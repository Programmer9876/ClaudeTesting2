#!/usr/bin/env python3
"""Joint tuning of numeric strategy weights with SPSA (docs/TUNING.md).

    python3 scripts/tune_joint.py --list-groups
    python3 scripts/tune_joint.py --group robber --max-games 2400 --state runs/tune/robber.json --plan
    python3 scripts/tune_joint.py --group robber --max-games 2400 --games 36 --state runs/tune/robber.json --workers 3
    python3 scripts/tune_joint.py --state runs/tune/robber.json --resume --workers 3 --max-minutes 120
    python3 scripts/tune_joint.py --state runs/tune/robber.json --status          # trajectory + league command
    /home/user/venv_cat33/bin/python scripts/tune_joint.py --group robber --mode catanatron --opponent value \\
        --max-games 4000 --games 48 --state runs/tune/robber_value.json --workers 3
    python3 scripts/tune_joint.py --group trade --exclude politics.BASELINE,search.counter_margin \\
        --max-games 2400 --state runs/tune/trade.json        # restrict a group (politics rule, docs/TUNING.md)

Budget: ``--max-games`` (required for a new run) caps the games of the whole run; the number of
iterations defaults to what fits (``max_games // games per iteration``), the planned total is printed
before anything is played and a run never starts an iteration that would exceed the cap.

SPSA (simultaneous perturbation stochastic approximation; Spall 1992) moves every parameter of a group at
once.  Parameters live in normalised units ``u = (x - default) / scale``: the scale is the span of the
registry's default and candidate values (``catanbot/tuning.py``), the default bounds are that span, so
``u`` runs over an interval of width 1 containing 0 (``--scale`` / ``--bounds`` change them).  Iteration k:

* ``Delta`` = a Rademacher vector (+-1 per parameter; ``random.Random("spsa:<seed>:<k>")``);
  ``theta+- = clip(theta +- c_k Delta)``; an integer knob is perturbed by at least half a unit so its two
  rounded values always differ;
* a batch of paired games theta+ vs theta- gives ``D``, an estimate of f(theta+) - f(theta-) where f is the
  per-seat win probability (``--objective vp``: final VP / 10);
* ``g_i = D / (u+_i - u-_i)`` (the installed values after clipping / rounding; = ``D / (2 c_k Delta_i)``
  for an unclipped real parameter); ``theta += a_k g`` with every coordinate's step capped at
  ``--max-step``, then clipped to the bounds;
* ``a_k = a / (A + k + 1)^alpha``, ``c_k = c / (k + 1)^gamma`` (Spall's alpha 0.602, gamma 0.101; A = 10%
  of the planned iterations; c 0.25; ``a`` is set so that a one-standard-error estimate of D moves a
  parameter by ``--first-step`` (0.05 of its scale) at k = 0 - see ``auto_gain``).

Game modes (``--mode``):

* ``selfplay`` (default): 4-player games, 2 seats theta+ vs 2 seats theta- (both sides ``ParamBot`` around
  the base spec, so each seat sees only its own constants), seat patterns rotated over the 6 arrangements;
  the mirrored arrangements (CCDD / DDCC, CDCD / DCDC, CDDC / DCCD) share one seed, i.e. one board and
  dice.  ``D`` = mean over games of (1/2 if a theta+ seat won, -1/2 if a theta- seat won, 0 at the cap).
* ``catanatron``: theta+ and theta- each play 1 seat vs 3 Catanatron bots (``--opponent``) on the same
  seeds and seat (``s % 4``) through ``scripts/ablate_catanatron.py``'s game machinery (3.3 presets such as
  ``value`` need /home/user/venv_cat33/bin/python).  ``D`` = mean over seeds of won+ - won-.

Seeds are fresh every iteration and disjoint from the screening and league seeds.  The process runs with
``PYTHONHASHSEED=0`` (and ``CATANBOT_NO_ACCEL=1`` when a parameter is read by the Python static evaluator;
it re-executes itself to set them), so the whole trajectory is a deterministic function of the state file:
after every iteration the state (theta, k, the history of theta, Delta, the installed theta+-, D and the
gradient estimate, games played, seeds) is rewritten atomically and the iteration's game records are
appended to ``<state>.games.jsonl`` (errors and per-game timeouts included; an errored game is left out of
D).  ``--resume`` continues exactly where the last finished iteration left off.

The tuner never edits a default.  ``--status`` prints the trajectory, the tuned vector (the average of
the iterates over the second half of the run, and the last iterate) as a ParamBot override dict and as a
bot spec (``tune=``, see ``agents/param_bot.py``), and the ``scripts/league.py gate`` command that must
confirm it (docs/LEAGUE.md).  Flags are not tuned here: test them with ``scripts/ablate.py --factorial``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from catanbot import factorial as F  # noqa: E402
from catanbot import tuning  # noqa: E402  (imported again after a re-exec; see ensure_env)

SCHEMA = 1
PINNED_HASHSEED = "0"
SIGMA = {"win": 0.5, "vp": 0.25}   # rough s.d. of one unit's D (a 2v2 game, a Catanatron seed) for auto_gain

# Parameter groups (``--group``).  Every member must be a numeric registry tunable.  Politics / table-social
# terms (politics.*, coalitions.*, opponent_model.stage_late_drop, search.counter_margin) belong in a run only
# if their one-at-a-time screen was significant (docs/ABLATIONS.md "Politics rule"): restrict the group with
# --exclude or replace it with --params.
GROUPS: Dict[str, Dict[str, Any]] = {
    "robber": {
        "params": ["danger.TURNS_HALF", "danger.BLOCK_NEED", "danger.BLOCK_FLOOR", "devcards.KNIGHT_VALUE",
                   "heuristic.EXPOSURE_WEIGHT", "placement.PLACEMENT_ROBBER_Q"],
        "note": "who / where to rob, knight value, steal exposure; EXPOSURE_WEIGHT needs the search bot and, with "
                "PLACEMENT_ROBBER_Q, the Python evaluator (~5x slower in self-play, ~3x vs Catanatron; --exclude both "
                "for the C++ evaluator)"},
    "robber_leaf": {
        "params": ["robber_eval.INSURANCE_W", "robber_eval.BLOCK_DUR_W"],
        "fixed": {"robber_eval.PERSIST_W": 0.0},
        "note": "knight insurance (R1b) and block duration (R1c), the robber leaf corrections of catanbot/robber_eval.py; "
                "they only act with search.robber_corr=1, added to the base spec automatically unless --base-spec is "
                "given.  R1a persistence failed its gate (docs/ABLATIONS.md), so robber_eval.PERSIST_W is held at 0 "
                "in both arms (the group's fixed value; --fix overrides it)"},
    "trade": {
        "params": ["trading.accept_margin", "politics.MAX_SLACK", "politics.BASELINE", "coalitions.SCALE",
                   "opponent_model.stage_late_drop", "search.counter_margin"],
        "note": "POLITICS RULE: only the terms whose screen was significant (Holm p < 0.05) may be tuned; "
                "--exclude the rest.  search.counter_margin needs counter=1 and the counter-offer rules (added "
                "automatically unless --base-spec is given)"},
    "ports": {
        "params": ["ports.FLOW_KAPPA"],
        "note": "ports.flow_provider's static-points weight for our seat's port value (catanbot/portvalue.py); needs "
                "the search bot.  The F1 conversion weights placement.PORT_A0 / PORT_B are left out: they only act "
                "with placement.PORT_MODEL > 0, an ordinary weight override (not a SearchConfig spec key), so the "
                "spec_requirement()/--base-spec mechanism that turns on robber_corr or counter above cannot switch it "
                "on here"},
}
POLITICS = F.POLITICS_TERMS
TRADE_TERMS = ("trading.", "politics.", "coalitions.", "opponent_model.", "search.counter", "search.trade_proposals")


# ---------------------------------------------------------------------------
# Parameters and normalisation
# ---------------------------------------------------------------------------
class Param:
    """One tuned parameter (a plain class, not a dataclass: the script is also loaded by file path)."""
    FIELDS = ("name", "default", "scale", "lo", "hi", "integer", "start")

    def __init__(self, name: str, default: float, scale: float, lo: float, hi: float, integer: bool, start: float):
        self.name = name
        self.default = default
        self.scale = scale        # raw units per normalised unit
        self.lo = lo              # raw bounds
        self.hi = hi
        self.integer = integer
        self.start = start        # raw value of theta_0

    def asdict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.FIELDS}

    def raw(self, u: float) -> float:
        return self.default + self.scale * u

    def norm(self, x: float) -> float:
        return (x - self.default) / self.scale

    @property
    def u_lo(self) -> float:
        return self.norm(self.lo)

    @property
    def u_hi(self) -> float:
        return self.norm(self.hi)

    def install(self, u: float) -> Any:
        """The value the bot gets for normalised ``u``: clipped to the bounds, rounded for integer knobs."""
        x = min(self.hi, max(self.lo, self.raw(u)))
        if self.integer:
            x = int(math.floor(x + 0.5))
            x = min(int(math.floor(self.hi)), max(int(math.ceil(self.lo)), x))
        return x

    def pretty(self, x: float) -> Any:
        """``x`` for output: an int, or 4 significant digits (far below the tuning noise)."""
        if self.integer:
            return int(x)
        return float(f"{x:.4g}")


def numeric_problem(t: tuning.Tunable) -> Optional[str]:
    """Why a tunable cannot be tuned by SPSA (None if it can)."""
    if t.kind == "flag":
        return "a flag: test it with scripts/ablate.py --factorial"
    d = t.default
    if isinstance(d, bool) or not isinstance(d, (int, float)):
        return f"not a number (default {t.format(d)}): not supported by the tuner"
    vals = [d] + list(t.candidates)
    if t.name == "search.depth" or (isinstance(d, int) and all(v in (0, 1) for v in vals)):
        return "an on/off or regime switch: test it with scripts/ablate.py --factorial"
    return None


def spec_requirement(name: str) -> Optional[Tuple[str, str, bool]]:
    """(spec key, value, needs the counter-offer rules) without which parameter ``name`` never acts."""
    if name in ("search.counter_margin", "search.counter_aggr"):
        return ("counter", "1", True)
    if name.startswith("winpaths.") or name in ("search.paths_w", "search.paths_crowd"):
        return ("paths", "1", False)
    if name in ("robber_eval.INSURANCE_W", "robber_eval.BLOCK_DUR_W"):
        return ("robber_corr", "1", False)
    return None


def _kv(items: Optional[Sequence[str]], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        for part in item.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise SystemExit(f"error: {what}: expected NAME=VALUE, got {part!r}")
            k, v = part.split("=", 1)
            out[tuning.find(k.strip()).name] = v.strip()
    return out


def build_params(names: Sequence[str], bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 scales: Optional[Dict[str, float]] = None, starts: Optional[Dict[str, float]] = None) -> List[Param]:
    out: List[Param] = []
    for n in names:
        t = tuning.find(n)
        why = numeric_problem(t)
        if why:
            raise ValueError(f"{t.name} is {why}")
        if any(p.name == t.name for p in out):
            raise ValueError(f"{t.name} given twice")
        vals = [float(t.default)] + [float(v) for v in t.candidates]
        lo, hi = min(vals), max(vals)
        if bounds and t.name in bounds:
            lo, hi = bounds[t.name]
        scale = float(scales[t.name]) if scales and t.name in scales else max(vals) - min(vals)
        if not scale > 0 or not hi > lo:
            raise ValueError(f"{t.name}: scale {scale} / bounds [{lo}, {hi}] are empty")
        integer = t.parse is tuning._parse_int or (isinstance(t.default, int) and not isinstance(t.default, bool))
        start = float(starts[t.name]) if starts and t.name in starts else float(t.default)
        if not lo <= start <= hi:
            raise ValueError(f"{t.name}: start {start} outside the bounds [{lo}, {hi}] (see --bounds)")
        out.append(Param(t.name, float(t.default), scale, float(lo), float(hi), integer, start))
    return out


def params_from_state(state: Dict[str, Any]) -> List[Param]:
    return [Param(**p) for p in state["params"]]


# ---------------------------------------------------------------------------
# SPSA arithmetic
# ---------------------------------------------------------------------------
def gains(cfg: Dict[str, Any], k: int) -> Tuple[float, float]:
    """``(a_k, c_k)`` of iteration ``k`` (0-based)."""
    return (cfg["a"] / (cfg["A"] + k + 1) ** cfg["alpha"], cfg["c"] / (k + 1) ** cfg["gamma"])


def auto_gain(first_step: float, A: float, alpha: float, c: float, units: int, sigma: float) -> float:
    """``a`` such that at k = 0 a gradient estimate of one standard error, ``sigma / sqrt(units) / (2c)``,
    moves a parameter by ``first_step`` normalised units: ``a_0 = first_step * 2c sqrt(units) / sigma``."""
    return first_step * (A + 1) ** alpha * 2.0 * c * math.sqrt(max(1, units)) / sigma


def perturbation(seed: int, k: int, n: int) -> List[int]:
    rng = random.Random(f"spsa:{seed}:{k}")      # string seeding: independent of PYTHONHASHSEED
    return [1 if rng.random() < 0.5 else -1 for _ in range(n)]


def plus_minus(params: Sequence[Param], theta: Sequence[float], c_k: float, delta: Sequence[int]
               ) -> Tuple[List[Any], List[Any], List[float]]:
    """Installed values of theta+ and theta- and their per-coordinate difference in normalised units."""
    xp, xm, d = [], [], []
    for p, u, s in zip(params, theta, delta):
        ck = max(c_k, 0.5 / p.scale) if p.integer else c_k
        a = p.install(min(p.u_hi, max(p.u_lo, u + ck * s)))
        b = p.install(min(p.u_hi, max(p.u_lo, u - ck * s)))
        xp.append(a)
        xm.append(b)
        d.append(p.norm(a) - p.norm(b))
    return xp, xm, d


def spsa_step(params: Sequence[Param], theta: Sequence[float], D: float, d: Sequence[float], a_k: float,
              max_step: float) -> Tuple[List[float], List[float], List[float]]:
    """Gradient estimate, capped step and the next theta (clipped to the bounds)."""
    ghat = [D / di if di else 0.0 for di in d]
    step = [max(-max_step, min(max_step, a_k * g)) for g in ghat]
    nxt = [min(p.u_hi, max(p.u_lo, u + s)) for p, u, s in zip(params, theta, step)]
    return ghat, step, nxt


def overrides_of(params: Sequence[Param], values: Sequence[Any]) -> Dict[str, Any]:
    return {p.name: v for p, v in zip(params, values)}


# ---------------------------------------------------------------------------
# Jobs (one per game / seed) and seeds
# ---------------------------------------------------------------------------
def selfplay_seed(seed: int, k: int, pair: int) -> int:
    """Board / dice seed of mirrored pair ``pair`` of iteration ``k`` (disjoint from ablate.py's seeds)."""
    return (seed * 1000003 + 7919 + k) * 100000 + pair


def catanatron_seed(seed: int, k: int, j: int, games: int) -> int:
    """Seed number ``s`` (catanatron game seed ``s + 1``, our seat ``s % 4``) of unit ``j`` of iteration ``k``."""
    return 10_000_000 * (seed + 1) + k * games + j


def make_jobs(cfg: Dict[str, Any], k: int, plus: Dict[str, Any], minus: Dict[str, Any]) -> List[Dict[str, Any]]:
    fixed = cfg.get("fixed") or {}
    if fixed:                                  # held values (--fix / a group's "fixed"), the same in both arms
        plus, minus = {**fixed, **plus}, {**fixed, **minus}
    fake = cfg.get("fake")
    jobs = []
    if cfg["mode"] == "selfplay":
        pats = tuning.SEAT_PATTERNS[4]
        for j in range(cfg["games"]):
            jobs.append({"mode": "selfplay", "k": k, "j": j, "seed": selfplay_seed(cfg["seed"], k, j // 2),
                         "pattern": pats[j % len(pats)], "spec": cfg["base_spec"], "plus": plus, "minus": minus,
                         "max_turns": cfg["max_turns"], "counters": cfg["counters"], "timeout": cfg["game_timeout"],
                         "fake": fake})
        return jobs
    ctx = cfg["ctx"]
    for j in range(cfg["games"]):
        s = catanatron_seed(cfg["seed"], k, j, cfg["games"])
        arms = [dict(role=role, spec=cfg["base_spec"], overrides=ov, adapter=dict(cfg["adapter"]), label=role,
                     key=f"spsa-{role}") for role, ov in (("plus", plus), ("minus", minus))]
        jobs.append({"mode": "catanatron", "k": k, "j": j, "s": s, "arms": arms, "ctx": ctx, "code": cfg.get("code"),
                     "exp": "spsa", "timeout": cfg["game_timeout"], "fake": fake})
    return jobs


# ---------------------------------------------------------------------------
# Playing (runs in worker processes)
# ---------------------------------------------------------------------------
class GameTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise GameTimeout("game exceeded --game-timeout")


def seat_bots(spec: str, plus: Dict[str, Any], minus: Dict[str, Any], pattern: str, make: Optional[Callable] = None):
    """``ParamBot`` seats: 'C' = theta+, 'D' = theta- (both sides wrapped: each seat applies its own
    constants only inside its own hooks, so the two never see each other's values)."""
    from catanbot.agents.param_bot import ParamBot
    from catanbot.selfplay import make_bot
    make = make or make_bot
    return [ParamBot(make(spec), plus if side == "C" else minus, label="plus" if side == "C" else "minus")
            for side in pattern]


def _selfplay_body(job: Dict[str, Any]) -> Dict[str, Any]:
    if job.get("fake"):
        return _fake_selfplay(job)
    from catanbot.selfplay import play_game
    bots = seat_bots(job["spec"], job["plus"], job["minus"], job["pattern"])
    res = play_game(bots, rng=random.Random(job["seed"]), seed=job["seed"], max_turns=job["max_turns"],
                    allow_counters=bool(job["counters"]))
    pat = job["pattern"]
    C = [i for i, s in enumerate(pat) if s == "C"]
    D = [i for i, s in enumerate(pat) if s == "D"]
    w = res.winner
    diff = 1.0 / len(C) if w in C else (-1.0 / len(D) if w in D else 0.0)
    ms = lambda idx: [1000.0 * t for i in idx for t in bots[i].stats["times"]]   # noqa: E731
    mp_, mm = ms(C), ms(D)
    return {"winner": w, "vps": list(res.vps), "turns": res.turns, "actions": res.actions,
            "duration": round(res.duration, 4), "diff": diff,
            "vp_diff": sum(res.vps[i] for i in C) / len(C) - sum(res.vps[i] for i in D) / len(D),
            "ms_plus": round(sum(mp_) / len(mp_), 4) if mp_ else None,
            "ms_minus": round(sum(mm) / len(mm), 4) if mm else None,
            "evaluator_mode": tuning.evaluator_mode()}


def play_selfplay_game(job: Dict[str, Any]) -> Dict[str, Any]:
    """One 2v2 game theta+ vs theta-; an exception or ``--game-timeout`` becomes an error record."""
    rec = {"k": job["k"], "j": job["j"], "seed": job["seed"], "pattern": job["pattern"], "pid": os.getpid()}
    timeout = int(job.get("timeout") or 0)
    use_alarm = timeout > 0 and threading.current_thread() is threading.main_thread()
    try:
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(timeout)
        try:
            body = _selfplay_body(job)
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
    except KeyboardInterrupt:
        raise
    except BaseException as ex:  # noqa: BLE001
        tb = traceback.format_exception(type(ex), ex, ex.__traceback__)
        rec.update(status="error", error=(f"{type(ex).__name__}: {ex} | " + "".join(tb[-3:]))[-2000:])
        return rec
    rec.update(body)
    rec["status"] = "ok"
    return rec


def play_catanatron_seed(job: Dict[str, Any]) -> Dict[str, Any]:
    """Both arms of one seed (theta+ then theta-), each 1 seat vs 3 Catanatron bots, through
    ``ablate_catanatron.play_job`` (per-arm timeout and error records included)."""
    if job.get("fake"):
        return _fake_catanatron(job)
    res = AB.play_job(job)
    keep = ("arm", "s", "seat", "game_seed", "status", "error", "won", "our_vp", "opp_vps", "turns", "duration",
            "truncated", "evaluator", "hashseed", "pid")
    return {"k": job["k"], "j": job["j"], "s": job["s"],
            "records": [{k2: r.get(k2) for k2 in keep if k2 in r} | {"dec_ms": (r.get("dec") or {}).get("mean")}
                        for r in res["records"]]}


def error_result(job: Dict[str, Any], message: str) -> Dict[str, Any]:
    if job["mode"] == "selfplay":
        return {"k": job["k"], "j": job["j"], "seed": job["seed"], "pattern": job["pattern"], "status": "error",
                "error": message}
    return {"k": job["k"], "j": job["j"], "s": job["s"],
            "records": [{"arm": a["role"], "s": job["s"], "status": "error", "error": message} for a in job["arms"]]}


def play_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return (play_selfplay_game if job["mode"] == "selfplay" else play_catanatron_seed)(job)


# --- synthetic games (tests / --fake-optimum): a known optimum, the real bookkeeping -----------------------
def fake_strength(fake: Dict[str, Any], overrides: Dict[str, Any]) -> float:
    """``-sum ((x - x*) / scale)^2`` over the parameters with a planted optimum (maximum 0 at the optimum)."""
    return -sum(((float(overrides[n]) - opt) / sc) ** 2 for n, (opt, sc) in fake["optimum"].items() if n in overrides)


def _fake_selfplay(job: Dict[str, Any]) -> Dict[str, Any]:
    fk = job["fake"]
    p = 0.5 + fk["kappa"] * (fake_strength(fk, job["plus"]) - fake_strength(fk, job["minus"]))
    p = min(0.98, max(0.02, p))
    won = random.Random(f"fake:{job['seed']}:{job['pattern']}").random() < p
    pat = job["pattern"]
    w = pat.index("C") if won else pat.index("D")
    vps = [10 if i == w else 6 for i in range(4)]
    return {"winner": w, "vps": vps, "turns": 80, "actions": 500, "duration": 0.0, "diff": 0.5 if won else -0.5,
            "vp_diff": 2.0 if won else -2.0, "ms_plus": 1.0, "ms_minus": 1.0, "evaluator_mode": "fake"}


def _fake_catanatron(job: Dict[str, Any]) -> Dict[str, Any]:
    fk = job["fake"]
    u = random.Random(f"fake:{job['s']}").random()        # common to both arms (same board and dice)
    recs = []
    for arm in job["arms"]:
        p = min(0.98, max(0.02, 0.3 + fk["kappa"] * fake_strength(fk, arm["overrides"])))
        won = u < p
        recs.append({"arm": arm["role"], "s": job["s"], "seat": job["s"] % 4, "status": "ok", "won": won,
                     "our_vp": 10 if won else 6, "turns": 80, "duration": 0.0})
    return {"k": job["k"], "j": job["j"], "s": job["s"], "records": recs}


# ---------------------------------------------------------------------------
# Worker pool: a fork pool per batch, surviving crashed games and dead workers
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent handles ^C
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _new_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_worker_init)


def run_jobs(jobs: Sequence[Dict[str, Any]], workers: int, fn: Callable = None,
             progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
    """Results of ``fn(job)`` in job order.  A job whose worker process dies (no Python exception to record)
    is replayed alone in a fresh process; if that dies too it becomes an error record."""
    fn = fn or play_job
    out: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs):
            out[i] = fn(job)
            if progress:
                progress(out[i])
        return out  # type: ignore[return-value]
    lost: List[int] = []
    pool = _new_pool(workers)
    clean = False
    try:
        futs = {pool.submit(fn, job): i for i, job in enumerate(jobs)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                out[i] = f.result()
            except BrokenProcessPool:
                lost.append(i)
                continue
            except Exception as ex:  # noqa: BLE001  (e.g. an unpicklable result)
                out[i] = error_result(jobs[i], f"{type(ex).__name__}: {ex}")
            if progress:
                progress(out[i])
        clean = True
    finally:
        AB._close(pool, force=not clean)
    for i in sorted(lost):
        solo = _new_pool(1)
        try:
            out[i] = solo.submit(fn, jobs[i]).result()
        except BrokenProcessPool:
            out[i] = error_result(jobs[i], "worker process died while playing this game (hard crash)")
        except Exception as ex:  # noqa: BLE001
            out[i] = error_result(jobs[i], f"{type(ex).__name__}: {ex}")
        finally:
            AB._close(solo)
        if progress:
            progress(out[i])
    return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# One iteration
# ---------------------------------------------------------------------------
def mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def estimate(cfg: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """D (theta+ minus theta-, per-seat win rate or VP / 10) from a batch, errors left out."""
    wins, vps = [], []
    pw = mw = draws = errors = 0
    if cfg["mode"] == "selfplay":
        for r in results:
            if r.get("status") != "ok":
                errors += 1
                continue
            wins.append(r["diff"])
            vps.append(r["vp_diff"] / 10.0)
            pw += r["diff"] > 0
            mw += r["diff"] < 0
            draws += r["diff"] == 0
    else:
        for r in results:
            recs = {x["arm"]: x for x in r["records"]}
            p, m = recs.get("plus"), recs.get("minus")
            if not p or not m or p.get("status") != "ok" or m.get("status") != "ok":
                errors += 1
                continue
            wins.append(float(bool(p["won"])) - float(bool(m["won"])))
            vps.append((float(p["our_vp"]) - float(m["our_vp"])) / 10.0)
            pw += bool(p["won"])
            mw += bool(m["won"])
    D, se = mean_se(wins if cfg["objective"] == "win" else vps)
    return {"D": D, "se": se, "units_ok": len(wins), "units_err": errors, "plus_wins": pw, "minus_wins": mw,
            "draws": draws, "win_diff": mean_se(wins)[0], "vp_diff": mean_se(vps)[0] * 10.0 if vps else float("nan")}


def iteration(state: Dict[str, Any], runner: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
              ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Play iteration ``state["k"]`` and return its history row and game results (the state is not changed)."""
    cfg = state["config"]
    params = params_from_state(state)
    k = state["k"]
    theta = list(state["theta"])
    a_k, c_k = gains(cfg, k)
    delta = perturbation(cfg["seed"], k, len(params))
    xp, xm, d = plus_minus(params, theta, c_k, delta)
    plus, minus = overrides_of(params, xp), overrides_of(params, xm)
    t0 = time.time()
    results = runner(make_jobs(cfg, k, plus, minus))
    est = estimate(cfg, results)
    if not est["units_ok"]:
        raise RuntimeError(f"iteration {k}: every game failed (first error: "
                           f"{next((str(r.get('error'))[:300] for r in _flat(results) if r.get('error')), '?')})")
    ghat, step, nxt = spsa_step(params, theta, est["D"], d, a_k, cfg["max_step"])
    row = {"k": k, "a_k": a_k, "c_k": c_k, "delta": delta, "theta": theta, "plus": xp, "minus": xm, "d": d,
           "ghat": ghat, "step": step, "theta_next": nxt, "seconds": round(time.time() - t0, 3),
           "seeds": _seed_range(results), "code": cfg.get("code"), **est}
    return row, results


def _flat(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        out.extend(r["records"] if "records" in r else [r])
    return out


def _seed_range(results: Sequence[Dict[str, Any]]) -> List[int]:
    ss = [r.get("seed", r.get("s")) for r in results]
    return [min(ss), max(ss)] if ss else []


def apply_row(state: Dict[str, Any], row: Dict[str, Any]) -> None:
    state["history"].append(row)
    state["theta"] = list(row["theta_next"])
    state["k"] = row["k"] + 1
    state["games_played"] += row["units_ok"] * (2 if state["config"]["mode"] == "catanatron" else 1)
    state["game_errors"] += row["units_err"]
    state["seconds"] += row["seconds"]
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["result"] = recommendation(state)


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------
def json_safe(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def save_state(path: str, state: Dict[str, Any]) -> None:
    """Atomic rewrite (temp file + fsync + rename): a kill leaves the previous iteration's state intact."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(json_safe(state), fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_state(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        st = json.load(fh)
    if st.get("kind") != "spsa" or st.get("v") != SCHEMA:
        raise SystemExit(f"error: {path} is not a tune_joint state (schema {SCHEMA})")
    return st


def games_path(state_path: str) -> str:
    return state_path + ".games.jsonl"


def append_games(state_path: str, results: Sequence[Dict[str, Any]]) -> None:
    with open(games_path(state_path), "a") as fh:
        fh.write("".join(json.dumps(json_safe(r), sort_keys=True, separators=(",", ":")) + "\n" for r in results))
        fh.flush()
        os.fsync(fh.fileno())


def new_state(cfg: Dict[str, Any], params: Sequence[Param], env: Dict[str, Any]) -> Dict[str, Any]:
    st = {"v": SCHEMA, "kind": "spsa", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "updated": None,
          "config": cfg, "params": [p.asdict() for p in params], "k": 0,
          "theta": [p.norm(p.start) for p in params], "history": [], "games_played": 0, "game_errors": 0,
          "seconds": 0.0, "env": env,
          "rng": {"perturbation": f"random.Random('spsa:{cfg['seed']}:<k>'), +1 if random() < 0.5 else -1",
                  "seeds": ("selfplay_seed(seed, k, j // 2), pattern j % 6" if cfg["mode"] == "selfplay"
                            else "catanatron_seed(seed, k, j, games): game seed s + 1, seat s % 4")}}
    st["result"] = recommendation(st)
    return st


def run(state: Dict[str, Any], state_path: str, runner: Callable, deadline: Optional[float] = None,
        log=None) -> str:
    """Play iterations until ``config.iterations`` (``"done"``) or the deadline (``"deadline"``); the state
    file is rewritten after every iteration."""
    cfg = state["config"]
    params = params_from_state(state)
    log = log or sys.stdout
    while state["k"] < cfg["iterations"]:
        if deadline is not None and time.time() >= deadline:
            return "deadline"
        if (state["k"] + 1) * cfg["games_per_iteration"] > cfg["max_games"]:
            return "games cap reached"
        row, results = iteration(state, runner)
        append_games(state_path, results)
        apply_row(state, row)
        save_state(state_path, state)
        print(f"  k={row['k']:>4} a_k {row['a_k']:.4f} c_k {row['c_k']:.4f}  D {100 * row['D']:+6.1f}pp +- "
              f"{100 * (row['se'] if not math.isnan(row['se']) else 0):4.1f}  ({row['units_ok']} ok"
              + (f", {row['units_err']} err" if row["units_err"] else "") + f", {row['seconds']:.0f} s)  theta "
              + " ".join(f"{_short(p.name)}={p.pretty(p.install(u)):g}" for p, u in zip(params, state["theta"])),
              file=log, flush=True)
    return "done"


# ---------------------------------------------------------------------------
# Output: tuned vector, trajectory, league command
# ---------------------------------------------------------------------------
def _short(name: str) -> str:
    return name.split(".", 1)[1]


def recommendation(state: Dict[str, Any]) -> Dict[str, Any]:
    """The tuned vector: the average of the iterates over the second half of the run (Polyak-Ruppert
    averaging, less noisy than the last iterate) and the last iterate, as installed values."""
    params = params_from_state(state)
    hist = state["history"]
    K = len(hist)
    tail = [h["theta_next"] for h in hist[K // 2:]] or [state["theta"]]
    avg = [sum(t[i] for t in tail) / len(tail) for i in range(len(params))]
    avg_x = [p.pretty(p.install(u)) for p, u in zip(params, avg)]
    last_x = [p.pretty(p.install(u)) for p, u in zip(params, state["theta"])]
    changed = {p.name: x for p, x in zip(params, avg_x) if x != p.pretty(p.default)}
    changed = {**(state["config"].get("fixed") or {}), **changed}
    from catanbot.agents.param_bot import tuned_spec
    spec = tuned_spec(state["config"]["base_spec"], changed) if changed else state["config"]["base_spec"]
    return {"iterations": K, "averaged_over": len(tail) if hist else 0, "average": overrides_of(params, avg_x),
            "last": overrides_of(params, last_x), "overrides": changed, "spec": spec,
            "python_evaluator": any(tuning.find(n).needs_python_evaluator for n in changed)}


def gate_command(state: Dict[str, Any], state_path: str) -> str:
    rec = state["result"]
    cfg = state["config"]
    trade = any(n.startswith(TRADE_TERMS) for n in rec["overrides"])
    parts = ["PYTHONPATH=$PWD python3 scripts/league.py gate --candidate-commit <sha of the commit that ran the tuner>",
             f"    --candidate-spec '{rec['spec']}'"]
    if rec["python_evaluator"]:
        parts.append("    --env CATANBOT_NO_ACCEL=1 --allow-no-accel")
    if trade:
        parts.append("    --formats 4p2v2,3p1v2")
    notes = (f"SPSA {cfg.get('group') or 'params'} ({len(state['params'])} params), {rec['iterations']} iterations x "
             f"{cfg['games']} {'games' if cfg['mode'] == 'selfplay' else 'seeds'} ({cfg['mode']}), state {state_path}")
    parts.append(f"    --notes '{notes}'")
    return " \\\n".join(parts)


def trajectory_lines(state: Dict[str, Any], max_rows: int = 30) -> List[str]:
    params = params_from_state(state)
    hist = state["history"]
    step = max(1, math.ceil(len(hist) / max_rows))
    rows = [h for i, h in enumerate(hist) if i % step == 0 or i == len(hist) - 1]
    head = f"  {'k':>4} {'a_k':>7} {'c_k':>6} {'ok/err':>7} {'D (pp)':>8} {'+-se':>6} " + " ".join(
        f"{_short(p.name)[:14]:>14}" for p in params)
    out = [head]
    for h in rows:
        se = h.get("se")
        out.append(f"  {h['k']:>4} {h['a_k']:>7.4f} {h['c_k']:>6.3f} {h['units_ok']:>4}/{h['units_err']:<2} "
                   f"{100 * h['D']:>+8.1f} {100 * (se or 0):>6.1f} "
                   + " ".join(f"{p.pretty(p.install(u)):>14g}" for p, u in zip(params, h["theta_next"])))
    return out


def print_status(state: Dict[str, Any], state_path: str, out=None) -> None:
    p_ = lambda *a: print(*a, file=out or sys.stdout)   # noqa: E731
    cfg = state["config"]
    params = params_from_state(state)
    env = state.get("env") or {}
    K = state["k"]
    p_(f"{state_path}: SPSA {cfg.get('group') or 'custom'} ({len(params)} parameters), mode {cfg['mode']}"
       + (f" vs 3x {cfg['opponent']}" if cfg["mode"] == "catanatron" else "") + f"; {K}/{cfg['iterations']} iterations, "
       f"{state['games_played']} games played, {state['game_errors']} error unit(s), {state['seconds'] / 3600:.2f} h of play")
    p_(f"  base spec {cfg['base_spec']}" + ("; counter-offer rules ON" if cfg.get("counters") else "")
       + f"; {cfg['games']} {'games' if cfg['mode'] == 'selfplay' else 'seeds (x2 games)'} per iteration; objective "
       f"{cfg['objective']}; seed {cfg['seed']}; {env.get('evaluator', '?')}; code {env.get('code', '?')}"
       + (" [SYNTHETIC GAMES]" if cfg.get("fake") else ""))
    p_(f"  gains a {cfg['a']:.4g} A {cfg['A']:g} alpha {cfg['alpha']:g}; c {cfg['c']:g} gamma {cfg['gamma']:g}; "
       f"max step {cfg['max_step']:g} (normalised units)")
    if K:
        p_(f"  cost {state['seconds'] / K:.0f} s per iteration at the workers used "
           f"({3600.0 * state['games_played'] / max(state['seconds'], 1e-9):.0f} games/h)")
    rec = state["result"] = recommendation(state)       # recomputed from the history (not the stored copy)
    p_(f"  {'parameter':34} {'default':>9} {'bounds':>17} {'scale':>8} {'start':>9} {'last':>9} {'average':>9}")
    for p in params:
        p_(f"  {p.name:34} {p.pretty(p.default):>9g} {f'[{p.lo:g}, {p.hi:g}]':>17} {p.scale:>8.4g} "
           f"{p.pretty(p.start):>9g} {rec['last'][p.name]:>9g} {rec['average'][p.name]:>9g}"
           + ("  (int)" if p.integer else ""))
    if K:
        p_("  (a parameter nothing depends on drifts too - at a few thousand games no per-parameter statistic tells "
           "drift from signal; the vector is judged as a whole, by the league gate)")
    if state["history"]:
        p_("  trajectory (theta after each update, installed values):")
        for line in trajectory_lines(state):
            p_(line)
    p_(f"  tuned vector (average of the last {rec['averaged_over']} iterates) as ParamBot overrides "
       f"(parameters that moved off their default):")
    p_("    " + json.dumps(rec["overrides"]))
    p_(f"  bot spec: {rec['spec']}")
    if K < cfg["iterations"]:
        p_(f"  unfinished: rerun with --resume to play iterations {K}..{cfg['iterations'] - 1}")
    p_("  The tuner never changes a default.  Confirm the tuned set through the champion league gate "
       "(docs/LEAGUE.md) after committing the code it ran with:")
    for line in gate_command(state, state_path).splitlines():
        p_("    " + line)
    if rec["python_evaluator"]:
        p_("  (a static-evaluator weight: adopting it as a default also needs the constexpr copy in "
           "cpp/heuristic.cpp changed)")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _load(name: str, path: str):
    """A script as a module, registered in ``sys.modules`` so worker pools can pickle its functions."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AB = _load("_ablate_catanatron_for_tune_joint", os.path.join(ROOT, "scripts", "ablate_catanatron.py"))


def ensure_env(need_python_eval: bool) -> None:
    """Re-execute with ``PYTHONHASHSEED=0`` (and ``CATANBOT_NO_ACCEL=1`` for parameters the Python static
    evaluator reads) unless already set; both must exist before the interpreter / ``catanbot`` start."""
    want = AB.required_env(need_python_eval)
    if all(os.environ.get(k) == v for k, v in want.items()):
        return
    missing = ", ".join(f"{k}={v}" for k, v in want.items() if os.environ.get(k) != v)
    print(f"re-executing with {missing}", flush=True)
    env = dict(os.environ, **want)
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)


def environment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"python": sys.version.split()[0], "hashseed": os.environ.get("PYTHONHASHSEED"),
            "evaluator": tuning.evaluator_mode(), "catanatron": AB.catanatron_version(),
            "code": AB.code_fingerprint()}


# ---------------------------------------------------------------------------
# Configuration from the command line
# ---------------------------------------------------------------------------
CONFIG_OPTS = ("group", "params", "exclude", "fix", "mode", "base_spec", "counters", "opponent", "opponent_params", "trades",
               "games", "seed", "a", "A", "c", "alpha", "gamma", "first_step", "max_step", "objective", "bounds", "scale",
               "start", "max_turns", "fake_optimum", "fake_kappa")


def select_names(args) -> List[str]:
    if args.params:
        names = [x.strip() for x in args.params.split(",") if x.strip()]
    elif args.group:
        if args.group not in GROUPS:
            raise SystemExit(f"error: unknown group {args.group!r} (groups: {', '.join(GROUPS)})")
        names = list(GROUPS[args.group]["params"])
    else:
        raise SystemExit("error: give --group NAME or --params NAME,NAME,... (see --list-groups)")
    try:
        names = [tuning.find(n).name for n in names]
        drop = {tuning.find(x.strip()).name for x in (args.exclude or "").split(",") if x.strip()}
    except KeyError as ex:
        raise SystemExit(f"error: {ex}")
    unknown = drop - set(names)
    if unknown:
        raise SystemExit(f"error: --exclude {sorted(unknown)}: not in the parameter list")
    names = [n for n in names if n not in drop]
    if not names:
        raise SystemExit("error: no parameters left to tune")
    return names


def resolve_setting(names: List[str], args) -> Tuple[str, bool, List[str]]:
    """Base spec and rules every parameter acts in; notes about what was added.  A parameter that could not
    act with an explicitly given spec is an error (both sides would play identical games in it)."""
    from catanbot.selfplay import parse_spec
    ts = [tuning.find(n) for n in names]
    notes: List[str] = []
    spec = args.base_spec or (tuning.DEFAULT_DEEP_SPEC if any(t.requires_depth >= 2 for t in ts)
                              else tuning.DEFAULT_SEARCH_SPEC)
    counters = bool(args.counters)
    for t in ts:
        if t.requires_search and not spec.startswith("search"):
            raise SystemExit(f"error: {t.name} only acts in the search bot; use a search --base-spec")
        if tuning.spec_depth(spec) < t.requires_depth:
            raise SystemExit(f"error: {t.name} is only read at depth >= {t.requires_depth}; use a deeper --base-spec")
        req = spec_requirement(t.name)
        if req is None:
            continue
        key, val, rules = req
        _, kw = parse_spec(spec)
        if kw.get(key) != val:
            if args.base_spec:
                raise SystemExit(f"error: {t.name} never acts without {key}={val} in the base spec; add it or "
                                 f"--exclude {t.name}")
            spec = f"{spec},{key}={val}"
            notes.append(f"{t.name}: added {key}={val} to the base spec")
        if rules and args.mode == "selfplay" and not counters:
            counters = True
            notes.append(f"{t.name}: counter-offer rules switched on (every game of the run; the base game has none)")
        if rules and args.mode == "catanatron":
            raise SystemExit(f"error: {t.name} needs the counter-offer rules, which the Catanatron games do not "
                             f"have; --exclude it in --mode catanatron")
    if args.mode == "catanatron" and (args.trades or "off") == "off" and any(n.startswith(TRADE_TERMS) for n in names):
        notes.append("WARNING: trade terms against Catanatron with --trades off: catanbot neither offers nor "
                     "answers domestic trades, so they mostly cannot act (use --trades value on catanatron 3.3)")
    return spec, counters, notes


def build_config(args) -> Tuple[Dict[str, Any], List[Param], List[str]]:
    args.mode = args.mode or "selfplay"
    names = select_names(args)
    try:
        bounds = {k: tuple(float(x) for x in v.replace(",", ":").split(":")) for k, v in _kv(args.bounds, "--bounds").items()}
        if any(len(b) != 2 for b in bounds.values()):
            raise SystemExit("error: --bounds NAME=LO:HI")
        params = build_params(names, bounds, {k: float(v) for k, v in _kv(args.scale, "--scale").items()},
                              {k: float(v) for k, v in _kv(args.start, "--start").items()})
    except (ValueError, KeyError) as ex:
        raise SystemExit(f"error: {ex}")
    spec, counters, notes = resolve_setting(names, args)
    games = args.games or (36 if args.mode == "selfplay" else 48)
    if args.mode == "selfplay" and games % 2:
        games += 1
        notes.append(f"--games rounded up to {games} (mirrored pairs share a board)")
    if args.mode == "selfplay" and games % 6:
        notes.append(f"{games} games is not a multiple of 6: the seat arrangements are not all equally used")
    if args.mode == "catanatron" and games % 4:
        notes.append(f"{games} seeds is not a multiple of 4: our seat is not balanced within an iteration")
    per_iter = games * (2 if args.mode == "catanatron" else 1)
    if not args.max_games:
        raise SystemExit("error: --max-games N is required: the cap on the games of the whole run (e.g. 2400 in "
                         "self-play, 4000 = 2000 paired seeds against Catanatron; docs/TUNING.md)")
    iterations = args.iterations or args.max_games // per_iter
    if iterations < 1 or iterations * per_iter > args.max_games:
        raise SystemExit(f"error: {iterations} iterations x {per_iter} games = {iterations * per_iter} games exceed "
                         f"--max-games {args.max_games}: lower --iterations or --games")
    A = args.A if args.A is not None else max(1.0, round(0.1 * iterations))
    alpha = args.alpha if args.alpha is not None else 0.602
    gamma = args.gamma if args.gamma is not None else 0.101
    c = args.c if args.c is not None else 0.25
    objective = args.objective or "win"
    first_step = args.first_step if args.first_step is not None else 0.05
    a = args.a if args.a is not None else auto_gain(first_step, A, alpha, c, games, SIGMA[objective])
    fixed = dict(GROUPS[args.group].get("fixed", {})) if args.group and not args.params else {}
    try:
        fixed.update({k: float(v) for k, v in _kv(getattr(args, "fix", None), "--fix").items()})
        for n in fixed:
            tuning.find(n)
    except (ValueError, KeyError) as ex:
        raise SystemExit(f"error: --fix: {ex}")
    if set(fixed) & set(names):
        raise SystemExit(f"error: --fix names are also tuned: {sorted(set(fixed) & set(names))}")
    cfg = {"mode": args.mode, "group": args.group if not args.params else None, "names": names, "base_spec": spec,
           "counters": counters, "games": games, "games_per_iteration": per_iter, "iterations": iterations,
           "max_games": args.max_games, "seed": args.seed if args.seed is not None else 1,
           "a": a, "a_auto": args.a is None, "A": A, "alpha": alpha, "c": c, "gamma": gamma, "first_step": first_step,
           "max_step": args.max_step if args.max_step is not None else 0.1, "objective": objective,
           "sigma": SIGMA[objective], "max_turns": args.max_turns or 400, "game_timeout": args.game_timeout,
           "python_eval": any(tuning.find(n).needs_python_evaluator for n in names), "notes": notes}
    if fixed:
        cfg["fixed"] = fixed
        cfg["notes"] = list(notes) + [f"held (not tuned, both arms): {fixed}"]
    if args.mode == "catanatron":
        if not args.opponent:
            raise SystemExit("error: --mode catanatron needs --opponent (e.g. value on catanatron 3.3)")
        cfg["opponent"] = args.opponent
        cfg["opponent_params"] = AB.parse_opponent_params(args.opponent_params)
        cfg["trades"] = args.trades or "off"
        cfg["adapter"] = {"trades": cfg["trades"]}
    if args.fake_optimum:
        opt = {k: float(v) for k, v in _kv([args.fake_optimum], "--fake-optimum").items()}
        sc = {p.name: p.scale for p in params}
        cfg["fake"] = {"optimum": {n: [v, sc.get(n, 1.0)] for n, v in opt.items()}, "kappa": args.fake_kappa}
    return cfg, params, notes


def finish_context(cfg: Dict[str, Any], env: Dict[str, Any]) -> None:
    """Fields that need the running process: the Catanatron context and the code fingerprint."""
    cfg["code"] = env["code"]
    if cfg["mode"] == "catanatron":
        cfg["ctx"] = {"opponent": cfg["opponent"], "opponent_params": dict(cfg.get("opponent_params") or {}),
                      "python": env["python"],
                      "catanatron": env["catanatron"], "evaluator": env["evaluator"], "vps_to_win": 10,
                      "discard_limit": 7, "hashseed": env["hashseed"], "fake": None}


def validate(cfg: Dict[str, Any], params: Sequence[Param]) -> None:
    """Fail fast: the base spec builds, the overrides are accepted by it, the opponent resolves."""
    from catanbot.agents.param_bot import ParamBot
    from catanbot.selfplay import make_bot
    try:
        ParamBot(make_bot(cfg["base_spec"]), {p.name: p.install(p.norm(p.start)) for p in params})
    except (TypeError, KeyError, ValueError) as ex:
        raise SystemExit(f"error: {ex}")
    if cfg["mode"] == "catanatron" and not cfg.get("fake"):
        try:
            AB._opponent_maker(cfg["ctx"])
        except SystemExit as ex:
            raise SystemExit(f"error: opponent {cfg['opponent']!r}: {ex}")


def print_plan(cfg: Dict[str, Any], params: Sequence[Param], out=None) -> None:
    p_ = lambda *a: print(*a, file=out or sys.stdout)   # noqa: E731
    p_(f"SPSA plan: {len(params)} parameters, mode {cfg['mode']}" + (f" vs 3x {cfg.get('opponent')}" if cfg["mode"] ==
                                                                        "catanatron" else "")
       + f", base spec {cfg['base_spec']}" + ("; counter-offer rules ON" if cfg["counters"] else ""))
    p_(f"  {cfg['iterations']} iterations x {cfg['games']} {'games' if cfg['mode'] == 'selfplay' else 'seeds (x2 games)'}"
       f" = {cfg['iterations'] * cfg['games_per_iteration']} games planned (cap --max-games {cfg['max_games']})"
       f"; objective {cfg['objective']}; seed {cfg['seed']}; Python evaluator {'yes' if cfg['python_eval'] else 'no'}")
    a0, c0 = gains(cfg, 0)
    aK, cK = gains(cfg, cfg["iterations"] - 1)
    p_(f"  a {cfg['a']:.4g} ({'auto: first step ' + format(cfg['first_step'], 'g') if cfg['a_auto'] else 'given'}), "
       f"A {cfg['A']:g}, alpha {cfg['alpha']:g}: a_k {a0:.4f} -> {aK:.4f}; c {cfg['c']:g}, gamma {cfg['gamma']:g}: "
       f"c_k {c0:.3f} -> {cK:.3f}; max step {cfg['max_step']:g}")
    se = cfg["sigma"] / math.sqrt(cfg["games"])
    p_(f"  noise: one iteration's D has s.e. ~{100 * se:.1f}pp; its gradient estimate ~{se / (2 * c0):.2f} per "
       f"normalised unit (k=0) -> a first step of ~{a0 * se / (2 * c0):.3f}")
    p_(f"  {'parameter':34} {'default':>9} {'bounds':>17} {'scale':>8} {'start':>9} {'c_0 raw':>9}")
    for p in params:
        ck = max(c0, 0.5 / p.scale) if p.integer else c0
        p_(f"  {p.name:34} {p.pretty(p.default):>9g} {f'[{p.lo:g}, {p.hi:g}]':>17} {p.scale:>8.4g} "
           f"{p.pretty(p.start):>9g} {ck * p.scale:>9.4g}" + ("  (int)" if p.integer else "")
           + ("  POLITICS RULE: only if its screen was significant" if p.name.startswith(POLITICS) else ""))
    for n in cfg.get("notes") or []:
        p_(f"  note: {n}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", help="state JSON (rewritten after every iteration; games in <state>.games.jsonl)")
    p.add_argument("--resume", action="store_true", help="continue the run in --state (its configuration is kept)")
    p.add_argument("--status", action="store_true", help="print the trajectory, tuned vector and league command")
    p.add_argument("--plan", action="store_true", help="print the parameters, gains and noise; play nothing")
    p.add_argument("--list-groups", action="store_true")
    g = p.add_argument_group("what to tune")
    g.add_argument("--group", help=f"parameter preset: {', '.join(GROUPS)}")
    g.add_argument("--params", help="comma-separated registry names (replaces the group's list)")
    g.add_argument("--exclude", help="comma-separated names dropped from the list (politics rule)")
    g.add_argument("--bounds", action="append", metavar="NAME=LO:HI", help="raw bounds (default: the registry's "
                                                                           "default and candidate span)")
    g.add_argument("--scale", action="append", metavar="NAME=X", help="raw units per normalised unit "
                                                                      "(default: that span)")
    g.add_argument("--start", action="append", metavar="NAME=V", help="theta_0 (default: the registry default)")
    g.add_argument("--fix", action="append", metavar="NAME=V",
                   help="hold a registry weight at V in both arms, not tuned (a group may preset some)")
    g = p.add_argument_group("games")
    g.add_argument("--mode", choices=("selfplay", "catanatron"), default=None, help="default selfplay")
    g.add_argument("--base-spec", help=f"bot spec of both sides (default {tuning.DEFAULT_SEARCH_SPEC}; depth 2 for "
                                       f"knobs only the depth-2 search reads)")
    g.add_argument("--counters", action="store_true", help="self-play under the counter-offer rules variant")
    g.add_argument("--opponent", help="--mode catanatron: bench_catanatron preset (value / alphabeta on 3.3, vf / "
                                      "random ... on 3.2.1)")
    g.add_argument("--opponent-params", default=None, metavar="KEY=VAL,...",
                   help="--mode catanatron: constructor parameters of every opponent (bench_catanatron.py syntax)")
    g.add_argument("--trades", default=None, help="--mode catanatron: domestic trading as in ablate_catanatron.py "
                                                  "(off | native | value | fair, 3.3 only)")
    g.add_argument("--games", type=int, default=None, help="per iteration: 2v2 games (selfplay, default 36) or seeds "
                                                           "played by both sides (catanatron, default 48)")
    g.add_argument("--max-games", type=int, default=None,
                   help="REQUIRED for a new run: cap on the games of the whole run (catanatron: 2 per seed); may be "
                        "raised explicitly on --resume")
    g.add_argument("--iterations", type=int, default=None, help="total iterations (default: as many as --max-games "
                                                                "allows; may be raised on --resume within the cap)")
    g.add_argument("--seed", type=int, default=None, help="base seed (perturbations and game seeds; default 1)")
    g.add_argument("--max-turns", type=int, default=None)
    g.add_argument("--game-timeout", type=int, default=1800, help="seconds before a game is recorded as an error")
    g = p.add_argument_group("SPSA gains (normalised units)")
    g.add_argument("--a", type=float, default=None, help="step gain (default: auto from --first-step)")
    g.add_argument("--A", type=float, default=None, help="stability constant (default 10%% of --iterations)")
    g.add_argument("--alpha", type=float, default=None, help="a_k exponent (default 0.602)")
    g.add_argument("--c", type=float, default=None, help="perturbation size c (default 0.25)")
    g.add_argument("--gamma", type=float, default=None, help="c_k exponent (default 0.101)")
    g.add_argument("--first-step", type=float, default=None,
                   help="auto a: step of a one-s.e. gradient estimate at k=0 (default 0.05)")
    g.add_argument("--max-step", type=float, default=None, help="cap on any coordinate's step (default 0.1)")
    g.add_argument("--objective", choices=("win", "vp"), default=None,
                   help="win (per-seat win rate, default) or vp (final VP / 10: less noise, not the target)")
    g = p.add_argument_group("running")
    g.add_argument("--workers", type=int, default=1, help="parallel game processes (fork pool)")
    g.add_argument("--max-minutes", type=float, default=None,
                   help="start no new iteration after this many minutes (rerun with --resume)")
    g.add_argument("--quiet", action="store_true", help="no per-game lines")
    p.add_argument("--no-reexec", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fake-optimum", default=None, help=argparse.SUPPRESS)    # synthetic games (tests)
    p.add_argument("--fake-kappa", type=float, default=2.0, help=argparse.SUPPRESS)
    return p


def print_groups() -> None:
    for name, g in GROUPS.items():
        print(f"{name}:")
        for n in g["params"]:
            t = tuning.find(n)
            print(f"  {n:34} default {t.format(t.default):>7}  candidates {', '.join(t.format(c) for c in t.candidates)}"
                  + ("  [Python evaluator]" if t.needs_python_evaluator else "")
                  + ("  [search bot]" if t.requires_search else "")
                  + ("  [politics rule]" if n.startswith(POLITICS) else ""))
        print(f"  note: {g['note']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_groups:
        print_groups()
        return 0
    if not args.state:
        parser.print_usage()
        print("error: --state is required", file=sys.stderr)
        return 2
    if args.status:
        if not os.path.exists(args.state):
            print(f"error: no state file {args.state}", file=sys.stderr)
            return 2
        print_status(load_state(args.state), args.state)
        return 0
    if args.resume:
        if not os.path.exists(args.state):
            print(f"error: --resume: no state file {args.state}", file=sys.stderr)
            return 2
        state = load_state(args.state)
        given = [o for o in CONFIG_OPTS if getattr(args, o, None) not in (None, False)
                 and not (o == "fake_kappa" and args.fake_kappa == parser.get_default("fake_kappa"))]
        if given:
            print(f"note: --resume keeps the state's configuration; ignored: {', '.join('--' + o.replace('_', '-') for o in given)}")
        cfg = state["config"]
        if args.max_games:
            cfg["max_games"] = args.max_games       # an explicit new cap
        if args.iterations and args.iterations != cfg["iterations"]:
            if args.iterations < state["k"]:
                raise SystemExit(f"error: {state['k']} iterations are already done")
            if args.iterations * cfg["games_per_iteration"] > cfg["max_games"]:
                raise SystemExit(f"error: {args.iterations} iterations = {args.iterations * cfg['games_per_iteration']}"
                                 f" games exceed the run's cap of {cfg['max_games']} (raise it with --max-games)")
            cfg["iterations"] = args.iterations     # A stays as created, so the finished iterations are unchanged
        cfg["game_timeout"] = args.game_timeout
    else:
        if os.path.exists(args.state) and not args.plan:
            print(f"error: {args.state} exists: --resume it, --status it, or choose another --state", file=sys.stderr)
            return 2
        cfg, params, notes = build_config(args)
        if args.plan:
            print_plan(cfg, params)
            return 0
        state = None
    if not args.no_reexec and argv is None:
        ensure_env(cfg["python_eval"])
    elif os.environ.get("PYTHONHASHSEED") != PINNED_HASHSEED:
        print("WARNING: PYTHONHASHSEED is not pinned: games are not reproducible across processes", file=sys.stderr)
    env = environment(cfg)
    if state is None:
        finish_context(cfg, env)
        validate(cfg, params)
        state = new_state(cfg, params, env)
        print_plan(cfg, params)
        save_state(args.state, state)
    else:
        old = state.get("env") or {}
        for key in ("code", "evaluator", "hashseed", "catanatron", "python"):
            if old.get(key) != env.get(key):
                print(f"WARNING: {key} changed since the run started ({old.get(key)} -> {env.get(key)}): the "
                      f"iterations from here on play another objective", file=sys.stderr)
        state["config"]["code"] = env["code"]
        if cfg["mode"] == "catanatron":
            finish_context(cfg, env)
    print(f"tune_joint: {args.state}: iteration {state['k']}/{state['config']['iterations']}, workers {args.workers}, "
          f"{env['evaluator']}, PYTHONHASHSEED={env['hashseed']}, code {env['code']}", flush=True)
    deadline = time.time() + 60.0 * args.max_minutes if args.max_minutes else None

    def progress(r):
        if args.quiet:
            return
        if "records" in r:
            parts = [f"{x['arm']} " + ("ERROR" if x.get("status") != "ok" else ("WIN" if x.get("won") else "loss"))
                     for x in r["records"]]
            print(f"    k={r['k']} seed {r['s']}: " + ", ".join(parts), file=sys.stderr, flush=True)
        elif r.get("status") != "ok":
            print(f"    k={r['k']} game {r['j']} ({r['pattern']}): ERROR {str(r.get('error'))[:120]}", file=sys.stderr,
                  flush=True)
        else:
            print(f"    k={r['k']} game {r['j']} ({r['pattern']}): winner seat {r['winner']}, vps {r['vps']}, "
                  f"{r['duration']:.1f}s", file=sys.stderr, flush=True)

    def runner(jobs):
        return run_jobs(jobs, max(1, args.workers), play_job, progress)

    def on_term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old_term = signal.signal(signal.SIGTERM, on_term)
    t0 = time.time()
    rc = 0
    try:
        status = run(state, args.state, runner, deadline)
    except KeyboardInterrupt:
        status, rc = "interrupted (the unfinished iteration is replayed on --resume)", 130
    except RuntimeError as ex:
        status, rc = f"STOPPED: {ex}", 1
    finally:
        signal.signal(signal.SIGTERM, old_term)
    print(f"\n{status} after {time.time() - t0:.0f} s\n")
    print_status(load_state(args.state), args.state)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
