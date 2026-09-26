#!/usr/bin/env python3
"""Paired ablations against Catanatron's bots: is a strategy term worth it against a strong opponent?

    # a registry tunable at candidate values vs its default, catanbot in 1 seat vs 3 x catanatron "value" (3.3):
    /home/user/venv_cat33/bin/python scripts/ablate_catanatron.py --tunable danger.TURNS_HALF --values 2,4.5 \\
        --opponent value --seeds 2000 --workers 2 --out runs/turns_half_value.jsonl
    python3 scripts/ablate_catanatron.py --tunable danger.danger_multiplier --flag-off --opponent vf \\
        --seeds 400 --workers 2 --out runs/danger_off_vf.jsonl                        # 3.2.1 stand-in opponent
    # two bot specs (search vs heuristic bot, depth 2 vs depth 1, ...):
    python3 scripts/ablate_catanatron.py --cand-spec search:depth=2,beam=4,expand=8,evaluator=heuristic \\
        --def-spec search:depth=1,beam=4,expand=8,evaluator=heuristic --opponent vf --seeds 400 --out d2.jsonl
    python3 scripts/ablate_catanatron.py --report --out runs/turns_half_value.jsonl   # statistics from the file
    python3 scripts/ablate_catanatron.py --plan ...                                  # runs / keys / jobs only

Design (see docs/ABLATIONS.md, "Paired ablations against Catanatron"):

* Every seed ``s`` is played once per ARM: the candidate catanbot seat and the default catanbot seat
  (one default game per seed is shared by every candidate value of the invocation), each against 3
  copies of the same catanatron opponent, with the SAME board / dice seed (``s + 1``, the seed
  ``scripts/bench_catanatron.py --seed 0`` gives its game ``s``) and the SAME seat (``s % 4``).  A
  worker plays all arms of a seed back to back.  The process runs with ``PYTHONHASHSEED=0`` (the
  script re-executes itself when it is not set: catanatron iterates sets of ``Color`` enums, whose
  order depends on the string hash seed), so a game is a deterministic function of (seed, arm) and
  the two arms are identical up to the first catanbot decision that differs (checked per pair from
  the action-log fingerprints, "pairing" in the report).
* Candidate overrides go through ``ParamBot`` (weights / flags via ``tuning.apply``, search knobs via
  ``tuning.apply_to_bot``); tunables read by the static evaluator re-execute with ``CATANBOT_NO_ACCEL=1``.
* Results are APPEND-ONLY JSONL (``--out``): a ``run`` line declares each comparison (run key = hash
  of the candidate arm and the default arm; an arm key hashes spec, overrides, adapter options,
  opponent, interpreter / catanatron version, evaluator mode, rules), then one ``game`` line per
  finished game.  Rerunning the same command skips every (arm, seed) already in the file, so a
  killed run resumes where it stopped; ``--report`` recomputes everything from the file alone.
  Records carry a fingerprint of the game-relevant code; a pair is only formed from two games played
  with the same code.
* Statistics per candidate: wins / win rate per arm, the paired win difference with its standard
  error over seeds, 95% interval, the paired VP difference (more sensitive), per-seat breakdown,
  decision-time cost (ours and the opponents'), adapter statistics and a verdict.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCHEMA = 1
HASH_PROBE_TEXT = "catanbot-ablate-hash-probe"
PINNED_HASHSEED = "0"
TRACE_EVERY = 16          # action-log fingerprint checkpoint every N actions
OURS_MAX = 32             # first N non-trivial catanbot decisions fingerprinted per game
OURS_VERSION = 2          # 2: ``ours`` also holds the adapter's follow-ups ([index, hash, 1]); records
                          # without ``ours_v`` (older script) hold consulted decisions only
MIN_VERDICT_PAIRS = 30
Z95 = 1.959964
# catanbot/ files that cannot change a game (screenshot parsing, training, CLI); everything else
# under catanbot/ plus the bench script and the C++ extension is fingerprinted.
CODE_EXCLUDE = ("vision", "cli.py", "__main__.py", "train.py")
KNOWN_RESULT_KEYS = {"seed", "winner", "winner_seat", "colors", "vps", "turns", "actions", "duration"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha(obj: Any, n: int = 16) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:6]


def game_seed(s: int) -> int:
    """Non-zero catanatron seed of seed number ``s`` (= ``bench_catanatron.game_seed(0, s)``)."""
    return s + 1


def seat_of(s: int) -> int:
    return s % 4


def catanatron_version() -> str:
    try:
        from importlib.metadata import version
        return version("catanatron")
    except Exception:  # noqa: BLE001
        return "none"


def json_safe(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def _num(x: Any) -> Any:
    """JSON-friendly copy of a stat value (numbers stay numbers, anything else becomes a string)."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else None
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _num(v) for k, v in x.items()}
    return str(x)


def parse_kv(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``["a=1", "b=true", "c=off"]`` -> ``{"a": 1, "b": True, "c": "off"}`` (JSON literals where possible)."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"error: expected NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


def fmt_pp(x: Optional[float], nd: int = 1, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:+.{nd}f}pp" if sign else f"{100.0 * x:.{nd}f}pp"


def fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    """Mean and standard error of the mean (unbiased sample variance); ``nan`` when undefined."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


# ---------------------------------------------------------------------------
# Decision-time summaries (per game, pooled over games through a log histogram)
# ---------------------------------------------------------------------------
def _ms_bin(ms: float) -> int:
    return int(math.floor(4.0 * math.log2(max(ms, 1e-3))))


def _bin_value(b: int) -> float:
    return 2.0 ** ((b + 0.5) / 4.0)


def timing_summary(seconds: Sequence[float]) -> Dict[str, Any]:
    """Per-game summary of decision times: count, total / mean / exact p95 / max in ms, log2/4 histogram."""
    ms = [1000.0 * x for x in seconds]
    n = len(ms)
    if not n:
        return {"n": 0, "ms": 0.0, "mean": None, "p95": None, "max": None, "h": {}}
    srt = sorted(ms)
    hist = collections.Counter(_ms_bin(x) for x in ms)
    return {"n": n, "ms": round(sum(ms), 3), "mean": round(sum(ms) / n, 4),
            "p95": round(srt[min(n - 1, int(math.ceil(0.95 * n)) - 1)], 4), "max": round(srt[-1], 4),
            "h": {str(k): v for k, v in sorted(hist.items())}}


def pool_timing(summaries: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Pooled mean (exact) and p95 (from the merged histograms, ~19% bin resolution) over games."""
    n = 0
    total = 0.0
    hist: Dict[int, int] = collections.Counter()
    for t in summaries:
        if not t or not t.get("n"):
            continue
        n += int(t["n"])
        total += float(t.get("ms") or 0.0)
        for k, v in (t.get("h") or {}).items():
            hist[int(k)] += int(v)
    if not n:
        return {"n": 0, "mean": float("nan"), "p95": float("nan")}
    target = 0.95 * sum(hist.values())
    cum = 0
    p95 = float("nan")
    for b in sorted(hist):
        cum += hist[b]
        if cum >= target:
            p95 = _bin_value(b)
            break
    return {"n": n, "mean": total / n, "p95": p95}


def make_trace(items: Sequence[str]) -> Dict[str, Any]:
    """Fingerprint of an action log: full hash plus a cumulative hash every ``TRACE_EVERY`` actions."""
    h = hashlib.sha1()
    ck = []
    for i, it in enumerate(items, 1):
        h.update(it.encode())
        h.update(b"\n")
        if i % TRACE_EVERY == 0:
            ck.append(h.copy().hexdigest()[:6])
    return {"n": len(items), "h": h.hexdigest()[:16], "every": TRACE_EVERY, "ck": ck}


# ---------------------------------------------------------------------------
# Code fingerprint and environment
# ---------------------------------------------------------------------------
def code_files() -> List[str]:
    files = []
    base = os.path.join(ROOT, "catanbot")
    for dirpath, dirnames, filenames in os.walk(base):
        rel = os.path.relpath(dirpath, base)
        if rel.split(os.sep)[0] in CODE_EXCLUDE or "__pycache__" in dirpath:
            continue
        for fn in filenames:
            if fn in CODE_EXCLUDE:
                continue
            if fn.endswith(".py") or (fn.startswith("catanbot_core") and fn.endswith(".so") and "staged" not in fn):
                files.append(os.path.join(dirpath, fn))
    files.append(os.path.join(ROOT, "scripts", "bench_catanatron.py"))
    return sorted(f for f in files if os.path.exists(f))


def code_fingerprint() -> str:
    """Short hash of every file that can change a game (catanbot package minus vision/CLI/training,
    the C++ extension, the bench script) plus the catanatron version.  Games are only paired with
    games played by the same code."""
    h = hashlib.sha1(catanatron_version().encode())
    for f in code_files():
        h.update(os.path.relpath(f, ROOT).encode())
        with open(f, "rb") as fh:
            h.update(hashlib.sha1(fh.read()).digest())
    return h.hexdigest()[:12]


def required_env(need_python_eval: bool) -> Dict[str, str]:
    env = {"PYTHONHASHSEED": PINNED_HASHSEED}
    if need_python_eval:
        env["CATANBOT_NO_ACCEL"] = "1"
    return env


def ensure_env(need_python_eval: bool) -> None:
    """Re-execute this script with ``PYTHONHASHSEED=0`` (and ``CATANBOT_NO_ACCEL=1`` for tunables read by
    the Python static evaluator) unless they are already set.  Both must be set before the interpreter
    starts / before ``catanbot`` is imported, hence the re-exec; forked workers inherit them."""
    want = required_env(need_python_eval)
    if all(os.environ.get(k) == v for k, v in want.items()):
        return
    missing = ", ".join(f"{k}={v}" for k, v in want.items() if os.environ.get(k) != v)
    print(f"re-executing with {missing}", flush=True)
    env = dict(os.environ, **want)
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)


def evaluator_mode() -> str:
    try:
        from catanbot import tuning
        return tuning.evaluator_mode()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Arms and runs
# ---------------------------------------------------------------------------
def arm_key(arm: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    """Identity of one arm's games: two records with the same key and seed are the same game."""
    key_ctx = {k: ctx.get(k) for k in ("opponent", "opponent_params", "python", "catanatron", "evaluator",
                                       "vps_to_win", "discard_limit", "hashseed")}
    if ctx.get("fake"):
        key_ctx["fake"] = ctx["fake"]["effect"]   # synthetic games: the effect size defines them, not the crash hooks
    return sha({"role": arm["role"], "spec": arm["spec"], "overrides": arm["overrides"], "adapter": arm["adapter"],
                "ctx": key_ctx})


def make_arm(role: str, spec: str, overrides: Dict[str, Any], adapter: Dict[str, Any], label: str) -> Dict[str, Any]:
    return {"role": role, "spec": spec, "overrides": dict(overrides), "adapter": dict(adapter), "label": label}


def _load_ablate():
    import importlib.util
    path = os.path.join(ROOT, "scripts", "ablate.py")
    spec = importlib.util.spec_from_file_location("_ablate_selfplay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_overrides(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``--set`` / ``--cand-set NAME=VALUE`` through the registry's parsers."""
    from catanbot import tuning
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"error: expected NAME=VALUE, got {item!r}")
        name, text = item.split("=", 1)
        try:
            t = tuning.find(name.strip())
        except KeyError as ex:
            raise SystemExit(f"error: {ex}")
        out[t.name] = t.parse(text.strip())
    return out


def build_runs(args) -> Tuple[List[Dict[str, Any]], bool]:
    """The comparisons of this invocation: ``[{cand arm, def arm, labels}]`` and whether the Python
    evaluator is needed.  Keys are filled in by :func:`finalize_runs` once the context is known."""
    from catanbot import tuning
    common = parse_overrides(args.set)
    cand_extra = parse_overrides(args.cand_set)
    adapter = parse_kv(args.adapter_opt)
    adapter["trades"] = args.trades
    info = getattr(args, "info", None) or "full"
    if info != "full":
        # only non-default information options enter the arm (full-information arm keys stay unchanged)
        adapter["info"] = info
        adapter["info_samples"] = int(getattr(args, "info_samples", None) or 4)
        if getattr(args, "discards_public", False):
            adapter["discards_public"] = True
    cand_adapter = dict(adapter, **parse_kv(args.cand_adapter_opt))
    runs: List[Dict[str, Any]] = []
    if args.tunable:
        if args.cand_spec or args.def_spec:
            raise SystemExit("error: use either --tunable or --cand-spec/--def-spec")
        ab = _load_ablate()
        try:
            t = tuning.find(args.tunable)
        except KeyError as ex:
            raise SystemExit(f"error: {ex}")
        values = ab.parse_values(t, args.values, args.flag_off)
        base = args.base_spec or (tuning.DEFAULT_DEEP_SPEC if t.requires_depth >= 2 else tuning.DEFAULT_SEARCH_SPEC)
        base = ab.resolve_spec(t, base)
        seen = set()
        for v in values:
            text = t.format(v)
            if text in seen:
                continue
            seen.add(text)
            v = json.loads(json.dumps(v))
            over = dict(common)
            over[t.name] = v
            over.update(cand_extra)
            cand = make_arm("cand", base, over, cand_adapter, f"{t.name}={text}")
            dflt = make_arm("def", base, common, adapter, f"default ({t.name}={t.format(t.default)})")
            runs.append({"tunable": t.name, "value": v, "value_text": text, "default_text": t.format(t.default),
                         "cand": cand, "def": dflt})
    else:
        if not (args.cand_spec and args.def_spec):
            raise SystemExit("error: --tunable, or both --cand-spec and --def-spec, are required")
        extra = ",".join(f"{k}={v}" for k, v in cand_extra.items())
        cand = make_arm("cand", args.cand_spec, dict(common, **cand_extra), cand_adapter,
                        args.cand_spec + (f" [{extra}]" if extra else ""))
        dflt = make_arm("def", args.def_spec, common, adapter, args.def_spec)
        if cand["spec"] == dflt["spec"] and cand["overrides"] == dflt["overrides"] and cand["adapter"] == dflt["adapter"]:
            print("note: candidate and default arms are identical (an A/A calibration run: every pair should be "
                  "an identical game)", flush=True)
        runs.append({"tunable": None, "value": None, "value_text": args.cand_spec, "default_text": args.def_spec,
                     "cand": cand, "def": dflt})
    need_python = any(tuning.find(n).needs_python_evaluator for r in runs for n in r["cand"]["overrides"])
    return runs, need_python


def parse_opponent_params(text: Optional[str]) -> Dict[str, str]:
    """``--opponent-params KEY=VAL,...`` (the bench's syntax; values stay strings, the bench coerces them)."""
    out: Dict[str, str] = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise SystemExit(f"error: --opponent-params: {item!r} is not KEY=VAL")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def make_context(args) -> Dict[str, Any]:
    fake = None
    if args.fake_games is not None:
        fake = {"effect": args.fake_games, "crash": sorted(args.fake_crash_seeds), "die": sorted(args.fake_die_seeds),
                "sleep": args.fake_sleep}
    return {"opponent": args.opponent, "opponent_params": parse_opponent_params(args.opponent_params),
            "python": platform.python_version(), "catanatron": catanatron_version(),
            "evaluator": evaluator_mode(), "vps_to_win": args.vps_to_win, "discard_limit": args.discard_limit,
            "hashseed": os.environ.get("PYTHONHASHSEED"), "fake": fake}


def finalize_runs(runs: List[Dict[str, Any]], ctx: Dict[str, Any], exp: str, seeds: Tuple[int, int]) -> None:
    for r in runs:
        r["cand_key"] = arm_key(r["cand"], ctx)
        r["def_key"] = arm_key(r["def"], ctx)
        r["run_key"] = sha([r["cand_key"], r["def_key"]])
        r["exp"] = exp
        r["seed_base"], r["seeds"] = seeds


def run_line(r: Dict[str, Any], ctx: Dict[str, Any], code: str) -> Dict[str, Any]:
    return {"v": SCHEMA, "kind": "run", "run_key": r["run_key"], "cand_key": r["cand_key"], "def_key": r["def_key"],
            "exp": r["exp"], "tunable": r["tunable"], "value": r["value"], "value_text": r["value_text"],
            "default_text": r["default_text"], "cand": r["cand"], "def": r["def"], "ctx": ctx,
            "seed_base": r["seed_base"], "seeds": r["seeds"], "code": code, "t": time.time()}


# ---------------------------------------------------------------------------
# Playing one arm of one seed (runs in a worker process)
# ---------------------------------------------------------------------------
class GameTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise GameTimeout("game exceeded --game-timeout")


def _worker_init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent handles ^C and shuts the pool down
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


_BENCH = None
_TRACED = None
_TIMED: Dict[type, type] = {}


def _bench():
    """``scripts/bench_catanatron.py`` as a module (opponent presets / resolution)."""
    global _BENCH
    if _BENCH is None:
        import importlib.util
        path = os.path.join(ROOT, "scripts", "bench_catanatron.py")
        spec = importlib.util.spec_from_file_location("_bench_catanatron", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BENCH = mod
    return _BENCH


def _traced_class():
    """``CatanbotPlayer`` that remembers the game and fingerprints its first non-trivial decisions."""
    global _TRACED
    if _TRACED is None:
        from catanbot.bench.catanatron_adapter import CatanbotPlayer, action_log

        class TracedCatanbotPlayer(CatanbotPlayer):
            """Records ``[action index, hash of the returned action]`` for the first ``OURS_MAX`` decisions
            in which the bot was consulted (its ParamBot counters moved) - with 3.3 trading that includes
            prompts with a single playable action, where the bot may still choose to offer a trade - and
            ``[action index, hash, 1]`` for the adapter's FOLLOW-UPS of an earlier decision, played without
            consulting the bot: the robber move of a knight (the bot picks hex and victim with the
            ``PLAY_KNIGHT``, catanatron logs ``PLAY_KNIGHT_CARD`` then ``MOVE_ROBBER``) and the 2nd..nth
            card of a 3.3 per-card discard.  Two arms whose knights differ only in the victim log the same
            ``PLAY_KNIGHT_CARD`` and part at the ``MOVE_ROBBER``; without the follow-up entry that pair was
            classified "inconsistent" although only our own decision made it diverge (``OURS_VERSION``)."""

            def decide(self, game, playable_actions):
                playable = list(playable_actions)
                self._ab_game = game
                st = getattr(self.bot, "stats", None)
                before = st.get("decisions", 0) + st.get("trivial", 0) if isinstance(st, dict) else None
                mine = self.stats
                follow_before = mine.get("pending_robber", 0) + mine.get("pending_discard", 0)
                a = super().decide(game, playable)
                ours = self.__dict__.setdefault("_ab_ours", [])
                if len(ours) < OURS_MAX:
                    if before is not None:
                        consulted = st.get("decisions", 0) + st.get("trivial", 0) != before
                    else:
                        consulted = len(playable) > 1
                    if consulted:
                        ours.append([len(action_log(game.state)), short_hash(repr(a))])
                    elif mine.get("pending_robber", 0) + mine.get("pending_discard", 0) != follow_before:
                        ours.append([len(action_log(game.state)), short_hash(repr(a)), 1])
                return a

        _TRACED = TracedCatanbotPlayer
    return _TRACED


def _timed_class(cls: type) -> type:
    """Subclass of an opponent class that times its decisions with more than one playable action."""
    if cls not in _TIMED:
        def decide(self, game, playable_actions):
            t0 = time.perf_counter()
            try:
                return cls.decide(self, game, playable_actions)
            finally:
                dt = time.perf_counter() - t0
                try:
                    many = len(playable_actions) > 1
                except TypeError:
                    many = True
                if many:
                    self.__dict__.setdefault("_ab_times", []).append(dt)
        _TIMED[cls] = type(cls.__name__, (cls,), {"decide": decide, "__module__": cls.__module__})
    return _TIMED[cls]


def _opponent_times(p) -> List[float]:
    """Seconds of an opponent's decisions with more than one playable action: the adapter's
    ``BenchOpponent.choice_times`` when the seat is wrapped in one, else our own timing."""
    v = p.__dict__.get("choice_times")
    if isinstance(v, list):
        return [float(x) for x in v]
    inner = p.__dict__.get("inner")
    if inner is not None and inner is not p:
        return _opponent_times(inner)
    return list(p.__dict__.get("_ab_times", []))


def _opponent_maker(ctx: Dict[str, Any]) -> Callable:
    """``make(color)`` for the opponent seats, through the bench's ``opponent_factory`` when it has one
    (``--opponent-params``) - the same construction as ``scripts/bench_catanatron.py``."""
    bench = _bench()
    cls = bench.resolve_opponent(ctx["opponent"])
    params = ctx.get("opponent_params") or None
    factory = getattr(bench, "opponent_factory", None)
    if factory is not None:
        return factory(cls, params, ctx["opponent"])
    if params:
        raise SystemExit("error: --opponent-params needs a bench_catanatron.py with opponent_factory")
    return cls


def _make_opponent(ad, make: Callable, color, adapter_opts: Dict[str, Any], vps_to_win: int):
    """An opponent seat as the bench builds it: ``BenchOpponent(player, trade_rule, vps_to_win)`` when the
    adapter has that wrapper (per-decision timing, the trade-answer rule), else the bare player; the
    player's class is swapped for its :func:`_timed_class` subclass so its decisions are timed anyway."""
    player = make(color)
    wrapper = getattr(ad, "BenchOpponent", None)
    if wrapper is None:
        player.__class__ = _timed_class(type(player))    # same instance, decide() timed
        return player
    return wrapper(player, trade_rule=adapter_opts.get("trades") or "off", vps_to_win=vps_to_win)


def _catanbot_kwargs(adapter_opts: Dict[str, Any]) -> Dict[str, Any]:
    """``CatanbotPlayer`` keyword arguments from the arm's adapter options: ``trades`` (the bench's
    ``--trades``) becomes ``suppress_trades=(trades == "off")``, the rest is passed through."""
    kw = {k: v for k, v in adapter_opts.items() if k != "trades"}
    if "trades" in adapter_opts:
        kw["suppress_trades"] = adapter_opts["trades"] in (None, "off")
    return kw


def _real_game(job: Dict[str, Any], arm: Dict[str, Any], s: int) -> Dict[str, Any]:
    from catanbot.agents.param_bot import ParamBot
    from catanbot.bench import catanatron_adapter as ad
    from catanbot.selfplay import make_bot
    ctx = job["ctx"]
    seat, gseed = seat_of(s), game_seed(s)
    make = _opponent_maker(ctx)
    pb = ParamBot(make_bot(arm["spec"]), arm["overrides"], label=arm["role"])
    me = _traced_class()(ad.COLORS[seat], spec=arm["spec"], bot=pb, seed=gseed, **_catanbot_kwargs(arm["adapter"]))
    players = [me if i == seat else _make_opponent(ad, make, c, arm["adapter"], ctx["vps_to_win"])
               for i, c in enumerate(ad.COLORS)]
    res = ad.play_game(players, seed=gseed, vps_to_win=ctx["vps_to_win"], discard_limit=ctx["discard_limit"])
    game = me.__dict__.get("_ab_game")
    log = ad.action_log(game.state) if game is not None else []
    items = [f"{ad.log_action(e)!r}|{ad.log_result(e)!r}" for e in log]
    opp_times: List[float] = []
    opp_extra: Dict[str, Any] = {}
    for i, p in enumerate(players):
        if i == seat:
            continue
        opp_times.extend(_opponent_times(p))
        ts = getattr(p, "trade_stats", None)
        if isinstance(ts, dict):
            for k, v in ts.items():
                if isinstance(v, (int, float)):
                    opp_extra[k] = opp_extra.get(k, 0) + v
    extra = {}
    for k, v in res.items():
        if k in KNOWN_RESULT_KEYS:
            continue
        try:
            text = json.dumps(_num(v))
        except (TypeError, ValueError):
            continue
        if len(text) <= 4000:
            extra[k] = _num(v)
    vps = [int(x) for x in res["vps"]]
    winner_seat = int(res.get("winner_seat", -1))
    return {
        "winner": res.get("winner"), "winner_seat": winner_seat, "won": winner_seat == seat,
        "our_vp": vps[seat], "opp_vps": [v for i, v in enumerate(vps) if i != seat], "vps": vps,
        "turns": int(res.get("turns", 0)), "actions": int(res.get("actions", len(log))),
        "truncated": res.get("winner") is None, "duration": round(float(res.get("duration", 0.0)), 4),
        "dec": timing_summary(pb.stats["times"]), "trivial": int(pb.stats["trivial"]),
        "dec_adapter": timing_summary(me.__dict__.get("choice_times") or []),
        "overhead_ms": round(1000.0 * pb.stats["overhead"], 3),
        "opp_dec": timing_summary(opp_times), "opp_trade_stats": _num(opp_extra),
        "adapter": _num(dict(me.stats)), "unmapped": _num(dict(getattr(me, "unmapped_kinds", {}))),
        "trace": make_trace(items), "ours": list(me.__dict__.get("_ab_ours", [])), "ours_v": OURS_VERSION,
        "extra": extra,
    }


def _fake_game(job: Dict[str, Any], arm: Dict[str, Any], s: int) -> Dict[str, Any]:
    """Synthetic game (tests): both arms share the seed's luck ``u``; the candidate wins with probability
    ``0.25 + effect``, the default with 0.25, so pairs differ only when ``0.25 <= u < 0.25 + effect``."""
    fk = job["ctx"]["fake"]
    cand = arm["role"] == "cand"
    if cand and s in fk["crash"]:
        raise RuntimeError(f"synthetic crash in seed {s}")
    if cand and s in fk["die"]:
        os._exit(3)   # a hard crash: the worker process dies without a Python exception
    if fk.get("sleep"):
        time.sleep(fk["sleep"])
    u = random.Random(f"fake-{s}").random()
    p = 0.25 + (fk["effect"] if cand else 0.0)
    won = u < p
    seat = seat_of(s)
    our_vp = 10 if won else 2 + int(u * 7)
    opp = [10 if (not won and i == 0) else 3 + (s + i) % 5 for i in range(3)]
    vps = opp[:seat] + [our_vp] + opp[seat:]
    differs = cand and (0.25 <= u < p)
    base = [f"a{s}-{i}" for i in range(60)]
    items = base[:40] + ([f"c{s}-{i}" for i in range(20)] if differs else base[40:])
    ours = [[10 * k + 3, short_hash(f"{s}-{k}")] for k in range(6)]
    if differs:
        ours[4] = [43, short_hash(f"{s}-cand")]
    return {
        "winner": "RED" if won else "BLUE", "winner_seat": seat if won else (seat + 1) % 4, "won": won,
        "our_vp": our_vp, "opp_vps": opp, "vps": vps, "turns": 80, "actions": 60, "truncated": False,
        "duration": 0.01, "dec": timing_summary([0.002 + 0.001 * cand] * 5 + [0.01]), "trivial": 3,
        "overhead_ms": 0.0, "opp_dec": timing_summary([0.001] * 10), "opp_trade_stats": {},
        "adapter": {"errors": 0, "fallback": 0, "trade_prompts": 0}, "unmapped": {},
        "trace": make_trace(items), "ours": ours, "extra": {},
    }


def _base_record(job: Dict[str, Any], arm: Dict[str, Any]) -> Dict[str, Any]:
    ctx = job["ctx"]
    s = job["s"]
    rec = {"v": SCHEMA, "kind": "game", "arm_key": arm["key"], "arm": arm["role"], "label": arm["label"],
           "exp": job["exp"], "spec": arm["spec"], "overrides": arm["overrides"], "adapter_opts": arm["adapter"],
           "opponent": ctx["opponent"], "catanatron": ctx["catanatron"], "python": ctx["python"],
           "evaluator": evaluator_mode() if not ctx.get("fake") else ctx["evaluator"], "code": job["code"],
           "s": s, "game_seed": game_seed(s), "seat": seat_of(s), "pid": os.getpid(),
           "hashseed": os.environ.get("PYTHONHASHSEED"), "hash_probe": hash(HASH_PROBE_TEXT)}
    if arm["role"] == "cand":
        rec["run_key"] = arm.get("run_key")
        rec["def_key"] = arm.get("def_key")
    return rec


def _error_record(job: Dict[str, Any], arm: Dict[str, Any], message: str) -> Dict[str, Any]:
    rec = _base_record(job, arm)
    rec.update({"status": "error", "error": message[-2000:], "t": time.time()})
    return rec


def play_arm(job: Dict[str, Any], arm: Dict[str, Any]) -> Dict[str, Any]:
    """One game of one arm; a Python exception (or ``--game-timeout``) becomes an error record."""
    timeout = int(job.get("timeout") or 0)
    use_alarm = timeout > 0 and threading.current_thread() is threading.main_thread()
    t0 = time.time()
    try:
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
        try:
            body = (_fake_game if job["ctx"].get("fake") else _real_game)(job, arm, job["s"])
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
    except KeyboardInterrupt:
        raise
    except BaseException as ex:  # noqa: BLE001  (SystemExit from an opponent resolver included)
        tb = traceback.format_exception(type(ex), ex, ex.__traceback__)
        return _error_record(job, arm, f"{type(ex).__name__}: {ex} | " + "".join(tb[-3:]))
    rec = _base_record(job, arm)
    rec.update(body)
    rec.update({"status": "ok", "wall": round(time.time() - t0, 4), "t": time.time()})
    return rec


def play_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """All arms of one seed, back to back in this process."""
    return {"s": job["s"], "records": [play_arm(job, arm) for arm in job["arms"]]}


# ---------------------------------------------------------------------------
# The JSONL store
# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """All parseable lines of a JSONL file (a line cut short by a kill is skipped) and the bad-line count."""
    out: List[Dict[str, Any]] = []
    bad = 0
    if not os.path.exists(path):
        return out, 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(obj, dict):
                out.append(obj)
            else:
                bad += 1
    return out, bad


class JsonlWriter:
    """Append-only writer: one ``write`` per line (O_APPEND), flushed and fsynced; a partial last line
    left by a killed writer is terminated first so it cannot swallow the next record."""

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                last = fh.read(1)
            if last != b"\n":
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n")

    def append(self, records: Sequence[Dict[str, Any]]) -> None:
        if not records:
            return
        data = "".join(json.dumps(json_safe(r), sort_keys=True, separators=(",", ":")) + "\n" for r in records)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())


class Index:
    """Records of one JSONL file keyed by (arm key, seed); runs by run key (latest declaration wins)."""

    def __init__(self, records: Iterable[Dict[str, Any]] = ()):
        self.games: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.stops: Dict[str, Dict[str, Any]] = {}
        self.order = 0
        for r in records:
            self.add(r)

    def add(self, r: Dict[str, Any]) -> None:
        kind = r.get("kind")
        self.order += 1
        if kind == "game":
            r["_order"] = self.order
            self.games[(r.get("arm_key"), int(r.get("s", -1)))].append(r)
        elif kind == "run":
            self.runs[r["run_key"]] = r
        elif kind == "stop":
            self.stops[r["run_key"]] = r

    def get(self, key: str, s: int) -> List[Dict[str, Any]]:
        return self.games.get((key, s), [])

    def has(self, key: str, s: int, code: str, retry_errors: bool = False) -> bool:
        return any(r.get("code") == code and (r.get("status") == "ok" or not retry_errors) for r in self.get(key, s))

    def pair(self, run: Dict[str, Any], s: int, retry_errors: bool = False
             ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """The latest (cand, def) records of seed ``s`` played with the same code (prefer ok/ok)."""
        cands = self.get(run["cand_key"], s)
        defs = self.get(run["def_key"], s)
        best = None
        for c in cands:
            for d in defs:
                if c.get("code") != d.get("code"):
                    continue
                ok = c.get("status") == "ok" and d.get("status") == "ok"
                if retry_errors and not ok:
                    continue
                rank = (ok, max(c["_order"], d["_order"]))
                if best is None or rank > best[0]:
                    best = (rank, c, d)
        return (best[1], best[2]) if best else (None, None)

    def seeds_of(self, run: Dict[str, Any]) -> List[int]:
        ss = {s for (k, s) in self.games if k in (run["cand_key"], run["def_key"])}
        return sorted(ss)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def divergence(c: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, Any]:
    """Where the two games of a pair part: ``identical`` (same action log), else the first catanbot
    decision that differs (``decision`` = its number, ``at`` = its action index) and whether the logs
    agreed up to it (``consistent``: nothing but our own decision made the games diverge)."""
    tc, td = c.get("trace"), d.get("trace")
    if not tc or not td:
        return {"known": False}
    if tc["h"] == td["h"] and tc["n"] == td["n"]:
        return {"known": True, "identical": True}
    every = tc.get("every", TRACE_EVERY)
    ckc, ckd = tc.get("ck", []), td.get("ck", [])
    k = next((i for i, (a, b) in enumerate(zip(ckc, ckd)) if a != b), None)
    lo = (k if k is not None else min(len(ckc), len(ckd))) * every      # first differing action index >= lo
    hi = (k + 1) * every if k is not None else None                     # ... and < hi when a checkpoint differs
    oc, od = c.get("ours") or [], d.get("ours") or []
    full = min(len(oc), len(od)) >= OURS_MAX
    if not (c.get("ours_v") == d.get("ours_v") == OURS_VERSION):
        # a record from an older script has no follow-up entries: compare the consulted decisions only
        oc = [x for x in oc if len(x) == 2]
        od = [x for x in od if len(x) == 2]
    j = next((i for i, (a, b) in enumerate(zip(oc, od)) if list(a) != list(b)), None)
    if j is None:
        last = max([x[0] for x in oc[-1:]] + [x[0] for x in od[-1:]] + [-1])
        # no stored decision differs: consistent only if the logs part after the last stored decision
        consistent = (hi is None or hi > last) if full else False
        return {"known": True, "identical": False, "decision": None, "at": None, "diverge_from": lo,
                "consistent": consistent}
    at_c, at_d = oc[j][0], od[j][0]
    if at_c != at_d:
        return {"known": True, "identical": False, "decision": j, "at": min(at_c, at_d), "diverge_from": lo,
                "consistent": False}
    consistent = lo <= at_c and (hi is None or at_c < hi)
    return {"known": True, "identical": False, "decision": j, "at": at_c, "diverge_from": lo, "consistent": consistent}


def _median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def verdict(delta: float, se: float, n: int, what: str = "candidate") -> str:
    if n < MIN_VERDICT_PAIRS or math.isnan(delta):
        return f"inconclusive (< {MIN_VERDICT_PAIRS} pairs)"
    if math.isnan(se):
        return "no detectable difference at 95%"
    lo, hi = delta - Z95 * se, delta + Z95 * se
    if lo > 0:
        return f"{what} better"
    if hi < 0:
        return f"{what} worse"
    return "no detectable difference at 95%"


def pair_stats(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """Statistics of complete (ok, ok) pairs ``(candidate record, default record)``.

    ``delta`` = mean over seeds of ``won_cand - won_def`` (each in {0, 1}) = the difference of the two
    arms' win rates; ``se`` its standard error from the per-seed differences (unbiased variance), ``ci95``
    ``delta +- 1.96 se``; ``vp_delta`` / ``vp_se`` likewise for catanbot's final VP."""
    n = len(pairs)
    diffs = [int(bool(c["won"])) - int(bool(d["won"])) for c, d in pairs]
    vpd = [float(c["our_vp"]) - float(d["our_vp"]) for c, d in pairs]
    delta, se = mean_se(diffs)
    vp_delta, vp_se = mean_se(vpd)
    wc = sum(1 for c, _ in pairs if c["won"])
    wd = sum(1 for _, d in pairs if d["won"])
    conc = {"both": sum(1 for c, d in pairs if c["won"] and d["won"]),
            "cand_only": sum(1 for c, d in pairs if c["won"] and not d["won"]),
            "def_only": sum(1 for c, d in pairs if d["won"] and not c["won"]),
            "neither": sum(1 for c, d in pairs if not c["won"] and not d["won"])}
    seats = {}
    for k in range(4):
        sub = [(c, d) for c, d in pairs if int(c["seat"]) == k]
        dk, sek = mean_se([int(bool(c["won"])) - int(bool(d["won"])) for c, d in sub])
        seats[str(k)] = {"n": len(sub), "cand_wins": sum(1 for c, _ in sub if c["won"]),
                         "def_wins": sum(1 for _, d in sub if d["won"]), "delta": dk, "se": sek}
    # Decision times are only comparable between games played together (same worker, same load): a
    # default game copied from another run (``reused_from``) ran at another time, so timing uses the
    # co-played pairs when there are any.
    co = [(c, d) for c, d in pairs if not c.get("reused_from") and not d.get("reused_from")]
    tpairs = co or list(pairs)
    tc = pool_timing(c.get("dec") for c, _ in tpairs)
    td = pool_timing(d.get("dec") for _, d in tpairs)
    oc = pool_timing(c.get("opp_dec") for c, _ in tpairs)
    od = pool_timing(d.get("opp_dec") for _, d in tpairs)

    def total(recs, key):
        out: Dict[str, float] = collections.Counter()
        for r in recs:
            for k2, v in (r.get(key) or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k2] += v
        return {k2: round(v, 4) for k2, v in sorted(out.items())}

    div = [divergence(c, d) for c, d in pairs]
    known = [x for x in div if x.get("known")]
    ident = sum(1 for x in known if x["identical"])
    nonid = [x for x in known if not x["identical"]]
    ci = [delta - Z95 * se, delta + Z95 * se] if not math.isnan(se) else [float("nan"), float("nan")]
    vci = [vp_delta - Z95 * vp_se, vp_delta + Z95 * vp_se] if not math.isnan(vp_se) else [float("nan")] * 2
    return {
        "pairs": n, "cand_wins": wc, "def_wins": wd,
        "win_rate_cand": wc / n if n else float("nan"), "win_rate_def": wd / n if n else float("nan"),
        "delta": delta, "se": se, "ci95": ci, "mde80": 2.8 * se if not math.isnan(se) else float("nan"),
        "vp_cand": sum(float(c["our_vp"]) for c, _ in pairs) / n if n else float("nan"),
        "vp_def": sum(float(d["our_vp"]) for _, d in pairs) / n if n else float("nan"),
        "vp_delta": vp_delta, "vp_se": vp_se, "vp_ci95": vci,
        "opp_vp_cand": _mean_of(sum(c["opp_vps"]) / 3.0 for c, _ in pairs),
        "opp_vp_def": _mean_of(sum(d["opp_vps"]) / 3.0 for _, d in pairs),
        "concordance": conc, "seats": seats,
        "ms_cand": tc["mean"], "ms_p95_cand": tc["p95"], "decisions_cand": tc["n"],
        "ms_def": td["mean"], "ms_p95_def": td["p95"], "decisions_def": td["n"],
        "extra_ms": tc["mean"] - td["mean"] if tc["n"] and td["n"] else float("nan"),
        "opp_ms_cand": oc["mean"], "opp_ms_p95_cand": oc["p95"], "opp_ms_def": od["mean"], "opp_ms_p95_def": od["p95"],
        "timing_pairs": len(tpairs), "timing_coplayed": bool(co),
        "turns_cand": _mean_of(c["turns"] for c, _ in pairs), "turns_def": _mean_of(d["turns"] for _, d in pairs),
        "sec_cand": _mean_of(c["duration"] for c, _ in pairs), "sec_def": _mean_of(d["duration"] for _, d in pairs),
        "truncated_cand": sum(1 for c, _ in pairs if c.get("truncated")),
        "truncated_def": sum(1 for _, d in pairs if d.get("truncated")),
        "adapter_cand": total([c for c, _ in pairs], "adapter"), "adapter_def": total([d for _, d in pairs], "adapter"),
        "opp_trades_cand": total([c for c, _ in pairs], "opp_trade_stats"),
        "opp_trades_def": total([d for _, d in pairs], "opp_trade_stats"),
        "identical": ident, "diverged": len(nonid),
        "consistent": sum(1 for x in nonid if x["consistent"]),
        "inconsistent": sum(1 for x in nonid if not x["consistent"]),
        "median_first_decision": _median([x["decision"] for x in nonid if x.get("decision") is not None]),
        "median_first_action": _median([x["at"] for x in nonid if x.get("at") is not None]),
        "verdict": verdict(delta, se, n), "vp_verdict": verdict(vp_delta, vp_se, n),
    }


def _mean_of(xs: Iterable[float]) -> float:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def run_stats(index: Index, run: Dict[str, Any], retry_errors: bool = False) -> Dict[str, Any]:
    """Pair the records of ``run`` seed by seed and summarise (plus bookkeeping counts)."""
    pairs = []
    errors = 0
    error_msgs: List[str] = []
    incomplete = 0
    codes = set()
    seeds = index.seeds_of(run)
    for s in seeds:
        c, d = index.pair(run, s, retry_errors)
        if c is None:
            incomplete += 1
            continue
        if c.get("status") != "ok" or d.get("status") != "ok":
            errors += 1
            for r in (c, d):
                if r.get("status") != "ok" and len(error_msgs) < 3:
                    error_msgs.append(f"seed {s} {r.get('arm')}: {str(r.get('error'))[:160]}")
            continue
        pairs.append((c, d))
        codes.add(c.get("code"))
    st = pair_stats(pairs)
    st.update({"errors": errors, "error_samples": error_msgs, "incomplete": incomplete, "codes": sorted(codes),
               "seed_min": min((c["s"] for c, _ in pairs), default=None),
               "seed_max": max((c["s"] for c, _ in pairs), default=None)})
    return st


def load_index(path: str) -> Tuple[Index, int]:
    recs, bad = read_jsonl(path)
    return Index(recs), bad


def report_runs(index: Index, retry_errors: bool = False) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    return [(run, run_stats(index, run, retry_errors)) for run in index.runs.values()]


def print_run(run: Dict[str, Any], st: Dict[str, Any], stopped: Optional[Dict[str, Any]] = None,
              out=sys.stdout) -> None:
    ctx = run.get("ctx") or {}
    p = lambda *a: print(*a, file=out)   # noqa: E731
    p(f"run {run['run_key']} [{run.get('exp')}]: {run['cand']['label']}  vs  {run['def']['label']}")
    p(f"  setting   base spec {run['cand']['spec']}" + (f" / default spec {run['def']['spec']}"
                                                           if run['def']['spec'] != run['cand']['spec'] else "")
      + f"; 1 catanbot seat vs 3x {ctx.get('opponent')} (catanatron {ctx.get('catanatron')}, python "
        f"{ctx.get('python')}, {ctx.get('evaluator')}); adapter {run['cand']['adapter'] or '{}'}"
      + (f" / {run['def']['adapter']}" if run['def']['adapter'] != run['cand']['adapter'] else ""))
    planned = run.get("seeds")
    p(f"  pairs     {st['pairs']} complete" + (f" of {planned} planned" if planned else "")
      + f" (seeds {st['seed_min']}..{st['seed_max']}); {st['errors']} with an error, {st['incomplete']} incomplete"
      + (f"; code versions {len(st['codes'])}" if len(st["codes"]) > 1 else "")
      + (f"; STOPPED EARLY at {stopped['pairs']} pairs (sequential stop, see the caveat)" if stopped else ""))
    if not st["pairs"]:
        for m in st["error_samples"]:
            p(f"  error     {m}")
        return
    lo, hi = st["ci95"]
    p(f"  win rate  cand {100 * st['win_rate_cand']:.1f}% ({st['cand_wins']})  default {100 * st['win_rate_def']:.1f}% "
      f"({st['def_wins']})  delta {fmt_pp(st['delta'])} +- {fmt_pp(st['se'], sign=False)}  95% CI "
      f"[{fmt_pp(lo)}, {fmt_pp(hi)}]  (80%-power detectable ~{fmt_pp(st['mde80'], sign=False)})")
    vlo, vhi = st["vp_ci95"]
    p(f"  avg VP    cand {fmt(st['vp_cand'])}  default {fmt(st['vp_def'])}  delta {st['vp_delta']:+.3f} +- "
      f"{fmt(st['vp_se'], 3)}  95% CI [{fmt(vlo, 3)}, {fmt(vhi, 3)}];  opponents avg {fmt(st['opp_vp_cand'])} / "
      f"{fmt(st['opp_vp_def'])}")
    cc = st["concordance"]
    p(f"  pairs by outcome: both won {cc['both']}, only cand {cc['cand_only']}, only default {cc['def_only']}, "
      f"neither {cc['neither']}")
    seats = "  ".join(f"s{k} n={v['n']} {v['cand_wins']}-{v['def_wins']} {fmt_pp(v['delta'])}+-{fmt_pp(v['se'], sign=False)}"
                      for k, v in st["seats"].items())
    p(f"  by seat   {seats}")
    tnote = (f"{st['timing_pairs']} co-played pairs" if st.get("timing_coplayed") else
             "WARNING: every default game was reused from another run, times not comparable")
    p(f"  time      [{tnote}] catanbot ms/decision cand {fmt(st['ms_cand'])} (p95 {fmt(st['ms_p95_cand'], 1)}) vs default "
      f"{fmt(st['ms_def'])} (p95 {fmt(st['ms_p95_def'], 1)}), extra {fmt(st['extra_ms'])} ms; opponents "
      f"{fmt(st['opp_ms_cand'])} (p95 {fmt(st['opp_ms_p95_cand'], 1)}) / {fmt(st['opp_ms_def'])} ms; "
      f"s/game {fmt(st['sec_cand'])} / {fmt(st['sec_def'])}; turns {fmt(st['turns_cand'], 1)} / {fmt(st['turns_def'], 1)}; "
      f"truncated {st['truncated_cand']} / {st['truncated_def']}")
    keys = ("errors", "fallback", "unmapped_top", "observe_errors", "trade_prompts")
    extra_keys = sorted(k for k in set(st["adapter_cand"]) | set(st["adapter_def"])
                        if ("trade" in k or "offer" in k) and k not in keys)
    ac = ", ".join(f"{k} {st['adapter_cand'].get(k, 0):g}/{st['adapter_def'].get(k, 0):g}" for k in keys + tuple(extra_keys))
    p(f"  adapter   (cand/default totals) {ac}")
    if any(st["opp_trades_cand"].values()) or any(st["opp_trades_def"].values()):
        p(f"  opp trades cand {st['opp_trades_cand']} / default {st['opp_trades_def']}")
    mf = st["median_first_decision"]
    p(f"  pairing   identical games {st['identical']}/{st['pairs']}; diverged {st['diverged']} "
      f"(consistent {st['consistent']}, inconsistent {st['inconsistent']})"
      + (f"; median first differing catanbot decision #{mf:.0f} at action {st['median_first_action']:.0f}"
         if mf is not None and st["median_first_action"] is not None else ""))
    p(f"  verdict   {st['verdict']} (win rate); {st['vp_verdict'].replace('candidate', 'candidate VP')} (VP)")


def print_report(path: str, retry_errors: bool = False, out=sys.stdout) -> List[Tuple[Dict, Dict]]:
    index, bad = load_index(path)
    rows = report_runs(index, retry_errors)
    ngames = sum(len(v) for v in index.games.values())
    print(f"{path}: {len(index.runs)} run(s), {ngames} game records" + (f", {bad} unparseable line(s) skipped" if bad else ""),
          file=out)
    for run, st in rows:
        stop = index.stops.get(run["run_key"])
        print_run(run, st, stop, out=out)
    return rows


# ---------------------------------------------------------------------------
# Planning / resuming
# ---------------------------------------------------------------------------
def should_stop(st: Dict[str, Any], stop_se: Optional[float], min_pairs: int) -> bool:
    """Sequential stop: enough pairs, s.e. below the target and |delta| beyond 3 s.e."""
    if not stop_se or st["pairs"] < max(min_pairs, 2) or math.isnan(st["se"]):
        return False
    return st["se"] <= stop_se and abs(st["delta"]) > 3.0 * st["se"]


class ReusePool:
    """Default-arm games already played elsewhere (other experiment files) with the same arm key and code."""

    def __init__(self, patterns: Sequence[str], keys: Sequence[str], code: str, exclude: str):
        self.found: Dict[Tuple[str, int], Tuple[Dict[str, Any], str]] = {}
        wanted = set(keys)
        files = sorted({f for pat in patterns for f in glob.glob(pat)})
        for f in files:
            if os.path.abspath(f) == os.path.abspath(exclude):
                continue
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"kind":"game"' not in line or not any(k in line for k in wanted):
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("arm_key") in wanted and r.get("status") == "ok" and r.get("code") == code:
                        self.found[(r["arm_key"], int(r["s"]))] = (r, f)

    def take(self, key: str, s: int) -> Optional[Dict[str, Any]]:
        got = self.found.get((key, s))
        if got is None:
            return None
        r, f = got
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        r["reused_from"] = os.path.basename(f)
        return r


class Campaign:
    """Plans the jobs (seed -> arms still missing), writes results, tracks sequential stops."""

    def __init__(self, runs: List[Dict[str, Any]], out: str, ctx: Dict[str, Any], code: str, seeds: Sequence[int],
                 exp: str, timeout: int, retry_errors: bool = False, stop_se: Optional[float] = None,
                 stop_min_pairs: int = 100, reuse: Optional[ReusePool] = None, quiet: bool = False,
                 log=sys.stderr):
        self.runs = runs
        self.out = out
        self.ctx = ctx
        self.code = code
        self.seeds = list(seeds)
        self.exp = exp
        self.timeout = timeout
        self.retry_errors = retry_errors
        self.stop_se = stop_se
        self.stop_min_pairs = stop_min_pairs
        self.reuse = reuse
        self.quiet = quiet
        self.log = log
        index, bad = load_index(out)
        if bad:
            print(f"note: {bad} unparseable line(s) in {out} skipped (a write cut short by a kill)", file=log)
        self.index = index
        self.writer = JsonlWriter(out)
        self.stopped: Dict[str, Dict[str, Any]] = {}
        self.played = 0
        self.reused = 0
        self.errors = 0
        self.t_start = time.time()
        decl = [run_line(r, ctx, code) for r in runs if self._needs_declaration(r)]
        self._write(decl)
        for r in runs:
            if self.stop_se:
                st = run_stats(self.index, r)
                if should_stop(st, self.stop_se, self.stop_min_pairs):
                    self.stopped[r["run_key"]] = {"pairs": st["pairs"]}

    def _needs_declaration(self, r: Dict[str, Any]) -> bool:
        old = self.index.runs.get(r["run_key"])
        return old is None or old.get("seeds") != r["seeds"] or old.get("seed_base") != r["seed_base"] \
            or old.get("exp") != r["exp"]

    def _write(self, records: Sequence[Dict[str, Any]]) -> None:
        self.writer.append(records)
        for rec in records:
            self.index.add(json.loads(json.dumps(json_safe(rec))))

    def active(self) -> List[Dict[str, Any]]:
        return [r for r in self.runs if r["run_key"] not in self.stopped]

    def arms_needed(self, s: int) -> List[Dict[str, Any]]:
        need: Dict[str, Dict[str, Any]] = {}
        for r in self.active():
            c, d = self.index.pair(r, s, self.retry_errors)
            if c is not None:
                continue
            if not self.index.has(r["def_key"], s, self.code, self.retry_errors):
                got = self.reuse.take(r["def_key"], s) if self.reuse else None
                if got is not None:
                    self._write([got])
                    self.reused += 1
                else:
                    need.setdefault(r["def_key"], dict(r["def"], key=r["def_key"]))
            if not self.index.has(r["cand_key"], s, self.code, self.retry_errors):
                need.setdefault(r["cand_key"], dict(r["cand"], key=r["cand_key"], run_key=r["run_key"],
                                                    def_key=r["def_key"]))
        # default arm first, then the candidates in declaration order
        return sorted(need.values(), key=lambda a: a["role"] != "def")

    def make_job(self, s: int, arms: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"s": s, "arms": arms, "ctx": self.ctx, "code": self.code, "exp": self.exp, "timeout": self.timeout}

    def jobs(self) -> Iterator[Dict[str, Any]]:
        for s in self.seeds:
            if not self.active():
                return
            arms = self.arms_needed(s)
            if arms:
                yield self.make_job(s, arms)

    def pending_jobs(self) -> int:
        """Seeds that still need at least one game (does not copy reusable records)."""
        n = 0
        for s in self.seeds:
            for r in self.active():
                if self.index.pair(r, s, self.retry_errors)[0] is None:
                    n += 1
                    break
        return n

    def handle(self, result: Dict[str, Any]) -> None:
        recs = result["records"]
        self._write(recs)
        self.played += sum(1 for r in recs if r.get("status") == "ok")
        self.errors += sum(1 for r in recs if r.get("status") != "ok")
        if not self.quiet:
            parts = []
            for r in recs:
                if r.get("status") != "ok":
                    parts.append(f"{r['arm']} ERROR {str(r.get('error'))[:100]}")
                else:
                    parts.append(f"{r['arm']}{'' if r['arm'] == 'def' else ' ' + r['label']}: "
                                 f"{'WIN ' if r['won'] else 'loss'} vp {r['our_vp']} {r['duration']:.1f}s")
            print(f"  seed {result['s']} seat {seat_of(result['s'])}: " + " | ".join(parts), file=self.log, flush=True)
        for r in self.active():
            if not any(rec.get("arm_key") in (r["cand_key"], r["def_key"]) for rec in recs):
                continue
            if self.stop_se:
                st = run_stats(self.index, r)
                if should_stop(st, self.stop_se, self.stop_min_pairs):
                    self.stopped[r["run_key"]] = {"pairs": st["pairs"]}
                    self._write([{"v": SCHEMA, "kind": "stop", "run_key": r["run_key"], "pairs": st["pairs"],
                                  "delta": st["delta"], "se": st["se"], "stop_se": self.stop_se, "t": time.time()}])
                    print(f"  sequential stop: {r['cand']['label']} after {st['pairs']} pairs, delta "
                          f"{fmt_pp(st['delta'])} +- {fmt_pp(st['se'], sign=False)}", file=self.log, flush=True)

    def error_result(self, job: Dict[str, Any], message: str) -> Dict[str, Any]:
        return {"s": job["s"], "records": [_error_record(job, arm, message) for arm in job["arms"]]}


# ---------------------------------------------------------------------------
# Execution: a fork pool over seeds, surviving crashed games and dead workers
# ---------------------------------------------------------------------------
class Stop(Exception):
    pass


def _close(executor: Optional[ProcessPoolExecutor], force: bool = False) -> None:
    """Shut a pool down and join its manager thread; ``force`` first terminates the worker processes
    (interrupt, broken pool).  Joining matters: an executor left half shut down makes the interpreter's
    exit hook write to a closed pipe."""
    if executor is None:
        return
    if force:
        for p in list((getattr(executor, "_processes", None) or {}).values()):
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass


def _new_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_worker_init)


def _run_solo(camp: Campaign, job: Dict[str, Any]) -> None:
    """Re-play a job whose worker died, one arm per fresh single-worker pool, so a hard crash is pinned
    on the arm that causes it (recorded as an error) and the other arms still count."""
    for arm in job["arms"]:
        one = dict(job, arms=[arm])
        pool = _new_pool(1)
        try:
            res = pool.submit(play_job, one).result()
        except BrokenProcessPool:
            res = camp.error_result(one, "worker process died while playing this game (hard crash)")
        except Exception as ex:  # noqa: BLE001
            res = camp.error_result(one, f"{type(ex).__name__}: {ex}")
        finally:
            _close(pool)
        camp.handle(res)


def execute(camp: Campaign, workers: int, deadline: Optional[float] = None) -> str:
    """Play every missing game; returns ``"done"`` or ``"deadline"``.

    At most ``workers`` seeds are in flight, so when a worker process dies (a hard crash: no Python
    exception to record) the suspects are exactly the in-flight seeds: they are re-played one game
    per fresh process (:func:`_run_solo`), the game that kills its process again is recorded as an
    error, and the pool is rebuilt."""
    jobs = camp.jobs()
    if workers <= 0:                      # in-process (debugging / tests)
        for job in jobs:
            if deadline and time.time() >= deadline:
                return "deadline"
            camp.handle(play_job(job))
        return "done"
    pool = _new_pool(workers)
    inflight: Dict[Any, Dict[str, Any]] = {}
    retry: collections.deque = collections.deque()   # jobs a broken pool refused (not suspects)
    exhausted = False
    status = "done"
    clean = False
    try:
        while True:
            broken = False
            while not exhausted and len(inflight) < workers:
                if deadline and time.time() >= deadline:
                    exhausted = True
                    status = "deadline"
                    break
                job = retry.popleft() if retry else next(jobs, None)
                if job is None:
                    exhausted = True
                    break
                try:
                    inflight[pool.submit(play_job, job)] = job
                except BrokenProcessPool:
                    retry.appendleft(job)
                    broken = True
                    break
            if not inflight and not broken:
                break
            suspects: List[Dict[str, Any]] = []
            if not broken:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for f in done:
                    job = inflight.pop(f)
                    try:
                        res = f.result()
                    except BrokenProcessPool:
                        suspects.append(job)
                        broken = True
                        continue
                    except Exception as ex:  # noqa: BLE001  (e.g. an unpicklable result)
                        res = camp.error_result(job, f"{type(ex).__name__}: {ex}")
                    camp.handle(res)
            if broken:
                suspects.extend(inflight.values())
                inflight.clear()
                _close(pool, force=True)
                if suspects:
                    print(f"  a worker process died; re-playing seed(s) {[j['s'] for j in suspects]} one game per "
                          f"process", file=camp.log, flush=True)
                for job in suspects:
                    _run_solo(camp, job)
                pool = _new_pool(workers)
        clean = True
    finally:
        _close(pool, force=not clean)
    return status


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("what to compare")
    g.add_argument("--tunable", help="registry name (danger.TURNS_HALF) or unique attribute (see scripts/ablate.py --list)")
    g.add_argument("--values", help="candidate values, comma separated (default: the registry candidates; flags on/off; "
                                    "resource vectors a/b/c/d/e separated by ';')")
    g.add_argument("--flag-off", action="store_true", help="flag tunables: the candidate is the flag switched off")
    g.add_argument("--base-spec", help="bot spec of both arms for --tunable (default "
                                       "search:depth=1,beam=4,expand=8,evaluator=heuristic; depth 2 for knobs only "
                                       "the depth-2 search reads)")
    g.add_argument("--cand-spec", help="candidate bot spec (with --def-spec, instead of --tunable)")
    g.add_argument("--def-spec", help="default bot spec")
    g.add_argument("--cand-set", action="append", metavar="NAME=VALUE", help="extra tunable override on the candidate arm")
    g.add_argument("--set", action="append", metavar="NAME=VALUE", help="tunable override on BOTH arms")
    g.add_argument("--adapter-opt", action="append", metavar="KEY=VALUE",
                   help="CatanbotPlayer keyword argument for both arms (JSON literal values)")
    g.add_argument("--cand-adapter-opt", action="append", metavar="KEY=VALUE", help="... for the candidate arm only")
    g.add_argument("--trades", default="off",
                   help="domestic trading as in bench_catanatron.py --trades (off | native | value | fair, 3.3 only; "
                        "see the adapter's TRADE_MODES): anything but off lets catanbot offer trades "
                        "(suppress_trades=False) and sets the opponents' answer rule (BenchOpponent.trade_rule); "
                        "--cand-adapter-opt trades=MODE changes it for the candidate arm only")
    g.add_argument("--info", choices=("full", "counted"), default="full",
                   help="information mode of both arms as in bench_catanatron.py --info (full = every card known, "
                        "the default; counted = Colonist public information); --cand-adapter-opt info=counted "
                        "switches the candidate arm only")
    g.add_argument("--info-samples", type=int, default=4, metavar="K",
                   help="--info counted: determinizations per searched decision (default 4)")
    g.add_argument("--discards-public", action="store_true",
                   help="--info counted: the cards of every discard are public")
    g.add_argument("--opponent-params", metavar="KEY=VAL,...",
                   help="constructor parameters of every opponent (bench_catanatron.py --opponent-params syntax)")
    g.add_argument("--opponent", help="bench_catanatron preset: value / alphabeta / sameturn (3.3), vf / ab / vp / "
                                      "weighted / random (3.2.1 and 3.3), or module:Class")
    g.add_argument("--vps-to-win", type=int, default=10)
    g.add_argument("--discard-limit", type=int, default=7)
    g = p.add_argument_group("how many / where")
    g.add_argument("--seeds", type=int, default=20, help="number of seeds (each played once per arm)")
    g.add_argument("--seed-base", type=int, default=0, help="first seed number (game seed = s + 1, seat = s %% 4)")
    g.add_argument("--workers", type=int, default=1, help="worker processes (0 = play in this process)")
    g.add_argument("--out", required=True, help="results JSONL (append-only; rerunning resumes)")
    g.add_argument("--exp-name", help="experiment name stored with the records")
    g.add_argument("--reuse", action="append", metavar="GLOB",
                   help="other JSONL files whose default-arm games (same arm key and code) are copied instead of replayed")
    g.add_argument("--stop-at-se", type=float, default=None,
                   help="sequential stop: end a candidate once its paired win-rate s.e. <= X and |delta| > 3 s.e.")
    g.add_argument("--stop-min-pairs", type=int, default=100, help="no sequential stop before this many pairs")
    g.add_argument("--max-minutes", type=float, default=None,
                   help="start no new seed after this many minutes (in-flight games finish; rerun to resume)")
    g.add_argument("--game-timeout", type=int, default=1800, help="seconds before a game is abandoned as an error")
    g.add_argument("--retry-errors", action="store_true", help="re-play games recorded as errors")
    g.add_argument("--quiet", action="store_true", help="no per-seed lines")
    g = p.add_argument_group("reporting")
    g.add_argument("--report", action="store_true", help="print the statistics of --out and exit (no games)")
    g.add_argument("--report-json", help="also write the statistics as JSON to this path")
    g.add_argument("--plan", action="store_true", help="print the runs, keys and missing games, play nothing")
    # test hooks (synthetic games through the real pool / JSONL / resume machinery)
    p.add_argument("--no-reexec", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fake-games", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--fake-crash-seeds", type=lambda t: [int(x) for x in t.split(",") if x], default=[],
                   help=argparse.SUPPRESS)
    p.add_argument("--fake-die-seeds", type=lambda t: [int(x) for x in t.split(",") if x], default=[],
                   help=argparse.SUPPRESS)
    p.add_argument("--fake-sleep", type=float, default=0.0, help=argparse.SUPPRESS)
    return p


def write_report_json(path: str, rows) -> None:
    data = [{"run": {k: v for k, v in run.items() if not k.startswith("_")}, "stats": st} for run, st in rows]
    with open(path, "w") as fh:
        json.dump(json_safe(data), fh, indent=1)


def _validate_real(runs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> None:
    """Fail fast (exit 2) on an opponent the running catanatron lacks, a bad adapter option or a search
    knob on a non-search spec - before any game is started."""
    import inspect
    from catanbot import tuning
    from catanbot.bench import catanatron_adapter as ad
    from catanbot.selfplay import make_bot
    bench = _bench()
    try:
        _opponent_maker(ctx)
    except SystemExit as ex:
        print(getattr(ex, "message", None) or str(ex), file=sys.stderr)
        raise SystemExit(2)
    params = inspect.signature(ad.CatanbotPlayer.__init__).parameters
    for r in runs:
        for arm in (r["cand"], r["def"]):
            for k in _catanbot_kwargs(arm["adapter"]):
                if k not in params:
                    raise SystemExit(f"error: CatanbotPlayer has no option {k!r} (options: {', '.join(list(params)[2:])})")
            mode = arm["adapter"].get("trades")
            if mode not in (None, "off"):
                modes = getattr(ad, "TRADE_MODES", ())
                if mode not in modes or not getattr(ad, "DOMESTIC_TRADING", False):
                    raise SystemExit(f"error: --trades {mode} needs catanatron 3.3 and an adapter with TRADE_MODES "
                                     f"(have {modes or 'none'} on catanatron {catanatron_version()})")
            tuning.check_overrides(make_bot(arm["spec"]), arm["overrides"])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report:
        rows = print_report(args.out)
        if args.report_json:
            write_report_json(args.report_json, rows)
        return 0
    if not args.opponent:
        print("error: --opponent is required", file=sys.stderr)
        return 2
    runs, need_python = build_runs(args)
    if not args.no_reexec and argv is None:
        ensure_env(need_python)
    elif os.environ.get("PYTHONHASHSEED") != PINNED_HASHSEED:
        print("WARNING: PYTHONHASHSEED is not pinned: games are not reproducible across processes", file=sys.stderr)
    fake = args.fake_games is not None
    ctx = make_context(args)
    if not fake:
        _validate_real(runs, ctx)
        # import everything a game touches now, so forked workers share one code version
        from catanbot.bench import catanatron_adapter  # noqa: F401
        from catanbot.agents import param_bot  # noqa: F401
        _bench()
    code = code_fingerprint()
    exp = args.exp_name or (f"{args.tunable}@{args.opponent}" if args.tunable else f"spec@{args.opponent}")
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    finalize_runs(runs, ctx, exp, (args.seed_base, args.seeds))
    keys = [r["def_key"] for r in runs]
    reuse = ReusePool(args.reuse, keys, code, args.out) if args.reuse else None
    print(f"ablate_catanatron: {len(runs)} candidate(s) vs default, opponent {args.opponent} x3, catanatron "
          f"{ctx['catanatron']}, python {ctx['python']}, {ctx['evaluator']}, PYTHONHASHSEED={ctx['hashseed']}, "
          f"code {code}; seeds {seeds[0] if seeds else '-'}..{seeds[-1] if seeds else '-'}, workers {args.workers}"
          + (" [SYNTHETIC GAMES]" if fake else ""), flush=True)
    for r in runs:
        print(f"  run {r['run_key']}: {r['cand']['label']} (arm {r['cand_key']}) vs {r['def']['label']} "
              f"(arm {r['def_key']}); spec {r['cand']['spec']}", flush=True)
    if args.plan:
        idx, _ = load_index(args.out)
        for r in runs:
            done = sum(1 for s in seeds if idx.pair(r, s)[0] is not None)
            print(f"  {r['cand']['label']}: {done}/{len(seeds)} seeds complete in {args.out}")
        return 0
    camp = Campaign(runs, args.out, ctx, code, seeds, exp, args.game_timeout, retry_errors=args.retry_errors,
                    stop_se=args.stop_at_se, stop_min_pairs=args.stop_min_pairs, reuse=reuse, quiet=args.quiet)
    todo = camp.pending_jobs()
    print(f"  {todo} seed(s) need games; results -> {args.out}", flush=True)
    deadline = time.time() + 60.0 * args.max_minutes if args.max_minutes else None

    def on_term(signum, frame):
        raise Stop(f"signal {signum}")

    old = signal.signal(signal.SIGTERM, on_term)
    t0 = time.time()
    try:
        status = execute(camp, args.workers, deadline)
    except (Stop, KeyboardInterrupt) as ex:
        status = f"interrupted ({ex or 'SIGINT'})"
    finally:
        signal.signal(signal.SIGTERM, old)
    wall = time.time() - t0
    rate = 3600.0 * camp.played / wall if wall > 0 else 0.0
    left = camp.pending_jobs()
    print(f"\n{status}: played {camp.played} game(s) ({camp.errors} error record(s), {camp.reused} default game(s) "
          f"reused) in {wall:.0f} s = {rate:.0f} games/hour at {max(args.workers, 1)} worker(s); "
          f"{left} seed(s) still need games" + (" - rerun the same command to resume" if left else ""), flush=True)
    print()
    rows = print_report(args.out)
    if args.report_json:
        write_report_json(args.report_json, rows)
    return 0 if status in ("done", "deadline") else 130


if __name__ == "__main__":
    raise SystemExit(main())
