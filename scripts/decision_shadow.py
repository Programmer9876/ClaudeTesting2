#!/usr/bin/env python3
"""Zero-game decision shadow: how often (and where) does a candidate change the default bot's decisions?

    # 40 games of the default bot; every robber decision and knight decision re-searched per candidate
    python3 scripts/decision_shadow.py --games 40 --seed 7300 --phases robber,knight \\
        --cand block_need0:danger.BLOCK_NEED=0 --cand dm_off:danger.danger_multiplier=off --json shadow.json
    # positions from the archived proof games instead (our seat; the interpreter must match the log's engine)
    /home/user/venv_cat33/bin/python scripts/decision_shadow.py --source proof --logs 'proof/T1/logs/*.gz' \\
        --max-games 20 --cand expand4:expand=4

Generalised from ``scripts/winpaths_shadow.py``.  The games are played by the UNCHANGED default bot (``--spec``);
at every covered decision with more than one legal action a :class:`ShadowBot` re-searches the root once with the
default configuration and once per candidate, each from ``random.Random(k)`` with the same ``k``, using the bot's
own evaluator, opponent model, belief and politics, and records the first actions and the process times.  The
shadow never touches the game's random stream, so the game is the one the default bot plays anyway
(tests/test_decision_shadow.py checks it).

Candidates (``--cand LABEL:NAME=VALUE,...``): a NAME with a dot is a registry tunable (``catanbot/tuning.py``:
weights and flags are installed with ``tuning.apply`` around the candidate search only, search knobs replace the
``SearchConfig`` field); a NAME without a dot is a bot-spec key (``expand=4``, ``paths=1``) whose ``SearchConfig``
fields are replaced.  An A/A candidate (``aa``: nothing changed) is always added: it must change 0 decisions
(``aa_ok``), else the shadow itself is nondeterministic and its numbers mean nothing.  A tunable read by the
static evaluator (``needs_python_evaluator``) re-executes the script with ``CATANBOT_NO_ACCEL=1``; with
``--py-shadow`` it does not: only the shadow searches (the default re-search, the A/A and every candidate) run on
the Python evaluator, while the game's own decisions stay on C++ (a proof replay ignores them anyway), which makes a
Python-evaluator screen of hundreds of proof games affordable.

Phase classes (``--phases``, default all): setup, roll, main, robber (the robber move after a 7), discard, trade
(answering / selecting), knight (any decision where playing a knight is legal - the knight's hex and victim are part
of that action), buy_dev (buying a development card is legal) and settle (a setup settlement placement, or a
decision with at least two legal settlement spots: the ports screen's "multi-spot settlement decisions").  A
decision can be in several classes (roll + knight, main + buy_dev, setup + settle).  Settle rows also record the
port under each arm's chosen settlement (``ports``: the port name, "" for a settlement off the ports, None for
another first action), summarised per candidate as ``settle_ports``.  ``--main-every N`` shadows only every N-th main-phase decision that is in no other
covered class (main decisions are ~80% of all; the robber shadow samples 1 in 5).

Output: per candidate the changed share overall and per class, the changed kinds (default kind -> candidate
kind), the mean and p95 ms of both searches and their ratio.  ``scripts/run_queue.py`` reads the JSON through
:func:`gate_share` to skip a prior-only row as NOOP(shadow) at 0 games when it changes fewer than 2% of the
decisions in its classes (the robber rows: robber + knight).  One process, no workers.

Robber step (docs/PRIORITY_PLAN.md step 5): ``--robber-gates r1a=persistence,kick=persistence,r1b=insurance,
r1c=duration`` tags every robber / knight / roll / main row with the robber classifier (``rob``: the default's and
each candidate's robber move - would-kick knight holders on the new hex, kicking within one roll or at all, a top-VP
opponent blocked, P(the stolen card is one we need) - whether we hold a playable knight, our blocked value and our
P_hit / D from robber_eval's insurance model), keeps each candidate search's robber_eval counters (``rstats``: the
insurance centring sums) and evaluates the design's gates G1-G5 per candidate (:func:`robber_gates`; written as
``robber_gates``).  ``--gate-report SHADOW_JSON --gate-label LABEL --json OUT`` re-evaluates one candidate's gates
from a finished shadow (no games): the queue's robber gate rows.  All of it is observation only.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PY_SHADOW = False     # --py-shadow: the shadow searches run on the Python evaluator (C++ switched off around them)
ROBBER_TAGS = False   # --robber-gates: robber / would-kick-holder tags per row and the candidates' robber_eval stats
CLASSES =("setup", "roll", "main", "robber", "discard", "trade", "knight", "buy_dev", "settle")


def _p95(xs: Sequence[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(0.95 * len(s)) - 1))] if s else float("nan")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Candidate:
    label: str
    overrides: Dict[str, Any]               # registry tunables (weights / flags / search knobs)
    spec_keys: Dict[str, str]               # bot-spec keys -> SearchConfig fields

    def needs_python(self) -> bool:
        from catanbot import tuning
        return any(tuning.find(n).needs_python_evaluator for n in self.overrides)


def parse_candidate(text: str) -> Candidate:
    """``LABEL:NAME=VALUE,NAME=VALUE`` (a dotted NAME is a registry tunable, otherwise a bot-spec key)."""
    from catanbot import tuning
    label, sep, rest = text.partition(":")
    if not sep or not label.strip():
        raise SystemExit(f"error: --cand wants LABEL:NAME=VALUE[,...], got {text!r}")
    overrides: Dict[str, Any] = {}
    keys: Dict[str, str] = {}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"error: --cand {label}: expected NAME=VALUE, got {part!r}")
        name, value = (x.strip() for x in part.split("=", 1))
        if "." in name:
            try:
                t = tuning.find(name)
            except KeyError as ex:
                raise SystemExit(f"error: --cand {label}: {ex}")
            overrides[t.name] = t.parse(value)
        else:
            keys[name] = value
    return Candidate(label.strip(), overrides, keys)


def classes_of(state, legal) -> Tuple[str, ...]:
    from catanbot import actions as A
    from catanbot import state as S
    ph = state.phase
    out = []
    if ph in (S.PHASE_SETUP_SETTLEMENT, S.PHASE_SETUP_ROAD):
        out.append("setup")
    elif ph == S.PHASE_ROLL:
        out.append("roll")
    elif ph == S.PHASE_MAIN:
        out.append("main")
    elif ph == S.PHASE_ROBBER:
        out.append("robber")
    elif ph == S.PHASE_DISCARD:
        out.append("discard")
    elif ph in (S.PHASE_TRADE_RESPONSE, S.PHASE_TRADE_SELECT):
        out.append("trade")
    if any(a[0] == A.PLAY_KNIGHT for a in legal):
        out.append("knight")
    if any(a[0] == A.BUY_DEV for a in legal):
        out.append("buy_dev")
    if ph == S.PHASE_SETUP_SETTLEMENT or sum(1 for a in legal if a[0] == A.BUILD_SETTLEMENT) >= 2:
        out.append("settle")
    return tuple(out)


def port_tag(state, action) -> Optional[str]:
    """The port under a settlement action (its name, "" off the ports); None for any other action."""
    from catanbot import actions as A
    from catanbot import board as B
    if not action or action[0] not in (A.SETUP_SETTLEMENT, A.BUILD_SETTLEMENT):
        return None
    t = state.ports.get(action[1])
    return "" if t is None else B.PORT_NAMES[t]


# ---------------------------------------------------------------------------
# The shadow wrapper
# ---------------------------------------------------------------------------
class ShadowBot:
    """Wraps the default ``SearchBot``: before each covered decision it re-searches the root with the default
    configuration and every candidate (same ``random.Random(k)``), then lets the inner bot decide as always."""

    def __init__(self, inner, candidates: Sequence[Candidate], phases: Sequence[str], seed: int, spec: str,
                 rows: List[Dict[str, Any]], tag: str = "", main_every: int = 1):
        self.inner = inner
        self.name = getattr(inner, "name", "search")
        self.candidates = list(candidates)
        self.phases = set(phases)
        self.seed = seed
        self.spec = spec
        self.rows = rows
        self.tag = tag
        self.k = 0
        self.main_every = max(1, int(main_every))
        self._main_seen = 0
        self._cfg_cache: Dict[str, Any] = {}

    # Bot protocol --------------------------------------------------------------------------
    def reset(self) -> None:
        self.inner.reset()

    def observe(self, state, action, player) -> None:
        self.inner.observe(state, action, player)

    def explain(self, state):
        return self.inner.explain(state)

    def __getattr__(self, item):
        if item in ("inner", "candidates", "phases", "rows"):
            raise AttributeError(item)
        return getattr(self.inner, item)

    def decide(self, state, legal, rng):
        if len(legal) > 1:
            cls = classes_of(state, legal)
            hit = self.phases.intersection(cls)
            if hit and self.main_every > 1 and hit <= {"main", "buy_dev"}:
                self._main_seen += 1
                if (self._main_seen - 1) % self.main_every:
                    hit = set()
            if hit:
                self._shadow(state, legal, cls)
        return self.inner.decide(state, legal, rng)

    # the shadow ----------------------------------------------------------------------------
    def _config(self, cand: Candidate):
        from catanbot import tuning
        from catanbot.selfplay import make_bot
        base = self.inner.config
        repl: Dict[str, Any] = {}
        if cand.spec_keys:
            key = json.dumps(cand.spec_keys, sort_keys=True)
            if key not in self._cfg_cache:
                spec = self.spec + ("," if ":" in self.spec else ":") + ",".join(f"{k}={v}" for k, v in
                                                                                 cand.spec_keys.items())
                ref = make_bot(self.spec).config
                cc = make_bot(spec).config
                self._cfg_cache[key] = {f.name: getattr(cc, f.name) for f in dataclasses.fields(cc)
                                        if getattr(cc, f.name) != getattr(ref, f.name)}
            repl.update(self._cfg_cache[key])
        for name, value in cand.overrides.items():
            t = tuning.find(name)
            if t.kind == "search":
                repl[t.attr] = value
        return dataclasses.replace(base, **repl) if repl else base

    def _search(self, state, me, cfg, k, overrides):
        from catanbot import accel, tuning
        from catanbot.search import Searcher
        token = tuning.apply(overrides) if overrides else None
        cpp = accel.AVAILABLE
        if PY_SHADOW:
            accel.AVAILABLE = False      # every evaluator site checks the flag at call time
        try:
            sr = Searcher(self.inner.evaluator, cfg, self.inner.model, self.inner.belief, self.inner.politics)
            t0 = time.process_time()
            res = sr.search(state, me, random.Random(k))
            dt = time.process_time() - t0
            self._last_hub = sr._corr        # the search's correction hub (robber_eval stats; --robber-gates)
        finally:
            accel.AVAILABLE = cpp
            if token is not None:
                tuning.restore(token)
        return (list(res[0].action) if res else None), dt

    def _shadow(self, state, legal, cls) -> None:
        from catanbot import engine as E
        inner = self.inner
        if inner.model is not None:
            inner.model.attach(state)
        inner._ensure_politics(state)
        me = E.acting_player(state)
        k = self.seed * 100003 + self.k
        self.k += 1
        a0, t0 = self._search(state, me, inner.config, k, None)
        row = {"g": self.tag, "k": k, "turn": getattr(state, "turn", None), "player": me, "cls": list(cls),
               "def": a0, "ms_def": round(1000.0 * t0, 3), "cand": {}}
        for c in self.candidates:
            a1, t1 = self._search(state, me, self._config(c), k, {n: v for n, v in c.overrides.items()})
            row["cand"][c.label] = {"a": a1, "ms": round(1000.0 * t1, 3)}
            if ROBBER_TAGS:
                rst = robber_stats(getattr(self, "_last_hub", None))
                if rst:
                    row["cand"][c.label]["rstats"] = rst
        if ROBBER_TAGS and set(cls) & {"robber", "knight", "roll", "main"}:
            row["rob"] = robber_tags(state, me, a0, row["cand"], inner.model)
        if "settle" in cls:
            row["ports"] = {"def": port_tag(state, a0)}
            row["ports"].update({lab: port_tag(state, got["a"]) for lab, got in row["cand"].items()})
        self.rows.append(row)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def run_selfplay(spec: str, candidates, phases, games: int, seed: int, max_turns: int, log=sys.stdout,
                 main_every: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from catanbot.selfplay import make_bot, play_game
    rows: List[Dict[str, Any]] = []
    results = []
    for g in range(games):
        gseed = seed + g
        bots = [ShadowBot(make_bot(spec), candidates, phases, gseed * 10 + i, spec, rows, tag=f"sp{gseed}",
                          main_every=main_every) for i in range(4)]
        res = play_game(bots, rng=random.Random(gseed), seed=gseed, max_turns=max_turns)
        results.append({"seed": gseed, "winner": res.winner, "vps": list(res.vps), "turns": res.turns,
                        "actions": res.actions})
        print(f"game {g + 1}/{games} seed {gseed}: {res.turns} turns, winner {res.winner}, {len(rows)} shadow "
              f"decisions so far", file=log, flush=True)
    return rows, results


def run_proof(spec: str, candidates, phases, paths: Sequence[str], max_games: Optional[int], log=sys.stdout,
              main_every: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Positions from archived proof games (catanbot-actionlog/1): our seat's CatanbotPlayer, whose bot is a
    ShadowBot around the default bot, is asked at each of its logged turns; its answer is ignored and the logged
    action is replayed, so every position is the proof game's own."""
    from catanatron.models.player import Color
    from catanbot.bench import catanatron_adapter as ad
    from catanbot.selfplay import make_bot
    api = "3.3" if ad.API_33 else "3.2"
    rows: List[Dict[str, Any]] = []
    results = []
    n = 0
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc.get("api") != api:
                    raise SystemExit(f"error: {p} was written by catanatron {doc.get('catanatron')} ({doc.get('api')} "
                                     f"API); run the shadow with the interpreter of that engine")
                seats = json.loads(doc["our_seats"]) if isinstance(doc["our_seats"], str) else doc["our_seats"]
                players = doc["players"]
                if isinstance(players, str):
                    import ast
                    players = ast.literal_eval(players)
                seat = int(seats[0])
                color = Color(doc["colors"][seat])
                bot_seed = int(players[seat].get("bot_seed", 0))
                game = ad.rebuild_game(doc)
                shadow = ShadowBot(make_bot(spec), candidates, phases, bot_seed, spec, rows, tag=f"pf{doc.get('game')}",
                                   main_every=main_every)
                me = ad.CatanbotPlayer(color, spec=spec, bot=shadow, seed=bot_seed, suppress_trades=True)
                errors = 0
                for item in doc["actions"]:
                    if item[0] == color.value:
                        try:
                            me.decide(game, ad.playable_actions_of(game))
                        except Exception:  # noqa: BLE001  (the logged action is replayed anyway)
                            errors += 1
                    ad.replay_log_action(game, item, check=False)
                results.append({"game": doc.get("game"), "seat": seat, "errors": errors,
                                "adapter_errors": int(me.stats.get("errors", 0))})
                n += 1
                print(f"proof game {doc.get('game')} ({os.path.basename(p)}): {len(rows)} shadow decisions so far",
                      file=log, flush=True)
                if max_games is not None and n >= max_games:
                    return rows, results
    return rows, results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def summarize(rows: Sequence[Dict[str, Any]], candidates: Sequence[Candidate]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in candidates:
        n = changed = 0
        by = {k: {"n": 0, "changed": 0} for k in CLASSES}
        kinds: Counter = Counter()
        ms_d, ms_c = [], []
        for r in rows:
            got = r["cand"].get(c.label)
            if got is None:
                continue
            n += 1
            ch = got["a"] != r["def"]
            changed += ch
            for k in r["cls"]:
                by[k]["n"] += 1
                by[k]["changed"] += int(ch)
            if ch:
                kinds[f"{r['def'][0] if r['def'] else '-'} -> {got['a'][0] if got['a'] else '-'}"] += 1
            ms_d.append(r["ms_def"])
            ms_c.append(got["ms"])
        md = sum(ms_d) / len(ms_d) if ms_d else float("nan")
        mc = sum(ms_c) / len(ms_c) if ms_c else float("nan")
        sp = settle_ports(rows, c.label)
        out[c.label] = {"overrides": {k: v for k, v in c.overrides.items()}, "spec_keys": dict(c.spec_keys),
                        "n": n, "changed": changed, "share": changed / n if n else None,
                        "by_class": {k: dict(v, share=(v["changed"] / v["n"] if v["n"] else None))
                                     for k, v in by.items() if v["n"]},
                        "kinds": dict(kinds.most_common()), "ms_def_mean": md, "ms_mean": mc,
                        "ms_ratio_mean": mc / md if md and md == md else None,
                        "ms_ratio_p95": (_p95(ms_c) / _p95(ms_d)) if ms_d and _p95(ms_d) > 0 else None}
        if sp is not None:
            out[c.label]["settle_ports"] = sp
    return out


def settle_ports(rows: Sequence[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    """Port picks over the settle rows (those with ``ports``), default vs candidate ``label``: settlement choices
    (``settled``), on a port (``port``), on a 3:1 port (``generic``), split into setup and main."""
    out = None
    for r in rows:
        tags = r.get("ports")
        if tags is None or label not in tags:
            continue
        if out is None:
            out = {ph: {arm: {"n": 0, "settled": 0, "port": 0, "generic": 0} for arm in ("def", "cand")}
                   for ph in ("setup", "main")}
        ph = "setup" if "setup" in r["cls"] else "main"
        for arm, tag in (("def", tags.get("def")), ("cand", tags.get(label))):
            d = out[ph][arm]
            d["n"] += 1
            if tag is not None:
                d["settled"] += 1
                d["port"] += int(tag != "")
                d["generic"] += int(tag == "3:1")
    return out


def gate_share(result: Dict[str, Any], label: str, classes: Sequence[str]) -> Optional[Tuple[float, int]]:
    """Changed share of candidate ``label`` over the decisions in any of ``classes`` (a decision counted once),
    from a shadow JSON (``rows`` kept) or, without rows, from the per-class counts (an upper bound: decisions
    in two classes are counted twice).  None when the candidate is not in the result."""
    classes = set(classes)
    rows = result.get("rows")
    if rows:
        n = ch = 0
        for r in rows:
            if label not in r["cand"] or not classes.intersection(r["cls"]):
                continue
            n += 1
            ch += int(r["cand"][label]["a"] != r["def"])
        return (ch / n if n else 0.0), n
    cand = (result.get("candidates") or {}).get(label)
    if cand is None:
        return None
    n = sum(v["n"] for k, v in cand["by_class"].items() if k in classes)
    ch = sum(v["changed"] for k, v in cand["by_class"].items() if k in classes)
    return (ch / n if n else 0.0), n


# ---------------------------------------------------------------------------
# Robber classifier and gates (docs/PRIORITY_PLAN.md step 5; --robber-gates)
# ---------------------------------------------------------------------------
# Gate sets of the robber design (docs/designs/priority_areas_2026-09-26.json designs[3]):
#   persistence (R1a; also the knight_kick alternative): G1 >= 30% of the default's robber / knight moves whose hex
#     blocks a would-kick knight holder kicking within one roll (t <= 1) change; G2 <= 3% of the other robber /
#     knight decisions change; G3 <= 1% of main-phase first actions (no knight legal) change; G4 ms ratio <= 1.2
#     mean and <= 1.5 p95.  Failing G1 or G4 unlocks the knight_kick fallback (``fallback_trigger``).
#   insurance (R1b): G1 >= 3% of knight holders' roll / main first actions change; G2 in >= 60% of those changes
#     the candidate keeps the knight while P_hit is above its median; G3 <= 1% of non-holders' roll / main first
#     actions change; G4 |mean C_ins| over insured leaves <= 0.05 points (centring; ``ins_offset_cal`` = the mean
#     D x P_hit that INS_OFFSET should be); G5 ms ratio <= 1.3 mean.
#   duration (R1c): G1 >= 2% of robber / knight decisions change (plus readouts).
GATE_SETS = ("persistence", "insurance", "duration")
ROBBER_MOVES = ("move_robber", "play_knight")


def parse_gate_map(text: Optional[str]) -> Dict[str, str]:
    """``LABEL=SET,...`` (SET in :data:`GATE_SETS`) -> ``{label: set}``."""
    out: Dict[str, str] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        label, sep, gset = (x.strip() for x in part.partition("="))
        if not sep or gset not in GATE_SETS:
            raise SystemExit(f"error: --robber-gates wants LABEL=SET with SET in {GATE_SETS}, got {part!r}")
        out[label] = gset
    return out


def robber_stats(hub) -> Optional[Dict[str, float]]:
    """The robber_eval provider's counters of one search (insurance centring / calibration sums), or None."""
    if hub is None:
        return None
    for p in getattr(hub, "providers", []):
        st = getattr(p, "stats", None)
        if st is not None and "ins_sum_dp" in st:
            return {k: st[k] for k in ("evals", "nonzero", "kick_sets", "ins_n", "ins_sum_c", "ins_sum_dp")}
    return None


def robber_move_info(state, me: int, action) -> Optional[Dict[str, Any]]:
    """For a robber move (MOVE_ROBBER / PLAY_KNIGHT): its would-kick set on the new hex, t <= 1 and any, whether a
    top-VP opponent is blocked, the opponents' blocked value and P(the stolen card is one we need)."""
    if not action or action[0] not in ROBBER_MOVES or len(action) < 3:
        return None
    from catanbot.robber import our_need_indicator
    from catanbot.robber_metrics import RobberCounters
    info = RobberCounters.classify_move(state, tuple(action), me)
    victim = action[2]
    p_need = None
    if victim is not None and victim >= 0:
        vp = state.players[victim]
        tot = vp.total_resources
        if vp.hand_known and tot > 0:
            need = our_need_indicator(state, me)
            p_need = sum(vp.resources[r] * need[r] for r in range(5)) / tot
    return {"kick_t1": any(t <= 1 for t, _k, _q in info["kick"]), "kick_any": bool(info["kick"]),
            "on_leader": bool(info["on_leader"]), "opp_pv": round(float(info["opp_pv"]), 4), "p_need": p_need,
            "knight": action[0] == "play_knight"}


def robber_tags(state, me: int, a0, cand: Dict[str, Dict[str, Any]], model=None) -> Dict[str, Any]:
    """The robber / would-kick-holder classification of one shadowed decision (observation only: no RNG, the
    state and the model are not changed): the default's and each candidate's robber move, whether we hold a
    playable knight (``holder``, roll / main phase), our blocked value, and our P_hit / D from R1b's model."""
    from catanbot import board as B
    from catanbot import state as S
    from catanbot.robber import production_blocked
    from catanbot.robber_eval import RobberContext
    p = state.players[me]
    holder = state.phase in (S.PHASE_ROLL, S.PHASE_MAIN) and p.dev_cards[B.DEV_KNIGHT] >= 1
    out: Dict[str, Any] = {"holder": bool(holder), "blocked": round(production_blocked(state, me), 4),
                           "def": robber_move_info(state, me, a0),
                           "cand": {lab: robber_move_info(state, me, got.get("a")) for lab, got in cand.items()}}
    if holder:
        ctx = RobberContext(state, me, model=model)
        p_hit, d = ctx.exposure(state, me)
        out["p_hit"] = round(p_hit, 6)
        out["d"] = round(d, 4)
    return out


def _share(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(rows)
    ch = sum(1 for r in rows if r["cand"][label]["a"] != r["def"])
    return {"n": n, "changed": ch, "share": (ch / n) if n else None}


def _gate(d: Dict[str, Any], ok: Optional[bool], rule: str) -> Dict[str, Any]:
    d = dict(d)
    d["rule"] = rule
    d["pass"] = bool(ok) if ok is not None else False
    return d


def robber_gates(rows: Sequence[Dict[str, Any]], label: str, gset: str) -> Dict[str, Any]:
    """Gate set ``gset`` (:data:`GATE_SETS`) for candidate ``label`` over the rows tagged by ``robber_tags``."""
    rows = [r for r in rows if label in r["cand"] and "rob" in r]
    rk = [r for r in rows if set(r["cls"]) & {"robber", "knight"}]
    ms_d = [r["ms_def"] for r in rows]
    ms_c = [r["cand"][label]["ms"] for r in rows]
    md = sum(ms_d) / len(ms_d) if ms_d else float("nan")
    mc = sum(ms_c) / len(ms_c) if ms_c else float("nan")
    ratio = mc / md if ms_d and md > 0 else None
    ratio95 = (_p95(ms_c) / _p95(ms_d)) if ms_d and _p95(ms_d) > 0 else None
    out: Dict[str, Any] = {"set": gset, "label": label, "rows": len(rows), "ms_ratio_mean": ratio,
                           "ms_ratio_p95": ratio95}
    g: Dict[str, Any] = {}
    if gset == "persistence":
        g1 = [r for r in rk if (r["rob"].get("def") or {}).get("kick_t1")]
        g1_ids = {id(r) for r in g1}
        g2 = [r for r in rk if id(r) not in g1_ids]
        g3 = [r for r in rows if "main" in r["cls"] and "knight" not in r["cls"]]
        s1, s2, s3 = _share(g1, label), _share(g2, label), _share(g3, label)
        g["G1"] = _gate(s1, s1["n"] > 0 and s1["share"] >= 0.30, ">= 30% of default moves onto a t<=1 would-kick "
                                                                  "holder change")
        g["G2"] = _gate(s2, s2["n"] == 0 or s2["share"] <= 0.03, "<= 3% of other robber / knight decisions change")
        g["G3"] = _gate(s3, s3["n"] == 0 or s3["share"] <= 0.01, "<= 1% of main-phase first actions change")
        g["G4"] = _gate({"mean": ratio, "p95": ratio95}, ratio is not None and ratio <= 1.2 and
                        (ratio95 is None or ratio95 <= 1.5), "ms ratio <= 1.2 mean, <= 1.5 p95")
        out["fallback_trigger"] = not (g["G1"]["pass"] and g["G4"]["pass"])
        cm = [r for r in g1 if r["cand"][label]["a"] != r["def"]]
        later = [r for r in g2 if (r["rob"].get("def") or {}).get("kick_any")]
        rest = [r for r in g2 if not (r["rob"].get("def") or {}).get("kick_any")]
        out["readout"] = {"g1_changed_to_robber_move": sum(1 for r in cm if (r["rob"]["cand"].get(label) or {})),
                          "g1_changed_still_t1": sum(1 for r in cm if (r["rob"]["cand"].get(label) or {})
                                                     .get("kick_t1")),
                          # G2's set split: the default's hex blocks a would-kick holder kicking after 2+ rolls
                          # (a smaller restore by design) / no would-kick holder (incl. no robber move at all)
                          "g2_kick_later": _share(later, label), "g2_no_kick_holder": _share(rest, label)}
    elif gset == "insurance":
        hold = [r for r in rows if r["rob"].get("holder")]
        non = [r for r in rows if not r["rob"].get("holder") and set(r["cls"]) & {"roll", "main"}]
        s1, s3 = _share(hold, label), _share(non, label)
        ph = sorted(r["rob"]["p_hit"] for r in hold if r["rob"].get("p_hit") is not None)
        med = ph[len(ph) // 2] if ph else None
        chg = [r for r in hold if r["cand"][label]["a"] != r["def"]]
        kept = [r for r in chg if (r["cand"][label]["a"] or [None])[0] != "play_knight"
                and med is not None and r["rob"].get("p_hit", -1.0) > med]
        s2 = {"n": len(chg), "kept_exposed": len(kept), "share": (len(kept) / len(chg)) if chg else None,
              "p_hit_median": med}
        tot = {"ins_n": 0, "ins_sum_c": 0.0, "ins_sum_dp": 0.0}
        for r in rows:
            rst = r["cand"][label].get("rstats") or {}
            for k in tot:
                tot[k] += rst.get(k, 0)
        mean_c = tot["ins_sum_c"] / tot["ins_n"] if tot["ins_n"] else None
        g["G1"] = _gate(s1, s1["n"] > 0 and s1["share"] >= 0.03, ">= 3% of knight holders' roll / main first "
                                                                  "actions change")
        g["G2"] = _gate(s2, bool(chg) and s2["share"] >= 0.60, ">= 60% of the changed holder decisions keep the "
                                                                "knight while P_hit is above its median")
        g["G3"] = _gate(s3, s3["n"] == 0 or s3["share"] <= 0.01, "<= 1% of non-holders' roll / main first actions "
                                                                  "change")
        g["G4"] = _gate({"insured_leaves": tot["ins_n"], "mean_c_ins": mean_c},
                        mean_c is not None and abs(mean_c) <= 0.05, "|mean C_ins| over insured leaves <= 0.05 points")
        g["G5"] = _gate({"mean": ratio}, ratio is not None and ratio <= 1.3, "ms ratio <= 1.3 mean")
        out["ins_offset_cal"] = (tot["ins_sum_dp"] / tot["ins_n"]) if tot["ins_n"] else None
    elif gset == "duration":
        s1 = _share(rk, label)
        g["G1"] = _gate(s1, s1["n"] > 0 and s1["share"] >= 0.02, ">= 2% of robber / knight decisions change")
        chg = [r for r in rk if r["cand"][label]["a"] != r["def"]]
        both = [r for r in chg if r["rob"].get("def") and (r["rob"]["cand"].get(label))]
        dn = [r["rob"]["cand"][label]["p_need"] - r["rob"]["def"]["p_need"] for r in both
              if r["rob"]["def"]["p_need"] is not None and r["rob"]["cand"][label]["p_need"] is not None]
        off = {"def": 0, "cand": 0}
        for r in rk:
            if r["rob"].get("blocked", 0.0) > 0.0:
                continue
            off["def"] += int(bool((r["rob"].get("def") or {}).get("knight")))
            off["cand"] += int(bool((r["rob"]["cand"].get(label) or {}).get("knight")))
        out["readout"] = {
            "changed_robber_moves": len(both),
            "d_p_need_mean": (sum(dn) / len(dn)) if dn else None,
            "d_leader_hit": sum(int(r["rob"]["cand"][label]["on_leader"]) - int(r["rob"]["def"]["on_leader"])
                                for r in both),
            "d_opp_pv_mean": (sum(r["rob"]["cand"][label]["opp_pv"] - r["rob"]["def"]["opp_pv"] for r in both)
                              / len(both)) if both else None,
            "offensive_knights": off}
    else:
        raise ValueError(f"unknown gate set {gset!r}")
    out["gates"] = g
    out["pass"] = all(x["pass"] for x in g.values())
    return out


def gate_report(path: str, label: str, out_path: Optional[str] = None) -> Dict[str, Any]:
    """A queue gate row's check: recompute ``label``'s robber gates from a shadow JSON (``--keep-rows``) or copy them
    from its ``robber_gates``; writes ``{label, set, pass, fallback_trigger, gates, ...}`` to ``out_path``."""
    with open(path) as fh:
        res = json.load(fh)
    gmap = res.get("robber_gate_map") or {}
    if label not in gmap:
        raise SystemExit(f"error: {path} has no robber gate set for candidate {label!r}")
    if res.get("aa_ok") is False:
        raise SystemExit(f"error: {path}: the A/A candidate changed decisions (the shadow is nondeterministic)")
    rows = res.get("rows")
    rep = robber_gates(rows, label, gmap[label]) if rows else (res.get("robber_gates") or {}).get(label)
    if rep is None:
        raise SystemExit(f"error: {path}: no rows and no robber_gates for {label!r}")
    rep = dict(rep, source=path)
    rep.setdefault("fallback_trigger", None)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rep, fh)
        os.replace(tmp, out_path)
    return rep


def print_robber_gates(gates: Dict[str, Dict[str, Any]], log=sys.stdout) -> None:
    for label, rep in gates.items():
        verdict = "PASS" if rep["pass"] else "FAIL"
        extra = ""
        if rep.get("fallback_trigger") is not None:
            extra += f"; knight_kick fallback trigger {rep['fallback_trigger']}"
        if rep.get("ins_offset_cal") is not None:
            extra += f"; INS_OFFSET calibration {rep['ins_offset_cal']:.3f}"
        print(f"  robber gates {label} ({rep['set']}): {verdict}{extra}", file=log)
        for gname, gd in rep["gates"].items():
            vals = ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in gd.items()
                             if k not in ("rule", "pass"))
            print(f"      {gname} {'ok  ' if gd['pass'] else 'FAIL'} {gd['rule']}: {vals}", file=log)
        if rep.get("readout"):
            print(f"      readout: {rep['readout']}", file=log)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("selfplay", "proof"), default="selfplay")
    ap.add_argument("--games", type=int, default=4, help="selfplay: games of the default bot (4 seats)")
    ap.add_argument("--seed", type=int, default=7300)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--logs", help="proof: glob of catanbot-actionlog/1 files (proof/T1/logs/*.gz)")
    ap.add_argument("--max-games", type=int, default=None, help="proof: at most this many games")
    ap.add_argument("--spec", default=None, help="the default bot spec (default: the shipped search bot)")
    ap.add_argument("--cand", action="append", default=[], metavar="LABEL:NAME=VALUE,...")
    ap.add_argument("--phases", default=",".join(CLASSES), help=f"classes to shadow ({','.join(CLASSES)})")
    ap.add_argument("--main-every", type=int, default=1, help="shadow every N-th main-phase decision only")
    ap.add_argument("--no-aa", action="store_true", help="do not add the A/A identity candidate")
    ap.add_argument("--keep-rows", action="store_true", help="keep the per-decision rows in --json")
    ap.add_argument("--py-shadow", action="store_true",
                    help="run the shadow searches (default, A/A and candidates) on the Python evaluator without "
                         "re-executing the whole script under CATANBOT_NO_ACCEL=1 (the games stay on C++)")
    ap.add_argument("--json", help="write the result here")
    ap.add_argument("--robber-gates", metavar="LABEL=SET,...",
                    help="tag robber / would-kick-holder decisions and evaluate the robber design's gates per "
                         f"candidate (SET in {', '.join(GATE_SETS)}); written as robber_gates")
    ap.add_argument("--gate-report", metavar="SHADOW_JSON",
                    help="no games: evaluate --gate-label's robber gates from a shadow JSON and write --json")
    ap.add_argument("--gate-label", help="with --gate-report: the candidate label")
    args = ap.parse_args(argv)
    if args.gate_report:
        if not args.gate_label:
            raise SystemExit("error: --gate-report needs --gate-label")
        rep = gate_report(args.gate_report, args.gate_label, args.json)
        print_robber_gates({args.gate_label: rep})
        return 0
    from catanbot import tuning
    spec = args.spec or tuning.DEFAULT_SEARCH_SPEC
    cands = [parse_candidate(t) for t in args.cand]
    if not args.no_aa and not any(c.label == "aa" for c in cands):
        cands.insert(0, Candidate("aa", {}, {}))
    if len({c.label for c in cands}) != len(cands):
        raise SystemExit("error: candidate labels must be unique")
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    bad = set(phases) - set(CLASSES)
    if bad:
        raise SystemExit(f"error: unknown phase class(es) {sorted(bad)}")
    global PY_SHADOW
    PY_SHADOW = bool(args.py_shadow)
    global ROBBER_TAGS
    gmap = parse_gate_map(args.robber_gates)
    ROBBER_TAGS = bool(gmap)
    unknown = sorted(set(gmap) - {c.label for c in cands})
    if unknown:
        raise SystemExit(f"error: --robber-gates names unknown candidate(s) {unknown}")
    if any(c.needs_python() for c in cands) and not PY_SHADOW:
        from catanbot import accel
        if accel.AVAILABLE and not accel.disabled_by_env() and argv is None:
            print("re-executing with CATANBOT_NO_ACCEL=1 (a candidate is read by the Python static evaluator)",
                  flush=True)
            os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
                      dict(os.environ, CATANBOT_NO_ACCEL="1"))
    t0 = time.time()
    cpu0 = time.process_time()
    if args.source == "selfplay":
        rows, games = run_selfplay(spec, cands, phases, args.games, args.seed, args.max_turns,
                                   main_every=args.main_every)
    else:
        if not args.logs:
            raise SystemExit("error: --source proof needs --logs")
        paths = sorted(glob.glob(args.logs))
        if not paths:
            raise SystemExit(f"error: no files match {args.logs}")
        rows, games = run_proof(spec, cands, phases, paths, args.max_games, main_every=args.main_every)
    summ = summarize(rows, cands)
    aa = summ.get("aa")
    aa_ok = None if aa is None else aa["changed"] == 0
    print(f"\n{len(rows)} shadowed decisions ({', '.join(phases)}) in {len(games)} game(s); evaluator "
          f"{tuning.evaluator_mode()}; {time.process_time() - cpu0:.0f} CPU-s, {time.time() - t0:.0f} s wall")
    if aa is not None:
        print(f"A/A identity: {aa['changed']} changed of {aa['n']} -> {'OK' if aa_ok else 'FAILED (nondeterministic)'}")
    for c in cands:
        if c.label == "aa":
            continue
        s = summ[c.label]
        cls = "  ".join(f"{k} {v['changed']}/{v['n']}" for k, v in s["by_class"].items())
        share = f"{s['share']:.1%}" if s["share"] is not None else "n/a"
        ratio = f"{s['ms_ratio_mean']:.2f}" if s["ms_ratio_mean"] else "n/a"
        print(f"  {c.label:24} changed {s['changed']}/{s['n']} ({share}); {cls}; ms ratio {ratio}")
        for kname, cnt in list(s["kinds"].items())[:5]:
            print(f"      {cnt:4d}  {kname}")
        for ph, arms in (s.get("settle_ports") or {}).items():
            d, c = arms["def"], arms["cand"]
            if d["n"]:
                print(f"      settle/{ph}: port picks {d['port']} -> {c['port']} (3:1 {d['generic']} -> "
                      f"{c['generic']}) of {d['n']} decisions")
    rgates = {lab: robber_gates(rows, lab, gs) for lab, gs in gmap.items()}
    if rgates:
        print_robber_gates(rgates)
    if args.json:
        out = {"source": args.source, "spec": spec,
               "evaluator": "python (shadow searches only)" if PY_SHADOW else tuning.evaluator_mode(), "phases": phases,
               "main_every": args.main_every,
               "games": games, "decisions": len(rows), "aa_ok": aa_ok, "candidates": summ,
               "cpu_s": time.process_time() - cpu0}
        if args.keep_rows:
            out["rows"] = rows
        if gmap:
            out["robber_gate_map"] = gmap
            out["robber_gates"] = rgates
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        tmp = args.json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, args.json)
    return 0 if aa_ok in (None, True) else 3


if __name__ == "__main__":
    sys.exit(main())
