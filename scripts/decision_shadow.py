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
static evaluator (``needs_python_evaluator``) re-executes the script with ``CATANBOT_NO_ACCEL=1``.

Phase classes (``--phases``, default all): setup, roll, main, robber (the robber move after a 7), discard, trade
(answering / selecting), knight (any decision where playing a knight is legal - the knight's hex and victim are part
of that action) and buy_dev (buying a development card is legal).  A decision can be in several classes (roll +
knight, main + buy_dev).  ``--main-every N`` shadows only every N-th main-phase decision that is in no other
covered class (main decisions are ~80% of all; the robber shadow samples 1 in 5).

Output: per candidate the changed share overall and per class, the changed kinds (default kind -> candidate
kind), the mean and p95 ms of both searches and their ratio.  ``scripts/run_queue.py`` reads the JSON through
:func:`gate_share` to skip a prior-only row as NOOP(shadow) at 0 games when it changes fewer than 2% of the
decisions in its classes (the robber rows: robber + knight).  One process, no workers.
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

CLASSES = ("setup", "roll", "main", "robber", "discard", "trade", "knight", "buy_dev")


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
    return tuple(out)


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
        from catanbot import tuning
        from catanbot.search import Searcher
        token = tuning.apply(overrides) if overrides else None
        try:
            sr = Searcher(self.inner.evaluator, cfg, self.inner.model, self.inner.belief, self.inner.politics)
            t0 = time.process_time()
            res = sr.search(state, me, random.Random(k))
            dt = time.process_time() - t0
        finally:
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
        out[c.label] = {"overrides": {k: v for k, v in c.overrides.items()}, "spec_keys": dict(c.spec_keys),
                        "n": n, "changed": changed, "share": changed / n if n else None,
                        "by_class": {k: dict(v, share=(v["changed"] / v["n"] if v["n"] else None))
                                     for k, v in by.items() if v["n"]},
                        "kinds": dict(kinds.most_common()), "ms_def_mean": md, "ms_mean": mc,
                        "ms_ratio_mean": mc / md if md and md == md else None,
                        "ms_ratio_p95": (_p95(ms_c) / _p95(ms_d)) if ms_d and _p95(ms_d) > 0 else None}
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
    ap.add_argument("--json", help="write the result here")
    args = ap.parse_args(argv)
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
    if any(c.needs_python() for c in cands):
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
    if args.json:
        out = {"source": args.source, "spec": spec, "evaluator": tuning.evaluator_mode(), "phases": phases,
               "main_every": args.main_every,
               "games": games, "decisions": len(rows), "aa_ok": aa_ok, "candidates": summ,
               "cpu_s": time.process_time() - cpu0}
        if args.keep_rows:
            out["rows"] = rows
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        tmp = args.json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, args.json)
    return 0 if aa_ok in (None, True) else 3


if __name__ == "__main__":
    sys.exit(main())
