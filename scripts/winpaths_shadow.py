#!/usr/bin/env python3
"""Shadow diagnostics for the win-path race term: how often (and how) does ``paths=1`` change the search's
first action, and what does it cost?

    python3 scripts/winpaths_shadow.py --games 4 --seed 7200

Plays ``--games`` games of the DEFAULT search bot (4 seats, ``--spec``) in ONE process.  At every main-phase
decision of a seat with more than one legal action it re-searches the root twice with the bot's own evaluator,
opponent model, belief and politics - once with ``paths=0`` and once with ``paths=1`` (plus ``--paths-keys``),
each from ``random.Random(k)`` with the same ``k`` - and records the two first actions and the two process
times.  The game itself is played by the unchanged default bot, so the positions are the default bot's.

Reports the share of changed first actions, a table of changes by action kind (paths=0 kind -> paths=1 kind),
the mean / p95 ms per decision of both searches and their ratio.  Stage 2 gates (docs/ABLATIONS_WINPATHS.md):
3-20 % changed first actions; ms ratio <= 1.5 mean and <= 2.0 at p95.  CPU: one process, no workers.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from collections import Counter
from typing import List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from catanbot import engine as E  # noqa: E402
from catanbot.search import Searcher  # noqa: E402
from catanbot.selfplay import make_bot, parse_spec, play_game  # noqa: E402
from catanbot.state import PHASE_MAIN  # noqa: E402
from catanbot.tuning import DEFAULT_SEARCH_SPEC  # noqa: E402


def _p95(xs: Sequence[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(0.95 * len(s)) - 1))] if s else float("nan")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7200)
    ap.add_argument("--spec", default=DEFAULT_SEARCH_SPEC, help="the default bot spec that plays the games")
    ap.add_argument("--paths-keys", default="", help="extra paths_* keys for the shadow search, e.g. paths_w=2")
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--json", help="write the per-decision rows here")
    args = ap.parse_args(argv)

    _, kw = parse_spec("search:paths=1" + ("," + args.paths_keys if args.paths_keys else ""))
    cand = make_bot("search:" + ",".join(f"{k}={v}" for k, v in kw.items())).config
    extra = {f.name: getattr(cand, f.name) for f in dataclasses.fields(cand) if f.name.startswith("paths")}
    rows: List[dict] = []
    t_start = time.time()
    for g in range(args.games):
        gseed = args.seed + g
        bots = [make_bot(args.spec) for _ in range(4)]
        k_counter = [0]

        def shadow(state, action, player):
            if state.phase != PHASE_MAIN or E.acting_player(state) != player:
                return
            legal = E.legal_actions(state)
            if len(legal) <= 1:
                return
            bot = bots[player]
            cfg0 = bot.config
            cfg1 = dataclasses.replace(cfg0, **extra)
            k = gseed * 100003 + k_counter[0]
            k_counter[0] += 1
            out = {}
            for tag, cfg in (("off", cfg0), ("on", cfg1)):
                sr = Searcher(bot.evaluator, cfg, bot.model, bot.belief, bot.politics)
                t0 = time.process_time()
                res = sr.search(state, player, random.Random(k))
                out[tag] = (res[0].action if res else None, time.process_time() - t0)
            rows.append({"seed": gseed, "turn": state.turn, "player": player, "played": list(action),
                         "off": list(out["off"][0]) if out["off"][0] else None,
                         "on": list(out["on"][0]) if out["on"][0] else None,
                         "ms_off": 1000.0 * out["off"][1], "ms_on": 1000.0 * out["on"][1]})

        res = play_game(bots, rng=random.Random(gseed), seed=gseed, max_turns=args.max_turns, on_action=shadow)
        print(f"game {g + 1}/{args.games} seed {gseed}: {res.turns} turns, winner {res.winner}, "
              f"{len(rows)} shadow decisions so far", flush=True)
    if not rows:
        print("no main-phase decisions recorded")
        return 1
    changed = [r for r in rows if r["off"] != r["on"]]
    kinds = Counter(f"{r['off'][0] if r['off'] else '-'} -> {r['on'][0] if r['on'] else '-'}" for r in changed)
    ms_off = [r["ms_off"] for r in rows]
    ms_on = [r["ms_on"] for r in rows]
    mean_off, mean_on = sum(ms_off) / len(ms_off), sum(ms_on) / len(ms_on)
    print(f"\n{len(rows)} main-phase roots, {len(changed)} changed first actions ({len(changed) / len(rows):.1%})")
    for kname, c in kinds.most_common():
        print(f"  {c:5d}  {kname}")
    print(f"ms per decision: off {mean_off:.1f} (p95 {_p95(ms_off):.1f}), on {mean_on:.1f} (p95 {_p95(ms_on):.1f}); "
          f"ratio mean {mean_on / mean_off:.2f}, p95 {_p95(ms_on) / _p95(ms_off):.2f}")
    print(f"{time.time() - t_start:.0f}s")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"spec": args.spec, "paths": extra, "rows": rows,
                       "changed_rate": len(changed) / len(rows), "by_kind": dict(kinds),
                       "ms_ratio_mean": mean_on / mean_off, "ms_ratio_p95": _p95(ms_on) / _p95(ms_off)}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
