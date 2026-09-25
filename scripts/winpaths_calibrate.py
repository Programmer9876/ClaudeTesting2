#!/usr/bin/env python3
"""Calibrate the win-path race model (catanbot/winpaths.py) on self-play: does its P(hold the award at game end)
predict the final Longest Road / Largest Army holder better than two simple baselines?

    python3 scripts/winpaths_calibrate.py --bot heuristic:temp=0.15 --games 80 --seed 7000
    python3 scripts/winpaths_calibrate.py --bot "search:depth=1,beam=4,expand=8,evaluator=heuristic" --games 20 \\
        --seed 7100 --json data/winpaths_calib_search.json

Plays ``--games`` games in ONE process (4 seats of ``--bot``), records every turn-start state (the roll phase of
each main-game turn) and labels it with the final holder of each award (or "nobody").  For each race it reports
the mean log-loss of

* ``model``      - the race model's P (``PathsContext.races``), including P(nobody) while nobody holds it;
* ``holder85``   - "the holder keeps it with 0.85" (0.15 spread over the other seats; uniform over the seats and
                   "nobody" while unclaimed);
* ``level_only`` - a softmax over the current levels only (``solve_race`` with a = G = 0: no hand, no growth).

It also fits the horizon regression (remaining rounds ~ a + b (10 - vmax)) on the finished games and reports its
MAE next to the MAE of the model's clamped horizon ``H``.  Stage 1 gates (docs/ABLATIONS_WINPATHS.md): the model
beats both baselines by >= 0.1 nats on both races, horizon MAE <= 3.5 rounds.  CPU: one process, no workers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from catanbot import board as B  # noqa: E402
from catanbot import winpaths as W  # noqa: E402
from catanbot.selfplay import make_bot, play_game  # noqa: E402
from catanbot.state import PHASE_ROLL  # noqa: E402

EPS = 1e-6


def _logloss(probs: Sequence[float], label: int) -> float:
    return -math.log(max(EPS, probs[label]))


def _dist(sol: W.RaceSolution, n: int) -> List[float]:
    """Seats 0..n-1 then "nobody" (index n)."""
    return list(sol.P) + [sol.nobody]


def _holder85(holder: int, n: int) -> List[float]:
    if holder < 0:
        return [1.0 / (n + 1)] * (n + 1)
    out = [0.15 / (n - 1)] * n + [0.0]
    out[holder] = 0.85
    return out


def _level_only(ctx: W.PathsContext, race: int, b_levels: Sequence[float], holder: int) -> List[float]:
    n = len(b_levels)
    zeros = tuple(0.0 for _ in range(n))
    min_level = ctx.params["MIN_LR"] if race == W.LR else ctx.params["MIN_LA"]
    cost = ctx.c_lr if race == W.LR else ctx.c_la
    sol = W.solve_race(tuple(b_levels), zeros, zeros, holder, min_level, cost, ctx.params)
    return _dist(sol, n)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="heuristic:temp=0.15", help="bot spec for all four seats")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--json", help="write the summary (and per-state rows) here")
    args = ap.parse_args(argv)

    t0 = time.time()
    loss: Dict[str, Dict[str, List[float]]] = {r: {"model": [], "holder85": [], "level_only": []} for r in ("lr", "la")}
    horizon_rows: List[tuple] = []
    per_state: List[dict] = []
    n_games = 0
    for g in range(args.games):
        gseed = args.seed + g
        states: List = []
        last = {}

        def rec(state, action, player):
            last["s"] = state
            last["a"] = action
            if state.phase == PHASE_ROLL and state.turn >= 2 * state.num_players:
                if not states or states[-1].turn != state.turn:
                    states.append(state.copy())

        bots = [make_bot(args.bot) for _ in range(4)]
        res = play_game(bots, rng=random.Random(gseed), seed=gseed, max_turns=args.max_turns, on_action=rec)
        # the state object is mutated in place by play_game: after the loop it is the final state
        fin = last.get("s")
        if fin is None:
            continue
        n_games += 1
        lr_final = fin.longest_road_owner
        la_final = fin.largest_army_owner
        finished = res.winner >= 0
        for s in states:
            n = s.num_players
            ctx = W.PathsContext(s, s.current)
            leaf = ctx._inputs(s)
            for race, (b, a, G), holder, final in ((W.LR, leaf.lr, leaf.lr_holder, lr_final),
                                                   (W.LA, leaf.la, leaf.la_holder, la_final)):
                key = "lr" if race == W.LR else "la"
                label = final if final >= 0 else n
                sol = ctx._solve(race, b, a, G, holder)
                levels = [b[i] - a[i] for i in range(n)]
                m = _dist(sol, n)
                h85 = _holder85(holder, n)
                lvl = _level_only(ctx, race, levels, holder)
                loss[key]["model"].append(_logloss(m, label))
                loss[key]["holder85"].append(_logloss(h85, label))
                loss[key]["level_only"].append(_logloss(lvl, label))
                if args.json:
                    per_state.append({"seed": gseed, "turn": s.turn, "race": key, "label": label,
                                      "model": m, "holder85": h85, "level_only": lvl})
            if finished:
                vmax = ctx.vmax
                remaining = (res.turns - s.turn) / float(n)
                horizon_rows.append((B.VP_TO_WIN - vmax, remaining, ctx.H))
        print(f"game {g + 1}/{args.games} seed {gseed}: {res.turns} turns, winner {res.winner}, {len(states)} states, "
              f"final LR {lr_final} LA {la_final}", flush=True)

    summary: Dict[str, object] = {"bot": args.bot, "games": n_games, "seed": args.seed, "seconds": time.time() - t0}
    print()
    print(f"{'race':5} {'states':>7} {'model':>8} {'holder85':>9} {'level_only':>11} {'gain_vs_h85':>12} {'gain_vs_lvl':>12}")
    for key in ("lr", "la"):
        d = loss[key]
        k = len(d["model"])
        if not k:
            continue
        mean = {m: sum(v) / k for m, v in d.items()}
        g1 = mean["holder85"] - mean["model"]
        g2 = mean["level_only"] - mean["model"]
        summary[key] = {"states": k, **{f"logloss_{m}": v for m, v in mean.items()},
                        "gain_vs_holder85": g1, "gain_vs_level_only": g2, "gate_pass": g1 >= 0.1 and g2 >= 0.1}
        print(f"{key:5} {k:7d} {mean['model']:8.3f} {mean['holder85']:9.3f} {mean['level_only']:11.3f} "
              f"{g1:12.3f} {g2:12.3f}")
    if len(horizon_rows) >= 3:
        xs = [r[0] for r in horizon_rows]
        ys = [r[1] for r in horizon_rows]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
        icpt = my - slope * mx
        mae_fit = sum(abs(icpt + slope * x - y) for x, y in zip(xs, ys)) / len(xs)
        mae_h = sum(abs(h - y) for _x, y, h in horizon_rows) / len(horizon_rows)
        summary["horizon"] = {"states": len(horizon_rows), "intercept": icpt, "slope": slope, "mae_fit": mae_fit,
                              "mae_model_H": mae_h, "gate_pass": mae_h <= 3.5}
        print(f"\nhorizon: remaining rounds = {icpt:.2f} + {slope:.2f} (10 - vmax) on {len(xs)} states, fit MAE "
              f"{mae_fit:.2f}; model H (H_A={W.H_A}, H_B={W.H_B}, clamped) MAE {mae_h:.2f} rounds")
    print(f"\n{n_games} games in {summary['seconds']:.0f}s")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "states": per_state}, f)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
