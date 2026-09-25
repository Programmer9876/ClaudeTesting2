#!/usr/bin/env python3
"""Round-robin ladder of catanatron players (4-player games, seat rotation).

Usage::

    python3 scripts/catanatron_ladder.py --games 12 [--types R,W,V,F,A] [--workers 2]
                                         [--seed 0] [--ab-budget 2000] [--smart-discard]
                                         [--json results.json]

Player types (single letters, repeatable in ``--types``):

    R  catanatron RandomPlayer              W  catanatron WeightedRandomPlayer
    V  catanatron VictoryPointPlayer        F  catanbot ValueFunctionPlayer
    A  catanbot AlphaBetaPlayer

Every 4-subset of ``--types`` (in order) plays ``--games`` games; the seat
order rotates with the game index and the engine additionally shuffles the
seating from the game seed.  Prints, per player type, games / wins / win
rate / average actual VP, plus the average game length and time.  Games
that reach catanatron's turn limit without a winner are counted as played
(no win).  ``--smart-discard`` runs the games through
``catanbot.bench.catanatron_players.play_game`` so the catanbot players choose
their discards (the stock engine loop discards at random for everyone).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanatron.game import Game  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402
from catanatron.players.search import VictoryPointPlayer  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402
from catanatron.state_functions import get_actual_victory_points  # noqa: E402

from catanbot.bench.catanatron_players import (  # noqa: E402
    AlphaBetaPlayer,
    ValueFunctionPlayer,
    play_game,
)

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
TYPE_NAMES = {
    "R": "RandomPlayer",
    "W": "WeightedRandomPlayer",
    "V": "VictoryPointPlayer",
    "F": "ValueFunctionPlayer",
    "A": "AlphaBetaPlayer",
}


def make_player(kind: str, color: Color, opts: dict):
    if kind == "R":
        return RandomPlayer(color)
    if kind == "W":
        return WeightedRandomPlayer(color)
    if kind == "V":
        return VictoryPointPlayer(color)
    if kind == "F":
        return ValueFunctionPlayer(color)
    if kind == "A":
        return AlphaBetaPlayer(color, budget=opts.get("ab_budget", 2000),
                               depth=opts.get("ab_depth", 3), beam=opts.get("ab_beam", 5))
    raise ValueError(f"unknown player type {kind!r} (use one of {''.join(TYPE_NAMES)})")


def play_one(job):
    """Worker: play one game. ``job`` = (kinds tuple in seat order, seed, opts)."""
    kinds, seed, opts = job
    players = [make_player(k, COLORS[i], opts) for i, k in enumerate(kinds)]
    game = Game(players, seed=seed, vps_to_win=opts.get("vps", 10))
    t0 = time.perf_counter()
    if opts.get("smart_discard"):
        winner = play_game(game, smart_discard=True)
    else:
        winner = game.play()
    dt = time.perf_counter() - t0
    result = {
        "seed": seed,
        "kinds": list(kinds),
        "winner": None,
        "vps": [],
        "turns": game.state.num_turns,
        "time": dt,
    }
    for i, k in enumerate(kinds):
        vp = get_actual_victory_points(game.state, COLORS[i])
        result["vps"].append(vp)
        if winner == COLORS[i]:
            result["winner"] = i
    return result


def build_jobs(types, games, seed, opts):
    jobs = []
    idx = 0
    for subset in itertools.combinations(range(len(types)), 4):
        kinds = [types[i] for i in subset]
        for g in range(games):
            rot = g % 4
            order = tuple(kinds[rot:] + kinds[:rot])
            jobs.append((order, seed + idx, opts))
            idx += 1
    return jobs


def summarize(results):
    games = defaultdict(int)
    wins = defaultdict(int)
    vps = defaultdict(float)
    for r in results:
        for i, k in enumerate(r["kinds"]):
            games[k] += 1
            vps[k] += r["vps"][i]
            if r["winner"] == i:
                wins[k] += 1
    rows = []
    for k in sorted(games, key=lambda k: -(wins[k] / games[k])):
        rows.append((k, TYPE_NAMES[k], games[k], wins[k], wins[k] / games[k], vps[k] / games[k]))
    return rows


def print_summary(results, elapsed):
    rows = summarize(results)
    print(f"{'type':<4} {'player':<22} {'games':>5} {'wins':>5} {'win%':>6} {'avg VP':>7}")
    for k, name, n, w, rate, vp in rows:
        print(f"{k:<4} {name:<22} {n:>5} {w:>5} {100 * rate:>5.1f}% {vp:>7.2f}")
    n = max(len(results), 1)
    unfinished = sum(1 for r in results if r["winner"] is None)
    print(f"games: {len(results)}  no-winner: {unfinished}  avg turns: "
          f"{sum(r['turns'] for r in results) / n:.1f}  avg game time: "
          f"{sum(r['time'] for r in results) / n:.2f}s  wall: {elapsed:.1f}s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=12, help="games per 4-player subset (default 12)")
    ap.add_argument("--types", default="R,W,V,F,A",
                    help="comma-separated player types, at least 4 (default R,W,V,F,A)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first game (default 0)")
    ap.add_argument("--workers", type=int, default=2, help="worker processes (default 2)")
    ap.add_argument("--ab-budget", type=int, default=2000, help="AlphaBetaPlayer node budget")
    ap.add_argument("--ab-depth", type=int, default=3, help="AlphaBetaPlayer own-action depth")
    ap.add_argument("--ab-beam", type=int, default=5, help="AlphaBetaPlayer beam width")
    ap.add_argument("--vps", type=int, default=10, help="victory points to win (default 10)")
    ap.add_argument("--smart-discard", action="store_true",
                    help="let catanbot players choose their discards (play_game loop)")
    ap.add_argument("--json", default=None, help="append the per-game results to this JSON-lines file")
    args = ap.parse_args(argv)

    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    if len(types) < 4:
        ap.error("--types needs at least 4 entries")
    for t in types:
        if t not in TYPE_NAMES:
            ap.error(f"unknown type {t!r}")
    opts = {
        "ab_budget": args.ab_budget,
        "ab_depth": args.ab_depth,
        "ab_beam": args.ab_beam,
        "vps": args.vps,
        "smart_discard": args.smart_discard,
    }
    jobs = build_jobs(types, args.games, args.seed, opts)
    print(f"{len(jobs)} games: types={','.join(types)} games/subset={args.games} seed={args.seed} "
          f"workers={args.workers} ab_budget={args.ab_budget} smart_discard={args.smart_discard}")
    t0 = time.perf_counter()
    if args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers) as pool:
            results = pool.map(play_one, jobs, chunksize=1)
    else:
        results = [play_one(job) for job in jobs]
    elapsed = time.perf_counter() - t0
    print_summary(results, elapsed)
    if args.json:
        with open(args.json, "a") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
