#!/usr/bin/env python3
"""Benchmark the rules engine with uniformly random self-play.

Usage::

    python3 scripts/bench_engine.py [--games N] [--players P] [--seed S] [--max-turns T]

Plays ``N`` random games (random boards, random legal actions) on one core
and prints games/sec, actions/sec, the average number of completed turns and
actions per game and how many games ended by victory rather than by the turn
cap.  Random play is much more trade-heavy than real play (every offer is a
separate action), so this is a conservative lower bound for the engine's
throughput under a real bot.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanbot import engine as E  # noqa: E402
from catanbot.state import new_game  # noqa: E402


def play_one(seed: int, players: int, max_turns: int) -> tuple:
    """Play one random game; returns (turns, actions, ended_by_vp)."""
    rng = random.Random(seed)
    s = new_game(players, rng)
    s.max_turns = max_turns
    actions = 0
    choice = rng.choice
    while not E.is_terminal(s):
        E.apply_inplace(s, choice(E.legal_actions(s)), rng)
        actions += 1
    return s.turn, actions, s.turn < max_turns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=50, help="number of random games (default 50)")
    ap.add_argument("--players", type=int, default=4, help="players per game, 3 or 4 (default 4)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first game (default 0)")
    ap.add_argument("--max-turns", type=int, default=400, help="turn cap per game (default 400)")
    args = ap.parse_args(argv)

    total_turns = total_actions = wins = 0
    t0 = time.perf_counter()
    for g in range(args.games):
        turns, actions, by_vp = play_one(args.seed + g, args.players, args.max_turns)
        total_turns += turns
        total_actions += actions
        wins += by_vp
    dt = time.perf_counter() - t0
    games = max(args.games, 1)
    print(f"games            : {args.games} ({args.players} players, max_turns={args.max_turns})")
    print(f"time             : {dt:.2f} s")
    print(f"games/sec        : {games / dt:.2f}")
    print(f"actions/sec      : {total_actions / dt:,.0f}")
    print(f"avg turns/game   : {total_turns / games:.1f}")
    print(f"avg actions/game : {total_actions / games:.0f}")
    print(f"ended by 10 VP   : {wins}/{args.games}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
