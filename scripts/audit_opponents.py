#!/usr/bin/env python3
"""Audit: did two proof tests really play different opponents?  (docs/SCRUTINY.md, Q1)

Reads the per-chunk result JSON and the action logs of two tests that share a base seed
(e.g. T2 = AlphaBeta and T3 = SameTurnAlphaBeta, both seed 900001) and reports:

* the opponent class each test recorded (in the result JSON and in every game's log);
* whether the games share seeds, seats, boards and dev decks (they should: same seed);
* how many games are identical action for action (a duplicated run: all of them);
* where each pair of games first differs, classified as an opponent's move, a chance
  outcome (dice, stolen card, drawn card: catanatron's searches draw from the game's own
  random stream, so a different search shifts later outcomes), or our bot's move (with the
  same history our bot is deterministic, so this should never happen);
* the game-by-game outcome cross-table and, when the totals are equal, the chance of that
  under two independent binomials at the pooled win rate.

Usage (the proof's output directory, or the archived ``proof/`` directory of the repo)::

    python3 scripts/audit_opponents.py --run .../scratchpad/proof/run T2 T3
    python3 scripts/audit_opponents.py --run proof T2 T3
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

CHANCE_TYPES = {"ROLL"}


def _files(run: str, test: str, sub: str, pattern: str) -> List[str]:
    """``<run>/<sub>/<test>/pattern`` (proof output layout) or the archive's ``<run>/<test>/{results,logs}/``."""
    out = sorted(glob.glob(os.path.join(run, sub, test, pattern)))
    return out or sorted(glob.glob(os.path.join(run, test, "results" if sub == "json" else sub, pattern)))


def load(run: str, test: str) -> Tuple[Dict[int, dict], Dict[int, dict], set]:
    results, meta = {}, set()
    for p in _files(run, test, "json", "*.json"):
        d = json.load(open(p))
        meta.add((d["opponent"], d["opponent_class"], d["format"], d["seed"], d["catanatron"]))
        for r in d["results"]:
            results[r["game"]] = r
    logs = {}
    for p in _files(run, test, "logs", "*.jsonl.gz"):
        with gzip.open(p, "rt") as f:
            for line in f:
                g = json.loads(line)
                logs[g["game"]] = g
    return results, logs, meta


def first_difference(a: list, b: list):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def classify(ga: dict, gb: dict, j: int) -> str:
    a, b = ga["actions"], gb["actions"]
    if j >= len(a) or j >= len(b):
        return "one game ended earlier"
    x, y = a[j], b[j]
    if x[0] == y[0] and x[1] == y[1] and (x[1] in CHANCE_TYPES or x[2] == y[2]):
        return "chance outcome (dice / stolen card / drawn card)"
    ours = {ga["colors"][s] for s in ga["our_seats"]}
    return "OUR BOT's move (unexpected)" if x[0] in ours else "opponent's move"


def equal_totals_probability(n: int, p: float) -> float:
    pmf = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    return sum(q * q for q in pmf)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="proof output directory (json/ + logs/) or the archive root")
    ap.add_argument("tests", nargs=2, metavar="TEST")
    args = ap.parse_args(argv)
    (ta, tb) = args.tests
    ra, la, ma = load(args.run, ta)
    rb, lb, mb = load(args.run, tb)
    print(f"{ta}: opponent recorded in the result JSON {sorted(ma)}")
    print(f"{tb}: opponent recorded in the result JSON {sorted(mb)}")
    for t, logs in ((ta, la), (tb, lb)):
        classes = Counter(p.get("class") for g in logs.values() for p in g["players"] if p["kind"] == "opponent")
        print(f"{t}: opponent seats in the {len(logs)} game logs: {dict(classes)}")
    games = sorted(set(ra) & set(rb) & set(la) & set(lb))
    if not games:
        print("no common games")
        return 1
    same = lambda k: sum(json.dumps(la[g][k], sort_keys=True) == json.dumps(lb[g][k], sort_keys=True) for g in games)
    print(f"common games: {len(games)}; same seed {same('seed')}, same seats {same('our_seats')}, "
          f"same board {same('board')}, same dev deck {same('dev_deck')}")
    firsts, kinds = [], Counter()
    for g in games:
        j = first_difference(la[g]["actions"], lb[g]["actions"])
        if j is None:
            kinds["identical game"] += 1
            continue
        firsts.append(j)
        kinds[classify(la[g], lb[g], j)] += 1
    print(f"identical games (every action and outcome): {kinds.pop('identical game', 0)} of {len(games)}")
    if firsts:
        firsts.sort()
        print(f"first difference at action index: median {firsts[len(firsts) // 2]}, "
              f"25 % by {firsts[len(firsts) // 4]}, 75 % by {firsts[3 * len(firsts) // 4]}")
    for k, v in kinds.most_common():
        print(f"  first difference is {k}: {v}")
    wa = {g for g in games if ra[g]["won"]}
    wb = {g for g in games if rb[g]["won"]}
    print(f"outcomes: won in both {len(wa & wb)}, only in {ta} {len(wa - wb)}, only in {tb} {len(wb - wa)}, "
          f"in neither {len(games) - len(wa | wb)}")
    print(f"same game length (turns): {sum(ra[g]['turns'] == rb[g]['turns'] for g in games)} of {len(games)}")
    print(f"totals: {ta} {len(wa)}/{len(games)}, {tb} {len(wb)}/{len(games)}")
    if len(wa) == len(wb):
        p = (len(wa) + len(wb)) / (2 * len(games))
        print(f"P(two independent runs of {len(games)} games at {p:.3f} give equal totals) = "
              f"{equal_totals_probability(len(games), p):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
