#!/usr/bin/env python3
"""Audit the development deck and the bank in logged Catanatron games (docs/SCRUTINY.md, Q21).

    python3 scripts/audit_devdeck.py proof bench_1v1

Checks, over every logged game under the given directories:
1. every game's deck is the standard 25 cards: 14 Knight, 5 Victory Point, 2 Road Building, 2 Year of Plenty,
   2 Monopoly;
2. at the end, bank + all hands = 19 of each resource, and deck left + held + played = 25;
3. draws come off the logged deck in order;
4. the shuffle is uniform: where the Victory Point cards sit in the distinct decks, against simulated
   uniform shuffles (tests reuse seeds, so duplicates are counted once);
5. nobody draws Victory Point cards more often than the cards still in the deck allow: for each purchase, the
   chance of a Victory Point card is (VP cards not yet drawn) / (cards not yet drawn); observed minus expected
   over all purchases, as a z-score.  A bot that saw the deck order and bought when a VP card was next
   would show a large positive z.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
import random
import sys

STANDARD = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2, "YEAR_OF_PLENTY": 2, "MONOPOLY": 2}
RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def records(dirs):
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.jsonl.gz"), recursive=True)):
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sims", type=int, default=300, help="uniform-shuffle simulations for check 4")
    args = ap.parse_args(argv)
    n = bad_deck = bad_bank = bad_dev = 0
    order = collections.Counter()
    decks = set()
    acc = {"ours": [0, 0.0, 0.0, 0], "opponents": [0, 0.0, 0.0, 0]}
    for rec in records(args.dirs):
        n += 1
        deck = rec["dev_deck"]
        decks.add(tuple(deck))
        bad_deck += dict(collections.Counter(deck)) != STANDARD
        st = rec["final"]["state"]
        bad_bank += any(st["bank"][r] + sum(p["resources"][r] for p in st["players"]) != 19 for r in RES)
        held = sum(sum(p["dev_cards"].values()) + sum(p["dev_played"].values()) for p in st["players"])
        bad_dev += held + st["dev_deck_left"] != 25
        ours = {p["color"] for p in rec["players"] if p.get("kind") == "catanbot"}
        drawn = []
        vp_left, left = 5, 25
        for color, kind, value, result in rec["actions"]:
            if kind != "BUY_DEVELOPMENT_CARD":
                continue
            card = result or value
            drawn.append(card)
            p = vp_left / left
            a = acc["ours" if color in ours else "opponents"]
            a[0] += card == "VICTORY_POINT"
            a[1] += p
            a[2] += p * (1 - p)
            a[3] += 1
            vp_left -= card == "VICTORY_POINT"
            left -= 1
        if drawn:
            back = deck[::-1][:len(drawn)] == drawn
            order["in order (from the end of the list)" if back else "out of order"] += 1
    print(f"games {n}; distinct decks {len(decks)}")
    print(f"1. decks that are not the standard 14/5/2/2/2: {bad_deck}")
    print(f"2. games where bank + hands != 19 of a resource: {bad_bank}; where deck + held + played != 25: {bad_dev}")
    print(f"3. draw order: {dict(order)}")
    exp = len(decks) * 5 / 25

    def chi2(ds):
        pos = collections.Counter(i for d in ds for i, c in enumerate(d) if c == "VICTORY_POINT")
        return sum((pos[i] - exp) ** 2 / exp for i in range(25))

    obs = chi2(decks)
    base = [c for c, k in STANDARD.items() for _ in range(k)]
    rng = random.Random(1)
    sims = []
    for _ in range(args.sims):
        ds = []
        for _ in range(len(decks)):
            d = base[:]
            rng.shuffle(d)
            ds.append(d)
        sims.append(chi2(ds))
    print(f"4. VP position chi2 {obs:.1f}; share of {args.sims} uniform-shuffle simulations at least as large: "
          f"{sum(s >= obs for s in sims) / len(sims):.2f}")
    for who, (o, e, v, k) in acc.items():
        if k:
            print(f"5. {who}: {k} purchases, {o} VP cards drawn, {e:.1f} expected from the cards left, "
                  f"z = {(o - e) / math.sqrt(v):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
