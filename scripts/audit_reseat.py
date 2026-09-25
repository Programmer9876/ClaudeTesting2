#!/usr/bin/env python3
"""Fairness audit: does the benchmark harness change how Catanatron's bots play?

For each seed, one Catanatron AlphaBetaPlayer and three ValueFunctionPlayers are
seated in an order Catanatron's own seat shuffle would *not* pick for that seed,
three ways:

* ``native``  - catanatron's ``Game`` with its seat shuffle forced to that order
  (the shuffle still runs and consumes the same random draws);
* ``harness`` - :func:`catanbot.bench.catanatron_adapter.make_game` (the re-seat
  every benchmark and proof game uses), raw catanatron players;
* ``wrapped`` - the same with every player inside ``BenchOpponent`` (the timing /
  trade-rule wrapper every benchmark opponent runs in).

The initial states are compared field by field, then all three games are played
to the end and every action - with its dice, steal and dev-draw outcome - is
compared.  ``IDENTICAL`` means the harness is exactly Catanatron's own game.
Needs the catanatron 3.3 engine (``catanatron.players.minimax``)::

    /home/user/venv_cat33/bin/python scripts/audit_reseat.py --games 8

The fast version of this check (random players, both catanatron versions) is
``tests/test_catanatron_adapter.py::test_reseating_equals_catanatron_native_seating``.
Result on 2026-09-25: 8 of 8 games identical (2,683 actions), docs/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanatron import Color, Game  # noqa: E402
from catanatron.players.minimax import AlphaBetaPlayer  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402

from catanbot.bench.catanatron_adapter import BenchOpponent, action_log, make_game, playable_actions_of  # noqa: E402

COLS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def players(order, ab_color, wrap=False):
    out = []
    for c in order:
        p = AlphaBetaPlayer(c) if c == ab_color else ValueFunctionPlayer(c)
        out.append(BenchOpponent(p, trade_rule="off") if wrap else p)
    return out


def native_seated(ps, seed):
    """catanatron's own Game with the seat shuffle's result forced to ``ps``."""
    orig = random.Random.sample

    def forced(self, population, k, *a, **kw):
        r = orig(self, population, k, *a, **kw)          # consume exactly the same draws
        return list(ps) if {id(p) for p in population} == {id(p) for p in ps} else r

    random.Random.sample = forced
    try:
        return Game(list(ps), seed=seed)
    finally:
        random.Random.sample = orig


def snapshot(g):
    s = g.state
    return {
        "colors": [c.value for c in s.colors], "players": [p.color.value for p in s.players],
        "color_to_index": sorted((c.value, i) for c, i in s.color_to_index.items()),
        "player_state": sorted(s.player_state.items()), "dev_deck": list(s.development_listdeck),
        "bank": list(s.resource_freqdeck),
        "tiles": sorted((str(k), str(t.resource), t.number) for k, t in s.board.map.land_tiles.items()),
        "robber": str(s.board.robber_coordinate), "rng": str(s.random.getstate()),
        "prompt": str(s.current_prompt), "current": [s.current_player_index, s.current_turn_index],
        "playable": [str(a) for a in playable_actions_of(g)],
        "discard_counts": list(s.discard_counts), "acceptees": list(s.acceptees),
    }


def first_difference(a, b):
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=910001)
    args = ap.parse_args(argv)
    rows = []
    for i in range(args.games):
        seed = args.seed + i
        ab = COLS[i % 4]
        own = [p.color for p in Game(players(COLS, ab), seed=seed).state.players]
        order = list(reversed(own)) if i % 2 == 0 else own[1:] + own[:1]
        t0 = time.time()
        games = {"native": native_seated(players(order, ab), seed),
                 "harness": make_game(players(order, ab), seed),
                 "wrapped": make_game(players(order, ab, wrap=True), seed)}
        snaps = {k: snapshot(g) for k, g in games.items()}   # each game has its own Random on 3.3
        played = {}
        for k, g in games.items():
            w = g.play()
            played[k] = (w.value if w else None, [str(a) for a in action_log(g.state)])
        row = {"seed": seed, "alphabeta_seat": order.index(ab), "catanatron_order": [c.value for c in own],
               "order": [c.value for c in order],
               "initial_state_differences": sorted(f"{k}:{f}" for k in ("harness", "wrapped")
                                                   for f in snaps["native"] if snaps[k][f] != snaps["native"][f]),
               "winner": [played[k][0] for k in games], "actions": [len(played[k][1]) for k in games],
               "first_action_difference": {k: first_difference(played["native"][1], played[k][1])
                                           for k in ("harness", "wrapped")},
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    ok = all(not r["initial_state_differences"] and all(v is None for v in r["first_action_difference"].values())
             for r in rows)
    print(json.dumps({"reseat_equivalence": "IDENTICAL" if ok else "DIFFERENT", "games": len(rows),
                      "actions_compared": sum(r["actions"][0] for r in rows)}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
