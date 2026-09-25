#!/usr/bin/env python3
"""Benchmark catanbot against the players shipped with the ``catanatron`` engine.

Usage::

    python3 scripts/bench_catanatron.py --games 20 --opponent vp \\
        --spec "search:depth=1,evaluator=heuristic" [--seed 0] [--workers 1] [--json out.json] \\
        [--vps-to-win 10] [--discard-limit 7]
    python3 scripts/bench_catanatron.py --ladder strong --games 20 --workers 2
    python3 scripts/bench_catanatron.py --list-opponents

Every game seats one :class:`catanbot.bench.catanatron_adapter.CatanbotPlayer`
(built from ``--spec``, see ``catanbot.selfplay.make_bot``) against three
copies of the chosen catanatron opponent:

* ``vp``       - ``VictoryPointPlayer`` (greedy one-ply VP maximiser, the strongest
  stock player of the PyPI 3.2.1 wheel)
* ``weighted`` - ``WeightedRandomPlayer`` (random, biased to cities > settlements > dev cards)
* ``random``   - ``RandomPlayer`` (uniform random)
* ``vf`` / ``ab`` - our own value-function / alpha-beta players built inside the
  catanatron engine (``catanbot.bench.catanatron_players``; both versions)
* ``value`` / ``alphabeta`` / ``sameturn`` / ``playouts`` / ``mcts`` - catanatron's
  own strong players (``catanatron.players.value`` / ``minimax`` / ``playouts`` /
  ``mcts``), shipped by the 3.3 engine of the GitHub checkout only

Both catanatron generations work (3.2.1 wheel, 3.3 checkout; see the adapter).
A preset that the running catanatron does not ship fails with a one-line
"not available on catanatron X" message and exit code 2 when it is the only
opponent asked for; inside a comma list or a ``--ladder`` it is skipped with
the same message (exit code 2 only if nothing could be played).
``--list-opponents`` shows which presets resolve here.

Seats rotate (game ``g`` puts catanbot in seat ``g % 4``), boards / dev decks
/ dice come from catanatron seeded per game, so a run is reproducible.  The
script prints the win rate (a random seat would win 25 %), average victory
points, per-seat results, turn counts and adapter statistics, plus a Markdown
table row ready for ``docs/BENCHMARKS.md``.  ``--workers`` plays games in
parallel processes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanatron.game import TURNS_LIMIT  # noqa: E402
from catanatron.models.player import RandomPlayer  # noqa: E402
from catanatron.players.search import VictoryPointPlayer  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402

from catanbot.bench.catanatron_adapter import (  # noqa: E402
    CATANATRON_VERSION,
    COLORS,
    DEFAULT_SPEC,
    CatanbotPlayer,
    play_game,
)

#: Stock players present in every catanatron version.
OPPONENTS = {
    "vp": VictoryPointPlayer,
    "weighted": WeightedRandomPlayer,
    "random": RandomPlayer,
}

# Presets that exist only in some catanatron versions (the GitHub checkout has the strong
# players; PyPI 3.2.1 does not) or in this repo; resolved lazily by import path.
PRESETS = {
    "alphabeta": "catanatron.players.minimax:AlphaBetaPlayer",
    "sameturn": "catanatron.players.minimax:SameTurnAlphaBetaPlayer",
    "value": "catanatron.players.value:ValueFunctionPlayer",
    "mcts": "catanatron.players.mcts:MCTSPlayer",
    "playouts": "catanatron.players.playouts:GreedyPlayoutsPlayer",
    "vf": "catanbot.bench.catanatron_players:ValueFunctionPlayer",
    "ab": "catanbot.bench.catanatron_players:AlphaBetaPlayer",
}
LADDERS = {
    "controls": ["random", "weighted", "vp"],
    "standins": ["vf", "ab"],
    "strong": ["value", "alphabeta", "sameturn", "playouts", "mcts"],
    "full": ["random", "weighted", "vp", "vf", "ab", "value", "alphabeta", "sameturn", "playouts", "mcts"],
}


class OpponentUnavailable(SystemExit):
    """An opponent that cannot be used here: exit code 2 and a one-line ``message`` (no traceback)."""

    def __init__(self, message: str):
        super().__init__(2)
        self.message = message

    def __str__(self) -> str:
        return self.message


def opponent_path(name: str) -> Optional[str]:
    """``module:Class`` of a preset or import path (``None`` for an unknown preset name)."""
    if name in OPPONENTS:
        cls = OPPONENTS[name]
        return f"{cls.__module__}:{cls.__name__}"
    path = PRESETS.get(name, name)
    if ":" in path:
        return path
    if "." in path:
        mod, cls = path.rsplit(".", 1)
        return f"{mod}:{cls}"
    return None


def resolve_opponent(name: str):
    """Opponent class from a preset key or an import path ``module:Class`` / ``module.Class``.

    Raises :class:`OpponentUnavailable` (a ``SystemExit`` with code 2 and a
    one-line message) for an unknown preset or one the running catanatron
    version does not ship.
    """
    if name in OPPONENTS:
        return OPPONENTS[name]
    path = opponent_path(name)
    if path is None:
        raise OpponentUnavailable(f"unknown opponent '{name}' (presets: {', '.join(sorted(OPPONENTS) + sorted(PRESETS))}, "
                                  "or an import path module:Class)")
    mod, cls = path.split(":", 1)
    try:
        module = importlib.import_module(mod)
        if hasattr(module, "USE_MULTIPROCESSING"):
            # catanatron 3.3's GreedyPlayoutsPlayer opens a Pool(cpu_count()) per decision, which
            # would exceed --workers and cannot run inside our (daemonic) worker processes; the
            # seeded playouts give the same result in-process.
            module.USE_MULTIPROCESSING = False
        return getattr(module, cls)
    except (ImportError, AttributeError) as ex:
        hint = ("; the strong players need the 3.3 engine: pip install -e <GitHub clone of catanatron>"
                if mod.startswith("catanatron.") else "")
        raise OpponentUnavailable(f"opponent '{name}' ({path}) is not available on catanatron {CATANATRON_VERSION}: "
                                  f"{ex}{hint}")


def list_opponents() -> List[Tuple[str, str, Optional[type], str]]:
    """``(preset, module:Class, class or None, status)`` for every preset on the running version."""
    rows = []
    for name in list(OPPONENTS) + list(PRESETS):
        path = opponent_path(name) or "?"
        try:
            cls = resolve_opponent(name)
            rows.append((name, path, cls, "available"))
        except OpponentUnavailable as ex:
            rows.append((name, path, None, ex.message.split(": ", 1)[-1] if ": " in ex.message else ex.message))
    return rows


def print_opponents() -> None:
    print(f"opponent presets on catanatron {CATANATRON_VERSION}:")
    for name, path, cls, status in list_opponents():
        mark = "available" if cls is not None else f"not available on catanatron {CATANATRON_VERSION}"
        print(f"  {name:<10} {path:<56} {mark}")
    for ladder, names in LADDERS.items():
        print(f"ladder {ladder:<9}: {', '.join(names)}")


def game_seed(base_seed: int, g: int) -> int:
    """Non-zero per-game seed (catanatron treats seed 0 as 'random')."""
    return base_seed * 100003 + g + 1


def run_one(job: tuple) -> Dict[str, object]:
    """Play game ``g`` of a batch; returns the summary dict plus adapter stats."""
    g, base_seed, spec, opponent, vps_to_win, discard_limit = job
    seat = g % len(COLORS)
    seed = game_seed(base_seed, g)
    opp_cls = resolve_opponent(opponent)
    players = []
    me = None
    for i, color in enumerate(COLORS):
        if i == seat:
            me = CatanbotPlayer(color, spec=spec, seed=seed)
            players.append(me)
        else:
            players.append(opp_cls(color))
    res = play_game(players, seed=seed, vps_to_win=vps_to_win, discard_limit=discard_limit)
    assert me is not None
    res["game"] = g
    res["seat"] = seat
    res["our_vp"] = res["vps"][seat]
    res["opp_vps"] = [v for i, v in enumerate(res["vps"]) if i != seat]
    res["won"] = res["winner_seat"] == seat
    res["stats"] = dict(me.stats)
    res["unmapped_kinds"] = dict(me.unmapped_kinds)
    return res


STAT_KEYS = ["decisions", "searched", "trivial", "pending_robber", "pending_discard", "trade_prompts",
             "unmapped_top", "fallback", "errors", "observe_errors", "observed", "search_time"]


def summarize(results: List[Dict[str, object]], spec: str, opponent: str, seed: int) -> Dict[str, object]:
    n = len(results)
    wins = sum(1 for r in results if r["won"])
    seats = len(COLORS)
    by_seat = {s: [0, 0] for s in range(seats)}
    for r in results:
        by_seat[r["seat"]][1] += 1
        if r["won"]:
            by_seat[r["seat"]][0] += 1
    opp_avg = [sum(r["opp_vps"]) / len(r["opp_vps"]) for r in results]
    opp_best = [max(r["opp_vps"]) for r in results]
    stats = {k: sum(r["stats"].get(k, 0) for r in results) for k in STAT_KEYS}
    truncated = sum(1 for r in results if r["winner"] is None)
    unmapped: Dict[str, int] = {}
    for r in results:
        for k, v in r.get("unmapped_kinds", {}).items():
            unmapped[k] = unmapped.get(k, 0) + v
    return {
        "spec": spec,
        "opponent": opponent,
        "opponent_class": resolve_opponent(opponent).__name__,
        "catanatron": CATANATRON_VERSION,
        "seed": seed,
        "games": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "avg_vp": sum(r["our_vp"] for r in results) / n if n else 0.0,
        "avg_opp_vp": sum(opp_avg) / n if n else 0.0,
        "avg_best_opp_vp": sum(opp_best) / n if n else 0.0,
        "avg_turns": sum(r["turns"] for r in results) / n if n else 0.0,
        "avg_duration": sum(r["duration"] for r in results) / n if n else 0.0,
        "truncated": truncated,
        "by_seat": {s: {"wins": w, "games": g} for s, (w, g) in by_seat.items()},
        "stats": stats,
        "unmapped_kinds": unmapped,
        "results": results,
    }


def print_summary(s: Dict[str, object], wall: float) -> None:
    n = max(1, int(s["games"]))
    st = s["stats"]
    print(f'catanbot "{s["spec"]}" vs 3 x {s["opponent_class"]}: {s["games"]} games, seat rotation, seed {s["seed"]}'
          f' (catanatron {s.get("catanatron", CATANATRON_VERSION)})')
    print(f'  wins        : {s["wins"]}/{s["games"]} = {100.0 * s["win_rate"]:.1f}%   (a random seat wins 25%)')
    print(f'  avg VP      : catanbot {s["avg_vp"]:.2f} | opponents {s["avg_opp_vp"]:.2f} (best opponent {s["avg_best_opp_vp"]:.2f})')
    seats = ", ".join(f'seat{k} {v["wins"]}/{v["games"]}' for k, v in s["by_seat"].items())
    print(f"  by seat     : {seats}")
    print(f'  turns/game  : {s["avg_turns"]:.1f} (catanatron num_turns; {s["truncated"]} game(s) hit the {TURNS_LIMIT}-turn cap)')
    print(f'  time/game   : {s["avg_duration"]:.2f} s ({st["search_time"] / n:.2f} s in search, '
          f'{st["searched"] / n:.1f} searched + {st["trivial"] / n:.1f} trivial decisions per game); wall {wall:.1f} s')
    extra = ""
    if st.get("pending_discard"):
        extra += f', {int(st["pending_discard"])} planned discard cards'
    if st.get("trade_prompts"):
        extra += f', {int(st["trade_prompts"])} trade prompts declined'
    print(f'  adapter     : {int(st["errors"])} errors, {int(st["fallback"])} fallbacks, '
          f'{int(st["unmapped_top"])} unmapped top actions, {int(st["observe_errors"])} observe errors, '
          f'{st["observed"] / n:.0f} observed actions/game{extra}'
          + (f'; unmapped top actions: {s["unmapped_kinds"]}' if s["unmapped_kinds"] else ""))
    print("  markdown    : | `%s` | %s | %d | %d | %.0f%% | %.2f | %.2f | %.1f | %.2f |" % (
        s["spec"], s["opponent_class"], s["games"], s["wins"], 100.0 * s["win_rate"], s["avg_vp"],
        s["avg_opp_vp"], s["avg_turns"], s["avg_duration"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=20, help="number of games (default 20)")
    ap.add_argument("--opponent", default="vp",
                    help="opponent preset (vp, weighted, random, vf, ab, value, alphabeta, sameturn, playouts, mcts), "
                         "an import path module:Class, or a comma-separated list (default vp)")
    ap.add_argument("--ladder", choices=sorted(LADDERS), default=None,
                    help="run a preset ladder of opponents (controls | standins | strong | full); "
                         "opponents the running catanatron does not ship are skipped")
    ap.add_argument("--list-opponents", action="store_true",
                    help="print every preset with its class and whether it is available on the running catanatron")
    ap.add_argument("--spec", default=DEFAULT_SPEC, help=f'catanbot bot spec (default "{DEFAULT_SPEC}")')
    ap.add_argument("--seed", type=int, default=0, help="base seed (default 0)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    ap.add_argument("--vps-to-win", type=int, default=10, help="victory points needed (default 10)")
    ap.add_argument("--discard-limit", type=int, default=7,
                    help="catanatron discard limit on a 7 (default 7; on 3.2.1 only the first discarder honours it, "
                         "see docs/BENCHMARKS.md limitation 12)")
    ap.add_argument("--json", default=None, help="write the full results to this JSON file")
    ap.add_argument("--verbose", action="store_true", help="print one line per game")
    args = ap.parse_args(argv)

    if args.list_opponents:
        print_opponents()
        return 0

    opponents = LADDERS[args.ladder] if args.ladder else [o.strip() for o in args.opponent.split(",") if o.strip()]
    single = args.ladder is None and len(opponents) == 1
    summaries: List[Dict[str, object]] = []
    for opponent in opponents:
        try:
            cls = resolve_opponent(opponent)
        except OpponentUnavailable as ex:
            if single:
                print(ex.message, file=sys.stderr)
                return 2
            print(f"skipping {opponent}: {ex.message}")
            continue
        print(f"== catanbot [{args.spec}] vs 3x {cls.__name__} ({opponent}), {args.games} games, "
              f"catanatron {CATANATRON_VERSION}")
        jobs = [(g, args.seed, args.spec, opponent, args.vps_to_win, args.discard_limit) for g in range(args.games)]
        results: List[Dict[str, object]] = []
        t0 = time.perf_counter()

        def report(r: Dict[str, object]) -> None:
            results.append(r)
            if args.verbose:
                print(f'  game {r["game"]:3d} seat {r["seat"]} seed {r["seed"]}: '
                      f'{"WIN " if r["won"] else "loss"} vp={r["vps"]} turns={r["turns"]} {r["duration"]:.1f}s',
                      flush=True)

        if args.workers > 1 and len(jobs) > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=args.workers) as pool:
                for r in pool.imap_unordered(run_one, jobs):
                    report(r)
        else:
            for job in jobs:
                report(run_one(job))
        results.sort(key=lambda r: r["game"])
        wall = time.perf_counter() - t0
        summary = summarize(results, args.spec, opponent, args.seed)
        summary["wall_time"] = wall
        print_summary(summary, wall)
        summaries.append(summary)
    if len(summaries) > 1:
        print(f"\n== ladder summary (win rate of catanbot in 4-player games; 25% = seat baseline; "
              f"catanatron {CATANATRON_VERSION})")
        for sm in summaries:
            print(f"  {sm['opponent_class']:<26} {sm['win_rate'] * 100:5.1f}%  avg VP {sm['avg_vp']:.2f} vs {sm['avg_opp_vp']:.2f}  ({sm['games']} games)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summaries if len(summaries) > 1 else (summaries[0] if summaries else {}), fh, indent=1, default=str)
        print(f"  json        : {args.json}")
    if not summaries:
        print(f"no opponent could be played on catanatron {CATANATRON_VERSION} (see --list-opponents)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
