#!/usr/bin/env python3
"""Benchmark ``Searcher.search`` with and without the C++ extension.

Usage::

    python3 scripts/bench_search.py [--positions 5] [--depths 1 2] [--evaluators heuristic net]
                                    [--model PATH] [--repeat 1] [--beam 4] [--expand 8] [--seed 0]

Times ``Searcher.search`` (depth 1 and 2, heuristic and value-net evaluators)
on mid-game positions of 4-player heuristic-bot games, once in a subprocess
with ``CATANBOT_NO_ACCEL=1`` (pure Python) and once accelerated
(``catanbot_core``: C++ feature extraction for the net, C++ static values +
softmax for the heuristic), and prints per configuration the total search
time, the share spent inside ``evaluator.evaluate`` and the speedup.  Both
runs search the same positions with the same seeds, so the node counts should
match (values are bit-identical, so the trees are the same).

``--model`` loads a trained ``ValueNet`` (default: a freshly initialised net,
which costs the same to evaluate).  ``--inprocess`` toggles
``catanbot.accel.AVAILABLE`` at run time instead of spawning subprocesses.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------
def mid_game_positions(count: int, seed: int, players: int = 4):
    """``count`` deterministic mid-game positions (turn >= 40, main phase, player 0 to move with >= 3 cards)."""
    from catanbot import engine as E
    from catanbot.agents.heuristic_bot import HeuristicBot
    from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, new_game

    out = []
    k = 0
    while len(out) < count:
        rng = random.Random(seed + 1000 * k)
        k += 1
        s = new_game(players, rng=rng)
        bots = [HeuristicBot() for _ in range(players)]
        steps = 0
        while s.phase != PHASE_GAME_OVER and steps < 6000:
            if s.turn >= 40 and s.phase == PHASE_MAIN and s.current == 0 and s.players[0].total_resources >= 3:
                break
            i = E.acting_player(s)
            s = E.apply_inplace(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
            steps += 1
        if s.phase != PHASE_GAME_OVER:
            out.append(s)
    return out


class TimedEvaluator:
    """Wraps an evaluator and accumulates the time / batches / states spent in ``evaluate``."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.seconds = 0.0
        self.calls = 0
        self.states = 0

    def evaluate(self, states, players):
        t0 = time.perf_counter()
        out = self.inner.evaluate(states, players)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.states += len(states)
        return out

    def reset(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self.states = 0


def make_evaluator(kind: str, model: str):
    from catanbot.heuristic import HeuristicEvaluator
    if kind == "heuristic":
        return HeuristicEvaluator()
    if kind == "net":
        from catanbot.model import ValueNet
        return ValueNet.load(model) if model else ValueNet(seed=0)
    raise ValueError(f"unknown evaluator {kind!r} (heuristic | net)")


# ---------------------------------------------------------------------------
# one mode (Python or C++), run inside a worker process
# ---------------------------------------------------------------------------
def run_mode(args) -> Dict:
    from catanbot import accel
    from catanbot.search import SearchConfig, Searcher

    positions = mid_game_positions(args.positions, args.seed)
    result: Dict = {"accel": None, "core": accel.core_file(), "positions": [s.turn for s in positions], "runs": []}
    for kind in args.evaluators:
        ev = TimedEvaluator(make_evaluator(kind, args.model))
        # evaluator alone: one batch of every position from every seat (what a search leaf batch looks like)
        batch_s = [s for s in positions for _ in range(s.num_players)]
        batch_p = [i for s in positions for i in range(s.num_players)]
        ev.evaluate(batch_s, batch_p)  # warm up (accel.verify(), numpy)
        best = min(_timed(ev.inner.evaluate, batch_s, batch_p) for _ in range(5))
        result["runs"].append({"evaluator": kind, "depth": 0, "search_s": best, "eval_s": best, "states": len(batch_s),
                               "nodes": 0, "calls": 1, "per_position": []})
        for depth in args.depths:
            cfg = SearchConfig(depth=depth, beam=args.beam, expand=args.expand)
            per_pos = []
            total = 0.0
            total_eval = 0.0
            total_nodes = 0
            total_states = 0
            total_calls = 0
            for pi, s in enumerate(positions):
                best_t = None
                for _ in range(args.repeat):
                    ev.reset()
                    searcher = Searcher(ev, cfg)
                    t0 = time.perf_counter()
                    res = searcher.search(s, 0, random.Random(1))
                    dt = time.perf_counter() - t0
                    if best_t is None or dt < best_t[0]:
                        best_t = (dt, ev.seconds, searcher.nodes, ev.states, ev.calls, str(res[0].action) if res else "")
                dt, es, nodes, nstates, calls, action = best_t
                per_pos.append({"position": pi, "search_s": dt, "eval_s": es, "nodes": nodes, "states": nstates,
                                "calls": calls, "best": action})
                total += dt
                total_eval += es
                total_nodes += nodes
                total_states += nstates
                total_calls += calls
            result["runs"].append({"evaluator": kind, "depth": depth, "search_s": total, "eval_s": total_eval,
                                   "nodes": total_nodes, "states": total_states, "calls": total_calls,
                                   "per_position": per_pos})
    result["accel"] = bool(accel.AVAILABLE)
    return result


def _timed(fn, *a) -> float:
    t0 = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def worker_args(args) -> List[str]:
    out = [sys.executable, os.path.abspath(__file__), "--worker", "--positions", str(args.positions), "--seed",
           str(args.seed), "--repeat", str(args.repeat), "--beam", str(args.beam), "--expand", str(args.expand),
           "--depths", *[str(d) for d in args.depths], "--evaluators", *args.evaluators]
    if args.model:
        out += ["--model", args.model]
    return out


def run_subprocess(args, no_accel: bool) -> Dict:
    env = dict(os.environ)
    env["CATANBOT_NO_ACCEL"] = "1" if no_accel else ""
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(worker_args(args), env=env, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"worker failed (no_accel={no_accel})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_inprocess(args, no_accel: bool) -> Dict:
    from catanbot import accel
    if no_accel:
        accel.AVAILABLE = False
    else:
        if accel.load_core() is None:
            raise SystemExit("catanbot_core is not built (scripts/build_cpp.sh)")
        accel._core = accel.load_core()
        accel._verified = False
        accel.AVAILABLE = True
    return run_mode(args)


def print_report(args, py: Dict, cpp: Dict) -> None:
    print(f"Searcher.search on {args.positions} mid-game positions (4 players, turns {py['positions']}), "
          f"beam={args.beam}, expand={args.expand}, best of {args.repeat}")
    print(f"python : accel.AVAILABLE={py['accel']}")
    print(f"c++    : accel.AVAILABLE={cpp['accel']}  ({cpp['core']})")
    if not cpp["accel"]:
        print("WARNING: the accelerated run did not use the extension (build it with scripts/build_cpp.sh)")
    header = (f"{'evaluator':<10} {'depth':>5} | {'python [s]':>10} {'eval%':>6} | {'c++ [s]':>9} {'eval%':>6} | "
              f"{'speedup':>8} | {'nodes py/c++':>14} {'states':>8}")
    print(header)
    print("-" * len(header))
    for a, b in zip(py["runs"], cpp["runs"]):
        assert a["evaluator"] == b["evaluator"] and a["depth"] == b["depth"]
        label = "eval only" if a["depth"] == 0 else str(a["depth"])
        share_a = 100.0 * a["eval_s"] / a["search_s"] if a["search_s"] else 0.0
        share_b = 100.0 * b["eval_s"] / b["search_s"] if b["search_s"] else 0.0
        speed = a["search_s"] / b["search_s"] if b["search_s"] else float("inf")
        nodes = f"{a['nodes']}/{b['nodes']}" if a["depth"] else "-"
        print(f"{a['evaluator']:<10} {label:>5} | {a['search_s']:>10.3f} {share_a:>5.0f}% | {b['search_s']:>9.3f} "
              f"{share_b:>5.0f}% | {speed:>7.2f}x | {nodes:>14} {a['states']:>8}")
    if args.verbose:
        print()
        for a, b in zip(py["runs"], cpp["runs"]):
            for pa, pb in zip(a["per_position"], b["per_position"]):
                print(f"  {a['evaluator']:<9} depth {a['depth']} pos {pa['position']}: python {pa['search_s']:.3f}s "
                      f"(eval {pa['eval_s']:.3f}s, {pa['nodes']} nodes) c++ {pb['search_s']:.3f}s "
                      f"(eval {pb['eval_s']:.3f}s, {pb['nodes']} nodes) best {pa['best']} / {pb['best']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--positions", type=int, default=5, help="number of mid-game positions (default 5)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2], help="search depths (default 1 2)")
    ap.add_argument("--evaluators", nargs="+", default=["heuristic", "net"], choices=["heuristic", "net"],
                    help="evaluators to time (default: heuristic net)")
    ap.add_argument("--model", default="", help="value net .npz for the net evaluator (default: fresh ValueNet)")
    ap.add_argument("--repeat", type=int, default=1, help="repetitions per position, best time kept (default 1)")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--expand", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", "-v", action="store_true", help="per-position lines")
    ap.add_argument("--inprocess", action="store_true", help="toggle accel.AVAILABLE in this process instead of "
                                                             "spawning CATANBOT_NO_ACCEL subprocesses")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        print(json.dumps(run_mode(args)))
        return 0
    runner = run_inprocess if args.inprocess else run_subprocess
    py = runner(args, no_accel=True)
    cpp = runner(args, no_accel=False)
    print_report(args, py, cpp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
