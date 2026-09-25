#!/usr/bin/env python3
"""Benchmark ``Searcher.search`` with and without the C++ extension, and tournaments between native / Python bots.

Usage::

    python3 scripts/bench_search.py [--positions 5] [--depths 1 2 3] [--evaluators heuristic net]
                                    [--modes python cpp native] [--python] [--native] [--time-limit 2.0]
                                    [--model PATH] [--repeat 1] [--beam 4] [--expand 8] [--seed 0]
    python3 scripts/bench_search.py --tournament --specs "search:depth=2,native=1" "search:depth=2,native=0" \\
                                    [--games 12] [--workers 2] [--players 3] [--max-turns 200] [--seed 0]

Times ``Searcher.search`` (depths 1-4, heuristic and value-net evaluators) on mid-game positions of 4-player
heuristic-bot games in up to three subprocess modes:

* ``python`` - ``CATANBOT_NO_ACCEL=1``: the pure-Python reference;
* ``cpp``    - the extension for features / heuristic / engine, the lookahead (``_future_values``) in Python
  (``CATANBOT_NO_NATIVE_SEARCH=1``);
* ``native`` - the extension including the native lookahead (opponents' turns, reduced our-turn search and
  leaf evaluation in C++; see docs/CPP.md "Native lookahead").

``--python`` / ``--native`` select just those modes (both = python and native).  Every mode searches the same
positions with the same seeds: ``python`` and ``cpp`` visit identical trees (bit-identical values); ``native``
simulates the opponents with every non-trade candidate and no proposals, so its node counts and moves can
differ (the differential tests pin the parity-mode equivalence).  ``--time-limit`` gives every search that
many seconds (use it for depth 4, where the Python path takes minutes per position).  ``--model`` loads a
trained ``ValueNet`` (default: a freshly initialised net, which costs the same to evaluate).  ``--inprocess``
toggles the switches in this process instead of spawning subprocesses.  The workers pin BLAS to one thread
(``--blas-threads``): the search evaluates batches of 10-100 rows, for which OpenBLAS's thread pool is pure
overhead and, on a shared machine, a large source of timing noise.

``--tournament`` plays games between bot specs (``selfplay.make_bot`` specs plus a ``native=0/1`` key that sets
``SearchConfig.native_future``), so a native and a Python search bot can sit at the same table; it prints win
rates with 95 % Wilson intervals, average VP and the average time per decision of every spec.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODES = ("python", "cpp", "native")


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
    """Wraps an evaluator and accumulates the time / batches / states spent in ``evaluate``.

    Only the Python-side batches are seen: with the native lookahead the leaves and the simulated opponents'
    trials are evaluated inside the extension, so the Searcher gets the wrapped evaluator itself there.
    """

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
# one mode (Python / C++ / native), run inside a worker process
# ---------------------------------------------------------------------------
def run_mode(args, mode: str) -> Dict:
    from catanbot import accel
    from catanbot.search import SearchConfig, Searcher

    positions = mid_game_positions(args.positions, args.seed)
    native = mode == "native"
    result: Dict = {"mode": mode, "accel": None, "native": None, "core": accel.core_file(),
                    "positions": [s.turn for s in positions], "runs": []}
    for kind in args.evaluators:
        ev = TimedEvaluator(make_evaluator(kind, args.model))
        # evaluator alone: one batch of every position from every seat (what a search leaf batch looks like)
        batch_s = [s for s in positions for _ in range(s.num_players)]
        batch_p = [i for s in positions for i in range(s.num_players)]
        ev.evaluate(batch_s, batch_p)  # warm up (accel.verify(), numpy)
        best = min(_timed(ev.inner.evaluate, batch_s, batch_p) for _ in range(5))
        result["runs"].append({"evaluator": kind, "depth": 0, "search_s": best, "eval_s": best, "states": len(batch_s),
                               "nodes": 0, "calls": 1, "per_position": [], "native": False})
        for depth in args.depths:
            cfg = SearchConfig(depth=depth, beam=args.beam, expand=args.expand, native_future=native,
                               time_limit=(args.time_limit or None))
            per_pos = []
            total = total_eval = 0.0
            total_nodes = total_states = total_calls = 0
            used_native = False
            for pi, s in enumerate(positions):
                best_t = None
                for _ in range(args.repeat):
                    ev.reset()
                    # The native path needs the evaluator itself (a wrapper has no C++ twin); eval% then covers
                    # only the Python-side (root) batches.
                    searcher = Searcher(ev.inner if native else ev, cfg)
                    used_native = used_native or searcher.native_active
                    t0 = time.perf_counter()
                    res = searcher.search(s, 0, random.Random(1))
                    dt = time.perf_counter() - t0
                    if best_t is None or dt < best_t[0]:
                        best_t = (dt, ev.seconds, searcher.nodes, ev.states, ev.calls, str(res[0].action) if res else "",
                                  float(res[0].value) if res else 0.0)
                dt, es, nodes, nstates, calls, action, value = best_t
                per_pos.append({"position": pi, "search_s": dt, "eval_s": es, "nodes": nodes, "states": nstates,
                                "calls": calls, "best": action, "value": value})
                total += dt
                total_eval += es
                total_nodes += nodes
                total_states += nstates
                total_calls += calls
            result["runs"].append({"evaluator": kind, "depth": depth, "search_s": total, "eval_s": total_eval,
                                   "nodes": total_nodes, "states": total_states, "calls": total_calls,
                                   "per_position": per_pos, "native": used_native})
    result["accel"] = bool(accel.AVAILABLE)
    result["native"] = bool(accel.native_search_available()) and native
    return result


def _timed(fn, *a) -> float:
    t0 = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def worker_args(args, mode: str) -> List[str]:
    out = [sys.executable, os.path.abspath(__file__), "--worker", mode, "--positions", str(args.positions), "--seed",
           str(args.seed), "--repeat", str(args.repeat), "--beam", str(args.beam), "--expand", str(args.expand),
           "--blas-threads", str(args.blas_threads), "--time-limit", str(args.time_limit),
           "--depths", *[str(d) for d in args.depths], "--evaluators", *args.evaluators]
    if args.model:
        out += ["--model", args.model]
    return out


def pin_blas_threads(n: int, env: Dict[str, str]) -> None:
    """Limit the BLAS thread pools (must happen before numpy is imported)."""
    if n > 0:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = str(n)


def mode_env(mode: str, env: Dict[str, str]) -> None:
    env["CATANBOT_NO_ACCEL"] = "1" if mode == "python" else ""
    env["CATANBOT_NO_NATIVE_SEARCH"] = "1" if mode == "cpp" else ""


def run_subprocess(args, mode: str) -> Dict:
    env = dict(os.environ)
    mode_env(mode, env)
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    pin_blas_threads(args.blas_threads, env)
    proc = subprocess.run(worker_args(args, mode), env=env, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"worker failed (mode={mode})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_inprocess(args, mode: str) -> Dict:
    from catanbot import accel
    if mode == "python":
        accel.AVAILABLE = False
    else:
        if accel.load_core() is None:
            raise SystemExit("catanbot_core is not built (scripts/build_cpp.sh)")
        accel._core = accel.load_core()
        accel._verified = False
        accel.AVAILABLE = True
        os.environ[accel.NATIVE_SEARCH_ENV] = "1" if mode == "cpp" else ""
    return run_mode(args, mode)


def print_report(args, results: Dict[str, Dict]) -> None:
    modes = [m for m in MODES if m in results]
    first = results[modes[0]]
    print(f"Searcher.search on {args.positions} mid-game positions (4 players, turns {first['positions']}), "
          f"beam={args.beam}, expand={args.expand}, best of {args.repeat}, BLAS threads "
          f"{args.blas_threads or 'default'}" + (f", time limit {args.time_limit} s" if args.time_limit else ""))
    for m in modes:
        r = results[m]
        print(f"{m:<7}: accel.AVAILABLE={r['accel']} native lookahead={r['native']}"
              + (f"  ({r['core']})" if m != "python" else ""))
    if "cpp" in results and not results["cpp"]["accel"]:
        print("WARNING: the accelerated run did not use the extension (build it with scripts/build_cpp.sh)")
    if "native" in results and not results["native"]["native"]:
        print("WARNING: the native run did not use the native lookahead (rebuild with scripts/build_cpp.sh)")
    head = f"{'evaluator':<10} {'depth':>5} |"
    for m in modes:
        head += f" {m + ' [s]':>11} {'eval%':>6} |"
    if len(modes) > 1:
        head += f" {'speedup':>18} |"
    head += f" {'nodes ' + '/'.join(modes):>22}"
    print(head)
    print("-" * len(head))
    n_runs = len(first["runs"])
    for k in range(n_runs):
        rows = [results[m]["runs"][k] for m in modes]
        depth = rows[0]["depth"]
        label = "eval only" if depth == 0 else str(depth)
        line = f"{rows[0]['evaluator']:<10} {label:>5} |"
        for r in rows:
            share = 100.0 * r["eval_s"] / r["search_s"] if r["search_s"] else 0.0
            share_txt = f"{share:5.0f}%" if not (r.get("native") and depth) else "   -  "
            line += f" {r['search_s']:>11.3f} {share_txt:>6} |"
        if len(modes) > 1:
            base = rows[0]["search_s"]
            speed = " ".join(f"{base / r['search_s']:.1f}x" if r["search_s"] else "inf" for r in rows[1:])
            line += f" {speed:>18} |"
        nodes = "/".join(str(r["nodes"]) for r in rows) if depth else "-"
        line += f" {nodes:>22}"
        print(line)
    if args.verbose:
        print()
        for k in range(n_runs):
            rows = [results[m]["runs"][k] for m in modes]
            if not rows[0]["depth"]:
                continue
            for pi in range(len(rows[0]["per_position"])):
                parts = [f"{m} {r['per_position'][pi]['search_s']:.3f}s ({r['per_position'][pi]['nodes']} nodes, "
                         f"best {r['per_position'][pi]['best']} v={r['per_position'][pi]['value']:.4f})"
                         for m, r in zip(modes, rows)]
                print(f"  {rows[0]['evaluator']:<9} depth {rows[0]['depth']} pos {pi}: " + " | ".join(parts))


# ---------------------------------------------------------------------------
# tournament: native and Python search bots at the same table
# ---------------------------------------------------------------------------
def make_tournament_bot(spec: str):
    """``selfplay.make_bot`` spec plus ``native=0/1`` (sets ``SearchConfig.native_future``); name = the spec."""
    from catanbot.selfplay import make_bot, parse_spec
    name, kw = parse_spec(spec)
    native = kw.pop("native", None)
    base = name + (":" + ",".join(f"{k}={v}" for k, v in kw.items()) if kw else "")
    bot = make_bot(base)
    if native is not None and hasattr(bot, "config"):
        bot.config.native_future = native.strip().lower() not in ("0", "false", "no", "")
    bot.name = spec
    return bot


class _Timed:
    """Wraps a bot's ``decide`` to accumulate its decision time."""

    def __init__(self, bot):
        self.bot = bot
        self.seconds = 0.0
        self.decisions = 0

    def __getattr__(self, item):
        return getattr(self.bot, item)

    def decide(self, state, legal, rng):
        t0 = time.perf_counter()
        a = self.bot.decide(state, legal, rng)
        self.seconds += time.perf_counter() - t0
        self.decisions += 1
        return a


def _play_job(job):
    seats, seed, max_turns = job
    from catanbot.selfplay import play_game
    bots = [_Timed(make_tournament_bot(s)) for s in seats]
    r = play_game(bots, rng=random.Random(seed), max_turns=max_turns, specs=list(seats))
    return {"seats": list(seats), "winner": r.winner, "vps": list(r.vps), "turns": r.turns, "seed": seed,
            "seconds": [b.seconds for b in bots], "decisions": [b.decisions for b in bots],
            "native": [bool(getattr(getattr(b.bot, "_searcher", None), "native_active", False)) for b in bots]}


def wilson(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run_tournament(args) -> int:
    import multiprocessing as mp
    specs = list(args.specs)
    if not specs:
        raise SystemExit("--tournament needs --specs")
    for s in specs:
        make_tournament_bot(s)  # validate before the workers start
    rng = random.Random(args.seed)
    jobs = []
    for g in range(args.games):
        pool = list(specs)
        rng.shuffle(pool)
        seats = pool[:args.players]
        while len(seats) < args.players:
            seats.append(rng.choice(specs))
        jobs.append((seats, args.seed * 100003 + g, args.max_turns))
    t0 = time.perf_counter()
    if args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as p:
            results = p.map(_play_job, jobs)
    else:
        results = [_play_job(j) for j in jobs]
    wall = time.perf_counter() - t0
    stats = {s: {"games": 0, "wins": 0, "vp": 0.0, "seconds": 0.0, "decisions": 0, "native": set()} for s in specs}
    for r in results:
        for i, s in enumerate(r["seats"]):
            st = stats[s]
            st["games"] += 1
            st["vp"] += r["vps"][i]
            st["seconds"] += r["seconds"][i]
            st["decisions"] += r["decisions"][i]
            st["native"].add(r["native"][i])
            if r["winner"] == i:
                st["wins"] += 1
    print(f"{args.games} games ({args.players} players, max {args.max_turns} turns, seed {args.seed}, "
          f"{args.workers} workers): avg {sum(r['turns'] for r in results) / len(results):.0f} turns, "
          f"{wall:.0f} s wall")
    print(f"{'bot':<36} {'seats':>5} {'win%':>6} {'95% CI':>14} {'avgVP':>6} {'s/decision':>10} {'native':>7}")
    for s, st in stats.items():
        g = max(1, st["games"])
        lo, hi = wilson(st["wins"], st["games"])
        nat = "/".join(sorted(str(x) for x in st["native"]))
        print(f"{s:<36} {st['games']:>5} {100.0 * st['wins'] / g:6.1f} {100 * lo:6.1f}-{100 * hi:5.1f} "
              f"{st['vp'] / g:6.2f} {st['seconds'] / max(1, st['decisions']):10.3f} {nat:>7}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1, default=list)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--positions", type=int, default=5, help="number of mid-game positions (default 5)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], help="search depths (default 1 2 3)")
    ap.add_argument("--evaluators", nargs="+", default=["heuristic", "net"], choices=["heuristic", "net"],
                    help="evaluators to time (default: heuristic net)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES),
                    help="which paths to time (default: python cpp native)")
    ap.add_argument("--python", action="store_true", help="time the pure-Python path (with --native: both)")
    ap.add_argument("--native", action="store_true", help="time the native-lookahead path (with --python: both)")
    ap.add_argument("--time-limit", type=float, default=0.0, help="SearchConfig.time_limit per search (0 = none)")
    ap.add_argument("--model", default="", help="value net .npz for the net evaluator (default: fresh ValueNet)")
    ap.add_argument("--repeat", type=int, default=1, help="repetitions per position, best time kept (default 1)")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--expand", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blas-threads", type=int, default=1,
                    help="BLAS threads for the value net (default 1; 0 = leave numpy's default)")
    ap.add_argument("--verbose", "-v", action="store_true", help="per-position lines")
    ap.add_argument("--inprocess", action="store_true", help="toggle the switches in this process instead of "
                                                             "spawning subprocesses")
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    # tournament
    ap.add_argument("--tournament", action="store_true", help="play games between --specs instead of timing searches")
    ap.add_argument("--specs", nargs="*", default=[], help="bot specs (make_bot specs + native=0/1)")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--players", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--json", default="", help="tournament: write the per-game results to this file")
    args = ap.parse_args(argv)
    if args.tournament:
        pin_blas_threads(args.blas_threads, os.environ)
        return run_tournament(args)
    if args.python or args.native:
        args.modes = [m for m in ("python", "native") if getattr(args, m)]
    if args.worker or args.inprocess:
        pin_blas_threads(args.blas_threads, os.environ)  # before catanbot / numpy are imported
    if args.worker:
        print(json.dumps(run_mode(args, args.worker)))
        return 0
    runner = run_inprocess if args.inprocess else run_subprocess
    results = {m: runner(args, m) for m in MODES if m in args.modes}
    print_report(args, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
