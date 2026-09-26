#!/usr/bin/env python3
"""2^k factorial tests against Catanatron: main effects and interactions of strategy terms (docs/TUNING.md).

    /home/user/venv_cat33/bin/python scripts/factorial_catanatron.py \\
        --factorial danger.danger_multiplier=off,danger.steal_factor=off --opponent value --seeds 1000 \\
        --workers 2 --out runs/fx/dm_x_sf_value.jsonl --reuse 'runs/campaign1/*.jsonl'
    python3 scripts/factorial_catanatron.py --report --out runs/fx/dm_x_sf_value.jsonl     # statistics only
    ... --plan                                                                               # runs and cost only

The self-play counterpart is ``scripts/ablate.py --factorial``.  Here every combination of the factors
(each at its test value or at its registry default; 2 to 4 factors, a bare flag name = off) is one
candidate arm of ``scripts/ablate_catanatron.py`` against ONE shared base arm (every factor at its
default), so each seed is played once per cell on the same board, dice and seat, 1 catanbot seat vs 3
copies of the opponent.  Everything that plays and stores games is ablate_catanatron's, unchanged: the
append-only JSONL, ``PYTHONHASHSEED=0``, the fork pool with crash isolation and ``--game-timeout``, resume
by rerunning, and ``--reuse`` of base (default-arm) games already played by a campaign with the same arm
key and code.  This script only declares the runs (the design is kept in each run line's ``value``) and
adds the factorial analysis (``catanbot/factorial.py``): per seed the win (0/1) and the final VP of every
cell, paired over the seeds where every cell and the base finished ok with one code version.

Cost: ``seeds x 2^k`` games (``seeds x (2^k - 1)`` when the base games are reused); the planned total is
printed before anything is played.  ``ablate_catanatron.py --report`` on the same file shows each cell
as an ordinary candidate-vs-base comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from catanbot import factorial as F  # noqa: E402
from catanbot import tuning  # noqa: E402


def _load(name: str, path: str):
    """A script as a module, registered in ``sys.modules`` so worker pools can pickle its functions."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AB = _load("_ablate_catanatron_for_factorial", os.path.join(ROOT, "scripts", "ablate_catanatron.py"))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def build_runs(args) -> Tuple[List[Dict[str, Any]], bool, List[F.Factor]]:
    """One ablate_catanatron run per non-base cell, all with the same base arm; the Python evaluator flag."""
    try:
        factors = F.parse_factors(args.factorial)
    except (ValueError, KeyError) as ex:
        raise SystemExit(f"error: --factorial: {ex}")
    ts = [tuning.find(n) for n, _ in factors]
    base = args.base_spec or (tuning.DEFAULT_DEEP_SPEC if any(t.requires_depth >= 2 for t in ts)
                              else tuning.DEFAULT_SEARCH_SPEC)
    base = AB._load_ablate().factorial_spec(ts, base)
    common = AB.parse_overrides(args.set)
    clash = sorted(set(common) & {n for n, _ in factors})
    if clash:
        raise SystemExit(f"error: {clash} both in --set and in --factorial")
    adapter = AB.parse_kv(args.adapter_opt)
    adapter["trades"] = args.trades
    if args.info != "full":          # as ablate_catanatron.build_runs: only non-default information options
        adapter["info"] = args.info
        adapter["info_samples"] = int(args.info_samples)
        if args.discards_public:
            adapter["discards_public"] = True
    fact = [[n, json.loads(json.dumps(v))] for n, v in factors]
    dflt = AB.make_arm("def", base, common, adapter, "base (every factor at its default)")
    runs = []
    for c in F.cells(len(factors))[1:]:
        cell = json.loads(json.dumps(F.cell_overrides(factors, c)))
        label = F.cell_label(factors, c)
        runs.append({"tunable": None, "value": {"factorial": fact, "cell": c}, "value_text": label,
                     "default_text": "base", "cand": AB.make_arm("cand", base, dict(common, **cell), adapter, label),
                     "def": dflt})
    need_python = any(tuning.find(n).needs_python_evaluator for n in list(common) + [n for n, _ in factors])
    return runs, need_python, factors


# ---------------------------------------------------------------------------
# Analysis (from the JSONL alone)
# ---------------------------------------------------------------------------
def factorial_groups(index) -> List[Tuple[List[List[Any]], Dict[int, Dict[str, Any]]]]:
    """The designs in a file: ``[(factors, {cell: run})]`` (runs sharing factors, experiment and base arm)."""
    groups: Dict[str, Tuple[List[List[Any]], Dict[int, Dict[str, Any]]]] = {}
    for run in index.runs.values():
        v = run.get("value")
        if not isinstance(v, dict) or "factorial" not in v:
            continue
        key = AB.sha([v["factorial"], run.get("exp"), run["def_key"]])
        groups.setdefault(key, (v["factorial"], {}))[1][int(v["cell"])] = run
    return list(groups.values())


def factorial_stats(index, factors: List[List[Any]], runs: Dict[int, Dict[str, Any]],
                    retry_errors: bool = False) -> Optional[Dict[str, Any]]:
    """Cells, main effects and interactions over the seeds where the base and every cell finished ok with
    one code version (None while a cell has not been declared)."""
    k = len(factors)
    if sorted(runs) != F.cells(k)[1:]:
        return None
    seeds = sorted(set().union(*(index.seeds_of(r) for r in runs.values())))
    won: Dict[int, List[float]] = {c: [] for c in F.cells(k)}
    vp: Dict[int, List[float]] = {c: [] for c in F.cells(k)}
    used, dropped = [], 0
    for s in seeds:
        pairs = {c: index.pair(r, s, retry_errors) for c, r in runs.items()}
        recs = [x for p in pairs.values() for x in p]
        if any(x is None or x.get("status") != "ok" for x in recs) or len({x.get("code") for x in recs}) != 1:
            dropped += 1
            continue
        base = pairs[1][1]
        for c in F.cells(k):
            rec = base if c == 0 else pairs[c][0]
            won[c].append(float(bool(rec["won"])))
            vp[c].append(float(rec["our_vp"]))
        used.append(s)
    out: Dict[str, Any] = {"factors": factors, "seeds": len(used), "dropped": dropped,
                           "seed_min": min(used, default=None), "seed_max": max(used, default=None)}
    if used:
        fs = [(n, v) for n, v in factors]
        out["win"] = F.analyse(fs, won, AB.MIN_VERDICT_PAIRS)
        out["vp"] = F.analyse(fs, vp, AB.MIN_VERDICT_PAIRS)
    return out


def print_factorials(index, retry_errors: bool = False, out=None) -> List[Dict[str, Any]]:
    out = out or sys.stdout
    results = []
    for factors, runs in factorial_groups(index):
        k = len(factors)
        names = ", ".join(f"{n}={tuning.find(n).format(v)}" for n, v in factors)
        res = factorial_stats(index, factors, runs, retry_errors)
        if res is None:
            print(f"factorial 2^{k} ({names}): only {len(runs)} of {2 ** k - 1} cells declared", file=out)
            continue
        results.append(res)
        run0 = next(iter(runs.values()))
        ctx = run0.get("ctx") or {}
        print(f"\nfactorial 2^{k} ({names}) vs 3x {ctx.get('opponent')} (catanatron {ctx.get('catanatron')}, "
              f"{ctx.get('evaluator')}), base spec {run0['def']['spec']}: {res['seeds']} seed(s) with every cell "
              f"and the base complete" + (f", {res['dropped']} incomplete / with an error skipped"
                                          if res["dropped"] else ""), file=out)
        if res["seeds"]:
            print("\n".join(F.format_report(res["win"], "seeds", True, "catanbot win rate (1 seat vs 3)")), file=out)
            print("\n".join(F.format_report(res["vp"], "seeds", False, "catanbot final VP")), file=out)
            if res["seeds"] < AB.MIN_VERDICT_PAIRS:
                print("  (smoke size: nothing here is significant)", file=out)
    return results


def report(path: str, json_path: Optional[str] = None, retry_errors: bool = False) -> List[Dict[str, Any]]:
    AB.print_report(path, retry_errors)
    index, _ = AB.load_index(path)
    res = print_factorials(index, retry_errors)
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(AB.json_safe(res), fh, indent=1)
        print(f"wrote {json_path}")
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def ensure_env(need_python_eval: bool) -> None:
    """``PYTHONHASHSEED=0`` (and ``CATANBOT_NO_ACCEL=1`` for a factor the Python static evaluator reads), set
    by re-executing this script (ablate_catanatron's rule)."""
    want = AB.required_env(need_python_eval)
    if all(os.environ.get(k) == v for k, v in want.items()):
        return
    print("re-executing with " + ", ".join(f"{k}={v}" for k, v in want.items() if os.environ.get(k) != v), flush=True)
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], dict(os.environ, **want))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("design")
    g.add_argument("--factorial", metavar="NAME=V,NAME2=V2", help="2 to 4 factors (a bare flag name = off)")
    g.add_argument("--base-spec", help="bot spec of every arm (default search:depth=1,beam=4,expand=8,"
                                       "evaluator=heuristic; depth 2 for knobs only the depth-2 search reads)")
    g.add_argument("--set", action="append", metavar="NAME=VALUE", help="tunable override on EVERY arm")
    g.add_argument("--adapter-opt", action="append", metavar="KEY=VALUE", help="CatanbotPlayer option (every arm)")
    g.add_argument("--trades", default="off", help="domestic trading as in ablate_catanatron.py (3.3 only)")
    g.add_argument("--info", choices=("full", "counted"), default="full", help="information mode of every arm")
    g.add_argument("--info-samples", type=int, default=4)
    g.add_argument("--discards-public", action="store_true")
    g.add_argument("--opponent", help="bench_catanatron preset (value / alphabeta on 3.3, vf / ab ... on 3.2.1)")
    g.add_argument("--opponent-params", metavar="KEY=VAL,...")
    g.add_argument("--vps-to-win", type=int, default=10)
    g.add_argument("--discard-limit", type=int, default=7)
    g = p.add_argument_group("how many / where")
    g.add_argument("--seeds", type=int, default=20, help="seeds; each is played once per cell and once by the base")
    g.add_argument("--seed-base", type=int, default=0)
    g.add_argument("--workers", type=int, default=1, help="worker processes (0 = in this process)")
    g.add_argument("--out", required=True, help="results JSONL (append-only; rerunning resumes)")
    g.add_argument("--exp-name")
    g.add_argument("--reuse", action="append", metavar="GLOB", help="JSONL files whose base (default-arm) games "
                                                                    "with the same arm key and code are copied")
    g.add_argument("--max-minutes", type=float, default=None)
    g.add_argument("--game-timeout", type=int, default=1800)
    g.add_argument("--retry-errors", action="store_true")
    g.add_argument("--quiet", action="store_true")
    g = p.add_argument_group("reporting")
    g.add_argument("--report", action="store_true", help="statistics of --out, no games")
    g.add_argument("--report-json", help="also write the factorial statistics as JSON")
    g.add_argument("--plan", action="store_true", help="print the runs and the planned games, play nothing")
    p.add_argument("--no-reexec", action="store_true", help=argparse.SUPPRESS)
    # ablate_catanatron's synthetic games (tests): every cell gets the same effect
    p.add_argument("--fake-games", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--fake-crash-seeds", type=lambda t: [int(x) for x in t.split(",") if x], default=[],
                   help=argparse.SUPPRESS)
    p.add_argument("--fake-die-seeds", type=lambda t: [int(x) for x in t.split(",") if x], default=[],
                   help=argparse.SUPPRESS)
    p.add_argument("--fake-sleep", type=float, default=0.0, help=argparse.SUPPRESS)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report:
        report(args.out, args.report_json, args.retry_errors)
        return 0
    if not args.factorial or not args.opponent:
        print("error: --factorial and --opponent are required", file=sys.stderr)
        return 2
    runs, need_python, factors = build_runs(args)
    if not args.no_reexec and argv is None:
        ensure_env(need_python)
    elif os.environ.get("PYTHONHASHSEED") != AB.PINNED_HASHSEED:
        print("WARNING: PYTHONHASHSEED is not pinned: games are not reproducible across processes", file=sys.stderr)
    fake = args.fake_games is not None
    ctx = AB.make_context(args)
    if not fake:
        AB._validate_real(runs, ctx)
        from catanbot.bench import catanatron_adapter  # noqa: F401  (imported before the workers fork)
        from catanbot.agents import param_bot  # noqa: F401
        AB._bench()
    code = AB.code_fingerprint()
    exp = args.exp_name or f"factorial@{args.opponent}"
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    AB.finalize_runs(runs, ctx, exp, (args.seed_base, args.seeds))
    reuse = AB.ReusePool(args.reuse, [runs[0]["def_key"]], code, args.out) if args.reuse else None
    k = len(factors)
    print(f"factorial_catanatron: 2^{k} = {2 ** k} arms (base + {2 ** k - 1} cells) x {len(seeds)} seeds = "
          f"{2 ** k * len(seeds)} games planned ({(2 ** k - 1) * len(seeds)} if every base game is reused); "
          f"opponent {args.opponent} x3, catanatron {ctx['catanatron']}, {ctx['evaluator']}, "
          f"PYTHONHASHSEED={ctx['hashseed']}, code {code}" + (" [SYNTHETIC GAMES]" if fake else ""), flush=True)
    print(f"  base arm {runs[0]['def_key']}: {runs[0]['def']['spec']} {runs[0]['def']['overrides'] or ''}", flush=True)
    for r in runs:
        print(f"  cell {r['value']['cell']}: {r['cand']['label']} (arm {r['cand_key']}, run {r['run_key']})", flush=True)
    note = F.politics_note([n for n, _ in factors])
    if note:
        print(f"  note: {note}", flush=True)
    if args.plan:
        idx, _ = AB.load_index(args.out)
        for r in runs:
            done = sum(1 for s in seeds if idx.pair(r, s)[0] is not None)
            print(f"  {r['cand']['label']}: {done}/{len(seeds)} seeds complete in {args.out}")
        return 0
    camp = AB.Campaign(runs, args.out, ctx, code, seeds, exp, args.game_timeout, retry_errors=args.retry_errors,
                       reuse=reuse, quiet=args.quiet)
    print(f"  {camp.pending_jobs()} seed(s) need games; results -> {args.out}", flush=True)
    deadline = time.time() + 60.0 * args.max_minutes if args.max_minutes else None

    def on_term(signum, frame):
        raise AB.Stop(f"signal {signum}")

    old = signal.signal(signal.SIGTERM, on_term)
    t0 = time.time()
    try:
        status = AB.execute(camp, args.workers, deadline)
    except (AB.Stop, KeyboardInterrupt) as ex:
        status = f"interrupted ({ex or 'SIGINT'})"
    finally:
        signal.signal(signal.SIGTERM, old)
    wall = time.time() - t0
    left = camp.pending_jobs()
    print(f"\n{status}: played {camp.played} game(s) ({camp.errors} error record(s), {camp.reused} base game(s) "
          f"reused) in {wall:.0f} s; {left} seed(s) still need games"
          + (" - rerun the same command to resume" if left else ""), flush=True)
    report(args.out, args.report_json, args.retry_errors)
    return 0 if status in ("done", "deadline") else 130


if __name__ == "__main__":
    raise SystemExit(main())
