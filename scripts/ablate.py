#!/usr/bin/env python3
"""Paired ablations of catanbot's strategy constants ("is this alpha strategy worth it?").

    python3 scripts/ablate.py --list
    python3 scripts/ablate.py --tunable danger.TURNS_HALF --values 2,4.5 --games 40 --workers 2 --seed 1 --players 4
    python3 scripts/ablate.py --tunable danger.danger_multiplier --flag-off --games 40 --json out.json
    python3 scripts/ablate.py --tunable heuristic.EXPOSURE_WEIGHT --values 0,0.5 \\
        --base-spec search:depth=1,beam=4,expand=8,evaluator=heuristic --games 20
    python3 scripts/ablate.py --sweep-all --games 4 --workers 2 --out docs/ABLATIONS.md
    python3 scripts/ablate.py --tunable search.counters --counters --games 1000 \
        --base-spec search:depth=1,beam=4,expand=8,evaluator=heuristic      # counter-offer rules on
    python3 scripts/ablate.py --factorial danger.danger_multiplier=off,danger.steal_factor=off --games 600 \
        --workers 3 --json dm_x_sf.json                                      # 2x2: main effects + interaction

``--counters`` plays every game under the counter-offer rules variant (``GameState.allow_counters``, off in
the base game): responders may counter, and the report adds the counters made / taken per side.

``--factorial NAME=V1,NAME2=V2[,...]`` (2 to 4 factors; a bare flag name means "off") plays every
combination of the factors at their test values (each against the all-default base, 2 seats vs 2, on
the SAME seeds and seat patterns for every cell) and reports each cell, each main effect and each
interaction (``AB - A - B + base``) with a paired 95% interval (``catanbot/factorial.py``, docs/TUNING.md).

Every game seats the SAME bot spec on both sides: the tunable at a candidate value
('C' seats) and at its default ('D' seats), 2-vs-2 in 4-player games and a rotating
2-vs-1 in 3-player games, with the seat pattern rotated per game and identical game
seeds (board, dice, steals) for every candidate value.  For each candidate the script
reports wins per side, the paired win-rate difference with its standard error and 95%
interval, average VP per side, mean and p95 decision time per side and "value per ms"
(win-rate delta / extra ms per decision).

A tunable that lives in ``heuristic.static_value`` needs the Python evaluator (the C++
port reads no Python constants): the script then re-executes itself with
``CATANBOT_NO_ACCEL=1`` set before ``catanbot`` is imported, so *both* sides run the
Python evaluator.  The evaluator mode that ran is printed and stored in the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from catanbot import tuning  # noqa: E402  (reads CATANBOT_NO_ACCEL at import; see maybe_reexec)
from catanbot.tuning import DEFAULT_CHEAP_SPEC, DEFAULT_DEEP_SPEC, DEFAULT_SEARCH_SPEC, Tunable, spec_depth  # noqa: E402

RESULTS_START = "<!-- ablate:results:start -->"
RESULTS_END = "<!-- ablate:results:end -->"


# ---------------------------------------------------------------------------
# Evaluator mode
# ---------------------------------------------------------------------------
def maybe_reexec(need_python: bool) -> None:
    """Re-run this script with ``CATANBOT_NO_ACCEL=1`` when the tunable needs the Python evaluator.

    The variable must be set before ``catanbot`` is imported (``catanbot.accel`` reads it at import
    time), hence the re-exec rather than a run-time flag; worker processes are forked from the
    re-executed process and inherit it.
    """
    from catanbot import accel
    if need_python and accel.AVAILABLE and not accel.disabled_by_env():
        print("re-executing with CATANBOT_NO_ACCEL=1 (the tunable is read by the Python static evaluator)", flush=True)
        env = dict(os.environ, CATANBOT_NO_ACCEL="1")
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def pct(x: float) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100.0 * x:+.1f}pp"


def parse_values(t: Tunable, text: Optional[str], flag_off: bool) -> List[Any]:
    if t.kind == "flag":
        if flag_off or not text:
            return [False]
        return [t.parse(v) for v in text.split(",") if v.strip()]
    if not text:
        return list(t.candidates)
    sep = ";" if t.parse is tuning._parse_vector else ","
    return [t.parse(v) for v in text.split(sep) if v.strip()]


def resolve_spec(t: Tunable, base_spec: Optional[str]) -> str:
    if base_spec:
        if t.requires_search and not base_spec.startswith("search"):
            raise SystemExit(f"error: {t.name} only affects the search bot (evaluator / search knob); "
                             f"use a search base spec such as {DEFAULT_SEARCH_SPEC}")
        if spec_depth(base_spec) < t.requires_depth:
            raise SystemExit(f"error: {t.name} is only read by the searcher at depth >= {t.requires_depth} "
                             f"(the opponents' turns are not simulated at depth {spec_depth(base_spec)}); both sides "
                             f"would do identical work.  Use a base spec such as {DEFAULT_DEEP_SPEC}")
        return base_spec
    if t.requires_depth >= 2:
        return DEFAULT_DEEP_SPEC
    return DEFAULT_SEARCH_SPEC if t.requires_search else DEFAULT_CHEAP_SPEC


def json_safe(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# One tunable
# ---------------------------------------------------------------------------
def run_tunable(t: Tunable, values: List[Any], base_spec: str, games: int, workers: int, seed: int,
                players: int, max_turns: int, quiet: bool = False, allow_counters: bool = False) -> Dict[str, Any]:
    mode = tuning.evaluator_mode()
    print(f"tunable {t.name} ({t.kind}): default {t.format(t.default)}; candidates "
          f"{', '.join(t.format(v) for v in values)}")
    print(f"base spec {base_spec}; {players} players; {games} paired games per candidate; seed {seed}; "
          f"workers {workers}" + ("; counter-offer rules ON" if allow_counters else ""))
    print(f"evaluator mode: {mode}")
    if t.needs_python_evaluator and not mode.startswith("python"):
        print("WARNING: this tunable is read by the static evaluator but the C++ evaluator is active; "
              "the two sides differ only where Python code reads the constant", flush=True)
    rows = []
    worker_modes = set()
    t_start = time.time()
    for value in values:
        overrides = {t.name: value}
        label = t.format(value)

        def progress(k, n, r, label=label):
            if not quiet:
                print(f"  [{label}] game {k}/{n}: winner seat {r['winner']} ({r['pattern']}), vps {r['vps']}, "
                      f"{r['turns']} turns, {r['duration']:.1f}s", file=sys.stderr, flush=True)

        results = tuning.run_paired(base_spec, overrides, games, seed, players, workers=workers,
                                    max_turns=max_turns, progress=progress, allow_counters=allow_counters)
        st = tuning.paired_stats(results)
        st["value"] = value
        st["value_text"] = label
        st["verdict"] = tuning.verdict(st)
        rows.append(st)
        worker_modes.update(st["evaluator_modes"])
        if st["evaluator_modes"] != [mode]:
            print(f"WARNING: the games were played with evaluator mode(s) {st['evaluator_modes']} but this "
                  f"process reports {mode!r}", flush=True)
        print_row(t, st)
        if allow_counters:
            g = max(1, st["games"])
            print(f"      counter-offers per game: cand {st['counters_cand'] / g:.2f} made / "
                  f"{st['counters_taken_cand'] / g:.2f} taken, default {st['counters_def'] / g:.2f} made / "
                  f"{st['counters_taken_def'] / g:.2f} taken")
    print(f"evaluator mode in the game processes: {', '.join(sorted(worker_modes)) or 'unknown'}")
    return {
        "tunable": t.name, "kind": t.kind, "default": t.default, "default_text": t.format(t.default),
        "description": t.description, "base_spec": base_spec, "players": players, "games": games, "seed": seed,
        "workers": workers, "max_turns": max_turns, "evaluator_mode": mode, "allow_counters": bool(allow_counters),
        "worker_evaluator_modes": sorted(worker_modes),
        "needs_python_evaluator": t.needs_python_evaluator, "seconds": time.time() - t_start, "results": rows,
    }


def print_row(t: Tunable, st: Dict[str, Any]) -> None:
    lo, hi = st["ci95"]
    print(f"  value {st['value_text']:>14} vs default {t.format(t.default)}: {st['games']} games, "
          f"cand {st['cand_wins']}W/{st['cand_seats']} seats ({100 * st['win_rate_cand']:.1f}%), "
          f"default {st['def_wins']}W/{st['def_seats']} seats ({100 * st['win_rate_def']:.1f}%), draws {st['draws']}")
    print(f"      delta {pct(st['delta'])} +- {100 * st['se']:.1f}pp  95% CI [{pct(lo)}, {pct(hi)}]  "
          f"side wins {st['cand_wins']}-{st['def_wins']}; avg VP {st['avg_vp_cand']:.2f} vs {st['avg_vp_def']:.2f}")
    vpm = f"{st['value_per_ms']:+.4f}" if st["value_per_ms"] is not None else f"n/a ({st['cost']})"
    print(f"      ms/decision cand {fmt(st['ms_mean_cand'], 2)} (p95 {fmt(st['ms_p95_cand'], 2)}) vs default "
          f"{fmt(st['ms_mean_def'], 2)} (p95 {fmt(st['ms_p95_def'], 2)}); extra {fmt(st['extra_ms'], 2)} ms "
          f"({st['cost']}); value/ms {vpm}; {st['verdict']}")
    print(f"      override apply/restore overhead (excluded from the times above): cand "
          f"{fmt(st['overhead_ms_cand'], 3)} ms, default {fmt(st['overhead_ms_def'], 3)} ms per decision; "
          f"{st['decisions_cand']} / {st['decisions_def']} decisions")


# ---------------------------------------------------------------------------
# Listing and markdown
# ---------------------------------------------------------------------------
def print_registry() -> None:
    print(f"{'name':34} {'kind':6} {'default':16} {'candidates':34} pyeval search depth description")
    for t in tuning.TUNABLES.values():
        cands = ", ".join(t.format(c) for c in t.candidates)
        print(f"{t.name:34} {t.kind:6} {t.format(t.default):16} {cands:34} {str(t.needs_python_evaluator):6} "
              f"{str(t.requires_search):6} {t.requires_depth:<5} {t.description}")


def markdown_table(reports: List[Dict[str, Any]]) -> str:
    head = ("| tunable | kind | value | default | spec | mode | games | cand W / def W (seats) | delta win-rate +- SE | "
            "95% CI | avg VP c / d | ms/decision c / d (p95) | extra ms | value / ms | verdict |")
    sep = "|" + "---|" * 15
    lines = [head, sep]
    for rep in reports:
        for st in rep["results"]:
            lo, hi = st["ci95"]
            spec = rep["base_spec"].split(":")[0]
            vpm = f"{st['value_per_ms']:+.4f}" if st["value_per_ms"] is not None else f"n/a ({st['cost']})"
            lines.append(
                f"| {rep['tunable']} | {rep['kind']} | {st['value_text']} | {rep['default_text']} | {spec} | "
                f"{rep['evaluator_mode'].split(' ')[0]} | {st['games']} | "
                f"{st['cand_wins']} / {st['def_wins']} ({st['cand_seats']}/{st['def_seats']}) | "
                f"{pct(st['delta'])} +- {100 * st['se']:.1f}pp | [{pct(lo)}, {pct(hi)}] | "
                f"{st['avg_vp_cand']:.2f} / {st['avg_vp_def']:.2f} | "
                f"{fmt(st['ms_mean_cand'], 2)} / {fmt(st['ms_mean_def'], 2)} ({fmt(st['ms_p95_cand'], 1)} / "
                f"{fmt(st['ms_p95_def'], 1)}) | {fmt(st['extra_ms'], 2)} | {vpm} | {st['verdict']} |")
    return "\n".join(lines) + "\n"


def write_markdown(path: str, table: str, header: str) -> None:
    block = f"{RESULTS_START}\n{header}\n{table}{RESULTS_END}"
    if os.path.exists(path):
        text = open(path).read()
        if RESULTS_START in text and RESULTS_END in text:
            pattern = re.compile(re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END), re.S)
            text = pattern.sub(lambda m: block, text, count=1)
        else:
            text = text.rstrip("\n") + "\n\n## Sweep results\n\n" + block + "\n"
    else:
        text = "# Ablation sweep\n\n" + block + "\n"
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(args) -> int:
    names = [t.name for t in tuning.TUNABLES.values()]
    if args.kinds:
        kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
        names = [n for n in names if tuning.TUNABLES[n].kind in kinds]
    if args.only:
        wanted = [tuning.find(n.strip()).name for n in args.only.split(",") if n.strip()]
        names = [n for n in names if n in wanted]
    if not names:
        raise SystemExit("error: no tunables selected")
    print(f"sweep: {len(names)} tunables, {args.games} games per candidate, workers {args.workers}, "
          f"seed {args.seed}, {args.players} players")
    reports = []
    tmpdir = tempfile.mkdtemp(prefix="ablate-sweep-")
    for name in names:
        out = os.path.join(tmpdir, name.replace(".", "_") + ".json")
        cmd = [sys.executable, os.path.abspath(__file__), "--tunable", name, "--games", str(args.games),
               "--workers", str(args.workers), "--seed", str(args.seed), "--players", str(args.players),
               "--max-turns", str(args.max_turns), "--json", out, "--quiet"]
        if args.base_spec:
            cmd += ["--base-spec", args.base_spec]
        if args.max_candidates:
            cmd += ["--max-candidates", str(args.max_candidates)]
        if args.counters:
            cmd += ["--counters"]
        print(f"--- {name}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}): {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}")
            continue
        with open(out) as f:
            reports.append(json.load(f))
    header = (f"Sweep of {len(reports)} tunables: {args.games} paired games per candidate, seed {args.seed}, "
              f"{args.players} players, workers {args.workers}, run {time.strftime('%Y-%m-%d %H:%M')} "
              f"({'smoke run: tiny game counts on a loaded machine, nothing here is significant' if args.games < 30 else 'full run'}).\n")
    table = markdown_table(reports)
    print(header)
    print(table)
    if args.out:
        write_markdown(args.out, table, header)
        print(f"wrote {args.out}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(json_safe({"sweep": reports}), f, indent=1)
        print(f"wrote {args.json}")
    return 0


# ---------------------------------------------------------------------------
# Factorial (2^k) tests
# ---------------------------------------------------------------------------
def factorial_spec(ts: List[Tunable], base_spec: Optional[str]) -> str:
    """One base spec every factor acts in (the most demanding default, or ``base_spec`` checked per factor)."""
    if base_spec:
        for t in ts:
            resolve_spec(t, base_spec)
        return base_spec
    if any(t.requires_depth >= 2 for t in ts):
        return DEFAULT_DEEP_SPEC
    return DEFAULT_SEARCH_SPEC if any(t.requires_search for t in ts) else DEFAULT_CHEAP_SPEC


def _game_vp_diff(rec: Dict[str, Any]) -> float:
    """Mean final VP of the candidate seats minus that of the default seats in one game."""
    pat, vps = rec["pattern"], rec["vps"]
    c = [vps[i] for i, s in enumerate(pat) if s == "C"]
    d = [vps[i] for i, s in enumerate(pat) if s == "D"]
    return sum(c) / len(c) - sum(d) / len(d)


def run_factorial(args) -> int:
    from catanbot import factorial as F
    try:
        factors = F.parse_factors(args.factorial)
    except (ValueError, KeyError) as ex:
        print(f"error: --factorial: {ex}", file=sys.stderr)
        return 2
    ts = [tuning.find(n) for n, _ in factors]
    maybe_reexec(any(t.needs_python_evaluator for t in ts))
    base_spec = factorial_spec(ts, args.base_spec)
    k = len(factors)
    mode = tuning.evaluator_mode()
    workers = max(1, args.workers)
    print(f"factorial 2^{k}: " + ", ".join(f"{n}={tuning.find(n).format(v)} (default {tuning.find(n).format(tuning.find(n).default)})"
                                          for n, v in factors))
    print(f"base spec {base_spec}; {args.players} players; {args.games} paired games per cell x {2 ** k - 1} cells "
          f"(= {args.games * (2 ** k - 1)} games; the base cell is 0 by construction); seed {args.seed}; workers "
          f"{workers}" + ("; counter-offer rules ON" if args.counters else ""))
    print(f"evaluator mode: {mode}")
    note = F.politics_note([n for n, _ in factors])
    if note:
        print(f"note: {note}")
    if args.plan:
        for c in F.cells(k)[1:]:
            print(f"  cell {F.cell_label(factors, c)}: overrides {F.cell_overrides(factors, c)}")
        print("seat patterns: " + " ".join(p for p, _ in tuning.paired_jobs(min(args.games, 6), args.seed, args.players)))
        return 0
    t_start = time.time()
    cells: Dict[int, Dict[str, Any]] = {}
    for c in F.cells(k)[1:]:
        label = F.cell_label(factors, c)

        def progress(i, n, r, label=label):
            if not args.quiet:
                print(f"  [{label}] game {i}/{n}: winner seat {r['winner']} ({r['pattern']}), vps {r['vps']}, "
                      f"{r['turns']} turns, {r['duration']:.1f}s", file=sys.stderr, flush=True)

        results = tuning.run_paired(base_spec, F.cell_overrides(factors, c), args.games, args.seed, args.players,
                                    workers=workers, max_turns=args.max_turns, progress=progress,
                                    allow_counters=args.counters)
        st = tuning.paired_stats(results)
        st.update(cell=c, label=label, overrides=F.cell_overrides(factors, c))
        cells[c] = st
        lo, hi = st["ci95"]
        print(f"  cell {label}: {st['games']} games, cand {st['cand_wins']}W / default {st['def_wins']}W, delta "
              f"{pct(st['delta'])} +- {100 * st['se']:.1f}pp [{pct(lo)}, {pct(hi)}], avg VP {st['avg_vp_cand']:.2f} vs "
              f"{st['avg_vp_def']:.2f}, extra {fmt(st['extra_ms'], 2)} ms/decision", flush=True)
    seeds = [[r["seed"] for r in cells[c]["records"]] for c in cells]
    if any(s != seeds[0] for s in seeds):
        raise SystemExit("internal error: the cells were not played on the same seeds")
    win = F.analyse(factors, {c: [r["diff"] for r in st["records"]] for c, st in cells.items()})
    vp = F.analyse(factors, {c: [_game_vp_diff(r) for r in st["records"]] for c, st in cells.items()})
    print(f"\nfactorial 2^{k} on {win['units']} common games per cell ({base_spec}, {mode}):")
    print("\n".join(F.format_report(win, "games", True, "per-seat win rate, candidate seats minus default seats")))
    print("\n".join(F.format_report(vp, "games", False, "final VP, candidate seats minus default seats")))
    if win["units"] < 30:
        print("  (smoke size: nothing here is significant; an interaction needs ~4x the games of a main effect)")
    if args.json:
        report = {"factorial": [[n, v] for n, v in factors], "base_spec": base_spec, "players": args.players,
                  "games": args.games, "seed": args.seed, "workers": workers, "max_turns": args.max_turns,
                  "allow_counters": bool(args.counters), "evaluator_mode": mode, "seconds": time.time() - t_start,
                  "cells": [cells[c] for c in sorted(cells)], "win": win, "vp": vp}
        with open(args.json, "w") as f:
            json.dump(json_safe(report), f, indent=1)
        print(f"wrote {args.json}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tunable", help="registry name (danger.TURNS_HALF) or unique attribute (TURNS_HALF)")
    p.add_argument("--values", help="comma separated candidate values (default: the registry candidates); "
                                    "resource vectors as a/b/c/d/e separated by ';'; flags on/off")
    p.add_argument("--flag-off", action="store_true", help="flag tunables: test the flag switched off")
    p.add_argument("--games", type=int, default=8, help="paired games per candidate value")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--players", type=int, choices=(2, 3, 4), default=4)
    p.add_argument("--base-spec", help=f"bot spec for both sides (default {DEFAULT_CHEAP_SPEC}, or "
                                       f"{DEFAULT_SEARCH_SPEC} for tunables only the search bot reads)")
    p.add_argument("--max-turns", type=int, default=400)
    p.add_argument("--counters", action="store_true",
                   help="play every game under the counter-offer rules variant (GameState.allow_counters)")
    p.add_argument("--max-candidates", type=int, default=0, help="only the first N registry candidates")
    p.add_argument("--json", help="write the report to this JSON file")
    p.add_argument("--quiet", action="store_true", help="no per-game progress lines")
    p.add_argument("--plan", action="store_true", help="print the plan (spec, mode, candidates) without playing")
    p.add_argument("--list", action="store_true", help="print the registry and exit")
    p.add_argument("--factorial", metavar="NAME=V,NAME2=V2",
                   help="2^k factorial test (2-4 factors; a bare flag name = off): every combination vs the base on "
                        "the same seeds; main effects and interactions with paired 95%% CIs")
    p.add_argument("--sweep-all", action="store_true", help="run every tunable at its candidates; write --out")
    p.add_argument("--kinds", help="sweep: only these kinds (weight,flag,search)")
    p.add_argument("--only", help="sweep: only these tunable names (comma separated)")
    p.add_argument("--out", default=os.path.join("docs", "ABLATIONS.md"),
                   help="sweep: markdown file whose results block is replaced (default docs/ABLATIONS.md)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print_registry()
        return 0
    if args.sweep_all:
        return sweep(args)
    if args.factorial:
        if args.tunable:
            print("error: use either --tunable or --factorial", file=sys.stderr)
            return 2
        return run_factorial(args)
    if not args.tunable:
        build_parser().print_usage()
        print("error: --tunable, --factorial, --list or --sweep-all is required", file=sys.stderr)
        return 2
    try:
        t = tuning.find(args.tunable)
    except KeyError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    maybe_reexec(t.needs_python_evaluator)
    values = parse_values(t, args.values, args.flag_off)
    if args.max_candidates:
        values = values[:args.max_candidates]
    base_spec = resolve_spec(t, args.base_spec)
    if args.players == 2 and 2 not in tuning.SEAT_PATTERNS:
        raise SystemExit("2-player games are not supported")
    if args.plan:
        print(f"tunable {t.name} ({t.kind}): default {t.format(t.default)}; candidates "
              f"{', '.join(t.format(v) for v in values)}")
        print(f"base spec {base_spec}; {args.players} players; {args.games} paired games per candidate; seed {args.seed}"
              + ("; counter-offer rules ON" if args.counters else ""))
        print(f"evaluator mode: {tuning.evaluator_mode()}")
        print("seat patterns: " + " ".join(p for p, _ in tuning.paired_jobs(min(args.games, 6), args.seed, args.players)))
        return 0
    report = run_tunable(t, values, base_spec, args.games, max(1, args.workers), args.seed, args.players,
                         args.max_turns, quiet=args.quiet, allow_counters=args.counters)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(json_safe(report), f, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
