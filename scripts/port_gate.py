#!/usr/bin/env python3
"""Ports mechanism gate: the pre-registered PASS rule over the gate cells' paired games (docs/PRIORITY_PLAN.md
step 4, test queue rows p12-p16; docs/QUEUE.md "Ports gate").  Read-only: it only reads the rows' JSONL files.

    python3 scripts/port_gate.py --dir runs/queue1 --json runs/queue1/ports_gate.json

Each gate cell is a queue row with one candidate against the shared default arm (300 paired seeds vs 3.2.1's
ValueFunction, full information, trades off; the Python evaluator for F1 / F2 / F1 mode 3, the C++ evaluator for the
flow cell).  Per cell, over the seeds whose candidate and default games both finished, the paired differences
(candidate - default) of

* ``cards_saved`` = 4 x cards received from the bank - cards given (the cards our ports saved over trading
  everything at 4:1; from each game's ``mech`` record: ``4 bank_trades - bank_cards_given``),
* ``settlements_built`` (post-setup settlements),
* ``won`` (a sanity readout only; the gate is not a win-rate test).

**PASS (pre-registered):** cards saved up by at least 1.1 a game (diff >= 1.1) AND settlements a game not lower by
0.15 or more (diff > -0.15: a drop of exactly 0.15 fails), with at least 90 % of the cell's seeds paired.  ``best`` = the passing cell with the largest gain
in cards saved (only it gets the 2,000-seed row).  ``f1_overshoot`` (the trigger of the milder F1 mode 3 cell): the
F1 cell raised cards saved by more than 2 paired se but failed the settlement guard.

The JSON carries ``cells`` (per cell: pairs, mean diffs and se, pass), ``any_pass``, ``best``, ``best_is`` (per cell)
and ``f1_overshoot``; scripts/run_queue.py command rows read one key each (``verdict_from``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS_MIN = 1.1          # cards saved a game, candidate - default (about 3 se at 300 seeds; our whole T1 gap)
SETTLE_MIN = -0.15       # settlements a game, candidate - default: PASS needs diff > SETTLE_MIN (strictly)
MIN_SHARE = 0.9          # share of the cell's seeds that must be paired
CELLS = {"f1": "ports_gate_f1@vf", "f2": "ports_gate_f2@vf", "flow": "ports_gate_flow@vf",
         "f1m3": "ports_gate_f1_mode3@vf"}


def _load_ablate():
    name = "_ablate_catanatron_for_port_gate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "ablate_catanatron.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mean_se(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    n = len(xs)
    if not n:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def cards_saved(rec: Dict[str, Any]) -> Optional[float]:
    m = rec.get("mech") or {}
    if "cards_saved" in m and m["cards_saved"] is not None:
        return float(m["cards_saved"])
    if m.get("bank_trades") is None or m.get("bank_cards_given") is None:
        return None
    return 4.0 * float(m["bank_trades"]) - float(m["bank_cards_given"])


def cell_pairs(path: str, row: str, base: int, count: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """``(candidate, default)`` game records of ``row``'s latest run over seeds ``base .. base + count - 1``
    (both ok, same code: ``ablate_catanatron.Index.pair``)."""
    AB = _load_ablate()
    idx, _bad = AB.load_index(path)
    runs = [r for r in idx.runs.values() if r.get("exp") == row]
    if not runs:
        return []
    run = max(runs, key=lambda r: r.get("t", 0))
    out = []
    for s in range(base, base + count):
        c, d = idx.pair(run, s)
        if c is not None and d is not None and c.get("status") == "ok" and d.get("status") == "ok":
            out.append((c, d))
    return out


def cell_stats(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], count: int) -> Dict[str, Any]:
    """Paired means / se of the gate metrics and the pre-registered PASS."""
    dc, ds, dw = [], [], []
    for c, d in pairs:
        a, b = cards_saved(c), cards_saved(d)
        if a is not None and b is not None:
            dc.append(a - b)
        mc, md = c.get("mech") or {}, d.get("mech") or {}
        if mc.get("settlements_built") is not None and md.get("settlements_built") is not None:
            ds.append(float(mc["settlements_built"]) - float(md["settlements_built"]))
        dw.append(float(int(bool(c.get("won"))) - int(bool(d.get("won")))))
    cm, cse = _mean_se(dc)
    sm, sse = _mean_se(ds)
    wm, wse = _mean_se(dw)
    complete = len(dc) >= MIN_SHARE * count and len(ds) >= MIN_SHARE * count
    ok = bool(complete and cm is not None and sm is not None and cm >= CARDS_MIN and sm > SETTLE_MIN)
    raised = bool(complete and cm is not None and cse is not None and cse > 0 and cm > 2.0 * cse)
    return {"pairs": len(pairs), "count": count, "complete": complete,
            "cards_saved": cm, "cards_saved_se": cse, "settlements": sm, "settlements_se": sse,
            "win": wm, "win_se": wse, "pass": ok, "raised_cards": raised,
            "settle_guard_failed": bool(complete and sm is not None and sm <= SETTLE_MIN)}


def evaluate(directory: str, cells: Optional[Dict[str, str]] = None, base: int = 0, count: int = 300
             ) -> Dict[str, Any]:
    cells = dict(CELLS if cells is None else cells)
    res: Dict[str, Any] = {"rule": {"cards_saved_min": CARDS_MIN, "settlements_min": SETTLE_MIN,
                                    "min_share": MIN_SHARE}, "cells": {}}
    for cell, row in cells.items():
        path = os.path.join(directory, f"{row}.jsonl")
        st = cell_stats(cell_pairs(path, row, base, count), count) if os.path.exists(path) else \
            dict(cell_stats([], count), missing=True)
        st["row"] = row
        res["cells"][cell] = st
    passing = [(st["cards_saved"], cell) for cell, st in res["cells"].items() if st["pass"]]
    best = max(passing)[1] if passing else None
    res["any_pass"] = bool(passing)
    res["best"] = best
    res["best_is"] = {cell: cell == best for cell in cells}
    f1 = res["cells"].get("f1") or {}
    res["f1_overshoot"] = bool(f1.get("raised_cards") and f1.get("settle_guard_failed"))
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the queue directory holding <row>.jsonl")
    ap.add_argument("--json", help="write the result here")
    ap.add_argument("--cell", action="append", default=[], metavar="CELL=ROW",
                    help="gate cells (default: f1, f2, flow, f1m3 = the ports_gate_* rows)")
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--count", type=int, default=300)
    args = ap.parse_args(argv)
    cells = dict(x.split("=", 1) for x in args.cell) if args.cell else None
    res = evaluate(args.dir, cells, args.base, args.count)
    for cell, st in res["cells"].items():
        fmt = (lambda x: "n/a" if x is None else f"{x:+.2f}")
        print(f"{cell:5} {st['row']:28} pairs {st['pairs']:4d}  cards saved {fmt(st['cards_saved'])} +- "
              f"{fmt(st['cards_saved_se'])}  settlements {fmt(st['settlements'])}  win {fmt(st['win'])}  "
              f"-> {'PASS' if st['pass'] else 'fail'}")
    print(f"any pass: {res['any_pass']}; best: {res['best']}; F1 overshoot (mode 3 trigger): {res['f1_overshoot']}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        tmp = args.json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(tmp, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
