#!/usr/bin/env python3
"""Evaluate the pre-registered 1v1 benchmark (``docs/BENCH_1V1_PROTOCOL.md``) from bench results.

Usage::

    python3 scripts/analyze_1v1.py --test H1=OUT/json/H1 --test H2=OUT/json/H2 [--markdown PATH] [--json PATH]

``PATH`` is what ``scripts/bench_catanatron.py --players 2 --json`` wrote: a file, a directory of
chunk files (``--game-range`` chunks, merged by game index) or a glob; a comma list is allowed.

Fixed by the protocol (edit nothing here after the first registered game):

* H1 (primary): catanbot vs catanatron 3.3.0 ``AlphaBetaPlayer`` (defaults), 1v1, base seed 910501,
  games 0..399; H2 (secondary): vs ``ValueFunctionPlayer`` (defaults), base seed 910601, games 0..399.
  Both: spec ``search:depth=1,beam=4,expand=8,evaluator=heuristic``, full information, trades off,
  PYTHONHASHSEED 0, 10 VP, discard limit 7, catanbot in seat ``g % 2``.
* A game is won when its winner seat is catanbot's (recomputed here); turn-cap games and games
  that crashed twice are losses.
* Primary endpoint: H1's win rate against 0.5, exact one-sided binomial test (p = P(X >= wins)
  under Bin(400, 1/2)), alpha 0.05.  H2 is tested the same way only if H1 rejects (fixed-sequence
  gatekeeping, family-wise error 0.05); otherwise H2 is reported descriptively.
* Intervals: central two-sided Clopper-Pearson, 95 % (and 99 % for reference).
* The HexMachina statement: H1's rate and 95 % interval next to the reported 54.1 % (8.2 VP) of
  HexMachina against AlphaBeta (AlphaBeta itself 51.0 %, 7.8 VP, in the same setup) - a
  side-by-side, not a test, because their format and sample size are unconfirmed.  The wording
  is chosen by where 54.1 % lies relative to our 95 % interval (below it / inside it / above it).
* The data must be the registered data: every field of the run metadata (format 1v1, 2 players,
  opponent preset and class, default parameters, engine 3.3.0, spec, seed, trades off, hash seed
  0, 10 VP, discard limit 7, information full) and every game (games exactly 0..399 once, per-game
  seed ``seed * 100003 + g + 1``, catanbot seat ``g % 2``, two seats RED / BLUE).  Anything else is
  reported as a deviation and the verdict says "not the registered data".

The exact statistics come from ``scripts/prove_strength.py`` (rational binomial tails, Clopper-
Pearson by bisection on log-space tails; tested against scipy there).
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import sys
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("prove_strength_1v1", os.path.join(_HERE, "prove_strength.py"))
PS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PS)

# ---------------------------------------------------------------------------
# The protocol (docs/BENCH_1V1_PROTOCOL.md, pre-registered 2026-09-26) - do not edit
# ---------------------------------------------------------------------------
PROTOCOL = "docs/BENCH_1V1_PROTOCOL.md"
SPEC = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
ENGINE = "3.3.0"
HASH_SEED = "0"
TRADES = "off"
VPS_TO_WIN = 10
DISCARD_LIMIT = 7
INFO = {"mode": "full"}
FORMAT = "1v1"
PLAYERS = 2
COLORS = ["RED", "BLUE"]
NULL = Fraction(1, 2)
ALPHA = Fraction(5, 100)
CONF = 0.95
#: HexMachina as reported in summaries of Belle et al., arXiv 2506.04651 (OpenReview V0Fb4pwhS4);
#: not verified against the paper, whose exact game format is unconfirmed.
HEXMACHINA = {"win_rate": 0.541, "avg_vp": 8.2, "alphabeta_win_rate": 0.510, "alphabeta_avg_vp": 7.8}


class TestDef(NamedTuple):
    opponent: str
    opponent_class: str
    seed: int
    games: int
    role: str


TESTS: Dict[str, TestDef] = {
    "H1": TestDef("alphabeta", "AlphaBetaPlayer", 910501, 400, "primary"),
    "H2": TestDef("value", "ValueFunctionPlayer", 910601, 400, "secondary"),
}
ERROR_KEYS = ("errors", "observe_errors", "fallback")


def game_seed(base_seed: int, g: int) -> int:
    return base_seed * 100003 + g + 1


# ---------------------------------------------------------------------------
# One test
# ---------------------------------------------------------------------------
def _meta_problems(t: TestDef, source: str, meta: Dict[str, object]) -> List[str]:
    if not meta:
        return [f"{source}: per-game records without run metadata (cannot be verified)"]
    want = {"format": FORMAT, "players": PLAYERS, "our_seats": 1, "null_win_rate": 0.5,
            "opponent": t.opponent, "opponent_class": t.opponent_class, "opponent_params": {},
            "catanatron": ENGINE, "spec": SPEC, "seed": t.seed, "trades": TRADES, "hash_seed": HASH_SEED,
            "vps_to_win": VPS_TO_WIN, "discard_limit": DISCARD_LIMIT, "info": INFO}
    out = []
    for k, v in want.items():
        if meta.get(k, "<missing>") != v:
            out.append(f"{os.path.basename(source)}: {k} = {meta.get(k, '<missing>')!r}, registered {v!r}")
    return out


def analyze_test(tid: str, chunks: Sequence) -> Dict[str, object]:
    """Statistics and registration checks of one test (``chunks`` from ``prove_strength.load_chunks``)."""
    t = TESTS[tid]
    problems: List[str] = []
    by_game: Dict[int, Dict[str, object]] = {}
    for source, meta, recs in chunks:
        problems.extend(_meta_problems(t, source, meta))
        for r in recs:
            g = int(r["game"])
            if g in by_game:
                problems.append(f"game {g} appears in two chunks")
                continue
            by_game[g] = r
    expected = range(t.games)
    missing = [g for g in expected if g not in by_game]
    extra = sorted(g for g in by_game if g >= t.games)
    if missing:
        problems.append(f"{t.games - len(missing)} of the {t.games} registered games (missing {PS._ranges(missing)})")
    if extra:
        problems.append(f"{len(extra)} game(s) outside 0..{t.games - 1} ({PS._ranges(extra)})")
    recs = [by_game[g] for g in expected if g in by_game]
    n = len(recs)
    wins = capped = crashed = attempts = unmapped = 0
    turns = 0
    our_vp = opp_vp = 0.0
    err = {k: 0 for k in ERROR_KEYS}
    by_seat = {0: [0, 0], 1: [0, 0]}
    for r in recs:
        g = int(r["game"])
        seat = g % 2
        if int(r.get("seed", -1)) != game_seed(t.seed, g):
            problems.append(f"game {g}: seed {r.get('seed')}, registered {game_seed(t.seed, g)}")
        if r.get("seat") != seat or list(r.get("our_seats") or []) != [seat]:
            problems.append(f"game {g}: catanbot seat {r.get('seat')} / {r.get('our_seats')}, registered {seat}")
        if list(r.get("colors") or []) != COLORS or len(r.get("vps") or []) != PLAYERS:
            problems.append(f"game {g}: seats {r.get('colors')}, registered {COLORS}")
        won = int(r.get("winner_seat", -1)) == seat and not r.get("crashed")
        wins += won
        by_seat[seat][1] += 1
        by_seat[seat][0] += won
        if r.get("crashed"):
            crashed += 1
        elif r.get("winner") is None:
            capped += 1
        attempts += int(r.get("crashes", 0))
        vps = r.get("vps") or [0, 0]
        our_vp += float(vps[seat]) if len(vps) == PLAYERS else 0.0
        opp_vp += float(vps[1 - seat]) if len(vps) == PLAYERS else 0.0
        turns += int(r.get("turns", 0))
        st = r.get("stats") or {}
        for k in ERROR_KEYS:
            err[k] += int(st.get(k, 0))
        unmapped += int(st.get("unmapped_top", 0))
    p_one = PS.binom_sf_exact(wins, n, NULL) if n else Fraction(1)
    ci95 = PS.clopper_pearson(wins, n, CONF)
    ci99 = PS.clopper_pearson(wins, n, 0.99)
    return {
        "test": tid, "role": t.role, "opponent": t.opponent, "opponent_class": t.opponent_class, "seed": t.seed,
        "registered_games": t.games, "games": n, "wins": wins, "win_rate": wins / n if n else 0.0,
        "p_one_sided": p_one, "ci95": ci95, "ci99": ci99,
        "avg_vp": our_vp / n if n else 0.0, "avg_opp_vp": opp_vp / n if n else 0.0,
        "avg_turns": turns / n if n else 0.0,
        "by_seat": {str(s): {"wins": w, "games": gm, "win_rate": w / gm if gm else 0.0}
                    for s, (w, gm) in by_seat.items()},
        "turn_cap_games": capped, "crashed_games": crashed, "crash_attempts": attempts,
        "adapter_errors": err["errors"] + err["observe_errors"], "fallbacks": err["fallback"],
        "unmapped_top": unmapped,
        "registered": not problems and n == t.games, "deviations": problems,
    }


# ---------------------------------------------------------------------------
# Verdict and the HexMachina statement
# ---------------------------------------------------------------------------
def hexmachina_position(ci95) -> str:
    """Where HexMachina's reported 54.1 % lies relative to our 95 % interval."""
    hm = HEXMACHINA["win_rate"]
    if ci95[0] > hm:
        return "above"      # our whole interval is above 54.1 %
    if ci95[1] < hm:
        return "below"
    return "inside"


def hexmachina_statement(a: Dict[str, object]) -> str:
    lo, hi = a["ci95"]
    rate = f'{100.0 * a["win_rate"]:.1f}%'
    ci = f"95% CI {100.0 * lo:.1f}-{100.0 * hi:.1f}%"
    pos = hexmachina_position(a["ci95"])
    head = (f'Against catanatron {ENGINE}\'s AlphaBetaPlayer (defaults) in 1v1 games our bot won {a["wins"]}/'
            f'{a["games"]} = {rate} ({ci}; average {a["avg_vp"]:.1f} VP vs {a["avg_opp_vp"]:.1f}). ')
    where = {
        "above": "HexMachina's reported 54.1% lies below our whole 95% interval: our rate is higher than their "
                 "reported point estimate.",
        "inside": "HexMachina's reported 54.1% lies inside our 95% interval: at this sample size our rate is not "
                  "distinguishable from their reported point estimate.",
        "below": "HexMachina's reported 54.1% lies above our whole 95% interval: our rate is lower than their "
                 "reported point estimate.",
    }[pos]
    caveat = (" HexMachina (Belle et al., arXiv 2506.04651) is reported at 54.1% (8.2 VP) against AlphaBeta, with "
              "AlphaBeta itself at 51.0% (7.8 VP) in the same setup; their exact game format, catanatron version, "
              "AlphaBeta settings and number of games are unconfirmed (the paper could not be read from our "
              "environment), and their uncertainty is unknown, so this is a side-by-side of two numbers, not a "
              "test of one against the other.")
    return head + where + caveat


def evaluate(analysis: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    h1 = analysis.get("H1")
    h2 = analysis.get("H2")
    out: Dict[str, object] = {"alpha": ALPHA}
    if h1 is None:
        out["primary"] = "no H1 results"
        return out
    rejects1 = h1["p_one_sided"] <= ALPHA
    out["primary_rejects"] = bool(rejects1)
    out["primary_registered"] = bool(h1["registered"])
    if not h1["registered"]:
        out["primary"] = ("NOT THE REGISTERED DATA (see the deviations): no confirmatory reading; "
                          f"descriptively p = {PS.fmt_p(h1['p_one_sided'])}")
    elif rejects1:
        out["primary"] = (f"REJECTED H0 (win rate <= 50%) at one-sided alpha 0.05: p = {PS.fmt_p(h1['p_one_sided'])}"
                          " - our bot beats catanatron's AlphaBetaPlayer head to head")
    else:
        out["primary"] = (f"H0 (win rate <= 50%) NOT rejected at one-sided alpha 0.05: p = "
                          f"{PS.fmt_p(h1['p_one_sided'])} - no evidence that our bot beats AlphaBeta head to head")
    out["hexmachina_position"] = hexmachina_position(h1["ci95"])
    out["hexmachina_statement"] = hexmachina_statement(h1)
    if h2 is not None:
        if not h2["registered"]:
            out["secondary"] = f"H2 not the registered data (descriptive p = {PS.fmt_p(h2['p_one_sided'])})"
        elif rejects1 and h1["registered"]:
            ok = h2["p_one_sided"] <= ALPHA
            out["secondary"] = (f"H2 (gated by H1) {'REJECTED' if ok else 'NOT rejected'} H0 (win rate <= 50%) "
                                f"at one-sided alpha 0.05: p = {PS.fmt_p(h2['p_one_sided'])}")
            out["secondary_rejects"] = bool(ok)
        else:
            out["secondary"] = (f"H2 descriptive only (H1 did not reject, so the gate is closed): p = "
                                f"{PS.fmt_p(h2['p_one_sided'])}")
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _line(a: Dict[str, object]) -> str:
    lo, hi = a["ci95"]
    lo99, hi99 = a["ci99"]
    s0, s1 = a["by_seat"]["0"], a["by_seat"]["1"]
    return (f'{a["test"]} ({a["role"]}) vs {a["opponent_class"]}: {a["wins"]}/{a["games"]} = '
            f'{100.0 * a["win_rate"]:.1f}%  95% CI {100.0 * lo:.1f}-{100.0 * hi:.1f}%  (99% {100.0 * lo99:.1f}-'
            f'{100.0 * hi99:.1f}%)  one-sided p vs 50% = {PS.fmt_p(a["p_one_sided"])}\n'
            f'    avg VP {a["avg_vp"]:.2f} vs {a["avg_opp_vp"]:.2f}; turns/game {a["avg_turns"]:.1f}; seat 0 (moves '
            f'first) {s0["wins"]}/{s0["games"]}, seat 1 {s1["wins"]}/{s1["games"]}; turn-cap games '
            f'{a["turn_cap_games"]}, crashed games {a["crashed_games"]} ({a["crash_attempts"]} crashed attempts); '
            f'adapter errors {a["adapter_errors"]}, fallbacks {a["fallbacks"]}, unmapped top {a["unmapped_top"]}\n'
            f'    registered data: {"yes" if a["registered"] else "NO"}'
            + "".join(f"\n      - {d}" for d in a["deviations"][:12])
            + (f"\n      ... {len(a['deviations']) - 12} more" if len(a["deviations"]) > 12 else ""))


def report_text(analysis: Dict[str, Dict[str, object]], verdict: Dict[str, object]) -> str:
    lines = [f"1v1 benchmark ({PROTOCOL}): catanbot [{SPEC}] vs catanatron {ENGINE}, head to head (null 50%)"]
    for tid in TESTS:
        if tid in analysis:
            lines.append(_line(analysis[tid]))
    lines.append(f"PRIMARY (H1): {verdict.get('primary')}")
    if verdict.get("secondary"):
        lines.append(f"SECONDARY: {verdict['secondary']}")
    if verdict.get("hexmachina_statement"):
        lines.append(f"HEXMACHINA COMPARISON: {verdict['hexmachina_statement']}")
    return "\n".join(lines)


def report_markdown(analysis: Dict[str, Dict[str, object]], verdict: Dict[str, object],
                    sources: Sequence[str]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [f"Analysis of {now} by `scripts/analyze_1v1.py` ({PROTOCOL}).", "",
           "| test | opponent | games | wins | win rate | 95% CI | one-sided p vs 50% | avg VP (ours / opp) | "
           "seat 0 / seat 1 wins | registered |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for tid in TESTS:
        a = analysis.get(tid)
        if a is None:
            continue
        lo, hi = a["ci95"]
        s0, s1 = a["by_seat"]["0"], a["by_seat"]["1"]
        out.append(f'| {tid} ({a["role"]}) | {a["opponent_class"]} | {a["games"]} | {a["wins"]} | '
                   f'{100.0 * a["win_rate"]:.1f}% | {100.0 * lo:.1f}-{100.0 * hi:.1f}% | {PS.fmt_p(a["p_one_sided"])} | '
                   f'{a["avg_vp"]:.2f} / {a["avg_opp_vp"]:.2f} | {s0["wins"]}/{s0["games"]}, {s1["wins"]}/'
                   f'{s1["games"]} | {"yes" if a["registered"] else "NO"} |')
    out += ["", f"**Primary (H1).** {verdict.get('primary')}", ""]
    if verdict.get("secondary"):
        out += [f"**Secondary (H2).** {verdict['secondary']}", ""]
    if verdict.get("hexmachina_statement"):
        out += [f"**Comparison with HexMachina.** {verdict['hexmachina_statement']}", ""]
    for tid in TESTS:
        a = analysis.get(tid)
        if a is not None and a["deviations"]:
            out.append(f"Deviations in {tid}: " + "; ".join(a["deviations"][:20]))
    out += ["", "Sources: " + ", ".join(f"`{s}`" for s in sources)]
    return "\n".join(out) + "\n"


def _jsonable(x):
    if isinstance(x, Fraction):
        return {"fraction": f"{x.numerator}/{x.denominator}", "float": PS.to_float(x), "text": PS.fmt_p(x)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test", action="append", default=[], metavar="ID=PATH",
                    help=f"results of a registered test ({', '.join(TESTS)}); PATH: file, directory, glob or comma list")
    ap.add_argument("--markdown", default=None, help="write the report as Markdown")
    ap.add_argument("--json", default=None, help="write the full analysis as JSON")
    args = ap.parse_args(argv)
    if not args.test:
        ap.error("give at least --test H1=PATH")
    analysis: Dict[str, Dict[str, object]] = {}
    sources: List[str] = []
    for item in args.test:
        tid, _, paths = item.partition("=")
        if tid not in TESTS or not paths:
            ap.error(f"--test {item!r}: want ID=PATH with ID one of {', '.join(TESTS)}")
        chunks = []
        for p in paths.split(","):
            try:
                chunks.extend(PS.load_chunks(p))
            except (FileNotFoundError, ValueError) as ex:
                print(f"{tid}: cannot read {p}: {ex}", file=sys.stderr)
                return 2
        if not chunks:
            print(f"{tid}: no results in {paths}", file=sys.stderr)
            return 2
        sources += [c[0] for c in chunks]
        analysis[tid] = analyze_test(tid, chunks)
    verdict = evaluate(analysis)
    print(report_text(analysis, verdict))
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(report_markdown(analysis, verdict, sources))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable({"protocol": PROTOCOL, "analysis": analysis, "verdict": verdict,
                                 "sources": sources}), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
