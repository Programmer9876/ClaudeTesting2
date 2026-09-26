#!/usr/bin/env python3
"""Evaluate the pre-registered multi-seat benchmark (``docs/BENCH_MULTI_PROTOCOL.md``) from bench results.

Usage::

    python3 scripts/analyze_multi.py --test M1=OUT/json/M1 --test M2=OUT/json/M2 --test M3=OUT/json/M3 \\
        [--markdown PATH] [--json PATH]

``PATH`` is what ``scripts/bench_catanatron.py --our-seats 3`` (M1, M2) or ``--our-seats 2
--mixed-opponents value,alphabeta`` (M3) wrote with ``--json``: a file, a directory of chunk files
(``--game-range`` chunks, merged by game index) or a glob; a comma list is allowed.

Fixed by the protocol (edit nothing here after the first registered game):

* M1: 3v1 (three catanbot seats, one opponent in seat ``g % 4``) against catanatron 3.3.0
  ``AlphaBetaPlayer`` (defaults), base seed 910701, games 0..399; M2: 3v1 against
  ``ValueFunctionPlayer`` (defaults), base seed 910801, games 0..399; M3: 2v2-mixed (catanbot in the
  seats ``combinations(range(4), 2)[g % 6]``, ``value`` and ``alphabeta`` in the other two seats in
  that turn order when ``(g // 6) % 2 == 0``, reversed otherwise), base seed 910901, games 0..399.
  All: spec ``search:depth=1,beam=4,expand=8,evaluator=heuristic``, full information, trades off,
  PYTHONHASHSEED 0, 10 VP, discard limit 7, colours RED, BLUE, ORANGE, WHITE in turn order.
* Wins are recomputed here from each game's winner seat and the registered seating.
* M1 / M2: H0 "the opponent wins at least 25 %" against H1 "less than 25 %".  The tested count is
  the number of games NOT won by one of our three seats (the opponent's wins plus turn-cap games and
  games that crashed twice: conservatively counted for the opponent); exact one-sided binomial test,
  lower tail: p = P(X <= count) under Bin(n, 1/4).
* M3: H0 "our two seats together win at most 50 %"; the count is the games won by either of our
  seats (turn-cap and crashed games are losses); exact one-sided binomial test, upper tail:
  p = P(X >= wins) under Bin(n, 1/2).
* Family: Holm-Bonferroni over M1-M3 at alpha 0.05, always with m = 3; a test that is missing or is
  not the registered data enters with p = 1 (it cannot be rejected, and it does not loosen the
  thresholds of the others).
* Intervals: central two-sided Clopper-Pearson, 95 %.
* Descriptive: per-seat rates, average VP, turns, turn-cap games, crashes, adapter errors.
* The data must be the registered data: every field of the run metadata and every game (games
  exactly 0..N-1 once, per-game seed ``seed * 100003 + g + 1``, the registered seating and lineup,
  four seats).  Anything else is reported as a deviation and the test's verdict says "not the
  registered data".

The exact statistics come from ``scripts/prove_strength.py`` (rational binomial tails, Clopper-
Pearson by bisection on log-space tails, Holm; tested against scipy there).
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import itertools
import json
import os
import sys
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("prove_strength_multi", os.path.join(_HERE, "prove_strength.py"))
PS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PS)

# ---------------------------------------------------------------------------
# The protocol (docs/BENCH_MULTI_PROTOCOL.md, pre-registered 2026-09-26) - do not edit
# ---------------------------------------------------------------------------
PROTOCOL = "docs/BENCH_MULTI_PROTOCOL.md"
SPEC = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
ENGINE = "3.3.0"
HASH_SEED = "0"
TRADES = "off"
VPS_TO_WIN = 10
DISCARD_LIMIT = 7
INFO = {"mode": "full"}
PLAYERS = 4
COLORS = ["RED", "BLUE", "ORANGE", "WHITE"]
ALPHA = Fraction(5, 100)
CONF = 0.95
CATANBOT = "catanbot"
#: The 6 catanbot seat pairs of 2v2 (turn order), as ``scripts/bench_catanatron.py`` ``ARRANGEMENTS_2V2``.
PAIRS: Tuple[Tuple[int, int], ...] = tuple(itertools.combinations(range(PLAYERS), 2))


class TestDef(NamedTuple):
    fmt: str                     # "3v1" or "2v2-mixed"
    opponent: str                # the bench's --opponent tag ("value,alphabeta" for the mixed run)
    opponent_class: str
    opponents: Tuple[str, ...]   # the --mixed-opponents presets in order (M3), () in 3v1
    seed: int
    games: int
    null: Fraction               # null value of the tested proportion
    tail: str                    # "lower" (M1, M2) or "upper" (M3)


TESTS: Dict[str, TestDef] = {
    "M1": TestDef("3v1", "alphabeta", "AlphaBetaPlayer", (), 910701, 400, Fraction(1, 4), "lower"),
    "M2": TestDef("3v1", "value", "ValueFunctionPlayer", (), 910801, 400, Fraction(1, 4), "lower"),
    "M3": TestDef("2v2-mixed", "value,alphabeta", "ValueFunctionPlayer+AlphaBetaPlayer", ("value", "alphabeta"),
                  910901, 400, Fraction(1, 2), "upper"),
}
ERROR_KEYS = ("errors", "observe_errors", "fallback")


def game_seed(base_seed: int, g: int) -> int:
    return base_seed * 100003 + g + 1


def registered_lineup(t: TestDef, g: int) -> List[str]:
    """Preset per seat (``catanbot`` for ours) of game ``g`` of test ``t``, in turn order."""
    if t.fmt == "3v1":
        return [t.opponent if s == g % PLAYERS else CATANBOT for s in range(PLAYERS)]
    pair = PAIRS[g % len(PAIRS)]
    names = list(t.opponents) if (g // len(PAIRS)) % 2 == 0 else list(t.opponents)[::-1]
    it = iter(names)
    return [CATANBOT if s in pair else next(it) for s in range(PLAYERS)]


def registered_seats(t: TestDef, g: int) -> List[int]:
    return [s for s, nm in enumerate(registered_lineup(t, g)) if nm == CATANBOT]


# ---------------------------------------------------------------------------
# One test
# ---------------------------------------------------------------------------
def _meta_problems(t: TestDef, source: str, meta: Dict[str, object]) -> List[str]:
    if not meta:
        return [f"{source}: per-game records without run metadata (cannot be verified)"]
    want: Dict[str, object] = {
        "format": t.fmt, "players": PLAYERS, "our_seats": 3 if t.fmt == "3v1" else 2,
        "null_win_rate": 0.75 if t.fmt == "3v1" else 0.5,
        "opponent": t.opponent, "opponent_class": t.opponent_class, "opponent_params": {},
        "catanatron": ENGINE, "spec": SPEC, "seed": t.seed, "trades": TRADES, "hash_seed": HASH_SEED,
        "vps_to_win": VPS_TO_WIN, "discard_limit": DISCARD_LIMIT, "info": INFO}
    if t.fmt == "3v1":
        want["opp_null_win_rate"] = 0.25
    else:
        want["opponents"] = list(t.opponents)
    got = dict(meta)
    got.setdefault("players", PLAYERS)       # the 4-player summaries carry no "players" key
    out = []
    for k, v in want.items():
        if got.get(k, "<missing>") != v:
            out.append(f"{os.path.basename(source)}: {k} = {got.get(k, '<missing>')!r}, registered {v!r}")
    return out


def _game_problems(t: TestDef, g: int, r: Dict[str, object]) -> List[str]:
    out = []
    if int(r.get("seed", -1)) != game_seed(t.seed, g):
        out.append(f"game {g}: seed {r.get('seed')}, registered {game_seed(t.seed, g)}")
    seats = registered_seats(t, g)
    if list(r.get("our_seats") or []) != seats:
        out.append(f"game {g}: catanbot seats {r.get('our_seats')}, registered {seats}")
    if r.get("format") != t.fmt:
        out.append(f"game {g}: format {r.get('format')!r}, registered {t.fmt!r}")
    if list(r.get("colors") or []) != COLORS or len(r.get("vps") or []) != PLAYERS:
        out.append(f"game {g}: seats {r.get('colors')}, registered {COLORS}")
    if t.fmt == "3v1":
        if r.get("opp_seat") != g % PLAYERS or r.get("opponent") != t.opponent:
            out.append(f"game {g}: opponent {r.get('opponent')!r} in seat {r.get('opp_seat')}, registered "
                       f"{t.opponent!r} in seat {g % PLAYERS}")
    elif list(r.get("lineup") or []) != registered_lineup(t, g):
        out.append(f"game {g}: lineup {r.get('lineup')}, registered {registered_lineup(t, g)}")
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
    names = [CATANBOT] + ([t.opponent] if t.fmt == "3v1" else list(t.opponents))
    wins = {nm: 0 for nm in names}                               # games won, per player (ours combined)
    seat_rows = {nm: {s: [0, 0] for s in range(PLAYERS)} for nm in names}   # wins by that seat / games in it
    vp_sum = {nm: 0.0 for nm in names}
    vp_cnt = {nm: 0 for nm in names}
    by_pattern: Dict[str, List[int]] = {}
    capped = crashed = attempts = unmapped = turns = 0
    err = {k: 0 for k in ERROR_KEYS}
    for r in recs:
        g = int(r["game"])
        problems.extend(_game_problems(t, g, r))
        lineup = registered_lineup(t, g)
        w = -1 if r.get("crashed") else int(r.get("winner_seat", -1))
        if w >= 0:
            wins[lineup[w]] += 1
        pattern = "".join("C" if nm == CATANBOT else "o" for nm in lineup)
        row = by_pattern.setdefault(pattern, [0, 0])
        row[1] += 1
        row[0] += w >= 0 and lineup[w] == CATANBOT
        vps = r.get("vps") or []
        for s, nm in enumerate(lineup):
            seat_rows[nm][s][1] += 1
            seat_rows[nm][s][0] += w == s
            if len(vps) == PLAYERS and not r.get("crashed"):
                vp_sum[nm] += float(vps[s])
                vp_cnt[nm] += 1
        if r.get("crashed"):
            crashed += 1
        elif r.get("winner") is None:
            capped += 1
        attempts += int(r.get("crashes", 0))
        turns += int(r.get("turns", 0))
        st = r.get("stats") or {}
        for k in ERROR_KEYS:
            err[k] += int(st.get(k, 0))
        unmapped += int(st.get("unmapped_top", 0))
    ours = wins[CATANBOT]
    no_winner = n - sum(wins.values())
    if t.tail == "lower":
        count = n - ours                   # opponent wins + no-winner games (conservatively the opponent's)
        p_one = PS.binom_cdf_exact(count, n, t.null) if n else Fraction(1)
    else:
        count = ours
        p_one = PS.binom_sf_exact(count, n, t.null) if n else Fraction(1)
    ci95 = PS.clopper_pearson(count, n, CONF)
    our_ci95 = PS.clopper_pearson(ours, n, CONF)
    return {
        "test": tid, "format": t.fmt, "opponent": t.opponent, "opponent_class": t.opponent_class,
        "opponents": list(t.opponents), "seed": t.seed, "registered_games": t.games, "games": n,
        "tail": t.tail, "null": t.null,
        "count": count, "rate": count / n if n else 0.0, "p_one_sided": p_one, "ci95": ci95,
        "our_wins": ours, "our_win_rate": ours / n if n else 0.0, "our_ci95": our_ci95,
        "wins": dict(wins), "win_rate": {nm: (w / n if n else 0.0) for nm, w in wins.items()},
        "no_winner": no_winner,
        "avg_vp": {nm: (vp_sum[nm] / vp_cnt[nm] if vp_cnt[nm] else 0.0) for nm in names},
        "by_seat": {nm: {str(s): {"wins": w, "games": gm, "win_rate": w / gm if gm else 0.0}
                         for s, (w, gm) in rows.items()} for nm, rows in seat_rows.items()},
        "by_pattern": {k: {"wins": w, "games": gm} for k, (w, gm) in sorted(by_pattern.items())},
        "avg_turns": turns / n if n else 0.0,
        "turn_cap_games": capped, "crashed_games": crashed, "crash_attempts": attempts,
        "adapter_errors": err["errors"] + err["observe_errors"], "fallbacks": err["fallback"],
        "unmapped_top": unmapped,
        "registered": not problems and n == t.games, "deviations": problems,
    }


# ---------------------------------------------------------------------------
# Verdict: Holm over M1-M3
# ---------------------------------------------------------------------------
def _h0(tid: str) -> str:
    t = TESTS[tid]
    if t.fmt == "3v1":
        return f"{t.opponent_class} wins >= 25% of the 3v1 games"
    return "our two seats together win <= 50% of the 2v2-mixed games"


def evaluate(analysis: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    pvals = {}
    for tid in TESTS:
        a = analysis.get(tid)
        pvals[tid] = a["p_one_sided"] if a is not None and a["registered"] else Fraction(1)
    h = PS.holm(pvals, ALPHA)
    out: Dict[str, object] = {"alpha": ALPHA, "holm": h, "tests": {}}
    for tid in TESTS:
        a = analysis.get(tid)
        if a is None:
            text = f"no {tid} results (enters the Holm family with p = 1)"
            rejected = False
        elif not a["registered"]:
            text = (f"NOT THE REGISTERED DATA (see the deviations): no confirmatory reading (enters the Holm "
                    f"family with p = 1); descriptively p = {PS.fmt_p(a['p_one_sided'])}")
            rejected = False
        else:
            rejected = bool(h["rejected"][tid])
            text = (f"H0 ({_h0(tid)}) {'REJECTED' if rejected else 'NOT rejected'} (Holm over M1-M3, alpha 0.05): "
                    f"p = {PS.fmt_p(a['p_one_sided'])}, Holm threshold {PS.fmt_p(h['threshold'][tid])}, "
                    f"Holm-adjusted p = {PS.fmt_p(h['adjusted'][tid])}")
        out["tests"][tid] = {"text": text, "rejected": rejected}
    missing = [tid for tid in TESTS if tid not in analysis or not analysis[tid]["registered"]]
    out["complete"] = not missing
    out["family"] = ("complete: M1-M3 are all the registered data" if not missing
                     else f"INCOMPLETE: {', '.join(missing)} missing or not the registered data")
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _ci(ci) -> str:
    return f"{100.0 * ci[0]:.1f}-{100.0 * ci[1]:.1f}%"


def _seats(a: Dict[str, object], nm: str) -> str:
    return ", ".join(f'seat {s} {v["wins"]}/{v["games"]}' for s, v in a["by_seat"][nm].items())


def _line(a: Dict[str, object]) -> str:
    n = a["games"]
    if a["format"] == "3v1":
        opp = a["opponent"]
        head = (f'{a["test"]} (3v1) vs 1 x {a["opponent_class"]}: games not won by our seats {a["count"]}/{n} = '
                f'{_pct(a["rate"])} (opponent wins {a["wins"][opp]}, no winner {a["no_winner"]})  95% CI '
                f'{_ci(a["ci95"])}  one-sided p vs 25% (lower tail) = {PS.fmt_p(a["p_one_sided"])}\n'
                f'    our three seats won {a["our_wins"]}/{n} = {_pct(a["our_win_rate"])} (95% CI '
                f'{_ci(a["our_ci95"])}; null 75%); avg VP per catanbot seat {a["avg_vp"][CATANBOT]:.2f}, opponent '
                f'{a["avg_vp"][opp]:.2f}\n'
                f'    opponent by seat: {_seats(a, opp)}\n'
                f'    catanbot by seat: {_seats(a, CATANBOT)}')
    else:
        per = " | ".join(f'{nm} {a["wins"][nm]}/{n} ({_pct(a["win_rate"][nm])}, avg VP {a["avg_vp"][nm]:.2f})'
                         for nm in [CATANBOT] + a["opponents"])
        head = (f'{a["test"]} (2v2-mixed) vs {" + ".join(a["opponents"])}: our two seats won {a["count"]}/{n} = '
                f'{_pct(a["rate"])}  95% CI {_ci(a["ci95"])}  one-sided p vs 50% (upper tail) = '
                f'{PS.fmt_p(a["p_one_sided"])}\n'
                f'    wins: {per} | no winner {a["no_winner"]}\n'
                f'    by pattern: ' + ", ".join(f'{k} {v["wins"]}/{v["games"]}' for k, v in a["by_pattern"].items())
                + "".join(f"\n    {nm} by seat: {_seats(a, nm)}" for nm in [CATANBOT] + a["opponents"]))
    return (head + f'\n    turns/game {a["avg_turns"]:.1f}; turn-cap games {a["turn_cap_games"]}, crashed games '
            f'{a["crashed_games"]} ({a["crash_attempts"]} crashed attempts); adapter errors {a["adapter_errors"]}, '
            f'fallbacks {a["fallbacks"]}, unmapped top {a["unmapped_top"]}\n'
            f'    registered data: {"yes" if a["registered"] else "NO"}'
            + "".join(f"\n      - {d}" for d in a["deviations"][:12])
            + (f"\n      ... {len(a['deviations']) - 12} more" if len(a["deviations"]) > 12 else ""))


NOTE = ("Three copies of our bot share one strategy and never cooperate or trade (trades off): M1 / M2 measure "
        "the lone opponent against a table of copies, not teamwork.")


def report_text(analysis: Dict[str, Dict[str, object]], verdict: Dict[str, object]) -> str:
    lines = [f"Multi-seat benchmark ({PROTOCOL}): catanbot [{SPEC}] vs catanatron {ENGINE}"]
    for tid in TESTS:
        if tid in analysis:
            lines.append(_line(analysis[tid]))
    for tid in TESTS:
        lines.append(f"{tid}: {verdict['tests'][tid]['text']}")
    lines.append(f"FAMILY (Holm over M1-M3, alpha 0.05): {verdict['family']}")
    lines.append(f"NOTE: {NOTE}")
    return "\n".join(lines)


def report_markdown(analysis: Dict[str, Dict[str, object]], verdict: Dict[str, object],
                    sources: Sequence[str]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [f"Analysis of {now} by `scripts/analyze_multi.py` ({PROTOCOL}).", "",
           "| test | format | opponent(s) | games | tested count | rate | 95% CI | null | one-sided p | Holm | "
           "our seats' wins | opponent wins | avg VP (ours / opp) | registered |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tid in TESTS:
        a = analysis.get(tid)
        if a is None:
            continue
        v = verdict["tests"][tid]
        opp_names = a["opponents"] or [a["opponent"]]
        opp_w = ", ".join(f'{nm} {a["wins"][nm]}' for nm in opp_names)
        opp_vp = "/".join(f'{a["avg_vp"][nm]:.2f}' for nm in opp_names)
        what = "not won by us" if a["tail"] == "lower" else "won by us"
        out.append(f'| {tid} | {a["format"]} | {a["opponent_class"]} | {a["games"]} | {a["count"]} ({what}) | '
                   f'{_pct(a["rate"])} | {_ci(a["ci95"])} | {PS.to_float(a["null"]):.0%} '
                   f'({"lower" if a["tail"] == "lower" else "upper"} tail) | {PS.fmt_p(a["p_one_sided"])} | '
                   f'{"rejected" if v["rejected"] else "not rejected"} | {a["our_wins"]} | {opp_w} | '
                   f'{a["avg_vp"][CATANBOT]:.2f} / {opp_vp} | {"yes" if a["registered"] else "NO"} |')
    out.append("")
    for tid in TESTS:
        out += [f"**{tid}.** {verdict['tests'][tid]['text']}", ""]
    out += [f"**Family.** {verdict['family']}", "", f"*{NOTE}*", ""]
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
        ap.error("give at least one --test ID=PATH")
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
