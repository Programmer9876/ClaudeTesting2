#!/usr/bin/env python3
"""Evaluate the pre-registered strength proof (``docs/PROOF_PROTOCOL.md``) from bench results.

Usage::

    python3 scripts/prove_strength.py --test T1=run/json/T1 --test T2=run/json/T2 ... [--markdown docs/PROOF.md]
    python3 scripts/prove_strength.py RESULTS ...            # test ids inferred from each file's metadata

``RESULTS`` / ``PATH`` are ``scripts/bench_catanatron.py --json`` files (a
summary with per-game ``results``, a list of summaries, or a plain list of
per-game records), ``.jsonl`` files of per-game records, directories (every
``*.json`` / ``*.jsonl`` below them) or globs; several chunks of one test are
merged by game index (bare per-game records carry no run metadata, so the
conformance check reports them as unverifiable).  ``--test ID=PATH`` (repeatable; ``PATH`` may be a
comma list) names the test; a positional file's test is inferred from its
opponent, format and catanatron version.

For every test (T1-T6, R1, R2) the script reports games, wins, win rate, the
exact one-sided binomial p-value against the protocol null (0.25 in 1v3,
0.5 for the 2v2 catanbot-win share), 95 % and 99 % Clopper-Pearson
intervals, the per-seat (1v3, with the per-seat one-sided p against 0.25) or
per-arrangement (2v2) breakdown, average VP of catanbot vs the opponents,
turn-cap games (counted as losses), adapter errors, illegal-action
fallbacks and crashes, and whether the data follows the protocol (opponent,
format, engine, seed and per-game seeds, seats, spec, trades off,
PYTHONHASHSEED=0, default rules, exactly games 0..N-1).  It then applies
Holm-Bonferroni over T1-T6 at family alpha 0.01 (claim 1) and 5.7e-7
(claim 2), the effect-size bounds (lower end of the 99 % Clopper-Pearson
interval >= 0.35 in T1-T3, >= 0.55 in T4-T6), seat robustness (every seat of
T1-T3: one-sided p < 0.05 against 0.25), the zero errors / fallbacks /
crashes condition over all proof games, and the R1 / R2 condition (two-sided
exact test against 0.25 has p > 0.01, or the rate is above 0.25), and prints
PASS / FAIL for both claims with the reason of every failed condition.
``--markdown PATH`` writes the same as a Markdown body for ``docs/PROOF.md``;
``--json PATH`` the full analysis.

Statistics are exact and pure Python: binomial tails and the two-sided test
(scipy's "minlike" rule) in rational arithmetic (``math.comb``, the nulls
are 1/4 and 1/2), Clopper-Pearson bounds by bisection on binomial tails
summed in log space from exact ``log(comb)``.  When scipy is installed its
values are computed alongside and the largest relative difference is
reported (the tests compare both).  The Clopper-Pearson intervals are the
usual central two-sided ones: "the lower 99 % bound" is the lower end of the
99 % interval (0.5 % in each tail), the conservative reading.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# The protocol (docs/PROOF_PROTOCOL.md, pre-registered 2026-09-25) - do not edit
# ---------------------------------------------------------------------------
PROTOCOL_SPEC = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
PROTOCOL_HASH_SEED = "0"
PROTOCOL_TRADES = "off"
PROTOCOL_VPS_TO_WIN = 10
PROTOCOL_DISCARD_LIMIT = 7
NUM_SEATS = 4
ARRANGEMENTS_2V2: Tuple[Tuple[int, int], ...] = tuple(combinations(range(NUM_SEATS), 2))


class TestDef(NamedTuple):
    tid: str
    opponent: str          # bench preset
    opponent_class: str
    fmt: str               # "1v3" | "2v2"
    games: int
    seed: int
    engine: str            # catanatron version (exact, as registered)


TESTS: Dict[str, TestDef] = {
    "T1": TestDef("T1", "value", "ValueFunctionPlayer", "1v3", 1000, 900001, "3.3.0"),
    "T2": TestDef("T2", "alphabeta", "AlphaBetaPlayer", "1v3", 400, 900001, "3.3.0"),
    "T3": TestDef("T3", "sameturn", "SameTurnAlphaBetaPlayer", "1v3", 400, 900001, "3.3.0"),
    "T4": TestDef("T4", "value", "ValueFunctionPlayer", "2v2", 1000, 900001, "3.3.0"),
    "T5": TestDef("T5", "alphabeta", "AlphaBetaPlayer", "2v2", 400, 900001, "3.3.0"),
    "T6": TestDef("T6", "sameturn", "SameTurnAlphaBetaPlayer", "2v2", 400, 900001, "3.3.0"),
    "R1": TestDef("R1", "vf", "ValueFunctionPlayer", "1v3", 1000, 900101, "3.2.1"),
    "R2": TestDef("R2", "ab", "AlphaBetaPlayer", "1v3", 400, 900101, "3.2.1"),
}
T_TESTS = ("T1", "T2", "T3", "T4", "T5", "T6")
R_TESTS = ("R1", "R2")
NULL: Dict[str, Fraction] = {"1v3": Fraction(1, 4), "2v2": Fraction(1, 2)}
CLAIM1_ALPHA = Fraction(1, 100)                 # family-wise, Holm over T1-T6
CLAIM2_ALPHA = Fraction(57, 10 ** 8)            # 5.7e-7, one-sided 5 sigma, Holm over T1-T6
EFFECT_CONF = 0.99
EFFECT_LOWER = {"1v3": 0.35, "2v2": 0.55}        # lower 99 % Clopper-Pearson bound
SEAT_ALPHA = Fraction(5, 100)                   # every seat of T1-T3, one-sided vs 0.25
R_ALPHA = Fraction(1, 100)                      # R1 / R2 two-sided vs 0.25
ERROR_KEYS = ("errors", "observe_errors", "fallback")


def game_seed(base_seed: int, g: int) -> int:
    """Per-game catanatron seed of ``scripts/bench_catanatron.py``."""
    return base_seed * 100003 + g + 1


def our_seats_for(g: int, fmt: str) -> Tuple[int, ...]:
    return (g % NUM_SEATS,) if fmt == "1v3" else ARRANGEMENTS_2V2[g % len(ARRANGEMENTS_2V2)]


def seat_pattern(seats: Sequence[int]) -> str:
    return "".join("C" if i in seats else "o" for i in range(NUM_SEATS))


# ---------------------------------------------------------------------------
# Exact binomial statistics (pure Python)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _terms(n: int, p: Fraction) -> Tuple[Tuple[int, ...], int]:
    """``(C(n,i) a^i b^(n-i) for i = 0..n, d^n)`` with ``p = a/d``: pmf(i) = term_i / d^n exactly."""
    a, d = p.numerator, p.denominator
    b = d - a
    return tuple(math.comb(n, i) * a ** i * b ** (n - i) for i in range(n + 1)), d ** n


def binom_pmf_exact(k: int, n: int, p: Fraction) -> Fraction:
    terms, den = _terms(n, Fraction(p))
    return Fraction(terms[k], den) if 0 <= k <= n else Fraction(0)


def binom_sf_exact(k: int, n: int, p: Fraction) -> Fraction:
    """P(X >= k), X ~ Bin(n, p), exactly (the one-sided p-value of k successes against H0: rate <= p)."""
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    terms, den = _terms(n, Fraction(p))
    return Fraction(sum(terms[k:]), den)


def binom_cdf_exact(k: int, n: int, p: Fraction) -> Fraction:
    """P(X <= k) exactly."""
    if k < 0:
        return Fraction(0)
    if k >= n:
        return Fraction(1)
    terms, den = _terms(n, Fraction(p))
    return Fraction(sum(terms[:k + 1]), den)


def binom_two_sided_exact(k: int, n: int, p: Fraction) -> Fraction:
    """Exact two-sided binomial p-value, "minlike" rule as in ``scipy.stats.binomtest`` / R's
    ``binom.test``: k's own tail plus the outcomes on the other side of the mode whose probability
    is at most pmf(k) (with scipy's relative tolerance 1e-7)."""
    p = Fraction(p)
    terms, den = _terms(n, p)
    if k * p.denominator == p.numerator * n:
        return Fraction(1)
    tk = terms[k]
    scale = 10 ** 7

    def small(t: int) -> bool:
        return t * scale <= tk * (scale + 1)

    np_ = p * n
    if k < np_:
        own = sum(terms[:k + 1])
        other = sum(t for t in terms[math.ceil(np_):] if small(t))
    else:
        own = sum(terms[k:])
        other = sum(t for t in terms[:math.floor(np_) + 1] if small(t))
    return min(Fraction(1), Fraction(own + other, den))


def log10_fraction(x: Fraction) -> float:
    """log10 of a positive Fraction without underflow (``-inf`` for 0)."""
    if x <= 0:
        return float("-inf")
    return math.log10(x.numerator) - math.log10(x.denominator)


def to_float(x: Fraction) -> float:
    """Correctly rounded float of a Fraction (0.0 when it underflows; use :func:`log10_fraction` then)."""
    return x.numerator / x.denominator


@lru_cache(maxsize=64)
def _log_comb(n: int) -> Tuple[float, ...]:
    return tuple(math.log(math.comb(n, i)) for i in range(n + 1))


def _logsumexp(xs: Iterable[float]) -> float:
    xs = list(xs)
    m = max(xs)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def log_tail_ge(k: int, n: int, p: float) -> float:
    """log P(X >= k) for a float p (exact log-binomial coefficients, log-space sum)."""
    if k <= 0:
        return 0.0
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return 0.0
    lc = _log_comb(n)
    lp, lq = math.log(p), math.log1p(-p)
    return _logsumexp(lc[i] + i * lp + (n - i) * lq for i in range(k, n + 1))


def log_tail_le(k: int, n: int, p: float) -> float:
    """log P(X <= k) for a float p."""
    if k >= n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("-inf")
    lc = _log_comb(n)
    lp, lq = math.log(p), math.log1p(-p)
    return _logsumexp(lc[i] + i * lp + (n - i) * lq for i in range(0, k + 1))


def clopper_pearson(k: int, n: int, conf: float) -> Tuple[float, float]:
    """Central (two-sided) Clopper-Pearson interval for k successes in n trials: the lower end
    solves P(X >= k | p) = (1-conf)/2, the upper end P(X <= k | p) = (1-conf)/2 (bisection)."""
    if n <= 0:
        return 0.0, 1.0
    target = math.log((1.0 - conf) / 2.0)

    def solve(f, increasing: bool) -> float:
        """The p in (0, 1) with f(p) = target, f monotone in p."""
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-17:
                break
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else solve(lambda q: log_tail_ge(k, n, q), increasing=True)
    upper = 1.0 if k == n else solve(lambda q: log_tail_le(k, n, q), increasing=False)
    return lower, upper


def holm(pvalues: Dict[str, Fraction], alpha: Fraction) -> Dict[str, object]:
    """Holm-Bonferroni step-down: sort ascending, reject H_(i) while p_(i) <= alpha / (m - i + 1)
    (i = 1..m), stop at the first acceptance.  Also returns the Holm-adjusted p-values."""
    order = sorted(pvalues, key=lambda t: (pvalues[t], t))
    m = len(order)
    rejected: Dict[str, bool] = {}
    threshold: Dict[str, Fraction] = {}
    adjusted: Dict[str, Fraction] = {}
    going = True
    running = Fraction(0)
    for i, t in enumerate(order):
        thr = Fraction(alpha) / (m - i)
        threshold[t] = thr
        running = max(running, min(Fraction(1), (m - i) * pvalues[t]))
        adjusted[t] = running
        if going and pvalues[t] <= thr:
            rejected[t] = True
        else:
            going = False
            rejected[t] = False
    return {"order": order, "threshold": threshold, "rejected": rejected, "adjusted": adjusted,
            "all_rejected": bool(order) and all(rejected.values()), "alpha": Fraction(alpha)}


def scipy_values(k: int, n: int, p: float) -> Optional[Dict[str, float]]:
    """The same statistics from scipy, or ``None`` when scipy is not installed."""
    try:
        from scipy import stats
    except ImportError:
        return None
    res = stats.binomtest(k, n, p, alternative="greater")
    two = stats.binomtest(k, n, p, alternative="two-sided")
    ci95 = two.proportion_ci(confidence_level=0.95, method="exact")
    ci99 = two.proportion_ci(confidence_level=0.99, method="exact")
    return {"p_greater": float(res.pvalue), "p_two_sided": float(two.pvalue),
            "cp95": (float(ci95.low), float(ci95.high)), "cp99": (float(ci99.low), float(ci99.high))}


def fmt_p(x: Fraction) -> str:
    """A p-value for display (no underflow: tiny values from their log10)."""
    if x == 0:
        return "0"
    v = to_float(x)
    if v >= 1e-4:
        return f"{v:.4f}"
    lg = log10_fraction(x)
    e = math.floor(lg)
    return f"{10 ** (lg - e):.2f}e{e:d}"


def fmt_frac(x: Fraction) -> str:
    return fmt_p(x) if x < Fraction(1, 10 ** 4) else f"{to_float(x):.4g}"


# ---------------------------------------------------------------------------
# Loading bench results
# ---------------------------------------------------------------------------
Chunk = Tuple[str, Dict[str, object], List[Dict[str, object]]]   # (source, metadata, per-game records)


def _expand(path: str) -> List[str]:
    if os.path.isdir(path):
        out = []
        for root, _dirs, files in os.walk(path):
            out.extend(os.path.join(root, f) for f in files if f.endswith((".json", ".jsonl")))
        return sorted(out)
    hits = sorted(glob.glob(path))
    if hits:
        out = []
        for h in hits:
            out.extend(_expand(h) if os.path.isdir(h) else [h])
        return out
    raise FileNotFoundError(path)


def load_chunks(path: str) -> List[Chunk]:
    """Every (metadata, records) chunk in ``path`` (file, directory or glob)."""
    chunks: List[Chunk] = []
    for f in _expand(path):
        if f.endswith(".jsonl"):
            with open(f) as fh:
                recs = [json.loads(line) for line in fh if line.strip()]
            chunks.append((f, {}, recs))
            continue
        with open(f) as fh:
            data = json.load(fh)
        items = data if isinstance(data, list) else [data]
        if items and all(isinstance(x, dict) and "won" in x and "results" not in x for x in items):
            chunks.append((f, {}, items))
            continue
        for x in items:
            if not isinstance(x, dict) or "results" not in x:
                raise ValueError(f"{f}: not a bench_catanatron.py result (no 'results')")
            meta = {k: v for k, v in x.items() if k != "results"}
            chunks.append((f, meta, list(x["results"])))
    return chunks


def infer_test(meta: Dict[str, object], recs: Sequence[Dict[str, object]]) -> Optional[str]:
    """Test id of a chunk from its opponent, format and catanatron version (``None`` if unknown)."""
    fmt = meta.get("format") or ("2v2" if any(len(r.get("our_seats") or [0]) == 2 for r in recs) else "1v3")
    version = str(meta.get("catanatron", ""))
    for tid, t in TESTS.items():
        if meta.get("opponent") == t.opponent and fmt == t.fmt and version.startswith(t.engine):
            return tid
    return None


# ---------------------------------------------------------------------------
# Per-test analysis
# ---------------------------------------------------------------------------
def _record_seats(r: Dict[str, object]) -> List[int]:
    seats = r.get("our_seats")
    if seats:
        return [int(s) for s in seats]
    return [int(r["seat"])] if r.get("seat") is not None else []


def _meta_problems(t: TestDef, source: str, meta: Dict[str, object]) -> List[str]:
    """Protocol deviations visible in a chunk's metadata.  Every registered field must be present
    and equal (a missing field is a deviation: it cannot be shown to be the registered one), and a
    chunk of bare per-game records without the bench's run metadata cannot be verified at all."""
    name = os.path.basename(source)
    if not meta:
        return [f"{name}: no run metadata (bare per-game records): engine, opponent, spec, trades, "
                f"PYTHONHASHSEED and rules cannot be verified"]
    probs = []

    def want(key, expected, got):
        if got != expected:
            probs.append(f"{name}: {key} is {got!r}, protocol {expected!r}")

    # Both the preset and the class: on 3.3 the stand-in presets vf / ab have the same class
    # names (ValueFunctionPlayer / AlphaBetaPlayer) as catanatron's own value / alphabeta.
    if meta.get("opponent") != t.opponent or meta.get("opponent_class") != t.opponent_class:
        probs.append(f"{name}: opponent {meta.get('opponent')!r} ({meta.get('opponent_class')}), "
                     f"protocol {t.opponent!r} ({t.opponent_class})")
    want("format", t.fmt, meta.get("format"))
    want("catanatron", t.engine, None if meta.get("catanatron") is None else str(meta.get("catanatron")))
    want("seed", t.seed, meta.get("seed"))
    want("PYTHONHASHSEED", PROTOCOL_HASH_SEED, None if meta.get("hash_seed") is None else str(meta.get("hash_seed")))
    want("trades", PROTOCOL_TRADES, meta.get("trades"))
    want("spec", PROTOCOL_SPEC, meta.get("spec"))
    if meta.get("opponent_params"):
        probs.append(f"{name}: opponent params {meta.get('opponent_params')}, protocol: defaults")
    want("vps_to_win", PROTOCOL_VPS_TO_WIN, meta.get("vps_to_win"))
    want("discard_limit", PROTOCOL_DISCARD_LIMIT, meta.get("discard_limit"))
    return probs


def _ranges(xs: Sequence[int]) -> str:
    xs = sorted(xs)
    out = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    txt = ",".join(out[:8])
    return txt + (",..." if len(out) > 8 else "")


def analyze_test(tid: str, chunks: Sequence[Chunk]) -> Dict[str, object]:
    """Statistics and protocol conformance of one test from its chunks."""
    t = TESTS[tid]
    problems: List[str] = []
    by_game: Dict[int, Dict[str, object]] = {}
    notes: List[str] = []
    duplicates = 0
    engines = set()
    opponents = set()
    for source, meta, recs in chunks:
        problems.extend(_meta_problems(t, source, meta))
        if meta:
            engines.add(str(meta.get("catanatron")))
            opponents.add(str(meta.get("opponent")))
        for r in recs:
            g = int(r["game"])
            if g in by_game:
                prev = by_game[g]
                key = ("seed", "winner_seat", "vps", "turns", "actions")
                if tuple(prev.get(k) for k in key) != tuple(r.get(k) for k in key):
                    problems.append(f"game {g} appears twice with different outcomes")
                duplicates += 1
                continue
            by_game[g] = r
    games = sorted(by_game)
    expected = set(range(t.games))
    missing = sorted(expected - set(games))
    extra = sorted(set(games) - expected)
    if missing:
        problems.append(f"{len(games) - len(extra)} of the {t.games} protocol games (missing {_ranges(missing)})")
    if extra:
        problems.append(f"{len(extra)} game(s) outside 0..{t.games - 1} ({_ranges(extra)})")
    if duplicates:
        notes.append(f"{duplicates} game(s) found in two chunks (overlapping ranges; identical unless listed as a "
                     f"deviation), counted once")
    recs = [by_game[g] for g in games if g in expected]
    null = NULL[t.fmt]
    n = len(recs)
    wins = 0
    capped = 0
    crashed_games = 0
    crash_attempts = 0
    err = {k: 0 for k in ERROR_KEYS}
    unmapped_top = 0
    opp_trade_errors = 0
    vp_ours: List[float] = []
    vp_opp: List[float] = []
    seat_rows = {s: [0, 0] for s in range(NUM_SEATS)}
    arr_rows = {seat_pattern(a): [0, 0] for a in ARRANGEMENTS_2V2}
    bad_seed = bad_seats = bad_won = 0
    for r in recs:
        g = int(r["game"])
        seats = _record_seats(r)
        if r.get("seed") != game_seed(t.seed, g):
            bad_seed += 1
        if tuple(seats) != our_seats_for(g, t.fmt):
            bad_seats += 1
        won = int(r.get("winner_seat", -1)) in seats and r.get("winner") is not None
        if bool(r.get("won")) != won:
            bad_won += 1
        wins += won
        if r.get("crashed"):
            crashed_games += 1
        elif r.get("winner") is None:
            capped += 1
        crash_attempts += int(r.get("crashes", 0) or 0)
        st = r.get("stats") or {}
        for k in ERROR_KEYS:
            err[k] += int(st.get(k, 0) or 0)
        unmapped_top += int(st.get("unmapped_top", 0) or 0)
        opp_trade_errors += int((r.get("trades") or {}).get("opp_errors", 0) or 0)
        if not r.get("crashed"):
            vps = r.get("vps") or []
            if vps:
                vp_ours.append(sum(vps[s] for s in seats) / len(seats))
                others = [v for i, v in enumerate(vps) if i not in seats]
                vp_opp.append(sum(others) / len(others))
        if t.fmt == "1v3":
            seat_rows[seats[0]][1] += 1
            seat_rows[seats[0]][0] += won
        else:
            row = arr_rows.setdefault(seat_pattern(seats), [0, 0])
            row[1] += 1
            row[0] += won
    if bad_seed:
        problems.append(f"{bad_seed} game(s) with a per-game seed other than {t.seed}*100003+g+1")
    if bad_seats:
        problems.append(f"{bad_seats} game(s) with catanbot seats other than the protocol rotation")
    if bad_won:
        problems.append(f"{bad_won} record(s) whose 'won' disagrees with the winner seat (recomputed here)")
    p_one = binom_sf_exact(wins, n, null) if n else Fraction(1)
    p_two = binom_two_sided_exact(wins, n, NULL["1v3"]) if n and t.fmt == "1v3" else None
    cp95 = clopper_pearson(wins, n, 0.95) if n else (0.0, 1.0)
    cp99 = clopper_pearson(wins, n, 0.99) if n else (0.0, 1.0)
    out: Dict[str, object] = {
        "test": tid, "opponent": t.opponent, "opponent_class": t.opponent_class, "format": t.fmt,
        "engine": t.engine, "protocol_games": t.games, "seed": t.seed,
        "games": n, "wins": wins, "win_rate": wins / n if n else 0.0, "null": null,
        "p_one_sided": p_one, "p_two_sided_vs_025": p_two, "cp95": cp95, "cp99": cp99,
        "avg_vp_ours": sum(vp_ours) / len(vp_ours) if vp_ours else 0.0,
        "avg_vp_opp": sum(vp_opp) / len(vp_opp) if vp_opp else 0.0,
        "turn_cap_games": capped, "crashed_games": crashed_games, "crash_attempts": crash_attempts,
        "errors": err["errors"], "observe_errors": err["observe_errors"], "fallbacks": err["fallback"],
        "unmapped_top": unmapped_top, "opp_trade_errors": opp_trade_errors,
        "problems": problems, "conforms": not problems, "notes": notes, "sources": sorted({c[0] for c in chunks}),
        "engines_seen": sorted(engines), "opponents_seen": sorted(opponents),
    }
    if t.fmt == "1v3":
        out["seats"] = {s: {"games": g_, "wins": w, "rate": w / g_ if g_ else 0.0,
                            "p_one_sided": binom_sf_exact(w, g_, NULL["1v3"]) if g_ else Fraction(1)}
                        for s, (w, g_) in seat_rows.items()}
    else:
        out["arrangements"] = {k: {"games": g_, "wins": w, "rate": w / g_ if g_ else 0.0}
                               for k, (w, g_) in arr_rows.items()}
    sp = scipy_values(wins, n, float(null)) if n else None
    if sp is not None:
        def rel(a: float, b: float) -> float:
            return abs(a - b) / max(abs(b), 1e-300) if (a or b) else 0.0
        diffs = [rel(to_float(p_one), sp["p_greater"]) if to_float(p_one) > 1e-290 else 0.0,
                 rel(cp95[0], sp["cp95"][0]), rel(cp95[1], sp["cp95"][1]),
                 rel(cp99[0], sp["cp99"][0]), rel(cp99[1], sp["cp99"][1])]
        if p_two is not None and to_float(p_two) > 1e-290:   # 1v3: the null is 0.25 already
            diffs.append(rel(to_float(p_two), sp["p_two_sided"]))
        out["scipy_max_rel_diff"] = max(diffs)
    return out


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------
def _cond(name: str, ok: bool, detail: str, failures: List[str]) -> Dict[str, object]:
    return {"name": name, "pass": bool(ok), "detail": detail, "failures": failures}


def evaluate(analysis: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Holm tables and the claim 1 / claim 2 verdicts (every failed condition with its reasons)."""
    present_t = [t for t in T_TESTS if t in analysis and analysis[t]["games"] > 0]
    missing_t = [t for t in T_TESTS if t not in present_t]
    # Holm is always over the six registered tests: a test without results enters with p = 1.
    pvals = {t: (analysis[t]["p_one_sided"] if t in present_t else Fraction(1)) for t in T_TESTS}
    holm1 = holm(pvals, CLAIM1_ALPHA)
    holm2 = holm(pvals, CLAIM2_ALPHA)

    def conformance(tests: Sequence[str]) -> Dict[str, object]:
        fails = []
        for t in tests:
            if t not in analysis or analysis[t]["games"] == 0:
                fails.append(f"{t}: no results")
            else:
                fails.extend(f"{t}: {p}" for p in analysis[t]["problems"])
        return _cond(f"results of {', '.join(tests)} complete and as pre-registered", not fails,
                     "opponent, format, engine, seeds, seats, spec, trades off, PYTHONHASHSEED=0, games 0..N-1",
                     fails)

    def holm_cond(h: Dict[str, object], label: str) -> Dict[str, object]:
        fails = []
        for t in h["order"]:
            if t in missing_t:
                fails.append(f"{t}: no results (enters Holm with p = 1)")
            elif not h["rejected"][t]:
                why = ("> Holm threshold" if pvals[t] > h["threshold"][t]
                       else "<= its threshold, but an earlier step already accepted; Holm threshold")
                fails.append(f"{t}: p = {fmt_p(pvals[t])} {why} {fmt_frac(h['threshold'][t])}")
        return _cond(f"T1-T6 reject their nulls at family-wise alpha = {label} (Holm)",
                     not fails and h["all_rejected"],
                     f"max Holm-adjusted p = {fmt_p(max(h['adjusted'].values())) if h['adjusted'] else 'n/a'}", fails)

    c1 = [conformance(T_TESTS), holm_cond(holm1, "0.01")]
    effect_fails = []
    for t in T_TESTS:
        if t not in analysis or analysis[t]["games"] == 0:
            effect_fails.append(f"{t}: no results")
            continue
        a = analysis[t]
        need = EFFECT_LOWER[a["format"]]
        if a["cp99"][0] < need:
            effect_fails.append(f"{t}: lower 99 % bound {a['cp99'][0]:.4f} < {need} "
                                f"({'win rate' if a['format'] == '1v3' else 'catanbot-win share'} "
                                f"{a['win_rate']:.3f}, {a['wins']}/{a['games']})")
    seat_fails = []
    for t in ("T1", "T2", "T3"):
        if t not in analysis or analysis[t]["games"] == 0:
            seat_fails.append(f"{t}: no results")
            continue
        for s, row in analysis[t]["seats"].items():
            if row["games"] == 0 or not (row["p_one_sided"] < SEAT_ALPHA):
                seat_fails.append(f"{t} seat {s}: {row['wins']}/{row['games']} = {row['rate']:.3f}, "
                                  f"one-sided p = {fmt_p(row['p_one_sided'])} >= 0.05")
    err_fails = []
    all_tests = T_TESTS + R_TESTS
    for t in all_tests:
        if t not in analysis:
            err_fails.append(f"{t}: no results")
            continue
        a = analysis[t]
        bits = [f"{a[k]} {label}" for k, label in (("errors", "adapter errors"), ("observe_errors", "observe errors"),
                                                   ("fallbacks", "illegal-action fallbacks"),
                                                   ("crash_attempts", "crashed attempts"),
                                                   ("crashed_games", "games crashed twice")) if a[k]]
        if bits:
            err_fails.append(f"{t}: " + ", ".join(bits))
    r_fails = []
    for t in R_TESTS:
        if t not in analysis or analysis[t]["games"] == 0:
            r_fails.append(f"{t}: no results")
            continue
        a = analysis[t]
        p2 = a["p_two_sided_vs_025"]
        if a["win_rate"] <= 0.25 and not (p2 > R_ALPHA):
            r_fails.append(f"{t}: win rate {a['win_rate']:.3f} ({a['wins']}/{a['games']}) is significantly below 0.25 "
                           f"(two-sided p = {fmt_p(p2)} <= 0.01)")
    c2 = [
        conformance(all_tests),
        holm_cond(holm2, "5.7e-7"),
        _cond("effect size: lower 99 % Clopper-Pearson bound >= 0.35 (T1-T3) / >= 0.55 (T4-T6)", not effect_fails,
              "; ".join(f"{t} {analysis[t]['cp99'][0]:.3f}" for t in T_TESTS if t in analysis and analysis[t]["games"]),
              effect_fails),
        _cond("seat robustness: every seat of T1-T3 above 0.25 (one-sided exact p < 0.05)", not seat_fails, "",
              seat_fails),
        _cond("zero adapter errors, illegal-action fallbacks and crashes over all proof games", not err_fails, "",
              err_fails),
        _cond("R1 / R2: 1v3 win rate not significantly below 0.25 (two-sided exact p > 0.01)", not r_fails,
              "; ".join(f"{t} {analysis[t]['win_rate']:.3f} (p = {fmt_p(analysis[t]['p_two_sided_vs_025'])})"
                        for t in R_TESTS if t in analysis and analysis[t]["games"]), r_fails),
    ]
    return {
        "holm_claim1": holm1, "holm_claim2": holm2,
        "claim1": {"pass": all(c["pass"] for c in c1), "conditions": c1},
        "claim2": {"pass": all(c["pass"] for c in c2), "conditions": c2},
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _cp(ci: Tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _errs(a: Dict[str, object]) -> str:
    return f"{a['errors'] + a['observe_errors']}/{a['fallbacks']}/{a['crash_attempts']}"


def report_text(analysis: Dict[str, Dict[str, object]], verdict: Dict[str, object]) -> str:
    L: List[str] = []
    L.append("Strength proof (docs/PROOF_PROTOCOL.md, pre-registered 2026-09-25)")
    L.append(f'bot "{PROTOCOL_SPEC}"; trades {PROTOCOL_TRADES}; PYTHONHASHSEED={PROTOCOL_HASH_SEED}; '
             f"turn-cap games count as losses")
    L.append("")
    L.append(f"{'test':<4} {'opponent':<10} {'fmt':<4} {'games':>10} {'wins':>5} {'rate':>6} {'null':>5} "
             f"{'p one-sided':>11} {'CP 95 %':>16} {'CP 99 %':>16} {'VP ours/opp':>12} {'cap':>4} "
             f"{'err/fb/crash':>12}  protocol")
    for tid in TESTS:
        if tid not in analysis:
            L.append(f"{tid:<4} (no results)")
            continue
        a = analysis[tid]
        L.append(f"{tid:<4} {a['opponent']:<10} {a['format']:<4} {a['games']:>5}/{a['protocol_games']:<4} "
                 f"{a['wins']:>5} {a['win_rate']:>6.3f} {to_float(a['null']):>5.2f} {fmt_p(a['p_one_sided']):>11} "
                 f"{_cp(a['cp95']):>16} {_cp(a['cp99']):>16} {a['avg_vp_ours']:>5.2f}/{a['avg_vp_opp']:<5.2f} "
                 f"{a['turn_cap_games']:>5} {_errs(a):>12}  {'ok' if a['conforms'] else 'DEVIATES'}")
    L.append("  (err = adapter + observe errors, fb = illegal-action fallbacks, crash = crashed attempts)")
    L.append("")
    for tid, a in analysis.items():
        if a["format"] == "1v3":
            cells = ", ".join(f"seat{s} {r['wins']}/{r['games']} = {r['rate']:.3f} (p {fmt_p(r['p_one_sided'])})"
                              for s, r in a["seats"].items())
            L.append(f"{tid} by seat (one-sided p vs 0.25): {cells}")
        else:
            cells = ", ".join(f"{k} {r['wins']}/{r['games']} = {r['rate']:.3f}" for k, r in a["arrangements"].items())
            L.append(f"{tid} by arrangement (C = catanbot, turn order): {cells}")
        if a["format"] == "1v3" and a["p_two_sided_vs_025"] is not None and tid in R_TESTS:
            L.append(f"{tid} two-sided exact p vs 0.25: {fmt_p(a['p_two_sided_vs_025'])}")
        for note in a.get("notes", []):
            L.append(f"{tid} note: {note}")
    L.append("")
    for key, label in (("holm_claim1", "0.01 (claim 1)"), ("holm_claim2", "5.7e-7 (claim 2)")):
        h = verdict[key]
        L.append(f"Holm-Bonferroni over T1-T6 at family alpha {label}:")
        for i, t in enumerate(h["order"]):
            pt = fmt_p(analysis[t]["p_one_sided"]) if t in analysis and analysis[t]["games"] else "1 (none)"
            L.append(f"  {i + 1}. {t} p = {pt:>10}  threshold alpha/{len(h['order']) - i}"
                     f" = {fmt_frac(h['threshold'][t]):>9}  adjusted p = {fmt_p(h['adjusted'][t]):>10}  "
                     f"{'reject' if h['rejected'][t] else 'ACCEPT'}")
        if not h["order"]:
            L.append("  (no T results)")
    L.append("")
    for key, label in (("claim1", "CLAIM 1 - better than Catanatron's strong bots"),
                       ("claim2", "CLAIM 2 - ready for (supervised) human testing")):
        c = verdict[key]
        L.append(f"{label}: {'PASS' if c['pass'] else 'FAIL'}")
        for i, cond in enumerate(c["conditions"]):
            L.append(f"  [{'pass' if cond['pass'] else 'FAIL'}] {cond['name']}"
                     + (f" - {cond['detail']}" if cond["detail"] else ""))
            for f in cond["failures"][:12]:
                L.append(f"         - {f}")
            if len(cond["failures"]) > 12:
                L.append(f"         - ... {len(cond['failures']) - 12} more")
    diffs = [a["scipy_max_rel_diff"] for a in analysis.values() if "scipy_max_rel_diff" in a]
    L.append("")
    L.append(f"exact pure-Python statistics; scipy cross-check: max relative difference {max(diffs):.1e}" if diffs
             else "exact pure-Python statistics (scipy not installed: no cross-check)")
    return "\n".join(L)


def report_markdown(analysis: Dict[str, Dict[str, object]], verdict: Dict[str, object], sources: Sequence[str]) -> str:
    L: List[str] = []
    today = datetime.date.today().isoformat()
    L.append("# Strength proof results")
    L.append("")
    L.append(f"Generated {today} by `scripts/prove_strength.py` from the bench results of the pre-registered "
             "protocol `docs/PROOF_PROTOCOL.md` (2026-09-25).  Bot under test: "
             f"`{PROTOCOL_SPEC}`; domestic trading off in every game (catanatron 3.3's players never answer "
             "offers with their own evaluation, see the protocol's tooling amendment); `PYTHONHASHSEED=0`; "
             "games that hit the turn cap count as losses.")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("| claim | verdict | failed conditions |")
    L.append("|---|---|---|")
    for key, label in (("claim1", "1 - better than Catanatron's strong bots"),
                       ("claim2", "2 - ready for supervised human testing")):
        c = verdict[key]
        failed = "; ".join(cond["name"] for cond in c["conditions"] if not cond["pass"]) or "-"
        L.append(f"| {label} | **{'PASS' if c['pass'] else 'FAIL'}** | {failed} |")
    L.append("")
    L.append("## Tests")
    L.append("")
    L.append("| test | opponent | format | games | wins | win rate | null | one-sided p | 95 % CP | 99 % CP | "
             "avg VP ours / opp | turn cap | errors / fallbacks / crashes | as registered |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tid in TESTS:
        if tid not in analysis:
            L.append(f"| {tid} | {TESTS[tid].opponent} | {TESTS[tid].fmt} | - | | | | | | | | | | no results |")
            continue
        a = analysis[tid]
        seen = ", ".join(a["opponents_seen"]) or a["opponent"]
        eng = ", ".join(a["engines_seen"]) or "?"
        L.append(f"| {tid} | `{seen}` (protocol `{a['opponent']}` = {a['opponent_class']}; catanatron {eng}) | {a['format']} | "
                 f"{a['games']} / {a['protocol_games']} | {a['wins']} | {a['win_rate']:.3f} | {to_float(a['null']):.2f} | "
                 f"{fmt_p(a['p_one_sided'])} | {_cp(a['cp95'])} | {_cp(a['cp99'])} | "
                 f"{a['avg_vp_ours']:.2f} / {a['avg_vp_opp']:.2f} | {a['turn_cap_games']} | "
                 f"{a['errors'] + a['observe_errors']} / {a['fallbacks']} / {a['crash_attempts']} | "
                 f"{'yes' if a['conforms'] else 'NO'} |")
    L.append("")
    L.append("## Seats (1v3) and arrangements (2v2)")
    L.append("")
    for tid, a in analysis.items():
        if a["format"] == "1v3":
            L.append(f"* {tid}: " + ", ".join(f"seat {s} {r['wins']}/{r['games']} = {r['rate']:.3f} "
                                              f"(p {fmt_p(r['p_one_sided'])})" for s, r in a["seats"].items()))
        else:
            L.append(f"* {tid}: " + ", ".join(f"`{k}` {r['wins']}/{r['games']} = {r['rate']:.3f}"
                                              for k, r in a["arrangements"].items()))
    L.append("")
    L.append("Seat 0 moves first; one-sided exact p against 0.25 per seat.  2v2 patterns list the turn order "
             "(`C` = catanbot, `o` = opponent).")
    L.append("")
    L.append("## Holm-Bonferroni over T1-T6")
    L.append("")
    L.append("| step | test | p | threshold (0.01) | claim 1 | threshold (5.7e-7) | claim 2 |")
    L.append("|---|---|---|---|---|---|---|")
    h1, h2 = verdict["holm_claim1"], verdict["holm_claim2"]
    for i, t in enumerate(h1["order"]):
        pt = fmt_p(analysis[t]["p_one_sided"]) if t in analysis and analysis[t]["games"] else "1 (no results)"
        L.append(f"| {i + 1} | {t} | {pt} | {fmt_frac(h1['threshold'][t])} | "
                 f"{'reject' if h1['rejected'][t] else 'accept'} | {fmt_frac(h2['threshold'][t])} | "
                 f"{'reject' if h2['rejected'][t] else 'accept'} |")
    L.append("")
    L.append("## Conditions")
    L.append("")
    for key, label in (("claim1", "Claim 1"), ("claim2", "Claim 2")):
        L.append(f"**{label}: {'PASS' if verdict[key]['pass'] else 'FAIL'}**")
        L.append("")
        for cond in verdict[key]["conditions"]:
            L.append(f"* {'PASS' if cond['pass'] else 'FAIL'} - {cond['name']}"
                     + (f" ({cond['detail']})" if cond["detail"] else ""))
            for f in cond["failures"]:
                L.append(f"  * {f}")
        L.append("")
    L.append("## Method")
    L.append("")
    L.append("Exact one-sided binomial tests against the protocol nulls (1v3 win rate 0.25, 2v2 catanbot-win "
             "share 0.5), computed in rational arithmetic; Clopper-Pearson intervals are the central two-sided "
             "ones (the \"lower 99 % bound\" is the lower end of the 99 % interval, 0.5 % per tail); R1 / R2 use "
             "the exact two-sided test (minlike rule, as scipy / R).  Holm-Bonferroni step-down over the six "
             "T tests.  Every game is replayable from the action logs (`scripts/replay_catanatron.py`).")
    diffs = [a["scipy_max_rel_diff"] for a in analysis.values() if "scipy_max_rel_diff" in a]
    if diffs:
        L.append(f"scipy cross-check of every p-value and interval: largest relative difference {max(diffs):.1e}.")
    L.append("")
    L.append("Result files: " + ", ".join(f"`{s}`" for s in sources[:40]) + (" ..." if len(sources) > 40 else ""))
    L.append("")
    return "\n".join(L)


def _jsonable(x):
    if isinstance(x, Fraction):
        return {"float": to_float(x), "log10": log10_fraction(x), "exact": f"{x.numerator}/{x.denominator}"
                if x.denominator < 10 ** 50 else None}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def collect(test_args: Sequence[str], positional: Sequence[str]) -> Tuple[Dict[str, List[Chunk]], List[str]]:
    by_test: Dict[str, List[Chunk]] = {}
    sources: List[str] = []
    for item in test_args:
        if "=" not in item:
            raise SystemExit(f"--test wants ID=PATH, got {item!r}")
        tid, paths = item.split("=", 1)
        tid = tid.strip().upper()
        if tid not in TESTS:
            raise SystemExit(f"unknown test id {tid!r} (known: {', '.join(TESTS)})")
        for p in paths.split(","):
            if not p.strip():
                continue
            chunks = load_chunks(p.strip())
            by_test.setdefault(tid, []).extend(chunks)
            sources.extend(c[0] for c in chunks)
    for p in positional:
        for chunk in load_chunks(p):
            tid = infer_test(chunk[1], chunk[2])
            if tid is None:
                raise SystemExit(f"{chunk[0]}: cannot infer the test id (opponent {chunk[1].get('opponent')!r}, "
                                 f"catanatron {chunk[1].get('catanatron')!r}); use --test ID=PATH")
            by_test.setdefault(tid, []).append(chunk)
            sources.append(chunk[0])
    return by_test, sorted(set(sources))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="*", help="bench JSON / JSONL files or directories (test id inferred)")
    ap.add_argument("--test", action="append", default=[], metavar="ID=PATH",
                    help="results of test ID (T1..T6, R1, R2); PATH is a file, directory, glob or comma list")
    ap.add_argument("--markdown", default=None, metavar="PATH", help="write the docs/PROOF.md body here")
    ap.add_argument("--json", default=None, metavar="PATH", help="write the full analysis as JSON")
    args = ap.parse_args(argv)
    try:
        by_test, sources = collect(args.test, args.results)
    except (FileNotFoundError, ValueError) as ex:
        print(f"cannot read results: {ex}", file=sys.stderr)
        return 2
    if not by_test:
        print("no results given (--test ID=PATH or positional files)", file=sys.stderr)
        return 2
    analysis = {tid: analyze_test(tid, by_test[tid]) for tid in TESTS if tid in by_test}
    verdict = evaluate(analysis)
    print(report_text(analysis, verdict))
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(report_markdown(analysis, verdict, sources))
        print(f"markdown: {args.markdown}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable({"analysis": analysis, "verdict": verdict}), fh, indent=1)
        print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
