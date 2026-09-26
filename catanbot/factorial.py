"""2^k factorial tests of strategy terms: main effects and interactions from paired games.

One-at-a-time ablations measure a term with every other term at its default.  Terms interact (a robber
on our hex hurts less while we hold a knight), so a suspected pair (or triple) is tested as a full 2^k
design: every combination of "factor at its test value" / "factor at its default" is played on the SAME
seeds (``scripts/ablate.py --factorial`` in self-play, ``scripts/ablate_catanatron.py --factorial``
against Catanatron; docs/TUNING.md).  This module holds the design and the arithmetic only (no games).

Cells are bit masks over the factors (bit ``i`` set = factor ``i`` at its test value; ``0`` = the base,
every factor at its default).  ``values[cell]`` is one number per unit (a game or a seed), aligned across
cells: unit ``u`` of every cell was played on the same board and dice.  For an effect ``E`` (a non-empty
set of factors) the per-unit contrast is

    c_u(E) = 2^-(k-|E|) * sum over cells S of  prod_{i in E} (+1 if i in S else -1) * values[S][u]

so a main effect (``|E| = 1``) is "factor on minus factor off, averaged over the levels of the other
factors" and a two-factor interaction in a 2x2 is ``AB - A - B + base`` (the difference of differences:
how much more factor A is worth when B is on).  In a 2^3 design a two-factor interaction is that
difference of differences averaged over the third factor.  The effect is the mean of ``c_u`` over units,
its standard error the paired one (unbiased sample variance of ``c_u`` / n): the common seeds cancel the
board / dice luck shared by the cells.  An interaction's contrast sums four cells, so at equal games its
s.e. is about twice a main effect's (it needs ~4x the games for the same precision).

In self-play the cells are measured *against the base* (candidate seats vs default seats in the same
game), so ``values[S][u]`` is already a paired difference and the base cell is identically 0; against
Catanatron every cell (the base included) is an absolute per-seed result (win 0/1, final VP).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

Z95 = 1.959964
MAX_FACTORS = 4          # 16 cells; beyond that a fractional design would be the tool, not this one

Factor = Tuple[str, Any]   # (registry name, test value); the other level is the registry default


def parse_factors(text: str) -> List[Factor]:
    """``"devcards.KNIGHT_VALUE=0.8,heuristic.EXPOSURE_WEIGHT=0"`` -> ``[(name, value), ...]``.

    Values go through the registry's parsers; a flag given without ``=`` (or ``=off``) is tested
    switched off.  Each factor must differ from its default and appear once."""
    from . import tuning
    out: List[Factor] = []
    for item in text.split(","):
        if not item.strip():
            continue
        name, sep, value = item.partition("=")
        try:
            t = tuning.find(name.strip())
        except KeyError as ex:
            raise ValueError(str(ex).strip("'\""))
        if not sep:
            if t.kind != "flag":
                raise ValueError(f"{t.name}: give the test value as {t.name}=VALUE")
            v: Any = False
        else:
            v = t.parse(value.strip())
        if t.format(v) == t.format(t.default):
            raise ValueError(f"{t.name}={t.format(v)} is its default: a factor needs a value that differs")
        if any(n == t.name for n, _ in out):
            raise ValueError(f"{t.name} given twice")
        out.append((t.name, v))
    if len(out) < 2:
        raise ValueError("a factorial test needs at least 2 factors (NAME=VALUE,NAME2=VALUE2)")
    if len(out) > MAX_FACTORS:
        raise ValueError(f"at most {MAX_FACTORS} factors ({2 ** MAX_FACTORS} cells)")
    return out


def cells(k: int) -> List[int]:
    """Every cell of a 2^k design, the base (0) first."""
    return list(range(2 ** k))


def effects(k: int) -> List[int]:
    """Every effect (non-empty factor set) as a mask: main effects first, then 2-way, 3-way, ..."""
    return sorted(range(1, 2 ** k), key=lambda m: (bin(m).count("1"), m))


def members(mask: int, k: int) -> List[int]:
    return [i for i in range(k) if mask >> i & 1]


def cell_overrides(factors: Sequence[Factor], mask: int) -> Dict[str, Any]:
    return {factors[i][0]: factors[i][1] for i in members(mask, len(factors))}


def _short(name: str) -> str:
    return name.split(".", 1)[1] if "." in name else name


def cell_label(factors: Sequence[Factor], mask: int) -> str:
    if mask == 0:
        return "base"
    from . import tuning
    return " + ".join(f"{_short(n)}={tuning.find(n).format(v)}" for n, v in
                      (factors[i] for i in members(mask, len(factors))))


def effect_label(factors: Sequence[Factor], mask: int) -> str:
    from . import tuning
    return " x ".join(f"{_short(factors[i][0])}={tuning.find(factors[i][0]).format(factors[i][1])}"
                      for i in members(mask, len(factors)))


def weights(k: int, effect: int) -> Dict[int, float]:
    """Contrast coefficient of every cell for ``effect`` (see the module docstring)."""
    scale = 2.0 ** -(k - bin(effect).count("1"))
    out = {}
    for c in cells(k):
        sign = 1.0
        for i in members(effect, k):
            sign *= 1.0 if c >> i & 1 else -1.0
        out[c] = sign * scale
    return out


def mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def summary(xs: Sequence[float]) -> Dict[str, Any]:
    m, se = mean_se(xs)
    ci = [m - Z95 * se, m + Z95 * se] if not math.isnan(se) else [float("nan"), float("nan")]
    if math.isnan(se) or math.isnan(m):
        p = float("nan")
    elif se <= 0:
        p = 1.0 if m == 0 else 0.0
    else:
        p = math.erfc(abs(m / se) / math.sqrt(2.0))
    return {"n": len(xs), "mean": m, "se": se, "ci95": ci, "p": p,
            "mde80": 2.8 * se if not math.isnan(se) else float("nan")}


def reading(st: Dict[str, Any], min_units: int = 30) -> str:
    lo, hi = st["ci95"]
    if st["n"] < min_units or math.isnan(lo):
        return "inconclusive (too few games)"
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "not significant"


def analyse(factors: Sequence[Factor], values: Dict[int, Sequence[float]], min_units: int = 30) -> Dict[str, Any]:
    """Cells, effects and the additivity check from aligned per-unit values.

    ``values`` maps every non-base cell (and optionally the base) to one number per unit; a missing
    base means 0 for every unit (self-play: the cells are already paired differences against it)."""
    k = len(factors)
    n = min(len(values[c]) for c in cells(k) if c)
    vals = {c: list(values[c])[:n] if c in values else [0.0] * n for c in cells(k)}
    base = vals[0]
    cell_rows = []
    for c in cells(k):
        if c == 0:
            continue
        st = summary([vals[c][u] - base[u] for u in range(n)])
        st.update(cell=c, label=cell_label(factors, c), overrides=cell_overrides(factors, c),
                  reading=reading(st, min_units))
        cell_rows.append(st)
    eff_rows = []
    for e in effects(k):
        w = weights(k, e)
        st = summary([sum(w[c] * vals[c][u] for c in w) for u in range(n)])
        st.update(effect=e, order=bin(e).count("1"), label=effect_label(factors, e),
                  factors=[factors[i][0] for i in members(e, k)], reading=reading(st, min_units))
        eff_rows.append(st)
    # additivity: every multi-factor cell against the sum of its single-factor cells (vs base)
    add_rows = []
    for c in cells(k):
        if bin(c).count("1") < 2:
            continue
        diff = [(vals[c][u] - base[u]) - sum(vals[1 << i][u] - base[u] for i in members(c, k)) for u in range(n)]
        st = summary(diff)
        st.update(cell=c, label=cell_label(factors, c),
                  observed=mean_se([vals[c][u] - base[u] for u in range(n)])[0],
                  predicted=sum(mean_se([vals[1 << i][u] - base[u] for u in range(n)])[0] for i in members(c, k)))
        add_rows.append(st)
    return {"factors": [[n_, v] for n_, v in factors], "k": k, "units": n, "cells": cell_rows, "effects": eff_rows,
            "additivity": add_rows}


def _pp(x: Optional[float], sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:+.1f}pp" if sign else f"{100.0 * x:.1f}pp"


def format_report(res: Dict[str, Any], unit: str = "games", scale_pp: bool = True, what: str = "win rate") -> List[str]:
    """Plain-text lines for a result of :func:`analyse` (percentage points when ``scale_pp``)."""
    fmt = (lambda x, sign=True: _pp(x, sign)) if scale_pp else \
        (lambda x, sign=True: "n/a" if x is None or math.isnan(x) else (f"{x:+.3f}" if sign else f"{x:.3f}"))
    lines = [f"  {what}: {res['units']} {unit} per cell, paired over the common seeds"]
    lines.append("  cells vs base (each alone = the one-at-a-time ablation on these seeds):")
    for r in res["cells"]:
        lo, hi = r["ci95"]
        lines.append(f"    {r['label']:58} {fmt(r['mean']):>9} +- {fmt(r['se'], False):>7}  95% CI [{fmt(lo)}, {fmt(hi)}]")
    lines.append("  effects (main = on minus off averaged over the other factors; interaction = difference of "
                 "differences, AB - A - B + base):")
    for r in res["effects"]:
        lo, hi = r["ci95"]
        kind = "main" if r["order"] == 1 else f"{r['order']}-way"
        lines.append(f"    {kind:6} {r['label']:51} {fmt(r['mean']):>9} +- {fmt(r['se'], False):>7}  95% CI "
                     f"[{fmt(lo)}, {fmt(hi)}]  p {r['p']:.3f}  {r['reading']}")
    for r in res["additivity"]:
        lines.append(f"    additivity {r['label']}: observed {fmt(r['observed'])} vs sum of the singles "
                     f"{fmt(r['predicted'])}")
    return lines
