"""Exact statistics for the champion league (pure Python, no scipy needed).

* :func:`binom_sf` / :func:`binom_cdf` - binomial tails through the regularised
  incomplete beta function (continued fraction, relative accuracy ~1e-13),
  so tiny p-values keep their relative precision.
* :func:`binom_test_greater` / :func:`binom_test_less` / :func:`binom_test_two_sided`
  - exact binomial tests of a win count against a null share ``p0`` (the
  two-sided one uses scipy's "minlike" rule).
* :func:`clopper_pearson` - central two-sided Clopper-Pearson interval.
* :func:`holm` - Holm-Bonferroni step-down adjusted p-values and rejections.
* :func:`fisher_less` - one-sided Fisher exact test "rate 1 < rate 2" (the
  external non-regression hook compares win rates with it).

Everything here is deterministic and cheap enough to recompute from the raw
game records on every status call.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

__all__ = ["betainc", "beta_ppf", "binom_sf", "binom_cdf", "binom_pmf", "binom_test_greater", "binom_test_less",
           "binom_test_two_sided", "clopper_pearson", "holm", "fisher_less", "norm_cdf", "norm_ppf"]

_FPMIN = 1e-300
_EPS = 1e-16


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz, Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 200000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            return h
    raise ArithmeticError(f"betainc continued fraction did not converge (a={a}, b={b}, x={x})")


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)`` for ``a, b > 0``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse of :func:`betainc` in ``x`` (bisection to ~1e-15)."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    lg = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(lg + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return betainc(k, n - k + 1, p)


def binom_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return betainc(n - k, k + 1, 1.0 - p)


def binom_test_greater(k: int, n: int, p0: float) -> float:
    """One-sided exact p-value for "share > p0": ``P(X >= k | p0)``."""
    return binom_sf(k, n, p0) if n > 0 else 1.0


def binom_test_less(k: int, n: int, p0: float) -> float:
    """One-sided exact p-value for "share < p0": ``P(X <= k | p0)``."""
    return binom_cdf(k, n, p0) if n > 0 else 1.0


def binom_test_two_sided(k: int, n: int, p0: float) -> float:
    """Two-sided exact p-value (scipy's "minlike": sum of outcomes no more likely than ``k``)."""
    if n <= 0:
        return 1.0
    if p0 <= 0.0 or p0 >= 1.0:
        return 1.0 if binom_pmf(k, n, p0) > 0 else 0.0
    thr = binom_pmf(k, n, p0) * (1.0 + 1e-7)
    lgn, lp, lq = math.lgamma(n + 1), math.log(p0), math.log1p(-p0)
    total = 0.0
    for i in range(n + 1):
        pm = math.exp(lgn - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
        if pm <= thr:
            total += pm
    return min(1.0, total)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Central two-sided Clopper-Pearson interval (``(1 - conf) / 2`` in each tail)."""
    if n <= 0:
        return 0.0, 1.0
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k <= 0 else beta_ppf(a, k, n - k + 1)
    hi = 1.0 if k >= n else beta_ppf(1.0 - a, k + 1, n - k)
    return lo, hi


def holm(pvalues: Mapping[str, float], alpha: float) -> Dict[str, Dict[str, float]]:
    """Holm-Bonferroni: ``{name: {"p", "p_adj", "reject"}}`` at family level ``alpha``."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out: Dict[str, Dict[str, float]] = {}
    running = 0.0
    still = True
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        reject = still and p <= alpha / (m - i)
        if not reject:
            still = False
        out[name] = {"p": p, "p_adj": running, "reject": bool(reject)}
    return out


def fisher_less(w1: int, n1: int, w2: int, n2: int) -> float:
    """One-sided Fisher exact p-value for "rate 1 < rate 2" (``P(W1 <= w1)`` given the margins)."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    total_w = w1 + w2
    N = n1 + n2
    lo = max(0, total_w - n2)

    def logc(a: int, b: int) -> float:
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    denom = logc(N, total_w)
    terms = [logc(n1, x) + logc(n2, total_w - x) - denom for x in range(lo, w1 + 1)]
    if not terms:
        return 0.0
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(q: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(q)
