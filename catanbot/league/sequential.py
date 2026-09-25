"""Group-sequential boundaries for the promotion gate (exact, binomial, alpha spending).

The gate plays the candidate against a champion in batches and looks at the
data after every batch.  Testing "one-sided p < alpha" at every look would
inflate the false-promotion rate (six looks at a nominal 0.01 give ~0.03).
Instead the gate uses a **Lan-DeMets alpha-spending function**:

* ``obf`` (default) - O'Brien-Fleming type,
  ``alpha(t) = 2 * (1 - Phi(z_{1 - alpha/2} / sqrt(t)))``: almost nothing is
  spent early (only an overwhelming lead stops a gate after one batch) and the
  final look keeps a nominal level close to ``alpha``.
* ``pocock`` - Pocock type, ``alpha(t) = alpha * ln(1 + (e - 1) * t)``: equal
  spending per look, earlier stops, a stricter final look.

``t`` is the information fraction: decisive games so far / planned maximum
(``t = 1`` at the final look, whatever the number of turn-cap draws).

The boundaries are computed **exactly for the binomial**, not from the normal
approximation: at look ``k`` the null distribution of the cumulative win count
over the paths that have not crossed yet is propagated by convolution with the
batch's ``Binomial(m_k, p0)``, and the critical count ``c_k`` is the smallest
count whose crossing probability fits in the alpha allowed so far minus what
earlier looks actually spent.  Hence

    P_null(candidate ever crosses)  =  sum of the spent increments  <=  alpha

exactly (up to float rounding, ~1e-15), for any batch sizes, including batch
sizes that depend on how many draws occurred.  The lower ("clearly worse")
boundary is the mirror image on the loss count.  Each boundary is computed as
if the other did not exist: stopping for the other reason only removes paths,
so both error rates stay below their nominal levels (non-binding boundaries);
the optional futility rule is non-binding for the same reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .stats import binom_sf, norm_cdf, norm_ppf

__all__ = ["SPENDING", "spend", "binom_pmf_vec", "Boundary", "upper_boundary", "lower_boundary",
           "simulate_type1", "nominal_levels"]


def spend_obf(t: float, alpha: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return alpha
    z = norm_ppf(1.0 - alpha / 2.0)
    return 2.0 * (1.0 - norm_cdf(z / math.sqrt(t)))


def spend_pocock(t: float, alpha: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return alpha
    return alpha * math.log(1.0 + (math.e - 1.0) * t)


SPENDING: Dict[str, Callable[[float, float], float]] = {"obf": spend_obf, "pocock": spend_pocock}


def spend(kind: str, t: float, alpha: float) -> float:
    try:
        return SPENDING[kind](t, alpha)
    except KeyError:
        raise ValueError(f"unknown spending function {kind!r} (choose from {sorted(SPENDING)})") from None


def binom_pmf_vec(m: int, p: float) -> np.ndarray:
    """``[P(X = 0), ..., P(X = m)]`` for ``X ~ Binomial(m, p)``."""
    if m <= 0:
        return np.ones(1)
    k = np.arange(m + 1, dtype=np.float64)
    lg = np.array([math.lgamma(i + 1) for i in range(m + 1)])
    logp = lg[m] - lg - lg[::-1] + k * math.log(p) + (m - k) * math.log1p(-p)
    return np.exp(logp)


@dataclass
class Boundary:
    """Critical counts of one boundary over the looks so far.

    ``crit[k]``: the boundary is crossed at look ``k`` when the (upper: win, lower:
    loss) count is ``>= crit[k]``; ``crit[k] = n_k + 1`` means "cannot cross here".
    ``spent[k]``: cumulative null crossing probability through look ``k``.
    ``allowed[k]``: the spending function's value at look ``k``.
    """
    ns: List[int] = field(default_factory=list)
    ts: List[float] = field(default_factory=list)
    crit: List[int] = field(default_factory=list)
    spent: List[float] = field(default_factory=list)
    allowed: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, list]:
        return {"ns": list(self.ns), "ts": [round(t, 6) for t in self.ts], "crit": list(self.crit),
                "spent": list(self.spent), "allowed": list(self.allowed)}


def upper_boundary(ns: Sequence[int], n_max: int, alpha: float, p0: float, spending: str = "obf",
                   final: Optional[Sequence[bool]] = None) -> Boundary:
    """Exact spending boundary for "win count too high" under ``Binomial(., p0)``.

    ``ns``: cumulative decisive games at each look (non-decreasing).  ``n_max``:
    planned maximum (information fraction ``n / n_max``).  ``final[k]`` forces
    ``t = 1`` at look ``k`` (the last look spends all remaining alpha).
    """
    b = Boundary()
    dist = np.ones(1)
    prev = 0
    spent = 0.0
    for k, n in enumerate(ns):
        n = int(n)
        if n < prev:
            raise ValueError(f"decisive game counts must not decrease: {list(ns)}")
        dist = np.convolve(dist, binom_pmf_vec(n - prev, p0))
        prev = n
        is_final = bool(final[k]) if final is not None else False
        t = 1.0 if is_final else (min(1.0, n / n_max) if n_max > 0 else 1.0)
        allowed = spend(spending, t, alpha)
        budget = allowed - spent
        # tail[c] = P(S >= c, not crossed before), c = 0..n+1
        tail = np.concatenate([np.cumsum(dist[::-1])[::-1], [0.0]])
        ok = np.nonzero(tail <= budget + 1e-15)[0]
        c = int(ok[0]) if len(ok) else n + 1
        if c == 0:
            c = 1 if n >= 1 else n + 1   # never "cross" with zero wins
        spent += float(tail[c])
        dist[c:] = 0.0
        b.ns.append(n)
        b.ts.append(t)
        b.crit.append(c)
        b.spent.append(spent)
        b.allowed.append(allowed)
    return b


def lower_boundary(ns: Sequence[int], n_max: int, alpha: float, p0: float, spending: str = "obf",
                   final: Optional[Sequence[bool]] = None) -> Boundary:
    """Boundary for "win count too low": the upper boundary on the *loss* count (null ``1 - p0``).

    Crossed at look ``k`` when ``losses >= crit[k]``, i.e. ``wins <= n_k - crit[k]``.
    """
    return upper_boundary(ns, n_max, alpha, 1.0 - p0, spending, final)


def nominal_levels(b: Boundary, p0: float) -> List[float]:
    """Unconditional one-sided nominal level of each look's critical count, ``P(Bin(n_k, p0) >= crit_k)``."""
    return [binom_sf(c, n, p0) if c <= n else 0.0 for n, c in zip(b.ns, b.crit)]


def simulate_type1(p: float, ns: Sequence[int], n_max: int, alpha: float, p0: float, spending: str,
                   sims: int, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo check of the upper boundary: share of simulated gates that cross it when the true share is ``p``.

    ``ns`` fixed (no draws); the boundary is computed once (it only depends on ``ns``).
    """
    rng = np.random.default_rng(seed)
    final = [False] * (len(ns) - 1) + [True]
    b = upper_boundary(ns, n_max, alpha, p0, spending, final)
    incs = np.diff(np.concatenate([[0], np.asarray(ns, dtype=np.int64)]))
    wins = np.cumsum(np.stack([rng.binomial(int(m), p, size=sims) for m in incs], axis=1), axis=1)
    crossed = wins >= np.asarray(b.crit)[None, :]
    any_cross = crossed.any(axis=1)
    first = np.where(any_cross, crossed.argmax(axis=1), len(ns))
    return {"rate": float(any_cross.mean()), "exact": b.spent[-1], "mean_stop_look": float(np.mean(np.minimum(first, len(ns) - 1) + 1)),
            "crit": list(b.crit)}
