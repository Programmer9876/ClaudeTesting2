#!/usr/bin/env python3
"""Sequential verdict engine for the budgeted test queue (docs/QUEUE.md, docs/PRIORITY_PLAN.md step 1).

A pure module outside ``catanbot/`` (the code fingerprint does not change).  ``scripts/run_queue.py`` calls it at
every look; nothing else changes behaviour because of it (campaign.py and ablate_catanatron.py keep their legacy
rules and only *honour* the ``{"kind": "stop", "source": "queue"}`` records it produces).

Units
    Catanatron rows: ``x_s = won_cand - won_def`` in {-1, 0, 1} per seed, "pp (1v3 win rate vs <opp>)".
    Self-play rows:  ``x_g = diff`` in {-1/2, 0, +1/2} per game, "pp (per seat, 2v2)".

Looks
    K = 5 equal looks, ``n_k = k * N_max / 5`` rounded to a multiple of 4 (the last look is ``N_max``).
    ``delta_k = mean(x)``, ``se_k = max(s_k / sqrt(n_k), 1 / n_k)``, ``Z_k = delta_k / se_k``,
    ``T_k = (delta_k - delta_min) / se_k``.

Boundaries (equal looks, pinned by tests/test_seqtest.py)
    ADOPT  a = 4.877, 3.357, 2.680, 2.290, 2.031  (Lan-DeMets O'Brien-Fleming spending, one-sided 0.025)
    REJECT r = 2.66, 2.46, 2.24, 2.02, 1.79       (Hwang-Shih-DeCani gamma = -2, one-sided 0.05)
    :func:`boundaries_general` (Armitage-McPherson-Rowe recursion on a 0.01-sd grid) reproduces both and is
    what the control-variate estimator uses at unequal information fractions.

Designs (``design`` plan field; the default follows ``polarity``)
    screen    new ideas.  ADOPT if Z >= a_k (with an exact McNemar cross-check below 50 discordant pairs);
              REJECT if T <= -r_k (delta_min 0.01): 'worse' when Z <= -1.96 else 'futile'; at looks 2-4 an
              early SHELVE when the conditional power under the current trend is below 0.10
              (Z < sqrt(t)(a_K - 1.2816 sqrt(1-t)) = 0.66 / 0.95 / 1.30); SHELVE at the cap.
    knockout  a flag switched off / a weight at 0 / a cheaper search.  Same boundaries, delta_min 0:
              ADOPT -> REMOVE (the term hurts), REJECT -> KEEP(proven), SHELVE -> KEEP(unproven).
    estimate  measurement rows: one look at N, delta +- 1.96 se, no verdict (ESTIMATE).  An A/A row
              (``aa``) PASSes at 400 pairs with 0 diverged pairs; any diverged pair is FAILED(nondeterminism).
    politics  the pre-registered fixed-N screen: one look at N, fixed-sample two-sided p (SCREENED); an ALERT
              at 400 pairs with 0 diverged pairs (possible wiring problem), no interim stop.  The label
              (SIGNIFICANT / INCONCLUSIVE) comes from Holm within the row's tier (run_queue.py report).
    confirm   one look, one-sided 0.05: CONFIRMED / NOT CONFIRMED.
    legacy    ablate_catanatron.should_stop, unchanged (the queue only passes --stop-at-se through).

Other outcomes
    NOOP      the Clopper-Pearson 95% upper bound of the diverged share is below ``noop_share`` (0.01: 0 of 400
              gives 0.0092): the change cannot move the win rate by 1 pp.  FAILED(inert-harness) instead when
              the row's area needs a harness mode and the mechanic never fired in either arm (trades: adapter
              offers + offers_received = 0; counting: info_samples = 0), so the row is re-queued, never lost.
    FAILED    errors > 1% of the prefix; inconsistent pairs > max(2, 2% of diverged); more than one code among the
              candidate records (code-mixed); identical traces with different winners; (plan changes: queue).

Precedence at a look: FAILED > ADOPT > NOOP > REJECT > SHELVE-early > look-1 promise re-check > continue.
(The design lists REJECT before NOOP; with the se floor a zero-divergence row would then always end
REJECT(futile) and the NOOP / inert-harness labels could never fire, so NOOP is checked first.  NOOP implies
|delta| < 1 pp, so no REJECT(worse) is hidden by it.)

Promise: at a terminal non-ADOPT, a naive upper 95% bound below ``promise_pp`` adds 'promise not met'; an ADOPT
with delta < promise / 2 is flagged 'small'.  Estimates of rows stopped early are biased away from 0 (winner's
curse): every verdict carries ``stopped_early``.

Stage-wise p (Holm, confirmations): the null probability of crossing ADOPT at an earlier look plus
P0(no earlier crossing, Z_k >= z_obs); it equals the fixed-sample p for one-look designs.

CLI (no games):  ``python3 scripts/seqtest.py --tables`` | ``--oc D DELTA [--design knockout]`` | ``--grid``.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Constants (pinned by tests)
# ---------------------------------------------------------------------------
K_LOOKS = 5
ADOPT_TABLE = (4.877, 3.357, 2.680, 2.290, 2.031)      # LD-OBF, one-sided 0.025, 5 equal looks
REJECT_TABLE = (2.66, 2.46, 2.24, 2.02, 1.79)          # HSD gamma -2, one-sided 0.05, 5 equal looks
ALPHA_ADOPT = 0.025
ALPHA_REJECT = 0.05
HSD_GAMMA = -2.0
Z_CP = 1.2816                                          # z_{0.90}: conditional power 0.10
CP_FUTILITY = 0.10
AA_PAIRS = 400                                         # A/A rows PASS here with 0 diverged pairs
ALERT_PAIRS = 400                                      # politics: ALERT at this look with 0 diverged pairs
ERROR_SHARE = 0.01                                     # FAILED(errors) above this share of the prefix
INCONSISTENT_SHARE = 0.02
MCNEMAR_BELOW = 50                                     # exact cross-check under this many discordant pairs
NOOP_SHARE = 0.01                                      # NOOP when the CP upper bound of the diverged share < this
Z95 = 1.959964
CONFIRM_Z = 1.644854                                   # one-sided 0.05
LOOK1_POWER_FLOOR = 0.30                               # re-check after look 1 at the observed discordance
UNPROVEN_POWER = 0.50                                  # intake: below this power a row is not run alone

TERMINAL = {"ADOPT", "REJECT", "SHELVE", "NOOP", "FAILED", "REMOVE", "KEEP", "ESTIMATE", "PASS", "SCREENED",
            "CONFIRMED", "NOT CONFIRMED", "LEGACY"}


def _load_stats():
    """``catanbot/league/stats.py`` (exact binomial tails, norm cdf/ppf, Holm) - import only, unchanged."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from catanbot.league import stats   # noqa: WPS433  (pure Python, no numpy)
    return stats


_STATS = None


def S():
    global _STATS
    if _STATS is None:
        _STATS = _load_stats()
    return _STATS


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_ppf(q: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(q)


# ---------------------------------------------------------------------------
# Look schedule
# ---------------------------------------------------------------------------
def look_sizes(n_max: int, looks: int = K_LOOKS, multiple: int = 4) -> List[int]:
    """``n_k = k * n_max / looks`` rounded to a multiple of ``multiple`` (the last look is ``n_max``)."""
    n_max = int(n_max)
    out = []
    for k in range(1, looks + 1):
        if k == looks:
            n = n_max
        else:
            n = int(round(k * n_max / looks / multiple)) * multiple
        n = max(multiple if n_max >= multiple else n_max, min(n, n_max))
        if out and n <= out[-1]:
            n = min(n_max, out[-1] + multiple)
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Design:
    name: str
    looks: int = K_LOOKS
    delta_min: float = 0.01
    adopt: Tuple[float, ...] = ADOPT_TABLE
    reject: Tuple[float, ...] = REJECT_TABLE
    early_shelve: bool = True
    noop_share: Optional[float] = NOOP_SHARE
    labels: Tuple[Tuple[str, str], ...] = (("ADOPT", "ADOPT"), ("REJECT", "REJECT"), ("SHELVE", "SHELVE"))
    vp_guard: bool = False          # optional: an early SHELVE also needs the VP trend to be unpromising

    def label(self, what: str) -> str:
        return dict(self.labels).get(what, what)


DESIGNS: Dict[str, Design] = {
    "screen": Design("screen"),
    "knockout": Design("knockout", delta_min=0.0,
                       labels=(("ADOPT", "REMOVE"), ("REJECT", "KEEP(proven)"), ("SHELVE", "KEEP(unproven)"))),
    "estimate": Design("estimate", looks=1, early_shelve=False, noop_share=None),
    "politics": Design("politics", looks=1, early_shelve=False, noop_share=None),
    "confirm": Design("confirm", looks=1, early_shelve=False, noop_share=None),
    "legacy": Design("legacy", looks=1, early_shelve=False, noop_share=None),
}
POLARITY_DESIGN = {"new": "screen", "knockout": "knockout", "measure": "estimate"}


def design_of(name: str) -> Design:
    try:
        return DESIGNS[name]
    except KeyError:
        raise ValueError(f"unknown design {name!r} (choose from {sorted(DESIGNS)})") from None


def shelve_thresholds(adopt: Sequence[float] = ADOPT_TABLE) -> List[Optional[float]]:
    """Early-SHELVE thresholds on Z at each look: conditional power under the current trend < 0.10
    (``Z < sqrt(t) (a_K - 1.2816 sqrt(1 - t))``) at looks 2 .. K-1; None at look 1 and at the cap."""
    kk = len(adopt)
    out: List[Optional[float]] = []
    for k in range(1, kk + 1):
        t = k / kk
        out.append(math.sqrt(t) * (adopt[-1] - Z_CP * math.sqrt(1.0 - t)) if 1 < k < kk else None)
    return out


SHELVE_TABLE = tuple(round(x, 4) if x is not None else None for x in shelve_thresholds())


def conditional_power(z: float, t: float, a_final: float = ADOPT_TABLE[-1]) -> float:
    """P(Z at the cap >= a_final | Z_t = z) under the current trend (drift = z / sqrt(t))."""
    if t >= 1.0:
        return 1.0 if z >= a_final else 0.0
    return norm_sf((a_final - z / math.sqrt(t)) / math.sqrt(1.0 - t))


# ---------------------------------------------------------------------------
# General Lan-DeMets boundaries (Armitage-McPherson-Rowe recursion)
# ---------------------------------------------------------------------------
def spend(kind: str, t: float, alpha: float, gamma: float = HSD_GAMMA) -> float:
    """Cumulative one-sided alpha spent at information fraction ``t``."""
    if t <= 0.0:
        return 0.0
    t = min(1.0, t)
    if kind == "obf":
        return 2.0 * norm_sf(norm_ppf(1.0 - alpha / 2.0) / math.sqrt(t))
    if kind == "hsd":
        if gamma == 0.0:
            return alpha * t
        return alpha * (1.0 - math.exp(-gamma * t)) / (1.0 - math.exp(-gamma))
    if kind == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    raise ValueError(f"unknown spending function {kind!r}")


def _simpson_weights(n: int, h: float):
    import numpy as np
    if n < 3:
        return np.full(n, h)
    w = np.ones(n)
    if n % 2 == 1:
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w * h / 3.0
    # even count: Simpson on the first n-1 points, trapezoid on the last interval
    w[1:-2:2] = 4.0
    w[2:-2:2] = 2.0
    w = w * h / 3.0
    w[-2] += h / 2.0 - h / 3.0
    w[-1] = h / 2.0
    return w


class _Recursion:
    """Sub-density of Z_k on the continuation region ``Z_j < c_j`` (j <= k), on a grid of step ``step``."""

    def __init__(self, step: float = 0.01, lower: float = -9.0):
        self.step = step
        self.lower = lower
        self.t_prev = 0.0
        self.grid = None
        self.dens = None

    def cross_prob(self, t: float, c: float) -> float:
        """P0(Z_t >= c, no earlier crossing)."""
        import numpy as np
        if self.grid is None:
            return norm_sf(c)
        sd = math.sqrt(t - self.t_prev)
        arg = (c * math.sqrt(t) - self.grid * math.sqrt(self.t_prev)) / sd
        return float(np.sum(self.w * self.dens * _erfc_half(arg)))

    def advance(self, t: float, c: float) -> None:
        """Move to look ``t`` with continuation region ``z < c``."""
        import numpy as np
        hi = min(c, 12.0)
        n = max(3, int(math.ceil((hi - self.lower) / self.step)) + 1)
        grid = np.linspace(self.lower, hi, n)
        h = grid[1] - grid[0]
        if self.grid is None:
            dens = np.exp(-0.5 * grid * grid) / math.sqrt(2.0 * math.pi)
        else:
            sd = math.sqrt(t - self.t_prev)
            st, sp = math.sqrt(t), math.sqrt(self.t_prev)
            # f_new(z) = int f(u) phi((z st - u sp) / sd) st / sd du
            diff = (grid[:, None] * st - self.grid[None, :] * sp) / sd
            ker = np.exp(-0.5 * diff * diff) / math.sqrt(2.0 * math.pi) * (st / sd)
            dens = ker @ (self.w * self.dens)
        self.grid, self.dens, self.w = grid, dens, _simpson_weights(n, h)
        self.t_prev = t


def _erfc_half(x):
    """0.5 * erfc(x / sqrt 2) = upper normal tail, vectorised without scipy (math.erfc per element: the
    polynomial approximations are too coarse for the 1e-7 tails of the first O'Brien-Fleming look)."""
    return _ERFC(x / math.sqrt(2.0)) * 0.5


def _make_erfc():
    import numpy as np
    return np.frompyfunc(math.erfc, 1, 1)


class _Erfc:
    def __init__(self):
        self.f = None

    def __call__(self, x):
        import numpy as np
        if self.f is None:
            self.f = _make_erfc()
        return np.asarray(self.f(x), dtype=float)


_ERFC = _Erfc()


def boundaries_general(ts: Sequence[float], alpha: float, kind: str = "obf", gamma: float = HSD_GAMMA,
                       step: float = 0.01) -> List[float]:
    """One-sided boundaries ``c_k`` for a standard Brownian motion observed at information fractions ``ts``
    (increasing, the last usually 1) that spend ``alpha`` with the Lan-DeMets function ``kind``
    (obf | hsd | pocock).  Each look spends exactly the increment of the spending function (numerically, by
    bisection on the Armitage-McPherson-Rowe recursion).  Non-binding: other boundaries are ignored."""
    rec = _Recursion(step=step)
    out: List[float] = []
    spent = 0.0
    for t in ts:
        target = spend(kind, t, alpha, gamma)
        inc = max(0.0, target - spent)
        if inc <= 1e-15:
            c = 12.0
        elif rec.grid is None:
            c = norm_ppf(1.0 - inc)
        else:
            lo, hi = -3.0, 12.0
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                if rec.cross_prob(t, mid) > inc:
                    lo = mid
                else:
                    hi = mid
            c = 0.5 * (lo + hi)
        spent += rec.cross_prob(t, c) if rec.grid is not None else norm_sf(c)
        out.append(c)
        rec.advance(t, c)
    return out


def crossing_probs(bounds: Sequence[float], ts: Sequence[float], step: float = 0.01) -> List[float]:
    """Null probability of first crossing the upper boundary at each look."""
    rec = _Recursion(step=step)
    out = []
    for t, c in zip(ts, bounds):
        out.append(rec.cross_prob(t, c))
        rec.advance(t, c)
    return out


@functools.lru_cache(maxsize=64)
def _cached_cross(bounds: Tuple[float, ...], ts: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(crossing_probs(bounds, ts))


def stagewise_p(k: int, z_obs: float, bounds: Sequence[float] = ADOPT_TABLE,
                ts: Optional[Sequence[float]] = None) -> float:
    """Stage-wise ordering p-value at look ``k`` (1-based): the null probability of crossing at an earlier look
    plus P0(no earlier crossing, Z_k >= z_obs).  One look: the fixed-sample one-sided p."""
    if ts is None:
        ts = [(j + 1) / len(bounds) for j in range(len(bounds))]
    if k <= 1:
        return norm_sf(z_obs)
    earlier = _cached_cross(tuple(float(b) for b in bounds[:k - 1]), tuple(float(t) for t in ts[:k - 1]))
    rec = _Recursion()
    for t, c in zip(ts[:k - 1], bounds[:k - 1]):
        rec.advance(t, c)
    return min(1.0, sum(earlier) + rec.cross_prob(ts[k - 1], z_obs))


# ---------------------------------------------------------------------------
# Look statistics
# ---------------------------------------------------------------------------
@dataclass
class LookStats:
    """Everything a look needs, from the complete pairs of the seed prefix (see :func:`look_units`)."""
    n_target: int                   # prefix size n_k
    ok: int = 0                     # complete (ok, ok) pairs
    errors: int = 0                 # pairs with an error record
    incomplete: int = 0             # seeds of the prefix without a pair yet
    sum_x: float = 0.0
    sum_x2: float = 0.0
    plus: int = 0                   # x > 0 (candidate only won)
    minus: int = 0                  # x < 0 (default only won)
    sum_vp: float = 0.0
    sum_vp2: float = 0.0
    known: int = 0                  # pairs with trace fingerprints
    identical: int = 0
    diverged: int = 0
    inconsistent: int = 0
    ident_diff_winner: int = 0
    cand_codes: Tuple[str, ...] = ()
    adapter_cand: Dict[str, float] = field(default_factory=dict)
    adapter_def: Dict[str, float] = field(default_factory=dict)
    win_def: int = 0
    unit: float = 1.0               # 1 for Catanatron pairs, 0.5 for 2v2 self-play games

    @property
    def complete(self) -> bool:
        return self.incomplete == 0 and self.ok + self.errors >= self.n_target

    @property
    def n(self) -> int:
        return self.ok

    @property
    def delta(self) -> float:
        return self.sum_x / self.ok if self.ok else float("nan")

    @property
    def sd(self) -> float:
        n = self.ok
        if n < 2:
            return float("nan")
        var = (self.sum_x2 - self.sum_x * self.sum_x / n) / (n - 1)
        return math.sqrt(max(0.0, var))

    @property
    def se(self) -> float:
        n = self.ok
        if n < 1:
            return float("nan")
        sd = self.sd
        raw = sd / math.sqrt(n) if not math.isnan(sd) else 0.0
        return max(raw, self.unit / n)

    @property
    def vp_delta(self) -> float:
        return self.sum_vp / self.ok if self.ok else float("nan")

    @property
    def vp_se(self) -> float:
        n = self.ok
        if n < 2:
            return float("nan")
        var = (self.sum_vp2 - self.sum_vp * self.sum_vp / n) / (n - 1)
        return math.sqrt(max(0.0, var) / n)

    @property
    def discordant(self) -> int:
        return self.plus + self.minus

    @property
    def p_def(self) -> float:
        return self.win_def / self.ok if self.ok else float("nan")

    def add(self, x: float, vp: float = 0.0, won_def: bool = False) -> None:
        self.ok += 1
        self.sum_x += x
        self.sum_x2 += x * x
        self.sum_vp += vp
        self.sum_vp2 += vp * vp
        if x > 0:
            self.plus += 1
        elif x < 0:
            self.minus += 1
        if won_def:
            self.win_def += 1

    def summary(self) -> Dict[str, Any]:
        return {"n": self.ok, "errors": self.errors, "delta": _r(self.delta), "se": _r(self.se),
                "vp_delta": _r(self.vp_delta), "vp_se": _r(self.vp_se), "plus": self.plus, "minus": self.minus,
                "identical": self.identical, "diverged": self.diverged, "inconsistent": self.inconsistent,
                "known": self.known, "p_def": _r(self.p_def)}


def _r(x: float, nd: int = 6) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def _p(p: Optional[float]) -> Optional[float]:
    """A p-value kept to 6 significant digits (tiny p-values keep their magnitude for Holm)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return None
    return float(f"{float(p):.6g}")


def _load_ablate():
    name = "_ablate_catanatron_for_seqtest"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "ablate_catanatron.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def look_units(index, run: Dict[str, Any], base: int, n: int, ab=None) -> LookStats:
    """The complete pairs of ``run`` in the seed prefix ``[base, base + n)`` of an ablate_catanatron Index.

    Error pairs are counted but excluded; seeds without a pair make the prefix incomplete (no look yet)."""
    ab = ab or _load_ablate()
    st = LookStats(n_target=n)
    codes = set()
    for s in range(base, base + n):
        c, d = index.pair(run, s)
        if c is None:
            st.incomplete += 1
            continue
        if c.get("status") != "ok" or d.get("status") != "ok":
            st.errors += 1
            continue
        wc, wd = bool(c.get("won")), bool(d.get("won"))
        st.add(float(int(wc) - int(wd)), float(c.get("our_vp", 0)) - float(d.get("our_vp", 0)), wd)
        codes.add(str(c.get("code")))
        div = ab.divergence(c, d)
        if div.get("known"):
            st.known += 1
            if div["identical"]:
                st.identical += 1
                if wc != wd:
                    st.ident_diff_winner += 1
            else:
                st.diverged += 1
                if not div["consistent"]:
                    st.inconsistent += 1
        for tgt, rec in ((st.adapter_cand, c), (st.adapter_def, d)):
            for k2, v in (rec.get("adapter") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    tgt[k2] = tgt.get(k2, 0) + v
    st.cand_codes = tuple(sorted(codes))
    return st


def selfplay_units(records: Iterable[Dict[str, Any]], n: Optional[int] = None,
                   one_candidate_only: bool = False) -> LookStats:
    """Self-play games (``ablate.py --json`` records, in game order) as look units: ``x_g = diff``.

    ``one_candidate_only`` keeps the games whose seat pattern has exactly one 'C' (the 1v2 no-harm check in
    3-player games) and uses ``x_g = c_g - d_g`` with ``c_g`` in {0, 1} and ``d_g`` in {0, 1/2}."""
    recs = list(records)
    if n is not None:
        recs = recs[:n]
    st = LookStats(n_target=len(recs), unit=0.5 if not one_candidate_only else 0.5)
    for r in recs:
        pat = r.get("pattern", "")
        if one_candidate_only and pat.count("C") != 1:
            continue
        if one_candidate_only:
            C = [i for i, x in enumerate(pat) if x == "C"]
            D = [i for i, x in enumerate(pat) if x == "D"]
            w = r.get("winner")
            c_g = 1.0 if w in C else 0.0
            d_g = (1.0 / len(D)) if w in D else 0.0
            x = c_g - d_g
        else:
            x = float(r.get("diff", 0.0))
        vps = r.get("vps") or []
        vp = 0.0
        if vps and pat:
            cc = [vps[i] for i, x2 in enumerate(pat) if x2 == "C"]
            dd = [vps[i] for i, x2 in enumerate(pat) if x2 == "D"]
            if cc and dd:
                vp = sum(cc) / len(cc) - sum(dd) / len(dd)
        st.add(x, vp, r.get("winning_side") == "default")
    st.n_target = st.ok
    return st


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    status: str                         # "continue" | "final"
    verdict: str = ""                   # ADOPT / REJECT / SHELVE / NOOP / FAILED / REMOVE / KEEP / ESTIMATE / ...
    reason: str = ""
    label: str = ""                     # e.g. "REJECT(worse)", "KEEP(unproven)", "FAILED(inert-harness)"
    look: int = 0
    pairs: int = 0
    delta: Optional[float] = None
    se: Optional[float] = None
    z: Optional[float] = None
    p: Optional[float] = None           # stage-wise one-sided p (two-sided for politics)
    design: str = ""
    flags: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> bool:
        return self.status == "final"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, "", [], {}) or k in ("status", "verdict")}


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    if n <= 0:
        return 1.0
    return S().clopper_pearson(k, n, conf)[1]


def mcnemar_p(plus: int, minus: int) -> float:
    """Exact one-sided p of ``plus`` candidate-only wins among the discordant pairs (p0 = 1/2)."""
    n = plus + minus
    if n <= 0:
        return 1.0
    return S().binom_sf(plus, n, 0.5)


def _final(design: Design, what: str, reason: str, k: int, st: LookStats, z: float, extra_flags=(),
           p: Optional[float] = None) -> Verdict:
    lab = design.label(what)
    base = lab.split("(")[0]
    label = lab if "(" in lab else (f"{lab}({reason})" if reason else lab)
    return Verdict(status="final", verdict=base, reason=reason, label=label, look=k, pairs=st.ok,
                   delta=_r(st.delta), se=_r(st.se), z=_r(z), p=_p(p),
                   design=design.name, flags=list(extra_flags), stats=st.summary())


def failed_checks(st: LookStats, aa: bool = False) -> Optional[str]:
    """The FAILED reason of a look, or None."""
    total = st.ok + st.errors
    if total and st.errors > ERROR_SHARE * total:
        return "errors"
    if st.ok < (1.0 - ERROR_SHARE) * st.n_target:
        return "errors"
    if len(st.cand_codes) > 1:
        return "code-mixed"
    if st.ident_diff_winner > 0:
        return "identical-trace-different-winner"
    if not aa and st.inconsistent > max(2, INCONSISTENT_SHARE * st.diverged):
        return "inconsistent"
    return None


def mechanic_fired(needs: Optional[str], st: LookStats) -> bool:
    """Did the harness exercise the row's mechanic at all (either arm)?"""
    if not needs:
        return True
    tot: Dict[str, float] = {}
    for d in (st.adapter_cand, st.adapter_def):
        for k, v in d.items():
            tot[k] = tot.get(k, 0) + v
    if needs == "trades":
        return tot.get("offers", 0) + tot.get("offers_received", 0) > 0
    if needs == "counting":
        return tot.get("info_samples", 0) > 0
    return True


def decide(st: LookStats, design: str, k: int, n_max: int, *, aa: bool = False, needs: Optional[str] = None,
           promise_pp: Optional[float] = None, power_fn: Optional[Callable[[float, float, int], float]] = None,
           fired: Optional[bool] = None) -> Verdict:
    """Verdict at look ``k`` (1-based) of a row with cap ``n_max`` from the prefix statistics ``st``.

    ``needs``: the harness mode the row's mechanic needs ('trades' | 'counting' | None) for FAILED(inert-harness);
    ``fired`` overrides the adapter-counter check.  ``promise_pp`` (percentage points) drives 'promise not met',
    the 'small' flag and, with ``power_fn(promise, D_obs, n_max)``, the non-binding re-check after look 1."""
    dz = design_of(design)
    se = st.se
    delta = st.delta
    z = delta / se if st.ok and se and not math.isnan(se) and se > 0 else 0.0
    fail = failed_checks(st, aa=aa)
    looks = look_sizes(n_max, dz.looks) if dz.looks > 1 else [n_max]
    last = k >= len(looks)
    promise = (promise_pp / 100.0) if promise_pp else None
    if fail:
        return _final(dz, "FAILED", fail, k, st, z)
    # ---- one-look designs ------------------------------------------------------------------------
    if dz.name == "estimate":
        if aa:
            if st.diverged > 0:
                return _final(dz, "FAILED", "nondeterminism", k, st, z)
            return _final(dz, "PASS", "A/A identical", k, st, z)
        return _final(dz, "ESTIMATE", "fixed N", k, st, z, p=2.0 * norm_sf(abs(z)))
    if dz.name == "politics":
        v = _final(dz, "SCREENED", "fixed N", k, st, z, p=2.0 * norm_sf(abs(z)))
        return v
    if dz.name == "confirm":
        what = "CONFIRMED" if z >= CONFIRM_Z else "NOT CONFIRMED"
        return _final(dz, what, "one-sided 0.05", k, st, z, p=norm_sf(z))
    if dz.name == "legacy":
        return _final(dz, "LEGACY", "legacy --stop-at-se", k, st, z)
    # ---- sequential designs (screen / knockout) --------------------------------------------------------
    a_k = dz.adopt[k - 1]
    r_k = dz.reject[k - 1]
    t_stat = (delta - dz.delta_min) / se if se else 0.0
    p_stage = stagewise_p(k, z, dz.adopt)
    flags: List[str] = []
    if not last:
        flags.append("stopped early")
    # ADOPT
    if z >= a_k:
        ok = True
        if st.discordant < MCNEMAR_BELOW:
            ok = mcnemar_p(st.plus, st.minus) <= norm_sf(a_k)
        if ok:
            if promise is not None and delta < promise / 2.0:
                flags.append("small")
            return _final(dz, "ADOPT", "", k, st, z, flags, p_stage)
    # NOOP / inert harness
    if dz.noop_share is not None and st.known >= st.ok and st.ok > 0:
        upper = clopper_pearson_upper(st.diverged, st.known)
        if upper < dz.noop_share:
            is_fired = mechanic_fired(needs, st) if fired is None else fired
            if needs and not is_fired:
                return _final(dz, "FAILED", "inert-harness", k, st, z, p=p_stage)
            return _final(dz, "NOOP", f"diverged share < {dz.noop_share:g} (CP upper {upper:.4f})", k, st, z,
                          flags, p_stage)
    # REJECT
    if t_stat <= -r_k:
        why = "worse" if z <= -Z95 else "futile"
        if promise is not None and delta + Z95 * se < promise:
            flags.append("promise not met")
        return _final(dz, "REJECT", why, k, st, z, flags, p_stage)
    # early SHELVE (futility, non-binding)
    thr = shelve_thresholds(dz.adopt)[k - 1]
    if dz.early_shelve and thr is not None and z < thr:
        vp_ok = True
        if dz.vp_guard and not math.isnan(st.vp_se) and st.vp_se > 0:
            vp_ok = st.vp_delta / st.vp_se < thr
        if vp_ok:
            tag = "no gain" if delta <= 0 else "too small to prove"
            if promise is not None and delta + Z95 * se < promise:
                flags.append("promise not met")
            return _final(dz, "SHELVE", f"{tag}; conditional power < {CP_FUTILITY:g}", k, st, z, flags, p_stage)
    # look-1 re-check of the promise at the observed discordance (non-binding futility)
    if k == 1 and not last and promise is not None and power_fn is not None and st.ok:
        d_obs = max(st.discordant / st.ok, 1e-3)
        pw = power_fn(promise_pp, d_obs, n_max)
        if pw < LOOK1_POWER_FLOOR:
            flags.append("promise not met") if delta + Z95 * se < promise else None
            return _final(dz, "SHELVE", f"too small to prove; unprovable at promise (power {pw:.2f} at D "
                                        f"{d_obs:.2f})", k, st, z, flags, p_stage)
    if last:
        tag = "no gain" if delta <= 0 else "too small to prove"
        if "stopped early" in flags:
            flags.remove("stopped early")
        if promise is not None and delta + Z95 * se < promise:
            flags.append("promise not met")
        return _final(dz, "SHELVE", f"{tag}; cap", k, st, z, flags, p_stage)
    v = Verdict(status="continue", look=k, pairs=st.ok, delta=_r(delta), se=_r(se), z=_r(z), p=_p(p_stage),
                design=dz.name, stats=st.summary())
    return v


def politics_alert(st: LookStats) -> Optional[str]:
    """Politics rows look once, but a 400-pair prefix with 0 diverged pairs raises an ALERT (wiring?)."""
    if st.known and st.diverged == 0:
        return f"ALERT: 0 diverged pairs in the first {st.ok} pairs (is the term reachable in this harness?)"
    return None


def stop_record(v: Verdict, run_key: str, row: str = "", value_text: str = "", extra: Optional[dict] = None
                ) -> Dict[str, Any]:
    """The ``{kind: stop}`` JSONL record carrying a queue verdict (ablate_catanatron honours ``source: queue``)."""
    rec = {"v": 1, "kind": "stop", "run_key": run_key, "pairs": v.pairs, "delta": v.delta, "se": v.se,
           "verdict": v.verdict, "label": v.label, "reason": v.reason, "look": v.look, "design": v.design,
           "p": v.p, "z": v.z, "flags": list(v.flags), "source": "queue", "row": row, "value_text": value_text,
           "t": time.time()}
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Remaining cost (scheduler) and operating characteristics (simulation)
# ---------------------------------------------------------------------------
def expected_remaining(design: str, n_done: int, n_max: int, z: Optional[float] = None, sd: float = 0.63,
                       paths: int = 2000, seed: int = 0, look_done: int = 0) -> float:
    """Expected pairs still to play.  Fixed designs: ``n_max - n_done``; a fresh sequential row: 0.49 n_max;
    a started one: ``paths`` Brownian continuations from (t_k, Z_k) with the current trend as drift."""
    import numpy as np
    dz = design_of(design)
    if dz.looks <= 1:
        return float(max(0, n_max - n_done))
    looks = look_sizes(n_max, dz.looks)
    if look_done <= 0 or z is None:
        return 0.49 * n_max if n_done <= 0 else max(0.0, 0.49 * n_max - n_done)
    rng = np.random.default_rng(seed)
    t0 = looks[look_done - 1] / n_max
    b = np.full(paths, z * math.sqrt(t0))
    drift = z / math.sqrt(t0)
    stop = np.full(paths, float(n_max))
    active = np.ones(paths, dtype=bool)
    thr = shelve_thresholds(dz.adopt)
    t_prev = t0
    for j in range(look_done, len(looks)):
        t = looks[j] / n_max
        b = b + drift * (t - t_prev) + rng.normal(0.0, math.sqrt(t - t_prev), paths)
        t_prev = t
        zj = b / math.sqrt(t)
        se_j = max(sd / math.sqrt(looks[j]), 1.0 / looks[j])
        tj = zj - dz.delta_min / se_j
        hit = (zj >= dz.adopt[j]) | (tj <= -dz.reject[j])
        if dz.early_shelve and thr[j] is not None:
            hit |= zj < thr[j]
        newly = active & hit
        stop[newly] = looks[j]
        active &= ~hit
    return float(max(0.0, stop.mean() - n_done))


def _binom_tail_table(nmax: int = MCNEMAR_BELOW):
    import numpy as np
    tab = np.zeros((nmax, nmax + 2))
    for n in range(nmax):
        for kk in range(n + 2):
            tab[n, kk] = S().binom_sf(kk, n, 0.5) if n > 0 else (1.0 if kk <= 0 else 0.0)
    return tab


_TAIL = None


def simulate(design: str, D: float, delta: float, n_max: int = 2000, rows: int = 4000, seed: int = 0,
             unit: float = 1.0) -> Dict[str, float]:
    """Operating characteristics of a sequential design by simulation (vectorised): per pair
    x = +1 w.p. (D + delta)/2, -1 w.p. (D - delta)/2, else 0 (``unit`` scales x: 0.5 for 2v2 self-play).
    Returns the share of each outcome and the mean number of pairs.  (NOOP is modelled with diverged =
    discordant pairs, i.e. conservatively; FAILED never happens here.)"""
    import numpy as np
    global _TAIL
    if _TAIL is None:
        _TAIL = _binom_tail_table()
    dz = design_of(design)
    looks = look_sizes(n_max, dz.looks) if dz.looks > 1 else [n_max]
    pp, pm = (D + delta) / 2.0, (D - delta) / 2.0
    if pp < 0 or pm < 0 or pp + pm > 1:
        raise ValueError(f"impossible cell D={D}, delta={delta}")
    rng = np.random.default_rng(seed)
    plus = np.zeros(rows)
    minus = np.zeros(rows)
    prev = 0
    outcome = np.full(rows, "", dtype=object)
    reason = np.full(rows, "", dtype=object)
    stop_n = np.full(rows, float(n_max))
    active = np.ones(rows, dtype=bool)
    thr = shelve_thresholds(dz.adopt)
    for j, n in enumerate(looks):
        m = n - prev
        prev = n
        draws = rng.multinomial(m, [pp, pm, 1.0 - pp - pm], size=rows)
        plus += draws[:, 0]
        minus += draws[:, 1]
        mean = (plus - minus) / n
        var = np.maximum(0.0, ((plus + minus) - n * mean * mean) / (n - 1)) * unit * unit
        se = np.maximum(np.sqrt(var / n), unit / n)
        dmean = mean * unit
        zz = dmean / se
        if dz.looks <= 1:
            if dz.name == "confirm":
                win = zz >= CONFIRM_Z
                outcome[active & win] = "CONFIRMED"
                outcome[active & ~win] = "NOT CONFIRMED"
            else:
                sig = np.abs(zz) >= Z95
                outcome[active & sig] = "SIGNIFICANT"
                outcome[active & ~sig] = "INCONCLUSIVE"
            active[:] = False
            break
        tt = (dmean - dz.delta_min) / se
        last = j == len(looks) - 1
        disc = (plus + minus).astype(int)
        adopt = zz >= dz.adopt[j]
        few = adopt & (disc < MCNEMAR_BELOW)
        if few.any():
            idx = np.where(few)[0]
            pv = _TAIL[disc[idx], plus[idx].astype(int)]
            adopt[idx] = pv <= norm_sf(dz.adopt[j])
        a = active & adopt
        outcome[a] = "ADOPT"
        stop_n[a] = n
        active &= ~a
        if dz.noop_share is not None:
            # CP upper (0.975) of diverged share with diverged ~ discordant: only 0 discordant can pass at n>=400
            k0 = disc == 0
            cp_hi = 1.0 - 0.025 ** (1.0 / n)
            noop = active & k0 & (cp_hi < dz.noop_share)
            outcome[noop] = "NOOP"
            stop_n[noop] = n
            active &= ~noop
        rej = active & (tt <= -dz.reject[j])
        outcome[rej] = "REJECT"
        reason[rej & (zz <= -Z95)] = "worse"
        reason[rej & (zz > -Z95)] = "futile"
        stop_n[rej] = n
        active &= ~rej
        if dz.early_shelve and thr[j] is not None:
            sh = active & (zz < thr[j])
            outcome[sh] = "SHELVE"
            stop_n[sh] = n
            active &= ~sh
        if last:
            outcome[active] = "SHELVE"
            stop_n[active] = n
            active[:] = False
    res = {"adopt": float(np.mean(outcome == "ADOPT")), "reject": float(np.mean(outcome == "REJECT")),
           "shelve": float(np.mean(outcome == "SHELVE")), "noop": float(np.mean(outcome == "NOOP")),
           "reject_worse": float(np.mean(reason == "worse")), "mean_pairs": float(np.mean(stop_n)),
           "rows": rows}
    if dz.looks <= 1:
        res.update({"significant": float(np.mean(outcome == "SIGNIFICANT")),
                    "confirmed": float(np.mean(outcome == "CONFIRMED"))})
    return res


# The operating-characteristic grid published in the design (screen, 2,000-pair cap): P(ADOPT) at the promised
# effect.  tests/test_seqtest.py checks it against a fresh simulation (+- 0.03); intake reads
# :func:`power_at`, which simulates a cell the first time it is asked for (about 0.3 s) and caches it.
POWER_TABLE = {
    0.40: {1: 0.08, 2: 0.23, 3: 0.48, 4: 0.74, 5: 0.89},
    0.30: {2: 0.33, 3: 0.62, 4: 0.84},
    0.25: {2: 0.36, 3: 0.69, 4: 0.90, 5: 0.98},
    0.10: {1: 0.24, 2: 0.73, 3: 0.97},
    0.04: {1: 0.54, 2: 0.98},
}


@functools.lru_cache(maxsize=512)
def _power_cell(promise_pp: float, D: float, n_max: int, design: str, rows: int, seed: int) -> Tuple[float, float]:
    delta = promise_pp / 100.0
    D = max(D, abs(delta) + 1e-9)
    res = simulate(design, D, delta, n_max, rows=rows, seed=seed)
    return res["adopt"], res["mean_pairs"]


def power_at(promise_pp: float, D: float, n_max: int = 2000, design: str = "screen", rows: int = 4000,
             seed: int = 20260926) -> float:
    """P(ADOPT) of ``design`` at a true effect of ``promise_pp`` percentage points and discordance ``D``
    (cached simulation; deterministic)."""
    return _power_cell(round(float(promise_pp), 4), round(float(D), 4), int(n_max), design, rows, seed)[0]


def mean_pairs_at(promise_pp: float, D: float, n_max: int = 2000, design: str = "screen", rows: int = 4000,
                  seed: int = 20260926) -> float:
    return _power_cell(round(float(promise_pp), 4), round(float(D), 4), int(n_max), design, rows, seed)[1]


# ---------------------------------------------------------------------------
# Control variate against an external default-arm pool (F7, tooling only)
# ---------------------------------------------------------------------------
def cv_variance_ratio(D: float, p: float, n: int, m: int) -> float:
    """Variance of the CV estimator relative to the paired one at ``n`` pairs with a pool of ``m`` games:
    ``[D - c + c n / M] / D`` with ``c = D^2 / (4 p (1 - p))`` (the design's formula)."""
    c = D * D / (4.0 * p * (1.0 - p))
    return (D - c + c * n / m) / D


def cv_stat(x: Sequence[float], d: Sequence[float], pool_mean: float, pool_m: int) -> Dict[str, float]:
    """theta = xbar - beta (dbar - p_pool), beta = Cov(x, d) / Var(d) on the prefix;
    Var = [Var(x) - Cov(x, d)^2 / Var(d)] / n + beta^2 p_pool (1 - p_pool) / M."""
    n = len(x)
    if n < 3:
        raise ValueError("cv_stat needs at least 3 pairs")
    xb = sum(x) / n
    db = sum(d) / n
    vx = sum((a - xb) ** 2 for a in x) / (n - 1)
    vd = sum((b - db) ** 2 for b in d) / (n - 1)
    cxd = sum((a - xb) * (b - db) for a, b in zip(x, d)) / (n - 1)
    beta = cxd / vd if vd > 0 else 0.0
    theta = xb - beta * (db - pool_mean)
    resid = max(vx - (cxd * cxd / vd if vd > 0 else 0.0), 0.0)
    var = resid / n + beta * beta * pool_mean * (1.0 - pool_mean) / pool_m
    return {"theta": theta, "var": var, "beta": beta, "resid_var": resid, "dbar": db, "n": n,
            "paired_var": vx / n}


def cv_pool_mismatch(dbar: float, n: int, pool_mean: float, pool_m: int, limit_se: float = 3.0) -> bool:
    """FAILED(pool-mismatch): the row's default prefix mean and the pool mean differ by more than 3 se."""
    p = pool_mean
    se = math.sqrt(max(p * (1 - p), 1e-9) * (1.0 / max(n, 1) + 1.0 / max(pool_m, 1)))
    return abs(dbar - pool_mean) > limit_se * se


def cv_information_fractions(vars_k: Sequence[float], var_final: float) -> List[float]:
    """Information fractions of the CV looks: t_k = Var_K / Var_k (capped at 1)."""
    return [min(1.0, var_final / v) if v > 0 else 1.0 for v in vars_k]


def _cv_population(D: float, delta: float, p: float) -> Tuple[float, float, float]:
    """Var(x), Cov(x, d), Var(d) of one pair under the flip model (default wins w.p. p; given the default, the
    candidate flips so that P(x=+1) = (D + delta)/2, P(x=-1) = (D - delta)/2)."""
    pp, pm = (D + delta) / 2.0, (D - delta) / 2.0
    ex = pp - pm
    vx = pp + pm - ex * ex
    # x = -1 only when d = 1, x = +1 only when d = 0: E[x d] = -pm
    cxd = -pm - ex * p
    vd = p * (1.0 - p)
    return vx, cxd, vd


def cv_design_boundaries(D: float, p: float, n_max: int, m: int, looks: int = K_LOOKS,
                         alpha: float = ALPHA_ADOPT) -> Tuple[List[float], List[float]]:
    """Pre-declared CV design: information fractions ``t_k = Var_K / Var_k`` from the row's class (D prior, base
    rate, pool size) and the Lan-DeMets O'Brien-Fleming boundaries at those fractions."""
    vx, cxd, vd = _cv_population(D, 0.0, p)
    resid = vx - cxd * cxd / vd
    beta = cxd / vd
    c = beta * beta * p * (1.0 - p) / m
    ns = look_sizes(n_max, looks)
    vs = [resid / n + c for n in ns]
    ts = cv_information_fractions(vs, vs[-1])
    return ts, boundaries_general(ts, alpha, "obf")


def simulate_cv(D: float, delta: float, p: float, m: int, n_max: int = 2000, rows: int = 2000, seed: int = 0,
                paired: bool = False) -> Dict[str, float]:
    """ADOPT rate of the sequential CV screen (``paired=True``: the same data with the paired statistic and the
    fixed table), futility and REJECT left out (they only lower false ADOPT)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    d = (rng.random((rows, n_max)) < p).astype(np.int8)
    a = min(1.0, max(0.0, (D + delta) / (2.0 * (1.0 - p))))
    b = min(1.0, max(0.0, (D - delta) / (2.0 * p)))
    u = rng.random((rows, n_max))
    c = np.where(d == 1, (u >= b).astype(np.int8), (u < a).astype(np.int8))
    x = (c - d).astype(np.float64)
    df = d.astype(np.float64)
    pool = rng.binomial(m, p, size=rows) / m
    ns = look_sizes(n_max)
    if paired:
        bounds = list(ADOPT_TABLE)
    else:
        _, bounds = cv_design_boundaries(D, p, n_max, m)
    adopt = np.zeros(rows, dtype=bool)
    stop = np.full(rows, float(n_max))
    for j, n in enumerate(ns):
        xs, ds = x[:, :n], df[:, :n]
        xb, db = xs.mean(1), ds.mean(1)
        vx = xs.var(1, ddof=1)
        vd = ds.var(1, ddof=1)
        cxd = ((xs - xb[:, None]) * (ds - db[:, None])).sum(1) / (n - 1)
        if paired:
            z = xb / np.maximum(np.sqrt(vx / n), 1.0 / n)
        else:
            beta = np.where(vd > 0, cxd / np.where(vd > 0, vd, 1.0), 0.0)
            theta = xb - beta * (db - pool)
            var = np.maximum(vx - np.where(vd > 0, cxd * cxd / np.where(vd > 0, vd, 1.0), 0.0), 0.0) / n \
                + beta * beta * pool * (1.0 - pool) / m
            z = theta / np.sqrt(np.maximum(var, 1e-12))
        hit = (~adopt) & (stop == n_max) & (z >= bounds[j])
        adopt |= hit
        stop[hit] = n
    return {"adopt": float(adopt.mean()), "rows": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true", help="print the boundary tables (published and recomputed)")
    ap.add_argument("--oc", nargs=2, type=float, metavar=("D", "DELTA"), help="simulate one cell")
    ap.add_argument("--design", default="screen")
    ap.add_argument("--n-max", type=int, default=2000)
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--grid", action="store_true", help="the power grid of the design (P(ADOPT) at +1..+5 pp)")
    args = ap.parse_args(argv)
    if args.tables:
        ts = [0.2, 0.4, 0.6, 0.8, 1.0]
        a = boundaries_general(ts, ALPHA_ADOPT, "obf")
        r = boundaries_general(ts, ALPHA_REJECT, "hsd", HSD_GAMMA)
        print("ADOPT  published " + " ".join(f"{x:.3f}" for x in ADOPT_TABLE) + "  recomputed "
              + " ".join(f"{x:.3f}" for x in a))
        print("REJECT published " + " ".join(f"{x:.2f}" for x in REJECT_TABLE) + "  recomputed "
              + " ".join(f"{x:.2f}" for x in r))
        print("SHELVE early (looks 2-4) " + " ".join(f"{x:.2f}" for x in SHELVE_TABLE if x is not None))
    if args.oc:
        res = simulate(args.design, args.oc[0], args.oc[1], args.n_max, rows=args.rows)
        print(json.dumps(res, indent=1))
    if args.grid:
        for D, row in POWER_TABLE.items():
            cells = " ".join(f"+{pp}pp {power_at(pp, D, args.n_max):.2f} (pub {pub:.2f})" for pp, pub in row.items())
            print(f"D {D:.2f}: {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
