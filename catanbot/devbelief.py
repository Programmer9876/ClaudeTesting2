"""Reading opponents' held development cards (docs/PRIORITY_PLAN.md, card counting, step 6).

A card that stays in a hand while its owner could have played it is less and less likely to be a
card that gets played: bots play a knight, Year of Plenty, Monopoly or Road Building on the first
eligible turn 82-85% of the time, and a VP card is never played.  So a card held for many turns
is most likely a VP card (a secret leader) - or, for a human, a Monopoly waiting for a payoff.
:class:`DevAgeModel` turns that into a posterior over the types of every opponent's hidden cards.

**Bookkeeping** (always on, RNG-free, and it can never raise into a tracker): per player the
purchase own-turn index of every hidden unplayed card (``recs``; type-less records, oldest first),
one record per own turn (``turns``: no dev card played, the knight context, the Monopoly payoff,
the highest public VP during the turn) and, per type, the ``(purchase, play)`` turn interval of
every played card (``cov``: a play removes the youngest eligible record).  No hazard is baked in:
the likelihood is computed lazily at query time from the module values below, so the ParamBot
overrides of a candidate arm reach it (``tuning.apply`` sets them around the tracker calls).

**Likelihood** (per type present, with a floor): an assignment of types to ``j``'s records has
``L_j = prod over the non-VP types t it contains of F_t``, ``F_t = (1 - EPS) prod_{tau in S_t}
(1 - h_t(tau)) + EPS``, where ``S_t`` is the set of completed own turns of ``j`` in which the
OLDEST record assigned ``t`` was eligible, ``j`` played no development card, and no played card of
type ``t`` was being held (its coverage interval).  A second knight does not make a no-play turn
twice as unlikely (measured: 0.907 vs 0.891 play rate).  Knights use ``H_KNIGHT_CTX`` in a turn
that started with the knight context: ``j`` the main robber victim with a block of at least
``KCTX_MIN_PIPS`` demand-weighted pips (the robber design's R3 event), or one knight short of
Largest Army.

**Win caps** (history aware): a holder at public ``p`` during one of its own turns holds at most
``V - 1 - p`` VP cards among the cards it held then (it would have won).

**Joint posterior** over every unknown holder's records, dealt without replacement from the
public pool ``pi``: ``P(A) ~ prod_t fall(pi_t, N_t) prod_j L_j(A_j) [caps]``.  It is sampled and
summarised exactly by dynamic programming: per player a forward pass over its records (state =
the set of non-VP types present and the type counts), then a convolution over players by the
cards used so far, then backward sampling.  Tables are cached per (model version, parameters,
pool, public VP).  Beyond ``TABLE_BUDGET`` entries the model falls back to uniform dealing.

Switch: ``devbelief.ENABLED`` (a 0/1 weight, default 0).  With 0 nothing here is consulted by the
bot and the dealing is byte-identical; with 1 :meth:`catanbot.public_belief.PublicBelief.determinize`
keeps the uniform deal on the shared random stream and then overwrites the dealt cards from a side
stream copied from the pre-deal state (RNG-neutral: the hands, the search seeds and the deck order
stay common to both arms).  The advisor (``--dev-model bot|human``) reads the same posterior.

Python only: ``counting.expected_hidden_vp`` and ``static_value`` (mirrored in C++) are untouched.
"""
from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import board as B

__all__ = ["DevAgeModel", "Preset", "BOT", "HUMAN", "UNIFORM", "preset", "register_tunables"]

# --- tunables (read at query time, never at bookkeeping time) --------------------------------------
ENABLED = 0          # 1 = counted-mode determinizations deal opponents' dev cards from the posterior
H = 0.85             # per-turn play hazard of a held non-VP card (bot preset)
H_KNIGHT_CTX = 0.85  # knight hazard in a knight-context turn (equal to H = inactive; R3 candidate 0.95)
EPS = 0.03           # per-type floor of the no-play likelihood (hoarders: Catanatron keeps Road Building)

# --- constants -------------------------------------------------------------------------------------
KCTX_MIN_PIPS = 3.0     # the robber design's EVIDENCE_MIN_PIPS (= robber_eval.KICK_MIN's default)
TABLE_BUDGET = 200_000  # DP / convolution entries before falling back to uniform dealing
_K, _VP, _RB, _YOP, _MONO = B.DEV_KNIGHT, B.DEV_VP, B.DEV_ROAD_BUILDING, B.DEV_YEAR_OF_PLENTY, B.DEV_MONOPOLY
_PLAYABLE = tuple(t for t in range(5) if t != _VP)
_BIT = {t: 1 << i for i, t in enumerate(_PLAYABLE)}
_ZERO = (0, 0, 0, 0, 0)

# turn record fields
_NOPLAY, _KPV, _LA, _PAY, _MAXPUB = range(5)


class Preset:
    """Per-turn play hazards ``h_t(turn record)`` and the floor ``eps``."""

    def __init__(self, name: str):
        self.name = name

    def params(self) -> Tuple:
        """Cache key of the current values (module values are read now: lazy under overrides)."""
        if self.name == "bot":
            return ("bot", float(H), float(H_KNIGHT_CTX), float(EPS), float(KCTX_MIN_PIPS))
        return (self.name,)

    def eps(self) -> float:
        if self.name == "bot":
            return float(EPS)
        if self.name == "human":
            return 0.10
        return 0.0

    def hazard(self, t: int, rec: Sequence) -> float:
        kctx = bool(rec[_LA]) or float(rec[_KPV]) >= KCTX_MIN_PIPS
        if self.name == "bot":
            return float(H_KNIGHT_CTX) if (t == _K and kctx) else float(H)
        if self.name == "human":   # unfitted priors (advisor only; docs/PRIORITY_PLAN.md advisor.dev_read)
            if t == _K:
                return 0.9 if kctx else 0.25
            if t == _MONO:
                return 0.15 + 0.65 * min(1.0, max(0.0, (float(rec[_PAY]) - 4.0) / 4.0))
            return 0.6 if t == _YOP else 0.3
        return 0.0                 # uniform: no aging (the caps still apply)


BOT = Preset("bot")
HUMAN = Preset("human")
UNIFORM = Preset("uniform")


def preset(name: str) -> Preset:
    return {"bot": BOT, "human": HUMAN, "uniform": UNIFORM}[name]


def _fall(x: int, n: int) -> int:
    """Falling factorial x (x-1) ... (x-n+1); 0 when n > x."""
    if n > x:
        return 0
    out = 1
    for i in range(n):
        out *= x - i
    return out


def _fall_vec(pool: Sequence[int], used: Sequence[int], n: Sequence[int]) -> int:
    out = 1
    for t in range(5):
        if n[t]:
            f = _fall(pool[t] - used[t], n[t])
            if not f:
                return 0
            out *= f
    return out


def _add(u: Tuple[int, ...], n: Sequence[int]) -> Tuple[int, ...]:
    return tuple(u[t] + n[t] for t in range(5))


def _sub(u: Tuple[int, ...], n: Sequence[int]) -> Tuple[int, ...]:
    return tuple(u[t] - n[t] for t in range(5))


def _pick(items: Sequence, weights: Sequence[float], rng):
    tot = sum(weights)
    x = rng.random() * tot
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if x < acc:
            return it
    for it, w in zip(reversed(items), reversed(weights)):
        if w > 0:
            return it
    return items[-1]


class _Fallback(Exception):
    """No usable posterior (table budget, no consistent assignment, bookkeeping mismatch)."""


def _guarded(fn):
    """Bookkeeping entry points never raise: a failure invalidates the model (uniform dealing)."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kw):
        try:
            return fn(self, *args, **kw)
        except Exception:  # noqa: BLE001  (error isolation: the tracker must not see it)
            self.invalid = True
            self.stats["errors"] += 1
            return None
    return wrapper


class _Player:
    """One player's DP: records oldest first, per-record per-type factors, caps, forward layers."""

    __slots__ = ("recs", "F", "caps", "layers", "W")


class DevAgeModel:
    """Sufficient statistics of every player's hidden development cards, and the posterior over them.

    ``eligible_same_turn``: a card can be played in the turn it was bought (catanatron 3.2.1); the
    official rule (3.3, Colonist) makes it playable from the next own turn."""

    def __init__(self, n: int, eligible_same_turn: bool = False):
        self.n = int(n)
        self.eligible_same_turn = bool(eligible_same_turn)
        self.recs: List[List[int]] = [[] for _ in range(self.n)]
        self.turns: List[List[list]] = [[] for _ in range(self.n)]
        self.cov: List[List[List[Tuple[int, int]]]] = [[[] for _ in range(5)] for _ in range(self.n)]
        self.open: List[bool] = [False] * self.n
        self.version = 0
        self.invalid = False
        self.stats: Dict[str, int] = {"errors": 0, "fallbacks": 0, "dealt": 0, "padded": 0, "trimmed": 0}
        self.log_events = False
        self.events: List[list] = []
        self._cache: Dict[Tuple, Any] = {}

    # --- bookkeeping ----------------------------------------------------------------------------
    def _touch(self) -> None:
        self.version += 1
        if self._cache:
            self._cache.clear()

    def _log(self, *ev) -> None:
        if self.log_events:
            self.events.append(list(ev))

    def is_open(self, j: int) -> bool:
        return bool(self.open[j])

    def turn_index(self, j: int) -> int:
        """Index of ``j``'s open turn (or of its next turn when none is open)."""
        return len(self.turns[j]) - 1 if self.open[j] else len(self.turns[j])

    @_guarded
    def on_turn_start(self, j: int, kpv: float = 0.0, la: bool = False, pay: float = 0.0, pub: int = 0) -> None:
        """``j``'s turn begins (its roll, or a development card played before it).  ``kpv``: ``j``'s
        block value when it is the main robber victim (else 0), ``la``: one knight gives it Largest
        Army, ``pay``: the best Monopoly payoff (human preset), ``pub``: its public VP."""
        for i in range(self.n):
            if self.open[i] and i != j:
                self._end(i)
        if self.open[j]:
            return
        self.turns[j].append([1, float(kpv), 1 if la else 0, float(pay), int(pub)])
        self.open[j] = True
        self._log("turn", j, len(self.turns[j]) - 1, round(float(kpv), 3), 1 if la else 0, round(float(pay), 3))
        self._touch()

    @_guarded
    def on_public(self, j: int, pub: int) -> None:
        """``j``'s public VP now (the highest value of its open turn is kept, for the win caps)."""
        if self.open[j] and int(pub) > self.turns[j][-1][_MAXPUB]:
            self.turns[j][-1][_MAXPUB] = int(pub)
            self._touch()

    @_guarded
    def on_buy(self, j: int) -> None:
        if not self.open[j]:
            self.on_turn_start(j)
        tau = len(self.turns[j]) - 1
        self.recs[j].append(tau)
        self._log("buy", j, tau)
        self._touch()

    @_guarded
    def on_play(self, j: int, t: int) -> None:
        """``j`` played a card of type ``t``: the youngest eligible record goes, and covers ``t``."""
        if not self.open[j]:
            self.on_turn_start(j)
        tau = len(self.turns[j]) - 1
        self.turns[j][-1][_NOPLAY] = 0
        recs = self.recs[j]
        if recs:
            idx = [i for i, b in enumerate(recs) if self._eligible(b, tau)]
            i = idx[-1] if idx else len(recs) - 1
            b = recs.pop(i)
            self.cov[j][int(t)].append((b, tau))
        self._log("play", j, tau, int(t))
        self._touch()

    @_guarded
    def on_turn_end(self, j: int) -> None:
        if self.open[j]:
            self._end(j)

    def _end(self, j: int) -> None:
        self.open[j] = False
        self._log("end", j, len(self.turns[j]) - 1)
        self._touch()

    @_guarded
    def sync_counts(self, counts: Sequence[int], skip: Sequence[int] = ()) -> None:
        """Pad (cards of unknown age) or trim (the youngest) every player's records to the public
        ``counts`` (a missed log line, a resync, a session that started mid-game)."""
        changed = False
        for j in range(self.n):
            if j in skip:
                continue
            want = max(0, int(counts[j]))
            recs = self.recs[j]
            while len(recs) < want:
                recs.append(len(self.turns[j]))
                self.stats["padded"] += 1
                changed = True
            while len(recs) > want:
                recs.pop()
                self.stats["trimmed"] += 1
                changed = True
        if changed:
            self._touch()

    @_guarded
    def forget(self) -> None:
        """After a resync: every held card's age becomes unknown and no turn is open (an open turn
        counts as one with a play, so it ages nothing)."""
        for j in range(self.n):
            if self.open[j]:
                self.turns[j][-1][_NOPLAY] = 0
                self.open[j] = False
            self.recs[j] = [len(self.turns[j])] * len(self.recs[j])
        self._touch()

    # --- persistence ----------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "eligible_same_turn": self.eligible_same_turn, "recs": self.recs, "turns": self.turns,
                "cov": [[[list(iv) for iv in c] for c in cj] for cj in self.cov], "open": self.open,
                "invalid": self.invalid, "stats": dict(self.stats), "log_events": self.log_events,
                "events": self.events}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DevAgeModel":
        m = cls(int(d["n"]), bool(d.get("eligible_same_turn", False)))
        m.recs = [[int(b) for b in r] for r in d["recs"]]
        m.turns = [[[int(x[0]), float(x[1]), int(x[2]), float(x[3]), int(x[4])] for x in tj] for tj in d["turns"]]
        m.cov = [[[(int(iv[0]), int(iv[1])) for iv in c] for c in cj] for cj in d["cov"]]
        m.open = [bool(x) for x in d["open"]]
        m.invalid = bool(d.get("invalid", False))
        m.stats.update({k: int(v) for k, v in (d.get("stats") or {}).items()})
        m.log_events = bool(d.get("log_events", False))
        m.events = [list(e) for e in d.get("events") or []]
        return m

    # --- likelihood pieces ------------------------------------------------------------------------
    def _eligible(self, b: int, tau: int) -> bool:
        return tau >= b if self.eligible_same_turn else tau > b

    def _closed(self, j: int) -> int:
        """Number of ``j``'s completed own turns (an open turn may still see a play)."""
        return len(self.turns[j]) - (1 if self.open[j] else 0)

    def aging_set(self, j: int, b: int, t: int) -> List[int]:
        """``S_t``: completed no-play turns of ``j`` in which a card bought at ``b`` was eligible and no
        played card of type ``t`` was being held."""
        cov = self.cov[j][t]
        out = []
        for tau in range(max(0, b), self._closed(j)):
            if not self.turns[j][tau][_NOPLAY] or not self._eligible(b, tau):
                continue
            if any(b0 <= tau <= t0 for b0, t0 in cov):
                continue
            out.append(tau)
        return out

    def age(self, j: int, i: int) -> int:
        """No-play eligible completed turns of ``j``'s ``i``-th record (any type; for reports)."""
        b = self.recs[j][i]
        return sum(1 for tau in range(max(0, b), self._closed(j))
                   if self.turns[j][tau][_NOPLAY] and self._eligible(b, tau))

    def _factor(self, j: int, b: int, t: int, pr: Preset, eps: float) -> float:
        prod = 1.0
        for tau in self.aging_set(j, b, t):
            prod *= 1.0 - pr.hazard(t, self.turns[j][tau])
        return (1.0 - eps) * prod + eps

    def _hist_caps(self, j: int, vps_to_win: int) -> List[float]:
        """Per record (oldest first): VP cards allowed among it and the older ones, from the highest
        public VP of every own turn since its purchase (inf when no such turn)."""
        caps = []
        for b in self.recs[j]:
            mp = [tj[_MAXPUB] for tj in self.turns[j][max(0, b):]]
            caps.append(vps_to_win - 1 - max(mp) if mp else math.inf)
        return caps

    def informative(self, holders: Sequence[int], vps_to_win: int) -> bool:
        """True when some holder's cards were aged or a history cap can bind (otherwise the
        posterior is the uniform deal the default already makes)."""
        key = ("informative", self.version, tuple(holders), int(vps_to_win))
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = self._informative(holders, vps_to_win)
        return hit

    def _informative(self, holders: Sequence[int], vps_to_win: int) -> bool:
        for j in holders:
            recs = self.recs[j]
            for i, cap in enumerate(self._hist_caps(j, vps_to_win)):
                if cap < i + 1:
                    return True
            for b in recs:
                if any(self.aging_set(j, b, t) for t in _PLAYABLE):
                    return True
        return False

    # --- exact DP ---------------------------------------------------------------------------------
    def _player_dp(self, j: int, pool: Sequence[int], vps_to_win: int, cap_now: float, pr: Preset, eps: float,
                   budget: List[int]) -> _Player:
        P = _Player()
        P.recs = list(self.recs[j])
        P.F = [{t: self._factor(j, b, t, pr, eps) for t in _PLAYABLE} for b in P.recs]
        hist = self._hist_caps(j, vps_to_win)
        P.caps = []
        layer: Dict[Tuple[int, Tuple[int, ...]], float] = {(0, _ZERO): 1.0}
        P.layers = [layer]
        for k, b in enumerate(P.recs):
            cap = min(cap_now, hist[k])
            P.caps.append(cap)
            nxt: Dict[Tuple[int, Tuple[int, ...]], float] = {}
            for (mask, cnt), w in layer.items():
                for t in range(5):
                    if cnt[t] >= pool[t]:
                        continue
                    if t == _VP:
                        if cnt[_VP] + 1 > cap:
                            continue
                        m2, f = mask, 1.0
                    elif mask & _BIT[t]:
                        m2, f = mask, 1.0
                    else:
                        m2, f = mask | _BIT[t], P.F[k][t]
                    c2 = cnt[:t] + (cnt[t] + 1,) + cnt[t + 1:]
                    key = (m2, c2)
                    nxt[key] = nxt.get(key, 0.0) + w * f
            layer = nxt
            P.layers.append(layer)
            budget[0] -= len(layer)
            if budget[0] < 0:
                raise _Fallback("table budget")
        W: Dict[Tuple[int, ...], float] = {}
        for (mask, cnt), w in layer.items():
            W[cnt] = W.get(cnt, 0.0) + w
        P.W = W
        return P

    def _tables(self, pool: Sequence[int], holders: Sequence[int], counts: Sequence[int],
                public: Sequence[int], vps_to_win: int, pr: Preset):
        key = (self.version, pr.params(), tuple(pool), tuple(holders), tuple(int(counts[j]) for j in holders),
               tuple(int(public[j]) for j in holders), int(vps_to_win))
        hit = self._cache.get(key)
        if hit is not None:
            if isinstance(hit, _Fallback):
                raise hit
            return hit
        try:
            if self.invalid:
                raise _Fallback("invalid bookkeeping")
            for j in holders:
                if len(self.recs[j]) != int(counts[j]):
                    raise _Fallback("record count differs from the public count")
            eps = pr.eps()
            budget = [TABLE_BUDGET]
            players = {}
            for j in holders:
                # a card held while at public p on an own turn is not one of V - p or more VP cards
                players[j] = self._player_dp(j, pool, vps_to_win, vps_to_win - 1 - int(public[j]), pr, eps, budget)
            Z = [{_ZERO: 1.0}]
            for j in holders:
                prev = Z[-1]
                cur: Dict[Tuple[int, ...], float] = {}
                for u, zu in prev.items():
                    for nj, w in players[j].W.items():
                        f = _fall_vec(pool, u, nj)
                        if f:
                            v = _add(u, nj)
                            cur[v] = cur.get(v, 0.0) + zu * w * f
                budget[0] -= len(cur)
                if budget[0] < 0:
                    raise _Fallback("table budget")
                Z.append(cur)
            if sum(Z[-1].values()) <= 0.0:
                raise _Fallback("no consistent assignment")
            out = (players, Z)
        except _Fallback as ex:
            self._cache[key] = ex
            raise
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = out
        return out

    # --- sampling ---------------------------------------------------------------------------------
    def sample(self, pool: Sequence[int], holders: Sequence[int], counts: Sequence[int], public: Sequence[int],
               vps_to_win: int, rng, pr: Optional[Preset] = None) -> Optional[Dict[int, List[int]]]:
        """One draw of every holder's record types (oldest first) from the posterior, with ``rng``;
        ``None`` when the model is not informative or has no usable posterior (then the caller keeps
        the uniform deal; a fallback is counted)."""
        pr = pr or BOT
        try:
            if not holders or not self.informative(holders, vps_to_win):
                return None
            players, Z = self._tables(pool, holders, counts, public, vps_to_win, pr)
        except _Fallback:
            self.stats["fallbacks"] += 1
            return None
        except Exception:  # noqa: BLE001
            self.invalid = True
            self.stats["errors"] += 1
            return None
        try:
            items = sorted(Z[-1])
            u = _pick(items, [Z[-1][x] for x in items], rng)
            ns: Dict[int, Tuple[int, ...]] = {}
            for idx in range(len(holders) - 1, -1, -1):
                j = holders[idx]
                prev = Z[idx]
                cands, ws = [], []
                for nj in sorted(players[j].W):
                    base = _sub(u, nj)
                    if min(base) < 0 or base not in prev:
                        continue
                    w = prev[base] * players[j].W[nj] * _fall_vec(pool, base, nj)
                    if w > 0:
                        cands.append(nj)
                        ws.append(w)
                nj = _pick(cands, ws, rng)
                ns[j] = nj
                u = _sub(u, nj)
            out = {j: self._sample_types(players[j], ns[j], rng) for j in holders}
        except Exception:  # noqa: BLE001
            self.invalid = True
            self.stats["errors"] += 1
            return None
        self.stats["dealt"] += 1
        return out

    @staticmethod
    def _sample_types(P: _Player, n: Tuple[int, ...], rng) -> List[int]:
        m = len(P.recs)
        last = P.layers[m]
        masks = sorted(mask for (mask, cnt) in last if cnt == n)
        mask = _pick(masks, [last[(mk, n)] for mk in masks], rng)
        cnt = n
        types = [0] * m
        for k in range(m - 1, -1, -1):
            prev = P.layers[k]
            cands, ws = [], []
            for t in range(5):
                if cnt[t] == 0:
                    continue
                pc = cnt[:t] + (cnt[t] - 1,) + cnt[t + 1:]
                if t == _VP:
                    opts = [(mask, 1.0)]
                elif mask & _BIT[t]:
                    opts = [(mask, 1.0), (mask & ~_BIT[t], P.F[k][t])]
                else:
                    opts = []
                for pm, f in opts:
                    w = prev.get((pm, pc), 0.0) * f
                    if w > 0:
                        cands.append((t, pm, pc))
                        ws.append(w)
            t, mask, cnt = _pick(cands, ws, rng)
            types[k] = t
        return types

    # --- posterior summaries ------------------------------------------------------------------------
    def posterior(self, pool: Sequence[int], holders: Sequence[int], counts: Sequence[int], public: Sequence[int],
                  vps_to_win: int, pr: Optional[Preset] = None) -> Optional[Dict[int, Dict[str, Any]]]:
        """Exact per-holder summaries: ``p_vp`` (P(hidden VP = k)), ``e_vp``, ``p_type`` (P(holds at
        least one of type t)), ``records`` (per record, oldest first: P(type t)) and ``ages``.
        ``None`` without a usable posterior."""
        pr = pr or BOT
        try:
            players, _Z = self._tables(pool, holders, counts, public, vps_to_win, pr)
            out: Dict[int, Dict[str, Any]] = {}
            for j in holders:
                # external weight of j's final counts: every other holder dealt first (order-free)
                others = [{_ZERO: 1.0}]
                for i in holders:
                    if i == j:
                        continue
                    cur: Dict[Tuple[int, ...], float] = {}
                    for u, zu in others[-1].items():
                        for ni, w in players[i].W.items():
                            f = _fall_vec(pool, u, ni)
                            if f:
                                v = _add(u, ni)
                                cur[v] = cur.get(v, 0.0) + zu * w * f
                    others.append(cur)
                zo = others[-1]
                P = players[j]
                ext = {nj: sum(zu * _fall_vec(pool, u, nj) for u, zu in zo.items()) for nj in P.W}
                out[j] = self._summarise(P, ext)
                out[j]["ages"] = [self.age(j, i) for i in range(len(self.recs[j]))]
            return out
        except _Fallback:
            return None
        except Exception:  # noqa: BLE001
            self.invalid = True
            self.stats["errors"] += 1
            return None

    @staticmethod
    def _summarise(P: _Player, ext: Dict[Tuple[int, ...], float]) -> Dict[str, Any]:
        m = len(P.recs)
        tot = sum(P.W[n] * ext[n] for n in P.W)
        if tot <= 0.0:
            raise _Fallback("no consistent assignment")
        p_vp = [0.0] * (m + 1)
        p_type = [0.0] * 5
        for n, w in P.W.items():
            p = w * ext[n] / tot
            p_vp[n[_VP]] += p
            for t in range(5):
                if n[t]:
                    p_type[t] += p
        # backward pass: completion weight of every layer state, times the external weight
        back: List[Dict[Tuple[int, Tuple[int, ...]], float]] = [dict() for _ in range(m + 1)]
        back[m] = {(mask, cnt): ext.get(cnt, 0.0) for (mask, cnt) in P.layers[m]}
        for k in range(m - 1, -1, -1):
            cur: Dict[Tuple[int, Tuple[int, ...]], float] = {}
            nxt = back[k + 1]
            for (mask, cnt) in P.layers[k]:
                s = 0.0
                for t in range(5):
                    c2 = cnt[:t] + (cnt[t] + 1,) + cnt[t + 1:]
                    if t == _VP or mask & _BIT[t]:
                        m2, f = mask, 1.0
                    else:
                        m2, f = mask | _BIT[t], P.F[k][t]
                    v = nxt.get((m2, c2))
                    if v:
                        s += f * v
                cur[(mask, cnt)] = s
            back[k] = cur
        records = []
        for k in range(m):
            pt = [0.0] * 5
            for (mask, cnt), fw in P.layers[k].items():
                for t in range(5):
                    c2 = cnt[:t] + (cnt[t] + 1,) + cnt[t + 1:]
                    if t == _VP or mask & _BIT[t]:
                        m2, f = mask, 1.0
                    else:
                        m2, f = mask | _BIT[t], P.F[k][t]
                    v = back[k + 1].get((m2, c2))
                    if v:
                        pt[t] += fw * f * v
            s = sum(pt)
            records.append([x / s for x in pt] if s > 0 else [0.0] * 5)
        return {"p_vp": p_vp, "e_vp": sum(k * p for k, p in enumerate(p_vp)), "p_type": p_type, "records": records}


# ---------------------------------------------------------------------------------------------------
# Advisor helpers
# ---------------------------------------------------------------------------------------------------
def monopoly_hazard(pr: Preset, pay: float) -> float:
    """Play probability of a held Monopoly at payoff ``pay`` (cards the best resource would take)."""
    return pr.hazard(_MONO, [1, 0.0, 0, float(pay), 0])


def register_tunables(registry: Dict[str, object]) -> None:
    """``devbelief.ENABLED`` / ``H`` / ``H_KNIGHT_CTX`` / ``EPS`` (kind weight, not needs_python_evaluator:
    the belief layer only changes which cards the determinizations deal)."""
    import importlib
    import sys
    tuning = sys.modules.get("catanbot.tuning") or importlib.import_module("catanbot.tuning")
    g = globals()
    for attr, cands, parse, meaning in (
            ("ENABLED", [1], tuning._parse_int,
             "counted mode: deal opponents' hidden dev cards from the held-age posterior (RNG-neutral; 0 = uniform)"),
            ("H", [0.5, 0.95], tuning._parse_float, "per-turn play hazard of a held non-VP dev card (bot preset)"),
            ("H_KNIGHT_CTX", [0.95], tuning._parse_float,
             "knight hazard when blocked as main robber victim or one knight from Largest Army (R3; = H inactive)"),
            ("EPS", [0.0, 0.1], tuning._parse_float, "per-type floor of the no-play likelihood (hoarders)")):
        t = tuning.Tunable(name=f"devbelief.{attr}", module=__name__, attr=attr, default=g[attr], kind="weight",
                           candidates=list(cands), requires_search=False, needs_python_evaluator=False,
                           clear_caches=False, parse=parse,
                           description=f"{meaning}; read only in counted mode with devbelief.ENABLED=1")
        registry[t.name] = t
