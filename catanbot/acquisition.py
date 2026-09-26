"""Acquisition: get what we need from the cheapest source (docs/PRIORITY_PLAN.md step 3, area "trades").

Everything here is off by default; with every switch at its default nothing of this module is imported by the bot
(tested in a fresh interpreter), ``heuristic.static_value``, the placement score and ``cpp/*`` are untouched and no
switch needs the Python evaluator.  Pieces (the design is ``docs/designs/priority_areas_2026-09-26.json``, area
"trades", as revised after both critiques):

``acq.progress`` (``search.acq``: 1 = conversions only, a Stage 0 diagnostic; 2 = conversions + production, the
tested arm; ``search.acq_w``, ``search.acq_self``).  A hub provider (:class:`AcqContext`, catanbot/corrections.py)
that replaces static's progress term ``heuristic._progress_to_build`` for our own seat by an *effective* progress:

* hand ``h`` (``n = sum h``), our ratios ``rho_r = port_ratio(i, r)``, bank stock ``b_r``; **gate**: ``n > 7``
  gives no correction (a hand above 7 is not a plan, it is a discard risk; this replaces the keep factor);
* per target ``T`` of static's (city 3.2 with a settlement and fewer than 4 cities, settlement 2.8 with a buildable
  spot, dev card 1.0 with a non-empty deck), cost ``c``: ``have = sum min(h_r, c_r)``, missing ``m_r``, surplus
  ``s_r``; ``K = sum floor(s_r / rho_r)`` conversions are allocated one card at a time to missing resources with
  capacity ``min(m_r, b_r)`` left (the bank caps them), least likely to be produced first; ``m'`` what is left;
* mode 2 credits production as ``prod = sum_r E[min(Y_r^(k), m'_r)] = sum_r sum_{j <= m'_r} P(Y_r^(k) >= j)``
  from the exact ``k``-roll distribution of our yield of ``r`` (robber hex excluded, 7 yields nothing), not the
  expected cards; mode 1 credits no production;
* ``conv_frac = min(0.5 sum_r frac(s_r / rho_r), max(0, sum_r min(m'_r, b_r - alloc_r) - prod))`` (a started
  conversion, half credit, capped by what is still missing and what the bank still holds after the integer ones);
* ``f_T = min(1, (have + conv_int + prod + conv_frac) / |c|)``, ``E = max_T 0.45 V_T f_T^2`` and ``P`` = the same
  with no credit, which is exactly ``heuristic._progress_to_build`` (same targets, same Python arithmetic), so
  ``C_me = acq_w (E - P)`` is 0.0 exactly when nothing is credited (the hub's fast path), else in
  ``(0, 1.44 acq_w]`` static points (at most 0.14 VP);
* **horizon** ``k`` = rolls up to and including the seat's next roll: our own seat gets ``k = n`` at every node of
  our turn *and* after END_TURN (the next player then has ``k = 1``); opponents (``acq_self = 0`` only)
  ``(i - current) mod n``, plus one in the roll phase, 0 for the current player after the roll.
  ``discard.opponent_rolls_before_my_turn`` is not used (it returns ``n - 1`` for the player about to roll);
* no pending-port credit (a port about to be built belongs to the ports area).

It applies at every leaf of our searches, also when we *answer* an offer: the reject leaf keeps the surplus that our
own port or the bank converts on our next turn, so accept-vs-reject sees the port alternative.

Composition with ``conversion.py`` (``search.conv``): the two are providers of one hub, summed per leaf, and value
different things.  ``conv`` is a *flow*: the long-run cost of the resources our buildings do not produce, paid in
bank / port conversions over the rolls left; it moves only with buildings and ports (a settlement, a city).
``acq.progress`` is a *stock*: how close the cards in hand (plus what they convert into now and what we are likely
to roll before our next turn) are to the next build; it moves with the hand.  Neither reads the other's output and
they cancel disjoint parts of static (conv: the port credit; acq: the hand-progress term).  The one shared input is
the port ratios: a new port settlement earns conv its future savings and acq the one-off convertibility of the
current hand at the new ratio (bounded by 1.44 points), which are different cards.

``acq.breadth`` (B) (``search.acq_shapes``): :func:`mixed_offers` injects mixed-give 2-for-1 proposals (one card
each of two surplus types for one missing card) as real candidates of our main-phase nodes, ranked by
``p_any = 1 - prod_j (1 - p_j)`` exactly as ``Searcher._trade_outcomes`` computes it; (A) is the existing
``search.trade_proposals = 5``; ``search.acq_breadth = 1`` switches both on as one knob (the self-play screen runs
one tunable per ablate.py call).

``acq_floor`` (``search.acq_floor``, weight :data:`W_PREMIUM`; the user's direction, 2026-09-26): a player trade is
riskier than the same exchange with the bank, because the partner also gets something they want.  Every player
trade we would propose or accept is compared with our best bank / port alternative for the same cards (now in our
turn, on our next turn when we answer an offer) in win-probability terms of the search's own evaluator (the softmax
over all seats already charges the partner's gain against us); the trade must beat it by
``W_PREMIUM x partner's gain x danger.danger_multiplier(partner)``; ties go to the bank.  See
:func:`bank_plans`, :func:`trade_premium` and ``Searcher._floor_*``.  The feed-the-leader and late-game rules are
unchanged (an extra requirement, not a replacement).  :func:`dominated_by_bank` is the cheap card-count version used
by ``trading.candidate_offers(port_floor=True)`` and the :class:`TradeObserver` counters.

``acq.flow``: :func:`port_flow_value`, the trade-flow value of a port (cards saved over the rest of the game) with
its table :data:`FLOW_R` and give-share model :data:`FLOW_W`, fitted on the T1 proof logs and validated on T2
(scripts/acq_flow_fit.py).  A pure function for the ports area (its provider and ``ports.FLOW_KAPPA`` are step 4);
the bot never calls it.

``acq.calib`` (``opponent_model.CALIB_RATE`` / ``CALIB_PRIOR``, fallback ``opponent_model.REJECT_STREAK``; politics
rule): :class:`AcceptCalibrator`, an online recalibration of ``OpponentModel.predict_accept`` per opponent from every
observed answer, and the rejection-streak rule.

Tunables register through :func:`register_tunables` (one import line at the end of catanbot/tuning.py).
"""
from __future__ import annotations

import importlib
import math
import sys
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import board as B
from .state import GameState, PHASE_ROLL, TradeOffer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GATE = 7                 # acq.progress: a hand above this many cards gets no correction
FRAC_W = 0.5             # weight of a started (fractional) conversion
MAX_J = 3                # production tails up to 3 cards of one resource (a city needs 3 ore)
CACHE_CAP = 20000        # a memo dict is cleared when it grows beyond this
SHAPES_KEEP = 2          # acq.breadth (B): injected offers kept per node ...
SHAPES_MIN_P = 0.15      # ... with p_any at least this
W_PREMIUM = 1.0          # acq_floor: premium per unit of the partner's win-probability gain x danger multiplier
FLOOR_WINDOW = 3         # acq_floor: proposals checked beyond the node's proposal slots (the rest are not tried)



# ---------------------------------------------------------------------------
# acq.progress: horizon, production tails, targets, effective progress (pure functions)
# ---------------------------------------------------------------------------
def horizon_k(state: GameState, i: int, me: int) -> int:
    """Rolls up to and including seat ``i``'s next roll (``me`` = the searching seat, see the module docstring)."""
    n = state.num_players
    cur = state.current
    if i == me and cur == me:
        return n
    if state.phase == PHASE_ROLL:
        return (i - cur) % n + 1
    if i == cur:
        return 0
    return (i - cur) % n


def roll_yields(state: GameState, i: int) -> Dict[int, List[int]]:
    """Seat ``i``'s cards per resource for every dice total 2..12 (cities 2, robber hex excluded, 7 = nothing)."""
    out = {d: [0, 0, 0, 0, 0] for d in range(2, 13)}
    p = state.players[i]
    for mult, verts in ((1, p.settlements), (2, p.cities)):
        for v in verts:
            for h in B.VERTEX_HEXES[v]:
                res, num = state.hexes[h]
                if res == B.DESERT or not num or h == state.robber:
                    continue
                out[num][res] += mult
    return out


def roll_tails(yields: Dict[int, Sequence[int]], k: int, max_j: int = MAX_J) -> List[List[float]]:
    """``t[r][j - 1] = P(Y_r^(k) >= j)`` for ``j = 1..max_j``: the exact ``k``-fold convolution of the per-roll
    yield (sums saturate at ``max_j``, which keeps every event ``>= j`` exact)."""
    out = []
    for r in range(5):
        dist = [1.0] + [0.0] * max_j
        for _ in range(max(0, k)):
            new = [0.0] * (max_j + 1)
            for s, ps in enumerate(dist):
                if ps == 0.0:
                    continue
                for d, pd in B.ROLL_PROB.items():
                    y = 0 if d == 7 else yields[d][r]
                    new[min(max_j, s + y)] += ps * pd
            dist = new
        tail = []
        acc = 0.0
        for j in range(max_j, 0, -1):
            acc += dist[j]
            tail.append(acc)
        out.append(tail[::-1])
    return out


def targets(state: GameState, i: int) -> List[Tuple[Tuple[int, ...], float]]:
    """Static's progress targets for seat ``i``, in ``heuristic._progress_to_build``'s order: ``[(cost, V)]``."""
    from .heuristic import BUILD_VALUE
    from .placement import buildable_settlements
    p = state.players[i]
    out: List[Tuple[Tuple[int, ...], float]] = []
    if p.settlements and len(p.cities) < B.MAX_CITIES:
        out.append((B.COST_CITY, BUILD_VALUE["city"]))
    if len(p.settlements) < B.MAX_SETTLEMENTS and buildable_settlements(state, i):
        out.append((B.COST_SETTLEMENT, BUILD_VALUE["settlement"]))
    if sum(state.dev_deck) > 0:
        out.append((B.COST_DEV, BUILD_VALUE["dev card"]))
    return out


def target_credit(hand: Sequence[int], cost: Sequence[int], ratios: Sequence[int], bank: Sequence[int],
                  tails: Optional[Sequence[Sequence[float]]]) -> Dict[str, Any]:
    """The acquisition credit of one target (``tails = None``: mode 1, no production): every intermediate."""
    total = sum(cost)
    have = sum(min(hand[r], cost[r]) for r in range(5))
    miss = [max(0, cost[r] - hand[r]) for r in range(5)]
    surplus = [max(0, hand[r] - cost[r]) for r in range(5)]
    K = sum(surplus[g] // ratios[g] for g in range(5))
    alloc = [0] * 5
    left = K
    if left and any(miss):
        order = sorted(range(5), key=lambda r: (tails[r][0] if tails is not None else 0.0, r))
        cap = [min(miss[r], bank[r]) for r in range(5)]
        while left > 0:
            got = False
            for r in order:
                if alloc[r] < cap[r]:
                    alloc[r] += 1
                    left -= 1
                    got = True
                    break
            if not got:
                break
    conv_int = sum(alloc)
    m2 = [miss[r] - alloc[r] for r in range(5)]
    prod = 0.0
    if tails is not None:
        for r in range(5):
            for j in range(min(m2[r], len(tails[r]))):
                prod += tails[r][j]
    frac_sum = sum((surplus[g] % ratios[g]) / ratios[g] for g in range(5))
    room = max(0.0, sum(min(m2[r], max(0, bank[r] - alloc[r])) for r in range(5)) - prod)
    conv_frac = min(FRAC_W * frac_sum, room)
    f = min(1.0, (have + conv_int + prod + conv_frac) / total)
    return {"have": have, "missing": miss, "surplus": surplus, "K": K, "alloc": alloc, "conv_int": conv_int,
            "left": m2, "prod": prod, "conv_frac": conv_frac, "f": f, "f0": have / total}


def effective_progress(hand: Sequence[int], tgts: Sequence[Tuple[Sequence[int], float]], ratios: Sequence[int],
                       bank: Sequence[int], tails: Optional[Sequence[Sequence[float]]]) -> Tuple[float, float]:
    """``(E, P)``: ``max_T 0.45 V_T f_T^2`` with and without the acquisition credit (``P`` is exactly
    ``heuristic._progress_to_build`` for the same targets)."""
    E = P = 0.0
    for cost, value in tgts:
        total = sum(cost)
        have = sum(min(hand[r], cost[r]) for r in range(5))
        frac = have / total
        p = value * 0.45 * frac * frac
        if p > P:
            P = p
        if have < total:
            f = target_credit(hand, cost, ratios, bank, tails)["f"]
        else:
            f = frac
        e = value * 0.45 * f * f
        if e > E:
            E = e
    return E, P


def p_complete(yields: Dict[int, Sequence[int]], missing: Sequence[int], k: int) -> float:
    """P(``k`` rolls produce at least ``missing[r]`` of every resource): exact (joint over the same dice)."""
    need = tuple(max(0, int(x)) for x in missing)
    if not any(need):
        return 1.0
    dist = {tuple([0] * 5): 1.0}
    for _ in range(max(0, k)):
        new: Dict[tuple, float] = {}
        for st, ps in dist.items():
            for d, pd in B.ROLL_PROB.items():
                y = (0, 0, 0, 0, 0) if d == 7 else yields[d]
                key = tuple(min(need[r], st[r] + y[r]) for r in range(5))
                new[key] = new.get(key, 0.0) + ps * pd
        dist = new
    return dist.get(need, 0.0)


def best_target_left(state: GameState, i: int, mode: int = 2) -> Tuple[Optional[str], List[int], float]:
    """Seat ``i``'s target with the largest effective progress now: ``(name, cards still missing after the
    conversions, P(rolling them before i's next build))`` (Stage 0 diagnostics and advice text)."""
    names = {B.COST_CITY: "city", B.COST_SETTLEMENT: "settlement", B.COST_DEV: "dev card"}
    k = horizon_k(state, i, i)
    ys = roll_yields(state, i)
    tails = roll_tails(ys, k) if mode >= 2 else None
    p = state.players[i]
    ratios = [state.port_ratio(i, r) for r in range(5)]
    best = (-1.0, None, [0] * 5)
    for cost, value in targets(state, i):
        d = target_credit(p.resources, cost, ratios, state.bank, tails)
        e = value * 0.45 * d["f"] * d["f"]
        if e > best[0]:
            best = (e, names.get(tuple(cost)), d["left"])
    if best[1] is None:
        return None, [0] * 5, 0.0
    return best[1], best[2], p_complete(ys, best[2], k)


# ---------------------------------------------------------------------------
# acq.progress: the hub provider
# ---------------------------------------------------------------------------
def _osig(state: GameState) -> tuple:
    return tuple(tuple(sorted(q.settlements + q.cities)) for q in state.players)


class AcqContext:
    """acq.progress for one ``Searcher.search`` call: a :class:`corrections.CorrectionHub` provider."""

    priors = False     # no move-ordering hook

    def __init__(self, root: GameState, me: int, mode: int = 2, weight: float = 1.0, self_only: bool = True):
        self.me = int(me)
        self.n = root.num_players
        self.mode = int(mode)
        self.weight = float(weight)
        self.self_only = bool(self_only)
        self.stats: Dict[str, int] = {"evals": 0, "nonzero": 0, "gated": 0, "tail_calls": 0, "tail_misses": 0,
                                      "target_calls": 0, "target_misses": 0}
        self._tails: Dict[tuple, List[List[float]]] = {}
        self._targets: Dict[tuple, list] = {}

    @classmethod
    def for_search(cls, root: GameState, me: int, cfg) -> "AcqContext":
        return cls(root, me, mode=int(getattr(cfg, "acq", 2)), weight=float(getattr(cfg, "acq_w", 1.0)),
                   self_only=bool(getattr(cfg, "acq_self", 1)))

    def _put(self, d: dict, key, value):
        if len(d) >= CACHE_CAP:
            d.clear()
        d[key] = value
        return value

    def tails(self, state: GameState, i: int, k: int) -> List[List[float]]:
        """Production tails of seat ``i`` over ``k`` rolls (memo: its buildings, the robber hex, ``k``)."""
        self.stats["tail_calls"] += 1
        p = state.players[i]
        key = (i, tuple(sorted(p.settlements)), tuple(sorted(p.cities)), state.robber, k)
        hit = self._tails.get(key)
        if hit is None:
            self.stats["tail_misses"] += 1
            hit = self._put(self._tails, key, roll_tails(roll_yields(state, i), k))
        return hit

    def targets(self, state: GameState, i: int) -> list:
        """Static's targets of seat ``i`` (memo: its buildings and roads, every seat's buildings, the dev deck)."""
        self.stats["target_calls"] += 1
        p = state.players[i]
        key = (i, len(p.settlements), len(p.cities), tuple(p.roads), _osig(state), sum(state.dev_deck) > 0)
        hit = self._targets.get(key)
        if hit is None:
            self.stats["target_misses"] += 1
            hit = self._put(self._targets, key, targets(state, i))
        return hit

    def seat_correction(self, state: GameState, i: int) -> float:
        """``acq_w (E_i - P_i)`` static points (0.0 above the gate or without a known hand)."""
        p = state.players[i]
        if not p.hand_known:
            return 0.0
        hand = p.resources
        if sum(hand) > GATE:
            self.stats["gated"] += 1
            return 0.0
        tgts = self.targets(state, i)
        if not tgts:
            return 0.0
        tails = self.tails(state, i, horizon_k(state, i, self.me)) if self.mode >= 2 else None
        ratios = [state.port_ratio(i, r) for r in range(5)]
        E, P = effective_progress(hand, tgts, ratios, state.bank, tails)
        return self.weight * (E - P)

    def corrections(self, state: GameState) -> Optional[List[float]]:
        """Per-seat points (our seat only unless ``acq_self = 0``); ``None`` when every correction is 0.0."""
        self.stats["evals"] += 1
        seats = [self.me] if self.self_only else range(self.n)
        out: Optional[List[float]] = None
        for i in seats:
            c = self.seat_correction(state, i)
            if c != 0.0:
                if out is None:
                    out = [0.0] * self.n
                out[i] = c
        if out is not None:
            self.stats["nonzero"] += 1
        return out


# ---------------------------------------------------------------------------
# acq.breadth (B): injected mixed-give 2-for-1 offers
# ---------------------------------------------------------------------------
def _unit(r: int, n: int = 1) -> Tuple[int, ...]:
    return tuple(n if x == r else 0 for x in range(5))


def mixed_offers(state: GameState, me: int, accept_fn, keep: int = SHAPES_KEEP,
                 min_p: float = SHAPES_MIN_P) -> List[Tuple[float, tuple, int]]:
    """``[(p_any, PROPOSE_TRADE action, likely partner)]``, best first: one card each of two surplus types for one
    missing card (``discard.needed_vector``), skipped where the bank or a port already gives that card for two
    surplus cards; ``p_any`` over the responders who can pay with ``accept_fn(state, j, offer, me)`` (the search's
    ``_accept_probability``); no offer whose likely accepter is at 9 VP or would be fed a build."""
    from . import actions as A
    from . import trading as T
    from .discard import needed_vector
    from .robber import estimated_vp
    p = state.players[me]
    hand = p.resources
    needed = needed_vector(state, me)
    missing = [r for r in range(5) if needed[r] > hand[r]]
    surplus = [g for g in range(5) if hand[g] - needed[g] >= 1]
    if not missing or len(surplus) < 2:
        return []
    out: List[Tuple[float, tuple, int]] = []
    for r in missing:
        if state.bank[r] >= 1 and any(g != r and state.port_ratio(me, g) <= 2
                                      and hand[g] - needed[g] >= state.port_ratio(me, g) for g in range(5)):
            continue      # a 2:1 port on a surplus card already gives r for two cards, for certain
        for a, b in combinations([g for g in surplus if g != r], 2):
            give = tuple(1 if x in (a, b) else 0 for x in range(5))
            get = _unit(r)
            offer = TradeOffer(me, list(give), list(get))
            probs: Dict[int, float] = {}
            for j in range(state.num_players):
                if j == me:
                    continue
                q = state.players[j]
                if q.hand_known and q.resources[r] < 1:
                    continue
                probs[j] = accept_fn(state, j, offer, me)
            if not probs:
                continue
            p_any = 1.0 - math.prod(1.0 - x for x in probs.values())
            if p_any < min_p:
                continue
            jstar = max(probs, key=probs.get)
            if T.offer_is_feeding_leader(state, me, jstar, list(give))[0] \
                    or estimated_vp(state, jstar) >= B.VP_TO_WIN - 1:
                continue
            out.append((p_any, (A.PROPOSE_TRADE, give, get), jstar))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out[:keep]


# ---------------------------------------------------------------------------
# acq_floor: bank / port alternatives and the player-trade premium
# ---------------------------------------------------------------------------
def dominated_by_bank(state: GameState, i: int, give: Sequence[int], get: Sequence[int]) -> bool:
    """True when seat ``i``'s own bank / port rates deliver ``get`` for at most the cards ``give`` (the bank holds
    them): the player trade is no better than a certain exchange with the bank (card counts only)."""
    n_get = sum(get)
    if n_get <= 0 or any(state.bank[r] < get[r] for r in range(5)):
        return False
    conv = sum(give[g] // state.port_ratio(i, g) for g in range(5) if give[g] and not get[g])
    return conv >= n_get


def bank_plans(state: GameState, i: int, get: Sequence[int], max_plans: int = 12) -> List[Tuple[int, ...]]:
    """Distinct cost vectors ``C`` of the bank / port conversions that give seat ``i`` the cards ``get`` from its
    hand (a card of ``r`` costs ``port_ratio(i, g)`` cards of one ``g`` outside ``get``); ``[]`` when the bank lacks
    them or the hand cannot pay.  Up to two cards every combination, beyond that one greedy plan."""
    if sum(get) <= 0 or any(state.bank[r] < get[r] for r in range(5)):
        return []
    hand = list(state.players[i].resources)
    ratios = [state.port_ratio(i, g) for g in range(5)]
    payers = [g for g in range(5) if not get[g]]
    cards = [r for r in range(5) for _ in range(get[r])]
    if len(cards) > 2:
        cost = [0] * 5
        for _r in cards:
            opts = [g for g in payers if hand[g] - cost[g] >= ratios[g]]
            if not opts:
                return []
            g = min(opts, key=lambda x: (ratios[x], -(hand[x] - cost[x]), x))
            cost[g] += ratios[g]
        return [tuple(cost)]
    plans = set()

    def rec(k: int, cost: List[int]) -> None:
        if k == len(cards):
            plans.add(tuple(cost))
            return
        for g in payers:
            if hand[g] - cost[g] >= ratios[g]:
                cost[g] += ratios[g]
                rec(k + 1, cost)
                cost[g] -= ratios[g]

    rec(0, [0] * 5)
    return sorted(plans)[:max_plans]


def trade_state(state: GameState, i: int, j: int, give: Sequence[int], get: Sequence[int]) -> GameState:
    """A copy of ``state`` with the player trade done: ``i`` gives ``give`` to ``j`` and receives ``get``."""
    s = state.copy()
    a, b = s.players[i], s.players[j]
    for r in range(5):
        a.resources[r] += get[r] - give[r]
        b.resources[r] += give[r] - get[r]
    a.hand_size = sum(a.resources)
    b.hand_size = sum(b.resources)
    return s


def bank_state(state: GameState, i: int, cost: Sequence[int], get: Sequence[int]) -> GameState:
    """A copy of ``state`` with seat ``i``'s bank / port conversions done (pays ``cost``, receives ``get``)."""
    s = state.copy()
    a = s.players[i]
    for r in range(5):
        a.resources[r] += get[r] - cost[r]
        s.bank[r] += cost[r] - get[r]
    a.hand_size = sum(a.resources)
    return s


def danger_of(state: GameState, j: int, belief=None):
    """``(danger.danger_multiplier(win path of j), the win path)`` through the module attributes (so the
    ``danger.danger_multiplier`` registry flag applies)."""
    from . import danger as D
    wp = D.win_paths(state)[j] if belief is None else D.win_path(state, j, belief)
    return D.danger_multiplier(wp), wp


def trade_premium(partner_gain: float, danger_mult: float, weight: Optional[float] = None) -> float:
    """``W_PREMIUM x max(0, partner's gain) x danger multiplier`` (win-probability units)."""
    w = W_PREMIUM if weight is None else float(weight)
    return w * max(0.0, float(partner_gain)) * float(danger_mult)


def floor_reason(state: GameState, me: int, j: int, cost: Sequence[int], get: Sequence[int], wp,
                 answering: bool) -> str:
    """Why the bank / port is preferred: 'the bank gives you the same 1 brick for 2 wheat on your next turn without
    helping blue (~2 turns from winning)'."""
    def cards(v):
        return " + ".join(f"{v[r]} {B.RESOURCE_NAMES[r]}" for r in range(5) if v[r]) or "nothing"
    ratios = {state.port_ratio(me, g) for g in range(5) if cost[g]}
    src = "the bank" if ratios == {4} else ("your port" if len(ratios) == 1 else "your ports / the bank")
    q = state.players[j]
    who = q.name or q.color
    when = " on your next turn" if answering else ""
    far = f" (~{wp.turns:.0f} turns from winning)" if wp is not None and wp.need_vp > 0 else ""
    return f"{src} gives you the same {cards(get)} for {cards(cost)}{when} without helping {who}{far}"


# ---------------------------------------------------------------------------
# acq.flow: trade-flow port value (fitted table; a pure function for the ports area)
# ---------------------------------------------------------------------------
# scripts/acq_flow_fit.py (fit: 1,000 T1 proof games, our seat vs 3 x ValueFunction, 6,708 bank trades; validation:
# the 400 T2 games vs AlphaBeta).  FLOW_T / FLOW_R: bank trades (any ratio) our bot still makes after t post-setup
# player turns, averaged over the games still running at t (linear interpolation; 0 from 120 on: 20 of 1,000 games
# run longer).  The unconditional 4:1-only table reproduces the design's 5.04 / 4.23 / 3.01 / 1.59 / 0.56 at
# 0 / 20 / 40 / 60 / 80 (T2: 5.16 / 4.38 / 3.08 / 1.67 / 0.55).  FLOW_W: multinomial logit of which resource a bank
# trade gives: intercepts per resource (wood 0) + slope on our production share of r + slope on "r is wheat / ore
# while a city can still be built" (fitted ~0: the production share carries it).  Validation on T2, mean absolute
# error per resource per game: the give split given the trades 0.69 (flat shares 1.27, share-only 0.71); the whole
# R(t) w_r at t = 0 / 20 / 40 / 60: 0.96 / 0.89 / 0.76 / 0.62 (flat shares 1.37 / 1.22 / 1.00 / 0.77).
FLOW_T = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
FLOW_R = [6.71, 6.4, 5.81, 5.18, 4.39, 3.62, 2.97, 2.52, 2.19, 1.77, 1.63, 1.42, 0.0]
FLOW_W = {"intercept": [0.0, -0.067, -0.171, -0.277, -0.449], "share": 6.606, "city": 0.038}
FLOW_FIT = {"source": "scripts/acq_flow_fit.py", "fit": "T1 (1,000 games)", "validate": "T2 (400 games)"}


def flow_remaining(t: float) -> float:
    """``R(t)``: bank trades still to come after ``t`` post-setup player turns (:data:`FLOW_R`, interpolated)."""
    if t <= FLOW_T[0]:
        return FLOW_R[0]
    for k in range(1, len(FLOW_T)):
        if t <= FLOW_T[k]:
            a, b = FLOW_T[k - 1], FLOW_T[k]
            return FLOW_R[k - 1] + (FLOW_R[k] - FLOW_R[k - 1]) * (t - a) / (b - a)
    return 0.0


def flow_features(prod: Sequence[float], city_open: bool) -> List[Tuple[float, float]]:
    """``[(production share of r, 1 if r is a city resource and a city can still be built)]`` per resource."""
    tot = sum(prod)
    return [((prod[r] / tot) if tot > 0 else 0.2, 1.0 if (city_open and r in (B.WHEAT, B.ORE)) else 0.0)
            for r in range(5)]


def flow_weights(prod: Sequence[float], city_open: bool, w: Optional[Dict[str, Any]] = None) -> List[float]:
    """``w_r``: the share of bank gives in ``r`` (the :data:`FLOW_W` logit), summing to 1."""
    w = FLOW_W if w is None else w
    z = [w["intercept"][r] + w["share"] * sh + w["city"] * ct
         for r, (sh, ct) in enumerate(flow_features(prod, city_open))]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    tot = sum(e)
    return [x / tot for x in e]


def post_setup_turn(state: GameState) -> int:
    """Player turns completed since the setup (``GameState.turn`` counts the ``2 n`` setup turns too)."""
    return max(0, int(state.turn) - 2 * state.num_players)


def port_flow_value(state: GameState, i: int, port: int, t: Optional[float] = None) -> float:
    """Cards seat ``i`` saves over the rest of the game by owning ``port`` (``B.PORT_GENERIC`` or a resource):
    ``sum_r N_r(i, t) (rho_r - rho'_r)`` with ``N_r = R(t) w_r(i)`` the expected bank trades giving ``r`` still to
    come, ``rho`` i's ratios now and ``rho'`` with the port.  ``t`` defaults to the state's post-setup turn.  Pure:
    the state is not modified."""
    from .placement import player_production
    p = state.players[i]
    t = post_setup_turn(state) if t is None else float(t)
    R = flow_remaining(t)
    if R <= 0.0:
        return 0.0
    rho = [state.port_ratio(i, r) for r in range(5)]
    new = [min(x, 3) for x in rho] if port == B.PORT_GENERIC else [2 if r == port else rho[r] for r in range(5)]
    city_open = bool(p.settlements) and len(p.cities) < B.MAX_CITIES
    w = flow_weights(player_production(state, i, ignore_robber=True), city_open)
    return sum(R * w[r] * (rho[r] - new[r]) for r in range(5))


# ---------------------------------------------------------------------------
# acq.calib: online P(accept) calibration per opponent and the rejection streak
# ---------------------------------------------------------------------------
CALIB_STEP_A0 = 0.10          # step sizes (x CALIB_RATE) of the shared intercept, the per-opponent intercept, ...
CALIB_STEP_AJ = 0.25
CALIB_STEP_BETA = 0.02        # ... and the shared slope
CALIB_CLIP_A = (-4.0, 2.0)
CALIB_CLIP_BETA = (0.3, 1.5)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clip(x: float, lo_hi: Tuple[float, float]) -> float:
    return max(lo_hi[0], min(lo_hi[1], x))


class AcceptCalibrator:
    """``p = sigmoid(a0 + a_j + beta l) x can_pay`` with ``l`` the raw logit of ``OpponentModel.predict_accept``;
    ``a0`` starts at ``prior`` (``opponent_model.CALIB_PRIOR``), ``a_j = 0``, ``beta = 1``; after every observed
    answer ``y`` (1 accept, 0 reject) with ``e = y - sigmoid(a0 + a_j + beta l)``:
    ``a0 += 0.10 eta e``, ``a_j += 0.25 eta e`` (both clipped to [-4, 2]), ``beta += 0.02 eta e l`` ([0.3, 1.5]).
    Also the per-opponent rejection streaks of the fallback rule (``opponent_model.REJECT_STREAK``).  Lives on an
    ``OpponentModel`` (one per game in the search bot) and is created only when one of the switches is on."""

    def __init__(self, prior: float = 0.0):
        self.a0 = float(prior)
        self.a: Dict[str, float] = {}
        self.beta = 1.0
        self.streak: Dict[str, int] = {}
        self.updates = 0

    def logit(self, name: str, raw: float) -> float:
        return self.a0 + self.a.get(name, 0.0) + self.beta * raw

    def prob(self, name: str, raw: float) -> float:
        return _sigmoid(self.logit(name, raw))

    def update(self, name: str, raw: float, accepted: bool, eta: float) -> None:
        e = (1.0 if accepted else 0.0) - self.prob(name, raw)
        self.a0 = _clip(self.a0 + CALIB_STEP_A0 * eta * e, CALIB_CLIP_A)
        self.a[name] = _clip(self.a.get(name, 0.0) + CALIB_STEP_AJ * eta * e, CALIB_CLIP_A)
        self.beta = _clip(self.beta + CALIB_STEP_BETA * eta * e * raw, CALIB_CLIP_BETA)
        self.updates += 1

    def note_answer(self, name: str, accepted: bool) -> None:
        self.streak[name] = 0 if accepted else self.streak.get(name, 0) + 1

    def excluded(self, name: str, streak: int) -> bool:
        return streak > 0 and self.streak.get(name, 0) >= streak


# ---------------------------------------------------------------------------
# Self-play mechanism counters (seat observer "trades", tuning.SEAT_OBSERVERS)
# ---------------------------------------------------------------------------
class TradeObserver:
    """Per seat: proposals (all / mixed-give / dominated by the proposer's own bank-port rate), answers to other
    players' offers (accepts, rejects, accepts dominated by the responder's own rate), trades executed as proposer
    and as partner (and how many of them were dominated for that side), cancelled offers."""

    KEYS = ("proposals", "proposals_mixed", "proposals_dominated", "accepts", "rejects", "accepts_dominated",
            "executed", "executed_dominated", "partnered", "partnered_dominated", "cancels")

    def __init__(self, num_seats: int):
        n = int(num_seats)
        self.c = {k: [0] * n for k in self.KEYS}

    def on_action(self, state: GameState, action, player: int) -> None:
        from . import actions as A
        k = action[0]
        c = self.c
        if k == A.PROPOSE_TRADE:
            give, get = action[1], action[2]
            c["proposals"][player] += 1
            if sum(1 for x in give if x) >= 2:
                c["proposals_mixed"][player] += 1
            if dominated_by_bank(state, player, give, get):
                c["proposals_dominated"][player] += 1
        elif k in (A.ACCEPT_TRADE, A.REJECT_TRADE) and state.pending_trade is not None \
                and state.pending_trade.origin is None:
            o = state.pending_trade
            if k == A.ACCEPT_TRADE:
                c["accepts"][player] += 1
                if dominated_by_bank(state, player, o.get, o.give):
                    c["accepts_dominated"][player] += 1
            else:
                c["rejects"][player] += 1
        elif k == A.EXECUTE_TRADE and state.pending_trade is not None:
            o = state.pending_trade
            j = action[1]
            c["executed"][player] += 1
            c["partnered"][j] += 1
            if dominated_by_bank(state, player, o.give, o.get):
                c["executed_dominated"][player] += 1
            if dominated_by_bank(state, j, o.get, o.give):
                c["partnered_dominated"][j] += 1
        elif k == A.CANCEL_TRADE:
            c["cancels"][player] += 1

    def result(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self.c.items()}


# ---------------------------------------------------------------------------
# Tunable registration (called from the last lines of catanbot/tuning.py)
# ---------------------------------------------------------------------------
_ONLY_ACQ = "only with search.acq > 0 (base spec ...,acq=2)"


def register_tunables(registry: Dict[str, object]) -> None:
    """Add the trades-area switches (docs/PRIORITY_PLAN.md step 3) to the ``tuning.TUNABLES`` registry."""
    tuning = sys.modules.get("catanbot.tuning") or importlib.import_module("catanbot.tuning")
    Tunable = tuning.Tunable
    for attr, default, cands, parse, desc in (
            ("acq", 0, [2, 1], tuning._parse_int,
             "acq.progress (catanbot/acquisition.py): our hand's progress to the next build credits bank / port "
             "conversions (capped by the bank) and, with 2, the chance to roll the missing cards before our next "
             "build; no correction above 7 cards; 1 = conversions only (a diagnostic); 0 = off (the default bot); "
             "budgeted at depth 1"),
            ("acq_w", 1.0, [0.0, 0.5, 2.0], tuning._parse_float,
             "value weight of the acq.progress correction (0 = an exact A/A of the hub); " + _ONLY_ACQ),
            ("acq_self", 1, [0], tuning._parse_int,
             "1 = acq.progress for our own seat only; 0 = every seat with a known hand (a self-play variant); "
             + _ONLY_ACQ),
            ("acq_shapes", 0, [1], tuning._parse_int,
             "acq.breadth (B): inject mixed-give 2-for-1 player offers (two surplus types for one missing card) as "
             "proposal candidates, ranked by P(someone accepts); 0 = off"),
            ("acq_breadth", 0, [1], tuning._parse_int,
             "the acq.breadth bundle as one switch: at least 5 proposal candidates per node (search.trade_proposals) "
             "and the mixed-give offers (search.acq_shapes); 0 = off"),
            ("acq_floor", 0, [1], tuning._parse_int,
             "player-trade premium: never propose or accept a player trade that does not beat our best bank / port "
             "alternative for the same cards (win probability; ties go to the bank) by acquisition.W_PREMIUM x the "
             "partner's gain x danger_multiplier(partner); candidate_offers drops offers our own port matches; "
             "0 = off")):
        t = Tunable(name=f"search.{attr}", module="catanbot.search", attr=attr, default=default, kind="search",
                    candidates=list(cands), requires_search=True, requires_depth=1, spec_key=attr, parse=parse,
                    description=desc)
        registry[t.name] = t
    t = Tunable(name="acquisition.W_PREMIUM", module=__name__, attr="W_PREMIUM", default=W_PREMIUM, kind="weight",
                candidates=[0.0, 0.5, 2.0], requires_search=True, needs_python_evaluator=False, clear_caches=False,
                parse=tuning._parse_float,
                description="player-trade premium per unit of the partner's win-probability gain x their danger "
                            "multiplier (0 = plain bank floor: ties go to the bank); only with search.acq_floor=1")
    registry[t.name] = t
    om = "catanbot.opponent_model"
    for attr, default, cands, parse, desc in (
            ("CALIB_RATE", 0.0, [1.0, 0.5], tuning._parse_float,
             "acq.calib: online calibration of predict_accept per opponent from every observed answer (0 = off)"),
            ("CALIB_PRIOR", 0.0, [-1.0], tuning._parse_float,
             "acq.calib: starting shared intercept of the calibration; only with opponent_model.CALIB_RATE > 0"),
            ("REJECT_STREAK", 0, [3], tuning._parse_int,
             "acq.calib fallback: a seat that rejected this many offers in a row gets P(accept) = 0 until it accepts "
             "one (0 = off)")):
        t = Tunable(name=f"opponent_model.{attr}", module=om, attr=attr, default=default, kind="weight",
                    candidates=list(cands), requires_search=True, needs_python_evaluator=False, clear_caches=False,
                    parse=parse, description=desc + " (the search bot's opponent model)")
        registry[t.name] = t
    tuning.register_seat_observer("trades", TradeObserver)     # self-play mechanism counters (observers=...)
