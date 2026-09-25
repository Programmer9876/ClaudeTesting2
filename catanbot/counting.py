"""Card counting: bank stock, development deck and opponent hand beliefs.

Two usage modes:

* **Live play** - a :class:`HandBelief` is updated from public events
  (production, builds, bank trades, discards, monopolies, steals) so the bot
  knows what opponents are likely holding.
* **Screenshot** - only hand sizes are known; :func:`expected_opponent_hands`
  builds a prior from each opponent's production mix.

Everything here is an *expected count* model (one float per resource per
player) which is cheap and good enough for trade / monopoly decisions.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from . import board as B
from .placement import player_production
from .state import GameState

TOTAL_RESOURCE_CARDS = B.BANK_PER_RESOURCE * 5


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------
def bank_remaining(state: GameState) -> List[int]:
    """Cards left in the bank per resource (as tracked in the state)."""
    return list(state.bank)


def bank_is_short(state: GameState, res: int, need: int = 1) -> bool:
    """True if the bank cannot supply ``need`` of ``res`` (relevant for trades)."""
    return state.bank[res] < need


def resources_in_play(state: GameState) -> List[int]:
    """Cards outside the bank per resource (known hands only; unknown hands add to totals)."""
    out = [B.BANK_PER_RESOURCE - b for b in state.bank]
    return out


# ---------------------------------------------------------------------------
# Development deck
# ---------------------------------------------------------------------------
def dev_pool(state: GameState) -> List[int]:
    """Cards of each type that are *not accounted for publicly*.

    pool = standard deck - knights played - dev cards held by players whose
    hands are known.  Unknown opponent hands and the undrawn deck are both
    drawn from this pool.
    """
    pool = list(B.DEV_DECK_COUNTS)
    for p in state.players:
        pool[B.DEV_KNIGHT] -= p.played_knights
        if p.dev_known:
            for t in range(5):
                pool[t] -= p.dev_cards[t] + p.dev_cards_new[t]
    return [max(0, x) for x in pool]


def dev_draw_probabilities(state: GameState) -> List[float]:
    """Probability that the next bought card is of each type."""
    deck = state.dev_deck
    total = sum(deck)
    if total <= 0:
        pool = dev_pool(state)
        total = sum(pool)
        if total <= 0:
            return [0.0] * 5
        return [x / total for x in pool]
    return [x / total for x in deck]


def dev_cards_remaining(state: GameState) -> int:
    return sum(state.dev_deck)


def expected_hidden_vp(state: GameState, player: int) -> float:
    """Expected number of VP cards held by ``player`` (exact if known)."""
    p = state.players[player]
    if p.dev_known:
        return float(p.vp_cards)
    pool = dev_pool(state)
    total = sum(pool)
    if total <= 0 or p.dev_count <= 0:
        return 0.0
    return p.dev_count * pool[B.DEV_VP] / total


# ---------------------------------------------------------------------------
# Opponent hands
# ---------------------------------------------------------------------------
def hand_prior_weights(state: GameState, player: int, smoothing: float = 0.04) -> List[float]:
    """Relative likelihood of each resource in an opponent's unknown hand.

    Production-weighted (cards come from rolls) with smoothing so trades,
    steals and starting resources are covered.  Sums to 1.
    """
    prod = player_production(state, player, ignore_robber=True)
    w = [prod[r] + smoothing for r in range(5)]
    # Cards the bank has run out of are proportionally more likely to be in hands.
    for r in range(5):
        w[r] *= 1.0 + 0.5 * (1.0 - state.bank[r] / B.BANK_PER_RESOURCE)
    tot = sum(w)
    return [x / tot for x in w]


def expected_opponent_hands(state: GameState, me: Optional[int] = None) -> List[List[float]]:
    """Expected resource counts for every player (exact where the hand is known)."""
    out: List[List[float]] = []
    for i, p in enumerate(state.players):
        if p.hand_known or i == me:
            out.append([float(x) for x in p.resources])
        else:
            w = hand_prior_weights(state, i)
            out.append([w[r] * p.hand_size for r in range(5)])
    return out


class HandBelief:
    """Expected-count belief over every player's hand, updated from public events.

    ``expected[i][r]`` is the expected number of cards of resource ``r`` in
    player ``i``'s hand; ``size[i]`` is the exact hand size (always public).
    """

    def __init__(self, state: GameState):
        self.n = state.num_players
        self.expected: List[List[float]] = []
        self.size: List[int] = []
        for i, p in enumerate(state.players):
            if p.hand_known:
                self.expected.append([float(x) for x in p.resources])
                self.size.append(sum(p.resources))
            else:
                w = hand_prior_weights(state, i)
                self.expected.append([w[r] * p.hand_size for r in range(5)])
                self.size.append(p.hand_size)

    # --- helpers ------------------------------------------------------------
    def _renormalise(self, i: int) -> None:
        e = self.expected[i]
        for r in range(5):
            if e[r] < 0:
                e[r] = 0.0
        tot = sum(e)
        n = self.size[i]
        if n <= 0:
            self.expected[i] = [0.0] * 5
        elif tot <= 1e-9:
            self.expected[i] = [n / 5.0] * 5
        elif abs(tot - n) > 1e-9:
            f = n / tot
            self.expected[i] = [x * f for x in e]

    def set_exact(self, i: int, resources: Sequence[int]) -> None:
        self.expected[i] = [float(x) for x in resources]
        self.size[i] = int(sum(resources))

    # --- public events --------------------------------------------------------
    def observe_gain(self, i: int, res: int, n: int = 1) -> None:
        self.expected[i][res] += n
        self.size[i] += n

    def observe_spend(self, i: int, counts: Sequence[int]) -> None:
        for r in range(5):
            self.expected[i][r] -= counts[r]
            self.size[i] -= counts[r]
        self._renormalise(i)

    def observe_bank_trade(self, i: int, give_res: int, ratio: int, get_res: int) -> None:
        self.expected[i][give_res] -= ratio
        self.size[i] -= ratio
        self._renormalise(i)
        self.observe_gain(i, get_res, 1)

    def observe_player_trade(self, a: int, give: Sequence[int], b: int, get: Sequence[int]) -> None:
        """``a`` gives ``give`` to ``b`` and receives ``get`` from ``b``."""
        self.observe_spend(a, give)
        self.observe_spend(b, get)
        for r in range(5):
            if give[r]:
                self.observe_gain(b, r, give[r])
            if get[r]:
                self.observe_gain(a, r, get[r])

    def observe_discard(self, i: int, counts: Sequence[int]) -> None:
        self.observe_spend(i, counts)

    def observe_steal(self, victim: int, thief: int, res: Optional[int] = None) -> None:
        """A random card moved from victim to thief; ``res`` if we saw it."""
        if self.size[victim] <= 0:
            return
        if res is not None:
            self.expected[victim][res] -= 1
            self.size[victim] -= 1
            self._renormalise(victim)
            self.observe_gain(thief, res, 1)
            return
        n = self.size[victim]
        probs = [self.expected[victim][r] / n for r in range(5)]
        for r in range(5):
            self.expected[victim][r] -= probs[r]
            self.expected[thief][r] += probs[r]
        self.size[victim] -= 1
        self.size[thief] += 1
        self._renormalise(victim)

    def observe_monopoly(self, i: int, res: int, taken: Optional[int] = None) -> None:
        """Player ``i`` monopolised ``res``.  ``taken`` = number of cards collected (public);
        if unknown, the expected total is used.  Card counts are conserved exactly."""
        shares = [self.expected[j][res] if j != i else 0.0 for j in range(self.n)]
        total = sum(shares)
        if taken is None:
            taken = int(round(total))
        taken = max(0, min(taken, sum(self.size[j] for j in range(self.n) if j != i)))
        # integer shares per victim by largest remainder, bounded by their hand sizes
        ints = [0] * self.n
        if taken > 0 and total > 0:
            raw = [taken * s / total for s in shares]
            ints = [min(int(r), self.size[j]) for j, r in enumerate(raw)]
            rem = taken - sum(ints)
            order = sorted((j for j in range(self.n) if j != i), key=lambda j: -(raw[j] - int(raw[j])))
            k = 0
            while rem > 0 and order:
                j = order[k % len(order)]
                if ints[j] < self.size[j]:
                    ints[j] += 1
                    rem -= 1
                k += 1
                if k > 4 * self.n:
                    break
        for j in range(self.n):
            if j == i:
                continue
            self.size[j] -= ints[j]
            self.expected[j][res] = 0.0
            self._renormalise(j)
        self.expected[i][res] += sum(ints)
        self.size[i] += sum(ints)

    def observe_hand_size(self, i: int, size: int) -> None:
        """Resynchronise with the public hand size (e.g. after an unobserved event)."""
        self.size[i] = size
        self._renormalise(i)

    def sync(self, state: GameState) -> None:
        """Snap known hands to their exact contents and sizes to the state."""
        for i, p in enumerate(state.players):
            if p.hand_known:
                self.set_exact(i, p.resources)
            else:
                self.observe_hand_size(i, p.hand_size)

    # --- queries -------------------------------------------------------------
    def expected_hand(self, i: int) -> List[float]:
        return list(self.expected[i])

    def probability_has(self, i: int, res: int, at_least: int = 1) -> float:
        """P(player i holds >= ``at_least`` of ``res``) under a binomial approximation."""
        n = self.size[i]
        if n <= 0 or at_least > n:
            return 0.0
        p = min(1.0, max(0.0, self.expected[i][res] / n))
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        # 1 - P(X < at_least), X ~ Binomial(n, p)
        from math import comb
        below = sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(at_least))
        return max(0.0, min(1.0, 1.0 - below))


def monopoly_expected_take(state: GameState, player: int, res: int,
                           belief: Optional[HandBelief] = None) -> float:
    """Expected number of cards a Monopoly on ``res`` would collect."""
    if belief is None:
        hands = expected_opponent_hands(state, me=player)
        return sum(hands[j][res] for j in range(state.num_players) if j != player)
    return sum(belief.expected[j][res] for j in range(state.num_players) if j != player)
