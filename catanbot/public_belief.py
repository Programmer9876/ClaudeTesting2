"""The public-information belief shared by the card-counting trackers.

Two trackers turn a public event stream into a :class:`catanbot.counting.CardCounter` plus the
development-card bookkeeping of a Colonist.io player:

* :class:`catanbot.bench.public_info.PublicInfoTracker` - catanatron games, from catanatron's
  action log (the ``counted`` information mode of the benchmarks);
* :class:`catanbot.colonist_log.ColonistLogTracker` - the advisor, from Colonist.io's game log
  (pasted text or screenshots).

Both hand the bot the same views, implemented once here in :class:`PublicBelief` (no catanatron
import): :meth:`PublicBelief.public_view` (every opponent ``hand_known=False`` with the exact
``hand_size``, ``dev_known=False`` with the exact ``dev_count``, the deck as the public pool's
expected composition), :meth:`PublicBelief.canonical_view` (the most likely hand hypothesis) and
:meth:`PublicBelief.determinize` (a joint hand hypothesis drawn from the counter, development
cards dealt from the public pool, conditioned on no opponent already holding enough VP cards to
have won).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from . import board as B
from .counting import CardCounter
from .state import PHASE_TRADE_RESPONSE, GameState

__all__ = ["PublicBelief", "apportion", "redact_state"]


def apportion(pool: Sequence[int], k: int) -> List[int]:
    """``k`` cards spread over the types in proportion to ``pool`` (largest remainder, capped by
    ``pool``): the expected composition of a ``k``-card deck drawn from ``pool``."""
    pool = [max(0, int(x)) for x in pool]
    tot = sum(pool)
    k = max(0, int(k))
    if tot <= 0 or k <= 0:
        return [0] * len(pool)
    k = min(k, tot)
    raw = [k * x / tot for x in pool]
    out = [min(int(r), pool[t]) for t, r in enumerate(raw)]
    order = sorted(range(len(pool)), key=lambda t: (-(raw[t] - int(raw[t])), t))
    i = 0
    while sum(out) < k and i < 10 * len(pool):
        t = order[i % len(order)]
        if out[t] < pool[t]:
            out[t] += 1
        i += 1
    return out


def _public_pool_estimate(cb: GameState, me: int) -> List[int]:
    """Public development pool from a converted state alone: the deck minus played knights minus
    our own cards (the trackers' :meth:`PublicBelief.dev_pool` also removes the other played
    types, which a :class:`GameState` does not record)."""
    pool = list(B.DEV_DECK_COUNTS)
    for p in cb.players:
        pool[B.DEV_KNIGHT] -= p.played_knights
    mine = cb.players[me]
    for t in range(5):
        pool[t] -= mine.dev_cards[t] + mine.dev_cards_new[t]
    return [max(0, x) for x in pool]


def redact_state(cb: GameState, me: int, deck: Optional[Sequence[int]] = None) -> GameState:
    """The public view of a converted state for seat ``me``.

    Every opponent: ``hand_known=False`` with the exact ``hand_size`` and zeroed ``resources``;
    ``dev_known=False`` with the exact ``dev_count`` and zeroed cards.  The development deck
    becomes ``deck`` (default: the public pool's expected composition, with the true size).  A
    pending offer keeps only the answers already given (the adapter marks a seat still to be
    asked that cannot pay as rejected - that reads its hand).  Our own seat is untouched.
    """
    s = cb.copy()
    for j, p in enumerate(s.players):
        if j == me:
            continue
        if p.hand_known:
            p.hand_size = sum(p.resources)
        p.resources = [0] * 5
        p.hand_known = False
        if p.dev_known:
            p.dev_count = sum(p.dev_cards) + sum(p.dev_cards_new)
        p.dev_cards = [0] * 5
        p.dev_cards_new = [0] * 5
        p.dev_known = False
    size = sum(cb.dev_deck)
    s.dev_deck = list(deck) if deck is not None else apportion(_public_pool_estimate(cb, me), size)
    if s.phase == PHASE_TRADE_RESPONSE and s.pending_trade is not None:
        offer = s.pending_trade
        for i in list(offer.responses):
            if i > s.trade_responder:
                del offer.responses[i]
    return s


def _weighted_pick(weights: Sequence[int], rng) -> int:
    tot = sum(weights)
    if tot <= 0:
        return -1
    x = rng.randrange(tot)
    for t, w in enumerate(weights):
        if x < w:
            return t
        x -= w
    return len(weights) - 1


class PublicBelief:
    """What a card counter believes, and the states it hands the bot (mixin of both trackers).

    The subclass keeps these attributes up to date: ``me`` / ``n`` (our seat, the number of
    players), ``counter`` (:class:`~catanbot.counting.CardCounter` over every hand),
    ``dev_count`` (every player's unplayed development cards), ``played`` (per player the played
    types), ``bought_this_turn``, ``known_dev`` (per player the exact types, or ``None`` when only
    the count is public), ``my_dev`` (our own types), ``deck`` (cards left in the deck),
    ``pending_discards`` (seat -> hidden cards discarded on the 7 being resolved, applied jointly
    later) and ``vps_to_win``.
    ``new_devs_unplayable``: cards bought this turn are dealt as not yet playable to the player
    on turn (the official rule; catanatron 3.2.1 lets them be played).
    """

    new_devs_unplayable = True

    me: int
    n: int
    counter: Optional[CardCounter]
    dev_count: List[int]
    played: List[List[int]]
    bought_this_turn: List[int]
    known_dev: List[Optional[List[int]]]
    my_dev: List[int]
    deck: int
    pending_discards: Dict[int, int]
    vps_to_win: int

    # --- development cards ---------------------------------------------------
    def dev_pool(self) -> List[int]:
        """Cards whose location is not public: the 25-card deck minus every played card minus
        our own (and, with ``reveal_hidden``, minus the opponents' tracked cards).  Opponents'
        unknown cards and the undrawn deck are both drawn from it."""
        pool = list(B.DEV_DECK_COUNTS)
        for j in range(self.n):
            for t in range(5):
                pool[t] -= self.played[j][t]
                kd = self.known_dev[j]
                if kd is not None:
                    pool[t] -= kd[t]
        for t in range(5):
            pool[t] -= self.my_dev[t]
        return [max(0, x) for x in pool]

    def unknown_dev_holders(self) -> List[int]:
        return [j for j in range(self.n) if j != self.me and self.known_dev[j] is None]

    def pool_is_consistent(self) -> bool:
        """``sum(pool) == deck size + the opponents' unknown card counts`` (public identity)."""
        return sum(self.dev_pool()) == self.deck + sum(self.dev_count[j] for j in self.unknown_dev_holders())

    # --- the belief (pending hidden discards included) ---------------------------
    def is_exact(self, j: int) -> bool:
        """True when seat ``j``'s hand is known exactly from the public information."""
        c = self.counter
        if not c.is_exact(j):
            return False
        k = self.pending_discards.get(j, 0)
        return k <= 0 or sum(1 for x in next(iter(c.hyps))[j] if x > 0) <= 1

    def support_weight(self, hands: Sequence[Sequence[int]]) -> float:
        """Probability of the exact joint assignment ``hands`` (0: excluded by the public information)."""
        pend = self.pending_discards
        if not pend:
            return self.counter.weight_of(hands)
        want = [tuple(int(x) for x in h) for h in hands]
        total = 0.0
        for joint, w in self.counter.hyps.items():
            p = w
            for j in range(self.n):
                if j in pend:
                    d = [joint[j][r] - want[j][r] for r in range(5)]
                    if min(d) < 0 or sum(d) != pend[j]:
                        p = 0.0
                        break
                    p *= math.prod(math.comb(joint[j][r], d[r]) for r in range(5)) / math.comb(sum(joint[j]), pend[j])
                elif joint[j] != want[j]:
                    p = 0.0
                    break
            total += p
        return total

    def expected_hand(self, j: int) -> List[float]:
        """Expected hand of seat ``j`` (a pending hidden discard removes cards in proportion)."""
        e = list(self.counter.expected[j])
        k = self.pending_discards.get(j, 0)
        n = sum(e)
        if k and n > 0:
            e = [x * (n - k) / n for x in e]
        return e

    def _discard_pending(self, hand: List[int], k: int, rng=None) -> List[int]:
        """``hand`` after a pending hidden discard of ``k`` cards: uniformly random cards (``rng``)
        or, deterministically, from the most held resources."""
        h = list(hand)
        for _ in range(min(k, sum(h))):
            if rng is None:
                r = max(range(5), key=lambda x: (h[x], -x))
            else:
                r = _weighted_pick(h, rng)
            h[r] -= 1
        return h

    # --- views for the bot ----------------------------------------------------
    def public_view(self, cb: GameState) -> GameState:
        """The state handed to the bot's hooks: :func:`redact_state` with the deck as the
        expected composition of the public pool."""
        return redact_state(cb, self.me, apportion(self.dev_pool(), sum(cb.dev_deck)))

    def canonical_view(self, pub: GameState) -> GameState:
        """``pub`` with the opponents' hands of the most likely hypothesis (deterministic): the
        state our own legal actions are generated on."""
        s = pub.copy()
        joint = self.counter.most_likely()
        for j, p in enumerate(s.players):
            if j != self.me:
                p.resources = self._discard_pending(list(joint[j]), self.pending_discards.get(j, 0))
                p.hand_known = True
                p.hand_size = sum(p.resources)
        return s

    def determinize(self, pub: GameState, rng) -> GameState:
        """A fully specified state sampled from the public information: the opponents' hands are
        one joint hypothesis of the counter (drawn with its probability), their development cards
        are dealt from the public pool (re-dealt, up to 20 times, while an opponent would already
        hold enough VP cards to have won) and the rest of the pool is the deck.  During a 7's
        discards an opponent's hidden discard so far is a uniformly random subset of its hand."""
        s = pub.copy()
        me = self.me
        joint = self.counter.sample(rng)
        for j, p in enumerate(s.players):
            if j != me:
                k = self.pending_discards.get(j, 0)
                p.resources = self._discard_pending(list(joint[j]), k, rng) if k else list(joint[j])
                p.hand_known = True
                p.hand_size = sum(p.resources)
        self._deal_devs(s, rng)
        if s.phase == PHASE_TRADE_RESPONSE and s.pending_trade is not None:
            offer = s.pending_trade
            for i in range(len(s.players)):
                if i in (offer.proposer, s.trade_responder) or i in offer.responses or i < s.trade_responder:
                    continue
                if any(s.players[i].resources[r] < offer.get[r] for r in range(5)):
                    offer.responses[i] = False   # catanbot's engine never asks a seat that cannot pay
        return s

    def _deal_devs(self, s: GameState, rng) -> None:
        pool = self.dev_pool()
        unknown = self.unknown_dev_holders()
        hands: Dict[int, List[int]] = {}
        left = list(pool)
        for _attempt in range(20):
            left = list(pool)
            hands = {}
            for j in unknown:
                cards = [0] * 5
                for _ in range(self.dev_count[j]):
                    t = _weighted_pick(left, rng)
                    if t < 0:
                        break
                    cards[t] += 1
                    left[t] -= 1
                hands[j] = cards
            if all(s.public_vp(j) + hands[j][B.DEV_VP] < self.vps_to_win for j in unknown):
                break
        else:
            # Still a winner after 20 deals (a near-empty pool of VP cards): trade its VP cards for
            # other types left in the pool where possible.
            for j in unknown:
                cards = hands[j]
                while s.public_vp(j) + cards[B.DEV_VP] >= self.vps_to_win and cards[B.DEV_VP] > 0:
                    alt = [t for t in range(5) if t != B.DEV_VP and left[t] > 0]
                    if not alt:
                        break
                    t = alt[rng.randrange(len(alt))]
                    cards[B.DEV_VP] -= 1
                    left[B.DEV_VP] += 1
                    cards[t] += 1
                    left[t] -= 1
        for j, p in enumerate(s.players):
            if j == self.me:
                continue
            cards = hands.get(j)
            if cards is None:
                cards = list(self.known_dev[j] or [0] * 5)
            new = [0] * 5
            n_new = self.bought_this_turn[j] if (self.new_devs_unplayable and j == s.current) else 0
            if n_new:
                # bought this turn: not playable yet (the official rule; VP cards count anyway)
                movable = [t for t in range(5) if t != B.DEV_VP for _ in range(cards[t])]
                rng.shuffle(movable)
                for t in movable[:n_new]:
                    new[t] += 1
            p.dev_cards = [cards[t] - new[t] for t in range(5)]
            p.dev_cards_new = new
            p.dev_known = True
            p.dev_count = sum(cards)
        s.dev_deck = left
