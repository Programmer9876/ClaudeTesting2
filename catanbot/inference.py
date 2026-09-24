"""Determinization of hidden information.

A state parsed from a screenshot knows opponents' hand sizes and dev card
counts but not their contents.  The engine and the search need fully
specified states, so we *sample* them: each unknown hand is filled with
cards drawn from a production-weighted prior (or from a live
:class:`~catanbot.counting.HandBelief`) constrained by what the bank can
still hold, and unknown dev cards are drawn from the public dev pool.

Averaging search results over several samples (``sample_states``) gives a
Perfect-Information-Monte-Carlo style estimate.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from . import board as B
from .counting import HandBelief, dev_pool, hand_prior_weights
from .state import GameState


def _weighted_pick(weights: Sequence[float], rng) -> int:
    tot = 0.0
    for w in weights:
        tot += w
    if tot <= 0:
        return -1
    x = rng.random() * tot
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x <= acc:
            return i
    return len(weights) - 1


def determinize(state: GameState, me: Optional[int] = None, rng=None,
                belief: Optional[HandBelief] = None) -> GameState:
    """Return a copy of ``state`` with every hidden hand filled in.

    * Players with ``hand_known=False`` receive ``hand_size`` cards sampled
      without replacement from the bank, weighted by their production prior
      (or ``belief.expected``).  The bank is decremented accordingly.
    * Players with ``dev_known=False`` receive ``dev_count`` dev cards drawn
      from the public dev pool; whatever remains of the pool becomes
      ``dev_deck`` (the undrawn deck).
    * ``me`` (if given) is always treated as known.
    """
    import random as _random
    rng = rng or _random.Random(0)
    s = state.copy()
    bank = list(s.bank)
    # ---- resources ---------------------------------------------------------
    for i, p in enumerate(s.players):
        if p.hand_known or i == me:
            if not p.hand_known:
                p.hand_known = True
                p.hand_size = sum(p.resources)
            continue
        n = p.hand_size
        if belief is not None:
            base = belief.expected_hand(i)
            tot = sum(base)
            w = [x / tot for x in base] if tot > 0 else hand_prior_weights(state, i)
        else:
            w = hand_prior_weights(state, i)
        counts = [0] * 5
        for _ in range(n):
            avail = [w[r] if bank[r] > 0 else 0.0 for r in range(5)]
            r = _weighted_pick(avail, rng)
            if r < 0:
                choices = [x for x in range(5) if bank[x] > 0]
                if not choices:
                    break
                r = choices[rng.randrange(len(choices))]
            counts[r] += 1
            bank[r] -= 1
        p.resources = counts
        p.hand_known = True
        p.hand_size = sum(counts)
    s.bank = bank
    # ---- dev cards ---------------------------------------------------------
    pool = dev_pool(state)
    # Remove dev cards of players that are known (already excluded by dev_pool)
    # and sample for the unknown ones.
    for i, p in enumerate(s.players):
        if p.dev_known or i == me:
            if not p.dev_known:
                p.dev_known = True
                p.dev_count = p.total_dev
            continue
        cards = [0] * 5
        for _ in range(p.dev_count):
            t = _weighted_pick(pool, rng)
            if t < 0:
                break
            cards[t] += 1
            pool[t] -= 1
        p.dev_cards = cards
        p.dev_cards_new = [0] * 5
        p.dev_known = True
        p.dev_count = sum(cards)
    if any(not q.dev_known for q in state.players):
        s.dev_deck = pool
    else:
        # Everything was known: keep the state's own deck estimate but never
        # let it exceed the public pool.
        s.dev_deck = [min(state.dev_deck[t], pool[t]) for t in range(5)]
    return s


def sample_states(state: GameState, me: Optional[int] = None, n: int = 8, rng=None,
                  belief: Optional[HandBelief] = None) -> List[GameState]:
    """``n`` independent determinizations of ``state``."""
    import random as _random
    rng = rng or _random.Random(0)
    if all(p.hand_known and p.dev_known for p in state.players):
        return [state.copy() for _ in range(max(1, n))]
    return [determinize(state, me, rng, belief) for _ in range(max(1, n))]


def is_fully_known(state: GameState) -> bool:
    return all(p.hand_known and p.dev_known for p in state.players)
