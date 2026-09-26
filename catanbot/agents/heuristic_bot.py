"""Rule-based bot: picks the action with the highest heuristic prior.

Used as the training opponent / bootstrap and as a strong-ish baseline.
``temperature`` > 0 samples from a softmax over priors for diversity in
self-play; ``epsilon`` adds uniformly random moves.

Trading style randomisation (DESIGN section 11, self-play only):

* ``accept_bias`` - half-width of a per-game acceptance bias ``b ~ U(-a, a)``
  (sampled from the game rng on the first decision) that is added to
  ``trading.should_accept``'s threshold: positive games are generous,
  negative ones stingy.  The bias used in the current game is ``trade_bias``.
* ``offer_temp`` - temperature (on the prior scale) over the *proposal*
  ranking: instead of always making the top-ranked offer, sample one.
* ``trade_eps`` - probability of answering an offer at random / making a
  random proposal on our turn.
"""
from __future__ import annotations

import math
from typing import List, Optional

from .. import actions as A
from ..actions import Action
from ..devcards import without_dev_buys
from ..heuristic import action_priors
from ..state import GameState, PHASE_MAIN, PHASE_TRADE_RESPONSE
from ..trading import should_accept
from .base import Bot


class HeuristicBot(Bot):
    name = "heuristic"

    def __init__(self, temperature: float = 0.0, epsilon: float = 0.0, max_trades_per_turn: int = 2,
                 accept_bias: float = 0.0, offer_temp: float = 0.0, trade_eps: float = 0.0):
        self.temperature = temperature
        self.epsilon = epsilon
        self.max_trades_per_turn = max_trades_per_turn
        self.accept_bias = accept_bias
        self.offer_temp = offer_temp
        self.trade_eps = trade_eps
        self.trade_bias: Optional[float] = None     # this game's sampled acceptance bias
        self._last_reason: Optional[str] = None

    def reset(self) -> None:
        self.trade_bias = None

    def _game_bias(self, rng) -> float:
        if self.trade_bias is None:
            self.trade_bias = rng.uniform(-self.accept_bias, self.accept_bias) if self.accept_bias > 0 else 0.0
        return self.trade_bias

    def decide(self, state: GameState, legal_actions: List[Action], rng) -> Action:
        legal_actions = without_dev_buys(legal_actions)     # devcards.buy off (Q22 check); the same list when on
        if len(legal_actions) == 1:
            return legal_actions[0]
        bias = self._game_bias(rng)
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return legal_actions[rng.randrange(len(legal_actions))]
        player = self._acting(state)
        if state.phase == PHASE_TRADE_RESPONSE and state.pending_trade is not None:
            if self.trade_eps > 0 and rng.random() < self.trade_eps:
                return legal_actions[rng.randrange(len(legal_actions))]
            if bias != 0.0:
                ok, _ = should_accept(state, player, state.pending_trade, accept_bias=bias)
                accept = (A.ACCEPT_TRADE,)
                return accept if ok and accept in legal_actions else (A.REJECT_TRADE,)
        actions = legal_actions
        # Limit proposal spam: only propose when we have not already proposed twice.
        if state.trades_this_turn >= self.max_trades_per_turn:
            filtered = [a for a in actions if a[0] != A.PROPOSE_TRADE]
            if filtered:
                actions = filtered
        elif self.trade_eps > 0 and state.phase == PHASE_MAIN and rng.random() < self.trade_eps:
            proposals = [a for a in actions if a[0] == A.PROPOSE_TRADE]
            if proposals:
                return proposals[rng.randrange(len(proposals))]
        priors = action_priors(state, actions, player)
        chosen = self._sample(actions, priors, self.temperature, rng)
        if self.offer_temp > 0 and chosen[0] == A.PROPOSE_TRADE:
            # Temperature over the offer ranking only: which deal we ask for varies between games.
            proposals = [(a, p) for a, p in zip(actions, priors) if a[0] == A.PROPOSE_TRADE]
            chosen = self._sample([a for a, _ in proposals], [p for _, p in proposals], self.offer_temp, rng)
        return chosen

    @staticmethod
    def _sample(actions: List[Action], priors: List[float], temperature: float, rng) -> Action:
        if temperature <= 0:
            best = max(range(len(actions)), key=lambda i: priors[i])
            # Break exact ties randomly for diversity.
            top = [i for i in range(len(actions)) if priors[i] == priors[best]]
            return actions[top[rng.randrange(len(top))]]
        m = max(priors)
        weights = [math.exp((p - m) / temperature) for p in priors]
        x = rng.random() * sum(weights)
        acc = 0.0
        for a, w in zip(actions, weights):
            acc += w
            if x <= acc:
                return a
        return actions[-1]

    @staticmethod
    def _acting(state: GameState) -> int:
        from ..state import PHASE_DISCARD
        if state.phase == PHASE_DISCARD and state.discard_queue:
            return state.discard_queue[0]
        if state.phase == PHASE_TRADE_RESPONSE:
            return state.trade_responder
        return state.current
