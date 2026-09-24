"""Rule-based bot: picks the action with the highest heuristic prior.

Used as the training opponent / bootstrap and as a strong-ish baseline.
``temperature`` > 0 samples from a softmax over priors for diversity in
self-play; ``epsilon`` adds uniformly random moves.
"""
from __future__ import annotations

import math
from typing import List, Optional

from .. import actions as A
from ..actions import Action
from ..heuristic import action_priors
from ..state import GameState
from .base import Bot


class HeuristicBot(Bot):
    name = "heuristic"

    def __init__(self, temperature: float = 0.0, epsilon: float = 0.0, max_trades_per_turn: int = 2):
        self.temperature = temperature
        self.epsilon = epsilon
        self.max_trades_per_turn = max_trades_per_turn
        self._last_reason: Optional[str] = None

    def decide(self, state: GameState, legal_actions: List[Action], rng) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return legal_actions[rng.randrange(len(legal_actions))]
        player = self._acting(state)
        actions = legal_actions
        # Limit proposal spam: only propose when we have not already proposed twice.
        if state.trades_this_turn >= self.max_trades_per_turn:
            filtered = [a for a in actions if a[0] != A.PROPOSE_TRADE]
            if filtered:
                actions = filtered
        priors = action_priors(state, actions, player)
        if self.temperature <= 0:
            best = max(range(len(actions)), key=lambda i: priors[i])
            # Break exact ties randomly for diversity.
            top = [i for i in range(len(actions)) if priors[i] == priors[best]]
            return actions[top[rng.randrange(len(top))]]
        m = max(priors)
        weights = [math.exp((p - m) / self.temperature) for p in priors]
        x = rng.random() * sum(weights)
        acc = 0.0
        for a, w in zip(actions, weights):
            acc += w
            if x <= acc:
                return a
        return actions[-1]

    @staticmethod
    def _acting(state: GameState) -> int:
        from ..state import PHASE_DISCARD, PHASE_TRADE_RESPONSE
        if state.phase == PHASE_DISCARD and state.discard_queue:
            return state.discard_queue[0]
        if state.phase == PHASE_TRADE_RESPONSE:
            return state.trade_responder
        return state.current
