"""Uniformly random baseline (never proposes trades, which would only add noise)."""
from __future__ import annotations

from typing import List

from .. import actions as A
from ..actions import Action
from ..state import GameState
from .base import Bot


class RandomBot(Bot):
    name = "random"

    def __init__(self, end_turn_bias: float = 0.0):
        # Probability of ending the turn immediately when allowed; a small
        # bias keeps random games short enough for benchmarks.
        self.end_turn_bias = end_turn_bias

    def decide(self, state: GameState, legal_actions: List[Action], rng) -> Action:
        candidates = [a for a in legal_actions if a[0] != A.PROPOSE_TRADE] or legal_actions
        if self.end_turn_bias > 0 and (A.END_TURN,) in candidates and rng.random() < self.end_turn_bias:
            return (A.END_TURN,)
        return candidates[rng.randrange(len(candidates))]
