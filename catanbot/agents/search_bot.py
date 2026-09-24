"""The real bot: value net (or heuristic) + expectimax search + opponent model.

``epsilon`` / ``temperature`` add exploration for self-play data generation:
with temperature > 0 the action is sampled from a softmax over the search
values (scaled so that a 0.05 win-probability edge is decisive at
temperature 1), with epsilon a uniformly random legal action is played.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

from .. import actions as A
from .. import engine as E
from ..actions import Action
from ..counting import HandBelief
from ..heuristic import HeuristicEvaluator, action_priors
from ..opponent_model import OpponentModel
from ..search import ScoredAction, SearchConfig, Searcher
from ..state import GameState
from .base import Bot


class SearchBot(Bot):
    name = "search"

    def __init__(self, evaluator=None, config: Optional[SearchConfig] = None, epsilon: float = 0.0,
                 temperature: float = 0.0, model: Optional[OpponentModel] = None,
                 track_opponents: bool = True, name: Optional[str] = None):
        self.evaluator = evaluator or HeuristicEvaluator()
        self.config = config or SearchConfig(depth=1, beam=4, expand=8)
        self.epsilon = epsilon
        self.temperature = temperature
        self.model = model
        self.track_opponents = track_opponents
        self.belief: Optional[HandBelief] = None
        self.last_results: List[ScoredAction] = []
        self._searcher: Optional[Searcher] = None
        if name:
            self.name = name

    def reset(self) -> None:
        if self.track_opponents:
            self.model = OpponentModel()
        self.belief = None
        self.last_results = []
        self._searcher = None

    def _searcher_for(self) -> Searcher:
        if self._searcher is None or self._searcher.model is not self.model:
            self._searcher = Searcher(self.evaluator, self.config, self.model, self.belief)
        return self._searcher

    def decide(self, state: GameState, legal_actions: List[Action], rng) -> Action:
        if len(legal_actions) == 1:
            self.last_results = [ScoredAction(legal_actions[0], 0.0)]
            return legal_actions[0]
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return legal_actions[rng.randrange(len(legal_actions))]
        if self.model is not None:
            self.model.attach(state)
        searcher = self._searcher_for()
        me = E.acting_player(state)
        results = searcher.search(state, me, random.Random(rng.random()))
        self.last_results = results
        if not results:
            return legal_actions[0]
        if self.temperature > 0 and len(results) > 1:
            scale = 20.0 / self.temperature  # 0.05 edge ~ e^1
            m = results[0].value
            weights = [math.exp((r.value - m) * scale) for r in results]
            x = rng.random() * sum(weights)
            acc = 0.0
            for r, w in zip(results, weights):
                acc += w
                if x <= acc:
                    return r.action
        return results[0].action

    def observe(self, state: GameState, action: Action, player: int) -> None:
        if self.model is None or not self.track_opponents:
            return
        predicted = None
        if action[0] in (A.PROPOSE_TRADE, A.ACCEPT_TRADE, A.REJECT_TRADE, A.MOVE_ROBBER, A.PLAY_KNIGHT,
                         A.BUILD_CITY, A.BUILD_SETTLEMENT, A.BUY_DEV):
            try:
                legal = E.legal_actions(state)
                if 1 < len(legal) <= 40:
                    priors = action_priors(state, legal, player)
                    predicted = legal[max(range(len(legal)), key=lambda i: priors[i])]
            except Exception:
                predicted = None
        self.model.observe(state, action, player, predicted)

    def explain(self, state: GameState) -> Optional[str]:
        if not self.last_results:
            return None
        r = self.last_results[0]
        return f"{A.describe(r.action, state)} (value {r.value:.3f}): {r.explanation}"
