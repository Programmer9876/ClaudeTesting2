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
from ..politics import PoliticalState
from ..search import ScoredAction, SearchConfig, Searcher
from ..state import GameState, PHASE_MAIN, PHASE_TRADE_RESPONSE
from .base import Bot


class SearchBot(Bot):
    name = "search"

    def __init__(self, evaluator=None, config: Optional[SearchConfig] = None, epsilon: float = 0.0,
                 temperature: float = 0.0, model: Optional[OpponentModel] = None,
                 track_opponents: bool = True, name: Optional[str] = None,
                 accept_bias: float = 0.0, offer_temp: float = 0.0, trade_eps: float = 0.0):
        self.evaluator = evaluator or HeuristicEvaluator()
        self.config = config or SearchConfig(depth=1, beam=4, expand=8)
        self.epsilon = epsilon
        self.temperature = temperature
        # Trading-style randomisation for self-play (DESIGN section 11), see agents/heuristic_bot.py:
        # per-game acceptance bias (half-width), temperature over the proposal ranking, random trades.
        self.accept_bias = accept_bias
        self.offer_temp = offer_temp
        self.trade_eps = trade_eps
        self.trade_bias: Optional[float] = None
        self.model = model
        self.track_opponents = track_opponents
        self.belief: Optional[HandBelief] = None
        self.politics: Optional[PoliticalState] = None
        self.last_results: List[ScoredAction] = []
        self._searcher: Optional[Searcher] = None
        if name:
            self.name = name

    def reset(self) -> None:
        if self.track_opponents:
            self.model = OpponentModel()
            self.politics = None
        self.belief = None
        self.last_results = []
        self._searcher = None
        self.trade_bias = None

    def _game_bias(self, rng) -> float:
        if self.trade_bias is None:
            self.trade_bias = rng.uniform(-self.accept_bias, self.accept_bias) if self.accept_bias > 0 else 0.0
        return self.trade_bias

    def _searcher_for(self) -> Searcher:
        if (self._searcher is None or self._searcher.model is not self.model
                or self._searcher.politics is not self.politics):
            self._searcher = Searcher(self.evaluator, self.config, self.model, self.belief, self.politics)
        return self._searcher

    def _ensure_politics(self, state: GameState) -> None:
        if self.track_opponents and (self.politics is None or self.politics.n != state.num_players):
            self.politics = PoliticalState(state.num_players)

    def decide(self, state: GameState, legal_actions: List[Action], rng) -> Action:
        if state.allow_counters and not self.config.counters and state.phase == PHASE_TRADE_RESPONSE:
            # Counter-offer rules, bot without counters: answer like a bot that does not know them.
            legal_actions = [a for a in legal_actions if a[0] != A.COUNTER_TRADE] or legal_actions
        if len(legal_actions) == 1:
            self.last_results = [ScoredAction(legal_actions[0], 0.0)]
            return legal_actions[0]
        bias = self._game_bias(rng)
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return legal_actions[rng.randrange(len(legal_actions))]
        if self.trade_eps > 0 and rng.random() < self.trade_eps:
            if state.phase == PHASE_TRADE_RESPONSE:
                a = legal_actions[rng.randrange(len(legal_actions))]
                self.last_results = [ScoredAction(a, 0.0, "random response (trade_eps)")]
                return a
            proposals = [a for a in legal_actions if a[0] == A.PROPOSE_TRADE]
            if proposals and state.phase == PHASE_MAIN:
                a = proposals[rng.randrange(len(proposals))]
                self.last_results = [ScoredAction(a, 0.0, "random proposal (trade_eps)")]
                return a
        if self.model is not None:
            self.model.attach(state)
        self._ensure_politics(state)
        searcher = self._searcher_for()
        me = E.acting_player(state)
        results = searcher.search(state, me, random.Random(rng.random()))
        if bias != 0.0 and state.phase == PHASE_TRADE_RESPONSE and len(results) > 1:
            # Per-game acceptance bias: 0.02 win probability per unit, like should_accept's margin.
            results = [ScoredAction(r.action, r.value + (0.02 * bias if r.action[0] == A.ACCEPT_TRADE else 0.0),
                                    r.explanation, r.line, r.static) for r in results]
            results.sort(key=lambda r: -r.value)
        self.last_results = results
        if not results:
            return legal_actions[0]
        chosen = self._sample(results, self.temperature, rng)
        if self.offer_temp > 0 and chosen[0] == A.PROPOSE_TRADE:
            # Temperature over the offer ranking only: which deal we ask for varies between games.
            proposals = [r for r in results if r.action[0] == A.PROPOSE_TRADE]
            chosen = self._sample(proposals, self.offer_temp, rng)
        return chosen

    @staticmethod
    def _sample(results: List[ScoredAction], temperature: float, rng) -> Action:
        if temperature > 0 and len(results) > 1:
            scale = 20.0 / temperature  # 0.05 edge ~ e^1
            m = max(r.value for r in results)
            weights = [math.exp((r.value - m) * scale) for r in results]
            x = rng.random() * sum(weights)
            acc = 0.0
            for r, w in zip(results, weights):
                acc += w
                if x <= acc:
                    return r.action
        return results[0].action

    def observe(self, state: GameState, action: Action, player: int) -> None:
        if not self.track_opponents:
            return
        self._ensure_politics(state)
        if self.politics is not None:
            try:
                self.politics.observe(state, action, player)
            except Exception:
                pass
        if self.model is None:
            return
        predicted = None
        if action[0] == A.COUNTER_TRADE and self.belief is not None:
            try:     # card counting: they hold what they offered, and probably lack what they asked for
                self.belief.observe_counter(player, action[1], action[2])
            except Exception:
                pass
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
