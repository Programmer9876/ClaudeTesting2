"""Expectimax search with beam pruning ("Stockfish style" for Catan).

Structure
---------
* **Decision nodes** are states where it is our move.  We expand the most
  promising legal actions (ordered by ``heuristic.action_priors``), keep a
  global beam of the best partial action sequences per level and stop when
  the turn ends (``END_TURN``) or the per-turn action cap is reached.
* **Chance nodes** appear after actions with random outcomes: the dice
  (exact expectation over the 11 totals), a dev card purchase (weighted by
  the remaining deck), a robber steal (weighted by the victim's hand) and
  trade proposals (accept / reject with the probability from the opponent
  model).  Every outcome is expanded, so the value of a sequence is a true
  expectation.
* **Finished nodes** (our turn is over) get a *future value*: the opponents
  play their turns with a greedy version of the same value function
  (max^n), with dice rolls sampled with common random numbers, until it is
  our turn again; then the leaf is evaluated by the value net (or, for
  ``depth >= 3``, by a reduced recursive search - for every leaf or for
  none, ``_leaves_affordable``).  Every finished node gets
  it (``finished_lookahead = 0``), so no candidate is valued at a different
  horizon than its siblings; the sampled part of the future value (its
  difference from the static value, minus the mean difference) is shrunk by
  its reliability ``n / (n + lookahead_shrink)`` because with few roll
  samples its noise is larger than the margins between candidates.
* Every state that needs a value is evaluated in a **batch** so a numpy
  value net can be used efficiently.

The search works for any phase in which we must act (roll, main, discard,
robber, trade response / select, setup).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import accel as _accel  # optional C++ extension: native opponent simulation for _future_values (docs/CPP.md)
from . import actions as A
from . import board as B
from . import engine as E
from .actions import Action
from .counting import HandBelief
from .devcards import should_buy_dev
from .discard import choose_discard, default_keep_targets, explain_seven_risk, seven_risk, surplus_dump_actions
from .heuristic import action_priors
from .opponent_model import OpponentModel, trade_stage_factor
from .placement import road_targets, score_city, vertex_production
from .politics import PoliticalState, political_trade_options
from .robber import best_robber_move, hex_damage, should_play_knight
from .state import (GameState, PHASE_DISCARD, PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER, PHASE_ROLL,
                    PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT, PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT)
from .trading import plan_trades, should_accept


@dataclass
class SearchConfig:
    depth: int = 2                  # turns of lookahead: 1 = end of our turn, 2 = + opponents' turns, 3 = + our next turn
    beam: int = 6                   # partial sequences kept per level
    expand: int = 10                # actions tried per decision node
    max_actions_per_turn: int = 6   # actions per turn (END_TURN forced afterwards)
    roll_samples: int = 11          # our own roll: 11 = exact expectation, fewer = most likely rolls
    opp_roll_samples: int = 12      # sampled roll sequences for opponents' turns (common random numbers)
    opponent_actions: int = 4       # greedy actions per opponent turn
    opponent_expand: int = 6        # candidates evaluated per opponent decision
    finished_lookahead: int = 0     # end-of-turn nodes that get the future value: 0 = all of them, N = the top N
    #                                 by static value (the pre-fix behaviour, which mixes horizons: DESIGN section 4)
    lookahead_shrink: float = 12.0  # k: a node's sampled lookahead delta counts n / (n + k) with n roll samples
    #                                 (the rest is the mean delta of all lookahead nodes); 0 = raw future values
    max_nodes: int = 40000
    time_limit: Optional[float] = None
    trade_proposals: int = 3        # PROPOSE_TRADE candidates per node (scaled down late in the game)
    trade_cap_early: int = 4        # proposals per turn early in the game (DESIGN section 11: 4 early, 2 late) ...
    trade_cap_late: int = 2         # ... interpolated with trade_stage_factor; 0 disables proposals
    opponent_proposals: int = 1     # profile-ranked proposals a simulated opponent may make per turn (0 = never)
    dump_candidates: int = 3        # surplus dumps (discard.surplus_dump_actions) always tried with > 7 cards
    discard_candidates: int = 3
    use_opponent_model: bool = True
    # C++ opponent simulation + reduced lookahead for ``_future_values`` when ``catanbot_core`` is built (see
    # docs/CPP.md "Native lookahead").  The simulated opponents then evaluate every non-trade legal action instead
    # of the prior-ordered top ``opponent_expand`` and never propose trades (``opponent_proposals`` is ignored);
    # the Python code in ``_future_values`` / ``_greedy_turn`` / ``_reduced_search_values`` stays the reference
    # and is used when the extension is missing, ``CATANBOT_NO_ACCEL`` / ``CATANBOT_NO_NATIVE_SEARCH`` is set,
    # the evaluator is not a HeuristicEvaluator / float32 ValueNet / BlendedEvaluator, or a state is unsupported.
    native_future: bool = True
    # Win-path races with crowding (catanbot/winpaths.py, docs/STRATEGY.md "Win-path races").  Off by default: with
    # paths = 0 no winpaths code runs and the search is exactly the old one.  Budgeted for depth 1; at depth >= 2
    # paths = 1 makes _future_values take the Python path (the C++ lookahead cannot see the term).  The five fields
    # never reach C++ (native_level_dict is unchanged).
    paths: int = 0                  # 1 = wrap the evaluator in winpaths.PathsEvaluator (race-aware LR / LA credit)
    paths_w: float = 1.0            # value weight g of the correction (0 = the base evaluator's values)
    paths_crowd: float = 1.0        # crowding strength: scales the waste cost of fighting a crowded race (0 = none)
    paths_priors: int = 1           # with paths = 1: race-aware move-ordering nudges (dev buys, Longest Road roads)
    paths_spots: int = 0            # with paths = 1: rescale the settlement-reach terms by our chance at contested spots
    # Counter-offers and out-of-turn trade analysis (docs/STRATEGY.md "Counter-offers").  Off by default: with
    # counters = 0 COUNTER_TRADE is never a candidate (it only exists under the rules flag
    # GameState.allow_counters anyway) and with respond_lookahead = 0 an answer to an offer is valued exactly as
    # before.  None of these fields reaches C++ (native_level_dict is unchanged); a state with the rules flag
    # on always takes the Python lookahead (accel.python_only).
    counters: int = 0               # 1 = consider counter-offers when answering an offer (needs the rules flag)
    counter_candidates: int = 2     # counters expanded per offer, ranked by P(proposer accepts) x our gain
    counter_aggr: float = 1.0       # ranking score P^(1 / aggr) x gain: > 1 greedier counters, < 1 safer ones
    counter_margin: float = 0.002   # a counter must beat the plain answers by this (win probability): it stands for
    #                                 what the model does not price (the proposer's patience, what the counter reveals)
    respond_lookahead: int = 0      # 1 = value accept / reject / counter after the rest of the proposer's turn


@dataclass
class ScoredAction:
    action: Action
    value: float
    explanation: str = ""
    line: List[Action] = field(default_factory=list)
    static: float = 0.0

    def to_dict(self, state: Optional[GameState] = None) -> dict:
        return {"action": A.to_json(self.action), "text": A.describe(self.action, state), "value": round(self.value, 4),
                "explanation": self.explanation, "line": [A.describe(a, state) for a in self.line]}


class _Node:
    __slots__ = ("state", "prob", "children", "static", "value", "finished", "line", "group")

    def __init__(self, state: GameState, prob: float, line: List[Action], group: int = 0):
        self.state = state
        self.prob = prob
        self.children: List[Tuple[Action, List[Tuple[float, "_Node"]]]] = []
        self.static = 0.0
        self.value: Optional[float] = None
        self.finished = False
        self.line = line
        self.group = group          # siblings (outcomes of the same chance node) share a group


_ROLL_ORDER = sorted(B.ROLL_PROB.items(), key=lambda kv: -kv[1])

# Node budget one leaf of the depth >= 3 lookahead needs for its reduced sub-search (``reduced_config``'s floor).
# ``_future_values`` runs the sub-search for every leaf or for none (``REDUCED_SEARCH_MIN_NODES`` x leaves must be
# left in ``max_nodes`` after the opponents' turns): a partial set, valued one turn deeper than the rest, would
# rank the end-of-turn nodes by their order in the tree.  Mirrored by cpp/search.cpp (future_values).
REDUCED_SEARCH_MIN_NODES = 500


def roll_distribution(samples: int) -> List[Tuple[int, float]]:
    """Most likely ``samples`` roll totals, renormalised."""
    if samples >= 11:
        return [(v, B.ROLL_PROB[v]) for v in range(2, 13)]
    top = _ROLL_ORDER[:max(1, samples)]
    tot = sum(p for _, p in top)
    return [(v, p / tot) for v, p in top]


def reduced_config(cfg: SearchConfig, depth: int, budget: int) -> SearchConfig:
    """The config of the reduced sub-search that scores the leaves of ``_future_values`` at ``depth >= 2``.

    Shared by the Python path (``_reduced_search_values``) and the native one (one entry per lookahead level).
    ``budget`` is the node budget left per leaf; the sub-search gets at least ``REDUCED_SEARCH_MIN_NODES`` nodes.
    """
    return SearchConfig(depth=depth, beam=max(2, cfg.beam // 3), expand=max(4, cfg.expand // 2),
                        max_actions_per_turn=4, roll_samples=min(cfg.roll_samples, 5),
                        opp_roll_samples=max(2, cfg.opp_roll_samples // 3),
                        opponent_actions=3, opponent_expand=4, finished_lookahead=2,
                        lookahead_shrink=cfg.lookahead_shrink,
                        max_nodes=max(REDUCED_SEARCH_MIN_NODES, budget),
                        trade_proposals=1, discard_candidates=2, use_opponent_model=cfg.use_opponent_model,
                        trade_cap_early=cfg.trade_cap_early, trade_cap_late=cfg.trade_cap_late,
                        opponent_proposals=0, dump_candidates=min(2, cfg.dump_candidates),
                        native_future=cfg.native_future, paths=cfg.paths, paths_w=cfg.paths_w,
                        paths_crowd=cfg.paths_crowd, paths_priors=cfg.paths_priors, paths_spots=cfg.paths_spots,
                        counters=cfg.counters, counter_candidates=cfg.counter_candidates,
                        counter_aggr=cfg.counter_aggr, counter_margin=cfg.counter_margin,
                        respond_lookahead=cfg.respond_lookahead)


def lookahead_weight(cfg: SearchConfig) -> float:
    """``n / (n + k)``: the weight of a node's own sampled lookahead delta (``n`` roll samples, ``k =
    lookahead_shrink``); the remaining ``1 - weight`` is the mean delta of all lookahead nodes.  1.0 when ``k <= 0``."""
    k = float(cfg.lookahead_shrink)
    if k <= 0.0:
        return 1.0
    n = float(max(1, cfg.opp_roll_samples))
    return n / (n + k)


def apply_lookahead(statics: Sequence[float], futures: Sequence[float], terminal: Sequence[bool],
                    weight: float) -> Tuple[List[float], float]:
    """Values of the lookahead nodes and the mean shift for every other leaf.

    ``shift`` is the mean of ``future - static`` over the non-terminal lookahead nodes (a finished game has an
    exact value: it keeps it and does not enter the mean).  A non-terminal node gets
    ``static + shift + weight * (future - static - shift)``: with ``weight = 1`` the raw sampled future value
    (exactly), with ``weight < 1`` its noisy deviation from the common mean counts less (``lookahead_weight``).  Mirrored by
    ``apply_lookahead`` in cpp/search.cpp for the native reduced search.
    """
    deltas = [f - s for s, f in zip(statics, futures)]
    live = [d for d, t in zip(deltas, terminal) if not t]
    shift = sum(live) / len(live) if live else 0.0
    values = []
    for s, f, d, t in zip(statics, futures, deltas, terminal):
        values.append(float(f) if (t or weight >= 1.0) else s + shift + weight * (d - shift))
    return values, shift


_NATIVE_LEVEL_FIELDS = ("opponent_actions", "beam", "expand", "max_actions_per_turn", "roll_samples",
                        "opp_roll_samples", "finished_lookahead", "discard_candidates", "max_nodes")
_NATIVE_LEVEL_FLOAT_FIELDS = ("lookahead_shrink",)


def native_level_dict(cfg: SearchConfig) -> dict:
    """The ``SearchConfig`` fields the extension reads for one lookahead level (``core.future_values`` levels)."""
    d = {k: int(getattr(cfg, k)) for k in _NATIVE_LEVEL_FIELDS}
    d.update({k: float(getattr(cfg, k)) for k in _NATIVE_LEVEL_FLOAT_FIELDS})
    return d


class Searcher:
    """Expectimax + beam search over one turn with max^n opponent simulation."""

    def __init__(self, evaluator, config: Optional[SearchConfig] = None,
                 model: Optional[OpponentModel] = None, belief: Optional[HandBelief] = None,
                 politics: Optional[PoliticalState] = None):
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.model = model
        self.belief = belief
        self.politics = politics
        self.nodes = 0
        self._deadline: Optional[float] = None
        self._rng = random.Random(12345)
        self._political_reasons: Dict[Action, str] = {}
        self._chain_next: Dict[Action, Action] = {}   # first step of an intermediary deal -> its second step
        self._arb_cache: Dict[tuple, list] = {}        # arbitrage deals per (hands, trades) within one search
        self._shift = 0.0            # mean(future - static) of the lookahead set, applied to static leaves
        self._paths = None           # winpaths.PathsEvaluator of the current search (config.paths = 1 only)
        self._counter_rank: Dict[Action, int] = {}     # counters kept by _counter_filter (config.counters = 1)
        # Native lookahead (C++): the evaluator's twin handle, or None -> the Python _future_values below.
        self._native_ev = None
        self._native_key = None
        if self.config.native_future and _accel.native_search_available():
            self._native_ev = _accel.native_evaluator(evaluator)
            self._native_key = _accel.evaluator_key(evaluator) if self._native_ev is not None else None

    @property
    def native_active(self) -> bool:
        """True when ``_future_values`` runs in the C++ extension (see ``SearchConfig.native_future``)."""
        return self._native_ev is not None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def best_action(self, state: GameState, rng=None) -> Action:
        return self.search(state, rng=rng)[0].action

    def search(self, state: GameState, player: Optional[int] = None, rng=None) -> List[ScoredAction]:
        cfg = self.config
        me = E.acting_player(state) if player is None else player
        if E.acting_player(state) != me:
            raise ValueError("search() must be called when the player has to act")
        self.nodes = 0
        self._deadline = (time.time() + cfg.time_limit) if cfg.time_limit else None
        self._rng = rng or random.Random(12345)
        self._political_reasons = {}
        self._chain_next = {}
        self._arb_cache = {}
        self._paths = None
        self._counter_rank = {}
        if self._native_ev is not None and _accel.evaluator_key(self.evaluator) != self._native_key:
            # The net's arrays were replaced (set_params / load): rebuild the native twin.
            self._native_ev = _accel.native_evaluator(self.evaluator)
            self._native_key = _accel.evaluator_key(self.evaluator) if self._native_ev is not None else None
        legal = E.legal_actions(state)
        if not legal:
            return []
        if len(legal) == 1 and legal[0][0] != A.ROLL:
            v = float(self._eval([state], [me])[0])
            return [ScoredAction(legal[0], v, self.explain(state, legal[0], me), [legal[0]], v)]
        if cfg.paths and state.phase not in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
            from . import winpaths   # lazy: nothing of it is imported or run with paths = 0
            self._paths = winpaths.PathsEvaluator.for_search(self.evaluator, state, me, cfg)
        root = _Node(state, 1.0, [])
        finished: List[_Node] = []
        frontier = [root]
        self._shift = 0.0
        group_id = 0
        # Answering another player's offer with respond_lookahead: every outcome has the rest of the proposer's
        # turn played out (_outcomes), so it is an end-of-decision node even when we are the next to roll.
        out_of_turn = bool(cfg.respond_lookahead) and state.phase == PHASE_TRADE_RESPONSE and state.current != me
        for level in range(cfg.max_actions_per_turn + 1):
            if not frontier or self._budget_exhausted():
                break
            force_end = level >= cfg.max_actions_per_turn
            new_nodes: List[_Node] = []
            for node in frontier:
                cands = self._candidates(node.state, me, force_end, node.line)
                for a in cands:
                    kids = []
                    try:
                        outcomes = self._outcomes(node.state, a, me)
                    except E.IllegalActionError:
                        continue
                    group_id += 1
                    for p, s2 in outcomes:
                        child = _Node(s2, node.prob * p, node.line + [a], group_id)
                        child.finished = out_of_turn or self._is_finished(s2, me)
                        kids.append((p, child))
                        new_nodes.append(child)
                    if kids:
                        node.children.append((a, kids))
            if not new_nodes:
                break
            vals = self._eval([n.state for n in new_nodes], [me] * len(new_nodes))
            for n, v in zip(new_nodes, vals):
                n.static = float(v)
            unfinished = [n for n in new_nodes if not n.finished]
            finished.extend(n for n in new_nodes if n.finished)
            unfinished.sort(key=lambda n: -n.static)
            # Beam by node, but never split a chance node: every outcome of a kept
            # action stays in the frontier so its expectation is over real continuations.
            keep_groups = {n.group for n in unfinished[:cfg.beam]}
            frontier = [n for n in unfinished if n.group in keep_groups]
        # Future values for the end-of-turn nodes (all of them by default: a candidate valued at the
        # lookahead horizon next to siblings valued statically is preferred or avoided by that alone).
        # The mean shift between lookahead and static values is applied to every other leaf (pruned
        # branches) so that lines with and without lookahead are compared at the same horizon, and a
        # node's own deviation from that mean is weighted by its sample reliability (apply_lookahead).
        if cfg.depth >= 2 and finished and not self._budget_exhausted():
            top = finished
            if cfg.finished_lookahead > 0:
                finished.sort(key=lambda n: -(n.static + 0.05 * math.log(max(n.prob, 1e-6))))
                top = finished[:cfg.finished_lookahead]
            fv = self._future_values([n.state for n in top], me, cfg.depth - 1)
            vals, self._shift = apply_lookahead([n.static for n in top], [float(v) for v in fv],
                                                [n.state.phase == PHASE_GAME_OVER for n in top],
                                                lookahead_weight(cfg))
            for n, v in zip(top, vals):
                n.value = v
        # Backup (unclamped: a shifted static value may leave [0, 1], and clamping it would collapse
        # the ordering of every leaf below 0); the root values are clamped for the caller after ranking.
        self._backup(root, me)
        results: List[ScoredAction] = []
        for a, kids in root.children:
            v = sum(p * (k.value if k.value is not None else k.static) for p, k in kids)
            if a[0] == A.COUNTER_TRADE:
                v -= cfg.counter_margin     # counters (config.counters = 1 only) must beat accept / reject by a margin
            best_kid = max(kids, key=lambda pk: pk[1].value if pk[1].value is not None else pk[1].static)[1]
            line = self._principal_line(best_kid)
            results.append(ScoredAction(a, v, "", [a] + line, sum(p * k.static for p, k in kids)))
        results.sort(key=lambda r: -r.value)
        for r in results:
            r.value = min(1.0, max(0.0, r.value))
            r.explanation = self.explain(state, r.action, me)
        return results

    # ------------------------------------------------------------------
    # tree helpers
    # ------------------------------------------------------------------
    def _budget_exhausted(self) -> bool:
        if self.nodes >= self.config.max_nodes:
            return True
        if self._deadline is not None and time.time() > self._deadline:
            return True
        return False

    def _eval(self, states: Sequence[GameState], players: Sequence[int]):
        ev = self.evaluator if self._paths is None else self._paths
        return ev.evaluate(list(states), list(players))

    def _apply(self, state: GameState, action: Action) -> GameState:
        self.nodes += 1
        return E.apply(state, action, self._rng)

    @staticmethod
    def _is_finished(s: GameState, me: int) -> bool:
        if s.phase == PHASE_GAME_OVER:
            return True
        return s.current != me

    def _backup(self, node: _Node, me: int) -> float:
        if not node.children:
            if node.value is None:
                node.value = node.static + self._shift
            return node.value
        best = -math.inf
        for a, kids in node.children:
            v = 0.0
            for p, k in kids:
                v += p * self._backup(k, me)
            if v > best:
                best = v
        # A pruned/unexpanded alternative might still be better than the expanded ones? No:
        # the static value of this node is what we would get without acting; END_TURN is
        # always among the candidates, so ``best`` already covers "do nothing".
        node.value = best
        return best

    def _principal_line(self, node: _Node) -> List[Action]:
        line: List[Action] = []
        while node.children:
            a, kids = max(node.children,
                          key=lambda ak: sum(p * (k.value if k.value is not None else k.static) for p, k in ak[1]))
            line.append(a)
            node = max(kids, key=lambda pk: pk[0])[1]
        return line

    # ------------------------------------------------------------------
    # candidate generation
    # ------------------------------------------------------------------
    def trade_cap(self, state: GameState) -> int:
        """Proposals allowed per turn: ``trade_cap_early`` early, ``trade_cap_late`` late (DESIGN section 11)."""
        cfg = self.config
        stage_f = trade_stage_factor(state)                      # 1.0 early .. 0.3 late
        t = max(0.0, min(1.0, (1.0 - stage_f) / 0.7))
        cap = cfg.trade_cap_early + (cfg.trade_cap_late - cfg.trade_cap_early) * t
        return max(0, min(E.MAX_TRADE_PROPOSALS_PER_TURN, int(round(cap))))

    def _arbitrage(self, state: GameState, me: int) -> list:
        """Exploitable deals (cached per search and proposals made: profiles do not change mid-search
        and the deals only order candidates, which are filtered for legality at every node)."""
        key = (me, state.trades_this_turn)
        arbs = self._arb_cache.get(key)
        if arbs is None:
            try:
                arbs = self.model.arbitrage_opportunities(state, me, belief=self.belief, politics=self.politics)
            except Exception:
                arbs = []
            self._arb_cache[key] = arbs
        return arbs

    def _candidate_priors(self, state: GameState, legal: Sequence[Action], me: int) -> List[float]:
        """Move-ordering priors for our own decision node.

        On top of ``heuristic.action_priors`` (which already sees the opponent
        model and the politics): PROPOSE_TRADE priors are multiplied by
        ``trade_stage_factor``, the first steps of the best exploitable deals
        (``OpponentModel.arbitrage_opportunities``) are promoted above
        ordinary proposals and their explanation records the counterpart's
        revealed valuation, and with more than 7 cards the cheapest surplus
        dumps (``discard.surplus_dump_actions``) rank at least like the trade
        plan so they are never crowded out of the beam by road / offer spam.
        """
        cfg = self.config
        model = self.model if cfg.use_opponent_model else None
        priors = list(action_priors(state, legal, me, self.belief, model=model, politics=self.politics))
        idx = {a: i for i, a in enumerate(legal)}
        stage_f = trade_stage_factor(state)
        if state.phase == PHASE_MAIN:
            has_proposals = any(a[0] == A.PROPOSE_TRADE for a in legal)
            if has_proposals and stage_f < 1.0:
                for i, a in enumerate(legal):
                    if a[0] == A.PROPOSE_TRADE:
                        priors[i] *= stage_f
            if has_proposals and model is not None:
                for d in self._arbitrage(state, me)[:3]:
                    first = d["steps"][0]
                    i = idx.get(first)
                    if i is None:
                        continue
                    priors[i] = max(priors[i], (58.0 + 10.0 * d["gain"]) * stage_f)   # plan proposals sit at 55
                    self._political_reasons.setdefault(
                        first, "exploit: " + d["reason"]
                        + (f"; then {A.describe(d['steps'][1], state)}" if len(d["steps"]) > 1 else ""))
                    if len(d["steps"]) > 1:
                        self._chain_next.setdefault(first, d["steps"][1])
            if cfg.dump_candidates > 0 and state.players[me].total_resources > 7:
                for k, a in enumerate(surplus_dump_actions(state, me, list(legal))[:cfg.dump_candidates]):
                    i = idx.get(a)
                    if i is not None:
                        priors[i] = max(priors[i], 62.0 - k)                     # plan bank trades sit at 60
        if self._paths is not None and cfg.paths_priors:
            priors = self._paths.ctx.adjust_priors(state, legal, priors)
        if state.phase == PHASE_TRADE_RESPONSE and self._counter_rank:
            for i, a in enumerate(legal):          # our ranked counters (_counter_filter) are always expanded
                k = self._counter_rank.get(a)
                if k is not None:
                    priors[i] = max(priors[i], 75.0 - k)
        return priors

    def _counter_filter(self, state: GameState, legal: List[Action], me: int) -> List[Action]:
        """Counter-offer rules: which COUNTER_TRADE candidates this node keeps.

        ``counters = 0`` (default) drops all of them.  Otherwise the engine's bounded edits are ranked by
        ``counteroffers.rank_counters`` (P(the proposer accepts) x our gain, affordable, never feeding the
        leader, no counters in the late-game "no trades with a player ahead" situation) and the best
        ``counter_candidates`` are kept; their explanation records the deal and the acceptance probability.
        """
        counters = [a for a in legal if a[0] == A.COUNTER_TRADE]
        if not counters:
            return legal
        rest = [a for a in legal if a[0] != A.COUNTER_TRADE]
        cfg = self.config
        if not cfg.counters or cfg.counter_candidates <= 0:
            return rest
        from .counteroffers import rank_counters     # lazy: never imported while counters = 0
        ev = self.evaluator if self._paths is None else self._paths
        model = self.model if cfg.use_opponent_model else None
        try:
            ranked = rank_counters(state, me, counters, evaluator=ev, model=model, belief=self.belief,
                                   politics=self.politics, aggr=cfg.counter_aggr)
        except Exception:
            ranked = []
        keep = [d for d in ranked if d["score"] > 0.0][:cfg.counter_candidates]
        for k, d in enumerate(keep):
            self._counter_rank[d["action"]] = k
            self._political_reasons.setdefault(d["action"], d["reason"])
        return rest + [d["action"] for d in keep]

    def _candidates(self, state: GameState, me: int, force_end: bool,
                    line: Optional[Sequence[Action]] = None) -> List[Action]:
        cfg = self.config
        legal = E.legal_actions(state)
        if not legal:
            return []
        if state.allow_counters and state.phase == PHASE_TRADE_RESPONSE:
            legal = self._counter_filter(state, legal, me)
        if force_end and (A.END_TURN,) in legal:
            return [(A.END_TURN,)]
        if len(legal) == 1:
            return legal
        if state.phase == PHASE_TRADE_SELECT and state.pending_trade is not None:
            # Never complete a trade with a player about to win / whom it would hand a build.
            from .robber import estimated_vp
            from .trading import offer_is_feeding_leader
            safe = [a for a in legal if a[0] != A.EXECUTE_TRADE
                    or not (offer_is_feeding_leader(state, me, a[1], state.pending_trade.give)[0]
                            or estimated_vp(state, a[1]) >= B.VP_TO_WIN - 1)]
            if safe:
                legal = safe
        priors = self._candidate_priors(state, legal, me)
        stage_f = trade_stage_factor(state)
        order = sorted(range(len(legal)), key=lambda i: -priors[i])
        out: List[Action] = []
        n_trades = 0
        n_discards = 0
        # Per node: fewer proposal candidates late in the game (but at least one while proposals
        # are enabled); per turn: the documented cap (4 early, 2 late).
        max_trades = max(1 if cfg.trade_proposals > 0 else 0, int(round(cfg.trade_proposals * stage_f)))
        if state.trades_this_turn >= self.trade_cap(state):
            max_trades = 0
        for i in order:
            a = legal[i]
            k = a[0]
            if k == A.PROPOSE_TRADE:
                if n_trades >= max_trades:
                    continue
                n_trades += 1
            elif k == A.DISCARD:
                if n_discards >= cfg.discard_candidates:
                    continue
                n_discards += 1
            out.append(a)
            if len(out) >= cfg.expand:
                break
        if (A.END_TURN,) in legal and (A.END_TURN,) not in out:
            out.append((A.END_TURN,))
        # Second step of an intermediary deal: once the first trade executed (we now hold the
        # bought card, so the follow-up offer is legal) it is tried first at this level.
        if line and self._chain_next and max_trades > 0:
            last = next((a for a in reversed(line) if a[0] == A.PROPOSE_TRADE), None)
            nxt = self._chain_next.get(last) if last is not None else None
            if nxt is not None and nxt in legal and nxt not in out:
                out.insert(0, nxt)
                self._political_reasons.setdefault(nxt, "exploit: second leg of the intermediary deal")
        # Political options: trades that let a trailing player take an award off the
        # leader at no cost to our own win probability ("buy runway").
        if (state.phase == PHASE_MAIN and state.free_roads == 0
                and any(a[0] == A.PROPOSE_TRADE for a in legal)):
            try:
                ev = self.evaluator if self._paths is None else self._paths
                for opt in political_trade_options(state, me, ev, self.politics, model=self.model)[:2]:
                    a = opt["action"]
                    if a not in out:
                        out.append(a)
                    self._political_reasons[a] = opt["reason"]
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------
    # chance outcomes
    # ------------------------------------------------------------------
    def _outcomes(self, state: GameState, action: Action, me: int) -> List[Tuple[float, GameState]]:
        kind = action[0]
        outs: List[Tuple[float, GameState]] = []
        if kind == A.ROLL and len(action) == 1:
            for v, p in roll_distribution(self.config.roll_samples):
                outs.append((p, self._apply(state, (A.ROLL, v))))
        elif kind == A.BUY_DEV:
            outs = self._dev_outcomes(state, action, me)
        elif kind in (A.MOVE_ROBBER, A.PLAY_KNIGHT) and len(action) >= 3 and action[2] is not None and action[2] >= 0:
            outs = self._steal_outcomes(state, action, me)
        elif kind == A.PROPOSE_TRADE:
            outs = self._trade_outcomes(state, action, me)
        elif kind in (A.ACCEPT_TRADE, A.REJECT_TRADE) and state.phase == PHASE_TRADE_RESPONSE:
            outs = self._response_outcomes(state, action, me)
        elif kind == A.COUNTER_TRADE:
            outs = self._counter_outcomes(state, action, me)
        else:
            outs = [(1.0, self._apply(state, action))]
        outs = [(p, self._autoplay_others(s, me)) for p, s in outs]
        if self.config.respond_lookahead and state.phase == PHASE_TRADE_RESPONSE and state.current != me:
            # (2) out-of-turn analysis: value every answer after the rest of the proposer's turn.
            outs = [(p, self._finish_proposer_turn(s, me)) for p, s in outs]
        return outs

    def _finish_proposer_turn(self, s: GameState, me: int) -> GameState:
        """The rest of the current (proposing) player's turn, played greedily like the lookahead's opponents
        (``_greedy_turn``; they have rolled already, so only dev draws / steals are random) up to and including
        their END_TURN.  ``s`` is returned unchanged when the game is over or it is our turn."""
        if s.phase == PHASE_GAME_OVER or s.current == me:
            return s
        return self._greedy_turn(s, s.current, 0, me)

    def _response_outcomes(self, state: GameState, action: Action, me: int) -> List[Tuple[float, GameState]]:
        """Our ACCEPT / REJECT: let the other responders answer and the proposer choose a
        partner (uniformly among accepters), so the leaf contains the executed trade."""
        return self._resolve_offer(self._apply(state, action), me)

    def _counter_accept_probability(self, s: GameState, ctr, me: int) -> float:
        """P(the current player accepts our counter ``ctr``, shown to them now); lower when somebody accepted
        the original offer as proposed (they could trade with them instead)."""
        cur = s.current
        orig = ctr.origin
        alternatives = orig is not None and any(orig.responses.values())
        if self.model is not None and self.config.use_opponent_model:
            return self.model.predict_counter_accept(s, cur, orig, ctr.give, ctr.get, counterer=me,
                                                     belief=self.belief, politics=self.politics,
                                                     alternatives=alternatives)
        ok, _ = should_accept(s, cur, ctr, politics=self.politics)
        return (0.75 if ok else 0.1) * (0.5 if alternatives else 1.0)

    def _counter_outcomes(self, state: GameState, action: Action, me: int) -> List[Tuple[float, GameState]]:
        """Our COUNTER_TRADE: the other responders answer the original offer (accept / reject, as in
        ``_response_outcomes``), then the proposer sees our counter - a chance node with
        :meth:`_counter_accept_probability`: taken (the trade executes, the round closes) or rejected (the
        original offer continues: other counters, then the proposer's partner choice)."""
        outs: List[Tuple[float, GameState]] = []
        for q, s1 in self._resolve_offer(self._apply(state, action), me):
            ctr = s1.pending_trade
            if not (s1.phase == PHASE_TRADE_RESPONSE and ctr is not None and ctr.origin is not None
                    and ctr.proposer == me):
                outs.append((q, s1))             # dropped (they cannot pay) or the round closed earlier
                continue
            p = min(1.0, max(0.0, self._counter_accept_probability(s1, ctr, me)))
            if p > 1e-4:
                outs.append((q * p, self._apply(s1, (A.ACCEPT_TRADE,))))
            if 1.0 - p > 1e-4:
                for q2, s2 in self._resolve_offer(self._apply(s1, (A.REJECT_TRADE,)), me):
                    outs.append((q * (1.0 - p) * q2, s2))
        return outs

    def _resolve_offer(self, s: GameState, me: int) -> List[Tuple[float, GameState]]:
        """After our answer: the remaining responders answer (``should_accept``), then the proposer picks a
        partner uniformly among the accepters.  Stops early (one outcome) at a decision of ours (counter-offer
        rules: a counter shown to us as the current player, or our own partner choice) and when our own counter
        is shown to the proposer (``_counter_outcomes`` makes that a chance node)."""
        guard = 0
        while (s.phase == PHASE_TRADE_RESPONSE and s.pending_trade is not None and guard < 8
               and E.acting_player(s) != me):
            po = s.pending_trade
            if po.origin is not None and po.proposer == me:
                break
            j = E.acting_player(s)
            ok, _ = should_accept(s, j, po, politics=self.politics)
            s = self._apply(s, (A.ACCEPT_TRADE,) if ok and (A.ACCEPT_TRADE,) in E.legal_actions(s) else (A.REJECT_TRADE,))
            guard += 1
        if s.phase != PHASE_TRADE_SELECT or s.pending_trade is None or s.current == me:
            return [(1.0, s)]
        accepters = [j for j, ok in s.pending_trade.responses.items() if ok]
        legal = E.legal_actions(s)
        outs = []
        if accepters:
            p = 1.0 / len(accepters)
            for j in accepters:
                a = (A.EXECUTE_TRADE, j)
                if a in legal:
                    outs.append((p, self._apply(s, a)))
        if not outs:
            outs.append((1.0, self._apply(s, (A.CANCEL_TRADE,)) if (A.CANCEL_TRADE,) in legal else s))
        return outs

    def _dev_outcomes(self, state: GameState, action: Action, me: int) -> List[Tuple[float, GameState]]:
        deck = state.dev_deck
        total = sum(deck)
        base = self._apply(state, action)
        if total <= 0:
            return [(1.0, base)]
        drawn = -1
        for t in range(5):
            if base.dev_deck[t] != deck[t]:
                drawn = t
                break
        if drawn < 0:
            return [(1.0, base)]
        outs = []
        for t in range(5):
            if deck[t] <= 0:
                continue
            p = deck[t] / total
            if t == drawn:
                outs.append((p, base))
                continue
            s2 = base.copy()
            pl = s2.players[me]
            pl.dev_cards_new[drawn] -= 1
            pl.dev_cards_new[t] += 1
            s2.dev_deck[drawn] += 1
            s2.dev_deck[t] -= 1
            if t == B.DEV_VP and s2.total_vp(me) >= B.VP_TO_WIN and s2.phase != PHASE_GAME_OVER:
                s2.phase = PHASE_GAME_OVER
                s2.winner = me
            elif drawn == B.DEV_VP and s2.phase == PHASE_GAME_OVER and s2.winner == me and s2.total_vp(me) < B.VP_TO_WIN:
                s2.phase = PHASE_MAIN
                s2.winner = -1
            outs.append((p, s2))
        return outs

    def _steal_outcomes(self, state: GameState, action: Action, me: int) -> List[Tuple[float, GameState]]:
        victim = action[2]
        before = list(state.players[victim].resources)
        total = sum(before)
        base = self._apply(state, action)
        if total <= 0:
            return [(1.0, base)]
        after = base.players[victim].resources
        stolen = -1
        for r in range(5):
            if after[r] != before[r]:
                stolen = r
                break
        if stolen < 0:
            return [(1.0, base)]
        outs = []
        for r in range(5):
            if before[r] <= 0:
                continue
            p = before[r] / total
            if r == stolen:
                outs.append((p, base))
                continue
            s2 = base.copy()
            v = s2.players[victim]
            t = s2.players[me]
            v.resources[stolen] += 1
            v.resources[r] -= 1
            t.resources[stolen] -= 1
            t.resources[r] += 1
            outs.append((p, s2))
        return outs

    def _accept_probability(self, state: GameState, j: int, offer, me: int) -> float:
        if self.model is not None and self.config.use_opponent_model:
            return self.model.predict_accept(state, j, offer.give, offer.get, proposer=me, belief=self.belief,
                                             politics=self.politics)
        p = state.players[j]
        will = self.politics.trade_willingness(state, j, me) if self.politics is not None else 1.0
        if p.hand_known:
            ok, _ = should_accept(state, j, offer, politics=self.politics)
            return min(1.0, (0.8 if ok else 0.08) * will)
        # unknown hand: can they pay?  crude production-based guess
        from .trading import partner_likelihood
        prob = 1.0
        for r in range(5):
            if offer.get[r]:
                prob *= partner_likelihood(state, me, j, r, self.belief)
        return 0.5 * prob

    def _trade_outcomes(self, state: GameState, action: Action, me: int) -> List[Tuple[float, GameState]]:
        s1 = self._apply(state, action)
        if s1.phase != PHASE_TRADE_RESPONSE or s1.pending_trade is None:
            return [(1.0, s1)]
        offer = s1.pending_trade
        probs: Dict[int, float] = {}
        for j in range(state.num_players):
            if j == me:
                continue
            pj = state.players[j]
            if pj.hand_known and any(pj.resources[r] < offer.get[r] for r in range(5)):
                continue
            probs[j] = self._accept_probability(state, j, offer, me)
        if not probs:
            rejected = self._respond_all(s1, me, accept_from=-1)
            return [(1.0, rejected)]
        p_any = 1.0 - math.prod(1.0 - p for p in probs.values())
        partner = max(probs, key=probs.get)
        accepted = self._respond_all(s1, me, accept_from=partner)
        rejected = self._respond_all(s1, me, accept_from=-1)
        outs = []
        if p_any > 1e-4:
            outs.append((p_any, accepted))
        if 1.0 - p_any > 1e-4:
            outs.append((1.0 - p_any, rejected))
        return outs or [(1.0, rejected)]

    def _respond_all(self, s: GameState, me: int, accept_from: int) -> GameState:
        guard = 0
        while s.phase == PHASE_TRADE_RESPONSE and guard < 8:
            j = E.acting_player(s)
            s = self._apply(s, (A.ACCEPT_TRADE,) if j == accept_from else (A.REJECT_TRADE,))
            guard += 1
        if s.phase == PHASE_TRADE_SELECT and E.acting_player(s) == me:
            if accept_from >= 0 and (A.EXECUTE_TRADE, accept_from) in E.legal_actions(s):
                s = self._apply(s, (A.EXECUTE_TRADE, accept_from))
            else:
                s = self._apply(s, (A.CANCEL_TRADE,))
        return s

    def _autoplay_others(self, s: GameState, me: int) -> GameState:
        """Resolve forced decisions of other players inside our turn (discards, responses)."""
        guard = 0
        while s.phase != PHASE_GAME_OVER and s.current == me and E.acting_player(s) != me and guard < 12:
            j = E.acting_player(s)
            legal = E.legal_actions(s)
            if not legal:
                break
            if s.phase == PHASE_DISCARD:
                a = choose_discard(s, j, legal_actions=legal)
            elif s.phase == PHASE_TRADE_RESPONSE and s.pending_trade is not None:
                ok, _ = should_accept(s, j, s.pending_trade, politics=self.politics, model=self.model)
                a = (A.ACCEPT_TRADE,) if ok and (A.ACCEPT_TRADE,) in legal else (A.REJECT_TRADE,)
            else:
                a = legal[0]
            s = self._apply(s, a)
            guard += 1
        return s

    def _opponent_proposal(self, s: GameState, j: int, legal: Sequence[Action]) -> Optional[Action]:
        """The one player trade a simulated opponent ``j`` would propose now (profile-ranked), or None."""
        if not any(a[0] == A.PROPOSE_TRADE for a in legal):
            return None
        legal_set = set(legal)
        model = self.model if self.config.use_opponent_model else None
        try:
            # Like trade_advice: plan for the build they are closest to, then the next one.
            for cost, _ in default_keep_targets(s, j)[:2]:
                for st in plan_trades(s, j, cost, self.belief, model=model, politics=self.politics):
                    if st["kind"] == "player" and st["action"] in legal_set:
                        return st["action"]
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # opponents' turns
    # ------------------------------------------------------------------
    def _future_values(self, states: Sequence[GameState], me: int, depth: int) -> List[float]:
        """Value of each end-of-turn state after the opponents have played (sampled rolls)."""
        cfg = self.config
        n_samples = max(1, cfg.opp_roll_samples)
        # Common random numbers: the same roll sequences for every state.
        rng = random.Random(self._rng.random())
        seqs = [[rng.randint(1, 6) + rng.randint(1, 6) for _ in range(3 * 4)] for _ in range(n_samples)]
        if self._native_ev is not None and self._paths is None and states:   # C++ cannot see the win-path term
            out = self._native_future_values(states, me, depth, seqs)
            if out is not None:
                return out
        leaves: List[GameState] = []
        owner: List[int] = []
        for si, s0 in enumerate(states):
            for seq in seqs:
                s = self._simulate_until_my_turn(s0, me, seq)
                leaves.append(s)
                owner.append(si)
        if depth >= 2 and not self._budget_exhausted() and self._leaves_affordable(len(leaves)):
            vals = self._reduced_search_values(leaves, me, depth)
        else:
            vals = list(self._eval(leaves, [me] * len(leaves)))
        out = [0.0] * len(states)
        cnt = [0] * len(states)
        for o, v in zip(owner, vals):
            out[o] += float(v)
            cnt[o] += 1
        return [out[i] / max(1, cnt[i]) for i in range(len(states))]

    def _leaves_affordable(self, n_leaves: int) -> bool:
        """depth >= 3: the reduced sub-search runs for every lookahead leaf or for none of them.

        ``REDUCED_SEARCH_MIN_NODES`` nodes per leaf must be left in ``max_nodes`` after the opponents' turns were
        simulated (``self.nodes`` counts them).  Searching only the leaves the budget still covers would value the
        first end-of-turn nodes of the tree one turn deeper than their siblings, i.e. rank them by tree order.
        """
        return self.config.max_nodes - self.nodes >= REDUCED_SEARCH_MIN_NODES * max(1, n_leaves)

    def _native_rng(self):
        """The ``rng`` argument of the native call: a seed drawn from our stream (tests may return a Random)."""
        return self._rng.getrandbits(63)

    def _native_future_values(self, states: Sequence[GameState], me: int, depth: int,
                              seqs: Sequence[Sequence[int]]) -> Optional[List[float]]:
        """``_future_values`` in the extension (opponents' greedy turns, reduced lookahead, leaf evaluation).

        Returns ``None`` when a state cannot be represented natively; the caller then runs the Python body.
        """
        cfg = self.config
        if any(s.allow_counters for s in states):
            # Counter-offer rules: the C++ engine does not implement them (accel.python_only), Python lookahead.
            return None
        if any(s.phase == PHASE_TRADE_RESPONSE and s.pending_trade is not None for s in states):
            # Other responders still have to answer a pending offer: the Python simulation asks should_accept
            # for them, the extension only rejects.  Searcher.search never produces such end-of-turn states
            # (_response_outcomes / _respond_all resolve the offer first), direct callers may.
            return None
        levels = [cfg]
        d = depth
        while d >= 2:
            budget = (cfg.max_nodes - self.nodes) // max(1, len(states) * max(1, len(seqs)))
            levels.append(reduced_config(levels[-1], d, budget))
            d -= 1
        level_dicts = [native_level_dict(lv) for lv in levels]
        model = self.model if cfg.use_opponent_model else None
        robber = _accel.robber_weights_bundle(states[0], self.politics, model)
        res = _accel.future_values(states, me, depth, level_dicts, [list(q) for q in seqs], self._native_ev, robber,
                                   rng=self._native_rng(), deadline=self._deadline,
                                   node_budget=max(0, cfg.max_nodes - self.nodes))
        if res is None:
            return None
        vals, nodes = res
        self.nodes += nodes
        return vals

    def _reduced_search_values(self, leaves: Sequence[GameState], me: int, depth: int) -> List[float]:
        sub_cfg = reduced_config(self.config, depth, (self.config.max_nodes - self.nodes) // max(1, len(leaves)))
        vals = []
        for s in leaves:
            if s.phase == PHASE_GAME_OVER or E.acting_player(s) != me:
                vals.append(float(self._eval([s], [me])[0]))
                continue
            sub = Searcher(self.evaluator, sub_cfg, self.model, self.belief, self.politics)
            sub._rng = random.Random(self._rng.random())
            res = sub.search(s, me)
            self.nodes += sub.nodes
            vals.append(res[0].value if res else float(self._eval([s], [me])[0]))
        return vals

    def _simulate_until_my_turn(self, s: GameState, me: int, rolls: Sequence[int]) -> GameState:
        """Greedy max^n play by the opponents until it is our roll phase (or game over)."""
        ri = 0
        guard = 0
        while s.phase != PHASE_GAME_OVER and not (s.current == me and s.phase == PHASE_ROLL) and guard < 6:
            if s.current == me:
                # Left-over forced decisions in our own turn (should be rare): finish the turn greedily.
                s = self._greedy_turn(s, me, rolls[ri % len(rolls)] if s.phase == PHASE_ROLL else 0, me)
            else:
                roll = rolls[ri % len(rolls)] if s.phase == PHASE_ROLL else 0
                s = self._greedy_turn(s, s.current, roll, me)
            ri += 1
            guard += 1
        return s

    def _greedy_turn(self, s: GameState, j: int, roll: int, me: int) -> GameState:
        cfg = self.config
        steps = 0
        acted = 0
        while s.phase != PHASE_GAME_OVER and s.current == j and steps < 3 * cfg.opponent_actions + 6:
            steps += 1
            acting = E.acting_player(s)
            legal = E.legal_actions(s)
            if not legal:
                break
            if acting != j:
                # Forced decision of another player (maybe us): heuristics.
                if s.phase == PHASE_DISCARD:
                    a = choose_discard(s, acting, legal_actions=legal)
                elif s.phase == PHASE_TRADE_RESPONSE and s.pending_trade is not None:
                    ok, _ = should_accept(s, acting, s.pending_trade, politics=self.politics, model=self.model)
                    a = (A.ACCEPT_TRADE,) if ok and (A.ACCEPT_TRADE,) in legal else (A.REJECT_TRADE,)
                else:
                    a = legal[0]
                s = self._apply(s, a)
                continue
            if s.phase == PHASE_ROLL:
                if roll and (A.ROLL, roll) not in legal and (A.ROLL,) in legal:
                    s = self._apply(s, (A.ROLL, roll))
                elif roll:
                    s = self._apply(s, (A.ROLL, roll))
                else:
                    s = self._apply(s, (A.ROLL,))
                continue
            if s.phase == PHASE_DISCARD:
                s = self._apply(s, choose_discard(s, j, legal_actions=legal))
                continue
            if s.phase == PHASE_ROBBER:
                # Opponents rob by threat x grudge x their observed habits (whom they keep hitting,
                # whether they go for the leader).
                model = self.model if cfg.use_opponent_model else None
                if self.politics is not None:
                    tw = self.politics.robber_target_weights(s, j, model=model)
                elif model is not None:
                    tw = model.robber_habit_weights(s, j)
                else:
                    tw = None
                h, v, _ = best_robber_move(s, j, target_weights=tw)
                a = (A.MOVE_ROBBER, h, v)
                if a not in legal:
                    a = legal[0]
                s = self._apply(s, a)
                continue
            if s.phase == PHASE_TRADE_SELECT:
                a = next((x for x in legal if x[0] == A.EXECUTE_TRADE), legal[0])
                s = self._apply(s, a)
                continue
            # main phase: greedy one-step lookahead with the value function from j's perspective
            if acted >= cfg.opponent_actions or len(legal) == 1:
                s = self._apply(s, (A.END_TURN,)) if (A.END_TURN,) in legal else self._apply(s, legal[0])
                continue
            cands = [a for a in legal if a[0] != A.PROPOSE_TRADE]
            # At most ``opponent_proposals`` profile-ranked player trades per simulated turn, so the
            # lookahead (and the value net trained on it) sees us being offered deals.
            if cfg.opponent_proposals > 0 and s.trades_this_turn < cfg.opponent_proposals:
                prop = self._opponent_proposal(s, j, legal)
                if prop is not None:
                    cands.append(prop)
            model = self.model if cfg.use_opponent_model else None
            priors = action_priors(s, cands, j, self.belief, model=model, politics=self.politics)
            order = sorted(range(len(cands)), key=lambda i: -priors[i])[:cfg.opponent_expand]
            trial = []
            for i in order:
                a = cands[i]
                s2 = self._apply(s, a)
                s2 = self._autoplay_others(s2, j)
                if a[0] == A.PROPOSE_TRADE and s2.phase == PHASE_TRADE_SELECT and s2.current == j:
                    # Resolve the proposal so its value reflects the executed (or cancelled) trade.
                    lg = E.legal_actions(s2)
                    ex = next((x for x in lg if x[0] == A.EXECUTE_TRADE), None)
                    s2 = self._apply(s2, ex if ex is not None else (A.CANCEL_TRADE,))
                trial.append((a, s2))
            if not trial:
                break
            vals = self._eval([t[1] for t in trial], [j] * len(trial))
            k = max(range(len(trial)), key=lambda i: float(vals[i]))
            a, s2 = trial[k]
            s = s2
            acted += 1
            if a[0] == A.END_TURN:
                break
        if s.phase != PHASE_GAME_OVER and s.current == j:
            legal = E.legal_actions(s)
            if (A.END_TURN,) in legal:
                s = self._apply(s, (A.END_TURN,))
        return s

    # ------------------------------------------------------------------
    # explanations
    # ------------------------------------------------------------------
    def explain(self, state: GameState, action: Action, me: int) -> str:
        if action in self._political_reasons:
            return self._political_reasons[action]
        return explain_action(state, action, me, self.model, self.belief, self.politics)


def explain_action(state: GameState, action: Action, me: int, model: Optional[OpponentModel] = None,
                   belief: Optional[HandBelief] = None, politics: Optional[PoliticalState] = None) -> str:
    """Short human readable rationale for an action (uses the strategy modules).

    ``politics`` makes the robber victim and the printed acceptance
    probabilities agree with the search (grudges, favour slack).
    """
    k = action[0]
    p = state.players[me]
    tw = politics.robber_target_weights(state, me) if politics is not None else None
    try:
        if k in (A.BUILD_SETTLEMENT, A.SETUP_SETTLEMENT):
            v = action[1]
            prod = vertex_production(state, v, ignore_robber=True)
            pips = sum(B.PIPS[state.hexes[h][1]] for h in B.VERTEX_HEXES[v]
                       if state.hexes[h][0] != B.DESERT and state.hexes[h][1])
            res = ", ".join(f"{B.RESOURCE_NAMES[r]} {prod[r] * 36:.0f}" for r in range(5) if prod[r] > 0)
            port = state.ports.get(v)
            txt = f"{pips} pips ({res})"
            if port is not None:
                txt += f", {B.PORT_NAMES[port]} port"
            return txt + ("" if k == A.SETUP_SETTLEMENT else "; +1 VP")
        if k == A.BUILD_CITY:
            v = action[1]
            prod = vertex_production(state, v, ignore_robber=True)
            res = ", ".join(f"{B.RESOURCE_NAMES[r]} +{prod[r] * 36:.0f}" for r in range(5) if prod[r] > 0)
            return f"+1 VP and doubles production: {res} pips"
        if k in (A.BUILD_ROAD, A.SETUP_ROAD):
            targets = road_targets(state, me, max_roads=3, k=4)
            for t in targets:
                if t["first_edge"] == action[1]:
                    return f"towards vertex {t['vertex']} (spot score {t['spot_score']:.1f}, {t['roads']} road(s) away)"
            if state.free_roads > 0:
                return "free road (Road Building)"
            return "extends the road network" + (" (longest road)" if state.longest_road_owner != me else "")
        if k == A.BUY_DEV:
            _, why = should_buy_dev(state, me, belief)
            return why + " (playable next turn; a VP card counts immediately)"
        if k == A.PLAY_KNIGHT:
            ok, why = should_play_knight(state, me)
            h, victim, reason = best_robber_move(state, me, target_weights=tw)
            if (action[1], action[2]) == (h, victim):
                return f"{why}; {reason}"
            opp, own = hex_damage(state, action[1], me, tw)
            return f"{why}; blocks {opp:.1f} pips-equivalent of opponents' production"
        if k == A.MOVE_ROBBER:
            h, victim, reason = best_robber_move(state, me, target_weights=tw)
            if (action[1], action[2]) == (h, victim):
                return reason
            opp, own = hex_damage(state, action[1], me, tw)
            return f"blocks {opp:.1f} pips-equivalent" + (f", costs us {own:.1f}" if own else "")
        if k == A.BANK_TRADE:
            ratio = state.port_ratio(me, action[1])
            after = list(p.resources)
            after[action[1]] -= ratio
            after[action[2]] += 1
            for cost, name in ((B.COST_CITY, "city"), (B.COST_SETTLEMENT, "settlement"), (B.COST_DEV, "dev card"), (B.COST_ROAD, "road")):
                if all(after[r] >= cost[r] for r in range(5)) and not p.can_afford(cost):
                    return f"{ratio}:1 {'port' if ratio < 4 else 'bank'} trade enables a {name}"
            return f"{ratio}:1 {'port' if ratio < 4 else 'bank'} trade"
        if k == A.PROPOSE_TRADE:
            give, get = action[1], action[2]
            txt = ""
            if model is not None:
                best = max(((model.predict_accept(state, j, give, get, proposer=me, belief=belief, politics=politics), j)
                            for j in range(state.num_players) if j != me), default=(0.0, -1))
                if best[1] >= 0:
                    nm = state.players[best[1]].name or state.players[best[1]].color
                    txt = f"{nm} accepts with ~{best[0]:.0%}; "
            after = list(p.resources)
            for r in range(5):
                after[r] += get[r] - give[r]
            for cost, name in ((B.COST_CITY, "city"), (B.COST_SETTLEMENT, "settlement"), (B.COST_DEV, "dev card"), (B.COST_ROAD, "road")):
                if all(after[r] >= cost[r] for r in range(5)) and not p.can_afford(cost):
                    txt += f"enables a {name}; "
            txt += f"trade stage factor {trade_stage_factor(state):.2f}"
            return txt
        if k == A.DISCARD:
            return "keeps the cards needed for the next build"
        if k == A.END_TURN:
            if p.total_resources > 7:
                return explain_seven_risk(state, me)
            return "nothing better to do this turn"
        if k == A.ACCEPT_TRADE or k == A.REJECT_TRADE:
            if state.pending_trade is not None:
                ok, why = should_accept(state, me, state.pending_trade, model=model, politics=politics)
                return why
        if k == A.PLAY_ROAD_BUILDING:
            return "two free roads (saves 2 wood + 2 brick)"
        if k == A.PLAY_YEAR_OF_PLENTY:
            from .devcards import best_year_of_plenty
            r1, r2, why = best_year_of_plenty(state, me)
            return why if (action[1], action[2]) == (r1, r2) else "alternative pick"
        if k == A.PLAY_MONOPOLY:
            from .devcards import best_monopoly
            r, v, why = best_monopoly(state, me, belief)
            return why if action[1] == r else "alternative resource"
        if k == A.ROLL:
            return "roll the dice"
    except Exception:  # explanations must never break the search
        return ""
    return ""


def search_determinized(state: GameState, me: int, evaluator, config: Optional[SearchConfig] = None,
                        samples: int = 4, rng=None, model: Optional[OpponentModel] = None,
                        belief: Optional[HandBelief] = None,
                        politics: Optional[PoliticalState] = None) -> List[ScoredAction]:
    """Average search results over several determinizations of a partially observed state."""
    from .inference import is_fully_known, sample_states
    rng = rng or random.Random(0)
    if is_fully_known(state):
        return Searcher(evaluator, config, model, belief, politics).search(state, me, rng)
    states = sample_states(state, me, samples, rng, belief)
    agg: Dict[Action, List[float]] = {}
    expl: Dict[Action, str] = {}
    lines: Dict[Action, List[Action]] = {}
    for s in states:
        res = Searcher(evaluator, config, model, belief, politics).search(s, me, rng)
        for r in res:
            agg.setdefault(r.action, []).append(r.value)
            expl.setdefault(r.action, r.explanation)
            lines.setdefault(r.action, r.line)
    out = [ScoredAction(a, sum(v) / len(v), expl[a], lines[a], sum(v) / len(v)) for a, v in agg.items()]
    out.sort(key=lambda r: -r.value)
    return out
