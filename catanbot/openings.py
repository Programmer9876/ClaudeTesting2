"""Alternative opening (setup-phase) policies, installable as the tunable ``openings.policy``.

Where the bots decide their setup moves
---------------------------------------
* ``HeuristicBot`` plays the highest ``heuristic.action_priors`` prior.  In the setup phases the
  priors come from two names ``heuristic.py`` imported from ``placement``:
  ``SETUP_SETTLEMENT`` -> ``50 + setup_pick(state, player, k=60)[v]`` and ``SETUP_ROAD`` -> 80 for
  ``setup_road_pick(state, player, setup_last_settlement)[0]``, 20 for every other road.
* ``SearchBot`` runs the expectimax search in the setup phases too: ``action_priors`` only orders
  the candidates (the top ``expand`` settlements), every settlement + road line is played out and
  the end-of-line states are ranked by the evaluator (``static_value`` softmax / value net).  Its
  opening is therefore the evaluator's argmax among the top setup_pick candidates, *not*
  ``setup_pick``'s top spot, and patching the priors alone would not change it.

:func:`install` therefore patches three things (and :func:`uninstall` puts the saved objects back):
``heuristic.setup_pick`` / ``heuristic.setup_road_pick`` (the priors: HeuristicBot, the search's
move ordering, opponent predictions) and ``SearchBot.decide``, which in the two setup phases
returns the policy's choice among the legal actions instead of searching (the search would have
exactly one candidate left, so this is the "filter the legal actions to the variant's choice"
override without running it).  ``placement.setup_pick`` itself is never patched: the policies
below call the real one.  Outside the setup phases nothing changes.

Policies (every ``pick`` has ``placement.setup_pick``'s shape ``(state, player, k) -> [(vertex,
score), ...]`` best first over the free vertices, every ``road`` ``setup_road_pick``'s shape
``(state, player, settlement) -> (edge, score)``; scores are on a pips-like scale so that the
heuristic bot's temperature means the same thing as with the current scorer):

* ``standin_book`` - a port of the opening book of our in-engine Catanatron stand-ins
  (``bench/catanatron_players.py``: ``opening_spot_score`` / ``choose_initial_road``):
  production x resource weight x board scarcity x (1 + 0.15 per extra resource type), for the
  second settlement x (1 + 0.2 per new type) x 1.15 wood+brick+sheep+wheat x 1.08 wheat+ore,
  port bonus, + 0.15 x the best free spot two edges away; road toward the best free spot two
  edges beyond it (three-edge spots at half value).  Scores are the stand-in's x 36.
* ``pips_diversity`` - pips x sqrt(clipped scarcity) x (1 + ``PD_DIVERSITY`` per extra type);
  for the second settlement resources the first one lacks weigh ``1 + PD_COMPLEMENT`` and the
  gain in ``max(min(ore, wheat), min(wood, brick))`` pips is worth ``PD_PAIR``; road toward the
  best free spot two edges away by the same score.
* ``denial`` - ``placement.setup_pick`` plus ``DENIAL_WEIGHT`` x the drop in the best spot score
  (``score_settlement_spot(setup=True)`` from *their* point of view) of the next opponent to pick
  in snake order when we take the spot (taking their favourite, or a neighbour of it, denies it);
  road = ``setup_road_pick``'s per-edge score + ``DENIAL_WEIGHT`` x ``placement.road_block_values``.
  ``denial:W`` is the same policy with weight ``W`` (a valid tunable value, e.g. ``denial:2``).  At
  the default weight 1.0 (a point denied to the next picker counts like a point for us) it changes
  ~20 % of the first and ~10 % of the second settlements of the heuristic ranking (0.5: 12 % / 5 %,
  2.0: 26 % / 19 %, 3.0: 31 % / 24 %; 50 random boards).
* ``setup_pick`` - the current heuristic ranking forced (no search).  A control: identical to the
  default for the heuristic bot; for the search bot it measures what the search adds to the
  opening, so a ``denial`` result can be read against it.

The tunable is registered by ``catanbot/tuning.py`` (``register_tunables``): a weight-kind
tunable whose single target ``openings.CONTROL.policy`` is a property - reading it returns the
installed policy name ("current" when none), assigning it calls :func:`install` /
:func:`uninstall`, so ``tuning.apply`` / ``restore`` and ``ParamBot`` nest and restore it like any
other override.  ``python -m catanbot.openings --boards 50`` prints the average starting
production each policy picks on random boards.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from . import placement as P
from .state import GameState, PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT

POLICY_DEFAULT = "current"
SETUP_PHASES = (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)

# The real scorers, captured at import (the policies call these, never the patched import sites).
_SETUP_PICK = P.setup_pick
_SETUP_ROAD_PICK = P.setup_road_pick

# --- standin_book constants (bench/catanatron_players.py DEFAULT_WEIGHTS / opening_spot_score) ---
STANDIN_RESOURCE_WEIGHTS = (1.0, 1.05, 0.85, 1.2, 1.15)   # wood brick sheep wheat ore
STANDIN_SCALE = 36.0            # the stand-in scores probabilities; x 36 = pips

# --- pips_diversity constants ---------------------------------------------------------------
PD_DIVERSITY = 0.12             # per resource type beyond the first
PD_COMPLEMENT = 0.35            # second settlement: extra weight of a type the first one lacks
PD_PAIR = 0.4                   # per pip gained in max(min(ore, wheat), min(wood, brick))
PD_SCARCITY_CLIP = (0.6, 1.6)   # board scarcity (mean pips / pips) clipped, then square-rooted

# --- denial constants -----------------------------------------------------------------------
DENIAL_WEIGHT = 1.0             # value of one point of the next picker's best-spot drop ("denial:W" overrides)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _free_vertices(occ: Dict[int, int]) -> List[int]:
    return [v for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)]


def _ranked(scored: Sequence[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
    """Best first; exact ties by vertex id so the ranking is deterministic."""
    return sorted(scored, key=lambda t: (-t[1], t[0]))[:max(0, k)]


def vertex_pips_by_resource(state: GameState, v: int) -> List[int]:
    """Pips per resource at ``v`` (the robber is ignored: setup scoring is about the dice)."""
    out = [0] * 5
    for h in B.VERTEX_HEXES[v]:
        res, num = state.hexes[h]
        if res == B.DESERT or num == 0:
            continue
        out[res] += B.PIPS[num]
    return out


def player_pips_by_resource(state: GameState, player: int) -> List[int]:
    p = state.players[player]
    out = [0] * 5
    for v in p.settlements:
        for r, x in enumerate(vertex_pips_by_resource(state, v)):
            out[r] += x
    for v in p.cities:
        for r, x in enumerate(vertex_pips_by_resource(state, v)):
            out[r] += 2 * x
    return out


def pick_order(num_players: int) -> List[int]:
    """Snake order of the setup settlements: 0..n-1, n-1..0."""
    return list(range(num_players)) + list(range(num_players - 1, -1, -1))


def pick_position(state: GameState) -> int:
    """Index of the current setup pick in :func:`pick_order`."""
    n = state.num_players
    c = state.current
    return c if state.setup_round == 0 else n + (n - 1 - c)


def next_opponent_picker(state: GameState, player: int) -> Optional[int]:
    """The first *other* player to place a setup settlement after this pick (``None`` after the last)."""
    order = pick_order(state.num_players)
    for q in order[pick_position(state) + 1:]:
        if q != player:
            return q
    return None


def _settlement_of(state: GameState, player: int, settlement: int) -> int:
    if settlement is not None and settlement >= 0:
        return settlement
    p = state.players[player]
    return p.settlements[-1] if p.settlements else -1


def _free_setup_edges(state: GameState, settlement: int, eocc: Dict[int, int]) -> List[int]:
    return [e for e in B.VERTEX_EDGES[settlement] if e not in eocc]


# ---------------------------------------------------------------------------
# standin_book: port of bench/catanatron_players.py's opening book
# ---------------------------------------------------------------------------
def _standin_scarcity(state: GameState) -> List[float]:
    board_vec = [0.0] * 5
    for res, num in state.hexes:
        if res == B.DESERT or num == 0:
            continue
        board_vec[res] += B.PIPS[num] / 36.0
    mean = sum(board_vec) / 5.0
    return [min(1.6, max(0.6, mean / x)) if x > 0 else 1.6 for x in board_vec]


class _StandinTables:
    """Per-state lookups of the stand-in scorer (MapTables: node production and scarcity)."""

    __slots__ = ("state", "vec", "total", "scarcity", "occ")

    def __init__(self, state: GameState):
        self.state = state
        self.vec = [P.vertex_production(state, v, ignore_robber=True) for v in range(B.NUM_VERTICES)]
        self.total = [sum(x) for x in self.vec]
        self.scarcity = _standin_scarcity(state)
        self.occ = state.occupied_vertices()

    def spot(self, v: int, own: Sequence[float]) -> float:
        """``opening_spot_score`` (probability units)."""
        vec = self.vec[v]
        sc = self.scarcity
        total = 0.0
        kinds = 0
        new_kinds = 0
        for ri in range(5):
            p = vec[ri]
            if p <= 0.0:
                continue
            kinds += 1
            if own[ri] <= 0.0:
                new_kinds += 1
            total += p * STANDIN_RESOURCE_WEIGHTS[ri] * sc[ri]
        if kinds == 0:
            return 0.0
        score = total * (1.0 + 0.15 * (kinds - 1))
        if any(p > 0.0 for p in own):
            score *= 1.0 + 0.2 * new_kinds
            union = [own[i] + vec[i] for i in range(5)]
            if union[B.WOOD] > 0 and union[B.BRICK] > 0 and union[B.SHEEP] > 0 and union[B.WHEAT] > 0:
                score *= 1.15
            if union[B.WHEAT] > 0 and union[B.ORE] > 0:
                score *= 1.08
        port = self.state.ports.get(v)
        if port is not None:
            if port == B.PORT_GENERIC:
                score += 0.012
            else:
                score += 0.02 if vec[port] > 0 else 0.005
        near = B.VERTEX_NEIGHBORS[v]
        best_far = 0.0
        for n in near:
            for m in B.VERTEX_NEIGHBORS[n]:
                if m != v and m not in near and P.is_free_vertex(self.occ, m):
                    t = self.total[m]
                    if t > best_far:
                        best_far = t
        return score + 0.15 * best_far


def standin_spot_scores(state: GameState, player: int) -> Dict[int, float]:
    """Stand-in opening score of every free vertex, in the stand-in's own (probability) units."""
    tb = _StandinTables(state)
    own = P.player_production(state, player, ignore_robber=True)
    return {v: tb.spot(v, own) for v in _free_vertices(tb.occ)}


def standin_pick(state: GameState, player: int, k: int = 5) -> List[Tuple[int, float]]:
    scores = standin_spot_scores(state, player)
    return _ranked([(v, STANDIN_SCALE * s) for v, s in scores.items()], k)


def standin_road_scores(state: GameState, player: int, settlement: int) -> Dict[int, float]:
    """``choose_initial_road``'s score per free edge (probability units)."""
    settlement = _settlement_of(state, player, settlement)
    tb = _StandinTables(state)
    occ = tb.occ
    own = P.player_production(state, player, ignore_robber=True)
    cache: Dict[int, float] = {}

    def spot(v: int) -> float:
        s = cache.get(v)
        if s is None:
            s = cache[v] = tb.spot(v, own)
        return s

    out: Dict[int, float] = {}
    for e in _free_setup_edges(state, settlement, state.occupied_edges()):
        a, b = B.EDGE_VERTICES[e]
        far = b if a == settlement else a
        score = -1.0 if far in occ else 0.0
        for m in B.VERTEX_NEIGHBORS[far]:
            if m == settlement or not P.is_free_vertex(occ, m):
                continue
            s = spot(m)
            # a spot three edges away counts a little, so dead ends are avoided
            for q in B.VERTEX_NEIGHBORS[m]:
                if q != far and P.is_free_vertex(occ, q):
                    s = max(s, 0.5 * spot(q))
            if s > score:
                score = s
        out[e] = score
    return out


def _best_edge(scores: Dict[int, float], settlement: int) -> Tuple[int, float]:
    if not scores:
        for e in B.VERTEX_EDGES[settlement]:
            return e, 0.0
        return -1, 0.0
    e = max(scores, key=lambda x: (scores[x], -x))
    return e, scores[e]


def standin_road(state: GameState, player: int, settlement: int) -> Tuple[int, float]:
    settlement = _settlement_of(state, player, settlement)
    scores = standin_road_scores(state, player, settlement)
    e, s = _best_edge(scores, settlement)
    return e, STANDIN_SCALE * s


# ---------------------------------------------------------------------------
# pips_diversity
# ---------------------------------------------------------------------------
def _pd_resource_weights(state: GameState) -> List[float]:
    pips = P.board_pips_by_resource(state)
    mean = sum(pips) / 5.0
    lo, hi = PD_SCARCITY_CLIP
    return [(min(hi, max(lo, mean / p)) if p > 0 else hi) ** 0.5 for p in pips]


def _pair(x: Sequence[float]) -> float:
    return max(min(x[B.ORE], x[B.WHEAT]), min(x[B.WOOD], x[B.BRICK]))


def pips_diversity_spot(state: GameState, v: int, own_pips: Sequence[int], weights: Sequence[float]) -> float:
    vp = vertex_pips_by_resource(state, v)
    has_own = any(x > 0 for x in own_pips)
    base = 0.0
    kinds = 0
    for r in range(5):
        if vp[r] <= 0:
            continue
        kinds += 1
        w = weights[r]
        if has_own and own_pips[r] <= 0:
            w *= 1.0 + PD_COMPLEMENT
        base += vp[r] * w
    if kinds == 0:
        return 0.0
    score = base * (1.0 + PD_DIVERSITY * (kinds - 1))
    union = [own_pips[r] + vp[r] for r in range(5)]
    return score + PD_PAIR * (_pair(union) - _pair(own_pips))


def pips_diversity_pick(state: GameState, player: int, k: int = 5) -> List[Tuple[int, float]]:
    occ = state.occupied_vertices()
    own = player_pips_by_resource(state, player)
    w = _pd_resource_weights(state)
    return _ranked([(v, pips_diversity_spot(state, v, own, w)) for v in _free_vertices(occ)], k)


def pips_diversity_road_scores(state: GameState, player: int, settlement: int) -> Dict[int, float]:
    """Per free edge: the best free spot two edges away (+0.01 per such spot, against dead ends)."""
    settlement = _settlement_of(state, player, settlement)
    occ = state.occupied_vertices()
    own = player_pips_by_resource(state, player)
    w = _pd_resource_weights(state)
    out: Dict[int, float] = {}
    for e in _free_setup_edges(state, settlement, state.occupied_edges()):
        a, b = B.EDGE_VERTICES[e]
        far = b if a == settlement else a
        best = 0.0
        count = 0
        if far not in occ:
            for m in B.VERTEX_NEIGHBORS[far]:
                if m == settlement or not P.is_free_vertex(occ, m):
                    continue
                count += 1
                best = max(best, pips_diversity_spot(state, m, own, w))
        out[e] = best + 0.01 * count
    return out


def pips_diversity_road(state: GameState, player: int, settlement: int) -> Tuple[int, float]:
    settlement = _settlement_of(state, player, settlement)
    return _best_edge(pips_diversity_road_scores(state, player, settlement), settlement)


# ---------------------------------------------------------------------------
# denial
# ---------------------------------------------------------------------------
def denial_bonus(state: GameState, player: int, vertices: Sequence[int]) -> Dict[int, float]:
    """Drop in the next opponent picker's best setup-spot score if we take each of ``vertices``."""
    opp = next_opponent_picker(state, player)
    if opp is None:
        return {v: 0.0 for v in vertices}
    occ = state.occupied_vertices()
    own = P.player_production(state, opp, ignore_robber=True)
    scarcity = P.resource_scarcity(state)
    bctx = P.BlockContext(state, opp, scarcity)
    theirs = sorted(((P.score_settlement_spot(state, opp, w, occ=occ, own_prod=own, scarcity=scarcity, setup=True,
                                              block_ctx=bctx), w) for w in _free_vertices(occ)), reverse=True)
    if not theirs:
        return {v: 0.0 for v in vertices}
    best = theirs[0][0]
    out: Dict[int, float] = {}
    for v in vertices:
        blocked = set(B.VERTEX_NEIGHBORS[v])
        blocked.add(v)
        after = next((s for s, w in theirs if w not in blocked), 0.0)
        out[v] = max(0.0, best - after)
    return out


def denial_pick(state: GameState, player: int, k: int = 5, weight: Optional[float] = None) -> List[Tuple[int, float]]:
    w = DENIAL_WEIGHT if weight is None else weight
    base = _SETUP_PICK(state, player, k=B.NUM_VERTICES)
    bonus = denial_bonus(state, player, [v for v, _ in base])
    return _ranked([(v, s + w * bonus[v]) for v, s in base], k)


def current_road_scores(state: GameState, player: int, settlement: int) -> Dict[int, float]:
    """``placement.setup_road_pick``'s score of every free edge (its argmax is setup_road_pick's edge)."""
    settlement = _settlement_of(state, player, settlement)
    occ = state.occupied_vertices()
    eocc = state.occupied_edges()
    own_prod = P.player_production(state, player, ignore_robber=True)
    scarcity = P.resource_scarcity(state)
    bctx = P.BlockContext(state, player, scarcity)
    out: Dict[int, float] = {}
    for e in B.VERTEX_EDGES[settlement]:
        if e in eocc:
            continue
        a, b = B.EDGE_VERTICES[e]
        w = b if a == settlement else a
        if w in occ:
            continue
        s = 0.0
        for e2 in B.VERTEX_EDGES[w]:
            if e2 in eocc or e2 == e:
                continue
            a2, b2 = B.EDGE_VERTICES[e2]
            x = b2 if a2 == w else a2
            if P.is_free_vertex(occ, x) and x != settlement:
                s = max(s, P.score_settlement_spot(state, player, x, occ=occ, own_prod=own_prod, scarcity=scarcity,
                                                   block_ctx=bctx))
        out[e] = s
    return out


def denial_road(state: GameState, player: int, settlement: int, weight: Optional[float] = None) -> Tuple[int, float]:
    w = DENIAL_WEIGHT if weight is None else weight
    settlement = _settlement_of(state, player, settlement)
    scores = current_road_scores(state, player, settlement)
    if not scores:
        return _SETUP_ROAD_PICK(state, player, settlement)
    block = P.road_block_values(state, player, list(scores))
    total = {e: s + w * block.get(e, 0.0) for e, s in scores.items()}
    return _best_edge(total, settlement)


# ---------------------------------------------------------------------------
# Policy table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpeningPolicy:
    name: str
    pick: Callable[..., List[Tuple[int, float]]]      # (state, player, k) -> [(vertex, score)], best first
    road: Callable[..., Tuple[int, float]]            # (state, player, settlement) -> (edge, score)
    description: str


def _control_pick(state: GameState, player: int, k: int = 5) -> List[Tuple[int, float]]:
    return _SETUP_PICK(state, player, k=k)


def _control_road(state: GameState, player: int, settlement: int) -> Tuple[int, float]:
    return _SETUP_ROAD_PICK(state, player, _settlement_of(state, player, settlement))


POLICIES: Dict[str, OpeningPolicy] = {
    "standin_book": OpeningPolicy("standin_book", standin_pick, standin_road,
                                  "port of the Catanatron stand-ins' opening book"),
    "pips_diversity": OpeningPolicy("pips_diversity", pips_diversity_pick, pips_diversity_road,
                                    "pips x diversity x scarcity, second-settlement complement and pair terms"),
    "denial": OpeningPolicy("denial", denial_pick, denial_road,
                            "setup_pick + denying the next picker's favourite spot; road + road_block_values"),
    "setup_pick": OpeningPolicy("setup_pick", _control_pick, _control_road,
                                "control: placement.setup_pick / setup_road_pick forced (no search)"),
}
CANDIDATES = ["standin_book", "pips_diversity", "denial", "setup_pick"]


_PARAMETRIC: Dict[str, OpeningPolicy] = {}


def policy(name: str) -> OpeningPolicy:
    """The policy called ``name``; ``denial:W`` is denial with ``DENIAL_WEIGHT = W`` (e.g. ``denial:2``)."""
    pol = POLICIES.get(name) or _PARAMETRIC.get(name)
    if pol is not None:
        return pol
    base, sep, arg = str(name).partition(":")
    if sep and base == "denial":
        try:
            w = float(arg)
        except ValueError:
            w = float("nan")
        if not (0.0 <= w < float("inf")):
            raise ValueError(f"denial weight must be a number >= 0, got {name!r}")
        pol = OpeningPolicy(name, partial(denial_pick, weight=w), partial(denial_road, weight=w),
                            f"denial with weight {w:g}")
        _PARAMETRIC[name] = pol
        return pol
    raise ValueError(f"unknown opening policy {name!r}; known: {POLICY_DEFAULT}, {', '.join(POLICIES)}, denial:W")


def choose(state: GameState, legal_actions: Sequence[A.Action], name: Optional[str] = None) -> Optional[A.Action]:
    """The policy's setup action among ``legal_actions`` (``None`` outside the setup phases / when none fits).

    ``name`` defaults to the installed policy.
    """
    name = name or _active
    if name is None or name == POLICY_DEFAULT:
        return None
    pol = policy(name)
    player = state.current
    if state.phase == PHASE_SETUP_SETTLEMENT:
        legal_v = {a[1]: a for a in legal_actions if a[0] == A.SETUP_SETTLEMENT}
        if not legal_v:
            return None
        for v, _ in pol.pick(state, player, B.NUM_VERTICES):
            if v in legal_v:
                return legal_v[v]
        return None
    if state.phase == PHASE_SETUP_ROAD:
        settlement = _settlement_of(state, player, state.setup_last_settlement)
        if settlement < 0:
            return None
        e, _ = pol.road(state, player, settlement)
        a = (A.SETUP_ROAD, e)
        return a if a in legal_actions else None
    return None


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------
_active: Optional[str] = None
_saved: Optional[List[Tuple[object, str, object]]] = None


def _patched_setup_pick(state: GameState, player: int, k: int = 5) -> List[Tuple[int, float]]:
    return policy(_active).pick(state, player, k) if _active else _SETUP_PICK(state, player, k=k)


def _patched_setup_road_pick(state: GameState, player: int, settlement: int) -> Tuple[int, float]:
    return policy(_active).road(state, player, settlement) if _active else _SETUP_ROAD_PICK(state, player, settlement)


def _search_decide(self, state, legal_actions, rng):
    """``SearchBot.decide`` while a policy is installed: the policy's move in the setup phases."""
    if _active is not None and len(legal_actions) > 1 and state.phase in SETUP_PHASES:
        a = choose(state, legal_actions)
        if a is not None:
            from .search import ScoredAction
            self._game_bias(rng)
            rng.random()     # the search path draws exactly one number (its Random seed): keep the stream aligned
            self.last_results = [ScoredAction(a, 0.0, f"opening policy {_active}", [a])]
            return a
    return _original_search_decide()(self, state, legal_actions, rng)


def _original_search_decide():
    for obj, attr, value in _saved or ():
        if attr == "decide":
            return value
    from .agents.search_bot import SearchBot
    return SearchBot.decide


def installed() -> str:
    """Name of the installed policy ("current" when none)."""
    return _active or POLICY_DEFAULT


def install(name: Optional[str]) -> None:
    """Make both bot types open with policy ``name`` ("current" / ``None`` = :func:`uninstall`).

    Installing over an installed policy only switches the policy; the saved originals stay.
    """
    global _active, _saved
    if name is None or name == POLICY_DEFAULT:
        uninstall()
        return
    policy(name)   # validate before touching anything
    if _saved is None:
        from . import heuristic
        from .agents.search_bot import SearchBot
        saved = [(heuristic, "setup_pick", heuristic.setup_pick),
                 (heuristic, "setup_road_pick", heuristic.setup_road_pick),
                 (SearchBot, "decide", SearchBot.__dict__["decide"])]
        heuristic.setup_pick = _patched_setup_pick
        heuristic.setup_road_pick = _patched_setup_road_pick
        SearchBot.decide = _search_decide
        _saved = saved
    _active = name


def uninstall() -> None:
    """Restore exactly the objects :func:`install` replaced (no-op when nothing is installed)."""
    global _active, _saved
    saved, _saved = _saved, None
    _active = None
    for obj, attr, value in reversed(saved or ()):
        setattr(obj, attr, value)


class using:
    """``with using("denial"): ...`` - install on entry, restore the previous policy on exit."""

    def __init__(self, name: Optional[str]):
        self.name = name
        self.prev = POLICY_DEFAULT

    def __enter__(self):
        self.prev = installed()
        install(self.name)
        return self

    def __exit__(self, *exc):
        install(self.prev)
        return False


class _Control:
    """The tuning target: ``CONTROL.policy`` reads the installed policy; assigning installs one."""

    @property
    def policy(self) -> str:
        return installed()

    @policy.setter
    def policy(self, name: str) -> None:
        install(name)


CONTROL = _Control()


# ---------------------------------------------------------------------------
# Tunable registration (called from the last line of catanbot/tuning.py)
# ---------------------------------------------------------------------------
def parse_policy(text: str) -> str:
    name = text.strip()
    if name != POLICY_DEFAULT:
        policy(name)
    return name


def register_tunables(registry: Dict[str, object]) -> None:
    """Add ``openings.policy`` to the ``tuning.TUNABLES`` registry."""
    tuning = sys.modules.get("catanbot.tuning") or importlib.import_module("catanbot.tuning")
    keep = tuning.KEEP

    def make(value):
        return keep if value in (None, POLICY_DEFAULT) else parse_policy(value)

    t = tuning.Tunable(
        name="openings.policy", module=__name__, attr="CONTROL.policy", default=POLICY_DEFAULT, kind="weight",
        candidates=list(CANDIDATES), targets=(tuning.Target(__name__, "CONTROL.policy"),), make=make,
        parse=parse_policy,
        description="opening (setup-phase) policy for both bots: standin_book = the Catanatron stand-ins' book, "
                    "pips_diversity, denial = setup_pick + deny the next picker's favourite (denial:W sets its weight, "
                    "default 1), setup_pick = the "
                    "heuristic ranking forced without search (control); implemented by patching "
                    "heuristic.setup_pick / setup_road_pick and SearchBot.decide (catanbot/openings.py)")
    registry[t.name] = t


# ---------------------------------------------------------------------------
# Starting-production report: python -m catanbot.openings --boards 50
# ---------------------------------------------------------------------------
def play_setup(state: GameState, deciders: Sequence[Callable[[GameState, List[A.Action]], A.Action]]) -> GameState:
    """Play the setup phase; ``deciders[i](state, legal)`` picks seat ``i``'s move."""
    from . import engine as E
    while state.phase in SETUP_PHASES:
        legal = E.legal_actions(state)
        a = legal[0] if len(legal) == 1 else deciders[state.current](state, legal)
        state = E.apply(state, a, None)
    return state


def starting_pips(state: GameState, player: int) -> Tuple[int, int]:
    """(total pips of the player's buildings, distinct resource types they produce)."""
    pips = player_pips_by_resource(state, player)
    return sum(pips), sum(1 for x in pips if x > 0)


def _report(boards: int, seed: int, with_search: bool) -> None:
    import random
    import time
    from .agents.heuristic_bot import HeuristicBot
    from .selfplay import make_bot
    from .state import new_game

    heur = HeuristicBot()
    rng = random.Random(seed)

    def heuristic_decider(s, legal):
        return heur.decide(s, legal, rng)

    def policy_decider(name):
        return lambda s, legal: choose(s, legal, name)

    arms: List[Tuple[str, Callable]] = [("current (heuristic bot)", heuristic_decider)]
    if with_search:
        sbot = make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic")
        sbot.reset()
        arms.append(("current (search bot)", lambda s, legal: sbot.decide(s, legal, rng)))
    arms += [(n, policy_decider(n)) for n in CANDIDATES]
    print(f"{boards} random 4-player boards (seed {seed}); candidate seat = board % 4, the other three seats "
          f"play the current heuristic opening; 'all' = every seat uses the arm")
    print(f"{'arm':26} {'cand pips':>9} {'cand types':>10} {'all pips':>8} {'all types':>9} {'s':>6}")
    for label, decider in arms:
        t0 = time.perf_counter()
        cp = ct = ap = at = 0.0
        for b in range(boards):
            base = new_game(4, rng=random.Random(seed * 1000 + b))
            seat = b % 4
            ds = [heuristic_decider] * 4
            ds[seat] = decider
            end = play_setup(base.copy(), ds)
            p, t = starting_pips(end, seat)
            cp += p
            ct += t
            end = play_setup(base.copy(), [decider] * 4)
            for i in range(4):
                p, t = starting_pips(end, i)
                ap += p / 4.0
                at += t / 4.0
        n = float(boards)
        print(f"{label:26} {cp / n:9.2f} {ct / n:10.2f} {ap / n:8.2f} {at / n:9.2f} {time.perf_counter() - t0:6.1f}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Average starting production of each opening policy on random boards")
    ap.add_argument("--boards", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-search", action="store_true", help="skip the search bot's current opening (slower)")
    args = ap.parse_args(argv)
    _report(args.boards, args.seed, not args.no_search)


if __name__ == "__main__":
    main()
