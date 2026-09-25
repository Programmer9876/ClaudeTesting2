"""Placement heuristics: settlement spots, cities and road targets.

These functions only depend on the board tables and the state; they are used
by the heuristic evaluator, by move ordering in the search and to produce
human readable explanations ("best spot: 10 pips, wheat/ore, gives a 2:1 ore
port").  Nothing here mutates the state.

Spot scores also charge *blockability*: buildings stacked on one hex (or a
city on a hex we already work, or a hex the leader also works) are worth less
than their pips because one robber placement blocks all of them - see the
"Blockability" section below.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import board as B
from .state import GameState

# How much each resource is worth over a whole game, derived from the build
# costs (road 1W1B, settlement 1W1B1S1Wh, city 2Wh3O, dev 1S1Wh1O) with a
# little extra weight on ore/wheat because cities and dev cards dominate the
# late game.  Normalised to mean 1.
_RAW_DEMAND = [1.0, 1.0, 0.9, 1.25, 1.2]
_MEAN = sum(_RAW_DEMAND) / 5.0
RESOURCE_DEMAND = [d / _MEAN for d in _RAW_DEMAND]


# ---------------------------------------------------------------------------
# Production helpers
# ---------------------------------------------------------------------------
def vertex_production(state: GameState, v: int, ignore_robber: bool = False) -> List[float]:
    """Expected cards per roll for a settlement on ``v`` (per resource)."""
    out = [0.0] * 5
    for h in B.VERTEX_HEXES[v]:
        res, num = state.hexes[h]
        if res == B.DESERT or num == 0:
            continue
        if not ignore_robber and h == state.robber:
            continue
        out[res] += B.PIPS[num] / 36.0
    return out


def vertex_pips(state: GameState, v: int, ignore_robber: bool = True) -> int:
    """Total pips (dots) at a vertex."""
    total = 0
    for h in B.VERTEX_HEXES[v]:
        res, num = state.hexes[h]
        if res == B.DESERT or num == 0:
            continue
        if not ignore_robber and h == state.robber:
            continue
        total += B.PIPS[num]
    return total


def player_production(state: GameState, player: int, ignore_robber: bool = False) -> List[float]:
    """Expected cards per roll for a player (per resource), cities count double."""
    p = state.players[player]
    out = [0.0] * 5
    for v in p.settlements:
        pr = vertex_production(state, v, ignore_robber)
        for i in range(5):
            out[i] += pr[i]
    for v in p.cities:
        pr = vertex_production(state, v, ignore_robber)
        for i in range(5):
            out[i] += 2.0 * pr[i]
    return out


def board_pips_by_resource(state: GameState) -> List[int]:
    pips = [0] * 5
    for res, num in state.hexes:
        if res != B.DESERT and num:
            pips[res] += B.PIPS[num]
    return pips


def resource_scarcity(state: GameState) -> List[float]:
    """Weight > 1 for resources that are scarce on this board (mean pips / pips)."""
    pips = board_pips_by_resource(state)
    mean = sum(pips) / 5.0
    return [mean / max(p, 1) for p in pips]


# ---------------------------------------------------------------------------
# Occupancy / legality helpers (duplicated from the engine on purpose so this
# module stays independent of it)
# ---------------------------------------------------------------------------
def is_free_vertex(occ: Dict[int, int], v: int) -> bool:
    """Distance rule: v and all its neighbours are unoccupied."""
    if v in occ:
        return False
    for n in B.VERTEX_NEIGHBORS[v]:
        if n in occ:
            return False
    return True


def network_vertices(state: GameState, player: int) -> List[int]:
    """Vertices touched by the player's roads or buildings."""
    p = state.players[player]
    verts = set(p.settlements) | set(p.cities)
    for e in p.roads:
        a, b = B.EDGE_VERTICES[e]
        verts.add(a)
        verts.add(b)
    return sorted(verts)


def buildable_settlements(state: GameState, player: int, occ: Optional[Dict[int, int]] = None) -> List[int]:
    """Free vertices adjacent to one of the player's roads."""
    occ = state.occupied_vertices() if occ is None else occ
    p = state.players[player]
    seen = set()
    out = []
    for e in p.roads:
        for v in B.EDGE_VERTICES[e]:
            if v in seen:
                continue
            seen.add(v)
            if is_free_vertex(occ, v):
                out.append(v)
    return out


def reachable_spots(state: GameState, player: int, max_roads: int = 3,
                    occ: Optional[Dict[int, int]] = None,
                    eocc: Optional[Dict[int, int]] = None) -> Dict[int, Tuple[int, int]]:
    """Free settlement spots reachable by building roads.

    Returns ``{vertex: (roads_needed, first_edge)}`` where ``roads_needed`` is
    the number of additional roads (0 = buildable now) and ``first_edge`` the
    first road to build towards it (-1 when none needed).  Paths stop at
    opponent buildings and never use occupied edges.  ``eocc`` (edge -> owner)
    defaults to the state's occupied edges; pass a modified copy to ask "what
    if this edge were taken" (see :func:`road_block_values`).
    """
    occ = state.occupied_vertices() if occ is None else occ
    eocc = state.occupied_edges() if eocc is None else eocc
    p = state.players[player]
    own_build = set(p.settlements) | set(p.cities)
    start = set(own_build)
    for e in p.roads:
        start.update(B.EDGE_VERTICES[e])
    # BFS over vertices; distance = roads to build.  A start vertex that is an
    # opponent building cannot be expanded from (it cuts the road).
    dist: Dict[int, int] = {}
    first: Dict[int, int] = {}
    dq = deque()
    for v in start:
        if v in occ and occ[v] != player:
            continue
        dist[v] = 0
        first[v] = -1
        dq.append(v)
    while dq:
        v = dq.popleft()
        d = dist[v]
        if d >= max_roads:
            continue
        for e in B.VERTEX_EDGES[v]:
            if e in eocc:
                continue
            a, b = B.EDGE_VERTICES[e]
            w = b if a == v else a
            if w in dist and dist[w] <= d + 1:
                continue
            if w in occ and occ[w] != player:
                continue  # opponent building blocks the path (cannot pass through)
            dist[w] = d + 1
            first[w] = e if first[v] == -1 else first[v]
            dq.append(w)
    out: Dict[int, Tuple[int, int]] = {}
    for v, d in dist.items():
        if is_free_vertex(occ, v):
            out[v] = (d, first[v])
    return out


# ---------------------------------------------------------------------------
# Blockability: how much of our income one robber placement can switch off
# ---------------------------------------------------------------------------
# The robber sits on one hex.  Two of our buildings on the same hex (or a city
# on it) concentrate our income where a single robber placement - and the
# robber goes to the juiciest hex of its victim - blocks all of it.  The spot
# scorers therefore charge a candidate building for the *increase* of our
# robber exposure, on the same pips-equivalent scale as their production term:
#
#   W(h)        = our demand-weighted pips on hex h (settlement = pips, city = 2 x pips,
#                 times RESOURCE_DEMAND[res] x scarcity[res] ** 0.5; the robber's current
#                 position is ignored - the term is about where it *can* go)
#   P_block(h)  = PLACEMENT_ROBBER_Q x W(h)^2 / sum_h' W(h')^2 x (1 + 0.5 x shared_strong(h))
#   exposure    = sum_h P_block(h) x W(h)
#   penalty     = PLACEMENT_BLOCK_WEIGHT x (exposure_after - exposure_before)
#
# The square in P_block models opponents aiming the robber at our best hex, so
# concentration is costly; shared_strong(h) = 1 when an opponent with
# robber.threat >= PLACEMENT_STRONG_THREAT (5+ estimated VP) has a building on
# h, because the robber visits the hexes of strong players anyway.  See
# docs/STRATEGY.md (Placement) for worked examples.  cpp/heuristic.cpp mirrors
# every function below bit for bit (score_spot / BlockContext).
PLACEMENT_ROBBER_Q = 0.35
"""Fraction of the time the robber sits on one of *our* hexes when we are an
ordinary target (nobody singles us out).  Tunable; the penalty scales with it."""

PLACEMENT_BLOCK_WEIGHT = 1.5
"""Weight of the blockability penalty on the pips-equivalent scale of
``score_settlement_spot`` / ``score_city``.  1.5 makes a second settlement on a
6 we already build on cost ~1.7 pips against a second 6 elsewhere (a 5-pip hex
with ordinary demand weights); set to 0 to switch the term off."""

PLACEMENT_STRONG_THREAT = 1.3
"""An opponent whose ``robber.threat`` is at least this (5+ estimated VP: the
threat is 1 + 0.3 x (VP - 4)) attracts the robber to their hexes regardless of
us; a hex we share with such a player counts as blocked 1.5x as often.  An
absolute threshold (not "the strongest at the table") so that early in the
game, when nobody is a robber magnet yet, only concentration is charged."""


def hex_block_weights(state: GameState, scarcity: Optional[Sequence[float]] = None) -> List[float]:
    """Per hex: pips x RESOURCE_DEMAND x scarcity ** 0.5 (0 for the desert) - a settlement's W(h)."""
    scarcity = resource_scarcity(state) if scarcity is None else scarcity
    out = [0.0] * B.NUM_HEXES
    for h in range(B.NUM_HEXES):
        res, num = state.hexes[h]
        if res == B.DESERT or num == 0:
            continue
        out[h] = B.PIPS[num] * RESOURCE_DEMAND[res] * (scarcity[res] ** 0.5)
    return out


def strong_opponent_hexes(state: GameState, player: int) -> List[int]:
    """1 for every hex on which an opponent with ``robber.threat >= PLACEMENT_STRONG_THREAT`` has a building."""
    from .robber import threat  # local import: robber.py imports this module

    strong = [0] * B.NUM_HEXES
    for i in range(state.num_players):
        if i == player:
            continue
        if threat(state, i) < PLACEMENT_STRONG_THREAT:
            continue
        q = state.players[i]
        for v in q.settlements:
            for h in B.VERTEX_HEXES[v]:
                strong[h] = 1
        for v in q.cities:
            for h in B.VERTEX_HEXES[v]:
                strong[h] = 1
    return strong


def _building_block_weights(state: GameState, player: int, hex_w: Sequence[float],
                            extra_settlement: Optional[int] = None,
                            extra_city: Optional[int] = None) -> List[float]:
    """W(h) of the player's buildings (settlements, then cities, then the extras; a city counts twice).

    ``extra_city`` equal to one of our settlements is the upgrade of that settlement.
    """
    w = [0.0] * B.NUM_HEXES
    p = state.players[player]
    for v in p.settlements:
        if v == extra_city:
            continue
        for h in B.VERTEX_HEXES[v]:
            w[h] += hex_w[h]
    for v in p.cities:
        for h in B.VERTEX_HEXES[v]:
            w[h] += 2.0 * hex_w[h]
    if extra_settlement is not None:
        for h in B.VERTEX_HEXES[extra_settlement]:
            w[h] += hex_w[h]
    if extra_city is not None:
        for h in B.VERTEX_HEXES[extra_city]:
            w[h] += 2.0 * hex_w[h]
    return w


def _exposure_of(w: Sequence[float], strong: Sequence[int]) -> float:
    """sum_h P_block(h) x W(h) for the per-hex weights ``w`` (0 without buildings)."""
    ssq = 0.0
    for h in range(B.NUM_HEXES):
        ssq += w[h] * w[h]
    if ssq <= 0.0:
        return 0.0
    total = 0.0
    for h in range(B.NUM_HEXES):
        x = w[h]
        if x <= 0.0:
            continue
        p_block = PLACEMENT_ROBBER_Q * (x * x) / ssq * (1.0 + 0.5 * strong[h])
        total += p_block * x
    return total


def robber_exposure(state: GameState, player: int, extra_settlement: Optional[int] = None,
                    extra_city: Optional[int] = None, scarcity: Optional[Sequence[float]] = None,
                    hex_w: Optional[Sequence[float]] = None, strong: Optional[Sequence[int]] = None) -> float:
    """Expected demand-weighted pips of ``player`` blocked by the robber (see the module comment).

    ``extra_settlement`` / ``extra_city`` evaluate the position *after* that
    building is added (``extra_city`` on one of our settlements = its upgrade).
    Pure function of the state, O(#our buildings x 3 + #opponent buildings x 3).
    """
    hex_w = hex_block_weights(state, scarcity) if hex_w is None else hex_w
    strong = strong_opponent_hexes(state, player) if strong is None else strong
    return _exposure_of(_building_block_weights(state, player, hex_w, extra_settlement, extra_city), strong)


def block_penalty(state: GameState, player: int, extra_settlement: Optional[int] = None,
                  extra_city: Optional[int] = None, scarcity: Optional[Sequence[float]] = None) -> float:
    """``PLACEMENT_BLOCK_WEIGHT x (exposure after the building - exposure before)``; 0 for a first building."""
    hex_w = hex_block_weights(state, scarcity)
    strong = strong_opponent_hexes(state, player)
    before = robber_exposure(state, player, hex_w=hex_w, strong=strong)
    after = robber_exposure(state, player, extra_settlement, extra_city, hex_w=hex_w, strong=strong)
    return PLACEMENT_BLOCK_WEIGHT * (after - before)


class BlockContext:
    """Everything the blockability term of :func:`score_settlement_spot` needs once per (state, player).

    ``settlement_penalty(v)`` is ``block_penalty(state, player, extra_settlement=v)`` computed from the
    cached weights; callers scoring many candidates build one context and pass it as ``block_ctx``.
    """

    __slots__ = ("hex_w", "strong", "base", "exposure_before")

    def __init__(self, state: GameState, player: int, scarcity: Optional[Sequence[float]] = None):
        self.hex_w = hex_block_weights(state, scarcity)
        self.strong = strong_opponent_hexes(state, player)
        self.base = _building_block_weights(state, player, self.hex_w)
        self.exposure_before = _exposure_of(self.base, self.strong)

    def exposure_with_settlement(self, v: int) -> float:
        w = list(self.base)
        for h in B.VERTEX_HEXES[v]:
            w[h] += self.hex_w[h]
        return _exposure_of(w, self.strong)

    def settlement_penalty(self, v: int) -> float:
        return PLACEMENT_BLOCK_WEIGHT * (self.exposure_with_settlement(v) - self.exposure_before)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _expansion_potential(state: GameState, v: int, occ: Dict[int, int], eocc: Dict[int, int]) -> int:
    """Number of free spots two roads away from ``v`` (rough future growth)."""
    count = 0
    for e in B.VERTEX_EDGES[v]:
        if e in eocc:
            continue
        a, b = B.EDGE_VERTICES[e]
        w = b if a == v else a
        if w in occ:
            continue
        for e2 in B.VERTEX_EDGES[w]:
            if e2 in eocc:
                continue
            a2, b2 = B.EDGE_VERTICES[e2]
            x = b2 if a2 == w else a2
            if x != v and is_free_vertex(occ, x):
                count += 1
    return count


def score_settlement_spot(state: GameState, player: int, v: int,
                          occ: Optional[Dict[int, int]] = None,
                          own_prod: Optional[Sequence[float]] = None,
                          scarcity: Optional[Sequence[float]] = None,
                          setup: bool = False,
                          block_ctx: Optional["BlockContext"] = None) -> float:
    """Heuristic value of placing a settlement on ``v`` for ``player``.

    Combines pips weighted by resource demand and board scarcity, diminishing
    returns on resources the player already produces, diversity, new
    resource types, port synergy and expansion potential, minus the
    blockability penalty (the extra robber exposure of stacking buildings on
    a hex, see :func:`robber_exposure`; ``block_ctx`` caches its per-player
    part).  Scale: roughly "pips-equivalents" (a 10-pip diverse spot scores
    ~12-16).  ``cpp/heuristic.cpp::score_spot`` is the bit-exact port.
    """
    occ = state.occupied_vertices() if occ is None else occ
    own_prod = player_production(state, player, ignore_robber=True) if own_prod is None else own_prod
    scarcity = resource_scarcity(state) if scarcity is None else scarcity
    prod = vertex_production(state, v, ignore_robber=True)
    score = 0.0
    distinct = 0
    new_types = 0
    for r in range(5):
        if prod[r] <= 0:
            continue
        distinct += 1
        have = own_prod[r]
        if have <= 0:
            new_types += 1
        complement = 1.0 / (1.0 + 4.0 * have)      # 1.0 when we have none, ~0.5 at 9 pips
        score += prod[r] * 36.0 * RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5) * (0.6 + 0.8 * complement)
    score += 0.6 * distinct + 1.2 * new_types
    if setup:
        # First placements: value ore+wheat / wood+brick combos that unlock builds.
        wb = min(prod[B.WOOD], prod[B.BRICK]) * 36.0
        ow = min(prod[B.ORE], prod[B.WHEAT]) * 36.0
        score += 0.35 * (wb + ow)
    port = state.ports.get(v)
    if port is not None:
        if port == B.PORT_GENERIC:
            score += 1.0
        else:
            score += 0.5 + 6.0 * (own_prod[port] + prod[port])
    eocc = state.occupied_edges()
    score += 0.35 * _expansion_potential(state, v, occ, eocc)
    # Blockability: how much more of our income one robber placement could switch off.
    if block_ctx is None:
        block_ctx = BlockContext(state, player, scarcity)
    score -= PLACEMENT_BLOCK_WEIGHT * (block_ctx.exposure_with_settlement(v) - block_ctx.exposure_before)
    return score


def blocking_value(state: GameState, player: int, v: int, occ: Optional[Dict[int, int]] = None) -> float:
    """How much taking ``v`` hurts opponents (best opponent score for v if reachable within 1 road)."""
    occ = state.occupied_vertices() if occ is None else occ
    best = 0.0
    for i in range(state.num_players):
        if i == player:
            continue
        reach = reachable_spots(state, i, max_roads=1, occ=occ)
        if v in reach:
            s = score_settlement_spot(state, i, v, occ=occ)
            best = max(best, s / (1.0 + reach[v][0]))
    return best


def road_block_values(state: GameState, player: int, edges: Iterable[int],
                      occ: Optional[Dict[int, int]] = None) -> Dict[int, float]:
    """How much building a road on each of ``edges`` cuts the opponents off.

    For every edge the value is the largest drop (over opponents) of that
    opponent's best reachable settlement spot score (within two roads,
    distance discounted) when the edge is occupied by ``player``: an edge that
    severs an opponent's only path to their best spot scores that spot, an
    edge nobody else wanted scores 0.  Same scale as ``blocking_value``.
    """
    occ = state.occupied_vertices() if occ is None else occ
    eocc = state.occupied_edges()
    edges = [e for e in edges if e not in eocc]
    out: Dict[int, float] = {e: 0.0 for e in edges}
    if not edges:
        return out
    scarcity = None
    for i in range(state.num_players):
        if i == player:
            continue
        reach = reachable_spots(state, i, max_roads=2, occ=occ, eocc=eocc)
        if not reach:
            continue
        # Spot scores are computed once; taking an edge can only remove spots / lengthen paths.
        scarcity = resource_scarcity(state) if scarcity is None else scarcity
        own_prod = player_production(state, i, ignore_robber=True)
        bctx = BlockContext(state, i, scarcity)
        scores = {v: score_settlement_spot(state, i, v, occ=occ, own_prod=own_prod, scarcity=scarcity, block_ctx=bctx)
                  for v in reach}
        base = max(scores[v] / (1.0 + 0.9 * d) for v, (d, _) in reach.items())
        if base <= 0.0:
            continue
        # Only edges on the opponent's frontier (touching a vertex they can reach with at most
        # one road) can change their two-road reach; everything else is skipped without a BFS.
        near = set()
        start = set(state.players[i].settlements) | set(state.players[i].cities)
        for e in state.players[i].roads:
            start.update(B.EDGE_VERTICES[e])
        for v in start:
            near.add(v)
            for e in B.VERTEX_EDGES[v]:
                if e not in eocc:
                    a, b = B.EDGE_VERTICES[e]
                    near.add(b if a == v else a)
        for e in edges:
            a, b = B.EDGE_VERTICES[e]
            if a not in near and b not in near:
                continue
            eocc2 = dict(eocc)
            eocc2[e] = player
            reach2 = reachable_spots(state, i, max_roads=2, occ=occ, eocc=eocc2)
            after = max((scores.get(v, 0.0) / (1.0 + 0.9 * d) for v, (d, _) in reach2.items()), default=0.0)
            drop = base - after
            if drop > out[e]:
                out[e] = drop
    return out


def best_settlement_spots(state: GameState, player: int, k: int = 5,
                          candidates: Optional[Iterable[int]] = None,
                          setup: bool = False, include_blocking: bool = True) -> List[Tuple[int, float]]:
    """Top-k ``(vertex, score)`` among ``candidates`` (default: all free vertices)."""
    occ = state.occupied_vertices()
    if candidates is None:
        candidates = [v for v in range(B.NUM_VERTICES) if is_free_vertex(occ, v)]
    own_prod = player_production(state, player, ignore_robber=True)
    scarcity = resource_scarcity(state)
    bctx = BlockContext(state, player, scarcity)
    scored = []
    for v in candidates:
        s = score_settlement_spot(state, player, v, occ=occ, own_prod=own_prod, scarcity=scarcity, setup=setup,
                                  block_ctx=bctx)
        if include_blocking:
            s += 0.25 * blocking_value(state, player, v, occ=occ)
        scored.append((v, s))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def score_city(state: GameState, player: int, v: int) -> float:
    """Value of upgrading the settlement on ``v`` (extra production, demand-weighted).

    Minus the blockability penalty: the city doubles W(h) on every hex of ``v``,
    so a city next to our other buildings (or on a hex the leader also works)
    is charged for concentrating our income (:func:`block_penalty`).
    """
    prod = vertex_production(state, v, ignore_robber=True)
    scarcity = resource_scarcity(state)
    score = 0.0
    for r in range(5):
        score += prod[r] * 36.0 * RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5)
    # A city on a hex the robber currently sits on is slightly less attractive.
    if state.robber in B.VERTEX_HEXES[v]:
        score *= 0.85
    score -= block_penalty(state, player, extra_city=v, scarcity=scarcity)
    return score


def best_city_spots(state: GameState, player: int, k: int = 3) -> List[Tuple[int, float]]:
    p = state.players[player]
    scored = [(v, score_city(state, player, v)) for v in p.settlements]
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def road_targets(state: GameState, player: int, max_roads: int = 3, k: int = 5) -> List[dict]:
    """Best settlement spots reachable by road with the first edge to build.

    Returns dicts ``{"vertex", "roads", "first_edge", "score", "spot_score", "block"}``
    sorted by ``score`` (spot value discounted by distance and by the chance
    an opponent grabs it first, plus a bonus when the first road also cuts an
    opponent off from their best spot - ``block``, see :func:`road_block_values`).
    """
    occ = state.occupied_vertices()
    reach = reachable_spots(state, player, max_roads=max_roads, occ=occ)
    own_prod = player_production(state, player, ignore_robber=True)
    scarcity = resource_scarcity(state)
    blocks = road_block_values(state, player, {e for _, e in reach.values() if e >= 0}, occ=occ)
    bctx = BlockContext(state, player, scarcity)
    out = []
    for v, (d, e) in reach.items():
        spot = score_settlement_spot(state, player, v, occ=occ, own_prod=own_prod, scarcity=scarcity, block_ctx=bctx)
        contest = blocking_value(state, player, v, occ=occ)
        block = blocks.get(e, 0.0)
        score = (spot + 0.15 * contest) / (1.0 + 0.9 * d) + 0.15 * block
        out.append({"vertex": v, "roads": d, "first_edge": e, "score": score, "spot_score": spot, "block": block})
    out.sort(key=lambda t: -t["score"])
    return out[:k]


def setup_pick(state: GameState, player: int, k: int = 5) -> List[Tuple[int, float]]:
    """Ranked starting-settlement candidates for the setup phase."""
    return best_settlement_spots(state, player, k=k, setup=True)


def setup_road_pick(state: GameState, player: int, settlement: int) -> Tuple[int, float]:
    """Best starting road from ``settlement``: towards the best free spot two steps away."""
    occ = state.occupied_vertices()
    eocc = state.occupied_edges()
    own_prod = player_production(state, player, ignore_robber=True)
    scarcity = resource_scarcity(state)
    bctx = BlockContext(state, player, scarcity)
    best_e, best_s = -1, -1.0
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
            if is_free_vertex(occ, x) and x != settlement:
                s = max(s, score_settlement_spot(state, player, x, occ=occ, own_prod=own_prod, scarcity=scarcity,
                                                 block_ctx=bctx))
        if s > best_s:
            best_s, best_e = s, e
    if best_e == -1:
        for e in B.VERTEX_EDGES[settlement]:
            if e not in eocc:
                return e, 0.0
    return best_e, best_s
