"""Placement heuristics: settlement spots, cities and road targets.

These functions only depend on the board tables and the state; they are used
by the heuristic evaluator, by move ordering in the search and to produce
human readable explanations ("best spot: 10 pips, wheat/ore, gives a 2:1 ore
port").  Nothing here mutates the state.
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
                          setup: bool = False) -> float:
    """Heuristic value of placing a settlement on ``v`` for ``player``.

    Combines pips weighted by resource demand and board scarcity, diminishing
    returns on resources the player already produces, diversity, new
    resource types, port synergy and expansion potential.  Scale: roughly
    "pips-equivalents" (a 10-pip diverse spot scores ~12-16).
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
        scores = {v: score_settlement_spot(state, i, v, occ=occ, own_prod=own_prod, scarcity=scarcity)
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
    scored = []
    for v in candidates:
        s = score_settlement_spot(state, player, v, occ=occ, own_prod=own_prod, scarcity=scarcity, setup=setup)
        if include_blocking:
            s += 0.25 * blocking_value(state, player, v, occ=occ)
        scored.append((v, s))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def score_city(state: GameState, player: int, v: int) -> float:
    """Value of upgrading the settlement on ``v`` (extra production, demand-weighted)."""
    prod = vertex_production(state, v, ignore_robber=True)
    scarcity = resource_scarcity(state)
    score = 0.0
    for r in range(5):
        score += prod[r] * 36.0 * RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5)
    # A city on a hex the robber currently sits on is slightly less attractive.
    if state.robber in B.VERTEX_HEXES[v]:
        score *= 0.85
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
    out = []
    for v, (d, e) in reach.items():
        spot = score_settlement_spot(state, player, v, occ=occ, own_prod=own_prod, scarcity=scarcity)
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
                s = max(s, score_settlement_spot(state, player, x, occ=occ, own_prod=own_prod, scarcity=scarcity))
        if s > best_s:
            best_s, best_e = s, e
    if best_e == -1:
        for e in B.VERTEX_EDGES[settlement]:
            if e not in eocc:
                return e, 0.0
    return best_e, best_s
