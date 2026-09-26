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
# next to our settlement on it) concentrate our income where a single robber
# placement - and the robber goes to the juiciest hex of its victim - blocks
# all of it.  The spot scorers therefore charge a candidate building for the
# *increase* of our robber exposure, on the same pips-equivalent scale as
# their production term:
#
#   W_b(h)     = demand-weighted pips of our building b on hex h (settlement = pips,
#                city = 2 x pips, times RESOURCE_DEMAND[res] x scarcity[res] ** 0.5;
#                the robber's current position is ignored - the term is about
#                where it *can* go)
#   W(h)       = sum_b W_b(h)                     our weight on the hex
#   P_block(h) = PLACEMENT_ROBBER_Q x W(h)^2 / sum_h' W(h')^2 x (1 + 0.5 x shared_strong(h))
#   stack(h)   = 1 - sum_b W_b(h)^3 / W(h)^3     the share of the hex's weight that is
#                                                 there because buildings share it
#                                                 (0 for one building, 0.75 for two equal ones)
#   exposure   = sum_h P_block(h) x W(h) x stack(h)
#   penalty    = PLACEMENT_BLOCK_WEIGHT x (exposure_after - exposure_before)
#
# The square in P_block models opponents aiming the robber at our best hex, so
# concentration is costly; stack(h) measures the exposure *in excess of* the
# same buildings standing on separate hexes, so a first building - and any
# layout without shared hexes - has exposure 0 and the scorers never trade
# raw pips against it, only stacking.  shared_strong(h) = 1 when an opponent
# with robber.threat >= PLACEMENT_STRONG_THREAT (5+ estimated VP) has a
# building on h: the robber visits the hexes of strong players anyway, so
# stacking there is blocked 1.5x as often.  Worked numbers: docs/STRATEGY.md
# (Placement).  cpp/heuristic.cpp mirrors every function below bit for bit
# (score_spot / BlockContext).
PLACEMENT_ROBBER_Q = 0.35
"""Fraction of the time the robber sits on one of *our* hexes when we are an
ordinary target (nobody singles us out).  Tunable; the penalty scales with it."""

PLACEMENT_BLOCK_WEIGHT = 1.0
"""Weight of the blockability penalty on the pips-equivalent scale of
``score_settlement_spot`` / ``score_city``.  With 1.0 a second settlement on
the brick 6 our first settlement already works costs 1.9-2.0 points (~1.9 pips
of production value) against the same pips on hexes we do not work; set to 0
to switch the term off (the scores are then exactly the old ones)."""

PLACEMENT_STRONG_THREAT = 1.3
"""An opponent whose ``robber.threat`` is at least this (5+ estimated VP: the
threat is 1 + 0.3 x (VP - 4)) attracts the robber to their hexes regardless of
us; stacking on a hex we share with such a player counts 1.5x.  An absolute
threshold (not "the strongest at the table") so that early in the game, when
nobody is a robber magnet yet, only concentration is charged."""


# ---------------------------------------------------------------------------
# Ports (docs/PRIORITY_PLAN.md step 4, area "ports"; docs/STRATEGY.md "Ports")
# ---------------------------------------------------------------------------
# The spot score's port bonus.  Tunables (catanbot/portvalue.py registers them), all needs_python_evaluator:
# cpp/heuristic.cpp (score_spot) keeps constexpr copies of these defaults, so an override reaches the C++ static
# value only after an ADOPT hard-codes it there.  The defaults are today's literals (byte-identical).
PORT_SPOT_GENERIC = 1.0        # a 3:1 spot: flat bonus
PORT_SPOT_2TO1_BASE = 0.5      # a 2:1 spot on t: base + slope x (our production of t + the spot's own)
PORT_SPOT_2TO1_SLOPE = 6.0
# ports.constants (F2): 1 = a generic 3:1 port counts once - static's +PORT_STATIC_GENERIC once however many of our
# buildings stand on 3:1 ports, and a 3:1 spot's bonus is 0 when we already own a 3:1 (every ratio <= 3).
PORT_GENERIC_ONCE = 0
# ports.surplus_value (F1): 0 = the constants above (default); 1 = the calibrated conversion model in the spot score
# and in static_value's port term; 2 = static only; 3 = 3:1 only (spot and static; the 2:1 constants are kept).
# heuristic.static_value reads PORT_MODEL / PORT_GENERIC_ONCE through this module (``_pl.PORT_MODEL``), never
# through a from-import, which would freeze the value and make an override a silent A/A.
PORT_MODEL = 0
PORT_A0 = 0.08                 # F1: cards per roll converted at 4:1 whatever the surplus (the fitted intercept) ...
PORT_B = 0.75                  # ... plus this share of the surplus (fit on our no-port rolls: T1 0.060 + 0.87 x
#                                model, R1 0.116 + 0.63 x model)
PORT_ELAST = 0.25              # F1: half credit for the extra conversion at a cheaper ratio (x1.19 at 3:1, x1.43 at 2:1)
PORT_SPOT_W = 1.0              # F1: weight of the calibrated spot bonus (pips-equivalents x the complement factor)
PORT_STATIC_W = 0.8            # F1: static points per pips-equivalent (0.55 + 0.25: static's weight of one pip)


def ratios_with(rho: Sequence[int], port: int) -> List[int]:
    """Bank ratios ``rho`` after adding a port of type ``port`` (``B.PORT_GENERIC`` or a resource)."""
    if port == B.PORT_GENERIC:
        return [min(x, 3) for x in rho]
    out = list(rho)
    out[port] = 2
    return out


def ratios_of(state: GameState, buildings: Iterable[int], generic: bool = True) -> List[int]:
    """Bank ratios given by the ports under ``buildings`` (``generic=False``: the 2:1 ports only)."""
    rho = [4] * 5
    for v in buildings:
        t = state.ports.get(v)
        if t is None or (t == B.PORT_GENERIC and not generic):
            continue
        rho = ratios_with(rho, t)
    return rho


def port_ratios(state: GameState, player: int, extra_port: Optional[int] = None) -> List[int]:
    """``state.port_ratio(player, r)`` for every resource, with a port of type ``extra_port`` added."""
    p = state.players[player]
    rho = ratios_of(state, p.settlements + p.cities)
    return rho if extra_port is None else ratios_with(rho, extra_port)


def owns_generic_port(state: GameState, player: int) -> bool:
    p = state.players[player]
    return any(state.ports.get(v) == B.PORT_GENERIC for v in p.settlements + p.cities)


def demand_shares() -> List[float]:
    """``D_r``: RESOURCE_DEMAND as shares (RESOURCE_DEMAND / 5 at mean 1: 0.187 / 0.187 / 0.168 / 0.234 / 0.224)."""
    tot = sum(RESOURCE_DEMAND)
    return [d / tot for d in RESOURCE_DEMAND]


def port_conversion(prod: Sequence[float], a0: Optional[float] = None, b: Optional[float] = None) -> List[float]:
    """F1 ``conv_r = A0 prod_r / I + B s_r``: cards of ``r`` per roll a seat with robber-free production ``prod``
    (cards per roll) gives to the bank, ``s_r = max(0, prod_r - D_r I)`` its surplus and ``I = sum prod``.  The
    intercept is the lumpy conversion of a low-surplus economy (our bot converts 0.06-0.12 cards per roll even so)."""
    a0 = PORT_A0 if a0 is None else a0
    b = PORT_B if b is None else b
    inc = sum(prod)
    if inc <= 0.0:
        return [0.0] * 5
    D = demand_shares()
    return [a0 * prod[r] / inc + b * max(0.0, prod[r] - D[r] * inc) for r in range(5)]


def port_gain(prod: Sequence[float], rho: Sequence[int], rho2: Sequence[int], elast: Optional[float] = None,
              conv: Optional[Sequence[float]] = None) -> float:
    """F1 ``G = sum_r conv_r (1/rho2_r - 1/rho_r) (1 + ELAST (4/rho2_r - 1))``: extra needed cards per roll that
    the ratios ``rho2`` buy over ``rho`` from the same conversions (0 when no ratio improves: a second 3:1, a 3:1
    on top of a 2:1 that already covers the surplus)."""
    e = PORT_ELAST if elast is None else elast
    conv = port_conversion(prod) if conv is None else conv
    g = 0.0
    for r in range(5):
        if rho2[r] < rho[r]:
            g += conv[r] * (1.0 / rho2[r] - 1.0 / rho[r]) * (1.0 + e * (4.0 / rho2[r] - 1.0))
    return g


def port_need_weights(prod: Sequence[float], scarcity: Sequence[float]) -> Tuple[float, float]:
    """F1 ``(w, c)`` over the deficits ``d_r = max(0, D_r I - prod_r)``: ``w`` the demand x sqrt(scarcity) value of
    a needed card (1 without a deficit) and ``c`` their mean complement factor ``1 / (1 + 4 prod_r)`` (0.5)."""
    inc = sum(prod)
    D = demand_shares()
    d = [max(0.0, D[r] * inc - prod[r]) for r in range(5)]
    tot = sum(d)
    if tot <= 0.0:
        return 1.0, 0.5
    w = sum(d[r] * RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5) for r in range(5)) / tot
    c = sum(d[r] / (1.0 + 4.0 * prod[r]) for r in range(5)) / tot
    return w, c


def port_pe(prod: Sequence[float], rho: Sequence[int], rho2: Sequence[int],
            scarcity: Sequence[float]) -> Tuple[float, float]:
    """F1 ``(PE, c)``: the pips-equivalent ``36 G w`` of going from ratios ``rho`` to ``rho2`` for the production
    ``prod``, and the complement factor ``c`` of :func:`port_need_weights`."""
    g = port_gain(prod, rho, rho2)
    if g <= 0.0:
        return 0.0, 0.5
    w, c = port_need_weights(prod, scarcity)
    return 36.0 * g * w, c


def spot_port_bonus(state: GameState, player: int, v: int, own_prod: Optional[Sequence[float]] = None,
                    prod_v: Optional[Sequence[float]] = None, scarcity: Optional[Sequence[float]] = None) -> float:
    """The port part of :func:`score_settlement_spot` for ``player`` settling ``v`` (0 without a port).

    Default: ``PORT_SPOT_GENERIC`` for a 3:1 spot, ``PORT_SPOT_2TO1_BASE + PORT_SPOT_2TO1_SLOPE x (own[t] +
    prod_v[t])`` for a 2:1 ``t`` spot (today's 1.0 / 0.5 + 6 x, bit for bit); ``PORT_GENERIC_ONCE``: a 3:1 spot is
    worth 0 when ``player`` already owns a 3:1.  ``PORT_MODEL`` 1 (every spot) / 3 (3:1 spots): ``PORT_SPOT_W x
    PE(own + prod_v; rho_now -> rho_with_v) x (0.6 + 0.8 c)``, the calibrated card value in the units of land pips
    (the same complement factor), so the argmax over spots decides "port vs land" directly."""
    port = state.ports.get(v)
    if port is None:
        return 0.0
    own = player_production(state, player, ignore_robber=True) if own_prod is None else own_prod
    pv = vertex_production(state, v, ignore_robber=True) if prod_v is None else prod_v
    model = PORT_MODEL
    if model == 1 or (model == 3 and port == B.PORT_GENERIC):
        rho = port_ratios(state, player)
        prod = [own[r] + pv[r] for r in range(5)]
        sc = resource_scarcity(state) if scarcity is None else scarcity
        pe, c = port_pe(prod, rho, ratios_with(rho, port), sc)
        return PORT_SPOT_W * pe * (0.6 + 0.8 * c)
    if port == B.PORT_GENERIC:
        if PORT_GENERIC_ONCE and owns_generic_port(state, player):
            return 0.0
        return PORT_SPOT_GENERIC
    return PORT_SPOT_2TO1_BASE + PORT_SPOT_2TO1_SLOPE * (own[port] + pv[port])


def hex_block_weights(state: GameState, scarcity: Optional[Sequence[float]] = None) -> List[float]:
    """Per hex: pips x RESOURCE_DEMAND x scarcity ** 0.5 (0 for the desert) - a settlement's W_b(h)."""
    scarcity = resource_scarcity(state) if scarcity is None else scarcity
    res_w = [RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5) for r in range(5)]
    out = [0.0] * B.NUM_HEXES
    for h in range(B.NUM_HEXES):
        res, num = state.hexes[h]
        if res == B.DESERT or num == 0:
            continue
        out[h] = B.PIPS[num] * res_w[res]
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


def _stack_weights(state: GameState, player: int, hex_w: Sequence[float],
                   extra_settlement: Optional[int] = None,
                   extra_city: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """Per hex: ``W(h)`` (our buildings' weights summed) and ``K(h)`` (the same weights cubed and summed).

    Settlements, then cities, then the extras; a city weighs twice a settlement.
    ``extra_city`` equal to one of our settlements is the upgrade of that settlement.
    """
    w = [0.0] * B.NUM_HEXES
    k = [0.0] * B.NUM_HEXES
    p = state.players[player]
    for v in p.settlements:
        if v == extra_city:
            continue
        for h in B.VERTEX_HEXES[v]:
            c = hex_w[h]
            w[h] += c
            k[h] += c * c * c
    for v in p.cities:
        for h in B.VERTEX_HEXES[v]:
            c = 2.0 * hex_w[h]
            w[h] += c
            k[h] += c * c * c
    if extra_settlement is not None:
        for h in B.VERTEX_HEXES[extra_settlement]:
            c = hex_w[h]
            w[h] += c
            k[h] += c * c * c
    if extra_city is not None:
        for h in B.VERTEX_HEXES[extra_city]:
            c = 2.0 * hex_w[h]
            w[h] += c
            k[h] += c * c * c
    return w, k


def _exposure_of(w: Sequence[float], k: Sequence[float], strong: Sequence[int]) -> float:
    """``sum_h P_block(h) x W(h) x stack(h)`` = ``q x sum_h (1 + 0.5 strong) (W^3 - K) / sum W^2``."""
    ssq = 0.0
    num = 0.0
    for h in range(B.NUM_HEXES):
        x = w[h]
        if x <= 0.0:
            continue
        ssq += x * x
        # W^3 - K is 0 for a hex with a single building and grows with stacking.
        num += (x * x * x - k[h]) * (1.0 + 0.5 * strong[h])
    if ssq <= 0.0:
        return 0.0
    return PLACEMENT_ROBBER_Q * num / ssq


def robber_exposure(state: GameState, player: int, extra_settlement: Optional[int] = None,
                    extra_city: Optional[int] = None, scarcity: Optional[Sequence[float]] = None,
                    hex_w: Optional[Sequence[float]] = None, strong: Optional[Sequence[int]] = None) -> float:
    """Demand-weighted pips of ``player`` one robber placement blocks *because* buildings share hexes.

    See the module comment for the model; it is 0 for a layout in which no two
    of our buildings share a hex.  ``extra_settlement`` / ``extra_city``
    evaluate the position *after* that building is added (``extra_city`` on
    one of our settlements = its upgrade).  Pure function of the state,
    O(#our buildings x 3 + #opponent buildings x 3).
    """
    hex_w = hex_block_weights(state, scarcity) if hex_w is None else hex_w
    strong = strong_opponent_hexes(state, player) if strong is None else strong
    w, k = _stack_weights(state, player, hex_w, extra_settlement, extra_city)
    return _exposure_of(w, k, strong)


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

    ``settlement_penalty(v)`` equals ``block_penalty(state, player, extra_settlement=v)`` computed from
    the cached weights; callers scoring many candidates build one context and pass it as ``block_ctx``.
    """

    __slots__ = ("hex_w", "strong", "base_w", "base_k", "exposure_before")

    def __init__(self, state: GameState, player: int, scarcity: Optional[Sequence[float]] = None):
        self.hex_w = hex_block_weights(state, scarcity)
        self.strong = strong_opponent_hexes(state, player)
        self.base_w, self.base_k = _stack_weights(state, player, self.hex_w)
        self.exposure_before = _exposure_of(self.base_w, self.base_k, self.strong)

    def exposure_with_settlement(self, v: int) -> float:
        w = list(self.base_w)
        k = list(self.base_k)
        for h in B.VERTEX_HEXES[v]:
            c = self.hex_w[h]
            w[h] += c
            k[h] += c * c * c
        return _exposure_of(w, k, self.strong)

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
    if v in state.ports:
        # the port bonus (spot_port_bonus: today's constants by default; ports.constants / ports.surplus_value)
        score += spot_port_bonus(state, player, v, own_prod, prod, scarcity)
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
