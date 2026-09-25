"""Blockability term of the spot scorers (``placement.robber_exposure`` / ``block_penalty``).

Two of our buildings on one hex (or a city next to our settlement on it) can be
switched off by a single robber placement; ``score_settlement_spot`` and
``score_city`` charge that concentration, 1.5x on a hex an opponent with 5+ VP
(``robber.threat >= PLACEMENT_STRONG_THREAT``) also works.  Most boards below
are uniform (18 x wood 8, desert in a corner, no ports) so that every interior
vertex has the same 15 pips and the only differences between candidates are
expansion room and the blockability penalty.
"""
from __future__ import annotations

import contextlib
import random

from catanbot import board as B
from catanbot import placement as P
from catanbot.robber import threat
from catanbot.state import PHASE_MAIN, new_game

from tests.test_placement_counting import mid_game_state

CENTRE = 9  # the middle hex: its corners HEX_VERTICES[9] are pairwise two steps apart in cyclic order


@contextlib.contextmanager
def block_weight(w: float):
    """Temporarily set ``placement.PLACEMENT_BLOCK_WEIGHT`` (the scorers read it at call time)."""
    old = P.PLACEMENT_BLOCK_WEIGHT
    P.PLACEMENT_BLOCK_WEIGHT = w
    try:
        yield
    finally:
        P.PLACEMENT_BLOCK_WEIGHT = old


def uniform_board(num_players: int = 4):
    s = new_game(num_players, hexes=[(B.DESERT, 0)] + [(B.WOOD, 8)] * 18, ports={})
    s.phase = PHASE_MAIN
    return s


def _interior(v: int) -> bool:
    return len(B.VERTEX_HEXES[v]) == 3 and 0 not in B.VERTEX_HEXES[v]


def _hexes(*vs) -> set:
    return {h for v in vs for h in B.VERTEX_HEXES[v]}


def _far_vertex(occ, *avoid) -> int:
    """A free interior vertex sharing no hex with any of ``avoid``."""
    return next(v for v in range(B.NUM_VERTICES)
                if _interior(v) and P.is_free_vertex(occ, v) and not _hexes(v) & _hexes(*avoid))


# ---------------------------------------------------------------------------
# the previous scorers (verbatim), the reference for PLACEMENT_BLOCK_WEIGHT = 0
# ---------------------------------------------------------------------------
def _old_score_settlement_spot(state, player, v, setup=False):
    occ = state.occupied_vertices()
    own_prod = P.player_production(state, player, ignore_robber=True)
    scarcity = P.resource_scarcity(state)
    prod = P.vertex_production(state, v, ignore_robber=True)
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
        complement = 1.0 / (1.0 + 4.0 * have)
        score += prod[r] * 36.0 * P.RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5) * (0.6 + 0.8 * complement)
    score += 0.6 * distinct + 1.2 * new_types
    if setup:
        wb = min(prod[B.WOOD], prod[B.BRICK]) * 36.0
        ow = min(prod[B.ORE], prod[B.WHEAT]) * 36.0
        score += 0.35 * (wb + ow)
    port = state.ports.get(v)
    if port is not None:
        if port == B.PORT_GENERIC:
            score += 1.0
        else:
            score += 0.5 + 6.0 * (own_prod[port] + prod[port])
    score += 0.35 * P._expansion_potential(state, v, occ, state.occupied_edges())
    return score


def _old_score_city(state, player, v):
    prod = P.vertex_production(state, v, ignore_robber=True)
    scarcity = P.resource_scarcity(state)
    score = 0.0
    for r in range(5):
        score += prod[r] * 36.0 * P.RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5)
    if state.robber in B.VERTEX_HEXES[v]:
        score *= 0.85
    return score


def _random_layouts(n: int = 12):
    """Random building layouts (stacked hexes, cities, strong opponents) on random boards."""
    rng = random.Random(3)
    out = []
    for _ in range(n):
        s = new_game(rng.choice([3, 4]), rng=rng)
        s.phase = PHASE_MAIN
        occ = {}
        for i in range(s.num_players):
            p = s.players[i]
            h0 = rng.randrange(B.NUM_HEXES)
            cands = [v for h in [h0] + B.HEX_NEIGHBORS[h0] for v in B.HEX_VERTICES[h]]
            rng.shuffle(cands)
            for v in cands:
                if len(p.settlements) + len(p.cities) >= rng.randint(1, 5):
                    break
                if P.is_free_vertex(occ, v):
                    (p.cities if rng.random() < 0.4 else p.settlements).append(v)
                    occ[v] = i
            p.roads = [B.VERTEX_EDGES[v][0] for v in p.settlements + p.cities][:3]
        if rng.random() < 0.5:
            s.longest_road_owner = rng.randrange(s.num_players)
            s.longest_road_len = 5
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def test_first_building_has_no_penalty():
    for seed in range(3):
        s = new_game(4, rng=random.Random(seed))
        for i in range(4):
            assert P.robber_exposure(s, i) == 0.0
        vs = list(range(0, B.NUM_VERTICES, 3))
        assert all(P.block_penalty(s, 0, extra_settlement=v) == 0.0 for v in vs)
        new = [P.score_settlement_spot(s, 0, v, setup=setup) for v in vs for setup in (False, True)]
        with block_weight(0.0):
            old = [P.score_settlement_spot(s, 0, v, setup=setup) for v in vs for setup in (False, True)]
        assert new == old
        # a lone building anywhere - even on the leader's hex - is not concentration either
        s.players[1].settlements = [B.HEX_VERTICES[CENTRE][0]]
        s.players[1].cities = [B.HEX_VERTICES[CENTRE][2], B.HEX_VERTICES[CENTRE][4]]
        assert threat(s, 1) >= P.PLACEMENT_STRONG_THREAT
        occ = s.occupied_vertices()
        free = [v for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)]
        assert all(P.block_penalty(s, 0, extra_settlement=v) == 0.0 for v in free)


def test_stacking_two_settlements_scores_below_spreading():
    s = uniform_board()
    v1, v_stack = B.HEX_VERTICES[CENTRE][0], B.HEX_VERTICES[CENTRE][2]  # share the centre hex only
    s.players[0].settlements = [v1]
    occ = s.occupied_vertices()
    assert P.is_free_vertex(occ, v_stack) and len(_hexes(v1) & _hexes(v_stack)) == 1
    v_spread = _far_vertex(occ, v1)
    assert P.vertex_pips(s, v_stack) == P.vertex_pips(s, v_spread) == 15
    pen_stack = P.block_penalty(s, 0, extra_settlement=v_stack)
    pen_spread = P.block_penalty(s, 0, extra_settlement=v_spread)
    new_stack, new_spread = P.score_settlement_spot(s, 0, v_stack), P.score_settlement_spot(s, 0, v_spread)
    with block_weight(0.0):
        old_stack, old_spread = P.score_settlement_spot(s, 0, v_stack), P.score_settlement_spot(s, 0, v_spread)
    assert pen_spread == 0.0 and new_spread == old_spread
    assert pen_stack > 0.0 and abs((old_stack - new_stack) - pen_stack) < 1e-9
    assert new_stack < new_spread
    # the penalty is a real fraction of the hex: between 1 and 3 pips of the production term
    per_pip = (old_stack - 0.6 - 0.35 * P._expansion_potential(s, v_stack, occ, {})) / 15.0
    assert 1.0 * per_pip < pen_stack < 3.0 * per_pip
    # exposure grows with stacking: three buildings on the hex, or a city, beat two settlements
    e2 = P.robber_exposure(s, 0, extra_settlement=v_stack)
    s.players[0].settlements.append(v_stack)
    assert abs(P.robber_exposure(s, 0) - e2) < 1e-12
    assert P.robber_exposure(s, 0, extra_settlement=B.HEX_VERTICES[CENTRE][4]) > e2
    assert P.robber_exposure(s, 0, extra_city=v1) > e2
    # ranking: best_settlement_spots puts the spread spot ahead of the stacked one
    s.players[0].settlements = [v1]
    ranked = dict(P.best_settlement_spots(s, 0, k=60, candidates=[v_stack, v_spread], include_blocking=False))
    assert ranked[v_spread] > ranked[v_stack]


def test_city_next_to_our_other_building_is_penalised_more():
    s = uniform_board()
    v1, v2 = B.HEX_VERTICES[CENTRE][0], B.HEX_VERTICES[CENTRE][2]  # share the centre hex
    v3 = _far_vertex({v1: 0, v2: 0}, v1, v2)                      # alone on its three hexes
    s.players[0].settlements = [v1, v2, v3]
    assert P.vertex_pips(s, v1) == P.vertex_pips(s, v3) == 15
    pen_shared, pen_alone = P.block_penalty(s, 0, extra_city=v1), P.block_penalty(s, 0, extra_city=v3)
    assert pen_shared > 0.0 and pen_shared > pen_alone
    assert pen_alone <= 0.0  # a lone city concentrates nothing (it only dilutes the stacked hex)
    new_shared, new_alone = P.score_city(s, 0, v1), P.score_city(s, 0, v3)
    with block_weight(0.0):
        old_shared, old_alone = P.score_city(s, 0, v1), P.score_city(s, 0, v3)
    assert old_shared == old_alone  # equal pips, same resources, robber elsewhere
    assert new_shared < new_alone
    assert abs((old_shared - new_shared) - pen_shared) < 1e-9 and abs((old_alone - new_alone) - pen_alone) < 1e-9
    assert [v for v, _ in P.best_city_spots(s, 0, k=3)][0] == v3
    # extra_city on one of our settlements is its upgrade: hexes of v1 count twice, not three times
    e_upgrade = P.robber_exposure(s, 0, extra_city=v1)
    s.players[0].settlements.remove(v1)
    s.players[0].cities.append(v1)
    assert abs(P.robber_exposure(s, 0) - e_upgrade) < 1e-12


def test_hex_shared_with_the_leader_is_penalised_more():
    base = uniform_board()
    v1, v_stack, v_opp = (B.HEX_VERTICES[CENTRE][k] for k in (0, 2, 4))
    base.players[0].settlements = [v1]
    occ = {v1: 0, v_opp: 1, v_stack: 0}  # v_stack stays free for the candidate
    far = []                            # two interior vertices away from the stacked hexes, distance rule respected
    for v in range(B.NUM_VERTICES):
        if len(far) < 2 and _interior(v) and P.is_free_vertex(occ, v) and not _hexes(v) & _hexes(v1, v_stack):
            far.append(v)
            occ[v] = 1
    assert len(far) == 2
    weak = base.copy()
    weak.players[1].settlements = [v_opp, far[0]]                       # 2 VP
    strong = base.copy()
    strong.players[1].settlements = [v_opp]
    strong.players[1].cities = list(far)                                # 5 VP ...
    strong.longest_road_owner, strong.longest_road_len = 1, 5           # ... + 2 = 7 VP
    assert threat(weak, 1) < P.PLACEMENT_STRONG_THREAT <= threat(strong, 1)
    assert P.strong_opponent_hexes(weak, 0) == [0] * B.NUM_HEXES
    assert P.strong_opponent_hexes(strong, 0)[CENTRE] == 1
    pen_weak = P.block_penalty(weak, 0, extra_settlement=v_stack)
    pen_strong = P.block_penalty(strong, 0, extra_settlement=v_stack)
    assert 0.0 < pen_weak < pen_strong
    assert abs(pen_strong / pen_weak - 1.5) < 1e-9  # only the shared hex is stacked: the 1 + 0.5 factor
    for st, pen in ((weak, pen_weak), (strong, pen_strong)):
        new = P.score_settlement_spot(st, 0, v_stack)
        with block_weight(0.0):
            old = P.score_settlement_spot(st, 0, v_stack)
        assert abs((old - new) - pen) < 1e-9
    # the factor is about stacking on the leader's hex: a lone spot next to the leader is still free
    occ = strong.occupied_vertices()
    v_lone = next(v for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)
                  and _hexes(v) & _hexes(v_opp) and not _hexes(v) & _hexes(v1))
    assert P.block_penalty(strong, 0, extra_settlement=v_lone) == 0.0
    # a city on the shared hex (a third building there) is charged 1.5x as well
    strong.players[0].settlements.append(v_stack)
    weak.players[0].settlements.append(v_stack)
    city_weak, city_strong = P.block_penalty(weak, 0, extra_city=v1), P.block_penalty(strong, 0, extra_city=v1)
    assert 0.0 < city_weak < city_strong and abs(city_strong / city_weak - 1.5) < 1e-9


def test_penalty_is_a_pure_function_of_the_state():
    for s in _random_layouts(4):
        before = s.to_dict()
        for i in range(s.num_players):
            occ = s.occupied_vertices()
            vs = [v for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)][:6]
            a = [P.block_penalty(s, i, extra_settlement=v) for v in vs]
            b = [P.block_penalty(s, i, extra_settlement=v) for v in vs]
            assert a == b
            ctx = P.BlockContext(s, i)
            assert [ctx.settlement_penalty(v) for v in vs] == a
            assert all(P.score_city(s, i, v) == P.score_city(s, i, v) for v in s.players[i].settlements)
        assert s.to_dict() == before


# ---------------------------------------------------------------------------
# wiring: weight 0 restores the old scores exactly; setup / city pickers still work
# ---------------------------------------------------------------------------
def test_zero_weight_restores_the_previous_scores_exactly():
    states = _random_layouts(10) + [mid_game_state()]
    checked = 0
    with block_weight(0.0):
        for s in states:
            occ = s.occupied_vertices()
            for i in range(s.num_players):
                for v in [v for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)][::4]:
                    for setup in (False, True):
                        assert P.score_settlement_spot(s, i, v, setup=setup) == _old_score_settlement_spot(s, i, v, setup)
                        checked += 1
                for v in s.players[i].settlements:
                    assert P.score_city(s, i, v) == _old_score_city(s, i, v)
    assert checked > 200
    # with the weight on, the scores differ from the old ones by exactly the penalty (and by nothing else)
    for s in states:
        occ = s.occupied_vertices()
        for i in range(s.num_players):
            for v in [v for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)][::7]:
                got = P.score_settlement_spot(s, i, v)
                ref = _old_score_settlement_spot(s, i, v) - P.block_penalty(s, i, extra_settlement=v)
                assert abs(got - ref) < 1e-9
            for v in s.players[i].settlements:
                ref = _old_score_city(s, i, v) - P.block_penalty(s, i, extra_city=v)
                assert abs(P.score_city(s, i, v) - ref) < 1e-9
    # somewhere the term is active (a stacked hex or a city next to a settlement exists in the layouts)
    assert any(P.robber_exposure(s, i) > 0.0 for s in states for i in range(s.num_players))


def test_setup_pick_and_city_spots_stay_valid():
    for seed in (1, 2, 3):
        s = new_game(4, rng=random.Random(seed))
        occ = {}
        for i in range(4):  # a first round of settlements
            v = next(v for v in range(seed * 7, B.NUM_VERTICES) if P.is_free_vertex(occ, v))
            s.players[i].settlements.append(v)
            occ[v] = i
        for i in range(4):
            picks = P.setup_pick(s, i, k=5)
            assert len(picks) == 5
            assert all(P.is_free_vertex(occ, v) for v, _ in picks)
            assert all(picks[k][1] >= picks[k + 1][1] for k in range(4))
            e, _ = P.setup_road_pick(s, i, s.players[i].settlements[0])
            assert s.players[i].settlements[0] in B.EDGE_VERTICES[e]
    s = mid_game_state()
    cities = P.best_city_spots(s, 0, k=3)
    assert [v for v, _ in cities] == s.players[0].settlements[:3] or len(cities) == len(s.players[0].settlements)
    assert P.road_targets(s, 0, k=3)


def test_degenerate_boards_and_threat_boundary():
    """No division by zero when every W(h) is 0 (all-desert board, players without buildings, desert-only
    vertices); the strong-opponent threshold includes exactly 5 estimated VP (threat == 1.3)."""
    d = new_game(3, hexes=[(B.DESERT, 0)] * 19, ports={})
    d.phase = PHASE_MAIN
    d.players[0].settlements = [B.HEX_VERTICES[9][0], B.HEX_VERTICES[9][2]]  # stacked, but on deserts
    d.players[0].cities = [B.HEX_VERTICES[9][4]]
    assert P.hex_block_weights(d) == [0.0] * B.NUM_HEXES
    assert P.robber_exposure(d, 0) == 0.0 and P.robber_exposure(d, 1) == 0.0
    occ = d.occupied_vertices()
    for v in [v for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)][:10]:
        assert P.block_penalty(d, 0, extra_settlement=v) == 0.0
        assert P.score_settlement_spot(d, 0, v) == P.score_settlement_spot(d, 1, v)  # finite, no buildings vs some
    assert P.block_penalty(d, 0, extra_city=d.players[0].settlements[0]) == 0.0
    # standard board: the desert hex weighs 0 and a lone desert-corner settlement adds nothing on it
    s = new_game(4, hexes=B.STANDARD_HEXES)
    s.phase = PHASE_MAIN
    assert P.hex_block_weights(s)[s.robber] == 0.0 and s.hexes[s.robber][0] == B.DESERT
    assert P.block_penalty(s, 0, extra_settlement=B.HEX_VERTICES[s.robber][0]) == 0.0
    # exactly 5 VP (threat 1.0 + 0.3 == PLACEMENT_STRONG_THREAT) is a strong opponent, 4 VP is not;
    # stacked on the ore 8 (hex 11; the standard board's centre hex 9 is the desert and weighs 0)
    ore8 = 11
    assert s.hexes[ore8] == (B.ORE, 8)
    v1, v_stack, v_opp = (B.HEX_VERTICES[ore8][k] for k in (0, 2, 4))
    far = [44, 47]  # two corners of the brick 5, not adjacent to each other
    assert far[1] not in B.VERTEX_NEIGHBORS[far[0]] and not _hexes(*far) & _hexes(v1, v_stack)
    s.players[0].settlements = [v1]
    s.players[1].settlements = [v_opp]
    s.players[1].cities = far
    assert threat(s, 1) == P.PLACEMENT_STRONG_THREAT
    assert P.strong_opponent_hexes(s, 0)[ore8] == 1
    pen5 = P.block_penalty(s, 0, extra_settlement=v_stack)
    s.players[1].cities = far[:1]
    assert threat(s, 1) < P.PLACEMENT_STRONG_THREAT
    assert P.strong_opponent_hexes(s, 0)[ore8] == 0
    pen3 = P.block_penalty(s, 0, extra_settlement=v_stack)
    assert pen3 > 0.0 and abs(pen5 / pen3 - 1.5) < 1e-9
