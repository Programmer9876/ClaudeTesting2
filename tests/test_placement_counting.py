import random

from catanbot import board as B
from catanbot import counting as C
from catanbot import inference as I
from catanbot import placement as P
from catanbot.state import PHASE_MAIN, new_game


def mid_game_state():
    s = new_game(4, hexes=B.STANDARD_HEXES)
    occ = {}

    def place(pi, v, city=False):
        assert P.is_free_vertex(occ, v), v
        (s.players[pi].cities if city else s.players[pi].settlements).append(v)
        occ[v] = pi

    place(0, B.HEX_VERTICES[9][0])          # desert top corner: touches hexes 4/5 too
    place(0, B.HEX_VERTICES[16][3], True)
    place(1, B.HEX_VERTICES[2][1])
    place(2, B.HEX_VERTICES[12][4])
    place(3, B.HEX_VERTICES[6][2])
    for pi in range(4):
        v = (s.players[pi].settlements or s.players[pi].cities)[0]
        s.players[pi].roads.append(B.VERTEX_EDGES[v][0])
    s.phase = PHASE_MAIN
    s.players[0].resources = [2, 1, 0, 3, 1]
    for pi in (1, 2, 3):
        p = s.players[pi]
        p.hand_known = False
        p.hand_size = 5
        p.dev_known = False
        p.dev_count = 1
    s.players[2].played_knights = 2
    # keep the bank consistent with the known hand
    for r in range(5):
        s.bank[r] -= s.players[0].resources[r]
    return s


def test_vertex_production_and_robber():
    s = new_game(4, hexes=B.STANDARD_HEXES)
    v = B.HEX_VERTICES[4][0]  # brick 6 hex top corner
    prod = P.vertex_production(s, v, ignore_robber=True)
    assert prod[B.BRICK] >= B.PIPS[6] / 36.0
    s.robber = 4
    prod_r = P.vertex_production(s, v)
    assert prod_r[B.BRICK] < prod[B.BRICK]


def test_city_doubles_production():
    s = mid_game_state()
    v = s.players[0].cities[0]
    single = P.vertex_production(s, v, ignore_robber=True)
    total = P.player_production(s, 0, ignore_robber=True)
    for r in range(5):
        assert total[r] >= 2 * single[r] - 1e-9


def test_best_spots_respect_distance_rule():
    s = mid_game_state()
    occ = s.occupied_vertices()
    spots = P.best_settlement_spots(s, 0, k=10)
    assert spots and all(P.is_free_vertex(occ, v) for v, _ in spots)
    # sorted descending
    assert all(spots[i][1] >= spots[i + 1][1] for i in range(len(spots) - 1))


def test_reachable_and_road_targets():
    s = mid_game_state()
    reach = P.reachable_spots(s, 0, max_roads=2)
    occ = s.occupied_vertices()
    assert all(P.is_free_vertex(occ, v) for v in reach)
    assert all(d >= 0 for d, _ in reach.values())
    targets = P.road_targets(s, 0, max_roads=3, k=3)
    assert targets and targets[0]["first_edge"] >= 0 or targets[0]["roads"] == 0
    e = targets[0]["first_edge"]
    if e >= 0:
        assert e not in s.occupied_edges()


def test_setup_road_touches_settlement():
    s = mid_game_state()
    v = s.players[0].settlements[0]
    e, _ = P.setup_road_pick(s, 0, v)
    assert v in B.EDGE_VERTICES[e]


def test_counting_priors_and_pool():
    s = mid_game_state()
    hands = C.expected_opponent_hands(s, me=0)
    assert hands[0] == [2.0, 1.0, 0.0, 3.0, 1.0]
    for i in (1, 2, 3):
        assert abs(sum(hands[i]) - 5.0) < 1e-6
    pool = C.dev_pool(s)
    assert pool[B.DEV_KNIGHT] == 12  # 14 - 2 played
    assert sum(pool) == 23
    assert 0 < C.expected_hidden_vp(s, 2) < 1


def test_hand_belief_updates():
    s = mid_game_state()
    b = C.HandBelief(s)
    b.observe_gain(1, B.WHEAT, 2)
    assert b.size[1] == 7
    b.observe_spend(1, (0, 0, 0, 1, 0))
    assert b.size[1] == 6 and abs(sum(b.expected[1]) - 6) < 1e-6
    before = b.size[0]
    b.observe_steal(1, 0)
    assert b.size[1] == 5 and b.size[0] == before + 1
    b.observe_monopoly(0, B.WHEAT)
    assert all(b.expected[j][B.WHEAT] == 0 for j in (1, 2, 3))
    assert 0 <= b.probability_has(2, B.ORE) <= 1


def test_determinize_conserves_cards():
    s = mid_game_state()
    rng = random.Random(5)
    for _ in range(5):
        d = I.determinize(s, 0, rng)
        assert I.is_fully_known(d)
        for i in (1, 2, 3):
            assert sum(d.players[i].resources) == 5
            assert sum(d.players[i].dev_cards) == 1
        total = sum(d.bank) + sum(sum(p.resources) for p in d.players)
        assert total == 95
        # dev conservation: deck + held + played knights == 25
        held = sum(p.total_dev for p in d.players)
        played = sum(p.played_knights for p in d.players)
        assert sum(d.dev_deck) + held + played == 25
    # original untouched
    assert not s.players[1].hand_known
    samples = I.sample_states(s, 0, n=3, rng=rng)
    assert len(samples) == 3


# ---------------------------------------------------------------------------
# Road blocking (cutting opponents off) and the settlement blocking bonus
# ---------------------------------------------------------------------------
def test_reachable_spots_accepts_an_edge_occupancy_override():
    s = mid_game_state()
    assert P.reachable_spots(s, 0, max_roads=2) == P.reachable_spots(s, 0, max_roads=2, eocc=s.occupied_edges())
    blocked = {e: 9 for e in range(B.NUM_EDGES)}
    reach = P.reachable_spots(s, 0, max_roads=2, eocc=blocked)      # no road can be built: only "now" spots
    default = P.reachable_spots(s, 0, max_roads=2)
    assert all(d == 0 for d, _ in reach.values())
    assert set(reach) == {v for v, (d, _) in default.items() if d == 0}


def _blocking_scenario():
    """Opponent 1: settlement v1 + road to w.  We (0): settlement z next to w with the road z-w, so our
    road on w-x is legal and severs their path to their best spot (x or beyond); found by search so the
    test does not depend on hand-picked vertex ids."""
    from catanbot import engine as E
    from catanbot import actions as A
    for v1 in range(B.NUM_VERTICES):
        for e1 in B.VERTEX_EDGES[v1]:
            a, b = B.EDGE_VERTICES[e1]
            w = b if a == v1 else a
            others = [e for e in B.VERTEX_EDGES[w] if e != e1]
            for e2 in others:
                for e3 in others:
                    if e3 == e2:
                        continue
                    a, b = B.EDGE_VERTICES[e2]
                    x = b if a == w else a
                    a, b = B.EDGE_VERTICES[e3]
                    z = b if a == w else a
                    if not P.is_free_vertex({v1: 1}, z):
                        continue
                    s = new_game(4, hexes=B.STANDARD_HEXES)
                    s.phase, s.dice, s.current = PHASE_MAIN, 6, 0
                    s.players[1].settlements, s.players[1].roads = [v1], [e1]
                    s.players[0].settlements, s.players[0].roads = [z], [e3]
                    s.players[0].resources = [2, 2, 0, 0, 0]
                    for r in range(5):
                        s.bank[r] = 19 - s.players[0].resources[r]
                    occ = s.occupied_vertices()
                    reach = P.reachable_spots(s, 1, max_roads=2, occ=occ)
                    scored = sorted(((P.score_settlement_spot(s, 1, v, occ=occ) / (1 + 0.9 * d), v, fe)
                                     for v, (d, fe) in reach.items()), reverse=True)
                    if len(scored) < 2 or scored[0][2] != e2 or scored[1][2] == e2 or scored[0][0] < scored[1][0] + 0.5:
                        continue
                    if (A.BUILD_ROAD, e2) in E.legal_actions(s):
                        return s, e2, x
    raise AssertionError("no blocking scenario found on the standard board")


def test_road_block_values_and_blocking_bonus():
    from catanbot import engine as E
    from catanbot import actions as A
    from catanbot.heuristic import action_priors
    s, e2, x = _blocking_scenario()
    legal = E.legal_actions(s)
    roads = [a[1] for a in legal if a[0] == A.BUILD_ROAD]
    bv = P.road_block_values(s, 0, roads)
    assert bv[e2] > 0 and all(bv[e] == 0 for e in roads if e != e2)
    assert P.road_block_values(s, 0, [s.players[1].roads[0]]) == {}       # occupied edges are skipped
    targets = P.road_targets(s, 0, max_roads=2, k=20)
    t = next(t for t in targets if t["first_edge"] == e2)
    assert abs(t["block"] - bv[e2]) < 1e-9 and all("block" in t for t in targets)
    # the prior of the blocking road exceeds its "towards our spot" value alone
    priors = dict(zip(legal, action_priors(s, legal, 0)))
    towards = max((t["score"] for t in P.road_targets(s, 0, max_roads=3, k=6) if t["first_edge"] == e2), default=0.0)
    assert priors[(A.BUILD_ROAD, e2)] > 40.0 + 2.0 * towards + 1e-9
    # a settlement on the contested spot x scores higher than the same spot without the blocking bonus
    contested = P.best_settlement_spots(s, 0, candidates=[x], include_blocking=True)[0][1]
    plain = P.best_settlement_spots(s, 0, candidates=[x], include_blocking=False)[0][1]
    assert contested > plain and P.blocking_value(s, 0, x) > 0
