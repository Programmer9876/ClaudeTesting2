"""Tests for catanbot.features."""
from __future__ import annotations

import random

import numpy as np
import pytest

from catanbot import board as B
from catanbot import features as F
from catanbot.state import (GameState, Player, TradeOffer, new_game, PHASE_DISCARD, PHASE_MAIN,
                            PHASE_TRADE_RESPONSE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fi(name: str) -> int:
    return F.feature_index(name)


def hex_chain_edges(h: int, k: int):
    """First ``k`` consecutive edges around hex ``h`` (a simple path of k roads)."""
    vs = B.HEX_VERTICES[h]
    return [B.edge_between(vs[i], vs[(i + 1) % 6]) for i in range(k)]


def custom_hexes():
    """Standard board with hex 9 = wheat 8, hex 10 = ore 8, hex 11 = desert."""
    hexes = list(B.STANDARD_HEXES)
    hexes[9] = (B.WHEAT, 8)
    hexes[10] = (B.ORE, 8)
    hexes[11] = (B.DESERT, 0)
    return hexes


def shared_vertex(h1: int, h2: int) -> int:
    common = [v for v in B.HEX_VERTICES[h1] if v in B.HEX_VERTICES[h2]]
    assert common
    return common[0]


def rotate_state(s: GameState, k: int) -> GameState:
    """Same game with the seats rotated so that old seat ``k`` becomes seat 0."""
    n = s.num_players
    r = s.copy()

    def m(i: int) -> int:
        return -1 if i < 0 else (i - k) % n

    r.players = [s.players[(j + k) % n].copy() for j in range(n)]
    r.current = m(s.current)
    r.longest_road_owner = m(s.longest_road_owner)
    r.largest_army_owner = m(s.largest_army_owner)
    r.winner = m(s.winner)
    r.trade_responder = m(s.trade_responder)
    r.discard_queue = [m(i) for i in s.discard_queue]
    if s.pending_trade is not None:
        t = s.pending_trade
        r.pending_trade = TradeOffer(m(t.proposer), list(t.give), list(t.get),
                                     {m(i): a for i, a in t.responses.items()})
    return r


def busy_state(num_players: int = 4, seed: int = 1) -> GameState:
    """A legal-looking mid-game position with pieces for every player."""
    rng = random.Random(seed)
    s = new_game(num_players, rng=rng)
    s.phase = PHASE_MAIN
    s.turn = 37
    s.dice = 8
    s.current = 1 % num_players
    occupied = set()
    hex_order = [0, 2, 7, 11, 16, 18]  # pairwise non-adjacent: no shared vertices / edges
    for i, p in enumerate(s.players):
        h = hex_order[i]
        vs = B.HEX_VERTICES[h]
        p.settlements = [vs[0]]
        p.cities = [vs[2]]
        occupied.update(vs[0:3])
        p.roads = hex_chain_edges(h, 2 + i)
        p.resources = [rng.randint(0, 4) for _ in range(5)]
        p.dev_cards = [rng.randint(0, 1) for _ in range(5)]
        p.dev_cards_new = [1 if i == 0 else 0, 0, 0, 0, 0]
        p.played_knights = i
        p.hand_size = sum(p.resources)
        p.dev_count = p.total_dev
    s.players[0].hand_known = False
    s.players[0].hand_size = 6
    s.players[0].resources = [0] * 5
    s.players[2 % num_players].dev_known = False
    s.players[2 % num_players].dev_count = 2
    s.bank = [12, 15, 9, 17, 11]
    s.dev_deck = [9, 3, 1, 2, 1]
    s.longest_road_owner = num_players - 1
    s.longest_road_len = 5
    s.largest_army_owner = 2 % num_players
    s.robber = 4
    s.trades_this_turn = 2
    return s


# ---------------------------------------------------------------------------
# shape / dtype / bounds
# ---------------------------------------------------------------------------
def test_feature_names_and_count():
    assert F.NUM_FEATURES == len(F.FEATURE_NAMES)
    assert len(set(F.FEATURE_NAMES)) == F.NUM_FEATURES
    assert F.NUM_FEATURES < 400
    assert F.NUM_FEATURES == 4 * F.PLAYER_BLOCK + F.GLOBAL_BLOCK
    assert F.FEATURE_NAMES[0].startswith("me_")
    assert F.FEATURE_NAMES[F.PLAYER_BLOCK].startswith("opp1_")
    assert F.FEATURE_NAMES[-1].startswith("g_")
    assert set(F.FEATURE_SCALES) == set(F.FEATURE_NAMES)


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_extract_shape_dtype_bounds(num_players):
    s = busy_state(num_players)
    for pl in range(num_players):
        x = F.extract(s, pl)
        assert x.shape == (F.NUM_FEATURES,)
        assert x.dtype == np.float32
        assert np.all(np.isfinite(x))
        assert x.min() >= -1.0 - 1e-6
        assert x.max() <= 1.5  # production of extreme positions may slightly exceed 1
    # padding blocks are all zero for missing seats
    x = F.extract(s, 0)
    for k in range(num_players, 4):
        blk = x[k * F.PLAYER_BLOCK:(k + 1) * F.PLAYER_BLOCK]
        assert not blk.any()
    assert x[fi(f"g_players_{num_players}")] == 1.0


def test_fresh_game_is_mostly_zero():
    s = new_game(4)
    x = F.extract(s, 0)
    assert x[fi("me_public_vp")] == 0
    assert x[fi("me_spots_now")] == 0
    assert x[fi("g_phase_setup_settlement")] == 1.0
    assert x[fi("g_bank_wood")] == 1.0
    assert x[fi("g_deck_total")] == 1.0
    assert x[fi("g_my_turn")] == 1.0
    assert x[fi("me_present")] == 1.0 and x[fi("opp3_present")] == 1.0


# ---------------------------------------------------------------------------
# production
# ---------------------------------------------------------------------------
def test_city_on_8_beats_settlement():
    hexes = custom_hexes()
    v = shared_vertex(9, 10)
    s1 = new_game(4, hexes=hexes)
    s1.players[0].settlements = [v]
    s2 = new_game(4, hexes=hexes)
    s2.players[0].cities = [v]
    assert s1.robber == 11 and s2.robber == 11
    x1, x2 = F.extract(s1, 0), F.extract(s2, 0)
    for r in ("wheat", "ore"):
        assert x1[fi(f"me_prod_{r}")] == pytest.approx(5 / 36 / 2)
        assert x2[fi(f"me_prod_{r}")] == pytest.approx(2 * 5 / 36 / 2)
        assert x2[fi(f"me_prod_{r}")] > x1[fi(f"me_prod_{r}")]
    assert x2[fi("me_public_vp")] == pytest.approx(0.2)
    assert x2[fi("me_cities")] == pytest.approx(0.25)
    touched = {hexes[h][0] for h in B.VERTEX_HEXES[v] if hexes[h][0] != B.DESERT}
    assert x1[fi("me_prod_types")] == pytest.approx(len(touched) / 5)
    assert 0.0 < x1[fi("me_prod_entropy")] <= 1.0
    # helper agrees
    p = F.production_per_roll(s2, 0)
    assert p[B.WHEAT] == pytest.approx(10 / 36) and p[B.ORE] == pytest.approx(10 / 36)


def test_robber_zeroes_hex_production():
    hexes = custom_hexes()
    v = shared_vertex(9, 10)
    s = new_game(4, hexes=hexes)
    s.players[0].cities = [v]
    base = F.extract(s, 0)
    s.robber = 9  # on the wheat 8
    x = F.extract(s, 0)
    assert x[fi("me_prod_wheat")] == 0.0
    assert x[fi("me_prod_ore")] == base[fi("me_prod_ore")]
    assert x[fi("me_prod_nr_wheat")] == base[fi("me_prod_nr_wheat")] > 0
    assert x[fi("me_robber_on_my_hex")] == 1.0 and base[fi("me_robber_on_my_hex")] == 0.0
    assert x[fi("me_robber_loss")] == pytest.approx(10 / 36 / 2)
    assert x[fi("me_prod_total")] < base[fi("me_prod_total")]
    assert x[fi("me_prod_nr_total")] == base[fi("me_prod_nr_total")]
    # robber elsewhere does not touch us
    s.robber = 0
    x0 = F.extract(s, 0)
    assert x0[fi("me_robber_on_my_hex")] == 0.0 and x0[fi("me_robber_loss")] == 0.0


# ---------------------------------------------------------------------------
# expansion / longest road
# ---------------------------------------------------------------------------
def test_settlement_spots_and_reach():
    s = new_game(4)
    s.phase = PHASE_MAIN
    vs = B.HEX_VERTICES[9]  # desert ring in the standard layout
    me = s.players[0]
    me.settlements = [vs[0]]
    me.roads = [B.edge_between(vs[0], vs[1]), B.edge_between(vs[1], vs[2])]
    now, one, two = F.settlement_spots(s, 0)
    assert now == [vs[2]]           # vs[1] is too close to our own settlement
    assert vs[2] not in one and vs[2] not in two
    assert len(one) > 0 and len(two) > 0
    assert not (set(now) & set(one)) and not (set(one) & set(two))
    x = F.extract(s, 0)
    assert x[fi("me_spots_now")] == pytest.approx(0.1)
    assert x[fi("me_spots_1road")] == pytest.approx(len(one) / 10)
    assert x[fi("me_spots_2roads")] == pytest.approx(len(two) / 15)
    assert x[fi("me_best_spot_pips_now")] == pytest.approx(B.vertex_pip_total(vs[2], s.hexes) / 15)
    assert x[fi("me_best_spot_pips_2roads")] >= x[fi("me_best_spot_pips_now")]
    # an opponent settlement on vs[2] blocks that spot and stops the expansion through it
    s.players[1].settlements = [vs[2]]
    now2, one2, two2 = F.settlement_spots(s, 0)
    assert now2 == []
    assert len(one2) + len(two2) < len(one) + len(two)
    # affordability flags require a spot
    me.resources = [1, 1, 1, 1, 0]
    x = F.extract(s, 0)
    assert x[fi("me_can_build_settlement")] == 0.0
    assert x[fi("me_can_build_road")] == 1.0
    s.players[1].settlements = []
    x = F.extract(s, 0)
    assert x[fi("me_can_build_settlement")] == 1.0
    assert x[fi("me_can_build_city")] == 0.0
    me.resources = [0, 0, 0, 2, 3]
    x = F.extract(s, 0)
    assert x[fi("me_can_build_city")] == 1.0 and x[fi("me_can_buy_dev")] == 0.0
    me.resources = [0, 0, 1, 1, 1]
    x = F.extract(s, 0)
    assert x[fi("me_can_buy_dev")] == 1.0
    s.dev_deck = [0] * 5
    x = F.extract(s, 0)
    assert x[fi("me_can_buy_dev")] == 0.0


def test_longest_road_length():
    s = new_game(4)
    vs = B.HEX_VERTICES[9]
    me = s.players[0]
    assert F.longest_road_length(s, 0) == 0
    me.roads = hex_chain_edges(9, 5)          # path vs0-vs1-...-vs5
    assert F.longest_road_length(s, 0) == 5
    me.roads = hex_chain_edges(9, 6)          # full ring
    assert F.longest_road_length(s, 0) == 6
    me.roads = hex_chain_edges(9, 5)
    s.players[1].settlements = [vs[2]]        # opponent breaks the path at vs2
    assert F.longest_road_length(s, 0) == 3   # vs2-vs3-vs4-vs5
    s.players[1].settlements = []
    me.settlements = [vs[2]]                  # own building does not break it
    assert F.longest_road_length(s, 0) == 5
    # a fork: ring minus one edge plus a spur off vs0
    spur = [e for e in B.VERTEX_EDGES[vs[0]] if e not in hex_chain_edges(9, 6)][0]
    me.roads = hex_chain_edges(9, 5) + [spur]
    assert F.longest_road_length(s, 0) == 6
    x = F.extract(s, 0)
    assert x[fi("me_longest_road")] == pytest.approx(6 / 15)
    assert x[fi("g_lr_gap")] == pytest.approx(6 / 5)


# ---------------------------------------------------------------------------
# perspective rotation / batch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_players", [3, 4])
def test_perspective_rotation(num_players):
    s = busy_state(num_players)
    s.pending_trade = TradeOffer(1 % num_players, [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], {})
    s.phase = PHASE_TRADE_RESPONSE
    s.trade_responder = 2 % num_players
    for k in range(num_players):
        r = rotate_state(s, k)
        for pl in range(num_players):
            a = F.extract(s, pl)
            b = F.extract(r, (pl - k) % num_players)
            np.testing.assert_array_equal(a, b)
    # and the perspective actually matters: two players see different vectors
    assert not np.array_equal(F.extract(s, 0), F.extract(s, 1))
    # opponent blocks are in seat order after me
    x = F.extract(s, 1)
    for k in range(1, num_players):
        j = (1 + k) % num_players
        blk = x[k * F.PLAYER_BLOCK:(k + 1) * F.PLAYER_BLOCK]
        assert blk[F._P["knights"]] == pytest.approx(s.players[j].played_knights / 5)


def test_extract_batch_matches_extract():
    states = [busy_state(4, seed=1), busy_state(3, seed=2), new_game(4), busy_state(4, seed=3)]
    pairs = [(states[0], 0), (states[0], 1), (states[0], 2), (states[0], 3), (states[1], 2), (states[2], 0),
             (states[3], 1), (states[0], 3), (states[1], 0)]
    X = F.extract_batch([s for s, _ in pairs], [p for _, p in pairs])
    assert X.shape == (len(pairs), F.NUM_FEATURES) and X.dtype == np.float32
    for i, (s, p) in enumerate(pairs):
        np.testing.assert_array_equal(X[i], F.extract(s, p))
    assert F.extract_batch([], []).shape == (0, F.NUM_FEATURES)
    with pytest.raises(ValueError):
        F.extract_batch(states, [0])


# ---------------------------------------------------------------------------
# hand / dev / awards / global details
# ---------------------------------------------------------------------------
def test_unknown_hand_and_dev_estimates():
    s = busy_state(4)
    x = F.extract(s, 0)
    # player 0: hand unknown (6 cards) -> expected counts sum to 6, flag off
    assert x[fi("me_hand_known")] == 0.0
    assert x[fi("me_hand_size")] == pytest.approx(6 / 15)
    assert sum(x[fi(f"me_res_{r}")] for r in B.RESOURCE_NAMES[:5]) == pytest.approx(0.6)
    assert x[fi("me_can_build_road")] == 0.0  # unknown hand -> no affordability claims
    # player 2 has 2 unknown dev cards
    x2 = F.extract(s, 2)
    assert x2[fi("me_dev_known")] == 0.0
    assert x2[fi("me_dev_total")] == pytest.approx(2 / 5)
    dev_sum = sum(x2[fi(f"me_dev_{d}")] for d in B.DEV_NAMES) * 3
    assert dev_sum == pytest.approx(2.0)
    assert 0.0 <= x2[fi("me_hidden_vp")] <= 2 / 5
    # player 0 has one freshly bought knight (not playable): shows up in devnew, not dev
    assert x[fi("me_devnew_knight")] == pytest.approx(1 / 3)
    assert x[fi("me_dev_knight")] == pytest.approx(s.players[0].dev_cards[0] / 3)


def test_discard_exposure_and_ports():
    s = new_game(4)
    s.phase = PHASE_MAIN
    me = s.players[0]
    me.resources = [3, 3, 2, 1, 0]  # 9 cards
    x = F.extract(s, 0)
    assert x[fi("me_cards_over_7")] == pytest.approx(2 / 8)
    assert x[fi("me_discard_exposure")] == pytest.approx(4 / 8)
    me.resources = [2, 2, 2, 1, 0]
    x = F.extract(s, 0)
    assert x[fi("me_cards_over_7")] == 0.0 and x[fi("me_discard_exposure")] == 0.0
    # ports
    ore_port_v = next(v for v, t in s.ports.items() if t == B.ORE)
    generic_v = next(v for v, t in s.ports.items() if t == B.PORT_GENERIC)
    me.settlements = [ore_port_v]
    x = F.extract(s, 0)
    assert x[fi("me_port_ore")] == 1.0 and x[fi("me_port_wood")] == 0.0
    me.cities = [generic_v]
    x = F.extract(s, 0)
    assert x[fi("me_port_ore")] == 1.0 and x[fi("me_port_wood")] == 0.5
    assert x[fi("me_port_ore")] == (4 - s.port_ratio(0, B.ORE)) / 2
    assert x[fi("me_port_wood")] == (4 - s.port_ratio(0, B.WOOD)) / 2


def test_awards_gaps_and_phase_flags():
    s = busy_state(4)
    x = F.extract(s, 3)
    assert x[fi("me_has_longest_road")] == 1.0
    assert x[fi("opp3_has_largest_army")] == 1.0  # seat 2 is 3 seats after seat 3
    assert x[fi("g_leader_vp")] == pytest.approx(max(s.public_vp(i) for i in range(4)) / 10)
    best_opp = max(s.public_vp(i) for i in range(4) if i != 3)
    assert x[fi("g_vp_gap")] == pytest.approx((s.public_vp(3) - best_opp) / 10, abs=1e-6)
    assert x[fi("g_i_am_leader")] == (1.0 if s.public_vp(3) >= best_opp else 0.0)
    assert x[fi("g_knight_gap")] == pytest.approx((3 - 2) / 5)
    assert x[fi("g_trades_this_turn")] == pytest.approx(0.5)
    assert x[fi("g_phase_main")] == 1.0 and x[fi("g_phase_roll")] == 0.0
    assert x[fi("g_turn")] == pytest.approx(37 / 200)
    assert x[fi("g_stage")] == pytest.approx(37 / 120)
    assert x[fi("g_bank_sheep")] == pytest.approx(9 / 19)
    assert x[fi("g_deck_knight")] == pytest.approx(9 / 14)
    assert x[fi("g_deck_total")] == pytest.approx(16 / 25)
    # discard phase: the acting player is the head of the queue
    s.phase = PHASE_DISCARD
    s.discard_queue = [3, 0]
    x3, x0 = F.extract(s, 3), F.extract(s, 0)
    assert x3[fi("g_i_am_acting")] == 1.0 and x0[fi("g_i_am_acting")] == 0.0
    assert x3[fi("g_my_turn")] == 0.0 and F.extract(s, 1)[fi("g_my_turn")] == 1.0
    # pending trade from proposer / responder / bystander perspective
    s.phase = PHASE_TRADE_RESPONSE
    s.pending_trade = TradeOffer(1, [0, 2, 0, 0, 0], [0, 0, 0, 0, 1], {})
    s.trade_responder = 2
    xp, xr, xb = F.extract(s, 1), F.extract(s, 2), F.extract(s, 3)
    assert xp[fi("g_trade_pending")] == xr[fi("g_trade_pending")] == 1.0
    assert xp[fi("g_trade_i_propose")] == 1.0 and xr[fi("g_trade_i_propose")] == 0.0
    assert xr[fi("g_trade_i_respond")] == 1.0 and xb[fi("g_trade_i_respond")] == 0.0
    assert xr[fi("g_trade_give_brick")] == pytest.approx(2 / 3) and xr[fi("g_trade_get_ore")] == pytest.approx(1 / 3)


def test_batch_is_fast():
    import time
    s = busy_state(4)
    t = time.perf_counter()
    for _ in range(100):
        F.extract_batch([s] * 4, [0, 1, 2, 3])
    per = (time.perf_counter() - t) / 100
    assert per < 0.02  # generous; typically ~0.2 ms
