"""Tests for ``catanbot.engine``: every rule bullet of DESIGN.md section 3 plus fuzzing.

Focused tests build small explicit positions straight from the board tables
(so vertex / edge ids are derived, never guessed); the fuzz tests play whole
random games and check invariants after every action.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Sequence, Set

import pytest

from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot.state import (
    PHASE_DISCARD,
    PHASE_GAME_OVER,
    PHASE_MAIN,
    PHASE_ROBBER,
    PHASE_ROLL,
    PHASE_SETUP_ROAD,
    PHASE_SETUP_SETTLEMENT,
    PHASE_TRADE_RESPONSE,
    PHASE_TRADE_SELECT,
    GameState,
    new_game,
)

Illegal = E.IllegalActionError

# Geometry used by the focused tests (all derived from the board tables).
CENTER = 9                                   # desert hex in the middle: every corner is interior
RING = list(B.HEX_EDGES[CENTER])             # 6 edges around the centre hex; edge k joins corner k and k+1
CORNER = list(B.HEX_VERTICES[CENTER])        # corner k of the centre hex
BRICK6 = 4                                   # (BRICK, 6) in the standard layout
FAR_A = 2                                    # top-right hex, no vertex shared with the centre hex
FAR_B = 16                                   # bottom-left hex, no vertex shared with hex 2 or the centre


def outward_edge(h: int, corner: int) -> int:
    """The one edge at corner ``corner`` of hex ``h`` that is not part of the hex ring."""
    v = B.HEX_VERTICES[h][corner]
    ring = set(B.HEX_EDGES[h])
    cands = [e for e in B.VERTEX_EDGES[v] if e not in ring]
    assert len(cands) == 1, (h, corner, cands)
    return cands[0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def blank_main(n: int = 4) -> GameState:
    """Standard board, no pieces, player 0 in PHASE_MAIN (already rolled)."""
    s = new_game(n, hexes=B.STANDARD_HEXES)
    s.phase = PHASE_MAIN
    s.dice = 8
    return s


def set_hand(s: GameState, i: int, hand: Sequence[int]) -> None:
    """Give player ``i`` exactly ``hand`` while keeping the 95-card total intact."""
    p = s.players[i]
    for r in range(5):
        s.bank[r] += p.resources[r] - hand[r]
    p.resources = list(hand)
    p.hand_size = sum(hand)
    assert min(s.bank) >= 0, s.bank


def play_setup(s: GameState, rng: random.Random) -> GameState:
    while s.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        E.apply_inplace(s, rng.choice(E.legal_actions(s)), rng)
    return s


def kinds(actions: Sequence[A.Action], kind: str) -> List[A.Action]:
    return [a for a in actions if a[0] == kind]


def naive_production(s: GameState, value: int) -> List[List[int]]:
    """Independent reference implementation of the production rule."""
    n = len(s.players)
    gains = [[0] * 5 for _ in range(n)]
    if value == 7:
        return gains
    owed = [0] * 5
    for h, (res, num) in enumerate(s.hexes):
        if num != value or h == s.robber or res == B.DESERT:
            continue
        for v in B.HEX_VERTICES[h]:
            for i, p in enumerate(s.players):
                if v in p.settlements:
                    gains[i][res] += 1
                    owed[res] += 1
                elif v in p.cities:
                    gains[i][res] += 2
                    owed[res] += 2
    for r in range(5):
        if owed[r] > s.bank[r]:
            rec = [i for i in range(n) if gains[i][r]]
            if len(rec) == 1:
                gains[rec[0]][r] = s.bank[r]
            else:
                for i in rec:
                    gains[i][r] = 0
    return gains


def naive_discards(res: Sequence[int], k: int) -> Set[tuple]:
    out = set()
    for combo in itertools.product(*[range(c + 1) for c in res]):
        if sum(combo) == k:
            out.add(combo)
    return out


# ---------------------------------------------------------------------------
# setup phase
# ---------------------------------------------------------------------------
def test_setup_snake_order_and_end_of_setup():
    rng = random.Random(1)
    s = new_game(4, rng)
    assert s.phase == PHASE_SETUP_SETTLEMENT and s.current == 0 and s.turn == 0
    order = []
    while s.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        acts = E.legal_actions(s)
        assert acts
        if s.phase == PHASE_SETUP_SETTLEMENT:
            order.append(E.acting_player(s))
            assert all(a[0] == A.SETUP_SETTLEMENT for a in acts)
        else:
            assert all(a[0] == A.SETUP_ROAD for a in acts)
        E.apply_inplace(s, rng.choice(acts), rng)
    assert order == [0, 1, 2, 3, 3, 2, 1, 0]
    assert s.phase == PHASE_ROLL and s.current == 0 and s.turn == 8 and s.setup_round == 1
    for p in s.players:
        assert len(p.settlements) == 2 and len(p.roads) == 2 and not p.cities
    assert sum(s.bank) + sum(sum(p.resources) for p in s.players) == 95


def test_setup_snake_order_three_players():
    rng = random.Random(2)
    s = new_game(3, rng)
    order = []
    while s.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        if s.phase == PHASE_SETUP_SETTLEMENT:
            order.append(s.current)
        E.apply_inplace(s, rng.choice(E.legal_actions(s)), rng)
    assert order == [0, 1, 2, 2, 1, 0]
    assert s.phase == PHASE_ROLL and s.current == 0 and s.turn == 6


def test_second_settlement_pays_adjacent_hexes_first_does_not():
    rng = random.Random(3)
    s = new_game(4, rng)
    # first round: nobody receives anything
    for _ in range(4):
        E.apply_inplace(s, rng.choice(E.legal_actions(s)), rng)
        E.apply_inplace(s, rng.choice(E.legal_actions(s)), rng)
    assert all(sum(p.resources) == 0 for p in s.players)
    assert s.setup_round == 1 and s.current == 3
    while s.phase != PHASE_ROLL:
        E.apply_inplace(s, rng.choice(E.legal_actions(s)), rng)
    for p in s.players:
        v = p.settlements[1]
        expected = [0] * 5
        for h in B.VERTEX_HEXES[v]:
            res = s.hexes[h][0]
            if res != B.DESERT:
                expected[res] += 1
        assert p.resources == expected
        assert p.hand_size == sum(expected)
    assert sum(s.bank) + sum(sum(p.resources) for p in s.players) == 95


def test_setup_road_must_touch_settlement_just_placed():
    s = new_game(4, hexes=B.STANDARD_HEXES)
    v = CORNER[0]
    s = E.apply(s, (A.SETUP_SETTLEMENT, v))
    assert s.phase == PHASE_SETUP_ROAD and s.setup_last_settlement == v
    acts = E.legal_actions(s)
    assert sorted(a[1] for a in acts) == sorted(B.VERTEX_EDGES[v])
    far = RING[3]  # does not touch corner 0
    with pytest.raises(Illegal):
        E.apply(s, (A.SETUP_ROAD, far))
    s2 = E.apply(s, (A.SETUP_ROAD, B.VERTEX_EDGES[v][0]))
    assert s2.players[0].roads == [B.VERTEX_EDGES[v][0]]
    assert s2.phase == PHASE_SETUP_SETTLEMENT and s2.current == 1 and s2.setup_last_settlement == -1


def test_setup_distance_rule():
    s = new_game(4, hexes=B.STANDARD_HEXES)
    v = CORNER[0]
    s = E.apply(s, (A.SETUP_SETTLEMENT, v))
    s = E.apply(s, (A.SETUP_ROAD, B.VERTEX_EDGES[v][0]))
    legal = {a[1] for a in E.legal_actions(s)}
    assert v not in legal
    for w in B.VERTEX_NEIGHBORS[v]:
        assert w not in legal
        with pytest.raises(Illegal):
            E.apply(s, (A.SETUP_SETTLEMENT, w))
    assert len(legal) == B.NUM_VERTICES - 1 - len(B.VERTEX_NEIGHBORS[v])
    # every other vertex is still legal
    for u in range(B.NUM_VERTICES):
        if u != v and u not in B.VERTEX_NEIGHBORS[v]:
            assert u in legal


def test_setup_actions_illegal_outside_setup():
    s = blank_main()
    with pytest.raises(Illegal):
        E.apply(s, (A.SETUP_SETTLEMENT, CORNER[0]))
    with pytest.raises(Illegal):
        E.apply(s, (A.SETUP_ROAD, RING[0]))


# ---------------------------------------------------------------------------
# rolling, knights before the roll, production
# ---------------------------------------------------------------------------
def test_roll_forced_random_and_phase_restrictions():
    s = blank_main()
    s.phase = PHASE_ROLL
    assert E.legal_actions(s) == [(A.ROLL,)]
    s8 = E.apply(s, (A.ROLL, 8))
    assert s8.dice == 8 and s8.phase == PHASE_MAIN and s8.rolls_history_len == 1
    s_rand = E.apply(s, (A.ROLL,), random.Random(5))
    assert 2 <= s_rand.dice <= 12 and s_rand.phase == PHASE_MAIN
    # deterministic given the rng
    assert E.apply(s, (A.ROLL,), random.Random(5)).dice == s_rand.dice
    with pytest.raises(Illegal):
        E.apply(s8, (A.ROLL,))
    with pytest.raises(Illegal):
        E.apply(s, (A.ROLL, 13))
    with pytest.raises(Illegal):
        E.apply(s, (A.END_TURN,))       # must roll first
    with pytest.raises(Illegal):
        E.apply(s, (A.BUY_DEV,))


def test_roll_outcomes_and_apply_roll():
    s = blank_main()
    s.phase = PHASE_ROLL
    outs = E.roll_outcomes(s)
    assert [v for _, v in outs] == list(range(2, 13))
    assert abs(sum(p for p, _ in outs) - 1.0) < 1e-12
    assert dict((v, p) for p, v in outs)[7] == pytest.approx(6 / 36)
    s2 = E.apply_roll(s, 6)
    assert s2.dice == 6 and s2.phase == PHASE_MAIN and s.phase == PHASE_ROLL and s.dice == 8   # input untouched


def test_knight_before_roll_only_knight_only_old_knight_once():
    s = blank_main()
    s.phase = PHASE_ROLL
    p = s.players[0]
    s.players[1].settlements.append(B.HEX_VERTICES[BRICK6][0])
    set_hand(s, 1, [0, 0, 1, 0, 0])
    p.dev_cards = [1, 0, 1, 1, 1]        # one old knight + other old cards
    p.dev_cards_new = [1, 0, 0, 0, 0]     # a fresh knight: not playable this turn
    p.dev_count = p.total_dev
    acts = E.legal_actions(s)
    assert (A.ROLL,) in acts
    knights = kinds(acts, A.PLAY_KNIGHT)
    assert knights and all(a[1] != s.robber for a in knights)
    assert (A.PLAY_KNIGHT, BRICK6, 1) in knights
    assert {a[0] for a in acts} == {A.ROLL, A.PLAY_KNIGHT}  # no other dev card, no builds
    for bad in ((A.PLAY_ROAD_BUILDING,), (A.PLAY_MONOPOLY, 0), (A.PLAY_YEAR_OF_PLENTY, 0, 1)):
        with pytest.raises(Illegal):
            E.apply(s, bad)
    s2 = E.apply(s, (A.PLAY_KNIGHT, BRICK6, 1), random.Random(0))
    assert s2.phase == PHASE_ROLL and s2.robber == BRICK6
    assert s2.players[0].dev_cards[B.DEV_KNIGHT] == 0 and s2.players[0].played_knights == 1
    assert s2.players[0].resources == [0, 0, 1, 0, 0] and s2.players[1].resources == [0] * 5
    assert s2.players[0].hand_size == 1 and s2.players[1].hand_size == 0
    assert s2.players[0].dev_count == s2.players[0].total_dev
    assert s2.dev_played_this_turn
    assert E.legal_actions(s2) == [(A.ROLL,)]     # one dev card per turn, new knight unplayable
    with pytest.raises(Illegal):
        E.apply(s2, (A.PLAY_KNIGHT, CENTER, -1))
    # a knight bought this turn only (no old knight) is not playable before the roll
    s3 = blank_main()
    s3.phase = PHASE_ROLL
    s3.players[0].dev_cards_new = [1, 0, 0, 0, 0]
    assert E.legal_actions(s3) == [(A.ROLL,)]
    # dev card already played this turn: no knight
    s4 = blank_main()
    s4.phase = PHASE_ROLL
    s4.players[0].dev_cards = [1, 0, 0, 0, 0]
    s4.dev_played_this_turn = True
    assert E.legal_actions(s4) == [(A.ROLL,)]


def _brick6_position() -> GameState:
    s = blank_main()
    s.players[0].settlements.append(B.HEX_VERTICES[BRICK6][0])   # hexes 0, 1, 4
    s.players[1].cities.append(B.HEX_VERTICES[BRICK6][3])        # hexes 4, 8, 9
    return s


def test_production_settlement_one_city_two_robber_blocks():
    s = _brick6_position()
    gains = E.production_for_roll(s, 6)
    assert gains[0] == [0, 1, 0, 0, 0] and gains[1] == [0, 2, 0, 0, 0]
    assert gains[2] == [0] * 5 and gains[3] == [0] * 5
    assert E.production_for_roll(s, 7) == [[0] * 5 for _ in range(4)]
    s.robber = BRICK6
    assert E.production_for_roll(s, 6) == [[0] * 5 for _ in range(4)]
    # a roll actually pays out and keeps the bank / hand sizes in sync
    s.robber = CENTER
    s.phase = PHASE_ROLL
    s2 = E.apply(s, (A.ROLL, 6))
    assert s2.players[0].resources[B.BRICK] == 1 and s2.players[1].resources[B.BRICK] == 2
    assert s2.bank[B.BRICK] == 19 - 3 and s2.players[1].hand_size == 2 and s2.phase == PHASE_MAIN


def test_production_bank_shortage_rule():
    s = _brick6_position()
    s.bank[B.BRICK] = 2                       # owed 3 to two players -> nobody
    assert E.production_for_roll(s, 6)[0][B.BRICK] == 0
    assert E.production_for_roll(s, 6)[1][B.BRICK] == 0
    s.bank[B.BRICK] = 3                       # exactly enough -> everybody
    assert E.production_for_roll(s, 6)[1][B.BRICK] == 2
    s.players[0].settlements.clear()          # single player owed 2, bank has 1 -> gets the 1
    s.bank[B.BRICK] = 1
    g = E.production_for_roll(s, 6)
    assert g[1][B.BRICK] == 1 and g[0][B.BRICK] == 0
    s.phase = PHASE_ROLL
    s2 = E.apply(s, (A.ROLL, 6))
    assert s2.bank[B.BRICK] == 0 and s2.players[1].resources[B.BRICK] == 1


def test_production_matches_reference_on_random_states():
    rng = random.Random(11)
    for seed in range(4):
        s = new_game(4, random.Random(seed))
        s.max_turns = 120
        while not E.is_terminal(s):
            E.apply_inplace(s, rng.choice(E.legal_actions(s)), rng)
            if s.turn % 7 == 0 and s.phase == PHASE_ROLL:
                for v in range(2, 13):
                    assert E.production_for_roll(s, v) == naive_production(s, v)


# ---------------------------------------------------------------------------
# sevens: discard queue, discard options, robber
# ---------------------------------------------------------------------------
def test_seven_discard_queue_in_seat_order_from_roller():
    s = blank_main()
    s.phase = PHASE_ROLL
    s.current = 2
    set_hand(s, 0, [3, 2, 2, 1, 1])   # 9 -> discard 4
    set_hand(s, 1, [2, 2, 2, 2, 0])   # 8 -> discard 4
    set_hand(s, 2, [2, 2, 2, 1, 0])   # 7 -> keeps everything
    set_hand(s, 3, [4, 3, 3, 0, 0])   # 10 -> discard 5
    s = E.apply(s, (A.ROLL, 7))
    assert s.dice == 7 and s.phase == PHASE_DISCARD and s.discard_queue == [3, 0, 1]
    assert E.acting_player(s) == 3
    acts = E.legal_actions(s)
    assert acts and all(a[0] == A.DISCARD for a in acts)
    assert all(sum(a[1]) == 5 for a in acts)
    assert all(all(a[1][r] <= s.players[3].resources[r] for r in range(5)) for a in acts)
    assert len({a[1] for a in acts}) == len(acts) == len(naive_discards([4, 3, 3, 0, 0], 5))
    with pytest.raises(Illegal):
        E.apply(s, (A.DISCARD, (4, 0, 0, 0, 0)))           # wrong total
    with pytest.raises(Illegal):
        E.apply(s, (A.DISCARD, (0, 0, 0, 5, 0)))           # more than held
    with pytest.raises(Illegal):
        E.apply(s, (A.MOVE_ROBBER, CENTER + 1, -1))        # not yet
    s = E.apply(s, (A.DISCARD, (2, 2, 1, 0, 0)))
    assert s.players[3].resources == [2, 1, 2, 0, 0] and s.players[3].hand_size == 5
    assert s.bank[B.WOOD] == 19 - 11 + 2
    assert s.discard_queue == [0, 1] and E.acting_player(s) == 0
    s = E.apply(s, E.legal_actions(s)[0])
    assert E.acting_player(s) == 1
    s = E.apply(s, E.legal_actions(s)[0])
    assert s.phase == PHASE_ROBBER and s.discard_queue == [] and E.acting_player(s) == 2
    assert sum(s.bank) + sum(sum(p.resources) for p in s.players) == 95


def test_seven_without_discards_goes_straight_to_robber():
    s = blank_main()
    s.phase = PHASE_ROLL
    set_hand(s, 1, [2, 2, 2, 1, 0])
    s = E.apply(s, (A.ROLL, 7))
    assert s.phase == PHASE_ROBBER and E.acting_player(s) == 0 and not s.discard_queue


def test_discard_options_complete_small_hands_capped_large_hands():
    for hand in ([3, 2, 2, 1, 1], [8, 0, 0, 0, 0], [2, 2, 2, 2, 2], [5, 4, 3, 2, 2]):
        opts = E.discard_options(hand, sum(hand) // 2)
        assert set(opts) == naive_discards(hand, sum(hand) // 2)
        assert len(set(opts)) == len(opts)
    big = [5, 5, 5, 5, 5]                     # 25 cards -> 780 vectors, capped
    opts = E.discard_options(big, 12)
    assert len(opts) == E.DISCARD_ENUM_CAP == 200
    assert all(sum(o) == 12 for o in opts)
    assert len(set(opts)) == 200
    # preference: dump the most-held resources first
    skew = [9, 1, 1, 1, 8]
    opts = E.discard_options(skew, 10)
    assert opts[0] == (9, 0, 0, 0, 1)
    assert all(o[0] >= opts[-1][0] for o in opts)
    assert E.discard_options([1, 0, 0, 0, 0], 2) == []


def test_move_robber_must_change_hex_and_pick_valid_victim():
    s = blank_main()
    s.phase = PHASE_ROBBER
    s.robber = CENTER
    top = B.HEX_VERTICES[BRICK6][0]           # hexes 0, 1, 4
    bottom = B.HEX_VERTICES[BRICK6][3]        # hexes 4, 8, 9
    s.players[1].settlements.append(top)
    s.players[2].cities.append(bottom)
    s.players[0].settlements.append(B.HEX_VERTICES[FAR_B][0])
    set_hand(s, 1, [1, 0, 0, 0, 0])           # player 2 has no cards
    acts = E.legal_actions(s)
    assert all(a[0] == A.MOVE_ROBBER and a[1] != CENTER for a in acts)
    by_hex: Dict[int, List[int]] = {}
    for a in acts:
        by_hex.setdefault(a[1], []).append(a[2])
    assert set(by_hex) == set(range(B.NUM_HEXES)) - {CENTER}
    assert by_hex[BRICK6] == [1]              # player 2 is adjacent but has no card
    assert by_hex[0] == [1] and by_hex[8] == [-1]
    assert by_hex[FAR_B] == [-1]              # own building is never a victim
    assert E.robber_victims(s, BRICK6, 0) == [1]
    assert E.robber_victims(s, BRICK6, 1) == []
    for bad in ((A.MOVE_ROBBER, BRICK6, -1), (A.MOVE_ROBBER, BRICK6, 2), (A.MOVE_ROBBER, CENTER, -1),
                (A.MOVE_ROBBER, 8, 1), (A.MOVE_ROBBER, 99, -1)):
        with pytest.raises(Illegal):
            E.apply(s, bad)
    s2 = E.apply(s, (A.MOVE_ROBBER, BRICK6, 1), random.Random(0))
    assert s2.robber == BRICK6 and s2.phase == PHASE_MAIN
    assert s2.players[0].resources == [1, 0, 0, 0, 0] and s2.players[1].resources == [0] * 5
    assert s2.players[0].hand_size == 1 and s2.players[1].hand_size == 0
    s3 = E.apply(s, (A.MOVE_ROBBER, 8, -1))
    assert s3.robber == 8 and s3.players[0].resources == [0] * 5


def test_steal_is_a_uniformly_random_card():
    s = blank_main()
    s.phase = PHASE_ROBBER
    s.robber = CENTER
    s.players[1].settlements.append(B.HEX_VERTICES[BRICK6][0])
    set_hand(s, 1, [2, 0, 0, 0, 2])
    seen = [0] * 5
    for seed in range(60):
        s2 = E.apply(s, (A.MOVE_ROBBER, BRICK6, 1), random.Random(seed))
        got = s2.players[0].resources
        assert sum(got) == 1 and sum(s2.players[1].resources) == 3
        seen[got.index(1)] += 1
    assert seen[B.WOOD] > 10 and seen[B.ORE] > 10 and seen[B.BRICK] == seen[B.SHEEP] == seen[B.WHEAT] == 0


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def test_build_road_connectivity_cost_occupancy_and_cut():
    s = blank_main()
    v0, v1, v2 = CORNER[0], CORNER[1], CORNER[2]
    s.players[0].settlements.append(v0)
    s.players[0].roads.extend([RING[0], RING[1]])          # v0 - v1 - v2
    s.players[1].settlements.append(v2)                    # opponent building cuts at v2
    set_hand(s, 0, [1, 1, 0, 0, 0])
    legal = {a[1] for a in kinds(E.legal_actions(s), A.BUILD_ROAD)}
    expected = (set(B.VERTEX_EDGES[v0]) | set(B.VERTEX_EDGES[v1])) - {RING[0], RING[1]}
    assert legal == expected
    assert RING[2] not in legal and outward_edge(CENTER, 2) not in legal
    assert set(E.buildable_road_edges(s, 0)) == expected
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_ROAD, RING[2]))                # through the opponent's settlement
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_ROAD, RING[0]))                # occupied
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_ROAD, RING[4]))                # not connected
    e = outward_edge(CENTER, 1)
    s2 = E.apply(s, (A.BUILD_ROAD, e))
    assert e in s2.players[0].roads and s2.players[0].resources == [0] * 5
    assert s2.bank[B.WOOD] == 19 and s2.bank[B.BRICK] == 19 and s2.players[0].hand_size == 0
    # the cut vertex still lets the opponent extend from their own building
    s.current = 1
    set_hand(s, 1, [1, 1, 0, 0, 0])
    legal1 = {a[1] for a in kinds(E.legal_actions(s), A.BUILD_ROAD)}
    assert legal1 == set(B.VERTEX_EDGES[v2]) - {RING[1]}
    # cost
    s.current = 0
    set_hand(s, 0, [0, 1, 0, 0, 0])
    assert not kinds(E.legal_actions(s), A.BUILD_ROAD)
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_ROAD, e))
    # piece limit
    set_hand(s, 0, [1, 1, 0, 0, 0])
    s.players[0].roads = list(range(15))
    assert not kinds(E.legal_actions(s), A.BUILD_ROAD)
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_ROAD, 70))


def test_build_settlement_distance_rule_road_requirement_cost_and_limit():
    s = blank_main()
    v0, v1, v2, v3 = CORNER[0], CORNER[1], CORNER[2], CORNER[3]
    s.players[0].settlements.append(v0)
    s.players[0].roads.extend([RING[0], RING[1]])          # reaches v1 (too close) and v2
    set_hand(s, 0, [1, 1, 1, 1, 0])
    legal = {a[1] for a in kinds(E.legal_actions(s), A.BUILD_SETTLEMENT)}
    assert legal == {v2}
    assert E.buildable_settlement_vertices(s, 0) == [v2]
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_SETTLEMENT, v1))               # distance rule
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_SETTLEMENT, v3))               # no own road there
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_SETTLEMENT, v0))               # occupied
    s2 = E.apply(s, (A.BUILD_SETTLEMENT, v2))
    assert s2.players[0].settlements == [v0, v2] and s2.players[0].resources == [0] * 5
    assert s2.bank == [19] * 5 and s2.players[0].hand_size == 0
    assert E.count_vp(s2, 0) == 2
    # an opponent building next to v2 blocks it
    s.players[1].settlements.append(v3)
    assert not kinds(E.legal_actions(s), A.BUILD_SETTLEMENT)
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_SETTLEMENT, v2))
    s.players[1].settlements.clear()
    # cost
    set_hand(s, 0, [1, 1, 1, 0, 0])
    assert not kinds(E.legal_actions(s), A.BUILD_SETTLEMENT)
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_SETTLEMENT, v2))
    # piece limit
    set_hand(s, 0, [1, 1, 1, 1, 0])
    s.players[0].settlements = [v0, 1, 3, 5, 7]
    assert not kinds(E.legal_actions(s), A.BUILD_SETTLEMENT)
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_SETTLEMENT, v2))


def test_build_city_replaces_own_settlement_and_returns_piece():
    s = blank_main()
    v0, v2 = CORNER[0], CORNER[2]
    s.players[0].settlements.extend([v0, v2])
    s.players[1].settlements.append(B.HEX_VERTICES[FAR_A][0])
    set_hand(s, 0, [0, 0, 0, 2, 3])
    assert {a[1] for a in kinds(E.legal_actions(s), A.BUILD_CITY)} == {v0, v2}
    assert E.buildable_city_vertices(s, 0) == sorted([v0, v2])
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_CITY, B.HEX_VERTICES[FAR_A][0]))   # opponent's settlement
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_CITY, CORNER[4]))                   # empty vertex
    s2 = E.apply(s, (A.BUILD_CITY, v0))
    p = s2.players[0]
    assert p.settlements == [v2] and p.cities == [v0] and p.resources == [0] * 5
    assert s2.bank == [19] * 5 and E.count_vp(s2, 0) == 3
    # settlement piece is back: with 4 settlements + this city the player may still build a 5th settlement
    assert len(p.settlements) < B.MAX_SETTLEMENTS
    with pytest.raises(Illegal):
        E.apply(s2, (A.BUILD_CITY, v0))                          # already a city
    # cost and city limit
    set_hand(s, 0, [0, 0, 0, 2, 2])
    assert not kinds(E.legal_actions(s), A.BUILD_CITY)
    set_hand(s, 0, [0, 0, 0, 2, 3])
    s.players[0].cities = [40, 42, 44, 46]
    assert not kinds(E.legal_actions(s), A.BUILD_CITY)
    with pytest.raises(Illegal):
        E.apply(s, (A.BUILD_CITY, v0))


# ---------------------------------------------------------------------------
# development cards
# ---------------------------------------------------------------------------
def test_buy_dev_draws_weighted_lands_in_new_and_moves_at_end_of_turn():
    s = blank_main()
    s.dev_deck = [0, 0, 0, 0, 1]
    set_hand(s, 0, [0, 0, 1, 1, 1])
    assert (A.BUY_DEV,) in E.legal_actions(s)
    s2 = E.apply(s, (A.BUY_DEV,), random.Random(0))
    p = s2.players[0]
    assert p.dev_cards_new == [0, 0, 0, 0, 1] and p.dev_cards == [0] * 5 and s2.dev_deck == [0] * 5
    assert p.resources == [0] * 5 and s2.bank == [19] * 5 and p.dev_count == 1 and p.hand_size == 0
    assert not kinds(E.legal_actions(s2), A.PLAY_MONOPOLY)       # bought this turn
    with pytest.raises(Illegal):
        E.apply(s2, (A.PLAY_MONOPOLY, B.WOOD))
    assert (A.BUY_DEV,) not in E.legal_actions(s2)                # deck empty
    with pytest.raises(Illegal):
        E.apply(s2, (A.BUY_DEV,))
    s3 = E.apply(s2, (A.END_TURN,))
    assert s3.players[0].dev_cards == [0, 0, 0, 0, 1] and s3.players[0].dev_cards_new == [0] * 5
    # cost
    set_hand(s, 0, [0, 0, 1, 1, 0])
    assert (A.BUY_DEV,) not in E.legal_actions(s)
    # drawing the whole deck yields exactly the deck composition (weighted draw without replacement)
    s = blank_main()
    rng = random.Random(7)
    for _ in range(25):
        set_hand(s, 0, [0, 0, 1, 1, 1])
        E.apply_inplace(s, (A.BUY_DEV,), rng)
    assert s.dev_deck == [0] * 5 and s.players[0].dev_cards_new == B.DEV_DECK_COUNTS
    assert s.players[0].dev_count == 25


def test_one_dev_card_per_turn_and_vp_never_played():
    s = blank_main()
    p = s.players[0]
    p.dev_cards = [1, 2, 1, 1, 1]
    p.dev_count = 6
    acts = E.legal_actions(s)
    assert kinds(acts, A.PLAY_KNIGHT) and kinds(acts, A.PLAY_ROAD_BUILDING)
    assert kinds(acts, A.PLAY_YEAR_OF_PLENTY) and kinds(acts, A.PLAY_MONOPOLY)
    assert not any("victory" in a[0] for a in acts)
    assert E.count_vp(s, 0) == 2 and E.count_vp(s, 0, include_hidden=False) == 0
    s2 = E.apply(s, (A.PLAY_MONOPOLY, B.WOOD))
    assert s2.dev_played_this_turn
    acts2 = E.legal_actions(s2)
    for k in (A.PLAY_KNIGHT, A.PLAY_ROAD_BUILDING, A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY):
        assert not kinds(acts2, k)
    with pytest.raises(Illegal):
        E.apply(s2, (A.PLAY_ROAD_BUILDING,))
    s3 = E.apply(s2, (A.END_TURN,))
    assert not s3.dev_played_this_turn
    assert s3.players[0].dev_cards[B.DEV_VP] == 2       # VP cards stay forever


def test_largest_army_first_to_three_and_strictly_more_takeover():
    s = blank_main()
    s.robber = CENTER
    s.players[0].dev_cards = [3, 0, 0, 0, 0]
    s.players[1].dev_cards = [4, 0, 0, 0, 0]

    def knight(state: GameState, who: int) -> GameState:
        state.current = who
        state.dev_played_this_turn = False
        target = FAR_A if state.robber != FAR_A else FAR_B
        return E.apply(state, (A.PLAY_KNIGHT, target, -1))

    s = knight(s, 0)
    s = knight(s, 0)
    assert s.largest_army_owner == -1 and s.players[0].played_knights == 2
    s = knight(s, 0)
    assert s.largest_army_owner == 0 and E.count_vp(s, 0) == 2
    s = knight(s, 1)
    s = knight(s, 1)
    s = knight(s, 1)
    assert s.largest_army_owner == 0                     # 3 == 3: no takeover
    s = knight(s, 1)
    assert s.largest_army_owner == 1 and E.count_vp(s, 0) == 0 and E.count_vp(s, 1) == 2
    s.players[0].dev_cards[B.DEV_KNIGHT] = 1
    s = knight(s, 0)
    assert s.players[0].played_knights == 4 and s.largest_army_owner == 1   # tie keeps the holder


def test_road_building_free_roads_consumed_first_and_forfeited():
    s = blank_main()
    v0 = CORNER[0]
    s.players[0].settlements.append(v0)
    s.players[0].roads.append(RING[0])
    s.players[0].dev_cards = [0, 0, 1, 0, 0]
    set_hand(s, 0, [1, 1, 0, 0, 0])
    s = E.apply(s, (A.PLAY_ROAD_BUILDING,))
    assert s.free_roads == 2 and s.dev_played_this_turn and s.players[0].dev_cards[B.DEV_ROAD_BUILDING] == 0
    acts = E.legal_actions(s)
    assert {a[0] for a in acts} == {A.BUILD_ROAD, A.END_TURN}   # roads must be placed first
    with pytest.raises(Illegal):
        E.apply(s, (A.BANK_TRADE, B.WOOD, B.ORE))
    with pytest.raises(Illegal):
        E.apply(s, (A.BUY_DEV,))
    s = E.apply(s, (A.BUILD_ROAD, RING[1]))
    assert s.free_roads == 1 and s.players[0].resources == [1, 1, 0, 0, 0]   # free: nothing paid
    s = E.apply(s, (A.BUILD_ROAD, RING[2]))
    assert s.free_roads == 0 and s.players[0].resources == [1, 1, 0, 0, 0]
    s = E.apply(s, (A.BUILD_ROAD, RING[3]))                                 # now it costs
    assert s.players[0].resources == [0] * 5 and len(s.players[0].roads) == 4
    # only one road piece left -> only one free road
    s2 = blank_main()
    s2.players[0].settlements.append(v0)
    s2.players[0].roads = [RING[0]] + list(range(50, 63))     # 14 roads
    s2.players[0].dev_cards = [0, 0, 1, 0, 0]
    s2 = E.apply(s2, (A.PLAY_ROAD_BUILDING,))
    assert s2.free_roads == 1
    s2 = E.apply(s2, (A.BUILD_ROAD, RING[1]))
    assert s2.free_roads == 0 and len(s2.players[0].roads) == 15
    # no legal road at all -> the free roads are forfeited immediately
    s3 = blank_main()
    s3.players[0].settlements.append(v0)
    s3.players[0].roads.append(RING[0])
    for e in (set(B.VERTEX_EDGES[v0]) | set(B.VERTEX_EDGES[CORNER[1]])) - {RING[0]}:
        s3.players[1].roads.append(e)
    s3.players[0].dev_cards = [0, 0, 1, 0, 0]
    s3 = E.apply(s3, (A.PLAY_ROAD_BUILDING,))
    assert s3.free_roads == 0 and s3.dev_played_this_turn
    assert (A.END_TURN,) in E.legal_actions(s3)
    # END_TURN with a free road pending forfeits it
    s4 = blank_main()
    s4.players[0].settlements.append(v0)
    s4.players[0].dev_cards = [0, 0, 1, 0, 0]
    s4 = E.apply(s4, (A.PLAY_ROAD_BUILDING,))
    assert s4.free_roads == 2
    s4 = E.apply(s4, (A.END_TURN,))
    assert s4.free_roads == 0 and s4.current == 1


def test_year_of_plenty_limited_by_bank():
    s = blank_main()
    s.players[0].dev_cards = [0, 0, 0, 1, 0]
    s.bank = [0, 1, 19, 19, 19]
    acts = kinds(E.legal_actions(s), A.PLAY_YEAR_OF_PLENTY)
    pairs = {(a[1], a[2]) for a in acts}
    assert all(r1 <= r2 for r1, r2 in pairs)
    assert (B.WOOD, B.ORE) not in pairs and (B.BRICK, B.BRICK) not in pairs
    assert (B.BRICK, B.SHEEP) in pairs and (B.ORE, B.ORE) in pairs
    assert len(pairs) == 3 + 6                  # brick+{s,w,o} and all 6 multisets of size 2 from {s,w,o}
    with pytest.raises(Illegal):
        E.apply(s, (A.PLAY_YEAR_OF_PLENTY, B.BRICK, B.BRICK))
    with pytest.raises(Illegal):
        E.apply(s, (A.PLAY_YEAR_OF_PLENTY, B.ORE, B.SHEEP))    # must be ordered
    s2 = E.apply(s, (A.PLAY_YEAR_OF_PLENTY, B.BRICK, B.ORE))
    assert s2.players[0].resources == [0, 1, 0, 0, 1] and s2.bank == [0, 0, 19, 19, 18]
    assert s2.players[0].dev_cards == [0] * 5 and s2.dev_played_this_turn and s2.players[0].hand_size == 2


def test_monopoly_takes_all_cards_of_a_resource_from_every_opponent():
    s = blank_main()
    s.players[0].dev_cards = [0, 0, 0, 0, 1]
    set_hand(s, 0, [0, 0, 0, 1, 0])
    set_hand(s, 1, [1, 0, 0, 2, 0])
    set_hand(s, 2, [0, 0, 0, 3, 1])
    set_hand(s, 3, [0, 0, 0, 0, 2])
    assert set(kinds(E.legal_actions(s), A.PLAY_MONOPOLY)) == {(A.PLAY_MONOPOLY, r) for r in range(5)}
    bank = list(s.bank)
    s2 = E.apply(s, (A.PLAY_MONOPOLY, B.WHEAT))
    assert s2.players[0].resources == [0, 0, 0, 6, 0]
    assert s2.players[1].resources == [1, 0, 0, 0, 0] and s2.players[2].resources == [0, 0, 0, 0, 1]
    assert s2.players[3].resources == [0, 0, 0, 0, 2]
    assert s2.bank == bank and all(p.hand_size == sum(p.resources) for p in s2.players)
    assert s2.players[0].dev_cards == [0] * 5 and s2.dev_played_this_turn


# ---------------------------------------------------------------------------
# longest road
# ---------------------------------------------------------------------------
def test_longest_road_length_paths_loops_branches_and_cuts():
    s = blank_main()
    p0 = s.players[0]
    assert E.longest_road_length(s, 0) == 0
    p0.roads = RING[:5]                                   # corners 0..5 in a line
    assert E.longest_road_length(s, 0) == 5
    p0.roads = list(RING)                                 # a loop counts every road
    assert E.longest_road_length(s, 0) == 6
    p0.roads = list(RING) + [outward_edge(CENTER, 0)]     # loop + tail: vertices may repeat, roads may not
    assert E.longest_road_length(s, 0) == 7
    p0.roads = RING[:4] + [outward_edge(CENTER, 2)]       # a branch off the middle does not extend
    assert E.longest_road_length(s, 0) == 4
    p0.roads = RING[:4] + [outward_edge(CENTER, 4)]       # a branch at the end does
    assert E.longest_road_length(s, 0) == 5
    p0.roads = [RING[0], RING[3]]                         # disconnected pieces
    assert E.longest_road_length(s, 0) == 1
    # an opponent building on a vertex breaks the path there (the roads into it still count)
    p0.roads = RING[:5]
    s.players[1].settlements.append(CORNER[2])
    assert E.longest_road_length(s, 0) == 3               # corners 2-3-4-5
    s.players[1].settlements[:] = [CORNER[3]]
    assert E.longest_road_length(s, 0) == 3               # corners 0-1-2-3
    s.players[1].settlements[:] = []
    s.players[0].settlements.append(CORNER[2])            # own building never blocks
    assert E.longest_road_length(s, 0) == 5


def test_longest_road_award_needs_five_ties_keep_holder_and_takeover():
    s = blank_main()
    s.players[0].settlements.append(CORNER[0])
    s.players[0].roads = RING[:4]
    s.players[1].settlements.append(B.HEX_VERTICES[FAR_A][0])
    s.players[1].roads = list(B.HEX_EDGES[FAR_A][:4])
    set_hand(s, 0, [2, 2, 0, 0, 0])
    set_hand(s, 1, [2, 2, 0, 0, 0])
    assert s.longest_road_owner == -1
    s = E.apply(s, (A.BUILD_ROAD, RING[4]))               # 5th road
    assert s.longest_road_owner == 0 and s.longest_road_len == 5 and E.count_vp(s, 0) == 3
    s.current = 1
    s = E.apply(s, (A.BUILD_ROAD, B.HEX_EDGES[FAR_A][4]))  # player 1 ties at 5
    assert s.longest_road_owner == 0 and s.longest_road_len == 5
    s = E.apply(s, (A.BUILD_ROAD, B.HEX_EDGES[FAR_A][5]))  # 6 > 5: takeover
    assert s.longest_road_owner == 1 and s.longest_road_len == 6
    assert E.count_vp(s, 0) == 1 and E.count_vp(s, 1) == 3
    # fewer than 5 roads never qualifies even with a holder-free board
    s2 = blank_main()
    s2.players[0].settlements.append(CORNER[0])
    s2.players[0].roads = RING[:3]
    set_hand(s2, 0, [1, 1, 0, 0, 0])
    s2 = E.apply(s2, (A.BUILD_ROAD, RING[3]))
    assert s2.longest_road_owner == -1 and s2.longest_road_len == 0


def _cut_position() -> GameState:
    """Player 0 holds Longest Road (5 along the centre ring); player 1 can settle on corner 3."""
    s = blank_main()
    s.players[0].settlements.append(CORNER[0])
    s.players[0].roads = RING[:4]
    set_hand(s, 0, [1, 1, 0, 0, 0])
    s = E.apply(s, (A.BUILD_ROAD, RING[4]))
    assert s.longest_road_owner == 0
    s.players[1].roads.append(outward_edge(CENTER, 3))
    set_hand(s, 1, [1, 1, 1, 1, 0])
    s.current = 1
    return s


def test_settlement_cut_drops_holder_to_nobody_unique_or_tie():
    s = _cut_position()
    s = E.apply(s, (A.BUILD_SETTLEMENT, CORNER[3]))
    assert E.longest_road_length(s, 0) == 3
    assert s.longest_road_owner == -1 and s.longest_road_len == 0
    # unique new maximum takes over
    s = _cut_position()
    s.players[2].roads = list(B.HEX_EDGES[FAR_A][:5])
    s = E.apply(s, (A.BUILD_SETTLEMENT, CORNER[3]))
    assert s.longest_road_owner == 2 and s.longest_road_len == 5
    # tie between the remaining candidates: nobody
    s = _cut_position()
    s.players[2].roads = list(B.HEX_EDGES[FAR_A][:5])
    s.players[3].roads = list(B.HEX_EDGES[FAR_B][:5])
    s = E.apply(s, (A.BUILD_SETTLEMENT, CORNER[3]))
    assert s.longest_road_owner == -1 and s.longest_road_len == 0
    # ... and the tie is resolved by the next road that makes a unique maximum
    s.current = 2
    set_hand(s, 2, [1, 1, 0, 0, 0])
    s = E.apply(s, (A.BUILD_ROAD, B.HEX_EDGES[FAR_A][5]))
    assert s.longest_road_owner == 2 and s.longest_road_len == 6


def _external_state(red_roads: Sequence[int], blue_roads: Sequence[int], owner: int, length: int) -> GameState:
    """A hand-built 2-player position round-tripped through JSON with the award fields forced.

    ``GameState.from_dict`` (like ``vision.schema.parsed_to_state`` with a badge
    from the player panel) takes ``longest_road_owner`` verbatim, so the recorded
    holder's real trail may be shorter than 5.  Red (player 0, to move) can afford
    one road.
    """
    s = blank_main(2)
    s.players[0].settlements.append(CORNER[0])
    s.players[0].roads = list(red_roads)
    s.players[1].settlements.append(B.HEX_VERTICES[FAR_B][0])
    s.players[1].roads = list(blue_roads)
    set_hand(s, 0, [1, 1, 0, 0, 0])
    d = s.to_dict()
    d["longest_road_owner"] = owner
    d["longest_road_len"] = length
    return GameState.from_dict(d)


def test_road_build_on_external_state_never_awards_or_keeps_a_sub_five_trail():
    """The fast path must not trust a recorded holder whose trail is < 5 (missed road in a screenshot)."""
    blue = list(B.HEX_EDGES[FAR_B][:3])                    # blue's real trail is 3 but it wears the badge
    # a branch off the middle: 5 roads yet still a 4-trail -> nobody qualifies (not "4 > 3, award")
    s = _external_state(RING[:4], blue, owner=1, length=5)
    assert E.longest_road_length(s, 1) == 3 and s.longest_road_owner == 1
    s = E.apply(s, (A.BUILD_ROAD, outward_edge(CENTER, 2)))
    assert E.longest_road_length(s, 0) == 4
    assert s.longest_road_owner == -1 and s.longest_road_len == 0
    assert E.count_vp(s, 0) == 1 and E.count_vp(s, 1) == 1
    # the builder only ties the stale holder below 5: the holder is dropped, not kept at length 3
    scattered = RING[:2] + [B.HEX_EDGES[FAR_A][0], B.HEX_EDGES[FAR_A][2], B.HEX_EDGES[FAR_A][4]]
    s = _external_state(scattered, blue, owner=1, length=5)
    s = E.apply(s, (A.BUILD_ROAD, RING[2]))
    assert E.longest_road_length(s, 0) == 3 == E.longest_road_length(s, 1)
    assert s.longest_road_owner == -1 and s.longest_road_len == 0
    # a build that really reaches 5 takes the card from the stale holder
    s = _external_state(RING[:4], blue, owner=1, length=5)
    s = E.apply(s, (A.BUILD_ROAD, RING[4]))
    assert s.longest_road_owner == 0 and s.longest_road_len == 5 and E.count_vp(s, 0) == 3
    # the recorded holder is the builder itself and is still short of 5 after the build
    s = _external_state(RING[:3] + [B.HEX_EDGES[FAR_A][0], B.HEX_EDGES[FAR_A][2]], blue, owner=0, length=5)
    s = E.apply(s, (A.BUILD_ROAD, RING[3]))
    assert E.longest_road_length(s, 0) == 4
    assert s.longest_road_owner == -1 and s.longest_road_len == 0
    set_hand(s, 0, [1, 1, 0, 0, 0])
    s = E.apply(s, (A.BUILD_ROAD, RING[4]))                # ... and qualifies with the next road
    assert s.longest_road_owner == 0 and s.longest_road_len == 5 and E.count_vp(s, 0) == 3
    # a stale recorded length is re-derived from the holder's real trail
    s = _external_state(RING[:5], blue, owner=0, length=9)
    s = E.apply(s, (A.BUILD_ROAD, outward_edge(CENTER, 5)))
    assert s.longest_road_owner == 0 and s.longest_road_len == 6
    # a genuine holder is still handled by the shortcut exactly as before (tie keeps, 6 > 5 takes over)
    s = _external_state(RING[:4], list(B.HEX_EDGES[FAR_B][:5]), owner=1, length=5)
    s = E.apply(s, (A.BUILD_ROAD, RING[4]))
    assert s.longest_road_owner == 1 and s.longest_road_len == 5
    set_hand(s, 0, [1, 1, 0, 0, 0])
    s = E.apply(s, (A.BUILD_ROAD, outward_edge(CENTER, 5)))
    assert s.longest_road_owner == 0 and s.longest_road_len == 6


# ---------------------------------------------------------------------------
# trading
# ---------------------------------------------------------------------------
def test_bank_trade_ratios_and_stock():
    s = blank_main()
    set_hand(s, 0, [4, 0, 0, 0, 0])
    trades = set(kinds(E.legal_actions(s), A.BANK_TRADE))
    assert trades == {(A.BANK_TRADE, B.WOOD, t) for t in range(5) if t != B.WOOD}
    s2 = E.apply(s, (A.BANK_TRADE, B.WOOD, B.ORE))
    assert s2.players[0].resources == [0, 0, 0, 0, 1] and s2.bank[B.WOOD] == 19 and s2.bank[B.ORE] == 18
    assert s2.players[0].hand_size == 1
    set_hand(s, 0, [3, 0, 0, 0, 0])
    assert not kinds(E.legal_actions(s), A.BANK_TRADE)
    with pytest.raises(Illegal):
        E.apply(s, (A.BANK_TRADE, B.WOOD, B.ORE))
    generic = next(v for v, t in B.STANDARD_PORTS.items() if t == B.PORT_GENERIC)
    s.players[0].settlements.append(generic)
    assert s.port_ratio(0, B.WOOD) == 3
    assert (A.BANK_TRADE, B.WOOD, B.ORE) in E.legal_actions(s)
    s3 = E.apply(s, (A.BANK_TRADE, B.WOOD, B.ORE))
    assert s3.players[0].resources == [0, 0, 0, 0, 1] and s3.bank[B.WOOD] == 19
    wood_port = next(v for v, t in B.STANDARD_PORTS.items() if t == B.WOOD)
    s.players[0].cities.append(wood_port)
    set_hand(s, 0, [2, 0, 0, 0, 0])
    assert s.port_ratio(0, B.WOOD) == 2
    s4 = E.apply(s, (A.BANK_TRADE, B.WOOD, B.WHEAT))
    assert s4.players[0].resources == [0, 0, 0, 1, 0]
    # the 2:1 port only helps for its resource; other resources use the 3:1
    set_hand(s, 0, [0, 2, 0, 0, 0])
    assert not kinds(E.legal_actions(s), A.BANK_TRADE)
    set_hand(s, 0, [0, 3, 0, 0, 0])
    assert (A.BANK_TRADE, B.BRICK, B.WOOD) in E.legal_actions(s)
    # bank stock
    set_hand(s, 0, [4, 0, 0, 0, 0])
    s.bank[B.ORE] = 0
    assert (A.BANK_TRADE, B.WOOD, B.ORE) not in E.legal_actions(s)
    assert (A.BANK_TRADE, B.WOOD, B.WHEAT) in E.legal_actions(s)
    with pytest.raises(Illegal):
        E.apply(s, (A.BANK_TRADE, B.WOOD, B.ORE))
    with pytest.raises(Illegal):
        E.apply(s, (A.BANK_TRADE, B.WOOD, B.WOOD))


def test_player_trade_protocol_responders_select_execute_cancel():
    s = blank_main()
    set_hand(s, 0, [2, 0, 0, 0, 0])
    set_hand(s, 1, [0, 1, 0, 0, 0])
    set_hand(s, 2, [0, 0, 0, 0, 0])
    set_hand(s, 3, [0, 1, 0, 0, 0])
    proposals = set(kinds(E.legal_actions(s), A.PROPOSE_TRADE))
    assert proposals == {(A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
                         (A.PROPOSE_TRADE, (2, 0, 0, 0, 0), (0, 1, 0, 0, 0))}
    give, get = (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)
    s = E.apply(s, (A.PROPOSE_TRADE, give, get))
    assert s.phase == PHASE_TRADE_RESPONSE and s.trades_this_turn == 1
    assert s.trade_responder == 1 and E.acting_player(s) == 1
    assert s.pending_trade is not None and s.pending_trade.responses == {2: False}   # 2 cannot pay: skipped
    assert E.legal_actions(s) == [(A.ACCEPT_TRADE,), (A.REJECT_TRADE,)]
    with pytest.raises(Illegal):
        E.apply(s, (A.END_TURN,))
    with pytest.raises(Illegal):
        E.apply(s, (A.EXECUTE_TRADE, 1))
    s = E.apply(s, (A.ACCEPT_TRADE,))
    assert s.phase == PHASE_TRADE_RESPONSE and s.trade_responder == 3 and E.acting_player(s) == 3
    s = E.apply(s, (A.REJECT_TRADE,))
    assert s.phase == PHASE_TRADE_SELECT and E.acting_player(s) == 0 and s.trade_responder == -1
    assert s.pending_trade.responses == {1: True, 2: False, 3: False}
    assert E.legal_actions(s) == [(A.EXECUTE_TRADE, 1), (A.CANCEL_TRADE,)]
    with pytest.raises(Illegal):
        E.apply(s, (A.EXECUTE_TRADE, 3))
    with pytest.raises(Illegal):
        E.apply(s, (A.ACCEPT_TRADE,))
    done = E.apply(s, (A.EXECUTE_TRADE, 1))
    assert done.players[0].resources == [1, 1, 0, 0, 0] and done.players[1].resources == [1, 0, 0, 0, 0]
    assert done.players[0].hand_size == 2 and done.players[1].hand_size == 1
    assert done.phase == PHASE_MAIN and done.pending_trade is None and done.trades_this_turn == 1
    cancelled = E.apply(s, (A.CANCEL_TRADE,))
    assert cancelled.phase == PHASE_MAIN and cancelled.pending_trade is None
    assert cancelled.players[0].resources == [2, 0, 0, 0, 0] and cancelled.players[1].resources == [0, 1, 0, 0, 0]


def test_player_trade_everyone_rejects_or_nobody_can_pay_returns_to_main():
    s = blank_main()
    set_hand(s, 0, [2, 0, 0, 0, 0])
    set_hand(s, 1, [0, 1, 0, 0, 0])
    set_hand(s, 3, [0, 1, 0, 0, 0])
    s2 = E.apply(s, (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    s2 = E.apply(s2, (A.REJECT_TRADE,))
    s2 = E.apply(s2, (A.REJECT_TRADE,))
    assert s2.phase == PHASE_MAIN and s2.pending_trade is None and s2.trades_this_turn == 1
    assert s2.players[0].resources == [2, 0, 0, 0, 0]
    # nobody holds sheep: no candidate is generated, but a hand-written offer is auto-rejected by all
    assert not any(a[2][B.SHEEP] for a in kinds(E.legal_actions(s), A.PROPOSE_TRADE))
    s3 = E.apply(s, (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 0, 1, 0, 0)))
    assert s3.phase == PHASE_MAIN and s3.pending_trade is None and s3.trades_this_turn == 1


def test_player_trade_limits_and_validation():
    s = blank_main()
    set_hand(s, 0, [4, 0, 0, 0, 0])
    offer = (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 0, 1, 0, 0))     # nobody can pay -> instant return
    for k in range(E.MAX_TRADE_PROPOSALS_PER_TURN):
        s = E.apply(s, offer)
        assert s.phase == PHASE_MAIN and s.trades_this_turn == k + 1
    assert E.MAX_TRADE_PROPOSALS_PER_TURN == 4
    set_hand(s, 1, [0, 0, 1, 0, 0])
    assert not kinds(E.legal_actions(s), A.PROPOSE_TRADE)
    with pytest.raises(Illegal):
        E.apply(s, offer)
    s = E.apply(s, (A.END_TURN,))
    assert s.trades_this_turn == 0
    # validation of hand-written offers
    s = blank_main()
    set_hand(s, 0, [1, 0, 0, 0, 0])
    set_hand(s, 1, [1, 1, 0, 0, 0])
    for bad in (
        (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (1, 0, 0, 0, 0)),     # not disjoint
        (A.PROPOSE_TRADE, (0, 0, 0, 0, 0), (0, 1, 0, 0, 0)),     # empty give
        (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)),     # empty get
        (A.PROPOSE_TRADE, (2, 0, 0, 0, 0), (0, 1, 0, 0, 0)),     # cannot afford
        (A.PROPOSE_TRADE, (1, 0, 0, 0), (0, 1, 0, 0, 0)),        # malformed
        (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, -1, 0, 0, 0)),    # negative
    ):
        with pytest.raises(Illegal):
            E.apply(s, bad)
    s.phase = PHASE_ROLL
    with pytest.raises(Illegal):
        E.apply(s, (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))   # only after rolling
    assert not kinds(E.legal_actions(s), A.PROPOSE_TRADE)


def test_trade_candidates_are_one_for_one_and_two_for_one_only():
    s = blank_main()
    set_hand(s, 0, [3, 1, 0, 0, 0])
    set_hand(s, 1, [0, 0, 1, 1, 1])
    cands = kinds(E.legal_actions(s), A.PROPOSE_TRADE)
    for _, give, get in cands:
        assert sum(get) == 1 and sum(give) in (1, 2) and max(give) == sum(give)
        assert all(not (give[r] and get[r]) for r in range(5))
        assert get[B.WOOD] == 0 and get[B.BRICK] == 0          # no opponent holds those
        assert s.players[0].can_afford(give)
    expect = 3 * 2 + 3 * 1     # wood: 1 or 2 for each of sheep/wheat/ore; brick: 1 for each
    assert len(cands) == len(set(cands)) == expect


# ---------------------------------------------------------------------------
# end of turn, winning, cap, queries
# ---------------------------------------------------------------------------
def test_end_turn_bookkeeping():
    s = blank_main()
    p = s.players[0]
    p.dev_cards_new = [1, 1, 0, 0, 0]
    p.dev_cards = [0, 0, 1, 0, 0]
    p.dev_count = 3
    s.dev_played_this_turn = True
    s.trades_this_turn = 3
    s.turn = 10
    assert (A.END_TURN,) in E.legal_actions(s)
    s2 = E.apply(s, (A.END_TURN,))
    q = s2.players[0]
    assert q.dev_cards == [1, 1, 1, 0, 0] and q.dev_cards_new == [0] * 5 and q.dev_count == 3
    assert not s2.dev_played_this_turn and s2.free_roads == 0 and s2.trades_this_turn == 0
    assert s2.turn == 11 and s2.current == 1 and s2.phase == PHASE_ROLL and s2.dice == 0
    assert s2.pending_trade is None and s2.trade_responder == -1
    s3 = blank_main()
    s3.current = 3
    assert E.apply(s3, (A.END_TURN,)).current == 0


def test_win_detected_during_own_turn_including_hidden_vp():
    s = blank_main()
    p = s.players[0]
    p.cities = [CORNER[0], CORNER[2], CORNER[4], B.HEX_VERTICES[FAR_A][0]]
    p.settlements = [B.HEX_VERTICES[FAR_B][0]]
    assert E.count_vp(s, 0) == 9 and not E.is_terminal(s)
    s.dev_deck = [0, 1, 0, 0, 0]
    set_hand(s, 0, [0, 0, 1, 1, 1])
    s2 = E.apply(s, (A.BUY_DEV,), random.Random(0))
    assert s2.players[0].dev_cards_new[B.DEV_VP] == 1
    assert E.count_vp(s2, 0) == 10 and E.count_vp(s2, 0, include_hidden=False) == 9
    assert E.is_terminal(s2) and s2.phase == PHASE_GAME_OVER and s2.winner == 0
    assert E.legal_actions(s2) == []
    with pytest.raises(Illegal):
        E.apply(s2, (A.END_TURN,))
    # a player who reaches 10 VP outside their own turn wins only when their turn comes
    s3 = blank_main()
    q = s3.players[1]
    q.cities = [CORNER[0], CORNER[2], CORNER[4], B.HEX_VERTICES[FAR_A][0]]
    q.settlements = [B.HEX_VERTICES[FAR_B][0], B.HEX_VERTICES[FAR_B][3]]
    assert E.count_vp(s3, 1) == 10
    set_hand(s3, 0, [4, 0, 0, 0, 0])
    s4 = E.apply(s3, (A.BANK_TRADE, B.WOOD, B.ORE))
    assert not E.is_terminal(s4)              # still player 0's turn
    s5 = E.apply(s4, (A.END_TURN,))
    assert E.is_terminal(s5) and s5.winner == 1


def test_longest_road_win_on_own_turn():
    s = blank_main()
    p = s.players[0]
    p.cities = [B.HEX_VERTICES[FAR_A][0], B.HEX_VERTICES[FAR_A][3], B.HEX_VERTICES[FAR_B][0]]
    p.settlements = [CORNER[0], B.HEX_VERTICES[FAR_B][3]]
    p.roads = RING[:4]
    set_hand(s, 0, [1, 1, 0, 0, 0])
    assert E.count_vp(s, 0) == 8
    s2 = E.apply(s, (A.BUILD_ROAD, RING[4]))
    assert s2.longest_road_owner == 0 and E.count_vp(s2, 0) == 10
    assert E.is_terminal(s2) and s2.winner == 0


def test_max_turns_cap_highest_vp_wins_ties_nobody():
    s = blank_main()
    s.turn = 41
    s.max_turns = 42
    s.players[2].settlements = [CORNER[0], CORNER[2]]
    s.players[0].settlements = [B.HEX_VERTICES[FAR_A][0]]
    s2 = E.apply(s, (A.END_TURN,))
    assert s2.phase == PHASE_GAME_OVER and s2.winner == 2 and s2.turn == 42
    s.players[1].cities = [B.HEX_VERTICES[FAR_B][0]]      # ties player 2 at 2 VP
    s3 = E.apply(s, (A.END_TURN,))
    assert s3.phase == PHASE_GAME_OVER and s3.winner == -1
    s.max_turns = 100
    assert not E.is_terminal(E.apply(s, (A.END_TURN,)))


def test_count_vp_components():
    s = blank_main()
    p = s.players[0]
    p.settlements = [CORNER[0]]
    p.cities = [CORNER[2], CORNER[4]]
    p.dev_cards = [0, 1, 0, 0, 0]
    p.dev_cards_new = [0, 1, 0, 0, 0]
    s.longest_road_owner = 0
    s.largest_army_owner = 0
    assert E.count_vp(s, 0) == 1 + 4 + 2 + 2 + 2
    assert E.count_vp(s, 0, include_hidden=False) == 9
    assert E.count_vp(s, 1) == 0
    assert E.count_vp(s, 0, include_hidden=False) == s.public_vp(0) and E.count_vp(s, 0) == s.total_vp(0)


def test_acting_player_by_phase():
    s = blank_main()
    s.current = 2
    assert E.acting_player(s) == 2
    s.phase = PHASE_DISCARD
    s.discard_queue = [3, 0]
    assert E.acting_player(s) == 3
    s.phase = PHASE_TRADE_RESPONSE
    s.trade_responder = 1
    assert E.acting_player(s) == 1
    for ph in (PHASE_ROLL, PHASE_ROBBER, PHASE_TRADE_SELECT, PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        s.phase = ph
        assert E.acting_player(s) == 2


def test_apply_never_mutates_input_and_rejected_actions_leave_state_untouched():
    rng = random.Random(21)
    s = play_setup(new_game(4, random.Random(21)), rng)
    for _ in range(300):
        before = s.to_dict()
        acts = E.legal_actions(s)
        a = rng.choice(acts)
        nxt = E.apply(s, a, random.Random(1))
        assert s.to_dict() == before
        assert nxt is not s and nxt.players[0] is not s.players[0]
        for bad in ((A.BUILD_CITY, 999), (A.SETUP_SETTLEMENT, 0), ("nonsense",), (A.MOVE_ROBBER, 0, 7)):
            with pytest.raises(Illegal):
                E.apply_inplace(s, bad, rng)
            assert s.to_dict() == before
        E.apply_inplace(s, a, rng)
        if E.is_terminal(s):
            break


def test_buildable_helpers_and_setup_variant():
    s = blank_main()
    s.players[0].settlements.append(CORNER[0])
    s.players[0].roads.extend([RING[0], RING[1]])
    s.players[1].settlements.append(CORNER[4])
    assert E.buildable_settlement_vertices(s, 0) == [CORNER[2]]
    free = E.buildable_settlement_vertices(s, 0, setup=True)
    blocked = {CORNER[0], CORNER[4]} | set(B.VERTEX_NEIGHBORS[CORNER[0]]) | set(B.VERTEX_NEIGHBORS[CORNER[4]])
    assert set(free) == set(range(B.NUM_VERTICES)) - blocked
    assert E.buildable_road_edges(s, 1) == sorted(B.VERTEX_EDGES[CORNER[4]])
    assert E.buildable_city_vertices(s, 1) == [CORNER[4]]
    assert E.buildable_road_edges(s, 2) == [] and E.buildable_settlement_vertices(s, 2) == []


def test_hand_size_and_dev_count_stay_in_sync_over_a_game():
    rng = random.Random(4)
    s = new_game(4, rng)
    while not E.is_terminal(s):
        E.apply_inplace(s, rng.choice(E.legal_actions(s)), rng)
        for p in s.players:
            assert p.hand_size == sum(p.resources)
            assert p.dev_count == sum(p.dev_cards) + sum(p.dev_cards_new)


def test_random_playout_terminates_and_respects_max_turns():
    rng = random.Random(9)
    start = new_game(4, rng)
    end = E.random_playout(start, rng, max_turns=60)
    assert E.is_terminal(end) and end.phase == PHASE_GAME_OVER
    assert end.turn <= 60 and start.phase == PHASE_SETUP_SETTLEMENT   # input untouched
    end2 = E.random_playout(new_game(3, random.Random(9)), random.Random(9), max_turns=400)
    assert E.is_terminal(end2)
    if end2.winner >= 0 and end2.turn < 400:
        assert E.count_vp(end2, end2.winner) >= B.VP_TO_WIN


# ---------------------------------------------------------------------------
# fuzzing: invariants after every action, legal_actions soundness & completeness
# ---------------------------------------------------------------------------
_PIECE_ACTIONS = {A.SETUP_SETTLEMENT, A.SETUP_ROAD, A.BUILD_ROAD, A.BUILD_SETTLEMENT, A.BUILD_CITY}
_AWARD_ACTIONS = _PIECE_ACTIONS | {A.PLAY_KNIGHT}
_OTHER_DEV = {A.PLAY_ROAD_BUILDING, A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY}


def _check_cards(s: GameState, played_other: int) -> None:
    assert sum(s.bank) + sum(sum(p.resources) for p in s.players) == 95
    assert min(s.bank) >= 0
    held = 0
    for p in s.players:
        assert min(p.resources) >= 0 and min(p.dev_cards) >= 0 and min(p.dev_cards_new) >= 0
        assert p.hand_size == sum(p.resources)
        assert p.dev_count == sum(p.dev_cards) + sum(p.dev_cards_new)
        held += p.dev_count + p.played_knights
    assert min(s.dev_deck) >= 0
    assert sum(s.dev_deck) + held + played_other == 25
    assert 0 <= s.robber < B.NUM_HEXES


def _check_board(s: GameState) -> None:
    occ: Dict[int, int] = {}
    for i, p in enumerate(s.players):
        assert len(p.roads) <= B.MAX_ROADS
        assert len(p.settlements) <= B.MAX_SETTLEMENTS
        assert len(p.cities) <= B.MAX_CITIES
        for v in p.settlements + p.cities:
            assert v not in occ, "two buildings on one vertex"
            occ[v] = i
    for v in occ:
        for w in B.VERTEX_NEIGHBORS[v]:
            assert w not in occ, "adjacent buildings"
    eo: Dict[int, int] = {}
    for i, p in enumerate(s.players):
        for e in p.roads:
            assert e not in eo, "two roads on one edge"
            eo[e] = i
    for p in s.players:
        if not p.roads:
            continue
        roads = set(p.roads)
        seen_v = set(p.settlements + p.cities)
        frontier = list(seen_v)
        reached: Set[int] = set()
        while frontier:
            v = frontier.pop()
            for e in B.VERTEX_EDGES[v]:
                if e in roads and e not in reached:
                    reached.add(e)
                    a, b = B.EDGE_VERTICES[e]
                    w = b if a == v else a
                    if w not in seen_v:
                        seen_v.add(w)
                        frontier.append(w)
        assert reached == roads, "road not connected to an own building"


def _check_awards(s: GameState) -> None:
    n = len(s.players)
    lens = [E.longest_road_length(s, i) for i in range(n)]
    o = s.longest_road_owner
    if o >= 0:
        assert lens[o] >= 5 and lens[o] == s.longest_road_len
        assert all(lens[j] <= lens[o] for j in range(n))
    else:
        assert s.longest_road_len == 0
        mx = max(lens)
        assert mx < 5 or lens.count(mx) >= 2
    ks = [p.played_knights for p in s.players]
    la = s.largest_army_owner
    if la >= 0:
        assert ks[la] >= 3 and ks[la] == max(ks)
    else:
        assert max(ks) < 3


@pytest.mark.parametrize("seed", list(range(30)))
def test_fuzz_random_playout_invariants(seed: int):
    rng = random.Random(seed)
    n = 3 if seed % 3 == 2 else 4
    s = new_game(n, rng)
    s.max_turns = 400
    played_other = 0
    steps = 0
    saw_main = False
    while not E.is_terminal(s):
        acts = E.legal_actions(s)
        assert acts, f"no legal action in phase {s.phase}"
        actor = E.acting_player(s)
        assert 0 <= actor < n
        if s.phase == PHASE_MAIN:
            assert (A.END_TURN,) in acts
            saw_main = True
        assert len(set(acts)) == len(acts), "duplicate legal actions"
        a = rng.choice(acts)
        E.apply_inplace(s, a, rng)
        steps += 1
        if a[0] in _OTHER_DEV:
            played_other += 1
        _check_cards(s, played_other)
        if a[0] in _PIECE_ACTIONS:
            _check_board(s)
        if a[0] in _AWARD_ACTIONS:
            _check_awards(s)
        assert steps < 200_000
    assert saw_main and s.phase == PHASE_GAME_OVER and E.legal_actions(s) == []
    _check_board(s)
    _check_awards(s)
    if s.turn < 400:
        assert s.winner >= 0 and E.count_vp(s, s.winner) >= B.VP_TO_WIN
        assert s.winner == s.current
    else:
        vps = [E.count_vp(s, i) for i in range(n)]
        best = max(vps)
        assert s.winner == (vps.index(best) if vps.count(best) == 1 else -1)


def _action_universe(n: int) -> List[A.Action]:
    acts: List[A.Action] = [(A.ROLL,), (A.END_TURN,), (A.BUY_DEV,), (A.PLAY_ROAD_BUILDING,),
                            (A.ACCEPT_TRADE,), (A.REJECT_TRADE,), (A.CANCEL_TRADE,)]
    acts += [(A.SETUP_SETTLEMENT, v) for v in range(B.NUM_VERTICES)]
    acts += [(A.SETUP_ROAD, e) for e in range(B.NUM_EDGES)]
    acts += [(A.BUILD_ROAD, e) for e in range(B.NUM_EDGES)]
    acts += [(A.BUILD_SETTLEMENT, v) for v in range(B.NUM_VERTICES)]
    acts += [(A.BUILD_CITY, v) for v in range(B.NUM_VERTICES)]
    acts += [(A.MOVE_ROBBER, h, i) for h in range(B.NUM_HEXES) for i in range(-1, n)]
    acts += [(A.PLAY_KNIGHT, h, i) for h in range(B.NUM_HEXES) for i in range(-1, n)]
    acts += [(A.PLAY_YEAR_OF_PLENTY, r1, r2) for r1 in range(5) for r2 in range(r1, 5)]
    acts += [(A.PLAY_MONOPOLY, r) for r in range(5)]
    acts += [(A.BANK_TRADE, g, t) for g in range(5) for t in range(5)]
    acts += [(A.EXECUTE_TRADE, i) for i in range(n)]
    return acts


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_legal_actions_sound_and_complete(seed: int):
    """Every listed action applies; every unlisted candidate from a fixed universe is rejected.

    ``PROPOSE_TRADE`` (bounded candidate set), forced rolls and discards beyond
    the enumeration cap are the documented exceptions and are not in the universe.
    """
    rng = random.Random(seed)
    n = 3 if seed == 102 else 4
    s = new_game(n, rng)
    s.max_turns = 150
    universe = _action_universe(n)
    step = 0
    while not E.is_terminal(s):
        acts = E.legal_actions(s)
        step += 1
        if step % 4 == 0:
            for a in acts:                                      # soundness
                E.apply(s, a, random.Random(step))
        if step % 25 == 0:
            legal = set(acts)
            for a in universe:                                  # completeness
                ok = True
                try:
                    E.apply(s, a, random.Random(step))
                except Illegal:
                    ok = False
                assert ok == (a in legal), (s.phase, a, ok)
        E.apply_inplace(s, rng.choice(acts), rng)
