"""Tests for catanbot.vision.schema: parse dict <-> GameState and validation."""
from __future__ import annotations

import copy
import json

import pytest

from catanbot import board as B
from catanbot.state import PHASE_MAIN, PLAYER_COLORS, GameState, Player
from catanbot.vision import schema as S


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_midgame_state() -> GameState:
    """Legal 4-player mid-game position built from the board tables."""
    hexes = list(B.STANDARD_HEXES)
    st = GameState()
    st.hexes = hexes
    st.robber = 13
    st.ports = dict(B.STANDARD_PORTS)
    st.players = [Player(color=c, name=n) for c, n in zip(PLAYER_COLORS, ["Alice", "Bob", "Carol", "Dave"])]
    order = sorted(range(B.NUM_VERTICES), key=lambda v: -B.vertex_pip_total(v, hexes))
    occupied: dict = {}

    def free(v: int) -> bool:
        return v not in occupied and all(n not in occupied for n in B.VERTEX_NEIGHBORS[v])

    for rnd in range(3):
        seats = range(4) if rnd % 2 == 0 else range(3, -1, -1)
        for i in seats:
            v = next(v for v in order if free(v))
            occupied[v] = i
            st.players[i].settlements.append(v)
    used_edges: set = set()
    for i, p in enumerate(st.players):
        for v in list(p.settlements):
            e = next(e for e in B.VERTEX_EDGES[v] if e not in used_edges)
            used_edges.add(e)
            p.roads.append(e)
    # extend player 0's first road into a chain of 6 (longest road)
    p0 = st.players[0]
    cur_v = p0.settlements[0]
    path_edges = [p0.roads[0]]
    a, b = B.EDGE_VERTICES[path_edges[0]]
    cur_v = b if a == cur_v else a
    while len(path_edges) < 6:
        nxt = None
        for e in B.VERTEX_EDGES[cur_v]:
            if e in used_edges:
                continue
            ea, eb = B.EDGE_VERTICES[e]
            other = eb if ea == cur_v else ea
            if other in occupied and occupied[other] != 0:
                continue
            nxt = (e, other)
            break
        assert nxt is not None
        used_edges.add(nxt[0])
        path_edges.append(nxt[0])
        cur_v = nxt[1]
    p0.roads = sorted(set(p0.roads) | set(path_edges))
    for i in (0, 1):
        p = st.players[i]
        p.cities.append(p.settlements.pop(0))
    st.players[0].resources = [1, 1, 0, 2, 0]
    st.players[0].dev_cards = [1, 0, 0, 0, 0]
    st.players[0].played_knights = 1
    st.players[1].played_knights = 2
    st.players[3].played_knights = 3
    for i in (1, 2, 3):
        p = st.players[i]
        p.hand_known = False
        p.hand_size = [0, 6, 3, 8][i]
        p.dev_known = False
        p.dev_count = [0, 2, 0, 1][i]
    st.longest_road_owner = 0
    st.longest_road_len = 6
    st.largest_army_owner = 3
    st.current = 0
    st.phase = PHASE_MAIN
    st.dice = 8
    st.bank = [17, 18, 19, 16, 19]
    st.dev_deck = [7, 4, 1, 2, 2]
    return st


# ---------------------------------------------------------------------------
# schema + example
# ---------------------------------------------------------------------------
def test_schema_is_json_and_example_matches_shape():
    txt = json.dumps(S.PARSE_SCHEMA)
    assert "hexes" in S.PARSE_SCHEMA["properties"]
    assert S.PARSE_SCHEMA["properties"]["hexes"]["minItems"] == 19
    for key in S.PARSE_SCHEMA["required"]:
        assert key in S.EXAMPLE_PARSED
    json.loads(json.dumps(S.EXAMPLE_PARSED))
    assert len(txt) > 100


def test_example_validates_against_jsonschema_if_available():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(S.EXAMPLE_PARSED, S.PARSE_SCHEMA)


def test_example_has_no_warnings_and_converts():
    assert S.validate(S.EXAMPLE_PARSED) == []
    st = S.parsed_to_state(S.EXAMPLE_PARSED)
    assert st.num_players == 4
    assert st.phase == PHASE_MAIN
    assert st.current == 0
    assert st.dice == 8
    assert st.robber == 13
    assert st.hexes == list(B.STANDARD_HEXES)
    assert st.ports == B.STANDARD_PORTS
    me = st.players[0]
    assert me.hand_known and me.resources == [1, 1, 0, 2, 0] and me.hand_size == 4
    for p in st.players[1:]:
        assert not p.hand_known and not p.dev_known
    assert [p.hand_size for p in st.players] == [4, 6, 3, 8]
    assert [p.dev_count for p in st.players] == [1, 2, 0, 1]
    assert [p.played_knights for p in st.players] == [1, 2, 0, 3]
    assert st.longest_road_owner == 0 and st.longest_road_len >= 5
    assert st.largest_army_owner == 3


def test_priors_bank_and_dev_deck():
    st = S.parsed_to_state(S.EXAMPLE_PARSED)
    # bank = 19 minus my known cards only (opponents' hands are unknown by type)
    assert st.bank == [18, 18, 19, 17, 19]
    # dev deck: 25 - 6 played knights - 4 held (unknown type, apportioned)
    assert sum(st.dev_deck) == 25 - 6 - 4
    assert st.dev_deck[B.DEV_KNIGHT] <= 14 - 6
    assert all(x >= 0 for x in st.dev_deck)
    # explicit dev_deck_remaining overrides the held-card inference
    parsed = copy.deepcopy(S.EXAMPLE_PARSED)
    parsed["dev_deck_remaining"] = 10
    st2 = S.parsed_to_state(parsed)
    assert sum(st2.dev_deck) == 10
    # explicit bank is used verbatim
    parsed["bank"] = {"wood": 3, "brick": 0, "sheep": 5, "wheat": 2, "ore": 9}
    assert S.parsed_to_state(parsed).bank == [3, 0, 5, 2, 9]


def test_bank_never_below_zero():
    parsed = copy.deepcopy(S.EXAMPLE_PARSED)
    parsed["players"][0]["resources"] = {"wood": 25, "brick": 0, "sheep": 0, "wheat": 0, "ore": 0}
    st = S.parsed_to_state(parsed)
    assert st.bank[B.WOOD] == 0 and st.bank[B.BRICK] == 19


def test_aliases_and_missing_optionals():
    parsed = copy.deepcopy(S.EXAMPLE_PARSED)
    names = {"wood": "Lumber", "brick": "hills", "sheep": "Wool", "wheat": "grain", "ore": "Mountains"}
    for h in parsed["hexes"]:
        h["resource"] = names.get(h["resource"], h["resource"])
    del parsed["ports"]
    del parsed["robber"]
    del parsed["current_player"]
    del parsed["dice"]
    parsed["me"] = "blue"
    parsed["players"][0].pop("resources")
    parsed["players"][1]["resources"] = {"wool": 2, "grain": 1}
    assert S.validate(parsed) == []
    st = S.parsed_to_state(parsed)
    assert st.hexes == list(B.STANDARD_HEXES)
    assert st.ports == B.STANDARD_PORTS
    assert st.robber == 9  # desert
    assert st.current == 1  # defaults to me
    assert st.dice == 0
    assert st.players[1].hand_known and st.players[1].resources == [0, 0, 2, 1, 0]
    assert not st.players[0].hand_known and st.players[0].hand_size == 4


def test_me_dev_cards_by_type():
    parsed = copy.deepcopy(S.EXAMPLE_PARSED)
    parsed["players"][0]["dev_cards"] = {"knight": 1, "victory_point": 1}
    parsed["players"][0]["vp"] = 7
    assert S.validate(parsed) == []
    st = S.parsed_to_state(parsed)
    assert st.players[0].dev_known and st.players[0].dev_cards == [1, 1, 0, 0, 0]
    assert st.total_vp(0) == 7
    assert st.dev_deck[B.DEV_VP] <= 4


def test_port_type_from_name():
    assert S.port_type_from_name("3:1") == B.PORT_GENERIC
    assert S.port_type_from_name("generic") == B.PORT_GENERIC
    assert S.port_type_from_name("wood") == B.WOOD
    assert S.port_type_from_name("2:1 ore") == B.ORE
    assert S.port_type_from_name("Wool port") == B.SHEEP
    assert S.port_type_from_name("desert") is None
    assert S.port_type_from_name("bogus") is None
    assert S.resource_from_name("Grain") == B.WHEAT


def test_longest_road_length():
    assert S.longest_road_length([]) == 0
    vs = B.HEX_VERTICES[9]
    ring = [B.edge_between(vs[k], vs[(k + 1) % 6]) for k in range(6)]
    assert S.longest_road_length(ring) == 6
    assert S.longest_road_length(ring[:4]) == 4
    # an opponent building in the middle of a 4-chain breaks it into 2 + 2
    mid = vs[2]
    assert S.longest_road_length(ring[:4], blocked=[mid]) == 2


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------
def test_round_trip_state_parsed_state():
    st = make_midgame_state()
    parsed = S.state_to_parsed(st, me=0)
    assert S.validate(parsed) == [], S.validate(parsed)
    json.dumps(parsed)
    st2 = S.parsed_to_state(parsed)
    assert st2.hexes == st.hexes
    assert st2.robber == st.robber
    assert st2.ports == st.ports
    assert st2.current == st.current
    assert st2.dice == st.dice
    assert st2.bank == st.bank
    assert sum(st2.dev_deck) == sum(st.dev_deck)
    assert st2.longest_road_owner == st.longest_road_owner
    assert st2.largest_army_owner == st.largest_army_owner
    assert st2.longest_road_len == st.longest_road_len
    for a, b in zip(st.players, st2.players):
        assert a.color == b.color and a.name == b.name
        assert sorted(a.settlements) == sorted(b.settlements)
        assert sorted(a.cities) == sorted(b.cities)
        assert sorted(a.roads) == sorted(b.roads)
        assert a.played_knights == b.played_knights
        assert (sum(a.resources) if a.hand_known else a.hand_size) == b.hand_size
        assert (a.total_dev if a.dev_known else a.dev_count) == b.dev_count
    assert st2.players[0].hand_known and st2.players[0].resources == st.players[0].resources
    assert st2.players[0].dev_known and st2.players[0].dev_cards == st.players[0].dev_cards
    # and back again: parsed -> state -> parsed is stable
    assert S.state_to_parsed(st2, me=0) == parsed


def test_state_to_parsed_me_by_colour_and_public_vp():
    st = make_midgame_state()
    parsed = S.state_to_parsed(st, me="blue")
    assert parsed["me"] == "blue"
    assert "resources" not in parsed["players"][0]
    assert parsed["players"][0]["vp"] == st.public_vp(0)
    assert len(parsed["ports"]) == 9
    assert {p["edge"] for p in parsed["ports"]} == {e for e, _ in B.STANDARD_PORT_EDGES}


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def _example():
    return copy.deepcopy(S.EXAMPLE_PARSED)


def test_validate_flags_wrong_number_multiset():
    parsed = _example()
    parsed["hexes"][0]["number"] = 11  # was 10 -> three 11s, one 10
    w = S.validate(parsed)
    assert any("number multiset" in x for x in w), w
    assert any("missing [10]" in x and "extra [11]" in x for x in w), w


def test_validate_flags_wrong_resource_multiset_and_desert_number():
    parsed = _example()
    parsed["hexes"][0]["resource"] = "wood"  # 5 wood, 2 ore
    parsed["hexes"][9]["number"] = 7
    w = S.validate(parsed)
    assert any("resource multiset" in x for x in w), w
    assert any("desert has a number" in x for x in w), w


def test_validate_flags_adjacent_buildings():
    parsed = _example()
    v = parsed["players"][1]["settlements"][0]
    nb = B.VERTEX_NEIGHBORS[v][0]
    parsed["players"][2]["settlements"].append(nb)
    parsed["players"][2]["roads"].append(B.edge_between(v, nb))
    w = S.validate(parsed)
    assert any("distance rule" in x for x in w), w
    assert not any("not connected" in x for x in w), w


def test_validate_flags_unknown_colour():
    parsed = _example()
    parsed["players"][2]["color"] = "teal"
    w = S.validate(parsed)
    assert any("unknown colour 'teal'" in x for x in w), w


def test_validate_flags_bad_ids_overlaps_and_disconnected_roads():
    parsed = _example()
    parsed["players"][0]["settlements"].append(99)
    parsed["players"][1]["roads"].append(200)
    parsed["players"][2]["roads"].append(parsed["players"][3]["roads"][0])
    # a road far away from everything owned by orange
    far_edge = next(e for e in range(B.NUM_EDGES)
                    if all(e not in p["roads"] for p in parsed["players"])
                    and all(v not in parsed["players"][2]["settlements"] for v in B.EDGE_VERTICES[e])
                    and all(v not in parsed["players"][2]["cities"] for v in B.EDGE_VERTICES[e])
                    and not any(e2 in parsed["players"][2]["roads"] for e2 in B.EDGE_NEIGHBORS[e]))
    parsed["players"][2]["roads"].append(far_edge)
    w = S.validate(parsed)
    assert any("settlement id 99" in x for x in w), w
    assert any("road id 200" in x for x in w), w
    assert any("overlaps road" in x for x in w), w
    assert any("not connected" in x for x in w), w


def test_validate_flags_vp_and_award_inconsistencies():
    parsed = _example()
    parsed["players"][2]["vp"] = 1          # has 3 settlements
    parsed["players"][1]["largest_army"] = True  # only 2 knights, and Dave already has it
    parsed["players"][2]["longest_road"] = True  # 3 roads
    parsed["me"] = "teal"
    parsed["dice"] = 13
    w = S.validate(parsed)
    assert any("vp 1 is less than" in x for x in w), w
    assert any("largest army flagged with only 2" in x for x in w), w
    assert any("flagged for several players" in x for x in w), w
    assert any("shorter than 5" in x for x in w), w
    assert any("'me' colour" in x for x in w), w
    assert any("dice value" in x for x in w), w


def test_validate_flags_ports_off_coast():
    parsed = _example()
    parsed["ports"][0]["edge"] = B.HEX_EDGES[9][0]  # interior edge
    parsed["ports"][1]["type"] = "gold"
    w = S.validate(parsed)
    assert any("not a coastal edge" in x for x in w), w
    assert any("unknown type 'gold'" in x for x in w), w


def test_validate_handles_garbage_without_raising():
    assert S.validate("nope")
    assert S.validate({})
    assert S.validate({"hexes": [None] * 19, "players": [{"color": 5}]})
    st = S.parsed_to_state({"hexes": [], "players": []})
    assert st.num_players == 0 and len(st.hexes) == 19
