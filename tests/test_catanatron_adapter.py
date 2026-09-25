"""Tests for the catanatron <-> catanbot adapter (skipped when catanatron is not installed)."""
from __future__ import annotations

import pytest

pytest.importorskip("catanatron")

from catanatron.game import TURNS_LIMIT  # noqa: E402
from catanatron.models.enums import CITY, ROAD, SETTLEMENT, ActionPrompt, ActionType  # noqa: E402
from catanatron.models.map import CatanMap  # noqa: E402
from catanatron.models.player import Color  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402

from catanbot import actions as A  # noqa: E402
from catanbot import board as B  # noqa: E402
from catanbot import engine as E  # noqa: E402
from catanbot.bench import catanatron_adapter as AD  # noqa: E402
from catanbot.state import (  # noqa: E402
    PHASE_DISCARD,
    PHASE_MAIN,
    PHASE_ROBBER,
    PHASE_ROLL,
    PHASE_SETUP_ROAD,
    PHASE_SETUP_SETTLEMENT,
)

ALL_TYPES = set(ActionType)


def fresh_game(seed: int = 123, seat_players=None):
    players = seat_players or [WeightedRandomPlayer(c) for c in AD.COLORS]
    return AD.make_game(players, seed=seed)


# ---------------------------------------------------------------------------
# Board mapping
# ---------------------------------------------------------------------------
def test_tile_node_edge_bijection_and_incidences():
    g = fresh_game()
    cmap = g.state.board.map
    m = AD.derive_mapping(cmap)
    assert AD.verify_mapping(cmap, m) == []
    assert sorted(m.tile_to_hex.values()) == list(range(B.NUM_HEXES))
    assert sorted(m.coord_to_hex.values()) == list(range(B.NUM_HEXES))
    assert sorted(m.node_to_vertex.values()) == list(range(B.NUM_VERTICES))
    assert sorted(m.vertex_to_node) == sorted(m.node_to_vertex)
    assert sorted(set(m.edge_to_id.values())) == list(range(B.NUM_EDGES))
    assert len(m.edge_to_id) == 2 * B.NUM_EDGES          # both orientations
    assert all(a < b for a, b in m.id_to_edge)
    for h in range(B.NUM_HEXES):
        assert m.coord_to_hex[m.hex_to_coord[h]] == h
        assert m.tile_to_hex[m.hex_to_tile[h]] == h
    # A node shared by tiles A, B, C maps to the vertex shared by hexes f(A), f(B), f(C).
    for node, tiles in cmap.adjacent_tiles.items():
        assert {m.tile_to_hex[t.id] for t in tiles} == set(B.VERTEX_HEXES[m.node_to_vertex[node]])
    # Every tile's six nodes / edges are exactly the mapped hex's corners / sides.
    for coord, tile in cmap.land_tiles.items():
        h = m.coord_to_hex[coord]
        assert {m.node_to_vertex[n] for n in tile.nodes.values()} == set(B.HEX_VERTICES[h])
        assert {m.edge_to_id[e] for e in tile.edges.values()} == set(B.HEX_EDGES[h])
        for a, b in tile.edges.values():
            assert m.edge_to_id[(a, b)] == m.edge_to_id[(b, a)] == B.edge_between(m.node_to_vertex[a], m.node_to_vertex[b])
    # Ports sit on coastal edges: both nodes of a port are the endpoints of one coastal catanbot edge.
    for port in cmap.ports_by_id.values():
        from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
        ra, rb = PORT_DIRECTION_TO_NODEREFS[port.direction]
        e = B.edge_between(m.node_to_vertex[port.nodes[ra]], m.node_to_vertex[port.nodes[rb]])
        assert len(B.EDGE_HEXES[e]) == 1


def test_natural_orientation_is_preferred():
    """With catanatron's own coordinates the identity symmetry fits: NORTH = corner 0, clockwise."""
    g = fresh_game()
    cmap = g.state.board.map
    m = AD.derive_mapping(cmap)
    assert m.symmetry == (0, False)
    assert m.coord_to_hex[(0, 0, 0)] == B.HEX_INDEX[(0, 0)]
    for coord, tile in cmap.land_tiles.items():
        h = m.coord_to_hex[coord]
        for k, ref in enumerate(AD.NODE_REFS):
            assert m.node_to_vertex[tile.nodes[ref]] == B.HEX_VERTICES[h][k]


def test_mapping_survives_rotated_and_mirrored_coordinates():
    """Re-keying the tiles by a board symmetry must still yield a verified bijection.

    The 19 coordinates are invariant under every symmetry, so the identity still
    fits; the tiles simply land on the rotated / mirrored hexes and the node
    mapping follows them consistently.
    """
    g = fresh_game()
    orig = g.state.board.map
    identity = AD.derive_mapping(orig)
    for (k, mirrored), f in AD.board_symmetries():
        if (k, mirrored) == (0, False):
            continue
        rotated = CatanMap.from_tiles({f(coord): tile for coord, tile in orig.tiles.items()})
        m = AD.derive_mapping(rotated)
        assert AD.verify_mapping(rotated, m) == []
        assert sorted(m.tile_to_hex.values()) == list(range(B.NUM_HEXES))
        assert m.tile_to_hex != identity.tile_to_hex
        for coord, tile in rotated.land_tiles.items():
            assert m.tile_to_hex[tile.id] == B.HEX_INDEX[(coord[0], coord[2])]
        for node, tiles in rotated.adjacent_tiles.items():
            assert {m.tile_to_hex[t.id] for t in tiles} == set(B.VERTEX_HEXES[m.node_to_vertex[node]])


def test_non_identity_symmetry_search(monkeypatch):
    """When the natural orientation is unavailable the search falls back to another symmetry."""
    g = fresh_game()
    cmap = g.state.board.map
    original = AD.board_symmetries
    for drop in (1, 6, 9):
        monkeypatch.setattr(AD, "board_symmetries", lambda drop=drop: original()[drop:])
        m = AD.derive_mapping(cmap)
        assert m.symmetry == original()[drop][0] and m.symmetry != (0, False)
        assert AD.verify_mapping(cmap, m) == []
        for node, tiles in cmap.adjacent_tiles.items():
            assert {m.tile_to_hex[t.id] for t in tiles} == set(B.VERTEX_HEXES[m.node_to_vertex[node]])
        # a rotation keeps the corner order cyclic, a mirror reverses it
        k, mirrored = m.symmetry
        for coord, tile in cmap.land_tiles.items():
            h = m.coord_to_hex[coord]
            corners = [B.HEX_VERTICES[h].index(m.node_to_vertex[tile.nodes[ref]]) for ref in AD.NODE_REFS]
            step = (corners[1] - corners[0]) % 6
            assert step == (5 if mirrored else 1)
            assert all((corners[i + 1] - corners[i]) % 6 == step for i in range(5))


def test_mapping_is_cached_per_board_structure():
    g1, g2 = fresh_game(1), fresh_game(2)
    assert AD.mapping_for(g1.state.board.map) is AD.mapping_for(g2.state.board.map)


# ---------------------------------------------------------------------------
# State conversion
# ---------------------------------------------------------------------------
def _check_conversion(g, m):
    st = g.state
    cb = AD.to_catanbot_state(g, st.current_color(), m)
    ps = st.player_state
    # board
    for coord, tile in st.board.map.land_tiles.items():
        res, num = cb.hexes[m.coord_to_hex[coord]]
        if tile.resource is None:
            assert res == B.DESERT and num == 0
        else:
            assert res == AD.RESOURCE_TO_CB[tile.resource] and num == tile.number
    assert cb.robber == m.coord_to_hex[st.board.robber_coordinate]
    for res, nodes in st.board.map.port_nodes.items():
        t = B.PORT_GENERIC if res is None else AD.RESOURCE_TO_CB[res]
        for n in nodes:
            assert cb.ports[m.node_to_vertex[n]] == t
    assert len(cb.ports) == 18
    # players
    assert [p.color for p in cb.players] == [AD.COLOR_NAMES[c] for c in st.colors]
    for i, color in enumerate(st.colors):
        key = f"P{i}"
        p = cb.players[i]
        bb = st.buildings_by_color[color]
        assert sorted(p.settlements) == sorted(m.node_to_vertex[n] for n in bb[SETTLEMENT])
        assert sorted(p.cities) == sorted(m.node_to_vertex[n] for n in bb[CITY])
        assert sorted(p.roads) == sorted(m.edge_to_id[e] for e in bb[ROAD])
        assert len(p.settlements) == B.MAX_SETTLEMENTS - ps[f"{key}_SETTLEMENTS_AVAILABLE"]
        assert len(p.cities) == B.MAX_CITIES - ps[f"{key}_CITIES_AVAILABLE"]
        assert len(p.roads) == B.MAX_ROADS - ps[f"{key}_ROADS_AVAILABLE"]
        assert p.resources == [ps[f"{key}_{r}_IN_HAND"] for r in AD.CB_TO_RESOURCE]
        assert p.dev_cards == [ps[f"{key}_{d}_IN_HAND"] for d in AD.CB_TO_DEV]
        assert p.dev_cards_new == [0] * 5
        assert p.played_knights == ps[f"{key}_PLAYED_KNIGHT"]
        assert p.hand_known and p.dev_known and p.hand_size == sum(p.resources) and p.dev_count == sum(p.dev_cards)
        assert (cb.longest_road_owner == i) == bool(ps[f"{key}_HAS_ROAD"])
        assert (cb.largest_army_owner == i) == bool(ps[f"{key}_HAS_ARMY"])
        assert cb.total_vp(i) == ps[f"{key}_ACTUAL_VICTORY_POINTS"]
        assert cb.public_vp(i) == ps[f"{key}_VICTORY_POINTS"]
    # supplies
    assert cb.bank == list(st.resource_freqdeck)
    assert sum(cb.dev_deck) == len(st.development_listdeck)
    assert cb.dev_deck[B.DEV_KNIGHT] == st.development_listdeck.count("KNIGHT")
    assert cb.dev_deck[B.DEV_VP] == st.development_listdeck.count("VICTORY_POINT")
    # turn structure
    assert cb.current == st.current_turn_index or cb.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)
    assert st.colors[E.acting_player(cb)] == st.current_color()
    assert cb.turn == st.num_turns and cb.max_turns > cb.turn
    cur = f"P{st.current_turn_index}"
    assert cb.dev_played_this_turn == ps[f"{cur}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
    prompt = st.current_prompt
    if prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        assert cb.phase == PHASE_SETUP_SETTLEMENT
    elif prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        assert cb.phase == PHASE_SETUP_ROAD
        assert cb.setup_last_settlement == m.node_to_vertex[st.buildings_by_color[st.current_color()][SETTLEMENT][-1]]
    elif prompt == ActionPrompt.DISCARD:
        assert cb.phase == PHASE_DISCARD and cb.dice == 7
        assert cb.discard_queue[0] == st.current_player_index
        assert all(cb.players[j].total_resources > 7 for j in cb.discard_queue)
    elif prompt == ActionPrompt.MOVE_ROBBER:
        assert cb.phase == PHASE_ROBBER
    else:
        assert cb.phase == (PHASE_MAIN if ps[f"{cur}_HAS_ROLLED"] else PHASE_ROLL)
        assert (cb.dice > 0) == bool(ps[f"{cur}_HAS_ROLLED"])
        assert cb.free_roads == (st.free_roads_available if st.is_road_building else 0)
    legal = E.legal_actions(cb)
    assert legal, cb.phase
    assert not any(a[0] == A.PROPOSE_TRADE for a in legal)
    return cb, legal


def test_converted_states_match_catanatron_through_a_game():
    # catanatron orders some playable actions from sets of Color enums, so a seeded game is
    # only reproducible within one process: play seeds until every prompt has been seen.
    prompts = set()
    ticks = 0
    for seed in range(123, 133):
        g = fresh_game(seed=seed)
        m = AD.mapping_for(g.state.board.map)
        while g.winning_color() is None and g.state.num_turns < TURNS_LIMIT:
            prompts.add(g.state.current_prompt)
            _check_conversion(g, m)
            g.play_tick()
            ticks += 1
        if g.winning_color() is not None:
            cb = AD.to_catanbot_state(g)
            assert cb.phase == "game_over" and g.state.colors[cb.winner] == g.winning_color()
        if prompts == set(ActionPrompt):
            break
    assert ticks > 100
    assert prompts == set(ActionPrompt)


def test_me_color_must_be_seated():
    g = fresh_game()
    with pytest.raises(ValueError):
        AD.to_catanbot_state(g, me_color="PURPLE")


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------
def test_action_round_trips_for_every_action_type():
    seen = set()
    for seed in range(1, 11):
        g = fresh_game(seed=seed)
        m = AD.mapping_for(g.state.board.map)
        while g.winning_color() is None and g.state.num_turns < TURNS_LIMIT:
            st = g.state
            cb = AD.to_catanbot_state(g)
            legal = set(E.legal_actions(cb))
            index = AD.index_playable(st.playable_actions)
            assert len(index) == len(st.playable_actions)
            for a in st.playable_actions:
                seen.add(a.action_type)
                cb_action = AD.catanatron_action_to_catanbot(a, cb, m)
                assert cb_action is not None or (a.action_type == ActionType.PLAY_KNIGHT_CARD
                                                 or (a.action_type == ActionType.PLAY_YEAR_OF_PLENTY and len(a.value) == 1))
                if cb_action is None:
                    continue
                key = AD.catanbot_action_to_key(cb_action, cb, m, st.colors)
                assert key == AD.playable_key(a)
                assert index[key] is a
                # Whatever catanatron offers, catanbot's rules agree it is legal
                # (except the engine-chosen random discard).
                if a.action_type != ActionType.DISCARD:
                    assert cb_action in legal, (a, cb_action, cb.phase)
            # ... and every mappable catanbot action catanatron does not offer is one of the
            # two documented rule differences.
            for cb_action in legal:
                key = AD.catanbot_action_to_key(cb_action, cb, m, st.colors)
                if key is not None and key not in index:
                    assert cb_action[0] == A.PLAY_ROAD_BUILDING or (cb_action == (A.END_TURN,) and cb.free_roads > 0)
            g.play_tick()
        if seen == ALL_TYPES:
            break
    assert seen == ALL_TYPES


def test_logged_actions_convert_with_context():
    g = fresh_game(seed=123)
    m = AD.mapping_for(g.state.board.map)
    cb = AD.to_catanbot_state(g)
    colors = g.state.colors
    knight = AD.CAction(colors[0], ActionType.PLAY_KNIGHT_CARD, None)
    robber = AD.CAction(colors[0], ActionType.MOVE_ROBBER, ((0, 0, 0), colors[1], "WOOD"))
    assert AD.catanatron_action_to_catanbot(knight, cb, m) is None
    assert AD.catanatron_action_to_catanbot(robber, cb, m, knight) == (A.PLAY_KNIGHT, B.HEX_INDEX[(0, 0)], 1)
    assert AD.catanatron_action_to_catanbot(robber, cb, m, None) == (A.MOVE_ROBBER, B.HEX_INDEX[(0, 0)], 1)
    nobody = AD.CAction(colors[0], ActionType.MOVE_ROBBER, ((1, -1, 0), None, None))
    assert AD.catanatron_action_to_catanbot(nobody, cb, m) == (A.MOVE_ROBBER, m.coord_to_hex[(1, -1, 0)], -1)
    roll = AD.CAction(colors[0], ActionType.ROLL, (3, 4))
    assert AD.catanatron_action_to_catanbot(roll, cb, m) == (A.ROLL, 7)
    discard = AD.CAction(colors[0], ActionType.DISCARD, ["WOOD", "ORE", "ORE"])
    assert AD.catanatron_action_to_catanbot(discard, cb, m) == (A.DISCARD, (1, 0, 0, 0, 2))
    trade = AD.CAction(colors[0], ActionType.MARITIME_TRADE, ("SHEEP", "SHEEP", "SHEEP", None, "ORE"))
    assert AD.catanatron_action_to_catanbot(trade, cb, m) == (A.BANK_TRADE, B.SHEEP, B.ORE)
    yop = AD.CAction(colors[0], ActionType.PLAY_YEAR_OF_PLENTY, ("ORE", "WOOD"))
    assert AD.catanatron_action_to_catanbot(yop, cb, m) == (A.PLAY_YEAR_OF_PLENTY, B.WOOD, B.ORE)
    assert AD.catanatron_action_to_catanbot(AD.CAction(colors[0], ActionType.PLAY_YEAR_OF_PLENTY, ("ORE",)), cb, m) is None
    # catanbot actions without a catanatron equivalent
    assert AD.catanbot_action_to_key((A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)), cb, m, colors) is None
    assert AD.catanbot_action_to_key((A.ROLL, 8), cb, m, colors) is None
    assert AD.catanbot_action_to_key((A.ACCEPT_TRADE,), cb, m, colors) is None


def test_fallback_action_prefers_building():
    colors = AD.COLORS
    end = AD.CAction(colors[0], ActionType.END_TURN, None)
    road = AD.CAction(colors[0], ActionType.BUILD_ROAD, (0, 1))
    city = AD.CAction(colors[0], ActionType.BUILD_CITY, 3)
    assert AD.fallback_action([end, road, city]) is city
    assert AD.fallback_action([end, road]) is road
    assert AD.fallback_action([end]) is end
    with pytest.raises(ValueError):
        AD.fallback_action([])


def test_make_game_seats_players_in_order():
    for seat in range(4):
        players = [WeightedRandomPlayer(c) for c in AD.COLORS]
        players = players[seat:] + players[:seat]
        g = AD.make_game(players, seed=seat + 1)
        assert g.state.colors == tuple(p.color for p in players)
        assert g.state.players == players
        assert all(a.color == players[0].color for a in g.state.playable_actions)
        assert g.state.color_to_index[players[0].color] == 0
    with pytest.raises(ValueError):
        AD.make_game(players, seed=0)


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------
def test_catanbot_player_smoke_game():
    me = AD.CatanbotPlayer(AD.COLORS[2], spec="search:depth=1,beam=2,expand=4,actions=3,evaluator=heuristic",
                           strict=True, seed=1)
    players = [WeightedRandomPlayer(c) if i != 2 else me for i, c in enumerate(AD.COLORS)]
    res = AD.play_game(players, seed=5)
    assert res["colors"][2] == Color.ORANGE.value
    assert res["turns"] > 0 and (res["winner"] is not None or res["turns"] >= TURNS_LIMIT)
    st = me.stats
    assert st["errors"] == 0 and st["observe_errors"] == 0 and st["fallback"] == 0
    assert st["decisions"] > 20 and st["searched"] > 0 and st["trivial"] > 0
    # every logged action except PLAY_KNIGHT_CARD / single-card YoP was replayed through observe
    assert st["observed"] >= 0.8 * res["actions"]
    assert me.last_explanation
    # a second game with the same player instance resets cleanly
    res2 = AD.play_game(players, seed=6)
    assert res2["turns"] > 0 and me.stats["errors"] == 0


# ---------------------------------------------------------------------------
# Engine rule differences (documented in docs/BENCHMARKS.md, limitations 11 and 12)
# ---------------------------------------------------------------------------
def _hand_build(st, color, node=None, edge=None):
    """Place a settlement / road on a catanatron ``State`` the way ``apply_action`` does (free)."""
    from catanatron import state_functions as SF
    if node is not None:
        st.board.build_settlement(color, node, True)
        SF.build_settlement(st, color, node, True)
    if edge is not None:
        prev, road_color, road_lengths = st.board.build_road(color, edge)
        SF.build_road(st, color, edge, True)
        SF.mantain_longest_road(st, prev, road_color, road_lengths)


def test_road_ending_at_enemy_settlement_counts_in_catanbot_but_not_catanatron():
    """Limitation 11: catanatron's ``longest_acyclic_path`` never steps onto an enemy node.

    RED owns a 5-road path whose last road ends at BLUE's settlement.  The
    official rules (and ``engine.longest_road_length``) count 5 and award
    Longest Road; catanatron counts 4 and awards nothing.  The adapter copies
    catanatron's award so victory points agree, while catanbot's own engine,
    applying the same road on the converted state, books the +2 VP.
    """
    import random
    import networkx as nx
    from catanatron.models.board import STATIC_GRAPH, longest_acyclic_path

    g = fresh_game(seed=7)
    st = g.state
    m = AD.mapping_for(st.board.map)
    graph = STATIC_GRAPH.subgraph(st.board.map.land_nodes)
    start = 0
    dist = nx.single_source_shortest_path_length(graph, start)
    end = min(n for n, d in dist.items() if d == 5)          # simple 5-edge path, endpoints 5 apart
    path = nx.shortest_path(graph, start, end)
    red, blue = st.colors[0], st.colors[1]
    _hand_build(st, red, node=start)
    _hand_build(st, blue, node=end)
    for i in range(4):
        _hand_build(st, red, edge=(path[i], path[i + 1]))
    cb = AD.to_catanbot_state(g)
    assert st.player_state["P0_LONGEST_ROAD_LENGTH"] == E.longest_road_length(cb, 0) == 4
    assert cb.longest_road_owner == -1

    # catanatron lets RED build the road *into* BLUE's settlement ...
    last = tuple(sorted((path[4], path[5])))
    assert last in st.board.buildable_edges(red)
    # ... and catanbot's engine expects that road to complete Longest Road (+2 VP):
    sim = cb.copy()
    sim.phase, sim.current, sim.dice = PHASE_MAIN, 0, 8
    sim.players[0].resources = [1, 1, 0, 0, 0]
    road = (A.BUILD_ROAD, m.edge_to_id[last])
    assert road in E.legal_actions(sim)
    after = E.apply(sim, road, random.Random(0))
    assert after.longest_road_owner == 0 and after.longest_road_len == 5
    assert after.total_vp(0) == cb.total_vp(0) + 2

    # catanatron does not: the road never joins the component and is not counted.
    _hand_build(st, red, edge=last)
    component = next(c for c in st.board.connected_components[red] if start in c)
    assert end not in component
    assert len(longest_acyclic_path(st.board, component, red)) == 4
    assert st.player_state["P0_LONGEST_ROAD_LENGTH"] == 4 and not st.player_state["P0_HAS_ROAD"]
    cb2 = AD.to_catanbot_state(g)
    assert E.longest_road_length(cb2, 0) == 5                         # catanbot / official counting
    assert cb2.longest_road_owner == -1 and cb2.longest_road_len == 0  # adapter copies catanatron's award
    assert cb2.total_vp(0) == st.player_state["P0_ACTUAL_VICTORY_POINTS"] == cb.total_vp(0)

    # Without the enemy building both engines agree on the same five roads.
    g2 = fresh_game(seed=7)
    st2 = g2.state
    _hand_build(st2, st2.colors[0], node=start)
    for i in range(5):
        _hand_build(st2, st2.colors[0], edge=(path[i], path[i + 1]))
    cb3 = AD.to_catanbot_state(g2)
    assert st2.player_state["P0_LONGEST_ROAD_LENGTH"] == E.longest_road_length(cb3, 0) == 5
    assert st2.player_state["P0_HAS_ROAD"] and cb3.longest_road_owner == 0 and cb3.longest_road_len == 5


def test_longest_road_lengths_never_differ_by_more_than_one_through_games():
    """Through whole games catanatron's length is catanbot's or exactly one less (limitation 11),
    the adapter copies catanatron's holder / length and the victory points always agree."""
    award_ticks = 0
    for seed in range(201, 205):
        g = fresh_game(seed=seed)
        while g.winning_color() is None and g.state.num_turns < TURNS_LIMIT:
            st = g.state
            ps = st.player_state
            cb = AD.to_catanbot_state(g)
            for i in range(len(st.colors)):
                cat = int(ps[f"P{i}_LONGEST_ROAD_LENGTH"])
                ours = E.longest_road_length(cb, i)
                if ps[f"P{i}_HAS_ROAD"]:
                    award_ticks += 1
                    assert cb.longest_road_owner == i and cb.longest_road_len == cat
                    assert cat in (ours, ours - 1), (seed, i, cat, ours)
                else:
                    assert cb.longest_road_owner != i
                assert cb.total_vp(i) == ps[f"P{i}_ACTUAL_VICTORY_POINTS"]
            g.play_tick()
    assert award_ticks > 0


def test_discard_queue_mirrors_catanatron_hard_coded_limit():
    """Limitation 12: catanatron chooses the first discarder with ``discard_limit`` but the later
    ones with a hard-coded ``> 7``; the converted ``discard_queue`` lists exactly the seats
    catanatron goes on to prompt, for any limit."""
    from catanatron.models.enums import Action as CAction
    from catanatron.state import apply_action

    for limit, hands in ((7, [9, 8, 7, 10]), (9, [10, 8, 6, 9]), (5, [6, 6, 8, 3]), (9, [4, 10, 8, 3])):
        g = AD.make_game([WeightedRandomPlayer(c) for c in AD.COLORS], seed=11, discard_limit=limit)
        st = g.state
        assert st.discard_limit == limit
        while not (st.current_prompt == ActionPrompt.PLAY_TURN
                   and not st.player_state[f"P{st.current_turn_index}_HAS_ROLLED"]):
            g.play_tick()
        for i in range(4):
            for r in AD.CB_TO_RESOURCE:
                st.player_state[f"P{i}_{r}_IN_HAND"] = 0
            st.player_state[f"P{i}_WOOD_IN_HAND"] = hands[i]
        apply_action(st, CAction(st.current_color(), ActionType.ROLL, (3, 4)))
        assert st.current_prompt == ActionPrompt.DISCARD
        cb = AD.to_catanbot_state(g)
        assert cb.phase == PHASE_DISCARD and cb.discard_queue[0] == st.current_player_index
        prompted = []
        while st.current_prompt == ActionPrompt.DISCARD:
            prompted.append(st.current_player_index)
            apply_action(st, CAction(st.current_color(), ActionType.DISCARD, None))
        assert cb.discard_queue == prompted, (limit, hands, cb.discard_queue, prompted)
        assert st.current_prompt == ActionPrompt.MOVE_ROBBER
    # the default limit is the one case where the two rules coincide
    assert [j for j in range(4) if [9, 8, 7, 10][j] > 7] == [0, 1, 3]


def test_merged_knight_observation_uses_the_state_before_the_knight():
    """A PLAY_KNIGHT_CARD + MOVE_ROBBER pair is observed once, as PLAY_KNIGHT, with the state
    the card was played in (PHASE_ROLL / PHASE_MAIN, knight still in hand, so the action is
    among that state's legal actions and SearchBot.observe predicts from the same set)."""
    from catanbot.state import PHASE_GAME_OVER

    me = AD.CatanbotPlayer(AD.COLORS[1], spec="search:depth=1,beam=2,expand=4,actions=3,evaluator=heuristic",
                           strict=True, seed=3)
    seen = []
    original = me.bot.observe

    def recording_observe(state, action, seat):
        if action[0] in (A.PLAY_KNIGHT, A.MOVE_ROBBER):
            seen.append((state.phase, action, seat, action in E.legal_actions(state),
                         state.players[seat].dev_cards[B.DEV_KNIGHT], state.dev_played_this_turn))
        assert state.phase != PHASE_GAME_OVER
        return original(state, action, seat)

    me.bot.observe = recording_observe
    players = [WeightedRandomPlayer(c) if i != 1 else me for i, c in enumerate(AD.COLORS)]
    knights = []
    for seed in range(31, 40):
        AD.play_game(players, seed=seed)
        assert me.stats["errors"] == 0 and me.stats["observe_errors"] == 0
        knights = [s for s in seen if s[1][0] == A.PLAY_KNIGHT]
        if len(knights) >= 5 and any(s[2] == 1 for s in knights) and any(s[2] != 1 for s in knights):
            break
    assert len(knights) >= 5
    for phase, action, seat, legal, knights_in_hand, played_dev in knights:
        assert phase in (PHASE_ROLL, PHASE_MAIN), (phase, action, seat)
        assert knights_in_hand >= 1 and not played_dev
        assert legal, (phase, action, seat)
    # plain robber moves (after a 7) still arrive in PHASE_ROBBER, as before
    robbers = [s for s in seen if s[1][0] == A.MOVE_ROBBER]
    assert robbers and all(s[0] == PHASE_ROBBER and s[3] for s in robbers)
    assert me._knight_state is None
