"""Tests for the catanatron-engine opponents (catanbot.bench.catanatron_players).

They run on catanatron 3.2.1 (PyPI wheel) and on the 3.3 engine (GitHub checkout).
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

catanatron = pytest.importorskip("catanatron")

from catanatron.game import TURNS_LIMIT, Game  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, ActionPrompt, ActionType  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402
from catanatron.state_functions import get_actual_victory_points, player_key  # noqa: E402

from catanbot.bench.catanatron_players import (  # noqa: E402
    DEFAULT_WEIGHTS,
    DISCARD_RESOURCE,
    AlphaBetaPlayer,
    ValueFunctionPlayer,
    choose_initial_settlement,
    get_map_tables,
    plan_discard,
    play_game,
    playable,
    robber_candidates,
    value_function,
)

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = os.path.join(ROOT, "scripts", "catanatron_ladder.py")


def load_ladder():
    spec = importlib.util.spec_from_file_location("catanatron_ladder", LADDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh(game):
    """Regenerate the playable actions after editing the state by hand."""
    acts = generate_playable_actions(game.state)
    game.state.playable_actions = acts
    if hasattr(game, "playable_actions"):
        game.playable_actions = acts
    return acts


def records(game):
    recs = getattr(game.state, "action_records", None)
    if recs is None:
        return list(game.state.actions)
    return [getattr(r, "action", r) for r in recs]


def has_rolled(game, color):
    return game.state.player_state[f"{player_key(game.state, color)}_HAS_ROLLED"]


def game_after_setup(seed):
    """A random game advanced past the initial placement phase."""
    game = Game([RandomPlayer(c) for c in COLORS], seed=seed)
    while game.state.is_initial_build_phase:
        game.play_tick()
    return game


def set_hand(game, color, wood=0, brick=0, sheep=0, wheat=0, ore=0):
    key = player_key(game.state, color)
    ps = game.state.player_state
    for name, amount in (("WOOD", wood), ("BRICK", brick), ("SHEEP", sheep), ("WHEAT", wheat), ("ORE", ore)):
        ps[f"{key}_{name}_IN_HAND"] = amount


def test_value_function_monotonic_in_vp():
    game = game_after_setup(1)
    color = game.state.colors[0]
    other = game.state.colors[1]
    base = value_function(game, color)

    more = game.copy()
    key = player_key(more.state, color)
    more.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1
    assert value_function(more, color) > base

    even_more = more.copy()
    even_more.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1
    assert value_function(even_more, color) > value_function(more, color)

    # an opponent's point hurts, and reaching the target is a huge jump
    opp = game.copy()
    okey = player_key(opp.state, other)
    opp.state.player_state[f"{okey}_ACTUAL_VICTORY_POINTS"] += 1
    assert value_function(opp, color) < base
    win = game.copy()
    win.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = game.vps_to_win
    assert value_function(win, color) > base + 500
    lose = game.copy()
    lose.state.player_state[f"{okey}_ACTUAL_VICTORY_POINTS"] = game.vps_to_win
    assert value_function(lose, color) < base - 500


def test_value_function_monotonic_in_production():
    game = game_after_setup(2)
    state = game.state
    color = state.colors[0]
    tables = get_map_tables(state.board.map)
    base = value_function(game, color)

    # a free extra settlement (no VP change) on a producing node raises the value,
    # and a better-producing node raises it more
    free = sorted(state.board.board_buildable_ids, key=lambda n: tables.node_total[n])
    weak, strong = free[0], free[-1]
    assert tables.node_total[strong] > tables.node_total[weak]

    def with_settlement(node):
        g = game.copy()
        g.state.board.buildings[node] = (color, SETTLEMENT)
        g.state.buildings_by_color[color][SETTLEMENT].append(node)
        return g

    v_weak = value_function(with_settlement(weak), color)
    v_strong = value_function(with_settlement(strong), color)
    assert v_strong > v_weak
    if tables.node_total[weak] > 0:
        assert v_weak > base

    # a city doubles production: upgrading the best settlement beats upgrading the worst
    settlements = list(state.buildings_by_color[color][SETTLEMENT])
    best = max(settlements, key=lambda n: tables.node_total[n])
    worst = min(settlements, key=lambda n: tables.node_total[n])

    def with_city(node):
        g = game.copy()
        g.state.board.buildings[node] = (color, CITY)
        g.state.buildings_by_color[color][SETTLEMENT].remove(node)
        g.state.buildings_by_color[color][CITY].append(node)
        return g

    if tables.node_total[best] > tables.node_total[worst]:
        assert value_function(with_city(best), color) > value_function(with_city(worst), color)
    assert value_function(with_city(best), color) > base

    # the robber on one of our producing tiles (shared with nobody else) lowers the value
    for node in settlements:
        for coord, _, proba in tables.node_tiles[node]:
            occupants = {state.board.buildings[n][0] for n in tables.tile_nodes[coord] if n in state.board.buildings}
            if proba > 0 and occupants == {color} and coord != state.board.robber_coordinate:
                blocked = game.copy()
                blocked.state.board.robber_coordinate = coord
                assert value_function(blocked, color) < base
                return


def test_players_return_playable_actions_in_every_prompt_of_random_games():
    seen = set()
    for seed in (11, 12):
        game = Game([RandomPlayer(c) for c in COLORS], seed=seed)
        vf = {c: ValueFunctionPlayer(c) for c in COLORS}
        ab = {c: AlphaBetaPlayer(c, budget=200) for c in COLORS}
        decisions = 0
        while game.winning_color() is None and decisions < 350:
            state = game.state
            color = state.current_color()
            actions = playable(game)
            seen.add(state.current_prompt)
            if len(actions) > 1 or decisions % 20 == 0:
                for player in (vf[color], ab[color]):
                    chosen = player.decide(game, actions)
                    assert chosen in actions, (player, chosen, state.current_prompt)
                    if DISCARD_RESOURCE is not None and actions[0].action_type == DISCARD_RESOURCE:
                        # 3.3: the discard is chosen one card at a time, least valuable first
                        assert chosen.value == plan_discard(game, color, 1)[0]
            game.play_tick()
            decisions += 1
    assert {ActionPrompt.BUILD_INITIAL_SETTLEMENT, ActionPrompt.BUILD_INITIAL_ROAD,
            ActionPrompt.PLAY_TURN, ActionPrompt.MOVE_ROBBER} <= seen


def test_players_in_a_full_engine_game_play_legal_actions():
    players = [ValueFunctionPlayer(COLORS[0]), AlphaBetaPlayer(COLORS[1], budget=300),
               RandomPlayer(COLORS[2]), WeightedRandomPlayer(COLORS[3])]
    game = Game(players, seed=7)
    ticks = 0
    while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT and ticks < 4000:
        state = game.state
        actions = playable(game)
        chosen = state.current_player().decide(game, actions)
        assert chosen in actions
        game.execute(chosen)  # validated by the engine
        ticks += 1
    assert game.winning_color() is not None


def test_smoke_match_value_function_vs_three_weighted_random():
    players = [ValueFunctionPlayer(COLORS[0])] + [WeightedRandomPlayer(c) for c in COLORS[1:]]
    game = Game(players, seed=5)
    winner = game.play()
    assert winner is not None and game.state.num_turns < TURNS_LIMIT
    vps = {c: get_actual_victory_points(game.state, c) for c in COLORS}
    assert vps[COLORS[0]] == max(vps.values())


def post_roll_decision(seed, color_index=0, min_actions=4):
    """A post-roll PLAY_TURN state of a random game with at least ``min_actions`` options."""
    game = Game([RandomPlayer(c) for c in COLORS], seed=seed)
    ticks = 0
    while ticks < 3000:
        state = game.state
        actions = playable(game)
        color = state.current_color()
        if (state.current_prompt == ActionPrompt.PLAY_TURN and not state.is_road_building
                and color == state.colors[color_index] and has_rolled(game, color)
                and len(actions) >= min_actions):
            return game
        game.play_tick()
        ticks += 1
    raise AssertionError("no suitable decision state found")


def test_alphabeta_decides_within_node_budget():
    game = post_roll_decision(21)
    color = game.state.current_color()
    actions = playable(game)
    for budget in (60, 150, 600):
        player = AlphaBetaPlayer(color, budget=budget)
        chosen = player.decide(game, actions)
        assert chosen in actions
        # a handful of copies may be in flight when the budget is hit (chance outcomes)
        assert player.nodes <= budget + 12, (budget, player.nodes)
        assert player.depth_reached >= 1
    big = AlphaBetaPlayer(color, budget=5000)
    big.decide(game, actions)
    assert big.nodes > 60


def test_never_ends_turn_with_affordable_settlement_or_city():
    game = post_roll_decision(23, min_actions=2)
    color = game.state.current_color()
    settlements = game.state.buildings_by_color[color][SETTLEMENT]
    assert settlements
    set_hand(game, color, wood=1, brick=1, sheep=1, wheat=3, ore=3)
    actions = refresh(game)
    assert any(a.action_type == ActionType.BUILD_CITY for a in actions)
    for player in (ValueFunctionPlayer(color), AlphaBetaPlayer(color, budget=400)):
        chosen = player.decide(game, actions)
        assert chosen in actions
        assert chosen.action_type != ActionType.END_TURN
        assert chosen.action_type in (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT,
                                      ActionType.BUILD_ROAD, ActionType.BUY_DEVELOPMENT_CARD,
                                      ActionType.MARITIME_TRADE, ActionType.PLAY_KNIGHT_CARD,
                                      ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY,
                                      ActionType.PLAY_ROAD_BUILDING)
    # with only the city affordable, the greedy player upgrades the best-producing
    # settlement (robber-aware, resource-weighted production, plus a 2:1 port's bonus)
    tables = get_map_tables(game.state.board.map)
    set_hand(game, color, wheat=2, ore=3)
    actions = refresh(game)
    chosen = ValueFunctionPlayer(color).decide(game, actions)
    assert chosen.action_type == ActionType.BUILD_CITY
    robber = game.state.board.robber_coordinate
    rw = DEFAULT_WEIGHTS["resource_weights"]

    def effective(node):
        prod = [0.0] * 5
        for coord, ri, p in tables.node_tiles[node]:
            if coord != robber:
                prod[ri] += p
        value = DEFAULT_WEIGHTS["production"] * sum(p * w for p, w in zip(prod, rw))
        port = tables.node_port.get(node, "none")
        if port is not None and port != "none":
            value += DEFAULT_WEIGHTS["port"] * prod[RESOURCES.index(port)]
        return value

    assert effective(chosen.value) == max(effective(n) for n in settlements)


def test_robber_never_on_own_tile_and_discard_keeps_next_build():
    game = game_after_setup(4)
    state = game.state
    color = state.colors[0]
    # robber prompt: fabricate it from the current state
    state.current_player_index = 0
    state.current_prompt = ActionPrompt.MOVE_ROBBER
    actions = refresh(game)
    tables = get_map_tables(state.board.map)
    own_nodes = set(state.buildings_by_color[color][SETTLEMENT])
    for action in robber_candidates(game, color, 4):
        coord = action.value[0]
        assert not (own_nodes & set(tables.tile_nodes[coord]))
    chosen = ValueFunctionPlayer(color).decide(game, actions)
    assert chosen in actions
    assert not (own_nodes & set(tables.tile_nodes[chosen.value[0]]))

    # discard: 10 cards holding a city (2 wheat 3 ore) discards exactly the other five
    set_hand(game, color, wood=2, brick=2, sheep=1, wheat=2, ore=3)
    discard = plan_discard(game, color)
    assert len(discard) == 5
    assert sorted(discard) == ["BRICK", "BRICK", "SHEEP", "WOOD", "WOOD"]
    # 8 cards: one city card has to go, the three non-city cards go first
    set_hand(game, color, wood=1, brick=1, sheep=1, wheat=2, ore=3)
    discard = plan_discard(game, color)
    assert len(discard) == 4
    assert {"WOOD", "BRICK", "SHEEP"} <= set(discard)
    assert discard.count("WHEAT") + discard.count("ORE") == 1


def test_opening_book_picks_a_high_production_spot():
    game = Game([RandomPlayer(c) for c in COLORS], seed=9)
    state = game.state
    color = state.current_color()
    actions = playable(game)
    chosen = choose_initial_settlement(game, color, actions)
    assert chosen in actions
    tables = get_map_tables(state.board.map)
    ranked = sorted((a.value for a in actions), key=lambda n: -tables.node_total[n])
    assert chosen.value in ranked[:6]
    player = ValueFunctionPlayer(color)
    assert player.decide(game, actions) == chosen


def test_play_game_with_smart_discards_completes():
    players = [ValueFunctionPlayer(COLORS[0]), AlphaBetaPlayer(COLORS[1], budget=300),
               WeightedRandomPlayer(COLORS[2]), WeightedRandomPlayer(COLORS[3])]
    game = Game(players, seed=13)
    winner = play_game(game, smart_discard=True)
    assert winner is not None
    for action in records(game):
        if action.action_type.value == "DISCARD":  # 3.2.1: the chosen cards are logged
            assert isinstance(action.value, list)
        elif action.action_type.value == "DISCARD_RESOURCE":  # 3.3: one resource at a time
            assert action.value in RESOURCES


class _Invariants:
    """Accumulator for play_game: checks every decision of the catanbot players."""

    def __init__(self, watched):
        self.watched = watched  # colours of the catanbot players
        self.decisions = 0
        self.discards = 0
        self.robber_moves = 0
        self.dev_bought = 0

    def before(self, game):
        pass

    def after(self, game):
        pass

    def step(self, game, action):
        if action.color not in self.watched:
            return
        self.decisions += 1
        state = game.state
        actions = playable(game)
        assert action in actions or action.action_type.value == "DISCARD", action
        t = action.action_type
        if t == ActionType.END_TURN:
            # never end the turn while a settlement / city is affordable and placeable
            assert not any(a.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY)
                           for a in actions), (action, actions)
        elif t == ActionType.MOVE_ROBBER:
            self.robber_moves += 1
            tables = get_map_tables(state.board.map)
            own = set(state.buildings_by_color[action.color][SETTLEMENT])
            own |= set(state.buildings_by_color[action.color][CITY])
            others = [a for a in actions if not (own & set(tables.tile_nodes[a.value[0]]))]
            if others:  # an own tile only when nothing else is available
                assert not (own & set(tables.tile_nodes[action.value[0]])), action
        elif t == ActionType.BUY_DEVELOPMENT_CARD:
            self.dev_bought += 1
        elif t.value == "DISCARD":  # 3.2.1: play_game applies the planned cards
            self.discards += 1
            assert isinstance(action.value, list)
            assert action.value == plan_discard(game, action.color)
        elif t.value == "DISCARD_RESOURCE":  # 3.3: one card at a time
            self.discards += 1
            assert action.value == plan_discard(game, action.color, 1)[0]


def test_full_game_invariants_of_both_players():
    """Over whole games: legal actions, no END_TURN with a buildable settlement /
    city, the robber never on an own tile, discards follow plan_discard."""
    totals = _Invariants(set())
    for seed in (31, 32):
        players = [ValueFunctionPlayer(COLORS[0]), AlphaBetaPlayer(COLORS[1], budget=300),
                   WeightedRandomPlayer(COLORS[2]), RandomPlayer(COLORS[3])]
        game = Game(players, seed=seed)
        acc = _Invariants({COLORS[0], COLORS[1]})
        winner = play_game(game, smart_discard=True, accumulators=[acc])
        assert winner is not None
        totals.decisions += acc.decisions
        totals.discards += acc.discards
        totals.robber_moves += acc.robber_moves
        totals.dev_bought += acc.dev_bought
    assert totals.decisions > 100
    assert totals.robber_moves > 0
    assert totals.discards > 0, "no discard happened in the sampled games"
    assert totals.dev_bought > 0


def test_ladder_defaults_to_smart_discards_and_records_them():
    ladder = load_ladder()
    opts = {"ab_budget": 200}
    found = False
    for seed in range(40, 52):
        game, winner = ladder.play_seeded(("F", "W", "R", "A"), seed, opts)
        discards = [a for a in records(game) if a.action_type.value in ("DISCARD", "DISCARD_RESOURCE")
                    and a.color in (COLORS[0], COLORS[3])]
        if discards:
            found = True
            for a in discards:
                if a.action_type.value == "DISCARD":
                    assert isinstance(a.value, list), a  # chosen, not the engine's random None
                else:
                    assert a.value in RESOURCES
            break
    assert found, "no discard by a catanbot player in the sampled seeds"
    result = ladder.play_one((("F", "W", "R", "V"), 3, opts))
    assert result["smart_discard"] is True
    assert result["winner"] == 0
    control = ladder.play_one((("F", "W", "R", "V"), 3, dict(opts, smart_discard=False)))
    assert control["smart_discard"] is False


@pytest.mark.skipif(os.name != "posix", reason="re-exec with a pinned hash seed is tested on posix")
def test_ladder_replays_identical_games_across_processes(tmp_path):
    """The ladder pins PYTHONHASHSEED (re-exec), so a seed replays the same game."""
    outs = []
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    for i in range(2):
        out = tmp_path / f"r{i}.jsonl"
        proc = subprocess.run(
            [sys.executable, LADDER, "--games", "2", "--types", "F,R,W,V", "--workers", "2",
             "--seed", "77", "--json", str(out)],
            env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp_path),
        )
        assert proc.returncode == 0, proc.stderr
        assert "hashseed=0" in proc.stdout.splitlines()[0]
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        for row in rows:
            row.pop("time")
        outs.append(rows)
    assert len(outs[0]) == 2
    assert outs[0] == outs[1]
