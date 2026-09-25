"""Differential tests: the C++ heuristic (``catanbot_core.static_values`` / ``heuristic_evaluate``) against the
Python reference ``catanbot.heuristic.static_value`` / ``HeuristicEvaluator.evaluate``.

The tests are skipped when the extension is not built (``scripts/build_cpp.sh``) or predates the heuristic port.
Inside this module ``catanbot.accel.AVAILABLE`` is forced to ``False`` (autouse fixture) so the Python functions
run the pure-Python code and serve as the reference; the extension is called directly through ``core``.
"""
from __future__ import annotations

import random
import time
import warnings
from typing import List

import numpy as np
import pytest

from catanbot import accel
from catanbot import board as B
from catanbot import features as F
from catanbot import heuristic as H
from catanbot import placement as P
from catanbot.counting import expected_hidden_vp
from catanbot.heuristic import HeuristicEvaluator, static_value
from catanbot.selfplay import make_bot, play_game
from catanbot.state import (GameState, new_game, PHASE_DISCARD, PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER,
                            PHASE_ROLL, PHASE_TRADE_RESPONSE)

core = accel.load_core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "heuristic_evaluate"),
                                reason="catanbot_core is not built with the heuristic port (run scripts/build_cpp.sh)")

ATOL = 1e-6
TEMPERATURES = (16.0, 1.0, 4.0, 50.0)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def python_reference(monkeypatch):
    """Route ``HeuristicEvaluator.evaluate`` (and features) to the pure-Python implementation."""
    monkeypatch.setattr(accel, "AVAILABLE", False)


def _play(seed: int, n: int, max_turns: int = 250) -> List[GameState]:
    """Every state of one heuristic-bot game (copies), including the terminal one."""
    rng = random.Random(seed)
    state = new_game(n, rng=rng)
    states: List[GameState] = []

    def on_action(s: GameState, a, i: int) -> None:
        states.append(s.copy())

    bots = [make_bot("heuristic:temp=0.7,eps=0.1") for _ in range(n)]
    play_game(bots, state=state, rng=rng, max_turns=max_turns, seed=seed, on_action=on_action)
    states.append(state.copy())  # game_over (or turn-capped) state
    return states


@pytest.fixture(scope="module")
def game_states() -> List[GameState]:
    states: List[GameState] = []
    for seed, n in [(11, 4), (12, 3), (13, 4), (14, 3), (15, 2), (16, 4)]:
        states += _play(seed, n)
    assert len(states) >= 2000
    return states


def _desert(s: GameState) -> int:
    return next(i for i, (r, _) in enumerate(s.hexes) if r == B.DESERT)


@pytest.fixture(scope="module")
def variant_states(game_states) -> List[GameState]:
    """Hidden-information, terminal, award and board variants of the game states."""
    rng = random.Random(77)
    out: List[GameState] = []
    for s in game_states[::4]:
        v = s.copy()  # opponent hands / dev cards as seen from a screenshot
        for p in v.players:
            if rng.random() < 0.45:
                p.hand_known = False
                p.hand_size = sum(p.resources)
                p.resources = [0] * 5
            if rng.random() < 0.45:
                p.dev_known = False
                p.dev_count = p.total_dev
                p.dev_cards = [0] * 5
                p.dev_cards_new = [0] * 5
        out.append(v)
    for s in game_states[::25]:
        n = s.num_players
        v = s.copy()  # decided game: +-1000 static values, 1 / 0 evaluations
        v.phase = PHASE_GAME_OVER
        v.winner = rng.randrange(n)
        out.append(v)
        v = s.copy()  # turn cap without a winner: evaluated like a live position
        v.phase = PHASE_GAME_OVER
        v.winner = -1
        out.append(v)
        v = s.copy()
        v.robber = _desert(s)
        out.append(v)
        v = s.copy()
        v.robber = -1  # unknown robber position: no hex is blocked
        out.append(v)
        v = s.copy()  # awards held by someone else / by us, holder lengths
        v.longest_road_owner = rng.randrange(n)
        v.longest_road_len = rng.randint(4, 8)
        v.largest_army_owner = rng.randrange(n)
        out.append(v)
        v = s.copy()  # awards nobody holds yet
        v.longest_road_owner = -1
        v.longest_road_len = 0
        v.largest_army_owner = -1
        out.append(v)
        v = s.copy()  # shuffled ports (port synergy terms) and an empty dev deck (no dev-card option)
        types = list(v.ports.values())
        rng.shuffle(types)
        v.ports = dict(zip(v.ports.keys(), types))
        v.dev_deck = [0] * 5
        out.append(v)
        v = s.copy()  # big hands (discard liability) and dev cards of every type
        for p in v.players:
            p.resources = [rng.randint(0, 4) for _ in range(5)]
            p.hand_size = sum(p.resources)
            p.dev_cards = [rng.randint(0, 2) for _ in range(5)]
            p.dev_cards_new = [rng.randint(0, 1) for _ in range(5)]
            p.dev_count = p.total_dev
            p.played_knights = rng.randint(0, 4)
        out.append(v)
        v = s.copy()
        v.phase = rng.choice([PHASE_ROLL, PHASE_DISCARD, PHASE_ROBBER, PHASE_TRADE_RESPONSE, "not_a_phase"])
        out.append(v)
    return out


@pytest.fixture(scope="module")
def network_states() -> List[GameState]:
    """Random (not necessarily legal) building / road layouts: cycles, forks, blocked paths, full piece sets."""
    rng = random.Random(9)
    out: List[GameState] = []
    for _ in range(300):
        n = rng.choice([2, 3, 4])
        s = new_game(n, rng=rng)
        s.phase = PHASE_MAIN
        s.turn = rng.randint(0, 120)
        s.dev_deck = [rng.randint(0, c) for c in B.DEV_DECK_COUNTS]
        free_v = list(range(B.NUM_VERTICES))
        rng.shuffle(free_v)
        for i in range(n):
            p = s.players[i]
            h0 = rng.randrange(B.NUM_HEXES)
            hexes = [h0] + rng.sample(B.HEX_NEIGHBORS[h0], k=min(len(B.HEX_NEIGHBORS[h0]), rng.randint(0, 3)))
            edges = sorted({e for h in hexes for e in B.HEX_EDGES[h]})
            p.roads = rng.sample(edges, k=rng.randint(0, min(B.MAX_ROADS, len(edges))))
            verts = sorted({v for e in p.roads for v in B.EDGE_VERTICES[e]}) or [free_v.pop()]
            k_s = rng.randint(0, min(B.MAX_SETTLEMENTS, len(verts)))
            p.settlements = rng.sample(verts, k=k_s)
            rest = [v for v in verts if v not in p.settlements]
            p.cities = rng.sample(rest, k=min(len(rest), rng.randint(0, B.MAX_CITIES)))
            p.resources = [rng.randint(0, 5) for _ in range(5)]
            p.hand_size = sum(p.resources)
            p.played_knights = rng.randint(0, 3)
        if rng.random() < 0.5:
            s.longest_road_owner = rng.randrange(n)
            s.longest_road_len = rng.randint(5, 9)
        if rng.random() < 0.3:
            s.largest_army_owner = rng.randrange(n)
        out.append(s)
    return out


def _pairs(states):
    ss = [s for s in states for _ in range(s.num_players)]
    pp = [i for s in states for i in range(s.num_players)]
    return ss, pp


def _assert_static_values_match(states) -> None:
    worst = 0.0
    bad = []
    checked = 0
    for s in states:
        got = core.static_values(s)
        ref = [static_value(s, i) for i in range(s.num_players)]
        assert len(got) == len(ref) == s.num_players
        for i, (a, b) in enumerate(zip(got, ref)):
            checked += 1
            d = abs(a - b)
            worst = max(worst, d)
            if not (d <= ATOL):
                bad.append(f"phase={s.phase} n={s.num_players} player={i}: python={b!r} cpp={a!r}")
    if bad:
        pytest.fail(f"{len(bad)} of {checked} static values differ by more than {ATOL}, e.g.\n" + "\n".join(bad[:15]))
    print(f"\nstatic_value: {checked} values checked, max |cpp - python| = {worst:.3g}")


def _assert_evaluate_matches(states, temperatures=TEMPERATURES) -> None:
    ss, pp = _pairs(states)
    for T in temperatures:
        ref = HeuristicEvaluator(T).evaluate(ss, pp)  # pure Python (see the autouse fixture)
        got = core.heuristic_evaluate(ss, pp, T)
        assert got.shape == ref.shape == (len(ss),)
        assert got.dtype == np.float64
        assert np.all(np.isfinite(got)) and np.all((got >= 0.0) & (got <= 1.0))
        if not np.allclose(got, ref, atol=ATOL):
            bad = np.argwhere(~np.isclose(got, ref, atol=ATOL)).ravel()
            lines = [f"row {r} (phase={ss[r].phase}, n={ss[r].num_players}, player={pp[r]}, T={T}): "
                     f"python={ref[r]!r} cpp={got[r]!r}" for r in bad[:15]]
            pytest.fail(f"{len(bad)} mismatching evaluations, e.g.\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# coverage of the generated states
# ---------------------------------------------------------------------------
def test_state_coverage(game_states, variant_states):
    assert len(game_states) >= 2000
    assert {s.num_players for s in game_states} == {2, 3, 4}
    phases = {s.phase for s in game_states}
    assert {PHASE_ROLL, PHASE_MAIN, PHASE_DISCARD, PHASE_ROBBER, PHASE_TRADE_RESPONSE, PHASE_GAME_OVER} <= phases
    assert any(s.phase == PHASE_GAME_OVER and s.winner >= 0 for s in game_states)
    assert any(s.phase == PHASE_GAME_OVER and s.winner < 0 for s in variant_states)
    assert any(not p.hand_known for s in variant_states for p in s.players)
    assert any(not p.dev_known for s in variant_states for p in s.players)
    assert any(len(p.settlements) >= B.MAX_SETTLEMENTS for s in game_states for p in s.players)
    assert any(p.total_resources > 7 for s in game_states for p in s.players)
    assert any(s.longest_road_owner >= 0 for s in game_states)
    assert any(s.largest_army_owner >= 0 for s in game_states)
    assert max(H.longest_road_length(s, i) for s in game_states for i in range(s.num_players)) >= 5


# ---------------------------------------------------------------------------
# differential: static_value
# ---------------------------------------------------------------------------
def test_static_values_match_python_on_game_states(game_states):
    _assert_static_values_match(game_states)


def test_static_values_match_python_on_variants(variant_states):
    _assert_static_values_match(variant_states)


def test_static_values_match_python_on_random_networks(network_states):
    _assert_static_values_match(network_states)


def test_static_value_single_matches(game_states):
    for s in game_states[::91]:
        vals = core.static_values(s)
        for i in range(s.num_players):
            a = core.static_value(s, i)
            assert a == vals[i]
            assert abs(a - static_value(s, i)) <= ATOL


def test_terminal_values():
    s = new_game(3)
    s.phase = PHASE_GAME_OVER
    s.winner = 1
    assert core.static_values(s) == [-1000.0, 1000.0, -1000.0]
    assert [static_value(s, i) for i in range(3)] == [-1000.0, 1000.0, -1000.0]
    np.testing.assert_array_equal(core.heuristic_evaluate([s] * 3, [0, 1, 2], 16.0), [0.0, 1.0, 0.0])
    s.winner = -1  # turn cap without a winner: ordinary evaluation
    assert core.static_values(s) == [static_value(s, i) for i in range(3)]
    assert 0.0 < core.heuristic_evaluate([s], [0], 16.0)[0] < 1.0


# ---------------------------------------------------------------------------
# differential: HeuristicEvaluator.evaluate
# ---------------------------------------------------------------------------
def test_evaluate_matches_python_on_game_states(game_states):
    _assert_evaluate_matches(game_states)


def test_evaluate_matches_python_on_variants(variant_states):
    _assert_evaluate_matches(variant_states)


def test_evaluate_matches_python_on_random_networks(network_states):
    _assert_evaluate_matches(network_states, temperatures=(16.0,))


def test_evaluate_batch_identity_cache_and_index_types(game_states):
    a, b, c = game_states[300], game_states[700], game_states[1100]
    states = [a, b, a, c, a, b]  # repeated, non-consecutive objects hit the per-call cache
    players = [0, 1, 1, 0, 2 % a.num_players, 0]
    ref = HeuristicEvaluator().evaluate(states, players)
    got = core.heuristic_evaluate(states, players, 16.0)
    assert np.allclose(got, ref, atol=ATOL)
    np.testing.assert_array_equal(got, core.heuristic_evaluate(tuple(states), np.array(players, dtype=np.int8), 16.0))
    np.testing.assert_array_equal(got[:1], core.heuristic_evaluate([a], range(1), 16.0))
    assert core.heuristic_evaluate([], [], 16.0).shape == (0,)
    assert core.heuristic_evaluate([a], [0]).dtype == np.float64  # default temperature


def test_evaluate_temperature_default_is_16():
    s = new_game(4, rng=random.Random(2))
    s.phase = PHASE_MAIN
    s.players[0].settlements = [B.HEX_VERTICES[9][0]]
    np.testing.assert_array_equal(core.heuristic_evaluate([s] * 4, range(4)),
                                  core.heuristic_evaluate([s] * 4, range(4), 16.0))
    row = core.heuristic_evaluate([s] * 4, range(4), 16.0)
    assert abs(row.sum() - 1.0) < 1e-12  # softmax over the players


# ---------------------------------------------------------------------------
# differential: the helpers static_value is built from
# ---------------------------------------------------------------------------
def _own_network_edges(s: GameState, i: int) -> set:
    p = s.players[i]
    verts = set(p.settlements) | set(p.cities)
    for e in p.roads:
        verts.update(B.EDGE_VERTICES[e])
    return {e for v in verts for e in B.VERTEX_EDGES[v]}


def test_helpers_match_python(game_states, network_states):
    eps = 1e-12
    checked = 0
    for s in game_states[::9] + network_states:
        assert core.resource_scarcity(s) == P.resource_scarcity(s)
        eocc = s.occupied_edges()
        for i in range(s.num_players):
            checked += 1
            assert core.heuristic_longest_road_length(s, i) == H.longest_road_length(s, i)
            assert abs(core.expected_hidden_vp(s, i) - expected_hidden_vp(s, i)) <= eps
            assert abs(core.progress_to_build(s, i) - H._progress_to_build(s, i)) <= eps
            for ignore in (False, True):
                got = core.player_production(s, i, ignore)
                ref = P.player_production(s, i, ignore_robber=ignore)
                assert all(abs(a - b) <= eps for a, b in zip(got, ref)), (got, ref)
            for max_roads in (1, 2, 3):
                got = core.reachable_spots(s, i, max_roads)
                ref = P.reachable_spots(s, i, max_roads=max_roads)
                assert {v: d for v, (d, _) in got.items()} == {v: d for v, (d, _) in ref.items()}
                allowed = _own_network_edges(s, i)
                for v, (d, first) in got.items():
                    if d == 0:
                        assert first == -1
                    else:  # a real first road: free, and leaving our network
                        assert first in allowed and first not in eocc
            for v in list(P.reachable_spots(s, i, max_roads=2))[:4]:
                for setup in (False, True):
                    got = core.score_settlement_spot(s, i, v, setup)
                    ref = P.score_settlement_spot(s, i, v, setup=setup)
                    assert abs(got - ref) <= ATOL, (v, setup, got, ref)
    assert checked >= 1000


def test_reachable_spots_default_max_roads(game_states):
    s = game_states[400]
    assert core.reachable_spots(s, s.current) == core.reachable_spots(s, s.current, 3)


def test_heuristic_longest_road_semantics():
    """heuristic.longest_road_length differs from features.longest_road_length in two corner cases."""
    s = new_game(4)
    vs = B.HEX_VERTICES[9]
    # a road between two opponent buildings: the path may end but not start on an opponent building
    s.players[0].roads = [B.edge_between(vs[0], vs[1]), B.edge_between(vs[1], vs[2])]
    s.players[1].settlements = [vs[0]]
    s.players[2].cities = [vs[2]]
    assert core.heuristic_longest_road_length(s, 0) == H.longest_road_length(s, 0) == 1
    assert core.longest_road_length(s, 0) == F.longest_road_length(s, 0) == 2
    # a duplicated road id is a single road for the heuristic (its "used" set is keyed by edge id)
    d = new_game(4)
    d.players[0].roads = [B.HEX_EDGES[9][0], B.HEX_EDGES[9][0]]
    assert core.heuristic_longest_road_length(d, 0) == H.longest_road_length(d, 0) == 1
    assert core.longest_road_length(d, 0) == F.longest_road_length(d, 0) == 2
    assert core.heuristic_longest_road_length(d, 1) == 0
    # ordinary networks agree with the features version
    for seed in range(20):
        rng = random.Random(seed)
        g = new_game(4, rng=rng)
        h0 = rng.randrange(B.NUM_HEXES)
        edges = sorted({e for h in [h0] + B.HEX_NEIGHBORS[h0][:2] for e in B.HEX_EDGES[h]})
        g.players[0].roads = rng.sample(edges, k=rng.randint(1, 12))
        assert core.heuristic_longest_road_length(g, 0) == H.longest_road_length(g, 0)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
def test_error_cases(game_states):
    s = game_states[10]
    with pytest.raises(ValueError):
        core.heuristic_evaluate([s, s], [0], 16.0)
    with pytest.raises(IndexError):
        core.heuristic_evaluate([s], [s.num_players], 16.0)
    with pytest.raises(IndexError):
        core.static_value(s, -1)
    with pytest.raises(IndexError):
        core.score_settlement_spot(s, 0, B.NUM_VERTICES)
    bad = s.copy()
    bad.players[0].resources = [1, 2, 3]
    with pytest.raises(ValueError):
        core.static_values(bad)
    bad = s.copy()
    bad.players[0].roads = [B.NUM_EDGES]
    with pytest.raises(ValueError):
        core.heuristic_evaluate([bad], [0], 16.0)


# ---------------------------------------------------------------------------
# accel switch
# ---------------------------------------------------------------------------
def test_accel_switch_routes_evaluator_to_cpp(monkeypatch, game_states):
    s = game_states[300]
    n = s.num_players
    ev = HeuristicEvaluator(temperature=8.0)
    py = ev.evaluate([s] * n, range(n))  # Python (AVAILABLE is False here)
    monkeypatch.setattr(accel, "_core", core)  # accel may not have loaded it (CATANBOT_NO_ACCEL=1)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    monkeypatch.setattr(accel, "_verified", False)
    calls = []
    real = core.heuristic_evaluate

    def spy(states, players, temperature):
        calls.append((len(states), temperature))
        return real(states, players, temperature)

    monkeypatch.setattr(core, "heuristic_evaluate", spy)
    x = ev.evaluate([s] * n, range(n))
    assert calls == [(n, 8.0)]
    assert accel.verify() is True and accel.AVAILABLE is True
    assert np.allclose(x, py, atol=ATOL)
    assert np.allclose(ev([s] * n, range(n)), py, atol=ATOL)  # __call__ alias
    assert accel.static_values(s) == core.static_values(s)
    assert accel.static_value(s, 0) == core.static_value(s, 0)
    assert calls[-1][0] == n and len(calls) == 2


def test_accel_wrappers_fall_back_to_python(game_states):
    s = game_states[500]
    n = s.num_players
    assert accel.AVAILABLE is False
    assert accel.static_values(s) == [static_value(s, i) for i in range(n)]
    assert accel.static_value(s, 1) == static_value(s, 1)
    np.testing.assert_array_equal(accel.heuristic_evaluate([s] * n, range(n), 4.0),
                                  HeuristicEvaluator(4.0).evaluate([s] * n, range(n)))


def test_accel_disables_a_build_without_the_heuristic_port(monkeypatch, game_states):
    s = game_states[300]

    class OldCore:  # feature layout matches, but no heuristic entry points (a build predating the port)
        @staticmethod
        def num_features():
            return F.NUM_FEATURES

        @staticmethod
        def feature_names():
            return list(F.FEATURE_NAMES)

        @staticmethod
        def extract_batch(states, players):  # pragma: no cover - must not be reached
            raise AssertionError("stale extension was used")

        extract = longest_road_length = extract_batch

    monkeypatch.setattr(accel, "_core", OldCore)
    monkeypatch.setattr(accel, "_verified", False)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = HeuristicEvaluator().evaluate([s], [0])
    assert accel.AVAILABLE is False
    assert any(issubclass(m.category, RuntimeWarning) for m in w)
    np.testing.assert_array_equal(x, HeuristicEvaluator().evaluate([s], [0]))


# ---------------------------------------------------------------------------
# unsupported states: rejected by the extension (UnsupportedStateError), served by the Python fallback
# ---------------------------------------------------------------------------
def _unsupported_variants(s: GameState) -> List[GameState]:
    """States the C++ structs cannot hold but the Python reference evaluates."""
    out: List[GameState] = []
    v = s.copy()
    v.robber = 2 ** 32 + 5  # Python: no hex matches -> "no robber" (C++ used to wrap it to hex 5)
    out.append(v)
    v = s.copy()
    v.phase = PHASE_GAME_OVER
    v.winner = 2 ** 35  # Python: nobody is the winner -> -1000 / 0.0 for everyone (C++ used to say player 0)
    out.append(v)
    v = s.copy()
    v.hexes = list(s.hexes) + [(B.WOOD, 8)]  # a 20th hex nobody indexes
    out.append(v)
    v = s.copy()
    v.bank = list(s.bank) + [0]  # a 6th bank entry
    out.append(v)
    five = new_game(5, rng=random.Random(5))
    five.phase = PHASE_MAIN
    vs = B.HEX_VERTICES[9]
    five.players[0].settlements = [vs[0]]
    five.players[0].roads = [B.edge_between(vs[0], vs[1])]
    five.players[4].settlements = [vs[3]]
    five.players[4].resources = [1, 2, 0, 1, 0]
    out.append(five)
    return out


def test_out_of_range_integers_are_rejected(game_states):
    """Integers outside the 32-bit fields raise instead of wrapping (winner = 2**35 is nobody, not player 0)."""
    s = game_states[400]
    assert issubclass(core.UnsupportedStateError, ValueError)
    for value in (2 ** 31, 2 ** 32 + 5, 2 ** 35, 2 ** 63, 2 ** 70, -2 ** 31 - 1):
        for attr in ("robber", "winner", "current", "longest_road_owner", "largest_army_owner", "longest_road_len"):
            bad = s.copy()
            setattr(bad, attr, value)
            with pytest.raises(core.UnsupportedStateError):
                core.static_values(bad)
            with pytest.raises(ValueError):
                core.static_value(bad, 0)
            with pytest.raises(ValueError):
                core.heuristic_evaluate([bad], [0], 16.0)
            with pytest.raises(ValueError):
                core.reachable_spots(bad, 0)
        bad = s.copy()
        bad.players[0].played_knights = value
        with pytest.raises(core.UnsupportedStateError):
            core.static_values(bad)
        bad = s.copy()
        bad.dev_deck = [value, 0, 0, 0, 0]
        with pytest.raises(core.UnsupportedStateError):
            core.static_values(bad)
    # the 32-bit boundaries themselves are accepted and agree with Python
    ok = []
    for value in (2 ** 31 - 1, -2 ** 31):
        v = s.copy()
        v.robber = value  # no hex matches: "no robber" on both sides
        ok.append(v)
        v = s.copy()
        v.phase = PHASE_GAME_OVER
        v.winner = value
        ok.append(v)
        v = s.copy()
        v.longest_road_owner = (s.current + 1) % s.num_players
        v.longest_road_len = value
        ok.append(v)
    _assert_static_values_match(ok)
    _assert_evaluate_matches(ok, temperatures=(16.0,))


def test_desert_number_is_ignored(game_states):
    """(DESERT, None) - or any number on the desert - is accepted like in Python, robber on it or not."""
    eps = 1e-12
    states = []
    for s in game_states[::400]:
        d = _desert(s)
        v = s.copy()
        v.hexes = list(s.hexes)  # copy() shares the hexes list
        v.hexes[d] = (B.DESERT, None)
        states.append(v)
        w = v.copy()
        w.robber = d
        states.append(w)
        u = s.copy()
        u.hexes = list(s.hexes)
        u.hexes[d] = [B.DESERT, 8]
        states.append(u)
    _assert_static_values_match(states)
    _assert_evaluate_matches(states, temperatures=(16.0,))
    for s in states:
        assert core.resource_scarcity(s) == P.resource_scarcity(s)
        for i in range(s.num_players):
            for ignore in (False, True):
                got = core.player_production(s, i, ignore)
                ref = P.player_production(s, i, ignore_robber=ignore)
                assert all(abs(a - b) <= eps for a, b in zip(got, ref)), (got, ref)


def test_accel_falls_back_to_python_for_unsupported_states(monkeypatch, game_states):
    """HeuristicEvaluator.evaluate / accel.static_value* keep the pure-Python behaviour for rejected states."""
    s = game_states[400]
    n = s.num_players
    variants = _unsupported_variants(s)
    refs = [(HeuristicEvaluator(4.0).evaluate([v] * v.num_players, range(v.num_players)),
             [static_value(v, i) for i in range(v.num_players)]) for v in variants]  # Python (AVAILABLE False)
    assert refs[1][1] == [-1000.0] * n and list(refs[1][0]) == [0.0] * n  # winner = 2**35: nobody won
    monkeypatch.setattr(accel, "_core", core)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    monkeypatch.setattr(accel, "_verified", False)
    calls = []
    real = core.heuristic_evaluate

    def spy(states, players, temperature):
        calls.append((len(states), temperature))
        return real(states, players, temperature)

    monkeypatch.setattr(core, "heuristic_evaluate", spy)
    ev = HeuristicEvaluator(4.0)
    for v, (ref_ev, ref_sv) in zip(variants, refs):
        m = v.num_players
        with pytest.raises(core.UnsupportedStateError):
            core.static_values(v)
        np.testing.assert_array_equal(ev.evaluate([v] * m, range(m)), ref_ev)  # accel -> core raises -> Python
        assert accel.static_values(v) == ref_sv
        assert accel.static_value(v, m - 1) == ref_sv[m - 1]
        assert accel.AVAILABLE is True  # the fallback is per call: the extension stays enabled
    assert calls == [(v.num_players, 4.0) for v in variants]
    calls.clear()  # a supported state still goes through the extension afterwards
    x = ev.evaluate([s] * n, range(n))
    assert calls == [(n, 4.0)]
    np.testing.assert_array_equal(x, real([s] * n, range(n), 4.0))
    with pytest.raises(ValueError):  # other errors are not swallowed
        ev.evaluate([s, s], [0])
    with pytest.raises(IndexError):
        accel.static_value(s, n)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def _timed(fn, *args) -> float:
    t = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t


def test_benchmark_speedup(game_states):
    """HeuristicEvaluator.evaluate over 500 distinct states: Python vs C++ (printed with -s; target >= 20x)."""
    sub = [s for s in game_states if s.phase in (PHASE_MAIN, PHASE_TRADE_RESPONSE)][:500]
    assert len(sub) == 500
    players = [s.current for s in sub]
    ev = HeuristicEvaluator()
    best_py = min(_timed(ev.evaluate, sub, players) for _ in range(2))
    best_cpp = min(_timed(core.heuristic_evaluate, sub, players, 16.0) for _ in range(5))
    speedup = best_py / best_cpp
    print(f"\nheuristic evaluate 500 states: python {best_py * 1e3:.1f} ms, c++ {best_cpp * 1e3:.2f} ms, "
          f"speedup {speedup:.1f}x ({best_cpp / 500 * 1e6:.1f} us/state, all players scored)")
    assert speedup > 5.0, f"C++ heuristic only {speedup:.1f}x faster than Python"
