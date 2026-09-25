"""Differential tests: the C++ extension ``catanbot_core`` against the Python reference ``catanbot.features``.

The tests are skipped when the extension is not built (``scripts/build_cpp.sh``).
Inside this module ``catanbot.accel.AVAILABLE`` is forced to ``False`` (autouse
fixture) so ``features.extract*`` run the pure-Python code and serve as the
reference; the extension is called directly through ``core``.
"""
from __future__ import annotations

import importlib.util
import os
import random
import time
import warnings
from typing import List

import numpy as np
import pytest

from catanbot import accel
from catanbot import board as B
from catanbot import features as F
from catanbot.selfplay import make_bot, play_game
from catanbot.state import (GameState, TradeOffer, new_game, PHASE_DISCARD, PHASE_GAME_OVER, PHASE_MAIN,
                            PHASE_ROBBER, PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT)

core = accel.load_core()
pytestmark = pytest.mark.skipif(core is None, reason="catanbot_core is not built (run scripts/build_cpp.sh)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOL = 1e-6


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def python_reference(monkeypatch):
    """Route ``features.extract`` / ``extract_batch`` to the pure-Python implementation."""
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
    for seed, n in [(1, 4), (2, 3), (3, 4), (4, 3), (5, 2), (6, 4)]:
        states += _play(seed, n)
    assert len(states) >= 2000
    return states


def _desert(s: GameState) -> int:
    return next(i for i, (r, _) in enumerate(s.hexes) if r == B.DESERT)


@pytest.fixture(scope="module")
def variant_states(game_states) -> List[GameState]:
    """Hidden-information and forced-phase variants of the game states."""
    rng = random.Random(99)
    out: List[GameState] = []
    for s in game_states[::5]:
        v = s.copy()
        for p in v.players:
            if rng.random() < 0.45:  # opponent hand as seen from a screenshot
                p.hand_known = False
                p.hand_size = sum(p.resources)
                p.resources = [0] * 5
            if rng.random() < 0.45:
                p.dev_known = False
                p.dev_count = p.total_dev
                p.dev_cards = [0] * 5
                p.dev_cards_new = [0] * 5
        out.append(v)
    for s in game_states[::40]:
        v = s.copy()
        v.robber = _desert(s)
        out.append(v)
    for s in game_states[::60]:
        n = s.num_players
        v = s.copy()
        v.phase = PHASE_DISCARD
        v.discard_queue = [(s.current + 1) % n, s.current]
        out.append(v)
        v = s.copy()
        v.phase = PHASE_ROBBER
        out.append(v)
        v = s.copy()
        v.phase = PHASE_TRADE_RESPONSE
        v.pending_trade = TradeOffer(s.current, [0, 2, 0, 0, 0], [0, 0, 0, 0, 1], {})
        v.trade_responder = (s.current + 2) % n
        out.append(v)
        v = s.copy()
        v.phase = PHASE_TRADE_SELECT
        v.pending_trade = TradeOffer(s.current, [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], {(s.current + 1) % n: True})
        v.trades_this_turn = 3
        out.append(v)
        v = s.copy()
        v.phase = PHASE_GAME_OVER
        v.winner = s.current
        out.append(v)
        v = s.copy()
        v.phase = "not_a_phase"  # no phase one-hot is set
        out.append(v)
        v = s.copy()
        v.phase = PHASE_MAIN
        v.free_roads = 2
        v.dev_played_this_turn = True
        out.append(v)
    return out


def _pairs(states):
    ss = [s for s in states for _ in range(s.num_players)]
    pp = [i for s in states for i in range(s.num_players)]
    return ss, pp


def _assert_match(states) -> None:
    ss, pp = _pairs(states)
    ref = F.extract_batch(ss, pp)  # pure Python (see the autouse fixture)
    got = core.extract_batch(ss, pp)
    assert got.shape == ref.shape == (len(ss), F.NUM_FEATURES)
    assert got.dtype == np.float32
    assert np.all(np.isfinite(got))
    if not np.allclose(got, ref, atol=ATOL):
        bad = np.argwhere(~np.isclose(got, ref, atol=ATOL))
        lines = []
        for r, c in bad[:15]:
            s = ss[r]
            lines.append(f"row {r} (phase={s.phase}, n={s.num_players}, player={pp[r]}) {F.FEATURE_NAMES[c]}: "
                         f"python={ref[r, c]!r} cpp={got[r, c]!r}")
        pytest.fail(f"{len(bad)} mismatching entries, e.g.\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# layout / generated tables
# ---------------------------------------------------------------------------
def test_layout_matches_python():
    assert core.num_features() == F.NUM_FEATURES == core.NUM_FEATURES
    assert list(core.feature_names()) == list(F.FEATURE_NAMES)
    assert list(core.phase_names()) == list(F._PHASES)
    assert core.PLAYER_BLOCK == F.PLAYER_BLOCK
    assert core.GLOBAL_BLOCK == F.GLOBAL_BLOCK
    assert core.MAX_PLAYERS == F.MAX_PLAYERS


def test_generated_headers_are_up_to_date():
    path = os.path.join(ROOT, "scripts", "gen_board_tables.py")
    spec = importlib.util.spec_from_file_location("gen_board_tables", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)  # type: ignore[union-attr]
    for header, text in ((gen.BOARD_HEADER, gen.gen_board_tables()), (gen.LAYOUT_HEADER, gen.gen_feature_layout())):
        with open(header, encoding="utf-8") as fh:
            assert fh.read() == text, f"{os.path.relpath(header, ROOT)} is stale: run scripts/gen_board_tables.py"


# ---------------------------------------------------------------------------
# differential: features
# ---------------------------------------------------------------------------
def test_state_coverage(game_states, variant_states):
    phases = {s.phase for s in game_states} | {s.phase for s in variant_states}
    assert set(F._PHASES) <= phases, phases
    assert {s.num_players for s in game_states} == {2, 3, 4}
    assert any(s.phase == PHASE_DISCARD and s.discard_queue for s in game_states)
    assert any(s.phase == PHASE_TRADE_SELECT for s in game_states)
    assert any(not p.hand_known for s in variant_states for p in s.players)
    assert any(not p.dev_known for s in variant_states for p in s.players)
    assert any(s.robber == _desert(s) for s in variant_states)


def test_features_match_python_on_game_states(game_states):
    _assert_match(game_states)


def test_features_match_python_on_variants(variant_states):
    _assert_match(variant_states)


def test_features_match_python_on_handcrafted_states():
    """Fresh games and the positions the Python unit tests use."""
    states = [new_game(2), new_game(3), new_game(4), new_game(4, rng=random.Random(3))]
    s = new_game(4)
    s.phase = PHASE_MAIN
    vs = B.HEX_VERTICES[9]
    me = s.players[0]
    me.settlements = [vs[0]]
    me.roads = [B.edge_between(vs[0], vs[1]), B.edge_between(vs[1], vs[2])]
    me.resources = [3, 3, 2, 1, 0]
    s.players[1].settlements = [vs[2]]
    states.append(s)
    s2 = s.copy()
    s2.players[0].cities = [next(v for v, t in s2.ports.items() if t == B.PORT_GENERIC)]
    s2.players[0].settlements.append(next(v for v, t in s2.ports.items() if t == B.ORE))
    s2.longest_road_owner = 0
    s2.largest_army_owner = 1
    s2.players[1].played_knights = 3
    s2.dev_deck = [0, 0, 0, 0, 0]
    states.append(s2)
    s3 = s2.copy()
    s3.players[2].dev_known = False
    s3.players[2].dev_count = 3
    s3.players[3].dev_known = False
    s3.players[3].dev_count = 1
    s3.players[1].dev_cards = [0, 2, 0, 0, 0]
    s3.dev_deck = [3, 1, 0, 1, 0]
    states.append(s3)
    _assert_match(states)
    # a state whose robber is out of range (unknown) is accepted by both
    s4 = s3.copy()
    s4.robber = -1
    _assert_match([s4])


def test_extract_single_matches_batch(game_states):
    for s in game_states[::97]:
        for pl in range(s.num_players):
            x = core.extract(s, pl)
            assert x.shape == (F.NUM_FEATURES,) and x.dtype == np.float32
            np.testing.assert_array_equal(x, core.extract_batch([s], [pl])[0])
            assert np.allclose(x, F.extract(s, pl), atol=ATOL)


def test_batch_shares_analysis_and_accepts_index_types(game_states):
    s = game_states[500]
    n = s.num_players
    ref = core.extract_batch([s] * n, list(range(n)))
    np.testing.assert_array_equal(ref, core.extract_batch([s] * n, range(n)))
    np.testing.assert_array_equal(ref, core.extract_batch([s] * n, np.arange(n)))
    np.testing.assert_array_equal(ref, core.extract_batch(tuple([s] * n), np.arange(n, dtype=np.int8)))
    mixed = core.extract_batch([s, game_states[600], s], [0, 1, n - 1])
    np.testing.assert_array_equal(mixed[0], ref[0])
    np.testing.assert_array_equal(mixed[2], ref[n - 1])
    assert core.extract_batch([], []).shape == (0, F.NUM_FEATURES)


def test_error_cases(game_states):
    s = game_states[10]
    with pytest.raises(ValueError):
        core.extract_batch([s, s], [0])
    with pytest.raises(IndexError):
        core.extract_batch([s], [s.num_players])
    with pytest.raises(IndexError):
        core.extract(s, -1)
    bad = s.copy()
    bad.players[0].resources = [1, 2, 3]
    with pytest.raises(ValueError):
        core.extract(bad, 0)
    bad = s.copy()
    bad.players[0].settlements = [99]
    with pytest.raises(ValueError):
        core.extract(bad, 0)
    bad = s.copy()
    bad.hexes = s.hexes[:-1]
    with pytest.raises(ValueError):
        core.extract(bad, 0)


# ---------------------------------------------------------------------------
# differential: longest road
# ---------------------------------------------------------------------------
def test_longest_road_matches_python_on_game_states(game_states):
    checked = 0
    for s in game_states[::2]:
        for i in range(s.num_players):
            assert core.longest_road_length(s, i) == F.longest_road_length(s, i)
            checked += 1
    assert checked >= 3000
    assert max(F.longest_road_length(s, i) for s in game_states for i in range(s.num_players)) >= 5


def test_longest_road_matches_python_on_random_networks():
    """Dense road sets with cycles and forks, opponent buildings on the network, duplicate ids."""
    rng = random.Random(7)
    for _ in range(400):
        n = rng.choice([3, 4])
        s = new_game(n, rng=rng)
        h0 = rng.randrange(B.NUM_HEXES)
        hexes = [h0] + rng.sample(B.HEX_NEIGHBORS[h0], k=min(len(B.HEX_NEIGHBORS[h0]), rng.randint(0, 2)))
        edges = sorted({e for h in hexes for e in B.HEX_EDGES[h]})
        me = rng.randrange(n)
        s.players[me].roads = rng.sample(edges, k=rng.randint(1, min(15, len(edges))))
        verts = sorted({v for e in s.players[me].roads for v in B.EDGE_VERTICES[e]})
        for j in range(n):
            if j != me and rng.random() < 0.5:
                s.players[j].settlements = rng.sample(verts, k=min(len(verts), rng.randint(1, 2)))
        if rng.random() < 0.3:
            s.players[me].settlements = rng.sample(verts, k=1)
        for i in range(n):
            assert core.longest_road_length(s, i) == F.longest_road_length(s, i)
    s = new_game(4)
    s.players[0].roads = [B.HEX_EDGES[9][0], B.HEX_EDGES[9][0]]  # duplicated id acts like a parallel road
    assert core.longest_road_length(s, 0) == F.longest_road_length(s, 0) == 2
    assert core.longest_road_length(s, 1) == 0


# ---------------------------------------------------------------------------
# accel switch
# ---------------------------------------------------------------------------
def test_accel_switch_routes_features_to_cpp(monkeypatch, game_states):
    s = game_states[300]
    n = s.num_players
    py = F.extract_batch([s] * n, range(n))  # Python (AVAILABLE is False here)
    monkeypatch.setattr(accel, "_core", core)  # accel may not have loaded it (CATANBOT_NO_ACCEL=1)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    monkeypatch.setattr(accel, "_verified", False)
    calls = []
    real = core.extract_batch

    def spy(states, players):
        calls.append(len(states))
        return real(states, players)

    monkeypatch.setattr(core, "extract_batch", spy)
    x = F.extract_batch([s] * n, range(n))
    assert calls == [n]
    assert accel.verify() is True and accel.AVAILABLE is True
    assert np.allclose(x, py, atol=ATOL)
    assert np.allclose(F.extract(s, 1), py[1], atol=ATOL)
    assert accel.longest_road_length(s, 0) == F.longest_road_length(s, 0)


def test_accel_falls_back_on_layout_mismatch(monkeypatch, game_states):
    s = game_states[300]

    class FakeCore:
        @staticmethod
        def num_features():
            return F.NUM_FEATURES + 1

        @staticmethod
        def feature_names():
            return ["nope"]

        @staticmethod
        def extract_batch(states, players):  # pragma: no cover - must not be reached
            raise AssertionError("stale extension was used")

    monkeypatch.setattr(accel, "_core", FakeCore)
    monkeypatch.setattr(accel, "_verified", False)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = F.extract_batch([s], [0])
    assert accel.AVAILABLE is False
    assert any(issubclass(m.category, RuntimeWarning) for m in w)
    monkeypatch.setattr(accel, "AVAILABLE", False)
    np.testing.assert_array_equal(x, F.extract_batch([s], [0]))


def test_env_switch_helper(monkeypatch):
    monkeypatch.setenv("CATANBOT_NO_ACCEL", "1")
    assert accel.disabled_by_env()
    monkeypatch.setenv("CATANBOT_NO_ACCEL", "0")
    assert not accel.disabled_by_env()
    monkeypatch.delenv("CATANBOT_NO_ACCEL")
    assert not accel.disabled_by_env()


# ---------------------------------------------------------------------------
# unsupported states: rejected by the extension (UnsupportedStateError), served by the Python fallback
# ---------------------------------------------------------------------------
def _five_player_state() -> GameState:
    five = new_game(5, rng=random.Random(5))
    five.phase = PHASE_MAIN
    vs = B.HEX_VERTICES[9]
    five.players[0].settlements = [vs[0]]
    five.players[0].roads = [B.edge_between(vs[0], vs[1])]
    five.players[4].settlements = [vs[3]]
    five.players[4].resources = [1, 2, 0, 1, 0]
    return five


def _unsupported_variants(s: GameState) -> List[GameState]:
    """States the C++ structs cannot hold but the Python reference evaluates."""
    n = s.num_players
    out: List[GameState] = []
    v = s.copy()
    v.robber = 2 ** 32 + 5  # Python: no hex matches -> "no robber" (C++ used to wrap it to hex 5)
    out.append(v)
    v = s.copy()
    v.turn = 2 ** 40 + 3  # Python: the turn / stage features saturate at 1
    out.append(v)
    v = s.copy()
    v.hexes = list(s.hexes) + [(B.WOOD, 8)]  # a 20th hex nobody indexes
    out.append(v)
    v = s.copy()
    v.bank = list(s.bank) + [0]  # a 6th bank entry
    out.append(v)
    v = s.copy()
    v.phase = PHASE_DISCARD
    v.discard_queue = [i % n for i in range(9)]  # longer than the C++ queue
    out.append(v)
    out.append(_five_player_state())
    v = new_game(4)  # 65 road entries; every vertex holds an opponent building so the Python trail DFS stays cheap
    v.phase = PHASE_MAIN
    v.players[0].roads = list(range(65))
    v.players[1].settlements = list(range(B.NUM_VERTICES))
    out.append(v)
    return out


def test_unsupported_state_error_is_a_value_error():
    assert issubclass(core.UnsupportedStateError, ValueError)
    with pytest.raises(core.UnsupportedStateError, match="unsupported GameState"):
        core.extract(_five_player_state(), 0)


def test_out_of_range_integers_are_rejected(game_states):
    """Integers outside the 32-bit fields raise instead of wrapping (robber = 2**32 + 5 is not hex 5)."""
    s = game_states[400]
    for value in (2 ** 31, 2 ** 32 + 5, 2 ** 40 + 3, 2 ** 63, 2 ** 70, -2 ** 31 - 1):
        for attr in ("robber", "turn", "winner", "current", "dice", "free_roads", "longest_road_len", "max_turns"):
            bad = s.copy()
            setattr(bad, attr, value)
            with pytest.raises(core.UnsupportedStateError):
                core.extract(bad, 0)
            with pytest.raises(ValueError):
                core.extract_batch([bad], [0])
            with pytest.raises(ValueError):
                core.longest_road_length(bad, 0)
        for attr in ("played_knights", "hand_size", "dev_count"):
            bad = s.copy()
            setattr(bad.players[1], attr, value)
            with pytest.raises(core.UnsupportedStateError):
                core.extract(bad, 0)
        bad = s.copy()
        bad.players[0].resources = [value, 0, 0, 0, 0]
        with pytest.raises(core.UnsupportedStateError):
            core.extract(bad, 0)
        bad = s.copy()
        bad.bank = [0, 0, value, 0, 0]
        with pytest.raises(core.UnsupportedStateError):
            core.extract(bad, 0)
        bad = s.copy()
        bad.phase = PHASE_DISCARD
        bad.discard_queue = [value]
        with pytest.raises(core.UnsupportedStateError):
            core.extract(bad, 0)
        bad = s.copy()
        bad.players[0].roads = [value]
        with pytest.raises(core.UnsupportedStateError):
            core.extract(bad, 0)
    # the 32-bit boundaries themselves are accepted and agree with Python
    ok = []
    for value in (2 ** 31 - 1, -2 ** 31):
        v = s.copy()
        v.robber = value  # no hex matches: "no robber" on both sides
        ok.append(v)
        v = s.copy()
        v.turn = value
        ok.append(v)
        v = s.copy()
        v.phase = PHASE_GAME_OVER
        v.winner = value
        ok.append(v)
    _assert_match(ok)


def test_huge_dict_keys_are_ignored(game_states):
    """A port / trade-response key that does not fit in a C long can never match and must not leave an error pending."""
    s = game_states[400].copy()
    s.ports = dict(s.ports)
    s.ports[2 ** 70] = B.PORT_GENERIC
    s.ports[-2 ** 70] = B.ORE
    s.ports["x"] = B.WOOD
    s.phase = PHASE_TRADE_RESPONSE
    s.pending_trade = TradeOffer(s.current, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], {2 ** 70: True, -1: True})
    s.trade_responder = (s.current + 1) % s.num_players
    _assert_match([s])


def test_desert_number_is_ignored(game_states):
    """(DESERT, None) - or any number on the desert - is accepted like in Python, robber on it or not."""
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
        u.hexes[d] = [B.DESERT, 8]  # a list instead of a tuple, with a number
        states.append(u)
    _assert_match(states)
    for s in states:
        for i in range(s.num_players):
            assert core.longest_road_length(s, i) == F.longest_road_length(s, i)


def test_accel_falls_back_to_python_for_unsupported_states(monkeypatch, game_states):
    """features.extract* keep the pure-Python behaviour for states the extension rejects."""
    s = game_states[400]
    variants = _unsupported_variants(s)
    refs = [F.extract_batch([v] * v.num_players, range(v.num_players)) for v in variants]  # Python (AVAILABLE False)
    monkeypatch.setattr(accel, "_core", core)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    monkeypatch.setattr(accel, "_verified", False)
    calls = []
    real_batch, real_single, real_lr = core.extract_batch, core.extract, core.longest_road_length

    def spy_batch(states, players):
        calls.append("batch")
        return real_batch(states, players)

    def spy_single(state, player):
        calls.append("single")
        return real_single(state, player)

    def spy_lr(state, player):
        calls.append("lr")
        return real_lr(state, player)

    monkeypatch.setattr(core, "extract_batch", spy_batch)
    monkeypatch.setattr(core, "extract", spy_single)
    monkeypatch.setattr(core, "longest_road_length", spy_lr)
    for v, ref in zip(variants, refs):
        n = v.num_players
        with pytest.raises(core.UnsupportedStateError):
            real_single(v, 0)
        np.testing.assert_array_equal(F.extract_batch([v] * n, range(n)), ref)  # accel -> core raises -> Python
        np.testing.assert_array_equal(F.extract(v, n - 1), ref[n - 1])
        assert accel.longest_road_length(v, 0) == F.longest_road_length(v, 0)
        assert accel.AVAILABLE is True  # the fallback is per call: the extension stays enabled
    assert calls == ["batch", "single", "lr"] * len(variants)
    calls.clear()  # a supported state still goes through the extension afterwards
    x = F.extract_batch([s], [s.current])
    assert calls == ["batch"]
    np.testing.assert_array_equal(x, real_batch([s], [s.current]))
    with pytest.raises(ValueError):  # other errors are not swallowed
        F.extract_batch([s, s], [0])
    with pytest.raises(IndexError):
        F.extract(s, s.num_players)


def test_fallback_is_thread_safe(monkeypatch, game_states):
    """Concurrent fallbacks (which briefly clear accel.AVAILABLE) restore the switch and return Python's result."""
    import threading

    s = game_states[400]
    five = _five_player_state()
    ref_five = F.extract(five, 0)  # Python
    ref_s = core.extract(s, 0)
    monkeypatch.setattr(accel, "_core", core)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    monkeypatch.setattr(accel, "_verified", False)
    errors: List[BaseException] = []

    def work() -> None:
        try:
            for _ in range(25):
                np.testing.assert_array_equal(F.extract(five, 0), ref_five)
                np.testing.assert_array_equal(F.extract_batch([s], [0])[0], ref_s)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert accel.AVAILABLE is True


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def test_benchmark_speedup(game_states):
    """extract_batch over 500 distinct states: Python vs C++ (printed with -s; target >= 30x)."""
    sub = [s for s in game_states if s.phase in (PHASE_MAIN, PHASE_TRADE_RESPONSE)][:500]
    assert len(sub) == 500
    players = [s.current for s in sub]
    best_py = min(_timed(F.extract_batch, sub, players) for _ in range(3))
    best_cpp = min(_timed(core.extract_batch, sub, players) for _ in range(5))
    speedup = best_py / best_cpp
    print(f"\nextract_batch 500 states: python {best_py * 1e3:.1f} ms, c++ {best_cpp * 1e3:.2f} ms, "
          f"speedup {speedup:.1f}x ({best_cpp / 500 * 1e6:.2f} us/state)")
    assert speedup > 5.0, f"C++ extraction only {speedup:.1f}x faster than Python"


def _timed(fn, states, players) -> float:
    t = time.perf_counter()
    fn(states, players)
    return time.perf_counter() - t
