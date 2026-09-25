"""Differential tests: the C++ engine in ``catanbot_core`` against the Python reference ``catanbot.engine``.

Skipped when the extension is not built (``scripts/build_cpp.sh``).  The Python
engine is the reference: the opt-in hook (``accel.ENGINE_ACTIVE``) is forced
off inside this module so ``engine.legal_actions`` / ``apply`` run the pure
Python code, and the extension is called directly through ``core``.

Randomness is made comparable with *forced draws*: rolls are applied as
``(ROLL, value)``, steals and dev-card draws get the index of the drawn card
(``core.apply_forced`` on the C++ side, a stand-in ``random.Random`` returning
that index on the Python side), so both engines see exactly the same chance
outcomes and must produce identical ``to_dict()`` states.
"""
from __future__ import annotations

import gc
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from catanbot import accel
from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot.state import (GameState, Player, new_game, PHASE_DISCARD, PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER,
                            PHASE_ROLL, PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT)
from tests.test_engine import _action_universe, _check_awards, _check_board, _check_cards

core = accel.load_core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "apply_forced"),
                                reason="catanbot_core (with the engine port) is not built (run scripts/build_cpp.sh)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Action = Tuple


@pytest.fixture(autouse=True)
def python_reference(monkeypatch):
    """Keep ``engine.*`` on the pure-Python path: it is the reference here."""
    monkeypatch.setattr(accel, "ENGINE_ACTIVE", False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Forced:
    """``random.Random`` stand-in returning one forced draw, the Python twin of ``core.apply_forced``.

    ``randrange(n)`` returns the index; the two ``randint(1, 6)`` calls of an unforced roll return
    ``1 + index // 6`` and ``1 + index % 6``.  With ``index < 0`` any call is an error (the action is
    not supposed to consume randomness).
    """

    def __init__(self, index: int) -> None:
        self.index = index
        self.calls = 0

    def randrange(self, n: int) -> int:
        assert 0 <= self.index < n, f"forced draw {self.index} out of range for {n}"
        self.calls += 1
        return self.index

    def randint(self, lo: int, hi: int) -> int:
        span = hi - lo + 1
        assert 0 <= self.index < span * span, f"forced roll index {self.index} out of range"
        v = self.index // span if self.calls == 0 else self.index % span
        self.calls += 1
        return lo + v


_CONSTRUCTIVE = {A.BUILD_ROAD, A.BUILD_SETTLEMENT, A.BUILD_CITY, A.BUY_DEV, A.PLAY_KNIGHT, A.PLAY_ROAD_BUILDING,
                 A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY, A.EXECUTE_TRADE, A.ACCEPT_TRADE}
_WEIGHTS = {A.BUILD_CITY: 4, A.BUILD_SETTLEMENT: 3, A.BUILD_ROAD: 2, A.BUY_DEV: 2}


def _policy(rng: random.Random, acts: List[Action]) -> Action:
    """Random policy biased towards building so games reach cities, dev cards, awards and wins."""
    r = rng.random()
    if r < 0.8:
        c = [a for a in acts if a[0] in _CONSTRUCTIVE]
        if c:
            return rng.choices(c, weights=[_WEIGHTS.get(a[0], 1) for a in c])[0]
    if r < 0.9 and (A.END_TURN,) in acts:
        return (A.END_TURN,)
    return rng.choice(acts)


def _force(rng: random.Random, s: GameState, a: Action) -> Tuple[Action, int]:
    """Draw the chance outcome of ``a`` with ``rng``: forced roll value / steal index / dev draw index."""
    if a[0] == A.ROLL and len(a) == 1:
        return (A.ROLL, rng.randint(1, 6) + rng.randint(1, 6)), -1
    if a[0] in (A.MOVE_ROBBER, A.PLAY_KNIGHT) and a[2] >= 0:
        return a, rng.randrange(sum(s.players[a[2]].resources))
    if a[0] == A.BUY_DEV:
        return a, rng.randrange(sum(s.dev_deck))
    return a, -1


def _same_objects(s: GameState, players: List[Player], lists: List[list]) -> bool:
    if list(s.players) != players:
        return False
    cur = [p.resources for p in s.players] + [p.roads for p in s.players] + [s.bank]
    return all(a is b for a, b in zip(cur, lists))


def _play_differential(seed: int, n: int, max_turns: int, stats: Dict[str, int]) -> GameState:
    """One game: at every step legal_actions and apply_forced must agree with the Python engine."""
    rng = random.Random(seed)
    s = new_game(n, rng=rng)
    s.max_turns = max_turns
    step = 0
    while s.phase != PHASE_GAME_OVER:
        py_acts = E.legal_actions(s)
        cpp_acts = core.legal_actions(s)
        assert cpp_acts == py_acts, (seed, step, s.phase)
        assert all(type(a) is tuple for a in cpp_acts)
        a, draw = _force(rng, s, _policy(rng, py_acts))
        before = s.to_dict()
        py_next = E.apply(s, a, _Forced(draw))
        cpp_next = core.apply_forced(s, a, draw)
        assert isinstance(cpp_next, GameState) and cpp_next is not s
        d_py, d_cpp = py_next.to_dict(), cpp_next.to_dict()
        assert d_cpp == d_py, (seed, step, s.phase, a, [k for k in d_py if d_py[k] != d_cpp[k]])
        assert cpp_next.rolls_history_len == py_next.rolls_history_len
        assert cpp_next.hexes is s.hexes and cpp_next.ports is s.ports
        assert [p.color for p in cpp_next.players] == [p.color for p in s.players]
        if step % 7 == 0:
            assert s.to_dict() == before, "apply_forced modified its input"
            # the in-place variant, driven by a random.Random-like object: same result, same objects
            s2 = s.copy()
            players, lists = list(s2.players), [p.resources for p in s2.players] + [p.roads for p in s2.players] + [s2.bank]
            r2 = core.apply_inplace(s2, a, _Forced(draw))
            assert r2 is s2 and s2.to_dict() == d_py and s2.rolls_history_len == py_next.rolls_history_len
            assert _same_objects(s2, players, lists), "apply_inplace replaced objects instead of updating them"
        stats[a[0]] = stats.get(a[0], 0) + 1
        stats["phase:" + s.phase] = stats.get("phase:" + s.phase, 0) + 1
        if s.longest_road_owner != py_next.longest_road_owner:
            stats["longest_road_change"] = stats.get("longest_road_change", 0) + 1
        if s.largest_army_owner != py_next.largest_army_owner:
            stats["largest_army_change"] = stats.get("largest_army_change", 0) + 1
        # continue from either engine's result (the C++-made GameState must be a valid input too)
        s = cpp_next if step % 2 else py_next
        step += 1
        assert step < 100_000
    assert E.legal_actions(s) == core.legal_actions(s) == []
    if s.turn < max_turns:
        stats["won"] = stats.get("won", 0) + 1
        assert s.winner == s.current and E.count_vp(s, s.winner) >= B.VP_TO_WIN
    else:
        stats["capped"] = stats.get("capped", 0) + 1
    stats["steps"] = stats.get("steps", 0) + step
    return s


# ---------------------------------------------------------------------------
# differential fuzz: >= 300 games, every step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_differential_fuzz_games(num_players: int):
    """105 games per player count (315 in total): identical legal_actions and apply results at every step."""
    stats: Dict[str, int] = {}
    caps = [60, 120, 250]
    for g in range(105):
        seed = 1000 * num_players + g
        _play_differential(seed, num_players, caps[g % 3], stats)
    assert all(kind in stats for kind in A.ALL_KINDS), sorted(set(A.ALL_KINDS) - set(stats))
    for phase in (PHASE_ROLL, PHASE_MAIN, PHASE_DISCARD, PHASE_ROBBER, PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT):
        assert stats.get("phase:" + phase, 0) > 0, phase
    assert stats.get("won", 0) >= 1 and stats.get("capped", 0) >= 1, stats
    assert stats.get("longest_road_change", 0) >= 1 and stats.get("largest_army_change", 0) >= 1, stats
    assert stats["steps"] >= 20_000


# ---------------------------------------------------------------------------
# random.Random path: same rng object -> same game and same rng state afterwards
# ---------------------------------------------------------------------------
def _sample_states(seeds, n: int, max_turns: int = 150, every: int = 3) -> List[Tuple[GameState, List[Action]]]:
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        s = new_game(n, rng=rng)
        s.max_turns = max_turns
        step = 0
        while s.phase != PHASE_GAME_OVER:
            acts = E.legal_actions(s)
            if step % every == 0:
                out.append((s.copy(), acts))
            a, draw = _force(rng, s, _policy(rng, acts))
            s = E.apply(s, a, _Forced(draw))
            step += 1
    return out


@pytest.fixture(scope="module")
def sampled():
    return _sample_states([11, 12, 13], 4) + _sample_states([14, 15], 3) + _sample_states([16], 2)


def test_random_object_path_matches_python(sampled):
    """``core.apply(state, action, random.Random(k))`` consults the rng exactly like the Python engine."""
    checked = 0
    for i, (s, acts) in enumerate(sampled):
        for a in acts[:: max(1, len(acts) // 4)]:
            k = 7919 * i + checked
            r_py, r_cpp = random.Random(k), random.Random(k)
            py = E.apply(s, a, r_py)
            cpp = core.apply(s, a, r_cpp)
            assert cpp.to_dict() == py.to_dict(), (i, a)
            assert r_py.getstate() == r_cpp.getstate(), (i, a)
            if a[0] in (A.ROLL, A.BUY_DEV, A.MOVE_ROBBER, A.PLAY_KNIGHT):
                s2, r_in = s.copy(), random.Random(k)
                assert core.apply_inplace(s2, a, r_in) is s2
                assert s2.to_dict() == py.to_dict() and r_in.getstate() == r_py.getstate()
            checked += 1
    assert checked >= 500


def test_seed_and_none_rng():
    s = new_game(4, rng=random.Random(3))
    while s.phase != PHASE_ROLL:
        s = E.apply(s, E.legal_actions(s)[0])
    a = (A.ROLL,)
    assert core.apply(s, a, 5).to_dict() == core.apply(s, a, 5).to_dict()
    seen = {core.apply(s, a, seed).dice for seed in range(40)}
    assert len(seen) > 3 and all(2 <= d <= 12 for d in seen)
    out = core.apply(s, a)  # None -> a fresh C++ generator
    assert 2 <= out.dice <= 12 and out.rolls_history_len == s.rolls_history_len + 1
    with pytest.raises(AttributeError):  # like the Python engine: 'str' object has no attribute 'randint'
        core.apply(s, a, "not an rng")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
def test_illegal_actions_raise_the_engine_class_with_the_same_message(sampled):
    checked = 0
    for s, acts in sampled[::4]:
        n = s.num_players
        legal = set(acts)
        for u in _action_universe(n) + [(A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)), (A.DISCARD, (1, 0, 0, 0, 0)),
                                        (A.ROLL, 7), (A.ROLL, 13), ("bogus",), (A.END_TURN, 1), (), 5]:
            err_py = err_cpp = None
            try:
                E.apply(s, u, random.Random(1))
            except E.IllegalActionError as exc:
                err_py = (type(exc), str(exc))
            try:
                core.apply(s, u, random.Random(1))
            except E.IllegalActionError as exc:
                err_cpp = (type(exc), str(exc))
            assert err_py == err_cpp, (s.phase, u, err_py, err_cpp)
            if u in legal:
                assert err_py is None
            elif not isinstance(u, tuple) or not u or (u[0] != A.PROPOSE_TRADE and u != (A.ROLL, 7)):
                assert err_py is not None
            checked += 1
    assert checked > 2000
    over = new_game(4)
    over.phase = PHASE_GAME_OVER
    with pytest.raises(E.IllegalActionError, match="game is over"):
        core.apply(over, (A.END_TURN,))
    with pytest.raises(E.IllegalActionError, match=r"unknown action \('nope', 1\)"):
        core.apply(new_game(4), ("nope", 1))
    assert issubclass(core.IllegalActionError, ValueError)


def test_apply_forced_validates_the_index():
    s = new_game(4, rng=random.Random(5))
    while s.phase != PHASE_ROLL:
        s = E.apply(s, E.legal_actions(s)[0])
    assert core.apply_forced(s, (A.ROLL,), 6 * 2 + 4).dice == 3 + 5   # d1 = 3, d2 = 5
    with pytest.raises(ValueError):
        core.apply_forced(s, (A.ROLL,), 36)
    # a steal: index into the victim's cards in resource order
    t = s.copy()
    t.phase = PHASE_ROBBER
    t.players[1].resources = [0, 2, 0, 3, 0]
    t.players[1].settlements = [B.HEX_VERTICES[(t.robber + 1) % 19][0]]
    victim_hex = next(h for h in range(19) if h != t.robber and t.players[1].settlements[0] in B.HEX_VERTICES[h])
    for idx, res in [(0, B.BRICK), (1, B.BRICK), (2, B.WHEAT), (4, B.WHEAT)]:
        out = core.apply_forced(t, (A.MOVE_ROBBER, victim_hex, 1), idx)
        assert out.players[0].resources[res] == t.players[0].resources[res] + 1
        assert out.to_dict() == E.apply(t, (A.MOVE_ROBBER, victim_hex, 1), _Forced(idx)).to_dict()
    with pytest.raises(ValueError):
        core.apply_forced(t, (A.MOVE_ROBBER, victim_hex, 1), 5)
    with pytest.raises(ValueError):
        core.apply_forced(t, (A.MOVE_ROBBER, victim_hex, 1), -1)  # a draw is needed


def test_unsupported_states_raise_and_the_wrappers_fall_back():
    s = new_game(5, rng=random.Random(1))  # more than MAX_PLAYERS
    with pytest.raises(core.UnsupportedStateError):
        core.legal_actions(s)
    with pytest.raises(core.UnsupportedStateError):
        core.apply(s, (A.SETUP_SETTLEMENT, 0))
    assert accel.engine_legal_actions(s) is None
    assert accel.engine_apply(s, (A.SETUP_SETTLEMENT, 0)) is None
    assert accel.engine_apply_inplace(s, (A.SETUP_SETTLEMENT, 0)) is None


# ---------------------------------------------------------------------------
# random_playout_fast
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_random_playout_fast_replays_in_python_and_keeps_the_invariants(seed: int):
    n = 3 if seed % 3 == 2 else 4
    start = new_game(n, rng=random.Random(seed))
    max_turns = 80 if seed % 2 else 400
    before = start.to_dict()
    final, trace = core.random_playout_fast(start, seed, max_turns, trace=True)
    assert start.to_dict() == before and final is not start
    assert final.phase == PHASE_GAME_OVER and final.turn <= max_turns and final.max_turns == max_turns
    assert len(trace) > 50
    # replay every recorded (action, draw) with the Python engine
    s = start.copy()
    s.max_turns = max_turns
    played_other = 0
    for a, draw in trace:
        assert a in E.legal_actions(s) or (a[0] == A.ROLL and len(a) == 2 and (A.ROLL,) in E.legal_actions(s))
        E.apply_inplace(s, a, _Forced(draw))
        if a[0] in (A.PLAY_ROAD_BUILDING, A.PLAY_YEAR_OF_PLENTY, A.PLAY_MONOPOLY):
            played_other += 1
    assert s.to_dict() == final.to_dict()
    assert final.rolls_history_len == sum(1 for a, _ in trace if a[0] == A.ROLL)
    _check_cards(final, played_other)
    _check_board(final)
    _check_awards(final)
    if final.turn < max_turns:
        assert final.winner == final.current and E.count_vp(final, final.winner) >= B.VP_TO_WIN
    # deterministic in the seed, and the untraced call gives the same game
    assert core.random_playout_fast(start, seed, max_turns).to_dict() == final.to_dict()
    assert core.random_playout_fast(start, seed + 1000, max_turns).to_dict() != final.to_dict()


def test_random_playout_fast_api():
    start = new_game(4, rng=random.Random(9))
    with pytest.raises(RuntimeError, match="max_actions"):
        core.random_playout_fast(start, 1, 400, max_actions=10)
    assert accel.random_playout_fast(start, 1, 50).turn <= 50
    over = new_game(4)
    over.phase = PHASE_GAME_OVER
    assert core.random_playout_fast(over, 1).to_dict() == over.to_dict()


# ---------------------------------------------------------------------------
# helpers exposed for the tests
# ---------------------------------------------------------------------------
def test_helpers_match_python(sampled):
    for s, acts in sampled[::5]:
        for v in range(2, 13):
            assert core.production_for_roll(s, v) == E.production_for_roll(s, v)
        assert core.acting_player(s) == E.acting_player(s)
        for i in range(s.num_players):
            assert core.count_vp(s, i) == E.count_vp(s, i)
            assert core.count_vp(s, i, False) == E.count_vp(s, i, include_hidden=False)
            res = s.players[i].resources
            k = sum(res) // 2
            assert core.discard_options(res, k) == E.discard_options(res, k)
    rng = random.Random(4)
    for _ in range(200):
        res = [rng.randint(0, 6) for _ in range(5)]
        k = rng.randint(0, sum(res) + 1)
        assert core.discard_options(res, k) == E.discard_options(res, k)
        assert core.discard_options(res, k, 7) == E.discard_options(res, k, 7)
    big = [9, 8, 7, 6, 5]
    assert core.discard_options(big, 17) == E.discard_options(big, 17)
    assert len(core.discard_options(big, 17)) == E.DISCARD_ENUM_CAP


def test_hidden_information_bookkeeping_is_preserved():
    """hand_size / dev_count of untouched players are copied as they are (only touched players are re-synced)."""
    s = new_game(4, rng=random.Random(2))
    while s.phase != PHASE_MAIN:
        s = E.apply(s, E.legal_actions(s)[0], random.Random(1))
    s.players[2].hand_known = False
    s.players[2].hand_size = 5
    s.players[2].resources = [0, 0, 0, 0, 0]
    s.players[3].dev_known = False
    s.players[3].dev_count = 2
    a = (A.END_TURN,)
    py, cpp = E.apply(s, a), core.apply(s, a)
    assert cpp.to_dict() == py.to_dict()
    assert cpp.players[2].hand_known is False and cpp.players[2].hand_size == 5
    assert cpp.players[3].dev_known is False and cpp.players[3].dev_count == 2


# ---------------------------------------------------------------------------
# CState: a converted state kept in C++
# ---------------------------------------------------------------------------
def test_cstate_handle_matches_the_gamestate_path(sampled):
    for i, (s, acts) in enumerate(sampled[::3]):
        n = s.num_players
        h = core.CState(s)
        assert h.legal_actions() == acts == core.legal_actions(h)
        assert (h.phase, h.current, h.turn, h.dice, h.winner, h.num_players, h.max_turns) == \
            (s.phase, s.current, s.turn, s.dice, s.winner, n, s.max_turns)
        assert h.acting_player == E.acting_player(s) and h.is_terminal == E.is_terminal(s) and h.origin is s
        assert [h.count_vp(p) for p in range(n)] == [E.count_vp(s, p) for p in range(n)]
        back = h.to_state()
        assert back.to_dict() == s.to_dict() and back.rolls_history_len == s.rolls_history_len and back.hexes is s.hexes
        a, draw = _force(random.Random(i), s, acts[i % len(acts)])
        ref = E.apply(s, a, _Forced(draw))
        h2 = h.apply_forced(a, draw)
        assert h2.to_state().to_dict() == ref.to_dict() and h2.rolls_history_len == ref.rolls_history_len
        assert h.to_state().to_dict() == s.to_dict(), "apply_forced modified the handle"
        assert h2.legal_actions() == E.legal_actions(ref)
        k = 31 * i + 7
        assert h.apply(a, random.Random(k)).to_state().to_dict() == E.apply(s, a, random.Random(k)).to_dict()
        h3 = core.CState(h)  # a copy
        assert h3.apply_inplace(a, _Forced(draw)) is h3 and h3.to_state().to_dict() == ref.to_dict()
        assert h.copy().to_state().to_dict() == s.to_dict()
        # the evaluators take handles directly
        np.testing.assert_array_equal(core.extract_batch([h2] * n, range(n)), core.extract_batch([ref] * n, range(n)))
        np.testing.assert_array_equal(core.extract(h2, 0), core.extract(ref, 0))
        assert core.static_values(h2) == core.static_values(ref)
        assert core.static_value(h2, 1 % n) == core.static_value(ref, 1 % n)
        np.testing.assert_array_equal(core.heuristic_evaluate([h2, ref], [0, 0]), core.heuristic_evaluate([ref, ref], [0, 0]))
    s, acts = sampled[10]
    h = core.CState(s)
    bad = (A.BUILD_CITY, 99)
    with pytest.raises(E.IllegalActionError) as e1:
        E.apply(s, bad)
    with pytest.raises(E.IllegalActionError) as e2:
        h.apply(bad)
    assert str(e1.value) == str(e2.value)
    assert h.random_playout(5, 100).to_state().to_dict() == core.random_playout_fast(s, 5, 100).to_dict()
    assert h.to_state().to_dict() == s.to_dict()  # the playout worked on a copy
    h.max_turns = 33
    assert h.max_turns == 33 and h.to_state().max_turns == 33
    over = h.random_playout(1)
    assert over.is_terminal and over.turn <= 33
    with pytest.raises(E.IllegalActionError, match="game is over"):
        over.apply((A.END_TURN,))
    assert over.legal_actions() == []
    with pytest.raises(core.UnsupportedStateError):
        core.CState(new_game(5))


# ---------------------------------------------------------------------------
# the switch in engine.py
# ---------------------------------------------------------------------------
def test_engine_hook_is_off_by_default_and_routes_when_active(monkeypatch):
    monkeypatch.delenv(accel.ENGINE_ENV, raising=False)
    assert not accel.engine_enabled_by_env()
    monkeypatch.setenv(accel.ENGINE_ENV, "1")
    assert accel.engine_enabled_by_env()
    monkeypatch.setenv(accel.ENGINE_ENV, "0")
    assert not accel.engine_enabled_by_env()
    assert accel.engine_available()

    s = new_game(4, rng=random.Random(8))
    for _ in range(12):
        s = E.apply(s, E.legal_actions(s)[0], random.Random(1))
    ref_acts = E.legal_actions(s)  # ENGINE_ACTIVE is False here (autouse fixture)
    ref_next = E.apply(s, ref_acts[0], random.Random(1)).to_dict()

    monkeypatch.setattr(accel, "_core", core)
    monkeypatch.setattr(accel, "ENGINE_ACTIVE", True)
    calls: List[str] = []
    for name in ("legal_actions", "apply", "apply_inplace"):
        real = getattr(core, name)

        def spy(*args, _name=name, _real=real):
            calls.append(_name)
            return _real(*args)

        monkeypatch.setattr(core, name, spy)
    assert E.legal_actions(s) == ref_acts
    assert E.apply(s, ref_acts[0], random.Random(1)).to_dict() == ref_next
    t = s.copy()
    assert E.apply_inplace(t, ref_acts[0], random.Random(1)) is t and t.to_dict() == ref_next
    assert calls == ["legal_actions", "apply", "apply_inplace"]
    # random_playout goes through apply_inplace -> C++
    end = E.random_playout(s, random.Random(3), max_turns=40)
    assert end.phase == PHASE_GAME_OVER and calls.count("apply_inplace") > 10
    # unsupported states fall back to the Python code
    five = new_game(5, rng=random.Random(1))
    calls.clear()
    assert E.legal_actions(five) == [(A.SETUP_SETTLEMENT, v) for v in range(B.NUM_VERTICES)]
    assert E.apply(five, (A.SETUP_SETTLEMENT, 0)).players[0].settlements == [0]
    # apply -> C++ refused -> Python apply(copy) -> apply_inplace (hooked as well, refused again) -> Python
    assert calls == ["legal_actions", "apply", "apply_inplace"]


def test_engine_hook_activates_from_the_environment():
    code = (
        "from catanbot import accel, engine as E\n"
        "from catanbot.state import new_game\n"
        "assert accel.ENGINE_ACTIVE, accel.ENGINE_ACTIVE\n"
        "core = accel.load_core(); seen = []\n"
        "real = core.legal_actions; core.legal_actions = lambda s: seen.append(1) or real(s)\n"
        "acts = E.legal_actions(new_game(4))\n"
        "assert seen == [1] and len(acts) == 54\n"
        "print('routed')\n"
    )
    env = dict(os.environ, PYTHONPATH=ROOT, CATANBOT_ACCEL_ENGINE="1")
    env.pop("CATANBOT_NO_ACCEL", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "routed" in out.stdout


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def _best(fn, reps: int) -> float:
    best = float("inf")
    gc.disable()
    try:
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
    finally:
        gc.enable()
    return best


def test_benchmark_speedup(sampled):
    """legal_actions / apply per call and random playouts: Python vs C++ (printed with -s)."""
    states = [s for s, _ in sampled if s.phase in (PHASE_MAIN, PHASE_ROLL)][:400]
    acts = [E.legal_actions(s)[0] for s in states]
    pairs = list(zip(states, acts))
    n = len(states)
    rng = random.Random(0)
    py_l = _best(lambda: [E.legal_actions(s) for s in states], 5)
    cpp_l = _best(lambda: [core.legal_actions(s) for s in states], 5)
    py_a = _best(lambda: [E.apply(s, a, rng) for s, a in pairs], 5)
    cpp_a = _best(lambda: [core.apply(s, a, 1) for s, a in pairs], 5)
    handles = [core.CState(s) for s in states]
    hpairs = list(zip(handles, acts))
    cs_l = _best(lambda: [h.legal_actions() for h in handles], 5)
    cs_a = _best(lambda: [h.apply(a, 1) for h, a in hpairs], 5)
    start = new_game(4, rng=random.Random(7))
    n_py, n_cpp = 3, 20
    py_p = _best(lambda: [E.random_playout(start, random.Random(i), max_turns=100) for i in range(n_py)], 1) / n_py
    cpp_p = _best(lambda: [core.random_playout_fast(start, i, 100) for i in range(n_cpp)], 2) / n_cpp
    print(f"\nlegal_actions: python {py_l / n * 1e6:.1f} us/call, c++ {cpp_l / n * 1e6:.1f} us/call, "
          f"speedup {py_l / cpp_l:.1f}x ({n} states, avg {sum(len(E.legal_actions(s)) for s in states) / n:.1f} actions)")
    print(f"apply:         python {py_a / n * 1e6:.1f} us/call, c++ {cpp_a / n * 1e6:.1f} us/call, "
          f"speedup {py_a / cpp_a:.1f}x")
    print(f"CState (no GameState conversion): legal_actions {cs_l / n * 1e6:.2f} us/call ({py_l / cs_l:.0f}x), "
          f"apply {cs_a / n * 1e6:.2f} us/call ({py_a / cs_a:.0f}x)")
    print(f"random playout (4 players, 100 turns): python {py_p * 1e3:.1f} ms/game, c++ {cpp_p * 1e3:.2f} ms/game, "
          f"speedup {py_p / cpp_p:.0f}x")
    assert py_p / cpp_p > 5.0, "C++ playouts should be far faster than Python ones"
    assert cpp_l < 2 * py_l and cpp_a < 2 * py_a, "the C++ calls must not be slower than the Python engine"
