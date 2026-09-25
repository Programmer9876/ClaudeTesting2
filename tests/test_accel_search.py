"""Differential tests: the native lookahead of ``catanbot_core`` (``future_values`` and the strategy ports it uses)
against the Python reference ``catanbot.search.Searcher._future_values`` and ``discard`` / ``danger`` / ``robber`` /
``politics`` / ``opponent_model``.

Skipped when the extension is not built with the lookahead (``scripts/build_cpp.sh``).  Two layers:

* **Component parity (exact)** on every state of heuristic-bot games (2 / 3 / 4 players, hidden-hand and
  terminal variants): ``choose_discard``, ``needed_vector``, ``win_path`` (numeric fields), ``best_robber_move``
  (hex, victim) with and without politics / habit weights, the robber weight chain itself, the evaluator handles
  (heuristic bitwise, value net to float32 precision, blend).
* **End-to-end parity (exact)**: ``_future_values`` in *parity mode* - the Python searcher with
  ``opponent_expand=1000`` (every non-trade candidate), ``opponent_proposals=0`` and legal-order priors for the
  simulated opponents, both sides drawing from one shared ``random.Random`` - must give bit-identical values,
  node counts, leaves and rng state; ``Searcher.search`` at depth 2 must then rank identically.  The default
  configuration (prior top-k, proposals) is only *equivalent*: a trace replay checks every simulated decision
  against its Python rule and a depth-3 run reports the agreement.
"""
from __future__ import annotations

import contextlib
import os
import random
import subprocess
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np
import pytest

from catanbot import accel
from catanbot import actions as A
from catanbot import engine as E
from catanbot import search as S
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.danger import win_path
from catanbot.discard import choose_discard, needed_vector
from catanbot.heuristic import HeuristicEvaluator
from catanbot.model import ValueNet
from catanbot.opponent_model import OpponentModel
from catanbot.politics import PoliticalState
from catanbot.robber import best_robber_move
from catanbot.search import SearchConfig, Searcher, reduced_config
from catanbot.selfplay import BlendedEvaluator, make_bot, play_game
from catanbot.state import GameState, new_game, PHASE_DISCARD, PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER, PHASE_ROLL
from tests.test_accel_engine import _Forced

core = accel.load_core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "future_values"),
                                reason="catanbot_core is not built with the native lookahead (run scripts/build_cpp.sh)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "value_net_candidate.npz")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def native_available(monkeypatch):
    """The extension is loaded and the native lookahead enabled, whatever the environment says."""
    monkeypatch.setattr(accel, "_core", core)
    monkeypatch.setattr(accel, "AVAILABLE", True)
    monkeypatch.setattr(accel, "_verified", False)
    monkeypatch.delenv(accel.NATIVE_SEARCH_ENV, raising=False)
    assert accel.native_search_available()


def _play(seed: int, n: int, max_turns: int = 220) -> List[GameState]:
    rng = random.Random(seed)
    state = new_game(n, rng=rng)
    states: List[GameState] = []

    def on_action(s: GameState, a, i: int) -> None:
        states.append(s.copy())

    bots = [make_bot("heuristic:temp=0.7,eps=0.1") for _ in range(n)]
    play_game(bots, state=state, rng=rng, max_turns=max_turns, seed=seed, on_action=on_action)
    states.append(state.copy())
    return states


def _hidden_variant(s: GameState, me: int) -> GameState:
    """Opponents' hands and dev cards unknown (screenshot-style state)."""
    v = s.copy()
    for j, p in enumerate(v.players):
        if j == me:
            continue
        p.hand_size = p.total_resources
        p.hand_known = False
        p.dev_count = p.total_dev
        p.dev_known = False
    return v


@pytest.fixture(scope="module")
def game_states() -> List[GameState]:
    states = _play(1, 4) + _play(2, 3) + _play(3, 2)
    extra = [_hidden_variant(s, s.current % s.num_players) for s in states[::17]]
    return states + extra


def _positions(count: int, seed: int) -> List[GameState]:
    """Decision positions (main phase, >= 3 cards) taken at three moments of heuristic-bot games (3 / 4 players)."""
    out: List[GameState] = []
    k = 0
    while len(out) < count:
        rng = random.Random(seed + 7919 * k)
        n = 3 if k % 3 == 2 else 4
        k += 1
        s = new_game(n, rng=rng)
        bots = [HeuristicBot() for _ in range(n)]
        targets = [24, 44, 64]
        steps = 0
        while s.phase != PHASE_GAME_OVER and steps < 6000 and targets:
            if s.turn >= targets[0] and s.phase == PHASE_MAIN and s.players[s.current].total_resources >= 3 \
                    and E.acting_player(s) == s.current:
                out.append(s.copy())
                targets.pop(0)
                if len(out) >= count:
                    break
            i = E.acting_player(s)
            s = E.apply_inplace(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
            steps += 1
    return out


@pytest.fixture(scope="module")
def positions() -> List[GameState]:
    return _positions(60, 100)


def _end_of_turn_states(s: GameState, me: int, k: int = 6, rng_seed: int = 3) -> List[GameState]:
    """End-of-turn states reachable from ``s`` (a non-trade action, then END_TURN)."""
    rng = random.Random(rng_seed)
    out = []
    for a in E.legal_actions(s):
        if a[0] == A.PROPOSE_TRADE:
            continue
        try:
            s2 = E.apply(s, a, rng)
        except E.IllegalActionError:
            continue
        guard = 0
        while s2.phase != PHASE_GAME_OVER and s2.current == me and guard < 6:
            lg = E.legal_actions(s2)
            s2 = E.apply(s2, (A.END_TURN,) if (A.END_TURN,) in lg else lg[0], rng)
            guard += 1
        if s2.current != me or s2.phase == PHASE_GAME_OVER:
            out.append(s2)
        if len(out) >= k:
            break
    return out


@pytest.fixture(scope="module")
def end_of_turn(positions) -> List[Tuple[GameState, int, List[GameState]]]:
    """Per position: (position, me, end-of-turn states), including hidden-hand determinizations."""
    from catanbot.inference import sample_states
    out = []
    for pi, s in enumerate(positions[:40]):
        me = s.current
        states = _end_of_turn_states(s, me)
        if pi % 5 == 4:  # a determinized hidden-hand position
            hidden = _hidden_variant(s, me)
            for det in sample_states(hidden, me, 1, random.Random(pi), None):
                states = states + _end_of_turn_states(det, me, k=3)
        if states:
            out.append((s, me, states))
    return out


def _random_politics(n: int, seed: int) -> PoliticalState:
    pol = PoliticalState(n)
    r = random.Random(seed)
    for i in range(n):
        for j in range(n):
            if i != j:
                pol.capital[i][j] = r.uniform(-1.0, 1.0)
                pol.coalitions.favour[i][j] = r.uniform(0.0, 1.5)
    return pol


def _random_model(s: GameState, seed: int) -> OpponentModel:
    model = OpponentModel(s)
    r = random.Random(seed)
    for i in range(s.num_players):
        prof = model.profile_of(s, i)
        for j in range(s.num_players):
            if i != j and r.random() < 0.7:
                prof.robbed[s.players[j].name or s.players[j].color] = r.uniform(0.0, 3.0)
        for _ in range(r.randrange(3)):
            prof.robs_leader.add(r.random())
    return model


def _legal_order_priors(state, actions, player=None, belief=None, model=None, politics=None):
    return [-i for i in range(len(actions))]


@contextlib.contextmanager
def _opponents_in_legal_order():
    """``action_priors`` seen by ``_greedy_turn`` returns legal-order priors (parity mode)."""
    saved = S.action_priors
    S.action_priors = _legal_order_priors
    try:
        yield
    finally:
        S.action_priors = saved


def _parity_pair(ev, depth, model=None, politics=None, seed=1, **kw):
    """A Python and a native searcher in parity mode sharing one random stream."""
    cfg_py = SearchConfig(depth=depth, beam=4, expand=8, opponent_expand=1000, opponent_proposals=0, native_future=False, **kw)
    cfg_nat = SearchConfig(depth=depth, beam=4, expand=8, opponent_expand=1000, opponent_proposals=0, native_future=True, **kw)
    se_py = Searcher(ev, cfg_py, model, None, politics)
    se_nat = Searcher(ev, cfg_nat, model, None, politics)
    assert not se_py.native_active and se_nat.native_active
    se_py._rng = random.Random(seed)
    se_nat._rng = random.Random(seed)
    se_nat._native_rng = lambda se=se_nat: se._rng      # the shared random.Random instead of a seed
    orig_fv = se_py._future_values

    def fv(states, me, d, _o=orig_fv):
        with _opponents_in_legal_order():
            return _o(states, me, d)

    se_py._future_values = fv
    return se_py, se_nat


def _tw_for(state, actor, politics, model):
    if politics is not None:
        return politics.robber_target_weights(state, actor, model=model)
    if model is not None:
        return model.robber_habit_weights(state, actor)
    return None


# ---------------------------------------------------------------------------
# component parity (exact)
# ---------------------------------------------------------------------------
def test_choose_discard_and_needed_vector_match_python(game_states):
    checked = 0
    for s in game_states:
        for i in range(s.num_players):
            assert core.choose_discard(s, i) == choose_discard(s, i), (s.turn, i)
            assert list(core.needed_vector(s, i)) == list(needed_vector(s, i)), (s.turn, i)
            checked += 1
        if s.phase == PHASE_DISCARD:
            acting = E.acting_player(s)
            legal = E.legal_actions(s)
            assert core.choose_discard(s, acting, legal=True) == choose_discard(s, acting, legal_actions=legal)
    assert checked > 3000


def test_win_path_matches_python(game_states):
    fields = ("vp", "need_vp", "cost", "hand", "hand_known", "missing", "need_share", "eff_need", "prod", "supply",
              "turns", "min_turns", "can_win_now", "danger", "blocked_now")
    checked = 0
    for s in game_states[::2]:
        for i in range(s.num_players):
            wp = win_path(s, i)
            d = core.win_path(s, i)
            for f in fields:
                assert d[f] == getattr(wp, f), (s.turn, i, f, d[f], getattr(wp, f))
            assert d["n_steps"] == len(wp.steps)
            checked += 1
    assert checked > 1500


def test_best_robber_move_matches_python(game_states):
    checked = 0
    for k, s in enumerate(game_states[::3]):
        n = s.num_players
        pol = _random_politics(n, k) if k % 3 == 1 else None
        model = _random_model(s, k) if k % 3 != 0 else None
        bundle = accel.robber_weights_bundle(s, pol, model)
        for i in range(n):
            tw = _tw_for(s, i, pol, model)
            if bundle is not None:
                assert list(core.robber_weights(s, i, bundle)) == list(tw), (s.turn, i)
            h, v, _ = best_robber_move(s, i, target_weights=tw)
            nh, nv, score = core.best_robber_move(s, i, bundle)
            assert (nh, nv) == (h, v), (s.turn, i, (nh, nv), (h, v), score)
            checked += 1
    assert checked > 1000


def test_robber_weight_chain_is_exact_for_every_combination(game_states):
    """politics only / politics + model / model only, on every state (the leader is recomputed per state)."""
    for k, s in enumerate(game_states[::7]):
        n = s.num_players
        pol = _random_politics(n, 1000 + k)
        model = _random_model(s, 2000 + k)
        for p, m in ((pol, None), (pol, model), (None, model)):
            bundle = accel.robber_weights_bundle(s, p, m)
            for actor in range(n):
                assert list(core.robber_weights(s, actor, bundle)) == list(_tw_for(s, actor, p, m))
    assert accel.robber_weights_bundle(game_states[0], None, None) is None


def test_heuristic_eval_handle_is_bitwise(game_states):
    sample = game_states[::5]
    states = [s for s in sample for _ in range(s.num_players)]
    players = [i for s in sample for i in range(s.num_players)]
    for temp in (16.0, 4.0):
        ev = HeuristicEvaluator(temp)
        py = accel._python(ev.evaluate, states, players)   # the pure-Python reference
        h = accel.native_evaluator(ev)
        assert h is not None and h.kind == "heuristic"
        nat = h.evaluate(states, players)
        assert nat.dtype == np.float64 and nat.shape == (len(states),)
        assert np.array_equal(nat, py)
    assert len(states) > 800


def _nets():
    nets = [("fresh 256/128", ValueNet(seed=0)), ("fresh 64/32", ValueNet(hidden=(64, 32), seed=3))]
    if os.path.exists(MODEL_PATH):
        nets.append(("candidate", ValueNet.load(MODEL_PATH)))
    return nets


def test_mlp_eval_matches_valuenet_to_float32_precision(game_states):
    from catanbot.features import extract_batch
    sample = game_states[::2]
    states = [s for s in sample for _ in range(s.num_players)][:1200]
    players = [i for s in sample for i in range(s.num_players)][:1200]
    assert len(states) >= 1000
    X = extract_batch(states, players)
    for label, net in _nets():
        h = accel.native_evaluator(net)
        assert h is not None and h.kind == "mlp", label
        py_p = net.predict(X).astype(np.float64)
        py_z = net.logits(X).astype(np.float64)
        nat_p = h.evaluate(states, players)
        nat_z = h.logits(states, players).astype(np.float64)
        assert nat_p.dtype == np.float64
        dp = float(np.max(np.abs(nat_p - py_p)))
        dz = float(np.max(np.abs(nat_z - py_z)))
        print(f"MlpEval vs ValueNet.predict ({label}, hidden {net.hidden}): {len(states)} rows, "
              f"max |dp| = {dp:.2e}, max |dlogit| = {dz:.2e}")
        assert dp <= 1e-6, (label, dp)
        assert dz <= 1e-5, (label, dz)
        # the whole evaluator interface, like the search uses it
        assert np.allclose(h.evaluate(states[:50], players[:50]), net.evaluate(states[:50], players[:50]), atol=1e-6)
        blend = BlendedEvaluator(net, 0.35)
        hb = accel.native_evaluator(blend)
        assert hb is not None and hb.kind == "blend"
        assert np.max(np.abs(hb.evaluate(states, players) - blend.evaluate(states, players))) <= 1e-6


def test_evaluator_adapters_reject_unknown_objects_and_track_replaced_weights(game_states):
    class Wrapper:
        name = "heuristic"

        def __init__(self, inner):
            self.inner = inner

        def evaluate(self, states, players):
            return self.inner.evaluate(states, players)

    assert accel.native_evaluator(Wrapper(HeuristicEvaluator())) is None
    assert accel.native_evaluator(ValueNet(seed=0, dtype=np.float64)) is None
    assert accel.native_evaluator(object()) is None
    assert accel.native_evaluator(BlendedEvaluator(ValueNet(seed=0, dtype=np.float64), 0.5)) is None
    net = ValueNet(hidden=(16, 8), seed=5)
    se = Searcher(net, SearchConfig(depth=1))
    assert se.native_active
    key = se._native_key
    net.set_params(net.get_params() * 0.5)           # arrays replaced -> the handle is rebuilt on the next search
    s = game_states[200]
    se.search(s, E.acting_player(s), random.Random(1))
    assert se._native_key != key and se.native_active
    h = se._native_ev
    assert np.allclose(h.evaluate([s], [0]), net.evaluate([s], [0]), atol=1e-6)


# ---------------------------------------------------------------------------
# end-to-end parity (exact, parity mode)
# ---------------------------------------------------------------------------
def test_future_values_bitwise_parity(end_of_turn):
    ev = HeuristicEvaluator()
    checked = 0
    n_positions = 0
    t_py = t_nat = 0.0
    for pi, (s, me, states) in enumerate(end_of_turn):
        n_positions += 1
        for variant in ("plain", "politics", "habit"):
            model = _random_model(s, pi) if variant != "plain" else None
            pol = _random_politics(s.num_players, pi) if variant == "politics" else None
            se_py, se_nat = _parity_pair(ev, 2, model, pol, seed=1000 + pi)
            t0 = time.perf_counter()
            vp = np.asarray(se_py._future_values(states, me, 1))
            t_py += time.perf_counter() - t0
            t0 = time.perf_counter()
            vn = np.asarray(se_nat._future_values(states, me, 1))
            t_nat += time.perf_counter() - t0
            np.testing.assert_array_equal(vn, vp, err_msg=f"position {pi} ({variant})")
            assert se_py.nodes == se_nat.nodes, (pi, variant)
            assert se_py._rng.getstate() == se_nat._rng.getstate(), (pi, variant)
            assert np.all((vn >= 0.0) & (vn <= 1.0))
            checked += len(states)
    print(f"future_values parity: {checked} end-of-turn states from {n_positions} positions x 3 weight variants, "
          f"python {t_py:.2f} s, native {t_nat:.2f} s")
    assert checked >= 200 and n_positions >= 12


def test_future_values_leaves_match_python_simulation(end_of_turn):
    """Trace mode: the native leaves are the states Python's _simulate_until_my_turn reaches (same rng)."""
    ev = HeuristicEvaluator()
    checked = 0
    for pi, (s, me, states) in enumerate(end_of_turn[:6]):
        se_py, se_nat = _parity_pair(ev, 2, seed=77 + pi)
        cfg = se_nat.config
        rng = random.Random(se_nat._rng.random())            # what _future_values does first
        seqs = [[rng.randint(1, 6) + rng.randint(1, 6) for _ in range(12)] for _ in range(cfg.opp_roll_samples)]
        se_py._rng.random()
        levels = [{k: int(getattr(cfg, k)) for k in S._NATIVE_LEVEL_FIELDS}]
        vals, nodes, steps, leaves, leaf_vals = core.future_values(states, me, 1, levels, seqs, se_nat._native_ev, None,
                                                                   se_nat._rng, None, None, True)
        k = 0
        with _opponents_in_legal_order():
            for s0 in states:
                for seq in seqs:
                    leaf = se_py._simulate_until_my_turn(s0, me, seq)
                    assert leaves[k].to_state().to_dict() == leaf.to_dict(), (pi, k)
                    assert float(ev.evaluate([leaf], [me])[0]) == leaf_vals[k]
                    k += 1
                    checked += 1
        assert se_py._rng.getstate() == se_nat._rng.getstate()
        assert len(vals) == len(states) and nodes == se_py.nodes
    assert checked >= 100


def test_search_depth2_ranks_identically_in_parity_mode(positions):
    ev = HeuristicEvaluator()
    agree = 0
    t_py = t_nat = 0.0
    for pi, s in enumerate(positions):
        me = s.current
        se_py, se_nat = _parity_pair(ev, 2, seed=5 + pi)
        t0 = time.perf_counter()
        rp = se_py.search(s, me, random.Random(1))
        t_py += time.perf_counter() - t0
        t0 = time.perf_counter()
        rn = se_nat.search(s, me, random.Random(1))
        t_nat += time.perf_counter() - t0
        assert [r.action for r in rp] == [r.action for r in rn], pi
        assert [r.value for r in rp] == [r.value for r in rn], pi
        assert se_py.nodes == se_nat.nodes
        assert se_py._shift == se_nat._shift
        agree += 1
    print(f"depth-2 search parity: {agree}/{len(positions)} positions identical (python {t_py:.2f} s, native {t_nat:.2f} s)")
    assert agree == len(positions) >= 50


# ---------------------------------------------------------------------------
# default configuration: trace replay + agreement (equivalent, not identical)
# ---------------------------------------------------------------------------
def test_trace_replays_through_the_python_engine_with_the_python_rules(end_of_turn):
    ev = HeuristicEvaluator()
    decisions = {"roll": 0, "discard": 0, "robber": 0, "main": 0}
    for pi, (s, me, states) in enumerate(end_of_turn[:12]):
        pol = _random_politics(s.num_players, pi) if pi % 2 else None
        model = _random_model(s, pi) if pi % 3 else None
        cfg = SearchConfig(depth=2, beam=4, expand=8)
        levels = [{k: int(getattr(cfg, k)) for k in S._NATIVE_LEVEL_FIELDS}]
        rng = random.Random(pi)
        seqs = [[rng.randint(1, 6) + rng.randint(1, 6) for _ in range(12)] for _ in range(4)]
        bundle = accel.robber_weights_bundle(s, pol, model)
        ev_h = accel.native_evaluator(ev)
        vals, nodes, steps, leaves, leaf_vals = core.future_values(states, me, 1, levels, seqs, ev_h, bundle,
                                                                   random.Random(pi + 1), None, None, True)
        assert len(steps) == len(states) * len(seqs) == len(leaves) == len(leaf_vals)
        assert nodes >= sum(len(st) for st in steps) > 0     # every trial candidate counts, the trace keeps the chosen ones
        for k, st in enumerate(steps):
            s0 = states[k // len(seqs)]
            seq = seqs[k % len(seqs)]
            cur = s0
            ri = 0
            turn_of = None
            for player, action, draw in st:
                assert player == E.acting_player(cur)
                legal = E.legal_actions(cur)
                if cur.phase == PHASE_ROLL:
                    if cur.current != turn_of:
                        turn_of = cur.current
                    assert action == (A.ROLL, seq[ri % len(seq)])
                    ri += 1
                    decisions["roll"] += 1
                elif cur.phase == PHASE_DISCARD:
                    assert action == choose_discard(cur, player, legal_actions=legal)
                    decisions["discard"] += 1
                elif cur.phase == PHASE_ROBBER:
                    tw = _tw_for(cur, player, pol, model)
                    h, v, _ = best_robber_move(cur, player, target_weights=tw)
                    expect = (A.MOVE_ROBBER, h, v)
                    assert action == (expect if expect in legal else legal[0])
                    decisions["robber"] += 1
                else:
                    assert action in legal and action[0] != A.PROPOSE_TRADE
                    if cur.phase == PHASE_MAIN:
                        decisions["main"] += 1
                cur = E.apply(cur, action, _Forced(draw))
            assert leaves[k].to_state().to_dict() == cur.to_dict(), (pi, k)
            assert cur.phase == PHASE_GAME_OVER or (cur.current == me and cur.phase == PHASE_ROLL) or \
                len(st) >= 3 * cfg.opponent_actions + 6
            assert float(ev.evaluate([cur], [me])[0]) == leaf_vals[k]
        # the returned values are the per-state means of the leaf values
        for i in range(len(states)):
            acc = 0.0
            for j in range(len(seqs)):
                acc += leaf_vals[i * len(seqs) + j]
            assert vals[i] == acc / len(seqs)
    print("trace replay decisions:", decisions)
    assert decisions["roll"] > 100 and decisions["main"] > 100 and decisions["robber"] > 0


def test_default_config_depth2_and_depth3_agree_with_python(positions):
    """Default configuration (prior top-k + proposals in Python vs every candidate natively): equivalence report."""
    ev = HeuristicEvaluator()
    report = {}
    for depth in (2, 3):
        agree = 0
        dv = []
        t_py = t_nat = 0.0
        subset = positions[:24] if depth == 2 else positions[:16]
        for pi, s in enumerate(subset):
            me = s.current
            cfg_py = SearchConfig(depth=depth, beam=4, expand=8, native_future=False)
            cfg_nat = SearchConfig(depth=depth, beam=4, expand=8, native_future=True)
            se_py, se_nat = Searcher(ev, cfg_py), Searcher(ev, cfg_nat)
            t0 = time.perf_counter()
            rp = se_py.search(s, me, random.Random(1))
            t_py += time.perf_counter() - t0
            t0 = time.perf_counter()
            rn = se_nat.search(s, me, random.Random(1))
            t_nat += time.perf_counter() - t0
            assert all(0.0 <= r.value <= 1.0 for r in rn)
            assert {r.action for r in rp} == {r.action for r in rn}
            agree += rp[0].action == rn[0].action
            vn = {r.action: r.value for r in rn}
            dv.append(abs(rp[0].value - vn[rp[0].action]))
        report[depth] = (agree, len(subset), float(np.median(dv)), t_py, t_nat)
        print(f"depth {depth} default config: best action agreement {agree}/{len(subset)}, median |dvalue| "
              f"{np.median(dv):.4f}, python {t_py:.2f} s, native {t_nat:.2f} s")
    agree2, n2, med2, _, _ = report[2]
    agree3, n3, med3, _, _ = report[3]
    # Python vs Python with different dice seeds agrees on ~11/16 positions at both depths (the noise floor of
    # the sampled lookahead); the native path is held to a looser bound and the report above is the real result.
    assert agree2 >= 0.5 * n2 and med2 < 0.05
    assert agree3 >= 0.35 * n3 and med3 < 0.1


# ---------------------------------------------------------------------------
# invariants, budget, fallbacks
# ---------------------------------------------------------------------------
def test_reduced_search_invariants_and_budget(positions):
    ev = HeuristicEvaluator()
    h = accel.native_evaluator(ev)
    cfg = SearchConfig(depth=3, beam=4, expand=8)
    sub = reduced_config(cfg, 2, 5000)
    lv = [{k: int(getattr(c, k)) for k in S._NATIVE_LEVEL_FIELDS} for c in (cfg, sub)]
    for s in positions[:8]:
        me = s.current
        legal = E.legal_actions(s)
        v, nodes = core.reduced_search(s, me, 2, [lv[1]], h, None, 7, None, None)
        assert 0.0 <= v <= 1.0 and nodes > 0 and len(legal) > 1
        # a past deadline: nothing is expanded, the static value comes back
        v2, nodes2 = core.reduced_search(s, me, 2, [lv[1]], h, None, 7, 1.0, None)
        assert nodes2 == 0 and v2 == float(ev.evaluate([s], [me])[0])
        # a zero node budget on the reduced level: depth 2 collapses to the leaf evaluation of depth 1
        eot = _end_of_turn_states(s, me, k=3)
        if not eot:
            continue
        rolls = [[6, 8, 7, 5, 9, 4, 10, 3, 11, 2, 12, 6]]
        v1, n1 = core.future_values(eot, me, 1, [lv[0]], rolls, h, None, 3, None, None, False)
        capped = dict(lv[1], max_nodes=0)
        v3, n3 = core.future_values(eot, me, 2, [lv[0], capped], rolls, h, None, 3, None, None, False)
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v3))
        assert n1 == n3
        # the real depth-2 lookahead does more work and stays in range
        v4, n4 = core.future_values(eot, me, 2, lv, rolls, h, None, 3, None, None, False)
        assert n4 > n1 and np.all((np.asarray(v4) >= 0.0) & (np.asarray(v4) <= 1.0))
        # the global node budget stops the reduced searches (values still valid)
        v5, n5 = core.future_values(eot, me, 2, lv, rolls, h, None, 3, None, n1 + 1, False)
        assert n5 <= n4 and np.all((np.asarray(v5) >= 0.0) & (np.asarray(v5) <= 1.0))


def test_searcher_accounts_nodes_and_shift_with_the_native_path(positions):
    ev = HeuristicEvaluator()
    s = positions[3]
    me = s.current
    se = Searcher(ev, SearchConfig(depth=3, beam=4, expand=8))
    assert se.native_active
    calls = []
    real = accel.future_values

    def spy(*args, **kw):
        out = real(*args, **kw)
        calls.append(out[1])
        return out

    accel.future_values = spy
    try:
        before = se.nodes
        res = se.search(s, me, random.Random(3))
    finally:
        accel.future_values = real
    assert calls and se.nodes >= before + sum(calls)
    assert res and all(0.0 <= r.value <= 1.0 for r in res)
    assert np.isfinite(se._shift)


def test_python_path_is_used_for_unsupported_states_and_unknown_evaluators(monkeypatch):
    ev = HeuristicEvaluator()
    five = new_game(5, rng=random.Random(1))
    bots = [HeuristicBot() for _ in range(5)]
    rng = random.Random(2)
    while not (five.phase == PHASE_MAIN and five.current == 0 and five.turn >= 20):
        i = E.acting_player(five)
        five = E.apply_inplace(five, bots[i].decide(five, E.legal_actions(five), rng), rng)
    se = Searcher(ev, SearchConfig(depth=2, beam=3, expand=6))
    assert se.native_active
    seen = []
    orig = se._native_future_values

    def spy(states, me, depth, seqs):
        out = orig(states, me, depth, seqs)
        seen.append(out)
        return out

    se._native_future_values = spy
    res = se.search(five, 0, random.Random(1))
    assert res and seen and all(o is None for o in seen)     # UnsupportedStateError -> Python body
    ref = Searcher(ev, SearchConfig(depth=2, beam=3, expand=6, native_future=False)).search(five, 0, random.Random(1))
    assert [r.action for r in res] == [r.action for r in ref] and [r.value for r in res] == [r.value for r in ref]

    class Custom:
        name = "custom"

        def evaluate(self, states, players):
            return ev.evaluate(states, players)

    assert not Searcher(Custom(), SearchConfig(depth=2)).native_active
    assert not Searcher(ev, SearchConfig(depth=2, native_future=False)).native_active


def test_switches_disable_the_native_path(monkeypatch, positions):
    ev = HeuristicEvaluator()
    s = positions[0]
    monkeypatch.setenv(accel.NATIVE_SEARCH_ENV, "1")
    assert not accel.native_search_available()
    assert not Searcher(ev, SearchConfig(depth=2)).native_active
    monkeypatch.delenv(accel.NATIVE_SEARCH_ENV)
    assert accel.native_search_available()
    # CATANBOT_NO_ACCEL semantics: accel.AVAILABLE False -> Python everywhere (like the other accel tests)
    monkeypatch.setattr(accel, "AVAILABLE", False)
    se = Searcher(ev, SearchConfig(depth=2, beam=4, expand=8))
    assert not se.native_active
    res = se.search(s, s.current, random.Random(1))
    assert res and all(0.0 <= r.value <= 1.0 for r in res)
    assert accel.native_evaluator(ev) is None
    monkeypatch.setattr(accel, "AVAILABLE", True)
    # a stale build (older .so without the lookahead) is disabled as a whole by verify()
    from catanbot import features as F

    class OldCore:
        UnsupportedStateError = ValueError

        @staticmethod
        def num_features():
            return F.NUM_FEATURES

        @staticmethod
        def feature_names():
            return list(F.FEATURE_NAMES)

    for name in accel._REQUIRED:
        if name not in ("future_values", "HeuristicEval", "MlpEval", "BlendEval", "UnsupportedStateError"):
            setattr(OldCore, name, staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(accel, "_core", OldCore)
    monkeypatch.setattr(accel, "_verified", False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert accel.native_search_available() is False
    assert accel.AVAILABLE is False and any(issubclass(x.category, RuntimeWarning) for x in w)
    assert not Searcher(ev, SearchConfig(depth=2)).native_active


def test_no_accel_env_runs_the_python_path_in_a_subprocess():
    code = (
        "import random\n"
        "from catanbot import accel\n"
        "from catanbot.heuristic import HeuristicEvaluator\n"
        "from catanbot.search import Searcher, SearchConfig\n"
        "from tests.test_accel_search import _positions\n"
        "s = _positions(1, 3)[0]\n"
        "se = Searcher(HeuristicEvaluator(), SearchConfig(depth=2, beam=3, expand=6))\n"
        "assert not accel.AVAILABLE and not accel.native_search_available() and not se.native_active\n"
        "res = se.search(s, s.current, random.Random(1))\n"
        "assert res and 0.0 <= res[0].value <= 1.0\n"
        "print('python path ok', res[0].action)\n"
    )
    env = dict(os.environ, CATANBOT_NO_ACCEL="1", PYTHONPATH=ROOT)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "python path ok" in proc.stdout


def test_search_suite_behaviours_hold_with_the_native_path(positions):
    """A few behaviours of tests/test_search.py exercised with native_future on: winning city, roll phase."""
    from tests.test_search import mid_game
    ev = HeuristicEvaluator()
    s = mid_game(5)
    s.players[0].settlements = s.players[0].settlements[:1] + s.players[0].settlements[1:]
    for depth in (2, 3):
        se = Searcher(ev, SearchConfig(depth=depth, beam=4, expand=8))
        assert se.native_active
        res = se.search(s, 0, random.Random(1))
        assert res and 0.0 <= res[0].value <= 1.0
    # the roll phase
    r = positions[1]
    bots = [HeuristicBot() for _ in range(r.num_players)]
    rng = random.Random(9)
    st = E.apply(r, (A.END_TURN,), rng)
    guard = 0
    while not (st.phase == PHASE_ROLL and st.current == r.current) and st.phase != PHASE_GAME_OVER and guard < 300:
        i = E.acting_player(st)
        st = E.apply_inplace(st, bots[i].decide(st, E.legal_actions(st), rng), rng)
        guard += 1
    if st.phase == PHASE_ROLL:
        res = Searcher(ev, SearchConfig(depth=2, beam=4, expand=8)).search(st, st.current, random.Random(1))
        assert res and res[0].action[0] in (A.ROLL, A.PLAY_KNIGHT)


# ---------------------------------------------------------------------------
# benchmark (printed)
# ---------------------------------------------------------------------------
def test_benchmark_future_values(end_of_turn):
    ev = HeuristicEvaluator()
    states = [st for _, _, sts in end_of_turn[:10] for st in sts][:24]
    me = end_of_turn[0][1]
    s0 = end_of_turn[0][0]
    states = _end_of_turn_states(s0, me, k=12)
    cfg_py = SearchConfig(depth=2, beam=4, expand=8, native_future=False)
    cfg_nat = SearchConfig(depth=2, beam=4, expand=8, native_future=True)
    se_py, se_nat = Searcher(ev, cfg_py), Searcher(ev, cfg_nat)
    t0 = time.perf_counter()
    se_py._future_values(states, me, 1)
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    se_nat._future_values(states, me, 1)
    t_nat = time.perf_counter() - t0
    print(f"_future_values on {len(states)} end-of-turn states x {cfg_py.opp_roll_samples} samples: python {t_py * 1e3:.1f} ms, "
          f"native {t_nat * 1e3:.1f} ms ({t_py / max(t_nat, 1e-9):.1f}x)")
    assert t_nat < t_py


# ---------------------------------------------------------------------------
# regressions found by the adversarial verification (independent fuzz, see docs/CPP.md "Verification")
# ---------------------------------------------------------------------------
def _finished_copy(s: GameState, winner: int) -> GameState:
    f = s.copy()
    f.phase = PHASE_GAME_OVER
    f.winner = winner
    return f


def test_net_and_blend_handles_are_exact_on_finished_games(positions):
    """ValueNet.evaluate forces 1 / 0 once the game is decided; the native twins must too (the first build
    returned the net's own guess, up to 0.87 off, exactly on the leaves where an opponent wins)."""
    net = ValueNet(hidden=(16, 8), seed=11)
    blend = BlendedEvaluator(net, 0.3, HeuristicEvaluator(8.0))
    states, players = [], []
    for k, s in enumerate(positions[:12]):
        w = k % s.num_players
        f = _finished_copy(s, w)
        for p in range(s.num_players):
            states.append(f)
            players.append(p)
        states.append(s)
        players.append(w)
    for ev in (net, blend):
        h = accel.native_evaluator(ev)
        assert h is not None
        got = np.asarray(h.evaluate(states, players), dtype=np.float64)
        want = np.asarray(ev.evaluate(states, players), dtype=np.float64)
        over = np.array([s.phase == PHASE_GAME_OVER for s in states])
        assert np.array_equal(got[over], want[over]) and set(got[over].tolist()) <= {0.0, 1.0}
        assert np.abs(got[~over] - want[~over]).max() <= 1e-6


def test_input_mask_is_folded_into_the_native_net(positions):
    from catanbot.model import feature_mask, HAND_BLIND_FEATURES
    from catanbot.features import extract_batch
    states = [s for s in positions[:20] for _ in range(s.num_players)]
    players = [i for s in positions[:20] for i in range(s.num_players)]
    net = ValueNet(hidden=(16, 8), seed=2)
    net.mean = np.abs(np.asarray(extract_batch(states, players))).mean(axis=0).astype(np.float32) + 0.5
    key_plain = accel.evaluator_key(net)
    plain = np.asarray(accel.native_evaluator(net).evaluate(states, players))
    net.input_mask = feature_mask(HAND_BLIND_FEATURES)
    assert net.input_mask is not None and accel.evaluator_key(net) != key_plain
    masked = np.asarray(accel.native_evaluator(net).evaluate(states, players))
    want = np.asarray(net.evaluate(states, players), dtype=np.float64)
    assert np.abs(masked - want).max() <= 1e-6
    assert np.abs(plain - want).max() > 1e-3          # the mask matters on these states
    # A searcher built before the mask was set rebuilds its twin on the next search.
    net2 = ValueNet(hidden=(16, 8), seed=2)
    se = Searcher(net2, SearchConfig(depth=1))
    assert se.native_active
    net2.input_mask = feature_mask(HAND_BLIND_FEATURES)
    s = positions[0]
    se.search(s, E.acting_player(s), random.Random(1))
    got = np.asarray(se._native_ev.evaluate(states, players))
    assert np.abs(got - np.asarray(net2.evaluate(states, players), dtype=np.float64)).max() <= 1e-6


def test_pending_trade_response_states_take_the_python_path(game_states):
    """An end-of-turn state in which other responders still have to answer an offer: the Python simulation
    asks should_accept for them, the extension only rejects, so _future_values must run the Python body."""
    from catanbot.state import PHASE_TRADE_RESPONSE
    found = None
    for s in game_states:
        if s.num_players < 3 or s.phase != PHASE_MAIN:
            continue
        props = [a for a in E.legal_actions(s) if a[0] == A.PROPOSE_TRADE]
        if not props:
            continue
        s1 = E.apply(s, props[0], random.Random(0))
        if s1.phase != PHASE_TRADE_RESPONSE or s1.pending_trade is None:
            continue
        me = E.acting_player(s1)
        s2 = E.apply(s1, (A.REJECT_TRADE,), random.Random(0))
        if s2.phase == PHASE_TRADE_RESPONSE and s2.pending_trade is not None and E.acting_player(s2) != me:
            found = (s2, me)
            break
    assert found is not None
    s2, me = found
    ev = HeuristicEvaluator()
    se_n = Searcher(ev, SearchConfig(depth=2, opponent_expand=1000, opponent_proposals=0))
    se_p = Searcher(ev, SearchConfig(depth=2, opponent_expand=1000, opponent_proposals=0, native_future=False))
    assert se_n.native_active
    assert se_n._native_future_values([s2], me, 1, [[7] * 12]) is None
    se_n._rng = random.Random(3)
    se_p._rng = random.Random(3)
    with _opponents_in_legal_order():
        vn = se_n._future_values([s2], me, 1)
        vp = se_p._future_values([s2], me, 1)
    assert vn == vp and se_n.nodes == se_p.nodes and se_n._rng.getstate() == se_p._rng.getstate()
    # ... and an ordinary end-of-turn state next to it still runs natively.
    plain = E.apply(s2, (A.REJECT_TRADE,), random.Random(0))
    if not (plain.phase == PHASE_TRADE_RESPONSE and plain.pending_trade is not None):
        assert se_n._native_future_values([plain], me, 1, [[7] * 12]) is not None


def test_subclasses_overriding_evaluate_keep_the_python_path(positions):
    class Noisy(HeuristicEvaluator):
        def evaluate(self, states, players):
            return np.asarray(super().evaluate(states, players)) * 0.5

    class NetPlus(ValueNet):
        def evaluate(self, states, players):
            return np.asarray(super().evaluate(states, players)) * 0.5

    class BlendPlus(BlendedEvaluator):
        def evaluate(self, states, players):
            return np.asarray(super().evaluate(states, players)) * 0.5

    for ev in (Noisy(), NetPlus(hidden=(8,), seed=1), BlendPlus(ValueNet(hidden=(8,), seed=1), 0.5),
               BlendedEvaluator(NetPlus(hidden=(8,), seed=1), 0.5), BlendedEvaluator(ValueNet(hidden=(8,), seed=1), 0.5, Noisy())):
        assert accel.evaluator_key(ev) is None and accel.native_evaluator(ev) is None
        assert not Searcher(ev, SearchConfig(depth=2)).native_active
    # Plain instances (and a subclass that does not touch evaluate) keep the native path.
    class Tagged(HeuristicEvaluator):
        tag = "x"

    assert Searcher(Tagged(), SearchConfig(depth=2)).native_active
    assert Searcher(BlendedEvaluator(ValueNet(hidden=(8,), seed=1), 0.5), SearchConfig(depth=2)).native_active
