"""Tests for catanbot.selfplay: sibling-afterstate recording for the value net's ranking term."""
from __future__ import annotations

import random

import numpy as np

from catanbot import actions as A
from catanbot import features as F
from catanbot.selfplay import (SIBLING_KINDS, SIBLING_KIND_OTHER, SIBLING_KIND_SETUP, SIBLING_OTHER_KINDS, GameResult,
                               make_bot, play_game, sibling_arrays, sibling_kind_id)

END = sibling_kind_id((A.END_TURN,))
ACCEPT, REJECT = sibling_kind_id((A.ACCEPT_TRADE,)), sibling_kind_id((A.REJECT_TRADE,))
ROBBER, DISCARD = sibling_kind_id((A.MOVE_ROBBER, 0, -1)), sibling_kind_id((A.DISCARD, (0, 0, 0, 0, 0)))


def _game(sibling_rate: float, seed: int = 3, **kw) -> GameResult:
    bots = [make_bot("heuristic:temp=0.3,eps=0.03") for _ in range(4)]
    return play_game(bots, rng=random.Random(seed), record=True, sample_every=2, seed=seed, sibling_rate=sibling_rate,
                     **kw)


def _node_kinds(r: GameResult):
    """{node id: set of kinds} of a result's sibling nodes."""
    out = {}
    for n, k in zip(r.s_node.tolist(), r.s_kind.tolist()):
        out.setdefault(n, set()).add(k)
    return out


def test_sibling_recording_main_phase():
    r = _game(0.3)
    r0 = _game(0.0)
    # recording siblings does not change the game (separate rng) nor the ordinary samples
    assert r.winner == r0.winner and r.turns == r0.turns and np.array_equal(r.X, r0.X)
    assert r0.Xs is None and r0.s_node is None
    assert r.Xs is not None and r.Xs.dtype == np.float16 and r.Xs.shape[1] == F.NUM_FEATURES
    assert r.Xm is not None and r.Xm.shape == r.Xs.shape and r.Xm.dtype == np.float16
    n = len(r.s_node)
    assert n >= 20 and r.s_h.shape == (n,) and r.s_kind.shape == (n,) and r.s_kind.dtype == np.int8
    assert r.s_h.min() >= 0.0 and r.s_h.max() <= 1.0
    nodes = np.unique(r.s_node)
    assert nodes.min() == 0 and len(nodes) == r.s_node.max() + 1 and len(nodes) >= 5
    X = r.Xs.astype(np.float32)
    Xm = r.Xm.astype(np.float32)
    assert END == 0
    kinds_of = _node_kinds(r)
    main_nodes = [nd for nd, ks in kinds_of.items() if END in ks]
    assert len(main_nodes) >= 5
    main_rows = np.isin(r.s_node, main_nodes)
    # every main-phase sibling is the state after the action *and* END_TURN: the next player's roll phase, so
    # no "still my turn" feature can separate END_TURN from the other actions (a sibling that wins the game ends
    # in game over on our own turn, which is the only exception)
    over = X[:, F.feature_index("g_phase_game_over")] == 1.0
    assert np.all(X[main_rows & ~over, F.feature_index("g_my_turn")] == 0.0)
    assert np.all(X[main_rows, F.feature_index("g_phase_roll")] + X[main_rows, F.feature_index("g_phase_game_over")] == 1.0)
    # the mid-turn twin of every main-phase sibling is still my main phase (the decision state itself for END_TURN)
    over_m = Xm[:, F.feature_index("g_phase_game_over")] == 1.0
    assert np.all(Xm[main_rows & ~over_m, F.feature_index("g_my_turn")] == 1.0)
    assert np.all(Xm[main_rows & ~over_m, F.feature_index("g_phase_main")] == 1.0)
    for node in main_nodes:
        rows = r.s_node == node
        assert rows.sum() >= 2
        assert int((r.s_kind[rows] == END).sum()) == 1   # exactly one END_TURN sibling per main-phase node
        # the END_TURN sibling's mid-turn twin is the decision state: same hand as its END_TURN afterstate
        e = np.flatnonzero(rows & (r.s_kind == END))[0]
        assert Xm[e, F.feature_index("me_hand_size")] == X[e, F.feature_index("me_hand_size")]
    kinds = set(r.s_kind.tolist())
    assert sibling_kind_id((A.BUILD_ROAD, 1)) in kinds and sibling_kind_id((A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0))) in kinds
    assert sibling_kind_id((A.ROLL,)) == SIBLING_KIND_OTHER == len(SIBLING_KINDS)
    # building spends cards (roads can be free after Road Building): a settlement / city / dev-card sibling
    # always holds fewer cards than the END_TURN sibling of the same node
    hand = F.feature_index("me_hand_size")
    paid = {sibling_kind_id((A.BUILD_SETTLEMENT, 0)), sibling_kind_id((A.BUILD_CITY, 0)), sibling_kind_id((A.BUY_DEV,))}
    checked = 0
    for node in main_nodes:
        rows = np.flatnonzero(r.s_node == node)
        end_row = rows[r.s_kind[rows] == END][0]
        for i in rows:
            if r.s_kind[i] in paid:
                assert X[i, hand] < X[end_row, hand]
                checked += 1
    assert checked > 0


def test_setup_siblings_and_setup_samples():
    """Every setup-settlement decision is a sibling node (the best spots, each with its setup road), and the
    ordinary samples always contain the setup states (also the setup-road ones ``sample_every`` would skip)."""
    r = _game(0.3, seed=4)
    X = r.Xs.astype(np.float32)
    Xm = r.Xm.astype(np.float32)
    setup_rows = r.s_kind == SIBLING_KIND_SETUP
    setup_nodes = np.unique(r.s_node[setup_rows])
    assert len(setup_nodes) == 8                       # 4 players x 2 placements
    for node in setup_nodes:
        rows = np.flatnonzero(r.s_node == node)
        assert 2 <= len(rows) <= 12 and np.all(r.s_kind[rows] == SIBLING_KIND_SETUP)   # setup nodes are pure
        # the finished afterstate has the settlement and the road; its mid-turn twin only the settlement
        assert np.all(X[rows, F.feature_index("me_roads")] > Xm[rows, F.feature_index("me_roads")])
        assert np.all(Xm[rows, F.feature_index("g_phase_setup_road")] == 1.0)
        assert np.all(Xm[rows, F.feature_index("g_my_turn")] == 1.0)
        assert np.all(X[rows, F.feature_index("g_phase_setup_settlement")] + X[rows, F.feature_index("g_phase_roll")] == 1.0)
        # the spots differ: distinct production per sibling (at least two distinct values per node)
        assert len(np.unique(X[rows, F.feature_index("me_prod_total")])) >= 2
    # setup decision states are always in the ordinary samples, from every perspective
    Xo = r.X.astype(np.float32)
    n_settle = int(Xo[:, F.feature_index("g_phase_setup_settlement")].sum())
    n_road = int(Xo[:, F.feature_index("g_phase_setup_road")].sum())
    assert n_settle == 8 * 4 and n_road == 8 * 4
    # setup siblings off, main-phase siblings on
    r2 = _game(0.3, seed=4, sibling_setup_rate=0.0)
    assert not np.any(r2.s_kind == SIBLING_KIND_SETUP) and np.any(r2.s_kind == END)
    # setup siblings alone (no main-phase nodes) still produce a result
    r3 = _game(0.0, seed=4, sibling_setup_rate=1.0, sibling_other_rate=0.0)
    assert r3.Xs is not None and np.all(r3.s_kind == SIBLING_KIND_SETUP) and len(np.unique(r3.s_node)) == 8


def test_other_phase_siblings():
    """Incoming offers (accept / reject), robber moves and discards are recorded as sibling nodes at one
    common horizon per node; by default at 5 x the main-phase rate."""
    r = _game(0.05, seed=5, sibling_other_rate=1.0)
    assert set(SIBLING_OTHER_KINDS) == {ACCEPT, REJECT, ROBBER, DISCARD}
    X = r.Xs.astype(np.float32)
    Xm = r.Xm.astype(np.float32)
    kinds_of = _node_kinds(r)
    trade_nodes = [nd for nd, ks in kinds_of.items() if ks & {ACCEPT, REJECT}]
    robber_nodes = [nd for nd, ks in kinds_of.items() if ROBBER in ks]
    discard_nodes = [nd for nd, ks in kinds_of.items() if DISCARD in ks]
    assert len(trade_nodes) >= 10 and len(robber_nodes) >= 2 and len(discard_nodes) >= 1
    res_cols = [F.feature_index(f"me_res_{c}") for c in ("wood", "brick", "sheep", "wheat", "ore")]
    for nd in trade_nodes:
        rows = np.flatnonzero(r.s_node == nd)
        assert sorted(r.s_kind[rows].tolist()) == [ACCEPT, REJECT]      # exactly the two answers
        # answered offers are resolved: the leaf is the proposer's turn, not ours, and no offer is pending
        assert np.all(X[rows, F.feature_index("g_my_turn")] == 0.0)
        assert np.all(X[rows, F.feature_index("g_trade_pending")] == 0.0)
        np.testing.assert_array_equal(X[rows], Xm[rows])                # already finished: one horizon
        # the counterfactual is deterministic: ACCEPT is the trade executed with me (my hand changed by the
        # offer), REJECT the offer resolved without me (my hand unchanged)
        a, b = rows[r.s_kind[rows] == ACCEPT][0], rows[r.s_kind[rows] == REJECT][0]
        assert not np.array_equal(X[a, res_cols], X[b, res_cols])
        assert X[a, F.feature_index("g_trades_this_turn")] >= X[b, F.feature_index("g_trades_this_turn")]
    for nd in robber_nodes:
        rows = np.flatnonzero(r.s_node == nd)
        assert 2 <= len(rows) <= 12 and np.all(r.s_kind[rows] == ROBBER)
        # mid-turn twin: our main phase after the move; finished sibling: after END_TURN (one horizon per node)
        assert np.all(Xm[rows, F.feature_index("g_my_turn")] == 1.0)
        assert np.all(X[rows, F.feature_index("g_my_turn")] == X[rows[0], F.feature_index("g_my_turn")])
        assert len(np.unique(X[rows, F.feature_index("me_robber_loss")].round(3) * 1000
                             + X[rows, F.feature_index("opp1_robber_loss")].round(3))) >= 2
    for nd in discard_nodes:
        rows = np.flatnonzero(r.s_node == nd)
        assert 2 <= len(rows) <= 12 and np.all(r.s_kind[rows] == DISCARD)
        np.testing.assert_array_equal(X[rows], Xm[rows])
        assert len(np.unique(X[rows][:, [F.feature_index(f"me_res_{c}") for c in ("wood", "brick", "sheep", "wheat", "ore")]], axis=0)) >= 2
    # heuristic values are per row and differ between accept and reject at least sometimes
    gaps = [abs(float(np.diff(r.s_h[r.s_node == nd])[0])) for nd in trade_nodes]
    assert max(gaps) > 0.0
    # default other-phase rate: 5 x the main rate; 0 switches them off
    r0 = _game(0.05, seed=5, sibling_other_rate=0.0)
    assert not np.any(np.isin(r0.s_kind, SIBLING_OTHER_KINDS))
    r5 = _game(0.05, seed=5)
    n_default = len([nd for nd, ks in _node_kinds(r5).items() if ks & set(SIBLING_OTHER_KINDS)])
    assert 0 < n_default < len(trade_nodes) + len(robber_nodes) + len(discard_nodes)


def test_sibling_arrays_ids_and_rate_zero():
    r = _game(0.3, seed=5)
    Xs, node, h, kind, Xm = sibling_arrays([r, r, _game(0.0, seed=6)], base_id=70)
    assert len(node) == 2 * len(r.s_node) and Xs.shape == (len(node), F.NUM_FEATURES) and Xm.shape == Xs.shape
    np.testing.assert_array_equal(Xm[:len(r.s_node)], r.Xm)
    assert node.dtype == np.int64 and node[0] == 70 and node[len(r.s_node)] == 70 + 1000
    assert len(np.unique(node)) == 2 * len(np.unique(r.s_node))
    np.testing.assert_array_equal(h[:len(r.s_h)], r.s_h)
    np.testing.assert_array_equal(kind[len(r.s_h):], r.s_kind)
    empty = sibling_arrays([_game(0.0, seed=7)])
    assert empty[0].shape == (0, F.NUM_FEATURES) and len(empty[1]) == 0 and empty[4].shape == (0, F.NUM_FEATURES)
