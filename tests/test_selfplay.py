"""Tests for catanbot.selfplay: sibling-afterstate recording for the value net's ranking term."""
from __future__ import annotations

import random

import numpy as np

from catanbot import actions as A
from catanbot import features as F
from catanbot.selfplay import (SIBLING_KINDS, SIBLING_KIND_OTHER, GameResult, make_bot, play_game, sibling_arrays,
                               sibling_kind_id)


def _game(sibling_rate: float, seed: int = 3) -> GameResult:
    bots = [make_bot("heuristic:temp=0.3,eps=0.03") for _ in range(4)]
    return play_game(bots, rng=random.Random(seed), record=True, sample_every=2, seed=seed, sibling_rate=sibling_rate)


def test_sibling_recording():
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
    # every sibling is the state after the action *and* END_TURN: the next player's roll phase, so no
    # "still my turn" feature can separate END_TURN from the other actions
    assert np.all(X[:, F.feature_index("g_my_turn")] == 0.0)
    assert np.all(X[:, F.feature_index("g_phase_roll")] + X[:, F.feature_index("g_phase_game_over")] == 1.0)
    # the mid-turn twin of every sibling is still my main phase (the decision state itself for END_TURN)
    Xm = r.Xm.astype(np.float32)
    assert np.all(Xm[:, F.feature_index("g_my_turn")] == 1.0) and np.all(Xm[:, F.feature_index("g_phase_main")] == 1.0)
    end_id = sibling_kind_id((A.END_TURN,))
    assert end_id == 0
    for node in nodes:
        rows = r.s_node == node
        assert rows.sum() >= 2
        assert int((r.s_kind[rows] == end_id).sum()) == 1   # exactly one END_TURN sibling per node
        # the END_TURN sibling's mid-turn twin is the decision state: same hand as its END_TURN afterstate
        e = np.flatnonzero(rows & (r.s_kind == end_id))[0]
        assert Xm[e, F.feature_index("me_hand_size")] == X[e, F.feature_index("me_hand_size")]
    kinds = set(r.s_kind.tolist())
    assert sibling_kind_id((A.BUILD_ROAD, 1)) in kinds and sibling_kind_id((A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0))) in kinds
    assert sibling_kind_id((A.ROLL,)) == SIBLING_KIND_OTHER == len(SIBLING_KINDS)
    # building spends cards (roads can be free after Road Building): a settlement / city / dev-card sibling
    # always holds fewer cards than the END_TURN sibling of the same node
    hand = F.feature_index("me_hand_size")
    paid = {sibling_kind_id((A.BUILD_SETTLEMENT, 0)), sibling_kind_id((A.BUILD_CITY, 0)), sibling_kind_id((A.BUY_DEV,))}
    checked = 0
    for node in nodes:
        rows = np.flatnonzero(r.s_node == node)
        end_row = rows[r.s_kind[rows] == end_id][0]
        for i in rows:
            if r.s_kind[i] in paid:
                assert X[i, hand] < X[end_row, hand]
                checked += 1
    assert checked > 0


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
