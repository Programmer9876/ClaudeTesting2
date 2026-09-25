"""Tests for catanbot.model."""
from __future__ import annotations

import time

import numpy as np
import pytest

from catanbot import features as F
from catanbot.model import ValueNet, binary_auc
from catanbot.state import new_game, PHASE_MAIN


def xor_data(n: int, seed: int = 0, noise: float = 0.1):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, 2))
    X = bits + rng.normal(0.0, noise, size=(n, 2))
    y = (bits[:, 0] ^ bits[:, 1]).astype(np.float32)
    return X.astype(np.float32), y


def test_learns_xor():
    X, y = xor_data(600, seed=1)
    Xv, yv = xor_data(300, seed=2)
    net = ValueNet(n_in=2, hidden=(16, 16), seed=0)
    t0 = time.perf_counter()
    hist = net.fit(X, y, epochs=150, batch_size=32, lr=0.01, weight_decay=1e-5, X_val=Xv, y_val=yv,
                   patience=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0
    p = net.predict(Xv)
    assert p.shape == (300,) and p.dtype == np.float32
    assert p.min() >= 0.0 and p.max() <= 1.0
    acc = float(((p >= 0.5) == (yv >= 0.5)).mean())
    assert acc > 0.95, (acc, hist["train_loss"][-1])
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert hist["val_auc"][-1] > 0.95
    # __call__ alias
    np.testing.assert_array_equal(net(Xv), p)


def test_fit_history_keys_and_early_stopping():
    X, y = xor_data(200, seed=3)
    Xv, yv = xor_data(100, seed=4)
    net = ValueNet(n_in=2, hidden=(8,), seed=1)
    logs = []
    hist = net.fit(X, y, epochs=5, batch_size=16, lr=0.01, weight_decay=0.0, X_val=Xv, y_val=yv,
                   log=logs.append, patience=2)
    for k in ("train_loss", "val_loss", "val_auc", "val_acc", "epochs", "best_epoch", "stopped_early"):
        assert k in hist
    assert len(hist["train_loss"]) == hist["epochs"] <= 5
    assert len(hist["val_loss"]) == len(hist["val_auc"]) == len(hist["val_acc"]) == hist["epochs"]
    assert len(logs) == hist["epochs"]
    assert net.norm_fitted
    # without a validation set the val lists stay empty
    net2 = ValueNet(n_in=2, hidden=(8,), seed=1)
    h2 = net2.fit(X, y, epochs=2, batch_size=16, lr=0.01, weight_decay=0.0)
    assert h2["val_loss"] == [] and h2["epochs"] == 2
    # early stopping actually triggers with a huge learning rate on a fixed val set
    net3 = ValueNet(n_in=2, hidden=(8,), seed=1)
    h3 = net3.fit(X, y, epochs=60, batch_size=200, lr=5.0, weight_decay=0.0, X_val=Xv, y_val=yv, patience=3)
    assert h3["epochs"] < 60 and h3["stopped_early"]
    # float16 input (replay buffer dtype) is accepted
    net4 = ValueNet(n_in=2, hidden=(8,), seed=1)
    net4.fit(X.astype(np.float16), y, epochs=1, batch_size=16, lr=0.01, weight_decay=0.0)
    assert net4.predict(X.astype(np.float16)).shape == (200,)


def test_gradient_check_float64():
    rng = np.random.default_rng(5)
    net = ValueNet(n_in=4, hidden=(5, 3), seed=2, dtype=np.float64)
    X = rng.normal(size=(7, 4))
    y = rng.integers(0, 2, size=7).astype(np.float64)
    net.fit_normalisation(rng.normal(size=(50, 4)) * 2 + 1)
    wd = 0.01
    loss, gW, gb = net.loss_and_grads(X, y, weight_decay=wd)
    analytic = np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in zip(gW, gb)])
    theta = net.get_params()
    numeric = np.empty_like(theta)
    h = 1e-6
    for i in range(theta.size):
        tp = theta.copy()
        tp[i] += h
        net.set_params(tp)
        lp, _, _ = net.loss_and_grads(X, y, weight_decay=wd)
        tm = theta.copy()
        tm[i] -= h
        net.set_params(tm)
        lm, _, _ = net.loss_and_grads(X, y, weight_decay=wd)
        numeric[i] = (lp - lm) / (2 * h)
    net.set_params(theta)
    rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))
    assert rel < 1e-4, rel
    assert np.isfinite(loss)


def test_save_load_round_trip(tmp_path):
    X, y = xor_data(300, seed=6)
    net = ValueNet(n_in=2, hidden=(12, 6), seed=3)
    net.fit(X, y, epochs=20, batch_size=32, lr=0.01, weight_decay=1e-4)
    path = str(tmp_path / "net.npz")
    net.save(path)
    net2 = ValueNet.load(path)
    assert net2.n_in == 2 and net2.hidden == (12, 6) and net2.norm_fitted
    np.testing.assert_array_equal(net.predict(X), net2.predict(X))
    np.testing.assert_array_equal(net.mean, net2.mean)
    # continuing training after load keeps the stored normalisation
    m = net2.mean.copy()
    net2.fit(X * 5 + 3, y, epochs=1, batch_size=32, lr=0.001, weight_decay=0.0)
    np.testing.assert_array_equal(net2.mean, m)


def test_default_net_and_evaluate():
    net = ValueNet()
    assert net.n_in == F.NUM_FEATURES and net.hidden == (256, 128)
    assert net.W[0].shape == (F.NUM_FEATURES, 256) and net.W[-1].shape == (128, 1)
    assert net.W[0].dtype == np.float32
    s = new_game(4)
    s.phase = PHASE_MAIN
    v = net.evaluate([s, s, s], [0, 1, 2])
    assert v.shape == (3,) and v.min() >= 0.0 and v.max() <= 1.0
    assert net.evaluate([], []).shape == (0,)
    # untrained net accepts raw features (standardisation is identity) and stays finite
    X = F.extract_batch([s] * 4, [0, 1, 2, 3])
    assert np.all(np.isfinite(net.predict(X)))
    with pytest.raises(ValueError):
        net.predict(np.zeros((2, F.NUM_FEATURES + 1), np.float32))
    # chunked prediction equals single-shot
    Xr = np.random.default_rng(0).normal(size=(50, F.NUM_FEATURES)).astype(np.float32)
    np.testing.assert_allclose(net.predict(Xr, chunk=7), net.predict(Xr), rtol=1e-6, atol=1e-6)


def test_binary_auc():
    assert binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert binary_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert binary_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert binary_auc(np.array([1, 1]), np.array([0.3, 0.7])) == 0.5
    assert binary_auc(np.array([0, 1, 1, 0]), np.array([0.2, 0.9, 0.6, 0.6])) == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# input mask (hand-blind net) and exact terminal values
# ---------------------------------------------------------------------------
from catanbot.model import HAND_BLIND_FEATURES, MODEL_VERSION, feature_mask  # noqa: E402
from catanbot.state import PHASE_GAME_OVER  # noqa: E402


def test_feature_mask_patterns():
    m = feature_mask(HAND_BLIND_FEATURES)
    assert m.shape == (F.NUM_FEATURES,) and m.dtype == np.float32
    hidden = [n for n, k in zip(F.FEATURE_NAMES, m) if k == 0]
    assert hidden == ["me_res_wood", "me_res_brick", "me_res_sheep", "me_res_wheat", "me_res_ore", "me_hand_size",
                      "me_cards_over_7", "me_discard_exposure", "me_can_build_road", "me_can_build_settlement",
                      "me_can_build_city", "me_can_buy_dev"]
    # opponents' hands and everything else stay visible
    assert m[F.feature_index("opp1_hand_size")] == 1 and m[F.feature_index("me_public_vp")] == 1
    # comma-separated string == iterable; no patterns == no mask
    np.testing.assert_array_equal(feature_mask(",".join(HAND_BLIND_FEATURES)), m)
    assert feature_mask("") is None and feature_mask(None) is None and feature_mask([]) is None
    with pytest.raises(ValueError):
        feature_mask("me_res_*,no_such_feature_*")
    # custom name list
    np.testing.assert_array_equal(feature_mask("a?", names=["a1", "a2", "b"]), [0.0, 0.0, 1.0])


def test_masked_net_ignores_hidden_features(tmp_path):
    rng = np.random.default_rng(7)
    n_in = 6
    mask = np.array([1, 0, 1, 1, 0, 1], np.float32)
    # the label depends on features 1 and 4 only: a masked net cannot learn it, an unmasked one can
    X = rng.normal(size=(800, n_in)).astype(np.float32)
    y = ((X[:, 1] + X[:, 4]) > 0).astype(np.float32)
    plain = ValueNet(n_in=n_in, hidden=(16,), seed=0)
    blind = ValueNet(n_in=n_in, hidden=(16,), seed=0, input_mask=mask)
    assert blind.masked_features(["f0", "f1", "f2", "f3", "f4", "f5"]) == ["f1", "f4"]
    plain.fit(X, y, epochs=30, batch_size=64, lr=0.01, weight_decay=0.0)
    blind.fit(X, y, epochs=30, batch_size=64, lr=0.01, weight_decay=0.0, input_noise=0.3)
    acc_plain = float(((plain.predict(X) >= 0.5) == (y >= 0.5)).mean())
    acc_blind = float(((blind.predict(X) >= 0.5) == (y >= 0.5)).mean())
    assert acc_plain > 0.9 and acc_blind < 0.65, (acc_plain, acc_blind)
    # predictions are invariant to the hidden columns, exactly
    X2 = X.copy()
    X2[:, [1, 4]] = rng.normal(size=(800, 2)) * 50
    np.testing.assert_array_equal(blind.predict(X2), blind.predict(X))
    # the mask survives save / load, files without a mask keep the old version number
    path = str(tmp_path / "blind.npz")
    blind.save(path)
    with np.load(path) as z:
        assert int(z["version"]) == MODEL_VERSION == 2 and "input_mask" in z
    loaded = ValueNet.load(path)
    np.testing.assert_array_equal(loaded.input_mask, mask)
    np.testing.assert_array_equal(loaded.predict(X2), blind.predict(X))
    assert "masked=2" in repr(loaded)
    plain.save(str(tmp_path / "plain.npz"))
    with np.load(str(tmp_path / "plain.npz")) as z:
        assert int(z["version"]) == 1 and "input_mask" not in z
    assert ValueNet.load(str(tmp_path / "plain.npz")).input_mask is None
    # an all-ones mask is no mask; a wrong shape is rejected; the mask can be changed after loading
    plain.input_mask = np.ones(n_in, np.float32)
    assert plain.input_mask is None
    with pytest.raises(ValueError):
        plain.input_mask = np.ones(n_in + 1, np.float32)
    loaded.input_mask = None
    assert not np.array_equal(loaded.predict(X2), loaded.predict(X))
    # gradients respect the mask: no gradient flows into the hidden input columns
    _, gW, _ = blind.loss_and_grads(X, y, weight_decay=0.0)
    assert np.all(gW[0][[1, 4], :] == 0.0) and np.any(gW[0][[0, 2, 3, 5], :] != 0.0)


def test_evaluate_exact_on_finished_games():
    net = ValueNet(hidden=(8,), seed=3)
    s = new_game(4)
    s.phase = PHASE_MAIN
    v_live = net.evaluate([s, s], [0, 1])
    assert 0.0 < v_live[0] < 1.0
    over = s.copy()
    over.phase = PHASE_GAME_OVER
    over.winner = 2
    v = net.evaluate([over, over, over, s], [2, 0, 3, 1])
    assert v[0] == 1.0 and v[1] == 0.0 and v[2] == 0.0 and v[3] == v_live[1]
    assert v.dtype == np.float32


def _synthetic_buffer(tmp_path, n_games=30, per_game=20, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_games * per_game, F.NUM_FEATURES)).astype(np.float16)
    g = np.repeat(np.arange(n_games, dtype=np.int32), per_game)
    y = (X[:, F.feature_index("me_public_vp")].astype(np.float32) > 0).astype(np.float32)
    buf = str(tmp_path / "buf.npz")
    np.savez(buf, X=X, y=y, g=g, bias=np.zeros(len(y), np.float32))
    return buf, X, y, g


def test_train_fit_only_and_mask_flags(tmp_path):
    """``python -m catanbot.train --fit-only --replay BUF --out NET`` fits on an existing buffer;
    ``--mask-features`` stores the mask with the net (off by default)."""
    from catanbot import train as T
    buf, X, y, g = _synthetic_buffer(tmp_path)
    out = str(tmp_path / "net.npz")
    common = ["--fit-only", "--replay", buf, "--epochs", "2", "--hidden", "8", "--batch-size", "64", "--seed", "0",
              "--rank-weight", "0"]
    assert T.main(common + ["--out", out]) == 0
    net = ValueNet.load(out)
    assert net.hidden == (8,) and net.input_mask is None
    assert (tmp_path / "net_train.log").exists()
    assert not (tmp_path / "net_replay.npz").exists()   # fit-only never writes a buffer
    out2 = str(tmp_path / "blind.npz")
    assert T.main(common + ["--mask-features", ",".join(HAND_BLIND_FEATURES), "--out", out2]) == 0
    assert ValueNet.load(out2).masked_features() == [n for n, m in zip(F.FEATURE_NAMES, feature_mask(HAND_BLIND_FEATURES)) if m == 0]
    out3 = str(tmp_path / "custom.npz")
    assert T.main(common + ["--mask-features", "g_turn,opp*_hand_size", "--out", out3]) == 0
    assert ValueNet.load(out3).masked_features() == ["opp1_hand_size", "opp2_hand_size", "opp3_hand_size", "g_turn"]
    # the same split / recipe through fit_replay directly
    args = T.build_parser().parse_args(common + ["--out", out])
    net2, hist, n_tr, n_va = T.fit_replay(X, y, g, args, seed=0)
    assert n_tr + n_va == len(y) and 0 < n_va < len(y) and hist["epochs"] == 2 and hist["n_pairs"] == 0
    np.testing.assert_array_equal(net2.predict(X[:50]), net.predict(X[:50]))


def test_build_pairs():
    from catanbot.train import build_pairs
    # node 0: END_TURN 0.30, road 0.31 (too close), settlement 0.45, bad bank trade 0.20
    # node 1: single row (no pairs); node 2: END_TURN 0.5 vs proposals 0.5 (no gap -> no pairs)
    node = np.array([0, 0, 0, 0, 1, 2, 2])
    h = np.array([0.30, 0.31, 0.45, 0.20, 0.9, 0.5, 0.5], np.float32)
    kind = np.array([0, 1, 2, 5, 0, 0, 6], np.int8)
    pos, neg = build_pairs(node, h, kind, gap=0.02, per_node=10, seed=0)
    pairs = set(zip(pos.tolist(), neg.tolist()))
    assert (2, 0) in pairs                       # top (settlement) beats END_TURN: always first
    assert (pos[0], neg[0]) == (2, 0)
    assert (0, 3) in pairs and (2, 3) in pairs and (2, 1) in pairs and (1, 3) in pairs
    assert (1, 0) not in pairs and (0, 1) not in pairs   # gap 0.01 < 0.02
    assert all(h[i] - h[j] >= 0.02 for i, j in pairs) and all(node[i] == node[j] for i, j in pairs)
    assert not any(node[i] in (1, 2) for i in pos)
    # per_node caps the count, keeping the top-vs-END_TURN pair
    pos2, neg2 = build_pairs(node, h, kind, gap=0.02, per_node=2, seed=0)
    assert len(pos2) == 2 and (pos2[0], neg2[0]) == (2, 0)
    # rows of a node need not be contiguous
    perm = np.array([3, 0, 5, 2, 6, 1, 4])
    pos3, neg3 = build_pairs(node[perm], h[perm], kind[perm], gap=0.02, per_node=10, seed=0)
    assert set(zip(node[perm][pos3].tolist(), node[perm][neg3].tolist())) == {(0, 0)}
    assert set((int(perm[i]), int(perm[j])) for i, j in zip(pos3, neg3)) == pairs


def test_build_pairs_other_phase_gap():
    """Offer / robber / discard nodes (all rows of ``SIBLING_OTHER_KINDS``) use the smaller ``gap_other``:
    accept vs reject differ by a card or two, far below the main-phase gap."""
    from catanbot.selfplay import SIBLING_OTHER_KINDS, sibling_kind_id
    from catanbot.train import build_pairs
    from catanbot import actions as A
    acc, rej = sibling_kind_id((A.ACCEPT_TRADE,)), sibling_kind_id((A.REJECT_TRADE,))
    rob, setup = sibling_kind_id((A.MOVE_ROBBER, 0, -1)), sibling_kind_id((A.SETUP_SETTLEMENT, 0))
    assert acc in SIBLING_OTHER_KINDS and rej in SIBLING_OTHER_KINDS and rob in SIBLING_OTHER_KINDS
    assert setup not in SIBLING_OTHER_KINDS
    # node 0: an offer, reject 0.300 vs accept 0.302; node 1: two robber moves 0.30 / 0.32;
    # node 2: a setup node with the same tiny gap (main gap applies: no pair); node 3: a main-phase node
    # whose END_TURN row and a road differ by 0.005 (no pair) and a settlement by 0.05 (pair)
    node = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    h = np.array([0.300, 0.302, 0.30, 0.32, 0.300, 0.302, 0.30, 0.305, 0.35], np.float32)
    kind = np.array([rej, acc, rob, rob, setup, setup, 0, 1, 2], np.int8)
    pos, neg = build_pairs(node, h, kind, gap=0.01, per_node=10, seed=0, gap_other=0.001,
                           other_kinds=SIBLING_OTHER_KINDS)
    pairs = set(zip(pos.tolist(), neg.tolist()))
    assert pairs == {(1, 0), (3, 2), (8, 6), (8, 7)}
    # without gap_other the offer node forms no pair (0.002 < 0.01)
    pos0, neg0 = build_pairs(node, h, kind, gap=0.01, per_node=10, seed=0)
    assert set(zip(pos0.tolist(), neg0.tolist())) == {(3, 2), (8, 6), (8, 7)}
    # a node mixing other and main kinds keeps the main gap
    kind2 = kind.copy()
    kind2[0] = 1
    pos2, neg2 = build_pairs(node, h, kind2, gap=0.01, per_node=10, seed=0, gap_other=0.001,
                             other_kinds=SIBLING_OTHER_KINDS)
    assert (1, 0) not in set(zip(pos2.tolist(), neg2.tolist()))


def test_pair_rank_gradient_and_fit():
    from catanbot.model import pair_rank_loss
    rng = np.random.default_rng(9)
    net = ValueNet(n_in=3, hidden=(4,), seed=2, dtype=np.float64)
    net.fit_normalisation(rng.normal(size=(40, 3)))
    X = rng.normal(size=(6, 3))
    y = rng.integers(0, 2, size=6).astype(np.float64)
    P = (rng.normal(size=(5, 3)), rng.normal(size=(5, 3)), np.array([0.3, 0.1, 0.5, 0.2, 0.4]))
    C = (rng.normal(size=(4, 3)), rng.normal(size=(4, 3)))
    kw = dict(weight_decay=0.01, pairs=P, pair_weight=0.7, consistency=C, consistency_weight=0.4)
    loss, gW, gb = net.loss_and_grads(X, y, **kw)
    analytic = np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in zip(gW, gb)])
    theta = net.get_params()
    numeric = np.empty_like(theta)
    h = 1e-6
    for i in range(theta.size):
        tp = theta.copy(); tp[i] += h; net.set_params(tp)
        lp = net.loss_and_grads(X, y, **kw)[0]
        tm = theta.copy(); tm[i] -= h; net.set_params(tm)
        lm = net.loss_and_grads(X, y, **kw)[0]
        numeric[i] = (lp - lm) / (2 * h)
    net.set_params(theta)
    rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))
    assert rel < 1e-6, rel
    l0, g0 = pair_rank_loss(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.5)
    assert l0 == pytest.approx((np.log1p(np.exp(-1.5)) + np.log1p(np.exp(0.5))) / 2) and g0.shape == (2,) and (g0 < 0).all()
    # ranking is learned without hurting the outcome fit: y depends on f0, the pairs order f1
    N = 3000
    X = rng.normal(size=(N, 4)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=N) > 0).astype(np.float32)
    Xp = rng.normal(size=(1500, 4)).astype(np.float32)
    Xn = Xp.copy()
    Xn[:, 1] -= 1.0
    ranked = ValueNet(n_in=4, hidden=(16,), seed=0)
    hist = ranked.fit(X, y, epochs=12, batch_size=64, lr=0.01, weight_decay=0.0, X_val=X[:400], y_val=y[:400],
                      pairs=(Xp[200:], Xn[200:]), val_pairs=(Xp[:200], Xn[:200]), pair_weight=1.0,
                      pair_margin=0.5, patience=0, input_noise=0.1)
    plain = ValueNet(n_in=4, hidden=(16,), seed=0)
    hist0 = plain.fit(X, y, epochs=12, batch_size=64, lr=0.01, weight_decay=0.0, X_val=X[:400], y_val=y[:400],
                      input_noise=0.1)
    for k in ("pair_loss", "val_pair_loss", "val_pair_acc"):
        assert k in hist and len(hist[k]) == hist["epochs"]
    assert k not in hist0
    d_ranked = ranked.logits(Xp[:200]) - ranked.logits(Xn[:200])
    d_plain = plain.logits(Xp[:200]) - plain.logits(Xn[:200])
    assert (d_ranked > 0).mean() > 0.9 and hist["val_pair_acc"][-1] > 0.9
    assert abs((d_plain > 0).mean() - 0.5) < 0.2          # BCE alone knows nothing about f1
    # horizon consistency: rows that differ only in f2 are pulled to the same logit
    Ca = rng.normal(size=(1500, 4)).astype(np.float32)
    Cb = Ca.copy()
    Cb[:, 2] += 2.0
    cons = ValueNet(n_in=4, hidden=(16,), seed=0)
    hc = cons.fit(X, y, epochs=12, batch_size=64, lr=0.01, weight_decay=0.0, consistency=(Ca[200:], Cb[200:]),
                  val_consistency=(Ca[:200], Cb[:200]), consistency_weight=1.0, pair_batch=64)
    assert len(hc["cons_loss"]) == len(hc["val_cons_loss"]) == hc["epochs"] and "pair_loss" not in hc
    gap_c = np.abs(cons.logits(Ca[:200]) - cons.logits(Cb[:200])).mean()
    gap_p = np.abs(plain.logits(Ca[:200]) - plain.logits(Cb[:200])).mean()
    assert gap_c < 0.25 * gap_p + 0.05, (gap_c, gap_p)
    # the pairs force a dependence on f1 that the labels do not have, so the outcome fit pays a little
    # (in the real data the pairs constrain states the labels are silent about); it must not be destroyed
    assert hist["val_auc"][-1] > hist0["val_auc"][-1] - 0.06


def test_fit_replay_with_siblings(tmp_path):
    """fit_replay builds train / validation pairs from sibling arrays and reports their counts."""
    from catanbot import train as T
    buf, X, y, g = _synthetic_buffer(tmp_path, n_games=20, per_game=10)
    rng = np.random.default_rng(3)
    n_nodes, per_node = 40, 5
    Xs = rng.normal(size=(n_nodes * per_node, F.NUM_FEATURES)).astype(np.float16)
    sn = np.repeat(np.arange(n_nodes, dtype=np.int64) * 7, per_node)
    sh = rng.uniform(0.1, 0.6, size=len(sn)).astype(np.float32)
    sk = np.tile(np.array([0, 1, 2, 4, 5], np.int8), n_nodes)
    Xm = (Xs.astype(np.float32) + rng.normal(scale=0.1, size=Xs.shape)).astype(np.float16)
    args = T.build_parser().parse_args(["--fit-only", "--replay", buf, "--out", str(tmp_path / "n.npz"),
                                        "--epochs", "2", "--hidden", "8", "--batch-size", "32", "--rank-batch", "16"])
    net, hist, _, _ = T.fit_replay(X, y, g, args, seed=0, siblings=(Xs, sn, sh, sk, Xm))
    assert hist["n_pairs"] > 0 and hist["n_val_pairs"] > 0 and len(hist["pair_loss"]) == hist["epochs"]
    assert len(hist["val_pair_acc"]) == hist["epochs"]
    assert hist["n_cons"] > 0 and len(hist["cons_loss"]) == len(hist["val_cons_loss"]) == hist["epochs"]
    assert net.predict(X[:5]).shape == (5,)
    # without the mid-turn twins (older sibling data) only the ranking term is used
    _, hist4, _, _ = T.fit_replay(X, y, g, args, seed=0, siblings=(Xs, sn, sh, sk))
    assert hist4["n_cons"] == 0 and "cons_loss" not in hist4 and hist4["n_pairs"] == hist["n_pairs"]
    # sibling npz given on the command line is used by --fit-only (no game generation)
    sib = str(tmp_path / "sib.npz")
    np.savez(sib, Xs=Xs, sn=sn, sh=sh, sk=sk, Xm=Xm)
    out = str(tmp_path / "ranked.npz")
    assert T.main(["--fit-only", "--replay", buf, "--siblings", sib, "--out", out, "--epochs", "1", "--hidden", "8",
                   "--batch-size", "32", "--rank-batch", "16"]) == 0
    assert ValueNet.load(out).hidden == (8,)
    log = (tmp_path / "ranked_train.log").read_text()
    assert "ranking pairs:" in log and "val_pair_acc" in log and "horizon-consistency pairs" in log
