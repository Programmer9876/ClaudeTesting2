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


def test_train_fit_only_applies_mask(tmp_path):
    """``python -m catanbot.train --fit-only --replay BUF --out NET`` fits on an existing buffer and
    stores the hand-blind mask with the net; ``--no-mask`` gives a plain net."""
    from catanbot import train as T
    rng = np.random.default_rng(11)
    n_games, per_game = 30, 20
    X = rng.normal(size=(n_games * per_game, F.NUM_FEATURES)).astype(np.float16)
    g = np.repeat(np.arange(n_games, dtype=np.int32), per_game)
    y = (X[:, F.feature_index("me_public_vp")].astype(np.float32) > 0).astype(np.float32)
    buf = str(tmp_path / "buf.npz")
    np.savez(buf, X=X, y=y, g=g, bias=np.zeros(len(y), np.float32))
    out = str(tmp_path / "net.npz")
    common = ["--fit-only", "--replay", buf, "--epochs", "2", "--hidden", "8", "--batch-size", "64", "--seed", "0"]
    assert T.main(common + ["--out", out]) == 0
    net = ValueNet.load(out)
    assert net.hidden == (8,) and net.masked_features() == [n for n, m in zip(F.FEATURE_NAMES, feature_mask(HAND_BLIND_FEATURES)) if m == 0]
    assert (tmp_path / "net_train.log").exists()
    assert not (tmp_path / "net_replay.npz").exists()   # fit-only never writes a buffer
    out2 = str(tmp_path / "plain.npz")
    assert T.main(common + ["--no-mask", "--out", out2]) == 0
    assert ValueNet.load(out2).input_mask is None
    out3 = str(tmp_path / "custom.npz")
    assert T.main(common + ["--mask-features", "g_turn,opp*_hand_size", "--out", out3]) == 0
    assert ValueNet.load(out3).masked_features() == ["opp1_hand_size", "opp2_hand_size", "opp3_hand_size", "g_turn"]
    # the same split / recipe through fit_replay directly
    args = T.build_parser().parse_args(common + ["--out", out])
    net2, hist, n_tr, n_va = T.fit_replay(X, y, g, args, seed=0)
    assert n_tr + n_va == len(y) and 0 < n_va < len(y) and hist["epochs"] == 2
    np.testing.assert_array_equal(net2.predict(X[:50]), net.predict(X[:50]))
