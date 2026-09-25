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
