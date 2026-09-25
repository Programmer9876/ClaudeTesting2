"""Tests for the number-token classifier (catanbot.vision.digits)."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from catanbot import board as B
from catanbot.vision import digits as D

MODEL_PATH = D.DEFAULT_MODEL_PATH
_HAVE_MODEL = os.path.isfile(MODEL_PATH)
needs_model = pytest.mark.skipif(not _HAVE_MODEL, reason="models/digits.npz missing (run scripts/train_digits.py)")

# A font subset different from the full training pool (regular weights only).
_TEST_FONTS = [f for f in D.available_fonts() if "Bold" not in os.path.basename(f)]


def _fresh_tokens(n: int, seed: int, fonts=None):
    rng = np.random.default_rng(seed)
    labels = [D.CLASSES[i % D.NUM_CLASSES] for i in range(n)]
    rng.shuffle(labels)
    fonts = fonts or _TEST_FONTS or D.available_fonts()
    imgs = [D.make_token_image(v, rng, fonts=fonts) for v in labels]
    return imgs, labels


# ---------------------------------------------------------------------------
# Synthetic renderer / pipeline
# ---------------------------------------------------------------------------
def test_classes_order():
    assert D.CLASSES == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    assert all(v in B.PIPS for v in D.CLASSES)
    assert D.NUM_FEATURES == 32 * 32 + 640


def test_available_fonts_exist():
    fonts = D.available_fonts()
    assert fonts, "expected at least one TrueType font on this machine"
    assert all(os.path.isfile(f) for f in fonts)


def test_make_token_image_shapes_and_determinism():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = D.make_token_image(11, rng1)
    b = D.make_token_image(11, rng2)
    assert a.mode == "RGB" and a.size == b.size
    assert np.array_equal(np.asarray(a), np.asarray(b))
    img = D.make_token_image(6, np.random.default_rng(1), radius=12)
    assert 20 <= img.size[0] <= 45
    img = D.make_token_image(12, np.random.default_rng(2), radius=40)
    assert 70 <= img.size[0] <= 140
    with pytest.raises(ValueError):
        D.make_token_image(7, np.random.default_rng(0))


def test_preprocess_and_features():
    img = D.make_token_image(8, np.random.default_rng(3))
    x = D.preprocess(img)
    assert x.shape == (32, 32) and x.dtype == np.float32
    assert abs(float(x.mean())) < 1e-3 and abs(float(x.std()) - 1.0) < 0.05
    X = D.extract_features(D.preprocess_batch([img, img]))
    assert X.shape == (2, D.NUM_FEATURES)
    assert np.isfinite(X).all()
    assert np.array_equal(X[0], X[1])
    # HOG on a flat image is all zeros, on an edge concentrates in one bin
    assert not D.hog_features(np.zeros((1, 32, 32), np.float32)).any()
    edge = np.zeros((32, 32), np.float32)
    edge[:, 16:] = 1.0
    h = D.hog_features(edge)[0, :512].reshape(8, 8, 8)
    assert h[:, 3:5, 0].sum() > 0.9 * h.sum()   # horizontal gradient -> bin 0


def test_locate_token_finds_disc_and_falls_back():
    rng = np.random.default_rng(5)
    img = D.make_token_image(4, rng, radius=30, background=(40, 110, 50))
    w, h = img.size
    x0, y0, x1, y1 = D.locate_token(np.asarray(img))
    assert 48 <= (x1 - x0) <= 72 and 48 <= (y1 - y0) <= 72
    flat = np.full((40, 40, 3), 120, np.uint8)
    assert D.locate_token(flat) == (0, 0, 40, 40)


def test_red_score_separates_ink_colours():
    rng = np.random.default_rng(11)
    fonts = _TEST_FONTS or None
    reds = [D.red_score(D.make_token_image(int(rng.choice([6, 8])), rng, fonts=fonts, red_ink=True))
            for _ in range(40)]
    blacks = [D.red_score(D.make_token_image(int(rng.choice(D.CLASSES)), rng, fonts=fonts, red_ink=False))
              for _ in range(40)]
    assert min(reds) > 0.45
    assert max(blacks) < 0.3
    assert D.red_score(np.zeros((30, 30), np.uint8)) == 0.5        # grayscale: undecidable


def test_red_prior_direction():
    clf = D.DigitClassifier()
    hi, lo = clf.red_prior(0.9), clf.red_prior(0.05)
    i6, i8, i5 = D.CLASS_INDEX[6], D.CLASS_INDEX[8], D.CLASS_INDEX[5]
    assert hi[i6] > hi[i5] and hi[i8] > hi[i5]
    assert lo[i6] < lo[i5] and lo[i8] < lo[i5]


def test_fit_small_and_roundtrip(tmp_path):
    X, y, _ = D.generate_dataset(400, 3)
    clf = D.DigitClassifier(hidden=(32, 16), seed=1)
    hist = clf.fit(X, y, epochs=8, batch_size=64, X_val=X[:100], y_val=y[:100], dropout=0.0, input_dropout=0.0)
    assert hist["loss"][-1] < hist["loss"][0]
    assert clf.accuracy(X, y) > 0.8
    p = tmp_path / "small.npz"
    clf.save(str(p))
    re = D.DigitClassifier.load(str(p))
    assert re.hidden == (32, 16)
    assert np.array_equal(re.predict_proba_features(X[:50]), clf.predict_proba_features(X[:50]))


def test_assign_standard_multiset():
    rng = np.random.default_rng(0)
    truth = list(B.STANDARD_NUMBERS)
    rng.shuffle(truth)
    P = np.full((18, 10), 0.01)
    for i, v in enumerate(truth):
        P[i, D.CLASS_INDEX[v]] = 0.9
    # corrupt one token: make the second "2" look like a 3 (there is only one 2)
    twos = [i for i, v in enumerate(truth) if v == 3]
    P[twos[0], D.CLASS_INDEX[3]] = 0.3
    P[twos[0], D.CLASS_INDEX[2]] = 0.6
    P /= P.sum(axis=1, keepdims=True)
    out = D.assign_standard_multiset(P)
    assert sorted(out) == sorted(B.STANDARD_NUMBERS)
    assert out == truth


# ---------------------------------------------------------------------------
# Trained model
# ---------------------------------------------------------------------------
@needs_model
def test_saved_model_accuracy_on_fresh_tokens():
    clf = D.DigitClassifier.load(MODEL_PATH)
    imgs, labels = _fresh_tokens(300, seed=20240917)
    pred = clf.predict(imgs)
    acc = float(np.mean([p == t for p, t in zip(pred, labels)]))
    assert acc >= 0.95, f"accuracy {acc:.3f} < 0.95"
    pred_np = clf.predict(imgs, use_red_prior=False)
    acc_np = float(np.mean([p == t for p, t in zip(pred_np, labels)]))
    assert acc_np >= 0.95, f"accuracy without red prior {acc_np:.3f} < 0.95"


@needs_model
def test_saved_model_load_save_roundtrip_exact(tmp_path):
    clf = D.DigitClassifier.load(MODEL_PATH)
    imgs, _ = _fresh_tokens(20, seed=99)
    p1 = clf.predict_proba(imgs)
    out = tmp_path / "copy.npz"
    clf.save(str(out))
    re = D.DigitClassifier.load(str(out))
    p2 = re.predict_proba(imgs)
    assert p1.shape == (20, 10)
    assert np.array_equal(p1, p2)
    assert np.allclose(p1.sum(axis=1), 1.0, atol=1e-4)


@needs_model
def test_numpy_and_pil_inputs_agree():
    clf = D.DigitClassifier.load(MODEL_PATH)
    imgs, _ = _fresh_tokens(12, seed=5)
    arrays = [np.asarray(im) for im in imgs]
    assert arrays[0].ndim == 3 and arrays[0].shape[2] == 3
    p_pil = clf.predict_proba(imgs)
    p_np = clf.predict_proba(arrays)
    assert np.array_equal(p_pil, p_np)
    # RGBA numpy and float [0,1] inputs are accepted too
    rgba = [np.dstack([a, np.full(a.shape[:2], 255, np.uint8)]) for a in arrays]
    assert np.array_equal(clf.predict_proba(rgba), p_np)
    floats = [a.astype(np.float32) / 255.0 for a in arrays]
    assert np.allclose(clf.predict_proba(floats), p_np, atol=1e-3)
    assert clf.predict_proba([]).shape == (0, 10)


@needs_model
def test_red_prior_changes_probabilities_sensibly():
    clf = D.DigitClassifier.load(MODEL_PATH)
    rng = np.random.default_rng(42)
    # a black-ink 8 (forced by drawing until the ink is black) should have 6/8 boosted less
    img = D.make_token_image(9, rng, fonts=_TEST_FONTS or None)
    p_prior = clf.predict_proba([img])[0]
    p_plain = clf.predict_proba([img], use_red_prior=False)[0]
    i6, i8 = D.CLASS_INDEX[6], D.CLASS_INDEX[8]
    assert p_prior[i6] + p_prior[i8] <= p_plain[i6] + p_plain[i8] + 1e-6


# ---------------------------------------------------------------------------
# Regression tests for the robustness review (vision-io-9, vision-io-10)
# ---------------------------------------------------------------------------
def test_save_and_load_agree_on_the_npz_suffix(tmp_path):
    clf = D.DigitClassifier(hidden=(8,), seed=0)
    written = clf.save(str(tmp_path / "digits_noext"))
    assert written.endswith(".npz") and os.path.isfile(written)
    re = D.DigitClassifier.load(str(tmp_path / "digits_noext"))
    assert re.hidden == (8,)
    assert D.DigitClassifier.load(written).hidden == (8,)


def test_find_model_path_env_override_and_clear_error(tmp_path, monkeypatch):
    clf = D.DigitClassifier(hidden=(8,), seed=0)
    path = clf.save(str(tmp_path / "custom.npz"))
    monkeypatch.setenv(D.MODEL_PATH_ENV, path)
    assert D.find_model_path() == path
    assert D.DigitClassifier.load().hidden == (8,)
    monkeypatch.setenv(D.MODEL_PATH_ENV, str(tmp_path / "missing.npz"))
    with pytest.raises(FileNotFoundError, match="missing.npz"):
        D.find_model_path()
    monkeypatch.delenv(D.MODEL_PATH_ENV)
    monkeypatch.setattr(D, "DEFAULT_MODEL_PATH", str(tmp_path / "nope.npz"))
    monkeypatch.setattr(D, "_PACKAGE_MODEL_PATH", str(tmp_path / "nope2.npz"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="train_digits.py"):
        D.find_model_path()


def test_docstring_feature_counts_are_current():
    assert "640" in D.__doc__ and "1664" in D.__doc__ and "512" not in D.__doc__.split("Colonist draws")[0]
    assert D.NUM_HOG == 640 and D.NUM_FEATURES == 1664
