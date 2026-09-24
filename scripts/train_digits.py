#!/usr/bin/env python3
"""Train the number-token classifier on synthetic Colonist-style tokens.

Usage::

    python3 scripts/train_digits.py [--n 30000] [--epochs 30] [--workers 4]
                                    [--seed 0] [--out models/digits.npz]

Generates ``n`` synthetic tokens (balanced classes, every font found on the
machine), holds out 10 %, trains :class:`catanbot.vision.digits.DigitClassifier`
with Adam, prints held-out accuracy plus the confusion matrix and saves the
model.  Exits with status 1 when the held-out accuracy is below ``--min-acc``
(default 0.98).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanbot.vision import digits as D  # noqa: E402


def _gen_chunk(args: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    n, seed = args
    X, y, _ = D.generate_dataset(n, np.random.default_rng(seed))
    return X, y


def generate(n: int, seed: int, workers: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` samples using ``workers`` processes (deterministic per seed)."""
    n_chunks = max(1, min(64, n // 500))
    sizes = [n // n_chunks + (1 if i < n % n_chunks else 0) for i in range(n_chunks)]
    jobs = [(sz, seed * 100003 + i) for i, sz in enumerate(sizes)]
    if workers > 1:
        with mp.Pool(workers) as pool:
            parts = pool.map(_gen_chunk, jobs)
    else:
        parts = [_gen_chunk(j) for j in jobs]
    X = np.concatenate([p[0] for p in parts], axis=0)
    y = np.concatenate([p[1] for p in parts], axis=0)
    return X, y


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((D.NUM_CLASSES, D.NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def format_confusion(cm: np.ndarray) -> str:
    labels = [str(c) for c in D.CLASSES]
    w = max(5, max(len(str(int(cm.max()))), 2) + 2)
    lines = ["true\\pred".ljust(10) + "".join(l.rjust(w) for l in labels)]
    for i, l in enumerate(labels):
        lines.append(l.ljust(10) + "".join(str(int(v)).rjust(w) for v in cm[i]))
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30000, help="number of synthetic samples")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-acc", type=float, default=0.98)
    ap.add_argument("--out", default=D.DEFAULT_MODEL_PATH)
    args = ap.parse_args(argv)

    fonts = D.available_fonts()
    print(f"fonts ({len(fonts)}): " + ", ".join(os.path.basename(f) for f in fonts))
    t0 = time.time()
    X, y = generate(args.n, args.seed, args.workers)
    print(f"generated {X.shape[0]} samples x {X.shape[1]} features in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(X.shape[0])
    n_val = int(round(args.val_frac * X.shape[0]))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr, X_val, y_val = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    clf = D.DigitClassifier(hidden=(256, 128), seed=args.seed)
    t1 = time.time()
    clf.fit(X_tr, y_tr, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, X_val=X_val, y_val=y_val, seed=args.seed, log=print)
    print(f"trained in {time.time() - t1:.1f}s")

    pred = clf.predict_proba_features(X_val).argmax(axis=1)
    acc = float((pred == y_val).mean())
    cm = confusion(y_val, pred)
    print(f"held-out accuracy: {acc:.4f} ({int((pred == y_val).sum())}/{len(y_val)})")
    print(format_confusion(cm))

    clf.save(args.out)
    print(f"saved {args.out}")
    # sanity: the saved model reproduces the predictions exactly
    re = D.DigitClassifier.load(args.out)
    assert np.array_equal(re.predict_proba_features(X_val[:64]), clf.predict_proba_features(X_val[:64]))
    print(f"total time {time.time() - t0:.1f}s")
    if acc < args.min_acc:
        print(f"FAILED: accuracy {acc:.4f} < {args.min_acc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
