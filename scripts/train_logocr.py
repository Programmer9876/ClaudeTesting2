#!/usr/bin/env python3
"""Train the log OCR's glyph classifier (``models/logocr_glyphs.npz``).

Usage::

    python3 scripts/train_logocr.py                    # default: 1400 panels, 10 epochs (~8 min on 4 cores)
    python3 scripts/train_logocr.py --panels 300 --epochs 3 --out /tmp/glyphs.npz   # smoke run

Training data is synthetic and deterministic (``--seed``): panels of rows of log words, random words
and numbers rendered with the allowed fonts (:data:`catanbot.vision.logocr_glyphs.TRAIN_FONTS`;
never the benchmark's held-out Liberation Sans / FreeSans), degraded like a screen capture (blur,
noise, JPEG, zoom) and read by the live pipeline (layout analysis, cutting); each candidate segment
is labelled from the renderer's character positions (see :func:`generate_dataset`).  Garbage
segments are subsampled to ``--garbage`` of the data.  The model is a numpy MLP trained with Adam,
cosine learning-rate decay and dropout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from catanbot.vision import logocr_glyphs as G  # noqa: E402


def build(panels: int, seed: int, garbage: float, log, max_samples: int = 0) -> tuple:
    X, y = G.generate_dataset(panels, seed=seed, log=log)
    rng = np.random.default_rng(seed + 17)
    g = np.flatnonzero(y == G.GARBAGE)
    c = np.flatnonzero(y != G.GARBAGE)
    if max_samples and c.size > (1.0 - garbage) * max_samples:
        c = rng.choice(c, size=int((1.0 - garbage) * max_samples), replace=False)
    keep_g = int(min(g.size, garbage / max(1e-6, 1.0 - garbage) * c.size))
    idx = np.concatenate([c, rng.choice(g, size=keep_g, replace=False)]) if keep_g else c
    idx = np.sort(idx)
    return X[idx], y[idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panels", type=int, default=1400, help="training panels to render (default 1400)")
    ap.add_argument("--val-panels", type=int, default=200, help="validation panels (default 200)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", default="320,160", help="hidden layer widths (default 320,160)")
    ap.add_argument("--garbage", type=float, default=0.4, help="share of garbage segments kept (default 0.4)")
    ap.add_argument("--max-samples", type=int, default=450000, help="cap on training segments (default 450000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=G.DEFAULT_MODEL_PATH, help="model file (default models/logocr_glyphs.npz)")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731
    np.random.seed(args.seed)
    Xtr, ytr = build(args.panels, args.seed, args.garbage, log, args.max_samples)
    Xva, yva = build(args.val_panels, args.seed + 100003, args.garbage, None, 40000)
    log(f"train {Xtr.shape[0]} segments ({(ytr == G.GARBAGE).mean():.2f} garbage), val {Xva.shape[0]}")
    clf = G.GlyphClassifier(hidden=[int(h) for h in args.hidden.split(",") if h], seed=args.seed)
    clf.fit(Xtr, ytr, epochs=args.epochs, seed=args.seed, X_val=Xva, y_val=yva, log=log)
    path = clf.save(args.out)
    size = os.path.getsize(path) / 1e6
    log(f"saved {path} ({size:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
