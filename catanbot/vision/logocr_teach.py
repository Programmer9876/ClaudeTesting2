"""Teaching the log OCR on one labelled screenshot of a real screen (:func:`catanbot.vision.logocr.teach`).

What is learned (stored in ``profile.ocr``; arrays in a sidecar ``.npz`` next to the profile, or
inline as base64 when the profile has no file yet):

* the panel region (``profile.regions["log"]``);
* ``name_colours``: colour word -> the RGB its names are drawn in on this screen (measured on the
  names of the labelled entries), used by later reads' name classifier;
* ``glyph_adapt``: an adaptation of the glyph classifier's output layer to this screen's font.
  Every labelled entry's text runs are force-aligned to the words the truth line says they hold
  (the character-level dynamic programme of the decoder, restricted to the one word sequence);
  the aligned segments are character examples, the other candidate segments garbage examples;
  the output layer is refitted on them with an L2 pull towards the shipped weights (so letters
  absent from the screen keep their shipped behaviour).

The truth lines are canonical texts (names as colour words, see
:mod:`catanbot.vision.logocr`), oldest first; they are aligned to the panel's fully visible entries.
"""
from __future__ import annotations

import base64
import difflib
import io
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import colonist_log as L
from . import logocr_decode as D
from . import logocr_glyphs as G
from . import logocr_layout as LL

__all__ = ["drawn_tokens", "align_truth", "forced_alignment", "AdaptedClassifier", "fit_adaptation",
           "encode_arrays", "decode_arrays"]

_COLOUR_WORDS = ("red", "blue", "orange", "white", "green", "brown", "purple", "pink", "black", "yellow", "grey",
                 "gray", "teal", "cyan", "bronze", "silver", "gold", "mysticblue", "plum")


def drawn_tokens(text: str, icons: bool) -> List[Tuple[str, str]]:
    """How a canonical line is drawn: ``[(kind, text)]`` with kind ``name`` / ``word`` / ``icon``
    (the synthetic panel's tokenizer with colour words as the player names)."""
    from . import synth
    players = {c: c for c in _COLOUR_WORDS}
    out = []
    for t in synth.log_line_tokens(text, players, icons=icons):
        if t.kind == "name":
            out.append(("name", t.colour or t.text))
        elif t.kind == "icon":
            out.append(("icon", t.text))
        else:
            out.append(("word", t.text))
    # a multiplier number glued to its icon is read as a word before the icon
    return out


def _structure(entry: LL.Entry) -> List[Tuple[str, Any]]:
    toks = []
    for b in entry.bands:
        for t in b.items:
            if isinstance(t, LL.NameTok):
                toks.append(("name", t))
            elif isinstance(t, LL.Icon):
                toks.append(("icon", t))
            else:
                toks.append(("run", t))
    return toks


def match_structure(entry: LL.Entry, text: str) -> Optional[Tuple[List[Tuple[Any, List[str]]], List[Tuple[Any, str]]]]:
    """Map a labelled line onto an entry's tokens: ``([(run, words)], [(name token, colour)])`` or
    ``None`` when the drawn structure does not match (icons on / off are both tried)."""
    det = _structure(entry)
    for icons in (True, False):
        exp = drawn_tokens(text, icons)
        # collapse consecutive words into runs; icons into one symbol per icon
        seq: List[Tuple[str, Any]] = []
        for kind, tx in exp:
            if kind == "word":
                if seq and seq[-1][0] == "run":
                    seq[-1][1].append(tx)
                else:
                    seq.append(("run", [tx]))
            else:
                seq.append((kind, tx))
        # a glued ":" after a name is its own run on screen
        if [k for k, _ in seq] != [k for k, _ in det]:
            continue
        runs, names = [], []
        for (k, v), (_, t) in zip(seq, det):
            if k == "run":
                runs.append((t, list(v)))
            elif k == "name":
                names.append((t, v))
        return runs, names
    return None


def align_truth(texts: Sequence[str], truth: Sequence[str]) -> List[Optional[int]]:
    """Order-preserving alignment of read texts to truth lines (index of the truth line per read
    line, or None)."""
    n, m = len(texts), len(truth)
    sim = [[difflib.SequenceMatcher(None, texts[i].lower(), truth[j].lower()).ratio() for j in range(m)]
           for i in range(n)]
    Dm = np.zeros((n + 1, m + 1))
    Dm[:, 0] = np.arange(n + 1)
    Dm[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Dm[i, j] = min(Dm[i - 1, j - 1] + (1.0 - sim[i - 1][j - 1]) * 1.5, Dm[i - 1, j] + 1.0, Dm[i, j - 1] + 1.0)
    out: List[Optional[int]] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if abs(Dm[i, j] - (Dm[i - 1, j - 1] + (1.0 - sim[i - 1][j - 1]) * 1.5)) < 1e-9:
            if sim[i - 1][j - 1] >= 0.5:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif abs(Dm[i, j] - (Dm[i - 1, j] + 1.0)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return out


def forced_alignment(lp: np.ndarray, pairs: Sequence[Tuple[int, int]], bounds: Sequence[int],
                     gaps: Sequence[float], xh: float, words: Sequence[str]) -> Optional[List[Tuple[int, int]]]:
    """The best segmentation of a run as exactly ``words``: ``[(segment index, class)]`` per
    character, or ``None`` when no path exists."""
    chars: List[Tuple[int, bool]] = []
    for w in words:
        for k, ch in enumerate(w):
            if ch not in G.CHAR_INDEX:
                return None
            chars.append((G.CHAR_INDEX[ch], k == 0))
    nb = len(bounds)
    if not chars or nb < 2:
        return None
    spl = [D._space_logp(g, xh) for g in gaps]
    by_end: Dict[int, List[Tuple[int, int]]] = {}
    for n, (i, j) in enumerate(pairs):
        by_end.setdefault(j, []).append((n, i))
    NEG = -1e9
    dp = np.full((len(chars) + 1, nb), NEG)
    bp = np.full((len(chars) + 1, nb, 2), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for k, (c, wstart) in enumerate(chars, start=1):
        for j in range(1, nb):
            best, arg = NEG, (-1, -1)
            for n, i in by_end.get(j, ()):
                if dp[k - 1, i] <= NEG / 2:
                    continue
                if i == 0:
                    t = 0.0 if k == 1 else NEG
                else:
                    t = spl[i][0] if wstart else spl[i][1]
                v = dp[k - 1, i] + float(lp[n, c]) + t
                if v > best:
                    best, arg = v, (n, i)
            dp[k, j] = best
            bp[k, j] = arg
    if dp[len(chars), nb - 1] <= NEG / 2:
        return None
    out = []
    j = nb - 1
    for k in range(len(chars), 0, -1):
        n, i = bp[k, j]
        out.append((int(n), chars[k - 1][0]))
        j = int(i)
    out.reverse()
    return out


class AdaptedClassifier:
    """The shipped glyph classifier with its output layer replaced by a taught one."""

    def __init__(self, base: G.GlyphClassifier, W_last: np.ndarray, b_last: np.ndarray) -> None:
        self.base = base
        self.W_last = W_last.astype(np.float32)
        self.b_last = b_last.astype(np.float32)

    def hidden(self, X: np.ndarray) -> np.ndarray:
        W0, b0 = self.base._folded()
        h = np.asarray(X, dtype=np.float32) @ W0 + b0
        np.maximum(h, 0.0, out=h)
        for i in range(1, len(self.base.weights) - 1):
            h = h @ self.base.weights[i] + self.base.biases[i]
            np.maximum(h, 0.0, out=h)
        return h

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0, G.NUM_CLASSES), dtype=np.float32)
        z = (self.hidden(X) @ self.W_last + self.b_last).astype(np.float64)
        z = z - z.max(axis=1, keepdims=True)
        return (z - np.log(np.exp(z).sum(axis=1, keepdims=True))).astype(np.float32)


def fit_adaptation(base: G.GlyphClassifier, X: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.01,
                   pull: float = 1e-3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Refit the output layer on taught examples (``X`` in classifier units, ``y`` classes) with an
    L2 pull towards the shipped weights; ``(W, b, stats)``."""
    ad = AdaptedClassifier(base, base.weights[-1], base.biases[-1])
    H = ad.hidden(X).astype(np.float64)
    W0 = base.weights[-1].astype(np.float64)
    b0 = base.biases[-1].astype(np.float64)
    W, b = W0.copy(), b0.copy()
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    n = H.shape[0]
    rng = np.random.default_rng(seed)
    # balance characters and garbage
    wts = np.ones(n)
    g = y == G.GARBAGE
    if g.any() and (~g).any():
        wts[g] = (~g).sum() / max(1, g.sum())
    wts = wts / wts.mean()

    def acc(Wc: np.ndarray, bc: np.ndarray) -> float:
        return float(((H @ Wc + bc).argmax(axis=1) == y).mean())

    before = acc(W0, b0)
    for t in range(1, steps + 1):
        idx = rng.choice(n, size=min(n, 256), replace=False)
        z = H[idx] @ W + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(idx.size), y[idx]] -= 1.0
        p *= wts[idx, None] / idx.size
        gW = H[idx].T @ p + pull * (W - W0)
        gb = p.sum(axis=0) + pull * (b - b0)
        for par, gr, m_, v_ in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m_ *= 0.9
            m_ += 0.1 * gr
            v_ *= 0.999
            v_ += 0.001 * gr * gr
            par -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
    return W.astype(np.float32), b.astype(np.float32), {"train_acc_before": before, "train_acc_after": acc(W, b)}


def encode_arrays(**arrays: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_arrays(text: str) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(base64.b64decode(text.encode("ascii")))) as z:
        return {k: z[k] for k in z.files}
