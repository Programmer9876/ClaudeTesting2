"""Glyph classifier of the local log OCR: over-segmented text runs -> character posteriors.

A *text run* (consecutive text-coloured glyph pieces of one row, see
:mod:`catanbot.vision.logocr_layout`) is turned into a soft ink image (the pixels' projection on the
panel's text colour, 0 background .. 1 full ink), cut at every plausible glyph boundary (gaps and
column-ink minima: :func:`cut_candidates`) and every span of up to :data:`MAX_SPAN` consecutive
pieces is classified by a numpy MLP (:class:`GlyphClassifier`) into one of :data:`CHARS` or
*garbage* (half a glyph, two glyphs, a glyph with a neighbour).  The decoder
(:mod:`catanbot.vision.logocr_decode`) searches the segmentations with a lexicon.

Patch (:func:`segment_features`)
    the run is resampled so the x-height is :data:`XH_PX` pixels; the patch is the segment's columns
    over a fixed vertical window around the baseline (``-2.1 .. +0.8`` x-heights: ascenders,
    capitals and descenders keep their place, so ``o`` / ``O`` or ``p`` / ``P`` differ by position
    and size), centred in :data:`PATCH_W` columns (squashed when wider), plus scalar features
    (width, the gaps to the neighbouring ink, the ink the cut goes through).

Training data (:func:`generate_dataset`, ``scripts/train_logocr.py``) is synthetic: panels of rows
of words (the log vocabulary, random words, numbers) rendered with the allowed fonts at 8-34 px on
random light / dark colours, blurred / noised / JPEG'd / zoomed, then read by the same layout
analysis and cutting as the live reader; each candidate segment is labelled from the renderer's
character positions.  The held-out benchmark fonts (Liberation Sans, FreeSans) are never used.
"""
from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

__all__ = [
    "CHARS",
    "GARBAGE",
    "NUM_CLASSES",
    "PATCH_H",
    "PATCH_W",
    "NUM_FEATURES",
    "MAX_SPAN",
    "MODEL_PATH_ENV",
    "DEFAULT_MODEL_PATH",
    "TRAIN_FONTS",
    "HELD_OUT_FONTS",
    "find_model_path",
    "RunInk",
    "run_ink",
    "cut_candidates",
    "segment_features",
    "GlyphClassifier",
    "generate_dataset",
    "load_classifier",
]

#: Character classes (the garbage class is appended: index :data:`GARBAGE`).
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:,-.'!?()/_#&+"
GARBAGE = len(CHARS)
NUM_CLASSES = len(CHARS) + 1
CHAR_INDEX = {c: i for i, c in enumerate(CHARS)}
#: Patch geometry: x-height in patch pixels and the window around the baseline (x-heights).
XH_PX = 8.0
WIN_UP, WIN_DN = 2.1, 0.8
PATCH_H = int(round(XH_PX * (WIN_UP + WIN_DN)))          # 23
PATCH_W = 20
N_EXTRA = 8
NUM_FEATURES = PATCH_H * PATCH_W + N_EXTRA
#: Longest segment in elementary pieces (between consecutive cut candidates).
MAX_SPAN = 4
#: Widest segment (x-heights): a capital W / M in a wide font.
MAX_SEG_XH = 2.3
#: At or below this x-height (pixels) every ink column is a cut candidate; above it, only the
#: intervals between natural cuts wider than ``DENSE_WIDE`` x-heights get every column.
DENSE_XH_ALL = 5.5
DENSE_WIDE = 0.95
DENSE_XH = 8.0

MODEL_PATH_ENV = "CATANBOT_LOGOCR_MODEL"
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "logocr_glyphs.npz")
_PACKAGE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "logocr_glyphs.npz")

#: The benchmark's held-out fonts: never used for training, templates or thresholds.
HELD_OUT_FONTS = ("LiberationSans-", "FreeSans")
#: Training fonts (path, weight); only the existing ones are used.  Sans faces dominate (the log is
#: a sans face), serif / mono faces add shape variety.
TRAIN_FONTS: List[Tuple[str, float]] = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 3.0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0.7),
    ("/opt/ruby-3.3.6/lib/ruby/3.3.0/rdoc/generator/template/darkfish/fonts/Lato-Regular.ttf", 3.0),
    ("/opt/ruby-3.3.6/lib/ruby/3.3.0/rdoc/generator/template/darkfish/fonts/Lato-Light.ttf", 1.0),
    ("/usr/share/fonts/opentype/tlwg/Loma.otf", 2.5),
    ("/usr/share/fonts/opentype/tlwg/Loma-Bold.otf", 0.7),
    ("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf", 2.5),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 2.0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 0.8),
    ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 0.8),
    ("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", 0.6),
    ("/usr/share/fonts/X11/Type1/c0648bt_.pfb", 0.6),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0.6),
    ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 0.5),
    ("/opt/ruby-3.3.6/lib/ruby/3.3.0/rdoc/generator/template/darkfish/fonts/SourceCodePro-Regular.ttf", 0.5),
    ("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", 0.4),
]


def _with_npz(path: str) -> str:
    return path if path.lower().endswith(".npz") else path + ".npz"


def find_model_path(path: Optional[str] = None) -> str:
    """The glyph model file: ``path`` or ``$CATANBOT_LOGOCR_MODEL`` when given, else the first
    existing of the repository's ``models/logocr_glyphs.npz``, the package-data copy and
    ``./models/logocr_glyphs.npz``.  ``FileNotFoundError`` with a hint otherwise."""
    explicit = path or os.environ.get(MODEL_PATH_ENV)
    if explicit:
        for cand in (explicit, _with_npz(explicit)):
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"log OCR glyph model not found at {explicit!r}")
    cands = [DEFAULT_MODEL_PATH, _PACKAGE_MODEL_PATH, os.path.join(os.getcwd(), "models", "logocr_glyphs.npz")]
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("log OCR glyph model not found (looked in " + ", ".join(cands) + "); train it with "
                            f"scripts/train_logocr.py or point {MODEL_PATH_ENV} at a logocr_glyphs.npz file")


def _is_held_out(path: str) -> bool:
    base = os.path.basename(path)
    return any(base.startswith(p) for p in HELD_OUT_FONTS)


def available_train_fonts() -> List[Tuple[str, float]]:
    out = []
    for p, w in TRAIN_FONTS:
        if os.path.isfile(p) and not _is_held_out(p):
            out.append((p, w))
    return out


# ---------------------------------------------------------------------------
# run ink and cuts
# ---------------------------------------------------------------------------
@dataclass
class RunInk:
    """Soft ink of one text run: ``A`` (rows ``y0..``, columns ``x0..``, panel-inner coordinates),
    the baseline row (in ``A`` coordinates) and the x-height (pixels)."""

    A: np.ndarray
    x0: int
    y0: int
    baseline: float
    xh: float
    plab: Optional[np.ndarray] = None                    # piece index per pixel (-1 none)
    pieces: Optional[List[Tuple[int, int]]] = None       # piece column extents (A coordinates)


def run_ink(lay: Any, band: Any, run: Any, pad: int = 2) -> RunInk:
    """The soft ink of ``run`` (a :class:`~catanbot.vision.logocr_layout.TextRun`) on ``band``:
    the projection of each pixel's difference to its row background on the panel's text colour
    direction, divided by the text contrast (so full ink is ~1 whatever the colours), clipped to
    0..1; ink of other tokens (names, icons) inside the columns is removed."""
    ya, yb = band._ya, band._yb
    x0 = max(0, run.x0 - pad)
    x1 = min(lay.rgb.shape[1], run.x1 + pad)
    rgb = lay.rgb[ya:yb, x0:x1].astype(np.float32)
    bg = lay.bg_rows[ya:yb].astype(np.float32)[:, None, :]
    uT = np.asarray(getattr(lay, "_uT", np.ones(3) / math.sqrt(3.0)), dtype=np.float32)
    norm = float(getattr(lay, "_text_norm", 150.0))
    A = ((rgb - bg) @ uT) / max(20.0, norm)
    A = np.clip(A, 0.0, 1.0)
    # remove other tokens' ink (a name glued to a colon, an icon's edge)
    lab = band._lab[:, x0:x1]
    own = set(k for p in run.pieces for k in p.comp_ids)
    other = (lab > 0) & ~np.isin(lab, list(own)) if own else lab > 0
    if other.any():
        g = other.copy()
        g[:, 1:] |= other[:, :-1]
        g[:, :-1] |= other[:, 1:]
        A = np.where(g & ~np.isin(lab, list(own)), 0.0, A)
    plab, pieces = _piece_labels(lab, [p.comp_ids for p in run.pieces], A)
    return RunInk(A=A.astype(np.float32), x0=x0, y0=ya, baseline=float(band.baseline - ya), xh=float(lay.xh),
                  plab=plab, pieces=pieces)


def _piece_labels(lab: np.ndarray, piece_comps: Sequence[Sequence[int]], A: np.ndarray
                  ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Piece index of every inked pixel of a run (components' pixels, then their anti-aliased
    fringe by two steps of dilation) and each piece's column extent."""
    plab = np.full(lab.shape, -1, dtype=np.int32)
    for k, comps in enumerate(piece_comps):
        if comps:
            plab[np.isin(lab, list(comps))] = k
    fringe = A > 0.04
    for _ in range(2):
        un = (plab < 0) & fringe
        if not un.any():
            break
        nb = plab.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.full_like(plab, -1)
            ys = slice(max(0, dy), plab.shape[0] + min(0, dy))
            yd = slice(max(0, -dy), plab.shape[0] + min(0, -dy))
            xs = slice(max(0, dx), plab.shape[1] + min(0, dx))
            xd = slice(max(0, -dx), plab.shape[1] + min(0, -dx))
            sh[yd, xd] = plab[ys, xs]
            nb = np.where((nb < 0) & (sh >= 0), sh, nb)
        plab = np.where(un, nb, plab)
    pieces = []
    for k in range(len(piece_comps)):
        cols = np.flatnonzero((plab == k).any(axis=0))
        pieces.append((int(cols[0]), int(cols[-1]) + 1) if cols.size else (0, 0))
    return plab, pieces


def cut_candidates(A: np.ndarray, xh: float) -> Tuple[List[int], List[float]]:
    """Candidate glyph boundaries of a run's soft ink ``A``: ``(bounds, gaps)``.

    ``bounds`` are column positions (a boundary at ``x`` separates column ``x - 1`` from ``x``),
    sorted, first / last at the ink's ends: the middle of every gap in the strong ink and the
    column-ink minima inside touching glyphs; an interval still wider than a narrow glyph pair
    (touching glyphs without a clear minimum - small or blurred text) gets every column as a
    candidate (all columns for very small text).  ``gaps[k]`` is the empty width (pixels) at bound
    ``k`` (0 for a cut through ink)."""
    strong = A > 0.45
    occ = strong.any(axis=0)
    cols = np.flatnonzero(occ)
    if cols.size == 0:
        return [], []
    lo, hi = int(cols[0]), int(cols[-1]) + 1
    prof = A.sum(axis=0).astype(np.float64)
    sm = prof.copy()
    if prof.size >= 3:
        sm[1:-1] = 0.25 * prof[:-2] + 0.5 * prof[1:-1] + 0.25 * prof[2:]
    bounds: Dict[int, float] = {lo: 99.0, hi: 99.0}
    d = np.diff(np.concatenate(([1], occ[lo:hi].astype(np.int8), [1])))
    gs = np.flatnonzero(d == -1)
    ge = np.flatnonzero(d == 1)
    for g0, g1 in zip(gs.tolist(), ge.tolist()):
        if g1 > g0:
            x0, x1 = lo + g0, lo + g1
            bounds[(x0 + x1) // 2 if x1 - x0 > 1 else x0] = float(x1 - x0)
    # minima inside ink runs
    win = max(2, int(round(0.5 * xh)))
    for x in range(lo + 1, hi - 1):
        if not occ[x]:
            continue
        v = sm[x]
        if v > sm[x - 1] or v > sm[x + 1]:
            continue
        lpk = sm[max(lo, x - win):x].max()
        rpk = sm[x + 1:min(hi, x + 1 + win)].max()
        if v < 0.9 * min(lpk, rpk):
            bounds.setdefault(x, 0.0)
            e = x                      # a flat minimum: its other end as well
            while e + 1 < hi - 1 and sm[e + 1] == v:
                e += 1
            if e > x + 1:
                bounds.setdefault(e, 0.0)
    # dense candidates where the natural cuts leave wide intervals
    tau = 0.0 if xh <= DENSE_XH_ALL else max(4.0, DENSE_WIDE * xh)
    bs = sorted(bounds)
    for a, b in zip(bs[:-1], bs[1:]):
        if b - a > tau and occ[a:b].any():
            for x in range(a + 1, b):
                if occ[x] or occ[x - 1]:
                    bounds.setdefault(x, 0.0)
    bs = sorted(bounds)
    return bs, [bounds[b] for b in bs]


def _strip(A: np.ndarray, baseline: float, xh: float) -> Tuple[np.ndarray, float]:
    """The run resampled to patch scale: rows ``baseline - WIN_UP xh .. baseline + WIN_DN xh`` ->
    :data:`PATCH_H` rows, columns scaled by the same factor.  Returns ``(strip, scale)``."""
    s = XH_PX / max(1.0, xh)
    top = baseline - WIN_UP * xh
    h, w = A.shape
    out_w = max(1, int(round(w * s)))
    if cv2 is not None:
        M = np.array([[s, 0.0, 0.0], [0.0, s, -top * s]], dtype=np.float64)
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        strip = cv2.warpAffine(A, M, (out_w, PATCH_H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        if s < 0.9:      # warpAffine does not area-average: pre-blur a little on downscale
            pass
    else:  # nearest-neighbour fallback
        ys = np.clip(np.floor(top + (np.arange(PATCH_H) + 0.5) / s).astype(int), -1, h)
        xs = np.clip(np.floor((np.arange(out_w) + 0.5) / s).astype(int), 0, w - 1)
        pad = np.zeros((h + 2, w), dtype=A.dtype)
        pad[1:-1] = A
        strip = pad[ys + 1][:, xs]
    return strip.astype(np.float32), s


def segment_features(ri: RunInk, bounds: Sequence[int], gaps: Sequence[float], max_span: Optional[int] = None,
                     info: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Features of every candidate segment ``bounds[i]..bounds[j]`` (no wider than
    :data:`MAX_SEG_XH` x-heights; at most ``max_span`` pieces when given): ``(X float32
    (n, NUM_FEATURES), [(i, j)])``.  The patch values are 0..1, the extras about 0..2."""
    A, xh = ri.A, ri.xh
    nb = len(bounds)
    if nb < 2:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), []
    strip, s = _strip(A, ri.baseline, xh)
    w = A.shape[1]
    prof = A.sum(axis=0)
    pmax = float(prof.max()) if prof.size and prof.max() > 0 else 1.0
    occ = (A > 0.45).any(axis=0)
    # first occupied column >= x and last occupied column < x
    idx = np.arange(w + 1)
    nxt = np.full(w + 1, w, dtype=np.int64)
    oc = np.flatnonzero(occ)
    if oc.size == 0:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), []
    pos = np.searchsorted(oc, idx)
    nxt = np.where(pos < oc.size, oc[np.minimum(pos, oc.size - 1)], w)
    prv = np.where(pos > 0, oc[np.maximum(pos - 1, 0)], -1)
    B = np.asarray(bounds, dtype=np.int64)
    G_ = np.asarray(gaps, dtype=np.float64)
    maxw = MAX_SEG_XH * xh + 1.0
    I, J = [], []
    for i in range(nb - 1):
        lim = nb if max_span is None else min(nb, i + 1 + max_span)
        for j in range(i + 1, lim):
            if B[j] - B[i] > maxw and j > i + 1:
                break
            I.append(i)
            J.append(j)
    if not I:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), []
    I = np.asarray(I)
    J = np.asarray(J)
    x0, x1 = B[I], B[J]
    ix0 = nxt[x0]
    ix1 = prv[x1] + 1
    ok = ix1 > ix0
    I, J, x0, x1, ix0, ix1 = I[ok], J[ok], x0[ok], x1[ok], ix0[ok], ix1[ok]
    n = I.size
    sw = strip.shape[1]
    sx0 = np.floor(ix0 * s).astype(np.int64)
    sx1 = np.minimum(sw, np.maximum(sx0 + 1, np.ceil(ix1 * s).astype(np.int64)))
    pw = sx1 - sx0
    o = (PATCH_W - pw) // 2
    c = np.arange(PATCH_W)[None, :]
    colidx = sx0[:, None] + c - o[:, None]
    valid = (c >= o[:, None]) & (c < (o + pw)[:, None])
    colidx = np.where(valid, colidx, sw)                       # sw -> the zero column
    stripz = np.concatenate([strip, np.zeros((PATCH_H, 1), dtype=np.float32)], axis=1)
    patches = stripz[:, colidx]                                 # (PATCH_H, n, PATCH_W)
    patches = np.transpose(patches, (1, 0, 2)).reshape(n, PATCH_H * PATCH_W)
    squashed = pw > PATCH_W
    if squashed.any():
        for k in np.flatnonzero(squashed):
            sub = strip[:, sx0[k]:sx1[k]]
            if cv2 is not None:
                sub = cv2.resize(sub, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
            else:
                sub = sub[:, np.linspace(0, sub.shape[1] - 1, PATCH_W).astype(int)]
            patches[k] = sub.reshape(-1)
    lg = np.where(G_[I] < 50, G_[I], 5.0 * xh)
    rg = np.where(G_[J] < 50, G_[J], 5.0 * xh)
    lcut = np.where(G_[I] == 0, prof[np.clip(x0, 0, w - 1)] / pmax, 0.0)
    rcut = np.where(G_[J] == 0, prof[np.clip(x1 - 1, 0, w - 1)] / pmax, 0.0)
    extra = np.stack([(ix1 - ix0) / xh, np.minimum(lg / xh, 2.0), np.minimum(rg / xh, 2.0), lcut, rcut,
                      squashed.astype(np.float64), np.full(n, min(1.0, xh / 20.0)),
                      ((I == 0) | (J == nb - 1)).astype(np.float64)], axis=1).astype(np.float32)
    X = np.concatenate([patches.astype(np.float32), extra], axis=1)
    pairs = list(zip(I.tolist(), J.tolist()))
    # kerned pairs (Yo, Ye, Te, ff...): a single-glyph piece reaching past a cut is taken whole by
    # the segment its centre is in and removed from the neighbour's
    if ri.plab is not None and ri.pieces:
        single = [k for k, (a, b) in enumerate(ri.pieces) if b > a and (b - a) <= 1.3 * xh]
        # kerned pairs: pieces overlapping a neighbouring piece in x; only a cut inside the overlap
        # zone (the boundary between the two glyphs) needs the masked patch
        zones = []
        crossing = set()
        for k in single:
            ak, bk = ri.pieces[k]
            for q in range(len(ri.pieces)):
                if q == k:
                    continue
                aq, bq = ri.pieces[q]
                lo_, hi_ = max(ak, aq), min(bk, bq)
                if hi_ > lo_:
                    zones.append((lo_ - 1, hi_ + 1))
                    crossing.add(k)
        if crossing:
            crossing = sorted(crossing)
            for n in range(len(pairs)):
                c0, c1 = int(x0[n]), int(x1[n])
                if not any(z0 <= c0 <= z1 or z0 <= c1 <= z1 for z0, z1 in zones):
                    continue
                owned, excl = [], []
                for k in crossing:
                    a, b = ri.pieces[k]
                    cx = 0.5 * (a + b)
                    if c0 <= cx < c1:
                        if a < c0 or b > c1:
                            owned.append(k)
                    elif a < c1 and b > c0:
                        excl.append(k)
                if not owned and not excl:
                    continue
                v = _masked_features(ri, strip, s, c0, c1, owned, excl, xh)
                if v is not None:
                    X[n, :PATCH_H * PATCH_W] = v[0]
                    X[n, PATCH_H * PATCH_W] = v[1]
                    X[n, PATCH_H * PATCH_W + 5] = v[2]
                    if info is not None:
                        info.setdefault("masked", set()).add(n)
    return X, pairs


def _masked_features(ri: RunInk, strip: np.ndarray, s: float, c0: int, c1: int, owned: Sequence[int],
                     excl: Sequence[int], xh: float) -> Optional[Tuple[np.ndarray, float, float]]:
    """Patch of a segment whose kerned neighbour pieces are masked (``excl``) and whose own crossing
    pieces are taken whole (``owned``): ``(patch, width / xh, squashed)``."""
    A = ri.A
    plab = ri.plab
    xa = min([c0] + [ri.pieces[k][0] for k in owned])
    xb = max([c1] + [ri.pieces[k][1] for k in owned])
    sub = A[:, xa:xb].copy()
    pl = plab[:, xa:xb]
    cols = np.arange(xa, xb)[None, :]
    inside = (cols >= c0) & (cols < c1)
    keep = np.broadcast_to(inside, pl.shape).copy()
    for k in excl:
        keep &= pl != k
    for k in owned:
        keep |= pl == k
    sub = np.where(keep, sub, 0.0)
    occ = (sub > 0.45).any(axis=0)
    nz = np.flatnonzero(occ)
    if nz.size == 0:
        return None
    ia, ib = int(nz[0]), int(nz[-1]) + 1
    sub = sub[:, ia:ib]
    stripe, _ = _strip(sub, ri.baseline, xh)
    pw = stripe.shape[1]
    patch = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    squashed = 0.0
    if pw > PATCH_W:
        if cv2 is not None:
            patch[:] = cv2.resize(stripe, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
        else:
            patch[:] = stripe[:, np.linspace(0, pw - 1, PATCH_W).astype(int)]
        squashed = 1.0
    else:
        o = (PATCH_W - pw) // 2
        patch[:, o:o + pw] = stripe
    return patch.reshape(-1), (ib - ia) / xh, squashed


# ---------------------------------------------------------------------------
# the MLP
# ---------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class GlyphClassifier:
    """Numpy MLP over :func:`segment_features` (ReLU hidden layers, softmax over
    :data:`NUM_CLASSES`); trained with Adam, cosine learning-rate decay and dropout (the
    :class:`catanbot.vision.digits.DigitClassifier` recipe)."""

    def __init__(self, hidden: Sequence[int] = (384, 192), seed: int = 0) -> None:
        self.hidden = tuple(int(h) for h in hidden)
        rng = np.random.default_rng(seed)
        sizes = (NUM_FEATURES, *self.hidden, NUM_CLASSES)
        self.weights = [(rng.standard_normal((a, b)) * math.sqrt(2.0 / a)).astype(np.float32)
                        for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.feat_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        self.feat_std = np.ones(NUM_FEATURES, dtype=np.float32)
        self.history: Dict[str, List[float]] = {}

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feat_mean) / self.feat_std

    def _folded(self) -> Tuple[np.ndarray, np.ndarray]:
        """The first layer with the feature standardisation folded in (inference)."""
        f = getattr(self, "_fold", None)
        if f is None or f[2] is not self.weights[0]:
            W0 = self.weights[0] / self.feat_std[:, None]
            b0 = self.biases[0] - (self.feat_mean / self.feat_std) @ self.weights[0]
            f = (W0.astype(np.float32), b0.astype(np.float32), self.weights[0])
            self._fold = f
        return f[0], f[1]

    def logits(self, X: np.ndarray) -> np.ndarray:
        W0, b0 = self._folded()
        h = np.asarray(X, dtype=np.float32) @ W0 + b0
        n = len(self.weights)
        if n > 1:
            np.maximum(h, 0.0, out=h)
        for i in range(1, n):
            h = h @ self.weights[i] + self.biases[i]
            if i < n - 1:
                np.maximum(h, 0.0, out=h)
        return h

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """Log posteriors ``(N, NUM_CLASSES)``."""
        if X.shape[0] == 0:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        z = self.logits(X).astype(np.float64)
        z = z - z.max(axis=1, keepdims=True)
        return (z - np.log(np.exp(z).sum(axis=1, keepdims=True))).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 12, batch_size: int = 256, lr: float = 1.5e-3,
            weight_decay: float = 2e-5, dropout: float = 0.2, input_dropout: float = 0.05, seed: int = 0,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, log=None,
            class_weight: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Adam on softmax cross-entropy with cosine decay and dropout; ``X`` rows are features
        (uint8 or float), standardisation statistics are fitted on ``X``."""
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        Xf = X.astype(np.float32)
        self.feat_mean = Xf.mean(axis=0).astype(np.float32)
        self.feat_std = (Xf.std(axis=0) + 0.05).astype(np.float32)
        del Xf
        params = self.weights + self.biases
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        steps_total = max(1, epochs * math.ceil(n / batch_size))
        nl = len(self.weights)
        hist: Dict[str, List[float]] = {"loss": [], "train_acc": [], "val_acc": []}
        cw = None if class_weight is None else np.asarray(class_weight, dtype=np.float32)
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot, corr = 0.0, 0
            for st in range(0, n, batch_size):
                idx = perm[st:st + batch_size]
                xb = self._standardise(X[idx].astype(np.float32))
                yb = y[idx]
                bs = xb.shape[0]
                if input_dropout > 0:
                    xb = xb * ((rng.random(xb.shape, dtype=np.float32) >= input_dropout) / (1 - input_dropout))
                acts = [xb]
                masks: List[Optional[np.ndarray]] = [None]
                h = xb
                for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                    z = h @ W + b
                    if i < nl - 1:
                        h = np.maximum(z, 0.0)
                        if dropout > 0:
                            mk = (rng.random(h.shape, dtype=np.float32) >= dropout).astype(np.float32) / (1 - dropout)
                            h = h * mk
                            masks.append(mk)
                        else:
                            masks.append(None)
                    else:
                        h = z
                        masks.append(None)
                    acts.append(h)
                p = _softmax(h)
                wts = cw[yb] if cw is not None else np.ones(bs, dtype=np.float32)
                tot += float(-(np.log(p[np.arange(bs), yb] + 1e-12) * wts).sum())
                corr += int((p.argmax(axis=1) == yb).sum())
                g = p.copy()
                g[np.arange(bs), yb] -= 1.0
                g *= (wts / bs)[:, None]
                gw: List[np.ndarray] = [np.empty(0)] * nl
                gb: List[np.ndarray] = [np.empty(0)] * nl
                for i in range(nl - 1, -1, -1):
                    gw[i] = acts[i].T @ g + weight_decay * self.weights[i]
                    gb[i] = g.sum(axis=0)
                    if i > 0:
                        g = (g @ self.weights[i].T) * (acts[i] > 0)
                        if masks[i] is not None:
                            g = g * masks[i]
                step += 1
                cur = lr * 0.5 * (1.0 + math.cos(math.pi * step / steps_total))
                for k, (par, gr) in enumerate(zip(params, gw + gb)):
                    m[k] = b1 * m[k] + (1 - b1) * gr
                    v[k] = b2 * v[k] + (1 - b2) * (gr * gr)
                    par -= (cur * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)).astype(np.float32)
            hist["loss"].append(tot / n)
            hist["train_acc"].append(corr / n)
            if X_val is not None and y_val is not None:
                pv = np.concatenate([self.logits(X_val[k:k + 4096].astype(np.float32)).argmax(axis=1)
                                     for k in range(0, X_val.shape[0], 4096)])
                hist["val_acc"].append(float((pv == y_val).mean()))
            if log is not None:
                msg = f"epoch {ep + 1:3d}/{epochs} loss {hist['loss'][-1]:.4f} train_acc {hist['train_acc'][-1]:.4f}"
                if hist["val_acc"]:
                    msg += f" val_acc {hist['val_acc'][-1]:.4f}"
                log(msg)
        self.history = hist
        return hist

    def save(self, path: str) -> str:
        path = _with_npz(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload: Dict[str, np.ndarray] = {
            "chars": np.frombuffer(CHARS.encode("utf-8"), dtype=np.uint8),
            "hidden": np.asarray(self.hidden, dtype=np.int64),
            "num_features": np.asarray(NUM_FEATURES, dtype=np.int64),
            "patch": np.asarray([PATCH_H, PATCH_W, N_EXTRA], dtype=np.int64),
            "feat_mean": self.feat_mean.astype(np.float32),
            "feat_std": self.feat_std.astype(np.float32),
        }
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{i}"] = W.astype(np.float16)
            payload[f"b{i}"] = b.astype(np.float32)
        np.savez_compressed(path, **payload)
        return path

    @staticmethod
    def load(path: Optional[str] = None) -> "GlyphClassifier":
        path = find_model_path(path)
        with np.load(path) as z:
            chars = bytes(z["chars"].astype(np.uint8)).decode("utf-8")
            if chars != CHARS:
                raise ValueError("glyph model was trained with another character set")
            if int(z["num_features"]) != NUM_FEATURES:
                raise ValueError("glyph model was trained with another feature layout")
            hidden = tuple(int(h) for h in z["hidden"])
            clf = GlyphClassifier(hidden=hidden)
            clf.feat_mean = z["feat_mean"].astype(np.float32)
            clf.feat_std = z["feat_std"].astype(np.float32)
            clf.weights = [z[f"W{i}"].astype(np.float32) for i in range(len(hidden) + 1)]
            clf.biases = [z[f"b{i}"].astype(np.float32) for i in range(len(hidden) + 1)]
        return clf


_CLF_CACHE: Dict[str, GlyphClassifier] = {}


def load_classifier(path: Optional[str] = None) -> GlyphClassifier:
    """The glyph classifier (loaded once per process and path)."""
    p = find_model_path(path)
    clf = _CLF_CACHE.get(p)
    if clf is None:
        clf = GlyphClassifier.load(p)
        _CLF_CACHE[p] = clf
    return clf


# ---------------------------------------------------------------------------
# synthetic training data
# ---------------------------------------------------------------------------
#: Words of the game log (as drawn) - half the training text.
LOG_WORDS = ("got rolled stole from wants to give for traded with bank gave and took used Monopoly Year of Plenty "
             "Knight Road Building bought Development Card built a Settlement City placed received starting "
             "resources counter-offered to: moved Robber discarded cards card wood brick sheep wheat ore lumber "
             "grain wool You you desert Victory Point accepted rejected the trade turn ended their is selecting "
             "wood, brick, sheep, wheat, ore, lumber, grain, wool, an stole: gets won game Largest Army Longest").split()
_NAMEISH = "Alice Bob Carol Dave Kaz Mira_7 TheBaron Nyx Zed k8 Tobi Robbie settler_x Marguerite GrandPa Joe Q Ana Luz".split()


def _rand_word(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        w = rng.choice(LOG_WORDS)
    elif r < 0.62:
        w = str(rng.choice([rng.randint(0, 12), rng.randint(0, 12), rng.randint(0, 99)]))
    elif r < 0.7:
        w = rng.choice(_NAMEISH)
    elif r < 0.9:
        n = rng.randint(1, 8)
        alpha = "abcdefghijklmnopqrstuvwxyz"
        w = "".join(rng.choice(alpha) for _ in range(n))
        if rng.random() < 0.3:
            w = w.capitalize()
        elif rng.random() < 0.08:
            w = w.upper()
    else:
        n = rng.randint(1, 6)
        w = "".join(rng.choice(CHARS) for _ in range(n))
    if rng.random() < 0.06:
        w = w + rng.choice(",.:!?")
    return w


def _font(path: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, px)


def _render_panel(rng: random.Random, font_path: str, px: float, width_em: float, n_rows: int,
                  dark: bool) -> Tuple[Image.Image, List[Tuple[str, int, float, float, float, float, float]], Tuple[int, int, int, int]]:
    """A panel-like image of rows of text: ``(image, chars, panel box)``; chars are
    ``(char, row, x_ink0, x_ink1, baseline, x_adv0, x_adv1)`` in final pixels."""
    ss = rng.choice((1, 2, 2, 3))
    fpx = max(6, int(round(px * ss)))
    font = _font(font_path, fpx)
    pitch = px * rng.uniform(1.35, 1.7)
    pad = px * 0.6
    W = int(round(width_em * px + 2 * pad))
    H = int(round(n_rows * pitch + 2 * pad))
    if dark:
        bgc = tuple(rng.randint(25, 70) for _ in range(3))
        stc = tuple(min(255, c + rng.randint(6, 16)) for c in bgc)
        tl = rng.randint(170, 240)
        txt = tuple(max(0, min(255, tl + rng.randint(-12, 12))) for _ in range(3))
    else:
        bgc = tuple(rng.randint(225, 255) for _ in range(3))
        stc = tuple(max(0, c - rng.randint(6, 18)) for c in bgc)
        tl = rng.randint(15, 90)
        txt = tuple(max(0, min(255, tl + rng.randint(-10, 14))) for _ in range(3))
    img = Image.new("RGB", (W * ss, H * ss), bgc)
    d = ImageDraw.Draw(img)
    chars = []
    stripe = rng.random() < 0.6
    for r in range(n_rows):
        ry0 = pad + r * pitch
        if stripe and r % 2 == 1:
            d.rectangle([0, int(ry0 * ss), W * ss, int((ry0 + pitch) * ss)], fill=stc)
        base = ry0 + 0.5 * pitch + 0.36 * px
        x = pad + (px if rng.random() < 0.15 else 0.0)
        words = []
        while True:
            w = _rand_word(rng)
            tw = font.getlength(" ".join(words + [w])) / ss
            if tw > width_em * px - (x - pad) and words:
                break
            words.append(w)
            if len(words) > 12:
                break
        text = " ".join(words)
        d.text((x * ss, base * ss), text, font=font, fill=txt, anchor="ls")
        for k, ch in enumerate(text):
            if ch == " ":
                continue
            adv_end = font.getlength(text[:k + 1])
            adv0 = adv_end - font.getlength(ch)
            bb = font.getbbox(ch, anchor="ls")
            if bb[2] <= bb[0]:
                continue
            chars.append((ch, r, x + (adv0 + bb[0]) / ss, x + (adv0 + bb[2]) / ss, base, x + adv0 / ss,
                          x + adv_end / ss))
    if ss > 1:
        img = img.resize((W, H), Image.LANCZOS)
    return img, chars, (0, 0, W, H)


def _degrade(img: Image.Image, rng: random.Random) -> Tuple[Image.Image, float]:
    """Zoom / blur / noise / JPEG like the benchmark's conditions; returns (image, zoom)."""
    zoom = 1.0
    r = rng.random()
    if r < 0.15:
        zoom = rng.choice((0.8, 1.25, 0.9, 1.1))
        img = img.resize((max(8, int(round(img.width * zoom))), max(8, int(round(img.height * zoom)))), Image.BICUBIC)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    if rng.random() < 0.3:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, rng.uniform(2.0, 10.0), arr.shape)
        img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    if rng.random() < 0.4:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(55, 93))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img, zoom


def _label_segments(bounds: Sequence[int], pairs: Sequence[Tuple[int, int]], run_x0: int,
                    chars: Sequence[Tuple[str, float, float]], xh: float, masked: Optional[set] = None) -> np.ndarray:
    """Class of each candidate segment from the true character ink extents (run coordinates):
    one character's ink mostly inside and no other's -> that character; a mix -> garbage; nothing
    -> -1 (skipped)."""
    labs = np.full(len(pairs), -1, dtype=np.int64)
    # the glyph boxes include the anti-aliased fringe: shrink them a little
    shr = []
    for ch, a, b in chars:
        s = min(0.7, 0.15 * (b - a))
        shr.append((ch, a + s, b - s))
    chars = shr
    for n, (i, j) in enumerate(pairs):
        x0, x1 = bounds[i] + run_x0, bounds[j] + run_x0
        best, best_f, others = -1, 0.0, 0.0
        if masked is not None and n in masked:
            # a masked segment holds exactly the glyphs whose centres are inside it
            inside = [(ch, a, b) for ch, a, b in chars if x0 <= 0.5 * (a + b) < x1]
            if len(inside) == 1:
                labs[n] = CHAR_INDEX.get(inside[0][0], GARBAGE)
            elif inside:
                labs[n] = GARBAGE
            continue
        for ch, a, b in chars:
            wdt = max(0.5, b - a)
            ov = max(0.0, min(x1, b) - max(x0, a))
            f = ov / wdt
            if f > best_f:
                if best >= 0:
                    others = max(others, best_f)
                best, best_f = CHAR_INDEX.get(ch, -1), f
                best_w = ov
            elif f > others:
                others = f
        if best_f < 0.15:
            continue
        if best >= 0 and best_f >= 0.8 and others <= 0.22:
            labs[n] = best
        else:
            labs[n] = GARBAGE
    return labs


def generate_dataset(n_panels: int, seed: int = 0, log=None, keep_char: float = 0.35,
                     keep_garbage: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic segments: ``(X uint8 (N, NUM_FEATURES), y (N,))`` from ``n_panels`` rendered
    panels read by the live pipeline (layout analysis, runs, cuts, features); character / garbage
    segments are kept with probability ``keep_char`` / ``keep_garbage`` (the dense cuts give
    several variants of each character and many garbage spans)."""
    from . import logocr_layout as LL
    rng = random.Random(seed)
    fonts = available_train_fonts()
    if not fonts:
        raise RuntimeError("no training font available")
    fw = np.array([w for _, w in fonts])
    fw = fw / fw.sum()
    npr = np.random.default_rng(seed)
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for k in range(n_panels):
        fp = fonts[int(npr.choice(len(fonts), p=fw))][0]
        px = float(np.exp(rng.uniform(math.log(8.0), math.log(34.0))))
        img, chars, box = _render_panel(rng, fp, px, rng.uniform(11.0, 20.0), rng.randint(3, 6), rng.random() < 0.35)
        img, zoom = _degrade(img, rng)
        arr = np.asarray(img)
        W, H = img.size
        try:
            lay = LL.analyse_panel(arr, (0, 0, W, H), text_only=True)
        except Exception:
            continue
        ix0, iy0 = lay.inner[0], lay.inner[1]
        for band in lay.bands:
            # the true characters of this row: the row whose baseline is nearest the band's
            by = band.baseline + iy0
            rows = {}
            for ch, r, a, b, base, _, _ in chars:
                rows.setdefault(r, []).append((ch, a * zoom - ix0, b * zoom - ix0, base * zoom))
            if not rows:
                continue
            r_best = min(rows, key=lambda r: abs(rows[r][0][3] - by))
            if abs(rows[r_best][0][3] - by) > 0.5 * lay.xh + 2:
                continue
            row_chars = [(ch, a, b) for ch, a, b, _ in rows[r_best]]
            for run in band.items:
                if not isinstance(run, LL.TextRun):
                    continue
                ri = run_ink(lay, band, run)
                bounds, gaps = cut_candidates(ri.A, ri.xh)
                if len(bounds) < 2:
                    continue
                info: Dict[str, Any] = {}
                X, pairs = segment_features(ri, bounds, gaps, info=info)
                labs = _label_segments(bounds, pairs, ri.x0, row_chars, ri.xh, info.get("masked"))
                u = npr.random(labs.shape[0])
                keep = (labs >= 0) & np.where(labs == GARBAGE, u < keep_garbage, u < keep_char)
                if not keep.any():
                    continue
                Xs.append(np.clip(np.rint(X[keep] * np.r_[np.full(PATCH_H * PATCH_W, 255.0),
                                                          np.full(N_EXTRA, 100.0)]), 0, 255).astype(np.uint8))
                ys.append(labs[keep])
        if log is not None and (k + 1) % 200 == 0:
            log(f"panels {k + 1}/{n_panels}: {sum(len(y) for y in ys)} segments")
    X = np.concatenate(Xs) if Xs else np.zeros((0, NUM_FEATURES), dtype=np.uint8)
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
    return X, y


#: Features are stored / fed to the classifier in these units (uint8-sized: patch x255, extras x100).
STORE_SCALE = np.r_[np.full(PATCH_H * PATCH_W, 255.0), np.full(N_EXTRA, 100.0)].astype(np.float32)
