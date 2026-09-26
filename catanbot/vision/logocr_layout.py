"""Geometry of Colonist's game-log panel for the local log OCR (:mod:`catanbot.vision.logocr`).

Everything here is measured from the image; nothing assumes the synthetic renderer's exact
colours or sizes (its palette only seeds the name-colour classifier).

Pipeline
    image -> :func:`as_rgb` (path / PIL / numpy -> ``H x W x 3`` uint8)
    -> :func:`detect_panel` (long straight vertical edges paired into a rounded rectangle whose
       inside is mostly one or two flat colours: the panel box, pixels ``x0, y0, x1, y1``)
    -> :func:`analyse_panel`:
       * the border is skipped (``inner``), the panel fill and the stripe colour are estimated
         from the per-row colour statistics, each pixel row gets its background colour;
       * ink = colour distance to the row's background; ink rows form *bands* (text rows);
       * connected components of the ink become glyph pieces, card / die icons (solid or
         framed blobs about one em high) and words (pieces grouped by gaps narrower than a
         space); each word gets an ink colour (the most saturated core pixels, unmixed from
         the background) so names (coloured bold words) are told from text;
       * bands are grouped into log entries by the stripe background (alternating per entry)
         and by the continuation-row indent; the top entry is ``partial`` when the panel's
         top edge cuts it (its expected row top, from the baseline, lies above the edge; its
         ink touches the edge; or its head row is missing).

The result (:class:`PanelLayout`) holds per-entry :class:`Word` / :class:`Icon` tokens for
the glyph reader (:mod:`catanbot.vision.logocr_glyphs`) and the decoder
(:mod:`catanbot.vision.logocr_decode`).
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # OpenCV is optional: numpy fallbacks keep the reader working without it
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only without OpenCV
    cv2 = None

from PIL import Image

__all__ = [
    "Box",
    "as_rgb",
    "detect_panel",
    "Icon",
    "Word",
    "Band",
    "Entry",
    "PanelLayout",
    "analyse_panel",
    "ICON_REFS",
    "NAME_REFS",
]

Box = Tuple[int, int, int, int]
RGB = Tuple[int, int, int]

#: Reference fill colours of the log's card icons (Colonist-like; the synthetic panel's) - used
#: only as the nearest-colour prior, the dark theme / jitter are covered by the distance margin.
ICON_REFS: Dict[str, List[RGB]] = {
    "wood": [(34, 105, 48), (40, 110, 50)],
    "brick": [(185, 78, 40), (200, 95, 50)],
    "sheep": [(150, 200, 80), (140, 195, 85)],
    "wheat": [(240, 200, 60), (232, 190, 60)],
    "ore": [(105, 115, 135), (130, 130, 138)],
    "card": [(135, 45, 40), (110, 40, 40)],
    "dev": [(120, 70, 170), (140, 80, 190)],
}
#: Name colours per colour word (light-panel and dark-panel variants).  A reader given
#: ``players`` ({colour: RGB}) or a taught profile uses those instead.
NAME_REFS: Dict[str, List[RGB]] = {
    "red": [(226, 74, 64), (240, 100, 90)],
    "blue": [(48, 120, 212), (95, 155, 240)],
    "orange": [(238, 140, 40)],
    "white": [(150, 150, 150), (235, 235, 235)],
    "green": [(60, 160, 80), (85, 195, 105)],
    "brown": [(120, 80, 50), (185, 135, 95)],
    "purple": [(140, 80, 190), (180, 130, 230)],
    "pink": [(230, 120, 180)],
}


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def as_rgb(img: Any) -> np.ndarray:
    """``img`` (file path, PIL image or numpy array in RGB order) -> contiguous ``H x W x 3`` uint8."""
    if isinstance(img, (str, os.PathLike)):
        with Image.open(img) as im:
            return np.ascontiguousarray(np.asarray(im.convert("RGB")))
    if isinstance(img, Image.Image):
        return np.ascontiguousarray(np.asarray(img.convert("RGB")))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected an RGB image, got an array of shape {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        if a.size and a.max() <= 1.0:
            a = a * 255.0
        arr = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_area(arr: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return arr
    h, w = arr.shape[0] // f, arr.shape[1] // f
    if cv2 is not None:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    a = arr[:h * f, :w * f].reshape(h, f, w, f, -1).mean(axis=(1, 3))
    return np.rint(a).astype(np.uint8)


def _components(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """8-connected components: ``(n, labels, stats)`` with stats rows ``x, y, w, h, area``
    (label 0 is the background), like ``cv2.connectedComponentsWithStats``."""
    m = mask.astype(np.uint8)
    if cv2 is not None:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        return n, lab, stats
    # numpy / python fallback: union-find over runs
    h, w = m.shape
    lab = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = 1
    for y in range(h):
        row = m[y]
        for x in range(w):
            if not row[x]:
                continue
            neigh = []
            if x > 0 and lab[y, x - 1]:
                neigh.append(lab[y, x - 1])
            if y > 0:
                for dx in (-1, 0, 1):
                    if 0 <= x + dx < w and lab[y - 1, x + dx]:
                        neigh.append(lab[y - 1, x + dx])
            if not neigh:
                parent.append(nxt)
                lab[y, x] = nxt
                nxt += 1
            else:
                r = min(find(a) for a in neigh)
                lab[y, x] = r
                for a in neigh:
                    ra = find(a)
                    if ra != r:
                        parent[ra] = r
    roots = {}
    for a in range(1, nxt):
        r = find(a)
        roots.setdefault(r, len(roots) + 1)
    remap = np.zeros(nxt, dtype=np.int32)
    for a in range(1, nxt):
        remap[a] = roots[find(a)]
    lab = remap[lab]
    n = len(roots) + 1
    stats = np.zeros((n, 5), dtype=np.int32)
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        stats[k] = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, xs.size)
    return n, lab, stats


# ---------------------------------------------------------------------------
# panel detection
# ---------------------------------------------------------------------------
def _runs(col: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Runs of True in a 1-D boolean array at least ``min_len`` long: ``[(start, end)]``."""
    if not col.any():
        return []
    d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_len
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def _close1(e: np.ndarray, axis: int) -> np.ndarray:
    """Fill one-pixel holes along ``axis`` (a pixel between two edge pixels becomes edge)."""
    out = e.copy()
    if axis == 0:
        out[1:-1] |= e[:-2] & e[2:]
    else:
        out[:, 1:-1] |= e[:, :-2] & e[:, 2:]
    return out


def _interior_stats(small: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[float, int]:
    """Of the inside of ``box`` (downsampled image): the share of pixels within 14 of the two most
    common (4-bit quantised) colours ("flat") and the number of text-like bands (runs of rows with
    a little ink between clean rows)."""
    x0, y0, x1, y1 = box
    mx, my = max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 40)
    sub = small[y0 + my:y1 - my, x0 + mx:x1 - mx].astype(np.int16)
    if sub.size == 0:
        return 0.0, 0
    flat_px = sub.reshape(-1, 3)
    q = flat_px >> 4
    codes = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    cnt = np.bincount(codes, minlength=4096)
    top = np.argsort(cnt)[::-1][:2]
    flat = np.zeros(codes.shape, dtype=bool)
    for c in top:
        if cnt[c] <= 0:
            continue
        ref = np.median(flat_px[codes == c], axis=0)
        flat |= np.abs(flat_px - ref).max(axis=1) <= 14
    frac = float(flat.mean())
    row_ink = 1.0 - flat.reshape(sub.shape[:2]).mean(axis=1)
    texty = (row_ink > 0.015) & (row_ink < 0.7)
    bands = len(_runs(texty, 1))
    return frac, bands


def _vertical_lines(ex: np.ndarray, min_len: int) -> List[Tuple[int, int, int, int]]:
    """Long vertical edge runs grouped over neighbouring columns: ``(xa, xb, y0, y1)``."""
    runs: List[Tuple[int, int, int]] = []
    cols = np.flatnonzero(ex.sum(axis=0) >= min_len)
    for x in cols.tolist():
        for a, b in _runs(ex[:, x], min_len):
            runs.append((x, a, b))
    lines: List[List[int]] = []
    for x, a, b in runs:
        for ln in lines:
            if x - ln[1] <= 1 and min(b, ln[3]) - max(a, ln[2]) >= 0.5 * min(b - a, ln[3] - ln[2]):
                ln[1] = x
                ln[2] = min(ln[2], a)
                ln[3] = max(ln[3], b)
                break
        else:
            lines.append([x, x, a, b])
    return [tuple(ln) for ln in lines]  # type: ignore[misc]


def detect_panel(arr: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None
                 ) -> Tuple[Optional[Box], float]:
    """Find the log panel in ``arr`` (``H x W x 3`` uint8): ``(box or None, score)``.

    The panel is a rounded rectangle with a thin border whose inside is mostly one or two flat
    colours (fill and stripe) with rows of text.  Long straight vertical edges (at least 15 % of the
    image height) are paired left / right, closed by horizontal edges at the top and the bottom,
    and scored by edge coverage, flatness of the inside and its text-like rows; the best box is
    refined at full resolution.  ``region`` (pixels) limits the search.
    """
    H, W = arr.shape[:2]
    ox = oy = 0
    if region is not None:
        rx0, ry0, rx1, ry1 = (int(round(v)) for v in region)
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(W, rx1), min(H, ry1)
        arr = arr[ry0:ry1, rx0:rx1]
        ox, oy = rx0, ry0
        H, W = arr.shape[:2]
    if H < 40 or W < 40:
        return None, 0.0
    f = max(1, int(round(max(W, H) / 640.0)))
    small = _resize_area(arr, f)
    h, w = small.shape[:2]
    s = small.astype(np.int16)
    T = 20
    ex = np.abs(s[:, 1:] - s[:, :-1]).max(axis=2) > T      # edge between x and x + 1
    ey = np.abs(s[1:] - s[:-1]).max(axis=2) > T            # edge between y and y + 1
    ex = _close1(ex, 0)
    ey = _close1(ey, 1)
    ey2 = ey.copy()
    ey2[:-1] |= ey[1:]                                     # an edge smeared over two rows
    min_len = max(12, int(0.15 * h))
    lines = _vertical_lines(ex, min_len)
    if len(lines) < 2:
        return None, 0.0
    lines.sort()
    best: Optional[Tuple[float, Tuple[int, int, int, int]]] = None
    min_w = max(16, int(0.04 * w))
    cov_rows = np.cumsum(np.concatenate([np.zeros((ey2.shape[0], 1), dtype=np.int32),
                                         ey2.astype(np.int32)], axis=1), axis=1)
    for i, (la, lb, al, bl) in enumerate(lines):
        for (ra, rb, ar, br) in lines[i + 1:]:
            if ra - lb < min_w or ra - lb > 0.7 * w:
                continue
            lo, hi = max(al, ar), min(bl, br)
            if hi - lo < min_len or hi - lo < 0.6 * min(bl - al, br - ar):
                continue
            xs0, xs1 = la + 1, rb + 1            # inside columns (small image)
            pad = max(6, int(0.02 * h))
            # coverage of horizontal edges per row between the two lines
            def best_row(ya: int, yb: int, outermost_first: bool) -> Tuple[int, float]:
                # the outermost covered row just past the end of the straight edge (the rounded
                # corner): not a farther panel's edge, not a stripe / text edge inside
                ya, yb = max(0, ya), min(ey2.shape[0], yb)
                if yb <= ya:
                    return ya, 0.0
                cov = (cov_rows[ya:yb, xs1] - cov_rows[ya:yb, xs0]) / max(1, xs1 - xs0)
                ok = np.flatnonzero(cov >= 0.5)
                if ok.size == 0:
                    k = int(np.argmax(cov))
                else:
                    k = int(ok[0] if outermost_first else ok[-1])
                return ya + k, float(cov[k])
            yt, cov_t = best_row(lo - pad, lo + 3, True)
            yb_, cov_b = best_row(hi - 3, hi + pad, False)
            if cov_t < 0.5 or cov_b < 0.5:
                continue
            y0, y1 = yt + 1, yb_ + 1
            if y1 - y0 < min_len:
                continue
            flat, bands = _interior_stats(small, (xs0, y0, xs1, y1))
            if flat < 0.45:
                continue
            vcov = min(1.0, (hi - lo) / max(1.0, (y1 - y0) * 0.8))
            score = (cov_t + cov_b + vcov + 2.0 * flat + 1.5 * min(bands, 8) / 8.0
                     + 0.5 * min(1.0, (y1 - y0) / (0.4 * h)))
            if best is None or score > best[0]:
                best = (score, (xs0, y0, xs1, y1))
    if best is None:
        return None, 0.0
    score, (x0, y0, x1, y1) = best
    box = _refine_box(arr, (x0 * f, y0 * f, x1 * f, y1 * f), f)
    bx0, by0, bx1, by1 = box
    return (bx0 + ox, by0 + oy, bx1 + ox, by1 + oy), float(score)


def _refine_box(arr: np.ndarray, box: Tuple[int, int, int, int], f: int) -> Box:
    """Snap a box found at ``1/f`` scale to the strongest full-resolution edges nearby."""
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = box
    r = f + 2
    ym0, ym1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
    xm0, xm1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
    a = arr.astype(np.int16)

    def vedge(xc: int, outside_left: bool) -> int:
        best, bx = -1.0, xc
        for x in range(max(1, xc - r), min(W - 1, xc + r + 1)):
            g = float(np.abs(a[ym0:ym1, x] - a[ym0:ym1, x - 1]).max(axis=1).mean())
            if g > best:
                best, bx = g, x
        return bx          # first column of the right-hand side of the edge

    def hedge(yc: int) -> int:
        best, by = -1.0, yc
        for y in range(max(1, yc - r), min(H - 1, yc + r + 1)):
            g = float(np.abs(a[y, xm0:xm1] - a[y - 1, xm0:xm1]).max(axis=1).mean())
            if g > best:
                best, by = g, y
        return by

    nx0 = vedge(x0, True)
    nx1 = vedge(x1, False)
    ny0 = hedge(y0)
    ny1 = hedge(y1)
    if nx1 - nx0 < 10 or ny1 - ny0 < 10:
        return (x0, y0, x1, y1)
    return (int(nx0), int(ny0), int(nx1), int(ny1))


# ---------------------------------------------------------------------------
# panel analysis
# ---------------------------------------------------------------------------
@dataclass
class Icon:
    """A card / die icon on a text row (panel-inner pixel coordinates)."""

    x0: int
    x1: int
    y0: int
    y1: int
    kind: str = "?"              # wood brick sheep wheat ore card dev die
    face: int = 0                # die face 1..6
    conf: float = 0.0
    fill: RGB = (0, 0, 0)


@dataclass
class Word:
    """A run of glyph pieces between spaces (panel-inner coordinates).

    ``colour`` is the unmixed core ink colour, ``cls`` ``"text"`` or ``"name"`` (set by the
    colour classifier), ``pieces`` the column extents of the connected glyph groups.
    """

    x0: int
    x1: int
    y0: int
    y1: int
    colour: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    contrast: float = 0.0
    stroke: float = 0.0
    cls: str = "text"
    name: Optional[str] = None           # colour word when cls == "name"
    name_conf: float = 0.0
    pieces: List[Tuple[int, int]] = field(default_factory=list)
    comp_ids: List[int] = field(default_factory=list)
    suffix: str = ""                     # glued text-coloured punctuation after a name (":")


@dataclass
class Band:
    """One text row: ink extent ``y0..y1`` (exclusive), baseline, left / right ink."""

    y0: int
    y1: int
    baseline: float = 0.0
    xline: float = 0.0
    left: int = 0
    right: int = 0
    head: bool = True
    bg: RGB = (0, 0, 0)
    stripe: int = 0                      # background class of the band's rows (0 fill, 1 stripe)
    items: List[Any] = field(default_factory=list)   # Word / Icon in x order


@dataclass
class Entry:
    """A log entry: its bands (rows), top / bottom (pixels, panel-inner), partial flag, key."""

    bands: List[Band]
    top: float = 0.0
    bottom: float = 0.0
    partial: bool = False
    partial_why: str = ""
    key: str = ""
    stripe: int = 0


@dataclass
class PanelLayout:
    """Everything measured on one panel (coordinates relative to ``inner`` unless noted)."""

    box: Box                             # panel (image pixels)
    inner: Box                           # content area (image pixels)
    rgb: np.ndarray                      # inner crop, uint8
    bg_rows: np.ndarray                  # (h, 3) float background colour per row
    row_cls: np.ndarray                  # (h,) background class per row (0 fill, 1 stripe, -1 unknown)
    fill: Tuple[float, float, float]
    stripe: Optional[Tuple[float, float, float]]
    noise: float
    dist: np.ndarray                     # (h, w) float32 colour distance to the row background
    ink: np.ndarray                      # (h, w) bool
    labels: np.ndarray                   # component labels of ``ink``
    stats: np.ndarray
    text_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    text_contrast: float = 1.0
    dark: bool = False
    pitch: float = 0.0
    xh: float = 0.0
    em: float = 0.0
    bands: List[Band] = field(default_factory=list)
    entries: List[Entry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    content_top: int = 0                 # first fill row below the top border (inner coords)


def _inner_box(arr: np.ndarray, box: Box) -> Tuple[Box, Tuple[float, float, float]]:
    """Skip the panel border: ``(inner box, fill colour)`` (image pixels)."""
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = arr[y0:y1, x0:x1].astype(np.int16)
    ch, cw = crop.shape[:2]
    if ch < 8 or cw < 8:
        return (x0, y0, x1, y1), tuple(float(v) for v in crop.reshape(-1, 3).mean(axis=0))  # type: ignore
    core = crop[ch // 8: ch - ch // 8, cw // 6: cw - cw // 6].reshape(-1, 3)
    q = core >> 3
    codes = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    top = int(np.argmax(np.bincount(codes)))
    fill = np.median(core[codes == top], axis=0)
    lim = max(3, min(ch, cw) // 6)

    def first_fill(lines: np.ndarray) -> int:
        # lines: (k, n, 3) from the edge inward; first index after the border that looks like fill
        seen_border = False
        for i in range(min(lim, lines.shape[0])):
            ln = lines[i]
            med = np.median(ln, axis=0)
            near = float((np.abs(ln - fill).max(axis=1) <= 22).mean())
            if near > 0.5 and np.abs(med - fill).max() <= 22:
                if seen_border or i > 0:
                    return i
                return i
            seen_border = True
        return min(lim, lines.shape[0])

    ys = slice(ch // 4, ch - ch // 4)
    xs = slice(cw // 4, cw - cw // 4)
    left = first_fill(np.transpose(crop[ys, :], (1, 0, 2)))
    right = first_fill(np.transpose(crop[ys, ::-1], (1, 0, 2)))
    top = first_fill(crop[:, xs])
    bottom = first_fill(crop[::-1, xs])
    return (x0 + left, y0 + top, x1 - right, y1 - bottom), (float(fill[0]), float(fill[1]), float(fill[2]))


def _row_backgrounds(crop: np.ndarray, fill: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, Tuple, Optional[Tuple],
                                                                        float]:
    """Per-row background colour and class (0 fill, 1 stripe), the stripe colour, the noise sigma."""
    h, w = crop.shape[:2]
    c = crop.astype(np.float32)
    med = np.median(c, axis=1)                                        # (h, 3)
    # noise: the per-row median absolute deviation (green channel) of the quietest rows
    mad = np.median(np.abs(c[:, :, 1] - med[:, None, 1]), axis=1)
    noise = float(1.4826 * np.percentile(mad, 20)) if h else 0.0
    dev = np.abs(c - med[:, None, :]).max(axis=2)                     # (h, w)
    support = (dev <= max(10.0, 3.5 * noise)).mean(axis=1)
    fill_a = np.asarray(fill, dtype=np.float32)
    good = support >= 0.55
    # candidate colours: fill and (maybe) stripe = the most common well-supported row colour far from fill
    stripe: Optional[Tuple[float, float, float]] = None
    if good.any():
        gm = med[good]
        far = np.abs(gm - fill_a).max(axis=1)
        cand = gm[(far >= max(6.0, 2.5 * noise)) & (far <= 60.0)]
        if cand.shape[0] >= max(3, 0.08 * good.sum()):
            q = np.rint(cand / 5.0).astype(np.int32)
            keys, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
            k = int(np.argmax(cnt))
            sel = cand[inv.reshape(-1) == k]
            near = cand[np.abs(cand - sel.mean(axis=0)).max(axis=1) <= 6.0]
            st = near.mean(axis=0)
            if np.abs(st - fill_a).max() >= 3.0 and near.shape[0] >= max(3, 0.06 * good.sum()):
                stripe = (float(st[0]), float(st[1]), float(st[2]))
    tol = max(6.0, 3.0 * noise)
    if stripe is not None:
        tol = min(tol, 0.5 * float(np.abs(np.asarray(stripe) - fill_a).max()) + 3.0)
    tol = max(tol, 2.5 * noise)
    near_f = (np.abs(c - fill_a).max(axis=2) <= tol).mean(axis=1)
    if stripe is not None:
        st_a = np.asarray(stripe, dtype=np.float32)
        near_s = (np.abs(c - st_a).max(axis=2) <= tol).mean(axis=1)
        # a well-supported row: its median decides (robust to noise)
        dm_f = np.abs(med - fill_a).max(axis=1)
        dm_s = np.abs(med - st_a).max(axis=1)
        cls = np.where(good, np.where(dm_s < dm_f, 1, 0), np.where(near_s > near_f, 1, 0)).astype(np.int8)
    else:
        near_s = np.zeros(h, dtype=np.float32)
        cls = np.zeros(h, dtype=np.int8)
    weak = (~good) & (np.maximum(near_f, near_s) < 0.12)
    cls[weak] = -1
    # rows with no clear background inherit it from the nearest clear row above / below
    if weak.any() and (~weak).any():
        idx = np.arange(h)
        clear = np.flatnonzero(~weak)
        pos = np.searchsorted(clear, idx)
        lo = clear[np.clip(pos - 1, 0, clear.size - 1)]
        hi = clear[np.clip(pos, 0, clear.size - 1)]
        pick = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)
        cls[weak] = cls[pick[weak]]
    elif weak.all():
        cls[:] = 0
    bg = np.empty((h, 3), dtype=np.float32)
    bg[:] = fill_a
    if stripe is not None:
        bg[cls == 1] = np.asarray(stripe, dtype=np.float32)
    # refine: the actual per-row median where the row is well supported and near its class colour
    ok = good & (np.abs(med - bg).max(axis=1) <= tol)
    bg[ok] = med[ok]
    return bg, cls, (float(fill_a[0]), float(fill_a[1]), float(fill_a[2])), stripe, noise


def _mode_int(vals: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return 0.0
    wts = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    iv = np.rint(v).astype(np.int64)
    lo = iv.min()
    cnt = np.bincount(iv - lo, weights=wts)
    # smooth by one neighbour so a split mode still wins
    sm = cnt.copy()
    sm[1:] += 0.5 * cnt[:-1]
    sm[:-1] += 0.5 * cnt[1:]
    return float(np.argmax(sm) + lo)


@dataclass
class _Comps:
    """Per-component measurements (index = label; 0 unused)."""

    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    area: np.ndarray
    maxd: np.ndarray            # max colour distance to the background
    core: np.ndarray            # (n, 3) mean RGB of the component's high-contrast pixels
    core_diff: np.ndarray       # (n, 3) mean (pixel - background) of those pixels


def _measure_components(crop: np.ndarray, bg: np.ndarray, dist: np.ndarray, labels: np.ndarray,
                        stats: np.ndarray) -> _Comps:
    n = stats.shape[0]
    x0 = stats[:, 0].astype(np.int32)
    y0 = stats[:, 1].astype(np.int32)
    x1 = x0 + stats[:, 2]
    y1 = y0 + stats[:, 3]
    area = stats[:, 4].astype(np.int32)
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    d = dist[ys, xs]
    maxd = np.zeros(n, dtype=np.float32)
    np.maximum.at(maxd, lab, d)
    core_sel = d >= 0.75 * maxd[lab]
    cl = lab[core_sel]
    rgb = crop[ys[core_sel], xs[core_sel]].astype(np.float64)
    dif = rgb - bg[ys[core_sel]]
    cnt = np.bincount(cl, minlength=n).astype(np.float64)
    core = np.zeros((n, 3))
    core_diff = np.zeros((n, 3))
    for ch in range(3):
        core[:, ch] = np.bincount(cl, weights=rgb[:, ch], minlength=n)
        core_diff[:, ch] = np.bincount(cl, weights=dif[:, ch], minlength=n)
    cnt[cnt == 0] = 1.0
    core /= cnt[:, None]
    core_diff /= cnt[:, None]
    return _Comps(x0, y0, x1, y1, area, maxd, core, core_diff)


def _find_bands(ink_rows: np.ndarray, min_px: int) -> List[Tuple[int, int]]:
    """Text rows: runs of pixel rows with at least ``min_px`` ink pixels (holes of one row closed);
    slivers (a descender's tail, an accent) are merged into the nearest row when close to it."""
    on = ink_rows >= min_px
    on = on.copy()
    on[1:-1] |= on[:-2] & on[2:]
    runs = [list(r) for r in _runs(on, 1)]
    if len(runs) < 2:
        return [tuple(r) for r in runs]  # type: ignore[misc]
    hs = np.array([b - a for a, b in runs], dtype=np.float64)
    med = float(np.median(hs))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        hs = np.array([b - a for a, b in runs], dtype=np.float64)
        order = np.argsort(hs)
        for k in order.tolist():
            a, b = runs[k]
            if b - a >= 0.45 * med:
                break
            # a sliver joins the nearest row (a descender's tail, an i-dot over a row without
            # ascenders); a sliver at the very top is the remains of a row cut by the panel edge
            # and stays on its own
            if k == 0 and a <= max(2.0, 0.5 * med):
                continue
            gap_up = a - runs[k - 1][1] if k > 0 else 1e9
            gap_dn = runs[k + 1][0] - b if k + 1 < len(runs) else 1e9
            j = k - 1 if gap_up <= gap_dn else k + 1
            if min(gap_up, gap_dn) > max(2.0, 0.3 * med):
                continue
            runs[j] = [min(runs[j][0], a), max(runs[j][1], b)]
            del runs[k]
            changed = True
            break
    return [tuple(r) for r in runs]  # type: ignore[misc]


def _profile_pitch(prof: np.ndarray, min_lag: int, max_lag: int) -> float:
    """The period of a (row-ink) profile: the highest autocorrelation peak between the lags."""
    p = prof.astype(np.float64) - float(prof.mean())
    n = p.size
    max_lag = min(max_lag, n - 2)
    if max_lag <= min_lag:
        return 0.0
    ac = np.array([float(np.dot(p[:n - L], p[L:])) / (n - L) for L in range(min_lag, max_lag + 1)])
    if ac.size < 3:
        return 0.0
    k = int(np.argmax(ac))
    # the shortest strong peak (a peak at twice the pitch can win when rows alternate in density)
    top = ac[k]
    for j in range(1, ac.size - 1):
        if ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1] and ac[j] >= 0.6 * top:
            k = j
            break
    L = min_lag + k
    # sub-pixel: parabola through the peak
    if 0 < k < ac.size - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        if den < 0:
            L += 0.5 * (a - c) / den
    return float(L)


def _split_tall(bands: List[Tuple[int, int]], prof: np.ndarray, pitch: float) -> List[Tuple[int, int]]:
    """Split bands spanning several text rows (rows bridged by blur / JPEG halos) at the profile
    minima nearest the expected row boundaries."""
    if pitch <= 0:
        return bands
    out: List[Tuple[int, int]] = []
    sm = np.convolve(prof.astype(np.float64), np.ones(3) / 3.0, mode="same")
    for a, b in bands:
        H = b - a
        if H <= 1.3 * pitch:
            out.append((a, b))
            continue
        k = max(2, int(round((H + 0.3 * pitch) / pitch)))
        cuts = [a]
        for i in range(1, k):
            yc = a + i * H / k
            lo = int(max(cuts[-1] + 2, yc - 0.3 * pitch))
            hi = int(min(b - 2, yc + 0.3 * pitch))
            if hi <= lo:
                continue
            y = lo + int(np.argmin(sm[lo:hi + 1]))
            cuts.append(y)
        cuts.append(b)
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            # trim empty rows at the cut
            while y0 < y1 and prof[y0] == 0:
                y0 += 1
            while y1 > y0 and prof[y1 - 1] == 0:
                y1 -= 1
            if y1 > y0:
                out.append((y0, y1))
    return out


def _saturation(rgb: Sequence[float]) -> float:
    r, g, b = (float(v) for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    return (mx - mn)


def analyse_panel(arr: np.ndarray, box: Box, palette: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
                  text_only: bool = False) -> PanelLayout:
    """Measure the panel ``box`` of ``arr``: background rows, ink, bands, tokens and entries
    (words are not read here: see :mod:`catanbot.vision.logocr`).  ``palette`` (colour word ->
    RGBs, :func:`name_palette`) names the name colours; ``text_only`` (training panels) takes every
    glyph as text and looks for no icons."""
    inner, fill = _inner_box(arr, box)
    ix0, iy0, ix1, iy1 = inner
    crop = np.ascontiguousarray(arr[iy0:iy1, ix0:ix1])
    h, w = crop.shape[:2]
    bg, cls, fill, stripe, noise = _row_backgrounds(crop, fill)
    c = crop.astype(np.float32)
    if noise >= 2.0 and cv2 is not None:      # noisy capture: measure ink on a lightly smoothed copy
        c = cv2.GaussianBlur(c, (3, 3), 0.7)
        noise_eff = 0.55 * noise
    else:
        noise_eff = noise
    diff = c - bg[:, None, :]
    dist = np.abs(diff).max(axis=2).astype(np.float32)
    thr = max(26.0, 5.0 * noise_eff + 14.0)
    ink = dist > thr
    n, labels, stats = _components(ink)
    fill_lum = float(np.mean(fill))
    lay = PanelLayout(box=tuple(int(round(v)) for v in box), inner=inner, rgb=crop, bg_rows=bg, row_cls=cls,
                      fill=fill, stripe=stripe, noise=noise, dist=dist, ink=ink, labels=labels, stats=stats,
                      dark=fill_lum < 110.0)
    lay._palette = palette                          # type: ignore[attr-defined]
    lay._text_only = text_only                      # type: ignore[attr-defined]
    if n <= 1 or h < 6 or w < 6:
        return lay
    cm = _measure_components(crop, bg, dist, labels, stats)
    # specks (noise, JPEG ringing) are not ink; neither are the border's remains: text never
    # touches the panel's sides or its bottom (padding), the rounded corners and a thin line
    # along the top edge are border
    speck = (cm.area <= 2) & (cm.maxd < 2.0 * thr)
    side = (cm.x0 <= 0) | (cm.x1 >= w) | (cm.y1 >= h)
    topline = (cm.y0 <= 0) & (cm.y1 - cm.y0 <= 2) & (cm.x1 - cm.x0 >= 0.5 * w)
    speck |= side | topline
    speck[0] = True
    keep = ~speck
    valid = keep[labels] & ink
    # text rows from the strong ink only (JPEG halos and blur tails bridge the gaps between rows)
    dv = dist[valid]
    c95 = float(np.percentile(dv, 95)) if dv.size else thr
    strong = valid & (dist > max(thr, 0.3 * c95))
    ink_rows = strong.sum(axis=1)
    bands_raw = _find_bands(ink_rows, 1)
    if not bands_raw:
        return lay
    pitch = 0.0
    if len(bands_raw) >= 2:
        # consecutive rows are one pitch apart (no gaps inside the stack): the median centre
        # difference of ordinary bands, refined by the autocorrelation of the row-ink profile
        hs = np.array([b - a for a, b in bands_raw], dtype=np.float64)
        ctr = np.array([(a + b) / 2.0 for a, b in bands_raw])
        dd = np.diff(ctr)
        ok = (hs[:-1] <= 1.25 * np.median(hs)) & (hs[1:] <= 1.25 * np.median(hs))
        p0 = float(np.median(dd[ok])) if ok.any() else float(np.median(dd))
        pitch = _profile_pitch(ink_rows, max(3, int(0.75 * p0)), max(4, int(np.ceil(1.3 * p0))))
        if pitch <= 0 or abs(pitch - p0) > 0.3 * p0:
            pitch = p0
    bands_raw = _split_tall(bands_raw, ink_rows, pitch)
    lay.pitch = pitch
    lay.ink = valid
    lay.thr = thr                                   # type: ignore[attr-defined]
    lay._thr_weak = max(12.0, 4.0 * noise_eff + 8.0)      # type: ignore[attr-defined]
    # each band reaches halfway into the gaps around it; components are taken per band so a
    # halo bridging two rows cannot join their glyphs
    ext: List[Tuple[int, int]] = []
    for j, (a, b) in enumerate(bands_raw):
        top = (bands_raw[j - 1][1] + a + 1) // 2 if j > 0 else max(0, a - 2)
        bot = (b + bands_raw[j + 1][0]) // 2 if j + 1 < len(bands_raw) else min(h, b + 2)
        ext.append((max(0, min(top, a)), min(h, max(bot, b))))
    heights = [b - a for a, b in bands_raw]
    em0 = lay.pitch / 1.5 if lay.pitch > 0 else max(4.0, float(np.median(heights)) / 0.75)
    lay.em = em0
    per_band: List[Tuple[Band, _Comps, np.ndarray]] = []
    for (a, b), (ya, yb) in zip(bands_raw, ext):
        sub = strong[ya:yb]
        n2, lab2, st2 = _components(sub)
        if n2 <= 1:
            continue
        cm2 = _measure_components(crop[ya:yb], bg[ya:yb], dist[ya:yb], lab2, st2)
        cm2.y0 = cm2.y0 + ya
        cm2.y1 = cm2.y1 + ya
        band = Band(y0=int(a), y1=int(b))
        band._ya, band._yb, band._lab = ya, yb, lab2      # type: ignore[attr-defined]
        vals = lay.row_cls[a:b]
        vals = vals[vals >= 0]
        band.stripe = int(np.bincount(vals).argmax()) if vals.size else 0
        per_band.append((band, cm2, st2))
    lay.xh = 0.53 * em0                  # refined by _band_metrics
    bands: List[Band] = []
    for band, cm2, st2 in per_band:
        _band_items(lay, band, cm2, st2, em0)
        if band.items:
            bands.append(band)
    lay.bands = bands
    _band_metrics(lay)
    _classify_colours(lay)
    for band in lay.bands:
        _group_band(lay, band)
    _group_entries(lay)
    return lay


# ---------------------------------------------------------------------------
# icons
# ---------------------------------------------------------------------------
def _ring_score(mask: np.ndarray) -> Tuple[float, float]:
    """(share of the bbox border band covered, share of the inner box covered) of a mask."""
    hh, ww = mask.shape
    t = max(1, int(0.09 * min(hh, ww)))
    border = np.zeros_like(mask)
    border[:t] = True
    border[-t:] = True
    border[:, :t] = True
    border[:, -t:] = True
    b = float(mask[border].mean()) if border.any() else 0.0
    iy0, iy1 = int(0.3 * hh), int(np.ceil(0.7 * hh))
    ix0, ix1 = int(0.3 * ww), int(np.ceil(0.7 * ww))
    core = mask[iy0:iy1, ix0:ix1]
    i = float(core.mean()) if core.size else 1.0
    return b, i


def _split_positions(prof: np.ndarray, n: int) -> List[int]:
    """Cut columns of a run of ``n`` glued icons: profile minima near the equal split."""
    wtot = prof.size
    cuts = [0]
    for i in range(1, n):
        xc = i * wtot / n
        lo = int(max(cuts[-1] + 2, xc - 0.25 * wtot / n))
        hi = int(min(wtot - 2, xc + 0.25 * wtot / n))
        if hi <= lo:
            cuts.append(int(round(xc)))
            continue
        cuts.append(lo + int(np.argmin(prof[lo:hi + 1])))
    cuts.append(wtot)
    return cuts


_DIE_LAYOUTS: Dict[int, List[List[Tuple[int, int]]]] = {
    1: [[(0, 0)]],
    2: [[(-1, -1), (1, 1)], [(1, -1), (-1, 1)]],
    3: [[(-1, -1), (0, 0), (1, 1)], [(1, -1), (0, 0), (-1, 1)]],
    4: [[(-1, -1), (1, -1), (-1, 1), (1, 1)]],
    5: [[(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)]],
    6: [[(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
        [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]],
}


def _die_face(dark_map: np.ndarray) -> Tuple[int, float]:
    """Face value of a die from a pip-darkness map of the whole die box (0 face .. 1 pip): the
    pips sit on a 3 x 3 grid at +-0.26 of the die size (standard layouts, either diagonal for 2 and
    3); each layout is rendered as Gaussian dots and correlated with the inner part of the map.
    ``(face, confidence)``."""
    hh, ww = dark_map.shape
    if hh < 4 or ww < 4:
        return 0, 0.0
    t = max(1, int(round(0.12 * min(hh, ww))))
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    cy, cx = (hh - 1) / 2.0, (ww - 1) / 2.0
    sig = max(0.55, 0.075 * min(hh, ww))
    inner = (slice(t, hh - t), slice(t, ww - t))
    m = dark_map[inner].astype(np.float64)
    m = m - m.mean()
    mn = float(np.linalg.norm(m))
    if mn < 1e-6:
        return 0, 0.0
    dots: Dict[Tuple[int, int], np.ndarray] = {}
    for gy in (-1, 0, 1):
        for gx in (-1, 0, 1):
            py, px = cy + gy * 0.26 * hh, cx + gx * 0.26 * ww
            dots[(gx, gy)] = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sig * sig))
    scores = []
    for face, alts in _DIE_LAYOUTS.items():
        best = -2.0
        for pips in alts:
            tpl = sum(dots[p] for p in pips)[inner]
            tpl = tpl - tpl.mean()
            c = float((tpl * m).sum()) / (float(np.linalg.norm(tpl)) * mn + 1e-9)
            best = max(best, c)
        scores.append((best, face))
    scores.sort(reverse=True)
    margin = scores[0][0] - scores[1][0]
    conf = float(min(1.0, margin / 0.15)) * (1.0 if scores[0][0] > 0.6 else 0.5)
    return scores[0][1], conf


def _classify_icon(lay: PanelLayout, ic: Icon) -> None:
    """Resource card / card back / development card / die from the icon's pixels."""
    crop = lay.rgb
    x0, x1, y0, y1 = ic.x0, ic.x1, ic.y0, ic.y1
    hh, ww = y1 - y0, x1 - x0
    m = max(1, int(round(0.16 * min(hh, ww))))
    body = crop[y0 + m:y1 - m, x0 + m:x1 - m].astype(np.float32)
    if body.size == 0 or hh < 3 or ww < 3:
        ic.kind, ic.conf = "?", 0.0
        return
    px = body.reshape(-1, 3)
    lum = px.mean(axis=1)
    chroma = px.max(axis=1) - px.min(axis=1)
    bgc = lay.bg_rows[min(lay.bg_rows.shape[0] - 1, (y0 + y1) // 2)]
    near_bg = np.abs(px - bgc).max(axis=1) <= 24
    face_like = ((px.min(axis=1) >= 200) & (chroma <= 40)) | (near_bg & (lum > 150))
    white_frac = float(face_like.mean())
    full = crop[y0:y1, x0:x1].astype(np.float32)
    aspect = ww / max(1.0, float(hh))
    face_lum0 = float(np.median(lum))
    # a die's face is light (white, or the light panel's own colour) with a few dark pips; a card is
    # narrower (0.75) and its fill is a colour
    med = np.median(px, axis=0)
    light_med = float(med.mean()) >= 165.0 and float(med.max() - med.min()) <= 35.0
    is_die = (white_frac >= 0.55 and aspect >= 0.8) or (aspect >= 0.86 and (white_frac >= 0.4 or light_med))
    if is_die:
        # a die (square, light face): dark pips
        fl = full.mean(axis=2)
        t = max(1, int(round(0.12 * min(hh, ww))))
        face = fl[t:hh - t, t:ww - t]
        if face.size == 0:
            ic.kind, ic.conf = "?", 0.0
            return
        face_lum = float(np.percentile(face, 75))
        pip_lum = float(np.percentile(face, 2))
        span = max(25.0, face_lum - pip_lum)
        dark_map = np.clip((face_lum - fl) / span, 0.0, 1.0)
        ic.face, ic.conf = _die_face(dark_map)
        ic.kind = "die"
        ic.fill = (int(face_lum), int(face_lum), int(face_lum))
        return
    body_px = px[~face_like] if (~face_like).sum() >= 3 else px
    if not lay.dark:
        # blur mixes the white pictogram into the fill: keep the darker half
        lb = body_px.mean(axis=1)
        body_px = body_px[lb <= np.percentile(lb, 60)]
    colour = np.median(body_px, axis=0)
    ic.fill = (int(colour[0]), int(colour[1]), int(colour[2]))
    # the face-down card's tan pattern vs the white pictograms of the resource cards
    r_, g_, b_ = px[:, 0], px[:, 1], px[:, 2]
    tan = float(((r_ > 170) & (g_ > 0.7 * r_) & (b_ < 0.62 * r_) & (g_ - b_ > 35)).mean())
    t = max(1, int(round(0.18 * min(hh, ww))))
    ring = np.ones((hh, ww), dtype=bool)
    ring[t:hh - t, t:ww - t] = False
    rp = full[ring]
    teal = float(((rp[:, 1] - rp[:, 0] > 60) & (rp[:, 2] - rp[:, 0] > 50)).mean()) if rp.size else 0.0
    dists = []
    for kind, refs in ICON_REFS.items():
        d = min(float(np.sqrt(((colour - np.asarray(r, dtype=np.float32)) ** 2).sum())) for r in refs)
        if kind == "dev":
            d -= 60.0 * min(1.0, teal / 0.2)
        if kind == "card":
            d -= 50.0 * min(1.0, tan / 0.12)
        elif kind in ("brick", "wood", "ore", "dev"):
            d += 30.0 * min(1.0, tan / 0.12)
        dists.append((d, kind))
    dists.sort()
    ic.kind = dists[0][1]
    margin = dists[1][0] - dists[0][0]
    ic.conf = float(max(0.0, min(1.0, margin / 40.0)) * (1.0 if dists[0][0] < 90 else 0.5))


# ---------------------------------------------------------------------------
# per-band items
# ---------------------------------------------------------------------------
def _trim(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Rows / columns barely covered at the edges of a mask (JPEG ringing, blur tails) trimmed from
    the outside in: ``(r0, r1, c0, c1)`` (a die's outline row is fully covered and stops it)."""
    hh, ww = mask.shape
    rcov, ccov = mask.mean(axis=1), mask.mean(axis=0)
    r0, r1, c0, c1 = 0, hh, 0, ww
    while r0 < r1 - 3 and rcov[r0] < 0.3:
        r0 += 1
    while r1 > r0 + 3 and rcov[r1 - 1] < 0.3:
        r1 -= 1
    while c0 < c1 - 3 and ccov[c0] < 0.3:
        c0 += 1
    while c1 > c0 + 3 and ccov[c1 - 1] < 0.3:
        c1 -= 1
    return r0, r1, c0, c1


def _icon_shape(mask: np.ndarray, span: float, shape_only: bool = False) -> Tuple[bool, bool, bool]:
    """``(is_icon, is_card, is_ring)`` of one (trimmed) blob mask: a (rounded) rectangle - its
    middle rows / columns span nearly the whole box and its rows filled between their outermost
    pixels cover the box (an O / Q of a bold name only ~0.8) - either solid (a card, a die on the
    dark panel) or a ring (a die's outline on the light panel)."""
    hh, ww = mask.shape
    if hh < 4 or ww < 3:
        return False, False, False
    mr = mask[int(0.2 * hh):int(np.ceil(0.8 * hh))]
    mc = mask[:, int(0.2 * ww):int(np.ceil(0.8 * ww))]
    if mr.size == 0 or mc.size == 0:
        return False, False, False
    rs = np.where(mr.any(axis=1), (ww - np.argmax(mr[:, ::-1], axis=1)) - np.argmax(mr, axis=1), 0) / ww
    cs = np.where(mc.any(axis=0), (hh - np.argmax(mc[::-1], axis=0)) - np.argmax(mc, axis=0), 0) / hh
    if not (float((rs >= span).mean()) >= 0.8 and float((cs >= span).mean()) >= 0.8):
        return False, False, False
    anyr = mask.any(axis=1)
    spans = np.where(anyr, ww - np.argmax(mask[:, ::-1], axis=1) - np.argmax(mask, axis=1), 0)
    if float(spans.sum()) / float(hh * ww) < 0.86:
        return False, False, False
    if shape_only:
        return True, False, False
    solid = float(mask.mean())
    ring_b, ring_i = _ring_score(mask)
    is_card = solid >= 0.55
    is_ring = ring_b >= 0.5 and ring_i <= 0.35 and ww >= 0.6 * hh
    return (is_card or is_ring), is_card, is_ring


def _icon_parts(full: np.ndarray, die_like: bool = False) -> List[Tuple[int, int]]:
    """Column ranges of the icons in a blob that may be several glued icons: split at the column
    dips, else by the pitch of a run of cards (0.88 of their height) or dice (1.22)."""
    hh, ww = full.shape
    if ww <= (1.35 if die_like else 1.2 * 0.78) * hh:
        return [(0, ww)]
    prof = full.sum(axis=0).astype(np.float64)
    dips = [x for x in range(2, ww - 2) if prof[x] <= 0.55 * hh and prof[x] <= prof[x - 1]
            and prof[x] <= prof[x + 1] and min(prof[max(0, x - 3):x].max(), prof[x + 1:x + 4].max()) >= 0.8 * hh]
    merged: List[int] = []
    for x in dips:
        if merged and x - merged[-1] <= 2:
            continue
        merged.append(x)
    if merged:
        cand = [0] + merged + [ww]
        widths = np.diff(cand)
        if widths.min() >= 0.55 * hh and widths.max() <= 1.25 * hh:
            return list(zip(cand[:-1], cand[1:]))
    # n icons of equal width w and n - 1 gaps g: (ww + g) = n (w + g); a card is 0.75 of its height
    # and the gap ~0.13, a die is square with a ~0.22 gap.  Equal cuts: the pictograms make deeper
    # column dips than the gaps
    w1, g1 = (1.0 * hh, 0.22 * hh) if die_like else (0.75 * hh, 0.133 * hh)
    n = max(1, int(round((ww + g1) / (w1 + g1))))
    if n == 1:
        return [(0, ww)]
    cuts = [int(round(i * (ww + g1) / n - (g1 / 2 if 0 < i < n else 0))) for i in range(n + 1)]
    cuts[0], cuts[-1] = 0, ww
    return list(zip(cuts[:-1], cuts[1:]))


def _die_like(rgb: Optional[np.ndarray], bgc: Optional[np.ndarray]) -> bool:
    """Whether a blob's box is mostly light / panel-coloured inside (a die's face) rather than a
    card's coloured fill."""
    if rgb is None or rgb.size == 0:
        return False
    hh, ww = rgb.shape[:2]
    m = max(1, int(round(0.2 * min(hh, ww))))
    body = rgb[m:hh - m, m:ww - m].reshape(-1, 3).astype(np.float32)
    if body.size == 0:
        return False
    chroma = body.max(axis=1) - body.min(axis=1)
    light = (body.min(axis=1) >= 195) & (chroma <= 40)
    if bgc is not None:
        light |= np.abs(body - bgc).max(axis=1) <= 22
    return float(light.mean()) >= 0.45


def _find_icons(lab: np.ndarray, cm: _Comps, ids: Sequence[int], ya: int, hmin: float, skip=(),
                span: float = 0.85, rgb: Optional[np.ndarray] = None, bg: Optional[np.ndarray] = None
                ) -> List[Tuple[List[Tuple[int, int, int, int]], List[int], bool]]:
    """Card / die icons among the components ``ids`` of a band label map: ``[(boxes, member ids,
    ring)]`` (a glued run of icons gives several boxes).  ``rgb`` / ``bg``: the band's pixels and
    row backgrounds (tell a run of dice from a run of cards)."""
    out = []
    used = set(skip)
    for k in sorted(ids, key=lambda k: -int(cm.area[k])):
        if k in used:
            continue
        x0, x1, y0, y1 = int(cm.x0[k]), int(cm.x1[k]), int(cm.y0[k]), int(cm.y1[k])
        hh, ww = y1 - y0, x1 - x0
        if hh < hmin or ww < 0.35 * hh:
            continue
        mask = lab[y0 - ya:y1 - ya, x0:x1] == k
        r0, r1, c0, c1 = _trim(mask)
        y0, y1, x0, x1 = y0 + r0, y0 + r1, x0 + c0, x0 + c1
        hh, ww = y1 - y0, x1 - x0
        if hh < hmin or ww < 0.35 * hh:
            continue
        members = [k]
        for j in ids:        # the components inside (pips, marks)
            if j == k or j in used:
                continue
            if cm.x0[j] >= x0 - 1 and cm.x1[j] <= x1 + 1 and cm.y0[j] >= y0 - 1 and cm.y1[j] <= y1 + 1:
                members.append(j)
        mask = lab[y0 - ya:y1 - ya, x0:x1] == k
        full = np.isin(lab[y0 - ya:y1 - ya, x0:x1], members)
        # glued icons all span the full height (most columns of a blob of bold letters only span
        # the x-height)
        colspan = np.where(full.any(axis=0), hh - np.argmax(full[::-1], axis=0) - np.argmax(full, axis=0), 0)
        if float((colspan >= 0.85 * hh).mean()) < 0.7:
            continue
        boxes = []
        any_ring = False
        dl = False
        if rgb is not None:
            dl = _die_like(rgb[y0 - ya:y1 - ya, x0:x1], None if bg is None else bg[(y0 + y1) // 2 - ya])
        for pc0, pc1 in _icon_parts(full, dl):
            sub = mask[:, pc0:pc1]
            cols = np.flatnonzero(sub.any(axis=0))
            if cols.size == 0:
                continue
            q0, q1 = pc0 + int(cols[0]), pc0 + int(cols[-1]) + 1
            sub = mask[:, q0:q1]
            t0, t1, u0, u1 = _trim(sub)
            ok, is_card, is_ring = _icon_shape(sub[t0:t1, u0:u1], span)
            if not ok:
                continue
            any_ring |= is_ring and not is_card
            boxes.append((x0 + q0 + u0, x0 + q0 + u1, y0 + t0, y0 + t1))
        if not boxes:
            continue
        used.update(members)
        out.append((boxes, members, any_ring))
    return out


def _band_items(lay: PanelLayout, band: Band, cm: _Comps, st: np.ndarray, em0: float) -> None:
    """Icons and glyph pieces of one band (fills ``band.items`` in x order; pieces are
    :class:`Word` objects of one piece each until :func:`_group_band`).

    Glyphs are the components of the strong ink (so halos do not glue letters); icons are looked
    for there first and then in the weaker ink (a dark card on the dark panel)."""
    lab = band._lab                                   # type: ignore[attr-defined]
    ya, yb = band._ya, band._yb                       # type: ignore[attr-defined]
    n = st.shape[0]
    ids = [k for k in range(1, n) if cm.area[k] > 0]
    hmin = 0.88 * em0
    if getattr(lay, "_text_only", False):
        hmin = 1e9
    brgb = lay.rgb[ya:yb].astype(np.float32)
    bbg = lay.bg_rows[ya:yb].astype(np.float32)
    found = _find_icons(lab, cm, ids, ya, hmin, rgb=brgb, bg=bbg)
    used = set(j for _, mem, _ in found for j in mem)
    boxes = [bx for bxs, _, _ in found for bx in bxs]
    # weak-ink pass for low-contrast icons
    weak = lay.ink[ya:yb].copy()
    grow = max(1, int(round(0.15 * em0)))
    for (x0, x1, y0, y1) in boxes:
        weak[max(0, y0 - ya - grow):max(0, y1 - ya + grow), max(0, x0 - grow):x1 + grow] = False
    n2, lab2, st2 = _components(weak) if hmin < 1e8 else (0, None, None)
    if n2 > 1:
        cm2 = _measure_components(lay.rgb[ya:yb], lay.bg_rows[ya:yb], lay.dist[ya:yb], lab2, st2)
        cm2.y0 = cm2.y0 + ya
        cm2.y1 = cm2.y1 + ya
        for bxs, _, _ in _find_icons(lab2, cm2, [k for k in range(1, n2) if cm2.area[k] > 0], ya, hmin,
                                     span=0.85, rgb=brgb, bg=bbg):
            for (x0, x1, y0, y1) in bxs:
                if any(min(x1, b[1]) - max(x0, b[0]) > 0.3 * (x1 - x0) for b in boxes):
                    continue
                boxes.append((x0, x1, y0, y1))
                for j in ids:
                    if j in used:
                        continue
                    if cm.x0[j] >= x0 - 1 and cm.x1[j] <= x1 + 1 and cm.y0[j] >= y0 - 1 and cm.y1[j] <= y1 + 1:
                        used.add(j)
    icons = [Icon(x0=x0, x1=x1, y0=y0, y1=y1) for (x0, x1, y0, y1) in boxes]
    for ic in icons:
        _classify_icon(lay, ic)
    # glyph pieces: the remaining components, merged when stacked (i / j dots, colons)
    comps = sorted((k for k in ids if k not in used), key=lambda k: int(cm.x0[k]))
    pieces: List[List[int]] = []
    ext: List[List[int]] = []
    for k in comps:
        x0, x1 = int(cm.x0[k]), int(cm.x1[k])
        if pieces:
            lx0, lx1 = ext[-1]
            ov = min(x1, lx1) - max(x0, lx0)
            if ov > 0 and ov >= 0.5 * min(x1 - x0, lx1 - lx0):
                pieces[-1].append(k)
                ext[-1] = [min(lx0, x0), max(lx1, x1)]
                continue
        pieces.append([k])
        ext.append([x0, x1])
    items: List[Any] = list(icons)
    for pc in pieces:
        x0 = min(int(cm.x0[j]) for j in pc)
        x1 = max(int(cm.x1[j]) for j in pc)
        y0 = min(int(cm.y0[j]) for j in pc)
        y1 = max(int(cm.y1[j]) for j in pc)
        wts = np.array([float(cm.area[j]) for j in pc])
        core = (np.array([cm.core[j] for j in pc]) * wts[:, None]).sum(axis=0) / max(1.0, wts.sum())
        cdiff = (np.array([cm.core_diff[j] for j in pc]) * wts[:, None]).sum(axis=0) / max(1.0, wts.sum())
        maxd = float(max(float(cm.maxd[j]) for j in pc))
        wd = Word(x0=x0, x1=x1, y0=y0, y1=y1, colour=(float(core[0]), float(core[1]), float(core[2])),
                  contrast=maxd, pieces=[(x0, x1)], comp_ids=list(pc))
        wd._area = float(wts.sum())                   # type: ignore[attr-defined]
        wd._diff = cdiff                              # type: ignore[attr-defined]
        items.append(wd)
    items.sort(key=lambda it: it.x0)
    band.items = items
    band.left = min(it.x0 for it in items) if items else 0
    band.right = max(it.x1 for it in items) if items else 0


def _text_profile(lay: PanelLayout, band: Band) -> Tuple[int, np.ndarray]:
    """``(y offset, per-row ink)`` of the band's glyph pieces (icons left out), soft ink."""
    ya, yb = band._ya, band._yb                        # type: ignore[attr-defined]
    lab = band._lab                                    # type: ignore[attr-defined]
    ids = [k for it in band.items if isinstance(it, Word) for k in it.comp_ids]
    if not ids:
        return ya, np.zeros(yb - ya)
    sel = np.isin(lab, ids)
    d = lay.dist[ya:yb] * sel
    c = max(1.0, float(np.percentile(d[sel], 90))) if sel.any() else 1.0
    return ya, np.clip(d / c, 0.0, 1.0).sum(axis=1)


def _band_metrics(lay: PanelLayout) -> None:
    """Per-band baseline and the panel's x-height / cap height from the rows' ink profiles.

    A band's baseline is where its glyph ink drops most sharply going down (the bottom of the
    x-height body; descenders are sparse); the profiles of all bands aligned at their baselines
    give the x-line (the sharpest rise above the baseline) and the cap / ascender top."""
    em = lay.em
    profs = []
    for band in lay.bands:
        ya, prof = _text_profile(lay, band)
        band._prof = (ya, prof)                        # type: ignore[attr-defined]
        if prof.sum() <= 0:
            icons = [it for it in band.items if isinstance(it, Icon)]
            if icons:   # icons are centred on the text: baseline ~ icon centre + 0.36 em
                band.baseline = float(np.median([(it.y0 + it.y1) / 2.0 for it in icons])) + 0.36 * em
            else:
                band.baseline = float(band.y1)
            continue
        # the x-height body is the dense part of the profile; descenders below it are sparse
        nzp = np.sort(prof[prof > 0])
        body = float(np.median(nzp[nzp.size // 2:])) if nzp.size else 0.0
        rows = np.flatnonzero(prof >= 0.45 * body)
        k = int(rows[-1]) if rows.size else int(np.argmax(prof))
        band.baseline = float(ya + k + 1)
        pk = prof / max(1e-6, float(prof.max()))
        profs.append((band.baseline - ya, pk))
    # the panel profile around the baseline
    span_up = int(np.ceil(1.3 * em)) + 2
    span_dn = int(np.ceil(0.6 * em)) + 2
    acc = np.zeros(span_up + span_dn)
    for off, pk in profs:
        o = int(round(off))
        for i in range(-span_up, span_dn):
            y = o + i
            if 0 <= y < pk.size:
                acc[i + span_up] += pk[y]
    if profs and acc.max() > 0:
        acc /= acc.max()
        # x-line: the sharpest rise going down, between 0.3 and 1.0 em above the baseline
        rise = acc[1:] - acc[:-1]                 # rise[j]: acc[j + 1] - acc[j]
        j0 = span_up - int(round(1.0 * em))
        j1 = span_up - max(2, int(round(0.25 * em)))
        if j1 > j0 >= 0:
            j = j0 + int(np.argmax(rise[j0:j1]))
            lay.xh = float(span_up - (j + 1))
        top = np.flatnonzero(acc[:span_up] > 0.08)
        lay._cap = float(span_up - top[0]) if top.size else 1.4 * lay.xh     # type: ignore[attr-defined]
    else:
        lay._cap = 1.4 * lay.xh                    # type: ignore[attr-defined]
    for band in lay.bands:
        band.xline = band.baseline - lay.xh


# ---------------------------------------------------------------------------
# colours: text vs player names
# ---------------------------------------------------------------------------
def name_palette(players: Optional[Dict[str, Sequence[int]]] = None, extra: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, List[Tuple[float, float, float]]]:
    """Colour word -> reference RGBs for the name classifier: ``players`` ({colour: RGB}) when given
    (only those colours), else :data:`NAME_REFS`; ``extra`` (a taught profile's colours) is added."""
    pal: Dict[str, List[Tuple[float, float, float]]] = {}
    if players:
        for col, rgb in players.items():
            if rgb is None:
                continue
            pal.setdefault(str(col).strip().lower(), []).append(tuple(float(v) for v in rgb)[:3])  # type: ignore
    else:
        for col, refs in NAME_REFS.items():
            pal[col] = [tuple(float(v) for v in r) for r in refs]  # type: ignore[misc]
    for col, refs in (extra or {}).items():
        for r in (refs if refs and isinstance(refs[0], (list, tuple)) else [refs]):
            pal.setdefault(str(col).lower(), []).append(tuple(float(v) for v in r)[:3])  # type: ignore
    return pal


def _unmix_residual(d: np.ndarray, bg: np.ndarray, ref: Sequence[float]) -> Tuple[float, float]:
    """How well a core difference ``d`` (pixel - background) is explained by ink of colour ``ref``
    at some coverage: ``(residual, coverage)``."""
    v = np.asarray(ref, dtype=np.float64) - bg
    vv = float(v @ v)
    if vv < 1.0:
        return 1e9, 0.0
    a = float(d @ v) / vv
    a_c = min(1.15, max(0.35, a))
    res = float(np.linalg.norm(d - a_c * v))
    return res, a


def _classify_colours(lay: PanelLayout, palette: Optional[Dict[str, List[Tuple[float, float, float]]]] = None
                      ) -> None:
    """Text vs name (and which colour word) for every glyph piece of the panel.

    The text colour is not assumed: it is the dominant ink direction (the core pixels' difference
    to the background) among the reliable pieces; achromatic pieces along it split into text and a
    white / grey player's name by contrast (the name is the fainter one on a light panel, the
    brighter one on a dark panel).  Other pieces take the nearest palette colour (unmixed from the
    background).  Unreliable (small / faint) pieces follow their close neighbours.
    """
    pal = palette if palette is not None else getattr(lay, "_palette", None) or name_palette()
    pcs: List[Tuple[Band, Word]] = [(b, it) for b in lay.bands for it in b.items if isinstance(it, Word)]
    if not pcs:
        return
    D = np.array([it._diff for _, it in pcs], dtype=np.float64)          # type: ignore[attr-defined]
    area = np.array([it._area for _, it in pcs], dtype=np.float64)       # type: ignore[attr-defined]
    norm = np.linalg.norm(D, axis=1) + 1e-6
    U = D / norm[:, None]
    p90 = float(np.percentile(norm, 90))
    rel = (norm >= 0.4 * p90) & (area >= max(3.0, 0.15 * lay.xh * lay.xh))
    if rel.sum() < 2:
        rel = norm >= 0.4 * p90
    idx = np.flatnonzero(rel)
    # log text is grey (on either theme): its direction is chosen among the near-achromatic pieces
    grey = np.ones(3) / math.sqrt(3.0)
    achro = idx[np.abs(U[idx] @ grey) >= math.cos(math.radians(9.0))]
    if achro.size >= 1:
        idx = achro
    cos6 = math.cos(math.radians(6.0))
    sims = U[idx] @ U[idx].T
    votes = (sims >= cos6).sum(axis=1).astype(np.float64)
    k = idx[int(np.argmax(votes))]
    members = idx[(U[idx] @ U[k]) >= cos6]
    uT = U[members].mean(axis=0)
    uT /= np.linalg.norm(uT)
    lay._uT = uT                                                         # type: ignore[attr-defined]
    # achromatic (text-direction) pieces: the angular tolerance widens for faint pieces.  A white /
    # grey player's name is achromatic too: the decoder tells it from text (a player slot filled by
    # a word that is not "you")
    cosang = U @ uT
    tol = np.radians(np.clip(7.0 + 250.0 / np.maximum(norm, 1.0), 7.0, 25.0))
    along = cosang >= np.cos(tol)
    text_norm = []
    if getattr(lay, "_text_only", False):
        along[:] = True
    else:
        # doubtful pieces (a brown / green name on a light panel is only ~12-16 degrees from grey,
        # JPEG washes colours out): compare the unmixed, chroma-restored ink colour with the text
        # colour and the name palette
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        doubt = (ang >= 3.5) & (ang <= 28.0) & (area >= max(4.0, 0.2 * lay.xh * lay.xh))
        if doubt.any():
            ref_idx = [i for i in members.tolist() if ang[i] < 3.5] or members.tolist()
            bg0 = np.asarray(lay.fill, dtype=np.float64)
            tcol = np.median(np.array([pcs[i][1].colour for i in ref_idx]), axis=0)
            txt_pal = {"text": [tuple(float(v) for v in tcol)]}
            name_pal = {k: v for k, v in pal.items() if k != "white"}
            for i in np.flatnonzero(doubt).tolist():
                band, it = pcs[i]
                est = _token_colour(lay, band, [it])
                bgc = lay.bg_rows[min(lay.bg_rows.shape[0] - 1, (it.y0 + it.y1) // 2)]
                _, _, dT = classify_name_colour(est, txt_pal, bg=bgc)
                _, _, dN = classify_name_colour(est, name_pal, bg=bgc) if name_pal else (None, 0, 1e9)
                along[i] = dT <= dN
    for i, (band, it) in enumerate(pcs):
        if along[i]:
            it.cls, it.name, it.name_conf = "text", None, 0.9
            if rel[i]:
                text_norm.append(norm[i])
        else:
            it.cls, it.name, it.name_conf = "name", "?", 0.0
    lay._text_norm = float(np.median(text_norm)) if text_norm else p90    # type: ignore[attr-defined]
    # unreliable pieces between two name pieces belong to the name ("Mira_7", "k8", an i-dot)
    for band in lay.bands:
        ws = [it for it in band.items if isinstance(it, Word)]
        pos = {id(p): k for k, (_, p) in enumerate(pcs)}
        for j, it in enumerate(ws):
            if rel[pos[id(it)]] or it.cls == "name":
                continue
            if 0 < j < len(ws) - 1 and ws[j - 1].cls == "name" and ws[j + 1].cls == "name" \
                    and it.x0 - ws[j - 1].x1 <= 0.35 * lay.xh and ws[j + 1].x0 - it.x1 <= 0.35 * lay.xh:
                it.cls, it.name = "name", "?"


_YCC = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
_YCC_INV = np.linalg.inv(_YCC)


def _token_colour(lay: PanelLayout, band: Band, pieces: Sequence[Word]) -> Tuple[float, float, float]:
    """The ink colour of a name token, robust to anti-aliasing and JPEG's chroma subsampling.

    Luma is sharp, chroma is blurred but its total is kept: over the token's pixels grown by two
    pixels (other tokens' ink excluded) the chroma difference to the background summed and divided
    by the summed coverage (luma difference / the token's strongest luma difference) gives the
    chroma of full coverage."""
    ya, yb = band._ya, band._yb                        # type: ignore[attr-defined]
    lab = band._lab                                    # type: ignore[attr-defined]
    own_ids = [k for p in pieces for k in p.comp_ids]
    own = np.isin(lab, own_ids)
    x0 = max(0, min(p.x0 for p in pieces) - 3)
    x1 = min(lab.shape[1], max(p.x1 for p in pieces) + 3)
    own = own[:, x0:x1]
    other = (lab[:, x0:x1] > 0) & ~own
    grown = own.copy()
    for _ in range(2):
        g = grown.copy()
        g[1:] |= grown[:-1]
        g[:-1] |= grown[1:]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    region = grown & ~other
    rgb = lay.rgb[ya:yb, x0:x1].astype(np.float64)
    bg = lay.bg_rows[ya:yb].astype(np.float64)[:, None, :]
    d = (rgb - bg) @ _YCC.T                                 # dY, dCb, dCr per pixel
    dy_own = d[..., 0][own]
    if dy_own.size == 0:
        c = np.mean([p.colour for p in pieces], axis=0)
        return (float(c[0]), float(c[1]), float(c[2]))
    sgn = 1.0 if np.median(dy_own) >= 0 else -1.0
    ref = float(np.percentile(sgn * dy_own, 95)) * sgn
    if abs(ref) < 8.0:
        c = np.mean([p.colour for p in pieces], axis=0)
        return (float(c[0]), float(c[1]), float(c[2]))
    alpha = np.clip(d[..., 0] / ref, 0.0, 1.0)[region]
    s = float(alpha.sum())
    dcb = float(d[..., 1][region].sum()) / max(1e-6, s)
    dcr = float(d[..., 2][region].sum()) / max(1e-6, s)
    drgb = _YCC_INV @ np.array([ref, dcb, dcr])
    bgm = lay.bg_rows[(ya + yb) // 2].astype(np.float64)
    col = np.clip(bgm + drgb, 0, 255)
    return (float(col[0]), float(col[1]), float(col[2]))


def _lab(rgb: Sequence[float]) -> np.ndarray:
    """CIE L*a*b* of an sRGB colour (0..255)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ c
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def classify_name_colour(rgb: Sequence[float], palette: Dict[str, List[Tuple[float, float, float]]],
                         bg: Optional[Sequence[float]] = None, exclude_grey: bool = False
                         ) -> Tuple[str, float, float]:
    """Nearest palette colour word of a name colour: ``(word, confidence, distance)``.

    With ``bg`` the colour may be a partial-coverage mix with the background (blur, thin strokes):
    each reference is scaled along its mixing line (coverage 0.45..1.2) before the CIE76 distance.
    """
    lab = _lab(rgb)
    best: List[Tuple[float, str]] = []
    bgv = None if bg is None else np.asarray(bg, dtype=np.float64)
    obs = np.asarray(rgb, dtype=np.float64)
    for col, refs in palette.items():
        if exclude_grey and col == "white":
            continue
        ds = []
        for r in refs:
            rv = np.asarray(r, dtype=np.float64)
            if bgv is not None:
                v = rv - bgv
                vv = float(v @ v)
                a = float((obs - bgv) @ v) / vv if vv > 1.0 else 1.0
                a = min(1.2, max(0.45, a))
                rv = np.clip(bgv + a * v, 0, 255)
            ds.append(float(np.linalg.norm(lab - _lab(rv))))
        best.append((min(ds), col))
    if not best:
        return "?", 0.0, 1e9
    best.sort()
    margin = best[1][0] - best[0][0] if len(best) > 1 else 50.0
    conf = min(1.0, margin / 12.0) * (1.0 if best[0][0] < 25.0 else 0.6)
    return best[0][1], float(conf), float(best[0][0])


# ---------------------------------------------------------------------------
# tokens per band, entries, keys
# ---------------------------------------------------------------------------
@dataclass
class TextRun:
    """Consecutive text-coloured glyph pieces between names / icons (panel-inner coordinates)."""

    x0: int
    x1: int
    y0: int
    y1: int
    pieces: List[Word] = field(default_factory=list)
    gap_before: float = 1e9          # ink gap to the previous token (pixels)


@dataclass
class NameTok:
    """A player name: consecutive pieces of one name colour."""

    x0: int
    x1: int
    y0: int
    y1: int
    colour: str
    conf: float
    pieces: List[Word] = field(default_factory=list)
    gap_before: float = 1e9


def _group_band(lay: PanelLayout, band: Band) -> None:
    """Band items -> tokens: :class:`Icon`, :class:`NameTok` (same-colour pieces, spaces inside a
    name allowed), :class:`TextRun`."""
    toks: List[Any] = []
    for it in band.items:
        gap = (it.x0 - toks[-1].x1) if toks else 1e9
        if isinstance(it, Icon):
            it.gap_before = gap                                          # type: ignore[attr-defined]
            toks.append(it)
            continue
        last = toks[-1] if toks else None
        if it.cls == "name":
            if isinstance(last, NameTok) and gap <= 1.6 * lay.xh:
                last.pieces.append(it)
                last.x1, last.y0, last.y1 = max(last.x1, it.x1), min(last.y0, it.y0), max(last.y1, it.y1)
                continue
            toks.append(NameTok(it.x0, it.x1, it.y0, it.y1, "?", 0.0, [it], gap))
        else:
            if isinstance(last, TextRun):
                last.pieces.append(it)
                last.x1, last.y0, last.y1 = max(last.x1, it.x1), min(last.y0, it.y0), max(last.y1, it.y1)
                continue
            toks.append(TextRun(it.x0, it.x1, it.y0, it.y1, [it], gap))
    # a tiny text-coloured piece between two name pieces ("Mira_7", "k8", a name's dot) is the name's
    merged: List[Any] = []
    for t in toks:
        if (isinstance(t, NameTok) and len(merged) >= 2 and isinstance(merged[-1], TextRun)
                and isinstance(merged[-2], NameTok) and len(merged[-1].pieces) == 1
                and merged[-1].x1 - merged[-1].x0 <= 0.9 * lay.xh
                and merged[-1].x0 - merged[-2].x1 <= 0.35 * lay.xh and t.x0 - merged[-1].x1 <= 0.35 * lay.xh):
            mid = merged.pop()
            nm = merged[-1]
            nm.pieces.extend(mid.pieces + t.pieces)
            nm.x1, nm.y0, nm.y1 = max(nm.x1, t.x1), min(nm.y0, mid.y0, t.y0), max(nm.y1, mid.y1, t.y1)
            continue
        merged.append(t)
    toks = merged
    pal = getattr(lay, "_palette", None) or name_palette()
    for t in toks:
        if isinstance(t, NameTok):
            rgb = _token_colour(lay, band, t.pieces)
            t.rgb = rgb                                                  # type: ignore[attr-defined]
            # a coloured token is never the grey player (achromatic pieces were text)
            bgc = lay.bg_rows[min(lay.bg_rows.shape[0] - 1, (t.y0 + t.y1) // 2)]
            t.colour, t.conf, t.dist = classify_name_colour(rgb, pal, bg=bgc, exclude_grey=True)  # type: ignore
    band.items = toks


def _entry_key(lay: PanelLayout, bands: Sequence[Band]) -> str:
    """Content hash of an entry's normalised ink: coverage (colour distance / the text contrast) in
    4 levels plus the colour of the strong pixels, trimmed to the ink - independent of the vertical
    position and (up to anti-aliasing) of the stripe colour."""
    h = hashlib.blake2b(digest_size=12)
    cmax = max(30.0, float(getattr(lay, "_text_norm", 150.0)))
    for band in bands:
        ya, yb = band._ya, band._yb                    # type: ignore[attr-defined]
        d = lay.dist[ya:yb]
        lev = np.clip(np.floor(d / cmax * 4.0), 0, 4).astype(np.uint8)
        rows = np.flatnonzero(lev.max(axis=1) > 0)
        cols = np.flatnonzero(lev.max(axis=0) > 0)
        if rows.size == 0:
            h.update(b"|empty")
            continue
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        sub = lev[r0:r1, c0:c1]
        rgb = lay.rgb[ya + r0:ya + r1, c0:c1] >> 5
        colour = np.where(sub[:, :, None] >= 3, rgb, 0).astype(np.uint8)
        h.update(np.asarray(sub.shape, dtype=np.int32).tobytes())
        h.update(np.asarray([c0], dtype=np.int32).tobytes() if False else b"")
        h.update(sub.tobytes())
        h.update(colour.tobytes())
        h.update(b"|")
    return h.hexdigest()


def _group_entries(lay: PanelLayout) -> None:
    """Bands -> entries (head row + indented continuation rows; the stripe background changes
    between entries); the top entry is partial when the panel edge cuts it."""
    bands = lay.bands
    if not bands:
        return
    em = lay.em
    lefts = np.array([b.left for b in bands], dtype=np.float64)
    head_x = float(np.percentile(lefts, 10))
    for b in bands:
        b.head = b.left <= head_x + 0.45 * em
    entries: List[Entry] = []
    cur: Optional[Entry] = None
    for b in bands:
        if b.head or cur is None:
            cur = Entry(bands=[b], stripe=b.stripe)
            entries.append(cur)
        else:
            cur.bands.append(b)
    # stripe boundaries give the row top of each head row: baseline offset within a row
    cls = lay.row_cls
    offs = []
    for e in entries[1:]:
        hb = e.bands[0]
        y = hb.y0
        lim = max(0, int(hb.y0 - 0.8 * lay.pitch))
        while y > lim and cls[y - 1] == cls[min(len(cls) - 1, hb.y0)]:
            y -= 1
        if y > lim and cls[y - 1] != cls[min(len(cls) - 1, hb.y0)]:
            offs.append(hb.baseline - y)
    if len(offs) >= 2:
        base_off = float(np.median(offs))
    else:
        base_off = 0.5 * lay.pitch + 0.5 * float(getattr(lay, "_cap", 1.4 * lay.xh))
    lay._base_off = base_off                                          # type: ignore[attr-defined]
    for e in entries:
        e.top = float(e.bands[0].y0)
        e.bottom = float(e.bands[-1].y1)
        e.key = _entry_key(lay, e.bands)
    first = entries[0]
    hb = first.bands[0]
    exp_top = hb.baseline - base_off
    if not hb.head:
        first.partial, first.partial_why = True, "head row missing (first visible row is indented)"
    elif hb.y0 <= 1:
        first.partial, first.partial_why = True, "ink touches the panel's top edge"
    elif exp_top < -0.05 * lay.pitch:
        first.partial, first.partial_why = True, f"row top {exp_top:.1f} px above the panel edge"
    lay.entries = entries
