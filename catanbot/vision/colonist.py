"""Computer-vision parser for Colonist.io screenshots.

Pipeline (see docs/DESIGN.md section 7):

1. Sea mask (HSV blue) -> land mask.
2. Board lattice fit: number tokens (or eroded tile blobs) give candidate
   registrations of the 19-hex lattice (least squares against the known
   hex centres, seeded around the median of the points); among the
   candidates the one whose lattice overlaps the land mask best is kept, so
   a board partly hidden under a panel or cropped is not shifted by a row.
   If neither works a closed land blob + IoU coordinate descent is used and
   verified.
3. Tile resource classification from ring colour statistics with a
   constrained (Hungarian) assignment to the standard 4/3/4/4/3/1 multiset.
4. Number tokens: digit classifier (``catanbot.vision.digits``) on a crop
   around every hex centre, then a constrained assignment to the standard
   multiset of 18 numbers - so a token hidden by the robber still gets the
   only number left.
5. Robber: pawn-sized box filter over the dark, unsaturated mask inside the
   board (thin piece outlines touching the pawn do not matter), falling
   back to a connected-component search.
6. Buildings at ``board.VERTEX_POS`` and roads along ``board.EDGE_POS`` by
   player-colour pixel statistics against a per-image background palette
   (measured sea and tile colours); city vs settlement by probe discs that
   lie inside a city but outside a settlement and the road directions.
   Colours of the whole calibration are rescaled when the number tokens
   show a global brightness change.
7. Ports: beige icons just outside coastal edges, typed by the resource
   square.
8. UI chrome (best effort): player panel rows (colour, VP, cards, dev,
   knights, badges, current-player border), hand bar (my cards), dice
   pips, bank panel.  Everything that cannot be read becomes a warning.
   Each reader searches a default area of the screen (:func:`default_ui_regions`,
   the synthetic layout) or a pixel box given by a *layout* (a
   :class:`~catanbot.vision.profile.UiProfile` measured on the user's screen,
   or a dict of boxes, see :func:`layout_boxes`).  A layout's ``log`` box
   (the game-log panel) is painted with the measured sea colour before the
   board analysis, so its text and icons cannot become land, number tokens,
   ports or pieces; hexes / port slots of the fitted board under it are
   named in a warning, and a log box covering most of the screen is ignored.

The parser never uses ``synth.board_geometry``; it finds the board itself.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import board as B
from ..state import PLAYER_COLORS
from . import schema as S
from .profile import REGION_NAMES, UiProfile
from .result import ParseResult

try:  # optional, used for connected components / morphology / resize
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

RGB = Tuple[int, int, int]
PixelBox = Tuple[int, int, int, int]      # x0, y0, x1, y1 in pixels, end exclusive
SQRT3 = math.sqrt(3.0)
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    """Reference colours and thresholds (defaults match synth + real Colonist)."""

    sea: RGB = (66, 150, 225)
    sea_hue: Tuple[float, float] = (190.0, 235.0)   # degrees
    sea_min_sat: float = 0.35
    sea_min_val: float = 0.35
    tile: Dict[int, RGB] = field(default_factory=lambda: {
        B.WOOD: (40, 110, 50),
        B.SHEEP: (140, 195, 85),
        B.WHEAT: (232, 190, 60),
        B.BRICK: (200, 95, 50),
        B.ORE: (130, 130, 138),
        B.DESERT: (222, 203, 150),
    })
    token: RGB = (242, 228, 196)
    port: RGB = (228, 208, 160)
    robber: RGB = (60, 60, 65)
    players: Dict[str, RGB] = field(default_factory=lambda: {
        "red": (226, 74, 64),
        "blue": (48, 120, 212),
        "orange": (238, 140, 40),
        "green": (60, 160, 80),
        "white": (235, 235, 235),
        "purple": (140, 80, 190),
        "brown": (120, 80, 50),
        "pink": (230, 120, 180),
    })
    dev_card: RGB = (120, 70, 170)    # dev-card icon / card colour in the UI
    panel: RGB = (30, 40, 56)         # dark UI panel colour (hand bar, bank panel)
    white: int = 255                  # level of white UI elements (text, dice, current-player border)
    player_max_dist: float = 60.0      # RGB distance to accept a pixel as a player colour
    building_min_fraction: float = 0.22
    road_min_fraction: float = 0.35
    # Probe offsets (hex sizes, relative to the vertex) that lie >= 0.03 hs inside the city
    # artwork and >= 0.03 hs away from the settlement and from the three road capsules.  "Y"
    # vertices (edges up, down-left, down-right; y = 0.5 mod 1.5 hex sizes) use the first
    # list, inverted-Y vertices the second.  Re-calibrate for other artwork.
    city_probes_y: Tuple[Tuple[float, float], ...] = ((-0.145, -0.16), (-0.15, -0.04))
    city_probes_inv: Tuple[Tuple[float, float], ...] = ((0.15, 0.12), (-0.155, 0.12))


def _copy_calibration(cal: Calibration) -> Calibration:
    return Calibration(**{k: (dict(v) if isinstance(v, dict) else v) for k, v in cal.__dict__.items()})


def _scale_rgb(c: Sequence[float], ratio: float) -> RGB:
    return tuple(int(min(255, max(0, round(v * ratio)))) for v in c)  # type: ignore[return-value]


def _luma(c: Sequence[float]) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def scaled_calibration(cal: Calibration, ratio: float, token: Optional[RGB] = None) -> Calibration:
    """Copy of ``cal`` with every reference colour multiplied by ``ratio`` (a global brightness change)."""
    new = _copy_calibration(cal)
    new.sea = _scale_rgb(cal.sea, ratio)
    new.token = token if token is not None else _scale_rgb(cal.token, ratio)
    new.port = _scale_rgb(cal.port, ratio)
    new.robber = _scale_rgb(cal.robber, ratio)
    new.dev_card = _scale_rgb(cal.dev_card, ratio)
    new.panel = _scale_rgb(cal.panel, ratio)
    new.white = int(min(255, max(0, round(cal.white * ratio))))
    new.tile = {k: _scale_rgb(v, ratio) for k, v in cal.tile.items()}
    new.players = {k: _scale_rgb(v, ratio) for k, v in cal.players.items()}
    return new


# ---------------------------------------------------------------------------
# Small image helpers
# ---------------------------------------------------------------------------
def _to_rgb_array(img: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def rgb_to_hsv(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB (uint8, ...x3) -> (hue degrees, sat 0..1, val 0..1)."""
    a = arr.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    d = mx - mn
    val = mx
    sat = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    hue = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = (60.0 * ((g - b)[rm] / d[rm])) % 360.0
    hue[gm] = 60.0 * ((b - r)[gm] / d[gm]) + 120.0
    hue[bm] = 60.0 * ((r - g)[bm] / d[bm]) + 240.0
    return hue, sat, val


def _components(mask: np.ndarray):
    """Connected components -> (n, labels, stats[x, y, w, h, area], centroids)."""
    m = mask.astype(np.uint8)
    if cv2 is not None:
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        return n, labels, stats, cents
    # numpy fallback (slow-ish, only used without OpenCV)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 1
    stats = [[0, 0, 0, 0, 0]]
    cents = [[0.0, 0.0]]
    ys, xs = np.nonzero(m)
    todo = set(zip(ys.tolist(), xs.tolist()))
    while todo:
        y0, x0 = todo.pop()
        stack = [(y0, x0)]
        pts = []
        labels[y0, x0] = n
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (yy, xx) in todo:
                        todo.discard((yy, xx))
                        labels[yy, xx] = n
                        stack.append((yy, xx))
        py = np.array([p[0] for p in pts])
        px = np.array([p[1] for p in pts])
        stats.append([px.min(), py.min(), px.max() - px.min() + 1, py.max() - py.min() + 1, len(pts)])
        cents.append([px.mean(), py.mean()])
        n += 1
    return n, labels, np.array(stats), np.array(cents)


def _close(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return mask
    if cv2 is not None:
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    # crude fallback: dilate then erode with box filters
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k // 2
    m = np.pad(mask, pad)
    d = sliding_window_view(m, (k, k)).max(axis=(2, 3))
    m2 = np.pad(d, pad, constant_values=True)
    return sliding_window_view(m2, (k, k)).min(axis=(2, 3)).astype(bool)


def _median3(arr: np.ndarray) -> np.ndarray:
    """3x3 median filter of an RGB uint8 image."""
    if cv2 is not None:
        return cv2.medianBlur(arr, 3)
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = arr.shape[:2]
    stack = np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _noise_level(arr: np.ndarray, region: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Median absolute difference between the image and its 3x3 median (over ``region``), and the median image.

    Flat renders and JPEG re-encodes score ~0; Gaussian sensor noise of
    sigma 4 / 8 / 16 scores ~2.7 / 5.7 / 11.
    """
    med = _median3(arr)
    d = np.abs(arr.astype(np.int16) - med.astype(np.int16)).mean(axis=2)
    if region is not None and region.any():
        d = d[region]
    return float(np.median(d)), med


def _resize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(arr).resize(size, Image.BILINEAR))


# ---------------------------------------------------------------------------
# Hungarian assignment (square cost matrix)
# ---------------------------------------------------------------------------
def hungarian(cost: np.ndarray) -> List[int]:
    """Minimum-cost assignment for an n x m (n <= m) cost matrix; returns column for each row."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n > m:
        raise ValueError("hungarian needs n <= m")
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def constrained_assignment(cost: np.ndarray, slot_counts: Sequence[int]) -> List[int]:
    """Assign each row to a class given per-class capacities (min total cost).

    ``cost[i, c]`` is the cost of giving row i class c; ``slot_counts[c]``
    how many rows may get class c (sum must be >= number of rows).
    """
    cols = []
    for c, k in enumerate(slot_counts):
        cols.extend([c] * int(k))
    big = np.asarray(cost)[:, cols]
    assign = hungarian(big)
    return [cols[j] for j in assign]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _lattice_pixels(geom: Dict[str, float], what: str = "hex") -> np.ndarray:
    cx, cy, hs = geom["cx"], geom["cy"], geom["hex_size"]
    if what == "hex":
        pts = np.array(B.HEX_CENTERS)
    elif what == "vertex":
        pts = np.array(B.VERTEX_POS)
    else:
        pts = np.array(B.EDGE_POS)
    return np.stack([cx + pts[:, 0] * hs, cy + pts[:, 1] * hs], axis=1)


def _hex_inside_mask(shape: Tuple[int, int], geom: Dict[str, float], scale: int = 1,
                     shrink: float = 1.0) -> np.ndarray:
    """Boolean mask (possibly downsampled by ``scale``) of pixels inside any of the 19 hexes."""
    h, w = shape
    hh, ww = h // scale, w // scale
    ys, xs = np.mgrid[0:hh, 0:ww]
    xs = xs.astype(np.float32) * scale + scale / 2.0
    ys = ys.astype(np.float32) * scale + scale / 2.0
    hs = geom["hex_size"] * shrink
    centers = _lattice_pixels(geom, "hex")
    inside = np.zeros((hh, ww), dtype=bool)
    for cx, cy in centers:
        dx = np.abs(xs - cx)
        dy = np.abs(ys - cy)
        inside |= (dx <= SQRT3 / 2.0 * hs) & (dy <= hs - dx / SQRT3)
    return inside


def sea_mask(arr: np.ndarray, cal: Calibration, hsv: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
             ) -> np.ndarray:
    """Boolean mask of sea pixels (HSV hue band); ``hsv`` may be passed to reuse a conversion."""
    hue, sat, val = rgb_to_hsv(arr) if hsv is None else hsv
    lo, hi = cal.sea_hue
    m = (hue >= lo) & (hue <= hi) & (sat >= cal.sea_min_sat) & (val >= cal.sea_min_val)
    frac = float(m.mean())
    if frac < 0.05:  # widen if the sea looks different
        m = (hue >= lo - 25) & (hue <= hi + 25) & (sat >= 0.2) & (val >= 0.25)
    return m


def _fit_geometry_from_tiles(cents: np.ndarray, hs0: float, cx0: float, cy0: float,
                             iters: int = 3) -> Tuple[Dict[str, float], float, int]:
    """Least squares (cx, cy, hs) from tile centroids matched to the nearest lattice centre."""
    geom = {"cx": cx0, "cy": cy0, "hex_size": hs0}
    base = np.array(B.HEX_CENTERS)
    matched = 0
    resid = 1e9
    for _ in range(iters):
        lat = _lattice_pixels(geom, "hex")
        d = np.linalg.norm(cents[:, None, :] - lat[None, :, :], axis=2)
        j = d.argmin(axis=1)
        dmin = d[np.arange(len(cents)), j]
        ok = dmin < 0.45 * geom["hex_size"]
        # one centroid per lattice point (closest)
        best: Dict[int, int] = {}
        for i in np.nonzero(ok)[0]:
            jj = int(j[i])
            if jj not in best or dmin[i] < dmin[best[jj]]:
                best[jj] = int(i)
        if len(best) < 4:
            break
        rows_i = np.array(list(best.values()))
        rows_j = np.array(list(best.keys()))
        # pixel = c + base * hs  -> solve for cx, cy, hs
        A = np.zeros((2 * len(rows_i), 3))
        bvec = np.zeros(2 * len(rows_i))
        A[0::2, 0] = 1.0
        A[0::2, 2] = base[rows_j, 0]
        A[1::2, 1] = 1.0
        A[1::2, 2] = base[rows_j, 1]
        bvec[0::2] = cents[rows_i, 0]
        bvec[1::2] = cents[rows_i, 1]
        sol, _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)
        geom = {"cx": float(sol[0]), "cy": float(sol[1]), "hex_size": float(sol[2])}
        matched = len(best)
        pred = A @ sol
        resid = float(np.sqrt(np.mean((pred - bvec) ** 2)))
    return geom, resid, matched


def _land_iou(land: np.ndarray, geom: Dict[str, float], scale: int = 4, shrink: float = 0.96,
              valid: Optional[np.ndarray] = None) -> float:
    """IoU between the (slightly shrunk) 19-hex lattice under ``geom`` and the land mask.

    ``valid`` (optional boolean mask) limits the comparison to the known part of the screen: a
    painted-over log panel is unknown, not sea, and must not favour a lattice that avoids it."""
    small = land[::scale, ::scale]
    m = _hex_inside_mask(land.shape, geom, scale=scale, shrink=shrink)
    m = m[:small.shape[0], :small.shape[1]]
    s = small[:m.shape[0], :m.shape[1]]
    if valid is not None:
        vs = valid[::scale, ::scale][:m.shape[0], :m.shape[1]]
        m, s = m & vs, s & vs
    inter = np.logical_and(m, s).sum()
    union = np.logical_or(m, s).sum()
    return float(inter / max(1, union))


def _iou_refine(land: np.ndarray, geom: Dict[str, float], scale: int = 4) -> Dict[str, float]:
    """Coordinate descent on (cx, cy, hs) maximising IoU between lattice and land mask."""
    best = dict(geom)

    def score(g):
        return _land_iou(land, g, scale=scale, shrink=0.96)

    best_s = score(best)
    for step in (0.06, 0.03, 0.012, 0.005):
        improved = True
        while improved:
            improved = False
            hs = best["hex_size"]
            for dcx, dcy, dhs in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, step), (0, 0, -step)):
                g = {"cx": best["cx"] + dcx * hs, "cy": best["cy"] + dcy * hs, "hex_size": hs * (1 + dhs)}
                s = score(g)
                if s > best_s + 1e-6:
                    best, best_s = g, s
                    improved = True
    best["iou"] = best_s
    return best


def _erode(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return mask
    if cv2 is not None:
        kernel = np.ones((k, k), np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k // 2
    m = np.pad(mask, pad, constant_values=False)
    return sliding_window_view(m, (k, k)).min(axis=(2, 3)).astype(bool)


def _fit_candidates(pts: np.ndarray, min_matched: int) -> List[Tuple[Dict[str, float], float, int]]:
    """Every distinct lattice registration consistent with candidate hex-centre points.

    Returns ``[(geometry, residual, matched)]`` for fits with at least
    ``min_matched`` points matched and a residual below 0.12 hex sizes.
    Seeds are placed on a +-2 hex-size grid around the *median* of the
    points (the mean is biased when a whole column of the board is hidden),
    and fits whose centres agree within 0.2 hex sizes are merged.
    """
    if len(pts) < min_matched:
        return []
    # nearest-neighbour spacing ~ sqrt(3) * hs
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    spacing = float(np.median(np.sort(nn)[: max(3, len(nn) * 2 // 3)]))
    if not np.isfinite(spacing) or spacing <= 0:
        return []
    hs0 = spacing / SQRT3
    core_mask = nn < 2.5 * spacing
    core = pts[core_mask] if int(core_mask.sum()) >= min_matched else pts   # drop isolated blobs
    cx0, cy0 = float(np.median(core[:, 0])), float(np.median(core[:, 1]))
    cands: List[Tuple[Dict[str, float], float, int]] = []
    offs = np.arange(-2.0, 2.01, 0.5)
    for ox in offs:
        for oy in offs:
            g, resid, matched = _fit_geometry_from_tiles(pts, hs0, cx0 + ox * hs0, cy0 + oy * hs0, iters=4)
            if matched < min_matched or g["hex_size"] <= 0 or resid > 0.12 * g["hex_size"]:
                continue
            dup = next((k for k, (c, _, _) in enumerate(cands)
                        if abs(g["cx"] - c["cx"]) < 0.2 * hs0 and abs(g["cy"] - c["cy"]) < 0.2 * hs0), -1)
            if dup >= 0:
                if (matched, -resid) > (cands[dup][2], -cands[dup][1]):
                    cands[dup] = (g, resid, matched)
                continue
            cands.append((g, resid, matched))
    return cands


def _fit_from_points(pts: np.ndarray, min_matched: int) -> Optional[Tuple[Dict[str, float], float, int]]:
    """Best lattice fit (most matched points, then least residual) from candidate hex-centre points."""
    cands = _fit_candidates(pts, min_matched)
    if not cands:
        return None
    g, resid, matched = max(cands, key=lambda c: (c[2], -c[1]))
    return g, resid, matched


def _pick_registration(cands: Sequence[Tuple[Dict[str, float], float, int]], land: np.ndarray,
                       valid: Optional[np.ndarray] = None
                       ) -> Tuple[Dict[str, float], float, int, float, List[Tuple[float, int]]]:
    """Choose among lattice candidates by land-mask overlap.

    Candidates within 3 matches of the best are ranked by the IoU between
    their lattice and the land mask (ties by matches, then residual); the
    translated registrations that match a subset of the visible tokens
    overlap the sea and lose.  ``valid`` restricts the IoU to the known part
    of the screen (see :func:`_land_iou`).  Returns ``(geometry, residual,
    matched, iou, ranking)`` with ``ranking = [(iou, matched), ...]`` for
    debugging.
    """
    best_m = max(m for _, _, m in cands)
    pool = [(g, r, m) for g, r, m in cands if m >= best_m - 3]
    scored = sorted(((_land_iou(land, g, valid=valid), m, -r, g) for g, r, m in pool),
                    key=lambda t: (t[0], t[1], t[2]), reverse=True)
    iou, m, neg_r, g = scored[0]
    return g, -neg_r, m, iou, [(round(float(s), 3), int(mm)) for s, mm, _, _ in scored[:5]]


def _token_points(arr: np.ndarray, cal: Calibration) -> np.ndarray:
    """Centroids of round cream blobs (number tokens)."""
    h, w = arr.shape[:2]
    a = arr.astype(np.int16)
    ref = np.array(cal.token, dtype=np.int16)
    m = np.abs(a - ref).sum(axis=1 + 1) < 75
    n, labels, stats, cents = _components(m)
    pts = []
    areas = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.00015 * h * w or area > 0.02 * h * w:
            continue
        fill = area / float(bw * bh)
        aspect = bw / float(bh)
        # a disc with digits punched out: fill ~0.6-0.8, aspect ~1
        if 0.5 <= fill <= 0.85 and 0.8 <= aspect <= 1.25:
            pts.append(cents[i])
            areas.append(area)
    if len(pts) >= 6:
        med = float(np.median(areas))
        keep = [k for k, a in enumerate(areas) if 0.5 * med <= a <= 1.8 * med]
        pts = [pts[k] for k in keep]
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def find_board(arr: np.ndarray, cal: Calibration, sea: Optional[np.ndarray] = None,
               valid: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], Dict[str, Any], List[str]]:
    """Locate the board: returns (geometry, debug, warnings).

    ``sea`` may be a precomputed :func:`sea_mask`; ``valid`` (optional
    boolean mask) marks the known part of the screen for the land-overlap
    ranking of lattice registrations (a painted-over log panel is unknown).
    ``debug`` carries the method (``tokens`` / ``tiles`` / ``blob``), the
    number of matched points, the land IoU of the chosen registration and
    ``confidence`` (0..1).
    """
    h, w = arr.shape[:2]
    warnings: List[str] = []
    if sea is None:
        sea = sea_mask(arr, cal)
    land = ~sea
    debug: Dict[str, Any] = {"sea_fraction": float(sea.mean())}
    geom = None
    # 1. number tokens
    pts = _token_points(arr, cal)
    cands = _fit_candidates(pts, min_matched=8)
    if cands:
        geom, resid, matched, iou, ranking = _pick_registration(cands, land, valid)
        debug.update({"method": "tokens", "points": int(len(pts)), "matched": matched, "residual": resid,
                      "land_iou": iou, "candidates": ranking, "confidence": float(min(1.0, matched / 14.0))})
        if matched < 12:
            warnings.append(f"board partially hidden ({matched} of 18 number tokens found); "
                            "lattice may be mis-registered - check the debug overlay")
    # 2. eroded tiles (bridges removed) at several scales
    if geom is None:
        img_area = h * w
        for frac in (0.008, 0.012, 0.018, 0.026):
            k = max(3, int(frac * min(h, w)))
            er = _erode(land, k)
            n, labels, stats, cents = _components(er)
            cands_t = []
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < 0.0003 * img_area or area > 0.06 * img_area:
                    continue
                if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                    continue
                fill = area / float(bw * bh)
                aspect = bw / float(bh)
                if 0.6 <= fill <= 0.88 and 0.75 <= aspect <= 1.2:
                    cands_t.append((i, area))
            if len(cands_t) < 8:
                continue
            areas = np.array([c[1] for c in cands_t], dtype=np.float64)
            med = np.median(np.sort(areas)[-19:])
            good = [c for c in cands_t if 0.55 * med <= c[1] <= 1.6 * med]
            cc = np.array([cents[c[0]] for c in good], dtype=np.float64)
            cands = _fit_candidates(cc, min_matched=8)
            if cands:
                geom, resid, matched, iou, ranking = _pick_registration(cands, land, valid)
                debug.update({"method": "tiles", "erode": k, "points": int(len(cc)), "matched": matched,
                              "residual": resid, "land_iou": iou, "candidates": ranking,
                              "confidence": float(min(1.0, matched / 14.0))})
                if matched < 12:
                    warnings.append(f"board partially hidden ({matched} of 19 tiles found); "
                                    "lattice may be mis-registered - check the debug overlay")
                break
    # 3. closed blob + IoU refinement
    if geom is None:
        k = max(3, int(0.006 * max(h, w)))
        closed = _close(land, k)
        n2, labels2, stats2, cents2 = _components(closed)
        best_i, best_score = -1, -1.0
        img_area = h * w
        for i in range(1, n2):
            x, y, bw, bh, area = stats2[i]
            if area < 0.02 * img_area:
                continue
            fill = area / float(bw * bh)
            aspect = bw / float(bh)
            score = area * (1.0 - abs(aspect - 1.08)) * (1.0 - abs(fill - 0.66))
            if 0.7 <= aspect <= 1.5 and fill < 0.9 and score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            raise ValueError("could not locate the board (no sea/land structure found)")
        x, y, bw, bh, area = stats2[best_i]
        # area of the 19-hex board = 19 * 1.5*sqrt(3) hs^2 -> hs (robust to port icons inflating the bbox)
        hs0 = math.sqrt(area / (19 * 1.5 * SQRT3))
        geom = {"cx": x + bw / 2.0, "cy": y + bh / 2.0, "hex_size": float(hs0)}
        geom = _iou_refine(land & (labels2 == best_i), geom)
        iou = float(geom.pop("iou", 0.0))
        debug.update({"method": "blob", "land_iou": iou, "confidence": float(min(1.0, iou))})
        if iou < 0.5:
            warnings.append(f"board not located reliably (blob fallback, land overlap {iou:.2f}); the board below "
                            "is probably wrong - use --state / --fix or the LLM parser")
        else:
            warnings.append("board located by blob fallback (tokens/tiles not found); geometry may be approximate")
    geom["width"] = float(w)
    geom["height"] = float(h)
    debug["geometry"] = dict(geom)
    return geom, debug, warnings


# ---------------------------------------------------------------------------
# Tile classification
# ---------------------------------------------------------------------------
def _ring_pixels(arr: np.ndarray, cx: float, cy: float, r0: float, r1: float) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, x1 = max(0, int(cx - r1)), min(w, int(cx + r1) + 1)
    y0, y1 = max(0, int(cy - r1)), min(h, int(cy + r1) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3), np.uint8)
    sub = arr[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy)
    m = (d >= r0) & (d <= r1)
    return sub[m]


def _colour_features(rgb: np.ndarray) -> np.ndarray:
    """Perceptual-ish features for a colour: (hue cos, hue sin scaled by sat, sat, val)."""
    hue, sat, val = rgb_to_hsv(np.asarray(rgb, dtype=np.uint8).reshape(-1, 3))
    hr = np.radians(hue)
    return np.stack([np.cos(hr) * sat * 1.5, np.sin(hr) * sat * 1.5, sat, val], axis=1)


def _tile_cost(median_rgb: np.ndarray, cal: Calibration) -> np.ndarray:
    """Cost matrix (19 x 6) of assigning each hex to each resource class."""
    refs = np.array([cal.tile[r] for r in range(6)], dtype=np.uint8)
    fr = _colour_features(refs)
    fm = _colour_features(median_rgb)
    d = np.linalg.norm(fm[:, None, :] - fr[None, :, :], axis=2)
    return d


def classify_tiles(arr: np.ndarray, geom: Dict[str, float], cal: Calibration,
                   assume_standard: bool = True) -> Tuple[List[int], List[float], np.ndarray]:
    hs = geom["hex_size"]
    centers = _lattice_pixels(geom, "hex")
    meds = []
    for cx, cy in centers:
        px = _ring_pixels(arr, cx, cy, 0.40 * hs, 0.62 * hs)
        meds.append(np.median(px, axis=0) if len(px) else np.array([0, 0, 0]))
    meds = np.array(meds, dtype=np.float32)
    cost = _tile_cost(meds, cal)
    if assume_standard:
        counts = [B.STANDARD_RESOURCE_COUNTS[r] for r in range(6)]
        assign = constrained_assignment(cost, counts)
    else:
        assign = [int(np.argmin(row)) for row in cost]
    conf = []
    for i, c in enumerate(assign):
        srt = np.sort(cost[i])
        margin = (srt[1] - srt[0]) if len(srt) > 1 else 1.0
        conf.append(float(min(1.0, margin / 0.35)) if cost[i][c] <= srt[0] + 1e-6 else float(min(1.0, 0.5 * margin / 0.35)))
    return assign, conf, meds


# ---------------------------------------------------------------------------
# Number tokens
# ---------------------------------------------------------------------------
_DIGITS = None


def _digit_classifier():
    """The shared :class:`~catanbot.vision.digits.DigitClassifier` (located via ``digits.find_model_path``)."""
    global _DIGITS
    if _DIGITS is None:
        from .digits import DigitClassifier
        _DIGITS = DigitClassifier.load()
    return _DIGITS


def _cream_fraction(arr: np.ndarray, cx: float, cy: float, r: float, cal: Calibration, r0: float = 0.0) -> float:
    """Fraction of token-coloured pixels in the disc (or ring ``r0..r``) around (cx, cy)."""
    px = _ring_pixels(arr, cx, cy, r0, r).astype(np.int16)
    if len(px) == 0:
        return 0.0
    ref = np.array(cal.token, dtype=np.int16)
    d = np.abs(px - ref).sum(axis=1)
    return float((d < 60).mean())


def _token_visibility(arr: np.ndarray, geom: Dict[str, float], cal: Calibration) -> List[float]:
    """Per hex: fraction of token-coloured pixels within 0.28 hex sizes of the centre."""
    hs = geom["hex_size"]
    return [_cream_fraction(arr, cx, cy, 0.28 * hs, cal) for cx, cy in _lattice_pixels(geom, "hex")]


def _token_like(arr: np.ndarray, cx: float, cy: float, hs: float, cal: Calibration) -> bool:
    """A number token is cream inside and tile-coloured around it; a sandy desert is cream everywhere."""
    inner = _cream_fraction(arr, cx, cy, 0.28 * hs, cal)
    ring = _cream_fraction(arr, cx, cy, 0.60 * hs, cal, r0=0.42 * hs)
    return inner > 0.5 and ring < 0.3


def read_numbers(arr: np.ndarray, geom: Dict[str, float], resources: Sequence[int], cal: Calibration,
                 assume_standard: bool = True, robber_hex: int = -1,
                 creams: Optional[Sequence[float]] = None) -> Tuple[List[int], List[float], List[str]]:
    """Number token of every hex (0 = none / unknown), per-hex confidence and warnings.

    With ``assume_standard`` the digits are assigned under the standard
    multiset of 18 numbers, so a single hidden token (e.g. under the robber)
    still gets the only number left.  When two or more tokens are hidden
    (hand bar, crop, robber plus another) their numbers cannot be told apart
    and are reported as 0 with one warning listing the leftover numbers,
    instead of being guessed.  ``creams`` are the per-hex token visibilities
    from :func:`_token_visibility` (computed here when not given).
    """
    from .digits import CLASSES
    hs = geom["hex_size"]
    centers = _lattice_pixels(geom, "hex")
    warnings: List[str] = []
    numbers = [0] * B.NUM_HEXES
    conf = [0.0] * B.NUM_HEXES
    numbered = [i for i in range(B.NUM_HEXES) if resources[i] != B.DESERT]
    if creams is None:
        creams = _token_visibility(arr, geom, cal)
    try:
        clf = _digit_classifier()
    except (FileNotFoundError, OSError, ValueError) as ex:
        warnings.append(f"number tokens not read ({ex}); set them with --fix \"hex N=RESOURCE NUMBER\"")
        return numbers, conf, warnings
    crops = []
    half = int(round(0.42 * hs))
    h, w = arr.shape[:2]
    for cx, cy in centers:
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        x1, y1 = x0 + 2 * half, y0 + 2 * half
        crop = np.zeros((2 * half, 2 * half, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
        crops.append(crop)
    probs = np.asarray(clf.predict_proba(crops), dtype=np.float64)
    probs = np.clip(probs, 1e-6, 1.0)
    # A hex whose token is hidden (robber on it, UI panel, crop) gets a flat distribution.
    hidden: List[int] = []
    for i in range(B.NUM_HEXES):
        if creams[i] < 0.25 and resources[i] != B.DESERT:
            probs[i] = np.full(len(CLASSES), 1.0 / len(CLASSES))
            hidden.append(i)
            if i != robber_hex:
                warnings.append(f"hex {i}: number token not clearly visible")
        if resources[i] == B.DESERT and _token_like(arr, centers[i][0], centers[i][1], hs, cal):
            warnings.append(f"hex {i}: looks like it has a number token but was classified as desert")
    if assume_standard and len(numbered) == 18:
        from collections import Counter
        std = Counter(B.STANDARD_NUMBERS)
        counts = [std[c] for c in CLASSES]
        cost = -np.log(probs[numbered])
        assign = constrained_assignment(cost, counts)
        for k, i in enumerate(numbered):
            numbers[i] = CLASSES[assign[k]]
            conf[i] = float(probs[i][assign[k]])
        if len(hidden) == 1:
            conf[hidden[0]] = 0.9      # the unique number left over from the other 17 tokens
        elif len(hidden) >= 2:
            leftover = sorted(numbers[i] for i in hidden)
            for i in hidden:
                numbers[i] = 0
                conf[i] = 0.0
            warnings.append(f"hexes {', '.join(str(i) for i in hidden)} hidden; their numbers are {leftover} in "
                            "unknown order - set them with --fix \"hex N=RESOURCE NUMBER\"")
    else:
        for i in numbered:
            if i in hidden:
                continue
            k = int(np.argmax(probs[i]))
            numbers[i] = CLASSES[k]
            conf[i] = float(probs[i][k])
        if hidden:
            warnings.append(f"hexes {', '.join(str(i) for i in hidden)} hidden; numbers unknown - set them with "
                            "--fix \"hex N=RESOURCE NUMBER\"")
        if assume_standard:
            warnings.append("non-standard number of desert hexes; numbers read independently")
    return numbers, conf, warnings


# ---------------------------------------------------------------------------
# Robber
# ---------------------------------------------------------------------------
def _box_mean(a: np.ndarray, kw: int, kh: int) -> np.ndarray:
    """Mean of ``a`` over a ``kw x kh`` window centred on every pixel (float32 in, float32 out)."""
    if cv2 is not None:
        return cv2.boxFilter(a, -1, (kw, kh), normalize=True)
    # integral-image fallback (zero padding at the border)
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1), np.float64)
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - kh // 2, 0, h)
    y1 = np.clip(ys - kh // 2 + kh, 0, h)
    x0 = np.clip(xs - kw // 2, 0, w)
    x1 = np.clip(xs - kw // 2 + kw, 0, w)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    return (s / float(kw * kh)).astype(np.float32)


def find_robber(arr: np.ndarray, geom: Dict[str, float], cal: Calibration) -> Tuple[int, float]:
    """Hex holding the robber pawn (-1 if none found) and a confidence 0..1.

    Works on the board's bounding box only.  A pawn-sized (0.20 x 0.50 hex
    sizes) box filter over the dark, unsaturated mask scores every hex by
    its densest dark patch; the pawn scores 0.6-0.95 (clean to heavy JPEG)
    while dark tile art stays below ~0.5.  Unlike a connected-component search this is immune to
    the thin piece outlines that touch the pawn on crowded hexes.  The
    component search is kept as a fallback for weak scores.
    """
    hs = geom["hex_size"]
    h, w = arr.shape[:2]
    cx, cy = geom["cx"], geom["cy"]
    x0, x1 = max(0, int(cx - 4.6 * hs)), min(w, int(cx + 4.6 * hs))
    y0, y1 = max(0, int(cy - 4.3 * hs)), min(h, int(cy + 4.3 * hs))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return -1, 0.0
    sub = arr[y0:y1, x0:x1]
    hue, sat, val = rgb_to_hsv(sub)
    g2 = {"cx": cx - x0, "cy": cy - y0, "hex_size": hs}
    inside_small = _hex_inside_mask(sub.shape[:2], g2, scale=2, shrink=0.94)
    inside = np.repeat(np.repeat(inside_small, 2, axis=0), 2, axis=1)[: sub.shape[0], : sub.shape[1]]
    if inside.shape != sub.shape[:2]:   # odd sizes: pad the last row / column
        full = np.zeros(sub.shape[:2], dtype=bool)
        full[: inside.shape[0], : inside.shape[1]] = inside
        inside = full
    centers = _lattice_pixels(g2, "hex")
    # 1. box-filter score (the value threshold follows the image brightness: cal.white)
    dark = ((val < 0.36 * cal.white / 255.0) & (sat < 0.40)).astype(np.float32)
    kw, kh = max(3, int(round(0.20 * hs))), max(3, int(round(0.50 * hs)))
    box = _box_mean(dark, kw, kh)
    box[~inside] = 0.0
    best_h, best_s = -1, 0.0
    for i, (hx, hy) in enumerate(centers):
        yy0, yy1 = max(0, int(hy - 0.9 * hs)), min(box.shape[0], int(hy + 0.9 * hs))
        xx0, xx1 = max(0, int(hx - 0.9 * hs)), min(box.shape[1], int(hx + 0.9 * hs))
        if yy1 <= yy0 or xx1 <= xx0:
            continue
        m = float(box[yy0:yy1, xx0:xx1].max())
        if m > best_s:
            best_h, best_s = i, m
    if best_s >= 0.55:
        return best_h, float(min(1.0, (best_s - 0.55) / 0.35))
    # 2. fallback: dark connected component of pawn size near a hex centre
    dark2 = (val < 0.42) & (sat < 0.30) & inside
    n, labels, stats, cents = _components(dark2)
    best_h, best_score = -1, 0.0
    amin, amax = 0.06 * hs * hs, 0.30 * hs * hs
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < amin or area > amax:
            continue
        if bw > 0.7 * hs or bh > 0.8 * hs:
            continue
        bx, by = cents[i]
        d = np.hypot(centers[:, 0] - bx, centers[:, 1] - by)
        hidx = int(d.argmin())
        if d[hidx] > 0.8 * hs:
            continue
        # pawn-ish: taller than wide, decent fill
        fill = area / float(bw * bh)
        score = area / (hs * hs) * (1.0 + 0.5 * (bh >= bw)) * (0.5 + fill)
        if score > best_score:
            best_score, best_h = score, hidx
    conf = min(1.0, best_score / 0.12) if best_h >= 0 else 0.0
    return best_h, conf


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------
def _player_palette(cal: Calibration) -> Tuple[List[str], np.ndarray]:
    names = list(cal.players.keys())
    return names, np.array([cal.players[k] for k in names], dtype=np.float32)


def _background_palette(cal: Calibration) -> np.ndarray:
    """Colours a piece pixel must be farther from than from its player colour (sea, tokens, UI, tiles)."""
    cols = [cal.sea, cal.token, cal.port, cal.robber, (35, 35, 35), (cal.white,) * 3, cal.dev_card, cal.panel]
    cols += [cal.tile[r] for r in range(6)]
    # shaded tile-art colours
    for r in range(6):
        c = cal.tile[r]
        cols.append((int(c[0] * 0.6), int(c[1] * 0.6), int(c[2] * 0.6)))
    return np.array(cols, dtype=np.float32)


def _classify_pixels(px: np.ndarray, pal: np.ndarray, bg: np.ndarray, max_dist: float) -> np.ndarray:
    """Return index into ``pal`` for each pixel, or -1 if closer to a background colour / too far.

    Squared distances are computed as ``|p|^2 + |r|^2 - 2 p.r`` with one
    matrix product, so only an ``N x K`` array is allocated (the naive
    ``N x K x 3`` differences cost ~800 MB on a 2560x1440 panel region).
    """
    if len(px) == 0:
        return np.zeros(0, dtype=np.int64)
    p = np.asarray(px, dtype=np.float32).reshape(-1, 3)
    pal = np.asarray(pal, dtype=np.float32).reshape(-1, 3)
    bg = np.asarray(bg, dtype=np.float32).reshape(-1, 3)
    refs = np.concatenate([pal, bg], axis=0)
    d2 = (p * p).sum(axis=1)[:, None] + (refs * refs).sum(axis=1)[None, :] - 2.0 * (p @ refs.T)
    k = len(pal)
    dp = d2[:, :k]
    j = dp.argmin(axis=1)
    dmin = dp[np.arange(len(p)), j]
    db = d2[:, k:].min(axis=1) if len(bg) else np.full(len(p), np.inf, dtype=np.float32)
    ok = (dmin < max_dist * max_dist) & (dmin < db)
    return np.where(ok, j, -1)


def _piece_palettes(cals: Sequence[Calibration], extra_bg: Sequence[Sequence[float]] = ()
                    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """``(names, player_palette, background_palette)`` for piece / panel classification.

    Several calibrations (e.g. the reference one and a brightness-scaled
    copy) are concatenated, so ``names`` may repeat; ``extra_bg`` are
    per-image background colours (the measured sea and tile medians) that
    keep a differently shaded sea or tile from passing as a player colour.
    """
    names: List[str] = []
    pals: List[np.ndarray] = []
    bgs: List[np.ndarray] = []
    for c in cals:
        n, p = _player_palette(c)
        names.extend(n)
        pals.append(p)
        bgs.append(_background_palette(c))
    if len(extra_bg):
        bgs.append(np.asarray(extra_bg, dtype=np.float32).reshape(-1, 3))
    return names, np.concatenate(pals, axis=0), np.concatenate(bgs, axis=0)


def _is_y_vertex(v: int) -> bool:
    """"Y" vertex (edges up, down-left, down-right) iff its y is 0.5 mod 1.5 hex sizes."""
    return abs((B.VERTEX_POS[v][1] % 1.5) - 0.5) < 0.1


def detect_pieces(arr: np.ndarray, geom: Dict[str, float], cal: Calibration,
                  known_colors: Optional[Sequence[str]] = None,
                  palette: Optional[Tuple[List[str], np.ndarray, np.ndarray]] = None,
                  details: Optional[Dict[str, Any]] = None):
    """Buildings per vertex and roads per edge.

    Returns ``(buildings, roads, conf)`` with ``buildings = {vertex: (colour, is_city, fraction)}``
    and ``roads = {edge: (colour, fraction)}``.  ``known_colors`` restricts the
    player colours (normally left ``None``: a panel row that was missed must
    not delete a player's pieces), ``palette`` is a precomputed
    :func:`_piece_palettes` result, and ``details`` (a dict) receives
    ``city_hits`` = per-vertex maximum probe hit (the city evidence).
    """
    hs = geom["hex_size"]
    if palette is None:
        names, pal, bg = _piece_palettes([cal])
    else:
        names, pal, bg = palette
        names = list(names)
    if known_colors:
        keep = [i for i, nme in enumerate(names) if nme in known_colors]
        if keep:
            names = [names[i] for i in keep]
            pal = pal[keep]
    uniq = list(dict.fromkeys(names))
    uidx = np.array([uniq.index(n) for n in names], dtype=np.int64)

    def colour_counts(cls: np.ndarray) -> np.ndarray:
        c = cls[cls >= 0]
        if len(c) == 0:
            return np.zeros(len(uniq), dtype=np.int64)
        return np.bincount(uidx[c], minlength=len(uniq))

    h, w = arr.shape[:2]
    buildings: Dict[int, Tuple[str, bool, float]] = {}
    roads: Dict[int, Tuple[str, float]] = {}
    city_hits: Dict[int, float] = {}
    fracs = []
    vpix = _lattice_pixels(geom, "vertex")
    for v, (vx, vy) in enumerate(vpix):
        px = _ring_pixels(arr, vx, vy, 0.0, 0.16 * hs)
        cls = _classify_pixels(px, pal, bg, cal.player_max_dist)
        if len(cls) == 0:
            continue
        counts = colour_counts(cls)
        frac = counts.max() / len(cls)
        if frac < cal.building_min_fraction:
            continue
        c = int(counts.argmax())
        # City vs settlement: probe small discs that lie inside the city artwork but outside a
        # settlement and off the three road directions; coloured for a city only.  The probe
        # set depends on the vertex orientation (Y / inverted Y).
        probes = cal.city_probes_y if _is_y_vertex(v) else cal.city_probes_inv
        hits = []
        for ox, oy in probes:
            px2 = _ring_pixels(arr, vx + ox * hs, vy + oy * hs, 0.0, 0.03 * hs)
            cls2 = _classify_pixels(px2, pal, bg, cal.player_max_dist)
            hits.append(float((uidx[cls2[cls2 >= 0]] == c).sum() / len(cls2)) if len(cls2) else 0.0)
        best_hit = max(hits) if hits else 0.0
        is_city = best_hit >= 0.6
        buildings[v] = (uniq[c], bool(is_city), float(frac))
        city_hits[v] = float(best_hit)
        fracs.append(frac)
    for e, (a, b) in enumerate(B.EDGE_VERTICES):
        (x1, y1), (x2, y2) = vpix[[a, b]]
        # sample the middle 50 % of the edge, half-width 0.05 hs
        ts = np.linspace(0.3, 0.7, 9)
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        pts = []
        for t in ts:
            for s in (-0.05 * hs, 0.0, 0.05 * hs):
                pts.append((x1 + dx * t + nx * s, y1 + dy * t + ny * s))
        pts = np.array(pts)
        # samples outside the image are skipped (clamping them to the border would read the UI)
        ok = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
        if ok.sum() < len(pts) // 2:
            continue
        xs = np.clip(np.round(pts[ok, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[ok, 1]).astype(int), 0, h - 1)
        px = arr[ys, xs]
        cls = _classify_pixels(px, pal, bg, cal.player_max_dist)
        counts = colour_counts(cls)
        frac = counts.max() / len(cls)
        if frac >= cal.road_min_fraction:
            roads[e] = (uniq[int(counts.argmax())], float(frac))
            fracs.append(frac)
    conf = float(np.mean([min(1.0, (f - 0.2) / 0.5) for f in fracs])) if fracs else 1.0
    if details is not None:
        details["city_hits"] = city_hits
    return buildings, roads, conf


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
def _port_slot_points(geom: Dict[str, float]) -> List[Tuple[int, float, float]]:
    """``(edge, x, y)`` for every coastal edge: where its port icon sits (0.62 hex sizes out from the
    edge midpoint, away from the land hex).  :func:`detect_ports` looks for an icon at each point."""
    hs = geom["hex_size"]
    out: List[Tuple[int, float, float]] = []
    for e in B.COASTAL_EDGES:
        hx, hy = B.HEX_CENTERS[B.EDGE_HEXES[e][0]]
        ex, ey = B.EDGE_POS[e]
        dx, dy = ex - hx, ey - hy
        n = math.hypot(dx, dy) or 1.0
        dx, dy = dx / n, dy / n
        mx, my = geom["cx"] + ex * hs, geom["cy"] + ey * hs
        out.append((e, mx + dx * 0.62 * hs, my + dy * 0.62 * hs))
    return out


def detect_ports(arr: np.ndarray, geom: Dict[str, float], cal: Calibration) -> Tuple[List[Dict[str, Any]], float, List[str]]:
    hs = geom["hex_size"]
    h, w = arr.shape[:2]
    ports = []
    warnings: List[str] = []
    ref = np.array(cal.port, dtype=np.int16)
    tile_refs = np.array([cal.tile[r] for r in range(5)], dtype=np.float32)
    scores = []
    for e, cx, cy in _port_slot_points(geom):
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        samples = []
        for ox, oy, rr in ((-0.30, 0.0, 0.05), (0.0, -0.16, 0.04), (0.0, 0.16, 0.04)):
            px = _ring_pixels(arr, cx + ox * hs, cy + oy * hs, 0.0, rr * hs).astype(np.int16)
            if len(px) == 0:
                samples.append(0.0)
            else:
                samples.append(float((np.abs(px - ref).sum(axis=1) < 80).mean()))
        beige = float(np.mean(samples))
        if beige < 0.5 or min(samples) < 0.2:
            continue
        # resource square at cx + 0.22 hs (icon drawn along the image x axis in synth; Colonist similar)
        sq = _ring_pixels(arr, cx + 0.22 * hs, cy, 0.0, 0.08 * hs).astype(np.float32)
        ptype = B.PORT_GENERIC
        if len(sq):
            med = np.median(sq, axis=0)
            d = np.linalg.norm(tile_refs - med[None, :], axis=1)
            dbeige = np.linalg.norm(med - ref.astype(np.float32))
            if d.min() < 55 and d.min() < dbeige:
                ptype = int(d.argmin())
        ports.append({"edge": int(e), "type": B.PORT_NAMES[ptype], "_score": beige, "_xy": (cx, cy)})
        scores.append(beige)
    # de-duplicate: two detections closer than 0.8 hs are the same icon
    ports.sort(key=lambda d: -d["_score"])
    kept: List[Dict[str, Any]] = []
    for d in ports:
        if all(math.hypot(d["_xy"][0] - k["_xy"][0], d["_xy"][1] - k["_xy"][1]) > 0.8 * hs for k in kept):
            kept.append(d)
    ports = [{"edge": d["edge"], "type": d["type"]} for d in sorted(kept, key=lambda d: d["edge"])]
    scores = [d["_score"] for d in kept]
    if len(ports) != 9:
        msg = f"detected {len(ports)} ports (expected 9)"
        if len(ports) < 9:
            msg += "; ports under UI panels (e.g. the hand bar at the bottom) cannot be seen - add them with --fix 'port EDGE=TYPE'"
        warnings.append(msg)
    conf = float(np.mean(scores)) if scores else 0.0
    return ports, conf, warnings


# ---------------------------------------------------------------------------
# Tiny digit OCR for UI numbers (0-9), template matching against DejaVu Bold
# ---------------------------------------------------------------------------
_TEMPLATES: Optional[np.ndarray] = None
_TW, _TH = 14, 20


def _norm_glyph(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((_TH, _TW), np.float32)
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    hh, ww = sub.shape
    scale = min(_TH / hh, _TW / ww)
    nh, nw = max(1, int(round(hh * scale))), max(1, int(round(ww * scale)))
    img = Image.fromarray((sub * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((_TH, _TW), np.float32)
    y0, x0 = (_TH - nh) // 2, (_TW - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = np.asarray(img, dtype=np.float32) / 255.0
    return out


_TEMPLATE_SIZES = (10, 12, 14, 20, 32, 64)


def _templates() -> np.ndarray:
    """``(10, len(_TEMPLATE_SIZES), _TH, _TW)``: every digit rendered at several pixel sizes.

    UI numbers can be tiny (a 9 px glyph in a small screenshot); a template
    rasterised at a similar size, with the same hinting / thresholding
    artefacts, matches it far better than one downsampled from 64 px
    (which confuses 9 with 0 at that scale).
    """
    global _TEMPLATES
    if _TEMPLATES is None:
        temps = []
        for d in range(10):
            per_size = []
            for px in _TEMPLATE_SIZES:
                try:
                    font = ImageFont.truetype(DEFAULT_FONT_PATH, px)
                except Exception:
                    font = ImageFont.load_default()
                img = Image.new("L", (80, 90), 0)
                ImageDraw.Draw(img).text((10, 5), str(d), fill=255, font=font)
                per_size.append(_norm_glyph(np.asarray(img) > 128))
            temps.append(np.stack(per_size))
        _TEMPLATES = np.stack(temps)
    return _TEMPLATES


_LETTER_TEMPLATES: Optional[Dict[str, np.ndarray]] = None


def _letter_templates() -> Dict[str, np.ndarray]:
    global _LETTER_TEMPLATES
    if _LETTER_TEMPLATES is None:
        try:
            font = ImageFont.truetype(DEFAULT_FONT_PATH, 64)
        except Exception:
            font = ImageFont.load_default()
        out = {}
        for ch in "AR":
            img = Image.new("L", (80, 90), 0)
            ImageDraw.Draw(img).text((10, 5), ch, fill=255, font=font)
            out[ch] = _norm_glyph(np.asarray(img) > 128)
        _LETTER_TEMPLATES = out
    return _LETTER_TEMPLATES


def _match_letter(mask: np.ndarray) -> str:
    g = _norm_glyph(mask)
    gz = g - g.mean()
    best, best_s = "?", -2.0
    for ch, t in _letter_templates().items():
        tz = t - t.mean()
        denom = math.sqrt((gz ** 2).sum() * (tz ** 2).sum()) or 1.0
        sc = float((gz * tz).sum() / denom)
        if sc > best_s:
            best, best_s = ch, sc
    return best


def _match_digit(mask: np.ndarray) -> Tuple[int, float]:
    g = _norm_glyph(mask)
    t = _templates()
    gz = g - g.mean()
    scores = []
    for k in range(10):
        best = -1.0
        for tk in t[k]:   # best over the template sizes
            tz = tk - tk.mean()
            denom = math.sqrt((gz ** 2).sum() * (tz ** 2).sum()) or 1.0
            best = max(best, float((gz * tz).sum() / denom))
        scores.append(best)
    k = int(np.argmax(scores))
    return k, scores[k]


def read_number_in_region(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, light_text: bool = True,
                          min_h: int = 6, bg_rgb: Optional[Sequence[float]] = None,
                          white: int = 255) -> Tuple[Optional[int], float]:
    """OCR a (possibly multi-digit) number made of light (or dark) glyphs inside a region.

    With ``bg_rgb`` (the colour behind the text) the glyph mask is *relative*:
    a pixel is text when it is much closer to white (black) than to the
    background, gated by a loose absolute level.  JPEG chroma bleeding and
    blur pull the glyph edges towards the background (e.g. orange), which
    fragments an absolute ``min(RGB) > 205`` mask into wrong digits.
    """
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0
    sub = arr[y0:y1, x0:x1].astype(np.int16)
    white = int(white)
    if bg_rgb is None:
        mask = sub.min(axis=2) > int(0.80 * white) if light_text else sub.max(axis=2) < 70
    else:
        bgc = np.asarray(bg_rgb, dtype=np.int16).reshape(3)
        if light_text:
            # anything at least as bright as the white level counts as white
            d_t = np.abs(np.minimum(sub, white) - white).sum(axis=2)
        else:
            d_t = np.abs(sub).sum(axis=2)
        d_b = np.abs(sub - bgc).sum(axis=2)
        # loose absolute gate: keeps coloured icons (the yellow VP star) out of the text mask
        gate = (sub.min(axis=2) > int(0.55 * white)) if light_text else (sub.max(axis=2) < 110)
        mask = (d_t < 0.6 * d_b) & (d_t < 300) & gate
    n, labels, stats, cents = _components(mask)
    glyphs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < min_h or area < 4:
            continue
        fill = area / float(bw * bh)
        if fill > 0.9 and bw > 0.5 * bh:   # solid icon rectangles
            continue
        if bw > 1.3 * bh:                  # too wide for a glyph
            continue
        glyphs.append((x, y, bw, bh, i))
    if not glyphs:
        return None, 0.0
    glyphs.sort(key=lambda g: g[0])
    # group into numbers by horizontal gap
    groups: List[List[Tuple[int, int, int, int, int]]] = []
    for g in glyphs:
        if groups and g[0] - (groups[-1][-1][0] + groups[-1][-1][2]) < 0.6 * g[3]:
            groups[-1].append(g)
        else:
            groups.append([g])
    # take the first group (left-most number)
    grp = groups[0]
    digits = []
    confs = []
    for x, y, bw, bh, i in grp:
        m = labels[y:y + bh, x:x + bw] == i
        d, c = _match_digit(m)
        digits.append(str(d))
        confs.append(c)
    try:
        return int("".join(digits)), float(np.mean(confs))
    except ValueError:
        return None, 0.0


# ---------------------------------------------------------------------------
# UI chrome
# ---------------------------------------------------------------------------
# Not-found warnings: every reader words its own "not found" warning, saying where it looked
# (its default area, "in region (x0, y0, x1, y1)" or "region ... lies outside the WxH image").
# read_player_panel returns its warnings (third item of its result); read_hand_bar, read_dice and
# read_bank_panel keep their return values and append to an optional ``warnings`` list instead.
# The player panel and the hand bar are always reported (the parse depends on them).  The dice and
# the bank are optional panels, so a miss in their default area stays silent; a miss inside a
# given region is reported (a region says the panel is there).  parse_image forwards all of them.
def default_ui_regions(size: Tuple[int, int]) -> Dict[str, PixelBox]:
    """Where the UI readers search when no region is given: pixel boxes ``(x0, y0, x1, y1)``.

    They follow the synthetic layout (:mod:`catanbot.vision.synth`): player
    panel in the left 34 % of the screen, hand bar in the bottom 20 %, dice
    in the bottom-right corner, bank panel in the top-right corner.  A real
    Colonist.io window needs its own boxes (a :class:`~catanbot.vision.profile.UiProfile`).
    """
    w, h = size
    return {"player_panel": (0, 0, int(0.34 * w), h),
            "hand_bar": (0, int(0.80 * h), w, h),
            "dice": (int(0.7 * w), int(0.8 * h), w, h),
            "bank": (int(0.6 * w), 0, w, int(0.15 * h))}


#: A ``log`` box larger than this fraction of the screen is ignored (with a warning): the log panel is a
#: column beside the board, and painting most of the screen as sea would hide the board itself.
MAX_LOG_BOX_FRACTION = 0.45


def _box_numbers(label: str, box: Any) -> List[float]:
    """The four numbers of a pixel box ``(x0, y0, x1, y1)``; ``ValueError`` naming ``label`` otherwise."""
    if isinstance(box, (str, bytes)):
        raise ValueError(f"{label}: got the text {box!r}; a box typed as text (\"x,y,w,h\" pixels or "
                         "\"x0,y0,x1,y1\" fractions) goes through profile.parse_box(text, size) and "
                         "UiProfile.set_region, and the profile is passed as the layout")
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        raise ValueError(f"{label}: expected 4 numbers x0,y0,x1,y1 in pixels, got {box!r}") from None
    if len(vals) != 4:
        raise ValueError(f"{label}: expected 4 numbers x0,y0,x1,y1 in pixels, got {box!r}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{label}: expected 4 finite numbers x0,y0,x1,y1 in pixels, got {box!r}")
    return vals


def _clip_box(box: Sequence[float], w: int, h: int) -> Optional[PixelBox]:
    """``box`` (finite numbers) rounded to whole pixels and clipped to a ``w x h`` image; ``None`` when
    nothing is left."""
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _reader_box(arr: np.ndarray, name: str, region: Optional[Sequence[float]]) -> Optional[PixelBox]:
    """The box a UI reader searches: ``region`` clipped to the image (``None`` when it lies outside),
    or the default area ``name``.  A malformed or non-finite region raises ``ValueError``."""
    h, w = arr.shape[:2]
    if region is None:
        return default_ui_regions((w, h))[name]
    return _clip_box(_box_numbers(f"{name} region", region), w, h)


def _where(arr: np.ndarray, region: Optional[Sequence[float]], box: Optional[PixelBox]) -> str:
    """Where a UI reader looked, for its not-found warning: ``""`` (its default area),
    ``" in region (x0, y0, x1, y1)"`` or why the region could not be searched."""
    if region is None:
        return ""
    if box is None:
        h, w = arr.shape[:2]
        return f": region ({', '.join(f'{float(v):g}' for v in region)}) lies outside the {w}x{h} image"
    return f" in region {box}"


def layout_boxes(layout: Union[UiProfile, Mapping, None], size: Tuple[int, int]) -> Dict[str, PixelBox]:
    """Pixel boxes ``{region name: (x0, y0, x1, y1)}`` (end exclusive, clipped to the image) of a layout.

    ``layout`` is one of

    * ``None`` - no regions: every reader uses its default area;
    * a :class:`~catanbot.vision.profile.UiProfile` - fraction regions,
      converted with ``pixel_box(name, size)``;
    * a dict of **pixel** boxes ``(x0, y0, x1, y1)`` - two corners, end
      exclusive - keyed by region name (``log``, ``player_panel``,
      ``hand_bar``, ``dice``, ``bank``; ``None`` values are skipped).  This is
      *not* the ``x,y,w,h`` form :func:`catanbot.vision.profile.parse_box`
      reads from text: a box typed by the user (e.g. ``--log-region
      x,y,w,h``) goes ``parse_box(text, size)`` ->
      ``UiProfile.set_region(name, box)`` and the profile is passed as the
      layout.  Never put CLI text boxes into the dict (a string is rejected;
      numbers taken from ``x,y,w,h`` would silently give a wrong box).

    Every box, from a dict or from a profile, is validated the same way and
    raises ``ValueError`` naming the region: an unknown region name, a
    malformed box (not four numbers), a non-finite number, a box that is
    empty or outside the image once clipped, and (dicts only) one that looks
    like screen fractions (all four numbers within 0..1).  A profile region
    with invalid fractions raises :class:`~catanbot.vision.profile.ProfileError`
    (a ``ValueError``) from ``pixel_box``.  :func:`parse_image` additionally
    ignores a ``log`` box covering more than :data:`MAX_LOG_BOX_FRACTION` of
    the screen (with a warning).
    """
    if layout is None:
        return {}
    w, h = int(size[0]), int(size[1])
    out: Dict[str, PixelBox] = {}
    if not isinstance(layout, Mapping):
        if not hasattr(layout, "pixel_box"):
            raise TypeError(f"layout must be a UiProfile or a dict of pixel boxes, not {type(layout).__name__}")
        for name in REGION_NAMES:
            box = layout.pixel_box(name, (w, h))    # UiProfile checks the fractions itself (ProfileError)
            if box is None:
                continue
            vals = _box_numbers(f"layout region '{name}' (from the profile)", box)
            clipped = _clip_box(vals, w, h)
            if clipped is None:
                raise ValueError(f"layout region '{name}' {tuple(vals)} (from the profile) is empty or outside "
                                 f"the {w}x{h} image")
            out[name] = clipped
        return out
    for name, box in layout.items():
        if box is None:
            continue
        if name not in REGION_NAMES:
            raise ValueError(f"unknown layout region '{name}' (known: {', '.join(REGION_NAMES)})")
        vals = _box_numbers(f"layout region '{name}'", box)
        if all(0.0 <= v <= 1.0 for v in vals):   # a 1-pixel box in the corner is never meant
            raise ValueError(f"layout region '{name}' {tuple(vals)} looks like screen fractions; a dict layout "
                             "takes pixel boxes (use a UiProfile for fractions)")
        clipped = _clip_box(vals, w, h)
        if clipped is None:
            raise ValueError(f"layout region '{name}' {tuple(vals)} is empty or outside the {w}x{h} image")
        out[name] = clipped
    return out


def _blank_box(arr: np.ndarray, sea: np.ndarray, box: PixelBox, cal: Calibration
               ) -> Tuple[np.ndarray, np.ndarray, RGB]:
    """Copy of ``arr`` with ``box`` painted in the measured sea colour, and ``sea`` with the box set.

    Used for the game-log panel: its light background, beige / cream / grey
    text and resource icons must not become land, token candidates, ports or
    pieces.  The colour is the median of the sea outside the box (``cal.sea``
    when there is none), so the painted area also looks like sea to every
    colour test that does not use the mask.
    """
    x0, y0, x1, y1 = box
    sea_out = sea.copy()
    sea_out[y0:y1, x0:x1] = False
    n = int(sea_out.sum())
    fill: RGB = tuple(int(v) for v in cal.sea)  # type: ignore[assignment]
    if n:
        step = max(1, int(math.sqrt(n / 100000.0)))   # ~100k samples are plenty for a median
        px = arr[::step, ::step][sea_out[::step, ::step]]
        if len(px) == 0:
            px = arr[sea_out]
        m = np.median(px, axis=0)
        fill = (int(m[0]), int(m[1]), int(m[2]))
    out = arr.copy()
    out[y0:y1, x0:x1] = fill
    sea_out[y0:y1, x0:x1] = True
    return out, sea_out, fill


def _board_under_box(geom: Dict[str, float], box: PixelBox) -> Tuple[List[int], List[int]]:
    """Hexes whose centre and coastal edges whose port-icon slot (:func:`_port_slot_points`) lie inside
    ``box`` for the fitted board ``geom``: ``(hex ids, edge ids)``, both sorted."""
    x0, y0, x1, y1 = box
    hexes = [i for i, (x, y) in enumerate(_lattice_pixels(geom, "hex")) if x0 <= x < x1 and y0 <= y < y1]
    edges = sorted(e for e, x, y in _port_slot_points(geom) if x0 <= x < x1 and y0 <= y < y1)
    return hexes, edges


def _log_box_warning(box: PixelBox, hexes: Sequence[int], edges: Sequence[int]) -> str:
    parts = []
    if hexes:
        parts.append(f"{len(hexes)} hex{'es' if len(hexes) > 1 else ''} ({', '.join(str(i) for i in hexes)})")
    if edges:
        parts.append(f"{len(edges)} port slot{'s' if len(edges) > 1 else ''} "
                     f"(coastal edge{'s' if len(edges) > 1 else ''} {', '.join(str(e) for e in edges)})")
    return (f"layout log box {box} covers {' and '.join(parts)} of the board; that area was painted as sea, so "
            "a hex there is a guess and a port there is missed - shrink the log region to the log panel")


def read_player_panel(arr: np.ndarray, cal: Calibration,
                      palette: Optional[Tuple[List[str], np.ndarray, np.ndarray]] = None, *,
                      region: Optional[Tuple[int, int, int, int]] = None
                      ) -> Tuple[List[Dict[str, Any]], float, List[str]]:
    """Rows of the player panel (left side): colour, VP, cards, dev, knights, badges, current.

    Rows are located on a downsampled copy of the left third of the image
    (they are large solid rectangles; this keeps memory flat at 4K) and the
    numbers are read at full resolution with a text mask relative to the
    measured row colour.  Rows must share the panel's x position and width
    (a row merged with adjacent pieces on the board is dropped).  ``region``
    (pixels ``x0, y0, x1, y1``) replaces the left third as the search area;
    row coordinates are always full-image pixels.  The not-found warning is
    the third item of the result (see "Not-found warnings" above).
    """
    h, w = arr.shape[:2]
    warnings: List[str] = []
    if palette is None:
        names, pal, bg = _piece_palettes([cal])
    else:
        names, pal, bg = palette
        names = list(names)
    search = _reader_box(arr, "player_panel", region)
    if search is None:
        warnings.append(f"player panel not found{_where(arr, region, search)}; player list inferred from pieces")
        return [], 0.0, warnings
    rx0, ry0, rx1, ry1 = search
    sub = arr[ry0:ry1, rx0:rx1]
    # the working scale keeps the searched area ~320 px wide (memory stays flat at 4K)
    f = max(1, int(round(sub.shape[1] / 320.0)))
    small = sub[::f, ::f]
    cls = _classify_pixels(small.reshape(-1, 3), pal, bg, 40.0).reshape(small.shape[:2])
    rows: List[Dict[str, Any]] = []
    for nme in dict.fromkeys(names):
        idx = [i for i, n in enumerate(names) if n == nme]
        m = np.isin(cls, idx)
        # Size limits stay relative to the whole screen even inside a region: rows (and the pieces
        # they must not be confused with) scale with the screen, not with the box drawn around them.
        if m.sum() * f * f < 0.002 * h * w:
            continue
        n, labels, stats, cents = _components(m)
        for i in range(1, n):
            x, y, bw, bh = [int(v) * f for v in stats[i][:4]]
            x, y = x + rx0, y + ry0
            area = int(stats[i][4]) * f * f
            fill = area / float(bw * bh)
            if area < 0.002 * h * w or bw < 0.08 * w or bh < 0.03 * h or fill < 0.55 or bw < 1.1 * bh:
                continue
            rows.append({"color": nme, "x": x, "y": y, "w": bw, "h": bh, "area": area})
    # keep one row per colour (largest), sort top to bottom
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["color"] not in best or r["area"] > best[r["color"]]["area"]:
            best[r["color"]] = r
    rows = sorted(best.values(), key=lambda r: r["y"])
    if len(rows) >= 2:
        # the panel's rows share their x position and width; a "row" that does not is a piece / merged blob
        mx = float(np.median([r["x"] for r in rows]))
        mw = float(np.median([r["w"] for r in rows]))
        rows = [r for r in rows if abs(r["x"] - mx) <= 0.15 * mw and abs(r["w"] - mw) <= 0.15 * mw]
    confs = []
    for r in rows:
        x, y, bw, bh = r["x"], r["y"], r["w"], r["h"]
        light = r["color"] != "white"
        # measured row colour for the relative text mask (text, icons and badges are a minority)
        box = arr[y:y + bh:2, x:x + bw:2]
        row_rgb = np.median(box.reshape(-1, 3), axis=0) if box.size else np.array(cal.players.get(r["color"], (0, 0, 0)))
        sy0, sy1 = y + int(0.52 * bh), y + int(0.92 * bh)
        # stat columns as drawn: vp at 6 % .. 30 %, cards 30 % .. 50 %, dev 50 % .. 70 %, knights 70 % .. 95 %
        cols = [(0.05, 0.30), (0.29, 0.50), (0.49, 0.70), (0.69, 0.96)]
        vals = []
        for a, b in cols:
            v, cf = read_number_in_region(arr, x + int(a * bw), sy0, x + int(b * bw), sy1, light_text=light,
                                          min_h=max(5, int(0.12 * bh)), bg_rgb=row_rgb, white=cal.white)
            vals.append(v)
            confs.append(cf if v is not None else 0.0)
        r["vp"], r["cards"], r["dev_cards"], r["knights"] = vals
        # badges: yellow rounded rects in the top-right of the row
        sub = arr[y:y + int(0.5 * bh), x + int(0.55 * bw):x + bw].astype(np.int16)
        yellow = (sub[:, :, 0] > 200) & (sub[:, :, 1] > 160) & (sub[:, :, 2] < 120)
        nb, lb, sb, cb = _components(yellow)
        badges = [i for i in range(1, nb) if sb[i][4] > 0.01 * bw * bh and sb[i][2] > sb[i][3]]
        r["badges"] = len(badges)
        r["badge_letters"] = []
        for i in badges:
            bx, by, bbw, bbh, _ = sb[i]
            # dark letters inside the badge: "LA" vs "LR" -> compare second letter's fill/aspect crudely
            glyph = arr[y + by:y + by + bbh, x + int(0.55 * bw) + bx:x + int(0.55 * bw) + bx + bbw].astype(np.int16)
            dark = glyph.max(axis=2) < 90
            nn, ll, ss, cc = _components(dark)
            letters = sorted([(k, ss[k]) for k in range(1, nn) if ss[k][4] > 3], key=lambda t: t[1][0])
            if len(letters) >= 2:
                k, (lx, ly, lw, lh, la) = letters[-1]
                r["badge_letters"].append(_match_letter(ll[ly:ly + lh, lx:lx + lw] == k))
            else:
                r["badge_letters"].append("?")
        # current player: thick white border just outside the coloured rect
        bx0, bx1 = max(0, x - int(0.06 * bh)), min(w, x + bw + int(0.06 * bh))
        by0, by1 = max(0, y - int(0.06 * bh)), min(h, y + bh + int(0.06 * bh))
        frame = arr[by0:by1, bx0:bx1].astype(np.int16)
        white = frame.min(axis=2) > int(0.92 * cal.white)
        inner = np.zeros_like(white)
        inner[y - by0:y - by0 + bh, x - bx0:x - bx0 + bw] = True
        ring = white & ~inner
        r["current"] = bool(ring.mean() > 0.02)
    conf = float(np.mean(confs)) if confs else 0.0
    if not rows:
        warnings.append(f"player panel not found{_where(arr, region, search)}; player list inferred from pieces")
    return rows, conf, warnings


def read_hand_bar(arr: np.ndarray, cal: Calibration, *, region: Optional[Tuple[int, int, int, int]] = None,
                  warnings: Optional[List[str]] = None) -> Tuple[Optional[List[int]], Optional[int], float]:
    """My hand from the bottom card bar: 5 resource counts and the dev-card count.

    The six cards must form a co-aligned group (same y and height, similar
    width, solid fill) so that board tiles reaching into the bottom of the
    image are never mistaken for cards.  ``(None, None, 0.0)`` when the bar
    or any resource count cannot be read (the hand is then unknown rather
    than a silent zero); the not-found warning is then appended to
    ``warnings`` when a list is given.  ``region`` (pixels ``x0, y0, x1,
    y1``) replaces the bottom 20 % of the screen as the search area.
    """
    box = _reader_box(arr, "hand_bar", region)
    out = (None, None, 0.0) if box is None else _read_hand_cards(arr, cal, box)
    if out[0] is None and warnings is not None:
        warnings.append(f"hand bar not found or not fully readable{_where(arr, region, box)}; your resources are "
                        "unknown (use --fix me.hand=...)")
    return out


def _read_hand_cards(arr: np.ndarray, cal: Calibration, box: PixelBox
                     ) -> Tuple[Optional[List[int]], Optional[int], float]:
    """:func:`read_hand_bar` inside the pixel box ``box``."""
    h, w = arr.shape[:2]
    ox, oy, ex, ey = box
    sub = arr[oy:ey, ox:ex]
    tile_refs = np.array([cal.tile[r] for r in range(5)] + [cal.dev_card], dtype=np.float32)
    bg = np.array([cal.sea, cal.panel, (255, 255, 255), (35, 35, 35)], dtype=np.float32)
    # Working scale and the card size limits below follow the whole screen (cards scale with the
    # screen, not with the box drawn around them).
    f = max(1, int(round(w / 640.0)))
    small = sub[::f, ::f]
    cls = _classify_pixels(small.reshape(-1, 3), tile_refs, bg, 45.0).reshape(small.shape[:2])
    cands: List[Tuple[int, int, int, int, int, int]] = []   # (class, x, y, w, h, area) in full-res pixels
    for c in range(6):
        n, labels, stats, cents = _components(cls == c)
        boxes = [[int(v) * f for v in stats[i][:4]] + [int(stats[i][4]) * f * f] for i in range(1, n)
                 if int(stats[i][4]) * f * f >= 0.0002 * h * w]
        # the card label (white text with a dark stroke) can split a card into an upper and a lower
        # part at the working scale: merge same-class parts that overlap in x and touch vertically
        boxes.sort(key=lambda b: b[1])
        merged: List[List[int]] = []
        for b in boxes:
            for m in merged:
                same_column = abs(m[0] - b[0]) < 0.15 * m[2] and abs(m[2] - b[2]) < 0.15 * m[2]
                gap = max(m[1], b[1]) - min(m[1] + m[3], b[1] + b[3])
                if same_column and gap < 0.25 * max(m[3], b[3]):
                    x0, y0_, x1, y1 = min(m[0], b[0]), min(m[1], b[1]), max(m[0] + m[2], b[0] + b[2]), max(m[1] + m[3], b[1] + b[3])
                    m[:] = [x0, y0_, x1 - x0, y1 - y0_, m[4] + b[4]]
                    break
            else:
                merged.append(list(b))
        for x, y, bw, bh, area in merged:
            if area < 0.0008 * h * w or bh < bw * 0.9 or area / float(bw * bh) < 0.6:
                continue
            cands.append((c, x, y, bw, bh, area))
    best: Optional[Dict[int, Tuple[int, int, int, int, int, int]]] = None
    for c0 in cands:   # the six cards are aligned and equally sized
        group: Dict[int, Tuple[int, int, int, int, int, int]] = {}
        for c1 in cands:
            if (abs(c1[2] - c0[2]) < 0.2 * c0[4] and abs(c1[4] - c0[4]) < 0.2 * c0[4]
                    and abs(c1[3] - c0[3]) < 0.3 * c0[3]):
                if c1[0] not in group or c1[5] > group[c1[0]][5]:
                    group[c1[0]] = c1
        if best is None or len(group) > len(best):
            best = group
    if best is None or any(c not in best for c in range(5)):
        return None, None, 0.0
    counts: List[Optional[int]] = [None] * 5
    confs = []
    dev = None
    for c, (_, x, y, bw, bh, _) in best.items():
        x, y = ox + x, oy + y          # full-image pixels
        card = arr[y:y + bh, x:x + bw]
        card_rgb = np.median(card.reshape(-1, 3), axis=0) if card.size else tile_refs[c]
        v, cf = read_number_in_region(arr, x, y + int(0.45 * bh), x + bw, y + bh, light_text=True,
                                      min_h=max(5, int(0.15 * bh)), bg_rgb=card_rgb, white=cal.white)
        confs.append(cf if v is not None else 0.0)
        if c < 5:
            counts[c] = v
        else:
            dev = v
    if any(v is None for v in counts):
        return None, dev, 0.0
    return [int(v) for v in counts], dev, float(np.mean(confs)) if confs else 0.0   # type: ignore[arg-type]


def _count_pips(face: np.ndarray, bw: int, bh: int, dark_level: int = 80) -> int:
    dark = face.max(axis=2) < dark_level
    nn, ll, ss, cc = _components(dark)
    pips = [k for k in range(1, nn) if 0.002 * bw * bh < ss[k][4] < 0.08 * bw * bh
            and abs(ss[k][2] - ss[k][3]) <= max(2, 0.4 * max(ss[k][2], ss[k][3]))
            and ss[k][4] / float(ss[k][2] * ss[k][3]) > 0.55]
    return len(pips)


def read_dice(arr: np.ndarray, cal: Optional[Calibration] = None, *,
              region: Optional[Tuple[int, int, int, int]] = None,
              warnings: Optional[List[str]] = None) -> Tuple[int, float]:
    """Sum of the pips on the two dice in the bottom-right corner (0 if not found).

    The dice are two equal white squares next to each other, each showing
    1-6 pips; any other pair of white blobs (a white player's pieces, text)
    is rejected.  ``cal.white`` sets the white level.  ``region`` (pixels
    ``x0, y0, x1, y1``) replaces the bottom-right corner as the search area;
    dice not found in a given region add a warning to ``warnings`` (when a
    list is given; a miss in the default area is silent).
    """
    box = _reader_box(arr, "dice", region)
    out = (0, 0.0) if box is None else _read_dice_in(arr, cal, box)
    if out[0] == 0 and region is not None and warnings is not None:
        warnings.append(f"dice not found{_where(arr, region, box)}; the roll is unknown (read as 0)")
    return out


def _read_dice_in(arr: np.ndarray, cal: Optional[Calibration], box: PixelBox) -> Tuple[int, float]:
    """:func:`read_dice` inside the pixel box ``box``."""
    h, w = arr.shape[:2]
    white_level = cal.white if cal is not None else 255
    rx0, ry0, rx1, ry1 = box
    sub = arr[ry0:ry1, rx0:rx1].astype(np.int16)
    white = sub.min(axis=2) > int(0.88 * white_level)
    n, labels, stats, cents = _components(white)
    squares = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # minimum die size relative to the screen (dice scale with it, not with the box)
        if area < 0.0005 * h * w or abs(bw - bh) > 0.3 * max(bw, bh):
            continue
        fill = area / float(bw * bh)
        if fill < 0.5:
            continue
        squares.append((int(x), int(y), int(bw), int(bh)))
    pairs = []
    for a in range(len(squares)):
        for b in range(a + 1, len(squares)):
            xa, ya, wa, ha = squares[a]
            xb, yb, wb, hb = squares[b]
            size = max(wa, wb)
            if abs(wa - wb) > 0.25 * size or abs(ha - hb) > 0.25 * size:
                continue
            if abs(ya - yb) > 0.5 * size or abs(xa - xb) > 2.5 * size:
                continue
            pairs.append((wa * ha + wb * hb, a, b))
    if not pairs:
        return 0, 0.0
    _, a, b = max(pairs)
    total = 0
    for x, y, bw, bh in (squares[a], squares[b]):
        mx, my = int(0.1 * bw), int(0.1 * bh)
        face = sub[y + my:y + bh - my, x + mx:x + bw - mx]
        pips = _count_pips(face, bw, bh, dark_level=max(40, int(0.31 * white_level)))
        if not 1 <= pips <= 6:
            return 0, 0.0
        total += pips
    if 2 <= total <= 12:
        return total, 0.9
    return 0, 0.0


def read_bank_panel(arr: np.ndarray, cal: Calibration, *, region: Optional[Tuple[int, int, int, int]] = None,
                    warnings: Optional[List[str]] = None) -> Tuple[Optional[Dict[str, int]], Optional[int]]:
    """Bank stock per resource and the dev-deck size from the panel top-right (best effort).

    ``region`` (pixels ``x0, y0, x1, y1``) replaces the top-right corner as
    the search area; a bank not found (or not fully readable) in a given
    region adds a warning to ``warnings`` (when a list is given; a miss in
    the default area is silent).
    """
    box = _reader_box(arr, "bank", region)
    out = (None, None) if box is None else _read_bank_in(arr, cal, box)
    if out[0] is None and region is not None and warnings is not None:
        warnings.append(f"bank panel not found or not fully readable{_where(arr, region, box)}")
    return out


def _read_bank_in(arr: np.ndarray, cal: Calibration, box: PixelBox
                  ) -> Tuple[Optional[Dict[str, int]], Optional[int]]:
    """:func:`read_bank_panel` inside the pixel box ``box``."""
    h, w = arr.shape[:2]
    ox, oy, ex, ey = box
    sub = arr[oy:ey, ox:ex].astype(np.int16)
    navy = np.array(cal.panel, dtype=np.int16)
    m = np.abs(sub - navy).sum(axis=2) < 60
    n, labels, stats, cents = _components(m)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # minimum panel size relative to the screen (the panel scales with it, not with the box)
        if area > 0.003 * h * w and bw > 2 * bh and (best is None or area > best[4]):
            best = (x, y, bw, bh, area)
    if best is None:
        return None, None
    x, y, bw, bh, _ = best
    x, y = ox + x, oy + y          # full-image pixels
    cell = bw / 6.0
    vals = []
    for k in range(6):
        v, cf = read_number_in_region(arr, x + int(k * cell), y + int(0.55 * bh), x + int((k + 1) * cell),
                                      y + bh, light_text=True, min_h=max(5, int(0.15 * bh)), bg_rgb=cal.panel,
                                      white=cal.white)
        vals.append(v)
    if any(v is None for v in vals[:5]):
        return None, vals[5]
    return {B.RESOURCE_NAMES[r]: int(vals[r]) for r in range(5)}, vals[5]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _estimate_token_colour(arr: np.ndarray, geom: Dict[str, float]) -> Optional[RGB]:
    """Median colour of the bright, low-saturation pixels near the hex centres (the number tokens)."""
    hs = geom["hex_size"]
    cols = []
    for cx, cy in _lattice_pixels(geom, "hex"):
        px = _ring_pixels(arr, cx, cy, 0.0, 0.25 * hs)
        if len(px) < 10:
            continue
        hue, sat, val = rgb_to_hsv(px)
        m = (sat < 0.35) & (val > 0.45)
        if m.mean() > 0.3:
            cols.append(np.median(px[m], axis=0))
    if len(cols) < 6:
        return None
    m = np.median(np.array(cols), axis=0)
    return (int(m[0]), int(m[1]), int(m[2]))


def parse_image(path_or_image: Union[str, Image.Image, np.ndarray], me: Optional[str] = None,
                assume_standard: bool = True, calibration: Optional[Calibration] = None,
                read_ui: bool = True,
                layout: Optional[Union[UiProfile, Dict[str, Tuple[int, int, int, int]]]] = None) -> ParseResult:
    """Parse a Colonist.io screenshot (path, PIL image or RGB array) into a :class:`ParseResult`.

    ``me`` is the colour of the screen owner (defaults to the first panel
    row), ``assume_standard`` constrains tiles / numbers to the standard
    multisets, ``calibration`` overrides the reference colours and
    ``read_ui`` enables the panel / hand bar / dice / bank readers.
    ``layout`` (a :class:`~catanbot.vision.profile.UiProfile` or a dict of
    pixel boxes, see :func:`layout_boxes`) tells the readers where the UI
    panels are; a reader without a box searches its default area.  The
    ``log`` box is excluded from the board analysis (painted with the
    measured sea colour); a warning names the hexes and port slots of the
    fitted board it covers (``debug["log_covers"]``), and a ``log`` box
    larger than :data:`MAX_LOG_BOX_FRACTION` of the screen is ignored with a
    warning (``debug["layout_ignored"]``).  ``debug["layout"]`` records the
    pixel boxes used.
    Everything that cannot be read becomes a warning; ``confidence`` holds a
    0..1 value per stage (``geometry``, ``hexes``, ``numbers``,
    ``numbers_min``, ``robber``, ``ports``, ``pieces``, ``panel``, ``hand``,
    ``dice``, ``bank``).
    """
    cal = calibration or Calibration()
    arr = _to_rgb_array(path_or_image)
    boxes = layout_boxes(layout, (arr.shape[1], arr.shape[0]))
    warnings: List[str] = []
    confidence: Dict[str, float] = {}
    ignored: Dict[str, PixelBox] = {}
    log_box = boxes.get("log")
    if log_box is not None:
        share = (log_box[2] - log_box[0]) * (log_box[3] - log_box[1]) / float(arr.shape[0] * arr.shape[1])
        if share > MAX_LOG_BOX_FRACTION:   # painting it as sea would hide the board: a wrong region
            ignored["log"] = boxes.pop("log")
            log_box = None
            warnings.append(f"layout log box {ignored['log']} covers {100 * share:.0f}% of the "
                            f"{arr.shape[1]}x{arr.shape[0]} screen; ignored (the log panel is a column beside the "
                            "board) - fix the log region")
    sea = sea_mask(arr, cal)
    noise_region = sea
    if log_box is not None:   # the log panel's text must not count as (or hide) sensor noise
        noise_region = sea.copy()
        noise_region[log_box[1]:log_box[3], log_box[0]:log_box[2]] = False
    # A noisy screenshot (photo of a monitor, heavy re-encoding) breaks the colour masks; a
    # light median filter removes the grain without hurting the digits.
    noise, med = _noise_level(arr, noise_region)
    if noise > 2.0:
        arr = med
        sea = sea_mask(arr, cal)
        warnings.append(f"noisy image (noise level {noise:.1f}); a 3x3 median filter was applied")
    # The UI readers given an explicit box read the image as it is (``ui_arr``); the board
    # analysis and the readers searching their default areas see the log panel painted as sea.
    ui_arr = arr
    log_fill: Optional[RGB] = None
    known: Optional[np.ndarray] = None       # the painted log box is unknown, not sea, to the lattice ranking
    if log_box is not None:
        arr, sea, log_fill = _blank_box(arr, sea, log_box, cal)
        known = np.ones(sea.shape, dtype=bool)
        known[log_box[1]:log_box[3], log_box[0]:log_box[2]] = False
    geom, debug, w0 = find_board(arr, cal, sea=sea, valid=known)
    debug["noise_level"] = noise
    debug["layout"] = dict(boxes)
    if ignored:
        debug["layout_ignored"] = ignored
    if log_fill is not None:
        debug["log_fill"] = log_fill
    warnings.extend(w0)
    # ---- brightness / palette adaptation ------------------------------------------
    # The number tokens are the one element with a known colour on every board: if they are
    # much brighter / darker than the reference, the whole screenshot is (monitor gamma, night
    # mode, re-encoding) and every reference colour is rescaled accordingly.
    cal_img = cal
    tok = _estimate_token_colour(arr, geom)
    if tok is not None and sum(abs(a - b) for a, b in zip(tok, cal.token)) > 40:
        ratio = _luma(tok) / max(1.0, _luma(cal.token))
        cal_img = scaled_calibration(cal, ratio, token=tok)
        debug["adapted_token"] = tok
        debug["brightness_ratio"] = ratio
        warnings.append(f"image colours differ from the reference palette (number tokens read as RGB {tok}); "
                        f"reference colours scaled by {ratio:.2f}")
        if debug.get("method") != "tokens":
            pts = _token_points(arr, cal_img)
            cands = _fit_candidates(pts, min_matched=8)
            if cands:
                g, resid, matched, iou, ranking = _pick_registration(cands, ~sea, known)
                geom = dict(g, width=float(arr.shape[1]), height=float(arr.shape[0]))
                debug.update({"method": "tokens(adapted)", "points": int(len(pts)), "matched": matched,
                              "residual": resid, "land_iou": iou, "candidates": ranking,
                              "confidence": float(min(1.0, matched / 14.0)), "geometry": dict(geom)})
                warnings = [w for w in warnings if "blob fallback" not in w and "not located reliably" not in w]
    if log_box is not None:   # the painted log box must not hide part of the (final) board
        covered_hexes, covered_slots = _board_under_box(geom, log_box)
        debug["log_covers"] = {"hexes": covered_hexes, "port_slots": covered_slots}
        if covered_hexes or covered_slots:
            warnings.append(_log_box_warning(log_box, covered_hexes, covered_slots))
    confidence["geometry"] = float(debug.get("confidence", 0.0))
    cals = [cal] if cal_img is cal else [cal, cal_img]
    # ---- board ----------------------------------------------------------------------
    resources, res_conf, meds = classify_tiles(arr, geom, cal_img, assume_standard)
    confidence["hexes"] = float(np.mean(res_conf))
    # Per-image background colours: the 19 tile ring medians and the sea median.  A "tile"
    # whose median is a player colour is a hex hidden under a panel row - adding it would
    # veto that player's pieces, so it is skipped (real tiles differ from every player colour).
    player_rgbs = np.array([c for cc in cals for c in cc.players.values()], dtype=np.float32)
    extra_bg = [tuple(int(v) for v in m) for m in meds
                if np.linalg.norm(player_rgbs - np.asarray(m, dtype=np.float32)[None, :], axis=1).min() > 16.0]
    if sea.any():
        sm = np.median(arr[sea], axis=0)
        extra_bg.append((int(sm[0]), int(sm[1]), int(sm[2])))
    palette = _piece_palettes(cals, extra_bg)
    creams = _token_visibility(arr, geom, cal_img)
    if debug.get("method") == "blob":
        # verify the fallback geometry: a correct registration puts a cream token disc on most hex centres
        n_vis = sum(1 for c in creams if c > 0.25)
        debug["tokens_at_centres"] = n_vis
        if n_vis < 8:
            confidence["geometry"] = min(confidence["geometry"], 0.25)
            warnings = [w for w in warnings if "blob fallback" not in w]
            warnings.append(f"board not located reliably (blob fallback; number tokens found at only {n_vis} of 19 hex "
                            "centres); the board below is probably wrong - use --state / --fix or the LLM parser")
    robber, rob_conf = find_robber(arr, geom, cal_img)
    if robber < 0:
        rob_conf = 0.0
        hidden = [i for i in range(B.NUM_HEXES) if resources[i] != B.DESERT and creams[i] < 0.25]
        desert = next((i for i, r in enumerate(resources) if r == B.DESERT), 0)
        if len(hidden) == 1:
            robber = hidden[0]
            warnings.append(f"robber pawn not found; assuming it covers the number token of hex {robber} - "
                            "set it with --fix robber=HEX if wrong")
        else:
            robber = desert
            warnings.append(f"robber pawn not found; assuming it is on the desert (hex {desert}, no production "
                            "blocked) - set it with --fix robber=HEX if wrong")
    confidence["robber"] = float(rob_conf)
    numbers, num_conf, w1 = read_numbers(arr, geom, resources, cal_img, assume_standard, robber, creams=creams)
    warnings.extend(w1)
    num_confs = [c for i, c in enumerate(num_conf) if resources[i] != B.DESERT]
    confidence["numbers"] = float(np.mean(num_confs)) if num_confs else 0.0
    confidence["numbers_min"] = float(min(num_confs)) if num_confs else 0.0
    ports, port_conf, w2 = detect_ports(arr, geom, cal_img)
    warnings.extend(w2)
    confidence["ports"] = port_conf
    # ---- UI chrome ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    hand = None
    dev = None
    dice = 0
    bank = None
    deck_left = None
    if read_ui:
        def ui_image(name: str) -> np.ndarray:
            return ui_arr if name in boxes else arr

        try:
            rows, panel_conf, w3 = read_player_panel(ui_image("player_panel"), cal_img, palette=palette,
                                                     region=boxes.get("player_panel"))
            warnings.extend(w3)
            confidence["panel"] = panel_conf
        except Exception as ex:  # pragma: no cover - best effort
            warnings.append(f"player panel unreadable: {ex}")
        # each reader words its own not-found warning ("Not-found warnings" at the top of the UI chrome section)
        try:
            w4: List[str] = []
            hand, dev, hand_conf = read_hand_bar(ui_image("hand_bar"), cal_img, region=boxes.get("hand_bar"),
                                                 warnings=w4)
            confidence["hand"] = hand_conf
            warnings.extend(w4)
        except Exception as ex:  # pragma: no cover
            warnings.append(f"hand bar unreadable: {ex}")
        try:
            w5: List[str] = []
            dice, dice_conf = read_dice(ui_image("dice"), cal_img, region=boxes.get("dice"), warnings=w5)
            confidence["dice"] = float(dice_conf)
            warnings.extend(w5)
        except Exception:
            dice = 0
        try:
            w6: List[str] = []
            bank, deck_left = read_bank_panel(ui_image("bank"), cal_img, region=boxes.get("bank"), warnings=w6)
            confidence["bank"] = 1.0 if bank is not None else 0.0
            warnings.extend(w6)
        except Exception:
            bank = None
    # ---- pieces ---------------------------------------------------------------------
    # Never restricted to the panel colours: a missed panel row must not delete a player.
    piece_details: Dict[str, Any] = {}
    buildings, roads, piece_conf = detect_pieces(arr, geom, cal, None, palette=palette, details=piece_details)
    confidence["pieces"] = piece_conf
    row_by_color = {r["color"]: r for r in rows}
    n_pieces: Dict[str, int] = {}
    for v, (c, _, _) in buildings.items():
        n_pieces[c] = n_pieces.get(c, 0) + 1
    for e, (c, _) in roads.items():
        n_pieces[c] = n_pieces.get(c, 0) + 1
    dropped = {c for c, n in n_pieces.items() if c not in row_by_color and n < 2}
    if dropped:
        buildings = {v: t for v, t in buildings.items() if t[0] not in dropped}
        roads = {e: t for e, t in roads.items() if t[0] not in dropped}
    colors_seen = [r["color"] for r in rows]
    for c in sorted(n_pieces, key=lambda c: -n_pieces[c]):
        if c not in colors_seen and c not in dropped:
            colors_seen.append(c)
            warnings.append(f"{c}: pieces found but no panel row; VP / cards / dev cards unknown "
                            f"(use --fix {c}.vp=N etc.)")
    if not colors_seen:
        colors_seen = ["red"]
        warnings.append("no player pieces or panel found")
    city_hits = piece_details.get("city_hits", {})
    uncertain = sorted(v for v, hit in city_hits.items() if 0.45 <= hit < 0.75)
    if uncertain:
        warnings.append("city / settlement uncertain at vertex " + ", ".join(
            f"{v} ({'city' if buildings[v][1] else 'settlement'}, {city_hits[v]:.2f})" for v in uncertain)
            + " - check the debug overlay")
    # ---- players ------------------------------------------------------------
    players = []
    for c in colors_seen:
        r = row_by_color.get(c, {})
        setts = sorted(v for v, (cc, city, _) in buildings.items() if cc == c and not city)
        cities = sorted(v for v, (cc, city, _) in buildings.items() if cc == c and city)
        rds = sorted(e for e, (cc, _) in roads.items() if cc == c)
        vp_pieces = len(setts) + 2 * len(cities)
        badges = r.get("badge_letters", [])
        p = {
            "color": c, "name": c,
            "vp": r.get("vp") if r.get("vp") is not None else vp_pieces,
            "cards": r.get("cards") if r.get("cards") is not None else 0,
            "dev_cards": r.get("dev_cards") if r.get("dev_cards") is not None else 0,
            "knights": r.get("knights") if r.get("knights") is not None else 0,
            "longest_road": "R" in badges,
            "largest_army": "A" in badges,
            "settlements": setts, "cities": cities, "roads": rds,
        }
        if r and (r.get("vp") is None or r.get("cards") is None):
            warnings.append(f"{c}: could not read all panel numbers")
        players.append(p)
    # me / current
    me_color = None
    if me:
        me_color = me.lower()
        if me_color not in colors_seen:
            warnings.append(f"--me {me} is not among the detected players {colors_seen}")
            me_color = None
    if me_color is None:
        me_color = colors_seen[0]
        if len(colors_seen) > 1 and not me:
            warnings.append(f"'me' not given; assuming {me_color} (first panel row). Use --me COLOR to override")
    current = next((r["color"] for r in rows if r.get("current")), me_color)
    for p in players:
        if p["color"] == me_color:
            if hand is not None:
                p["resources"] = {B.RESOURCE_NAMES[r]: int(hand[r]) for r in range(5)}
                if p["cards"] in (0, None) or p["cards"] != sum(hand):
                    p["cards"] = int(sum(hand))
            if dev is not None:
                p["dev_cards"] = int(dev)
    # Award sanity: if no badge was read, infer holders from pieces / knights.
    if not any(p["longest_road"] for p in players):
        lens = {p["color"]: S.longest_road_length(p["roads"]) for p in players}
        best = max(lens.values()) if lens else 0
        if best >= 5 and list(lens.values()).count(best) == 1:
            for p in players:
                p["longest_road"] = lens[p["color"]] == best
    if not any(p["largest_army"] for p in players):
        ks = {p["color"]: p["knights"] for p in players}
        best = max(ks.values()) if ks else 0
        if best >= 3 and list(ks.values()).count(best) == 1:
            for p in players:
                p["largest_army"] = ks[p["color"]] == best
    parsed: Dict[str, Any] = {
        "hexes": [{"resource": B.RESOURCE_NAMES[r], "number": (numbers[i] or None) if r != B.DESERT else None}
                  for i, r in enumerate(resources)],
        "robber": int(robber),
        "ports": ports if ports else None,
        "players": players,
        "me": me_color,
        "current_player": current,
        "dice": int(dice),
    }
    if bank is not None:
        parsed["bank"] = bank
    if deck_left is not None:
        parsed["dev_deck_remaining"] = int(deck_left)
    if parsed["ports"] is None:
        del parsed["ports"]
        warnings.append("no ports detected; using the standard port layout")
    warnings.extend(S.validate(parsed))
    state = S.parsed_to_state(parsed)
    debug["buildings"] = buildings
    debug["roads"] = roads
    debug["city_hits"] = city_hits
    debug["numbers_conf"] = num_conf
    debug["token_visibility"] = creams
    debug["tile_medians"] = meds.tolist()
    debug["panel_rows"] = rows
    return ParseResult(parsed=parsed, state=state, confidence=confidence, warnings=warnings, debug=debug)


# ---------------------------------------------------------------------------
# Debug overlay and calibration
# ---------------------------------------------------------------------------
def draw_debug(image: Union[str, Image.Image, np.ndarray], result: ParseResult) -> Image.Image:
    img = Image.fromarray(_to_rgb_array(image)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for name, (x0, y0, x1, y1) in (result.debug.get("layout") or {}).items():   # UI regions given by a layout
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=2)
        draw.text((x0 + 3, y0 + 2), name, fill=(0, 255, 0))
    geom = result.debug.get("geometry")
    if not geom:
        return img
    hs = geom["hex_size"]
    try:
        font = ImageFont.truetype(DEFAULT_FONT_PATH, max(10, int(0.22 * hs)))
    except Exception:
        font = ImageFont.load_default()
    for i, (cx, cy) in enumerate(_lattice_pixels(geom, "hex")):
        pts = [B.hex_corner(cx, cy, k, hs) for k in range(6)]
        draw.polygon(pts, outline=(255, 0, 255))
        hx = result.parsed["hexes"][i]
        label = f"{i}:{hx['resource'][:2]}{hx['number'] or ''}"
        draw.text((cx - 0.3 * hs, cy - 0.62 * hs), label, fill=(255, 255, 0), font=font)
    rx, ry = _lattice_pixels(geom, "hex")[result.parsed["robber"]]
    draw.ellipse([rx - 0.7 * hs, ry - 0.7 * hs, rx + 0.7 * hs, ry + 0.7 * hs], outline=(255, 0, 0), width=3)
    vpix = _lattice_pixels(geom, "vertex")
    for v, (c, city, frac) in result.debug.get("buildings", {}).items():
        x, y = vpix[v]
        r = 0.22 * hs
        draw.rectangle([x - r, y - r, x + r, y + r], outline=(0, 255, 255) if city else (255, 255, 255), width=2)
        draw.text((x + r, y - r), f"{c[:2]}{'C' if city else 'S'}", fill=(255, 255, 255), font=font)
    epix = _lattice_pixels(geom, "edge")
    for e, (c, frac) in result.debug.get("roads", {}).items():
        x, y = epix[e]
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 0, 0))
        draw.text((x + 4, y), c[:2], fill=(255, 255, 255), font=font)
    for p in result.parsed.get("ports", []) or []:
        x, y = epix[p["edge"]]
        draw.text((x, y), "P:" + p["type"], fill=(255, 0, 255), font=font)
    return img


def calibrate_from_image(image: Union[str, Image.Image, np.ndarray], known_state, me: Optional[str] = None,
                         base: Optional[Calibration] = None) -> Calibration:
    """Re-estimate tile / player / sea colours from a screenshot whose state is known."""
    cal = base or Calibration()
    arr = _to_rgb_array(image)
    geom, _, _ = find_board(arr, cal)
    hs = geom["hex_size"]
    new = Calibration(**{k: (dict(v) if isinstance(v, dict) else v) for k, v in cal.__dict__.items()})
    sums: Dict[int, List[np.ndarray]] = {}
    for i, (cx, cy) in enumerate(_lattice_pixels(geom, "hex")):
        res = known_state.hexes[i][0]
        px = _ring_pixels(arr, cx, cy, 0.40 * hs, 0.62 * hs)
        if len(px):
            sums.setdefault(res, []).append(np.median(px, axis=0))
    for res, lst in sums.items():
        m = np.median(np.array(lst), axis=0)
        new.tile[res] = (int(m[0]), int(m[1]), int(m[2]))
    vpix = _lattice_pixels(geom, "vertex")
    for p in known_state.players:
        cols = []
        for v in list(p.settlements) + list(p.cities):
            px = _ring_pixels(arr, vpix[v][0], vpix[v][1], 0.0, 0.08 * hs)
            if len(px):
                cols.append(np.median(px, axis=0))
        if cols:
            m = np.median(np.array(cols), axis=0)
            new.players[p.color] = (int(m[0]), int(m[1]), int(m[2]))
    sea = sea_mask(arr, cal)
    if sea.any():
        m = np.median(arr[sea], axis=0)
        new.sea = (int(m[0]), int(m[1]), int(m[2]))
    return new
