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

The parser never uses ``synth.board_geometry``; it finds the board itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import board as B
from ..state import PLAYER_COLORS
from . import schema as S
from .result import ParseResult

try:  # optional, used for connected components / morphology / resize
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

RGB = Tuple[int, int, int]
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


def _land_iou(land: np.ndarray, geom: Dict[str, float], scale: int = 4, shrink: float = 0.96) -> float:
    """IoU between the (slightly shrunk) 19-hex lattice under ``geom`` and the land mask."""
    small = land[::scale, ::scale]
    m = _hex_inside_mask(land.shape, geom, scale=scale, shrink=shrink)
    m = m[:small.shape[0], :small.shape[1]]
    s = small[:m.shape[0], :m.shape[1]]
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


def _pick_registration(cands: Sequence[Tuple[Dict[str, float], float, int]], land: np.ndarray
                       ) -> Tuple[Dict[str, float], float, int, float, List[Tuple[float, int]]]:
    """Choose among lattice candidates by land-mask overlap.

    Candidates within 3 matches of the best are ranked by the IoU between
    their lattice and the land mask (ties by matches, then residual); the
    translated registrations that match a subset of the visible tokens
    overlap the sea and lose.  Returns ``(geometry, residual, matched, iou,
    ranking)`` with ``ranking = [(iou, matched), ...]`` for debugging.
    """
    best_m = max(m for _, _, m in cands)
    pool = [(g, r, m) for g, r, m in cands if m >= best_m - 3]
    scored = sorted(((_land_iou(land, g), m, -r, g) for g, r, m in pool), key=lambda t: (t[0], t[1], t[2]), reverse=True)
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


def find_board(arr: np.ndarray, cal: Calibration, sea: Optional[np.ndarray] = None
               ) -> Tuple[Dict[str, float], Dict[str, Any], List[str]]:
    """Locate the board: returns (geometry, debug, warnings).

    ``sea`` may be a precomputed :func:`sea_mask`.  ``debug`` carries the
    method (``tokens`` / ``tiles`` / ``blob``), the number of matched points,
    the land IoU of the chosen registration and ``confidence`` (0..1).
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
        geom, resid, matched, iou, ranking = _pick_registration(cands, land)
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
                geom, resid, matched, iou, ranking = _pick_registration(cands, land)
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
    global _DIGITS
    if _DIGITS is None:
        from .digits import DEFAULT_MODEL_PATH, DigitClassifier
        _DIGITS = DigitClassifier.load(DEFAULT_MODEL_PATH)
    return _DIGITS


def _cream_fraction(arr: np.ndarray, cx: float, cy: float, r: float, cal: Calibration) -> float:
    px = _ring_pixels(arr, cx, cy, 0.0, r).astype(np.int16)
    if len(px) == 0:
        return 0.0
    ref = np.array(cal.token, dtype=np.int16)
    d = np.abs(px - ref).sum(axis=1)
    return float((d < 60).mean())


def read_numbers(arr: np.ndarray, geom: Dict[str, float], resources: Sequence[int], cal: Calibration,
                 assume_standard: bool = True, robber_hex: int = -1) -> Tuple[List[int], List[float], List[str]]:
    from .digits import CLASSES
    hs = geom["hex_size"]
    centers = _lattice_pixels(geom, "hex")
    clf = _digit_classifier()
    crops = []
    creams = []
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
        creams.append(_cream_fraction(arr, cx, cy, 0.28 * hs, cal))
    probs = np.asarray(clf.predict_proba(crops), dtype=np.float64)
    probs = np.clip(probs, 1e-6, 1.0)
    warnings: List[str] = []
    numbers = [0] * B.NUM_HEXES
    conf = [0.0] * B.NUM_HEXES
    numbered = [i for i in range(B.NUM_HEXES) if resources[i] != B.DESERT]
    # A hex whose token is hidden (robber on it) or missing gets a flat distribution.
    for i in range(B.NUM_HEXES):
        if creams[i] < 0.25 and resources[i] != B.DESERT:
            probs[i] = np.full(len(CLASSES), 1.0 / len(CLASSES))
            if i != robber_hex:
                warnings.append(f"hex {i}: number token not clearly visible")
        if creams[i] > 0.5 and resources[i] == B.DESERT:
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
    else:
        for i in numbered:
            k = int(np.argmax(probs[i]))
            numbers[i] = CLASSES[k]
            conf[i] = float(probs[i][k])
        if assume_standard:
            warnings.append("non-standard number of desert hexes; numbers read independently")
    return numbers, conf, warnings


# ---------------------------------------------------------------------------
# Robber
# ---------------------------------------------------------------------------
def find_robber(arr: np.ndarray, geom: Dict[str, float], cal: Calibration) -> Tuple[int, float]:
    hs = geom["hex_size"]
    hue, sat, val = rgb_to_hsv(arr)
    dark = (val < 0.42) & (sat < 0.30)
    # restrict to the board area
    inside = _hex_inside_mask(arr.shape[:2], geom, scale=1, shrink=1.0)
    dark &= inside
    n, labels, stats, cents = _components(dark)
    centers = _lattice_pixels(geom, "hex")
    best_h, best_score = -1, 0.0
    amin, amax = 0.06 * hs * hs, 0.30 * hs * hs
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < amin or area > amax:
            continue
        if bw > 0.7 * hs or bh > 0.8 * hs:
            continue
        cx, cy = cents[i]
        d = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
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
    cols = [cal.sea, cal.token, cal.port, cal.robber, (35, 35, 35), (255, 255, 255)]
    cols += [cal.tile[r] for r in range(6)]
    # shaded tile-art colours
    for r in range(6):
        c = cal.tile[r]
        cols.append((int(c[0] * 0.6), int(c[1] * 0.6), int(c[2] * 0.6)))
    return np.array(cols, dtype=np.float32)


def _classify_pixels(px: np.ndarray, pal: np.ndarray, bg: np.ndarray, max_dist: float) -> np.ndarray:
    """Return index into ``pal`` for each pixel, or -1 if closer to a background colour / too far."""
    if len(px) == 0:
        return np.zeros(0, dtype=np.int64)
    p = px.astype(np.float32)
    dp = np.linalg.norm(p[:, None, :] - pal[None, :, :], axis=2)
    db = np.linalg.norm(p[:, None, :] - bg[None, :, :], axis=2).min(axis=1)
    j = dp.argmin(axis=1)
    dmin = dp[np.arange(len(p)), j]
    ok = (dmin < max_dist) & (dmin < db)
    return np.where(ok, j, -1)


def detect_pieces(arr: np.ndarray, geom: Dict[str, float], cal: Calibration,
                  known_colors: Optional[Sequence[str]] = None):
    """Buildings per vertex and roads per edge.

    Returns ``(buildings, roads, conf)`` with ``buildings = {vertex: (colour, is_city, fraction)}``
    and ``roads = {edge: (colour, fraction)}``.
    """
    hs = geom["hex_size"]
    names, pal = _player_palette(cal)
    bg = _background_palette(cal)
    if known_colors:
        keep = [i for i, nme in enumerate(names) if nme in known_colors]
        if keep:
            names = [names[i] for i in keep]
            pal = pal[keep]
    h, w = arr.shape[:2]
    buildings: Dict[int, Tuple[str, bool, float]] = {}
    roads: Dict[int, Tuple[str, float]] = {}
    fracs = []
    for v, (vx, vy) in enumerate(_lattice_pixels(geom, "vertex")):
        px = _ring_pixels(arr, vx, vy, 0.0, 0.16 * hs)
        cls = _classify_pixels(px, pal, bg, cal.player_max_dist)
        if len(cls) == 0:
            continue
        counts = np.bincount(cls[cls >= 0], minlength=len(names))
        frac = counts.max() / len(cls) if len(cls) else 0.0
        if frac < cal.building_min_fraction:
            continue
        c = int(counts.argmax())
        # City vs settlement: a city is ~0.46 hs wide, a settlement ~0.30 hs.  Roads leave the
        # vertex along the edge directions (never horizontally on a pointy-top board), so probe
        # small discs left and right of the vertex at 0.21 hs: coloured for a city only.
        # City vs settlement: a city has a tower on its upper-left, so the disc at
        # ``cal.city_probe`` (hex-size units, relative to the vertex) is coloured for a
        # city only; it lies >= 30 degrees off every road direction for both vertex
        # orientations.  "Y" vertices (edge straight up) get a second, lower tower probe.
        up_edge = any(abs(B.VERTEX_POS[n][0] - B.VERTEX_POS[v][0]) < 0.1 and B.VERTEX_POS[n][1] < B.VERTEX_POS[v][1]
                      for n in B.VERTEX_NEIGHBORS[v])
        probes = cal.city_probes_y if up_edge else cal.city_probes_inv
        hits = []
        for ox, oy in probes:
            px2 = _ring_pixels(arr, vx + ox * hs, vy + oy * hs, 0.0, 0.03 * hs)
            cls2 = _classify_pixels(px2, pal, bg, cal.player_max_dist)
            hits.append(float((cls2 == c).mean()) if len(cls2) else 0.0)
        is_city = any(hit >= 0.6 for hit in hits)
        buildings[v] = (names[c], bool(is_city), float(frac))
        fracs.append(frac)
    for e, (a, b) in enumerate(B.EDGE_VERTICES):
        (x1, y1), (x2, y2) = _lattice_pixels(geom, "vertex")[[a, b]]
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
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        px = arr[ys, xs]
        cls = _classify_pixels(px, pal, bg, cal.player_max_dist)
        counts = np.bincount(cls[cls >= 0], minlength=len(names))
        frac = counts.max() / len(cls)
        if frac >= cal.road_min_fraction:
            roads[e] = (names[int(counts.argmax())], float(frac))
            fracs.append(frac)
    conf = float(np.mean([min(1.0, (f - 0.2) / 0.5) for f in fracs])) if fracs else 1.0
    return buildings, roads, conf


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
def detect_ports(arr: np.ndarray, geom: Dict[str, float], cal: Calibration) -> Tuple[List[Dict[str, Any]], float, List[str]]:
    hs = geom["hex_size"]
    h, w = arr.shape[:2]
    ports = []
    warnings: List[str] = []
    ref = np.array(cal.port, dtype=np.int16)
    tile_refs = np.array([cal.tile[r] for r in range(5)], dtype=np.float32)
    scores = []
    for e in B.COASTAL_EDGES:
        hx, hy = B.HEX_CENTERS[B.EDGE_HEXES[e][0]]
        ex, ey = B.EDGE_POS[e]
        dx, dy = ex - hx, ey - hy
        n = math.hypot(dx, dy) or 1.0
        dx, dy = dx / n, dy / n
        mx, my = geom["cx"] + ex * hs, geom["cy"] + ey * hs
        cx, cy = mx + dx * 0.62 * hs, my + dy * 0.62 * hs
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


def _templates() -> np.ndarray:
    global _TEMPLATES
    if _TEMPLATES is None:
        try:
            font = ImageFont.truetype(DEFAULT_FONT_PATH, 64)
        except Exception:
            font = ImageFont.load_default()
        temps = []
        for d in range(10):
            img = Image.new("L", (80, 90), 0)
            ImageDraw.Draw(img).text((10, 5), str(d), fill=255, font=font)
            temps.append(_norm_glyph(np.asarray(img) > 128))
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
        tz = t[k] - t[k].mean()
        denom = math.sqrt((gz ** 2).sum() * (tz ** 2).sum()) or 1.0
        scores.append(float((gz * tz).sum() / denom))
    k = int(np.argmax(scores))
    return k, scores[k]


def read_number_in_region(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, light_text: bool = True,
                          min_h: int = 6) -> Tuple[Optional[int], float]:
    """OCR a (possibly multi-digit) number made of light (or dark) glyphs inside a region."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0
    sub = arr[y0:y1, x0:x1].astype(np.int16)
    if light_text:
        mask = sub.min(axis=2) > 205
    else:
        mask = sub.max(axis=2) < 70
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
def read_player_panel(arr: np.ndarray, cal: Calibration) -> Tuple[List[Dict[str, Any]], float, List[str]]:
    """Rows of the player panel (left side): colour, VP, cards, dev, knights, badges, current."""
    h, w = arr.shape[:2]
    warnings: List[str] = []
    names, pal = _player_palette(cal)
    bg = _background_palette(cal)
    region = arr[:, : int(0.34 * w)]
    cls = _classify_pixels(region.reshape(-1, 3), pal, bg, 40.0).reshape(region.shape[:2])
    rows: List[Dict[str, Any]] = []
    for c, nme in enumerate(names):
        m = cls == c
        if m.sum() < 0.002 * h * w:
            continue
        n, labels, stats, cents = _components(m)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            fill = area / float(bw * bh)
            if area < 0.002 * h * w or bw < 0.08 * w or bh < 0.03 * h or fill < 0.55 or bw < 1.5 * bh:
                continue
            rows.append({"color": nme, "x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "area": int(area)})
    # keep one row per colour (largest), sort top to bottom
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["color"] not in best or r["area"] > best[r["color"]]["area"]:
            best[r["color"]] = r
    rows = sorted(best.values(), key=lambda r: r["y"])
    confs = []
    for r in rows:
        x, y, bw, bh = r["x"], r["y"], r["w"], r["h"]
        light = r["color"] != "white"
        sy0, sy1 = y + int(0.52 * bh), y + int(0.92 * bh)
        # stat columns as drawn: vp at 6 % .. 30 %, cards 30 % .. 50 %, dev 50 % .. 70 %, knights 70 % .. 95 %
        cols = [(0.05, 0.30), (0.29, 0.50), (0.49, 0.70), (0.69, 0.96)]
        vals = []
        for a, b in cols:
            v, cf = read_number_in_region(arr, x + int(a * bw), sy0, x + int(b * bw), sy1, light_text=light,
                                          min_h=max(5, int(0.12 * bh)))
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
        white = frame.min(axis=2) > 235
        inner = np.zeros_like(white)
        inner[y - by0:y - by0 + bh, x - bx0:x - bx0 + bw] = True
        ring = white & ~inner
        r["current"] = bool(ring.mean() > 0.02)
    conf = float(np.mean(confs)) if confs else 0.0
    if not rows:
        warnings.append("player panel not found; player list inferred from pieces")
    return rows, conf, warnings


def read_hand_bar(arr: np.ndarray, cal: Calibration) -> Tuple[Optional[List[int]], Optional[int], float]:
    """My hand from the bottom card bar: 5 resource counts and the dev-card count."""
    h, w = arr.shape[:2]
    y0 = int(0.80 * h)
    region = arr[y0:, :]
    tile_refs = np.array([cal.tile[r] for r in range(5)] + [(120, 70, 170)], dtype=np.float32)
    bg = np.array([cal.sea, (30, 40, 56), (255, 255, 255), (35, 35, 35)], dtype=np.float32)
    cls = _classify_pixels(region.reshape(-1, 3), tile_refs, bg, 45.0).reshape(region.shape[:2])
    cards: Dict[int, Tuple[int, int, int, int]] = {}
    for c in range(6):
        m = cls == c
        n, labels, stats, cents = _components(m)
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.0008 * h * w or bh < bw * 0.9:
                continue
            if best is None or area > best[4]:
                best = (x, y, bw, bh, area)
        if best is not None:
            cards[c] = best[:4]
    if len(cards) < 5:
        return None, None, 0.0
    counts = [0] * 5
    confs = []
    dev = None
    for c, (x, y, bw, bh) in cards.items():
        v, cf = read_number_in_region(arr, x, y0 + y + int(0.45 * bh), x + bw, y0 + y + bh, light_text=True,
                                      min_h=max(5, int(0.15 * bh)))
        confs.append(cf if v is not None else 0.0)
        if c < 5:
            counts[c] = v if v is not None else 0
        else:
            dev = v
    return counts, dev, float(np.mean(confs)) if confs else 0.0


def read_dice(arr: np.ndarray) -> Tuple[int, float]:
    """Sum of the pips on the two dice in the bottom-right corner (0 if not found)."""
    h, w = arr.shape[:2]
    region = arr[int(0.8 * h):, int(0.7 * w):].astype(np.int16)
    white = region.min(axis=2) > 225
    n, labels, stats, cents = _components(white)
    dice = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.0005 * h * w or abs(bw - bh) > 0.3 * max(bw, bh):
            continue
        fill = area / float(bw * bh)
        if fill < 0.5:
            continue
        dice.append((x, y, bw, bh))
    if len(dice) != 2:
        return 0, 0.0
    total = 0
    for x, y, bw, bh in dice:
        mx, my = int(0.1 * bw), int(0.1 * bh)
        face = region[y + my:y + bh - my, x + mx:x + bw - mx]
        dark = face.max(axis=2) < 80
        nn, ll, ss, cc = _components(dark)
        pips = [k for k in range(1, nn) if 0.002 * bw * bh < ss[k][4] < 0.08 * bw * bh
                and abs(ss[k][2] - ss[k][3]) <= max(2, 0.4 * max(ss[k][2], ss[k][3]))
                and ss[k][4] / float(ss[k][2] * ss[k][3]) > 0.55]
        total += len(pips)
    if 2 <= total <= 12:
        return total, 0.9
    return 0, 0.0


def read_bank_panel(arr: np.ndarray, cal: Calibration) -> Tuple[Optional[Dict[str, int]], Optional[int]]:
    h, w = arr.shape[:2]
    region = arr[: int(0.15 * h), int(0.6 * w):].astype(np.int16)
    navy = np.array([30, 40, 56], dtype=np.int16)
    m = np.abs(region - navy).sum(axis=2) < 60
    n, labels, stats, cents = _components(m)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > 0.003 * h * w and bw > 2 * bh and (best is None or area > best[4]):
            best = (x, y, bw, bh, area)
    if best is None:
        return None, None
    x, y, bw, bh, _ = best
    ox = int(0.6 * w)
    cell = bw / 6.0
    vals = []
    for k in range(6):
        v, cf = read_number_in_region(arr, ox + x + int(k * cell), y + int(0.55 * bh), ox + x + int((k + 1) * cell),
                                      y + bh, light_text=True, min_h=max(5, int(0.15 * bh)))
        vals.append(v)
    if any(v is None for v in vals[:5]):
        return None, vals[5]
    return {B.RESOURCE_NAMES[r]: int(vals[r]) for r in range(5)}, vals[5]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_image(path_or_image: Union[str, Image.Image, np.ndarray], me: Optional[str] = None,
                assume_standard: bool = True, calibration: Optional[Calibration] = None,
                read_ui: bool = True) -> ParseResult:
    cal = calibration or Calibration()
    arr = _to_rgb_array(path_or_image)
    warnings: List[str] = []
    confidence: Dict[str, float] = {}
    geom, debug, w0 = find_board(arr, cal)
    warnings.extend(w0)
    resources, res_conf, meds = classify_tiles(arr, geom, cal, assume_standard)
    confidence["hexes"] = float(np.mean(res_conf))
    robber, rob_conf = find_robber(arr, geom, cal)
    if robber < 0:
        # default: the desert
        robber = next((i for i, r in enumerate(resources) if r == B.DESERT), 0)
        warnings.append("robber not found; assuming it is on the desert")
    confidence["robber"] = rob_conf
    numbers, num_conf, w1 = read_numbers(arr, geom, resources, cal, assume_standard, robber)
    warnings.extend(w1)
    confidence["numbers"] = float(np.mean([c for i, c in enumerate(num_conf) if resources[i] != B.DESERT]) or 0.0)
    ports, port_conf, w2 = detect_ports(arr, geom, cal)
    warnings.extend(w2)
    confidence["ports"] = port_conf
    rows: List[Dict[str, Any]] = []
    hand = None
    dev = None
    dice = 0
    bank = None
    deck_left = None
    if read_ui:
        try:
            rows, panel_conf, w3 = read_player_panel(arr, cal)
            warnings.extend(w3)
            confidence["panel"] = panel_conf
        except Exception as ex:  # pragma: no cover - best effort
            warnings.append(f"player panel unreadable: {ex}")
        try:
            hand, dev, hand_conf = read_hand_bar(arr, cal)
            confidence["hand"] = hand_conf
            if hand is None:
                warnings.append("hand bar not found; your resources are unknown (use --fix me.hand=...)")
        except Exception as ex:  # pragma: no cover
            warnings.append(f"hand bar unreadable: {ex}")
        try:
            dice, dice_conf = read_dice(arr)
        except Exception:
            dice = 0
        try:
            bank, deck_left = read_bank_panel(arr, cal)
        except Exception:
            bank = None
    known_colors = [r["color"] for r in rows] or None
    buildings, roads, piece_conf = detect_pieces(arr, geom, cal, known_colors)
    confidence["pieces"] = piece_conf
    colors_seen = []
    for r in rows:
        colors_seen.append(r["color"])
    for v, (c, _, _) in buildings.items():
        if c not in colors_seen:
            colors_seen.append(c)
    for e, (c, _) in roads.items():
        if c not in colors_seen:
            colors_seen.append(c)
    if not colors_seen:
        colors_seen = ["red"]
        warnings.append("no player pieces or panel found")
    # ---- players ------------------------------------------------------------
    players = []
    row_by_color = {r["color"]: r for r in rows}
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
    if dice:
        from ..state import PHASE_MAIN
        state.phase = PHASE_MAIN
        state.dice = int(dice)
    debug["buildings"] = buildings
    debug["roads"] = roads
    debug["numbers_conf"] = num_conf
    debug["tile_medians"] = meds.tolist()
    debug["panel_rows"] = rows
    return ParseResult(parsed=parsed, state=state, confidence=confidence, warnings=warnings, debug=debug)


# ---------------------------------------------------------------------------
# Debug overlay and calibration
# ---------------------------------------------------------------------------
def draw_debug(image: Union[str, Image.Image, np.ndarray], result: ParseResult) -> Image.Image:
    img = Image.fromarray(_to_rgb_array(image)).convert("RGB")
    draw = ImageDraw.Draw(img)
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
