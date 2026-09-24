"""Number-token classifier for the Colonist.io screenshot parser (pure numpy).

The CV parser (:mod:`catanbot.vision.colonist`) locates the cream number
tokens near the hex centres and hands square crops of them to
:class:`DigitClassifier`, which answers with a probability over the ten
token values ``2, 3, 4, 5, 6, 8, 9, 10, 11, 12`` (:data:`CLASSES`).

Pipeline (:func:`preprocess`)
    crop (any size, RGB / RGBA / grayscale, PIL or numpy)
    -> grayscale + "creamness" map
    -> locate the token (bright, low-saturation blob; whole crop as fallback)
    -> square box, resized to 32x32, per-image standardised
    -> features = raw pixels (1024) ++ HOG-like orientation histograms (512)
    -> MLP 1536 -> 256 -> 128 -> 10 (ReLU, softmax)

Colonist draws 6 and 8 in red ink.  :func:`red_score` measures how red the
digit ink is and :meth:`DigitClassifier.predict_proba` folds it in as a soft
Bayesian prior (``use_red_prior=True``): red ink boosts 6 / 8, black ink
penalises them.

Training data is synthetic: :func:`make_token_image` renders one token the
way Colonist draws it (cream disc, bold black / red digits, probability
pips, tile-coloured background, rotation, blur, noise, occlusion, JPEG
artefacts) and :func:`generate_dataset` turns thousands of them into
feature matrices.  ``scripts/train_digits.py`` trains and writes
``models/digits.npz``; :meth:`DigitClassifier.load` restores it.

Only numpy and Pillow are required.
"""
from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .. import board as B

__all__ = [
    "CLASSES",
    "CLASS_INDEX",
    "IMG_SIZE",
    "NUM_FEATURES",
    "FONT_CANDIDATES",
    "DEFAULT_MODEL_PATH",
    "available_fonts",
    "make_token_image",
    "generate_dataset",
    "preprocess",
    "preprocess_batch",
    "hog_features",
    "extract_features",
    "red_score",
    "DigitClassifier",
    "assign_standard_multiset",
]

#: Token values in class-index order.
CLASSES: List[int] = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
CLASS_INDEX: Dict[int, int] = {v: i for i, v in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
RED_CLASSES = (CLASS_INDEX[6], CLASS_INDEX[8])

IMG_SIZE = 32
_HOG_CELLS_FINE = 8    # 8x8 cells of 4px
_HOG_CELLS_COARSE = 4  # 4x4 cells of 8px
_HOG_BINS = 8          # unsigned orientation bins over [0, 180)
NUM_HOG = (_HOG_CELLS_FINE ** 2 + _HOG_CELLS_COARSE ** 2) * _HOG_BINS   # 640
NUM_FEATURES = IMG_SIZE * IMG_SIZE + NUM_HOG        # 1664

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "digits.npz"
)

#: Fonts we try to use for synthetic tokens (only the ones that exist are used).
FONT_CANDIDATES: List[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]

# Hex tile colours a token can sit on (Colonist palette + a few variations).
_BACKGROUNDS: List[Tuple[int, int, int]] = [
    (40, 110, 50), (55, 125, 60), (30, 95, 45),          # wood greens
    (140, 195, 85), (150, 205, 95), (125, 180, 80),      # sheep light greens
    (232, 190, 60), (225, 180, 55), (240, 200, 80),      # wheat yellows
    (200, 95, 50), (210, 105, 55), (190, 85, 45),        # brick oranges
    (130, 130, 138), (120, 120, 128), (145, 145, 150),   # ore greys
    (222, 203, 150), (215, 195, 140),                    # desert beige (rare in practice)
    (66, 150, 225),                                      # sea blue (crop spill-over)
]
_TOKEN_CREAM = (242, 228, 196)
_TOKEN_OUTLINE = (196, 176, 136)
_INK_BLACK = (20, 20, 20)
_INK_RED = (205, 30, 30)
_ROBBER = (60, 60, 65)
_PIECE_COLOURS: List[Tuple[int, int, int]] = [
    (226, 74, 64), (48, 120, 212), (238, 140, 40), (60, 160, 80), (235, 235, 235), (140, 80, 190),
]

ArrayLike = Union[np.ndarray, Image.Image]


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
def available_fonts(candidates: Optional[Sequence[str]] = None) -> List[str]:
    """Return the subset of ``candidates`` (default :data:`FONT_CANDIDATES`) that exist on disk."""
    cands = FONT_CANDIDATES if candidates is None else list(candidates)
    return [p for p in cands if os.path.isfile(p)]


@lru_cache(maxsize=2048)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


# ---------------------------------------------------------------------------
# Synthetic token rendering
# ---------------------------------------------------------------------------
def _jitter_colour(rgb: Tuple[int, int, int], rng: np.random.Generator, amount: int = 18) -> Tuple[int, int, int]:
    d = rng.integers(-amount, amount + 1, size=3)
    return tuple(int(min(255, max(0, c + int(dc)))) for c, dc in zip(rgb, d))  # type: ignore[return-value]


def _draw_background(draw: ImageDraw.ImageDraw, w: int, h: int, base: Tuple[int, int, int],
                     rng: np.random.Generator) -> None:
    """Flat tile colour plus a few darker / lighter shapes imitating tile art."""
    draw.rectangle([0, 0, w, h], fill=base)
    n_shapes = int(rng.integers(0, 5))
    for _ in range(n_shapes):
        f = float(rng.uniform(0.6, 1.3))
        col = tuple(int(min(255, max(0, c * f))) for c in base)
        x = float(rng.uniform(0, w))
        y = float(rng.uniform(0, h))
        s = float(rng.uniform(0.08, 0.3)) * max(w, h)
        kind = int(rng.integers(0, 3))
        if kind == 0:
            draw.ellipse([x - s, y - s, x + s, y + s], fill=col)
        elif kind == 1:
            draw.polygon([(x - s, y + s), (x + s, y + s), (x, y - s)], fill=col)
        else:
            draw.rectangle([x - s, y - s * 0.4, x + s, y + s * 0.4], fill=col)


def make_token_image(
    value: int,
    rng: np.random.Generator,
    font_path: Optional[str] = None,
    radius: Optional[float] = None,
    fonts: Optional[Sequence[str]] = None,
    background: Optional[Tuple[int, int, int]] = None,
    red_ink: Optional[bool] = None,
    supersample: int = 2,
) -> Image.Image:
    """Render one synthetic Colonist-style number token as an RGB crop.

    Parameters
    ----------
    value:
        Token value (one of :data:`CLASSES`).
    rng:
        ``numpy.random.Generator`` driving every random choice.
    font_path:
        Font to use; ``None`` picks a random font from ``fonts``
        (default: :func:`available_fonts`).
    radius:
        Token radius in pixels; ``None`` draws one from ``U(12, 40)``.
    background:
        Tile colour behind the token; ``None`` picks a random one.
    red_ink:
        Force red (``True``) or black (``False``) ink; ``None`` draws red
        with 85 % probability for 6 / 8 and black otherwise.

    The returned image is a roughly square crop of ``1.0 .. 1.7`` token
    diameters with the token slightly off-centre, exactly the kind of crop
    the screenshot parser produces.  Randomised: font, font size (0.9-1.5 x
    radius), ink colour (red for 6/8 with 85 %), small offsets, rotation
    (+-8 deg), pips (omitted 30 % of the time), an outline ring, partial
    occlusion (robber / road / building), blur, noise, brightness and
    JPEG artefacts.
    """
    if value not in CLASS_INDEX:
        raise ValueError(f"unsupported token value {value!r}; expected one of {CLASSES}")
    if font_path is None:
        pool = list(fonts) if fonts else available_fonts()
        if not pool:
            raise RuntimeError("no TrueType fonts found; pass font_path explicitly")
        font_path = pool[int(rng.integers(0, len(pool)))]
    if radius is None:
        radius = float(rng.uniform(12.0, 40.0))
    ss = max(1, int(supersample))
    r = radius * ss

    # --- crop geometry ------------------------------------------------------
    side_factor = float(rng.uniform(1.0, 1.7)) if rng.random() < 0.85 else float(rng.uniform(0.92, 1.0))
    side = int(round(2 * r * side_factor))
    side = max(side, int(2 * r * 0.9))
    max_off = max(0.0, (side / 2.0 - r) * 0.9) + 0.06 * r
    cx = side / 2.0 + float(rng.uniform(-max_off, max_off))
    cy = side / 2.0 + float(rng.uniform(-max_off, max_off))

    base = background if background is not None else _BACKGROUNDS[int(rng.integers(0, len(_BACKGROUNDS)))]
    base = _jitter_colour(base, rng)
    img = Image.new("RGB", (side, side), base)
    _draw_background(ImageDraw.Draw(img), side, side, base, rng)

    # --- token layer (drawn axis-aligned, then rotated) ---------------------
    layer_half = int(math.ceil(r * 1.25)) + 2
    L = 2 * layer_half
    layer = Image.new("RGBA", (L, L), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    lc = layer_half
    cream = _jitter_colour(_TOKEN_CREAM, rng, 10)
    outline = _jitter_colour(_TOKEN_OUTLINE, rng, 14)
    ring = float(rng.uniform(0.0, 0.11)) * r if rng.random() < 0.8 else 0.0
    ld.ellipse([lc - r, lc - r, lc + r, lc + r], fill=cream, outline=outline if ring > 0.5 else None,
               width=max(1, int(round(ring))))

    is_red = (value in (6, 8) and rng.random() < 0.85) if red_ink is None else bool(red_ink)
    ink = _jitter_colour(_INK_RED if is_red else _INK_BLACK, rng, 12)
    font_size = int(round(r * float(rng.uniform(0.9, 1.5))))
    font = _font(font_path, max(6, font_size))
    text = str(value)
    show_pips = rng.random() >= 0.30
    # Anchor "mm" centres the text box; move the number up when pips are shown.
    ty = lc - (0.18 * r if show_pips else 0.02 * r) + float(rng.uniform(-0.06, 0.06)) * r
    tx = lc + float(rng.uniform(-0.06, 0.06)) * r
    ld.text((tx, ty), text, font=font, fill=ink, anchor="mm")
    if show_pips:
        pips = B.PIPS[value]
        spacing = float(rng.uniform(0.2, 0.26)) * r
        pr = float(rng.uniform(0.055, 0.085)) * r
        py = lc + float(rng.uniform(0.52, 0.66)) * r
        x0 = tx - (pips - 1) * spacing / 2.0
        for i in range(pips):
            px = x0 + i * spacing
            ld.ellipse([px - pr, py - pr, px + pr, py + pr], fill=ink)

    angle = float(rng.uniform(-8.0, 8.0))
    layer = layer.rotate(angle, resample=Image.BICUBIC, expand=False)
    img.paste(layer, (int(round(cx - layer_half)), int(round(cy - layer_half))), layer)

    # --- occlusion ----------------------------------------------------------
    d = ImageDraw.Draw(img)
    u = rng.random()
    if u < 0.12:  # robber pawn overlapping the token edge
        ang = float(rng.uniform(0, 2 * math.pi))
        dist = float(rng.uniform(1.15, 1.6)) * r
        ox, oy = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
        rr = float(rng.uniform(0.3, 0.6)) * r
        col = _jitter_colour(_ROBBER, rng, 10)
        d.ellipse([ox - rr, oy - rr * 1.3, ox + rr, oy + rr * 1.3], fill=col, outline=(30, 30, 30))
    elif u < 0.24:  # road piece crossing a corner of the crop / the token edge
        ang = float(rng.uniform(0, 2 * math.pi))
        dist = float(rng.uniform(1.0, 1.6)) * r
        ox, oy = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
        length = float(rng.uniform(1.0, 2.5)) * r
        width = float(rng.uniform(0.25, 0.4)) * r
        t = ang + math.pi / 2 + float(rng.uniform(-0.5, 0.5))
        dx, dy = math.cos(t), math.sin(t)
        nx, ny = -dy * width / 2, dx * width / 2
        pts = [(ox - dx * length / 2 + nx, oy - dy * length / 2 + ny),
               (ox + dx * length / 2 + nx, oy + dy * length / 2 + ny),
               (ox + dx * length / 2 - nx, oy + dy * length / 2 - ny),
               (ox - dx * length / 2 - nx, oy - dy * length / 2 - ny)]
        col = _PIECE_COLOURS[int(rng.integers(0, len(_PIECE_COLOURS)))]
        d.polygon(pts, fill=col, outline=(35, 35, 35))
    elif u < 0.30:  # settlement-ish blob near the edge
        ang = float(rng.uniform(0, 2 * math.pi))
        dist = float(rng.uniform(1.1, 1.6)) * r
        ox, oy = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
        s = float(rng.uniform(0.25, 0.45)) * r
        col = _PIECE_COLOURS[int(rng.integers(0, len(_PIECE_COLOURS)))]
        d.polygon([(ox - s, oy + s), (ox + s, oy + s), (ox + s, oy), (ox, oy - s), (ox - s, oy)],
                  fill=col, outline=(35, 35, 35))

    # --- downsample, blur, photometric noise, JPEG --------------------------
    if ss > 1:
        out_side = max(8, int(round(side / ss)))
        img = img.resize((out_side, out_side), Image.LANCZOS)
    blur = float(rng.uniform(0.0, 1.2))
    if blur > 0.15:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, dtype=np.float32)
    gain = float(rng.uniform(0.75, 1.2))
    bias = float(rng.uniform(-15.0, 15.0))
    arr = arr * gain + bias
    sigma = float(rng.uniform(0.0, 8.0))
    if sigma > 0.3:
        arr = arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "RGB")
    if rng.random() < 0.5:
        q = int(rng.integers(35, 96))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()
    return img


def generate_dataset(
    n: int,
    rng: Union[np.random.Generator, int],
    fonts: Optional[Sequence[str]] = None,
    keep_images: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Image.Image]]:
    """Generate ``n`` synthetic tokens with balanced classes.

    Returns ``(X, y, images)`` where ``X`` is the ``(n, NUM_FEATURES)``
    float32 feature matrix (see :func:`extract_features`), ``y`` the class
    indices and ``images`` the PIL crops when ``keep_images`` else ``[]``.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    pool = list(fonts) if fonts else available_fonts()
    if not pool:
        raise RuntimeError("no fonts available for synthetic data")
    labels = np.arange(n) % NUM_CLASSES
    rng.shuffle(labels)
    X = np.empty((n, NUM_FEATURES), dtype=np.float32)
    images: List[Image.Image] = []
    chunk = 256
    for start in range(0, n, chunk):
        crops = [make_token_image(CLASSES[int(labels[i])], rng, fonts=pool) for i in range(start, min(n, start + chunk))]
        X[start:start + len(crops)] = extract_features(preprocess_batch(crops))
        if keep_images:
            images.extend(crops)
    return X, labels.astype(np.int64), images


# ---------------------------------------------------------------------------
# Input pipeline
# ---------------------------------------------------------------------------
def _to_array(crop: ArrayLike) -> np.ndarray:
    """Any supported crop -> uint8 array ``(H, W)`` or ``(H, W, 3)``."""
    if isinstance(crop, Image.Image):
        if crop.mode in ("L", "RGB"):
            arr = np.asarray(crop)
        elif crop.mode in ("I", "F", "I;16", "1", "LA"):
            arr = np.asarray(crop.convert("L"))
        else:
            arr = np.asarray(crop.convert("RGB"))
    else:
        arr = np.asarray(crop)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] != 3:
            raise ValueError(f"unsupported crop shape {arr.shape}")
    elif arr.ndim != 2:
        raise ValueError(f"unsupported crop shape {arr.shape}")
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if a.size and a.max() <= 1.0:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"crop too small: {arr.shape}")
    return arr


def _luminance(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32)
    a = arr.astype(np.float32)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold of a float map in ``[0, 255]``."""
    hist, edges = np.histogram(values, bins=64, range=(0.0, 256.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return 128.0
    var = np.where(valid, w0 * w1 * (m0 - m1) ** 2, -1.0)
    k = int(np.argmax(var))
    return float(edges[k + 1])


def _creamness(arr: np.ndarray) -> np.ndarray:
    """Bright + unsaturated -> high.  Used to find the token disc."""
    lum = _luminance(arr)
    if arr.ndim == 2:
        return lum
    a = arr.astype(np.float32)
    sat = a.max(axis=2) - a.min(axis=2)
    return lum - 0.6 * sat


def locate_token(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Bounding box ``(x0, y0, x1, y1)`` (exclusive ends) of the token disc.

    Thresholds the creamness map (Otsu) and takes the contiguous run of
    rows / columns around the mask's centre of mass whose coverage exceeds
    a quarter of the peak coverage (this trims thin attachments such as a
    white settlement or road touching the disc).  Falls back to the whole crop when
    no convincing bright blob exists.
    """
    h, w = arr.shape[:2]
    cm = _creamness(arr)
    thr = _otsu(cm)
    mask = cm >= thr
    frac = float(mask.mean())
    if frac < 0.04 or frac > 0.97:
        return 0, 0, w, h
    rows = mask.sum(axis=1).astype(np.float32)
    cols = mask.sum(axis=0).astype(np.float32)
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    cyi = int(round(float((rows * ys).sum() / max(rows.sum(), 1.0))))
    cxi = int(round(float((cols * xs).sum() / max(cols.sum(), 1.0))))
    cyi = min(max(cyi, 0), h - 1)
    cxi = min(max(cxi, 0), w - 1)

    def run(profile: np.ndarray, centre: int, n: int) -> Tuple[int, int]:
        peak = float(profile.max())
        cut = 0.25 * peak
        lo = centre
        while lo > 0 and profile[lo - 1] > cut:
            lo -= 1
        hi = centre
        while hi < n - 1 and profile[hi + 1] > cut:
            hi += 1
        return lo, hi + 1

    y0, y1 = run(rows, cyi, h)
    x0, x1 = run(cols, cxi, w)
    if (y1 - y0) < 6 or (x1 - x0) < 6:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _square_box(box: Tuple[int, int, int, int], margin: float = 0.04) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1.0 + 2 * margin)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = side / 2.0
    return int(math.floor(cx - half)), int(math.floor(cy - half)), int(math.ceil(cx + half)), int(math.ceil(cy + half))


def _crop_pad(gray: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``gray`` to ``box`` padding with edge values where the box leaves the image."""
    h, w = gray.shape
    x0, y0, x1, y1 = box
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        gray = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    return gray[y0:y1, x0:x1]


def preprocess(crop: ArrayLike) -> np.ndarray:
    """Crop -> standardised ``(32, 32)`` float32 image (ink positive).

    Grayscale, token localisation (:func:`locate_token`), square crop with
    edge padding, Lanczos resize, inversion (ink bright) and per-image
    standardisation to zero mean / unit variance.
    """
    arr = _to_array(crop)
    gray = _luminance(arr)
    box = _square_box(locate_token(arr))
    patch = _crop_pad(gray, box)
    pil = Image.fromarray(np.clip(patch + 0.5, 0, 255).astype(np.uint8), "L")
    small = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.float32)
    x = 1.0 - small / 255.0
    x -= x.mean()
    x /= (x.std() + 1e-3)
    return x.astype(np.float32)


def preprocess_batch(crops: Iterable[ArrayLike]) -> np.ndarray:
    """Stack :func:`preprocess` over an iterable of crops -> ``(N, 32, 32)``."""
    items = [preprocess(c) for c in crops]
    if not items:
        return np.empty((0, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    return np.stack(items, axis=0)


def hog_features(imgs: np.ndarray) -> np.ndarray:
    """HOG-like gradient orientation histograms for ``(N, 32, 32)`` images.

    8 unsigned orientation bins (linearly interpolated) per cell at two
    scales, 4x4-pixel cells (8x8 grid) and 8x8-pixel cells (4x4 grid),
    gradient magnitude weighted, each scale L2-normalised per image
    -> ``(N, 640)`` float32.  Fully vectorised numpy.
    """
    imgs = np.asarray(imgs, dtype=np.float32)
    if imgs.ndim == 2:
        imgs = imgs[None]
    n = imgs.shape[0]
    gx = np.zeros_like(imgs)
    gy = np.zeros_like(imgs)
    gx[:, :, 1:-1] = imgs[:, :, 2:] - imgs[:, :, :-2]
    gy[:, 1:-1, :] = imgs[:, 2:, :] - imgs[:, :-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)                      # (-pi, pi]
    ang = np.mod(ang, np.pi)                      # unsigned [0, pi)
    pos = ang * (_HOG_BINS / np.pi)               # continuous bin position
    b0 = np.floor(pos).astype(np.int64)
    w1 = (pos - b0).astype(np.float32)            # linear interpolation between bins
    w0 = 1.0 - w1
    b0 = np.mod(b0, _HOG_BINS)
    b1 = np.mod(b0 + 1, _HOG_BINS)
    hist = np.zeros((n, IMG_SIZE, IMG_SIZE, _HOG_BINS), dtype=np.float32)
    ii = np.arange(n)[:, None, None]
    yy = np.arange(IMG_SIZE)[None, :, None]
    xx = np.arange(IMG_SIZE)[None, None, :]
    hist[ii, yy, xx, b0] += mag * w0
    hist[ii, yy, xx, b1] += mag * w1
    feats = []
    for cells_per_side in (_HOG_CELLS_FINE, _HOG_CELLS_COARSE):
        px = IMG_SIZE // cells_per_side
        cells = hist.reshape(n, cells_per_side, px, cells_per_side, px, _HOG_BINS).sum(axis=(2, 4))
        feat = cells.reshape(n, -1)
        norm = np.sqrt((feat * feat).sum(axis=1, keepdims=True)) + 1e-6
        feats.append(feat / norm * float(cells_per_side))
    return np.concatenate(feats, axis=1).astype(np.float32)


def extract_features(imgs: np.ndarray) -> np.ndarray:
    """``(N, 32, 32)`` preprocessed images -> ``(N, NUM_FEATURES)`` features."""
    imgs = np.asarray(imgs, dtype=np.float32)
    if imgs.ndim == 2:
        imgs = imgs[None]
    raw = imgs.reshape(imgs.shape[0], -1)
    return np.concatenate([raw, hog_features(imgs)], axis=1).astype(np.float32)


def red_score(crop: ArrayLike) -> float:
    """How red the digit ink is, in ``[0, 1]`` (0.5 when undecidable).

    Looks at dark ("ink") pixels inside the central 75 % of the located
    token disc and returns the mean of ``clip((R - max(G, B)) / 128, 0, 1)``
    over them.  Black ink scores ~0, Colonist's red 6 / 8 ink ~0.9-1.0.
    Grayscale crops (no colour information) return 0.5.
    """
    arr = _to_array(crop)
    if arr.ndim == 2:
        return 0.5
    x0, y0, x1, y1 = locate_token(arr)
    sub = arr[y0:y1, x0:x1].astype(np.float32)
    h, w = sub.shape[:2]
    if h < 4 or w < 4:
        return 0.5
    lum = 0.299 * sub[:, :, 0] + 0.587 * sub[:, :, 1] + 0.114 * sub[:, :, 2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = 0.75 * min(h, w) / 2.0
    inside = ((yy - cy) ** 2 + (xx - cx) ** 2) <= rad * rad
    if not inside.any():
        return 0.5
    bright = float(np.percentile(lum[inside], 80))
    ink = inside & (lum < bright - 55.0)
    if ink.sum() < 4:
        return 0.5
    px = sub[ink]
    redness = np.clip((px[:, 0] - np.maximum(px[:, 1], px[:, 2])) / 128.0, 0.0, 1.0)
    return float(redness.mean())


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class DigitClassifier:
    """Numpy MLP over :func:`extract_features` with an optional red-ink prior.

    Parameters
    ----------
    hidden:
        Hidden layer widths (default ``(256, 128)``).
    seed:
        Weight initialisation seed.
    use_red_prior:
        Default for :meth:`predict_proba`'s ``use_red_prior`` argument.
    """

    #: P(red ink | token is 6 or 8) and P(red ink | other token) used by the prior.
    P_RED_GIVEN_68 = 0.85
    P_RED_GIVEN_OTHER = 0.03

    def __init__(self, hidden: Sequence[int] = (256, 128), seed: int = 0, use_red_prior: bool = True) -> None:
        self.hidden: Tuple[int, ...] = tuple(int(h) for h in hidden)
        self.use_red_prior = use_red_prior
        self.feat_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        self.feat_std = np.ones(NUM_FEATURES, dtype=np.float32)
        rng = np.random.default_rng(seed)
        sizes = (NUM_FEATURES, *self.hidden, NUM_CLASSES)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.weights.append((rng.standard_normal((a, b)) * math.sqrt(2.0 / a)).astype(np.float32))
            self.biases.append(np.zeros(b, dtype=np.float32))
        self.history: Dict[str, List[float]] = {}

    # --- forward / backward --------------------------------------------------
    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        acts = [X]
        h = X
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < n_layers - 1 else z
            acts.append(h)
        return h, acts

    def _forward_train(self, X: np.ndarray, dropout: float, rng: np.random.Generator):
        """Forward pass with inverted dropout on hidden activations (training only)."""
        acts = [X]
        masks: List[Optional[np.ndarray]] = [None]
        h = X
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            if i < n_layers - 1:
                h = np.maximum(z, 0.0)
                if dropout > 0:
                    mask = (rng.random(h.shape, dtype=np.float32) >= dropout).astype(np.float32) / (1 - dropout)
                    h = h * mask
                    masks.append(mask)
                else:
                    masks.append(None)
            else:
                h = z
                masks.append(None)
            acts.append(h)
        return h, acts, masks

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.feat_mean) / self.feat_std).astype(np.float32)

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Raw network outputs for a feature matrix ``(N, NUM_FEATURES)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None]
        return self._forward(self._standardise(X))[0]

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        """Softmax over :meth:`logits` (no red prior)."""
        return _softmax(self.logits(X).astype(np.float64)).astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        seed: int = 0,
        dropout: float = 0.3,
        input_dropout: float = 0.1,
        log=None,
    ) -> Dict[str, List[float]]:
        """Train with Adam on softmax cross-entropy.

        ``X`` is ``(N, NUM_FEATURES)`` (from :func:`extract_features`), ``y``
        class indices.  Feature standardisation statistics are fitted on
        ``X``.  The learning rate decays with cosine annealing; inverted
        dropout (``dropout`` on hidden units, ``input_dropout`` on features)
        regularises training and is disabled at inference.  ``log`` is an
        optional callable receiving one line of text per epoch.  Returns the
        training history (``loss``, ``train_acc``, ``val_acc``).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = X.shape[0]
        self.feat_mean = X.mean(axis=0).astype(np.float32)
        self.feat_std = (X.std(axis=0) + 1e-3).astype(np.float32)
        Xs = self._standardise(X)
        Xv = self._standardise(np.asarray(X_val, dtype=np.float32)) if X_val is not None else None
        rng = np.random.default_rng(seed)
        params = self.weights + self.biases
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        hist: Dict[str, List[float]] = {"loss": [], "train_acc": [], "val_acc": []}
        n_layers = len(self.weights)
        steps_total = max(1, epochs * math.ceil(n / batch_size))
        for ep in range(epochs):
            perm = rng.permutation(n)
            total_loss = 0.0
            correct = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = Xs[idx], y[idx]
                bs = xb.shape[0]
                if input_dropout > 0:
                    xb = xb * (rng.random(xb.shape, dtype=np.float32) >= input_dropout) / (1 - input_dropout)
                out, acts, masks = self._forward_train(xb, dropout, rng)
                p = _softmax(out)
                total_loss += float(-np.log(p[np.arange(bs), yb] + 1e-12).sum())
                correct += int((p.argmax(axis=1) == yb).sum())
                # backward
                grad = p.copy()
                grad[np.arange(bs), yb] -= 1.0
                grad /= bs
                grads_w: List[np.ndarray] = [np.empty(0)] * n_layers
                grads_b: List[np.ndarray] = [np.empty(0)] * n_layers
                for i in range(n_layers - 1, -1, -1):
                    grads_w[i] = acts[i].T @ grad + weight_decay * self.weights[i]
                    grads_b[i] = grad.sum(axis=0)
                    if i > 0:
                        grad = (grad @ self.weights[i].T) * (acts[i] > 0)
                        if masks[i] is not None:
                            grad = grad * masks[i]
                # Adam with cosine lr schedule
                step += 1
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * step / steps_total))
                grads = grads_w + grads_b
                for k, (par, g) in enumerate(zip(params, grads)):
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * (g * g)
                    mhat = m[k] / (1 - b1 ** step)
                    vhat = v[k] / (1 - b2 ** step)
                    par -= (cur_lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
            hist["loss"].append(total_loss / n)
            hist["train_acc"].append(correct / n)
            if Xv is not None and y_val is not None:
                pv = self._forward(Xv)[0].argmax(axis=1)
                hist["val_acc"].append(float((pv == np.asarray(y_val)).mean()))
            if log is not None:
                msg = f"epoch {ep + 1:3d}/{epochs}  loss {hist['loss'][-1]:.4f}  train_acc {hist['train_acc'][-1]:.4f}"
                if hist["val_acc"]:
                    msg += f"  val_acc {hist['val_acc'][-1]:.4f}"
                log(msg)
        self.history = hist
        return hist

    # --- inference on crops ---------------------------------------------------
    def red_prior(self, score: float) -> np.ndarray:
        """Likelihood vector over classes for a given :func:`red_score`.

        The score is mapped to ``P(red ink)`` with a logistic centred at
        0.35; the per-class likelihood mixes :attr:`P_RED_GIVEN_68` /
        :attr:`P_RED_GIVEN_OTHER` accordingly.  Multiplying the network's
        posterior by it and renormalising applies the prior.
        """
        p_red = 1.0 / (1.0 + math.exp(-10.0 * (score - 0.35)))
        like = np.full(NUM_CLASSES, self.P_RED_GIVEN_OTHER * p_red + (1 - self.P_RED_GIVEN_OTHER) * (1 - p_red))
        for k in RED_CLASSES:
            like[k] = self.P_RED_GIVEN_68 * p_red + (1 - self.P_RED_GIVEN_68) * (1 - p_red)
        return like.astype(np.float32)

    def predict_proba(self, crops: Sequence[ArrayLike], use_red_prior: Optional[bool] = None) -> np.ndarray:
        """Class probabilities ``(N, 10)`` for a list of crops (PIL or numpy).

        ``use_red_prior`` (default: the instance flag) folds :func:`red_score`
        into the posterior; grayscale crops carry no colour and are left
        untouched.
        """
        crops = list(crops)
        if not crops:
            return np.empty((0, NUM_CLASSES), dtype=np.float32)
        use_prior = self.use_red_prior if use_red_prior is None else bool(use_red_prior)
        X = extract_features(preprocess_batch(crops))
        p = self.predict_proba_features(X).astype(np.float64)
        if use_prior:
            for i, c in enumerate(crops):
                s = red_score(c)
                if s == 0.5:
                    continue
                q = p[i] * self.red_prior(s)
                p[i] = q / max(q.sum(), 1e-12)
        return p.astype(np.float32)

    def predict(self, crops: Sequence[ArrayLike], use_red_prior: Optional[bool] = None) -> List[int]:
        """Token values (``2..12``) for a list of crops."""
        p = self.predict_proba(crops, use_red_prior=use_red_prior)
        return [CLASSES[int(k)] for k in p.argmax(axis=1)]

    def predict_index(self, crops: Sequence[ArrayLike], use_red_prior: Optional[bool] = None) -> np.ndarray:
        """Class indices (into :data:`CLASSES`) for a list of crops."""
        return self.predict_proba(crops, use_red_prior=use_red_prior).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on a feature matrix / label vector."""
        return float((self.predict_proba_features(X).argmax(axis=1) == np.asarray(y)).mean())

    # --- persistence ------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save weights, feature statistics and configuration to ``path`` (``.npz``)."""
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        payload: Dict[str, np.ndarray] = {
            "classes": np.asarray(CLASSES, dtype=np.int64),
            "hidden": np.asarray(self.hidden, dtype=np.int64),
            "img_size": np.asarray(IMG_SIZE, dtype=np.int64),
            "num_features": np.asarray(NUM_FEATURES, dtype=np.int64),
            "use_red_prior": np.asarray(int(self.use_red_prior), dtype=np.int64),
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
        }
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{i}"] = W
            payload[f"b{i}"] = b
        np.savez(path, **payload)

    @staticmethod
    def load(path: Optional[str] = None) -> "DigitClassifier":
        """Restore a classifier saved with :meth:`save` (default :data:`DEFAULT_MODEL_PATH`)."""
        path = DEFAULT_MODEL_PATH if path is None else path
        with np.load(path) as z:
            classes = [int(c) for c in z["classes"]]
            if classes != CLASSES:
                raise ValueError(f"model classes {classes} do not match {CLASSES}")
            if int(z["num_features"]) != NUM_FEATURES:
                raise ValueError("model was trained with a different feature layout")
            hidden = tuple(int(h) for h in z["hidden"])
            clf = DigitClassifier(hidden=hidden, use_red_prior=bool(int(z["use_red_prior"])))
            clf.feat_mean = z["feat_mean"].astype(np.float32)
            clf.feat_std = z["feat_std"].astype(np.float32)
            clf.weights = [z[f"W{i}"].astype(np.float32) for i in range(len(hidden) + 1)]
            clf.biases = [z[f"b{i}"].astype(np.float32) for i in range(len(hidden) + 1)]
        return clf


# ---------------------------------------------------------------------------
# Multiset-constrained decoding (used by the board parser)
# ---------------------------------------------------------------------------
def assign_standard_multiset(probs: np.ndarray, numbers: Sequence[int] = B.STANDARD_NUMBERS) -> List[int]:
    """Assign token values to ``probs`` rows so that the multiset matches ``numbers``.

    ``probs`` is ``(N, 10)`` in :data:`CLASSES` order.  When ``N`` equals
    ``len(numbers)`` every value is used exactly once (greedy by confidence
    followed by pairwise-swap hill climbing on total log-probability);
    when ``N`` is smaller the counts act as upper bounds, and when larger
    the extra rows get their unconstrained argmax.
    """
    P = np.asarray(probs, dtype=np.float64)
    n = P.shape[0]
    logp = np.log(P + 1e-9)
    quota = np.zeros(NUM_CLASSES, dtype=np.int64)
    for v in numbers:
        if v in CLASS_INDEX:
            quota[CLASS_INDEX[v]] += 1
    assign = -np.ones(n, dtype=np.int64)
    if n > int(quota.sum()):
        assign[:] = P.argmax(axis=1)
        return [CLASSES[int(k)] for k in assign]
    order = np.argsort(-P.max(axis=1))
    for i in order:
        for k in np.argsort(-logp[i]):
            if quota[k] > 0:
                assign[i] = k
                quota[k] -= 1
                break
    # pairwise swap improvement
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = assign[i], assign[j]
                if a == b:
                    continue
                if logp[i, b] + logp[j, a] > logp[i, a] + logp[j, b] + 1e-12:
                    assign[i], assign[j] = b, a
                    improved = True
    return [CLASSES[int(k)] for k in assign]
