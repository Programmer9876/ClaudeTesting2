"""Synthetic Colonist.io-style screenshot renderer (PIL only).

:func:`render_state` draws a :class:`~catanbot.state.GameState` the way
Colonist.io shows a game: blue sea, pointy-top hex tiles in Colonist colours
with simple tile art, cream number tokens (red 6 / 8) with probability pips,
the robber, coloured settlements / cities / roads, port icons off the coast,
a player panel on the left, a hand bar for "me" at the bottom, a bank panel
top-right and two dice bottom-right.

The board geometry is derived from the image size (the board spans roughly
70 % of the image height) and, with ``jitter=True``, randomised
deterministically from ``seed``: centre offset, hex size, per-colour jitter
and an optional JPEG re-compression.  :func:`board_geometry` and the
``pixel_of_*`` helpers expose the exact geometry so tests and the CV parser
can compare against ground truth.

Everything is rendered at ``style.supersample`` times the target size and
downsampled with a Lanczos filter for anti-aliased edges.
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from .. import board as B
from ..state import GameState, Player
from .schema import port_edge_pairs

__all__ = [
    "ColonistStyle",
    "PLAYER_RGB",
    "DEFAULT_FONT_PATH",
    "board_geometry",
    "pixel_of_hex",
    "pixel_of_vertex",
    "pixel_of_edge",
    "render_state",
    "render_to_file",
]

RGB = Tuple[int, int, int]
Geometry = Dict[str, float]

DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

#: Player colours as Colonist.io draws them (keys match ``state.PLAYER_COLORS``).
PLAYER_RGB: Dict[str, RGB] = {
    "red": (226, 74, 64),
    "blue": (48, 120, 212),
    "orange": (238, 140, 40),
    "green": (60, 160, 80),
    "white": (235, 235, 235),
    "purple": (140, 80, 190),
    "brown": (120, 80, 50),
    "pink": (230, 120, 180),
}
_UNKNOWN_PLAYER_RGB: RGB = (150, 150, 150)


@dataclass
class ColonistStyle:
    """Colours and rendering knobs imitating Colonist.io's look."""

    sea: RGB = (66, 150, 225)
    tile: Dict[int, RGB] = field(default_factory=lambda: {
        B.WOOD: (40, 110, 50),
        B.SHEEP: (140, 195, 85),
        B.WHEAT: (232, 190, 60),
        B.BRICK: (200, 95, 50),
        B.ORE: (130, 130, 138),
        B.DESERT: (222, 203, 150),
    })
    token: RGB = (242, 228, 196)
    token_outline: RGB = (196, 176, 136)
    number: RGB = (20, 20, 20)
    number_red: RGB = (205, 30, 30)
    robber: RGB = (60, 60, 65)
    outline: RGB = (35, 35, 35)
    port: RGB = (228, 208, 160)
    port_outline: RGB = (150, 125, 80)
    panel: RGB = (30, 40, 56)
    panel_text: RGB = (255, 255, 255)
    card_text: RGB = (255, 255, 255)
    dice: RGB = (245, 245, 245)
    dice_pip: RGB = (30, 30, 30)
    badge: RGB = (240, 200, 60)
    tile_gap: float = 0.045          # fraction of the hex size left as sea between tiles
    token_radius: float = 0.34       # fraction of the hex size
    supersample: int = 2             # render scale before Lanczos downsampling
    font_path: str = DEFAULT_FONT_PATH
    jpeg_quality: Optional[int] = None   # fixed JPEG quality when re-compressing (None = seeded)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
_BOARD_HEIGHT_UNITS = 8.0   # vertex y range is -4..4 in hex-size units
_BOARD_FRACTION = 0.70      # of the image height


def board_geometry(size: Tuple[int, int] = (1280, 800), seed: int = 0, jitter: bool = True) -> Geometry:
    """Board centre and hex size (pixels) for an image of ``size``.

    Without jitter the board centre sits slightly right of the image centre
    (leaving room for the player panel) and slightly above the vertical
    centre (leaving room for the hand bar); the hex size is such that the
    board's vertical extent is 70 % of the height.  With jitter the centre
    moves by up to +-4 % of the image size and the hex size varies by
    +-8 %, deterministically from ``seed``.

    Returns ``{"cx", "cy", "hex_size", "width", "height"}``.
    """
    w, h = int(size[0]), int(size[1])
    hex_size = _BOARD_FRACTION * h / _BOARD_HEIGHT_UNITS
    cx = 0.53 * w
    cy = 0.46 * h
    if jitter:
        rng = random.Random(seed * 7919 + 17)
        cx += rng.uniform(-0.04, 0.04) * w
        cy += rng.uniform(-0.04, 0.04) * h
        hex_size *= 1.0 + rng.uniform(-0.08, 0.08)
    return {"cx": cx, "cy": cy, "hex_size": hex_size, "width": float(w), "height": float(h)}


def pixel_of_hex(h: int, geom: Geometry) -> Tuple[float, float]:
    """Pixel centre of hex ``h`` under ``geom``."""
    x, y = B.HEX_CENTERS[h]
    return (geom["cx"] + x * geom["hex_size"], geom["cy"] + y * geom["hex_size"])


def pixel_of_vertex(v: int, geom: Geometry) -> Tuple[float, float]:
    """Pixel position of vertex ``v`` under ``geom``."""
    x, y = B.VERTEX_POS[v]
    return (geom["cx"] + x * geom["hex_size"], geom["cy"] + y * geom["hex_size"])


def pixel_of_edge(e: int, geom: Geometry) -> Tuple[float, float]:
    """Pixel midpoint of edge ``e`` under ``geom``."""
    x, y = B.EDGE_POS[e]
    return (geom["cx"] + x * geom["hex_size"], geom["cy"] + y * geom["hex_size"])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
_FONT_CACHE: Dict[Tuple[str, int], ImageFont.ImageFont] = {}


def _font(path: str, px: float) -> ImageFont.ImageFont:
    """Bold TrueType font of ``px`` pixels, falling back to PIL's default font."""
    px_i = max(6, int(round(px)))
    key = (path, px_i)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(path, px_i)
        except (OSError, ValueError):
            try:
                f = ImageFont.load_default(size=px_i)
            except TypeError:  # Pillow < 10.1
                f = ImageFont.load_default()
        _FONT_CACHE[key] = f
    return f


def _clamp(c: float) -> int:
    return 0 if c < 0 else 255 if c > 255 else int(c)


def _jitter_rgb(c: RGB, rng: random.Random, amount: int = 8) -> RGB:
    return (_clamp(c[0] + rng.randint(-amount, amount)),
            _clamp(c[1] + rng.randint(-amount, amount)),
            _clamp(c[2] + rng.randint(-amount, amount)))


def _shade(c: RGB, factor: float) -> RGB:
    return (_clamp(c[0] * factor), _clamp(c[1] * factor), _clamp(c[2] * factor))


def _jittered_style(style: ColonistStyle, rng: random.Random) -> ColonistStyle:
    """Copy of ``style`` with every colour perturbed by +-8 per channel."""
    st = replace(style, tile={k: _jitter_rgb(v, rng) for k, v in style.tile.items()})
    for name in ("sea", "token", "token_outline", "robber", "port", "panel", "dice", "badge"):
        setattr(st, name, _jitter_rgb(getattr(style, name), rng))
    return st


class _Canvas:
    """Thin ImageDraw wrapper working in *final* pixel coordinates on a supersampled image."""

    def __init__(self, width: int, height: int, ss: int, background: RGB, font_path: str):
        self.ss = ss
        self.img = Image.new("RGB", (width * ss, height * ss), background)
        self.d = ImageDraw.Draw(self.img)
        self.font_path = font_path

    # coordinate scaling -------------------------------------------------
    def _pt(self, p: Sequence[float]) -> Tuple[float, float]:
        return (p[0] * self.ss, p[1] * self.ss)

    def _pts(self, pts: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
        s = self.ss
        return [(p[0] * s, p[1] * s) for p in pts]

    # primitives ---------------------------------------------------------
    def polygon(self, pts: Sequence[Sequence[float]], fill: Optional[RGB], outline: Optional[RGB] = None,
                width: float = 0.0) -> None:
        self.d.polygon(self._pts(pts), fill=fill, outline=outline, width=max(0, int(round(width * self.ss))))

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, fill: Optional[RGB],
                outline: Optional[RGB] = None, width: float = 0.0) -> None:
        s = self.ss
        box = [(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s]
        self.d.ellipse(box, fill=fill, outline=outline, width=max(0, int(round(width * s))))

    def circle(self, cx: float, cy: float, r: float, fill: Optional[RGB], outline: Optional[RGB] = None,
               width: float = 0.0) -> None:
        self.ellipse(cx, cy, r, r, fill, outline, width)

    def line(self, pts: Sequence[Sequence[float]], fill: RGB, width: float = 1.0) -> None:
        self.d.line(self._pts(pts), fill=fill, width=max(1, int(round(width * self.ss))))

    def rounded_rect(self, x0: float, y0: float, x1: float, y1: float, radius: float, fill: Optional[RGB],
                     outline: Optional[RGB] = None, width: float = 0.0) -> None:
        s = self.ss
        self.d.rounded_rectangle([x0 * s, y0 * s, x1 * s, y1 * s], radius=max(0.0, radius * s), fill=fill,
                                 outline=outline, width=max(0, int(round(width * s))))

    def bar(self, p0: Sequence[float], p1: Sequence[float], thickness: float, fill: RGB,
            outline: Optional[RGB] = None, outline_w: float = 0.0) -> None:
        """Rounded bar (capsule) from p0 to p1."""
        if outline is not None and outline_w > 0:
            self._capsule(p0, p1, thickness + 2 * outline_w, outline)
        self._capsule(p0, p1, thickness, fill)

    def _capsule(self, p0: Sequence[float], p1: Sequence[float], thickness: float, fill: RGB) -> None:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length * thickness / 2.0, dx / length * thickness / 2.0
        quad = [(p0[0] + nx, p0[1] + ny), (p1[0] + nx, p1[1] + ny),
                (p1[0] - nx, p1[1] - ny), (p0[0] - nx, p0[1] - ny)]
        self.polygon(quad, fill)
        r = thickness / 2.0
        self.circle(p0[0], p0[1], r, fill)
        self.circle(p1[0], p1[1], r, fill)

    def text(self, cx: float, cy: float, s: str, px: float, fill: RGB, anchor: str = "mm",
             stroke: Optional[RGB] = None, stroke_w: float = 0.0) -> None:
        """Text with its ``anchor`` ("mm" centre, "lm" left-middle, "rm" right-middle) at (cx, cy)."""
        font = _font(self.font_path, px * self.ss)
        x0, y0, x1, y1 = self.d.textbbox((0, 0), s, font=font)
        tw, th = x1 - x0, y1 - y0
        X, Y = cx * self.ss, cy * self.ss
        if anchor[0] == "m":
            X -= tw / 2.0 + x0
        elif anchor[0] == "r":
            X -= tw + x0
        else:
            X -= x0
        Y -= th / 2.0 + y0
        sw = int(round(stroke_w * self.ss)) if stroke is not None else 0
        self.d.text((X, Y), s, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke)

    def finish(self, size: Tuple[int, int]) -> Image.Image:
        if self.ss == 1:
            return self.img
        return self.img.resize(size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Board pieces
# ---------------------------------------------------------------------------
def _hex_polygon(cx: float, cy: float, size: float) -> List[Tuple[float, float]]:
    return [B.hex_corner(cx, cy, k, size) for k in range(6)]


def _draw_tile_art(cv: _Canvas, res: int, cx: float, cy: float, hs: float, style: ColonistStyle,
                   rng: random.Random) -> None:
    """Simple tile decorations placed around the token in a ring."""
    base = style.tile[res]
    ring = 0.62 * hs
    angles = [30, 90, 150, 210, 270, 330]
    if res == B.WOOD:
        dark = _shade(base, 0.6)
        trunk = (90, 60, 30)
        for a in angles[::2] + [angles[1]]:
            x = cx + ring * math.cos(math.radians(a)) + rng.uniform(-0.05, 0.05) * hs
            y = cy + ring * math.sin(math.radians(a)) + rng.uniform(-0.05, 0.05) * hs
            w, h = 0.16 * hs, 0.26 * hs
            cv.polygon([(x - w / 2, y + h * 0.35), (x + w / 2, y + h * 0.35), (x, y - h * 0.65)], dark)
            cv.polygon([(x - w * 0.12, y + h * 0.35), (x + w * 0.12, y + h * 0.35),
                        (x + w * 0.12, y + h * 0.6), (x - w * 0.12, y + h * 0.6)], trunk)
    elif res == B.SHEEP:
        for a in (60, 180, 300):
            x = cx + ring * math.cos(math.radians(a)) + rng.uniform(-0.05, 0.05) * hs
            y = cy + ring * math.sin(math.radians(a))
            cv.ellipse(x, y, 0.12 * hs, 0.08 * hs, (245, 245, 240))
            cv.circle(x + 0.1 * hs, y - 0.02 * hs, 0.035 * hs, (50, 50, 50))
    elif res == B.WHEAT:
        stroke = _shade(base, 0.72)
        for a in angles:
            x = cx + ring * math.cos(math.radians(a))
            y = cy + ring * math.sin(math.radians(a))
            for k in (-1, 0, 1):
                xx = x + k * 0.06 * hs
                cv.line([(xx, y + 0.14 * hs), (xx + 0.03 * hs, y - 0.14 * hs)], stroke, 0.018 * hs)
    elif res == B.BRICK:
        brick = _shade(base, 0.7)
        for a in angles[1::2]:
            x = cx + ring * math.cos(math.radians(a))
            y = cy + ring * math.sin(math.radians(a))
            for row in range(2):
                off = 0.07 * hs * (row % 2)
                for col in range(2):
                    x0 = x - 0.15 * hs + col * 0.16 * hs + off
                    y0 = y - 0.08 * hs + row * 0.09 * hs
                    cv.polygon([(x0, y0), (x0 + 0.13 * hs, y0), (x0 + 0.13 * hs, y0 + 0.07 * hs),
                                (x0, y0 + 0.07 * hs)], brick)
    elif res == B.ORE:
        dark = _shade(base, 0.55)
        snow = (225, 225, 230)
        for a in angles[::2]:
            x = cx + ring * math.cos(math.radians(a))
            y = cy + ring * math.sin(math.radians(a))
            w, h = 0.34 * hs, 0.28 * hs
            cv.polygon([(x - w / 2, y + h / 2), (x + w / 2, y + h / 2), (x, y - h / 2)], dark)
            cv.polygon([(x - w * 0.12, y - h * 0.15), (x + w * 0.12, y - h * 0.15), (x, y - h / 2)], snow)
    else:  # desert dunes
        dune = _shade(base, 0.88)
        for a in (30, 150, 270):
            x = cx + ring * math.cos(math.radians(a))
            y = cy + ring * math.sin(math.radians(a))
            cv.line([(x - 0.18 * hs, y + 0.03 * hs), (x - 0.06 * hs, y - 0.04 * hs),
                     (x + 0.06 * hs, y + 0.03 * hs), (x + 0.18 * hs, y - 0.04 * hs)], dune, 0.02 * hs)


def _draw_token(cv: _Canvas, cx: float, cy: float, hs: float, number: int, style: ColonistStyle) -> None:
    r = style.token_radius * hs
    cv.circle(cx, cy, r, style.token, style.token_outline, 0.035 * hs)
    color = style.number_red if number in (6, 8) else style.number
    cv.text(cx, cy - 0.06 * hs, str(number), 0.40 * hs, color)
    pips = B.PIPS.get(number, 0)
    if pips:
        spacing = 0.085 * hs
        x0 = cx - (pips - 1) * spacing / 2.0
        py = cy + 0.2 * hs
        for i in range(pips):
            cv.circle(x0 + i * spacing, py, 0.028 * hs, color)


def _draw_robber(cv: _Canvas, cx: float, cy: float, hs: float, style: ColonistStyle) -> None:
    """Pawn-shaped robber beside (right of / below) the token."""
    x = cx + 0.47 * hs
    y = cy + 0.12 * hs
    fill = style.robber
    outline = style.outline
    cv.ellipse(x, y + 0.2 * hs, 0.17 * hs, 0.07 * hs, fill, outline, 0.012 * hs)           # base
    cv.polygon([(x - 0.12 * hs, y + 0.2 * hs), (x + 0.12 * hs, y + 0.2 * hs),
                (x + 0.06 * hs, y - 0.08 * hs), (x - 0.06 * hs, y - 0.08 * hs)], fill, outline, 0.012 * hs)
    cv.circle(x, y - 0.17 * hs, 0.1 * hs, fill, outline, 0.012 * hs)                       # head


def _draw_settlement(cv: _Canvas, x: float, y: float, hs: float, color: RGB, style: ColonistStyle) -> None:
    w, h = 0.30 * hs, 0.30 * hs
    pts = [(x - w / 2, y + h / 2), (x + w / 2, y + h / 2), (x + w / 2, y - h * 0.05),
           (x, y - h / 2), (x - w / 2, y - h * 0.05)]
    cv.polygon(pts, color, style.outline, 0.035 * hs)


def _draw_city(cv: _Canvas, x: float, y: float, hs: float, color: RGB, style: ColonistStyle) -> None:
    w, h = 0.46 * hs, 0.40 * hs
    # house body (right) with a tower (left)
    pts = [(x - w / 2, y + h / 2), (x + w / 2, y + h / 2), (x + w / 2, y - h * 0.05),
           (x + w * 0.18, y - h * 0.05), (x + w * 0.18, y - h * 0.05),
           (x - w * 0.12, y - h * 0.05), (x - w * 0.12, y - h * 0.5),
           (x - w * 0.31, y - h * 0.75), (x - w / 2, y - h * 0.5)]
    cv.polygon(pts, color, style.outline, 0.035 * hs)
    cv.polygon([(x + w * 0.18, y - h * 0.05), (x + w / 2, y - h * 0.05), (x + w * 0.34, y - h * 0.32)],
               color, style.outline, 0.03 * hs)


def _draw_road(cv: _Canvas, e: int, geom: Geometry, color: RGB, style: ColonistStyle) -> None:
    a, b = B.EDGE_VERTICES[e]
    (x1, y1), (x2, y2) = pixel_of_vertex(a, geom), pixel_of_vertex(b, geom)
    hs = geom["hex_size"]
    # shorten so roads do not cover the buildings at the vertices
    t = 0.2
    p0 = (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    p1 = (x2 + (x1 - x2) * t, y2 + (y1 - y2) * t)
    cv.bar(p0, p1, 0.17 * hs, color, style.outline, 0.03 * hs)


def _port_entries(ports: Dict[int, int]) -> List[Tuple[int, int]]:
    """``{vertex: type}`` -> ``[(coastal_edge, type)]`` (shared with ``schema.state_to_parsed``)."""
    return port_edge_pairs(ports)


def _draw_port(cv: _Canvas, e: int, ptype: int, geom: Geometry, style: ColonistStyle) -> None:
    hs = geom["hex_size"]
    h = B.EDGE_HEXES[e][0]
    hx, hy = B.HEX_CENTERS[h]
    ex, ey = B.EDGE_POS[e]
    dx, dy = ex - hx, ey - hy
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    mx, my = pixel_of_edge(e, geom)
    cx, cy = mx + dx * 0.62 * hs, my + dy * 0.62 * hs
    # dock lines from the icon to both vertices
    for v in B.EDGE_VERTICES[e]:
        vx, vy = pixel_of_vertex(v, geom)
        cv.line([(cx, cy), (vx + (cx - vx) * 0.25, vy + (cy - vy) * 0.25)], style.port, 0.05 * hs)
    w, hgt = 0.72 * hs, 0.42 * hs
    cv.rounded_rect(cx - w / 2, cy - hgt / 2, cx + w / 2, cy + hgt / 2, 0.08 * hs, style.port,
                    style.port_outline, 0.025 * hs)
    if ptype == B.PORT_GENERIC:
        cv.text(cx, cy, "3:1", 0.26 * hs, (40, 40, 40))
    else:
        cv.text(cx - 0.1 * hs, cy, "2:1", 0.24 * hs, (40, 40, 40))
        sq = 0.13 * hs
        sx = cx + 0.22 * hs
        cv.polygon([(sx - sq, cy - sq), (sx + sq, cy - sq), (sx + sq, cy + sq), (sx - sq, cy + sq)],
                   style.tile[ptype], style.outline, 0.015 * hs)


# ---------------------------------------------------------------------------
# UI chrome
# ---------------------------------------------------------------------------
def _player_rgb(color: str) -> RGB:
    return PLAYER_RGB.get(color, _UNKNOWN_PLAYER_RGB)


def _draw_star(cv: _Canvas, cx: float, cy: float, r: float, fill: RGB) -> None:
    pts = []
    for k in range(10):
        ang = math.radians(-90 + k * 36)
        rr = r if k % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    cv.polygon(pts, fill)


def _draw_player_panel(cv: _Canvas, state: GameState, me: int, w: int, h: int, style: ColonistStyle) -> None:
    n = max(1, len(state.players))
    pw = max(150.0, 0.21 * w)
    row_h = min(0.10 * h, 0.7 * h / n)
    x0, y0 = 0.012 * w, 0.02 * h
    pad = 0.06 * row_h
    fs = 0.30 * row_h
    small = 0.26 * row_h
    for i, p in enumerate(state.players):
        ry = y0 + i * (row_h + pad)
        col = _player_rgb(p.color)
        border = (255, 255, 255) if i == state.current else style.outline
        cv.rounded_rect(x0, ry, x0 + pw, ry + row_h, 0.15 * row_h, col, border,
                        0.06 * row_h if i == state.current else 0.02 * row_h)
        text_col = (30, 30, 30) if p.color == "white" else style.panel_text
        name = (p.name or p.color)[:12] + (" (you)" if i == me else "")
        cv.text(x0 + 0.06 * pw, ry + 0.28 * row_h, name, fs, text_col, "lm")
        # stats row: VP star, cards, dev cards, knights
        sy = ry + 0.7 * row_h
        vp = state.total_vp(i) if i == me else state.public_vp(i)
        x = x0 + 0.06 * pw
        _draw_star(cv, x + 0.4 * small, sy, 0.55 * small, style.badge)
        cv.text(x + 1.0 * small, sy, str(vp), small, text_col, "lm")
        x += 0.24 * pw
        cards = sum(p.resources) if p.hand_known else p.hand_size
        cv.rounded_rect(x, sy - 0.5 * small, x + 0.7 * small, sy + 0.5 * small, 0.1 * small, (250, 250, 250),
                        style.outline, 0.05 * small)
        cv.text(x + 1.0 * small, sy, str(cards), small, text_col, "lm")
        x += 0.2 * pw
        dev = p.total_dev if p.dev_known else p.dev_count
        cv.rounded_rect(x, sy - 0.5 * small, x + 0.7 * small, sy + 0.5 * small, 0.1 * small, (120, 70, 170),
                        style.outline, 0.05 * small)
        cv.text(x + 1.0 * small, sy, str(dev), small, text_col, "lm")
        x += 0.2 * pw
        # knight shield
        sh = 0.55 * small
        cv.polygon([(x, sy - sh), (x + 1.1 * sh, sy - sh), (x + 1.1 * sh, sy + 0.2 * sh),
                    (x + 0.55 * sh, sy + sh), (x, sy + 0.2 * sh)], (70, 70, 80), style.outline, 0.04 * small)
        cv.text(x + 1.5 * sh, sy, str(p.played_knights), small, text_col, "lm")
        # badges
        bx = x0 + pw - 0.05 * pw
        for label, held in (("LA", state.largest_army_owner == i), ("LR", state.longest_road_owner == i)):
            if held:
                bw = 1.7 * small
                cv.rounded_rect(bx - bw, ry + 0.12 * row_h, bx, ry + 0.12 * row_h + 0.9 * small, 0.2 * small,
                                style.badge, style.outline, 0.04 * small)
                cv.text(bx - bw / 2, ry + 0.12 * row_h + 0.45 * small, label, 0.6 * small, (40, 30, 10))
                bx -= bw + 0.3 * small


def _draw_hand_bar(cv: _Canvas, player: Optional[Player], w: int, h: int, style: ColonistStyle) -> None:
    if player is None:
        return
    bar_h = 0.13 * h
    y1 = 0.985 * h
    y0 = y1 - bar_h
    card_w = 0.055 * w
    gap = 0.012 * w
    n_cards = 6  # 5 resources + dev cards
    total = n_cards * card_w + (n_cards - 1) * gap
    x = 0.5 * w - total / 2.0
    cv.rounded_rect(x - gap, y0 - 0.02 * h, x + total + gap, y1 + 0.01 * h, 0.15 * bar_h, style.panel)
    # an unknown hand is drawn as "?" (never as zeros, which a parser would read as an empty hand)
    counts = [str(c) for c in player.resources] if player.hand_known else ["?"] * 5
    fs = 0.32 * bar_h
    for r in range(5):
        col = style.tile[r]
        cv.rounded_rect(x, y0, x + card_w, y1, 0.08 * card_w, col, style.outline, 0.02 * card_w)
        cv.text(x + card_w / 2, y0 + 0.3 * bar_h, B.RESOURCE_NAMES[r][:4], 0.18 * bar_h, (255, 255, 255),
                "mm", style.outline, 0.02 * bar_h)
        cv.text(x + card_w / 2, y0 + 0.68 * bar_h, counts[r], fs, style.card_text, "mm", style.outline,
                0.04 * bar_h)
        x += card_w + gap
    dev = player.total_dev if player.dev_known else player.dev_count
    cv.rounded_rect(x, y0, x + card_w, y1, 0.08 * card_w, (120, 70, 170), style.outline, 0.02 * card_w)
    cv.text(x + card_w / 2, y0 + 0.3 * bar_h, "dev", 0.18 * bar_h, (255, 255, 255), "mm", style.outline,
            0.02 * bar_h)
    cv.text(x + card_w / 2, y0 + 0.68 * bar_h, str(dev), fs, style.card_text, "mm", style.outline, 0.04 * bar_h)


def _draw_bank_panel(cv: _Canvas, state: GameState, w: int, h: int, style: ColonistStyle) -> None:
    pw = max(120.0, 0.16 * w)
    ph = 0.085 * h
    x0, y0 = w - pw - 0.012 * w, 0.02 * h
    cv.rounded_rect(x0, y0, x0 + pw, y0 + ph, 0.12 * ph, style.panel, style.outline, 0.02 * ph)
    fs = 0.3 * ph
    cell = pw / 6.0
    for r in range(5):
        cx = x0 + cell * (r + 0.5)
        sq = 0.16 * ph
        cv.polygon([(cx - sq, y0 + 0.2 * ph), (cx + sq, y0 + 0.2 * ph), (cx + sq, y0 + 0.2 * ph + 2 * sq),
                    (cx - sq, y0 + 0.2 * ph + 2 * sq)], style.tile[r], style.outline, 0.01 * ph)
        cv.text(cx, y0 + 0.76 * ph, str(state.bank[r]), fs, style.panel_text)
    cx = x0 + cell * 5.5
    sq = 0.16 * ph
    cv.polygon([(cx - sq, y0 + 0.2 * ph), (cx + sq, y0 + 0.2 * ph), (cx + sq, y0 + 0.2 * ph + 2 * sq),
                (cx - sq, y0 + 0.2 * ph + 2 * sq)], (120, 70, 170), style.outline, 0.01 * ph)
    cv.text(cx, y0 + 0.76 * ph, str(sum(state.dev_deck)), fs, style.panel_text)


_DIE_PIPS = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
    6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
}


def _split_dice(total: int, rng: random.Random) -> Tuple[int, int]:
    if not 2 <= total <= 12:
        return (0, 0)
    pairs = [(a, total - a) for a in range(1, 7) if 1 <= total - a <= 6]
    return pairs[rng.randrange(len(pairs))]


def _draw_dice(cv: _Canvas, dice: int, w: int, h: int, style: ColonistStyle, rng: random.Random) -> None:
    d = 0.06 * h
    gap = 0.012 * w
    x1 = w - 0.02 * w
    y1 = h - 0.02 * h
    faces = _split_dice(dice, rng)
    for k, face in enumerate(reversed(faces)):
        bx1 = x1 - k * (d + gap)
        bx0 = bx1 - d
        by0 = y1 - d
        cv.rounded_rect(bx0, by0, bx1, y1, 0.18 * d, style.dice, style.outline, 0.03 * d)
        cx, cy = (bx0 + bx1) / 2.0, (by0 + y1) / 2.0
        for px, py in _DIE_PIPS.get(face, []):
            cv.circle(cx + px * 0.26 * d, cy + py * 0.26 * d, 0.075 * d, style.dice_pip)
    if dice:
        cv.text(x1 - (2 * d + gap) / 2.0, y1 - d - 0.35 * d, str(dice), 0.3 * d, (255, 255, 255), "mm",
                style.outline, 0.04 * d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _resolve_me(state: GameState, me: Union[int, str, None]) -> int:
    if me is None:
        return 0 if state.players else -1
    if isinstance(me, str):
        key = me.strip().lower()
        for i, p in enumerate(state.players):
            if p.color == key:
                return i
        return 0
    return int(me)


def render_state(state: GameState, size: Tuple[int, int] = (1280, 800), style: Optional[ColonistStyle] = None,
                 seed: int = 0, me: Union[int, str, None] = None, jitter: bool = True,
                 jpeg: Optional[bool] = None, geometry: Optional[Geometry] = None) -> Image.Image:
    """Render ``state`` as a Colonist.io-look-alike screenshot (RGB image of ``size``).

    ``me`` (index or colour, default player 0) is the player whose hand is
    shown in the hand bar.  ``seed`` drives every random choice (jitter,
    tile art wobble, dice faces, JPEG quality) so renders are deterministic.
    With ``jitter`` the board centre / hex size / colours are perturbed (see
    :func:`board_geometry`) and the image is JPEG re-compressed with
    probability one half; ``jpeg=True`` / ``False`` forces that on / off.
    ``geometry`` overrides the board placement (a :func:`board_geometry` dict).
    """
    style = style or ColonistStyle()
    w, h = int(size[0]), int(size[1])
    rng = random.Random(seed)
    geom = dict(geometry) if geometry is not None else board_geometry((w, h), seed, jitter)
    if jitter:
        style = _jittered_style(style, rng)
    hs = geom["hex_size"]
    me_idx = _resolve_me(state, me)
    ss = max(1, int(style.supersample))
    cv = _Canvas(w, h, ss, style.sea, style.font_path)

    # tiles
    tile_scale = 1.0 - style.tile_gap
    for hidx, (res, num) in enumerate(state.hexes):
        cx, cy = pixel_of_hex(hidx, geom)
        cv.polygon(_hex_polygon(cx, cy, hs * tile_scale), style.tile.get(res, style.tile[B.DESERT]))
        _draw_tile_art(cv, res, cx, cy, hs, style, rng)
    # ports (in the sea, under everything else)
    for e, t in _port_entries(state.ports):
        _draw_port(cv, e, t, geom, style)
    # tokens + robber
    for hidx, (res, num) in enumerate(state.hexes):
        cx, cy = pixel_of_hex(hidx, geom)
        if res != B.DESERT and num:
            _draw_token(cv, cx, cy, hs, num, style)
    if 0 <= state.robber < B.NUM_HEXES:
        cx, cy = pixel_of_hex(state.robber, geom)
        _draw_robber(cv, cx, cy, hs, style)
    # roads then buildings
    for p in state.players:
        col = _player_rgb(p.color)
        for e in p.roads:
            _draw_road(cv, e, geom, col, style)
    for p in state.players:
        col = _player_rgb(p.color)
        for v in p.settlements:
            x, y = pixel_of_vertex(v, geom)
            _draw_settlement(cv, x, y, hs, col, style)
        for v in p.cities:
            x, y = pixel_of_vertex(v, geom)
            _draw_city(cv, x, y, hs, col, style)
    # chrome
    _draw_player_panel(cv, state, me_idx, w, h, style)
    _draw_bank_panel(cv, state, w, h, style)
    _draw_hand_bar(cv, state.players[me_idx] if 0 <= me_idx < len(state.players) else None, w, h, style)
    _draw_dice(cv, state.dice, w, h, style, rng)

    img = cv.finish((w, h))
    do_jpeg = jpeg if jpeg is not None else (jitter and rng.random() < 0.5)
    if do_jpeg:
        quality = style.jpeg_quality if style.jpeg_quality is not None else rng.randint(70, 95)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()
    return img


def render_to_file(state: GameState, path: str, **kw: Any) -> Geometry:
    """Render ``state`` to ``path`` (format from the extension); returns the geometry used."""
    size = kw.get("size", (1280, 800))
    geom = kw.get("geometry") or board_geometry(size, kw.get("seed", 0), kw.get("jitter", True))
    img = render_state(state, geometry=geom, **{k: v for k, v in kw.items() if k != "geometry"})
    img.save(path)
    return geom
