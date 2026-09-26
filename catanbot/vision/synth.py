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

Game-log panel (opt-in)
-----------------------
``render_state(..., log=[...])`` also draws Colonist's game-log panel in the
right column (:func:`log_panel_box`): one entry per log line (the wording of
:data:`catanbot.colonist_log.PHRASES`), newest at the bottom, alternating
stripes, player names in bold in their colour, and - with
``LogPanelStyle.icons`` - the cards of card slots as small card icons
(resource cards, face-down card backs, the development card) and the two dice
of a roll as dice.  The pipeline is

1. :func:`log_line_tokens` - a line -> :class:`LogToken` s (names, words,
   icons) through the phrase table: names are taken from the matched
   phrase's player slots (case-insensitive), cards from its card slots; a
   multiplier number is glued to its icon; the robber line and the letters
   notation ``WWB`` always stay text;
2. :func:`layout_log_panel` - token widths, wrapping at token boundaries
   (continuation rows indented by 1 em), bottom-aligned stacking with the
   top-most entry optionally cut by the panel edge (``cut_top``); returns the
   ground truth :class:`LogEntryLayout` per visible entry, including its
   *canonical* text (the text a log OCR must produce: names -> colour words,
   icon runs -> ``N res``, card backs -> ``a card`` / ``N cards``, the
   development card icon -> ``Development Card``, dice -> ``d1 d2``);
3. the renderer draws exactly that layout.  With ``log=None`` (the default)
   nothing is drawn and the render is byte-identical to a render without the
   feature; the panel never draws from the render's random stream.
"""
from __future__ import annotations

import io
import math
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from .. import board as B
from ..state import GameState, Player
from .schema import port_edge_pairs

__all__ = [
    "ColonistStyle",
    "PLAYER_RGB",
    "DEFAULT_FONT_PATH",
    "LOG_FONT_PATH",
    "LOG_RESOURCE_ICONS",
    "LOG_ICON_IDS",
    "LogPanelStyle",
    "LogToken",
    "LogEntryLayout",
    "board_geometry",
    "pixel_of_hex",
    "pixel_of_vertex",
    "pixel_of_edge",
    "log_panel_box",
    "log_line_tokens",
    "canonical_log_text",
    "layout_log_panel",
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
# Game-log panel (opt-in: render_state(..., log=[...]))
# ---------------------------------------------------------------------------
Box = Tuple[float, float, float, float]

#: Regular text face of the log panel (names use the bold :data:`DEFAULT_FONT_PATH`).
LOG_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
#: Resource card icon ids in resource order (wood, brick, sheep, wheat, ore): the canonical words.
LOG_RESOURCE_ICONS: Tuple[str, ...] = ("wood", "brick", "sheep", "wheat", "ore")
#: Every icon id a :class:`LogToken` of kind ``icon`` may carry.
LOG_ICON_IDS: Tuple[str, ...] = LOG_RESOURCE_ICONS + ("card", "dev") + tuple(f"die{k}" for k in range(1, 7))
#: Longest run of identical card icons drawn one by one; a longer run is "N" + one icon.
LOG_MAX_ICON_RUN = 7
#: The default render scale (:class:`ColonistStyle`), which :func:`layout_log_panel` measures at.
_DEFAULT_SUPERSAMPLE: int = ColonistStyle.supersample

#: Name colours on the light panel: the player colours, but a white name would vanish on white.
_LOG_NAME_RGB: Dict[str, RGB] = dict(PLAYER_RGB, white=(150, 150, 150))
#: Name colours on the dark panel (blue / brown / purple / green / red brightened for contrast).
_LOG_NAME_RGB_DARK: Dict[str, RGB] = dict(PLAYER_RGB, red=(240, 100, 90), blue=(95, 155, 240),
                                          green=(85, 195, 105), purple=(180, 130, 230), brown=(185, 135, 95),
                                          white=(235, 235, 235))
#: Card icon colours: the tile colours, with brick and ore pushed away from the red / grey names.
_LOG_ICON_RGB: Dict[str, RGB] = {
    "wood": (34, 105, 48), "brick": (185, 78, 40), "sheep": (150, 200, 80), "wheat": (240, 200, 60),
    "ore": (105, 115, 135), "card": (135, 45, 40), "dev": (120, 70, 170), "die": (250, 250, 250),
}


@dataclass
class LogPanelStyle:
    """Look of the synthetic game-log panel (light theme by default, :meth:`dark` for the dark one).

    Text is ``font_scale * image height`` pixels (at least ``min_font_px``), rows are
    ``round(text_px * line_spacing)`` high.  The background must not be navy, beige, cream or the
    white player's grey (235, 235, 235): the board / chrome parser reads those colours (the light
    stripe and the dark theme come close to grey / navy and are safe only by where the panel sits:
    right of the board, below the bank, above the hand bar - see ``tests/test_synth_log.py``).

    Icons are told from names by shape and marks, not by fill colour alone: a card fill can be as
    close as L1 50 to a name colour (the dev card's purple fill vs the purple name, the card back vs
    the brown name), so every card has a dark outline and a light pictogram, the card back a tan
    frame and diamond (``card_back_mark``), and the development card a thick teal border
    (``dev_border``, a colour no name or card uses) around a white four-point star.
    """

    background: RGB = (252, 252, 253)
    stripe: RGB = (239, 242, 248)          # odd entries (absolute index)
    border: RGB = (190, 196, 208)
    text: RGB = (40, 44, 54)
    font_path: str = LOG_FONT_PATH                 # regular words
    name_font_path: str = DEFAULT_FONT_PATH       # player names (bold)
    font_scale: float = 0.0175
    min_font_px: float = 9.0
    line_spacing: float = 1.5
    icons: bool = True                     # card slots / dice / dev card as icons (False: text only)
    name_colours: Dict[str, RGB] = field(default_factory=lambda: dict(_LOG_NAME_RGB))
    icon_colours: Dict[str, RGB] = field(default_factory=lambda: dict(_LOG_ICON_RGB))
    icon_mark: RGB = (255, 255, 255)       # pictogram on the resource cards / dev card
    card_back_mark: RGB = (220, 175, 110)  # pattern on the face-down card back
    dev_border: RGB = (40, 180, 170)       # the development card's border (never a name colour)
    die_outline: RGB = (120, 124, 136)
    die_pip: RGB = (40, 40, 46)
    theme: str = "light"

    @classmethod
    def dark(cls, **kw: Any) -> "LogPanelStyle":
        """The dark theme: dark slate panel, light text, brightened name colours."""
        base = dict(background=(38, 42, 54), stripe=(48, 53, 66), border=(70, 76, 92), text=(190, 196, 208),
                    name_colours=dict(_LOG_NAME_RGB_DARK), card_back_mark=(225, 180, 115),
                    die_outline=(20, 20, 24), theme="dark")
        base.update(kw)
        return cls(**base)

    def text_px(self, height: float) -> float:
        """Text size (em, pixels) on an image ``height`` pixels high."""
        return max(float(self.min_font_px), float(self.font_scale) * float(height))

    def row_height(self, height: float) -> int:
        return max(1, int(round(self.text_px(height) * float(self.line_spacing))))

    def name_rgb(self, colour: Optional[str]) -> RGB:
        return self.name_colours.get(str(colour or "").lower(), _UNKNOWN_PLAYER_RGB)


@dataclass(frozen=True)
class LogToken:
    """One drawn item of a log line: ``kind`` ``text`` (a word), ``name`` (a player name drawn in
    the colour of ``colour``) or ``icon`` (``text`` is an id of :data:`LOG_ICON_IDS`).  ``space``
    tells whether a space precedes the token; tokens without one are glued (never wrapped apart)."""

    kind: str
    text: str
    colour: Optional[str] = None
    space: bool = True


@dataclass
class LogEntryLayout:
    """Ground truth of one visible log entry.

    ``index`` is the absolute log index (``first_index`` + position in the lines), ``text`` the input
    line, ``canonical`` the text a log OCR must produce for it, ``partial`` whether the panel's top
    edge cuts it, ``rows`` the visible part of each of its rows (pixel boxes clipped to the panel's
    inner area, top to bottom; fully hidden rows are left out), ``row_count`` all its rows,
    ``tokens`` its tokens with ``token_boxes[k] = (row, (x0, y0, x1, y1))`` (unclipped cell of token
    k: advance width x row height), ``stripe`` whether it has the stripe background and ``box`` the
    union of ``rows``.
    """

    index: int
    text: str
    canonical: str
    partial: bool
    rows: List[Box]
    tokens: List[LogToken]
    token_boxes: List[Tuple[int, Box]] = field(default_factory=list)
    row_count: int = 0
    stripe: bool = False
    box: Box = (0.0, 0.0, 0.0, 0.0)


def log_panel_box(size: Tuple[int, int]) -> Box:
    """Default log panel (pixels ``x0, y0, x1, y1``): right column between the bank and the dice."""
    w, h = float(size[0]), float(size[1])
    x1 = w - 0.012 * w
    x0 = x1 - max(120.0, 0.14 * w)
    return (x0, 0.15 * h, x1, 0.78 * h)


def _normalise_log_box(box: Optional[Sequence[float]], size: Tuple[int, int]) -> Box:
    """``box`` (pixels ``x0, y0, x1, y1``; ``None``: :func:`log_panel_box`) clipped to the image.

    Raises ``ValueError`` for anything that is not 4 finite numbers, an inverted or empty box
    (``x1 <= x0`` or ``y1 <= y0``) and a box with nothing inside the image; a box reaching past an
    image edge is clipped to it.
    """
    w, h = float(size[0]), float(size[1])
    if box is None:
        return log_panel_box((w, h))
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        raise ValueError(f"log_box must be 4 numbers (pixels x0, y0, x1, y1), got {box!r}") from None
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"log_box must be 4 finite numbers (pixels x0, y0, x1, y1), got {box!r}")
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"log_box {tuple(vals)} is inverted or empty: need x0 < x1 and y0 < y1 "
                         f"(pixels x0, y0, x1, y1)")
    cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(w, x1), min(h, y1)
    if cx1 - cx0 < 1.0 or cy1 - cy0 < 1.0:
        raise ValueError(f"log_box {tuple(vals)} lies outside the {int(w)}x{int(h)} image")
    return (cx0, cy0, cx1, cy1)


# --- tokenization ---------------------------------------------------------------------------
_TWO_DICE = re.compile(r"([1-6])(?:\s*(?:[+,&/]|and)\s*|\s+)([1-6])", re.IGNORECASE)
_DEV_CARD = re.compile(r"\bdevelopment card\b", re.IGNORECASE)
_CARD_SLOT_KINDS = ("gain", "year_of_plenty", "steal", "discard", "bank_trade", "player_trade", "offer", "counter")
_SLOT_NUMBER_WORDS = {"a", "an", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
_SLOT_FILLERS = {"and", "of", "the", "x"}


def _player_names(players: Any) -> Dict[str, str]:
    """``players`` (``state.players``, ``{name: colour}`` or ``[(name, colour)]``) -> ``{name: colour}``;
    a player without a name is known by its colour word."""
    out: Dict[str, str] = {}
    if not players:
        return out
    if isinstance(players, Mapping):
        items = list(players.items())
    else:
        items = []
        for p in players:
            if isinstance(p, (tuple, list)):
                items.append((p[0], p[1]))
            else:
                col = getattr(p, "color", None) or getattr(p, "colour", None) or ""
                items.append((getattr(p, "name", None) or col, col))
    for name, colour in items:
        colour = str(colour or "").strip().lower()
        name = re.sub(r"\s+", " ", str(name or colour or "")).strip()
        if name and name not in out:
            out[name] = colour
    return out


def _icon_run(icon: str, n: int, space: bool = True) -> List[LogToken]:
    """``n`` identical card icons (glued); a run longer than :data:`LOG_MAX_ICON_RUN` is "n" + one
    icon, the icon glued to its number (a row never breaks between a multiplier and its icon)."""
    if n > LOG_MAX_ICON_RUN:
        return [LogToken("text", str(n), None, space), LogToken("icon", icon, None, False)]
    return [LogToken("icon", icon, None, space if k == 0 else False) for k in range(n)]


def _resource_word(word: str) -> Optional[int]:
    """Resource index of one card word (``wood``, ``lumber``, ``grain``, plurals ...), else ``None``.
    Lower-cased first, so the letters notation (``W``, ``WWB``) is never a word."""
    from .. import colonist_log as L
    cp = L.parse_cards(word.lower())
    if cp.ok and not cp.unknown and sum(cp.counts) == 1:
        return next(r for r in range(5) if cp.counts[r])
    return None


def _card_slot_tokens(slot: str) -> Optional[List[LogToken]]:
    """Icons for the cards of a card slot (``2 wood, 1 ore``, ``wood wood ore``, ``2x wood`` /
    ``wood x2``, ``a card`` / ``N cards``), resources in the order the slot names them; ``None`` to
    keep the slot as text (the letters notation ``WWB``, ``?``, known and face-down cards mixed,
    anything :func:`catanbot.colonist_log.parse_cards` cannot read)."""
    from .. import colonist_log as L
    order: List[int] = []
    for tok in re.findall(r"\d+|[A-Za-z]+|[^\sA-Za-z\d]", slot):
        low = tok.lower()
        if tok.isdigit() or tok == "," or low in _SLOT_NUMBER_WORDS or low in _SLOT_FILLERS:
            continue
        if low in ("card", "cards"):
            continue
        r = _resource_word(low) if tok.isalpha() else None
        if r is None:
            return None
        if r not in order:
            order.append(r)
    cp = L.parse_cards(slot)
    if not cp.ok or (cp.unknown and any(cp.counts)):
        return None
    if cp.unknown:
        return _icon_run("card", cp.unknown)
    out: List[LogToken] = []
    for r in order:
        if cp.counts[r] > 0:
            out.extend(_icon_run(LOG_RESOURCE_ICONS[r], cp.counts[r]))
    return out or None


PhraseMatch = Tuple[Any, "re.Match[str]", int]


def _phrase_match(text: str) -> Optional[PhraseMatch]:
    """``(phrase, match, offset)``: the row of :data:`catanbot.colonist_log.PHRASES` that
    ``parse_log_line`` would use for ``text`` (whitespace already collapsed), its match on the
    cleaned line and where that line starts in ``text``; ``None`` when no row matches.  The only
    place the tokenizer touches the phrase table's internals (its compiled rows and line cleaner)."""
    from .. import colonist_log as L
    core = L._clean_line(text)
    if not core:
        return None
    end = len(text.rstrip(" .!"))
    p = end - len(core)
    if p < 0 or text[p:end] != core:
        p = text.find(core)
        if p < 0:
            return None
    for ph, rx in L._COMPILED:
        m = rx.match(core)
        if m:
            return ph, m, p
    return None


def _icon_spans(text: str, pm: Optional[PhraseMatch] = None) -> List[Tuple[int, int, List[LogToken]]]:
    """``(start, end, tokens)`` of the parts of ``text`` drawn as icons (card slots, dice, dev card)."""
    pm = pm if pm is not None else _phrase_match(text)
    if pm is None:
        return []
    ph, match, p = pm
    core = match.string
    kind = ph.kind
    spans: List[Tuple[int, int, List[LogToken]]] = []
    groups = match.groupdict()
    if kind == "roll" and groups.get("dice"):
        dm = _TWO_DICE.fullmatch(groups["dice"])
        if dm:
            s, e = match.span("dice")
            spans.append((s, e, [LogToken("icon", f"die{dm.group(1)}"), LogToken("icon", f"die{dm.group(2)}", None,
                                                                                  False)]))
    elif kind == "monopoly" and groups.get("n") is not None and groups.get("res"):
        r = _resource_word(groups["res"])
        if r is not None:       # the number glued to its icon, as a long run
            spans.append((match.start("n"), match.end("res"),
                          [LogToken("text", groups["n"]), LogToken("icon", LOG_RESOURCE_ICONS[r], None, False)]))
    elif kind == "buy_dev":
        dm = _DEV_CARD.search(core, match.end("a"))
        if dm:
            spans.append((dm.start(), dm.end(), [LogToken("icon", "dev")]))
    elif kind in _CARD_SLOT_KINDS:
        for grp in ("cards", "get"):
            if grp in groups and groups[grp]:
                toks = _card_slot_tokens(groups[grp])
                if toks:
                    spans.append((match.start(grp), match.end(grp), toks))
    return [(s + p, e + p, toks) for s, e, toks in spans]


def _name_spans(text: str, names: Dict[str, str], taken: Sequence[Tuple[int, int, Any]],
                pm: Optional[PhraseMatch] = None) -> List[Tuple[int, int, List[LogToken]]]:
    """Player names drawn as names, outside the ``taken`` spans.

    Names are looked for only in the player slots (``{A}`` / ``{B}``) of the matched phrase, so a
    player called "bank", "Robber", "7" or "wood" is a name only where the line names a player;
    a line no phrase matches (or one whose phrase has no player slot) is searched whole.  Whole
    words, longest name first; an exact-case match wins, then a case-insensitive one ("BOB" for a
    player "Bob"; between players differing only in case, the first in ``names`` wins).  The name
    token's text is the line's own spelling.
    """
    regions: List[Tuple[int, int]] = []
    if pm is not None:
        _, m, p = pm
        for g in ("a", "b"):
            if g in m.re.groupindex and m.group(g) is not None:
                regions.append((m.start(g) + p, m.end(g) + p))
    if not regions:
        regions = [(0, len(text))]
    busy = [(s, e) for s, e, _ in taken]
    order = sorted(enumerate(names), key=lambda kn: (-len(kn[1]), kn[0]))
    out: List[Tuple[int, int, List[LogToken]]] = []
    for flags in (0, re.IGNORECASE):
        for _, name in order:
            rx = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", flags)
            for rs, re_ in regions:
                for m in rx.finditer(text, rs, re_):
                    s, e = m.span()
                    if any(s < be and bs < e for bs, be in busy):
                        continue
                    busy.append((s, e))
                    out.append((s, e, [LogToken("name", text[s:e], names[name])]))
    return out


def log_line_tokens(line: str, players: Any, icons: bool = True) -> List[LogToken]:
    """One log line -> the tokens the panel draws, in order.

    ``players`` names the players (``state.players``, ``{name: colour}`` or ``[(name, colour)]``).
    The line is matched against the phrase table (:data:`catanbot.colonist_log.PHRASES`); the
    player names in its player slots become ``LogToken("name", name, colour)`` (case-insensitive,
    see :func:`_name_spans`); with ``icons`` the cards of card slots become card icons
    (``"2 wood, 1 ore"`` -> wood wood ore; more than :data:`LOG_MAX_ICON_RUN` of a kind -> the
    number as text glued to one icon; ``a card`` / ``N cards`` -> card backs), Monopoly's ``N res``
    -> the number glued to one icon, ``rolled d1 d2`` -> two dice and ``Development Card`` -> the
    dev card icon.  The robber line and the letters notation (``WWB``) stay text.  Every other word
    is a ``text`` token.  Whitespace is collapsed; ``space`` records where the line had a space.
    """
    text = re.sub(r"\s+", " ", str(line)).strip()
    if not text:
        return []
    pm = _phrase_match(text)
    spans = _icon_spans(text, pm) if icons else []
    spans = spans + _name_spans(text, _player_names(players), spans, pm)
    spans.sort(key=lambda sp: sp[0])
    out: List[LogToken] = []
    pos = 0
    for s, e, toks in spans + [(len(text), len(text), [])]:
        for m in re.finditer(r"\S+", text[pos:s]):
            a = pos + m.start()
            out.append(LogToken("text", m.group(), None, a > 0 and text[a - 1] == " "))
        if toks:
            out.append(replace(toks[0], space=s > 0 and text[s - 1] == " "))
            out.extend(toks[1:])
        pos = e
    return out


def _canonical_from_tokens(tokens: Sequence[LogToken]) -> str:
    """The canonical OCR text of drawn tokens (see the module docstring)."""
    parts: List[Tuple[str, bool]] = []
    n = len(tokens)

    def card_icon(k: int) -> bool:
        return 0 <= k < n and tokens[k].kind == "icon" and (tokens[k].text in LOG_RESOURCE_ICONS
                                                             or tokens[k].text == "card")

    def multiplier(k: int) -> bool:
        return tokens[k].kind == "text" and tokens[k].text.isdigit() and card_icon(k + 1)

    i = 0
    while i < n:
        t = tokens[i]
        if t.kind == "name":
            parts.append((t.colour or t.text, t.space))
            i += 1
        elif t.kind == "icon" and t.text.startswith("die"):
            after_die = i > 0 and tokens[i - 1].kind == "icon" and tokens[i - 1].text.startswith("die")
            parts.append((t.text[3:], t.space or after_die))        # two dice read "d1 d2"
            i += 1
        elif t.kind == "icon" and t.text == "dev":
            parts.append(("Development Card", t.space))
            i += 1
        elif card_icon(i) or multiplier(i):
            space = t.space
            runs: List[List[Any]] = []
            while i < n and (card_icon(i) or multiplier(i)):
                mult = 1
                if multiplier(i):
                    mult = int(tokens[i].text)
                    i += 1
                icon = tokens[i].text
                k = 0
                while i < n and card_icon(i) and tokens[i].text == icon and (k == 0 or not tokens[i].space):
                    k += 1
                    i += 1
                runs.append([icon, k * mult])
            words = []
            for icon, k in runs:
                words.append(("a card" if k == 1 else f"{k} cards") if icon == "card" else f"{k} {icon}")
            parts.append((", ".join(words), space))
        else:
            parts.append((t.text, t.space))
            i += 1
    return "".join((" " if sp and k else "") + s for k, (s, sp) in enumerate(parts))


def canonical_log_text(line: str, players: Any, icons: bool = True) -> str:
    """The canonical OCR text of ``line`` as the panel draws it (``icons`` as :class:`LogPanelStyle`)."""
    return _canonical_from_tokens(log_line_tokens(line, players, icons))


# --- layout -----------------------------------------------------------------------------------
@dataclass
class _LogMetrics:
    px: float                  # text size (em)
    row_h: int
    box: Tuple[int, int, int, int]      # panel (integer pixels)
    inner: Tuple[int, int, int, int]    # clip area inside the border
    pad_x: float
    border_w: float
    radius: float
    card_w: float
    card_h: float
    die: float
    space_w: float
    ss: int


def _log_metrics(size: Tuple[int, int], style: LogPanelStyle, box: Optional[Sequence[float]], ss: int) -> _LogMetrics:
    w, h = int(size[0]), int(size[1])
    px = style.text_px(h)
    bx = _normalise_log_box(box, (w, h))
    x0, y0, x1, y1 = (int(round(v)) for v in bx)
    border_w = max(1.0, 0.07 * px)
    radius = 0.5 * px
    inset = int(math.ceil(border_w + 0.3 * radius))
    inner = (x0 + inset, y0 + inset, max(x0 + inset + 1, x1 - inset), max(y0 + inset + 1, y1 - inset))
    card_h = 1.05 * px
    # a die must stay below the dice reader's minimum area (0.0005 w h) whatever the text size
    die = min(1.0 * px, 0.9 * math.sqrt(0.0005 * w * h))
    font = _font(style.font_path, px * ss)
    return _LogMetrics(px=px, row_h=style.row_height(h), box=(x0, y0, x1, y1), inner=inner,
                       pad_x=max(2.0, 0.45 * px), border_w=border_w, radius=radius, card_w=0.75 * card_h,
                       card_h=card_h, die=die, space_w=font.getlength(" ") / ss, ss=ss)


def _token_width(tok: LogToken, style: LogPanelStyle, mt: _LogMetrics) -> float:
    if tok.kind == "icon":
        return mt.die if tok.text.startswith("die") else mt.card_w
    path = style.name_font_path if tok.kind == "name" else style.font_path
    return _font(path, mt.px * mt.ss).getlength(tok.text) / mt.ss


def _token_gap(prev: LogToken, tok: LogToken, mt: _LogMetrics) -> float:
    both_icons = prev.kind == "icon" and tok.kind == "icon"
    if not tok.space:
        if both_icons:
            return (0.22 if tok.text.startswith("die") else 0.14) * mt.px
        if tok.kind == "icon" and prev.kind == "text" and prev.text.isdigit():
            return 0.18 * mt.px          # a multiplier and its icon: glued, a thin gap
        return 0.0
    return max(mt.space_w, 0.3 * mt.px) if both_icons else mt.space_w


def _wrap(tokens: Sequence[LogToken], style: LogPanelStyle, mt: _LogMetrics, avail: float
          ) -> List[List[Tuple[int, float, float]]]:
    """Rows of ``(token index, x offset, width)``; breaks only before a token preceded by a space."""
    widths = [_token_width(t, style, mt) for t in tokens]
    groups: List[List[int]] = []
    for k, t in enumerate(tokens):
        if not groups or t.space:
            groups.append([k])
        else:
            groups[-1].append(k)
    rows: List[List[Tuple[int, float, float]]] = [[]]
    x = 0.0
    for g in groups:
        # the group laid out on its own
        offs, gx = [], 0.0
        for j, k in enumerate(g):
            if j:
                gx += _token_gap(tokens[g[j - 1]], tokens[k], mt)
            offs.append(gx)
            gx += widths[k]
        row = rows[-1]
        limit = avail - (mt.px if len(rows) > 1 else 0.0)
        gap = _token_gap(tokens[row[-1][0]], tokens[g[0]], mt) if row else 0.0
        if row and x + gap + gx > limit:
            rows.append([])
            row, x, gap = rows[-1], 0.0, 0.0
        for k, o in zip(g, offs):
            row.append((k, x + gap + o, widths[k]))
        x += gap + gx
    return rows


def layout_log_panel(lines: Sequence[str], players: Any, size: Tuple[int, int],
                     style: Optional[LogPanelStyle] = None, box: Optional[Sequence[float]] = None,
                     first_index: int = 0, cut_top: float = 0.0,
                     supersample: Optional[int] = None) -> List[LogEntryLayout]:
    """Ground-truth layout of the log panel showing ``lines`` (oldest first) on an image of ``size``.

    Entries are bottom-aligned (the newest at the bottom), wrapped at token boundaries (continuation
    rows indented by 1 em) and dropped from the top when they do not fit.  ``cut_top`` in 0..0.9
    scrolls the stack so that that fraction of the top-most visible row is hidden above the panel
    edge (the entry it belongs to is ``partial``; without it the top-most entries are whole).
    ``first_index`` is the absolute index of ``lines[0]`` (stripe parity).  ``box`` is the panel
    (pixels, clipped to the image; ``ValueError`` when inverted / empty / off the image, see
    :func:`_normalise_log_box`).  ``supersample`` is the renderer's scale (text is measured at it,
    so the wrapping can differ between scales): ``None`` is the default :class:`ColonistStyle`'s;
    pass ``style.supersample`` for a render with another :class:`ColonistStyle`, or take the
    layout actually drawn from ``render_state(..., log_layout=[])``.  Returns the visible entries,
    oldest first.
    """
    style = style or LogPanelStyle()
    ss = max(1, int(_DEFAULT_SUPERSAMPLE if supersample is None else supersample))
    mt = _log_metrics(size, style, box, ss)
    ix0, iy0, ix1, iy1 = mt.inner
    avail = (ix1 - ix0) - 2 * mt.pad_x
    row_h = mt.row_h
    cut = min(0.9, max(0.0, float(cut_top)))
    height = iy1 - iy0
    delta = int(round((height % row_h) + cut * row_h)) % row_h if cut > 0 else 0
    bottom = iy1 - delta
    out: List[LogEntryLayout] = []
    for i in range(len(lines) - 1, -1, -1):
        if bottom <= iy0:
            break
        line = lines[i]
        tokens = log_line_tokens(line, players, style.icons)
        rows = _wrap(tokens, style, mt, avail) if tokens else [[]]
        top = bottom - len(rows) * row_h
        partial = top < iy0
        if partial and cut <= 0:
            break
        idx = int(first_index) + i
        boxes: List[Tuple[int, Box]] = [(0, (0.0, 0.0, 0.0, 0.0))] * len(tokens)
        vis: List[Box] = []
        for r, row in enumerate(rows):
            ry0 = top + r * row_h
            ry1 = ry0 + row_h
            indent = mt.px if r else 0.0
            for k, xo, wd in row:
                x = ix0 + mt.pad_x + indent + xo
                boxes[k] = (r, (x, float(ry0), x + wd, float(ry1)))
            if ry1 > iy0:
                vis.append((float(ix0), float(max(ry0, iy0)), float(ix1), float(ry1)))
        ub = (vis[0][0], vis[0][1], vis[-1][2], vis[-1][3]) if vis else (0.0, 0.0, 0.0, 0.0)
        out.append(LogEntryLayout(index=idx, text=str(line), canonical=_canonical_from_tokens(tokens),
                                  partial=partial, rows=vis, tokens=list(tokens), token_boxes=boxes,
                                  row_count=len(rows), stripe=idx % 2 == 1, box=ub))
        bottom = top
        if partial:
            break
    out.reverse()
    return out


# --- drawing ----------------------------------------------------------------------------------
def _draw_card_icon(pc: _Canvas, icon: str, x: float, cy: float, mt: _LogMetrics, style: LogPanelStyle) -> None:
    """A small card (resource / face-down back / development card) with its left edge at ``x``."""
    cw, ch = mt.card_w, mt.card_h
    y0, y1 = cy - ch / 2.0, cy + ch / 2.0
    fill = style.icon_colours.get(icon, (150, 150, 150))
    if icon == "dev":         # a thick border in a colour no name has: never mistaken for a purple name
        pc.rounded_rect(x, y0, x + cw, y1, 0.16 * cw, fill, style.dev_border, max(1.0, 0.12 * mt.px))
    else:
        pc.rounded_rect(x, y0, x + cw, y1, 0.16 * cw, fill, _shade(fill, 0.6), max(0.5, 0.06 * mt.px))
    cx = x + cw / 2.0
    mark = style.icon_mark
    if icon == "wood":        # a fir tree
        pc.polygon([(cx, cy - 0.33 * ch), (cx + 0.3 * cw, cy + 0.14 * ch), (cx - 0.3 * cw, cy + 0.14 * ch)], mark)
        pc.polygon([(cx - 0.07 * cw, cy + 0.14 * ch), (cx + 0.07 * cw, cy + 0.14 * ch),
                    (cx + 0.07 * cw, cy + 0.3 * ch), (cx - 0.07 * cw, cy + 0.3 * ch)], mark)
    elif icon == "brick":     # three bricks
        bw, bh = 0.27 * cw, 0.11 * ch
        for bx, by in ((cx - 0.3 * cw, cy - 0.16 * ch), (cx + 0.03 * cw, cy - 0.16 * ch),
                       (cx - 0.135 * cw, cy + 0.02 * ch)):
            pc.polygon([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], mark)
    elif icon == "sheep":     # a woolly cloud
        pc.ellipse(cx, cy + 0.02 * ch, 0.3 * cw, 0.15 * ch, mark)
        pc.circle(cx - 0.14 * cw, cy - 0.08 * ch, 0.13 * cw, mark)
        pc.circle(cx + 0.12 * cw, cy - 0.08 * ch, 0.13 * cw, mark)
    elif icon == "wheat":     # a stalk with grains
        pc.line([(cx, cy + 0.34 * ch), (cx, cy - 0.3 * ch)], mark, 0.08 * cw)
        for k, yy in enumerate((-0.2, -0.04, 0.12)):
            dx = 0.12 * cw
            pc.ellipse(cx - dx, cy + yy * ch, 0.09 * cw, 0.06 * ch, mark)
            pc.ellipse(cx + dx, cy + yy * ch, 0.09 * cw, 0.06 * ch, mark)
    elif icon == "ore":       # a rock
        pc.polygon([(cx - 0.32 * cw, cy + 0.22 * ch), (cx - 0.18 * cw, cy - 0.12 * ch),
                    (cx + 0.02 * cw, cy - 0.26 * ch), (cx + 0.3 * cw, cy - 0.04 * ch),
                    (cx + 0.3 * cw, cy + 0.22 * ch)], mark)
    elif icon == "card":      # face-down: inner frame and a diamond
        m = style.card_back_mark
        pc.rounded_rect(x + 0.2 * cw, y0 + 0.16 * ch, x + cw - 0.2 * cw, y1 - 0.16 * ch, 0.1 * cw, None, m,
                        max(0.5, 0.05 * mt.px))
        pc.polygon([(cx, cy - 0.17 * ch), (cx + 0.16 * cw, cy), (cx, cy + 0.17 * ch), (cx - 0.16 * cw, cy)], m)
    elif icon == "dev":       # a four-point star
        r1, r2 = 0.3 * cw, 0.09 * cw
        pts = []
        for k in range(8):
            ang = math.radians(-90 + 45 * k)
            rr = r1 if k % 2 == 0 else r2
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang) * (ch / cw) * 0.8))
        pc.polygon(pts, mark)


def _draw_die_icon(pc: _Canvas, face: int, x: float, cy: float, mt: _LogMetrics, style: LogPanelStyle) -> None:
    d = mt.die
    pc.rounded_rect(x, cy - d / 2.0, x + d, cy + d / 2.0, 0.18 * d, style.icon_colours.get("die", (250, 250, 250)),
                    style.die_outline, max(0.5, 0.06 * d))
    cx = x + d / 2.0
    for px, py in _DIE_PIPS.get(face, []):
        pc.circle(cx + px * 0.26 * d, cy + py * 0.26 * d, 0.085 * d, style.die_pip)


def _draw_log_panel(cv: _Canvas, players: Any, lines: Sequence[str], size: Tuple[int, int],
                    box: Optional[Sequence[float]], first_index: int, cut_top: float,
                    style: Optional[LogPanelStyle]) -> List[LogEntryLayout]:
    """Draw the log panel on ``cv`` (no randomness); returns the layout drawn."""
    style = style or LogPanelStyle()
    ss = cv.ss
    entries = layout_log_panel(lines, players, size, style, box, first_index, cut_top, supersample=ss)
    mt = _log_metrics(size, style, box, ss)
    x0, y0, x1, y1 = mt.box
    cv.rounded_rect(x0, y0, x1, y1, mt.radius, style.background, style.border, mt.border_w)
    ix0, iy0, ix1, iy1 = mt.inner
    # everything inside is drawn on a crop of the inner area: rows cut by the top edge are clipped
    sub = cv.img.crop((ix0 * ss, iy0 * ss, ix1 * ss, iy1 * ss))
    pc = _Canvas.__new__(_Canvas)
    pc.ss, pc.img, pc.d, pc.font_path = ss, sub, ImageDraw.Draw(sub), style.font_path
    text_font = _font(style.font_path, mt.px * ss)
    name_font = _font(style.name_font_path, mt.px * ss)
    row_h = mt.row_h
    for e in entries:
        if e.stripe and e.rows:
            etop = e.rows[-1][3] - e.row_count * row_h       # unclipped: the crop clips it
            pc.polygon([(0, etop - iy0), (ix1 - ix0, etop - iy0), (ix1 - ix0, e.rows[-1][3] - iy0),
                        (0, e.rows[-1][3] - iy0)], style.stripe)
        for tok, (r, (tx0, ty0, tx1, ty1)) in zip(e.tokens, e.token_boxes):
            cy = (ty0 + ty1) / 2.0 - iy0
            lx = tx0 - ix0
            if tok.kind == "icon":
                if tok.text.startswith("die"):
                    _draw_die_icon(pc, int(tok.text[3:]), lx, cy, mt, style)
                else:
                    _draw_card_icon(pc, tok.text, lx, cy, mt, style)
                continue
            font = name_font if tok.kind == "name" else text_font
            fill = style.name_rgb(tok.colour) if tok.kind == "name" else style.text
            pc.d.text((lx * ss, (cy + 0.36 * mt.px) * ss), tok.text, font=font, fill=fill, anchor="ls")
    cv.img.paste(sub, (ix0 * ss, iy0 * ss))
    return entries


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
                 jpeg: Optional[bool] = None, geometry: Optional[Geometry] = None,
                 log: Optional[Sequence[str]] = None, log_box: Optional[Tuple[float, float, float, float]] = None,
                 log_first_index: int = 0, log_cut_top: float = 0.0,
                 log_style: Optional[LogPanelStyle] = None,
                 log_layout: Optional[List[LogEntryLayout]] = None) -> Image.Image:
    """Render ``state`` as a Colonist.io-look-alike screenshot (RGB image of ``size``).

    ``me`` (index or colour, default player 0) is the player whose hand is
    shown in the hand bar.  ``seed`` drives every random choice (jitter,
    tile art wobble, dice faces, JPEG quality) so renders are deterministic.
    With ``jitter`` the board centre / hex size / colours are perturbed (see
    :func:`board_geometry`) and the image is JPEG re-compressed with
    probability one half; ``jpeg=True`` / ``False`` forces that on / off.
    ``geometry`` overrides the board placement (a :func:`board_geometry` dict).

    ``log`` (entries oldest first, player names as ``state.players[i].name``)
    adds the game-log panel in ``log_box`` (pixels, default
    :func:`log_panel_box`; clipped to the image, ``ValueError`` when inverted
    or off the image); ``log_first_index`` is the absolute index of ``log[0]``
    (stripe parity), ``log_cut_top`` the fraction of the top-most visible row
    hidden above the panel edge and ``log_style`` its look.  The ground truth
    of what is drawn is ``layout_log_panel(log, state.players, size,
    log_style, log_box, log_first_index, log_cut_top,
    supersample=style.supersample)``; a list passed as ``log_layout`` receives
    exactly the entries drawn.  ``log=None`` draws nothing (the default render
    is unchanged, ``log_layout`` stays empty).
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
    if log is not None:     # opt-in; draws nothing random (the render stream stays as it was)
        drawn = _draw_log_panel(cv, state.players, list(log), (w, h), log_box, log_first_index, log_cut_top,
                                log_style)
        if log_layout is not None:
            log_layout.extend(drawn)

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
