#!/usr/bin/env python3
"""Benchmark a local game-log reader (log OCR) on synthetic Colonist.io log panels.

Usage::

    python3 scripts/eval_logocr.py --quick                     # ~90 samples, catanbot.vision.logocr
    python3 scripts/eval_logocr.py                             # full benchmark, ~900 samples
    python3 scripts/eval_logocr.py --reader oracle --quick     # self-check of the harness: must be 100 %
    python3 scripts/eval_logocr.py --reader oracle:dup,rekey --quick   # a deliberately bad oracle
    python3 scripts/eval_logocr.py --reader my.module:read_fn --save-dir /tmp/fails --json summary.json

The reader contract (``catanbot.vision.logocr``): ``read_log_panel(img, box=None, profile=None,
players=None, cache=None)`` returns a result whose ``lines`` (oldest first) each have ``text``
(canonical text: names as colour words, card icons as ``N res``, ...), ``partial``, ``confidence``
and ``key`` (a stable hash of the entry's normalised pixels); an optional ``box`` is the panel it
read.  ``--reader module:function`` points at any other implementation; ``LineCache`` and
``find_log_panel(img, profile=None)`` are looked up in the same module (no cache / no separate
detection when it has none).  Nothing is imported from the reader until the benchmark runs.
``--reader oracle`` returns the ground truth; ``oracle:FAULT[,FAULT]`` injects the faults of
:data:`ORACLE_FAULTS` (each must make its metric fail: the harness's own check).

Samples (deterministic from ``--seed``): heuristic-bot games rendered as Colonist logs
(``tests/test_colonist_log.py::play``, card styles counts / words / colonist) with hand-made
counter-offer, Monopoly, Year of Plenty, discard, steal, bank-trade and offer lines injected; per
sample a window of recent entries is drawn with :func:`catanbot.vision.synth.render_state` on a
random board and the truth is the layout it drew (``render_state(..., log_layout=[])``); as on
Colonist, the panel draws the item of "built a" / "placed a" as a building icon in the player's
colour and writes a colon after the verbs of ~40 % of the lines ("got:", "for:" ...:
``LogPanelStyle`` defaults, so both are in every condition).  Each
sample is drawn twice: the window, and the window without its newest entry (the previous frame of
a live screen: the same panel one entry earlier, every entry one entry-height lower).

Conditions (reported per value): ``size`` (final image size; a browser-zoomed sample is labelled
``1280x800@x0.8``), ``icons`` on / off, ``theme`` light / dark, ``cut`` 0 / 0.3 / 0.6 (top entry
cut by the panel edge), ``jpeg`` (quality 60-92), ``noise`` (Gaussian noise sigma 4-10 and / or
blur radius 0.6-1.0), ``font`` (DejaVu, and the held-out Liberation Sans / FreeSans when
installed), ``font_px`` (text forced to 9 / 11 / 14 / 20 / 28 px; ``default``: 0.0175 x height),
``scale`` (browser zoom: a 1280x800 render resized by 0.8 / 1.25), ``box`` (the panel at its default
place or jittered in the right column: x0 / width / y0 / y1), ``colour`` (panel colour jitter) and
``players`` (recoloured, so white / purple / brown / pink names appear; renamed).

Metrics (overall and per condition value):

* ``exact`` - fully visible entries read as the right event (the tracker key of
  :func:`event_signature`); ``emitted`` - fully visible entries emitted as a non-partial line;
* ``precision`` - emitted non-partial lines matched one-to-one to a distinct panel entry with the
  same event (multiset matching; the order-preserving alignment's pairs first); ``extra`` - the
  other emitted non-partial lines (duplicates, misreads, garbage: what makes the tracker
  double-count or invent events), of which ``false`` have an event that is not on the panel at all;
* ``part_viol`` - samples whose cut top entry came out as a full line with confidence >= 0.6 (it
  must be flagged partial or omitted); calibration - accuracy of emitted lines with confidence
  >= 0.6 vs < 0.6;
* ``box`` - the result's panel box has IoU >= 0.8 with the true panel; ``detect`` - panel
  detection: ``find_log_panel(img)`` (or, without one, the result's box; with ``--give-box`` a read
  without the box) has IoU >= 0.8, also as mean IoU - scored with and without ``--give-box``;
* ``scroll`` - entries fully visible in both frames of a scroll pair that got the same text, flag
  and ``key`` (a non-empty one) in both, reading the previous frame then the current one with one
  cache; ``scroll_fresh`` - the same with a fresh cache for each frame (the key is a content hash,
  not a cache serial);
* ``warm_same`` - a re-read of the identical image with the same cache returned identical lines
  (text, flag, confidence, key) and box; read time per panel cold (fresh cache) and warm;
* ``after`` - the warm read of the scrolled (new) frame, scored like the cold read (exact and no
  extra lines): a cache must never return a stale reading for a changed panel;
* ``key_clash`` - emitted lines that share a key but read differently (keys are content hashes:
  only identically drawn entries may share one).  Keys are compared across a scroll only on
  lossless, unzoomed frames (noise, JPEG blocks and resampling legitimately change a scrolled
  entry's pixels); the text and the partial flag must be stable on every frame.

Exit status: 2 for bad usage, 1 when ``--min-exact`` or ``--max-failed`` is not met, else 0.

A sample fails when an entry is not exact, a line is extra, the cut entry violates, the box or the
detection misses (a finder returning ``None`` is a miss), an entry is unstable across the scroll
pair, the warm re-read differs, the warm read of the new frame is wrong, or keys clash.
"""
from __future__ import annotations

# JUDGE_READY: benchmark v2 (after-read scoring, None detections, lossless-only key checks, key clashes, exit codes)

import argparse
import dataclasses
import difflib
import importlib
import io
import json
import os
import random
import re
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from catanbot import board as B  # noqa: E402
from catanbot import colonist_log as L  # noqa: E402
from catanbot.vision import synth  # noqa: E402

Box = Tuple[float, float, float, float]
SIZES: Tuple[Tuple[int, int], ...] = ((1000, 640), (1280, 800), (1366, 768), (1440, 900), (1920, 1080), (2560, 1440))
STYLES = ("counts", "words", "colonist")
CUTS = (0.0, 0.3, 0.6)
SCALES = (0.8, 1.25)
FONT_PXS = (9, 11, 14, 20, 28)
#: A panel is at least this many em wide (a forced text size picks a screen whose panel fits it).
MIN_PANEL_EMS = 11.0
FONT_DIR = "/usr/share/fonts/truetype"
FONTS: Dict[str, Tuple[str, str]] = {
    "dejavu": (synth.LOG_FONT_PATH, synth.DEFAULT_FONT_PATH),
    "liberation": (os.path.join(FONT_DIR, "liberation/LiberationSans-Regular.ttf"),
                   os.path.join(FONT_DIR, "liberation/LiberationSans-Bold.ttf")),
    "freesans": (os.path.join(FONT_DIR, "freefont/FreeSans.ttf"), os.path.join(FONT_DIR, "freefont/FreeSansBold.ttf")),
}
USERNAMES = ("Kaz", "Mira_7", "TheBaron", "lumberjack99", "Ana Luz", "Q", "Tobi", "settler_x", "Nyx", "GrandPa Joe",
             "Zed", "k8", "Marguerite", "Robbie")
CONF_HI = 0.6
BOX_IOU = 0.8
CONDITIONS = ("size", "icons", "theme", "cut", "jpeg", "noise", "font", "font_px", "scale", "box", "colour",
              "players")


# ---------------------------------------------------------------------------
# event keys
# ---------------------------------------------------------------------------
def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def event_signature(ev: Optional[L.LogEvent], name_to_colour: Optional[Dict[str, str]] = None) -> Optional[tuple]:
    """Tracker-style key of an event: ``(kind, player, other, cards, get, count, value, item, resource,
    free)`` with players as colour words (``name_to_colour`` maps names; "you" for the viewer).
    Players resolve as the tracker's ``seat_of`` does: exactly, then without punctuation."""
    if ev is None:
        return None
    names = {_norm(k): v for k, v in (name_to_colour or {}).items()}
    stripped = {re.sub(r"[^\w#]+", "", k): v for k, v in names.items()}

    def who(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        k = _norm(x)
        if k in names:
            return names[k]
        k2 = re.sub(r"[^\w#]+", "", k)
        if k2 in ("you", "me", "yourself", "your"):
            return "you"
        return stripped.get(k2, k2)

    def cs(v: Optional[Sequence[int]]) -> Optional[tuple]:
        return None if v is None else tuple(int(x) for x in v)

    return (ev.kind, who(ev.player), who(ev.other), cs(ev.cards), cs(ev.get), ev.count, ev.value, ev.item,
            ev.resource, bool(ev.free))


def text_signature(text: str) -> Optional[tuple]:
    return event_signature(L.parse_log_line(text or ""))


# ---------------------------------------------------------------------------
# games and samples
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Game:
    seed: int
    style: str
    lines: List[str]
    checkpoints: List[Tuple[Any, int]]      # (state copy, number of log lines at that point)


def build_games(seeds: Sequence[int], styles: Sequence[str], per_game: int) -> List[Game]:
    from tests.test_colonist_log import play
    games = []
    for seed in seeds:
        for style in styles:
            rng = random.Random(seed * 131 + len(style))
            cps: List[Tuple[Any, int]] = []
            lines: List[str] = []
            for pre, a, s, ls in play(seed, style=style):
                lines = ls
                if len(ls) >= 12 and len(cps) < per_game and rng.random() < 0.03:
                    if not cps or len(ls) > cps[-1][1]:
                        cps.append((s.copy(), len(ls)))
            if not cps:
                cps.append((s.copy(), len(lines)))
            games.append(Game(seed, style, list(lines), cps))
    return games


def _cards(rng: random.Random, style: str, n: int) -> str:
    counts = [0] * 5
    for _ in range(n):
        counts[rng.randrange(5)] += 1
    return L.format_cards(counts, style)


def hand_made_lines(rng: random.Random, names: Sequence[str], style: str) -> List[str]:
    """One injected entry (or two for the two-line Monopoly) in the game's card style."""
    a, b = rng.sample(list(names[1:]), 2)
    res = rng.randrange(5)
    res_word = (L.COLONIST_CARD_NAMES if style == "colonist" else B.RESOURCE_NAMES)[res]
    c = lambda n: _cards(rng, style, n)  # noqa: E731
    k = rng.randrange(11)
    if k == 0:
        return [f"{a} counter-offered {c(rng.randint(1, 3))} for {c(rng.randint(1, 2))}"]
    if k == 1:
        return [f"{a} counter-offered to {b}: {c(rng.randint(1, 2))} for {c(rng.randint(1, 3))}"]
    if k == 2:
        return [f"{a} used Monopoly and stole {rng.randint(0, 9)} {res_word}"]
    if k == 3:
        return [f"{a} used Monopoly", f"{a} stole {rng.randint(0, 11)} {res_word}"]
    if k == 4:
        return [f"{a} used Year of Plenty and took {c(2)}"]
    if k == 5:
        if rng.random() < 0.5:
            return [f"{a} discarded {c(rng.randint(4, 9))}"]
        return [f"{a} discarded {rng.randint(4, 12)} cards"]
    if k == 6:
        return [f"You stole {c(1)} from {b}"]
    if k == 7:
        return [f"{a} stole {c(1)} from you"]
    if k == 8:
        return [f"{a} gave {c(rng.choice((2, 3, 4)))} and got {c(1)} from bank"]
    if k == 9:
        return [f"{a} wants to give {c(rng.randint(1, 3))} for {c(rng.randint(1, 2))}"]
    return [f"{a} stole a card from {b}"]


@dataclasses.dataclass
class Look:
    """How a sample's frames are drawn and degraded (shared by both frames of a scroll pair)."""

    size: Tuple[int, int]                  # render size (before the browser zoom)
    style: Any                             # synth.LogPanelStyle
    box: Optional[Box]                     # log_box given to the renderer (None: the default place)
    cut: float
    scale: float = 1.0
    blur: float = 0.0
    sigma: float = 0.0
    jpeg_q: Optional[int] = None
    render_seed: int = 0
    noise_seed: int = 0


@dataclasses.dataclass
class Sample:
    id: int
    img: Image.Image
    box: Box                                           # the panel in the final image (pixels)
    truth: List[Any]                                   # LogEntryLayout (boxes in the final image)
    cond: Dict[str, str]
    lines: List[str]
    names: Dict[str, str]                              # name -> colour word
    name_rgb: Dict[str, Tuple[int, int, int]]          # colour word -> name RGB on this panel
    prev_img: Optional[Image.Image] = None             # the previous frame (lines[:-1]), same look
    prev_truth: Optional[List[Any]] = None


def _jitter_rgb(c: Sequence[int], rng: random.Random, amt: int) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(v) + rng.randint(-amt, amt))) for v in c)  # type: ignore[return-value]


def _scale_entry(e: Any, f: float) -> Any:
    sc = lambda b: tuple(v * f for v in b)  # noqa: E731
    return dataclasses.replace(e, rows=[sc(r) for r in e.rows], token_boxes=[(r, sc(b)) for r, b in e.token_boxes],
                               box=sc(e.box))


def jitter_box(size: Tuple[int, int], text_px: float, rng: random.Random) -> Box:
    """A panel box in the safe right column: x1 0.975-0.996 w, x0 >= 0.83 w, width >= max(120 px,
    0.10 w, :data:`MIN_PANEL_EMS` em), y0 0.15-0.35 h (below the bank), y1 0.55-0.78 h (above the
    dice and the hand bar)."""
    w, h = size
    x1 = rng.uniform(0.975, 0.996) * w
    wmin = max(120.0, 0.10 * w, MIN_PANEL_EMS * text_px)
    wmax = max(wmin, min(0.16 * w, x1 - 0.83 * w))
    x0 = x1 - rng.uniform(wmin, wmax)
    return (x0, rng.uniform(0.15, 0.35) * h, x1, rng.uniform(0.55, 0.78) * h)


def render_frame(state: Any, lines: Sequence[str], first_index: int, look: Look
                 ) -> Tuple[Image.Image, List[Any], Box]:
    """One frame: (image, truth entries, panel box), all in final-image pixels."""
    truth: List[Any] = []
    img = synth.render_state(state, size=look.size, seed=look.render_seed, me=0, jitter=True, jpeg=False,
                             log=list(lines), log_box=look.box, log_first_index=first_index, log_cut_top=look.cut,
                             log_style=look.style, log_layout=truth)
    box: Box = tuple(float(v) for v in synth._log_metrics(look.size, look.style, look.box, 2).box)  # type: ignore
    if look.scale != 1.0:
        img = img.resize((int(round(look.size[0] * look.scale)), int(round(look.size[1] * look.scale))),
                         Image.BICUBIC)
        truth = [_scale_entry(e, look.scale) for e in truth]
        box = tuple(v * look.scale for v in box)  # type: ignore[assignment]
    if look.blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(look.blur))
    if look.sigma > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + np.random.default_rng(look.noise_seed).normal(0.0, look.sigma, arr.shape).astype(np.float32)
        img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    if look.jpeg_q is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=look.jpeg_q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()
    return img, truth, box


def make_sample(k: int, games: Sequence[Game], seed: int, fonts: Dict[str, Tuple[str, str]],
                want: Optional[Dict[str, str]] = None, scroll: bool = True) -> Optional[Sample]:
    """Sample ``k`` (deterministic); ``None`` when its conditions do not match ``want`` (not rendered).
    With ``scroll`` the previous frame (the window without its newest entry) is rendered too."""
    rng = random.Random(seed * 1000003 + k * 7919)
    game = games[k % len(games)]
    state0, n = game.checkpoints[rng.randrange(len(game.checkpoints))]
    state = state0.copy()
    cond: Dict[str, str] = {}
    # players: recolour (so every name colour appears) and rename
    colours = [p.color for p in state.players]
    if rng.random() < 0.4:
        colours = rng.sample(list(synth.PLAYER_RGB), len(state.players))
        cond["players"] = "recoloured"
    else:
        cond["players"] = "default"
    old_names = [p.name or p.color for p in state.players]
    new_names = list(old_names)
    if rng.random() < 0.5:
        new_names = rng.sample(USERNAMES, len(state.players))
        cond["players"] += "+renamed"
    for p, col, nm in zip(state.players, colours, new_names):
        p.color, p.name = col, nm
    # the window of recent entries, with hand-made lines injected
    K = rng.randint(6, 45)
    start = max(0, n - K)
    window = list(game.lines[start:n])
    for _ in range(rng.choice((0, 1, 1, 2, 3))):
        pos = rng.randint(0, len(window))
        window[pos:pos] = hand_made_lines(rng, old_names, game.style)
    table = dict(zip(old_names, new_names))
    rx = re.compile(r"(?<!\w)(" + "|".join(re.escape(x) for x in sorted(old_names, key=len, reverse=True)) + r")(?!\w)")
    window = [rx.sub(lambda m: table[m.group(1)], line) for line in window]
    # the look
    zoom = rng.random()
    scale = 0.8 if zoom < 0.1 else 1.25 if zoom < 0.2 else 1.0
    font_px: Optional[int] = FONT_PXS[rng.randrange(len(FONT_PXS))] if rng.random() < 0.3 else None
    if font_px is not None:
        scale = 1.0                          # the forced size is the final size
        fits = [s for s in SIZES if max(120.0, 0.14 * s[0]) >= MIN_PANEL_EMS * font_px]
        size = fits[rng.randrange(len(fits))]
    elif scale != 1.0:
        size = (1280, 800)
    else:
        size = SIZES[rng.randrange(len(SIZES))]
    icons = rng.random() < 0.75
    dark = rng.random() < 0.3
    cut = CUTS[rng.randrange(len(CUTS))]
    jpeg_q = rng.randint(60, 92) if rng.random() < 0.5 else None
    nr = rng.random()
    noise = "none" if nr < 0.55 else "noise" if nr < 0.7 else "blur" if nr < 0.85 else "noise+blur"
    sigma = rng.uniform(4.0, 10.0) if "noise" in noise else 0.0
    blur = rng.uniform(0.6, 1.0) if "blur" in noise else 0.0
    fr = rng.random()
    font = "dejavu" if fr < 0.6 else "liberation" if fr < 0.8 else "freesans"
    if font not in fonts:
        font = "dejavu"
    style = synth.LogPanelStyle.dark(icons=icons) if dark else synth.LogPanelStyle(icons=icons)
    style.font_path, style.name_font_path = fonts[font]
    if font_px is not None:
        style.font_scale = font_px / float(size[1])
    if rng.random() < 0.25:
        amt = 10
        for attr in ("background", "stripe", "text"):
            setattr(style, attr, _jitter_rgb(getattr(style, attr), rng, amt // 2))
        style.name_colours = {c: _jitter_rgb(v, rng, amt) for c, v in style.name_colours.items()}
        style.icon_colours = {c: _jitter_rgb(v, rng, amt) for c, v in style.icon_colours.items()}
        cond["colour"] = "jittered"
    else:
        cond["colour"] = "plain"
    box = jitter_box(size, style.text_px(size[1]), rng) if rng.random() < 0.3 else None
    final = size if scale == 1.0 else (int(round(size[0] * scale)), int(round(size[1] * scale)))
    cond.update(size=f"{final[0]}x{final[1]}" if scale == 1.0 else f"{size[0]}x{size[1]}@x{scale:g}",
                icons="on" if icons else "off", theme="dark" if dark else "light", cut=f"{cut:g}",
                jpeg="on" if jpeg_q else "off", noise=noise, font=font,
                font_px="default" if font_px is None else str(font_px),
                scale="none" if scale == 1.0 else f"x{scale:g}", box="default" if box is None else "jittered")
    if want and any(cond.get(c) != v for c, v in want.items()):
        return None
    look = Look(size=size, style=style, box=box, cut=cut, scale=scale, blur=blur, sigma=sigma, jpeg_q=jpeg_q,
                render_seed=seed * 31 + k, noise_seed=seed * 7907 + k)
    first_index = start
    img, truth, pbox = render_frame(state, window, first_index, look)
    prev_img = prev_truth = None
    if scroll and len(window) > 1:
        prev_img, prev_truth, _ = render_frame(state, window[:-1], first_index, look)
    names = {p.name: p.color for p in state.players}
    name_rgb = {p.color: style.name_rgb(p.color) for p in state.players}
    return Sample(k, img, pbox, truth, cond, window, names, name_rgb, prev_img, prev_truth)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
#: Faults the built-in oracle can inject (``--reader oracle:dup,rekey``); each one must make the
#: named metric fail, which is how the harness checks itself.
ORACLE_FAULTS: Dict[str, str] = {
    "dup": "emits the newest full entry twice (extra / precision)",
    "rekey": "keys depend on the entry's vertical position, so they change on scroll (scroll)",
    "cachekey": "keys are serials handed out by the cache, newest first (scroll_fresh)",
    "defaultbox": "returns and finds the default log_panel_box(img.size), ignoring a given box (box / detect)",
    "flaky": "a re-read of the same image with the same cache lowers a confidence (warm_same)",
    "stale": "a warm cache returns the previous frame's lines for a new image (after)",
    "nodetect": "find_log_panel finds nothing (detect)",
    "samekey": "every line gets the same key (key_clash)",
}
#: id(image) -> (truth entries, panel box) of the frames being read (the oracle's answer sheet).
_ORACLE_FRAMES: Dict[int, Tuple[List[Any], Box]] = {}


class OracleCache(dict):
    """The oracle's cache (a dict): serial keys for ``cachekey``, the last image for ``flaky``."""


@dataclasses.dataclass
class Reader:
    read: Callable[..., Any]
    cache_cls: Optional[Callable[[], Any]] = None
    find: Optional[Callable[..., Any]] = None


def make_oracle(faults: Sequence[str] = ()) -> Reader:
    """The ground-truth reader (the harness self-check: every metric must be 100 %), optionally
    with the faults of :data:`ORACLE_FAULTS`."""
    faults = set(faults)
    bad = faults - set(ORACLE_FAULTS)
    if bad:
        raise ValueError(f"unknown oracle fault(s) {', '.join(sorted(bad))}; known: {', '.join(ORACLE_FAULTS)}")

    def find(img: Any, profile: Any = None) -> Optional[Box]:
        if "nodetect" in faults:
            return None
        if "defaultbox" in faults:
            return synth.log_panel_box(img.size)
        return _ORACLE_FRAMES[id(img)][1]

    def read(img: Any, box: Any = None, profile: Any = None, players: Any = None, cache: Any = None) -> Any:
        if "stale" in faults and cache is not None and "result" in cache:
            return cache["result"]
        truth, true_box = _ORACLE_FRAMES[id(img)]
        serials = cache.setdefault("serials", {}) if cache is not None else {}
        keys: Dict[int, str] = {}
        for pos in range(len(truth) - 1, -1, -1):          # newest first
            e = truth[pos]
            key = f"oracle:{e.canonical}|{e.row_count}"
            if "rekey" in faults:
                key += f"@{int(e.box[1])}"
            if "cachekey" in faults:
                key = str(serials.setdefault(key, len(serials)))
            if "samekey" in faults:
                key = "K"
            keys[pos] = key
        lines = [SimpleNamespace(text=e.canonical, partial=e.partial, confidence=0.3 if e.partial else 1.0,
                                 box=e.box, key=keys[pos], event=L.parse_log_line(e.canonical))
                 for pos, e in enumerate(truth)]
        if "dup" in faults:
            full = [k for k, ln in enumerate(lines) if not ln.partial]
            if full:
                lines.insert(full[-1] + 1, SimpleNamespace(**vars(lines[full[-1]])))
        if "flaky" in faults and cache is not None:
            if cache.get("last") == id(img) and lines:
                lines[-1].confidence = round(lines[-1].confidence * 0.99, 4)
            cache["last"] = id(img)
        if "defaultbox" in faults:
            rbox: Any = synth.log_panel_box(img.size)
        else:
            rbox = tuple(box) if box is not None else true_box
        result = SimpleNamespace(lines=lines, box=rbox, warnings=[], confidence=1.0)
        if "stale" in faults and cache is not None:
            cache["result"] = result
        return result

    return Reader(read, OracleCache, find)


def load_reader(spec: str) -> Reader:
    """``oracle[:FAULT,...]`` or ``module:function`` (function defaults to read_log_panel) -> a
    :class:`Reader` (the module's ``LineCache`` and ``find_log_panel`` when it has them).  Raises
    ImportError (no module), AttributeError (no such function) or ValueError (unknown fault)."""
    if spec == "oracle" or spec.startswith("oracle:"):
        faults = [f.strip() for f in spec.partition(":")[2].split(",") if f.strip()]
        return make_oracle(faults)
    mod_name, _, fn = spec.partition(":")
    mod = importlib.import_module(mod_name)
    name = fn or "read_log_panel"
    read = getattr(mod, name, None)
    if read is None or not callable(read):
        raise AttributeError(f"module '{mod_name}' has no function '{name}'")
    find = getattr(mod, "find_log_panel", None)
    return Reader(read, getattr(mod, "LineCache", None), find if callable(find) else None)


OcrLine = Tuple[str, bool, float, Any]       # text, partial, confidence, key


def _lines_of(result: Any) -> List[OcrLine]:
    out = []
    for ln in getattr(result, "lines", None) or []:
        conf = getattr(ln, "confidence", 1.0)
        out.append((str(getattr(ln, "text", "") or ""), bool(getattr(ln, "partial", False)),
                    float(1.0 if conf is None else conf), getattr(ln, "key", None)))
    return out


def _box_of(result: Any) -> Optional[Box]:
    b = getattr(result, "box", None)
    try:
        return None if b is None else tuple(float(v) for v in b)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def align(truth_sigs: Sequence[Any], truth_texts: Sequence[str], ocr_sigs: Sequence[Any],
          ocr_texts: Sequence[str]) -> List[Optional[int]]:
    """Order-preserving alignment (edit distance; a pair costs 0 when the events agree, else
    1 - 0.5 * text similarity; a gap costs 1).  Returns the OCR index matched to each truth entry."""
    n, m = len(truth_sigs), len(ocr_sigs)
    memo: Dict[Tuple[int, int], float] = {}

    def cost(i: int, j: int) -> float:
        c = memo.get((i, j))
        if c is None:
            if truth_sigs[i] is not None and truth_sigs[i] == ocr_sigs[j]:
                c = 0.0
            else:
                c = 1.0 - 0.5 * difflib.SequenceMatcher(None, _norm(truth_texts[i]), _norm(ocr_texts[j])).ratio()
            memo[(i, j)] = c
        return c

    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = float(i)
    for j in range(1, m + 1):
        D[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + cost(i - 1, j - 1), D[i - 1][j] + 1.0, D[i][j - 1] + 1.0)
    match: List[Optional[int]] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if abs(D[i][j] - (D[i - 1][j - 1] + cost(i - 1, j - 1))) < 1e-9:
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif abs(D[i][j] - (D[i - 1][j] + 1.0)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return match


def match_lines(t_sigs: Sequence[Any], o_sigs: Sequence[Any], o_partial: Sequence[bool],
                aligned: Sequence[Optional[int]]) -> Dict[int, int]:
    """OCR line -> truth entry for emitted (non-partial) lines matched one-to-one to distinct
    entries with the same (known) event: the alignment's pairs first, then the rest in order."""
    def known(sig: Any) -> bool:
        return sig is not None and sig[0] != "unknown"

    pairs: Dict[int, int] = {}
    used = set()
    for i, j in enumerate(aligned):
        if j is not None and not o_partial[j] and known(o_sigs[j]) and o_sigs[j] == t_sigs[i]:
            pairs[j] = i
            used.add(i)
    for j, sig in enumerate(o_sigs):
        if j in pairs or o_partial[j] or not known(sig):
            continue
        for i, ts in enumerate(t_sigs):
            if i not in used and ts == sig:
                pairs[j] = i
                used.add(i)
                break
    return pairs


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(sample: Sample, ocr: Sequence[Sequence[Any]], result_box: Any) -> Dict[str, Any]:
    """Per-sample counts of one read (``ocr``: ``(text, partial, confidence[, key])`` lines)."""
    truth = sample.truth
    t_sigs = [text_signature(e.canonical) for e in truth]
    t_texts = [e.canonical for e in truth]
    o_sigs = [text_signature(o[0]) for o in ocr]
    o_texts = [o[0] for o in ocr]
    o_part = [bool(o[1]) for o in ocr]
    match = align(t_sigs, t_texts, o_sigs, o_texts)
    matched_by = {j: i for i, j in enumerate(match) if j is not None}
    tp = match_lines(t_sigs, o_sigs, o_part, match)
    visible_sigs = set(t_sigs)
    r: Dict[str, Any] = dict(entries=0, exact=0, emitted=0, lines=len(ocr), emitted_lines=0, tp=0, extra=0, dup=0,
                             false=0, partial_samples=0, part_viol=0, hi=0, hi_ok=0, lo=0, lo_ok=0, box_checked=0,
                             box_ok=0)
    verdicts = []
    for i, e in enumerate(truth):
        j = match[i]
        if e.partial:
            continue
        r["entries"] += 1
        got = ocr[j] if j is not None else None
        emitted = got is not None and not got[1]
        ok = emitted and o_sigs[j] == t_sigs[i]
        r["emitted"] += int(emitted)
        r["exact"] += int(ok)
        verdicts.append((i, j, "ok" if ok else "wrong" if emitted else "missing"))
    extras = []
    for j, o in enumerate(ocr):
        if o[1]:
            continue
        r["emitted_lines"] += 1
        if j in tp:
            r["tp"] += 1
        else:
            extras.append(j)
            r["extra"] += 1
            if o_sigs[j] is None or o_sigs[j][0] == "unknown" or o_sigs[j] not in visible_sigs:
                r["false"] += 1
            else:
                r["dup"] += 1
        ok = j in tp and not truth[tp[j]].partial
        if o[2] >= CONF_HI:
            r["hi"] += 1
            r["hi_ok"] += int(ok)
        else:
            r["lo"] += 1
            r["lo_ok"] += int(ok)
    parts = [i for i, e in enumerate(truth) if e.partial]
    if parts:
        r["partial_samples"] = 1
        first_full = min((match[i] for i, e in enumerate(truth) if not e.partial and match[i] is not None),
                         default=len(ocr))
        bad = False
        for i in parts:
            j = match[i]
            if j is not None and not ocr[j][1] and ocr[j][2] >= CONF_HI:
                bad = True
        for j in range(first_full):
            if j not in matched_by and not ocr[j][1] and ocr[j][2] >= CONF_HI:
                bad = True
        r["part_viol"] = int(bad)
    if result_box is not None:
        try:
            rb = tuple(float(v) for v in result_box)
            r["box_checked"] = 1
            r["box_ok"] = int(_iou(rb, sample.box) >= BOX_IOU)
        except (TypeError, ValueError):
            pass
    r["verdicts"] = verdicts
    r["match"] = match
    r["extras"] = extras
    return r


def scroll_check(prev_truth: Sequence[Any], prev_ocr: Sequence[OcrLine], truth: Sequence[Any],
                 ocr: Sequence[OcrLine], match: Optional[Sequence[Optional[int]]] = None, check_key: bool = True
                 ) -> Tuple[int, int, List[str]]:
    """``(pairs, stable, problems)``: the entries fully visible in both frames, how many got the same
    text, partial flag and - with ``check_key`` - (non-empty) key in both, and a description of each
    unstable one.  Keys are only compared on lossless, unzoomed frames: noise, JPEG blocks and
    resampling are fixed to image positions, so a scrolled entry's pixels legitimately change."""
    def aligned(tr: Sequence[Any], oc: Sequence[OcrLine]) -> List[Optional[int]]:
        return align([text_signature(e.canonical) for e in tr], [e.canonical for e in tr],
                     [text_signature(o[0]) for o in oc], [o[0] for o in oc])

    ma = aligned(prev_truth, prev_ocr)
    mb = list(match) if match is not None else aligned(truth, ocr)
    in_prev = {e.index: i for i, e in enumerate(prev_truth) if not e.partial}
    pairs = stable = 0
    problems: List[str] = []
    for i, e in enumerate(truth):
        if e.partial or e.index not in in_prev:
            continue
        pairs += 1
        ja, jb = ma[in_prev[e.index]], mb[i]
        a = prev_ocr[ja] if ja is not None else None
        b = ocr[jb] if jb is not None else None
        if a is not None and b is not None and a[0] == b[0] and a[1] == b[1] and a[3] not in (None, "") \
                and b[3] not in (None, "") and (a[3] == b[3] or not check_key):
            stable += 1
        else:
            problems.append(f"[{e.index}] {e.canonical!r}: before {a[:2] + (a[3],) if a else None} "
                            f"after {b[:2] + (b[3],) if b else None}")
    return pairs, stable, problems


_SUM_KEYS = ("samples", "entries", "exact", "emitted", "lines", "emitted_lines", "tp", "extra", "dup", "false",
             "partial_samples", "part_viol", "hi", "hi_ok", "lo", "lo_ok", "box_checked", "box_ok", "det_checked",
             "det_ok", "det_iou", "scroll_pairs", "scroll_ok", "fresh_pairs", "fresh_ok", "cold_s", "warm_s", "timed",
             "warm_same", "after_entries", "after_exact", "after_extra", "key_clash")


def _add(acc: Dict[str, float], r: Dict[str, Any]) -> None:
    for k in _SUM_KEYS:
        acc[k] = acc.get(k, 0) + r.get(k, 0)


def _rates(acc: Dict[str, float]) -> Dict[str, Any]:
    def div(a: float, b: float) -> Optional[float]:
        return round(a / b, 4) if b else None
    g = lambda k: acc.get(k, 0)  # noqa: E731
    return {
        "samples": int(g("samples")), "entries": int(g("entries")),
        "exact": div(g("exact"), g("entries")), "emitted": div(g("emitted"), g("entries")),
        "lines": int(g("emitted_lines")), "precision": div(g("tp"), g("emitted_lines")),
        "extra": int(g("extra")), "dup": int(g("dup")),
        "false": int(g("false")), "false_rate": div(g("false"), g("emitted_lines")),
        "partial_samples": int(g("partial_samples")), "part_viol": int(g("part_viol")),
        "partial_ok": div(g("partial_samples") - g("part_viol"), g("partial_samples")),
        "hi_n": int(g("hi")), "hi_acc": div(g("hi_ok"), g("hi")),
        "lo_n": int(g("lo")), "lo_acc": div(g("lo_ok"), g("lo")),
        "box_ok": div(g("box_ok"), g("box_checked")),
        "detect": div(g("det_ok"), g("det_checked")), "detect_iou": div(g("det_iou"), g("det_checked")),
        "scroll_pairs": int(g("scroll_pairs")), "scroll": div(g("scroll_ok"), g("scroll_pairs")),
        "scroll_fresh": div(g("fresh_ok"), g("fresh_pairs")),
        "cold_ms": div(1000 * g("cold_s"), g("timed")), "warm_ms": div(1000 * g("warm_s"), g("timed")),
        "warm_same": div(g("warm_same"), g("timed")),
        "after_exact": div(g("after_exact"), g("after_entries")), "after_extra": int(g("after_extra")),
        "key_clash": int(g("key_clash")),
    }


def _pct(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{100 * v:5.1f}"


def format_table(summary: Dict[str, Any]) -> str:
    head = (f"{'condition':<28}{'n':>5}{'ent':>6}{'exact%':>8}{'emit%':>7}{'prec%':>7}{'extra':>6}{'false':>6}"
            f"{'partOK%':>8}{'hi-acc%':>8}{'(n)':>6}{'lo-acc%':>8}{'(n)':>5}{'box%':>6}{'det%':>6}{'scrl%':>6}"
            f"{'fresh%':>7}{'same%':>6}{'aftr%':>6}{'clash':>6}{'cold ms':>9}{'warm ms':>8}")
    out = [head, "-" * len(head)]

    def row(label: str, m: Dict[str, Any]) -> str:
        ms = lambda v: "   -  " if v is None else f"{v:8.1f}"  # noqa: E731
        return (f"{label:<28}{m['samples']:>5}{m['entries']:>6}{_pct(m['exact']):>8}{_pct(m['emitted']):>7}"
                f"{_pct(m['precision']):>7}{m['extra']:>6}{m['false']:>6}{_pct(m['partial_ok']):>8}"
                f"{_pct(m['hi_acc']):>8}{m['hi_n']:>6}{_pct(m['lo_acc']):>8}{m['lo_n']:>5}{_pct(m['box_ok']):>6}"
                f"{_pct(m['detect']):>6}{_pct(m['scroll']):>6}{_pct(m['scroll_fresh']):>7}{_pct(m['warm_same']):>6}"
                f"{_pct(m['after_exact']):>6}{m['key_clash']:>6}{ms(m['cold_ms']):>9}{ms(m['warm_ms']):>8}")
    out.append(row("OVERALL", summary["overall"]))
    for cond, vals in summary["by_condition"].items():
        for v, m in sorted(vals.items()):
            out.append(row(f"{cond}={v}", m))
    return "\n".join(out)


def _save_failure(save_dir: str, s: Sample, ocr: Sequence[OcrLine], r: Dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    x0, y0, x1, y1 = s.box
    pad = 8
    crop_box = (max(0, int(x0) - pad), max(0, int(y0) - pad), min(s.img.width, int(x1) + pad),
                min(s.img.height, int(y1) + pad))
    stem = os.path.join(save_dir, f"sample_{s.id:04d}")
    s.img.crop(crop_box).save(stem + ".png")
    match = r["match"]
    with open(stem + ".txt", "w") as f:
        f.write("conditions: " + json.dumps(s.cond, sort_keys=True) + "\n")
        f.write("names: " + json.dumps(s.names) + f"\npanel box: {tuple(round(v, 1) for v in s.box)}  "
                f"result box ok: {r.get('box_ok') if r.get('box_checked') else '-'}  "
                f"detection IoU: {r.get('det_iou') if r.get('det_checked') else '-'}\n")
        f.write("\nTRUTH (canonical | partial | drawn line) -> OCR\n")
        for i, e in enumerate(s.truth):
            j = match[i]
            got = ocr[j] if j is not None else None
            ok = got is not None and text_signature(got[0]) == text_signature(e.canonical) and not got[1]
            mark = "ok " if ok else ("-- " if e.partial else "XX ")
            f.write(f"{mark}[{e.index}] {e.canonical!r}{' (partial)' if e.partial else ''}  <- {e.text!r}\n")
            f.write(f"      ocr: {got[0]!r} partial={got[1]} conf={got[2]:.2f} key={got[3]!r}\n" if got
                    else "      ocr: (none)\n")
        if r.get("extras"):
            f.write("\nEXTRA lines (no distinct entry with their event):\n")
            for j in r["extras"]:
                f.write(f"  {ocr[j][0]!r} partial={ocr[j][1]} conf={ocr[j][2]:.2f}\n")
        for what in ("scroll_problems", "fresh_problems"):
            if r.get(what):
                f.write(f"\n{what.replace('_', ' ').upper()} (previous frame -> this frame):\n")
                f.writelines(f"  {p}\n" for p in r[what])
        if not r.get("warm_same", 1):
            f.write("\nthe warm re-read (same image, same cache) returned different lines\n")
    if s.prev_img is not None and (r.get("scroll_problems") or r.get("fresh_problems")):
        s.prev_img.crop(crop_box).save(stem + "_prev.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def failed(r: Dict[str, Any]) -> bool:
    """Whether a sample's result (:func:`evaluate_sample`) fails any metric."""
    return bool(r["exact"] < r["entries"] or r["extra"] or r["part_viol"] or not r.get("warm_same", 1)
                or (r["box_checked"] and not r["box_ok"]) or (r.get("det_checked") and not r.get("det_ok"))
                or r.get("scroll_ok", 0) < r.get("scroll_pairs", 0) or r.get("fresh_ok", 0) < r.get("fresh_pairs", 0)
                or r.get("after_exact", 0) < r.get("after_entries", 0) or r.get("after_extra", 0)
                or r.get("key_clash", 0))


def key_clashes(ocr: Sequence[OcrLine]) -> int:
    """Pairs of emitted (non-partial) lines that share a key but read differently: a content hash
    may only collide for entries drawn identically."""
    seen: Dict[Any, str] = {}
    clashes = 0
    for text, partial, _conf, key in ocr:
        if partial or key in (None, ""):
            continue
        if key in seen and seen[key] != text:
            clashes += 1
        seen.setdefault(key, text)
    return clashes


def lossless(s: Sample) -> bool:
    """Frames whose entry pixels move with the entry on a scroll (no noise, JPEG or zoom)."""
    return s.cond.get("jpeg") == "off" and s.cond.get("noise") == "none" and s.cond.get("scale") == "none"


def evaluate_sample(s: Sample, reader: Reader, give_box: bool = False, give_players: bool = False
                    ) -> Tuple[Dict[str, Any], List[OcrLine]]:
    """Read sample ``s`` with ``reader`` and score it: ``(counts, lines of the cold read)``.

    Reads: cold (fresh cache) and warm (same cache, same image); the panel detection
    (``reader.find``, else the result's box; with ``give_box`` and no finder a read without the
    box); the scroll pair: the previous frame then this frame with one fresh cache (``scroll``),
    and the previous frame's read against the cold read (a fresh cache each: ``scroll_fresh``).
    """
    _ORACLE_FRAMES.clear()
    _ORACLE_FRAMES[id(s.img)] = (s.truth, s.box)
    if s.prev_img is not None:
        _ORACLE_FRAMES[id(s.prev_img)] = (s.prev_truth or [], s.box)
    new_cache = (lambda: reader.cache_cls()) if reader.cache_cls is not None else (lambda: None)  # noqa: E731
    box = s.box if give_box else None
    kw = dict(profile=None, players=s.name_rgb if give_players else None)
    try:
        cache = new_cache()
        t1 = time.perf_counter()
        res = reader.read(s.img, box=box, cache=cache, **kw)
        cold = time.perf_counter() - t1
        t1 = time.perf_counter()
        res2 = reader.read(s.img, box=box, cache=cache, **kw)
        warm = time.perf_counter() - t1
        ocr = _lines_of(res)
        r = score(s, ocr, _box_of(res))
        r.update(samples=1, cold_s=cold, warm_s=warm, timed=1, key_clash=key_clashes(ocr),
                 warm_same=int(_lines_of(res2) == ocr and _box_of(res2) == _box_of(res)))
        # panel detection, with and without --give-box
        if reader.find is not None:
            det = reader.find(s.img, profile=None)
        elif not give_box:
            det = _box_of(res)
        else:
            det = _box_of(reader.read(s.img, box=None, cache=new_cache(), **kw))
        if det is not None or reader.find is not None or not give_box:
            iou = _iou(tuple(float(v) for v in det), s.box) if det is not None else 0.0   # None = a miss
            r.update(det_checked=1, det_ok=int(iou >= BOX_IOU), det_iou=iou)
        # the scroll pair
        if s.prev_img is not None and s.prev_truth is not None:
            c2 = new_cache()
            prev = _lines_of(reader.read(s.prev_img, box=box, cache=c2, **kw))     # c2 is fresh here
            after = _lines_of(reader.read(s.img, box=box, cache=c2, **kw))
            ra = score(s, after, None)          # the warm read of the new frame is scored like the cold one
            r.update(after_entries=ra["entries"], after_exact=ra["exact"], after_extra=ra["extra"])
            p, ok, probs = scroll_check(s.prev_truth, prev, s.truth, after, check_key=lossless(s))
            r.update(scroll_pairs=p, scroll_ok=ok, scroll_problems=probs)
            p, ok, probs = scroll_check(s.prev_truth, prev, s.truth, ocr, r["match"], check_key=lossless(s))
            r.update(fresh_pairs=p, fresh_ok=ok, fresh_problems=probs)
    finally:
        _ORACLE_FRAMES.clear()
    return r, ocr


def run(args: argparse.Namespace, reader: Optional[Reader] = None) -> Dict[str, Any]:
    reader = reader if reader is not None else load_reader(args.reader)
    fonts = {k: v for k, v in FONTS.items() if all(os.path.exists(p) for p in v)}
    n = args.n if args.n else (90 if args.quick else 900)
    seeds = list(range(1, (3 if n <= 100 else 9) + 1))
    want: Dict[str, str] = {}
    for kv in args.only:
        c, _, v = kv.partition("=")
        if c not in CONDITIONS or not v:
            raise SystemExit(f"--only {kv!r}: expected CONDITION=VALUE with CONDITION one of {', '.join(CONDITIONS)}")
        want[c] = v
    t0 = time.perf_counter()
    games = build_games(seeds, STYLES, per_game=max(4, n // (len(seeds) * len(STYLES)) + 4))
    t_games = time.perf_counter() - t0
    overall: Dict[str, float] = {}
    by: Dict[str, Dict[str, Dict[str, float]]] = {c: {} for c in CONDITIONS}
    startup_ms = None
    failures = 0
    t_render = 0.0
    scroll = not getattr(args, "no_scroll", False)
    k = done = 0
    while done < n and k < 100 * n:
        t1 = time.perf_counter()
        s = make_sample(k, games, args.seed, fonts, want, scroll=scroll)
        t_render += time.perf_counter() - t1
        k += 1
        if s is None:
            continue
        done += 1
        if startup_ms is None:       # first call: templates / models load; not part of the timings
            _ORACLE_FRAMES[id(s.img)] = (s.truth, s.box)
            t1 = time.perf_counter()
            reader.read(s.img, box=s.box if args.give_box else None, profile=None,
                        players=s.name_rgb if args.give_players else None, cache=None)
            startup_ms = 1000 * (time.perf_counter() - t1)
        r, ocr = evaluate_sample(s, reader, args.give_box, args.give_players)
        _add(overall, r)
        for c in CONDITIONS:
            _add(by[c].setdefault(s.cond[c], {}), r)
        bad = failed(r)
        failures += int(bad)
        if bad and args.save_dir:
            _save_failure(args.save_dir, s, ocr, r)
        if args.verbose:
            print(f"sample {s.id}: {s.cond} exact {r['exact']}/{r['entries']} extra {r['extra']} "
                  f"part_viol {r['part_viol']} scroll {r.get('scroll_ok', 0)}/{r.get('scroll_pairs', 0)} "
                  f"cold {1000 * r['cold_s']:.1f} ms", file=sys.stderr)
    return {
        "reader": args.reader, "mode": "quick" if args.quick else "full", "samples": done, "seed": args.seed,
        "only": want, "scroll": scroll,
        "give_box": bool(args.give_box), "give_players": bool(args.give_players),
        "fonts": sorted(fonts), "startup_ms": None if startup_ms is None else round(startup_ms, 1),
        "games_s": round(t_games, 2), "render_s": round(t_render, 2), "failed_samples": failures,
        "overall": _rates(overall),
        "by_condition": {c: {v: _rates(acc) for v, acc in vals.items()} for c, vals in by.items()},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="catanbot.vision.logocr:read_log_panel",
                    help="module:function of the reader, or 'oracle' (the ground truth; self-check), or "
                         "'oracle:FAULT,...' with FAULT one of " + ", ".join(ORACLE_FAULTS))
    ap.add_argument("--quick", action="store_true", help="~90 samples (default: full, ~900)")
    ap.add_argument("--n", type=int, default=0, help="number of samples (overrides --quick)")
    ap.add_argument("--seed", type=int, default=0, help="benchmark seed (default 0)")
    ap.add_argument("--only", action="append", default=[], metavar="COND=VALUE",
                    help="only samples with this condition value, e.g. font=liberation, theme=dark, cut=0.6, "
                         "size=1920x1080, size=1280x800@x1.25, font_px=9, noise=blur, box=jittered (repeatable)")
    ap.add_argument("--give-box", action="store_true", help="pass the true panel box (default: the reader finds it)")
    ap.add_argument("--give-players", action="store_true",
                    help="pass the panel's name colours {colour: RGB} (default: the reader's palette)")
    ap.add_argument("--no-scroll", action="store_true", help="skip the scroll pairs (half the rendering)")
    ap.add_argument("--json", default="", help="write the JSON summary here")
    ap.add_argument("--save-dir", default="", help="dump failing samples here (panel PNG + truth / OCR text)")
    ap.add_argument("--min-exact", type=float, default=None, help="exit 1 when the overall exact rate is below this")
    ap.add_argument("--max-failed", type=float, default=None,
                    help="exit 1 when more than this fraction of the samples fail any metric (0 = every sample "
                         "must pass)")
    ap.add_argument("-v", "--verbose", action="store_true", help="one line per sample on stderr")
    args = ap.parse_args(argv)
    if args.n < 0:
        ap.error("--n must be >= 0")
    try:
        reader = load_reader(args.reader)
    except (ImportError, AttributeError, ValueError) as exc:
        print(f"cannot load the reader '{args.reader}': {exc}", file=sys.stderr)
        return 2
    summary = run(args, reader)
    if not summary["samples"]:
        print(f"no sample matches --only {' '.join(args.only)} (see the condition values in a full run's table)",
              file=sys.stderr)
        return 2
    print(f"reader {summary['reader']}  samples {summary['samples']}  failed samples {summary['failed_samples']}  "
          f"scroll pairs {summary['overall']['scroll_pairs']}  startup {summary['startup_ms']} ms  "
          f"(games {summary['games_s']} s, rendering {summary['render_s']} s)")
    print(format_table(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
    if args.min_exact is not None and (summary["overall"]["exact"] or 0.0) < args.min_exact:
        return 1
    if args.max_failed is not None and summary["failed_samples"] > args.max_failed * summary["samples"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
