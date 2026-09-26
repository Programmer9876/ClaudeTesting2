"""Local game-log reader (log OCR) for Colonist.io screenshots: numpy / Pillow / OpenCV only.

Pipeline (``read_log_panel``)
    image (path / PIL / numpy RGB)
    -> the panel box: the caller's, a profile's ``log`` region, or :func:`find_log_panel`
    -> :func:`catanbot.vision.logocr_layout.analyse_panel`: background rows (panel fill and stripe
       estimated from the image), ink, text rows, glyph pieces, card / die / building icons (shape
       + colour),
       player names (coloured words; the colour word from the unmixed ink colour), entries (stripe
       changes and the continuation-row indent), the cut top entry (``partial``) and each entry's
       content key (a hash of its normalised ink);
    -> for each entry not in the :class:`LineCache`: text runs are over-segmented and every
       candidate glyph is classified by the numpy MLP of :mod:`catanbot.vision.logocr_glyphs`
       (one batch for the whole panel); :mod:`catanbot.vision.logocr_decode` aligns the log
       vocabulary to the cuts (word lattice) and picks the words with a bigram phrase model;
    -> canonical text (names as colour words, icons as ``N res`` / ``a card`` / ``N cards`` /
       ``Development Card`` / dice faces / ``Road`` / ``Settlement`` / ``City``; a verb's glued
       colon, "got:", read and left out) that must parse with
       :func:`catanbot.colonist_log.parse_log_line`, with a confidence: the weakest of the name
       colour, icon and word evidence; lines below 0.6 are not to be fed to the card counter.

``teach(img, truth_lines)`` learns from one labelled screenshot of a real screen: the panel
region, the name colours and word exemplars of the screen's font (a sidecar ``.npz`` next to the
profile), which later reads use (``read_log_panel(profile=...)``).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import colonist_log as L
from . import logocr_decode as D
from . import logocr_glyphs as G
from . import logocr_layout as LL
from .profile import UiProfile

__all__ = [
    "LogLine",
    "LogReadResult",
    "LineCache",
    "find_log_panel",
    "read_log_panel",
    "teach",
    "event_signature",
    "CONF_FEED",
]

Box = Tuple[int, int, int, int]
#: Lines at or above this confidence may be fed to the card counter.
CONF_FEED = 0.6
_READER_VERSION = "b2.2"


# ---------------------------------------------------------------------------
# results and cache
# ---------------------------------------------------------------------------
@dataclass
class LogLine:
    """One log entry as read: canonical ``text``, its parsed ``event``, ``confidence`` 0..1, ``box``
    (image pixels), ``key`` (content hash of the entry's normalised ink), ``partial`` (cut by the
    panel's top edge / head row missing) and debug ``tokens``."""

    text: str
    event: Any
    confidence: float
    box: Tuple[float, float, float, float]
    key: str
    partial: bool = False
    tokens: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LogReadResult:
    """A read panel: ``lines`` oldest first, the panel ``box``, ``warnings``, overall ``confidence``
    and ``debug`` details."""

    lines: List[LogLine]
    box: Optional[Tuple[int, int, int, int]]
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)

    def texts(self, min_conf: float = CONF_FEED, include_partial: bool = False) -> List[str]:
        """Texts of the lines at or above ``min_conf`` (partial lines only with ``include_partial``)."""
        return [ln.text for ln in self.lines if ln.confidence >= min_conf and (include_partial or not ln.partial)]


class LineCache:
    """Bounded LRU cache: entry key -> reading (so an unchanged entry reads identically on every
    frame), plus the panel box of recently seen frames (a warm re-read skips the detection)."""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = max(1, int(maxsize))
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self._boxes: "OrderedDict[str, Any]" = OrderedDict()
        self._frames: "OrderedDict[str, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
            self.hits += 1
        else:
            self.misses += 1
        return v

    def put(self, key: str, value: Any) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def get_frame(self, key: str) -> Any:
        return self._frames.get(key)

    def put_frame(self, key: str, res: Any) -> None:
        self._frames[key] = res
        self._frames.move_to_end(key)
        while len(self._frames) > 4:
            self._frames.popitem(last=False)

    def get_box(self, fp: str) -> Any:
        v = self._boxes.get(fp)
        if v is not None:
            self._boxes.move_to_end(fp)
        return v

    def put_box(self, fp: str, box: Any) -> None:
        self._boxes[fp] = box
        self._boxes.move_to_end(fp)
        while len(self._boxes) > 8:
            self._boxes.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def clear(self) -> None:
        self._d.clear()
        self._boxes.clear()
        self._frames.clear()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def event_signature(ev: Any, name_to_colour: Optional[Dict[str, str]] = None) -> Optional[tuple]:
    """Tracker-style key of an event with players as colour words ("you" for the viewer):
    ``(kind, player, other, cards, get, count, value, item, resource, free)``."""
    if ev is None:
        return None
    names = {re.sub(r"\s+", " ", k.strip().lower()): v for k, v in (name_to_colour or {}).items()}
    stripped = {re.sub(r"[^\w#]+", "", k): v for k, v in names.items()}

    def who(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        k = re.sub(r"\s+", " ", x.strip().lower())
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


def _fingerprint(arr: np.ndarray) -> str:
    """Cheap identity of a frame (a strided sample of its pixels)."""
    h = hashlib.blake2b(digest_size=12)
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    step = max(1, int(math.sqrt(arr.shape[0] * arr.shape[1] / 40000.0)))
    h.update(np.ascontiguousarray(arr[::step, ::step]).tobytes())
    return h.hexdigest()


def _profile_box(profile: Optional[UiProfile], size: Tuple[int, int]) -> Optional[Box]:
    if profile is None:
        return None
    try:
        return profile.pixel_box("log", size)
    except Exception:
        return None


def find_log_panel(img: Any, profile: Optional[UiProfile] = None) -> Optional[Box]:
    """The log panel in ``img`` as pixels ``(x0, y0, x1, y1)``: the profile's ``log`` region when it
    has one, else detected (a rounded rectangle of flat colour with text rows); ``None`` when none
    is found."""
    arr = LL.as_rgb(img)
    pb = _profile_box(profile, (arr.shape[1], arr.shape[0]))
    if pb is not None:
        return pb
    box, _ = LL.detect_panel(arr)
    return box


def _palette(players: Optional[Dict[str, Any]], profile: Optional[UiProfile]) -> Dict[str, List[Tuple[float, ...]]]:
    extra = None
    if profile is not None and isinstance(profile.ocr, dict):
        extra = profile.ocr.get("name_colours")
    if players:
        pal = LL.name_palette(players, extra)
    else:
        pal = LL.name_palette(None, extra)
    return pal


# ---------------------------------------------------------------------------
# entry tokens -> text
# ---------------------------------------------------------------------------
_PLAYER_BEFORE = {"from", "with", "to"}


def _entry_tokens(entry: LL.Entry) -> List[Any]:
    out = []
    for b in entry.bands:
        for t in b.items:
            out.append((b, t))
    return out


def _lm_symbol_of(tok: Any) -> str:
    if isinstance(tok, LL.NameTok):
        return "NAME"
    if isinstance(tok, LL.Icon):
        if tok.kind == "die":
            return "DICE"
        if tok.kind == "dev":
            return "DEV"
        if tok.kind in D.BUILDING_WORDS:
            return D.BUILDING_WORDS[tok.kind]
        return "CARDS"
    return "UNK"


def _assemble(lay: LL.PanelLayout, entry: LL.Entry, readings: Dict[int, D.RunReading]
              ) -> Tuple[str, float, List[Dict[str, Any]], List[str]]:
    """Canonical text of an entry from its tokens and the runs' readings:
    ``(text, confidence, debug tokens, notes)``."""
    toks = _entry_tokens(entry)
    parts: List[List[Any]] = []          # [text, glue_to_previous, kind]
    confs: List[float] = []
    dbg: List[Dict[str, Any]] = []
    notes: List[str] = []
    em = max(1.0, lay.em)
    k = 0
    n = len(toks)
    while k < n:
        band, t = toks[k]
        if isinstance(t, LL.NameTok):
            glue = False
            parts.append([t.colour, glue, "name"])
            confs.append(t.conf)
            dbg.append({"kind": "name", "colour": t.colour, "conf": round(t.conf, 3),
                        "rgb": [int(v) for v in getattr(t, "rgb", (0, 0, 0))], "x": [t.x0, t.x1]})
            k += 1
            continue
        if isinstance(t, LL.Icon):
            group = [t]
            k += 1
            while k < n and isinstance(toks[k][1], LL.Icon) and toks[k][0] is band \
                    and toks[k][1].x0 - group[-1].x1 <= 0.6 * em:
                group.append(toks[k][1])
                k += 1
            kinds = [("die%d" % g.face) if g.kind == "die" else g.kind for g in group]
            mult = None
            cards = [x for x in kinds if x in D._RES or x == "card"]
            if cards and parts and parts[-1][2] == "word" and re.fullmatch(r"\d+", parts[-1][0]):
                mult = int(parts[-1][0])
                parts.pop()
            txt = D.icon_text(kinds, mult)
            parts.append([txt, False, "icons"])
            confs.extend(g.conf for g in group)
            dbg.append({"kind": "icons", "icons": kinds, "mult": mult,
                        "conf": [round(g.conf, 3) for g in group]})
            continue
        # a text run
        rr = readings.get(id(t))
        k += 1
        if rr is None or not rr.words:
            confs.append(0.0)
            notes.append("unread text run")
            continue
        for wi, w in enumerate(rr.words):
            if w == D.NAME_WORD:
                parts.append(["white", False, "name"])
                confs.append(0.7)
                notes.append("text-coloured name read as the grey player")
                continue
            glue = w in (":", ",") or (w.startswith(":") and wi == 0)
            if not rr.unknown[wi]:
                w = D.drop_verb_colon(w)      # "got:" reads as "got": the colon is punctuation
            parts.append([w, glue, "unk" if rr.unknown[wi] else "word"])
        confs.append(rr.conf)
        dbg.append({"kind": "text", "words": list(rr.words), "unknown": list(rr.unknown),
                    "scores": [round(s, 3) for s in rr.scores], "conf": round(rr.conf, 3)})
    # player slots filled by an unknown text-coloured word: the grey (white) player
    def is_player_slot(i: int) -> bool:
        if i == 0:
            return True
        return parts[i - 1][2] == "word" and parts[i - 1][0].lower().rstrip(":") in _PLAYER_BEFORE
    i = 0
    while i < len(parts):
        if parts[i][2] == "unk" and is_player_slot(i):
            j = i
            while j + 1 < len(parts) and parts[j + 1][2] == "unk" and j - i < 2:
                j += 1
            parts[i:j + 1] = [["white", False, "name"]]
            notes.append("text-coloured name read as the grey player")
        i += 1
    text = ""
    for s, glue, kind in parts:
        if not text:
            text = s
        elif glue:
            text += s
        else:
            text += " " + s
    conf = min(confs) if confs else 0.0
    return text.strip(), float(conf), dbg, notes


def _symbolic_key(text: str, rows: int) -> str:
    """Public entry key: what the entry shows (its canonical reading) and on how many rows - not its
    pixels, so noise, JPEG blocks and scrolling do not change it; entries share a key only when they
    read the same (the pixel hash stays the cache key)."""
    norm = re.sub(r"\s+", " ", text.strip())
    return hashlib.blake2b(f"{norm}|{rows}".encode(), digest_size=12).hexdigest()


def _parse_ok(text: str) -> Tuple[Any, bool]:
    ev = L.parse_log_line(text) if text else None
    ok = ev is not None and ev.kind != "unknown" and not ev.problem
    return ev, ok


def _repair_numbers(text: str) -> Optional[str]:
    """A line that does not parse because a number's digits were read as two words ("stole 1 1
    brick": wide digit spacing in some fonts): the first join of adjacent digit words that makes
    it parse, else None."""
    words = text.split(" ")
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        if a.isdigit() and b.isdigit() and len(a) + len(b) <= 2:
            cand = " ".join(words[:i] + [a + b] + words[i + 2:])
            if _parse_ok(cand)[1]:
                return cand
    return None


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------
def _decode_entries(lay: LL.PanelLayout, entries: Sequence[LL.Entry], clf: G.GlyphClassifier
                    ) -> Dict[int, Tuple[str, float, List[Dict[str, Any]], List[str]]]:
    """Read ``entries`` (one classifier batch for all their runs)."""
    jobs = []            # (entry, band, run, ri, bounds, gaps, pairs, lo, hi)
    feats = []
    off = 0
    for e in entries:
        for band in e.bands:
            for t in band.items:
                if not isinstance(t, LL.TextRun):
                    continue
                ri = G.run_ink(lay, band, t)
                bounds, gaps = G.cut_candidates(ri.A, ri.xh)
                if len(bounds) < 2:
                    jobs.append((e, band, t, ri, bounds, gaps, [], off, off))
                    continue
                X, pairs = G.segment_features(ri, bounds, gaps)
                feats.append(X)
                jobs.append((e, band, t, ri, bounds, gaps, pairs, off, off + X.shape[0]))
                off += X.shape[0]
    if feats:
        Xall = np.concatenate(feats) * G.STORE_SCALE
        LP = clf.log_proba(Xall)
    else:
        LP = np.zeros((0, G.NUM_CLASSES), dtype=np.float32)
    readings: Dict[int, D.RunReading] = {}
    prev_sym: Dict[int, str] = {}
    next_sym: Dict[int, str] = {}
    # LM context: the tokens around each run
    for e in entries:
        toks = _entry_tokens(e)
        prev = "<s>"
        for k, (band, t) in enumerate(toks):
            prev_sym[id(t)] = prev
            nt = toks[k + 1][1] if k + 1 < len(toks) else None
            next_sym[id(t)] = "</s>" if nt is None else (_lm_symbol_of(nt) if not isinstance(nt, LL.TextRun) else "")
            prev = _lm_symbol_of(t) if not isinstance(t, LL.TextRun) else "UNK"
    for (e, band, t, ri, bounds, gaps, pairs, lo, hi) in jobs:
        if hi <= lo:
            continue
        rr = D.decode_run(LP[lo:hi], pairs, bounds, gaps, ri.xh, prev=prev_sym.get(id(t), "<s>"), run_x0=ri.x0,
                          nxt=next_sym.get(id(t)) or None)
        readings[id(t)] = rr
    out = {}
    for e in entries:
        out[id(e)] = _assemble(lay, e, readings)
    return out


def read_log_panel(img: Any, box: Optional[Sequence[float]] = None, profile: Optional[UiProfile] = None,
                   players: Optional[Dict[str, Any]] = None, cache: Optional[LineCache] = None) -> LogReadResult:
    """Read the game-log panel of ``img`` (see the module docstring).

    ``box`` (pixels) wins over the profile's ``log`` region and the detection; ``players``
    ({colour word: RGB}) are this screen's name colours (default: the palette, or what ``teach``
    stored in ``profile``); ``cache`` (:class:`LineCache`) keeps each entry's reading by its content
    key across frames.
    """
    t0 = time.perf_counter()
    arr = LL.as_rgb(img)
    H, W = arr.shape[:2]
    warnings: List[str] = []
    fp = None
    if box is not None:
        bx = tuple(int(round(float(v))) for v in box)
    else:
        bx = _profile_box(profile, (W, H))
        if bx is None:
            if cache is not None:
                fp = _fingerprint(arr)
                bx = cache.get_box(fp)
            if bx is None:
                bx, _ = LL.detect_panel(arr)
                if cache is not None and fp is not None:
                    cache.put_box(fp, bx if bx is not None else ())
            elif bx == ():
                bx = None
    if not bx:
        return LogReadResult(lines=[], box=None, warnings=["no log panel found"], confidence=0.0,
                             debug={"ms": 1000 * (time.perf_counter() - t0)})
    bx = (max(0, bx[0]), max(0, bx[1]), min(W, bx[2]), min(H, bx[3]))
    pal = _palette(players, profile)
    # an identical panel (same pixels, same settings) reads identically: the whole result is reused
    frame_key = None
    if cache is not None:
        hh = hashlib.blake2b(digest_size=16)
        hh.update(repr((bx, sorted((k, [tuple(r) for r in v]) for k, v in pal.items()), _profile_token(profile),
                        _READER_VERSION)).encode())
        hh.update(np.ascontiguousarray(arr[bx[1]:bx[3], bx[0]:bx[2]]).tobytes())
        frame_key = hh.hexdigest()
        prev_res = cache.get_frame(frame_key)
        if prev_res is not None:
            return _copy_result(prev_res, t0)
    lay = LL.analyse_panel(arr, bx, palette=pal)
    ix0, iy0 = lay.inner[0], lay.inner[1]
    clf = _classifier(profile)
    todo = []
    cached: Dict[int, Any] = {}
    first_of: Dict[str, LL.Entry] = {}
    for e in lay.entries:
        ck = f"{_READER_VERSION}:{e.key}:{int(e.partial)}"
        hit = cache.get(ck) if cache is not None else None
        if hit is not None:
            cached[id(e)] = hit
        elif ck not in first_of:        # identical entries (same key) are read once
            first_of[ck] = e
            todo.append(e)
    decoded = _decode_entries(lay, todo, clf) if todo else {}
    fresh: Dict[str, Any] = {}
    lines: List[LogLine] = []
    for e in lay.entries:
        ck = f"{_READER_VERSION}:{e.key}:{int(e.partial)}"
        if id(e) in cached:
            text, conf, dbg, notes = cached[id(e)]
        elif ck in fresh:
            text, conf, dbg, notes = fresh[ck]
        else:
            text, conf, dbg, notes = decoded[id(first_of[ck])]
            ev0, ok0 = _parse_ok(text)
            if not ok0:
                fixed = _repair_numbers(text)
                if fixed is not None:
                    text, conf = fixed, min(conf, 0.55)
                    notes = list(notes) + ["digits split by a wide gap joined into one number"]
                else:
                    conf = min(conf, 0.35)
            fresh[ck] = (text, conf, dbg, notes)
            if cache is not None:
                cache.put(ck, (text, conf, dbg, notes))
        ev, ok = _parse_ok(text)
        c = float(conf)
        if e.partial:
            c = min(c, 0.3)
        y0 = min(b.y0 for b in e.bands) + iy0
        y1 = max(b.y1 for b in e.bands) + iy0
        lines.append(LogLine(text=text, event=ev, confidence=round(c, 4),
                             box=(float(lay.inner[0]), float(y0), float(lay.inner[2]), float(y1)),
                             key=_symbolic_key(text, len(e.bands)), partial=e.partial, tokens=list(dbg)))
    full = [ln.confidence for ln in lines if not ln.partial]
    conf = float(np.mean(full)) if full else 0.0
    dbg = {"ms": round(1000 * (time.perf_counter() - t0), 2), "em": lay.em, "xh": lay.xh, "pitch": lay.pitch,
           "dark": lay.dark, "noise": lay.noise, "inner": lay.inner, "decoded": len(todo),
           "cached": len(cached)}
    res = LogReadResult(lines=lines, box=tuple(int(v) for v in bx), warnings=warnings + lay.warnings,
                        confidence=conf, debug=dbg)
    if cache is not None and frame_key is not None:
        cache.put_frame(frame_key, res)
        res = _copy_result(res, None)
    return res


def _copy_result(res: LogReadResult, t0: Optional[float]) -> LogReadResult:
    lines = [LogLine(text=ln.text, event=ln.event, confidence=ln.confidence, box=ln.box, key=ln.key,
                     partial=ln.partial, tokens=list(ln.tokens)) for ln in res.lines]
    dbg = dict(res.debug)
    if t0 is not None:
        dbg["ms"] = round(1000 * (time.perf_counter() - t0), 2)
        dbg["frame_cache"] = True
    return LogReadResult(lines=lines, box=res.box, warnings=list(res.warnings), confidence=res.confidence, debug=dbg)


def _profile_token(profile: Optional[UiProfile]) -> Any:
    if profile is None:
        return None
    return (id(profile), len(profile.ocr) if isinstance(profile.ocr, dict) else 0, profile.ocr.get("taught_at")
            if isinstance(profile.ocr, dict) else None)


_ADAPT_CACHE: Dict[Any, Any] = {}


def _classifier(profile: Optional[UiProfile] = None) -> Any:
    """The glyph classifier: the shipped one, or its adaptation taught on this screen."""
    base = G.load_classifier()
    ocr = profile.ocr if profile is not None and isinstance(profile.ocr, dict) else {}
    ad = ocr.get("glyph_adapt")
    if not ad:
        return base
    key = (id(base), repr(ad)[:200], profile.path if profile is not None else None)
    clf = _ADAPT_CACHE.get(key)
    if clf is None:
        from . import logocr_teach as TE
        try:
            if isinstance(ad, dict) and ad.get("file"):
                path = profile.sidecar_path(ad["file"]) if profile is not None else None
                with np.load(path) as z:
                    arrs = {k: z[k] for k in z.files}
            elif isinstance(ad, dict) and ad.get("b64"):
                arrs = TE.decode_arrays(ad["b64"])
            else:
                return base
            if arrs["W"].shape != base.weights[-1].shape:
                return base
            clf = TE.AdaptedClassifier(base, arrs["W"], arrs["b"])
        except Exception:
            return base
        _ADAPT_CACHE[key] = clf
    return clf


# ---------------------------------------------------------------------------
# teaching
# ---------------------------------------------------------------------------
def teach(img: Any, truth_lines: Sequence[str], box: Optional[Sequence[float]] = None,
          profile: Optional[UiProfile] = None, players: Optional[Dict[str, Any]] = None
          ) -> Tuple[UiProfile, Dict[str, Any]]:
    """Learn this screen's look from one labelled screenshot.

    ``truth_lines`` are the canonical texts of the log entries (names as colour words, oldest first;
    at least the fully visible ones).  Stores in ``profile`` (a new :class:`UiProfile` when None):
    the panel region, the measured name colours (``ocr["name_colours"]``) and an adaptation of the
    glyph classifier to this screen's font (``ocr["glyph_adapt"]``: a sidecar ``.npz`` next to the
    profile file when it has a path, else inline).  Returns ``(profile, report)``; the report
    counts the aligned entries and taught glyphs and the entries read right before / after.
    """
    from . import logocr_teach as TE
    prof = profile if profile is not None else UiProfile()
    arr = LL.as_rgb(img)
    H, W = arr.shape[:2]
    bx = tuple(int(round(float(v))) for v in box) if box is not None else find_log_panel(arr, prof)
    report: Dict[str, Any] = {"box": bx}
    if not bx:
        report["error"] = "no log panel found"
        return prof, report
    prof.set_region("log", (bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H))
    prof.screen = (W, H)
    truth = [str(t) for t in truth_lines if str(t).strip()]
    pal = _palette(players, prof)
    lay = LL.analyse_panel(arr, bx, palette=pal)
    base = G.load_classifier()
    entries = [e for e in lay.entries if not e.partial]
    decoded = _decode_entries(lay, entries, base) if entries else {}
    texts = [decoded[id(e)][0] for e in entries]
    al = TE.align_truth(texts, truth)
    sig = lambda t: event_signature(L.parse_log_line(t))  # noqa: E731
    before = sum(1 for e, ti in zip(entries, al) if ti is not None and sig(texts[entries.index(e)]) == sig(truth[ti]))
    name_rgbs: Dict[str, List[Tuple[float, ...]]] = {}
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    used = 0
    rng = np.random.default_rng(0)
    for e, ti in zip(entries, al):
        if ti is None:
            continue
        ms = TE.match_structure(e, truth[ti])
        if ms is None:
            continue
        used += 1
        runs, names = ms
        for nt, colour in names:
            rgb = getattr(nt, "rgb", None)
            if rgb is not None and colour != "white":
                name_rgbs.setdefault(colour, []).append(tuple(float(v) for v in rgb))
        band_of = {id(t): b for b in e.bands for t in b.items}
        for run, words in runs:
            band = band_of.get(id(run))
            if band is None or not words:
                continue
            ri = G.run_ink(lay, band, run)
            bounds, gaps = G.cut_candidates(ri.A, ri.xh)
            if len(bounds) < 2:
                continue
            X, pairs = G.segment_features(ri, bounds, gaps)
            if X.shape[0] == 0:
                continue
            Xu = X * G.STORE_SCALE
            lp = base.log_proba(Xu)
            path = TE.forced_alignment(lp, pairs, bounds, gaps, ri.xh, words)
            if path is None:
                continue
            on = {n for n, _ in path}
            spans = [(bounds[pairs[n][0]], bounds[pairs[n][1]]) for n, _ in path]
            for n, c in path:
                Xs.append(Xu[n])
                ys.append(c)
            # garbage: other segments overlapping the path's glyphs
            cand = [n for n, (i, j) in enumerate(pairs) if n not in on
                    and any(min(bounds[j], b1) - max(bounds[i], a1) > 0 for a1, b1 in spans)]
            if cand:
                pick = rng.choice(len(cand), size=min(len(cand), 2 * len(path)), replace=False)
                for k in pick:
                    Xs.append(Xu[cand[int(k)]])
                    ys.append(G.GARBAGE)
    report.update(entries=len(entries), aligned=int(sum(1 for a in al if a is not None)), taught_entries=used,
                  read_right_before=int(before))
    ocr = dict(prof.ocr) if isinstance(prof.ocr, dict) else {}
    if name_rgbs:
        ocr["name_colours"] = {c: [list(np.mean(np.array(v), axis=0).round(1))] for c, v in name_rgbs.items()}
    nchar = int(sum(1 for y in ys if y != G.GARBAGE))
    report["glyphs"] = nchar
    if nchar >= 20:
        Wn, bn, st = TE.fit_adaptation(base, np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64))
        report.update({k: round(v, 4) for k, v in st.items()})
        fname = f"logocr_adapt_{hashlib.blake2b(Wn.tobytes(), digest_size=6).hexdigest()}.npz"
        side = prof.sidecar_path(fname)
        if side:
            np.savez_compressed(side, W=Wn, b=bn)
            ocr["glyph_adapt"] = {"file": fname, "glyphs": nchar}
            ocr["exemplars"] = fname
        else:
            ocr["glyph_adapt"] = {"b64": TE.encode_arrays(W=Wn, b=bn), "glyphs": nchar}
    ocr["taught_at"] = time.time()
    ocr["reader"] = _READER_VERSION
    prof.ocr = ocr
    # how the taught profile reads this screen
    after_res = read_log_panel(arr, box=bx, profile=prof, players=players)
    after_texts = [ln.text for ln in after_res.lines if not ln.partial]
    al2 = TE.align_truth(after_texts, truth)
    report["read_right_after"] = int(sum(1 for t, ti in zip(after_texts, al2) if ti is not None
                                         and sig(t) == sig(truth[ti])))
    return prof, report
