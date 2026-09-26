"""Live local screen reader: board + player cards + game log, frame after frame (no API).

:class:`LiveSession` turns a stream of screenshots of a Colonist.io game into a stable picture
of the game, for the ``watch`` advisor (:mod:`catanbot.cli`).  A real screen is noisy in time:
popups and trade windows cover the board for a few frames, pieces flicker in and out of the
detector, the newest log entry is read while it slides in, the log scrolls, the user looks at
another window.  The pipeline of one :meth:`LiveSession.step` is

1. **change detection** - a downsampled, quantised thumbnail of the frame is split into the
   log region and the rest of the screen; a region whose thumbnail matches the one last
   processed (hash, then a small tolerance) is not read again.  An unchanged frame costs a
   few milliseconds.
2. **game log** (only when the log region changed, or an entry is still unconfirmed) - the
   local log OCR (:mod:`catanbot.vision.logocr`, imported lazily; any reader with its
   ``read_log_panel`` contract can be injected) with a persistent ``LineCache``, then
   :class:`LogStream`: the frame's confident, non-partial entries are aligned with the
   confirmed tail, an overlapping entry keeps its confirmed text (it is never re-read
   differently), a new bottom entry is confirmed when it reads the same in two consecutive
   frames (or with confidence >= 0.9), a frame without a panel (a popup) changes nothing and a
   gap (the log scrolled past while not watched) is accepted once two frames agree on it and
   reported.  Every confirmed entry is emitted exactly once, in log order.
3. **board** (only when the rest of the screen changed) - the board / chrome parser
   (:func:`catanbot.vision.colonist.parse_image`, told the log box so the panel stays out of
   the board analysis), then :class:`BoardMemory`: the static board (tiles, numbers, ports)
   is locked once two consecutive confident parses agree and never re-read (a popup over the
   board cannot change it), pieces are monotonic (added once seen in 2 of the last 3 parses,
   settlement -> city upgrades, never removed because a frame missed them) and a new game
   (the board or the pieces replaced for 3 frames) resets everything.
4. **offers** - a newly confirmed ``offer`` / ``counter`` entry of an opponent that is still
   open is evaluated at once with :func:`evaluate_offer` (the trade rules, the accept /
   reject / counter search of :mod:`catanbot.counteroffers` and, when the card counter runs,
   the counted opponents' hands).
5. **card counting** - the :class:`~catanbot.colonist_log.ColonistLogTracker` of the
   ``--session`` file is fed only the stream's confirmed window (stable texts: the tracker
   would double count an entry whose reading changed) plus the latest state and bank, once
   the log has settled, and the session is saved.
6. **recording** (``record_dir``) - changed frames as ``frames/NNNNN.png`` and JSON lines of
   the confirmed entries (``events.jsonl``), the board parses (``parses.jsonl``), the offer
   verdicts (``offers.jsonl``) and the raw log reads (``ocr.jsonl``): a dataset of the user's
   own games for teaching the readers later.

Warnings are reported once per session (numbers in them are ignored when comparing), so an
hour of watching does not repeat the same line every frame.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:  # optional: the numpy fallback does the same area downsampling, only slower
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is installed in the repo's environment
    cv2 = None

from .. import board as B
from ..state import GameState, PHASE_GAME_OVER, PHASE_MAIN, PHASE_TRADE_RESPONSE, TradeOffer
from .profile import UiProfile

__all__ = [
    "LOGOCR_MODULE",
    "load_logocr",
    "logocr_missing_message",
    "LogEntry",
    "LogStream",
    "BoardMemory",
    "BoardUpdate",
    "OfferVerdict",
    "evaluate_offer",
    "LiveUpdate",
    "LiveSession",
    "frame_thumbnail",
    "compact_card_count",
    "name_colours",
    "layout_for",
    "confident_texts",
]

PixelBox = Tuple[int, int, int, int]
RGB = Tuple[int, int, int]

#: The local game-log OCR module (imported lazily: the live loop runs without it, board only).
LOGOCR_MODULE = "catanbot.vision.logocr"


def load_logocr() -> Any:
    """The log OCR module, or ``None`` when it is not installed (or fails to import)."""
    try:
        return importlib.import_module(LOGOCR_MODULE)
    except ImportError:
        return None


def logocr_missing_message() -> str:
    return (f"the local game-log reader ({LOGOCR_MODULE}) is not available in this installation, so the game "
            "log is not read (no card counting or offer verdicts from the log; the board is still read)")


def _norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).rstrip(" .!")


def _warn_key(msg: str) -> str:
    """Warnings are compared without their numbers (``noise level 2.3`` vs ``2.4`` is one warning)."""
    return re.sub(r"\d+(?:\.\d+)?", "#", _norm_text(msg))


def _hhmmss(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


# ---------------------------------------------------------------------------
# game log: confirmed, append-only entries
# ---------------------------------------------------------------------------
@dataclass
class LogEntry:
    """One confirmed log entry.  ``text`` never changes once confirmed; ``countable`` entries are
    fed to the card counter (an entry that stayed unreadable while later ones were read is kept
    in order, but not counted); ``gap_before`` marks the first entry after a gap."""

    index: int
    text: str
    kind: str
    event: Any = None                 # colonist_log.LogEvent (None when the text parses to nothing)
    confidence: float = 0.0
    key: str = ""
    time: float = 0.0
    frame: int = -1
    countable: bool = True
    gap_before: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "time": round(self.time, 3), "frame": self.frame, "text": self.text,
                "kind": self.kind, "confidence": round(float(self.confidence), 4), "countable": self.countable,
                "gap_before": self.gap_before, "key": self.key}


@dataclass
class _Seen:
    """One line of a log read, normalised."""

    text: str
    norm: str
    key: str
    conf: float
    confident: bool
    event: Any


def _parse_line(text: str) -> Any:
    from ..colonist_log import parse_log_line
    try:
        return parse_log_line(text)
    except Exception:  # pragma: no cover - the phrase table never raises on text
        return None


def _seen_lines(lines: Optional[Sequence[Any]], min_conf: float) -> List[_Seen]:
    """The reader's lines (oldest first) without the partial ones (cut at the panel's top edge)."""
    out: List[_Seen] = []
    for ln in lines or []:
        if getattr(ln, "partial", False):
            continue
        text = str(getattr(ln, "text", "") or "").strip()
        conf = float(getattr(ln, "confidence", 0.0) or 0.0)
        key = str(getattr(ln, "key", "") or "")
        norm = _norm_text(text)
        ev = getattr(ln, "event", None)
        if ev is None and text:
            ev = _parse_line(text)
        out.append(_Seen(text, norm, key, conf, bool(norm) and conf >= min_conf, ev))
    return out


_CLOSE_RE = re.compile(r"^(?P<who>\S+(?:\s\S+){0,2}?)\s+(?P<verb>accepted|rejected|declined|cancell?ed|canceled|"
                       r"withdrew|withdrawn)\b", re.IGNORECASE)


class LogStream:
    """Successive OCR windows of the log panel -> a stable, append-only list of confirmed entries.

    :meth:`update` takes one read's lines (``text``, ``confidence``, ``key``, ``partial``;
    ``None`` = no panel on screen) and returns the entries confirmed by it, oldest first:

    * the frame's non-partial lines are aligned with the confirmed entries (a line matches an
      entry by its pixel ``key`` or, when confident, by its text; low-confidence lines are
      neutral) - the best offset wins when it has enough matches and few mismatches.  The
      confirmed entries are indexed by text and key, so a panel scrolled up to any older part of
      the log is recognised ("inside": nothing new);
    * lines in the overlap keep their confirmed text (a different re-read is ignored);
    * a line after the overlap is confirmed when its confidence is >= ``confirm_conf`` or it
      read the same (text) in the previous frame at the same log position; confirmation stops
      at the first line that is not ready (the order is kept).  A line that stays unreadable
      for ``stall_frames`` frames while later lines are read is confirmed as not countable;
    * a frame sharing nothing with the confirmed entries is a gap candidate: it is accepted when
      the next frame agrees with it and either shows new entries at the bottom (the live end
      of the log) or ``gap_seconds`` passed since the last aligned read (a panel scrolled far up
      is not taken for a gap at once); the first entry after it has ``gap_before`` and a
      warning is reported.

    The stream also follows offers: an ``offer`` / ``counter`` stays open until its proposer
    trades, cancels or withdraws, someone accepts, or the turn moves on (a roll / turn line) -
    :meth:`is_open`.
    """

    def __init__(self, min_conf: float = 0.6, confirm_conf: float = 0.9, stall_frames: int = 3,
                 gap_seconds: float = 20.0, clock: Callable[[], float] = time.time):
        self.min_conf = float(min_conf)
        self.confirm_conf = float(confirm_conf)
        self.stall_frames = int(stall_frames)
        self.gap_seconds = float(gap_seconds)
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.entries: List[LogEntry] = []
        self._refs: List[Tuple[str, str]] = []  # (normalised text, key) per entry
        self._by_text: Dict[str, List[int]] = {}
        self._by_key: Dict[str, List[int]] = {}
        self.pending: List[_Seen] = []          # read after the confirmed end, not confirmed yet
        self._stall = 0
        self._stall_pos = -1
        self._gap_prev: Optional[List[_Seen]] = None
        self._last_aligned: Optional[float] = None
        self.gaps = 0
        self.no_panel = 0
        self.warnings: List[str] = []           # of the last update
        self.open_offers: Dict[int, LogEntry] = {}
        self.last_gap_index: int = -1

    # --- alignment -----------------------------------------------------------------------
    @staticmethod
    def _score(refs: Sequence[Tuple[str, str]], items: Sequence[_Seen], s: int) -> Tuple[int, int]:
        """Matches / mismatches of the frame placed at offset ``s``: a line matches an entry by its
        key, or (when confident) by its text; a low-confidence line with another key is neutral."""
        m = x = 0
        n = len(refs)
        for i in range(max(0, -s), min(len(items), n - s)):
            it = items[i]
            norm, key = refs[s + i]
            if it.key and key and it.key == key:
                m += 1
            elif it.confident:
                if it.norm == norm:
                    m += 1
                else:
                    x += 1
        return m, x

    @staticmethod
    def _valid(m: int, x: int) -> bool:
        return m >= 1 and 3 * x <= m

    def _best_offset(self, items: Sequence[_Seen], recent: Optional[int] = 400) -> Optional[Tuple[int, int, int]]:
        """``(offset, matches, mismatches)`` of the frame against the confirmed entries (frame line
        ``i`` is entry ``offset + i``), or ``None`` when no offset is acceptable.  Offsets come from
        the entries sharing a key / text with a frame line (the ``recent`` last entries first, the
        whole log when none of them fits)."""
        n = len(self.entries)
        lo = max(0, n - recent) if recent else 0
        cands = set()
        for i, it in enumerate(items):
            if it.key:
                for p in reversed(self._by_key.get(it.key, ())):
                    if p < lo:
                        break
                    cands.add(p - i)
            if it.confident:
                for p in reversed(self._by_text.get(it.norm, ())):
                    if p < lo:
                        break
                    cands.add(p - i)
        best = None
        for s in cands:
            m, x = self._score(self._refs, items, s)
            if not self._valid(m, x):
                continue
            score = (m - 2 * x + (1.5 if s + len(items) >= n else 0.0), m, s)
            if best is None or score > best[0]:
                best = (score, s, m, x)
        if best is None and lo > 0:
            return self._best_offset(items, None)
        return None if best is None else (best[1], best[2], best[3])

    # --- confirmation ----------------------------------------------------------------------
    def _append(self, it: _Seen, now: float, frame: int, countable: bool = True, gap: bool = False) -> LogEntry:
        ev = it.event if countable else None
        kind = getattr(ev, "kind", None) or ("unknown" if countable else "unreadable")
        e = LogEntry(index=len(self.entries), text=it.text, kind=kind, event=ev, confidence=it.conf, key=it.key,
                     time=now, frame=frame, countable=countable, gap_before=gap)
        self.entries.append(e)
        self._refs.append((it.norm, it.key))
        self._by_text.setdefault(it.norm, []).append(e.index)
        if it.key:
            self._by_key.setdefault(it.key, []).append(e.index)
        if gap:
            self.last_gap_index = e.index
        self._track_offer(e)
        return e

    def _confirm_new(self, new: List[_Seen], now: float, frame: int, gap: bool = False) -> List[LogEntry]:
        """Confirm the lines after the confirmed end, in order; the rest waits in :attr:`pending`
        (compared at the same log position with the next frame's read)."""
        out: List[LogEntry] = []
        prev = self.pending
        k = 0
        while k < len(new):
            it = new[k]
            same = k < len(prev) and it.confident and prev[k].confident and prev[k].norm == it.norm
            if it.confident and (it.conf >= self.confirm_conf or same):
                out.append(self._append(it, now, frame, gap=gap and not out))
                k += 1
                continue
            # not ready: wait - unless it stays unreadable while the entries after it are read
            pos = len(self.entries)
            if any(x.confident for x in new[k + 1:]):
                self._stall = self._stall + 1 if self._stall_pos == pos else 1
                self._stall_pos = pos
                if self._stall >= self.stall_frames:
                    self.warnings.append(f"a log entry stayed unreadable (last read '{it.text}', confidence "
                                         f"{it.conf:.2f}) while the ones after it were read; kept in order, not counted")
                    out.append(self._append(it, now, frame, countable=False, gap=gap and not out))
                    self._stall, self._stall_pos = 0, -1
                    k += 1
                    continue
            else:
                self._stall, self._stall_pos = 0, -1
            break
        self.pending = list(new[k:])
        return out

    def update(self, lines: Optional[Sequence[Any]], now: Optional[float] = None, frame: int = -1,
               allow_gap: bool = True) -> List[LogEntry]:
        """Feed one read of the panel (``None``: no panel on screen); returns the newly confirmed
        entries, oldest first (each entry is returned exactly once over the stream's life).
        ``allow_gap=False`` holds a gap (the screen may be showing another game)."""
        now = self.clock() if now is None else float(now)
        self.warnings = []
        if lines is None:
            self.no_panel += 1
            return []
        items = _seen_lines(lines, self.min_conf)
        if not items:
            self.no_panel += 1
            return []
        n = len(self.entries)
        if n == 0 and self._gap_prev is None:
            # nothing confirmed yet: the waiting lines are matched to this frame by content (the
            # panel may have scrolled between the two reads)
            self._last_aligned = now
            s = self._align_frames(self.pending, items, 1) if self.pending else None
            self.pending = self._shifted(self.pending, s, len(items)) if s is not None else []
            return self._confirm_new(items, now, frame)
        best = self._best_offset(items) if n else None
        if best is not None:
            s, _, _ = best
            self._gap_prev = None
            self._last_aligned = now
            if s + len(items) <= n:
                return []                       # inside the confirmed log (scrolled up / nothing new)
            return self._confirm_new(list(items[n - s:]), now, frame)
        return self._gap(items, now, frame, allow_gap)

    def _gap(self, items: List[_Seen], now: float, frame: int, allow: bool = True) -> List[LogEntry]:
        """A frame sharing nothing with the confirmed entries."""
        if sum(1 for it in items if it.confident) < 2:
            return []
        prev = self._gap_prev
        self._gap_prev = items
        if prev is None or not allow:
            return []
        s = self._align_frames(prev, items, 2)
        if s is None:
            return []
        extended = s + len(items) > len(prev)
        waited = self._last_aligned is None or now - self._last_aligned >= self.gap_seconds
        if not (extended or waited):
            return []
        # accepted: lines read the same in both frames are confirmed, the rest waits as usual
        self.pending = self._shifted(prev, s, len(items))
        self.gaps += 1
        before = len(self.entries)
        out = self._confirm_new(items, now, frame, gap=True)
        if out:
            self._gap_prev = None
            self._last_aligned = now
            self.warnings.append(f"the game log jumped: the entries on screen do not continue the {before} read so far "
                                 "(it scrolled past while the screen was not watched?); entries in between are missing "
                                 "and the card count resynchronises from the hand sizes")
        else:
            self.gaps -= 1
        return out

    def _align_frames(self, prev: Sequence[_Seen], items: Sequence[_Seen], min_matches: int) -> Optional[int]:
        """Offset of ``items`` against the previous read ``prev`` (line ``i`` is ``prev[s + i]``)."""
        refs = [(it.norm, it.key) for it in prev]
        best = None
        for s in range(-len(items) + 1, len(prev)):
            m, x = self._score(refs, items, s)
            if m >= min_matches and 3 * x <= m and (best is None or (m - 2 * x, s) > best[0]):
                best = ((m - 2 * x, s), s)
        return None if best is None else best[1]

    @staticmethod
    def _shifted(prev: Sequence[_Seen], s: Optional[int], n: int) -> List[_Seen]:
        """``prev`` re-indexed to a frame at offset ``s`` (missing positions: a blank line)."""
        blank = _Seen("", "", "", 0.0, False, None)
        if s is None:
            return []
        return [prev[s + i] if 0 <= s + i < len(prev) else blank for i in range(n)]

    # --- offers ------------------------------------------------------------------------------
    def _who(self, text: Optional[str]) -> str:
        return _norm_text(text)

    def _track_offer(self, e: LogEntry) -> None:
        ev = e.event
        kind = e.kind
        if kind in ("offer", "counter") and ev is not None:
            who = self._who(ev.player)
            for idx in [i for i, o in self.open_offers.items() if self._who(o.event.player) == who]:
                del self.open_offers[idx]            # a new offer replaces the proposer's older one
            self.open_offers[e.index] = e
        elif kind in ("roll", "turn"):
            self.open_offers.clear()
        elif kind == "player_trade" and ev is not None:
            parties = {self._who(ev.player), self._who(ev.other)}
            for idx in [i for i, o in self.open_offers.items() if self._who(o.event.player) in parties]:
                del self.open_offers[idx]
        elif kind == "ignored":
            m = _CLOSE_RE.match(e.text.strip())
            if m:
                verb = m.group("verb").lower()
                who = self._who(m.group("who"))
                if verb == "accepted":
                    self.open_offers.clear()
                elif verb.startswith(("cancel", "withdr")):
                    for idx in [i for i, o in self.open_offers.items() if self._who(o.event.player) == who]:
                        del self.open_offers[idx]

    def is_open(self, entry: LogEntry) -> bool:
        return entry.index in self.open_offers

    # --- views -------------------------------------------------------------------------------
    def texts(self, start: int = 0) -> List[str]:
        return [e.text for e in self.entries[start:]]

    def window(self, start: int = 0) -> List[Any]:
        """The confirmed, countable events from entry ``start`` on (the card counter's window)."""
        return [e.event for e in self.entries[start:] if e.countable and e.event is not None]


# ---------------------------------------------------------------------------
# board memory
# ---------------------------------------------------------------------------
@dataclass
class BoardUpdate:
    parsed: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    new_game: bool = False
    locked: bool = False


class BoardMemory:
    """What stays true between frames of one game.

    * The static board - tiles with their numbers, and the ports - is locked once two
      consecutive confident parses (``confidence["hexes"]`` and ``["numbers"]`` >= ``min_conf``;
      ports: ``["ports"]`` >= ``port_conf``) agree; afterwards every frame gets the locked values
      and a frame that disagrees is reported once (a popup or a trade window over the board).
    * Pieces are monotonic: a settlement / city / road is added once seen in ``votes`` of the
      last ``window`` parses, a settlement becomes a city the same way, and nothing disappears
      because one frame missed it.  The merged parse shows the remembered pieces plus the pieces
      this frame shows on free spots (provisionally: a new road is visible at once, but only the
      memory keeps it when a later frame misses it).
    * The player list (colours, in panel order) is locked the same way; a player missing from a
      frame keeps their last values, and a value a frame could not read (``None``) keeps the last
      one read.
    * The robber, dice, current player, bank, the panel numbers and our hand come from the
      latest frame.
    * A new game - at least ``new_board_diff`` of the 19 locked tiles different in
      ``new_game_frames`` consecutive confident frames (every number token read) that agree with
      each other, or nearly all pieces gone for as many frames while the board is clearly the
      locked one - resets the memory (``BoardUpdate.new_game``).  A popup over part of the board
      changes far fewer tiles and hides tokens, so it never counts.  :attr:`suspect` is set while
      such frames are being seen (the live session holds log gaps and the card count meanwhile).
    """

    def __init__(self, min_conf: float = 0.6, port_conf: float = 0.5, votes: int = 2, window: int = 3,
                 new_game_frames: int = 3, new_board_diff: int = 15):
        self.min_conf = float(min_conf)
        self.port_conf = float(port_conf)
        self.votes = int(votes)
        self.window = int(window)
        self.new_game_frames = int(new_game_frames)
        self.new_board_diff = int(new_board_diff)
        self.games = 0
        self.reset()

    def reset(self) -> None:
        self.hexes: Optional[Tuple[Tuple[Any, Any], ...]] = None
        self.ports: Optional[List[Dict[str, Any]]] = None
        self._prev_hexes: Optional[Tuple[Tuple[Any, Any], ...]] = None
        self._prev_ports: Optional[Tuple[Tuple[int, str], ...]] = None
        self.buildings: Dict[int, Tuple[str, bool]] = {}
        self.roads: Dict[int, str] = {}
        self._history: Deque[Tuple[Dict[int, Tuple[str, bool]], Dict[int, str]]] = deque(maxlen=self.window)
        self.colours: Optional[List[str]] = None
        self._prev_colours: Optional[List[str]] = None
        self._last_player: Dict[str, Dict[str, Any]] = {}
        self._disagree: List[Tuple[Tuple[Any, Any], ...]] = []
        self._vanish = 0
        self.suspect = False          # the last frame looked like another game (not confirmed yet)
        self._warned: set = set()
        self.last: Optional[Tuple[Dict[str, Any], Dict[str, float]]] = None
        self.frames = 0

    @property
    def locked(self) -> bool:
        return self.hexes is not None

    @property
    def pending(self) -> bool:
        """Something seen in the last parse is not confirmed yet (a piece, the board, the players)."""
        if self.last is None:
            return False
        parsed, _ = self.last
        if self.colours is None or (self.hexes is None and self._prev_hexes is not None):
            return True
        b, r = self._pieces(parsed)
        for v, (col, city) in b.items():
            cur = self.buildings.get(v)
            if cur is None or (cur[0] == col and city and not cur[1]):
                return True
        return any(e not in self.roads for e in r)

    @staticmethod
    def _hex_sig(parsed: Dict[str, Any]) -> Optional[Tuple[Tuple[Any, Any], ...]]:
        hexes = parsed.get("hexes")
        if not isinstance(hexes, list) or len(hexes) != B.NUM_HEXES:
            return None
        out = []
        for h in hexes:
            if not isinstance(h, dict):
                return None
            out.append((str(h.get("resource") or "").lower(), h.get("number")))
        return tuple(out)

    @staticmethod
    def _port_sig(parsed: Dict[str, Any]) -> Optional[Tuple[Tuple[int, str], ...]]:
        ports = parsed.get("ports")
        if not isinstance(ports, list) or not ports:
            return None
        try:
            return tuple(sorted((int(p["edge"]), str(p["type"])) for p in ports if isinstance(p, dict)))
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _pieces(parsed: Dict[str, Any]) -> Tuple[Dict[int, Tuple[str, bool]], Dict[int, str]]:
        b: Dict[int, Tuple[str, bool]] = {}
        r: Dict[int, str] = {}
        for p in parsed.get("players") or []:
            if not isinstance(p, dict):
                continue
            col = str(p.get("color") or "").lower()
            for v in p.get("settlements") or []:
                b.setdefault(int(v), (col, False))
            for v in p.get("cities") or []:
                b[int(v)] = (col, True)
            for e in p.get("roads") or []:
                r.setdefault(int(e), col)
        return b, r

    def _warn_once(self, out: List[str], tag: str, msg: str) -> None:
        if tag not in self._warned:
            self._warned.add(tag)
            out.append(msg)

    def update(self, parsed: Dict[str, Any], confidence: Optional[Dict[str, float]] = None,
               fresh: bool = True) -> BoardUpdate:
        """Merge one frame's parse; returns the merged parse (a new dict; ``parsed`` is not modified).
        ``fresh=False`` (:meth:`revote`) repeats the last parse: it votes, but never starts a new game."""
        conf = dict(confidence or {})
        warns: List[str] = []
        new_game = False
        hsig = self._hex_sig(parsed)
        confident = (hsig is not None and conf.get("hexes", 0.0) >= self.min_conf
                     and conf.get("numbers", 0.0) >= self.min_conf)
        b, r = self._pieces(parsed)
        # --- new game? ----------------------------------------------------------------------
        # A new map differs from the old one in nearly every tile (two random standard boards share
        # well under one tile on average), a popup or trade window over part of the board in far
        # fewer - and it hides number tokens, so a frame only counts with every token read.
        if self.hexes is not None and confident and fresh:
            diff = sum(1 for a, c in zip(hsig, self.hexes) if a != c)
            tokens_ok = conf.get("numbers_min", 1.0) >= self.min_conf
            if diff >= self.new_board_diff and tokens_ok:
                self._disagree.append(hsig)
                self._disagree = self._disagree[-self.new_game_frames:]
                if len(self._disagree) >= self.new_game_frames and len(set(self._disagree)) == 1:
                    new_game = True
            else:
                self._disagree = []
            if diff and not new_game:
                self._warn_once(warns, "board-differs",
                                f"a frame shows {diff} tile(s) different from the locked board (a popup or trade window "
                                "over the board?); the locked board is kept")
            # the same map with the pieces gone (a rematch): the board clearly visible, nearly no piece
            known = len(self.buildings) + len(self.roads)
            seen = sum(1 for v in b if v in self.buildings) + sum(1 for e in r if e in self.roads)
            if diff <= 2 and tokens_ok and known >= 8 and seen * 10 <= known:
                self._vanish += 1
                if self._vanish >= self.new_game_frames:
                    new_game = True
            else:
                self._vanish = 0
        self.suspect = bool(self._disagree) or self._vanish > 0
        if new_game:
            self.reset()
            self.games += 1
            warns.append("a new game started (the board / pieces changed for several frames): the board, the log and "
                         "the card count start over")
            self._history.clear()
        self.frames += 1
        self.last = (parsed, conf)
        # --- lock the static board ------------------------------------------------------------
        if self.hexes is None:
            if confident and self._prev_hexes == hsig:
                self.hexes = hsig
            self._prev_hexes = hsig if confident else None
        psig = self._port_sig(parsed)
        if self.ports is None:
            if psig is not None and conf.get("ports", 0.0) >= self.port_conf and self._prev_ports == psig:
                self.ports = [dict(p) for p in parsed["ports"] if isinstance(p, dict)]
            self._prev_ports = psig if (psig is not None and conf.get("ports", 0.0) >= self.port_conf) else None
        elif psig is not None and psig != self._port_sig({"ports": self.ports}):
            self._warn_once(warns, "ports-differ", "a frame shows other ports than the locked board; the locked ports "
                                                   "are kept")
        # --- pieces: 2 of the last 3 parses ---------------------------------------------------
        self._history.append((b, r))
        for v, (col, _) in b.items():
            n_any = sum(1 for hb, _ in self._history if hb.get(v, ("",))[0] == col)
            n_city = sum(1 for hb, _ in self._history if hb.get(v) == (col, True))
            cur = self.buildings.get(v)
            if cur is None:
                if n_any >= self.votes:
                    self.buildings[v] = (col, n_city >= self.votes)
            elif cur[0] == col:
                if not cur[1] and n_city >= self.votes:
                    self.buildings[v] = (col, True)
            elif n_any >= self.votes:
                self._warn_once(warns, f"vertex-{v}", f"vertex {v}: a {col} building is seen where {cur[0]}'s was "
                                                      "recorded; the first one is kept")
        for e, col in r.items():
            if e not in self.roads and sum(1 for _, hr in self._history if hr.get(e) == col) >= self.votes:
                self.roads[e] = col
        # --- players ------------------------------------------------------------------------------
        frame_players = [p for p in parsed.get("players") or [] if isinstance(p, dict)]
        cols = [str(p.get("color") or "").lower() for p in frame_players]
        by_col = {c: p for c, p in zip(cols, frame_players)}
        if self.colours is None:
            if cols and cols == self._prev_colours:
                self.colours = list(cols)
            self._prev_colours = cols if cols else None
        elif cols != self.colours:
            extra = [c for c in cols if c not in self.colours]
            missing = [c for c in self.colours if c not in cols]
            if extra or missing:
                self._warn_once(warns, "players-differ",
                                "a frame shows other players (" + ", ".join(cols) + ") than the game ("
                                + ", ".join(self.colours) + "); the missing keep their last values")
        order = self.colours if self.colours is not None else cols
        # shown: the confirmed pieces plus what this frame shows on free spots (provisional: a piece
        # seen once is on screen now, but only the memory keeps it when a later frame misses it)
        shown_b = dict(self.buildings)
        for v, (col, city) in b.items():
            cur = shown_b.get(v)
            if cur is None or (cur[0] == col and city and not cur[1]):
                shown_b[v] = (col, city)
        shown_r = dict(self.roads)
        for e, col in r.items():
            shown_r.setdefault(e, col)
        players: List[Dict[str, Any]] = []
        for c in order:
            base = dict(self._last_player.get(c) or {"color": c, "name": c, "vp": 0, "cards": 0, "dev_cards": 0,
                                                    "knights": 0, "longest_road": False, "largest_army": False})
            cur = by_col.get(c)
            if cur is not None:
                for k, v in cur.items():
                    if k in ("settlements", "cities", "roads"):
                        continue
                    if v is not None:
                        base[k] = v
                if "resources" not in cur:
                    base.pop("resources", None)   # the hand is only known when this frame shows it
            base["color"] = c
            self._last_player[c] = {k: v for k, v in base.items() if k not in ("settlements", "cities", "roads")}
            base["settlements"] = sorted(v for v, (col, city) in shown_b.items() if col == c and not city)
            base["cities"] = sorted(v for v, (col, city) in shown_b.items() if col == c and city)
            base["roads"] = sorted(e for e, col in shown_r.items() if col == c)
            players.append(base)
        merged = {k: v for k, v in parsed.items() if k not in ("hexes", "ports", "players")}
        if self.hexes is not None:
            merged["hexes"] = [{"resource": res, "number": num} for res, num in self.hexes]
        else:
            merged["hexes"] = parsed.get("hexes")
        if self.ports is not None:
            merged["ports"] = [dict(p) for p in self.ports]
        elif parsed.get("ports") is not None:
            merged["ports"] = parsed.get("ports")
        merged["players"] = players
        return BoardUpdate(parsed=merged, warnings=warns, new_game=new_game, locked=self.hexes is not None)

    def revote(self) -> Optional[BoardUpdate]:
        """The frame did not change: its (last) parse counts again - a piece still on an unchanged
        screen is confirmed without parsing the same pixels twice."""
        if self.last is None:
            return None
        parsed, conf = self.last
        return self.update(parsed, conf, fresh=False)


# ---------------------------------------------------------------------------
# offers
# ---------------------------------------------------------------------------
@dataclass
class OfferVerdict:
    """The advice on one incoming offer.  ``verdict`` is ``accept``, ``reject`` or ``counter: give
    X for Y``; ``accept`` / ``rule_reason`` are the trade rules' answer (:func:`catanbot.trading.should_accept`),
    ``best`` the accept / reject / counter search's (after the proposer's turn) and ``lines`` its
    report (best answer first)."""

    verdict: str
    reason: str
    lines: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    accept: bool = False
    rule_reason: str = ""
    best: str = ""
    proposer: int = -1
    proposer_colour: str = ""
    give: List[int] = field(default_factory=lambda: [0] * 5)
    get: List[int] = field(default_factory=lambda: [0] * 5)
    kind: str = "offer"
    text: str = ""
    entry: int = -1
    time: float = 0.0

    def summary(self) -> str:
        """``you get 1 wood, you give 1 ore -> ACCEPT: reason``."""
        from .. import actions as A
        return (f"you get {A._counts_str(self.give)}, you give {A._counts_str(self.get)} -> "
                f"{self.verdict.upper() if not self.verdict.startswith('counter') else 'COUNTER' + self.verdict[7:]}"
                f": {self.reason}")

    def to_dict(self) -> Dict[str, Any]:
        return {"time": round(self.time, 3), "entry": self.entry, "text": self.text, "kind": self.kind,
                "proposer": self.proposer_colour, "give": list(self.give), "get": list(self.get),
                "verdict": self.verdict, "reason": self.reason, "accept": self.accept, "rule_reason": self.rule_reason,
                "best": self.best, "lines": list(self.lines), "warnings": list(self.warnings)}


#: A counter-offer becomes the verdict when the look-ahead values it this much above the rules' answer.
COUNTER_MARGIN = 0.01


def _short_action(a: Sequence[Any]) -> str:
    from .. import actions as A
    k = a[0]
    if k == A.ACCEPT_TRADE:
        return "accept"
    if k == A.REJECT_TRADE:
        return "reject"
    return f"counter: give {A._counts_str(a[1])} for {A._counts_str(a[2])}"


def evaluate_offer(state: GameState, me: int, proposer: int, give: Sequence[int], get: Sequence[int],
                   tracker: Any = None, evaluator: Any = None, model: Any = None, politics: Any = None,
                   seed: int = 0) -> OfferVerdict:
    """Accept / reject / counter an incoming offer: ``proposer`` gives us ``give`` and wants ``get``.

    The decision state is a copy of ``state`` in the trade-response phase with us as the
    responder.  With a card counter (``tracker``: a :class:`~catanbot.colonist_log.ColonistLogTracker`
    that has counted) the opponents' hands are one determinization of the count and the trade
    rules (:func:`catanbot.trading.should_accept`) compare values with ``evaluator``; without one
    the rules work on what is public.  The rules give the accept / reject verdict.  The
    accept / reject / counter look-ahead (:func:`catanbot.counteroffers.offer_response_report`,
    with the count as its belief) values every answer after the proposer's turn: its lines are
    the verdict's ``lines``, a counter-offer it values :data:`COUNTER_MARGIN` above the rules'
    answer becomes the verdict, and a disagreement on accept / reject is said in the reason.  A
    hand that cannot pay ``get`` is always a reject (with the warning of
    :func:`catanbot.cli.offer_affordability_warnings`).
    """
    from ..cli import offer_affordability_warnings
    from ..counteroffers import offer_response_report
    from ..opponent_model import OpponentModel
    from ..politics import PoliticalState
    from ..trading import should_accept
    give = [int(x) for x in give]
    get = [int(x) for x in get]
    pc = state.players[proposer].color if 0 <= proposer < state.num_players else str(proposer)
    base = OfferVerdict(verdict="reject", reason="", give=list(give), get=list(get), proposer=proposer,
                        proposer_colour=pc)
    if not (0 <= me < state.num_players) or not (0 <= proposer < state.num_players) or proposer == me:
        base.reason = "not an offer to us"
        return base
    if sum(give) <= 0 or sum(get) <= 0:
        base.reason = "one side of the offer is empty or unreadable"
        return base
    if state.phase == PHASE_GAME_OVER:
        base.reason = "the game is over"
        return base
    warns = offer_affordability_warnings(state, me, proposer, give, get)
    base.warnings = warns
    s = state.copy()
    offer = TradeOffer(proposer, list(give), list(get))
    s.pending_trade = offer
    s.current = proposer
    s.phase = PHASE_TRADE_RESPONSE
    s.trade_responder = me
    s.dice = s.dice or 8
    s.discard_queue = []
    if evaluator is None:
        from ..heuristic import HeuristicEvaluator
        evaluator = HeuristicEvaluator()
    model = model or OpponentModel(s)
    politics = politics or PoliticalState(s.num_players)
    rng = random.Random(seed)
    decision = s
    belief = None
    if tracker is not None and getattr(tracker, "counter", None) is not None:
        try:
            decision = tracker.determinize(tracker.public_view(s), rng)
            belief = tracker.counter
        except Exception as ex:   # a count that does not fit this state: judge on the public view
            base.warnings.append(f"card count not usable for this offer ({ex})")
            decision = s
    try:
        if decision is s:
            ok, why = should_accept(s, me, offer, model=model, politics=politics)
        else:
            ok, why = should_accept(decision, me, decision.pending_trade, evaluator=evaluator, model=model,
                                    politics=politics)
    except Exception as ex:  # pragma: no cover - defensive: an odd parsed state
        ok, why = False, f"trade rules failed ({ex})"
    base.accept, base.rule_reason = bool(ok), why
    rule = "accept" if ok else "reject"
    try:
        rep = offer_response_report(decision, me, evaluator, model=model, politics=politics, belief=belief,
                                    rng=random.Random(seed))
    except Exception as ex:  # pragma: no cover - defensive
        rep = {"options": [], "best": "", "lines": [f"(offer search unavailable: {ex})"]}
    base.lines = list(rep.get("lines") or [])
    base.best = str(rep.get("best") or "")
    values = {}
    for o in rep.get("options") or []:
        try:
            values[_short_action(o["action"])] = float(o["v_turn"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    if any(w.startswith("you cannot pay") for w in warns):
        base.verdict = "reject"
        base.reason = warns[0].split(" (your hand")[0]
        return base
    best = base.best
    base.verdict, base.reason = rule, why
    if best.startswith("counter") and best in values and rule in values and values[best] - values[rule] >= COUNTER_MARGIN:
        base.verdict = best
        base.reason = (f"a counter-offer is worth more after {pc}'s turn ({values[best]:.3f} vs {rule} "
                       f"{values[rule]:.3f}); the offer as made: {rule} - {why}")
    elif best and best != rule and best in values and rule in values:
        base.reason = f"{why} (the look-ahead over {pc}'s turn prefers {best}: {values[best]:.3f} vs {values[rule]:.3f})"
    return base


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------
def frame_thumbnail(arr: np.ndarray, factor: int) -> np.ndarray:
    """Area-averaged ``1/factor`` thumbnail (uint8) of an RGB array."""
    h, w = arr.shape[:2]
    tw, th = max(1, w // factor), max(1, h // factor)
    if factor <= 1:
        return np.ascontiguousarray(arr[:, :, :3])
    if cv2 is not None:
        return cv2.resize(np.ascontiguousarray(arr[:, :, :3]), (tw, th), interpolation=cv2.INTER_AREA)
    a = arr[:th * factor, :tw * factor, :3].reshape(th, factor, tw, factor, 3)
    return a.mean(axis=(1, 3)).astype(np.uint8)


def _thumb_hash(t: np.ndarray) -> str:
    return hashlib.blake2b((t >> 3).tobytes(), digest_size=16).hexdigest() + str(t.shape)


class _Region:
    """The last processed thumbnail of one screen region."""

    def __init__(self, tol: int):
        self.tol = int(tol)
        self.ref: Optional[np.ndarray] = None
        self.hash: Optional[str] = None

    def changed(self, t: np.ndarray) -> bool:
        if self.ref is None or self.ref.shape != t.shape:
            return True
        if _thumb_hash(t) == self.hash:
            return False
        return bool(np.abs(t.astype(np.int16) - self.ref).max() > self.tol)

    def accept(self, t: np.ndarray) -> None:
        self.ref = t
        self.hash = _thumb_hash(t)

    def clear(self) -> None:
        self.ref = None
        self.hash = None


# ---------------------------------------------------------------------------
# the live session
# ---------------------------------------------------------------------------
@dataclass
class LiveUpdate:
    """What one :meth:`LiveSession.step` found.  ``warnings`` holds only warnings not reported
    before in this session; ``card_count`` is the tracker's report when it was updated this
    step; ``parse_ms`` / ``ocr_ms`` are ``None`` when that part was not re-read."""

    time: float
    frame: int = 0
    changed_board: bool = False
    changed_log: bool = False
    state: Optional[GameState] = None
    parsed: Optional[Dict[str, Any]] = None
    me: Optional[int] = None
    new_entries: List[LogEntry] = field(default_factory=list)
    offers: List[OfferVerdict] = field(default_factory=list)
    card_count: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    new_game: bool = False
    parse_ms: Optional[float] = None
    ocr_ms: Optional[float] = None
    total_ms: float = 0.0
    skipped: bool = False
    parse_warnings: List[str] = field(default_factory=list)


def _default_parse(img: Any, me: Optional[str] = None, layout: Any = None) -> Any:
    from .colonist import parse_image
    return parse_image(img, me=me, layout=layout)


def _as_image(img: Any) -> Tuple[Image.Image, np.ndarray]:
    if isinstance(img, (str, os.PathLike)):
        with Image.open(img) as im:
            pil = im.convert("RGB")
    elif isinstance(img, np.ndarray):
        a = img
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        pil = Image.fromarray(np.ascontiguousarray(a[:, :, :3]).astype(np.uint8))
    elif isinstance(img, Image.Image):
        pil = img if img.mode == "RGB" else img.convert("RGB")
    else:
        raise TypeError(f"expected an image path, a PIL image or an RGB array, got {type(img).__name__}")
    return pil, np.asarray(pil)


def _sig(parsed: Optional[Dict[str, Any]], keys: Sequence[str]) -> str:
    if parsed is None:
        return ""
    return hashlib.md5(json.dumps({k: parsed.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def compact_card_count(report: Optional[Dict[str, Any]]) -> str:
    """One line of the tracker's report: ``blue 4 = 1 wood, 3 ore | orange 3 ~ 2 brick, 1 ore (60%)``."""
    if not report:
        return ""
    parts = []
    for col, p in (report.get("players") or {}).items():
        size = int(p.get("size", 0))
        if size == 0:
            parts.append(f"{col} 0")
            continue
        if p.get("exact"):
            cards = ", ".join(f"{n} {r}" for r, n in (p.get("certain") or {}).items() if n)
            parts.append(f"{col} {size} = {cards}")
        else:
            ml = ", ".join(f"{n} {r}" for r, n in (p.get("most_likely") or {}).items() if n)
            parts.append(f"{col} {size} ~ {ml} ({float(p.get('most_likely_p', 0.0)):.0%})")
    return " | ".join(parts)


class _Recorder:
    """``record_dir``: changed frames and JSON lines of what was read (the user's own dataset)."""

    def __init__(self, root: str):
        self.root = root
        self.frames_dir = os.path.join(root, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        nums = [int(m.group(1)) for f in os.listdir(self.frames_dir) for m in [re.match(r"(\d+)\.png$", f)] if m]
        self.offset = max(nums) + 1 if nums else 0   # a second run continues the numbering

    def frame(self, n: int, pil: Image.Image) -> str:
        path = os.path.join(self.frames_dir, f"{n:05d}.png")
        pil.save(path, compress_level=1)
        return path

    def write(self, name: str, obj: Dict[str, Any]) -> None:
        with open(os.path.join(self.root, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


#: Name colours of the log panel (Colonist draws player names in the player's colour; a white
#: name is grey on the light panel).  Light theme / dark theme.
_NAME_RGB_LIGHT: Dict[str, RGB] = {"red": (226, 74, 64), "blue": (48, 120, 212), "orange": (238, 140, 40),
                                   "green": (60, 160, 80), "white": (150, 150, 150), "purple": (140, 80, 190),
                                   "brown": (120, 80, 50), "pink": (230, 120, 180)}
_NAME_RGB_DARK: Dict[str, RGB] = dict(_NAME_RGB_LIGHT, red=(240, 100, 90), blue=(95, 155, 240), green=(85, 195, 105),
                                      purple=(180, 130, 230), brown=(185, 135, 95), white=(235, 235, 235))


def layout_for(profile: Optional[UiProfile], box: Optional[Sequence[int]], size: Tuple[int, int]) -> Any:
    """The board parser's ``layout``: the profile (its ``log`` region replaced by ``box`` when that
    differs, on a copy), or ``{"log": box}`` in pixels without a profile, or ``None``."""
    if profile is not None and profile.regions:
        try:
            same = box is None or profile.pixel_box("log", size) == tuple(int(v) for v in box)
        except ValueError:
            same = False
        if same:
            return profile
        w, h = size
        p2 = UiProfile(regions=dict(profile.regions), screen=profile.screen, ocr=profile.ocr, path=profile.path)
        try:
            p2.set_region("log", (max(0.0, box[0] / w), max(0.0, box[1] / h), min(1.0, box[2] / w),  # type: ignore
                                  min(1.0, box[3] / h)))                                              # type: ignore
        except ValueError:
            return profile
        return p2
    if box is not None:
        return {"log": tuple(int(v) for v in box)}
    return None


def confident_texts(result: Any, min_conf: float = 0.6) -> List[str]:
    """The confident, complete entries of a log read (``result.texts()`` when the reader has it)."""
    f = getattr(result, "texts", None)
    if callable(f):
        try:
            return [str(t) for t in f(min_conf=min_conf, include_partial=False)]
        except TypeError:
            pass
    return [str(getattr(ln, "text", "")) for ln in getattr(result, "lines", None) or []
            if float(getattr(ln, "confidence", 0.0) or 0.0) >= min_conf and not getattr(ln, "partial", False)
            and str(getattr(ln, "text", "")).strip()]


def name_colours(arr: np.ndarray, box: Optional[PixelBox], colours: Sequence[str],
                 profile: Optional[UiProfile] = None) -> Optional[Dict[str, RGB]]:
    """``players`` for the log reader: the name colours of the players in the parse.

    A profile the reader was taught on keeps the reader's own palette (``None``), or gives the
    learned colours when it stores them by colour word; otherwise the default name colours of the
    panel's theme (a dark panel -> the brightened set), restricted to ``colours``."""
    cols = [str(c).lower() for c in colours if c]
    if not cols:
        return None
    if profile is not None and profile.ocr:
        for k in ("name_colours", "name_colors", "players"):
            v = profile.ocr.get(k)
            if isinstance(v, dict):
                out = {c: tuple(int(x) for x in v[c][:3]) for c in cols
                       if isinstance(v.get(c), (list, tuple)) and len(v[c]) >= 3}
                return out or None       # type: ignore[return-value]
        return None
    dark = False
    if box is not None:
        x0, y0, x1, y1 = box
        crop = arr[y0:y1:4, x0:x1:4]
        if crop.size:
            dark = float(np.median(crop[..., :3].mean(axis=2))) < 110.0
    table = _NAME_RGB_DARK if dark else _NAME_RGB_LIGHT
    return {c: table[c] for c in cols if c in table} or None


class LiveSession:
    """The live reader: :meth:`step` one screenshot at a time (see the module doc).

    ``me`` is our colour (default: what the parser infers), ``profile`` a
    :class:`~catanbot.vision.profile.UiProfile` (its ``log`` region is the log panel; it is also
    the parser's layout), ``session_path`` the card-counting session file (created, or continued
    when it follows the same game), ``read_log`` switches the game log off, ``log_box`` is the
    log panel as pixels ``(x0, y0, x1, y1)`` or screen fractions (it overrides the profile's
    region; with neither, the panel is found on the first frame - ``find_log_panel``, or the
    reader asked with ``box=None`` - and again after reads keep failing), ``log_reader`` /
    ``parse_fn`` replace the log OCR's ``read_log_panel`` and the board parser
    (``parse_fn(img, me=, layout=) -> ParseResult``), ``evaluator`` values offers (default: the
    heuristic evaluator), ``record_dir`` records the session and ``clock`` gives the time.
    ``log_finder`` / ``log_cache`` replace ``find_log_panel`` / the ``LineCache``, ``players``
    the name colours given to the reader, ``tolerance`` the change detection's (0..255 on the
    block-averaged thumbnail) and ``postprocess(parsed)`` corrects each merged parse in place
    (the CLI's ``--fix``).
    """

    def __init__(self, me: Optional[str] = None, profile: Optional[UiProfile] = None,
                 session_path: Optional[str] = None, read_log: bool = True,
                 log_box: Optional[Sequence[float]] = None, log_reader: Optional[Callable[..., Any]] = None,
                 parse_fn: Optional[Callable[..., Any]] = None, evaluator: Any = None,
                 record_dir: Optional[str] = None, clock: Callable[[], float] = time.time,
                 log_finder: Optional[Callable[..., Any]] = None, log_cache: Any = None,
                 players: Optional[Dict[str, RGB]] = None, tolerance: int = 16, seed: int = 0,
                 postprocess: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.me_colour = me.strip().lower() if me else None
        self.postprocess = postprocess
        self.profile = profile
        self.session_path = session_path
        self.read_log = bool(read_log)
        self.log_box_arg = tuple(float(v) for v in log_box) if log_box is not None else None
        self.parse_fn = parse_fn or _default_parse
        self.evaluator = evaluator
        self.clock = clock
        self.players_rgb = dict(players) if players else None
        self.seed = int(seed)
        self.memory = BoardMemory()
        self.stream = LogStream(clock=clock)
        self.tracker: Any = None
        self._tracker_off = False
        self._tracker_due = False
        self._defer = 0
        self._fed_upto = 0
        self.tracker_overlap = 12
        self.card_count: Optional[Dict[str, Any]] = None
        self._warned: Dict[str, int] = {}
        self.frames = 0
        self.skipped = 0
        self.parses = 0
        self.log_reads = 0
        self.offers_evaluated = 0
        self.new_games = 0
        self.state: Optional[GameState] = None
        self.parsed: Optional[Dict[str, Any]] = None
        self.me: Optional[int] = None
        self._board_sig = ""
        self._count_sig = ""
        self._size: Optional[Tuple[int, int]] = None
        self._board_region = _Region(tolerance)
        self._log_region = _Region(tolerance)
        self._box: Optional[PixelBox] = None
        self._box_source = ""
        self._detect_wait = 0
        self._read_failures = 0
        self._pending_warnings: List[str] = []
        # the log reader
        self._logocr = None
        self._reader = log_reader
        self._finder = log_finder
        self.cache = log_cache
        if self.read_log and self._reader is None:
            self._logocr = load_logocr()
            if self._logocr is None:
                self.read_log = False
                self._pending_warnings.append(logocr_missing_message())
            else:
                self._reader = getattr(self._logocr, "read_log_panel")
        if self.read_log and self._finder is None and self._logocr is not None:
            self._finder = getattr(self._logocr, "find_log_panel", None)   # an injected reader finds its own panel
        if self.read_log and self.cache is None and self._logocr is not None and hasattr(self._logocr, "LineCache"):
            try:
                self.cache = self._logocr.LineCache()
            except Exception:  # pragma: no cover - a reader without a default-constructible cache
                self.cache = None
        # recording
        self.recorder: Optional[_Recorder] = None
        if record_dir:
            try:
                self.recorder = _Recorder(record_dir)
            except OSError as ex:
                self._pending_warnings.append(f"cannot record to {record_dir}: {ex}; recording is off")

    # --- warnings --------------------------------------------------------------------------
    def _warn(self, upd: LiveUpdate, msg: str) -> None:
        if not msg:
            return
        k = _warn_key(msg)
        n = self._warned.get(k)
        if n is None:
            if len(self._warned) >= 5000:        # an endless variety of warnings: stop reporting new ones
                return
            upd.warnings.append(msg)
            n = 0
        self._warned[k] = n + 1

    # --- the log box -------------------------------------------------------------------------
    def _resolve_box(self, pil: Image.Image, size: Tuple[int, int], upd: LiveUpdate) -> Optional[PixelBox]:
        w, h = size
        if self.log_box_arg is not None:
            b = self.log_box_arg
            if all(0.0 <= v <= 1.0 for v in b):
                b = (b[0] * w, b[1] * h, b[2] * w, b[3] * h)
            x0, y0, x1, y1 = (int(round(v)) for v in b)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 > x0 and y1 > y0:
                return (x0, y0, x1, y1)
            self._warn(upd, f"the log region {self.log_box_arg} lies outside the {w}x{h} screen")
            return None
        if self.profile is not None and self.profile.region("log") is not None:
            try:
                return self.profile.pixel_box("log", size)
            except ValueError as ex:
                self._warn(upd, f"the profile's log region is not usable: {ex}")
                return None
        if self._box is not None:
            return self._box
        if not self.read_log or pil is None:
            return None
        if self._detect_wait > 0:            # nothing found lately: look again in a while, not every frame
            self._detect_wait -= 1
            return None
        if self._finder is None:
            return None                      # the reader is asked with box=None (it finds the panel itself)
        try:
            box = self._finder(pil, profile=self.profile)
        except Exception as ex:
            self._warn(upd, f"finding the log panel failed: {type(ex).__name__}: {ex}")
            box = None
        if box is None:
            return None
        try:
            self._box = tuple(int(round(float(v))) for v in box)  # type: ignore[assignment]
        except (TypeError, ValueError):
            return None
        self._box_source = "detected"
        return self._box

    def _layout(self, size: Tuple[int, int], box: Optional[PixelBox]) -> Any:
        return layout_for(self.profile, box, size)

    def _name_colours(self, arr: np.ndarray, box: Optional[PixelBox]) -> Optional[Dict[str, RGB]]:
        cols = self.memory.colours or [str(p.get("color", "")).lower() for p in (self.parsed or {}).get("players") or []]
        if self.players_rgb:
            return {c: self.players_rgb[c] for c in cols if c in self.players_rgb} or None
        return name_colours(arr, box, cols, self.profile)

    # --- one frame ---------------------------------------------------------------------------
    def step(self, img: Any) -> LiveUpdate:
        """Read one screenshot (path, PIL image or RGB array); never raises for a bad frame (the
        problem becomes a warning)."""
        t_start = time.perf_counter()
        now = float(self.clock())
        frame_no = self.frames + (self.recorder.offset if self.recorder is not None else 0)
        upd = LiveUpdate(time=now, frame=frame_no)
        self.frames += 1
        for w in self._pending_warnings:
            self._warn(upd, w)
        self._pending_warnings = []
        try:
            pil, arr = _as_image(img)
        except Exception as ex:
            self._warn(upd, f"frame not readable: {ex}")
            upd.total_ms = (time.perf_counter() - t_start) * 1e3
            return upd
        size = (arr.shape[1], arr.shape[0])
        if size != self._size:
            self._size = size
            self._board_region.clear()
            self._log_region.clear()
            if self._box_source == "detected":
                self._box = None
        box = self._resolve_box(pil, size, upd) if self.read_log else self._resolve_box_static(size, upd)
        factor = max(1, int(round(max(size) / 480.0)))
        thumb = frame_thumbnail(arr, factor)
        log_t, rest_t = self._split(thumb, box, factor)
        board_changed = self._board_region.changed(rest_t)
        log_changed = self.read_log and log_t is not None and self._log_region.changed(log_t)
        need_log = self.read_log and (log_changed or bool(self.stream.pending)
                                      or (box is None and self._detect_wait == 0))
        changed_any = False
        # --- the game log (first: it may find the panel the board parse must leave out) -------------
        if need_log and self._reader is not None:
            changed_any = True
            t0 = time.perf_counter()
            self._read_log(pil, arr, box, upd)
            upd.ocr_ms = (time.perf_counter() - t0) * 1e3
            if box is None and self._box is not None:      # the reader found the panel itself
                box = self._box
                log_t, rest_t = self._split(thumb, box, factor)
                board_changed = self._board_region.changed(rest_t)
            if log_t is not None:
                self._log_region.accept(log_t)
        # --- the board -------------------------------------------------------------------------
        if board_changed:
            changed_any = True
            t0 = time.perf_counter()
            self._read_board(pil, size, box, upd)
            upd.parse_ms = (time.perf_counter() - t0) * 1e3
            self._board_region.accept(rest_t)
            if upd.new_game and self.read_log and self._reader is not None:
                t0 = time.perf_counter()               # the new game's log, read into the fresh stream
                self._read_log(pil, arr, box, upd)
                upd.ocr_ms = (upd.ocr_ms or 0.0) + (time.perf_counter() - t0) * 1e3
        elif self.memory.pending:
            bu = self.memory.revote()                      # an unchanged frame confirms what it shows
            if bu is not None:
                self._apply_board(bu, [], upd)
        upd.skipped = not changed_any
        if not changed_any:
            self.skipped += 1
        # --- offers, card count, recording -------------------------------------------------------
        self._offers(upd)
        self._card_count(upd, board_changed)
        upd.state, upd.parsed, upd.me = self.state, self.parsed, self.me
        self._record(pil, upd, changed_any)
        upd.total_ms = (time.perf_counter() - t_start) * 1e3
        return upd

    def _resolve_box_static(self, size: Tuple[int, int], upd: LiveUpdate) -> Optional[PixelBox]:
        """Without log reading the log box still keeps the panel out of the board analysis."""
        if (self.profile is not None and self.profile.region("log") is not None) or self.log_box_arg is not None:
            return self._resolve_box(None, size, upd)  # type: ignore[arg-type]
        return None

    @staticmethod
    def _split(thumb: np.ndarray, box: Optional[PixelBox], f: int) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if box is None:
            return None, thumb
        th, tw = thumb.shape[:2]
        x0, y0, x1, y1 = box
        lx0, ly0 = max(0, x0 // f), max(0, y0 // f)
        lx1, ly1 = min(tw, -(-x1 // f)), min(th, -(-y1 // f))
        if lx1 <= lx0 or ly1 <= ly0:
            return None, thumb
        log_t = np.ascontiguousarray(thumb[ly0:ly1, lx0:lx1])
        rest = thumb.copy()
        rest[max(0, ly0 - 1):ly1 + 1, max(0, lx0 - 1):lx1 + 1] = 0
        return log_t, rest

    def _read_log(self, pil: Image.Image, arr: np.ndarray, box: Optional[PixelBox], upd: LiveUpdate) -> None:
        self.log_reads += 1
        try:
            res = self._reader(pil, box=box, profile=self.profile, players=self._name_colours(arr, box),
                               cache=self.cache)
        except Exception as ex:
            self._warn(upd, f"reading the game log failed: {type(ex).__name__}: {ex}")
            self._note_read_failure()
            return
        for w in getattr(res, "warnings", None) or []:
            self._warn(upd, f"log: {w}")
        lines = list(getattr(res, "lines", None) or [])
        rbox = getattr(res, "box", None)
        if box is None:
            try:
                self._box = tuple(int(round(float(v))) for v in rbox) if rbox is not None else None  # type: ignore
            except (TypeError, ValueError):
                self._box = None
            if self._box is not None:
                self._box_source = "detected"
            else:
                self._detect_wait = 10
                self._warn(upd, "no game-log panel found on the screen; give its region with --log-region x,y,w,h "
                                "(or ui-profile FILE --detect SCREENSHOT / --set log=...)")
                return
        conf = getattr(res, "confidence", None)
        if not lines or (conf is not None and float(conf) < 0.3 and not any(
                float(getattr(ln, "confidence", 0.0) or 0.0) >= 0.6 for ln in lines)):
            self._note_read_failure()
        else:
            self._read_failures = 0
        if self.recorder is not None:
            self._safe_record("ocr.jsonl", {
                "time": round(upd.time, 3), "frame": upd.frame, "box": list(box or rbox or []) or None,
                "confidence": None if conf is None else float(conf),
                "lines": [{"text": str(getattr(ln, "text", "")), "confidence": float(getattr(ln, "confidence", 0) or 0),
                           "partial": bool(getattr(ln, "partial", False)), "key": str(getattr(ln, "key", "") or ""),
                           "box": list(getattr(ln, "box", None) or []) or None} for ln in lines]}, upd)
        new = self.stream.update(lines if lines else None, now=upd.time, frame=upd.frame,
                                 allow_gap=not self.memory.suspect)
        for w in self.stream.warnings:
            self._warn(upd, w)
        if new:
            upd.new_entries.extend(new)
            upd.changed_log = True
            if any(e.countable for e in new):
                self._tracker_due = True

    def _note_read_failure(self) -> None:
        self._read_failures += 1
        if self._read_failures >= 5 and self._box_source == "detected":
            self._box = None                   # the panel moved (window resized, layout changed): find it again
            self._box_source = ""
            self._read_failures = 0
            self._log_region.clear()
            self._board_region.clear()

    def _read_board(self, pil: Image.Image, size: Tuple[int, int], box: Optional[PixelBox], upd: LiveUpdate) -> None:
        self.parses += 1
        try:
            layout = self._layout(size, box)
            result = self.parse_fn(pil, me=self.me_colour, layout=layout)
        except Exception as ex:
            self._warn(upd, f"board not read in this frame: {ex}")
            return
        parsed = getattr(result, "parsed", None)
        if not isinstance(parsed, dict):
            self._warn(upd, "board not read in this frame: the parser returned no parse")
            return
        upd.parse_warnings = list(getattr(result, "warnings", None) or [])
        for w in upd.parse_warnings:
            self._warn(upd, w)
        bu = self.memory.update(parsed, getattr(result, "confidence", None))
        if self.recorder is not None:
            self._safe_record("parses.jsonl", {"time": round(upd.time, 3), "frame": upd.frame, "parsed": bu.parsed,
                                               "raw": parsed, "confidence": getattr(result, "confidence", None),
                                               "warnings": upd.parse_warnings}, upd)
        self._apply_board(bu, bu.warnings, upd)

    def _apply_board(self, bu: BoardUpdate, warns: Sequence[str], upd: LiveUpdate) -> None:
        from .schema import parsed_to_state
        for w in warns:
            self._warn(upd, w)
        if bu.new_game:
            self._new_game(upd)
        parsed = bu.parsed
        if self.postprocess is not None:          # e.g. the CLI's --fix corrections
            fixed = json.loads(json.dumps(parsed, default=_json_default))
            try:
                self.postprocess(fixed)
                parsed = fixed
            except Exception as ex:
                self._warn(upd, f"{ex}")
        try:
            state = parsed_to_state(parsed)
        except Exception as ex:  # pragma: no cover - parsed_to_state never raises on missing fields
            self._warn(upd, f"the parse could not be turned into a game state: {ex}")
            return
        if parsed.get("dice"):
            state.phase = PHASE_MAIN
            state.dice = int(parsed["dice"])
        me = None
        want = self.me_colour or parsed.get("me") or (state.players[0].color if state.players else None)
        if want:
            k = str(want).strip().lower()
            me = next((i for i, p in enumerate(state.players) if p.color == k or (p.name or "").lower() == k), None)
            if me is None and state.players:
                self._warn(upd, f"your colour '{want}' is not among the players read ({[p.color for p in state.players]})")
        self.state, self.parsed, self.me = state, parsed, me
        sig = _sig(parsed, ("hexes", "robber", "players", "dice", "current_player"))
        if sig != self._board_sig:
            self._board_sig = sig
            upd.changed_board = True
        csig = _sig(parsed, ("players", "bank"))
        if csig != self._count_sig:
            self._count_sig = csig
            self._tracker_due = True

    def _new_game(self, upd: LiveUpdate) -> None:
        upd.new_game = True
        self.new_games += 1
        self.stream.reset()
        self._fed_upto = 0
        self._tracker_due = False
        if self.tracker is not None and self.session_path and os.path.exists(self.session_path):
            try:
                os.replace(self.session_path, self.session_path + ".previous")
                self._warn(upd, f"new game: the card-count session of the last game was kept as "
                                f"{self.session_path}.previous")
            except OSError as ex:
                self._warn(upd, f"new game: could not keep the old session file: {ex}")
        self.tracker = None
        self._tracker_off = False
        self.card_count = None

    # --- offers ------------------------------------------------------------------------------
    def _seat(self, who: Optional[str]) -> Optional[int]:
        if who is None or self.state is None:
            return None
        if self.tracker is not None:
            s = self.tracker.seat_of(who)
            if s is not None:
                return s
        k = _norm_text(who).strip(":")
        if k in ("you", "me", "yourself", "your"):
            return self.me
        for i, p in enumerate(self.state.players):
            if p.color == k or _norm_text(p.name) == k:
                return i
        return None

    def _offers(self, upd: LiveUpdate) -> None:
        if not upd.new_entries or self.state is None or self.me is None:
            return
        me = self.me
        for e in upd.new_entries:
            if e.kind not in ("offer", "counter") or not e.countable or not self.stream.is_open(e):
                continue
            ev = e.event
            proposer = self._seat(ev.player)
            if proposer is None:
                self._warn(upd, f"offer from an unknown player: '{e.text}'")
                continue
            if proposer == me:
                continue                              # our own offer
            if e.kind == "counter":
                other = self._seat(ev.other) if ev.other else None
                if ev.other and other != me:
                    continue                          # a counter to someone else's offer
                ours_open = any(self._seat(o.event.player) == me for o in self.stream.open_offers.values()
                                if o.event is not None)
                if not ev.other and not (self.state.current == me or ours_open):
                    continue
            if ev.cards is None or ev.get is None:
                self._warn(upd, f"offer not readable (cards missing): '{e.text}'")
                continue
            try:
                v = evaluate_offer(self.state, me, proposer, ev.cards, ev.get, tracker=self.tracker,
                                   evaluator=self._evaluator(), seed=self.seed + e.index)
            except Exception as ex:
                self._warn(upd, f"offer evaluation failed for '{e.text}': {type(ex).__name__}: {ex}")
                continue
            v.kind, v.text, v.entry, v.time = e.kind, e.text, e.index, upd.time
            self.offers_evaluated += 1
            upd.offers.append(v)
            if self.recorder is not None:
                self._safe_record("offers.jsonl", dict(v.to_dict(), frame=upd.frame), upd)

    def _evaluator(self) -> Any:
        if self.evaluator is None:
            from ..heuristic import HeuristicEvaluator
            self.evaluator = HeuristicEvaluator()
        return self.evaluator

    # --- card counting -----------------------------------------------------------------------
    def _card_count(self, upd: LiveUpdate, board_changed: bool) -> None:
        if not self.read_log or self._tracker_off or self.state is None or self.me is None:
            return
        if not self.state.players or self.memory.colours is None:
            return                                    # wait until the player list is settled
        if self.tracker is None:
            from ..colonist_log import ColonistLogTracker, SessionError
            try:
                self.tracker = ColonistLogTracker.open_session(self.session_path, self.state, self.me)
            except (SessionError, OSError) as ex:
                self._tracker_off = True
                self._warn(upd, f"card counting is off: {ex}")
                return
            self._tracker_due = True
        if [p.color for p in self.state.players] != list(self.tracker.colors):
            self._warn(upd, "the players on screen differ from the card count's; the count is not updated")
            return
        if not self._tracker_due or self.memory.suspect:
            return
        # the screen's hand sizes must not be ahead of the confirmed log: never while an entry on
        # screen waits for confirmation; right after a board change wait a frame (animations),
        # but at most two
        if self.stream.pending:
            return
        if board_changed and self._defer < 2:
            self._defer += 1
            return
        self._defer = 0
        self._tracker_due = False
        # the confirmed entries not fed yet, after an overlap with the ones fed before (the tracker
        # aligns on it); a gap among them splits the window so the tracker sees the gap too
        start = max(0, self._fed_upto - self.tracker_overlap)
        g = self.stream.last_gap_index
        if g >= self._fed_upto and g > start:
            windows = [[e.event for e in self.stream.entries[start:g] if e.countable and e.event is not None],
                       self.stream.window(g)]
        else:
            windows = [self.stream.window(start)]
        windows = [w for w in windows if w]
        bank = list(self.state.bank) if (self.parsed or {}).get("bank") else None
        try:
            self.tracker.update(windows, self.state, bank)
        except Exception as ex:
            self._warn(upd, f"card count update failed: {type(ex).__name__}: {ex}")
            return
        self._fed_upto = len(self.stream.entries)
        for w in self.tracker.warnings:
            self._warn(upd, f"card count: {w}")
        try:
            rep = self.tracker.report()
        except Exception as ex:  # pragma: no cover - defensive
            self._warn(upd, f"card count report failed: {ex}")
            return
        self.card_count = rep
        upd.card_count = rep
        if self.session_path:
            try:
                self.tracker.save(self.session_path)
            except OSError as ex:
                self._warn(upd, f"could not save the card-count session {self.session_path}: {ex}")

    # --- recording ---------------------------------------------------------------------------
    def _safe_record(self, name: str, obj: Dict[str, Any], upd: LiveUpdate) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.write(name, obj)
        except (OSError, TypeError, ValueError) as ex:
            self._warn(upd, f"recording stopped: {ex}")
            self.recorder = None

    def _record(self, pil: Image.Image, upd: LiveUpdate, changed: bool) -> None:
        if self.recorder is None:
            return
        try:
            if changed:
                self.recorder.frame(upd.frame, pil)
            for e in upd.new_entries:
                self.recorder.write("events.jsonl", {"time": round(e.time, 3), "frame": upd.frame, "index": e.index,
                                                     "text": e.text, "kind": e.kind,
                                                     "confidence": round(float(e.confidence), 4),
                                                     "countable": e.countable, "gap_before": e.gap_before})
        except (OSError, TypeError, ValueError) as ex:
            self._warn(upd, f"recording stopped: {ex}")
            self.recorder = None

    # --- summary -----------------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {"frames": self.frames, "skipped": self.skipped, "parses": self.parses, "log_reads": self.log_reads,
                "entries": len(self.stream.entries), "offers": self.offers_evaluated, "gaps": self.stream.gaps,
                "new_games": self.new_games,
                "card_count": (self.card_count or {}).get("entries") if self.card_count else None}

    def close(self) -> None:
        """Save the card-count session (it is also saved after every update)."""
        if self.tracker is not None and self.session_path:
            try:
                self.tracker.save(self.session_path)
            except OSError:
                pass
