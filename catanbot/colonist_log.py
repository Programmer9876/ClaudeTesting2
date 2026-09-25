"""Card counting from Colonist.io's public game log (the advisor's ``--session`` / ``--game-log``).

The advisor reads one screenshot at a time: the board, every player's hand *size* and
development-card count, and our own hand.  Everything else a strong human counts is public and
printed in Colonist's log panel: dice and what each player received, builds, development-card
purchases and plays, bank / port trades, trades between players, offers and counter-offers,
Monopoly, Year of Plenty, Road Building, knights and robber moves.  Hidden to third parties are
only the card of a robber steal (the thief and the victim see it), the cards discarded on a 7
(the count is public) and the type of every unplayed development card.  This module turns that
log into the exact card counting the benchmarks' ``counted`` information mode does from
catanatron's action log (:mod:`catanbot.bench.public_info`): a
:class:`~catanbot.counting.CardCounter` over every hand (exact while nothing hidden happened, the
precise set of still-possible hands with their probabilities after a hidden steal / discard) plus
the development-card bookkeeping and the determinizations of
:class:`~catanbot.public_belief.PublicBelief`.

Pieces
------
* :class:`LogEvent` - the event schema, one object per log line (kinds :data:`EVENT_KINDS`).
  Players are referred to as written (name, colour or "you") and mapped to seats by the tracker.
* :func:`parse_log_text` - pasted / typed log text -> events through the phrase table
  :data:`PHRASES` (one regex per Colonist wording, first match wins) and the card notation of
  :func:`parse_cards` (``2 wood, 1 brick`` / ``wood wood brick`` / ``WWB`` / Colonist's names
  lumber, wool, grain).  Unrecognised lines and unreadable cards are reported, never dropped.
* :func:`events_from_json` - the ``log`` list of a parsed screenshot (the Claude-vision parser
  transcribes the visible log panel into it, :mod:`catanbot.vision.llm`) -> events.
* :class:`ColonistLogTracker` - the session (JSON file): consumes overlapping windows of the log
  (deduplicated by aligning each window with the entries already consumed), starts from the game
  start or - mid-game - from a production prior fitted to the hand sizes, repairs the count when
  a payment / hand size / own hand shows a missed or misread entry, and reports the "Card count".
* :class:`LogRenderer` - the inverse: catanbot engine actions as Colonist-style log lines (the
  end-to-end tests feed a whole engine game through it; it is also the reference of the wording).
"""
from __future__ import annotations

import heapq
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from .counting import CardCounter, _sub_multisets, hand_prior_weights
from .public_belief import PublicBelief
from .state import GameState
from .vision.schema import LOG_EVENT_KINDS, dev_type_from_name, resource_from_name

__all__ = [
    "EVENT_KINDS",
    "LogEvent",
    "Phrase",
    "PHRASES",
    "parse_cards",
    "format_cards",
    "parse_log_line",
    "parse_log_text",
    "events_from_json",
    "ColonistLogTracker",
    "SessionError",
    "LogRenderer",
]

#: Event kinds: the schema's (:data:`catanbot.vision.schema.LOG_EVENT_KINDS`, whose ``other`` is
#: read as ``ignored``) plus ``ignored`` (a known line without card information) and ``unknown``
#: (no phrase matched: reported to the user).
EVENT_KINDS: Tuple[str, ...] = tuple(k for k in LOG_EVENT_KINDS if k != "other") + ("ignored", "unknown")
BUILD_COSTS: Dict[str, Tuple[int, ...]] = {"road": B.COST_ROAD, "settlement": B.COST_SETTLEMENT, "city": B.COST_CITY}
#: Colonist's discard rule: more than this many cards on a 7 -> discard half (rounded down).
DISCARD_LIMIT = 7
#: Resource names on Colonist's cards (lumber / wool / grain) in resource order.
COLONIST_CARD_NAMES = ("lumber", "brick", "wool", "grain", "ore")


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------
@dataclass
class LogEvent:
    """One public log entry.

    ``player`` / ``other`` are the players as written (name, colour or "you"); ``cards`` / ``get``
    are 5 resource counts or ``None`` when the cards could not be read (``problem`` says why).
    Per kind: ``roll`` (``value``), ``gain`` (``cards``; ``count`` = cards of unknown type;
    ``setup`` for starting resources), ``build`` (``item`` road / settlement / city, ``free`` for a
    setup placement), ``buy_dev``, ``play_dev`` (``item`` a :data:`board.DEV_NAMES` type),
    ``monopoly`` (``resource``, ``count`` = the total taken, ``taken_by`` per victim when shown),
    ``year_of_plenty`` (``cards``), ``bank_trade`` (``cards`` given, ``get`` received), ``player_trade``
    (``player`` gave ``cards`` to ``other`` and got ``get``), ``offer`` / ``counter`` (``player``
    offers ``cards`` for ``get``), ``steal`` (``player`` stole from ``other``: ``cards`` when the
    card is shown, else hidden), ``discard`` (``cards`` when shown, else ``count``), ``robber``,
    ``turn``, ``ignored``, ``unknown``.
    """

    kind: str
    player: Optional[str] = None
    other: Optional[str] = None
    cards: Optional[List[int]] = None
    get: Optional[List[int]] = None
    count: Optional[int] = None
    value: Optional[int] = None
    item: Optional[str] = None
    resource: Optional[int] = None
    taken_by: Optional[Dict[str, int]] = None
    free: bool = False
    setup: bool = False
    text: str = ""
    problem: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON form (the schema of :data:`catanbot.vision.schema.LOG_ENTRY_SCHEMA`, cards by name)."""
        d: Dict[str, Any] = {"kind": self.kind}
        for k in ("player", "other"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        for k in ("cards", "get"):
            v = getattr(self, k)
            if v is not None:
                d[k] = {B.RESOURCE_NAMES[r]: int(v[r]) for r in range(5) if v[r]}
        for k in ("count", "value", "item"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.resource is not None:
            d["resource"] = B.RESOURCE_NAMES[self.resource]
        if self.taken_by:
            d["taken_by"] = dict(self.taken_by)
        if self.free:
            d["free"] = True
        if self.setup:
            d["setup"] = True
        if self.text:
            d["text"] = self.text
        if self.problem:
            d["problem"] = self.problem
        return d


def _counts_text(counts: Sequence[int]) -> str:
    return A._counts_str(counts)


# ---------------------------------------------------------------------------
# Card notation
# ---------------------------------------------------------------------------
_CARD_WORDS: Dict[str, int] = {k: v for k, v in B.RESOURCE_ALIASES.items() if v != B.DESERT}
_CARD_WORDS.update({"lumbers": B.WOOD, "woods": B.WOOD, "logs": B.WOOD, "log": B.WOOD, "bricks": B.BRICK,
                    "wools": B.SHEEP, "sheeps": B.SHEEP, "grains": B.WHEAT, "wheats": B.WHEAT, "ores": B.ORE})
_UNKNOWN_WORDS = {"card", "cards", "?", "unknown", "resource", "resources", "hidden", "cardback"}
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10}
_FILLER_WORDS = {"and", "x", "of", "the"}
#: Compact letters (upper case only): W/L wood, B/C brick, S sheep, G wheat, O ore, ? unknown.
_LETTERS: Dict[str, int] = {"W": B.WOOD, "L": B.WOOD, "B": B.BRICK, "C": B.BRICK, "S": B.SHEEP,
                            "G": B.WHEAT, "O": B.ORE}


@dataclass
class CardsParse:
    """Result of :func:`parse_cards`: ``counts`` of known resources, ``unknown`` face-down cards,
    ``bad`` tokens that are not cards, ``empty`` when there was nothing at all (lost icons)."""

    counts: List[int] = field(default_factory=lambda: [0] * 5)
    unknown: int = 0
    bad: List[str] = field(default_factory=list)
    empty: bool = True

    @property
    def ok(self) -> bool:
        return not self.bad and not self.empty


def parse_cards(text: Optional[str]) -> CardsParse:
    """Cards written as ``2 wood, 1 brick`` / ``wood wood brick`` / ``2x wood`` / ``wood x2`` /
    ``WWB`` (letters W/L wood, B/C brick, S sheep, G wheat, O ore, ? unknown) / Colonist's names
    (lumber, wool, grain), plurals and ``a card`` / ``2 cards`` / ``?`` for face-down cards."""
    out = CardsParse()
    if text is None:
        return out
    toks = re.findall(r"(?<![A-Za-z])[xX]\d+|\d+[xX](?![A-Za-z])|\d+|[A-Za-z]+|\?", str(text).replace("×", "x"))
    pending: Optional[int] = None
    last: Optional[Tuple[str, int]] = None      # the previous card item, for a trailing 'x2'
    for tok in toks:
        out.empty = False
        low = tok.lower()
        if tok.isdigit() or re.fullmatch(r"\d+x", low):
            k = int(low.rstrip("x"))
            if pending is not None or k <= 0:
                out.bad.append(tok)
            pending = k
            continue
        if re.fullmatch(r"x\d+", low):
            if last is None or pending is not None:
                out.bad.append(tok)
                continue
            kind, r = last                     # 'wood x3': the previous item counts 3 in total
            k = int(low[1:]) - 1
            if kind == "res":
                out.counts[r] += k
            else:
                out.unknown += k
            last = None
            continue
        if low in _NUMBER_WORDS and pending is None:
            pending = _NUMBER_WORDS[low]
            continue
        if low in _FILLER_WORDS:
            continue
        n = 1 if pending is None else pending
        pending = None
        if low in _CARD_WORDS:
            out.counts[_CARD_WORDS[low]] += n
            last = ("res", _CARD_WORDS[low])
        elif low in _UNKNOWN_WORDS:
            out.unknown += n
            last = ("unknown", -1)
        elif tok.isupper() and all(ch in _LETTERS for ch in tok) and (n == 1 or len(tok) == 1):
            for ch in tok:
                out.counts[_LETTERS[ch]] += n
            last = ("res", _LETTERS[tok[-1]])
        else:
            out.bad.append(tok)
            last = None
    if pending is not None:
        out.bad.append(str(pending))
    return out


def format_cards(counts: Sequence[int], style: str = "counts") -> str:
    """Cards as text in one of :data:`LogRenderer.STYLES` (the notations :func:`parse_cards` reads)."""
    if style == "letters":
        return "".join("WBSGO"[r] * int(counts[r]) for r in range(5))
    if style in ("words", "colonist"):
        names = B.RESOURCE_NAMES if style == "words" else COLONIST_CARD_NAMES
        return " ".join(names[r] for r in range(5) for _ in range(int(counts[r])))
    return ", ".join(f"{int(counts[r])} {B.RESOURCE_NAMES[r]}" for r in range(5) if counts[r])


# ---------------------------------------------------------------------------
# The phrase table (pasted / typed log text)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Phrase:
    """One row of the phrase table: an event ``kind``, a ``pattern`` (a case-insensitive regex
    over the whole cleaned line, with the slots of :data:`_SLOTS`), an ``example`` line and fixed
    event attributes ``attrs`` (e.g. ``free`` for a setup placement)."""

    kind: str
    pattern: str
    example: str
    attrs: Tuple[Tuple[str, Any], ...] = ()


#: Slots usable in a phrase pattern: {A} the acting player (up to 3 words), {B} the other player,
#: {CARDS} / {GET} cards (see :func:`parse_cards`), {N} a number, {RES} a resource word, {DICE}
#: "5 3" / "8", {ITEM} road / settlement / city, {DEV} a development card, {REST} anything.  A
#: card / resource slot written after a space is optional: a pasted line whose card icons were
#: lost ("Bob got") still matches, and the event reports its cards as not readable.
_SLOTS: Dict[str, str] = {
    " {CARDS}": r"(?:\s(?P<cards>.*?))?",
    " {GET}": r"(?:\s(?P<get>.*?))?",
    " {RES}": r"(?:\s(?P<res>[a-z]+))?",
    "{A}": r"(?P<a>\S+(?:\s\S+){0,2}?)",
    "{B}": r"(?P<b>\S+(?:\s\S+){0,2}?)",
    "{N}": r"(?P<n>\d+)",
    "{DICE}": r"(?P<dice>\d{1,2}(?:\s*(?:[+,&/]|and)?\s*\d)?)",
    "{ITEM}": r"(?P<item>road|settlement|city)",
    "{DEV}": r"(?P<dev>knight|soldier|road ?building|year ?of ?plenty|monopoly|victory ?points?|vp)",
    "{REST}": r"(?P<rest>.*?)",
}
_BANK = r"(?:the )?bank"

#: The phrase table: Colonist.io log wordings (best knowledge, see docs/USAGE.md "Card counting"),
#: tried in order - the first matching row makes the event.  Correct or extend it here.
PHRASES: Tuple[Phrase, ...] = (
    # --- setup ---------------------------------------------------------------------------
    Phrase("gain", r"{A} received starting resources:? {CARDS}", "Alice received starting resources wood brick ore",
           (("setup", True),)),
    Phrase("build", r"{A} placed(?: an?)?(?: {ITEM})?", "Alice placed a Settlement", (("free", True),)),
    # --- dice ----------------------------------------------------------------------------
    Phrase("roll", r"{A} rolled:?(?: an?)?(?: {DICE})?", "Alice rolled 5 3"),
    # --- development cards (the one-line forms before the plain "used") -------------------------
    Phrase("monopoly", r"{A} (?:used|played) (?:an? |the )?monopoly(?: card)?,? and (?:stole|took|got|received) "
                       r"{N} {RES}(?: cards?)?(?: in total)?", "Bob used Monopoly and stole 5 ore"),
    Phrase("year_of_plenty", r"{A} (?:used|played) (?:an? |the )?year of plenty(?: card)?,? and (?:took|got|received)"
                             r":? {CARDS}(?: from " + _BANK + r")?", "Bob used Year of Plenty and took wood ore"),
    Phrase("play_dev", r"{A} (?:used|played) (?:an? |the )?{DEV}(?: card)?", "Bob used Knight"),
    Phrase("monopoly", r"{A} (?:stole|monopolized|monopolised)(?: all)? {N} {RES}(?: cards?)?"
                       r"(?: from (?:everyone|everybody|all players|all))?", "Bob stole 5 ore"),
    # --- robber --------------------------------------------------------------------------
    Phrase("steal", r"{A} stole:? {CARDS} from:? {B}", "Carol stole a card from Bob"),
    Phrase("robber", r"{A} (?:moved|placed|put) (?:the )?robber{REST}", "Bob moved Robber to 6 wheat"),
    Phrase("discard", r"{A} discarded:? {CARDS}", "Bob discarded 4 cards"),
    # --- trades --------------------------------------------------------------------------
    Phrase("bank_trade", r"{A} gave " + _BANK + r":? {CARDS} and took:? {GET}", "Bob gave bank 4 wood and took 1 ore"),
    Phrase("bank_trade", r"{A} gave {CARDS} and (?:got|took|received) {GET} from " + _BANK,
           "Bob gave 4 wood and got 1 ore from bank"),
    Phrase("bank_trade", r"{A} traded:? {CARDS} for:? {GET} with:? " + _BANK, "Bob traded 3 wool for 1 ore with bank"),
    Phrase("player_trade", r"{A} traded:? {CARDS} for:? {GET} with:? {B}", "Alice traded 1 wood for 1 ore with Bob"),
    Phrase("counter", r"{A} (?:counter[- ]?offered|countered|proposed a counter[- ]?offer|made a counter[- ]?offer)"
                      r"(?: to {B})?:?(?: to give)? {CARDS} for:? {GET}", "Bob counter-offered 1 ore for 2 wood"),
    Phrase("offer", r"{A} (?:wants to give|offered|offers|proposed|proposes)(?: to give)?:? {CARDS} for:? {GET}",
           "Alice wants to give 1 wood for 1 ore"),
    # --- Year of Plenty's take (after the trades: "gave bank ... and took ...") ------------------
    Phrase("year_of_plenty", r"{A} took(?: from " + _BANK + r")?:? {CARDS}(?: from " + _BANK + r")?",
           "Bob took from bank wood ore"),
    # --- known lines without card information (before the generic "got") ---------------------
    Phrase("ignored", r".*\b(?:largest army|longest road)\b.*", "Bob received Largest Army"),
    Phrase("ignored", r"{A} (?:won|wins|has won) the game.*", "Bob won the game!"),
    Phrase("ignored", r"(?:no player|nobody|no one|no-one) (?:gets|got|received|receives) (?:any )?(?:resources?|cards?).*",
           "No player gets resources"),
    Phrase("ignored", r"{A} (?:accepted|rejected|declined|cancelled|canceled|withdrew)\b.*", "Bob rejected the trade"),
    Phrase("ignored", r"{A} (?:is selecting|is choosing|is placing|is discarding|joined|left|reconnected|"
                      r"disconnected)\b.*", "Bob is selecting a player to steal from"),
    # --- production and other gains ------------------------------------------------------------
    Phrase("gain", r"{A} (?:got|received|gets|receives):? {CARDS}", "Bob got 2 wood, 1 ore"),
    # --- builds and purchases --------------------------------------------------------------------
    Phrase("build", r"{A} built(?: an?)?(?: {ITEM})?", "Alice built a Road"),
    Phrase("buy_dev", r"{A} bought(?: an?)?(?: development cards?| dev cards?| development| devcard)?",
           "Bob bought Development Card"),
    # --- turn structure --------------------------------------------------------------------------
    Phrase("turn", r"{A} ended (?:their |his |her |the |your )?turn", "Bob ended their turn"),
    Phrase("turn", r"(?:it'?s |it is )?{A}'s turn", "Bob's turn"),
)


def _compile(ph: Phrase) -> "re.Pattern[str]":
    pat = ph.pattern
    for slot, rx in _SLOTS.items():
        pat = pat.replace(slot, rx)
    return re.compile(r"^" + pat + r"$", re.IGNORECASE)


_COMPILED: List[Tuple[Phrase, "re.Pattern[str]"]] = [(ph, _compile(ph)) for ph in PHRASES]


def _clean_line(raw: str) -> str:
    """Whitespace collapsed; leading bullets / timestamps and trailing punctuation dropped."""
    line = re.sub(r"\s+", " ", str(raw)).strip()
    line = re.sub(r"^(?:[-*•>]+\s*|\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s+)", "", line)
    return line.rstrip(" .!").strip()


def _dice_value(text: Optional[str]) -> Tuple[Optional[int], str]:
    if not text:
        return None, ""
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) == 2 and all(1 <= x <= 6 for x in nums):
        return sum(nums), ""
    if len(nums) == 1 and 2 <= nums[0] <= 12:
        return nums[0], ""
    return None, f"dice '{text}' not readable"


def _single_card(cp: CardsParse) -> Optional[int]:
    """The resource of a one-card steal, ``None`` when face down / not shown."""
    if cp.ok and sum(cp.counts) == 1 and cp.unknown == 0:
        return next(r for r in range(5) if cp.counts[r])
    return None


def _cards_or_problem(ev: LogEvent, text: Optional[str], what: str, attr: str = "cards") -> CardsParse:
    cp = parse_cards(text)
    if cp.ok and cp.unknown == 0 and sum(cp.counts) > 0:
        setattr(ev, attr, list(cp.counts))
    elif cp.ok and cp.unknown and not any(cp.counts) and attr == "cards":
        ev.count = cp.unknown                 # "got 2 cards": the number without the types
    else:
        bad = f" (not cards: {', '.join(cp.bad)})" if cp.bad else ""
        ev.problem = (ev.problem + "; " if ev.problem else "") + (
            f"{what} not readable{bad}" if not cp.empty else f"{what} missing (card icons lost in the paste?)")
    return cp


def _event_from_match(ph: Phrase, m: "re.Match[str]", line: str) -> LogEvent:
    g = m.groupdict()
    ev = LogEvent(ph.kind, player=(g.get("a") or None), other=(g.get("b") or None), text=line)
    for k, v in ph.attrs:
        setattr(ev, k, v)
    kind = ph.kind
    if kind == "roll":
        ev.value, problem = _dice_value(g.get("dice"))
        ev.problem = problem
    elif kind in ("gain", "year_of_plenty"):
        _cards_or_problem(ev, g.get("cards"), "cards")
    elif kind == "build":
        ev.item = g["item"].lower() if g.get("item") else None
        if ev.item is None and not ev.free:
            ev.problem = "building type not readable (road / settlement / city)"
    elif kind == "play_dev":
        t = dev_type_from_name(re.sub(r"\s+", "", g.get("dev") or ""))
        ev.item = B.DEV_NAMES[t] if t is not None else None
        if ev.item is None:
            ev.problem = "development card not readable"
    elif kind == "monopoly":
        ev.count = int(g["n"])
        r = resource_from_name(g.get("res"))
        if r is None or r == B.DESERT:
            r = _CARD_WORDS.get((g.get("res") or "").lower())
        ev.resource = r
        if r is None:
            ev.problem = f"monopoly resource '{g['res']}' not readable" if g.get("res") else "monopoly resource missing"
    elif kind == "steal":
        cp = parse_cards(g.get("cards"))
        r = _single_card(cp)
        if r is not None:
            ev.cards = [1 if x == r else 0 for x in range(5)]
        elif cp.bad or (cp.ok and sum(cp.counts) + cp.unknown > 1):
            ev.problem = f"stolen card '{g.get('cards')}' not readable (taken as hidden)"
    elif kind == "discard":
        cp = parse_cards(g.get("cards"))
        if cp.ok and cp.unknown == 0 and sum(cp.counts) > 0:
            ev.cards = list(cp.counts)
        elif cp.ok and cp.unknown and not any(cp.counts):
            ev.count = cp.unknown
        elif not cp.empty:     # nothing at all: the count is half the hand (the tracker knows its size)
            ev.problem = "discarded cards not readable" + (f" (not cards: {', '.join(cp.bad)})" if cp.bad else "")
    elif kind in ("bank_trade", "player_trade", "offer", "counter"):
        _cards_or_problem(ev, g.get("cards"), "given cards")
        _cards_or_problem(ev, g.get("get"), "received cards", attr="get")
    return ev


def parse_log_line(line: str) -> Optional[LogEvent]:
    """One log line -> an event (``None`` for an empty line); ``unknown`` when no phrase matches."""
    line = _clean_line(line)
    if not line:
        return None
    for ph, rx in _COMPILED:
        m = rx.match(line)
        if m:
            return _event_from_match(ph, m, line)
    return LogEvent("unknown", text=line, problem="no phrase in the table matches")


def parse_log_text(text: str) -> List[LogEvent]:
    """Pasted / typed Colonist log text (one entry per line, oldest first) -> events."""
    out: List[LogEvent] = []
    for raw in str(text).splitlines():
        ev = parse_log_line(raw)
        if ev is not None:
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# JSON events (the parsed screenshot's ``log``)
# ---------------------------------------------------------------------------
def _json_cards(v: Any) -> Tuple[Optional[List[int]], int, str]:
    """``{"wood": 2, "unknown": 1}`` / a 5-list -> (counts or None, face-down count, problem)."""
    if v is None:
        return None, 0, ""
    counts = [0] * 5
    unknown = 0
    if isinstance(v, dict):
        for k, x in v.items():
            try:
                n = int(x)
            except (TypeError, ValueError):
                return None, 0, f"bad card count {x!r}"
            key = str(k).strip().lower()
            r = _CARD_WORDS.get(key)
            if r is None and key in _UNKNOWN_WORDS:
                unknown += max(0, n)
            elif r is None:
                return None, 0, f"unknown resource {k!r}"
            else:
                counts[r] += max(0, n)
    elif isinstance(v, (list, tuple)) and len(v) == 5:
        try:
            counts = [max(0, int(x)) for x in v]
        except (TypeError, ValueError):
            return None, 0, f"bad cards {v!r}"
    elif isinstance(v, str):
        cp = parse_cards(v)
        if not cp.ok:
            return None, 0, f"cards '{v}' not readable"
        counts, unknown = list(cp.counts), cp.unknown
    else:
        return None, 0, f"bad cards {v!r}"
    return counts, unknown, ""


def events_from_json(entries: Any) -> Tuple[List[LogEvent], List[str]]:
    """The ``log`` list of a parsed screenshot -> ``(events, warnings)``.

    Entries follow :data:`catanbot.vision.schema.LOG_ENTRY_SCHEMA`; ``other`` becomes
    ``ignored``.  Two conveniences for model output: a ``roll`` may carry ``gains``
    (``{player: cards}``, expanded to ``gain`` events) and a ``play_dev`` of a Monopoly / Year of
    Plenty may carry its effect (expanded to a ``monopoly`` / ``year_of_plenty`` event)."""
    events: List[LogEvent] = []
    warns: List[str] = []
    if entries is None:
        return events, warns
    if not isinstance(entries, (list, tuple)):
        return events, ["log: expected a list of entries"]
    for i, e in enumerate(entries):
        if isinstance(e, str):
            ev = parse_log_line(e)
            if ev is not None:
                events.append(ev)
            continue
        if not isinstance(e, dict):
            warns.append(f"log entry {i}: malformed {e!r}")
            continue
        kind = str(e.get("kind") or "").strip().lower()
        if kind == "other":
            kind = "ignored"
        text = str(e.get("text") or "")
        if kind not in EVENT_KINDS:
            events.append(LogEvent("unknown", text=text or json.dumps(e, sort_keys=True),
                                   problem=f"unknown log entry kind {e.get('kind')!r}"))
            continue
        ev = LogEvent(kind, player=(str(e["player"]) if e.get("player") else None),
                      other=(str(e["other"]) if e.get("other") else None), text=text,
                      free=bool(e.get("free")), setup=bool(e.get("setup")))
        problems = []
        cards, unknown, pr = _json_cards(e.get("cards"))
        if pr:
            problems.append(pr)
        get, _unk, pr2 = _json_cards(e.get("get"))
        if pr2:
            problems.append(pr2)
        if kind == "steal":
            ev.cards = cards if cards is not None and sum(cards) == 1 and not unknown else None
        elif cards is not None and sum(cards) > 0:
            ev.cards = cards
        elif unknown:
            ev.count = unknown
        ev.get = get if get is not None and sum(get) > 0 else None
        for k in ("count", "value"):
            if e.get(k) is not None:
                try:
                    setattr(ev, k, int(e[k]))
                except (TypeError, ValueError):
                    problems.append(f"bad {k} {e[k]!r}")
        item = e.get("item")
        if item is not None:
            it = str(item).strip().lower()
            if it in BUILD_COSTS:
                ev.item = it
            else:
                t = dev_type_from_name(it)
                ev.item = B.DEV_NAMES[t] if t is not None else None
                if t is None:
                    problems.append(f"unknown item {item!r}")
        if e.get("resource") is not None:
            r = resource_from_name(e.get("resource"))
            ev.resource = r if r is not None and r != B.DESERT else None
            if ev.resource is None:
                problems.append(f"unknown resource {e.get('resource')!r}")
        if isinstance(e.get("taken_by"), dict):
            try:
                ev.taken_by = {str(k): int(v) for k, v in e["taken_by"].items()}
            except (TypeError, ValueError):
                problems.append("bad taken_by")
        if kind in ("gain", "year_of_plenty", "bank_trade", "player_trade", "offer", "counter") \
                and ev.cards is None and ev.count is None:
            problems.append("cards not readable")
        if kind in ("bank_trade", "player_trade", "offer", "counter") and ev.get is None:
            problems.append("received cards not readable")
        if kind == "build" and ev.item not in BUILD_COSTS and not ev.free:
            problems.append("building type not readable")
        if kind == "monopoly" and ev.resource is None:
            problems.append("monopoly resource not readable")
        ev.problem = "; ".join(problems)
        events.append(ev)
        if kind == "roll" and isinstance(e.get("gains"), dict):
            for who, cs in e["gains"].items():
                c2, u2, pr3 = _json_cards(cs)
                events.append(LogEvent("gain", player=str(who), cards=c2 if c2 and sum(c2) else None,
                                       count=u2 or None, problem=pr3))
        if kind == "play_dev" and ev.item == "monopoly" and ev.resource is not None and ev.count is not None:
            events.append(LogEvent("monopoly", player=ev.player, resource=ev.resource, count=ev.count,
                                   taken_by=ev.taken_by, text=text))
            ev.count, ev.resource, ev.taken_by = None, None, None
        if kind == "play_dev" and ev.item == "year_of_plenty" and ev.cards is not None:
            events.append(LogEvent("year_of_plenty", player=ev.player, cards=ev.cards, text=text))
            ev.cards = None
    return events, warns


# ---------------------------------------------------------------------------
# Hand distributions
# ---------------------------------------------------------------------------
def _multinomial(hand: Sequence[int], w: Sequence[float]) -> float:
    n = sum(hand)
    p = math.factorial(n)
    for r in range(5):
        p = p / math.factorial(hand[r]) * (w[r] ** hand[r])
    return p


def _compositions(k: int, w: Sequence[float], top: Optional[int] = None) -> List[Tuple[Tuple[int, ...], float]]:
    """Every ``k``-card hand with its multinomial probability under per-card weights ``w``
    (most likely first; ``top`` keeps only the most likely)."""
    tot = sum(w)
    w = [x / tot for x in w] if tot > 0 else [0.2] * 5
    out: List[Tuple[Tuple[int, ...], float]] = []

    def rec(r: int, left: int, cur: List[int]) -> None:
        if r == 4:
            h = tuple(cur + [left])
            p = _multinomial(h, w)
            if p > 0:
                out.append((h, p))
            return
        for x in range(left + 1):
            rec(r + 1, left - x, cur + [x])

    rec(0, int(k), [])
    out.sort(key=lambda hp: (-hp[1], hp[0]))
    return out[:top] if top else out


def _best_first(dists: Sequence[Sequence[Tuple[Tuple[int, ...], float]]], cap: int, accept,
                max_scan: int) -> List[Tuple[Tuple[Tuple[int, ...], ...], float]]:
    """The ``cap`` most likely joint hands of independent per-player distributions (each sorted,
    most likely first) that pass ``accept`` - best-first over the product, at most ``max_scan``."""
    k = len(dists)
    if any(not d for d in dists):
        return []

    def prob(idx):
        return math.prod(dists[i][idx[i]][1] for i in range(k))

    start = (0,) * k
    heap = [(-prob(start), start)]
    seen = {start}
    out = []
    scanned = 0
    while heap and len(out) < cap and scanned < max_scan:
        negp, idx = heapq.heappop(heap)
        scanned += 1
        joint = tuple(dists[i][idx[i]][0] for i in range(k))
        if accept(joint):
            out.append((joint, -negp))
        for i in range(k):
            if idx[i] + 1 < len(dists[i]):
                nxt = idx[:i] + (idx[i] + 1,) + idx[i + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (-prob(nxt), nxt))
    return out


def _counter_to_dict(c: CardCounter) -> Dict[str, Any]:
    hyps = sorted(c.hyps.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"hypotheses": [[[list(h) for h in joint], w] for joint, w in hyps], "size": list(c.size),
            "last_bank": c.last_bank, "max_hypotheses": c.max_hypotheses, "stats": dict(c.stats)}


def _counter_from_dict(d: Dict[str, Any], n: int) -> CardCounter:
    c = CardCounter([[0] * 5] * n, max_hypotheses=int(d.get("max_hypotheses", CardCounter.MAX_HYPOTHESES)))
    hyps = {}
    for joint, w in d["hypotheses"]:
        hyps[tuple(tuple(int(x) for x in h) for h in joint)] = float(w)
    if not hyps or any(len(j) != n for j in hyps):
        raise ValueError("bad counter hypotheses")
    c.hyps = hyps
    c.size = [int(x) for x in d["size"]]
    c.last_bank = [int(x) for x in d["last_bank"]] if d.get("last_bank") is not None else None
    c.stats.update(d.get("stats") or {})
    c._marg = None
    return c


# ---------------------------------------------------------------------------
# The session tracker
# ---------------------------------------------------------------------------
class SessionError(ValueError):
    """A session file that cannot be used with this screenshot (other game, other seat, not a session)."""


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


class ColonistLogTracker(PublicBelief):
    """Card counting over a Colonist game log, across advisor calls (a JSON session file).

    :meth:`update` takes the log windows of one advisor call (each a list of :class:`LogEvent`,
    e.g. the entries visible on this screenshot and / or pasted text) and the screenshot's state:

    1. every window is aligned with the entries already consumed (longest overlap of the window's
       head with the consumed tail - a window may also contain older entries before it, or lie
       entirely inside it; candidate alignments that contradict the screenshot's hand sizes are
       skipped) and only the new entries are applied; a window sharing nothing with the tail is
       applied whole with a warning (entries may be missing);
    2. the first window starts the count: at the game start (setup placements / starting
       resources before any roll) every hand starts empty and the count is exact; otherwise
       (mid-game) the hands just before the window's last stretch of fully readable entries start
       from a production-weighted prior fitted to the hand sizes (``estimated`` players, said so in
       the report) and that stretch is applied;
    3. :meth:`reconcile` checks the screenshot: our own hand, every hand size (a mismatch means a
       missed / misread entry: reported, and the difference is branched in as unknown cards drawn
       from the production prior or removed like a hidden discard), and the bank when visible.

    A payment no hypothesis can make (a missed gain) is repaired the same way (the smallest
    unrecorded gain that makes it possible), so the count never collapses to a false certainty.
    The search's determinizations come from :class:`~catanbot.public_belief.PublicBelief` (the
    same code as the benchmarks' ``counted`` information mode).
    """

    FORMAT = "catanbot.colonist_log session"
    VERSION = 1
    MAX_TAIL = 5000
    #: most likely hands per player kept in a mid-game prior, and the joint hypotheses built from them
    PRIOR_TOP = 128
    PRIOR_JOINT = 1024

    def __init__(self, colors: Sequence[str], names: Sequence[str], me: int,
                 max_hypotheses: int = CardCounter.MAX_HYPOTHESES, vps_to_win: int = 10):
        self.colors = [_norm(c) for c in colors]
        self.names = [str(x) for x in names]
        self.n = len(self.colors)
        self.me = int(me)
        self.max_hypotheses = int(max_hypotheses)
        self.vps_to_win = int(vps_to_win)
        self.aliases: Dict[str, int] = {}
        self._learn_names(self.names)
        self.counter: Optional[CardCounter] = None
        self.origin: Optional[str] = None            # "start" / "mid-game" once counting started
        self.estimated = [False] * self.n            # hand began as a mid-game prior (not yet washed out)
        self.reasons: List[List[str]] = [[] for _ in range(self.n)]
        self.log_played = [[0] * 5 for _ in range(self.n)]
        self.played = [[0] * 5 for _ in range(self.n)]
        self.dev_count = [0] * self.n
        self.known_dev: List[Optional[List[int]]] = [None] * self.n
        self.my_dev = [0] * 5
        self.deck = sum(B.DEV_DECK_COUNTS)
        self.bought_this_turn = [0] * self.n
        self.free_roads = [0] * self.n
        self.pending_play: Dict[int, int] = {}       # seat -> Monopoly / Year of Plenty awaiting its effect line
        self.pending_discards: Dict[int, int] = {}   # always empty: Colonist discards are applied at once
        self.setup = True
        self.turn_player = -1
        self.tail: List[str] = []
        self.entries = 0
        self.reported: List[str] = []                # unknown lines already reported (normalised text)
        self.bank_known = False
        self.stats: Dict[str, int] = {"entries": 0, "unknown_lines": 0, "unreadable": 0, "hidden_steals": 0,
                                      "hidden_discards": 0, "repairs": 0, "gaps": 0}
        # per update (not persisted)
        self.warnings: List[str] = []
        self.new_entries = 0
        self._state: Optional[GameState] = None
        self._rest: Sequence[LogEvent] = ()
        self._target: Optional[List[int]] = None
        self._window_bank: Optional[List[int]] = None
        self._hidden_7: Dict[int, int] = {}          # hidden discards of the 7 being resolved (seat -> cards)
        self._unknown_names: List[str] = []

    # --- identities -------------------------------------------------------------------
    @classmethod
    def for_state(cls, state: GameState, me: int, **kw) -> "ColonistLogTracker":
        return cls([p.color for p in state.players], [p.name or p.color for p in state.players], me, **kw)

    def _learn_names(self, names: Iterable[str]) -> None:
        for i, c in enumerate(self.colors):
            self.aliases[c] = i
        for i, nm in enumerate(names):
            k = _norm(nm)
            if k and i < self.n and (k not in self.aliases or k == self.colors[i]):
                self.aliases[k] = i
                if k != self.colors[i]:
                    self.names[i] = str(nm)

    def seat_of(self, who: Optional[str]) -> Optional[int]:
        """Seat of a player as written in the log (name, colour, "you"); ``None`` if unknown."""
        if who is None:
            return None
        k = _norm(who)
        if k in ("you", "me", "yourself", "your"):
            return self.me
        if k in self.aliases:
            return self.aliases[k]
        k2 = re.sub(r"[^\w#]+", "", k)
        for a, i in self.aliases.items():
            if re.sub(r"[^\w#]+", "", a) == k2:
                return i
        m = re.fullmatch(r"(.+?) ?\((.+)\)", k)          # "Bob (blue)"
        if m:
            return self.seat_of(m.group(1)) if self.seat_of(m.group(1)) is not None else self.seat_of(m.group(2))
        return None

    def label(self, j: int) -> str:
        c = self.colors[j]
        nm = self.names[j]
        return c if not nm or _norm(nm) == c else f"{c} ({nm})"

    # --- persistence ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.FORMAT, "version": self.VERSION, "colors": self.colors, "names": self.names,
            "me": self.me, "aliases": self.aliases, "origin": self.origin, "estimated": self.estimated,
            "reasons": self.reasons, "max_hypotheses": self.max_hypotheses, "vps_to_win": self.vps_to_win,
            "counter": _counter_to_dict(self.counter) if self.counter is not None else None,
            "log_played": self.log_played, "bought_this_turn": self.bought_this_turn,
            "free_roads": self.free_roads, "pending_play": {str(k): v for k, v in self.pending_play.items()},
            "setup": self.setup, "turn_player": self.turn_player, "entries": self.entries,
            "tail": self.tail[-self.MAX_TAIL:], "reported": self.reported[-1000:], "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColonistLogTracker":
        if not isinstance(d, dict) or d.get("format") != cls.FORMAT:
            raise SessionError("not a card-counting session file")
        if int(d.get("version", 0)) != cls.VERSION:
            raise SessionError(f"session format version {d.get('version')} is not supported")
        tr = cls(d["colors"], d["names"], int(d["me"]), max_hypotheses=int(d.get("max_hypotheses", 4096)),
                 vps_to_win=int(d.get("vps_to_win", 10)))
        tr.aliases.update({str(k): int(v) for k, v in (d.get("aliases") or {}).items()})
        tr.origin = d.get("origin")
        tr.estimated = [bool(x) for x in d.get("estimated") or [False] * tr.n]
        tr.reasons = [list(x) for x in d.get("reasons") or [[] for _ in range(tr.n)]]
        tr.counter = _counter_from_dict(d["counter"], tr.n) if d.get("counter") else None
        tr.log_played = [list(x) for x in d.get("log_played") or tr.log_played]
        tr.bought_this_turn = list(d.get("bought_this_turn") or tr.bought_this_turn)
        tr.free_roads = list(d.get("free_roads") or tr.free_roads)
        tr.pending_play = {int(k): int(v) for k, v in (d.get("pending_play") or {}).items()}
        tr.setup = bool(d.get("setup", True))
        tr.turn_player = int(d.get("turn_player", -1))
        tr.entries = int(d.get("entries", 0))
        tr.tail = [str(x) for x in d.get("tail") or []]
        tr.reported = [str(x) for x in d.get("reported") or []]
        tr.stats.update(d.get("stats") or {})
        return tr

    @classmethod
    def open_session(cls, path: Optional[str], state: GameState, me: int) -> "ColonistLogTracker":
        """The session in ``path`` (checked against this screenshot's players and seat), or a new
        one when ``path`` is empty / does not exist yet."""
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError) as ex:
                raise SessionError(f"{path} is not a card-counting session file ({ex}); refusing to overwrite it")
            try:
                tr = cls.from_dict(d)
            except SessionError as ex:
                raise SessionError(f"{path}: {ex}; refusing to overwrite it")
            except (KeyError, TypeError, ValueError) as ex:
                raise SessionError(f"{path}: damaged session file ({type(ex).__name__}: {ex})")
            colors = [_norm(p.color) for p in state.players]
            if colors != tr.colors:
                raise SessionError(f"session {path} follows a game with players {tr.colors}, this position has "
                                   f"{colors}: use a new --session file for a new game, or fix the players "
                                   "(--fix 'players=...')")
            if me != tr.me:
                raise SessionError(f"session {path} counts cards for {tr.colors[tr.me]}, this position is for "
                                   f"{colors[me]} (--me)")
            tr._learn_names(p.name or p.color for p in state.players)
            return tr
        return cls.for_state(state, me)

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))
        os.replace(tmp, path)

    # --- one advisor call -------------------------------------------------------------
    def update(self, windows: Sequence[Sequence[LogEvent]], state: GameState,
               bank: Optional[Sequence[int]] = None) -> None:
        """Consume the log ``windows`` of one advisor call, then reconcile with the screenshot
        ``state`` (hand sizes, our hand) and the ``bank`` when it is visible."""
        self.warnings = []
        self.new_entries = 0
        self._unknown_names = []
        self._state = state
        self._learn_names(p.name or p.color for p in state.players)
        target = [self._screen_size(state, j) for j in range(self.n)]
        try:
            for k, events in enumerate(windows):
                last = k == len(windows) - 1
                self._window_bank = [int(x) for x in bank] if (last and bank is not None) else None
                self._consume(list(events), target if last else None)
            if self.counter is None:          # no log yet: estimates from the screenshot alone
                self._window_bank = [int(x) for x in bank] if bank is not None else None
                self._start([], [], target)
            self.reconcile(state, bank)
        finally:
            self._state = None
            self._rest = ()
            self._target = None
            self._window_bank = None

    @staticmethod
    def _screen_size(state: GameState, j: int) -> int:
        p = state.players[j]
        return sum(p.resources) if p.hand_known else int(p.hand_size)

    def _weights(self, j: int) -> List[float]:
        if self._state is not None:
            try:
                return hand_prior_weights(self._state, j)
            except Exception:  # pragma: no cover - a malformed screenshot board
                pass
        return [0.2] * 5

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _mark(self, j: int, reason: str) -> None:
        if reason not in self.reasons[j]:
            self.reasons[j].append(reason)

    def _refresh_flags(self) -> None:
        c = self.counter
        for j in range(self.n):
            if self.estimated[j] and c.size[j] == 0:
                self.estimated[j] = False
            if self.reasons[j] and not self.estimated[j] and c.is_exact(j):
                self.reasons[j] = []

    # --- keys and alignment ------------------------------------------------------------
    def _ref(self, who: Optional[str]) -> str:
        s = self.seat_of(who)
        return "-" if who is None else (str(s) if s is not None else "?" + _norm(who))

    def _key(self, ev: LogEvent) -> str:
        def cs(v):
            return "" if v is None else ".".join(str(int(x)) for x in v)
        return "|".join(str(x) for x in (ev.kind, self._ref(ev.player), self._ref(ev.other), cs(ev.cards), cs(ev.get),
                                          ev.count, ev.value, ev.item, ev.resource, int(ev.free)))

    def _align(self, keys: List[str], events: List[LogEvent], target: Optional[List[int]]) -> Tuple[int, int]:
        """``(index of the first new entry in the window, overlap length)``; ``(-1, 0)`` for a window
        lying inside the consumed log; ``(0, 0)`` when it shares nothing with it (a gap)."""
        tail = self.tail
        if not tail:
            return 0, 0
        L = len(keys)
        cands: List[Tuple[int, int]] = []
        for j in range(1, L + 1):
            m = 0
            while m < j and m < len(tail) and keys[j - 1 - m] == tail[-1 - m]:
                m += 1
            if m > 0 and (m == j or m == len(tail)):
                cands.append((j, m))
        if not cands:
            if L >= 2 and self._inside_tail(keys):
                return -1, 0
            return 0, 0
        cands.sort(key=lambda jm: (-jm[1], -jm[0]))
        if target is not None and self.counter is not None:
            for j, m in cands:
                if self._sizes_fit(events[j:], target):
                    return j, m
        return cands[0]

    def _inside_tail(self, keys: List[str]) -> bool:
        tail = self.tail
        L = len(keys)
        return any(tail[i:i + L] == keys for i in range(len(tail) - L + 1))

    def _net(self, events: Sequence[LogEvent]) -> Tuple[List[int], set, Optional[List[int]]]:
        """Hand-size change per seat over ``events`` (seats whose change is not readable in
        ``unknown``) and our own per-resource change (``None`` when not fully readable)."""
        n = self.n
        d = [0] * n
        unknown: set = set()
        mine: Optional[List[int]] = [0] * 5
        setup = self.setup
        free = list(self.free_roads)
        me = self.me

        def add(j, cards, sign):
            nonlocal mine
            d[j] += sign * sum(cards)
            if j == me and mine is not None:
                mine = [mine[r] + sign * cards[r] for r in range(5)]

        def lost(j):
            nonlocal mine
            unknown.add(j)
            if j == me:
                mine = None

        for ev in events:
            p = self.seat_of(ev.player)
            o = self.seat_of(ev.other)
            k = ev.kind
            if k in ("roll", "turn"):
                if k == "roll":
                    setup = False
                free = [0] * n
                continue
            if p is None:
                continue
            if k == "gain":
                if ev.cards is not None:
                    add(p, ev.cards, +1)
                else:
                    lost(p)
            elif k == "build":
                if ev.free or setup:
                    pass
                elif ev.item == "road" and free[p] > 0:
                    free[p] -= 1
                elif ev.item in BUILD_COSTS:
                    add(p, BUILD_COSTS[ev.item], -1)
                else:
                    lost(p)
            elif k == "buy_dev":
                add(p, B.COST_DEV, -1)
            elif k == "play_dev":
                if ev.item == "road_building":
                    free[p] = 2
            elif k == "year_of_plenty":
                if ev.cards is not None:
                    add(p, ev.cards, +1)
                else:
                    lost(p)
            elif k == "monopoly":
                if ev.taken_by:
                    for who, x in ev.taken_by.items():
                        v = self.seat_of(who)
                        if v is None:
                            lost(p)
                        else:
                            add(v, [x if r == ev.resource else 0 for r in range(5)], -1)
                            add(p, [x if r == ev.resource else 0 for r in range(5)], +1)
                else:
                    for j in range(n):
                        lost(j)
            elif k in ("bank_trade", "player_trade"):
                if ev.cards is None or ev.get is None or (k == "player_trade" and o is None):
                    lost(p)
                    if o is not None:
                        lost(o)
                    continue
                add(p, ev.get, +1)
                add(p, ev.cards, -1)
                if k == "player_trade":
                    add(o, ev.cards, +1)
                    add(o, ev.get, -1)
            elif k == "steal":
                if o is None:
                    lost(p)
                    continue
                d[p] += 1
                d[o] -= 1
                if me in (p, o):
                    if ev.cards is None:
                        mine = None
                    elif mine is not None:
                        s = 1 if p == me else -1
                        mine = [mine[r] + s * ev.cards[r] for r in range(5)]
            elif k == "discard":
                if ev.cards is not None:
                    add(p, ev.cards, -1)
                elif ev.count is not None:
                    d[p] -= ev.count
                    if p == me:
                        mine = None
                else:
                    lost(p)
        return d, unknown, mine

    def _sizes_fit(self, events: Sequence[LogEvent], target: Sequence[int]) -> bool:
        d, unknown, _ = self._net(events)
        c = self.counter
        return all(c.size[j] + d[j] == target[j] for j in range(self.n) if j not in unknown)

    # --- consuming windows ------------------------------------------------------------
    def _consume(self, events: List[LogEvent], target: Optional[List[int]]) -> None:
        counting: List[LogEvent] = []
        unknown_lines: List[str] = []
        for ev in events:
            if ev.kind == "unknown":
                key = _norm(ev.text)
                if key not in self.reported:
                    self.reported.append(key)
                    self.stats["unknown_lines"] += 1
                    unknown_lines.append(ev.text + (f" ({ev.problem})" if ev.problem and "no phrase" not in ev.problem
                                                    else ""))
                continue
            if ev.kind == "ignored":
                continue
            counting.append(ev)
        if unknown_lines:
            shown = "; ".join(f"'{t}'" for t in unknown_lines[:5])
            more = f" and {len(unknown_lines) - 5} more" if len(unknown_lines) > 5 else ""
            self._warn(f"{len(unknown_lines)} log line(s) not understood (skipped; extend the phrase table or "
                       f"retype them, see docs/USAGE.md): {shown}{more}")
        keys = [self._key(ev) for ev in counting]
        if self.counter is None or (not self.tail and self.origin != "start"):
            self._start(counting, keys, target)
            return
        start, m = self._align(keys, counting, target)
        if start < 0:
            self._warn("the log window lies inside the part already consumed (scrolled up?): nothing new")
            return
        if m == 0 and counting:
            self.stats["gaps"] += 1
            self._warn(f"the log window shares no entry with the {len(self.tail)} consumed so far: entries between "
                       "the two may be missing; the hand sizes on screen resynchronise the count")
        self._apply_all(counting[start:], keys[start:], target)

    def _apply_all(self, events: Sequence[LogEvent], keys: Sequence[str], target: Optional[List[int]]) -> None:
        for i, (ev, key) in enumerate(zip(events, keys)):
            if self._hidden_7 and ev.kind != "discard":
                self._flush_discards(events[i:])
            self._rest = events[i + 1:]
            self._target = target
            self.apply(ev)
            self.tail.append(key)
            self.entries += 1
            self.new_entries += 1
            self.stats["entries"] += 1
        if self._hidden_7:
            self._flush_discards([])
        if len(self.tail) > self.MAX_TAIL:
            del self.tail[:len(self.tail) - self.MAX_TAIL]
        self._rest = ()

    @staticmethod
    def _starts_at_setup(events: Sequence[LogEvent]) -> bool:
        for ev in events:
            if ev.kind == "roll":
                return False
            if (ev.kind == "build" and ev.free) or (ev.kind == "gain" and ev.setup):
                return True
        return False

    def _start(self, events: List[LogEvent], keys: List[str], target: Optional[List[int]]) -> None:
        """Start counting on the first window: from the game start, or mid-game from a prior."""
        n = self.n
        if self._starts_at_setup(events):
            self.origin = "start"
            self.setup = True
            self.counter = CardCounter([[0] * 5] * n, max_hypotheses=self.max_hypotheses)
            self.estimated = [False] * n
            self._apply_all(events, keys, target)
            return
        self.origin = "mid-game"
        self.setup = False
        state = self._state
        end = list(target) if target is not None else (
            [self._screen_size(state, j) for j in range(n)] if state is not None else [0] * n)
        mine_end = list(state.players[self.me].resources) if state is not None and state.players[self.me].hand_known \
            else None
        # the last stretch of fully readable entries is applied on top of the prior
        s = 0
        for i, ev in enumerate(events):
            _d, unk, mine = self._net([ev])
            if unk or mine is None:
                s = i + 1
        d, unk, mine = self._net(events[s:])
        sizes = [end[j] - d[j] for j in range(n)]
        my_start = None if mine_end is None or mine is None else [mine_end[r] - mine[r] for r in range(5)]
        if min(sizes) < 0 or my_start is None or min(my_start) < 0 or sum(my_start) != sizes[self.me]:
            s = len(events)
            sizes = list(end)
            my_start = mine_end
        bank = None
        if self._window_bank is not None:     # the bank when the prior applies: before the stretch's changes
            d_bank = self._bank_net(events[s:])
            if d_bank is not None and min(self._window_bank[r] - d_bank[r] for r in range(5)) >= 0:
                bank = [self._window_bank[r] - d_bank[r] for r in range(5)]
        self.counter = self._prior(sizes, my_start, bank)
        self.estimated = [j != self.me and sizes[j] > 0 for j in range(n)]
        for j in range(n):
            if self.estimated[j]:
                self._mark(j, "the session started mid-game")
        # the entries before the stretch happened before the prior: consumed, not applied
        for key in keys[:s]:
            self.tail.append(key)
            self.entries += 1
        self._apply_all(events[s:], keys[s:], target)

    def _prior(self, sizes: Sequence[int], my_hand: Optional[Sequence[int]],
               bank: Optional[Sequence[int]] = None) -> CardCounter:
        """Joint hand hypotheses for a mid-game start: every opponent's ``sizes[j]`` cards drawn
        from its production-weighted prior (:func:`catanbot.counting.hand_prior_weights`), the most
        likely combinations whose per-resource totals the 19-card decks allow.  With the ``bank``
        (at that point) the totals are exact: the last opponent's hand is what the others leave."""
        n = self.n
        dists: List[List[Tuple[Tuple[int, ...], float]]] = []
        weights: Dict[int, List[float]] = {}
        for j in range(n):
            if j == self.me and my_hand is not None:
                dists.append([(tuple(int(x) for x in my_hand), 1.0)])
            elif sizes[j] <= 0:
                dists.append([((0,) * 5, 1.0)])
            else:
                weights[j] = self._weights(j)
                dists.append(_compositions(sizes[j], weights[j], top=self.PRIOR_TOP))
        cap = min(self.PRIOR_JOINT, self.max_hypotheses)
        joint: List[Tuple[Tuple[Tuple[int, ...], ...], float]] = []
        free = [j for j in range(n) if len(dists[j]) > 1]
        if bank is not None and free:
            cols = [B.BANK_PER_RESOURCE - int(bank[r]) for r in range(5)]
            last = free[-1]
            w_last = weights[last]
            tot_last = sum(w_last)
            rest = [j for j in range(n) if j != last]

            def forced(part) -> Optional[Tuple[int, ...]]:
                h = [cols[r] - sum(x[r] for x in part) for r in range(5)]
                return tuple(h) if min(h) >= 0 and sum(h) == sizes[last] else None

            for part, w in _best_first([dists[j] for j in rest], 4 * cap, lambda part: forced(part) is not None,
                                       max_scan=50 * cap):
                h = forced(part)
                full = list(part)
                full.insert(last, h)
                joint.append((tuple(full), w * _multinomial(h, [x / tot_last for x in w_last])))
            joint = sorted(joint, key=lambda jw: -jw[1])[:cap]
            if not joint:
                self._warn("the bank on screen fits no estimate of the hands (a misread hand size or bank?): "
                           "the estimate ignores it")

        def fits(joint) -> bool:
            return all(sum(h[r] for h in joint) <= B.BANK_PER_RESOURCE for r in range(5))

        if not joint:
            joint = _best_first(dists, cap, fits, max_scan=50 * self.PRIOR_JOINT)
        c = CardCounter([[0] * 5] * n, max_hypotheses=self.max_hypotheses)
        joint = [(jt, w) for jt, w in joint if w > 0]
        if joint:
            tot = sum(w for _, w in joint)
            c.hyps = {jt: w / tot for jt, w in joint}
        else:   # nothing fits the decks (a misread size): the single most likely hand per player
            c.hyps = {tuple(d[0][0] for d in dists): 1.0}
        c.size = [int(x) for x in sizes]
        c._marg = None
        return c

    # --- the events -------------------------------------------------------------------
    def _seat(self, who: Optional[str], ev: LogEvent) -> Optional[int]:
        s = self.seat_of(who)
        if s is None and who is not None and _norm(who) not in self._unknown_names:
            self._unknown_names.append(_norm(who))
            self._warn(f"unknown player '{who}' in the log (players: {', '.join(self.label(j) for j in range(self.n))}):"
                       f" tell me their colour with --fix 'COLOUR.name={who}'; their entries are skipped")
        return s

    def _unreadable(self, ev: LogEvent, what: str) -> None:
        self.stats["unreadable"] += 1
        self._warn(f"log entry '{ev.text or ev.kind}': {what}; the hand sizes on screen correct the count")

    def _new_turn(self, p: Optional[int]) -> None:
        self.bought_this_turn = [0] * self.n
        self.free_roads = [0] * self.n
        self.pending_play = {}
        self.turn_player = -1 if p is None else p

    def apply(self, ev: LogEvent) -> None:
        """Apply one (new) log entry to the count."""
        k = ev.kind
        c = self.counter
        if k in ("ignored", "unknown"):
            return
        p = self._seat(ev.player, ev)
        if k == "roll":
            self.setup = False
            if p is None or p != self.turn_player:
                self._new_turn(p)
            return
        if k == "turn":
            self._new_turn(None if "ended" in ev.text.lower() else p)
            return
        if p is None:
            return
        if not (k == "build" and ev.item == "road") and not (k == "play_dev" and ev.item == "road_building") \
                and k not in ("robber", "offer", "counter", "steal"):
            self.free_roads[p] = 0            # Road Building's roads are placed right away
        if ev.problem and k not in ("steal", "roll"):
            self._unreadable(ev, ev.problem)
        if k == "gain":
            if ev.cards is not None:
                c.observe_delta(p, ev.cards)
            elif ev.count:
                self._gain_unknown(p, ev.count, "cards received of unknown type")
        elif k == "build":
            if ev.free or self.setup:
                return
            if ev.item == "road" and self.free_roads[p] > 0:
                self.free_roads[p] -= 1
                return
            if ev.item in BUILD_COSTS:
                self._pay(p, [-x for x in BUILD_COSTS[ev.item]], f"built a {ev.item}")
        elif k == "buy_dev":
            self._pay(p, [-x for x in B.COST_DEV], "bought a development card")
            self.bought_this_turn[p] += 1
        elif k == "play_dev":
            if ev.item is None:
                return
            t = B.DEV_NAMES.index(ev.item)
            self.log_played[p][t] += 1
            if t == B.DEV_ROAD_BUILDING:
                self.free_roads[p] = 2
            elif t in (B.DEV_MONOPOLY, B.DEV_YEAR_OF_PLENTY):
                self.pending_play[p] = t
        elif k == "year_of_plenty":
            self._effect_of(p, B.DEV_YEAR_OF_PLENTY)
            if ev.cards is not None:
                c.observe_delta(p, ev.cards)
            else:
                self._gain_unknown(p, ev.count or 2, "Year of Plenty cards not read")
        elif k == "monopoly":
            self._effect_of(p, B.DEV_MONOPOLY)
            if ev.resource is not None:
                self._monopoly(p, ev)
        elif k == "bank_trade":
            if ev.cards is not None and ev.get is not None:
                self._pay(p, [ev.get[r] - ev.cards[r] for r in range(5)], "traded with the bank")
        elif k == "player_trade":
            o = self._seat(ev.other, ev)
            if o is not None and ev.cards is not None and ev.get is not None:
                self._pay(p, [ev.get[r] - ev.cards[r] for r in range(5)], f"traded with {self.label(o)}")
                self._pay(o, [ev.cards[r] - ev.get[r] for r in range(5)], f"traded with {self.label(p)}")
        elif k in ("offer", "counter"):
            if p != self.me and ev.cards is not None:
                if not any(all(j[p][r] >= ev.cards[r] for r in range(5)) for j in c.hyps):
                    self._warn(f"{self.label(p)} offered {_counts_text(ev.cards)}, which the count says they do not "
                               "hold: a missed or misread log entry?")
                else:
                    c.observe_offer(p, ev.cards, ev.get or [0] * 5)
        elif k == "steal":
            self._steal(p, ev)
        elif k == "discard":
            self._discard(p, ev)
        self._refresh_flags()

    def _effect_of(self, p: int, t: int) -> None:
        """A Monopoly / Year of Plenty effect line: the card was played (if its own line is missing)."""
        if self.pending_play.get(p) == t:
            del self.pending_play[p]
        else:
            self.log_played[p][t] += 1

    def _pay(self, j: int, d: Sequence[int], what: str) -> None:
        """A public change of ``j``'s hand; a payment no hypothesis can make is repaired with the
        smallest unrecorded gain (a missed / misread entry) instead of collapsing the count."""
        c = self.counter
        d = [int(x) for x in d]
        if min(d) >= 0 or any(all(joint[j][r] + d[r] >= 0 for r in range(5)) for joint in c.hyps):
            c.observe_delta(j, d)
            return
        if self.estimated[j] and self._redraw(j, d):
            c.observe_delta(j, d)
            return
        short = {joint: sum(max(0, -(joint[j][r] + d[r])) for r in range(5)) for joint in c.hyps}
        m = min(short.values())
        rep = c._replace

        def f(joint, w):
            if short[joint] != m:
                return ()
            h = joint[j]
            return ((rep(joint, j, [max(0, h[r] + d[r]) for r in range(5)]), w),)

        c.size[j] += sum(d) + m
        c._update(f, "repair")
        self.stats["repairs"] += 1
        self._mark(j, "a missed log entry")
        self._warn(f"{self.label(j)} {what} but the count had them {m} card(s) short: a missed or misread log "
                   "entry (assumed an unrecorded gain)")

    def _redraw(self, j: int, d: Sequence[int]) -> bool:
        """``j``'s hand is still a mid-game estimate and no hypothesis can make the change ``d``:
        the prior was wrong, not the log.  Every hypothesis trades as many of its other cards (a
        uniformly random subset) for the missing ones, keeping its size.  False if a hypothesis
        does not hold enough other cards."""
        c = self.counter
        rep = c._replace
        plans = {}
        for joint in c.hyps:
            h = joint[j]
            short = [max(0, -(h[r] + d[r])) for r in range(5)]
            pool = [0 if short[r] else max(0, h[r] + min(0, d[r])) for r in range(5)]
            m = sum(short)
            if sum(pool) < m:
                return False
            plans[joint] = (short, pool, m)

        def f(joint, w):
            short, pool, m = plans[joint]
            if m == 0:
                return ((joint, w),)
            h = joint[j]
            return [(rep(joint, j, [h[r] - sub[r] + short[r] for r in range(5)]), w * p)
                    for sub, p in _sub_multisets(pool, m)]

        return c._update(f, "prior correction")

    def _gain_unknown(self, j: int, k: int, reason: str) -> None:
        """``j`` gained ``k`` cards of unknown type: branch over them (production prior)."""
        if k <= 0:
            return
        c = self.counter
        comps = _compositions(k, self._weights(j), top=None if k <= 4 else 70)
        rep = c._replace

        def f(joint, w):
            h = joint[j]
            return [(rep(joint, j, [h[r] + x[r] for r in range(5)]), w * p) for x, p in comps]

        c.size[j] += k
        c._update(f, "unknown gain")
        self._mark(j, reason)

    def _lose_unknown(self, j: int, k: int, reason: str) -> None:
        """``j`` lost ``k`` cards of unknown type (like a hidden discard)."""
        c = self.counter
        k = min(k, c.size[j])
        if k > 0:
            c.observe_discard(j, n=k)
            self._mark(j, reason)

    def _steal(self, thief: int, ev: LogEvent) -> None:
        c = self.counter
        victim = self._seat(ev.other, ev)
        if victim is None or victim == thief:
            return
        res = next((r for r in range(5) if ev.cards[r]), None) if ev.cards is not None else None
        if res is not None:
            one = [1 if r == res else 0 for r in range(5)]
            if not any(joint[victim][res] > 0 for joint in c.hyps) and not (
                    self.estimated[victim] and self._redraw(victim, [-x for x in one])):
                c.observe_delta(victim, one)
                self.stats["repairs"] += 1
                self._mark(victim, "a missed log entry")
                self._warn(f"{self.label(thief)} stole {B.RESOURCE_NAMES[res]} from {self.label(victim)}, who had none "
                           "by the count: a missed or misread log entry")
            c.observe_steal(victim, thief, res)
            return
        if c.size[victim] <= 0:
            self._gain_unknown(victim, 1, "a missed log entry")
            self._warn(f"{self.label(thief)} stole from {self.label(victim)}, who had no cards by the count: a "
                       "missed or misread log entry")
        self.stats["hidden_steals"] += 1
        c.observe_steal(victim, thief, None)
        self._mark(thief, f"a hidden steal with {self.label(victim)}")
        self._mark(victim, f"a hidden steal with {self.label(thief)}")

    def _discard(self, p: int, ev: LogEvent) -> None:
        c = self.counter
        if ev.cards is not None:
            if not any(all(joint[p][r] >= ev.cards[r] for r in range(5)) for joint in c.hyps):
                self._pay(p, [-x for x in ev.cards], f"discarded {_counts_text(ev.cards)}")
            else:
                c.observe_discard(p, ev.cards)
            return
        k = ev.count
        if k is None:
            k = c.size[p] // 2 if c.size[p] > DISCARD_LIMIT else None
        if k is None or k <= 0:
            if k is None:
                self._unreadable(ev, "discard count not readable")
            return
        if c.size[p] < 2 * k:
            held = c.size[p]
            self._gain_unknown(p, 2 * k - held, "a missed log entry")
            self._warn(f"{self.label(p)} discarded {k} cards but held only {held} by the count: a missed or misread "
                       "log entry")
        self.stats["hidden_discards"] += 1
        self._hidden_7[p] = self._hidden_7.get(p, 0) + k

    def _flush_discards(self, after: Sequence[LogEvent]) -> None:
        """Apply the 7's hidden discards together (Colonist's discards are simultaneous).  With the
        bank right after them - the bank on screen minus the public bank changes of the entries
        ``after`` them - only the combinations matching it are kept (a single discarder is then
        pinned down, several keep only their split open), as :class:`PublicInfoTracker` does."""
        hidden, self._hidden_7 = self._hidden_7, {}
        c = self.counter
        bank = None
        if self._window_bank is not None:
            d = self._bank_net(after)
            if d is not None:
                bank = [self._window_bank[r] - d[r] for r in range(5)]
                if min(bank) < 0:
                    bank = None
        if bank is not None:
            saved = (dict(c.hyps), list(c.size), c.stats["resets"])
            c.observe_discards(hidden, bank)
            if c.stats["resets"] > saved[2]:
                c.hyps, c.size, c.stats["resets"] = saved[0], saved[1], saved[2]
                c._marg = None
                bank = None
                self._warn("the cards discarded on the 7 do not fit the bank on screen (a missed entry, or a misread "
                           "bank): the discards were resolved without it")
        if bank is None:
            for p, k in sorted(hidden.items()):
                c.observe_discard(p, n=k)
        for p in hidden:
            self._mark(p, "a hidden discard")

    def _bank_net(self, events: Sequence[LogEvent]) -> Optional[List[int]]:
        """Public change of the bank over ``events`` (``None`` when one of them is not readable)."""
        d = [0] * 5
        setup = self.setup
        free = list(self.free_roads)
        for ev in events:
            k = ev.kind
            p = self.seat_of(ev.player)
            if k in ("roll", "turn"):
                setup = setup and k != "roll"
                free = [0] * self.n
            elif k in ("gain", "year_of_plenty"):
                if ev.cards is None:
                    return None
                d = [d[r] - ev.cards[r] for r in range(5)]
            elif k == "build":
                if ev.free or setup:
                    continue
                if ev.item == "road" and p is not None and free[p] > 0:
                    free[p] -= 1
                elif ev.item in BUILD_COSTS:
                    d = [d[r] + BUILD_COSTS[ev.item][r] for r in range(5)]
                else:
                    return None
            elif k == "buy_dev":
                d = [d[r] + B.COST_DEV[r] for r in range(5)]
            elif k == "play_dev" and ev.item == "road_building" and p is not None:
                free[p] = 2
            elif k == "bank_trade":
                if ev.cards is None or ev.get is None:
                    return None
                d = [d[r] + ev.cards[r] - ev.get[r] for r in range(5)]
            elif k == "discard":
                if ev.cards is None:
                    return None
                d = [d[r] + ev.cards[r] for r in range(5)]
        return d

    def _monopoly(self, p: int, ev: LogEvent) -> None:
        c = self.counter
        res = ev.resource
        victims = [j for j in range(self.n) if j != p]
        if ev.taken_by:
            want = {j: 0 for j in victims}
            for who, x in ev.taken_by.items():
                v = self._seat(who, ev)
                if v is not None and v != p:
                    want[v] = int(x)
            split = tuple(want[j] for j in victims)
            groups = {split: 1.0}
        else:
            groups: Dict[Tuple[int, ...], float] = {}
            for joint, w in c.hyps.items():
                s = tuple(joint[j][res] for j in victims)
                if ev.count is None or sum(s) == ev.count:
                    groups[s] = groups.get(s, 0.0) + w
        ok = [s for s in groups if any(tuple(j[v][res] for v in victims) == s for j in c.hyps)]
        if not ok:
            # the take matches no hypothesis: the given split, else the total split by the expected
            # shares; the counter restarts from its marginals with it
            if ev.taken_by:
                shares = list(next(iter(groups)))
            else:
                exp = [c.expected[j][res] for j in victims]
                shares = _largest_remainder(int(ev.count or 0), exp, [c.size[j] for j in victims])
            self._warn(f"{self.label(p)}'s Monopoly on {B.RESOURCE_NAMES[res]} took {sum(shares)}, which no hand in "
                       "the count allows: a missed or misread log entry (the count restarts from its estimates)")
            self.stats["repairs"] += 1
            c.observe_monopoly(p, res, taken_by=dict(zip(victims, shares)))
            for j in range(self.n):
                if j != self.me:
                    self._mark(j, "a Monopoly that did not fit the count")
            return
        ranked = sorted(ok, key=lambda s: (-groups[s], s))
        pick = ranked[0]
        if len(ranked) > 1:
            fitting = [s for s in ranked if self._split_fits(p, victims, s)]
            if fitting:
                pick = fitting[0]
            if len(fitting) != 1:
                self._warn(f"{self.label(p)}'s Monopoly: how the {sum(pick)} {B.RESOURCE_NAMES[res]} split between the "
                           "victims is not certain; took the most likely split")
        c.observe_monopoly(p, res, taken_by=dict(zip(victims, pick)))

    def _split_fits(self, p: int, victims: Sequence[int], split: Sequence[int]) -> bool:
        """Would this Monopoly split end the window with the hand sizes on screen?"""
        if self._target is None:
            return False
        d, unknown, _ = self._net(self._rest)
        size = list(self.counter.size)
        for j, x in zip(victims, split):
            size[j] -= x
        size[p] += sum(split)
        return all(size[j] + d[j] == self._target[j] for j in range(self.n) if j not in unknown)

    # --- the screenshot ---------------------------------------------------------------
    def reconcile(self, state: GameState, bank: Optional[Sequence[int]] = None) -> None:
        """Check the count against the screenshot (our hand, hand sizes, the bank) and take the
        development-card counts from it."""
        c = self.counter
        me = self.me
        mp = state.players[me]
        if mp.hand_known:
            want = tuple(int(x) for x in mp.resources)
            if not any(joint[me] == want for joint in c.hyps):
                guess = c.most_likely()[me]
                self._warn(f"your hand on screen ({_counts_text(want)}) differs from the log count "
                           f"({_counts_text(guess)}): a missed or misread log entry; the count takes the screen")
                rep = c._replace
                c._update(lambda joint, w: ((rep(joint, me, want), w),), "own hand")
                c.size[me] = sum(want)
            else:
                c.observe_hand(me, want)
        for j, p in enumerate(state.players):
            if j == me:
                continue
            size = self._screen_size(state, j)
            diff = size - c.size[j]
            if diff:
                self.stats["repairs"] += 1
                self._warn(f"{self.label(j)} holds {size} card(s) on screen but {c.size[j]} by the log count: a missed "
                           f"or misread log entry; {abs(diff)} card(s) of unknown type "
                           + ("added" if diff > 0 else "removed"))
                if diff > 0:
                    self._gain_unknown(j, diff, "a missed log entry")
                else:
                    self._lose_unknown(j, -diff, "a missed log entry")
            if p.hand_known:                  # an opponent's hand given exactly (--fix COLOUR.hand=)
                want = tuple(int(x) for x in p.resources)
                if any(joint[j] == want for joint in c.hyps):
                    c.observe_hand(j, want)
                else:
                    rep = c._replace
                    c._update(lambda joint, w, j=j, want=want: ((rep(joint, j, want), w),), "given hand")
                    c.size[j] = sum(want)
        self.bank_known = bank is not None
        if bank is not None:
            b = [int(x) for x in bank]
            cols = [B.BANK_PER_RESOURCE - x for x in b]
            if any(all(sum(h[r] for h in joint) == cols[r] for r in range(5)) for joint in c.hyps):
                c.observe_bank(b)
            else:
                self._warn("the bank on screen does not match the log count (a missed entry, or a misread bank): "
                           "the bank was not used")
        self._refresh_flags()
        # development cards: counts from the screen, played types from the log (knights also from the screen)
        self.dev_count = [(p.total_dev if p.dev_known else int(p.dev_count)) for p in state.players]
        self.known_dev = [None if (j == me or not p.dev_known) else [p.dev_cards[t] + p.dev_cards_new[t] for t in range(5)]
                          for j, p in enumerate(state.players)]
        self.my_dev = [mp.dev_cards[t] + mp.dev_cards_new[t] for t in range(5)]
        self.played = [list(x) for x in self.log_played]
        for j, p in enumerate(state.players):
            self.played[j][B.DEV_KNIGHT] = max(self.played[j][B.DEV_KNIGHT], int(p.played_knights))
        self.deck = sum(state.dev_deck)

    # --- the bot's views ----------------------------------------------------------------
    def determinize(self, pub: GameState, rng) -> GameState:
        """:meth:`PublicBelief.determinize`, plus the screenshot conventions: our hand from the
        count when the screen did not show it, our unknown development types taken as known (as
        :func:`catanbot.inference.determinize` does) and, unless the bank is visible, the bank as
        what the sampled hands leave of the 19-card decks."""
        s = super().determinize(pub, rng)
        mp = s.players[self.me]
        if not mp.hand_known:
            mp.resources = list(self.counter.most_likely()[self.me])
            mp.hand_known = True
            mp.hand_size = sum(mp.resources)
        if not mp.dev_known:
            mp.dev_known = True
            mp.dev_count = mp.total_dev
        if not self.bank_known:
            s.bank = [max(0, B.BANK_PER_RESOURCE - sum(p.resources[r] for p in s.players)) for r in range(5)]
        return s

    # --- the report -----------------------------------------------------------------------
    def player_summary(self, j: int) -> Dict[str, Any]:
        """``j``'s hand: the certain part (held in every possible hand), the uncertain cards with
        the probability of each resource, the most likely hand."""
        c = self.counter
        marg = c.marginal(j)
        support = [h for h, w in marg.items() if w > 1e-12]
        size = c.size[j]
        certain = [min(h[r] for h in support) for r in range(5)] if support else [0] * 5
        unc = size - sum(certain)
        exp = c.expected[j]
        probs = [max(0.0, exp[r] - certain[r]) / unc for r in range(5)] if unc > 0 else [0.0] * 5
        best, pbest = min(marg.items(), key=lambda kv: (-kv[1], kv[0]))
        return {"size": size, "exact": bool(c.is_exact(j) and not self.estimated[j]),
                "estimated": bool(self.estimated[j]), "certain": certain, "uncertain": unc,
                "uncertain_probs": probs, "most_likely": list(best), "most_likely_p": pbest,
                "expected": [round(x, 3) for x in exp], "reasons": list(self.reasons[j])}

    def report(self) -> Dict[str, Any]:
        """The "Card count" section: a status line, one line per opponent, the warnings."""
        lines: List[str] = []
        c = self.counter
        if self.origin == "start":
            lines.append(f"Counting from the start of the game: {self.entries} log entries ({self.new_entries} new), "
                         f"{c.num_hypotheses} hand hypothesis(es).")
        else:
            lines.append(f"Session started mid-game: opponents' hands began as production-weighted estimates fitted to "
                         f"their hand sizes; exact counting takes over as they spend them. {self.entries} log entries "
                         f"({self.new_entries} new), {c.num_hypotheses} hand hypothesis(es).")
        players: Dict[str, Any] = {}
        for j in range(self.n):
            if j == self.me:
                continue
            s = self.player_summary(j)
            players[self.colors[j]] = {
                "size": s["size"], "exact": s["exact"], "estimated": s["estimated"],
                "certain": _named(s["certain"]), "uncertain": s["uncertain"],
                "uncertain_probs": {B.RESOURCE_NAMES[r]: round(s["uncertain_probs"][r], 3) for r in range(5)
                                    if s["uncertain_probs"][r] > 0},
                "most_likely": _named(s["most_likely"]), "most_likely_p": round(s["most_likely_p"], 3),
                "reasons": s["reasons"],
            }
            lines.append("  " + self._summary_line(j, s))
        for w in self.warnings:
            lines.append("  ! " + w)
        return {"origin": self.origin, "entries": self.entries, "new_entries": self.new_entries,
                "hypotheses": c.num_hypotheses, "players": players, "warnings": list(self.warnings),
                "stats": dict(self.stats), "lines": lines}

    def _summary_line(self, j: int, s: Dict[str, Any]) -> str:
        who = self.label(j)
        size = s["size"]
        if size == 0:
            return f"{who}: no cards"
        probs = sorted(((p, r) for r, p in enumerate(s["uncertain_probs"]) if p >= 0.005), key=lambda x: (-x[0], x[1]))
        dist = " / ".join(f"{B.RESOURCE_NAMES[r]} {p:.0%}" for p, r in probs)
        best = f"most likely {_counts_text(s['most_likely'])} ({s['most_likely_p']:.0%})"
        if s["estimated"]:
            return f"{who}: {_cards(size)}, estimated (the session started mid-game): " + (f"{dist}; " if dist else "") + best
        if s["uncertain"] == 0:
            return f"{who}: {_counts_text(s['certain'])} (exact, {_cards(size)})"
        why = f" from {_reasons_text(s['reasons'])}" if s["reasons"] else ""
        cert = _counts_text(s["certain"]) + " certain" if any(s["certain"]) else "nothing certain"
        return f"{who}: {cert}; {_cards(s['uncertain'])} uncertain{why}: {dist}; {best}"


def _cards(n: int) -> str:
    return f"{n} card" + ("" if n == 1 else "s")


def _reasons_text(reasons: Sequence[str]) -> str:
    """The causes of a hand's uncertainty, steals merged: 'hidden steals with blue, green'."""
    steals = [r[len("a hidden steal with "):] for r in reasons if r.startswith("a hidden steal with ")]
    rest = [r for r in reasons if not r.startswith("a hidden steal with ")]
    parts = []
    if steals:
        parts.append(("a hidden steal with " if len(steals) == 1 else "hidden steals with ") + ", ".join(steals))
    return ", ".join(parts + rest)


def _named(counts: Sequence[int]) -> Dict[str, int]:
    return {B.RESOURCE_NAMES[r]: int(counts[r]) for r in range(5) if counts[r]}


def _largest_remainder(total: int, weights: Sequence[float], caps: Sequence[int]) -> List[int]:
    tot = sum(weights)
    raw = [total * w / tot for w in weights] if tot > 0 else [total / len(weights)] * len(weights)
    out = [min(int(x), caps[i]) for i, x in enumerate(raw)]
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - int(raw[i])))
    guard = 0
    while sum(out) < total and guard < 10 * len(out) + total:
        i = order[guard % len(order)]
        if out[i] < caps[i]:
            out[i] += 1
        guard += 1
    return out


# ---------------------------------------------------------------------------
# Engine actions -> Colonist-style log lines
# ---------------------------------------------------------------------------
class LogRenderer:
    """Renders catanbot engine actions as Colonist-style log lines, as seat ``me`` sees them.

    ``render(pre, action, post)`` diffs the hands around one engine action and returns its log
    lines (the wording of :data:`PHRASES`): hidden information stays hidden - a steal shows its
    card only to the thief and the victim ("You stole ..." / "... from you"), a discard only its
    count unless it is ours (or ``discards_public``).  ``reveal_hidden`` (tests / diagnostics)
    shows every stolen and discarded card: a count fed with it must equal the true hands.
    ``style`` picks the card notation (:data:`STYLES`, or ``"mixed"``: a random one per line,
    from ``rng``).
    """

    STYLES = ("counts", "words", "letters", "colonist")

    def __init__(self, me: int, names: Sequence[str], rng=None, style: str = "counts",
                 discards_public: bool = False, reveal_hidden: bool = False):
        import random as _random
        self.me = int(me)
        self.names = list(names)
        self.rng = rng or _random.Random(0)
        self.style = style
        self.discards_public = bool(discards_public)
        self.reveal_hidden = bool(reveal_hidden)

    def cards(self, counts: Sequence[int]) -> str:
        style = self.rng.choice(self.STYLES) if self.style == "mixed" else self.style
        return format_cards(counts, style)

    def _one_line(self) -> bool:
        return self.style == "mixed" and self.rng.random() < 0.5

    def render(self, pre: GameState, action: Tuple, post: GameState) -> List[str]:
        kind = action[0]
        cur = pre.current
        nm = self.names
        a = nm[cur]
        n = len(pre.players)

        def delta(j):
            return [post.players[j].resources[r] - pre.players[j].resources[r] for r in range(5)]

        if kind == A.SETUP_SETTLEMENT:
            out = [f"{a} placed a Settlement"]
            d = delta(cur)
            if any(d):
                out.append(f"{a} received starting resources {self.cards(d)}")
            return out
        if kind == A.SETUP_ROAD:
            return [f"{a} placed a Road"]
        if kind == A.ROLL:
            v = post.dice
            d1 = self.rng.randint(max(1, v - 6), min(6, v - 1))
            out = [f"{a} rolled {d1} {v - d1}"]
            for j in range(n):
                d = delta(j)
                if any(x > 0 for x in d):
                    out.append(f"{nm[j]} got {self.cards([max(0, x) for x in d])}")
            return out
        if kind == A.DISCARD:
            i = pre.discard_queue[0]
            d = [-x for x in delta(i)]
            if i == self.me or self.discards_public or self.reveal_hidden:
                return [f"{nm[i]} discarded {self.cards(d)}"]
            return [f"{nm[i]} discarded {sum(d)} cards"]
        if kind in (A.MOVE_ROBBER, A.PLAY_KNIGHT):
            out = [f"{a} used Knight"] if kind == A.PLAY_KNIGHT else []
            res, num = pre.hexes[action[1]]
            out.append(f"{a} moved Robber to " + (f"{num} {B.RESOURCE_NAMES[res]}" if num else "desert"))
            v = action[2]
            if v >= 0:
                got = [max(0, x) for x in delta(cur)]
                if any(got):
                    if self.me in (cur, v) or self.reveal_hidden:
                        thief = "You" if cur == self.me else a
                        victim = "you" if v == self.me else nm[v]
                        out.append(f"{thief} stole {self.cards(got)} from {victim}")
                    else:
                        out.append(f"{a} stole a card from {nm[v]}")
            return out
        if kind in (A.BUILD_ROAD, A.BUILD_SETTLEMENT, A.BUILD_CITY):
            what = {A.BUILD_ROAD: "Road", A.BUILD_SETTLEMENT: "Settlement", A.BUILD_CITY: "City"}[kind]
            return [f"{a} built a {what}"]
        if kind == A.BUY_DEV:
            return [f"{a} bought Development Card"]
        if kind == A.PLAY_ROAD_BUILDING:
            return [f"{a} used Road Building"]
        if kind == A.PLAY_YEAR_OF_PLENTY:
            got = delta(cur)
            if self._one_line():
                return [f"{a} used Year of Plenty and took {self.cards(got)}"]
            return [f"{a} used Year of Plenty", f"{a} took from bank {self.cards(got)}"]
        if kind == A.PLAY_MONOPOLY:
            r = action[1]
            total = delta(cur)[r]
            if self._one_line():
                return [f"{a} used Monopoly and stole {total} {B.RESOURCE_NAMES[r]}"]
            return [f"{a} used Monopoly", f"{a} stole {total} {B.RESOURCE_NAMES[r]}"]
        if kind == A.BANK_TRADE:
            d = delta(cur)
            give = [max(0, -x) for x in d]
            get = [max(0, x) for x in d]
            if self._one_line():
                return [f"{a} gave {self.cards(give)} and got {self.cards(get)} from bank"]
            return [f"{a} gave bank {self.cards(give)} and took {self.cards(get)}"]
        if kind == A.PROPOSE_TRADE:
            return [f"{a} wants to give {self.cards(action[1])} for {self.cards(action[2])}"]
        if kind == A.COUNTER_TRADE:
            r = pre.trade_responder
            return [f"{nm[r]} counter-offered {self.cards(action[1])} for {self.cards(action[2])}"]
        if kind == A.ACCEPT_TRADE:
            offer = pre.pending_trade
            if offer is not None and offer.origin is not None and any(delta(offer.proposer)):
                # the player on turn accepted a counter-offer: the trade happens now
                return [f"{nm[offer.proposer]} traded {self.cards(offer.give)} for {self.cards(offer.get)} "
                        f"with {nm[pre.trade_responder]}"]
            return []
        if kind == A.EXECUTE_TRADE:
            offer = pre.pending_trade
            return [f"{nm[offer.proposer]} traded {self.cards(offer.give)} for {self.cards(offer.get)} "
                    f"with {nm[action[1]]}"]
        return []
