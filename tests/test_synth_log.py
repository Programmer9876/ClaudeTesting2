"""Tests for the synthetic Colonist.io game-log panel of catanbot.vision.synth.

The panel is the ground truth of the local log OCR: every drawn entry carries its canonical text
(names -> colour words, card icons -> ``N res``, card backs -> ``a card`` / ``N cards``, the dev card
icon -> ``Development Card``, dice -> ``d1 d2``), which must parse to the same event as the line it
was drawn from.  The panel is opt-in and must not disturb the default render or the CV parser.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest

from catanbot import board as B
from catanbot import colonist_log as L
from catanbot.vision import synth
from catanbot.vision.synth import LogPanelStyle, LogToken

from tests.test_colonist_log import play
from tests.test_colonist_parser import played_state
from tests.test_schema import make_midgame_state

NAMES = {"Alice": "red", "Bob": "blue", "Carol": "orange", "Dave": "green"}
FONT_DIR = "/usr/share/fonts/truetype"


def event_signature(ev, name_to_colour=None):
    """The tracker-style key of an event with players as colour words ("you" for the viewer)."""
    if ev is None:
        return None
    names = {re.sub(r"\s+", " ", k.strip().lower()): v for k, v in (name_to_colour or {}).items()}
    stripped = {re.sub(r"[^\w#]+", "", k): v for k, v in names.items()}

    def who(x):
        # as the tracker's seat_of: exact name / colour, then without punctuation ("Alice:" -> alice)
        if x is None:
            return None
        k = re.sub(r"\s+", " ", x.strip().lower())
        if k in names:
            return names[k]
        k2 = re.sub(r"[^\w#]+", "", k)
        if k2 in ("you", "me", "yourself", "your"):
            return "you"
        return stripped.get(k2, k2)

    def cs(v):
        return None if v is None else tuple(int(x) for x in v)

    return (ev.kind, who(ev.player), who(ev.other), cs(ev.cards), cs(ev.get), ev.count, ev.value, ev.item,
            ev.resource, bool(ev.free))


def check_canonical(line, players, icons):
    names = synth._player_names(players)
    canon = synth.canonical_log_text(line, players, icons)
    ev0, ev1 = L.parse_log_line(line), L.parse_log_line(canon)
    assert event_signature(ev1) == event_signature(ev0, names), (line, canon)
    assert bool(ev1.problem) == bool(ev0.problem), (line, canon, ev0.problem, ev1.problem)
    for name, colour in names.items():
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", line):
            assert re.search(r"(?<!\w)" + re.escape(colour) + r"(?!\w)", canon), (line, canon)
    return canon


# ---------------------------------------------------------------------------
# canonical text == the same event
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("style", ["counts", "words", "colonist"])
@pytest.mark.parametrize("seed", [1, 2])
def test_canonical_text_of_whole_games_parses_to_the_same_events(seed, style):
    lines = None
    state = None
    for pre, a, s, ls in play(seed, style=style):
        lines, state = ls, s
    assert len(lines) > 300
    kinds = set()
    for line in lines:
        ev = L.parse_log_line(line)
        assert ev.kind != "unknown", line
        kinds.add(ev.kind)
        c_icons = check_canonical(line, state.players, True)
        c_text = check_canonical(line, state.players, False)
        # text-only mode: only the names change
        assert c_text == synth._canonical_from_tokens(synth.log_line_tokens(line, NAMES, False))
        toks = synth.log_line_tokens(line, state.players, True)
        icons = [t.text for t in toks if t.kind == "icon"]
        if ev.kind == "roll":
            assert len(icons) == 2 and all(i.startswith("die") for i in icons)
        if ev.kind == "robber":
            assert not icons and c_icons == c_text
        if ev.kind in ("gain", "bank_trade", "player_trade", "offer", "year_of_plenty"):
            assert icons and all(i in synth.LOG_RESOURCE_ICONS for i in icons), line
        if ev.kind == "buy_dev":
            assert icons == ["dev"]
    assert {"roll", "gain", "build", "robber", "steal", "offer", "player_trade", "bank_trade", "buy_dev"} <= kinds


HAND_MADE = [
    "Bob counter-offered 1 ore for 2 wood",
    "Bob counter-offered to Alice: 1 ore for 2 wood",
    "Carol countered 2 sheep, 1 wheat for 1 brick",
    "Dave made a counter-offer to Bob: wool wool for grain",
    "Alice wants to give 1 wood for 1 ore",
    "Alice offered lumber lumber for ore",
    "Carol used Monopoly and stole 5 wheat",
    "Carol used Monopoly and stole 3 grain",
    "Carol used Monopoly",
    "Carol stole 4 ore",
    "Carol stole 11 wool",
    "Dave used Year of Plenty and took 1 brick, 1 ore",
    "Dave used Year of Plenty and took wood wood",
    "Dave used Year of Plenty",
    "Dave took from bank 2 sheep",
    "Bob discarded 4 cards",
    "Bob discarded 10 cards",
    "You discarded 2 wood, 2 ore",
    "Bob discarded 9 wood, 2 ore",
    "Alice discarded 1 wood, 1 brick, 1 sheep, 1 wheat",
    "You stole 1 wood from Bob",
    "You stole ore from Bob",
    "Bob stole wood from you",
    "Bob stole a card from Carol",
    "Carol stole 1 lumber from you",
    "Bob moved Robber to 9 wheat",
    "Bob moved Robber to desert",
    "Bob used Knight",
    "Alice gave bank 4 sheep and took 1 ore",
    "Alice gave 2 wheat and got 1 brick from bank",
    "Alice traded 3 wool for 1 ore with the bank",
    "Alice traded 1 wood for 1 ore with Bob",
    "Carol traded grain grain for brick with Dave",
    "Alice rolled 6 6",
    "Alice rolled 12",
    "Alice rolled 5+3",
    "Bob got 2 wood, 1 ore",
    "Bob got WWB",
    "Bob got 2 cards",
    "Alice received starting resources wood brick ore",
    "Bob bought Development Card",
    "Bob bought a development card",
    "Alice built a City",
    "Alice placed a Settlement",
    "Bob received Largest Army",
    "Bob won the game!",
    "- Bob got 2 wood.",
]


@pytest.mark.parametrize("icons", [True, False])
@pytest.mark.parametrize("line", HAND_MADE)
def test_canonical_text_of_hand_made_lines(line, icons):
    check_canonical(line, NAMES, icons)


def test_tokens_and_canonical_details():
    t = synth.log_line_tokens("Alice rolled 5 3", NAMES)
    assert t == [LogToken("name", "Alice", "red", False), LogToken("text", "rolled"),
                 LogToken("icon", "die5"), LogToken("icon", "die3", None, False)]
    t = synth.log_line_tokens("Bob got 2 wood, 1 ore", NAMES)
    assert [x.text for x in t if x.kind == "icon"] == ["wood", "wood", "ore"]
    assert [x.space for x in t if x.kind == "icon"] == [True, False, True]    # runs split by a space
    assert synth.canonical_log_text("Bob got lumber lumber grain", NAMES) == "blue got 2 wood, 1 wheat"
    # a monopoly: the number stays text in front of one icon; long runs too
    t = synth.log_line_tokens("Carol stole 4 ore", NAMES)
    assert [(x.kind, x.text) for x in t[1:]] == [("text", "stole"), ("text", "4"), ("icon", "ore")]
    assert [(x.kind, x.text) for x in synth.log_line_tokens("Bob discarded 10 cards", NAMES)[2:]] == [
        ("text", "10"), ("icon", "card")]
    assert synth.canonical_log_text("Bob discarded 10 cards", NAMES) == "blue discarded 10 cards"
    assert synth.canonical_log_text("Bob discarded 9 wood, 2 ore", NAMES) == "blue discarded 9 wood, 2 ore"
    assert synth.canonical_log_text("Bob stole a card from Carol", NAMES) == "blue stole a card from orange"
    assert synth.canonical_log_text("You stole ore from Bob", NAMES) == "You stole 1 ore from blue"
    assert synth.canonical_log_text("Bob bought Development Card", NAMES) == "blue bought Development Card"
    # never icons: the robber line, the letters notation, a single dice total
    for line in ("Bob moved Robber to 9 wheat", "Bob got WWB", "Alice rolled 12"):
        assert not [x for x in synth.log_line_tokens(line, NAMES) if x.kind == "icon"], line
    # names: longest first, whole words, punctuation glued, "You" stays text
    t = synth.log_line_tokens("Bob Jr traded 1 wood for 1 ore with Bob:", {"Bob": "blue", "Bob Jr": "red"})
    assert (t[0].kind, t[0].text, t[0].colour) == ("name", "Bob Jr", "red")
    assert (t[-2].kind, t[-2].colour, t[-1].text, t[-1].space) == ("name", "blue", ":", False)
    assert synth.log_line_tokens("You stole ore from Bob", NAMES)[0] == LogToken("text", "You", None, False)
    # players as state.players: an unnamed player is known by its colour
    st = make_midgame_state()
    st.players[1].name = ""
    assert synth.canonical_log_text("blue got 1 ore", st.players) == "blue got 1 ore"
    assert synth.log_line_tokens("blue got 1 ore", st.players)[0].kind == "name"
    assert synth.log_line_tokens("", NAMES) == []


# players whose names are phrase words, card words or numbers: a name only where the line names a player
@pytest.mark.parametrize("icons", [True, False])
@pytest.mark.parametrize("players,line,canonical", [
    ({"bank": "red", "Bob": "blue"}, "bank gave bank 4 sheep and took 1 ore", "red gave bank 4 sheep and took 1 ore"),
    ({"bank": "red", "Bob": "blue"}, "Bob traded 1 wood for 1 ore with bank", "blue traded 1 wood for 1 ore with bank"),
    ({"bank": "red", "Bob": "blue"}, "bank stole a card from Bob", "red stole a card from blue"),
    ({"Robber": "red", "Bob": "blue"}, "Robber moved Robber to 9 wheat", "red moved Robber to 9 wheat"),
    ({"Robber": "red", "Bob": "blue"}, "Bob moved Robber to desert", "blue moved Robber to desert"),
    ({"Robber": "red", "Bob": "blue"}, "Robber stole a card from Bob", "red stole a card from blue"),
    ({"7": "red", "Bob": "blue"}, "Bob stole 7 wood", "blue stole 7 wood"),
    ({"7": "red", "Bob": "blue"}, "7 stole 7 wood", "red stole 7 wood"),
    ({"7": "red", "Bob": "blue"}, "Bob moved Robber to 7 wood", "blue moved Robber to 7 wood"),
    ({"7": "red", "Bob": "blue"}, "Bob discarded 7 cards", "blue discarded 7 cards"),
    ({"7": "red", "Bob": "blue"}, "Bob rolled 7", "blue rolled 7"),
    ({"7": "red", "Bob": "blue"}, "7 rolled 3 4", "red rolled 3 4"),
    ({"wood": "red", "Bob": "blue"}, "wood got 2 wood, 1 ore", "red got 2 wood, 1 ore"),
    ({"wood": "red", "Bob": "blue"}, "Bob stole 1 wood from wood", "blue stole 1 wood from red"),
    ({"wood": "red", "Bob": "blue"}, "Bob traded 1 wood for 1 ore with wood", "blue traded 1 wood for 1 ore with red"),
])
def test_names_come_from_the_phrase_player_slots(players, line, canonical, icons):
    got = synth.canonical_log_text(line, players, icons)
    assert got == canonical
    ev0, ev1 = L.parse_log_line(line), L.parse_log_line(got)
    assert ev1.kind != "unknown" and event_signature(ev1) == event_signature(ev0, players)
    names = [t for t in synth.log_line_tokens(line, players, icons) if t.kind == "name"]
    assert len(names) == len(re.findall(r"\b(?:red|blue)\b", canonical))


def test_names_match_case_insensitively_and_exact_case_wins():
    one = {"Bob": "blue", "Alice": "red"}
    assert synth.canonical_log_text("BOB got 1 ore", one) == "blue got 1 ore"
    assert synth.log_line_tokens("BOB got 1 ore", one)[0] == LogToken("name", "BOB", "blue", False)  # as written
    assert synth.canonical_log_text("alice stole a card from bob", one) == "red stole a card from blue"
    two = {"Bob": "blue", "BOB": "red"}        # players differing only in case
    assert synth.canonical_log_text("BOB got 1 ore", two) == "red got 1 ore"
    assert synth.canonical_log_text("Bob got 1 ore", two) == "blue got 1 ore"
    assert synth.canonical_log_text("Bob stole a card from BOB", two) == "blue stole a card from red"
    assert synth.canonical_log_text("bob got 1 ore", two) == "blue got 1 ore"     # ambiguous: the first player
    # a phrase without a player slot (and an unknown line) is searched whole, whole words only
    assert synth.canonical_log_text("BOB received Largest Army", one) == "blue received Largest Army"
    assert synth.canonical_log_text("Bobby received Largest Army", one) == "Bobby received Largest Army"
    assert synth.canonical_log_text("hello bob and Alice", one) == "hello blue and red"


@pytest.mark.parametrize("line", ["Bob discarded 9 wood, 2 ore", "Bob discarded 10 cards", "Carol stole 11 wool",
                                  "Carol used Monopoly and stole 5 wheat", "Bob got 8 ore, 1 wood"])
def test_multiplier_numbers_are_glued_to_their_icon(line):
    toks = synth.log_line_tokens(line, NAMES)
    mult = [k for k in range(len(toks) - 1)
            if toks[k].kind == "text" and toks[k].text.isdigit() and toks[k + 1].kind == "icon"]
    assert mult and all(not toks[k + 1].space for k in mult)
    style = LogPanelStyle(font_scale=0.03)
    size = (1280, 800)
    rows_seen = set()
    for width in range(120, 330, 2):
        box = (1270 - width, 150, 1270, 600)
        e = synth.layout_log_panel([line], NAMES, size, style, box)[0]
        mt = synth._log_metrics(size, style, box, 2)
        rows_seen.add(e.row_count)
        for k in mult:
            (ra, a), (rb, b) = e.token_boxes[k], e.token_boxes[k + 1]
            assert ra == rb and b[0] - a[2] == pytest.approx(0.18 * mt.px), (width, line)   # same row, thin gap
        assert e.canonical == synth.canonical_log_text(line, NAMES)
    assert len(rows_seen) >= 2          # the widths do wrap the line


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [(1000, 640), (1280, 800), (1366, 768), (1920, 1080), (2560, 1440)])
def test_log_panel_box(size):
    w, h = size
    x0, y0, x1, y1 = synth.log_panel_box(size)
    assert x1 == pytest.approx(w - 0.012 * w) and x1 - x0 == pytest.approx(max(120, 0.14 * w))
    assert (y0, y1) == (pytest.approx(0.15 * h), pytest.approx(0.78 * h))


def _game_lines(seed=1, style="counts", n=None):
    out = None
    for pre, a, s, ls in play(seed, style=style):
        out = (s, list(ls))
        if n is not None and len(ls) >= n:
            break
    return out


@pytest.fixture(scope="module")
def game():
    return _game_lines(1, "counts", 260)


@pytest.mark.parametrize("size", [(1000, 640), (1280, 800), (1920, 1080)])
def test_layout_bottom_aligned_wrapped_and_dropped_from_the_top(game, size):
    s, lines = game
    style = LogPanelStyle()
    entries = synth.layout_log_panel(lines, s.players, size, style, first_index=100)
    mt = synth._log_metrics(size, style, None, 2)
    ix0, iy0, ix1, iy1 = mt.inner
    row_h = style.row_height(size[1])
    assert 5 < len(entries) < len(lines)
    # a suffix of the log, oldest first, absolute indices, stripes by parity
    assert [e.index for e in entries] == list(range(100 + len(lines) - len(entries), 100 + len(lines)))
    assert [e.text for e in entries] == lines[-len(entries):]
    assert all(e.stripe == (e.index % 2 == 1) and not e.partial for e in entries)
    # stacked without gaps, newest flush with the bottom, all inside the panel
    assert entries[-1].rows[-1][3] == iy1
    for a, b in zip(entries, entries[1:]):
        assert a.rows[-1][3] == b.rows[0][1]
    # the next older entry did not fit: all its rows (it may wrap) would cross the top edge
    older = lines[len(lines) - len(entries) - 1]
    older_rows = synth.layout_log_panel([older], s.players, size, style)[0].row_count
    assert entries[0].rows[0][1] >= iy0 and entries[0].rows[0][1] - older_rows * row_h < iy0
    wrapped = 0
    for e in entries:
        assert len(e.rows) == e.row_count and all(r[3] - r[1] == row_h for r in e.rows)
        assert e.canonical == synth.canonical_log_text(e.text, s.players)
        for k, (r, (x0, y0, x1, y1)) in enumerate(e.token_boxes):
            assert e.rows[r][1] == y0 and e.rows[r][3] == y1
            assert x0 >= ix0 + mt.pad_x - 1e-6
            if r:
                assert x0 >= ix0 + mt.pad_x + mt.px - 1e-6        # continuation rows indented by 1 em
            if k and e.token_boxes[k - 1][0] == r:
                assert x0 >= e.token_boxes[k - 1][1][2] - 1e-6   # left to right, no overlap
            assert x1 <= ix1 - mt.pad_x + 1e-6 or len([b for b in e.token_boxes if b[0] == r]) == 1
        if e.row_count > 1:
            wrapped += 1
            assert e.token_boxes[0][0] == 0 and e.token_boxes[-1][0] == e.row_count - 1
    assert wrapped > 0


@pytest.mark.parametrize("cut", [0.3, 0.6, 0.9])
def test_layout_cut_top(game, cut):
    s, lines = game
    size = (1280, 800)
    style = LogPanelStyle()
    entries = synth.layout_log_panel(lines, s.players, size, style, cut_top=cut)
    mt = synth._log_metrics(size, style, None, 2)
    row_h = style.row_height(size[1])
    first = entries[0]
    assert first.partial and not any(e.partial for e in entries[1:])
    # the top-most visible row is cut by the panel edge by ~cut of its height
    top_row = entries[0].rows[0]
    assert top_row[1] == mt.inner[1]
    hidden = row_h - (top_row[3] - top_row[1])
    assert abs(hidden - cut * row_h) <= 0.5 + 1e-9
    assert first.row_count >= len(first.rows)
    # the whole-entry layout (no cut) shows the same entries except the partial one
    whole = synth.layout_log_panel(lines, s.players, size, style)
    assert [e.index for e in whole][-len(entries) + 1:] == [e.index for e in entries[1:]]


def test_layout_font_scale_icons_off_and_held_out_fonts(game):
    s, lines = game
    size = (1280, 800)
    small = synth.layout_log_panel(lines, s.players, size, LogPanelStyle(font_scale=0.013))
    base = synth.layout_log_panel(lines, s.players, size)
    assert len(small) > len(base)
    text_only = synth.layout_log_panel(lines, s.players, size, LogPanelStyle(icons=False))
    assert all(not any(t.kind == "icon" for t in e.tokens) for e in text_only)
    assert all(e.canonical == synth.canonical_log_text(e.text, s.players, False) for e in text_only)
    for reg, bold in (("liberation/LiberationSans-Regular.ttf", "liberation/LiberationSans-Bold.ttf"),
                      ("freefont/FreeSans.ttf", "freefont/FreeSansBold.ttf")):
        if not os.path.exists(os.path.join(FONT_DIR, reg)):
            continue
        st = LogPanelStyle(font_path=os.path.join(FONT_DIR, reg), name_font_path=os.path.join(FONT_DIR, bold))
        ents = synth.layout_log_panel(lines, s.players, size, st)
        assert ents and [e.canonical for e in ents] == [synth.canonical_log_text(e.text, s.players) for e in ents]
        img = synth.render_state(s, size=size, seed=1, me=0, jitter=False, log=lines, log_style=st)
        assert img.size == size


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _outside_panel_equal(a, b, box, margin):
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask = np.ones(a.shape[:2], bool)
    mask[max(0, y0 - margin): y1 + margin, max(0, x0 - margin): x1 + margin] = False
    return np.array_equal(a[mask], b[mask])


def test_default_render_is_unchanged_and_the_panel_uses_no_render_randomness(game):
    s, lines = game
    for kw in (dict(jitter=False), dict(jitter=True), dict(jitter=True, jpeg=True), dict(jitter=True, jpeg=False)):
        for seed in (0, 4):
            plain = synth.render_state(s, size=(1280, 800), seed=seed, me=0, **kw)
            none = synth.render_state(s, size=(1280, 800), seed=seed, me=0, log=None, log_style=LogPanelStyle.dark(),
                                      log_cut_top=0.5, log_first_index=3, **kw)
            assert plain.tobytes() == none.tobytes()
            with_log = synth.render_state(s, size=(1280, 800), seed=seed, me=0, log=lines, **kw)
            box = synth.log_panel_box((1280, 800))
            # dice faces, colours and the JPEG choice / quality are the same: only the panel differs
            assert _outside_panel_equal(plain, with_log, box, 24 if kw.get("jpeg", kw["jitter"]) else 3)
            assert not np.array_equal(np.asarray(plain), np.asarray(with_log))


def test_render_matches_layout_colours(game):
    s, lines = game
    size = (1280, 800)
    for style in (LogPanelStyle(), LogPanelStyle.dark()):
        img = synth.render_state(s, size=size, seed=2, me=0, jitter=False, jpeg=False, log=lines, log_style=style)
        arr = np.asarray(img).astype(int)
        entries = synth.layout_log_panel(lines, s.players, size, style)
        mt = synth._log_metrics(size, style, None, 2)
        seen_icons = set()
        for e in entries:
            # the right end of every row shows the entry's background (stripe or plain)
            want = style.stripe if e.stripe else style.background
            for r in e.rows:
                y = int((r[1] + r[3]) / 2)
                x = int(r[2]) - 2
                assert np.abs(arr[y, x] - np.array(want)).sum() <= 6, (e.text, arr[y, x], want)
            for tok, (row, (x0, y0, x1, y1)) in zip(e.tokens, e.token_boxes):
                if tok.kind == "icon" and not tok.text.startswith("die"):
                    # the card colour just inside its left edge, away from the pictogram
                    cy = int((y0 + y1) / 2 + 0.3 * mt.card_h)
                    px = arr[cy, int(x0 + 0.2 * mt.card_w + 0.5)]
                    assert np.abs(px - np.array(style.icon_colours[tok.text])).sum() < 40, (tok, px)
                    seen_icons.add(tok.text)
                elif tok.kind == "name":
                    patch = arr[int(y0):int(y1), int(x0):int(x1) + 1].reshape(-1, 3)
                    d = np.abs(patch - np.array(style.name_rgb(tok.colour))).sum(axis=1)
                    assert d.min() < 40, (tok, style.theme)
        assert {"wood", "card", "dev"} & seen_icons
        # the panel stays between the bank and the dice / hand bar
        x0, y0, x1, y1 = mt.box
        assert y0 >= 0.15 * size[1] - 0.5 and y1 <= 0.78 * size[1] + 0.5


def test_dice_icons_stay_below_the_dice_readers_minimum_area():
    for size in ((1000, 640), (1280, 800), (1920, 1080), (2560, 1440), (1280, 1600)):
        for scale in (0.0175, 0.03):
            mt = synth._log_metrics(size, LogPanelStyle(font_scale=scale), None, 2)
            assert mt.die ** 2 < 0.0005 * size[0] * size[1]


def _l1(a, b):
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))


# The board / chrome parser's colour hazards (catanbot/vision/colonist.py, L1 distances):
#   navy  (30, 40, 56)    panel colour: a navy area within L1 60 that touches the bank panel merges with it
#   beige (228, 208, 160) port colour: a beige area within L1 80 over a port position reads as a 3:1 port
#   cream (242, 228, 196) token colour: blobs within L1 75 with fill 0.5-0.85 and aspect 0.8-1.25 are tokens
#   grey  (235, 235, 235) the white player: pixels nearer to it than to any background colour (and within
#                         Euclid 40) at a board vertex / edge read as a white piece
NAVY, BEIGE, CREAM, GREY = (30, 40, 56), (228, 208, 160), (242, 228, 196), (235, 235, 235)
TOLERANCE = {NAVY: 60, BEIGE: 80, CREAM: 75, GREY: 30}
# Documented margins of the panel colours that come within a tolerance (theme, attribute, hazard, minimum
# L1 asserted, why it is safe anyway); every other panel area colour must clear the tolerance.
PANEL_COLOUR_EXCEPTIONS = {
    ("light", "stripe", CREAM): (65, "shape: a full-width band (aspect > 1.5, asserted below), never a round blob"),
    ("light", "stripe", GREY): (20, "place: the panel never covers a board vertex / edge (asserted below)"),
    ("dark", "background", NAVY): (10, "place: the panel starts below the bank and ends above the hand bar"),
    ("dark", "stripe", NAVY): (35, "place: as the dark background"),
}


def test_panel_colours_avoid_parser_hazards():
    for theme, st in (("light", LogPanelStyle()), ("dark", LogPanelStyle.dark())):
        for attr in ("background", "stripe", "border"):
            for hazard, tol in TOLERANCE.items():
                d = _l1(getattr(st, attr), hazard)
                floor, _why = PANEL_COLOUR_EXCEPTIONS.get((theme, attr, hazard), (tol + 1, "colour"))
                assert d >= floor, (theme, attr, hazard, d)
    # the stripe is nearer to the white player's grey than to white: only its place keeps it off the pieces
    st = LogPanelStyle()
    assert math.dist(st.stripe, GREY) < math.dist(st.stripe, (255, 255, 255))
    # card colours: apart from each other; a fill may come as close as L1 50 to a name colour (dev card vs
    # the purple name, card back vs brown), which is why the icons carry marks (see LogPanelStyle)
    cards = {k: st.icon_colours[k] for k in synth.LOG_RESOURCE_ICONS + ("card", "dev")}
    for a in cards:
        for b in cards:
            if a < b:
                assert _l1(cards[a], cards[b]) > 60, (a, b)
        for name, rgb in st.name_colours.items():
            assert _l1(cards[a], rgb) >= 50, (a, name)
    # the dev card's border is a colour no name, card, panel or text has (in both themes)
    for sty in (LogPanelStyle(), LogPanelStyle.dark()):
        others = list(sty.name_colours.values()) + list(sty.icon_colours.values()) + [
            sty.background, sty.stripe, sty.border, sty.text, sty.card_back_mark, sty.icon_mark]
        assert min(_l1(sty.dev_border, c) for c in others) > 100
    dark = LogPanelStyle.dark()
    assert dark.name_rgb("white") == (235, 235, 235) and dark.theme == "dark"
    assert _l1(dark.text, dark.name_rgb("white")) > 60


@pytest.mark.parametrize("size", [(1000, 640), (1280, 800), (1366, 768), (1440, 900), (1920, 1080), (2560, 1440)])
def test_log_panel_sits_clear_of_the_board_bank_and_hand_bar(game, size):
    """Why the stripe / dark-theme margins above are safe: where the default panel is."""
    w, h = size
    x0, y0, x1, y1 = synth.log_panel_box(size)
    assert y0 > 0.02 * h + 0.085 * h                  # the bank panel's bottom
    assert y1 < 0.985 * h - 0.13 * h - 0.02 * h       # the hand bar's navy backing
    for seed in range(60):
        g = synth.board_geometry(size, seed, True)
        vx = max(synth.pixel_of_vertex(v, g)[0] for v in range(len(B.VERTEX_POS)))
        assert x0 - vx > 0.3 * g["hex_size"], (seed, x0, vx)    # pieces sit on vertices / edges
    # stripes are wide bands: never within the token blobs' aspect range (0.8 - 1.25)
    s, lines = game
    style = LogPanelStyle()
    mt = synth._log_metrics(size, style, None, 2)
    for e in synth.layout_log_panel(lines, s.players, size, style):
        assert (mt.inner[2] - mt.inner[0]) / (e.row_count * mt.row_h) > 1.5, e.text


def test_dev_card_icon_is_told_from_a_purple_name_by_its_border():
    st = make_midgame_state()
    st.players[2].color, st.players[2].name = "purple", "Carol"
    lines = ["Carol bought Development Card", "Carol got 1 ore", "Carol bought Development Card"]
    size = (1280, 800)
    for style in (LogPanelStyle(), LogPanelStyle.dark()):
        style.icon_colours["dev"] = style.name_rgb("purple")         # worst case: the very same fill
        img = synth.render_state(st, size=size, seed=0, me=0, jitter=False, jpeg=False, log=lines, log_style=style)
        arr = np.asarray(img).astype(int)
        seen = {"dev": 0, "name": 0}
        for e in synth.layout_log_panel(lines, st.players, size, style):
            for tok, (_, (x0, y0, x1, y1)) in zip(e.tokens, e.token_boxes):
                if tok.kind == "name" or (tok.kind == "icon" and tok.text == "dev"):
                    patch = arr[int(y0):int(y1) + 1, int(x0):int(x1) + 1].reshape(-1, 3)
                    near = int((np.abs(patch - np.array(style.dev_border)).sum(axis=1) < 60).sum())
                    if tok.kind == "name":
                        assert near == 0, style.theme
                        seen["name"] += 1
                    else:
                        assert near >= 4, (style.theme, near)
                        seen["dev"] += 1
        assert seen == {"dev": 2, "name": 3}


def test_render_exposes_the_drawn_layout_and_the_supersample_default(game):
    s, lines = game
    size = (1000, 640)
    for ss in (1, 2, 3):
        drawn = []
        synth.render_state(s, size=size, style=synth.ColonistStyle(supersample=ss), seed=1, me=0, jitter=False,
                           log=lines, log_cut_top=0.3, log_first_index=5, log_layout=drawn)
        assert drawn and drawn == synth.layout_log_panel(lines, s.players, size, cut_top=0.3, first_index=5,
                                                         supersample=ss)
    # the default is the default ColonistStyle's scale
    assert synth.layout_log_panel(lines, s.players, size) == synth.layout_log_panel(
        lines, s.players, size, supersample=synth.ColonistStyle().supersample)
    none = []
    synth.render_state(s, size=size, seed=1, me=0, log=None, log_layout=none)
    assert none == []


def test_log_box_is_validated_and_clipped(game):
    s, lines = game
    size = (1280, 800)
    for bad in ((1100, 600, 1000, 100), (1000, 100, 1000, 600), (1300, 100, 1400, 600), (0, -300, 100, -10),
                (1, 2, 3), (float("nan"), 0, 10, 10), ("a", 0, 1, 1)):
        with pytest.raises(ValueError, match="log_box"):
            synth.layout_log_panel(lines, s.players, size, box=bad)
        with pytest.raises(ValueError, match="log_box"):
            synth.render_state(s, size=size, seed=0, me=0, jitter=False, log=lines, log_box=bad)
    with pytest.raises(ValueError, match="inverted"):
        synth.layout_log_panel(lines, s.players, size, box=(1100, 600, 1000, 100))
    # a box reaching past the image edges is clipped to the image
    clipped = synth.layout_log_panel(lines, s.players, size, box=(1100, 100, 1500, 900))
    assert clipped and clipped == synth.layout_log_panel(lines, s.players, size, box=(1100, 100, 1280, 800))
    assert max(r[2] for e in clipped for r in e.rows) <= size[0] and max(r[3] for e in clipped for r in e.rows) <= 800
    img = synth.render_state(s, size=size, seed=0, me=0, jitter=False, log=lines, log_box=(1100, 100, 1500, 900))
    assert img.size == size
    # without a log the box is not looked at (the default render is untouched)
    synth.render_state(s, size=size, seed=0, me=0, log=None, log_box=(1100, 600, 1000, 100))


def test_render_to_file_passes_the_log_through(tmp_path, game):
    s, lines = game
    p = str(tmp_path / "shot.png")
    synth.render_to_file(s, p, size=(1000, 640), seed=3, me=0, log=lines[-10:], log_cut_top=0.3)
    from PIL import Image
    img = Image.open(p).convert("RGB")
    direct = synth.render_state(s, size=(1000, 640), seed=3, me=0, log=lines[-10:], log_cut_top=0.3,
                                geometry=synth.board_geometry((1000, 640), 3, True))
    assert np.array_equal(np.asarray(img), np.asarray(direct))


def test_empty_log_draws_an_empty_panel(game):
    s, _ = game
    img = synth.render_state(s, size=(1280, 800), seed=0, me=0, jitter=False, log=[])
    x0, y0, x1, y1 = synth.log_panel_box((1280, 800))
    arr = np.asarray(img).astype(int)
    assert np.abs(arr[int((y0 + y1) / 2), int((x0 + x1) / 2)] - np.array(LogPanelStyle().background)).sum() <= 3
    assert synth.layout_log_panel([], s.players, (1280, 800)) == []


# ---------------------------------------------------------------------------
# the board / chrome parser does not see the panel
# ---------------------------------------------------------------------------
def _log_for(s, seed, n=40):
    """Log lines naming the players of ``s`` by their colour (played_state has no names)."""
    names = {i: (p.name or p.color) for i, p in enumerate(s.players)}
    out = []
    for pre, a, st, ls in play(seed, style="counts"):
        if len(ls) >= 150:
            out = list(ls[-n:])
            break
    table = dict(zip(["Alice", "Bob", "Carol", "Dave"], [names[i] for i in range(4)]))
    return [re.sub(r"\b(Alice|Bob|Carol|Dave)\b", lambda m: table[m.group(1)], x) for x in out]


def _parse(img):
    from catanbot.vision.colonist import parse_image
    return parse_image(img, me="red").parsed


def _load_harness():
    import importlib.util
    import sys
    name = "eval_logocr_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "eval_logocr.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_ocr_benchmark_harness_oracle_is_perfect_and_mistakes_are_counted(tmp_path):
    import json
    ev = _load_harness()
    out = tmp_path / "summary.json"
    assert ev.main(["--reader", "oracle", "--n", "8", "--json", str(out)]) == 0
    summ = json.loads(out.read_text())
    o = summ["overall"]
    assert summ["samples"] == 8 and summ["failed_samples"] == 0 and o["entries"] > 40
    assert o["exact"] == 1.0 and o["emitted"] == 1.0 and o["false"] == 0 and o["hi_acc"] == 1.0
    assert o["precision"] == 1.0 and o["extra"] == 0 and o["dup"] == 0
    assert o["box_ok"] == 1.0 and o["detect"] == 1.0 and o["partial_ok"] in (1.0, None) and o["warm_same"] == 1.0
    assert o["scroll_pairs"] > 30 and o["scroll"] == 1.0 and o["scroll_fresh"] == 1.0
    assert set(summ["by_condition"]) == set(ev.CONDITIONS)
    # the scoring sees a dropped entry, a misread player, a confident full line for the cut entry and garbage
    games = ev.build_games([1], ["counts"], 6)
    s = next(sm for sm in (ev.make_sample(k, games, 0, ev.FONTS, scroll=False) for k in range(40))
             if sm.truth[0].partial and len(sm.truth) > 6)
    ocr = [(e.canonical, False, 0.95) for e in s.truth]
    del ocr[3]
    ocr[4] = (re.sub(r"^\w+", "pink", ocr[4][0]), False, 0.95)
    ocr.append(("zz garbage", False, 0.2))
    r = ev.score(s, ocr, s.box)
    full = len(s.truth) - 1
    assert r["entries"] == full and r["emitted"] == full - 1 and r["exact"] == full - 2
    assert r["part_viol"] == 1 and r["false"] >= 2 and r["lo"] == 1 and r["lo_ok"] == 0 and r["box_ok"] == 1
    assert r["extra"] == r["false"] + r["dup"] >= 2 and r["tp"] == r["emitted_lines"] - r["extra"]
    assert ev.failed(r)
    # flagged partial (or unconfident) is fine
    ocr2 = [(e.canonical, e.partial, 0.95) for e in s.truth]
    r2 = ev.score(s, ocr2, None)
    assert r2["part_viol"] == 0 and r2["exact"] == full and r2["false"] == 0 and r2["extra"] == 0
    assert r2["tp"] == r2["emitted_lines"] == full and not ev.failed(r2)
    # a duplicated line: every entry is still exact, but the copy is extra (the tracker would count it twice)
    for pos in (len(ocr2), 3):
        ocr3 = list(ocr2)
        ocr3.insert(pos, ocr2[pos - 1])
        r3 = ev.score(s, ocr3, None)
        assert r3["exact"] == full and r3["extra"] == 1 and r3["dup"] == 1 and r3["false"] == 0, pos
        assert r3["tp"] == full and ev.failed(r3)
    # a line repeated out of order is extra too (multiset matching: one entry, one line)
    ocr4 = ocr2 + [ocr2[2]]
    r4 = ev.score(s, ocr4, None)
    assert r4["exact"] == full and r4["extra"] == 1 == r4["dup"]


@pytest.fixture(scope="module")
def bench():
    """A few benchmark samples (with their scroll frames): one with the panel moved, one cut at the top."""
    ev = _load_harness()
    fonts = {k: v for k, v in ev.FONTS.items() if all(os.path.exists(p) for p in v)}
    games = ev.build_games([1, 2], ["counts", "colonist"], 6)
    picks = []
    for want in ({"box": "jittered"}, {"box": "jittered", "icons": "on"}, {"cut": "0.3", "box": "default"}):
        s = next(sm for sm in (ev.make_sample(k, games, 0, fonts, want) for k in range(400)) if sm is not None)
        picks.append(s)
    return ev, games, fonts, picks


def test_ocr_benchmark_each_oracle_fault_fails_its_metric(bench):
    ev, _, _, samples = bench
    for s in samples:
        assert s.prev_img is not None and s.prev_truth
        assert s.prev_truth[-1].index == s.truth[-1].index - 1         # one entry earlier
        for give_box in (False, True):
            r, _ = ev.evaluate_sample(s, ev.make_oracle(), give_box=give_box)
            assert not ev.failed(r) and r["scroll_pairs"] >= 3 and r["det_checked"] and r["box_checked"]
    jittered = samples[:2]

    def results(faults, give_box=False, which=samples):
        return [ev.evaluate_sample(s, ev.make_oracle(faults), give_box=give_box)[0] for s in which]

    for r in results(["dup"]):
        assert r["extra"] == r["dup"] == 1 and r["exact"] == r["entries"] and ev.failed(r)
        assert r["scroll_ok"] == r["scroll_pairs"] and r["warm_same"] and r["box_ok"]
    for r in results(["rekey"]):
        assert r["scroll_ok"] < r["scroll_pairs"] and r["fresh_ok"] < r["fresh_pairs"] and r["extra"] == 0
    for r in results(["cachekey"]):          # stable with one cache, not across fresh caches
        assert r["scroll_ok"] == r["scroll_pairs"] and r["fresh_ok"] < r["fresh_pairs"] and ev.failed(r)
    for give_box in (False, True):           # a moved panel: box and detection both miss, given the box or not
        for r in results(["defaultbox"], give_box, jittered):
            assert r["box_checked"] and not r["box_ok"] and r["det_checked"] and not r["det_ok"] and ev.failed(r)
            assert r["det_iou"] < ev.BOX_IOU and r["exact"] == r["entries"]
    for r in results(["flaky"]):
        assert not r["warm_same"] and r["scroll_ok"] == r["scroll_pairs"] and ev.failed(r)
    with pytest.raises(ValueError, match="unknown oracle fault"):
        ev.make_oracle(["nope"])


def test_ocr_benchmark_conditions(bench):
    ev, games, fonts, samples = bench
    for s in samples[:2]:                    # a jittered panel: inside the safe right column, not the default
        w, h = s.img.size
        x0, y0, x1, y1 = s.box
        assert s.cond["box"] == "jittered" and s.cond["scale"] == "none"
        assert 0.83 * w - 1 <= x0 < x1 <= w and 0.15 * h - 1 <= y0 < y1 <= 0.78 * h + 1
        assert ev._iou(s.box, synth.log_panel_box((w, h))) < 1.0
        assert all(x0 <= e.box[0] and e.box[2] <= x1 and y0 <= e.box[1] and e.box[3] <= y1 for e in s.truth)

    def one(want):
        return next(sm for sm in (ev.make_sample(k, games, 0, fonts, want, scroll=False) for k in range(600))
                    if sm is not None)

    s = one({"scale": "x0.8"})                # labelled by the final size
    assert s.cond["size"] == "1280x800@x0.8" and s.img.size == (1024, 640)
    for px in (9, 28):                        # forced text size: rows of round(1.5 px), every token inside the panel
        s = one({"font_px": str(px)})
        assert s.cond["scale"] == "none" and s.cond["size"] == f"{s.img.size[0]}x{s.img.size[1]}"
        x0, y0, x1, y1 = s.box
        for e in s.truth:
            if not e.partial:
                assert [r[3] - r[1] for r in e.rows] == [round(1.5 * px)] * e.row_count
            assert all(x0 < b[0] and b[2] < x1 for _, b in e.token_boxes), e.text
    s = one({"noise": "noise+blur"})
    assert s.cond["noise"] == "noise+blur" and s.img.size[0] > 0
    assert {c for c in ev.CONDITIONS} >= {"box", "font_px", "noise"}


def test_ocr_benchmark_cli_errors(tmp_path, monkeypatch):
    ev = _load_harness()
    # a missing module / function / oracle fault is reported, not a crash
    assert ev.main(["--reader", "catanbot.vision.no_such_reader", "--n", "1"]) == 2
    assert ev.main(["--reader", "catanbot.vision.synth:no_such_function", "--n", "1"]) == 2
    assert ev.main(["--reader", "oracle:nope", "--n", "1"]) == 2
    # no sample matches --only: an error, not an empty 100 %
    assert ev.main(["--reader", "oracle", "--n", "2", "--only", "size=999x999"]) == 2
    with pytest.raises(SystemExit):
        ev.main(["--reader", "oracle", "--n", "1", "--only", "nonsense=1"])
    # an ImportError raised inside the reader is the reader's bug: it propagates
    (tmp_path / "broken_log_reader.py").write_text(
        "def read_log_panel(img, box=None, profile=None, players=None, cache=None):\n"
        "    import no_such_module_inside_the_reader  # noqa: F401\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match="no_such_module_inside_the_reader"):
        ev.main(["--reader", "broken_log_reader", "--n", "1", "--no-scroll"])


@pytest.mark.parametrize("seed,size,dark", [(3, (1280, 800), False), (7, (1280, 800), False),
                                            (13, (1280, 800), False), (5, (1920, 1080), False),
                                            (11, (1280, 800), True)])
def test_parser_reads_the_same_with_and_without_the_log_panel(seed, size, dark):
    s = played_state(seed)
    geom = synth.board_geometry(size, seed, True)
    log = _log_for(s, seed)
    style = LogPanelStyle.dark() if dark else LogPanelStyle()
    plain = synth.render_state(s, size=size, seed=seed, me=0, jitter=True, geometry=geom)
    shot = synth.render_state(s, size=size, seed=seed, me=0, jitter=True, geometry=geom, log=log, log_style=style,
                              log_cut_top=0.3)
    assert len(synth.layout_log_panel(log, s.players, size, style, cut_top=0.3)) >= 8
    a, b = _parse(plain), _parse(shot)
    for key in ("hexes", "robber", "ports", "players", "me", "current_player", "dice", "bank", "dev_deck_remaining"):
        assert a.get(key) == b.get(key), key
    assert a == b
