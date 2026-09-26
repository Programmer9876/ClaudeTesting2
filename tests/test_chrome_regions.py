"""UI chrome readers pointed at screen regions (``region=`` / ``parse_image(layout=...)``).

Colonist.io's panels are not where the synthetic renderer puts them, so every
chrome reader accepts a pixel box and ``parse_image`` accepts a layout (a
:class:`~catanbot.vision.profile.UiProfile` or a dict of pixel boxes).  These
tests move the synthetic panels to other places and check that the readers
follow the boxes, that ``layout=None`` changes nothing, and that a ``log``
box keeps log-panel clutter out of the board analysis (and says so when it
covers part of the board).  The old-vs-new comparison of ``parse_image``
without a layout on many more screens lives outside the test suite (bench).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from catanbot import board as B
from catanbot.vision import synth
from catanbot.vision.colonist import (DEFAULT_FONT_PATH, MAX_LOG_BOX_FRACTION, Calibration, _blank_box,
                                      _board_under_box, default_ui_regions, draw_debug, layout_boxes, parse_image,
                                      read_bank_panel, read_dice, read_hand_bar, read_player_panel, sea_mask)
from catanbot.vision.profile import ProfileError, UiProfile
from catanbot.vision.schema import state_to_parsed
from tests.test_colonist_parser import played_state, render

SIZE = (1280, 800)
SEA = synth.ColonistStyle().sea


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _state(seed=1, turns=60, current=2):
    s = played_state(seed, 4, turns)
    s.current = current          # a current player other than me: its border must be found too
    return s


def _small_board(size=SIZE):
    """A smaller board in the upper middle: room to move the panels around it."""
    w, h = size
    return {"cx": 0.53 * w, "cy": 0.40 * h, "hex_size": 0.06 * h, "width": float(w), "height": float(h)}


def _render(s, size=SIZE, geom=None):
    img = synth.render_state(s, size=size, seed=0, me=0, jitter=False, geometry=geom or _small_board(size))
    return np.asarray(img)


def _panel_box(n, size=SIZE, m=3):
    """Pixel box around the synthetic player panel (synth._draw_player_panel) plus ``m`` px of sea."""
    w, h = size
    pw, row_h = max(150, 0.21 * w), min(0.10 * h, 0.7 * h / n)
    x0, y0 = 0.012 * w, 0.02 * h
    y1 = y0 + n * row_h + (n - 1) * 0.06 * row_h
    return (int(x0) - m, int(y0) - m, int(math.ceil(x0 + pw)) + m, int(math.ceil(y1)) + m)


def _hand_box(size=SIZE, m=3):
    w, h = size
    return (int(0.293 * w) - m, int(0.835 * h) - m, int(math.ceil(0.707 * w)) + m, min(h, int(math.ceil(0.995 * h)) + m))


def _bank_box(size=SIZE, m=3):
    w, h = size
    pw = max(120, 0.16 * w)
    x0 = w - pw - 0.012 * w
    return (int(x0) - m, int(0.02 * h) - m, int(math.ceil(w - 0.012 * w)) + m, int(math.ceil(0.105 * h)) + m)


def _dice_box(size=SIZE, m=3):
    """The two dice and the total drawn above them (synth._draw_dice)."""
    w, h = size
    d, gap, x1 = 0.06 * h, 0.012 * w, 0.98 * w
    return (int(x1 - 2 * d - gap) - m, int(0.899 * h - 0.02 * h) - m, int(math.ceil(x1)) + m, int(math.ceil(0.98 * h)) + m)


def _move(arr, box, dx, dy):
    """Cut ``box`` out (the hole is painted sea-blue) and paste it ``dx, dy`` pixels away."""
    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1].copy()
    out = arr.copy()
    out[y0:y1, x0:x1] = SEA
    out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = patch
    return out, (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def _board_bbox(geom):
    """Generous bounding box of the board incl. port icons (4.6 x 4.6 hex sizes around the centre)."""
    cx, cy, hs = geom["cx"], geom["cy"], geom["hex_size"]
    return (cx - 4.7 * hs, cy - 4.7 * hs, cx + 4.7 * hs, cy + 4.7 * hs)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _moved_screen():
    """Player panel to the right (lower half), hand bar up, bank down the right side, dice to the top."""
    s = _state()
    arr = _render(s)
    w, h = SIZE
    pbox = _panel_box(len(s.players))
    arr, new_panel = _move(arr, pbox, int(0.754 * w) - pbox[0], int(0.375 * h) - pbox[1])
    hbox = _hand_box()
    arr, new_hand = _move(arr, hbox, 0, -int(0.1375 * h))
    bbox = _bank_box()
    arr, new_bank = _move(arr, bbox, 0, int(0.25 * h))
    dbox = _dice_box()
    arr, new_dice = _move(arr, dbox, 0, int(0.13 * h) - dbox[1])
    boxes = {"player_panel": new_panel, "hand_bar": new_hand, "bank": new_bank, "dice": new_dice}
    board = _board_bbox(_small_board())
    for b in boxes.values():
        assert not _overlaps(b, board), (b, board)          # the moves leave the board intact
        assert 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h
    names = list(boxes)
    assert not any(_overlaps(boxes[a], boxes[b]) for i, a in enumerate(names) for b in names[i + 1:])
    return s, arr, boxes


def _check_rows(rows, s):
    truth = state_to_parsed(s, me=0)
    assert [r["color"] for r in rows] == [p["color"] for p in truth["players"]]
    for r, tp in zip(rows, truth["players"]):
        dev = tp["dev_cards"] if isinstance(tp["dev_cards"], int) else sum(tp["dev_cards"].values())
        assert (r["vp"], r["cards"], r["dev_cards"], r["knights"]) == (tp["vp"], tp["cards"], dev, tp["knights"]), r
    assert [r["current"] for r in rows] == [i == s.current for i in range(len(s.players))]


# ---------------------------------------------------------------------------
# (a) no layout = today's behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,players,size", [(0, 4, (1280, 800)), (2, 3, (800, 500))])
def test_no_layout_is_unchanged(seed, players, size):
    s = played_state(seed, players, 40 + 10 * seed)
    img, _ = render(s, size, seed)
    arr = np.asarray(img)
    base = parse_image(arr, me=s.players[0].color)          # layout=None, the default
    assert base.debug["layout"] == {} and "log_fill" not in base.debug and "log_covers" not in base.debug
    # an empty dict and an empty profile are no layout either (one parse covers both: same boxes)
    assert layout_boxes({}, size) == layout_boxes(UiProfile(), size) == {}
    res = parse_image(arr, me=s.players[0].color, layout=UiProfile())
    assert res.parsed == base.parsed
    assert res.confidence == base.confidence and res.warnings == base.warnings
    assert res.debug["panel_rows"] == base.debug["panel_rows"]
    assert res.debug["layout"] == {}
    # every reader given its default area as an explicit box reads exactly the same
    cal = Calibration()
    boxes = default_ui_regions(size)
    assert read_player_panel(arr, cal) == read_player_panel(arr, cal, region=boxes["player_panel"])
    assert read_hand_bar(arr, cal) == read_hand_bar(arr, cal, region=boxes["hand_bar"])
    assert read_dice(arr, cal) == read_dice(arr, cal, region=boxes["dice"])
    assert read_bank_panel(arr, cal) == read_bank_panel(arr, cal, region=boxes["bank"])
    # ... and so does the whole parse with those boxes as its layout
    res = parse_image(arr, me=s.players[0].color, layout=boxes)
    assert res.parsed == base.parsed
    assert res.debug["layout"] == boxes


# ---------------------------------------------------------------------------
# (b) moved panels are read from their boxes
# ---------------------------------------------------------------------------
def test_readers_follow_moved_regions():
    s, arr, boxes = _moved_screen()
    cal = Calibration()
    me = s.players[0]
    # default areas: the panels are gone from there
    rows, conf, warns = read_player_panel(arr, cal)
    assert rows == [] and any("player panel not found" in w for w in warns)
    assert read_hand_bar(arr, cal)[0] is None
    assert read_bank_panel(arr, cal) == (None, None)
    assert read_dice(arr, cal) == (0, 0.0)
    assert read_dice(arr, cal, region=boxes["dice"]) == (s.dice, 0.9)
    # with the boxes everything is read
    rows, conf, warns = read_player_panel(arr, cal, region=boxes["player_panel"])
    _check_rows(rows, s)
    x0, y0, x1, y1 = boxes["player_panel"]
    assert all(x0 <= r["x"] and r["x"] + r["w"] <= x1 and y0 <= r["y"] and r["y"] + r["h"] <= y1 for r in rows)
    assert conf > 0.5 and not warns
    hand, dev, hconf = read_hand_bar(arr, cal, region=boxes["hand_bar"])
    assert hand == list(me.resources) and dev == me.total_dev
    bank, deck = read_bank_panel(arr, cal, region=boxes["bank"])
    assert bank == {B.RESOURCE_NAMES[r]: s.bank[r] for r in range(5)} and deck == sum(s.dev_deck)
    # a box that misses the panel finds nothing (and says where it looked)
    rows, _, warns = read_player_panel(arr, cal, region=(0, 0, 300, 400))
    assert rows == [] and "(0, 0, 300, 400)" in warns[0]
    # a box outside the image
    assert read_player_panel(arr, cal, region=(5000, 0, 6000, 100))[0] == []
    assert read_hand_bar(arr, cal, region=(5000, 0, 6000, 100)) == (None, None, 0.0)
    assert read_dice(arr, cal, region=(5000, 0, 6000, 100)) == (0, 0.0)
    assert read_bank_panel(arr, cal, region=(5000, 0, 6000, 100)) == (None, None)


def test_readers_word_their_not_found_warnings():
    """Every reader reports a miss itself, the same way (see "Not-found warnings" in colonist.py)."""
    s, arr, boxes = _moved_screen()
    cal = Calibration()
    miss, outside = (0, 0, 300, 400), (5000, 0, 6000, 100)
    for region, where in ((None, ""), (miss, " in region (0, 0, 300, 400)"),
                          (outside, ": region (5000, 0, 6000, 100) lies outside the 1280x800 image")):
        _, _, wp = read_player_panel(arr, cal, region=region)
        assert wp == [f"player panel not found{where}; player list inferred from pieces"]
        wh, wd, wb = [], [], []
        assert read_hand_bar(arr, cal, region=region, warnings=wh)[0] is None
        assert read_dice(arr, cal, region=region, warnings=wd)[0] == 0
        assert read_bank_panel(arr, cal, region=region, warnings=wb)[0] is None
        assert wh == [f"hand bar not found or not fully readable{where}; your resources are unknown "
                      "(use --fix me.hand=...)"]
        # dice and bank are optional panels: silent in their default area, reported inside a given region
        assert wd == ([] if region is None else [f"dice not found{where}; the roll is unknown (read as 0)"])
        assert wb == ([] if region is None else [f"bank panel not found or not fully readable{where}"])
    # found: nothing is reported
    wh, wd, wb = [], [], []
    read_hand_bar(arr, cal, region=boxes["hand_bar"], warnings=wh)
    read_dice(arr, cal, region=boxes["dice"], warnings=wd)
    read_bank_panel(arr, cal, region=boxes["bank"], warnings=wb)
    assert wh == wd == wb == []
    # parse_image forwards the readers' warnings: a dice box that misses the dice is reported
    res = parse_image(arr, me="red", layout=dict(boxes, dice=miss), read_ui=True)
    assert "dice not found in region (0, 0, 300, 400); the roll is unknown (read as 0)" in res.warnings
    # a malformed reader region is an error naming the region
    for bad in ((0, 0, float("inf"), 10), (float("nan"), 0, 10, 10), (1, 2, 3), "0,0,10,10"):
        with pytest.raises(ValueError, match="dice region"):
            read_dice(arr, cal, region=bad)


def test_parse_image_with_layout_reads_moved_panels():
    s, arr, boxes = _moved_screen()
    truth = state_to_parsed(s, me=0)
    plain = parse_image(arr, me="red")
    assert plain.debug["panel_rows"] == []
    assert any("player panel not found" in w for w in plain.warnings)
    assert "bank" not in plain.parsed and "resources" not in plain.parsed["players"][0]
    assert plain.parsed["dice"] == 0
    res = parse_image(arr, me="red", layout=boxes)
    assert res.debug["layout"] == boxes
    _check_rows(res.debug["panel_rows"], s)
    ps = {p["color"]: p for p in res.parsed["players"]}
    for tp in truth["players"]:
        p = ps[tp["color"]]
        assert (p["vp"], p["cards"], p["knights"]) == (tp["vp"], tp["cards"], tp["knights"])
        assert (p["longest_road"], p["largest_army"]) == (tp["longest_road"], tp["largest_army"])
    assert ps["red"]["resources"] == truth["players"][0]["resources"]
    assert res.parsed["bank"] == truth["bank"] and res.parsed["dev_deck_remaining"] == sum(s.dev_deck)
    assert res.parsed["current_player"] == truth["current_player"]
    assert res.parsed["dice"] == s.dice
    assert not any("panel not found" in w or "hand bar not found" in w or "bank panel" in w for w in res.warnings)
    # the board is untouched by the layout
    assert res.parsed["hexes"] == plain.parsed["hexes"] and res.parsed["robber"] == plain.parsed["robber"]
    assert res.parsed["ports"] == plain.parsed["ports"]
    # the debug overlay outlines the regions
    over = np.asarray(draw_debug(arr, res))
    x0, y0, x1, y1 = boxes["bank"]
    assert tuple(over[(y0 + y1) // 2, x0]) == (0, 255, 0)
    # an explicit reader box wins over an overlapping log box (readers with a box see the screen as is)
    px0, py0, px1, py1 = boxes["player_panel"]
    res2 = parse_image(arr, me="red", layout=dict(boxes, log=(px0, py0, px1, (py0 + py1) // 2)))
    _check_rows(res2.debug["panel_rows"], s)
    assert res2.parsed["hexes"] == plain.parsed["hexes"] and res2.parsed["ports"] == plain.parsed["ports"]


# ---------------------------------------------------------------------------
# (c) a log box keeps the log panel out of the board analysis
# ---------------------------------------------------------------------------
def _font(px):
    try:
        return ImageFont.truetype(DEFAULT_FONT_PATH, px)
    except Exception:  # pragma: no cover
        return ImageFont.load_default()


def _add_log_clutter(arr, geom):
    """A log-like panel in the empty sea right of the board, with parser hazards in it.

    * a light-grey highlight stripe along its left edge, next to two coastal
      vertices (looks like white player pieces),
    * a beige "offer" box where the icon of the (portless) east edge 38 would
      sit (looks like a 3:1 port),
    * cream discs with numbers (look like number tokens) and lines of text.
    """
    cx, cy, hs = geom["cx"], geom["cy"], geom["hex_size"]
    w, h = SIZE
    box = (int(cx + 4.36 * hs), int(cy - 1.0 * hs), int(0.988 * w), int(0.78 * h))
    x0, y0, x1, y1 = box
    img = Image.fromarray(arr.copy())
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(252, 252, 253))
    d.rectangle([x0, cy - 0.75 * hs, x0 + 0.22 * hs, cy + 0.75 * hs], fill=(235, 235, 235))
    bx0 = x0 + 0.22 * hs
    d.rectangle([bx0, cy - 0.35 * hs, x0 + 1.4 * hs, cy + 0.35 * hs], fill=(228, 208, 160))
    font = _font(max(10, int(0.3 * hs)))
    d.text((bx0 + 0.75 * hs, cy), "3:1", fill=(40, 44, 54), font=font, anchor="mm")
    d.text((x0 + 1.6 * hs, cy - 0.25 * hs), "offers", fill=(40, 44, 54), font=font)
    r = 0.34 * hs
    for k, num in enumerate((5, 8, 11)):
        tx, ty = x0 + (1.0 + 1.8 * k) * hs, cy + 1.6 * hs
        d.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(242, 228, 196))
        d.text((tx, ty), str(num), fill=(20, 20, 20), font=font, anchor="mm")
    lines = ["red rolled 4 6", "blue got 2 wood", "orange built a road", "green wants 1 ore",
             "red stole a card", "blue bought a Development Card"]
    for i, line in enumerate(lines):
        ty = cy + 2.3 * hs + i * 0.45 * hs
        if ty + 0.4 * hs < y1:
            d.rectangle([x0, ty - 2, x1 - 1, ty + 0.4 * hs], fill=(239, 242, 248) if i % 2 else (252, 252, 253))
            d.text((x0 + 6, ty), line, fill=(40, 44, 54), font=font)
    return np.asarray(img), box


def _dim(a):
    return (a.astype(np.float32) * 0.75).astype(np.uint8)


def _noisy(a):
    rng = np.random.default_rng(1)
    return np.clip(a.astype(np.int16) + rng.normal(0, 9, a.shape), 0, 255).astype(np.uint8)


def _no_log_warning(warnings):
    return [w for w in warnings if not w.startswith("layout log box")]


@pytest.mark.parametrize("variant", ["clean", "dimmed", "noisy"])
def test_log_region_is_excluded_from_the_board(variant):
    s = _state()
    geom = _small_board()
    clean = _render(s)
    cluttered, log_box = _add_log_clutter(clean, geom)
    assert not _overlaps(log_box, (0, 0, SIZE[0], int(0.15 * SIZE[1])))      # clear of the bank's default area
    change = {"clean": lambda a: a, "dimmed": _dim, "noisy": _noisy}[variant]
    base = parse_image(change(clean), me="red")
    truth_ports = {p["edge"] for p in base.parsed["ports"]}
    assert 38 not in truth_ports and len(truth_ports) == 9
    # with the log box nothing of it reaches the board analysis
    good = parse_image(change(cluttered), me="red", layout={"log": log_box})
    assert good.parsed == base.parsed
    assert good.debug["layout"] == {"log": log_box}
    assert good.debug["buildings"] == base.debug["buildings"] and good.debug["roads"] == base.debug["roads"]
    assert good.debug["points"] == base.debug["points"]          # no token look-alikes in the lattice fit
    # the box covers the icon slot of east edge 38, which is not a harbour position on the standard
    # board: nothing is hidden, so no warning (it would push useful warnings out of the live loop)
    assert good.debug["log_covers"] == {"hexes": [], "port_slots": []}
    assert 38 in _board_under_box(good.debug["geometry"], log_box, standard_ports=False)[1]
    assert not [w for w in good.warnings if w.startswith("layout log box")]
    if variant == "noisy":   # the noise level is measured without the log box: its number may differ
        assert [w.split("(noise level")[0] for w in _no_log_warning(good.warnings)] == \
            [w.split("(noise level")[0] for w in base.warnings]
        assert good.debug["noise_level"] > 2.0
    else:
        assert _no_log_warning(good.warnings) == base.warnings
    # the box is painted with the sea as measured on this screen (not the reference colour)
    fill = np.array(good.debug["log_fill"], dtype=float)
    sea_here = np.array(SEA, dtype=float) * (0.75 if variant == "dimmed" else 1.0)
    assert np.abs(fill - sea_here).max() <= 3
    if variant == "dimmed":
        assert np.abs(fill - np.array(SEA)).max() > 10
    if variant != "clean":
        return
    # Without the log box the clutter is read as board (a false port, false white pieces or token
    # look-alikes).  Only a precondition: a parser that learns to ignore it makes this test moot.
    bad = parse_image(cluttered, me="red")
    false_ports = {p["edge"] for p in bad.parsed.get("ports") or []} - truth_ports
    white = [v for v, (c, _, _) in bad.debug["buildings"].items() if c == "white"]
    white += [e for e, (c, _) in bad.debug["roads"].items() if c == "white"]
    if not (false_ports or white or bad.debug["points"] > base.debug["points"]):
        pytest.skip("the log clutter no longer disturbs the parser without a log box")


def test_blank_box_paints_the_measured_sea():
    """``_blank_box``: the box becomes sea in the mask and is filled with the median of the sea outside it."""
    cal = Calibration()
    h, w = 100, 200
    sea_rgb = (30, 90, 150)                          # a sea unlike the reference colour
    assert sea_rgb != tuple(cal.sea)
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:] = sea_rgb
    arr[:, :40] = (200, 180, 120)                    # land
    arr[:5, 40:60] = (0, 0, 255)                     # a few odd sea pixels: the median ignores them
    box = (60, 0, 200, 100)
    arr[:, 60:] = (250, 250, 252)                    # log panel, mostly (wrongly) flagged as sea ...
    sea = np.zeros((h, w), bool)
    sea[:, 40:] = True
    sea[40:45, 60:] = False                          # ... except a line of text
    arr0, sea0 = arr.copy(), sea.copy()
    out, sea_out, fill = _blank_box(arr, sea, box, cal)
    assert fill == sea_rgb                           # the whole-sea median would be the panel colour
    assert (out[:, 60:] == sea_rgb).all() and (out[:, :60] == arr0[:, :60]).all()
    assert sea_out[:, 60:].all() and (sea_out[:, :60] == sea0[:, :60]).all()
    assert (arr == arr0).all() and (sea == sea0).all()           # inputs untouched
    # no sea outside the box: the reference colour
    _, _, fill = _blank_box(arr, np.zeros((h, w), bool), box, cal)
    assert fill == tuple(cal.sea)
    # on a dimmed screen the fill is the dimmed sea, not the reference
    dim = _dim(_render(_state()))
    sea = sea_mask(dim, cal)
    log_box = synth.log_panel_box(SIZE)
    log_box = tuple(int(round(v)) for v in log_box)
    out, sea_out, fill = _blank_box(dim, sea, log_box, cal)
    x0, y0, x1, y1 = log_box
    outside = sea.copy()
    outside[y0:y1, x0:x1] = False
    measured = np.median(dim[outside], axis=0)
    assert np.abs(np.array(fill) - measured).max() <= 1
    assert np.abs(np.array(fill) - np.array(cal.sea)).max() > 10
    assert sea_out[y0:y1, x0:x1].all() and (out[y0:y1, x0:x1] == fill).all()


def test_log_box_over_the_board_is_reported():
    s = _state()
    arr = _render(s)
    g = _small_board()
    cx, cy, hs = g["cx"], g["cy"], g["hex_size"]
    base = parse_image(arr, me="red", read_ui=False)
    # the right part of the board: its left edge 0.43 hex sizes from the nearest column of hex centres
    box = (int(cx + 1.3 * hs), int(cy - 5 * hs), SIZE[0], int(cy + 5 * hs))
    res = parse_image(arr, me="red", read_ui=False, layout={"log": box})
    # the painted box is unknown (not sea) to the lattice ranking: the fit is not pushed one column
    # to the left, away from the "sea" the box became
    fit = res.debug["geometry"]
    assert abs(fit["cx"] - cx) < 0.1 * hs and abs(fit["cy"] - cy) < 0.1 * hs
    hexes = [i for i, (x, _) in enumerate(B.HEX_CENTERS) if cx + x * hs >= box[0]]
    assert len(hexes) == 6
    covers = res.debug["log_covers"]
    assert covers["hexes"] == hexes
    assert (covers["hexes"], covers["port_slots"]) == _board_under_box(res.debug["geometry"], box)
    all_slots = _board_under_box(res.debug["geometry"], box, standard_ports=False)[1]
    assert len(all_slots) >= 5 and set(covers["port_slots"]) < set(all_slots)
    assert covers["port_slots"] and all(e in {e for e, _ in B.STANDARD_PORT_EDGES} for e in covers["port_slots"])
    extra = [w for w in res.warnings if w not in base.warnings]
    warn = [w for w in extra if w.startswith("layout log box")]
    assert len(warn) == 1
    n = len(covers["port_slots"])
    assert f"covers 6 hexes ({', '.join(map(str, hexes))}) and {n} port slot{'s' if n > 1 else ''}" in warn[0]
    assert "shrink the log region" in warn[0]
    # a log box covering most of the screen (here: all of it, clipped) is ignored, with a warning
    big = parse_image(arr, me="red", read_ui=False, layout={"log": (-50, -50, 5000, 5000)})
    assert MAX_LOG_BOX_FRACTION < 1.0
    assert big.debug["layout"] == {} and big.debug["layout_ignored"] == {"log": (0, 0, 1280, 800)}
    assert "log_fill" not in big.debug and "log_covers" not in big.debug
    assert big.warnings[0] == ("layout log box (0, 0, 1280, 800) covers 100% of the 1280x800 screen; ignored "
                               "(the log panel is a column beside the board) - fix the log region")
    assert big.warnings[1:] == base.warnings and big.parsed == base.parsed


# ---------------------------------------------------------------------------
# (d) a UiProfile with fraction regions
# ---------------------------------------------------------------------------
def test_parse_image_accepts_a_ui_profile():
    s, arr, boxes = _moved_screen()
    w, h = SIZE
    prof = UiProfile()
    for name, (x0, y0, x1, y1) in boxes.items():
        prof.set_region(name, (x0 / w, y0 / h, x1 / w, y1 / h))
    assert layout_boxes(prof, SIZE) == boxes
    res = parse_image(arr, me="red", layout=prof)
    assert res.debug["layout"] == {k: prof.pixel_box(k, SIZE) for k in boxes}
    _check_rows(res.debug["panel_rows"], s)
    assert res.parsed["players"][0]["resources"] == {B.RESOURCE_NAMES[r]: s.players[0].resources[r] for r in range(5)}
    assert res.parsed["bank"] == {B.RESOURCE_NAMES[r]: s.bank[r] for r in range(5)}


def test_layout_validation():
    assert layout_boxes(None, SIZE) == {}
    assert layout_boxes({"log": (1000.4, 100, 1300, 700), "dice": None}, SIZE) == {"log": (1000, 100, 1280, 700)}
    with pytest.raises(ValueError, match="unknown layout region"):
        layout_boxes({"panel": (0, 0, 10, 10)}, SIZE)
    for fractions in ((0.8, 0.15, 0.99, 0.78), (0, 0, 1, 1)):
        with pytest.raises(ValueError, match="fractions"):
            layout_boxes({"log": fractions}, SIZE)
    with pytest.raises(ValueError, match="4 numbers"):
        layout_boxes({"log": (1, 2, 3)}, SIZE)
    with pytest.raises(ValueError, match="outside"):
        layout_boxes({"bank": (2000, 0, 2100, 50)}, SIZE)
    with pytest.raises(TypeError):
        layout_boxes([(0, 0, 10, 10)], SIZE)
    # non-finite numbers: the documented ValueError naming the region
    for bad in ((0, 0, math.inf, 500), (math.nan, 0, 100, 100), (0, -math.inf, 100, 100)):
        with pytest.raises(ValueError, match="layout region 'log'.*finite"):
            layout_boxes({"log": bad}, SIZE)
    # text boxes (CLI) must go through profile.parse_box + UiProfile.set_region, never into the dict
    for text in ("1085,120,180,504", "1234"):
        with pytest.raises(ValueError, match="parse_box"):
            layout_boxes({"log": text}, SIZE)


def test_profile_boxes_are_validated_like_dict_boxes():
    prof = UiProfile()
    prof.set_region("log", (0.85, 0.15, 0.99, 0.78))
    prof.set_region("dice", (0.7, 0.8, 1.0, 1.0))
    assert layout_boxes(prof, SIZE) == {"log": (1088, 120, 1267, 624), "dice": (896, 640, 1280, 800)}
    # pixel values assigned directly (skipping set_region) are not fractions: an error naming the region
    prof.regions["log"] = (1085, 120, 1265, 624)
    with pytest.raises(ValueError, match="'log'"):
        layout_boxes(prof, SIZE)
    with pytest.raises(ProfileError):
        layout_boxes(prof, SIZE)

    class Duck:
        """Anything with ``pixel_box``: its boxes go through the same checks as a dict's."""

        def __init__(self, box):
            self.box = box

        def pixel_box(self, name, size):
            return self.box if name == "log" else None

    assert layout_boxes(Duck((-5, 10, 5000, 700)), SIZE) == {"log": (0, 10, 1280, 700)}
    with pytest.raises(ValueError, match="'log'.*outside"):
        layout_boxes(Duck((1388800, 96000, 1280, 800)), SIZE)
    with pytest.raises(ValueError, match="'log'.*finite"):
        layout_boxes(Duck((0, 0, math.nan, 10)), SIZE)
    with pytest.raises(ValueError, match="'log'.*4 numbers"):
        layout_boxes(Duck((0, 0, 10)), SIZE)


def test_log_box_that_breaks_the_board_fit_is_named():
    # a log box over most of the board breaks the lattice fit itself: the covered-hex check then sees
    # a wrong geometry, so the fallback (or the failure) must name the log box
    arr = _render(_state())
    g = _small_board()
    cx, cy, hs = g["cx"], g["cy"], g["hex_size"]
    box = (int(cx - 4.4 * hs), int(cy - 4.1 * hs), int(cx + 3.0 * hs), int(cy + 4.1 * hs))
    try:
        res = parse_image(arr, me="red", read_ui=False, layout={"log": box})
    except ValueError as ex:
        assert f"a layout log box {box} is set" in str(ex)
    else:
        assert any(str(box) in w and ("fell back" in w or "covers" in w) for w in res.warnings), res.warnings
