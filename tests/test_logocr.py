"""Tests of the local game-log reader (catanbot.vision.logocr) on rendered Colonist-style panels.

The panels come from catanbot.vision.synth (the log OCR's ground truth: every drawn entry carries its
canonical text).  Readings are compared at event level (the tracker's key), as the benchmark
(scripts/eval_logocr.py) does.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
from PIL import Image

from catanbot import colonist_log as L
from catanbot.vision import logocr, synth
from catanbot.vision.profile import UiProfile

from tests.test_colonist_log import NAMES, play

FONT_DIR = "/usr/share/fonts/truetype"
LIB = (os.path.join(FONT_DIR, "liberation/LiberationSans-Regular.ttf"),
       os.path.join(FONT_DIR, "liberation/LiberationSans-Bold.ttf"))


def _game(seed: int = 3, n: int = 60):
    for pre, a, s, ls in play(seed, style="counts"):
        if len(ls) >= n:
            return s.copy(), list(ls)
    return s.copy(), list(ls)


EXTRA = ["Bob counter-offered to Carol: 2 wood, 1 ore for 1 wheat", "Carol used Monopoly and stole 9 ore",
         "Dave discarded 8 cards", "You stole 1 sheep from Bob", "Bob stole a card from Carol",
         "Carol bought Development Card", "Alice wants to give 2 brick for 1 sheep", "Dave rolled 5 3"]


@pytest.fixture(scope="module")
def game():
    state, lines = _game()
    return state, lines[-22:] + EXTRA


def render(state, lines, size=(1280, 800), style=None, cut=0.0, box=None, first=0, seed=1):
    truth = []
    img = synth.render_state(state, size=size, seed=seed, me=0, jpeg=False, log=list(lines), log_box=box,
                             log_first_index=first, log_cut_top=cut, log_style=style or synth.LogPanelStyle(),
                             log_layout=truth)
    return img, truth


def sig(text):
    return logocr.event_signature(L.parse_log_line(text or ""))


def score(res, truth):
    """(right, visible full entries): full truth entries whose event is read, in order (a longest
    common subsequence of the event signatures)."""
    import difflib
    full = [sig(e.canonical) for e in truth if not e.partial]
    got = [sig(ln.text) for ln in res.lines if not ln.partial]
    sm = difflib.SequenceMatcher(None, [repr(s) for s in full], [repr(s) for s in got], autojunk=False)
    right = sum(b.size for b in sm.get_matching_blocks())
    return right, len(full)


# ---------------------------------------------------------------------------
def test_api_shape_and_inputs(game, tmp_path):
    state, lines = game
    img, truth = render(state, lines)
    arr = np.asarray(img)
    path = str(tmp_path / "shot.png")
    img.save(path)
    results = [logocr.read_log_panel(x) for x in (img, arr, path)]
    for res in results:
        assert isinstance(res, logocr.LogReadResult)
        assert res.box is not None and len(res.box) == 4
        assert isinstance(res.warnings, list) and isinstance(res.debug, dict)
        assert 0.0 <= res.confidence <= 1.0
        for ln in res.lines:
            assert isinstance(ln, logocr.LogLine)
            assert isinstance(ln.text, str) and isinstance(ln.key, str) and ln.key
            assert 0.0 <= ln.confidence <= 1.0 and len(ln.box) == 4
            assert ln.event is None or hasattr(ln.event, "kind")
        assert res.texts() == [ln.text for ln in res.lines if ln.confidence >= 0.6 and not ln.partial]
    # the same pixels read the same whatever the input type
    assert [ln.text for ln in results[0].lines] == [ln.text for ln in results[1].lines] == \
        [ln.text for ln in results[2].lines]


@pytest.mark.parametrize("dark,icons", [(False, True), (True, True), (False, False)])
def test_canonical_output_light_dark_icons_off(game, dark, icons):
    state, lines = game
    style = synth.LogPanelStyle.dark(icons=icons) if dark else synth.LogPanelStyle(icons=icons)
    img, truth = render(state, lines, style=style)
    res = logocr.read_log_panel(img)
    right, n = score(res, truth)
    assert n >= 10
    assert right >= n - 1, [(e.canonical, ln.text) for e, ln in zip(truth, res.lines)]
    # confident lines are right
    good = {sig(e.canonical) for e in truth}
    for ln in res.lines:
        if ln.confidence >= 0.6 and not ln.partial:
            assert sig(ln.text) in good, ln.text


def test_wrapped_entries_are_joined(game):
    state, lines = game
    box = synth.log_panel_box((1280, 800))
    narrow = (box[2] - 150, box[1], box[2], box[3])       # a narrow panel wraps most entries
    img, truth = render(state, lines, box=narrow)
    assert any(e.row_count > 1 and not e.partial for e in truth)
    res = logocr.read_log_panel(img, box=narrow)
    right, n = score(res, truth)
    assert right >= n - 1
    wrapped = [e for e in truth if e.row_count > 1 and not e.partial]
    texts = {sig(ln.text) for ln in res.lines if not ln.partial}
    assert sum(sig(e.canonical) in texts for e in wrapped) >= len(wrapped) - 1


@pytest.mark.parametrize("cut", [0.3, 0.6])
def test_cut_top_entry_is_partial(game, cut):
    state, lines = game
    img, truth = render(state, lines, cut=cut)
    assert truth[0].partial
    res = logocr.read_log_panel(img)
    assert res.lines and res.lines[0].partial
    assert res.lines[0].confidence < 0.6
    assert all(not ln.partial for ln in res.lines[1:])
    assert res.lines[0].text not in res.texts()
    right, n = score(res, truth)
    assert right >= n - 1


def test_cache_determinism_and_key_stability(game):
    state, lines = game
    img, truth = render(state, lines)
    cache = logocr.LineCache(maxsize=64)
    a = logocr.read_log_panel(img, cache=cache)
    b = logocr.read_log_panel(img, cache=cache)
    assert [(ln.text, ln.partial, ln.confidence, ln.key) for ln in a.lines] == \
        [(ln.text, ln.partial, ln.confidence, ln.key) for ln in b.lines]
    assert a.box == b.box
    # the previous frame (one entry less, every entry lower): unchanged entries keep text and key
    prev, ptruth = render(state, lines[:-1])
    c2 = logocr.LineCache()
    p = logocr.read_log_panel(prev, cache=c2)
    q = logocr.read_log_panel(img, cache=c2)
    fresh = logocr.read_log_panel(img)
    pk = {ln.key: ln.text for ln in p.lines if not ln.partial}
    qk = {ln.key: ln.text for ln in q.lines if not ln.partial}
    shared = set(pk) & set(qk)
    assert len(shared) >= len(pk) - 2
    assert all(pk[k] == qk[k] for k in shared)
    assert [ln.key for ln in q.lines] == [ln.key for ln in fresh.lines]
    # distinct content, distinct keys
    by_key = {}
    for ln in fresh.lines:
        by_key.setdefault(ln.key, set()).add(ln.text)
    assert all(len(v) == 1 for v in by_key.values())
    # the cache is bounded
    small = logocr.LineCache(maxsize=3)
    logocr.read_log_panel(img, cache=small)
    assert len(small) <= 3


def test_key_independent_of_position_and_stripe(game):
    state, lines = game
    ref = logocr.read_log_panel(render(state, lines)[0])
    b = ref.box
    # the same panel elsewhere on the screen: the same keys
    moved = (b[0] - 37, b[1] + 21, b[2] - 37, b[3] + 21)
    r0 = logocr.read_log_panel(render(state, lines, box=moved)[0])
    k0 = [(ln.text, ln.key) for ln in ref.lines if not ln.partial]
    k1 = [(ln.text, ln.key) for ln in r0.lines if not ln.partial]
    assert sum(1 for a in k0 if a in k1) >= 0.9 * len(k0)
    # every entry's stripe flips: the key is taken relative to the entry's own background, as ink
    # coverage, so many keys survive (not all: the renderer's resampling ringing is clipped
    # differently on each stripe, which moves a few coverage blocks across a level)
    r1 = logocr.read_log_panel(render(state, lines, first=1)[0])
    t1 = [(ln.text, ln.key) for ln in ref.lines if not ln.partial]
    t2 = [(ln.text, ln.key) for ln in r1.lines if not ln.partial]
    n = min(len(t1), len(t2))
    same = sum(1 for a, c in zip(t1[-n:], t2[-n:]) if a[1] == c[1])
    assert same >= 0.3 * n
    assert sum(1 for a, c in zip(t1[-n:], t2[-n:]) if a[0] == c[0]) >= 0.9 * n


@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080), (1366, 768)])
def test_find_log_panel_on_moved_panels(game, size):
    state, lines = game
    rng = random.Random(size[0])
    w, h = size
    for _ in range(2):
        x1 = rng.uniform(0.975, 0.996) * w
        x0 = x1 - rng.uniform(max(150.0, 0.11 * w), 0.16 * w)
        box = (x0, rng.uniform(0.15, 0.35) * h, x1, rng.uniform(0.55, 0.78) * h)
        img, truth = render(state, lines, size=size, box=box)
        tb = synth._log_metrics(size, synth.LogPanelStyle(), box, 2).box
        found = logocr.find_log_panel(img)
        assert found is not None

        def iou(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
            return ix * iy / u
        assert iou(found, tb) >= 0.9
    # a profile region wins over the detection
    prof = UiProfile()
    prof.set_region("log", (0.8, 0.2, 0.99, 0.7))
    assert logocr.find_log_panel(img, profile=prof) == prof.pixel_box("log", img.size)


def test_teach_improves_a_held_out_font_screen(tmp_path):
    if not all(os.path.exists(p) for p in LIB):
        pytest.skip("held-out font not installed")
    state, lines = _game(5, 90)
    style = synth.LogPanelStyle(font_path=LIB[0], name_font_path=LIB[1], font_scale=11.0 / 800)
    img_a, truth_a = render(state, lines[:40][-24:], style=style, seed=2)
    img_b, truth_b = render(state, lines[-24:], style=style, seed=3)
    before, n = score(logocr.read_log_panel(img_b), truth_b)
    prof = UiProfile(path=str(tmp_path / "profile.json"))
    prof, report = logocr.teach(img_a, [e.canonical for e in truth_a], profile=prof)
    assert report["taught_entries"] >= 5 and report["glyphs"] >= 20
    assert prof.region("log") is not None and "glyph_adapt" in prof.ocr
    prof.save()
    loaded = UiProfile.load(str(tmp_path / "profile.json"))
    after, n2 = score(logocr.read_log_panel(img_b, profile=loaded), truth_b)
    assert n2 == n
    assert after >= before
    assert report["read_right_after"] >= report["read_right_before"]


def test_players_palette_and_empty_image():
    blank = Image.new("RGB", (800, 500), (60, 140, 220))
    res = logocr.read_log_panel(blank)
    assert res.lines == [] and res.box is None and res.warnings
    assert logocr.find_log_panel(blank) is None


# ---------------------------------------------------------------------------
# building icons ("built a [road]") and Colonist's colons ("got:")
# ---------------------------------------------------------------------------
BUILD_LINES = ["Alice built a Road", "Bob built a Settlement", "Carol built a City", "Dave placed a Settlement",
               "Bob got 1 wood, 2 ore", "Alice placed a Road", "Carol bought Development Card", "Dave built a City",
               "Bob stole a card from Alice", "Alice built a Settlement", "Carol wants to give 2 brick for 1 sheep",
               "Bob built a Road"]


def _icon_kinds(ln):
    return [k for t in ln.tokens if t["kind"] == "icons" for k in t["icons"]]


@pytest.mark.parametrize("dark,colours,size", [
    (False, ("red", "blue", "orange", "white"), (1280, 800)),
    (True, ("purple", "brown", "green", "pink"), (1920, 1080)),
    (False, ("brown", "white", "purple", "green"), (1366, 768)),
])
def test_building_icons_are_read_by_shape_and_player_colour(game, dark, colours, size):
    state, _ = game
    state = state.copy()
    for p, c, nm in zip(state.players, colours, ("Alice", "Bob", "Carol", "Dave")):
        p.color, p.name = c, nm
    style = synth.LogPanelStyle.dark() if dark else synth.LogPanelStyle()
    img, truth = render(state, BUILD_LINES, size=size, style=style)
    res = logocr.read_log_panel(img)
    right, n = score(res, truth)
    assert n == len(BUILD_LINES) and right >= n - 1, [(e.canonical, ln.text) for e, ln in zip(truth, res.lines)]
    by_text = {e.text: e for e in truth}
    for e, ln in zip(truth, res.lines):
        kinds = _icon_kinds(ln)
        item = e.canonical.split()[-1]
        if item in ("Road", "Settlement", "City"):
            if sig(ln.text) == sig(e.canonical):
                # the building icon (not a card, not a name) gives the item
                assert kinds == [item.lower()] and ln.text.endswith(" " + item), ln.text
                assert ln.event.kind == "build" and ln.event.item == item.lower()
        else:
            # no building is ever read in a line without one
            assert not set(kinds) & {"road", "settlement", "city"}, (e.text, ln.text)
    assert by_text["Dave placed a Settlement"].canonical.startswith(colours[3])


def test_building_icons_are_not_confused_with_cards_or_names(game):
    """Card icons stay cards and bold names stay names on a panel full of both (the game's own lines)."""
    state, lines = game
    for style in (synth.LogPanelStyle(), synth.LogPanelStyle.dark()):
        img, truth = render(state, lines, style=style)
        res = logocr.read_log_panel(img)
        builds = {i for i, e in enumerate(truth) if e.canonical.split()[-1] in ("Road", "Settlement", "City")}
        for i, (e, ln) in enumerate(zip(truth, res.lines)):
            if i not in builds:
                assert not set(_icon_kinds(ln)) & {"road", "settlement", "city"}, (e.canonical, ln.text)
        right, n = score(res, truth)
        assert right >= n - 1


@pytest.mark.parametrize("font", ["dejavu", "liberation"])
def test_colons_glued_to_verbs_are_read(game, font):
    """Colonist's "got:", "gave bank:", "for:" ...: the colon is read as part of its word, left out of
    the text, and costs no accuracy or confidence against the same panel without colons."""
    state, _ = game
    lines = ["Alice got 2 wood, 1 ore", "Bob gave bank 4 sheep and took 1 ore", "Carol wants to give 1 wood for 1 ore",
             "Dave stole a card from Bob", "Alice traded 1 wood for 1 ore with Carol", "Bob got 1 sheep",
             "Carol gave bank 2 wheat and took 1 brick", "Dave wants to give 2 ore for 1 wheat"]
    state = state.copy()
    for p, nm in zip(state.players, ("Alice", "Bob", "Carol", "Dave")):
        p.name = nm
    if font == "liberation" and not all(os.path.exists(p) for p in LIB):
        pytest.skip("held-out font not installed")
    for size in ((1280, 800), (1920, 1080)):
        reads = {}
        for share in (0.0, 1.0):
            style = synth.LogPanelStyle(colons=share)
            if font == "liberation":
                style.font_path, style.name_font_path = LIB
            img, truth = render(state, lines, size=size, style=style)
            assert all(any(t.text.endswith(":") for t in e.tokens) == (share > 0) for e in truth)
            res = logocr.read_log_panel(img)
            right, n = score(res, truth)
            assert right == n, [(e.canonical, ln.text) for e, ln in zip(truth, res.lines)]
            assert [ln.text for ln in res.lines] == [e.canonical for e in truth]     # no colon in the text
            reads[share] = res
        # the colons are seen (most of the 12 are read as part of their word) ...
        words = [w for ln in reads[1.0].lines for t in ln.tokens if t["kind"] == "text" for w in t["words"]]
        assert sum(w.endswith(":") and len(w) > 1 for w in words) >= 7
        # ... and cost no confidence worth the name: a line fed without colons is fed with them
        for a, b in zip(reads[0.0].lines, reads[1.0].lines):
            assert b.confidence >= min(a.confidence - 0.1, logocr.CONF_FEED), (a.text, a.confidence, b.confidence)
