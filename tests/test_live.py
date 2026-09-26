"""The live local reader (catanbot.vision.live) and the watch / ocr / ocr-teach / ui-profile commands.

The game-log OCR (catanbot.vision.logocr) is not needed: :class:`FakeReader` implements its
``read_log_panel`` contract on synthetic frames by returning the ground truth the renderer drew
(``synth.render_state(..., log_layout=[])``: the canonical text of every visible entry), looked up
by the frame's pixels, optionally with injected misreads.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image, ImageDraw

from catanbot import board as B
from catanbot import cli
from catanbot import colonist_log as L
from catanbot.vision import live as LV
from catanbot.vision import synth
from catanbot.vision.profile import UiProfile
from catanbot.vision.result import ParseResult
from catanbot.vision.schema import parsed_to_state, state_to_parsed

from tests.test_colonist_log import play

SIZE = (1280, 800)


# ---------------------------------------------------------------------------
# helpers: a fake log reader, a truth parser, rendered replays
# ---------------------------------------------------------------------------
def digest(img) -> str:
    arr = np.asarray(img if not isinstance(img, str) else Image.open(img).convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def line(text, conf=0.97, key=None, partial=False, box=None):
    return NS(text=text, event=L.parse_log_line(text) if text else None, confidence=conf,
              key=key if key is not None else "k:" + text, partial=partial, box=box, tokens=[])


class FakeReader:
    """``read_log_panel(img, box=None, profile=None, players=None, cache=None)`` of the log-OCR
    contract, answering from the renderer's ground truth.  ``misread[(digest, k)] = (text, conf)``
    replaces the reading of entry ``k`` (negative: from the bottom) of that frame."""

    def __init__(self, conf=0.97):
        self.conf = conf
        self.frames = {}
        self.misread = {}
        self.calls = 0
        self.players_seen = []

    def add(self, img, layout, box):
        self.frames[digest(img)] = (list(layout), tuple(int(round(v)) for v in box))
        return img

    def find(self, img, profile=None):
        return self.frames[digest(img)][1]

    def __call__(self, img, box=None, profile=None, players=None, cache=None):
        self.calls += 1
        self.players_seen.append(players)
        d = digest(img)
        layout, true_box = self.frames[d]
        lines = []
        for k, e in enumerate(layout):
            text, conf = e.canonical, (0.3 if e.partial else self.conf)
            for kk in (k, k - len(layout)):
                if (d, kk) in self.misread:
                    text, conf = self.misread[(d, kk)]
            lines.append(line(text, conf, key=f"k:{text}|{e.row_count}", partial=e.partial, box=e.box))

        def texts(min_conf=0.6, include_partial=False):
            return [ln.text for ln in lines if ln.confidence >= min_conf and (include_partial or not ln.partial)]
        return NS(lines=lines, box=tuple(box) if box is not None else true_box, warnings=[], confidence=1.0,
                  texts=texts, debug={})

    def module(self):
        """A stand-in for the logocr module (for the CLI via ``live.load_logocr``)."""
        return NS(read_log_panel=self, find_log_panel=self.find, LineCache=dict,
                  teach=lambda img, truth, box=None, profile=None, players=None: (profile, {"learned": len(truth)}))


def colour_parsed(s):
    """What a screenshot of ``s`` shows (seat 0 = me), names as colours like the CV parser."""
    p = state_to_parsed(s, me=0)
    for q in p["players"]:
        q["name"] = q["color"]
    return p


class TruthParser:
    """``parse_fn`` returning the true parse of the frame (fast; the real parser is tested separately)."""

    def __init__(self):
        self.frames = {}
        self.calls = 0

    def add(self, img, s):
        self.frames[digest(img)] = colour_parsed(s)

    def __call__(self, img, me=None, layout=None):
        self.calls += 1
        parsed = json.loads(json.dumps(self.frames[digest(img)]))
        conf = {"geometry": 1.0, "hexes": 1.0, "numbers": 1.0, "ports": 1.0, "pieces": 0.9, "panel": 0.9}
        return ParseResult(parsed=parsed, state=parsed_to_state(parsed), confidence=conf, warnings=[])


class Screen:
    """A scripted screen for :class:`LiveSession` (no rendering): :meth:`show` sets the log reader's
    lines and the parse (with its confidence) of the next frame; the image changes where they did."""

    BOX = (280, 40, 396, 296)

    def __init__(self, parsed):
        self.lines, self.parsed, self.conf, self.fail = [], parsed, None, None
        self.log_ver = self.board_ver = self.reads = self.parses = 0

    def image(self):
        a = np.full((300, 400, 3), 90, np.uint8)
        a[:40, :40] = (self.board_ver * 37) % 256
        a[50:80, 290:320] = (self.log_ver * 53) % 256
        return a

    def reader(self, img, box=None, profile=None, players=None, cache=None):
        self.reads += 1
        return NS(lines=list(self.lines), box=box or self.BOX, warnings=[], confidence=1.0 if self.lines else 0.0)

    def parse(self, img, me=None, layout=None):
        self.parses += 1
        if self.fail:
            raise self.fail
        p = json.loads(json.dumps(self.parsed))
        conf = dict(self.conf or {"hexes": 1.0, "numbers": 1.0, "numbers_min": 1.0, "ports": 1.0})
        return ParseResult(parsed=p, state=parsed_to_state(p), confidence=conf, warnings=[])

    def session(self, **kw):
        return LV.LiveSession(me="red", log_reader=self.reader, parse_fn=self.parse, log_box=self.BOX, **kw)

    def show(self, sess, texts=None, parsed=None):
        """One frame: ``texts`` (a new log view: strings or ``line`` objects) and / or ``parsed`` (a
        board change: a new parse, or the same one redrawn when ``True``)."""
        if texts is not None:
            self.lines = [t if isinstance(t, NS) else line(t) for t in texts]
            self.log_ver += 1
        if parsed is not None:
            self.parsed = self.parsed if parsed is True else parsed
            self.board_ver += 1
        return sess.step(self.image())


def render(s, lines, reader, size=SIZE, seed=5, keep=40, **kw):
    lay = []
    first = max(0, len(lines) - keep)
    img = synth.render_state(s, size=size, seed=seed, me=0, jitter=False, jpeg=False, log=list(lines[first:]),
                             log_first_index=first, log_layout=lay, **kw)
    reader.add(img, lay, synth.log_panel_box(size))
    return img


def replay(seed=2, frames=60, every=5, reveal=True):
    """``(state, lines)`` snapshots of a heuristic-bot game: one whenever ``every`` new log lines
    accumulated, and right after an opponent's offer (so it is on screen while still open)."""
    out = []
    last = 0
    for pre, a, s, lines in play(seed, reveal=reveal, style="counts"):
        new = lines[last:]
        offer = bool(new) and new[-1].split(" ")[0] != "Alice" and " wants to give " in new[-1]
        if len(lines) - last >= every or (offer and len(lines) > last):
            out.append((s.copy(), list(lines)))
            last = len(lines)
            if len(out) >= frames:
                break
    return out


@pytest.fixture(scope="module")
def game():
    return replay()


@pytest.fixture(scope="module")
def rendered(game):
    reader, truth = FakeReader(), TruthParser()
    imgs = []
    for s, lines in game:
        img = render(s, lines, reader)
        truth.add(img, s)
        imgs.append(img)
    return reader, truth, imgs


# ---------------------------------------------------------------------------
# LiveSession end to end
# ---------------------------------------------------------------------------
def test_live_session_replay_counts_cards_reads_log_and_offers_once(game, rendered, tmp_path):
    reader, truth, imgs = rendered
    rec = tmp_path / "rec"
    sess_path = tmp_path / "session.json"
    t = [1000.0]
    sess = LV.LiveSession(me="red", session_path=str(sess_path), log_reader=reader, parse_fn=truth,
                          record_dir=str(rec), clock=lambda: t[0])
    entries, verdicts, skipped_ms = [], [], []
    expected_offers = log_only = 0
    prev = None
    for k, img in enumerate(imgs + [imgs[-1]]):          # the last frame twice: unchanged -> skipped
        t[0] += 2.0
        upd = sess.step(img)
        entries.extend(upd.new_entries)
        verdicts.extend(upd.offers)
        if upd.skipped:
            skipped_ms.append(upd.total_ms)
        cur = {k2: v for k2, v in truth.frames[digest(img)].items() if k2 != "turn"}
        if upd.parse_ms is None:                          # the board was not re-read: it did not change
            assert cur == prev
            log_only += upd.ocr_ms is not None
        prev = cur
        for e in upd.new_entries:
            if e.kind == "offer" and e.event.player != "red" and sess.stream.is_open(e):
                expected_offers += 1
        assert all(not w.startswith("card count:") for w in upd.warnings), upd.warnings
    s_end, lines_end = game[-1]
    canon = [synth.canonical_log_text(x, s_end.players) for x in lines_end]
    assert [e.text for e in entries] == canon            # every entry once, in order, exactly as drawn
    assert len({e.index for e in entries}) == len(entries)
    # the card count follows the truth (every hidden card is revealed in this replay)
    tr = sess.tracker
    assert tr is not None and tr.origin == "start"
    assert [list(h) for h in tr.counter.most_likely()] == [p.resources for p in s_end.players]
    assert all(tr.is_exact(j) for j in range(4))
    assert tr.stats["gaps"] == 0 and tr.stats["repairs"] == 0
    assert os.path.exists(sess_path)
    # offers: a verdict for every opponent offer that was open when read, exactly once
    assert expected_offers >= 1 and len(verdicts) == expected_offers
    assert len({v.entry for v in verdicts}) == len(verdicts)
    for v in verdicts:
        assert v.verdict in ("accept", "reject") or v.verdict.startswith("counter:")
        assert v.reason and v.lines and v.proposer_colour != "red"
    # unchanged frames cost a few ms and read nothing
    assert skipped_ms and max(skipped_ms) < 60
    assert reader.calls == len(imgs) and truth.calls == len(imgs) - log_only and log_only >= 3
    # recording
    frames = sorted(os.listdir(rec / "frames"))
    assert len(frames) == len(imgs)
    ev = [json.loads(x) for x in (rec / "events.jsonl").read_text().splitlines()]
    assert [e["text"] for e in ev] == canon and {"time", "text", "kind", "confidence"} <= set(ev[0])
    parses = [json.loads(x) for x in (rec / "parses.jsonl").read_text().splitlines()]
    assert len(parses) == truth.calls and {"time", "parsed", "warnings"} <= set(parses[0])
    offers = [json.loads(x) for x in (rec / "offers.jsonl").read_text().splitlines()]
    assert len(offers) == len(verdicts) and offers[0]["verdict"] == verdicts[0].verdict
    summ = sess.summary()
    assert summ["frames"] == len(imgs) + 1 and summ["skipped"] == 1 and summ["entries"] == len(canon)


def test_live_session_with_the_real_parser_keeps_the_board_under_a_popup(game, rendered):
    """The CV parser on rendered frames: the log box is kept out of the board, the board locks, a
    popup over the board changes neither tiles nor pieces, and the log / count stay right."""
    reader, _, _ = rendered
    picks = [game[k] for k in (20, 21, 22)]
    imgs = [render(s, lines, reader) for s, lines in picks]
    popup = imgs[1].copy()
    d = ImageDraw.Draw(popup)
    w, h = SIZE
    d.rounded_rectangle((int(0.36 * w), int(0.28 * h), int(0.70 * w), int(0.62 * h)), 12, fill=(245, 240, 225),
                        outline=(60, 60, 60), width=3)
    lay = reader.frames[digest(imgs[1])][0]
    reader.add(popup, lay, synth.log_panel_box(SIZE))
    layouts = []
    calls = []

    def parse(img, me=None, layout=None):
        from catanbot.vision.colonist import parse_image
        layouts.append(layout)
        calls.append(1)
        return parse_image(img, me=me, layout=layout)

    sess = LV.LiveSession(me="red", log_reader=reader, parse_fn=parse)
    ups, kept = [], []
    for im in (imgs[0], imgs[0], imgs[1], popup, imgs[1], imgs[2]):
        ups.append(sess.step(im))
        kept.append((dict(sess.memory.buildings), dict(sess.memory.roads)))
    assert ups[1].parse_ms is None and ups[1].skipped                  # the repeated frame is not parsed again
    assert ups[3].parse_ms is not None and ups[4].parse_ms is not None and len(calls) in (4, 5)
    assert all(isinstance(l, dict) and "log" in l for l in layouts)
    truth_hexes = [{"resource": B.RESOURCE_NAMES[r], "number": (n or None)} for r, n in picks[0][0].hexes]
    assert sess.memory.locked and ups[1].parsed["hexes"] == truth_hexes

    def pieces(u):
        b = {v: (p["color"], False) for p in u.parsed["players"] for v in p["settlements"]}
        b.update({v: (p["color"], True) for p in u.parsed["players"] for v in p["cities"]})
        return b, {e: p["color"] for p in u.parsed["players"] for e in p["roads"]}
    under = ups[3]                                                     # the popup hides part of the board
    assert under.parsed["hexes"] == truth_hexes and under.parsed.get("ports") == ups[2].parsed.get("ports")
    b, r = pieces(under)
    assert kept[2][0].items() <= b.items() and kept[2][1].items() <= r.items()   # nothing confirmed is lost
    truth21 = picks[1][0]
    b, r = pieces(ups[4])                                              # the popup gone: everything is back
    assert r == {e: p.color for p in truth21.players for e in p.roads}
    assert b == {**{v: (p.color, False) for p in truth21.players for v in p.settlements},
                 **{v: (p.color, True) for p in truth21.players for v in p.cities}}
    assert kept[4][1] == r                                             # seen in 2 of the last 3 parses: kept
    assert not any(u.new_game for u in ups)
    s_end, lines_end = picks[-1]
    got = [e.text for u in ups for e in u.new_entries]
    canon = [synth.canonical_log_text(x, s_end.players) for x in lines_end]
    assert got == canon[-len(got):] and len(got) >= 10
    tr = sess.tracker
    assert tr is not None and tr.counter.size == [sum(p.resources) for p in s_end.players]
    assert tr.counter.weight_of([p.resources for p in s_end.players]) > 0


def test_live_session_new_game_starts_over(game, rendered, tmp_path):
    """A second game on screen: detected once, the log / count / session start over, and the new
    game's log is read from its first entry (nothing of it lands in the old game's count)."""
    reader, truth, imgs = rendered
    other = replay(seed=5, frames=6, every=4)
    imgs2 = []
    for s, lines in other:
        img = render(s, lines, reader)
        truth.add(img, s)
        imgs2.append(img)
    sp = tmp_path / "s.json"
    sess = LV.LiveSession(me="red", session_path=str(sp), log_reader=reader, parse_fn=truth)
    ups = [sess.step(im) for im in imgs[:12] + imgs2 + [imgs2[-1]]]
    assert sum(u.new_game for u in ups) == 1 and ups[12 + 2].new_game
    old_entries = [e.text for u in ups[:12 + 2] for e in u.new_entries]
    s1, l1 = game[11]
    assert old_entries == [synth.canonical_log_text(x, s1.players) for x in l1]    # nothing of game 2 before
    s2, l2 = other[-1]
    assert [e.text for e in sess.stream.entries] == [synth.canonical_log_text(x, s2.players) for x in l2]
    tr = sess.tracker
    assert tr.origin == "start" and [list(h) for h in tr.counter.most_likely()] == [p.resources for p in s2.players]
    assert (tmp_path / "s.json.previous").exists() and sp.exists()
    assert sum("new game" in w for u in ups for w in u.warnings) >= 1
    assert not any("jumped" in w or "card count:" in w for u in ups for w in u.warnings)


def test_live_session_players_lock_on_fresh_parses_and_a_corrected_list_restarts_the_count(tmp_path):
    """An unchanged frame is no second look (nothing locks from it); a player list locked from two
    frames that missed a player is corrected by 3 frames showing all, and the count starts over."""
    p = parsed_of()
    three = json.loads(json.dumps(p))
    three["players"] = three["players"][:3]
    sc = Screen(three)
    sp = tmp_path / "s.json"
    sess = sc.session(session_path=str(sp))
    sc.show(sess, ["green rolled 1 2"], parsed=True)
    sc.show(sess)                                                    # the same pixels again
    assert sc.parses == 1 and sess.memory.colours is None and not sess.memory.locked
    for _ in range(3):
        sc.show(sess, parsed=True)
    assert sess.memory.colours == ["red", "blue", "orange"] and sess.tracker.colors == ["red", "blue", "orange"]
    ups = [sc.show(sess, parsed=p) for _ in range(3)] + [sc.show(sess) for _ in range(2)]
    assert sess.memory.colours == ["red", "blue", "orange", "green"]
    assert [q.color for q in sess.state.players] == ["red", "blue", "orange", "green"]
    assert sess.tracker.colors == ["red", "blue", "orange", "green"] and (tmp_path / "s.json.previous").exists()
    warned = [w for u in ups for w in u.warnings]
    assert any("player list is corrected" in w for w in warned) and any("players were re-read" in w for w in warned)


def test_live_session_judges_an_offer_with_the_count_up_to_it_and_after_a_late_board():
    """The count is fed the entries up to an offer before it is judged (the proposer's cards got just
    before it are counted), and an offer read before the board was is judged once there is a board."""
    p = parsed_of()
    blue = next(q for q in p["players"] if q["color"] == "blue")
    blue["cards"] = 0
    sc = Screen(p)
    sess = sc.session()
    sc.show(sess, ["green rolled 1 2"], parsed=True)
    sc.show(sess, parsed=True)
    sc.show(sess)
    assert sess.tracker is not None and tuple(sess.tracker.counter.most_likely()[1]) == (0, 0, 0, 0, 0)
    seen = []
    real = LV.evaluate_offer

    def spy(state, me, proposer, give, get, tracker=None, **kw):
        seen.append(tuple(tracker.counter.most_likely()[proposer]))
        return real(state, me, proposer, give, get, tracker=tracker, **kw)
    p2 = json.loads(json.dumps(p))
    next(q for q in p2["players"] if q["color"] == "blue")["cards"] = 2
    try:
        LV.evaluate_offer = spy
        u = sc.show(sess, ["green rolled 1 2", "blue got 2 ore", "blue wants to give 2 ore for 1 sheep"], parsed=p2)
    finally:
        LV.evaluate_offer = real
    assert len(u.offers) == 1 and seen == [(0, 0, 0, 0, 2)]
    assert not any("cannot hold" in w for w in u.offers[0].warnings)
    # an offer read while the board is not read yet waits (while it is open) for a game state
    sc2 = Screen(p)
    sc2.fail = ValueError("no board yet")
    s2 = sc2.session()
    u1 = sc2.show(s2, ["green rolled 1 2", "blue wants to give 2 ore for 1 sheep"], parsed=True)
    sc2.fail = None
    u2 = sc2.show(s2, parsed=True)
    assert u1.offers == [] and [v.text for v in u2.offers] == ["blue wants to give 2 ore for 1 sheep"]
    assert sc2.show(s2, parsed=True).offers == []                   # judged once


def test_live_session_stops_waiting_on_a_bottom_line_that_never_reads():
    """A bottom entry the reader never reads holds the card count and the re-reads of an unchanged
    panel only for ``stall_frames`` reads."""
    sc = Screen(parsed_of())
    sess = sc.session()
    sc.show(sess, ["green rolled 1 2", line("#?%", 0.2, key="junk")], parsed=True)
    sc.show(sess, parsed=True)
    ups = [sc.show(sess) for _ in range(6)]
    assert sc.reads == 3 and not sess.stream.waiting and sess.stream.pending
    assert [e.text for e in sess.stream.entries] == ["green rolled 1 2"]
    assert sess.card_count is not None and any(u.card_count for u in ups)
    assert sum(u.skipped for u in ups) >= 4


def test_load_logocr_treats_a_broken_module_as_missing(monkeypatch, tmp_path, capsys):
    import types
    name = LV.LOGOCR_MODULE
    mod = types.ModuleType(name)
    mod.find_log_panel = lambda img, profile=None: None               # no read_log_panel, no teach
    monkeypatch.setitem(sys.modules, name, mod)
    assert LV.load_logocr() is None and "it has no read_log_panel, teach" in LV.logocr_missing_message()
    sess = LV.LiveSession(me="red", parse_fn=TruthParser())
    assert not sess.read_log and sess._reader is None
    assert cli.main(["ocr", str(tmp_path / "shot.png"), "--log-region", "1000,120,265,504"]) == 2
    assert "catanbot.vision.logocr" in capsys.readouterr().err
    monkeypatch.delitem(sys.modules, name, raising=False)
    real = LV.importlib.import_module

    def broken(n, *a, **k):
        if n == name:
            raise RuntimeError("cv2 build lacks feature X")
        return real(n, *a, **k)
    monkeypatch.setattr(LV.importlib, "import_module", broken)
    assert LV.load_logocr() is None
    assert "not available in this installation (importing it failed: RuntimeError: cv2 build lacks feature X)" \
        in LV.logocr_missing_message()
    assert not LV.LiveSession(me="red", parse_fn=TruthParser()).read_log


def test_live_session_with_the_installed_log_reader(game, rendered):
    """Runs only once catanbot.vision.logocr is installed: the real OCR in the live loop on the
    replay's first frames (board from the truth parser) - entries in order, each at most once, no
    gap, and nearly every entry the reader itself reads exactly (in some frame) comes out exactly:
    the loop loses nothing the reader gets right (the reader's own accuracy is its benchmark's)."""
    import difflib
    logocr = pytest.importorskip("catanbot.vision.logocr")
    reader, truth, imgs = rendered
    read_ok = set()

    def spy(img, **kw):
        res = logocr.read_log_panel(img, **kw)
        lay = reader.frames[digest(img)][0]
        if len(res.lines) == len(lay):
            read_ok.update(e.index for e, ln in zip(lay, res.lines)
                           if not e.partial and ln.text == e.canonical and ln.confidence >= 0.6)
        return res
    sess = LV.LiveSession(me="red", parse_fn=truth, log_reader=spy, log_finder=logocr.find_log_panel,
                          log_cache=logocr.LineCache())
    got = []
    for img in imgs[:20] + [imgs[19]]:
        got.extend(e.text for e in sess.step(img).new_entries)
    s_end, lines_end = game[19]
    canon = [synth.canonical_log_text(x, s_end.players) for x in lines_end]
    sm = difflib.SequenceMatcher(a=canon, b=got, autojunk=False)
    exact = {b.a + i for b in sm.get_matching_blocks() for i in range(b.size)}
    assert len(exact & read_ok) >= 0.9 * len(read_ok) and len(got) <= len(canon) + 2, \
        (len(exact & read_ok), len(read_ok), len(exact), len(canon), len(got))
    assert sess.stream.gaps == 0


# ---------------------------------------------------------------------------
# LogStream
# ---------------------------------------------------------------------------
def canon_game(seed=2, n=None):
    for pre, a, s, lines in play(seed, reveal=True, style="counts"):
        pass
    out = [synth.canonical_log_text(x, s.players) for x in lines]
    return out[:n] if n else out


@pytest.fixture(scope="module")
def log_lines():
    return canon_game()


def window(log, end, size=14, conf=0.95):
    return [line(t, conf) for t in log[max(0, end - size):end]]


def test_stream_emits_each_entry_once_over_a_replayed_game_with_scrolls_popups_and_misreads(log_lines):
    """A panel of 16 entries read every frame while the log grows by 0-4 entries: popups hide the
    panel, the user scrolls up, confident misreads hit entries already read (a misread of a new
    entry at confidence >= 0.9 cannot be told from the truth) and the newest entry is not sure."""
    rng = random.Random(4)
    st = LV.LogStream()
    out = []
    end = seen = 0
    frames = 0
    while end < len(log_lines):
        end = min(len(log_lines), end + rng.randint(0, min(4, 12 - (end - seen))))
        roll = rng.random()
        if roll < 0.08:
            got = st.update(None)                                         # a popup hides the panel
        elif roll < 0.16 and end > 30:
            back = rng.randint(5, 25)                                     # the user scrolled up
            got = st.update(window(log_lines, end - back, rng.randint(6, 14)))
        else:
            w = window(log_lines, end, 16)
            if len(w) > 8 and rng.random() < 0.4:                         # a confident misread of an old entry
                k = rng.randrange(0, max(1, len(w) - (end - seen) - 1))
                w[k] = line(w[k].text.replace("1", "7").replace("got", "gat") + " x", 0.95, key="other")
            if w and rng.random() < 0.3:                                  # the newest entry not sure yet
                w[-1] = line(w[-1].text, 0.7)
            got = st.update(w)
            seen = end
        out.extend(got)
        frames += 1
    for _ in range(2):
        out.extend(st.update(window(log_lines, len(log_lines), 16)))
    assert frames > 60
    assert [e.text for e in out] == log_lines
    assert [e.index for e in out] == list(range(len(log_lines))) and st.gaps == 0


def test_stream_flip_flopping_bottom_line_is_confirmed_once_stable(log_lines):
    st = LV.LogStream()
    base = window(log_lines, 20)
    assert len(st.update(base)) == 14
    good, bad = log_lines[20], log_lines[20].replace("1", "2") + " z"
    seq = [(good, 0.7), (bad, 0.7), (good, 0.72), (bad, 0.65), (good, 0.7), (good, 0.7)]
    got = []
    for k, (t, c) in enumerate(seq):
        new = st.update(base[1:] + [line(t, c, key=f"flip{k}")])
        got.append([e.text for e in new])
    assert got == [[], [], [], [], [], [good]]
    assert st.update(base[2:] + [line(good, 0.7, key="x"), line(log_lines[21], 0.95)])[0].text == log_lines[21]


def test_stream_keeps_the_confirmed_text_of_an_overlapping_line(log_lines):
    st = LV.LogStream()
    st.update(window(log_lines, 30))
    w = window(log_lines, 31)
    w[5] = line("blue got 9 ore", 0.99, key="misread")                     # an old entry read differently
    new = st.update(w)
    assert [e.text for e in new] == [log_lines[30]]
    assert [e.text for e in st.entries] == log_lines[16:31]


def test_stream_scrolled_up_and_back(log_lines):
    st = LV.LogStream()
    for end in range(10, 60, 3):
        st.update(window(log_lines, end))
    n = len(st.entries)
    assert st.update(window(log_lines, 30, 10)) == [] and len(st.entries) == n     # an older part: nothing new
    assert n == 58 and [e.text for e in st.update(window(log_lines, 62))] == log_lines[58:62]


def test_stream_gap_is_accepted_when_two_frames_agree_and_reported(log_lines):
    t = [0.0]
    st = LV.LogStream(clock=lambda: t[0])
    st.update(window(log_lines, 40))
    t[0] += 1
    # a far older part of the log (never read, the panel scrolled up): a static view is not a gap at once
    assert st.update(window(log_lines, 5)) == []
    assert st.update(window(log_lines, 5)) == [] and st.gaps == 0
    # the log moved on while the screen was not watched: the live end shows up, one entry later
    st2 = LV.LogStream(clock=lambda: t[0])
    st2.update(window(log_lines, 40))
    assert st2.update(window(log_lines, 120)) == []
    new = st2.update(window(log_lines, 121))                  # (with the entry that scrolled out in between)
    assert [e.text for e in new] == log_lines[106:121] and new[0].gap_before and not new[1].gap_before
    assert st2.gaps == 1 and any("jumped" in w for w in st2.warnings)
    assert [e.text for e in st2.update(window(log_lines, 123))] == log_lines[121:123]
    # a static view that shares nothing is never a gap, however long it stays (a panel left scrolled
    # up past everything read); a quiet log after a real gap is taken once it grows
    t[0] += 30
    assert st.update(window(log_lines, 5)) == [] and st.gaps == 0 and st.entries[-1].text == log_lines[39]


def test_stream_no_panel_and_unreadable_entries(log_lines):
    st = LV.LogStream()
    st.update(window(log_lines, 20))
    w = window(log_lines, 21)
    w[-1] = line(w[-1].text, 0.7)
    assert st.update(w) == []
    assert st.update(None) == [] and st.no_panel == 1                          # a popup: nothing changes
    assert [e.text for e in st.update(w)] == [log_lines[20]]                  # read the same twice
    # an entry that stays unreadable while later ones are read is kept in order, not counted
    for k in range(3):
        w = window(log_lines, 24)
        w[-3] = line("#?%", 0.2, key=f"junk{k}")
        new = st.update(w)
    assert [e.text for e in new] == ["#?%", log_lines[22], log_lines[23]] and not new[0].countable
    assert new[0].kind == "unreadable" and all(e.countable for e in new[1:])
    assert all(e.countable is False or e.event is not None for e in st.entries)


def test_stream_overlap_misread_at_the_live_end_and_a_gap_keep_the_log_single(log_lines):
    """The log moved 11 entries between two reads of a 14-line panel: of the 3 overlap lines one is
    re-read differently (another key) - aligned at the end of the log, not taken for a gap, the
    overlap keeps its text.  A jump accepted as a gap does not take again an entry it still shows."""
    st = LV.LogStream()
    st.update(window(log_lines, 40))
    w = window(log_lines, 51)
    w[1] = line(w[1].text.replace("1", "2") + " x", 0.95, key="re-read")
    assert [e.text for e in st.update(w)] == log_lines[40:51] and st.gaps == 0
    st2 = LV.LogStream()
    st2.update(window(log_lines, 40))
    assert st2.update(window(log_lines, 53)) == []                  # one line of overlap: not proof enough
    new = st2.update(window(log_lines, 54))
    assert [e.text for e in new] == log_lines[40:54] and new[0].gap_before and st2.gaps == 1
    assert [e.text for e in st2.entries] == log_lines[26:54]


def test_stream_panel_left_scrolled_up_is_never_a_gap():
    """Scrolled up past everything read, for any time: nothing is appended, and back at the live end
    the log continues without a duplicate or a gap."""
    t = [0.0]
    st = LV.LogStream(clock=lambda: t[0])
    log = [f"red got {i} wood" for i in range(1, 200)]
    st.update([line(x) for x in log[100:114]])
    for _ in range(4):
        t[0] += 8
        assert st.update([line(x) for x in log[50:64]]) == []
    got = []
    for end in range(115, 160):
        t[0] += 2
        got += st.update([line(x) for x in log[end - 14:end]])
    assert [e.text for e in got] == log[114:159] and st.gaps == 0


def test_stream_repeated_line_after_a_jump_is_reported_not_lost_silently():
    """Entries 20 and 21 read the same; the next frame starts at 21: its top line matching 20 is no
    proof of an overlap, so the jump is reported (the card count resynchronises)."""
    log = [f"blue got {i} ore" for i in range(1, 21)] + ["green wants to give 1 sheep for 1 brick"] * 2 + \
          [f"red got {i} wood" for i in range(1, 30)]
    st = LV.LogStream()
    st.update([line(x) for x in log[7:21]])
    assert st.update([line(x) for x in log[21:35]]) == [] and st.update([line(x) for x in log[21:35]]) == []
    new = st.update([line(x) for x in log[22:36]])
    assert new[0].gap_before and st.gaps == 1 and any("jumped" in w for w in st.warnings)
    assert [e.text for e in st.entries] == log[7:21] + log[22:36]


def test_stream_unreadable_lines_are_released_together_and_scrolled_past(log_lines):
    """Several entries the reader cannot read (dice icons) are released at once after
    ``stall_frames`` frames; a panel that moved past lines still waiting aligns on them (no gap)."""
    def frame(end, bad):
        w = window(log_lines, end)
        return [line("#?%", 0.3, key=f"bad{end - len(w) + i}") if end - len(w) + i in bad else x
                for i, x in enumerate(w)]
    st = LV.LogStream()
    st.update(window(log_lines, 20))
    outs = [st.update(frame(24, (20, 22))) for _ in range(3)]
    assert outs[:2] == [[], []] and [e.text for e in outs[2]] == ["#?%", log_lines[21], "#?%", log_lines[23]]
    assert [e.countable for e in outs[2]] == [False, True, False, True]
    st2 = LV.LogStream()
    st2.update(window(log_lines, 20))
    assert st2.update(frame(30, (20,))) == [] and len(st2.pending) == 10
    new = st2.update(window(log_lines, 37))                         # 16..22 scrolled out, 20 never read
    assert [e.text for e in new] == ["#?%"] + log_lines[21:37] and not new[0].countable and st2.gaps == 0


def test_stream_follows_whether_an_offer_is_open():
    st = LV.LogStream()
    got = st.update([line("red rolled 3 4"), line("blue wants to give 1 wood for 1 ore")])
    offer = got[1]
    assert offer.kind == "offer" and st.is_open(offer)
    st.update([line("blue wants to give 1 wood for 1 ore"), line("green rejected the trade")])
    assert st.is_open(offer)
    st.update([line("green rejected the trade"), line("blue traded 1 wood for 1 ore with red")])
    assert not st.is_open(offer)
    o2 = st.update([line("blue traded 1 wood for 1 ore with red"), line("orange wants to give 2 sheep for 1 brick")])[0]
    assert st.is_open(o2)
    st.update([line("orange wants to give 2 sheep for 1 brick"), line("orange cancelled the trade")])
    assert not st.is_open(o2)
    o3 = st.update([line("orange cancelled the trade"), line("orange wants to give 1 sheep for 1 brick")])[0]
    st.update([line("orange wants to give 1 sheep for 1 brick"), line("green rolled 2 2")])
    assert not st.is_open(o3)


# ---------------------------------------------------------------------------
# BoardMemory
# ---------------------------------------------------------------------------
def parsed_of(seed=3, turns=40):
    from tests.test_cli import played_state
    return colour_parsed(played_state(seed=seed, turns=turns))


CONF = {"hexes": 0.95, "numbers": 0.95, "ports": 0.9, "panel": 0.9}


def test_board_memory_locks_and_keeps_the_board_under_a_popup():
    p = parsed_of()
    m = LV.BoardMemory()
    assert not m.update(p, CONF).locked
    assert m.update(p, CONF).locked
    m.update(p, CONF)                                      # the pieces: voted by the 2 frames since the lock
    popup = json.loads(json.dumps(p))
    for i in (3, 4, 8):
        popup["hexes"][i] = {"resource": "desert", "number": None}
    popup["ports"] = popup["ports"][:4]
    for q in popup["players"]:
        q["roads"] = q["roads"][:1]
    warned = []
    for _ in range(3):
        u = m.update(popup, dict(CONF, numbers_min=0.05))  # a popup hides number tokens
        warned += u.warnings
        assert u.parsed["hexes"] == p["hexes"] and u.parsed["ports"] == p["ports"] and not u.new_game
        for a, b in zip(u.parsed["players"], p["players"]):
            assert a["roads"] == b["roads"] and a["settlements"] == b["settlements"]
    assert len([w for w in warned if "tile(s) different" in w]) == 1 and len(warned) == 2    # each said once


def test_board_memory_pieces_are_monotonic_with_city_upgrades():
    p = parsed_of()
    m = LV.BoardMemory()
    m.update(p, CONF)
    m.update(p, CONF)
    red = lambda u: next(q for q in u.parsed["players"] if q["color"] == "red")
    edge = next(e for e in range(B.NUM_EDGES) if all(e not in q["roads"] for q in p["players"]))
    p1 = json.loads(json.dumps(p))
    p1["players"][0]["roads"].append(edge)
    assert edge in red(m.update(p1, CONF))["roads"]                              # shown at once ...
    assert edge not in red(m.update(p, CONF))["roads"]                           # ... a one-frame blip is not kept
    assert edge not in m.roads
    m.update(p1, CONF)
    m.update(p1, CONF)
    assert edge in m.roads and edge in red(m.update(p, CONF))["roads"]           # 2 of 3: kept when missed later
    v = p["players"][0]["settlements"][0]
    p2 = json.loads(json.dumps(p1))
    p2["players"][0]["settlements"].remove(v)
    p2["players"][0]["cities"].append(v)
    m.update(p2, CONF)
    m.update(p2, CONF)
    u = m.update(p1, CONF)                                                       # a frame missing the city
    assert v in red(u)["cities"] and v not in red(u)["settlements"]
    gone = json.loads(json.dumps(p1))
    gone["players"][0]["roads"] = []
    assert red(m.update(gone, CONF))["roads"] == red(u)["roads"]


def test_board_memory_popup_pieces_never_count_and_a_steady_reading_corrects_a_piece():
    """A popup (number tokens hidden; its cream colour read as a 'white' player's pieces) on the first
    frames or later, seen again unchanged, adds no piece, player or lock; a remembered piece read
    otherwise in 4 of the last 5 clean frames is replaced or removed."""
    p = parsed_of()
    taken = {v for q in p["players"] for v in q["settlements"] + q["cities"]}
    free_v = next(v for v in range(B.NUM_VERTICES) if v not in taken)
    blue_v = p["players"][1]["settlements"][0]
    popup = json.loads(json.dumps(p))
    popup["players"].append({"color": "white", "name": "white", "vp": 0, "cards": 0, "settlements": [free_v],
                             "cities": [blue_v], "roads": list(p["players"][1]["roads"])})
    hidden = dict(CONF, numbers_min=0.05)
    m = LV.BoardMemory()
    m.update(popup, hidden)
    m.update(popup, hidden, fresh=False)
    assert not m.locked and m.colours is None and not m.buildings and not m.roads
    for _ in range(3):
        m.update(p, CONF)
    assert m.buildings[blue_v] == ("blue", False) and m.colours == ["red", "blue", "orange", "green"]
    for _ in range(4):
        m.update(popup, hidden)
        m.update(popup, hidden, fresh=False)
    assert "white" not in {c for c, _ in m.buildings.values()} | set(m.roads.values()) and free_v not in m.buildings
    # blue's settlement really is red's, and a road of blue's is not there: 4 of the last 5 clean frames
    fixed = json.loads(json.dumps(p))
    fixed["players"][1]["settlements"].remove(blue_v)
    fixed["players"][0]["settlements"].append(blue_v)
    gone = fixed["players"][1]["roads"].pop()
    for k in range(4):
        u = m.update(fixed, CONF)
        assert (m.buildings[blue_v] == ("red", False)) == (k == 3) and (gone in m.roads) == (k < 3)
    assert any(f"vertex {blue_v}: a red settlement read in 4" in w for w in u.warnings)


def test_board_memory_locks_on_clean_frames_and_corrects_a_wrong_lock():
    p = parsed_of()
    wrong = json.loads(json.dumps(p))
    wrong["hexes"][0], wrong["hexes"][1] = wrong["hexes"][1], {"resource": "desert", "number": None}
    assert sum(a != b for a, b in zip(wrong["hexes"], p["hexes"])) == 2
    m = LV.BoardMemory()
    m.update(wrong, dict(CONF, numbers_min=0.05))                  # a popup over two tiles on the first frame ...
    m.update(wrong, dict(CONF, numbers_min=0.05), fresh=False)     # ... and the same pixels again
    assert not m.locked
    m.update(p, CONF)
    assert m.update(p, CONF).locked and m.hexes == LV.BoardMemory._hex_sig(p)
    # a wrong first lock (two clean frames misread alike) is corrected by 3 clean frames in a row
    m2 = LV.BoardMemory()
    m2.update(wrong, CONF)
    m2.update(wrong, CONF)
    ups = [m2.update(p, CONF) for _ in range(3)]
    assert ups[1].parsed["hexes"] == wrong["hexes"] and ups[2].parsed["hexes"] == p["hexes"] and m2.locked
    assert any("locked board is corrected" in w for w in ups[2].warnings) and not any(u.new_game for u in ups)
    # the player list: two frames showed 3 of the 4 players, then all 4 for good
    three = json.loads(json.dumps(p))
    three["players"] = three["players"][:3]
    m3 = LV.BoardMemory()
    m3.update(three, CONF)
    m3.update(three, CONF)
    assert m3.colours == ["red", "blue", "orange"]
    ups = [m3.update(p, CONF) for _ in range(3)]
    assert [q["color"] for q in ups[1].parsed["players"]] == ["red", "blue", "orange"]
    assert m3.colours == [q["color"] for q in ups[2].parsed["players"]] == ["red", "blue", "orange", "green"]


def test_board_memory_detects_a_new_game():
    p = parsed_of()
    other = parsed_of(seed=11, turns=10)
    m = LV.BoardMemory()
    m.update(p, CONF)
    m.update(p, CONF)
    news = [m.update(other, CONF).new_game for _ in range(3)]
    assert news == [False, False, True] and not m.locked
    assert m.update(other, CONF).locked and m.update(other, CONF).parsed["hexes"] == other["hexes"]
    # the same board with the pieces gone (a rematch on the same map)
    m2 = LV.BoardMemory()
    for _ in range(3):
        m2.update(p, CONF)
    empty = json.loads(json.dumps(p))
    for q in empty["players"]:
        q["settlements"], q["cities"], q["roads"] = [], [], []
    assert [m2.update(empty, CONF).new_game for _ in range(3)] == [False, False, True]
    # an unconfident frame never starts a new game, nor does a trade window hiding most of the board
    m3 = LV.BoardMemory()
    m3.update(p, CONF)
    m3.update(p, CONF)
    assert not any(m3.update(other, {"hexes": 0.3, "numbers": 0.3}).new_game for _ in range(5))
    covered = json.loads(json.dumps(p))
    covered["hexes"][:12] = other["hexes"][:12]
    assert sum(a != b for a, b in zip(covered["hexes"], p["hexes"])) >= 10
    assert not any(m3.update(covered, c).new_game for c in [CONF] * 2 + [dict(CONF, numbers_min=0.1)] * 4)
    assert not any(m3.update(other, dict(CONF, numbers_min=0.2)).new_game for _ in range(6))   # tokens hidden
    assert m3.locked and m3.update(p, CONF).parsed["hexes"] == p["hexes"] and not m3.suspect


# ---------------------------------------------------------------------------
# evaluate_offer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def offer_state():
    s = parsed_to_state(parsed_of(seed=3, turns=30))
    s.players[0].resources = [0, 0, 1, 2, 1]          # a settlement and 1 sheep, 2 wheat, 1 ore
    s.players[0].hand_size = 4
    return s


def test_evaluate_offer_accepts_a_good_offer_and_rejects_a_bad_one(offer_state):
    s = offer_state
    assert s.players[0].settlements and s.players[0].hand_known
    good = LV.evaluate_offer(s, 0, 1, [0, 0, 0, 0, 2], [0, 0, 1, 0, 0])     # 2 ore for our sheep: a city
    assert good.verdict == "accept" and good.accept and "city" in good.reason
    assert good.lines and good.lines[0].startswith("Best answer after") and good.proposer_colour == s.players[1].color
    assert good.summary().startswith("you get 2 ore, you give 1 sheep -> ACCEPT: ")
    bad = LV.evaluate_offer(s, 0, 1, [0, 0, 1, 0, 0], [0, 0, 0, 2, 1])      # our city cards for a sheep
    assert bad.verdict == "reject" and not bad.accept and bad.reason
    poor = LV.evaluate_offer(s, 0, 1, [1, 0, 0, 0, 0], [0, 0, 0, 0, 5])     # we cannot pay
    assert poor.verdict == "reject" and "cannot pay" in poor.reason
    assert any(w.startswith("you cannot pay 5 ore") for w in poor.warnings)
    assert "you get 1 wood, you give 5 ore -> REJECT" in poor.summary()
    assert json.loads(json.dumps(good.to_dict()))["verdict"] == "accept"


def test_evaluate_offer_with_a_card_counter(game):
    s, lines = game[25]
    st = parsed_to_state(colour_parsed(s))
    tr = L.ColonistLogTracker.for_state(st, 0)
    tr.update([L.parse_log_text("\n".join(synth.canonical_log_text(x, s.players) for x in lines))], st)
    me_hand = st.players[0].resources
    counted = tr.counter.most_likely()[1]
    give = [0] * 5
    give[max(range(5), key=lambda r: counted[r])] = 1                         # a card blue holds (by the count)
    get = [0] * 5
    get[max(range(5), key=lambda r: me_hand[r])] = 1
    v = LV.evaluate_offer(st, 0, 1, give, get, tracker=tr)
    assert v.verdict in ("accept", "reject") or v.verdict.startswith("counter:")
    assert v.rule_reason and v.lines and not any("not usable" in w or "cannot hold" in w for w in v.warnings)
    assert LV.evaluate_offer(st, 0, 0, give, get).reason == "not an offer to us"
    # a count behind the log (blue offers cards it says blue cannot hold): judged on the public view
    assert tr.is_exact(1)
    lack = next(r for r in range(5) if counted[r] == 0)
    stale = LV.evaluate_offer(st, 0, 1, [2 if r == lack else 0 for r in range(5)], get, tracker=tr)
    assert any(w.startswith("the card count says blue cannot hold 2 ") for w in stale.warnings)
    assert stale.rule_reason and stale.lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@pytest.fixture
def frames_dir(tmp_path, game):
    """Four consecutive replay frames ending with an opponent's open offer, as PNG files, and the
    fake log reader that knows them."""
    reader = FakeReader()
    k = next(i for i in range(4, len(game)) if " wants to give " in game[i][1][-1]
             and not game[i][1][-1].startswith("Alice"))
    d = tmp_path / "frames"
    d.mkdir()
    for j, (s, lines) in enumerate(game[k - 3:k + 1]):
        render(s, lines, reader).save(d / f"{j:03d}.png")
    return d, reader


def test_cli_watch_reads_the_log_and_evaluates_offers(frames_dir, tmp_path, capsys, monkeypatch):
    d, reader = frames_dir
    monkeypatch.setattr(LV, "load_logocr", reader.module)
    rec = tmp_path / "rec"
    rc = cli.main(["watch", "--from-dir", str(d), "--me", "red", "--depth", "1", "--samples", "1", "--beam", "2",
                   "--model", "heuristic", "--session", str(tmp_path / "s.json"), "--record", str(rec)])
    out = capsys.readouterr().out
    assert rc == 0 and "watch mode" in out and "game log" in out
    assert "OFFER from " in out and " -> " in out and "Best answer after" in out
    assert out.count("turn of") >= 1 and "cards: " in out and "stopped: 4 frames" in out
    assert any(l.split("] ", 1)[1].split(" ")[0] in ("red", "blue", "orange", "green")
               for l in out.splitlines() if l.startswith("[") and "] " in l)
    assert (tmp_path / "s.json").exists() and len(os.listdir(rec / "frames")) == 4
    assert (rec / "offers.jsonl").read_text().count("\n") == out.count("OFFER from ")


def test_cli_watch_without_logocr_reads_the_board_and_survives_bad_frames(frames_dir, capsys, monkeypatch):
    d, _ = frames_dir
    monkeypatch.setitem(sys.modules, "catanbot.vision.logocr", None)
    calls = []
    real = LV.LiveSession.step

    def flaky(self, img):
        calls.append(img)
        if len(calls) in (2, 3):
            raise RuntimeError("boom 42")
        return real(self, img)
    monkeypatch.setattr(LV.LiveSession, "step", flaky)
    rc = cli.main(["watch", "--from-dir", str(d), "--me", "red", "--depth", "1", "--samples", "1", "--beam", "2",
                   "--model", "heuristic"])
    out = capsys.readouterr().out
    assert rc == 0 and out.count("not available in this installation") == 1
    assert out.count("boom") == 1 and out.count("turn of") >= 1 and "stopped: " in out

    def interrupt(self, img):
        raise KeyboardInterrupt
    monkeypatch.setattr(LV.LiveSession, "step", interrupt)
    rc = cli.main(["watch", "--from-dir", str(d), "--me", "red", "--model", "heuristic"])
    out = capsys.readouterr().out
    assert rc == 0 and "stopped: 0 frames" in out


def test_cli_ocr_commands_without_logocr_fail_clearly(frames_dir, tmp_path, capsys, monkeypatch):
    d, _ = frames_dir
    img = str(sorted(d.iterdir())[0])
    monkeypatch.setitem(sys.modules, "catanbot.vision.logocr", None)
    truth = tmp_path / "truth.txt"
    truth.write_text("red rolled 5 3\n")
    for argv in (["ocr", img], ["ocr-teach", img, "--truth", str(truth), "--ui-profile", str(tmp_path / "p.json")],
                 ["ui-profile", str(tmp_path / "p.json"), "--detect", img]):
        rc = cli.main(argv)
        err = capsys.readouterr().err
        assert rc == 2 and err.startswith("error:") and "catanbot.vision.logocr" in err, argv
    assert not (tmp_path / "p.json").exists()


def test_cli_ui_profile_set_and_detect(frames_dir, tmp_path, capsys, monkeypatch):
    d, reader = frames_dir
    prof = tmp_path / "screen.json"
    assert cli.main(["ui-profile", str(prof), "--set", "log=0.8,0.15,0.99,0.78"]) == 0
    out = capsys.readouterr().out
    assert "log: 0.8000,0.1500,0.9900,0.7800" in out
    assert cli.main(["ui-profile", str(prof), "--set", "bank=1000,10,200,60"]) == 2               # pixels need a size
    assert "screen size" in capsys.readouterr().err
    assert cli.main(["ui-profile", str(prof), "--screen", "1280x800", "--set", "bank=1000,10,200,60"]) == 0
    out = capsys.readouterr().out
    assert "bank:" in out and "pixels (1000, 10, 1200, 70)" in out
    monkeypatch.setattr(LV, "load_logocr", reader.module)
    img = str(sorted(d.iterdir())[0])
    assert cli.main(["ui-profile", str(prof), "--detect", img]) == 0
    p = UiProfile.load(str(prof))
    x0, y0, x1, y1 = synth.log_panel_box(SIZE)
    assert p.pixel_box("log", SIZE) == (round(x0), round(y0), round(x1), round(y1)) and p.region("bank")
    assert cli.main(["ui-profile", str(tmp_path / "t.json"), "--set", "nope=0,0,1,1"]) == 2
    capsys.readouterr()


def test_cli_ocr_and_ocr_teach_with_a_reader(game, tmp_path, capsys, monkeypatch):
    reader = FakeReader()
    s, lines = game[30]
    img = tmp_path / "shot.png"
    render(s, lines, reader, log_cut_top=0.4).save(img)                   # the top entry cut by the panel edge
    img = str(img)
    monkeypatch.setattr(LV, "load_logocr", reader.module)
    lay = reader.frames[digest(img)][0]
    assert lay[0].partial and not lay[-1].partial
    dbg = tmp_path / "dbg.png"
    assert cli.main(["ocr", img, "--debug", str(dbg)]) == 0
    out = capsys.readouterr().out
    kind = L.parse_log_line(lay[-1].canonical).kind
    assert out.startswith("log panel (") and f"[0.97] {lay[-1].canonical}   ({kind})" in out
    assert f"[0.30] {lay[0].canonical}   (" in out and ", partial)" in out
    assert Image.open(dbg).size == SIZE
    assert cli.main(["ocr", img, "--json", "--log-region", "1000,120,265,504"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["box"] == [1000, 120, 1265, 624] and js["lines"][-1]["text"] == lay[-1].canonical
    assert js["lines"][0]["partial"] and js["lines"][-1]["kind"] == kind
    truth = tmp_path / "truth.txt"
    truth.write_text("# the panel, oldest first\n" + "\n".join(e.canonical for e in lay if not e.partial) + "\n")
    prof = tmp_path / "p.json"
    assert cli.main(["ocr-teach", img, "--truth", str(truth), "--ui-profile", str(prof)]) == 0
    out = capsys.readouterr().out
    n = sum(1 for e in lay if not e.partial)
    assert f"before {n}/{n}, after {n}/{n}" in out and "learned: " in out
    assert prof.exists() and UiProfile.load(str(prof)).region("log")


def test_cli_analyze_reads_the_log_locally_and_no_log(frames_dir, tmp_path, capsys, monkeypatch):
    d, reader = frames_dir
    img = str(sorted(d.iterdir())[-1])
    args = [img, "--parser", "cv", "--me", "red", "--depth", "1", "--samples", "1", "--beam", "2", "--model",
            "heuristic"]
    monkeypatch.setattr(LV, "load_logocr", reader.module)
    st = tmp_path / "st.json"
    assert cli.main(["analyze"] + args + ["--session", str(tmp_path / "s.json"), "--save-state", str(st)]) == 0
    out = capsys.readouterr().out
    lay = reader.frames[digest(img)][0]
    saved = json.loads(st.read_text())["parsed"]
    assert saved["log"] == [e.canonical for e in lay if not e.partial] and "== Card count ==" in out
    assert cli.main(["analyze"] + args + ["--no-log", "--save-state", str(st)]) == 0
    capsys.readouterr()
    assert "log" not in json.loads(st.read_text())["parsed"]
    monkeypatch.setitem(sys.modules, "catanbot.vision.logocr", None)
    monkeypatch.setattr(LV, "load_logocr", lambda: None)
    assert cli.main(["analyze"] + args + ["--log-region", "1000,120,265,504", "--save-state", str(st)]) == 0
    out = capsys.readouterr().out
    assert "not available in this installation" in out and "log" not in json.loads(st.read_text())["parsed"]
