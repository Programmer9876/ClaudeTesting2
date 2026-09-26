"""Card counting from the Colonist.io game log (catanbot.colonist_log, the advisor's --session / --game-log)."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
from PIL import Image

from catanbot import actions as A
from catanbot import board as B
from catanbot import cli
from catanbot import colonist_log as L
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, new_game
from catanbot.vision import llm
from catanbot.vision.result import ParseResult
from catanbot.vision.schema import EXAMPLE_PARSED, parsed_to_state, state_to_parsed

NAMES = ["Alice", "Bob", "Carol", "Dave"]
RES = ("wood", "brick", "sheep", "wheat", "ore")


def counts(**kw):
    return [kw.get(r, 0) for r in RES]


# ---------------------------------------------------------------------------
# phrase table and card notation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ph", L.PHRASES, ids=lambda ph: ph.example)
def test_every_phrase_example_parses_to_its_kind(ph):
    ev = L.parse_log_line(ph.example)
    assert ev.kind == ph.kind and not ev.problem, ev.to_dict()


@pytest.mark.parametrize("line,expect", [
    ("Alice rolled 5 3", dict(kind="roll", player="Alice", value=8)),
    ("Alice rolled 12", dict(kind="roll", value=12)),
    ("Bob got 2 wood, 1 ore", dict(kind="gain", player="Bob", cards=counts(wood=2, ore=1))),
    ("Bob got lumber lumber grain", dict(kind="gain", cards=counts(wood=2, wheat=1))),
    ("Bob got WWB", dict(kind="gain", cards=counts(wood=2, brick=1))),
    ("Bob got 2 cards", dict(kind="gain", count=2, cards=None)),
    ("Alice received starting resources wood brick ore", dict(kind="gain", setup=True, cards=counts(wood=1, brick=1, ore=1))),
    ("Alice placed a Settlement", dict(kind="build", item="settlement", free=True)),
    ("Alice placed a Road", dict(kind="build", item="road", free=True)),
    ("Alice built a City", dict(kind="build", item="city", free=False)),
    ("Bob bought Development Card", dict(kind="buy_dev", player="Bob")),
    ("Bob used Knight", dict(kind="play_dev", item="knight")),
    ("Bob used Road Building", dict(kind="play_dev", item="road_building")),
    ("Bob played Year of Plenty", dict(kind="play_dev", item="year_of_plenty")),
    ("Bob used Victory Point", dict(kind="play_dev", item="victory_point")),
    ("Bob used Monopoly", dict(kind="play_dev", item="monopoly")),
    ("Bob stole 5 ore", dict(kind="monopoly", count=5, resource=B.ORE)),
    ("Bob used Monopoly and stole 3 grain", dict(kind="monopoly", count=3, resource=B.WHEAT)),
    ("Bob took from bank wood ore", dict(kind="year_of_plenty", cards=counts(wood=1, ore=1))),
    ("Bob used Year of Plenty and took 2 brick", dict(kind="year_of_plenty", cards=counts(brick=2))),
    ("Bob gave bank 4 wood and took 1 ore", dict(kind="bank_trade", cards=counts(wood=4), get=counts(ore=1))),
    ("Bob gave 2 sheep and got 1 wheat from bank", dict(kind="bank_trade", cards=counts(sheep=2), get=counts(wheat=1))),
    ("Bob traded 3 wool for 1 ore with the bank", dict(kind="bank_trade", cards=counts(sheep=3), get=counts(ore=1))),
    ("Alice traded 1 wood for 1 ore with Bob", dict(kind="player_trade", player="Alice", other="Bob",
                                                     cards=counts(wood=1), get=counts(ore=1))),
    ("Alice wants to give 1 wood for 1 ore", dict(kind="offer", cards=counts(wood=1), get=counts(ore=1))),
    ("Bob counter-offered 1 ore for 2 wood", dict(kind="counter", cards=counts(ore=1), get=counts(wood=2))),
    ("Carol stole a card from Bob", dict(kind="steal", player="Carol", other="Bob", cards=None)),
    ("You stole ore from Bob", dict(kind="steal", player="You", other="Bob", cards=counts(ore=1))),
    ("Bob stole wood from you", dict(kind="steal", other="you", cards=counts(wood=1))),
    ("Bob discarded 4 cards", dict(kind="discard", count=4, cards=None)),
    ("You discarded 2 wood, 2 ore", dict(kind="discard", cards=counts(wood=2, ore=2))),
    ("Bob moved Robber to 6 wheat", dict(kind="robber", player="Bob")),
    ("Bob ended their turn", dict(kind="turn")),
    ("Bob received Largest Army", dict(kind="ignored")),
    ("  - Bob got 2 wood.", dict(kind="gain", player="Bob", cards=counts(wood=2))),
    ("Big Bob Jr got 1 ore", dict(kind="gain", player="Big Bob Jr", cards=counts(ore=1))),
])
def test_phrase_table_events(line, expect):
    ev = L.parse_log_line(line)
    for k, v in expect.items():
        assert getattr(ev, k) == v, (line, k, ev.to_dict())
    assert not ev.problem, ev.to_dict()


@pytest.mark.parametrize("line,kind,problem", [
    ("Happy settling!", "unknown", "no phrase"),
    ("Bob got", "gain", "cards missing"),                     # the card icons were lost in the paste
    ("Bob built a", "build", "building type"),
    ("Bob stole 5", "monopoly", "monopoly resource"),
    ("Bob gave bank and took", "bank_trade", "cards missing"),
    ("Bob got 2 gold", "gain", "not cards: gold"),
    ("Bob rolled 9 9", "roll", "dice"),
])
def test_unknown_and_partial_lines_are_reported(line, kind, problem):
    ev = L.parse_log_line(line)
    assert ev.kind == kind and problem in ev.problem, ev.to_dict()


def test_card_notation_round_trips():
    for text, want in [("2 wood, 1 brick", counts(wood=2, brick=1)), ("wood wood brick", counts(wood=2, brick=1)),
                       ("WWB", counts(wood=2, brick=1)), ("wood x2", counts(wood=2)), ("2x wood", counts(wood=2)),
                       ("two ore and a wool", counts(ore=2, sheep=1)), ("G O S", counts(wheat=1, ore=1, sheep=1))]:
        cp = L.parse_cards(text)
        assert cp.ok and cp.counts == want and cp.unknown == 0, text
    assert L.parse_cards("a card").unknown == 1 and L.parse_cards("?").unknown == 1
    assert L.parse_cards("").empty and not L.parse_cards("").ok
    assert L.parse_cards("gold").bad == ["gold"]
    rng = random.Random(3)
    for style in L.LogRenderer.STYLES:
        for _ in range(20):
            c = [rng.randint(0, 3) for _ in range(5)]
            if any(c):
                assert L.parse_cards(L.format_cards(c, style)).counts == c, (style, c)


def test_parse_log_text_keeps_every_nonempty_line():
    text = "Alice rolled 5 3\n\nBob got 1 ore\nsomething odd\n"
    evs = L.parse_log_text(text)
    assert [e.kind for e in evs] == ["roll", "gain", "unknown"]


def test_events_from_json_schema_and_conveniences():
    entries = [
        {"kind": "roll", "player": "red", "value": 8, "gains": {"blue": {"ore": 2}}},
        {"kind": "play_dev", "player": "blue", "item": "Monopoly", "resource": "wheat", "count": 3},
        {"kind": "steal", "player": "blue", "other": "red", "cards": {"unknown": 1}},
        {"kind": "steal", "player": "red", "other": "blue", "cards": {"ore": 1}},
        {"kind": "other", "text": "gg"},
        {"kind": "teleport"},
        {"kind": "gain", "player": "red"},
        "Bob got 1 wood",
    ]
    evs, warns = L.events_from_json(entries)
    kinds = [e.kind for e in evs]
    assert kinds == ["roll", "gain", "play_dev", "monopoly", "steal", "steal", "ignored", "unknown", "gain", "gain"]
    assert evs[1].player == "blue" and evs[1].cards == counts(ore=2)
    assert evs[2].item == "monopoly" and evs[2].resource is None
    assert evs[3].resource == B.WHEAT and evs[3].count == 3
    assert evs[4].cards is None and evs[5].cards == counts(ore=1)
    assert "cards not readable" in evs[8].problem
    assert L.events_from_json("nope")[1]
    # to_dict / events_from_json round trip
    back, _ = L.events_from_json([e.to_dict() for e in evs if e.kind != "unknown"])
    assert [b.to_dict() for b in back] == [e.to_dict() for e in evs if e.kind != "unknown"]


# ---------------------------------------------------------------------------
# a small hand-made game
# ---------------------------------------------------------------------------
def small_state(sizes, me_hand, names=("Alice", "Bob", "Carol")):
    """A parsed-screenshot state of 3 players (red = me) with the given hand sizes."""
    parsed = json.loads(json.dumps(EXAMPLE_PARSED))
    parsed["players"] = parsed["players"][:3]
    for p, nm, n in zip(parsed["players"], names, sizes):
        p["name"] = nm
        p["cards"] = n
        p.pop("resources", None)
    parsed["players"][0]["resources"] = dict(zip(RES, me_hand))
    parsed["me"] = "red"
    parsed.pop("bank", None)
    return parsed_to_state(parsed)


SETUP = """Alice placed a Settlement
Alice placed a Road
Bob placed a Settlement
Bob placed a Road
Carol placed a Settlement
Carol placed a Road
Carol placed a Settlement
Carol received starting resources wood wheat ore
Carol placed a Road
Bob placed a Settlement
Bob received starting resources sheep sheep ore
Bob placed a Road
Alice placed a Settlement
Alice received starting resources brick wheat
Alice placed a Road"""


def test_tracker_from_the_start_is_exact_then_branches_on_a_hidden_steal():
    st = small_state([2, 3, 3], [0, 1, 0, 1, 0])
    tr = L.ColonistLogTracker.for_state(st, 0)
    tr.update([L.parse_log_text(SETUP)], st)
    assert tr.origin == "start" and tr.counter.num_hypotheses == 1 and not tr.warnings
    assert tr.counter.most_likely() == ((0, 1, 0, 1, 0), (0, 0, 2, 0, 1), (1, 0, 0, 1, 1))
    more = SETUP + "\nCarol rolled 3 4\nCarol moved Robber to 8 ore\nCarol stole a card from Bob"   # hidden to Alice
    st = small_state([2, 2, 4], [0, 1, 0, 1, 0])
    tr.update([L.parse_log_text(more)], st)
    assert tr.new_entries == 3 and tr.counter.num_hypotheses == 2 and not tr.warnings
    assert tr.counter.weight_of([(0, 1, 0, 1, 0), (0, 0, 1, 0, 1), (1, 0, 1, 1, 1)]) == pytest.approx(2 / 3)
    rep = tr.report()
    carol = rep["players"]["orange"]
    assert carol["certain"] == {"wood": 1, "wheat": 1, "ore": 1} and carol["uncertain"] == 1
    assert carol["uncertain_probs"] == {"sheep": pytest.approx(0.667, abs=1e-3), "ore": pytest.approx(0.333, abs=1e-3)}
    line = next(x for x in rep["lines"] if x.strip().startswith("orange (Carol)"))
    assert "1 wood, 1 wheat, 1 ore certain; 1 card uncertain from a hidden steal with blue (Bob): sheep 67% / ore 33%" in line
    # Bob pays 2 sheep: he still held both, so the stolen card was the ore
    tr.update([L.parse_log_text(more + "\nBob gave bank 2 sheep and took 1 wheat")], small_state([2, 1, 4], [0, 1, 0, 1, 0]))
    assert tr.counter.num_hypotheses == 1 and tr.report()["players"]["orange"]["uncertain"] == 0
    assert tr.counter.most_likely()[1:] == ((0, 0, 0, 1, 0), (1, 0, 0, 1, 2))
    assert not tr.reasons[1] and not tr.reasons[2]


def test_unknown_player_names_and_unknown_lines_are_reported_once():
    st = small_state([2, 3, 3], [0, 1, 0, 1, 0])
    tr = L.ColonistLogTracker.for_state(st, 0)
    text = SETUP + "\nKelsey got 1 ore\nHappy settling!"
    tr.update([L.parse_log_text(text)], st)
    assert any("unknown player 'Kelsey'" in w for w in tr.warnings)
    assert any("not understood" in w and "Happy settling" in w for w in tr.warnings)
    tr.update([L.parse_log_text(text)], st)          # the same window again: nothing new, nothing re-reported
    assert tr.new_entries == 0 and not tr.warnings


def test_a_name_learned_late_keeps_the_windows_aligned():
    """The CV parser knows colours only: Kelsey's lines are skipped until her name is given, and the
    entries already consumed still align with the next window afterwards."""
    text = SETUP.replace("Bob", "Kelsey")
    st = small_state([2, 3, 3], [0, 1, 0, 1, 0], names=("Alice", "blue", "Carol"))
    tr = L.ColonistLogTracker.for_state(st, 0)
    tr.update([L.parse_log_text(text)], st)
    assert any("unknown player 'Kelsey'" in w for w in tr.warnings)
    assert any(w.startswith("blue holds 3 card(s) on screen but 0") for w in tr.warnings)
    st = small_state([2, 4, 3], [0, 1, 0, 1, 0], names=("Alice", "Kelsey", "Carol"))   # --fix blue.name=Kelsey
    tr.update([L.parse_log_text(text + "\nAlice rolled 4 4\nKelsey got 1 wood")], st)
    assert tr.new_entries == 2 and not tr.warnings, tr.warnings
    assert tr.counter.size[1] == 4 and tr.label(1) == "blue (Kelsey)"


def test_mid_game_start_is_a_prior_fitted_to_the_hand_sizes():
    st = small_state([3, 5, 2], [1, 0, 0, 1, 1])
    tr = L.ColonistLogTracker.for_state(st, 0)
    tr.update([L.parse_log_text("Bob rolled 2 4\nBob got 1 ore\nCarol got 1 wheat")], st)
    assert tr.origin == "mid-game" and tr.counter.size == [3, 5, 2] and not tr.warnings
    rep = tr.report()
    assert rep["lines"][0].startswith("Session started mid-game")
    # the window's readable stretch was applied on top of the prior: they hold the cards they just got
    assert rep["players"]["blue"]["certain"].get("ore", 0) >= 1
    assert rep["players"]["orange"]["certain"].get("wheat", 0) >= 1
    assert "the mid-game start" in rep["players"]["blue"]["reasons"]
    # small hands: the prior holds every possible hand, so it is an exact posterior (not "estimated")
    assert not any(tr.estimated)
    # the next window: a city Bob can pay in some hypotheses is plain conditioning
    tr.update([L.parse_log_text("Carol got 1 wheat\nBob built a City")], small_state([3, 0, 2], [1, 0, 0, 1, 1]))
    assert tr.counter.size == [3, 0, 2] and not tr.warnings, tr.warnings


def test_estimated_hands_are_redrawn_but_known_hands_report_a_missed_entry():
    st = small_state([3, 5, 2], [1, 0, 0, 1, 1])
    for estimated in (True, False):
        tr = L.ColonistLogTracker.for_state(st, 0)
        tr.update([L.parse_log_text("Bob rolled 2 4\nBob got 1 ore\nCarol got 1 wheat")], st)
        # the prior (or the count) had Bob without the wheat and ore for a city
        tr.counter.hyps = {((1, 0, 0, 1, 1), (2, 1, 1, 0, 1), (1, 0, 0, 1, 0)): 1.0}
        tr.counter._marg = None
        tr.estimated[1] = estimated
        tr.update([L.parse_log_text("Carol got 1 wheat\nBob built a City")], small_state([3, 0, 2], [1, 0, 0, 1, 1]))
        assert tr.counter.size == [3, 0, 2]
        if estimated:      # a wrong estimate, not a log problem: the 4 other cards were wheat and ore
            assert not tr.warnings and tr.counter.num_hypotheses == 1
        else:              # a counted hand that cannot pay: a missed entry, repaired and reported
            assert any("4 card(s) short" in w for w in tr.warnings) and tr.stats["repairs"] >= 1


def test_session_round_trip_and_refusals(tmp_path):
    st = small_state([2, 3, 3], [0, 1, 0, 1, 0])
    tr = L.ColonistLogTracker.for_state(st, 0)
    tr.update([L.parse_log_text(SETUP + "\nCarol rolled 3 4\nCarol stole a card from Bob")], small_state([2, 2, 4], [0, 1, 0, 1, 0]))
    path = tmp_path / "s.json"
    tr.save(str(path))
    back = L.ColonistLogTracker.open_session(str(path), small_state([2, 2, 4], [0, 1, 0, 1, 0]), 0)
    assert back.counter.hyps == tr.counter.hyps and back.tail == tr.tail and back.reasons == tr.reasons
    assert back.to_dict() == tr.to_dict()
    with pytest.raises(L.SessionError):
        L.ColonistLogTracker.open_session(str(path), small_state([2, 2, 4], [0, 1, 0, 1, 0]), 1)   # another seat
    other = parsed_to_state(EXAMPLE_PARSED)                                                       # another game
    with pytest.raises(L.SessionError):
        L.ColonistLogTracker.open_session(str(path), other, 0)
    junk = tmp_path / "profiles.json"
    junk.write_text(json.dumps({"profiles": {}}))
    with pytest.raises(L.SessionError, match="refusing to overwrite"):
        L.ColonistLogTracker.open_session(str(junk), st, 0)


# ---------------------------------------------------------------------------
# overlapping windows
# ---------------------------------------------------------------------------
GAME = SETUP + """
Alice rolled 4 4
Bob got 1 wood
Carol got 1 brick
Bob rolled 2 1
Carol got 1 ore
Carol rolled 6 2
Bob got 1 wood
Carol got 1 brick
Carol built a Road
Alice rolled 5 3
Bob got 1 wood
Carol got 1 brick
Bob gave bank 3 wood and took 1 brick"""


def game_sizes(text):
    """(hand sizes, my hand) after the lines of ``text`` (a full count of the plain GAME)."""
    st = small_state([2, 3, 3], [0, 1, 0, 1, 0])
    tr = L.ColonistLogTracker.for_state(st, 0)
    evs = L.parse_log_text(text)
    tr.counter = None
    tr._state = st
    tr._start(evs, [tr._key(e) for e in evs], None)
    return list(tr.counter.size), tr.counter.most_likely()


def screen(text):
    sizes, hands = game_sizes(text)
    return small_state(sizes, list(hands[0]))


def feed(windows):
    lines = GAME.splitlines()
    tr = None
    for a, b in windows:
        st = screen("\n".join(lines[:b]))
        tr = tr or L.ColonistLogTracker.for_state(st, 0)
        tr.update([L.parse_log_text("\n".join(lines[a:b]))], st)
        assert not tr.warnings, (a, b, tr.warnings)
    return tr


def test_identical_and_shifted_windows_count_every_entry_once():
    lines = GAME.splitlines()
    whole = feed([(0, len(lines))])
    shifted = feed([(0, 18), (12, 22), (15, 25), (20, len(lines))])
    twice = feed([(0, 20), (0, 20), (5, 20), (0, len(lines)), (0, len(lines))])
    for tr in (shifted, twice):
        assert tr.counter.hyps == whole.counter.hyps
        assert tr.entries == whole.entries == len(lines)
    assert whole.counter.most_likely() == ((0, 1, 0, 1, 0), (0, 1, 2, 0, 1), (0, 2, 0, 1, 2))
    assert whole.counter.num_hypotheses == 1


def test_window_inside_the_consumed_log_adds_nothing():
    lines = GAME.splitlines()
    tr = feed([(0, len(lines))])
    before = dict(tr.counter.hyps)
    tr.update([L.parse_log_text("\n".join(lines[16:21]))], screen(GAME))    # scrolled up
    assert tr.new_entries == 0 and tr.counter.hyps == before
    assert any("inside the part already consumed" in w for w in tr.warnings)


def test_repeated_lines_are_aligned_by_the_hand_sizes():
    base = SETUP + "\nAlice rolled 4 4\nBob got 1 wood\nAlice rolled 4 4\nBob got 1 wood"
    tr = L.ColonistLogTracker.for_state(screen(base), 0)
    tr.update([L.parse_log_text(base)], screen(base))
    more = base + "\nAlice rolled 4 4\nBob got 1 wood"
    # the window shows the last 4 lines: the longest overlap (all 4 already seen) contradicts Bob's
    # hand size on screen, the 2-line overlap does not
    window = "\n".join(more.splitlines()[-4:])
    tr.update([L.parse_log_text(window)], screen(more))
    assert tr.new_entries == 2 and not tr.warnings, tr.warnings
    assert tr.counter.size[1] == 3 + 3


def test_a_gap_is_reported_and_resynchronised_from_the_hand_sizes():
    lines = GAME.splitlines()
    tr = feed([(0, 16)])
    # the next window starts after 4 lines that were never seen (production for Bob and Carol)
    st = screen(GAME)
    tr.update([L.parse_log_text("\n".join(lines[20:]))], st)
    assert any("shares no entry" in w for w in tr.warnings)
    assert any(w.startswith("blue (Bob) traded with the bank but the count had them 1 card(s) short") for w in tr.warnings)
    assert any(w.startswith("orange (Carol) holds 5 card(s) on screen but 3") for w in tr.warnings)
    assert tr.counter.size == [sum(p.resources) if p.hand_known else p.hand_size for p in st.players]
    truth = game_sizes(GAME)[1]
    assert tr.counter.weight_of(truth) > 0 and tr.stats["gaps"] == 1


# ---------------------------------------------------------------------------
# end to end: engine games rendered as Colonist logs
# ---------------------------------------------------------------------------
def play(seed, reveal=False, style="mixed"):
    """Yields ``(pre, action, post, lines)`` after every action of a heuristic-bot game (seat 0 = me)."""
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    for p, nm in zip(s.players, NAMES):
        p.name = nm
    s.allow_counters = bool(seed % 2)
    bots = [HeuristicBot(temperature=0.3) for _ in range(4)]
    rend = L.LogRenderer(0, NAMES, rng=random.Random(seed + 7), style=style, reveal_hidden=reveal)
    lines = []
    while s.phase != PHASE_GAME_OVER:
        i = E.acting_player(s)
        a = bots[i].decide(s, E.legal_actions(s), rng)
        pre = s.copy()
        s = E.apply_inplace(s, a, rng)
        lines.extend(rend.render(pre, a, s))
        yield pre, a, s, lines


def shot(s, bank=True):
    parsed = state_to_parsed(s, me=0, include_bank=bank)
    return parsed_to_state(parsed), ([parsed["bank"][r] for r in RES] if bank else None)


def hidden_events(pre, a):
    """Seats a hidden steal / discard touched (hidden to seat 0)."""
    if a[0] in (A.MOVE_ROBBER, A.PLAY_KNIGHT) and a[2] >= 0 and 0 not in (pre.current, a[2]):
        return {pre.current, a[2]}
    if a[0] == A.DISCARD and pre.discard_queue[0] != 0:
        return {pre.discard_queue[0]}
    return set()


@pytest.mark.parametrize("seed", [1, 2, 4])
def test_engine_game_through_overlapping_windows(seed):
    """The true hands are always among the hypotheses; a hand no hidden steal / discard touched is exact."""
    tr = None
    touched = set()
    last_end = 0
    wrng = random.Random(seed + 3)
    checks = 0
    seen_uncertain = False
    for pre, a, s, lines in play(seed):
        touched |= hidden_events(pre, a)
        if wrng.random() < 0.15 and len(lines) > last_end:
            start = max(0, last_end - wrng.randint(1, 8)) if last_end else 0
            st, bank = shot(s)
            tr = tr or L.ColonistLogTracker.for_state(st, 0)
            tr.update([L.parse_log_text("\n".join(lines[start:]))], st, bank)
            truth = [p.resources for p in s.players]
            assert not tr.warnings, tr.warnings
            assert tr.counter.weight_of(truth) > 0
            for j in range(4):
                if j not in touched:
                    assert tr.is_exact(j) and list(tr.counter.most_likely()[j]) == truth[j]
            seen_uncertain |= tr.counter.num_hypotheses > 1
            last_end = len(lines)
            checks += 1
    assert checks > 50 and tr.stats["hidden_steals"] > 0 and seen_uncertain
    assert tr.stats["unknown_lines"] == 0 and tr.stats["repairs"] == 0 and tr.stats["gaps"] == 0


def test_engine_game_with_every_hidden_card_revealed_is_exact_throughout():
    tr = None
    wrng = random.Random(5)
    last_end = 0
    for pre, a, s, lines in play(3, reveal=True):
        if wrng.random() < 0.2 and len(lines) > last_end:
            st, bank = shot(s, bank=False)
            tr = tr or L.ColonistLogTracker.for_state(st, 0)
            tr.update([L.parse_log_text("\n".join(lines[max(0, last_end - 3):]))], st, bank)
            assert tr.counter.num_hypotheses == 1 and tr.counter.weight_of([p.resources for p in s.players]) == 1.0
            last_end = len(lines)


def test_hand_size_reconciliation_catches_a_dropped_entry():
    """A production line lost from the window: the hand size on screen shows it, the count branches the
    missing cards in (the true hand stays possible) and says so."""
    tr = None
    last_end = prev = 0
    for pre, a, s, lines in play(2, style="counts"):
        new, prev = lines[prev:], len(lines)
        if tr is None:
            if s.turn >= 8 and s.phase == PHASE_MAIN:
                st, _ = shot(s, bank=False)
                tr = L.ColonistLogTracker.for_state(st, 0)
                tr.update([L.parse_log_text("\n".join(lines))], st)
                assert not tr.warnings
                last_end = len(lines)
            continue
        dropped = next((ln for ln in new if a[0] == A.ROLL and " got " in ln and not ln.startswith("Alice")), None)
        if dropped is None:
            continue
        victim = NAMES.index(dropped.split(" got ")[0])
        st, _ = shot(s, bank=False)
        tr.update([L.parse_log_text("\n".join(ln for ln in lines[last_end - 2:] if ln != dropped))], st)
        label = f"{st.players[victim].color} ({NAMES[victim]})"
        assert any(w.startswith(f"{label} holds") and "missed or misread" in w for w in tr.warnings), tr.warnings
        assert tr.counter.size[victim] == st.players[victim].hand_size
        assert tr.counter.weight_of([p.resources for p in s.players]) > 0
        assert "a missed log entry" in tr.reasons[victim]
        break
    else:
        pytest.fail("no production of an opponent to drop")


def test_determinizations_follow_the_count():
    """The search's sampled states hold the count's hands (and the screenshot conventions)."""
    tr = None
    for pre, a, s, lines in play(1):
        if s.turn >= 30 and s.phase == PHASE_MAIN and s.current == 0:
            st, bank = shot(s, bank=False)
            tr = L.ColonistLogTracker.for_state(st, 0)
            tr.update([L.parse_log_text("\n".join(lines))], st)
            break
    pub = tr.public_view(st)
    rng = random.Random(0)
    for _ in range(5):
        d = tr.determinize(pub, rng)
        hands = [tuple(p.resources) for p in d.players]
        assert tr.counter.weight_of(hands) > 0
        assert all(p.hand_known and p.dev_known for p in d.players)
        assert d.bank == [19 - sum(h[r] for h in hands) for r in range(5)]
        assert [p.dev_count for p in d.players] == [p.dev_count if not p.dev_known else p.total_dev for p in st.players]
        assert E.legal_actions(d)


# ---------------------------------------------------------------------------
# screenshots: the Claude-vision parser transcribes the log (mocked client)
# ---------------------------------------------------------------------------
def fake_response(tool_input):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="t1", name=llm.TOOL_NAME, input=tool_input)],
                           stop_reason="tool_use", stop_details=None, model="claude-opus-5",
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._response = response

    def _create(self, **kw):
        self.calls.append(kw)
        return self._response


LLM_LOG = [
    {"kind": "build", "player": "red", "item": "settlement", "free": True, "text": "Alice placed a Settlement"},
    {"kind": "build", "player": "blue", "item": "settlement", "free": True},
    {"kind": "gain", "player": "blue", "cards": {"ore": 1, "sheep": 2}, "setup": True},
    {"kind": "gain", "player": "red", "cards": {"brick": 1, "wheat": 1}, "setup": True},
    {"kind": "roll", "player": "red", "value": 8, "gains": {"blue": {"ore": 1}}},
    {"kind": "robber", "player": "red"},
    {"kind": "steal", "player": "red", "other": "blue", "cards": {"ore": 1}, "text": "You stole ore from Bob"},
    {"kind": "other", "text": "Alice: gl hf"},
]


def llm_tool_input(log=None):
    parsed = json.loads(json.dumps(EXAMPLE_PARSED))
    parsed["players"] = parsed["players"][:2]
    for p in parsed["players"]:
        p["settlements"], p["cities"], p["roads"] = [], [], []
    parsed["players"][0]["resources"] = {"brick": 1, "wheat": 1, "ore": 1}
    parsed["players"][0]["cards"] = 3
    parsed["players"][1]["cards"] = 3
    parsed["confidence"] = {"overall": 0.9}
    parsed.pop("ports")
    if log is not None:
        parsed["log"] = log
    return parsed


def test_llm_log_is_requested_only_with_read_log_and_feeds_the_tracker():
    img = Image.new("RGB", (320, 200), (20, 90, 160))
    client = FakeClient(fake_response(llm_tool_input()))
    llm.parse_with_claude(img, client=client)
    req = client.calls[0]
    assert "log" not in req["tools"][0]["input_schema"]["properties"] and "Game log" not in req["system"]
    assert req["tools"][0] == llm.build_tool() and req["system"] == llm.build_prompt()

    client = FakeClient(fake_response(llm_tool_input(LLM_LOG)))
    result = llm.parse_with_claude(img, client=client, read_log=True)
    req = client.calls[0]
    schema = req["tools"][0]["input_schema"]["properties"]["log"]
    assert schema["items"]["properties"]["kind"]["enum"][0] == "roll"
    assert "Game log" in req["system"] and "log" in req["messages"][0]["content"][1]["text"]
    log = result.parsed["log"]
    assert [e["kind"] for e in log] == ["build", "build", "gain", "gain", "roll", "gain", "robber", "steal", "ignored"]
    tr = L.ColonistLogTracker.for_state(result.state, 0)
    events, warns = L.events_from_json(log)
    tr.update([events], result.state)
    assert not warns and not tr.warnings, tr.warnings
    assert tr.origin == "start" and tr.counter.num_hypotheses == 1
    assert tr.counter.most_likely()[1] == (0, 0, 2, 0, 1)
    # a model that did not transcribe the log: a warning, no crash
    client = FakeClient(fake_response(llm_tool_input()))
    result = llm.parse_with_claude(img, client=client, read_log=True)
    assert any("did not transcribe the game log" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# the advisor
# ---------------------------------------------------------------------------
ARGS = ["--model", "heuristic", "--beam", "2", "--samples", "2", "--depth", "1"]


def test_advisor_unchanged_without_the_new_options(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("card counting must stay off")

    monkeypatch.setattr(L.ColonistLogTracker, "__init__", boom)
    parsed = llm_tool_input(LLM_LOG)          # a log in the parse is ignored without --session / --game-log
    parsed.pop("confidence")
    path = tmp_path / "p.json"
    path.write_text(json.dumps(parsed))
    assert cli.main(["recommend", "--state", str(path)] + ARGS) == 0
    out = capsys.readouterr().out
    assert "Card count" not in out
    assert cli.main(["recommend", "--state", str(path), "--json"] + ARGS) == 0
    d = json.loads(capsys.readouterr().out)
    assert "card_count" not in d and "card_count" not in d["advice"]
    assert cli.build_parser().parse_args(["recommend", "--state", "x"]).session is None


def test_advisor_session_from_the_parse_log_and_game_log(tmp_path, capsys):
    parsed = llm_tool_input(LLM_LOG)
    parsed.pop("confidence")
    path = tmp_path / "p.json"
    path.write_text(json.dumps(parsed))
    sess = tmp_path / "s.json"
    assert cli.main(["recommend", "--state", str(path), "--session", str(sess)] + ARGS) == 0
    out = capsys.readouterr().out
    assert "== Card count ==" in out and "Counting from the start of the game: 8 log entries (8 new)" in out
    assert "blue (Bob): 2 sheep, 1 ore (exact, 3 cards)" in out
    assert out.index("== Card count ==") < out.index("== Trading ==")
    assert json.loads(sess.read_text())["format"] == L.ColonistLogTracker.FORMAT
    # the next call with the same screenshot: nothing new; a pasted log adds its new lines
    text = tmp_path / "log.txt"
    text.write_text("You stole ore from Bob\nAlice rolled 4 4\nBob got 1 ore\nHappy settling!\n")
    assert cli.main(["recommend", "--state", str(path), "--session", str(sess), "--json"] + ARGS) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["card_count"]["new_entries"] == 0 and d["advice"]["card_count"]
    assert cli.main(["recommend", "--state", str(path), "--game-log", str(text), "--session", str(sess), "--json",
                     "--fix", "blue.cards=4"] + ARGS) == 0
    d = json.loads(capsys.readouterr().out)
    cc = d["card_count"]
    assert cc["new_entries"] == 2 and cc["players"]["blue"]["most_likely"] == {"sheep": 2, "ore": 2}
    assert any("Happy settling" in w for w in cc["warnings"])
    # another game in the same session file is refused
    other = tmp_path / "other.json"
    other.write_text(json.dumps(EXAMPLE_PARSED))
    assert cli.main(["recommend", "--state", str(other), "--session", str(sess)] + ARGS) == 2
    assert "follows a game with players" in capsys.readouterr().err


def test_advisor_game_log_without_session_and_name_fix(tmp_path, capsys):
    parsed = llm_tool_input()
    parsed.pop("confidence")
    for p in parsed["players"]:
        p["name"] = p["color"]            # the CV parser knows colours only
    path = tmp_path / "p.json"
    path.write_text(json.dumps(parsed))
    text = tmp_path / "log.txt"
    text.write_text("Kelsey placed a Settlement\nKelsey received starting resources sheep sheep ore\n"
                    "red received starting resources brick wheat ore\n")
    assert cli.main(["recommend", "--state", str(path), "--game-log", str(text)] + ARGS) == 0
    out = capsys.readouterr().out
    assert "unknown player 'Kelsey'" in out
    assert cli.main(["recommend", "--state", str(path), "--game-log", str(text), "--fix", "blue.name=Kelsey"] + ARGS) == 0
    out = capsys.readouterr().out
    assert "blue (Kelsey): 2 sheep, 1 ore (exact, 3 cards)" in out and "unknown player" not in out


def test_analyze_with_session_asks_the_llm_for_the_log(tmp_path, capsys, monkeypatch):
    calls = []

    def fake(image, me=None, read_log=False):
        calls.append(read_log)
        parsed = llm_tool_input(LLM_LOG if read_log else None)
        parsed.pop("confidence")
        return ParseResult(parsed=parsed, state=parsed_to_state(parsed))

    monkeypatch.setattr(llm, "parse_with_claude", fake)
    img = tmp_path / "shot.png"
    Image.new("RGB", (64, 64)).save(img)
    sess = tmp_path / "s.json"
    assert cli.main(["analyze", str(img), "--parser", "llm", "--session", str(sess)] + ARGS) == 0
    assert "== Card count ==" in capsys.readouterr().out and sess.exists()
    assert cli.main(["analyze", str(img), "--parser", "llm"] + ARGS) == 0
    assert "Card count" not in capsys.readouterr().out
    assert calls == [True, False]


# Colonist.io's own wordings, as an open-source Colonist log tracker (glasperfan/explorer) matches them:
# colons after the verbs, card icons, "stole all of" for a Monopoly (no total), discards with their cards.
COLONIST_WORDINGS = [
    ("Bob got: 1 wood, 2 ore", dict(kind="gain", cards=[1, 0, 0, 0, 2])),
    ("Bob gave bank: 4 wood and took 1 ore", dict(kind="bank_trade", cards=[4, 0, 0, 0, 0], get=[0, 0, 0, 0, 1])),
    ("Bob wants to give: 1 wood for: 1 ore", dict(kind="offer", cards=[1, 0, 0, 0, 0], get=[0, 0, 0, 0, 1])),
    ("Bob stole: 1 wood from: you", dict(kind="steal", cards=[1, 0, 0, 0, 0], other="you")),
    ("Bob stole a card from: Carol", dict(kind="steal", cards=None, other="Carol")),
    ("Bob discarded 2 wood, 2 ore", dict(kind="discard", cards=[2, 0, 0, 0, 2])),
    ("Bob stole all of: 1 ore", dict(kind="monopoly", count=None, resource=B.ORE)),
    ("Bob stole all of wool", dict(kind="monopoly", count=None, resource=B.SHEEP)),
    ("Bob stole 5 ore", dict(kind="monopoly", count=5, resource=B.ORE)),
    ("Giving out starting resources", dict(kind="ignored")),
    ("Bob's turn to place", dict(kind="ignored")),
]


@pytest.mark.parametrize("line,want", COLONIST_WORDINGS)
def test_colonist_wordings(line, want):
    ev = L.parse_log_line(line)
    for k, v in want.items():
        assert getattr(ev, k) == v, (line, k, getattr(ev, k))
    assert not ev.problem


def test_monopoly_without_a_total_is_split_by_the_hand_sizes():
    st = small_state([2, 3, 3], [0, 1, 0, 1, 0])
    tr = L.ColonistLogTracker.for_state(st, 0)
    tr.update([L.parse_log_text(SETUP)], st)
    assert tr.counter.most_likely() == ((0, 1, 0, 1, 0), (0, 0, 2, 0, 1), (1, 0, 0, 1, 1))
    # Colonist writes "stole all of [ore]" without the total: Carol's one ore moves to Bob
    more = SETUP + "\nBob rolled 3 4\nBob used Monopoly\nBob stole all of: 1 ore"
    st = small_state([2, 4, 2], [0, 1, 0, 1, 0])
    tr.update([L.parse_log_text(more)], st)
    assert tr.counter.num_hypotheses == 1 and not tr.warnings, tr.warnings
    assert tr.counter.most_likely() == ((0, 1, 0, 1, 0), (0, 0, 2, 0, 2), (1, 0, 0, 1, 0))
