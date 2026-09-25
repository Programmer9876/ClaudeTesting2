"""Tests for the opening policies (catanbot/openings.py) and their tunable ``openings.policy``."""
import random

import pytest

from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot import heuristic, openings, placement, tuning
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.param_bot import ParamBot
from catanbot.agents.search_bot import SearchBot
from catanbot.search import Searcher
from catanbot.selfplay import make_bot
from catanbot.state import PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT, new_game

SEARCH_SPEC = tuning.DEFAULT_SEARCH_SPEC
ORIGINAL_SEARCH_DECIDE = SearchBot.__dict__["decide"]


def _originals():
    return (heuristic.setup_pick, heuristic.setup_road_pick, SearchBot.__dict__["decide"],
            placement.setup_pick, placement.setup_road_pick, HeuristicBot.__dict__["decide"])


def _assert_pristine():
    assert openings.installed() == "current"
    assert heuristic.setup_pick is placement.setup_pick
    assert heuristic.setup_road_pick is placement.setup_road_pick
    assert SearchBot.__dict__["decide"] is ORIGINAL_SEARCH_DECIDE


@pytest.fixture(autouse=True)
def _clean():
    openings.uninstall()
    yield
    openings.uninstall()
    _assert_pristine()


def _setup_states(num_players, seed):
    """Every state of a setup phase played with random legal moves (both rounds, settlements and roads)."""
    rng = random.Random(seed)
    s = new_game(num_players, rng=random.Random(seed))
    out = []
    while s.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        out.append(s)
        legal = E.legal_actions(s)
        s = E.apply(s, legal[rng.randrange(len(legal))], None)
    return out


# ---------------------------------------------------------------------------
# the policies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_players", [3, 4])
def test_policies_return_legal_spots_in_every_setup_state(num_players):
    seen = set()
    for seed in range(3):
        for s in _setup_states(num_players, seed):
            legal = E.legal_actions(s)
            me = s.current
            seen.add((s.phase, s.setup_round))
            for name in openings.CANDIDATES:
                pol = openings.policy(name)
                if s.phase == PHASE_SETUP_SETTLEMENT:
                    free = {a[1] for a in legal}
                    top = pol.pick(s, me, 5)
                    assert 1 <= len(top) <= 5
                    assert all(v in free for v, _ in top), name
                    scores = [x for _, x in top]
                    assert scores == sorted(scores, reverse=True), name
                    full = pol.pick(s, me, 60)                      # heuristic.action_priors asks for k=60
                    assert {v for v, _ in full} == free, name       # every legal spot gets its own score
                    assert full[:5] == top
                else:
                    e, _ = pol.road(s, me, s.setup_last_settlement)
                    assert (A.SETUP_ROAD, e) in legal, name
                a = openings.choose(s, legal, name)
                assert a in legal, name
    assert seen == {(PHASE_SETUP_SETTLEMENT, 0), (PHASE_SETUP_ROAD, 0), (PHASE_SETUP_SETTLEMENT, 1),
                    (PHASE_SETUP_ROAD, 1)}


def test_snake_order_next_picker():
    s = new_game(4, rng=random.Random(1))
    assert openings.pick_order(4) == [0, 1, 2, 3, 3, 2, 1, 0]
    s.current, s.setup_round = 0, 0
    assert openings.next_opponent_picker(s, 0) == 1
    s.current, s.setup_round = 3, 0                 # seat 3 picks twice in a row: the next *opponent* is 2
    assert openings.next_opponent_picker(s, 3) == 2
    s.current, s.setup_round = 3, 1
    assert openings.next_opponent_picker(s, 3) == 2
    s.current, s.setup_round = 1, 1
    assert openings.next_opponent_picker(s, 1) == 0
    s.current, s.setup_round = 0, 1                 # the very last pick
    assert openings.next_opponent_picker(s, 0) is None
    t = new_game(3, rng=random.Random(1))
    t.current, t.setup_round = 2, 0
    assert openings.next_opponent_picker(t, 2) == 1


def test_denial_reduces_to_setup_pick(monkeypatch):
    s = new_game(4, rng=random.Random(4))
    base = placement.setup_pick(s, 0, k=60)
    bonus = openings.denial_bonus(s, 0, [v for v, _ in base])
    assert all(b >= 0.0 for b in bonus.values()) and max(bonus.values()) > 0.0
    # taking the next picker's favourite spot denies them something
    occ = s.occupied_vertices()
    fav = max((placement.score_settlement_spot(s, 1, v, occ=occ, setup=True), v)
              for v in range(B.NUM_VERTICES) if placement.is_free_vertex(occ, v))[1]
    assert bonus[fav] > 0.0
    monkeypatch.setattr(openings, "DENIAL_WEIGHT", 0.0)
    assert [v for v, _ in openings.denial_pick(s, 0, 60)] == [v for v, _ in sorted(base, key=lambda t: (-t[1], t[0]))]
    # the last pick of the setup has nobody to deny
    monkeypatch.setattr(openings, "DENIAL_WEIGHT", 0.5)
    last = [st for st in _setup_states(4, 2) if st.phase == PHASE_SETUP_SETTLEMENT][-1]
    assert (last.current, last.setup_round) == (0, 1)
    assert openings.denial_pick(last, 0, 60) == sorted(placement.setup_pick(last, 0, k=60), key=lambda t: (-t[1], t[0]))


def test_parametric_denial_weight():
    s = next(st for st in _setup_states(4, 6) if st.phase == PHASE_SETUP_SETTLEMENT and st.setup_round == 1)
    me = s.current
    ranked = sorted(placement.setup_pick(s, me, k=60), key=lambda t: (-t[1], t[0]))
    assert openings.policy("denial:0").pick(s, me, 60) == ranked
    assert openings.policy("denial:1").pick(s, me, 60) == openings.denial_pick(s, me, 60, weight=1.0)
    assert openings.policy("denial:1.0").pick(s, me, 60) == openings.policy("denial").pick(s, me, 60)   # default 1.0
    assert openings.policy("denial:2.5") is openings.policy("denial:2.5")
    for bad in ("denial:x", "denial:-1", "denial:inf", "pips_diversity:2", "nope"):
        with pytest.raises(ValueError):
            openings.policy(bad)
    t = tuning.find("openings.policy")
    assert t.parse("denial:2") == "denial:2"
    with tuning.overridden({"openings.policy": "denial:2"}):
        assert openings.installed() == "denial:2"
        legal = E.legal_actions(s)
        assert openings.choose(s, legal) == openings.choose(s, legal, "denial:2")


def test_current_road_scores_match_setup_road_pick():
    for s in _setup_states(4, 5):
        if s.phase != PHASE_SETUP_ROAD:
            continue
        e, best = placement.setup_road_pick(s, s.current, s.setup_last_settlement)
        scores = openings.current_road_scores(s, s.current, s.setup_last_settlement)
        assert scores[e] == pytest.approx(best) and max(scores.values()) == pytest.approx(best)
        assert openings.policy("setup_pick").road(s, s.current, s.setup_last_settlement) == (e, best)


def test_pips_diversity_second_settlement_complements_the_first():
    s = new_game(4, rng=random.Random(7))
    w = [1.0] * 5
    own_wheat = [0, 0, 0, 9, 0]
    ore_spot = [v for v in range(B.NUM_VERTICES) if openings.vertex_pips_by_resource(s, v)[B.ORE] > 0][0]
    # a resource the first settlement lacks weighs more, and ore next to wheat completes the pair
    assert (openings.pips_diversity_spot(s, ore_spot, own_wheat, w)
            > openings.pips_diversity_spot(s, ore_spot, [0, 0, 0, 0, 0], w))


# ---------------------------------------------------------------------------
# install / uninstall / tunable
# ---------------------------------------------------------------------------
def test_default_untouched_when_not_installed():
    _assert_pristine()
    s = new_game(4, rng=random.Random(1))
    assert openings.choose(s, E.legal_actions(s)) is None          # no policy installed -> no opinion
    assert tuning.apply({"openings.policy": "current"}) == []      # the default installs nothing
    _assert_pristine()


def test_install_uninstall_round_trip():
    before = _originals()
    for name in openings.CANDIDATES:
        openings.install(name)
        assert openings.installed() == name
        assert heuristic.setup_pick is openings._patched_setup_pick
        assert heuristic.setup_road_pick is openings._patched_setup_road_pick
        assert SearchBot.__dict__["decide"] is openings._search_decide
        assert placement.setup_pick is before[3]                     # the real scorer is never patched
        openings.install("denial" if name != "denial" else "standin_book")   # switch: originals stay saved
        assert heuristic.setup_pick is openings._patched_setup_pick
        openings.uninstall()
        assert _originals() == before
        openings.uninstall()                                         # idempotent
        assert _originals() == before
    with pytest.raises(ValueError):
        openings.install("no_such_policy")
    assert _originals() == before
    with openings.using("pips_diversity"):
        assert openings.installed() == "pips_diversity"
        with openings.using("denial"):
            assert openings.installed() == "denial"
        assert openings.installed() == "pips_diversity"
    assert _originals() == before


def test_tunable_registered_and_nests_through_tuning():
    t = tuning.find("openings.policy")
    assert t is tuning.TUNABLES["openings.policy"] and t.default == "current" and t.kind in ("weight", "flag")
    assert set(t.candidates) >= {"standin_book", "pips_diversity", "denial"}
    assert t.parse(" denial ") == "denial" and t.format("denial") == "denial"
    with pytest.raises(ValueError):
        t.parse("bogus")
    assert not t.needs_python_evaluator and not t.requires_search
    assert tuning.verify_registry() == []
    before = _originals()
    tok = tuning.apply({"openings.policy": "denial"})
    try:
        assert openings.installed() == "denial" and [tg.get() for tg in t.targets] == ["denial"]
        with tuning.overridden({"openings.policy": "standin_book"}):
            assert openings.installed() == "standin_book"
        assert openings.installed() == "denial"
    finally:
        tuning.restore(tok)
    assert _originals() == before and [tg.get() for tg in t.targets] == ["current"]


# ---------------------------------------------------------------------------
# both bots follow the installed policy
# ---------------------------------------------------------------------------
def _score_of(name, s, a):
    if a[0] == A.SETUP_SETTLEMENT:
        return dict(openings.policy(name).pick(s, s.current, 60))[a[1]]
    return None


def _check_bot_follows(bot, name, seeds=range(10)):
    decisions = 0
    for seed in seeds:
        s = new_game(4, rng=random.Random(seed))
        rng = random.Random(seed)
        bot.reset()
        with openings.using(name):
            while s.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
                legal = E.legal_actions(s)
                expected = openings.choose(s, legal, name)
                got = bot.decide(s, legal, rng)
                if len(legal) > 1:
                    decisions += 1
                    if got != expected:
                        # the heuristic bot breaks exact prior ties at random: same policy score is fine
                        assert isinstance(bot, HeuristicBot) and got[0] == A.SETUP_SETTLEMENT, (name, seed, got, expected)
                        assert _score_of(name, s, got) == _score_of(name, s, expected)
                s = E.apply(s, got, None)
    assert decisions >= 10 * 15


@pytest.mark.parametrize("name", openings.CANDIDATES)
def test_heuristic_bot_follows_installed_policy(name):
    _check_bot_follows(HeuristicBot(), name)


@pytest.mark.parametrize("name", openings.CANDIDATES)
def test_search_bot_follows_installed_policy_without_searching(name, monkeypatch):
    def no_search(self, *a, **k):
        raise AssertionError("the search must not run in the setup phases while a policy is installed")

    monkeypatch.setattr(Searcher, "search", no_search)
    bot = make_bot(SEARCH_SPEC)
    _check_bot_follows(bot, name)
    assert bot.last_results and name in bot.last_results[0].explanation


def test_search_bot_outside_setup_and_rng_stream(monkeypatch):
    """Outside the setup phases the original decide runs; the forced path draws one number like the search."""
    calls = []
    real_search = Searcher.search

    def counted(self, *a, **k):
        calls.append(1)
        return real_search(self, *a, **k)

    monkeypatch.setattr(Searcher, "search", counted)
    s = next(st for st in _setup_states(4, 3) if st.phase == PHASE_SETUP_SETTLEMENT)
    legal = E.legal_actions(s)
    bot = make_bot(SEARCH_SPEC)
    r1, r2 = random.Random(9), random.Random(9)
    with openings.using("standin_book"):
        bot.decide(s, legal, r1)
    assert not calls
    bot.decide(s, legal, r2)                                         # the real search
    assert len(calls) == 1 and r1.random() == r2.random()
    # main phase: the installed wrapper hands over to the search
    g = new_game(4, rng=random.Random(3))
    while g.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        g = E.apply(g, E.legal_actions(g)[0], None)
    g = E.apply(g, (A.ROLL, 8), random.Random(0))
    legal = E.legal_actions(g)
    assert len(legal) > 1
    with openings.using("denial"):
        a = bot.decide(g, legal, random.Random(1))
    assert len(calls) == 2
    assert a == bot.decide(g, legal, random.Random(1))


def test_parambot_applies_policy_only_to_the_candidate_seats():
    cand = {"openings.policy": "standin_book"}
    bots = [ParamBot(make_bot(SEARCH_SPEC), cand), ParamBot(HeuristicBot(), {}),
            ParamBot(make_bot(SEARCH_SPEC), cand), ParamBot(HeuristicBot(), {})]
    for b in bots:
        b.reset()
    s = new_game(4, rng=random.Random(12))
    rng = random.Random(12)
    checked = 0
    while s.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        legal = E.legal_actions(s)
        i = s.current
        a = bots[i].decide(s, legal, rng)
        _assert_pristine()                                           # restored after every hook
        if i in (0, 2):
            assert a == openings.choose(s, legal, "standin_book")
        elif s.phase == PHASE_SETUP_SETTLEMENT:
            top = placement.setup_pick(s, i, k=60)                   # default seats keep the current ranking
            assert dict(top)[a[1]] == top[0][1]
        else:
            assert a == (A.SETUP_ROAD, placement.setup_road_pick(s, i, s.setup_last_settlement)[0])
        checked += 1
        s = E.apply(s, a, None)
    assert checked == 16


# ---------------------------------------------------------------------------
# standin_book is a faithful port of the stand-ins' opening book
# ---------------------------------------------------------------------------
def test_standin_book_matches_the_catanatron_stand_ins():
    pytest.importorskip("catanatron")
    from catanatron.models.enums import SETTLEMENT, ActionType
    from catanatron.models.player import Color, RandomPlayer

    from catanbot.bench import catanatron_adapter as AD
    from catanbot.bench import catanatron_players as CP

    compared = roads = 0
    for seed in (3, 11, 29):
        colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
        game = AD.make_game([RandomPlayer(c) for c in colors], seed=seed)
        m = AD.mapping_for(game.state.board.map)
        while game.state.current_prompt.name.startswith("BUILD_INITIAL"):
            st = game.state
            color = st.colors[st.current_player_index]
            playable = AD.playable_actions_of(game)
            cb = AD.to_catanbot_state(game, color, m)
            me = cb.current
            tables = CP.get_map_tables(st.board.map)
            own = CP._production(st, color, tables, None)
            if st.current_prompt.name == "BUILD_INITIAL_SETTLEMENT":
                mine = openings.standin_spot_scores(cb, me)
                for a in playable:
                    ref = CP.opening_spot_score(tables, st.board, a.value, own)
                    assert mine[m.node_to_vertex[a.value]] == pytest.approx(ref, rel=1e-9, abs=1e-12)
                    compared += 1
                ref_choice = CP.choose_initial_settlement(game, color, playable)
                top_v, top_s = openings.standin_pick(cb, me, 1)[0]
                assert top_s == pytest.approx(openings.STANDIN_SCALE * mine[m.node_to_vertex[ref_choice.value]])
                action = ref_choice
            else:
                settlement = st.buildings_by_color[color][SETTLEMENT][-1]
                mine = openings.standin_road_scores(cb, me, m.node_to_vertex[settlement])
                ref_choice = CP.choose_initial_road(game, color, playable)
                e_ref = m.edge_to_id[(ref_choice.value[0], ref_choice.value[1])]
                e_mine, s_mine = openings.standin_road(cb, me, m.node_to_vertex[settlement])
                assert s_mine == pytest.approx(openings.STANDIN_SCALE * mine[e_ref])   # same best score
                assert set(mine) == {m.edge_to_id[(a.value[0], a.value[1])] for a in playable
                                     if a.action_type == ActionType.BUILD_ROAD}
                roads += 1
                action = ref_choice
            game.execute(action)
    assert compared > 300 and roads == 3 * 8
