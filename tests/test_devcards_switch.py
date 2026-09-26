"""The ``devcards.buy`` switch (docs/SCRUTINY.md Q22): off removes BUY_DEV from our bot's own choices only.

Default play must stay byte-identical: the pinned digests below were produced by the code before the switch
existed (python3, C++ core built), and ``tests/test_corrections.py``'s pinned default games and searches still hold.
"""
import hashlib
import random

import pytest

from catanbot import actions as A
from catanbot import devcards
from catanbot import engine as E
from catanbot import tuning
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.param_bot import ParamBot
from catanbot.heuristic import HeuristicEvaluator, action_priors
from catanbot.search import SearchConfig, Searcher
from catanbot.selfplay import make_bot, play_game
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, new_game

DEFAULT = tuning.DEFAULT_SEARCH_SPEC
OFF = {"devcards.buy": False}


def game_hash(specs, seed, max_turns, bots=None):
    h = hashlib.sha256()
    res = play_game(bots or [make_bot(sp) for sp in specs], rng=random.Random(seed), seed=seed, max_turns=max_turns,
                    on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    return h.hexdigest()[:16]


def dev_positions(n=6, seeds=(101, 202, 303)):
    """Main-phase states of heuristic self-play in which buying a development card is legal."""
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        s = new_game(4, rng=rng)
        bots = [HeuristicBot(temperature=0.15) for _ in range(4)]
        while s.phase != PHASE_GAME_OVER and s.turn < 150 and len(out) < n * (seeds.index(seed) + 1):
            i = E.acting_player(s)
            legal = E.legal_actions(s)
            if s.phase == PHASE_MAIN and (A.BUY_DEV,) in legal and s.turn >= 12:
                out.append((s.copy(), i))
            s = E.apply(s, bots[i].decide(s, legal, rng), rng)
    return out[:n * len(seeds)]


def buys_per_seat(bots, seed, max_turns):
    n = [0] * len(bots)

    def count(state, action, player):
        if action[0] == A.BUY_DEV:
            n[player] += 1

    play_game(bots, rng=random.Random(seed), seed=seed, max_turns=max_turns, on_action=count)
    return n


# ---------------------------------------------------------------------------
# registry and the filter
# ---------------------------------------------------------------------------
def test_registered_flag_defaults_on():
    t = tuning.find("devcards.buy")
    assert t is tuning.TUNABLES["devcards.buy"] is tuning.find("BUY_ENABLED")
    assert (t.kind, t.default, t.candidates) == ("flag", True, [False])
    assert [str(tg) for tg in t.targets] == ["devcards.BUY_ENABLED"]
    assert not t.requires_search and not t.needs_python_evaluator and t.description
    assert devcards.BUY_ENABLED is True and tuning.verify_registry() == []
    assert t.parse("off") is False and t.parse("on") is True and t.format(False) == "off"
    assert tuning.apply({"devcards.buy": True}) == []          # on installs nothing
    with tuning.overridden(OFF):
        assert devcards.BUY_ENABLED is False
    assert devcards.BUY_ENABLED is True


def test_without_dev_buys_is_the_identity_when_on():
    legal = [(A.BUILD_ROAD, 3), (A.BUY_DEV,), (A.END_TURN,)]
    assert devcards.without_dev_buys(legal) is legal
    with tuning.overridden(OFF):
        assert devcards.without_dev_buys(legal) == [(A.BUILD_ROAD, 3), (A.END_TURN,)]
        only = [(A.BUY_DEV,)]
        assert devcards.without_dev_buys(only) is only        # never leaves a decision without an action


# ---------------------------------------------------------------------------
# our own choices: search and heuristic bot
# ---------------------------------------------------------------------------
def test_search_never_considers_buy_dev_when_off():
    pos = dev_positions()
    assert len(pos) >= 12
    cfg = SearchConfig(depth=1, beam=4, expand=8)
    ranked_default = 0
    for s, me in pos:
        res = Searcher(HeuristicEvaluator(), cfg).search(s, me, random.Random(9))
        ranked_default += any(r.action == (A.BUY_DEV,) for r in res)
        with tuning.overridden(OFF):
            off = Searcher(HeuristicEvaluator(), cfg).search(s, me, random.Random(9))
        assert off and all(r.action[0] != A.BUY_DEV and all(a[0] != A.BUY_DEV for a in r.line) for r in off)
        with tuning.overridden({"devcards.buy": True}):       # spelled out at its default: the same search
            on = Searcher(HeuristicEvaluator(), cfg).search(s, me, random.Random(9))
        assert [(r.action, r.value, r.line) for r in on] == [(r.action, r.value, r.line) for r in res]
    assert ranked_default >= len(pos) // 2        # the default search does weigh the purchase


def test_bots_decide_without_buy_dev_when_off():
    pos = dev_positions()
    heur_default = 0
    for s, me in pos:
        legal = E.legal_actions(s)
        heur_default += HeuristicBot().decide(s, legal, random.Random(1)) == (A.BUY_DEV,)
        for bot in (ParamBot(HeuristicBot(), OFF), ParamBot(HeuristicBot(temperature=5.0, epsilon=1.0), OFF),
                    ParamBot(make_bot(DEFAULT), OFF)):
            for k in range(3):
                assert bot.decide(s, legal, random.Random(k)) != (A.BUY_DEV,)
                assert all(r.action[0] != A.BUY_DEV for r in (getattr(bot, "last_results", None) or []))
    assert heur_default >= 1


def test_opponent_model_is_unchanged():
    """The switch reads nothing the opponents' prediction / simulation uses: action_priors, should_buy_dev."""
    for s, me in dev_positions(n=3):
        legal = E.legal_actions(s)
        for j in range(4):
            base = (action_priors(s, legal, j), devcards.should_buy_dev(s, j))
            with tuning.overridden(OFF):
                assert (action_priors(s, legal, j), devcards.should_buy_dev(s, j)) == base


@pytest.mark.parametrize("kind", ["heuristic", "search"])
def test_candidate_seats_never_buy_in_a_mixed_game(kind):
    total_default = 0
    for seed in (5, 6):
        spec = "heuristic:temp=0.15" if kind == "heuristic" else DEFAULT
        turns = 400 if kind == "heuristic" else 160
        bots = [ParamBot(make_bot(spec), OFF if i % 2 == 0 else {}) for i in range(4)]
        n = buys_per_seat(bots, seed, turns)
        assert n[0] == n[2] == 0, n
        total_default += n[1] + n[3]
    assert total_default > 0          # the default seats in the same games still buy


# ---------------------------------------------------------------------------
# default play: byte-identical
# ---------------------------------------------------------------------------
# sha256[:16] of full games, produced by the code BEFORE the switch existed (python3; the venv agrees)
PINNED_HEURISTIC = {31: "f113440ef5bef9b6", 32: "de14c2670755d210"}   # heuristic:temp=0.15 x4, 400 turns
PINNED_HEURISTIC0 = {33: "f17c7635c349899f"}                          # heuristic:temp=0 x4, 400 turns
PINNED_SEARCH = {41: "0766876d4ee1cdd3"}                              # the default search spec x4, 60 turns


def test_pinned_default_games_are_unchanged():
    for seed, want in PINNED_HEURISTIC.items():
        assert game_hash(["heuristic:temp=0.15"] * 4, seed, 400) == want
        assert game_hash(["heuristic:temp=0.15,tune=devcards.buy:on"] * 4, seed, 400) == want
    for seed, want in PINNED_HEURISTIC0.items():
        assert game_hash(["heuristic:temp=0"] * 4, seed, 400) == want
    for seed, want in PINNED_SEARCH.items():
        assert game_hash([DEFAULT] * 4, seed, 60) == want
        assert game_hash([DEFAULT + ",tune=devcards.buy:on"] * 4, seed, 60) == want
    # the pinned games do contain purchases, so the pin covers the switch's code path
    assert sum(buys_per_seat([make_bot("heuristic:temp=0.15") for _ in range(4)], 31, 400)) > 0


def test_off_changes_the_game():
    assert game_hash(["heuristic:temp=0.15,tune=devcards.buy:off"] * 4, 31, 400) != PINNED_HEURISTIC[31]
