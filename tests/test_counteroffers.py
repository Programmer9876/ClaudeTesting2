"""Counter-offers (Colonist.io rules variant) and out-of-turn trade analysis.

* engine protocol of ``COUNTER_TRADE`` under ``GameState.allow_counters`` (legality, execution, limits);
* flag-off identity (default games byte-identical to the pre-counteroffers commit, new code never entered);
* old-bot compatibility (a counter is an ordinary offer to a bot that does not know counters; ``to_dict``
  unchanged with the flag off) and the C++ guard (``accel.python_only``);
* the bot: a responder counters when accepting is slightly bad and a +1 edit is likely taken, the respond
  lookahead rejects a trade that lets the proposer settle our target spot, the proposer takes a good counter;
* opponent model / card counting updates, the advisor text, specs / tunables / self-play plumbing.
"""
from __future__ import annotations

import hashlib
import json
import random

import pytest

from catanbot import accel, cli, counting, engine as E, tuning
from catanbot import actions as A
from catanbot import board as B
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.random_bot import RandomBot
from catanbot.agents.search_bot import SearchBot
from catanbot.counteroffers import counter_blocked, describe_edit, offer_response_report, rank_counters
from catanbot.heuristic import HeuristicEvaluator
from catanbot.opponent_model import OpponentModel, OpponentProfile
from catanbot.politics import PoliticalState
from catanbot.search import SearchConfig, Searcher
from catanbot.selfplay import make_bot, play_game
from catanbot.state import (GameState, PHASE_MAIN, PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT, TradeOffer,
                            new_game)

ACCEPT, REJECT = (A.ACCEPT_TRADE,), (A.REJECT_TRADE,)
WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)


def vec(**kw):
    out = [0] * 5
    for k, v in kw.items():
        out[B.RESOURCE_INDEX[k]] = v
    return tuple(out)


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------
# Standard board.  Blue (seat 1) is to move with a road reaching vertex 28 (wheat 9 / wood 11 / wood 8, not a
# port): with one brick it builds a settlement there.  Red (seat 0, "us") has a road to vertex 33, next to 28,
# so 28 is one road away for us (placement.road_targets) and a blue settlement on 28 also kills our spot on 33.
SPOT, BLUE_ROAD_MID, BLUE_HOME, RED_ROAD_END, RED_ROAD_MID, RED_HOME = 28, 22, 17, 33, 38, 43
OTHER_SETTLEMENTS = {0: [RED_HOME, 2], 1: [BLUE_HOME, 0], 2: [8, 9], 3: [11, 15]}


def board_state(hands, current=1, allow_counters=False):
    s = new_game(4)
    for i, vs in OTHER_SETTLEMENTS.items():
        s.players[i].settlements = list(vs)
    eb = B.edge_between
    s.players[1].roads = [eb(BLUE_HOME, BLUE_ROAD_MID), eb(BLUE_ROAD_MID, SPOT)]
    s.players[0].roads = [eb(RED_HOME, RED_ROAD_MID), eb(RED_ROAD_MID, RED_ROAD_END)]
    occ = s.occupied_vertices()
    for v in occ:                                         # the distance rule holds
        assert not any(w in occ for w in B.VERTEX_NEIGHBORS[v]), v
    for i, h in hands.items():
        s.players[i].resources = list(h)
        s.players[i].hand_size = sum(h)
    s.bank = [B.BANK_PER_RESOURCE - sum(p.resources[r] for p in s.players) for r in range(5)]
    s.current = current
    s.phase = PHASE_MAIN
    s.dice = 8
    s.turn = 30
    s.allow_counters = allow_counters
    return s


def protocol_state():
    """Blue proposes 1 wood for 1 ore under the counters rule; every other seat can pay."""
    s = board_state({0: [1, 1, 1, 1, 2], 1: [2, 1, 1, 1, 1], 2: [1, 1, 1, 1, 2], 3: [0, 2, 1, 1, 1]},
                    allow_counters=True)
    return E.apply(s, (A.PROPOSE_TRADE, vec(wood=1), vec(ore=1)))


def lookahead_position():
    """Blue offers red 2 ore for 1 brick; nobody else holds brick.  Statically a good deal for red, but the brick
    completes blue's settlement on vertex 28 - red's target spot."""
    s = board_state({1: [1, 0, 1, 1, 3], 0: [1, 2, 0, 2, 1], 2: [0, 0, 2, 1, 0], 3: [1, 0, 1, 0, 0]})
    return s, E.apply(s, (A.PROPOSE_TRADE, vec(ore=2), vec(brick=1)))


class LikelyModel(OpponentModel):
    """An opponent model that is sure the proposer takes counters (85 %)."""

    def predict_counter_accept(self, *args, **kw):
        return 0.85


# ---------------------------------------------------------------------------
# engine protocol
# ---------------------------------------------------------------------------
def test_counter_protocol_round_accept_reject_and_select():
    s = protocol_state()
    assert s.phase == PHASE_TRADE_RESPONSE and s.trade_responder == 2 and s.trades_this_turn == 1
    legal = E.legal_actions(s)
    assert legal[:2] == [ACCEPT, REJECT]
    counters = [a for a in legal if a[0] == A.COUNTER_TRADE]
    assert counters == E.counter_candidates(s) and counters
    # orange (2) asks one more card: gives ore, wants wood + wheat
    ask_more = (A.COUNTER_TRADE, vec(ore=1), vec(wood=1, wheat=1))
    assert ask_more in counters
    swap = (A.COUNTER_TRADE, vec(sheep=1), vec(wood=1))
    assert swap in counters
    s = E.apply(s, ask_more)
    assert s.pending_trade.responses[2] is False and s.pending_trade.counters == {2: ([0, 0, 0, 0, 1], [1, 0, 0, 1, 0])}
    assert s.trade_responder == 3
    s = E.apply(s, ACCEPT)                                  # green accepts the offer as proposed
    s = E.apply(s, swap)                                    # red counters: sheep instead of ore
    # everyone answered: orange's counter is shown to blue as an ordinary offer
    ctr = s.pending_trade
    assert s.phase == PHASE_TRADE_RESPONSE and s.trade_responder == 1 and E.acting_player(s) == 1
    assert (ctr.proposer, ctr.give, ctr.get) == (2, [0, 0, 0, 0, 1], [1, 0, 0, 1, 0])
    assert ctr.responses == {0: False, 3: False} and ctr.is_counter
    assert ctr.origin.give == [1, 0, 0, 0, 0] and ctr.origin.responses == {2: False, 3: True, 0: False}
    assert E.legal_actions(s) == [ACCEPT, REJECT]          # a counter cannot be countered
    with pytest.raises(E.IllegalActionError, match="cannot be countered"):
        E.apply(s, (A.COUNTER_TRADE, vec(ore=1), vec(wood=1)))
    # branch 1: blue takes orange's counter -> trade executes, the round closes
    before = [list(p.resources) for p in s.players]
    done = E.apply(s, ACCEPT)
    assert done.phase == PHASE_MAIN and done.pending_trade is None and done.trade_responder == -1
    assert done.players[1].resources == [before[1][0] - 1, 1, 1, before[1][3] - 1, before[1][4] + 1]
    assert done.players[2].resources == [before[2][0] + 1, 1, 1, before[2][3] + 1, before[2][4] - 1]
    assert done.players[1].hand_size == sum(done.players[1].resources)
    assert done.trades_this_turn == 1                       # counters count toward nothing
    # branch 2: blue rejects -> red's counter; rejects again -> partner selection among plain accepters
    s2 = E.apply(s, REJECT)
    assert s2.pending_trade.proposer == 0 and s2.pending_trade.give == [0, 0, 1, 0, 0] and s2.trade_responder == 1
    s3 = E.apply(s2, REJECT)
    assert s3.phase == PHASE_TRADE_SELECT and s3.pending_trade.proposer == 1 and not s3.pending_trade.counters
    assert E.legal_actions(s3) == [(A.EXECUTE_TRADE, 3), (A.CANCEL_TRADE,)]
    s4 = E.apply(s2, ACCEPT)                                # or blue takes red's swap
    assert s4.players[0].resources[SHEEP] == 0 and s4.players[1].resources[SHEEP] == 2 and s4.phase == PHASE_MAIN


def test_counter_limits_and_validation():
    s = protocol_state()
    snap = json.dumps(s.to_dict(), sort_keys=True)
    bad = [((A.COUNTER_TRADE, vec(ore=1), vec(wood=1)), "change the deal"),       # = accepting
           ((A.COUNTER_TRADE, vec(ore=3), vec(wood=1)), "does not hold"),
           ((A.COUNTER_TRADE, vec(ore=1), vec(ore=1)), "disjoint"),
           ((A.COUNTER_TRADE, (0, 0, 0, 0, 0), vec(wood=1)), "non-empty"),
           ((A.COUNTER_TRADE, (0, 0, 0, -1, 1), vec(wood=1)), "negative"),
           ((A.COUNTER_TRADE, (1, 0), vec(wood=1)), "5 counts")]
    for a, msg in bad:
        with pytest.raises(E.IllegalActionError, match=msg):
            E.apply(s, a)
        assert json.dumps(s.to_dict(), sort_keys=True) == snap        # a refused action changes nothing
    # any well-formed counter the counterer holds is accepted, listed or not (here: not listed)
    s2 = E.apply(s, (A.COUNTER_TRADE, vec(ore=2), vec(wood=1, brick=1)))
    assert s2.pending_trade.counters[2] == ([0, 0, 0, 0, 2], [1, 1, 0, 0, 0])
    # one answer per responder: orange is done, the next answer is green's (asks 3 wood; blue holds 2)
    s3 = E.apply(s2, (A.COUNTER_TRADE, vec(brick=1), vec(wood=3)))
    assert set(s3.pending_trade.counters) == {2, 3} and s3.trade_responder == 0
    s4 = E.apply(s3, REJECT)
    assert s4.pending_trade.proposer == 2 and s4.trade_responder == 1 and s4.trades_this_turn == 1
    # green's counter is dropped when it would be shown: blue cannot pay 3 wood -> nobody accepted -> main
    s5 = E.apply(s4, REJECT)
    assert s5.phase == PHASE_MAIN and s5.pending_trade is None and s5.trade_responder == -1


def test_non_payers_may_counter_under_the_flag():
    # green holds no ore (cannot pay the offer) but holds cards -> still asked under the flag
    hands = {0: [0, 0, 0, 0, 0], 1: [2, 1, 1, 1, 0], 2: [1, 1, 1, 1, 2], 3: [0, 2, 1, 1, 0]}
    s = board_state(hands, allow_counters=True)
    s = E.apply(s, (A.PROPOSE_TRADE, vec(wood=1), vec(ore=1)))
    assert s.pending_trade.responses == {0: False}          # red has an empty hand: auto-rejected
    s = E.apply(s, (A.COUNTER_TRADE, vec(ore=1), vec(wood=1, wheat=1)))   # orange asks one more wheat
    assert s.trade_responder == 3 and E.legal_actions(s)[0] == REJECT      # green cannot accept ...
    assert any(a[0] == A.COUNTER_TRADE for a in E.legal_actions(s))        # ... but may counter
    s = E.apply(s, (A.COUNTER_TRADE, vec(brick=1), vec(wood=1)))
    assert s.pending_trade.proposer == 2 and s.trade_responder == 1
    s = E.apply(s, REJECT)
    assert s.pending_trade.proposer == 3 and s.pending_trade.give == [0, 1, 0, 0, 0]
    s = E.apply(s, ACCEPT)
    assert s.players[1].resources == [1, 2, 1, 1, 0] and s.players[3].resources == [1, 1, 1, 1, 0]
    # the same offer without the flag: green is auto-rejected and nobody may counter
    s0 = board_state(hands)
    s0 = E.apply(s0, (A.PROPOSE_TRADE, vec(wood=1), vec(ore=1)))
    assert s0.pending_trade.responses == {0: False, 3: False}
    assert E.legal_actions(s0) == [ACCEPT, REJECT] and E.counter_candidates(s0) == []
    with pytest.raises(E.IllegalActionError, match="counter-offers are off"):
        E.apply(s0, (A.COUNTER_TRADE, vec(ore=1), vec(wood=1, wheat=1)))


def test_random_games_with_counters_keep_the_invariants():
    rng = random.Random(4)
    s = new_game(4, rng=rng)
    s.allow_counters = True
    s.max_turns = 120
    kinds = {}
    total = None
    n = 0
    while not E.is_terminal(s):
        acts = E.legal_actions(s)
        a = rng.choice(acts)
        kinds[a[0]] = kinds.get(a[0], 0) + 1
        if n % 7 == 0:
            d = s.to_dict()
            assert GameState.from_dict(d).to_dict() == d
        s = E.apply_inplace(s, a, rng)
        n += 1
        cards = [s.bank[r] + sum(p.resources[r] for p in s.players) for r in range(5)]
        assert cards == [B.BANK_PER_RESOURCE] * 5
        assert all(min(p.resources) >= 0 for p in s.players)
        if s.pending_trade is not None and s.pending_trade.origin is not None:
            assert s.phase == PHASE_TRADE_RESPONSE and s.trade_responder == s.current
            assert E.legal_actions(s) in ([ACCEPT, REJECT], [REJECT])
    assert kinds.get(A.COUNTER_TRADE, 0) > 20


# ---------------------------------------------------------------------------
# flag off: identical to before
# ---------------------------------------------------------------------------
# sha256[:16] of the (seat, action) sequence of these games, produced by the pre-counteroffers commit 20818e2
# (catanbot/ extracted read-only from .git objects, same C++ build).  The default bot must play them unchanged.
PINNED = [(["search:depth=1,beam=4,expand=8,evaluator=heuristic,accept_bias=0.3,trade_eps=0.05"] * 2
           + ["search:depth=1,beam=4,expand=8,evaluator=heuristic"] * 2, 9, 40, 443, "d2708a9ec181f883"),
          (["heuristic:temp=0.15"] * 4, 10, 400, 476, "26c38d3c09a61bcc")]


@pytest.mark.parametrize("specs,seed,max_turns,n_actions,digest", PINNED)
def test_default_games_are_byte_identical_to_the_pre_counteroffers_commit(specs, seed, max_turns, n_actions, digest):
    for variant in ("plain", "explicit"):
        sp = specs if variant == "plain" else [s + ",counter=0,resp_la=0" if s.startswith("search") else s
                                                for s in specs]
        log = []
        kw = {"allow_counters": False} if variant == "explicit" else {}
        play_game([make_bot(x) for x in sp], rng=random.Random(seed), seed=seed, max_turns=max_turns,
                  on_action=lambda st, a, p: log.append((p, a)), **kw)
        assert (len(log), hashlib.sha256(repr(log).encode()).hexdigest()[:16]) == (n_actions, digest), variant


def test_flag_off_never_enters_the_counter_code(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("counter code reached in a default game")

    from catanbot import counteroffers, opponent_model, search
    monkeypatch.setattr(E, "counter_candidates", boom)
    monkeypatch.setattr(E, "_present_counter", boom)
    monkeypatch.setattr(E, "_answer_counter", boom)
    monkeypatch.setitem(E._HANDLERS, A.COUNTER_TRADE, boom)
    for name in ("_counter_filter", "_counter_outcomes", "_finish_proposer_turn", "_counter_accept_probability"):
        monkeypatch.setattr(search.Searcher, name, boom)
    monkeypatch.setattr(opponent_model.OpponentModel, "predict_counter_accept", boom)
    monkeypatch.setattr(counteroffers, "rank_counters", boom)
    seen_keys = set()
    bots = [make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic,accept_bias=0.3,trade_eps=0.1"),
            HeuristicBot(temperature=0.3), make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic"),
            HeuristicBot(temperature=0.3, trade_eps=0.2)]

    def check(state, a, p):
        d = state.to_dict()
        seen_keys.update(d)
        assert "allow_counters" not in d
        if d["pending_trade"]:
            assert set(d["pending_trade"]) == {"proposer", "give", "get", "responses"}
        assert a[0] != A.COUNTER_TRADE

    play_game(bots, rng=random.Random(21), seed=21, max_turns=30, on_action=check)
    assert "pending_trade" in seen_keys


def test_to_dict_is_unchanged_with_the_flag_off_and_round_trips_with_it_on():
    old_keys = {"version", "hexes", "robber", "ports", "players", "bank", "dev_deck", "current", "phase", "turn",
                "setup_round", "setup_last_settlement", "dice", "dev_played_this_turn", "free_roads",
                "discard_queue", "pending_trade", "trade_responder", "trades_this_turn", "longest_road_owner",
                "longest_road_len", "largest_army_owner", "winner"}
    s = board_state({1: [1, 0, 1, 1, 3], 0: [1, 2, 0, 2, 1]})
    s1 = E.apply(s, (A.PROPOSE_TRADE, vec(ore=2), vec(brick=1)))
    for st in (s, s1):
        assert set(st.to_dict()) == old_keys
    assert set(s1.to_dict()["pending_trade"]) == {"proposer", "give", "get", "responses"}
    assert GameState.from_dict(s1.to_dict()).allow_counters is False
    # the class-level default: states built without __init__ (copy(), the C++ engine) read False
    bare = GameState.__new__(GameState)
    assert bare.allow_counters is False and TradeOffer.__new__(TradeOffer).origin is None
    # flag on: the key and the counter bookkeeping survive a round trip
    on = protocol_state()
    on = E.apply(on, (A.COUNTER_TRADE, vec(ore=1), vec(wood=1, wheat=1)))
    on = E.apply(on, (A.COUNTER_TRADE, vec(brick=1), vec(wood=1)))
    on = E.apply(on, REJECT)
    d = on.to_dict()
    assert d["allow_counters"] is True and d["pending_trade"]["proposer"] == 2
    assert d["pending_trade"]["origin"]["counters"] == {"3": {"give": [0, 1, 0, 0, 0], "get": [1, 0, 0, 0, 0]}}
    back = GameState.from_dict(json.loads(json.dumps(d)))
    assert back.to_dict() == d and back.pending_trade.origin.responses == on.pending_trade.origin.responses
    assert back.copy().to_dict() == d


# ---------------------------------------------------------------------------
# old bots / C++
# ---------------------------------------------------------------------------
def _counter_shown_to_blue():
    """Blue offered 1 ore for 1 brick; orange countered 'brick for 2 ore'; now blue must answer it."""
    s = board_state({1: [1, 0, 1, 1, 3], 0: [0, 2, 0, 1, 0], 2: [0, 1, 0, 0, 0], 3: [0, 0, 1, 0, 0]},
                    allow_counters=True)
    s = E.apply(s, (A.PROPOSE_TRADE, vec(ore=1), vec(brick=1)))
    assert s.trade_responder == 2
    s = E.apply(s, (A.COUNTER_TRADE, vec(brick=1), vec(ore=2)))
    while s.pending_trade.origin is None:
        s = E.apply(s, REJECT)
    return s


def _old_reader_view(state):
    """What a commit without counters reads from ``state.to_dict()`` (it ignores the unknown keys)."""
    d = json.loads(json.dumps(state.to_dict()))
    d.pop("allow_counters", None)
    if d.get("pending_trade"):
        d["pending_trade"].pop("origin", None)
        d["pending_trade"].pop("counters", None)
    return GameState.from_dict(d)


def test_old_bot_answers_a_counter_as_an_ordinary_offer():
    s = _counter_shown_to_blue()
    old = _old_reader_view(s)
    assert old.phase == PHASE_TRADE_RESPONSE and E.acting_player(old) == 1 == s.current
    assert E.legal_actions(old) == [ACCEPT, REJECT] == E.legal_actions(s)
    rng = random.Random(0)
    for bot in (HeuristicBot(), make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic")):
        bot.reset()
        a = bot.decide(old, E.legal_actions(old), rng)
        assert a in (ACCEPT, REJECT)
        E.apply(s, a)                                        # the old bot's answer is legal in the new engine
    # the old engine's own model of accepting (partner selection by the proposer) moves the same cards
    o1 = E.apply(old, ACCEPT)
    assert o1.phase == PHASE_TRADE_SELECT and (A.EXECUTE_TRADE, 1) in E.legal_actions(o1)
    o2 = E.apply(o1, (A.EXECUTE_TRADE, 1))
    n1 = E.apply(s, ACCEPT)
    assert [p.resources for p in o2.players] == [p.resources for p in n1.players]


def test_cpp_engine_and_native_search_fall_back_to_python_under_the_flag(monkeypatch):
    s = protocol_state()
    assert accel.python_only(s) and not accel.python_only(board_state({1: [1, 0, 0, 0, 0]}))
    assert accel.engine_legal_actions(s) is None
    assert accel.engine_apply(s, REJECT) is None and accel.engine_apply_inplace(s.copy(), REJECT) is None
    assert accel.future_values([s], 0, 1, [], [], None) is None
    with pytest.raises(ValueError, match="counter-offer"):
        accel.random_playout_fast(s)
    if accel.engine_available():
        monkeypatch.setattr(accel, "ENGINE_ACTIVE", True)                # the C++ engine hook switched on ...
        assert any(a[0] == A.COUNTER_TRADE for a in E.legal_actions(s))  # ... still the Python rules
        s2 = E.apply(s, (A.COUNTER_TRADE, vec(ore=1), vec(wood=1, wheat=1)))
        assert s2.pending_trade.counters and s2.allow_counters
    # the searcher's native lookahead refuses flag-on states; the Python lookahead runs instead

    def no_native(*a, **k):
        raise AssertionError("native lookahead called for a counters-rule state")

    monkeypatch.setattr(accel, "future_values", no_native)
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=3000))
    assert se._native_future_values([s], 2, 1, [[6] * 12]) is None
    main = board_state({1: [1, 0, 1, 1, 3], 0: [1, 2, 0, 2, 1]}, allow_counters=True)
    assert se.search(main, 1, random.Random(0))


# ---------------------------------------------------------------------------
# the bot
# ---------------------------------------------------------------------------
def test_responder_counters_when_accept_is_slightly_bad_and_a_plus_one_edit_is_likely():
    s = board_state({1: [1, 0, 2, 1, 1], 0: [0, 1, 1, 1, 2], 2: [0, 0, 0, 0, 0], 3: [0, 0, 0, 0, 0]},
                    allow_counters=True)
    s = E.apply(s, (A.PROPOSE_TRADE, vec(sheep=1), vec(wheat=1)))   # blue: 1 sheep for red's 1 wheat
    assert s.trade_responder == 0
    ev = HeuristicEvaluator()
    model = LikelyModel(s)
    plain = Searcher(ev, SearchConfig(depth=1, beam=4, expand=8), model).search(s, 0, random.Random(1))
    vals = {r.action: r.value for r in plain}
    assert vals[ACCEPT] < vals[REJECT]                                  # accepting is slightly bad
    assert all(r.action[0] != A.COUNTER_TRADE for r in plain)          # counters = 0: never considered
    res = Searcher(ev, SearchConfig(depth=1, beam=4, expand=8, counters=1), model).search(s, 0, random.Random(1))
    top = res[0]
    assert top.action[0] == A.COUNTER_TRADE and top.value > vals[REJECT]
    give, get = top.action[1], top.action[2]
    assert list(give) == s.pending_trade.get and sum(get) == sum(s.pending_trade.give) + 1   # a +1 edit
    assert get[WOOD] == 1 and "asks 1 more wood" in top.explanation and "85%" in top.explanation
    # the margin: a counter is only played when it beats the plain answers by counter_margin
    edge = top.value - vals[REJECT]
    picky = Searcher(ev, SearchConfig(depth=1, beam=4, expand=8, counters=1, counter_margin=edge + 0.004),
                     model).search(s, 0, random.Random(1))
    assert picky[0].action == REJECT and picky[1].action[0] == A.COUNTER_TRADE
    # SearchBot with counter=1 plays it; the default bot never counters
    bot = make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic,counter=1")
    bot.reset()
    bot.model = model
    assert bot.decide(s, E.legal_actions(s), random.Random(0)) == top.action
    plain_bot = make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic")
    plain_bot.reset()
    assert plain_bot.decide(s, E.legal_actions(s), random.Random(0))[0] != A.COUNTER_TRADE


def test_counter_ranking_filters_and_aggressiveness():
    s = board_state({1: [1, 0, 2, 1, 1], 0: [0, 1, 1, 1, 2], 2: [0, 0, 0, 0, 0], 3: [0, 0, 0, 0, 0]},
                    allow_counters=True)
    s = E.apply(s, (A.PROPOSE_TRADE, vec(sheep=1), vec(wheat=1)))
    ev = HeuristicEvaluator()
    rows = rank_counters(s, 0, evaluator=ev, model=OpponentModel(s))
    assert rows and all(r["action"] in E.legal_actions(s) for r in rows)
    assert all(r["score"] == pytest.approx(r["p_accept"] * max(0.0, r["gain"])) for r in rows)
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    greedy = rank_counters(s, 0, evaluator=ev, model=OpponentModel(s), aggr=50.0)
    assert all(r["score"] == pytest.approx(r["p_accept"] ** 0.02 * max(0.0, r["gain"])) for r in greedy)
    # late game against a proposer ahead of us: no counters at all (should_accept's rule)
    late = s.copy()
    late.players[1].settlements = late.players[1].settlements + [5]
    late.players[1].cities = [24, 49, 51]
    late.turn = 200
    assert counter_blocked(late, 0, 1) and rank_counters(late, 0, evaluator=ev) == []
    assert describe_edit(s.pending_trade, (0, 0, 0, 1, 0), (1, 0, 1, 0, 0)) == "asks 1 more wood"
    assert describe_edit(s.pending_trade, (0, 0, 0, 0, 1), (0, 0, 1, 0, 0)) == "gives ore instead of wheat"
    # never feed the leader: blue (4 VP vs our 2) lacks only wood for a settlement; our wood swap is dropped
    lead = board_state({1: [0, 1, 2, 1, 1], 0: [1, 1, 1, 1, 2], 2: [0, 0, 0, 0, 0], 3: [0, 0, 0, 0, 0]},
                       allow_counters=True)
    lead.players[1].cities, lead.players[1].settlements = [BLUE_HOME, 0], []
    lead = E.apply(lead, (A.PROPOSE_TRADE, vec(sheep=1), vec(wheat=1)))
    feeding = (A.COUNTER_TRADE, vec(wood=1), vec(sheep=1))
    assert feeding in E.legal_actions(lead) and not counter_blocked(lead, 0, 1)
    kept = rank_counters(lead, 0, evaluator=ev, model=OpponentModel(lead))
    assert kept and all(r["action"][1][WOOD] == 0 for r in kept)


def test_respond_lookahead_rejects_a_trade_that_lets_the_proposer_take_our_spot():
    base, s = lookahead_position()
    assert s.trade_responder == 0 and s.pending_trade.responses == {2: False, 3: False}
    ev = HeuristicEvaluator()
    static = Searcher(ev, SearchConfig(depth=1, beam=4, expand=8), OpponentModel(s)).search(s, 0, random.Random(1))
    assert static[0].action == ACCEPT                         # at the moment cards change hands: accept
    la = Searcher(ev, SearchConfig(depth=1, beam=4, expand=8, respond_lookahead=1),
                  OpponentModel(s)).search(s, 0, random.Random(1))
    assert la[0].action == REJECT                             # after blue's turn: reject
    v = {r.action: r.value for r in la}
    assert v[ACCEPT] < v[REJECT] - 0.03
    line = next(r.line for r in la if r.action == ACCEPT)
    assert line == [ACCEPT]                                   # an end-of-decision node (nothing searched after it)
    # the lookahead played blue's turn: accepting ends with blue's settlement on the spot
    se = Searcher(ev, SearchConfig(depth=1, respond_lookahead=1), OpponentModel(s))
    se._rng = random.Random(3)
    outs = se._outcomes(s, ACCEPT, 0)
    assert len(outs) == 1 and SPOT in outs[0][1].players[1].settlements and outs[0][1].current == 2
    rej = se._outcomes(s, REJECT, 0)
    assert all(SPOT not in st.players[1].settlements for _, st in rej)
    # a depth-2 search sees it too (the Python lookahead after blue's END_TURN)
    d2 = Searcher(ev, SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=4000,
                                   respond_lookahead=1), OpponentModel(s)).search(s, 0, random.Random(1))
    assert d2[0].action == REJECT


def test_proposer_accepts_a_good_counter_and_rejects_a_bad_one():
    s = _counter_shown_to_blue()                              # orange: "brick for 2 ore" -> blue can settle 28
    bot = make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic")    # a default bot is enough
    bot.reset()
    assert bot.decide(s, E.legal_actions(s), random.Random(0)) == ACCEPT
    assert (A.BUILD_SETTLEMENT, SPOT) in bot.last_results[0].line               # searched our turn after it
    # a counter that does not bring the brick (sheep instead) is refused
    s2 = board_state({1: [1, 0, 1, 1, 3], 0: [0, 2, 0, 1, 0], 2: [0, 1, 1, 0, 0], 3: [0, 0, 1, 0, 0]},
                     allow_counters=True)
    s2 = E.apply(s2, (A.PROPOSE_TRADE, vec(ore=1), vec(brick=1)))
    s2 = E.apply(s2, (A.COUNTER_TRADE, vec(sheep=1), vec(ore=2)))
    while s2.pending_trade.origin is None:
        s2 = E.apply(s2, REJECT)
    bot.reset()
    assert bot.decide(s2, E.legal_actions(s2), random.Random(0)) == REJECT
    # rejecting it continues the original offer: red accepted it -> blue picks its partner itself
    s3 = board_state({1: [1, 0, 1, 1, 3], 0: [0, 2, 0, 1, 0], 2: [0, 1, 1, 0, 0], 3: [0, 0, 1, 0, 0]},
                     allow_counters=True)
    s3 = E.apply(s3, (A.PROPOSE_TRADE, vec(ore=1), vec(brick=1)))
    s3 = E.apply(s3, (A.COUNTER_TRADE, vec(sheep=1), vec(ore=2)))
    s3 = E.apply(s3, REJECT)
    s3 = E.apply(s3, ACCEPT)                                  # red accepts the original
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8))
    res = se.search(s3, 1, random.Random(0))
    rej = next(r for r in res if r.action == REJECT)
    assert rej.line[:2] == [REJECT, (A.EXECUTE_TRADE, 0)]      # our own partner choice, searched
    assert res[0].action == REJECT


# ---------------------------------------------------------------------------
# opponent model / card counting / politics
# ---------------------------------------------------------------------------
def test_opponent_model_learns_from_counters():
    s = protocol_state()
    m = OpponentModel(s)
    prof = m.profile_of(s, 2)
    v0 = list(prof.value)
    acc0 = prof.accept_get[WHEAT].weight
    a = (A.COUNTER_TRADE, vec(ore=1), vec(wood=1, wheat=1))
    m.observe(s, a, 2)
    assert prof.counters.num == 1.0 and prof.accept.weight == 1.0 and prof.accept.num == 0.5
    assert prof.value[WHEAT] > v0[WHEAT] and prof.value[WOOD] > v0[WOOD] and prof.value[ORE] < v0[ORE]
    assert prof.accept_get[WHEAT].weight == acc0 + 1 and prof.accept_give[ORE].mean() > 0.5
    assert m.shortage_hint(s, 2)[WHEAT] == pytest.approx(0.6) and m.shortage_hint(s, 2)[ORE] == 0.0
    # counters are strong evidence: a bigger valuation step than a plain accept of the same deal
    p_acc = OpponentProfile("x")
    p_acc.note_accept([1, 0, 0, 1, 0], [0, 0, 0, 0, 1], True)
    assert prof.value[WHEAT] - prof.value[ORE] > p_acc.value[WHEAT] - p_acc.value[ORE]
    # the proposer answers the counter: counter_accept statistic and a trade partner recorded
    s2 = E.apply(s, a)
    s2 = E.apply(E.apply(s2, REJECT), REJECT)
    assert s2.pending_trade.is_counter
    m.observe(s2, ACCEPT, 1)
    pb = m.profile_of(s2, 1)
    assert pb.counter_accept.weight == 1 and pb.counter_accept.mean() > 0.5
    assert pb.traded_with["orange"] == 1.0 and m.profile_of(s2, 2).traded_with["blue"] == 1.0
    # the shortage hint fades turn by turn and survives serialisation
    back = OpponentProfile.from_dict(json.loads(json.dumps(prof.to_dict())))
    assert back.short == prof.short and back.counters.to_list() == prof.counters.to_list()
    end = board_state({1: [1, 0, 0, 0, 0]})
    m.observe(end, (A.END_TURN,), 1)
    assert m.shortage_hint(s, 2)[WHEAT] == pytest.approx(0.36)
    assert "makes counter-offers" in prof.style_summary()


def test_predict_counter_accept_orders_the_edits():
    s = protocol_state()                        # blue offered 1 wood for 1 ore
    m = OpponentModel(s)
    orig = s.pending_trade

    def p(give, get, **kw):                     # our counter (give, get) as seat 2
        return m.predict_counter_accept(s, 1, orig, give, get, counterer=2, **kw)

    same = p(vec(ore=1), vec(wood=1))
    swap = p(vec(sheep=1), vec(wood=1))         # they asked for ore, get sheep
    more_offered = p(vec(ore=1), vec(wood=2))    # one more of the card they offered
    more_other = p(vec(ore=1), vec(wood=1, wheat=1))
    assert same > swap > more_other and same > more_offered > more_other
    assert 0.1 < swap < 0.5 and more_other < 0.25
    assert p(vec(ore=1), vec(wood=1, wheat=1), alternatives=True) < more_other
    for _ in range(6):
        m.profile_of(s, 1).note_counter_answer(True)
    assert p(vec(ore=1), vec(wood=1, wheat=1)) > more_other
    s.players[1].resources[WHEAT] = 0
    assert p(vec(ore=1), vec(wood=1, wheat=1)) == 0.0            # a known hand that cannot pay


def test_shortage_hint_lowers_predict_accept_for_hidden_hands():
    s = board_state({1: [1, 0, 1, 1, 3], 0: [1, 2, 0, 2, 1], 2: [1, 1, 2, 1, 1]})
    s.players[2].hand_known = False
    m = OpponentModel(s)
    before = m.predict_accept(s, 2, vec(ore=1), vec(wheat=1), proposer=1)
    m.profile_of(s, 2).short[WHEAT] = 0.8
    assert m.predict_accept(s, 2, vec(ore=1), vec(wheat=1), proposer=1) == pytest.approx(before * 0.6)
    s.players[2].hand_known = True                          # known hands ignore the hint
    m2 = OpponentModel(s)
    k = m2.predict_accept(s, 2, vec(ore=1), vec(wheat=1), proposer=1)
    m2.profile_of(s, 2).short[WHEAT] = 0.8
    assert m2.predict_accept(s, 2, vec(ore=1), vec(wheat=1), proposer=1) == k


def test_offers_feed_card_counting_counters_and_plain_proposals():
    # CardCounter: an unseen steal leaves "blue got wood" / "blue got ore"; offering ore drops the first
    cc = counting.CardCounter([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    cc.observe_steal(0, 1)
    assert cc.num_hypotheses == 2
    s = board_state({1: [0, 0, 0, 0, 1], 0: [1, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0], 3: [0, 0, 1, 0, 0]})
    bot = make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic")
    bot.reset()
    bot.belief = cc
    bot.observe(s, (A.PROPOSE_TRADE, vec(ore=1), vec(brick=1)), 1)   # a plain proposal: they hold the ore
    assert cc.num_hypotheses == 1 and cc.probability_has(1, ORE) == 1.0 and cc.probability_has(0, WOOD) == 1.0
    # soft: asking for a resource they already hold makes that hypothesis less likely
    cc2 = counting.CardCounter([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    cc2.observe_steal(0, 1)
    cc2.observe_counter(1, (0, 0, 0, 0, 0), vec(wood=1))       # counters: the same evidence as offers
    assert cc2.probability_has(1, WOOD) == pytest.approx(1.0 / 3.0)
    # a contradiction (offering what no hypothesis holds) leaves the mixture unchanged
    cc2.observe_offer(1, vec(brick=1), vec(wood=1))
    assert cc2.num_hypotheses == 2
    # HandBelief: the offered cards' expectation rises to what was offered, the asked one shrinks
    s.players[1].hand_known = False
    s.players[1].hand_size = 4
    hb = counting.HandBelief(s)
    e0 = list(hb.expected[1])
    hb.observe_offer(1, vec(ore=2), vec(brick=1))
    assert hb.expected[1][ORE] >= 2.0 and hb.expected[1][BRICK] < e0[BRICK]
    assert sum(hb.expected[1]) == pytest.approx(4.0) and hb.size[1] == 4
    bot.belief = hb
    bot.observe(protocol_state(), (A.COUNTER_TRADE, vec(ore=1), vec(wood=1, wheat=1)), 2)
    assert hb.is_exact(2)                                    # an exact hand is left alone


def test_politics_credits_an_accepted_counter_as_a_trade():
    s = _counter_shown_to_blue()
    pol = PoliticalState(4)
    before = pol.get(1, 2), pol.get(2, 1)
    pol.observe(s, ACCEPT, 1)
    assert pol.get(1, 2) > before[0] and pol.get(2, 1) > before[1]      # goodwill both ways, like EXECUTE_TRADE


# ---------------------------------------------------------------------------
# advisor
# ---------------------------------------------------------------------------
def test_offer_response_report_explains_the_proposers_build():
    base, s = lookahead_position()
    rep = offer_response_report(s, 0, HeuristicEvaluator(), model=OpponentModel(s), politics=PoliticalState(4))
    text = "\n".join(rep["lines"])
    assert rep["lines"][0].startswith("Best answer after blue's turn:") and "judged when the cards change hands: accept" \
        in rep["lines"][0]
    assert "accepting lets blue build a settlement on vertex 28 (the 8-9-11 spot you are heading for)" in text
    assert "Reject:" in text and "Counter: you give" in text and "P(blue takes it)" in text
    acc = next(o for o in rep["options"] if o["action"] == ["accept_trade"])
    assert acc["v_trade"] > acc["v_turn"] and ["settlement", SPOT] in acc["builds"]
    assert s.pending_trade.counters is None and not s.allow_counters          # the input state is untouched


def test_cli_offer_prints_the_new_section_and_keeps_the_old_lines(tmp_path, capsys):
    base, _ = lookahead_position()
    path = tmp_path / "state.json"
    path.write_text(json.dumps(base.to_dict()))
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic",
                   "--offer", "blue:give=ore:2;get=brick:1"])
    out = capsys.readouterr().out
    assert rc == 0 and "offer from blue" in out and "Recommended actions" in out and "Trading" in out
    assert "== Offer response" in out
    assert "accepting lets blue build a settlement on vertex 28 (the 8-9-11 spot you are heading for)" in out
    assert "Counter: you give" in out and "P(blue takes it)" in out
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic",
                   "--offer", "blue:give=ore:2;get=brick:1", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["advice"]["offer"] and d["offer_response"]["options"]
    # an unaffordable offer: the section has no accept line (only reject / counters)
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic",
                   "--offer", "blue:give=ore:2;get=brick:9"])
    out = capsys.readouterr().out
    assert "Accept the trade offer" not in out and "Accept: " not in out and "Reject: " in out


# ---------------------------------------------------------------------------
# specs, tunables, self-play
# ---------------------------------------------------------------------------
def test_specs_and_tunables():
    bot = make_bot("search:depth=1,evaluator=heuristic,counter=1,resp_la=1,counter_n=3,counter_aggr=2,"
                   "counter_margin=0.01")
    c = bot.config
    assert (c.counters, c.respond_lookahead, c.counter_candidates, c.counter_aggr, c.counter_margin) == \
        (1, 1, 3, 2.0, 0.01)
    d = make_bot("search:depth=1,evaluator=heuristic").config
    assert (d.counters, d.respond_lookahead) == (0, 0) and SearchConfig().counters == 0
    for name, key in (("search.counters", "counter"), ("search.respond_lookahead", "resp_la"),
                      ("search.counter_aggr", "counter_aggr"), ("search.counter_margin", "counter_margin")):
        t = tuning.TUNABLES[name]
        assert t.kind == "search" and t.spec_key == key and t.default == getattr(d, t.attr)
    assert tuning.verify_registry() == []
    from catanbot.search import reduced_config
    r = reduced_config(c, 2, 1000)
    assert (r.counters, r.respond_lookahead, r.counter_aggr, r.counter_margin) == (1, 1, 2.0, 0.01)


def test_selfplay_with_counters_and_paired_counts():
    made = [0] * 4
    specs = ["search:depth=1,beam=3,expand=6,evaluator=heuristic,counter=1,resp_la=1",
             "search:depth=1,beam=3,expand=6,evaluator=heuristic"] * 2

    def hook(state, a, p):
        if a[0] == A.COUNTER_TRADE:
            made[p] += 1

    r = play_game([make_bot(x) for x in specs], rng=random.Random(2), seed=2, max_turns=24, allow_counters=True,
                  on_action=hook)
    assert r.turns >= 20 and made[0] + made[2] > 0 and made[1] == made[3] == 0
    # random bots counter too (the engine lists the bounded edits) and the paired harness counts them
    res = tuning.play_paired_game("heuristic:temp=0.15", {}, "CCDD", 5, max_turns=20, allow_counters=True)
    assert res["allow_counters"] is True and len(res["seat_counters"]) == 4
    st = tuning.paired_stats([res])
    assert st["allow_counters"] and st["counters_cand"] == 0      # heuristic bots never counter
    g = play_game([RandomBot(), RandomBot(), RandomBot(), RandomBot()], rng=random.Random(1), seed=1, max_turns=30,
                  allow_counters=True)
    assert g.turns > 0


def test_ablate_has_a_counters_rules_switch(capsys):
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("ablate_script_c", os.path.join(root, "scripts", "ablate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_parser().parse_args(["--tunable", "search.counters", "--counters"]).counters is True
    assert mod.main(["--tunable", "search.counters", "--counters", "--plan", "--games", "2"]) == 0
    out = capsys.readouterr().out
    assert "counter-offer rules ON" in out and "search.counters" in out
