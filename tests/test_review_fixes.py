"""Regression tests for behaviours found by the adversarial reviews."""
import random

from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot import placement as P
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.counting import HandBelief
from catanbot.heuristic import HeuristicEvaluator, action_priors
from catanbot.politics import PoliticalState, political_trade_options
from catanbot.search import SearchConfig, Searcher
from catanbot.state import (PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROLL, PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT,
                            TradeOffer, new_game)


def base_state():
    s = new_game(4, hexes=B.STANDARD_HEXES)
    occ = {}

    def place(pi, v, city=False):
        assert P.is_free_vertex(occ, v), v
        (s.players[pi].cities if city else s.players[pi].settlements).append(v)
        occ[v] = pi

    place(0, B.HEX_VERTICES[4][0])
    place(1, B.HEX_VERTICES[2][1])
    place(2, B.HEX_VERTICES[12][4])
    place(3, B.HEX_VERTICES[11][2])
    for pi in range(4):
        v = s.players[pi].settlements[0]
        s.players[pi].roads.append(B.VERTEX_EDGES[v][0])
    s.phase = PHASE_MAIN
    s.dice = 6
    return s


def test_execute_trade_never_feeds_a_player_at_nine_vp():
    s = base_state()
    # player 1 at 9 VP: 4 cities + settlement = 9
    s.players[1].cities = [B.HEX_VERTICES[16][3], B.HEX_VERTICES[18][2], B.HEX_VERTICES[0][5], B.HEX_VERTICES[7][4]]
    assert s.public_vp(1) == 9
    s.players[0].resources = [0, 0, 0, 0, 2]
    s.players[1].resources = [0, 0, 1, 2, 2]
    s.pending_trade = TradeOffer(0, [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], responses={1: True, 2: False, 3: False})
    s.phase = PHASE_TRADE_SELECT
    s.current = 0
    legal = E.legal_actions(s)
    assert (A.EXECUTE_TRADE, 1) in legal and (A.CANCEL_TRADE,) in legal
    priors = dict(zip(legal, action_priors(s, legal, 0)))
    assert priors[(A.CANCEL_TRADE,)] > priors[(A.EXECUTE_TRADE, 1)]
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=3, expand=6))
    res = se.search(s, 0, random.Random(1))
    assert res[0].action == (A.CANCEL_TRADE,)


def test_political_offers_are_engine_legal():
    s = base_state()
    s.longest_road_owner = 1
    s.longest_road_len = 5
    s.players[0].resources = [3, 3, 2, 2, 2]
    for pi in (1, 2, 3):
        p = s.players[pi]
        p.hand_known = False
        p.hand_size = 5
    for r in range(5):
        s.bank[r] = 19 - s.players[0].resources[r]
    for opt in political_trade_options(s, 0, HeuristicEvaluator(), PoliticalState(4)):
        a = opt["action"]
        assert not any(a[1][r] and a[2][r] for r in range(5))
        E.apply(s, a, random.Random(0))  # must not raise


def test_monopoly_belief_conserves_cards():
    s = base_state()
    for pi in (1, 2, 3):
        s.players[pi].hand_known = False
        s.players[pi].hand_size = 4
    s.players[0].resources = [1, 1, 1, 1, 1]
    b = HandBelief(s)
    total_before = sum(b.size)
    b.observe_monopoly(0, B.ORE, taken=3)
    assert sum(b.size) == total_before
    assert b.size[0] == 8
    for i in range(4):
        assert abs(sum(b.expected[i]) - b.size[i]) < 1e-6
        assert all(x >= 0 for x in b.expected[i])
    b2 = HandBelief(s)
    b2.observe_monopoly(0, B.WHEAT)
    assert sum(b2.size) == total_before


def test_pre_roll_knight_outranks_roll_when_recommended():
    s = base_state()
    s.phase = PHASE_ROLL
    s.dice = 0
    s.current = 0
    s.robber = 4  # brick 6 next to player 0's settlement
    s.players[0].dev_cards[B.DEV_KNIGHT] = 1
    legal = E.legal_actions(s)
    knights = [a for a in legal if a[0] == A.PLAY_KNIGHT]
    assert knights and (A.ROLL,) in legal
    priors = dict(zip(legal, action_priors(s, legal, 0)))
    assert max(priors[a] for a in knights) > priors[(A.ROLL,)]
    assert all(priors[a] <= max(priors[k] for k in knights) for a in knights)


def test_responder_search_sees_executed_trade():
    """Accepting must be evaluated on the post-trade hand, not the phase flag."""
    s = base_state()
    s.players[0].resources = [0, 1, 1, 1, 0]      # one wood from a settlement
    s.players[1].resources = [2, 0, 0, 0, 1]
    s.players[0].roads = [e for e in s.players[0].roads]
    s.pending_trade = TradeOffer(1, [1, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    s.pending_trade.get = [0, 0, 0, 1, 0]          # blue gives wood, wants wheat
    s.phase = PHASE_TRADE_RESPONSE
    s.current = 1
    s.trade_responder = 0
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=3, expand=6))
    outs = se._outcomes(s, (A.ACCEPT_TRADE,), 0)
    assert abs(sum(p for p, _ in outs) - 1.0) < 1e-9
    # after acceptance and execution our hand must have changed in at least one outcome
    assert any(st.players[0].resources != s.players[0].resources for _, st in outs)
    outs_r = se._outcomes(s, (A.REJECT_TRADE,), 0)
    assert all(st.players[0].resources == s.players[0].resources for _, st in outs_r)


def test_selfplay_records_roll_states():
    from catanbot.selfplay import make_bot, play_game
    from catanbot.features import FEATURE_NAMES
    bots = [make_bot("heuristic:temp=0.3"), make_bot("heuristic"), make_bot("random:end=0.5")]
    r = play_game(bots, rng=random.Random(4), record=True, max_turns=60, sample_every=1)
    names = list(FEATURE_NAMES)
    roll_cols = [i for i, n in enumerate(names) if "phase_roll" in n]
    assert roll_cols, names[:10]
    assert (r.X[:, roll_cols[0]] > 0.5).sum() > 0
