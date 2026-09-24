import json
import os
import random

from catanbot import actions as A
from catanbot import board as B
from catanbot import placement as P
from catanbot import trading as T
from catanbot.opponent_model import OpponentModel, OpponentProfile, game_stage, trade_stage_factor
from catanbot.state import PHASE_MAIN, TradeOffer, new_game


def mid_game_state(vps=(1, 1, 1, 1)):
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
    s.players[0].resources = [3, 0, 1, 4, 2]
    s.players[1].resources = [1, 2, 1, 0, 0]
    s.players[2].resources = [0, 0, 2, 2, 3]
    s.players[3].resources = [4, 1, 0, 1, 0]
    for r in range(5):
        s.bank[r] -= sum(p.resources[r] for p in s.players)
    return s


def test_game_stage_monotone():
    s = mid_game_state()
    early = game_stage(s)
    s.players[1].cities = [B.HEX_VERTICES[16][3]]
    s.players[1].cities += [B.HEX_VERTICES[18][2]]
    s.players[1].settlements += [B.HEX_VERTICES[0][5]]
    late = game_stage(s)
    assert 0 <= early < late <= 1
    assert trade_stage_factor(s) < 1.0


def test_profile_learns_valuations_and_acceptance():
    prof = OpponentProfile("bob")
    for _ in range(5):
        prof.note_accept([0, 0, 0, 0, 1], [1, 0, 0, 0, 0], True)   # takes ore, gives wood
    assert prof.value[B.ORE] > prof.value[B.WOOD]
    assert prof.acceptance_rate() > 0.6
    for _ in range(8):
        prof.note_accept([0, 1, 0, 0, 0], [0, 0, 0, 1, 0], False)  # refuses wheat for brick
    assert prof.value[B.WHEAT] > prof.value[B.BRICK]
    assert prof.gives_easily(B.WOOD) > prof.gives_easily(B.WHEAT)
    d = prof.to_dict()
    back = OpponentProfile.from_dict(json.loads(json.dumps(d)))
    assert back.value == prof.value and back.acceptance_rate() == prof.acceptance_rate()


def test_observe_actions_and_predict_accept():
    s = mid_game_state()
    m = OpponentModel(s)
    offer = TradeOffer(0, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0])   # p0 gives wood wants brick
    s2 = s.copy()
    s2.pending_trade = offer
    s2.trade_responder = 1
    m.observe(s2, (A.ACCEPT_TRADE,), 1)
    m.observe(s2, (A.ACCEPT_TRADE,), 1)
    m.observe(s, (A.MOVE_ROBBER, 11, 3), 1)
    m.observe(s, (A.BUILD_CITY, 5), 1)
    m.observe(s, (A.END_TURN,), 1)
    prof = m.profile_of(s, 1)
    assert prof.acceptance_rate() > 0.5
    assert prof.builds["city"] > 0
    assert prof.robbed.get("green", 0) > 0
    p_generous = m.predict_accept(s, 1, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], proposer=0)
    fresh = OpponentModel(s)
    p_prior = fresh.predict_accept(s, 1, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], proposer=0)
    assert 0 <= p_prior <= 1 and p_generous > p_prior
    # cannot pay -> 0
    assert m.predict_accept(s, 1, [1, 0, 0, 0, 0], [0, 0, 0, 0, 3], proposer=0) == 0.0


def test_observe_event_parsing():
    s = mid_game_state()
    m = OpponentModel(s)
    assert m.observe_event(s, "blue accepted give ore get wood") is None
    assert m.observe_event(s, "blue rejected give 2 wheat get brick") is None
    assert m.observe_event(s, "orange proposed give sheep get ore") is None
    assert m.observe_event(s, "green robbed red") is None
    assert m.observe_event(s, "blue bank 4 wood for 1 ore") is None
    assert m.observe_event(s, "nobody accepted give ore get wood") is not None
    assert m.profile_of(s, 1).observations >= 3


def test_rank_and_arbitrage(tmp_path):
    s = mid_game_state()
    m = OpponentModel(s)
    # Teach: blue dumps brick for anything; orange pays ore for brick.
    for _ in range(6):
        m.profile_of(s, 1).note_accept([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], True)
        m.profile_of(s, 2).note_accept([0, 1, 0, 0, 0], [0, 0, 0, 0, 1], True)
    offers = T.candidate_offers(s, 0, B.COST_SETTLEMENT, model=m)
    assert offers
    ranked = m.rank_offers(s, 0, offers, B.COST_SETTLEMENT)
    assert ranked[0]["p_accept"] >= ranked[-1]["p_accept"] - 1e-9 or ranked[0]["score"] >= ranked[-1]["score"]
    arbs = m.arbitrage_opportunities(s, 0)
    assert isinstance(arbs, list)
    path = tmp_path / "profiles.json"
    m.save(str(path))
    m2 = OpponentModel.load(str(path))
    assert m2.profile("blue").acceptance_rate() == m.profile("blue").acceptance_rate()
    advice = T.trade_advice(s, 0, model=m)
    assert any("Game stage" in l for l in advice)


def test_should_accept_gets_stricter_late():
    s = mid_game_state()
    offer = TradeOffer(1, [0, 0, 0, 0, 1], [1, 0, 0, 0, 0])   # we (p0) receive ore, pay wood
    early, _ = T.should_accept(s, 0, offer)
    # Make proposer a runaway leader: 3 cities
    s.players[1].cities = [B.HEX_VERTICES[16][3], B.HEX_VERTICES[18][2], B.HEX_VERTICES[0][5]]
    s.players[1].settlements = [B.HEX_VERTICES[2][1], B.HEX_VERTICES[7][4]]
    late, why = T.should_accept(s, 0, offer)
    assert early is True
    assert late is False and ("leader" in why or "late" in why)
