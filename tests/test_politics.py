import random

from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.heuristic import HeuristicEvaluator
from catanbot.politics import (PoliticalState, award_threat_opportunities, political_trade_options,
                               relative_position, runway_advice)
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, new_game


def played_state(seed=5, turns=50):
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    bots = [HeuristicBot() for _ in range(4)]
    n = 0
    while s.phase != PHASE_GAME_OVER and s.turn < turns:
        i = E.acting_player(s)
        legal = E.legal_actions(s)
        s = E.apply_inplace(s, bots[i].decide(s, legal, rng), rng)
        n += 1
    return s


def test_relative_position_and_pressure():
    s = played_state()
    pol = PoliticalState(4)
    pos = [relative_position(s, i) for i in range(4)]
    assert all(0 <= x <= 1 for x in pos)
    s.largest_army_owner = 0
    s.longest_road_owner = 0
    assert relative_position(s, 0) > pos[0]
    assert 0 <= pol.target_pressure(s, 0) <= 1


def test_robbing_costs_capital_and_decays():
    s = played_state()
    pol = PoliticalState(4)
    # find a hex where player 1 produces
    h = next(h for h in range(B.NUM_HEXES)
             if any(v in s.players[1].settlements or v in s.players[1].cities for v in B.HEX_VERTICES[h]) and h != s.robber)
    before = pol.get(0, 1)
    pol.observe(s, (A.MOVE_ROBBER, h, 1), 0)
    after = pol.get(0, 1)
    assert after < before
    for _ in range(50):
        pol.decay()
    assert abs(pol.get(0, 1) - pol.baseline) < abs(after - pol.baseline)
    w = pol.robber_target_weights(s, 1)
    assert w[1] == 0.0 and w[0] > 0


def test_trades_build_capital_and_willingness():
    s = played_state()
    pol = PoliticalState(4)
    from catanbot.state import TradeOffer
    s.pending_trade = TradeOffer(0, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0])
    pol.observe(s, (A.EXECUTE_TRADE, 2), 0)
    assert pol.get(0, 2) > pol.baseline and pol.get(2, 0) > pol.baseline
    assert 0.4 <= pol.trade_willingness(s, 2, 0) <= 1.4
    d = pol.to_dict()
    back = PoliticalState.from_dict(d)
    assert back.get(0, 2) == pol.get(0, 2)


def test_award_opportunities_and_political_trades():
    s = played_state(seed=8, turns=60)
    s.phase = PHASE_MAIN
    s.current = 0
    s.dice = 6
    # leader 1 holds longest road with length 5; player 2 has 4-length road potential
    s.longest_road_owner = 1
    s.longest_road_len = 5
    s.players[1].cities = s.players[1].cities or []
    s.players[0].resources = [3, 3, 1, 1, 1]
    for r in range(5):
        s.bank[r] = max(0, 19 - sum(p.resources[r] for p in s.players))
    opts = award_threat_opportunities(s, 0)
    assert isinstance(opts, list)
    trades = political_trade_options(s, 0, HeuristicEvaluator(), PoliticalState(4))
    for t in trades:
        assert t["action"][0] == A.PROPOSE_TRADE and t["delta_me"] >= -0.01 and t["delta_leader"] < 0
    lines = runway_advice(s, 0, PoliticalState(4))
    assert any("position" in l for l in lines)
