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


def test_stage_weight_and_slack_bounds():
    from catanbot.politics import MAX_SLACK, stage_weight
    s = played_state()
    early = stage_weight(s)
    s.players[1].cities = [B.HEX_VERTICES[16][3], B.HEX_VERTICES[18][2]]
    s.players[1].settlements = s.players[1].settlements + [B.HEX_VERTICES[0][5]]
    assert stage_weight(s) >= early
    pol = PoliticalState(4)
    pol.capital[0][2] = 1.0
    assert abs(pol.favor_slack(s, 2, 0)) <= MAX_SLACK
    pol.capital[0][2] = -1.0
    assert pol.favor_slack(s, 2, 0) <= 0.0
    # award transfer detection without state_after
    pol2 = PoliticalState(4)
    s.longest_road_owner = 1
    pol2.observe(s, (A.BUILD_ROAD, s.players[0].roads[0]), 0)
    s2 = s.copy()
    s2.longest_road_owner = 0
    pol2.observe(s2, (A.END_TURN,), 0)
    assert pol2.get(0, 1) < pol2.baseline


# ---------------------------------------------------------------------------
# Advice wording, monopolies against hidden hands, robber habits from the opponent model
# ---------------------------------------------------------------------------
def four_settlements():
    from catanbot import placement as P
    s = new_game(4, hexes=B.STANDARD_HEXES)
    occ = {}

    def place(pi, v):
        assert P.is_free_vertex(occ, v), v
        s.players[pi].settlements.append(v)
        occ[v] = pi

    place(0, B.HEX_VERTICES[4][0])
    place(1, B.HEX_VERTICES[2][1])
    place(2, B.HEX_VERTICES[12][4])
    place(3, B.HEX_VERTICES[11][2])
    for pi in range(4):
        s.players[pi].roads.append(B.VERTEX_EDGES[s.players[pi].settlements[0]][0])
    s.phase = PHASE_MAIN
    s.dice = 6
    s.current = 0
    return s


def test_award_line_when_the_challenger_already_holds_a_card():
    s = four_settlements()
    s.largest_army_owner = 2
    s.players[2].played_knights = 3
    s.players[2].cities = [B.HEX_VERTICES[16][3], B.HEX_VERTICES[18][2]]
    s.players[1].played_knights = 3
    s.players[1].dev_known = False
    s.players[1].dev_count = 1
    s.players[3].played_knights = 3
    opts = {o["player"]: o for o in award_threat_opportunities(s, 0)}
    assert opts[1]["needs"] == [0, 0, 0, 0, 0] and sum(opts[3]["needs"]) == 3
    lines = runway_advice(s, 0)
    assert not any("(needs )" in l for l in lines)
    assert any(l.startswith("blue is 1 step") and "already holds a dev card" in l for l in lines)
    assert any(l.startswith("green is 1 step") and "needs 1 sheep, 1 wheat, 1 ore" in l for l in lines)


def test_monopoly_against_hidden_hands_costs_capital():
    from catanbot.counting import expected_opponent_hands
    s = four_settlements()
    for j in (1, 2):
        s.players[j].hand_known = False
        s.players[j].hand_size = 6
        s.players[j].resources = [0] * 5
    s.players[3].resources = [0] * 5
    exp = expected_opponent_hands(s, me=0)
    res = max(range(5), key=lambda r: exp[1][r])
    pol = PoliticalState(4)
    before = [pol.get(0, j) for j in range(4)]
    pol.observe(s, (A.PLAY_MONOPOLY, res), 0)
    assert pol.get(0, 1) < before[1]                       # expected take from the hidden hand
    assert pol.get(0, 3) == before[3]                      # known empty hand: nothing taken
    # with the state after the play the public take is split among the hidden hands
    pol2 = PoliticalState(4)
    after = s.copy()
    after.players[0].resources[res] += 4
    pol2.observe(s, (A.PLAY_MONOPOLY, res), 0, state_after=after)
    assert pol2.get(0, 1) < before[1] and pol2.get(0, 3) == before[3]
    assert any("monopolised" in e for e in pol2.events)


def test_robber_target_weights_take_the_actors_habits_from_the_model():
    from catanbot.opponent_model import OpponentModel
    s = four_settlements()
    for p in s.players:
        p.resources = [1, 1, 1, 1, 1]
    m = OpponentModel(s)
    pol = PoliticalState(4)
    base = pol.robber_target_weights(s, 1)
    assert pol.robber_target_weights(s, 1, model=m) == base   # nothing observed yet
    for _ in range(4):
        m.observe(s, (A.MOVE_ROBBER, 11, 3), 1)               # blue always robs green
    w = pol.robber_target_weights(s, 1, model=m)
    assert w[1] == 0.0 and w[3] > base[3] and w[2] < base[2]
    # the simulated opponent robs its habitual victim
    from catanbot.robber import best_robber_move
    _, victim, _ = best_robber_move(s, 1, target_weights=w)
    assert victim == 3
