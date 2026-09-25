import random

from catanbot import board as B
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.discard import explain_seven_risk
from catanbot.heuristic import static_value
from catanbot.robber import steal_exposure
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, new_game


def played(seed=5, turns=50):
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    bots = [HeuristicBot() for _ in range(4)]
    while s.phase != PHASE_GAME_OVER and s.turn < turns:
        i = E.acting_player(s)
        s = E.apply_inplace(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
    s.phase = PHASE_MAIN
    s.current = 0
    s.dice = 6
    return s


def test_exposure_grows_with_hand_and_lead():
    s = played()
    s.players[0].resources = [0, 0, 0, 0, 0]
    p0, l0, _ = steal_exposure(s, 0)
    assert p0 == 0.0 and l0 == 0.0
    s.players[0].resources = [2, 2, 1, 1, 1]
    p1, l1, detail = steal_exposure(s, 0)
    assert 0.0 < p1 < 1.0 and l1 > 0 and "robbed" in detail
    # make player 0 the runaway leader: opponents should target them more
    s.players[0].cities = list(s.players[0].settlements[:2])
    s.players[0].settlements = s.players[0].settlements[2:]
    s.largest_army_owner = 0
    s.longest_road_owner = 0
    p2, l2, _ = steal_exposure(s, 0)
    assert p2 >= p1 - 1e-9
    # the exposure lowers the static value (compare against the same state with the penalty switched off)
    v_with = static_value(s, 0)
    import catanbot.heuristic as H
    orig = H.steal_exposure_fast
    H.steal_exposure_fast = lambda st, pl: 0.0
    try:
        v_without = static_value(s, 0)
    finally:
        H.steal_exposure_fast = orig
    assert v_with < v_without
    assert v_without - v_with < 2.0  # bounded penalty: a 7-card hand is still worth holding
    txt = explain_seven_risk(s, 0)
    assert isinstance(txt, str) and len(txt) > 10
