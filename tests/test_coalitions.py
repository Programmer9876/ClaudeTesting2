import random

from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.coalitions import CoalitionDetector, signal
from catanbot.politics import PoliticalState
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, TradeOffer, new_game


def played(seed=5, turns=40):
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    bots = [HeuristicBot() for _ in range(4)]
    while s.phase != PHASE_GAME_OVER and s.turn < turns:
        i = E.acting_player(s)
        s = E.apply_inplace(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
    s.phase = PHASE_MAIN
    s.dice = 6
    return s


def test_signal_is_quadratic_and_capped():
    assert signal(0.0) == 0.0
    assert signal(0.5) == 1.0
    assert abs(signal(0.25) - 0.25) < 1e-9
    assert signal(1.0) == 4.0 and signal(10.0) == 4.0
    # ten subtle deals < one blatant deal
    assert 10 * signal(0.1) < signal(0.5)


def test_blatant_losing_acceptance_creates_a_bloc():
    s = played()
    s.players[1].resources = [0, 0, 0, 0, 3]
    s.players[2].resources = [2, 2, 0, 0, 0]
    det = CoalitionDetector(4)
    # orange (2) accepts giving 2 wood + 2 brick for 1 ore: clearly losing
    offer = TradeOffer(1, [0, 0, 0, 0, 1], [2, 2, 0, 0, 0], responses={2: True, 3: False, 0: False})
    det.observe_trade(s, 1, 2, offer, offer.responses)
    assert det.strength(1, 2) >= 1.0
    assert [1, 2] in det.blocs()
    assert det.against(0) >= 1.0 and det.against(1) == 0.0
    # subtle, fair 1:1 deals do not create a bloc
    det2 = CoalitionDetector(4)
    fair = TradeOffer(1, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], responses={2: True})
    for _ in range(5):
        det2.observe_trade(s, 1, 2, fair, fair.responses)
    assert det2.strength(1, 2) < 1.0


def test_robber_sparing_and_decay():
    s = played(seed=8, turns=60)
    det = CoalitionDetector(4)
    # find the selfish best target for player 0 and a hex that hits nobody relevant (the desert)
    from catanbot.robber import best_robber_move
    h, victim, _ = best_robber_move(s, 0)
    desert = next(i for i, (r, n) in enumerate(s.hexes) if r == B.DESERT)
    if desert == s.robber:
        return
    det.observe_robber(s, 0, desert, -1)
    if victim >= 0:
        assert det.favour[0][victim] > 0
        before = det.favour[0][victim]
        det.decay()
        assert det.favour[0][victim] < before


def test_politics_integration_and_persistence():
    s = played(seed=3, turns=50)
    pol = PoliticalState(4)
    offer = TradeOffer(1, [0, 0, 0, 0, 1], [2, 2, 0, 0, 0], responses={2: True})
    s.players[2].resources = [2, 2, 0, 0, 0]
    s.pending_trade = offer
    pol.observe(s, (A.EXECUTE_TRADE, 2), 1)
    assert pol.coalitions.strength(1, 2) >= 1.0
    w_before = PoliticalState(4).robber_target_weights(s, 1)
    w_after = pol.robber_target_weights(s, 1)
    assert w_after[2] < w_before[2]            # blue spares its ally orange
    assert pol.favor_slack(s, 2, 1) > PoliticalState(4).favor_slack(s, 2, 1)   # ally floats ally
    lines = pol.summary(s, 0)
    assert any("Bloc" in l for l in lines)
    d = pol.to_named_dict(s)
    back = PoliticalState.from_named_dict(d, s)
    assert abs(back.coalitions.strength(1, 2) - pol.coalitions.strength(1, 2)) < 1e-9
    assert pol.observe_event(s, "blue traded orange give 2 ore get 1 wood") is None
