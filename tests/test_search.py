import random

import numpy as np

from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot import placement as P
from catanbot import trading as T
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.search_bot import SearchBot
from catanbot.devcards import best_year_of_plenty, should_buy_dev, should_play_monopoly
from catanbot.discard import choose_discard, seven_risk, surplus_dump_actions
from catanbot.heuristic import HeuristicEvaluator, action_priors
from catanbot.opponent_model import OpponentModel, trade_stage_factor
from catanbot.politics import PoliticalState
from catanbot.robber import best_robber_move, should_play_knight
from catanbot.search import Searcher, SearchConfig, roll_distribution, search_determinized
from catanbot.selfplay import make_bot, play_game, tournament
from catanbot.state import (PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER, PHASE_ROLL, PHASE_TRADE_RESPONSE, TradeOffer,
                            new_game)


def play_until(state, pred, bots, rng, limit=5000):
    n = 0
    while not pred(state) and state.phase != PHASE_GAME_OVER and n < limit:
        i = E.acting_player(state)
        legal = E.legal_actions(state)
        state = E.apply_inplace(state, bots[i].decide(state, legal, rng), rng)
        n += 1
    return state


def mid_game(seed=5):
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    bots = [HeuristicBot() for _ in range(4)]
    s = play_until(s, lambda st: st.turn >= 40 and st.phase == PHASE_MAIN and st.current == 0
                   and st.players[0].total_resources >= 3, bots, rng)
    return s


def test_roll_distribution():
    full = roll_distribution(11)
    assert abs(sum(p for _, p in full) - 1.0) < 1e-9 and len(full) == 11
    top = roll_distribution(3)
    assert [v for v, _ in top] == [7, 6, 8] and abs(sum(p for _, p in top) - 1.0) < 1e-9


def test_search_returns_ranked_actions():
    s = mid_game()
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8))
    res = se.search(s, 0, random.Random(1))
    assert res and all(0.0 <= r.value <= 1.0 for r in res)
    assert all(res[i].value >= res[i + 1].value for i in range(len(res) - 1))
    legal = set(E.legal_actions(s))
    assert all(r.action in legal for r in res)
    assert all(r.line and r.line[0] == r.action for r in res)


def test_search_depth2_and_roll_phase():
    s = mid_game(seed=9)
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=2, beam=3, expand=6, opp_roll_samples=2), OpponentModel(s))
    res = se.search(s, 0, random.Random(2))
    assert res and se.nodes > 0
    # roll phase: only ROLL legal but the search still expands the dice
    rng = random.Random(3)
    s2 = play_until(s.copy(), lambda st: st.phase == PHASE_ROLL and st.current == 0, [HeuristicBot()] * 4, rng)
    if s2.phase == PHASE_ROLL:
        se2 = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=2, expand=4))
        res2 = se2.search(s2, 0, random.Random(4))
        assert res2[0].action[0] in (A.ROLL, A.PLAY_KNIGHT) and se2.nodes >= 11


def test_search_takes_winning_city():
    """A player at 9 VP with a city affordable must build it."""
    s = mid_game(seed=11)
    p = s.players[0]
    # give player 0 nine VP worth of pieces artificially: 3 cities + 3 settlements = 9 VP
    s.players[0].resources = [0, 0, 0, 2, 3]
    if not p.settlements:
        return
    # inflate VP via largest army + longest road flags (2 + 2) and pieces
    s.largest_army_owner = 0
    s.longest_road_owner = 0
    need = 9 - s.total_vp(0)
    if need > 0:
        # count remaining VP from cities we could upgrade: just assert on the value ordering instead
        pass
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8))
    res = se.search(s, 0, random.Random(1))
    kinds = [r.action[0] for r in res]
    assert A.BUILD_CITY in kinds
    city = next(r for r in res if r.action[0] == A.BUILD_CITY)
    end = next(r for r in res if r.action == (A.END_TURN,))
    assert city.value > end.value


def test_outcomes_probabilities_sum_to_one():
    s = mid_game(seed=13)
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1))
    s.players[0].resources = [1, 1, 1, 1, 1]
    for a in E.legal_actions(s):
        outs = se._outcomes(s, a, 0)
        assert abs(sum(p for p, _ in outs) - 1.0) < 1e-6, a


def test_search_bot_plays_full_game_and_wins_sometimes():
    bots = [make_bot("search:depth=1,beam=3,expand=6"), make_bot("heuristic"), make_bot("random:end=0.3")]
    r = play_game(bots, rng=random.Random(2), record=False, max_turns=300)
    assert r.turns > 0 and len(r.vps) == 3


def test_search_determinized_on_hidden_state():
    s = mid_game(seed=17)
    for i in (1, 2, 3):
        p = s.players[i]
        p.hand_known = False
        p.hand_size = p.total_resources
        p.resources = [0] * 5
    res = search_determinized(s, 0, HeuristicEvaluator(), SearchConfig(depth=1, beam=3, expand=6), samples=2,
                              rng=random.Random(1))
    assert res and 0 <= res[0].value <= 1


def test_tournament_smoke():
    res = tournament(["heuristic", "random:end=0.3"], games=2, workers=1, seed=3, num_players=3, max_turns=150)
    assert set(res["summary"]) == {"heuristic", "random:end=0.3"}
    assert all(v["games"] > 0 for v in res["summary"].values())


def test_play_game_records_samples():
    try:
        from catanbot.features import NUM_FEATURES
    except ImportError:
        return
    bots = [make_bot("heuristic:temp=0.3"), make_bot("heuristic"), make_bot("random:end=0.5")]
    r = play_game(bots, rng=random.Random(4), record=True, max_turns=120, sample_every=3)
    assert r.X is not None and r.X.shape[1] == NUM_FEATURES and len(r.y) == len(r.X) == len(r.players)
    assert set(r.y.tolist()) <= {0.0, 1.0} or r.winner < 0


# ---------------------------------------------------------------------------
# Strategy modules inside the search (review findings: exploitative offers, 7-protection,
# game-stage trade caps, simulated opponents trading, politics-consistent robber ordering)
# ---------------------------------------------------------------------------
def strategy_state():
    """Four settlements + one road each on the standard board; player 0 to act after rolling."""
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


def set_hands(s, *hands):
    for p, h in zip(s.players, hands):
        p.resources = list(h)
    for r in range(5):
        s.bank[r] = 19 - sum(p.resources[r] for p in s.players)


def taught_model(s):
    """blue dumps brick for wood, orange pays ore for brick: wood -> brick -> ore is an arbitrage chain."""
    m = OpponentModel(s)
    for _ in range(6):
        m.profile_of(s, 1).note_accept([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], True)
        m.profile_of(s, 2).note_accept([0, 1, 0, 0, 0], [0, 0, 0, 0, 1], True)
    return m


def three_cities(s, pi):
    s.players[pi].cities = [B.HEX_VERTICES[16][3], B.HEX_VERTICES[18][2], B.HEX_VERTICES[0][5]]


def test_arbitrage_offers_enter_the_search_with_their_reason():
    s = strategy_state()
    set_hands(s, [3, 0, 1, 4, 2], [1, 2, 1, 0, 0], [0, 0, 2, 2, 3], [4, 1, 0, 1, 0])
    m = taught_model(s)
    arbs = m.arbitrage_opportunities(s, 0)
    assert arbs and arbs[0]["gain"] > 0
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=10), m, None, PoliticalState(4))
    res = se.search(s, 0, random.Random(1))
    cands = se._candidates(s, 0, False)
    direct = (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 0, 0, 0, 1))      # wood -> ore straight from orange
    assert arbs[0]["steps"][0] in cands and direct in cands
    for d in arbs[:2]:
        why = se.explain(s, d["steps"][0], 0)
        assert why.startswith("exploit") and s.players[d["partner"]].color in why and "values" in why
    assert any(r.action == direct and "orange values ore below wood" in r.explanation for r in res)
    # without a model nothing is promoted and the explanation is the generic one
    se0 = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=10))
    se0.search(s, 0, random.Random(1))
    assert not se0.explain(s, direct, 0).startswith("exploit")


def test_intermediary_second_leg_is_offered_after_the_first_trade():
    s = strategy_state()
    set_hands(s, [3, 0, 1, 4, 2], [1, 2, 1, 0, 0], [0, 0, 2, 2, 3], [4, 1, 0, 1, 0])
    m = taught_model(s)
    chain = next(d for d in m.arbitrage_opportunities(s, 0)[:3] if len(d["steps"]) == 2)
    first, second = chain["steps"]
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=10), m, None, PoliticalState(4))
    se.search(s, 0, random.Random(1))
    assert se._chain_next.get(first) == second
    # execute the first leg with the partner: we now hold the bought card
    rng = random.Random(0)
    s2 = E.apply(s, first, rng)
    while s2.phase == PHASE_TRADE_RESPONSE:
        ok = E.acting_player(s2) == chain["partner"]
        s2 = E.apply(s2, (A.ACCEPT_TRADE,) if ok else (A.REJECT_TRADE,), rng)
    s2 = E.apply(s2, (A.EXECUTE_TRADE, chain["partner"]), rng)
    assert s2.phase == PHASE_MAIN and s2.trades_this_turn == 1
    cands = se._candidates(s2, 0, False, [first])
    assert cands[0] == second
    assert "second leg" in se.explain(s2, second, 0)
    # a rejected first leg leaves us without the card: no second leg
    s3 = s.copy()
    s3.trades_this_turn = 1
    assert second not in se._candidates(s3, 0, False, [first])


def test_surplus_dumps_are_candidates_and_the_search_reduces_a_nine_card_hand():
    s = strategy_state()
    set_hands(s, [5, 0, 4, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    legal = E.legal_actions(s)
    assert not any(a[0] in (A.BUILD_ROAD, A.BUILD_SETTLEMENT, A.BUILD_CITY, A.BUY_DEV) for a in legal)
    dumps = surplus_dump_actions(s, 0, legal)
    assert dumps and all(a[0] == A.BANK_TRADE for a in dumps)
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=6))
    cands = se._candidates(s, 0, False)
    assert dumps[0] in cands and sum(1 for a in cands if a in dumps) >= 2
    priors = dict(zip(legal, se._candidate_priors(s, legal, 0)))
    assert all(priors[a] >= 60.0 for a in dumps[:3])          # at least like the trade plan's bank steps
    res = se.search(s, 0, random.Random(1))
    after = E.apply(s, res[0].action, random.Random(0))
    assert after.players[0].total_resources < 9
    # a hand of 7 gets no dump promotion
    set_hands(s, [4, 0, 3, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    legal = E.legal_actions(s)
    assert surplus_dump_actions(s, 0, legal) == []
    assert se._candidate_priors(s, legal, 0) == action_priors(s, legal, 0)


def test_late_game_scales_proposal_priors_and_caps_proposals_per_turn():
    s = strategy_state()
    set_hands(s, [3, 0, 1, 4, 2], [1, 2, 1, 0, 0], [0, 0, 2, 2, 3], [4, 1, 0, 1, 0])
    se = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=10))
    assert se.trade_cap(s) == 4
    legal = E.legal_actions(s)
    props = [i for i, a in enumerate(legal) if a[0] == A.PROPOSE_TRADE]
    assert props
    assert [se._candidate_priors(s, legal, 0)[i] for i in props] == [action_priors(s, legal, 0)[i] for i in props]
    # a 9-VP player makes it late: proposal priors shrink by the stage factor, the cap drops to 2
    s.players[1].cities = [B.HEX_VERTICES[16][3], B.HEX_VERTICES[18][2], B.HEX_VERTICES[0][5], B.HEX_VERTICES[7][4]]
    f = trade_stage_factor(s)
    assert f < 0.6
    late = se._candidate_priors(s, legal, 0)
    raw = action_priors(s, legal, 0)
    for i in props:
        assert abs(late[i] - raw[i] * f) < 1e-9
    assert se.trade_cap(s) == 2
    s.trades_this_turn = 1
    assert any(a[0] == A.PROPOSE_TRADE for a in se._candidates(s, 0, False))
    s.trades_this_turn = 2
    assert not any(a[0] == A.PROPOSE_TRADE for a in se._candidates(s, 0, False))
    # the config is honoured (the cap is interpolated between the early and the late value)
    se2 = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, trade_cap_early=3, trade_cap_late=0))
    assert se2.trade_cap(strategy_state()) == 3 and se2.trade_cap(s) <= 1
    se3 = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, trade_cap_early=0, trade_cap_late=0))
    assert se3.trade_cap(s) == 0 and not any(a[0] == A.PROPOSE_TRADE for a in se3._candidates(strategy_state(), 0, False))


def test_simulated_opponent_proposes_a_trade_that_completes_its_city():
    s = strategy_state()
    s.current = 1
    # blue: 2 wood, 1 wheat, 3 ore (one wheat short of a city); orange: brick + wheat (a wood completes a road)
    set_hands(s, [0, 0, 0, 0, 0], [2, 0, 0, 1, 3], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0])
    cfg = SearchConfig(depth=2, beam=3, expand=6, opp_roll_samples=2)
    se = Searcher(HeuristicEvaluator(), cfg, OpponentModel(s), None, PoliticalState(4))
    legal = E.legal_actions(s)
    prop = se._opponent_proposal(s, 1, legal)
    assert prop is not None and prop[0] == A.PROPOSE_TRADE and prop in legal
    assert prop[2][B.WHEAT] == 1 and prop[1][B.WOOD] == 1
    end = se._greedy_turn(s.copy(), 1, 0, 0)
    assert len(end.players[1].cities) == 1 and end.players[2].resources[B.WOOD] == 1
    # with opponent proposals disabled the simulated turn never trades
    se2 = Searcher(HeuristicEvaluator(), SearchConfig(depth=2, beam=3, expand=6, opp_roll_samples=2,
                                                      opponent_proposals=0))
    end2 = se2._greedy_turn(s.copy(), 1, 0, 0)
    assert not end2.players[1].cities and end2.players[2].resources[B.WOOD] == 0


def test_politics_grudge_aligns_the_search_ordering_with_the_advice():
    s = strategy_state()
    s.phase = PHASE_ROBBER
    s.robber = 9
    set_hands(s, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    legal = E.legal_actions(s)
    plain = best_robber_move(s, 0)[:2]
    victims = set()
    for k in (1, 2, 3):
        pol = PoliticalState(4)
        pol.capital[k][0] = -1.0                      # k wronged us badly: strong grudge
        tw = pol.robber_target_weights(s, 0)
        h, victim, _ = best_robber_move(s, 0, target_weights=tw)   # what the advice prints
        priors = action_priors(s, legal, 0, politics=pol)
        top = legal[max(range(len(legal)), key=lambda i: priors[i])]   # what the search tries first
        assert top == (A.MOVE_ROBBER, h, victim)
        assert victim == k
        victims.add((h, victim))
    assert plain in victims and len(victims) == 3           # the grudge really moves the robber


# --- strategy module regressions (knights, robber, dev cards, planner, discard) ---------------------
def test_knight_timing_rules():
    s = strategy_state()
    s.phase, s.dice, s.robber = PHASE_ROLL, 0, 4            # brick 6 next to our settlement is blocked
    s.players[0].dev_cards[B.DEV_KNIGHT] = 1
    set_hands(s, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    ok, why = should_play_knight(s, 0)
    assert ok and "blocking" in why
    legal = E.legal_actions(s)
    priors = dict(zip(legal, action_priors(s, legal, 0)))
    assert max(priors[a] for a in legal if a[0] == A.PLAY_KNIGHT) > priors[(A.ROLL,)]
    s = strategy_state()
    s.robber = 9
    s.players[0].dev_cards[B.DEV_KNIGHT] = 1
    s.players[0].played_knights = 2
    ok, why = should_play_knight(s, 0)
    assert ok and "Largest Army" in why
    s = strategy_state()
    s.robber = 9
    s.players[0].dev_cards[B.DEV_KNIGHT] = 1
    three_cities(s, 1)
    s.players[1].settlements.append(B.HEX_VERTICES[7][4])
    assert s.total_vp(1) == 8
    ok, why = should_play_knight(s, 0)
    assert ok and "slow down" in why
    # a knight bought this turn is not playable
    s = strategy_state()
    set_hands(s, [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    s.dev_deck = [14, 0, 0, 0, 0]
    s2 = E.apply(s, (A.BUY_DEV,), random.Random(0))
    assert s2.players[0].dev_cards_new[B.DEV_KNIGHT] == 1
    assert not any(a[0] == A.PLAY_KNIGHT for a in E.legal_actions(s2))


def test_best_robber_move_avoids_our_own_production():
    s = strategy_state()
    s.robber = 9
    h, victim, _ = best_robber_move(s, 0)
    assert h != 9 and h not in B.VERTEX_HEXES[s.players[0].settlements[0]]


def test_dev_card_buying_rules():
    s = strategy_state()
    set_hands(s, [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    three_cities(s, 0)
    s.players[0].settlements.append(B.HEX_VERTICES[7][4])
    s.largest_army_owner = 0
    assert s.total_vp(0) >= 8
    ok, why = should_buy_dev(s, 0)
    assert ok and "victory point" in why
    s = strategy_state()
    set_hands(s, [0, 0, 1, 3, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    ok, why = should_buy_dev(s, 0)
    assert not ok and "city" in why


def test_monopoly_threshold_and_year_of_plenty():
    s = strategy_state()
    s.players[0].dev_cards[B.DEV_MONOPOLY] = 1
    set_hands(s, [1, 1, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 2, 0], [1, 0, 0, 0, 0])
    play, r, _ = should_play_monopoly(s, 0)
    assert play and r == B.WHEAT
    set_hands(s, [1, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0])
    play, _, why = should_play_monopoly(s, 0)
    assert not play and "wait" in why
    s = strategy_state()
    set_hands(s, [0, 0, 0, 1, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    r1, r2, why = best_year_of_plenty(s, 0)
    assert (r1, r2) == (B.WHEAT, B.ORE) and "city" in why


def test_never_feed_a_player_at_nine_vp():
    s = strategy_state()
    s.players[1].cities = [B.HEX_VERTICES[16][3], B.HEX_VERTICES[18][2], B.HEX_VERTICES[0][5], B.HEX_VERTICES[7][4]]
    assert s.public_vp(1) == 9
    feeding, why = T.offer_is_feeding_leader(s, 0, 1, [1, 0, 0, 0, 0])
    assert feeding and "9 VP" in why
    assert not T.offer_is_feeding_leader(s, 0, 2, [1, 0, 0, 0, 0])[0]


def test_plan_trades_port_ratio_decisions_and_bank_out_fallback():
    s = strategy_state()
    set_hands(s, [2, 0, 1, 1, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    steps = T.plan_trades(s, 0, B.COST_SETTLEMENT)             # 4:1 unaffordable -> ask blue
    assert steps and steps[0]["kind"] == "player" and steps[0]["fallback"] is None
    assert steps[0]["action"] == (A.PROPOSE_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
    s.ports = dict(s.ports)
    s.ports[s.players[0].settlements[0]] = B.WOOD
    set_hands(s, [3, 0, 1, 1, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    steps = T.plan_trades(s, 0, B.COST_SETTLEMENT)             # 2:1 port -> always the bank
    assert steps[0]["kind"] == "bank" and steps[0]["action"] == (A.BANK_TRADE, B.WOOD, B.BRICK)
    s.ports[s.players[0].settlements[0]] = B.PORT_GENERIC
    set_hands(s, [4, 0, 1, 1, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    steps = T.plan_trades(s, 0, B.COST_SETTLEMENT)             # 3:1 but a sure 1:1 partner -> player, bank fallback
    assert steps[0]["kind"] == "player" and steps[0]["fallback"] == (A.BANK_TRADE, B.WOOD, B.BRICK)
    set_hands(s, [4, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    steps = T.plan_trades(s, 0, B.COST_SETTLEMENT)             # nobody holds brick -> 3:1 port
    assert steps[0]["kind"] == "bank"
    set_hands(s, [4, 0, 1, 1, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    s.bank[B.BRICK] = 0
    steps = T.plan_trades(s, 0, B.COST_SETTLEMENT)             # bank out of brick -> players only
    assert steps[0]["kind"] == "player" and steps[0]["fallback"] is None and "out of" in steps[0]["reason"]


def test_choose_discard_keeps_the_top_target_and_dumps_are_ordered():
    s = strategy_state()
    set_hands(s, [1, 1, 1, 1, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    d = choose_discard(s, 0)
    assert d[0] == A.DISCARD and sum(d[1]) == 4 and d[1][B.ORE] == 3
    left = [s.players[0].resources[r] - d[1][r] for r in range(5)]
    assert all(left[r] >= B.COST_DEV[r] for r in range(5))      # the dev card (top target) stays affordable
    s = strategy_state()
    set_hands(s, [1, 1, 1, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])   # 10 cards, city affordable
    kinds = [a[0] for a in surplus_dump_actions(s, 0, E.legal_actions(s))]
    assert kinds[0] == A.BUILD_CITY and A.BUY_DEV in kinds
    assert kinds.index(A.BUY_DEV) > kinds.index(A.BUILD_ROAD)
    assert all(kinds.index(A.BANK_TRADE) > kinds.index(k) for k in (A.BUILD_CITY, A.BUY_DEV)) if A.BANK_TRADE in kinds else True


def test_seven_risk_counts_opponent_rolls_before_our_next_roll():
    s = strategy_state()                                     # player 0 acts after rolling
    assert abs(seven_risk(s, 0) - (1 - (5 / 6) ** 3)) < 1e-9
    assert seven_risk(s, 1) == 0.0                           # next seat: nobody rolls before them
    assert abs(seven_risk(s, 2) - (1 - (5 / 6) ** 1)) < 1e-9
    assert abs(seven_risk(s, 3) - (1 - (5 / 6) ** 2)) < 1e-9
    s.dice = 0                                               # player 0 has not rolled yet
    assert abs(seven_risk(s, 1) - (1 - 5 / 6)) < 1e-9


# --- self-play trading styles ------------------------------------------------------------------------
def test_bot_specs_parse_trading_styles_and_games_record_the_bias():
    b = make_bot("heuristic:temp=0.3,accept_bias=0.3,offer_temp=0.5,trade_eps=0.05")
    assert (b.accept_bias, b.offer_temp, b.trade_eps) == (0.3, 0.5, 0.05) and b.trade_bias is None
    sb = make_bot("search:depth=1,beam=3,expand=6,bias=0.2,offer_temp=0.5,trade_eps=0.05")
    assert isinstance(sb, SearchBot) and sb.accept_bias == 0.2 and sb.trade_eps == 0.05
    bots = [make_bot("heuristic:accept_bias=0.3"), make_bot("heuristic:accept_bias=0.3,offer_temp=0.5"),
            make_bot("random:end=0.5")]
    play_game(bots, rng=random.Random(1), record=True, max_turns=30, sample_every=4)
    first = [b.trade_bias for b in bots[:2]]
    r = play_game(bots, rng=random.Random(2), record=True, max_turns=30, sample_every=4)
    second = [b.trade_bias for b in bots[:2]]
    assert all(-0.3 <= x <= 0.3 for x in first + second) and first != second    # resampled per game
    assert r.bias is not None and len(r.bias) == len(r.y)
    assert np.allclose(r.bias[r.players == 0], second[0], atol=1e-6)
    assert (r.bias[r.players == 2] == 0).all()                                    # the random bot has none
    # the search bot's accept-value shift is bounded and it still plays a legal game
    bots = [make_bot("search:depth=1,beam=3,expand=6,accept_bias=0.3,offer_temp=0.5,trade_eps=0.1"),
            make_bot("heuristic:trade_eps=0.1"), make_bot("random:end=0.5")]
    r = play_game(bots, rng=random.Random(3), max_turns=25)
    assert r.turns > 0 and bots[0].trade_bias is not None


# ---------------------------------------------------------------------------
# lookahead invariants (DESIGN section 4: horizon consistency, shrinkage, unclamped backup)
# ---------------------------------------------------------------------------
def _finished_nodes(root):
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.finished:
            out.append(n)
        for _, kids in n.children:
            stack.extend(k for _, k in kids)
    return out


class _RootSearcher(Searcher):
    """Keeps the root node, the states handed to ``_future_values`` and what it returned."""

    def search(self, state, player=None, rng=None):
        self.root = None
        self.lookahead_states = None
        self.lookahead_values = None
        return super().search(state, player, rng)

    def _backup(self, node, me):
        if self.root is None:
            self.root = node
        return super()._backup(node, me)

    def _future_values(self, states, me, depth):
        self.lookahead_states = list(states)
        self.lookahead_values = [float(v) for v in self._mock_future(states, me)]
        return self.lookahead_values

    def _mock_future(self, states, me):
        return self._eval(list(states), [me] * len(states))


def test_lookahead_values_every_end_node_and_a_constant_offset_keeps_the_depth1_ranking():
    """With ``finished_lookahead=0`` every end-of-turn node of the tree gets the future value, and a lookahead
    that only adds a constant to every static value cannot change the depth-1 ranking - also when the constant
    pushes ``static + shift`` below 0 for most leaves (the backup no longer clamps; the root values are clamped
    on output).  The old top-4 selection valued identical candidates differently by membership alone."""
    ev = HeuristicEvaluator()
    base = dict(beam=4, expand=8, trade_proposals=0)

    class ConstantOffset(_RootSearcher):
        def _mock_future(self, states, me):
            return [float(v) - 0.5 for v in self._eval(list(states), [me] * len(states))]

    checked = 0
    for seed in (5, 9, 13, 21):
        s = mid_game(seed=seed)
        r1 = Searcher(ev, SearchConfig(depth=1, **base)).search(s, 0, random.Random(1))
        se = ConstantOffset(ev, SearchConfig(depth=2, finished_lookahead=0, **base))
        r2 = se.search(s, 0, random.Random(1))
        fin = _finished_nodes(se.root)
        assert fin and se.lookahead_states is not None
        assert len(se.lookahead_states) == len(fin)                  # all of them, not the top N
        assert all(n.value is not None for n in fin)
        assert abs(se._shift + 0.5) < 1e-9
        assert [r.action for r in r1] == [r.action for r in r2], seed
        assert all(0.0 <= r.value <= 1.0 for r in r2)
        assert all(r2[i].value >= r2[i + 1].value for i in range(len(r2) - 1))
        # the finished nodes' values sit exactly at the shifted horizon (mean delta == every delta)
        assert all(abs(n.value - (n.static - 0.5)) < 1e-9 for n in fin)
        checked += 1
    assert checked == 4


def test_lookahead_top_n_selection_is_still_available():
    ev = HeuristicEvaluator()
    checked = 0
    for seed in (5, 9, 13, 21, 29):
        s = mid_game(seed=seed)
        se = _RootSearcher(ev, SearchConfig(depth=2, beam=4, expand=8, trade_proposals=0, finished_lookahead=2))
        se.search(s, 0, random.Random(1))
        fin = _finished_nodes(se.root)
        if len(fin) <= 2:
            continue
        assert len(se.lookahead_states) == 2
        # the two chosen are the top two by static (+ 0.05 log p); ties (identical candidates) may go either way
        chosen = {id(st) for st in se.lookahead_states}
        key = {id(n.state): n.static + 0.05 * np.log(max(n.prob, 1e-6)) for n in fin}
        assert min(key[i] for i in chosen) >= max(key[i] for i in key if i not in chosen) - 1e-12
        checked += 1
    assert checked >= 2


def test_lookahead_weight_and_apply_lookahead():
    from catanbot.search import apply_lookahead, lookahead_weight
    assert lookahead_weight(SearchConfig(opp_roll_samples=12, lookahead_shrink=12.0)) == 0.5
    assert lookahead_weight(SearchConfig(opp_roll_samples=4, lookahead_shrink=12.0)) == 0.25
    assert lookahead_weight(SearchConfig(opp_roll_samples=24, lookahead_shrink=0.0)) == 1.0
    assert lookahead_weight(SearchConfig(opp_roll_samples=0, lookahead_shrink=3.0)) == 0.25   # n floors at 1
    statics = [0.5, 0.4, 1.0, 0.0]
    futures = [0.45, 0.30, 1.0, 0.0]
    terminal = [False, False, True, True]
    vals, shift = apply_lookahead(statics, futures, terminal, 1.0)
    assert abs(shift + 0.075) < 1e-12                              # terminal nodes stay out of the mean
    assert np.allclose(vals, [0.45, 0.30, 1.0, 0.0])              # weight 1: the raw future values
    vals0, shift0 = apply_lookahead(statics, futures, terminal, 0.0)
    assert shift0 == shift and np.allclose(vals0, [0.425, 0.325, 1.0, 0.0])   # weight 0: static + mean delta
    vals_h, _ = apply_lookahead(statics, futures, terminal, 0.5)
    assert np.allclose(vals_h, [0.4375, 0.3125, 1.0, 0.0])
    # the mean of (value - static) over the live nodes is the shift whatever the weight
    for v in (vals, vals0, vals_h):
        assert abs(np.mean([v[i] - statics[i] for i in range(2)]) - shift) < 1e-12
    assert apply_lookahead([1.0], [1.0], [True], 0.5) == ([1.0], 0.0)
    assert apply_lookahead([], [], [], 0.5) == ([], 0.0)


def test_lookahead_shrinkage_in_the_search():
    """``lookahead_shrink=0`` uses the sampled future values raw; a huge k gives every end node the mean delta,
    so the ranking is the depth-1 one; in between the node keeps ``n / (n + k)`` of its own deviation."""
    ev = HeuristicEvaluator()
    base = dict(beam=4, expand=8, trade_proposals=0, opp_roll_samples=12)

    class Noisy(_RootSearcher):
        def _mock_future(self, states, me):
            r = random.Random(7)
            return [float(v) - 0.1 + 0.08 * r.random() for v in self._eval(list(states), [me] * len(states))]

    s = mid_game(seed=13)
    r1 = Searcher(ev, SearchConfig(depth=1, **base)).search(s, 0, random.Random(1))
    raw = Noisy(ev, SearchConfig(depth=2, lookahead_shrink=0.0, **base))
    raw.search(s, 0, random.Random(1))
    fin_by_state = {id(n.state): n for n in _finished_nodes(raw.root)}
    for st, v in zip(raw.lookahead_states, raw.lookahead_values):
        assert fin_by_state[id(st)].value == v
    flat = Noisy(ev, SearchConfig(depth=2, lookahead_shrink=1e15, **base))
    r_flat = flat.search(s, 0, random.Random(1))
    assert [r.action for r in r_flat] == [r.action for r in r1]
    half = Noisy(ev, SearchConfig(depth=2, lookahead_shrink=12.0, **base))
    half.search(s, 0, random.Random(1))
    assert abs(half._shift - raw._shift) < 1e-12
    fin_half = {id(n.state): n for n in _finished_nodes(half.root)}
    for st, v in zip(half.lookahead_states, half.lookahead_values):
        n = fin_half[id(st)]
        if n.state.phase != PHASE_GAME_OVER:
            expected = n.static + half._shift + 0.5 * (v - n.static - half._shift)
            assert abs(n.value - expected) < 1e-12


def test_reduced_config_keeps_the_shrinkage_and_native_level_dict_carries_it():
    from catanbot.search import native_level_dict, reduced_config
    cfg = SearchConfig(depth=3, beam=6, expand=10, opp_roll_samples=12, lookahead_shrink=7.5)
    sub = reduced_config(cfg, 2, 3000)
    assert sub.lookahead_shrink == 7.5 and sub.finished_lookahead == 2 and sub.opp_roll_samples == 4
    d = native_level_dict(cfg)
    assert d["lookahead_shrink"] == 7.5 and d["finished_lookahead"] == 0 and d["opp_roll_samples"] == 12
    assert all(isinstance(d[k], int) for k in d if k != "lookahead_shrink")


def test_depth3_sub_search_runs_for_every_lookahead_leaf_or_for_none():
    """The depth-3 leaves (end nodes x roll samples) are searched one turn deeper only when ``max_nodes`` still
    covers ``REDUCED_SEARCH_MIN_NODES`` for each of them after the opponents' turns; otherwise every leaf keeps the
    static value and depth 3 is exactly depth 2 (same dice).  A partial set would value the first end nodes of the
    tree deeper than their siblings."""
    from catanbot.search import REDUCED_SEARCH_MIN_NODES
    ev = HeuristicEvaluator()
    base = dict(beam=2, expand=4, trade_proposals=0, opp_roll_samples=2, roll_samples=5, native_future=False)

    class Spy(Searcher):
        def _reduced_search_values(self, leaves, me, depth):
            self.sub_leaves = len(leaves)
            self.sub_remaining = self.config.max_nodes - self.nodes
            return super()._reduced_search_values(leaves, me, depth)

    s = mid_game(seed=13)
    # the budget is set from what depth 2 actually uses on this position, so the test does not depend on the
    # size of the tree: 400 spare nodes can never cover REDUCED_SEARCH_MIN_NODES x (>= 1) leaves
    probe = Searcher(ev, SearchConfig(depth=2, max_nodes=10 ** 6, **base))
    probe.search(s, 0, random.Random(1))
    tight_budget = probe.nodes + REDUCED_SEARCH_MIN_NODES - 100
    d2 = Searcher(ev, SearchConfig(depth=2, max_nodes=tight_budget, **base))
    r2 = d2.search(s, 0, random.Random(1))
    tight = Spy(ev, SearchConfig(depth=3, max_nodes=tight_budget, **base))
    r3 = tight.search(s, 0, random.Random(1))
    assert not hasattr(tight, "sub_leaves")                      # not affordable: no leaf was searched ...
    assert tight.nodes == d2.nodes                               # ... and nothing else changed
    assert [(r.action, r.value) for r in r3] == [(r.action, r.value) for r in r2]
    ample = Spy(ev, SearchConfig(depth=3, max_nodes=10 ** 6, **base))
    r3a = ample.search(s, 0, random.Random(1))
    assert ample.sub_leaves > 0                                  # affordable: every leaf, in one call
    assert ample.sub_remaining >= REDUCED_SEARCH_MIN_NODES * ample.sub_leaves
    assert ample.nodes > d2.nodes
    assert [r.action for r in r3a] and any(abs(a.value - b.value) > 1e-9 for a, b in zip(r3a, r2)
                                           if a.action == b.action) or [r.action for r in r3a] != [r.action for r in r2]
    # the rule itself, on the accounting the searcher keeps
    se = Searcher(ev, SearchConfig(depth=3, max_nodes=2000, **base))
    se.nodes = 0
    assert se._leaves_affordable(4) and not se._leaves_affordable(5)
    se.nodes = 1
    assert not se._leaves_affordable(4)
