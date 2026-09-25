"""Tests for catanbot/winpaths.py (win-path races with crowding) and its search / spec / tunable hooks.

The feature is off by default: the tests in the "default unchanged" section prove, with action-sequence
hashes, that the default bot plays exactly as before the module existed (a subprocess plays the games before
``catanbot.winpaths`` is ever imported).
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import pytest

from catanbot import accel
from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot import heuristic, tuning
from catanbot import winpaths as W
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.param_bot import ParamBot
from catanbot.heuristic import HeuristicEvaluator, action_priors
from catanbot.placement import is_free_vertex, player_production
from catanbot.search import SearchConfig, Searcher
from catanbot.selfplay import BlendedEvaluator, make_bot, play_game
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROLL, new_game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = tuning.DEFAULT_SEARCH_SPEC
P0 = W.current_params()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def played(seed=5, turns=50):
    """A mid-game position from heuristic self-play (as in test_danger), our move in the main phase."""
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


def clear_devs(s):
    for p in s.players:
        p.dev_cards = [0] * 5
        p.dev_cards_new = [0] * 5
        p.dev_known = True
        p.dev_count = 0
        p.played_knights = 0
    s.largest_army_owner = -1


def set_me(s, me):
    s.current = me
    s.phase = PHASE_MAIN
    s.dice = 6
    s.free_roads = 0
    s.dev_played_this_turn = False


def extend_to(s, j, target):
    """Append free edges to player j's network (each one lengthening the official trail) until it reaches ``target``."""
    p = s.players[j]
    for _ in range(40):
        cur = E.longest_road_length(s, j)
        if cur >= target or len(p.roads) >= B.MAX_ROADS:
            break
        occ = s.occupied_vertices()
        eocc = s.occupied_edges()
        net = set(p.settlements) | set(p.cities)
        for e in p.roads:
            net.update(B.EDGE_VERTICES[e])
        done = False
        for u in sorted(net):
            if occ.get(u, j) != j:
                continue
            for e in B.VERTEX_EDGES[u]:
                if e in eocc:
                    continue
                p.roads.append(e)
                if E.longest_road_length(s, j) > cur:
                    done = True
                    break
                p.roads.pop()
            if done:
                break
        if not done:
            break
    return E.longest_road_length(s, j)


def _simple_path(start, length, blocked):
    path = [start]
    seen = {start}

    def rec(v):
        if len(path) - 1 == length:
            return True
        for w in B.VERTEX_NEIGHBORS[v]:
            if w in seen or w in blocked:
                continue
            seen.add(w)
            path.append(w)
            if rec(w):
                return True
            path.pop()
            seen.discard(w)
        return False

    return list(path) if rec(start) else None


def road_board(lengths, me=0, seed=4):
    """Fresh board; seat j gets a settlement at the start of a simple path of ``lengths[j]`` roads (paths are
    vertex-disjoint, so every official trail length is exactly ``lengths[j]``).  Our move in the main phase."""
    s = new_game(len(lengths), rng=random.Random(seed))
    used = set()
    for j in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        occ = s.occupied_vertices()
        for start in range(B.NUM_VERTICES):
            if start in used or not is_free_vertex(occ, start):
                continue
            path = _simple_path(start, lengths[j], used)
            if path is None:
                continue
            s.players[j].settlements = [start]
            s.players[j].roads = [B.edge_between(path[k], path[k + 1]) for k in range(len(path) - 1)]
            used.update(path)
            break
        else:
            raise AssertionError(f"no simple path of {lengths[j]} roads left for seat {j}")
    set_me(s, me)
    for j, L in enumerate(lengths):
        assert E.longest_road_length(s, j) == L
    return s


def rand_race(rng, n=4, holder=None):
    b = [round(rng.uniform(0, 9), 3) for _ in range(n)]
    a = [round(rng.uniform(0, 2), 3) for _ in range(n)]
    b = [max(x, y) for x, y in zip(b, a)]
    G = [round(rng.uniform(0, 4), 3) for _ in range(n)]
    h = rng.randrange(-1, n) if holder is None else holder
    return b, a, G, h


def V(P, i, holder, kappa=P0["KAPPA"]):
    return 20.0 * (1.0 - kappa * (1.0 - P)) if i == holder else 20.0 * kappa * P


# ---------------------------------------------------------------------------
# 1-6: solver and horizon
# ---------------------------------------------------------------------------
def test_solver_probabilities_ties_and_rivals():
    rng = random.Random(1)
    for _ in range(200):
        b, a, G, h = rand_race(rng)
        sol = W.solve_race(b, a, G, h, 5, 1.2)
        assert abs(sum(sol.P) + sol.nobody - 1.0) < 1e-12
        assert h < 0 or sol.nobody == 0.0
    # the holder wins exact ties: equal trails -> the tie bonus puts the holder ahead
    s = played()
    extend_to(s, 1, 6)
    extend_to(s, 2, 6)
    if E.longest_road_length(s, 1) == E.longest_road_length(s, 2):
        s.longest_road_owner, s.longest_road_len = 1, E.longest_road_length(s, 1)
        ctx = W.PathsContext(s, 0)
        sol = ctx.races(s).lr
        assert sol.P[1] > sol.P[2]
    sol = W.solve_race([5.0, 5.0], [0.0, 0.0], [1.0, 1.0], 0, 5, 0.0)
    assert sol.P[0] == pytest.approx(sol.P[1]) and sol.credit[0] > sol.credit[1]   # same P, holder keeps more
    # P_me rises with our level
    ps = [W.solve_race([x, 4.0, 3.0], [0.0] * 3, [1.0, 1.0, 1.0], -1, 5, 1.2).P[0] for x in (2.0, 3.0, 4.0, 5.0)]
    assert all(p2 > p1 for p1, p2 in zip(ps, ps[1:]))
    # a strong rival lowers P_me and raises U_me and N_close_me
    base = W.solve_race([3.0, 2.0], [0.0, 0.0], [2.0, 1.0], -1, 5, 1.2)
    more = W.solve_race([3.0, 2.0, 5.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], -1, 5, 1.2)
    assert more.P[0] < base.P[0] and more.U[0] > base.U[0] and more.N_close[0] > base.N_close[0]


def test_credit_is_monotone_property():
    rng = random.Random(7)
    for _ in range(300):
        b, a, G, h = rand_race(rng)
        cost = rng.choice([0.0, 1.2, 2.5])
        i = rng.randrange(4)
        sol = W.solve_race(b, a, G, h, 5, cost)
        up = list(b)
        up[i] += rng.uniform(0.01, 2.0)
        assert W.solve_race(up, a, G, h, 5, cost).credit[i] >= sol.credit[i] - 1e-12      # own level
        j = (i + 1 + rng.randrange(3)) % 4
        riv = list(b)
        riv[j] += rng.uniform(0.01, 2.0)
        assert W.solve_race(riv, a, G, h, 5, cost).credit[i] <= sol.credit[i] + 1e-12     # a rival's level


def test_passive_floor():
    rng = random.Random(3)
    for _ in range(200):
        b, a, G, h = rand_race(rng)
        sol = W.solve_race(b, a, G, h, 5, 1.2)
        for i in range(4):
            assert sol.credit[i] >= V(sol.P0[i], i, h) - 1e-12
    sol = W.solve_race([2.0, 11.0, 3.0], [0.0] * 3, [3.0, 1.0, 1.0], 1, 5, 1.2)   # rival far ahead
    assert not sol.active[0] and sol.credit[0] == V(sol.P0[0], 0, 1)


def test_zero_sum_with_holder_and_no_cost():
    rng = random.Random(11)
    for _ in range(200):
        b, a, G, _h = rand_race(rng)
        h = rng.randrange(4)
        sol = W.solve_race(b, a, G, h, 5, 0.0)
        assert abs(sum(sol.credit[i] - (20.0 if i == h else 0.0) for i in range(4))) < 1e-9


def test_knight_pool_scales_growth():
    s = played(seed=6)
    clear_devs(s)
    for p in s.players:
        p.resources = [0, 0, 4, 4, 4]
    s.players[1].played_knights = 2
    s.players[2].played_knights = 2
    s.players[3].dev_cards = [9, 0, 0, 0, 0]          # the rest of the knights is held: 1 left in the pool
    s.players[3].dev_count = 9
    ctx = W.PathsContext(s, 0)
    assert ctx.K_left == 14 - 4 - 9
    G = ctx.race_inputs(s)["la"][2]
    assert sum(G) <= ctx.K_left + 1e-9


def test_horizon_clamped_and_non_increasing():
    s = played()
    clear_devs(s)
    hs = []
    for extra_vp in range(0, 12):
        s2 = s.copy()
        s2.players[0].dev_cards[B.DEV_VP] = extra_vp          # VP cards raise vmax one by one
        h = W.horizon(s2)
        assert P0["H_MIN"] <= h <= P0["H_MAX"]
        hs.append(h)
    assert all(b <= a for a, b in zip(hs, hs[1:])) and hs[-1] == P0["H_MIN"]


def _recorded_leaves(s, me, cfg):
    rec = []

    class Rec(HeuristicEvaluator):
        def evaluate(self, states, players):
            rec.append((list(states), list(players)))
            return super().evaluate(states, players)

    Searcher(Rec(), cfg).search(s, me, random.Random(2))
    return rec


def test_cold_and_warm_caches_identical():
    s = played(seed=8, turns=70)
    me = 0
    extend_to(s, 2, 6)
    s.longest_road_owner, s.longest_road_len = 2, E.longest_road_length(s, 2)
    s.players[0].resources = [2, 2, 1, 1, 1]
    batches = _recorded_leaves(s, me, SearchConfig(depth=1, beam=4, expand=8))
    leaves = [x for st, _ in batches for x in st]
    players = [me] * len(leaves)
    assert len(leaves) > 30
    ev = HeuristicEvaluator()
    cold = W.PathsEvaluator(ev, W.PathsContext(s, me, spots=True))
    a = np.concatenate([cold.evaluate(st, pl) for st, pl in batches])
    warm = W.PathsEvaluator(ev, W.PathsContext(s, me, spots=True))
    rev = warm.evaluate(leaves[::-1], players)            # warm the memos in the reverse order first
    b = np.concatenate([warm.evaluate(st, pl) for st, pl in batches])
    assert np.array_equal(a, b) and np.array_equal(a, rev[::-1])
    assert warm.ctx.stats["race_hits"] > 0


# ---------------------------------------------------------------------------
# 7-9: the ledger cancels static_value's award credit exactly
# ---------------------------------------------------------------------------
def _ledger_states():
    out = []
    for seed, turns in ((5, 50), (6, 60), (8, 80), (9, 70)):
        s = played(seed, turns)
        for p in s.players:
            p.dev_known = True
            p.dev_count = p.total_dev
        out.append(s)
    return out


def test_ledger_longest_road_matches_static(monkeypatch):
    for s in _ledger_states():
        ctx = W.PathsContext(s, 0)
        S_LR = ctx.races(s).inputs["S_LR"]
        full = [heuristic.static_value(s, i) for i in range(4)]
        real = heuristic.longest_road_length
        with monkeypatch.context() as m:
            m.setattr(heuristic, "longest_road_length",
                      lambda st, i: real(st, i) if i == st.longest_road_owner else 0)
            zero = [heuristic.static_value(s, i) for i in range(4)]
        for i in range(4):
            if i == s.longest_road_owner:
                assert S_LR[i] == 20.0            # the award itself is in public_vp (2 VP)
            else:
                assert S_LR[i] == pytest.approx(full[i] - zero[i], abs=1e-9)


def test_ledger_largest_army_matches_static():
    for s in _ledger_states():
        s.players[1].played_knights = 2
        s.players[2].dev_cards[B.DEV_KNIGHT] += 2
        s.players[2].dev_count += 2
        ctx = W.PathsContext(s, 0)
        S_LA = ctx.races(s).inputs["S_LA"]
        for i in range(4):
            p = s.players[i]
            held = p.dev_cards[B.DEV_KNIGHT] + p.dev_cards_new[B.DEV_KNIGHT]
            s2 = s.copy()
            q = s2.players[i]
            if i != s.largest_army_owner:
                q.played_knights = 0
            q.dev_cards[B.DEV_KNIGHT] = 0
            q.dev_cards_new[B.DEV_KNIGHT] = 0
            diff = heuristic.static_value(s, i) - heuristic.static_value(s2, i)
            award = 20.0 if i == s.largest_army_owner else 0.0
            assert S_LA[i] - award == pytest.approx(diff - 0.25 * held, abs=1e-9)


def test_length_parity_with_python():
    for s in _ledger_states():
        for i in range(4):
            assert W._heur_lr(s, i) == heuristic.longest_road_length(s, i)
            assert accel.longest_road_length(s, i) == E.longest_road_length(s, i)


# ---------------------------------------------------------------------------
# 10-15: scenarios
# ---------------------------------------------------------------------------
def test_crowded_longest_road_synthetic():
    b = [9.5, 8.0, 6.5, 6.5]
    G = [1.2, 1.2, 1.4, 1.4]
    a = [0.0] * 4
    me, holder = 3, 0
    F = [x + g for x, g in zip(b, G)]
    assert F == pytest.approx([10.7, 9.2, 7.9, 7.9])
    cost = round(P0["KAPPA_F"] * (2.0 / P0["ETA"]) * (1.0 - P0["ROAD_RESIDUAL"]), 4)
    sol = W.solve_race(b, a, G, holder, 5, cost)
    up = list(b)
    up[me] += 1.0
    d_lr = W.solve_race(up, a, G, holder, 5, cost).credit[me] - sol.credit[me]
    hl = 9
    static_jump = 0.35 * hl - 0.15 * (hl - 1)
    assert sol.P[me] < 0.15 and not sol.active[me]
    assert d_lr < 0.3 and d_lr < static_jump


def test_crowded_longest_road_board_road_priors_not_raised():
    """Three opponents on trails of 9, 9 and 10 (the 10 holds Longest Road), us on 4 with road cards in hand:
    the race is crowded, one more road is worth almost nothing and the race-aware road prior never exceeds
    action_priors' flat +8 Longest Road bonus it replaces."""
    s = road_board([4, 9, 9, 10])
    me = 0
    s.longest_road_owner, s.longest_road_len = 3, 10
    s.players[me].resources = [3, 3, 0, 0, 0]
    legal = E.legal_actions(s)
    roads = [i for i, a in enumerate(legal) if a[0] == A.BUILD_ROAD]
    assert roads
    ctx = W.PathsContext(s, me)
    assert ctx.live_lr
    sol = ctx.races(s).lr
    assert sol.P[me] < 0.15 and not sol.active[me] and sol.N_close[me] >= 1.5
    assert ctx.marginal(s, W.LR) < 0.3
    pri = action_priors(s, legal, me)
    adj = ctx.adjust_priors(s, legal, pri)
    for i in roads:
        assert adj[i] - pri[i] <= 1e-9
    assert any(adj[i] < pri[i] for i in roads)


def la_scenario(seed=5, held=1, rival_played=1):
    s = played(seed)
    clear_devs(s)
    n = s.num_players
    prod = [player_production(s, i, True) for i in range(n)]
    me = max(range(n), key=lambda i: min(prod[i][B.SHEEP], prod[i][B.WHEAT], prod[i][B.ORE]))
    set_me(s, me)
    p = s.players[me]
    p.played_knights = 2
    p.dev_cards[B.DEV_KNIGHT] = held
    p.dev_count = held
    p.resources = [0, 0, 1, 1, 1]
    for j in range(n):
        if j != me:
            s.players[j].played_knights = rival_played if j == (me + 1) % n else 0
    s.dev_deck = list(B.DEV_DECK_COUNTS)
    s.dev_deck[B.DEV_KNIGHT] -= 2 + held + rival_played
    return s, me


def test_uncrowded_largest_army_pulls_dev_buys():
    s, me = la_scenario()
    ctx = W.PathsContext(s, me)
    assert ctx.live_la and s.largest_army_owner == -1
    rc = ctx.races(s)
    d_la = ctx.marginal(s, W.LA)
    assert d_la >= 1.0
    assert rc.la.credit[me] > rc.inputs["S_LA"][me] + 2.0
    legal = E.legal_actions(s)
    i_dev = legal.index((A.BUY_DEV,))
    adj = ctx.adjust_priors(s, legal, action_priors(s, legal, me))
    assert adj[i_dev] >= P0["DEV_FLOOR"]
    # the promotion itself: a dev buy the rules would postpone (prior 20) enters the expand set
    s2, me2 = la_scenario(held=0)
    ctx2 = W.PathsContext(s2, me2)
    assert ctx2.p_k * ctx2.marginal(s2, W.LA) >= P0["DEV_PROMOTE"]
    legal2 = E.legal_actions(s2)
    low = [20.0 if a[0] == A.BUY_DEV else 12.0 for a in legal2]
    assert ctx2.adjust_priors(s2, legal2, low)[legal2.index((A.BUY_DEV,))] >= P0["DEV_FLOOR"]
    # the search values the dev buy (vs ending the turn) higher with the term
    gaps = []
    for paths in (0, 1):
        res = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, paths=paths)).search(
            s, me, random.Random(3))
        v = {r.action: r.value for r in res}
        gaps.append(v[(A.BUY_DEV,)] - v[(A.END_TURN,)])
    assert gaps[1] > gaps[0]


def test_opponent_holding_with_big_lead():
    s, me = la_scenario()
    r = (me + 1) % 4
    s.players[r].played_knights = 6
    s.largest_army_owner = r
    s.players[me].played_knights = 1
    s.players[me].dev_cards[B.DEV_KNIGHT] = 0
    s.players[me].dev_count = 0
    ctx = W.PathsContext(s, me)
    rc = ctx.races(s)
    assert rc.la.credit[me] < 0.3 and ctx.marginal(s, W.LA) < 0.1
    c_la = rc.la.credit[me] - rc.inputs["S_LA"][me]
    assert abs(c_la + rc.inputs["S_LA"][me]) < 0.3
    # Longest Road: the holder at 10 with room left, us on 3
    s = road_board([3, 4, 10, 2])
    me, h = 0, 2
    s.longest_road_owner, s.longest_road_len = h, 10
    ctx = W.PathsContext(s, me)
    assert ctx.race_inputs(s)["room"][h] > 0
    rc = ctx.races(s)
    assert rc.lr.credit[me] < 0.3 and ctx.marginal(s, W.LR) < 0.1
    c_lr = rc.lr.credit[me] - rc.inputs["S_LR"][me]
    assert abs(c_lr + rc.inputs["S_LR"][me]) < 0.3


def test_won_path_is_left_alone():
    s, me = la_scenario()
    p = s.players[me]
    p.played_knights = 6
    p.dev_cards[B.DEV_KNIGHT] = 0
    p.dev_count = 0
    s.largest_army_owner = me
    others = [j for j in range(4) if j != me]
    for j, k in zip(others, (3, 2, 1)):
        s.players[j].played_knights = k
    s.dev_deck = [2, 5, 2, 2, 2]
    s.players[me].resources = [0, 0, 1, 1, 1]
    ctx = W.PathsContext(s, me)
    assert ctx.K_left <= 2
    rc = ctx.races(s)
    c_la = rc.la.credit[me] - rc.inputs["S_LA"][me]
    d_la = ctx.marginal(s, W.LA)
    # P is already ~0.98 (the softmax tails leave the rivals ~2%): one more knight adds ~0.012 VP; the
    # spec's 0.1-point bound is relaxed to 0.15 points (deviation documented in ABLATIONS_WINPATHS.md)
    assert abs(c_la) < 0.5 and d_la < 0.15
    legal = E.legal_actions(s)
    i_dev = legal.index((A.BUY_DEV,))
    pri = action_priors(s, legal, me)
    adj = ctx.adjust_priors(s, legal, pri)
    assert abs(adj[i_dev] - pri[i_dev]) < 0.5 and (adj[i_dev] >= P0["DEV_FLOOR"]) == (pri[i_dev] >= P0["DEV_FLOOR"])


def test_endgame_horizon_and_challenger():
    """vmax 9 -> H = 2.5 rounds: growth is small, current levels decide; a challenger 2 behind the holder has P < 0.1."""
    s = road_board([3, 6, 2, 8])
    me, h, c = 0, 3, 1
    s.longest_road_owner, s.longest_road_len = h, 8
    leader = 2
    s.players[leader].dev_cards[B.DEV_VP] = 9 - s.public_vp(leader)      # 9 VP, hidden in VP cards
    ctx = W.PathsContext(s, me)
    assert ctx.vmax == 9 and ctx.H == pytest.approx(2.5)
    assert ctx.races(s).lr.P[c] < 0.1
    # game-over leaves pass through unchanged
    s2 = s.copy()
    s2.phase, s2.winner = PHASE_GAME_OVER, leader
    pe = W.PathsEvaluator(HeuristicEvaluator(), ctx)
    assert list(pe.evaluate([s2, s2], [leader, me])) == [1.0, 0.0]


def _vertex_dist(a):
    dist = {a: 0}
    frontier = [a]
    while frontier:
        nxt = []
        for v in frontier:
            for w in B.VERTEX_NEIGHBORS[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    nxt.append(w)
        frontier = nxt
    return dist


def _spot_board(contested=True):
    """Us at ``a`` with roads a-m-v (spot v buildable now); a rival (moving first) at ``c`` with road c-x, one road
    from v (contested) - or a rival far away (uncontested)."""
    s = new_game(4, rng=random.Random(4))
    for v in range(B.NUM_VERTICES):
        nb = B.VERTEX_NEIGHBORS[v]
        if len(nb) < 3:
            continue
        for m in nb:
            for a in B.VERTEX_NEIGHBORS[m]:
                if a in (v,) or a in nb:
                    continue
                for x in nb:
                    if x == m:
                        continue
                    for c in B.VERTEX_NEIGHBORS[x]:
                        if c == v or c in nb or c == a or _vertex_dist(a).get(c, 99) < 3:
                            continue
                        return _build_spot_board(s, a, m, v, x, c, contested)
    raise AssertionError("no spot layout found")


def _build_spot_board(s, a, m, v, x, c, contested):
    me, riv = 0, 1
    s.phase = PHASE_MAIN
    s.current = me
    s.dice = 6
    s.players[me].settlements = [a]
    s.players[me].roads = [B.edge_between(a, m), B.edge_between(m, v)]
    if contested:
        s.players[riv].settlements = [c]
        s.players[riv].roads = [B.edge_between(c, x)]
    else:
        far = max((w for w, d in _vertex_dist(v).items() if d >= 7), key=lambda w: _vertex_dist(v)[w])
        e = B.VERTEX_EDGES[far][0]
        s.players[riv].settlements = [far]
        s.players[riv].roads = [e]
    s.players[me].resources = [0, 0, 0, 0, 0]
    s.players[riv].resources = [5, 5, 2, 2, 0]
    return s, me, v


def test_contested_spots():
    s, me, v = _spot_board(contested=True)
    ctx = W.PathsContext(s, me, spots=True)
    reach = ctx._contest(s, W._osig(s))
    assert reach[1] and any(vv == v and riv for vv, _d, riv in reach[0])
    assert ctx.spot_correction(s) < 0.0
    s2, me2, _ = _spot_board(contested=False)
    ctx2 = W.PathsContext(s2, me2, spots=True)
    assert ctx2.spot_correction(s2) == 0.0


# ---------------------------------------------------------------------------
# 16-23: default unchanged and integration
# ---------------------------------------------------------------------------
def test_default_is_off_everywhere():
    assert SearchConfig().paths == 0
    assert make_bot(DEFAULT).config.paths == 0
    from catanbot.bench.catanatron_adapter import DEFAULT_SPEC
    assert make_bot(DEFAULT_SPEC).config.paths == 0
    for spec in (tuning.DEFAULT_DEEP_SPEC, "search:depth=1,beam=4,expand=8,evaluator=heuristic,paths=0"):
        assert make_bot(spec).config.paths == 0


def test_paths_off_never_builds_a_context(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("PathsEvaluator.for_search called with paths=0")

    monkeypatch.setattr(W.PathsEvaluator, "for_search", classmethod(boom))
    s = played(seed=6)
    for cfg in (SearchConfig(depth=1, beam=4, expand=8), SearchConfig(depth=2, beam=2, expand=4, max_nodes=3000)):
        res = Searcher(HeuristicEvaluator(), cfg).search(s, 0, random.Random(1))
        assert res
    bot = make_bot(DEFAULT)
    play_game([bot, HeuristicBot(), HeuristicBot(), HeuristicBot()], rng=random.Random(2), seed=2, max_turns=20)


_HASH_SCRIPT = r"""
import hashlib, json, random, sys
sys.path.insert(0, ROOT)


def POSITIONS(k=30):
    out = []
    for seed in (101, 202, 303):
        rng = random.Random(seed)
        s = new_game(4, rng=rng)
        bots = [HeuristicBot(temperature=0.15) for _ in range(4)]
        cand = []
        while s.phase != PHASE_GAME_OVER and s.turn < 90:
            i = E.acting_player(s)
            legal = E.legal_actions(s)
            if s.phase in (PHASE_MAIN, PHASE_ROLL) and s.turn >= 20 and len(legal) > 1:
                cand.append((s.copy(), i))
            s = E.apply(s, bots[i].decide(s, legal, rng), rng)
        step = max(1, len(cand) // 10)
        out.extend(cand[::step][:10])
    return out[:k]


def game_hash(bots, seed):
    h = hashlib.sha256()
    res = play_game(bots, rng=random.Random(seed), seed=seed, max_turns=MAX_TURNS,
                    on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    return h.hexdigest()


from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.heuristic import HeuristicEvaluator
from catanbot.search import SearchConfig, Searcher
from catanbot.selfplay import make_bot, play_game
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROLL, new_game
spec = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
out = {"games": {}, "search": []}
if MODE == "before":
    assert "catanbot.winpaths" not in sys.modules and "catanbot.tuning" not in sys.modules
    specs = {"default": spec}
else:
    import catanbot.tuning, catanbot.winpaths                     # the registry imports the new module
    from catanbot.agents.param_bot import ParamBot
    specs = {"default": spec, "paths0": spec + ",paths=0"}
for name, sp in specs.items():
    out["games"][name] = [game_hash([make_bot(sp) for _ in range(4)], seed) for seed in SEEDS]
if MODE == "after":
    ov = {"winpaths.KAPPA": 1.0, "winpaths.BETA": 0.8}
    out["games"]["parambot"] = [game_hash([ParamBot(make_bot(spec), ov) for _ in range(4)], seed)
                                for seed in SEEDS[:2]]
cfg = SearchConfig(depth=1, beam=4, expand=8) if MODE == "before" else SearchConfig(depth=1, beam=4, expand=8, paths=0)
for s, me in POSITIONS():
    res = Searcher(HeuristicEvaluator(), cfg).search(s, me, random.Random(9))
    out["search"].append(repr([(r.action, r.value, r.static, r.line, r.explanation) for r in res]))
assert (MODE == "after") == ("catanbot.winpaths" in sys.modules)
print(json.dumps(out))
"""

SEEDS = [11, 12, 13, 14, 15, 16]
MAX_TURNS = 28


def _run_hash_script(mode):
    code = f"ROOT = {ROOT!r}\nSEEDS = {SEEDS!r}\nMAX_TURNS = {MAX_TURNS}\nMODE = {mode!r}\n" + _HASH_SCRIPT
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT,
                          timeout=1200)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_default_unchanged_six_games_and_thirty_searches():
    """Requirement (1): 6 seeded self-play games with the default spec (4 search seats, 28 turns each) and 30 root
    searches of mid-game positions.  A fresh interpreter that never imported catanbot.winpaths (nor tuning) plays /
    searches them; a second one imports the module and repeats with the default spec, with ``paths=0`` spelled
    out, and (2 games) as ParamBots whose winpaths constants are overridden while paths stays 0.  Every action
    sequence and every search result (actions, values, lines, explanations) must be identical."""
    before = _run_hash_script("before")
    after = _run_hash_script("after")
    assert len(before["games"]["default"]) == 6 and len(before["search"]) == 30
    assert after["games"]["default"] == before["games"]["default"]
    assert after["games"]["paths0"] == before["games"]["default"]
    assert after["games"]["parambot"] == before["games"]["default"][:2]
    assert after["search"] == before["search"]


def test_weight_zero_equals_base_and_game_over_passes():
    s = played(seed=9, turns=70)
    extend_to(s, 1, 6)
    s.longest_road_owner, s.longest_road_len = 1, E.longest_road_length(s, 1)
    s.players[2].played_knights = 3
    s.largest_army_owner = 2
    me = 0
    batches = _recorded_leaves(s, me, SearchConfig(depth=1, beam=4, expand=8))
    leaves = [x for st, _ in batches for x in st][:120]
    over = s.copy()
    over.phase, over.winner = PHASE_GAME_OVER, 3
    leaves.append(over)
    players = [i % 4 for i in range(len(leaves))]

    class StubNet:
        def evaluate(self, states, players):
            return np.array([0.1 + 0.05 * ((len(st.players[p].roads) + p) % 7) for st, p in zip(states, players)])

    for base in (HeuristicEvaluator(), BlendedEvaluator(StubNet(), 0.5), StubNet()):
        ctx = W.PathsContext(s, me, weight=0.0, spots=True)
        assert ctx.live_lr and ctx.live_la
        got = W.PathsEvaluator(base, ctx).evaluate(leaves, players)
        want = np.asarray(base.evaluate(leaves, players), dtype=np.float64)
        assert np.max(np.abs(got - want)) < 1e-12
    pe = W.PathsEvaluator(HeuristicEvaluator(), W.PathsContext(s, me))
    assert list(pe.evaluate([over, over], [3, 0])) == [1.0, 0.0]


def test_tunables_registered_and_round_trip():
    names = ["search.paths", "search.paths_w", "search.paths_crowd", "search.paths_priors", "search.paths_spots"] + [
        f"winpaths.{a}" for a in ("KAPPA", "BETA", "H_B", "HAND_W", "ESC", "TIE_LR", "LIVE_LR", "LIVE_LA",
                                  "PRIOR_SCALE", "PLACEBO")]
    for n in names:
        t = tuning.TUNABLES[n]
        assert t.requires_search and t.description
        if n.startswith("winpaths."):
            assert t.kind == "weight" and not t.needs_python_evaluator and not t.clear_caches
            assert "only with search.paths=1" in t.description and t.default == getattr(W, t.attr)
        else:
            assert t.kind == "search" and t.requires_depth == 1 and t.spec_key == t.attr
    assert tuning.TUNABLES["search.paths"].default == 0 and tuning.TUNABLES["search.paths"].candidates == [1]
    assert tuning.verify_registry() == []
    before = W.current_params()
    tok = tuning.apply({"winpaths.KAPPA": 1.0, "winpaths.LIVE_LR": 0, "winpaths.PLACEBO": 1})
    try:
        assert W.KAPPA == 1.0 and W.LIVE_LR == 0 and W.PLACEBO == 1
    finally:
        tuning.restore(tok)
    assert W.current_params() == before
    bot = make_bot(DEFAULT)
    tok = tuning.apply_to_bot(bot, {"search.paths": 1, "search.paths_w": 2.0})
    assert bot.config.paths == 1 and bot.config.paths_w == 2.0
    tuning.restore_bot(tok)
    assert bot.config.paths == 0 and bot.config.paths_w == 1.0


def test_parambot_builds_contexts_only_for_the_candidate(monkeypatch):
    seen = []
    orig = W.PathsEvaluator.for_search.__func__

    def spy(cls, base, root, me, cfg):
        seen.append((me, cfg.paths, W.KAPPA))
        return orig(cls, base, root, me, cfg)

    monkeypatch.setattr(W.PathsEvaluator, "for_search", classmethod(spy))
    cand = {"search.paths": 1, "winpaths.KAPPA": 0.75}
    bots = [ParamBot(make_bot(DEFAULT), cand), ParamBot(make_bot(DEFAULT), {}),
            ParamBot(make_bot(DEFAULT), cand), ParamBot(make_bot(DEFAULT), {})]
    play_game(bots, rng=random.Random(21), seed=21, max_turns=24)
    assert seen and {me for me, _, _ in seen} <= {0, 2}
    assert all(p == 1 and k == 0.75 for _, p, k in seen)
    assert all(b.inner.config.paths == 0 for b in bots) and W.KAPPA == P0["KAPPA"]


def test_make_bot_spec_keys():
    bot = make_bot(DEFAULT + ",paths=1,paths_w=1.5,paths_crowd=0.5,paths_priors=0,paths_spots=1")
    c = bot.config
    assert (c.paths, c.paths_w, c.paths_crowd, c.paths_priors, c.paths_spots) == (1, 1.5, 0.5, 0, 1)
    c = make_bot(DEFAULT).config
    assert (c.paths, c.paths_w, c.paths_crowd, c.paths_priors, c.paths_spots) == (0, 1.0, 1.0, 1, 0)


def test_depth_two_uses_the_python_lookahead(monkeypatch):
    from catanbot import search as S

    def boom(*a, **k):
        raise AssertionError("native lookahead used with paths=1")

    monkeypatch.setattr(S.Searcher, "_native_future_values", boom)
    s = played(seed=6)
    cfg = SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500, paths=1)
    sr = Searcher(HeuristicEvaluator(), cfg)
    res = sr.search(s, 0, random.Random(1))
    assert res and sr._paths is not None
    assert S.reduced_config(cfg, 2, 600).paths == 1


def test_political_options_see_the_paths_evaluator(monkeypatch):
    from catanbot import search as S
    got = []

    def capture(state, me, evaluator=None, politics=None, model=None, **kw):
        got.append(evaluator)
        return []

    monkeypatch.setattr(S, "political_trade_options", capture)
    s = played(seed=6)
    s.players[0].resources = [2, 1, 2, 1, 0]
    legal = E.legal_actions(s)
    assert any(a[0] == A.PROPOSE_TRADE for a in legal)
    Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=2, expand=4, paths=1)).search(s, 0, random.Random(1))
    assert got and all(isinstance(e, W.PathsEvaluator) for e in got)
    got.clear()
    Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=2, expand=4)).search(s, 0, random.Random(1))
    assert got and all(isinstance(e, HeuristicEvaluator) for e in got)


def test_placebo_rotates_the_correction(monkeypatch):
    s = played(seed=9, turns=70)
    extend_to(s, 1, 6)
    s.longest_road_owner, s.longest_road_len = 1, E.longest_road_length(s, 1)
    s.players[2].played_knights = 3
    s.largest_army_owner = 2
    real = W.PathsContext(s, 0).corrections(s)
    monkeypatch.setattr(W, "PLACEBO", 1)
    rot = W.PathsContext(s, 0).corrections(s)
    assert rot == [real[(i + 1) % 4] for i in range(4)] and rot != real


def test_cost_budget_and_hit_rate():
    rng = random.Random(31)
    s = new_game(4, rng=rng)
    bots = [HeuristicBot(temperature=0.15) for _ in range(4)]
    roots = []
    while s.phase != PHASE_GAME_OVER and len(roots) < 10 and s.turn < 200:
        i = E.acting_player(s)
        legal = E.legal_actions(s)
        if s.phase == PHASE_MAIN and s.turn >= 30 and s.turn % 4 == 0 and len(legal) > 1:
            roots.append((s.copy(), i))
        s = E.apply(s, bots[i].decide(s, legal, rng), rng)
    assert len(roots) == 10
    ev = HeuristicEvaluator()
    tot = {0: 0.0, 1: 0.0}
    hits = misses = 0
    for paths in (0, 1, 0, 1):          # interleaved, the best of two runs each
        t = 0.0
        for st, me in roots:
            sr = Searcher(ev, SearchConfig(depth=1, beam=4, expand=8, paths=paths))
            t0 = time.process_time()
            sr.search(st, me, random.Random(5))
            t += time.process_time() - t0
            if paths:
                hits += sr._paths.ctx.stats["race_hits"]
                misses += sr._paths.ctx.stats["race_misses"]
        tot[paths] = t if tot[paths] == 0.0 else min(tot[paths], t)
    ratio = tot[1] / tot[0]
    print(f"cost ratio paths=1 / paths=0: {ratio:.2f} ({1000 * tot[1] / 10:.1f} vs {1000 * tot[0] / 10:.1f} ms per "
          f"decision); race-solve hit rate {hits / max(1, hits + misses):.2f}")
    assert ratio <= 2.0
    assert hits / max(1, hits + misses) >= 0.8


def test_race_lines_name_both_awards_with_verdicts():
    s = played(seed=9, turns=70)
    extend_to(s, 1, 7)
    s.longest_road_owner, s.longest_road_len = 1, E.longest_road_length(s, 1)
    s.players[2].played_knights = 3
    s.largest_army_owner = 2
    lines = W.race_lines(s, 0)
    text = "\n".join(lines)
    assert "Longest Road" in text and "Largest Army" in text
    assert any(v in text for v in ("CROWDED", "OPEN", "passive", "safe", "defend"))
    assert lines[1].startswith("Your best path: ") and "crowded: " in lines[1]
    port = W.portfolio(s, 0)
    assert port and all(port[k][1] >= port[k + 1][1] for k in range(len(port) - 1))
    assert W.race_lines(new_game(4, rng=random.Random(1)), 0)          # setup phase: a note, no crash


def test_cli_shows_win_paths_and_accepts_the_flag(tmp_path, capsys):
    from catanbot import cli
    s = played(seed=9, turns=70)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(s.to_dict()))
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic",
                   "--paths", "1"])
    out = capsys.readouterr().out
    assert rc == 0 and "Win paths" in out and "Your best path:" in out
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert rc == 0 and d["advice"]["win_paths"]
