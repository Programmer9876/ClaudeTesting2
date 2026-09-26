"""Tests for the trades area (catanbot/acquisition.py; docs/PRIORITY_PLAN.md step 3), all off by default.

* default identity: the default bot's games reproduce the pinned pre-feature digests (tests/test_corrections.py,
  tests/test_counteroffers.py), also with every new switch spelled out at its default and as ParamBots with the new
  weights overridden while the switches stay off; a default game never imports the module or enters its code;
* acq.progress: horizon, production tails, conversions / bank cap, the 7-card gate, caches, the hub's evaluator
  sites, composition with paths and conv, the acq_w = 0 A/A, P = static's own progress term;
* acq.breadth: injected mixed-give offers (legal in both engines and in play_game, p_any as _trade_outcomes, caps,
  feed-the-leader veto, the bundle switch);
* acq_floor: the cheap dominance check, bank plans, the premium rule on answers and proposals, candidate_offers;
* acq.flow: the table and port_flow_value;
* acq.calib: calibration off = identical, updates, clips, per-game reset, ParamBot scoping, the rejection streak.
"""
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from catanbot import accel
from catanbot import acquisition as Q
from catanbot import actions as A
from catanbot import board as B
from catanbot import corrections as H
from catanbot import engine as E
from catanbot import opponent_model as OM
from catanbot import tuning
from catanbot.agents.param_bot import ParamBot, tuned_spec
from catanbot.heuristic import HeuristicEvaluator, _progress_to_build
from catanbot.opponent_model import OpponentModel
from catanbot.search import SearchConfig, Searcher, reduced_config
from catanbot.selfplay import _valid_extra_action, make_bot, play_game
from catanbot.state import PHASE_MAIN, PHASE_ROLL, PHASE_TRADE_RESPONSE, TradeOffer
from tests.test_conversion import V, blank, put
from tests.test_corrections import PINNED_GAMES, game_hash, positions, search_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = tuning.DEFAULT_SEARCH_SPEC
W_, BR, SH, WH, OR = B.WOOD, B.BRICK, B.SHEEP, B.WHEAT, B.ORE
OFF_KEYS = ",acq=0,acq_w=1,acq_self=1,acq_shapes=0,acq_breadth=0,acq_floor=0"


def vec(**kw):
    out = [0] * 5
    for k, v in kw.items():
        out[B.RESOURCE_INDEX[k]] = v
    return tuple(out)


# ---------------------------------------------------------------------------
# hand-built positions (every hex a desert unless a test gives it a resource: test_conversion's blank board)
# ---------------------------------------------------------------------------
def board(ports=None):
    """Seat 0 on V0 (wheat 6 / ore 8 / wheat 5) and V1 (ore 6 / wheat 9 / sheep 4); V2 wood/brick/wood for seat 1."""
    s = blank()
    put(s, V[0], [(WH, 6), (OR, 8), (WH, 5)])
    put(s, V[1], [(OR, 6), (WH, 9), (SH, 4)])
    put(s, V[2], [(W_, 6), (BR, 8), (W_, 4)])
    s.players[0].settlements = [V[0], V[1]]
    s.players[1].settlements = [V[2]]
    s.ports = dict(ports or {})
    for p in s.players:
        p.resources = [0] * 5
        p.hand_size = 0
        p.dev_known = True
    return s


def set_hand(s, i, hand):
    s.players[i].resources = list(hand)
    s.players[i].hand_size = sum(hand)
    s.bank = [B.BANK_PER_RESOURCE - sum(p.resources[r] for p in s.players) for r in range(5)]


# ---------------------------------------------------------------------------
# default identity
# ---------------------------------------------------------------------------
def test_default_games_reproduce_the_pinned_digests_with_the_switches_spelled_out():
    spec, want = PINNED_GAMES["default"]
    assert [game_hash([spec + OFF_KEYS] * 4, seed, 36) for seed in (11, 12)] == want[:2]


def test_counteroffers_pinned_digest_with_the_switches_spelled_out():
    from tests.test_counteroffers import PINNED
    specs, seed, max_turns, n_actions, digest = PINNED[0]
    sp = [s + OFF_KEYS if s.startswith("search") else s for s in specs]
    log = []
    play_game([make_bot(x) for x in sp], rng=random.Random(seed), seed=seed, max_turns=max_turns,
              on_action=lambda st, a, p: log.append((p, a)))
    assert (len(log), hashlib.sha256(repr(log).encode()).hexdigest()[:16]) == (n_actions, digest)


def test_parambot_with_the_new_weights_but_switches_off_plays_the_default():
    ov = {"acquisition.W_PREMIUM": 2.0, "search.acq_w": 0.5, "search.acq_self": 0,
          "opponent_model.CALIB_PRIOR": -1.0}
    h = hashlib.sha256()
    res = play_game([ParamBot(make_bot(DEFAULT), ov) for _ in range(4)], rng=random.Random(12), seed=12,
                    max_turns=36, on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    assert h.hexdigest()[:16] == PINNED_GAMES["default"][1][1]


_FRESH = r"""
import hashlib, json, random, sys
sys.path.insert(0, ROOT)
from catanbot.selfplay import make_bot, play_game
h = hashlib.sha256()
spec = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
res = play_game([make_bot(spec) for _ in range(4)], rng=random.Random(11), seed=11, max_turns=36,
                on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
print(json.dumps({"hash": h.hexdigest()[:16],
                  "imported": [m for m in ("catanbot.acquisition", "catanbot.corrections") if m in sys.modules]}))
"""


def test_default_game_never_imports_the_module():
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, "-c", f"ROOT = {ROOT!r}\n" + _FRESH], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"hash": PINNED_GAMES["default"][1][0], "imported": []}


def test_switches_off_never_enter_the_code(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("acquisition code reached with every switch off")

    for name in ("mixed_offers", "bank_plans", "dominated_by_bank", "horizon_k", "roll_tails", "port_flow_value"):
        monkeypatch.setattr(Q, name, boom)
    monkeypatch.setattr(Q.AcqContext, "__init__", boom)
    monkeypatch.setattr(Q.AcceptCalibrator, "__init__", boom)
    for name in ("_inject_shapes", "_floor_check", "_floor_proposals", "_floor_answers"):
        monkeypatch.setattr(Searcher, name, boom)
    for name in ("_calibrate", "_calibrated", "_calibrator"):
        monkeypatch.setattr(OpponentModel, name, boom)
    bots = [make_bot(DEFAULT + ",accept_bias=0.3,trade_eps=0.05"), make_bot(DEFAULT), make_bot(DEFAULT),
            make_bot(DEFAULT + ",trades=5")]
    res = play_game(bots, rng=random.Random(4), seed=4, max_turns=40)
    assert res.actions > 100


# ---------------------------------------------------------------------------
# acq.progress: the model
# ---------------------------------------------------------------------------
def test_horizon_k_for_every_seat_and_phase():
    s = board()
    s.phase, s.current = PHASE_MAIN, 0
    assert [Q.horizon_k(s, i, 0) for i in range(4)] == [4, 1, 2, 3]       # our turn, after our roll: us k = n
    s.phase = PHASE_ROLL
    assert Q.horizon_k(s, 0, 0) == 4                                    # every node of our turn: k = n
    s.current = 1                                                       # after our END_TURN: seat 1 rolls next
    assert [Q.horizon_k(s, i, 0) for i in range(4)] == [4, 1, 2, 3]
    s.phase = PHASE_MAIN                                                # seat 1 has rolled
    assert [Q.horizon_k(s, i, 0) for i in range(4)] == [3, 0, 1, 2]
    import inspect
    src = inspect.getsource(Q)
    assert "opponent_rolls_before_my_turn(" not in src and "seven_risk(" not in src   # no P7 anywhere


def test_own_seat_has_one_horizon_before_and_after_end_turn():
    s = board()
    s.phase, s.current, s.dice = PHASE_MAIN, 0, 6
    set_hand(s, 0, vec(sheep=3, wheat=1, ore=1))
    ctx = Q.AcqContext(s, 0, mode=2)
    after = E.apply(s, (A.END_TURN,))
    assert after.current == 1 and after.phase == PHASE_ROLL
    a, b = ctx.corrections(s), ctx.corrections(after)
    assert a is not None and a == b


def test_conversions_ports_and_the_bank_cap():
    city = B.COST_CITY
    hand = vec(sheep=4, wheat=2)                                        # city: have 2 wheat, missing 3 ore
    ok = [19] * 5

    def conv(ratios, bank=ok):
        d = Q.target_credit(hand, city, ratios, bank, None)
        return d["conv_int"] + d["conv_frac"]

    assert conv([4] * 5) == 1                                           # 4 sheep -> 1 ore at the bank
    assert conv([3] * 5) == pytest.approx(1 + 0.5 * (1 / 3))            # 3:1: one conversion + half the started one
    assert conv([4, 4, 2, 4, 4]) == 2                                   # 2:1 sheep port: two ore
    capped = [19, 19, 19, 19, 1]
    assert conv([4, 4, 2, 4, 4], capped) == 1 and conv([3] * 5, capped) == 1   # the bank holds one ore
    # adding a port never lowers E
    rng = random.Random(3)
    tgts = [(B.COST_CITY, 3.2), (B.COST_SETTLEMENT, 2.8), (B.COST_DEV, 1.0)]
    for _ in range(300):
        h = [rng.randint(0, 3) for _ in range(5)]
        base = [4] * 5
        e0 = Q.effective_progress(h, tgts, base, ok, None)[0]
        e3 = Q.effective_progress(h, tgts, [3] * 5, ok, None)[0]
        r = rng.randrange(5)
        e2 = Q.effective_progress(h, tgts, [2 if x == r else 4 for x in range(5)], ok, None)[0]
        assert e3 >= e0 - 1e-12 and e2 >= e0 - 1e-12
    # no surplus: mode 1 credits nothing, C = 0 exactly (E == P bit for bit)
    E1, P1 = Q.effective_progress(vec(wheat=1, ore=2), tgts, base, ok, None)
    assert E1 == P1


def test_effective_progress_without_credit_is_statics_progress_term():
    checked = 0
    for s, me in positions(k=12):
        p = s.players[me]
        tg = Q.targets(s, me)
        E0, P0 = Q.effective_progress(p.resources, tg, [4] * 5, [0] * 5, None)   # empty bank: no conversions
        assert P0 == _progress_to_build(s, me)
        checked += 1
    assert checked == 12


def _brute_tails(yields, k, r, max_j=3):
    tot = [0.0] * (max_j + 1)
    rolls = list(B.ROLL_PROB.items())
    for seq in itertools.product(rolls, repeat=k):
        pr = 1.0
        y = 0
        for d, pd in seq:
            pr *= pd
            y += 0 if d == 7 else yields[d][r]
        tot[min(max_j, y)] += pr
    return [sum(tot[j:]) for j in range(1, max_j + 1)]


def test_production_tails_are_exact_and_exclude_the_robber():
    s = board()
    s.players[0].cities = [V[0]]
    s.players[0].settlements = [V[1]]
    ys = Q.roll_yields(s, 0)
    assert ys[6][WH] == 2 and ys[6][OR] == 1 and ys[8][OR] == 2 and sum(ys[7]) == 0
    for k in range(1, 5):
        t = Q.roll_tails(ys, k)
        for r in (SH, WH, OR):
            assert t[r] == pytest.approx(_brute_tails(ys, k, r), abs=1e-12)
        for r in range(5):                                       # sum_j t(j) = E[min(Y, m)]
            for m in range(4):
                dist = _brute_tails(ys, k, r, max_j=m) if m else []
                assert sum(t[r][:m]) == pytest.approx(sum(dist), abs=1e-12)
    s.robber = B.VERTEX_HEXES[V[0]][1]                           # the ore 8 hex
    assert Q.roll_yields(s, 0)[8][OR] == 0 and Q.roll_yields(s, 0)[6][WH] == 2


def test_gate_above_seven_cards():
    s = board()
    s.phase, s.current = PHASE_MAIN, 0
    set_hand(s, 0, vec(sheep=4, wheat=2, ore=1))
    ctx = Q.AcqContext(s, 0, mode=2)
    assert ctx.corrections(s) is not None
    set_hand(s, 0, vec(sheep=5, wheat=2, ore=1))                   # 8 cards
    assert ctx.corrections(s) is None and ctx.stats["gated"] == 1


def test_target_cache_follows_a_road_that_opens_a_spot():
    s = board()
    s.phase, s.current = PHASE_MAIN, 0
    v0 = V[1]
    a = next(x for x in B.VERTEX_NEIGHBORS[v0])
    occ = s.occupied_vertices()
    b = next(x for x in B.VERTEX_NEIGHBORS[a]
             if x != v0 and not any(y in occ for y in list(B.VERTEX_NEIGHBORS[x]) + [x]))
    s.players[0].roads = [B.edge_between(v0, a)]
    set_hand(s, 0, vec(wood=1, brick=1, wheat=1, ore=4))
    ctx = Q.AcqContext(s, 0, mode=1)
    c1 = ctx.corrections(s)
    s2 = s.copy()
    s2.players[0].roads.append(B.edge_between(a, b))
    assert [c for c, _ in Q.targets(s2, 0)] == [B.COST_CITY, B.COST_SETTLEMENT, B.COST_DEV]
    assert B.COST_SETTLEMENT not in [c for c, _ in Q.targets(s, 0)]
    c2 = ctx.corrections(s2)
    assert c2 == Q.AcqContext(s2, 0, mode=1).corrections(s2) and c2 != c1


def test_cost_counts_are_deterministic_and_the_caches_hit():
    stats = []
    for rep in range(2):
        tot = {}
        for s, me in positions(k=8):
            sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, acq=2))
            sr.search(s, me, random.Random(3))
            if sr._corr is None:
                continue
            st = sr._corr.providers[0].stats
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
        stats.append(tot)
    assert stats[0] == stats[1]
    st = stats[0]
    assert st["evals"] > 100 and st["tail_misses"] < st["tail_calls"] / 5 and st["target_misses"] < st["target_calls"]


# ---------------------------------------------------------------------------
# acq.progress: search integration and composition
# ---------------------------------------------------------------------------
def test_zero_weight_is_an_exact_aa():
    pos = positions(k=10)
    assert search_hash(SearchConfig(depth=1, beam=4, expand=8, acq=2, acq_w=0.0), pos) == \
        search_hash(SearchConfig(depth=1, beam=4, expand=8), pos)


def test_reduced_config_copies_the_trade_fields():
    cfg = SearchConfig(acq=2, acq_w=0.5, acq_self=0, acq_shapes=0, acq_breadth=1, acq_floor=1, trade_proposals=3)
    sub = reduced_config(cfg, 2, 600)
    assert (sub.acq, sub.acq_w, sub.acq_self, sub.acq_shapes, sub.acq_breadth, sub.acq_floor) == (2, 0.5, 0, 1, 0, 1)
    assert sub.trade_proposals == 1
    d = reduced_config(SearchConfig(), 2, 600)
    assert (d.acq, d.acq_shapes, d.acq_breadth, d.acq_floor) == (0, 0, 0, 0)


def test_all_evaluator_sites_use_the_hub_with_acq(monkeypatch):
    from catanbot import search as S
    from tests.test_winpaths import played
    got = []
    monkeypatch.setattr(S, "political_trade_options",
                        lambda state, me, evaluator=None, politics=None, model=None, **kw: got.append(evaluator) or [])
    s = played(seed=6)
    s.players[0].resources = [2, 1, 2, 1, 0]
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=2, expand=4, acq=2))
    sr.search(s, 0, random.Random(1))
    assert got and all(isinstance(e, H.CorrectionHub) for e in got)
    assert [type(p) for p in sr._corr.providers] == [Q.AcqContext] and sr._corr.stats["corrected"] > 0
    monkeypatch.setattr(S.Searcher, "_native_future_values", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500, acq=2)
    assert Searcher(HeuristicEvaluator(), cfg).search(s, 0, random.Random(1))


def test_hub_composes_paths_conv_and_acq(monkeypatch):
    import catanbot.winpaths as W
    from catanbot.conversion import ConversionContext
    s, me = positions(k=1)[0]
    calls = []
    orig = W.PathsContext.adjust_priors

    def spy(self, state, legal, priors):
        calls.append(1)
        return orig(self, state, legal, priors)

    monkeypatch.setattr(W.PathsContext, "adjust_priors", spy)
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, paths=1, conv=1, acq=2))
    sr.search(s, me, random.Random(1))
    kinds = [type(p) for p in sr._corr.providers]
    assert kinds == [W.PathsContext, ConversionContext, Q.AcqContext] and calls
    # the hub's correction of a leaf is the sum of the three providers' (list order)
    ls = [st for st, _ in positions(k=6)]
    for leaf in ls:
        parts = [p.corrections(leaf) for p in sr._corr.providers]
        tot = sr._corr.corrections(leaf)
        want = [0.0] * 4
        for c in parts:
            if c:
                want = [a + b for a, b in zip(want, c)]
        assert (tot is None and not any(want)) or tot == pytest.approx(want, abs=1e-12)


def test_conv_and_acq_together_on_a_few_positions():
    """Composition (docs/STRATEGY.md): the two providers are summed per leaf; on a position where our hand holds
    convertible surplus only acq corrects the unbuilt leaves, where we build only conv changes the buildings."""
    s = board()
    s.phase, s.current, s.dice = PHASE_MAIN, 0, 6
    set_hand(s, 0, vec(sheep=4, wheat=2))
    from catanbot.conversion import ConversionContext
    hub = H.CorrectionHub(HeuristicEvaluator(), [ConversionContext(s, 0), Q.AcqContext(s, 0, mode=2)])
    c = hub.corrections(s)
    assert c is not None and c[0] > 0 and c[1:] == [0.0, 0.0, 0.0]
    assert hub.providers[0].corrections(s) is None                    # nothing built: conversion is silent
    city = s.copy()
    city.players[0].settlements = [V[1]]
    city.players[0].cities = [V[0]]
    set_hand(city, 0, vec(sheep=4))
    cc = [p.corrections(city) for p in hub.providers]
    assert cc[0] is not None and cc[1] is not None and hub.corrections(city)[0] == pytest.approx(cc[0][0] + cc[1][0])
    for s0, me in positions(k=4, seeds=(404,)):
        res = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, conv=1, acq=2)).search(
            s0, me, random.Random(2))
        assert res


def test_answering_an_offer_credits_our_port_on_the_reject_leaf():
    """Scope addition (1): when we answer an offer, the reject leaf keeps surplus that our own 2:1 port converts on our
    next turn, so accept-vs-reject sees the port alternative."""
    s = board(ports={V[1]: WH})
    s.current, s.phase, s.dice = 1, PHASE_MAIN, 6
    set_hand(s, 0, vec(wood=1, sheep=1, wheat=3))
    set_hand(s, 1, vec(brick=2))
    s1 = E.apply(s, (A.PROPOSE_TRADE, vec(brick=1), vec(wheat=2)))
    assert s1.phase == PHASE_TRADE_RESPONSE and E.acting_player(s1) == 0
    reject = E.apply(s1, (A.REJECT_TRADE,))
    ctx = Q.AcqContext(s1, 0, mode=2)
    c = ctx.corrections(reject)
    assert c is not None and c[0] > 0                                  # 2 surplus wheat -> brick via the port
    d = Q.target_credit(reject.players[0].resources, B.COST_SETTLEMENT, [4, 4, 4, 2, 4], reject.bank,
                        None)
    assert d["alloc"][BR] == 1


# ---------------------------------------------------------------------------
# acq.breadth
# ---------------------------------------------------------------------------
def breadth_position():
    """Seat 0 to move with wood / brick / sheep surplus and no wheat / ore; seats 1-3 hold wheat and ore."""
    s = board()
    s.phase, s.current, s.dice, s.turn = PHASE_MAIN, 0, 6, 30
    set_hand(s, 0, vec(wood=2, brick=2, sheep=2))
    set_hand(s, 1, vec(wheat=2, ore=2, wood=1))
    set_hand(s, 2, vec(wheat=1, ore=1))
    set_hand(s, 3, vec(ore=2, sheep=1))
    return s


def searcher(cfg=None, model=True):
    return Searcher(HeuristicEvaluator(), cfg or SearchConfig(depth=1, beam=4, expand=12, acq_shapes=1),
                    OpponentModel() if model else None)


def test_mixed_offers_shape_ranking_and_p_any_as_trade_outcomes():
    s = breadth_position()
    sr = searcher()
    inj = Q.mixed_offers(s, 0, sr._accept_probability, keep=10, min_p=0.0)
    assert inj and all(sum(1 for x in a[1] if x) == 2 and sum(a[1]) == 2 and sum(a[2]) == 1 for _p, a, _j in inj)
    assert [p for p, _a, _j in inj] == sorted((p for p, _a, _j in inj), reverse=True)
    for p_any, a, _j in inj:
        outs = sr._trade_outcomes(s, a, 0)
        assert sum(q for q, _ in outs) == pytest.approx(1.0)
        acc = [q for q, st in outs if st.players[0].resources != s.players[0].resources]
        assert acc and acc[0] == pytest.approx(p_any, abs=1e-12)


def test_injected_offers_are_legal_in_both_engines_and_in_play_game():
    s = breadth_position()
    legal = E.legal_actions(s)
    for _p, a, _j in Q.mixed_offers(s, 0, searcher()._accept_probability, keep=10, min_p=0.0):
        assert a not in legal and _valid_extra_action(s, a, legal)
        py = E.apply(s, a)
        assert py.phase == PHASE_TRADE_RESPONSE and py.pending_trade.give == list(a[1])
        if accel.AVAILABLE and accel.engine_available():
            cc = accel.engine_apply(s, a, random.Random(1))
            assert cc is not None and cc.to_dict() == py.to_dict()


def test_injected_offers_count_against_the_node_and_turn_caps():
    from catanbot.opponent_model import trade_stage_factor
    s = breadth_position()
    sr = searcher()
    legal = E.legal_actions(s)
    priors = sr._candidate_priors(s, legal, 0)
    f = trade_stage_factor(s)
    leg2, pri2 = sr._inject_shapes(s, legal, priors, 0, f)
    added = leg2[len(legal):]
    assert added and all(sum(1 for x in a[1] if x) == 2 for a in added) and leg2[:len(legal)] == legal
    plan_has = any(priors[i] >= 55.0 * f - 1e-9 for i, a in enumerate(legal) if a[0] == A.PROPOSE_TRADE)
    assert pri2[len(legal)] == pytest.approx((35.0 if plan_has else 55.0) * f)
    assert all(p == pytest.approx(35.0 * f) for p in pri2[len(legal) + 1:])
    assert all(a in sr._political_reasons for a in added)
    cands = sr._candidates(s, 0, False)
    assert sum(1 for a in cands if a[0] == A.PROPOSE_TRADE) <= 3
    s.trades_this_turn = sr.trade_cap(s)
    assert not any(a[0] == A.PROPOSE_TRADE for a in sr._candidates(s, 0, False))
    s.trades_this_turn = 0
    plain = searcher(SearchConfig(depth=1, beam=4, expand=12))
    assert not any(sum(1 for x in a[1] if x) == 2 for a in plain._candidates(s, 0, False) if a[0] == A.PROPOSE_TRADE)


def test_bundle_switch_widens_proposals_to_five():
    s = breadth_position()
    s.players[0].resources = [3, 3, 3, 0, 0]
    wide = searcher(SearchConfig(depth=1, beam=4, expand=20, acq_breadth=1))._candidates(s, 0, False)
    base = searcher(SearchConfig(depth=1, beam=4, expand=20))._candidates(s, 0, False)
    n_wide = sum(1 for a in wide if a[0] == A.PROPOSE_TRADE)
    n_base = sum(1 for a in base if a[0] == A.PROPOSE_TRADE)
    assert n_base == 3 and n_wide == 5


def test_no_pair_no_injection_and_the_feed_leader_veto():
    s = breadth_position()
    set_hand(s, 0, vec(wood=4))                                       # one surplus type only
    assert Q.mixed_offers(s, 0, searcher()._accept_probability, min_p=0.0) == []
    s = breadth_position()
    set_hand(s, 2, [0] * 5)
    set_hand(s, 3, [0] * 5)                                           # seat 1 is the only possible accepter ...
    occ = s.occupied_vertices()
    free = [v for v in range(B.NUM_VERTICES) if v not in occ and all(u not in occ for u in B.VERTEX_NEIGHBORS[v])]
    picks = []
    for v in free:
        if all(v not in B.VERTEX_NEIGHBORS[u] and v != u for u in picks):
            picks.append(v)
        if len(picks) == 4:
            break
    s.players[1].cities = picks                                       # ... and at 9 VP (8 in cities + V2)
    assert s.public_vp(1) == 9
    assert Q.mixed_offers(s, 0, searcher()._accept_probability, min_p=0.0) == []


def test_injected_offer_maps_to_offer_trade_in_the_adapter():
    from catanbot.bench import catanatron_adapter as ad
    if not ad.DOMESTIC_TRADING:
        pytest.skip(f"catanatron {ad.CATANATRON_VERSION} has no player-to-player trading")
    from catanatron.models.player import RandomPlayer
    game = ad.make_game([RandomPlayer(c) for c in ad.COLORS], seed=3)
    while game.state.num_turns < 12 and game.winning_color() is None:
        game.play_tick()
    st = game.state
    color = st.colors[st.current_turn_index]
    seat = st.color_to_index[color]
    me = ad.CatanbotPlayer(color, spec=DEFAULT, suppress_trades=False)
    for r, name in enumerate(ad.CB_TO_RESOURCE):
        st.player_state[f"P{seat}_{name}_IN_HAND"] = [1, 1, 0, 0, 0][r]
    a = (A.PROPOSE_TRADE, vec(wood=1, brick=1), vec(ore=1))
    ca = me._offer_action(a, st)
    assert ca is not None and ca.action_type == ad.AT_OFFER and tuple(ca.value) == (1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
    assert me._offer_action((A.PROPOSE_TRADE, vec(wood=1, sheep=1), vec(ore=1)), st) is None   # unaffordable


# ---------------------------------------------------------------------------
# acq_floor: dominance, bank plans, the premium rule
# ---------------------------------------------------------------------------
def test_dominated_by_bank_and_bank_plans():
    s = board(ports={V[1]: WH})                                       # 2:1 wheat for seat 0
    set_hand(s, 0, vec(wheat=4, ore=4, sheep=1))
    assert Q.dominated_by_bank(s, 0, vec(wheat=2), vec(brick=1))
    assert Q.dominated_by_bank(s, 0, vec(wheat=3), vec(brick=1))
    assert not Q.dominated_by_bank(s, 0, vec(wheat=1), vec(brick=1))
    assert Q.dominated_by_bank(s, 0, vec(ore=4), vec(brick=1))
    assert not Q.dominated_by_bank(s, 0, vec(ore=3), vec(brick=1))
    s.bank[BR] = 0
    assert not Q.dominated_by_bank(s, 0, vec(wheat=2), vec(brick=1)) and Q.bank_plans(s, 0, vec(brick=1)) == []
    s.bank[BR] = 5
    assert Q.bank_plans(s, 0, vec(brick=1)) == [vec(ore=4), vec(wheat=2)]
    two = Q.bank_plans(s, 0, vec(brick=2))
    assert vec(wheat=4) in two and vec(wheat=2, ore=4) in two and vec(ore=8) not in two
    set_hand(s, 0, vec(sheep=3))
    assert Q.bank_plans(s, 0, vec(brick=1)) == []                      # cannot pay any ratio


def answer_position(ports=None):
    """Seat 1 offers 1 brick for 2 wheat; seats 2 and 3 cannot pay, seat 0 (us) answers."""
    s = board(ports=ports)
    s.current, s.phase, s.dice, s.turn = 1, PHASE_MAIN, 6, 30
    set_hand(s, 0, vec(wood=1, sheep=1, wheat=3))
    set_hand(s, 1, vec(brick=2, ore=1))
    s1 = E.apply(s, (A.PROPOSE_TRADE, vec(brick=1), vec(wheat=2)))
    assert s1.phase == PHASE_TRADE_RESPONSE and E.acting_player(s1) == 0
    return s1


def test_floor_drops_accept_when_our_port_gives_the_same_card():
    s1 = answer_position(ports={V[1]: WH})
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, acq_floor=1), OpponentModel())
    cands = sr._candidates(s1, 0, False)
    assert (A.ACCEPT_TRADE,) not in cands and cands == [(A.REJECT_TRADE,)]
    why = sr._political_reasons[(A.REJECT_TRADE,)]
    assert "your port gives you the same 1 brick for 2 wheat on your next turn without helping" in why
    plain = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8), OpponentModel())
    assert (A.ACCEPT_TRADE,) in plain._candidates(s1, 0, False)
    # without the port the bank wants 4 wheat: no bank plan, the rule does not apply
    s2 = answer_position()
    sr2 = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, acq_floor=1), OpponentModel())
    assert (A.ACCEPT_TRADE,) in sr2._candidates(s2, 0, False)
    res = sr.search(s1, 0, random.Random(1))
    assert [r.action for r in res] == [(A.REJECT_TRADE,)] and "without helping" in res[0].explanation


def test_floor_premium_grows_with_the_partners_gain_and_danger(monkeypatch):
    s1 = answer_position(ports={V[1]: WH})
    s1.players[0].resources = [1, 0, 1, 5, 0]                        # 5 wheat: the port still has a plan
    s1.players[0].hand_size = 7
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, acq_floor=1), OpponentModel())
    item = [((A.ACCEPT_TRADE,), vec(wheat=2), vec(brick=1), 1, True)]
    with tuning.overridden({"acquisition.W_PREMIUM": 0.0}):
        base = sr._floor_check(s1, 0, item)[(A.ACCEPT_TRADE,)]
    assert Q.trade_premium(0.01, 2.0, 1.0) == pytest.approx(0.02) and Q.trade_premium(-0.01, 2.0, 1.0) == 0.0
    assert Q.trade_premium(0.01, 0.4, 1.0) < Q.trade_premium(0.01, 2.0, 1.0)
    assert not base[0]                                                # equal cards, the partner gains: bank wins
    # a 1-for-1 that beats the port on value passes at W = 0 and fails once the premium is large
    item1 = [((A.ACCEPT_TRADE,), vec(wheat=1), vec(brick=1), 1, True)]
    with tuning.overridden({"acquisition.W_PREMIUM": 0.0}):
        assert sr._floor_check(s1, 0, item1)[(A.ACCEPT_TRADE,)][0]
    from catanbot import danger as D
    monkeypatch.setattr(D, "danger_multiplier", lambda wp: 1e6)
    assert not sr._floor_check(s1, 0, item1)[(A.ACCEPT_TRADE,)][0]


def test_floor_drops_proposals_our_port_matches_and_names_the_bank_trade():
    s = board(ports={V[1]: WH})
    s.phase, s.current, s.dice, s.turn = PHASE_MAIN, 0, 6, 30
    set_hand(s, 0, vec(wheat=4, sheep=1))
    set_hand(s, 1, vec(brick=2, ore=2))
    set_hand(s, 2, vec(brick=1))
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=12, acq_floor=1), OpponentModel())
    two_for_one = (A.PROPOSE_TRADE, vec(wheat=2), vec(brick=1))
    assert two_for_one in E.legal_actions(s)
    ok = sr._floor_proposals(s, [two_for_one], 0, E.legal_actions(s))
    assert two_for_one not in ok
    assert "instead of a player trade" in sr._political_reasons[(A.BANK_TRADE, WH, BR)]
    cands = sr._candidates(s, 0, False)
    assert two_for_one not in cands
    plain = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=12), OpponentModel())
    assert any(a[0] == A.PROPOSE_TRADE for a in plain._candidates(s, 0, False))


def test_candidate_offers_port_floor():
    from catanbot.discard import needed_vector
    from catanbot.trading import candidate_offers
    s = board(ports={V[1]: WH})
    s.phase, s.current = PHASE_MAIN, 0
    set_hand(s, 0, vec(wheat=4, sheep=1))
    set_hand(s, 1, vec(brick=2, wood=2, ore=2))
    need = needed_vector(s, 0)
    plain = candidate_offers(s, 0, need)
    floor = candidate_offers(s, 0, need, port_floor=True)
    dom = [a for a in plain if Q.dominated_by_bank(s, 0, a[1], a[2])]
    assert dom and not any(Q.dominated_by_bank(s, 0, a[1], a[2]) for a in floor)
    assert [a for a in plain if a not in dom] == floor[:len(plain) - len(dom)]


# ---------------------------------------------------------------------------
# acq.flow
# ---------------------------------------------------------------------------
def test_flow_table_loads_and_is_monotone():
    assert len(Q.FLOW_T) == len(Q.FLOW_R) and Q.FLOW_T == sorted(Q.FLOW_T)
    assert all(a >= b for a, b in zip(Q.FLOW_R, Q.FLOW_R[1:])) and Q.FLOW_R[-1] == 0.0
    assert set(Q.FLOW_W) >= {"intercept", "share", "city"} and len(Q.FLOW_W["intercept"]) == 5
    assert Q.flow_remaining(-5) == Q.FLOW_R[0] and Q.flow_remaining(10 ** 6) == 0.0
    w = Q.flow_weights([0.1, 0.1, 0.1, 0.3, 0.2], True)
    assert sum(w) == pytest.approx(1.0) and min(w) > 0


def test_port_flow_value_nonnegative_zero_after_horizon_and_pure():
    s = board()
    s.turn = 8 + 20
    before = json.dumps(s.to_dict(), sort_keys=True)
    vals = {p: Q.port_flow_value(s, 0, p) for p in (B.PORT_GENERIC, W_, BR, SH, WH, OR)}
    assert json.dumps(s.to_dict(), sort_keys=True) == before
    assert all(v >= 0.0 for v in vals.values())
    assert all(Q.port_flow_value(s, 0, p, t=Q.FLOW_T[-1] + 1) == 0.0 for p in vals)
    # generic: one card saved on every trade still to come (all at 4:1 now)
    assert vals[B.PORT_GENERIC] == pytest.approx(Q.flow_remaining(20))
    # seat 0 produces wheat / ore only: a 2:1 wheat port beats a 2:1 wood port by far
    assert vals[WH] > vals[B.PORT_GENERIC] > vals[W_] and vals[W_] < 0.25 * vals[WH]
    s.ports = {V[0]: B.PORT_GENERIC}                                   # already 3:1: a second 3:1 saves nothing
    assert Q.port_flow_value(s, 0, B.PORT_GENERIC) == 0.0 and Q.port_flow_value(s, 0, WH) > 0.0


# ---------------------------------------------------------------------------
# acq.calib
# ---------------------------------------------------------------------------
def offer_state(give, get, proposer=0, hands=None):
    s = board()
    s.current, s.phase, s.dice, s.turn = proposer, PHASE_MAIN, 6, 30
    for i, h in (hands or {0: vec(wood=2, brick=2, sheep=2), 1: vec(wheat=2, ore=2), 2: vec(wheat=2, ore=2),
                           3: vec(wheat=2, ore=2)}).items():
        set_hand(s, i, h)
    s1 = E.apply(s, (A.PROPOSE_TRADE, give, get))
    return s, s1


def test_calibration_off_is_identical_and_keeps_no_state():
    from tests.test_winpaths import played
    rng = random.Random(5)
    m = OpponentModel()
    n = 0
    for seed in range(8):
        s = played(seed=seed, turns=40)
        for _ in range(25):
            j = rng.randrange(1, 4)
            give = [0] * 5
            get = [0] * 5
            give[rng.randrange(5)] = rng.randint(1, 2)
            r = rng.randrange(5)
            if give[r]:
                continue
            get[r] = 1
            p = m.predict_accept(s, j, give, get, proposer=0)
            logit, can = m.accept_terms(s, j, give, get, proposer=0)
            want = 0.0 if logit is None else max(0.0, min(1.0, (1.0 / (1.0 + np.exp(-logit))) * can))
            assert p == pytest.approx(want, abs=0.0, rel=1e-15)
            n += 1
    assert n > 150 and m.calibrator is None
    s0, s1 = offer_state(vec(wood=1), vec(wheat=1))
    m.observe(s1, (A.REJECT_TRADE,), E.acting_player(s1))
    assert m.calibrator is None


def test_calibration_learns_an_always_rejecting_seat():
    give, get = vec(wood=1, brick=1, sheep=1), vec(ore=1)                 # 3 for 1: generous
    s0, s1 = offer_state(give, get)
    m = OpponentModel(s1)
    j = E.acting_player(s1)
    other = next(x for x in (1, 2, 3) if x != j)
    p0 = m.predict_accept(s0, j, give, get, proposer=0)
    assert p0 >= 0.8
    with tuning.overridden({"opponent_model.CALIB_RATE": 1.0}):
        raw_other = m.accept_terms(s0, other, give, get, proposer=0)[0]
        for _ in range(20):
            m.observe(s1, (A.REJECT_TRADE,), j)
        p1 = m.predict_accept(s0, j, give, get, proposer=0)
        cal = m.calibrator
        assert p1 <= 0.1 and cal.updates == 20
        # the other seat moves only through the shared a0 and beta (its own a_j is untouched)
        name_o = OM._pname(s0, other)
        assert name_o not in cal.a
        assert m.predict_accept(s0, other, give, get, proposer=0) == pytest.approx(
            1.0 / (1.0 + np.exp(-(cal.a0 + cal.beta * m.accept_terms(s0, other, give, get, proposer=0)[0]))))
        assert m.accept_terms(s0, other, give, get, proposer=0)[0] == raw_other


def test_calibration_always_accepting_and_clips():
    cal = Q.AcceptCalibrator(0.0)
    for _ in range(2000):
        cal.update("x", -3.0, True, 1.0)
    assert cal.a0 <= 2.0 and cal.a["x"] <= 2.0 and Q.CALIB_CLIP_BETA[0] <= cal.beta <= Q.CALIB_CLIP_BETA[1]
    assert cal.a["x"] == 2.0 and cal.prob("x", -3.0) > 0.5
    for _ in range(2000):
        cal.update("x", 3.0, False, 1.0)
    assert cal.a0 >= -4.0 and cal.a["x"] == -4.0 and cal.beta >= 0.3


def test_calibration_uses_the_logit_before_note_accept(monkeypatch):
    give, get = vec(wood=1), vec(wheat=1)
    s0, s1 = offer_state(give, get)
    m = OpponentModel(s1)
    j = E.acting_player(s1)
    want = m.accept_terms(s1, j, s1.pending_trade.give, s1.pending_trade.get, proposer=0)[0]
    seen = []
    orig = Q.AcceptCalibrator.update

    def spy(self, name, raw, accepted, eta):
        seen.append(raw)
        return orig(self, name, raw, accepted, eta)

    monkeypatch.setattr(Q.AcceptCalibrator, "update", spy)
    with tuning.overridden({"opponent_model.CALIB_RATE": 1.0}):
        m.observe(s1, (A.ACCEPT_TRADE,), j)
    assert seen == [want]
    assert m.accept_terms(s1, j, s1.pending_trade.give, s1.pending_trade.get, proposer=0)[0] != want  # profile moved


class HalfBelief:
    def probability_has(self, j, r, n=1):
        return 0.5


def test_calibration_skips_uncertain_payers():
    give, get = vec(wood=1), vec(wheat=1)
    s0, s1 = offer_state(give, get)
    j = E.acting_player(s1)
    s1.players[j].hand_known = False
    m = OpponentModel(s1)
    with tuning.overridden({"opponent_model.CALIB_RATE": 1.0}):
        m.observe(s1, (A.REJECT_TRADE,), j, belief=HalfBelief())
        assert m.calibrator is not None and m.calibrator.updates == 0
        s1.players[j].hand_known = True
        m.observe(s1, (A.REJECT_TRADE,), j, belief=HalfBelief())
        assert m.calibrator.updates == 1


def test_calibration_resets_per_game_and_stays_on_candidate_seats():
    bot = make_bot(DEFAULT)
    bot.reset()
    bot.model.calibrator = Q.AcceptCalibrator(0.0)
    bot.reset()
    assert bot.model.calibrator is None
    ov = {"opponent_model.CALIB_RATE": 1.0}
    bots = [ParamBot(make_bot(DEFAULT), ov), ParamBot(make_bot(DEFAULT), {}), ParamBot(make_bot(DEFAULT), ov),
            ParamBot(make_bot(DEFAULT), {})]
    play_game(bots, rng=random.Random(8), seed=8, max_turns=40)
    assert bots[1].inner.model.calibrator is None and bots[3].inner.model.calibrator is None
    assert bots[0].inner.model.calibrator is not None and bots[0].inner.model.calibrator.updates > 0
    assert OM.CALIB_RATE == 0.0


def test_reject_streak_excludes_a_seat_until_it_accepts():
    give, get = vec(wood=1, brick=1), vec(ore=1)
    s0, s1 = offer_state(give, get)
    m = OpponentModel(s1)
    j = E.acting_player(s1)
    with tuning.overridden({"opponent_model.REJECT_STREAK": 3}):
        for k in range(3):
            assert m.predict_accept(s0, j, give, get, proposer=0) > 0.0
            m.observe(s1, (A.REJECT_TRADE,), j)
        assert m.predict_accept(s0, j, give, get, proposer=0) == 0.0
        sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8), m)
        assert sr._accept_probability(s0, j, TradeOffer(0, list(give), list(get)), 0) == 0.0
        m.observe(s1, (A.ACCEPT_TRADE,), j)
        assert m.predict_accept(s0, j, give, get, proposer=0) > 0.0
        assert m.calibrator.updates == 0                              # the streak alone never recalibrates


# ---------------------------------------------------------------------------
# registry, specs, observer
# ---------------------------------------------------------------------------
def test_registry_specs_and_round_trip():
    names = ("search.acq", "search.acq_w", "search.acq_self", "search.acq_shapes", "search.acq_breadth",
             "search.acq_floor")
    for n in names:
        t = tuning.TUNABLES[n]
        assert t.kind == "search" and t.requires_search and t.spec_key == n.split(".")[1]
        assert getattr(SearchConfig(), t.attr) == t.default == getattr(make_bot(DEFAULT).config, t.attr)
    for n in ("acquisition.W_PREMIUM", "opponent_model.CALIB_RATE", "opponent_model.CALIB_PRIOR",
              "opponent_model.REJECT_STREAK"):
        t = tuning.TUNABLES[n]
        assert t.kind == "weight" and not t.needs_python_evaluator and t.requires_search
    assert tuning.verify_registry() == []
    ov = {"search.acq": 2, "search.acq_w": 0.5, "search.acq_shapes": 1, "search.acq_breadth": 1,
          "search.acq_floor": 1, "acquisition.W_PREMIUM": 2.0, "opponent_model.CALIB_RATE": 1.0}
    spec = tuned_spec(DEFAULT, ov)
    bot = make_bot(spec)
    cfg = bot.config
    assert (cfg.acq, cfg.acq_w, cfg.acq_shapes, cfg.acq_breadth, cfg.acq_floor) == (2, 0.5, 1, 1, 1)
    assert bot.overrides == {"acquisition.W_PREMIUM": 2.0, "opponent_model.CALIB_RATE": 1.0}
    from catanbot.bench.catanatron_adapter import DEFAULT_SPEC
    c = make_bot(DEFAULT_SPEC).config
    assert (c.acq, c.acq_shapes, c.acq_breadth, c.acq_floor) == (0, 0, 0, 0)
    assert "trades" in tuning.SEAT_OBSERVERS


def test_trade_observer_counts():
    s0, s1 = offer_state(vec(wood=2), vec(wheat=1))
    ob = Q.TradeObserver(4)
    ob.on_action(s0, (A.PROPOSE_TRADE, vec(wood=2), vec(wheat=1)), 0)
    j = E.acting_player(s1)
    ob.on_action(s1, (A.ACCEPT_TRADE,), j)
    r = ob.result()
    assert r["proposals"][0] == 1 and r["accepts"][j] == 1 and r["proposals_mixed"] == [0] * 4
    s2 = board(ports={V[1]: W_})
    s2.phase, s2.current = PHASE_MAIN, 0
    set_hand(s2, 0, vec(wood=2))
    ob.on_action(s2, (A.PROPOSE_TRADE, vec(wood=2), vec(wheat=1)), 0)
    ob.on_action(s2, (A.PROPOSE_TRADE, vec(wood=1, sheep=1), vec(wheat=1)), 0)
    r = ob.result()
    assert r["proposals"][0] == 3 and r["proposals_dominated"][0] == 1 and r["proposals_mixed"][0] == 1


def test_cli_trade_floor_rejects_with_the_reason(tmp_path, capsys):
    from catanbot import cli
    s = board(ports={V[1]: WH})
    s.current, s.phase, s.dice, s.turn = 1, PHASE_MAIN, 6, 30
    set_hand(s, 0, vec(wood=1, sheep=1, wheat=3))
    set_hand(s, 1, vec(brick=2, ore=1))
    path = tmp_path / "state.json"
    path.write_text(json.dumps(s.to_dict()))
    me, them = s.players[0].color, s.players[1].color
    args = ["recommend", "--state", str(path), "--me", me, "--depth", "1", "--model", "heuristic",
            "--offer", f"{them}:give=brick:1;get=wheat:2", "--json"]
    assert cli.main(args + ["--trade-floor", "1"]) == 0
    d = json.loads(capsys.readouterr().out)
    top = d["actions"][0]
    assert "Reject" in top["text"] and "without helping" in top["explanation"]
    assert Q.W_PREMIUM == 1.0 and cli.main(args) == 0                  # restored; the default advisor is unchanged
    d0 = json.loads(capsys.readouterr().out)
    assert all("without helping" not in (a.get("explanation") or "") for a in d0["actions"])


def test_calibration_and_streak_observe_native_trades_through_the_adapter():
    """The Catanatron adapter delivers the native bots' REJECT_TRADE answers (with the pending offer) to observe, so
    acq_calib_native@value / acq_reject_streak_native@value exercise the calibration and the streak."""
    from catanbot.bench import catanatron_adapter as ad
    if not ad.DOMESTIC_TRADING:
        pytest.skip(f"catanatron {ad.CATANATRON_VERSION} has no player-to-player trading")
    from catanatron.players.value import ValueFunctionPlayer
    spec = DEFAULT + ",tune=opponent_model.CALIB_RATE:1;opponent_model.REJECT_STREAK:3"
    me = ad.CatanbotPlayer(ad.COLORS[0], spec=spec, suppress_trades=False)
    players = [me] + [ValueFunctionPlayer(c) for c in ad.COLORS[1:]]
    for p in players:
        if hasattr(p, "reset_state"):
            p.reset_state()
    game = ad.make_game(players, seed=11)
    while game.winning_color() is None and game.state.num_turns < 40:
        game.play_tick()
    cal = me.bot.inner.model.calibrator
    assert me.stats["offers"] > 0 and me.stats["observe_errors"] == 0
    assert cal is not None and cal.updates > 0 and cal.a and min(cal.a.values()) < 0.0
    assert any(v >= 1 for v in cal.streak.values()) and OM.CALIB_RATE == 0.0
