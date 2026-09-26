"""Tests for the offer cost (acquisition.OFFER_COST / OFFER_LEAK, Searcher._offer_value; docs/PRIORITY_PLAN.md
"Backlog": the trade-offer spam), off by default.

* default identity: both weights 0 by default; ParamBots / overrides with the weights spelled out at 0 play the pinned
  default games and searches; with the cost off the pricing code is never entered;
* the cost itself: offer_reveal (cards asked, +1 for a completed settlement / city), offer_cost;
* the valuation: min(E, V_no + P x (V_acc - V_rej)) - cost on hand-built outcome nodes (missing branches, no sibling);
  the pricing never raises a root value; a trade we value at or below nothing is never proposed;
* the combination: with every seat on a rejection streak (P = 0) the old search still proposes (a free, tied or
  deeper-searching offer), the priced one never does.
"""
import hashlib
import random

import pytest

from catanbot import acquisition as Q
from catanbot import actions as A
from catanbot import board as B
from catanbot import engine as E
from catanbot import opponent_model as OM
from catanbot import tuning
from catanbot.agents.param_bot import ParamBot
from catanbot.heuristic import HeuristicEvaluator
from catanbot.opponent_model import OpponentModel
from catanbot.search import SearchConfig, Searcher, _Node, _offer_pricing
from catanbot.selfplay import make_bot, play_game
from catanbot.state import PHASE_MAIN
from tests.test_corrections import PINNED_GAMES, PINNED_SEARCH, positions, search_hash

DEFAULT = tuning.DEFAULT_SEARCH_SPEC
ZERO = {"acquisition.OFFER_COST": 0.0, "acquisition.OFFER_LEAK": 0.0}
CFG = SearchConfig(depth=1, beam=4, expand=8)


@pytest.fixture(scope="module")
def mains():
    """The main-phase roots of test_corrections.positions (heuristic self-play, turn >= 20)."""
    return [(s, me) for s, me in positions() if s.phase == PHASE_MAIN]


def search(s, me, model=None, over=None):
    with tuning.overridden(over or {}):
        return Searcher(HeuristicEvaluator(), CFG, model or OpponentModel(s)).search(s, me, random.Random(9))


# ---------------------------------------------------------------------------
# off by default
# ---------------------------------------------------------------------------
def test_registered_off_by_default():
    for name in ZERO:
        t = tuning.TUNABLES[name]
        assert t.kind == "weight" and t.default == 0.0 and t.requires_search and not t.needs_python_evaluator
        assert all(v > 0 for v in t.candidates) and t.parse("0.004") == 0.004
    assert Q.OFFER_COST == 0.0 and Q.OFFER_LEAK == 0.0 and not Q.offer_cost_on()
    assert _offer_pricing() is None
    assert tuning.verify_registry() == []
    with tuning.overridden({"acquisition.OFFER_LEAK": 0.001}):
        assert _offer_pricing() is Q
    assert _offer_pricing() is None


def test_zero_weights_spelled_out_play_the_pinned_default_game():
    h = hashlib.sha256()
    res = play_game([ParamBot(make_bot(DEFAULT), dict(ZERO)) for _ in range(4)], rng=random.Random(12), seed=12,
                    max_turns=36, on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    assert h.hexdigest()[:16] == PINNED_GAMES["default"][1][1]


def test_zero_weights_reproduce_the_pinned_searches():
    pos = positions()
    cfg, want = PINNED_SEARCH["paths0"]
    with tuning.overridden(ZERO):
        assert search_hash(cfg, pos) == want


def test_cost_off_never_enters_the_pricing_code(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("offer-cost code reached with the cost off")

    for name in ("offer_cost", "offer_reveal"):
        monkeypatch.setattr(Q, name, boom)
    for name in ("_priced_children", "_offer_value"):
        monkeypatch.setattr(Searcher, name, boom)
    res = play_game([make_bot(DEFAULT) for _ in range(4)], rng=random.Random(4), seed=4, max_turns=30)
    assert res.actions > 100


# ---------------------------------------------------------------------------
# the cost
# ---------------------------------------------------------------------------
def _hand(s, i, **kw):
    v = [0] * 5
    for k, n in kw.items():
        v[B.RESOURCE_INDEX[k]] = n
    s.players[i].resources = v
    s.players[i].hand_size = sum(v)
    return tuple(v)


def _vec(**kw):
    v = [0] * 5
    for k, n in kw.items():
        v[B.RESOURCE_INDEX[k]] = n
    return tuple(v)


def test_offer_reveal_counts_asked_cards_and_a_completed_build(mains):
    s = mains[0][0].copy()
    _hand(s, 0, wheat=2, ore=2, sheep=1)
    assert Q.offer_reveal(s, 0, _vec(sheep=1), _vec(ore=1)) == 2           # completes a city
    assert Q.offer_reveal(s, 0, _vec(sheep=1), _vec(wood=1)) == 1          # asks one card, completes nothing
    assert Q.offer_reveal(s, 0, _vec(ore=1), _vec(wood=1, brick=1)) == 3    # 2 cards + a settlement completed
    _hand(s, 0, wheat=2, ore=3, sheep=1)
    assert Q.offer_reveal(s, 0, _vec(sheep=1), _vec(ore=1)) == 1           # the city was affordable already
    _hand(s, 0, wheat=2, ore=2)
    assert Q.offer_reveal(s, 0, _vec(wheat=1), _vec(ore=1)) == 1           # gives away what the city needs


def test_offer_cost_is_fixed_plus_leak(mains):
    s = mains[0][0].copy()
    _hand(s, 0, wheat=2, ore=2, sheep=1)
    give, get = _vec(sheep=1), _vec(ore=1)
    assert Q.offer_cost(s, 0, give, get) == 0.0
    with tuning.overridden({"acquisition.OFFER_COST": 0.004}):
        assert Q.offer_cost(s, 0, give, get) == pytest.approx(0.004)
    with tuning.overridden({"acquisition.OFFER_COST": 0.004, "acquisition.OFFER_LEAK": 0.001}):
        assert Q.offer_cost(s, 0, give, get) == pytest.approx(0.006)
        assert Q.offer_cost(s, 0, give, _vec(wood=1)) == pytest.approx(0.005)
    assert Q.OFFER_COST == 0.0 and Q.OFFER_LEAK == 0.0                      # restored


# ---------------------------------------------------------------------------
# the valuation
# ---------------------------------------------------------------------------
def _kid(state, value, traded):
    s = state.copy()
    if traded:
        s.players[0].resources[B.RESOURCE_INDEX["ore"]] += 1
        s.players[0].resources[B.RESOURCE_INDEX["sheep"]] -= 1
    n = _Node(s, 1.0, [])
    n.value = value
    return n


def test_offer_value_formula_on_hand_built_outcomes(mains):
    s = mains[0][0].copy()
    _hand(s, 0, wheat=2, ore=2, sheep=1)
    a = (A.PROPOSE_TRADE, _vec(sheep=1), _vec(ore=1))
    sr = Searcher(HeuristicEvaluator(), CFG)
    with tuning.overridden({"acquisition.OFFER_COST": 0.004}):
        sr._offer_q = _offer_pricing()

        def val(kids, base):
            ev = sum(p * k.value for p, k in kids)
            return sr._offer_value(s, a, kids, ev, base, 0)

        acc, rej = _kid(s, 0.60, True), _kid(s, 0.50, False)
        # both branches: base + P x (V_acc - V_rej) - cost, capped by the plain expectation
        assert val([(0.3, acc), (0.7, rej)], 0.50) == pytest.approx(0.50 + 0.3 * 0.10 - 0.004)
        # a rejected branch that searched deeper than the siblings no longer carries the offer
        deep = _kid(s, 0.58, False)
        assert val([(0.3, _kid(s, 0.57, True)), (0.7, deep)], 0.50) < 0.50
        # a trade we value below nothing is never worth it, whatever P
        assert val([(0.9, _kid(s, 0.45, True)), (0.1, rej)], 0.50) < 0.50
        # a rejected branch below the siblings (pruned) keeps the plain expectation as the cap
        low = _kid(s, 0.40, False)
        assert val([(0.5, acc), (0.5, low)], 0.50) == pytest.approx(0.5 * 0.60 + 0.5 * 0.40 - 0.004)
        # P(accept) about 0: only the rejected branch -> below not offering
        assert val([(1.0, rej)], 0.50) == pytest.approx(0.50 - 0.004)
        # P(accept) about 1: only the accepted branch -> its value net of the cost
        assert val([(1.0, acc)], 0.50) == pytest.approx(0.60 - 0.004)
        # no non-proposal sibling: the plain expectation net of the cost
        assert val([(0.3, acc), (0.7, rej)], float("-inf")) == pytest.approx(0.53 - 0.004)


def test_pricing_never_raises_an_offer_above_its_expectation_minus_the_cost(mains, monkeypatch):
    calls = []
    orig = Searcher._offer_value

    def spy(self, state, action, kids, ev, base, me):
        v = orig(self, state, action, kids, ev, base, me)
        calls.append((v, ev, self._offer_q.offer_cost(state, me, action[1], action[2])))
        return v

    monkeypatch.setattr(Searcher, "_offer_value", spy)
    changed = 0
    for s, me in mains:
        plain = search(s, me)
        priced = search(s, me, over={"acquisition.OFFER_COST": 0.003, "acquisition.OFFER_LEAK": 0.001})
        assert {r.action for r in plain} == {r.action for r in priced}     # the same candidates at the root
        changed += priced[0].action != plain[0].action
    assert calls and all(v <= ev - c + 1e-12 and c >= 0.004 - 1e-12 for v, ev, c in calls)
    assert changed > 0


def test_beam_ignores_rejected_outcomes_with_the_cost_on(mains, monkeypatch):
    """P(accept) = 0: a proposal's only outcome is the rejection (the parent again).  The old beam lets such nodes
    crowd out the real alternatives (some get expanded); with the cost on they never rank a group."""
    monkeypatch.setattr(OpponentModel, "predict_accept", lambda self, *a, **k: 0.0)
    seen = []
    orig = Searcher._backup

    def spy(self, node, me):
        if not node.line and node.children:
            seen.append([kids[0][1].children != [] for a, kids in node.children if a[0] == A.PROPOSE_TRADE])
        return orig(self, node, me)

    monkeypatch.setattr(Searcher, "_backup", spy)
    expanded = {}
    for key, over in (("off", {}), ("on", {"acquisition.OFFER_COST": 0.001})):
        seen.clear()
        for s, me in mains:
            search(s, me, over=over)
        expanded[key] = sum(sum(x) for x in seen)
        assert sum(len(x) for x in seen) > 0
    assert expanded["off"] > 0 and expanded["on"] == 0


def test_proposal_on_top_only_when_its_own_gain_beats_the_cost(mains, monkeypatch):
    """Every root proposal the priced search ranks first has P x (V_acc - V_rej) > cost."""
    seen = []
    orig = Searcher._offer_value

    def spy(self, state, action, kids, ev, base, me):
        v = orig(self, state, action, kids, ev, base, me)
        mine = state.players[me].resources
        p_acc = sum(p for p, k in kids if k.state.players[me].resources != mine)
        v_acc = [k.value for p, k in kids if k.state.players[me].resources != mine]
        v_rej = [k.value for p, k in kids if k.state.players[me].resources == mine]
        gain = p_acc * (v_acc[0] - (v_rej[0] if v_rej else base)) if v_acc else 0.0
        seen.append((id(state), action, v, gain, base))
        return v

    monkeypatch.setattr(Searcher, "_offer_value", spy)
    tops = 0
    for s, me in mains:
        seen.clear()
        res = search(s, me, over={"acquisition.OFFER_COST": 0.003})
        if res[0].action[0] != A.PROPOSE_TRADE:
            continue
        tops += 1
        mine = [x for x in seen if x[0] == id(s) and x[1] == res[0].action]
        assert mine and all(g > 0.003 for _i, _a, _v, g, base in mine if base > float("-inf"))
    assert tops > 0


# ---------------------------------------------------------------------------
# the combination with the rejection streak / calibration
# ---------------------------------------------------------------------------
def _streaked_model(s, me, streak=3):
    m = OpponentModel(s)
    cal = m._calibrator()
    for j in range(s.num_players):
        if j != me:
            for _ in range(streak):
                cal.note_answer(OM._pname(s, j), False)
    return m


def test_streak_plus_cost_stops_offering_the_old_search_does_not(mains):
    old = new = 0
    over = {"opponent_model.REJECT_STREAK": 3}
    for s, me in mains:
        r0 = search(s, me, _streaked_model(s, me), over)
        r1 = search(s, me, _streaked_model(s, me), dict(over, **{"acquisition.OFFER_COST": 0.001}))
        old += r0[0].action[0] == A.PROPOSE_TRADE
        new += r1[0].action[0] == A.PROPOSE_TRADE
    assert old > 0 and new == 0


def test_calibrated_zero_probability_seats_get_no_priced_offers(mains, monkeypatch):
    """Whatever supplies P (calibration, streak, the base model): P = 0 for every seat -> no offer with a cost on."""
    monkeypatch.setattr(OpponentModel, "predict_accept", lambda self, *a, **k: 0.0)
    for s, me in mains:
        res = search(s, me, over={"acquisition.OFFER_COST": 0.0005})
        assert res[0].action[0] != A.PROPOSE_TRADE
        assert all(r.value < res[0].value for r in res if r.action[0] == A.PROPOSE_TRADE)


def test_priced_parambot_game_runs_and_restores(mains):
    ov = {"acquisition.OFFER_COST": 0.003, "acquisition.OFFER_LEAK": 0.001, "opponent_model.REJECT_STREAK": 3}
    props = {0: 0, 1: 0}

    def count(st, a, p):
        if a[0] == A.PROPOSE_TRADE:
            props[p % 2] += 1

    bots = [ParamBot(make_bot(DEFAULT), ov) if i % 2 == 0 else make_bot(DEFAULT) for i in range(4)]
    res = play_game(bots, rng=random.Random(7), seed=7, max_turns=40, on_action=count)
    assert res.actions > 100 and props[1] > 0
    assert Q.OFFER_COST == 0.0 and Q.OFFER_LEAK == 0.0 and OM.REJECT_STREAK == 0
