"""Tests for the ports area (docs/PRIORITY_PLAN.md step 4): ``ports.constants`` (F2) and ``ports.surplus_value`` (F1)
in catanbot/placement.py and catanbot/heuristic.py, the ports flow provider, the advice and the rival-want helper
in catanbot/portvalue.py, and the gate script scripts/port_gate.py.

Everything is off by default: the default bot's pinned game digests (tests/test_corrections.py) are reproduced with
every new switch spelled out at its default, in a fresh interpreter the default bot never imports portvalue, and the
C++ static value (which keeps constexpr copies of the port constants) still matches the Python one.
"""
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys

import pytest

from catanbot import accel
from catanbot import acquisition as AQ
from catanbot import board as B
from catanbot import corrections as H
from catanbot import engine as E
from catanbot import heuristic
from catanbot import openings
from catanbot import placement as P
from catanbot import portvalue as PV
from catanbot import tuning
from catanbot import winpaths as W
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.param_bot import ParamBot
from catanbot.heuristic import HeuristicEvaluator, static_value
from catanbot.search import SearchConfig, Searcher
from catanbot.selfplay import make_bot, play_game
from catanbot.state import PHASE_MAIN, PHASE_SETUP_SETTLEMENT, new_game
from tests.test_conversion import V, blank, put, with_building
from tests.test_corrections import PINNED_GAMES, positions, search_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = tuning.DEFAULT_SEARCH_SPEC
W_, BR, SH, WH, OR = B.WOOD, B.BRICK, B.SHEEP, B.WHEAT, B.ORE
F2_NAMES = ("placement.PORT_SPOT_GENERIC", "placement.PORT_SPOT_2TO1_BASE", "placement.PORT_SPOT_2TO1_SLOPE",
            "heuristic.PORT_STATIC_GENERIC", "heuristic.PORT_STATIC_2TO1_BASE", "heuristic.PORT_STATIC_2TO1_SLOPE",
            "placement.PORT_GENERIC_ONCE")
F1_NAMES = ("placement.PORT_MODEL", "placement.PORT_A0", "placement.PORT_B", "placement.PORT_ELAST",
            "placement.PORT_SPOT_W", "placement.PORT_STATIC_W")
P3 = {"heuristic.PORT_STATIC_GENERIC": 0.8, "placement.PORT_GENERIC_ONCE": 1}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def midgame_states(n_games=4, every=17, seed=300):
    out = []
    for g in range(n_games):
        rng = random.Random(seed + g)
        s = new_game(4, rng=rng)
        bots = [HeuristicBot(temperature=0.3) for _ in range(4)]
        k = 0
        while not E.is_terminal(s) and k < 600:
            a = bots[E.acting_player(s)].decide(s, E.legal_actions(s), rng)
            if s.turn > 12 and k % every == 0:
                out.append(s.copy())
            s = E.apply(s, a, rng)
            k += 1
    return out


STATES = midgame_states()


def old_spot_bonus(state, player, v, own):
    """The pre-feature port block of score_settlement_spot (literals)."""
    t = state.ports.get(v)
    if t is None:
        return 0.0
    if t == B.PORT_GENERIC:
        return 1.0
    return 0.5 + 6.0 * (own[t] + P.vertex_production(state, v, ignore_robber=True)[t])


def old_static_port(state, player):
    """The pre-feature port loop of static_value (literals), summed from 0."""
    p = state.players[player]
    prod = P.player_production(state, player, ignore_robber=True)
    out = 0.0
    for v in p.settlements + p.cities:
        t = state.ports.get(v)
        if t is None:
            continue
        out += 0.2 if t == B.PORT_GENERIC else 0.15 + 4.0 * prod[t]
    return out


def best_reach(state, player):
    occ = state.occupied_vertices()
    own = P.player_production(state, player, ignore_robber=True)
    sc = P.resource_scarcity(state)
    if len(state.players[player].settlements) >= B.MAX_SETTLEMENTS:
        return 0.0
    best = 0.0
    for v, (d, _e) in P.reachable_spots(state, player, max_roads=2, occ=occ).items():
        best = max(best, P.score_settlement_spot(state, player, v, occ=occ, own_prod=own, scarcity=sc) / (1 + 0.9 * d))
    return best


def game_digest(bots, seed, max_turns=36):
    h = hashlib.sha256()
    res = play_game(bots, rng=random.Random(seed), seed=seed, max_turns=max_turns,
                    on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    return h.hexdigest()[:16]


def port_board(port, at=3):
    """ore_wheat_board of test_conversion (seat 0 on V0, V1: wheat / ore / sheep) with a port at V[at]."""
    s = blank()
    put(s, V[0], [(WH, 6), (OR, 8), (WH, 5)])
    put(s, V[1], [(OR, 6), (WH, 9), (SH, 4)])
    put(s, V[2], [(W_, 6), (BR, 8), (W_, 4)])
    put(s, V[3], [(W_, 9), (BR, 5), (SH, 10)])
    s.players[0].settlements = [V[0], V[1]]
    s.ports = {} if port is None else {V[at]: port}
    return s


# ---------------------------------------------------------------------------
# defaults: byte-identical, C++ parity, registry
# ---------------------------------------------------------------------------
def test_defaults_equal_the_pre_feature_literals_and_cpp():
    checked = 0
    for s in STATES:
        occ = s.occupied_vertices()
        for p in range(4):
            own = P.player_production(s, p, ignore_robber=True)
            for v in s.ports:
                if P.is_free_vertex(occ, v):
                    assert P.spot_port_bonus(s, p, v, own) == old_spot_bonus(s, p, v, own)
                    assert openings.spot_port_bonus(s, p, v, own) == old_spot_bonus(s, p, v, own)
                    checked += 1
            pl = s.players[p]
            assert heuristic.static_port_term(s, pl.settlements + pl.cities, own) == old_static_port(s, p)
            if accel.AVAILABLE:
                assert abs(static_value(s, p) - accel.static_value(s, p)) <= 1e-9
                for v in list(s.ports)[:6]:
                    if P.is_free_vertex(occ, v):
                        py = P.score_settlement_spot(s, p, v)
                        assert abs(py - accel._core.score_settlement_spot(s, p, v)) <= 1e-9
    assert checked > 100


def test_registry_entries_defaults_flags_and_round_trip():
    reg = tuning.TUNABLES
    for name in F2_NAMES + F1_NAMES:
        t = reg[name]
        assert t.needs_python_evaluator and t.kind == "weight" and t.candidates and t.description
        assert t.default == t.targets[0].get()
    assert reg["placement.PORT_GENERIC_ONCE"].parse("1") == 1 and isinstance(reg["placement.PORT_MODEL"].parse("3"), int)
    assert (reg["placement.PORT_MODEL"].default, reg["placement.PORT_GENERIC_ONCE"].default) == (0, 0)
    assert [reg[n].default for n in F2_NAMES[:6]] == [1.0, 0.5, 6.0, 0.2, 0.15, 4.0]    # today's literals
    fk = reg["ports.FLOW_KAPPA"]
    assert (fk.default, fk.needs_python_evaluator, fk.requires_search, fk.qualname) == (0.0, False, True,
                                                                                         "portvalue.FLOW_KAPPA")
    assert tuning.find("FLOW_KAPPA") is fk
    ad = reg["portvalue.ADVICE"]
    assert (ad.default, ad.needs_python_evaluator, ad.candidates) == (0, False, [1])
    assert tuning.verify_registry() == []
    names = F2_NAMES + F1_NAMES + ("ports.FLOW_KAPPA", "portvalue.ADVICE")
    before = {n: reg[n].targets[0].get() for n in names}
    tok = tuning.apply({n: reg[n].candidates[0] for n in names})
    try:
        assert P.PORT_MODEL == 1 and P.PORT_GENERIC_ONCE == 1 and heuristic.PORT_STATIC_GENERIC == 0.8
        assert PV.FLOW_KAPPA == reg["ports.FLOW_KAPPA"].candidates[0] and PV.ADVICE == 1
    finally:
        tuning.restore(tok)
    assert {n: reg[n].targets[0].get() for n in names} == before


def test_default_games_with_every_switch_spelled_out_and_advice_on():
    """ParamBot with the new switches at their defaults, and with portvalue.ADVICE = 1 (display only): the pinned
    pre-feature digests (tests/test_corrections.py)."""
    spelled = {n: tuning.TUNABLES[n].default for n in F2_NAMES + F1_NAMES + ("ports.FLOW_KAPPA",)}
    want = PINNED_GAMES["default"][1][1]
    assert game_digest([ParamBot(make_bot(DEFAULT), spelled) for _ in range(4)], 12) == want
    assert game_digest([ParamBot(make_bot(DEFAULT), {"portvalue.ADVICE": 1}) for _ in range(4)], 12) == want


_FRESH = r"""
import hashlib, json, random, sys
sys.path.insert(0, ROOT)
from catanbot.selfplay import make_bot, play_game
h = hashlib.sha256()
spec = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
res = play_game([make_bot(spec) for _ in range(4)], rng=random.Random(11), seed=11, max_turns=36,
                on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
print(json.dumps({"hash": h.hexdigest()[:16], "imported": [m for m in ("catanbot.portvalue", "catanbot.corrections")
                                                           if m in sys.modules]}))
"""


def test_default_game_never_imports_portvalue():
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, "-c", f"ROOT = {ROOT!r}\n" + _FRESH], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["imported"] == [] and out["hash"] == PINNED_GAMES["default"][1][0]


# ---------------------------------------------------------------------------
# ports.constants (F2)
# ---------------------------------------------------------------------------
def test_generic_once_counts_one_3to1_and_zeroes_a_second_3to1_spot():
    s = port_board(B.PORT_GENERIC, at=0)
    s.ports[V[1]] = B.PORT_GENERIC                                   # two of our buildings on 3:1 ports
    prod = P.player_production(s, 0, ignore_robber=True)
    blds = s.players[0].settlements
    assert heuristic.static_port_term(s, blds, prod) == pytest.approx(0.4)
    s.ports[V[3]] = B.PORT_GENERIC                                   # a free 3:1 spot
    assert P.spot_port_bonus(s, 0, V[3]) == 1.0
    with tuning.overridden({"placement.PORT_GENERIC_ONCE": 1}):
        assert heuristic.static_port_term(s, blds, prod) == pytest.approx(0.2)
        assert P.spot_port_bonus(s, 0, V[3]) == 0.0                  # we already own a 3:1
        assert P.spot_port_bonus(s, 1, V[3]) == 1.0                  # seat 1 does not
        s2 = port_board(W_, at=0)                                    # a 2:1 port does not make a 3:1 worthless
        s2.ports[V[3]] = B.PORT_GENERIC
        assert P.spot_port_bonus(s2, 0, V[3]) == 1.0


def test_p3_raises_static_by_exactly_06_through_the_module_attribute():
    """P3' (static 3:1 = 0.8, counted once) on a board whose only port is under our building: static_value rises by
    exactly 0.6; the C++ value does not move (why the Python evaluator is needed)."""
    s = port_board(B.PORT_GENERIC, at=0)
    before = static_value(s, 0)
    with tuning.overridden(P3):
        assert static_value(s, 0) - before == pytest.approx(0.6, abs=1e-12)
        if accel.AVAILABLE:
            assert accel.static_value(s, 0) == pytest.approx(before, abs=1e-9)
    with tuning.overridden({"placement.PORT_GENERIC_ONCE": 1}):   # read through placement, not a from-import copy
        assert heuristic._pl is P and static_value(s, 0) == pytest.approx(before, abs=1e-12)


# ---------------------------------------------------------------------------
# ports.surplus_value (F1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("elast", [0.25, 0.0, 0.5])
def test_port_gain_identities(elast):
    prod = [x / 36 for x in (2, 3, 1, 9, 7)]
    conv = P.port_conversion(prod)
    with tuning.overridden({"placement.PORT_ELAST": elast}):
        g3 = P.port_gain(prod, [4] * 5, [3] * 5)
        assert g3 == pytest.approx(sum(conv) / 12 * (1 + elast / 3), rel=1e-12)
        g2 = P.port_gain(prod, [4] * 5, P.ratios_with([4] * 5, WH))
        assert g2 == pytest.approx(conv[WH] / 4 * (1 + elast), rel=1e-12)
        assert P.port_gain(prod, [3] * 5, P.ratios_with([3] * 5, WH)) == pytest.approx(conv[WH] / 6 * (1 + elast),
                                                                                        rel=1e-12)
        assert P.port_gain(prod, [4, 4, 4, 2, 4], [4, 4, 4, 2, 4]) == 0.0      # already 2
        assert P.port_gain(prod, [3] * 5, P.ratios_with([3] * 5, B.PORT_GENERIC)) == 0.0   # a second 3:1
    assert P.ratios_with([4, 4, 4, 2, 4], B.PORT_GENERIC) == [3, 3, 3, 2, 3]


def test_port_conversion_intercept_monotone_and_need_weights():
    prod = [x / 36 for x in (5, 5, 4, 6, 6)]                           # about the demand mix: small surplus
    D = P.demand_shares()
    inc = sum(prod)
    conv = P.port_conversion(prod)
    for r in range(5):
        assert conv[r] >= P.PORT_A0 * prod[r] / inc > 0.0             # the intercept converts even without surplus
    more = list(prod)
    more[WH] += 3 / 36
    c2 = P.port_conversion(more)
    s1, s2 = max(0, prod[WH] - D[WH] * inc), max(0, more[WH] - D[WH] * sum(more))
    assert s2 > s1 and c2[WH] > conv[WH]
    assert sum(D) == pytest.approx(1.0) and D[WH] == pytest.approx(P.RESOURCE_DEMAND[WH] / 5)
    exact = [D[r] for r in range(5)]                                   # no deficit at all
    assert P.port_need_weights(exact, P.resource_scarcity(port_board(None))) == (1.0, 0.5)
    w, c = P.port_need_weights([0.3, 0.0, 0.0, 0.1, 0.1], [1.0] * 5)
    assert 0.0 < c <= 1.0 and w > 0.0


def test_spot_bonus_counts_the_spots_own_production_and_the_complement():
    s = port_board(WH, at=3)
    s.players[0].settlements = [V[0]]                                  # wheat-rich seat
    with tuning.overridden({"placement.PORT_MODEL": 1}):
        rich = P.spot_port_bonus(s, 0, V[3])
        s2 = s.copy()
        s2.hexes = list(s.hexes)
        put(s2, V[3], [(WH, 9), (WH, 5), (SH, 10)])                    # the port spot itself produces wheat
        richer = P.spot_port_bonus(s2, 0, V[3])
        assert richer > rich > 0.0
        # the complement factor in [0.6, 1.4]: bonus / PE
        own = P.player_production(s2, 0, ignore_robber=True)
        pv = P.vertex_production(s2, V[3], ignore_robber=True)
        prod = [own[r] + pv[r] for r in range(5)]
        rho = P.port_ratios(s2, 0)
        pe, c = P.port_pe(prod, rho, P.ratios_with(rho, WH), P.resource_scarcity(s2))
        assert richer == pytest.approx(P.PORT_SPOT_W * pe * (0.6 + 0.8 * c)) and 0.6 <= 0.6 + 0.8 * c <= 1.4


def test_model_override_reaches_static_setup_and_spot_priors():
    s = port_board(B.PORT_GENERIC, at=0)
    before = static_value(s, 0)
    with tuning.overridden({"placement.PORT_MODEL": 1}):
        assert static_value(s, 0) != before
    st = new_game(4, rng=random.Random(4))
    assert st.phase == PHASE_SETUP_SETTLEMENT
    me = st.current
    base = dict(P.setup_pick(st, me, k=60))
    with tuning.overridden({"placement.PORT_MODEL": 1}):
        m1 = dict(P.setup_pick(st, me, k=60))
    moved = [v for v in base if v in m1 and base[v] != m1[v]]
    assert moved and all(v in st.ports for v in moved)                   # only port spots move


@pytest.mark.parametrize("model", [1, 2, 3])
def test_static_difference_is_the_port_term_plus_the_reach_term(model):
    checked = 0
    for s in STATES[::3]:
        for p in range(4):
            pl = s.players[p]
            own = P.player_production(s, p, ignore_robber=True)
            t0, r0, v0 = heuristic.static_port_term(s, pl.settlements + pl.cities, own), best_reach(s, p), \
                static_value(s, p)
            with tuning.overridden({"placement.PORT_MODEL": model}):
                t1, r1, v1 = heuristic.static_port_term(s, pl.settlements + pl.cities, own), best_reach(s, p), \
                    static_value(s, p)
            assert v1 - v0 == pytest.approx((t1 - t0) + 0.12 * (r1 - r0), abs=1e-9)
            owns = any(v in s.ports for v in pl.settlements + pl.cities)
            near = any(v in s.ports for v in P.reachable_spots(s, p, max_roads=2))
            if not owns and not near:
                assert v1 == v0
            checked += owns
    assert checked > 5


def test_mode2_keeps_the_spot_score_and_mode3_keeps_every_2to1_term():
    for s in STATES[::4]:
        occ = s.occupied_vertices()
        for p in range(4):
            own = P.player_production(s, p, ignore_robber=True)
            free = [v for v in s.ports if P.is_free_vertex(occ, v)]
            base = {v: P.spot_port_bonus(s, p, v, own) for v in free}
            with tuning.overridden({"placement.PORT_MODEL": 2}):
                assert {v: P.spot_port_bonus(s, p, v, own) for v in free} == base
            with tuning.overridden({"placement.PORT_MODEL": 3}):
                for v in free:
                    if s.ports[v] != B.PORT_GENERIC:
                        assert P.spot_port_bonus(s, p, v, own) == base[v]
    s = port_board(WH, at=0)                                            # a 2:1 wheat building only
    prod = P.player_production(s, 0, ignore_robber=True)
    blds = s.players[0].settlements
    t0 = heuristic.static_port_term(s, blds, prod)
    with tuning.overridden({"placement.PORT_MODEL": 3}):
        assert heuristic.static_port_term(s, blds, prod) == t0
        s.ports[V[1]] = B.PORT_GENERIC                                  # plus a 3:1: calibrated on top of the 2:1
        t3 = heuristic.static_port_term(s, blds, prod)
        rho = P.ratios_of(s, blds)
        extra = P.PORT_STATIC_W * P.port_pe(prod, [4, 4, 4, 2, 4], rho, P.resource_scarcity(s))[0]
        assert t3 == pytest.approx(t0 + extra, abs=1e-12)


def test_joint_ratios_make_a_second_generic_port_worth_nothing_in_static():
    s = port_board(B.PORT_GENERIC, at=0)
    prod = P.player_production(s, 0, ignore_robber=True)
    blds = s.players[0].settlements
    with tuning.overridden({"placement.PORT_MODEL": 1}):
        one = heuristic.static_port_term(s, blds, prod)
        s.ports[V[1]] = B.PORT_GENERIC
        two = heuristic.static_port_term(s, blds, prod)
    assert one == two > 0.0


# ---------------------------------------------------------------------------
# ports.flow_provider
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("port", [B.PORT_GENERIC, WH, W_])
def test_owned_flow_cards_rise_by_port_flow_value(port):
    s0 = port_board(None)
    s1 = port_board(port, at=1)                                         # the port under an existing building
    got = PV.owned_flow_cards(s1, 0) - PV.owned_flow_cards(s0, 0)
    assert got == pytest.approx(AQ.port_flow_value(s0, 0, port), rel=1e-12)
    s2 = port_board(B.PORT_GENERIC, at=1)
    s2.ports[V[0]] = B.PORT_GENERIC                                     # a second 3:1 adds nothing
    assert PV.owned_flow_cards(s2, 0) == pytest.approx(PV.owned_flow_cards(port_board(B.PORT_GENERIC, at=1), 0))
    late = port_board(port, at=1)
    late.turn = 2 * 4 + 130
    assert PV.owned_flow_cards(late, 0) == 0.0


@pytest.mark.parametrize("port", [B.PORT_GENERIC, WH])
def test_flow_context_replaces_the_static_credit_with_the_flow_value(port):
    s = port_board(port, at=3)
    with tuning.overridden({"ports.FLOW_KAPPA": 0.18}):
        ctx = PV.PortFlowContext(s, 0)
    leaf = with_building(s, "settlement", V[3])
    c = ctx.corrections(leaf)
    f0, f1 = PV.owned_flow_cards(s, 0, ctx.t), PV.owned_flow_cards(leaf, 0, ctx.t)
    pr1 = P.player_production(leaf, 0, ignore_robber=True)
    credit = heuristic.static_port_term(leaf, leaf.players[0].settlements, pr1)
    assert f1 > f0 == 0.0 and credit > 0.0
    assert c[0] == pytest.approx(0.18 * (f1 - f0) - credit, abs=1e-12) and c[1:] == [0.0] * 3
    assert ctx.corrections(s.copy()) is None                            # nothing built: the hub's fast path
    # leaf value z = static + C: static's port credit is gone, the flow value is in (the one owner)
    if accel.AVAILABLE:
        noport = s.copy()
        noport.ports = {}
        lb = with_building(noport, "settlement", V[3])
        za = H.static_values(leaf)[0] + c[0]
        zb = H.static_values(lb)[0]
        assert za - zb == pytest.approx(0.18 * f1, abs=1e-6)              # no port spot left free: reach is equal


def test_hub_owner_the_flow_provider_owns_our_port_value_also_next_to_conv():
    from catanbot.conversion import ConversionContext
    s, me = positions(k=1)[0]
    with tuning.overridden({"ports.FLOW_KAPPA": 0.18}):
        sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8))
        sr.search(s, me, random.Random(1))
        assert [type(p) for p in sr._corr.providers] == [PV.PortFlowContext] and sr._corr.port_owner == "flow"
        sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, conv=1))
        sr.search(s, me, random.Random(1))
        assert [type(p) for p in sr._corr.providers] == [ConversionContext, PV.PortFlowContext]
        cv = sr._corr.providers[0]
        assert sr._corr.port_owner == "flow" and cv.ports is False and cv.ledger is False
        sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, paths=1))
        sr.search(s, me, random.Random(1))
        assert [type(p) for p in sr._corr.providers] == [W.PathsContext, PV.PortFlowContext]
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, conv=1))
    sr.search(s, me, random.Random(1))
    cv = sr._corr.providers[0]
    assert sr._corr.port_owner == "conv" and cv.ports is True and cv.ledger is True     # conv alone: unchanged
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8))
    sr.search(s, me, random.Random(1))
    assert sr._corr is None                                             # FLOW_KAPPA = 0: no hub


@pytest.mark.parametrize("port", [B.PORT_GENERIC, WH])
@pytest.mark.parametrize("reach_w", [0.0, 0.5])
def test_conv_plus_flow_counts_our_port_value_once(port, reach_w):
    """conv=1 and the flow provider together: the port changes our leaf correction by exactly the flow value minus
    static's port credit (so static + C holds the port once, as kappa x F), static's credit is cancelled once, and
    the conversion term is the port-free one (its value on the same board without the port)."""
    from catanbot.conversion import ConversionContext
    kappa = 0.25
    cfg = SearchConfig(depth=1, beam=4, expand=8, conv=1)
    s = port_board(port, at=3)
    noport = port_board(None)
    leaf, leaf0 = with_building(s, "settlement", V[3]), with_building(noport, "settlement", V[3])
    with tuning.overridden({"ports.FLOW_KAPPA": kappa, "conversion.REACH_W": reach_w}):
        hub = H.CorrectionHub.for_search(HeuristicEvaluator(), s, 0, cfg)
        hub0 = H.CorrectionHub.for_search(HeuristicEvaluator(), noport, 0, cfg)
        conv_alone = ConversionContext(s, 0)                            # conv's own port-aware term (the ledger)
    assert hub.port_owner == "flow" and [type(p) for p in hub.providers] == [ConversionContext, PV.PortFlowContext]
    c, c0 = hub.corrections(leaf)[0], hub0.corrections(leaf0)[0]
    cv, fl = hub.providers
    assert cv.corrections(leaf)[0] == pytest.approx(hub0.providers[0].corrections(leaf0)[0], abs=1e-12)
    f1 = PV.owned_flow_cards(leaf, 0, fl.t)
    credit = heuristic.static_port_term(leaf, leaf.players[0].settlements, P.player_production(leaf, 0, True))
    assert f1 > 0.0 and credit > 0.0
    assert c - c0 == pytest.approx(kappa * f1 - credit, abs=1e-12)     # the port: flow value in, static credit out
    assert c == pytest.approx(cv.corrections(leaf)[0] + fl.corrections(leaf)[0], abs=1e-12)
    # conv alone would also move with the port (its routing and ledger): with flow on, that part is gone
    assert conv_alone.corrections(leaf)[0] != pytest.approx(cv.corrections(leaf)[0], abs=1e-6)
    if accel.AVAILABLE:                                                  # static + C: the port counted once
        za, zb = H.static_values(leaf)[0] + c, H.static_values(leaf0)[0] + c0
        assert za - zb == pytest.approx(kappa * f1, abs=1e-6)


def test_conv_alone_keeps_its_pre_ports_search_digest():
    """conv=1 without the flow switch (also spelled out at 0) searches exactly as before the ports step: the digests
    were computed on the epoch-B2 snapshot's conversion.py (before ``ConversionContext(ports=...)``)."""
    pos = positions(k=8)
    cfg = SearchConfig(depth=1, beam=4, expand=8, conv=1)
    assert search_hash(cfg, pos) == "32280282458ad41b"
    with tuning.overridden({"ports.FLOW_KAPPA": 0.0, "conversion.REACH_W": 0.5}):
        assert search_hash(cfg, pos) == "182308e77800c55c"


def test_flow_candidate_is_built_only_for_the_candidate_seats(monkeypatch):
    seen = []
    orig = PV.PortFlowContext.__init__

    def spy(self, root, me, params=None):
        seen.append((me, PV.FLOW_KAPPA))
        orig(self, root, me, params)

    monkeypatch.setattr(PV.PortFlowContext, "__init__", spy)
    cand = {"ports.FLOW_KAPPA": 0.18}
    bots = [ParamBot(make_bot(DEFAULT), cand), ParamBot(make_bot(DEFAULT), {}),
            ParamBot(make_bot(DEFAULT), cand), ParamBot(make_bot(DEFAULT), {})]
    play_game(bots, rng=random.Random(21), seed=21, max_turns=40)
    assert seen and {me for me, _ in seen} <= {0, 2} and all(k == 0.18 for _, k in seen)
    assert PV.FLOW_KAPPA == 0.0


# ---------------------------------------------------------------------------
# ports.spot_want helper (F4) and ports.advice (F3)
# ---------------------------------------------------------------------------
def test_rival_want_clamps_relative_to_the_best_discounted_spot(monkeypatch):
    s = port_board(None)
    reach = {10: (0, -1), 11: (1, -1), 12: (2, -1)}
    scores = {10: 10.0, 11: 9.5, 12: 5.0}
    monkeypatch.setattr(W, "_spot_score", lambda st, j, x: scores[x])
    assert PV.rival_want(s, 1, 10, reach=reach) == 1.0
    assert PV.rival_want(s, 1, 11, reach=reach) == pytest.approx(9.5 / 1.9 / 10.0)      # 0.5
    assert PV.rival_want(s, 1, 12, reach=reach) == PV.WANT_FLOOR                         # 5 / 2.8 / 10 < 0.3
    assert PV.rival_want(s, 1, 13, reach=reach) is None
    assert PV.rival_want(s, 1, 12, reach=reach, floor=0.1) == pytest.approx(5.0 / 2.8 / 10.0)


def test_leader_check_modes(monkeypatch):
    s = port_board(None)
    vps = {0: 6.0, 1: 4.0, 2: 3.0, 3: 4.5}
    monkeypatch.setattr(PV, "_vp", lambda st, i: vps[i])
    lead = PV.leader(s)
    assert lead == (0, 1.5)
    assert PV.will_block(s, 0, 1, 0.3, mode=1, lead=lead) == (PV.LEADER_DENY, "you lead")
    assert PV.will_block(s, 0, 1, 0.8, mode=1, lead=lead) == (0.8, "")
    assert PV.will_block(s, 0, 1, 0.3, mode=0, lead=lead) == (0.3, "")
    vps.update({0: 3.0, 1: 7.0})
    lead = PV.leader(s)
    assert lead == (1, 2.5)
    assert PV.will_block(s, 0, 1, 0.4, mode=2, lead=lead) == (1.0, "they lead")
    assert PV.will_block(s, 0, 1, 0.4, mode=1, lead=lead) == (0.4, "")
    vps.update({1: 3.5, 3: 4.0})
    assert PV.leader(s) is None


def test_port_advice_numbers_and_the_verdict_flip():
    s = port_board(WH, at=3)
    s.players[0].settlements = [V[0]]
    spots = {V[3]: 0, V[2]: 0}
    a = PV.port_advice(s, 0, V[3], spots)
    own = P.player_production(s, 0, ignore_robber=True)
    pv = P.vertex_production(s, V[3], ignore_robber=True)
    prod = [own[r] + pv[r] for r in range(5)]
    rho = P.port_ratios(s, 0)
    G = P.port_gain(prod, rho, P.ratios_with(rho, WH))
    w, _c = P.port_need_weights(prod, P.resource_scarcity(s))
    H_ = W.horizon(s)
    assert a["saved"] == pytest.approx(4 * H_ * G * w, rel=1e-12) and a["land"] == V[2]
    sc = P.resource_scarcity(s)
    assert a["extra"] == pytest.approx(4 * H_ * (PV.pe_pips(s, V[2], sc) - PV.pe_pips(s, V[3], sc)) / 36)
    assert a["worth"] == (a["saved"] > a["extra"] + a["road_cost"])
    # a weak land spot: the port wins; a strong one: the land wins
    weak = s.copy()
    weak.hexes = list(s.hexes)
    put(weak, V[2], [(W_, 2), (BR, 12), (W_, 3)])
    strong = s.copy()
    strong.hexes = list(s.hexes)
    put(strong, V[2], [(W_, 6), (BR, 8), (SH, 5)])
    assert PV.port_advice(weak, 0, V[3], spots)["worth"] is True
    assert PV.port_advice(strong, 0, V[3], spots)["worth"] is False
    # roads: the port two roads further costs two roads of cards
    far = PV.port_advice(s, 0, V[3], {V[3]: 2, V[2]: 0})
    assert far["road_cost"] == pytest.approx(2 * PV.road_cards(sc))
    assert PV.road_cards(sc) == pytest.approx(sum(P.RESOURCE_DEMAND[r] * sc[r] ** 0.5 for r in (W_, BR)))


def _contested_port_position():
    for seed in range(40, 90):
        rng = random.Random(seed)
        s = new_game(4, rng=rng)
        bots = [HeuristicBot(temperature=0.3) for _ in range(4)]
        k = 0
        while not E.is_terminal(s) and k < 900:
            i = E.acting_player(s)
            if s.phase == PHASE_MAIN and s.turn > 24:
                ctx = W.PathsContext(s, i)
                for v, d, pm, info in ctx._spot_table(s, ctx._inputs(s)):
                    if v in s.ports and info:
                        return s, i, v
            s = E.apply(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
            k += 1
    raise AssertionError("no contested port spot found")


def test_race_line_matches_the_spot_table_and_the_want_helper():
    s, me, v = _contested_port_position()
    ctx = W.PathsContext(s, me)
    row = next(r for r in ctx._spot_table(s, ctx._inputs(s)) if r[0] == v)
    r = PV.race_advice(s, me, v)
    assert r["p_me"] == row[2] and [(x["seat"], x["roads"], x["rounds"]) for x in r["rivals"]] == list(row[3])
    for x in r["rivals"]:
        assert x["want"] == PV.rival_want(s, x["seat"], v)
        assert (x["will"], x["why"]) == PV.will_block(s, me, x["seat"], x["want"], lead=PV.leader(s))
    lines = PV.advice_lines(s, me, [("build_settlement", v)])
    assert f"port at {v} " in lines[0] and lines[1].startswith(f"  race for {v}: we get there first ~")


def test_search_never_calls_the_advice(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the search called the port advice")

    for name in ("advice_lines", "port_advice", "race_advice", "rival_want", "leader"):
        monkeypatch.setattr(PV, name, boom)
    with tuning.overridden({"portvalue.ADVICE": 1}):
        play_game([make_bot(DEFAULT) for _ in range(4)], rng=random.Random(5), seed=5, max_turns=60)


def test_cli_port_advice_section(tmp_path, capsys):
    from catanbot import cli
    s = new_game(4, rng=random.Random(8))                               # setup: every free port spot is in play
    path = tmp_path / "state.json"
    path.write_text(json.dumps(s.to_dict()))
    me = s.players[s.current].color
    args = ["recommend", "--state", str(path), "--me", me, "--depth", "1", "--model", "heuristic", "--json"]
    assert cli.main(args) == 0
    assert "ports" not in json.loads(capsys.readouterr().out)["advice"]
    assert cli.main(args + ["--port-advice"]) == 0
    lines = json.loads(capsys.readouterr().out)["advice"]["ports"]
    assert lines and all("port at" in ln and ("port worth it" in ln or "take the land spot" in ln) for ln in lines)
    assert cli.main(args[:-1] + ["--port-advice"]) == 0
    assert "Ports (is this port worth it?)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scripts/port_gate.py
# ---------------------------------------------------------------------------
def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gate_jsonl(path, row, deltas, n=300):
    """A synthetic ablate_catanatron JSONL: one run, n seeds, candidate cards / settlements shifted by deltas(s)."""
    recs = [{"v": 1, "kind": "run", "run_key": "rk", "cand_key": "C", "def_key": "D", "exp": row, "t": 1.0,
             "value_text": "1"}]
    for s in range(n):
        base_cards, base_settle = s % 7, s % 3
        dc, dset = deltas(s)
        for arm, cards, settle, won in (("D", base_cards, base_settle, s % 4 == 0),
                                         ("C", base_cards + dc, base_settle + dset, s % 4 == 0)):
            recs.append({"v": 1, "kind": "game", "arm_key": arm, "s": s, "code": "x", "status": "ok", "won": won,
                         "mech": {"bank_trades": 5, "bank_cards_given": 20 - cards, "settlements_built": settle}})
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_port_gate_pass_rule_best_cell_and_mode3_trigger(tmp_path):
    G = _load_script("port_gate")
    _gate_jsonl(tmp_path / "ports_gate_f1@vf.jsonl", "ports_gate_f1@vf", lambda s: (2 + (s % 2), -(s % 2)))
    _gate_jsonl(tmp_path / "ports_gate_f2@vf.jsonl", "ports_gate_f2@vf", lambda s: (1 + (s % 2), 0))
    _gate_jsonl(tmp_path / "ports_gate_flow@vf.jsonl", "ports_gate_flow@vf", lambda s: (s % 2, 0))
    res = G.evaluate(str(tmp_path))
    f1, f2, fl = res["cells"]["f1"], res["cells"]["f2"], res["cells"]["flow"]
    assert f1["cards_saved"] == pytest.approx(2.5) and f1["settlements"] == pytest.approx(-0.5) and not f1["pass"]
    assert f2["cards_saved"] == pytest.approx(1.5) and f2["pass"] and not fl["pass"]      # 0.5 < 1.1
    assert res["cells"]["f1m3"].get("missing") and not res["cells"]["f1m3"]["pass"]
    assert res["any_pass"] and res["best"] == "f2" and res["best_is"] == {"f1": False, "f2": True, "flow": False,
                                                                          "f1m3": False}
    assert res["f1_overshoot"] is True                                  # raised cards, failed the settlement guard
    out = tmp_path / "gate.json"
    assert G.main(["--dir", str(tmp_path), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["best"] == "f2"
    # an incomplete cell never passes
    _gate_jsonl(tmp_path / "ports_gate_f2@vf.jsonl", "ports_gate_f2@vf", lambda s: (3, 0), n=200)
    assert not G.evaluate(str(tmp_path))["cells"]["f2"]["pass"]
    # the boundaries: cards +1.1 exactly passes; settlements -0.15 exactly ("lower by 0.15 or more") fails
    _gate_jsonl(tmp_path / "ports_gate_f2@vf.jsonl", "ports_gate_f2@vf", lambda s: (1.1, 0))
    _gate_jsonl(tmp_path / "ports_gate_flow@vf.jsonl", "ports_gate_flow@vf",
                lambda s: (2, -1 if s % 20 < 3 else 0))
    res = G.evaluate(str(tmp_path))
    assert res["cells"]["f2"]["pass"] and res["cells"]["flow"]["settlements"] == -0.15
    assert not res["cells"]["flow"]["pass"] and res["cells"]["flow"]["settle_guard_failed"]
