"""Tests for catanbot/conversion.py (the port-aware conversion-cost term, ``search.conv``) and the ``conversion``
opening policy.

Hand-built boards: every hex is a desert except the ones a test gives a resource, so production, shortfalls and
port ratios are known exactly.  The feature is off by default; test_corrections.py pins the default bot's games
(also with ``conv=0`` spelled out) to the digests of the code before this feature existed.
"""
import json
import os
import random
import subprocess
import sys

import pytest

from catanbot import board as B
from catanbot import conversion as C
from catanbot import engine as E
from catanbot import openings, placement, tuning
from catanbot.agents.param_bot import ParamBot
from catanbot.corrections import static_values
from catanbot.heuristic import HeuristicEvaluator, static_value
from catanbot.search import SearchConfig
from catanbot.selfplay import make_bot, play_game
from catanbot.state import PHASE_MAIN, PHASE_SETUP_SETTLEMENT, new_game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = tuning.DEFAULT_SEARCH_SPEC
W_, BR, SH, WH, OR = B.WOOD, B.BRICK, B.SHEEP, B.WHEAT, B.ORE
SHARES = C.need_shares()


# ---------------------------------------------------------------------------
# hand-built boards
# ---------------------------------------------------------------------------
def spots(k):
    """``k`` interior vertices with pairwise disjoint hexes and no common neighbour (so all can hold buildings)."""
    out = []
    for v in range(B.NUM_VERTICES):
        if len(B.VERTEX_HEXES[v]) != 3:
            continue
        if any(set(B.VERTEX_HEXES[v]) & set(B.VERTEX_HEXES[u]) or v in B.VERTEX_NEIGHBORS[u]
               or set(B.VERTEX_NEIGHBORS[v]) & set(B.VERTEX_NEIGHBORS[u]) for u in out):
            continue
        out.append(v)
        if len(out) == k:
            return out
    raise AssertionError("not enough disjoint interior vertices")


V = spots(5)


def blank(seed=1):
    s = new_game(4, rng=random.Random(seed))
    s.hexes = [(B.DESERT, 0)] * B.NUM_HEXES
    s.ports = {}
    for p in s.players:
        p.settlements, p.cities, p.roads = [], [], []
    s.phase, s.current, s.setup_round, s.turn, s.dice = PHASE_MAIN, 0, 2, 30, 6
    s.robber = next(h for h in range(B.NUM_HEXES) if not any(h in B.VERTEX_HEXES[v] for v in V))
    return s


def put(s, v, hexes):
    for h, rn in zip(B.VERTEX_HEXES[v], hexes):
        s.hexes[h] = rn


def ore_wheat_board():
    """Seat 0 on two wheat/ore/sheep vertices (V0, V1); V2 is a free wood/brick/wood vertex, V3 wood/brick/sheep."""
    s = blank()
    put(s, V[0], [(WH, 6), (OR, 8), (WH, 5)])
    put(s, V[1], [(OR, 6), (WH, 9), (SH, 4)])
    put(s, V[2], [(W_, 6), (BR, 8), (W_, 4)])
    put(s, V[3], [(W_, 9), (BR, 5), (SH, 10)])
    s.players[0].settlements = [V[0], V[1]]
    return s


def with_building(s, kind, v, player=0):
    s2 = s.copy()
    p = s2.players[player]
    if kind == "city":
        p.settlements.remove(v)
        p.cities.append(v)
    else:
        p.settlements.append(v)
    return s2


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def test_extra_cards_routes_and_relative_weights():
    short = [0.1, 0.05, 0.05, 0.0, 0.0]
    rich = [0.0, 0.0, 0.0, 0.5, 0.2]          # wheat surplus 0.5 carries 0.25 >= the 0.2 short at 2:1
    S = sum(short)
    bank = C.extra_cards(short, rich, [4] * 5)
    generic = C.extra_cards(short, rich, [3] * 5)
    wheat21 = C.extra_cards(short, rich, [4, 4, 4, 2, 4])
    assert (bank, generic, wheat21) == (pytest.approx(3 * S), pytest.approx(2 * S), pytest.approx(S))
    assert (generic / bank, wheat21 / bank) == (pytest.approx(2 / 3), pytest.approx(1 / 3))   # the 3 / 2 / 1 weights
    # a 2:1 port helps only as far as its resource's surplus pays: 0.1 wheat carries 0.05 cards
    poor = [0.0, 0.0, 0.0, 0.1, 0.2]
    assert C.extra_cards(short, poor, [4, 4, 4, 2, 4]) == pytest.approx(0.05 * 1 + 0.15 * 3)
    assert C.extra_cards(short, poor, [3, 3, 3, 2, 3]) == pytest.approx(0.05 * 1 + 0.15 * 2)
    assert C.extra_cards(short, poor, [4, 4, 4, 2, 2]) == pytest.approx(0.15 * 1 + 0.05 * 3)   # two 2:1 routes
    assert C.extra_cards(short, poor, [4, 4, 2, 4, 4]) == pytest.approx(3 * S)        # 2:1 on a resource we lack
    assert C.extra_cards([0.0] * 5, rich, [4] * 5) == 0.0


def test_shortfall_surplus_balance():
    prod = [x / 36 for x in (2, 0, 3, 9, 7)]
    short, sur = C.shortfall_surplus(prod, SHARES)
    assert sum(short) == pytest.approx(sum(sur)) and min(short + sur) >= 0.0
    assert short[W_] > 0 and short[BR] == pytest.approx(SHARES[BR] * sum(prod)) and sur[WH] > 0
    assert C.shortfall_surplus([0.0] * 5, SHARES) == ([0.0] * 5, [0.0] * 5)
    assert SHARES == pytest.approx([x / sum(placement.RESOURCE_DEMAND) for x in placement.RESOURCE_DEMAND])
    assert C.need_shares([1, 1, 1, 1, 1]) == [0.2] * 5


def test_economy_ports_and_cities():
    s = ore_wheat_board()
    prod, ratios = C.economy(s, [V[0], V[1]], [])
    assert prod == pytest.approx(placement.player_production(s, 0, ignore_robber=True)) and ratios == [4] * 5
    prod2, _ = C.economy(s, [V[1]], [V[0]])
    assert prod2[WH] == pytest.approx(prod[WH] + (5 + 4) / 36) and prod2[OR] == pytest.approx(prod[OR] + 5 / 36)
    s.ports = {V[0]: B.PORT_GENERIC}
    assert C.economy(s, [V[0], V[1]], [])[1] == [3] * 5
    s.ports = {V[0]: B.PORT_GENERIC, V[1]: WH}
    assert C.economy(s, [V[0], V[1]], [])[1] == [3, 3, 3, 2, 3]
    s.ports = {V[1]: WH}
    assert C.economy(s, [V[0], V[1]], [])[1] == [4, 4, 4, 2, 4]
    s.robber = B.VERTEX_HEXES[V[0]][0]                               # production is robber-free
    assert C.economy(s, [V[0], V[1]], [])[0] == pytest.approx(prod)


def test_no_port_3to1_2to1_with_and_without_surplus():
    s = ore_wheat_board()
    c = C.ConversionContext(s, 0)
    base = c.cost(s)
    b = C.breakdown(s, 0)
    assert base == pytest.approx(3 * sum(b["short"])) and b["surplus"][WH] > 0 and b["surplus"][OR] > 0
    s3 = s.copy()
    s3.ports = {V[1]: B.PORT_GENERIC}                                # GameState.copy shares the ports dict
    assert C.ConversionContext(s3, 0).cost(s3) == pytest.approx(2 * sum(b["short"]))
    s2 = s.copy()
    s2.ports = {V[1]: WH}                                            # 2:1 on our surplus wheat
    carried = min(sum(b["short"]), b["surplus"][WH] / 2)
    assert C.ConversionContext(s2, 0).cost(s2) == pytest.approx(carried + 3 * (sum(b["short"]) - carried))
    s0 = s.copy()
    s0.ports = {V[1]: W_}                                            # 2:1 on wood, which we lack: no help
    assert C.ConversionContext(s0, 0).cost(s0) == pytest.approx(base)


def test_horizon_early_vs_late():
    s = ore_wheat_board()
    early = C.ConversionContext(s, 0)
    late_s = s.copy()
    far = [v for v in range(B.NUM_VERTICES) if all(v not in B.VERTEX_NEIGHBORS[u] and v != u for u in V)
           and all(v not in B.VERTEX_NEIGHBORS[w] for w in V)]
    late_s.players[1].cities = far[:3]
    picks = [v for v in far[3:] if all(v not in B.VERTEX_NEIGHBORS[u] and u != v for u in far[:3])]
    late_s.players[1].settlements = picks[:2]                       # 8 VP on desert: horizon shrinks, economy equal
    assert late_s.public_vp(1) == 8
    late = C.ConversionContext(late_s, 0)
    assert early.R == 4 * 16.0 and late.R == pytest.approx(4 * (0.5 + 2.0 * 2))
    city_e = early.corrections(with_building(s, "city", V[0]))[0]
    city_l = late.corrections(with_building(late_s, "city", V[0]))[0]
    assert city_e < 0.0 and city_l == pytest.approx(city_e * late.R / early.R)


def test_city_on_surplus_loses_and_settlement_adding_missing_types_gains():
    s = ore_wheat_board()
    ctx = C.ConversionContext(s, 0)
    city = ctx.corrections(with_building(s, "city", V[0]))
    settle = ctx.corrections(with_building(s, "settlement", V[2]))
    assert city[0] < -0.5 and settle[0] > 0.5 and city[1:] == [0.0] * 3 == settle[1:]
    # with a generic port both effects shrink to two thirds (every card on the generic route)
    s3 = s.copy()
    s3.ports = {V[1]: B.PORT_GENERIC}
    ctx3 = C.ConversionContext(s3, 0)
    assert ctx3.corrections(with_building(s3, "city", V[0]))[0] == pytest.approx(2 / 3 * city[0])
    assert ctx3.corrections(with_building(s3, "settlement", V[2]))[0] == pytest.approx(2 / 3 * settle[0])
    # nothing built since the root: no correction (the hub's fast path)
    assert ctx.corrections(s) is None and ctx.corrections(s.copy()) is None


@pytest.mark.parametrize("port", [B.PORT_GENERIC, WH, W_])
@pytest.mark.parametrize("ledger", [1, 0])
def test_port_settlement_gains_exactly_the_trade_savings(port, ledger):
    """Leaf value z_me = static + correction: a settlement on a port vertex vs the same vertex without the port
    differs by exactly KAPPA R (c_without - c_with) under the ledger (static's port credit cancelled)."""
    s = ore_wheat_board()
    s.ports = {V[3]: port}
    noport = s.copy()
    noport.ports = {}
    with tuning.overridden({"conversion.PORT_LEDGER": ledger}):
        ctx, ctx_b = C.ConversionContext(s, 0), C.ConversionContext(noport, 0)   # memo keys: our buildings only
        a, b = with_building(s, "settlement", V[3]), with_building(noport, "settlement", V[3])
        assert ctx.v0 == ctx_b.v0
        za = static_values(a)[0] + ctx.corrections(a)[0]
        zb = static_values(b)[0] + ctx_b.corrections(b)[0]
    saving = ctx.scale * (ctx_b.cost(b) - ctx.cost(a))
    credit = C.static_port_credit(a, a.players[0].settlements, [])
    assert credit == pytest.approx(0.2 if port == B.PORT_GENERIC else 0.15 + 4.0 * C.economy(a, a.players[0].settlements, [])[0][port])
    assert za - zb == pytest.approx(saving + (0.0 if ledger else credit), abs=1e-9)
    if port == W_:
        assert saving == 0.0                                       # a 2:1 wood port without wood surplus saves nothing
    else:
        assert saving > 0.0


def test_static_port_credit_mirrors_static_value():
    checked = 0
    for seed in range(8):
        rng = random.Random(seed)
        s = new_game(4, rng=rng)
        while s.phase in (PHASE_SETUP_SETTLEMENT, "setup_road"):
            s = E.apply(s, rng.choice(E.legal_actions(s)), rng)
        for i in range(4):
            p = s.players[i]
            if not any(v in s.ports for v in p.settlements):
                continue
            bare = s.copy()
            bare.ports = {k: t for k, t in s.ports.items() if k not in p.settlements}
            d = static_value(s, i) - static_value(bare, i)
            assert C.static_port_credit(s, p.settlements, p.cities) == pytest.approx(d, abs=1e-9)
            checked += 1
    assert checked >= 3


def test_reach_component_credits_a_road_toward_a_missing_type():
    """Seat 0 on a wheat/ore vertex; a wood/brick/sheep spot x three edges away.  With REACH_W > 0 the first road of
    the path brings x within static's two-road reach and gains; a road the other way (only desert spots in reach)
    gains nothing; with REACH_W = 0 roads never change the term."""
    s = blank()
    v0 = V[0]
    put(s, v0, [(WH, 6), (OR, 8), (WH, 5)])
    s.players[0].settlements = [v0]
    near = {v0} | set(B.VERTEX_NEIGHBORS[v0])
    target = None
    for a in B.VERTEX_NEIGHBORS[v0]:
        for b in B.VERTEX_NEIGHBORS[a]:
            for x in B.VERTEX_NEIGHBORS[b]:
                if x in near or b == v0 or set(B.VERTEX_HEXES[x]) & set(B.VERTEX_HEXES[v0]) or len(B.VERTEX_HEXES[x]) < 2:
                    continue
                target = (a, b, x)
                break
            if target:
                break
        if target:
            break
    a, b, x = target
    hexes = [(W_, 6), (BR, 8), (SH, 9)][:len(B.VERTEX_HEXES[x])]
    put(s, x, hexes)
    toward = B.edge_between(v0, a)

    def reach_after(e):
        s2 = s.copy()
        s2.players[0].roads = [e]
        return s2, placement.reachable_spots(s2, 0, max_roads=2)

    assert x in reach_after(toward)[1] and x not in placement.reachable_spots(s, 0, max_roads=2)
    edges = [e for e in B.VERTEX_EDGES[v0]]
    with tuning.overridden({"conversion.REACH_W": 0.3}):
        ctx = C.ConversionContext(s, 0)
        gain = {e: (ctx.corrections(reach_after(e)[0]) or [0.0])[0] for e in edges}
    assert gain[toward] > 0.0 and gain[toward] == max(gain.values())
    def best(st):
        return max([0.0] + [-C.spot_delta(st, 0, u) / (1.0 + 0.9 * d)
                            for u, (d, _e) in placement.reachable_spots(st, 0, max_roads=2).items()])

    assert gain[toward] == pytest.approx(0.3 * ctx.scale * (best(reach_after(toward)[0]) - best(s)))   # root-anchored
    best = best(reach_after(toward)[0])
    assert best >= -C.spot_delta(s, 0, x) / (1.0 + 0.9 * reach_after(toward)[1][x][0]) > 0.0
    ctx0 = C.ConversionContext(s, 0)
    for e in edges:
        assert ctx0.corrections(reach_after(e)[0]) is None             # REACH_W = 0: roads never change the term


def test_context_memo_is_pure_and_constants_frozen():
    s = ore_wheat_board()
    leaves = [with_building(s, "city", V[0]), with_building(s, "settlement", V[2]), s.copy(),
              with_building(s, "settlement", V[3])]
    cold = [C.ConversionContext(s, 0).corrections(x) for x in leaves]
    warm_ctx = C.ConversionContext(s, 0)
    for x in leaves + leaves:
        warm_ctx.corrections(x)
    assert [warm_ctx.corrections(x) for x in leaves] == cold
    ctx = C.ConversionContext(s, 0)
    with tuning.overridden({"conversion.KAPPA_CONV": 1.0}):
        assert ctx.corrections(leaves[0]) == cold[0]               # frozen at construction
        assert C.ConversionContext(s, 0).corrections(leaves[0])[0] == pytest.approx(cold[0][0] / 0.12)


# ---------------------------------------------------------------------------
# default unchanged, spec keys, registry
# ---------------------------------------------------------------------------
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
                  "imported": [m for m in ("catanbot.corrections", "catanbot.conversion") if m in sys.modules]}))
"""


def test_default_game_never_imports_the_hub_or_the_term():
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, "-c", f"ROOT = {ROOT!r}\n" + _FRESH], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["imported"] == [] and out["hash"] == "bb4d5a60408db227"   # the pre-feature digest (test_corrections)


def test_parambot_with_conversion_weights_but_conv_off_plays_the_default():
    from tests.test_corrections import PINNED_GAMES
    import hashlib
    ov = {"conversion.KAPPA_CONV": 0.5, "conversion.REACH_W": 0.3, "conversion.NEED": [1.0] * 5,
          "conversion.PORT_LEDGER": 0}
    h = hashlib.sha256()
    res = play_game([ParamBot(make_bot(DEFAULT), ov) for _ in range(4)], rng=random.Random(12), seed=12,
                    max_turns=36, on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    assert h.hexdigest()[:16] == PINNED_GAMES["default"][1][1]


def test_parambot_builds_contexts_only_for_the_candidate(monkeypatch):
    seen = []
    orig = C.ConversionContext.__init__

    def spy(self, root, me, params=None):
        seen.append((me, C.KAPPA_CONV))
        orig(self, root, me, params)

    monkeypatch.setattr(C.ConversionContext, "__init__", spy)
    cand = {"search.conv": 1, "conversion.KAPPA_CONV": 0.25}
    bots = [ParamBot(make_bot(DEFAULT), cand), ParamBot(make_bot(DEFAULT), {}),
            ParamBot(make_bot(DEFAULT), cand), ParamBot(make_bot(DEFAULT), {})]
    play_game(bots, rng=random.Random(21), seed=21, max_turns=24)
    assert seen and {me for me, _ in seen} <= {0, 2} and all(k == 0.25 for _, k in seen)
    assert all(b.inner.config.conv == 0 for b in bots) and C.KAPPA_CONV == 0.12


def test_registry_spec_and_defaults():
    assert SearchConfig().conv == 0 and make_bot(DEFAULT).config.conv == 0
    assert make_bot(DEFAULT + ",conv=1").config.conv == 1
    from catanbot.bench.catanatron_adapter import DEFAULT_SPEC
    assert make_bot(DEFAULT_SPEC).config.conv == 0
    t = tuning.TUNABLES["search.conv"]
    assert (t.kind, t.default, t.candidates, t.spec_key, t.requires_depth) == ("search", 0, [1], "conv", 1)
    for name in ("KAPPA_CONV", "REACH_W", "NEED", "PORT_LEDGER"):
        t = tuning.TUNABLES[f"conversion.{name}"]
        assert t.kind == "weight" and t.requires_search and not t.needs_python_evaluator
        assert "only with search.conv=1" in t.description and t.default == getattr(C, name)
    assert tuning.verify_registry() == []
    before = C.current_params()
    tok = tuning.apply({"conversion.NEED": [1, 1, 1, 1, 1], "conversion.PORT_LEDGER": 0})
    try:
        assert C.NEED == [1, 1, 1, 1, 1] and C.PORT_LEDGER == 0
    finally:
        tuning.restore(tok)
    assert C.current_params() == before
    assert "economy" in tuning.SEAT_OBSERVERS


def test_search_with_conv_prefers_the_diversifying_settlement():
    """One decision where a settlement on a missing type and a city on surplus ore/wheat are both affordable:
    the corrections favour the settlement by more than they shift the city."""
    s = ore_wheat_board()
    me = s.players[0]
    me.roads = []
    for u in (V[2],):     # a road next to V2 so the settlement is legal
        e = next(e for e in B.VERTEX_EDGES[u])
        me.roads.append(e)
        a, b = B.EDGE_VERTICES[e]
        link = b if a == u else a
        e2 = next(x for x in B.VERTEX_EDGES[link] if x != e)
        me.roads.append(e2)
    me.resources = [1, 1, 1, 3, 3]
    ctx = C.ConversionContext(s, 0)
    ev = HeuristicEvaluator()
    city, settle = with_building(s, "city", V[0]), with_building(s, "settlement", V[2])
    base = ev.evaluate([city, settle], [0, 0])
    from catanbot.corrections import CorrectionHub
    corr = CorrectionHub(ev, [ctx]).evaluate([city, settle], [0, 0])
    assert (corr[1] - corr[0]) > (base[1] - base[0])


# ---------------------------------------------------------------------------
# the conversion opening policy
# ---------------------------------------------------------------------------
def _setup_state(seed=6, picks=0):
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    n = 0
    while n < picks:
        legal = E.legal_actions(s)
        a = openings.choose(s, legal, "setup_pick") or legal[0]
        if s.phase == PHASE_SETUP_SETTLEMENT:
            n += 1
        s = E.apply(s, a, rng)
    while s.phase != PHASE_SETUP_SETTLEMENT:
        s = E.apply(s, openings.choose(s, E.legal_actions(s), "setup_pick"), rng)
    return s


def test_policy_registered_parametric_and_reduces_to_setup_pick():
    assert "conversion" in openings.CANDIDATES and "conversion" in tuning.TUNABLES["openings.policy"].candidates
    s = _setup_state(picks=5)
    me = s.current
    ranked = sorted(placement.setup_pick(s, me, k=60), key=lambda t: (-t[1], t[0]))
    with tuning.overridden({"conversion.PORT_LEDGER": 0}):
        assert openings.policy("conversion:0").pick(s, me, 60) == ranked
    own = placement.player_production(s, me, ignore_robber=True)
    led = sorted(((v, sc - openings.spot_port_bonus(s, me, v, own)) for v, sc in ranked), key=lambda t: (-t[1], t[0]))
    assert openings.policy("conversion:0").pick(s, me, 60) == led
    assert openings.policy("conversion:1.25").pick(s, me, 60) == openings.conversion_pick(s, me, 60)
    for bad in ("conversion:x", "conversion:-2"):
        with pytest.raises(ValueError):
            openings.policy(bad)


def test_policy_adds_exactly_the_conversion_points():
    s = _setup_state(picks=4)                     # our second settlement (seat 3 picks twice in a row)
    me = s.current
    base = dict(placement.setup_pick(s, me, k=60))
    pts = openings.conversion_points(s, me, list(base))
    own = placement.player_production(s, me, ignore_robber=True)
    got = dict(openings.conversion_pick(s, me, 60))
    for v in base:
        assert got[v] == pytest.approx(base[v] - openings.CONV_WEIGHT * pts[v]
                                       - openings.spot_port_bonus(s, me, v, own))
    # points: KAPPA R (c(own + v) - c(own)); a spot adding only surplus costs, a complementary one saves
    assert max(pts.values()) > 0.0 > min(pts.values())


def test_policy_second_pick_complements_the_first():
    better = 0
    total = 0
    for seed in range(12):
        s = _setup_state(seed=seed, picks=4)
        me = s.current
        own = [x > 0 for x in placement.player_production(s, me, ignore_robber=True)]
        v_conv = openings.conversion_pick(s, me, 1)[0][0]
        v_base = placement.setup_pick(s, me, k=1)[0][0]
        new = lambda v: sum(1 for r, x in enumerate(placement.vertex_production(s, v, True)) if x > 0 and not own[r])
        better += new(v_conv) >= new(v_base)
        total += 1
    assert better == total
