"""Tests for catanbot/knightkick.py (``search.knight_kick``, docs/PRIORITY_PLAN.md step 5): the leaf chance node
"a blocked knight holder kicks the robber before our next turn", the alternative to robber_eval's persistence
correction, through the hub's leaf-chance hook (catanbot/corrections.py).  Off by default (``search.kick = 0``) and
off at depth >= 2."""
import random

import numpy as np
import pytest

from catanbot import board as B
from catanbot import corrections as H
from catanbot import engine as E
from catanbot import knightkick as KK
from catanbot import robber_eval as R
from catanbot import tuning
from catanbot import winpaths as W
from catanbot.heuristic import HeuristicEvaluator
from catanbot.search import SearchConfig, Searcher
from catanbot.selfplay import make_bot
from catanbot.state import PHASE_MAIN
from tests.test_corrections import positions, search_hash
from tests.test_robber_eval import blocked_state, roll_of

K = B.DEV_KNIGHT
D1 = dict(depth=1, beam=4, expand=8)


def finished_leaf(knight=True, t=0):
    """A finished leaf of seat ``me``: the robber blocks seat ``j`` (>= KICK_MIN), j rolls after ``t`` rolls."""
    s, h, j = blocked_state()
    roll_of(s, j, t)
    me = (s.current - 1) % 4          # the seat whose turn just ended
    if me == j:
        me = (j + 1) % 4
        s.current = (me + 1) % 4
    if knight:
        s.players[j].dev_cards[K] = 1
    return s, h, j, me


def test_kick_zero_bit_identical_and_off_at_depth_two():
    pos = positions(k=12)
    assert search_hash(SearchConfig(**D1, kick=0.0), pos) == search_hash(SearchConfig(**D1), pos)
    d2 = dict(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500)
    assert search_hash(SearchConfig(**d2, kick=0.95), pos[:5]) == search_hash(SearchConfig(**d2), pos[:5])
    assert not SearchConfig(**d2, kick=0.95).kick_active and SearchConfig(**D1, kick=0.95).kick_active
    s, me = pos[0]
    sr = Searcher(HeuristicEvaluator(), SearchConfig(**d2, kick=0.95))
    sr.search(s, me, random.Random(1))
    assert sr._corr is None                       # no hub at depth 2: the native lookahead stays available


def test_single_holder_mixture_value():
    s, h, j, me = finished_leaf()
    assert s.current != me
    kc = KK.KickChance(s, me, 0.95)
    outs = kc.leaf_outcomes(s, me)
    assert outs is not None and len(outs) == 2 and outs[0][1] is s
    (p0, _), (p1, s2) = outs
    assert abs(p0 - 0.05) < 1e-12 and abs(p1 - 0.95) < 1e-12
    # s' by hand: j plays its knight onto its best robber move from the root with the robber on h
    r = s.copy()
    hh, victim, _ = __import__("catanbot.robber", fromlist=["best_robber_move"]).best_robber_move(r, j)
    want = s.copy()
    want.robber = hh
    want.players[j].dev_cards[K] -= 1
    want.players[j].dev_count = sum(want.players[j].dev_cards) + sum(want.players[j].dev_cards_new)
    want.players[j].played_knights += 1
    E._update_largest_army(want, j)
    assert s2 == want and s2.robber != h
    hub = H.CorrectionHub(HeuristicEvaluator(), [], chance=[kc])
    v = float(HeuristicEvaluator().evaluate([s], [me])[0])
    v2 = float(HeuristicEvaluator().evaluate([want], [me])[0])
    got = hub.leaf_chance([s], [True], me, [v])
    assert abs(got[0] - (0.05 * v + 0.95 * v2)) < 1e-12
    assert hub.leaf_chance([s], [False], me, [v]) == [v]            # an unfinished node is unchanged


def test_no_holder_no_change_no_extra_evaluation(monkeypatch):
    s, h, j, me = finished_leaf(knight=False)
    kc = KK.KickChance(s, me, 0.95)
    assert kc.leaf_outcomes(s, me) is None
    calls = []
    hub = H.CorrectionHub(HeuristicEvaluator(), [], chance=[kc])
    monkeypatch.setattr(hub, "evaluate", lambda *a, **k: calls.append(1) or [])
    assert hub.leaf_chance([s], [True], me, [0.3]) == [0.3] and not calls
    # a small block is never kicked (the knight rule), even with a knight
    s.players[j].dev_cards[K] = 1
    s.hexes = list(s.hexes)
    res, num = s.hexes[h]
    s.hexes[h] = (res, 12)
    if R.block_values(s, h)[j] < KK.KICK_MIN:
        assert KK.KickChance(s, me, 0.95).leaf_outcomes(s, me) is None


def test_earliest_holder_in_roll_order_and_dealt_seats():
    s, h, j, me = finished_leaf(t=1)
    k = (j + 1) % 4
    if k == me:
        pytest.skip("construction put us next to the holder")
    v = next(v for v in B.HEX_VERTICES[h] if not any(v in p.settlements or v in p.cities for p in s.players))
    s.players[k].cities.append(v)
    s.players[k].dev_cards[K] = 1
    t = R.roll_offsets(s)
    first = min((j, k), key=lambda x: t[x])
    assert KK.KickChance(s, me, 0.95).kicker(s, me)[0] == first
    with R.knight_hint(dealt=[first]):
        kc = KK.KickChance.for_search(s, me, SearchConfig(**D1, kick=0.95))
        other = k if first == j else j
        assert kc.kicker(s, me)[0] == other
    with R.knight_hint(dealt=[j, k], p_knight={first: 0.5}):
        kc = KK.KickChance.for_search(s, me, SearchConfig(**D1, kick=0.95))
        (p0, _), (p1, _) = kc.leaf_outcomes(s, me)
        assert abs(p1 - 0.475) < 1e-12
    with R.knight_hint(dealt=[j, k]):
        assert KK.KickChance.for_search(s, me, SearchConfig(**D1, kick=0.95)).leaf_outcomes(s, me) is None


def test_kick_that_takes_largest_army_updates_vp():
    s, h, j, me = finished_leaf()
    s.players[j].played_knights = 2
    s.largest_army_owner = -1
    vp0 = s.public_vp(j)
    (_, _), (_, s2) = KK.KickChance(s, me, 0.95).leaf_outcomes(s, me)
    assert s2.largest_army_owner == j and s2.public_vp(j) == vp0 + 2 and s.largest_army_owner == -1
    s.players[(j + 2) % 4].played_knights = 3
    s.largest_army_owner = (j + 2) % 4
    (_, _), (_, s3) = KK.KickChance(s, me, 0.95).leaf_outcomes(s, me)
    assert s3.largest_army_owner == (j + 2) % 4                      # a tie never takes the army


def test_composes_with_paths_and_target_cache_per_search(monkeypatch):
    s, me = positions(k=1)[0]
    sr = Searcher(HeuristicEvaluator(), SearchConfig(**D1, kick=0.95, paths=1))
    assert sr.search(s, me, random.Random(1))
    assert [type(p) for p in sr._corr.providers] == [W.PathsContext]
    assert [type(c) for c in sr._corr.chance] == [KK.KickChance]
    leaf, h, j, me2 = finished_leaf()
    kc = KK.KickChance(leaf, me2, 0.95)
    calls = []
    import catanbot.knightkick as mod
    orig = mod.best_robber_move
    monkeypatch.setattr(mod, "best_robber_move", lambda st, p, *a, **k: calls.append((st.robber, p)) or orig(st, p))
    kc.leaf_outcomes(leaf, me2)
    kc.leaf_outcomes(leaf.copy(), me2)
    assert calls == [(h, j)] and kc.stats["target_misses"] == 1   # cached per (h, j*) within the search
    KK.KickChance(leaf, me2, 0.95).leaf_outcomes(leaf, me2)       # a new search: a new cache
    assert len(calls) == 2
    # the mixture uses the hub's values: with a paths provider the kicked state is valued with the correction
    hub = H.CorrectionHub(HeuristicEvaluator(), [W.PathsContext(leaf, me2)], chance=[KK.KickChance(leaf, me2, 1.0)])
    (_, _), (_, s2) = KK.KickChance(leaf, me2, 1.0).leaf_outcomes(leaf, me2)
    v = float(hub.evaluate([leaf], [me2])[0])
    assert abs(hub.leaf_chance([leaf], [True], me2, [v])[0] - float(hub.evaluate([s2], [me2])[0])) < 1e-12


def test_search_uses_the_chance_node_at_depth_one():
    pos = positions(k=24)
    changed = 0
    kicks = 0
    for s, me in pos:
        sr = Searcher(HeuristicEvaluator(), SearchConfig(**D1, kick=0.95))
        res = sr.search(s, me, random.Random(9))
        base = Searcher(HeuristicEvaluator(), SearchConfig(**D1)).search(s, me, random.Random(9))
        kicks += sr._corr.chance[0].stats["kicks"]
        if sr._corr.chance[0].stats["kicks"] == 0:
            assert [(r.action, r.value) for r in res] == [(r.action, r.value) for r in base]
        else:
            changed += [r.value for r in res] != [r.value for r in base]
    assert kicks > 0 and changed > 0


def test_registry_and_spec_key():
    t = tuning.TUNABLES["search.kick"]
    assert t.default == 0.0 and t.candidates == [0.95] and t.spec_key == "kick" and t.kind == "search"
    assert make_bot(tuning.DEFAULT_SEARCH_SPEC + ",kick=0.95").config.kick == 0.95
    assert make_bot(tuning.DEFAULT_SEARCH_SPEC).config.kick == 0.0
