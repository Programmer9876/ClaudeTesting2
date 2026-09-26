"""Tests for catanbot/robber_eval.py, the robber leaf corrections (docs/PRIORITY_PLAN.md step 5): R1a persistence
(``search.robber_corr``), R1b knight insurance (``robber_eval.INSURANCE_W``), R1c block duration
(``robber_eval.BLOCK_DUR_W``), all providers of the catanbot/corrections.py hub and all off by default.

The default must be untouched: the pinned digests below were produced by the code before this step (the queue's
epoch-B1 snapshot, /home/user/queue_runs/queue1/code/B1, which has no robber_eval), on python3 (catanatron 3.2.1)
and venv_cat33 (3.3.0) alike.
"""
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import pytest

from catanbot import actions as A
from catanbot import board as B
from catanbot import corrections as H
from catanbot import engine as E
from catanbot import robber_eval as R
from catanbot import tuning
from catanbot import winpaths as W
from catanbot.agents.param_bot import ParamBot
from catanbot.heuristic import HeuristicEvaluator
from catanbot.opponent_model import OpponentModel
from catanbot.search import SearchConfig, Searcher, reduced_config
from catanbot.selfplay import make_bot
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER, PHASE_ROLL, new_game
from tests.test_corrections import leaves, positions, search_hash
from tests.test_winpaths import clear_devs, played

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = tuning.DEFAULT_SEARCH_SPEC
K = B.DEV_KNIGHT
A_KEEP = 1.0 - 1.0 / R.R_REF


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def owners(s, h):
    return sorted({i for i, p in enumerate(s.players) for v in B.HEX_VERTICES[h] if v in p.settlements or v in p.cities})


def blocked_state(seed=5, min_pv=3.0):
    """A mid-game state (devs cleared) with the robber on a hex where exactly one seat ``j`` has buildings worth at
    least ``min_pv`` demand-weighted pips: ``(s, h, j)``."""
    for sd in range(seed, seed + 40):
        s = played(seed=sd, turns=50)
        clear_devs(s)
        for h in range(B.NUM_HEXES):
            o = owners(s, h)
            if len(o) == 1 and R.block_values(s, h)[o[0]] >= min_pv:
                s.robber = h
                return s, h, o[0]
    raise AssertionError("no blocked position found")


def roll_of(s, j, t):
    """Put the game in the roll phase of the seat ``t`` rolls before ``j`` (so ``t_j = t``)."""
    s.phase = PHASE_ROLL
    s.dice = 0
    s.dev_played_this_turn = False
    s.current = (j - t) % s.num_players
    return s


def ctx_of(s, me, **params):
    P = R.current_params()
    P.update(params)
    return R.RobberContext(s, me, params=P)


# ---------------------------------------------------------------------------
# the default is untouched
# ---------------------------------------------------------------------------
_HASH_SCRIPT = r"""
import hashlib, json, random, sys
sys.path.insert(0, ROOT)
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.heuristic import HeuristicEvaluator
from catanbot.search import SearchConfig, Searcher
from catanbot.selfplay import make_bot, play_game
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROLL, PHASE_ROBBER, new_game
SEEDS = [11, 12, 13, 14, 15, 16]


def positions(k=30):
    out = []
    for seed in (101, 202, 303):
        rng = random.Random(seed)
        s = new_game(4, rng=rng)
        bots = [HeuristicBot(temperature=0.15) for _ in range(4)]
        cand = []
        while s.phase != PHASE_GAME_OVER and s.turn < 90:
            i = E.acting_player(s)
            legal = E.legal_actions(s)
            if s.phase in (PHASE_MAIN, PHASE_ROLL, PHASE_ROBBER) and s.turn >= 20 and len(legal) > 1:
                cand.append((s.copy(), i))
            s = E.apply(s, bots[i].decide(s, legal, rng), rng)
        step = max(1, len(cand) // 10)
        out.extend(cand[::step][:10])
    return out[:k]


def game_hash(bots, seed):
    h = hashlib.sha256()
    res = play_game(bots, rng=random.Random(seed), seed=seed, max_turns=28,
                    on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    return h.hexdigest()[:16]


def search_hash(cfg, pos):
    h = hashlib.sha256()
    for s, me in pos:
        res = Searcher(HeuristicEvaluator(), cfg).search(s, me, random.Random(9))
        h.update(repr([(r.action, r.value, r.static, r.line, r.explanation) for r in res]).encode())
    return h.hexdigest()[:16]


spec = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
pos = positions()
out = {"games": [game_hash([make_bot(spec) for _ in range(4)], sd) for sd in SEEDS],
       "search_d1": search_hash(SearchConfig(depth=1, beam=4, expand=8), pos),
       "search_d2": search_hash(SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500), pos[:8])}
out["robber_eval_imported_default"] = "catanbot.robber_eval" in sys.modules
out["knightkick_imported_default"] = "catanbot.knightkick" in sys.modules
from catanbot import tuning
from catanbot.agents.param_bot import ParamBot
out["games_spelled"] = [game_hash([make_bot(spec + ",robber_corr=0,kick=0") for _ in range(4)], sd) for sd in SEEDS]
ov = {"robber_eval.PERSIST_W": 0.3, "robber_eval.INSURANCE_W": 2.0, "robber_eval.BLOCK_DUR_W": 0.5,
      "robber_eval.KICK_MIN": 1.0, "robber_eval.PLACEBO": 1}
out["games_parambot"] = [game_hash([ParamBot(make_bot(spec), ov) for _ in range(4)], sd) for sd in SEEDS[:2]]
out["search_d1_spelled"] = search_hash(SearchConfig(depth=1, beam=4, expand=8, robber_corr=0, kick=0.0), pos)
out["search_d2_kick"] = search_hash(SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500,
                                                 kick=0.95), pos[:8])
out["knightkick_imported"] = "catanbot.knightkick" in sys.modules
print(json.dumps(out))
"""

# sha256[:16] digests of the pre-step code (queue epoch B1 snapshot; python3 and venv_cat33 agree)
PINNED_GAMES = ["114017e87a7c9666", "4c3848122aa9dc06", "981527ce906dc306", "c215623717ca25fc", "78a8371348f5a995",
                "d497eb2042e626ce"]
PINNED_SEARCH_D1 = "1d6d82b9f443c426"
PINNED_SEARCH_D2 = "83320445c260caed"


def test_robber_corr_off_byte_identical():
    """A fresh interpreter plays 6 seeded default games (28 turns) and 30 depth-1 + 8 depth-2 root searches: equal
    to the pre-step code's digests, without importing robber_eval / knightkick.  Then, with the registry
    imported: the switches spelled out at 0, ParamBots whose robber_eval weights are overridden while
    robber_corr = 0, and kick = 0.95 at depth 2 (off there) all give the same digests."""
    code = f"ROOT = {ROOT!r}\n" + _HASH_SCRIPT
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=1200,
                          env=dict(os.environ, PYTHONHASHSEED="0"))
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["games"] == PINNED_GAMES == out["games_spelled"]
    assert out["games_parambot"] == PINNED_GAMES[:2]
    assert out["search_d1"] == PINNED_SEARCH_D1 == out["search_d1_spelled"]
    assert out["search_d2"] == PINNED_SEARCH_D2 == out["search_d2_kick"]
    assert not out["robber_eval_imported_default"] and not out["knightkick_imported_default"]
    assert not out["knightkick_imported"]


def test_off_builds_no_hub_and_no_context(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("RobberContext built with robber_corr = 0")

    monkeypatch.setattr(R.RobberContext, "__init__", boom)
    s, me = positions(k=1)[0]
    for cfg in (SearchConfig(depth=1, beam=4, expand=8), SearchConfig(depth=2, beam=2, expand=4, max_nodes=1500,
                                                                      opp_roll_samples=2)):
        sr = Searcher(HeuristicEvaluator(), cfg)
        assert sr.search(s, me, random.Random(1)) and sr._corr is None


def test_robber_corr_zero_weights_action_identical():
    """robber_corr = 1 with every term at weight 0: the hub is built, every correction is None, and the searches
    are the default's bit for bit (the hub's fast path)."""
    pos = positions(k=12)
    base = search_hash(SearchConfig(depth=1, beam=4, expand=8), pos)
    with tuning.overridden({"robber_eval.PERSIST_W": 0.0}):
        assert search_hash(SearchConfig(depth=1, beam=4, expand=8, robber_corr=1), pos) == base
        s, me = pos[0]
        sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, robber_corr=1))
        sr.search(s, me, random.Random(1))
        assert isinstance(sr._corr.providers[-1], R.RobberContext) and sr._corr.stats["corrected"] == 0


# ---------------------------------------------------------------------------
# R1a persistence
# ---------------------------------------------------------------------------
def test_persistence_zero_without_would_kick_holders():
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    me = (j + 2) % 4
    assert R.RobberContext(s, me).corrections(s) is None           # nobody holds a knight
    s.players[(j + 1) % 4].dev_cards[K] = 1                         # a knight holder who is not blocked
    assert R.RobberContext(s, me).corrections(s) is None
    s.robber = next(h2 for h2 in range(B.NUM_HEXES) if s.hexes[h2][0] == B.DESERT)
    s.players[j].dev_cards[K] = 1
    assert R.RobberContext(s, me).corrections(s) is None           # the robber on the desert blocks nobody


def test_small_block_knight_holder_restores_nothing():
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    s.players[j].dev_cards[K] = 1
    pv = R.block_values(s, h)[j]
    me = (j + 2) % 4
    assert ctx_of(s, me, KICK_MIN=pv + 0.01).corrections(s) is None      # pv_k(h) < KICK_MIN: rho = 0
    assert ctx_of(s, me, KICK_MIN=pv).corrections(s) is not None


def test_next_roller_restore_equals_kick_lambda():
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    s.players[j].dev_cards[K] = 1
    me = (j + 2) % 4
    pv = R.block_values(s, h)
    C = R.RobberContext(s, me).corrections(s)
    for i in range(4):
        want = R.BLOCK_SHARE * pv[i] * R.KICK_LAMBDA if pv[i] > 0 else 0.0
        assert abs(C[i] - want) < 1e-12
    assert C[j] > 0 and abs(C[j] / (R.BLOCK_SHARE * pv[j]) - 0.95) < 1e-12
    # a knight bought this turn (dev_cards_new) counts too: it is playable at the next turn start
    s.players[j].dev_cards[K], s.players[j].dev_cards_new[K] = 0, 1
    assert R.RobberContext(s, me).corrections(s) == C


def test_geometric_keep():
    s, h, j = blocked_state()
    s.players[j].dev_cards[K] = 1
    me = (j + 2) % 4
    pv = R.block_values(s, h)[j]
    for t in (0, 1, 2, 3):
        roll_of(s, j, t)
        C = R.RobberContext(s, me).corrections(s)
        assert abs(C[j] - R.BLOCK_SHARE * pv * R.KICK_LAMBDA * A_KEEP ** t) < 1e-12, t
    # after the roll (main phase of the seat before j): j still rolls next, t = 0
    roll_of(s, j, 1)
    s.phase = PHASE_MAIN
    s.dice = 6
    assert abs(R.RobberContext(s, me).corrections(s)[j] - R.BLOCK_SHARE * pv * R.KICK_LAMBDA) < 1e-12
    # j itself in its roll phase after already playing a development card: the next chance is n rolls away
    roll_of(s, j, 0)
    s.dev_played_this_turn = True
    assert abs(R.RobberContext(s, me).corrections(s)[j] - R.BLOCK_SHARE * pv * R.KICK_LAMBDA * A_KEEP ** 4) < 1e-12


def _two_holders():
    """Seats j and k (k rolls two turns after j) both blocked on the robber hex, both worth >= KICK_MIN."""
    s, h, j = blocked_state()
    k = (j + 2) % 4
    v = next(v for v in B.HEX_VERTICES[h] if not any(v in p.settlements or v in p.cities for p in s.players))
    s.players[k].cities.append(v)                  # a city: at least as much as j's pips there
    assert R.block_values(s, h)[k] >= R.KICK_MIN
    roll_of(s, j, 0)
    return s, h, j, k


def test_earliest_kicker_order():
    s, h, j, k = _two_holders()
    me = (j + 1) % 4
    pv = R.block_values(s, h)
    lam, a = R.KICK_LAMBDA, A_KEEP
    s.players[j].dev_cards[K] = 1
    s.players[k].dev_cards[K] = 1
    both = R.RobberContext(s, me).corrections(s)
    rho = lam + (1 - lam) * lam * a ** 2            # the earlier roller dominates
    for i in (j, k):
        assert abs(both[i] - R.BLOCK_SHARE * pv[i] * rho) < 1e-12
    s.players[j].dev_cards[K] = 0
    late = R.RobberContext(s, me).corrections(s)    # only the later roller holds a knight: a smaller restore
    assert abs(late[j] - R.BLOCK_SHARE * pv[j] * lam * a ** 2) < 1e-12 and late[j] < both[j]
    assert R.first_kicks([(0, j, lam), (2, k, lam)]) == [(0, j, lam), (2, k, (1 - lam) * lam)]


def test_dealt_seat_knights_not_trusted():
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    s.players[j].dev_cards[K] = 1
    me = (j + 2) % 4
    pv = R.block_values(s, h)[j]
    assert R.RobberContext(s, me).corrections(s) is not None       # full information: trusted
    with R.knight_hint(dealt=[j]):
        assert R.RobberContext.for_search(s, me, SearchConfig()).corrections(s) is None
    with R.knight_hint(dealt=[j], p_knight={j: 0.5}):
        C = R.RobberContext.for_search(s, me, SearchConfig()).corrections(s)
        assert abs(C[j] - R.BLOCK_SHARE * pv * R.KICK_LAMBDA * 0.5) < 1e-12
    assert R.current_hint() is None
    s.players[j].dev_known = False                                  # unknown (public view): not trusted either
    assert R.RobberContext(s, me).corrections(s) is None
    s.players[j].dev_known = True
    # our own knights are always known, a hint never hides them
    roll_of(s, j, 3)
    with R.knight_hint(dealt=[0, 1, 2, 3]):
        assert R.RobberContext(s, j).corrections(s) is not None


def test_search_determinized_hides_dealt_knights(monkeypatch):
    """The advisor path (search.search_determinized): the seats whose dev cards the samples dealt are hinted."""
    from catanbot import search as S
    seen = []
    orig = R.RobberContext.for_search.__func__

    def spy(cls, root, me, cfg):
        seen.append(R.current_hint())
        return orig(cls, root, me, cfg)

    monkeypatch.setattr(R.RobberContext, "for_search", classmethod(spy))
    s, me = positions(k=1)[0]
    s = s.copy()
    hidden = (me + 1) % 4
    s.players[hidden].dev_known = False
    s.players[hidden].dev_count = 1
    S.search_determinized(s, me, HeuristicEvaluator(), SearchConfig(depth=1, beam=2, expand=4, robber_corr=1),
                          samples=2, rng=random.Random(3))
    assert seen and all(h is not None and h["dealt"] == frozenset([hidden]) for h in seen)
    assert R.current_hint() is None


def test_cache_cold_vs_warm_identical():
    """The pv tables are keyed on every seat's buildings: an opponent's new building inside the search (as with
    respond_lookahead) is a new key; warm and cold contexts give the same corrections."""
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    for i in range(4):
        s.players[i].dev_cards[K] = 1
    me = (j + 2) % 4
    s2 = s.copy()
    opp = (me + 1) % 4
    v = next(v for v in range(B.NUM_VERTICES) if not any(v in p.settlements or v in p.cities for p in s2.players)
             and B.VERTEX_HEXES[v] and h not in B.VERTEX_HEXES[v])
    s2.players[opp].settlements.append(v)
    kw = dict(INSURANCE_W=1.0, RETAL_W=1.0, INS_OFFSET=1.0)
    warm = ctx_of(s, me, **kw)
    a1 = warm.corrections(s)
    a2 = warm.corrections(s2)
    assert warm.stats["table_misses"] == 2 and a1 != a2
    assert ctx_of(s2, me, **kw).corrections(s2) == a2 and ctx_of(s, me, **kw).corrections(s) == a1
    assert warm.corrections(s) == a1 and warm.stats["table_misses"] == 2


def test_setup_and_game_over_pass_through():
    ls, me = leaves(n=40)
    over, setup = ls[-2], ls[-1]
    ctx = ctx_of(ls[0], me, BLOCK_DUR_W=0.5, INSURANCE_W=1.0)
    assert ctx.corrections(over) is None and ctx.corrections(setup) is None
    hub = H.CorrectionHub(HeuristicEvaluator(), [ctx])
    got = hub.evaluate(ls, [me] * len(ls))
    base = HeuristicEvaluator().evaluate(ls, [me] * len(ls))
    assert got[-1] == base[-1] and got[-2] == base[-2]
    assert hub.stats["corrected"] > 0


def _two_target_position():
    """Our robber move after a 7 (seat 0): hex hA blocks only seat 1 (the next roller, a knight holder), hex hB
    only seat 2 (no knight); every other numbered hex is a 12."""
    s = played(seed=12, turns=50)
    clear_devs(s)
    s.current, s.phase, s.dice = 0, PHASE_ROBBER, 7
    hA = next(h for h in range(B.NUM_HEXES) if h != s.robber and owners(s, h) == [1])
    hB = next(h for h in range(B.NUM_HEXES) if h != s.robber and owners(s, h) == [2])
    hexes = []
    for h, (res, num) in enumerate(s.hexes):
        if h in (hA, hB):
            hexes.append((B.WHEAT, 6 if h == hA else 8))
        elif res != B.DESERT and num and h != s.robber:
            hexes.append((res, 12 if num != 2 else 2))
        else:
            hexes.append((res, num))
    s.hexes = hexes
    s.players[1].dev_cards[K] = 1
    for i in (1, 2):
        s.players[i].resources = [1, 1, 1, 1, 0]
        s.players[i].hand_size = 4
    return s, hA, hB


def test_search_avoids_next_roller_knight_holder():
    s, hA, hB = _two_target_position()
    base = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8)).search(s, 0, random.Random(1))
    cand = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, robber_corr=1)).search(
        s, 0, random.Random(1))
    assert base[0].action[0] == A.MOVE_ROBBER and base[0].action[1] == hA        # the default blocks the kicker
    assert cand[0].action[0] == A.MOVE_ROBBER and cand[0].action[1] != hA
    after = s.copy()
    after.robber, after.phase = cand[0].action[1], PHASE_MAIN
    assert not [x for x in R.kick_plan(after, 0, R.block_values(after, after.robber)) if x[1] != 0 and x[0] <= 1]
    s.players[1].dev_cards[K] = 0                                                  # no knight: no change
    again = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, robber_corr=1)).search(
        s, 0, random.Random(1))
    plain = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8)).search(s, 0, random.Random(1))
    assert again[0].action == plain[0].action


def test_paths_and_robber_corr_compose():
    """The hub replaced the old 'robber_corr + paths refused' rule: both are providers (winpaths first)."""
    s, me = positions(k=1)[0]
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, paths=1, robber_corr=1))
    assert sr.search(s, me, random.Random(1))
    assert [type(p) for p in sr._corr.providers] == [W.PathsContext, R.RobberContext] and sr._paths is None
    assert reduced_config(SearchConfig(robber_corr=1, kick=0.95), 2, 600).robber_corr == 1
    assert make_bot(DEFAULT + ",robber_corr=1,paths=1").config.robber_corr == 1


def test_persistence_and_kick_never_stacked():
    s, me = positions(k=1)[0]
    with pytest.raises(ValueError, match="never stack"):
        Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, robber_corr=1, kick=0.95)).search(
            s, me, random.Random(1))
    with tuning.overridden({"robber_eval.PERSIST_W": 0.0, "robber_eval.INSURANCE_W": 1.0}):
        assert Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, robber_corr=1,
                                                           kick=0.95)).search(s, me, random.Random(1))


def test_param_bot_builds_context_only_for_candidate_seats(monkeypatch):
    built = []
    orig = R.RobberContext.__init__

    def spy(self, root, me, *a, **k):
        built.append(me)
        orig(self, root, me, *a, **k)

    monkeypatch.setattr(R.RobberContext, "__init__", spy)
    s, me = positions(k=1)[0]
    legal = E.legal_actions(s)
    make_bot(DEFAULT).decide(s, legal, random.Random(1))
    ParamBot(make_bot(DEFAULT), {}).decide(s, legal, random.Random(1))
    assert built == []
    cand = ParamBot(make_bot(DEFAULT), {"search.robber_corr": 1})
    cand.decide(s, legal, random.Random(1))
    assert built and set(built) == {me} and cand.inner.config.robber_corr == 0     # restored after the decision


def test_registry_robber_eval():
    assert tuning.verify_registry() == []
    for n in ("search.robber_corr", "search.kick"):
        t = tuning.TUNABLES[n]
        assert t.kind == "search" and t.spec_key == t.attr and t.requires_depth == 1 and t.requires_search
        assert not t.needs_python_evaluator
        assert getattr(SearchConfig(), t.attr) == t.default == getattr(make_bot(DEFAULT).config, t.attr)
    assert tuning.TUNABLES["search.robber_corr"].default == 0 and tuning.TUNABLES["search.kick"].default == 0.0
    names = [f"robber_eval.{a}" for a in R._PARAM_NAMES]
    for n in names:
        t = tuning.TUNABLES[n]
        assert t.kind == "weight" and t.requires_search and not t.needs_python_evaluator and not t.clear_caches
        assert t.default == getattr(R, t.attr) and "only with search.robber_corr=1" in t.description
    for n, off in (("robber_eval.INSURANCE_W", 0.0), ("robber_eval.BLOCK_DUR_W", 0.0), ("robber_eval.RETAL_W", 0.0),
                   ("robber_eval.PLACEBO", 0)):
        assert tuning.TUNABLES[n].default == off
    assert (R.PERSIST_W, R.KICK_LAMBDA, R.KICK_MIN, R.R_REF) == (1.0, 0.95, 3.0, 4.34)
    tok = tuning.apply({"robber_eval.INSURANCE_W": 2.0, "robber_eval.PLACEBO": 1})
    try:
        assert R.INSURANCE_W == 2.0 and R.PLACEBO == 1 and R.current_params()["INSURANCE_W"] == 2.0
    finally:
        tuning.restore(tok)
    assert R.INSURANCE_W == 0.0 and R.PLACEBO == 0
    assert "robber" in tuning.SEAT_OBSERVERS


def test_placebo_rotates_corrections():
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    s.players[j].dev_cards[K] = 1
    me = (j + 2) % 4
    C = ctx_of(s, me).corrections(s)
    P = ctx_of(s, me, PLACEBO=1).corrections(s)
    assert P == [C[(i + 1) % 4] for i in range(4)] and P != C


def test_cost_budget_fixed_positions():
    """ms ratio of robber_corr = 1 over the default on 30 fixed roots (process time, best of two) <= 1.3."""
    pos = positions(k=30)

    def run(cfg):
        best = None
        for _ in range(2):
            t0 = time.process_time()
            for s, me in pos:
                Searcher(HeuristicEvaluator(), cfg).search(s, me, random.Random(9))
            dt = time.process_time() - t0
            best = dt if best is None else min(best, dt)
        return best

    base = run(SearchConfig(depth=1, beam=4, expand=8))
    cand = run(SearchConfig(depth=1, beam=4, expand=8, robber_corr=1))
    assert cand / base <= 1.3, (cand, base)


# ---------------------------------------------------------------------------
# R1b knight insurance
# ---------------------------------------------------------------------------
def _ins_state():
    s, h, j = blocked_state()
    roll_of(s, j, 1)
    me = (j + 2) % 4
    holder = (j + 3) % 4                    # not blocked by the robber
    assert R.block_values(s, h)[holder] == 0.0
    s.players[holder].dev_cards[K] = 1
    return s, h, j, me, holder


def _c_ins(s, me, i, **kw):
    P = dict(PERSIST_W=0.0, INSURANCE_W=1.0)
    P.update(kw)
    C = ctx_of(s, me, **P).corrections(s)
    return 0.0 if C is None else C[i]


def test_insurance_independent_of_hand_size():
    s, h, j, me, holder = _ins_state()
    c0 = _c_ins(s, me, holder)
    assert c0 != 0.0
    for extra in ([3, 0, 0, 0, 0], [2, 2, 2, 2, 2]):
        s2 = s.copy()
        s2.players[holder].resources = [a + b for a, b in zip(s2.players[holder].resources, extra)]
        s2.players[holder].hand_size = sum(s2.players[holder].resources)
        assert _c_ins(s2, me, holder) == c0


def test_insurance_spare_knight_for_blocked_holder_with_two():
    s, h, j, me, holder = _ins_state()
    s.players[j].dev_cards[K] = 1              # blocked (>= KICK_MIN): its only knight goes to the kick
    assert _c_ins(s, me, j, INS_OFFSET=0.0) == 0.0
    ctx = ctx_of(s, me, PERSIST_W=0.0, INSURANCE_W=1.0)
    assert ctx.insured(s, j, True) == 0.0 and ctx.insured(s, j, False) == 1.0
    s.players[j].dev_cards[K] = 2              # a spare knight: insured after kicking
    assert ctx.insured(s, j, True) == 1.0 and _c_ins(s, me, j, INS_OFFSET=0.0) > 0.0


def test_insurance_first_knight_only():
    s, h, j, me, holder = _ins_state()
    c1 = _c_ins(s, me, holder)
    s.players[holder].dev_cards[K] = 3
    assert _c_ins(s, me, holder) == c1


def test_insurance_centred_mean_zero_on_fixture_leaves():
    """INS_OFFSET = the mean D x P_hit over the insured players at a set of leaves (the shadow's calibration)
    makes the mean C_ins over those leaves 0: the term moves knight value, it does not add it."""
    ls, me = leaves(n=80)
    ls = [s.copy() for s in ls[:-2]]
    for s in ls:
        s.players[(me + 1) % 4].dev_cards[K] = 1
        s.players[(me + 2) % 4].dev_cards[K] = 1
    probe = ctx_of(ls[0], me, PERSIST_W=0.0, INSURANCE_W=1.0, INS_OFFSET=0.0)
    for s in ls:
        probe.corrections(s)
    assert probe.stats["ins_n"] >= len(ls)
    cal = probe.stats["ins_sum_dp"] / probe.stats["ins_n"]
    centred = ctx_of(ls[0], me, PERSIST_W=0.0, INSURANCE_W=1.0, INS_OFFSET=cal)
    for s in ls:
        centred.corrections(s)
    assert centred.stats["ins_n"] == probe.stats["ins_n"]
    assert abs(centred.stats["ins_sum_c"] / centred.stats["ins_n"]) < 1e-9


def test_insurance_uses_opponent_model_shares_with_trust_ramp():
    s, h, j, me, holder = _ins_state()
    ctx = ctx_of(s, me, PERSIST_W=0.0, INSURANCE_W=1.0)
    model = OpponentModel(s)
    robber_seat = (holder + 1) % 4
    name = lambda i: s.players[i].name or s.players[i].color   # noqa: E731
    bp = ctx.blockable(s, ctx.pv_table(s))
    thr = [R.threat(s, x) for x in range(4)]
    prior = ctx.p_target(s, robber_seat, bp, thr)
    ctx.bind_model(model)                                        # no observations yet: the prior
    assert ctx.p_target(s, robber_seat, bp, thr) == prior
    model.profiles[name(robber_seat)].robbed = {name(holder): 1.5}
    ctx.bind_model(model)                                        # tau = 0.5, share 1.0 on the holder
    got = ctx.p_target(s, robber_seat, bp, thr)
    assert abs(got[holder] - (0.5 * prior[holder] + 0.5)) < 1e-12
    model.profiles[name(robber_seat)].robbed = {name(holder): 6.0}
    ctx.bind_model(model)                                        # tau = 1
    assert ctx.p_target(s, robber_seat, bp, thr)[holder] == 1.0
    before = _c_ins(s, me, holder)
    c = ctx_of(s, me, PERSIST_W=0.0, INSURANCE_W=1.0)
    c.bind_model(model)
    assert c.corrections(s)[holder] > before                     # a habitual robber of the holder: more insurance
    assert len(model.profiles) == 4                              # read only: no profile created


def test_insurance_monotone_in_p_hit_and_exposure():
    s, h, j, me, holder = _ins_state()
    base = _c_ins(s, me, holder)
    ctx = ctx_of(s, me, PERSIST_W=0.0, INSURANCE_W=1.0)
    p_hit, d = ctx.exposure(s, holder)
    assert 0.0 < p_hit < 1.0 and d > 0.0
    s2 = s.copy()                                                # more exposure: another building on its best hex
    table = ctx.pv_table(s2)
    best = max((x for x in range(B.NUM_HEXES) if x != s2.robber), key=lambda x: table[holder][x])
    s2.players[holder].cities.append(next(v for v in B.HEX_VERTICES[best]
                                          if not any(v in p.settlements or v in p.cities for p in s2.players)))
    assert ctx_of(s2, me).exposure(s2, holder)[1] > d
    assert _c_ins(s2, me, holder) > base
    s3 = s.copy()                                                # higher P_hit: every opponent can knight us
    for x in range(4):
        if x != holder:
            s3.players[x].dev_cards[K] = 1
    assert ctx_of(s3, me).exposure(s3, holder)[0] > p_hit and _c_ins(s3, me, holder) > base
    lo, hi = _c_ins(s, me, holder, INSURANCE_W=1.0), _c_ins(s, me, holder, INSURANCE_W=2.0)
    assert abs(hi - 2 * lo) < 1e-12


def test_insurance_zero_for_dealt_seats_without_posterior():
    s, h, j, me, holder = _ins_state()
    assert _c_ins(s, me, holder) != 0.0
    P = dict(R.current_params(), PERSIST_W=0.0, INSURANCE_W=1.0)
    with R.knight_hint(dealt=[holder]):
        ctx = R.RobberContext(s, me, params=P, hint=R.current_hint())
        assert ctx.insured(s, holder, False) == 0.0
        C = ctx.corrections(s)
        assert C is None or C[holder] == 0.0
    with R.knight_hint(dealt=[holder], p_knight={holder: 0.4}):
        ctx = R.RobberContext(s, me, params=P, hint=R.current_hint())
        assert ctx.insured(s, holder, False) == 0.4 and ctx.insured(s, holder, True) == 0.0


def test_insurance_w_zero_action_identical():
    pos = positions(k=10)
    base = search_hash(SearchConfig(depth=1, beam=4, expand=8), pos)
    with tuning.overridden({"robber_eval.PERSIST_W": 0.0, "robber_eval.INSURANCE_W": 0.0,
                            "robber_eval.INS_OFFSET": 2.0, "robber_eval.TGT_LEAD": 0.9}):
        assert search_hash(SearchConfig(depth=1, beam=4, expand=8, robber_corr=1), pos) == base


# ---------------------------------------------------------------------------
# R1c block duration
# ---------------------------------------------------------------------------
def test_block_dur_zero_is_zero():
    s, h, j = blocked_state()
    roll_of(s, j, 0)                                 # a block, no knight: R1a silent
    assert ctx_of(s, 0, BLOCK_DUR_W=0.0).corrections(s) is None


def test_block_dur_restores_fraction():
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    pv = R.block_values(s, h)
    for w in (0.25, 0.5, 0.8):
        C = ctx_of(s, (j + 2) % 4, BLOCK_DUR_W=w).corrections(s)
        for i in range(4):
            assert abs(C[i] - w * R.BLOCK_SHARE * pv[i]) < 1e-12


def test_block_dur_composes_with_persistence():
    s, h, j = blocked_state()
    s.players[j].dev_cards[K] = 1
    roll_of(s, j, 2)
    pv = R.block_values(s, h)[j]
    rho = R.KICK_LAMBDA * A_KEEP ** 2
    w = 0.5
    C = ctx_of(s, (j + 2) % 4, BLOCK_DUR_W=w).corrections(s)
    assert abs(C[j] - R.BLOCK_SHARE * pv * (rho + w * (1 - rho))) < 1e-12
    only = ctx_of(s, (j + 2) % 4, BLOCK_DUR_W=w, PERSIST_W=0.0).corrections(s)
    assert abs(only[j] - R.BLOCK_SHARE * pv * w) < 1e-12


def test_block_dur_zero_action_identical():
    pos = positions(k=10)
    base = search_hash(SearchConfig(depth=1, beam=4, expand=8), pos)
    with tuning.overridden({"robber_eval.PERSIST_W": 0.0, "robber_eval.BLOCK_DUR_W": 0.0}):
        assert search_hash(SearchConfig(depth=1, beam=4, expand=8, robber_corr=1), pos) == base


# ---------------------------------------------------------------------------
# retaliation (politics rule: 0 by default)
# ---------------------------------------------------------------------------
def test_retaliation_only_with_its_weight():
    s, h, j = blocked_state()
    roll_of(s, j, 0)
    s.players[j].dev_cards[K] = 1
    me = (j + 2) % 4
    base = ctx_of(s, me).corrections(s)
    ret = ctx_of(s, me, RETAL_W=1.0).corrections(s)
    assert ret[me] < base[me] and [ret[i] for i in range(4) if i != me] == [base[i] for i in range(4) if i != me]
    assert tuning.TUNABLES["robber_eval.RETAL_W"].default == 0.0
