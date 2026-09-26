"""Tests for the leaf-value correction hub (catanbot/corrections.py) and the per-seat observer hook of
tuning.play_paired_game (docs/PRIORITY_PLAN.md step 2: ``shared.corrections_hook``, ``shared.seat_events``).

The hub must be invisible when unused: the default bot and ``search.paths`` (winpaths) play and search exactly as
before the refactor - the digests below were produced by the pre-hub code (same C++ build, python3 and
venv_cat33 agree) - and a hub whose corrections are all zero returns the base evaluator's values bit for bit.
"""
import hashlib
import random

import numpy as np
import pytest

from catanbot import actions as A
from catanbot import corrections as H
from catanbot import engine as E
from catanbot import tuning
from catanbot import winpaths as W
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.heuristic import HeuristicEvaluator
from catanbot.search import SearchConfig, Searcher, reduced_config
from catanbot.selfplay import BlendedEvaluator, make_bot, play_game
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROLL, PHASE_SETUP_SETTLEMENT, new_game

DEFAULT = tuning.DEFAULT_SEARCH_SPEC


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def positions(k=24, seeds=(101, 202, 303), per=8, max_turn=110):
    """Mid-game roots (roll / main phase, more than one legal action) from heuristic self-play."""
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        s = new_game(4, rng=rng)
        bots = [HeuristicBot(temperature=0.15) for _ in range(4)]
        cand = []
        while s.phase != PHASE_GAME_OVER and s.turn < max_turn:
            i = E.acting_player(s)
            legal = E.legal_actions(s)
            if s.phase in (PHASE_MAIN, PHASE_ROLL) and s.turn >= 20 and len(legal) > 1:
                cand.append((s.copy(), i))
            s = E.apply(s, bots[i].decide(s, legal, rng), rng)
        step = max(1, len(cand) // per)
        out.extend(cand[::step][:per])
    return out[:k]


def game_hash(specs, seed, max_turns):
    h = hashlib.sha256()
    res = play_game([make_bot(sp) for sp in specs], rng=random.Random(seed), seed=seed, max_turns=max_turns,
                    on_action=lambda st, a, p: h.update(repr((p, a)).encode()))
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    return h.hexdigest()[:16]


def search_hash(cfg, pos, ev=None):
    h = hashlib.sha256()
    for s, me in pos:
        res = Searcher(ev or HeuristicEvaluator(), cfg).search(s, me, random.Random(9))
        h.update(repr([(r.action, r.value, r.static, r.line, r.explanation) for r in res]).encode())
    return h.hexdigest()[:16]


def leaves(seed=6, n=80):
    """Leaf states of one depth-1 search (varied: builds, trades, rolls) plus a finished game and a setup state."""
    s = positions(k=1, seeds=(seed,))[0][0]
    me = E.acting_player(s)
    got = []

    class Spy(HeuristicEvaluator):
        def evaluate(self, states, players):
            got.extend(states)
            return super().evaluate(states, players)

    Searcher(Spy(), SearchConfig(depth=1, beam=4, expand=8)).search(s, me, random.Random(1))
    out = got[:n]
    over = out[0].copy()
    over.phase, over.winner = PHASE_GAME_OVER, 2
    setup = new_game(4, rng=random.Random(3))
    assert setup.phase == PHASE_SETUP_SETTLEMENT
    return out + [over, setup], me


class StubNet:
    def evaluate(self, states, players):
        return np.array([0.1 + 0.05 * ((len(st.players[p].roads) + p) % 7) for st, p in zip(states, players)])


def softmax(z, T):
    m = max(z)
    e = [np.exp((v - m) / T) for v in z]
    return [x / sum(e) for x in e]


# ---------------------------------------------------------------------------
# the hub as an evaluator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base", [HeuristicEvaluator(), BlendedEvaluator(StubNet(), 0.5), StubNet()])
def test_fast_path_is_bit_identical_when_every_correction_is_zero(base):
    ls, me = leaves()
    players = [i % 4 for i in range(len(ls))]
    zero = H.ConstantProvider([0.0, 0.0, 0.0, 0.0])
    hub = H.CorrectionHub(base, [zero, H.ConstantProvider([0.0, -0.0, 0.0, 0.0])])
    got = hub.evaluate(ls, players)
    want = np.asarray(base.evaluate(ls, players), dtype=np.float64)
    assert got.tobytes() == want.tobytes()
    assert hub.stats["corrected"] == 0 and hub.stats["passed"] == len(ls)
    assert zero.calls == len(ls) - 2           # setup and finished game never reach the providers


def test_corrected_leaves_setup_and_game_over():
    ls, me = leaves()
    over, setup = ls[-2], ls[-1]
    players = [i % 4 for i in range(len(ls))]
    C = [1.5, -2.0, 0.0, 4.0]
    for base in (HeuristicEvaluator(), BlendedEvaluator(StubNet(), 0.3), StubNet()):
        hub = H.CorrectionHub(base, [H.ConstantProvider(C)])
        got = hub.evaluate(ls, players)
        plain = np.asarray(base.evaluate(ls, players), dtype=np.float64)
        T = 16.0
        for k, (s, p) in enumerate(zip(ls, players)):
            if s is over or s is setup:
                assert got[k] == plain[k]          # pass through
                continue
            V = H.static_values(s)
            new = softmax([V[i] + C[i] for i in range(4)], T)[p]
            if isinstance(base, HeuristicEvaluator):
                assert abs(got[k] - new) < 1e-12
            else:
                w = 1.0 - base.alpha if isinstance(base, BlendedEvaluator) else 1.0
                assert abs(got[k] - (plain[k] + w * (new - softmax(V, T)[p]))) < 1e-12
    hub = H.CorrectionHub(HeuristicEvaluator(), [H.ConstantProvider(C)])
    assert list(hub.evaluate([over, over], [2, 0])) == [1.0, 0.0]


def test_provider_order_sum_and_prior_chain():
    ls, me = leaves(n=10)
    p1 = H.ConstantProvider([0.1, 0.0, 0.0, 0.0], priors=lambda s, legal, pr: [2.0 * x for x in pr])
    p2 = H.ConstantProvider([1e16, 0.0, 0.0, 0.0])
    p3 = H.ConstantProvider([-1e16, 0.0, 0.0, 0.0], priors=lambda s, legal, pr: [x + 1.0 for x in pr])
    s = ls[0]
    assert H.CorrectionHub(HeuristicEvaluator(), [p1, p2, p3]).corrections(s)[0] == (0.1 + 1e16) - 1e16
    assert H.CorrectionHub(HeuristicEvaluator(), [p2, p3, p1]).corrections(s)[0] == 0.1   # float sum: list order
    assert H.CorrectionHub(HeuristicEvaluator(), [p1, p2, p3]).adjust_priors(s, [], [1.0, 3.0]) == [3.0, 7.0]
    assert H.CorrectionHub(HeuristicEvaluator(), [p3, p2, p1]).adjust_priors(s, [], [1.0, 3.0]) == [4.0, 8.0]
    assert H.CorrectionHub(HeuristicEvaluator(), [p2]).adjust_priors(s, [], [1.0]) == [1.0]   # no hook: unchanged


def test_paths_context_as_a_provider_matches_paths_evaluator():
    from tests.test_winpaths import extend_to, played
    s = played(seed=9, turns=70)
    extend_to(s, 1, 6)
    s.longest_road_owner, s.longest_road_len = 1, E.longest_road_length(s, 1)
    s.players[2].played_knights = 3
    s.largest_army_owner = 2
    ls, _ = leaves(seed=6, n=60)
    players = [i % 4 for i in range(len(ls))]
    for base in (HeuristicEvaluator(), BlendedEvaluator(StubNet(), 0.5), StubNet()):
        pe = W.PathsEvaluator(base, W.PathsContext(s, 0, spots=True))
        hub = H.CorrectionHub(base, [W.PathsContext(s, 0, spots=True)])
        a, b = pe.evaluate(ls, players), hub.evaluate(ls, players)
        assert np.max(np.abs(a - b)) < 1e-12
        if isinstance(base, HeuristicEvaluator):
            corrected = [k for k, st in enumerate(ls) if hub.corrections(st) is not None]
            assert corrected and all(a[k] == b[k] for k in corrected)   # same arithmetic on corrected leaves


# ---------------------------------------------------------------------------
# winpaths and the default bot: bit-identical before and after the refactor
# ---------------------------------------------------------------------------
# sha256[:16] digests produced by the pre-hub code (search.py before step 2; python3 and venv_cat33 agree).
PINNED_GAMES = {
    "default": (DEFAULT, ["bb4d5a60408db227", "7d579e7613b8fb57", "d4a7f5ace38ef8fe"]),
    "paths1": (DEFAULT + ",paths=1", ["3336b3d2adb9cea4", "6a3b3ac203006199", "c48533e8b7f1419f"]),
}
PINNED_MIXED = "5ab121e8a200c22d"      # seed 21, 60 turns: paths=1 / default / paths=1 spots w1.5 / default
PINNED_SEARCH = {
    "paths0": (SearchConfig(depth=1, beam=4, expand=8), "c7737a77f0898bcd"),
    "paths1": (SearchConfig(depth=1, beam=4, expand=8, paths=1), "ba04855873bd1633"),
    "paths1_spots": (SearchConfig(depth=1, beam=4, expand=8, paths=1, paths_spots=1, paths_priors=0), "70dbb3bf2391c6d9"),
    "paths1_d2": (SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500, paths=1),
                  "3b8cc56ce67bba97"),
}


@pytest.mark.parametrize("name", sorted(PINNED_GAMES))
def test_pinned_games_default_and_paths(name):
    spec, want = PINNED_GAMES[name]
    assert [game_hash([spec] * 4, seed, 36) for seed in (11, 12, 13)] == want
    if name == "default":      # the new switch spelled out at its default plays the same games
        assert [game_hash([spec + ",conv=0"] * 4, seed, 36) for seed in (11, 12)] == want[:2]


def test_pinned_mixed_paths_game():
    sp = DEFAULT
    assert game_hash([sp + ",paths=1", sp, sp + ",paths=1,paths_spots=1,paths_w=1.5", sp], 21, 60) == PINNED_MIXED


def test_pinned_searches_paths_zero_and_one():
    pos = positions()
    for name, (cfg, want) in PINNED_SEARCH.items():
        assert search_hash(cfg, pos) == want, name


def test_paths_alone_keeps_the_paths_evaluator():
    s, me = positions(k=1)[0]
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, paths=1))
    sr.search(s, me, random.Random(1))
    assert isinstance(sr._paths, W.PathsEvaluator) and sr._corr is None and sr._value_ev() is sr._paths
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8))
    sr.search(s, me, random.Random(1))
    assert sr._paths is None and sr._corr is None and sr._value_ev() is sr.evaluator


def test_zero_weight_hub_search_is_an_exact_aa():
    """conv=1 with KAPPA_CONV = 0: the hub is built but every correction is zero, so every search result is the
    default's bit for bit (fast path; the conversion context adds no prior hook)."""
    pos = positions(k=12)
    base = search_hash(SearchConfig(depth=1, beam=4, expand=8), pos)
    with tuning.overridden({"conversion.KAPPA_CONV": 0.0}):
        assert search_hash(SearchConfig(depth=1, beam=4, expand=8, conv=1), pos) == base


# ---------------------------------------------------------------------------
# search integration
# ---------------------------------------------------------------------------
def test_hub_off_is_never_built(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("CorrectionHub.for_search called with every provider off")

    monkeypatch.setattr(H.CorrectionHub, "for_search", classmethod(boom))
    s, me = positions(k=1)[0]
    for cfg in (SearchConfig(depth=1, beam=4, expand=8), SearchConfig(depth=1, beam=4, expand=8, paths=1),
                SearchConfig(depth=2, beam=2, expand=4, max_nodes=2000)):
        assert Searcher(HeuristicEvaluator(), cfg).search(s, me, random.Random(1))
    play_game([make_bot(DEFAULT), HeuristicBot(), make_bot(DEFAULT + ",paths=1"), HeuristicBot()],
              rng=random.Random(2), seed=2, max_turns=20)


def test_hub_providers_and_prior_hooks(monkeypatch):
    from catanbot.conversion import ConversionContext
    s, me = positions(k=1)[0]
    for paths, pri in ((0, 1), (1, 1), (1, 0)):
        cfg = SearchConfig(depth=1, beam=4, expand=8, conv=1, paths=paths, paths_priors=pri)
        sr = Searcher(HeuristicEvaluator(), cfg)
        calls = []
        orig = W.PathsContext.adjust_priors

        def spy(self, state, legal, priors):
            calls.append(1)
            return orig(self, state, legal, priors)

        monkeypatch.setattr(W.PathsContext, "adjust_priors", spy)
        sr.search(s, me, random.Random(1))
        kinds = [type(p) for p in sr._corr.providers]
        assert kinds == ([W.PathsContext] if paths else []) + [ConversionContext] and sr._paths is None
        assert bool(calls) == bool(paths and pri)
        monkeypatch.undo()


def test_all_evaluator_sites_use_the_hub(monkeypatch):
    from catanbot import search as S
    got = []

    def capture(state, me, evaluator=None, politics=None, model=None, **kw):
        got.append(evaluator)
        return []

    monkeypatch.setattr(S, "political_trade_options", capture)
    from tests.test_winpaths import played
    s = played(seed=6)
    s.players[0].resources = [2, 1, 2, 1, 0]
    assert any(a[0] == A.PROPOSE_TRADE for a in E.legal_actions(s))
    seen = []
    orig_eval = H.CorrectionHub.evaluate

    def spy_eval(self, states, players):
        seen.append(len(states))
        return orig_eval(self, states, players)

    monkeypatch.setattr(H.CorrectionHub, "evaluate", spy_eval)
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=2, expand=4, conv=1))
    sr.search(s, 0, random.Random(1))
    assert got and all(isinstance(e, H.CorrectionHub) for e in got) and seen      # _eval + political options
    # counter-offers: rank_counters gets the hub
    from catanbot import counteroffers
    cap = []
    monkeypatch.setattr(counteroffers, "rank_counters", lambda *a, **k: cap.append(k.get("evaluator")) or [])
    sr._corr = H.CorrectionHub(sr.evaluator, [])
    sr.config.counters = 1
    sr._counter_filter(s, [(A.COUNTER_TRADE, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0))], 0)
    assert cap == [sr._corr]
    # native lookahead: never with the hub (C++ cannot see the corrections)
    monkeypatch.setattr(S.Searcher, "_native_future_values", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500, conv=1)
    assert Searcher(HeuristicEvaluator(), cfg).search(s, 0, random.Random(1))
    assert reduced_config(cfg, 2, 600).conv == 1 and reduced_config(SearchConfig(), 2, 600).conv == 0


def test_setup_roots_build_no_hub():
    s = new_game(4, rng=random.Random(5))
    sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, conv=1, paths=1))
    assert sr.search(s, 0, random.Random(1)) and sr._corr is None and sr._paths is None


# ---------------------------------------------------------------------------
# leaf-chance hook
# ---------------------------------------------------------------------------
class GiftChance:
    """Toy chance provider: with probability q the finished leaf's next player loses 2 VP-equivalents (a copy
    with one of their settlements removed); the rest keeps the leaf itself."""

    def __init__(self, q=0.4, fire=True):
        self.q = q
        self.fire = fire
        self.calls = 0

    def leaf_outcomes(self, s, me):
        self.calls += 1
        if not self.fire:
            return None
        j = (me + 1) % s.num_players
        if not s.players[j].settlements:
            return None
        s2 = s.copy()
        s2.players[j].settlements = s2.players[j].settlements[1:]
        return [(1.0 - self.q, s), (self.q, s2)]


def test_leaf_chance_mixture_values():
    ls, me = leaves(n=30)
    fin = [k % 2 == 0 for k in range(len(ls))]
    hub = H.CorrectionHub(HeuristicEvaluator(), [], chance=[GiftChance(0.0, fire=False), GiftChance(0.4)])
    base = list(HeuristicEvaluator().evaluate(ls, [me] * len(ls)))
    out = hub.leaf_chance(ls, fin, me, base)
    changed = 0
    for k, s in enumerate(ls):
        outs = GiftChance(0.4).leaf_outcomes(s, me)
        if not fin[k] or s.phase in (PHASE_GAME_OVER, PHASE_SETUP_SETTLEMENT) or not outs:
            assert out[k] == base[k]
            continue
        v2 = float(HeuristicEvaluator().evaluate([outs[1][1]], [me])[0])
        assert abs(out[k] - (0.6 * base[k] + 0.4 * v2)) < 1e-12 and out[k] > base[k]
        changed += 1
    assert changed >= 5 and hub.stats["chance_leaves"] == changed
    assert hub.chance[0].calls >= changed      # asked first, answered None: the second provider decides


def test_leaf_chance_in_the_search_depth_one_only(monkeypatch):
    s, me = positions(k=1)[0]
    orig = H.CorrectionHub.for_search.__func__
    chance = GiftChance(0.5)

    def with_chance(cls, base, root, me_, cfg):
        hub = orig(cls, base, root, me_, cfg)
        hub.chance.append(chance)
        return hub

    monkeypatch.setattr(H.CorrectionHub, "for_search", classmethod(with_chance))
    with tuning.overridden({"conversion.KAPPA_CONV": 0.0}):
        sr = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8, conv=1))
        res = sr.search(s, me, random.Random(1))
        assert res and chance.calls > 0 and sr._corr.stats["chance_leaves"] > 0
        base = Searcher(HeuristicEvaluator(), SearchConfig(depth=1, beam=4, expand=8)).search(s, me, random.Random(1))
        assert [r.value for r in res] != [r.value for r in base]
        chance.calls = 0
        Searcher(HeuristicEvaluator(), SearchConfig(depth=2, beam=2, expand=4, opp_roll_samples=2, max_nodes=1500,
                                                    conv=1)).search(s, me, random.Random(1))
        assert chance.calls == 0


# ---------------------------------------------------------------------------
# shared.seat_events: per-seat observers in play_paired_game
# ---------------------------------------------------------------------------
class CountKinds:
    def __init__(self, n):
        self.n = [0] * n

    def on_action(self, state, action, player):
        if action[0] == A.END_TURN:
            self.n[player] += 1

    def result(self):
        return list(self.n)


def test_seat_observers_in_paired_games(monkeypatch):
    monkeypatch.setitem(tuning.SEAT_OBSERVERS, "ends", CountKinds)
    plain = tuning.play_paired_game("heuristic:temp=0.15", {}, "CDDC", 5, max_turns=30)
    seen = tuning.play_paired_game("heuristic:temp=0.15", {}, "CDDC", 5, max_turns=30, observers=["ends", "economy"])
    assert "seat_events" not in plain
    ev = seen.pop("seat_events")
    for k in ("winner", "vps", "turns", "actions", "seat_decisions", "seat_counters"):
        assert seen[k] == plain[k]                  # observers never touch the game
    assert sum(ev["ends"]) > 0 and len(ev["ends"]) == 4
    eco = ev["economy"]
    assert set(eco) >= {"bank_4", "bank_3", "bank_2", "settlements", "cities", "first_settlement_turn",
                        "first_city_turn", "types_after_setup"}
    assert all(1 <= t <= 5 for t in eco["types_after_setup"])
    both = tuning.play_paired_game("heuristic:temp=0.15", {}, "CDDC", 5, max_turns=30, allow_counters=True,
                                   observers=["ends"])
    assert "seat_counters" in both and both["seat_events"]["ends"]
    st = tuning.paired_stats([dict(seen, seat_events=ev), plain])
    assert st["records"][0]["seat_events"] == ev and "seat_events" not in st["records"][1]
    with pytest.raises(KeyError):
        tuning.play_paired_game("heuristic:temp=0.15", {}, "CDDC", 5, max_turns=5, observers=["nope"])
