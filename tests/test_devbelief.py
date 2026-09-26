"""Held-age reading of opponents' development cards (catanbot/devbelief.py; docs/PRIORITY_PLAN.md step 6):
the bookkeeping, the per-type-present likelihood, the exact DP against brute force, the history-aware
win caps, RNG-neutral dealing in the trackers, the counted adapter's hooks and the advisor reading."""
from __future__ import annotations

import itertools
import json
import math
import random

import pytest

from catanbot import board as B
from catanbot import devbelief as DB
from catanbot import tuning

K, VP, RB, YOP, MONO = B.DEV_KNIGHT, B.DEV_VP, B.DEV_ROAD_BUILDING, B.DEV_YEAR_OF_PLENTY, B.DEV_MONOPOLY


@pytest.fixture(autouse=True)
def _defaults():
    saved = (DB.ENABLED, DB.H, DB.H_KNIGHT_CTX, DB.EPS)
    yield
    DB.ENABLED, DB.H, DB.H_KNIGHT_CTX, DB.EPS = saved


def model_with(history, n=2, same_turn=False):
    """``history``: per player a list of own turns, each a list of events ('buy' | ('play', t) | ('pub', v)
    | 'ctx'); every listed turn is completed."""
    m = DB.DevAgeModel(n, eligible_same_turn=same_turn)
    for j, turns in enumerate(history):
        for turn in turns:
            m.on_turn_start(j, la="ctx" in turn)
            for ev in turn:
                if ev == "buy":
                    m.on_buy(j)
                elif isinstance(ev, tuple) and ev[0] == "play":
                    m.on_play(j, ev[1])
                elif isinstance(ev, tuple) and ev[0] == "pub":
                    m.on_public(j, ev[1])
            m.on_turn_end(j)
    return m


def brute_posterior(m, pool, holders, public, V, pr):
    """Exact joint posterior by enumerating every type assignment of every holder's records."""
    eps = pr.eps()
    recs = {j: list(m.recs[j]) for j in holders}
    hist = {j: m._hist_caps(j, V) for j in holders}
    out = {}
    total = 0.0
    slots = [(j, i) for j in holders for i in range(len(recs[j]))]
    for assign in itertools.product(range(5), repeat=len(slots)):
        N = [0] * 5
        for t in assign:
            N[t] += 1
        w = 1.0
        for t in range(5):
            w *= DB._fall(pool[t], N[t])
        if not w:
            continue
        per = {j: [] for j in holders}
        for (j, i), t in zip(slots, assign):
            per[j].append(t)
        ok = True
        for j in holders:
            types = per[j]
            for t in DB._PLAYABLE:
                if t in types:
                    b = recs[j][types.index(t)]            # the oldest record assigned t
                    prod = 1.0
                    for tau in m.aging_set(j, b, t):
                        prod *= 1.0 - pr.hazard(t, m.turns[j][tau])
                    w *= (1.0 - eps) * prod + eps
            nvp = 0
            for k, t in enumerate(types):
                nvp += t == VP
                if nvp > min(hist[j][k], V - 1 - public[j]):
                    ok = False
        if not ok or w <= 0:
            continue
        total += w
        key = tuple(tuple(per[j]) for j in holders)
        out[key] = out.get(key, 0.0) + w
    return {k: v / total for k, v in out.items()}


def summaries(joint, holders):
    res = {}
    for idx, j in enumerate(holders):
        m = len(next(iter(joint))[idx])
        p_vp = [0.0] * (m + 1)
        rec = [[0.0] * 5 for _ in range(m)]
        p_type = [0.0] * 5
        for key, p in joint.items():
            types = key[idx]
            p_vp[types.count(VP)] += p
            for i, t in enumerate(types):
                rec[i][t] += p
            for t in set(types):
                p_type[t] += p
        res[j] = {"p_vp": p_vp, "records": rec, "p_type": p_type}
    return res


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------
def test_per_type_present_likelihood_and_eps_floor():
    # two knights held through the same no-play turns cost what one does; the floor bounds F from below
    m = model_with([[["buy", "buy"], [], []], []])
    DB.H, DB.EPS = 0.85, 0.03
    f = m._factor(0, m.recs[0][0], K, DB.BOT, DB.EPS)
    assert f == pytest.approx(0.97 * 0.15 ** 2 + 0.03)
    post2 = m.posterior([14, 5, 2, 2, 2], [0], [2, 0], [2, 2], 10, DB.BOT)
    m1 = model_with([[["buy"], [], []], []])
    post1 = m1.posterior([14, 5, 2, 2, 2], [0], [1, 0], [2, 2], 10, DB.BOT)
    # P(KK) / P(K?) ratios follow the per-type factor once, not squared
    joint = brute_posterior(m, [14, 5, 2, 2, 2], [0], [2, 2], 10, DB.BOT)
    assert joint[((K, K),)] / joint[((VP, VP),)] == pytest.approx(14 * 13 / (5 * 4) * f)
    assert post1[0]["records"][0][K] < 14 / 25 and post2[0]["p_type"][K] > 0
    DB.H = 1.0
    assert m._factor(0, m.recs[0][0], K, DB.BOT, DB.EPS) == pytest.approx(0.03)


def test_one_card_per_turn_and_same_turn_eligibility():
    # a turn with a play ages nothing; a card bought this turn is not aged (3.3) / is aged (3.2.1)
    m = model_with([[["buy"], [("play", YOP)], []], []])
    assert m.recs[0] == [] and m.cov[0][YOP] == [(0, 1)]
    m = model_with([[["buy", "buy"], [("play", K)], [], []], []])
    assert m.recs[0] == [0] and m.aging_set(0, 0, RB) == [2, 3]
    assert m.aging_set(0, 0, K) == [2, 3]             # the played knight's interval (0, 1) is over
    m33 = model_with([[["buy"]], []])
    m321 = model_with([[["buy"]], []], same_turn=True)
    assert m33.aging_set(0, 0, K) == [] and m321.aging_set(0, 0, K) == [0]


def test_coverage_interval_does_not_age_that_type():
    m = model_with([[["buy"], ["buy"], [], [("play", K)], []], []])
    # the knight bought in turn 1 was played in turn 3 (youngest eligible): records keep turn 0's card
    assert m.recs[0] == [0] and m.cov[0][K] == [(1, 3)]
    assert m.aging_set(0, 0, K) == [4]               # turns 1-3 are explained by the held knight
    assert m.aging_set(0, 0, MONO) == [1, 2, 4]


# ---------------------------------------------------------------------------
# exact DP vs brute force, sampling
# ---------------------------------------------------------------------------
CASES = [
    # (history, pool, holders, public, V)
    ([[["buy"], ["buy"], ["buy"], [], []], []], [14, 5, 2, 2, 2], [0], [3, 3], 10),
    ([[["buy"], ["buy"], [], []], [["buy", "buy"], ["ctx"], []]], [12, 5, 2, 2, 2], [0, 1], [4, 6], 10),
    ([[["buy", "buy"], [], []], [["buy"], ["buy"], []]], [3, 1, 1, 1, 1], [0, 1], [2, 2], 10),   # few VP left
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("pr_name", ["bot", "human", "uniform"])
def test_dp_posterior_matches_brute_force(case, pr_name):
    hist, pool, holders, public, V = CASES[case]
    m = model_with(hist)
    DB.H_KNIGHT_CTX = 0.97
    pr = DB.preset(pr_name)
    counts = [len(m.recs[j]) for j in range(m.n)]
    post = m.posterior(pool, holders, counts, public, V, pr)
    ref = summaries(brute_posterior(m, pool, holders, public, V, pr), holders)
    for j in holders:
        assert close(post[j]["p_vp"], ref[j]["p_vp"])
        assert close(post[j]["p_type"], ref[j]["p_type"])
        for a, b in zip(post[j]["records"], ref[j]["records"]):
            assert close(a, b)


def test_sampler_matches_the_exact_joint():
    hist, pool, holders, public, V = CASES[1]
    m = model_with(hist)
    counts = [len(m.recs[j]) for j in range(m.n)]
    joint = brute_posterior(m, pool, holders, public, V, DB.BOT)
    rng = random.Random(3)
    freq = {}
    n = 20000
    for _ in range(n):
        d = m.sample(pool, holders, counts, public, V, rng)
        key = tuple(tuple(d[j]) for j in holders)
        freq[key] = freq.get(key, 0) + 1
    for key, p in joint.items():
        assert abs(freq.get(key, 0) / n - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-3, key


def test_history_aware_caps():
    V = 10
    # a card held while its owner stood at public 9 on its own turn is never a VP card
    m = model_with([[["buy"], [("pub", 9)], []], []])
    post = m.posterior([14, 5, 2, 2, 2], [0], [1, 0], [7, 0], V, DB.BOT)   # public dropped to 7 since
    assert post[0]["p_vp"][1] == 0.0
    rng = random.Random(1)
    assert all(m.sample([14, 5, 2, 2, 2], [0], [1, 0], [7, 0], V, rng)[0] != [VP] for _ in range(200))
    # a public-8 holder of two old cards holds at most one VP card
    m = model_with([[["buy"], ["buy"], [], []], []])
    post = m.posterior([0, 5, 0, 0, 1], [0], [2, 0], [8, 0], V, DB.BOT)
    assert post[0]["p_vp"][2] == 0.0 and post[0]["p_vp"][1] == pytest.approx(1.0)
    assert m.informative([0], V)                      # aged
    m = model_with([[["buy", ("pub", 9)]], []])
    assert m.aging_set(0, 0, K) == [] and m.informative([0], V)   # the cap binds without any aging


def test_conditioned_alert_numbers():
    # public 8, two long-held cards: E[hidden VP] <= 1 and P(at least 9) is high (critique 1's example)
    DB.EPS = 0.0                                      # two cards at P(VP) ~ 0.9+ each unconditioned
    m = model_with([[["buy"], ["buy"]] + [[] for _ in range(6)], []])
    post = m.posterior([14, 5, 2, 2, 2], [0], [2, 0], [8, 0], 10, DB.BOT)
    assert post[0]["e_vp"] <= 1.0 + 1e-12 and post[0]["p_vp"][1] >= 0.5   # the alert rule P(>= V - 1) >= 0.5


def test_budget_and_mismatch_fall_back(monkeypatch):
    m = model_with([[["buy"], []], []])
    assert m.sample([14, 5, 2, 2, 2], [0], [2, 0], [0, 0], 10, random.Random(0)) is None   # count mismatch
    assert m.stats["fallbacks"] == 1
    monkeypatch.setattr(DB, "TABLE_BUDGET", 1)
    assert m.sample([14, 5, 2, 2, 2], [0], [1, 0], [0, 0], 10, random.Random(0)) is None
    assert m.stats["fallbacks"] == 2 and m.stats["errors"] == 0


def test_bookkeeping_errors_are_isolated_and_the_model_round_trips():
    m = model_with([[["buy"], [("play", K)], ["buy"], []], [["buy"]]])
    m.log_events = True
    m.on_turn_start(1)
    back = DB.DevAgeModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert back.to_dict() == m.to_dict()
    assert m.on_play(7, K) is None and m.invalid and m.stats["errors"] == 1   # bad seat: flagged, not raised
    assert m.sample([14, 5, 2, 2, 2], [1], [0, 1], [0, 0], 10, random.Random(0)) is None


def test_registry_entries():
    assert tuning.verify_registry() == []
    for name, default in (("ENABLED", 0), ("H", 0.85), ("H_KNIGHT_CTX", 0.85), ("EPS", 0.03)):
        t = tuning.find(f"devbelief.{name}")
        assert t.default == default and t.kind == "weight" and not t.needs_python_evaluator


# ---------------------------------------------------------------------------
# trackers and the counted adapter (catanatron)
# ---------------------------------------------------------------------------
def _catanatron():
    pytest.importorskip("catanatron")
    from catanatron.players.weighted_random import WeightedRandomPlayer
    from catanbot.bench import catanatron_adapter as AD
    from catanbot.bench import public_info as PI
    return WeightedRandomPlayer, AD, PI


def weighted_game(seed):
    WR, AD, _PI = _catanatron()
    g = AD.make_game([WR(c) for c in AD.COLORS], seed=seed)
    g.play()
    return g


def _positions(seed, every=29):
    """(tracker, public view) at every ``every``-th log entry of a WeightedRandom game, seat 0 counting."""
    _WR, AD, PI = _catanatron()
    g = weighted_game(seed)
    tr = PI.PublicInfoTracker(AD.COLORS[0])
    for i, (entry, sh) in enumerate(tr.replay_log(g.state)):
        if i % every == every - 1:
            cb = AD.state_to_catanbot(sh, 10, AD.mapping_for(sh.board.map))
            yield tr, sh, cb, tr.public_view(cb)


def test_rng_neutral_dealing_and_default_identity():
    n_aged = n_diff = 0
    for tr, _sh, _cb, pub in _positions(5):
        dm = tr.dev_age
        for k in range(2):
            DB.ENABLED = 0
            r0 = random.Random(k)
            d0 = tr.determinize(pub, r0)
            tr.dev_age = None                    # without the bookkeeping at all: the same state (ENABLED=0)
            rn = random.Random(k)
            dn = tr.determinize(pub, rn)
            tr.dev_age = dm
            assert d0.to_dict() == dn.to_dict() and r0.getstate() == rn.getstate()
            DB.ENABLED = 1
            r1 = random.Random(k)
            d1 = tr.determinize(pub, r1)
            DB.ENABLED = 0
            assert r1.getstate() == r0.getstate()
            assert [p.resources for p in d1.players] == [p.resources for p in d0.players]
            informative = dm.informative(tr.unknown_dev_holders(), tr.vps_to_win)
            n_aged += informative
            if not informative:
                assert d1.to_dict() == d0.to_dict()
            elif d1.to_dict() != d0.to_dict():
                n_diff += 1
                for j in tr.unknown_dev_holders():
                    assert d1.players[j].total_dev == tr.dev_count[j]
                assert sum(d1.dev_deck) == sum(d0.dev_deck)
    assert n_aged > 0 and n_diff > 0
    assert dm.stats["errors"] == 0


def test_tracker_records_match_counts_and_block_values_match_robber_eval():
    from catanbot import robber_eval as RE
    _WR, AD, PI = _catanatron()
    for tr, sh, cb, _pub in _positions(7, every=11):
        for j in tr.unknown_dev_holders():
            assert len(tr.dev_age.recs[j]) == tr.dev_count[j]
        pv = PI.robber_block_values(sh, tr.color_to_index, tr._scarcity)
        assert close(pv, RE.block_values(cb, cb.robber))
        assert tr.dev_age.stats["errors"] == 0


def test_reveal_devs_oracle_only_reveals_development_draws():
    _WR, AD, PI = _catanatron()
    g = weighted_game(9)
    tr = PI.PublicInfoTracker(AD.COLORS[0], reveal_devs=True)
    plain = PI.PublicInfoTracker(AD.COLORS[0])
    steps = list(plain.replay_log(g.state))
    for (entry, sh), _ in zip(tr.replay_log(g.state), steps):
        ps = sh.player_state
        for j in range(1, 4):
            true = [int(ps[f"P{j}_{d}_IN_HAND"]) for d in AD.CB_TO_DEV]
            assert tr.known_dev[j] == true
    assert tr.unknown_dev_holders() == [] and tr.stats["hidden_dev_draws"] == 0
    assert tr.stats["hidden_steals"] == plain.stats["hidden_steals"] > 0
    assert tr.stats["hidden_discards"] == plain.stats["hidden_discards"]
    assert PI.PublicInfoTracker(AD.COLORS[0]).reveal_devs is False


def test_paramBot_scope_applies_only_devbelief_overrides():
    from catanbot.agents.param_bot import ParamBot
    from catanbot.selfplay import make_bot
    pb = ParamBot(make_bot("heuristic"), {"devbelief.H": 0.95, "danger.TURNS_HALF": 2.0})
    from catanbot import danger
    th = danger.TURNS_HALF
    with pb.scope(prefixes=("devbelief.",)):
        assert DB.H == 0.95 and danger.TURNS_HALF == th
    assert DB.H == 0.85
    with ParamBot(make_bot("heuristic"), {}).scope(prefixes=("devbelief.",)):
        assert DB.H == 0.85


def test_candidate_hazard_deals_differently_from_the_default():
    # same rng and history: H=0.95 and H=0.85 deal differently somewhere; ENABLED=0 overrides nothing
    seen = 0
    for tr, _sh, _cb, pub in _positions(5):
        if not tr.dev_age.informative(tr.unknown_dev_holders(), 10):
            continue
        outs = []
        for h in (0.85, 0.2):
            tok = tuning.apply({"devbelief.ENABLED": 1, "devbelief.H": h})
            try:
                outs.append([tuple(p.dev_cards) for p in tr.determinize(pub, random.Random(4)).players])
            finally:
                tuning.restore(tok)
        seen += outs[0] != outs[1]
    assert seen > 0


def _counted_game(seed, monkeypatch=None, raise_model=False, **kw):
    """One short counted game of our heuristic seat vs WeightedRandom seats: (action log items, stats)."""
    WR, AD, PI = _catanatron()
    from catanbot.agents.param_bot import ParamBot
    from catanbot.selfplay import make_bot
    over = kw.pop("overrides", {})
    me = AD.CatanbotPlayer(AD.COLORS[1], spec="heuristic", bot=ParamBot(make_bot("heuristic"), over),
                           info="counted", seed=seed, **kw)
    if raise_model:
        def boom(*a, **k):
            raise RuntimeError("boom")
        for name in ("_informative", "_tables", "aging_set"):
            monkeypatch.setattr(DB.DevAgeModel, name, boom)
        for name in ("on_turn_start", "on_buy", "on_play", "on_turn_end", "on_public"):
            monkeypatch.setattr(DB.DevAgeModel, name, DB._guarded(boom))
    players = [me if i == 1 else WR(c) for i, c in enumerate(AD.COLORS)]
    g = AD.make_game(players, seed=seed)
    g.play()
    me.log = [repr(AD.log_action(e)) for e in AD.action_log(g.state)]
    return me


def _log_of(me):
    return me.log


def test_raising_model_leaves_counted_games_unchanged(monkeypatch):
    base = _counted_game(21)
    bad = _counted_game(21, monkeypatch, raise_model=True)
    assert _log_of(bad) == _log_of(base)
    assert bad.stats["info_errors"] == base.stats["info_errors"]
    assert bad.stats["devbelief_errors"] > 0 and base.stats["devbelief_errors"] == 0
    # with the flag on the broken model falls back to the uniform deal (the same game) and counts it
    on = _counted_game(21, monkeypatch, raise_model=True, overrides={"devbelief.ENABLED": 1})
    assert _log_of(on) == _log_of(base) and on.stats["devbelief_errors"] > 0


def test_seeded_samples_and_knight_hint(monkeypatch):
    from catanbot import robber_eval as RE
    hints = []
    real = RE.knight_hint

    def spy(dealt=(), p_knight=None):
        hints.append((tuple(dealt), dict(p_knight or {})))
        return real(dealt=dealt, p_knight=p_knight)
    monkeypatch.setattr(RE, "knight_hint", spy)
    a = _counted_game(23, seeded_samples=True)
    b = _counted_game(23, seeded_samples=True)
    assert _log_of(a) == _log_of(b) and hints
    assert all(1 not in d for d, _ in hints) and all(p == {} for _, p in hints)   # ENABLED=0: dealt knights absent
    hints.clear()
    _counted_game(23, overrides={"devbelief.ENABLED": 1})
    assert any(p for _, p in hints) and all(0.0 <= v <= 1.0 for _, p in hints for v in p.values())
    from catanbot.bench import catanatron_adapter as AD
    r1, r2 = a._sample_rng(3, 1), a._sample_rng(3, 1)
    assert r1.random() == r2.random() and a._sample_rng(3, 2).random() != a._sample_rng(3, 1).random()
    assert AD.CatanbotPlayer(AD.COLORS[0], spec="heuristic").seeded_samples is False


# ---------------------------------------------------------------------------
# the advisor (Colonist log)
# ---------------------------------------------------------------------------
def _colonist_tracker():
    from catanbot.colonist_log import ColonistLogTracker
    return ColonistLogTracker(["red", "blue", "orange"], ["Alice", "Bob", "Carol"], 0)


def ev(kind, player, text="", item=None):
    from catanbot.colonist_log import LogEvent
    return LogEvent(kind=kind, player=player, item=item, text=text)


def test_colonist_pre_roll_knight_starts_the_turn():
    tr = _colonist_tracker()
    dm = tr.dev_age
    for e in (ev("roll", "Bob"), ev("buy_dev", "Bob"), ev("buy_dev", "Bob"), ev("turn", "Bob", "Bob ended their turn"),
              ev("roll", "Carol"), ev("turn", "Carol", "Carol ended their turn"),
              ev("roll", "Bob"), ev("turn", "Bob", "Bob ended their turn"),          # a no-play turn: ages
              ev("roll", "Carol"),                                                  # Carol's end line is missing
              ev("play_dev", "Bob", item="knight"), ev("roll", "Bob"), ev("turn", "Bob", "Bob ended their turn")):
        tr._dev_age_event(ev_kind(e), tr.seat_of(e.player), e)
    assert not any(dm.open) and dm.recs[1] == [0] and len(dm.turns[1]) == 3 and len(dm.turns[2]) == 2
    assert dm.turns[1][2][0] == 0                   # the knight turn ages nothing
    assert dm.aging_set(1, 0, MONO) == [1]


def ev_kind(e):
    return e.kind


def test_trackers_build_the_same_statistics():
    # PublicInfoTracker and ColonistLogTracker: the same buys / plays / turns give the same records
    col = _colonist_tracker()
    pi = DB.DevAgeModel(3)
    seq = [("roll", 1), ("buy_dev", 1), ("end", 1), ("roll", 2), ("end", 2), ("roll", 1), ("end", 1),
           ("knight", 1), ("roll", 1), ("buy_dev", 1), ("end", 1)]
    names = ["Alice", "Bob", "Carol"]
    for kind, p in seq:
        e = (ev("turn", names[p], f"{names[p]} ended their turn") if kind == "end" else
             ev("play_dev", names[p], item="knight") if kind == "knight" else ev(kind, names[p]))
        col._dev_age_event(e.kind, p, e)
        if kind in ("roll", "knight") and not pi.is_open(p):
            pi.on_turn_start(p)
        if kind == "buy_dev":
            pi.on_buy(p)
        elif kind == "knight":
            pi.on_play(p, K)
        elif kind == "end":
            pi.on_turn_end(p)
    a, b = col.dev_age.to_dict(), pi.to_dict()
    for key in ("recs", "turns", "cov", "open"):
        assert a[key] == b[key], key


def test_session_round_trip_and_old_sessions():
    from catanbot.colonist_log import ColonistLogTracker
    tr = _colonist_tracker()
    tr._dev_age_event("roll", 1, ev("roll", "Bob"))
    tr._dev_age_event("buy_dev", 1, ev("buy_dev", "Bob"))
    tr.dev_age.log_events = True
    tr._dev_age_event("turn", 1, ev("turn", "Bob", "Bob ended their turn"))
    d = json.loads(json.dumps(tr.to_dict()))
    assert ColonistLogTracker.from_dict(d).to_dict() == tr.to_dict()
    d.pop("dev_age")                                  # a session written before step 6
    old = ColonistLogTracker.from_dict(d)
    assert old.dev_age.recs == [[], [], []] and not old.warnings
    old.dev_age.sync_counts([0, 2, 0], skip=[0])
    assert old.dev_age.recs[1] == [0, 0] and old.dev_age.informative([1], 10) is False


def test_monopoly_forecast_rises_with_the_payoff_and_old_cards_lose_monopoly_mass():
    assert DB.monopoly_hazard(DB.HUMAN, 2) < DB.monopoly_hazard(DB.HUMAN, 6) < DB.monopoly_hazard(DB.HUMAN, 9)
    m = DB.DevAgeModel(2)
    m.on_turn_start(0)
    m.on_buy(0)
    m.on_turn_end(0)
    m.on_turn_start(0, pay=9.0)                      # held through a high-payoff turn
    m.on_turn_end(0)
    low = DB.DevAgeModel.from_dict(m.to_dict())
    low.turns[0][1][3] = 1.0                         # the same turn at a small payoff
    low._touch()
    hi = m.posterior([14, 5, 2, 2, 2], [0], [1, 0], [0, 0], 10, DB.HUMAN)[0]["records"][0][MONO]
    lo = low.posterior([14, 5, 2, 2, 2], [0], [1, 0], [0, 0], 10, DB.HUMAN)[0]["records"][0][MONO]
    assert hi < lo


def test_advisor_reading_is_off_by_default_and_reads_a_secret_leader():
    from catanbot.colonist_log import ColonistLogTracker
    from catanbot.counting import CardCounter
    tr = _colonist_tracker()
    tr.counter = CardCounter([[1, 0, 0, 0, 0], [2, 1, 0, 0, 0], [0, 0, 3, 0, 0]])
    tr.origin = "start"
    tr.dev_count = [0, 2, 0]
    tr.public_vp_seen = [3, 8, 4]
    for e in [ev("roll", "Bob"), ev("buy_dev", "Bob"), ev("turn", "Bob", "Bob ended their turn")] + \
            [ev("roll", "Bob"), ev("buy_dev", "Bob"), ev("turn", "Bob", "Bob ended their turn")] + \
            [x for _ in range(5) for x in (ev("roll", "Bob"), ev("turn", "Bob", "Bob ended their turn"))]:
        tr._dev_age_event(e.kind, tr.seat_of(e.player), e)
    base = tr.report()
    assert tr.report(dev_model="uniform") == base and "dev" not in json.dumps(base["players"])
    rep = tr.report(dev_model="bot")
    dev = rep["players"]["blue"]["dev"]
    assert dev["e_vp"] <= 1.0 + 1e-9 and dev["alert"] and len(dev["cards"]) == 2
    assert rep["lines"][:2] == base["lines"][:2] and any("SECRET LEADER" in x for x in rep["lines"])
    assert "unfitted" in " ".join(tr.report(dev_model="human")["lines"])
    assert isinstance(ColonistLogTracker.from_dict(json.loads(json.dumps(tr.to_dict()))), ColonistLogTracker)


# ---------------------------------------------------------------------------
# scripts: gate (A) calibration and the dev-oracle shadow
# ---------------------------------------------------------------------------
def _script(name):
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibration_gate_smoke_on_a_fixture_log(tmp_path):
    WR, AD, _PI = _catanatron()
    path = tmp_path / "games.jsonl"
    with open(path, "w") as fh:
        for s in (1, 2, 3):
            g = AD.make_game([WR(c) for c in AD.COLORS], seed=s)
            header = AD.game_log_header(g)
            g.play()
            doc = AD.game_log(g, header)
            doc.update({"game": s, "our_seats": [0], "players": [{"seat": 0, "kind": "catanbot"}] +
                        [{"seat": i, "kind": "opponent", "preset": "random"} for i in range(1, 4)]})
            fh.write(json.dumps(doc) + "\n")
    cal = _script("dev_belief_calibrate")
    out = tmp_path / "gate.json"
    assert cal.main(["--logs", str(path), "--json", str(out), "--min-n", "10"]) == 0
    res = json.loads(out.read_text())
    assert res["games"] == 3 and res["observations"] > 20 and res["errors"] == 0 and res["fallbacks"] == 0
    c = res["classes"]["random"]
    assert c["hazard"]["mse"] < c["uniform"]["mse"] and set(c["buckets"]) <= {"1", "2", "3-5", "6-9", "10+"}
    assert res["verdict"] in ("PASS", "FAIL") and res["pass"] == (not res["fails"])


def test_shadow_regret_statistics_and_aggregation():
    bs = _script("belief_shadow")
    rows = [{"evaluated": True, "sub": True, "regret_u": 0.02, "regret_h": 0.0, "regret_o": 0.0},
            {"evaluated": True, "sub": False, "regret_u": 0.04, "regret_h": 0.01, "regret_o": 0.0},
            {"sub": False}, {"sub": False}]
    st = bs.regret_stats(rows, per_game=50)
    assert st["captured"] == pytest.approx((0.02 + 0.03) / 4) and st["headroom"] == pytest.approx(0.02)
    assert st["implied_pp"] == pytest.approx(100 * 0.0125 * 50) and st["oracle_sane"]
    # the counted adapter's rule: actions ranked by at least half the samples first, then the mean value
    res = [("a", [("a", 0.5), ("b", 0.9)]), ("a", [("a", 0.5)]), ("c", [("c", 0.99)])]
    assert bs.aggregate(res, 3) == "a"


def test_forced_root_evaluates_any_listed_action():
    bs = _script("belief_shadow")
    from catanbot import engine as E
    from catanbot.agents.heuristic_bot import HeuristicBot
    from catanbot.state import new_game
    from catanbot.selfplay import make_bot
    bot = make_bot("search:depth=1,beam=2,expand=4,evaluator=heuristic")
    rng = random.Random(1)
    s = new_game(4, rng=rng)
    h = HeuristicBot()
    k = 0
    while not (s.turn > 8 and len(E.legal_actions(s)) >= 3) and k < 800:
        s = E.apply(s, h.decide(s, E.legal_actions(s), rng), rng)
        k += 1
    legal = E.legal_actions(s)
    assert len(legal) >= 3
    pick = [legal[-1], legal[0]]
    with bs.forced_root(s, pick):
        bot.decide(s, list(pick), random.Random(2))
    assert {r.action for r in bot.last_results} == set(pick)
    assert E.legal_actions(s) == legal                   # restored


def test_shadow_live_smoke():
    _catanatron()
    bs = _script("belief_shadow")
    rows = []
    shadow = bs.Shadow(2, 2, 1.0, rows, 4)

    class Args:
        games, seed, opponents, spec, max_positions, discards_public = 1, 3, "random", None, 4, True
    from catanbot.bench import catanatron_adapter as AD
    Args.spec = "search:depth=1,beam=2,expand=4,evaluator=heuristic"
    src = bs.run_live(Args, shadow)
    out = bs.summarize(rows, src)
    assert out["positions"] == 4 and out["aa_ok"] and out["crn_ok"] and src["games"] == 1
    assert all(r.get("evaluated") for r in rows)          # --sub 1: every position scored
    assert AD.CatanbotPlayer(AD.COLORS[0], spec="heuristic", info="counted").reveal_devs is False
