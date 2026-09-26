"""Tests for the tunable registry (catanbot/tuning.py), ParamBot and scripts/ablate.py."""
import importlib.util
import json
import os
import random
import subprocess
import sys

import pytest

from catanbot import coalitions, danger, heuristic, opponent_model, placement, politics, robber, trading, tuning
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.param_bot import ParamBot
from catanbot.selfplay import make_bot, play_game
from catanbot.state import new_game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "ablate.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("ablate_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_cpp_static_value_constants_are_flagged():
    """Every tunable that changes ``heuristic.static_value`` must be flagged ``needs_python_evaluator``: the C++
    port (cpp/heuristic.cpp) hard-codes its constants, so with the extension loaded an unflagged override would
    silently reach only the Python-side code (the C++ static value keeps the default)."""
    from catanbot import accel, engine as E
    if not accel.AVAILABLE:
        pytest.skip("C++ extension not built / disabled")
    rng = random.Random(5)
    states = []
    for g in range(3):
        s = new_game(4, rng=random.Random(200 + g))
        bots = [HeuristicBot(temperature=0.3) for _ in range(4)]
        k = 0
        while not E.is_terminal(s) and k < 500 and len(states) < 8 * (g + 1):
            a = bots[E.acting_player(s)].decide(s, E.legal_actions(s), rng)
            if s.turn > 15 and k % 23 == 0:
                states.append(s.copy() if hasattr(s, "copy") else __import__("copy").deepcopy(s))
            s = E.apply(s, a, rng)
            k += 1
    assert states
    for s in states:     # the port is bit-identical at the defaults
        for p in range(4):
            assert abs(heuristic.static_value(s, p) - accel.static_value(s, p)) <= 1e-9
    unflagged = []
    for name, t in tuning.TUNABLES.items():
        if t.kind == "search" or t.needs_python_evaluator:
            continue
        for v in t.candidates:
            with tuning.overridden({name: v}):
                if any(abs(heuristic.static_value(s, p) - accel.static_value(s, p)) > 1e-9
                       for s in states for p in range(4)):
                    unflagged.append(f"{name}={t.format(v)}")
                    break
    assert unflagged == [], f"static_value reads these but the C++ port ignores them: {unflagged}"


def test_registry_resolves_to_real_attributes():
    assert tuning.verify_registry() == []
    required = {"danger.TURNS_HALF", "danger.BLOCK_FLOOR", "danger.BLOCK_NEED", "danger.danger_multiplier",
                "heuristic.EXPOSURE_WEIGHT", "coalitions.SCALE", "coalitions.BLOC_THRESHOLD", "politics.MAX_SLACK",
                "opponent_model.stage_late_drop", "trading.accept_margin", "search.trade_proposals",
                "search.dump_candidates", "search.opponent_proposals", "search.opponent_actions",
                "search.opp_roll_samples", "search.beam", "search.expand", "search.depth"}
    assert required <= set(tuning.TUNABLES)
    for t in tuning.TUNABLES.values():
        assert t.kind in ("weight", "flag", "search") and t.candidates and t.description
        assert all(c != t.default for c in t.candidates), t.name
        if t.kind != "search":
            for tg in t.targets:
                tg.get()   # resolves
    assert tuning.TUNABLES["danger.TURNS_HALF"].default == danger.TURNS_HALF
    assert tuning.TUNABLES["heuristic.EXPOSURE_WEIGHT"].needs_python_evaluator
    assert tuning.TUNABLES["heuristic.EXPOSURE_WEIGHT"].default == 0.25
    for name in ("PLACEMENT_BLOCK_WEIGHT", "PLACEMENT_ROBBER_Q"):
        assert (f"placement.{name}" in tuning.TUNABLES) == hasattr(placement, name)
        if hasattr(placement, name):   # static_value reads it; cpp/heuristic.cpp has a constexpr copy
            assert tuning.TUNABLES[f"placement.{name}"].needs_python_evaluator
    assert tuning.find("TURNS_HALF") is tuning.TUNABLES["danger.TURNS_HALF"]
    for name in ("search.opp_roll_samples", "search.opponent_actions", "search.opponent_expand", "search.opponent_proposals"):
        assert tuning.TUNABLES[name].requires_depth == 2      # Searcher._future_values only runs at depth >= 2
    assert tuning.TUNABLES["search.beam"].requires_depth == 1
    assert tuning.spec_depth("search:depth=2,beam=4") == 2 and tuning.spec_depth("heuristic:temp=0.1") == 1
    assert tuning.spec_depth("search:beam=4") == 1
    with pytest.raises(KeyError):
        tuning.find("no_such_tunable")
    with pytest.raises(KeyError):
        tuning.find("DECAY")   # ambiguous: coalitions / politics / opponent_model


def test_apply_restore_round_trip():
    before = {t.name: [tg.get() for tg in t.targets] for t in tuning.TUNABLES.values() if t.kind != "search"}
    overrides = {"danger.TURNS_HALF": 2.0, "danger.BLOCK_NEED": 0.0, "coalitions.BLOC_THRESHOLD": 5.0,
                 "politics.DECAY": 0.5, "politics.BASELINE": 0.3, "opponent_model.DECAY": 0.5,
                 "trading.accept_margin": 0.03, "placement.RESOURCE_DEMAND": [2, 2, 2, 2, 2],
                 "opponent_model.stage_late_drop": 0.0, "danger.danger_multiplier": False,
                 "danger.steal_factor": False, "danger.rob_break": False,
                 "trading.feed_leader_guard": False, "search.beam": 2}
    demand_obj = placement.RESOURCE_DEMAND
    token = tuning.apply(overrides)
    try:
        assert danger.TURNS_HALF == 2.0 and danger.BLOCK_NEED == 0.0
        assert coalitions.BLOC_THRESHOLD == 5.0
        assert coalitions.CoalitionDetector.allies.__defaults__ == (5.0,)   # def-time default rewritten
        assert politics.PoliticalState.decay.__defaults__ == (0.5,)
        assert politics.PoliticalState(3).baseline == 0.3
        assert opponent_model._EW.add.__defaults__ == (0.5,) and opponent_model.DECAY == 0.5
        assert trading.should_accept.__defaults__[1] == 0.03
        assert placement.RESOURCE_DEMAND == [1.0] * 5 and placement.RESOURCE_DEMAND is demand_obj   # in place, normalised
        assert robber.danger_multiplier(None) == 1.0 and danger.danger_multiplier(None) == 1.0
        assert robber.steal_factor(None, [1.0] * 5) == 1.0 and danger.steal_factor(None) == 1.0
        assert robber.rob_break_probability(None) == 0.0 and danger.rob_break_probability(None) == 0.0
        assert trading.offer_is_feeding_leader(None, 0, 1, [1, 0, 0, 0, 0]) == (False, "")
        import catanbot.search as S
        s = new_game(4, rng=random.Random(1))
        assert S.trade_stage_factor(s) == 1.0 and trading.trade_stage_factor(s) == 1.0
        assert S.SearchConfig.beam == S.SearchConfig().beam   # search kinds never touch globals
    finally:
        tuning.restore(token)
    after = {t.name: [tg.get() for tg in t.targets] for t in tuning.TUNABLES.values() if t.kind != "search"}
    assert after == before
    assert robber.danger_multiplier is danger.danger_multiplier
    assert robber.steal_factor is danger.steal_factor
    assert robber.rob_break_probability is danger.rob_break_probability
    assert tuning.verify_registry() == []


def test_flag_on_and_stage_default_are_noops():
    assert tuning.apply({"danger.danger_multiplier": True, "trading.feed_leader_guard": True,
                         "danger.steal_factor": True, "danger.rob_break": True}) == []
    s = new_game(4, rng=random.Random(3))
    f = tuning.TUNABLES["opponent_model.stage_late_drop"].make(0.7)
    assert f(s) == pytest.approx(opponent_model.trade_stage_factor(s))


def test_overridden_context_manager_and_nesting():
    with tuning.overridden({"danger.TURNS_HALF": 2.0}):
        assert danger.TURNS_HALF == 2.0
        with tuning.overridden({"danger.TURNS_HALF": 9.0}):
            assert danger.TURNS_HALF == 9.0
        assert danger.TURNS_HALF == 2.0
    assert danger.TURNS_HALF == 3.0


# ---------------------------------------------------------------------------
# ParamBot
# ---------------------------------------------------------------------------
class Boom(HeuristicBot):
    def decide(self, state, legal, rng):
        assert danger.TURNS_HALF == 1.0    # the override is live inside the inner bot
        raise RuntimeError("boom")


def test_parambot_restores_when_inner_raises():
    bot = ParamBot(Boom(), {"danger.TURNS_HALF": 1.0, "danger.danger_multiplier": False})
    s = new_game(4, rng=random.Random(1))
    with pytest.raises(RuntimeError):
        bot.decide(s, [("a",), ("b",)], random.Random(0))
    assert danger.TURNS_HALF == 3.0 and robber.danger_multiplier is danger.danger_multiplier
    assert bot.stats["decisions"] == 1 and len(bot.stats["times"]) == 1


def test_parambot_plays_a_game_and_records_times():
    cand = {"danger.TURNS_HALF": 2.0, "coalitions.SCALE": 1.0}
    bots = [ParamBot(HeuristicBot(temperature=0.15), cand), ParamBot(HeuristicBot(temperature=0.15), {}),
            ParamBot(HeuristicBot(temperature=0.15), cand), ParamBot(HeuristicBot(temperature=0.15), {})]
    res = play_game(bots, rng=random.Random(5), seed=5, max_turns=120)
    assert res.turns > 0 and len(res.vps) == 4
    for b in bots:
        assert b.stats["decisions"] == len(b.stats["times"]) > 0
        assert all(t >= 0.0 for t in b.stats["times"])
    assert danger.TURNS_HALF == 3.0 and coalitions.SCALE == 0.5
    assert "[cand:" in bots[0].name and "[default]" in bots[1].name
    assert bots[0].trade_bias == 0.0     # attribute delegation to the inner bot


def test_parambot_decision_time_excludes_override_overhead(monkeypatch):
    import time
    from catanbot.engine import legal_actions
    orig_apply, orig_restore = tuning.apply, tuning.restore

    def slow_apply(ov):
        time.sleep(0.02)
        return orig_apply(ov)

    def slow_restore(tok):
        time.sleep(0.02)
        orig_restore(tok)

    monkeypatch.setattr(tuning, "apply", slow_apply)
    monkeypatch.setattr(tuning, "restore", slow_restore)
    bot = ParamBot(HeuristicBot(temperature=0.15), {"danger.TURNS_HALF": 2.0})
    s = new_game(4, rng=random.Random(1))
    legal = legal_actions(s)
    assert len(legal) > 1
    for _ in range(3):
        bot.decide(s, legal, random.Random(0))
    assert bot.stats["decisions"] == 3 and all(t < 0.02 for t in bot.stats["times"])   # the 40 ms sleeps are not in it
    assert bot.stats["overhead"] >= 3 * 0.04                                          # ... they are here
    assert danger.TURNS_HALF == 3.0


NONSEARCH = [t for t in tuning.TUNABLES.values() if t.kind != "search"]


def _all_targets():
    return {t.name: [tg.get() for tg in t.targets] for t in NONSEARCH}


def test_parambot_restores_every_registry_target_after_midgame_raise():
    before = _all_targets()
    every_candidate = {t.name: t.candidates[0] for t in NONSEARCH}   # every non-search tunable at once

    class Boom(HeuristicBot):
        calls = 0

        def decide(self, state, legal, rng):
            Boom.calls += 1
            if Boom.calls == 5:
                raise RuntimeError("boom on decision 5")
            return super().decide(state, legal, rng)

    bots = [ParamBot(Boom(temperature=0.15), every_candidate), ParamBot(HeuristicBot(temperature=0.15), {}),
            ParamBot(Boom(temperature=0.15), every_candidate), ParamBot(HeuristicBot(temperature=0.15), {})]
    with pytest.raises(RuntimeError, match="decision 5"):
        play_game(bots, rng=random.Random(3), seed=3, max_turns=100)
    assert Boom.calls == 5
    assert _all_targets() == before
    assert tuning.verify_registry() == []
    assert robber.danger_multiplier is danger.danger_multiplier
    assert trading.offer_is_feeding_leader is heuristic.offer_is_feeding_leader
    import catanbot.search as S
    assert S.trade_stage_factor is opponent_model.trade_stage_factor is trading.trade_stage_factor
    assert not danger._cache


def test_two_parambots_each_see_their_own_values(monkeypatch):
    """Two candidate bots with different overrides plus two default bots in one game: the value seen
    inside a strategy function (danger.win_path, called from the bots' decisions) is the deciding
    bot's own, and the defaults never see an override."""
    before = _all_targets()
    seen_in_win_path = []
    current = ["-"]
    orig_win_path, orig_dm = danger.win_path, danger.danger_multiplier

    def logged_win_path(state, i, belief=None):
        assert robber.danger_multiplier is danger.danger_multiplier          # both import sites always agree
        seen_in_win_path.append((current[0], danger.TURNS_HALF, coalitions.SCALE, danger.danger_multiplier is orig_dm))
        return orig_win_path(state, i, belief)

    monkeypatch.setattr(danger, "win_path", logged_win_path)   # win_paths looks the name up at call time

    class Tagged(HeuristicBot):
        def __init__(self, tag):
            super().__init__(temperature=0.15)
            self.tag = tag

        def decide(self, state, legal, rng):
            current[0] = self.tag
            try:
                return super().decide(state, legal, rng)
            finally:
                current[0] = "-"

    A = {"danger.TURNS_HALF": 1.5, "coalitions.SCALE": 2.0, "danger.danger_multiplier": False}
    B = {"danger.TURNS_HALF": 6.0, "coalitions.SCALE": 0.25}
    bots = [ParamBot(Tagged("A"), A), ParamBot(Tagged("D1"), {}), ParamBot(Tagged("B"), B), ParamBot(Tagged("D2"), {})]
    play_game(bots, rng=random.Random(11), seed=11, max_turns=150)
    expect = {"A": (1.5, 2.0, False), "B": (6.0, 0.25, True), "D1": (3.0, 0.5, True), "D2": (3.0, 0.5, True),
              "-": (3.0, 0.5, True)}
    assert {tag for tag, *_ in seen_in_win_path} >= {"A", "B", "D1", "D2"}
    for tag, th, sc, orig in seen_in_win_path:
        assert (th, sc, orig) == expect[tag], (tag, th, sc, orig)
    assert _all_targets() == before


def test_parambot_search_knob_only_on_search_bot():
    with pytest.raises(TypeError):
        ParamBot(HeuristicBot(), {"search.beam": 2})
    bot = ParamBot(make_bot("search:depth=1,beam=4,expand=8,evaluator=heuristic"), {"search.beam": 2})
    assert bot.inner.config.beam == 4
    token, bot_token = bot._enter()
    assert bot.inner.config.beam == 2
    bot._exit((token, bot_token))
    assert bot.inner.config.beam == 4


# ---------------------------------------------------------------------------
# paired design and statistics
# ---------------------------------------------------------------------------
def test_seat_patterns_balance_and_seeds_repeat():
    for n in (3, 4):
        jobs = tuning.paired_jobs(6, seed=3, num_players=n)
        assert sum(p.count("C") for p, _ in jobs) == sum(p.count("D") for p, _ in jobs)
        assert [s for _, s in jobs] == [tuning.game_seed(3, g) for g in range(6)]
    assert tuning.paired_jobs(4, 3, 4) == tuning.paired_jobs(4, 3, 4)   # identical across candidate values
    assert [p for p, _ in tuning.paired_jobs(2, 0, 3)] == ["CCD", "DDC"]


def test_paired_stats_two_vs_two_matches_binomial():
    def game(pattern, winner, seed):
        vps = [10 if i == winner else 6 for i in range(4)]
        return {"seed": seed, "pattern": pattern, "winner": winner, "vps": vps, "turns": 50, "actions": 100,
                "duration": 1.0, "seat_times": [[0.001], [0.002], [0.001], [0.002]], "seat_decisions": [1, 1, 1, 1]}
    games = [game("CCDD", 0, 1), game("DDCC", 0, 2), game("CDCD", 1, 3), game("DCDC", 1, 4)]  # C wins 2, D wins 2
    st = tuning.paired_stats(games)
    assert st["games"] == 4 and st["cand_wins"] == 2 and st["def_wins"] == 2 and st["draws"] == 0
    assert st["win_rate_cand"] == st["win_rate_def"] == 0.25 and st["delta"] == 0.0
    # paired differences are +-0.5: unbiased sample variance 1/3 -> se = sqrt((1/3) / 4); this is the
    # binomial SE of the side win rate (0.5 * sqrt(p(1-p)/N) * 2 = 0.25) times the N/(N-1) factor
    assert st["se"] == pytest.approx(((1.0 / 3.0) / 4) ** 0.5)
    assert st["se"] == pytest.approx(2 * st["side_se"] * 0.5 * (4 / 3) ** 0.5)
    assert st["ci95"][0] < 0 < st["ci95"][1]
    assert st["ms_mean_cand"] == pytest.approx(1.5) and st["ms_mean_def"] == pytest.approx(1.5)
    assert st["value_per_ms"] is None and "inconclusive" in tuning.verdict(st)
    games = [game("CCDD", 0, 1), game("DDCC", 2, 2), game("CDCD", 0, 3), game("DCDC", 1, 4)]  # C wins all
    st = tuning.paired_stats(games)
    assert st["delta"] == 0.5 and st["se"] == 0.0 and st["win_rate_cand"] == 0.5
    assert st["avg_vp_cand"] > st["avg_vp_def"]


def test_paired_stats_per_game_records():
    def game(pattern, winner, seed, mode):
        return {"seed": seed, "pattern": pattern, "winner": winner, "vps": [10 if i == winner else 6 for i in range(4)],
                "turns": 50, "actions": 100, "duration": 1.0, "seat_times": [[0.001, 0.003], [0.002], [0.001], [0.002]],
                "seat_decisions": [2, 1, 1, 1], "seat_overhead": [0.0004, 0.0, 0.0002, 0.0], "evaluator_mode": mode,
                "pid": 42}
    st = tuning.paired_stats([game("CCDD", 0, 1, "c++ (catanbot_core)"), game("DDCC", 0, 2, "c++ (catanbot_core)"),
                              game("CDCD", 3, 3, "c++ (catanbot_core)")])
    recs = st["records"]
    assert [r["seed"] for r in recs] == [1, 2, 3] and [r["pattern"] for r in recs] == ["CCDD", "DDCC", "CDCD"]
    assert [r["winning_side"] for r in recs] == ["cand", "default", "default"]
    assert [r["diff"] for r in recs] == [0.5, -0.5, -0.5]
    assert st["delta"] == pytest.approx(sum(r["diff"] for r in recs) / 3)
    assert recs[0]["seat_ms_mean"][0] == pytest.approx(2.0) and recs[0]["seat_decisions"] == [2, 1, 1, 1]
    assert recs[0]["seat_overhead_ms"][0] == pytest.approx(0.4) and recs[0]["evaluator_mode"].startswith("c++")
    assert st["evaluator_modes"] == ["c++ (catanbot_core)"]
    # candidate seats: CCDD -> 0,1 (3 decisions, overhead 0.0004); DDCC -> 2,3 (2, 0.0002); CDCD -> 0,2 (3, 0.0006)
    assert st["decisions_cand"] == 8 and st["decisions_def"] == 7
    assert st["overhead_ms_cand"] == pytest.approx(1000 * (0.0004 + 0.0002 + 0.0006) / 8)
    assert st["overhead_ms_def"] == pytest.approx(1000 * (0.0002 + 0.0004 + 0.0) / 7)


def test_paired_stats_three_player_rotation_is_unbiased():
    # Equal strength: every seat wins its share -> the paired delta is zero on average.
    rows = []
    k = 0
    for pattern in ["CCD", "DDC", "CDC", "DCD", "DCC", "CDD"]:
        for w in range(3):
            rows.append({"seed": k, "pattern": pattern, "winner": w, "vps": [10, 5, 5], "turns": 1, "actions": 1,
                         "duration": 0.0, "seat_times": [[0.0]] * 3, "seat_decisions": [1] * 3})
            k += 1
    st = tuning.paired_stats(rows)
    assert st["delta"] == pytest.approx(0.0) and st["cand_seats"] == st["def_seats"]


# ---------------------------------------------------------------------------
# scripts/ablate.py
# ---------------------------------------------------------------------------
def _run(args, env=None):
    e = dict(os.environ, PYTHONPATH=ROOT)
    e.pop("CATANBOT_NO_ACCEL", None)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, SCRIPT] + args, cwd=ROOT, env=e, text=True, capture_output=True,
                          timeout=540)


def test_ablate_list_and_plan():
    p = _run(["--list"])
    assert p.returncode == 0 and "danger.TURNS_HALF" in p.stdout and "search.beam" in p.stdout
    p = _run(["--tunable", "TURNS_HALF", "--plan", "--games", "3", "--players", "3"])
    assert p.returncode == 0
    assert "base spec heuristic:temp=0.15" in p.stdout and "seat patterns: CCD DDC CDC" in p.stdout
    p = _run(["--tunable", "search.beam", "--plan"])
    assert p.returncode == 0 and "base spec search:depth=1,beam=4,expand=8,evaluator=heuristic" in p.stdout
    p = _run(["--tunable", "search.beam", "--plan", "--base-spec", "heuristic"])
    assert p.returncode != 0 and "search base spec" in (p.stdout + p.stderr)
    # the opponent knobs are only read at depth >= 2: depth-2 spec by default, a depth-1 spec is refused
    p = _run(["--tunable", "search.opp_roll_samples", "--plan"])
    assert p.returncode == 0 and "base spec search:depth=2,beam=4,expand=8,evaluator=heuristic" in p.stdout
    p = _run(["--tunable", "search.opponent_actions", "--plan", "--base-spec", "search:depth=1,beam=4,expand=8"])
    assert p.returncode != 0 and "depth >= 2" in (p.stdout + p.stderr)
    p = _run(["--tunable", "search.trade_proposals", "--plan", "--base-spec", "search:depth=1,beam=4,expand=8"])
    assert p.returncode == 0
    p = _run(["--tunable", "nope"])
    assert p.returncode == 2 and "unknown tunable" in p.stderr


def test_ablate_smoke_run_writes_json(tmp_path):
    out = tmp_path / "turns.json"
    p = _run(["--tunable", "danger.TURNS_HALF", "--values", "2", "--games", "2", "--workers", "1", "--seed", "1",
              "--players", "4", "--json", str(out), "--quiet"])
    assert p.returncode == 0, p.stderr
    assert "evaluator mode:" in p.stdout and "delta" in p.stdout
    d = json.loads(out.read_text())
    for key in ("tunable", "kind", "default", "base_spec", "players", "games", "seed", "evaluator_mode", "results"):
        assert key in d
    assert d["tunable"] == "danger.TURNS_HALF" and d["games"] == 2 and d["players"] == 4
    assert len(d["results"]) == 1
    r = d["results"][0]
    for key in ("value", "games", "cand_wins", "def_wins", "draws", "cand_seats", "def_seats", "win_rate_cand",
                "win_rate_def", "delta", "se", "ci95", "avg_vp_cand", "avg_vp_def", "ms_mean_cand", "ms_p95_cand",
                "ms_mean_def", "ms_p95_def", "extra_ms", "value_per_ms", "verdict"):
        assert key in r, key
    assert r["value"] == 2.0 and r["games"] == 2 and r["cand_seats"] == 4 and r["def_seats"] == 4
    assert r["cand_wins"] + r["def_wins"] + r["draws"] == 2
    assert r["ms_mean_cand"] > 0 and r["ms_mean_def"] > 0
    # per-game records: both settings in every game, seats rotated, seeds from the seed formula
    recs = r["records"]
    assert len(recs) == 2 and [x["pattern"] for x in recs] == ["CCDD", "DDCC"]
    assert [x["seed"] for x in recs] == [tuning.game_seed(1, 0), tuning.game_seed(1, 1)]
    for x in recs:
        assert len(x["vps"]) == 4 and len(x["seat_decisions"]) == 4 and len(x["seat_ms_mean"]) == 4
        assert x["evaluator_mode"] == d["evaluator_mode"]
    # ... and the statistics recompute from them
    diffs = [x["diff"] for x in recs]
    delta = sum(diffs) / 2
    assert r["delta"] == pytest.approx(delta)
    assert r["se"] == pytest.approx((sum((x - delta) ** 2 for x in diffs) / 1 / 2) ** 0.5)
    assert r["ci95"] == pytest.approx([max(-1, delta - 1.96 * r["se"]), min(1, delta + 1.96 * r["se"])])
    assert d["worker_evaluator_modes"] == [d["evaluator_mode"]] == r["evaluator_modes"]
    assert r["overhead_ms_cand"] >= 0 and r["overhead_ms_def"] >= 0


def test_ablate_python_evaluator_path_sets_env(tmp_path):
    from catanbot import accel
    out = tmp_path / "demand.json"
    p = _run(["--tunable", "placement.RESOURCE_DEMAND", "--values", "1/1/1/1/1", "--games", "1", "--workers", "1",
              "--players", "3", "--json", str(out), "--quiet"])
    assert p.returncode == 0, p.stderr
    if accel.load_core() is not None:
        assert "re-executing with CATANBOT_NO_ACCEL=1" in p.stdout
    assert "evaluator mode: python (CATANBOT_NO_ACCEL=1)" in p.stdout
    d = json.loads(out.read_text())
    assert d["evaluator_mode"] == "python (CATANBOT_NO_ACCEL=1)" and d["needs_python_evaluator"]
    assert d["results"][0]["value"] == [1.0] * 5
    # the process that played the game reports the Python evaluator too (both sides run in it)
    assert d["worker_evaluator_modes"] == ["python (CATANBOT_NO_ACCEL=1)"]
    assert all(x["evaluator_mode"] == "python (CATANBOT_NO_ACCEL=1)" for x in d["results"][0]["records"])
    assert "evaluator mode in the game processes: python (CATANBOT_NO_ACCEL=1)" in p.stdout
    # A tunable that does not need it keeps the default mode (no re-exec).
    p = _run(["--tunable", "danger.BLOCK_NEED", "--plan"])
    assert p.returncode == 0 and "re-executing" not in p.stdout
    assert ("evaluator mode: c++" in p.stdout) == bool(accel.load_core() is not None)


def test_markdown_table_and_results_block(tmp_path):
    mod = _load_script()
    st = tuning.paired_stats([{"seed": 1, "pattern": "CCDD", "winner": 0, "vps": [10, 6, 6, 5], "turns": 80,
                               "actions": 300, "duration": 1.0, "seat_times": [[0.001]] * 4, "seat_decisions": [1] * 4}])
    st.update(value=2.0, value_text="2", verdict=tuning.verdict(st))
    rep = {"tunable": "danger.TURNS_HALF", "kind": "weight", "default_text": "3", "base_spec": "heuristic:temp=0.15",
           "evaluator_mode": "c++ (catanbot_core)", "results": [st]}
    table = mod.markdown_table([rep])
    assert table.startswith("| tunable |") and "| danger.TURNS_HALF | weight | 2 | 3 | heuristic | c++ | 1 |" in table
    doc = tmp_path / "A.md"
    doc.write_text("# Doc\n\nintro\n\n<!-- ablate:results:start -->\nold\n<!-- ablate:results:end -->\n\ntail\n")
    mod.write_markdown(str(doc), table, "header line")
    text = doc.read_text()
    assert "old" not in text and "header line" in text and text.startswith("# Doc") and text.rstrip().endswith("tail")
    mod.write_markdown(str(tmp_path / "new.md"), table, "h")
    assert "| danger.TURNS_HALF |" in (tmp_path / "new.md").read_text()
