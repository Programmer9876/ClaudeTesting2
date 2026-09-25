"""Tests for the tunable registry (catanbot/tuning.py), ParamBot and scripts/ablate.py."""
import importlib.util
import json
import os
import random
import subprocess
import sys

import pytest

from catanbot import coalitions, danger, opponent_model, placement, politics, robber, trading, tuning
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
    assert tuning.find("TURNS_HALF") is tuning.TUNABLES["danger.TURNS_HALF"]
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
    assert tuning.verify_registry() == []


def test_flag_on_and_stage_default_are_noops():
    assert tuning.apply({"danger.danger_multiplier": True, "trading.feed_leader_guard": True}) == []
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
