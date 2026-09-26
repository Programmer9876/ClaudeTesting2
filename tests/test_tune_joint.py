"""Tests for scripts/tune_joint.py (SPSA joint tuning) and the ``tune=`` bot spec.

Almost everything runs on synthetic games: a mock runner in-process (a noisy objective with a planted
optimum) or the hidden ``--fake-optimum`` option through the command line, which go through the real
iteration / state / resume code.  ParamBot isolation is checked in a real game; two smoke tests play real
games (2 self-play games; 2 seeds x 2 arms against Catanatron).
"""
import copy
import importlib.util
import io
import json
import os
import random
import subprocess
import sys

import pytest

from catanbot import danger, devcards, heuristic, opponent_model, robber, trading, tuning
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.agents.param_bot import ParamBot, format_tune, parse_tune, tuned_spec
from catanbot.selfplay import make_bot, play_game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "tune_joint.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TJ = _load("tune_joint_under_test", SCRIPT)
ENV = {"code": "test", "python": "3", "catanatron": "none", "evaluator": "test", "hashseed": "0"}


def _config(argv):
    cfg, params, notes = TJ.build_config(TJ.build_parser().parse_args(list(argv) + ["--state", "unused.json"]))
    TJ.finish_context(cfg, ENV)
    return cfg, params


def _mock_runner(params, optimum, kappa, noise):
    """2v2 games whose theta+ side wins with probability 0.5 + kappa (f(theta+) - f(theta-)),
    f = -sum ((x - x*) / scale)^2: a noisy objective with a known optimum."""
    sc = {p.name: p.scale for p in params}
    calls = []

    def f(ov):
        return -sum(((float(ov[n]) - x) / sc[n]) ** 2 for n, x in optimum.items())

    def runner(jobs):
        calls.append(jobs)
        out = []
        for j in jobs:
            p = min(0.98, max(0.02, 0.5 + kappa * (f(j["plus"]) - f(j["minus"]))))
            won = random.Random(f"{noise}:{j['seed']}:{j['pattern']}").random() < p
            out.append({"k": j["k"], "j": j["j"], "seed": j["seed"], "pattern": j["pattern"], "status": "ok",
                        "diff": 0.5 if won else -0.5, "vp_diff": 2.0 if won else -2.0})
        return out

    return runner, calls


NONSEARCH = [t for t in tuning.TUNABLES.values() if t.kind != "search"]


def _all_targets():
    return {t.name: [tg.get() for tg in t.targets] for t in NONSEARCH}


# ---------------------------------------------------------------------------
# parameters, normalisation, bounds
# ---------------------------------------------------------------------------
def test_parameters_normalisation_and_bounds():
    params = TJ.build_params(TJ.GROUPS["robber"]["params"])
    th = params[0]
    assert th.name == "danger.TURNS_HALF" and (th.default, th.lo, th.hi, th.scale) == (3.0, 1.5, 6.0, 4.5)
    assert not th.integer and th.norm(th.default) == 0.0 and th.raw(th.norm(4.2)) == pytest.approx(4.2)
    assert th.install(10.0) == 6.0 and th.install(-10.0) == 1.5          # clipped to the bounds
    for p in params:     # default bounds = the registry's default + candidate span = one normalised unit
        t = tuning.find(p.name)
        vals = [t.default] + list(t.candidates)
        assert (p.lo, p.hi) == (min(vals), max(vals)) and p.u_hi - p.u_lo == pytest.approx(1.0)
        assert p.u_lo <= 0.0 <= p.u_hi and p.start == t.default
    # an integer knob: rounded, and perturbed by at least half a unit so the two sides always differ
    [tp] = TJ.build_params(["search.trade_proposals"])
    assert tp.integer and (tp.lo, tp.hi, tp.scale) == (0.0, 5.0, 5.0)
    assert tp.install(tp.norm(2.4)) == 2 and tp.install(tp.norm(2.6)) == 3 and isinstance(tp.install(0.1), int)
    xp, xm, d = TJ.plus_minus([tp], [0.0], 0.01, [1])
    assert (xp, xm) == ([4], [3]) and d == [pytest.approx(0.2)]
    # a real parameter at its upper bound: theta+ is clipped, the difference is one-sided
    xp, xm, d = TJ.plus_minus([th], [th.u_hi], 0.2, [1])
    assert xp == [6.0] and xm == [pytest.approx(6.0 - 0.2 * 4.5)] and d == [pytest.approx(0.2)]
    xp, xm, d = TJ.plus_minus([th], [0.0], 0.2, [-1])
    assert xp == [pytest.approx(3.0 - 0.9)] and xm == [pytest.approx(3.9)] and d == [pytest.approx(-0.4)]
    # --bounds / --scale / --start
    [b] = TJ.build_params(["TURNS_HALF"], bounds={"danger.TURNS_HALF": (2.0, 9.0)},
                          scales={"danger.TURNS_HALF": 2.0}, starts={"danger.TURNS_HALF": 5.0})
    assert (b.lo, b.hi, b.scale) == (2.0, 9.0, 2.0) and b.norm(b.start) == 1.0
    with pytest.raises(ValueError, match="outside the bounds"):
        TJ.build_params(["danger.TURNS_HALF"], starts={"danger.TURNS_HALF": 7.0})
    with pytest.raises(ValueError, match="twice"):
        TJ.build_params(["danger.TURNS_HALF", "TURNS_HALF"])
    # flags, vectors, categorical and on/off knobs belong to the factorial tool
    for name, why in (("danger.danger_multiplier", "flag"), ("trading.feed_leader_guard", "flag"),
                      ("placement.RESOURCE_DEMAND", "not a number"), ("openings.policy", "not a number"),
                      ("search.depth", "switch"), ("search.counters", "switch"), ("search.paths", "switch"),
                      ("search.respond_lookahead", "switch")):
        with pytest.raises(ValueError, match=why):
            TJ.build_params([name])


def test_gains_perturbation_auto_gain_and_step():
    cfg = {"a": 1.0, "A": 4, "alpha": 0.602, "c": 0.2, "gamma": 0.101}
    assert TJ.gains(cfg, 0) == (pytest.approx(1.0 / 5 ** 0.602), pytest.approx(0.2))
    assert TJ.gains(cfg, 9) == (pytest.approx(1.0 / 14 ** 0.602), pytest.approx(0.2 / 10 ** 0.101))
    d = TJ.perturbation(7, 3, 6)
    assert d == TJ.perturbation(7, 3, 6) and set(d) <= {-1, 1}             # deterministic
    allk = [x for k in range(200) for x in TJ.perturbation(7, k, 6)]
    assert abs(sum(allk)) < 4 * len(allk) ** 0.5                            # balanced
    assert len({tuple(TJ.perturbation(7, k, 6)) for k in range(50)}) > 20   # varies with k
    # auto gain: a one-standard-error estimate of D moves a parameter by first_step at k = 0
    a = TJ.auto_gain(0.05, 10, 0.602, 0.25, 36, 0.5)
    g_one_se = (0.5 / 36 ** 0.5) / (2 * 0.25)
    assert a / 11 ** 0.602 * g_one_se == pytest.approx(0.05)
    params = TJ.build_params(["danger.TURNS_HALF", "devcards.KNIGHT_VALUE"])
    ghat, step, nxt = TJ.spsa_step(params, [0.0, 0.0], 0.1, [0.4, -0.4], a_k=0.2, max_step=0.1)
    assert ghat == [pytest.approx(0.25), pytest.approx(-0.25)] and step == [pytest.approx(0.05), pytest.approx(-0.05)]
    assert nxt == step
    ghat, step, nxt = TJ.spsa_step(params, [0.0, 0.0], 0.1, [0.4, -0.4], a_k=0.5, max_step=0.1)
    assert step == [0.1, -0.1]                                                # capped
    ghat, step, nxt = TJ.spsa_step(params, [params[0].u_hi - 0.01, 0.2], 0.1, [0.4, 0.0], 0.5, 0.1)
    assert nxt[0] == params[0].u_hi and ghat[1] == 0.0 and nxt[1] == 0.2     # clipped; no difference, no step


# ---------------------------------------------------------------------------
# jobs, seeds, configuration
# ---------------------------------------------------------------------------
def test_jobs_mirrored_pairs_share_a_board_and_seeds_are_fresh():
    cfg, params = _config(["--params", "danger.TURNS_HALF,devcards.KNIGHT_VALUE", "--max-games", "120",
                           "--games", "12", "--base-spec", "heuristic:temp=0.15"])
    assert cfg["iterations"] == 10 and cfg["games_per_iteration"] == 12
    jobs = TJ.make_jobs(cfg, 3, {"danger.TURNS_HALF": 4.0}, {"danger.TURNS_HALF": 2.0})
    assert [j["pattern"] for j in jobs] == tuning.SEAT_PATTERNS[4] * 2
    for a, b in zip(jobs[::2], jobs[1::2]):
        assert a["seed"] == b["seed"] and a["pattern"] == b["pattern"].translate(str.maketrans("CD", "DC"))
    assert len({j["seed"] for j in jobs}) == 6
    assert all(j["plus"] == {"danger.TURNS_HALF": 4.0} and j["minus"] == {"danger.TURNS_HALF": 2.0} for j in jobs)
    nxt = TJ.make_jobs(cfg, 4, {}, {})
    assert not {j["seed"] for j in jobs} & {j["seed"] for j in nxt}
    # against Catanatron: consecutive seed numbers, our seat s % 4 balanced, both arms on every seed
    cfgc, _ = _config(["--params", "danger.TURNS_HALF", "--mode", "catanatron", "--opponent", "value",
                       "--games", "8", "--max-games", "160"])
    assert cfgc["iterations"] == 10 and cfgc["games_per_iteration"] == 16
    jc = TJ.make_jobs(cfgc, 2, {"danger.TURNS_HALF": 4.0}, {"danger.TURNS_HALF": 2.0})
    ss = [j["s"] for j in jc]
    assert ss == list(range(ss[0], ss[0] + 8)) and sorted(s % 4 for s in ss) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [a["role"] for a in jc[0]["arms"]] == ["plus", "minus"]
    assert jc[0]["arms"][0]["overrides"] == {"danger.TURNS_HALF": 4.0} and jc[0]["ctx"]["opponent"] == "value"
    assert TJ.make_jobs(cfgc, 3, {}, {})[0]["s"] == ss[-1] + 1
    # the budget: --max-games is required and bounds the plan
    with pytest.raises(SystemExit, match="--max-games N is required"):
        _config(["--params", "danger.TURNS_HALF"])
    with pytest.raises(SystemExit, match="exceed --max-games"):
        _config(["--params", "danger.TURNS_HALF", "--max-games", "100", "--games", "36", "--iterations", "3"])
    cfg5, _ = _config(["--params", "danger.TURNS_HALF", "--max-games", "60", "--games", "5"])
    assert cfg5["games"] == 6 and cfg5["iterations"] == 10          # rounded up to whole mirrored pairs


def test_settings_follow_the_parameters_and_groups_restrict():
    cfg, params = _config(["--group", "robber", "--max-games", "2400"])
    assert cfg["base_spec"] == tuning.DEFAULT_SEARCH_SPEC and cfg["python_eval"] and not cfg["counters"]
    assert cfg["iterations"] == 66 and cfg["games"] == 36 and cfg["c"] == 0.25 and cfg["a_auto"]
    assert cfg["A"] == 7 and cfg["alpha"] == 0.602 and cfg["gamma"] == 0.101
    cfg, params = _config(["--group", "robber", "--exclude", "heuristic.EXPOSURE_WEIGHT,placement.PLACEMENT_ROBBER_Q",
                           "--max-games", "2400"])
    assert len(params) == 4 and not cfg["python_eval"]
    # the trade group: counter_margin brings counter=1 and the counter-offer rules unless excluded
    cfg, params = _config(["--group", "trade", "--max-games", "2400"])
    assert cfg["base_spec"].endswith(",counter=1") and cfg["counters"] and len(params) == 6
    cfg, params = _config(["--group", "trade", "--exclude", "search.counter_margin,politics.BASELINE",
                           "--max-games", "2400"])
    assert cfg["base_spec"] == tuning.DEFAULT_SEARCH_SPEC and not cfg["counters"]
    assert [p.name for p in params] == ["trading.accept_margin", "politics.MAX_SLACK", "coalitions.SCALE",
                                        "opponent_model.stage_late_drop"]
    cfg, params = _config(["--group", "trade", "--params", "trading.accept_margin", "--max-games", "2400"])
    assert [p.name for p in params] == ["trading.accept_margin"] and cfg["group"] is None
    for argv, msg in ((["--group", "trade", "--base-spec", tuning.DEFAULT_SEARCH_SPEC], "never acts without counter=1"),
                      (["--group", "robber", "--base-spec", "heuristic:temp=0.1"], "search bot"),
                      (["--group", "trade", "--mode", "catanatron", "--opponent", "value"], "counter-offer rules"),
                      (["--group", "robber", "--exclude", "nope.X"], "unknown tunable"),
                      (["--group", "robber", "--exclude", "coalitions.SCALE"], "not in the parameter list"),
                      (["--params", "danger.danger_multiplier"], "flag"),
                      (["--group", "nope"], "unknown group"),
                      (["--params", "winpaths.KAPPA", "--base-spec", tuning.DEFAULT_SEARCH_SPEC], "paths=1")):
        with pytest.raises(SystemExit, match=msg):
            _config(argv + ["--max-games", "2400"])


# ---------------------------------------------------------------------------
# SPSA on a synthetic objective; resume
# ---------------------------------------------------------------------------
OPT = {"danger.TURNS_HALF": 4.8, "danger.BLOCK_NEED": 1.0, "devcards.KNIGHT_VALUE": 0.7}


def test_spsa_converges_on_a_noisy_objective_with_a_known_optimum(tmp_path):
    cfg, params = _config(["--params", ",".join(OPT), "--max-games", str(200 * 36), "--games", "36", "--seed", "3"])
    state = TJ.new_state(cfg, params, ENV)
    runner, calls = _mock_runner(params, OPT, kappa=0.5, noise=3)
    path = str(tmp_path / "spsa.json")
    assert TJ.run(state, path, runner, log=io.StringIO()) == "done"
    assert state["k"] == 200 and len(calls) == 200 and all(len(c) == 36 for c in calls)
    assert state["games_played"] == 200 * 36 and state["game_errors"] == 0
    rec = state["result"]["average"]
    for p in params:
        start_err = abs(p.norm(OPT[p.name]))
        assert start_err >= 0.25                                       # the optimum is far from the default ...
        assert abs(p.norm(rec[p.name]) - p.norm(OPT[p.name])) < 0.1, (p.name, rec[p.name])   # ... and found
    assert TJ.load_state(path)["theta"] == state["theta"]            # the file holds the in-memory state
    # the history holds everything needed to audit an iteration
    h = state["history"][17]
    assert h["k"] == 17 and set(h["delta"]) <= {-1, 1} and len(h["plus"]) == len(h["minus"]) == 3
    assert h["ghat"] == [pytest.approx(h["D"] / d) for d in h["d"]]
    assert h["theta_next"] == state["history"][18]["theta"]
    assert h["D"] == pytest.approx(sum(r["diff"] for r in runner(TJ.make_jobs(cfg, 17, *(
        TJ.overrides_of(params, x) for x in (h["plus"], h["minus"]))))) / 36)


def _strip(hist):
    return [{k: v for k, v in h.items() if k != "seconds"} for h in hist]


def test_resume_after_an_interrupted_iteration_reproduces_the_trajectory(tmp_path):
    cfg, params = _config(["--params", ",".join(OPT), "--max-games", "120", "--games", "12", "--seed", "5"])
    runner, _ = _mock_runner(params, OPT, kappa=0.5, noise=5)
    straight = TJ.new_state(copy.deepcopy(cfg), params, ENV)
    p1 = str(tmp_path / "a.json")
    TJ.run(straight, p1, runner, log=io.StringIO())
    assert straight["k"] == 10
    calls = [0]

    def flaky(jobs):              # killed in the middle of iteration 4 (after some of its games)
        calls[0] += 1
        if calls[0] == 5:
            runner(jobs[:5])
            raise KeyboardInterrupt
        return runner(jobs)

    p2 = str(tmp_path / "b.json")
    st = TJ.new_state(copy.deepcopy(cfg), params, ENV)
    TJ.save_state(p2, st)
    with pytest.raises(KeyboardInterrupt):
        TJ.run(st, p2, flaky, log=io.StringIO())
    resumed = TJ.load_state(p2)
    assert resumed["k"] == 4 and len(resumed["history"]) == 4
    TJ.run(resumed, p2, runner, log=io.StringIO())
    assert _strip(resumed["history"]) == _strip(straight["history"])
    assert resumed["theta"] == straight["theta"] and resumed["result"] == straight["result"]
    ga = [json.loads(x) for x in open(TJ.games_path(p1))]
    gb = [json.loads(x) for x in open(TJ.games_path(p2))]
    assert ga == gb and len(ga) == 120             # the unfinished iteration left no game records


def _cli(args, env_extra=None, python=None, timeout=300):
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONHASHSEED", None)       # the script must pin it itself
    env.pop("CATANBOT_NO_ACCEL", None)
    env.update(env_extra or {})
    return subprocess.run([python or sys.executable, SCRIPT] + args, cwd=ROOT, env=env, text=True,
                          capture_output=True, timeout=timeout)


def test_cli_resume_status_and_guards(tmp_path):
    common = ["--params", "danger.TURNS_HALF,devcards.KNIGHT_VALUE,search.trade_proposals", "--games", "12",
              "--A", "1", "--fake-optimum", "danger.TURNS_HALF=4.5,devcards.KNIGHT_VALUE=0.7,search.trade_proposals=1",
              "--quiet"]
    a, b = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    p = _cli(common + ["--state", a, "--max-games", "72"])
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "re-executing with PYTHONHASHSEED=0" in p.stdout and "6 iterations x 12 games = 72 games planned" in p.stdout
    p = _cli(common + ["--state", b, "--max-games", "72", "--iterations", "3"])
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.load(open(b))["k"] == 3
    p = _cli(common + ["--state", b])                                   # exists, no --resume
    assert p.returncode == 2 and "exists" in p.stderr
    p = _cli(["--state", b, "--resume", "--iterations", "7"])          # 84 games > the cap of 72
    assert p.returncode != 0 and "exceed the run's cap" in p.stderr
    p = _cli(["--state", b, "--resume", "--iterations", "6", "--games", "99"])
    assert p.returncode == 0, p.stderr[-2000:]
    assert "ignored: --games" in p.stdout
    sa, sb = json.load(open(a)), json.load(open(b))
    assert sb["k"] == 6 and _strip(sa["history"]) == _strip(sb["history"]) and sa["theta"] == sb["theta"]
    assert sa["env"]["hashseed"] == "0" and sa["config"]["fake"]
    p = _cli(["--state", a, "--status"])
    assert p.returncode == 0
    out = p.stdout
    assert "trajectory" in out and "bot spec: search:depth=1,beam=4,expand=8,evaluator=heuristic" in out
    assert "scripts/league.py gate" in out and "--candidate-spec" in out and "never changes a default" in out
    assert "--formats 4p2v2,3p1v2" in out                   # search.trade_proposals is a trading term
    p = _cli(["--state", str(tmp_path / "none.json"), "--resume"])
    assert p.returncode == 2
    p = _cli(["--group", "robber", "--state", str(tmp_path / "c.json")])
    assert p.returncode != 0 and "--max-games N is required" in p.stderr


# ---------------------------------------------------------------------------
# error records, pool
# ---------------------------------------------------------------------------
def _job_fn(job):         # module level: the pool pickles it by reference
    if job.get("die"):
        os._exit(3)
    if job.get("raise"):
        raise RuntimeError("worker-side failure")
    return {"k": job["k"], "j": job["j"], "seed": job["seed"], "pattern": job["pattern"], "status": "ok",
            "diff": 0.5, "vp_diff": 1.0}


def test_run_jobs_survives_a_dead_worker_and_keeps_job_order():
    jobs = [{"mode": "selfplay", "k": 0, "j": j, "seed": j, "pattern": "CCDD", "die": j == 2, "raise": j == 4}
            for j in range(6)]
    out = TJ.run_jobs(jobs, 2, fn=_job_fn)
    assert [r["j"] for r in out] == list(range(6))
    assert out[2]["status"] == "error" and "died" in out[2]["error"]
    assert out[4]["status"] == "error" and "worker-side failure" in out[4]["error"]
    assert all(out[j]["status"] == "ok" for j in (0, 1, 3, 5))


def test_estimate_leaves_errors_out_and_a_failed_iteration_stops(monkeypatch):
    cfg, params = _config(["--params", "danger.TURNS_HALF", "--max-games", "8", "--games", "4"])
    rs = [{"status": "ok", "diff": 0.5, "vp_diff": 2.0}, {"status": "ok", "diff": -0.5, "vp_diff": -1.0},
          {"status": "error", "error": "x"}, {"status": "ok", "diff": 0.5, "vp_diff": 1.0}]
    est = TJ.estimate(cfg, rs)
    assert est["D"] == pytest.approx(1 / 6) and (est["units_ok"], est["units_err"]) == (3, 1)
    assert (est["plus_wins"], est["minus_wins"], est["draws"]) == (2, 1, 0)
    assert est["vp_diff"] == pytest.approx(2.0 / 3)
    assert TJ.estimate(dict(cfg, objective="vp"), rs)["D"] == pytest.approx(0.2 / 3)
    cfgc = dict(cfg, mode="catanatron")
    rc = [{"records": [{"arm": "plus", "status": "ok", "won": True, "our_vp": 10},
                       {"arm": "minus", "status": "ok", "won": False, "our_vp": 7}]},
          {"records": [{"arm": "plus", "status": "ok", "won": False, "our_vp": 6},
                       {"arm": "minus", "status": "error", "error": "timeout"}]},
          {"records": [{"arm": "plus", "status": "ok", "won": True, "our_vp": 10},
                       {"arm": "minus", "status": "ok", "won": True, "our_vp": 10}]}]
    est = TJ.estimate(cfgc, rc)
    assert est["D"] == pytest.approx(0.5) and (est["units_ok"], est["units_err"]) == (2, 1)
    state = TJ.new_state(cfg, params, ENV)
    with pytest.raises(RuntimeError, match="every game failed"):
        TJ.iteration(state, lambda jobs: [TJ.error_result(j, "boom") for j in jobs])
    assert state["k"] == 0 and state["history"] == []


def test_real_selfplay_job_is_deterministic_and_records_errors_and_timeouts(monkeypatch):
    before = _all_targets()
    cfg, params = _config(["--params", "danger.TURNS_HALF,devcards.KNIGHT_VALUE", "--base-spec", "heuristic:temp=0.15",
                           "--max-games", "2", "--games", "2"])
    job = TJ.make_jobs(cfg, 0, {"danger.TURNS_HALF": 4.1, "devcards.KNIGHT_VALUE": 0.7},
                       {"danger.TURNS_HALF": 1.9, "devcards.KNIGHT_VALUE": 0.4})[0]
    r1, r2 = TJ.play_selfplay_game(job), TJ.play_selfplay_game(job)
    assert r1["status"] == "ok" and r1["diff"] in (0.5, -0.5, 0.0)
    for key in ("winner", "vps", "turns", "actions", "diff", "vp_diff"):
        assert r1[key] == r2[key], key
    assert _all_targets() == before
    bad = TJ.play_selfplay_game(dict(job, spec="nope:x"))
    assert bad["status"] == "error" and "unknown bot spec" in bad["error"] and bad["seed"] == job["seed"]

    def slow(job):
        import time
        time.sleep(5)

    monkeypatch.setattr(TJ, "_selfplay_body", slow)
    t = TJ.play_selfplay_game(dict(job, timeout=1))
    assert t["status"] == "error" and "GameTimeout" in t["error"]


# ---------------------------------------------------------------------------
# ParamBot isolation: theta+ and theta- seats in one game
# ---------------------------------------------------------------------------
def test_plus_and_minus_seats_never_see_each_others_values(monkeypatch):
    before = _all_targets()
    seen = []
    current = ["-"]
    orig = danger.win_path

    def logged(state, i, belief=None):
        seen.append((current[0], danger.TURNS_HALF, danger.BLOCK_NEED, devcards.KNIGHT_VALUE))
        return orig(state, i, belief)

    monkeypatch.setattr(danger, "win_path", logged)

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

    plus = {"danger.TURNS_HALF": 4.2, "danger.BLOCK_NEED": 3.1, "devcards.KNIGHT_VALUE": 0.71}
    minus = {"danger.TURNS_HALF": 1.8, "danger.BLOCK_NEED": 0.9, "devcards.KNIGHT_VALUE": 0.39}
    pattern = "CDDC"
    tags = iter("P" if c == "C" else "M" for c in pattern)
    bots = TJ.seat_bots("heuristic:temp=0.15", plus, minus, pattern, make=lambda spec: Tagged(next(tags)))
    assert [b.inner.tag for b in bots] == ["P", "M", "M", "P"]
    assert "[plus:" in bots[0].name and "[minus:" in bots[1].name
    play_game(bots, rng=random.Random(5), seed=5, max_turns=150)
    expect = {"P": (4.2, 3.1, 0.71), "M": (1.8, 0.9, 0.39), "-": (3.0, 2.0, 0.55)}
    assert {tag for tag, *_ in seen} >= {"P", "M"}
    for tag, *vals in seen:
        assert tuple(vals) == expect[tag], (tag, vals)
    assert _all_targets() == before
    assert robber.danger_multiplier is danger.danger_multiplier
    assert trading.offer_is_feeding_leader is heuristic.offer_is_feeding_leader
    import catanbot.search as S
    assert S.trade_stage_factor is opponent_model.trade_stage_factor


# ---------------------------------------------------------------------------
# the tuned set as a bot spec, and the league command
# ---------------------------------------------------------------------------
def test_tune_spec_builds_a_parambot_and_nothing_else_changes():
    base = tuning.DEFAULT_SEARCH_SPEC
    b = make_bot(base + ",tune=danger.TURNS_HALF:2.4;devcards.KNIGHT_VALUE:0.62")
    assert isinstance(b, ParamBot) and b.overrides == {"danger.TURNS_HALF": 2.4, "devcards.KNIGHT_VALUE": 0.62}
    assert b.inner.name == base and b.config.beam == 4
    assert not isinstance(make_bot(base), ParamBot) and not isinstance(make_bot("heuristic:temp=0.1"), ParamBot)
    ov = {"danger.TURNS_HALF": 2.5, "danger.danger_multiplier": False, "placement.RESOURCE_DEMAND": [1.0] * 5}
    assert parse_tune(format_tune(ov)) == ov
    with pytest.raises(ValueError):
        parse_tune("danger.TURNS_HALF=2")
    spec = tuned_spec(base, {"danger.TURNS_HALF": 2.4, "search.trade_proposals": 1, "search.dump_candidates": 2,
                             "danger.danger_multiplier": False})
    assert ",trades=1," in spec and spec.endswith(
        "tune=danger.TURNS_HALF:2.4;search.dump_candidates:2;danger.danger_multiplier:off")
    bot = make_bot(spec)
    assert bot.config.trade_proposals == 1 and bot.overrides == {
        "danger.TURNS_HALF": 2.4, "search.dump_candidates": 2, "danger.danger_multiplier": False}
    assert tuned_spec(spec, {"danger.TURNS_HALF": 2.0}).count("tune=") == 1
    with pytest.raises(ValueError, match="search knob"):
        tuned_spec("heuristic:temp=0.1", {"search.trade_proposals": 1})
    # a tuned spec plays and leaves the process at the defaults
    before = _all_targets()
    bots = [make_bot("heuristic:temp=0.15,tune=danger.TURNS_HALF:1.7") for _ in range(2)] + \
           [make_bot("heuristic:temp=0.15") for _ in range(2)]
    play_game(bots, rng=random.Random(2), seed=2, max_turns=120)
    assert _all_targets() == before and danger.TURNS_HALF == 3.0


def test_recommendation_and_gate_command():
    cfg, params = _config(["--group", "robber", "--max-games", "360", "--games", "36"])
    state = TJ.new_state(cfg, params, ENV)
    assert state["result"]["overrides"] == {} and state["result"]["spec"] == cfg["base_spec"]
    names = [p.name for p in params]
    moved = [0.0] * len(params)
    moved[names.index("danger.TURNS_HALF")] = 0.2
    moved[names.index("heuristic.EXPOSURE_WEIGHT")] = -0.1
    row = {"a_k": 0.1, "c_k": 0.2, "units_ok": 36, "units_err": 0, "D": 0.0, "se": 0.08}
    state["history"] = [dict(row, k=i, theta_next=t) for i, t in enumerate(
        [[0.0] * len(params), [0.0] * len(params), moved, [2 * x for x in moved]])]
    state["theta"] = [2 * x for x in moved]
    rec = TJ.recommendation(state)
    assert rec["averaged_over"] == 2
    assert rec["overrides"] == {"danger.TURNS_HALF": pytest.approx(3.0 + 0.3 * 4.5),
                                "heuristic.EXPOSURE_WEIGHT": pytest.approx(0.25 - 0.15 * 0.6)}
    assert rec["last"]["danger.TURNS_HALF"] == pytest.approx(3.0 + 0.4 * 4.5) and rec["python_evaluator"]
    assert make_bot(rec["spec"]).overrides == rec["overrides"]
    state["result"] = rec
    cmd = TJ.gate_command(state, "runs/tune/robber.json")
    assert "scripts/league.py gate" in cmd and f"--candidate-spec '{rec['spec']}'" in cmd
    assert "--env CATANBOT_NO_ACCEL=1 --allow-no-accel" in cmd and "--formats" not in cmd
    buf = io.StringIO()
    TJ.print_status(state, "runs/tune/robber.json", out=buf)
    assert "cpp/heuristic.cpp" in buf.getvalue() and json.dumps(rec["overrides"]) in buf.getvalue()


def test_list_groups_and_plan(capsys, tmp_path):
    assert TJ.main(["--list-groups"]) == 0
    out = capsys.readouterr().out
    assert "robber:" in out and "trade:" in out and "POLITICS RULE" in out and "[politics rule]" in out
    for n in TJ.GROUPS["robber"]["params"] + TJ.GROUPS["trade"]["params"]:
        assert TJ.numeric_problem(tuning.find(n)) is None, n
    assert TJ.main(["--group", "trade", "--max-games", "2400", "--state", str(tmp_path / "t.json"), "--plan"]) == 0
    out = capsys.readouterr().out
    assert "counter=1" in out and "counter-offer rules ON" in out and "POLITICS RULE" in out
    assert "66 iterations x 36 games = 2376 games planned (cap --max-games 2400)" in out
    assert not os.path.exists(tmp_path / "t.json")


# ---------------------------------------------------------------------------
# real games (smoke)
# ---------------------------------------------------------------------------
def test_real_selfplay_smoke(tmp_path):
    st = str(tmp_path / "sp.json")
    p = _cli(["--params", "danger.TURNS_HALF,devcards.KNIGHT_VALUE", "--base-spec", "heuristic:temp=0.15",
              "--games", "2", "--max-games", "2", "--state", st, "--quiet"])
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    s = json.load(open(st))
    assert s["k"] == 1 and s["games_played"] == 2 and s["env"]["hashseed"] == "0"
    games = [json.loads(x) for x in open(st + ".games.jsonl")]
    assert len(games) == 2 and all(g["status"] == "ok" for g in games)
    assert games[0]["seed"] == games[1]["seed"] and [g["pattern"] for g in games] == ["CCDD", "DDCC"]
    h = s["history"][0]
    assert h["D"] == pytest.approx(sum(g["diff"] for g in games) / 2)


def test_real_catanatron_smoke(tmp_path):
    pytest.importorskip("catanatron")
    from importlib.metadata import version
    opponent = "value" if tuple(int(x) for x in version("catanatron").split(".")[:2]) >= (3, 3) else "random"
    st = str(tmp_path / "ct.json")
    p = _cli(["--params", "danger.TURNS_HALF,devcards.KNIGHT_VALUE", "--mode", "catanatron", "--opponent", opponent,
              "--games", "2", "--max-games", "4", "--state", st, "--quiet"], timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    s = json.load(open(st))
    assert s["k"] == 1 and s["games_played"] == 4 and s["config"]["mode"] == "catanatron"
    units = [json.loads(x) for x in open(st + ".games.jsonl")]
    assert [u["s"] for u in units] == [20000000, 20000001]
    for u in units:
        assert [r["arm"] for r in u["records"]] == ["plus", "minus"]
        assert all(r["status"] == "ok" and r["hashseed"] == "0" and r["seat"] == u["s"] % 4 for r in u["records"])
        assert u["records"][0]["game_seed"] == u["records"][1]["game_seed"] == u["s"] + 1
