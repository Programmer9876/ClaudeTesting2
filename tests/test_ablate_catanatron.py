"""Tests for scripts/ablate_catanatron.py (paired ablations against Catanatron) and scripts/campaign.py.

Almost everything runs on synthetic games (``--fake-games`` in a subprocess, or a monkeypatched game
runner in-process), which go through the real pool / JSONL / resume / statistics code.  One test plays
two real games (an A/A pair against catanatron's RandomPlayer) to check the determinism the pairing
rests on.
"""
import importlib.util
import io
import json
import math
import os
import statistics
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "ablate_catanatron.py")
CAMPAIGN = os.path.join(ROOT, "scripts", "campaign.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load("ablate_catanatron_under_test", SCRIPT)


def _env(**extra):
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONHASHSEED", None)      # the script must pin it itself
    env.pop("CATANBOT_NO_ACCEL", None)
    env.update(extra)
    return env


def _run(args, env=None, timeout=300, check=True):
    proc = subprocess.run([sys.executable, SCRIPT] + args, env=env or _env(), capture_output=True, text=True,
                          timeout=timeout, cwd=ROOT)
    if check:
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc


def _fake(out, seeds=8, extra=(), tunable="danger.TURNS_HALF", values="2", workers=2, effect=0.2, env=None):
    args = ["--tunable", tunable, "--opponent", "vf", "--seeds", str(seeds), "--workers", str(workers),
            "--out", str(out), "--fake-games", str(effect), "--quiet"]
    if values is not None:
        args += ["--values", values]
    return _run(args + list(extra), env=env)


def _games(path):
    recs, bad = AB.read_jsonl(str(path))
    return [r for r in recs if r.get("kind") == "game"], bad


# ---------------------------------------------------------------------------
# statistics, recomputed by hand
# ---------------------------------------------------------------------------
def _rec(s, won, vp, arm="cand", ms=(10.0, 20.0), opp_ms=(1.0,), trace=None, ours=None):
    return {"kind": "game", "status": "ok", "s": s, "seat": s % 4, "arm": arm, "won": won, "our_vp": vp,
            "opp_vps": [5, 6, 7], "turns": 80, "duration": 1.0, "truncated": False,
            "dec": AB.timing_summary([x / 1000.0 for x in ms]), "opp_dec": AB.timing_summary([x / 1000.0 for x in opp_ms]),
            "adapter": {"errors": 0, "fallback": 1}, "trace": trace or AB.make_trace([f"x{s}-{i}" for i in range(40)]),
            "ours": ours or [[3, "aaaaaa"]]}


def test_pair_stats_by_hand():
    cw = [1, 1, 0, 0, 1, 0, 1, 1]
    dw = [0, 1, 0, 1, 0, 0, 0, 1]
    cv = [10, 10, 7, 6, 10, 5, 10, 10]
    dv = [8, 10, 7, 10, 9, 6, 7, 10]
    pairs = [(_rec(s, bool(cw[s]), cv[s], ms=(10.0, 30.0)), _rec(s, bool(dw[s]), dv[s], arm="def", ms=(10.0, 10.0)))
             for s in range(8)]
    st = AB.pair_stats(pairs)
    diffs = [c - d for c, d in zip(cw, dw)]
    vpd = [c - d for c, d in zip(cv, dv)]
    assert st["pairs"] == 8 and st["cand_wins"] == 5 and st["def_wins"] == 3
    assert st["win_rate_cand"] == pytest.approx(5 / 8) and st["win_rate_def"] == pytest.approx(3 / 8)
    assert st["delta"] == pytest.approx(statistics.mean(diffs))
    assert st["se"] == pytest.approx(statistics.stdev(diffs) / math.sqrt(8))
    assert st["ci95"][0] == pytest.approx(st["delta"] - 1.959964 * st["se"])
    assert st["vp_delta"] == pytest.approx(statistics.mean(vpd))
    assert st["vp_se"] == pytest.approx(statistics.stdev(vpd) / math.sqrt(8))
    assert st["concordance"] == {"both": 2, "cand_only": 3, "def_only": 1, "neither": 2}
    # seat k holds seeds k and k + 4
    for k in range(4):
        sub = [diffs[k], diffs[k + 4]]
        assert st["seats"][str(k)]["n"] == 2
        assert st["seats"][str(k)]["delta"] == pytest.approx(statistics.mean(sub))
    assert st["ms_cand"] == pytest.approx(20.0) and st["ms_def"] == pytest.approx(10.0)
    assert st["extra_ms"] == pytest.approx(10.0)
    assert st["opp_ms_cand"] == pytest.approx(1.0)
    assert st["adapter_cand"]["fallback"] == 8
    assert st["verdict"].startswith("inconclusive")


def test_verdicts_and_identical_pairs():
    better = [(_rec(s, True, 10), _rec(s, s % 3 == 0, 8, arm="def")) for s in range(40)]
    assert AB.pair_stats(better)["verdict"] == "candidate better"
    worse = [(_rec(s, s % 3 == 0, 8), _rec(s, True, 10, arm="def")) for s in range(40)]
    assert AB.pair_stats(worse)["verdict"] == "candidate worse"
    same = [(_rec(s, s % 2 == 0, 9), _rec(s, s % 2 == 0, 9, arm="def")) for s in range(40)]
    st = AB.pair_stats(same)
    assert st["delta"] == 0 and st["se"] == 0 and st["verdict"] == "no detectable difference at 95%"
    assert st["identical"] == 40 and st["diverged"] == 0
    assert AB.should_stop(st, 0.05, 10) is False      # |delta| = 0 never stops
    assert AB.should_stop(AB.pair_stats(better), 0.1, 10) is True
    assert AB.should_stop(AB.pair_stats(better), 0.1, 100) is False   # not enough pairs yet


def test_pool_timing_mean_is_exact():
    a = AB.timing_summary([0.001, 0.003])
    b = AB.timing_summary([0.010])
    pooled = AB.pool_timing([a, b, None, {"n": 0}])
    assert pooled["n"] == 3 and pooled["mean"] == pytest.approx(14.0 / 3)
    assert 8.0 < pooled["p95"] < 12.0          # histogram bin of 10 ms
    assert a["p95"] == pytest.approx(3.0)


def test_divergence_classification():
    base = [f"a{i}" for i in range(64)]
    t_same = AB.make_trace(base)
    assert AB.divergence({"trace": t_same, "ours": []}, {"trace": t_same, "ours": []})["identical"]
    # our decision #2 at action 37 differs and the logs part exactly there -> consistent
    other = base[:37] + [f"b{i}" for i in range(27)]
    ours_c = [[5, "h1"], [20, "h2"], [37, "cand"]]
    ours_d = [[5, "h1"], [20, "h2"], [37, "dflt"]]
    div = AB.divergence({"trace": AB.make_trace(other), "ours": ours_c}, {"trace": t_same, "ours": ours_d})
    assert div == {"known": True, "identical": False, "decision": 2, "at": 37, "diverge_from": 32, "consistent": True}
    # the logs part at action 3, before any catanbot decision differs -> inconsistent (not our doing)
    early = base[:3] + [f"c{i}" for i in range(61)]
    div = AB.divergence({"trace": AB.make_trace(early), "ours": ours_c}, {"trace": t_same, "ours": ours_d})
    assert div["consistent"] is False


# ---------------------------------------------------------------------------
# runs / arms
# ---------------------------------------------------------------------------
def _args(argv):
    return AB.build_parser().parse_args(argv + ["--out", "x.jsonl"])


def test_build_runs_tunable_and_specs():
    from catanbot import tuning
    runs, need_py = AB.build_runs(_args(["--tunable", "danger.TURNS_HALF", "--values", "2,4.5,2", "--opponent", "vf"]))
    assert [r["value"] for r in runs] == [2.0, 4.5] and not need_py          # duplicates dropped
    for r in runs:
        assert r["cand"]["overrides"] == {"danger.TURNS_HALF": r["value"]} and r["def"]["overrides"] == {}
        assert r["cand"]["spec"] == r["def"]["spec"] == tuning.DEFAULT_SEARCH_SPEC
        assert r["cand"]["adapter"] == r["def"]["adapter"] == {"trades": "off"}
    runs, _ = AB.build_runs(_args(["--tunable", "search.opp_roll_samples", "--values", "2", "--opponent", "vf"]))
    assert runs[0]["cand"]["spec"] == tuning.DEFAULT_DEEP_SPEC                # knob only read at depth >= 2
    _, need_py = AB.build_runs(_args(["--tunable", "heuristic.EXPOSURE_WEIGHT", "--values", "0", "--opponent", "vf"]))
    assert need_py                                                           # -> CATANBOT_NO_ACCEL=1 re-exec
    runs, _ = AB.build_runs(_args(["--tunable", "danger.danger_multiplier", "--flag-off", "--opponent", "vf"]))
    assert runs[0]["cand"]["overrides"] == {"danger.danger_multiplier": False}
    runs, _ = AB.build_runs(_args(["--cand-spec", "heuristic:temp=0", "--def-spec", tuning.DEFAULT_SEARCH_SPEC,
                                   "--opponent", "value", "--cand-set", "danger.TURNS_HALF=2", "--trades", "value",
                                   "--cand-adapter-opt", "trades=off"]))
    r = runs[0]
    assert r["cand"]["spec"] == "heuristic:temp=0" and r["cand"]["overrides"] == {"danger.TURNS_HALF": 2.0}
    assert r["cand"]["adapter"] == {"trades": "off"} and r["def"]["adapter"] == {"trades": "value"}
    ctx = {"opponent": "value", "python": "3.11", "catanatron": "3.3.0", "evaluator": "c++", "vps_to_win": 10,
           "discard_limit": 7, "hashseed": "0", "fake": None}
    AB.finalize_runs(runs, ctx, "exp", (0, 10))
    assert r["cand_key"] != r["def_key"]
    # the arm key ignores the experiment / candidate labels but not the opponent or the engine
    same = dict(r["def"], label="whatever")
    assert AB.arm_key(same, ctx) == r["def_key"]
    assert AB.arm_key(r["def"], dict(ctx, opponent="alphabeta")) != r["def_key"]
    assert AB.arm_key(r["def"], dict(ctx, catanatron="3.2.1")) != r["def_key"]
    # an A/A run still plays both arms (the role is part of the key)
    aa, _ = AB.build_runs(_args(["--cand-spec", "heuristic:temp=0", "--def-spec", "heuristic:temp=0", "--opponent", "vf"]))
    AB.finalize_runs(aa, ctx, "aa", (0, 4))
    assert aa[0]["cand_key"] != aa[0]["def_key"]


def test_catanbot_kwargs_follow_the_bench():
    assert AB._catanbot_kwargs({"trades": "off"}) == {"suppress_trades": True}
    assert AB._catanbot_kwargs({"trades": "value", "strict": False}) == {"suppress_trades": False, "strict": False}
    assert AB._catanbot_kwargs({}) == {}


# ---------------------------------------------------------------------------
# in-process run with a monkeypatched game runner
# ---------------------------------------------------------------------------
def test_monkeypatched_runner_pairs_arms(tmp_path, monkeypatch):
    calls = []

    def runner(job, arm, s):
        calls.append((s, arm["role"], dict(arm["overrides"]), AB.seat_of(s), AB.game_seed(s), os.getpid()))
        won = (s % 3 == 0) or (arm["role"] == "cand" and s % 3 == 1)
        body = AB._fake_game(dict(job, ctx=dict(job["ctx"], fake={"effect": 0.0, "crash": [], "die": [], "sleep": 0})),
                             arm, s)
        body.update({"won": won, "our_vp": 10 if won else 6})
        return body

    monkeypatch.setattr(AB, "_real_game", runner)
    args = _args(["--tunable", "danger.TURNS_HALF", "--values", "2,4.5", "--opponent", "vf", "--seeds", "9"])
    runs, _ = AB.build_runs(args)
    ctx = AB.make_context(args)
    out = tmp_path / "r.jsonl"
    AB.finalize_runs(runs, ctx, "mp", (0, 9))
    camp = AB.Campaign(runs, str(out), ctx, "code1", range(9), "mp", timeout=0, quiet=True)
    assert AB.execute(camp, workers=0) == "done"
    # one default game per seed, shared by both candidate values, then one game per candidate
    assert len(calls) == 9 * 3
    for s in range(9):
        mine = [c for c in calls if c[0] == s]
        assert [c[1] for c in mine] == ["def", "cand", "cand"]
        assert {c[3] for c in mine} == {s % 4} and {c[4] for c in mine} == {s + 1}
        assert mine[0][2] == {} and mine[1][2] == {"danger.TURNS_HALF": 2.0} and mine[2][2] == {"danger.TURNS_HALF": 4.5}
    index, _ = AB.load_index(str(out))
    st = AB.run_stats(index, runs[0])
    assert st["pairs"] == 9 and st["cand_wins"] == 6 and st["def_wins"] == 3
    assert st["delta"] == pytest.approx(3 / 9)
    # resume: nothing left to play
    calls.clear()
    camp2 = AB.Campaign(runs, str(out), ctx, "code1", range(9), "mp", timeout=0, quiet=True)
    assert camp2.pending_jobs() == 0 and AB.execute(camp2, workers=0) == "done" and calls == []
    # a code change: complete pairs stay complete, nothing is replayed
    camp3 = AB.Campaign(runs, str(out), ctx, "code2", range(9), "mp", timeout=0, quiet=True)
    assert camp3.pending_jobs() == 0


def test_lone_arm_with_old_code_is_replayed_as_a_pair(tmp_path, monkeypatch):
    played = []

    def runner(job, arm, s):
        played.append((s, arm["role"], job["code"]))
        return AB._fake_game(dict(job, ctx=dict(job["ctx"], fake={"effect": 0.0, "crash": [], "die": [], "sleep": 0})),
                             arm, s)

    monkeypatch.setattr(AB, "_real_game", runner)
    args = _args(["--tunable", "danger.TURNS_HALF", "--values", "2", "--opponent", "vf"])
    runs, _ = AB.build_runs(args)
    ctx = AB.make_context(args)
    AB.finalize_runs(runs, ctx, "lone", (0, 1))
    out = tmp_path / "l.jsonl"
    camp = AB.Campaign(runs, str(out), ctx, "old", [0], "lone", timeout=0, quiet=True)
    job = camp.make_job(0, [dict(runs[0]["def"], key=runs[0]["def_key"])])     # only the default arm (a kill)
    camp.handle(AB.play_job(job))
    played.clear()
    camp2 = AB.Campaign(runs, str(out), ctx, "new", [0], "lone", timeout=0, quiet=True)
    AB.execute(camp2, workers=0)
    assert sorted(played) == [(0, "cand", "new"), (0, "def", "new")]     # the pair is formed within one code


# ---------------------------------------------------------------------------
# subprocess runs with synthetic games: pairing, pinned hash seed, resume, append-only, crashes
# ---------------------------------------------------------------------------
def test_pairing_and_pinned_hashseed_in_workers(tmp_path):
    out = tmp_path / "p.jsonl"
    proc = _fake(out, seeds=8, workers=2)
    assert "re-executing with PYTHONHASHSEED=0" in proc.stdout
    probe = subprocess.run([sys.executable, "-c", f"print(hash({AB.HASH_PROBE_TEXT!r}))"],
                           env=_env(PYTHONHASHSEED="0"), capture_output=True, text=True).stdout.strip()
    games, bad = _games(out)
    assert bad == 0 and len(games) == 16
    by_seed = {}
    for g in games:
        by_seed.setdefault(g["s"], []).append(g)
        assert g["hashseed"] == "0" and str(g["hash_probe"]) == probe
    for s, gs in by_seed.items():
        assert sorted(g["arm"] for g in gs) == ["cand", "def"]
        assert {g["seat"] for g in gs} == {s % 4} and {g["game_seed"] for g in gs} == {s + 1}
        assert len({g["pid"] for g in gs}) == 1              # both arms back to back in one worker
        assert len({g["code"] for g in gs}) == 1
    lines = [json.loads(x) for x in open(out)]
    assert lines[0]["kind"] == "run" and lines[0]["ctx"]["hashseed"] == "0"


def test_resume_skips_finished_seeds_and_is_append_only(tmp_path):
    out = tmp_path / "r.jsonl"
    _fake(out, seeds=6)
    first = out.read_bytes()
    proc = _fake(out, seeds=6)
    assert "0 seed(s) need games" in proc.stdout
    assert out.read_bytes() == first                          # nothing rewritten, nothing added
    proc = _fake(out, seeds=10)
    assert "4 seed(s) need games" in proc.stdout
    data = out.read_bytes()
    assert data.startswith(first)                             # append-only
    games, _ = _games(out)
    assert len(games) == 20 and sorted({g["s"] for g in games}) == list(range(10))
    # a line cut short by a kill is skipped and does not swallow the next record
    with open(out, "ab") as fh:
        fh.write(b'{"kind":"game","arm_key":"trunc')
    proc = _fake(out, seeds=12)
    assert "unparseable" in proc.stdout + proc.stderr
    games, bad = _games(out)
    assert bad == 1 and len(games) == 24
    assert out.read_bytes().startswith(data)
    rep = _run(["--report", "--out", str(out)])
    assert "12 complete" in rep.stdout


def test_crash_isolation_errors_are_recorded(tmp_path):
    out = tmp_path / "c.jsonl"
    proc = _fake(out, seeds=10, extra=["--fake-crash-seeds", "2", "--fake-die-seeds", "5"])
    assert "worker process died" in proc.stderr + proc.stdout
    games, _ = _games(out)
    errs = {(g["s"], g["arm"]): g["error"] for g in games if g["status"] == "error"}
    assert set(errs) == {(2, "cand"), (5, "cand")}
    assert "synthetic crash" in errs[(2, "cand")] and "hard crash" in errs[(5, "cand")]
    assert sum(1 for g in games if g["status"] == "ok") == 18
    index, _ = AB.load_index(str(out))
    st = AB.run_stats(index, next(iter(index.runs.values())))
    assert st["pairs"] == 8 and st["errors"] == 2
    before = out.read_bytes()
    _fake(out, seeds=10, extra=["--fake-crash-seeds", "2", "--fake-die-seeds", "5"])
    assert out.read_bytes() == before                        # errors count as played ...
    _fake(out, seeds=10, extra=["--retry-errors"])           # ... unless asked to retry them
    index, _ = AB.load_index(str(out))
    st = AB.run_stats(index, next(iter(index.runs.values())))
    assert st["pairs"] == 10 and st["errors"] == 0


def test_default_arm_shared_and_reused_across_files(tmp_path):
    a = tmp_path / "a.jsonl"
    _fake(a, seeds=6, values="2,4.5")
    games, _ = _games(a)
    assert len(games) == 6 * 3                                # one default game per seed for both values
    b = tmp_path / "b.jsonl"
    proc = _fake(b, seeds=6, tunable="coalitions.SCALE", values="1", extra=["--reuse", str(tmp_path / "*.jsonl")])
    assert "6 default game(s) reused" in proc.stdout
    games, _ = _games(b)
    reused = [g for g in games if g.get("reused_from")]
    assert len(reused) == 6 and all(g["arm"] == "def" and g["reused_from"] == "a.jsonl" for g in reused)
    index, _ = AB.load_index(str(b))
    assert AB.run_stats(index, next(iter(index.runs.values())))["pairs"] == 6


def test_sequential_stop(tmp_path):
    out = tmp_path / "s.jsonl"
    proc = _fake(out, seeds=400, workers=1, effect=0.6,
                 extra=["--stop-at-se", "0.08", "--stop-min-pairs", "20"])
    assert "sequential stop" in proc.stderr
    recs, _ = AB.read_jsonl(str(out))
    stops = [r for r in recs if r.get("kind") == "stop"]
    assert len(stops) == 1 and stops[0]["pairs"] >= 20
    games = [r for r in recs if r.get("kind") == "game"]
    assert len(games) < 2 * 400
    assert "STOPPED EARLY" in _run(["--report", "--out", str(out)]).stdout


def test_deadline_stops_submitting(tmp_path):
    out = tmp_path / "d.jsonl"
    proc = _fake(out, seeds=50, workers=1, extra=["--fake-sleep", "0.05", "--max-minutes", "0.005"])
    assert "deadline" in proc.stdout and "rerun the same command to resume" in proc.stdout
    games, _ = _games(out)
    assert 0 < len(games) < 100


# ---------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------
def test_campaign_runs_resumes_and_summarises(tmp_path):
    camp = _load("campaign_under_test", CAMPAIGN)
    plan = tmp_path / "plan.json"
    d = tmp_path / "runs"
    plan.write_text(json.dumps({
        "defaults": {"seeds": {"count": 8, "base": 0}, "workers": 2, "extra_args": ["--fake-games", "0.2", "--quiet"]},
        "experiments": [
            {"name": "turns@vf", "interpreter": "py321", "opponent": "vf", "tunable": "danger.TURNS_HALF",
             "values": [2, 4.5], "priority": 2},
            {"name": "broken@vf", "interpreter": "py321", "opponent": "vf", "tunable": "no.such_tunable", "priority": 1},
            {"name": "depth2@vf", "interpreter": "py321", "opponent": "vf", "priority": 3,
             "cand_spec": "search:depth=2,beam=4,expand=8,evaluator=heuristic",
             "def_spec": "search:depth=1,beam=4,expand=8,evaluator=heuristic"},
            {"name": "off@vf", "interpreter": "py321", "opponent": "vf", "tunable": "danger.danger_multiplier",
             "flag_off": True, "priority": 4},
        ]}))
    interp = {"py321": sys.executable}
    raw = json.loads(plan.read_text())
    raw["interpreters"] = interp
    plan.write_text(json.dumps(raw))
    cmd = [sys.executable, CAMPAIGN, "--plan", str(plan), "--dir", str(d)]
    proc = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 1, proc.stdout + proc.stderr          # the broken experiment failed ...
    assert "[turns@vf] exit 0" in proc.stdout and "[off@vf] exit 0" in proc.stdout   # ... the others ran
    assert proc.stdout.index("[broken@vf]") < proc.stdout.index("[turns@vf]")       # priority order
    summary = (d / "SUMMARY.md").read_text()
    rows = [line for line in summary.splitlines() if line.startswith("| ") and "experiment" not in line]
    assert len(rows) == 5                                  # 2 + 1 + 1 candidates + the failed experiment
    assert any("FAILED" in r and "broken@vf" in r for r in rows)
    assert any("`danger.TURNS_HALF=4.5`" in r and "8 / 8" in r and "done" in r for r in rows)
    # the default games of off@vf were reused from turns@vf (same default arm)
    games, _ = _games(d / "off@vf.jsonl")
    assert sum(1 for g in games if g.get("reused_from")) == 8
    snapshot = {p.name: p.read_bytes() for p in d.glob("*.jsonl")}
    proc = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=600)
    assert "[turns@vf] complete" in proc.stdout
    assert {p.name: p.read_bytes() for p in d.glob("*.jsonl")} == snapshot    # idempotent
    # status in-process on the JSONL files
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert camp.main(["--plan", str(plan), "--dir", str(d), "--status"]) == 0
    text = buf.getvalue()
    assert "3 complete" in text and "FAILED(1)" in text and "2:8, 4.5:8" in text
    exps, _ = camp.load_plan(str(plan), 2)
    prog = camp.progress(next(e for e in exps if e["name"] == "turns@vf"), str(d))
    assert prog["complete"] and prog["games"] == 24 and prog["remaining"] == 0 and prog["rate"]


def test_campaign_plan_validation_and_command(tmp_path):
    camp = _load("campaign_under_test2", CAMPAIGN)
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps([{"name": "x", "interpreter": "py330", "opponent": "value", "tunable": "t",
                                 "seed": 5}]))
    with pytest.raises(SystemExit, match="unknown field"):
        camp.load_plan(str(plan), 2)
    plan.write_text(json.dumps([
        {"name": "v", "interpreter": "py330", "opponent": "alphabeta", "tunable": "placement.RESOURCE_DEMAND",
         "values": [[1, 1, 1, 1, 1], [0.9, 0.9, 0.8, 1.4, 1.4]], "workers": 8, "seeds": 400, "trades": "value",
         "opponent_params": {"depth": 1}, "cand_set": {"danger.TURNS_HALF": 2}, "stop_at_se": 0.02}]))
    exps, interps = camp.load_plan(str(plan), 2)
    e = exps[0]
    assert e["workers"] == 2 and e["seeds"] == {"count": 400, "base": 0}
    cmd = camp.command(e, str(tmp_path), interps, 15.0)
    assert cmd[0] == interps["py330"]
    joined = " ".join(cmd)
    assert "--values 1/1/1/1/1;0.9/0.9/0.8/1.4/1.4" in joined and "--trades value" in joined
    assert "--opponent-params depth=1" in joined and "--cand-set danger.TURNS_HALF=2" in joined
    assert "--max-minutes 15.00" in joined and "--stop-at-se 0.02" in joined and "--workers 2" in joined
    assert camp.expected_candidates(e) == 2


# ---------------------------------------------------------------------------
# real games (the determinism the pairing rests on)
# ---------------------------------------------------------------------------
def test_real_aa_pair_is_identical(tmp_path):
    pytest.importorskip("catanatron")
    out = tmp_path / "real.jsonl"
    _run(["--tunable", "search.trade_proposals", "--values", "0", "--opponent", "random", "--seeds", "2",
          "--workers", "1", "--out", str(out), "--quiet"], timeout=600)
    games, _ = _games(out)
    assert len(games) == 4 and all(g["status"] == "ok" for g in games)
    for s in (0, 1):
        c = next(g for g in games if g["s"] == s and g["arm"] == "cand")
        d = next(g for g in games if g["s"] == s and g["arm"] == "def")
        # trades are off against catanatron, so trade_proposals cannot change a decision: same game
        assert c["trace"] == d["trace"] and c["ours"] == d["ours"] and c["vps"] == d["vps"]
        assert c["seat"] == d["seat"] == s % 4 and c["game_seed"] == d["game_seed"] == s + 1
        assert c["dec"]["n"] > 0 and c["opp_dec"]["n"] >= 0 and "errors" in c["adapter"]
        assert c["hashseed"] == "0" and c["evaluator"]
