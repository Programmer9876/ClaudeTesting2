"""Tests for the 2^k factorial tests: catanbot/factorial.py (design and arithmetic), scripts/ablate.py
--factorial (self-play) and scripts/factorial_catanatron.py (against Catanatron).

The arithmetic is checked by hand and on mock runners with a planted interaction; the scripts run on a
monkeypatched game runner, synthetic games (--fake-games) and two tiny real smokes.
"""
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

from catanbot import factorial as F
from catanbot import tuning

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATE = os.path.join(ROOT, "scripts", "ablate.py")
FCT = os.path.join(ROOT, "scripts", "factorial_catanatron.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FC = _load("factorial_catanatron_under_test", FCT)


def _run(script, args, python=None, timeout=300):
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONHASHSEED", None)
    env.pop("CATANBOT_NO_ACCEL", None)
    return subprocess.run([python or sys.executable, script] + args, cwd=ROOT, env=env, text=True,
                          capture_output=True, timeout=timeout)


# ---------------------------------------------------------------------------
# design and arithmetic
# ---------------------------------------------------------------------------
def test_contrast_weights_by_hand():
    assert F.cells(2) == [0, 1, 2, 3] and F.effects(2) == [1, 2, 3]
    assert F.effects(3) == [1, 2, 4, 3, 5, 6, 7]                     # main effects, 2-way, 3-way
    # 2x2: main A = ((A + AB) - (base + B)) / 2; interaction = AB - A - B + base
    assert F.weights(2, 1) == {0: -0.5, 1: 0.5, 2: -0.5, 3: 0.5}
    assert F.weights(2, 2) == {0: -0.5, 1: -0.5, 2: 0.5, 3: 0.5}
    assert F.weights(2, 3) == {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0}
    # 2^3: main effects over 4 differences, 2-way over 2, the 3-way undivided
    assert set(abs(w) for w in F.weights(3, 1).values()) == {0.25}
    assert set(abs(w) for w in F.weights(3, 3).values()) == {0.5}
    assert F.weights(3, 7)[7] == 1.0 and F.weights(3, 7)[0] == -1.0 and F.weights(3, 7)[1] == 1.0
    for k in (2, 3, 4):
        for e in F.effects(k):
            assert sum(F.weights(k, e).values()) == 0.0                 # every contrast ignores a common shift


def test_parse_factors():
    fs = F.parse_factors("danger.danger_multiplier,danger.steal_factor=off,devcards.KNIGHT_VALUE=0.8")
    assert fs == [("danger.danger_multiplier", False), ("danger.steal_factor", False), ("devcards.KNIGHT_VALUE", 0.8)]
    assert F.cell_overrides(fs, 5) == {"danger.danger_multiplier": False, "devcards.KNIGHT_VALUE": 0.8}
    assert F.cell_label(fs, 0) == "base" and F.cell_label(fs, 3) == "danger_multiplier=off + steal_factor=off"
    assert F.effect_label(fs, 6) == "steal_factor=off x KNIGHT_VALUE=0.8"
    assert F.parse_factors("TURNS_HALF=2,search.trade_proposals=0")[1] == ("search.trade_proposals", 0)
    # politics / table-social terms get the politics-rule note (docs/ABLATIONS.md); the robber terms do not
    note = F.politics_note(["politics.MAX_SLACK", "coalitions.SCALE", "danger.TURNS_HALF"])
    assert "politics.MAX_SLACK, coalitions.SCALE" in note and "INCONCLUSIVE" in note
    assert F.politics_note(["devcards.KNIGHT_VALUE", "heuristic.EXPOSURE_WEIGHT", "search.trade_proposals"]) is None
    assert all(F.is_politics(n) for n in ("trading.feed_leader_guard", "search.counters", "search.counter_margin",
                                          "search.respond_lookahead", "opponent_model.stage_late_drop"))
    for text, msg in (("danger.TURNS_HALF=3,danger.BLOCK_NEED=0", "is its default"),
                      ("danger.TURNS_HALF=2", "at least 2 factors"),
                      ("danger.TURNS_HALF,danger.BLOCK_NEED=0", "danger.TURNS_HALF=VALUE"),
                      ("danger.TURNS_HALF=2,TURNS_HALF=4", "twice"),
                      ("nope.X=1,danger.TURNS_HALF=2", "unknown tunable"),
                      ("a.TURNS_HALF=2", "unknown tunable"),
                      (",".join(f"{n}=off" for n in ("danger.danger_multiplier", "danger.steal_factor",
                                                     "danger.rob_break", "trading.feed_leader_guard")) +
                       ",danger.TURNS_HALF=2", "at most 4")):
        with pytest.raises((ValueError, KeyError), match=msg):
            F.parse_factors(text)


def _planted(units, a, b, i, noise=0.3, board=1.0, seed=1):
    """Absolute per-unit values y = board_u + a A + b B + i A B + eps (0/1 coding of the factors)."""
    rng = random.Random(seed)
    vals = {c: [] for c in F.cells(2)}
    for _ in range(units):
        u = rng.gauss(0, board)
        for c in F.cells(2):
            A, B = c & 1, c >> 1 & 1
            vals[c].append(u + a * A + b * B + i * A * B + rng.gauss(0, noise))
    return vals


def test_analyse_recovers_a_planted_interaction():
    fs = [("devcards.KNIGHT_VALUE", 0.8), ("heuristic.EXPOSURE_WEIGHT", 0.0)]
    a, b, i = 0.05, -0.03, 0.08
    vals = _planted(4000, a, b, i)
    res = F.analyse(fs, vals)
    eff = {r["effect"]: r for r in res["effects"]}
    for e, truth in ((1, a + i / 2), (2, b + i / 2), (3, i)):      # main effects average over the other factor
        assert abs(eff[e]["mean"] - truth) < 3 * eff[e]["se"], (e, eff[e]["mean"], truth)
    # the shared board luck (s.d. 1.0) cancels: the s.e. reflects the per-cell noise only
    assert eff[3]["se"] < 2 * 0.3 * 4 ** 0.5 / 4000 ** 0.5
    assert eff[3]["se"] == pytest.approx(2 * eff[1]["se"], rel=0.1)   # interaction ~2x the s.e. (4x the games)
    assert eff[3]["reading"] == "positive" and eff[3]["order"] == 2 and eff[3]["p"] < 1e-6
    cells = {r["cell"]: r for r in res["cells"]}
    assert abs(cells[3]["mean"] - (a + b + i)) < 3 * cells[3]["se"]
    add = res["additivity"][0]
    assert add["observed"] - add["predicted"] == pytest.approx(eff[3]["mean"])
    # the self-play form: cells already paired against the base, no base cell -> the same effects exactly
    vs = {c: [v - b0 for v, b0 in zip(vals[c], vals[0])] for c in (1, 2, 3)}
    res2 = F.analyse(fs, vs)
    for r1, r2 in zip(res["effects"], res2["effects"]):
        assert r1["mean"] == pytest.approx(r2["mean"]) and r1["se"] == pytest.approx(r2["se"])
    # no interaction planted -> none found (at this seed), main effects still right
    res3 = F.analyse(fs, _planted(4000, a, b, 0.0, seed=2))
    e3 = {r["effect"]: r for r in res3["effects"]}
    assert e3[3]["reading"] == "not significant" and abs(e3[1]["mean"] - a) < 3 * e3[1]["se"]
    assert "inconclusive" in F.analyse(fs, _planted(10, a, b, i))["effects"][2]["reading"]
    lines = F.format_report(res)
    assert any("2-way" in x and "KNIGHT_VALUE=0.8 x EXPOSURE_WEIGHT=0" in x for x in lines)


# ---------------------------------------------------------------------------
# scripts/ablate.py --factorial (self-play)
# ---------------------------------------------------------------------------
def test_ablate_factorial_on_a_mock_runner_with_a_planted_interaction(tmp_path, monkeypatch):
    ab = _load("ablate_for_factorial_test", ABLATE)
    effect = {frozenset(["danger.TURNS_HALF"]): 0.1, frozenset(["danger.danger_multiplier"]): 0.0,
              frozenset(["danger.TURNS_HALF", "danger.danger_multiplier"]): 0.3}
    calls = []

    def fake_run_paired(base_spec, overrides, games, seed, num_players, workers=1, max_turns=400, progress=None,
                        allow_counters=False):
        calls.append((base_spec, dict(overrides), games, seed))
        out = []
        for pattern, gseed in tuning.paired_jobs(games, seed, num_players):
            u = random.Random(gseed).random()                       # common to every cell (same board / dice)
            cand = u < 0.5 + effect[frozenset(overrides)]
            winner = pattern.index("C") if cand else pattern.index("D")
            out.append({"seed": gseed, "pattern": pattern, "winner": winner,
                        "vps": [10 if i == winner else 6 for i in range(4)], "turns": 80, "actions": 500,
                        "duration": 0.0, "seat_times": [[0.001]] * 4, "seat_decisions": [1] * 4})
        return out

    monkeypatch.setattr(ab.tuning, "run_paired", fake_run_paired)
    out = tmp_path / "fx.json"
    with redirect_stdout(io.StringIO()) as buf:
        rc = ab.main(["--factorial", "danger.TURNS_HALF=2,danger.danger_multiplier", "--games", "600", "--seed", "4",
                      "--json", str(out), "--quiet"])
    assert rc == 0
    assert [c[1] for c in calls] == [{"danger.TURNS_HALF": 2.0}, {"danger.danger_multiplier": False},
                                     {"danger.TURNS_HALF": 2.0, "danger.danger_multiplier": False}]
    assert {c[0] for c in calls} == {tuning.DEFAULT_CHEAP_SPEC} and {c[3] for c in calls} == {4}
    d = json.loads(out.read_text())
    eff = {r["effect"]: r for r in d["win"]["effects"]}
    for e, truth in ((1, (0.1 + 0.3 - 0.0) / 2), (2, (0.0 + 0.3 - 0.1) / 2), (3, 0.3 - 0.1 - 0.0)):
        assert abs(eff[e]["mean"] - truth) < 3 * eff[e]["se"] + 1e-9, (e, eff[e]["mean"], truth)
    assert eff[3]["reading"] == "positive" and d["win"]["units"] == 600
    # the per-game contrast recomputed from the stored records
    recs = {c["cell"]: c["records"] for c in d["cells"]}
    contrast = [recs[3][g]["diff"] - recs[1][g]["diff"] - recs[2][g]["diff"] for g in range(600)]
    assert eff[3]["mean"] == pytest.approx(sum(contrast) / 600)
    assert "2-way  TURNS_HALF=2 x danger_multiplier=off" in buf.getvalue()


def test_ablate_factorial_plan_errors_and_real_smoke(tmp_path):
    p = _run(ABLATE, ["--factorial", "devcards.KNIGHT_VALUE=0.8,heuristic.EXPOSURE_WEIGHT=0", "--plan", "--games", "6"])
    assert p.returncode == 0, p.stderr
    assert "re-executing with CATANBOT_NO_ACCEL=1" in p.stdout or "python" in p.stdout
    assert "base spec search:depth=1,beam=4,expand=8,evaluator=heuristic" in p.stdout
    assert p.stdout.count("  cell ") == 3 and "(= 18 games" in p.stdout
    p = _run(ABLATE, ["--factorial", "danger.TURNS_HALF=3,danger.BLOCK_NEED=1"])
    assert p.returncode == 2 and "is its default" in p.stderr
    p = _run(ABLATE, ["--factorial", "search.beam=2,danger.BLOCK_NEED=1", "--base-spec", "heuristic", "--plan"])
    assert p.returncode != 0 and "search base spec" in (p.stdout + p.stderr)
    p = _run(ABLATE, ["--factorial", "danger.BLOCK_NEED=1,danger.TURNS_HALF=2", "--tunable", "danger.TURNS_HALF"])
    assert p.returncode == 2
    out = tmp_path / "real.json"
    p = _run(ABLATE, ["--factorial", "danger.danger_multiplier,danger.steal_factor=off", "--games", "2", "--json",
                      str(out), "--quiet"])
    assert p.returncode == 0, p.stderr[-2000:]
    d = json.loads(out.read_text())
    assert [c["cell"] for c in d["cells"]] == [1, 2, 3] and d["base_spec"] == tuning.DEFAULT_CHEAP_SPEC
    assert len({tuple(r["seed"] for r in c["records"]) for c in d["cells"]}) == 1      # the same seeds per cell
    recs = {c["cell"]: c["records"] for c in d["cells"]}
    inter = sum(recs[3][g]["diff"] - recs[1][g]["diff"] - recs[2][g]["diff"] for g in range(2)) / 2
    assert d["win"]["effects"][2]["mean"] == pytest.approx(inter) and d["win"]["units"] == 2
    assert "smoke size" in p.stdout


# ---------------------------------------------------------------------------
# scripts/factorial_catanatron.py
# ---------------------------------------------------------------------------
def test_factorial_catanatron_monkeypatched_runner_planted_interaction(tmp_path, monkeypatch):
    AB = FC.AB
    p_win = {(): 0.2, ("devcards.KNIGHT_VALUE",): 0.3, ("danger.steal_factor",): 0.3,
             ("danger.steal_factor", "devcards.KNIGHT_VALUE"): 0.6}          # interaction 0.6 - 0.3 - 0.3 + 0.2
    calls = []

    def runner(job, arm, s):
        cell = tuple(sorted(arm["overrides"]))
        calls.append((s, arm["role"], cell))
        body = AB._fake_game(dict(job, ctx=dict(job["ctx"], fake={"effect": 0.0, "crash": [], "die": [], "sleep": 0})),
                             arm, s)
        won = random.Random(f"luck-{s}").random() < p_win[cell]              # the same luck in every cell
        body.update({"won": won, "our_vp": 10 if won else 6})
        return body

    monkeypatch.setattr(AB, "_real_game", runner)
    args = FC.build_parser().parse_args(["--factorial", "devcards.KNIGHT_VALUE=0.8,danger.steal_factor", "--opponent",
                                         "vf", "--seeds", "400", "--out", str(tmp_path / "x.jsonl")])
    runs, need_python, factors = FC.build_runs(args)
    assert len(runs) == 3 and not need_python and len({r["def"]["label"] for r in runs}) == 1
    ctx = AB.make_context(args)
    out = str(tmp_path / "x.jsonl")
    AB.finalize_runs(runs, ctx, "fx", (0, 400))
    assert len({r["def_key"] for r in runs}) == 1 and len({r["cand_key"] for r in runs}) == 3
    camp = AB.Campaign(runs, out, ctx, "code1", range(400), "fx", timeout=0, quiet=True)
    assert AB.execute(camp, workers=0) == "done"
    assert len(calls) == 400 * 4                                             # one base game per seed, shared
    assert [c[1] for c in calls[:4]] == ["def", "cand", "cand", "cand"]
    index, _ = AB.load_index(out)
    groups = FC.factorial_groups(index)
    assert len(groups) == 1 and sorted(groups[0][1]) == [1, 2, 3]
    res = FC.factorial_stats(index, *groups[0])
    assert res["seeds"] == 400 and res["dropped"] == 0
    eff = {r["effect"]: r for r in res["win"]["effects"]}
    assert abs(eff[3]["mean"] - 0.2) < 3 * eff[3]["se"] and eff[3]["reading"] == "positive"
    assert abs(eff[1]["mean"] - (0.3 + 0.6 - 0.2 - 0.3) / 2) < 3 * eff[1]["se"]
    # resume: nothing left to play; an error record drops that seed from the factorial only
    calls.clear()
    camp2 = AB.Campaign(runs, out, ctx, "code1", range(400), "fx", timeout=0, quiet=True)
    assert camp2.pending_jobs() == 0 and AB.execute(camp2, workers=0) == "done" and calls == []
    AB.JsonlWriter(out).append([AB._error_record({"s": 7, "ctx": ctx, "exp": "fx", "code": "code1"},
                                                 dict(runs[1]["cand"], key=runs[1]["cand_key"]), "boom")])
    index, _ = AB.load_index(out)
    assert FC.factorial_stats(index, *FC.factorial_groups(index)[0])["seeds"] == 400   # ok pair preferred
    buf = io.StringIO()
    FC.print_factorials(index, out=buf)
    assert "factorial 2^2" in buf.getvalue() and "2-way  KNIGHT_VALUE=0.8 x steal_factor=off" in buf.getvalue()


def test_factorial_catanatron_cli_synthetic_games(tmp_path):
    out = tmp_path / "fx.jsonl"
    p = _run(FCT, ["--factorial", "danger.danger_multiplier,danger.steal_factor", "--opponent", "vf", "--seeds", "8",
                   "--workers", "2", "--out", str(out), "--fake-games", "0.2", "--quiet", "--report-json",
                   str(tmp_path / "r.json")])
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "4 arms (base + 3 cells) x 8 seeds = 32 games planned" in p.stdout
    assert "8 seed(s) with every cell and the base complete" in p.stdout
    r = json.loads((tmp_path / "r.json").read_text())
    assert len(r) == 1 and r[0]["seeds"] == 8 and len(r[0]["win"]["effects"]) == 3
    p = _run(FCT, ["--report", "--out", str(out)])                        # statistics from the file alone
    assert p.returncode == 0 and "factorial 2^2" in p.stdout
    p = _run(os.path.join(ROOT, "scripts", "ablate_catanatron.py"), ["--report", "--out", str(out)])
    assert p.returncode == 0 and p.stdout.count("vs  base (every factor at its default)") == 3
    p = _run(FCT, ["--factorial", "danger.danger_multiplier,danger.steal_factor", "--opponent", "vf", "--seeds", "8",
                   "--out", str(out), "--fake-games", "0.2", "--plan"])
    assert p.returncode == 0 and p.stdout.count("8/8 seeds complete") == 3


def test_factorial_catanatron_real_smoke(tmp_path):
    pytest.importorskip("catanatron")
    from importlib.metadata import version
    opponent = "value" if tuple(int(x) for x in version("catanatron").split(".")[:2]) >= (3, 3) else "random"
    out = tmp_path / "real.jsonl"
    p = _run(FCT, ["--factorial", "devcards.KNIGHT_VALUE=0.8,danger.steal_factor", "--opponent", opponent, "--seeds",
                   "1", "--workers", "1", "--out", str(out), "--quiet"], timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    games = [json.loads(x) for x in out.read_text().splitlines() if '"kind":"game"' in x]
    assert len(games) == 4 and all(g["status"] == "ok" and g["hashseed"] == "0" for g in games)
    assert len({g["game_seed"] for g in games}) == 1 and len({g["seat"] for g in games}) == 1
    assert "1 seed(s) with every cell and the base complete" in p.stdout
