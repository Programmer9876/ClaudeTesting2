"""Tests for scripts/decision_shadow.py, the zero-game decision shadow (generalised from winpaths_shadow.py).

One short self-play game per test (one process, niced by the caller); the proof-log source replays one archived
game of the engine generation this interpreter runs (T1 on catanatron 3.3, R1 on 3.2.1).
"""
import glob
import importlib.util
import json
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DS = _load("decision_shadow_under_test", os.path.join(SCRIPTS, "decision_shadow.py"))
SPEC = "search:depth=1,beam=4,expand=8,evaluator=heuristic"


def test_candidate_parsing():
    c = DS.parse_candidate("mix:danger.BLOCK_NEED=0,expand=4,danger.danger_multiplier=off,search.beam=2")
    assert c.label == "mix" and c.overrides == {"danger.BLOCK_NEED": 0.0, "danger.danger_multiplier": False,
                                                 "search.beam": 2}
    assert c.spec_keys == {"expand": "4"}
    with pytest.raises(SystemExit):
        DS.parse_candidate("nolabel")
    with pytest.raises(SystemExit):
        DS.parse_candidate("x:no.such_tunable=1")


def test_shadow_identity_zero_changes_and_game_unchanged():
    from catanbot.selfplay import make_bot, play_game
    cands = [DS.Candidate("aa", {}, {}), DS.parse_candidate("dm:danger.danger_multiplier=off"),
             DS.parse_candidate("e4:expand=4")]
    rows, games = DS.run_selfplay(SPEC, cands, ["robber", "knight"], 1, 7301, 400, log=open(os.devnull, "w"))
    assert rows and all(set(r["cls"]) & {"robber", "knight"} for r in rows)
    summ = DS.summarize(rows, cands)
    assert summ["aa"]["changed"] == 0 and summ["aa"]["n"] == len(rows)
    assert summ["e4"]["n"] == len(rows) and summ["e4"]["ms_ratio_mean"] is not None
    # the shadow never touches the game's random stream: the same seed without it is the same game
    bots = [make_bot(SPEC) for _ in range(4)]
    res = play_game(bots, rng=random.Random(7301), seed=7301, max_turns=400)
    g = games[0]
    assert (res.winner, list(res.vps), res.turns, res.actions) == (g["winner"], g["vps"], g["turns"], g["actions"])


def test_gate_share_from_rows_and_counts():
    rows = [{"cls": ["robber"], "def": ["a"], "cand": {"x": {"a": ["a"]}}},
            {"cls": ["roll", "knight"], "def": ["a"], "cand": {"x": {"a": ["b"]}}},
            {"cls": ["main"], "def": ["a"], "cand": {"x": {"a": ["b"]}}}]
    res = {"rows": rows}
    assert DS.gate_share(res, "x", ["robber", "knight"]) == (0.5, 2)
    assert DS.gate_share(res, "x", ["main"]) == (1.0, 1)
    summ = DS.summarize([dict(r, ms_def=1.0, cand={"x": dict(r["cand"]["x"], ms=1.0)}) for r in rows],
                        [DS.Candidate("x", {}, {})])
    assert DS.gate_share({"candidates": summ}, "x", ["robber", "knight"]) == (0.5, 2)
    assert DS.gate_share({"candidates": summ}, "y", ["main"]) is None


def test_main_every_samples_main_phase_only():
    cands = [DS.Candidate("aa", {}, {})]
    rows, _ = DS.run_selfplay(SPEC, cands, ["main", "robber"], 1, 7302, 60, log=open(os.devnull, "w"),
                              main_every=4)
    rows_all, _ = DS.run_selfplay(SPEC, cands, ["main", "robber"], 1, 7302, 60, log=open(os.devnull, "w"))
    n_main = sum(1 for r in rows if set(r["cls"]) <= {"main", "buy_dev"})
    n_main_all = sum(1 for r in rows_all if set(r["cls"]) <= {"main", "buy_dev"})
    assert 0 < n_main < n_main_all and abs(n_main - n_main_all / 4) <= 4
    assert sum(1 for r in rows if "robber" in r["cls"]) == sum(1 for r in rows_all if "robber" in r["cls"])


def test_cli_json_and_aa_exit_code(tmp_path):
    out = tmp_path / "s.json"
    rc = DS.main(["--games", "1", "--seed", "7303", "--phases", "robber", "--max-turns", "120",
                  "--cand", "bn0:danger.BLOCK_NEED=0", "--json", str(out), "--keep-rows"])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["aa_ok"] is True and "bn0" in doc["candidates"] and doc["rows"] is not None
    assert DS.gate_share(doc, "bn0", ["robber"])[1] == doc["candidates"]["bn0"]["by_class"].get("robber", {}).get("n", 0)


def test_proof_source_replays_one_archived_game():
    ad = pytest.importorskip("catanbot.bench.catanatron_adapter")
    pat = "T1" if ad.API_33 else "R1"
    paths = sorted(glob.glob(os.path.join(ROOT, "proof", pat, "logs", "*.jsonl.gz")))
    if not paths:
        pytest.skip("proof archive not present")
    cands = [DS.Candidate("aa", {}, {}), DS.parse_candidate("bn0:danger.BLOCK_NEED=0")]
    rows, games = DS.run_proof(SPEC, cands, ["robber", "knight"], paths[:1], 1, log=open(os.devnull, "w"))
    assert len(games) == 1 and games[0]["errors"] == 0
    assert rows and DS.summarize(rows, cands)["aa"]["changed"] == 0
