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


def test_settle_class_port_tags_and_summary():
    """The ports screen's class (docs/PRIORITY_PLAN.md step 4): setup settlement placements and decisions with at
    least two legal settlement spots; settle rows record the port under each arm's chosen settlement."""
    from catanbot import actions as A
    from catanbot import board as B
    from catanbot import engine as E
    from catanbot.state import new_game
    s = new_game(4, rng=random.Random(3))
    legal = E.legal_actions(s)
    assert "settle" in DS.classes_of(s, legal) and "setup" in DS.classes_of(s, legal)
    m = s.copy()
    m.phase = "main"
    assert "settle" not in DS.classes_of(m, [(A.BUILD_SETTLEMENT, 1), (A.END_TURN,)])
    assert "settle" in DS.classes_of(m, [(A.BUILD_SETTLEMENT, 1), (A.BUILD_SETTLEMENT, 7), (A.END_TURN,)])
    v_port = next(iter(s.ports))
    v_land = next(v for v in range(B.NUM_VERTICES) if v not in s.ports)
    assert DS.port_tag(s, (A.SETUP_SETTLEMENT, v_port)) == B.PORT_NAMES[s.ports[v_port]]
    assert DS.port_tag(s, [A.BUILD_SETTLEMENT, v_land]) == "" and DS.port_tag(s, (A.END_TURN,)) is None
    rows = [{"cls": ["setup", "settle"], "def": ["x"], "ms_def": 1.0, "cand": {"c": {"a": ["y"], "ms": 1.0}},
             "ports": {"def": "", "c": "3:1"}},
            {"cls": ["main", "settle"], "def": ["x"], "ms_def": 1.0, "cand": {"c": {"a": ["x"], "ms": 1.0}},
             "ports": {"def": "wheat", "c": "wheat"}},
            {"cls": ["main"], "def": ["x"], "ms_def": 1.0, "cand": {"c": {"a": ["x"], "ms": 1.0}}}]
    sp = DS.summarize(rows, [DS.Candidate("c", {}, {})])["c"]["settle_ports"]
    assert sp["setup"]["def"] == {"n": 1, "settled": 1, "port": 0, "generic": 0}
    assert sp["setup"]["cand"] == {"n": 1, "settled": 1, "port": 1, "generic": 1}
    assert sp["main"]["def"]["port"] == sp["main"]["cand"]["port"] == 1
    assert "settle_ports" not in DS.summarize(rows[2:], [DS.Candidate("c", {}, {})])["c"]


# ---------------------------------------------------------------------------
# robber classifier and gates (docs/PRIORITY_PLAN.md step 5; --robber-gates)
# ---------------------------------------------------------------------------
def _rrow(cls, d, c, rob, ms=(1.0, 1.0), rst=None):
    got = {"a": c, "ms": ms[1]}
    if rst:
        got["rstats"] = rst
    return {"cls": cls, "def": d, "ms_def": ms[0], "cand": {"x": got}, "rob": rob}


def _mv(t1, knight=False, leader=True, opp=5.0, need=0.5, later=False):
    return {"kick_t1": t1, "kick_any": t1 or later, "on_leader": leader, "opp_pv": opp, "p_need": need,
            "knight": knight}


def test_robber_gates_on_synthetic_rows():
    R, K, M = ["move_robber", 3, 1], ["move_robber", 4, 2], ["end_turn"]
    hit = {"def": _mv(True), "cand": {"x": _mv(False)}, "holder": False, "blocked": 0.0}
    hit_same = {"def": _mv(True), "cand": {"x": _mv(True)}, "holder": False, "blocked": 0.0}
    other = {"def": _mv(False), "cand": {"x": _mv(False)}, "holder": False, "blocked": 0.0}
    plain = {"def": None, "cand": {"x": None}, "holder": False, "blocked": 0.0}
    late = {"def": _mv(False, later=True), "cand": {"x": _mv(False)}, "holder": False, "blocked": 0.0}
    rows = [_rrow(["robber"], R, K, hit), _rrow(["robber"], R, R, hit_same), _rrow(["robber"], R, R, hit_same),
            _rrow(["robber"], K, K, other), _rrow(["knight", "roll"], R, K, late)] + \
        [_rrow(["main"], M, M, plain) for _ in range(99)] + [_rrow(["main"], M, ["buy_dev"], plain)]
    g = DS.robber_gates(rows, "x", "persistence")
    assert g["gates"]["G1"]["n"] == 3 and g["gates"]["G1"]["changed"] == 1 and g["gates"]["G1"]["pass"]
    assert g["gates"]["G2"]["n"] == 1 and g["gates"]["G2"]["pass"]
    assert g["gates"]["G3"]["n"] == 100 and g["gates"]["G3"]["share"] == 0.01 and g["gates"]["G3"]["pass"]
    assert g["gates"]["G4"]["pass"] and g["pass"] and g["fallback_trigger"] is False
    assert g["readout"]["kick_later"] == {"n": 1, "changed": 1, "share": 1.0}   # t >= 2 holders: not gated
    slow = [dict(r, ms_def=1.0, cand={"x": dict(r["cand"]["x"], ms=1.3)}) for r in rows]
    g = DS.robber_gates(slow, "x", "persistence")
    assert not g["gates"]["G4"]["pass"] and not g["pass"] and g["fallback_trigger"] is True
    inert = [dict(r, cand={"x": dict(r["cand"]["x"], a=r["def"])}) for r in rows]
    assert DS.robber_gates(inert, "x", "persistence")["fallback_trigger"] is True    # G1 fails: inert
    # insurance: holders' decisions, P_hit median, centring from the candidates' robber_eval counters
    hold = lambda p: {"def": None, "cand": {"x": None}, "holder": True, "blocked": 0.0, "p_hit": p, "d": 5.0}  # noqa
    rst = {"ins_n": 10, "ins_sum_c": 0.2, "ins_sum_dp": 21.0, "evals": 10, "nonzero": 10, "kick_sets": 0}
    rows = [_rrow(["roll", "knight"], ["play_knight", 1, 2], ["roll"], hold(0.9), rst=rst),      # kept, exposed
            _rrow(["roll", "knight"], ["play_knight", 1, 2], ["roll"], hold(0.8), rst=rst),      # kept, exposed
            _rrow(["roll", "knight"], ["roll"], ["roll"], hold(0.1), rst=rst),
            _rrow(["main", "knight"], ["end_turn"], ["end_turn"], hold(0.2), rst=rst),
            _rrow(["main", "knight"], ["end_turn"], ["play_knight", 2, 1], hold(0.05), rst=rst),  # spent, quiet
            _rrow(["roll", "knight"], ["roll"], ["play_knight", 2, 1], hold(0.85), rst=rst),     # spent, exposed
            _rrow(["roll", "knight"], ["play_knight", 1, 2], ["play_knight", 2, 1], hold(0.7), rst=rst),  # no flip
            _rrow(["main"], M, M, plain, rst=rst)]
    g = DS.robber_gates(rows, "x", "insurance")
    assert g["gates"]["G1"]["n"] == 7 and g["gates"]["G1"]["changed"] == 5 and g["gates"]["G1"]["pass"]
    g2 = g["gates"]["G2"]
    assert g2["p_hit_median"] == 0.7 and g2["n"] == 4 and g2["consistent"] == 3 and g2["share"] == 0.75 and g2["pass"]
    assert g2["changed"] == 5 and g2["kept_exposed"] == 2 and g2["kept_exposed_share"] == 0.4   # the literal count
    assert g["gates"]["G3"]["n"] == 1 and g["gates"]["G3"]["pass"]
    assert abs(g["gates"]["G4"]["mean_c_ins"] - 0.02) < 1e-12 and g["gates"]["G4"]["pass"]
    assert abs(g["ins_offset_cal"] - 2.1) < 1e-12 and g["pass"]
    # duration: >= 2% of robber / knight decisions, readouts on the changed robber moves
    rows = [_rrow(["robber"], R, K, {"def": _mv(False, need=0.2), "cand": {"x": _mv(False, need=0.6, leader=False,
                                                                                    opp=3.0)},
                                     "holder": False, "blocked": 0.0})] + \
        [_rrow(["robber"], R, R, other) for _ in range(9)]
    g = DS.robber_gates(rows, "x", "duration")
    assert g["gates"]["G1"]["share"] == 0.1 and g["pass"]
    assert abs(g["readout"]["d_p_need_mean"] - 0.4) < 1e-12 and g["readout"]["d_leader_hit"] == -1
    with pytest.raises(SystemExit):
        DS.parse_gate_map("r1a=nope")
    assert DS.parse_gate_map("a=persistence, b=insurance") == {"a": "persistence", "b": "insurance"}


def test_robber_shadow_identity_tags_and_gate_report(tmp_path):
    """One short game with the robber classifier on: the A/A and a candidate equal to the default (robber_corr=0
    spelled out) change 0 decisions, the robber / knight rows carry the tags, every candidate search of
    robber_corr=1 reports its robber_eval counters, the game is the default bot's own, and the gate report
    reproduces the in-run gates."""
    from catanbot.selfplay import make_bot, play_game
    out = tmp_path / "rob.json"
    rc = DS.main(["--games", "1", "--seed", "7304", "--phases", "robber,knight,main", "--main-every", "6",
                  "--max-turns", "70", "--keep-rows", "--robber-gates", "same=persistence,r1a=persistence",
                  "--cand", "same:robber_corr=0", "--cand", "r1a:robber_corr=1", "--json", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["aa_ok"] is True and doc["candidates"]["same"]["changed"] == 0
    assert doc["robber_gate_map"] == {"same": "persistence", "r1a": "persistence"}
    rows = doc["rows"]
    assert rows and all("rob" in r for r in rows if set(r["cls"]) & {"robber", "knight", "main", "roll"})
    assert all("rstats" in r["cand"]["r1a"] and "rstats" not in r["cand"]["same"] for r in rows)
    for r in rows:
        if r["def"] and r["def"][0] in ("move_robber", "play_knight"):
            assert r["rob"]["def"] is not None and set(r["rob"]["def"]) >= {"kick_t1", "kick_any", "on_leader"}
        if r["rob"]["holder"]:
            assert 0.0 <= r["rob"]["p_hit"] <= 1.0
    same = doc["robber_gates"]["same"]
    assert same["gates"]["G2"]["changed"] == 0 and same["gates"]["G3"]["changed"] == 0
    rep = DS.gate_report(str(out), "r1a", str(tmp_path / "gate.json"))
    assert rep["gates"] == doc["robber_gates"]["r1a"]["gates"] and rep["pass"] == doc["robber_gates"]["r1a"]["pass"]
    assert json.loads((tmp_path / "gate.json").read_text())["fallback_trigger"] == rep["fallback_trigger"]
    bots = [make_bot(SPEC) for _ in range(4)]
    res = play_game(bots, rng=random.Random(7304), seed=7304, max_turns=70)
    g = doc["games"][0]
    assert (res.winner, list(res.vps), res.turns, res.actions) == (g["winner"], g["vps"], g["turns"], g["actions"])
    with pytest.raises(SystemExit):
        DS.gate_report(str(out), "nope")
