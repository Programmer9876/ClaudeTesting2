"""Tests for scripts/mechanics.py (per-game mechanism metrics from the action log) and its --mech wiring.

Golden values come from archived proof games (proof/T1/logs, read only); the replayed hands must equal the logged
final hands in every seat-game, which checks the whole resource replay (production with catanatron's
bank-depletion rule, setup yield, builds, trades, steals, discards, Monopoly, Year of Plenty).
"""
import glob
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
T1 = sorted(glob.glob(os.path.join(ROOT, "proof", "T1", "logs", "*.jsonl.gz")))
R1 = sorted(glob.glob(os.path.join(ROOT, "proof", "R1", "logs", "*.jsonl.gz")))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("mechanics_under_test", os.path.join(SCRIPTS, "mechanics.py"))
AB = _load("ablate_catanatron_for_mech_tests", os.path.join(SCRIPTS, "ablate_catanatron.py"))

GOLDEN = [
    {"setup_distinct": 5, "setup_pips": {"WOOD": 7, "BRICK": 4, "SHEEP": 2, "WHEAT": 3, "ORE": 4},
     "first_settle_round": 17, "first_city_round": 13, "settle_before_city": 0, "port_settled": 1, "port_round": 22,
     "port_kind": "SHEEP", "bank_4to1": 2, "bank_3to1": 0, "bank_2to1": 0, "bank_cards_given": 8, "robber_moves": 5,
     "robber_on_leader": 5, "robber_leader_pips": 79, "robber_on_us_rolls": 14, "cards_lost_block": 2,
     "stolen_by_us": 5, "stolen_from_us": 0, "discarded": 8, "monopoly_plays": 0, "monopoly_haul": 0, "yop_plays": 0,
     "dev_bought": 3, "dev_held10": 2, "knights_played": 0, "longest_road": 1, "largest_army": 0, "hand_check": True,
     "rounds": 22},
    {"setup_distinct": 4, "setup_pips": {"WOOD": 5, "BRICK": 0, "SHEEP": 2, "WHEAT": 8, "ORE": 5},
     "first_settle_round": 15, "first_city_round": 3, "settle_before_city": 0, "port_settled": 0, "port_round": None,
     "port_kind": None, "bank_4to1": 8, "bank_3to1": 0, "bank_2to1": 0, "bank_cards_given": 32, "robber_moves": 9,
     "robber_on_leader": 8, "robber_leader_pips": 70, "robber_on_us_rolls": 4, "cards_lost_block": 0,
     "stolen_by_us": 9, "stolen_from_us": 2, "discarded": 13, "monopoly_plays": 0, "monopoly_haul": 0,
     "yop_plays": 0, "dev_bought": 5, "dev_held10": 1, "knights_played": 4, "longest_road": 1, "largest_army": 1,
     "hand_check": True, "rounds": 17},
    {"setup_distinct": 4, "setup_pips": {"WOOD": 5, "BRICK": 3, "SHEEP": 0, "WHEAT": 5, "ORE": 8},
     "first_settle_round": 10, "first_city_round": 4, "settle_before_city": 0, "port_settled": 1, "port_round": 0,
     "port_kind": "3:1", "bank_4to1": 0, "bank_3to1": 7, "bank_2to1": 0, "bank_cards_given": 21, "robber_moves": 2,
     "robber_on_leader": 2, "robber_leader_pips": 22, "robber_on_us_rolls": 32, "cards_lost_block": 13,
     "stolen_by_us": 2, "stolen_from_us": 5, "discarded": 0, "monopoly_plays": 0, "monopoly_haul": 0, "yop_plays": 0,
     "dev_bought": 2, "dev_held10": 1, "knights_played": 0, "longest_road": 1, "largest_army": 0, "hand_check": True,
     "rounds": 16},
]


@pytest.mark.skipif(not T1, reason="proof archive not present")
def test_mechanics_on_canned_proof_game():
    docs = list(M.iter_proof_games(T1[:1], 3))
    for doc, gold in zip(docs, GOLDEN):
        m = M.proof_game_metrics(doc)
        assert {k: m[k] for k in gold} == gold
        # every seat's replayed hand equals the logged final hand
        for s in range(4):
            assert M.proof_game_metrics(doc, s)["hand_check"] is True


@pytest.mark.skipif(not T1, reason="proof archive not present")
def test_mechanics_base_rates_t1_sample():
    br = M.base_rates(T1, 300)
    assert br["games"] == 300 and br["hand_checks"] == [1200, 1200]
    ours, opp = br["ours"], br["opponents"]
    assert 0.70 <= br["ours"]["pooled_share_4to1"] <= 0.80          # the proof analysis: 76% (1,495 / 1,966)
    assert 0.55 <= ours["port_settled"]["mean"] <= 0.63              # 59% of games
    assert 0.66 <= opp["port_settled"]["mean"] <= 0.76               # Catanatron 71%
    assert 6.0 <= ours["robber_moves"]["mean"] <= 6.8                # 6.37 a game
    assert 6.0 <= ours["bank_trades"]["mean"] <= 7.2                 # 6.55 maritime trades a game
    # the expansion / diversity readout (coordinator's measurement on 400 T1 games)
    assert 3.7 <= ours["setup_distinct"]["mean"] <= 4.0 and 4.5 <= opp["setup_distinct"]["mean"] <= 4.8
    assert ours["setup_pips_mean"]["WHEAT"] > 5.5 and ours["setup_pips_mean"]["ORE"] > 4.5
    assert ours["first_settle_round"]["median"] > opp["first_settle_round"]["median"]
    assert ours["first_city_round"]["median"] < opp["first_city_round"]["median"]
    assert ours["settle_before_city"]["mean"] < 0.45 < opp["settle_before_city"]["mean"]


@pytest.mark.skipif(not R1, reason="proof archive not present")
def test_mechanics_reads_catanatron_321_logs():
    br = M.base_rates(R1, 40)          # 3.2.1 format: DISCARD lists, MOVE_ROBBER [coord, victim, stolen]
    assert br["hand_checks"] == [160, 160]


def test_hex_nodes_match_catanatron_topology():
    cat = pytest.importorskip("catanatron.models.map")
    all_tiles = {}
    auto = 0
    land = {}
    for coord, tt in cat.BASE_MAP_TEMPLATE.topology.items():
        nodes, edges, auto = cat.get_nodes_and_edges(all_tiles, coord, auto)
        all_tiles[coord] = cat.Water(nodes, edges)
        if tt == cat.LandTile:
            land[coord] = tuple(sorted(nodes.values()))
    assert land == M.HEX_NODES


def test_synthetic_log_resource_flow_and_counters():
    board = {"tiles": [{"c": [0, 0, 0], "t": "land", "resource": "WHEAT", "number": 6},
                       {"c": [1, -1, 0], "t": "land", "resource": "ORE", "number": 8},
                       {"c": [0, -1, 1], "t": "land", "resource": None, "number": None}],
             "ports": {"3:1": [1], "WHEAT": [3]}}
    items = [
        ["RED", "BUILD_SETTLEMENT", 1, None], ["RED", "BUILD_ROAD", [1, 2], None],
        ["BLUE", "BUILD_SETTLEMENT", 3, None], ["BLUE", "BUILD_ROAD", [3, 4], None],
        ["BLUE", "BUILD_SETTLEMENT", 5, None], ["BLUE", "BUILD_ROAD", [4, 5], None],
        ["RED", "BUILD_SETTLEMENT", 2, None], ["RED", "BUILD_ROAD", [2, 9], None],    # 2nd: yields wheat + ore
        ["RED", "ROLL", [3, 3], [3, 3]],                                               # 6: RED wheat x2, BLUE x2
        ["RED", "MARITIME_TRADE", ["WHEAT", "WHEAT", "WHEAT", None, "ORE"], None],     # 3:1 (RED has 3 wheat)
        ["RED", "END_TURN", None, None],
        ["BLUE", "ROLL", [3, 4], [3, 4]],
        ["BLUE", "MOVE_ROBBER", [[1, -1, 0], "RED"], "ORE"],
        ["BLUE", "END_TURN", None, None],
        ["RED", "ROLL", [4, 4], [4, 4]],                                               # 8 on the robber: blocked
        ["RED", "END_TURN", None, None],
    ]
    m = M.game_mechanics(items, board, "RED", colors=["RED", "BLUE"])
    assert m["bank_3to1"] == 1 and m["bank_trades"] == 1 and m["share_4to1"] == 0.0
    assert m["port_settled"] == 1 and m["port_round"] == 0 and m["port_kind"] == "3:1"
    assert m["stolen_from_us"] == 1 and m["robber_moves"] == 0
    assert m["cards_lost_block"] == 2              # settlements on nodes 1 and 2 of the 8-ore hex
    assert m["setup_distinct"] == 2 and m["setup_pips"]["WHEAT"] == 10 and m["setup_pips"]["ORE"] == 10
    b = M.game_mechanics(items, board, "BLUE", colors=["RED", "BLUE"])
    assert b["robber_moves"] == 1 and b["stolen_by_us"] == 1 and b["port_kind"] == "WHEAT"


def test_overshoot_rule_uses_paired_mechanism_se():
    def pair(c, d):
        return ({"mech": {"setup_distinct": c}}, {"mech": {"setup_distinct": d}})
    pairs = [pair(5, 4)] * 30 + [pair(4, 4)] * 10
    s = M.summarize(pairs, "setup_distinct")
    assert s["n_pairs"] == 40 and abs(s["diff"] - 0.75) < 1e-9 and s["diff_se"] > 0
    assert M.overshoot(s, "+", -0.01) is True          # moved up by > 2 se while the win rate did not rise
    assert M.overshoot(s, "+", 0.02) is False          # a win-rate gain is no overshoot
    assert M.overshoot(s, "-", -0.01) is False         # moved the other way
    noisy = M.summarize([pair(5, 4), pair(3, 4)] * 20, "setup_distinct")
    assert M.overshoot(noisy, "+", -0.01) is False
    # records without mech are excluded; medians for turn-number metrics
    mixed = [({"mech": {"first_settle_round": 12}}, {"mech": {"first_settle_round": 8}}),
             ({"mech": {"first_settle_round": 14}}, {}), ({"mech": {"first_settle_round": None}},
                                                          {"mech": {"first_settle_round": 9}})]
    s2 = M.summarize(mixed, "first_settle_round")
    assert s2["n_pairs"] == 1 and s2["cand_median"] == 12 and s2["def_median"] == 8.5


def test_inert_harness_detection_from_adapter_counters():
    SEQ = _load("seqtest_for_mech_tests", os.path.join(SCRIPTS, "seqtest.py"))
    st = SEQ.LookStats(n_target=400)
    for _ in range(400):
        st.add(0.0)
    st.known, st.identical = 400, 400
    assert not SEQ.mechanic_fired("trades", st)
    st.adapter_def = {"offers_received": 3}
    assert SEQ.mechanic_fired("trades", st)
    assert not SEQ.mechanic_fired("counting", st) and SEQ.mechanic_fired(None, st)


def test_real_game_mech_field_only_with_flag_arm_key_unchanged(tmp_path):
    pytest.importorskip("catanatron")
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONHASHSEED", None)
    outs = {}
    for tag, extra in (("plain", []), ("mech", ["--mech"])):
        out = tmp_path / f"{tag}.jsonl"
        proc = subprocess.run([sys.executable, os.path.join(SCRIPTS, "ablate_catanatron.py"), "--tunable",
                               "search.trade_proposals", "--values", "0", "--opponent", "random", "--seeds", "1",
                               "--workers", "1", "--out", str(out), "--quiet"] + extra, env=env, capture_output=True,
                              text=True, timeout=600, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr[-2000:]
        recs, _ = AB.read_jsonl(str(out))
        outs[tag] = [r for r in recs if r.get("kind") == "game"]
    assert all("mech" not in g for g in outs["plain"])
    assert all(isinstance(g.get("mech"), dict) and g["mech"].get("hand_check") is True for g in outs["mech"])
    key = lambda gs: sorted((g["arm"], g["arm_key"], g["trace"]["h"]) for g in gs)   # noqa: E731
    assert key(outs["plain"]) == key(outs["mech"])          # same arms, same games
