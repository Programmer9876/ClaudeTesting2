"""Tests of scripts/replay_export.py (docs/designs/replay_archive.md, section 9.3).

Run under both interpreters: ``/home/user/venv_cat33/bin/python -m pytest tests/test_replay_export.py`` (catanatron
3.3, logs T2 / T4 / T8 / H1 / M3) and ``python3 -m pytest tests/test_replay_export.py`` (catanatron 3.2.1, logs R1 / R2).
Version-specific tests skip on the other generation (``AD.API_33``).  Tests that need the proof's own tracker run it
in a subprocess with ``/home/user/proof_snapshot2`` first on ``sys.path`` (the repository's ``catanbot`` is already
imported in this process) and skip if the snapshot is absent.

The round-trip tests decode the compact records with a small pure-Python decoder that mirrors what the page's
``replay_core.js`` derives (spec 3.5) and compare every position with ``AD.state_summary`` of an independent replay.
"""
from __future__ import annotations

import gzip
import importlib.util
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "replay_export.py")
SNAP = "/home/user/proof_snapshot2"
PY33 = "/home/user/venv_cat33/bin/python"

_spec = importlib.util.spec_from_file_location("replay_export", SCRIPT)
RE = importlib.util.module_from_spec(_spec)
sys.modules["replay_export"] = RE
_spec.loader.exec_module(RE)

from catanbot.bench import catanatron_adapter as AD  # noqa: E402

RE.load_engine(None)          # this process: the repository's catanbot (whichever catanatron is installed)

needs33 = pytest.mark.skipif(not AD.API_33, reason="needs catanatron 3.3 (run under /home/user/venv_cat33/bin/python)")
needs32 = pytest.mark.skipif(AD.API_33, reason="needs catanatron 3.2.1 (run under python3)")
needs_snap = pytest.mark.skipif(not os.path.isdir(SNAP), reason="proof snapshot /home/user/proof_snapshot2 absent")


def game_rec(t, g):
    return RE.read_games(RE.log_files(t), g, g + 1)[0]


def make_ctx(t, rec):
    metas, by_game, _ = RE.load_results(t)
    info = RE.test_info(metas[0])
    geo = RE.global_geometry(AD.rebuild_game(rec))
    ti, ei = RE.geo_indexes(geo)
    return ({"test": t, "geo": geo, "tile_index": ti, "edge_index": ei, "info": info,
             "counted": info.get("mode") == "counted", "vps_to_win": int(metas[0].get("vps_to_win", 10))}, by_game)


# ---------------------------------------------------------------------------
# Pure-Python decoder mirroring replay_core.js (spec 3.4 / 3.5)
# ---------------------------------------------------------------------------
def decode_game(enc, geo):
    """Per-position snapshots ``k = 0..N`` derived from the compact record alone."""
    n = enc["n"]
    board = RE.parse_board_code(enc["b"])
    desert = next(i for i, (r, _) in enumerate(board["land"]) if r is None)
    st = {"hand": [[0] * 5 for _ in range(n)], "held": [[0] * 5 for _ in range(n)],
          "played": [[0] * 5 for _ in range(n)], "vp": [0] * n, "pvp": [0] * n, "lrl": [0] * n, "LR": -1, "LA": -1,
          "sett": [set() for _ in range(n)], "city": [set() for _ in range(n)], "road": [set() for _ in range(n)],
          "robber": list(geo["tiles"][desert][0]), "deck": 25, "rolls": 0}

    def snap():
        return {"hand": [list(h) for h in st["hand"]], "held": [list(h) for h in st["held"]],
                "played": [list(h) for h in st["played"]], "vp": list(st["vp"]), "pvp": list(st["pvp"]),
                "lrl": list(st["lrl"]), "LR": st["LR"], "LA": st["LA"], "sett": [sorted(s) for s in st["sett"]],
                "city": [sorted(s) for s in st["city"]], "road": [sorted(s) for s in st["road"]],
                "robber": list(st["robber"]), "deck": st["deck"], "rolls": st["rolls"],
                "bank": [19 - sum(h[r] for h in st["hand"]) for r in range(5)]}

    out = [snap()]
    for step in enc["s"]:
        op = step[0]
        v = step[1] if len(step) > 1 else None
        h = step[2] if len(step) > 2 else 0
        x = step[3] if len(step) > 3 else []
        kind, seat = RE.KINDS[op // 4], op % 4
        for i in range(0, len(h or []), 2):
            s, r = divmod(h[i], 5)
            st["hand"][s][r] += h[i + 1]
        for i in range(0, len(x), 2):
            code, val = x[i], x[i + 1]
            if code == 12:
                st["LR"] = val
            elif code == 13:
                st["LA"] = val
            else:
                f, s = divmod(code, 4)
                st[("vp", "pvp", "lrl")[f]][s] = val
        if kind == "BUILD_SETTLEMENT":
            st["sett"][seat].add(v)
        elif kind == "BUILD_CITY":
            st["sett"][seat].discard(v)
            st["city"][seat].add(v)
        elif kind == "BUILD_ROAD":
            st["road"][seat].add(v)
        elif kind == "ROLL":
            st["rolls"] += 1
        elif kind == "BUY_DEVELOPMENT_CARD":
            st["held"][seat][v] += 1
            st["deck"] -= 1
        elif kind in RE.PLAY_DEV:
            t = RE.PLAY_DEV[kind]
            st["held"][seat][t] -= 1
            st["played"][seat][t] += 1
        elif kind == "MOVE_ROBBER":
            st["robber"] = list(geo["tiles"][v // 100][0])
        out.append(snap())
    return out


def summary_as_snap(s, edge_index):
    P = s["players"]
    return {"hand": [[p["resources"][r] for r in RE.RES] for p in P],
            "held": [[p["dev_cards"][d] for d in RE.DEV] for p in P],
            "played": [[p["dev_played"].get(d, 0) for d in RE.DEV] for p in P],
            "vp": [p["vp"] for p in P], "pvp": [p["public_vp"] for p in P], "lrl": [p["longest_road_length"] for p in P],
            "LR": next((p["seat"] for p in P if p["has_longest_road"]), -1),
            "LA": next((p["seat"] for p in P if p["has_largest_army"]), -1),
            "sett": [sorted(p["settlements"]) for p in P], "city": [sorted(p["cities"]) for p in P],
            "road": [sorted(edge_index[(e[0], e[1])] for e in p["roads"]) for p in P],
            "robber": list(s["robber"]), "deck": s["dev_deck_left"], "bank": [s["bank"][r] for r in RE.RES]}


def check_roundtrip(t, g):
    rec = game_rec(t, g)
    ctx, by_game = make_ctx(t, rec)
    res = RE.encode_game(rec, ctx, by_game.get(g), want_truth=True)
    enc = res["record"]
    json.loads(json.dumps(enc))                      # JSON-ready
    dec = decode_game(enc, ctx["geo"])
    game = AD.rebuild_game(rec)
    rolls = 0
    for k in range(len(rec["actions"]) + 1):
        if k:
            AD.replay_log_action(game, rec["actions"][k - 1], check=True)
            rolls += rec["actions"][k - 1][1] == "ROLL"
        want = summary_as_snap(AD.state_summary(game.state), ctx["edge_index"])
        got = {key: dec[k][key] for key in want}
        assert got == want, f"{t}-{g} position {k}: " + ", ".join(
            f"{key}: {got[key]} != {want[key]}" for key in want if got[key] != want[key])
        assert dec[k]["rolls"] == rolls
        tk = res["truth"]["pos"][k]                   # the worker's truth file has the same position
        assert tk["P"][0][2] == want["hand"][0] and tk["bank"] == want["bank"] and tk["turn"] == game.state.num_turns
    return rec, res, dec


# ---------------------------------------------------------------------------
# Paths and dispatch
# ---------------------------------------------------------------------------
def test_shard_path():
    assert RE.shard_path("T7", 537) == "shards/T7_05.json.gz"
    assert RE.shard_path("T8", 0) == "shards/T8_00.json.gz"
    assert RE.shard_path("T1", 999) == "shards/T1_09.json.gz"
    assert RE.shard_path("H1", 399) == "shards/H1_03.json.gz"


def test_interpreter_for():
    assert RE.interpreter_for(game_rec("R1", 0)["catanatron"], "PY33", "PY32") == "PY32"
    assert RE.interpreter_for(game_rec("T8", 0)["catanatron"], "PY33", "PY32") == "PY33"
    with pytest.raises(ValueError, match="no interpreter for logs of catanatron '4.0.0'"):
        RE.interpreter_for("4.0.0")


def test_code_root_for():
    for t in ("T7", "T8", "T9", "T10", "T11"):
        assert RE.code_root_for(t, "SNAP", "HEAD") == "SNAP"
    for t in ("T1", "T2", "T6", "R1", "R2", "H1", "H2", "M1", "M3"):
        assert RE.code_root_for(t, "SNAP", "HEAD") == "HEAD"
    with pytest.raises(ValueError):
        RE.code_root_for("T1", "SNAP", None)


def test_log_files_tile_games():
    for t, n in (("T8", 400), ("T7", 1000), ("R1", 1000), ("H1", 400), ("M3", 400)):
        files = RE.log_files(t)
        assert files[0]["from"] == 0 and files[-1]["to"] == n
        assert all(a["to"] == b["from"] for a, b in zip(files, files[1:]))
    recs = RE.read_games(RE.log_files("T8"), 38, 42)          # crosses a file boundary
    assert [r["game"] for r in recs] == [38, 39, 40, 41]


def test_board_code_roundtrip():
    for t, g in (("T8", 5), ("R1", 3), ("H1", 0)):
        rec = game_rec(t, g)
        code = RE.board_code(rec)
        assert len(code) == 48 and code[38] == "|"
        back = RE.parse_board_code(code)
        land = [x for x in rec["board"]["tiles"] if x["t"] == "land"]
        ports = [x for x in rec["board"]["tiles"] if x["t"] == "port"]
        assert back["land"] == [(x["resource"], x["number"]) for x in land]
        assert back["ports"] == [x["resource"] for x in ports]


def test_lineup_and_deal_key():
    a, b = game_rec("T1", 5), game_rec("T2", 5)       # T1-T6 share deals for games 0-399
    assert a["seed"] == b["seed"] and RE.deal_key(a) == RE.deal_key(b)
    assert RE.deal_key(game_rec("T7", 0)) == RE.deal_key(game_rec("T8", 0))
    assert RE.deal_key(game_rec("T8", 0)) != RE.deal_key(game_rec("T8", 1))
    assert RE.lineup_of(game_rec("T8", 1)) == "acaa"
    assert RE.lineup_of(game_rec("M3", 0)).count("c") == 2
    assert RE.lineup_of(game_rec("R1", 0)) == "cfff"


def test_truth_targets_deterministic():
    plan = RE.plan_test("T8")
    chosen, games = RE.truth_targets(plan, 4)
    rnd = random.Random("T8-20260926").sample(range(400), 4)
    assert all("random" in chosen[g] for g in rnd)
    assert "heaviest hypotheses" in chosen[375]
    reasons = {w for ws in chosen.values() for w in ws}
    assert {"steal from discarder", "monopoly with hidden discards", "our bot thief and victim"} <= reasons
    assert RE.truth_targets(plan, 4)[1] == games


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_step_tuple():
    assert RE.step_tuple(16, None, [], []) == [16]
    assert RE.step_tuple(12, 63, [], []) == [12, 63]
    assert RE.step_tuple(12, 63, [0, 1], []) == [12, 63, [0, 1]]
    assert RE.step_tuple(36, None, [], [13, 0]) == [36, None, 0, [13, 0]]
    assert RE.step_tuple(0, 21, [], [0, 1, 4, 1]) == [0, 21, 0, [0, 1, 4, 1]]


def test_encode_value_table():
    seat_of = {"RED": 0, "BLUE": 1, "ORANGE": 2, "WHITE": 3}
    ti = {(0, 0, 0): 0, (1, -1, 0): 1, (2, -2, 0): 7}
    ei = {(16, 21): 40, (8, 9): 12}
    ev = lambda k, v, r=None: RE.encode_value(k, v, r, seat_of, ti, ei)  # noqa: E731
    assert ev("BUILD_SETTLEMENT", 21) == 21
    assert ev("BUILD_CITY", 3) == 3
    assert ev("BUILD_ROAD", [21, 16]) == 40
    assert ev("ROLL", [6, 3], [6, 3]) == 63
    assert ev("END_TURN", None) is None
    assert ev("BUY_DEVELOPMENT_CARD", "MONOPOLY", "MONOPOLY") == 4
    assert ev("BUY_DEVELOPMENT_CARD", "KNIGHT") == 0                              # 3.2.1: the value
    assert ev("MOVE_ROBBER", [[1, -1, 0], "BLUE"], "ORE") == 1 * 100 + 2 * 10 + 5   # 3.3: stolen = result
    assert ev("MOVE_ROBBER", [[2, -2, 0], "RED", "SHEEP"], "SHEEP") == 700 + 10 + 3  # 3.2.1: value[2]
    assert ev("MOVE_ROBBER", [[2, -2, 0], "RED", "SHEEP"]) == 713
    assert ev("MOVE_ROBBER", [[0, 0, 0], None]) == 0                              # no victim
    assert ev("MOVE_ROBBER", [[1, -1, 0], "WHITE"]) == 140                        # victim without cards
    assert ev("MARITIME_TRADE", ["ORE", "ORE", "ORE", "ORE", "BRICK"]) == 4 * 100 + 40 + 1
    assert ev("MARITIME_TRADE", ["WHEAT", "WHEAT", "WHEAT", None, "WOOD"]) == 300 + 30 + 0
    assert ev("MARITIME_TRADE", ["SHEEP", "SHEEP", None, None, "ORE"]) == 200 + 20 + 4
    assert ev("DISCARD_RESOURCE", "WHEAT", "WHEAT") == 3
    assert ev("DISCARD", ["SHEEP", "SHEEP", "SHEEP", "WOOD", "ORE", "ORE", "BRICK", "WHEAT", "WOOD"]) == [2, 1, 3, 1, 2]
    assert ev("PLAY_KNIGHT_CARD", None) is None
    assert ev("PLAY_MONOPOLY", "ORE") == 4
    assert ev("PLAY_YEAR_OF_PLENTY", ["BRICK", "SHEEP"]) == [1, 2]
    assert ev("PLAY_YEAR_OF_PLENTY", ["WOOD"]) == [0]
    assert ev("PLAY_YEAR_OF_PLENTY", "WOOD") == [0]
    assert ev("PLAY_ROAD_BUILDING", None) is None
    with pytest.raises(RE.ExportError):
        ev("OFFER_TRADE", None)


def test_largest_remainder():
    assert RE.largest_remainder([1 / 3, 1 / 3, 1 / 3]) == [334, 333, 333]
    assert RE.largest_remainder([1.0]) == [1000]
    assert sum(RE.largest_remainder([0.62, 0.33, 0.05, 1e-9])) == 1000
    assert RE.largest_remainder([0.8104, 0.1896]) == [810, 190]


def test_min_chance_keeps_relative_precision():
    """bv.sum.minP of T11-186 is 2.47e-9 and of T7-278 6.05e-9: rounding to 8 decimals gave 0 and 1e-8 (found by the
    full-bundle validator run); significant digits keep them."""
    assert RE._sig(2.471724145016134e-9) == 2.47172e-9
    assert RE._sig(6.053408747539136e-9) == 6.05341e-9
    assert RE._sig(0.771234567) == 0.771235 and RE._sig(None) is None
    assert RE._r4(RE._sig(2.471724145016134e-9)) == 2.472e-9


def test_geometry_global():
    """Games of both generations, 2- and 4-player, give one geometry; it matches game_viewer.geometry."""
    code = ("import sys, json; sys.path.insert(0, %r); import replay_export as RE; RE.load_engine(%r); "
            "t, g = sys.argv[1], int(sys.argv[2]); rec = RE.read_games(RE.log_files(t), g, g + 1)[0]; "
            "print(json.dumps(RE.global_geometry(RE._ENV.AD.rebuild_game(rec))))") % (os.path.dirname(SCRIPT), REPO)
    geos = []
    for py, t, g in ((PY33, "T2", 0), (PY33, "H1", 7), ("python3", "R1", 0), ("python3", "R2", 3)):
        if shutil.which(py) is None and not os.path.exists(py):
            pytest.skip(f"{py} not available")
        r = subprocess.run([py, "-c", code, t, str(g)], capture_output=True, text=True, cwd="/")
        assert r.returncode == 0, r.stderr[-800:]
        geos.append(json.loads(r.stdout))
    assert all(x == geos[0] for x in geos)
    geo = geos[0]
    assert (len(geo["nodes"]), len(geo["edges"]), len(geo["tiles"]), len(geo["ports"])) == (96, 132, 19, 9)
    assert all(a < b for a, b in geo["edges"]) and geo["edges"] == sorted(geo["edges"])
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import game_viewer as GV
    t = "T2" if AD.API_33 else "R1"
    nodes, tiles, ports = GV.geometry(AD.rebuild_game(game_rec(t, 0)))
    assert [nodes[i] for i in range(len(nodes))] == geo["nodes"]
    assert [[x["c"], x["x"], x["y"]] for x in tiles] == geo["tiles"]
    assert [[p["x"], p["y"], p["nodes"]] for p in ports] == geo["ports"]


# ---------------------------------------------------------------------------
# Round trip: compact record -> every position == state_summary
# ---------------------------------------------------------------------------
@needs33
@pytest.mark.parametrize("t,g", [("T2", 0), ("T2", 11), ("T4", 3), ("H1", 7), ("M3", 5), ("T8", 0)])
def test_roundtrip_every_step_33(t, g):
    check_roundtrip(t, g)


@needs32
@pytest.mark.parametrize("t,g", [("R1", 0), ("R2", 3)])
def test_roundtrip_every_step_321(t, g):
    check_roundtrip(t, g)


@needs32
def test_roundtrip_big_discard_321():
    """A 3.2.1 DISCARD of >= 9 cards (one step with counts)."""
    plan = RE.plan_test("R1")
    chosen, _ = RE.truth_targets(plan, 0)
    g = next(g for g, why in chosen.items() if "big discard" in why)
    rec, res, _ = check_roundtrip("R1", g)
    big = [s for s in res["record"]["s"] if s[0] // 4 == RE.KIND_CODE["DISCARD"] and sum(s[1]) >= 9]
    assert big


def _index_fields_check(t, g):
    rec, res, dec = check_roundtrip(t, g)
    row = dict(zip(RE.INDEX_COLS, res["row"]))
    fin = rec["final"]
    ours = rec["our_seats"]
    fp = fin["state"]["players"]
    assert (row["t"], row["g"], row["seed"]) == (t, g, rec["seed"])
    assert row["lineup"] == RE.lineup_of(rec) and [i for i, c in enumerate(row["lineup"]) if c == "c"] == ours
    assert row["win"] == fin["winner_seat"] and row["vps"] == fin["vps"]
    assert row["pvps"] == [p["public_vp"] for p in fp]
    assert row["turns"] == fin["turns"] and row["acts"] == len(rec["actions"])
    assert row["rolls"] == sum(1 for a in rec["actions"] if a[1] == "ROLL")
    w = row["win"]
    assert row["hvp"] == int(fp[w]["public_vp"] < 10) and row["hv"] == fp[w]["vp"] - fp[w]["public_vp"]
    assert row["dev"] == sum(1 for a in rec["actions"] if a[1] == "BUY_DEVELOPMENT_CARD"
                             and rec["colors"].index(a[0]) in ours)
    assert row["lr"] == next((p["seat"] for p in fp if p["has_longest_road"]), -1)
    assert row["la"] == next((p["seat"] for p in fp if p["has_largest_army"]), -1)
    last_end = max(i for i, a in enumerate(rec["actions"]) if a[1] == "END_TURN")
    opp = [i for i in range(rec["colors"].__len__()) if i not in ours]
    dv = [max(d["vp"][j] for j in opp) - max(d["vp"][j] for j in ours) for d in dec[:last_end + 2]]
    dp = [max(d["pvp"][j] for j in opp) - max(d["pvp"][j] for j in ours) for d in dec[:last_end + 2]]
    assert row["def"] == max(0, max(dv)) and row["defp"] == max(0, max(dp))
    return row


@needs33
def test_index_fields_33():
    for t, g in (("T2", 4), ("M3", 1), ("H1", 2)):
        row = _index_fields_check(t, g)
        assert row["bk"] is None


@needs32
def test_index_fields_321():
    for t, g in (("R1", 7), ("R2", 1)):
        assert _index_fields_check(t, g)["bk"] is None


def test_worker_rejects_wrong_engine():
    t, g = ("R1", 0) if AD.API_33 else ("T8", 0)
    rec = game_rec(t, g)
    ctx = {"test": t, "geo": None, "tile_index": {}, "edge_index": {}, "info": {"mode": "full"}, "counted": False,
           "vps_to_win": 10}
    with pytest.raises(ValueError, match="interpreter"):
        RE.encode_game(rec, ctx)


# ---------------------------------------------------------------------------
# Belief mathematics (pure)
# ---------------------------------------------------------------------------
def test_p_true_hand_pending():
    """A pending hidden discard of 2: the hypergeometric marginal equals a brute-force enumeration over every
    ordered pair of discarded cards."""
    hyps = {((2, 1, 0, 1, 0), (0, 0, 0, 0, 0)): 0.5, ((1, 1, 1, 1, 0), (0, 0, 0, 0, 0)): 0.3,
            ((0, 0, 2, 0, 2), (0, 0, 0, 0, 0)): 0.2}
    got = RE.hand_marginal(hyps, 0, 2)
    brute = {}
    for joint, w in hyps.items():
        cards = [r for r in range(5) for _ in range(joint[0][r])]
        pairs = list(itertools.permutations(range(len(cards)), 2))
        for a, b in pairs:
            left = list(joint[0])
            left[cards[a]] -= 1
            left[cards[b]] -= 1
            brute[tuple(left)] = brute.get(tuple(left), 0.0) + w / len(pairs)
    assert set(got) == set(brute)
    assert all(abs(got[h] - brute[h]) < 1e-12 for h in got)
    assert abs(sum(got.values()) - 1) < 1e-12
    assert RE.p_true_hand(got, [1, 0, 0, 1, 0]) == pytest.approx(brute[(1, 0, 0, 1, 0)])
    assert RE.p_true_hand(got, [5, 0, 0, 0, 0]) == 0.0
    assert RE.hand_marginal(hyps, 0, 0) == {(2, 1, 0, 1, 0): 0.5, (1, 1, 1, 1, 0): 0.3, (0, 0, 2, 0, 2): 0.2}


def test_dev_vp_law_pure():
    # nobody near 10 VP: each holder's count is hypergeometric
    law = RE.dev_vp_law([10, 4, 2, 2, 2], [1, 2], [2, 3], [4, 5])
    T, V = 20, 4
    for j, c in ((1, 2), (2, 3)):
        want = [math.comb(V, v) * math.comb(T - V, c - v) / math.comb(T, c) for v in range(c + 1)]
        assert law[j] == pytest.approx(want, abs=1e-12)
    # a holder at 9 public VP with one card: conditioned on not having won, and the fallback also swaps it away
    law = RE.dev_vp_law([2, 1, 1, 0, 2], [0, 1, 2], [0, 1, 0], [7, 9, 7])
    assert law[1] == pytest.approx([1.0, 0.0], abs=1e-12) and law[0] == [1.0]
    # every deal wins and no non-VP card is left to swap: the fallback keeps the VP card
    law = RE.dev_vp_law([0, 1, 0, 0, 0], [1], [1], [9])
    assert law[1] == pytest.approx([0.0, 1.0])
    for lw in RE.dev_vp_law([5, 3, 1, 2, 1], [0, 2, 3], [3, 2, 1], [8, 9, 6]).values():
        assert abs(sum(lw) - 1) < 1e-12


def test_audit_rules_pure():
    """The rule table on hand-made calls: allowed and forbidden inputs for the same entry."""
    colors = ["RED", "BLUE", "ORANGE"]
    items = [["BLUE", "MOVE_ROBBER", [[0, 0, 0], "ORANGE"], "ORE"]]    # a steal between two opponents of RED
    hands = [[[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2]],
             [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]]
    banks = [[18, 19, 19, 19, 17], [18, 19, 19, 19, 17]]
    fields = [([18, 19, 19, 19, 17], [1, 1, 1])]
    ok = [(0, "observe_steal", (2, 1, None), {}, 0), (0, "_update", (), {}, 1),
          (0, "observe_hand", (0, [1, 0, 0, 0, 0]), {}, 0), (0, "observe_hand_size", (1, 1), {}, 0),
          (0, "observe_bank", ([18, 19, 19, 19, 17],), {}, 0)]
    assert RE.audit_violations(ok, [], fields, items, hands, banks, colors, 0) == []
    for bad in [(0, "observe_steal", (2, 1, 4), {}, 0),           # third-party stolen card revealed
                (0, "_revealed_result", (2,), {}, 0),               # chance result read
                (0, "_update", (), {}, 0),                          # a write outside an audited call
                (0, "_reset", (), {}, 1), (0, "observe_gain", (1, 4), {}, 0),
                (0, "observe_hand", (1, [0, 0, 0, 0, 1]), {}, 0),   # an opponent's hand
                (0, "observe_delta", (1, [0, 0, 0, 0, 1]), {}, 0)]:  # a delta on a robber move
        assert len(RE.audit_violations([bad], [], fields, items, hands, banks, colors, 0)) == 1, bad
    assert RE.audit_violations([], [(0, "x")], fields, items, hands, banks, colors, 0)
    assert RE.audit_violations([], [], [([0] * 5, [1, 1, 1])], items, hands, banks, colors, 0)


# ---------------------------------------------------------------------------
# "What our bot knew": the proof's tracker in a subprocess (catanatron 3.3 + snapshot)
# ---------------------------------------------------------------------------
def _hidden_steal_game():
    """First T8 game with a steal between two opponents of our bot, and its action indexes."""
    for rec in RE.iter_all_games(RE.log_files("T8")):
        me = rec["our_seats"][0]
        seat = {c: i for i, c in enumerate(rec["colors"])}
        idx = [i for i, a in enumerate(rec["actions"]) if a[1] == "MOVE_ROBBER" and a[2][1] is not None
               and a[3] is not None and me not in (seat[a[0]], seat[a[2][1]])]
        if idx:
            return int(rec["game"]), idx
    raise AssertionError("no hidden steal in T8")


@pytest.fixture(scope="module")
def t8_worker(tmp_path_factory):
    if not (AD.API_33 and os.path.isdir(SNAP)):
        pytest.skip("needs catanatron 3.3 and the proof snapshot")
    g_steal, _ = _hidden_steal_game()
    games = sorted({0, 1, g_steal})
    d = tmp_path_factory.mktemp("t8w")
    out, work = str(d / "out"), str(d / "work")
    r = subprocess.run([sys.executable, SCRIPT, "--worker", "T8", "--shard", "0", "--code-root", SNAP, "--out", out,
                        "--work", work, "--games", ",".join(map(str, games)), "--truth", ",".join(map(str, games))],
                       capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    shard = json.loads(gzip.decompress(open(os.path.join(out, "shards", "T8_00.json.gz"), "rb").read()))
    checks = json.load(open(os.path.join(work, "parts", "T8_00.checks.json")))
    truth = json.loads(gzip.decompress(open(os.path.join(work, "truth", "T8_00.steps.json.gz"), "rb").read()))
    rows = json.load(open(os.path.join(work, "parts", "T8_00.rows.json")))["rows"]
    snap = json.load(open(os.path.join(work, "digests", "T8_00.snap.json")))
    return {"shard": shard, "checks": checks, "truth": truth, "rows": rows, "games": games, "g_steal": g_steal,
            "snap": snap, "work": work}


def decode_bv(bv, N, opps):
    """Per-position bot view from the sparse ``bv`` rows (what ReplayCore.botView derives)."""
    cur = {j: {"exact": True, "own": [0] * 5, "lp": 0, "lt": 0, "lc": 0, "u": 0, "E": [0] * 5} for j in opps}
    nh, pool, law = 1, [14, 5, 2, 2, 2], {j: [1000] for j in opps}
    qi = nhi = dpi = dvi = 0
    out = [None]
    q, nhs, dp, dv = bv["q"], bv["nh"], bv["dp"], bv["dv"]
    for k in range(1, N + 1):
        while qi < len(q) and q[qi][0] == k:
            r = q[qi]
            if r[2] == 1:
                cur[r[1]] = {"exact": True, "own": r[3:8], "lp": 0, "lt": 0, "lc": 0, "u": 0, "E": [100 * x for x in r[3:8]]}
            else:
                cur[r[1]] = {"exact": False, "own": None, "lp": r[3], "lt": r[4], "lc": r[5], "u": r[6], "E": r[7:12]}
            qi += 1
        while nhi < len(nhs) and nhs[nhi] == k:
            nh = nhs[nhi + 1]
            nhi += 2
        while dpi < len(dp) and dp[dpi][0] == k:
            pool = dp[dpi][1:]
            dpi += 1
        while dvi < len(dv) and dv[dvi][0] == k:
            law[dv[dvi][1]] = dv[dvi][2:]
            dvi += 1
        out.append({"opp": {j: dict(cur[j]) for j in opps}, "nh": nh, "pool": list(pool),
                    "law": {j: list(law[j]) for j in opps}})
    assert qi == len(q) and nhi == len(nhs) and dpi == len(dp) and dvi == len(dv)
    return out


@needs33
@needs_snap
def test_worker_imports_snapshot(t8_worker):
    code = t8_worker["checks"]["code"]
    assert code["catanbot"].startswith(SNAP + "/")
    for rel, prefix in RE.SNAPSHOT_SHA_PREFIX.items():
        assert code["sha256"][rel].startswith(prefix)
    for c in t8_worker["checks"]["games"]:
        assert c["tracker"]["class"] == "catanbot.bench.public_info.PublicInfoTracker"


@needs33
@needs_snap
def test_bot_view_settings_from_metadata(t8_worker):
    """The tracker is built with the run's own settings: results metadata == record == what the worker used."""
    metas, _, _ = RE.load_results("T8")
    assert all(m["info"] == RE.COUNTED_INFO for m in metas)
    assert t8_worker["checks"]["info"] == metas[0]["info"]
    for c in t8_worker["checks"]["games"]:
        rec = game_rec("T8", int(c["id"].split("-")[1]))
        me = rec["our_seats"][0]
        assert rec["players"][me]["info"] == metas[0]["info"]
        assert c["tracker"] == {"class": "catanbot.bench.public_info.PublicInfoTracker", "me": me,
                                "discards_public": False, "reveal_hidden": False, "vps_to_win": 10,
                                "max_hypotheses": 4096}


@needs33
@needs_snap
def test_bot_view_audit(t8_worker):
    """Per game: statistics equal the live bot's, 0 audit violations, the reveal_hidden twin equals the real
    hands at every position, the audited belief equals the plain one."""
    _, by_game, _ = RE.load_results("T8")
    for c in t8_worker["checks"]["games"]:
        g = int(c["id"].split("-")[1])
        assert c["stats"]["equal"] and c["stats"]["ours"] == c["stats"]["logged"]
        logged = by_game[g]["stats"]
        assert c["stats"]["ours"]["hidden_steals"] == logged["hidden_steals"]
        assert c["audit"][0] > 100 and c["audit"][1] == 0
        assert c["twin"] and c["same"]
    for rec in t8_worker["shard"]["games"]:
        assert rec["bv"]["chk"] == {"code": "9599eed", "stats": 1, "audit": rec["bv"]["chk"]["audit"], "twin": 1,
                                    "same": 1, "head": None}


@needs33
@needs_snap
def test_bot_view_encoding_matches_truth_file(t8_worker):
    """The sparse rows decode to the worker's per-position belief; p > 0 always; exact => own == real hand; the
    summary is recomputable from the rows."""
    truth = t8_worker["truth"]
    for rec in t8_worker["shard"]["games"]:
        tr = truth[rec["id"]]
        N = len(rec["s"])
        me = rec["o"][0]
        opps = [j for j in range(rec["n"]) if j != me]
        dec = decode_bv(rec["bv"], N, opps)
        assert len(tr["pos"]) == N + 1 and len(tr["bv"]) == N + 1
        ps, cals, pu = [], [], []
        exact_all = 0
        for k in range(1, N + 1):
            bk, pk, dk = tr["bv"][k], tr["pos"][k], dec[k]
            assert dk["nh"] == bk["nh"] and dk["pool"] == bk["pool"]
            every = True
            for j in opps:
                b, d = bk["opp"][str(j)], dk["opp"][j]
                real = pk["P"][j][2]
                assert b["p"] > 0
                assert d["exact"] == b["exact"] and d["u"] == b["u"]
                assert abs(sum(b["E"]) - sum(real)) < 1e-6
                if b["exact"]:
                    assert d["own"] == b["own"] == real
                    p = 1.0
                else:
                    assert all(abs(d["E"][r] / 100 - b["E"][r]) <= 0.005 + 1e-9 for r in range(5))
                    p = 10 ** (-d["lp"] / 1000)
                    assert abs(p / b["p"] - 1) < 0.003
                    assert abs(10 ** (-d["lt"] / 1000) / b["top"] - 1) < 0.003
                    assert abs(10 ** (-d["lc"] / 1000) / b["calp"] - 1) < 0.003
                    assert d["lt"] <= d["lp"]
                    pu.append(p)
                every = every and b["exact"]
                ps.append(p)
                cals.append(b["calp"])
                assert sum(dk["law"][j]) == 1000
                assert all(abs(dk["law"][j][v] / 1000 - b["vpLaw"][v]) <= 0.001 for v in range(len(b["vpLaw"])))
            exact_all += every
        s = rec["bv"]["sum"]
        assert abs(s["meanP"] - sum(ps) / len(ps)) < 1e-3 and abs(s["calP"] - sum(cals) / len(cals)) < 1e-6
        assert abs(s["allExact"] - exact_all / N) < 1e-6 and s["nu"] == len(pu)
        if pu:
            assert abs(s["meanPu"] - sum(pu) / len(pu)) < 1e-3
        row = next(r for r in t8_worker["rows"] if r[1] == rec["g"])
        assert row[18][:6] == [RE._r4(s[k]) for k in ("allExact", "meanP", "minP", "calP", "meanPu", "calPu")]


@needs33
@needs_snap
def test_belief_indexing_hidden_steal(t8_worker):
    """Position k is the belief after entries 0..k-1: a hidden steal at action index i shows at position i + 1
    (not at i, not later) - in the victim's real hand size, belief size and uncertainty."""
    g = t8_worker["g_steal"]
    rec = game_rec("T8", g)
    tr = t8_worker["truth"][f"T8-{g}"]
    seat = {c: i for i, c in enumerate(rec["colors"])}
    me = rec["our_seats"][0]
    tested = 0
    for i, a in enumerate(rec["actions"]):
        if not (a[1] == "MOVE_ROBBER" and a[2][1] is not None and a[3] is not None):
            continue
        v = seat[a[2][1]]
        if me in (seat[a[0]], v):
            continue
        before = tr["pos"][i]["P"][v][2]
        if sum(1 for x in before if x) < 2:
            continue
        size = sum(before)
        assert sum(tr["pos"][i + 1]["P"][v][2]) == size - 1               # the real hand drops at i + 1
        assert abs(sum(tr["bv"][i]["opp"][str(v)]["E"]) - size) < 1e-6     # not yet at i
        assert abs(sum(tr["bv"][i + 1]["opp"][str(v)]["E"]) - (size - 1)) < 1e-6   # already at i + 1
        if tr["bv"][i]["opp"][str(v)]["exact"]:
            assert tr["bv"][i]["opp"][str(v)]["own"] == before                 # still the old hand at i
            assert not tr["bv"][i + 1]["opp"][str(v)]["exact"]                 # uncertain from i + 1 on
            tested += 1
        if tested >= 1:
            break
    assert tested == 1


def _run_snapshot(script: str, tmp_path) -> dict:
    path = tmp_path / "snap_check.py"
    path.write_text("import sys, json\nsys.path.insert(0, %r)\nimport replay_export as RE\n"
                    "RE.load_engine(%r, expect_snapshot=True)\nAD, PIT = RE._ENV.AD, RE._ENV.PIT\n"
                    % (os.path.dirname(SCRIPT), SNAP) + script)
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@needs33
@needs_snap
def test_audit_catches_leak(tmp_path):
    """The audit has teeth: a tracker that reads third-party stolen cards, or restarts its counter, is caught;
    the unmodified tracker is clean."""
    g, _ = _hidden_steal_game()
    res = _run_snapshot(f"""
rec = RE.read_games(RE.log_files("T8"), {g}, {g + 1})[0]
Aud = RE.make_audited_tracker(PIT)

class PeekSteals(Aud):
    def _observe_entry(self, entry, pre, post):
        old = self.reveal_hidden
        self.reveal_hidden = AD.encode_log_entry(entry)[1] == "MOVE_ROBBER"
        try:
            super()._observe_entry(entry, pre, post)
        finally:
            self.reveal_hidden = old

class Resets(Aud):
    def _observe_entry(self, entry, pre, post):
        super()._observe_entry(entry, pre, post)
        if self._aud_i == 60:
            self.counter._reset("test")

out = {{}}
for name, cls in (("clean", None), ("peek", PeekSteals), ("reset", Resets)):
    calls, viol = RE.audit_game(rec, RE.COUNTED_INFO, 10, cls)
    out[name] = [calls, len(viol), viol[:3]]
print(json.dumps(out))
""", tmp_path)
    assert res["clean"][1] == 0 and res["clean"][0] > 100
    assert res["peek"][1] >= 1 and any("observe_steal" in v or "_revealed_result" in v for v in res["peek"][2])
    assert res["reset"][1] >= 1 and any("_reset" in v for v in res["reset"][2])


@needs33
@needs_snap
def test_dev_vp_law(tmp_path):
    """dev_vp_law equals the frequencies of 20,000 calls of the snapshot's own _deal_devs (within 0.01) at 3
    positions of T7-111, one with an opponent at 9 public VP holding a development card."""
    res = _run_snapshot("""
import random
from catanbot import board as B
rec = RE.read_games(RE.log_files("T7"), 111, 112)[0]
me = rec["our_seats"][0]
color = AD.Color(rec["colors"][me])
n = len(rec["colors"])
final = AD.rebuild_game(rec)
for it in rec["actions"]:
    AD.replay_log_action(final, it, check=True)

def walk(visit):
    game = AD.rebuild_game(rec)
    tr = PIT(color, discards_public=False, vps_to_win=10)
    gen = tr.replay_log(final.state)
    for k, it in enumerate(rec["actions"], 1):
        AD.replay_log_action(game, it, check=True)
        next(gen)
        visit(k, game, tr)

cand = []
def pick(k, game, tr):
    H = tr.unknown_dev_holders()
    c = [tr.dev_count[j] for j in H]
    pv = [r["pvp"] for r in RE.seat_rows(game.state, n)]
    if sum(c) and tr.dev_pool()[1]:
        cand.append((k, sum(c), max(pv[j] for j, x in zip(H, c) if x)))
walk(pick)
first = cand[0][0]
nine = next(k for k, _, hi in cand if hi >= 9)
rest = [k for k, _, _ in cand if k not in (first, nine)]
want = {first, nine, rest[len(rest) // 2]}
out = []
def check(k, game, tr):
    if k not in want:
        return
    H = list(tr.unknown_dev_holders())
    counts = [tr.dev_count[j] for j in H]
    pv = [r["pvp"] for r in RE.seat_rows(game.state, n)]
    law = RE.dev_vp_law(tr.dev_pool(), H, counts, [pv[j] for j in H], 10)
    s = tr.public_view(AD.to_catanbot_state(game, color)).copy()
    assert [s.public_vp(j) for j in H] == [pv[j] for j in H]
    rng = random.Random(20260926)
    freq = {j: [0] * (c + 1) for j, c in zip(H, counts)}
    N = 20000
    for _ in range(N):
        tr._deal_devs(s, rng)
        for j in H:
            p = s.players[j]
            freq[j][p.dev_cards[B.DEV_VP] + p.dev_cards_new[B.DEV_VP]] += 1
    diff = max(abs(freq[j][v] / N - law[j][v]) for j in H for v in range(len(law[j])))
    out.append({"k": k, "pool": tr.dev_pool(), "counts": counts, "pvps": [pv[j] for j in H], "diff": diff,
                "law": {str(j): law[j] for j in H}, "freq": {str(j): freq[j] for j in H}})
walk(check)
print(json.dumps({"positions": out, "nine": nine}))
""", tmp_path)
    assert len(res["positions"]) == 3
    for p in res["positions"]:
        assert p["diff"] < 0.01, p
    nine = next(p for p in res["positions"] if p["k"] == res["nine"])
    assert max(nine["pvps"][i] for i, c in enumerate(nine["counts"]) if c) >= 9


@needs33
@needs_snap
def test_head_digest_job(t8_worker, tmp_path):
    """The head-digest job (repository tracker) produces one digest per position, equal to the snapshot's."""
    games = t8_worker["games"]
    r = subprocess.run([sys.executable, SCRIPT, "--digest", "T8", "--shard", "0", "--code-root", REPO,
                        "--work", str(tmp_path), "--games", ",".join(map(str, games))],
                       capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stderr[-2000:]
    head = json.load(open(tmp_path / "digests" / "T8_00.head.json"))
    snap = t8_worker["snap"]
    assert sorted(head) == sorted(snap) == sorted(f"T8-{g}" for g in games)
    for gid in snap:
        assert len(snap[gid]) == len(game_rec("T8", int(gid.split("-")[1]))["actions"])
        assert head[gid] == snap[gid]


def test_node_validator(tmp_path):
    js = RE.VALIDATOR
    if shutil.which("node") is None or not os.path.exists(js):
        pytest.skip("node or scripts/check_replay_bundle.mjs not available")
    t = "T2" if AD.API_33 else "R1"
    r = subprocess.run([sys.executable, SCRIPT, "--worker", t, "--shard", "0", "--code-root", REPO, "--out",
                        str(tmp_path / "out"), "--work", str(tmp_path / "work"), "--games", "0,1"],
                       capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run(["node", js, "--bundle", str(tmp_path / "out"), "--bundle-only", "--work", str(tmp_path / "work"),
                        "--repo", RE.REPO], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
