"""Tests for scripts/bench_catanatron.py on catanatron 3.2.1 (wheel) and 3.3 (checkout)."""
import importlib.util
import os

import pytest

pytest.importorskip("catanatron")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bench_catanatron", os.path.join(ROOT, "scripts", "bench_catanatron.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

from catanbot.bench import catanatron_adapter as AD  # noqa: E402

STRONG = {
    "value": "ValueFunctionPlayer",
    "alphabeta": "AlphaBetaPlayer",
    "sameturn": "SameTurnAlphaBetaPlayer",
    "playouts": "GreedyPlayoutsPlayer",
    "mcts": "MCTSPlayer",
}
needs_33 = pytest.mark.skipif(not AD.API_33, reason=f"catanatron {AD.CATANATRON_VERSION} ships no strong players")


def test_resolve_opponent_presets_and_paths():
    assert bench.resolve_opponent("vp").__name__ == "VictoryPointPlayer"
    assert bench.resolve_opponent("catanatron.players.weighted_random:WeightedRandomPlayer").__name__ == "WeightedRandomPlayer"
    assert bench.resolve_opponent("catanatron.players.weighted_random.WeightedRandomPlayer").__name__ == "WeightedRandomPlayer"
    with pytest.raises(SystemExit) as ex:
        bench.resolve_opponent("nonsense")
    assert ex.value.code == 2 and "unknown opponent" in str(ex.value)
    with pytest.raises(SystemExit) as ex:
        bench.resolve_opponent("no.such.module:Player")
    assert ex.value.code == 2 and "not available on catanatron" in str(ex.value)


def test_our_stand_ins_resolve_on_every_version():
    assert bench.resolve_opponent("vf").__name__ == "ValueFunctionPlayer"
    assert bench.resolve_opponent("ab").__name__ == "AlphaBetaPlayer"
    assert bench.resolve_opponent("vf").__module__ == "catanbot.bench.catanatron_players"


def test_strong_presets_resolve_only_on_the_33_engine():
    for name, cls_name in STRONG.items():
        if AD.API_33:
            cls = bench.resolve_opponent(name)
            assert cls.__name__ == cls_name and cls.__module__.startswith("catanatron.players.")
        else:
            with pytest.raises(bench.OpponentUnavailable) as ex:
                bench.resolve_opponent(name)
            msg = ex.value.message
            assert ex.value.code == 2 and "\n" not in msg
            assert f"not available on catanatron {AD.CATANATRON_VERSION}" in msg and name in msg


def test_list_opponents_reports_availability(capsys):
    rc = bench.main(["--list-opponents"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = {line.split()[0]: line for line in out.splitlines() if line.startswith("  ")}
    assert set(lines) == set(bench.OPPONENTS) | set(bench.PRESETS)
    for name in ("random", "weighted", "vp", "vf", "ab"):
        assert lines[name].rstrip().endswith("available") and "not available" not in lines[name]
    for name in STRONG:
        if AD.API_33:
            assert lines[name].rstrip().endswith("available") and "not available" not in lines[name]
        else:
            assert f"not available on catanatron {AD.CATANATRON_VERSION}" in lines[name]
    assert f"catanatron {AD.CATANATRON_VERSION}" in out
    for ladder in bench.LADDERS:
        assert f"ladder {ladder}" in out


def test_single_unavailable_opponent_exits_2_with_one_line(capsys, monkeypatch):
    monkeypatch.setitem(bench.PRESETS, "ghost", "catanatron.players.nowhere:GhostPlayer")
    rc = bench.main(["--games", "1", "--opponent", "ghost"])
    captured = capsys.readouterr()
    assert rc == 2
    err_lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(err_lines) == 1, captured.err
    assert f"not available on catanatron {AD.CATANATRON_VERSION}" in err_lines[0] and "ghost" in err_lines[0]
    assert "Traceback" not in captured.err and "==" not in captured.out


def test_unavailable_presets_in_a_list_are_skipped(capsys, monkeypatch):
    monkeypatch.setitem(bench.PRESETS, "ghost", "catanatron.players.nowhere:GhostPlayer")
    rc = bench.main(["--games", "1", "--opponent", "ghost,nonsense"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out.count("skipping") == 2 and "Traceback" not in captured.err
    assert "no opponent could be played" in captured.err


def test_ladder_skips_missing_and_runs_stock(capsys):
    rc = bench.main(["--games", "2", "--opponent", "weighted,alphabeta-missing", "--spec", "heuristic", "--seed", "3"])
    out = capsys.readouterr().out
    assert rc == 0 and "WeightedRandomPlayer" in out and "skipping" in out
    assert f"catanatron {AD.CATANATRON_VERSION}" in out
    assert "0 errors" in out


def test_standins_ladder_runs_one_game_each(capsys):
    rc = bench.main(["--games", "1", "--ladder", "standins", "--spec", "heuristic", "--seed", "8"])
    out = capsys.readouterr().out
    assert rc == 0 and "ladder summary" in out
    assert "vs 3x ValueFunctionPlayer (vf)" in out and "vs 3x AlphaBetaPlayer (ab)" in out
    assert "0 errors" in out and "Traceback" not in out


@needs_33
def test_smoke_against_catanatrons_value_player(capsys):
    rc = bench.main(["--games", "1", "--opponent", "value", "--spec", "heuristic", "--seed", "5"])
    out = capsys.readouterr().out
    assert rc == 0 and "vs 3x ValueFunctionPlayer (value)" in out and "catanatron 3.3" in out
    assert "0 errors" in out and "0 observe errors" in out


# ---------------------------------------------------------------------------
# --opponent-params, --trades, timing, --probe-trades, PYTHONHASHSEED pinning
# ---------------------------------------------------------------------------
SMALL = "search:depth=1,beam=2,expand=4,actions=3,evaluator=heuristic"
needs_trading = pytest.mark.skipif(not AD.DOMESTIC_TRADING,
                                   reason=f"catanatron {AD.CATANATRON_VERSION} has no player-to-player trading")


def test_parse_opponent_params():
    assert bench.parse_opponent_params(None) == {}
    assert bench.parse_opponent_params(" depth=3, prunning=true ,") == {"depth": "3", "prunning": "true"}
    with pytest.raises(bench.BadOpponentParams) as ex:
        bench.parse_opponent_params("depth")
    assert ex.value.code == 2


def test_opponent_factory_keyword_players():
    cls = bench.resolve_opponent("ab")
    make = bench.opponent_factory(cls, {"budget": "300", "depth": "2", "opening_book": "false"}, "ab")
    p = make(AD.COLORS[1])
    assert p.color == AD.COLORS[1] and p.budget == 300 and p.depth == 2 and p.opening_book is False
    assert bench.opponent_factory(cls, {}) is cls
    for bad in ({"nonsense": "1"}, {"budget": "lots"}):
        with pytest.raises(bench.BadOpponentParams) as ex:
            bench.opponent_factory(cls, bad, "ab")
        assert ex.value.code == 2 and "\n" not in ex.value.message and "ab" in ex.value.message
    with pytest.raises(bench.BadOpponentParams) as ex:
        bench.opponent_factory(bench.resolve_opponent("vp"), {"depth": "2"}, "vp")
    assert "takes no parameter depth" in ex.value.message


@needs_33
def test_opponent_factory_33_params():
    make = bench.opponent_factory(bench.resolve_opponent("alphabeta"), {"depth": "1", "prunning": "true"})
    p = make(AD.COLORS[2])
    assert p.params.depth == 1 and p.params.prunning is True and p.color == AD.COLORS[2]
    p = bench.opponent_factory(bench.resolve_opponent("mcts"), {"num_simulations": "7"})(AD.COLORS[0])
    assert p.params.num_simulations == 7
    p = bench.opponent_factory(bench.resolve_opponent("value"), {"value_fn": "contender"})(AD.COLORS[0])
    assert p.params.value_fn == "contender"
    for bad in ({"depth": "deep"}, {"value_fn": "mystery"}, {"num_playouts": "3"}):
        with pytest.raises(bench.BadOpponentParams):
            bench.opponent_factory(bench.resolve_opponent("alphabeta" if "depth" in bad else "value"), bad)


def test_bad_opponent_params_exit_2_with_one_line(capsys):
    rc = bench.main(["--games", "1", "--opponent", "ab", "--opponent-params", "wings=2"])
    captured = capsys.readouterr()
    assert rc == 2 and len([x for x in captured.err.splitlines() if x.strip()]) == 1
    assert "wings" in captured.err and "Traceback" not in captured.err


def test_per_game_timing_and_params_in_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = bench.main(["--games", "2", "--opponent", "ab", "--opponent-params", "budget=200,depth=1",
                     "--spec", SMALL, "--seed", "4", "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "compute     :" in text and "catanbot" in text and "AlphaBetaPlayer (each of 3)" in text
    import json
    data = json.loads(out.read_text())
    assert data["opponent_params"] == {"budget": "200", "depth": "1"} and data["trades"] == "off"
    assert "_times" not in json.dumps(data)
    for r in data["results"]:
        tm = r["timing"]
        assert tm["ours"]["n"] == r["stats"]["decisions"] > 0 and tm["opp"]["n"] > 0
        assert tm["ours"]["p95_ms"] >= tm["ours"]["p50_ms"] > 0 and len(tm["opp_seat_s"]) == 3
        assert r["trades"]["offers"] == 0 and r["trades"]["opp_asked"] == 0
    t = data["timing"]
    assert t["ours"]["n"] == sum(r["timing"]["ours"]["n"] for r in data["results"])
    assert t["opp"]["n"] == sum(r["timing"]["opp"]["n"] for r in data["results"])
    assert abs(t["our_s_per_game"] - sum(r["timing"]["ours"]["total_s"] for r in data["results"]) / 2) < 1e-9


@pytest.mark.skipif(AD.DOMESTIC_TRADING, reason="catanatron 3.3 has domestic trading")
def test_trades_need_the_33_engine(capsys):
    rc = bench.main(["--games", "1", "--opponent", "weighted", "--trades", "value"])
    assert rc == 2 and "needs catanatron 3.3" in capsys.readouterr().err
    rc = bench.main(["--probe-trades", "2", "--opponent", "weighted"])
    assert rc == 2


@needs_trading
def test_trades_value_game_reports_offers(tmp_path, capsys):
    out = tmp_path / "t.json"
    rc = bench.main(["--games", "2", "--opponent", "value", "--spec", SMALL, "--seed", "6", "--trades", "value",
                     "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "trades value" in text and "trades      : catanbot offered" in text
    assert "0 errors" in text and "0 observe errors" in text
    import json
    data = json.loads(out.read_text())
    tt = data["trade_totals"]
    assert tt["offers"] == sum(r["trades"]["offers"] for r in data["results"]) > 0
    assert tt["opp_accepted"] + tt["opp_rejected"] + tt["opp_cannot_pay"] == tt["opp_asked"] == 3 * tt["offers"]
    assert tt["confirmed"] + tt["cancelled"] == tt["offers_accepted"] and tt["opp_errors"] == 0


@needs_trading
def test_native_trades_against_alphabeta_count_errors_not_crashes(capsys):
    rc = bench.main(["--games", "1", "--opponent", "alphabeta", "--opponent-params", "depth=1", "--spec", SMALL,
                     "--seed", "2", "--trades", "native"])
    text = capsys.readouterr().out
    assert rc == 0 and "0 errors" in text
    line = next(x for x in text.splitlines() if x.startswith("  trades      :"))
    assert "accepted 0/" in line


@needs_trading
def test_playouts_print_is_silenced(capsys):
    from catanatron.models.enums import ActionType
    import catanatron.players.playouts as playouts_mod
    make = bench.opponent_factory(bench.resolve_opponent("playouts"), {"num_playouts": "1"})
    assert playouts_mod.print is bench._quiet_print and playouts_mod.USE_MULTIPROCESSING is False
    game = bench.probe_positions(1, seed=3)[0]
    import random as _random
    offer = bench.probe_offer(game, "1:1", _random.Random(1))
    game.execute(AD.CAction(game.state.colors[game.state.current_turn_index], ActionType.OFFER_TRADE, offer))
    player = make(game.state.current_color())
    capsys.readouterr()
    a = player.decide(game, game.playable_actions)
    assert a.action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE)
    assert capsys.readouterr().out == ""


@needs_trading
def test_probe_trades_tabulates_native_and_rule_answers(capsys, tmp_path):
    probe = bench.run_probe(["value", "alphabeta", "random"], 4, seed=2, opp_params={})
    table = probe["table"]
    for name in ("value", "alphabeta", "random"):
        for cat in bench.PROBE_CATEGORIES:
            row = table[name]["rows"][cat]
            assert row["n"] == 4 and row["accept"] + row["reject"] + row["error"] == 4
            assert 0 <= row["value_rule"] <= 4 and 0 <= row["fair_rule"] <= 4
    assert all(table["value"]["rows"][c]["accept"] == 0 for c in bench.PROBE_CATEGORIES)
    assert all(table["alphabeta"]["rows"][c]["error"] == 4 for c in bench.PROBE_CATEGORIES)
    assert all(table["value"]["rows"]["1:2"]["fair_rule"] == 0 for _ in [0])   # fair never pays 2 for 1
    bench.print_probe(probe)
    out = capsys.readouterr().out
    assert "trade-response probe" in out and "RuntimeError" in out
    rc = bench.main(["--probe-trades", "2", "--opponent", "value", "--seed", "1", "--json", str(tmp_path / "p.json")])
    assert rc == 0 and (tmp_path / "p.json").exists()


def test_hash_seed_reexec_makes_runs_reproducible(tmp_path):
    """Run as a script, the bench pins PYTHONHASHSEED (default 0): two processes play identical games."""
    import json
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONPATH"] = ROOT
    runs = []
    for i in range(2):
        out = tmp_path / f"h{i}.json"
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "bench_catanatron.py"), "--games", "2",
                               "--opponent", "weighted", "--spec", "heuristic", "--seed", "5", "--json", str(out)],
                              env=env, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        assert "PYTHONHASHSEED=0" in proc.stdout
        data = json.loads(out.read_text())
        assert data["hash_seed"] == "0"
        runs.append([(r["winner"], r["turns"], r["vps"], r["actions"]) for r in data["results"]])
    assert runs[0] == runs[1]
    # in-process calls (the tests above) never re-exec
    bench._reexec_with_hash_seed(-1)


# ---------------------------------------------------------------------------
# --mixed-opponents (1 catanbot seat + one copy of each of three presets) and --info
# ---------------------------------------------------------------------------
MIXED = ["random", "weighted", "vp"]     # present on every catanatron version, fast


def test_mixed_lineup_rotation_is_balanced_over_24_games():
    from collections import Counter
    names = ["value", "alphabeta", "sameturn"]
    for start in (0, 5, 17, 240):
        games = range(start, start + 24)
        lineups = [bench.mixed_lineup(g, names) for g in games]
        for g, lu in zip(games, lineups):
            assert lu[g % 4] == bench.CATANBOT                      # catanbot's seat rotates exactly as in 1v3
            assert sorted(x for x in lu if x != bench.CATANBOT) == sorted(names)
        rel = Counter((lu[(g % 4 + k) % 4], k) for g, lu in zip(games, lineups) for k in (1, 2, 3))
        assert set(rel.values()) == {8} and len(rel) == 9             # every preset x relative position: 8 games
        seats = Counter((nm, s) for lu in lineups for s, nm in enumerate(lu))
        assert set(seats.values()) == {6} and len(seats) == 16         # every player x absolute seat: 6 games
        assert len(set(lineups)) == 24                                 # the 24 (seat, permutation) cells, once each
    with pytest.raises(ValueError):
        bench.mixed_lineup(0, ["value", "alphabeta"])


def test_mixed_table_attributes_wins_and_positions():
    recs = [{"winner_name": "catanbot", "lineup": ["catanbot", "a", "b", "c"], "relative": ["a", "b", "c"],
             "vps": [10, 3, 4, 5]},
            {"winner_name": "b", "lineup": ["c", "catanbot", "a", "b"], "relative": ["a", "b", "c"],
             "vps": [2, 6, 7, 10]},
            {"winner_name": None, "lineup": ["b", "c", "catanbot", "a"], "relative": ["a", "b", "c"],
             "vps": [9, 9, 9, 9], "crashed": False}]
    t = bench.mixed_table(recs, ["a", "b", "c"])
    assert t["wins"] == {"catanbot": 1, "a": 0, "b": 1, "c": 0, "none": 1}
    assert t["avg_vp"]["catanbot"] == pytest.approx((10 + 6 + 9) / 3) and t["avg_vp"]["b"] == pytest.approx((4 + 10 + 9) / 3)
    assert t["positions"]["a"] == {"1": 3, "2": 0, "3": 0}


def test_mixed_games_record_lineup_and_winner(tmp_path, capsys):
    import json
    out = tmp_path / "mixed.json"
    logs = tmp_path / "logs"
    rc = bench.main(["--game-range", "3:5", "--mixed-opponents", ",".join(MIXED), "--spec", SMALL, "--seed", "4",
                     "--log-actions", str(logs), "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "mixed wins" in text and "0 errors" in text
    s = json.loads(out.read_text())
    assert s["format"] == "1v3-mixed" and s["opponents"] == MIXED and s["info"] == {"mode": "full"}
    assert s["opponent_class"] == "RandomPlayer+WeightedRandomPlayer+VictoryPointPlayer"
    for r in s["results"]:
        g = r["game"]
        assert r["lineup"] == list(bench.mixed_lineup(g, MIXED)) and r["seat"] == g % 4
        w = r["winner_seat"]
        assert r["winner_name"] == (r["lineup"][w] if w >= 0 else None)
        assert r["won"] == (r["winner_name"] == bench.CATANBOT)
        assert set(r["timing"]["opp_by_name"]) == set(MIXED)
    wins = s["mixed"]["wins"]
    assert sum(wins.values()) == s["games"] == 2 and wins["catanbot"] == s["wins"]
    assert set(s["timing"]["opp_by_name"]) == set(MIXED)
    assert all(t["all"]["n"] > 0 for t in s["timing"]["opp_by_name"].values())
    import gzip
    (path,) = list(logs.iterdir())
    assert "1v3-mixed" in path.name
    recs = [json.loads(line) for line in gzip.open(path, "rt")]
    for rec in recs:
        assert rec["match"] == "1v3-mixed" and rec["lineup"] == list(bench.mixed_lineup(rec["game"], MIXED))
        assert [p.get("preset") or bench.CATANBOT for p in rec["players"]] == rec["lineup"]


def test_mixed_chunks_reproduce_one_run(tmp_path, capsys):
    """Per-game seeds and lineups depend on the game index alone: chunks 0:1 + 1:3 == one run 0:3
    (also in the counted information mode)."""
    import json

    def run(rng, name, extra=()):
        out = tmp_path / name
        rc = bench.main(["--game-range", rng, "--mixed-opponents", ",".join(MIXED), "--spec", SMALL, "--seed", "9",
                         "--json", str(out)] + list(extra))
        capsys.readouterr()
        assert rc == 0
        return {r["game"]: (r["lineup"], r["winner_name"], r["vps"], r["turns"], r["actions"])
                for r in json.loads(out.read_text())["results"]}

    whole = run("0:3", "whole.json", ["--info", "counted", "--info-samples", "2"])
    parts = {**run("0:1", "a.json", ["--info", "counted", "--info-samples", "2"]),
             **run("1:3", "b.json", ["--info", "counted", "--info-samples", "2"])}
    assert whole == parts and sorted(whole) == [0, 1, 2]


def test_mixed_option_validation(capsys):
    with pytest.raises(SystemExit):
        bench.main(["--mixed-opponents", "random,weighted"])
    with pytest.raises(SystemExit):
        bench.main(["--mixed-opponents", ",".join(MIXED), "--our-seats", "2"])
    rc = bench.main(["--games", "1", "--mixed-opponents", "random,weighted,nonsense"])
    assert rc == 2 and "unknown opponent" in capsys.readouterr().err


def test_info_counted_is_recorded_in_json_and_logs(tmp_path, capsys):
    import gzip
    import json
    out = tmp_path / "c.json"
    rc = bench.main(["--games", "1", "--opponent", "weighted", "--spec", SMALL, "--info", "counted",
                     "--info-samples", "3", "--discards-public", "--log-actions", str(tmp_path / "l"),
                     "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "info counted K=3 discards-public" in text and "information : counted" in text
    s = json.loads(out.read_text())
    assert s["info"] == {"mode": "counted", "samples": 3, "discards_public": True}
    assert s["info_stats"]["info_errors"] == 0 and s["info_stats"]["hidden_discards"] == 0
    assert s["stats"]["errors"] == 0 and s["stats"]["fallback"] == 0
    (path,) = list((tmp_path / "l").iterdir())
    rec = json.loads(gzip.open(path, "rt").readline())
    ours = [p for p in rec["players"] if p["kind"] == "catanbot"]
    assert ours[0]["info"] == {"mode": "counted", "samples": 3, "discards_public": True}
    with pytest.raises(SystemExit):
        bench.main(["--info-samples", "0"])


# ---------------------------------------------------------------------------
# --players 2: 1v1 (catanbot against one opponent, catanbot in seat g % 2; docs/BENCH_1V1_PROTOCOL.md)
# ---------------------------------------------------------------------------
def _load_script(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def test_one_v_one_seats_alternate_and_formats():
    assert [bench.our_seats_for(g, 1, players=2) for g in range(5)] == [(0,), (1,), (0,), (1,), (0,)]
    plan = bench.plan_games(910501, range(400), players=2)
    assert [s for _, _, s in plan].count((0,)) == 200 and [s for _, _, s in plan].count((1,)) == 200
    assert bench.plan_games(7, range(3, 5), players=2) == [(3, bench.game_seed(7, 3), (1,)),
                                                           (4, bench.game_seed(7, 4), (0,))]
    assert bench.match_format(1, 2) == "1v1" and bench.match_format(1) == "1v3" and bench.match_format(2) == "2v2"
    assert bench.match_format(1, 4, mixed=True) == "1v3-mixed"
    with pytest.raises(ValueError):
        bench.our_seats_for(0, 2, players=2)
    with pytest.raises(ValueError):
        bench.our_seats_for(0, 1, players=3)
    # the 4-player seatings are untouched
    assert [bench.our_seats_for(g) for g in range(5)] == [(0,), (1,), (2,), (3,), (0,)]
    assert [bench.our_seats_for(g, 2) for g in range(6)] == list(bench.ARRANGEMENTS_2V2)


def test_one_v_one_games_results_logs_and_replay(tmp_path, capsys):
    import gzip
    import json
    out = tmp_path / "h2h.json"
    logs = tmp_path / "logs"
    rc = bench.main(["--players", "2", "--game-range", "0:2", "--opponent", "weighted", "--spec", SMALL, "--seed", "6",
                     "--rerun-crashes", "--log-actions", str(logs), "--verbose", "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "vs 1x WeightedRandomPlayer (weighted)" in text and "1v1 null is 50%" in text
    assert "0 errors, 0 fallbacks" in text and "0 observe errors" in text
    s = json.loads(out.read_text())
    assert s["format"] == "1v1" and s["players"] == 2 and s["our_seats"] == 1 and s["null_win_rate"] == 0.5
    assert s["by_seat"] == {"0": {"wins": s["results"][0]["won"] * 1, "games": 1},
                            "1": {"wins": s["results"][1]["won"] * 1, "games": 1}}
    assert s["game_range"] == [0, 2] and s["crash_attempts"] == 0 and s["seed"] == 6
    for r in s["results"]:
        g = r["game"]
        assert r["seat"] == g % 2 and r["our_seats"] == [g % 2] and r["players"] == 2
        assert r["colors"] == ["RED", "BLUE"] and len(r["vps"]) == 2 and len(r["opp_vps"]) == 1
        assert r["won"] == (r["winner_seat"] == g % 2) and r["seed"] == bench.game_seed(6, g)
        assert len(r["timing"]["opp_seat_s"]) == 1
    (path,) = list(logs.iterdir())
    assert path.name == "weighted_1v1_seed6_g00000-00002.jsonl.gz"
    recs = [json.loads(line) for line in gzip.open(path, "rt")]
    assert sorted(r["game"] for r in recs) == [0, 1]
    for rec in recs:
        g = rec["game"]
        assert rec["match"] == "1v1" and rec["colors"] == ["RED", "BLUE"] and rec["our_seats"] == [g % 2]
        assert [p["kind"] for p in rec["players"]] == (["catanbot", "opponent"] if g % 2 == 0
                                                       else ["opponent", "catanbot"])
    replay = _load_script("replay_catanatron")      # what `replay_catanatron.py --check` runs per game
    for rec in recs:
        ok, msg = replay.check_record(rec)
        assert ok, msg


def test_one_v_one_chunks_reproduce_one_run(tmp_path, capsys):
    import json

    def run(rng, name):
        out = tmp_path / name
        rc = bench.main(["--players", "2", "--game-range", rng, "--opponent", "weighted", "--spec", SMALL,
                         "--seed", "12", "--json", str(out)])
        capsys.readouterr()
        assert rc == 0
        return {r["game"]: (r["seat"], r["winner"], r["vps"], r["turns"], r["actions"])
                for r in json.loads(out.read_text())["results"]}

    whole = run("0:3", "whole.json")
    parts = {**run("0:1", "a.json"), **run("1:3", "b.json")}
    assert whole == parts and sorted(whole) == [0, 1, 2]


def test_one_v_one_crashed_games_are_losses(monkeypatch):
    calls = []

    def boom(job, opts):
        calls.append(job[0])
        raise RuntimeError("boom")

    monkeypatch.setattr(bench, "_play_one", boom)
    job = (3, 5, SMALL, "weighted", 10, 7, "off", {}, {"our_seats": 1, "players": 2, "rerun_crashes": True})
    r = bench.run_one(job)
    assert calls == [3, 3] and r["crashed"] and r["crashes"] == 2 and r["seat"] == 1 and r["our_seats"] == [1]
    assert r["colors"] == ["RED", "BLUE"] and r["vps"] == [0, 0] and r["opp_vps"] == [0] and r["players"] == 2
    s = bench.summarize([r], SMALL, "weighted", 5, players=2)
    assert s["format"] == "1v1" and s["wins"] == 0 and s["crashed_games"] == 1 and s["truncated"] == 0
    assert s["by_seat"] == {0: {"wins": 0, "games": 0}, 1: {"wins": 0, "games": 1}}
    assert bench.summarize([r], SMALL, "weighted", 5)["format"] == "1v1"     # players read from the records


def test_one_v_one_option_validation(capsys):
    with pytest.raises(SystemExit):
        bench.main(["--players", "2", "--our-seats", "2"])
    with pytest.raises(SystemExit):
        bench.main(["--players", "2", "--mixed-opponents", ",".join(MIXED)])
    with pytest.raises(SystemExit):
        bench.main(["--players", "3"])


def test_one_v_one_against_our_stand_ins(capsys):
    rc = bench.main(["--players", "2", "--games", "1", "--ladder", "standins", "--spec", SMALL, "--seed", "8"])
    out = capsys.readouterr().out
    assert rc == 0 and "vs 1x ValueFunctionPlayer (vf)" in out and "vs 1x AlphaBetaPlayer (ab)" in out
    assert "1v1 games; 50% = parity" in out and out.count("0 errors, 0 fallbacks") == 2 and "Traceback" not in out


@needs_33
def test_one_v_one_against_catanatrons_value_and_alphabeta(capsys):
    rc = bench.main(["--players", "2", "--games", "1", "--opponent", "value,alphabeta", "--spec", SMALL, "--seed", "5"])
    out = capsys.readouterr().out
    assert rc == 0 and "vs 1x ValueFunctionPlayer (value)" in out and "vs 1x AlphaBetaPlayer (alphabeta)" in out
    assert "catanatron 3.3" in out and out.count("0 errors, 0 fallbacks") == 2


def test_analyze_1v1_registration_statistics_and_statement(tmp_path, capsys):
    """scripts/analyze_1v1.py (the protocol's analysis) on synthetic chunks: wins recomputed from the
    winner seat, exact one-sided p against 1/2, the HexMachina wording, the H1 -> H2 gate and the
    registration checks."""
    import json
    from fractions import Fraction
    an = _load_script("analyze_1v1")

    def records(tid, wins):
        t = an.TESTS[tid]
        out = []
        for g in range(t.games):
            seat = g % 2
            won = g < wins
            w = seat if won else 1 - seat
            out.append({"game": g, "seed": an.game_seed(t.seed, g), "seat": seat, "our_seats": [seat],
                        "colors": ["RED", "BLUE"], "winner": ["RED", "BLUE"][w], "winner_seat": w,
                        "won": not won,     # ignored: the analysis recomputes wins from the winner seat
                        "vps": [10, 7] if w == 0 else [7, 10], "turns": 60, "crashes": 0,
                        "stats": {"errors": 0, "observe_errors": 0, "fallback": 0, "unmapped_top": 0}})
        return out

    def meta(tid):
        t = an.TESTS[tid]
        return {"format": "1v1", "players": 2, "our_seats": 1, "null_win_rate": 0.5, "opponent": t.opponent,
                "opponent_class": t.opponent_class, "opponent_params": {}, "catanatron": "3.3.0", "spec": an.SPEC,
                "seed": t.seed, "trades": "off", "hash_seed": "0", "vps_to_win": 10, "discard_limit": 7,
                "info": {"mode": "full"}}

    def write(tid, wins, tweak=None):
        d = tmp_path / f"{tid}_{wins}_{bool(tweak)}"
        d.mkdir()
        recs = records(tid, wins)
        for a, b in ((0, 200), (200, 400)):
            m = meta(tid)
            if tweak:
                tweak(m, recs)
            (d / f"g{a:05d}-{b:05d}.json").write_text(json.dumps({**m, "game_range": [a, b], "results": recs[a:b]}))
        return str(d)

    out = tmp_path / "a.json"
    rc = an.main(["--test", f"H1={write('H1', 240)}", "--test", f"H2={write('H2', 230)}", "--json", str(out),
                  "--markdown", str(tmp_path / "a.md")])
    text = capsys.readouterr().out
    assert rc == 0 and "REJECTED H0" in text and "H2 (gated by H1) REJECTED" in text
    res = json.loads(out.read_text())
    h1 = res["analysis"]["H1"]
    assert h1["registered"] and h1["deviations"] == [] and h1["games"] == 400 and h1["wins"] == 240
    assert h1["p_one_sided"]["fraction"] == str(an.PS.binom_sf_exact(240, 400, Fraction(1, 2)))
    assert h1["by_seat"]["0"]["games"] == 200
    # scipy.stats.binomtest(240, 400).proportion_ci(0.95, "exact") = (0.550146..., 0.648362...); p = 3.7133e-05
    assert abs(h1["ci95"][0] - 0.5501461617) < 1e-8 and abs(h1["ci95"][1] - 0.6483628520) < 1e-8
    assert abs(h1["p_one_sided"]["float"] - 3.713284038e-05) < 1e-12
    assert res["verdict"]["hexmachina_position"] == "above" and "54.1%" in res["verdict"]["hexmachina_statement"]
    assert "unconfirmed" in res["verdict"]["hexmachina_statement"]
    assert "| H1 (primary) | AlphaBetaPlayer | 400 | 240 | 60.0% |" in (tmp_path / "a.md").read_text()
    # 210/400 (52.5 %): not significant, 54.1 % inside the interval, the gate stays closed for H2
    an.main(["--test", f"H1={write('H1', 210)}", "--test", f"H2={write('H2', 300)}"])
    text = capsys.readouterr().out
    assert "NOT rejected" in text and "not distinguishable" in text and "gate is closed" in text
    a = an.analyze_test("H1", an.PS.load_chunks(write("H1", 190)))
    assert an.hexmachina_position(a["ci95"]) == "below"
    # deviations: a non-default opponent parameter and a game in the wrong seat -> not the registered data

    def tweak(m, recs):
        m["opponent_params"] = {"depth": "3"}
        recs[5]["seat"] = 0
    an.main(["--test", f"H1={write('H1', 240, tweak)}"])
    text = capsys.readouterr().out
    assert "NOT THE REGISTERED DATA" in text and "opponent_params" in text and "game 5: catanbot seat" in text


# ---------------------------------------------------------------------------
# --our-seats 3 (3v1) and --our-seats 2 --mixed-opponents A,B (2v2-mixed); docs/BENCH_MULTI_PROTOCOL.md.
# Games here use the protocol's smoke seeds only (424901, 425001, 425101).
# ---------------------------------------------------------------------------
PAIR = ["random", "weighted"]      # two presets present on every catanatron version, fast


def test_three_v_one_seats_rotate_the_lone_opponent():
    from collections import Counter
    for g in range(8):
        seats = bench.our_seats_for(g, 3)
        assert len(seats) == 3 and g % 4 not in seats and seats == bench.ARRANGEMENTS_3V1[g % 4]
        assert bench.arrangement_index(g, 3) == g % 4
    opp = Counter(next(s for s in range(4) if s not in bench.our_seats_for(g, 3)) for g in range(400))
    assert opp == {0: 100, 1: 100, 2: 100, 3: 100}
    assert bench.match_format(3) == "3v1" and bench.match_format(2, mixed=True) == "2v2-mixed"
    assert [bench.seat_pattern(a) for a in bench.ARRANGEMENTS_3V1] == ["oCCC", "CoCC", "CCoC", "CCCo"]
    assert bench.plan_games(7, range(5, 7), our_seats=3) == [(5, bench.game_seed(7, 5), (0, 2, 3)),
                                                             (6, bench.game_seed(7, 6), (0, 1, 3))]
    with pytest.raises(ValueError):
        bench.our_seats_for(0, 4)
    # the older formats' seatings and tags are untouched
    assert [bench.our_seats_for(g, 2) for g in range(6)] == list(bench.ARRANGEMENTS_2V2)
    assert bench.match_format(1) == "1v3" and bench.match_format(2) == "2v2" and bench.match_format(1, mixed=True) == "1v3-mixed"
    assert bench.arrangement_index(7, 2) == 1 and bench.arrangement_index(7, 2, mixed=True) == 7


def test_two_v_two_mixed_placements_cycle_through_all_twelve():
    from collections import Counter
    names = ["value", "alphabeta"]
    for start in (0, 5, 17, 396):
        games = range(start, start + 12)
        lineups = [bench.mixed_lineup_2v2(g, names) for g in games]
        assert len(set(lineups)) == 12                                   # the 12 placements, once each
        for g, lu in zip(games, lineups):
            assert tuple(s for s, x in enumerate(lu) if x == bench.CATANBOT) == bench.our_seats_for(g, 2)
            opp = [x for x in lu if x != bench.CATANBOT]
            assert opp == (names if (g // 6) % 2 == 0 else names[::-1])  # turn order of the two presets
            assert bench.lineup_for(g, names, 2) == lu
        seats = Counter((nm, s) for lu in lineups for s, nm in enumerate(lu))
        assert all(seats[(nm, s)] == 3 for nm in names for s in range(4))  # each preset in each seat 3 times
        assert all(seats[(bench.CATANBOT, s)] == 6 for s in range(4))
    assert bench.lineup_for(3, ["a", "b", "c"], 1) == bench.mixed_lineup(3, ["a", "b", "c"])
    assert bench.lineup_for(3, None, 2) is None
    with pytest.raises(ValueError):
        bench.mixed_lineup_2v2(0, ["value", "alphabeta", "sameturn"])


def test_three_v_one_games_results_logs_and_replay(tmp_path, capsys):
    import gzip
    import json
    out = tmp_path / "3v1.json"
    logs = tmp_path / "logs"
    rc = bench.main(["--our-seats", "3", "--game-range", "1:3", "--opponent", "weighted", "--spec", SMALL,
                     "--seed", "424901", "--rerun-crashes", "--log-actions", str(logs), "--verbose", "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "== 3x catanbot" in text and "vs 1x WeightedRandomPlayer (weighted)" in text
    assert "the 3v1 null is 75%" in text and "opp by seat :" in text and "0 errors, 0 fallbacks" in text
    s = json.loads(out.read_text())
    assert s["format"] == "3v1" and s["our_seats"] == 3 and s["null_win_rate"] == 0.75 and "players" not in s
    assert s["opp_null_win_rate"] == 0.25 and s["game_range"] == [1, 3] and s["crash_attempts"] == 0
    assert s["opp_wins"] == sum(r["opp_won"] for r in s["results"])
    assert s["wins"] + s["opp_wins"] + s["no_winner"] == s["games"] == 2
    assert s["opp_by_seat"] == {str(k): {"wins": int(s["results"][k - 1]["opp_won"]) if k in (1, 2) else 0,
                                         "games": 1 if k in (1, 2) else 0} for k in range(4)}
    assert set(s["by_arrangement"]) == {"oCCC", "CoCC", "CCoC", "CCCo"}
    for r in s["results"]:
        g = r["game"]
        assert r["format"] == "3v1" and r["opp_seat"] == g % 4 and r["opponent"] == "weighted"
        assert r["our_seats"] == [x for x in range(4) if x != g % 4] and r["arrangement"] == g % 4
        assert r["won"] == (r["winner_seat"] in r["our_seats"]) and r["opp_won"] == (r["winner_seat"] == g % 4)
        assert len(r["our_vps"]) == 3 and len(r["opp_vps"]) == 1 and len(r["timing"]["opp_seat_s"]) == 1
        assert set(r["stats_by_seat"]) == {str(x) for x in r["our_seats"]}
    (path,) = list(logs.iterdir())
    assert path.name == "weighted_3v1_seed424901_g00001-00003.jsonl.gz"
    recs = [json.loads(line) for line in gzip.open(path, "rt")]
    replay = _load_script("replay_catanatron")
    for rec in recs:
        g = rec["game"]
        assert rec["match"] == "3v1" and rec["our_seats"] == [x for x in range(4) if x != g % 4]
        kinds = [p["kind"] for p in rec["players"]]
        assert kinds.count("catanbot") == 3 and kinds[g % 4] == "opponent"
        seeds = [p["bot_seed"] for p in rec["players"] if p["kind"] == "catanbot"]
        assert len(set(seeds)) == 3 and seeds[0] == rec["seed"]           # independent bots, distinct seeds
        ok, msg = replay.check_record(rec)
        assert ok, msg


def test_two_v_two_mixed_games_results_logs_and_replay(tmp_path, capsys):
    import gzip
    import json
    out = tmp_path / "m.json"
    logs = tmp_path / "logs"
    rc = bench.main(["--our-seats", "2", "--mixed-opponents", ",".join(PAIR), "--game-range", "5:7", "--spec", SMALL,
                     "--seed", "425101", "--log-actions", str(logs), "--verbose", "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "2v2-mixed" in text and "mixed wins  :" in text and "the 2v2 null is 50%" in text
    assert "0 errors, 0 fallbacks" in text
    s = json.loads(out.read_text())
    assert s["format"] == "2v2-mixed" and s["opponents"] == PAIR and s["our_seats"] == 2 and s["null_win_rate"] == 0.5
    assert s["opponent_class"] == "RandomPlayer+WeightedRandomPlayer" and "opp_wins" not in s
    mx = s["mixed"]
    assert sum(mx["wins"].values()) == s["games"] == 2 and mx["wins"]["catanbot"] == s["wins"]
    assert set(mx["by_seat"]) == {"catanbot", *PAIR} and "positions" not in mx
    assert sum(v["games"] for v in mx["placements"].values()) == 2
    assert set(s["timing"]["opp_by_name"]) == set(PAIR)
    for r in s["results"]:
        g = r["game"]
        assert r["format"] == "2v2-mixed" and r["arrangement"] == g % 12 and "relative" not in r
        assert r["lineup"] == list(bench.mixed_lineup_2v2(g, PAIR)) and r["our_seats"] == list(bench.our_seats_for(g, 2))
        w = r["winner_seat"]
        assert r["winner_name"] == (r["lineup"][w] if w >= 0 else None)
        assert r["won"] == (r["winner_name"] == bench.CATANBOT)
        for nm in PAIR:
            seat = r["lineup"].index(nm)
            assert mx["by_seat"][nm][str(seat)]["games"] >= 1
    (path,) = list(logs.iterdir())
    assert path.name == "random_weighted_2v2-mixed_seed425101_g00005-00007.jsonl.gz"
    recs = [json.loads(line) for line in gzip.open(path, "rt")]
    replay = _load_script("replay_catanatron")
    for rec in recs:
        assert rec["match"] == "2v2-mixed" and rec["lineup"] == list(bench.mixed_lineup_2v2(rec["game"], PAIR))
        assert [p.get("preset") or bench.CATANBOT for p in rec["players"]] == rec["lineup"]
        ok, msg = replay.check_record(rec)
        assert ok, msg


def test_multi_seat_chunks_reproduce_one_run(tmp_path, capsys):
    """Per-game seeds and seatings depend on the game index alone: chunks 0:1 + 1:3 == one run 0:3."""
    import json

    def run(extra, rng, name):
        out = tmp_path / name
        rc = bench.main(extra + ["--game-range", rng, "--spec", SMALL, "--seed", "425001", "--json", str(out)])
        capsys.readouterr()
        assert rc == 0
        return {r["game"]: (r["our_seats"], r.get("lineup"), r["winner"], r["vps"], r["turns"], r["actions"])
                for r in json.loads(out.read_text())["results"]}

    for i, extra in enumerate((["--our-seats", "3", "--opponent", "weighted"],
                               ["--our-seats", "2", "--mixed-opponents", ",".join(PAIR)])):
        whole = run(extra, "0:3", f"w{i}.json")
        parts = {**run(extra, "0:1", f"a{i}.json"), **run(extra, "1:3", f"b{i}.json")}
        assert whole == parts and sorted(whole) == [0, 1, 2]


def test_multi_seat_crashed_games_are_losses(monkeypatch):
    def boom(job, opts):
        raise RuntimeError("boom")

    monkeypatch.setattr(bench, "_play_one", boom)
    r = bench.run_one((6, 5, SMALL, "weighted", 10, 7, "off", {}, {"our_seats": 3, "rerun_crashes": True}))
    assert r["crashed"] and r["crashes"] == 2 and r["format"] == "3v1" and r["opp_seat"] == 2
    assert r["opponent"] == "weighted" and r["opp_won"] is False and r["our_seats"] == [0, 1, 3]
    assert r["our_vps"] == [0, 0, 0] and r["opp_vps"] == [0] and r["arrangement"] == 2 and r["pattern"] == "CCoC"
    s = bench.summarize([r], SMALL, "weighted", 5)
    assert s["format"] == "3v1" and s["wins"] == 0 and s["opp_wins"] == 0 and s["no_winner"] == 1
    assert s["crashed_games"] == 1 and s["truncated"] == 0 and s["opp_by_seat"][2] == {"wins": 0, "games": 1}
    m = bench.run_one((7, 5, SMALL, "random,weighted", 10, 7, "off", {},
                       {"our_seats": 2, "mixed": PAIR, "rerun_crashes": True}))
    assert m["crashed"] and m["format"] == "2v2-mixed" and m["arrangement"] == 7 and m["winner_name"] is None
    assert m["lineup"] == list(bench.mixed_lineup_2v2(7, PAIR)) and "relative" not in m and m["our_vps"] == [0, 0]
    s = bench.summarize([m], SMALL, "random,weighted", 5, mixed=PAIR)
    assert s["format"] == "2v2-mixed" and s["mixed"]["wins"]["none"] == 1 and s["wins"] == 0
    # the older formats' records gain no key
    for opts in ({"our_seats": 1}, {"our_seats": 2}, {"our_seats": 1, "mixed": MIXED}, {"our_seats": 1, "players": 2}):
        c = bench.run_one((3, 5, SMALL, "weighted", 10, 7, "off", {}, dict(opts, rerun_crashes=True)))
        assert not {"format", "opp_seat", "opp_won", "opponent"} & set(c)
        assert not {"opp_wins", "opp_by_seat", "no_winner"} & set(bench.summarize([c], SMALL, "weighted", 5,
                                                                                 mixed=opts.get("mixed")))


def test_multi_seat_option_validation(capsys):
    for argv in (["--our-seats", "3", "--mixed-opponents", ",".join(MIXED)],
                 ["--our-seats", "3", "--mixed-opponents", ",".join(PAIR)],
                 ["--our-seats", "2", "--mixed-opponents", ",".join(MIXED)],
                 ["--our-seats", "2", "--mixed-opponents", "random"],
                 ["--players", "2", "--our-seats", "3"],
                 ["--our-seats", "4"]):
        with pytest.raises(SystemExit):
            bench.main(argv)
    rc = bench.main(["--games", "1", "--our-seats", "2", "--mixed-opponents", "random,nonsense"])
    assert rc == 2 and "unknown opponent" in capsys.readouterr().err


@needs_33
def test_multi_seat_formats_against_catanatrons_value_player(capsys):
    rc = bench.main(["--our-seats", "3", "--games", "1", "--opponent", "value", "--spec", SMALL, "--seed", "424901"])
    out = capsys.readouterr().out
    assert rc == 0 and "vs 1x ValueFunctionPlayer (value)" in out and "catanatron 3.3" in out
    assert "0 errors, 0 fallbacks" in out
    rc = bench.main(["--our-seats", "2", "--mixed-opponents", "value,vp", "--games", "1", "--spec", SMALL,
                     "--seed", "425101"])
    out = capsys.readouterr().out
    assert rc == 0 and "ValueFunctionPlayer+VictoryPointPlayer" in out and "0 errors, 0 fallbacks" in out


def test_analyze_multi_registration_statistics_and_holm(tmp_path, capsys):
    """scripts/analyze_multi.py (the protocol's analysis) on synthetic chunks: wins recomputed from the
    winner seat and the registered seating, the conservative 3v1 count (turn-cap / crashed games count
    for the opponent), exact lower / upper tails, Holm over M1-M3 and the registration checks."""
    import json
    from fractions import Fraction
    an = _load_script("analyze_multi")

    def records(tid, not_ours, capped=0):
        """``not_ours`` games not won by our seats (the first ``capped`` of them without a winner)."""
        t = an.TESTS[tid]
        out = []
        for g in range(t.games):
            lineup = an.registered_lineup(t, g)
            ours = [s for s, nm in enumerate(lineup) if nm == "catanbot"]
            theirs = [s for s, nm in enumerate(lineup) if nm != "catanbot"]
            w = -1 if g < capped else theirs[g % len(theirs)] if g < not_ours else ours[g % len(ours)]
            r = {"game": g, "seed": an.game_seed(t.seed, g), "our_seats": ours, "format": t.fmt,
                 "colors": ["RED", "BLUE", "ORANGE", "WHITE"], "winner": None if w < 0 else "RED", "winner_seat": w,
                 "won": False,      # ignored: recomputed from the winner seat
                 "vps": [10 if s == w else 6 for s in range(4)], "turns": 70, "crashes": 0,
                 "stats": {"errors": 0, "observe_errors": 0, "fallback": 0, "unmapped_top": 0}}
            if t.fmt == "3v1":
                r.update(opp_seat=g % 4, opponent=t.opponent)
            else:
                r["lineup"] = lineup
            out.append(r)
        return out

    def meta(tid):
        t = an.TESTS[tid]
        m = {"format": t.fmt, "our_seats": 3 if t.fmt == "3v1" else 2, "null_win_rate": 0.75 if t.fmt == "3v1" else 0.5,
             "opponent": t.opponent, "opponent_class": t.opponent_class, "opponent_params": {}, "catanatron": "3.3.0",
             "spec": an.SPEC, "seed": t.seed, "trades": "off", "hash_seed": "0", "vps_to_win": 10, "discard_limit": 7,
             "info": {"mode": "full"}}
        m.update({"opp_null_win_rate": 0.25} if t.fmt == "3v1" else {"opponents": list(t.opponents)})
        return m

    def write(tid, not_ours, capped=0, tweak=None):
        d = tmp_path / f"{tid}_{not_ours}_{capped}_{bool(tweak)}"
        d.mkdir()
        recs = records(tid, not_ours, capped)
        for a, b in ((0, 200), (200, 400)):
            m = meta(tid)
            if tweak:
                tweak(m, recs)
            (d / f"g{a:05d}-{b:05d}.json").write_text(json.dumps({**m, "game_range": [a, b], "results": recs[a:b]}))
        return str(d)

    out = tmp_path / "a.json"
    # M1: 60 games not won by us (55 opponent wins + 5 turn-cap games); M2: 90; M3: 170 not ours = 230 ours
    rc = an.main(["--test", f"M1={write('M1', 60, capped=5)}", "--test", f"M2={write('M2', 90)}",
                  "--test", f"M3={write('M3', 170)}", "--json", str(out), "--markdown", str(tmp_path / "a.md")])
    text = capsys.readouterr().out
    assert rc == 0 and "FAMILY (Holm over M1-M3, alpha 0.05): complete" in text
    res = json.loads(out.read_text())
    m1, m2, m3 = (res["analysis"][t] for t in ("M1", "M2", "M3"))
    assert m1["registered"] and m1["deviations"] == [] and m1["count"] == 60 and m1["our_wins"] == 340
    assert m1["wins"]["alphabeta"] == 55 and m1["no_winner"] == 5 and m1["turn_cap_games"] == 5
    assert m1["by_seat"]["alphabeta"]["0"]["games"] == 100
    assert m1["p_one_sided"]["fraction"] == str(an.PS.binom_cdf_exact(60, 400, Fraction(1, 4)))
    # scipy.stats.binomtest(60, 400, 0.25, alternative="less").pvalue = 7.8008e-07; CP 95 % (0.116458, 0.188822)
    assert abs(m1["p_one_sided"]["float"] - 7.800766112735117e-07) < 1e-15
    assert abs(m1["ci95"][0] - 0.11645847860739882) < 1e-8 and abs(m1["ci95"][1] - 0.18882213648142618) < 1e-8
    # binomtest(90, 400, 0.25, "less") = 0.135846; binomtest(230, 400, 0.5, "greater") = 0.0015645, CP (0.5249, 0.6240)
    assert abs(m2["p_one_sided"]["float"] - 0.1358459093754855) < 1e-12 and m2["tail"] == "lower"
    assert m3["count"] == 230 and m3["tail"] == "upper" and abs(m3["p_one_sided"]["float"] - 0.0015645080634072589) < 1e-12
    assert abs(m3["ci95"][0] - 0.5248998335723469) < 1e-8 and abs(m3["ci95"][1] - 0.6239822512451211) < 1e-8
    assert m3["wins"]["value"] + m3["wins"]["alphabeta"] == 170 and sum(v["games"] for v in m3["by_pattern"].values()) == 400
    # Holm: M1 (7.8e-7 <= 0.05/3) and M3 (0.0016 <= 0.05/2) rejected, M2 (0.136 > 0.05) not
    v = res["verdict"]["tests"]
    assert v["M1"]["rejected"] and v["M3"]["rejected"] and not v["M2"]["rejected"]
    assert "| M1 | 3v1 | AlphaBetaPlayer | 400 | 60 (not won by us) | 15.0% |" in (tmp_path / "a.md").read_text()
    # a missing test enters with p = 1 (the family stays m = 3): M1 alone still needs p <= 0.05 / 3
    an.main(["--test", f"M1={write('M1', 81)}"])
    text = capsys.readouterr().out
    assert "INCOMPLETE: M2, M3" in text and "M1: H0" in text and "REJECTED" in text      # 81 = the rejection point
    an.main(["--test", f"M1={write('M1', 82)}"])
    assert "M1: H0 (AlphaBetaPlayer wins >= 25% of the 3v1 games) NOT rejected" in capsys.readouterr().out

    # deviations: a non-default parameter, the opponent in the wrong seat, a wrong M3 lineup -> not the registered data
    def tweak(m, recs):
        m["opponent_params"] = {"depth": "3"}
        recs[5]["opp_seat"] = 0

    def tweak3(m, recs):
        recs[7]["lineup"] = recs[7]["lineup"][::-1]
    an.main(["--test", f"M1={write('M1', 40, tweak=tweak)}", "--test", f"M3={write('M3', 100, tweak=tweak3)}"])
    text = capsys.readouterr().out
    assert "NOT THE REGISTERED DATA" in text and "opponent_params" in text and "game 5: opponent" in text
    assert "game 7: lineup" in text and "registered data: NO" in text
