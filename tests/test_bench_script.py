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
