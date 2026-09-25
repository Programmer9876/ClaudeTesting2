import importlib.util
import sys

import pytest

spec = importlib.util.spec_from_file_location("bench_catanatron", "scripts/bench_catanatron.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_resolve_opponent_presets_and_paths():
    assert bench.resolve_opponent("vp").__name__ == "VictoryPointPlayer"
    assert bench.resolve_opponent("catanatron.players.weighted_random:WeightedRandomPlayer").__name__ == "WeightedRandomPlayer"
    assert bench.resolve_opponent("catanatron.players.weighted_random.WeightedRandomPlayer").__name__ == "WeightedRandomPlayer"
    with pytest.raises(SystemExit):
        bench.resolve_opponent("nonsense")
    with pytest.raises(SystemExit):
        bench.resolve_opponent("no.such.module:Player")


def test_ladder_skips_missing_and_runs_stock(capsys):
    rc = bench.main(["--games", "2", "--opponent", "weighted,alphabeta-missing", "--spec", "heuristic", "--seed", "3"])
    out = capsys.readouterr().out
    assert rc == 0 and "WeightedRandomPlayer" in out and "skipping" in out
