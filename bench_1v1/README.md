# 1v1 benchmark evidence

Everything the pre-registered 1v1 benchmark (docs/BENCH_1V1_PROTOCOL.md) produced: catanbot against
catanatron 3.3.0's AlphaBetaPlayer (H1, primary) and ValueFunctionPlayer (H2, secondary), head to head.
Copied from the run's scratch directory by `scripts/run_1v1.sh --archive` (which uses
`scripts/archive_proof.py`: every test must cover exactly games 0..N-1 once, in results and logs).

```
bench_1v1/
  <ID>/results/<ID>_g<a>-<b>.json   bench results of games [a, b) (run metadata, per-game results, timing)
  <ID>/logs/<ID>_<opponent>_1v1_seed<base>_g<a>-<b>.jsonl.gz
                                    replayable action logs of the same games, one game per line, sorted
  <ID>/console/<ID>_g<a>-<b>.txt    the bench's console output of the chunk
  <ID>/replay_check_<ID>.txt        replay --check of the original logs
  run/run_1v1.log                   what ran when
  run/result_1v1.txt, RESULT_1V1.md, result_1v1.json   the analysis (scripts/analyze_1v1.py)
  MANIFEST.sha256                   sha256 of every file here: (cd bench_1v1 && sha256sum -c MANIFEST.sha256)
```

Re-check from the repository root (`$PY33` = a Python with catanatron 3.3.0):

```
(cd bench_1v1 && sha256sum -c MANIFEST.sha256)
$PY33 scripts/replay_catanatron.py bench_1v1/H1/logs --check      # likewise H2
python3 scripts/analyze_1v1.py --test H1=bench_1v1/H1/results --test H2=bench_1v1/H2/results
$PY33 scripts/replay_catanatron.py bench_1v1/H1/logs --game 17 --turn 40   # any position of any game
```
