# Multi-seat benchmark evidence

Everything the pre-registered multi-seat benchmark (docs/BENCH_MULTI_PROTOCOL.md) produced: three catanbot seats
against one catanatron 3.3.0 AlphaBetaPlayer (M1) or ValueFunctionPlayer (M2) in 3v1 games, and two catanbot seats
against one ValueFunctionPlayer and one AlphaBetaPlayer in 2v2-mixed games (M3).  Copied from the run's scratch
directory by `scripts/run_multi.sh --archive` (which uses `scripts/archive_proof.py`: every test must cover exactly
games 0..N-1 once, in results and logs).

```
bench_multi/
  <ID>/results/<ID>_g<a>-<b>.json   bench results of games [a, b) (run metadata, per-game results, timing)
  <ID>/logs/<ID>_<opponent>_<format>_seed<base>_g<a>-<b>.jsonl.gz
                                    replayable action logs of the same games, one game per line, sorted
  <ID>/console/<ID>_g<a>-<b>.txt    the bench's console output of the chunk
  <ID>/replay_check_<ID>.txt        replay --check of the original logs
  run/run_multi.log                 what ran when
  run/result_multi.txt, RESULT_MULTI.md, result_multi.json   the analysis (scripts/analyze_multi.py)
  MANIFEST.sha256                   sha256 of every file here: (cd bench_multi && sha256sum -c MANIFEST.sha256)
```

Re-check from the repository root (`$PY33` = a Python with catanatron 3.3.0):

```
(cd bench_multi && sha256sum -c MANIFEST.sha256)
$PY33 scripts/replay_catanatron.py bench_multi/M1/logs --check      # likewise M2, M3
python3 scripts/analyze_multi.py --test M1=bench_multi/M1/results --test M2=bench_multi/M2/results \
    --test M3=bench_multi/M3/results
$PY33 scripts/replay_catanatron.py bench_multi/M1/logs --game 17 --turn 40   # any position of any game
```
