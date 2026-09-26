# Multi-seat benchmark: result

*Run 2026-09-26, 10:33-12:25 UTC, under the protocol pre-registered in `docs/BENCH_MULTI_PROTOCOL.md` (commit
`332775b`, pushed before the first game).  The games ran from a frozen worktree of that commit.  There, default
play was first checked to be identical to the queue's epoch-B1 code (24 of 24 games) and the tests passed.
Opponents: Catanatron 3.3.0 at GitHub commit `ecf9311`.*

## Verdict

All three registered hypotheses are rejected (Holm over M1-M3, alpha 0.05).

| test | seating | result | 95 % CI | null | one-sided p |
|---|---|---|---|---|---|
| M1 | 3 of ours vs 1 AlphaBeta | the lone AlphaBeta won **39/400 = 9.8 %** | 7.0-13.1 % | 25 % | 8.4e-15 |
| M2 | 3 of ours vs 1 ValueFunction | the lone ValueFunction won **21/400 = 5.2 %** | 3.3-7.9 % | 25 % | 6.1e-26 |
| M3 | 2 of ours vs 1 ValueFunction + 1 AlphaBeta | our two seats won **314/400 = 78.5 %** | 74.1-82.4 % | 50 % | 6.8e-32 |

- **Registered data:** every test matches the protocol (seeds, seats, opponents, settings, games 0..399 once).
- **Clean run:** no games hit the turn cap, none crashed, and there were no adapter errors or fallbacks.
- **Replay:** `replay --check` found 0 mismatches in all 1,200 games.
- **Per-seat figures** (the lone opponent wins more from later seats, as a moving-last advantage would give):

  | test | opponent wins by seat 0 / 1 / 2 / 3 (of 100 each) |
  |---|---|
  | M1 AlphaBeta | 9 / 4 / 12 / 14 |
  | M2 ValueFunction | 7 / 4 / 3 / 7 |

- **M3 by opponent:** AlphaBeta won 56 games (14.0 %) and ValueFunction 30 (7.5 %).

## Reading it

- **Three copies of our bot share one strategy and never cooperate or trade** (trading off).  M1 and M2 measure
  one Catanatron bot against a table of our copies, not teamwork.
- **Catanatron's bots have known gaps**: development cards nearly worthless to them, Victory Point cards
  invisible, one modelled enemy, two moves of lookahead, no trading, no politics (docs/BENCHMARKS.md, "What
  Catanatron's bots are, and what they are not").  These results complete the picture against Catanatron.
  They say nothing about human opponents.
- **The formats cannot be compared with each other's rates.**  1v3, 2v2, 3v1 and 1v1 have different nulls
  (25 %, 50 %, 75 % for our seats, 50 %).

## Evidence

`bench_multi/`: per-game results, compressed replay logs of all 1,200 games, console output, replay checks,
the run log and the analysis, with `MANIFEST.sha256` (74 files).

```
(cd bench_multi && sha256sum -c MANIFEST.sha256)
/home/user/venv_cat33/bin/python scripts/replay_catanatron.py bench_multi/M1/logs --check    # likewise M2, M3
python3 scripts/analyze_multi.py --test M1=bench_multi/M1/results --test M2=bench_multi/M2/results --test M3=bench_multi/M3/results
```
