# 1v1 benchmark: result

Run on 2026-09-26, 08:19-08:36 UTC, under the protocol pre-registered in `docs/BENCH_1V1_PROTOCOL.md`
(commit `35ef224`, pushed before the first game).  The games ran from a frozen worktree of that commit.
Its tests and the C++ parity tests passed there before the run.

## Verdict

| test | opponent (catanatron 3.3.0, defaults) | games | wins | win rate | 95 % CI | one-sided p vs 50 % | avg VP (ours / theirs) |
|---|---|---|---|---|---|---|---|
| H1 (primary) | AlphaBetaPlayer | 400 | 302 | **75.5 %** | 71.0-79.6 % | 1.4e-25 | 9.05 / 6.46 |
| H2 (secondary) | ValueFunctionPlayer | 400 | 293 | **73.2 %** | 68.6-77.5 % | 2.1e-21 | 8.98 / 6.55 |

- **H1:** H0 (win rate at most 50 %) is rejected.  Our bot beats Catanatron's AlphaBetaPlayer head to head.
  The registered rejection point was 217 wins.
- **H2:** tested because H1 rejected (fixed-sequence gate), and also rejected.
- **Both tests are the registered data:** metadata, games 0..399 once each, seeds and seats all match.

The comparison statement, word for word as `scripts/analyze_1v1.py` prints it (fixed in the protocol before
any game):

> Against catanatron 3.3.0's AlphaBetaPlayer (defaults) in 1v1 games our bot won 302/400 = 75.5% (95% CI
> 71.0-79.6%; average 9.1 VP vs 6.5). HexMachina's reported 54.1% lies below our whole 95% interval: our rate
> is higher than their reported point estimate. HexMachina (Belle et al., arXiv 2506.04651) is reported at
> 54.1% (8.2 VP) against AlphaBeta, with AlphaBeta itself at 51.0% (7.8 VP) in the same setup; their exact game
> format, catanatron version, AlphaBeta settings and number of games are unconfirmed (the paper could not be
> read from our environment), and their uncertainty is unknown, so this is a side-by-side of two numbers, not
> a test of one against the other.

The protocol set 237 wins as the bar for this wording; we have 302.  As registered, we do not claim to be
better than HexMachina.

## Descriptive figures

| | H1 (AlphaBeta) | H2 (ValueFunction) |
|---|---|---|
| wins moving first (seat 0) | 154 / 200 | 152 / 200 |
| wins moving second (seat 1) | 148 / 200 | 141 / 200 |
| turns per game | 69.4 | 70.8 |
| turn-cap games | 0 | 0 |
| crashed attempts / games | 0 / 0 | 0 / 0 |
| adapter errors, fallbacks, unmapped actions | 0, 0, 0 | 0, 0, 0 |
| slowest opponent decision | 0.73 s | 0.07 s |

AlphaBeta's 20 s deadline never came close to binding (its slowest decision took 0.73 s).  CPU load from other
jobs could not have changed a game.

## Reading it

- **Two bars, both cleared.**  A coin flip is 50 %; HexMachina's reported 54.1 % sits below our lower bound
  of 71.0 %.
- **This is not the 4-player result.**  In the strength proof (1v3 against three AlphaBetas, test T2) our bot
  won 54.5 % where chance is 25 %.  The formats differ, so the two rates must not be compared.
- **1v1 is outside our bot's tuning.**  It was tuned for 3-4 players, and its politics and coalition logic
  have nothing to act on with one opponent.  Catanatron's AlphaBeta, on the other hand, is built for exactly
  one enemy.  Despite both, the head-to-head rate is high.
- **Remaining caveats** (all in the protocol):
  - The rules are Catanatron's base rules with two seats, not an official two-player variant.
  - Both sides see the full state, as in proof test T2.
  - Player trading is off, and Catanatron's bots cannot trade anyway.
  - HexMachina's game format, Catanatron version, AlphaBeta settings and number of games are unconfirmed.

## Evidence

Everything is in `bench_1v1/`: per-game results, compressed replay logs of all 800 games, console output,
replay checks, the run log and the analysis, with `MANIFEST.sha256` (43 files).

- `replay --check`: 400 + 400 games, 0 mismatches (2.1 KiB per game).
- The analysis re-run on the archived copy is identical to the run's own.
- Checksums verify.

```
(cd bench_1v1 && sha256sum -c MANIFEST.sha256)
/home/user/venv_cat33/bin/python scripts/replay_catanatron.py bench_1v1/H1/logs --check     # likewise H2
python3 scripts/analyze_1v1.py --test H1=bench_1v1/H1/results --test H2=bench_1v1/H2/results
```

To watch any of these games, run `scripts/game_viewer.py bench_1v1/H1/logs --game N --out game.html`.
