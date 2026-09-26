# Multi-seat benchmark protocol: 3v1 and 2v2-mixed against Catanatron (pre-registered 2026-09-26, before any benchmark game)

This file fixes the hypotheses, opponents, formats, sample sizes, seeds,
statistics and commands **before** any benchmark game is played. It is
registered by the commit that adds it; the games start only after that
commit. Any later change must be a dated amendment below the original text,
and games played before an amendment keep the original rules.

## Why this benchmark

The strength proof (`docs/PROOF_PROTOCOL.md`, `docs/PROOF.md`) seated our
bot alone against three Catanatron bots (1v3), two against two (2v2), and
alone against a mixed table of three different Catanatron bots (1v3-mixed).
The 1v1 benchmark (`docs/BENCH_1V1_PROTOCOL.md`) adds head-to-head games.
Two seatings are still missing:

- **3v1:** one Catanatron bot alone against three copies of our bot. Does a
  lone AlphaBeta or ValueFunction player still win its fair quarter of the
  games when every other seat is our bot?
- **2v2-mixed:** two copies of our bot against one ValueFunction and one
  AlphaBeta player. Do our two seats together win more than half the games
  when the other two seats are two different Catanatron bots?

## Bot under test

- **Spec:** `search:depth=1,beam=4,expand=8,evaluator=heuristic`, our
  default bot, the same as in the proof and the 1v1 benchmark.
- **Code:** the commit that registers this file. Commits after it may change
  tooling only, not strategy. If strategy code changes in the repository
  before the run ends, the games run from a frozen worktree of the
  registration commit, with its own C++ build, as the proof's games did.
- **Copies:** every catanbot seat is its own bot instance with its own
  state and bot seed (seat k of a game, in turn order, gets bot seed
  `game seed + 7919 k`), exactly as in the proof's 2v2 games.
- **Information:** full, as in proof T2 and the 1v1 benchmark. Every seat
  of ours is handed Catanatron's true state, with every hand and every
  development card known. This is the bench default `--info full`.
  Catanatron's bots read the full state too.
- **Trading:** off (`--trades off`). Our bots never offer a domestic trade
  and decline every offer, also to and from each other. Catanatron's players
  never offer one. Bank and port trades are allowed for every seat.

## Opponents

| id | engine | player(s) | format |
|---|---|---|---|
| M1 | catanatron 3.3.0 | `AlphaBetaPlayer`, defaults | 3v1 |
| M2 | catanatron 3.3.0 | `ValueFunctionPlayer`, defaults | 3v1 |
| M3 | catanatron 3.3.0 | one `ValueFunctionPlayer` + one `AlphaBetaPlayer`, defaults | 2v2-mixed |

- AlphaBetaPlayer's defaults are: depth 2, value function `base`, no
  pruning, no epsilon, and a 20 s search deadline. ValueFunctionPlayer uses
  the `base` value function. No `--opponent-params` is passed.
- The engine is the one of the proof and the 1v1 benchmark: the GitHub
  checkout `/home/user/bcollazo/catanatron` at commit `ecf9311`, installed
  in `/home/user/venv_cat33` and run with `/home/user/venv_cat33/bin/python`.
  The result files record the engine version of every chunk.

## Formats

Both formats are 4-player games under the rules of the proof: Catanatron's
base rules, 10 VP to win, a player with more than 7 cards discards half of
them on a 7, Catanatron's turn cap of 1000 turns, colours RED, BLUE, ORANGE,
WHITE in turn order (seat 0 moves first).

**3v1** (`scripts/bench_catanatron.py --our-seats 3`, added for this
benchmark):

- three catanbot seats and one opponent seat;
- the opponent sits in seat `g % 4` in game `g`, our bot in the other three
  seats, so in 400 games each seat holds the opponent exactly 100 times;
- a game counts for our table when any of our three seats wins it.

**2v2-mixed** (`--our-seats 2 --mixed-opponents value,alphabeta`, added for
this benchmark):

- two catanbot seats, one ValueFunctionPlayer and one AlphaBetaPlayer;
- game `g` uses placement `g % 12` of the 12 placements: our two seats are
  the seat pair `g % 6` of the six pairs (0,1), (0,2), (0,3), (1,2), (1,3),
  (2,3), as in the proof's 2v2, and the two presets sit in the other two
  seats in turn order value, alphabeta when `(g // 6) % 2 == 0`, and
  alphabeta, value otherwise;
- any 12 consecutive games play each placement once; 400 = 33 x 12 + 4, so
  placements 0-3 get 34 games and placements 4-11 get 33;
- a game counts for our table when either of our two seats wins it.

The existing formats (1v3, 2v2, 1v3-mixed, 1v1) are unchanged: digests of 13
before/after runs matched on both engines (last section).

## What this measures, and what it does not

- **Three copies of our bot share one strategy, and they never cooperate or
  trade** (trades off). Each copy plays to win for itself and competes with
  the other copies as hard as with the opponent. M1 and M2 therefore measure
  a lone Catanatron bot against a table of copies of one strategy, **not
  teamwork**. The same holds for the two copies in M3.
- A lone opponent can profit from the copies' competition (for example,
  when they rob or block each other), and it suffers from facing three
  strong seats at once. Neither effect is isolated here.
- Catanatron's value function models a single enemy, the next player in
  turn order. In 3v1 that enemy is always a copy of our bot. Its AlphaBeta
  search assumes that every other seat minimises its value; our copies do
  not play that way.
- These rates must not be compared with the proof's 1v3, 2v2 or 1v3-mixed
  rates, or with the 1v1 rate: every format has its own null and its own
  dynamics.
- **Caveat:** read `docs/BENCHMARKS.md`, section "What Catanatron's bots
  are, and what they are not", before quoting any result. Catanatron's bots
  are an open-source baseline with known gaps (development cards, hidden
  Victory Point cards, one modelled enemy, two-move look-ahead, no trading,
  no table politics). Winning against them is not evidence of strength
  against good human players.

## Sample sizes and seeds

| test | format | opponent(s) | games | base seed | information |
|---|---|---|---|---|---|
| M1 | 3v1 | AlphaBetaPlayer (catanatron 3.3.0, defaults) | 400 | 910701 | full |
| M2 | 3v1 | ValueFunctionPlayer | 400 | 910801 | full |
| M3 | 2v2-mixed | ValueFunction + AlphaBeta | 400 | 910901 | full |

- **Seeds used before:** none of these base seeds has been used. The proof
  used 900001-900401, the 1v1 benchmark 910501 and 910601, its smoke runs
  424701 and 424801. Tooling smoke runs and tests of this benchmark use
  424901, 425001 and 425101 only.
- **Per-game seed:** game `g` has Catanatron seed `base * 100003 + g + 1`.
  `PYTHONHASHSEED` is pinned to 0 (`--hash-seed 0`).
- **Turn cap and crashes:** a crashed game is re-played once with the same
  seed (`--rerun-crashes`); a second crash is recorded and reported. How
  turn-cap and crashed games count is fixed under Statistics.

## Statistics

**Counting.** Every game's outcome is recomputed from its winner seat and
the registered seating, not read from the bench's `won` flags.

**M1 and M2 (3v1).** Let `p` be the probability that a game is not won by
one of our three seats.

- Hypotheses: H0 "the opponent wins at least 25 %" (`p >= 0.25`) against H1
  "less than 25 %" (`p < 0.25`).
- Tested count: the games not won by our seats, that is, the opponent's
  wins **plus** turn-cap games **plus** games that crashed twice. Counting
  the games without a winner for the opponent is conservative: the test
  stays a valid level-alpha test of "the opponent wins at least 25 %". The
  opponent's own wins and the games without a winner are also reported
  separately.
- Test: exact one-sided binomial test, lower tail. The p-value is
  `P(X <= count)` under Bin(400, 1/4).

**M3 (2v2-mixed).** Let `q` be the probability that one of our two seats
wins a game.

- Hypotheses: H0 "our two seats together win at most 50 %" (`q <= 0.5`)
  against H1 `q > 0.5`.
- Tested count: games won by either of our seats. Turn-cap games and games
  that crashed twice are losses.
- Test: exact one-sided binomial test, upper tail. The p-value is
  `P(X >= wins)` under Bin(400, 1/2).

**Family.** Holm-Bonferroni over M1-M3 at alpha = 0.05 (family-wise
error 0.05). Sorted by p-value, the smallest is compared with 0.05/3, the
next with 0.05/2 and the largest with 0.05, stopping at the first
non-rejection. The family always has m = 3: a test that is missing or is not
the registered data enters with p = 1, so it is never rejected and does not
loosen the thresholds of the others.

**Rejection points** (exact, n = 400). A test with p <= 0.05/3 is rejected
whatever the other two show; the looser points apply only when the tests
before it in the Holm order were rejected.

| test | at 0.05/3 (always enough) | at 0.05/2 | at 0.05 |
|---|---|---|---|
| M1, M2: games not won by our seats | <= 81 (20.25 %), exact size 0.0148 | <= 82, size 0.0200 | <= 85 (21.25 %), size 0.0452 |
| M1, M2: equivalently, our seats' wins | >= 319 | >= 318 | >= 315 |
| M3: games won by our seats | >= 222 (55.5 %), size 0.0157 | >= 221, size 0.0201 | >= 217 (54.25 %), size 0.0494 |

**Power** (exact binomial, n = 400). The first number is the power at
0.05/3, a lower bound on the Holm power; the second is the power at 0.05,
reached when the other two tests reject first.

| M1 / M2: true rate of games not won by our seats | 22.5 % | 20 % | 17.5 % | 15 % | 12.5 % or less |
|---|---|---|---|---|---|
| power | 0.15 / 0.30 | 0.58 / 0.76 | 0.93 / 0.98 | 0.998 / 0.9997 | > 0.99999 |

| M3: true rate of our seats' wins | 52.5 % | 55 % | 57.5 % | 60 % | 65 % or more |
|---|---|---|---|---|---|
| power | 0.12 / 0.26 | 0.44 / 0.64 | 0.81 / 0.91 | 0.970 / 0.992 | > 0.9999 |

For orientation only: in the proof our two seats won 80.8 % of 2v2 games
against two AlphaBeta players (T5) and 83.6 % against two ValueFunction
players (T4). These are other formats and are not a prediction.

**Intervals.** Central two-sided Clopper-Pearson intervals, 95 %, for the
tested proportion of each test, and for our seats' win rate. At n = 400 a
95 % interval is about ±3-5 points wide (for example 10.0 % -> 7.2-13.4 %,
80.0 % -> 75.7-83.8 %).

**Descriptive figures,** reported for every test, not tested:

- per-seat rates: in M1 and M2 the opponent's wins in each seat (100 games
  each) and each catanbot seat's wins; in M3 the wins of each player per
  seat, per turn-order pattern of our seats, and wins per preset
  (ValueFunction and AlphaBeta separately);
- the average VP per seat of ours and of each opponent;
- turns per game;
- turn-cap games;
- crashed attempts and games;
- adapter errors, illegal-action fallbacks and unmapped top actions.

The statistics code is `scripts/analyze_multi.py`, written before any game.
It computes the exact values with the functions of
`scripts/prove_strength.py` (rational binomial tails, Clopper-Pearson by
bisection, Holm), which are tested against scipy there; its own tests check
its p-values and intervals against scipy's.

**Registered data.** The analysis accepts only data that matches every
registered field. Anything else is listed as a deviation, and that test's
verdict then says "not the registered data" (and it enters the Holm family
with p = 1).

Run metadata that must match:

- the format (`3v1`, or `2v2-mixed` with presets `value`, `alphabeta` in
  that order), 4 players, 3 or 2 catanbot seats and the format's null;
- the opponent, as preset and as class;
- default opponent parameters;
- engine 3.3.0;
- the spec above;
- the base seed;
- trades off;
- hash seed 0;
- 10 VP;
- discard limit 7;
- information mode full.

Per-game checks:

- exactly games 0..399, each once;
- each game's Catanatron seed;
- our seats: every seat but `g % 4` (M1, M2), the pair `g % 6` (M3);
- in M1 and M2 the opponent's preset and seat `g % 4`; in M3 the whole
  lineup of placement `g % 12`;
- four seats, RED, BLUE, ORANGE, WHITE.

## Running (registered commands)

Nothing below has been run with the registered seeds. The runner is
`scripts/run_multi.sh`, built like `scripts/run_1v1.sh`:

- it plays chunked games;
- it can be resumed after an interruption, because a chunk whose JSON
  exists is skipped;
- a chunk that times out is split in two, down to single games;
- a chunk writes its JSON and log under temporary names, then moves them
  into place.

    nice -n 10 scripts/run_multi.sh            # M1, M2, M3; then replay --check of every log and the analysis
    nice -n 10 scripts/run_multi.sh M1         # one test (resumes where it stopped)
    scripts/run_multi.sh --analyze             # replay checks and analysis of what exists
    scripts/run_multi.sh --archive             # the evidence into bench_multi/ (next section)

Each chunk of games `[A, B)` runs exactly one of these commands, under
`timeout 1200`:

    # M1
    /home/user/venv_cat33/bin/python scripts/bench_catanatron.py --our-seats 3 --opponent alphabeta \
        --spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --seed 910701 --game-range A:B \
        --workers W --trades off --hash-seed 0 --rerun-crashes --log-actions DIR --verbose --json FILE
    # M2
    /home/user/venv_cat33/bin/python scripts/bench_catanatron.py --our-seats 3 --opponent value \
        --spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --seed 910801 --game-range A:B \
        --workers W --trades off --hash-seed 0 --rerun-crashes --log-actions DIR --verbose --json FILE
    # M3
    /home/user/venv_cat33/bin/python scripts/bench_catanatron.py --our-seats 2 --mixed-opponents value,alphabeta \
        --spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --seed 910901 --game-range A:B \
        --workers W --trades off --hash-seed 0 --rerun-crashes --log-actions DIR --verbose --json FILE

- **Chunk sizes:** 50 games for M1, 100 for M2, 40 for M3 (`BENCHMULTI_CHUNK_<ID>`
  overrides them; chunks cannot change a game).
- **Workers:** `BENCHMULTI_WORKERS`, default 1, because the machine is
  shared. The worker count cannot change a game: per-game seeds and
  seatings depend on the game index alone. The only exception is
  AlphaBeta's 20 s deadline, which no decision came near in the smoke
  (slowest 3.0 s).
- **Duration at 1 worker:** in the runner smoke (4 games per test, on a
  machine whose other 3 cores were busy) a game took about 5 s in M1, 2 s
  in M2 and 7 s in M3. That puts M1 at about 35 minutes, M2 at about 15
  minutes and M3 at about 50 minutes, roughly 1 hour 40 minutes in all.
  With 4 games per test the estimate is rough; allow up to twice as long.
- **Output:** everything goes to `BENCHMULTI_OUT`, by default the session
  scratchpad `.../scratchpad/benchmulti/run`:
  - `json/<ID>/g<a>-<b>.json`
  - `logs/<ID>/*.jsonl.gz`
  - `out/<ID>_g<a>-<b>.txt`
  - `replay_check_<ID>.txt`
  - `run_multi.log`
  - `result_multi.txt`, `RESULT_MULTI.md`, `result_multi.json`
- **Analysis:** `python3 scripts/analyze_multi.py --test M1=OUT/json/M1
  --test M2=OUT/json/M2 --test M3=OUT/json/M3 --markdown RESULT_MULTI.md`
- **Smoke runs:** `BENCHMULTI_SMOKE=1` plays 4 games per test in chunks of
  2 with the smoke seeds 424901, 425001 and 425101. Its `--archive` goes to
  the smoke directory, never into the repository.

## Evidence

Every game's evidence will be committed to the repository under
`bench_multi/`, the way `proof/` and `bench_1v1/` keep theirs. The step is
`scripts/run_multi.sh --archive`, run after the analysis. It copies the run
with the unchanged `scripts/archive_proof.py`, which checks that each test
covers exactly games 0..399 once, in both the results and the logs, and
writes a manifest.

    bench_multi/
      M1/results/M1_g<a>-<b>.json      per-game results + run metadata (unchanged bench JSON)
      M1/logs/M1_alphabeta_3v1_seed910701_g<a>-<b>.jsonl.gz
                                       compressed replay logs, one game per line, sorted by game
      M1/console/M1_g<a>-<b>.txt       console output of each chunk
      M1/replay_check_M1.txt           replay --check of every M1 game
      M2/...                           the same for M2 (value_3v1_seed910801)
      M3/...                           the same for M3 (value_alphabeta_2v2-mixed_seed910901)
      run/run_multi.log                what ran when
      run/result_multi.txt, RESULT_MULTI.md, result_multi.json
                                       the analysis
      README.md                        layout and re-check commands
      MANIFEST.sha256                  sha256 of every file above

**A result file** holds:

- the run metadata that the analysis checks;
- per game: game index, seed, our seats and pattern, the winner seat,
  whether one of ours won, the opponent's seat, preset and whether it won
  (3v1) or the lineup and the winner's name (2v2-mixed), VPs, turns,
  crashes, and adapter counters per catanbot seat;
- per-decision timing for every side.

**A log line** is one game:

- base seed and game index;
- the players in turn order, with the opponents' classes and each catanbot
  seat's bot seed;
- the board, the robber start and the development-deck order;
- every action with its chance outcome;
- the final state with a full-state fingerprint.

The logs are about 3 KiB per game, so the whole archive is a few MiB.

Re-check from the repository root:

    (cd bench_multi && sha256sum -c MANIFEST.sha256)
    /home/user/venv_cat33/bin/python scripts/replay_catanatron.py bench_multi/M1/logs --check   # likewise M2, M3
    python3 scripts/analyze_multi.py --test M1=bench_multi/M1/results --test M2=bench_multi/M2/results \
        --test M3=bench_multi/M3/results

The result, meaning the three verdicts, the Holm family line and the
descriptive table, goes to a results page in `docs/` with the caveat above.
That page records the commit the games were run from.

## Tooling checks done before registration (no registered seed played)

- **Format smokes on catanatron 3.2.1** (system `python3`): 2 games of 3v1
  against `weighted` (seed 424901) and 2 games of 2v2-mixed against
  `weighted` + our `vf` stand-in (seed 425101). All logged games passed
  `scripts/replay_catanatron.py --check`.
- **Runner smoke on catanatron 3.3.0** (the venv):
  `BENCHMULTI_SMOKE=1 scripts/run_multi.sh`, 4 games per test against the
  registered opponents with the smoke seeds. All 12 games replayed with 0
  mismatches, with 0 adapter errors, fallbacks, crashes or turn-cap games.
  Then `--archive` into the smoke directory and `sha256sum -c` of its
  manifest (26 files OK), a replay `--check` of the archived logs, and the
  analysis, which reports "not the registered data" for every test, as it
  must for smoke seeds.
- **Digests of the existing formats:** 13 before/after runs matched, 6 on
  catanatron 3.2.1 (1v3, logged 1v3, logged 2v2, 1v3-mixed counted, logged
  1v1, a ladder) and 7 on 3.3.0 (the same plus a trades run). The action-log
  bytes, the result JSON without timings (key order included) and the
  masked console output were identical. Both sides ran against one frozen
  snapshot of `catanbot/`, so only the bench script differed.
- **Tests:** `tests/test_bench_script.py` (with 9 new tests of 3v1,
  2v2-mixed and `analyze_multi.py`), `tests/test_catanatron_adapter.py`,
  `tests/test_prove_strength.py` and `tests/test_public_info.py`: 141 passed
  and 19 skipped on catanatron 3.2.1 (system `python3`; the skips need the
  3.3 engine), 158 passed and 2 skipped on 3.3.0 (the venv; the skips need
  the 3.2.1 engine).
