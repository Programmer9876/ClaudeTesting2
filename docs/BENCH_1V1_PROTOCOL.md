# 1v1 benchmark protocol: catanbot vs Catanatron's AlphaBetaPlayer (pre-registered 2026-09-26, before any benchmark game)

This file fixes the hypotheses, opponents, format, sample sizes, seeds,
statistics and the comparison statement **before** any benchmark game is
played. It is registered by the commit that adds it; the games start only
after that commit. Any later change must be a dated amendment below the
original text, and games played before an amendment keep the original rules.

## Why this benchmark

We want a number that can stand next to HexMachina's. HexMachina is Belle et
al., "Agents of Change: Self-Evolving LLM Agents for Strategic Planning"
(arXiv 2506.04651). On OpenReview it is "HexMachina: Self-Evolving
Multi-Agent System for Continual Learning of Catan" (id V0Fb4pwhS4).

What search summaries report:

- HexMachina wins **54.1 %** of its games (8.2 VP on average) against
  Catanatron's AlphaBeta player;
- AlphaBeta itself scores 51.0 % (7.8 VP) in the same setup.

These numbers read as head-to-head games, where parity is about 50 %.

**The exact format of their experiment is unconfirmed.** The paper cannot be
fetched from our environment, because the network blocks arxiv.org and
openreview.net. So we do not know any of these:

- the number of players per game;
- the number of games;
- the Catanatron version;
- AlphaBeta's settings;
- the seating;
- the VP target;
- whether trading was possible.

Our strength proof (`docs/PROOF_PROTOCOL.md`, `docs/PROOF.md`) played only
4-player games (1v3 and 2v2). Its numbers cannot be compared with a
head-to-head figure. This benchmark adds the 1v1 format.

## Bot under test

- **Spec:** `search:depth=1,beam=4,expand=8,evaluator=heuristic`, our
  default bot, the same as in the proof.
- **Code:** the commit that registers this file. The tooling commits that
  follow it may not change strategy.
- **Information:** full, as in proof test T2. Our bot is handed Catanatron's
  true state, with every hand and every development card known. This is the
  bench default `--info full`. Catanatron's bots read the full state too.
- **Trading:** off (`--trades off`). Our bot never offers a domestic trade
  and declines every offer. Catanatron's players never offer one, and its
  AlphaBetaPlayer cannot evaluate trade actions. Bank and port trades are
  allowed for both sides.

## Opponents

| id | engine | player | role |
|---|---|---|---|
| H1 | catanatron 3.3.0 | `AlphaBetaPlayer`, defaults | **primary** |
| H2 | catanatron 3.3.0 | `ValueFunctionPlayer`, defaults | secondary |

AlphaBetaPlayer's defaults are: depth 2, value function `base`, no pruning,
no epsilon, and a 20 s search deadline. No `--opponent-params` is passed to
either player.

The engine is the GitHub checkout `/home/user/bcollazo/catanatron` at commit
`ecf9311` (branch main). It is installed in `/home/user/venv_cat33` and run
with `/home/user/venv_cat33/bin/python`.

The PyPI wheel (3.2.1) does not ship AlphaBetaPlayer. Our stand-ins `vf` and
`ab` are not Catanatron's players and are not used here.

## Format: 1v1

- Two players: our bot against **one** copy of the opponent, colours RED and
  BLUE.
- Our bot sits in seat `g % 2` in game `g`. Seat 0 moves first, so each seat
  gets exactly 200 of the 400 games.
- The rules are Catanatron's base rules, seated with two players:
  - the full 19-hex board, from Catanatron 3.3's default board generation
    (random tiles and ports, number tokens along the official spiral);
  - 10 VP to win;
  - a player with more than 7 cards discards half of them on a 7;
  - the robber, moved on a 7 or a knight;
  - setup placement order 0-1-1-0;
  - Catanatron's turn cap of 1000 turns.
- No two-player variant rules are used (no trade tokens, no neutral
  players).
- Tooling: `scripts/bench_catanatron.py --players 2`, added for this
  benchmark. The default `--players 4` keeps every 4-player format
  byte-identical: digests of 9 before/after runs matched on both engines
  (next section).

## Sample sizes and seeds

| test | opponent | games | base seed |
|---|---|---|---|
| H1 | AlphaBetaPlayer | 400 | 910501 |
| H2 | ValueFunctionPlayer | 400 | 910601 |

- **Seeds used before:** neither base seed has been used. The proof used
  900001-900401. Tooling smoke runs of this benchmark use 424701 and 424801.
- **Per-game seed:** game `g` has Catanatron seed `base * 100003 + g + 1`.
  `PYTHONHASHSEED` is pinned to 0 (`--hash-seed 0`).
- **Turn cap:** a game that hits the turn cap counts as a loss for our bot.
- **Crashes:** a crashed game is re-played once with the same seed
  (`--rerun-crashes`). A second crash counts as a loss and is reported.

## Statistics

**Primary endpoint (H1).** Our win rate `p` against AlphaBetaPlayer.

- Hypotheses: H0 `p <= 0.5` against H1 `p > 0.5`.
- Test: exact one-sided binomial test. The p-value is `P(X >= wins)` under
  Bin(400, 1/2).
- Level: alpha = 0.05.
- Counting wins: a game's win is recomputed from its winner seat. Turn-cap
  games and games that crashed twice are losses.
- Rejection point: H0 is rejected from 217 wins (54.25 %) upwards.
- Power: 0.64 if the true rate is 55 %, 0.91 at 57.5 %, 0.99 at 60 %.

**Secondary endpoint (H2).** The same test against ValueFunctionPlayer,
gated by the primary (fixed-sequence testing). H2 is tested at alpha 0.05
only if H1 rejects; otherwise it is reported descriptively. This keeps the
family-wise error at 0.05.

**Intervals.** Central two-sided Clopper-Pearson intervals, 95 %, with 99 %
reported too. At n = 400 the 95 % interval is about ±4.9 points wide.

**Descriptive figures,** reported for both tests:

- the average VP of both sides;
- the win rate per seat (seat 0 moves first);
- turns per game;
- turn-cap games;
- crashed attempts and games;
- adapter errors, illegal-action fallbacks and unmapped top actions.

The statistics code is `scripts/analyze_1v1.py`, written before any game.
It computes the exact values with the functions of
`scripts/prove_strength.py`, which are tested against scipy there.

**Registered data.** The analysis accepts only data that matches every
registered field. Anything else is listed as a deviation, and the verdict
then says "not the registered data".

Run metadata that must match:

- format `1v1` and 2 players;
- opponent, as preset and as class;
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
- our bot in seat `g % 2`;
- two seats, RED and BLUE.

## The comparison statement we will make

We will report H1's rate and its 95 % interval next to HexMachina's reported
54.1 %, with the caveat that their format is unconfirmed. This is a
side-by-side of two numbers, not a test. HexMachina's sample size and
uncertainty are unknown, and their game format, Catanatron version and
AlphaBeta settings are unconfirmed.

`scripts/analyze_1v1.py` prints the statement, fixed now:

> Against catanatron 3.3.0's AlphaBetaPlayer (defaults) in 1v1 games our bot
> won W/400 = x% (95% CI a-b%; average y VP vs z). HexMachina's reported
> 54.1% lies {below our whole 95% interval: our rate is higher than their
> reported point estimate | inside our 95% interval: at this sample size our
> rate is not distinguishable from their reported point estimate | above our
> whole 95% interval: our rate is lower than their reported point estimate}.
> HexMachina (Belle et al., arXiv 2506.04651) is reported at 54.1% (8.2 VP)
> against AlphaBeta, with AlphaBeta itself at 51.0% (7.8 VP) in the same
> setup; their exact game format, catanatron version, AlphaBeta settings and
> number of games are unconfirmed (the paper could not be read from our
> environment), and their uncertainty is unknown, so this is a side-by-side
> of two numbers, not a test of one against the other.

The first branch ("our rate is higher") needs at least 237 wins (59.25 %).

We will not write "better than HexMachina" in any form. At most we will
write that our rate is higher, equal within uncertainty, or lower than
their *reported* figure, always with the caveat.

## What may limit comparability: two-player rules and engine

1. **The rules are Catanatron's base rules with two seats, not an official
   two-player variant.**
   - The full board leaves much more room than a 4-player game: less
     blocking and less competition for spots, ports and Longest Road.
   - Every 7 and every knight can only rob the one opponent.
   - The discard limit stays at 7.
2. **Turn order.**
   - Seat 0 places first and moves first. Seat 1 places its two setup
     settlements back to back (order 0-1-1-0).
   - Alternating seats balances this exactly, and we report per-seat rates.
   - Catanatron's own CLI shuffles seats at random. HexMachina's seating is
     unknown.
3. **Catanatron's bots are built for head-to-head play.**
   - Their value function (`base_fn`) models a single enemy: the next
     player in turn order. The source says "the base value function only
     considers 1 enemy player".
   - AlphaBeta is a max/min search in which every other seat minimises its
     value.
   - Both assumptions are exact in 1v1 and only approximations in 4-player
     games. AlphaBeta may therefore be relatively stronger here than in the
     proof's 1v3 and 2v2 games.
   - Our bot was tuned in 3-4 player self-play. Its political module finds
     no award-threat opportunities below 3 players, and its coalition logic
     has nothing to act on with one opponent. 1v1 is outside the setting it
     was tuned in.
   - For both reasons, the 1v1 rate must not be compared with the proof's
     1v3 and 2v2 rates either.
4. **Discards.** Catanatron 3.3 lets every player choose its discard, one
   card per prompt. Our bot plans the whole discard and hands it over card
   by card, as in the proof. (The 3.2.1 engine, not used here, discards at
   random.)
5. **Information.** Both sides see the full state, as in proof T2. In 1v1 a
   robber steal always involves both players, so Colonist-style hidden
   information would reduce to the opponent's unrevealed development cards.
   A tooling smoke in the counted mode (`--info counted`) saw 0 hidden
   steals, as expected. The information question matters less here than in
   4-player games.
6. **Engine version and settings.**
   - HexMachina's Catanatron version is unconfirmed. Older releases kept
     AlphaBetaPlayer in `catanatron_experimental` and placed the number
     tokens at random.
   - Their AlphaBeta settings (depth, pruning, value function) are also
     unconfirmed.
   - Catanatron 3.3's AlphaBeta stops a search at 20 s of wall time. Its 1v1
     decisions took 18 ms on average (141 ms at the 95th percentile) in the
     smoke, so the deadline should never bind. The logs record what was
     actually played.
7. **Engine rule differences** listed in `docs/BENCHMARKS.md` apply
   unchanged. For example, Catanatron never counts a road that runs through
   an enemy building towards Longest Road. Victory points are always
   Catanatron's.

## Running (registered commands)

Nothing below has been run with the registered seeds. The runner is
`scripts/run_1v1.sh`, built like `scripts/run_proof.sh`:

- it plays chunked games;
- it can be resumed after an interruption, because a chunk whose JSON
  exists is skipped;
- a chunk that times out is split in two, down to single games;
- a chunk writes its JSON and log under temporary names, then moves them
  into place.

    nice -n 10 scripts/run_1v1.sh              # H1, then H2; then replay --check of every log and the analysis
    nice -n 10 scripts/run_1v1.sh H1           # one test (resumes where it stopped)
    scripts/run_1v1.sh --analyze               # replay checks and analysis of what exists
    scripts/run_1v1.sh --archive               # the evidence into bench_1v1/ (next section)

Each chunk of games `[A, B)` runs exactly this command, under `timeout 1200`:

    /home/user/venv_cat33/bin/python scripts/bench_catanatron.py --players 2 --opponent alphabeta \
        --spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --seed 910501 --game-range A:B \
        --workers W --trades off --hash-seed 0 --rerun-crashes --log-actions DIR --verbose --json FILE

H2 runs the same command with `--opponent value --seed 910601`.

- **Chunk sizes:** 50 games for H1 and 100 for H2.
- **Workers:** `BENCH1V1_WORKERS`, default 1, because the machine is
  shared. The worker count cannot change a game: per-game seeds and seats
  depend on the game index alone. The only exception is AlphaBeta's 20 s
  deadline.
- **Duration:** in the smoke a 1v1 AlphaBeta game took about 2 s and a value
  game under 1 s. H1 should take about 15 minutes and H2 about 5 minutes at
  1 worker.
- **Output:** everything goes to `BENCH1V1_OUT`, by default the session
  scratchpad `.../scratchpad/bench1v1/run`:
  - `json/<ID>/g<a>-<b>.json`
  - `logs/<ID>/*.jsonl.gz`
  - `out/<ID>_g<a>-<b>.txt`
  - `replay_check_<ID>.txt`
  - `run_1v1.log`
  - `result_1v1.txt`, `RESULT_1V1.md`, `result_1v1.json`
- **Analysis:** `python3 scripts/analyze_1v1.py --test H1=OUT/json/H1 --test
  H2=OUT/json/H2 --markdown RESULT_1V1.md`
- **Smoke runs:** `BENCH1V1_SMOKE=1` plays 4 games per test with the smoke
  seeds 424701 and 424801. Its `--archive` goes to the smoke directory, never
  into the repository.

## Evidence

Every game's evidence will be committed to the repository under
`bench_1v1/`, the way `proof/` keeps the strength proof's evidence. The step
is `scripts/run_1v1.sh --archive`, run after the analysis. It copies the run
with `scripts/archive_proof.py`, which checks that each test covers exactly
games 0..399 once, in both the results and the logs.

    bench_1v1/
      H1/results/H1_g<a>-<b>.json      per-game results + run metadata (unchanged bench JSON)
      H1/logs/H1_alphabeta_1v1_seed910501_g<a>-<b>.jsonl.gz
                                       compressed replay logs, one game per line, sorted by game
      H1/console/H1_g<a>-<b>.txt       console output of each chunk
      H1/replay_check_H1.txt           replay --check of every H1 game
      H2/...                           the same for H2 (value_1v1_seed910601)
      run/run_1v1.log                  what ran when
      run/result_1v1.txt, RESULT_1V1.md, result_1v1.json
                                       the analysis
      README.md                        layout and re-check commands
      MANIFEST.sha256                  sha256 of every file above

**A result file** holds:

- the run metadata that the analysis checks;
- per game: game index, seed, seat, winner, VPs, turns, crashes, and
  adapter counters;
- per-decision timing for both sides.

**A log line** is one game:

- base seed and game index;
- the players with their classes and our bot's seed;
- the board, the robber start and the development-deck order;
- every action with its chance outcome;
- the final state with a full-state fingerprint.

The logs are about 2 KiB per game, so the whole archive is a few MiB.

Re-check from the repository root:

    (cd bench_1v1 && sha256sum -c MANIFEST.sha256)
    /home/user/venv_cat33/bin/python scripts/replay_catanatron.py bench_1v1/H1/logs --check   # likewise H2
    python3 scripts/analyze_1v1.py --test H1=bench_1v1/H1/results --test H2=bench_1v1/H2/results

The result, meaning the primary verdict, the HexMachina statement, the
secondary line and the descriptive table, goes to a results page in `docs/`.
That page records the commit the games were run from.

## Tooling checks done before registration (no registered seed played)

- **2-player smoke on catanatron 3.2.1** (system `python3`): 2 games each
  against our `vf` and `ab` stand-ins, seed 424701. All logged games passed
  `scripts/replay_catanatron.py --check`.
- **2-player smoke on catanatron 3.3.0** (the venv): 2 games each against
  `value` and `alphabeta`, seed 424701. All logged games passed replay
  `--check`.
- **Runner smoke:** `BENCH1V1_SMOKE=1 scripts/run_1v1.sh`, then `--archive`
  and `sha256sum -c`. 4 games per test, seeds 424701 and 424801.
- **4-player digests:** 9 before/after runs (1v3, logged 1v3, 2v2, mixed
  counted, trades) matched on both engines. The action-log bytes, the result
  JSON without timings and the masked console output were identical.
- **Tests:** the existing and new tests of the adapter and the bench pass on
  both interpreters: `tests/test_catanatron_adapter.py` and
  `tests/test_bench_script.py` (which includes the new 1v1 and
  `analyze_1v1.py` tests). The other suites that import the bench, adapter or
  proof tooling pass as well: `tests/test_prove_strength.py`,
  `tests/test_public_info.py` and `tests/test_ablate_catanatron.py`.
