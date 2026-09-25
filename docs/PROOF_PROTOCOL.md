# Strength proof protocol (pre-registered 2026-09-25, before any proof game)

This file fixes the hypotheses, opponents, sample sizes, seeds, statistics and
pass thresholds **before** the proof games are played, so the p-values cannot
be tuned after seeing results.  Any later change must be a dated amendment
below the original text, and games played before an amendment keep the
original rules.

## Bot under test

`search:depth=1,beam=4,expand=8,evaluator=heuristic` (the default bot), code
at the commit that adds this file plus the benchmark tooling commits that
follow it (no strategy change between this commit and the proof runs).
Trading in Catanatron 3.3 games: on only if the tooling verification judged
it honest (catanatron's players answering offers with their own evaluation);
otherwise off, and the proof says so.

## Opponents

| id | engine | player | role |
|---|---|---|---|
| V | catanatron 3.3.0 | `ValueFunctionPlayer` (defaults) | Catanatron strong bot |
| A | catanatron 3.3.0 | `AlphaBetaPlayer` (defaults, depth 2) | Catanatron's strongest bot |
| S | catanatron 3.3.0 | `SameTurnAlphaBetaPlayer` (defaults) | Catanatron strong bot |
| vf | catanatron 3.2.1 | our `ValueFunctionPlayer` stand-in | stronger-than-Catanatron reference |
| ab | catanatron 3.2.1 | our `AlphaBetaPlayer` stand-in | stronger-than-Catanatron reference |

## Formats and sample sizes

* **1v3**: one catanbot seat against three copies of the opponent, 4
  players, catanbot's seat rotated `g % 4` (every seat exactly 1/4 of the
  games; seat 0 moves first).  Null "no better than an opponent seat": win
  rate <= 0.25.
* **2v2 mixed**: two catanbot seats and two opponent seats, seat patterns
  rotated over the 6 arrangements.  Null "no better head to head": the
  winner is a catanbot seat with probability <= 0.5.

| test | opponent | format | games |
|---|---|---|---|
| T1 | V | 1v3 | 1000 |
| T2 | A | 1v3 | 400 |
| T3 | S | 1v3 | 400 |
| T4 | V | 2v2 | 1000 |
| T5 | A | 2v2 | 400 |
| T6 | S | 2v2 | 400 |
| R1 | vf | 1v3 | 1000 |
| R2 | ab | 1v3 | 400 |

Seeds: base seed 900001 for T1-T6 and 900101 for R1-R2 (never used before),
PYTHONHASHSEED pinned to 0, catanatron default rules (10 VP, discard above 7
cards, random boards with random ports and no adjacent 6/8), turn cap per
catanatron default.  Games that hit the turn cap count as losses for
catanbot.  A crashed game is re-run with the same seed once; a second crash
counts as a loss and is reported.

## Statistics

Exact one-sided binomial tests (Clopper-Pearson intervals), wins counted per
game from catanbot's side.  Multiple comparisons: Holm-Bonferroni over the six
tests T1-T6.

## Claims and pass thresholds

**Claim 1 - "better than Catanatron's strong bots".**  All six tests T1-T6
reject their null at family-wise alpha = 0.01 (Holm).

**Claim 2 - "ready for human testing".**  All of:

1. T1-T6 reject their null at family-wise alpha = 5.7e-7 (a one-sided
   5-sigma level, Holm);
2. effect size, not just significance: the lower 99 % Clopper-Pearson bound
   of the 1v3 win rate is >= 0.35 in T1-T3 and the lower 99 % bound of the
   2v2 catanbot-win share is >= 0.55 in T4-T6;
3. seat robustness: in each of T1-T3 every seat's win rate exceeds 0.25
   (one-sided exact test, p < 0.05 per seat);
4. zero adapter errors, illegal-action fallbacks and crashes over all proof
   games;
5. not beaten by our stronger stand-ins: in R1 and R2 the 1v3 win rate is not
   significantly below 0.25 (two-sided exact test, p > 0.01).

Claim 2 is about readiness for *supervised* human testing with the advisor;
it does not claim the bot beats strong humans.  The screenshot parser has
only been validated on synthetic renders; real Colonist screenshots are a
separate gate.

## If a claim fails

A loss analysis is produced for every opponent where a claim fails: action
logs of the lost games (replayable), where the VP gap opens (turn), opening
quality (production, diversity, port access vs the winner), build order,
robber exposure, dev-card use, trades, and the decisions where the
opponent's choice beat ours by rollout.  The summary states why the bot is
being outsmarted, with numbers.

## Amendment 2026-09-25: Tooling

Written before any proof game.  It changes none of the hypotheses,
opponents, formats, sample sizes, seeds, statistics or thresholds above; it
records how the tooling plays and reads them.

**Trading.**  The tooling verification found that catanatron 3.3's own
players never answer a trade offer with their own evaluation
(`ValueFunctionPlayer` always rejects, `AlphaBetaPlayer` /
`SameTurnAlphaBetaPlayer` raise on `REJECT_TRADE`), so by the rule under
"Bot under test" trading is **off** (`--trades off`) in every proof game.

**Running.**  `scripts/run_proof.sh` plays every test with
`scripts/bench_catanatron.py` and these arguments (T1-T6 with
`/home/user/venv_cat33/bin/python`, catanatron 3.3.0; R1-R2 with the system
`python3`, catanatron 3.2.1):

| test | bench arguments | chunk (games) |
|---|---|---|
| T1 | `--opponent value --our-seats 1 --seed 900001`, games 0..999 | 250 |
| T2 | `--opponent alphabeta --our-seats 1 --seed 900001`, games 0..399 | 40 |
| T3 | `--opponent sameturn --our-seats 1 --seed 900001`, games 0..399 | 50 |
| T4 | `--opponent value --our-seats 2 --seed 900001`, games 0..999 | 250 |
| T5 | `--opponent alphabeta --our-seats 2 --seed 900001`, games 0..399 | 40 |
| T6 | `--opponent sameturn --our-seats 2 --seed 900001`, games 0..399 | 50 |
| R1 | `--opponent vf --our-seats 1 --seed 900101`, games 0..999 | 250 |
| R2 | `--opponent ab --our-seats 1 --seed 900101`, games 0..399 | 80 |

plus, for every chunk, `--spec "search:depth=1,beam=4,expand=8,evaluator=heuristic"
--trades off --hash-seed 0 --workers 3 --rerun-crashes --log-actions DIR
--verbose --game-range A:B --json FILE`, under `timeout 1200` (20 minutes).
Opponent parameters are catanatron's defaults (no `--opponent-params`).
Game `g` always has catanatron seed `seed * 100003 + g + 1`; in 1v3
catanbot sits in seat `g % 4`, in 2v2 its two seats are arrangement `g % 6`
of (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) (turn-order patterns `CCoo`,
`CoCo`, `CooC`, `oCCo`, `oCoC`, `ooCC`); seat = turn-order position, seat 0
moves first.  1000 and 400 are not multiples of 6: the first four
arrangements get one game more (167 / 166 and 67 / 66).  The two catanbot
seats of a 2v2 game are independent bot instances (same spec; bot seeds
`seed` and `seed + 7919`).  Because every per-game quantity depends on the
game index alone, the chunks `--game-range A:B` that cover `0..N-1` play
exactly the games of one uninterrupted run.  A chunk whose JSON exists is
skipped (the script can be restarted after an interruption); a chunk that
times out is split in two, down to single games.  A game that crashes is
re-played once with the same seed (`--rerun-crashes`); a second crash is
recorded as a loss.  Commands:

    scripts/run_proof.sh                 # all tests, then replay --check of every log and the analysis
    scripts/run_proof.sh T2 T5           # some tests only (resumes where they stopped)
    scripts/run_proof.sh --analyze       # replay checks and analysis of what exists
    python3 scripts/prove_strength.py --test T1=OUT/json/T1 ... --test R2=OUT/json/R2 --markdown docs/PROOF.md
    PY scripts/replay_catanatron.py OUT/logs/T2 --check
    PY scripts/replay_catanatron.py OUT/logs/T2 --game 17 --turn 40

(`OUT` defaults to the session scratchpad `.../scratchpad/proof/run`, `PY`
is the interpreter that played the test.)  Smoke runs of the tooling
(`PROOF_SMOKE=1`) use the seeds 424201 / 424301, never the proof seeds.

**Action logs.**  Every game is logged (gzip JSONL, one line per game):
seeds, `PYTHONHASHSEED`, catanatron version, players in turn order
(catanbot spec / opponent class and parameters), board (resource and number
of every land tile, ports), robber start, the shuffled development deck,
every action with its colour and chance outcome (3.3: `ActionRecord`
results; 3.2.1 records dice, stolen card, drawn card and its random
discards in the logged action itself, so nothing needs the seed to be
re-run) and the final state with a full-state fingerprint.
`scripts/replay_catanatron.py` rebuilds any position by replaying the log
into a fresh game on the logged board; `--check` verifies every action and
the final VPs, winner, turns and fingerprint.  About 2.5-3 KiB per game
compressed (about 14 MiB for all 5 000 games).  Games are deterministic
given the seed and `PYTHONHASHSEED` (chunked and single-game runs of the
same seed reproduced each other exactly in the tooling tests), except that
catanatron's alpha-beta players stop a search after 20 s of wall time (their
slowest decision in `docs/BENCHMARKS.md` took 6.1 s); the logs record what
was actually played.

**Reading of the rules** (`scripts/prove_strength.py`, fixed here before
any proof game):

* one-sided p-value = P(X >= wins) under the null (exact, rational
  arithmetic); a game's win is recomputed from the winner seat; turn-cap
  games and games that crashed twice are losses;
* Clopper-Pearson intervals are the central two-sided ones: "the lower 99 %
  bound" is the lower end of the 99 % interval (0.5 % in each tail), the
  conservative reading;
* Holm-Bonferroni is always over the six tests T1-T6 (a test without
  results enters with p = 1);
* condition 3: every seat of T1-T3 has one-sided exact p < 0.05 against 0.25;
* condition 4, over all eight tests: adapter errors = the catanbot player's
  `errors` (a decision raised) + `observe_errors`, illegal-action fallbacks
  = `fallback` (no legal action had a catanatron equivalent, or the bot's
  own choice could not be played and a default was played instead, e.g. a
  discard the engine does not accept), crashes =
  every crashed attempt, even one whose re-run succeeded; `unmapped_top`
  (the search's first choice had no catanatron equivalent and the best
  mapped one was played) is reported but is not a failure;
* condition 5 fails only when a 1v3 rate <= 0.25 has two-sided exact
  p <= 0.01 (minlike rule, as scipy and R);
* both claims also require the data to be the registered data: exactly
  games 0..N-1 of the registered seed, the registered opponent with default
  parameters, engine, format and seats, the registered spec, trades off,
  `PYTHONHASHSEED=0`, 10 VP to win and discard limit 7, each read from the
  bench's run metadata (a field that is missing, or results without that
  metadata, count as not registered); the opponent must match both as
  preset and as class (on 3.3 the stand-ins `vf` / `ab` have catanatron's
  class names).

## Amendment 2026-09-25 (2): Colonist-level information (claim 3)

Written before any game in the new information mode.  It adds tests and a
claim; it changes nothing in T1-T6, R1-R2 or claims 1 and 2.

**Why.**  In T1-T6 both sides see every hand and dev card (Catanatron exposes
the full state, and Catanatron's own bots read it).  That is fair between
the two sides but it is not the Colonist game.  Claim 3 tests our bot with
exactly the information a Colonist player has, while Catanatron's bots keep
the full view (which can only help them).

**Information model for our bot ("counted" mode).**  Always known: board,
robber, buildings, roads, bank per resource, dev-deck size, every hand size
and dev-card count, every played dev card, awards, our own hand and dev
cards.  Public events with content: production per player from each roll,
builds, bank / port trades, domestic trades, Monopoly takes, Year of Plenty
picks, Road Building, dev purchases (count only).  Hidden from third parties:
the resource of a robber steal (known only to thief and victim), the types
of discarded cards (count public), the type of a dev card until it is
played.  Opponents' hands are a card-counting belief (exact except for the
hidden events above); the bot decides on determinizations sampled from that
belief (default 4 samples per decision).

**Tests.**  1v3, same format and rules as T1-T3, seed 900201, PYTHONHASHSEED
0, trading off, bot spec as in T1-T3 plus `--info counted` with default
samples, run from a frozen snapshot of the first commit that contains the
counted mode and its tests (recorded in docs/PROOF.md):

| test | opponent | format | games |
|---|---|---|---|
| T7 | V (catanatron ValueFunctionPlayer) | 1v3 | 1000 |
| T8 | A (catanatron AlphaBetaPlayer) | 1v3 | 400 |
| T9 | S (catanatron SameTurnAlphaBetaPlayer) | 1v3 | 400 |

**Claim 3 - "strong under Colonist information".**  T7-T9 each reject the
null "win rate <= 0.25" (exact one-sided binomial, Holm over T7-T9) at
family-wise alpha = 5.7e-7; the lower bound of the central 99 %
Clopper-Pearson interval is >= 0.35 in each; every seat's win rate exceeds
0.25 (one-sided exact, p < 0.05 per seat) in each; zero adapter errors,
fallbacks and crashes.

**Readiness for human testing** now requires claim 2 AND claim 3 (a
stricter bar than before; claim 2's own conditions are unchanged).

## Amendment 2026-09-25 (3): mixed Catanatron table (claim 4)

Written before any mixed-table game.  Adds tests and a claim; changes
nothing above.

**Format "1v3 mixed".**  One catanbot seat against one copy each of
catanatron's ValueFunctionPlayer, AlphaBetaPlayer and
SameTurnAlphaBetaPlayer (defaults).  catanbot's seat rotates `g % 4`; the
three opponents take the remaining seats in permutation `(g // 4) % 6` of
the order (value, alphabeta, sameturn), so every opponent sits in every
relative position equally often over each 24 consecutive games.  Null:
catanbot's win rate <= 0.25.

| test | information for our bot | games | seed |
|---|---|---|---|
| T10 | full (as T1-T6) | 400 | 900301 |
| T11 | counted (as T7-T9) | 400 | 900401 |

Rules, trading off, PYTHONHASHSEED 0, chunking, logging and crash handling
as for the other tests; T10 and T11 run from the snapshot that contains the
mixed-table option (recorded in docs/PROOF.md).

**Claim 4 - "beats a mixed Catanatron table".**  T10 and T11 each reject
their null at family-wise alpha = 5.7e-7 (Holm over T10-T11); the lower
bound of the central 99 % Clopper-Pearson interval is >= 0.35 in each;
every seat's win rate exceeds 0.25 (one-sided exact, p < 0.05 per seat);
zero adapter errors, fallbacks and crashes.

**Readiness for human testing** now requires claims 2, 3 and 4.
