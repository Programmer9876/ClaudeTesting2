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
