# Strength proof: claims 1 and 2 PASS

**Verdict (2026-09-26 00:25 UTC):** our bot beats Catanatron's strong bots
(claim 1) and meets the pre-registered bar for *supervised* human testing
(claim 2), under the protocol committed before any proof game
(docs/PROOF_PROTOCOL.md, `704849b`).

**Readiness is not decided yet.**  Amendments 2 and 3 made readiness also
require claims 3 and 4:
- claim 3, T7-T9: our bot sees only what a Colonist player sees;
- claim 4, T10-T11: a mixed table of one ValueFunction, one AlphaBeta and
  one SameTurnAlphaBeta.

Those tests started at 00:25 UTC and are reported below when they finish.

## What was played

- **Games:** 5,000 in total, 0 errors, 0 illegal-action fallbacks, 0 crashes,
  0 turn-cap games.
- **Our bot:** `search:depth=1,beam=4,expand=8,evaluator=heuristic`, run from
  a frozen worktree of commit `9984181` with its own C++ build.
- **Opponents:** Catanatron 3.3.0, unmodified GitHub checkout `ecf9311`, for
  T1-T6.  For R1-R2, our own stronger stand-ins on Catanatron 3.2.1 (PyPI).
- **Seeds:** base seeds 900001 (T1-T6) and 900101 (R1-R2), `PYTHONHASHSEED=0`.
- **Trading:** off in every game.
- **Replays:** every game is replayable, and all 5,000 replay action for
  action from the archived logs.

## How to re-check

- The evidence is in `proof/` (see proof/README.md).
- Re-run the registered analysis script on the archived copy:
  ```
  ARGS=""; for t in T1 T2 T3 T4 T5 T6 R1 R2; do ARGS="$ARGS --test $t=proof/$t/results"; done
  python3 scripts/prove_strength.py $ARGS
  ```
  That is the current script, extended for T7-T11.  It gives the same
  numbers and verdicts for T1-R2.  To use exactly the script the proof ran
  with (commit `9984181`):
  ```
  git show 9984181:scripts/prove_strength.py > /tmp/prove_9984181.py
  python3 /tmp/prove_9984181.py $ARGS
  ```
  On the archived copy it prints output identical to the original run's.
- Replay every logged game:
  - T1-T6: `$PY33 scripts/replay_catanatron.py proof/T1/logs --check`
    (Catanatron 3.3.0);
  - R1-R2: `python3 scripts/replay_catanatron.py proof/R1/logs --check`
    (Catanatron 3.2.1).
- A skeptic's questions and the evidence for each answer: docs/SCRUTINY.md.

## Opponents' thinking time (per test, from the result files)

| test | opponents' slowest single decision | our bot, s per game | each opponent seat, s per game |
|---|---|---|---|
| T1 / T4 (ValueFunction) | 0.1 / 0.3 s | 0.95 / 1.14 | 0.23 / 0.32 |
| T2 / T5 (AlphaBeta) | 6.5 / 7.7 s | 0.90 / 0.80 | 8.4 / 8.4 |
| T3 / T6 (SameTurnAlphaBeta) | 16.3 / 5.9 s | 1.18 / 0.79 | 10.7 / 7.8 |
| R1 / R2 (stand-ins) | 0.1 / 1.1 s | 0.81 / 0.78 | 0.15 / 2.3 |

- Catanatron's search bots stop at 20 s per decision; no decision reached it.
- Against them our bot used about 10x less time per game than each opponent
  seat.

## Notes on reading the numbers

- **The 5-sigma label:** alpha = 5.7e-7 is the protocol's registered
  threshold.  The protocol calls it "one-sided 5-sigma", but as a one-sided
  threshold it is 4.87 sigma; a strict one-sided 5 sigma is 2.87e-7.  Every
  test also clears that stricter line by far: the largest Holm-adjusted
  p-value is 1.85e-32.
- **T2 and T3** are different opponents that happen to have equal totals
  (docs/SCRUTINY.md Q1).  0 of 400 game pairs are identical; the chance of
  equal totals is 2.8 %.
- **R1-R2** are our own stronger stand-ins.  Condition 5 only requires our
  bot not to be significantly *below* 25 % against them.  It scored 27.8 %
  and 25.8 %: about even with a stronger-than-Catanatron version of the
  same bots.  In both, seat 0 (moving first) did best (36-38 %).
- **Scope:** full information in T1-T6 (Catanatron's bots see all hands too),
  no trading, Catanatron's bots as the only outside opponents.  This is not
  evidence about strong human players.

---

## Generated report (scripts/prove_strength.py on the archived copy)

Generated 2026-09-26 by `scripts/prove_strength.py` from the bench results of the pre-registered protocol `docs/PROOF_PROTOCOL.md` (2026-09-25).  Bot under test: `search:depth=1,beam=4,expand=8,evaluator=heuristic`; domestic trading off in every game (catanatron 3.3's players never answer offers with their own evaluation, see the protocol's tooling amendment); `PYTHONHASHSEED=0`; games that hit the turn cap count as losses.

### Verdict

| claim | verdict | failed conditions |
|---|---|---|
| 1 - better than Catanatron's strong bots | **PASS** | - |
| 2 - ready for supervised human testing | **PASS** | - |

### Tests

| test | opponent | format | games | wins | win rate | null | one-sided p | 95 % CP | 99 % CP | avg VP ours / opp | turn cap | errors / fallbacks / crashes | as registered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | `value` (protocol `value` = ValueFunctionPlayer; catanatron 3.3.0) | 1v3 | 1000 / 1000 | 628 | 0.628 | 0.25 | 3.80e-140 | [0.597, 0.658] | [0.588, 0.667] | 8.76 / 6.00 | 0 | 0 / 0 / 0 | yes |
| T2 | `alphabeta` (protocol `alphabeta` = AlphaBetaPlayer; catanatron 3.3.0) | 1v3 | 400 / 400 | 218 | 0.545 | 0.25 | 2.90e-36 | [0.495, 0.595] | [0.479, 0.610] | 8.47 / 6.18 | 0 | 0 / 0 / 0 | yes |
| T3 | `sameturn` (protocol `sameturn` = SameTurnAlphaBetaPlayer; catanatron 3.3.0) | 1v3 | 400 / 400 | 218 | 0.545 | 0.25 | 2.90e-36 | [0.495, 0.595] | [0.479, 0.610] | 8.50 / 6.10 | 0 | 0 / 0 / 0 | yes |
| T4 | `value` (protocol `value` = ValueFunctionPlayer; catanatron 3.3.0) | 2v2 | 1000 / 1000 | 836 | 0.836 | 0.50 | 2.50e-109 | [0.812, 0.858] | [0.804, 0.865] | 7.89 / 5.95 | 0 | 0 / 0 / 0 | yes |
| T5 | `alphabeta` (protocol `alphabeta` = AlphaBetaPlayer; catanatron 3.3.0) | 2v2 | 400 / 400 | 323 | 0.807 | 0.50 | 3.17e-37 | [0.765, 0.845] | [0.752, 0.856] | 7.87 / 6.12 | 0 | 0 / 0 / 0 | yes |
| T6 | `sameturn` (protocol `sameturn` = SameTurnAlphaBetaPlayer; catanatron 3.3.0) | 2v2 | 400 / 400 | 315 | 0.787 | 0.50 | 1.85e-32 | [0.744, 0.827] | [0.730, 0.838] | 7.76 / 6.07 | 0 | 0 / 0 / 0 | yes |
| R1 | `vf` (protocol `vf` = ValueFunctionPlayer; catanatron 3.2.1) | 1v3 | 1000 / 1000 | 278 | 0.278 | 0.25 | 0.0232 | [0.250, 0.307] | [0.242, 0.316] | 7.44 / 7.32 | 0 | 0 / 0 / 0 | yes |
| R2 | `ab` (protocol `ab` = AlphaBetaPlayer; catanatron 3.2.1) | 1v3 | 400 / 400 | 103 | 0.258 | 0.25 | 0.3831 | [0.215, 0.303] | [0.203, 0.318] | 7.09 / 7.25 | 0 | 0 / 0 / 0 | yes |

### Seats (1v3) and arrangements (2v2)

* T1: seat 0 171/250 = 0.684 (p 5.17e-47), seat 1 149/250 = 0.596 (p 5.48e-31), seat 2 151/250 = 0.604 (p 2.69e-32), seat 3 157/250 = 0.628 (p 2.13e-36)
* T2: seat 0 52/100 = 0.520 (p 6.58e-9), seat 1 55/100 = 0.550 (p 1.54e-10), seat 2 54/100 = 0.540 (p 5.59e-10), seat 3 57/100 = 0.570 (p 1.03e-11)
* T3: seat 0 54/100 = 0.540 (p 5.59e-10), seat 1 55/100 = 0.550 (p 1.54e-10), seat 2 51/100 = 0.510 (p 2.13e-8), seat 3 58/100 = 0.580 (p 2.51e-12)
* T4: `CCoo` 147/167 = 0.880, `CoCo` 132/167 = 0.790, `CooC` 150/167 = 0.898, `oCCo` 130/167 = 0.778, `oCoC` 135/166 = 0.813, `ooCC` 142/166 = 0.855
* T5: `CCoo` 58/67 = 0.866, `CoCo` 50/67 = 0.746, `CooC` 57/67 = 0.851, `oCCo` 47/67 = 0.701, `oCoC` 54/66 = 0.818, `ooCC` 57/66 = 0.864
* T6: `CCoo` 57/67 = 0.851, `CoCo` 51/67 = 0.761, `CooC` 52/67 = 0.776, `oCCo` 53/67 = 0.791, `oCoC` 49/66 = 0.742, `ooCC` 53/66 = 0.803
* R1: seat 0 91/250 = 0.364 (p 4.14e-5), seat 1 69/250 = 0.276 (p 0.1896), seat 2 63/250 = 0.252 (p 0.4951), seat 3 55/250 = 0.220 (p 0.8797)
* R2: seat 0 38/100 = 0.380 (p 0.0027), seat 1 25/100 = 0.250 (p 0.5383), seat 2 18/100 = 0.180 (p 0.9624), seat 3 22/100 = 0.220 (p 0.7886)

Seat 0 moves first; one-sided exact p against 0.25 per seat.  2v2 patterns list the turn order (`C` = catanbot, `o` = opponent).

### Holm-Bonferroni over T1-T6

| step | test | p | threshold (0.01) | claim 1 | threshold (5.7e-7) | claim 2 |
|---|---|---|---|---|---|---|
| 1 | T1 | 3.80e-140 | 0.001667 | reject | 9.50e-8 | reject |
| 2 | T4 | 2.50e-109 | 0.002 | reject | 1.14e-7 | reject |
| 3 | T5 | 3.17e-37 | 0.0025 | reject | 1.42e-7 | reject |
| 4 | T2 | 2.90e-36 | 0.003333 | reject | 1.90e-7 | reject |
| 5 | T3 | 2.90e-36 | 0.005 | reject | 2.85e-7 | reject |
| 6 | T6 | 1.85e-32 | 0.01 | reject | 5.70e-7 | reject |

### Conditions

**Claim 1: PASS**

* PASS - results of T1, T2, T3, T4, T5, T6 complete and as pre-registered (opponent, format, engine, seeds, seats, spec, trades off, PYTHONHASHSEED=0, games 0..N-1)
* PASS - T1-T6 reject their nulls at family-wise alpha = 0.01 (Holm) (max Holm-adjusted p = 1.85e-32)

**Claim 2: PASS**

* PASS - results of T1, T2, T3, T4, T5, T6, R1, R2 complete and as pre-registered (opponent, format, engine, seeds, seats, spec, trades off, PYTHONHASHSEED=0, games 0..N-1)
* PASS - T1-T6 reject their nulls at family-wise alpha = 5.7e-7 (Holm) (max Holm-adjusted p = 1.85e-32)
* PASS - effect size: lower 99 % Clopper-Pearson bound >= 0.35 (T1-T3) / >= 0.55 (T4-T6) (T1 0.588; T2 0.479; T3 0.479; T4 0.804; T5 0.752; T6 0.730)
* PASS - seat robustness: every seat of T1-T3 above 0.25 (one-sided exact p < 0.05)
* PASS - zero adapter errors, illegal-action fallbacks and crashes over all proof games
* PASS - R1 / R2: 1v3 win rate not significantly below 0.25 (two-sided exact p > 0.01) (R1 0.278 (p = 0.0445); R2 0.258 (p = 0.7291))

### Method

Exact one-sided binomial tests against the protocol nulls (1v3 win rate 0.25, 2v2 catanbot-win share 0.5), computed in rational arithmetic; Clopper-Pearson intervals are the central two-sided ones (the "lower 99 % bound" is the lower end of the 99 % interval, 0.5 % per tail); R1 / R2 use the exact two-sided test (minlike rule, as scipy / R).  Holm-Bonferroni step-down over the six T tests.  Every game is replayable from the action logs (`scripts/replay_catanatron.py`).
scipy cross-check of every p-value and interval: largest relative difference 7.1e-13.

Result files: `proof/R1/results/R1_g00000-00250.json`, `proof/R1/results/R1_g00250-00500.json`, `proof/R1/results/R1_g00500-00750.json`, `proof/R1/results/R1_g00750-01000.json`, `proof/R2/results/R2_g00000-00080.json`, `proof/R2/results/R2_g00080-00160.json`, `proof/R2/results/R2_g00160-00240.json`, `proof/R2/results/R2_g00240-00320.json`, `proof/R2/results/R2_g00320-00400.json`, `proof/T1/results/T1_g00000-00250.json`, `proof/T1/results/T1_g00250-00500.json`, `proof/T1/results/T1_g00500-00750.json`, `proof/T1/results/T1_g00750-01000.json`, `proof/T2/results/T2_g00000-00040.json`, `proof/T2/results/T2_g00040-00080.json`, `proof/T2/results/T2_g00080-00120.json`, `proof/T2/results/T2_g00120-00160.json`, `proof/T2/results/T2_g00160-00200.json`, `proof/T2/results/T2_g00200-00240.json`, `proof/T2/results/T2_g00240-00280.json`, `proof/T2/results/T2_g00280-00320.json`, `proof/T2/results/T2_g00320-00360.json`, `proof/T2/results/T2_g00360-00400.json`, `proof/T3/results/T3_g00000-00050.json`, `proof/T3/results/T3_g00050-00100.json`, `proof/T3/results/T3_g00100-00150.json`, `proof/T3/results/T3_g00150-00200.json`, `proof/T3/results/T3_g00200-00250.json`, `proof/T3/results/T3_g00250-00300.json`, `proof/T3/results/T3_g00300-00350.json`, `proof/T3/results/T3_g00350-00400.json`, `proof/T4/results/T4_g00000-00250.json`, `proof/T4/results/T4_g00250-00500.json`, `proof/T4/results/T4_g00500-00750.json`, `proof/T4/results/T4_g00750-01000.json`, `proof/T5/results/T5_g00000-00040.json`, `proof/T5/results/T5_g00040-00080.json`, `proof/T5/results/T5_g00080-00120.json`, `proof/T5/results/T5_g00120-00160.json`, `proof/T5/results/T5_g00160-00200.json` ...
