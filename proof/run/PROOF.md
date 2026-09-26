# Strength proof results

Generated 2026-09-26 by `scripts/prove_strength.py` from the bench results of the pre-registered protocol `docs/PROOF_PROTOCOL.md` (2026-09-25, with its amendments).  Bot under test: `search:depth=1,beam=4,expand=8,evaluator=heuristic`; domestic trading off in every game (catanatron 3.3's players never answer offers with their own evaluation, see the protocol's tooling amendment); `PYTHONHASHSEED=0`; games that hit the turn cap count as losses.  In T7-T9 and T11 the bot sees only what a Colonist player sees (`--info counted`, 4 determinizations per searched decision, discarded cards hidden; Catanatron's bots keep the full view); T10-T11 seat one catanbot against one ValueFunctionPlayer, one AlphaBetaPlayer and one SameTurnAlphaBetaPlayer (format `1v3-mixed`).

## Verdict

| claim | verdict | failed conditions |
|---|---|---|
| 1 - better than Catanatron's strong bots | **PASS** | - |
| 2 - ready for supervised human testing | **PASS** | - |
| 3 - strong under Colonist information (T7-T9) | **PASS** | - |
| 4 - beats a mixed Catanatron table (T10-T11) | **PASS** | - |
| readiness for supervised human testing (claims 2, 3 and 4) | **PASS** | - |

## Tests

| test | opponent | format | information | games | wins | win rate | null | one-sided p | 95 % CP | 99 % CP | avg VP ours / opp | turn cap | errors / fallbacks / crashes | as registered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | `value` (protocol `value` = ValueFunctionPlayer; catanatron 3.3.0) | 1v3 | full | 1000 / 1000 | 628 | 0.628 | 0.25 | 3.80e-140 | [0.597, 0.658] | [0.588, 0.667] | 8.76 / 6.00 | 0 | 0 / 0 / 0 | yes |
| T2 | `alphabeta` (protocol `alphabeta` = AlphaBetaPlayer; catanatron 3.3.0) | 1v3 | full | 400 / 400 | 218 | 0.545 | 0.25 | 2.90e-36 | [0.495, 0.595] | [0.479, 0.610] | 8.47 / 6.18 | 0 | 0 / 0 / 0 | yes |
| T3 | `sameturn` (protocol `sameturn` = SameTurnAlphaBetaPlayer; catanatron 3.3.0) | 1v3 | full | 400 / 400 | 218 | 0.545 | 0.25 | 2.90e-36 | [0.495, 0.595] | [0.479, 0.610] | 8.50 / 6.10 | 0 | 0 / 0 / 0 | yes |
| T4 | `value` (protocol `value` = ValueFunctionPlayer; catanatron 3.3.0) | 2v2 | full | 1000 / 1000 | 836 | 0.836 | 0.50 | 2.50e-109 | [0.812, 0.858] | [0.804, 0.865] | 7.89 / 5.95 | 0 | 0 / 0 / 0 | yes |
| T5 | `alphabeta` (protocol `alphabeta` = AlphaBetaPlayer; catanatron 3.3.0) | 2v2 | full | 400 / 400 | 323 | 0.807 | 0.50 | 3.17e-37 | [0.765, 0.845] | [0.752, 0.856] | 7.87 / 6.12 | 0 | 0 / 0 / 0 | yes |
| T6 | `sameturn` (protocol `sameturn` = SameTurnAlphaBetaPlayer; catanatron 3.3.0) | 2v2 | full | 400 / 400 | 315 | 0.787 | 0.50 | 1.85e-32 | [0.744, 0.827] | [0.730, 0.838] | 7.76 / 6.07 | 0 | 0 / 0 / 0 | yes |
| R1 | `vf` (protocol `vf` = ValueFunctionPlayer; catanatron 3.2.1) | 1v3 | full | 1000 / 1000 | 278 | 0.278 | 0.25 | 0.0232 | [0.250, 0.307] | [0.242, 0.316] | 7.44 / 7.32 | 0 | 0 / 0 / 0 | yes |
| R2 | `ab` (protocol `ab` = AlphaBetaPlayer; catanatron 3.2.1) | 1v3 | full | 400 / 400 | 103 | 0.258 | 0.25 | 0.3831 | [0.215, 0.303] | [0.203, 0.318] | 7.09 / 7.25 | 0 | 0 / 0 / 0 | yes |
| T7 | `value` (protocol `value` = ValueFunctionPlayer; catanatron 3.3.0) | 1v3 | counted (K = 4, discards hidden) | 1000 / 1000 | 617 | 0.617 | 0.25 | 1.66e-132 | [0.586, 0.647] | [0.576, 0.656] | 8.75 / 6.04 | 0 | 0 / 0 / 0 | yes |
| T8 | `alphabeta` (protocol `alphabeta` = AlphaBetaPlayer; catanatron 3.3.0) | 1v3 | counted (K = 4, discards hidden) | 400 / 400 | 234 | 0.585 | 0.25 | 9.79e-46 | [0.535, 0.634] | [0.520, 0.648] | 8.77 / 6.08 | 0 | 0 / 0 / 0 | yes |
| T9 | `sameturn` (protocol `sameturn` = SameTurnAlphaBetaPlayer; catanatron 3.3.0) | 1v3 | counted (K = 4, discards hidden) | 400 / 400 | 227 | 0.568 | 0.25 | 1.87e-41 | [0.517, 0.617] | [0.502, 0.631] | 8.57 / 6.15 | 0 | 0 / 0 / 0 | yes |
| T10 | `value,alphabeta,sameturn` (protocol `value,alphabeta,sameturn` = ValueFunctionPlayer+AlphaBetaPlayer+SameTurnAlphaBetaPlayer; catanatron 3.3.0) | 1v3-mixed | full | 400 / 400 | 244 | 0.610 | 0.25 | 3.13e-52 | [0.560, 0.658] | [0.545, 0.672] | 8.60 / 6.04 | 0 | 0 / 0 / 0 | yes |
| T11 | `value,alphabeta,sameturn` (protocol `value,alphabeta,sameturn` = ValueFunctionPlayer+AlphaBetaPlayer+SameTurnAlphaBetaPlayer; catanatron 3.3.0) | 1v3-mixed | counted (K = 4, discards hidden) | 400 / 400 | 247 | 0.618 | 0.25 | 2.87e-54 | [0.568, 0.665] | [0.553, 0.680] | 8.71 / 5.99 | 0 | 0 / 0 / 0 | yes |

Errors include the counted mode's tracker errors and belief resets (T7-T9, T11).

## Seats (1v3) and arrangements (2v2)

* T1: seat 0 171/250 = 0.684 (p 5.17e-47), seat 1 149/250 = 0.596 (p 5.48e-31), seat 2 151/250 = 0.604 (p 2.69e-32), seat 3 157/250 = 0.628 (p 2.13e-36)
* T2: seat 0 52/100 = 0.520 (p 6.58e-9), seat 1 55/100 = 0.550 (p 1.54e-10), seat 2 54/100 = 0.540 (p 5.59e-10), seat 3 57/100 = 0.570 (p 1.03e-11)
* T3: seat 0 54/100 = 0.540 (p 5.59e-10), seat 1 55/100 = 0.550 (p 1.54e-10), seat 2 51/100 = 0.510 (p 2.13e-8), seat 3 58/100 = 0.580 (p 2.51e-12)
* T4: `CCoo` 147/167 = 0.880, `CoCo` 132/167 = 0.790, `CooC` 150/167 = 0.898, `oCCo` 130/167 = 0.778, `oCoC` 135/166 = 0.813, `ooCC` 142/166 = 0.855
* T5: `CCoo` 58/67 = 0.866, `CoCo` 50/67 = 0.746, `CooC` 57/67 = 0.851, `oCCo` 47/67 = 0.701, `oCoC` 54/66 = 0.818, `ooCC` 57/66 = 0.864
* T6: `CCoo` 57/67 = 0.851, `CoCo` 51/67 = 0.761, `CooC` 52/67 = 0.776, `oCCo` 53/67 = 0.791, `oCoC` 49/66 = 0.742, `ooCC` 53/66 = 0.803
* R1: seat 0 91/250 = 0.364 (p 4.14e-5), seat 1 69/250 = 0.276 (p 0.1896), seat 2 63/250 = 0.252 (p 0.4951), seat 3 55/250 = 0.220 (p 0.8797)
* R2: seat 0 38/100 = 0.380 (p 0.0027), seat 1 25/100 = 0.250 (p 0.5383), seat 2 18/100 = 0.180 (p 0.9624), seat 3 22/100 = 0.220 (p 0.7886)
* T7: seat 0 167/250 = 0.668 (p 7.95e-44), seat 1 144/250 = 0.576 (p 7.70e-28), seat 2 147/250 = 0.588 (p 1.04e-29), seat 3 159/250 = 0.636 (p 8.00e-38)
* T8: seat 0 63/100 = 0.630 (p 1.18e-15), seat 1 51/100 = 0.510 (p 2.13e-8), seat 2 67/100 = 0.670 (p 1.21e-18), seat 3 53/100 = 0.530 (p 1.96e-9)
* T9: seat 0 57/100 = 0.570 (p 1.03e-11), seat 1 54/100 = 0.540 (p 5.59e-10), seat 2 67/100 = 0.670 (p 1.21e-18), seat 3 49/100 = 0.490 (p 1.99e-7)
* T10: seat 0 70/100 = 0.700 (p 4.37e-21), seat 1 55/100 = 0.550 (p 1.54e-10), seat 2 54/100 = 0.540 (p 5.59e-10), seat 3 65/100 = 0.650 (p 4.13e-17)
  * wins by player: catanbot 244, value 42, alphabeta 54, sameturn 60, none 0 (none = turn cap or crashed)
* T11: seat 0 59/100 = 0.590 (p 5.89e-13), seat 1 64/100 = 0.640 (p 2.26e-16), seat 2 61/100 = 0.610 (p 2.87e-14), seat 3 63/100 = 0.630 (p 1.18e-15)
  * wins by player: catanbot 247, value 24, alphabeta 58, sameturn 71, none 0 (none = turn cap or crashed)

Seat 0 moves first; one-sided exact p against 0.25 per seat.  2v2 patterns list the turn order (`C` = catanbot, `o` = opponent).  In the mixed table catanbot sits in seat `g % 4` and the three opponents follow it in turn order in permutation `(g // 4) % 6` of (value, alphabeta, sameturn).

## Holm-Bonferroni over T1-T6

| step | test | p | threshold (0.01) | claim 1 | threshold (5.7e-7) | claim 2 |
|---|---|---|---|---|---|---|
| 1 | T1 | 3.80e-140 | 0.001667 | reject | 9.50e-8 | reject |
| 2 | T4 | 2.50e-109 | 0.002 | reject | 1.14e-7 | reject |
| 3 | T5 | 3.17e-37 | 0.0025 | reject | 1.42e-7 | reject |
| 4 | T2 | 2.90e-36 | 0.003333 | reject | 1.90e-7 | reject |
| 5 | T3 | 2.90e-36 | 0.005 | reject | 2.85e-7 | reject |
| 6 | T6 | 1.85e-32 | 0.01 | reject | 5.70e-7 | reject |

## Holm-Bonferroni over T7-T9 (claim 3)

| step | test | p | threshold (5.7e-7) | adjusted p | claim 3 |
|---|---|---|---|---|---|
| 1 | T7 | 1.66e-132 | 1.90e-7 | 4.99e-132 | reject |
| 2 | T8 | 9.79e-46 | 2.85e-7 | 1.96e-45 | reject |
| 3 | T9 | 1.87e-41 | 5.70e-7 | 1.87e-41 | reject |

## Holm-Bonferroni over T10-T11 (claim 4)

| step | test | p | threshold (5.7e-7) | adjusted p | claim 4 |
|---|---|---|---|---|---|
| 1 | T11 | 2.87e-54 | 2.85e-7 | 5.74e-54 | reject |
| 2 | T10 | 3.13e-52 | 5.70e-7 | 3.13e-52 | reject |

## Conditions

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

**Claim 3: PASS**

* PASS - results of T7, T8, T9 complete and as pre-registered (opponent(s), format, information mode, engine, seeds, seats and lineups, spec, trades off, PYTHONHASHSEED=0, games 0..N-1)
* PASS - T7-T9 reject their nulls at family-wise alpha = 5.7e-7 (Holm over T7-T9) (max Holm-adjusted p = 1.87e-41)
* PASS - effect size: lower 99 % Clopper-Pearson bound >= 0.35 in each of T7-T9 (T7 0.576; T8 0.520; T9 0.502)
* PASS - seat robustness: every seat of T7-T9 above 0.25 (one-sided exact p < 0.05)
* PASS - zero adapter errors, illegal-action fallbacks, crashes and counted-mode tracker errors / belief resets over T7-T9

**Claim 4: PASS**

* PASS - results of T10, T11 complete and as pre-registered (opponent(s), format, information mode, engine, seeds, seats and lineups, spec, trades off, PYTHONHASHSEED=0, games 0..N-1)
* PASS - T10-T11 reject their nulls at family-wise alpha = 5.7e-7 (Holm over T10-T11) (max Holm-adjusted p = 3.13e-52)
* PASS - effect size: lower 99 % Clopper-Pearson bound >= 0.35 in each of T10-T11 (T10 0.545; T11 0.553)
* PASS - seat robustness: every seat of T10-T11 above 0.25 (one-sided exact p < 0.05)
* PASS - zero adapter errors, illegal-action fallbacks, crashes and counted-mode tracker errors / belief resets over T10-T11

**READINESS for (supervised) human testing = claims 2, 3 and 4 (amendments 2 and 3): PASS**

## Method

Exact one-sided binomial tests against the protocol nulls (1v3 win rate 0.25, 2v2 catanbot-win share 0.5), computed in rational arithmetic; Clopper-Pearson intervals are the central two-sided ones (the "lower 99 % bound" is the lower end of the 99 % interval, 0.5 % per tail); R1 / R2 use the exact two-sided test (minlike rule, as scipy / R).  Holm-Bonferroni step-down over the six T tests.  Claims 3 and 4 (amendments 2 and 3): Holm over T7-T9 and over T10-T11 at 5.7e-7, lower 99 % bound >= 0.35 and every seat above 0.25 (one-sided p < 0.05) in each test, and zero errors, fallbacks and crashes including the counted mode's tracker errors and belief resets.  Every game is replayable from the action logs (`scripts/replay_catanatron.py`).
scipy cross-check of every p-value and interval: largest relative difference 7.4e-13.

Result files: `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R1/g00000-00250.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R1/g00250-00500.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R1/g00500-00750.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R1/g00750-01000.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R2/g00000-00080.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R2/g00080-00160.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R2/g00160-00240.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R2/g00240-00320.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/R2/g00320-00400.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T1/g00000-00250.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T1/g00250-00500.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T1/g00500-00750.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T1/g00750-01000.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00000-00050.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00050-00100.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00100-00150.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00150-00200.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00200-00250.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00250-00300.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00300-00350.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T10/g00350-00400.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00000-00040.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00040-00080.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00080-00120.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00120-00160.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00160-00200.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00200-00240.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00240-00280.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00280-00320.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00320-00360.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T11/g00360-00400.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00000-00040.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00040-00080.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00080-00120.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00120-00160.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00160-00200.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00200-00240.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00240-00280.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00280-00320.json`, `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run/json/T2/g00320-00360.json` ...
