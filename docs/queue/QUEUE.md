# Test queue report

Plan `scripts/queue_plan.json` (806c3d3a0efc), results `/home/user/queue_runs/queue1`, written 2026-09-26 07:51:13.  One line per candidate.  Units: win-rate percentage points of the candidate minus the default (paired); `p` = stage-wise one-sided p (fixed two-sided p for politics / estimate rows), `Holm` = adjusted within the row's tier (provisional `*` until the tier is complete).  Estimates of rows stopped early are biased away from 0 (winner's curse): confirm on fresh seeds.

| area | row | candidate | polarity | design | label | look | pairs | estimate (pp) +- se | unit | base rate / relative | p | Holm | dVP +- se | discordant / diverged | mechanism (cand - def) | CPU-h | next action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| harness | t1_baseline_aa@value | cand | measure | estimate | PASS(A/A identical) | 1 | 400 | +0.0 +- 0.2 | pp (1v3 win rate vs value) | 65.8% / +0% |  |  | +0.00 +- 0.00 | 0.00 / 0.00 | first_city_round +0.00+-0.00; first_settle_round +0.00+-0.00; knights_held_end +0.00+-0.00; knights_played +0.00+-0.00; largest_army +0.00+-0.00; longest_road + | 0.25 | pipeline / pairing verified |
| harness | t1_baseline_aa@vf | cand | measure | estimate | PASS(A/A identical) | 1 | 400 | +0.0 +- 0.2 | pp (1v3 win rate vs vf) | 26.5% / +0% |  |  | +0.00 +- 0.00 | 0.00 / 0.00 | first_city_round +0.00+-0.00; first_settle_round +0.00+-0.00; knights_held_end +0.00+-0.00; knights_played +0.00+-0.00; largest_army +0.00+-0.00; longest_road + | 0.19 | pipeline / pairing verified |
| harness | smoke_expand4@vf | 4 | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 4 | 320 | +3.1 +- 3.2 | pp (1v3 win rate vs vf) | 27.5% / +11% | 0.168 |  | +0.10 +- 0.14 | 0.34 / 0.98 | first_city_round +0.80+-0.38; first_settle_round -0.61+-0.33; knights_held_end +0.05+-0.03; knights_played +0.09+-0.12; largest_army -0.00+-0.03; longest_road - | 0.07 | on ice (default unchanged) |
| harness | smoke_turns_half2@vf | 2 | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 4 | 320 | +0.6 +- 0.6 | pp (1v3 win rate vs vf) | 27.5% / +2% | 0.159 |  | +0.00 +- 0.03 | 0.01 / 0.06 | first_city_round +0.03+-0.03; first_settle_round -0.01+-0.01; knights_held_end +0.00+-0.01; knights_played +0.00+-0.02; largest_army -0.01+-0.00; longest_road + | 0.08 | on ice (default unchanged) |
| harness | crn_pilot_off@vf | 4 | measure | estimate | ESTIMATE(fixed N) | 1 | 400 | +2.8 +- 2.8 | pp (1v3 win rate vs vf) | 26.5% / +10% | 0.329 |  | -0.05 +- 0.13 | 0.32 / 0.98 | first_city_round +0.52+-0.34; first_settle_round -0.61+-0.30; knights_held_end +0.07+-0.03; knights_played +0.03+-0.10; largest_army -0.03+-0.03; longest_road - | 0.08 | measurement recorded |
| harness | crn_pilot_dice@vf | 4 | measure | estimate | ESTIMATE(fixed N) | 1 | 400 | -3.0 +- 2.3 | pp (1v3 win rate vs vf) | 29.2% / -10% | 0.195 |  | -0.02 +- 0.10 | 0.21 / 0.95 | first_city_round +0.51+-0.35; first_settle_round -0.53+-0.26; knights_held_end +0.03+-0.02; knights_played +0.16+-0.09; largest_army +0.04+-0.02; longest_road - | 0.17 | measurement recorded |
| harness | crn_aa_dice@vf | cand | measure | estimate | PASS(A/A identical) | 1 | 400 | +0.0 +- 0.2 | pp (1v3 win rate vs vf) | 29.2% / +0% |  |  | +0.00 +- 0.00 | 0.00 / 0.00 | first_city_round +0.00+-0.00; first_settle_round +0.00+-0.00; knights_held_end +0.00+-0.00; knights_played +0.00+-0.00; largest_army +0.00+-0.00; longest_road + | 0.09 | pipeline / pairing verified |
| trades | t1_trades0_vrule@value | 0 | measure | estimate | ESTIMATE(fixed N) | 1 | 2000 | -24.1 +- 1.2 | pp (1v3 win rate vs value, vs value-rule responders) | 84.3% / -29% | 6.14e-83 | 6.14e-83* | -1.04 +- 0.05 | 0.37 / 1.00 | first_city_round +1.54+-0.17; first_settle_round +3.58+-0.17; knights_held_end -0.01+-0.02; knights_played -0.63+-0.05; largest_army -0.13+-0.01; longest_road - | 2.49 | measurement recorded (headroom for its area) |
| trades | t2_dump0@value | 0 | knockout | knockout | KEEP(unproven) [stopped early] | 2 | 400 | -0.8 +- 0.8 | pp (1v3 win rate vs value) | 65.8% / -1% | 0.817 | 0.817* | -0.03 +- 0.03 | 0.03 / 0.35 | first_city_round -0.03+-0.02; first_settle_round -0.03+-0.04; knights_held_end +0.00+-0.01; knights_played -0.04+-0.02; largest_army -0.01+-0.01; longest_road - | 0.13 | keep the term (default unchanged) |
| trades | acq_breadth_bundle | 1 | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| trades | acq_breadth_confirm@value | 1 | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| trades | acq_breadth_noharm_3p | 1 | measure | estimate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after acq_breadth_bundle |
| trades | acq_calib_native@value | 1 | measure | estimate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| trades | acq_calib | 1 | new | politics | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | politics: one fixed-N screen, after acq_calib_native@value |
| trades | acq_reject_streak_native@value | 3 | measure | estimate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after acq_calib_native@value |
| trades | acq_calib_reject_streak | row |  | human | DEFERRED |  |  |  |  |  |  |  |  |  |  | 0.00 | deferred to human testing (no games) |
| trades | acq_flow_fit | row |  | gate | PASS |  |  |  |  |  |  |  |  |  |  | 0.00 | pipeline / pairing verified |
| trades | acq_floor_selfplay | 1 | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| trades | acq_floor_vrule@value | 1 | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| trades | t3_trades0_vrule@alphabeta | 0 | measure | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| diversification | t1_openings@value | pips_diversity | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | running |
| diversification | t1_openings@value | standin_book | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | running |
| diversification | t1_openings@value | conversion | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | running |
| diversification | t1_openings@value | setup_pick | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | running |
| diversification | t1_openings@vf | pips_diversity | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| diversification | t1_openings@vf | standin_book | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| diversification | t1_openings@vf | conversion | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| diversification | t1_openings@vf | setup_pick | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| diversification | demand_flat_pyeval@vf | 1/1/1/1/1 | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| diversification | demand_mild_pyeval@vf | 1.15/1.15/0.9/1/1 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | parent demand_flat_pyeval@vf (ELIGIBLE) |
| diversification | ports_conversion_cost@value | cand | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | bundle first: div_lr_bundle |
| diversification | ports_conversion_reach@value | cand | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | parent ports_conversion_cost@value (WAITING) |
| diversification | div_lr_bundle | cand | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| diversification | div_lr_bundle-no-conv | cand | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after div_lr_bundle |
| diversification | div_lr_bundle-no-paths | cand | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after div_lr_bundle |
| diversification | t3_openings@alphabeta | pips_diversity | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| diversification | t3_openings@alphabeta | standin_book | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| diversification | t3_openings@alphabeta | setup_pick | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| ports | ports_spot_leader | row |  | human | DEFERRED |  |  |  |  |  |  |  |  |  |  | 0.00 | deferred to human testing (no games) |
| robber | shadow_robber | row |  | gate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| robber | t2_danger_mult_off@value | off | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_robber |
| robber | t2_block_need0@value | 0 | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_robber |
| robber | t2_steal_factor_off@value | off | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_robber |
| robber | t2_rob_break_off@value | off | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_robber |
| robber | t2_turns_half2@value | 2 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_robber |
| robber | t2_knight03@value | 0.3 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_robber |
| robber | t2_exposure0_pyeval@value | 0 | knockout | knockout | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| robber | t1_blockw0_pyeval@value | 0 | knockout | knockout | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| robber | t1_blockw0_pyeval@vf | 0 | knockout | knockout | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| robber | robber_retal_w | row |  | human | DEFERRED |  |  |  |  |  |  |  |  |  |  | 0.00 | deferred to human testing (no games) |
| robber | t3_blockw0_pyeval@alphabeta | 0 | knockout | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| politics | t2_max_slack0_vrule@value | 0 | knockout | politics | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | politics: one fixed-N screen, queued |
| politics | t2_coal_scale2_vrule@value | 2 | knockout | politics | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | politics: one fixed-N screen, queued |
| politics | t2_feed_leader_off_vrule@value | off | knockout | politics | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | politics: one fixed-N screen, queued |
| politics | t2_late_drop0_vrule@value | 0 | knockout | politics | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | politics: one fixed-N screen, queued |
| politics | counters_selfplay | 1 | new | politics | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | politics: one fixed-N screen, queued |
| politics | respond_lookahead_selfplay | 1 | new | politics | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | politics: one fixed-N screen, queued |
| other | t1_search_vs_heur@value | cand | measure | estimate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| other | t1_search_vs_heur@vf | cand | measure | estimate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| other | t1_depth2@value | 2 | new | screen | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| other | t2x_beam2@value | 2 | knockout | knockout | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| other | t2x_expand4@value | 4 | knockout | knockout | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| other | shadow_paths | row |  | gate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| other | paths_main@value | 1 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_paths |
| other | paths_main@vf | 1 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | shadow shadow_paths |
| other | t3_search_vs_heur@alphabeta | cand | measure | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| other | t3_depth2@alphabeta | 2 | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |

## Open rows in scheduler order

| # | area | row | status | remaining CPU-h | weight | why |
|---|---|---|---|---|---|---|
| 1 | trades | acq_breadth_bundle | ELIGIBLE | 3.53 | 0.2031 |  |
| 2 | trades | acq_calib_native@value | ELIGIBLE | 0.07 | 0.1672 |  |
| 3 | trades | acq_floor_selfplay | ELIGIBLE | 3.53 | 0.1340 |  |
| 4 | trades | acq_floor_vrule@value | ELIGIBLE | 1.63 | 0.1294 |  |
| 5 | diversification | t1_openings@value | RUNNING | 2.06 | 0.0625 |  |
| 6 | diversification | t1_openings@vf | ELIGIBLE | 1.57 | 0.0625 |  |
| 7 | diversification | demand_flat_pyeval@vf | ELIGIBLE | 1.80 | 0.0583 |  |
| 8 | diversification | div_lr_bundle | ELIGIBLE | 0.96 | 0.0526 |  |
| 9 | robber | shadow_robber | ELIGIBLE | 1.30 | 0.0039 |  |
| 10 | robber | t2_exposure0_pyeval@value | ELIGIBLE | 1.50 | 0.0034 |  |
| 11 | robber | t1_blockw0_pyeval@value | ELIGIBLE | 2.99 | 0.0034 |  |
| 12 | robber | t1_blockw0_pyeval@vf | ELIGIBLE | 1.80 | 0.0034 |  |
| 13 | politics | t2_max_slack0_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 14 | politics | t2_coal_scale2_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 15 | politics | t2_feed_leader_off_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 16 | politics | t2_late_drop0_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 17 | politics | counters_selfplay | ELIGIBLE | 3.60 | 0.0002 |  |
| 18 | politics | respond_lookahead_selfplay | ELIGIBLE | 3.60 | 0.0002 |  |
| 19 | other | t1_search_vs_heur@value | ELIGIBLE | 1.57 | 0.0001 |  |
| 20 | other | t1_search_vs_heur@vf | ELIGIBLE | 1.02 | 0.0001 |  |
| 21 | other | t1_depth2@value | ELIGIBLE | 1.64 | 0.0001 |  |
| 22 | other | t2x_beam2@value | ELIGIBLE | 0.48 | 0.0001 |  |
| 23 | other | t2x_expand4@value | ELIGIBLE | 0.48 | 0.0001 |  |
| 24 | other | shadow_paths | ELIGIBLE | 0.15 | 0.0001 |  |

Waiting / blocked: acq_breadth_confirm@value (WAITING: confirmation waits for tier t1 to complete (Holm family)); acq_breadth_noharm_3p (WAITING: after acq_breadth_bundle); acq_calib (WAITING: after acq_calib_native@value); acq_reject_streak_native@value (WAITING: after acq_calib_native@value); t3_trades0_vrule@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); demand_mild_pyeval@vf (WAITING: parent demand_flat_pyeval@vf (ELIGIBLE)); ports_conversion_cost@value (WAITING: bundle first: div_lr_bundle); ports_conversion_reach@value (WAITING: parent ports_conversion_cost@value (WAITING)); div_lr_bundle-no-conv (WAITING: after div_lr_bundle); div_lr_bundle-no-paths (WAITING: after div_lr_bundle); t3_openings@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); t2_danger_mult_off@value (WAITING: shadow shadow_robber); t2_block_need0@value (WAITING: shadow shadow_robber); t2_steal_factor_off@value (WAITING: shadow shadow_robber); t2_rob_break_off@value (WAITING: shadow shadow_robber); t2_turns_half2@value (WAITING: shadow shadow_robber); t2_knight03@value (WAITING: shadow shadow_robber); t3_blockw0_pyeval@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); paths_main@value (WAITING: shadow shadow_paths); paths_main@vf (WAITING: shadow shadow_paths); t3_search_vs_heur@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); t3_depth2@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family))

## Per area

| area | rows | final | open | ADOPT | on ice (SHELVE/REJECT) | NOOP | FAILED | deferred | CPU-h spent |
|---|---|---|---|---|---|---|---|---|---|
| harness | 7 | 7 | 0 | 0 | 2 | 0 | 0 | 0 | 0.92 |
| trades | 13 | 4 | 4 | 0 | 0 | 0 | 0 | 1 | 2.62 |
| diversification | 10 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| ports | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0.00 |
| robber | 12 | 1 | 4 | 0 | 0 | 0 | 0 | 1 | 0.00 |
| politics | 6 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| other | 10 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0.00 |

## Bundle proposals (a human commits them; then the league gate)

(no ADOPT yet)

## Deferred to human testing (paste into docs/ABLATIONS.md, Politics rule)

| term | screen | result | status |
|---|---|---|---|
| opponent_model.REJECT_STREAK | none (no bot harness reacts) | n/a | deferred |
| winpaths.SPOT_LEADER | none (no bot harness reacts) | n/a | deferred |
| robber_eval.RETAL_W | none (no bot harness reacts) | n/a | deferred |

CRN pilot: discordant share 0.318 (off) -> 0.215 (dice), ratio 0.68; A/A identical -> PASS: rows with crn auto use --crn dice
