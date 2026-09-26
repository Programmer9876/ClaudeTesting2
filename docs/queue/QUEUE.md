# Test queue report

Plan `scripts/queue_plan.json` (c24712175bb9), results `/home/user/queue_runs/queue1`, written 2026-09-26 13:07:42.  One line per candidate.  Units: win-rate percentage points of the candidate minus the default (paired); `p` = stage-wise one-sided p (fixed two-sided p for politics / estimate rows), `Holm` = adjusted within the row's tier (provisional `*` until the tier is complete).  Estimates of rows stopped early are biased away from 0 (winner's curse): confirm on fresh seeds.

| area | row | candidate | polarity | design | label | look | pairs | estimate (pp) +- se | unit | base rate / relative | p | Holm | dVP +- se | discordant / diverged | mechanism (cand - def) | CPU-h | next action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| harness | t1_baseline_aa@value | cand | measure | estimate | PASS(A/A identical) | 1 | 400 | +0.0 +- 0.2 | pp (1v3 win rate vs value) | 65.8% / +0% |  |  | +0.00 +- 0.00 | 0.00 / 0.00 | first_city_round +0.00+-0.00; first_settle_round +0.00+-0.00; knights_held_end +0.00+-0.00; knights_played +0.00+-0.00; largest_army +0.00+-0.00; longest_road + | 0.25 | pipeline / pairing verified |
| harness | t1_baseline_aa@vf | cand | measure | estimate | PASS(A/A identical) | 1 | 400 | +0.0 +- 0.2 | pp (1v3 win rate vs vf) | 26.5% / +0% |  |  | +0.00 +- 0.00 | 0.00 / 0.00 | first_city_round +0.00+-0.00; first_settle_round +0.00+-0.00; knights_held_end +0.00+-0.00; knights_played +0.00+-0.00; largest_army +0.00+-0.00; longest_road + | 0.19 | pipeline / pairing verified |
| harness | smoke_expand4@vf | 4 | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 4 | 320 | +3.1 +- 3.2 | pp (1v3 win rate vs vf) | 27.5% / +11% | 0.168 |  | +0.10 +- 0.14 | 0.34 / 0.98 | first_city_round +0.80+-0.38; first_settle_round -0.61+-0.33; knights_held_end +0.05+-0.03; knights_played +0.09+-0.12; largest_army -0.00+-0.03; longest_road - | 0.07 | on ice (default unchanged) |
| harness | smoke_turns_half2@vf | 2 | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 4 | 320 | +0.6 +- 0.6 | pp (1v3 win rate vs vf) | 27.5% / +2% | 0.159 |  | +0.00 +- 0.03 | 0.01 / 0.06 | first_city_round +0.03+-0.03; first_settle_round -0.01+-0.01; knights_held_end +0.00+-0.01; knights_played +0.00+-0.02; largest_army -0.01+-0.00; longest_road + | 0.08 | on ice (default unchanged) |
| harness | crn_pilot_off@vf | 4 | measure | estimate | ESTIMATE(fixed N) | 1 | 400 | +2.8 +- 2.8 | pp (1v3 win rate vs vf) | 26.5% / +10% | 0.329 |  | -0.05 +- 0.13 | 0.32 / 0.98 | first_city_round +0.52+-0.34; first_settle_round -0.61+-0.30; knights_held_end +0.07+-0.03; knights_played +0.03+-0.10; largest_army -0.03+-0.03; longest_road - | 0.08 | measurement recorded |
| harness | crn_pilot_dice@vf | 4 | measure | estimate | ESTIMATE(fixed N) | 1 | 400 | -3.0 +- 2.3 | pp (1v3 win rate vs vf) | 29.2% / -10% | 0.195 |  | -0.02 +- 0.10 | 0.21 / 0.95 | first_city_round +0.51+-0.35; first_settle_round -0.53+-0.26; knights_held_end +0.03+-0.02; knights_played +0.16+-0.09; largest_army +0.04+-0.02; longest_road - | 0.17 | measurement recorded |
| harness | crn_aa_dice@vf | cand | measure | estimate | PASS(A/A identical) | 1 | 400 | +0.0 +- 0.2 | pp (1v3 win rate vs vf) | 29.2% / +0% |  |  | +0.00 +- 0.00 | 0.00 / 0.00 | first_city_round +0.00+-0.00; first_settle_round +0.00+-0.00; knights_held_end +0.00+-0.00; knights_played +0.00+-0.00; largest_army +0.00+-0.00; longest_road + | 0.09 | pipeline / pairing verified |
| trades | t1_trades0_vrule@value | 0 | measure | estimate | ESTIMATE(fixed N) | 1 | 2000 | -24.1 +- 1.2 | pp (1v3 win rate vs value, vs value-rule responders) | 84.3% / -29% | 6.14e-83 | 8.59e-82* | -1.04 +- 0.05 | 0.37 / 1.00 | first_city_round +1.54+-0.17; first_settle_round +3.58+-0.17; knights_held_end -0.01+-0.02; knights_played -0.63+-0.05; largest_army -0.13+-0.01; longest_road - | 2.49 | measurement recorded (headroom for its area) |
| trades | t2_dump0@value | 0 | knockout | knockout | KEEP(unproven) [stopped early] | 2 | 400 | -0.8 +- 0.8 | pp (1v3 win rate vs value) | 65.8% / -1% | 0.817 | 0.817* | -0.03 +- 0.03 | 0.03 / 0.35 | first_city_round -0.03+-0.02; first_settle_round -0.03+-0.04; knights_held_end +0.00+-0.01; knights_played -0.04+-0.02; largest_army -0.01+-0.01; longest_road - | 0.13 | keep the term (default unchanged) |
| trades | acq_breadth_bundle | 1 | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 2 | 960 | +0.2 +- 1.6 | pp (per seat, 2v2) | 49.8% / +0% | 0.449 | 1* | +0.05 +- 0.07 | 1.00 / 0.00 |  | 2.21 | on ice (default unchanged) |
| trades | acq_breadth_confirm@value | 1 | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| trades | acq_breadth_noharm_3p | 1 | measure | estimate | NOT TRIGGERED |  |  |  |  |  |  |  |  |  |  | 0.00 | after acq_breadth_bundle: condition not met |
| trades | acq_calib_native@value | 1 | measure | estimate | ESTIMATE(fixed N) | 1 | 40 | -12.5 +- 5.3 | pp (1v3 win rate vs value) | 67.5% / -19% | 0.0183 |  | -0.20 +- 0.17 | 0.12 / 1.00 | first_city_round -0.16+-0.24; first_settle_round -0.71+-0.33; knights_held_end +0.20+-0.11; knights_played -0.40+-0.21; largest_army -0.07+-0.07; longest_road - | 0.08 | measurement recorded |
| trades | acq_reject_streak_native@value | 3 | measure | estimate | ESTIMATE(fixed N) | 1 | 40 | -10.0 +- 7.8 | pp (1v3 win rate vs value) | 67.5% / -15% | 0.202 |  | -0.35 +- 0.27 | 0.25 / 1.00 | first_city_round -0.19+-0.28; first_settle_round -0.67+-0.45; knights_held_end +0.17+-0.13; knights_played -0.35+-0.24; largest_army -0.12+-0.08; longest_road - | 0.03 | measurement recorded |
| trades | acq_calib_reject_streak | row |  | human | DEFERRED |  |  |  |  |  |  |  |  |  |  | 0.00 | deferred to human testing (no games) |
| trades | acq_flow_fit | row |  | gate | PASS |  |  |  |  |  |  |  |  |  |  | 0.00 | pipeline / pairing verified |
| trades | acq_floor_selfplay | 1 | new | screen | SHELVE(no gain; conditional power < 0.1) [stopped early, promise not met] | 2 | 960 | -1.2 +- 1.6 | pp (per seat, 2v2) | 51.2% / -2% | 0.781 | 1* | -0.10 +- 0.07 | 1.00 / 0.00 |  | 2.33 | on ice (default unchanged) |
| trades | acq_floor_vrule@value | 1 | new | screen | SHELVE(no gain; conditional power < 0.1) [stopped early, promise not met] | 2 | 800 | -0.1 +- 1.0 | pp (1v3 win rate vs value, vs value-rule responders) | 85.2% / -0% | 0.548 | 1* | +0.00 +- 0.03 | 0.09 / 0.62 | first_city_round +0.01+-0.08; first_settle_round +0.10+-0.08; knights_held_end +0.01+-0.02; knights_played +0.02+-0.05; largest_army -0.01+-0.01; longest_road - | 1.53 | on ice (default unchanged) |
| trades | t3_trades0_vrule@alphabeta | 0 | measure | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| diversification | t1_openings@value | pips_diversity | new | screen | SHELVE(no gain; conditional power < 0.1) [stopped early, promise not met] | 2 | 800 | -1.5 +- 2.1 | pp (1v3 win rate vs value) | 63.5% / -2% | 0.764 | 1* | -0.06 +- 0.09 | 0.35 / 0.80 | first_city_round +1.48+-0.29; first_settle_round -1.65+-0.25; knights_held_end -0.03+-0.02; knights_played -0.18+-0.08; largest_army -0.04+-0.02; longest_road + | 1.45 | on ice (default unchanged) |
| diversification | t1_openings@value | standin_book | new | screen | SHELVE(no gain; conditional power < 0.1) [stopped early, promise not met] | 2 | 800 | -1.1 +- 2.0 | pp (1v3 win rate vs value) | 63.5% / -2% | 0.716 | 1* | -0.10 +- 0.08 | 0.31 / 0.80 | first_city_round +1.27+-0.30; first_settle_round -1.13+-0.27; knights_held_end -0.08+-0.02; knights_played -0.28+-0.08; largest_army -0.05+-0.02; longest_road + | 1.45 | on ice (default unchanged) |
| diversification | t1_openings@value | conversion | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 2 | 800 | +0.4 +- 2.2 | pp (1v3 win rate vs value) | 63.5% / +1% | 0.431 | 1* | +0.01 +- 0.09 | 0.38 / 0.88 | first_city_round +1.05+-0.30; first_settle_round -1.37+-0.30; knights_held_end -0.04+-0.02; knights_played -0.23+-0.09; largest_army -0.04+-0.02; longest_road + | 1.45 | on ice (default unchanged) |
| diversification | t1_openings@value | setup_pick | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 4 | 1600 | +1.4 +- 1.4 | pp (1v3 win rate vs value) | 62.6% / +2% | 0.149 | 1* | +0.07 +- 0.06 | 0.30 / 0.70 | first_city_round +1.10+-0.18; first_settle_round -0.78+-0.18; knights_held_end -0.03+-0.02; knights_played -0.18+-0.05; largest_army -0.03+-0.01; longest_road + | 1.45 | on ice (default unchanged) |
| diversification | t1_openings@vf | pips_diversity | new | screen | SHELVE(no gain; conditional power < 0.1) [stopped early, promise not met] | 2 | 800 | -1.1 +- 1.9 | pp (1v3 win rate vs vf) | 27.1% / -4% | 0.727 | 1* | -0.10 +- 0.10 | 0.28 / 0.79 | first_city_round +1.41+-0.36; first_settle_round -1.08+-0.28; knights_held_end -0.01+-0.01; knights_played -0.26+-0.08; largest_army -0.07+-0.02; longest_road + | 1.21 | on ice (default unchanged) |
| diversification | t1_openings@vf | standin_book | new | screen | ADOPT [stopped early] | 4 | 1600 | +3.8 +- 1.4 | pp (1v3 win rate vs vf) | 25.9% / +14% | 0.00567 | 0.0738* | +0.11 +- 0.07 | 0.31 / 0.85 | first_city_round +1.44+-0.29; first_settle_round -1.42+-0.21; knights_held_end -0.01+-0.01; knights_played -0.38+-0.06; largest_army -0.08+-0.02; longest_road + | 1.21 | league gate (--formats 4p2v2,3p1v2; power 0.38 at a 0.53 share, 0.86 at 0.55; ~4 h) (bundle the area's ADOPTs first) |
| diversification | t1_openings@vf | conversion | new | screen | SHELVE(no gain; conditional power < 0.1) [stopped early, promise not met] | 2 | 800 | -0.9 +- 1.9 | pp (1v3 win rate vs vf) | 27.1% / -3% | 0.677 | 1* | -0.03 +- 0.10 | 0.29 / 0.89 | first_city_round +0.27+-0.41; first_settle_round -0.96+-0.32; knights_held_end -0.02+-0.01; knights_played -0.40+-0.09; largest_army -0.09+-0.02; longest_road + | 1.21 | on ice (default unchanged) |
| diversification | t1_openings@vf | setup_pick | new | screen | SHELVE(no gain; conditional power < 0.1) [stopped early, promise not met] | 2 | 800 | -1.1 +- 1.7 | pp (1v3 win rate vs vf) | 27.1% / -4% | 0.745 | 1* | -0.06 +- 0.09 | 0.23 / 0.71 | first_city_round +0.64+-0.31; first_settle_round -0.33+-0.26; knights_held_end -0.01+-0.01; knights_played -0.26+-0.08; largest_army -0.07+-0.02; longest_road + | 1.21 | on ice (default unchanged) |
| diversification | demand_flat_pyeval@vf | 1/1/1/1/1 | new | screen | REJECT(worse) [stopped early, promise not met] | 2 | 800 | -3.8 +- 1.9 | pp (1v3 win rate vs vf) | 27.1% / -14% | 0.977 | 1* | -0.38 +- 0.10 | 0.28 / 0.94 | first_city_round +1.55+-0.39; first_settle_round -0.85+-0.29; knights_held_end +0.01+-0.02; knights_played -0.11+-0.08; largest_army -0.06+-0.02; longest_road + | 1.17 | on ice (default unchanged) |
| diversification | demand_mild_pyeval@vf | 1.15/1.15/0.9/1/1 | new | screen | NOT TRIGGERED |  |  |  |  |  |  |  |  |  |  | 0.00 | no overshoot of setup_distinct |
| diversification | ports_conversion_cost@value | cand | new | screen | SHELVE(with bundle) |  |  |  |  |  |  |  |  |  |  | 0.00 | on ice (default unchanged) |
| diversification | ports_conversion_reach@value | cand | new | screen | NOT TRIGGERED |  |  |  |  |  |  |  |  |  |  | 0.00 | parent SHELVE(with bundle) |
| diversification | div_lr_bundle | cand | new | screen | SHELVE(too small to prove; conditional power < 0.1) [stopped early] | 2 | 800 | +0.9 +- 1.7 | pp (1v3 win rate vs value) | 63.5% / +1% | 0.303 | 1* | +0.05 +- 0.06 | 0.23 / 0.94 | setup_distinct +0.00+-0.00; first_settle_round +0.26+-0.15; first_city_round +0.14+-0.12; settle_before_city -0.00+-0.01; port_settled -0.01+-0.02; share_4to1 + | 0.31 | on ice with its pieces (conv=1, paths=1): bundle first, no 2x2 |
| diversification | div_lr_bundle-no-conv | cand | knockout | knockout | NOT TRIGGERED |  |  |  |  |  |  |  |  |  |  | 0.00 | after div_lr_bundle: condition not met |
| diversification | div_lr_bundle-no-paths | cand | knockout | knockout | NOT TRIGGERED |  |  |  |  |  |  |  |  |  |  | 0.00 | after div_lr_bundle: condition not met |
| diversification | t3_openings@alphabeta | pips_diversity | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| diversification | t3_openings@alphabeta | standin_book | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| diversification | t3_openings@alphabeta | setup_pick | new | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| ports | ports_gate_f1@vf | 1 | measure | estimate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| ports | ports_gate_f2@vf | 0.8 | measure | estimate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| ports | ports_gate_flow@vf | 0.25 | measure | estimate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| ports | ports_gate_mode3_trigger | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate_f1@vf |
| ports | ports_gate_f1_mode3@vf | 3 | measure | estimate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate_mode3_trigger |
| ports | ports_gate | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate_f1@vf |
| ports | ports_gate_pick_f1 | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate |
| ports | ports_gate_pick_f2 | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate |
| ports | ports_gate_pick_flow | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate |
| ports | ports_gate_pick_f1m3 | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate |
| ports | ports_best_f1@vf | 1 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate_pick_f1 |
| ports | ports_best_f2@vf | 0.8 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate_pick_f2 |
| ports | ports_best_flow@vf | 0.25 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate_pick_flow |
| ports | ports_best_f1m3@vf | 3 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after ports_gate_pick_f1m3 |
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
| robber | shadow_robber_step5 | row |  | gate | queued |  |  |  |  |  |  |  |  |  |  | 0.00 | queued |
| robber | robber_gate_r1a | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after shadow_robber_step5 |
| robber | robber_gate_kick_trigger | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after shadow_robber_step5 |
| robber | robber_gate_r1b | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after shadow_robber_step5 |
| robber | robber_gate_r1c | row |  | gate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after shadow_robber_step5 |
| robber | robber_la_bundle | cand | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after robber_gate_r1a |
| robber | robber_la_bundle-no-robber_corr | cand | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after robber_la_bundle |
| robber | robber_la_bundle-no-paths | cand | knockout | knockout | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after robber_la_bundle |
| robber | robber_persistence | 1 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after robber_gate_r1a |
| robber | knight_kick | 0.95 | new | screen | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after robber_gate_kick_trigger |
| robber | robber_retal_w | row |  | human | DEFERRED |  |  |  |  |  |  |  |  |  |  | 0.00 | deferred to human testing (no games) |
| robber | t3_blockw0_pyeval@alphabeta | 0 | knockout | confirm | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | confirmation waits for tier t1 to complete (Holm family) |
| counting | devbelief_smoke@value | 1 | measure | estimate | WAITING |  |  |  |  |  |  |  |  |  |  | 0.00 | after devbelief_gate |
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
| 1 | ports | ports_gate_f1@vf | ELIGIBLE | 0.55 | 0.0156 |  |
| 2 | ports | ports_gate_f2@vf | ELIGIBLE | 0.55 | 0.0146 |  |
| 3 | ports | ports_gate_flow@vf | ELIGIBLE | 0.19 | 0.0136 |  |
| 4 | robber | shadow_robber | ELIGIBLE | 1.30 | 0.0039 |  |
| 5 | robber | t2_exposure0_pyeval@value | ELIGIBLE | 1.50 | 0.0034 |  |
| 6 | robber | t1_blockw0_pyeval@value | ELIGIBLE | 2.99 | 0.0034 |  |
| 7 | robber | t1_blockw0_pyeval@vf | ELIGIBLE | 1.80 | 0.0034 |  |
| 8 | robber | shadow_robber_step5 | ELIGIBLE | 0.50 | 0.0033 |  |
| 9 | politics | t2_max_slack0_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 10 | politics | t2_coal_scale2_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 11 | politics | t2_feed_leader_off_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 12 | politics | t2_late_drop0_vrule@value | ELIGIBLE | 1.67 | 0.0002 |  |
| 13 | politics | counters_selfplay | ELIGIBLE | 3.60 | 0.0002 |  |
| 14 | politics | respond_lookahead_selfplay | ELIGIBLE | 3.60 | 0.0002 |  |
| 15 | other | t1_search_vs_heur@value | ELIGIBLE | 1.57 | 0.0001 |  |
| 16 | other | t1_search_vs_heur@vf | ELIGIBLE | 1.02 | 0.0001 |  |
| 17 | other | t1_depth2@value | ELIGIBLE | 1.64 | 0.0001 |  |
| 18 | other | t2x_beam2@value | ELIGIBLE | 0.48 | 0.0001 |  |
| 19 | other | t2x_expand4@value | ELIGIBLE | 0.48 | 0.0001 |  |
| 20 | other | shadow_paths | ELIGIBLE | 0.15 | 0.0001 |  |

Waiting / blocked: acq_breadth_confirm@value (WAITING: confirmation waits for tier t1 to complete (Holm family)); t3_trades0_vrule@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); t3_openings@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); ports_gate_mode3_trigger (WAITING: after ports_gate_f1@vf); ports_gate_f1_mode3@vf (WAITING: after ports_gate_mode3_trigger); ports_gate (WAITING: after ports_gate_f1@vf); ports_gate_pick_f1 (WAITING: after ports_gate); ports_gate_pick_f2 (WAITING: after ports_gate); ports_gate_pick_flow (WAITING: after ports_gate); ports_gate_pick_f1m3 (WAITING: after ports_gate); ports_best_f1@vf (WAITING: after ports_gate_pick_f1); ports_best_f2@vf (WAITING: after ports_gate_pick_f2); ports_best_flow@vf (WAITING: after ports_gate_pick_flow); ports_best_f1m3@vf (WAITING: after ports_gate_pick_f1m3); t2_danger_mult_off@value (WAITING: shadow shadow_robber); t2_block_need0@value (WAITING: shadow shadow_robber); t2_steal_factor_off@value (WAITING: shadow shadow_robber); t2_rob_break_off@value (WAITING: shadow shadow_robber); t2_turns_half2@value (WAITING: shadow shadow_robber); t2_knight03@value (WAITING: shadow shadow_robber); robber_gate_r1a (WAITING: after shadow_robber_step5); robber_gate_kick_trigger (WAITING: after shadow_robber_step5); robber_gate_r1b (WAITING: after shadow_robber_step5); robber_gate_r1c (WAITING: after shadow_robber_step5); robber_la_bundle (WAITING: after robber_gate_r1a); robber_la_bundle-no-robber_corr (WAITING: after robber_la_bundle); robber_la_bundle-no-paths (WAITING: after robber_la_bundle); robber_persistence (WAITING: after robber_gate_r1a); knight_kick (WAITING: after robber_gate_kick_trigger); t3_blockw0_pyeval@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); devbelief_smoke@value (WAITING: after devbelief_gate); paths_main@value (WAITING: shadow shadow_paths); paths_main@vf (WAITING: shadow shadow_paths); t3_search_vs_heur@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family)); t3_depth2@alphabeta (WAITING: confirmation waits for tier t1 to complete (Holm family))

## Per area

| area | rows | final | open | ADOPT | on ice (SHELVE/REJECT) | NOOP | FAILED | deferred | CPU-h spent |
|---|---|---|---|---|---|---|---|---|---|
| harness | 7 | 7 | 0 | 0 | 2 | 0 | 0 | 0 | 0.92 |
| trades | 12 | 10 | 0 | 0 | 3 | 0 | 0 | 1 | 8.79 |
| diversification | 10 | 9 | 0 | 1 | 10 | 0 | 0 | 0 | 4.14 |
| ports | 15 | 1 | 3 | 0 | 0 | 0 | 0 | 1 | 0.00 |
| robber | 22 | 1 | 5 | 0 | 0 | 0 | 0 | 1 | 0.00 |
| counting | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| politics | 6 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| other | 10 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0.00 |

## Bundle proposals (a human commits them; then the league gate)

- **diversification**: t1_openings@vf=standin_book
  `python3 scripts/league.py gate --candidate-commit <sha> --candidate-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic,tune=openings.policy:standin_book" --formats 4p2v2,3p1v2`

## Deferred to human testing (paste into docs/ABLATIONS.md, Politics rule)

| term | screen | result | status |
|---|---|---|---|
| opponent_model.REJECT_STREAK | none (no bot harness reacts) | n/a | deferred |
| winpaths.SPOT_LEADER | none (no bot harness reacts) | n/a | deferred |
| robber_eval.RETAL_W | none (no bot harness reacts) | n/a | deferred |

CRN pilot: discordant share 0.318 (off) -> 0.215 (dice), ratio 0.68; A/A identical -> PASS: rows with crn auto use --crn dice
