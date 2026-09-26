# Priority areas: designs and plan (2026-09-26)

**Regrouped after the designs (user, 2026-09-26):**

| order | area | linked VP path |
|---|---|---|
| 1 | trading | |
| 2 | diversification / expansion (conversion cost, opening diversity, expansion pace) | Longest Road |
| 3 | ports (access) | |
| 4 | robber | Largest Army |
| 5 | card counting | |

Linked pieces are tested as bundles first, then knocked out one at a time
(docs/ABLATIONS.md "Regrouping").  The rest of this file is the design
output as written, with the old area names.

The design workflow's output for the user's four priority areas (trading, ports, robber, card counting) plus the budgeted test queue.
For each area, one agent mapped the code and measured the proof logs; two critics (Catan strategy, engineering) attacked the design; a reviser folded the fixes in; a planner merged everything into one queue.
Full detail (every formula, plug-in point and test): `docs/designs/priority_areas_2026-09-26.json`.

## Budgeted test queue (early stopping, priority queue with preemption cost, batching, shelve / fallback)

This is the revised design. I accepted almost every point from both critiques after re-checking the code; the one factual correction is that only half of the 3-player self-play patterns are 1v2. The queue lives entirely in scripts/. The one exception is an optional harness keyword in catanbot/bench/catanatron_adapter.py. So the default bot stays byte-identical, and neither static_value nor the placement score (mirrored in C++) is touched. None of this can run yet: a pre-registered proof run (T9, counted mode vs sameturn) is using 3 of the 4 cores, and the new proof guard blocks the queue until it finishes.

Features, in implementation order:
- **F1 `queue.seqtest`:** a sequential verdict engine with fixed K=5 look tables.
  - Designs by polarity: new-idea screen, knockout (REMOVE / KEEP), fixed-N estimate for headroom and A/A rows, pre-registered fixed-N politics screen, confirm, legacy.
  - NOOP becomes FAILED(inert-harness) when the mechanic never fired.
  - A 'promise not met' label.
  - Verdicts are carried as `{kind: stop}` JSONL records.
  - A newly found bug is fixed: ablate_catanatron's Campaign ignores stop records.
- **F2 `queue.intake`:** before any game is played:
  - required area / polarity / promise_pp fields;
  - routing lint: counting only in counted mode vs Catanatron, trades only with --trades or in politics self-play, no self-play on the Python evaluator;
  - a power check at the promised effect, from a simulated grid (D 0.40: +2pp 0.23, +3pp 0.48, +4pp 0.74);
  - headroom rows run first in their area;
  - an `--explain --simulate` dry run.
- **F3 `queue.scheduler`:**
  - Order is area first (harness, trades, ports, robber, counting, politics, other), then priority, then plan order.
  - Cost is counted in CPU seconds (getrusage) and is used only to decide preemption.
  - Chunks are time slices, so in-flight games finish.
  - Global proof guard, and a load gate measured after the queue drains its own work.
  - Fallbacks come in three kinds (simpler, milder, stronger), with milder allowed only on a mechanism overshoot.
  - Tier-3 confirmations use Holm within the tier; destinations depend on the area.
- **F4 `queue.snapshot`:** pinned code snapshots built from an allowlist, with one epoch per row. Look-synchronous groups are dropped and canary aliasing is deferred.
- **F5 `queue.mechanics`:** mechanism metrics per game: ports, trade rates, robber, steals, Monopoly, discards.
- **F6 `queue.crn`:** common random numbers. Dice are keyed by roll index, and counted-mode samples are seeded per decision. This needs a pilot first.
- **F7 `queue.cv_default`:** a control variate against a disjoint default-arm pool. With 8,000 pool games, false ADOPT is unchanged (0.022 to 0.023) and power at +3pp for D 0.40 rises from 0.49 to 0.68.
- **F8 `queue.report`:** labelled units, and the politics rule applied mechanically with an explicit tier family.
- **F9 `queue.selfplay_rows`:** limited to the two politics screens at the pre-registered 1,200 games, plus 1v2 no-harm checks for trade ADOPTs.

At null, the 27 enabled rows cost about 24 CPU-h (about 8 h wall on 3 cores), against 38.5 CPU-h for the legacy full-length plan.

| priority | feature | switch (off by default) |
|---|---|---|
| 1 | `queue.seqtest`: Sequential verdict engine by polarity and design (screen / knockout / estimate / politics / confirm / legacy), inert-harness detection, verdicts carried as JSONL stop records | Plan field `design` (screen / knockout / estimate / politics / confirm / legacy) and `polarity`, read only by scripts/queue.py. Run directly, campaign.py and ab |
| 2 | `queue.intake`: Row intake: area / polarity / promise fields, routing lint, feasibility at the promised effect, headroom-first gates, explain/simulate dry run | Only scripts/queue.py applies intake. `--no-intake` bypasses the power and headroom checks but still prints the lint. The new fields are optional for campaign.p |
| 3 | `queue.scheduler`: Area-first priority queue with a CPU-cost preemption test, time-sliced chunks, proof guard, fallback kinds and area-specific confirmations | A new entry point. campaign.py behaves as today, except that it skips kinds it cannot run and accepts the new optional keys. No tuning.py entry: the bot is unto |
| 4 | `queue.snapshot`: Pinned code snapshots from an allowlist (one epoch per row; explicit bumps at batch boundaries) | Queue option --snapshot, on by default in queue mode. campaign.py and direct ablate runs are unchanged. |
| 5 | `queue.mechanics`: Per-game mechanism metrics as secondary endpoints (steer fallbacks and inert detection; never ADOPT) | --mech flag, off by default; the queue turns it on. Records without `mech` are simply excluded from the mechanism summaries. |
| 6 | `queue.crn`: Common random numbers in the harness: dice keyed by roll index; per-decision seeded samples in counted mode (pilot, then queue default if discordance drops at least 25%) | --crn off by default (no patch, arm keys unchanged). seeded_samples=False by default (absent from the adapter dict, keys unchanged).  Neither is a tuning.py tun |
| 7 | `queue.cv_default`: Control variate with an external default-arm pool (pre-declared for high-divergence rows) | Plan field `estimator: paired` (default) or `cv`. Pool rows exist only where they are declared. No tuning.py entry. |
| 8 | `queue.report`: Ledger-driven QUEUE.md with labelled units, knockout labels, mechanism deltas, bundle proposals and the politics rule (explicit tier family) | Written only by scripts/queue.py. SUMMARY.md is byte-identical when no ledger exists. |
| 9 | `queue.selfplay_rows`: Self-play rows limited to the politics self-play screens (fixed 1,200 games) and 1v2 no-harm checks for trade ADOPTs | Plan field `kind: selfplay` (the default kind is catanatron). These rows run only under scripts/queue.py, and campaign.py skips them. |

## Trading: get what we need from the cheapest source (bank 4:1, ports 3:1/2:1, other players)

Revised after both critiques. I re-checked every disputed fact in the code, and the critiques are right on all of them except two small points.

**Unchanged findings.** Against Catanatron the bank/port side is close to its per-turn optimum. There are 0 impossible 4:1 trades. Only 2% of our 4:1 trades had a better give payable from the hand. 89-91% of the cards we buy are spent the same turn. The 4:1 gap is structural: we own fewer ports and trade away resources we produce in excess (wheat and ore). That gap belongs to the ports area.

**What changed**
1. **acq.progress (priority 1) is rebuilt and cut to one cheap arm.** The earlier draft had four problems:
   - The horizon was wrong: the player about to roll got k = 4 instead of 1, because `discard.opponent_rolls_before_my_turn` returns n-1 for the current player in any phase.
   - P7 left out the roller's own 7 (the engine puts the roller in the discard queue).
   - Our seat was valued at different horizons mid-turn and after END_TURN.
   - The Stage 0 evidence came from a different formula. The critics' re-runs of the specified mode 1 change 8-10% of decisions. Most of those are bank trade to END_TURN at 7 or fewer cards, and a quarter to a third of the changes above 7 cards end the turn holding more than 7.

   The new version:
   - applies to our own seat only;
   - credits bank/port conversions, capped by the bank;
   - credits production as E[min(Y_r, missing_r)] from the exact k-roll production distribution, not expected cards;
   - uses k = n for our seat at every node of our turn;
   - gives a hand above 7 cards no correction (this replaces the keep factor);
   - drops the pending-port credit.

   It plugs into one shared correction hook, which reuses winpaths.PathsEvaluator with winpaths.py unedited. That hook is used at all four evaluator sites plus the native guard and reduced_config.

   Test plan: a real-code Stage 0 audit with pre-registered pass criteria. Then a 200-seed mechanism check and A/A. Then one arm vs value that reuses the t1_baseline_aa default arm, with a futility look at 1,000 seeds (win delta and VP delta both 0 or below means stop). Nothing runs beside the proof except Stage-0-size audits. The expected effect (about 0.5-1.5 pp) is below the minimum detectable effect (about 3.5-4 pp), so SHELVE is the likely result. It is still run because it is cheap (about 0.5 h) and it is the only trading feature measurable against Catanatron.
2. **acq.offers is replaced by acq.breadth (priority 2).** The route table, density retargeting, ask-for-2, the port floor (0 measured reach) and propose_margin (a cancelled offer is free in the engine) are dropped. What remains is 5 proposal candidates per node (an existing registry candidate) plus injected mixed-give 2-for-1 offers. The injected offers are appended in `_candidates`, count against max_trades and trade_cap, and are ranked by p_any exactly as `_trade_outcomes` computes it. Screened in self-play on the same seeds. ablate.py pairs candidate and default seats inside each game, so there is no default arm to share there.
3. **New: acq.flow (priority 3), a trade-flow port value handed to the ports area.** We make 5.0-5.2 4:1 trades a game, 3.0 of them after global turn 40. By resource: wheat 1.65-1.78 a game, ore about 1.2. So a generic port held all game saves up to about 5 cards and a 2:1 wheat port about 3.6-3.8 cards, against the static value's flat +0.2 (0.02 VP). This replaces port_savings.
4. **acq.calib (priority 4) is now a calibration and compute item under the politics rule.** It first passes a mechanism gate (Brier score, offers per game, CPU), then gets exactly one 1,200-game self-play screen. Logits are computed at answer time before note_accept. There is no value-rule row. The fallback is a rejection-streak rule that goes straight to the human-testing list.
5. **Shelved by measured reach, no code:**
   - port-aware Monopoly, Year of Plenty and discards (0.036 and at most 0.09 changed decisions a game);
   - acq.mono_guard, kept as a spec note with a corrected formula.

**Cross-cutting**
- **Code freeze.** The code fingerprint hashes all of catanbot/, so every area's code and the SPSA/factorial agent's catanbot/ files must land before the first campaign default arm is played.
- **Harness counters.** Per-game counters (4:1 trades, cards to the bank, discards, steals, END_TURN with more than 7 cards, proposals, accepts) must be added to the harnesses first.
- **Registration.** All tunables register through acquisition.register_tunables, one import line in tuning.py.

| priority | feature | switch (off by default) |
|---|---|---|
| 1 | `acq.progress`: Acquisition-aware hand value for our own seat: bank/port conversions plus the chance of producing the missing cards before our next build, off above 7 cards | - search.acq (kind=search, int, default 0 = off; 1 = conversions only, a Stage 0 diagnostic; 2 = conversions + production, the tested arm; spec key acq). - sear |
| 2 | `acq.breadth`: Wider player-offer search: 5 proposal candidates per node and injected mixed-give 2-for-1 offers | - (A) search.trade_proposals (existing; kind=search; default 3; candidate 5). - (B) search.acq_shapes (kind=search, int, default 0 = off; 1 = inject mixed-give  |
| 3 | `acq.flow`: Trade-flow port value (a data product and function for the ports area) | Owned by the ports area, for example ports.FLOW_KAPPA (kind=weight, default 0.0 = off; read only by its provider). In this area it is a pure function that the d |
| 4 | `acq.calib`: Online calibration of P(accept) per opponent (a calibration and compute item under the politics rule) | - opponent_model.CALIB_RATE (kind=weight, default 0.0 = off, candidates [0.5, 1.0]). - opponent_model.CALIB_PRIOR (weight, default 0.0, candidate -1.0). - Fallb |
| 5 | `acq.minor_routes`: Port-aware Monopoly, Year of Plenty and discards (SHELVED by measured reach, no code) | None; nothing is built. If it is ever built: devcards.PORT_AWARE (kind=flag, default False). |
| 6 | `acq.mono_guard`: Do not hold resources into a likely opponent Monopoly (SHELVED; spec note only) | None; nothing is built. When built: search.acq_mono (float, default 0.0, read only when search.acq = 2). |

## Ports: when a port spot is worth it (production given up, roads, blocking race with a simple leader check)

**Revised verdict: ports are a small lever. This area gets one cheap mechanism gate and at most one win-rate row.**

**The port gap is an expansion-pace gap.**
- Our first post-setup settlement comes at median turn 40-45; Catanatron's comes at 29-32 (turns counted over all seats).
- Our first city comes at median turn 26-36 vs 35-37 for Catanatron. Our bot plays city-first.
- Our first port is about 9-10 turns later, which matches that pace gap.
- Port share of post-setup settlements is about equal (41-45% vs 43-45%). For the first expansion settlement ours is 0 to 6.5 points lower (R1 0.32 vs 0.38, about 3.5 s.e.).

**The whole port-efficiency gap is worth at most about 1.6 pp.**
- Cards saved vs trading everything at 4:1, per game: ours vs Catanatron 2.5 vs 3.6 (T1), 2.3 vs 3.6 (T2), 3.3 vs 3.1 (R1).
- 1.3 cards per game is about 0.6 pip, since 1 pip is about 2.2 cards per game. At +0.25 to +2.7 pp per pip, closing the gap completely is worth at most about 1.6 pp.
- 2,000 paired seeds detect about 3.1-3.6 pp. So no port-valuation feature can be proven within budget. By the user's rule, each one is gated and then shelved unless the mechanism turns out bigger than expected.

**Calibration corrected (critique accepted).**
- Our bot gives 0.06-0.12 cards per roll to the bank at 4:1 even with little surplus. Fit: obs = 0.060 + 0.87 x model (T1), 0.116 + 0.63 x model (R1). Catanatron fits near zero intercept.
- New conversion model: conv_r = A0 x prod_r / I + B x s_r.
- With it, today's 3:1 spot bonus (1.0) is about right: 1.2-1.3 calibrated, including the complement factor.
- The static 3:1 term is about 4x too small: 0.2 today vs 0.76-0.79 points calibrated. It also counts every 3:1 port, although a second one is worth nothing.
- Today's 2:1 constants are about 2x the calibrated card value (for example 2:1 wheat 2.3 vs 1.3 on T1).
- Our bank surplus is wheat, and on T1 also ore, so 2:1 wheat is our most valuable 2:1. The first draft's claim that 2:1 ore/wheat is overvalued "because we need them" was wrong.

**Decision rates measured on the real search bot, not the heuristic prior.**
- P1-rev: 3.7% of setup decisions (n=320) and 5.3% of settlement decisions with 2 or more legal spots (n=150); port settlements 54 -> 55.
- P3' (static 3:1 = 0.8, counted once): 1.9% / 4.7%; 3:1 picks 21 -> 27 of 150.
- The old road-target race (P2) changed only 4.8% of road decisions (critic, A/A = 0). The search picks moves by static value among its top 8 candidates. The race clock also gave the next seat an 80% win in a spot we could buy right now, and it ignored bank conversion. **P2 is dropped.**

**Revised features (all off by default):**
- **F1 ports.surplus_value:** a calibrated port gain for the spot score and static_value, with modes full / static-only / 3:1-only. Python evaluator for the screen; hard-coded constexpr in C++ only on ADOPT.
- **F2 ports.constants:** the six literals as tunables, plus a count-3:1-once switch. Candidate P3': static 3:1 = 0.8, counted once. Also the 'ports' group for the other agent's SPSA/factorial tools.
- **F3 ports.advice:** a zero-game "is this port worth it / who can block it / will they / is someone leading" line in the CLI advice. It must not go in explain_action, which the search runs for every root action at every decision.
- **F4 ports.spot_want:** "will they block" (rival need) and the leader check, placed inside winpaths paths_spots where they can affect decisions. Screened only if Stage 5 adopts paths_spots. The leader knob is deferred to human testing at once (no harness opponent conditions on the leader).
- **F5 ports.monopoly_liquidity:** a port-aware Monopoly value. It is small and changes priors only; screen it first and hand the reverse direction to card counting.

**Plan.**
1. Zero-game search-bot decision screens.
2. After the proof run: one 300-seed mechanism gate vs 3.2.1 vf with a shared base arm and two cells (F1, F2 P3'), Python evaluator, about 0.3 CPU-hours.
   - Pre-registered PASS: cards saved +1.1 per game or more, and settlements not lower by 0.15 or more. The expected outcome is FAIL -> SHELVE.
   - On PASS: one 2,000-seed row inside a frozen, shared-seed Python-evaluator batch.
3. The CPU saved goes to the expansion-pace question, which belongs to another area.

| priority | feature | switch (off by default) |
|---|---|---|
| 1 | `ports.surplus_value`: F1 (P1-rev): port value from a calibrated conversion model, in the spot score and static_value | placement.PORT_MODEL: int, parse _parse_int, default 0 (today's constants, byte-identical); candidates 1, 2, 3. Sub-knobs are read only when PORT_MODEL > 0: - p |
| 2 | `ports.constants`: F2 (P3'): the port constants as tunables, plus a count-3:1-once switch | Seven weight tunables whose defaults equal today's behaviour (byte-identical): - placement.PORT_SPOT_GENERIC 1.0 - placement.PORT_SPOT_2TO1_BASE 0.5 - placement |
| 3 | `ports.advice`: F3: 'is this port worth it, and can they block it' advice line (zero games) | portvalue.ADVICE: int, default 0 (off); registered in tuning.py with needs_python_evaluator=False. Display only: it never changes a decision. |
| 4 | `ports.spot_want`: F4: 'will they block' (rival need) and the leader check, inside winpaths' spot race | winpaths.SPOT_WANT: int 0 (candidate 1). winpaths.SPOT_WANT_FLOOR 0.3 (0.1, 0.5). winpaths.SPOT_LEADER: int 0 (1, 2), a politics knob. winpaths.SPOT_LEADER_DENY |
| 5 | `ports.monopoly_liquidity`: F5: port-aware Monopoly value (small; screen first; hand-off) | devcards.MONO_PORT: int 0 (candidates 1, 2). devcards.MONO_PORT_W 0.5. needs_python_evaluator=False (monopoly_value is not read by static_value). |

## Robber dynamics: block production and steal the right resources

This is the design revised after the two critiques. It stays read-only: I changed no repository files and ran no git. I re-checked every disputed fact in the code and ran one more niced replay of archived proof games (1,000 games, about 16 s of CPU; scratch scripts robstats4.py and robstats4b.py). The critiques were right on almost every point.

Verified in code:
- should_play_knight kicks only when production_blocked >= 3.0 (robber.py L378).
- _decide_counted calls tracker.determinize before bot.decide (catanatron_adapter.py L1488-1489), so registry overrides never reach determinization.
- _deal_devs re-deals everyone jointly up to 20 times, falls back to the last deal, and sets dev_known=True on dealt seats (public_belief.py L255-305).
- ablate_catanatron stores only a trace hash per game.
- The steal-swap in R4 breaks the per-resource totals.

New measurements:
- A block's realised share E[min(L,t)]/E[L] follows the geometric curve 1-(1-1/4.34)^t within 0.02. The linear curve was off by up to 0.29.
- Both bots kick almost always when they are still blocked and hold a knight: 822 of 825 blocked-with-knight turns. Only 2 of those 825 had a 1-2 pip block.
- At 10.1% of our clear-leader robber decisions the leader holds dev cards, and 93% of them hold no knight. In counted mode, pool dealing would call most of them kickers.
- Neither bot population retaliates. After being robbed, Catanatron steals from the robber 22.7% of the time and we do 35.6%, against a uniform 33%.

Revised plan:
- Priority 1, tooling (zero games): a shadow harness plus observation-only robber counters. They go in the adapter stats and in play_paired_game, not in ablate.py or tune_joint.py. This replaces the old Stage A and screens the five prior-only campaign rows before they cost games.
- Priority 2, R1a persistence: the only feature that gets its own A/B test. Changes: a geometric keep curve, a kick gate matching the knight rule, KICK_LAMBDA 0.95, and sampled dev cards never counted as knights in counted mode. It no longer refactors the C++-mirrored steal_exposure_fast. The cache is keyed on every seat's buildings, and a fast path keeps robber_corr=1 bit-identical when all corrections are zero. There is an optional retaliation weight, RETAL_W, defaulting to 0 (politics rule). Testing: self-play, 3 x 800 games on disjoint seeds, O'Brien-Fleming boundaries (3.71, 2.51, 1.99), paired VP difference as co-primary. Against Catanatron it is only a harm screen, run only if self-play adopts or shows at least +1.5 pp. Expect SHELVE: power is about 31% at +1.5 pp.
- Priority 3, R3 (blocked, holding dev cards, played none): handed to the card-counting area as one likelihood term. Fixes: exact per-player posterior dealing, a mixture likelihood with a hoarder prior, a cap on the event count, fitting on opponent seats only, and a gain measured against a held-age/per-type hazard baseline. It also fixes the blocker: determinization moves inside ParamBot's override scope.
- Priority 4, R1b knight insurance: redesigned. The target model now comes from the opponent model's robber habits, and the fallback prior uses pips and the leader, never hand size. A blocked player with two or more knights gets credit for the spare. The term is centred on its mean, so it moves knight value toward exposed positions instead of raising it. That keeps it separate from KNIGHT_VALUE. A standalone test would need about 10^5 seeds, so it becomes an SPSA knob.
- Priority 5, R1c block-duration knob (a critique's idea): the leaf treats every block as permanent. Shadow first, then an SPSA knob.
- R2 is not built: the normalisation bug is confirmed and the effect is ±1 pp of uncertain sign. R4 is dropped.

Every bot feature is off by default behind registered tunables, and none touches the C++-mirrored static_value or placement score.

| priority | feature | switch (off by default) |
|---|---|---|
| 1 | `robber.shadow_metrics`: Zero-game robber shadow and observation-only robber counters (tooling; gates everything else) | Not a bot feature: no tunable, because the bot is unchanged. The counters are observation-only and consume no RNG, and a test proves game results are identical  |
| 2 | `robber.persistence`: Robber persistence leaf correction: blocks on a would-kick knight holder are mostly temporary (R1a) | search.robber_corr: kind search, int, default 0 (spec key robber_corr). With 0 the wrapper is never built, so the default is byte-identical. Module weights, all |
| 3 | `robber.knight_evidence`: Blocked-no-play likelihood for held dev cards: secret-VP leader and knight holders (R3, handed to the card-counting area) | Weights (not flags: a flag tunable cannot be switched on from a False default): - public_belief.KNIGHT_EVIDENCE: int 0/1, default 0; - public_belief.KNIGHT_HOLD |
| 4 | `robber.knight_insurance`: Centred knight-insurance leaf correction with opponent-habit targeting (R1b; an SPSA knob, no standalone A/B) | Module weights, kind weight, requires_search, NOT needs_python_evaluator: - robber_eval.INSURANCE_W = 0.0 (off even with robber_corr=1); - INS_AVERT = 0.85; - I |
| 5 | `robber.block_duration`: Block-duration discount for every blocked player (R1c; a shadow plus an SPSA knob) | robber_eval.BLOCK_DUR_W: kind weight, requires_search, not needs_python_evaluator, default 0.0. Read only when search.robber_corr=1. |

## Card counting uses: resource odds for Monopoly and the robber, and reading opponents' held development cards (hidden VP / secret leader, knight threat and robber kicks, Monopoly plausibility)

Revised after both critiques. I re-checked every disputed fact in the code and ran three short niced log parses.

Every code claim the critiques made holds:
- ParamBot applies overrides only inside reset/decide/observe/explain.
- _follow_tracker resyncs the counter when the tracker raises.
- catanatron_players.py is pure Catanatron (it imports nothing from catanbot).
- bench_catanatron uses CATANBOT="catanbot" as our seat's label.
- arm_key hashes a fixed key_ctx.
- CatanbotPlayer defaults to seed=0.
- Colonist turns start at a roll or a 'turn' event.
- The default depth-1 search values finished leaves statically.
- static_value prices a held card at 0.12.

One error of my own: tuning.overridden does not exist; the API is tuning.apply/restore.

The diagnosis is unchanged. Counting is not the Monopoly bottleneck; reading opponents' development cards is the gap; and none of it can be proven against Catanatron. The proof data now finished make that firmer: full T1/T2/T3/T10 won 59.5% (n=2,200), counted T7/T8/T9/T11 won 60.3% (n=2,160). The campaign row t2_counted_info@value is therefore redundant.

New calibration, on T1-T11 logs with a tracker-view card model:
- The old one-parameter hazard (0.5) fails an age-stratified gate on ValueFunction. Its log-loss for cards held 10+ turns is 1.85 vs 1.06 for uniform: Catanatron hoards Road Building.
- A per-type-present joint likelihood with a per-type floor (H=0.85, EPS=0.03) beats uniform in every opponent class and every age bucket.
- Hidden-VP mean squared error: ours 0.237 vs 0.829, ValueFunction 0.216 vs 0.435, AlphaBeta 0.089 vs 0.389, SameTurn 0.099 vs 0.468.

Revised plan, all off by default:
1. **devbelief.hazard**, rebuilt:
   - bookkeeping stores sufficient statistics, and hazards are applied lazily under the ParamBot overrides;
   - per-type-present likelihood with an EPS floor;
   - exact dynamic-programming sampling with history-aware no-win caps;
   - dealing that consumes the shared random stream exactly as the default does;
   - error isolation;
   - four tunables;
   - Python only, no C++ impact.
2. **advisor.dev_read**: the first adoption, with no game budget. It adds per-opponent card odds, a conditioned secret-leader alert and a Monopoly-threat forecast from the bank. It uses an unfitted human preset and logs card ages for later fitting.
3. **shadow.dev_oracle**: a zero-game three-arm shadow (uniform / hazard / true dev cards dealt) scored by paired value regret. It gates every game-level test.
4. **search.knight_kick**, a new consumer. A robbed victim holding a knight kicks the robber next turn 99.4% of the time. This is a depth-1 leaf chance node in exact evaluator units; the critique measured it and I re-ran it.
5. **harness.selfbot_opponents**: its contract bugs are fixed, and its games are gated on the shadow result.

search.mono_threat and devcards.mono_convert are SHELVED at design; their formulas are corrected in case they are revived.

The game-level non-inferiority run against ValueFunction becomes a 200-game smoke run. It can optionally ride a shared counted default arm after a code freeze.

| priority | feature | switch (off by default) |
|---|---|---|
| 1 | `devbelief.hazard`: Age- and behaviour-aware posterior over opponents' hidden dev cards (per-type-present likelihood, eps floor, exact DP sampling, history-aware no-win caps, RNG-neutral dealing) | devbelief.ENABLED is a 0/1 weight, default 0: uniform dealing, byte-identical, and the existing re-deal guard stays untouched.  The other knobs are read only wh |
| 2 | `advisor.dev_read`: Advisor dev-card reading: per-opponent card odds, conditioned secret-leader alert, Monopoly-threat forecast from the bank, knight-kick note; human preset and age logging | CLI --dev-model uniform (default): report() and the CLI 'Card count' text are byte-unchanged (golden test). --log-dev-ages is off by default. It is not a bot be |
| 3 | `shadow.dev_oracle`: Zero-game dev-reading shadow: uniform vs hazard vs true-dev-cards arms, paired value regret, plus the reveal_devs oracle flag | Harness and diagnostic only. reveal_devs defaults to False and is not a bot behaviour, so there is no tuning.py flag. The script is new and nothing else calls i |
| 4 | `search.knight_kick`: Knight-kick leaf chance node: a robber block on a knight holder lasts until that player's turn | search.kick is a search-kind tunable, default 0.0: no correction and bit-identical values. Candidate 0.95. Spec key kick. Depth-1 only. |
| 5 | `harness.selfbot_opponents`: Dev-heavy hidden-information table: our default bot as the three opponents in the Catanatron harness (games gated on the shadow) | A harness option only (--opponent selfbot --opponent-spec SPEC). It is not a bot behaviour, so there is no tuning.py flag. Default runs, arm keys and the bot ar |
| 6 | `search.mono_threat`: Opponent Monopoly threat (SHELVED at design) | search.mono_threat, search-kind, default 0.0 (not registered until revived). |
| 7 | `devcards.mono_convert`: Monopoly resource by conversion value (SHELVED at design; formula corrected) | devcards.MONO_CONVERT, 0/1, default 0 (not registered unless revived). |

## Implementation order

1. **harness (budgeted test queue)**: `queue.seqtest`, `queue.intake`, `queue.scheduler`, `queue.snapshot`, `queue.mechanics`, `queue.report`, `queue.selfplay_rows`, `queue.crn (part a: --crn dice only)`, `queue.cv_default (tooling only; pools are separate queue rows)`, `robber.shadow_metrics (generic part only: a zero-game decision shadow)`
2. **shared plumbing (trades, ports, robber, counting)**: `shared.corrections_hook`, `shared.seat_events`
3. **trades**: `acq.progress`, `acq.breadth`, `acq.flow (function and fitted table only; its provider is in step 4)`, `acq.calib (with the REJECT_STREAK fallback)`
4. **ports**: `ports.surplus_value`, `ports.constants`, `ports.flow_provider (consumer of acq.flow)`, `ports.advice`, `ports.spot_want (only the rival-want and leader helper used by the advice text; the winpaths integration is in step 7)`
5. **robber**: `robber.shadow_metrics (RobberCounters plus the robber classifier in decision_shadow)`, `robber.persistence`, `robber.knight_insurance`, `robber.block_duration`, `search.knight_kick (moved here from card counting; the alternative to persistence)`
6. **card counting**: `devbelief.hazard (robber R3 folded in as devbelief.H_KNIGHT_CTX, using the robber design's event definition)`, `advisor.dev_read`, `shadow.dev_oracle`, `harness.selfbot_opponents (code, A/A and a 20-game pilot only; its game batch is gated)`, `queue.crn (part b: seeded_samples)`, `robber.knight_evidence (as H_KNIGHT_CTX)`
7. **conditional follow-ups (ports / robber / other)**: `ports.spot_want (winpaths integration: SPOT_WANT in _spot_table)`, `C++ constexpr port of an adopted port setting`, `robber_eval C++ port`

## Test queue

| priority | row | harness | games (cap) | est. hours | fallback / follow-up | politics rule |
|---|---|---|---|---|---|---|
| 1 | `queue.aa_smoke (existing t1_baseline_aa@value and @vf, plus a 3-row real smoke)` | catanatron (value and vf, full info, trades off); A/A estimate rows plus a smoke on 3.2.1 vs vf (A/A, search.e | 3200 | 0.5 |  |  |
| 2 | `queue.crn (a) dice pilot` | catanatron (vf on 3.2.1, full info, trades off): 400 seeds x {crn off, crn dice}, candidate search.expand=4 | 1600 | 0.2 |  |  |
| 3 | `t1_trades0_vrule@value (trades headroom; existing row, epoch A)` | catanatron (value, full info, --trades value): fixed-N estimate, 2,000 seeds; its trading-on default arm is sh | 4000 | 1 |  |  |
| 4 | `t2_dump0@value (existing knockout, epoch A)` | catanatron (value, full info, trades off): knockout design, N_max 1,000; default arm shared | 1000 | 0.2 |  |  |
| 5 | `acq.progress (search.acq=2, own seat)` | catanatron (value, full info, trades off). - Stage 0: zero-game audit on 300 positions. - Stage 0b: 200 seeds  | 4600 | 0.85 |  |  |
| 6 | `acq.breadth (bundle: search.trade_proposals=5 + search.acq_shapes=1)` | selfplay (4p 2v2, C++ evaluator, default search base spec): a 100-game mechanism check first, then a seqtest s | 2500 | 2.5 |  |  |
| 7 | `acq.breadth (A) alone: search.trade_proposals=5` | selfplay (4p 2v2, C++ evaluator), seqtest screen | 2400 | 2.4 | fallback: acq.breadth (simpler). It replaces the bundle when (B) gets 0 accepts in the mec |  |
| 8 | `acq.breadth follow-ups (value-rule confirmation, 3p 1v2 no-harm, league gate)` | catanatron (value, full info, --trades value, 2,000 seeds) + selfplay (3p, 1,200 games, only the 1v2 seatings  | 6400 | 6.3 | follow-up: only after an ADOPT of acq.breadth |  |
| 9 | `acq.calib (opponent_model.CALIB_RATE=1)` | Mechanism gate first: 50 self-play games plus 40 native-trading games vs value. Then selfplay (4p 2v2): the on | 1290 | 1.3 |  | yes |
| 10 | `acq.calib fallback: opponent_model.REJECT_STREAK=3` | catanatron (value, full info, --trades native): 40-game behaviour-only check, then straight to the deferred-to | 40 | 0.05 | fallback: acq.calib (simpler), if the calibration mechanism gate fails | yes |
| 11 | `acq.flow (trade-flow port value table)` | offline replay (fit on 1,000 T1 proof games, validate on T2); 0 games | 0 | 0.05 |  |  |
| 12 | `ports.surplus_value (F1, placement.PORT_MODEL=1)` | Zero-game decision screen first (PORT_MODEL 0 vs 1/2/3, A/A). Then a mechanism gate cell: catanatron (vf on 3. | 600 | 0.2 |  |  |
| 13 | `ports.constants (F2, P3': heuristic.PORT_STATIC_GENERIC=0.8 + placement.PORT_GENERIC_ONCE=1)` | catanatron (vf on 3.2.1, full info, Python evaluator): second gate cell, 300 seeds, shared base | 300 | 0.1 |  |  |
| 14 | `ports.flow_provider (acq.flow via the hub, ports.FLOW_KAPPA)` | catanatron (vf on 3.2.1, full info, C++ evaluator): third gate cell, 300 seeds with its own 300-seed C++ base  | 600 | 0.1 |  |  |
| 15 | `ports.surplus_value mode 3 (calibrated 3:1 only)` | catanatron (vf on 3.2.1, Python evaluator): 300-seed gate cell | 300 | 0.1 | fallback: ports.surplus_value (milder), only if F1 raises cards saved but fails the settle |  |
| 16 | `ports best-cell row` | catanatron (vf, full info; Python evaluator for F1/F2, C++ for the flow cell): one seqtest screen at N_max 2,0 | 4000 | 1.25 | follow-up: only if a gate cell PASSes |  |
| 17 | `ports.advice (portvalue.ADVICE)` | zero-game: unit tests plus a CLI smoke; a fixed-seed game trace must be byte-identical | 0 | 0 |  |  |
| 18 | `ports.spot_want (winpaths.SPOT_WANT)` | Zero-game screen first. Then catanatron (value, full info, C++ evaluator): one extra Stage-5 knock-out row, 2, | 2000 | 0.6 | follow-up: only if winpaths Stage 5 (other area) adopts paths_spots=1 |  |
| 19 | `ports.spot_want SPOT_LEADER` | deferred to human testing (no bot harness conditions on the leader) | 0 | 0 |  | yes |
| 20 | `robber.shadow_metrics (robber shadow)` | Zero-game shadow: 40 default self-play games, with every robber, knight-roll and main decision re-searched per | 0 | 1.3 |  |  |
| 21 | `robber existing C++ rows (t2_danger_mult_off, t2_block_need0, t2_steal_factor_off, t2_rob_break_off knockouts; t2_turns_half2, t2_knight03)` | catanatron (value, full info, trades off, C++ evaluator): seqtest knockout or screen, N_max 1,000 (pre-planned | 6000 | 1.2 |  |  |
| 22 | `robber existing Python-evaluator rows (t2_exposure0_pyeval@value; t1_blockw0_pyeval@value and @vf)` | catanatron (value and vf, full info, Python evaluator): knockout design; one shared Python-evaluator default a | 9000 | 3.8 |  |  |
| 23 | `robber.persistence (search.robber_corr=1; PERSIST_W=1, KICK_LAMBDA 0.95, geometric keep, R_REF 4.34)` | selfplay (4p 2v2, C++ evaluator): seqtest screen over 2,400 games, with the paired VP difference as co-primary | 2400 | 2.4 |  |  |
| 24 | `robber.persistence harm screen` | catanatron (value, full info, C++ evaluator): 2,000 paired seeds | 4000 | 0.7 | follow-up: only if robber.persistence ADOPTs or its pooled estimate is at least +1.5 pp; t |  |
| 25 | `search.knight_kick (search.kick=0.95)` | selfplay (4p 2v2, C++ evaluator): screen over 2,400 games with a mechanism look at 600 games | 2400 | 2.4 | fallback: robber.persistence (simpler, same mechanism). Only if R1a fails shadow gate G1 ( |  |
| 26 | `robber SPSA group (robber.knight_insurance INSURANCE_W 0-3 + robber.block_duration BLOCK_DUR_W 0-0.8, plus the C++ robber knobs not REMOVEd)` | catanatron (value, full info, C++ evaluator): tune_joint robber group with the base spec robber_corr=1; knobs  | 4000 | 0.8 |  |  |
| 27 | `robber.persistence RETAL_W (retaliation)` | deferred to human testing (neither bot population retaliates: 22.7% and 35.6% vs 33% uniform) | 0 | 0 |  | yes |
| 28 | `devbelief.hazard (devbelief.ENABLED=1, H 0.85, EPS 0.03)` | Gate (A): offline calibration on T1-T11 through the real DevAgeModel (0 games). Then smoke (C): catanatron (va | 200 | 0.3 |  |  |
| 29 | `robber.knight_evidence (as devbelief.H_KNIGHT_CTX)` | offline: the knight-context stratum inside gate (A); 0 games | 0 | 0 |  |  |
| 30 | `shadow.dev_oracle` | zero-game three-arm shadow (uniform / hazard / true dev cards) on about 2,000 counted positions from T4 logs,  | 0 | 1.5 |  |  |
| 31 | `advisor.dev_read (--dev-model bot/human, --log-dev-ages)` | zero-game: unit and golden tests; no game budget | 0 | 0 |  |  |
| 32 | `devbelief.hazard counted non-inferiority` | catanatron (value, --info counted, seeded samples): 2,000 seeds on a shared counted default arm | 4000 | 2.7 | follow-up: only if the user wants a game-level harm screen before making devbelief the cou |  |
| 33 | `queue.crn (b) seeded-samples pilot` | catanatron (vf, --info counted): 400 seeds x {off, seeded}, candidate discards_public=True | 1600 | 0.6 | follow-up: only before a counted game row beyond the smoke (the non-inferiority row or the |  |
| 34 | `harness.selfbot_opponents` | catanatron harness with 3 default catanbot opponents: an A/A plus a 20-game cost pilot now; the gated batch is | 40 | 0.1 |  |  |
| 35 | `politics t2 vrule rows (t2_max_slack0, t2_coal_scale2, t2_feed_leader_off, t2_late_drop0; existing, epoch A)` | catanatron (value, full info, --trades value): fixed-N politics screen, 1,000 seeds each; default arm shared w | 4000 | 1.1 |  | yes |
| 36 | `search.counters (self-play politics screen)` | selfplay (4p 2v2, --counters): fixed 1,200 games in 5 chunks of 240 | 1200 | 1.2 |  | yes |
| 37 | `search.respond_lookahead (self-play politics screen)` | selfplay (4p 2v2): fixed 1,200 games in 5 chunks of 240 | 1200 | 1.2 |  | yes |
| 38 | `other existing rows (t1_search_vs_heur x2, t1_openings x2 with 4 values each, t1_depth2, t2x_beam2, t2x_expand4, paths_main@value/@vf; epoch A)` | catanatron (value and vf, full info, trades off): estimate rows for search_vs_heur, screens for openings/depth | 24000 | 4.7 |  |  |
| 39 | `queue.cv_default pool (value C++ arm, M=8,000)` | catanatron (value, full info): default-only games on a disjoint seed block, one per arm key and epoch | 8000 | 1.3 | follow-up: only if the 'other' high-D rows (openings, depth2, beam2, expand4, paths, searc |  |
| 40 | `tier-3 AlphaBeta confirmations (t3 rows, disabled today)` | catanatron (alphabeta, full info): 400 seeds each; exclusive load gate | 1600 | 3.4 | follow-up: only for tier-1 ADOPTs |  |

## Shelved up front

- **acq.minor_routes** (port-aware Monopoly, Year of Plenty and discards): 0.036 changed Monopoly decisions and at most 0.09 discards a game. Proving it would need tens of thousands of games.
- **acq.mono_guard** (do not hold cards into an opponent's likely Monopoly): at most about 0.1 card a game. Kept as a spec note only.
- **ports.monopoly_liquidity (F5):** the same mechanism and reach as acq.minor_routes (a 2:1 port on the monopolised resource in about 3% of games). The trades area's measurement serves as its zero-game screen. Advisor text at most.
- **search.mono_threat:** the old version was mis-scaled 4-8x against the static value; hidden Monopolies are rare and played fast; 0.1-0.3 pp. The exact chance-node formula is kept in case it is revived on the shared leaf-chance hook.
- **devcards.mono_convert:** counting is already sufficient for our own Monopoly (T7 = T1); about 0.3 build units a game of headroom.
- **Robber R2** (need-aware block normalisation): its bug is confirmed, but the effect is plus or minus 1 pp of uncertain sign. **R4** (steal swap): breaks the per-resource totals. Neither is built.
- The robber design's **separate knight-evidence sampler and public_belief.KNIGHT_EVIDENCE tunable:** folded into devbelief.H_KNIGHT_CTX.
- **Standalone A/B tests** of robber.knight_insurance and robber.block_duration: +0.5 pp would need about 10^5 seeds. They become SPSA knobs instead.
- The planned **KNIGHT_VALUE x EXPOSURE_WEIGHT 2x2 factorial:** an interaction needs about 4x the games of a main effect. Replaced by INSURANCE_W in the SPSA robber group.
- The robber design's old **knight-after-roll rule** (wrong premise) and the **'robber package' bundle** (its members act in different populations, so bundling adds no power).
- **Parts of the old acq.offers:** the route table, density retargeting, summed targets and ask-for-2 (no mis-targeting found; almost never accepted). The port-rate floor (0 reach), propose_margin (a cancelled offer is free) and the mixed-offer etiquette go straight to the human-testing list.
- **The ports P2 road-target race:** changed only 4.8% of search decisions (A/A 0), its clock was biased against us, and it ignored conversion.
- **Other ports ideas, not queued:** a setup 'port plan' credit (static's reach weight makes it about 0.1 points), road cuts in winpaths _contest, and port-aware hand liquidity in the C++-mirrored static value.
- **t2_counted_info@value:** redundant. The finished proof already shows counted vs full at 60.3% vs 59.5% over 4 opponent sets; its CPU goes to shadow.dev_oracle.
- **Game-level tests of dev reading against Catanatron:** superiority cannot be shown with a gap of about 0, and the old 4,000-game non-inferiority run was effectively an A/A. Only a smoke run plus an optional follow-up remain.
- **Counting rows in self-play or the league** (belief is None and hands are fully known there), **any self-play row on the Python evaluator** (170 games/h: 2,400 games take about 14 h), and devbelief's per-type hazard, GIBBS_SWEEPS and H_CAP tunables (removed).
- The **acq_self=0 symmetric variant** and **Stage 2 of acq.progress** (vf, self-play, league) unless acq.progress ADOPTs first.
- **Queue tooling left out:** look-synchronous groups, canary aliasing and the proof-log divergence probe (all deferred), and scripts/pool_paired.py (replaced by seqtest chunks).
- **A trade_cap_early registry entry** (it can only lower the engine maximum) and **mirrored 1v3 self-play seat patterns** (they need a flag from ablate.py's owner).

## Cross-area conflicts and resolutions

- **Leaf-value corrections.** Four designs wrap the same four evaluator sites in catanbot/search.py (_eval, _counter_filter, the political options, the native-futures guard) plus reduced_config:
- trades: acq.progress's CorrectionHub via winpaths.PathsEvaluator;
- robber: RobberEvaluator with its own fast path, refusing paths=1;
- ports: the flow provider;
- counting: knight_kick's _leaf_chance hook.
Resolution: step 2 builds ONE hub (catanbot/corrections.py):
- a provider list;
- the robber design's bit-identical fast path;
- PathsContext as a provider;
- a leaf-chance hook;
- paths=1 alone keeps the existing PathsEvaluator path.
This removes the 'robber_corr + paths refused' rule and the 3x-per-leaf stacking risk.
- **knight_kick (counting) and robber.persistence R1a model the same mechanism:** a block on a knight holder is temporary. Stacking them double-counts. Resolution:
- both move to the robber step and share the shadow;
- R1a is primary (calibrated geometric lifetime, several kickers, composes with R1b/R1c);
- knight_kick is its fallback only if R1a fails shadow gate G1 or G4, never after a win-rate SHELVE, and never in the same arm.
- **R3 (robber knight evidence) and devbelief's H_KNIGHT_CTX (counting) are the same likelihood event.** Resolution:
- fold R3 into devbelief as H_KNIGHT_CTX, with the robber design's event definition (main victim, at least 3.0 demand-weighted pips, old cards, no play);
- one exact DP sampler (the counting design's);
- no public_belief.KNIGHT_EVIDENCE tunable and no separate per-player enumeration sampler.
- **Candidate overrides do not reach tracker.determinize.** The robber design proposes ParamBot.scope() around determinize; the counting design proposes applying and restoring the devbelief.* subset in _decide_counted. queue.crn(b) seeded_samples edits the same function. Resolution: one ParamBot.scope() in agents/param_bot.py, used around determinize, with all three adapter edits in step 6. devbelief's side-stream dealing stays RNG-neutral under seeded samples.
- **Mechanism counters.**
- ports wants port_* counters in CatanbotPlayer.stats (fingerprinted);
- robber wants robber counters in the adapter plus 'seat_robber' in play_paired_game;
- trades wants counters in _real_game and play_paired_game;
- the queue wants scripts/mechanics.py.
Resolution:
- Catanatron harness: only scripts/mechanics.py, from the in-memory action log (not fingerprinted; covers ports, bank rates, robber, steals, discards, Monopoly, dev holding). Adapter stats keep only bot-internal counters (offers, info_samples, devbelief_errors, spot_want_active).
- Self-play: one generic observer hook in tuning.play_paired_game (step 2), with robber and trade observers.
- Check that ablate.py --json carries play_paired_game's extra keys; if it does not, that is a request to ablate.py's owner.
- **Two port-value models:**
- ports F1/F2: a calibrated conversion model in the spot score and static value (C++-mirrored, Python evaluator);
- trades acq.flow: trade-flow value per resource and time, through the hub (main phase, C++ evaluator).
Resolution:
- the function and table live in acquisition.py (step 3); the provider and ports.FLOW_KAPPA are owned by ports (portvalue.py, step 4);
- all three are cells of ONE ports mechanism gate, and only the best passing cell gets a row;
- the flow provider is never stacked with F1 modes 1/2 (it subtracts the static port credit, so otherwise it would double-count).
- **Order of acq.progress vs ports.** The trades design suggests running acq.progress after the ports screen, because conversions matter more with more ports. The user's priority puts trades first. Resolution: run acq.progress in trades order on today's port behaviour. Re-screen it once, on top of any adopted port setting, only if that setting raises our port ownership.
- **Monopoly work is spread over four areas:** acq.minor_routes, acq.mono_guard, ports.monopoly_liquidity, search.mono_threat and devcards.mono_convert. The measured reach is 0.036 changed Monopoly decisions a game, with a port on the resource in 29 of 1,000 games, and counting is already sufficient (T7 = T1). Resolution: all shelved. One human-facing surface only: the Monopoly-threat forecast and odds text in advisor.dev_read and trade_advice.
- **Stopping rules.** Each design brings its own:
- robber: its own O'Brien-Fleming 3x800 plus pool_paired.py;
- trades: a futility look at 1,000 and legacy --stop-at-se;
- ports: should_stop harm guards and 'Holm over about 60 rows';
- counting: its own non-inferiority rules.
Resolution:
- every row goes through seqtest (screen / knockout / estimate / politics / confirm), with Holm within the tier (F8);
- the designs' own pre-registered rules become entry gates (mechanism and shadow) before a row, not alternative stopping rules;
- self-play screens use 240-game chunks with looks at multiples of 480;
- pool_paired.py is not built.
- **Self-play scope.** queue F9 and the F2 lint allow self-play only for politics screens and trade no-harm checks. But robber.persistence, knight_kick and acq.breadth need self-play as their primary harness, because they are inert against Catanatron. Resolution: the lint allows self-play for mechanics that do not fire in the Catanatron harness (player trades, robber on knight holders), with the C++ evaluator only. Counting stays banned from self-play and the league (belief is None there).
- **Code fingerprint.** Every area edits fingerprinted catanbot/ files, as does the other agent's catanbot/factorial.py, and each edit voids default-arm reuse. Resolution:
- queue snapshots (F4) with one epoch per area batch: epoch A = today's code (step 1 is scripts only); B1 after trades; B2 after ports; B3 after robber; B4 after counting;
- zero-game gates run from the working tree before each bump;
- default arms are mostly area-specific, so per-area bumps cost about 0.3-0.7 h of replayed defaults each.
- **Shared files and ownership.**
- search.py is edited in steps 2, 3 and 5; tuning.py in steps 2-6 and by the SPSA/factorial agent; selfplay.py make_bot in steps 3 and 5; catanatron_adapter.py only in step 6 (the counters moved to mechanics.py).
- Each new module registers through register_tunables plus one import line at the end of tuning.py.
- scripts/ablate.py and scripts/tune_joint.py are not edited. The SPSA group additions (INSURANCE_W, BLOCK_DUR_W, a 'ports' Python-evaluator group) are requests to their owner.
- **Explanation text on the search path.** acq.progress puts its 'keeps 4 sheep...' text in search.explain. The ports design measured that explanations for every root action cost 2.4 of 18.2 ms per decision. Resolution: all new advice text (acq.progress, ports.advice, advisor.dev_read) is computed only on the CLI advisor path.
- **C++ policy for port weights.** RESULTS.md says to port static-value weights to C++ with a parameter. The ports design rejects a runtime C++ setter: it is process-global, leaks into default seats in paired games, and the native opponent simulation and winpaths _spot_score call C++ directly. Resolution: needs_python_evaluator for the gate and for any row against Catanatron (300-2,000 seeds at about 3,000 games/h); a hard-coded constexpr only on ADOPT. A per-call C++ parameter is built only if self-play of port weights ever becomes necessary (170 games/h on the Python evaluator is refused).
- **Robber prior-only rows.** The queue plan runs t2_danger_mult_off, block_need0, turns_half2, steal_factor_off and rob_break_off as knockouts; the robber design shadows them first. Resolution: the step-1 generic shadow runs before them in epoch A. Rows that change fewer than 2% of robber/knight decisions are skipped as NOOP(shadow) at 0 games, unless the user wants inertness confirmed to justify removing code.
- **Throughput and proof state.**
- The task text assumes about 6,000 self-play games/h. RESULTS.md measured about 1,000 games/h on 3 cores (11.5 s per game), and 170 on the Python evaluator. This plan uses 1,000.
- The task says the proof is running. RESULTS.md reports T7-T11 finished at 04:10 UTC, and no bench, ablate or Python process was running when this plan was made. The proof guard stays for future pre-registered runs.

## Notes (total test hours 26.3)

**How the hours are counted.** Hours are wall-clock on 3 cores (x3 for CPU-hours), taken at each row's game cap. They assume:
- about 6,000 games/h vs value on the C++ evaluator, and about 9,000 vs vf;
- about 1,960 / 3,250 on the Python evaluator vs value / vf;
- about 3,550 with --trades value;
- about 1,500-2,800 in counted mode;
- about 1,000 in self-play (the RESULTS.md measurement, not the task's 6,000).

The total of 26.3 h counts every unconditional row, including 4.7 h of 'other' rows that run only if there is time. The four areas plus harness and politics come to 21.6 h. Expected time with seqtest early stopping at null is about 19 h. Conditional follow-ups add up to about 30 h more if everything triggers; about 8 h of that is league gates. The self-play rows (acq.breadth, persistence, 3 politics screens) are about 40% of the cost.

**Batching.** One snapshot epoch per area:
- Epoch A = today's code, available as soon as step 1 lands. Existing rows run at once, in area order: harness, trades headroom and dump0, shadow-pruned robber rows, politics, other.
- While steps 2-6 are implemented, each area's zero-game gates run from the working tree. Its epoch then starts and its rows preempt lower-area epoch-A rows at slice boundaries.

**C++ impact per feature:**
- Only ports.surplus_value and ports.constants touch the C++-mirrored static_value and placement score. Recommended: needs_python_evaluator for the gate and any Catanatron row, and a constexpr C++ change only on ADOPT (a runtime setter would leak across paired seats).
- Everything else is Python, with no C++ port: the acquisition and robber_eval providers on the hub, knight_kick, devbelief, opponent_model calibration, advice and harness options.

**Decisions for the user:**
1. **Rows below what the budget can show.** acq.progress, acq.breadth and robber.persistence are honestly expected at +0.5 to +2 pp; proving that would need 10-25k games. The plan screens them anyway, because at null they stop at about half their cap (about 4-5 h in total) and they are the only measurable levers in trades and robber. Strict reading of your rule: SHELVE them at intake with 0 games.
2. **Adoption without a game-level win.** May features be adopted on calibration + shadow + smoke evidence (devbelief), on non-inferiority (acq.calib's calibration and compute gain), or on mechanism + harm screen (insurance-type terms)? Otherwise they stay advisor-only or shelved.
3. **Politics Holm family.** Recommended: a 7-row politics tier (4 vrule rows, counters, respond_lookahead, acq.calib). The whole t2 tier is 15 rows, with power of only about 0.34 at +4 pp.
4. **Skip screens that cannot react.** SPOT_LEADER and RETAL_W would go straight to the deferred-to-human list without spending their one screen, since no bot harness reacts to them. The same for propose_margin, the port floor, mixed-offer etiquette, REJECT_STREAK and our dev-card tell (we hold only VP cards beyond one turn).
5. **Epoch bumps.** Per-area bumps (recommended; about 0.3-0.7 h of replayed default arms each), or one freeze after step 6, which delays the trade rows by the implementation time of steps 4-6.
6. **Shadow-inert robber rows.** Skip the prior-only robber rows the shadow finds inert, or run them to justify removing code? Knockouts KEEP unless removal is proven better; allowing removal on non-inferiority plus speed is your call.
7. **Other open choices:**
- whether t2 rows move from 1,000 to the policy's 2,000 seeds;
- whether to build the CV pool (only worth it if the 'other' high-D rows run);
- whether the discard.py advisor text fix lands before epoch B1;
- whether to turn on --log-dev-ages in human sessions, to fit the HUMAN hazard preset.
8. **Expansion pace.** The logs place the port gap in pace, not in port preference: our first post-setup settlement comes about 10-15 turns later, city-first, and our 4:1 trades turn wheat and ore into sheep and brick. It is outside the four areas and not designed yet. It deserves a zero-game screen in 'other' before any further port work.
9. **Doc fix.** ABLATIONS.md's politics premise ('self-play copies react as our model predicts') is refuted: predicted P(accept) 0.55 vs 0.18 realised. Also, lazy explanation text would save about 13% of decision time for free (a side finding for the search owner).
10. **Requests to the SPSA/factorial agent** (not edits):
- add robber_eval.INSURANCE_W and BLOCK_DUR_W to GROUPS['robber'] with robber_corr=1;
- add a 'ports' Python-evaluator group;
- confirm that play_paired_game observer keys reach ablate.py --json.
