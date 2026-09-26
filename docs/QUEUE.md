# Test queue (`scripts/run_queue.py`)

The budgeted test queue enforces the testing policy (docs/ABLATIONS.md, "Testing policy" and "Politics rule"):
- a few thousand games per candidate (about 2,000 paired seeds vs Catanatron, 2,400 self-play games);
- sequential looks with ADOPT / REJECT / SHELVE;
- a priority queue by area, where preemption must pay for itself in CPU seconds;
- default arms shared within a code epoch;
- fallback rows once an idea goes on ice;
- one fixed-N screen per politics term.

It is step 1 of docs/PRIORITY_PLAN.md, and the full design is in `docs/designs/priority_areas_2026-09-26.json` (infra).

Everything lives in `scripts/` and is test tooling, not a bot behaviour, so nothing registers in `catanbot/tuning.py`. That is deliberate: the bot is untouched, and a tuning.py entry would change the code fingerprint.

| file | role |
|---|---|
| `scripts/run_queue.py` | the queue: intake, scheduler, snapshots, chunks, ledger, `QUEUE.md` (named so because a `scripts/queue.py` would shadow the stdlib `queue` module) |
| `scripts/seqtest.py` | verdict engine: boundary tables, designs, operating characteristics, CV estimator |
| `scripts/mechanics.py` | per-game mechanism metrics from the action log (also `--proof DIR` base rates) |
| `scripts/decision_shadow.py` | zero-game decision shadow (default vs candidates on the same positions; `settle` class and port picks; `--py-shadow`) |
| `scripts/port_gate.py` | the ports mechanism gate's pre-registered PASS rule over the gate cells' paired games (read-only) |
| `scripts/queue_plan.json` | the plan: epoch-A rows, plus disabled rows for features not built yet |
| `ablate_catanatron.py` flags | `--mech`, `--crn dice`, `--default-only`; all off by default, and arm keys are unchanged when off |

## Running it

```bash
# 1. approve the order and the CPU budget (no games)
python3 scripts/run_queue.py --plan scripts/queue_plan.json --dir runs/queue1 --explain --simulate
# 2. run it: time slices of 15 min, verdicts as looks complete; rerun the same command to resume
nice -n 10 python3 scripts/run_queue.py --plan scripts/queue_plan.json --dir runs/queue1 --max-hours 8
# 3. where are we (also written after every slice to runs/queue1/QUEUE.md)
python3 scripts/run_queue.py --plan scripts/queue_plan.json --dir runs/queue1 --status
```

Other options:
- `--once` runs one slice.
- `--report-only` evaluates looks and rewrites QUEUE.md.
- `--rebuild-ledger` re-derives the verdicts from the rows' stop records.
- `--no-intake` skips the power and headroom checks (lint still applies).
- `--hard` preempts mid-slice with SIGTERM, which loses up to W in-flight games.
- `--order index` orders within an area by weight x P(ADOPT) / cost.
- `--max-workers` (default 3) and `--slice-minutes` set the chunk size.

Before each chunk the queue checks two guards:
- **Proof guard.** No chunk starts while a process command line matches `--guard` (default `proof_snapshot`) or a lock file exists (default `/home/user/PROOF_RUNNING`; add more with `--lock-file`).
- **Load gate.** AlphaBeta-class rows (`alphabeta`, `sameturn`, `ab`, or `exclusive: true`) start only when externally busy cores <= cores - workers + 0.25. Load is sampled over 10 s after the queue's own work has drained. An ALERT is raised after 2 h of waiting.

Where things go:
- Row results go to `<dir>/<row>.jsonl`, in ablate_catanatron's format. campaign.py can share the directory.
- Verdicts and CPU bookkeeping go to `<dir>/ledger.jsonl`.
- `scripts/archive_queue.py --dir <dir>` copies each finished row's per-game results into the repository,
  compressed, at `docs/queue/games/` (README there).
- Self-play chunks go to `<dir>/selfplay/<row>/<value>/chunk_<k>.json`.

## Reading verdicts

Units:
- **Catanatron rows:** candidate minus default win rate, paired over seeds (`pp (1v3 win rate vs <opp>)`). Rows under `trades value` add "vs value-rule responders".
- **Self-play rows:** per-seat 2v2 difference.

Every candidate line in QUEUE.md shows:
- the estimate +- se;
- the default's base rate and the relative effect;
- the stage-wise p and the Holm p within the row's `tier` (`*` = provisional, until the tier is complete);
- the VP difference;
- the discordant and diverged shares;
- the mechanism deltas;
- CPU-h spent;
- the next action.

Estimates of rows that stopped early are biased away from 0, so confirm them on fresh seeds.

| design (default by polarity) | looks | outcomes |
|---|---|---|
| `screen` (new) | 5 equal looks (400..2,000) | **ADOPT** (Z >= 4.877 / 3.357 / 2.680 / 2.290 / 2.031, O'Brien-Fleming; plus an exact McNemar check below 50 discordant pairs), then the league gate. **REJECT(worse / futile)** when T = (delta - 1 pp)/se <= -2.66 / -2.46 / -2.24 / -2.02 / -1.79. **SHELVE** early at looks 2-4 when the conditional power is below 0.10 (Z < 0.66 / 0.95 / 1.30), or at the cap. |
| `knockout` (knockout) | same, with delta_min 0 | **REMOVE** (the term hurts), **KEEP(proven)**, **KEEP(unproven)** |
| `estimate` (measure) | one look at N | **ESTIMATE** delta +- 1.96 se; A/A rows **PASS** at 400 pairs with 0 diverged pairs, else **FAILED(nondeterminism)** |
| `politics` (forced for politics-scope terms) | one look at N; an ALERT at 400 pairs if nothing diverged | **SCREENED**, then **SIGNIFICANT** if the Holm p < 0.05 in its tier, else **INCONCLUSIVE**: no more games, default unchanged, deferred to human testing |
| `confirm` (rows with `confirms`) | one look | **CONFIRMED / NOT CONFIRMED** (one-sided 0.05) |
| `legacy` | as ablate_catanatron `--stop-at-se` | **LEGACY** |

Outcomes for any design:
- **NOOP** means the 95% upper bound of the diverged share is below 1%, so the change cannot move the win rate by 1 pp.
- **FAILED(inert-harness)** replaces NOOP when the row's mechanic never fired: trades with no offers, or counting with no info samples.
- **FAILED(errors / inconsistent / code-mixed / identical-trace-different-winner / plan-changed / stalled / pool-mismatch / chunks-differ)** means fix the cause and re-queue. Failures never trigger fallbacks.
- **NOOP(shadow)** is a 0-game skip: the zero-game shadow changed fewer than `min_share` of the row's decisions.
- **SHELVE(intake)** is a 0-game skip: the power at the promised effect is below 0.5 (next section).

Flags:
- `promise not met`: the 95% upper bound is below `promise_pp`.
- `small`: an ADOPT smaller than half its promise.
- `stopped early`.

At null, a screen or knockout row falsely ADOPTs about 2.1% of the time and uses about 49% of its cap. The simulated OC grid is in seqtest.py (`POWER_TABLE`; `python3 scripts/seqtest.py --grid`).

The mechanism readout (`--mech`, scripts/mechanics.py) is a secondary endpoint and never ADOPTs a row. Metrics:
- **Diversification / expansion:** `setup_distinct`, `distinct_produced`, `first_settle_round`, `first_city_round`, `settle_before_city`, `roads_built`, `settlements_built`, `longest_road` (held at the end).
- **Ports and bank:** `port_settled`, `share_4to1`, `cards_saved` (4 x cards received - cards given), bank trades.
- **Robber, steals and cards:** robber on the leader, rolls with the robber on us, cards lost to blocks, `robber_income_share` (cards lost / (produced + lost)), steals, discards, Monopoly haul.
- **Knights and dev cards:** `knights_played`, `knights_held_end`, `largest_army` (held at the end), dev cards bought and held.

Turn numbers are our own turn count (the first turn after setup is 1), summarised per arm as medians. Everything else is a per-arm mean with a paired difference and its se.

Base rates on the proof logs: `python3 scripts/mechanics.py --proof proof/T1/logs`.

## Adding a row

A row is a campaign.py experiment plus queue fields. campaign.py accepts the queue fields, ignores them, and skips rows it cannot run.

```json
{"name": "demand_flat_pyeval@vf", "area": "ports", "polarity": "new", "promise_pp": 4, "tier": "t1",
 "priority": 18, "interpreter": "py321", "opponent": "vf", "tunable": "placement.RESOURCE_DEMAND",
 "values": [[1, 1, 1, 1, 1]], "mechanism": {"metric": "setup_distinct", "direction": "+"},
 "seeds": {"count": 2000, "base": 0}}
```

Required fields:
- `area`: one of harness | trades | ports | robber | counting | politics | other.
- `polarity`: new | knockout | measure.
- `tier`: the Holm family.
- `promise_pp`: for new rows, in pp.
- Explicit `values`.

Optional fields:

| field | effect |
|---|---|
| `kind` | `catanatron` (default), `selfplay` (`games`, `players`, `counters`, `base_spec`; chunks of `chunk_games` = 240, seeds 5000+k), `command` (`cmd` with `{python}` `{snapshot}` `{dir}` `{minutes}`, and `verdict_from {json, key, pass_if / min / max}`), `pool` (a CV pool: `--default-only` on seeds 10^7+), `human` (deferred, 0 games) |
| `design`, `d_prior` | override the design or the discordance prior (full 0.40, trades 0.25, medium 0.14, low 0.04) |
| `headroom: true` | runs first in its area. The area's new rows wait for it, and player-trade promises are capped at max(1 pp, 0.5 x its upper bound) |
| `after` | `{row, verdicts, cond}`: wait for another row's verdict (`cond` e.g. `"delta >= 0.015"`) |
| `parent`, `fallback` | a fallback row (next section) |
| `confirms` | a confirmation row. It runs only for parent candidates that ADOPTed with a Holm p < 0.05, once the parent's tier is complete |
| `shadow_gate` | `{row, key, classes, min_share}`: skip as NOOP(shadow) when the shadow changed fewer decisions |
| `bundle`, `knockouts` | bundle-first testing of linked pieces (next section) |
| `bundle_of` | a power bundle: member rows too weak alone (intake) run inside this row |
| `screen_once` | plan decision 1 (the user, 2026-09-26: screen the +0.5-2 pp ideas once within the budget): the row takes its one screen even though its power at `promise_pp` is below 0.5. It skips only intake's power SHELVE and the bundle step below (a CRN / CV reducer is still used when one exists); lint, the cap, early stopping and the look-1 'unprovable at promise' re-check still apply, and an unclear screen ends SHELVE. Also a plan-level list `"screen_once": [row names]` (the same effect; scripts/queue_plan.json uses the list so that a queue process started before the field existed, which refuses unknown row fields, can still re-read the file) |
| `estimator: cv`, `crn: dice / auto` | variance reducers. CV needs a pool with M >= 4 N_max on the same default arm. `auto` means `--crn dice` once the CRN pilot passes (discordance down >= 25%, crn A/A identical) |
| `mechanism` | `{metric, direction}`: the readout the milder fallback's overshoot rule uses |
| `exclusive`, `weight`, `est_cpu_h`, `requires` | load gate, preemption weight, a command's cost, why a disabled row waits |

Intake refuses a row before any game when:
- its routing cannot exercise the mechanic: counting outside counted mode vs Catanatron, player-trade terms without `trades`, or self-play on the Python evaluator or outside politics / trade / robber scope;
- a required field is missing.

A new row whose power at `promise_pp` is below 0.5 (and without `screen_once`) gets these, in order:
1. a reducer (CRN or CV);
2. its bundle row, if one exists;
3. otherwise SHELVE(intake) at 0 games, with a bundle proposal.

After look 1, a power below 0.30 at the observed discordance ends the row SHELVE('unprovable at promise').

The plan is re-read between slices:
- A parse error keeps the previous plan.
- A removed value becomes WITHDRAWN.
- An added value starts a new run that shares the default arm.
- A changed identity of a started row is FAILED(plan-changed).
- Caps are frozen once a row starts.

## Fallbacks ("on ice, then a more moderate version")

A fallback row names `parent` and `fallback`. It plays on a fresh seed block (base + 10^6 x depth), must pass intake, and ladders are at most 2 deep. Knockout rows are never parents.

| kind | eligible when the parent ended |
|---|---|
| `simpler` (a different, more modest idea) | SHELVE, REJECT(worse), REJECT(futile), or 'promise not met' |
| `milder` (a smaller step of the same change) | REJECT(worse) or SHELVE(no gain) **and** its `mechanism` moved in the declared direction by more than 2 paired se (overshoot) |
| `stronger` | NOOP, or SHELVE(too small to prove) |

A child whose trigger did not fire shows NOT TRIGGERED. Gate outcomes are recomputed at every refresh, so a plan edit can reopen them. Verdicts from games are permanent: the first one per candidate wins.

## Bundle first, then knock out (linked pieces)

Pieces that act through the same thing are tested together (docs/ABLATIONS.md, "Regrouping"): diversification with the win-path Longest Road race (the same roads), and the robber with the Largest Army race (the same knights).

A row with `"bundle": [...]` plays every piece on against the default. A piece is one of:
- `KEY=VALUE`, a bot-spec key;
- `NAME=VALUE`, a dotted registry tunable;
- the name of a plan row that tests the piece alone.

The queue builds the candidate spec and overrides from the pieces unless the row gives them.

With `"knockouts": true`, the queue generates one knockout row per piece, `<bundle>-no-<piece>`:
- the full bundle is the default arm, and the bundle minus the piece is the candidate;
- the design is knockout: REMOVE means the piece hurts inside the bundle, KEEP(proven) means it carries the gain;
- it runs on a fresh seed block (base + 5x10^6 + i x 10^6);
- it becomes eligible only after the bundle ADOPTs.

If the bundle does not ADOPT, the knockouts are NOT TRIGGERED and member rows are SHELVE(with bundle): both pieces go on ice, with no 2x2.

After an ADOPT, member rows are superseded by the knockouts. The mechanism readout says what went wrong in a failed bundle:
- roads built and Longest Road for diversification;
- knights played vs held, Largest Army, and income under the robber for the robber.

Plan rows: `div_lr_bundle` (`ports_conversion_cost@value` + `paths=1`) and `robber_la_bundle` (`robber_corr=1` + `paths=1`). `div_lr_bundle` is enabled; `robber_la_bundle` (step 5) waits for `--bump-code --areas robber`: its pieces are bot-spec keys, which the queue does not tunable-check, so it stays `enabled: false` until the bump, then runs first in its area's new rows (priority 22.9, after the R1a shadow gate), before the lone self-play screen of `robber_corr`.

## Order and preemption

Rows are ordered by:
1. area (harness, trades, diversification, ports, robber, counting, politics, other: "the rest only if there is time");
2. headroom rows first;
3. `priority`;
4. plan order.

Cost never reorders the queue. At a slice boundary, a higher-ranked newcomer q preempts the running row r only if `w_q C_r > 1.2 w_r (C_q + S)`, where:
- `w = 4^-area_rank x 2^-(priority - area min)/10`;
- C is the remaining CPU-hours: expected remaining pairs x arms x CPU-s per game;
- CPU is measured with `getrusage(RUSAGE_CHILDREN)` around each chunk (pool workers included), with the design's class priors until a row has measured its own.

So a nearly finished incumbent finishes first.

## Code epochs (snapshots)

Chunks never run from the live repository, which other agents edit. Each row is pinned at its first chunk to a code epoch: `<home>/code/<epoch>/`. The default `<home>` is `../queue_runs/<dir name>` when `--dir` is inside the repository.

A snapshot is an explicit allowlist:
- `catanbot/**/*.py` without `vision/`;
- the C++ `.so` (not `*.staged.so`);
- the chunk scripts.

The files are copied read-only. At creation the queue asserts, per interpreter, that the snapshot's code fingerprint equals the source's and that the C++ evaluator loads there.

Epochs:
- Epoch A (today's code) is created at the first chunk.
- After an area's code lands, `--bump-code --areas trades` creates B1 for that area's rows that have not started yet. Started rows stay pinned to their epoch.
- The bump is refused while a started row of those areas is open (a batch boundary), unless `--force`.
- A row whose tunable is missing from its epoch waits as BLOCKED(needs --bump-code).
- Default arms are shared only within an epoch.

## The epoch-A plan (scripts/queue_plan.json)

The enabled rows are the existing campaign rows, re-expressed:
- **harness:** the two A/A rows, the 3-row real smoke on 3.2.1 vs vf, and the CRN dice pilot (off / dice / crn A/A).
- **trades:** `t1_trades0_vrule` (headroom estimate) and the `t2_dump0` knockout.  Step 3 (epoch B1, after
  `--bump-code --areas trades`; the rows wait as BLOCKED until their tunables exist): the acq.breadth bundle
  (`search.acq_breadth`, self-play) with its value-rule confirmation and 3p no-harm follow-ups; the player-trade
  premium (`search.acq_floor`) in self-play and vs value-rule responders; acq.calib's 40-seed native-trading check,
  then its one politics self-play screen, and the rejection-streak fallback (40-seed check + deferred to human
  testing).  Their plan promises (+2 pp) are below what the budget can show (power 0.38-0.42), so the three
  screens and the calibration screen are in the plan-level `screen_once` list (plan decision 1).  A queue process
  started before step 3 ignores that list and SHELVEs them at intake: restart it after the bump.  acq.progress is disabled (it failed its Stage 0 audit) and the (A)-only breadth fallback is not
  triggered (injected offers were accepted).  `acq_flow_fit` is a zero-game command row run from the working tree.
- **diversification:** openings (pips_diversity, standin_book and the setup_pick control) vs value and vf, and flat resource demand, with the milder registry vector as its fallback. The port gap is an expansion / diversity gap. Disabled until built: `expansion_reach_credit`, `ports_conversion_cost` (conv=1) and the `div_lr_bundle`.
- **ports (port access only; step 4, its own epoch after `--bump-code --areas ports` - B2 went to the robber step; until then the gate cells wait as BLOCKED):** the mechanism gate (next section)
  and one best-cell screen per cell behind it; `ports_spot_want@value` stays disabled (step 7, after winpaths Stage 5)
  and SPOT_LEADER is deferred to human testing.
- **robber:** a 40-game zero-game shadow that gates the prior-only rows, plus the knockouts and new-direction rows.
  Step 5 (epoch B3, after `--bump-code --areas robber`): a second 40-game shadow, `shadow_robber_step5` (`decision_shadow.py --robber-gates`: the robber / would-kick-holder classifier), and four command gate rows reading it - `robber_gate_r1a` (persistence G1-G4), `robber_gate_kick_trigger` (PASS when R1a fails G1 or G4: the knight_kick fallback rule), `robber_gate_r1b` (insurance G1-G5) and `robber_gate_r1c` (block duration). These are `enabled: false` until the bump (command rows are not tunable-checked).  Behind them: `robber_la_bundle` (disabled until the bump) and its two knockouts, the self-play screens `robber_persistence` (after the R1a gate) and `knight_kick` (after the trigger), both in `screen_once` and both BLOCKED / WAITING until the bump because `search.robber_corr` / `search.kick` are not in epoch A.  Insurance and block duration get no standalone A/B: on a gate PASS they are requests to the SPSA owner (`robber_spsa` stays disabled).
- **politics:** four vrule rows and two 1,200-game self-play screens, all fixed-N, in one politics tier.
- **other:** search vs heuristic (estimates), depth 2, the cheaper-search knockouts, and win-path rows gated by their own shadow.
- **tier-3 AlphaBeta confirmations:** gated automatically.

### Ports gate (step 4)

The design's one cheap mechanism gate (docs/PRIORITY_PLAN.md "Ports"), all in area ports, run in this order:

| row | what |
|---|---|
| `ports_gate_f1@vf` | F1 `placement.PORT_MODEL=1`, 300 seeds vs 3.2.1 vf, Python evaluator (`estimate`, tier `smoke`) |
| `ports_gate_f2@vf` | F2 P3' (`heuristic.PORT_STATIC_GENERIC=0.8` + `placement.PORT_GENERIC_ONCE=1`), same default arm |
| `ports_gate_flow@vf` | the flow provider (`ports.FLOW_KAPPA=0.25`), C++ evaluator, its own default arm |
| `ports_gate_mode3_trigger` | command: PASS when F1 raised cards saved (> 2 se) but failed the settlement guard |
| `ports_gate_f1_mode3@vf` | the milder F1 mode 3 (`PORT_MODEL=3`), only after the trigger PASSes |
| `ports_gate` | command (`scripts/port_gate.py`): PASS when a cell passes the pre-registered rule |
| `ports_gate_pick_<cell>` | command: PASS for the passing cell with the largest gain in cards saved |
| `ports_best_<cell>@vf` | the one 2,000-seed screen (fresh seeds 300..), only after its pick PASSes (in `screen_once`) |

Pre-registered PASS: cards saved vs 4:1 (`cards_saved` = 4 x cards received - cards given; a `mech` metric) up by at
least 1.1 a game and settlements a game not lower by 0.15 or more, with at least 90 % of the 300 seeds paired (a drop of exactly 0.15 fails).  The
milder fallback and the best-cell choice are command rows because the built-in `milder` rule needs a REJECT / SHELVE
verdict, which an estimate row never gives, and `after` conditions read only a verdict's top-level numbers.  A gate
FAIL (the expected result) leaves every downstream row NOT TRIGGERED: the port valuation line is shelved with its
knobs registered.

Removed: `t2_counted_info` (the finished proof already measured counted vs full). Its numbers are the static counting headroom.

Rows of features not built yet are present with `enabled: false` and a `requires` note. Enable them when their code lands and the area's epoch is bumped.

`--explain --simulate` prints the cost:
- about 32 CPU-h at null for the unconditional epoch-A rows (about 10.6 h wall on 3 cores);
- about 30 CPU-h more for conditional rows at their caps (tier-3 confirmations, the milder demand vector).

Tests: `tests/test_seqtest.py`, `tests/test_queue.py`, `tests/test_mechanics.py`, `tests/test_decision_shadow.py`, `tests/test_ports.py` (port_gate.py). The golden files are in `tests/data/`. After a plan edit, regenerate `queue_explain_golden.txt` with `python3 -c "import tests.test_queue as t; t.write_explain_golden()"`.
