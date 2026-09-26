# Results log

Living document, updated by the overnight check-ins.  Newest entries first.
Everything here was measured on the cloud container (4 shared cores, 15 GB);
numbers vary with the load from concurrent jobs.

## 2026-09-26 09:22 UTC - check-in: two trade ideas shelved or stopped; the ports agent was stopped

- **Wider trade offers** (`acq_breadth_bundle`, self-play): +0.2 +- 1.6 points at 960 pairs, too small to
  prove.  SHELVE; the default is unchanged.
- **Acceptance calibration:**
  - It failed its behaviour check: against Catanatron's bots, which never accept, it should drop below 10
    offers a game, and it still made 58.0 (default 59.6).  The check also measured -12.5 +- 5.3 points at 40
    pairs (a smoke row).
  - Its self-play screen (`acq_calib`) was therefore switched off by hand, as the plan says.
  - Next comes the simpler fallback, a rejection-streak rule: a 40-game behaviour check, then the
    human-testing list.
- **Ports step 4:** the background agent was stopped by the user.  Its unfinished code is on the branch as
  "WIP ports step 4" commits and is not reviewed.  The ports queue rows stay blocked (they need a code epoch
  that is not made).  No new implementation agent starts without the user's word.

## 2026-09-26 08:50 UTC - 1v1 benchmark: 75.5 % against Catanatron's AlphaBeta (HexMachina comparison)

Pre-registered in `docs/BENCH_1V1_PROTOCOL.md` (commit `35ef224`, before any game) and run from a frozen
worktree of that commit.  Full write-up: `docs/BENCH_1V1.md`; evidence: `bench_1v1/`.

- **H1, vs AlphaBetaPlayer (primary):** 302/400 = **75.5 %** (95 % CI 71.0-79.6 %), p = 1.4e-25 against 50 %.
  Average VP 9.05 vs 6.46.
- **H2, vs ValueFunctionPlayer (secondary, gated):** 293/400 = **73.2 %** (68.6-77.5 %), p = 2.1e-21.
- **HexMachina's reported 54.1 %** is below our whole 95 % interval, so our rate is higher than their reported
  point estimate.  Their format is unconfirmed, so this is a side-by-side of two numbers, not a test, and we
  do not claim to be better than HexMachina.
- **Clean run:** 0 replay mismatches in 800 games, 0 crashes, 0 turn-cap games.  AlphaBeta's slowest decision
  took 0.73 s against its 20 s deadline.

## 2026-09-26 09:20 UTC - first queue results: player trading is worth about 24 points

- **Trading headroom** (`t1_trades0_vrule@value`, 2,000 paired seeds, vs 3x
  Catanatron ValueFunction whose seats answer our offers with a value rule:
  accept if their own value rises):
  - our bot wins **84.3 %** with its trade proposals on;
  - switching proposals off costs **-24.1 +- 1.2 points** (p = 6e-83);
  - without trading, our first extra settlement comes 3.6 rounds later and
    our first city 1.5 rounds later.

  So trading is the biggest lever measured so far.  Caveat: value-rule
  responders accept any deal that helps them.  Humans are warier, so the
  size of this edge against people is unknown; the direction is not in
  doubt.
- **Surplus dumping knockout** (`t2_dump0`): removing it changes -0.8 +- 0.8
  points at 400 pairs.  KEEP (unproven either way; the default is unchanged).
- **Common random numbers** (the same dice for both arms of a pair) cut
  discordance from 0.33 to 0.20-0.22, so rows marked `crn: auto` now use
  them.
- **Both A/A checks PASS** (identical games).

The trading step's rows (player-trade premium, wider offers, calibration)
run in code epoch B1 from 09:10 UTC.

## 2026-09-26 09:00 UTC - trading (step 3) built; the hand-value idea failed its pre-registered check

All switches are off by default, and the default bot's pinned game digests
are unchanged.  255 tests pass on catanatron 3.3 and 247 (+9 skipped) on
3.2.1.

| feature | zero-game result | status |
|---|---|---|
| hand value from conversions and roll odds (`acq=1/2`) | **failed Stage 0**: 8 % of decisions change, mostly "bank trade -> end turn" to wait for cards it had only about a 20 % chance to roll before its next turn (1 of 13 such waits was reasonable; the rule needs at least half) | stopped, no games |
| moderate version, roll odds only (`acq=3`) | 4 % change; one bad wait on held-out games | stopped; becomes advisor text ("keep: 58 % to roll the ore") |
| player-trade premium (`acq_floor`, the user's rule) | 1.6 % change: 25 accepts became rejects, 5 proposals became bank trades | queued: self-play, and vs Catanatron with value-rule answers |
| wider trade offers (`acq_breadth`: 5 proposals, mixed 2-for-1 offers) | 9.1 % change; injected offers were accepted 2 of 14 times | queued: self-play |
| learning who accepts (`CALIB_RATE`) | prediction error (Brier) 0.31 -> 0.15 | politics tier: native check, then one screen |
| port trade-flow model (`acq.flow`) | per-resource error 0.69 vs 1.27 for flat shares | a data product for the ports step |

**Why the hand-value idea failed.**  It credits surplus cards as if they
were already converted.  So a bank trade that does not finish a build this
turn looks worthless, while the static value still pays for every card held.
The bot then holds cards instead of trading, which exposes them to the
robber for a round.

**The player-trade premium measures a real leak.**  With the premium off,
the default bot accepted trades that fail the rule in 6.7 % of accepts
(0.6 % of proposals).  Those are trades that our own bank or port would have
given us without helping the partner.  Trades literally worse than our port
rate by card count never occurred, because the engine only offers 1-for-1
and single-resource 2-for-1 deals.

## 2026-09-26 07:20 UTC - port-aware diversification built; the opening is the real lever

Built off by default (step 2 of the plan and the user's diversification
request):
- **Corrections hub** (`catanbot/corrections.py`): one place where extra
  scoring terms are added on top of the C++ static values.  Win-path
  results are bit-identical before and after the refactor, and the default
  bot's pinned game digests are unchanged.
- **Conversion-cost term** (`catanbot/conversion.py`, spec key `conv=1`).
  Every missing resource costs 3 / 2 / 1 extra cards per needed card
  (bank / 3:1 port / 2:1 port with a surplus), times the rolls left.  It is
  priced at 0.12 points a card, the static value's own card price.  With no
  fitting, the model reproduces the logs:

  | | model | measured |
  |---|---|---|
  | extra cards a game at 4:1 | 12.5 | ~15 |
  | saved by a 3:1 port | 4.2 | ~5 |
  | saved by a 2:1 wheat port | 3.2 | 3.6-3.8 |

- **`conversion` opening policy** for setup placement.

**Zero-game comparison** (1,997 decisions from 60 proof games; the A/A
control changed 0):

| arm | decisions changed | CPU cost |
|---|---|---|
| `conv=1` | 1.1 % | 1.01-1.04x |
| `conv=1` + road credit | 2.4 % | 1.06-1.09x |
| `paths=1` | 11.1 % | 1.23x |
| `conv=1` + `paths=1` | 12.0 % | 1.24-1.27x |

- `conv=1` changes *which* city or settlement, almost never the *kind* of
  action.  A one-turn term cannot make the bot save cards for next turn's
  settlement.
- With `paths=1` the two are additive, with no double counting; the extra
  roads come from the win paths.
- The opening policy is the strong lever.  It changes 56 % of our setup
  settlements:

  | our opening | current | conversion policy |
  |---|---|---|
  | resource types (proof positions) | 3.78 | 4.28 |
  | resource types (200 random boards) | 4.07 | 4.62 |
  | five-type openings | 26 % | 62 % |
  | total pips | 20.2 | 18.9 |
  | on a port | 7 % | 13 % |

**Queue:** these rows are now enabled in `scripts/queue_plan.json`:
- the `conversion` policy in the openings rows;
- `div_lr_bundle` (`conv=1` + `paths=1`, bundle first);
- `ports_conversion_cost`, with the road credit as its stronger fallback.

216 tests pass on both Python environments (hub, conversion, search,
winpaths, openings, pinned default digests, C++ parity, queue).

## 2026-09-26 06:30 UTC - budgeted test queue built (step 1 of docs/PRIORITY_PLAN.md)

`scripts/run_queue.py` runs the user's testing policy (docs/QUEUE.md).
- **Verdict engine** (`scripts/seqtest.py`): sequential verdicts at 5 looks:
  - ADOPT (then the league gate);
  - REJECT (clearly bad, or clearly too small);
  - SHELVE (unclear at the cap);
  - for knockouts, REMOVE or KEEP;
  - NOOP when the feature never fired.
- **Scheduler:** in area order (trading > diversification > ports > robber
  > card counting > politics > other).  A higher row preempts the running
  one only when that is worth the switching cost.  Costs are counted in CPU
  seconds.
- **Fallbacks and bundles:** fallback rows open when their parent is
  shelved or rejected.  Bundle rows are tested first, and their knockouts
  open only after an ADOPT.
- **Snapshots:** every row is pinned to a frozen copy of the code (an
  "epoch").
- **Measurements:** per-arm behaviour metrics (`scripts/mechanics.py`), and
  a zero-game decision comparison (`scripts/decision_shadow.py`) that gates
  rows unlikely to change any decision.

Error rates verified by simulation:

| design | case | outcome |
|---|---|---|
| screen | null | false ADOPT 2.1 %, about 980 of 2,000 pairs used |
| screen | +4 points | ADOPT about 73 % |
| knockout | null | false REMOVE 2.1 % |
| control-variate estimator | +3 points | power 0.56 -> 0.74 |

The behaviour metrics reproduce the known gaps on 400 proof games:

| | ours | Catanatron's |
|---|---|---|
| resource types | 3.85 | 4.67 |
| share settling before the first city | 0.36 | 0.63 |
| 4:1 share of bank trades | 0.76 | 0.54 |
| port settlements | 0.59 | 0.71 |
| income under the robber | 0.06 | 0.09 |

155 tests pass on both Python environments.  About 31.7 CPU-hours (about
10.5 h on 3 cores) if every effect is zero.

It starts once the corrections hub and the conversion-cost term have landed
and passed their "default bot unchanged" tests.  Epoch A is copied from the
repository at the first game, so it must not include half-finished code.

## 2026-09-26 05:00 UTC - why our bot goes city-first (and the 4:1 habit)

The user asked why the bot builds cities first, while strong humans expand
with roads for resource diversity and so avoid bank trades.

**Measured in the proof games** (T1, T2, R1; openings from 400 T1 games):

| | our bot | Catanatron's bots |
|---|---|---|
| opening production, wheat + ore | 11.4 pips (wheat 6.3, ore 5.1) | 7.3 |
| opening production, wood + brick | 6.8 | 8.1 |
| opening production, sheep | 2.6 | 4.6 |
| distinct resources produced after setup | 3.85 | 4.67 |
| first settlement after setup, median round | 12 | 7-9 |
| first city, median round (T1 / T2) | 8-9 | 11-13 |
| settles before its first city | 35-41 % of games | 56-62 % |
| extra settlements per game | 1.6 | 2.1 |
| cities per game (T1) | 1.9 | 1.3 |

So the bot plays an ore/wheat city strategy: few resource types, few
settlements, and surplus wheat and ore traded 4:1 for wood, brick and sheep.
That is where the 75-78 % 4:1 share and the low port share come from.

**Why the code does this:**
- Resource weights favour wheat and ore (`placement.RESOURCE_DEMAND`: 1.25 /
  1.2 against 1.0 for wood and brick and 0.9 for sheep).
- The search looks one turn ahead, so a road only earns the small reach
  credit in the static value: 0.6 per spot buildable now, and 0.12 x the
  best reachable spot / (1 + 0.9 x distance), about 0.6-0.9 points per
  road.  A city earns its +1 VP at once.
- Diversity earns 0.4 per resource type produced.

It beats Catanatron, and ore/wheat cities is a real human strategy.  But
here it is the default on every board, not a choice made per board, and
against people who race for spots it may leak.

**Test** (ports area, docs/ABLATIONS.md policy): the existing knobs first.
- The `pips_diversity` and `standin_book` opening policies against the
  setup-pick control.
- `placement.RESOURCE_DEMAND` flat, and leaning to wood and brick.
- Then a stronger multi-turn settlement-plan credit for roads, still to be
  built.

Each is read both by win rate against Catanatron and by the mechanism:
diversity, round of the first settlement, 4:1 share, ports.

**Does the robber punish the city-first style?**  Not measurably.  At turns
40, 60 and 80 of 200 T1 games:

| | our bot | Catanatron's bots |
|---|---|---|
| best single hex, share of income | 28.4 % | 29.2 % |
| hexes worked | 6.3 | 6.7 |
| income under the robber at that moment | 7.3 % | 9.2 % |

The robber blocks one *hex*, not one *resource*.  Our cities stand on
separate hexes, so income is spread about as widely as Catanatron's.  The
blockability penalty charges only for stacking our own buildings on the same
hex, and here it works: we are not more blockable.

What the style lacks is resource *types* (3.85 vs 4.67), which the robber
does not act on.  Type diversity is set by the resource weights and by the
one-turn horizon, which are what the queued expansion tests change.

## 2026-09-26 04:30 UTC - strength proof complete: all four claims PASS, ready for supervised human testing

T7-T11 finished at 04:10 UTC.  With T1-T6 and R1-R2 that is 7,600 games,
with 0 errors, fallbacks, crashes, turn-cap games or card-counter resets.
Every game replays exactly.

| claim | tests | result | largest Holm-adjusted p |
|---|---|---|---|
| 1 better than Catanatron's strong bots | T1-T6 | PASS | 1.85e-32 |
| 2 ready for supervised human testing | T1-T6, R1-R2 | PASS | 1.85e-32 |
| 3 strong with only Colonist information | T7 61.7 %, T8 58.5 %, T9 56.8 % (fair 25 %) | PASS | 1.87e-41 |
| 4 beats a mixed Catanatron table | T10 61.0 % (full), T11 61.8 % (Colonist) | PASS | 3.13e-52 |

**Readiness for supervised human testing** (claims 2, 3 and 4): **PASS.**

* **Colonist information costs almost nothing against these bots.**  Win
  rates stay within noise of the full-information runs: 61.7 vs 62.8 %
  against ValueFunction, and 58.5 / 56.8 vs 54.5 % against the search bots.
  Our bot's time per game rises about 3.4x (4 guesses of the hidden cards
  per decision).
* **Mixed table:** SameTurnAlphaBeta was the strongest Catanatron seat (60 /
  71 wins in T10 / T11), then AlphaBeta (54 / 58), then ValueFunction
  (42 / 24).
* **Two container restarts** each cut off one chunk (T7 games 800-999, T9
  games 120-159).  Both chunks were discarded and replayed in full with the
  same seeds, and all 39 games finished before the cut came out identical.
  This is disclosed in docs/PROOF.md.
* **Timing:** Catanatron's slowest decision in T8-T11 was 7.4 s, against
  its 20 s search limit.
* **Evidence** archived in `proof/`: sha256 manifest of 312 files, replay
  checks, and the registered analysis re-run on the archive with identical
  output.

**Next** (docs/ABLATIONS.md testing policy):
- the budgeted test queue with the priority areas: trading, ports, robber,
  card counting;
- the new strategies through the league gate.

## 2026-09-26 02:00 UTC - joint tuning (SPSA) and 2x2 interaction tools built; what the budget can detect

Tools, all off the default path:
- `scripts/tune_joint.py`: SPSA over groups of weights, in self-play or
  against Catanatron.  `--max-games` is required, and the planned total is
  printed before any game.
- `scripts/ablate.py --factorial` and `scripts/factorial_catanatron.py`: 2x2
  (up to 2^4) tests reporting main effects and interactions.
- `docs/TUNING.md`: how to run them.
- A tuned set reaches the league gate as a `tune=` bot spec.

64 tests pass on both Python environments.  So far only smoke-sized real
runs have been played.

**What a budget of thousands of games can and cannot see** (measured on a
synthetic objective with a known optimum):

| method, budget | what it can detect |
|---|---|
| SPSA, 2,376 self-play games | a group's weights only when their combined edge is about 5 points: it recovers +3.8 on average, never worse than the defaults |
| the same, combined edge of about 2 points | +0.6 on average, and 4 of 12 runs ended *worse* than the defaults |
| 2x2 self-play test, 800 games per cell | main effects of about 4 points and interactions of about 9 points |

So only large effects show at this budget.  The joint tuner is judged as a
whole by the league gate, not per weight.

**Throughput correction** (it changes the time estimates):

| mode | games per hour on 3 cores |
|---|---|
| vs Catanatron, C++ evaluator | about 6,000 (1.1 s per game) |
| vs Catanatron, Python evaluator (needed for weights inside the static value) | about 3,000 (3.6 s per game) |
| self-play, 4 search bots, C++ evaluator | about 1,000 (11.5 s per game) |
| self-play, Python evaluator | about 170 (62 s per game): 2,400 games take about 14 h |

Consequence: test new ideas against Catanatron wherever the mechanic
exists there (bank and port trades, ports, robber, card counting in
counted mode).  Keep self-play for player trades.  Port static-value
weights to C++ with a parameter rather than paying the Python-evaluator
cost.

## 2026-09-26 01:00 UTC - advisor counts cards from Colonist's game log (off by default)

The advisor used to read one screenshot at a time, so it knew opponents'
hand *sizes* but not their cards.  Now, with `--session FILE`, it keeps a
running card count across calls from Colonist's public log.
- **How the log gets in:** the Claude-vision parser can read the log panel,
  or you can paste the log with `--game-log FILE|-`.
- **What it counts:** rolls and production, builds, bank and port trades,
  player trades, offers, Monopoly, Year of Plenty, and steals and discards
  as counts.  Overlapping screenshot windows are lined up so no entry
  counts twice.
- **Checks:** after each update the count is checked against your hand,
  every hand size and the bank (when visible).  A mismatch is reported and
  resynchronised.
- **Where it is used:** the search samples opponents' hands from the count,
  using the same code as the proof's Colonist-information mode.
- **New output:** a "Card count" section.  Example: `orange (Carol): 1 wood,
  2 ore certain; 1 card uncertain from hidden steals with green, blue:
  wood 88% / sheep 12%`.

**Evidence:**
- 94 new tests.  In simulated games rendered as Colonist-style log text and
  fed in overlapping windows, the true hands were always among the counted
  possibilities.  They were exact whenever no hidden steal or discard was
  pending, over 30 seeds with the bank visible.
- Without the new options the advisor output is unchanged.
- Tests pass on both Python environments.

**Must be checked against a real Colonist game:** the log wording.  It is
written from memory, since Colonist can't be reached from here.  The list
of phrasings to verify is in docs/USAGE.md ("Card counting from the game
log").

**Known limits:**
- With no visible bank, several big hidden discards on one 7 can overflow
  the 4,096-hypothesis cap; this happened in 2 of 6 test seeds.  A visible
  bank fixes it.
- A missed log line is caught by the hand-size check, but repaired with a
  production-weighted guess.
- Only the search uses the count so far; the trading, robber and
  offer-response sections still use their own estimates.

## 2026-09-26 00:40 UTC - strength proof: claims 1 and 2 PASS

The pre-registered proof (docs/PROOF_PROTOCOL.md) finished T1-T6 and R1-R2:
5,000 games, 0 errors / fallbacks / crashes / turn-cap games, and all 5,000
replay action for action.

| test | opponent | format | our wins | one-sided p | 99 % CI lower end |
|---|---|---|---|---|---|
| T1 | 3x ValueFunction | 1v3 (fair 25 %) | 628/1000 = 62.8 % | 3.8e-140 | 0.588 |
| T2 | 3x AlphaBeta | 1v3 | 218/400 = 54.5 % | 2.9e-36 | 0.479 |
| T3 | 3x SameTurnAlphaBeta | 1v3 | 218/400 = 54.5 % | 2.9e-36 | 0.479 |
| T4 | 2x ValueFunction | 2v2 (fair 50 %) | 836/1000 = 83.6 % | 2.5e-109 | 0.804 |
| T5 | 2x AlphaBeta | 2v2 | 323/400 = 80.8 % | 3.2e-37 | 0.752 |
| T6 | 2x SameTurnAlphaBeta | 2v2 | 315/400 = 78.8 % | 1.9e-32 | 0.730 |
| R1 | 3x our stronger ValueFunction stand-in | 1v3 | 278/1000 = 27.8 % | (two-sided 0.045) | - |
| R2 | 3x our stronger AlphaBeta stand-in | 1v3 | 103/400 = 25.8 % | (two-sided 0.73) | - |

* **Claim 1** (better than Catanatron's strong bots): PASS; the largest
  Holm-adjusted p-value is 1.85e-32.
* **Claim 2** (ready for supervised human testing): PASS on every condition.
  * T1-T6 significant at 5.7e-7; they also clear a strict one-sided
    5-sigma, 2.87e-7.
  * Effect sizes above the registered floors.
  * Every seat above 25 %: the worst seat p-value is 2.1e-8.
  * Zero errors.
  * Not significantly below 25 % against our stronger stand-ins.
* **Readiness for human testing also needs claims 3 and 4** (amendments 2
  and 3).  T7-T11 started at 00:25 UTC from the amendment-5 snapshot.
* **Evidence archived** in `proof/` (README, sha256 manifest).  The registered
  analysis re-run on the archived copy prints output identical to the run's.
  Written up in docs/PROOF.md; skeptic's Q&A in docs/SCRUTINY.md.
* **Against the stand-ins** our bot is about even (27.8 % and 25.8 %).  In
  both, the first seat did best (36-38 %).  So a stronger version of the
  same bots is roughly our level.  This is the headroom to watch as
  Catanatron-style opponents get stronger.

## 2026-09-25 23:00 UTC - counteroffers and out-of-turn offer analysis built (off by default)

* **Counteroffers** (Colonist rule, `allow_counters`, off by default): when
  someone offers a trade, each other player may accept, reject or counter
  once; the offerer then sees the counters one by one and may take one (the
  trade happens and the round closes) or reject it.  Counters do not use up
  the 4 offers per turn and a counter cannot be countered.  The bot's counters
  are small edits of the offer (ask one more card, give one fewer, swap a
  card), never ones that feed the leader, ranked by the chance the offerer
  takes it x our gain.  A 0.002 win-probability margin keeps it from
  countering everything (without it: 100-140 counters a game, ~10 % taken;
  with it: 2-5).  The predicted chance a counter is taken matches self-play
  (predicted 9-17 %, observed 9-16 %).
* **Out-of-turn offer analysis** (`resp_la=1`): accept, reject and each counter
  are valued after the offerer's turn is played out.  Example test position:
  the plain view says accept (0.277 vs 0.268); after the offerer's turn it
  says reject, because the brick lets them settle on the spot we are heading
  for.
* **Card counting from offers**: every offer and counter is public, and you
  can only offer cards you hold, so an offer rules out hands without those
  cards and hints that the offerer lacks what they ask for.
* **Advisor**: `recommend --offer` adds an "Offer response" section (best
  answer, value of accept / reject / up to 3 counters before and after the
  offerer's turn, and why).
* **Safety**: off by default; six fixed-seed default games replay
  byte-identically against the code before the change (two pinned as a test);
  the C++ engine refuses counter-rule states (they run in Python).  Tests
  pass on both Python environments.
* **Smoke only** (20 games, noise): counter bots made 2-5 counters a game,
  ~10 % more time per decision; answering an offer with `resp_la=1` takes
  ~3 ms (p95 10 ms).
* **Next, after the proof**: paired self-play ablations (1,200 games each for
  counters and for `resp_la`), then the league gate for `resp_la` (the league
  plays standard rules, so counters themselves are judged by the ablation).
  Catanatron's bots cannot trade, so neither can be measured against them.

## 2026-09-25 22:40 UTC - win-path portfolio built (off by default, not yet benchmarked)

The strategy the user asked for: weigh every route to 10 VP (Longest Road,
Largest Army, cities, settlements, VP cards) by how *crowded* it is - who else
is racing for it, from their production, ports, hand and progress - and only
spend on a race we can win.  Design chosen by a 3-design judge panel, then
implemented and independently reviewed (`catanbot/winpaths.py`,
docs/STRATEGY.md, docs/ABLATIONS_WINPATHS.md).

* **Model**: Longest Road and Largest Army are races; each seat's projected
  level (current length / knights + cards in hand + income-driven growth over
  the expected remaining game) gives a win probability per race.  A path's
  value is prize x P(win) minus the cards still needed to beat the strongest
  rival, with the option to quit (never below the passive value), replacing
  the heuristic's fixed award credit.  Optional (off): races for the same
  settlement spot, timed by each seat's income and turn order.
* **Calibration on 80 self-play games** (log-loss on who holds the award at
  the end): Longest Road 1.08 vs 1.41 for "the holder keeps it"; Largest
  Army 0.90 vs 1.29.  The current heuristic assumes an award holder always
  keeps it; in these games a Longest Road holder with a 1-road lead kept it
  only 61 % of the time.
* **Cost**: 1.25x per decision (1.64x with the spot races).
* **Safety**: off by default (`paths=1` in a bot spec switches it on); the
  review rebuilt a pristine package and showed the default bot plays
  identical games and searches with the module present.  The review also
  found and fixed three defects (two stale cache keys, a double-counted port
  conversion), each with a test that fails on the old code.  32 tests pass on
  both Python environments.
* **Next** (pre-registered in docs/ABLATIONS_WINPATHS.md, after the proof):
  calibration, paired games vs Catanatron's ValueFunction (2,000 seeds per
  arm), crowding / weight sweeps, knock-outs, a held-out confirmation, then
  the champion league gate before it can become the default.

## 2026-09-25 22:30 UTC - harness audit closed: it does not weaken Catanatron's bots

The last two controls:

| control | games | result | reading |
|---|---|---|---|
| C2b: AlphaBeta vs 3x ValueFunction in a plain Catanatron game (none of our code, Catanatron's own seat shuffle) | 100 | 32 % +- 4.7 | |
| C4: our bot fully blind (no hands, no dev cards, no counting) vs 3x AlphaBeta | 60 | 41.7 % +- 6.4 | fair share 25 %; the win does not come from seeing cards |

C2b (32 %) against C2 in our harness (20 %) looked like the harness might
weaken AlphaBeta, which would inflate our results against it.  It does not:

* **Move-for-move check** (`scripts/audit_reseat.py`): 8 seeds, AlphaBeta + 3
  ValueFunction seated in an order Catanatron's shuffle would not pick, played
  three ways in one process - Catanatron's own `Game` with its shuffle forced to
  that order, our `make_game` re-seat, and our re-seat with every player in the
  `BenchOpponent` wrapper.  Initial states identical field by field (hands,
  decks, board, RNG state, legal actions), and all 2,683 actions identical
  including every dice roll, steal and dev draw; same 8 winners.  The harness
  *is* Catanatron's game.  A fast version with random players runs in the test
  suite on both Catanatron versions
  (`test_reseating_equals_catanatron_native_seating`).
* **The gap is noise**: 20/100 vs 32/100, Fisher exact p = 0.076.  Pooled,
  AlphaBeta wins 52/200 = 26 % (95 % CI 20-33 %) against three ValueFunction
  players: in a 4-player game it is about as strong as ValueFunction.
* **No time truncation**: AlphaBeta stops searching at 20 s per decision;
  its slowest decision in the proof logs is about 6 s, so CPU load does not cut
  its search short.

Side note: the same seed gives a different game in a different Python process
unless `PYTHONHASHSEED` is fixed (Catanatron iterates over sets); every proof
and benchmark run pins it (`--hash-seed`), and the three arms above share a
process.

Verdict: the results against AlphaBeta and SameTurnAlphaBeta stand.

## 2026-09-25 22:00 UTC - Colonist-information mode and champion league landed

**Colonist-level information for our bot** (`--info counted`): opponents'
hands come from a card counter fed only by public events (production,
builds, trades, Monopoly / Year of Plenty, bank); hidden: steals seen by
neither thief nor victim, discard types, unplayed dev types (drawn from the
public pool).  Evidence: on 20 replayed games per engine (11,103 and 12,428
log steps) the true hands were always among the counter's hypotheses (mean
probability 0.76-0.80), each opponent's hand was known exactly 73 % of the
time, at most 44 hypotheses; with every hidden event revealed the counter is
exact at every step; swapping hidden cards between opponents never changed a
counted-mode decision (a full-information search changed at 18-20 of 20
positions).  The default full mode replays byte-identical games.  Cost: 3.4-4.2x
per decision (4 determinizations).  Mixed Catanatron tables
(`--mixed-opponents value,alphabeta,sameturn`) are in the benchmark.
Proof tests T7-T11 (amendments 2 and 3) are being wired into the analysis
before any of their games are played.

**Champion league** (`scripts/league.py`, docs/LEAGUE.md): champion-0 = the
proof bot (9984181).  Each champion plays with its own code (a frozen export
of its commit, served over a JSON protocol), candidates play 2v2 tables
against every champion, and promotion needs a significant win over the
current champion (one-sided p < 0.01, exact binomial with O'Brien-Fleming
style early stopping; simulated type-I error 0.0093, power 86 % at a 55 %
share) and no significant loss to any earlier one (Holm).  A candidate
identical to a champion wins exactly half of every block (built-in harness
check).  About 290 games per hour per gate on this machine; a gate needs at
most 1188 games per champion.

## 2026-09-25 21:20 UTC - fairness audit of the Catanatron benchmark

The user asked whether the wins could come from a harness bug or a leak.
Code audit: dice, steals and dev draws are Catanatron's own (our bot has a
separate RNG and never reads Catanatron's); the dev deck is exposed as
composition, not order; every action goes through Catanatron's validated
execute; opponents' decisions pass through our wrapper unchanged; the
re-seating happens before the first action when all seats are identical.
One real caveat: in T1-T6 both sides see every hand and dev card
(Catanatron exposes the full state; its bots read it too), which is fair
between the sides but is not the Colonist game - hence the pre-registered
counted-information tests T7-T9 and the mixed table T10-T11.

Controls through the same harness (non-proof seeds, 1 process):

| control | games | result | reading |
|---|---|---|---|
| C1: Catanatron ValueFunction in our rotated seat vs 3x ValueFunction | 400 | 23.5 % +- 2.1 | the harness favours no seat (fair = 25 %) |
| C2: Catanatron AlphaBeta in that seat vs 3x ValueFunction | 100 | 20 % +- 4 | AlphaBeta is not stronger than ValueFunction here |
| C3: our bot fully blind (opponents' cards replaced by random guesses, no counting) vs 3x ValueFunction | 300 | 56.7 % +- 2.9 | vs 62.8 % with full information: the view is worth ~6 points, not the win |

Pending: C4 (blind vs 3x AlphaBeta) and C2b (AlphaBeta vs ValueFunction in
a plain Catanatron game without any of our code, to separate "AlphaBeta is
just not stronger" from "our harness weakens it").  Both done: see the 22:30 entry.

## 2026-09-25 19:55 UTC - pre-registered strength proof launched

Protocol: docs/PROOF_PROTOCOL.md (pre-registered in 704849b, tooling
amendment only appended; the original text is byte-identical).  Tooling
verified (exact binomial tails equal to an independent rational
implementation, Clopper-Pearson against exact bisection and scipy, Holm,
every claim condition on synthetic data, chunked runs reproduce an
uninterrupted run game for game, replay of every logged game re-applies
exactly; full suite 550 / 561 passed on catanatron 3.2.1 / 3.3.0).  The
proof runs from a frozen git worktree of commit 9984181
(/home/user/proof_snapshot, its own C++ build), so development elsewhere
cannot change the bot under test.  About 4.3 h at 3 workers; every game is
logged (about 2.7 KiB per game compressed) and can be replayed.

Disclosed readings of the protocol, fixed before any proof game: alpha
5.7e-7 is used as registered (the text calls it one-sided 5-sigma; it is
the two-sided 5-sigma tail, i.e. stricter than one-sided) [correction
2026-09-25 23:45 UTC: wrong way round - as a one-sided threshold 5.7e-7 is
4.87 sigma, slightly *less* strict than one-sided 5 sigma (2.87e-7); the
registered number stands, see docs/SCRUTINY.md Q14]; condition 5
fails only when the bot is significantly *below* 0.25 against a stand-in
(its heading), not when it is significantly above; the 99 % bounds are the
lower ends of the central 99 % intervals (conservative).  Heads-up from
non-protocol timing runs (20 games each): SameTurnAlphaBeta was the
hardest Catanatron opponent (6/20), so claim 1 is at risk on T3.

## 2026-09-25 17:45 UTC - Catanatron test harness built; three facts that shape the proof

* **Catanatron's bots cannot trade.**  Its ValueFunction player rejects every
  offer (accepting moves no card in its own evaluation, so the tie keeps
  REJECT), and its AlphaBeta / SameTurn players crash on every offer
  (RuntimeError on REJECT_TRADE).  Trade-driven strategy terms can only be
  tested against Catanatron with our own opponent response rule (a seat
  accepts if its Catanatron value function improves), which is labelled as
  such; the pre-registered proof plays with trading off, as the protocol
  prescribes.
* **Compute is not in our favour against the strong bots.**  Our bot spends
  8-12 ms per decision (0.6-1.0 s per game); Catanatron's AlphaBeta spends
  about 105 ms per decision (8 s per seat per game), SameTurn 7 s, and our
  AlphaBeta stand-in 2.2 s.  Only Catanatron's ValueFunction thinks less
  (0.23 s).  The ~63-66 % results against AlphaBeta / SameTurn are with ~10x
  less thinking time.
* **Games are now reproducible.**  The benchmarks pin PYTHONHASHSEED; the same
  seed gives the same game in any process, so paired comparisons and exact
  replays are possible (earlier results were unpinned).

Also found: our bot keeps proposing trades to players who never accept (about
59 offers per game against Catanatron with native trading, all rejected) and
cancels about a quarter of accepted offers: an opponent-model calibration
issue, recorded for later.

Proof tooling (2v2 mixed tables, replayable action logs, exact-test analysis
with the protocol's verdicts, chunked runner) is being built and verified;
the proof games run right after.

## 2026-09-25 17:10 UTC - boards now fully random (port types were fixed)

Our engine shuffled tiles and number tokens per game but always put the
same port type at each of the 9 harbour slots, so every self-play game,
every self-play ablation and the whole value-net replay buffer shared one
harbour layout (an overfitting risk for the net and for port-related
terms).  `new_game(rng=...)` now also shuffles the 9 port types over the
standard slots (as Catanatron's base map does); the shuffle is seeded from
the random tiles, so a seed's dice stream is unchanged.  200 seeds give 200
distinct harbour layouts.  The Catanatron benchmarks were unaffected
(Catanatron already randomises ports).  Self-play ablation results so far
were measured on the fixed layout; the Catanatron campaign is the one to
trust.  Also found and fixed: the installed C++ extension was older than
its sources (a verification agent had built the depth-3 budget fix into a
staged file and never installed it); rebuilt.  Full suite: 474 passed, 14
skipped.

## 2026-09-25 16:30 UTC - five trade proposals vs three (200 paired games)

Five proposals per node instead of three: -3.0 pp (s.e. 3.5), +1.9 ms per
decision.  More proposals do not help; the default of three stays.  Next:
a paired campaign against catanatron's strong bots with thousands of games
per term (harness, 3.3 trade wiring and opening variants being built).

## 2026-09-25 16:00 UTC - late-game trade damping follow-up (400 games)

The weak +4-5 pp signal from sweep 2 did not hold: stage drop 0.0 is +0.8
pp and 0.4 is -3.5 pp (s.e. 2.5) against the default 0.7.  Default kept.

## 2026-09-25 15:40 UTC - lookahead fix verified: parity at best; depth 1 is the default everywhere

The de-noising fix landed (every end-of-turn node gets the lookahead, 12
stratified roll samples, no clamp inside the backup, reliability shrinkage
k = 12, all-or-none depth-3 sub-search; bitwise native parity, 60 search
tests).  Adversarial verification on new seeds and pooled over 840 / 360
games: fixed depth 2 is -1.7 +/- 2.1 points vs the ValueFunction stand-in
and -8.3 +/- 3.1 points vs the AlphaBeta stand-in relative to depth 1, at
4-5x the decision time, and it still flips 16 % of its own decisions by
dice sample.  Verdict: with the heuristic evaluator the sampled
opponents'-turn lookahead is a noisy correction with a signal about a third
of the static spread; it is not a strength lever.  Defaults now: advisor
`--depth 1`, self-play depth 1, bot specs `lookahead=0,opprolls=12` when
depth 2 is requested, depth 3 off.  Depth 2 remains useful for the advice
text (what each opponent can do to you next round).

What this means for the ML plan: the search cannot currently improve on the
evaluator, so iterated self-play (search-improved targets) has no
improvement operator to amortise; the remaining ML experiment (task #15,
end-of-round targets + hold-epsilon) has a small expected gain and is
parked for the user's decision.  The measured strength levers are
elsewhere: trade proposals (+17 pp), openings (the stand-ins' opening book
beats our setup picks from seats 2-3) and trade generation - all tunable
with the ablation harness, which should get a mode that pairs a tunable
against the Catanatron stand-ins.

**Opus switch notice sent at 15:40 UTC** (all implementation pieces are
integrated and verified; the remaining default work is benchmarking and
tuning).  Not to be repeated.

## 2026-09-25 13:40 UTC - ablation sweep 2 (six more tunables)

Trade proposals inside the search are the most valuable strategy term
measured: switching them off costs 17.5 pp (s.e. 4.3) for a saving of
about 12 ms per decision; one proposal instead of three costs 5 pp.  Weak
signals for follow-up at 400 games: late-game trade damping possibly too
strong (+4-5 pp when weakened), blockability weight on the right side
(-2 pp when removed).  Knight value, dump candidates, feed-the-leader
guard: neutral.  All 11 tunables of the first two sweeps: docs/ABLATIONS.md.

## 2026-09-25 11:45 UTC - lookahead diagnosis: noise, not strategy

The 15 % figure was 40-game noise; over 400 games vs the ValueFunction
stand-in depth 2 scores 24.8 % (7.13 VP) vs depth 1 26.2 % (7.34 VP), and at
the same table over 528 seats each 22.9 % vs 29.2 % (7.57 vs 7.81 VP): a
consistent small deficit for 1.5-2x the compute.  Cause, measured on 200
decision nodes: the sampled opponents'-turn correction has a per-sample
noise of 0.028 win-prob between two candidates (chaotic divergence of the
greedy opponents even with common dice), so with 4 roll samples the error
is 0.014 while the margin between the best two actions is 0.002-0.008; the
lookahead flips coin-flip decisions (30 % of main-phase top choices, 15 % of
offer answers), and rollouts rate those flips as neutral.  Two mechanisms
turn the noise into bias: only the top-4 end-of-turn nodes get the
lookahead (horizon mixing: identical candidates valued differently by
membership) and the [0,1] clamp in the backup.  Hypotheses ruled out by
measurement: hoarding / pessimistic opponents (END_TURN with an affordable
build 0.0 % at both depths), unrealistic opponent policy (oracle and
END_TURN-only opponents change nothing), node budget, native vs Python,
chance-node leaks.

Fix being implemented: lookahead for every finished node, 12-24 roll
samples with stratified / antithetic sequences, no clamp inside the backup,
reliability shrinkage of the correction, cheap rolls-only opponents by
default (the clever greedy opponents add cost but not accuracy), depth >= 3
off by default.  De-noised depth 2 measured at 26.7 % / 7.35 VP vs vf
(= depth 1).  Expectation to keep in mind: with the heuristic evaluator a
de-noised lookahead only reaches parity; a real gain needs an evaluator
whose error the simulation can correct, i.e. the value net trained on
end-of-round search targets (task #15).  Benchmark rule from now on: never
rank search settings on fewer than 400 games; report avg VP with the win
rate.

## 2026-09-25 11:35 UTC - first strategy ablation sweep (5 of 11 tunables)

120 paired games per candidate with the depth-1 search bot (2 vs 2 in the
same games): danger multiplier off +2.5 pp, danger BLOCK_NEED 0 / 1 +1.7 /
+3.3 pp, TURNS_HALF 1.5 / 2 -1.7 / 0 pp, coalition SCALE 0.25 / 1 +0.8 /
+1.7 pp, favour slack 0 / 0.15 -3.3 / -1.7 pp; standard error 4.6 pp, so
nothing is significant.  Full table: docs/ABLATIONS.md.  The remaining six
tunables (feed-the-leader guard, blockability weight, trade proposals,
dump candidates, knight value, trade-stage drop) run in a second batch.
Conclusion so far: the targeting / political terms neither help nor hurt
measurably in self-play among identical bots; they must be tested against
different opponents (stand-in ladder) before any is cut.

## 2026-09-25 09:55 UTC - value-net fix landed: regression gone, net at parity with heuristic search

Training-side fix (default on in `catanbot.train`): every generated game
also records sibling afterstates (main-phase builds, setup placements,
incoming offers as a deterministic counterfactual, robber, discards) and
the fit adds a ranking term that distils the heuristic's ordering of those
siblings (delta targets = heuristic logit gap, hold pairs weighted, offer
pairs weighted) plus a horizon-consistency term; old buffers still load
with `--resume`.

| net as search evaluator (depth 1, 2 seats vs 2 heuristic-search seats) | seat games | net | heuristic search |
|---|---|---|---|
| rejected candidate (outcome-only fit), blend 0.6 | 96 | 9.4 % | 40.6 % |
| fixed net, blend 0.6 (fix agent, seeds 5-6) | 96 | 22.9 % | 27.1 % |
| fixed net, blend 0.6 (independent, seed 11) | 48 | 25.0 % | 25.0 % |
| fixed net, blend 0.3 | 96 | 25.0 % | 25.0 % |

Honest reading: the catastrophic regression is gone (z = +2.8 vs the
candidate), but a net that distils the heuristic's sibling ordering cannot
beat heuristic search by construction: the self-play data has no
counterfactual for holding cards, and outcome labels reward states, not
actions.  The 40-game promotion tournament would promote such a net by coin
flip and the loop could then drift, so training is NOT restarted with this
recipe alone.

Path to a net that is actually better than the heuristic (next
implementation step, after the lookahead fix): search-distilled targets -
label recorded decision afterstates with the value of a deeper native
search (depth 2-3, now cheap) in addition to the outcome, so the net
amortises lookahead and depth-1 search with the net plays like depth 3;
plus a small hold-epsilon in the behaviour policy so the data contains
"hold vs build" counterfactuals.  Restart training then, with
`--eval-games 80 --promote-ratio 1.15` so parity nets are not promoted.

## 2026-09-25 08:50 UTC - lookahead regression found: depth 2 plays worse than depth 1

Stand-in ladder (catanatron 3.2.1, 40 games each, seed 5) with the depth-2
native search bot: 15 % vs ValueFunction (avg VP 7.0 vs 7.8) and 15 % vs
AlphaBeta (6.6 vs 7.5), against 24.5 % / 24.4 % for depth 1.  Together with
"depth 3 not better than depth 2" this says the opponents'-turn lookahead
systematically misvalues actions (search time is not the issue: 0.02 s per
decision).  A diagnosis-and-fix workflow is running (hypotheses: pessimistic
simulated opponents making visible progress look bad, horizon mixing, roll
sample noise, unrealistic opponent policy, chance-node handling).  Until it
lands the advisor, self-play and the ladders should use depth 1; the
native lookahead is still the right tool once the design is fixed.

## 2026-09-25 08:35 UTC - strong Catanatron ladder (catanatron 3.3.0) done and verified

Default bot (`search:depth=1,evaluator=heuristic`, C++ accel), 1 seat vs 3
copies of the opponent, seats rotated, 25 % = seat baseline.  Verified by an
independent re-run with new seeds (all rows compatible except sameturn,
whose pooled estimate over 146 games is 66 %).

| opponent x3 | games | win rate | avg VP us / them |
|---|---|---|---|
| VictoryPoint (control) | 100 | 100 % | 10.1 / 2.7 |
| ValueFunctionPlayer | 200 | 64 % (57-70) | 8.9 / 5.9 |
| AlphaBetaPlayer (depth 2) | 30 | 63 % (46-78) | 8.8 / 6.3 |
| SameTurnAlphaBeta | 60 (+86) | 66 % pooled | 9.1 / 5.9 |
| MCTS (10 simulations) | 40 | 100 % | 10.0 / 2.9 |
| GreedyPlayouts (25/action) | 2 | 2/2 (not informative, 15 min/game) | 10.5 / 3.8 |

Reading: against catanatron's three real search players the depth-1 bot
wins about two thirds of its seats (baseline 25 %), i.e. it is clearly the
strongest player at the table but not dominant; against our own stronger
in-engine stand-ins it is only at par.  Losses are economic (behind by
turn 40-60, openings), never mechanical (0 errors / fallbacks / stalls in
432 games).  Next: the same ladders at depth 2 with the native lookahead
(0.02 s per decision) and time-matched (task #9), then the value net.
Full tables and commands: docs/BENCHMARKS.md.

## 2026-09-25 07:20 UTC - usage-limit interruption, three workflows resumed

The session limit hit at ~06:40 UTC and cut three workflows short; it reset
at 07:20 and they are resumed from their cached prefixes.  What had landed:

* **Native C++ search lookahead** (`core.future_values`, opponents' turns +
  the depth >= 3 reduced sub-search + a native MLP/blend): bit-identical to
  the Python `_future_values` in parity mode (same dice, same candidate
  order), 101 search/accel tests passing.  Per move with the SearchBot
  configuration (heuristic evaluator): depth 2 0.045 s, depth 3 0.065 s,
  depth 4 0.17 s (Python: 0.6 s, 2.5 s, minutes); in games 0.02-0.04 s per
  decision.  Enabled by default when the extension is built
  (`SearchConfig.native_future`, `CATANBOT_NO_NATIVE_SEARCH=1` disables).
  Independent verification (07:40 UTC): own fuzz on 5138 states bit-identical in parity
  mode, 120/120 depth-2 searches identical, default-config agreement at the seed-to-seed
  noise floor, MLP within 5e-7, no undefined behaviour found; advisor default stays
  depth 2, depth 3 is an analysis option with a time budget.  Note: when the global node
  budget runs out at depth >= 3 the later leaves degrade to static values (uneven
  horizon), one candidate cause for the next item.
  **Open issue**: depth 3 is not stronger than depth 2 at the same table
  (20 % vs 30 % seat wins over 32 games, Python and native alike) - a
  search-design problem (reduced sub-search noise / horizon shift), to be
  investigated before deeper search is used for play.
* **Catanatron benchmark on both versions** (3.2.1 wheel and the 3.3.0
  checkout): adapter, script and tests pass on both (38+4 skipped / 41+1).
  Stand-in ladder on 3.2.1, default depth-1 heuristic bot, 3 copies of the
  opponent, seats rotated:

  | opponent x3 | games | win rate | avg VP us / them |
  |---|---|---|---|
  | random | 200 | 100 % | 10.1 / 2.4 |
  | weighted random | 200 | 100 % | 10.1 / 2.7 |
  | VP-greedy | 200 | 99.5 % | 10.1 / 2.7 |
  | ValueFunction stand-in | 200 | 24.5 % | 7.3 / 7.4 |
  | AlphaBeta stand-in | 160 | 24.4 % | 7.2 / 7.3 |

  Against our own strong stand-ins the depth-1 bot is exactly at the 25 %
  seat baseline: as strong as one of them, not stronger, with a seat
  handicap (31 % from seats 0-1, 18 % from seats 2-3): the stand-ins'
  opening book takes the best spots first.  No mechanical cause (0
  errors / fallbacks / stalls).  Next levers: depth 2 with the native
  lookahead (now 0.02 s per decision), better setup picks, the value net.
  The 3.3.0 strong ladder (catanatron's own AlphaBeta / ValueFunction /
  MCTS) is being re-run.
* **Value-net diagnosis** (why a 0.80-AUC net loses as evaluator): it is a
  good state predictor but a bad afterstate ranker.  The behaviour policy
  builds whenever it can, so the data has no counterfactual for holding
  cards and the net credits cards as much as the builds they become
  (+4 cards = +0.61 logit vs +1 VP = +0.32).  On 300 decision nodes the
  net's "build a settlement now minus end turn" delta is +0.004 (heuristic
  +0.159); it ranks END_TURN first in 17 % of nodes.  Fix being
  implemented: a sibling-ranking term (pairwise margin between the
  afterstate of the eventual winner's action and END_TURN / alternatives,
  with the heuristic ordering as an auxiliary target) on top of the outcome
  loss.  Training restarts when the fix passes its tournament.

## 2026-09-25 06:20 UTC - hourly check-in

* **Landed and verified**: the strategy ablation harness (`scripts/ablate.py`,
  30 tunables, paired games, per-decision cost; docs/ABLATIONS.md) and the
  blockability placement term (robber concentration penalty; C++ parity
  bit-exact; neutral within noise over 240 heuristic games, to be re-tested
  with the search bot in the full sweep).  Smoke hint from the harness: trade
  proposals are about half of the search bot's decision time at depth 1.
* **In progress**: value-net fix (training-side; agent is fitting a net on the
  saved buffer and running tournaments), native C++ search port (agent is
  building cpp/search.* and the MLP forward), Catanatron ladders on both
  versions (verification stage).
* Training stays paused until the fix is verified.  Load average ~10 on 4
  cores from the agents' experiments; the full ablation sweep waits for a
  quiet window.
* Opus switch notice: not yet sent (two implementation pieces still open).

## 2026-09-25 05:20 UTC - state before the overnight run

**Value net.**  Two candidates trained on the accelerated self-play data
(1.35M positions from 750-game iterations with trading-randomised bots) both
generalise as classifiers (validation AUC 0.80 / 0.78, loss 0.47 / 0.50, game-level
split) but lose the promotion tournament badly when used as the search evaluator:

| iteration | candidate wins | heuristic-search wins | avg VP candidate / heuristic |
|---|---|---|---|
| 0 | 11 % | 39 % | 6.5 / 8.3 |
| 1 | 13 % | 37 % | 6.3 / 7.9 |

Diagnosis in progress (workflow): the working hypothesis is that an
outcome-regression net learns cross-game correlations (a fat hand marks a
rich player) rather than the within-game effect of an action (spending the
hand on a settlement), so the search under-builds.  Training is paused until
the fix is verified; the replay buffer is kept for `--resume`.

**Speed.**  C++ extension (features, heuristic evaluator, engine) verified
bit-identical to Python: features 38-47x, evaluator 68x, self-play games
about 4 s instead of 14 s, depth-2 search 0.34 s (heuristic) / 0.61 s (net)
per position.  A native port of the search's opponent simulation is in
progress to make depth 3-4 practical.

**Review.**  58 findings (3 critical, 23 major, 32 minor) from the multi-lens
review were verified and fixed (engine Longest Road edge case, screenshot
parser robustness, CLI validation, arbitrage / kingmaker / card counting
wired into the search, self-play trading randomisation); full suite 385
passed.

**Benchmarks.**  Stock Catanatron bots: 99-100 % wins (control).  Strong
opponents (our in-engine ValueFunction / AlphaBeta stand-ins, catanatron 3.3's
own players) ladder in progress; a compute-fair (time-matched) ladder is
queued because the current ladder does not equalise thinking time.

**Pending overnight.**  Value-net fix and training restart; native search;
blockability placement term; ablation harness and full sweep; fair ladder.
