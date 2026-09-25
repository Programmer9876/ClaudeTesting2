# Results log

Living document, updated by the overnight check-ins.  Newest entries first.
Everything here was measured on the cloud container (4 shared cores, 15 GB);
numbers vary with the load from concurrent jobs.

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
