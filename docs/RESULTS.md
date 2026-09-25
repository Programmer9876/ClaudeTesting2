# Results log

Living document, updated by the overnight check-ins.  Newest entries first.
Everything here was measured on the cloud container (4 shared cores, 15 GB);
numbers vary with the load from concurrent jobs.

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
