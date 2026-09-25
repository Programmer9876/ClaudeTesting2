# Results log

Living document, updated by the overnight check-ins.  Newest entries first.
Everything here was measured on the cloud container (4 shared cores, 15 GB);
numbers vary with the load from concurrent jobs.

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
