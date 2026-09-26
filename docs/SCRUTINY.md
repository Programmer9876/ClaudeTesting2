# Scrutiny notes: the questions a skeptic should ask about the strength proof

Claim under test: our bot (`search:depth=1,beam=4,expand=8,evaluator=heuristic`)
beats Catanatron's strong bots, and is ready for supervised human testing
(docs/PROOF_PROTOCOL.md).  Every answer below says what the evidence is and how
to re-check it.

**Status (2026-09-26 00:25 UTC):**
- T1-T6 and R1-R2 are finished, and **claims 1 and 2 PASS** (docs/PROOF.md).
- T7-T11 (Colonist information and the mixed table; claims 3 and 4, needed
  for readiness) are running.
- The T1-T6 numbers below are final.
- `RUN` below is the archived evidence, `proof/` in this repository
  (proof/README.md).

| test | opponent (Catanatron 3.3.0) | format | games | our wins | one-sided p | lower end of 99 % CI |
|---|---|---|---|---|---|---|
| T1 | ValueFunction x3 | 1v3 (fair share 25 %) | 1000/1000 | 628 (62.8 %) | 3.8e-140 | 0.588 |
| T2 | AlphaBeta x3 | 1v3 | 400/400 | 218 (54.5 %) | 2.9e-36 | 0.479 |
| T3 | SameTurnAlphaBeta x3 | 1v3 | 400/400 | 218 (54.5 %) | 2.9e-36 | 0.479 |
| T4 | ValueFunction x2 | 2v2 (fair share 50 %) | 1000/1000 | 836 (83.6 %) | 2.5e-109 | 0.804 |
| T5 | AlphaBeta x2 | 2v2 | 400/400 | 323 (80.8 %) | 3.2e-37 | 0.752 |
| T6 | SameTurnAlphaBeta x2 | 2v2 | 400/400 | 315 (78.8 %) | 1.9e-32 | 0.730 |

Registered pass lines, from the protocol:
- Claim 1: Holm family-wise alpha = 0.01.
- Claim 2: family-wise alpha = 5.7e-7, plus the lower end of the 99 % interval
  at least 0.35 (1v3) / 0.55 (2v2).  Claim 2 also has seat, error and
  stand-in conditions, covered below.

---

## A. Are the opponents what we say they are?

### Q1. T2 (AlphaBeta) and T3 (SameTurnAlphaBeta) both scored exactly 218/400.  Is T3 just T2 duplicated?

No.

- **What was recorded:**
  - Every T2 result file and every one of its 400 game logs name
    `catanatron.players.minimax:AlphaBetaPlayer` in all three opponent seats.
    Every T3 file and log name `SameTurnAlphaBetaPlayer`.
  - The class name is taken from the Python class the harness actually
    builds the opponents from (`resolve_opponent` / `opponent_path` in
    `scripts/bench_catanatron.py`), not typed in by hand.  The move-level
    differences below confirm it: the two tests' opponents play differently.
- **Same starting positions, by design:** both tests use base seed 900001,
  so game *g* has the same board, dev-deck order and seats in both tests.
  That is the protocol's registered seed.
- **The games themselves differ:**
  - 0 of the 400 pairs are identical.  Half of them differ within the first
    5 actions (a quarter by action 1, three quarters by action 11).
  - In 321 pairs the first difference is an opponent choosing a different
    move.  Example, game 0, action 3: AlphaBeta builds the road 19-21,
    SameTurnAlphaBeta builds 16-21.
  - In the other 79 the first difference is a chance outcome: a dice roll,
    or the card taken in a steal.  Catanatron's searches draw from the
    game's own random stream (catanatron 3.3 shares it with the copies it
    searches on), so two different searches leave the stream at different
    points.
  - In none of the 400 did our bot make a different choice first.  Given
    the same history it is deterministic.
- **The outcomes differ game by game:**
  - Won in both: 127.  Only against AlphaBeta: 91.  Only against SameTurn:
    91.  In neither: 91.
  - Only 13 of 400 pairs lasted the same number of turns.
  - Per-chunk results also differ: T2 in chunks of 40: 12, 29, 22, 18, 21,
    19, 21, 23, 24, 29; T3 in chunks of 50: 25, 26, 24, 29, 25, 29, 31, 29.
- **How often equal totals happen by chance:** two independent runs of 400
  games at a 54.5 % win rate give the same total with probability 0.028
  (about 1 in 35).
- **Re-check:** `python3 scripts/audit_opponents.py --run RUN T2 T3`.
  Positive control: `... T2 T2` (a real duplicate) reports 400 of 400
  identical games, so the script would catch a duplicated run.

### Q2. Are they Catanatron's real bots, unmodified, with default settings?

Yes.

- **Code:**
  - T1-T6 and T7-T11 use Catanatron 3.3.0 from its GitHub repository at
    commit `ecf9311`.  `git status` in that checkout shows no local changes.
  - R1-R2 use Catanatron 3.2.1 from PyPI.
- **Settings:** each game log records the opponent's class path and
  `params: {}`.  AlphaBeta's defaults are depth 2 with pruning off
  (`ALPHABETA_DEFAULT_DEPTH = 2` in `catanatron/players/minimax.py`).
- **Enforced by the analysis:** `prove_strength.py` rejects data whose
  metadata does not show all of these:
  - the registered opponent, both as preset name and as class, with default
    parameters;
  - the registered engine, format, seats and seed;
  - trades off, `PYTHONHASHSEED=0`, 10 VP to win and discard limit 7.

### Q3. Does our harness change how Catanatron's bots play?

No.

- **Why it could:** our harness seats the players in a rotating order
  (`make_game` re-seats them after Catanatron's own shuffle) and wraps each
  opponent in `BenchOpponent`, which times its moves.
- **Test** (`scripts/audit_reseat.py`): 8 games of AlphaBeta + 3
  ValueFunction were each played three ways in one process:
  - Catanatron's own `Game`, with its seat shuffle forced to the same order;
  - our re-seat;
  - our re-seat with the wrapper.
- **Result:** the starting states were identical field by field (hands,
  decks, board, random-number state, legal moves).  All 2,683 actions were
  identical, including every dice roll, steal and dev draw, and the same
  player won all 8 games.
- **Also in the test suite:**
  `tests/test_catanatron_adapter.py::test_reseating_equals_catanatron_native_seating`
  runs a fast version with random players on both Catanatron versions.

### Q4. Were Catanatron's bots short of time or CPU?

No.

- **The limit that matters:** AlphaBeta and SameTurnAlphaBeta stop a search
  after 20 s of wall time.  Their slowest single decision in each test was
  6.5 s (T2), 16.3 s (T3), 7.7 s (T5) and 5.9 s (T6).  No decision in any
  of the 5,000 games reached the cutoff, so load never cut a search short.
  T3's 16.3 s was the closest call.
- **Which side used more time:**

  | test | our bot, s per game (all its decisions) | each opponent seat, s per game |
  |---|---|---|
  | T2 | 0.9 | 8.4 |
  | T3 | 1.2 | 10.7 |
  | T5 | 0.8 | 8.4 |
  | T6 | 0.8 | 7.6 |
  | T1, T4 (ValueFunction, a one-move-ahead bot) | 1.0-1.1 | 0.2-0.3 |

  Against the search bots ours thinks about 10x less than each of them.

### Q5. AlphaBeta beating ValueFunction only 26 % of the time looks weak.  Is our setup nerfing it?

No.  We measured that rate without our code:

| run | setup | AlphaBeta wins vs 3x ValueFunction |
|---|---|---|
| C2 | inside our harness | 20/100 |
| C2b | plain Catanatron game, no catanbot code, Catanatron's own seat shuffle | 32/100 |

- The two runs differ by chance: Fisher exact p = 0.076.
- Pooled: 52/200 = 26 % (95 % CI 20-33 %).
- Q3 shows the harness plays Catanatron's own game move for move.
- In a four-player game, depth-2 AlphaBeta is simply about as strong as
  ValueFunction.

---

## B. Is our bot cheating?

### Q6. Can it see the dice or the order of the dev-card deck?

No.

- **Its own random numbers:** our bot draws from its own random generator,
  seeded per game (`bot_seed` in each log).  The adapter never reads
  Catanatron's generator.
- **Dev deck:** the bot gets only the deck's *composition* (how many of each
  card type are left), never the order
  (`catanbot/bench/catanatron_adapter.py`, `state_to_catanbot`).
- **Dice:** Catanatron rolls them itself, when the roll action is executed,
  after our decision.

### Q7. Does it see opponents' hidden cards?

- **In T1-T6, yes, and so do Catanatron's bots.** Catanatron exposes the full
  state and its bots' value functions read it.  That is fair between the
  sides, but it is not how Colonist works.
- **Blind controls** (non-proof seeds): our bot with every opponent card
  hidden and replaced by random guesses, no card counting at all:

  | control | blind | with the full view |
  |---|---|---|
  | C3: vs 3x ValueFunction | 56.7 % (300 games) | 62.8 % (T1) |
  | C4: vs 3x AlphaBeta | 41.7 % (60 games) | 54.5 % (T2) |

  Seeing the cards is worth a few points.  It is not where the wins come
  from.
- **Pre-registered Colonist-information tests (T7-T11):** our bot sees only
  what a Colonist player sees, while Catanatron's bots keep the full view.
  - Public: production, builds, trades, Monopoly and Year of Plenty amounts,
    the bank, hand sizes, dev-card counts, played dev cards.
  - Hidden: steal cards (to third parties), discard types, unplayed dev
    types.
  - Readiness for human testing requires these to pass (claims 3 and 4).

### Q8. Could it make illegal moves or get special treatment from the engine?

No.

- Every move goes through Catanatron's `Game.execute` with
  `validate_action=True`, which raises on any move not in Catanatron's
  legal list.
- Over all 5,000 games of T1-T6 and R1-R2 there were zero adapter errors,
  zero illegal-action fallbacks and zero crashes.
- `scripts/replay_catanatron.py --check` re-executes every logged game in a
  fresh engine and checks every action, final VP, winner and a full-state
  fingerprint.  All 5,000 games pass, both from the run's logs and from the
  archived copy.
- Claim 2 requires zero errors, fallbacks and crashes over all proof games.

### Q9. Does it exploit trading?

No: trading is off in every proof game.  Catanatron's bots cannot answer
trade offers sensibly:
- ValueFunction always rejects;
- AlphaBeta and SameTurn crash on an offer.

The protocol's Tooling amendment turned trading off before any proof game.

### Q10. Does a rule difference between the engines favour us?

The known differences, docs/BENCHMARKS.md limitations 3, 4, 11 and 12:
- **Longest Road:** Catanatron does not count a road that ends at an
  opponent's building (the official rules do).  Our search expects Longest
  Road awards that Catanatron may not grant, which hurts us, if anything.
- **Road Building:** in Catanatron it needs wood and brick in hand.  This
  can only block our moves.
- **Dev cards:** on 3.2.1 they are playable the turn they are bought, for
  every player alike.
- **Discards:** 3.2.1 applies its discard rule oddly, but it coincides with
  the official one at the default discard limit.

None of these gives our bot information or actions the opponents lack.

---

## C. Were the statistics set up honestly?

### Q11. Were the thresholds or sample sizes picked after seeing results?

No.  Timeline (git history, UTC, 2026-09-25):

| time | event |
|---|---|
| 17:09 | protocol committed (`704849b`): hypotheses, opponents, sample sizes, seeds (900001 and 900101, never used before), statistics, thresholds |
| 19:20 | code frozen for the proof (`9984181`), run from a separate git worktree, `/home/user/proof_snapshot`, with its own C++ build |
| 19:55 | first proof game |
| 20:53 / 21:01 | amendments 2 and 3 committed (`c7c108a`, `1e709df`) |
| 21:58 / 21:59 | amendments 4 and 5, before any T7-T11 game |

- The Tooling amendment was written before any proof game.
- **Disclosure:** amendments 2 and 3 were written while T1-T2 were running,
  and part of their results were known.  Those amendments only *add* tests
  (T7-T11) and make readiness *harder* (claims 3 and 4 are now required).
  They change nothing about T1-T6, R1-R2 or claims 1-2.
- Amendments 4 and 5 fix how T7-T11 are run and which code they use.
  Amendment 5 records the code snapshot (`9599eed` plus four analysis and
  runner files, listed with sha256 prefixes).

### Q12. Could games have been dropped, or the run stopped when it looked good?

No.

- Each test has a fixed number of games.  The analysis accepts only exactly
  games 0..N-1 of the registered seed; missing or extra games fail
  conformance.
- A crashed game is re-run once with the same seed; a second crash counts as
  a loss.  Turn-cap games count as losses.  Final: 0 crashes, 0 turn-cap
  games.
- There is no early stopping.  Interim numbers were read for monitoring
  only and changed nothing.

### Q13. Six tests: isn't a multiple-comparisons correction needed?

Yes, and it is applied: Holm-Bonferroni over T1-T6 (and over T7-T9 and
T10-T11 for claims 3 and 4).
- Holm is valid under any dependence between the tests.
- That matters here: the tests share base seed 900001, so the same 400-1000
  boards appear against different opponents.
- Within a test, every game has its own seed: its own board, deck and dice.

### Q14. The protocol calls alpha = 5.7e-7 "one-sided 5-sigma".  Is it?

Not exactly.  The label is loose, and our earlier note in docs/RESULTS.md
got it wrong; this is the correction.
- 5.7e-7 is the **two-sided** 5-sigma tail.  A one-sided 5-sigma threshold
  would be 2.87e-7.
- Used as a one-sided threshold, 5.7e-7 is 4.87 sigma: slightly *less*
  strict than one-sided 5 sigma, not stricter.
- The registered number stays; changing it after registration is not
  allowed.
- Every test also passes the stricter 2.87e-7: the largest Holm-adjusted
  p-value is 1.85e-32.

### Q15. Could seat order or turn order produce the result?

No.

- **Seats:** in 1v3, game *g* puts our bot in seat `g % 4`, so each seat is
  exactly a quarter of the games.  Seat 0 moves first.
- **Per-seat results** (claim 2 requires p < 0.05 for every seat):

  | test | seat 0 | seat 1 | seat 2 | seat 3 | largest per-seat p |
  |---|---|---|---|---|---|
  | T1 | 171/250 | 149/250 | 151/250 | 157/250 | 5.5e-31 |
  | T2 | 52/100 | 55/100 | 54/100 | 57/100 | 6.6e-9 |
  | T3 | 54/100 | 55/100 | 51/100 | 58/100 | 2.1e-8 |

- **Harness control C1:** Catanatron's own ValueFunction placed in "our"
  rotating seat against 3 ValueFunctions won 23.5 % of 400 games (fair
  share 25 %).  The harness favours no seat.
- **2v2:** our two seats rotate over all 6 possible arrangements.

### Q16. Is 2v2 double counting?

No.  A 2v2 game is one outcome: did either of our two seats win?  The null
is 50 %, because the two opponent seats have the same combined chance.

### Q17. Why only 400 games against AlphaBeta and SameTurnAlphaBeta, but 1,000 against ValueFunction?

Because the search bots are far more expensive to play against, and 400
games is still more than enough.

- **Fixed in advance:** the sizes were set in the protocol before any game
  (docs/PROOF_PROTOCOL.md, commit `704849b`).  Every test with AlphaBeta or
  SameTurnAlphaBeta seats has 400 games: T2, T3, T5, T6, T8, T9, and the
  mixed tables T10 and T11.  R2, against our AlphaBeta stand-in, also has 400.
  The ValueFunction tests have 1,000.
- **Cost:** AlphaBeta searches two plies ahead at every decision (up to its
  20 s cutoff).  ValueFunction looks one move ahead.  From the proof's result
  files:

  | test | opponent | wall time per game | each opponent seat, s thinking per game | our bot, s per game |
  |---|---|---|---|---|
  | T1 | ValueFunction | 1.7 s | 0.23 | 0.95 |
  | T2 | AlphaBeta | 26.0 s | 8.4 | 0.90 |
  | T3 | SameTurnAlphaBeta | 33.3 s | 10.7 | 1.18 |

  So an AlphaBeta opponent thinks about 36x longer per game than a
  ValueFunction one, and a game takes 15-20x longer.  1,000 AlphaBeta games
  per test would have cost about 7 hours per test instead of about 3.
- **Statistically, 400 is plenty for these claims.**  In a 1v3 test, 400
  games give a 99 % interval of about +-6.4 points at a 50 % win rate (1,000
  games: +-4.1).
- **The smaller size raises the bar rather than lowering it:**

  | games | to pass the strictest significance step | to pass the effect-size floor (lower end of the 99 % CI >= 35 %) |
  |---|---|---|
  | 400 | at least 148 wins (37 %) | at least 166 wins (41.5 %) |
  | 1,000 | at least 324 wins (32.4 %) | at least 390 wins (39 %) |

  - Significance: Holm alpha 5.7e-7 / 6, one-sided exact test against 25 %.
  - The 2v2 tests likewise need about a 62-63 % share at 400 games,
    against about 58-59 % at 1,000.
  - With fewer games the bot has to *win more often* to pass.
  - T2 and T3 passed at 54.5 %, with a lower 99 % bound of 0.479.

---

## D. Can someone else reproduce it?

### Q18. Can I replay a specific game?

Yes.
- **What each log holds:** the board, the dev-deck order, every action with
  its chance outcome, the players with their classes and seeds, and a
  final-state fingerprint.
- **Replay one game:** `python3 scripts/replay_catanatron.py <log> --game <g>
  [--turn T | --action K]` rebuilds the game in a fresh engine and prints the
  position.
- **Check every game:** `python3 scripts/replay_catanatron.py <logs...> --check`
  re-plays every game and verifies each action and the final state.
- **Hash seed:** Catanatron iterates over Python sets, so the same seed gives
  a different game in a different process unless `PYTHONHASHSEED` is fixed.
  The proof pins it to 0.

### Q19. Where are the logs and result files?

In this repository, under `proof/` (layout and commands in
proof/README.md):
- per-chunk result JSON;
- action logs sorted by game number, one gzip JSONL per chunk;
- console output, replay checks and the analysis;
- a sha256 manifest.

All 5,000 archived games were replay-checked before the commit.  The
registered analysis re-run on the archived copy prints output identical to
the run's own.  T7-T11 will be added the same way when they finish.

### Q20. Exactly which code played?

- **T1-T6 and R1-R2:** commit `9984181`, from the frozen worktree
  `/home/user/proof_snapshot`.
- **T7-T11:** commit `9599eed` plus four analysis and runner files from
  `855bfd0` (sha256 prefixes in amendment 5), from `/home/user/proof_snapshot2`.
- **Isolation:** development after those commits (win paths, counteroffers,
  log reading) is off by default and never touches the snapshots.

---

## E. What the proof does not show

- **Beating Catanatron is not beating strong humans.**  Claim 2 is readiness
  for *supervised* human testing with the advisor, nothing more.
- **No trading** in any proof game, because Catanatron's bots cannot trade.
  Trading, counteroffers and table politics are untested against outside
  opponents.  Self-play and human testing are where they get judged.
- **T1-T6 are full-information games.**  The Colonist-information result is
  T7-T11 (pending).
- **Catanatron's bots are the only outside opponents.**  The stand-ins in
  R1-R2 are our own stronger versions of them, not independent programs.
- **The screenshot parser** has only been validated on synthetic renders.
  Real Colonist screenshots are a separate gate.
- **The advisor cannot count cards yet.**  It reads one screenshot at a time;
  counting from Colonist's game log is being built (off by default).  Its
  log phrasing is a best guess and must be checked against a real game.
- **Shared, loaded hardware** (4 cores, about 3 busy with the proof).
  Timing numbers are for this machine.  Q4 shows the opponents were never
  cut short.
