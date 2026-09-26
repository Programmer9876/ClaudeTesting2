# Scrutiny notes: the questions a skeptic should ask about the strength proof

Claim under test: our bot (`search:depth=1,beam=4,expand=8,evaluator=heuristic`)
beats Catanatron's strong bots, and is ready for supervised human testing
(docs/PROOF_PROTOCOL.md).  Every answer below says what the evidence is and how
to re-check it.

**Status (2026-09-26 04:11 UTC): the proof is complete.**
- **All four claims PASS**, and so does the readiness for supervised human
  testing, which needs claims 2, 3 and 4 (docs/PROOF.md).
- Every number below is final.
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
| T7 | ValueFunction x3, Colonist info | 1v3 | 1000/1000 | 617 (61.7 %) | 1.7e-132 | 0.576 |
| T8 | AlphaBeta x3, Colonist info | 1v3 | 400/400 | 234 (58.5 %) | 9.8e-46 | 0.520 |
| T9 | SameTurnAlphaBeta x3, Colonist info | 1v3 | 400/400 | 227 (56.8 %) | 1.9e-41 | 0.502 |
| T10 | one each of ValueFunction / AlphaBeta / SameTurn | 1v3 | 400/400 | 244 (61.0 %) | 3.1e-52 | 0.545 |
| T11 | the same, Colonist info | 1v3 | 400/400 | 247 (61.8 %) | 2.9e-54 | 0.553 |

Registered pass lines, from the protocol:
- Claim 1: Holm family-wise alpha = 0.01.
- Claim 2: family-wise alpha = 5.7e-7, plus the lower end of the 99 % interval
  at least 0.35 (1v3) / 0.55 (2v2).  Claim 2 also has seat, error and
  stand-in conditions, covered below.
- Claims 3 (T7-T9) and 4 (T10-T11): the same strict thresholds.  They add
  zero card-counter errors or resets.

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
    They did:

    | opponents | Colonist info | full info |
    |---|---|---|
    | 3x ValueFunction | 61.7 % (T7) | 62.8 % (T1) |
    | 3x AlphaBeta | 58.5 % (T8) | 54.5 % (T2) |
    | 3x SameTurnAlphaBeta | 56.8 % (T9) | 54.5 % (T3) |

  - The tracker had zero errors or belief resets in 2,200 counted-mode
    games.

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
- **Two container restarts** interrupted T7-T11, cutting off T7 games
  800-999 and T9 games 120-159.
  - The runner keeps a chunk only once it has completed, so both chunks
    were discarded and played again in full with the same seeds.
  - Nothing was chosen between runs.
  - All 39 games that had finished before an interruption came out
    identical in the replay: winner, every seat's VP and turn count.
  - The interrupted output is archived in `proof/T7/interrupted/` and
    `proof/T9/interrupted/`.

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

All 7,600 archived games (T1-T11, R1-R2) were replay-checked before the
commit.  The registered analysis re-run on the archived copy prints output
identical to the run's own.

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
- **T1-T6 and T10 are full-information games.**  T7-T9 and T11 are the
  Colonist-information results: our bot sees only public information, while
  Catanatron's bots still see everything.
- **Catanatron's bots are far from complete players** (Q22, and docs/BENCHMARKS.md, "What Catanatron's bots
  are, and what they are not").  Development cards are nearly worthless to them and Victory Point cards
  invisible; they model one enemy, look two moves ahead, never trade and play no politics.  Part of our edge
  exploits these gaps, and humans will not leave them open.
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

---

## F. Questions added after the proof

### Q21. Is the development deck the standard one, is the shuffle fair, and is the bank finite?

Yes.  `scripts/audit_devdeck.py proof bench_1v1` checks all 8,400 logged games (the proof's 7,600 and the
1v1 benchmark's 800):

- **Deck:** every game's deck is the standard 25 cards: 14 Knight, 5 Victory Point, 2 Road Building,
  2 Year of Plenty, 2 Monopoly.  Catanatron builds it in `models/decks.py`, `starting_devcard_bank`; our own
  engine uses the same counts (`catanbot/board.py`, `DEV_DECK_COUNTS`).
- **Finite bank:** at the end of every game, bank plus all hands is exactly 19 of each resource, and deck plus
  held plus played cards is exactly 25.  A card cannot be bought from an empty deck.
- **Shortage rule:** when the bank cannot pay everyone owed a resource, Catanatron pays nobody that resource
  (`apply_action.py`, `yield_resources`), even if only one player is owed.  The official rule, which our
  engine follows (`catanbot/engine.py`, the production step), gives a single owed player whatever is left.
  This is a small engine difference, and it applies to both sides equally.
- **Piece limits, measured** (`scripts/audit_limits.py`, replaying the 7,000 catanatron 3.3 games, that is the
  proof without R1-R2 plus the 1v1 benchmark; 26,400 player-games):
  - no player ever had more than 15 roads, 5 settlements or 4 cities on the board, and both engines refuse
    the move when the pieces run out (catanatron `ROADS_AVAILABLE` etc.; `catanbot/board.py`, `MAX_ROADS`,
    `MAX_SETTLEMENTS`, `MAX_CITIES`);
  - the limits do bind: 20.7 % of player-games used all 5 settlements at some point, 6.5 % all 4 cities and
    6.2 % all 15 roads.  Upgrading a settlement to a city returns the settlement piece.
- **Bank shortages, measured:** the bank ran out of a resource in 244 games for wheat, 180 ore, 103 sheep, 68
  wood and 30 brick.  A shortage cancelled production on 904 of 552,511 rolls (0.16 %, in 709 games; 4,963
  cards not paid).  In 73 of those the official rule would have paid a single owed player what was left.
  None of this happened in the 1v1 games.
- **Road Building:** it gives two free roads, and it cannot be played without a place to build.  If the
  player runs out of roads or places after the first road, it ends early.  In the 3.3 games it gave 2 roads
  3,611 times and 1 road 15 times.  Our engine does the same (`catanbot/engine.py`, `free_roads`).
- **Shuffle:** where the Victory Point cards sit in the 4,600 distinct decks looks exactly like uniform
  shuffling: a chi-square of 18.4, larger in 57 % of simulated uniform shuffles.  (Tests reuse seeds, so the
  8,400 games hold 4,600 distinct decks.)  Draws come off the logged deck in order in every game.
- **No peeking** (see also Q6): for each purchase, the chance of a Victory Point card is the share of Victory
  Point cards among the cards not yet drawn.  Our bot drew 12,582 Victory Point cards in 61,095 purchases,
  where 12,569.6 were expected (z = +0.13).  A bot that saw the order and bought when a Victory Point card was
  next would be far above that.  Catanatron's bots drew 4,575 where 4,700.3 were expected (z = -2.11, within
  chance for this number of checks, and if anything against them).

**Why the replay of game 23 shows three Victory Point cards in six purchases.**  That game was picked *because*
it was won on hidden points.  Three or more Victory Point cards in six draws has a 7 % chance.  Across all of
our bot's purchases, 55.4 % were Knights and 20.6 % Victory Point cards, as the deck predicts.  Our bot plays
its Knights, which everyone sees, and keeps its Victory Point cards hidden.  So the hidden points stand out.

### Q22. Why does Catanatron's AlphaBeta almost never buy development cards, and does our edge depend on it?

Its search handles a purchase correctly: it expands it as a chance node over the cards it cannot see
(`players/tree_search_utils.py`, `execute_spectrum`).  The cause is the hand-set value function it scores
positions with (`players/value.py`, `base_fn`, `DEFAULT_WEIGHTS`):

| feature | weight |
|---|---|
| public victory points | 3e14 |
| own production / next player's production | 1e8 / -1e8 |
| hand synergy | 100 |
| each development card in hand | 10 |
| each Knight played | 10.1 |
| each resource card in hand | 1 |

- **Development cards are tie-breakers.**  A card is worth 10, and it costs three resource cards, which also
  lowers hand synergy.  So buying one usually scores as a loss.  In the proof games each AlphaBeta bought
  0.15 cards per game; our bot bought 6.2.
- **Victory Point cards are worth nothing to it.**  The function counts *public* points
  (`VICTORY_POINTS`).  A Victory Point card raises only `ACTUAL_VICTORY_POINTS` (`state_functions.py`), even
  for its own cards, which it can see.  A finished game is scored with the same function (`minimax.py`,
  `alphabeta`), so a card that wins the game on the spot gets no extra credit.
- **Depth 2 is too short** to see Largest Army or the later payoff of Monopoly and Year of Plenty.

The code comments name other simplifications (Monopoly assumes perfect card counting; the value function
models one enemy) but say nothing about development cards.  The weights are a design choice, not a bug in the
search.

**What it means for us.**  Hidden Victory Point cards were part of the winning total in 168 of our 218 wins in
T2.  AlphaBeta neither competes for development cards nor suspects hidden points, and humans will do both.  So
this part of the edge is specific to Catanatron.  Two follow-up checks are proposed, to be reported next to
the registered results without changing them:
1. our bot with development-card purchases off, against AlphaBeta;
2. an AlphaBeta patched to count its own hidden points and to score a won game as a win.

### Q23. Are discards public?  The Colonist-information tests treated them as hidden.

On a 7, Colonist shows which cards each player discarded (the user, a regular Colonist player, confirms it).
In physical Catan the discards go back to the bank, face up.
- **The proof's Colonist-information tests (T7-T9, T11)** were registered with hidden discards: only the count
  was public (`discards_public: false`, protocol amendment 2).  That gave our bot *less* information than it
  has on Colonist, and it still passed.  So the claim is conservative on this point.
- **The advisor's log reader** (`catanbot/colonist_log.py`) already takes the discarded cards whenever the log
  line shows them, and treats a discard as hidden only when the log gives just a count.
- **From now on,** every Colonist-information queue row and benchmark uses `discards_public: true`
  (`--adapter-opt discards_public=true`, or `--discards-public` in the bench scripts).  The registered proof
  commands keep the old setting, so they replay unchanged.
