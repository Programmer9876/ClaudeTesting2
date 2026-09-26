# Replay archive: game picker and "what our bot knew" (design spec)

Status: design, revision 2, 2026-09-26. The measurements come from the scratch prototypes and checks listed in
section 12, run on the real logs. Revision 2 answers two critiques (data and correctness; UX and scrutiny). Section
11 lists the points that were rejected or changed. Nothing in `catanbot/`, `proof/`, `bench_1v1/`, `bench_multi/`
or the snapshot worktrees changes.

## 0. What gets built

One static page, **Catanbot Replay Archive**. It lists every game of the strength proof and the two benchmarks:
9,600 logged games. Any game opens in a step-through replay that reuses the board, controls, scoreboard, VP chart
and log of `scripts/game_viewer_template.html`.

For the tests where our bot had only public information (T7, T8, T9, T11), the page shows each opponent's real
hand at every step, next to what our bot believed. The belief is re-derived after the fact. It uses **the code that
played those games** (the frozen worktree `/home/user/proof_snapshot2`, commit 9599eed), fed the public log with
the test's own settings. For full-information games the page states that our bot saw every hand.

The page is published as a claude.ai Artifact with supporting data files.

| | measured / extrapolated |
|---|---|
| games | 9,600 in 18 tests (4 × 1,000 + 14 × 400), 0 crashed, 0 without a winner, 2,905,082 actions (max 654 in one game) |
| log files | 130, all listed in `proof/`, `bench_1v1/`, `bench_multi/` `MANIFEST.sha256`. Every file `…_g<a>-<b>.jsonl.gz` holds games a … b−1 in line order (checked on all 130) |
| game shards | 96 files (`shards/<T>_<kk>.json.gz`, 100 games each). 11.9 MB gzip measured with revision 1's bot layer. Revision 2's bot layer (5.6) is **not measured yet**; if it doubled, the total would be about 14.8 MB |
| largest shard | T7: about 270 KB gzip with revision 1's layer, at most about 430 KB if the bot layer doubled |
| index | 1 file (`index.json.gz`). Revision 1 measured 994 KiB JSON / 172 KiB gzip; revision 2 adds 4 columns and per-test provenance |
| bundle | **98 files**: `index.html` (with `replay_core.js` inlined), `index.json.gz`, 96 shards. Limits: 255 files and 64 MB per publish, 15 MB per binary file |
| export time | about 20 CPU-minutes, 10–15 minutes of wall time with 2 workers, heavy tail included (6.6) |

## 1. Facts established from the data

Sources: the survey of every log (`proto/survey.py`) and the critics' read-only checks (section 12).

* **Geometry.** Every game uses the same **BASE topology**, with the same land-tile and port-tile coordinate order
  in every log.
  * Node and edge ids and all positions are identical between catanatron 3.2.1 and 3.3.0, and between 2-player
    and 4-player games.
  * The geometry has 96 nodes and 132 edges (water-ring nodes included), 19 land tiles and 9 ports: 2,986 bytes of
    JSON, stored once.
* **Seats and rules.** Seats are always coloured `RED, BLUE, ORANGE, WHITE` in seat order (the first `n` of them).
  `vps_to_win` = 10, `discard_limit` = 7 and the turn limit is 1,000 everywhere.
* **Action kinds.** 3.3 has `BUILD_SETTLEMENT BUILD_ROAD BUILD_CITY ROLL END_TURN BUY_DEVELOPMENT_CARD MOVE_ROBBER
  MARITIME_TRADE DISCARD_RESOURCE PLAY_KNIGHT_CARD PLAY_MONOPOLY PLAY_YEAR_OF_PLENTY PLAY_ROAD_BUILDING`.
  * 3.2.1 has the same set, except that a discard is one `DISCARD` whose value lists the cards (9–11 cards occur in
    R1 and R2).
  * No trade-offer kinds appear: every test ran with `trades: off`.
* **Turns and rolls.** catanatron's `final.turns` = number of `END_TURN` + 2n − 2 (checked on 340 games).
  * The page's unit is the **roll**: "roll 57" is the 57th `ROLL`. The verify command (8.5) uses only `--action`.
  * catanatron's own turn count is shown next to it only if the per-step validator confirms the running count
    (9.1 step 5).
* **Information mode** comes from the results metadata `info`.
  * `{"discards_public": false, "mode": "counted", "samples": 4}` in every results file of T7, T8, T9 and T11
    (checked). These tests have exactly one catanbot seat, and the log record holds the same dict at
    `players[our_seat].info`.
  * `{"mode": "full"}` for T10, H1, H2 and M1–M3.
  * Absent for T1–T6, R1 and R2. Those runs predate the switch and are full information (SCRUTINY Q7).
* **Opponent presets.** `value`, `alphabeta` and `sameturn` (catanatron 3.3 classes), plus `vf` and `ab` (catanbot's
  3.2.1 stand-ins, R1 and R2). T10, T11 and M3 records carry `lineup` (a preset per seat).
* **Per-game tracker statistics** logged by the run are in `results[*].info_stats`: `hidden_steals`,
  `hidden_discards`, `hidden_dev_draws`, `info_resets`, `info_errors`, `info_max_hypotheses`, `info_uncertain`
  (searched decisions made while some opponent hand was not known exactly) and `info_samples`.
  * `info_resets` and `info_errors` are 0 in all 2,200 counted games.
  * `info_max_hypotheses` reaches 2,232 (T7-111), 1,697 (T7-492), 1,829 (T8-375), 1,589 (T9-208) and 750
    (T11-390). Games above 500 hypotheses: T7 15, T8 4, T9 3, T11 2.
* **Code that played each test** (docs/PROOF.md "What was played", SCRUTINY Q20, docs/BENCH_1V1.md,
  docs/BENCH_MULTI.md):

  | tests | code | worktree |
  |---|---|---|
  | T1–T6, R1, R2 | commit `9984181` | `/home/user/proof_snapshot` |
  | T7–T11 | commit `9599eed`, plus four analysis and runner files from `855bfd0` (PROOF_PROTOCOL amendment 5) | `/home/user/proof_snapshot2` (launcher `$S/proof/queue_t7_t11.sh` does `cd /home/user/proof_snapshot2`) |
  | H1, H2 | commit `35ef224` | frozen worktree of that commit |
  | M1–M3 | commit `332775b` | frozen worktree of that commit |

* **The repository's code is not the code that played T7–T11.**
  * Against the snapshot, `catanbot/bench/public_info.py` differs by 369 diff lines and `catanbot/counting.py` by
    59.
  * `catanbot/public_belief.py` and `catanbot/devbelief.py` do not exist in the snapshot.
  * `catanbot/bench/catanatron_adapter.py` is still being edited by other agents.
  * Today the two trackers give identical per-step belief digests: hypotheses, pending discards, dev pool, exact
    flags, expected hands, counter statistics and hidden-event counts. This held on 40 of 40 games (T8 20,
    T11 20; critic's `digest_tracker.py`).
  * Snapshot file hashes (sha256 prefixes): `public_info.py` `703e3fa8967837f3`, `counting.py` `718422ad13cc3e7f`,
    `catanatron_adapter.py` `b5aacaddfc164ee6`.
* **API of the snapshot's tracker** (the only API the exporter uses for beliefs).
  * `PublicInfoTracker(me_color, discards_public=False, reveal_hidden=False, vps_to_win=10, max_hypotheses=4096)`.
  * Belief methods: `replay_log(st)`, `is_exact(j)`, `expected_hand(j)` (pending hidden discards removed in
    proportion), `support_weight(hands)`, `dev_pool()`, `unknown_dev_holders()`, `pool_is_consistent()`,
    `_deal_devs(s, rng)`.
  * Counter: `counter.hyps` (joint hypothesis → weight), `counter.stats` (`events`, `branching_events`,
    `max_hypotheses`, `pruned_mass`, `contradictions`, `resets`). `CardCounter.expected` and `.exact` are
    properties. The only writers of `hyps` are `__init__`, `_update` and `_reset`.
  * The snapshot has no `reveal_devs` argument and no `PublicBelief` class.
  * The same method names exist in the repository's tracker, which is what the head cross-check (5.1) relies on.
* **Development-card belief in the proof.** The proof code predates `devbelief`. Its determinizations dealt
  opponents' development cards uniformly from the public pool. `_deal_devs` re-deals up to 20 times while any
  opponent's public VP plus its dealt VP cards would reach `vps_to_win`. After 20 failures it swaps VP cards for
  other types left in the pool. The bot did not use how long a card had been held.
* **Hidden-VP wins.** A win "needed hidden VP" when the winner's public VP at the end is below `vps_to_win`.
  Revision 1's definition (VP > public VP) also counted winners who already had 10 public VP: 233 games too many.
  About 57 % of all games need hidden VP, so a yes/no filter selects little; the page offers finer filters (4.3).

  | test | games | winner needed hidden VP | … of which our bot | revision 1 definition |
  |---|---|---|---|---|
  | T1 | 1,000 | 503 | 459 | 533 |
  | T2 | 400 | 174 | 165 | 177 |
  | T3 | 400 | 162 | 159 | 175 |
  | T4 | 1,000 | 654 | 635 | 676 |
  | T5 | 400 | 244 | 240 | 252 |
  | T6 | 400 | 254 | 247 | 264 |
  | T7 | 1,000 | 478 | 432 | 504 |
  | T8 | 400 | 177 | 170 | 192 |
  | T9 | 400 | 178 | 169 | 188 |
  | T10 | 400 | 191 | 181 | 201 |
  | T11 | 400 | 194 | 181 | 206 |
  | R1 | 1,000 | 658 | 222 | 680 |
  | R2 | 400 | 306 | 82 | 323 |
  | H1 | 400 | 228 | 222 | 235 |
  | H2 | 400 | 251 | 228 | 257 |
  | M1 | 400 | 283 | 282 | 292 |
  | M2 | 400 | 297 | 297 | 304 |
  | M3 | 400 | 243 | 238 | 249 |
  | all | 9,600 | 5,475 | 4,609 | 5,708 |

* **Shared deals.** Grouping every game's seed from the results files gives:
  * 400 groups of 6: T1–T6, games 0–399.
  * 400 groups of 3: T7–T9, games 0–399.
  * 1,000 pairs: T1/T4 games 400–999 and R1/R2 games 0–399.
  * 4,000 unique seeds.

  T7-0, T8-0 and T9-0 have identical boards, development decks and opening actions (critic's check). The exporter
  verifies board and deck equality for every group (4.1, column `d`).
* **Published totals** (results files; docs/PROOF.md and SCRUTINY report the same). Our bot's wins per test:
  T1 628/1,000, T2 218/400, T3 218/400, T4 836/1,000, T5 323/400, T6 315/400, T7 617/1,000, T8 234/400,
  T9 227/400, T10 244/400, T11 247/400, R1 278/1,000, R2 103/400, H1 302/400, H2 293/400, M1 361/400, M2 379/400,
  M3 314/400.

## 2. Files

### 2.1 New repository files (sources)

| path | role |
|---|---|
| `scripts/export_replay_archive.py` | exporter: orchestrator, per-shard worker and per-shard head-digest pass. Stdlib-only at import time |
| `scripts/replay_archive/index.html` | the page template. The exporter writes the bundle's `index.html` from it by inlining `replay_core.js` between `/* BEGIN replay_core.js */` and `/* END replay_core.js */` |
| `scripts/replay_archive/replay_core.js` | shared decoder and view models, UMD (`window.ReplayCore` in the page, `require()` in node). **Not published as a file** |
| `scripts/replay_archive/validate_archive.js` | node validator (CommonJS, node ≥ 18, no npm packages) |
| `tests/test_replay_archive.py` | pytest for the exporter |
| `docs/designs/replay_archive.md` | this spec |
| `docs/USAGE.md` | a short "Replay archive" section (export, validate, publish) added once the scripts exist |

**Names as implemented.** The files were created as `scripts/replay_export.py` (exporter),
`tests/test_replay_export.py` (pytest), `scripts/replay_archive_template.html` (page template),
`scripts/replay_core.js` (core) and `scripts/check_replay_bundle.mjs` (node validator, ES module; options as in 9.1
plus `--template`, `--expect-all`, and `--bundle-only [--work DIR]` for partial bundles). The roles are the ones above.

`scripts/game_viewer.py` and its template stay unchanged. The new page copies the template's CSS tokens and drawing
code; it does not import them.

### 2.2 Bundle (published as is)

`$S = /tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad`

```
$S/replay_archive/
  index.html                 page: <title>Catanbot Replay Archive</title>, own <style>, replay_core.js inlined,
                             no <html>/<head>/<body>, no external scripts
  index.json.gz              geometry + test metadata and provenance + 9,600 index rows
  shards/T1_00.json.gz       games 0-99 of T1
  ...                        96 shard files: T1, T4, T7, R1 have _00.._09; the other 14 tests _00.._03
```

A game's shard is `shards/${t}_${String(Math.floor(g / 100)).padStart(2, "0")}.json.gz`. The Python function is
`shard_path(t, g)` and the JS function is `ReplayCore.shardPath(t, g)`; both are tested.

### 2.3 Work files (never published)

```
$S/replay_archive_work/
  code_head/                     frozen copy of the repository's catanbot/ + scripts/replay_catanatron.py (6.2)
  code_head/SHA256SUMS           sha256 of every copied file, plus a tree hash
  parts/<T>_<kk>.rows.json       index rows of one shard (merged into index.json.gz)
  parts/<T>_<kk>.checks.json     per-game check results (audit counts, stats equality, twin, timings)
  digests/<T>_<kk>.snap.json     per-position belief digests from the snapshot tracker (counted tests)
  digests/<T>_<kk>.head.json     the same from the frozen head copy
  truth/<T>_<kk>.steps.json.gz   per-position truth and belief for the sampled games (6.5, 9.2)
  logs/<T>_<kk>.log              worker progress, one line per 10 games
  export.log                     orchestrator log, one line per shard job
```

## 3. Per-game encoding

### 3.1 Global geometry (once, inside `index.json.gz` as `geo`)

```
geo = {"nodes": [[x, y] × 96],
       "edges": [[a, b] × 132],            // sorted, a < b
       "tiles": [[[x, y, z], cx, cy] × 19],
       "ports": [[cx, cy, [n1, n2]] × 9]}
```

* The values come from `game_viewer.geometry(game)` (unit circumradius, pointy-top, y down) and the tiles' edge
  sets.
* `tiles` and `ports` follow the order of the land and port tiles in `rec["board"]["tiles"]` (catanatron order).
* The exporter computes `geo` from the first game and asserts **`global_geometry(game) == geo` for every game**,
  which costs milliseconds per game.

### 3.2 Board code (per game, 48 characters)

`b = <land: 19 × 2 chars> "|" <ports: 9 chars>`

* **Land tile i** (in the order of `geo.tiles`): a resource char, then a number char.
  * Resource: one of `W B S G O D` (wood, brick, sheep, grain/wheat, ore, desert).
  * Number: one of `2 3 4 5 6 7 8 9 A B C` (A = 10, B = 11, C = 12). The desert uses `-`.
  * Examples: `O8` is ore 8 and `D-` is the desert.
* **Port i** (in the order of `geo.ports`): one of `W B S G O 3` (`3` = generic 3:1).
* **Robber.** It starts on the desert. The decoder derives this, and the validator checks it against
  `rec.board.robber`.

### 3.3 Game record (the element of a shard's `games` array)

```json
{"id": "T8-123", "g": 123, "b": "O8B5...|3WG3...", "n": 4, "o": [0],
 "s": [[op, v, h, x], ...],
 "bv": { ... }}                       // counted games only (section 5)
```

* `n` is the number of players and `o` lists our seats.
* `s` has one step per logged action, in log order. Step i (0-based) is logged action i.
* **Position k** is the state after k actions (k = 0 … N). The belief shown at position k is the belief after
  the tracker processed entries 0 … k−1 (5.2).

Shard file: `{"v": 2, "t": "T8", "from": 100, "to": 200, "games": [ ... sorted by g ... ]}`. Game `g` is at
`games[g - from]`.

### 3.4 Step tuple `[op, v, h, x]`

Trailing empty parts are dropped: `[op]`, `[op, v]`, `[op, v, h]`, `[op, v, h, x]`. A missing middle part is
`null` (for `v`) or `0` (for `h`). Example: `[36, null, 0, [13, 0]]` is a Knight that takes Largest Army.

* `op = kind * 4 + seat`, which is always below 56. The kind codes are fixed:

| code | kind | `v` |
|---|---|---|
| 0 | BUILD_SETTLEMENT | node id |
| 1 | BUILD_ROAD | index into `geo.edges` |
| 2 | BUILD_CITY | node id |
| 3 | ROLL | `10 * d1 + d2` (e.g. 63) |
| 4 | END_TURN | — |
| 5 | BUY_DEVELOPMENT_CARD | drawn card 0–4 (`KNIGHT, VICTORY_POINT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY`), from the result, else the value |
| 6 | MOVE_ROBBER | `tile * 100 + (victim_seat + 1) * 10 + (stolen_res + 1)`. A 0 slot means none. 3.3: stolen = result; 3.2.1: stolen = `value[2]` |
| 7 | MARITIME_TRADE | `give_res * 100 + ratio * 10 + get_res` (ratio = number of non-null give slots: 2, 3 or 4) |
| 8 | DISCARD_RESOURCE (3.3) | resource 0–4 |
| 9 | PLAY_KNIGHT_CARD | — (the robber move is the next step) |
| 10 | PLAY_MONOPOLY | resource |
| 11 | PLAY_YEAR_OF_PLENTY | `[r1, r2]` (the one-card form never occurs, but it stays decodable) |
| 12 | PLAY_ROAD_BUILDING | — |
| 13 | DISCARD (3.2.1) | counts `[w, b, s, g, o]` |

  Resources `0..4` are `WOOD, BRICK, SHEEP, WHEAT, ORE` (`AD.CB_TO_RESOURCE`).
* `h` holds the hand deltas of **every** seat as a flat sparse list `[seat * 5 + res, delta, ...]` in seat and
  resource order, or `0`.
  * They are stored for every kind: build costs, trades, steals, discards and Monopoly included. The JS core
    therefore never re-implements catanatron's production, bank-shortage or free-road rules.
  * Deriving the non-roll deltas instead would save only 9–16 % gzip (measured), which is not worth the risk.
* `x` holds state changes, stored only when a value changed, as a flat list `[code, value, ...]`.
  * `code = f * 4 + seat` for f = 0 (actual VP), 1 (public VP) and 2 (longest-road length).
  * `12` = Longest Road holder (seat or −1); `13` = Largest Army holder (seat or −1).

### 3.5 What the JS core derives (and the validator checks against the engine)

* **Pieces.** Settlement node, city node (replaces the settlement) and road edge come from `v`.
* **Robber.** From `v / 100`.
* **Development cards.** A `BUY` adds 1 to held[card]. A `PLAY_*` subtracts 1 from held and adds 1 to played
  (Knight 0, Road Building 2, Year of Plenty 3, Monopoly 4). VP cards stay held.
* **Bank and deck.** The bank is 19 minus the sum of all hands, per resource. The deck left is 25 minus the number
  of buys.
* **Rolls and dice.** Rolls are the number of `ROLL`s; dice are `v / 10` and `v % 10`.
* **catanatron turns.** Counted as the engine advances them, checked per position on the sampled games (9.1
  step 5).
* **Free builds.** A `BUILD_*` without `h` is free (setup or Road Building); the log text uses this.
* **Hidden events** (relative to our seat `me` in counted games):
  * a `MOVE_ROBBER` that steals, with `me` neither thief nor victim;
  * a discard by another seat;
  * a `BUY_DEVELOPMENT_CARD` by another seat.
* **Hidden-discard runs.** A run is the consecutive discard steps of one 7. The bank our bot saw stays at its
  pre-run value until the first non-discard step (the tracker's rule, `_observe_entry`).

Prototype result: `proto/proto_decode.js` decoded all 261 sampled games (11 tests, both engine generations) to
exactly the logs' own `final.state`, with 0 mismatches. The comparison covered hands, development cards held and
played, VP, public VP, longest-road length, titles, settlements, cities, roads, robber, bank, deck, turns and VPs.

### 3.6 Measured sizes (layout A = the tuples above, revision 1 bot layer; `proto/measure.py`)

The samples are the first games of every log file of a test, spread over its chunks. "gzip/shard" is the gzip -9
size of all sampled games of the test in one JSON array, divided by the number of games.

| test | games sampled | actions/game | JSON B/game | gzip/shard B/game | of which bot view (gzip) |
|---|---|---|---|---|---|
| T1 (1v3 full) | 20 | 324 | 5,390 | 927 | — |
| T4 (2v2 full) | 24 | 320 | 5,405 | 938 | — |
| H1 (1v1 full) | 24 | 221 | 3,083 | 509 | — |
| M1 (3v1 full) | 24 | 346 | 5,787 | 1,002 | — |
| M3 (2v2-mixed full) | 20 | 323 | 5,500 | 965 | — |
| R1 (3.2.1) | 24 | 324 | 5,666 | 1,030 | — |
| R2 (3.2.1) | 20 | 297 | 5,234 | 970 | — |
| T7 (counted) | 25 | 313 | 10,918 | 2,665 | 1,613 |
| T8 (counted) | 40 | 301 | 9,248 | 2,136 | 1,124 |
| T9 (counted) | 20 | 288 | 8,236 | 1,855 | 885 |
| T11 (counted, mixed) | 20 | 299 | 9,606 | 2,243 | 1,236 |

* **Extrapolation to all 9,600 games** (T1 stands in for T2, T3 and T10, T4 for T5 and T6, H1 for H2, M1 for M2):
  **60.40 MB JSON, 11.90 MB gzip**.
* **Alternatives measured.** A columnar layout (kinds, values, deltas and changes in separate arrays) gives
  16.35 MB gzip; gzipping each game separately gives 19.3 MB. Layout A in 100-game shards is the choice.
* **Heavy tail** (critic's `time_heavy.py`): T7-111 has 448 `q` rows and a 14.8 KB raw-JSON bot view.
* **Revision 2's bot layer** adds a few numbers per uncertain row, five per exact row, and the `dv` rows (5.6).
  The exporter prints the measured per-test sizes; the acceptance limit is the hosting budget, not an estimate.

**Hosting budget.**
* 98 files (limit 255), about 12–15 MB (limit 64 MB per publish), largest file below 0.5 MB (limit 15 MB).
* Shards are published with `contentType: "application/gzip"`.
* Fallback, only if the host refuses `.gz`: publish plain `.json` shards (about 60–65 MB) over two publishes to
  the same URL (the version limit is 256 MB). The page code is the same either way (7.2, `fetchJson`).

## 4. Index (`index.json.gz`) and picker data

### 4.1 Layout

```json
{"v": 2, "built": "<UTC ISO time>", "shardSize": 100, "geo": {...},
 "tests": {"T8": {"suite": "proof", "label": "3× AlphaBeta", "group": "Proof · public information only",
                  "fmt": "1v3", "opp": "alphabeta", "catanatron": "3.3.0",
                  "info": {"mode": "counted", "samples": 4, "discards_public": false},
                  "spec": "search:depth=1,beam=4,expand=8,evaluator=heuristic",
                  "seed": 900201, "games": 400, "shards": 4,
                  "published": {"games": 400, "wins": 234, "resultFiles": 10},
                  "code": {"played": "9599eed + 4 tooling files from 855bfd0 (PROOF_PROTOCOL amendment 5)",
                           "worktree": "/home/user/proof_snapshot2", "exportRoot": "/home/user/proof_snapshot2",
                           "sha256": {"catanbot/bench/public_info.py": "703e3fa8…", "catanbot/counting.py": "718422ad…",
                                      "catanbot/bench/catanatron_adapter.py": "b5aacadd…"},
                           "headSame": [400, 400]},
                  "files": [{"path": "proof/T8/logs/T8_alphabeta_1v3_seed900201_g00000-00040.jsonl.gz",
                             "from": 0, "to": 40, "sha256": "…", "manifest": true}, ...],
                  "hvp": {"needed": 177, "ours": 170},
                  "counted": {"allExact": 0.51, "meanP": 0.771, "calP": 0.772, "meanPu": null, "calPu": null,
                              "devBelief": "uniform over the unseen pool, conditioned on no opponent having already won"}}, ...},
 "cols": ["t","g","seed","d","lineup","win","vps","pvps","turns","rolls","acts","hvp","hv","dev","lr","la","def","defp","bk"],
 "rows": [["T8",123,90022800604,7,"caaa",2,[7,9,10,6],[6,9,9,6],124,119,456,1,1,8,2,0,3,2,
           [0.51,0.77,0.0044,0.77,0.41,0.42,23,49,15,48,1]], ...]}
```

The row values above are illustrative. `meanPu` and `calPu` are shown as `null` because they have not been
measured yet (5.7).

| col | meaning |
|---|---|
| `t`, `g` | test id and game index (`id = t + "-" + g`; the deep link is `#T8-123`) |
| `seed` | the game seed (`rec.seed`) |
| `d` | deal id: equal for games whose board code and `dev_deck` are identical. The exporter asserts that games with the same seed share `d`; any exception is reported, and the page then says "same seed, different deal" |
| `lineup` | one char per seat: `c` our bot, `v` value, `a` alphabeta, `s` sameturn, `f` vf (3.2.1 stand-in), `b` ab (3.2.1 stand-in). Our seats and colours come from it |
| `win` | the winner's seat (0-based). Every one of the 9,600 games has a winner (asserted), so it is never missing. Our win = `lineup[win] == "c"` (for 2v2, our team) |
| `vps`, `pvps` | final actual and public VP per seat |
| `turns`, `rolls`, `acts` | catanatron `num_turns`, number of `ROLL`s, number of actions |
| `hvp` | 1 if the winner's **public** VP at the end is below `vps_to_win`: the win needed hidden VP cards |
| `hv` | the winner's VP − public VP at the end (0, 1, 2, …; shown as 3+ above 2) |
| `dev` | development cards bought by our seats |
| `lr`, `la` | holder of Longest Road / Largest Army at the end (seat or −1) |
| `def`, `defp` | largest deficit, in **true** VP (`def`) or **public** VP (`defp`): max over positions of (best opponent VP − best of our VPs), floored at 0. Positions run from 0 to the one right after the last `END_TURN`, so the final turn's finishing burst does not count |
| `bk` | counted games: `[allExact, meanP, minP, calP, meanPu, calPu, maxH, uncertain, hiddenSteals, hiddenDiscards, hiddenDevDraws]` (5.7). `uncertain` is the run's `info_uncertain`; the hidden counts are the re-derived tracker's (equal to the run's by 5.8 b). `null` for full-information games |

### 4.2 Test metadata and provenance

* `label` and `group` come from a static `TEST_LABELS` table in the exporter, cross-checked against the metadata
  (opponent, format, information mode, catanatron version).
* The groups are:
  * *Proof · all hands seen* (T1–T6, T10)
  * *Proof · public information only* (T7–T9, T11)
  * *Replication · catanatron 3.2.1* (R1, R2)
  * *Heads-up 1v1* (H1, H2)
  * *Several of our seats* (M1–M3)
* `published` is summed by the exporter from the test's results files (`games`, `wins`, per-game `winner_seat`).
  The index must agree game by game (the exporter and the validator both check).
* `files` lists the test's log files: `from`/`to` come from the `g<a>-<b>` name, and `sha256` is computed at export
  and must equal the entry in the suite's `MANIFEST.sha256` (`manifest: true`). Game g is line g − from + 1 of its
  file (checked on all 130 files).
* `code` records the code that played (1, table), the code root the export used, the sha256 of the belief-relevant
  files in that root, and `headSame` = [counted games whose per-position digest under the repository's code equals
  the snapshot's, counted games] (5.1).

### 4.3 Picker filters

All filtering happens in memory; 9,600 rows filter in well under 10 ms.

* **Test**: multi-select chips, grouped as in 4.2.
* **Format**: `1v3`, `1v3-mixed`, `2v2`, `2v2-mixed`, `1v1`, `3v1`.
* **Information**: "Saw all hands" / "Public info only".
* **Opponent type** (in any seat): ValueFunction, AlphaBeta, SameTurnAlphaBeta, the 3.2.1 stand-ins.
* **Result**: our win / our loss.
* **Our seat**: 1st–4th, or any of ours.
* **Hidden VP**:
  * "our win that needed hidden VP" (`hvp` and our win);
  * "opponent win that needed hidden VP";
  * "won from ≤ N public VP" (winner's final public VP ≤ N, N = 6…9).
* **Was behind by ≥ N** (0–5+, any result), with a switch "true VP" / "public VP" (`def` / `defp`). Together
  with "Result: our win" this is the comeback filter.
* **Rolls** between a minimum and a maximum.
* **catanatron version**.
* **Free text**: an id (`T8-123`), a game number or a seed. A seed search groups the games of one deal and says
  "N games share this deal (same board and development deck, different opponents)".
* An always-visible line "N filters on · Clear all" ("No filters" when N = 0).

### 4.4 Sort

Column headers are buttons with `aria-sort`; click toggles ascending/descending. On phones a "Sort by" select
replaces them (8.4). The sort keys are:

* game (test order, then g)
* seed
* rolls
* turns
* our VP
* best opponent VP
* margin (our best − their best)
* behind by
* development cards bought
* bot's average chance on the real hands (`meanP`; counted only; lowest first is "hardest to count")
* share of positions known exactly (`allExact`)
* most hidden events (sum of the three hidden counts)

### 4.5 Summary strip (updates with the filter)

Example for T8, with the numbers from the results files and the logs:

> 400 games · our bot won 234 (58.5 %; 95 % interval 53.6–63.2 %) · fair share if all seats were equally strong:
> 25 % · mean 80.8 rolls · 177 winners needed hidden VP (170 of them our bot)

* The fair share is the mean over rows of `|our seats| / n`.
* The interval is the Wilson score interval over the selected games. A note says that games sharing a deal (4.3)
  are not independent, so across several tests the interval is approximate.
* **Filters that select by outcome**: "Result", the hidden-VP filters and "won from ≤ N public VP". When any of
  them is on, the win rate and interval are replaced by "win rate not meaningful: this filter picks by result".
* "Was behind by ≥ N" does not select by outcome (it excludes the final turn, 4.1), so the win rate stays.

## 5. "What our bot knew" (counted games: T7, T8, T9, T11)

### 5.1 Code and settings: the tracker that played, with the run's settings

* **Code root.**
  * Workers for T7–T11 (T10 included, for provenance) start with `/home/user/proof_snapshot2` and
    `/home/user/proof_snapshot2/scripts` first on `sys.path`.
  * They assert that `catanbot.__file__` lies under `/home/user/proof_snapshot2/`.
  * They assert that the sha256 of the snapshot's `public_info.py` and `counting.py` start with `703e3fa8967837f3`
    and `718422ad13cc3e7f`, so a changed snapshot fails loudly.
  * The critic verified that the snapshot imports and replays under `/home/user/venv_cat33/bin/python`.
  * The hashes go into `tests[t].code.sha256`.
* **Settings.**
  * `info` comes from the results metadata. The exporter asserts that it equals `rec.players[me].info` and
    `{"mode": "counted", "samples": 4, "discards_public": false}`.
  * The tracker is built exactly as the snapshot's `CatanbotPlayer._begin` builds it (`catanatron_adapter.py`
    l.1160): `PublicInfoTracker(Color(colors[me]), discards_public=info["discards_public"],
    vps_to_win=rec["vps_to_win"])`, with the defaults `max_hypotheses` = 4096 and `reveal_hidden=False`.
  * There is no development-card-belief switch to assert. The snapshot predates `devbelief`; its dealing law is
    shown exactly in 5.4.
* **Head cross-check** (cheap, every counted game).
  * A separate process runs the same tracker from the frozen head copy (`code_head`, 6.2) over the same log.
  * The snapshot worker and the head process each write a per-position digest with one shared function,
    `belief_digest(tr)`. It is defined in the exporter and reads only names present in both codes:
    * `sorted(counter.hyps.items())`
    * `sorted(pending_discards.items())`
    * `dev_pool()`
    * `[is_exact(j)]`
    * `[expected_hand(j)]`
    * the six snapshot `counter.stats` keys
    * `stats[hidden_steals, hidden_discards, hidden_dev_draws]`
  * The digest is the first 16 hex characters of a sha256.
  * The orchestrator requires equal digests at every position of every counted game and records `headSame`.
  * `--allow-head-drift` (for future re-exports only) records the mismatches with the first differing position
    instead of failing. The page always shows the **snapshot's** belief.

### 5.2 Lockstep and indexing

```python
final = AD.rebuild_game(rec)
for item in rec["actions"]:
    AD.replay_log_action(final, item, check=True)            # replay_log needs the complete engine log

game  = AD.rebuild_game(rec)                                 # the truth, re-played with check=True
plain = PublicInfoTracker(*args)                             # displayed belief
audit = AuditedTracker(*args)                                # 5.8 (c)
twin  = PublicInfoTracker(*args, reveal_hidden=True)         # 5.8 (d)
g1, g2, g3 = (t.replay_log(final.state) for t in (plain, audit, twin))

record_position(0, START, truth_of(game))                    # position 0: every hand empty and exact
for k in range(1, N + 1):
    item = rec["actions"][k - 1]
    AD.replay_log_action(game, item, check=True)             # game is now at position k
    entry, _ = next(g1); next(g2); next(g3)                  # each tracker has now processed entries 0..k-1
    assert AD.encode_log_entry(entry) == list(item)          # the tracker saw exactly logged entry k-1
    record_position(k, belief_of(plain), truth_of(game))     # belief and truth of the same moment
for g in (g1, g2, g3):
    assert next(g, None) is None
```

* **Position k** means after k entries. Its belief is the tracker's state after entry k−1, and it is scored against
  the truth at position k.
  * Both are read at the same line of the loop above.
  * The per-position truth file (9.2) stores both for k = 0 … N.
  * A pytest pins the convention with a hidden steal at a known index (9.3).
* **Position 0 (START)** is the constant empty start: every hand is `[0,0,0,0,0]` and exact, there is 1 hypothesis
  and the pool is `[14,5,2,2,2]`. `_init_from` raises unless every hand is empty, so this is asserted by the
  snapshot itself.
* **The truth** is read only from `game`, never from a tracker. `replay_log` is the path the bot itself took:
  `start()` calls it, and `follow()` calls the same `step()` per entry.

### 5.3 The belief at position k (for each opponent j)

The exporter derives everything from one object, `M_j`: our bot's distribution over opponent j's **current** hand.
`M_j` is the marginal of `counter.hyps` for seat j. If `k_j = pending_discards[j] > 0`, each hypothesis hand is
replaced by the multivariate hypergeometric distribution of the hands left after removing `k_j` cards
(`Π_r C(h_r, d_r) / C(|h|, k_j)`), which is the marginal version of the snapshot's `support_weight`.

**Truth-independent** (what the bot knew; published and usable in "Bot's view only"):

| field | definition | assert |
|---|---|---|
| `size_j` | hand size (public) | equals the true size and `Σ E_j` within 1e-6 |
| `exact_j` | `plain.is_exact(j)` | ⇔ the support of `M_j` holds one hand |
| `own_j` | when exact: the one hand in `supp M_j` (the counter's own hand) | equals the true hand |
| `E_j` | `plain.expected_hand(j)` (5 floats) | equals the mean of `M_j` within 1e-9 |
| `u_j` | bit mask of resources whose count is not the same in every hand of `supp M_j` | 0 ⇔ exact |
| `top_j` | `max_h M_j(h)`: the chance of the bot's most likely hand | ≥ `p_j` |
| `calp_j` | `Σ_h M_j(h)²`: the chance the bot's own belief expects to put on the real hand | |
| `nh` | `len(plain.counter.hyps)` (joint hypotheses) | ≤ 4096 |
| `pool` | `plain.dev_pool()` | equals `[14,5,2,2,2]` − Σ played − our held |
| `vpLaw_j` | 5.4 | sums to 1 |

**Truth-dependent** (evaluation only; never used in "Bot's view only"):

| field | definition | assert |
|---|---|---|
| `p_j` | `M_j(truth_j)` | `> 0`: the truth is never excluded |

### 5.4 The VP-card belief the bot actually used

`vpLaw_j` is the exact distribution of the number of VP cards among opponent j's unknown development cards under
the snapshot's `_deal_devs` sampler. It is recomputed only when `dev_pool()`, any `dev_count` or any public VP
changes.

* **Inputs** (all public):
  * the pool: T cards, V of them VP (`pool[1]`);
  * the unknown holders `U = plain.unknown_dev_holders()`, in seat order, with counts `c_j = dev_count[j]`;
  * public VP `pvp_j`, from catanatron's state. The exporter asserts it equals settlements + 2·cities +
    2·[Longest Road] + 2·[Largest Army], the snapshot's `GameState.public_vp`.
* **Joint law of one deal:** `P(v) = Π_j C(c_j, v_j) · C(T − Σc, V − Σv) / C(T, V)`. It is enumerated over
  `v_j ∈ 0..min(c_j, V)`; this is at most a few hundred terms.
* **Accept event:** `A = {∀ j ∈ U: pvp_j + v_j < vps_to_win}`, with `q = P(A)`.
* **Fallback after 20 failed deals.** Starting from the 20th deal (drawn from `P(· | not A)`), each j in seat
  order swaps VP cards for non-VP cards left in the pool while `pvp_j + v_j ≥ vps_to_win`.
  * The non-VP cards left number `(T − V) − (Σc − Σv)`, and each swap uses one of them.
  * The resulting VP counts are therefore a deterministic function of `v`.
* **The law:** `(1 − (1 − q)^20) · P(v | A) + (1 − q)^20 · fallback(P(v | not A))`, marginalised per opponent.
* The pytest `test_dev_vp_law` checks it against 20,000 draws of the snapshot's own `_deal_devs` (9.3).

The page also shows the unconditioned per-card chance `pool[1] / Σ pool` (public), labelled "before ruling out
deals that would already have won". It says in words that the bot ignored how long a card had been held.

### 5.5 Per-position asserts in the exporter (every counted game, every position)

* every assert in the two tables of 5.3
* `plain.known_dev[j] is None` for every opponent
* `plain.pool_is_consistent()`
* `counter.stats["resets"] == 0`, `counter.stats["contradictions"] == 0`, `counter.stats["pruned_mass"] == 0`
* `plain.stats["resyncs"] == 0`
* the twin's belief is exact and equals the true hands (5.8 d)
* the audited tracker's `hyps` and `pending_discards` equal the plain tracker's (5.8 c)

### 5.6 Sparse encoding (`bv` in the game record)

```json
"bv": {"q":  [[k, j, 0, lp, lt, lc, u, e0, e1, e2, e3, e4],     // uncertain
              [k, j, 1, h0, h1, h2, h3, h4], ...],              // exact: the counter's own hand
       "nh": [k, count, k, count, ...],
       "dp": [[k, p0, p1, p2, p3, p4], ...],
       "dv": [[k, j, c0, c1, ...], ...],
       "sum": {...},                                            // 5.7
       "chk": {"code": "9599eed", "stats": 1, "audit": [1916, 0], "twin": 1, "same": 1, "head": 1}}
```

* **Keys.** Rows are keyed by **position** k (1 … N). A row is written when any field of opponent j's record
  changes at k.
* **Position 0** is implicit: every opponent is exact with hand `[0,0,0,0,0]`, `nh` = 1, the pool is
  `[14,5,2,2,2]`, and there are no `dv` rows.
* **Uncertain rows.**
  * `lp = round(1000 · −log10 p)`, `lt = round(1000 · −log10 top)`, `lc = round(1000 · −log10 calp)`: relative
    precision 0.23 %, and no clamping anywhere.
  * `u` is the 5-bit mask; `e_r = round(100 · E_r)`.
* **Exact rows** publish the counter's own hand, not the true one. p = 1 is not stored: the validator checks the
  own hand against the decoded true hand (9.1 step 6).
* **`dv` rows.** `c_v` is `vpLaw_j(v)` in thousandths (largest-remainder rounding to 1000), for v = 0 … `c_j`.
* The measured row counts under revision 1 were 104–177 `q` rows per sampled game and 448 for T7-111.

### 5.7 Per-game summary (`bv.sum`, copied into the index as `bk`) and per-test aggregates

| field | meaning |
|---|---|
| `allExact` | share of positions 1..N at which every opponent hand was known exactly |
| `meanP`, `minP`, `calP` | mean of `p_j`, minimum of `p_j`, mean of `calp_j`, over positions 1..N and opponents |
| `meanPu`, `calPu`, `nu` | the same means over **uncertain** (position, opponent) pairs only, and their number. Exact pairs contribute p = calp = 1 to the headline means, which inflates them (51 % of T8 positions are all-exact) |
| `maxH` | `counter.stats["max_hypotheses"]` |
| `hidden` | `[hidden_steals, hidden_discards (cards), hidden_dev_draws]` for the whole game |

`tests[t].counted` holds the means over the test's games.

Revision 1 prototype values (sampled games; headline means over all pairs; the uncertain-only means were not
computed):

| test | games | allExact | meanP | calP | minP | maxH in sample | maxH in the full results |
|---|---|---|---|---|---|---|---|
| T7 | 25 | 0.338 | 0.669 | 0.663 | 0.0008 | 233 | 2,232 |
| T8 | 40 | 0.510 | 0.771 | 0.772 | 0.0002 | 204 | 1,829 |
| T9 | 20 | 0.568 | 0.802 | 0.800 | 0.0019 | 88 | 1,589 |
| T11 | 20 | 0.390 | 0.720 | 0.722 | 0.0044 | 122 | 750 |

### 5.8 What the checks show (run for every counted game; results in `bv.chk`)

The structural checks come first, because they are the evidence. Calibration comes last, as a consistency check.

(a) **Re-run with the code and settings that played.** The belief comes from the snapshot's `PublicInfoTracker`
(5.1), fed the logged entries one at a time through the same `replay_log` the bot used.

(b) **Same statistics as the live bot.**
* After the entries before our seat's last action, the tracker is in the state the bot's own tracker had at its
  last `decide()`, because `_follow_tracker` runs at every decide before the action.
* Its statistics must equal the per-game statistics the run logged (`info_stats`): `hidden_steals`,
  `hidden_discards`, `hidden_dev_draws`, `info_resets` (= counter resets) and `info_max_hypotheses`.
* Prototype: 105/105 games equal (T8 40, T7 25, T11 20, T9 20). Critic: the 3 heaviest T7 games equal too.
  → `chk.stats`.

(c) **Input audit.**
* `AuditedTracker(PublicInfoTracker)` overrides `_init_from` (to wrap the new counter instance), `_observe_entry`
  (notes the entry, the public snapshot before and after, and a hypothesis digest before and after) and
  `_revealed_result` (records the call).
* It wraps the counter instance's public mutators and the two internal writers `_update` and `_reset` with
  recorders that delegate unchanged.
* Query methods (`is_exact`, `marginal`, `weight_of`, `most_likely`, `sample`, `probability_has`, `expected_hand`,
  `hand_belief` and the properties) are not restricted.
* Every recorded call must satisfy the public-information rule for its entry (actor a, victim v, our seat me):

| call | allowed when |
|---|---|
| `observe_hand(i, hand)` | `i == me` and `hand` = our true hand |
| `observe_hand_size(j, s)` | `s` = j's true size after the entry (sizes are public) |
| `observe_bank(bank)` | `bank` = the bank after the entry, and the entry is not a discard |
| `observe_delta(j, d)` | the entry kind is not `MOVE_ROBBER`, a discard or `PLAY_MONOPOLY`, and `d` = j's true delta (production, build cost, bank trade, Year of Plenty: public content) |
| `observe_steal(v', a', res)` | `v'` = the entry's victim, `a'` = its actor, and `res is None` unless `me ∈ {a, v}` |
| `observe_discard(i, counts)` | `i == me` (discards_public is false) and `counts` = our true discard |
| `observe_discards(pending, bank)` | `pending` = the public per-seat hidden-discard counts since the 7, `bank` = the bank before the entry (after the discards) |
| `observe_monopoly(i, res, taken_by=T)` | `i` = actor, `res` = the entry's resource, `T[j]` = j's public hand-size drop |
| `_revealed_result(entry, …)` | the entry is `MOVE_ROBBER` and `me ∈ {a, v}` |
| `_update(...)` | only nested inside one of the audited public calls above |
| `_reset(...)` | never |
| any other counter mutator (`observe_gain`, `observe_spend`, `observe_bank_trade`, `observe_player_trade`, `sync`, `set_exact`, `observe_offer` or anything added later) | never |

* **Direct field writes.** After each entry, `counter.last_bank` must be the public bank before or after it, and
  `counter.size[j]` must be j's public size (for a seat with a pending discard, its public size before the
  discard).
* **Hypothesis digest.** `hash(tuple(sorted(hyps.items())))` at the start of each `_observe_entry` must equal the
  one at the end of the previous entry, so nothing changes `hyps` between entries. Within an entry, a change needs
  at least one audited `_update`.
* **The wrappers change nothing.** The audited tracker's `counter.hyps` and `pending_discards` must equal the
  plain tracker's at every position.
* **Prototype:** 22,991 calls in 12 T8 games and 15,057 calls in 8 T11 games, 0 violations, belief identical
  20/20. Critic: 0 violations on T7-111, T7-492 and T7-260.
* **The audit has teeth:** a pytest feeds a deliberately leaking subclass and requires ≥ 1 violation (9.3).
  → `chk.audit = [calls, violations]`, `chk.same`.

(d) **Twin with `reveal_hidden=True`.** It receives the same entries but treats steals, discards and development
draws as public. It must be exact and equal to the true hands at every position. This shows that the bookkeeping is
right and that the hidden events are the only difference between the two trackers. Prototype: 20/20. → `chk.twin`.

(e) **Structure.**
* The truth for display is read from the exporter's own replay (`game`), never from a tracker.
* The tracker receives `final.state` only through `replay_log`, which restarts from `initial_state_like` and
  applies the log entries one at a time.
* The tracker's private shadow game does hold true hands, which is why (c) audits what reaches the counter.

(f) **The repository's current tracker agrees** position by position (5.1). → `chk.head`.

(g) **Calibration: a consistency check, not a proof.**
* An honest posterior puts on the truth, on average, what it expects to put: meanP ≈ calP.
* Any Bayesian tracker is calibrated on its own information set, **including one that peeks**:
  * a full peeker has meanP = calP = 1;
  * a partial peeker that conditions correctly also has meanP ≈ calP.
* So calibration shows the bot was **honest about its uncertainty**, not that it did not peek. The page reports
  meanPu and calPu next to the twin's 1.00.

**Claim scope.** (a)–(f) show that **the belief layer** (the card counter) used public information only. They say
nothing directly about the move search. For the decisions, the page and this spec cite
`tests/test_public_info.py::test_counted_player_never_reads_hidden_information`, which is in the snapshot too (l.371
in both). That test swaps the hidden cards and requires an identical ranking. The search receives only
`public_view` and determinizations sampled from this counter (`_decide_counted`).

The exporter fails the shard if any check fails.

## 6. Exporter (`scripts/export_replay_archive.py`)

### 6.1 Command line

```
python3 scripts/export_replay_archive.py [--tests T1,...,M3] [--out $S/replay_archive] [--work $S/replay_archive_work]
        [--workers 2] [--shard-size 100] [--truth-per-test 4] [--py33 /home/user/venv_cat33/bin/python]
        [--py32 python3] [--snapshot2 /home/user/proof_snapshot2] [--fresh-code] [--allow-head-drift] [--validate]
# internal: nice -n 10 <interp> scripts/export_replay_archive.py --worker T8 --shard 1 --code-root <root> --out ... --work ...
#           nice -n 10 <interp> scripts/export_replay_archive.py --digest T8 --shard 1 --code-root <work>/code_head --work ...
```

### 6.2 Code roots (fixed at export start)

* **Frozen head copy.** Before any job starts, the orchestrator copies the repository's `catanbot/` (the compiled
  `catanbot_core*.so` included) and `scripts/replay_catanatron.py` into `work/code_head/`.
  * It writes `SHA256SUMS` over every copied file and a tree hash.
  * It smoke-imports under both interpreters and asserts that `catanbot.__file__` lies under `code_head`.
  * An existing copy is reused only if its tree hash matches; `--fresh-code` re-copies.
  * A live edit to the repository cannot change the export halfway through.
* **Root per test:**
  * T7–T11 use `/home/user/proof_snapshot2`, the code that played them.
  * All other tests use `code_head`.
* **Why the other tests don't use their own snapshots.** For full-information games no belief is derived. Each
  replay is checked with `check=True` and against the log's own final fingerprint (6.4), so the code version
  cannot change what the page shows without failing. The page still shows the code that played each test (1,
  table).
* **What the exporter imports from `catanbot`:** `AD.rebuild_game`, `AD.replay_log_action`,
  `AD.encode_log_entry`, `AD.state_summary`, `AD.state_fingerprint`, `AD.CB_TO_RESOURCE`,
  `AD.CATANATRON_VERSION`, `AD.API_33`, `AD.Color` and `PublicInfoTracker`.
  * The adapter names exist in `proof_snapshot`, `proof_snapshot2` and the repository (checked by grep).
  * The analysis (marginals, p, VP law, audit, digests) lives in the exporter, so it is the same for both roots.

### 6.3 Orchestrator (stdlib only, any interpreter)

* **Setup.** For every test it finds the log directory (`proof/<T>`, `bench_1v1/<T>` or `bench_multi/<T>`) and
  reads the first record's `catanatron` field. `3.3.*` uses `--py33`, `3.2.*` uses `--py32`, and anything else is
  an error.
* **Jobs.**
  * One worker job per shard, and for counted tests one head-digest job per shard.
  * Each runs as `nice -n 10 <interpreter> …`, with at most `--workers` (default 2, the cap on this machine) at
    once.
  * There is no per-job timeout below 30 minutes: the heaviest counted game takes about 10 s with all checks
    (6.6).
  * One line per finished job goes to `export.log`.
* **Merge and checks.**
  * It compares `digests/*.snap.json` with `digests/*.head.json` (5.1).
  * It merges `parts/*.rows.json` into `index.json.gz`, with rows sorted by test order, then g.
  * It asserts 9,600 rows and each test's games 0..N−1 exactly once.
  * It asserts `published` against the results files and log sha256s against the manifests.
  * It asserts that same seed ⇒ same deal id.
* **Output.** It writes `index.html` with the inlined core, prints a per-test table (games, shard bytes, bot-layer
  bytes, check totals, headSame, slowest game), and with `--validate` runs `node validate_archive.js` (9.1). It
  exits 1 on any failure.

### 6.4 Worker (catanatron imported lazily, under its code root)

* **Engine version.** It asserts `AD.CATANATRON_VERSION == rec["catanatron"]` for every record; `AD.rebuild_game`
  also refuses the wrong API.
* **Inputs.** It reads the test's results files (metadata, and per-game results by `game`) and selects games
  `[100k, 100k + 100)`. For each game:
  `encode_game(rec, geo, meta, result_row) -> (game_record, index_row, checks, truth_positions | None, digests | None)`.
* **Replay check.** `encode_game` replays with `check=True` and asserts that the final VPs, winner, `turns` and
  `AD.state_fingerprint` equal `rec["final"]`, as `replay_catanatron.check_record` does. It also asserts
  `global_geometry(game) == geo`.
* **Outputs.**
  * `shards/<T>_<kk>.json.gz` (gzip level 9, `separators=(",", ":")`)
  * `parts/…rows.json`, `parts/…checks.json`
  * `digests/…snap.json` (counted tests)
  * for the sampled games (6.5), `truth/…steps.json.gz`
  * `logs/<T>_<kk>.log`: one progress line per 10 games (game id, seconds, maxH)
* **Unit-tested functions:** `shard_path`, `test_dir`, `interpreter_for`, `code_root_for`, `global_geometry`,
  `board_code`, `encode_value(kind, value, result, ...)`, `seat_rows(state, n)` (reads `player_state` directly:
  hands, VP, public VP, longest-road length, titles), `encode_game`, `bot_view_positions` (5.2–5.6),
  `hand_marginal`, `p_true_hand`, `dev_vp_law`, `belief_digest`, `AuditedTracker`, `audit_violations`,
  `index_row`, `TEST_LABELS`.

### 6.5 Per-position truth sample

The sample is written to `truth/` and compared by the validator (9.1 step 5). It includes:

* **Random games:** `random.Random(f"{T}-20260926").sample(range(N), 4)` per test (72 games; this covers 2-player
  H1/H2 games).
* **Targeted games,** chosen by deterministic scans in the orchestrator (first match by g) and passed to the
  workers:
  * the heaviest-hypothesis games: T7-111, T7-492, T8-375, T9-208, T11-390 (from `info_max_hypotheses`);
  * per counted test, a game where the robber move that ends a run of hidden discards steals from one of the
    discarders;
  * per counted test, a game with `PLAY_MONOPOLY` in the same turn as hidden discards;
  * in R1 and in R2, a game with a 3.2.1 `DISCARD` of ≥ 9 cards;
  * a game where Longest Road changes hands on another player's turn (x code 12 changes and the actor is not the
    new holder);
  * per counted test, a game where our bot is both a steal victim and a thief.

### 6.6 Runtime

* **Revision 1 prototype** (light samples, under load): replay 0.01–0.02 s per game; encoding 0.01–0.03 s for
  full-information games; about 0.15 s per counted game with the audit (2 replays + 3 trackers).
* **Critic's heavy-tail measurement** (`time_heavy.py`, `nice -n 10`, machine load about 3.5):

  | game | actions | hypotheses | encode + bot view | audit (3 trackers) | `q` rows | bot view (raw JSON) |
  |---|---|---|---|---|---|---|
  | T7-111 | 622 | 2,232 | 3.51 s | 6.38 s | 448 | 14.8 KB |
  | T7-492 | 527 | 1,697 | 1.02 s | 1.48 s | 398 | 13.1 KB |
  | T7-260 | 399 | 1,531 | 0.61 s | 1.01 s | 341 | 11.2 KB |

* **Estimate:** 7,400 × 0.04 s + 2,200 × about 0.4 s (three trackers, marginals, VP law and the head-digest
  process) plus the tail, about 20 CPU-minutes. That is 10–15 minutes of wall time with 2 workers. The per-shard
  logs show progress, so a slow shard is visible without timeouts.

## 7. `replay_core.js` (one decoder and view-model layer, used by the page and by node)

### 7.1 Interface

```js
(function (root) {
  const RC = {
    VERSION: 2,
    COLORS: ["RED", "BLUE", "ORANGE", "WHITE"],
    RES: ["wood", "brick", "sheep", "wheat", "ore"],
    DEV: ["Knight", "Victory Point", "Road Building", "Year of Plenty", "Monopoly"],
    KINDS: [/* the 14 codes of 3.4 */],
    OPP: {c: "our bot", v: "ValueFunction", a: "AlphaBeta", s: "SameTurnAlphaBeta",
          f: "ValueFunction stand-in", b: "AlphaBeta stand-in"},
    shardPath(t, g),                 // "shards/T8_01.json.gz"
    async fetchJson(url),            // 7.2
    parseJsonBytes(u8, gunzip),      // shared by fetchJson and node (node passes zlib.gunzipSync)
    parseIndex(doc),                 // → {geo, tests, rows: [{t,g,id,seed,d,lineup,ours,win,won,vps,pvps,turns,rolls,acts,hvp,hv,dev,lr,la,def,defp,bk}]}
    parseHash(hash, index),          // → {view: "game"|"picker"|"selftest", id, k, filters, message}  (8.2)
    filterToken(filters),            // inverse of parseHash for filters
    parseBoard(code, geo),           // → {tiles:[{i,c,x,y,res,num}], ports:[{x,y,res,nodes}], robber0}
    decodeGame(g, geo),              // → Decoded (below)
    publicView(D, me),               // → PV: the public projection (below)
    botView(g, D),                   // → BV (below); null for full-information games
    stepText(src, i, opts),          // English log line; opts.view = "full" | "truth" | "bot"
    viewModel(src, bv, k, mode),     // everything the replay renders at k, as plain data (8.6)
    indexFields(D),                  // {rolls, turns, dev, lr, la, hvp, hv, def, defp, vps, pvps}, for the validator
    wilson(wins, n),                 // 95 % interval for the summary strip
  };
  if (typeof module === "object" && module.exports) module.exports = RC; else root.ReplayCore = RC;
})(typeof self !== "undefined" ? self : this);
```

* **`Decoded`** = `{n, ours, board, N, steps: [{a, kind, v, dice, rob, add: [type, seat, where], free, hidden, h,
  x, r}], snap(k)}`.
  * `r` is the roll count and `hidden` marks the hidden events of 3.5. `h` expands to per-seat 5-vectors.
  * `snap(k)` returns `{P: [{hand[5], held[5], played[5], vp, pvp, lr, LR, LA}], robber, bank[5], deck, turns,
    rolls}` from a precomputed `Int16Array((N + 1) · n · 20)`.
  * Pieces at k are rebuilt by walking `add`, as the template's `stateAt` does.
* **`PV = publicView(D, me)`** is a new object built by copying **only** public fields:
  * for every seat: hand size, development-card count, played cards, public VP, road length, titles;
  * for `me` only: hand, held cards and actual VP;
  * the pieces, robber and dice;
  * the bank, frozen at its pre-run value inside hidden-discard runs (3.5);
  * the deck size;
  * steps with `kind`, actor, public `v` (tile and victim for a robber move; for a hidden event no stolen, drawn
    or discarded card), public deltas (for a hidden event, only hand-size deltas) and public `x` codes (f = 1 and
    2, titles; f = 0 for `me` only);
  * the hidden flags.

  It never holds a reference to `D`.
* **`BV`** is built by walking `q`, `nh`, `dp` and `dv` once into per-position arrays.
  * `BV.pub(k)` returns, for the opponents, `[{seat, exact, own[5] | null, E[5], u, top, calp, vpLaw}]` plus
    `nh(k)` and `pool(k)`. It is truth-independent.
  * `BV.p(k, j)` returns the truth-dependent chance on the real hand; it is 1 for exact rows.
  * `BV.summary` is `bv.sum`.

### 7.2 `fetchJson(url)`

1. `res = await fetch(url)`. If `!res.ok`, throw `{kind: "http", status: res.status, url}`.
2. Read the bytes. If `bytes[0] == 0x1f && bytes[1] == 0x8b`, gunzip them with `DecompressionStream("gzip")`
   (the host may already have decompressed them, hence the check).
3. Otherwise the first non-whitespace byte must be `{`. If it is not, throw `{kind: "notdata", url}`: a host
   fallback that serves an HTML page with status 200 must not surface as a JSON parse error.
4. If there is no `DecompressionStream` when gzip bytes arrive, throw `{kind: "nogzip"}`.

### 7.3 `stepText` views

* **`"full"`** (full-information games): content everywhere, extending `game_viewer.py`'s templates.
  * `rolls 9 · Red +1 wood +1 ore; Blue +1 wheat` / `rolls 7 · the robber moves` / `rolls 5 · nobody produces`
  * `builds a settlement` (+ `· starting cards +1 wood +1 ore` when `h` gives the builder cards)
  * `builds a road` (+ ` (free)` when `free`)
  * `upgrades a settlement to a city`
  * `trades 4 wood for 1 ore (4:1)`
  * `moves the robber to wheat 8 and steals ore from Blue`
  * `buys a development card (Knight)`
  * `plays Monopoly on wheat and collects 5`
  * `plays Year of Plenty: wood and ore`
  * `discards wood`, and for 3.2.1 `discards 4 cards: 2 wood, 1 sheep, 1 ore`
  * `ends the turn`
* **`"truth"`** (counted game, modes "Real hands" and "Real + bot's guess"): hidden events not involving `me` get
  a tag.
  * `steals a card from Blue · hidden from our bot (it was ore)`
  * `buys a development card · type hidden from our bot (Knight)`
  * `discards a card · hidden from our bot (wood)`
* **`"bot"`** (mode "Bot's view only") takes `PV`, not `D`.
  * Hidden events read `moves the robber to wheat 8 and steals a card from Blue`, `discards a card` and `buys a
    development card`.
  * Events involving `me` keep their content, because our bot saw it.

## 8. The page (`index.html`)

### 8.1 Identity and tokens

* **Title and fonts.** `<title>Catanbot Replay Archive</title>`. The page keeps the viewer's look: Bricolage
  Grotesque (display), Atkinson Hyperlegible (body) and JetBrains Mono (data), all from Google Fonts with fallback
  stacks.
* **Theme.** It keeps the sage ground, the sea board and the template's tokens.
  * The dark set goes under `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {…} }` and again
    under `:root[data-theme="dark"]`.
  * `body { background: var(--ground) }`.
* **New tokens** are declared in bare `:root` and redefined in both dark blocks. None of them reuses a seat hue
  (red, blue, orange, white):
  * `--known` / `--known-soft` (sage green), used with a check-mark icon (inline SVG) and the text "known exactly".
  * `--differs` (violet), always paired with the glyph "≠" and a text alternative ("differs from the real count").
    It replaces revision 1's brick-red `--miss`.
  * `--meter-track` / `--meter-fill` (neutral slate), for probabilities. A probability is plain text with a
    single-hue bar, never a traffic light: a low chance right after a hidden steal is normal, and the page says
    so.
  * `--hidden` (muted slate) for hidden-event tags.
  * `--delta-up` / `--delta-down`, replacing the template's hardcoded `#2f7d32` / `#b3261e`. Those measured
    3.10:1 and 2.43:1 on the dark panel.
* **Contrast** is checked by the validator (9.1 step 9).
  * Text tokens need ≥ 4.5:1 on `--panel` and `--ground` in both themes; strokes and outlines need ≥ 3:1.
  * WHITE pieces, dots and chart lines keep the template's `--piece-edge` casing (template l.378/384) everywhere,
    the belief chart included. WHITE `#f5f5f0` on the light panel is 1.03:1 without it.
* **No libraries and no emoji.** `replay_core.js` is inlined (2.1). Wide tables sit in their own
  `overflow-x: auto` wrapper, and the body never scrolls sideways at 400 px.

### 8.2 Boot, routing and deep links

* **Boot.** `fetchJson("index.json.gz")` runs first, while a status line (`role="status"`, `aria-live="polite"`)
  reads "Loading 9,600 games…". If it fails, the page shows "Could not load the game list (index.json.gz, HTTP
  404)." or "The server returned something other than game data." with a **Retry** button.
* **Routes** (only a bare token reaches `location.hash`: letters, digits and `. _ ~ -`):
  * `#T8-123` opens game 123 of T8 at its end.
  * `#T8-123.57` opens it at position 57.
  * `#T8` opens the picker with the filters reset to test T8. This **overrides** the filters stored in
    `localStorage`.
  * `#T7.T8.T9~won~b3` opens the picker with these filters. Tokens are joined by `~`:
    * tests: `T7.T8` or `all`
    * `won` / `lost`
    * `pub` / `full` (information)
    * `hvo` / `hvx` (a win that needed hidden VP: ours / an opponent's)
    * `wpN` (won from ≤ N public VP)
    * `bN` / `bpN` (behind by ≥ N true / public VP)
    * `rA-B` (rolls)
    * `sN` (our seat)
    * `o-<key>` / `o-<key>-d` (sort)
  * `#selftest` runs the self-test (8.9).
  * An empty hash opens the picker with the stored filters.
* **Edge cases.**
  * Test ids are case-insensitive (`#t8-5` opens T8-5).
  * An unknown game shows "There is no game T8-999 (T8 has games 0–399)" above the T8 list.
  * A position past the end shows "Game T8-5 has 312 actions; showing the end".
  * An unknown test or token shows "Unknown link part 'x9'; showing all games".
* **History.**
  * Opening or closing a game uses `history.pushState` (inside try/catch).
  * Stepping updates the position with `history.replaceState`, debounced to 300 ms.
  * A `popstate`/`hashchange` listener handles back and forward.
  * The page never builds a URL from `location.href`: in the Artifact iframe that is the sandbox origin, not the
    claude.ai link.
* **Sharing.** A "Link to this moment" control shows the token (`#T8-123.57`) with the text "Add this to the
  page's address to open this moment". It copies the token with `navigator.clipboard.writeText`, falling back to a
  selected read-only input. The picker has the same control for the current filter token.
* **`localStorage`** (every access wrapped in try/catch) keeps only the last filter set, the bot-view mode and the
  per-opponent expanded state.

### 8.3 Loading, errors and races when a game opens

* **Loading.**
  * If the shard is not cached, the replay view shows a skeleton (board outline, empty scoreboard rows) and the
    status line "Loading games 100–199 of T8…".
  * The step controls stay `disabled` until the data arrives.
* **Races.**
  * Every open request gets a token (an incrementing integer). A response is dropped if its token is no longer
    current, for example after the user opened another game or pressed prev/next.
  * Shards are cached in a `Map` (up to 8), with the in-flight promise shared.
* **Errors.** A shard failure shows "Could not load shards/T8_01.json.gz (HTTP 404)." with a **Retry** button that
  re-fetches the shard without reloading the page, and a "← All games" link.
* **Old browsers.** Without `DecompressionStream`, the page shows "This browser cannot unpack the game data. Use a
  current Chrome, Edge, Firefox or Safari."
* **Prefetch.** Once a game is open, the shards of the previous and next games in the current filter order are
  fetched in the background if they are not cached.

### 8.4 Picker view

* **Header.**
  * Eyebrow line: "9,600 logged games · catanatron 3.3.0 and 3.2.1".
  * h1: "Every game of the strength proof and the two benchmarks".
  * One sentence: every game replays exactly from its action log.
  * One line on what is excluded: "Not included: ablation, league and tuning runs, and the blind controls C3 and
    C4 (SCRUTINY Q7)."
* **Completeness line per test** (in a `<details>` "Check the counts"):
  * "T8: 400 of 400 games · our bot won 234 (58.5 %), as in the run's results files and docs/PROOF.md".
  * It is computed from the index rows and compared with `tests[t].published`; the exporter already asserted
    equality.
* **Filter bar** (4.3) and summary strip (4.5).
  * On a phone the filters collapse into `<details><summary>Filters (3 on)</summary>`.
  * A **Random game** button opens a uniformly random game from the current filter.
* **Table.** 50 rows per page, with First, Previous and Next controls and "Page 3 of 25 · rows 101–150 of 1,234".
  Paging is simpler and more accessible than virtual scrolling at this size.
* **Columns:**
  * **Game**: a real link `<a href="#T8-123">T8-123</a>`
  * **Opponents**: format + label
  * **Info**: "Saw all hands" / "Public info only"
  * **Our seat(s)**: colour dot + ordinal + visually hidden colour name
  * **Winner**: dot + "ours" / colour name
  * **Final VP**: "ours 10 · best opp 8"
  * **Rolls**
  * **Dev cards bought**
  * **LR / LA**: dot + text alternative
  * **Behind by**: `def` if > 0 ("3 true VP"; the tooltip gives `defp`)
  * **Bot's avg. chance on the real hands**: `meanP` as a percentage with a neutral bar; "saw all" for
    full-information rows
  * **Hidden VP**: tag "won with 2 hidden VP" when `hvp`
* **At ≤ 900 px** each row becomes a two-line card: id, result and VP on line 1; opponents, rolls and tags on line
  2. A **"Sort by"** select appears (4.4 keys, plus ascending/descending). No sideways scrolling is needed at
  641–900 px.

### 8.5 Replay view

The layout is the template's: board and controls | scoreboard, VP chart and log, in one column under 900 px.

* **Top bar.** "← All games" (keeps the filter and page, and returns focus to the row that was opened). Then the
  game id, test label, seed and catanatron version. Then prev/next game within the current filter.
* **Position line.** "action 57 of 312 · roll 12". " · catanatron turn 15" is added only if 9.1 step 5 confirms
  the running count. Log group headers read "Roll 12 · Red" (the template's "Turn N" meant the roll count).
* **Information badge** under the h1:
  * **Counted:**
    > "Public information only. Our bot saw the dice, builds, bank and port trades, the bank's card counts, hand
    > sizes, played development cards and its own cards. It did not see stolen cards between other players, other
    > players' discards (which Colonist does show, so this is stricter than Colonist) or unplayed development
    > cards. Catanatron's bots saw every hand."
  * **Full information:** "Full information: our bot saw every hand and development card, as Catanatron's bots
    do." For T1–T6, R1 and R2 add: "(These runs predate the information switch; SCRUTINY Q7.)"
* **"What you see" note.** It replaces template l.418 and changes with the game and mode:
  * Full information: "Every hand and development card is shown because the log records the full state. In this
    game our bot saw them too, as did Catanatron's bots."
  * Counted, "Real hands": "These are the real hands from the log. Our bot did not see them; choose 'Real + bot's
    guess' to compare."
  * Counted, "Real + bot's guess": "Each opponent shows the real hand and, below it, our bot's guess at that
    moment."
  * Counted, "Bot's view only": "Only what our bot saw: hand sizes, public VP, played development cards and its
    own belief. The real hands are not used to draw this view."
* **Controls.** The template's (‹ ›, Play, scrubber, speed).
  * Below 900 px, a **sticky bottom bar** holds ‹ Back, Play/Pause, Next › and "57 / 312". The scrubber and the
    other buttons stay in the board card, and the page gets bottom padding so the bar covers nothing.
  * This is tested at 400 px with T7-111, the tallest counted game.
* **Hidden-event navigation** (counted games): "‹ Previous hidden event" / "Next hidden event ›" buttons (keys `[`
  and `]`). Occurrences are public, so they work in every mode.
* **Board, scoreboard, VP race chart** (dashed public-VP line) **and log** are the template's, fed by
  `viewModel(…)`.
  * Log rows are `<button>`s, not click-only `div`s (template l.409).
  * The `.lastact` line uses the same `stepText` view as the log.
* **Check this yourself** (a `<details>` at the end of the replay):
  ```
  Game T8-123 is line 4 of proof/T8/logs/T8_alphabeta_1v3_seed900201_g00120-00160.jsonl.gz
  (sha256 3f2a…, listed in proof/MANIFEST.sha256). Played by commit 9599eed (/home/user/proof_snapshot2;
  docs/PROOF.md, SCRUTINY Q20). Rebuild this position with a Python that has catanatron 3.3.0
  (GitHub commit ecf9311):
    python scripts/replay_catanatron.py proof/T8/logs --game 123 --action 57 --json
  ```
  * `--action K` means "stop after K actions" (the script's argparse, l.275), which is the page's position k.
  * R1 and R2 say "`python3` with catanatron 3.2.1 (PyPI)".
  * The block links SCRUTINY Q18 by name.
* **Same deal.** "Same board and development deck, other opponents: T8-123, T9-123" (links), from the index's `d`
  column.

### 8.6 Bot-view layer (counted games)

* **Mode control.** A segmented control labelled "Opponent hands" (`role="radiogroup"`) with three options: "Real
  hands", "Real + bot's guess" (the default) and "Bot's view only".
* **Rendering.** Every mode renders from `viewModel(src, bv, k, mode)`:
  * "Real hands" and "Real + bot's guess" pass `src = D`;
  * "Bot's view only" passes `src = PV` and `BV.pub`, and never calls `BV.p`.
  * 9.1 step 7 tests this.
* **Per-opponent block** (in the scoreboard row).
  * **Collapsed** (one line):
    * "7 cards · known exactly" (check icon);
    * "7 cards · bot's chance on the real hand 34 % · unsure about wheat, ore" ("Real + bot's guess"). The unsure
      resources come from `u`.
    * "7 cards · bot's top guess 41 % · unsure about wheat, ore" ("Bot's view only").
  * **Expanded** (tap the line; a `<button aria-expanded>`): a grid with fixed columns (resource icons as column
    headers) and numbers only.
    * Rows are "real" and "bot's guess". "Bot's view only" shows the guess row alone. "Real hands" shows the real
      row alone, which is the template's chips.
    * The guess row shows `E_r` with one decimal, or the counter's own integer hand when exact.
    * In "Real + bot's guess", a column where `|E_r − real_r| ≥ 0.5` gets a `--differs` outline and the "≠" glyph.
  * **Defaults.** Desktop expands uncertain opponents; phones start collapsed. When every opponent is exact, one
    line replaces the blocks: "Our bot knew every opponent's hand exactly here."
  * The **"P" wording** says it is the chance of the exact whole hand, so low values right after a hidden steal are
    normal.
* **Development-card line** per opponent. Example in "Real + bot's guess":
  > "Our bot's guess: 2 unseen development cards, about 0.43 hidden points (0 with chance 62 %, 1 with 33 %, 2
  > with 5 %). Before ruling out deals that would already have won, each card is a VP with chance 5 in 22 (23 %).
  > Real: 1. (The guess ignores how long a card was held.)"
  * In "Bot's view only" the "Real: 1." sentence is omitted.
* **What each mode hides.**
  * "Real hands" and "Real + bot's guess" show the template's truth (dev held types, "N seen · M hidden" VP
    subtitle, delta chips).
  * "Bot's view only" shows, for opponents: hand size and its delta, development-card count, public VP (in the
    scoreboard and the VP chart), and the belief.
    * Delta chips of hidden events show only the size delta ("−1 card").
    * There are no held types, no "hidden" VP subtitle and no real VP lines.
    * Our own row shows our hand, cards and actual VP, because our bot knew them.
* **Belief chart.** It shares the X axis and cursor with the VP chart; step lines use seat colours with the WHITE
  casing, and labels use theme tokens.
  * "Real + bot's guess" plots "Bot's chance on the real hand" per opponent (0–100 %).
  * "Bot's view only" plots "Bot's confidence in its top guess" (`top`).
  * X-axis ticks mark the hidden events, so drops line up with their cause.
  * "Real hands" hides the chart.
* **Summary card.** It puts the evidence first, then calibration.
  1. "Card counter re-run afterwards from the public log, with the run's own code (commit 9599eed) and settings."
  2. "Its statistics equal those the live bot logged."
  3. "1,916 inputs audited, 0 violations."
  4. "A twin given the hidden cards becomes exact at every step."
  5. "The move search only receives the public view and guesses drawn from this counter
     (tests/test_public_info.py::test_counted_player_never_reads_hidden_information swaps the hidden cards and
     requires the same choice)."
  6. The game's numbers, with the calibration sentence labelled "a consistency check, not a proof": "Our bot knew
     every opponent hand exactly at 51 % of positions. When it was unsure, it gave the real hand 41 % on average;
     if its guesses were honest they should average 42 %. A twin that sees the hidden cards scores 100 %. It never
     ruled out the real hand. Hidden from it: 15 steals, 48 discarded cards, 1 development card."

     These numbers are illustrative. In "Bot's view only" the sentences that use the real hands are dropped;
     `allExact`, the hidden counts and `maxH` remain.

### 8.7 Full-information games

* Where the belief card would be, the page shows "No belief layer: our bot saw every card in this game."
* Each opponent row is labelled "seen by our bot".
* There is no mode control.

### 8.8 Keyboard and screen readers

* **Keys** follow the template: ←/→ step, Shift jumps a roll, Space plays or pauses, Home/End, and `[`/`]` for
  hidden events.
  * They act only in the replay view.
  * They do nothing while focus is in an input, select, the segmented control, the filter chips or the table.
* **Current step.** A visually hidden `aria-live="polite"` region announces it: every step when stepping by hand,
  at most once every 3 s during Play.
* **Colour-only cells** (seat dots, LR/LA) carry visually hidden text.
* **Focus and motion.** Every button has a visible focus ring, and `prefers-reduced-motion` disables the pulse.

### 8.9 `#selftest`

The published page runs this against its own host:
1. It fetches the index and shard `_00` of every test.
2. It reports which decode path each file took (gzip bytes or already decompressed).
3. It decodes every game in those shards and compares `indexFields(D)` with the index rows.
4. It prints "selftest: 18 shards, 1,800 games, 0 mismatches" (or the mismatches) in the status panel.

## 9. Validation

### 9.1 Node validator (`scripts/replay_archive/validate_archive.js`)

```
node scripts/replay_archive/validate_archive.js --bundle $S/replay_archive --repo /home/user/ClaudeTesting2 \
     --truth $S/replay_archive_work/truth --core scripts/replay_archive/replay_core.js
```

1. **Bundle limits.**
   * At most 255 files, total ≤ 64 MB, every file ≤ 15 MB; every shard named by the index exists.
   * No `.js` file in the bundle.
   * `index.html` has a `<title>`, no `<html>/<head>/<body>` and no `<script src>`.
2. **Inlined core.** The bytes between the `BEGIN`/`END replay_core.js` markers equal `--core` byte for byte, and
   the validator `require()`s that same file.
3. **Index and provenance.**
   * 9,600 rows; each test's games are 0..N−1 exactly once, matching `tests[t].games`; every row's shard holds
     that game.
   * The validator reads `<suite>/<T>/results/*.json` itself: games, wins and each game's winner must equal the
     index.
   * Each `tests[t].files[i].sha256` equals the suite's `MANIFEST.sha256` entry.
4. **Every game**, decoded with `decodeGame` from the bundle.
   * Its final `snap(N)` is compared with the **original log's** `final.state` and `final`, read directly with
     `zlib.gunzipSync` from `proof|bench_1v1|bench_multi/<T>/logs`, so the check does not depend on the exporter.
   * The comparison covers hands, development cards held and played, VP, public VP, longest-road length, titles,
     settlements, cities, roads, robber, bank, deck left, `turns`, `vps` and winner.
   * `indexFields(D)` must equal the index row.
   * Revision 1's prototype of this check gave 261/261.
5. **Per-position sample** (6.5; the truth file holds k = 0..N).
   * Every position must match `snap(k)`, the pieces and catanatron's running `num_turns`.
   * For counted games, `BV.pub(k)` and `BV.p(k, j)` must match the Python belief: p, top and calp within 0.3 %
     relative, E within 0.005, exact, own, u and nh equal, pool equal, and vpLaw within 0.001.
6. **Bot-view invariants, every counted game, every position.**
   * `|Σ E − size| ≤ 0.03`
   * `lp`, `lt` and `lc` are integers ≥ 0, and `lt ≤ lp` (top ≥ p; calp has no order with p)
   * exact rows: own hand = the decoded true hand, and `u = 0`
   * `pool` = `[14,5,2,2,2]` − Σ played − our held, recomputed from the decoded public state
   * `Σ pool = deck + Σ_opponents held`
   * every `dv` row sums to 1000 over `0..dev_count`
   * `meanP`, `minP`, `allExact`, `meanPu` and `calPu` are recomputed from `q` and equal `bv.sum` within 1e-3
   * `chk` is all ones, with 0 audit violations
7. **Swap test** (the bot view never reads hidden fields; every sampled counted game). The validator applies
   mutations that change only hidden fields:
   * (M1) at each position, one card moves from opponent A to opponent B and one card of another type back, so
     sizes and bank stay the same;
   * (M2) opponents' held development types are relabelled, keeping each count, together with their actual VP;
   * (M3) every stolen card in steals not involving `me`, every hidden discard card and every opponent's drawn
     card are changed, and so is the bank inside hidden-discard runs.

   It then requires `JSON.stringify(viewModel(publicView(D'), BV.pub, k, "bot"))` and the bot-mode log text to be
   identical to the unmutated ones at every k. This mirrors the decision-side test in 5.8.
8. **Text check.** In bot mode, no log line, `.lastact` text or chip label for a hidden event not involving `me`
   names a resource or a development-card type.
9. **Contrast.** It parses the three token blocks of `index.html` (`:root`, the media-query block,
   `[data-theme="dark"]`) and checks, in both themes, `--known`, `--differs`, `--hidden`, `--delta-up`,
   `--delta-down`, `--muted` and `--ink` against `--panel` and `--ground` (≥ 4.5:1), and `--meter-fill` and the
   WHITE casing (≥ 3:1).
10. **Parsers.**
    * `parseHash`: `#t8-5`, `#T8-999`, `#T8-5.9999`, `#T8`, `#T7.T8~won~b3`, `#x9`, `#selftest`.
    * `filterToken` round-trips.
    * `fetchJson` is fed gzip bytes, plain JSON bytes, an HTML page with status 200 (→ `notdata`) and a 404
      (→ `http`), through an injected `fetch`.
11. It prints one line per test (`T8: 400 games OK, 6 sampled games × 301 positions OK, swap OK`) and exits 1 after
    listing the first 20 mismatches.

### 9.2 Python per-position truth (written by the worker, not published)

`truth/<T>_<kk>.steps.json.gz = {"<id>": {"pos": [truth_k for k in 0..N], "bv": [belief_k for k in 0..N]}}`.

* `truth_k` is `AD.state_summary(game.state)` reduced to lists (roads as `geo.edges` indices) plus
  `state.num_turns`, independent of `seat_rows`.
* `belief_k` holds each opponent's `(exact, own, E, u, p, top, calp, vpLaw)`, `nh` and `pool`, read at the same
  loop line as `truth_k` (5.2).

### 9.3 pytest (`tests/test_replay_archive.py`)

The tests run under either interpreter. Version-specific tests pick a log of the running generation (R1 for
3.2.1; T8 and T4 for 3.3) and skip otherwise, using `AD.API_33`. Tests that need the snapshot skip if
`/home/user/proof_snapshot2` is absent.

* `test_shard_path` (`T7`, 537 gives `shards/T7_05.json.gz`) and `test_interpreter_for` (R1 gives py32, T8 gives
  py33; an unknown version raises a clear error)
* `test_code_root_for` (T7–T11 give the snapshot, others give `code_head`) and `test_worker_imports_snapshot` (a
  subprocess worker for one T8 game reports `catanbot.__file__` under the snapshot and the two sha256 prefixes)
* `test_board_code_roundtrip` (parsing the code back gives the record's resources, numbers and port resources)
* `test_geometry_global` (games from 3.2.1 and 3.3, 2-player and 4-player, give an equal `global_geometry`)
* `test_encode_value_table` (every kind, including a 3.2.1 `DISCARD` list, a robber move without a victim and a
  one-card Year of Plenty)
* `test_encode_game_final_state` (2 games: a pure-Python mini-decoder of `s` gives `final.state`)
* `test_belief_indexing_hidden_steal`: the first T8 game with a hidden steal between two opponents at action index
  i, where the victim held ≥ 2 resource types. The victim's belief changes at position i + 1, not at i or i + 2,
  and the truth file's size for the victim drops by 1 at i + 1.
* `test_bot_view_audit` (2 T8 games: 0 violations, twin exact, logged stats equal, `p > 0`, `exact ⇒ own == truth`)
* `test_audit_catches_leak` (a tracker subclass that passes the stolen card to `observe_steal` for a third-party
  steal gives ≥ 1 violation; one that calls `counter._reset` gives ≥ 1 violation)
* `test_p_true_hand_pending` (a hand-built counter with a pending discard of 2: the hypergeometric marginal equals a
  brute-force enumeration)
* `test_dev_vp_law` (3 positions from T7-111, including one with an opponent at 9 public VP: `dev_vp_law` equals
  the frequencies of 20,000 calls of the snapshot's `_deal_devs` within 0.01)
* `test_worker_rejects_wrong_engine` (a 3.2.1 record under 3.3 raises `ValueError` naming the interpreter)
* `test_node_validator` (skipped without `node`: export 2 games into `tmp_path` and run the validator in a
  bundle-only mode; exit code 0)

### 9.4 Acceptance

* **Exporter.** It exits 0 for all 18 tests: 9,600 games, 0 replay mismatches, and index = results files. For all
  2,200 counted games: stats equal, 0 audit violations, twin exact, belief identical, and head digests equal
  (`headSame` 2,200/2,200).
* **Validator and pytest.** The validator exits 0 and the pytest file passes under both interpreters.
* **Bundle.** It is within the limits.
* **After publishing, on the published claude.ai URL** (not a local server):
  * `#selftest` reports 0 mismatches.
  * `#T8-123` opens T8 game 123.
  * `#T8-123.57` opens position 57.
  * Filters and sort change the counts.
  * The mode control cycles through its three modes.
  * "Link to this moment" copies the token.
* **Manual layout check** at 400 px and at desktop width, in both themes, with T7-111 (tallest) and one 2-player
  game: no sideways page scroll, and the sticky bar never covers content.

## 10. Risks and decisions

* **The snapshot must stay in place.** The export depends on `/home/user/proof_snapshot2`. The sha256 asserts
  make a changed snapshot fail loudly, and `tests[t].code` records what was used.
* **Head drift.** If the repository's tracker changes behaviour later, `headSame` shows it. The page keeps showing
  the snapshot's belief, the one the bot had.
* **Host serving of `.gz`.** The magic-byte check makes both raw and pre-decompressed serving work. If `.gz` is
  refused, use the plain-JSON fallback of 3.6; `#selftest` shows which path was taken.
* **Explicit deltas and stored VP/titles over rule re-implementation.** This costs about 14 % of size; in exchange
  the JS core has no game rules. Everything it derives was validated on 261 games of both generations.
* **Calibration is a statistic, not a proof** (5.8 g). The evidence is (a)–(f) for the belief layer and the cited
  test for the decisions.
* **The dev-card law is the sampler's law, not a belief the bot stored.** The bot never held a VP-card
  distribution; it sampled deals. 5.4 computes the exact law of those samples. The pytest compares it with the
  sampler itself.

## 11. Rejected or modified critique points

* **Critique 1.1, "use `/home/user/proof_snapshot` for T1–T6, R1, R2" (offered as an option).** Modified: all
  non-T7–T11 tests use one frozen copy of the repository's code, made at export start. Full-information games
  derive no belief, and each replay is verified against the log's own final fingerprint, so one well-tested API
  for every test is less risky than three code generations. The page still names the code that played each test.
* **Critique 1.2, truth-free number per row.** The critique offered `calp` or the probability of the bot's most
  likely hand. Both are stored (`lc`, `lt`). The page shows `top` in "Bot's view only", because "chance of its top
  guess" is plain language and `Σ P(h)²` is not.
* **Critique 2.9, "as in docs/PROOF.md".** The exporter and the validator compare against the results files that
  PROOF.md is computed from, game by game. They do not parse PROOF.md prose; the page cites PROOF.md.
* **Critique 2.13, "was behind by ≥ N (any result)".** Modified: the deficit excludes the final turn (4.1). A
  deficit that includes the winner's finishing burst is large in almost every loss by construction, which would
  make the "any result" win rate select by outcome after all.
* **Critique 2.15, "plain text for P".** Both are used: plain text plus a neutral single-hue bar.
* **Critique 2.6, "prefetch the neighbouring shard when prev/next nears a 100-game boundary".** Replaced by
  prefetching the shards of the actual previous and next games in the current filter order. The filter order, not
  the 100-game boundary, decides which shard comes next.

## 12. Prototype files, critic checks and commands (scratch, not in the repository)

`$S/proto/`:

* `survey.py`: all logs, kinds and layouts. `python3 survey.py`, 3.1 s.
* `proto_encode.py`: layouts A and C plus the revision 1 bot view (repository code, not the snapshot).
  * 3.3: `nice -n 10 /home/user/venv_cat33/bin/python proto_encode.py <T> <n> --out out_<T>.json` for T8 40,
    T7 25, T11 20, T9 20, T4 24, M3 20, H1 24, T1 20, M1 24.
  * 3.2.1: `nice -n 10 python3 proto_encode.py R1 24` and `R2 20`.
* `measure.py`: table 3.6 and the extrapolation.
* `proto_index.py`: the index size.
* `proto_audit.py`: 5.8 (c) and (d). Run as `… proto_audit.py T8 12` and `… T11 8`.
* `proto_decode.js`: node decoding against the logs' final state (`node proto_decode.js out_<T>.json <logs dir>`
  for 11 tests, 261/261).

`$S/critic_archive/` (critique 1): `digest_tracker.py <code_root> <test> <n_per_file> <out.json>`, run with
`nice -n 10 /home/user/venv_cat33/bin/python` for the snapshot and the repository. Snapshot and head digests are
equal on 40/40 games (`snap_T8.json` = `head_T8.json`, `snap_T11.json` = `head_T11.json`; re-compared for this
revision). `time_heavy.py` produced the heavy-tail table in 6.6.

Checks run for this revision (read-only, from `/home/user/ClaudeTesting2`):

* `nice -n 10 python3 $S/hvp_check.py` (2.3 s): the hidden-VP table in section 1.
* An inline script over every results file gives:
  * the seed groups (4,000 × 1, 1,000 × 2, 400 × 3, 400 × 6);
  * wins per test;
  * the Wilson interval for T8 (53.6–63.2 %);
  * the maximum hypotheses per counted test, with 0 `info_resets` and 0 `info_errors`.
* An inline script over the T8 logs: mean 80.8 rolls per game.
* An inline script over all 130 log files: games a … b−1 in line order in every file.
* `diff` of `public_info.py` / `counting.py` between the snapshot and the repository (369 / 59 changed lines).
  `sha256sum` of the snapshot's `public_info.py`, `counting.py` and `catanatron_adapter.py`.
* grep of the three adapters for the functions listed in 6.2, and of `scripts/replay_catanatron.py` for
  `--action`.
