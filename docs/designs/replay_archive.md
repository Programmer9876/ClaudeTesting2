# Replay archive: game picker and "what our bot knew" (design spec)

Status: design, 2026-09-26. The measurements below come from the scratch prototypes listed in section 11.
They were run on the real logs. Nothing in `catanbot/`, `proof/`, `bench_1v1/` or `bench_multi/` changes.

## 0. What gets built

One static page, **Catanbot Replay Archive**. It lists all 9,600 logged games. Any game opens in a
step-through replay that reuses `scripts/game_viewer_template.html`'s board, controls, scoreboard, VP chart and
log. For the Colonist-information tests (T7, T8, T9, T11) the page also shows, at every step and for every
opponent, the true hand next to what our bot believed. That belief is re-derived by running the bot's own
`PublicInfoTracker` over the public log with the test's settings. For full-information games the page says plainly
that our bot saw every hand.

The page is published as a claude.ai Artifact with supporting data files.

| | measured / extrapolated |
|---|---|
| games | 9,600 in 18 tests (4 × 1,000 + 14 × 400), 0 crashed, 0 without a winner, 2,905,082 actions (max 654 in one game) |
| game shards | 96 files (`shards/<T>_<kk>.json.gz`, 100 games each). **11.9 MB gzip in total** (60.4 MB as raw JSON) |
| largest shard | T7, about 270 KB gzip / 1.1 MB JSON (100 counted games at 2.67 KB gzip each) |
| index | 1 file (`index.json.gz`), 9,600 rows, 994 KiB JSON / 172 KiB gzip |
| bundle | 99 files, about 12.2 MB. Limits are 255 files and 64 MB per publish, 15 MB per binary file |
| export time | 0.03–0.05 s per full-info game, about 0.15 s per counted game with all checks. About 12 CPU-minutes in total |

## 1. Facts established from the data (survey of every log, `proto/survey.py`)

* Every game is on the same **BASE topology**, and the land-tile and port-tile coordinate order is the same in
  every log. The node and edge ids and all positions are identical between catanatron 3.2.1 and 3.3.0, and between
  2-player and 4-player games (`proto_encode.py` asserts `global_geometry(game) == geo` for every sampled game;
  3.2.1 geometry == 3.3.0 geometry: True). The geometry has 96 nodes and 132 edges (water-ring nodes included)
  plus 19 land tiles and 9 ports: 2,986 bytes of JSON, stored once.
* Seats are always coloured `RED, BLUE, ORANGE, WHITE` in seat order (the first `n` of them). `vps_to_win` = 10,
  `discard_limit` = 7 and the turn limit is 1,000 everywhere.
* Action kinds: 3.3 has `BUILD_SETTLEMENT BUILD_ROAD BUILD_CITY ROLL END_TURN BUY_DEVELOPMENT_CARD MOVE_ROBBER
  MARITIME_TRADE DISCARD_RESOURCE PLAY_KNIGHT_CARD PLAY_MONOPOLY PLAY_YEAR_OF_PLENTY PLAY_ROAD_BUILDING`. 3.2.1
  has the same set, except that a discard is one `DISCARD` whose value is the list of cards. No trade-offer kinds
  appear (every test ran with `trades: off`).
* catanatron's `final.turns` = number of `END_TURN` + 2n − 2 (checked on 340 games: +6 for 4 players, +2 for 2
  players). The page's "turn N" is the number of `ROLL`s so far, as in `game_viewer`.
* Information mode comes from the results metadata `info`:
  * `{"mode": "counted", "samples": 4, "discards_public": false}` for T7, T8, T9 and T11 (exactly one catanbot
    seat each). The same dict is in the log record at `players[our_seat].info`.
  * `{"mode": "full"}` for T10, H1, H2 and M1–M3.
  * absent for T1–T6, R1 and R2. Those tests predate the flag and are full information.
* Opponent presets are `value`, `alphabeta` and `sameturn` (catanatron 3.3 classes), plus `vf` and `ab` (the
  catanbot 3.2.1 stand-ins, R1 and R2). T10, T11 and M3 records carry `lineup` (a preset per seat).
* Per-game tracker statistics logged during the run are in `results[*].info_stats` (and also in `stats`):
  `hidden_steals`, `hidden_discards`, `hidden_dev_draws`, `info_resets`, `info_max_hypotheses`, `info_uncertain`
  and `info_samples`.
* API note: `CardCounter.expected` and `CardCounter.exact` are **properties**, not methods. Use
  `PublicBelief.expected_hand(j)` and `PublicBelief.is_exact(j)`, which account for hidden discards that are still
  pending.

## 2. Files

### 2.1 New repository files (sources)

| path | role |
|---|---|
| `scripts/export_replay_archive.py` | exporter (orchestrator + per-shard worker), stdlib-only at import time |
| `scripts/replay_archive/index.html` | the page (static; no data inside). Copied into the bundle |
| `scripts/replay_archive/replay_core.js` | shared decoder, UMD: `window.ReplayCore` in the page, `require()` in node. Copied into the bundle |
| `scripts/replay_archive/validate_archive.js` | node validator (CommonJS, node ≥ 18, no npm packages) |
| `tests/test_replay_archive.py` | pytest for the exporter |
| `docs/designs/replay_archive.md` | this spec |
| `docs/USAGE.md` | a short "Replay archive" section (export, validate, publish commands) added once the scripts exist |

`scripts/game_viewer.py` and its template stay unchanged. The new page copies the template's CSS tokens and drawing
code into `index.html` and does not import them.

### 2.2 Bundle (published as is)

`$S = /tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad`

```
$S/replay_archive/
  index.html                 page (<title>Catanbot Replay Archive</title>, own <style>, no <html>/<head>/<body>)
  replay_core.js             shared decoder
  index.json.gz              geometry + test metadata + 9,600 index rows
  shards/T1_00.json.gz       games 0-99 of T1
  shards/T1_01.json.gz       games 100-199 ...
  ...                        96 shard files: T1, T4, T7, R1 have _00.._09; the other 14 tests have _00.._03
```

Shard of a game: `shards/${t}_${String(Math.floor(g / 100)).padStart(2, "0")}.json.gz`.
The Python function is `shard_path(t, g)` and the JS function is `ReplayCore.shardPath(t, g)`. Both are tested.

### 2.3 Work files (never published)

```
$S/replay_archive_work/
  parts/<T>_<kk>.rows.json     index rows of one shard (merged into index.json.gz)
  parts/<T>_<kk>.checks.json   per-game check results (audit counts, stats equality, twin, timings)
  truth/<T>_<kk>.steps.json.gz Python per-step truth for the sampled games (section 9.2)
  export.log                   orchestrator log (one line per shard job)
```

## 3. Per-game encoding

### 3.1 Global geometry (once, inside `index.json.gz` as `geo`)

`geo = {"nodes": [[x, y] × 96], "edges": [[a, b] × 132] (sorted, a < b), "tiles": [[[x,y,z], cx, cy] × 19],
"ports": [[cx, cy, [n1, n2]] × 9]}`. The values come from `game_viewer.geometry(game)` (unit circumradius,
pointy-top, y down) and the tiles' edge sets. `tiles` and `ports` follow the order of the land and port tiles in
`rec["board"]["tiles"]` (catanatron order). The exporter computes `geo` from the first game. For every game it
asserts that the land and port coordinate lists (and the port directions) equal the first game's. That makes it
the same topology, because `decode_board` walks the BASE template in a fixed order.

### 3.2 Board code (per game, 48 characters)

`b = <land: 19 × 2 chars> "|" <ports: 9 chars>`

* land tile *i* (the order of `geo.tiles`): a resource char from `W B S G O D` (wood, brick, sheep, grain/wheat,
  ore, desert), then a number char from `2 3 4 5 6 7 8 9 A B C` (A = 10, B = 11, C = 12). The desert uses `-`.
  Example: `O8` is ore 8 and `D-` is the desert.
* port *i* (the order of `geo.ports`): one of `W B S G O 3` (`3` = generic 3:1).
* The robber starts on the desert. The decoder derives this and the validator checks it against
  `rec.board.robber`.

### 3.3 Game record (the element of a shard's `games` array)

```json
{"id": "T8-123", "g": 123, "b": "O8B5...|3WG3...", "n": 4, "o": [0],
 "s": [[op, v, h, x], ...],
 "bv": { ... }}                       // counted games only (section 5)
```

`n` is the number of players, `o` is our seats and `s` has one step per logged action, in log order. Step *i*
(0-based) is logged action *i*, and "position k" means the state after k actions (k = 0 … N).

Shard file: `{"v": 1, "t": "T8", "from": 100, "to": 200, "games": [ ... sorted by g ... ]}`. Game `g` is at
`games[g - from]`.

### 3.4 Step tuple `[op, v, h, x]`

Trailing empty parts are dropped: `[op]`, `[op, v]`, `[op, v, h]`, `[op, v, h, x]`. A missing middle part is
`null` (for `v`) or `0` (for `h`), e.g. `[36, null, 0, [13, 0]]` (a Knight that takes Largest Army).

* `op = kind * 4 + seat`, which is always below 56. The kind codes are fixed:

| code | kind | `v` |
|---|---|---|
| 0 | BUILD_SETTLEMENT | node id |
| 1 | BUILD_ROAD | index into `geo.edges` |
| 2 | BUILD_CITY | node id |
| 3 | ROLL | `10 * d1 + d2` (e.g. 63) |
| 4 | END_TURN | — |
| 5 | BUY_DEVELOPMENT_CARD | drawn card 0–4 (`KNIGHT, VICTORY_POINT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY`), from the result, else the value |
| 6 | MOVE_ROBBER | `tile * 100 + (victim_seat + 1) * 10 + (stolen_res + 1)`. 0 in a slot means none. 3.3: stolen = result. 3.2.1: stolen = `value[2]` |
| 7 | MARITIME_TRADE | `give_res * 100 + ratio * 10 + get_res` (ratio = number of non-null give slots, 2/3/4) |
| 8 | DISCARD_RESOURCE (3.3) | resource 0–4 |
| 9 | PLAY_KNIGHT_CARD | — (the robber move is the next step) |
| 10 | PLAY_MONOPOLY | resource |
| 11 | PLAY_YEAR_OF_PLENTY | `[r1, r2]` (or `[r1]`) |
| 12 | PLAY_ROAD_BUILDING | — |
| 13 | DISCARD (3.2.1) | counts `[w, b, s, g, o]` (max 7 of one kind seen) |

  Resources are `0..4` = `WOOD, BRICK, SHEEP, WHEAT, ORE` (`AD.CB_TO_RESOURCE`).
* `h` holds the hand deltas of **every** seat, explicit and sparse: a flat list `[seat * 5 + res, delta, ...]` in
  seat and resource order, or `0`. They are stored for every kind, including build costs, trades, steals,
  discards and Monopoly. That way the JS core never re-implements catanatron's production, bank-shortage or
  free-road rules. Deriving the non-roll deltas would save only 9–16 % gzip (measured), which does not justify the
  risk.
* `x` holds state changes, stored only when a value changed, as a flat list `[code, value, ...]`:
  `code = f * 4 + seat` for f = 0 (actual VP), 1 (public VP) and 2 (longest-road length), then `12` = Longest
  Road holder (seat or −1) and `13` = Largest Army holder (seat or −1).

### 3.5 What the JS core derives (and the validator checks against the engine)

Pieces: settlement node, city node (replaces the settlement) and road edge come from `v`. The robber comes from
`v / 100`. Development cards: `BUY` adds 1 to held[card], and `PLAY_*` subtracts 1 from held and adds 1 to played
(Knight 0, Road Building 2, Year of Plenty 3, Monopoly 4). VP cards stay held. The bank is 19 minus the sum of all
hands per resource. The deck left is 25 minus the number of buys. Rolls are the number of `ROLL`s. Turns are the
number of `END_TURN`s + 2n − 2. Dice are `v / 10` and `v % 10`. A `BUILD_*` without `h` is free (setup or Road
Building), which the log text uses. **Prototype result**: `proto/proto_decode.js` decoded all 261 sampled games
(11 tests, both engine generations) to exactly the logs' own `final.state` (hands, development cards held and
played, VP, public VP, longest-road length, titles, settlements, cities, roads, robber, bank, deck, turns, VPs).
There were 0 mismatches.

### 3.6 Measured sizes (layout A = the tuples above; `proto/measure.py`)

The samples are the first games of every log file of a test (spread over its chunks). "gzip/shard" is the gzip -9
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

The extrapolation to all 9,600 games uses T1 for T2, T3 and T10, T4 for T5 and T6, H1 for H2, and M1 for M2:
**60.40 MB JSON, 11.90 MB gzip**. A columnar alternative (kinds, values, deltas and changes in separate arrays)
measured 16.35 MB gzip. Gzipping each game separately gives 19.3 MB. Layout A in 100-game shards is the choice.
Coarser bot-view quantisation (expected counts × 10, probability × 1000) would save only 12–18 % of the bot layer,
so the finer quantisation stays.

**Hosting budget**: 96 shards + index + page + core = 99 files (limit 255), about 12.2 MB (limit 64 MB per
publish), largest file about 0.3 MB (limit 15 MB). Shards are published with `contentType: "application/gzip"`.
Fallback, only if the host refuses `.gz`: publish plain `.json` shards (60.4 MB). That fits one publish only
barely, so split it over two publishes to the same URL (the version limit is 256 MB). The page code is the same
either way (section 7, `fetchJson`).

## 4. Index (`index.json.gz`) and picker data

```json
{"v": 1, "built": "<UTC ISO time>", "shardSize": 100, "geo": {...},
 "tests": {"T8": {"suite": "proof", "label": "3× AlphaBeta", "group": "Proof · Colonist information",
                  "fmt": "1v3", "opp": "alphabeta", "info": {"mode": "counted", "samples": 4, "discards_public": false},
                  "catanatron": "3.3.0", "spec": "search:depth=1,beam=4,expand=8,evaluator=heuristic",
                  "seed": 900201, "games": 400, "shards": 4,
                  "counted": {"allExact": 0.51, "meanP": 0.771, "calP": 0.772, "devBelief": "uniform pool"}}, ...},
 "cols": ["t","g","seed","lineup","win","vps","pvps","turns","rolls","acts","hvp","dev","lr","la","def","bk"],
 "rows": [["T8",123,90022800604,"caaa",2,[7,9,10,6],[6,9,10,6],124,119,456,0,8,2,0,3,[0.51,0.77,0.0044,0.77,23,49,36]], ...]}
```

| col | meaning |
|---|---|
| `t`, `g` | test id and game index (`id = t + "-" + g`; the deep link is `#T8-123`) |
| `seed` | the game seed (`rec.seed`) |
| `lineup` | one char per seat: `c` catanbot (ours), `v` value, `a` alphabeta, `s` sameturn, `f` vf (3.2.1 stand-in), `b` ab (3.2.1 stand-in). Our seats and colours come from it |
| `win` | winner seat (−1 = none; 0 today). Our win = `lineup[win] == "c"` |
| `vps`, `pvps` | final actual and public VP per seat |
| `turns`, `rolls`, `acts` | catanatron `num_turns`, number of `ROLL`s, number of actions |
| `hvp` | 1 if the winner's VP > its public VP at the end (a hidden-VP win) |
| `dev` | development cards bought by our seats |
| `lr`, `la` | holder of Longest Road / Largest Army at the end (seat or −1) |
| `def` | largest deficit at any position, in **actual** VP: max over k of (best opponent VP − best of our VPs), floored at 0. "Came back from N" = our win with `def ≥ N` |
| `bk` | counted games: `[allExact, meanP, minP, calP, maxH, searched, uncertain]` (section 5.3). The last two come from the run's results file (`stats.searched`, `info_stats.info_uncertain`). `null` for full-info games |

Format, information mode, opponent preset(s) and catanatron version come from `tests[t]`. `label` and `group` come
from a static `TEST_LABELS` table in the exporter, which is cross-checked against the metadata (opponent, format,
info mode, version). Groups are: *Proof · full information* (T1–T6, T10), *Proof · Colonist information* (T7–T9,
T11), *Replication · catanatron 3.2.1* (R1, R2), *Heads-up 1v1* (H1, H2), *Several of our seats* (M1–M3).

**Picker filters** (all in memory; 9,600 rows filter in well under 10 ms):

* test (multi-select chips, grouped as above)
* format (`1v3`, `1v3-mixed`, `2v2`, `2v2-mixed`, `1v1`, `3v1`)
* information (full / counted)
* opponent type (any seat; ValueFunction, AlphaBeta, SameTurnAlphaBeta, 3.2.1 stand-ins)
* result (our win / our loss)
* our seat (1st–4th; any of ours)
* hidden-VP win (yes / no)
* came back from ≥ N VP (0–5+)
* rolls between min and max
* catanatron version
* free text: an id (`T8-123`), a game number or a seed

**Sort** by column header (click to toggle asc/desc): game (test order, then g), seed, rolls, turns, our VP, best
opponent VP, margin (our best − their best), deficit, development cards bought, `meanP` or `allExact` (counted
only; sorting by "hardest to count" = lowest meanP is the skeptic's entry point).

**Summary strip** (updates with the filter): "1,234 games · our bot won 745 (60.4 %) · 25.0 % expected by chance ·
mean 72 rolls · 31 hidden-VP wins". The chance rate is the mean over rows of `|our seats| / n`.

## 5. "What our bot knew" (counted games: T7, T8, T9, T11)

### 5.1 Derivation: the same tracker, the same settings, in lockstep

For each counted game (`info.mode == "counted"`, exactly one catanbot seat `me`):

1. Settings come from the test's results metadata `info`. The exporter asserts that the value equals
   `rec.players[me].info` and that `reveal_devs` is absent or false. It also asserts
   `catanbot.devbelief.ENABLED == 0`, the code default: the results metadata does not record this switch, and
   with 0 it only affects the determinizations' deal, never the counter. The tracker is built exactly as
   `CatanbotPlayer._begin` builds it:
   `PublicInfoTracker(Color(colors[me]), discards_public=info["discards_public"], vps_to_win=rec["vps_to_win"])`
   (default `max_hypotheses` = 4096, `reveal_hidden=False`, `reveal_devs=False`).
2. Pass 1: `final = AD.rebuild_game(rec)`, then `AD.replay_log_action(final, item, check=True)` for every action.
   `replay_log` needs the complete engine log.
3. Pass 2 is the lockstep. The display comes from the plain tracker. The same entry also drives two check
   trackers:
   ```python
   game  = AD.rebuild_game(rec)                                   # the truth, re-played with check=True
   plain = PublicInfoTracker(...);            g1 = plain.replay_log(final.state)
   audit = AuditedTracker(...);               g2 = audit.replay_log(final.state)       # 5.4 (b)
   twin  = PublicInfoTracker(..., reveal_hidden=True); g3 = twin.replay_log(final.state)  # 5.4 (d)
   for i, item in enumerate(rec["actions"]):
       AD.replay_log_action(game, item, check=True)
       entry, _ = next(g1); next(g2); next(g3)
       assert AD.encode_log_entry(entry) == list(item)            # the tracker saw exactly logged entry i
       truth = hands of game.state                                 # the truth is read from `game`, never from a tracker
       ... record plain's belief after entry i ...
   assert every generator is exhausted
   ```
   `replay_log` is the path the bot itself took: `start()` calls it, and `follow()` calls the same `step()` per
   entry.
4. Belief after step i, for each opponent j (the truth is used only as the point at which the belief is scored):
   * `size_j` = true hand size (public).
   * `exact_j = plain.is_exact(j)`.
   * `E_j = plain.expected_hand(j)` (5 floats; pending hidden discards removed in proportion).
   * `p_j = p_true_hand(plain, j, truth_j)`, which is:
     `Σ_{joint, w ∈ counter.hyps} w · 1[joint[j] == truth]`. If `k = pending_discards[j] > 0`, the indicator is
     replaced by the multivariate hypergeometric `Π_r C(h_r, d_r) / C(|h|, k)` with `d = joint[j] − truth`
     (0 when some `d_r < 0` or `Σ d ≠ k`). This is the marginal version of `PublicBelief.support_weight`.
   * `calp_j = Σ_h P(h)²` over the same marginal (post-discard hands when pending). This is the probability the
     bot's own posterior expects to place on the truth, used for calibration.
   * `nh = plain.counter.num_hypotheses` (joint hypotheses held).
   * `pool = plain.dev_pool()` (the 25-card deck minus every played card minus our own cards). The number of
     unknown development cards of opponent j is `dev_count[j]` (public), and P(a given unknown card is a VP) is
     `pool[1] / Σ pool`. This is the uniform-pool belief, since development-card belief was off in the proof.

   Per-step asserts in the exporter:
   * `exact_j ⇒ round(E_j) == truth_j`
   * `p_j > 0` (the truth is never excluded)
   * `|Σ E_j − size_j| < 1e-6`
   * `plain.known_dev[j] is None` for every opponent
   * `plain.pool_is_consistent()`
   * `pool == [14,5,2,2,2] − Σ_all played − our held` (the public formula the page recomputes)
   * `plain.counter.stats["resets"] == 0`

### 5.2 Sparse per-step encoding (`bv` in the game record)

```json
"bv": {"q":  [[i, j, exact, p10000, e0, e1, e2, e3, e4], ...],
       "nh": [i, count, i, count, ...],
       "dp": [[i, p0, p1, p2, p3, p4], ...],
       "sum": {"allExact": 0.51, "meanP": 0.771, "minP": 0.0002, "calP": 0.772, "maxH": 23,
               "hidden": [15, 48, 1]},
       "chk": {"stats": 1, "audit": [1916, 0], "twin": 1, "same": 1}}
```

* `q` has one row each time opponent j's record `(exact, p, E)` changes after step i. For an exact record the
  row is `[i, j, 1, 10000]` and E is omitted (it equals the true hand). Otherwise `p10000 = max(1,
  round(p · 10000))` and `e_r = round(E_r · 100)`. Before step 0 every opponent is exact with p = 1.
* `nh` is the joint hypothesis count on change (initially 1). `dp` is the dev pool on change (initially
  `[14,5,2,2,2]`).
* Measured: 104–177 `q` rows per game; the bot layer is 885–1,613 B gzip per game (table 3.6).

### 5.3 Per-game summary (`bv.sum`, copied into the index as `bk`) and per-test aggregate

* `allExact`: share of positions 1..N at which every opponent hand was known exactly.
* `meanP`: mean over positions and opponents of `p_j`.
* `minP`: the minimum of `p_j`.
* `calP`: mean of `calp_j` over the same positions and opponents.
* `maxH = counter.stats["max_hypotheses"]`.
* `hidden = [hidden_steals, hidden_discards, hidden_dev_draws]` over the whole game.
* `tests[t].counted` holds the means over the test's games.

Prototype values (sampled games, exact including pending discards):

| test | games | allExact | meanP | calP (self-expected) | minP | maxH |
|---|---|---|---|---|---|---|
| T7 | 25 | 0.338 | 0.669 | 0.663 | 0.0008 | 233 |
| T8 | 40 | 0.510 | 0.771 | 0.772 | 0.0002 | 204 |
| T9 | 20 | 0.568 | 0.802 | 0.800 | 0.0019 | 88 |
| T11 | 20 | 0.390 | 0.720 | 0.722 | 0.0044 | 122 |

`meanP ≈ calP` is the calibration signature. If the belief is honest, the truth is distributed as the posterior
says, so the probability it places on the truth averages Σp². A tracker that peeked would put about 1.0 on the
truth. The page states both numbers.

### 5.4 Proof that the tracker never read hidden information (run for every counted game; results in `bv.chk`)

(a) **Same inputs as the run.** The tracker's statistics after the entries before our seat's last action are the
state of the bot's own tracker at its last `decide()`, because `_follow_tracker` runs at every decide before the
action. They must **equal** the per-game statistics the run logged (`info_stats`): `hidden_steals`,
`hidden_discards`, `hidden_dev_draws`, `info_resets` (= counter resets) and `info_max_hypotheses`.
Prototype: **105/105 games equal** (T8 40, T7 25, T11 20, T9 20). → `chk.stats`.

(b) **Input audit.** `AuditedTracker(PublicInfoTracker)` overrides only `_init_from` (it wraps the counter
instance's `observe_delta`, `observe_steal`, `observe_discard`, `observe_discards`, `observe_monopoly`,
`observe_hand`, `observe_hand_size`, `observe_bank` and `_reset` with recorders that delegate unchanged),
`_observe_entry` (it notes the entry and pre/post snapshots, then calls super) and `_revealed_result` (it records
the call). The counter's hypotheses change only through these methods, and the displayed belief is a function of
the counter plus `pending_discards` (sizes). Every recorded call must satisfy the public-information rule for its
entry (actor a, victim v, our seat me):

| call | allowed when |
|---|---|
| `observe_hand(i, hand)` | `i == me` |
| `observe_hand_size(j, s)` | `s` = j's true size after the entry (sizes are public) |
| `observe_bank(bank)` | `bank` = the bank before or after the entry (public) |
| `observe_delta(j, d)` | the entry kind is not `MOVE_ROBBER`, a discard or `PLAY_MONOPOLY`, and `d` = j's true delta (production, build cost, bank trade, Year of Plenty: public content) |
| `observe_steal(v, a, res)` | `res is None` unless `me ∈ {a, v}` |
| `observe_discard(i, counts)` | `i == me` (discards_public is false) |
| `observe_discards(pending, bank)`, `observe_monopoly(...)` | always (per-seat counts and amounts are public) |
| `_revealed_result(entry, …)` | entry is `MOVE_ROBBER` and `me ∈ {a, v}` |
| `_reset(...)` | never |

The audited tracker's `counter.hyps` and `pending_discards` must equal the plain tracker's at every step, which
shows the wrappers change nothing. Prototype: 22,991 calls in 12 T8 games and 15,057 calls in 8 T11 games,
**0 violations**, belief identical 20/20. → `chk.audit = [calls, violations]`, `chk.same`.

(c) **Truth in support and calibration**: the per-step asserts of 5.1, plus `meanP` against `calP` (5.3).

(d) **Twin with `reveal_hidden=True`.** It receives the same entries and treats steals, discards and dev draws as
public. It must be exact and equal to the true hands at every step. This shows the bookkeeping is right and that
the only difference between the two trackers is the hidden events. Prototype: **20/20** games. → `chk.twin`.

(e) **Structure.** The truth for display is read from the exporter's own replay (`game`), never from a tracker.
The tracker receives `final.state` only through `replay_log`, which restarts from `initial_state_like` and applies
the log entries one at a time (the same function the bot used).

The exporter fails the shard if any check fails. The page shows `chk` as one line ("Re-derived tracker matches
the run's own tracker statistics; 1,916 inputs audited, 0 hidden").

## 6. Exporter (`scripts/export_replay_archive.py`)

```
python3 scripts/export_replay_archive.py [--tests T1,...,M3] [--out $S/replay_archive] [--work $S/replay_archive_work]
        [--workers 2] [--shard-size 100] [--truth-per-test 4] [--py33 /home/user/venv_cat33/bin/python]
        [--py32 python3] [--validate]
# internal: <interpreter> scripts/export_replay_archive.py --worker T8 --shard 1 --out ... --work ...
```

* **Orchestrator** (stdlib only, any interpreter). For every test it finds the log directory
  (`proof/<T>`, `bench_1v1/<T>`, `bench_multi/<T>`) and reads the first record's `catanatron` field: `3.3.*` uses
  `--py33`, `3.2.*` uses `--py32`, anything else is an error. It runs one job per shard as
  `nice -n 10 <interpreter> … --worker T --shard k` with at most `--workers` (default 2, the cap on this machine)
  running at once. Then it merges `parts/*.rows.json` into `index.json.gz` (rows sorted by test order, then g),
  asserts 9,600 rows and each test's games = 0..N−1 once, copies `index.html` and `replay_core.js`, prints a
  per-test table (games, shard bytes, check totals), and with `--validate` runs `node validate_archive.js`
  (section 9). It exits 1 on any failure.
* **Worker** (catanatron imported lazily). It asserts `AD.CATANATRON_VERSION == rec["catanatron"]` for every
  record (`AD.rebuild_game` also refuses the wrong API). It reads the test's results files (metadata and per-game
  results by `game`), selects games `[100k, 100k + 100)`, and for each:
  `encode_game(rec, geo, meta, result_row) -> (game_record, index_row, checks, truth_steps | None)`.
  It writes `shards/<T>_<kk>.json.gz` (gzip level 9, `separators=(",", ":")`), `parts/…rows.json`,
  `parts/…checks.json` and, for the deterministic sample (`random.Random(f"{T}-20260926").sample(range(N), 4)`
  intersected with the shard), `truth/…steps.json.gz`.
* Functions (all unit-tested): `shard_path`, `test_dir`, `interpreter_for`, `global_geometry`, `board_code`,
  `encode_value(kind, value, result, ...)`, `seat_rows(state, n)` (reads `player_state` directly: hands, VP,
  public VP, longest-road length, titles), `encode_game`, `bot_view_steps` (5.1–5.2), `p_true_hand`,
  `self_expected_p`, `AuditedTracker`, `audit_violations`, `index_row`, `TEST_LABELS`.
* `encode_game` replays with `check=True` and asserts that the final VPs, winner, `turns` and
  `AD.state_fingerprint` equal `rec["final"]`, as `replay_catanatron.check_record` does.
* Runtime (prototype, under load): the replay pass takes 0.01–0.02 s per game, encoding 0.01–0.03 s for full
  games and 0.06–0.09 s for counted games with one tracker. The audit prototype (2 replays + 3 trackers) took
  0.14–0.16 s per counted game. Estimate: 7,400 × 0.04 s + 2,200 × 0.2 s ≈ 12 CPU-minutes, about 6–8 minutes of
  wall time with 2 workers.

## 7. `replay_core.js` (the one decoder, used by the page and by node)

```js
(function (root) {
  const RC = {
    VERSION: 1,
    COLORS: ["RED", "BLUE", "ORANGE", "WHITE"],
    RES: ["wood", "brick", "sheep", "wheat", "ore"],
    DEV: ["Knight", "Victory Point", "Road Building", "Year of Plenty", "Monopoly"],
    KINDS: [/* the 14 codes of 3.4 */],
    OPP: {c: "our bot", v: "ValueFunction", a: "AlphaBeta", s: "SameTurnAlphaBeta",
          f: "ValueFunction stand-in", b: "AlphaBeta stand-in"},
    shardPath(t, g),                 // "shards/T8_01.json.gz"
    async fetchJson(url),            // fetch → bytes; if bytes[0]==0x1f && bytes[1]==0x8b → DecompressionStream("gzip"); else UTF-8 text
    parseJsonBytes(u8, gunzip),      // shared by fetchJson and node (node passes zlib.gunzipSync)
    parseIndex(doc),                 // → {geo, tests, rows: [{t,g,id,seed,lineup,ours:[..],win,won,vps,pvps,turns,rolls,acts,hvp,dev,lr,la,def,bk}]}
    parseBoard(code, geo),           // → {tiles:[{i,c,x,y,res,num}], ports:[{x,y,res,nodes}], robber0}
    decodeGame(g, geo),              // → Decoded (below)
    stepText(D, i, opts),            // English log line; opts = {counted, me}
    botView(g, D),                   // → BV (below), null for full-info games
    indexFields(D),                  // {rolls, turns, dev, lr, la, hvp, def, vps, pvps}, recomputed for the validator
  };
  if (typeof module === "object" && module.exports) module.exports = RC; else root.ReplayCore = RC;
})(typeof self !== "undefined" ? self : this);
```

* `Decoded = {n, ours, board, N, steps: [{a, kind, v, dice, rob, add: [type, seat, where], free, h, x, r}],
  snap(k)}`. `r` is the roll count, and `h` expands to per-seat 5-vectors. `snap(k)` returns
  `{P: [{hand[5], held[5], played[5], vp, pvp, lr, LR, LA}], robber, bank[5], deck, turns, rolls}` from a
  precomputed `Int16Array((N + 1) · n · 20)`. Pieces at k are rebuilt by walking `add` (as the template's
  `stateAt` does).
* `BV.at(k)` returns `[{seat, size, exact, p, E[5] (the true hand when exact), devUnknown, pVP, eHiddenVP,
  trueVPcards}]` for the opponents. Also `BV.nh(k)`, `BV.pool(k)` and `BV.summary`, from arrays filled once by
  walking `q`, `nh` and `dp`. `pVP = pool[1] / Σ pool`, `eHiddenVP = devUnknown · pVP`, and `trueVPcards` is the
  opponent's true held VP cards.
* `stepText` templates, extending `game_viewer.py`'s:
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

  In counted games, when a hidden event does not involve `me`, the text carries a tag: `steals a card from Blue ·
  hidden from our bot (it was ore)`, `buys a development card · type hidden from our bot (Knight)`,
  `discards a card · hidden from our bot (wood)`.

## 8. The page (`index.html`)

**Name and identity.** `<title>Catanbot Replay Archive</title>`. The page keeps the viewer's look: Bricolage
Grotesque (display), Atkinson Hyperlegible (body) and JetBrains Mono (data), all from Google Fonts with fallback
stacks. It keeps the sage ground and sea board and the tokens from `game_viewer_template.html`, with the dark set
under `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {…} }` and again under
`:root[data-theme="dark"]`, and `body { background: var(--ground) }`. New tokens are declared in bare `:root` and
redefined in both dark blocks:

* belief state: `--known` (sage green) / `--known-soft`, `--unsure` (ochre) / `--unsure-soft`, `--miss` (brick
  red) / `--miss-soft`
* `--hidden` (muted slate) for hidden-event tags

There are no libraries and no emoji. Wide tables sit in their own `overflow-x: auto` wrapper, and the body never
scrolls sideways at 400 px.

**Boot.** `ReplayCore.fetchJson("index.json.gz")` runs first; the page shows "Loading 9,600 games…". It then
routes on `location.hash`:

* `#T8-123` opens the game at its end.
* `#T8-123.57` opens the game at position 57.
* `#T8` opens the picker filtered to T8.
* An empty hash opens the picker.

Opening or closing a game sets `location.hash` inside try/catch, and a `hashchange` listener handles back and
forward. Shards are cached in a `Map`, keeping up to 8. A fetch error shows "Could not load shards/T8_01.json.gz
(HTTP 404). Reload the page to retry." `DecompressionStream` is absent only in browsers older than about 2023;
then the page shows "This browser cannot unpack the game data. Use a current Chrome, Edge, Firefox or Safari."
`localStorage` (wrapped in try/catch) keeps only the last filter set and the bot-view mode.

**Picker view**

* Header: an eyebrow line ("9,600 logged games · catanatron 3.3.0 and 3.2.1"), h1 "Every game our bot played",
  and one sentence saying that every game replays exactly from its action log.
* Filter bar (section 4) and summary strip. On a phone the filters collapse into `<details><summary>Filters (3
  on)</summary>`.
* Table: 50 rows per page, with First, Previous and Next controls and "Page 3 of 25 · rows 101–150 of 1,234".
  Paging is simpler and more accessible than virtual scrolling at this size.
* Columns:
  * Game (`T8-123` as a link)
  * Table (format + label)
  * Info (a `full` or `Colonist` pill)
  * Our seat(s) (colour dots + ordinal)
  * Winner (dot + "ours" / colour)
  * Final VP ("ours 10 · best opp 8")
  * Rolls
  * Dev cards bought
  * LR / LA (dots)
  * Came back from (`def` if won and > 0)
  * Bot knew (counted only: `meanP` as a percentage, plus a small bar)
  * Hidden-VP win (tag)
* At ≤ 640 px each row becomes a two-line card: id, result pill and VP on line 1; opponents, rolls and tags on
  line 2.
* Keyboard: Enter opens the focused row. The column headers are buttons with `aria-sort`.

**Replay view** (the template layout: board and controls | scoreboard, VP chart, log; one column under 900 px)

* Top bar: "← All games" (keeps the filter and page), then the game id, test label, seed and catanatron version,
  then prev/next game within the current filter.
* Information badge under the h1:
  * counted: "Colonist information: our bot saw only what a Colonist.io player sees (dice, builds, trades, hand
    sizes; not stolen cards, other players' discards or their development cards)"
  * full: "Full information: our bot saw every hand and development card, as Catanatron's bots do in every game"
* Controls, board, scoreboard, VP race chart (dashed public-VP line) and log are the template's, fed by
  `snap(k)` and `stepText`.
* **Bot-view layer** (counted games only). A segmented control labelled "Opponent hands" with the options
  "True" / "True + what our bot knew" (the default) / "Only what our bot knew". Under each opponent's true hand in
  the scoreboard:
  * exact: a `--known` pill "our bot knew this hand exactly".
  * uncertain: five chips with the bot's expected count ("wood 1.4"). A chip gets a `--miss` outline when
    |E_r − true_r| ≥ 0.5 and a dashed border when E_r is fractional. A pill "P(true hand) 34 %" is `--known` at
    ≥ 90 %, `--unsure` at 30–90 % and `--miss` below 30 %. A note says "N cards (public)".
  * a development-card line: "2 cards our bot cannot see · each a VP with 5/22 = 23 % · expected hidden VP 0.45 ·
    actually holds 1".
  * our own row: "our own hand and development cards: always known".

  "Only what our bot knew" replaces every opponent hand with the belief chips, hides their development types, and
  draws only their public VP (the chart too). This is the table as our bot saw it.
* **Belief chart** (counted only): P(true hand) per opponent over the game, 0–100 %, as step lines in seat
  colours. It shares the X axis and cursor with the VP chart. Labels and ticks use theme tokens.
* **Summary card** (counted only): "Our bot knew every opponent hand exactly at 51 % of positions. On average it
  gave the true hand 77 % probability; its own posterior expected 77 %. It never ruled out the true hand. Hidden
  from it: 15 steals, 48 discarded cards, 1 development card." A second line shows the `chk` statement from 5.4.
* Log rows of hidden events carry a small `--hidden` tag, "hidden from our bot".
* Keys follow the template (←/→ step, Shift jumps a turn, Space plays or pauses, Home/End). `prefers-reduced-motion`
  disables the pulse. Every button has a visible focus ring.

## 9. Validation

### 9.1 Node validator (`scripts/replay_archive/validate_archive.js`)

```
node scripts/replay_archive/validate_archive.js --bundle $S/replay_archive --repo /home/user/ClaudeTesting2 \
     --truth $S/replay_archive_work/truth
```

1. **Bundle limits**: at most 255 files, total ≤ 64 MB, every file ≤ 15 MB, every shard path in the index exists.
2. **Index**: 9,600 rows; each test's games are 0..N−1 once, matching `tests[t].games`; every row's shard holds
   that game.
3. **Every game** is decoded with `ReplayCore.decodeGame` from the bundle. The decoded final `snap(N)` is compared
   with the **original log's** `final.state` and `final`, read directly with `zlib.gunzipSync` from
   `proof|bench_1v1|bench_multi/<T>/logs`, so the check is independent of the exporter. The comparison covers
   hands, development cards held and played, VP, public VP, longest-road length, titles, settlements, cities,
   roads, robber, bank, deck left, `turns`, `vps` and winner. `ReplayCore.indexFields(D)` must equal the index
   row. The prototype of this check (`proto_decode.js`) gave 261/261.
4. **Per-step sample**: for the sampled games (4 per test, 72 in all) the Python truth file holds
   `state_summary` after every action. Every position k = 0..N must match `snap(k)` and the pieces. For counted
   games the file also holds `(exact, p, E)` per opponent, `nh` and `dev_pool` from the tracker at every step.
   The expanded `BV.at(k)` must match them (p within 1e-4, E within 0.005, exact equal).
5. **Bot-view invariants, every counted game, every step**:
   * `|Σ E − size| ≤ 0.03`
   * `0 < p ≤ 1`
   * `pool` = `[14,5,2,2,2] − Σ played − our held`, recomputed in JS from the decoded public state
   * `Σ pool = deck + Σ_opponents held`
   * `chk` all ones with 0 violations
6. It prints one line per test (`T8: 400 games OK, 4 sampled games × 301 steps OK`) and exits 1 on the first 20
   mismatches (listed).

### 9.2 Python per-step truth (written by the worker, not published)

`truth/<T>_<kk>.steps.json.gz = {"<id>": {"steps": [state_summary_compact(k) for k in 0..N], "bv": [...]}}`.
The summary is `AD.state_summary(game.state)` reduced to lists (roads as `geo.edges` indices). It is independent
of `seat_rows`.

### 9.3 pytest (`tests/test_replay_archive.py`)

The tests run under either interpreter. Version-specific tests pick a log of the running generation (R1 for
3.2.1, T8 and T4 for 3.3) and skip otherwise, using `AD.API_33`.

* `test_shard_path` (`T7`, 537 gives `shards/T7_05.json.gz`) and `test_interpreter_for` (R1 gives py32, T8 gives
  py33; an unknown version raises a clear error)
* `test_board_code_roundtrip` (parsing the code back gives the record's resources, numbers and port resources)
* `test_geometry_global` (two games from different tests give an equal `global_geometry`)
* `test_encode_value_table` (every kind, including a 3.2.1 `DISCARD` list, a robber move without a victim and a
  one-card Year of Plenty)
* `test_encode_game_final_state` (2 games: a pure-Python mini-decoder of `s` gives `final.state`)
* `test_bot_view_audit` (3.3, 2 T8 games: 0 violations, twin exact, logged stats equal, `p > 0`,
  `exact ⇒ E == truth`)
* `test_p_true_hand_pending` (hand-built counter with a pending discard of 2: the hypergeometric value equals a
  brute-force enumeration)
* `test_worker_rejects_wrong_engine` (a 3.2.1 record under 3.3 raises `ValueError` naming the interpreter)
* `test_node_validator` (skipped without `node`: export 2 games into `tmp_path` and run the validator in a
  bundle-only mode; exit code 0)

### 9.4 Acceptance

* The exporter exits 0 for all 18 tests: 9,600 games, 0 replay mismatches. For all 2,200 counted games: stats
  equal, 0 audit violations, twin exact, belief identical.
* The validator exits 0.
* The bundle is within the limits.
* Manual check of the page at 400 px and at desktop width in both themes: `#T8-123` opens T8 game 123, filters and
  sort change the counts, and the bot-view toggle cycles through its three modes.

## 10. Risks and decisions

* **Host serving of `.gz`**: the magic-byte check makes both raw and pre-decompressed serving work. If `.gz` is
  refused, use the plain-JSON fallback in 3.6.
* **Explicit deltas and stored VP/titles over rule re-implementation**: this costs about 14 % of size, and in
  exchange the JS core has no game rules. Everything it derives (pieces, development cards, robber, bank, deck,
  turns) was validated on 261 games of both generations.
* **Calibration numbers are statistics, not proofs.** The proof is 5.4 (a), (b), (d) and (e). (c) is the
  readable summary for the page.
* `devbelief.ENABLED` is not recorded in the results metadata. The exporter records the value it ran with (0) and
  the page says "development-card belief: uniform over the unseen pool".

## 11. Prototype files and commands (scratch, not in the repository)

`$S/proto/`:

* `survey.py`: all logs, kinds and layouts. `python3 survey.py`, 3.1 s.
* `proto_encode.py`: layouts A and C plus the bot view.
  * 3.3: `nice -n 10 /home/user/venv_cat33/bin/python proto_encode.py <T> <n> --out out_<T>.json` for T8 40,
    T7 25, T11 20, T9 20, T4 24, M3 20, H1 24, T1 20, M1 24.
  * 3.2.1: `nice -n 10 python3 proto_encode.py R1 24` and `R2 20`.
* `measure.py`: table 3.6 and the extrapolation.
* `proto_index.py`: the index size.
* `proto_audit.py`: 5.4 (b) and (d). Run as `… proto_audit.py T8 12` and `… T11 8`.
* `proto_decode.js`: node decoding against the logs' final state (`node proto_decode.js out_<T>.json <logs dir>`
  for 11 tests, 261/261).
