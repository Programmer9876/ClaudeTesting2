# Benchmarks against catanatron

[catanatron](https://pypi.org/project/catanatron/) is an independent Python
Catan engine with its own rules implementation and a few built-in players.
`catanbot/bench/catanatron_adapter.py` lets a catanbot bot sit at a
catanatron table: it converts catanatron's state into a `GameState`, runs the
catanbot search on it and hands catanatron back one of its own
`playable_actions`.  `scripts/bench_catanatron.py` plays batches of games
against catanatron's players and prints win rates.  This gives an *external*
yardstick that does not share any code with our engine, heuristics or value
net.

Two catanatron generations are supported and tested, selected by whichever
interpreter runs the script: the **3.2.1 PyPI wheel** (system `python3`
here: only the weak stock bots) and the **3.3.0 engine of the GitHub
checkout** (`/home/user/venv_cat33/bin/python`, `pip install -e
/home/user/bcollazo/catanatron`: ships the strong `ValueFunctionPlayer`,
`AlphaBetaPlayer`, `SameTurnAlphaBetaPlayer`, `GreedyPlayoutsPlayer` and
`MCTSPlayer`).  The adapter detects the API by feature
(`catanatron_adapter.API_33`): 3.3 keeps the log as
`State.action_records` of `ActionRecord(action, result)`, the playable
actions on the `Game`, `apply_action` in its own module (and it must be
given the record to replay a logged roll / robbery exactly), discards one
`DISCARD_RESOURCE` per prompt with `State.discard_counts`, uses 2-tuple
robber values, follows the official "dev cards are not playable the turn
they are bought" rule and adds domestic-trade prompts.  See
"Running the full ladder" below for the commands on each version.

```bash
pip install catanatron                     # 3.2.1 wheel; only needed for this benchmark (pulls networkx)
python3 scripts/bench_catanatron.py --games 20 --opponent vp \
    --spec "search:depth=1,evaluator=heuristic" [--workers 2] [--seed 0] [--json out.json] [--verbose] \
    [--vps-to-win 10] [--discard-limit 7]
python3 scripts/bench_catanatron.py --list-opponents       # which presets resolve on the running catanatron
python3 -m pytest tests/test_catanatron_adapter.py tests/test_bench_script.py -q   # adapter tests (skipped without catanatron)
```

* `--opponent vp` = `VictoryPointPlayer` (greedy one-ply VP maximiser, the
  strongest player in the core `catanatron` 3.2.1 package), `weighted` =
  `WeightedRandomPlayer` (random, biased towards cities > settlements > dev
  cards), `random` = `RandomPlayer` (uniform); `vf` / `ab` are our own
  stand-in players built inside the engine (`docs/BENCHMARKS_OPPONENTS.md`,
  both versions); `value`, `alphabeta`, `sameturn`, `playouts`, `mcts` are
  catanatron's own strong players (3.3 only).  A preset the running
  catanatron does not ship fails with a one-line
  `... is not available on catanatron 3.2.1 ...` message and exit code 2
  (inside a comma list or a `--ladder` it is skipped with the same line).
* `--spec` is any `catanbot.selfplay.make_bot` spec (`search:...`,
  `heuristic`, `random`, `search:model=models/value_net.npz,...`).
* Every game seats one catanbot player against three copies of the opponent;
  game `g` puts catanbot in seat `g % 4` (colours RED, BLUE, ORANGE, WHITE by
  seat), so seats and first-move advantage rotate exactly.  Boards, ports,
  dev-deck order and dice are catanatron's, seeded per game.

## Results (2026-09-25, catanatron 3.2.1, 2 worker processes on 4 shared cores)

| catanbot spec | opponent (x3) | games | wins | win rate | avg VP catanbot | avg VP opponents | avg turns | s / game |
|---|---|---|---|---|---|---|---|---|
| `search:depth=1,evaluator=heuristic` | VictoryPointPlayer | 100 | 99 | 99% | 10.03 | 2.74 | 81.8 | 0.53 |
| `search:depth=1,evaluator=heuristic` | WeightedRandomPlayer | 100 | 100 | 100% | 10.08 | 2.85 | 81.5 | 0.40 |
| `search:depth=1,evaluator=heuristic` | RandomPlayer | 100 | 100 | 100% | 10.09 | 2.40 | 78.4 | 0.41 |
| `search:depth=2,evaluator=heuristic` | VictoryPointPlayer | 20 | 20 | 100% | 10.05 | 2.77 | 82.9 | 2.14 |
| `heuristic` (rule-based bot, no search) | VictoryPointPlayer | 100 | 97 | 97% | 10.04 | 2.94 | 86.4 | 0.16 |
| `random` (control: uniform random catanbot bot) | VictoryPointPlayer | 100 | 13 | 13% | 4.48 | 6.01 | 214.3 | 0.45 |

The first two rows are the batches requested for this benchmark (20-game
versions of them gave 20/20 and 20/20; the 100-game batches above took 27 s
and 21 s of wall time).  A random seat wins 25 % of the time.  "avg turns" is
catanatron's `num_turns` (setup turns included); no game hit catanatron's
1000-turn cap.  Adapter statistics over all 520 games: 0 errors, 0 fallbacks,
0 observe errors; 53 of ~22 000 bot decisions (0.4 % of the search rows'
decisions) had a top-ranked action without catanatron equivalent (every one
of them `PLAY_ROAD_BUILDING`, see limitation 3) and used the next-ranked
action instead.

Reading the numbers:

* catanatron's built-in players are weak.  `VictoryPointPlayer` only looks
  one action ahead and picks uniformly among all actions that do not gain a
  point immediately (bank trades, `END_TURN`, ...), so it rarely accumulates
  resources: after ~80 turns the three opponents average 2.7 VP, barely above
  the 2 VP everybody starts with.  The `random` control row shows the
  catanatron players are nevertheless better than uniform random play (13 %
  for catanbot's random bot, below the 25 % seat baseline).
* The ranking is what it should be: random (13 %) << catanatron players <
  catanbot heuristic bot (97 %) < depth-1 search (99 %) <= depth-2 search
  (100 %).  Against these opponents the benchmark is saturated, so it can
  confirm the adapter and gross bot strength but cannot separate the search
  depths or a value net from the heuristic evaluator; use self-play
  tournaments (`python -m catanbot eval`, `docs/USAGE.md`) or a stronger
  external opponent (`catanatron_experimental`'s `AlphaBetaPlayer`) for that.
* Cost: a depth-1 game is ~0.4-0.5 s (0.35 s of it in the search, ~40
  searched decisions per game; the other ~35 decisions are trivial: only one
  playable action, or every legal catanbot action maps to the same catanatron
  action).  Depth 2 is ~5x slower.  Opponents cost almost nothing.

## How the mapping works

### Board ids

| | catanatron | catanbot |
|---|---|---|
| tiles | cube coordinate `(x, y, z)`; `LandTile.id` 0..18 in template order (centre, ring 1, ring 2) | hex index 0..18, rows 3-4-5-4-3 top to bottom, left to right; axial `(q, r)` pointy-top (`board.HEX_COORDS`) |
| vertices | node id 0..53 (the water ring adds ids >= 54 that never carry buildings) | vertex id 0..53 sorted by `(y, x)` |
| edges | `(node_a, node_b)` tuples (both orientations stored in `board.roads`, `playable_actions` use `a < b`) | edge id 0..71 |
| corners | `NodeRef` NORTH, NORTHEAST, SOUTHEAST, SOUTH, SOUTHWEST, NORTHWEST | corner 0 (top) .. 5, clockwise (`board.HEX_VERTICES`) |

`derive_mapping(catan_map)` derives the bijections structurally instead of
hard-coding them:

1. **Tiles**: the 12 symmetries of the hexagonal board (6 rotations x
   mirror, `board_symmetries()`, identity first) are applied to the cube
   coordinates and projected to axial `(q, r) = (x, z)`; a symmetry is kept
   only if all 19 tiles land on 19 distinct catanbot hexes.
2. **Nodes** (`_match_nodes`): every node's *incidence set* (the tiles it
   belongs to, mapped to hexes) must equal the incidence set
   `board.VERTEX_HEXES[v]` of its vertex.  This pins down the 24 interior
   nodes (3 tiles) and the 12 coastal nodes shared by two tiles (the other
   vertex of that shared edge touches three hexes).  The 18 single-tile
   corners come in ambiguous pairs on the six corner hexes; they are resolved
   by propagation along catanatron's edges: a node adjacent to an already
   mapped node must map to a neighbour (`board.VERTEX_NEIGHBORS`) of that
   node's vertex.
3. **Edges**: `board.edge_between(node_to_vertex[a], node_to_vertex[b])`
   for every tile edge.
4. `verify_mapping` re-checks everything (tile/node/edge bijections, every
   tile's six nodes are exactly the hex's six corners, every tile edge is one
   of the hex's six sides, both orientations of every edge agree) and the
   first symmetry that passes wins.

Because the coordinate set of a hexagonal board is invariant under all 12
symmetries and the node incidence structure is too, every symmetry yields
*a* consistent isomorphism; trying the identity first keeps the natural
orientation whenever catanatron's convention matches ours - which it does in
3.2.1: `NORTH` is corner 0 and the refs run clockwise (asserted by
`test_natural_orientation_is_preferred`; `test_non_identity_symmetry_search`
covers the fallbacks).  Tile and node ids are identical for every `BASE_MAP`
game (only resources, numbers and ports are shuffled), so the mapping is
cached by structure (`mapping_for`).  The derived table:

```
              h0=t15 (0,2,-2)     h1=t16 (1,1,-2)     h2=t17 (2,0,-2)
      h3=t14 (-1,2,-1)    h4=t5 (0,1,-1)      h5=t6 (1,0,-1)      h6=t18 (2,-1,-1)
h7=t13 (-2,2,0)   h8=t4 (-1,1,0)    h9=t0 (0,0,0)     h10=t1 (1,-1,0)   h11=t7 (2,-2,0)
      h12=t12 (-2,1,1)    h13=t3 (-1,0,1)     h14=t2 (0,-1,1)     h15=t8 (1,-2,1)
              h16=t11 (-2,0,2)    h17=t10 (-1,-1,2)   h18=t9 (0,-2,2)
```

(`hN` = catanbot hex, `tM` = catanatron tile id, then the cube coordinate.)

### State (`to_catanbot_state(game, me_color)` / `state_to_catanbot(state)`)

| catanatron | catanbot |
|---|---|
| `LandTile.resource / number` (`None` = desert) | `hexes[h] = (resource, number)` (desert = `(DESERT, 0)`) |
| `board.robber_coordinate` | `robber` |
| `map.port_nodes` (`None` = 3:1) | `ports = {vertex: type}` (18 vertices) |
| `state.colors` (seating order) | `players[i]`, colour names `red/blue/orange/white` |
| `P{i}_{RES}_IN_HAND` | `resources` |
| `P{i}_{DEV}_IN_HAND` | 3.2.1: `dev_cards` (all playable, see limitation 4), `dev_cards_new = 0`.  3.3: a type with `P{i}_{DEV}_OWNED_AT_START` set (playable) goes to `dev_cards`, a type bought this turn to `dev_cards_new`; VP cards always to `dev_cards` |
| `P{i}_PLAYED_KNIGHT` | `played_knights` |
| `buildings_by_color[c][SETTLEMENT / CITY / ROAD]` | `settlements / cities / roads` |
| `P{i}_HAS_ROAD` + `LONGEST_ROAD_LENGTH`, `P{i}_HAS_ARMY` | `longest_road_owner / len`, `largest_army_owner` |
| `resource_freqdeck` | `bank` |
| `development_listdeck` | `dev_deck` counts by type |
| `current_turn_index`, `num_turns` | `current`, `turn` (`max_turns` = catanatron's 1000-turn cap) |
| `P{cur}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN` | `dev_played_this_turn` |
| `is_road_building` / `free_roads_available` | `free_roads` |
| prompt `BUILD_INITIAL_SETTLEMENT` / `BUILD_INITIAL_ROAD` | `PHASE_SETUP_SETTLEMENT` / `PHASE_SETUP_ROAD` (+ `setup_round`, `setup_last_settlement`) |
| prompt `PLAY_TURN`, not rolled / rolled | `PHASE_ROLL` / `PHASE_MAIN` (`dice` from the turn's logged `ROLL`) |
| prompt `DISCARD` | `PHASE_DISCARD`, `discard_queue` = current discarder + later seats holding > 7 cards (3.2.1, catanatron's hard-coded rule for the later discarders, limitation 12) / + later seats with `discard_counts > 0` (3.3, which also prompts each seat once per card: mid-way the hand is already reduced) |
| prompt `MOVE_ROBBER` | `PHASE_ROBBER` |
| prompt `DECIDE_TRADE` / `DECIDE_ACCEPTEES` (3.3 domestic trades) | the turn player's `PHASE_MAIN` (no catanbot phase; the player declines, limitation 13) |
| `ACTUAL_VICTORY_POINTS >= vps_to_win` | `winner`, `PHASE_GAME_OVER` |

`total_vp(i)` of the converted state equals catanatron's
`ACTUAL_VICTORY_POINTS` and `public_vp(i)` its `VICTORY_POINTS` at every
tick of a game (tested).  The Longest Road holder and length are *copied*
from catanatron, not recomputed, because the two engines count roads
differently (limitation 11).  **All hands are exact**: catanatron hands every
`Player` the complete `game.state`, so every seat is converted with
`hand_known=True` / `dev_known=True` - our "me" and the three opponents alike
(catanbot's own self-play engine is perfect-information too, so the search
takes the same code path; the card-counting / determinization layer is simply
not needed here).  `trades_this_turn` is set to the per-turn maximum so
`engine.legal_actions` never proposes player trades (catanatron has none) and
the search runs with `SearchConfig.trade_proposals = 0`.

### Actions (`catanbot_action_to_key` <-> `playable_key`)

| catanbot | catanatron `Action.value` |
|---|---|
| `(SETUP_SETTLEMENT / BUILD_SETTLEMENT, v)` | `BUILD_SETTLEMENT`, node id |
| `(SETUP_ROAD / BUILD_ROAD, e)` | `BUILD_ROAD`, `(node_a, node_b)` with `a < b` |
| `(BUILD_CITY, v)` | `BUILD_CITY`, node id |
| `(ROLL,)` | `ROLL`, `None` (the log stores the two dice) |
| `(MOVE_ROBBER, hex, victim)` | `MOVE_ROBBER`, `(cube_coordinate, Color or None, None)` on 3.2.1 (the log fills the stolen card), `(cube_coordinate, Color or None)` on 3.3 (the stolen card is the `ActionRecord.result`) |
| `(PLAY_KNIGHT, hex, victim)` | `PLAY_KNIGHT_CARD`, `None`; the `(hex, victim)` is remembered and answered at the following `MOVE_ROBBER` prompt |
| `(BUY_DEV,)` | `BUY_DEVELOPMENT_CARD`, `None` (log: the card) |
| `(PLAY_ROAD_BUILDING,)` | `PLAY_ROAD_BUILDING`, `None` |
| `(PLAY_YEAR_OF_PLENTY, r1, r2)` | `PLAY_YEAR_OF_PLENTY`, `("WOOD", "ORE")`-style pair (single-card picks exist only in catanatron) |
| `(PLAY_MONOPOLY, r)` | `PLAY_MONOPOLY`, resource string |
| `(BANK_TRADE, give, get)` | `MARITIME_TRADE`, 5-tuple: `ratio` copies of the given resource, `None` padding to four, then the asked resource, e.g. `("WHEAT","WHEAT","WHEAT",None,"BRICK")` |
| `(DISCARD, counts)` | 3.2.1: `DISCARD`, `None` (catanatron discards randomly; log: list of cards).  3.3: `DISCARD_RESOURCE`, the plan's first card; `CatanbotPlayer` queues the other cards for the engine's following one-card prompts, and merges a logged run of one player's `DISCARD_RESOURCE`s back into one `(DISCARD, counts)` observation |
| `(END_TURN,)` | `END_TURN`, `None` |
| `PROPOSE_TRADE` and the other player-trade actions, forced `(ROLL, v)` | no equivalent |

`test_action_round_trips_for_every_action_type` checks, over whole random
games, that every catanatron playable action converts to a catanbot action
that converts back to the same key, that catanbot's `legal_actions` on the
converted state contains it (except the discards and, on 3.3, the non-knight
dev cards offered before the roll, limitation 14), and that every mappable
catanbot action catanatron does *not* offer is one of the two documented
rule differences below.  The 3.3 domestic-trade actions (`OFFER_TRADE`,
`ACCEPT_TRADE`, `REJECT_TRADE`, `CONFIRM_TRADE`, `CANCEL_TRADE`) have no
catanbot equivalent and never appear in a stock game.

### `CatanbotPlayer.decide`

0. 3.3 only: a `DISCARD` prompt runs the bot once, on the full hand, over
   catanbot's `(DISCARD, counts)` options whose size is the engine's
   `discard_counts`; the first card is played and the rest are queued for the
   following one-card prompts (`stats["pending_discard"]`; if the queue no
   longer matches the hand the bot plans again).  A `DECIDE_TRADE` /
   `DECIDE_ACCEPTEES` prompt is answered with `REJECT_TRADE` /
   `CANCEL_TRADE` (`stats["trade_prompts"]`).
1. One playable action (roll-only turns, the 3.2.1 `DISCARD None` prompt): return it.
2. `MOVE_ROBBER` prompt right after our `PLAY_KNIGHT_CARD`: execute the
   remembered `(hex, victim)`; if catanatron no longer offers it, fall through.
3. Convert the state, take `engine.legal_actions`, keep the actions that map
   onto a playable catanatron action.  If they all map to the same catanatron
   action (e.g. only `ROLL`, or `MOVE_ROBBER` with one option) there is nothing
   to search.
4. Otherwise `bot.decide` (the `SearchBot` from the spec) runs on the converted
   state; its ranked `last_results` are walked and the best action with a
   catanatron equivalent is played.  Should nothing rank, the bot's own
   decision, then the highest `heuristic.action_priors` mappable action, then
   `fallback_action` (city > settlement > dev card > road > roll > robber >
   end turn) are used.  With `strict=False` (the default) any adapter
   exception also ends in `fallback_action`, so a benchmark never crashes;
   the tests run with `strict=True`.
5. Between decisions every action catanatron logged (`game.state.actions`
   on 3.2.1, `game.state.action_records` on 3.3; fully specified: dice,
   stolen card, drawn dev card, discarded cards) is replayed on a private
   *shadow* copy of catanatron's state (on 3.3 together with its
   `ActionRecord`, because `apply_action` would otherwise roll fresh dice /
   steal a fresh card from the `random.Random` the state shares with its
   copies), converting the state before each action and calling
   `bot.observe(state, action, seat)` - the same hook the self-play runner
   uses - so the opponent model and the political tracker see the whole
   game.  `PLAY_KNIGHT_CARD` + `MOVE_ROBBER` are merged into one
   `PLAY_KNIGHT` observation, delivered with the state the *card* was played
   in (`PHASE_ROLL` / `PHASE_MAIN`, knight still in hand) rather than the
   `PHASE_ROBBER` state after it, so that `SearchBot.observe` predicts the
   decision from the same legal-action set the observed
   `(PLAY_KNIGHT, hex, victim)` came from (tested).  Likewise a 3.3 run of
   one player's `DISCARD_RESOURCE` actions is one `(DISCARD, counts)`
   observation with the state before the first card, where it is a legal
   half-hand discard (tested on both versions).

## Limitations and rule differences

1. **No player trading.**  catanatron 3.2.1 has no domestic trades (3.3 has
   them, but no stock player offers one and the adapter never does), so
   `PROPOSE_TRADE` never enters the search (`trades_this_turn` = max,
   `trade_proposals = 0`) and the trade-related parts of the opponent model
   keep their priors.  Feature `trades_this_turn` reads 4/4 for a value net.
2. **Discards are random on 3.2.1.**  Its only `DISCARD` action has value
   `None`; the engine samples the cards.  Our 7-protection (dumping surplus
   before the roll) still applies, our discard *choice* does not.  On 3.3
   the bot chooses its discard (one search on the full hand, handed over
   card by card).
3. **Road Building needs wood + brick in catanatron** (`road_building_possibilities`
   checks the road cost even for the free roads), and during the free roads
   only `BUILD_ROAD` is offered (no `END_TURN`).  catanbot may rank
   `PLAY_ROAD_BUILDING` first without the resources; it is then skipped for
   the next-ranked action (about 0.4 % of searched decisions; counted as
   `unmapped_top`, reported per kind by the script).
4. **Dev cards are playable the turn they are bought** in catanatron 3.2.1
   (official rules and catanbot: next turn).  All held cards go into
   `dev_cards`; the search still assumes a card bought *during* the search is
   playable next turn only, which only makes it slightly pessimistic about
   `BUY_DEV`.  3.3 follows the official rule (`{DEV}_OWNED_AT_START`) and the
   adapter converts it exactly (`dev_cards` / `dev_cards_new`).
5. **Knight before rolling** is two catanatron decisions; the robber target is
   carried over.  If the search is re-run at the `MOVE_ROBBER` prompt before
   the roll, the converted state is `PHASE_ROBBER` with `dice = 0` and the
   simulated continuation skips the roll.
6. **Perfect information.**  Opponents' hands are exact (they are in
   catanatron), so robber-victim and monopoly decisions are easier than in
   real play where catanbot works from hand sizes and card counting.
7. **Weak opponents** (see above): win rates saturate near 100 %.
8. **Reproducibility.**  Seeds fix boards, decks, dice and our bot's RNG, but
   catanatron builds some action lists from `set`s of `Color` enums whose
   order depends on Python's per-process hash seed, so game trajectories are
   only reproducible within one process (or with `PYTHONHASHSEED` fixed).
   The tests therefore loop over seeds rather than pin one trajectory
   (confirmed: the same 30 `WeightedRandomPlayer` seeds gave 18 701, 17 042
   and 18 646 ticks in three separate processes).
9. Actions logged before our first decision (setup placements of earlier
   seats) are not replayed through `observe`; single-card Year of Plenty logs
   are skipped (no catanbot equivalent).
10. `CatanbotPlayer.reset_state()` is not called by catanatron's `Game`
    itself; `play_game` calls it, and a new `game.id` also resets the bot.
11. **Longest Road: a road ending at an opponent's building does not count
    in catanatron.**  Its `Board.longest_acyclic_path` refuses to step onto
    any enemy-owned node, so the last road of a path that *ends* at an
    opponent's settlement / city is never counted (and `Board.build_road`
    never adds that node to the road's component, although
    `buildable_edges` does allow building the road into it).  catanbot's
    `engine.longest_road_length`, an independent brute-force longest-trail
    search and the official rules count that road: the opponent's building
    only stops the path from continuing *through* the vertex.  Over 30
    `WeightedRandomPlayer` games catanatron's `P{i}_LONGEST_ROAD_LENGTH` was
    exactly one below catanbot's on 363 of 10 192 ticks on which the award
    was held (every mismatch was catanatron = catanbot - 1; the brute force
    agreed with catanbot on every tick), and on 37 ticks in 1 of the 30 games
    the unique holder under catanbot's counting was a different seat from
    catanatron's `HAS_ROAD` holder.  This is an engine rule difference, not an
    adapter bug: the adapter copies catanatron's holder and length, so
    `total_vp` / `public_vp` always equal `ACTUAL_VICTORY_POINTS` /
    `VICTORY_POINTS`.  The consequence for play is that the search, whose
    simulated road building uses catanbot's counting, can expect a Longest
    Road award (or a takeover) that catanatron will not grant when the
    decisive road ends at an opponent's building.
    `test_road_ending_at_enemy_settlement_counts_in_catanbot_but_not_catanatron`
    reproduces this on a hand-built board (catanbot's engine books +2 VP for
    the fifth road, catanatron scores the path 4 and awards nothing) and
    `test_longest_road_lengths_never_differ_by_more_than_one_through_games`
    checks the `cat in (cb, cb - 1)` invariant through whole games.
12. **`discard_limit` only governs the first discarder (3.2.1).**  On a 7
    catanatron 3.2.1 picks the first player to discard with
    `state.discard_limit` (default 7) but advances to the *later* discarders
    with a hard-coded `> 7` (`apply_action`'s `DISCARD` branch).
    `state_to_catanbot` builds the `discard_queue` with the same `> 7` rule
    for the later seats so it lists exactly the seats catanatron goes on to
    prompt; with the default limit the two rules coincide (every benchmark
    game), with a non-default `--discard-limit` they do not (e.g. limit 9,
    hands `[10, 8, 6, 9]`: catanatron prompts seats 0, 1 and 3 although only
    seats 0 and 3 exceed 9; limit 5, hands `[6, 6, 8, 3]`: only seats 0 and
    2 are prompted).  `test_discard_queue_mirrors_catanatron_hard_coded_limit`
    replays these.  3.3 applies the limit to everyone and fixes the counts at
    the roll (`State.discard_counts`), which the queue follows
    (`test_discard_queue_follows_discard_counts`).
13. **3.3 domestic-trade prompts are declined.**  catanbot's trade logic
    (`PROPOSE_TRADE` / `ACCEPT_TRADE` / ...) is not wired to catanatron
    3.3's `OFFER_TRADE` protocol: our player never offers, answers a
    `DECIDE_TRADE` prompt with `REJECT_TRADE` and a `DECIDE_ACCEPTEES`
    prompt with `CANCEL_TRADE` (counted in `stats["trade_prompts"]`), and
    the converted state of those prompts is the turn player's `PHASE_MAIN`.
    No stock catanatron player offers trades, so this never triggers in the
    ladder (`test_trade_prompts_are_declined` exercises it by hand).
14. **3.3 offers Year of Plenty, Monopoly and Road Building before the
    roll** (3.2.1 and catanbot: only the knight).  Those pre-roll options
    have no legal catanbot counterpart in `PHASE_ROLL`, so they are never
    chosen; the same card is played after the roll instead.  An opponent's
    logged pre-roll play is observed with the `PHASE_ROLL` state (not a
    legal action there; `SearchBot.observe` makes no prediction for those
    kinds, so nothing breaks).  When Road Building was played before the
    roll, 3.3 prompts the two free roads first: `state_to_catanbot` converts
    that (`is_road_building` while not rolled) to a `PHASE_MAIN` state with
    `free_roads` set and `dice = 0`, where the roads are legal.
15. **`GreedyPlayoutsPlayer` runs its playouts in-process.**  catanatron
    3.3's `playouts` preset opens a `multiprocessing.Pool(cpu_count())` per
    decision; the bench script sets its `USE_MULTIPROCESSING` flag off when
    it resolves the preset (the seeded playouts give the same result), so
    `--workers` stays the only parallelism and the preset also works inside
    the worker processes (which may not spawn children).

## Running the full ladder (strong Catanatron players)

The PyPI release of Catanatron (3.2.1) ships only the weak stock bots.  The
strong players live in the GitHub checkout (3.3.0 engine); install it into
its own interpreter (a venv, so the wheel stays available as well):

```bash
git clone https://github.com/bcollazo/catanatron.git        # here: /home/user/bcollazo/catanatron
python3 -m venv /home/user/venv_cat33 && /home/user/venv_cat33/bin/python -m pip install -e catanatron numpy pillow pytest
/home/user/venv_cat33/bin/python scripts/bench_catanatron.py --list-opponents   # value/alphabeta/... "available"
```

The same script and adapter run on both versions (`--list-opponents` says
which presets resolve on the interpreter in use; a preset that is not
available is a one-line error with exit code 2 when asked for alone, and
skipped inside a list or ladder).  Always run from the repository root with
`PYTHONPATH` set; keep to `--workers 2` on this 4-core machine.

Stand-in ladder on catanatron 3.2.1 (system `python3`; our `vf` / `ab`
players plus the stock controls):

```bash
cd /home/user/ClaudeTesting2
PYTHONPATH=/home/user/ClaudeTesting2 python3 scripts/bench_catanatron.py --list-opponents
PYTHONPATH=/home/user/ClaudeTesting2 python3 scripts/bench_catanatron.py --ladder controls --games 100 --workers 2 --seed 0 \
    --spec "search:depth=1,evaluator=heuristic" --json ladder_321_controls.json      # ~0.4-0.5 s/game: ~1.5 min
PYTHONPATH=/home/user/ClaudeTesting2 python3 scripts/bench_catanatron.py --ladder standins --games 100 --workers 2 --seed 0 \
    --spec "search:depth=1,evaluator=heuristic" --json ladder_321_standins.json      # vf ~1.3 s/game, ab ~7 s/game: ~7 min
# or both at once: --opponent random,weighted,vp,vf,ab
```

Strong ladder on catanatron 3.3.0 (the venv; catanatron's own players):

```bash
cd /home/user/ClaudeTesting2
PYTHONPATH=/home/user/ClaudeTesting2 /home/user/venv_cat33/bin/python scripts/bench_catanatron.py --list-opponents
PYTHONPATH=/home/user/ClaudeTesting2 /home/user/venv_cat33/bin/python scripts/bench_catanatron.py \
    --opponent value,alphabeta,sameturn --games 40 --workers 2 --seed 0 \
    --spec "search:depth=1,evaluator=heuristic" --json ladder_330_strong.json
PYTHONPATH=/home/user/ClaudeTesting2 /home/user/venv_cat33/bin/python scripts/bench_catanatron.py \
    --opponent mcts --games 4 --workers 2 --seed 0 --json ladder_330_mcts.json
PYTHONPATH=/home/user/ClaudeTesting2 /home/user/venv_cat33/bin/python scripts/bench_catanatron.py \
    --opponent playouts --games 2 --workers 2 --seed 0 --json ladder_330_playouts.json
# --ladder strong runs all five in this order: value, alphabeta, sameturn, playouts, mcts (mind the slow two)
```

Per-game cost of the 3.3 opponents with the depth-1 heuristic bot on this
machine (one game, one worker, `--seed 1`): `value` 1.1 s, `alphabeta` 19 s,
`sameturn` 22 s, so with two workers 20 minutes cover roughly 1000 / 120 /
110 games of those; PLAYOUTS_MCTS_TIMING.  The stand-ins on 3.2.1 cost 1.3 s
(`vf`) and 6.8 s (`ab`) per game, the stock controls 0.4-0.5 s.

Presets: `random`, `weighted`, `vp` (stock controls), `vf`, `ab` (our
stand-in value-function / alpha-beta players built inside the Catanatron
engine), `value`, `alphabeta`, `sameturn`, `playouts`, `mcts` (Catanatron's
own strong players).  Any other opponent can be given as an import path,
e.g. `--opponent mypkg.bots:MyPlayer`, and several as a comma list.  In
4-player games the seat baseline is 25%; use at least 100 games per
opponent for a win rate of 40%+ to be statistically clear.

Smoke results of the dual-version work (2026-09-25, `--games 2 --workers 1
--seed 1`, depth-1 heuristic bot; far too few games to be a benchmark, they
only show both paths run cleanly): 3.2.1 vs `random` 2/2 wins, 0.40 s/game,
0 adapter errors / fallbacks / observe errors; 3.3.0 vs `value` 0/2 wins
(4.5 vs 6.8 VP), 1.1 s/game, 0 errors, 3 planned discard cards handed over
card by card.  One game each vs `alphabeta` (9 VP, lost 9-10) and
`sameturn` (3 VP) on 3.3.0 also ran with 0 errors.  catanatron 3.3's
players are a real opponent, unlike the 3.2.1 stock bots: a proper
strong-ladder run (40+ games per opponent) is the next step.
