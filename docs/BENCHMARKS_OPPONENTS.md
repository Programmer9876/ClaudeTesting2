# Benchmark opponents inside the catanatron engine

`catanbot/bench/catanatron_players.py` provides strong opponents that run
*inside* the [catanatron](https://github.com/bcollazo/catanatron) engine.
They import nothing from catanbot, so they can be dropped into any
`catanatron.game.Game` as a reference point that is much stronger than the
stock players of the PyPI wheel (`RandomPlayer`, `WeightedRandomPlayer`,
`VictoryPointPlayer`).  They are in the spirit of the `ValueFunctionPlayer` /
`AlphaBetaPlayer` of the Catanatron repository, which the 3.2.1 wheel does
not ship (the 3.3 engine does, and they are used as controls below).

The module works with both engines found on this machine:

* **catanatron 3.2.1** (the PyPI wheel, `pip show catanatron`): random
  discards enforced by the engine, chance results encoded in action values,
  random number placement;
* **catanatron 3.3.0** (the GitHub checkout at `/home/user/bcollazo/catanatron`,
  which was installed as an editable package for part of this session):
  `game.playable_actions`, chance results carried by an `ActionRecord`,
  per-resource `DISCARD_RESOURCE` actions chosen by the players, 2-tuple
  robber values, domestic-trade prompts, official-spiral number placement.

`scripts/catanatron_ladder.py` plays a round-robin ladder of the player types
in 4-player games with seat rotation and prints win rates and average
victory points per player type.  All numbers below come from that script.

## Players

| letter | player | what it does |
| --- | --- | --- |
| `R` | `catanatron.models.player.RandomPlayer` | uniform random legal action |
| `W` | `catanatron.players.weighted_random.WeightedRandomPlayer` | random, cities / settlements / dev cards up-weighted |
| `V` | `catanatron.players.search.VictoryPointPlayer` | 1-ply greedy on actual VP, random among ties (so effectively random except for builds) |
| `F` | `catanbot.bench.catanatron_players.ValueFunctionPlayer` | 1-ply greedy on the hand-written `value_function` |
| `A` | `catanbot.bench.catanatron_players.AlphaBetaPlayer` | turn-level depth-2 expectimax on the same value function, node budget 2000 |
| `X` | `catanatron.players.value.ValueFunctionPlayer` (3.3 only) | catanatron's own value-function player, default weights |
| `Y` | `catanatron.players.minimax.AlphaBetaPlayer` (3.3 only) | catanatron's own alpha-beta, depth 2, pruned action list |

### `value_function(game, color) -> float`

Evaluation from `color`'s perspective, computed from `state.player_state`
and the board data structures with per-node production tables cached per
map (about 25 us per call):

* actual victory points (12 per VP, plus/minus 1000 at the winning threshold),
* production: sum over own settlements (cities x2) of the dice probability
  of the adjacent tiles, weighted per resource (ore / wheat highest), the
  tile under the robber excluded (so a robber on an own tile is a loss),
* resource diversity (distinct resources produced),
* settlement spots buildable now and spots reachable with one more road,
  plus the production of the best reachable spot,
* longest-road length, a bonus when one road away from the award, knights
  played, a bonus when one knight away from Largest Army (the awards
  themselves are in the VP term),
* non-VP development cards in hand,
* cards in hand (0.8 each) with a penalty above the discard limit,
* progress toward the next build (city / settlement / dev card), squared so
  nearly complete builds are protected (this is what stops the player from
  spending the ore and wheat of a city on a development card),
* 2:1 ports weighted by own production of the port's resource, 3:1 ports,
* minus half of the best opponent's score (VP, production, cards, roads,
  knights), which makes the robber go to the leader's best tile.

### `ValueFunctionPlayer`

1-ply greedy: every playable action is applied to a copy of the game and the
result evaluated; END_TURN is evaluated as the current state.  Chance
actions are averaged exactly: a development-card purchase over the card
types still in the deck with their starting probabilities (the deck's order
is never looked at), a robber steal over the composition of the victim's
hand.  Actions that only pay off with a follow-up get a one-step look-ahead:
maritime trades, Year of Plenty and Monopoly are valued together with the
best city / settlement / dev card they enable, a knight together with the
best robber move, Road Building with the two best free roads.

Fixes of the obvious errors of the stock `VictoryPointPlayer`:

* never ends the turn while a settlement or city is affordable and placeable,
* a city goes on the best producing settlement (robber-aware, resource-weighted),
* discards keep the cards of the next build (`plan_discard`; chosen card by
  card on 3.3, and through `play_game` on 3.2.1, see the engine notes),
* the robber goes to the leader's best tile and never onto an own tile
  (`robber_candidates`), the victim is the leader when tiles tie,
* knights are played before rolling when the robber blocks own production or
  when the knight takes Largest Army, otherwise they compete with the other
  actions after the roll,
* development cards are bought with surplus ore / sheep / wheat (the
  progress term keeps a nearly affordable city intact; ending the turn with
  an affordable dev card while holding ore and wheat for a city is therefore
  intended, it happens once or twice per game),
* maritime trades (4:1 / 3:1 / 2:1) are used when they complete a build, and
  to get under the discard limit,
* initial placements from an opening book: settlement spot maximising
  production x diversity x scarcity (board-level scarcity of each resource,
  complementing the first settlement, ports, expansion room), road toward
  the best free spot two edges away.

### `AlphaBetaPlayer`

At its own post-roll decisions it searches sequences of its own actions for
the rest of the turn (up to `beam=8` candidates per node ordered by a cheap
heuristic - cities and settlements by production, dev card, knight, dev
plays that complete a build, the two best roads including the one that
lengthens the longest road, the trades that complete a build plus the best
port trade - and up to `depth=3` actions before END_TURN).  Every END_TURN
leaf is valued by the next opponent's reply: a chance node over the three
most likely dice sums with their true probabilities (7 with our discard and
the robber on our best tile, 6, 8; the remaining mass is a "null roll")
followed by a MIN node over the opponent's most damaging replies (city, best
settlement, knight + robber, a longest-road road), with alpha pruning of
the opponent node.  The search deepens iteratively (1, 2, 3 own actions) and
keeps the last iteration that completed within the node budget (`budget=2000`
game copies per decision, about 0.3 s worst case; most decisions use far
fewer, a game with one `A` player takes 2-3 s).  Robber moves, pre-roll
knights, Road Building roads, discards and the initial placements use the
1-ply logic of `ValueFunctionPlayer`.

## Engine notes

* **3.2.1 discards are random in the engine.** `discard_possibilities` only
  exposes `Action(color, DISCARD, None)` and `Game.play` validates actions,
  so no player can choose its discard through `decide`.  The players
  implement `choose_discard(game)` (keep the next build's cards) and
  `play_game(game, smart_discard=True)` is a drop-in for `Game.play` that
  applies it with `validate_action=False`.  The ladder runs every game
  through `play_game` (`--stock-discard` uses the stock loop instead, as a
  control; the difference is about two points of win rate, see below).  On
  3.3 the engine asks for one `DISCARD_RESOURCE` at a time and `decide`
  answers with the least valuable card of the same plan, so `play_game` is
  just `Game.play` there.
* All hands and the development deck are visible in `state`.  The players
  use the hand composition of a robbed player for the exact steal
  expectation (the stock `VictoryPointPlayer` copies the same state) and
  never look at the order of the development deck.
* Chance outcomes are always specified when simulating (`execute(game,
  action, result)`: dice, drawn card, stolen resource), so the search never
  consumes the game's random stream (3.3 shares one `random.Random` between
  a game and its copies).
* Reproducibility: catanatron generates some actions from `set`s (the
  robber victims as a set of `Color`, maritime-trade and Year-of-Plenty
  offers as sets of resource tuples), so the order of `playable_actions`
  depends on the per-process string hash seed; that order fixes the choice
  of the random players and every exact tie-break of ours.  The ladder
  therefore pins `PYTHONHASHSEED=0` (it re-executes itself unless the
  variable is already set; the worker processes inherit it) and a seed
  replays the identical game in every run (`hashseed=` in the header line;
  checked by a test).  A plain `Game(...).play()` in a fresh interpreter
  without `PYTHONHASHSEED` still gives a statistically equivalent, not
  identical, game.

## Ladder

```bash
PYTHONPATH=. python3 scripts/catanatron_ladder.py --games 12                 # R,W,V,F,A round robin
PYTHONPATH=. python3 scripts/catanatron_ladder.py --games 24 --types F,V,V,V # one 4-player matchup
PYTHONPATH=. python3 scripts/catanatron_ladder.py --games 12 --stock-discard --json out.jsonl  # control: stock loop
# catanatron 3.3 controls (X, Y) from the GitHub checkout without changing the install:
PYTHONPATH=/home/user/bcollazo/catanatron/catanatron:. python3 scripts/catanatron_ladder.py --games 24 --types X,Y,F,A
# the same players as opponents of the catanbot search bot (presets vf / ab of the adapter bench, both versions):
PYTHONPATH=. python3 scripts/bench_catanatron.py --ladder standins --games 40 --workers 2
```

Every 4-subset of `--types` plays `--games` games; the seat order rotates
with the game index and the engine additionally shuffles the seating from
the game seed.  Games run in `--workers` processes (default 2; 3 were used
below on 4 shared cores, every batch under 70 s).  The catanbot players
choose their discards (`play_game`) unless `--stock-discard` is given, and
the hash seed is pinned, so every batch below replays exactly with the
listed seed.  The win rate is
wins / games played by that type; a type that is one of four players has a
25 % par.  With 24 games one player's win rate has a standard error of
about 9 points, with 48 games about 7 points.

## Results on catanatron 3.2.1 (installed wheel)

Round robin `--games 12 --seed 0` (60 games, every type in 48 games), the
catanbot players choosing their discards (the ladder default):

| type | player | games | wins | win % | avg VP |
| --- | --- | ---: | ---: | ---: | ---: |
| A | AlphaBetaPlayer | 48 | 32 | 66.7 | 8.88 |
| F | ValueFunctionPlayer | 48 | 28 | 58.3 | 8.12 |
| V | VictoryPointPlayer | 48 | 0 | 0.0 | 2.67 |
| W | WeightedRandomPlayer | 48 | 0 | 0.0 | 2.58 |
| R | RandomPlayer | 48 | 0 | 0.0 | 2.42 |

No game hit the turn limit; 73 turns per game on average; 3.2 s per game
with three workers on shared cores.  The control through the stock engine
loop (`--stock-discard`, same seeds, random discards for everyone): A 68.8 %
(33 wins, 8.79 VP), F 56.2 % (27 wins, 8.19), R / W / V 0 %, 75 turns per
game.  The smart discards move the two catanbot players by about two points
in opposite directions (well within the 7-point standard error), so the
random discards of the stock loop cost them little against these opponents.

Direct matchups, 24 games each (one tested player, three of the control):

| matchup | seed | tested player | control |
| --- | ---: | --- | --- |
| F vs V,V,V | 1000 | F 100 % (24/24, 10.12 VP) | V 0 % (2.97 VP) |
| A vs V,V,V | 1200 | A 100 % (24/24, 10.12 VP) | V 0 % (2.67 VP) |
| A vs F,F,F | 2000 | A 25.0 % (6/24, 7.29 VP) | F 25.0 % (7.40 VP) |
| F vs A,A,A | 3000 | F 25.0 % (6/24, 7.54 VP) | A 25.0 % (7.31 VP) |

Both A-vs-F matchups land exactly on the 25 % par.  An earlier paired
experiment with the scratch harness (36 seeds x all four seats for the
tested player = 144 games, so seat and board luck cancel; stock loop,
unpinned hash seed) gave the AlphaBetaPlayer 27.8 % against three
ValueFunctionPlayers; the same harness gives exactly 25.0 % for a
ValueFunctionPlayer against three copies of itself.

## Results on catanatron 3.3.0 (GitHub checkout, players choose discards)

Run with `PYTHONPATH=/home/user/bcollazo/catanatron/catanatron:.`; `X` and
`Y` are catanatron's own players ("base catanatron" controls).

| batch | seed | result |
| --- | ---: | --- |
| X,Y,F,A, 24 games | 5000 | A 41.7 % (8.17 VP), F 33.3 % (7.83), Y 16.7 % (6.71), X 8.3 % (6.75) |
| F vs X,X,X, 24 games | 5100 | F 50.0 % (8.46 VP), X 16.7 % (6.74) |
| A vs X,X,X, 24 games | 5200 | A 58.3 % (8.46 VP), X 13.9 % (6.14) |
| X vs F,F,F, 24 games | 5300 | X 12.5 % (5.96 VP), F 29.2 % (7.57) |
| R,W,V,F,A round robin, 12/subset | 5400 | A 68.8 % (9.04 VP), F 56.2 % (8.50), R / W / V 0 % |

(Pinned hash seed; an earlier run of the same seeds with an unpinned hash
seed gave F 41.7 % / A 33.3 % in the 4-way and F 45.8 % against three X,
i.e. the same picture within noise.)

Catanatron's `AlphaBetaPlayer` (depth 2, pruned) needs about 4 s per game
in a 4-player game, its `ValueFunctionPlayer` under 1 s.

## Assessment

* `ValueFunctionPlayer` clearly beats `VictoryPointPlayer` and
  `WeightedRandomPlayer`: 24/24 against three `VictoryPointPlayer`s, and the
  three stock players won none of the 180 round-robin games on the two
  engines (three batches).  It also beats catanatron 3.3's own
  `ValueFunctionPlayer` (50.0 % against three of them; that player gets
  12.5 % against three of ours) and its `AlphaBetaPlayer` (33.3 vs 16.7 %
  in the 4-way).
* `AlphaBetaPlayer` is at least as strong as `ValueFunctionPlayer` but not
  clearly stronger: both direct 24-game matchups (`A` against three `F`, `F`
  against three `A`) land exactly on the 25 % par, the paired harness gave
  it 27.8 %, and it has the higher round-robin win rate on both engines
  (66.7 vs 58.3 %, 68.8 vs 56.2 %) and the better result against
  catanatron 3.3's players (41.7 vs 33.3 % in the 4-way, 58.3 vs 50.0 %
  against three `X`).  The round-robin gap is about one standard error; the
  direct matchups say the two are equal against each other.
* Weight tuning by paired self-play (card value, build-progress weight,
  opening book on / off, 144 games each) moved win rates by at most a few
  points, within noise, so the defaults were kept.
* Limitations: the search models only the next opponent's single reply
  (no domestic trading, which no stock player initiates either), the value
  function sees full information like every catanatron player, and the
  smart discards only reach the 3.2.1 engine through `play_game` (which the
  ladder uses; a plain `Game.play` still discards at random for them).

## Tests

`python3 -m pytest tests/test_catanatron_players.py -q -p no:cacheprovider`
(13 tests, about 12 s; passes on the 3.2.1 wheel and on the 3.3 checkout):
value function monotonic in VP and in production (settlement, city on the
best settlement, robber on an own tile), both players return a playable
action in every prompt of two random games (including the 3.3 per-card
discards), a full engine game with both players plus random opponents plays
only validated actions, a smoke match of `ValueFunctionPlayer` against three
`WeightedRandomPlayer`s completes with the highest VP, `AlphaBetaPlayer`
stays within its node budget and reaches at least depth 1, never ends the
turn with a buildable city and upgrades the best settlement, the robber
never lands on an own tile, discards keep the next build, the opening book
picks a top-production spot, and `play_game` with smart discards completes.
Three whole-game checks: an accumulator over two `play_game` games verifies
every decision of both players (legal, never END_TURN with a buildable
settlement / city, the robber never on an own tile when another tile is
available, every discard equal to `plan_discard`, and that discards, robber
moves and dev-card purchases actually occurred); the ladder's `play_seeded`
logs chosen (list-valued) discards by default and `play_one` records
`smart_discard`; and the ladder run twice in subprocesses without
`PYTHONHASHSEED` prints `hashseed=0` and writes identical per-game results.
