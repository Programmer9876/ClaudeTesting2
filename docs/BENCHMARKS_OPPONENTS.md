# Benchmark opponents inside the catanatron engine

`catanbot/bench/catanatron_players.py` provides strong opponents that run
*inside* the [catanatron](https://github.com/bcollazo/catanatron) engine
(PyPI `catanatron==3.2.1`).  They import nothing from catanbot, so they can be
dropped into any `catanatron.game.Game` as a reference point that is much
stronger than the stock players shipped on PyPI (`RandomPlayer`,
`WeightedRandomPlayer`, `VictoryPointPlayer`).  They are in the spirit of the
`ValueFunctionPlayer` / `AlphaBetaPlayer` of the Catanatron repository,
which are not on PyPI.

`scripts/catanatron_ladder.py` plays a round-robin ladder of all five player
types in 4-player games with seat rotation and prints win rates and average
victory points per player type.  The numbers below come from that script.

## Players

| letter | player | what it does |
| --- | --- | --- |
| `R` | `catanatron.models.player.RandomPlayer` | uniform random legal action |
| `W` | `catanatron.players.weighted_random.WeightedRandomPlayer` | random, cities / settlements / dev cards up-weighted |
| `V` | `catanatron.players.search.VictoryPointPlayer` | 1-ply greedy on actual VP, random among ties (so effectively random except for builds) |
| `F` | `catanbot.bench.catanatron_players.ValueFunctionPlayer` | 1-ply greedy on the hand-written `value_function` |
| `A` | `catanbot.bench.catanatron_players.AlphaBetaPlayer` | turn-level depth-2 expectimax on the same value function, node budget 2000 |

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
* discards keep the cards of the next build (`plan_discard`; see the engine note below),
* the robber goes to the leader's best tile and never onto an own tile
  (`robber_candidates`), the victim is the leader when tiles tie,
* knights are played before rolling when the robber blocks own production or
  when the knight takes Largest Army, otherwise they compete with the other
  actions after the roll,
* development cards are bought with surplus ore / sheep / wheat (the
  progress term keeps a nearly affordable city intact),
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
plays that complete a build, the two best roads, the trades that complete a
build - and up to `depth=3` actions before END_TURN).  Every END_TURN leaf is
valued by the next opponent's reply: a chance node over the three most
likely dice sums with their true probabilities (7 with our discard and the
robber on our best tile, 6, 8; the remaining mass is a "null roll") followed
by a MIN node over the opponent's most damaging replies (city, best
settlement, knight + robber, a longest-road road), with alpha pruning of
the opponent node.  The search deepens iteratively (1, 2, 3 own actions) and
keeps the last iteration that completed within the node budget (`budget=2000`
game copies per decision, about 0.3 s worst case; most decisions use far
fewer).  Robber moves, pre-roll knights, Road Building roads and the initial
placements use the 1-ply logic of `ValueFunctionPlayer`.

## Engine notes (catanatron 3.2.1)

* **Discards are random in the engine.** `discard_possibilities` only
  exposes `Action(color, DISCARD, None)` and `Game.play` validates actions,
  so no player can choose its discard through `decide`.  The players
  implement `choose_discard(game)` (keep the next build's cards) and
  `play_game(game, smart_discard=True)` is a drop-in for `Game.play` that
  applies it with `validate_action=False`.  The ladder uses the stock loop
  unless `--smart-discard` is given, so all numbers below are pure engine.
* All hands and the development deck are visible in `state`.  The players
  use the hand composition of a robbed player for the exact steal
  expectation (the stock `VictoryPointPlayer` copies the same state) and
  never look at the order of the development deck.
* `Game.execute(action, validate_action=False)` accepts fully specified
  chance actions (`ROLL` with dice, `BUY_DEVELOPMENT_CARD` with the card,
  `MOVE_ROBBER` with the stolen resource, `DISCARD` with the cards), which is
  how the search averages over outcomes without consuming the game's random
  stream.

## Ladder

```bash
PYTHONPATH=. python3 scripts/catanatron_ladder.py --games 12                 # R,W,V,F,A round robin
PYTHONPATH=. python3 scripts/catanatron_ladder.py --games 24 --types F,V,V,V # one 4-player matchup
PYTHONPATH=. python3 scripts/catanatron_ladder.py --games 12 --smart-discard --json out.jsonl
```

Every 4-subset of `--types` plays `--games` games; the seat order rotates
with the game index and the engine additionally shuffles the seating from
the game seed.  Games run in `--workers` processes (default 2).  The player
win rate is wins / games played by that type; a type that is one of four
players has a 25 % par.

<<RESULTS>>
