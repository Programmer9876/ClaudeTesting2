# catanbot design & module contracts

This document is the contract every module is written against.  Read it fully
before touching code.  `catanbot/board.py`, `catanbot/state.py` and
`catanbot/actions.py` are already written and are the source of truth for
topology, the state model and the action encoding.

## 1. What the bot must do (user requirements)

1. **Play well** (target: beats heuristic bots and random bots decisively in
   4-player self-play, sensible recommendations on real positions).
2. **Trading**
   * Bank / port trades when we can afford them and no cheaper player trade
     is realistically available (a 1:1 player trade beats 4:1; a 2:1 port is
     usually preferable to feeding an opponent).
   * Player trades when the bank/port is out of a resource or we cannot make
     the bank rate; never feed the leader a resource that completes their
     build; evaluate every trade with the same value function used by search.
   * Accept / reject incoming offers with the same value function.
3. **7-protection**: when the hand is > 7 cards and opponents still roll
   before us, dump surplus (build, buy dev, bank trade) so a 7 costs less;
   discard choice = keep the cards needed for the planned build.
4. **Knights / robber**: play a knight when the robber blocks a critical tile,
   to stop the leader (block their best hex, steal from them), or to take /
   defend Largest Army.  Knights may be played before rolling.
5. **Dev cards**: buy when resources would otherwise rot, when going for
   Largest Army, when at 8-9 VP the hidden-VP gamble can win, and when no
   settlement spot is available; Monopoly when opponents collectively hold
   many of a resource (card counting); Year of Plenty to complete a build.
6. **Card counting**: track bank stock per resource, remaining dev deck
   composition, and opponents' hands (exact where public, probabilistic
   otherwise).
7. **Placement**: initial settlements by pips, diversity, scarce resources,
   ports and blocking; cities on the best producers; roads towards the best
   reachable spot and to cut opponents off / for Longest Road.
8. **Search**: Stockfish-style expectimax over dice rolls with an ML value
   net at the leaves, beam-pruned action sequences within a turn, opponents
   modelled with the same policy (max^n), plus a heuristic evaluator used as
   the bootstrap / fallback and for move ordering.
9. **Screenshot analysis**: parse a Colonist.io screenshot into a
   `GameState` (board, numbers, robber, pieces, ports, my hand, opponents'
   card counts, VP, knights, awards) and recommend the best action(s) with an
   explanation.  Two parsers: a pure computer-vision one (OpenCV / numpy) and
   an LLM-vision one (Claude, needs `ANTHROPIC_API_KEY`); plus a JSON state
   input for manual entry / corrections.

## 2. Package layout

```
catanbot/
  board.py        static topology (DONE)
  state.py        GameState / Player / TradeOffer + JSON (DONE)
  actions.py      action tuples + describe() (DONE)
  engine.py       rules: legal_actions, apply, roll outcomes, awards
  heuristic.py    hand-written evaluation + move ordering priors
  placement.py    settlement / city / road scoring helpers
  trading.py      bank vs player trade logic, offer generation, accept/reject
  discard.py      7-protection: surplus dumping and discard choice
  robber.py       knight timing, robber target + victim selection, out-of-turn steal exposure
  danger.py       distance-to-win model per player (win path, missing cards, rolls, turns)
  devcards.py     dev card buying / playing policy, monopoly & YOP timing
  counting.py     card counting: bank, dev deck, opponent hand beliefs
  inference.py    determinization of hidden information (sample full states)
  features.py     GameState -> numpy feature vector (perspective of a player)
  model.py        numpy MLP value network (train / predict / save / load)
  search.py       expectimax + beam search; returns ranked actions + explanation
  agents/
    base.py       Bot interface
    random_bot.py
    heuristic_bot.py
    search_bot.py ML + search bot (the real bot)
  selfplay.py     game runner, tournaments, dataset generation (multiprocessing)
  train.py        training loop (self-play -> fit -> evaluate -> promote)
  vision/
    schema.py     screenshot-parse result -> GameState, validation, standard priors
    synth.py      synthetic Colonist.io-style renderer (test data + classifier training)
    digits.py     number-token / digit classifier (numpy) trained on synthetic renders
    colonist.py   CV parser: image -> ParsedBoard/GameState
    llm.py        Claude vision parser (optional dependency `anthropic`)
  cli.py          `python -m catanbot ...`
  __main__.py
models/           trained weights (value net .npz, digit classifier .npz)
tests/            pytest
docs/             this file + user docs
```

Only `numpy` and `pillow` are hard dependencies.  `cv2` and `anthropic` are
imported lazily inside `vision/`.  Everything must run on CPU.

## 3. Engine contract (`catanbot/engine.py`)

The engine is perfect-information and deterministic given a `random.Random`.
It **never mutates its input state** in `apply` (use `state.copy()`), but a
faster `apply_inplace(state, action, rng)` is also exposed and used by the
self-play runner.

```python
acting_player(state) -> int
    # PHASE_DISCARD: state.discard_queue[0]; PHASE_TRADE_RESPONSE: state.trade_responder;
    # otherwise state.current.
legal_actions(state) -> list[Action]       # complete & only legal actions for acting_player
apply(state, action, rng) -> GameState     # copy + apply_inplace
apply_inplace(state, action, rng) -> GameState
is_terminal(state) -> bool
roll_outcomes(state) -> list[tuple[float, int]]   # [(prob, roll_value)] for 2..12
apply_roll(state, value, rng) -> GameState        # apply (ROLL, value) on a copy
production_for_roll(state, value) -> list[list[int]]   # per player resource gains (bank-limited)
longest_road_length(state, player) -> int
count_vp(state, player, include_hidden=True) -> int
```

Rules (base game, 3-4 players, Colonist.io defaults):

* Setup: snake order (0..n-1 then n-1..0).  Second settlement pays the
  resources of its adjacent hexes.  `SETUP_ROAD` must touch the settlement
  just placed (`state.setup_last_settlement`).  After setup, `phase = ROLL`,
  `current = 0`, `turn` counts completed turns.
* `ROLL`: `(ROLL,)` draws 2d6 from `rng`; `(ROLL, v)` forces the total `v`
  (used by search).  Legal in `PHASE_ROLL` only.  Knights (and only knights)
  may be played before rolling.
* Production: a hex with the robber pays nothing.  Settlement 1, city 2.  If
  the bank cannot pay everyone for a resource: if only one player is owed
  that resource, they get what is left; otherwise nobody gets it.
* 7: every player with > 7 cards discards `floor(n/2)` (queue in seat order
  starting from the roller), then the roller moves the robber (must change
  hex) and steals one random card from a chosen adjacent opponent with ≥ 1
  card (`victim = -1` only if no such opponent exists).
* Build costs and piece limits: `board.COST_*`, `MAX_ROADS=15`,
  `MAX_SETTLEMENTS=5`, `MAX_CITIES=4`.  Settlements need distance ≥ 2 from
  every other building and a connecting own road (except setup).  Roads need
  an own road / building at an endpoint and the endpoint must not be an
  opponent building (opponent buildings cut roads).  Cities replace own
  settlements (the settlement piece returns to supply).
* Dev cards: `BUY_DEV` draws from `state.dev_deck` weighted by remaining
  counts; the card lands in `dev_cards_new` and moves to `dev_cards` at end
  of turn.  At most one dev card per turn (`dev_played_this_turn`).  VP cards
  are never played; they count in `total_vp`.  Knight: move robber + steal,
  `played_knights += 1`, Largest Army to the first player with ≥ 3 knights,
  taken over only by strictly more.  Road Building: `free_roads =
  min(2, roads_left)`; `BUILD_ROAD` consumes free roads first (no cost); if
  no legal road exists the free roads are forfeited.  Year of Plenty limited
  by bank stock.  Monopoly takes all cards of that resource from every
  opponent.
* Longest Road: ≥ 5 roads, longest simple path (roads broken by opponent
  buildings on a vertex), first to reach keeps it on ties; if the holder
  drops below the new max it passes to the unique new max, else nobody.
* Bank trade `(BANK_TRADE, give, get)`: ratio = `state.port_ratio(cur, give)`,
  needs bank stock of `get` ≥ 1 and the player has `ratio` of `give`.
* Player trades: `(PROPOSE_TRADE, give, get)` only in `PHASE_MAIN` after
  rolling, at most `MAX_TRADE_PROPOSALS_PER_TURN = 4` per turn; `give` and
  `get` are 5-tuples, non-empty, disjoint, proposer must hold `give`.
  `legal_actions` in `PHASE_MAIN` generates a bounded candidate set: all
  1-for-1 and 2-for-1 offers (give ≤ 2 of one type, get 1 of another) that
  the proposer can afford and at least one opponent could pay.  Responders
  are asked in seat order after the proposer (`trade_responder`), skipping
  players who cannot pay `get` (auto-reject).  When all responded: if any
  accepted -> `PHASE_TRADE_SELECT` where the proposer plays `(EXECUTE_TRADE,
  partner)` or `(CANCEL_TRADE,)`; else back to `PHASE_MAIN`.
* `END_TURN`: moves `dev_cards_new` into `dev_cards`, resets per-turn flags,
  `turn += 1`, next player in `PHASE_ROLL`.  The game ends (`phase =
  GAME_OVER`, `winner`) as soon as the current player reaches ≥ 10 VP
  (including hidden VP cards) during their own turn, or when `turn >=
  max_turns` (winner = highest VP, ties -> -1).
* `Player.hand_size` and `Player.dev_count` are kept equal to the true totals
  by the engine whenever it changes a hand (the engine may just recompute
  them in `apply_inplace`).

## 4. Search contract (`catanbot/search.py`)

```python
@dataclass
class SearchConfig:
    depth: int = 2                  # number of *turns* to look ahead (own turn counts as 1)
    beam: int = 6                   # partial action sequences kept per level
    expand: int = 10                # actions tried per decision node
    roll_samples: int = 11          # 11 = exact expectation over all rolls; fewer = top-probability rolls
    opp_roll_samples: int = 12      # sampled dice sequences for the opponents' turns (common random numbers)
    opponent_actions: int = 4       # greedy actions per simulated opponent turn (opponent_expand candidates each)
    finished_lookahead: int = 0     # end-of-turn nodes that get the future value: 0 = all, N = the top N by static
    lookahead_shrink: float = 12.0  # k: a node's own sampled lookahead delta counts n / (n + k); 0 = raw
    max_nodes: int = 40000
    time_limit: float | None = None
    native_future: bool = True      # opponents' turns simulated by the C++ extension when built (docs/CPP.md)
    # + trade_proposals / trade_cap_early / trade_cap_late / opponent_proposals / dump_candidates /
    #   discard_candidates / use_opponent_model (sections 6, 11)

@dataclass
class ScoredAction:
    action: Action
    value: float              # expected value for the acting player (win probability scale, 0..1)
    explanation: str          # short reason ("+2.1 pips/turn", "completes city", ...)
    line: list[Action]        # principal variation for the current turn

class Searcher:
    def __init__(self, evaluator, config: SearchConfig, heuristics=None): ...
    def search(self, state, player: int | None = None, rng=None) -> list[ScoredAction]
        # ranked best-first; [0] is the recommendation.  ``player`` defaults to acting_player.
    def best_action(self, state, rng=None) -> Action
```

`evaluator` is any object with `evaluate(states: list[GameState], players: list[int]) -> np.ndarray`
returning one value per (state, player) pair in [0, 1].  Both `heuristic.HeuristicEvaluator`
and `model.ValueNet` implement this.  Leaves must be evaluated in **batches**.

Turn model: within a turn a sequence of actions ends with `END_TURN`.  The
searcher enumerates action sequences with beam pruning (ordering by
`heuristic.action_priors`), inserts chance nodes for `ROLL` (exact expectation
over the 11 outcomes weighted by `board.ROLL_PROB`, or the top-k), models
robber steals as expectation over the victim's hand, models `PROPOSE_TRADE` by
asking the opponent model whether the responder would accept, and evaluates
leaves with the evaluator from each player's own perspective (max^n).
Depth is in turns.  Discard decisions for opponents use `discard.choose_discard`.

**Lookahead horizon (depth >= 2).**  At depth 1 every leaf is a static value.
At depth 2 an end-of-turn node's *future value* is the mean, over
`opp_roll_samples` dice sequences shared by every node (common random
numbers), of the leaf value after the opponents played their turns greedily
(one-step lookahead with the same evaluator, discards / robber / knights from
the strategy modules); at depth 3 the leaf value is a reduced recursive
search instead.  Three rules keep the horizons consistent
(`Searcher.search` / `apply_lookahead`, mirrored natively):

1. **Every** finished node gets the future value (`finished_lookahead = 0`).
   Valuing only the top-N nodes by static value and giving the rest
   `static + shift` (the pre-fix default, still available with
   `finished_lookahead = N`) values two candidates with identical static
   values differently by membership alone, and the selection by static plus
   the winner's curse leave a residual offset between the two classes that
   the mean shift cannot remove.
2. The **mean shift** `mean(future - static)` over the non-terminal
   lookahead nodes is added to every leaf that has no future value (branches
   pruned from the beam), so lines with and without lookahead are compared at
   the same horizon.  A finished game keeps its exact value and stays out of
   the mean.  The backup is **not clamped**: `static + shift` may leave
   [0, 1] (the shift is about -0.06 with the heuristic evaluator), and
   clamping it collapsed the ordering of every leaf below 0 in 7.5 % / 13 % /
   19 % of mid-game nodes at depth 2 / 3 / 4 (which is why deeper searches
   diverged from each other).  Only the root's `ScoredAction.value` is
   clamped, after ranking.
3. The node's own deviation from the mean is **shrunk by its reliability**:
   `value = static + shift + w * (future - static - shift)` with
   `w = n / (n + lookahead_shrink)` (`lookahead_weight`; `n = opp_roll_samples`).
   Measured on mid-game nodes the sampled delta has a differential standard
   deviation of ~0.028 win-probability per sample under common random numbers
   while the depth-1 margin between the best two actions is 0.002-0.008, so
   with few samples the raw delta is mostly noise (split-half reliability
   0.40 at 12 samples).  `w` is the posterior weight of the node's own
   estimate against the common mean; `lookahead_shrink = 0` restores the raw
   future values, a huge value the depth-1 ranking (plus a constant).

4. At depth >= 3 the lookahead leaves (end nodes x roll samples) are scored
   by the reduced sub-search **for every leaf or for none**
   (`Searcher._leaves_affordable`, mirrored in `future_values` natively):
   the sub-search runs only when `REDUCED_SEARCH_MIN_NODES` (500) nodes per
   leaf are still left in `max_nodes` after the opponents' turns; otherwise
   every leaf keeps its static value and depth 3 is exactly depth 2 (same
   dice).  Before this rule the native path searched the leaves in tree order
   until the budget ran out, so the shallowest end nodes (END_TURN first)
   were valued one turn deeper than their siblings - the mixing of rule 1 one
   level down - while the Python path ignored the global budget altogether
   (137k nodes and 30 s per decision at `max_nodes = 20000`).  With the
   defaults (~30 end nodes x 12 samples) a real depth 3 needs
   `max_nodes >= ~250000` and costs seconds per decision; at the SearchBot
   budget of 20000 nodes "depth 3" was depth 2 on about half of the mid-game
   nodes and a tree-order-biased mixture on the rest, which is why the
   same-table depth-3-vs-depth-2 numbers below are a near-null comparison.

Invariants (tests/test_search.py): a lookahead that adds the same constant to
every static value never changes the depth-1 ranking; every finished node of
the tree is handed to `_future_values`; the mean of `value - static` over the
lookahead nodes equals the shift for any `w`; a depth-3 search whose budget
cannot cover its leaves values and ranks exactly like depth 2, an ample
budget searches every leaf.  Defaults changed with this
fix: `opp_roll_samples` 6 -> 12, `finished_lookahead` 4 -> 0 (all), and
`lookahead_shrink` (new) 12 (the split-half reliability of the centred delta
on 200 mid-game nodes is 0.50 at 12 vs 12 samples, i.e. a single-sample
reliability of 0.077 and `k = (1 - r) / r = 12`; with `opponent_actions = 0`
the per-sample differential noise falls from 0.028 to 0.008 but the signal
from 0.020 to 0.005, and the measured `k` is 3).  `selfplay.make_bot`'s
spec keys still default to `opprolls=4, lookahead=3`; pass
`lookahead=0,opprolls=12` for the fixed behaviour until those defaults follow.

Measured (catanatron 3.2.1 stand-ins, `scripts/bench_catanatron.py`, seat
rotation, 2 workers; `vf` = 1-ply value function, `ab` = depth-2 alpha-beta;
all bots `beam=4,expand=8,rolls=11,nodes=20000`, heuristic evaluator; win %
with 95 % Wilson intervals, then average VP):

| bot | vs vf, seeds 31+32 (120 games) | vs ab, seeds 31+32 (120) | vs vf, seeds 31-34 (720) | vs ab, seeds 31-33 (240) | s / searched decision |
| --- | --- | --- | --- | --- | --- |
| depth 1 | 22.5 (16-31), 6.97 | 26.7 (20-35), 7.20 | 24.9 (22-28), 7.27 | 28.8 (23-35), 7.34 | 0.015 |
| depth 2 before the fix (top-4 end nodes, 4 samples, raw, clamped) | 19.2 (13-27), 7.01 | 19.2 (13-27), 6.79 | 21.2 (18-25), 7.02 (420) | 17.9 (14-23), 6.85 | 0.021 |
| depth 2 after the fix (all end nodes, 12 samples, `k = 12`, unclamped) | 27.5 (20-36), 7.28 | 23.3 (17-32), 7.08 | 23.9 (21-27), 7.18 | 20.4 (16-26), 7.03 | 0.069 |
| same, `lookahead_shrink = 0` (raw deltas) | - | - | 21.7 (19-25), 7.13 (600) | 15.8 (10-23), 6.84 (120) | 0.068 |
| same, 24 samples | - | - | 23.3 (19-28), 7.30 (300) | 22.5 (16-31), 7.05 (120) | 0.125 |
| same, `opponent_actions = 0` (rolls / discards / robber only), 24 samples | 20.0 (13-30), 6.98 | 22.5 (14-34), 6.96 | 23.9 (21-27), 7.20 | 20.8 (17-25), 6.86 (360) | 0.021 |

The fix recovers the whole deficit against `vf` (parity with depth 1 within
+/-2.3 points over 720 games) and about a third of it against `ab`, where
every depth-2 variant, fixed or not, stays 6-8 points below depth 1 (the 95 %
intervals barely overlap).  The shrinkage matters (raw deltas are 2-5 points
worse on the same seeds), 24 samples add nothing measurable over 12 at twice
the cost, and simulating the opponents' builds adds nothing over rolls only
(`opponent_actions = 0`) at three times the cost - the max^n opponents are
kept as the default because they are the documented, parity-tested
semantics, not because they earn their time.  Same table, 4 players, 32
games, 2 seats each (`python3 -m catanbot eval`): depth 3 (same lookahead
defaults; each leaf's reduced sub-search still uses the old top-2 x 4-sample
lookahead, `search.reduced_config`) wins 12/64 seats = 18.8 % (11-30), 7.20
VP against depth 2's 20/64 = 31.2 % (21-43), 8.23 VP.  That run predates
rule 4: at `nodes = 20000` its "depth 3" bot was depth 2 on about half of
its decisions (the budget was gone after the simulations) and a tree-order
mixture on the rest, so the gap is mostly the noise of 32 games between two
nearly identical bots, not evidence about depth 3 (see "Adversarial re-run"
below).  The remaining lever is the
evaluator: a de-noised lookahead only reaches depth-1 strength because its
residual signal (0.01 win-probability across candidates) is what the static
evaluator already knows.

**Adversarial re-run** (new seeds 41 and 42, 60 games per cell, the same
commands and specs as the table above).  Against `vf`: depth 2 after the fix
24/120 = 20.0 % (14-28), 7.10 VP; depth 1 31/120 = 25.8 % (19-34), 7.47 VP
(-5.8 points, s.e. 5.4).  Against `ab`: depth 2 after the fix 20/120 =
16.7 % (11-24), 6.71 VP; depth 1 30/120 = 25.0 % (18-33), 6.92 VP (-8.3
points, s.e. 5.2).  Pooled over every seed (31-34 and 41-42): vs `vf` depth
2 196/840 = 23.3 % (21-26), 7.17 VP against depth 1 210/840 = 25.0 %
(22-28), 7.30 VP (-1.7 +/- 2.1 points); vs `ab` depth 2 69/360 = 19.2 %
(15-24), 6.92 VP against depth 1 99/360 = 27.5 % (23-32), 7.20 VP (-8.3 +/-
3.1 points).  The +5.0 points vs `vf` on seeds 31+32 were seed noise: the
fixed depth 2 is at best at parity with depth 1 against the 1-ply stand-in
and clearly weaker against the alpha-beta one, for 4-5x the search time.
Against the pre-fix depth 2 (21.2 % / 17.9 %) the fix gains 1-2 points,
inside the noise.  On 276 decision nodes from real search-bot games the
fixed depth 2 still disagrees with *itself* under different dice on 16 % of
its top choices (main phase 23 %, discards 31 %) - almost as often as it
disagrees with depth 1 (20 %) - so the sampled lookahead remains mostly
noise at 12 samples.  Pooled adversarial numbers (seeds 31-34 + 41-42): vs ValueFunction stand-in depth 2 fixed 23.3 % (7.17 VP) vs depth 1 25.0 % (7.30 VP) over 840 games (-1.7 +/- 2.1 points); vs AlphaBeta stand-in 19.2 % (6.92 VP) vs 27.5 % (7.20 VP) over 360 games (-8.3 +/- 3.1 points); depth 2 costs 0.07 s per decision vs 0.015 s.  On 276 real-game decision nodes the fixed depth 2 disagrees with itself under different dice on 16 % of top choices.  Conclusion: with the heuristic evaluator the sampled opponents'-turn lookahead is not a strength lever; the advisor and self-play default to depth 1.

## 5. ML contract

`features.py`

```python
FEATURE_NAMES: list[str]                     # length == NUM_FEATURES
NUM_FEATURES: int
def extract(state, player) -> np.ndarray     # float32, shape (NUM_FEATURES,)
def extract_batch(states, players) -> np.ndarray   # (N, NUM_FEATURES)
```

Features are from the perspective of `player` ("me" first, then opponents in
seat order after me, padded to 3 opponents).  Include per player: public VP,
hidden VP (own exact / opponents expected), settlements, cities, roads,
resources (5) and hand size, expected production per resource per roll
(pips-weighted, robber-aware, ×2 for cities), production diversity, number of
buildable settlement spots reachable now / within 1 road / 2 roads and the best
such spot's pips, can-afford flags (road, settlement, city, dev), longest road
length + award, knights played + award, dev cards by type (own exact /
opponents count), port ratios (5), robber currently on my hex (production
lost).  Global: bank (5), dev deck remaining (5, or counts), turn number,
number of players, phase flags.  Keep it dense and < 400 dims.

`model.py`

```python
class ValueNet:
    def __init__(self, n_in=NUM_FEATURES, hidden=(256, 128), seed=0): ...
    def predict(self, X: np.ndarray) -> np.ndarray       # (N,) probabilities in [0,1]
    def evaluate(self, states, players) -> np.ndarray    # extract_batch + predict
    def fit(self, X, y, epochs, batch_size, lr, weight_decay, X_val=None, y_val=None, log=None) -> dict
    def save(self, path); @staticmethod load(path) -> ValueNet
```

Pure numpy forward/backward (ReLU, sigmoid output, binary cross-entropy,
Adam, L2).  Weights stored with `np.savez`.

`selfplay.py`

```python
def play_game(bots: list[Bot], state=None, rng=None, record=False, max_turns=400) -> GameResult
    # GameResult: winner, vp per player, turns, and (if record) samples: list of (features, player, ...)
def generate_dataset(num_games, bot_factory, workers, seed, ...) -> (X, y, meta)
def tournament(bot_factories, games, workers, seed) -> win-rate matrix / summary
```

Training targets: for every decision state in a game, `y = 1` if `player`
eventually won else `0` (Monte-Carlo return); optionally TD(λ) blending with
the net's own later predictions.  Positions are rotated so each sample is
from one player's perspective; sample all players at every state.

`train.py`: iterations of (generate games with the current best net + the
heuristic bot + exploration ε) -> fit new net on the growing replay buffer ->
evaluate new vs old head-to-head -> promote if win rate ≥ 52 %.  CLI flags
for games, iterations, workers, output path.  Writes `models/value_net.npz`
and `models/train_log.json`.

## 6. Strategy modules (explicit logic; used by search priors and explanations)

* `heuristic.py`: `HeuristicEvaluator().evaluate(states, players)`, `static_value(state, player) -> float`,
  `action_priors(state, actions) -> list[float]` (higher = try first).
* `placement.py`: `score_settlement_spot(state, player, v)`, `best_settlement_spots(state, player, k)`,
  `score_city(state, player, v)`, `road_targets(state, player)`, `resource_scarcity(state)`.  Both spot
  scorers subtract the blockability penalty `block_penalty(state, player, extra_settlement=v | extra_city=v)`
  `= PLACEMENT_BLOCK_WEIGHT x (robber_exposure after - before)`: the demand-weighted pips one robber
  placement blocks *because* our buildings share hexes (`P_block ~ W(h)^2`, 1.5x on a hex a 5+ VP opponent
  works; 0 for a first building or any layout without shared hexes).  Constants `PLACEMENT_ROBBER_Q`,
  `PLACEMENT_BLOCK_WEIGHT` (0 = old scores), `PLACEMENT_STRONG_THREAT`; `BlockContext` caches the per-player
  part; `cpp/heuristic.cpp::score_spot` mirrors `score_settlement_spot` bit for bit (STRATEGY.md, Placement).
* `trading.py`: `plan_trades(state, player, target_cost) -> list[Action]` (bank/port first when
  affordable and no cheaper player deal; player proposals otherwise or when the bank is empty),
  `should_accept(state, responder, offer, evaluator) -> bool`, `offer_is_feeding_leader(...)`.
* `discard.py`: `choose_discard(state, player, keep_for=None) -> Action`, `seven_risk(state, player) -> float`,
  `surplus_dump_actions(state, player) -> list[Action]`.
* `robber.py`: `best_robber_move(state, player, evaluator=None, target_weights=None, belief=None) -> (hex, victim, reason)`,
  `should_play_knight(state, player) -> (bool, reason)`, `production_blocked(state, player) -> float`,
  `target_weight(state, i)` (= VP `threat` x `danger.danger_multiplier`), `hex_damage(..., paths=None)`
  (need-aware blocking), `choose_victim(..., paths=None, our_need=None)` (resource-aware stealing),
  `steal_exposure(state, player, politics=None) -> (p_robbed, expected_loss, detail)` and the O(n^2)
  `steal_exposure_fast` used by `heuristic.static_value` (mirrored in `cpp/heuristic.cpp`).
* `danger.py`: `win_paths(state, belief=None) -> {player: WinPath}` (cached per identical state) and
  `win_path(state, i, belief=None)`: the cheapest builds to 10 VP given the hand (city upgrades,
  settlements on spots <= 2 roads away, Longest Road, Largest Army, VP cards), `missing` cards,
  `need_share` / `eff_need` (direct need plus use as port currency for a needed resource), per-resource
  `supply` (production + surplus traded through the best port), the `rolls` that feed the path, `turns`
  to afford it, `can_win_now` and `danger` in 0..1 (`1 / (1 + turns / 3)`).  `block_factor(wp, res, pips)`,
  `steal_factor(wp, our_need)`, `rob_break_probability(wp)`, `danger_lines(state, me)` (advice).
* `devcards.py`: `should_buy_dev(state, player) -> (bool, reason)`, `monopoly_value(state, player, res) -> float`,
  `best_year_of_plenty(state, player) -> (r1, r2)`.
* `counting.py`: `BankTracker`, `DevDeckTracker(state) -> probabilities of next card`,
  `HandBelief` (per opponent per resource distribution) with `observe_*` methods for live play,
  `expected_opponent_hands(state) -> list[list[float]]` (from a screenshot: counts + production prior).
* `inference.py`: `determinize(state, me, rng, belief=None) -> GameState` (fills opponents' hidden
  resources & dev cards consistent with `hand_size`/`dev_count`, the bank, dev deck remaining),
  `sample_states(state, me, n, rng) -> list[GameState]`.

## 7. Vision

`vision/schema.py`: `PARSE_SCHEMA` (JSON schema dict) for a *parsed screenshot*:

```json
{
  "hexes": [{"resource": "wood", "number": 9}, ... 19 in board index order ...],
  "robber": 9,
  "ports": [{"edge": 6, "type": "3:1"}, ...]            // optional
  "players": [{"color": "red", "name": "Alice", "vp": 3, "cards": 5, "dev_cards": 1,
               "knights": 0, "longest_road": false, "largest_army": false,
               "settlements": [..vertex ids..], "cities": [...], "roads": [..edge ids..],
               "resources": {"wood": 1, ...} /* only for "me" */ }],
  "me": "red",
  "current_player": "red",
  "dice": 8,
  "bank": {...} /* optional */, "dev_deck_remaining": 14 /* optional */
}
```

`vision/schema.py` also provides `parsed_to_state(parsed: dict) -> GameState`
(applies standard priors: bank = 19 − visible; dev deck = 25 − held − played;
opponents get `hand_known=False`), `validate(parsed) -> list[str]` (warnings:
wrong multiset of numbers/resources, pieces on impossible vertices, ...).

`vision/synth.py`: `render_state(state, size=(1280, 800), style=ColonistStyle(), seed=0) -> PIL.Image`
draws a Colonist.io-look-alike: blue sea, hex tiles in Colonist colours with
subtle texture, cream circular number tokens (red 6/8) with pips, robber,
coloured settlements / cities / roads, port icons on coastal edges, a player
panel (name, VP, card count, dev count, knights, badges) and a hand bar for
"me".  Also `render_digit_samples(...)` for classifier training.  Randomises
scale, offset, colour jitter, JPEG quality.

`vision/colonist.py`:

```python
@dataclass
class ParseResult:
    parsed: dict          # schema above
    state: GameState
    confidence: dict      # per field 0..1
    warnings: list[str]
    debug: dict           # geometry (hex centers, size), for --debug rendering
def parse_image(path_or_image, me: str | None = None, assume_standard=True, calibration=None) -> ParseResult
```

Pipeline: sea colour mask -> land mask -> fit the 19-hex lattice (center, hex
size) -> classify each hex by colour (constrained assignment to the standard
4/3/4/4/3/1 multiset when `assume_standard`) -> locate number tokens near hex
centres, classify digits with `digits.DigitClassifier` (constrained to the
standard multiset), red tokens = 6/8 -> robber = dark blob -> buildings at
`VERTEX_POS`, roads at `EDGE_POS` by saturated-colour blobs clustered into
player colours -> ports from coastal icons (best effort) -> player panel /
hand by colour + digits (best effort; missing fields become warnings).  Must
round-trip on `synth.render_state` output at several scales.

`vision/llm.py`: `parse_with_claude(path, me=None, model=None) -> ParseResult` using the
Anthropic Messages API with the image + a tool/structured output whose schema is
`PARSE_SCHEMA`.  Fails with a clear message if `anthropic` or the API key is missing.

## 8. CLI (`python -m catanbot`)

```
catanbot analyze IMAGE [--me COLOR] [--parser auto|cv|llm] [--state STATE.json]
                      [--depth N] [--beam K] [--model models/value_net.npz]
                      [--fix "hex 4=wheat 8" ...] [--save-state out.json] [--debug out.png]
catanbot recommend --state STATE.json [--me COLOR] [...]
catanbot play [--players 4] [--bots search,heuristic,heuristic,random] [--seed S] [--verbose]
catanbot train [--iters N] [--games G] [--workers W] [--out models/value_net.npz]
catanbot eval [--games G] [--a search] [--b heuristic]
catanbot render STATE.json OUT.png
```

`analyze` prints: parsed board as ASCII, warnings, then the top actions with
values and explanations, plus the situational advice sections (trade plan,
7-risk, knight/robber, dev card).  Exit code 0.

## 9. Testing

`pytest -q` must pass.  Every module ships tests.  Engine tests cover every
rule bullet in §3.  Vision tests render synthetic states and check the parse
round-trips (hexes, numbers, robber, pieces).  Keep each test < 30 s.

## 10. Conventions

* Python 3.10+, type hints, docstrings on public functions.
* `random.Random` for game randomness, `numpy.random.Generator` in ML code.
* No global mutable state.  No prints in library code (return / log).
* Performance matters in `engine.py` / `features.py` (self-play runs
  thousands of games): avoid per-call object churn, precompute tables.

## 11. Exploitative play, game stage, dev-card timing (added requirements)

* `catanbot/opponent_model.py` (DONE): `OpponentProfile` per player *name*
  with exponentially decayed statistics (offer acceptance overall / per
  resource received / per resource paid, implied resource valuations updated
  from every accept / reject / proposal / bank trade, robber targets, risk of
  holding > 7, build tendencies, "surprise" vs our heuristic).
  `OpponentModel.observe(state_before, action, player, predicted=None)` is
  called for every public action (the self-play runner and the search bot
  must call it; `predicted` = our heuristic's top action for that decision
  when cheap to compute).  `predict_accept(state, j, receives, pays,
  proposer)` gives P(accept) used by the search for `PROPOSE_TRADE`
  branches: `EV = p * V(accepted) + (1 - p) * V(rejected)`.
  `rank_offers`, `arbitrage_opportunities` (direct and intermediary deals
  where the counterpart's implied valuation disagrees with ours) feed offer
  generation.  `save/load` JSON so profiles persist across games and
  screenshots; the CLI takes `--profiles FILE` and `--event "blue accepted
  give ore get wood"` lines.
* **Game stage**: `opponent_model.game_stage(state)` (0..1) and
  `trade_stage_factor(state)` (1.0 early -> 0.3 late).  Trading logic already
  uses them: `should_accept` raises its margin with stage and refuses late
  trades with anyone ahead of us; `plan_trades` prefers the bank late.  The
  search must multiply the prior of `PROPOSE_TRADE` actions by
  `trade_stage_factor` and cap proposals per turn (2 late, 4 early).
* **Dev cards bought this turn** cannot be played until the next turn (VP
  cards count immediately).  The engine enforces this via `dev_cards_new`;
  the search must not assume a freshly bought knight is playable in the same
  turn, and explanations must say "(playable next turn)" for `BUY_DEV`.
* **Self-play training** (`train.py`): opponents are sampled from
  {current best search bot, heuristic bot with temperature, previous nets}
  and trading behaviour is randomised (epsilon on proposals / responses,
  temperature over offer ranking, random per-game acceptance bias) so the
  value net sees varied trading styles instead of one deterministic
  policy.  Player count is sampled from {3, 4} per game.

## 12. Out-of-turn risk and distance-to-win targeting (added requirements)

* **Steal exposure** (`robber.steal_exposure`): every opponent who rolls before
  our next roll can rob us with a 7 (1/6) or a plausible knight (deck odds x
  their dev cards, 50% they play it); whether they *target* us comes from
  their best robber move (politics-weighted).  Expected loss = P(robbed) x
  the average value of our cards (demand x sqrt(scarcity)).  The static
  evaluator subtracts `0.25 x loss` (hands of 3+ cards) via the O(n^2)
  `steal_exposure_fast`, so the search prefers spending a valuable hand or
  keeping cheap cards when we are the natural target; the 7-risk advice
  reports it (`explain_seven_risk(state, me, politics)`).
* **Danger, not rank** (`danger.py`): robber / knight targets, the political
  target weights and the priors weight each player by VP threat x
  `(0.4 + 1.6 x danger)`, where danger comes from the estimated turns until
  their cheapest win path is affordable.  A loaded runner-up (cards in hand,
  a spot, the rolls to finish) outranks an overextended leader (all cities
  built, no spot, empty hand).  Blocking is need-aware: a hex counts for the
  share of the target's supply of a *needed* resource it removes, so
  blocking sheep is worth little when they hold sheep, produce it elsewhere
  or turn a surplus into what they need through a 2:1 / 3:1 port (the port
  currency itself becomes worth blocking).  Stealing is resource-aware: a
  hand rich in what its owner needs (or what we need) is a better steal, and
  a player who can win on their turn gets a large bonus because one stolen
  card may break the build.  The CLI prints a "Threat board" (per player:
  VP, loaded / building / overextended, ~turns to win, the path, the missing
  cards and the rolls that produce them).

## 13. Win-path races: search hooks (`catanbot/winpaths.py`; off by default)

Model and knobs: docs/STRATEGY.md "Win-path races"; experiments: docs/ABLATIONS_WINPATHS.md.

* **Config.**  Five `SearchConfig` fields after `native_future`: `paths = 0` (1 = on),
  `paths_w = 1.0` (value weight g), `paths_crowd = 1.0` (waste-cost scale),
  `paths_priors = 1`, `paths_spots = 0`.  `reduced_config` passes them through (depth
  >= 3 sub-searches); `native_level_dict` does not read them (they never reach C++).
  `selfplay.make_bot` reads the spec keys `paths`, `paths_w`, `paths_crowd`,
  `paths_priors`, `paths_spots`, so the Catanatron adapter, `ablate_catanatron
  --cand-spec` and campaign `cand_spec` strings can switch it on.
* **Hooks in `Searcher`** (all guarded by `self._paths`, which is `None` unless
  `config.paths` is set):
  1. `search()` resets `self._paths = None` with the other per-search caches; after the
     single-legal-action early return (forced decisions build nothing) and outside the
     setup phases it builds `winpaths.PathsEvaluator.for_search(self.evaluator, state,
     me, cfg)` - a lazy import, so with `paths = 0` the module is never imported or run.
  2. `_eval` evaluates through the wrapper when it exists (every leaf, the lookahead's
     greedy opponents and the reduced sub-search leaves).
  3. `_candidate_priors` ends with `ctx.adjust_priors(state, legal, priors)` when
     `paths_priors` (our own main-phase nodes only; the simulated opponents' priors are
     untouched).
  4. `_candidates` hands the wrapper to `political_trade_options` (its "at no cost to
     us" test then uses the same values as the search).
  5. `_future_values` takes the C++ lookahead only when `self._paths is None`: the
     native opponent simulation cannot see the term, so `paths = 1` at depth >= 2 runs
     the Python lookahead (slower; the feature is budgeted for depth 1 only).
* **Contract of the wrapper.**  `PathsEvaluator.evaluate(states, players)` has the
  evaluator interface.  For a `HeuristicEvaluator` base it returns
  `softmax_i((static_values(s)_i + corr_i) / T)[player]` with `T` the base's temperature
  and the base's max subtraction; a `BlendedEvaluator` gets the correction in its
  heuristic half; any other evaluator (a value net) gets
  `base + softmax((V + corr) / 16)[p] - softmax(V / 16)[p]`.  Finished games return 1 / 0,
  setup states the base's value; `paths_w = 0` reproduces the base within 1e-12.
* **Per-decision context.**  `PathsContext` holds the root quantities (horizon, dev pool,
  bank factor, liveness gates, the opponents' road room) and frozen copies of the module
  constants (so `ParamBot`'s apply / restore around `decide` is enough), plus memo dicts
  that live for one `search()` call: supply by (seat, buildings, robber), trail lengths by
  (buildings, seat, roads), our road room, race solutions by their exact inputs, reach sets
  and spot scores.  Every memo value is a pure function of its key and no RNG is used, so
  cold and warm caches give bit-identical values (tested) and the search's RNG streams are
  untouched.  Opponents cannot build during our turn, so their entries hit on every leaf
  after the first; the race-solve hit rate is ~96 %.
* **Invariant (tested).**  With `paths = 0` the default bot plays byte-identical action
  sequences and root searches return identical values, lines and explanations - before and
  after `catanbot.winpaths` is imported, with `paths=0` spelled out and with the module's
  constants overridden (tests/test_winpaths.py,
  `test_default_unchanged_six_games_and_thirty_searches`).
* **Cost** (depth 1, beam 4, expand 8): ~26 us per leaf on top of the batched C++
  evaluation, ~0.12 ms per context; 1.26x the default's mean decision time on 30
  mid-game positions (1.56x with spots), 1.18x mean / 1.25x p95 on the 281 main-phase
  roots of one shadow game.
