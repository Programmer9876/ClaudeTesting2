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
  robber.py       knight timing, robber target + victim selection
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
    depth: int = 2            # number of *turns* to look ahead (own turn counts as 1)
    beam: int = 6             # candidate actions kept per decision node
    roll_samples: int = 11    # 11 = exact expectation over all rolls; fewer = top-probability rolls
    max_nodes: int = 20000
    time_limit: float | None = None
    opponent_model: str = "maxn"     # "maxn" | "greedy" | "paranoid"
    trade_acceptance: str = "value"  # opponents accept if their own value increases

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
  `score_city(state, player, v)`, `road_targets(state, player)`, `resource_scarcity(state)`.
* `trading.py`: `plan_trades(state, player, target_cost) -> list[Action]` (bank/port first when
  affordable and no cheaper player deal; player proposals otherwise or when the bank is empty),
  `should_accept(state, responder, offer, evaluator) -> bool`, `offer_is_feeding_leader(...)`.
* `discard.py`: `choose_discard(state, player, keep_for=None) -> Action`, `seven_risk(state, player) -> float`,
  `surplus_dump_actions(state, player) -> list[Action]`.
* `robber.py`: `best_robber_move(state, player, evaluator=None) -> (hex, victim, reason)`,
  `should_play_knight(state, player) -> (bool, reason)`, `production_blocked(state, player) -> float`.
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
