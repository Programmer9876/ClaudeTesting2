# C++ acceleration (`catanbot_core`)

`catanbot_core` is an optional pybind11 extension that ports the hot,
low-level code of the bot to C++.  Today it contains **feature extraction**
(`catanbot/features.py::extract` / `extract_batch`), `longest_road_length`,
the **heuristic evaluator** (`catanbot/heuristic.py::static_value` /
`HeuristicEvaluator.evaluate`) and the **rules engine**
(`catanbot/engine.py::legal_actions` / `apply` / `apply_inplace` /
`random_playout`, see the last section; opt-in).  The Python modules stay
the tested reference implementation and the fallback: with the extension
missing (or disabled) everything works exactly as before, only slower.

The output is bit-identical to the numpy reference (the differential test
compares 4000+ game states from every phase at `atol=1e-6`; they match
exactly), and about **40x faster** on realistic states.

## Build

```bash
scripts/build_cpp.sh              # regenerate tables, compile, smoke test
scripts/build_cpp.sh --clean      # wipe build/ and the .so first
```

The script runs `python3 setup_cpp.py build_ext --inplace` and produces
`catanbot/catanbot_core.<abi>.so` (importable as `catanbot.catanbot_core`;
a top-level `catanbot_core` module is accepted as well).  Requirements:
`g++` with C++17, `python3` with `numpy`, `setuptools` and `pybind11`
(`pip install pybind11`).  Flags: `-O3 -march=native -std=c++17
-ffp-contract=off` (strict floating point so results match numpy bit for
bit).  The binary is tuned to the build machine; set
`CATANBOT_CPP_PORTABLE=1` to drop `-march=native`.  A build takes ~10 s.
The `.so` and `build/` are build artefacts (do not commit them).

## Files

| file | what |
| --- | --- |
| `scripts/gen_board_tables.py` | generates the two headers below from `board.py` / `features.py`; `--check` exits 1 if they are stale (run by the tests) |
| `cpp/board_tables.hpp` | *generated*: hex/vertex/edge adjacency (`HEX_VERTICES`, `HEX_EDGES`, `VERTEX_HEXES`, `VERTEX_NEIGHBORS`, `VERTEX_EDGES`, `EDGE_VERTICES`, `EDGE_HEXES`, ...), `PIPS`, costs, piece limits, standard layout / ports |
| `cpp/feature_layout.hpp` | *generated*: phase enum (in the one-hot order), `PLAYER_BLOCK`, `GLOBAL_BLOCK`, `NUM_FEATURES`, every feature offset (`P::prod_ore`, `G::vp_gap`, ...) and `FEATURE_NAMES` |
| `cpp/state.hpp` | `GameStateC` / `PlayerC` / `TradeC`: plain C++ mirror of `catanbot.state.GameState` + `state_from_python()` converter |
| `cpp/features.hpp`, `cpp/features.cpp` | the port of `features.py` (`analyse`, `assemble`, `longest_road_length`) |
| `cpp/heuristic.hpp`, `cpp/heuristic.cpp` | the port of `heuristic.py::static_value` + the placement / counting helpers it uses |
| `cpp/engine.hpp`, `cpp/engine.cpp` | the port of `engine.py` (legal actions, every action handler, production, awards, win detection, random playout) |
| `cpp/module.cpp` | pybind11 bindings (`PYBIND11_MODULE(catanbot_core)`), incl. action tuple <-> `ActionC`, `GameStateC` -> `GameState` and the `CState` handle |
| `setup_cpp.py`, `scripts/build_cpp.sh` | build |
| `catanbot/accel.py` | loader / switch used by `features.py`, `heuristic.py` and `engine.py` |
| `tests/test_accel_features.py` | differential tests + benchmark (features) |
| `tests/test_accel_heuristic.py` | differential tests + benchmark (heuristic) |
| `tests/test_accel_engine.py` | differential tests + benchmark (engine) |
| `scripts/bench_search.py` | `Searcher.search` timings, Python vs C++ |

Never edit the generated headers: change `board.py` / `features.py` and
re-run `scripts/gen_board_tables.py` (the build script does it).  Ragged
Python tables are emitted as fixed-width arrays padded with `-1` plus a
`*_N` count array (`VERTEX_EDGES[v][k]` joins `v` and
`VERTEX_NEIGHBORS[v][k]`).

## Python API

```python
import catanbot.catanbot_core as core   # or: from catanbot import accel; core = accel.load_core()

core.extract_batch(states, players) -> np.ndarray   # float32 (N, NUM_FEATURES)
core.extract(state, player)         -> np.ndarray   # float32 (NUM_FEATURES,)
core.longest_road_length(state, player) -> int
core.num_features() -> int;  core.feature_names() -> list[str];  core.phase_names() -> list[str]
core.NUM_FEATURES, core.PLAYER_BLOCK, core.GLOBAL_BLOCK, core.MAX_PLAYERS
```

`states` are ordinary `catanbot.state.GameState` objects.  Each *distinct
consecutive* state object is converted and analysed once (like the Python
`extract_batch`), so `extract_batch([s] * n, range(n))` costs one analysis.
Errors mirror Python: `ValueError` for a length mismatch, `IndexError`
for a player index outside the state.  A state the fixed-size C++ structs
cannot hold (more than 4 players, lists of the wrong length, ids or integers
outside the 32-bit fields, more than 64 road entries for one player) raises
`core.UnsupportedStateError`, a `ValueError` subclass; the `accel` wrappers
catch it and compute that call with the Python reference, so
`features.extract*` / `HeuristicEvaluator.evaluate` behave exactly as without
the extension for such states (only slower).

### The switch (`catanbot/accel.py`)

`features.extract` and `features.extract_batch` start with

```python
if _accel.AVAILABLE:
    return _accel.extract_batch(states, players)
```

`accel.AVAILABLE` is true when the extension imported and
`CATANBOT_NO_ACCEL` is unset.  The first accelerated call verifies that the
extension was compiled for the current `features.FEATURE_NAMES`; on a
mismatch (stale build after a layout change) it emits a `RuntimeWarning`,
sets `AVAILABLE = False` and falls back to Python, so a stale build can
never produce silently wrong features.  `model.ValueNet.evaluate`,
`selfplay.play_game(record=True)` and the search all go through
`features.extract_batch`, so they are accelerated automatically.

To disable: `CATANBOT_NO_ACCEL=1 python -m catanbot ...`, or
`catanbot.accel.AVAILABLE = False` at run time (the tests use this to get
the pure-Python reference in the same process).

## Design notes

* **Conversion, not serialisation.**  `state_from_python` reads the
  `GameState` attributes with the raw CPython API (interned attribute
  names, `PySequence_Fast`, `PyLong_AsLongAndOverflow`) and never calls `to_dict()`.
  Converting a 4-player state costs ~4 µs; the analysis itself ~1 µs, so
  ~5 µs per state in total.  The GIL is held (the work is too small to be worth releasing it).
* **`GameStateC`** is POD-like (fixed arrays, ~1.3 KB, `memcpy`-able):
  `hex_res/hex_num[19]`, `robber`, `ports[54]` (-1 = none), up to 4
  `PlayerC` (resources / dev cards / new dev cards `[5]`, knights, ordered
  settlement / city / road id lists **and** bitsets, `hand_known`,
  `hand_size`, `dev_known`, `dev_count`), `bank[5]`, `dev_deck[5]`,
  `current`, `phase` (enum in the one-hot order, `PHASE_UNKNOWN` for
  anything else), `turn`, `setup_round`, `setup_last_settlement`, `dice`,
  `dev_played_this_turn`, `free_roads`, `discard_queue`, `pending_trade`
  (`proposer`, `give[5]`, `get[5]`, `responses[4]`), `trade_responder`,
  `trades_this_turn`, the awards, `winner`, `max_turns`.  Player colour /
  name are not copied.  Helpers: `public_vp`, `total_vp`, `vertex_owner`,
  `edge_owner`, `acting_player`.  Define `CATAN_NO_PYTHON` to use the
  struct without `Python.h` (e.g. for a pure C++ engine port).
* **Exactness.**  `features.cpp` mirrors `features.py` statement by
  statement: the same double accumulation order (settlements then cities,
  a city added twice, the robber table built by copy-then-subtract), the
  same points where values are narrowed to float32 (the VP gaps are
  computed from the float32 block values like the numpy code), `math.log`
  via the same libm, `-ffp-contract=off` to forbid FMA contraction.
* **Longest road** is the same DFS over trails (roads not reused, vertices
  may repeat, an opponent's building ends a path) with a 64-bit "used
  roads" mask; per-entry bits, so duplicated road ids behave like the
  Python reference.
* **Settlement spots** replicate `_spots_for_player` (BFS over free edges
  from the road network, never through an opponent's building, shared
  `seen` set across the 0/1/2-road levels).

## Tests and benchmark

```bash
PYTHONPATH=. python3 -m pytest tests/test_accel_features.py tests/test_features.py -q -p no:cacheprovider
```

`tests/test_accel_features.py` plays heuristic-bot games (2, 3 and 4
players), keeps every intermediate state (all phases: setup, roll, main,
discard, robber, trade_response, trade_select, game_over) plus hidden-hand
variants (`hand_known=False`, `dev_known=False`, robber on the desert,
unknown phase string) and asserts `np.allclose(cpp, python, atol=1e-6)`
for every perspective of every state (> 4000 states), `longest_road_length`
equality (game states + random road subsets with cycles), the layout drift
check, the fallback behaviour of `accel`, and prints the benchmark.  The
tests skip when the extension is not built.

Measured on the development container (4 shared cores, background training
running), `extract_batch` over 500 distinct mid-game states:

| | per call | per state |
| --- | --- | --- |
| Python (`features.py`) | ~115-120 ms | ~240 µs |
| C++ (`catanbot_core`) | ~2.5-3.2 ms | ~5 µs |
| speedup | **38-47x** (varies with the shared-core load) | |

Of the ~5 µs per state about 4 µs is reading the Python `GameState`
(roughly 200 attribute / list-item reads) and ~1 µs is the actual analysis,
so a further speed-up would have to come from keeping states in C++ (an
engine port) rather than from the feature code itself.

## Limitations / notes

* Feature extraction, the heuristic evaluator and (opt-in) the rules engine;
  the move ordering (`action_priors`) and the search itself are still Python.
* At most 4 players (the feature layout pads to 3 opponents anyway) and
  32-bit integer fields.  Calling `core.*` directly on such a state (or on
  resource lists that are not length 5, ids out of range, > 64 road entries
  for one player) raises `core.UnsupportedStateError` (a `ValueError`); the
  `accel` wrappers fall back to the Python reference for it.  Other malformed
  input raises the usual Python errors (`TypeError` for a non-numeric hex
  number, ...); the desert's number is ignored like in Python.  Port types
  outside `0..5` are ignored.
* `-march=native` binaries are not portable between CPUs; rebuild on each
  machine (or use `CATANBOT_CPP_PORTABLE=1`).
* The extension must be rebuilt after changing `board.py` or the feature
  layout in `features.py` (`scripts/build_cpp.sh`; the tests and the
  run-time check in `accel.py` catch a stale build).

## Heuristic evaluator (`heuristic.static_value` / `HeuristicEvaluator.evaluate`)

The second port covers the hand-written evaluator: the search uses it as the
leaf evaluator when no value net is loaded (`HeuristicEvaluator`), the
politics module and the CLI fall back to it as well.  In a pure-Python search
it was ~80 % of `Searcher.search` (`static_value` costs ~120 µs per
(state, player)).

Ported, statement by statement, from `catanbot/heuristic.py::static_value`
and everything it calls:

* `placement.vertex_production` / `player_production` (robber-aware and
  robber-free), `resource_scarcity`, `RESOURCE_DEMAND` (computed at run time
  with the same float operations as the Python list comprehension);
* `placement.reachable_spots` (BFS from the own buildings / road endpoints
  over free edges; a vertex holding an opponent's building is never entered,
  and an own road endpoint that is an opponent's building cannot be expanded
  from; `first_edge` bookkeeping included);
* `placement.score_settlement_spot` incl. `_expansion_potential`, the port
  synergy terms and the `setup` flag (not used by `static_value`, exposed for
  completeness);
* `heuristic.longest_road_length`, `heuristic._progress_to_build`
  (`buildable_settlements` reduced to "any spot exists"), `counting.expected_hidden_vp`
  (`dev_pool`), the hand / dev-card / award / port terms and the terminal
  handling (`+-1000` once the game is over with a winner; a turn-capped game
  without a winner is evaluated like a live position);
* `HeuristicEvaluator.evaluate`: softmax of the players' static values at
  `temperature`, overridden by 1 / 0 for a decided game.  Like the Python
  `id(s)` cache, every distinct state object in a batch is converted and
  scored once, wherever it appears.

### Python API

```python
core.static_values(state) -> list[float]          # static_value of every player, VP-equivalents
core.static_value(state, player) -> float
core.heuristic_evaluate(states, players, temperature=16.0) -> np.ndarray   # float64 (N,)
# helpers (mainly for the differential tests):
core.heuristic_longest_road_length(state, player) -> int
core.reachable_spots(state, player, max_roads=3) -> {vertex: (roads_needed, first_edge)}
core.score_settlement_spot(state, player, vertex, setup=False) -> float
core.player_production(state, player, ignore_robber=False) -> list[float]  # 5 entries
core.resource_scarcity(state) -> list[float]
core.expected_hidden_vp(state, player) -> float
core.progress_to_build(state, player) -> float
```

`catanbot.accel` wraps the three main ones (`accel.static_values`,
`accel.static_value`, `accel.heuristic_evaluate`) with the usual fallback to
Python.  Errors mirror the feature functions: `ValueError` for a length
mismatch or a malformed state, `IndexError` for a player outside the state.

### The switch

`HeuristicEvaluator.evaluate` starts with

```python
if _accel.AVAILABLE:
    return _accel.heuristic_evaluate(states, players, self.temperature)
```

and the Python body below it stays the reference (`static_value` itself is
untouched and still pure Python; `action_priors` and the strategy modules do
not use the extension).  `accel.verify()` now also requires the heuristic
entry points, so a `.so` built before this port is disabled as a whole with
the usual `RuntimeWarning` instead of half-working.  `CATANBOT_NO_ACCEL=1` /
`accel.AVAILABLE = False` disable it like the features.

### Exactness

The values are **bit-identical** to the Python reference on the development
machine (Python 3.11, glibc libm): the differential test compares 18 758
static values from > 2 000 game states plus hidden-hand, terminal, award,
port and random-network variants and finds a maximum difference of 0, and the
same for `evaluate` at four temperatures.  What makes that work:

* the same accumulation order everywhere (`sum(...)` of five floats is a
  sequential double accumulation from 0.0; a city's production is added once
  as `2.0 * prod`, unlike `features.py` which adds it twice);
* `x ** 0.5` is libm `pow(x, 0.5)` in CPython, which is *not* guaranteed to
  equal `sqrt(x)` bit for bit, so `score_settlement_spot` calls `pow` with
  an exponent the compiler cannot see (`math.sqrt` in `static_value` maps
  to `sqrt`, `math.exp` to `exp`; same libm, `-ffp-contract=off`);
* `heuristic.longest_road_length` is ported on its own: unlike
  `features.longest_road_length` a path may *end* but not *start* on an
  opponent's building, and a duplicated road id is one road (the Python
  `used` set is keyed by edge id).  Both behaviours are pinned by a test.
* `reachable_spots` distances are exact; `first_edge` can differ from Python
  when several first roads reach a spot at the same distance (Python's pick
  depends on `set` iteration order).  `static_value` only uses the distances.
* Python >= 3.12 sums floats with Neumaier compensation, so there the last
  bits of `pv`, `pv_free` and the softmax denominator can differ (far below
  the 1e-6 test tolerance).

### Tests and benchmark

```bash
PYTHONPATH=. python3 -m pytest tests/test_accel_heuristic.py tests/test_search.py -q -p no:cacheprovider
python3 scripts/bench_search.py            # Searcher.search, Python vs C++ (two subprocesses)
python3 scripts/bench_search.py -v --repeat 3 --depths 1 2 3 --evaluators heuristic --positions 3
```

`tests/test_accel_heuristic.py` plays heuristic-bot games (2, 3 and 4
players), keeps every state, adds the variants listed above and asserts
`|C++ - Python| <= 1e-6` for every player's `static_value` and for
`evaluate()`; it also checks every helper against its Python original, the
error cases, the accel switch (routing, fallback, stale-build detection) and
prints the benchmark.  It skips when the extension is not built.

`scripts/bench_search.py` times `Searcher.search` (beam 4, expand 8, depth
1 and 2, heuristic and value-net evaluators) on 5 mid-game 4-player positions
in two subprocesses, `CATANBOT_NO_ACCEL=1` and accelerated, and prints the
totals, the share of the time spent in `evaluator.evaluate` and the speedups.
The workers pin BLAS to one thread (`--blas-threads`, see below).

Measured on the development container (4 shared cores, other jobs running;
best of 2):

| evaluator | depth | Python | of which evaluate | C++ | of which evaluate | speedup | nodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | evaluate only, 500 states | 244 ms | 100 % | 3.6 ms (7 µs/state, all seats) | 100 % | **68x** | - |
| heuristic | 1 | 0.59 s | 81 % | 0.119 s | 7 % | **5.0x** | 2511 / 2511 |
| heuristic | 2 | 1.90 s | 83 % | 0.336 s | 9 % | **5.7x** | 5325 / 5325 |
| net | 1 | 0.47 s | 70 % | 0.163 s | 13 % | **2.9x** | 2530 / 2530 |
| net | 2 | 1.66 s | 69 % | 0.611 s | 19 % | **2.7x** | 5498 / 5498 |

The node counts are identical, i.e. the accelerated search visits the same
tree and returns the same moves.  Of the ~7 µs per state about 4 µs is the
`GameState` conversion (see the design notes above) and ~3 µs the evaluation
of all seats.

### Where the time goes now

After the port the evaluator is 7-9 % (heuristic) / 13-19 % (net) of
`Searcher.search`; the remaining ~90 % is the pure-Python rules engine
(`engine.apply`, `legal_actions`) and the move ordering
(`heuristic.action_priors` with its strategy modules).  A further big step
needs the engine in C++ (`state.hpp` / `board_tables.hpp` are ready for it),
not more work on the evaluators.

One thing that is *not* the extension's doing but showed up while measuring:
the value net evaluates batches of 10-100 rows, for which OpenBLAS's default
thread pool is pure overhead, and on a loaded shared machine it made the
net-backed search 3-5x slower (the same 5 positions took 2.4-5.1 s instead
of 0.5-1.7 s in pure Python).  Set `OPENBLAS_NUM_THREADS=1` (and
`OMP_NUM_THREADS=1`) in search / self-play worker processes.

### Limitations / notes

* Same state limits as the features: at most 4 players, 32-bit integers,
  ids in range; otherwise `core.UnsupportedStateError` from `core.*`, while
  `accel.static_value*` / `HeuristicEvaluator.evaluate` fall back to the
  Python `static_value`.  Hex numbers outside 2..12 (other than 0) count as
  0 pips where Python raises `KeyError`; port types outside 0..5 are ignored
  where Python would raise or index oddly.
* `temperature=0` gives `inf` / `nan` instead of Python's
  `ZeroDivisionError`; a negative player index raises `IndexError` where
  Python would index from the end.
* `first_edge` of `reachable_spots` may differ on ties (see above).
* Rebuild after changing `heuristic.py`, `placement.py` or `counting.py`:
  there is no run-time check for the heuristic (only the differential test).

## Rules engine (`engine.legal_actions` / `apply` / `apply_inplace` / `random_playout`)

The third port covers the rules engine, so a search (or a rollout) can apply
actions without running Python.  `cpp/engine.cpp` is a statement-by-statement
translation of `catanbot/engine.py` over the `GameStateC` struct of
`state.hpp`: `legal_actions` (every phase, the same tuples in the same
order, `discard_options` with its ordering and cap, the bounded trade
candidates), every `_h_*` handler with the same checks in the same order and
the same error messages, production with the bank-shortage rule, robber /
steal, dev cards (buy, knight, road building with the free-road rules, year
of plenty, monopoly), Longest Road (`_update_longest_road_after_road` and the
full recompute after a cutting settlement), Largest Army, the trade protocol
(propose / respond / select / execute / cancel, auto-rejects), `END_TURN`,
the win check and the `max_turns` cap, `hand_size` / `dev_count` re-sync of
the touched players, and `random_playout`.

**Python-visible behaviour is identical**, including the random draws: the
engine asks a "draw source" exactly when and how the Python engine consults
its `random.Random` (`randint(1, 6)` twice for a roll, `randrange(total)` for
a steal and for a dev-card draw).  Three sources exist: a forced index
(`apply_forced`, for differential tests), a Python `random.Random` (or any
object with `randrange` / `randint`; the same rng object then produces the
same game in both engines and ends in the same state), and a C++ xoshiro256**
(`rng=None` or an int seed; used by `random_playout_fast`).

### Python API

```python
core.legal_actions(state) -> list[tuple]            # engine.legal_actions (state may be a CState)
core.apply(state, action, rng=None) -> GameState     # engine.apply: a NEW GameState (input untouched)
core.apply_inplace(state, action, rng=None) -> state # engine.apply_inplace: written back into the same objects
core.apply_forced(state, action, drawn_index) -> GameState
core.random_playout_fast(state, seed=None, max_turns=None, max_actions=2_000_000, trace=False)
#   -> GameState, or (GameState, [(action, draw), ...]) with trace=True
core.production_for_roll(state, value), core.discard_options(resources, k, cap=200),
core.acting_player(state), core.count_vp(state, player, include_hidden=True)   # helpers (tests)

h = core.CState(state)          # converted once, kept in C++ (~3 us); origin = state
h.legal_actions(); h.apply(action, rng=None) -> CState; h.apply_inplace(...); h.apply_forced(...)
h.random_playout(seed=None, max_turns=None) -> CState; h.copy(); h.to_state() -> GameState
h.phase, h.current, h.turn, h.dice, h.winner, h.num_players, h.acting_player, h.is_terminal,
h.max_turns (settable), h.rolls_history_len, h.origin; h.count_vp(player, include_hidden=True)
core.extract_batch / extract / static_values / static_value / heuristic_evaluate accept CState objects
```

* `rng`: `None` (a fresh C++ generator), an int seed (C++ generator, deterministic) or a
  `random.Random`.  `apply_forced`'s `drawn_index` is the index of the stolen card among the
  victim's cards in resource order, the index of the drawn dev card into the deck in dev-type
  order, or for an unforced `(ROLL,)` the dice-pair index `6 * (d1 - 1) + (d2 - 1)`; out of
  range raises `ValueError`.
* Illegal actions raise **`catanbot.engine.IllegalActionError` itself** (the class is looked up
  when first needed) with the Python message, so `except E.IllegalActionError` in the search
  works unchanged; `core.IllegalActionError` is only the stand-in when `catanbot.engine` cannot
  be imported.  `"game is over"` / `"unknown action <repr>"` behave like `engine.apply_inplace`.
* The new `GameState` is built like `GameState.copy()`: `hexes` / `ports` are the same objects
  as the input's, the players' `color` / `name` are copied, `rolls_history_len` is carried
  along (+1 per roll).  `apply_inplace` updates the existing `GameState`, `Player`, list and
  `TradeOffer` objects in place (item stores / slice assignment), so references held by the
  caller stay valid exactly as with the Python engine.
* States the structs cannot hold (> 4 players, ...) raise `core.UnsupportedStateError` as for
  the other entry points; `accel.engine_*` return `None` for them and `engine.py` falls back.
* `random_playout_fast(..., trace=True)` also returns every applied action with the random
  index it consumed (rolls are recorded as `(ROLL, value)`), so a C++ playout can be replayed
  through the Python engine with `apply_forced`-style stand-ins (the test does exactly that).

### The switch (`CATANBOT_ACCEL_ENGINE=1`, default off)

`engine.legal_actions`, `engine.apply` and `engine.apply_inplace` start with

```python
if _accel.ENGINE_ACTIVE:
    out = _accel.engine_apply(state, action, rng)   # None -> unsupported state -> Python path
    if out is not None:
        return out
```

`accel.ENGINE_ACTIVE` is true only when the extension loaded, it has the engine entry points
**and** the environment variable `CATANBOT_ACCEL_ENGINE` is set (`1` / `true` / `yes`).  The
Python engine stays the default; `random_playout`, `selfplay.play_game` and the search go
through these three functions, so setting the variable accelerates them without other changes
(`accel.verify()` now also requires the engine entry points, so a `.so` built before this port
is disabled as a whole).  With the switch on, `tests/test_engine.py` (78 tests, incl. the
30 fuzzed invariant playouts and the soundness / completeness universe) and `tests/test_search.py`
pass unchanged, i.e. every engine call of those suites runs through C++.

### Tests

```bash
PYTHONPATH=. python3 -m pytest tests/test_accel_engine.py tests/test_engine.py -q -p no:cacheprovider
CATANBOT_ACCEL_ENGINE=1 PYTHONPATH=. python3 -m pytest tests/test_engine.py -q -p no:cacheprovider
```

`tests/test_accel_engine.py` (skipped when the extension is not built) plays 315 games
(105 each with 2, 3 and 4 players, turn caps 60 / 120 / 250, a build-biased random policy so
games reach cities, dev cards, both awards, wins and the turn cap): at **every step**
`legal_actions` must be identical as lists and `apply_forced` must give the same `to_dict()`
as the Python engine driven with the same forced draw (the continuation alternates between the
two engines' results, so C++-made states are also checked as inputs); every 7th step the
in-place variant is checked too, incl. object identity.  Further tests: the `random.Random`
path (same rng object -> same state and same `rng.getstate()` afterwards, > 500 checks), the
error parity over the action universe of `test_engine.py` (same class, same message, > 2000
checks), `apply_forced` index semantics, unsupported states and the `accel` fallbacks,
`random_playout_fast` replayed through the Python engine (identical final state, the
`test_engine.py` invariants on cards / board / awards), the helpers, hidden-hand bookkeeping,
`CState` against the `GameState` path (incl. the evaluators taking handles), the `engine.py`
switch (routing spy, fallback, env activation in a subprocess) and the benchmark below.

### Benchmark

Measured on the development container (4 shared cores, background training running; states
from random / heuristic-bot games, best of 5):

| | Python (`engine.py`) | C++ via `GameState` objects | C++ via `CState` |
| --- | --- | --- | --- |
| `legal_actions` per call (random-play states, ~12 actions) | 4.2 us | 4.4 us (1.0x) | 0.54 us (**8x**) |
| `legal_actions` per call (heuristic-bot states, ~13 actions) | 5.8 us | 4.0 us (1.5x) | |
| `apply` per call, first legal action | 13-23 us | 8-11 us (**1.7-2.0x**) | 0.87 us (**15x**) |
| `apply` per call by kind | END_TURN 8 us, ROLL 15 us, BUILD_ROAD 22-42 us | 8-11 us flat (0.8x .. 3.8x) | ~1 us |
| random playout, 4 players, 100 turns (`random_playout` vs `random_playout_fast`) | 8.1 ms/game | 0.19 ms/game (**43x**) | same |
| random playout to the 400-turn cap | | 0.57 ms/game (19 / 20 end by 10 VP) | |

Reading: the engine itself costs ~0.4 us per action; on the `GameState` path the rest is the
conversion (~3 us to read a 4-player `GameState`, ~4.5 us to build the new one - the same as
Python's own `GameState.copy()`), so cheap actions gain nothing and expensive ones (roads with
the Longest Road DFS, settlements, rolls with production) gain 2-4x.  The 43x of the playout
and the 8-15x of `CState` are what the engine gives once the state stays in C++: a search that
keeps `CState` handles and evaluates them directly (the evaluators accept handles, so nothing
is converted at all) is the way to "apply actions without Python overhead"; converting back
with `to_state()` costs ~4.5 us only where a Python `GameState` is really needed.

### Limitations / notes

* Same state limits as the other ports (at most 4 players, 32-bit fields, ids in range) ->
  `core.UnsupportedStateError`; the `accel.engine_*` wrappers and the `engine.py` hook fall
  back to Python for them.  Hand-built states with a player index / `trade_responder` /
  award owner outside the player list raise `IndexError` like Python (from `std::out_of_range`),
  a state without players raises `ValueError` where Python raises `ZeroDivisionError`.
* Malformed action arguments (non-integers) raise `TypeError` when the action is parsed, i.e.
  before the phase checks, where Python raises it inside the handler after them; the
  well-formed-but-illegal cases (the whole `test_engine.py` universe) match message for message.
* Port types outside `0..5` are ignored (Python would `IndexError` in `_port_ratios`); a
  responder index >= 4 cannot be stored in `TradeC.responses`.
* The C++ random generator of `rng=None` / `seed` is not `random.Random`: `core.apply(s, a, 5)`
  and `engine.apply(s, a, random.Random(5))` draw different values (both are correct games);
  pass a `random.Random` object for bit-identical sequences.
* `-march=native` and the rebuild rules of the other ports apply; `setup_cpp.py` now lists
  `cpp/engine.cpp` as a source.
