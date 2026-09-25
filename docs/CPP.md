# C++ acceleration (`catanbot_core`)

`catanbot_core` is an optional pybind11 extension that ports the hot,
low-level code of the bot to C++.  Today it contains **feature extraction**
(`catanbot/features.py::extract` / `extract_batch`), `longest_road_length`
and the **heuristic evaluator** (`catanbot/heuristic.py::static_value` /
`HeuristicEvaluator.evaluate`, see the section at the end).  The Python
modules stay the tested reference implementation and the fallback: with the
extension missing (or disabled) everything works exactly as before, only
slower.

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
| `cpp/module.cpp` | pybind11 bindings (`PYBIND11_MODULE(catanbot_core)`) |
| `setup_cpp.py`, `scripts/build_cpp.sh` | build |
| `catanbot/accel.py` | loader / switch used by `features.py` and `heuristic.py` |
| `tests/test_accel_features.py` | differential tests + benchmark (features) |
| `tests/test_accel_heuristic.py` | differential tests + benchmark (heuristic) |
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

* Feature extraction and the heuristic evaluator only; the engine, the move
  ordering (`action_priors`) and the search are still Python (they can reuse
  `state.hpp` and `board_tables.hpp`).
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
