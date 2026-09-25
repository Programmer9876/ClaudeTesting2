# C++ acceleration (`catanbot_core`)

`catanbot_core` is an optional pybind11 extension that ports the hot,
low-level code of the bot to C++.  Today it contains **feature extraction**
(`catanbot/features.py::extract` / `extract_batch`) and
`longest_road_length`.  The Python modules stay the tested reference
implementation and the fallback: with the extension missing (or disabled)
everything works exactly as before, only slower.

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
| `cpp/module.cpp` | pybind11 bindings (`PYBIND11_MODULE(catanbot_core)`) |
| `setup_cpp.py`, `scripts/build_cpp.sh` | build |
| `catanbot/accel.py` | loader / switch used by `features.py` |
| `tests/test_accel_features.py` | differential tests + benchmark |

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
Errors mirror Python: `ValueError` for length mismatch or malformed states
(wrong list lengths, ids out of range, more than 4 players), `IndexError`
for a player index outside the state.

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
  names, `PySequence_Fast`, `PyLong_AsLong`) and never calls `to_dict()`.
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

* Feature extraction only; the engine, heuristics and search are still
  Python (they can reuse `state.hpp` and `board_tables.hpp`).
* At most 4 players (the feature layout pads to 3 opponents anyway).
  Malformed states (resource lists that are not length 5, ids out of
  range, > 64 roads for one player) raise `ValueError` instead of the
  assorted Python errors.  Port types outside `0..5` are ignored.
* `-march=native` binaries are not portable between CPUs; rebuild on each
  machine (or use `CATANBOT_CPP_PORTABLE=1`).
* The extension must be rebuilt after changing `board.py` or the feature
  layout in `features.py` (`scripts/build_cpp.sh`; the tests and the
  run-time check in `accel.py` catch a stale build).
