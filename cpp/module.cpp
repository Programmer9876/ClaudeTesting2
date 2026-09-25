// pybind11 module `catanbot_core`: C++ feature extraction for catanbot.
//
//   extract_batch(states, players) -> np.ndarray float32 (N, NUM_FEATURES)
//   extract(state, player)         -> np.ndarray float32 (NUM_FEATURES,)
//   longest_road_length(state, player) -> int
//   num_features() / feature_names() / phase_names()
//   static_values(state) -> list[float];  static_value(state, player) -> float
//   heuristic_evaluate(states, players, temperature=16.0) -> np.ndarray float64 (N,)
//   (+ the heuristic helpers heuristic_longest_road_length, reachable_spots,
//    score_settlement_spot, player_production, resource_scarcity,
//    expected_hidden_vp, progress_to_build, mainly for the differential tests)
//   HeuristicEval / MlpEval / BlendEval, future_values, reduced_search, choose_discard,
//   best_robber_move, win_path, robber_weights: the native lookahead (search_bindings.inc)
//
// `states` are catanbot.state.GameState objects; they are converted once per
// distinct consecutive object (like features.extract_batch) with the raw
// CPython API (see state.hpp).  The GIL is held throughout (the work per state
// is a few microseconds, releasing it would cost more than it saves).
//
// A state the C++ structs cannot represent (> 4 players, lists of the wrong
// length, ids / integers out of range, > 64 road entries) raises
// `UnsupportedStateError` (a ValueError subclass, from catanbot::unsupported_state);
// catanbot.accel catches it and falls back to the Python reference.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <random>

#include "engine.hpp"
#include "evaluator.hpp"
#include "features.hpp"
#include "heuristic.hpp"
#include "policy.hpp"
#include "search.hpp"
#include "state.hpp"

namespace py = pybind11;
using namespace catanbot;

namespace {

// A GameState converted once and kept in C++ (see the engine section): the search can apply
// actions and evaluate positions on it without converting a Python GameState each time.
// `origin` is the GameState it came from (hexes, ports and the players' colour / name are taken
// from it when the handle is materialised with to_state()).
struct CState {
    GameStateC s;
    py::object origin;
    long rolls_history_len = 0;
};

// The GameStateC behind a Python argument: a CState's own struct, or `buf` filled from a GameState.
const GameStateC* resolve(PyObject* obj, GameStateC& buf) {
    if (py::isinstance<CState>(py::handle(obj))) return &py::cast<const CState&>(py::handle(obj)).s;
    state_from_python(obj, buf);
    return &buf;
}

long index_of(PyObject* o) {
    long v = PyLong_AsLong(o);
    if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        PyObject* i = PyNumber_Index(o);
        if (!i) throw py::error_already_set();
        v = PyLong_AsLong(i);
        Py_DECREF(i);
        if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    }
    return v;
}

void check_player(const GameStateC& st, long player) {
    if (player < 0 || player >= st.num_players)
        throw py::index_error("player " + std::to_string(player) + " out of range for a " +
                              std::to_string(st.num_players) + "-player state");
}

py::array_t<float> extract_batch(py::handle states, py::handle players) {
    PyObject* sf = PySequence_Fast(states.ptr(), "states must be a sequence");
    if (!sf) throw py::error_already_set();
    py::object sf_keep = py::reinterpret_steal<py::object>(sf);
    PyObject* pf = PySequence_Fast(players.ptr(), "players must be a sequence");
    if (!pf) throw py::error_already_set();
    py::object pf_keep = py::reinterpret_steal<py::object>(pf);
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(sf);
    if (n != PySequence_Fast_GET_SIZE(pf)) throw py::value_error("states and players must have the same length");
    py::array_t<float> X({(py::ssize_t)n, (py::ssize_t)NUM_FEATURES});
    float* out = X.mutable_data();
    if (n == 0) return X;
    PyObject** sitems = PySequence_Fast_ITEMS(sf);
    PyObject** pitems = PySequence_Fast_ITEMS(pf);
    // Analysis is ~9 KB, GameStateC ~1.3 KB: heap-allocate once per call to keep the stack small.
    std::unique_ptr<GameStateC> st(new GameStateC());
    std::unique_ptr<Analysis> an(new Analysis());
    PyObject* last = nullptr;
    const GameStateC* cur = st.get();
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = sitems[i];
        if (s != last) {  // consecutive identical state objects share one analysis
            cur = resolve(s, *st);
            analyse(*cur, *an);
            last = s;
        }
        const long player = index_of(pitems[i]);
        check_player(*cur, player);
        assemble(*cur, *an, (int)player, out + i * NUM_FEATURES);
    }
    return X;
}

py::array_t<float> extract(py::handle state, long player) {
    std::unique_ptr<GameStateC> st(new GameStateC());
    std::unique_ptr<Analysis> an(new Analysis());
    const GameStateC* cur = resolve(state.ptr(), *st);
    check_player(*cur, player);
    analyse(*cur, *an);
    py::array_t<float> x({(py::ssize_t)NUM_FEATURES});
    assemble(*cur, *an, (int)player, x.mutable_data());
    return x;
}

int longest_road(py::handle state, long player) {
    std::unique_ptr<GameStateC> st(new GameStateC());
    state_from_python(state.ptr(), *st);
    check_player(*st, player);
    return longest_road_length(*st, (int)player);
}

// ---------------------------------------------------------------------------
// heuristic (catanbot.heuristic.static_value / HeuristicEvaluator.evaluate)
// ---------------------------------------------------------------------------
py::list static_values_py(py::handle state) {
    GameStateC buf;
    const GameStateC& st = *resolve(state.ptr(), buf);
    double vals[MAX_PLAYERS];
    static_values(st, vals);
    py::list out;
    for (int i = 0; i < st.num_players; ++i) out.append(vals[i]);
    return out;
}

double static_value_py(py::handle state, long player) {
    GameStateC buf;
    const GameStateC& st = *resolve(state.ptr(), buf);
    check_player(st, player);
    return static_value(st, (int)player);
}

py::array_t<double> heuristic_evaluate_py(py::handle states, py::handle players, double temperature) {
    PyObject* sf = PySequence_Fast(states.ptr(), "states must be a sequence");
    if (!sf) throw py::error_already_set();
    py::object sf_keep = py::reinterpret_steal<py::object>(sf);
    PyObject* pf = PySequence_Fast(players.ptr(), "players must be a sequence");
    if (!pf) throw py::error_already_set();
    py::object pf_keep = py::reinterpret_steal<py::object>(pf);
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(sf);
    if (n != PySequence_Fast_GET_SIZE(pf)) throw py::value_error("states and players must have the same length");
    py::array_t<double> out({(py::ssize_t)n});
    double* o = out.mutable_data();
    if (n == 0) return out;
    PyObject** sitems = PySequence_Fast_ITEMS(sf);
    PyObject** pitems = PySequence_Fast_ITEMS(pf);
    // Like HeuristicEvaluator.evaluate (cache keyed by id(state)), every distinct state object
    // is converted and scored once per call, wherever it appears in the batch.
    struct Entry {
        double vals[MAX_PLAYERS];
        int num_players;
        bool decided;  // game over with a winner: the softmax is overridden by 1 / 0
        int winner;
    };
    std::vector<Entry> entries;
    entries.reserve((size_t)n);
    std::unordered_map<PyObject*, size_t> cache;
    cache.reserve((size_t)n);
    GameStateC st;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = sitems[i];
        size_t idx;
        auto it = cache.find(s);
        if (it == cache.end()) {
            const GameStateC& cur = *resolve(s, st);
            Entry e;
            static_values(cur, e.vals);
            e.num_players = cur.num_players;
            e.decided = cur.phase == PHASE_GAME_OVER && cur.winner >= 0;
            e.winner = cur.winner;
            idx = entries.size();
            entries.push_back(e);
            cache.emplace(s, idx);
        } else {
            idx = it->second;
        }
        const Entry& e = entries[idx];
        const long player = index_of(pitems[i]);
        if (player < 0 || player >= e.num_players)
            throw py::index_error("player " + std::to_string(player) + " out of range for a " +
                                  std::to_string(e.num_players) + "-player state");
        o[i] = e.decided ? (e.winner == player ? 1.0 : 0.0)
                         : heuristic_softmax(e.vals, e.num_players, (int)player, temperature);
    }
    return out;
}

int heuristic_longest_road_py(py::handle state, long player) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    check_player(st, player);
    return heuristic_longest_road_length(st, (int)player);
}

py::dict reachable_spots_py(py::handle state, long player, int max_roads) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    check_player(st, player);
    Occupancy occ;
    occupancy(st, occ);
    ReachableSpots r;
    reachable_spots(st, (int)player, max_roads, occ, r);
    py::dict out;
    for (int v = 0; v < NUM_VERTICES; ++v)
        if (r.dist[v] >= 0) out[py::int_(v)] = py::make_tuple((int)r.dist[v], (int)r.first[v]);
    return out;
}

double score_settlement_spot_py(py::handle state, long player, int vertex, bool setup) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    check_player(st, player);
    if (vertex < 0 || vertex >= NUM_VERTICES) throw py::index_error("vertex " + std::to_string(vertex) + " out of range");
    Occupancy occ;
    occupancy(st, occ);
    double own[NUM_RESOURCES], sc[NUM_RESOURCES];
    player_production(st, (int)player, true, own);
    resource_scarcity(st, sc);
    return score_settlement_spot(st, (int)player, vertex, occ, own, sc, setup);
}

py::list player_production_py(py::handle state, long player, bool ignore_robber) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    check_player(st, player);
    double prod[NUM_RESOURCES];
    player_production(st, (int)player, ignore_robber, prod);
    py::list out;
    for (int r = 0; r < NUM_RESOURCES; ++r) out.append(prod[r]);
    return out;
}

py::list resource_scarcity_py(py::handle state) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    double sc[NUM_RESOURCES];
    resource_scarcity(st, sc);
    py::list out;
    for (int r = 0; r < NUM_RESOURCES; ++r) out.append(sc[r]);
    return out;
}

double expected_hidden_vp_py(py::handle state, long player) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    check_player(st, player);
    return expected_hidden_vp(st, (int)player);
}

double progress_to_build_py(py::handle state, long player) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    check_player(st, player);
    Occupancy occ;
    occupancy(st, occ);
    return progress_to_build(st, (int)player, occ);
}

// ---------------------------------------------------------------------------
// engine (catanbot.engine.legal_actions / apply / apply_inplace / random_playout)
// ---------------------------------------------------------------------------
// Action tuples are converted to ActionC (engine.hpp) and back; the tuples that do not
// depend on the state are built once and cached like engine.py's module tables.  A new
// GameState is produced with `object.__new__(GameState)` + attribute stores (like
// GameState.copy()); apply_inplace writes the result back into the existing objects
// (same GameState, Player, TradeOffer and list objects) so callers holding references
// see the update exactly as with the Python engine.
namespace engine_py {

using namespace catanbot::detail;

struct EngineNames {
    PyObject* kind[NUM_ACTION_KINDS];
    PyObject *color, *name, *rolls_history_len, *randrange, *randint;
    PyObject* empty_tuple;
};

const EngineNames& enames() {
    static const EngineNames* n = [] {
        EngineNames* m = new EngineNames();
        for (int k = 0; k < NUM_ACTION_KINDS; ++k) m->kind[k] = intern(ACTION_KIND_NAMES[k]);
        m->color = intern("color");
        m->name = intern("name");
        m->rolls_history_len = intern("rolls_history_len");
        m->randrange = intern("randrange");
        m->randint = intern("randint");
        m->empty_tuple = PyTuple_New(0);
        if (!m->empty_tuple) throw py::error_already_set();
        return m;
    }();
    return *n;
}

PyObject* check(PyObject* o) {
    if (!o) throw py::error_already_set();
    return o;
}

inline PyObject* incref(PyObject* o) {
    Py_INCREF(o);
    return o;
}

// --- the IllegalActionError class ------------------------------------------------------
// catanbot.engine.IllegalActionError itself (looked up lazily, after the package is imported),
// so callers catching the Python engine's exception catch ours as well; the module's own
// IllegalActionError (a ValueError) is used when catanbot.engine cannot be imported.
PyObject* fallback_illegal_type = nullptr;

PyObject* illegal_action_type() {
    static PyObject* cls = [] {
        PyObject* mod = PyImport_ImportModule("catanbot.engine");
        if (mod) {
            PyObject* c = PyObject_GetAttrString(mod, "IllegalActionError");
            Py_DECREF(mod);
            if (c && PyExceptionClass_Check(c)) return c;
            Py_XDECREF(c);
        }
        PyErr_Clear();
        Py_INCREF(fallback_illegal_type);
        return fallback_illegal_type;
    }();
    return cls;
}

[[noreturn]] void throw_illegal(const std::string& msg) {
    PyErr_SetString(illegal_action_type(), msg.c_str());
    throw py::error_already_set();
}

// --- action tuple cache ------------------------------------------------------------------
PyObject* tuple1(PyObject* kind) { return check(Py_BuildValue("(O)", kind)); }
PyObject* tuple2(PyObject* kind, long a) { return check(Py_BuildValue("(Ol)", kind, a)); }
PyObject* tuple3(PyObject* kind, long a, long b) { return check(Py_BuildValue("(Oll)", kind, a, b)); }
PyObject* vec_tuple(const int32_t* v) {
    return check(Py_BuildValue("(lllll)", (long)v[0], (long)v[1], (long)v[2], (long)v[3], (long)v[4]));
}

struct ActionTuples {
    PyObject* nullary[NUM_ACTION_KINDS] = {};
    PyObject* setup_settlement[NUM_VERTICES];
    PyObject* setup_road[NUM_EDGES];
    PyObject* build_road[NUM_EDGES];
    PyObject* build_settlement[NUM_VERTICES];
    PyObject* build_city[NUM_VERTICES];
    PyObject* execute_trade[MAX_PLAYERS];
    PyObject* monopoly[NUM_RESOURCES];
    PyObject* yop[NUM_RESOURCES][NUM_RESOURCES];
    PyObject* bank_trade[NUM_RESOURCES][NUM_RESOURCES];
    PyObject* move_robber[NUM_HEXES][MAX_PLAYERS + 1];  // [hex][victim + 1]
    PyObject* play_knight[NUM_HEXES][MAX_PLAYERS + 1];
    PyObject* propose[NUM_RESOURCES][2][NUM_RESOURCES];  // [give][amount - 1][get]
};

const ActionTuples& tuples() {
    static const ActionTuples* t = [] {
        const EngineNames& N = enames();
        ActionTuples* T = new ActionTuples();
        for (int k : {ACT_ROLL, ACT_BUY_DEV, ACT_PLAY_ROAD_BUILDING, ACT_ACCEPT_TRADE, ACT_REJECT_TRADE,
                      ACT_CANCEL_TRADE, ACT_END_TURN})
            T->nullary[k] = tuple1(N.kind[k]);
        for (int v = 0; v < NUM_VERTICES; ++v) {
            T->setup_settlement[v] = tuple2(N.kind[ACT_SETUP_SETTLEMENT], v);
            T->build_settlement[v] = tuple2(N.kind[ACT_BUILD_SETTLEMENT], v);
            T->build_city[v] = tuple2(N.kind[ACT_BUILD_CITY], v);
        }
        for (int e = 0; e < NUM_EDGES; ++e) {
            T->setup_road[e] = tuple2(N.kind[ACT_SETUP_ROAD], e);
            T->build_road[e] = tuple2(N.kind[ACT_BUILD_ROAD], e);
        }
        for (int i = 0; i < MAX_PLAYERS; ++i) T->execute_trade[i] = tuple2(N.kind[ACT_EXECUTE_TRADE], i);
        for (int r = 0; r < NUM_RESOURCES; ++r) T->monopoly[r] = tuple2(N.kind[ACT_PLAY_MONOPOLY], r);
        for (int r1 = 0; r1 < NUM_RESOURCES; ++r1)
            for (int r2 = 0; r2 < NUM_RESOURCES; ++r2) {
                T->yop[r1][r2] = tuple3(N.kind[ACT_PLAY_YEAR_OF_PLENTY], r1, r2);
                T->bank_trade[r1][r2] = tuple3(N.kind[ACT_BANK_TRADE], r1, r2);
            }
        for (int h = 0; h < NUM_HEXES; ++h)
            for (int i = 0; i <= MAX_PLAYERS; ++i) {
                T->move_robber[h][i] = tuple3(N.kind[ACT_MOVE_ROBBER], h, i - 1);
                T->play_knight[h][i] = tuple3(N.kind[ACT_PLAY_KNIGHT], h, i - 1);
            }
        for (int g = 0; g < NUM_RESOURCES; ++g)
            for (int amount = 1; amount <= 2; ++amount)
                for (int get = 0; get < NUM_RESOURCES; ++get) {
                    int32_t give[NUM_RESOURCES] = {0, 0, 0, 0, 0}, want[NUM_RESOURCES] = {0, 0, 0, 0, 0};
                    give[g] = amount;
                    want[get] = 1;
                    Ref gv(vec_tuple(give));
                    Ref wv(vec_tuple(want));
                    T->propose[g][amount - 1][get] = check(Py_BuildValue("(OOO)", N.kind[ACT_PROPOSE_TRADE], gv.p, wv.p));
                }
        return T;
    }();
    return *t;
}

// The single non-zero entry of a 5-vector (index) when it equals `amount`, else -1.
int single_entry(const int32_t* v, int amount) {
    int idx = -1;
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        if (v[r] == 0) continue;
        if (v[r] != amount || idx >= 0) return -1;
        idx = r;
    }
    return idx;
}

// ActionC -> Python tuple (new reference), cached tuples where possible.
PyObject* action_to_python(const ActionC& a) {
    const ActionTuples& T = tuples();
    const EngineNames& N = enames();
    const int k = a.kind;
    if (k < 0 || k >= NUM_ACTION_KINDS) throw py::value_error("bad action kind");
    PyObject* kind = N.kind[k];
    switch (k) {
        case ACT_ROLL:
        case ACT_BUY_DEV:
        case ACT_PLAY_ROAD_BUILDING:
        case ACT_ACCEPT_TRADE:
        case ACT_REJECT_TRADE:
        case ACT_CANCEL_TRADE:
        case ACT_END_TURN:
            if (a.nargs == 0) return incref(T.nullary[k]);
            break;
        case ACT_SETUP_SETTLEMENT:
            if (a.nargs == 1 && a.a >= 0 && a.a < NUM_VERTICES) return incref(T.setup_settlement[a.a]);
            break;
        case ACT_BUILD_SETTLEMENT:
            if (a.nargs == 1 && a.a >= 0 && a.a < NUM_VERTICES) return incref(T.build_settlement[a.a]);
            break;
        case ACT_BUILD_CITY:
            if (a.nargs == 1 && a.a >= 0 && a.a < NUM_VERTICES) return incref(T.build_city[a.a]);
            break;
        case ACT_SETUP_ROAD:
            if (a.nargs == 1 && a.a >= 0 && a.a < NUM_EDGES) return incref(T.setup_road[a.a]);
            break;
        case ACT_BUILD_ROAD:
            if (a.nargs == 1 && a.a >= 0 && a.a < NUM_EDGES) return incref(T.build_road[a.a]);
            break;
        case ACT_EXECUTE_TRADE:
            if (a.nargs == 1 && a.a >= 0 && a.a < MAX_PLAYERS) return incref(T.execute_trade[a.a]);
            break;
        case ACT_PLAY_MONOPOLY:
            if (a.nargs == 1 && a.a >= 0 && a.a < NUM_RESOURCES) return incref(T.monopoly[a.a]);
            break;
        case ACT_PLAY_YEAR_OF_PLENTY:
            if (a.nargs == 2 && a.a >= 0 && a.a < NUM_RESOURCES && a.b >= 0 && a.b < NUM_RESOURCES)
                return incref(T.yop[a.a][a.b]);
            break;
        case ACT_BANK_TRADE:
            if (a.nargs == 2 && a.a >= 0 && a.a < NUM_RESOURCES && a.b >= 0 && a.b < NUM_RESOURCES)
                return incref(T.bank_trade[a.a][a.b]);
            break;
        case ACT_MOVE_ROBBER:
            if (a.nargs == 2 && a.a >= 0 && a.a < NUM_HEXES && a.b >= -1 && a.b < MAX_PLAYERS)
                return incref(T.move_robber[a.a][a.b + 1]);
            break;
        case ACT_PLAY_KNIGHT:
            if (a.nargs == 2 && a.a >= 0 && a.a < NUM_HEXES && a.b >= -1 && a.b < MAX_PLAYERS)
                return incref(T.play_knight[a.a][a.b + 1]);
            break;
        case ACT_DISCARD: {
            Ref v(vec_tuple(a.vec1));
            return check(Py_BuildValue("(OO)", kind, v.p));
        }
        case ACT_PROPOSE_TRADE: {
            const int get = single_entry(a.vec2, 1);
            if (get >= 0) {
                for (int amount = 1; amount <= 2; ++amount) {
                    const int give = single_entry(a.vec1, amount);
                    if (give >= 0) return incref(T.propose[give][amount - 1][get]);
                }
            }
            Ref gv(vec_tuple(a.vec1));
            Ref wv(vec_tuple(a.vec2));
            return check(Py_BuildValue("(OOO)", kind, gv.p, wv.p));
        }
        default:
            break;
    }
    // generic (kind, a[, b]) - e.g. a recorded (ROLL, value)
    if (a.nargs <= 0) return tuple1(kind);
    if (a.nargs == 1) return tuple2(kind, a.a);
    return tuple3(kind, a.a, a.b);
}

// --- Python action -> ActionC -----------------------------------------------------------------
int32_t int_arg(PyObject* o) {
    // Integers outside the 32-bit range are clamped: they are rejected by the handlers'
    // range checks with the same message Python gives ("bad vertex", ...).
    int overflow = 0;
    long v = PyLong_AsLongAndOverflow(o, &overflow);
    if (v == -1 && !overflow && PyErr_Occurred()) {
        PyErr_Clear();
        PyObject* i = PyNumber_Index(o);  // TypeError for non-integers, like Python's comparisons
        if (!i) throw py::error_already_set();
        v = PyLong_AsLongAndOverflow(i, &overflow);
        Py_DECREF(i);
        if (v == -1 && !overflow && PyErr_Occurred()) throw py::error_already_set();
    }
    if (overflow) return overflow > 0 ? INT32_MAX : INT32_MIN;
    if (v > INT32_MAX) return INT32_MAX;
    if (v < INT32_MIN) return INT32_MIN;
    return (int32_t)v;
}

void vec_arg(PyObject* o, int32_t* out, int8_t* nvec) {
    Ref fast(PySequence_Fast(o, "trade / discard counts must be a sequence of 5 ints"));
    if (!fast.p) throw py::error_already_set();
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
    *nvec = (int8_t)(n > 127 ? 127 : n);
    PyObject** items = PySequence_Fast_ITEMS(fast.p);
    for (Py_ssize_t i = 0; i < n && i < NUM_RESOURCES; ++i) out[i] = int_arg(items[i]);
}

int kind_index(PyObject* o) {
    const EngineNames& N = enames();
    for (int k = 0; k < NUM_ACTION_KINDS; ++k)
        if (o == N.kind[k]) return k;
    if (!PyUnicode_Check(o)) return ACT_UNKNOWN;
    for (int k = 0; k < NUM_ACTION_KINDS; ++k)
        if (PyUnicode_CompareWithASCIIString(o, ACTION_KIND_NAMES[k]) == 0) return k;
    return ACT_UNKNOWN;
}

// Parses (kind, *args).  Anything that is not a sequence starting with a known kind string gets
// kind = ACT_UNKNOWN (-> "unknown action <repr>", as in Python); the argument *count* is kept as
// given and validated by the handlers.
void parse_action(PyObject* obj, ActionC& a) {
    a = ActionC();
    PyObject* fast = PySequence_Fast(obj, "");
    if (!fast) {
        PyErr_Clear();
        return;
    }
    Ref keep(fast);
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n == 0) return;
    PyObject** items = PySequence_Fast_ITEMS(fast);
    const int k = kind_index(items[0]);
    if (k < 0) return;
    a.kind = (int8_t)k;
    a.nargs = (int8_t)(n - 1 > 127 ? 127 : n - 1);
    switch (k) {
        case ACT_DISCARD:
            if (n >= 2) vec_arg(items[1], a.vec1, &a.nvec1);
            break;
        case ACT_PROPOSE_TRADE:
            if (n >= 2) vec_arg(items[1], a.vec1, &a.nvec1);
            if (n >= 3) vec_arg(items[2], a.vec2, &a.nvec2);
            break;
        default:
            if (n >= 2) a.a = int_arg(items[1]);
            if (n >= 3) a.b = int_arg(items[2]);
            break;
    }
}

// --- GameStateC -> Python ------------------------------------------------------------------------
struct StateClasses {
    PyObject *game_state, *player, *trade_offer;
};

const StateClasses& classes() {
    static const StateClasses* c = [] {
        StateClasses* k = new StateClasses();
        PyObject* mod = check(PyImport_ImportModule("catanbot.state"));
        k->game_state = check(PyObject_GetAttrString(mod, "GameState"));
        k->player = check(PyObject_GetAttrString(mod, "Player"));
        k->trade_offer = check(PyObject_GetAttrString(mod, "TradeOffer"));
        Py_DECREF(mod);
        return k;
    }();
    return *c;
}

// object.__new__(cls): a bare instance, like GameState.copy() does.
PyObject* new_instance(PyObject* cls) {
    return check(PyBaseObject_Type.tp_new((PyTypeObject*)cls, enames().empty_tuple, nullptr));
}

// Attribute stores / loads on one object.  Plain PyObject_SetAttr on purpose: storing straight
// into a materialised instance __dict__ is no faster and leaves objects that CPython 3.11 can no
// longer keep in its compact inline-attribute layout (their later copy() / reads get slower).
struct Writer {
    PyObject* obj;
    explicit Writer(PyObject* o) : obj(o) {}
    // setattr(obj, name, value), consuming the new reference `value`.
    void set(PyObject* name, PyObject* value) {
        check(value);
        const int r = PyObject_SetAttr(obj, name, value);
        Py_DECREF(value);
        if (r < 0) throw py::error_already_set();
    }
    void set_int(PyObject* name, long v) { set(name, PyLong_FromLong(v)); }
    void set_bool(PyObject* name, bool v) { set(name, incref(v ? Py_True : Py_False)); }
    // getattr(obj, name) as a new reference, nullptr (no exception pending) when absent.
    PyObject* get(PyObject* name) const {
        PyObject* v = PyObject_GetAttr(obj, name);
        if (!v) PyErr_Clear();
        return v;
    }
};

PyObject* int_list(const int32_t* v, int n) {
    PyObject* l = check(PyList_New(n));
    for (int i = 0; i < n; ++i) PyList_SET_ITEM(l, i, check(PyLong_FromLong(v[i])));
    return l;
}

PyObject* id_list(const uint8_t* v, int n) {
    PyObject* l = check(PyList_New(n));
    for (int i = 0; i < n; ++i) PyList_SET_ITEM(l, i, check(PyLong_FromLong(v[i])));
    return l;
}

PyObject* responses_dict(const TradeC& t) {
    PyObject* d = check(PyDict_New());
    for (int i = 0; i < MAX_PLAYERS; ++i) {
        if (t.responses[i] < 0) continue;
        Ref k(PyLong_FromLong(i));
        if (PyDict_SetItem(d, k.p, t.responses[i] ? Py_True : Py_False) < 0) {
            Py_DECREF(d);
            throw py::error_already_set();
        }
    }
    return d;
}

PyObject* make_trade(const TradeC& t) {
    const Names& N = names();
    const StateClasses& C = classes();
    PyObject* o = new_instance(C.trade_offer);
    Ref keep(o);
    Writer w(o);
    w.set_int(N.proposer, t.proposer);
    w.set(N.give, int_list(t.give, NUM_RESOURCES));
    w.set(N.get, int_list(t.get, NUM_RESOURCES));
    w.set(N.responses, responses_dict(t));
    return incref(o);
}

// A new Player object; colour / name are copied from `src` (the Python Player it came from).
PyObject* make_player(const PlayerC& p, PyObject* src) {
    const Names& N = names();
    const EngineNames& E = enames();
    const StateClasses& C = classes();
    PyObject* o = new_instance(C.player);
    Ref keep(o);
    Writer w(o);
    if (src) {
        w.set(E.color, getattr(src, E.color));
        w.set(E.name, getattr(src, E.name));
    } else {
        w.set(E.color, PyUnicode_FromString("red"));
        w.set(E.name, PyUnicode_FromString(""));
    }
    w.set(N.resources, int_list(p.resources, NUM_RESOURCES));
    w.set(N.dev_cards, int_list(p.dev_cards, NUM_DEV));
    w.set(N.dev_cards_new, int_list(p.dev_cards_new, NUM_DEV));
    w.set_int(N.played_knights, p.played_knights);
    w.set(N.settlements, id_list(p.settlements, p.n_settlements));
    w.set(N.cities, id_list(p.cities, p.n_cities));
    w.set(N.roads, id_list(p.roads, p.n_roads));
    w.set_bool(N.hand_known, p.hand_known);
    w.set_int(N.hand_size, p.hand_size);
    w.set_bool(N.dev_known, p.dev_known);
    w.set_int(N.dev_count, p.dev_count);
    return incref(o);
}

// The Python Player objects of `src` (borrowed, nullptr when the list is shorter / not a list).
struct SrcPlayers {
    Ref fast;
    Py_ssize_t n = 0;
    PyObject** items = nullptr;
    explicit SrcPlayers(PyObject* src) : fast(nullptr) {
        Ref pl(PyObject_GetAttr(src, names().players));
        if (!pl.p) {
            PyErr_Clear();
            return;
        }
        fast.p = PySequence_Fast(pl.p, "");
        if (!fast.p) {
            PyErr_Clear();
            return;
        }
        n = PySequence_Fast_GET_SIZE(fast.p);
        items = PySequence_Fast_ITEMS(fast.p);
    }
    PyObject* at(int i) const { return i < n ? items[i] : nullptr; }
};

void store_scalars(Writer& w, const GameStateC& s) {
    const Names& N = names();
    w.set_int(N.robber, s.robber);
    w.set_int(N.current, s.current);
    if (s.phase >= 0 && s.phase < NUM_PHASES) w.set(N.phase, incref(N.phase_str[s.phase]));
    w.set_int(N.turn, s.turn);
    w.set_int(N.setup_round, s.setup_round);
    w.set_int(N.setup_last_settlement, s.setup_last_settlement);
    w.set_int(N.dice, s.dice);
    w.set_bool(N.dev_played_this_turn, s.dev_played_this_turn);
    w.set_int(N.free_roads, s.free_roads);
    w.set_int(N.trade_responder, s.trade_responder);
    w.set_int(N.trades_this_turn, s.trades_this_turn);
    w.set_int(N.longest_road_owner, s.longest_road_owner);
    w.set_int(N.longest_road_len, s.longest_road_len);
    w.set_int(N.largest_army_owner, s.largest_army_owner);
    w.set_int(N.winner, s.winner);
    w.set_int(N.max_turns, s.max_turns);
}

// New GameState from `s`; hexes / ports (immutable during a game) and the players' colour /
// name come from `src`, exactly like GameState.copy().
PyObject* state_to_python_new(const GameStateC& s, PyObject* src, long rolls_history_len) {
    const Names& N = names();
    const EngineNames& E = enames();
    const StateClasses& C = classes();
    PyObject* o = new_instance(C.game_state);
    Ref keep(o);
    Writer w(o);
    w.set(N.hexes, getattr(src, N.hexes));
    w.set(N.ports, getattr(src, N.ports));
    if (s.phase < 0 || s.phase >= NUM_PHASES) w.set(N.phase, getattr(src, N.phase));
    store_scalars(w, s);
    {
        SrcPlayers sp(src);
        PyObject* pl = check(PyList_New(s.num_players));
        Ref keep_pl(pl);
        for (int i = 0; i < s.num_players; ++i) PyList_SET_ITEM(pl, i, make_player(s.players[i], sp.at(i)));
        w.set(N.players, incref(pl));
    }
    w.set(N.bank, int_list(s.bank, NUM_RESOURCES));
    w.set(N.dev_deck, int_list(s.dev_deck, NUM_DEV));
    w.set(N.discard_queue, int_list(s.discard_queue, s.n_discard));
    w.set(N.pending_trade, s.has_pending_trade ? make_trade(s.pending_trade) : incref(Py_None));
    w.set_int(E.rolls_history_len, rolls_history_len);
    return incref(o);
}

// --- in-place write-back (apply_inplace) ------------------------------------------------------
// Existing list objects are updated in place (item stores / slice assignment) so references held
// by the caller stay valid, exactly as with the Python engine's in-place mutation.
void store_ints_inplace(Writer& w, PyObject* name, const int32_t* v, int n) {
    Ref cur(w.get(name));
    if (cur.p && PyList_CheckExact(cur.p) && PyList_GET_SIZE(cur.p) == n) {
        for (int i = 0; i < n; ++i)
            if (PyList_SetItem(cur.p, i, check(PyLong_FromLong(v[i]))) < 0) throw py::error_already_set();
        return;
    }
    w.set(name, int_list(v, n));
}

void store_ids_inplace(Writer& w, PyObject* name, const uint8_t* ids, int n) {
    Ref cur(w.get(name));
    Ref nl(id_list(ids, n));
    if (cur.p && PyList_CheckExact(cur.p)) {
        if (PyList_SetSlice(cur.p, 0, PyList_GET_SIZE(cur.p), nl.p) < 0) throw py::error_already_set();
        return;
    }
    w.set(name, incref(nl.p));
}

void store_player_inplace(PyObject* o, const PlayerC& p) {
    const Names& N = names();
    Writer w(o);
    store_ints_inplace(w, N.resources, p.resources, NUM_RESOURCES);
    store_ints_inplace(w, N.dev_cards, p.dev_cards, NUM_DEV);
    store_ints_inplace(w, N.dev_cards_new, p.dev_cards_new, NUM_DEV);
    w.set_int(N.played_knights, p.played_knights);
    store_ids_inplace(w, N.settlements, p.settlements, p.n_settlements);
    store_ids_inplace(w, N.cities, p.cities, p.n_cities);
    store_ids_inplace(w, N.roads, p.roads, p.n_roads);
    w.set_int(N.hand_size, p.hand_size);
    w.set_int(N.dev_count, p.dev_count);
}

void store_trade_inplace(Writer& w, const GameStateC& s) {
    const Names& N = names();
    if (!s.has_pending_trade) {
        w.set(N.pending_trade, incref(Py_None));
        return;
    }
    const TradeC& t = s.pending_trade;
    Ref cur(w.get(N.pending_trade));
    if (!cur.p || cur.p == Py_None || t.proposer != get_int32(cur.p, N.proposer)) {
        // a new offer (PROPOSE_TRADE creates a new TradeOffer in Python as well)
        w.set(N.pending_trade, make_trade(t));
        return;
    }
    Writer tw(cur.p);
    store_ints_inplace(tw, N.give, t.give, NUM_RESOURCES);
    store_ints_inplace(tw, N.get, t.get, NUM_RESOURCES);
    Ref resp(tw.get(N.responses));
    if (resp.p && PyDict_CheckExact(resp.p)) {
        Ref fresh(responses_dict(t));
        PyDict_Clear(resp.p);
        if (PyDict_Update(resp.p, fresh.p) < 0) throw py::error_already_set();
    } else {
        tw.set(N.responses, responses_dict(t));
    }
}

long long_of(PyObject* v_new_ref, long dflt) {  // consumes the reference
    if (!v_new_ref) return dflt;
    Ref keep(v_new_ref);
    const long v = PyLong_AsLong(v_new_ref);
    if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return dflt;
    }
    return v;
}

void state_to_python_inplace(const GameStateC& s, PyObject* o, bool rolled) {
    const Names& N = names();
    const EngineNames& E = enames();
    Writer w(o);
    store_scalars(w, s);
    {
        SrcPlayers sp(o);
        if (sp.n == s.num_players) {
            for (int i = 0; i < s.num_players; ++i) store_player_inplace(sp.items[i], s.players[i]);
        } else {
            PyObject* pl = check(PyList_New(s.num_players));
            Ref keep_pl(pl);
            for (int i = 0; i < s.num_players; ++i) PyList_SET_ITEM(pl, i, make_player(s.players[i], sp.at(i)));
            w.set(N.players, incref(pl));
        }
    }
    store_ints_inplace(w, N.bank, s.bank, NUM_RESOURCES);
    store_ints_inplace(w, N.dev_deck, s.dev_deck, NUM_DEV);
    store_ints_inplace(w, N.discard_queue, s.discard_queue, s.n_discard);
    store_trade_inplace(w, s);
    if (rolled) w.set_int(E.rolls_history_len, long_of(w.get(E.rolls_history_len), 0) + 1);
}

long rolls_history_len_of(PyObject* src) {
    PyObject* r = PyObject_GetAttr(src, enames().rolls_history_len);
    if (!r) PyErr_Clear();
    return long_of(r, 0);
}

// --- random sources -----------------------------------------------------------------------------
// A Python random.Random (or anything with randrange / randint): called exactly like the Python
// engine calls it, so the same rng object gives the same game in both engines.
struct PyRngDraw : DrawSource {
    PyObject* rng;
    explicit PyRngDraw(PyObject* r) : rng(r) {}
    static int as_int(PyObject* r, int lo, int hi) {
        Ref keep(r);
        const long v = PyLong_AsLong(r);
        if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
        if (v < lo || v > hi)
            throw py::value_error("rng returned " + std::to_string(v) + ", outside [" + std::to_string(lo) + ", " +
                                  std::to_string(hi) + "]");
        return (int)v;
    }
    int randrange(int n) override {
        Ref arg(check(PyLong_FromLong(n)));
        return as_int(check(PyObject_CallMethodObjArgs(rng, enames().randrange, arg.p, nullptr)), 0, n - 1);
    }
    int randint(int lo, int hi) override {
        Ref a(check(PyLong_FromLong(lo)));
        Ref b(check(PyLong_FromLong(hi)));
        return as_int(check(PyObject_CallMethodObjArgs(rng, enames().randint, a.p, b.p, nullptr)), lo, hi);
    }
};

uint64_t fresh_seed() {
    static std::random_device rd;
    static std::mt19937_64 gen((uint64_t)rd() << 32 ^ rd());
    return gen();
}

uint64_t seed_of(py::handle seed) {
    if (seed.is_none()) return fresh_seed();
    if (!PyLong_Check(seed.ptr())) throw py::type_error("seed must be an int or None");
    const unsigned long long v = PyLong_AsUnsignedLongLongMask(seed.ptr());
    if (v == (unsigned long long)-1 && PyErr_Occurred()) throw py::error_already_set();
    return (uint64_t)v;
}

// rng argument of apply / apply_inplace: None (fresh C++ rng), an int seed, or a random.Random.
struct DrawHolder {
    Xoshiro xo{0};
    PyRngDraw* py = nullptr;
    DrawSource* src = nullptr;
    explicit DrawHolder(py::handle rng) {
        if (rng.is_none() || PyLong_Check(rng.ptr())) {
            xo.reseed(seed_of(rng));
            src = &xo;
        } else {
            py = new PyRngDraw(rng.ptr());
            src = py;
        }
    }
    ~DrawHolder() { delete py; }
};

// --- entry points --------------------------------------------------------------------------------
py::list legal_list(const GameStateC& st);

py::list legal_actions_py(py::handle state) {
    std::unique_ptr<GameStateC> st(new GameStateC());
    return legal_list(*resolve(state.ptr(), *st));
}

// engine.apply_inplace on a C++ state: the Python-level checks ("game is over" before "unknown
// action <repr>", like engine.apply_inplace) plus the handler; *rolled is set for a roll.
void apply_checked(GameStateC& st, py::handle action, DrawSource& draw, bool* rolled) {
    ActionC a;
    parse_action(action.ptr(), a);
    if (st.phase == PHASE_GAME_OVER) throw_illegal("game is over");
    if (a.kind == ACT_UNKNOWN) {
        Ref r(check(PyObject_Repr(action.ptr())));
        const char* s = PyUnicode_AsUTF8(r.p);
        if (!s) throw py::error_already_set();
        throw_illegal(std::string("unknown action ") + s);
    }
    apply_inplace(st, a, draw, rolled);
}

py::list legal_list(const GameStateC& st) {
    static thread_local ActionList acts;
    legal_actions(st, acts);
    PyObject* out = check(PyList_New(acts.n));
    for (int i = 0; i < acts.n; ++i) {
        PyObject* t;
        try {
            t = action_to_python(acts.items[i]);
        } catch (...) {
            Py_DECREF(out);
            throw;
        }
        PyList_SET_ITEM(out, i, t);
    }
    return py::reinterpret_steal<py::list>(out);
}

// --- CState: a converted state kept in C++ -------------------------------------------------------
std::unique_ptr<CState> cstate_new(py::handle state) {
    std::unique_ptr<CState> c(new CState());
    if (py::isinstance<CState>(state)) {
        const CState& o = py::cast<const CState&>(state);
        c->s = o.s;
        c->origin = o.origin;
        c->rolls_history_len = o.rolls_history_len;
    } else {
        state_from_python(state.ptr(), c->s);
        c->origin = py::reinterpret_borrow<py::object>(state);
        c->rolls_history_len = rolls_history_len_of(state.ptr());
    }
    return c;
}

void cstate_apply(CState& c, py::handle action, DrawSource& draw) {
    bool rolled = false;
    apply_checked(c.s, action, draw, &rolled);
    if (rolled) ++c.rolls_history_len;
}

std::unique_ptr<CState> cstate_apply_new(const CState& c, py::handle action, py::handle rng) {
    std::unique_ptr<CState> out(new CState(c));
    DrawHolder d(rng);
    cstate_apply(*out, action, *d.src);
    return out;
}

py::object cstate_apply_inplace(py::object self, py::handle action, py::handle rng) {
    CState& c = py::cast<CState&>(self);
    DrawHolder d(rng);
    cstate_apply(c, action, *d.src);
    return self;
}

std::unique_ptr<CState> cstate_apply_forced(const CState& c, py::handle action, int drawn_index) {
    std::unique_ptr<CState> out(new CState(c));
    ForcedDraw d(drawn_index);
    cstate_apply(*out, action, d);
    return out;
}

py::object cstate_to_state(const CState& c) {
    return py::reinterpret_steal<py::object>(state_to_python_new(c.s, c.origin.ptr(), c.rolls_history_len));
}

std::unique_ptr<CState> cstate_random_playout(const CState& c, py::handle seed, py::handle max_turns, long max_actions) {
    std::unique_ptr<CState> out(new CState(c));
    if (!max_turns.is_none()) out->s.max_turns = to_int32(max_turns.ptr(), "max_turns");
    Xoshiro rng(seed_of(seed));
    out->rolls_history_len += random_playout(out->s, rng, max_actions, nullptr);
    return out;
}

py::object cstate_phase(const CState& c) {
    if (c.s.phase >= 0 && c.s.phase < NUM_PHASES) return py::reinterpret_borrow<py::object>(names().phase_str[c.s.phase]);
    return c.origin.attr("phase");  // an unknown phase string is kept as it was
}

int cstate_count_vp(const CState& c, long player, bool include_hidden) {
    check_player(c.s, player);
    return count_vp(c.s, (int)player, include_hidden);
}

py::object apply_impl(py::handle state, py::handle action, DrawSource& draw, bool inplace) {
    std::unique_ptr<GameStateC> st(new GameStateC());
    state_from_python(state.ptr(), *st);
    bool rolled = false;
    apply_checked(*st, action, draw, &rolled);
    if (inplace) {
        state_to_python_inplace(*st, state.ptr(), rolled);
        return py::reinterpret_borrow<py::object>(state);
    }
    return py::reinterpret_steal<py::object>(
        state_to_python_new(*st, state.ptr(), rolls_history_len_of(state.ptr()) + (rolled ? 1 : 0)));
}

py::object apply_py(py::handle state, py::handle action, py::handle rng) {
    DrawHolder d(rng);
    return apply_impl(state, action, *d.src, false);
}

py::object apply_inplace_py(py::handle state, py::handle action, py::handle rng) {
    DrawHolder d(rng);
    return apply_impl(state, action, *d.src, true);
}

py::object apply_forced_py(py::handle state, py::handle action, int drawn_index) {
    ForcedDraw d(drawn_index);
    return apply_impl(state, action, d, false);
}

py::object random_playout_fast_py(py::handle state, py::handle seed, py::handle max_turns, long max_actions,
                                  bool trace) {
    std::unique_ptr<GameStateC> st(new GameStateC());
    state_from_python(state.ptr(), *st);
    if (!max_turns.is_none()) st->max_turns = to_int32(max_turns.ptr(), "max_turns");
    Xoshiro rng(seed_of(seed));
    std::vector<PlayoutStep> steps;
    const long rolls = random_playout(*st, rng, max_actions, trace ? &steps : nullptr);
    py::object out = py::reinterpret_steal<py::object>(
        state_to_python_new(*st, state.ptr(), rolls_history_len_of(state.ptr()) + rolls));
    if (!trace) return out;
    py::list tr(steps.size());
    for (size_t i = 0; i < steps.size(); ++i) {
        py::object act = py::reinterpret_steal<py::object>(action_to_python(steps[i].action));
        if (PyList_SetItem(tr.ptr(), (Py_ssize_t)i, check(Py_BuildValue("(Oi)", act.ptr(), (int)steps[i].draw))) < 0)
            throw py::error_already_set();
    }
    return py::make_tuple(out, tr);
}

py::list production_for_roll_py(py::handle state, int value) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    int32_t gains[MAX_PLAYERS][NUM_RESOURCES];
    production_for_roll(st, value, gains);
    py::list out;
    for (int i = 0; i < st.num_players; ++i) out.append(py::reinterpret_steal<py::object>(int_list(gains[i], NUM_RESOURCES)));
    return out;
}

py::list discard_options_py(py::handle resources, int k, int cap) {
    int32_t res[NUM_RESOURCES];
    Ref fast(PySequence_Fast(resources.ptr(), "resources must be a sequence of 5 ints"));
    if (!fast.p) throw py::error_already_set();
    if (PySequence_Fast_GET_SIZE(fast.p) != NUM_RESOURCES) throw py::value_error("resources must have 5 entries");
    for (int i = 0; i < NUM_RESOURCES; ++i) res[i] = to_int32(PySequence_Fast_ITEMS(fast.p)[i], "resources");
    std::vector<int32_t> buf((size_t)(cap > 0 ? cap : 1) * NUM_RESOURCES);
    int32_t(*rows)[NUM_RESOURCES] = reinterpret_cast<int32_t(*)[NUM_RESOURCES]>(buf.data());
    const int n = discard_options(res, k, cap, rows);
    py::list out(n);
    for (int i = 0; i < n; ++i)
        if (PyList_SetItem(out.ptr(), i, vec_tuple(rows[i])) < 0) throw py::error_already_set();
    return out;
}

int acting_player_py(py::handle state) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    return acting_player(st);
}

int count_vp_py(py::handle state, long player, bool include_hidden) {
    GameStateC st;
    state_from_python(state.ptr(), st);
    check_player(st, player);
    return count_vp(st, (int)player, include_hidden);
}

}  // namespace engine_py

// Native lookahead (search.hpp): evaluator handles, future_values, reduced_search and the strategy helpers.
#include "search_bindings.inc"

}  // namespace

PYBIND11_MODULE(catanbot_core, m) {
    m.doc() = "C++ feature extraction and heuristic evaluation for catanbot (exact ports of "
              "catanbot.features.extract and catanbot.heuristic.static_value)";
    py::register_exception<unsupported_state>(m, "UnsupportedStateError", PyExc_ValueError)
        .attr("__doc__") = "The GameState cannot be represented by the extension (more than 4 players, lists "
                           "of the wrong length, ids or integers out of range, more than 64 road entries); "
                           "catanbot.accel falls back to the Python implementation for such states.";
    m.def("extract_batch", &extract_batch, py::arg("states"), py::arg("players"),
          "Feature matrix (N, NUM_FEATURES) float32 for the (state, player) pairs.");
    m.def("extract", &extract, py::arg("state"), py::arg("player"),
          "Feature vector (NUM_FEATURES,) float32 from player's perspective.");
    m.def("longest_road_length", &longest_road, py::arg("state"), py::arg("player"),
          "Longest trail over the player's roads (features.longest_road_length semantics).");
    m.def("num_features", [] { return (int)NUM_FEATURES; });
    m.def("feature_names", [] {
        py::list names;
        for (int i = 0; i < NUM_FEATURES; ++i) names.append(py::str(FEATURE_NAMES[i]));
        return names;
    });
    m.def("phase_names", [] {
        py::list names;
        for (int i = 0; i < NUM_PHASES; ++i) names.append(py::str(PHASE_NAMES[i]));
        return names;
    });
    // --- heuristic ---------------------------------------------------------
    m.def("static_values", &static_values_py, py::arg("state"),
          "heuristic.static_value of every player of the state -> list[float] (VP-equivalents).");
    m.def("static_value", &static_value_py, py::arg("state"), py::arg("player"),
          "heuristic.static_value(state, player).");
    m.def("heuristic_evaluate", &heuristic_evaluate_py, py::arg("states"), py::arg("players"),
          py::arg("temperature") = 16.0,
          "HeuristicEvaluator.evaluate: win-probability-like values, float64 (N,), for the (state, player) "
          "pairs (softmax of the static values over the players; 1 / 0 once the game is over with a winner).");
    m.def("heuristic_longest_road_length", &heuristic_longest_road_py, py::arg("state"), py::arg("player"),
          "heuristic.longest_road_length (a path may not start on an opponent's building).");
    m.def("reachable_spots", &reachable_spots_py, py::arg("state"), py::arg("player"), py::arg("max_roads") = 3,
          "placement.reachable_spots -> {vertex: (roads_needed, first_edge)} (first_edge may differ on ties).");
    m.def("score_settlement_spot", &score_settlement_spot_py, py::arg("state"), py::arg("player"),
          py::arg("vertex"), py::arg("setup") = false, "placement.score_settlement_spot with the default context.");
    m.def("player_production", &player_production_py, py::arg("state"), py::arg("player"),
          py::arg("ignore_robber") = false, "placement.player_production -> list of 5 floats.");
    m.def("resource_scarcity", &resource_scarcity_py, py::arg("state"), "placement.resource_scarcity -> 5 floats.");
    m.def("expected_hidden_vp", &expected_hidden_vp_py, py::arg("state"), py::arg("player"),
          "counting.expected_hidden_vp.");
    m.def("progress_to_build", &progress_to_build_py, py::arg("state"), py::arg("player"),
          "heuristic._progress_to_build.");
    // --- engine ------------------------------------------------------------
    // Illegal actions raise catanbot.engine.IllegalActionError itself (looked up when first
    // needed); this class is only the stand-in when catanbot.engine cannot be imported.
    static py::exception<illegal_action> illegal_exc(m, "IllegalActionError", PyExc_ValueError);
    engine_py::fallback_illegal_type = illegal_exc.ptr();
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const illegal_action& e) {
            PyErr_SetString(engine_py::illegal_action_type(), e.what());
        }
    });
    m.def("legal_actions", &engine_py::legal_actions_py, py::arg("state"),
          "engine.legal_actions: the same action tuples in the same order as the Python engine "
          "(`state` may also be a CState).");
    py::class_<CState>(m, "CState",
                       "A GameState converted once and kept in C++.  legal_actions() / apply() / apply_inplace() / "
                       "apply_forced() / random_playout() work on it without converting a Python GameState, "
                       "extract_batch / heuristic_evaluate / static_values accept it directly, and to_state() "
                       "materialises an ordinary GameState (hexes, ports and the players' colour / name come from "
                       "the GameState the handle was created from).")
        .def(py::init(&engine_py::cstate_new), py::arg("state"), "From a GameState (converted) or another CState (copied).")
        .def("legal_actions", [](const CState& c) { return engine_py::legal_list(c.s); }, "engine.legal_actions.")
        .def("apply", &engine_py::cstate_apply_new, py::arg("action"), py::arg("rng") = py::none(),
             "engine.apply -> a new CState (rng: None / int seed / random.Random, like core.apply).")
        .def("apply_inplace", &engine_py::cstate_apply_inplace, py::arg("action"), py::arg("rng") = py::none(),
             "engine.apply_inplace on this handle; returns it.")
        .def("apply_forced", &engine_py::cstate_apply_forced, py::arg("action"), py::arg("drawn_index"),
             "core.apply_forced -> a new CState.")
        .def("to_state", &engine_py::cstate_to_state, "A new catanbot.state.GameState with this position.")
        .def("copy", [](const CState& c) { return std::unique_ptr<CState>(new CState(c)); })
        .def("random_playout", &engine_py::cstate_random_playout, py::arg("seed") = py::none(),
             py::arg("max_turns") = py::none(), py::arg("max_actions") = 2000000L,
             "engine.random_playout from this position -> the final CState (this handle is unchanged).")
        .def("count_vp", &engine_py::cstate_count_vp, py::arg("player"), py::arg("include_hidden") = true)
        .def_property_readonly("phase", &engine_py::cstate_phase)
        .def_property_readonly("current", [](const CState& c) { return c.s.current; })
        .def_property_readonly("turn", [](const CState& c) { return c.s.turn; })
        .def_property_readonly("dice", [](const CState& c) { return c.s.dice; })
        .def_property_readonly("winner", [](const CState& c) { return c.s.winner; })
        .def_property_readonly("num_players", [](const CState& c) { return c.s.num_players; })
        .def_property_readonly("acting_player", [](const CState& c) { return acting_player(c.s); })
        .def_property_readonly("is_terminal", [](const CState& c) { return is_terminal(c.s); })
        .def_property_readonly("rolls_history_len", [](const CState& c) { return c.rolls_history_len; })
        .def_property_readonly("origin", [](const CState& c) { return c.origin; },
                               "The GameState this handle (or the handle it was derived from) was created from.")
        .def_property(
            "max_turns", [](const CState& c) { return c.s.max_turns; },
            [](CState& c, int v) { c.s.max_turns = v; });
    m.def("apply", &engine_py::apply_py, py::arg("state"), py::arg("action"), py::arg("rng") = py::none(),
          "engine.apply: a NEW GameState with the action applied (the input is not modified).  `rng` is None "
          "(fresh C++ rng), an int seed (C++ rng) or a random.Random, which is consulted exactly like the Python "
          "engine does (randint(1, 6) twice for a roll, randrange(total) for a steal / dev card draw).  Illegal "
          "actions raise engine.IllegalActionError with the Python message.");
    m.def("apply_inplace", &engine_py::apply_inplace_py, py::arg("state"), py::arg("action"),
          py::arg("rng") = py::none(),
          "engine.apply_inplace: applies the action to `state` (written back into the same GameState / Player / "
          "list objects) and returns it.");
    m.def("apply_forced", &engine_py::apply_forced_py, py::arg("state"), py::arg("action"), py::arg("drawn_index"),
          "apply() with the random choice given: the index of the stolen card among the victim's cards in "
          "resource order, the index of the drawn dev card into the deck (in dev-type order), or for (ROLL,) "
          "the dice pair index d = 6 * (d1 - 1) + (d2 - 1).  ValueError when the index is out of range.");
    m.def("random_playout_fast", &engine_py::random_playout_fast_py, py::arg("state"), py::arg("seed") = py::none(),
          py::arg("max_turns") = py::none(), py::arg("max_actions") = 2000000L, py::arg("trace") = false,
          "engine.random_playout entirely in C++: uniformly random legal actions until the game is over, "
          "returns the final GameState (a new object).  With trace=True returns (state, [(action, draw), ...]) "
          "where draw is the random index the action consumed (-1 = none; rolls are recorded as (ROLL, value)) "
          "so the game can be replayed with apply_forced.");
    m.def("production_for_roll", &engine_py::production_for_roll_py, py::arg("state"), py::arg("value"),
          "engine.production_for_roll -> list of 5-int lists per player.");
    m.def("discard_options", &engine_py::discard_options_py, py::arg("resources"), py::arg("k"),
          py::arg("cap") = (int)DISCARD_ENUM_CAP, "engine.discard_options -> list of 5-tuples in the Python order.");
    m.def("acting_player", &engine_py::acting_player_py, py::arg("state"), "engine.acting_player.");
    m.def("count_vp", &engine_py::count_vp_py, py::arg("state"), py::arg("player"), py::arg("include_hidden") = true,
          "engine.count_vp.");
    m.attr("MAX_TRADE_PROPOSALS_PER_TURN") = (int)MAX_TRADE_PROPOSALS_PER_TURN;
    m.attr("DISCARD_ENUM_CAP") = (int)DISCARD_ENUM_CAP;
    m.attr("NUM_FEATURES") = (int)NUM_FEATURES;
    m.attr("PLAYER_BLOCK") = (int)PLAYER_BLOCK;
    m.attr("GLOBAL_BLOCK") = (int)GLOBAL_BLOCK;
    m.attr("MAX_PLAYERS") = (int)MAX_PLAYERS;
    // --- native lookahead ---------------------------------------------------
    search_py::register_search(m);
    m.attr("__version__") = "0.4.0";
}
