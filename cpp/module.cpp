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
//
// `states` are catanbot.state.GameState objects; they are converted once per
// distinct consecutive object (like features.extract_batch) with the raw
// CPython API (see state.hpp).  The GIL is held throughout (the work per state
// is a few microseconds, releasing it would cost more than it saves).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "features.hpp"
#include "heuristic.hpp"
#include "state.hpp"

namespace py = pybind11;
using namespace catanbot;

namespace {

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
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = sitems[i];
        if (s != last) {  // consecutive identical state objects share one analysis
            state_from_python(s, *st);
            analyse(*st, *an);
            last = s;
        }
        const long player = index_of(pitems[i]);
        check_player(*st, player);
        assemble(*st, *an, (int)player, out + i * NUM_FEATURES);
    }
    return X;
}

py::array_t<float> extract(py::handle state, long player) {
    std::unique_ptr<GameStateC> st(new GameStateC());
    std::unique_ptr<Analysis> an(new Analysis());
    state_from_python(state.ptr(), *st);
    check_player(*st, player);
    analyse(*st, *an);
    py::array_t<float> x({(py::ssize_t)NUM_FEATURES});
    assemble(*st, *an, (int)player, x.mutable_data());
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
    GameStateC st;
    state_from_python(state.ptr(), st);
    double vals[MAX_PLAYERS];
    static_values(st, vals);
    py::list out;
    for (int i = 0; i < st.num_players; ++i) out.append(vals[i]);
    return out;
}

double static_value_py(py::handle state, long player) {
    GameStateC st;
    state_from_python(state.ptr(), st);
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
            state_from_python(s, st);
            Entry e;
            static_values(st, e.vals);
            e.num_players = st.num_players;
            e.decided = st.phase == PHASE_GAME_OVER && st.winner >= 0;
            e.winner = st.winner;
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

}  // namespace

PYBIND11_MODULE(catanbot_core, m) {
    m.doc() = "C++ feature extraction and heuristic evaluation for catanbot (exact ports of "
              "catanbot.features.extract and catanbot.heuristic.static_value)";
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
    m.attr("NUM_FEATURES") = (int)NUM_FEATURES;
    m.attr("PLAYER_BLOCK") = (int)PLAYER_BLOCK;
    m.attr("GLOBAL_BLOCK") = (int)GLOBAL_BLOCK;
    m.attr("MAX_PLAYERS") = (int)MAX_PLAYERS;
    m.attr("__version__") = "0.2.0";
}
