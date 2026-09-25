// pybind11 module `catanbot_core`: C++ feature extraction for catanbot.
//
//   extract_batch(states, players) -> np.ndarray float32 (N, NUM_FEATURES)
//   extract(state, player)         -> np.ndarray float32 (NUM_FEATURES,)
//   longest_road_length(state, player) -> int
//   num_features() / feature_names() / phase_names()
//
// `states` are catanbot.state.GameState objects; they are converted once per
// distinct consecutive object (like features.extract_batch) with the raw
// CPython API (see state.hpp).  The GIL is held throughout (the work per state
// is a few microseconds, releasing it would cost more than it saves).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>

#include "features.hpp"
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

}  // namespace

PYBIND11_MODULE(catanbot_core, m) {
    m.doc() = "C++ feature extraction for catanbot (exact port of catanbot.features.extract)";
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
    m.attr("NUM_FEATURES") = (int)NUM_FEATURES;
    m.attr("PLAYER_BLOCK") = (int)PLAYER_BLOCK;
    m.attr("GLOBAL_BLOCK") = (int)GLOBAL_BLOCK;
    m.attr("MAX_PLAYERS") = (int)MAX_PLAYERS;
    m.attr("__version__") = "0.1.0";
}
