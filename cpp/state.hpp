// Plain C++ mirror of catanbot.state.GameState plus a converter from the Python object.
//
// The struct is POD-like (fixed-size arrays, no heap allocation) so a state can
// be copied with memcpy and lives on the stack.  Ordered lists (settlements,
// cities, roads, discard queue) keep the Python list order - the feature code
// accumulates floating point production in exactly that order so results are
// bit-identical to the Python reference - and additionally expose bitsets for
// O(1) membership tests.
//
// The converter (`state_from_python`) reads the attributes of a
// catanbot.state.GameState / Player / TradeOffer directly through the CPython
// C API (interned attribute names, PySequence_Fast, PyLong_AsLong) - it never
// goes through GameState.to_dict().  Define CATAN_NO_PYTHON to use the struct
// without Python.h.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "board_tables.hpp"
#include "feature_layout.hpp"

namespace catanbot {

constexpr int MAX_DISCARD_QUEUE = 8;

struct PlayerC {
    int32_t resources[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    int32_t dev_cards[NUM_DEV] = {0, 0, 0, 0, 0};       // playable (bought on an earlier turn)
    int32_t dev_cards_new[NUM_DEV] = {0, 0, 0, 0, 0};   // bought this turn
    int32_t played_knights = 0;
    // Buildings and roads as ordered id lists (Python order) ...
    int32_t n_settlements = 0;
    int32_t n_cities = 0;
    int32_t n_roads = 0;
    uint8_t settlements[NUM_VERTICES] = {};
    uint8_t cities[NUM_VERTICES] = {};
    uint8_t roads[NUM_EDGES] = {};
    // ... and as bitsets (bit v / bit e; edges >= 64 live in road_bits[1]).
    uint64_t settlement_bits = 0;
    uint64_t city_bits = 0;
    uint64_t road_bits[2] = {0, 0};
    // Hidden-information bookkeeping (see state.py).
    bool hand_known = true;
    int32_t hand_size = 0;
    bool dev_known = true;
    int32_t dev_count = 0;

    bool has_settlement(int v) const { return (settlement_bits >> v) & 1u; }
    bool has_city(int v) const { return (city_bits >> v) & 1u; }
    bool has_building(int v) const { return ((settlement_bits | city_bits) >> v) & 1u; }
    bool has_road(int e) const { return (road_bits[e >> 6] >> (e & 63)) & 1u; }
    int total_resources() const {
        return resources[0] + resources[1] + resources[2] + resources[3] + resources[4];
    }
    int total_dev() const {
        int t = 0;
        for (int k = 0; k < NUM_DEV; ++k) t += dev_cards[k] + dev_cards_new[k];
        return t;
    }
    int vp_cards() const { return dev_cards[DEV_VICTORY_POINT] + dev_cards_new[DEV_VICTORY_POINT]; }
};

struct TradeC {
    int32_t proposer = -1;
    int32_t give[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    int32_t get[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    int8_t responses[MAX_PLAYERS] = {-1, -1, -1, -1};  // -1 = no response yet, 0 = rejected, 1 = accepted
};

struct GameStateC {
    // Board
    int8_t hex_res[NUM_HEXES] = {};
    int8_t hex_num[NUM_HEXES] = {};
    int32_t robber = 0;
    int8_t ports[NUM_VERTICES] = {};   // vertex -> port type, -1 = none
    // Players and supplies
    int32_t num_players = 0;
    PlayerC players[MAX_PLAYERS];
    int32_t bank[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    int32_t dev_deck[NUM_DEV] = {0, 0, 0, 0, 0};
    // Turn structure
    int32_t current = 0;
    Phase phase = PHASE_UNKNOWN;
    int32_t turn = 0;
    int32_t setup_round = 0;
    int32_t setup_last_settlement = -1;
    int32_t dice = 0;
    bool dev_played_this_turn = false;
    int32_t free_roads = 0;
    int32_t n_discard = 0;
    int32_t discard_queue[MAX_DISCARD_QUEUE] = {};
    bool has_pending_trade = false;
    TradeC pending_trade;
    int32_t trade_responder = -1;
    int32_t trades_this_turn = 0;
    // Awards
    int32_t longest_road_owner = -1;
    int32_t longest_road_len = 0;
    int32_t largest_army_owner = -1;
    int32_t winner = -1;
    int32_t max_turns = 400;

    int public_vp(int i) const {
        const PlayerC& p = players[i];
        int vp = p.n_settlements + 2 * p.n_cities;
        if (longest_road_owner == i) vp += 2;
        if (largest_army_owner == i) vp += 2;
        return vp;
    }
    int total_vp(int i) const { return public_vp(i) + players[i].vp_cards(); }
    int vertex_owner(int v) const {
        for (int i = 0; i < num_players; ++i)
            if (players[i].has_building(v)) return i;
        return -1;
    }
    int edge_owner(int e) const {
        for (int i = 0; i < num_players; ++i)
            if (players[i].has_road(e)) return i;
        return -1;
    }
    // Player who must act now (engine.acting_player / features._acting_player).
    int acting_player() const {
        if (phase == PHASE_DISCARD && n_discard > 0) return discard_queue[0];
        if (phase == PHASE_TRADE_RESPONSE) return trade_responder;
        return current;
    }
};

}  // namespace catanbot

// ===========================================================================
// Python -> GameStateC converter
// ===========================================================================
#ifndef CATAN_NO_PYTHON
#include <pybind11/pybind11.h>

namespace catanbot {
namespace detail {

// Interned attribute names, created once (leaked on purpose: they must outlive
// every static destructor that could run after interpreter finalisation).
struct Names {
    PyObject *hexes, *robber, *ports, *players, *bank, *dev_deck, *current, *phase, *turn, *setup_round,
        *setup_last_settlement, *dice, *dev_played_this_turn, *free_roads, *discard_queue, *pending_trade,
        *trade_responder, *trades_this_turn, *longest_road_owner, *longest_road_len, *largest_army_owner,
        *winner, *max_turns;
    PyObject *resources, *dev_cards, *dev_cards_new, *played_knights, *settlements, *cities, *roads,
        *hand_known, *hand_size, *dev_known, *dev_count;
    PyObject *proposer, *give, *get, *responses;
    PyObject* phase_str[NUM_PHASES];
};

inline PyObject* intern(const char* s) {
    PyObject* o = PyUnicode_InternFromString(s);
    if (!o) throw pybind11::error_already_set();
    return o;
}

inline const Names& names() {
    static const Names* n = [] {
        Names* m = new Names();
        m->hexes = intern("hexes");
        m->robber = intern("robber");
        m->ports = intern("ports");
        m->players = intern("players");
        m->bank = intern("bank");
        m->dev_deck = intern("dev_deck");
        m->current = intern("current");
        m->phase = intern("phase");
        m->turn = intern("turn");
        m->setup_round = intern("setup_round");
        m->setup_last_settlement = intern("setup_last_settlement");
        m->dice = intern("dice");
        m->dev_played_this_turn = intern("dev_played_this_turn");
        m->free_roads = intern("free_roads");
        m->discard_queue = intern("discard_queue");
        m->pending_trade = intern("pending_trade");
        m->trade_responder = intern("trade_responder");
        m->trades_this_turn = intern("trades_this_turn");
        m->longest_road_owner = intern("longest_road_owner");
        m->longest_road_len = intern("longest_road_len");
        m->largest_army_owner = intern("largest_army_owner");
        m->winner = intern("winner");
        m->max_turns = intern("max_turns");
        m->resources = intern("resources");
        m->dev_cards = intern("dev_cards");
        m->dev_cards_new = intern("dev_cards_new");
        m->played_knights = intern("played_knights");
        m->settlements = intern("settlements");
        m->cities = intern("cities");
        m->roads = intern("roads");
        m->hand_known = intern("hand_known");
        m->hand_size = intern("hand_size");
        m->dev_known = intern("dev_known");
        m->dev_count = intern("dev_count");
        m->proposer = intern("proposer");
        m->give = intern("give");
        m->get = intern("get");
        m->responses = intern("responses");
        for (int i = 0; i < NUM_PHASES; ++i) m->phase_str[i] = intern(PHASE_NAMES[i]);
        return m;
    }();
    return *n;
}

[[noreturn]] inline void bad_state(const std::string& msg) {
    throw pybind11::value_error("catanbot_core: invalid GameState: " + msg);
}

// Owned reference helper (tiny RAII wrapper, no pybind11 overhead in the hot path).
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* o) : p(o) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
};

inline PyObject* getattr(PyObject* obj, PyObject* name) {
    PyObject* v = PyObject_GetAttr(obj, name);
    if (!v) throw pybind11::error_already_set();
    return v;  // new reference
}

inline long to_long(PyObject* v) {
    long r = PyLong_AsLong(v);
    if (r == -1 && PyErr_Occurred()) {
        // Lenient fallback for floats / numpy scalars that are not ints.
        PyErr_Clear();
        PyObject* i = PyNumber_Long(v);
        if (!i) throw pybind11::error_already_set();
        r = PyLong_AsLong(i);
        Py_DECREF(i);
        if (r == -1 && PyErr_Occurred()) throw pybind11::error_already_set();
    }
    return r;
}

inline long get_long(PyObject* obj, PyObject* name) {
    Ref v(getattr(obj, name));
    return to_long(v.p);
}

inline bool get_bool(PyObject* obj, PyObject* name) {
    Ref v(getattr(obj, name));
    int r = PyObject_IsTrue(v.p);
    if (r < 0) throw pybind11::error_already_set();
    return r != 0;
}

// Read a sequence of ints into out[0..cap); returns the length.  Raises when longer than cap.
inline int read_ints(PyObject* seq, int32_t* out, int cap, const char* what) {
    Ref fast(PySequence_Fast(seq, "expected a sequence"));
    if (!fast.p) throw pybind11::error_already_set();
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
    if (n > cap) bad_state(std::string(what) + " has " + std::to_string(n) + " entries (max " + std::to_string(cap) + ")");
    PyObject** items = PySequence_Fast_ITEMS(fast.p);
    for (Py_ssize_t i = 0; i < n; ++i) out[i] = (int32_t)to_long(items[i]);
    return (int)n;
}

inline void read_exact5(PyObject* obj, PyObject* name, int32_t* out, const char* what) {
    Ref v(getattr(obj, name));
    int32_t tmp[8];
    int n = read_ints(v.p, tmp, 8, what);
    if (n != 5) bad_state(std::string(what) + " must have exactly 5 entries, got " + std::to_string(n));
    std::memcpy(out, tmp, 5 * sizeof(int32_t));
}

// Read a list of ids (0 <= id < limit) into a uint8 list + bitset.
inline int read_ids(PyObject* obj, PyObject* name, uint8_t* out, int cap, int limit, uint64_t* bits,
                    const char* what) {
    Ref v(getattr(obj, name));
    Ref fast(PySequence_Fast(v.p, "expected a sequence"));
    if (!fast.p) throw pybind11::error_already_set();
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
    if (n > cap) bad_state(std::string(what) + " has " + std::to_string(n) + " entries (max " + std::to_string(cap) + ")");
    PyObject** items = PySequence_Fast_ITEMS(fast.p);
    for (Py_ssize_t i = 0; i < n; ++i) {
        long id = to_long(items[i]);
        if (id < 0 || id >= limit)
            bad_state(std::string(what) + " id " + std::to_string(id) + " out of range [0, " + std::to_string(limit) + ")");
        out[i] = (uint8_t)id;
        bits[id >> 6] |= (uint64_t)1 << (id & 63);
    }
    return (int)n;
}

inline Phase read_phase(PyObject* obj) {
    const Names& N = names();
    Ref v(getattr(obj, N.phase));
    for (int i = 0; i < NUM_PHASES; ++i)  // fast path: interned string identity
        if (v.p == N.phase_str[i]) return (Phase)i;
    if (!PyUnicode_Check(v.p)) return PHASE_UNKNOWN;
    Py_ssize_t len = 0;
    const char* s = PyUnicode_AsUTF8AndSize(v.p, &len);
    if (!s) throw pybind11::error_already_set();
    for (int i = 0; i < NUM_PHASES; ++i)
        if (std::strcmp(s, PHASE_NAMES[i]) == 0) return (Phase)i;
    return PHASE_UNKNOWN;  // features.py: no phase one-hot is set
}

inline void read_player(PyObject* obj, PlayerC& p) {
    const Names& N = names();
    read_exact5(obj, N.resources, p.resources, "Player.resources");
    read_exact5(obj, N.dev_cards, p.dev_cards, "Player.dev_cards");
    read_exact5(obj, N.dev_cards_new, p.dev_cards_new, "Player.dev_cards_new");
    p.played_knights = (int32_t)get_long(obj, N.played_knights);
    uint64_t sb[2] = {0, 0}, cb[2] = {0, 0};
    p.n_settlements = read_ids(obj, N.settlements, p.settlements, NUM_VERTICES, NUM_VERTICES, sb, "Player.settlements");
    p.n_cities = read_ids(obj, N.cities, p.cities, NUM_VERTICES, NUM_VERTICES, cb, "Player.cities");
    p.settlement_bits = sb[0];
    p.city_bits = cb[0];
    p.road_bits[0] = p.road_bits[1] = 0;
    p.n_roads = read_ids(obj, N.roads, p.roads, NUM_EDGES, NUM_EDGES, p.road_bits, "Player.roads");
    p.hand_known = get_bool(obj, N.hand_known);
    p.hand_size = (int32_t)get_long(obj, N.hand_size);
    p.dev_known = get_bool(obj, N.dev_known);
    p.dev_count = (int32_t)get_long(obj, N.dev_count);
}

inline void read_ports(PyObject* obj, int8_t* ports) {
    for (int v = 0; v < NUM_VERTICES; ++v) ports[v] = -1;
    Ref d(getattr(obj, names().ports));
    if (d.p == Py_None) return;
    Ref items(PyDict_Check(d.p) ? nullptr : PyMapping_Items(d.p));
    if (!PyDict_Check(d.p) && !items.p) throw pybind11::error_already_set();
    if (PyDict_Check(d.p)) {
        PyObject *key, *val;
        Py_ssize_t pos = 0;
        while (PyDict_Next(d.p, &pos, &key, &val)) {
            if (!PyLong_Check(key)) continue;  // a non-int key can never match an int vertex
            long v = PyLong_AsLong(key);
            if (v < 0 || v >= NUM_VERTICES) continue;  // never queried by the feature code
            long t = to_long(val);
            ports[v] = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t));
        }
    } else {
        Ref fast(PySequence_Fast(items.p, "ports.items()"));
        if (!fast.p) throw pybind11::error_already_set();
        Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
        PyObject** arr = PySequence_Fast_ITEMS(fast.p);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* kv = arr[i];
            if (!PyTuple_Check(kv) || PyTuple_GET_SIZE(kv) != 2) bad_state("ports items");
            PyObject* key = PyTuple_GET_ITEM(kv, 0);
            if (!PyLong_Check(key)) continue;
            long v = PyLong_AsLong(key);
            if (v < 0 || v >= NUM_VERTICES) continue;
            long t = to_long(PyTuple_GET_ITEM(kv, 1));
            ports[v] = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t));
        }
    }
}

inline void read_hexes(PyObject* obj, GameStateC& s) {
    Ref h(getattr(obj, names().hexes));
    Ref fast(PySequence_Fast(h.p, "hexes must be a sequence"));
    if (!fast.p) throw pybind11::error_already_set();
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
    if (n != NUM_HEXES) bad_state("hexes must have " + std::to_string(NUM_HEXES) + " entries, got " + std::to_string(n));
    PyObject** items = PySequence_Fast_ITEMS(fast.p);
    for (int i = 0; i < NUM_HEXES; ++i) {
        PyObject* pair = items[i];
        long res, num;
        if (PyTuple_Check(pair) && PyTuple_GET_SIZE(pair) == 2) {
            res = to_long(PyTuple_GET_ITEM(pair, 0));
            num = to_long(PyTuple_GET_ITEM(pair, 1));
        } else {
            int32_t tmp[2];
            if (read_ints(pair, tmp, 2, "hex") != 2) bad_state("hex entries must be (resource, number) pairs");
            res = tmp[0];
            num = tmp[1];
        }
        if (res < 0 || res > DESERT) bad_state("hex resource " + std::to_string(res) + " out of range");
        s.hex_res[i] = (int8_t)res;
        s.hex_num[i] = (int8_t)(num < -128 ? -128 : (num > 127 ? 127 : num));  // PIPS lookup treats it as "no pips"
    }
}

inline void read_trade(PyObject* obj, GameStateC& s) {
    const Names& N = names();
    Ref t(getattr(obj, N.pending_trade));
    s.has_pending_trade = false;
    if (t.p == Py_None) return;
    s.has_pending_trade = true;
    TradeC& tr = s.pending_trade;
    tr.proposer = (int32_t)get_long(t.p, N.proposer);
    read_exact5(t.p, N.give, tr.give, "TradeOffer.give");
    read_exact5(t.p, N.get, tr.get, "TradeOffer.get");
    for (int i = 0; i < MAX_PLAYERS; ++i) tr.responses[i] = -1;
    if (PyObject_HasAttr(t.p, N.responses)) {
        Ref r(getattr(t.p, N.responses));
        if (PyDict_Check(r.p)) {
            PyObject *key, *val;
            Py_ssize_t pos = 0;
            while (PyDict_Next(r.p, &pos, &key, &val)) {
                if (!PyLong_Check(key)) continue;
                long k = PyLong_AsLong(key);
                if (k < 0 || k >= MAX_PLAYERS) continue;
                int b = PyObject_IsTrue(val);
                if (b < 0) throw pybind11::error_already_set();
                tr.responses[k] = (int8_t)b;
            }
        }
    }
}

}  // namespace detail

// Fill `s` from a catanbot.state.GameState instance (attribute access, no to_dict()).
// Throws pybind11::error_already_set on Python errors and pybind11::value_error on
// malformed states (wrong list lengths, ids out of range, > MAX_PLAYERS players).
inline void state_from_python(PyObject* obj, GameStateC& s) {
    using namespace detail;
    const Names& N = names();
    s = GameStateC{};
    read_hexes(obj, s);
    s.robber = (int32_t)get_long(obj, N.robber);
    read_ports(obj, s.ports);
    {
        Ref pl(getattr(obj, N.players));
        Ref fast(PySequence_Fast(pl.p, "players must be a sequence"));
        if (!fast.p) throw pybind11::error_already_set();
        Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
        if (n > MAX_PLAYERS) bad_state("at most " + std::to_string(MAX_PLAYERS) + " players are supported, got " + std::to_string(n));
        s.num_players = (int32_t)n;
        PyObject** items = PySequence_Fast_ITEMS(fast.p);
        for (Py_ssize_t i = 0; i < n; ++i) read_player(items[i], s.players[i]);
    }
    read_exact5(obj, N.bank, s.bank, "GameState.bank");
    read_exact5(obj, N.dev_deck, s.dev_deck, "GameState.dev_deck");
    s.current = (int32_t)get_long(obj, N.current);
    s.phase = read_phase(obj);
    s.turn = (int32_t)get_long(obj, N.turn);
    s.setup_round = (int32_t)get_long(obj, N.setup_round);
    s.setup_last_settlement = (int32_t)get_long(obj, N.setup_last_settlement);
    s.dice = (int32_t)get_long(obj, N.dice);
    s.dev_played_this_turn = get_bool(obj, N.dev_played_this_turn);
    s.free_roads = (int32_t)get_long(obj, N.free_roads);
    {
        Ref q(getattr(obj, N.discard_queue));
        s.n_discard = read_ints(q.p, s.discard_queue, MAX_DISCARD_QUEUE, "GameState.discard_queue");
    }
    read_trade(obj, s);
    s.trade_responder = (int32_t)get_long(obj, N.trade_responder);
    s.trades_this_turn = (int32_t)get_long(obj, N.trades_this_turn);
    s.longest_road_owner = (int32_t)get_long(obj, N.longest_road_owner);
    s.longest_road_len = (int32_t)get_long(obj, N.longest_road_len);
    s.largest_army_owner = (int32_t)get_long(obj, N.largest_army_owner);
    s.winner = (int32_t)get_long(obj, N.winner);
    s.max_turns = (int32_t)get_long(obj, N.max_turns);
}

inline GameStateC state_from_python(pybind11::handle obj) {
    GameStateC s;
    state_from_python(obj.ptr(), s);
    return s;
}

}  // namespace catanbot
#endif  // CATAN_NO_PYTHON
