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
// C API (interned attribute names, PySequence_Fast, PyLong_AsLongAndOverflow) -
// it never goes through GameState.to_dict().  Every integer is range-checked
// against the 32-bit fields below; a state the struct cannot represent raises
// `unsupported_state` (never a silently wrapped value).  Define CATAN_NO_PYTHON
// to use the struct without Python.h.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "board_tables.hpp"
#include "feature_layout.hpp"

namespace catanbot {

constexpr int MAX_DISCARD_QUEUE = 8;

// Thrown (by the converter below and by the feature code) for a GameState the
// extension cannot represent: more than MAX_PLAYERS players, lists of the wrong
// length, ids out of range, integers outside the 32-bit fields, more than 64
// road entries for one player.  The Python reference implementation handles
// many of these states, so module.cpp exposes the type as
// `catanbot_core.UnsupportedStateError` (a ValueError subclass) and
// catanbot.accel falls back to Python when it sees one.
struct unsupported_state : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void unsupported(const std::string& msg) {
    throw unsupported_state("catanbot_core: unsupported GameState: " + msg);
}

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

// An interned attribute name plus the same text as a C string (for error messages).
struct Attr {
    PyObject* obj = nullptr;
    const char* name = "";
    operator PyObject*() const { return obj; }
};

// Interned attribute names, created once (leaked on purpose: they must outlive
// every static destructor that could run after interpreter finalisation).
struct Names {
    Attr hexes, robber, ports, players, bank, dev_deck, current, phase, turn, setup_round,
        setup_last_settlement, dice, dev_played_this_turn, free_roads, discard_queue, pending_trade,
        trade_responder, trades_this_turn, longest_road_owner, longest_road_len, largest_army_owner,
        winner, max_turns;
    Attr resources, dev_cards, dev_cards_new, played_knights, settlements, cities, roads,
        hand_known, hand_size, dev_known, dev_count;
    Attr proposer, give, get, responses;
    PyObject* phase_str[NUM_PHASES];
};

inline PyObject* intern(const char* s) {
    PyObject* o = PyUnicode_InternFromString(s);
    if (!o) throw pybind11::error_already_set();
    return o;
}

inline Attr attr(const char* s) { return Attr{intern(s), s}; }

inline const Names& names() {
    static const Names* n = [] {
        Names* m = new Names();
        m->hexes = attr("hexes");
        m->robber = attr("robber");
        m->ports = attr("ports");
        m->players = attr("players");
        m->bank = attr("bank");
        m->dev_deck = attr("dev_deck");
        m->current = attr("current");
        m->phase = attr("phase");
        m->turn = attr("turn");
        m->setup_round = attr("setup_round");
        m->setup_last_settlement = attr("setup_last_settlement");
        m->dice = attr("dice");
        m->dev_played_this_turn = attr("dev_played_this_turn");
        m->free_roads = attr("free_roads");
        m->discard_queue = attr("discard_queue");
        m->pending_trade = attr("pending_trade");
        m->trade_responder = attr("trade_responder");
        m->trades_this_turn = attr("trades_this_turn");
        m->longest_road_owner = attr("longest_road_owner");
        m->longest_road_len = attr("longest_road_len");
        m->largest_army_owner = attr("largest_army_owner");
        m->winner = attr("winner");
        m->max_turns = attr("max_turns");
        m->resources = attr("resources");
        m->dev_cards = attr("dev_cards");
        m->dev_cards_new = attr("dev_cards_new");
        m->played_knights = attr("played_knights");
        m->settlements = attr("settlements");
        m->cities = attr("cities");
        m->roads = attr("roads");
        m->hand_known = attr("hand_known");
        m->hand_size = attr("hand_size");
        m->dev_known = attr("dev_known");
        m->dev_count = attr("dev_count");
        m->proposer = attr("proposer");
        m->give = attr("give");
        m->get = attr("get");
        m->responses = attr("responses");
        for (int i = 0; i < NUM_PHASES; ++i) m->phase_str[i] = intern(PHASE_NAMES[i]);
        return m;
    }();
    return *n;
}

[[noreturn]] inline void bad_state(const std::string& msg) { unsupported(msg); }

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

// Python int (or anything int() accepts: floats and numpy scalars are truncated) -> long.
// An int beyond the range of a C long has no counterpart in GameStateC and is reported as
// unsupported (`what` names the field) instead of raising OverflowError.
inline long to_long(PyObject* v, const char* what) {
    int overflow = 0;
    long r = PyLong_AsLongAndOverflow(v, &overflow);
    if (overflow) bad_state(std::string(what) + " does not fit in 64 bits");
    if (r == -1 && PyErr_Occurred()) {
        // Lenient fallback for floats / numpy scalars that are not ints.
        PyErr_Clear();
        PyObject* i = PyNumber_Long(v);
        if (!i) throw pybind11::error_already_set();
        r = PyLong_AsLongAndOverflow(i, &overflow);
        Py_DECREF(i);
        if (overflow) bad_state(std::string(what) + " does not fit in 64 bits");
        if (r == -1 && PyErr_Occurred()) throw pybind11::error_already_set();
    }
    return r;
}

// Every integer field of GameStateC is 32 bits wide: a value outside that range is rejected
// rather than narrowed (robber = 2**32 + 5 must not silently become hex 5).
inline int32_t to_int32(PyObject* v, const char* what) {
    const long r = to_long(v, what);
    if (r < INT32_MIN || r > INT32_MAX)
        bad_state(std::string(what) + " = " + std::to_string(r) + " is outside the supported 32-bit range");
    return (int32_t)r;
}

inline int32_t get_int32(PyObject* obj, const Attr& a) {
    Ref v(getattr(obj, a.obj));
    return to_int32(v.p, a.name);
}

inline bool get_bool(PyObject* obj, PyObject* name) {
    Ref v(getattr(obj, name));
    int r = PyObject_IsTrue(v.p);
    if (r < 0) throw pybind11::error_already_set();
    return r != 0;
}

// Dict key -> int, or -1 when the key is not an int or is too big for a long: such a key can
// never name a vertex / player, and no Python error is left pending (PyLong_AsLong would set one).
inline long key_to_long(PyObject* key) {
    if (!PyLong_Check(key)) return -1;
    int overflow = 0;
    const long v = PyLong_AsLongAndOverflow(key, &overflow);
    return overflow ? -1 : v;
}

// Read a sequence of ints into out[0..cap); returns the length.  Raises when longer than cap.
inline int read_ints(PyObject* seq, int32_t* out, int cap, const char* what) {
    Ref fast(PySequence_Fast(seq, "expected a sequence"));
    if (!fast.p) throw pybind11::error_already_set();
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
    if (n > cap) bad_state(std::string(what) + " has " + std::to_string(n) + " entries (max " + std::to_string(cap) + ")");
    PyObject** items = PySequence_Fast_ITEMS(fast.p);
    for (Py_ssize_t i = 0; i < n; ++i) out[i] = to_int32(items[i], what);
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
        long id = to_long(items[i], what);
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
    p.played_knights = get_int32(obj, N.played_knights);
    uint64_t sb[2] = {0, 0}, cb[2] = {0, 0};
    p.n_settlements = read_ids(obj, N.settlements, p.settlements, NUM_VERTICES, NUM_VERTICES, sb, "Player.settlements");
    p.n_cities = read_ids(obj, N.cities, p.cities, NUM_VERTICES, NUM_VERTICES, cb, "Player.cities");
    p.settlement_bits = sb[0];
    p.city_bits = cb[0];
    p.road_bits[0] = p.road_bits[1] = 0;
    p.n_roads = read_ids(obj, N.roads, p.roads, NUM_EDGES, NUM_EDGES, p.road_bits, "Player.roads");
    p.hand_known = get_bool(obj, N.hand_known);
    p.hand_size = get_int32(obj, N.hand_size);
    p.dev_known = get_bool(obj, N.dev_known);
    p.dev_count = get_int32(obj, N.dev_count);
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
            const long v = key_to_long(key);
            if (v < 0 || v >= NUM_VERTICES) continue;  // never queried by the feature code
            const long t = to_long(val, "port type");
            ports[v] = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t));  // outside 0..5 = ignored
        }
    } else {
        Ref fast(PySequence_Fast(items.p, "ports.items()"));
        if (!fast.p) throw pybind11::error_already_set();
        Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.p);
        PyObject** arr = PySequence_Fast_ITEMS(fast.p);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* kv = arr[i];
            if (!PyTuple_Check(kv) || PyTuple_GET_SIZE(kv) != 2) bad_state("ports items");
            const long v = key_to_long(PyTuple_GET_ITEM(kv, 0));
            if (v < 0 || v >= NUM_VERTICES) continue;
            const long t = to_long(PyTuple_GET_ITEM(kv, 1), "port type");
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
        PyObject *res_o, *num_o;
        Ref seq(nullptr);
        if (PyTuple_Check(pair) && PyTuple_GET_SIZE(pair) == 2) {
            res_o = PyTuple_GET_ITEM(pair, 0);
            num_o = PyTuple_GET_ITEM(pair, 1);
        } else {
            seq.p = PySequence_Fast(pair, "hex entries must be (resource, number) pairs");
            if (!seq.p) throw pybind11::error_already_set();
            if (PySequence_Fast_GET_SIZE(seq.p) != 2) bad_state("hex entries must be (resource, number) pairs");
            res_o = PySequence_Fast_GET_ITEM(seq.p, 0);
            num_o = PySequence_Fast_GET_ITEM(seq.p, 1);
        }
        const long res = to_long(res_o, "hex resource");
        if (res < 0 || res > DESERT) bad_state("hex resource " + std::to_string(res) + " out of range");
        s.hex_res[i] = (int8_t)res;
        if (res == DESERT) {  // the desert's number is never read by the Python code (it may be None)
            s.hex_num[i] = 0;
            continue;
        }
        const long num = to_long(num_o, "hex number");
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
    tr.proposer = get_int32(t.p, N.proposer);
    read_exact5(t.p, N.give, tr.give, "TradeOffer.give");
    read_exact5(t.p, N.get, tr.get, "TradeOffer.get");
    for (int i = 0; i < MAX_PLAYERS; ++i) tr.responses[i] = -1;
    if (PyObject_HasAttr(t.p, N.responses)) {
        Ref r(getattr(t.p, N.responses));
        if (PyDict_Check(r.p)) {
            PyObject *key, *val;
            Py_ssize_t pos = 0;
            while (PyDict_Next(r.p, &pos, &key, &val)) {
                const long k = key_to_long(key);
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
// Throws pybind11::error_already_set on Python errors and catanbot::unsupported_state
// (-> catanbot_core.UnsupportedStateError, a ValueError) on states the struct cannot hold
// (wrong list lengths, ids out of range, > MAX_PLAYERS players, integers outside 32 bits).
inline void state_from_python(PyObject* obj, GameStateC& s) {
    using namespace detail;
    const Names& N = names();
    s = GameStateC{};
    read_hexes(obj, s);
    s.robber = get_int32(obj, N.robber);
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
    s.current = get_int32(obj, N.current);
    s.phase = read_phase(obj);
    s.turn = get_int32(obj, N.turn);
    s.setup_round = get_int32(obj, N.setup_round);
    s.setup_last_settlement = get_int32(obj, N.setup_last_settlement);
    s.dice = get_int32(obj, N.dice);
    s.dev_played_this_turn = get_bool(obj, N.dev_played_this_turn);
    s.free_roads = get_int32(obj, N.free_roads);
    {
        Ref q(getattr(obj, N.discard_queue));
        s.n_discard = read_ints(q.p, s.discard_queue, MAX_DISCARD_QUEUE, "GameState.discard_queue");
    }
    read_trade(obj, s);
    s.trade_responder = get_int32(obj, N.trade_responder);
    s.trades_this_turn = get_int32(obj, N.trades_this_turn);
    s.longest_road_owner = get_int32(obj, N.longest_road_owner);
    s.longest_road_len = get_int32(obj, N.longest_road_len);
    s.largest_army_owner = get_int32(obj, N.largest_army_owner);
    s.winner = get_int32(obj, N.winner);
    s.max_turns = get_int32(obj, N.max_turns);
}

inline GameStateC state_from_python(pybind11::handle obj) {
    GameStateC s;
    state_from_python(obj.ptr(), s);
    return s;
}

}  // namespace catanbot
#endif  // CATAN_NO_PYTHON
