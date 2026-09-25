// C++ port of catanbot/heuristic.py::static_value and HeuristicEvaluator.evaluate,
// including the placement / counting helpers they use.  See heuristic.cpp.
//
// Every function mirrors the Python function named in its comment; the
// floating point operations are performed in the same order so the values are
// bit-identical to the Python reference (tests/test_accel_heuristic.py).
#pragma once

#include <cstdint>

#include "board_tables.hpp"
#include "feature_layout.hpp"
#include "state.hpp"

namespace catanbot {

// placement.RESOURCE_DEMAND (computed at run time with the same arithmetic as the Python list).
const double* resource_demand();

// state.occupied_vertices() / occupied_edges() and placement.is_free_vertex as tables.
struct Occupancy {
    int8_t owner[NUM_VERTICES];         // building owner per vertex, -1 = none (later players overwrite)
    uint8_t edge_taken[NUM_EDGES];      // 1 when some player has a road on the edge
    uint8_t free_vertex[NUM_VERTICES];  // is_free_vertex: v and all its neighbours are unoccupied
};
void occupancy(const GameStateC& s, Occupancy& o);

// placement.vertex_production / player_production / resource_scarcity
void vertex_production(const GameStateC& s, int v, bool ignore_robber, double out[NUM_RESOURCES]);
void player_production(const GameStateC& s, int player, bool ignore_robber, double out[NUM_RESOURCES]);
void resource_scarcity(const GameStateC& s, double out[NUM_RESOURCES]);

// counting.expected_hidden_vp (exact VP cards when the hand is known, else dev_count * pool share)
double expected_hidden_vp(const GameStateC& s, int player);

// heuristic.longest_road_length.  NOT features.longest_road_length: here a path may not
// *start* on a vertex holding an opponent's building (only end there) and a duplicated
// road id counts as a single road (the "used" set is keyed by edge id).
int heuristic_longest_road_length(const GameStateC& s, int player);

// placement.reachable_spots(state, player, max_roads, occ): free settlement spots reachable by
// building roads.  dist[v] = roads needed (0 = buildable now, -1 = not a reachable free spot),
// first[v] = first road to build towards it (-1 when none needed).  `count` = number of spots.
// The distances are exactly the Python ones; `first` can differ from Python when several first
// edges lead to a spot at the same distance (Python's choice depends on set iteration order).
struct ReachableSpots {
    int8_t dist[NUM_VERTICES];
    int8_t first[NUM_VERTICES];
    int count;
};
void reachable_spots(const GameStateC& s, int player, int max_roads, const Occupancy& occ, ReachableSpots& out);

// placement.score_settlement_spot(state, player, v, occ, own_prod, scarcity, setup)
double score_settlement_spot(const GameStateC& s, int player, int v, const Occupancy& occ,
                             const double own_prod[NUM_RESOURCES], const double scarcity[NUM_RESOURCES],
                             bool setup);

// heuristic._progress_to_build
double progress_to_build(const GameStateC& s, int player, const Occupancy& occ);

// heuristic.static_value for one player, and for every player (out gets s.num_players entries;
// the per-state work - occupancy, scarcity - is shared).
double static_value(const GameStateC& s, int player);
void static_values(const GameStateC& s, double* out);

// HeuristicEvaluator.evaluate for one (state, player) pair given that state's static values:
// heuristic_softmax is the softmax over the n players' values at `temperature`;
// heuristic_win_probability adds the terminal override (1 / 0 when the game is over with a winner).
double heuristic_softmax(const double* values, int n, int player, double temperature);
double heuristic_win_probability(const GameStateC& s, const double* values, int player, double temperature);

}  // namespace catanbot
