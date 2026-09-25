// C++ port of catanbot/features.py (feature extraction).  See features.cpp.
#pragma once

#include "board_tables.hpp"
#include "feature_layout.hpp"
#include "state.hpp"

namespace catanbot {

// Everything features.py computes once per state (_analyse_players, _hidden_vp_estimates,
// _player_blocks).  Perspective-dependent parts are produced by `assemble`.
struct Analysis {
    double prod_tab[NUM_VERTICES][NUM_RESOURCES];    // expected cards per roll of a settlement, robber ignored
    double prod_r_tab[NUM_VERTICES][NUM_RESOURCES];  // same, the robber hex pays nothing
    double pips[NUM_VERTICES];                       // pip total per vertex
    double prod[MAX_PLAYERS][NUM_RESOURCES];         // robber-aware production per player
    double prod_nr[MAX_PLAYERS][NUM_RESOURCES];      // production ignoring the robber
    double robber_touch[MAX_PLAYERS];
    int spots_now[MAX_PLAYERS], spots_one[MAX_PLAYERS], spots_two[MAX_PLAYERS];
    double best_now[MAX_PLAYERS], best_two[MAX_PLAYERS];
    int lr[MAX_PLAYERS];                             // longest road per player
    double hidden[MAX_PLAYERS];                      // expected VP cards
    float blocks[MAX_PLAYERS][PLAYER_BLOCK];         // one player block per seat
};

// features.longest_road_length: longest trail over the player's roads (no edge reused,
// vertices may repeat, an opponent building ends the path).
int longest_road_length(const GameStateC& s, int player);

// Per-state analysis + all player blocks (features._analyse_players + _player_blocks).
void analyse(const GameStateC& s, Analysis& a);

// Write the NUM_FEATURES-float vector from `player`'s perspective (features._assemble).
void assemble(const GameStateC& s, const Analysis& a, int player, float* out);

}  // namespace catanbot
