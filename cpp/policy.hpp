// C++ ports of the strategy helpers the search's opponent simulation needs
// (catanbot/discard.py, danger.py, robber.py, counting.py and the robber
// target-weight chain of politics.py / opponent_model.py).  See policy.cpp.
//
// Every function mirrors the Python function named in its comment with the
// same floating point operations in the same order, so the choices (discard
// vectors, robber hex / victim) and the numbers (win-path fields, target
// weights) are identical to the Python reference
// (tests/test_accel_search.py).  Keep this file in sync with those modules:
// there is no run-time layout check, only the differential tests.
#pragma once

#include <cstdint>

#include "board_tables.hpp"
#include "engine.hpp"
#include "feature_layout.hpp"
#include "heuristic.hpp"
#include "state.hpp"

namespace catanbot {

// counting.dev_pool / robber.estimated_vp / robber.threat / state.port_ratio
void dev_pool(const GameStateC& s, int out[NUM_DEV]);
double estimated_vp(const GameStateC& s, int i);
double threat(const GameStateC& s, int i);
int port_ratio_of(const GameStateC& s, int i, int res);

// counting.hand_prior_weights (smoothing 0.04)
void hand_prior_weights(const GameStateC& s, int player, double out[NUM_RESOURCES]);

// discard.default_keep_targets: cost vectors (pointers into the COST_* tables), most important first.
struct KeepTargets {
    const int* cost[4];
    int n = 0;
};
void default_keep_targets(const GameStateC& s, int player, KeepTargets& out);

// discard.needed_vector / discard.choose_discard.  `choose_discard` writes the raw pick;
// `choose_discard_action` applies the legal-list fallback (closest legal discard by L1 distance)
// when `legal` is given, exactly like the Python function with `legal_actions=`.
void needed_vector(const GameStateC& s, int player, int out[NUM_RESOURCES]);
void choose_discard(const GameStateC& s, int player, int32_t out[NUM_RESOURCES]);
ActionC choose_discard_action(const GameStateC& s, int player, const ActionList* legal);

// danger.WinPath (the numeric fields; the text `steps` / `rolls` are not mirrored).
struct WinPathC {
    double vp = 0.0;
    int need_vp = 0;
    double cost[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    double hand[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    bool hand_known = true;
    double missing[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    double need_share[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    double eff_need[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    double prod[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    double supply[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    double turns = 0.0;
    double min_turns = 0.0;
    bool can_win_now = false;
    double danger = 0.0;
    double blocked_now = 0.0;
    int n_steps = 0;        // len(steps) excluding the "no way to gain" entry
    bool no_path = false;   // the "no way to gain k VP" case (cost + 10)
};
struct WinPaths {
    WinPathC p[MAX_PLAYERS];
};
// danger.win_path(state, i) with no belief (hidden hands use the production prior).
void win_path(const GameStateC& s, int i, WinPathC& out);
// danger.win_paths(state): every player.
void win_paths(const GameStateC& s, WinPaths& out);
// danger.danger_multiplier / block_factor / steal_factor / rob_break_probability
double danger_multiplier(const WinPathC& wp);
double block_factor(const WinPathC& wp, int res, int pips);
double steal_factor(const WinPathC& wp, const double* our_need);
double rob_break_probability(const WinPathC& wp);

// robber.target_weight / our_need_indicator / hex_pips_for_player / hex_damage / steal_candidates /
// choose_victim / best_robber_move (hex, victim and score; the reason text is not mirrored).
double target_weight(const GameStateC& s, int i, const WinPaths& paths);
void our_need_indicator(const GameStateC& s, int player, double out[NUM_RESOURCES]);
int hex_pips_for_player(const GameStateC& s, int h, int i);
void hex_damage(const GameStateC& s, int h, int player, const double* target_weights, const WinPaths& paths,
                double& opp, double& own);
int steal_candidates(const GameStateC& s, int h, int player, int out[MAX_PLAYERS]);
int choose_victim(const GameStateC& s, int h, int player, const double* target_weights, const WinPaths& paths,
                  const double* our_need);
struct RobberMove {
    int hex = -1;
    int victim = -1;
    double score = -1e9;
};
RobberMove best_robber_move(const GameStateC& s, int player, const double* target_weights);

// The factor chain of politics.PoliticalState.robber_target_weights (mode 1) and
// OpponentModel.robber_habit_weights (mode 2), with everything that does not depend on the
// simulated state precomputed in Python (catanbot.accel.robber_weights_bundle) and the two
// state-dependent parts - target_weight / threat and the leader - recomputed here per state.
struct RobberWeights {
    int mode = 0;   // 0 = none (best_robber_move's own threat x danger), 1 = politics, 2 = habit
    int n = 0;
    bool has_habit = false;                           // an OpponentModel was given (mode 1 only)
    double c[MAX_PLAYERS][MAX_PLAYERS][4] = {};       // per (actor, victim): grudge, friend, habit base, ally factors
    double habit_leader[MAX_PLAYERS][MAX_PLAYERS] = {};   // per (actor, victim): factor when the victim leads
    double coal[MAX_PLAYERS][MAX_PLAYERS] = {};       // per (victim, leader): coalition factor (1.0 = none)
};
// politics.robber_target_weights(state, actor, model) / model.robber_habit_weights(state, actor)
void robber_weights(const GameStateC& s, int actor, const RobberWeights& rw, double out[MAX_PLAYERS]);

}  // namespace catanbot
