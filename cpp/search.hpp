// Native lookahead for catanbot.search.Searcher._future_values (see search.cpp and docs/CPP.md).
//
//   future_values(states, me, depth)  ==  the opponents play their turns greedily (sampled dice
//   with common random numbers) until it is our roll again; the leaves are evaluated, or for
//   depth >= 2 searched with a reduced version of Searcher.search (our turn, no trades) whose
//   end-of-turn nodes recurse into future_values(depth - 1).
//
// Everything below lives on GameStateC; the Python side converts the batch of end-of-turn
// states once.  greedy_turn / simulate_until_my_turn mirror Searcher._greedy_turn /
// _simulate_until_my_turn step by step (with every non-trade legal action as a candidate and no
// opponent trade proposals); reduced_search mirrors Searcher.search without trade candidates.
#pragma once

#include <cstdint>
#include <vector>

#include "engine.hpp"
#include "evaluator.hpp"
#include "policy.hpp"
#include "state.hpp"

namespace catanbot {

// The SearchConfig fields the native side uses, one per lookahead level (search.reduced_config).
struct LevelCfg {
    int opponent_actions = 4;
    int beam = 6;
    int expand = 10;
    int max_actions_per_turn = 6;
    int roll_samples = 11;
    int opp_roll_samples = 12;
    int finished_lookahead = 0;      // 0 = every end-of-turn node gets the future value, N = the top N
    int discard_candidates = 3;
    long max_nodes = 40000;
    double lookahead_shrink = 12.0;  // k of search.lookahead_weight: a node's own delta counts n / (n + k)
};

// One applied action of a simulated opponent turn (trace mode): who acted, the action (rolls as
// (ROLL, value)) and the random index the action consumed (-1 = none), so a replay through the
// Python engine with a forced draw reproduces the state.
struct TraceStep {
    int state_idx;
    int sample_idx;
    int player;
    ActionC action;
    int draw;
};

struct SearchCtx {
    const std::vector<LevelCfg>* levels = nullptr;
    const Evaluator* ev = nullptr;
    const RobberWeights* rw = nullptr;   // nullptr / mode 0: best_robber_move's own weights
    DrawSource* shared_rng = nullptr;    // a Python random.Random (parity mode): one stream for everything
    uint64_t seed = 0;                   // else: xoshiro reseeded per (state, sample)
    double deadline = -1.0;              // time.time() value; < 0 = none
    long node_budget = -1;               // total applies allowed; < 0 = unlimited
    long nodes = 0;                      // applies performed (engine.apply calls in Python terms)
    bool trace = false;
    std::vector<TraceStep> steps;        // trace mode: every applied action of the simulations
    int cur_state = 0, cur_sample = 0;   // trace bookkeeping
};

// Budget check (deadline / node budget), Searcher._budget_exhausted.
bool budget_exhausted(const SearchCtx& ctx);

// Searcher._greedy_turn(s, j, roll, me) on `s` in place.
void greedy_turn(SearchCtx& ctx, GameStateC& s, int j, int roll, int me, const LevelCfg& cfg, DrawSource& draw);

// Searcher._simulate_until_my_turn(s, me, rolls) on `s` in place.
void simulate_until_my_turn(SearchCtx& ctx, GameStateC& s, int me, const std::vector<int>& rolls, const LevelCfg& cfg,
                            DrawSource& draw);

// Searcher._future_values(states, me, depth) with levels[L] as the caller's config; `rolls` are the
// sampled dice sequences (common random numbers).  Fills `out` (one value per state) and, when
// given, the leaves (states x samples) and their values.
void future_values(SearchCtx& ctx, const std::vector<GameStateC>& states, int me, int depth, int L,
                   const std::vector<std::vector<int>>& rolls, std::vector<double>& out,
                   std::vector<GameStateC>* leaves, std::vector<double>* leaf_values);

// Searcher.search(state, me)[0].value for a reduced searcher configured with levels[L]
// (no trade candidates); `depth` is that searcher's cfg.depth.
double reduced_search(SearchCtx& ctx, const GameStateC& root, int me, int depth, int L, DrawSource& draw);

// search.roll_distribution(samples): totals and renormalised probabilities.
int roll_distribution(int samples, int* totals, double* probs);

// search.lookahead_weight(cfg): n / (n + k) with n = max(1, opp_roll_samples), k = lookahead_shrink (1 when k <= 0).
double lookahead_weight(const LevelCfg& cfg);

// search.apply_lookahead: values of the lookahead nodes (static + shift + weight * (future - static - shift);
// a terminal node keeps its exact future value and stays out of the mean) and the mean shift for the other leaves.
double apply_lookahead(const std::vector<double>& statics, const std::vector<double>& futures,
                       const std::vector<bool>& terminal, double weight, std::vector<double>& values);

}  // namespace catanbot
