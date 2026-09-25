// Leaf / trial evaluators for the native search (see evaluator.cpp): the C++ twins of
// heuristic.HeuristicEvaluator, model.ValueNet and selfplay.BlendedEvaluator, evaluating
// GameStateC positions without any Python object in the loop.
#pragma once

#include <memory>
#include <vector>

#include "features.hpp"
#include "heuristic.hpp"
#include "state.hpp"

namespace catanbot {

struct Evaluator {
    virtual ~Evaluator() = default;
    // Win probability of `player` in `s` (the evaluator's value in [0, 1]).
    virtual double evaluate(const GameStateC& s, int player) const = 0;
    // Same for several (state, player) pairs; the default loops over evaluate().
    virtual void evaluate_batch(const GameStateC* const* states, const int* players, int n, double* out) const;
};

// HeuristicEvaluator(temperature): softmax over static_values (1 / 0 once the game is decided).
struct HeuristicEval : Evaluator {
    double temperature;
    explicit HeuristicEval(double t = 16.0) : temperature(t) {}
    double evaluate(const GameStateC& s, int player) const override;
};

// ValueNet (numpy float32 MLP): features -> standardise -> ReLU layers -> sigmoid.  Weights are
// stored transposed (out-major) as float32; the dot products accumulate float32 products in
// double (eight interleaved partial sums, a fixed order on every machine) and are rounded to
// float32 at the layer boundary, where numpy's sgemm result is float32 as well.  The results
// agree with ValueNet.predict to float32 precision, not bit for bit (OpenBLAS's blocked
// accumulation order cannot be reproduced).
struct MlpEval : Evaluator {
    int n_in = 0;
    std::vector<int> sizes;                  // n_in, hidden..., 1
    std::vector<std::vector<float>> Wt;      // per layer: out x in, row-major (transposed numpy W)
    std::vector<std::vector<float>> b;       // per layer: out
    std::vector<float> mean, std_;           // n_in
    MlpEval(std::vector<std::vector<float>> Wt_, std::vector<std::vector<float>> b_, std::vector<float> mean_,
            std::vector<float> std_dev, std::vector<int> sizes_);
    // The float32 logit of the standardised feature vector (`x` has n_in raw features).
    float logit_from_features(const float* x) const;
    float logit(const GameStateC& s, int player) const;
    double evaluate(const GameStateC& s, int player) const override;
    void evaluate_batch(const GameStateC* const* states, const int* players, int n, double* out) const override;
};

// BlendedEvaluator(net, alpha, heuristic): alpha * net + (1 - alpha) * heuristic (the net alone at alpha >= 1).
struct BlendEval : Evaluator {
    std::shared_ptr<const Evaluator> net;
    double alpha;
    std::shared_ptr<const Evaluator> heuristic;
    BlendEval(std::shared_ptr<const Evaluator> n, double a, std::shared_ptr<const Evaluator> h)
        : net(std::move(n)), alpha(a), heuristic(std::move(h)) {}
    double evaluate(const GameStateC& s, int player) const override;
};

}  // namespace catanbot
