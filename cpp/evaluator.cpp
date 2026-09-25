// Evaluators for the native search.  See evaluator.hpp.
#define CATAN_NO_PYTHON
#include "evaluator.hpp"
#undef CATAN_NO_PYTHON

#include <cmath>
#include <stdexcept>

namespace catanbot {

void Evaluator::evaluate_batch(const GameStateC* const* states, const int* players, int n, double* out) const {
    for (int i = 0; i < n; ++i) out[i] = evaluate(*states[i], players[i]);
}

// ---------------------------------------------------------------------------
// HeuristicEvaluator
// ---------------------------------------------------------------------------
double HeuristicEval::evaluate(const GameStateC& s, int player) const {
    double vals[MAX_PLAYERS];
    static_values(s, vals);
    return heuristic_win_probability(s, vals, player, temperature);
}

// ---------------------------------------------------------------------------
// ValueNet
// ---------------------------------------------------------------------------
namespace {

// Dot product of float32 vectors accumulated in double: eight interleaved partial sums combined
// pairwise (a fixed, machine-independent order; products of two float32 are exact in double).
inline double dot8(const float* a, const float* w, int n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 += (double)a[i] * (double)w[i];
        s1 += (double)a[i + 1] * (double)w[i + 1];
        s2 += (double)a[i + 2] * (double)w[i + 2];
        s3 += (double)a[i + 3] * (double)w[i + 3];
        s4 += (double)a[i + 4] * (double)w[i + 4];
        s5 += (double)a[i + 5] * (double)w[i + 5];
        s6 += (double)a[i + 6] * (double)w[i + 6];
        s7 += (double)a[i + 7] * (double)w[i + 7];
    }
    for (; i < n; ++i) s0 += (double)a[i] * (double)w[i];
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// model._sigmoid on a float32 logit: computed in double, narrowed to float32 like predict()'s output.
inline double sigmoid32(float z) {
    const double zd = (double)z;
    double p;
    if (zd >= 0.0) {
        p = 1.0 / (1.0 + std::exp(-zd));
    } else {
        const double ez = std::exp(zd);
        p = ez / (1.0 + ez);
    }
    return (double)(float)p;
}

}  // namespace

MlpEval::MlpEval(std::vector<std::vector<float>> Wt_, std::vector<std::vector<float>> b_, std::vector<float> mean_,
                 std::vector<float> std_dev, std::vector<int> sizes_)
    : sizes(std::move(sizes_)), Wt(std::move(Wt_)), b(std::move(b_)), mean(std::move(mean_)), std_(std::move(std_dev)) {
    if (sizes.size() < 2) throw std::invalid_argument("MlpEval needs at least one layer");
    n_in = sizes[0];
    if (n_in != NUM_FEATURES) throw std::invalid_argument("MlpEval: the net does not take NUM_FEATURES inputs");
    if (sizes.back() != 1) throw std::invalid_argument("MlpEval: the last layer must have one output");
    const size_t layers = sizes.size() - 1;
    if (Wt.size() != layers || b.size() != layers) throw std::invalid_argument("MlpEval: layer count mismatch");
    for (size_t l = 0; l < layers; ++l) {
        if (Wt[l].size() != (size_t)sizes[l] * (size_t)sizes[l + 1] || b[l].size() != (size_t)sizes[l + 1])
            throw std::invalid_argument("MlpEval: weight shape mismatch");
    }
    if (mean.size() != (size_t)n_in || std_.size() != (size_t)n_in)
        throw std::invalid_argument("MlpEval: normalisation shape mismatch");
}

float MlpEval::logit_from_features(const float* x) const {
    std::vector<float> h((size_t)n_in), h2;
    for (int i = 0; i < n_in; ++i) {  // (X - mean) / std in float32, like numpy
        const float d = x[i] - mean[i];
        h[i] = d / std_[i];
    }
    const int layers = (int)sizes.size() - 1;
    for (int l = 0; l < layers; ++l) {
        const int in = sizes[l], out = sizes[l + 1];
        h2.assign((size_t)out, 0.0f);
        const float* W = Wt[l].data();
        const float* bias = b[l].data();
        for (int j = 0; j < out; ++j) {
            float y = (float)dot8(h.data(), W + (size_t)j * (size_t)in, in);  // H @ W -> float32
            y = y + bias[j];                                                  // + b (float32)
            if (l < layers - 1 && y < 0.0f) y = 0.0f;                         // np.maximum(H, 0.0)
            h2[j] = y;
        }
        h.swap(h2);
    }
    return h[0];
}

float MlpEval::logit(const GameStateC& s, int player) const {
    Analysis an;
    analyse(s, an);
    float x[NUM_FEATURES];
    assemble(s, an, player, x);
    return logit_from_features(x);
}

double MlpEval::evaluate(const GameStateC& s, int player) const { return sigmoid32(logit(s, player)); }

void MlpEval::evaluate_batch(const GameStateC* const* states, const int* players, int n, double* out) const {
    Analysis an;
    const GameStateC* last = nullptr;
    float x[NUM_FEATURES];
    for (int i = 0; i < n; ++i) {
        if (states[i] != last) {  // consecutive identical states share one analysis
            analyse(*states[i], an);
            last = states[i];
        }
        assemble(*states[i], an, players[i], x);
        out[i] = sigmoid32(logit_from_features(x));
    }
}

// ---------------------------------------------------------------------------
// BlendedEvaluator
// ---------------------------------------------------------------------------
double BlendEval::evaluate(const GameStateC& s, int player) const {
    const double vn = net->evaluate(s, player);
    if (alpha >= 1.0) return vn;
    const double vh = heuristic->evaluate(s, player);
    return alpha * vn + (1.0 - alpha) * vh;
}

}  // namespace catanbot
