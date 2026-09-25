// Native lookahead: opponents' greedy turns + reduced our-turn search.  See search.hpp.
//
// greedy_turn / simulate_until_my_turn / autoplay_others mirror Searcher._greedy_turn /
// _simulate_until_my_turn / _autoplay_others statement by statement (with the documented
// differences: every non-trade legal action is a candidate, no opponent trade proposals,
// PHASE_TRADE_RESPONSE of another player is answered with REJECT - unreachable without
// proposals).  reduced_search mirrors Searcher.search without trade candidates; its move
// ordering is the afterstate value instead of heuristic.action_priors.
#define CATAN_NO_PYTHON
#include "search.hpp"
#undef CATAN_NO_PYTHON

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

namespace catanbot {

namespace {

// Forwards to the draw source and remembers the last randrange index (trace / replay).
struct RecordingDraw : DrawSource {
    DrawSource& inner;
    int last = -1;
    explicit RecordingDraw(DrawSource& d) : inner(d) {}
    int randrange(int n) override { return last = inner.randrange(n); }
    int randint(int lo, int hi) override { return inner.randint(lo, hi); }
};

inline double now_seconds() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

inline uint64_t mix_seed(uint64_t seed, int level, int si, int ri) {
    return seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(((level + 1) * 65536 + si) * 4096 + ri + 1);
}

inline bool same_action(const ActionC& a, const ActionC& b) {
    if (a.kind != b.kind || a.nargs != b.nargs) return false;
    if (a.kind == ACT_DISCARD || a.kind == ACT_PROPOSE_TRADE) {
        for (int r = 0; r < NUM_RESOURCES; ++r)
            if (a.vec1[r] != b.vec1[r] || a.vec2[r] != b.vec2[r]) return false;
        return true;
    }
    return a.a == b.a && a.b == b.b;
}

inline bool contains(const ActionList& legal, const ActionC& a) {
    for (int i = 0; i < legal.n; ++i)
        if (same_action(legal.items[i], a)) return true;
    return false;
}

inline bool has_kind(const ActionList& legal, ActionKind k) {
    for (int i = 0; i < legal.n; ++i)
        if (legal.items[i].kind == k) return true;
    return false;
}

inline bool is_finished(const GameStateC& s, int me) { return s.phase == PHASE_GAME_OVER || s.current != me; }

// Searcher._apply: apply + node count (+ the trace entry when `sink` is given).
void apply_counted(SearchCtx& ctx, GameStateC& s, ActionC a, DrawSource& draw, int player,
                   std::vector<TraceStep>* sink) {
    RecordingDraw rec(draw);
    bool rolled = false;
    apply_inplace(s, a, rec, &rolled);
    ++ctx.nodes;
    if (sink != nullptr) {
        if (rolled && a.nargs == 0) {  // record the rolled value so a replay needs no draw
            a.nargs = 1;
            a.a = s.dice;
        }
        sink->push_back(TraceStep{ctx.cur_state, ctx.cur_sample, player, a, rec.last});
    }
}

inline std::vector<TraceStep>* trace_sink(SearchCtx& ctx) { return ctx.trace ? &ctx.steps : nullptr; }

// Searcher._autoplay_others(s, j): forced decisions of other players inside j's turn.
void autoplay_others(SearchCtx& ctx, GameStateC& s, int j, DrawSource& draw, std::vector<TraceStep>* sink) {
    std::unique_ptr<ActionList> legal(new ActionList());
    int guard = 0;
    while (s.phase != PHASE_GAME_OVER && s.current == j && acting_player(s) != j && guard < 12) {
        const int acting = acting_player(s);
        legal_actions(s, *legal);
        if (legal->n == 0) break;
        ActionC a;
        if (s.phase == PHASE_DISCARD) a = choose_discard_action(s, acting, legal.get());
        else if (s.phase == PHASE_TRADE_RESPONSE && s.has_pending_trade) a = make_action(ACT_REJECT_TRADE);
        else a = legal->items[0];
        apply_counted(ctx, s, a, draw, acting, sink);
        ++guard;
    }
}

}  // namespace

bool budget_exhausted(const SearchCtx& ctx) {
    if (ctx.node_budget >= 0 && ctx.nodes >= ctx.node_budget) return true;
    if (ctx.deadline > 0.0 && now_seconds() > ctx.deadline) return true;
    return false;
}

int roll_distribution(int samples, int* totals, double* probs) {
    static const int ORDER[11] = {7, 6, 8, 5, 9, 4, 10, 3, 11, 2, 12};  // search._ROLL_ORDER
    auto prob_of = [](int v) { return v == 7 ? 6.0 / 36.0 : (double)PIPS[v] / 36.0; };  // board.ROLL_PROB
    if (samples >= 11) {
        for (int v = 2; v <= 12; ++v) {
            totals[v - 2] = v;
            probs[v - 2] = prob_of(v);
        }
        return 11;
    }
    const int n = std::max(1, samples);
    double tot = 0.0;
    for (int k = 0; k < n; ++k) tot += prob_of(ORDER[k]);
    for (int k = 0; k < n; ++k) {
        totals[k] = ORDER[k];
        probs[k] = prob_of(ORDER[k]) / tot;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Searcher._greedy_turn
// ---------------------------------------------------------------------------
void greedy_turn(SearchCtx& ctx, GameStateC& s, int j, int roll, int me, const LevelCfg& cfg, DrawSource& draw) {
    (void)me;  // the Python signature carries it; the policy does not depend on it
    std::unique_ptr<ActionList> legal(new ActionList());
    std::vector<TraceStep>* sink = trace_sink(ctx);
    int steps = 0, acted = 0;
    const int max_steps = 3 * cfg.opponent_actions + 6;
    std::vector<TraceStep> tmp, best_steps;
    while (s.phase != PHASE_GAME_OVER && s.current == j && steps < max_steps) {
        ++steps;
        const int acting = acting_player(s);
        legal_actions(s, *legal);
        if (legal->n == 0) break;
        if (acting != j) {
            // Forced decision of another player (maybe us): heuristics.
            ActionC a;
            if (s.phase == PHASE_DISCARD) a = choose_discard_action(s, acting, legal.get());
            else if (s.phase == PHASE_TRADE_RESPONSE && s.has_pending_trade) a = make_action(ACT_REJECT_TRADE);
            else a = legal->items[0];
            apply_counted(ctx, s, a, draw, acting, sink);
            continue;
        }
        if (s.phase == PHASE_ROLL) {
            apply_counted(ctx, s, roll ? make_action(ACT_ROLL, roll) : make_action(ACT_ROLL), draw, j, sink);
            continue;
        }
        if (s.phase == PHASE_DISCARD) {
            apply_counted(ctx, s, choose_discard_action(s, j, legal.get()), draw, j, sink);
            continue;
        }
        if (s.phase == PHASE_ROBBER) {
            double tw[MAX_PLAYERS];
            const double* twp = nullptr;
            if (ctx.rw != nullptr && ctx.rw->mode != 0) {
                robber_weights(s, j, *ctx.rw, tw);
                twp = tw;
            }
            const RobberMove m = best_robber_move(s, j, twp);
            ActionC a = make_action(ACT_MOVE_ROBBER, m.hex, m.victim);
            if (!contains(*legal, a)) a = legal->items[0];
            apply_counted(ctx, s, a, draw, j, sink);
            continue;
        }
        if (s.phase == PHASE_TRADE_SELECT) {
            ActionC a = legal->items[0];
            for (int i = 0; i < legal->n; ++i)
                if (legal->items[i].kind == ACT_EXECUTE_TRADE) {
                    a = legal->items[i];
                    break;
                }
            apply_counted(ctx, s, a, draw, j, sink);
            continue;
        }
        // Main phase: greedy one-step lookahead with the value function from j's perspective.
        if (acted >= cfg.opponent_actions || legal->n == 1) {
            apply_counted(ctx, s, has_kind(*legal, ACT_END_TURN) ? make_action(ACT_END_TURN) : legal->items[0], draw, j,
                          sink);
            continue;
        }
        double best_val = 0.0;
        int best_k = -1;
        GameStateC best_state;
        int n_trials = 0;
        for (int k = 0; k < legal->n; ++k) {
            const ActionC& a = legal->items[k];
            if (a.kind == ACT_PROPOSE_TRADE) continue;
            GameStateC s2 = s;
            tmp.clear();
            std::vector<TraceStep>* tsink = ctx.trace ? &tmp : nullptr;
            apply_counted(ctx, s2, a, draw, j, tsink);
            autoplay_others(ctx, s2, j, draw, tsink);
            const double v = ctx.ev->evaluate(s2, j);
            if (n_trials == 0 || v > best_val) {  // max(..., key=value): the first maximum
                best_val = v;
                best_k = k;
                best_state = s2;
                if (ctx.trace) best_steps = tmp;
            }
            ++n_trials;
        }
        if (n_trials == 0) break;
        s = best_state;
        if (sink != nullptr) sink->insert(sink->end(), best_steps.begin(), best_steps.end());
        ++acted;
        if (legal->items[best_k].kind == ACT_END_TURN) break;
    }
    if (s.phase != PHASE_GAME_OVER && s.current == j) {
        legal_actions(s, *legal);
        if (has_kind(*legal, ACT_END_TURN)) apply_counted(ctx, s, make_action(ACT_END_TURN), draw, j, sink);
    }
}

// ---------------------------------------------------------------------------
// Searcher._simulate_until_my_turn
// ---------------------------------------------------------------------------
void simulate_until_my_turn(SearchCtx& ctx, GameStateC& s, int me, const std::vector<int>& rolls, const LevelCfg& cfg,
                            DrawSource& draw) {
    int ri = 0, guard = 0;
    while (s.phase != PHASE_GAME_OVER && !(s.current == me && s.phase == PHASE_ROLL) && guard < 6) {
        const int roll = (s.phase == PHASE_ROLL && !rolls.empty()) ? rolls[(size_t)ri % rolls.size()] : 0;
        greedy_turn(ctx, s, s.current, roll, me, cfg, draw);
        ++ri;
        ++guard;
    }
}

// ---------------------------------------------------------------------------
// reduced Searcher.search (no trades)
// ---------------------------------------------------------------------------
namespace {

struct RChild {
    ActionC action;
    std::vector<std::pair<double, int>> kids;  // (probability, node index)
};

struct RNode {
    GameStateC state;
    double prob = 1.0;
    double stat = 0.0;
    double value = 0.0;
    bool has_value = false;
    bool finished = false;
    int group = 0;
    std::vector<RChild> children;
};

// Searcher._candidates without trades: force_end / single action / discards (choose_discard's
// pick first, then by L1 distance) / everything else ordered by its afterstate value.
void candidates(SearchCtx& ctx, const GameStateC& s, int me, bool force_end, const LevelCfg& cfg, DrawSource& draw,
                std::vector<ActionC>& out) {
    out.clear();
    std::unique_ptr<ActionList> legal(new ActionList());
    legal_actions(s, *legal);
    if (legal->n == 0) return;
    if (force_end && has_kind(*legal, ACT_END_TURN)) {
        out.push_back(make_action(ACT_END_TURN));
        return;
    }
    if (legal->n == 1) {
        out.push_back(legal->items[0]);
        return;
    }
    if (s.phase == PHASE_DISCARD) {
        const ActionC pick = choose_discard_action(s, me, legal.get());
        std::vector<std::pair<long, int>> order;  // (L1 distance to the pick, legal index)
        for (int i = 0; i < legal->n; ++i) {
            const ActionC& a = legal->items[i];
            if (a.kind != ACT_DISCARD) continue;
            long d = 0;
            for (int r = 0; r < NUM_RESOURCES; ++r) d += std::labs((long)a.vec1[r] - (long)pick.vec1[r]);
            order.push_back({d, i});
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const std::pair<long, int>& x, const std::pair<long, int>& y) { return x.first < y.first; });
        const int cap = std::max(1, cfg.discard_candidates);
        for (size_t k = 0; k < order.size() && (int)k < cap; ++k) out.push_back(legal->items[order[k].second]);
        return;
    }
    struct Cand {
        int idx;
        double val;
        bool ok;
    };
    std::vector<Cand> cands;
    cands.reserve((size_t)legal->n);
    for (int i = 0; i < legal->n; ++i) {
        const ActionC& a = legal->items[i];
        if (a.kind == ACT_PROPOSE_TRADE) continue;
        Cand c{i, 0.0, true};
        if (a.kind == ACT_ROLL) {
            c.val = 1e9;  // rolling always comes first
        } else {
            GameStateC s2 = s;
            try {
                apply_counted(ctx, s2, a, draw, me, nullptr);
                c.val = ctx.ev->evaluate(s2, me);
            } catch (const illegal_action&) {
                c.ok = false;
            }
        }
        if (c.ok) cands.push_back(c);
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.val > y.val; });
    const int cap = std::max(1, cfg.expand);
    for (size_t k = 0; k < cands.size() && (int)k < cap; ++k) out.push_back(legal->items[cands[k].idx]);
    if (has_kind(*legal, ACT_END_TURN)) {
        bool present = false;
        for (const ActionC& a : out)
            if (a.kind == ACT_END_TURN) present = true;
        if (!present) out.push_back(make_action(ACT_END_TURN));
    }
}

// Searcher._outcomes without trades: exact chance nodes for the roll, a dev card purchase and
// a robber steal (forced draws), then _autoplay_others on every outcome.
void outcomes(SearchCtx& ctx, const GameStateC& s, const ActionC& a, int me, const LevelCfg& cfg, DrawSource& draw,
              std::vector<std::pair<double, GameStateC>>& outs) {
    outs.clear();
    if (a.kind == ACT_ROLL && a.nargs == 0) {
        int totals[11];
        double probs[11];
        const int n = roll_distribution(cfg.roll_samples, totals, probs);
        for (int k = 0; k < n; ++k) {
            GameStateC s2 = s;
            apply_counted(ctx, s2, make_action(ACT_ROLL, totals[k]), draw, me, nullptr);
            outs.push_back({probs[k], s2});
        }
    } else if (a.kind == ACT_BUY_DEV && a.nargs == 0) {
        int total = 0;
        for (int t = 0; t < NUM_DEV; ++t) total += s.dev_deck[t];
        if (total <= 0) {
            GameStateC s2 = s;
            apply_counted(ctx, s2, a, draw, me, nullptr);  // raises "development deck is empty"
            outs.push_back({1.0, s2});
        } else {
            int cum = 0;
            for (int t = 0; t < NUM_DEV; ++t) {
                const int c = s.dev_deck[t];
                if (c > 0) {
                    GameStateC s2 = s;
                    ForcedDraw fd(cum);
                    apply_inplace(s2, a, fd, nullptr);
                    ++ctx.nodes;
                    outs.push_back({(double)c / (double)total, s2});
                }
                cum += c;
            }
        }
    } else if ((a.kind == ACT_MOVE_ROBBER || a.kind == ACT_PLAY_KNIGHT) && a.nargs >= 2 && a.b >= 0 &&
               a.b < s.num_players) {
        const PlayerC& victim = s.players[a.b];
        const int total = victim.total_resources();
        if (total <= 0) {
            GameStateC s2 = s;
            apply_counted(ctx, s2, a, draw, me, nullptr);
            outs.push_back({1.0, s2});
        } else {
            int cum = 0;
            for (int r = 0; r < NUM_RESOURCES; ++r) {
                const int c = victim.resources[r];
                if (c > 0) {
                    GameStateC s2 = s;
                    ForcedDraw fd(cum);
                    apply_inplace(s2, a, fd, nullptr);
                    ++ctx.nodes;
                    outs.push_back({(double)c / (double)total, s2});
                }
                cum += c;
            }
        }
    } else {
        GameStateC s2 = s;
        apply_counted(ctx, s2, a, draw, me, nullptr);
        outs.push_back({1.0, s2});
    }
    for (auto& o : outs) autoplay_others(ctx, o.second, me, draw, nullptr);
}

double backup(std::vector<RNode>& nodes, int i, double shift) {
    RNode& n = nodes[i];
    if (n.children.empty()) {
        if (!n.has_value) {
            n.value = std::min(1.0, std::max(0.0, n.stat + shift));
            n.has_value = true;
        }
        return n.value;
    }
    double best = -1.0;
    for (const RChild& ch : n.children) {
        double v = 0.0;
        for (const auto& pk : ch.kids) v += pk.first * backup(nodes, pk.second, shift);
        if (v > best) best = v;
    }
    nodes[i].value = best;
    nodes[i].has_value = true;
    return best;
}

}  // namespace

double reduced_search(SearchCtx& ctx, const GameStateC& root, int me, int depth, int L, DrawSource& draw) {
    const std::vector<LevelCfg>& levels = *ctx.levels;
    if (L >= (int)levels.size()) return ctx.ev->evaluate(root, me);
    const LevelCfg& cfg = levels[L];
    const long nodes_start = ctx.nodes;
    auto exhausted = [&]() { return budget_exhausted(ctx) || (ctx.nodes - nodes_start) >= cfg.max_nodes; };
    {
        std::unique_ptr<ActionList> legal(new ActionList());
        legal_actions(root, *legal);
        if (legal->n == 0) return ctx.ev->evaluate(root, me);
        if (legal->n == 1 && legal->items[0].kind != ACT_ROLL) return ctx.ev->evaluate(root, me);
    }
    const bool saved_trace = ctx.trace;
    ctx.trace = false;  // the trace covers the top-level simulations only
    std::vector<RNode> nodes;
    nodes.reserve(64);
    nodes.emplace_back();
    nodes[0].state = root;
    nodes[0].prob = 1.0;
    std::vector<int> frontier{0}, finished;
    double shift = 0.0;
    int group_id = 0;
    std::vector<ActionC> cands;
    std::vector<std::pair<double, GameStateC>> outs;
    for (int level = 0; level <= cfg.max_actions_per_turn; ++level) {
        if (frontier.empty() || exhausted()) break;
        const bool force_end = level >= cfg.max_actions_per_turn;
        std::vector<int> new_nodes;
        for (const int ni : frontier) {
            candidates(ctx, nodes[ni].state, me, force_end, cfg, draw, cands);
            for (const ActionC& a : cands) {
                try {
                    outcomes(ctx, nodes[ni].state, a, me, cfg, draw, outs);
                } catch (const illegal_action&) {
                    continue;
                }
                ++group_id;
                std::vector<std::pair<double, int>> kids;
                const double parent_prob = nodes[ni].prob;
                for (auto& o : outs) {
                    RNode child;
                    child.state = o.second;
                    child.prob = parent_prob * o.first;
                    child.finished = is_finished(o.second, me);
                    child.group = group_id;
                    nodes.push_back(std::move(child));
                    const int ci = (int)nodes.size() - 1;
                    kids.push_back({o.first, ci});
                    new_nodes.push_back(ci);
                }
                if (!kids.empty()) nodes[ni].children.push_back(RChild{a, std::move(kids)});
            }
        }
        if (new_nodes.empty()) break;
        for (const int ci : new_nodes) nodes[ci].stat = ctx.ev->evaluate(nodes[ci].state, me);
        std::vector<int> unfinished;
        for (const int ci : new_nodes) {
            if (nodes[ci].finished) finished.push_back(ci);
            else unfinished.push_back(ci);
        }
        std::stable_sort(unfinished.begin(), unfinished.end(),
                         [&](int x, int y) { return nodes[x].stat > nodes[y].stat; });
        // Beam by node, but never split a chance node.
        std::vector<int> keep_groups;
        for (size_t k = 0; k < unfinished.size() && (int)k < cfg.beam; ++k) keep_groups.push_back(nodes[unfinished[k]].group);
        frontier.clear();
        for (const int ci : unfinished)
            if (std::find(keep_groups.begin(), keep_groups.end(), nodes[ci].group) != keep_groups.end())
                frontier.push_back(ci);
    }
    if (depth >= 2 && !finished.empty() && !exhausted()) {
        std::stable_sort(finished.begin(), finished.end(), [&](int x, int y) {
            const double kx = nodes[x].stat + 0.05 * std::log(std::max(nodes[x].prob, 1e-6));
            const double ky = nodes[y].stat + 0.05 * std::log(std::max(nodes[y].prob, 1e-6));
            return kx > ky;
        });
        const size_t n_top = std::min(finished.size(), (size_t)std::max(0, cfg.finished_lookahead));
        if (n_top > 0) {
            std::vector<GameStateC> top_states;
            for (size_t k = 0; k < n_top; ++k) top_states.push_back(nodes[finished[k]].state);
            const int n_samples = std::max(1, cfg.opp_roll_samples);
            std::vector<std::vector<int>> rolls((size_t)n_samples, std::vector<int>(12, 0));
            for (int k = 0; k < n_samples; ++k)
                for (int t = 0; t < 12; ++t) rolls[k][t] = draw.randint(1, 6) + draw.randint(1, 6);
            std::vector<double> fv;
            future_values(ctx, top_states, me, depth - 1, L, rolls, fv, nullptr, nullptr);
            double acc = 0.0;
            for (size_t k = 0; k < n_top; ++k) {
                RNode& n = nodes[finished[k]];
                n.value = fv[k];
                n.has_value = true;
                acc += n.value - n.stat;
            }
            shift = acc / (double)n_top;
        }
    }
    ctx.trace = saved_trace;
    if (nodes[0].children.empty()) return ctx.ev->evaluate(root, me);
    return backup(nodes, 0, shift);
}

// ---------------------------------------------------------------------------
// Searcher._future_values
// ---------------------------------------------------------------------------
void future_values(SearchCtx& ctx, const std::vector<GameStateC>& states, int me, int depth, int L,
                   const std::vector<std::vector<int>>& rolls, std::vector<double>& out,
                   std::vector<GameStateC>* leaves_out, std::vector<double>* leaf_values_out) {
    const std::vector<LevelCfg>& levels = *ctx.levels;
    const LevelCfg& cfg = levels[std::min(L, (int)levels.size() - 1)];
    const int n_states = (int)states.size();
    const int n_samples = std::max(1, (int)rolls.size());
    const int saved_state = ctx.cur_state, saved_sample = ctx.cur_sample;
    std::vector<GameStateC> leaves;
    leaves.reserve((size_t)n_states * (size_t)n_samples);
    static const std::vector<int> no_rolls;
    for (int si = 0; si < n_states; ++si) {
        for (int ri = 0; ri < n_samples; ++ri) {
            GameStateC s = states[si];
            ctx.cur_state = si;
            ctx.cur_sample = ri;
            Xoshiro xo(mix_seed(ctx.seed, L, si, ri));
            DrawSource& draw = ctx.shared_rng != nullptr ? *ctx.shared_rng : static_cast<DrawSource&>(xo);
            simulate_until_my_turn(ctx, s, me, rolls.empty() ? no_rolls : rolls[(size_t)ri], cfg, draw);
            leaves.push_back(s);
        }
    }
    std::vector<double> vals(leaves.size(), 0.0);
    if (depth >= 2 && !budget_exhausted(ctx)) {
        for (size_t k = 0; k < leaves.size(); ++k) {
            const GameStateC& leaf = leaves[k];
            if (leaf.phase == PHASE_GAME_OVER || acting_player(leaf) != me) {
                vals[k] = ctx.ev->evaluate(leaf, me);
                continue;
            }
            Xoshiro xo(mix_seed(ctx.seed, L + 64, (int)(k / (size_t)n_samples), (int)(k % (size_t)n_samples)));
            DrawSource& draw = ctx.shared_rng != nullptr ? *ctx.shared_rng : static_cast<DrawSource&>(xo);
            vals[k] = reduced_search(ctx, leaf, me, depth, L + 1, draw);
        }
    } else {
        for (size_t k = 0; k < leaves.size(); ++k) vals[k] = ctx.ev->evaluate(leaves[k], me);
    }
    out.assign((size_t)n_states, 0.0);
    for (size_t k = 0; k < vals.size(); ++k) out[k / (size_t)n_samples] += vals[k];
    for (int si = 0; si < n_states; ++si) out[(size_t)si] = out[(size_t)si] / (double)n_samples;
    if (leaves_out != nullptr) *leaves_out = std::move(leaves);
    if (leaf_values_out != nullptr) *leaf_values_out = std::move(vals);
    ctx.cur_state = saved_state;
    ctx.cur_sample = saved_sample;
}

}  // namespace catanbot
