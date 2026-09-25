// Exact ports of the strategy helpers used by the native opponent simulation.  See policy.hpp.
//
// Sources (function names are kept): discard.py (default_keep_targets, _needed_tiers,
// needed_vector, choose_discard), counting.py (dev_pool, hand_prior_weights), danger.py
// (_expected_hand without a belief, _vp_sources, win_path, danger_multiplier, block_factor,
// steal_factor, rob_break_probability), robber.py (estimated_vp, threat, target_weight,
// our_need_indicator, hex_pips_for_player, hex_damage, steal_candidates, choose_victim,
// best_robber_move), politics.py (robber_target_weights) and opponent_model.py
// (robber_habit_factors, _leader).  Python `sum(...)` of floats is a sequential double
// accumulation, `x ** 0.5` is libm pow(x, 0.5) (kept opaque), `round(x, 3)` is the correctly
// rounded decimal (mirrored with snprintf("%.3f") + strtod), `max` / `min` keep the first of
// equal candidates.
#define CATAN_NO_PYTHON
#include "policy.hpp"
#undef CATAN_NO_PYTHON

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace catanbot {

namespace {

inline int pips_of(int num) { return (num >= 0 && num < 13) ? PIPS[num] : 0; }

// Python's `x ** 0.5` (float_pow -> libm pow); the exponent is opaque so it is not turned into sqrt.
inline double pow_half(double x) {
    static const volatile double half = 0.5;
    const double h = half;
    return std::pow(x, h);
}

// CPython's round(x, 3): correctly rounded to 3 decimals (round-half-even on the exact binary
// value) and back to the nearest double.  glibc's printf rounds the exact binary value the
// same way, strtod is correctly rounded.
inline double round3(double x) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", x);
    return std::strtod(buf, nullptr);
}

inline int cards_of(const PlayerC& p) { return p.hand_known ? p.total_resources() : p.hand_size; }

// danger.py constants
constexpr double TURNS_HALF = 3.0;
constexpr double MAX_TURNS = 30.0;
constexpr double BLOCK_FLOOR = 0.35;
constexpr double BLOCK_NEED = 2.0;

// ---------------------------------------------------------------------------
// danger._vp_sources
// ---------------------------------------------------------------------------
struct VpSource {
    double cost_per_vp;
    double cost[NUM_RESOURCES];
    int vp;
    double min_turns;
};

// Stable insertion sort by cost_per_vp (list.sort(key=lambda t: t[0]) is stable).
void sort_sources(VpSource* src, int n) {
    for (int i = 1; i < n; ++i) {
        VpSource x = src[i];
        int j = i - 1;
        while (j >= 0 && src[j].cost_per_vp > x.cost_per_vp) {
            src[j + 1] = src[j];
            --j;
        }
        src[j + 1] = x;
    }
}

inline double sum5(const double* v) {  // sum(list_of_5_floats): 0 + v0 + v1 + ...
    double acc = 0.0;
    for (int r = 0; r < NUM_RESOURCES; ++r) acc += v[r];
    return acc;
}

int vp_sources(const GameStateC& s, int i, VpSource* out) {
    const PlayerC& p = s.players[i];
    int n = 0;
    // City upgrades.
    const int cities = std::min(p.n_settlements, MAX_CITIES - p.n_cities);
    for (int k = 0; k < cities; ++k) {
        VpSource& v = out[n++];
        for (int r = 0; r < NUM_RESOURCES; ++r) v.cost[r] = (double)COST_CITY[r];
        int tot = 0;
        for (int r = 0; r < NUM_RESOURCES; ++r) tot += COST_CITY[r];
        v.cost_per_vp = (double)tot;  // float(sum(B.COST_CITY))
        v.vp = 1;
        v.min_turns = 0.0;
    }
    // Settlements on reachable spots (0-2 roads away): the first `slots` spots sorted by distance.
    const int slots = MAX_SETTLEMENTS - p.n_settlements;
    if (slots > 0) {
        Occupancy occ;
        occupancy(s, occ);
        ReachableSpots reach;
        reachable_spots(s, i, 2, occ, reach);
        int by_dist[3] = {0, 0, 0};
        for (int v = 0; v < NUM_VERTICES; ++v)
            if (reach.dist[v] >= 0 && reach.dist[v] <= 2) ++by_dist[reach.dist[v]];
        const int roads_left = MAX_ROADS - p.n_roads;
        int taken = 0;
        for (int d = 0; d <= 2 && taken < slots; ++d) {
            for (int k = 0; k < by_dist[d] && taken < slots; ++k) {
                ++taken;
                if (d > roads_left) continue;
                VpSource& v = out[n++];
                for (int r = 0; r < NUM_RESOURCES; ++r) v.cost[r] = (double)(COST_SETTLEMENT[r] + d * COST_ROAD[r]);
                v.cost_per_vp = sum5(v.cost);
                v.vp = 1;
                v.min_turns = 0.0;
            }
        }
    }
    // Longest Road.
    if (s.longest_road_owner != i && p.n_roads >= 3 && p.n_roads < MAX_ROADS) {
        const int target = (s.longest_road_owner >= 0 ? s.longest_road_len : 4) + 1;
        const int k = target - engine_longest_road_length(s, i);
        if (0 < k && k <= MAX_ROADS - p.n_roads) {
            VpSource& v = out[n++];
            for (int r = 0; r < NUM_RESOURCES; ++r) v.cost[r] = (double)(k * COST_ROAD[r]);
            v.cost_per_vp = sum5(v.cost) / 2.0;
            v.vp = 2;
            v.min_turns = 0.0;
        }
    }
    // Largest Army.
    int pool[NUM_DEV];
    dev_pool(s, pool);
    int pool_total = 0;
    for (int t = 0; t < NUM_DEV; ++t) pool_total += pool[t];
    const double p_knight = pool_total > 0 ? (double)pool[DEV_KNIGHT] / (double)pool_total : 0.0;
    const double p_vp = pool_total > 0 ? (double)pool[DEV_VICTORY_POINT] / (double)pool_total : 0.0;
    if (s.largest_army_owner != i) {
        const int holder = s.largest_army_owner;
        const int holder_k = holder >= 0 ? s.players[holder].played_knights : 2;
        const int k = holder_k + 1 - p.played_knights;
        double to_buy;
        if (p.dev_known) {
            const int held = p.dev_cards[DEV_KNIGHT] + p.dev_cards_new[DEV_KNIGHT];
            const int diff = k - held;
            to_buy = diff > 0 ? (double)diff : 0.0;           // max(0.0, int)
        } else {
            const double held = (double)p.dev_count * p_knight;
            const double diff = (double)k - held;
            to_buy = diff > 0.0 ? diff : 0.0;                 // max(0.0, float)
        }
        if (k > 0 && (to_buy <= 0 || p_knight > 0)) {
            const double per_card = p_knight > 0 ? (1.0 / p_knight) : 0.0;
            VpSource& v = out[n++];
            for (int r = 0; r < NUM_RESOURCES; ++r) v.cost[r] = to_buy * per_card * (double)COST_DEV[r];
            v.cost_per_vp = sum5(v.cost) / 2.0;
            v.vp = 2;
            v.min_turns = (double)k;
        }
    }
    // Hidden VP cards still in the pool.
    if (p_vp > 0) {
        double per_vp[NUM_RESOURCES];
        for (int r = 0; r < NUM_RESOURCES; ++r) per_vp[r] = (double)COST_DEV[r] / p_vp;
        const double total = sum5(per_vp);
        for (int k = 0; k < pool[DEV_VICTORY_POINT]; ++k) {
            VpSource& v = out[n++];
            for (int r = 0; r < NUM_RESOURCES; ++r) v.cost[r] = per_vp[r];
            v.cost_per_vp = total;
            v.vp = 1;
            v.min_turns = 0.0;
        }
    }
    sort_sources(out, n);
    return n;
}

// _expected_hand(state, i) without a belief.
void expected_hand(const GameStateC& s, int i, double hand[NUM_RESOURCES], bool& known) {
    const PlayerC& p = s.players[i];
    if (p.hand_known) {
        for (int r = 0; r < NUM_RESOURCES; ++r) hand[r] = (double)p.resources[r];
        known = true;
        return;
    }
    double w[NUM_RESOURCES];
    hand_prior_weights(s, i, w);
    for (int r = 0; r < NUM_RESOURCES; ++r) hand[r] = (double)p.hand_size * w[r];
    known = false;
}

}  // namespace

// ---------------------------------------------------------------------------
// counting / robber basics
// ---------------------------------------------------------------------------
void dev_pool(const GameStateC& s, int out[NUM_DEV]) {
    for (int t = 0; t < NUM_DEV; ++t) out[t] = DEV_DECK_COUNTS[t];
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& q = s.players[i];
        out[DEV_KNIGHT] -= q.played_knights;
        if (q.dev_known)
            for (int t = 0; t < NUM_DEV; ++t) out[t] -= q.dev_cards[t] + q.dev_cards_new[t];
    }
    for (int t = 0; t < NUM_DEV; ++t) out[t] = std::max(0, out[t]);
}

double estimated_vp(const GameStateC& s, int i) { return (double)s.public_vp(i) + expected_hidden_vp(s, i); }

double threat(const GameStateC& s, int i) {
    const double vp = estimated_vp(s, i);
    double t = 1.0 + 0.3 * std::max(0.0, vp - 4.0);
    if (vp >= 8) t *= 1.8;
    else if (vp >= 7) t *= 1.3;
    return t;
}

int port_ratio_of(const GameStateC& s, int i, int res) {
    const PlayerC& p = s.players[i];
    int ratio = 4;
    for (int pass = 0; pass < 2; ++pass) {
        const int cnt = pass == 0 ? p.n_settlements : p.n_cities;
        const uint8_t* ids = pass == 0 ? p.settlements : p.cities;
        for (int k = 0; k < cnt; ++k) {
            const int t = s.ports[ids[k]];
            if (t < 0) continue;
            if (t == PORT_GENERIC) ratio = std::min(ratio, 3);
            else if (t == res) return 2;
        }
    }
    return ratio;
}

void hand_prior_weights(const GameStateC& s, int player, double out[NUM_RESOURCES]) {
    const double smoothing = 0.04;
    double prod[NUM_RESOURCES];
    player_production(s, player, true, prod);
    double w[NUM_RESOURCES];
    for (int r = 0; r < NUM_RESOURCES; ++r) w[r] = prod[r] + smoothing;
    for (int r = 0; r < NUM_RESOURCES; ++r)
        w[r] *= 1.0 + 0.5 * (1.0 - (double)s.bank[r] / (double)BANK_PER_RESOURCE);
    double tot = 0.0;
    for (int r = 0; r < NUM_RESOURCES; ++r) tot += w[r];
    for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = w[r] / tot;
}

// ---------------------------------------------------------------------------
// discard.py
// ---------------------------------------------------------------------------
void default_keep_targets(const GameStateC& s, int player, KeepTargets& out) {
    const PlayerC& p = s.players[player];
    const int* t[4];
    int n = 0;
    if (p.n_settlements > 0 && p.n_cities < MAX_CITIES) t[n++] = COST_CITY;
    if (p.n_settlements < MAX_SETTLEMENTS) {
        // placement.buildable_settlements non-empty: a free vertex next to one of our roads.
        Occupancy occ;
        occupancy(s, occ);
        bool spots = false;
        for (int k = 0; k < p.n_roads && !spots; ++k) {
            const int e = p.roads[k];
            if (occ.free_vertex[EDGE_VERTICES[e][0]] || occ.free_vertex[EDGE_VERTICES[e][1]]) spots = true;
        }
        if (spots) {  // targets.insert(0, settlement)
            for (int k = n; k > 0; --k) t[k] = t[k - 1];
            t[0] = COST_SETTLEMENT;
            ++n;
        }
    }
    int deck = 0;
    for (int k = 0; k < NUM_DEV; ++k) deck += s.dev_deck[k];
    if (deck > 0) t[n++] = COST_DEV;
    if (p.n_roads < MAX_ROADS) t[n++] = COST_ROAD;
    // Stable sort by (missing, -sum(cost)).
    int key_missing[4], key_sum[4];
    for (int k = 0; k < n; ++k) {
        int m = 0, tot = 0;
        for (int r = 0; r < NUM_RESOURCES; ++r) {
            m += std::max(0, t[k][r] - p.resources[r]);
            tot += t[k][r];
        }
        key_missing[k] = m;
        key_sum[k] = -tot;
    }
    for (int i = 1; i < n; ++i) {
        const int* x = t[i];
        const int km = key_missing[i], ks = key_sum[i];
        int j = i - 1;
        while (j >= 0 && (key_missing[j] > km || (key_missing[j] == km && key_sum[j] > ks))) {
            t[j + 1] = t[j];
            key_missing[j + 1] = key_missing[j];
            key_sum[j + 1] = key_sum[j];
            --j;
        }
        t[j + 1] = x;
        key_missing[j + 1] = km;
        key_sum[j + 1] = ks;
    }
    out.n = n;
    for (int k = 0; k < n; ++k) out.cost[k] = t[k];
}

namespace {
// discard._needed_tiers: (top-target cost, ceil(second-target cost / 2)).
void needed_tiers(const GameStateC& s, int player, int top[NUM_RESOURCES], int second[NUM_RESOURCES]) {
    KeepTargets kt;
    default_keep_targets(s, player, kt);
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        top[r] = kt.n > 0 ? kt.cost[0][r] : 0;
        second[r] = kt.n > 1 ? (kt.cost[1][r] + 1) / 2 : 0;  // -(-c // 2) = ceil(c / 2) for c >= 0
    }
}
}  // namespace

void needed_vector(const GameStateC& s, int player, int out[NUM_RESOURCES]) {
    int top[NUM_RESOURCES], second[NUM_RESOURCES];
    needed_tiers(s, player, top, second);
    for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = std::max(top[r], second[r]);
}

void choose_discard(const GameStateC& s, int player, int32_t out[NUM_RESOURCES]) {
    const PlayerC& p = s.players[player];
    int hand[NUM_RESOURCES];
    for (int r = 0; r < NUM_RESOURCES; ++r) hand[r] = p.resources[r];
    const int n = p.total_resources() / 2;  // sum(hand) // 2 (hands are never negative)
    int top[NUM_RESOURCES], second[NUM_RESOURCES], needed[NUM_RESOURCES];
    needed_tiers(s, player, top, second);
    for (int r = 0; r < NUM_RESOURCES; ++r) needed[r] = std::max(top[r], second[r]);
    double prod[NUM_RESOURCES], scarcity[NUM_RESOURCES];
    player_production(s, player, true, prod);
    resource_scarcity(s, scarcity);
    const double* demand = resource_demand();
    for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = 0;
    for (int k = 0; k < n; ++k) {
        int best_r = -1;
        double best_score = -1e9;
        for (int r = 0; r < NUM_RESOURCES; ++r) {
            if (hand[r] <= 0) continue;
            const int surplus = hand[r] - needed[r];
            double tier;
            if (surplus > 0) tier = 10.0;
            else if (hand[r] > top[r]) tier = 4.0;
            else tier = 0.0;
            const double score = tier + 0.5 * (double)surplus + 6.0 * prod[r] - 1.5 * demand[r] * scarcity[r];
            if (score > best_score) {
                best_score = score;
                best_r = r;
            }
        }
        if (best_r < 0) break;
        out[best_r] += 1;
        hand[best_r] -= 1;
    }
}

ActionC choose_discard_action(const GameStateC& s, int player, const ActionList* legal) {
    ActionC a = make_action(ACT_DISCARD, 0);
    a.nvec1 = 5;
    choose_discard(s, player, a.vec1);
    if (legal == nullptr) return a;
    bool found = false;
    for (int i = 0; i < legal->n && !found; ++i) {
        const ActionC& b = legal->items[i];
        if (b.kind != ACT_DISCARD) continue;
        found = true;
        for (int r = 0; r < NUM_RESOURCES; ++r)
            if (b.vec1[r] != a.vec1[r]) found = false;
    }
    if (found) return a;
    // Closest legal discard (should not happen with a correct engine).
    const ActionC* best = nullptr;
    long best_d = 1000000000L;
    for (int i = 0; i < legal->n; ++i) {
        const ActionC& b = legal->items[i];
        if (b.kind != ACT_DISCARD) continue;
        long d = 0;
        for (int r = 0; r < NUM_RESOURCES; ++r) d += std::labs((long)b.vec1[r] - (long)a.vec1[r]);
        if (d < best_d) {
            best_d = d;
            best = &b;
        }
    }
    return best ? *best : a;
}

// ---------------------------------------------------------------------------
// danger.py
// ---------------------------------------------------------------------------
void win_path(const GameStateC& s, int i, WinPathC& out) {
    const PlayerC& p = s.players[i];
    const int n = s.num_players;
    out = WinPathC{};
    const double vp = estimated_vp(s, i);
    out.vp = vp;
    int need_vp = (int)((double)VP_TO_WIN - std::nearbyint(vp));  // round() is half-to-even
    double hand[NUM_RESOURCES];
    bool known;
    expected_hand(s, i, hand, known);
    out.hand_known = known;
    for (int r = 0; r < NUM_RESOURCES; ++r) out.hand[r] = hand[r];
    double prod[NUM_RESOURCES], prod_now[NUM_RESOURCES];
    player_production(s, i, true, prod);
    player_production(s, i, false, prod_now);
    out.blocked_now = sum5(prod) - sum5(prod_now);
    for (int r = 0; r < NUM_RESOURCES; ++r) out.prod[r] = prod[r];
    double cost[NUM_RESOURCES] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double min_turns = 0.0;
    int n_steps = 0;
    if (s.phase == PHASE_GAME_OVER && s.winner == i) need_vp = 0;
    if (need_vp > 0) {
        int left = need_vp;
        VpSource sources[24];
        int n_src = vp_sources(s, i, sources);
        double remaining[NUM_RESOURCES];
        for (int r = 0; r < NUM_RESOURCES; ++r) remaining[r] = hand[r];
        while (left > 0 && n_src > 0) {
            // min(sources, key=gap) with gap = (missing cards per VP, cost per VP): the first minimum.
            int best = 0;
            double best_g = 0.0, best_c = 0.0;
            for (int k = 0; k < n_src; ++k) {
                double acc = 0.0;
                for (int r = 0; r < NUM_RESOURCES; ++r) acc += std::max(0.0, sources[k].cost[r] - remaining[r]);
                const double g = acc / (double)sources[k].vp;
                const double c = sources[k].cost_per_vp;
                if (k == 0 || g < best_g || (g == best_g && c < best_c)) {
                    best = k;
                    best_g = g;
                    best_c = c;
                }
            }
            const VpSource b = sources[best];
            for (int k = best; k + 1 < n_src; ++k) sources[k] = sources[k + 1];
            --n_src;
            ++n_steps;
            for (int r = 0; r < NUM_RESOURCES; ++r) {
                cost[r] += b.cost[r];
                remaining[r] = std::max(0.0, remaining[r] - b.cost[r]);
            }
            min_turns = std::max(min_turns, b.min_turns);
            left -= b.vp;
        }
        if (left > 0) {  // no path at all: treat as very far
            out.no_path = true;
            for (int r = 0; r < NUM_RESOURCES; ++r) cost[r] = cost[r] + 10.0;
        }
    }
    out.need_vp = need_vp;
    out.n_steps = n_steps;
    out.min_turns = min_turns;
    for (int r = 0; r < NUM_RESOURCES; ++r) out.cost[r] = cost[r];
    double missing[NUM_RESOURCES];
    for (int r = 0; r < NUM_RESOURCES; ++r) missing[r] = std::max(0.0, cost[r] - hand[r]);
    const double tot_missing = sum5(missing);
    double need_share[NUM_RESOURCES];
    for (int r = 0; r < NUM_RESOURCES; ++r) need_share[r] = tot_missing > 1e-9 ? missing[r] / tot_missing : 0.0;
    // Supply per resource: own production plus what a surplus buys through the best port.
    double surplus[NUM_RESOURCES];
    int ratio[NUM_RESOURCES];
    for (int q = 0; q < NUM_RESOURCES; ++q) {
        surplus[q] = prod[q] * (1.0 - need_share[q]);
        ratio[q] = port_ratio_of(s, i, q);
    }
    double trade_in[NUM_RESOURCES] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        if (s.bank[r] <= 0) continue;
        double acc = 0.0;
        for (int q = 0; q < NUM_RESOURCES; ++q)
            if (q != r) acc += surplus[q] / (double)ratio[q];
        trade_in[r] = acc;
    }
    double supply[NUM_RESOURCES];
    for (int r = 0; r < NUM_RESOURCES; ++r) supply[r] = prod[r] + trade_in[r];
    double eff_need[NUM_RESOURCES];
    for (int r = 0; r < NUM_RESOURCES; ++r) eff_need[r] = need_share[r];
    for (int q = 0; q < NUM_RESOURCES; ++q)
        for (int r = 0; r < NUM_RESOURCES; ++r) {
            if (r == q || need_share[r] <= 0 || supply[r] <= 1e-9) continue;
            eff_need[q] += need_share[r] * (surplus[q] / (double)ratio[q]) / supply[r];
        }
    // Turns: per-resource bottleneck vs. total throughput, per round of n rolls.
    double turns = 0.0;
    if (need_vp > 0) {
        for (int r = 0; r < NUM_RESOURCES; ++r) {
            if (missing[r] <= 1e-9) continue;
            const double per_turn = supply[r] * (double)n;
            turns = std::max(turns, per_turn > 1e-9 ? missing[r] / per_turn : MAX_TURNS);
        }
        const double total_per_turn = sum5(prod) * (double)n;
        if (tot_missing > 1e-9) turns = std::max(turns, total_per_turn > 1e-9 ? tot_missing / total_per_turn : MAX_TURNS);
        turns = std::max(turns, min_turns - 1.0);
        turns = std::min(MAX_TURNS, turns);
    }
    const bool can_win_now = need_vp > 0 && tot_missing <= 1e-9 && min_turns <= 1.0;
    const double danger = (need_vp <= 0 || can_win_now) ? 1.0 : 1.0 / (1.0 + turns / TURNS_HALF);
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        out.missing[r] = missing[r];
        out.need_share[r] = need_share[r];
        out.eff_need[r] = eff_need[r];
        out.supply[r] = supply[r];
    }
    out.turns = turns;
    out.can_win_now = can_win_now;
    out.danger = danger;
}

void win_paths(const GameStateC& s, WinPaths& out) {
    for (int i = 0; i < s.num_players; ++i) win_path(s, i, out.p[i]);
}

double danger_multiplier(const WinPathC& wp) { return 0.4 + 1.6 * wp.danger; }

double block_factor(const WinPathC& wp, int res, int pips) {
    if (pips <= 0) return 1.0;
    const double need = BLOCK_FLOOR + BLOCK_NEED * std::min(1.0, wp.eff_need[res]);
    const double supply_pips = wp.supply[res] * 36.0;
    const double share = supply_pips > 1e-9 ? (double)pips / supply_pips : 1.0;
    return need * (0.6 + 0.4 * std::min(1.0, share));
}

double steal_factor(const WinPathC& wp, const double* our_need) {
    // WinPath.composition()
    double comp[NUM_RESOURCES];
    const double tot = sum5(wp.hand);
    if (tot <= 1e-9) {
        for (int r = 0; r < NUM_RESOURCES; ++r) comp[r] = 0.2;
    } else {
        for (int r = 0; r < NUM_RESOURCES; ++r) comp[r] = wp.hand[r] / tot;
    }
    double f = 0.0;
    for (int r = 0; r < NUM_RESOURCES; ++r)
        f += comp[r] * (0.6 + 0.9 * wp.need_share[r] + (our_need != nullptr ? 0.5 * our_need[r] : 0.0));
    return f;
}

double rob_break_probability(const WinPathC& wp) {
    if (!wp.can_win_now) return 0.0;
    const double tot = sum5(wp.hand);
    if (tot <= 0) return 0.0;
    return std::min(1.0, sum5(wp.cost) / tot);
}

// ---------------------------------------------------------------------------
// robber.py
// ---------------------------------------------------------------------------
double target_weight(const GameStateC& s, int i, const WinPaths& paths) {
    return threat(s, i) * danger_multiplier(paths.p[i]);
}

void our_need_indicator(const GameStateC& s, int player, double out[NUM_RESOURCES]) {
    const PlayerC& p = s.players[player];
    if (!p.hand_known) {
        for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = 0.0;
        return;
    }
    int needed[NUM_RESOURCES];
    needed_vector(s, player, needed);
    for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = needed[r] > p.resources[r] ? 1.0 : 0.0;
}

int hex_pips_for_player(const GameStateC& s, int h, int i) {
    const int res = s.hex_res[h];
    const int num = s.hex_num[h];
    if (res == DESERT || num == 0) return 0;
    const PlayerC& p = s.players[i];
    const int pips = pips_of(num);
    int total = 0;
    for (int k = 0; k < 6; ++k) {
        const int v = HEX_VERTICES[h][k];
        if (p.has_city(v)) total += 2 * pips;
        else if (p.has_settlement(v)) total += pips;
    }
    return total;
}

void hex_damage(const GameStateC& s, int h, int player, const double* target_weights, const WinPaths& paths,
                double& opp, double& own) {
    opp = 0.0;
    own = 0.0;
    const int res = s.hex_res[h];
    const int num = s.hex_num[h];
    if (res == DESERT || num == 0) return;
    double scarcity[NUM_RESOURCES];
    resource_scarcity(s, scarcity);
    const double w = resource_demand()[res] * pow_half(scarcity[res]);
    for (int i = 0; i < s.num_players; ++i) {
        const int pips = hex_pips_for_player(s, h, i);
        if (!pips) continue;
        if (i == player) {
            own += (double)pips * w;
        } else {
            const WinPathC& wp = paths.p[i];
            const double tw = target_weights != nullptr ? target_weights[i] : threat(s, i) * danger_multiplier(wp);
            opp += (double)pips * w * tw * block_factor(wp, res, pips);
        }
    }
}

int steal_candidates(const GameStateC& s, int h, int player, int out[MAX_PLAYERS]) {
    int n = 0;
    for (int i = 0; i < s.num_players; ++i) {
        if (i == player) continue;
        const PlayerC& p = s.players[i];
        if (cards_of(p) <= 0) continue;
        bool on_hex = false;
        for (int k = 0; k < 6 && !on_hex; ++k)
            if (p.has_building(HEX_VERTICES[h][k])) on_hex = true;
        if (on_hex) out[n++] = i;
    }
    return n;
}

int choose_victim(const GameStateC& s, int h, int player, const double* target_weights, const WinPaths& paths,
                  const double* our_need) {
    int cands[MAX_PLAYERS];
    const int nc = steal_candidates(s, h, player, cands);
    if (nc == 0) return -1;
    int best = -1;
    double best_val = 0.0;
    int best_c8 = 0, best_cards = 0;
    for (int k = 0; k < nc; ++k) {
        const int i = cands[k];
        const PlayerC& p = s.players[i];
        const int cards = cards_of(p);
        const WinPathC& wp = paths.p[i];
        const double tw = target_weights != nullptr ? target_weights[i] : threat(s, i) * danger_multiplier(wp);
        const double val = round3(tw * steal_factor(wp, our_need) + 3.0 * rob_break_probability(wp));
        const int c8 = std::min(cards, 8);
        // max(cands, key=(val, min(cards, 8), cards)): the first maximum wins ties.
        bool better;
        if (best < 0) better = true;
        else if (val != best_val) better = val > best_val;
        else if (c8 != best_c8) better = c8 > best_c8;
        else better = cards > best_cards;
        if (better) {
            best = i;
            best_val = val;
            best_c8 = c8;
            best_cards = cards;
        }
    }
    return best;
}

RobberMove best_robber_move(const GameStateC& s, int player, const double* target_weights) {
    const int exclude = s.robber;
    WinPaths paths;
    win_paths(s, paths);
    double our_need[NUM_RESOURCES];
    our_need_indicator(s, player, our_need);
    RobberMove best;
    best.score = -1e9;
    for (int h = 0; h < NUM_HEXES; ++h) {
        if (h == exclude) continue;
        double opp, own;
        hex_damage(s, h, player, target_weights, paths, opp, own);
        const int victim = choose_victim(s, h, player, target_weights, paths, our_need);
        double score = opp - 1.6 * own;
        if (victim >= 0) {
            const PlayerC& vp = s.players[victim];
            const int cards = cards_of(vp);
            const WinPathC& wp = paths.p[victim];
            const double tw = target_weights != nullptr ? target_weights[victim] : threat(s, victim) * danger_multiplier(wp);
            score += 1.5 + 0.35 * (double)std::min(cards, 6) * steal_factor(wp, our_need) + 0.8 * (tw - 1.0) +
                     4.0 * rob_break_probability(wp);
        }
        if (score > best.score) {
            best.score = score;
            best.hex = h;
            best.victim = victim;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// politics.robber_target_weights / OpponentModel.robber_habit_weights
// ---------------------------------------------------------------------------
namespace {
// max(range(n), key=_vp): the first player with the highest estimated VP.
int leader_all(const GameStateC& s) {
    int best = 0;
    double best_vp = estimated_vp(s, 0);
    for (int k = 1; k < s.num_players; ++k) {
        const double vp = estimated_vp(s, k);
        if (vp > best_vp) {
            best = k;
            best_vp = vp;
        }
    }
    return best;
}

// opponent_model._leader(state, exclude): the first player (other than `exclude`) with the highest VP.
int leader_excluding(const GameStateC& s, int exclude) {
    int best = -1;
    double best_vp = -1.0;
    for (int k = 0; k < s.num_players; ++k) {
        if (k == exclude) continue;
        const double vp = estimated_vp(s, k);
        if (vp > best_vp) {
            best = k;
            best_vp = vp;
        }
    }
    return best;
}
}  // namespace

void robber_weights(const GameStateC& s, int actor, const RobberWeights& rw, double out[MAX_PLAYERS]) {
    const int n = s.num_players;
    if (rw.mode == 0) {
        for (int j = 0; j < n; ++j) out[j] = 0.0;
        return;
    }
    const int lead_ex = (rw.mode == 2 || rw.has_habit) ? leader_excluding(s, actor) : -1;
    if (rw.mode == 2) {  // robber_habit_weights: threat(j) x robber_habit_factors
        for (int j = 0; j < n; ++j) {
            if (j == actor) {
                out[j] = 0.0;
                continue;
            }
            double f = 1.0;
            f *= rw.c[actor][j][2];
            if (j == lead_ex) f *= rw.habit_leader[actor][j];
            out[j] = threat(s, j) * f;
        }
        return;
    }
    WinPaths paths;
    win_paths(s, paths);
    const int leader = leader_all(s);
    for (int j = 0; j < n; ++j) {
        if (j == actor) {
            out[j] = 0.0;
            continue;
        }
        double w = target_weight(s, j, paths);   // VP threat x distance to win
        w *= rw.c[actor][j][0];                   // grudges
        w *= rw.c[actor][j][1];                   // friends get hit less
        if (rw.has_habit) {
            double f = 1.0;
            f *= rw.c[actor][j][2];
            if (j == lead_ex) f *= rw.habit_leader[actor][j];
            w *= f;                               // observed victim habits / leader hitting
        }
        w *= rw.c[actor][j][3];                   // coalitions: spare the allies
        if (leader != j && leader != actor) w *= rw.coal[j][leader];  // 1.0 when the bloc is weak
        out[j] = w;
    }
}

}  // namespace catanbot
