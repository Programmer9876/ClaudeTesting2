// Exact port of catanbot/heuristic.py::static_value (and HeuristicEvaluator.evaluate).
//
// Like features.cpp, every arithmetic expression mirrors the Python statement
// it comes from - the same accumulation order, the same int -> float points,
// `math.sqrt` -> sqrt, `x ** 0.5` -> libm pow(x, 0.5), `math.exp` -> exp - so
// the values are bit-identical to the Python reference on the same libm
// (tests/test_accel_heuristic.py checks |C++ - Python| <= 1e-6 over thousands
// of states).  The helpers come from placement.py (production, scarcity,
// reachable_spots, score_settlement_spot incl. its blockability term
// robber_exposure / BlockContext), robber.py (threat) and counting.py
// (expected_hidden_vp).
// Keep this file in sync with those modules.
#define CATAN_NO_PYTHON
#include "heuristic.hpp"
#undef CATAN_NO_PYTHON

#include <algorithm>
#include <cmath>
#include <cstring>

namespace catanbot {

namespace {

inline int pips_of(int num) { return (num >= 0 && num < 13) ? PIPS[num] : 0; }

// Python's `x ** 0.5` is libm pow(x, 0.5) (float_pow calls pow()), which is not
// guaranteed to equal sqrt(x) bit for bit.  The exponent is kept opaque so the
// compiler cannot rewrite the call into sqrt.
inline double pow_half(double x) {
    static const volatile double half = 0.5;
    const double h = half;
    return std::pow(x, h);
}

// heuristic.BUILD_VALUE
constexpr double VALUE_CITY = 3.2;
constexpr double VALUE_SETTLEMENT = 2.8;
constexpr double VALUE_DEV = 1.0;

// ---------------------------------------------------------------------------
// placement._expansion_potential: free spots two roads away from v (with multiplicity)
// ---------------------------------------------------------------------------
int expansion_potential(int v, const Occupancy& occ) {
    int count = 0;
    for (int k = 0; k < VERTEX_EDGES_N[v]; ++k) {
        if (occ.edge_taken[VERTEX_EDGES[v][k]]) continue;
        const int w = VERTEX_NEIGHBORS[v][k];
        if (occ.owner[w] != -1) continue;
        for (int k2 = 0; k2 < VERTEX_EDGES_N[w]; ++k2) {
            if (occ.edge_taken[VERTEX_EDGES[w][k2]]) continue;
            const int x = VERTEX_NEIGHBORS[w][k2];
            if (x != v && occ.free_vertex[x]) ++count;
        }
    }
    return count;
}

// robber.threat: how dangerous player i is (1.0 for a harmless player).
static double threat(const GameStateC& s, int i) {
    const double vp = (double)s.public_vp(i) + expected_hidden_vp(s, i);
    double t = 1.0 + 0.3 * std::max(0.0, vp - 4.0);
    if (vp >= 8) t *= 1.8;
    else if (vp >= 7) t *= 1.3;
    return t;
}

// ---------------------------------------------------------------------------
// placement: blockability (robber_exposure / BlockContext), see the module comment there
// ---------------------------------------------------------------------------
constexpr double PLACEMENT_ROBBER_Q = 0.35;       // placement.PLACEMENT_ROBBER_Q
constexpr double PLACEMENT_BLOCK_WEIGHT = 1.0;    // placement.PLACEMENT_BLOCK_WEIGHT
constexpr double PLACEMENT_STRONG_THREAT = 1.3;   // placement.PLACEMENT_STRONG_THREAT

// placement.BlockContext: the per-(state, player) part of the blockability term.
struct BlockContext {
    double hex_w[NUM_HEXES];    // placement.hex_block_weights: pips x demand x scarcity ** 0.5, 0 for the desert
    uint8_t strong[NUM_HEXES];  // placement.strong_opponent_hexes
    double base_w[NUM_HEXES];   // W(h) of the player's current buildings
    double base_k[NUM_HEXES];   // K(h): the same weights cubed and summed
    double exposure_before;
};

// placement.hex_block_weights with the `scarcity ** 0.5` table precomputed.
static void hex_block_weights(const GameStateC& s, const double* pow05, double out[NUM_HEXES]) {
    const double* demand = resource_demand();
    double res_w[NUM_RESOURCES];  // RESOURCE_DEMAND[r] * scarcity[r] ** 0.5
    for (int r = 0; r < NUM_RESOURCES; ++r) res_w[r] = demand[r] * pow05[r];
    for (int h = 0; h < NUM_HEXES; ++h) {
        const int res = s.hex_res[h];
        const int num = s.hex_num[h];
        if (res == DESERT || num == 0) {
            out[h] = 0.0;
            continue;
        }
        out[h] = (double)pips_of(num) * res_w[res];
    }
}

// placement.strong_opponent_hexes
static void strong_opponent_hexes(const GameStateC& s, int player, uint8_t out[NUM_HEXES]) {
    std::memset(out, 0, NUM_HEXES);
    for (int i = 0; i < s.num_players; ++i) {
        if (i == player) continue;
        if (threat(s, i) < PLACEMENT_STRONG_THREAT) continue;
        const PlayerC& q = s.players[i];
        for (int k = 0; k < q.n_settlements; ++k) {
            const int v = q.settlements[k];
            for (int j = 0; j < VERTEX_HEXES_N[v]; ++j) out[VERTEX_HEXES[v][j]] = 1;
        }
        for (int k = 0; k < q.n_cities; ++k) {
            const int v = q.cities[k];
            for (int j = 0; j < VERTEX_HEXES_N[v]; ++j) out[VERTEX_HEXES[v][j]] = 1;
        }
    }
}

// One building's contribution to W(h) / K(h) (placement._stack_weights inner loops): c = hex_w[h] for
// a settlement (weight 1.0: x * 1.0 == x exactly, so it matches Python's plain `c = hex_w[h]`) and
// 2.0 * hex_w[h] for a city.
static inline void add_building(int v, double weight, const double* hex_w, double* w, double* k) {
    for (int j = 0; j < VERTEX_HEXES_N[v]; ++j) {
        const int h = VERTEX_HEXES[v][j];
        const double c = weight * hex_w[h];
        w[h] += c;
        k[h] += c * c * c;
    }
}

// placement._stack_weights(state, player, hex_w, extra_settlement, extra_city); -1 = none.
static void stack_weights(const GameStateC& s, int player, const double* hex_w, int extra_settlement, int extra_city,
                          double w[NUM_HEXES], double k[NUM_HEXES]) {
    for (int h = 0; h < NUM_HEXES; ++h) w[h] = k[h] = 0.0;
    const PlayerC& p = s.players[player];
    for (int i = 0; i < p.n_settlements; ++i) {
        if (p.settlements[i] == extra_city) continue;
        add_building(p.settlements[i], 1.0, hex_w, w, k);
    }
    for (int i = 0; i < p.n_cities; ++i) add_building(p.cities[i], 2.0, hex_w, w, k);
    if (extra_settlement >= 0) add_building(extra_settlement, 1.0, hex_w, w, k);
    if (extra_city >= 0) add_building(extra_city, 2.0, hex_w, w, k);
}

// placement._exposure_of
static double exposure_of(const double* w, const double* k, const uint8_t* strong) {
    double ssq = 0.0;
    double num = 0.0;
    for (int h = 0; h < NUM_HEXES; ++h) {
        const double x = w[h];
        if (x <= 0.0) continue;
        ssq += x * x;
        // W^3 - K is 0 for a hex with a single building and grows with stacking.
        num += (x * x * x - k[h]) * (1.0 + 0.5 * (double)strong[h]);
    }
    if (ssq <= 0.0) return 0.0;
    return PLACEMENT_ROBBER_Q * num / ssq;
}

// placement.BlockContext.__init__
static void make_block_context(const GameStateC& s, int player, const double* pow05, BlockContext& b) {
    hex_block_weights(s, pow05, b.hex_w);
    strong_opponent_hexes(s, player, b.strong);
    stack_weights(s, player, b.hex_w, -1, -1, b.base_w, b.base_k);
    b.exposure_before = exposure_of(b.base_w, b.base_k, b.strong);
}

// placement.BlockContext.exposure_with_settlement
static double exposure_with_settlement(const BlockContext& b, int v) {
    double w[NUM_HEXES], k[NUM_HEXES];
    std::memcpy(w, b.base_w, sizeof(w));
    std::memcpy(k, b.base_k, sizeof(k));
    for (int j = 0; j < VERTEX_HEXES_N[v]; ++j) {
        const int h = VERTEX_HEXES[v][j];
        const double c = b.hex_w[h];
        w[h] += c;
        k[h] += c * c * c;
    }
    return exposure_of(w, k, b.strong);
}

// score_settlement_spot with the per-state `scarcity ** 0.5` table and the player's BlockContext precomputed.
double score_spot(const GameStateC& s, int v, const Occupancy& occ, const double* own_prod, const double* pow05,
                  bool setup, const BlockContext& block) {
    double prod[NUM_RESOURCES];
    vertex_production(s, v, true, prod);
    const double* demand = resource_demand();
    double score = 0.0;
    int distinct = 0;
    int new_types = 0;
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        if (prod[r] <= 0) continue;
        ++distinct;
        const double have = own_prod[r];
        if (have <= 0) ++new_types;
        const double complement = 1.0 / (1.0 + 4.0 * have);  // 1.0 when we have none, ~0.5 at 9 pips
        score += prod[r] * 36.0 * demand[r] * pow05[r] * (0.6 + 0.8 * complement);
    }
    score += 0.6 * distinct + 1.2 * new_types;
    if (setup) {
        // First placements: value ore+wheat / wood+brick combos that unlock builds.
        const double wb = std::min(prod[WOOD], prod[BRICK]) * 36.0;
        const double ow = std::min(prod[ORE], prod[WHEAT]) * 36.0;
        score += 0.35 * (wb + ow);
    }
    const int port = s.ports[v];
    if (port >= 0) {
        if (port == PORT_GENERIC) score += 1.0;
        else if (port < NUM_RESOURCES) score += 0.5 + 6.0 * (own_prod[port] + prod[port]);
    }
    score += 0.35 * expansion_potential(v, occ);
    // Blockability: how much more of our income one robber placement could switch off.
    score -= PLACEMENT_BLOCK_WEIGHT * (exposure_with_settlement(block, v) - block.exposure_before);
    return score;
}

// Everything static_value needs once per state.
struct Context {
    Occupancy occ;
    double scarcity[NUM_RESOURCES];
    double sqrt_scarcity[NUM_RESOURCES];  // math.sqrt(scarcity[r])
    double pow05_scarcity[NUM_RESOURCES]; // scarcity[r] ** 0.5
};

void make_context(const GameStateC& s, Context& c) {
    occupancy(s, c.occ);
    resource_scarcity(s, c.scarcity);
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        c.sqrt_scarcity[r] = std::sqrt(c.scarcity[r]);
        c.pow05_scarcity[r] = pow_half(c.scarcity[r]);
    }
}

// counting.dev_draw_probabilities()[DEV_KNIGHT]: undrawn deck, else the public pool.
static double knight_draw_probability(const GameStateC& s) {
    int total = 0;
    for (int t = 0; t < NUM_DEV; ++t) total += s.dev_deck[t];
    if (total > 0) return (double)s.dev_deck[DEV_KNIGHT] / (double)total;
    int pool[NUM_DEV];  // counting.dev_pool
    for (int t = 0; t < NUM_DEV; ++t) pool[t] = DEV_DECK_COUNTS[t];
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& q = s.players[i];
        pool[DEV_KNIGHT] -= q.played_knights;
        if (q.dev_known)
            for (int t = 0; t < NUM_DEV; ++t) pool[t] -= q.dev_cards[t] + q.dev_cards_new[t];
    }
    int pool_total = 0;
    for (int t = 0; t < NUM_DEV; ++t) {
        pool[t] = std::max(0, pool[t]);
        pool_total += pool[t];
    }
    if (pool_total <= 0) return 0.0;
    return (double)pool[DEV_KNIGHT] / (double)pool_total;
}

// Exact port of robber.steal_exposure_fast: expected card-value lost to an out-of-turn
// robbery (7 or plausible knight from every opponent who rolls before us).
struct VictimKey {
    double threat;
    int c8;
    int cards;
    bool operator>(const VictimKey& o) const {
        if (threat != o.threat) return threat > o.threat;
        if (c8 != o.c8) return c8 > o.c8;
        return cards > o.cards;
    }
};

double steal_exposure_fast(const GameStateC& s, int player, const double* pow05_scarcity) {
    const PlayerC& p = s.players[player];
    const int cards = p.hand_known ? p.total_resources() : p.hand_size;
    const int n = s.num_players;
    if (cards < 3 || n <= 1) return 0.0;
    int robbers[MAX_PLAYERS];
    int n_robbers = 0;
    if (s.current == player) {
        for (int i = 1; i < n; ++i) robbers[n_robbers++] = (player + i) % n;
    } else {
        int m = s.dice ? (s.current + 1) % n : s.current;
        while (m != player) {
            robbers[n_robbers++] = m;
            m = (m + 1) % n;
        }
    }
    if (n_robbers == 0) return 0.0;
    const double p_knight = knight_draw_probability(s);
    VictimKey keys[MAX_PLAYERS];
    for (int i = 0; i < n; ++i) {
        const PlayerC& q = s.players[i];
        const int c = q.hand_known ? q.total_resources() : q.hand_size;
        keys[i] = VictimKey{threat(s, i), std::min(c, 8), c};
    }
    double p_safe = 1.0;
    for (int k = 0; k < n_robbers; ++k) {
        const int j = robbers[k];
        const PlayerC& q = s.players[j];
        double has_knight;
        if (q.dev_known)
            has_knight = (q.dev_cards[DEV_KNIGHT] + q.dev_cards_new[DEV_KNIGHT] > 0) ? 1.0 : 0.0;
        else
            has_knight = 1.0 - std::pow(1.0 - p_knight, (double)std::max(0, q.dev_count));
        if (s.dev_played_this_turn && j == s.current) has_knight = 0.0;
        const double p_rob = 1.0 / 6.0 + (5.0 / 6.0) * has_knight * 0.5;
        int target = -1;
        for (int i = 0; i < n; ++i) {
            if (i == j || keys[i].cards <= 0) continue;
            if (target < 0 || keys[i] > keys[target]) target = i;
        }
        const double p_me = (target == player) ? 0.7 : 0.08;
        p_safe *= 1.0 - p_rob * p_me;
    }
    const double p_robbed = 1.0 - p_safe;
    double avg_card;
    if (p.hand_known) {
        const double* demand = resource_demand();
        double acc = 0.0;
        for (int r = 0; r < NUM_RESOURCES; ++r) acc += (double)p.resources[r] * demand[r] * pow05_scarcity[r];
        avg_card = acc / (double)cards;
    } else {
        avg_card = 1.0;
    }
    return p_robbed * avg_card;
}

double static_value_ctx(const GameStateC& s, int player, const Context& c) {
    if (s.phase == PHASE_GAME_OVER) {
        if (s.winner == player) return 1000.0;
        if (s.winner >= 0) return -1000.0;
    }
    const PlayerC& p = s.players[player];
    const double vp = (double)s.public_vp(player) + (p.dev_known ? (double)p.vp_cards() : expected_hidden_vp(s, player));
    double score = 10.0 * vp;
    double prod[NUM_RESOURCES], prod_free[NUM_RESOURCES];
    player_production(s, player, false, prod);
    player_production(s, player, true, prod_free);
    const double* demand = resource_demand();
    double pv = 0.0, pv_free = 0.0;  // sum(...) of floats: sequential double accumulation
    for (int r = 0; r < NUM_RESOURCES; ++r) pv += prod[r] * 36.0 * demand[r] * c.sqrt_scarcity[r];
    for (int r = 0; r < NUM_RESOURCES; ++r) pv_free += prod_free[r] * 36.0 * demand[r] * c.sqrt_scarcity[r];
    score += 0.55 * pv + 0.25 * pv_free;
    int types = 0;
    for (int r = 0; r < NUM_RESOURCES; ++r)
        if (prod_free[r] > 0) ++types;
    score += 0.4 * types;
    // Expansion potential.
    if (p.n_settlements < MAX_SETTLEMENTS) {
        ReachableSpots reach;
        reachable_spots(s, player, 2, c.occ, reach);
        int now = 0;
        for (int v = 0; v < NUM_VERTICES; ++v)
            if (reach.dist[v] == 0) ++now;
        score += 0.6 * std::min(now, 3);
        double best_spot = 0.0;
        BlockContext block;
        make_block_context(s, player, c.pow05_scarcity, block);
        for (int v = 0; v < NUM_VERTICES; ++v) {
            const int d = reach.dist[v];
            if (d < 0) continue;
            const double sc = score_spot(s, v, c.occ, prod_free, c.pow05_scarcity, false, block) / (1.0 + 0.9 * d);
            if (sc > best_spot) best_spot = sc;
        }
        score += 0.12 * best_spot;
    }
    // Hand: cards are worth something, too many are a liability.
    const int n = p.hand_known ? p.total_resources() : p.hand_size;
    score += 0.12 * std::min(n, 7) - 0.25 * std::max(0, n - 7);
    // Out-of-turn robber exposure (robber.steal_exposure_fast).
    if (n >= 3) score -= 0.25 * steal_exposure_fast(s, player, c.pow05_scarcity);
    if (p.hand_known) score += progress_to_build(s, player, c.occ);
    // Dev cards.
    if (p.dev_known) {
        const int knights_held = p.dev_cards[DEV_KNIGHT] + p.dev_cards_new[DEV_KNIGHT];
        score += 0.7 * knights_held + 0.9 * (p.dev_cards[DEV_ROAD_BUILDING] + p.dev_cards_new[DEV_ROAD_BUILDING]);
        score += 0.7 * (p.dev_cards[DEV_YEAR_OF_PLENTY] + p.dev_cards_new[DEV_YEAR_OF_PLENTY]);
        score += 0.9 * (p.dev_cards[DEV_MONOPOLY] + p.dev_cards_new[DEV_MONOPOLY]);
    } else {
        score += 0.6 * p.dev_count;
    }
    // Longest road / largest army progress (the awards themselves are in vp).
    const int lr = heuristic_longest_road_length(s, player);
    if (s.longest_road_owner != player) {
        const int holder_len = s.longest_road_owner >= 0 ? s.longest_road_len : 4;
        if (lr >= holder_len) score += 0.35 * lr;
        else score += 0.15 * lr;
    }
    if (s.largest_army_owner != player) score += 0.45 * p.played_knights;
    // Ports that match our production.
    for (int pass = 0; pass < 2; ++pass) {
        const int cnt = pass == 0 ? p.n_settlements : p.n_cities;
        const uint8_t* ids = pass == 0 ? p.settlements : p.cities;
        for (int k = 0; k < cnt; ++k) {
            const int t = s.ports[ids[k]];
            if (t < 0) continue;
            if (t == PORT_GENERIC) score += 0.2;
            else if (t < NUM_RESOURCES) score += 0.15 + 4.0 * prod_free[t];
        }
    }
    return score;
}

// ---------------------------------------------------------------------------
// longest road (heuristic.longest_road_length semantics)
// ---------------------------------------------------------------------------
struct HeuristicRoads {
    int head[NUM_VERTICES];
    int to[2 * NUM_EDGES];
    int edge[2 * NUM_EDGES];
    int next[2 * NUM_EDGES];
    uint8_t blocked[NUM_VERTICES];
    uint8_t used[NUM_EDGES];
    int best;

    void dfs(int v, int length) {
        if (length > best) best = length;
        if (blocked[v]) return;  // cannot continue through an opponent building
        for (int i = head[v]; i != -1; i = next[i]) {
            const int e = edge[i];
            if (used[e]) continue;
            used[e] = 1;
            dfs(to[i], length + 1);
            used[e] = 0;
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// public helpers
// ---------------------------------------------------------------------------
const double* resource_demand() {
    static const double* table = [] {
        static double out[NUM_RESOURCES];
        const double raw[NUM_RESOURCES] = {1.0, 1.0, 0.9, 1.25, 1.2};  // placement._RAW_DEMAND
        double total = 0.0;
        for (int r = 0; r < NUM_RESOURCES; ++r) total += raw[r];  // sum(_RAW_DEMAND)
        const double mean = total / 5.0;
        for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = raw[r] / mean;
        return out;
    }();
    return table;
}

void occupancy(const GameStateC& s, Occupancy& o) {
    std::memset(o.owner, -1, sizeof(o.owner));
    std::memset(o.edge_taken, 0, sizeof(o.edge_taken));
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& p = s.players[i];
        for (int k = 0; k < p.n_settlements; ++k) o.owner[p.settlements[k]] = (int8_t)i;
        for (int k = 0; k < p.n_cities; ++k) o.owner[p.cities[k]] = (int8_t)i;
        for (int k = 0; k < p.n_roads; ++k) o.edge_taken[p.roads[k]] = 1;
    }
    for (int v = 0; v < NUM_VERTICES; ++v) {
        bool free_ = o.owner[v] == -1;
        for (int k = 0; free_ && k < VERTEX_NEIGHBORS_N[v]; ++k)
            if (o.owner[VERTEX_NEIGHBORS[v][k]] != -1) free_ = false;
        o.free_vertex[v] = free_ ? 1 : 0;
    }
}

void vertex_production(const GameStateC& s, int v, bool ignore_robber, double out[NUM_RESOURCES]) {
    for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = 0.0;
    for (int k = 0; k < VERTEX_HEXES_N[v]; ++k) {
        const int h = VERTEX_HEXES[v][k];
        const int res = s.hex_res[h];
        const int num = s.hex_num[h];
        if (res == DESERT || num == 0) continue;
        if (!ignore_robber && h == s.robber) continue;
        out[res] += pips_of(num) / 36.0;
    }
}

void player_production(const GameStateC& s, int player, bool ignore_robber, double out[NUM_RESOURCES]) {
    const PlayerC& p = s.players[player];
    for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = 0.0;
    double pr[NUM_RESOURCES];
    for (int k = 0; k < p.n_settlements; ++k) {
        vertex_production(s, p.settlements[k], ignore_robber, pr);
        for (int r = 0; r < NUM_RESOURCES; ++r) out[r] += pr[r];
    }
    for (int k = 0; k < p.n_cities; ++k) {
        vertex_production(s, p.cities[k], ignore_robber, pr);
        for (int r = 0; r < NUM_RESOURCES; ++r) out[r] += 2.0 * pr[r];
    }
}

void resource_scarcity(const GameStateC& s, double out[NUM_RESOURCES]) {
    int pips[NUM_RESOURCES] = {0, 0, 0, 0, 0};  // placement.board_pips_by_resource
    for (int h = 0; h < NUM_HEXES; ++h) {
        const int res = s.hex_res[h];
        const int num = s.hex_num[h];
        if (res != DESERT && num) pips[res] += pips_of(num);
    }
    int total = 0;
    for (int r = 0; r < NUM_RESOURCES; ++r) total += pips[r];
    const double mean = total / 5.0;
    for (int r = 0; r < NUM_RESOURCES; ++r) out[r] = mean / (double)std::max(pips[r], 1);
}

double expected_hidden_vp(const GameStateC& s, int player) {
    const PlayerC& p = s.players[player];
    if (p.dev_known) return (double)p.vp_cards();
    int pool[NUM_DEV];  // counting.dev_pool
    for (int t = 0; t < NUM_DEV; ++t) pool[t] = DEV_DECK_COUNTS[t];
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& q = s.players[i];
        pool[DEV_KNIGHT] -= q.played_knights;
        if (q.dev_known)
            for (int t = 0; t < NUM_DEV; ++t) pool[t] -= q.dev_cards[t] + q.dev_cards_new[t];
    }
    int total = 0;
    for (int t = 0; t < NUM_DEV; ++t) {
        pool[t] = std::max(0, pool[t]);
        total += pool[t];
    }
    if (total <= 0 || p.dev_count <= 0) return 0.0;
    return (double)(p.dev_count * pool[DEV_VICTORY_POINT]) / (double)total;
}

int heuristic_longest_road_length(const GameStateC& s, int player) {
    const PlayerC& p = s.players[player];
    if (p.n_roads == 0) return 0;
    HeuristicRoads g;
    // occ = state.occupied_vertices(): later players overwrite earlier ones.
    int8_t owner[NUM_VERTICES];
    std::memset(owner, -1, sizeof(owner));
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& q = s.players[i];
        for (int k = 0; k < q.n_settlements; ++k) owner[q.settlements[k]] = (int8_t)i;
        for (int k = 0; k < q.n_cities; ++k) owner[q.cities[k]] = (int8_t)i;
    }
    for (int v = 0; v < NUM_VERTICES; ++v) {
        g.blocked[v] = (owner[v] != -1 && owner[v] != player) ? 1 : 0;
        g.head[v] = -1;
    }
    std::memset(g.used, 0, sizeof(g.used));
    int ne = 0;
    for (int k = 0; k < p.n_roads; ++k) {
        const int e = p.roads[k];
        const int a = EDGE_VERTICES[e][0], b = EDGE_VERTICES[e][1];
        g.to[ne] = b; g.edge[ne] = e; g.next[ne] = g.head[a]; g.head[a] = ne; ++ne;
        g.to[ne] = a; g.edge[ne] = e; g.next[ne] = g.head[b]; g.head[b] = ne; ++ne;
    }
    g.best = 0;
    for (int v = 0; v < NUM_VERTICES; ++v)
        if (g.head[v] != -1) g.dfs(v, 0);
    return g.best;
}

void reachable_spots(const GameStateC& s, int player, int max_roads, const Occupancy& occ, ReachableSpots& out) {
    const PlayerC& p = s.players[player];
    int8_t dist[NUM_VERTICES];
    int8_t first[NUM_VERTICES];
    uint8_t is_start[NUM_VERTICES];
    std::memset(dist, -1, sizeof(dist));
    std::memset(first, -1, sizeof(first));
    std::memset(is_start, 0, sizeof(is_start));
    for (int k = 0; k < p.n_settlements; ++k) is_start[p.settlements[k]] = 1;
    for (int k = 0; k < p.n_cities; ++k) is_start[p.cities[k]] = 1;
    for (int k = 0; k < p.n_roads; ++k) {
        is_start[EDGE_VERTICES[p.roads[k]][0]] = 1;
        is_start[EDGE_VERTICES[p.roads[k]][1]] = 1;
    }
    // BFS over vertices; distance = roads to build.  A start vertex that is an
    // opponent building cannot be expanded from (it cuts the road).
    int queue[4 * NUM_VERTICES];
    int qh = 0, qt = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (!is_start[v]) continue;
        if (occ.owner[v] != -1 && occ.owner[v] != player) continue;
        dist[v] = 0;
        first[v] = -1;
        queue[qt++] = v;
    }
    while (qh < qt) {
        const int v = queue[qh++];
        const int d = dist[v];
        if (d >= max_roads) continue;
        for (int k = 0; k < VERTEX_EDGES_N[v]; ++k) {
            const int e = VERTEX_EDGES[v][k];
            if (occ.edge_taken[e]) continue;
            const int w = VERTEX_NEIGHBORS[v][k];
            if (dist[w] != -1 && dist[w] <= d + 1) continue;
            if (occ.owner[w] != -1 && occ.owner[w] != player) continue;  // opponent building blocks the path
            dist[w] = (int8_t)(d + 1);
            first[w] = first[v] == -1 ? (int8_t)e : first[v];
            if (qt < 4 * NUM_VERTICES) queue[qt++] = w;  // never full: each vertex is discovered once
        }
    }
    out.count = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (dist[v] != -1 && occ.free_vertex[v]) {
            out.dist[v] = dist[v];
            out.first[v] = first[v];
            ++out.count;
        } else {
            out.dist[v] = -1;
            out.first[v] = -1;
        }
    }
}

double score_settlement_spot(const GameStateC& s, int player, int v, const Occupancy& occ,
                             const double own_prod[NUM_RESOURCES], const double scarcity[NUM_RESOURCES],
                             bool setup) {
    double pow05[NUM_RESOURCES];
    for (int r = 0; r < NUM_RESOURCES; ++r) pow05[r] = pow_half(scarcity[r]);
    BlockContext block;  // placement.score_settlement_spot builds one when block_ctx is None
    make_block_context(s, player, pow05, block);
    return score_spot(s, v, occ, own_prod, pow05, setup, block);
}

double progress_to_build(const GameStateC& s, int player, const Occupancy& occ) {
    const PlayerC& p = s.players[player];
    const int32_t* hand = p.resources;
    double best = 0.0;
    bool spots = false;  // placement.buildable_settlements(...) non-empty
    if (p.n_settlements < MAX_SETTLEMENTS) {
        for (int k = 0; k < p.n_roads && !spots; ++k) {
            const int e = p.roads[k];
            if (occ.free_vertex[EDGE_VERTICES[e][0]] || occ.free_vertex[EDGE_VERTICES[e][1]]) spots = true;
        }
    }
    const int* costs[3];
    double values[3];
    int n_opt = 0;
    if (p.n_settlements > 0 && p.n_cities < MAX_CITIES) { costs[n_opt] = COST_CITY; values[n_opt++] = VALUE_CITY; }
    if (spots) { costs[n_opt] = COST_SETTLEMENT; values[n_opt++] = VALUE_SETTLEMENT; }
    int deck = 0;
    for (int t = 0; t < NUM_DEV; ++t) deck += s.dev_deck[t];
    if (deck > 0) { costs[n_opt] = COST_DEV; values[n_opt++] = VALUE_DEV; }
    for (int i = 0; i < n_opt; ++i) {
        int total = 0, have = 0;
        for (int r = 0; r < NUM_RESOURCES; ++r) {
            total += costs[i][r];
            have += std::min(hand[r], costs[i][r]);
        }
        const double frac = (double)have / (double)total;
        const double cand = values[i] * 0.45 * frac * frac;
        if (cand > best) best = cand;
    }
    return best;
}

double static_value(const GameStateC& s, int player) {
    Context c;
    make_context(s, c);
    return static_value_ctx(s, player, c);
}

void static_values(const GameStateC& s, double* out) {
    Context c;
    make_context(s, c);
    for (int i = 0; i < s.num_players; ++i) out[i] = static_value_ctx(s, i, c);
}

double heuristic_softmax(const double* values, int n, int player, double temperature) {
    double m = values[0];
    for (int i = 1; i < n; ++i)
        if (values[i] > m) m = values[i];
    double exps[MAX_PLAYERS];
    double total = 0.0;  // sum(exps): sequential double accumulation
    for (int i = 0; i < n; ++i) {
        exps[i] = std::exp((values[i] - m) / temperature);
        total += exps[i];
    }
    return exps[player] / total;
}

double heuristic_win_probability(const GameStateC& s, const double* values, int player, double temperature) {
    if (s.phase == PHASE_GAME_OVER && s.winner >= 0) return s.winner == player ? 1.0 : 0.0;
    return heuristic_softmax(values, s.num_players, player, temperature);
}

}  // namespace catanbot
