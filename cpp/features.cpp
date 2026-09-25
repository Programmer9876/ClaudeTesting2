// Exact port of catanbot/features.py::extract.
//
// Every arithmetic expression below mirrors the Python statement it comes from,
// including the evaluation order of floating point accumulations and the point
// where a value is narrowed to float32, so the output is bit-identical to the
// numpy reference (the differential test in tests/test_accel_features.py
// checks it to 1e-6).  Keep this file in sync with features.py - the feature
// offsets (P::*, G::*) and NUM_FEATURES come from the generated
// feature_layout.hpp, so a layout change fails to compile or is caught by the
// feature_names() check in catanbot/accel.py.
#define CATAN_NO_PYTHON
#include "features.hpp"
#undef CATAN_NO_PYTHON

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace catanbot {

namespace {

inline int pips_of(int num) { return (num >= 0 && num < 13) ? PIPS[num] : 0; }

// Python's `//` for the (non-negative in practice) hand size.
inline int floordiv2(int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); }

// math.log(5.0) evaluated at run time by the same libm Python uses (no constant folding).
double log5() {
    static const double v = [] {
        volatile double five = 5.0;
        return std::log(five);
    }();
    return v;
}

// ---------------------------------------------------------------------------
// settlement spots (features._spots_for_player)
// ---------------------------------------------------------------------------
void spots_for_player(const GameStateC& s, int player, const int8_t* owner, const uint8_t* edge_taken,
                      const uint8_t* settlement_ok, const double* pips, Analysis& a) {
    const PlayerC& p = s.players[player];
    int n_now = 0, n_one = 0, n_two = 0;
    double best_now = 0.0, best_two = 0.0;
    if (p.n_roads > 0) {
        uint8_t seen[NUM_VERTICES];
        std::memset(seen, 0, sizeof(seen));
        int frontier[NUM_VERTICES];
        int nf = 0;
        for (int k = 0; k < p.n_roads; ++k) {
            const int e = p.roads[k];
            for (int j = 0; j < 2; ++j) {
                const int v = EDGE_VERTICES[e][j];
                if (!seen[v]) {
                    seen[v] = 1;
                    frontier[nf++] = v;
                }
            }
        }
        for (int i = 0; i < nf; ++i) {
            const int v = frontier[i];
            if (settlement_ok[v]) {
                ++n_now;
                if (pips[v] > best_now) best_now = pips[v];  // max(..., default=0.0)
            }
        }
        best_two = best_now;
        int nxt[NUM_VERTICES];
        for (int level = 0; level < 2; ++level) {
            int nn = 0;
            for (int i = 0; i < nf; ++i) {
                const int v = frontier[i];
                const int o = owner[v];
                if (o != -1 && o != player) continue;  // cannot build a road out of an opponent's building
                for (int k = 0; k < VERTEX_EDGES_N[v]; ++k) {
                    const int e = VERTEX_EDGES[v][k];
                    if (edge_taken[e]) continue;
                    const int w = VERTEX_NEIGHBORS[v][k];
                    if (!seen[w]) {
                        seen[w] = 1;
                        nxt[nn++] = w;
                    }
                }
            }
            int count = 0;
            for (int i = 0; i < nn; ++i) {
                const int w = nxt[i];
                if (settlement_ok[w]) {
                    ++count;
                    if (pips[w] > best_two) best_two = pips[w];
                }
            }
            if (level == 0) n_one = count; else n_two = count;
            std::memcpy(frontier, nxt, nn * sizeof(int));
            nf = nn;
        }
    }
    a.spots_now[player] = n_now;
    a.spots_one[player] = n_one;
    a.spots_two[player] = n_two;
    a.best_now[player] = best_now;
    a.best_two[player] = best_two;
}

// ---------------------------------------------------------------------------
// longest road DFS
// ---------------------------------------------------------------------------
struct RoadGraph {
    // adjacency as singly linked lists: head[v] -> entry index, entries hold (road bit, other vertex, next)
    int head[NUM_VERTICES];
    int bit[2 * NUM_EDGES];
    int to[2 * NUM_EDGES];
    int next[2 * NUM_EDGES];
    uint8_t blocked[NUM_VERTICES];

    int dfs(int v, uint64_t used) const {
        int best = 0;
        for (int i = head[v]; i != -1; i = next[i]) {
            const uint64_t b = (uint64_t)1 << bit[i];
            if (used & b) continue;
            const int w = to[i];
            const int length = blocked[w] ? 1 : 1 + dfs(w, used | b);
            if (length > best) best = length;
        }
        return best;
    }
};

}  // namespace

int longest_road_length(const GameStateC& s, int player) {
    const PlayerC& p = s.players[player];
    const int nr = p.n_roads;
    if (nr == 0) return 0;
    if (nr > 64)  // one "used" bit per entry; the Python reference has no such limit -> accel falls back to it
        unsupported("more than 64 road entries for one player (features.longest_road_length)");
    RoadGraph g;
    std::memset(g.blocked, 0, sizeof(g.blocked));
    for (int i = 0; i < s.num_players; ++i) {
        if (i == player) continue;
        const PlayerC& q = s.players[i];
        for (int k = 0; k < q.n_settlements; ++k) g.blocked[q.settlements[k]] = 1;
        for (int k = 0; k < q.n_cities; ++k) g.blocked[q.cities[k]] = 1;
    }
    for (int v = 0; v < NUM_VERTICES; ++v) g.head[v] = -1;
    int ne = 0;
    for (int k = 0; k < nr; ++k) {
        const int e = p.roads[k];
        const int a = EDGE_VERTICES[e][0], b = EDGE_VERTICES[e][1];
        g.bit[ne] = k; g.to[ne] = b; g.next[ne] = g.head[a]; g.head[a] = ne; ++ne;
        g.bit[ne] = k; g.to[ne] = a; g.next[ne] = g.head[b]; g.head[b] = ne; ++ne;
    }
    int best = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (g.head[v] == -1) continue;
        const int length = g.dfs(v, 0);
        if (length > best) {
            best = length;
            if (best == nr) break;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// per-state analysis
// ---------------------------------------------------------------------------
namespace {

void hidden_vp_estimates(const GameStateC& s, Analysis& a) {
    const int n = s.num_players;
    int known_vp = 0;
    int unknown_total = 0;
    for (int i = 0; i < n; ++i) {
        const PlayerC& p = s.players[i];
        if (p.dev_known) known_vp += p.vp_cards();
        else unknown_total += p.dev_count;
    }
    double frac = 0.0;
    if (unknown_total > 0) {
        const int deck_vp = s.dev_deck[DEV_VICTORY_POINT];
        const int vp_unknown = std::min(unknown_total, std::max(0, DEV_DECK_COUNTS[DEV_VICTORY_POINT] - known_vp - deck_vp));
        frac = (double)vp_unknown / (double)unknown_total;
    }
    for (int i = 0; i < n; ++i) {
        const PlayerC& p = s.players[i];
        a.hidden[i] = p.dev_known ? (double)p.vp_cards() : (double)p.dev_count * frac;
    }
}

void player_blocks(const GameStateC& s, Analysis& a) {
    const int n = s.num_players;
    int deck_total = 0;
    for (int k = 0; k < NUM_DEV; ++k) deck_total += s.dev_deck[k];
    const bool deck_left = deck_total > 0;
    for (int i = 0; i < n; ++i) {
        const PlayerC& p = s.players[i];
        float* row = a.blocks[i];
        std::memset(row, 0, PLAYER_BLOCK * sizeof(float));
        const int pub = s.public_vp(i);
        row[P::public_vp] = (float)(pub / 10.0);
        row[P::hidden_vp] = (float)(a.hidden[i] / 5.0);
        row[P::total_vp] = (float)((pub + a.hidden[i]) / 10.0);
        const int ns = p.n_settlements, nc = p.n_cities, nr = p.n_roads;
        row[P::settlements] = (float)(ns / 5.0);
        row[P::cities] = (float)(nc / 4.0);
        row[P::roads] = (float)(nr / 15.0);
        // --- hand ---------------------------------------------------------
        const int32_t* res = p.resources;
        const int hand = p.hand_known ? p.total_resources() : p.hand_size;
        if (p.hand_known) {
            for (int k = 0; k < 5; ++k) row[P::res_wood + k] = (float)(res[k] / 10.0);
        } else {
            double tot = 0.0;  // float(prod_nr[i].sum())
            for (int k = 0; k < 5; ++k) tot += a.prod_nr[i][k];
            if (tot > 0) {
                for (int k = 0; k < 5; ++k) row[P::res_wood + k] = (float)(hand * a.prod_nr[i][k] / tot / 10.0);
            } else {
                for (int k = 0; k < 5; ++k) row[P::res_wood + k] = (float)(hand / 5.0 / 10.0);
            }
        }
        row[P::hand_known] = p.hand_known ? 1.0f : 0.0f;
        row[P::hand_size] = (float)(hand / 15.0);
        if (hand > 7) {
            row[P::cards_over_7] = (float)((hand - 7) / 8.0);
            row[P::discard_exposure] = (float)(floordiv2(hand) / 8.0);
        }
        // --- production ---------------------------------------------------
        double tot_r = 0.0, tot_nr = 0.0, ent = 0.0;
        int types = 0;
        for (int k = 0; k < 5; ++k) {
            const double pa = a.prod[i][k];
            const double pb = a.prod_nr[i][k];
            row[P::prod_wood + k] = (float)(pa / 2.0);
            row[P::prod_nr_wood + k] = (float)(pb / 2.0);
            tot_r += pa;
            tot_nr += pb;
            if (pb > 0) ++types;
        }
        if (tot_nr > 0) {
            for (int k = 0; k < 5; ++k) {
                const double pb = a.prod_nr[i][k];
                if (pb > 0) {
                    const double f = pb / tot_nr;
                    ent -= f * std::log(f);
                }
            }
            ent /= log5();
        }
        row[P::prod_total] = (float)(tot_r / 5.0);
        row[P::prod_nr_total] = (float)(tot_nr / 5.0);
        row[P::robber_loss] = (float)((tot_nr - tot_r) / 2.0);
        row[P::prod_types] = (float)(types / 5.0);
        row[P::prod_entropy] = (float)ent;
        row[P::robber_on_my_hex] = (float)a.robber_touch[i];
        // --- expansion ----------------------------------------------------
        row[P::spots_now] = (float)(a.spots_now[i] / 10.0);
        row[P::spots_1road] = (float)(a.spots_one[i] / 10.0);
        row[P::spots_2roads] = (float)(a.spots_two[i] / 15.0);
        row[P::best_spot_pips_now] = (float)(a.best_now[i] / 15.0);
        row[P::best_spot_pips_2roads] = (float)(a.best_two[i] / 15.0);
        // --- affordability --------------------------------------------------
        if (p.hand_known) {
            const int w = res[0], b_ = res[1], sh = res[2], wh = res[3], o = res[4];
            row[P::can_build_road] = (w >= 1 && b_ >= 1 && nr < MAX_ROADS) ? 1.0f : 0.0f;
            row[P::can_build_settlement] =
                (w >= 1 && b_ >= 1 && sh >= 1 && wh >= 1 && ns < MAX_SETTLEMENTS && a.spots_now[i] > 0) ? 1.0f : 0.0f;
            row[P::can_build_city] = (wh >= 2 && o >= 3 && ns > 0 && nc < MAX_CITIES) ? 1.0f : 0.0f;
            row[P::can_buy_dev] = (sh >= 1 && wh >= 1 && o >= 1 && deck_left) ? 1.0f : 0.0f;
        }
        // --- awards ----------------------------------------------------------
        row[P::longest_road] = (float)(a.lr[i] / 15.0);
        row[P::has_longest_road] = (s.longest_road_owner == i) ? 1.0f : 0.0f;
        row[P::knights] = (float)(p.played_knights / 5.0);
        row[P::has_largest_army] = (s.largest_army_owner == i) ? 1.0f : 0.0f;
        // --- dev cards -------------------------------------------------------
        if (p.dev_known) {
            for (int k = 0; k < 5; ++k) {
                row[P::dev_knight + k] = (float)(p.dev_cards[k] / 3.0);
                row[P::devnew_knight + k] = (float)(p.dev_cards_new[k] / 3.0);
            }
            row[P::dev_total] = (float)(p.total_dev() / 5.0);
            row[P::dev_known] = 1.0f;
        } else {
            const int cnt = p.dev_count;
            if (cnt > 0) {
                const double rest = std::max(0.0, cnt - a.hidden[i]);
                int non_vp[5];
                int tot_i = 0;
                for (int k = 0; k < 5; ++k) {
                    non_vp[k] = (k != DEV_VICTORY_POINT) ? s.dev_deck[k] : 0;
                    tot_i += non_vp[k];
                }
                const double tot = (double)tot_i;
                for (int k = 0; k < 5; ++k) {
                    if (k == DEV_VICTORY_POINT) row[P::dev_knight + k] = (float)(a.hidden[i] / 3.0);
                    else if (tot > 0) row[P::dev_knight + k] = (float)(rest * non_vp[k] / tot / 3.0);
                    else row[P::dev_knight + k] = (float)(rest / 4.0 / 3.0);
                }
            }
            row[P::dev_total] = (float)(cnt / 5.0);
        }
        // --- ports -------------------------------------------------------------
        bool generic = false;
        for (int k = 0; k < ns; ++k) {
            const int t = s.ports[p.settlements[k]];
            if (t < 0) continue;
            if (t == PORT_GENERIC) generic = true;
            else if (t < NUM_RESOURCES) row[P::port_wood + t] = 1.0f;
        }
        for (int k = 0; k < nc; ++k) {
            const int t = s.ports[p.cities[k]];
            if (t < 0) continue;
            if (t == PORT_GENERIC) generic = true;
            else if (t < NUM_RESOURCES) row[P::port_wood + t] = 1.0f;
        }
        if (generic) {
            for (int k = 0; k < 5; ++k)
                if (row[P::port_wood + k] < 0.5f) row[P::port_wood + k] = 0.5f;
        }
        row[P::is_current] = (s.current == i) ? 1.0f : 0.0f;
        row[P::present] = 1.0f;
    }
}

}  // namespace

void analyse(const GameStateC& s, Analysis& a) {
    const int n = s.num_players;
    // --- vertex production tables (features.vertex_production_tables) ---------
    for (int v = 0; v < NUM_VERTICES; ++v) {
        double* pr = a.prod_tab[v];
        for (int r = 0; r < NUM_RESOURCES; ++r) pr[r] = 0.0;
        a.pips[v] = 0.0;
        for (int k = 0; k < VERTEX_HEXES_N[v]; ++k) {
            const int h = VERTEX_HEXES[v][k];
            const int res = s.hex_res[h];
            if (res == DESERT) continue;
            const int p = pips_of(s.hex_num[h]);
            pr[res] += p / 36.0;
            a.pips[v] += p;
        }
    }
    // --- occupancy ----------------------------------------------------------------
    int8_t owner[NUM_VERTICES];
    uint8_t edge_taken[NUM_EDGES];
    uint8_t settlement_ok[NUM_VERTICES];
    std::memset(owner, -1, sizeof(owner));
    std::memset(edge_taken, 0, sizeof(edge_taken));
    std::memset(settlement_ok, 1, sizeof(settlement_ok));
    for (int i = 0; i < n; ++i) {
        const PlayerC& p = s.players[i];
        for (int k = 0; k < p.n_settlements; ++k) owner[p.settlements[k]] = (int8_t)i;
        for (int k = 0; k < p.n_cities; ++k) owner[p.cities[k]] = (int8_t)i;
        for (int k = 0; k < p.n_roads; ++k) edge_taken[p.roads[k]] = 1;
    }
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (owner[v] == -1) continue;
        settlement_ok[v] = 0;
        for (int k = 0; k < VERTEX_NEIGHBORS_N[v]; ++k) settlement_ok[VERTEX_NEIGHBORS[v][k]] = 0;
    }
    // --- robber -----------------------------------------------------------------------
    const int robber = s.robber;
    uint64_t rob_mask = 0;
    std::memcpy(a.prod_r_tab, a.prod_tab, sizeof(a.prod_tab));
    if (robber >= 0 && robber < NUM_HEXES) {
        for (int k = 0; k < 6; ++k) rob_mask |= (uint64_t)1 << HEX_VERTICES[robber][k];
        const int rob_res = s.hex_res[robber];
        if (rob_res != DESERT) {
            const double rob_p = pips_of(s.hex_num[robber]) / 36.0;
            for (int k = 0; k < 6; ++k) a.prod_r_tab[HEX_VERTICES[robber][k]][rob_res] -= rob_p;
        }
    }
    // --- per player -----------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        const PlayerC& p = s.players[i];
        double* row_nr = a.prod_nr[i];
        double* row = a.prod[i];
        for (int r = 0; r < NUM_RESOURCES; ++r) row_nr[r] = row[r] = 0.0;
        a.robber_touch[i] = 0.0;
        for (int k = 0; k < p.n_settlements; ++k) {
            const int v = p.settlements[k];
            for (int r = 0; r < NUM_RESOURCES; ++r) {
                row_nr[r] += a.prod_tab[v][r];
                row[r] += a.prod_r_tab[v][r];
            }
            if ((rob_mask >> v) & 1u) a.robber_touch[i] = 1.0;
        }
        for (int k = 0; k < p.n_cities; ++k) {
            const int v = p.cities[k];
            for (int r = 0; r < NUM_RESOURCES; ++r) {
                row_nr[r] += a.prod_tab[v][r];
                row_nr[r] += a.prod_tab[v][r];
                row[r] += a.prod_r_tab[v][r];
                row[r] += a.prod_r_tab[v][r];
            }
            if ((rob_mask >> v) & 1u) a.robber_touch[i] = 1.0;
        }
        spots_for_player(s, i, owner, edge_taken, settlement_ok, a.pips, a);
        a.lr[i] = longest_road_length(s, i);
    }
    hidden_vp_estimates(s, a);
    player_blocks(s, a);
}

// ---------------------------------------------------------------------------
// perspective assembly (features._assemble + _global_block)
// ---------------------------------------------------------------------------
void assemble(const GameStateC& s, const Analysis& a, int player, float* out) {
    const int n = s.num_players;
    std::memset(out, 0, NUM_FEATURES * sizeof(float));
    std::memcpy(out, a.blocks[player], PLAYER_BLOCK * sizeof(float));
    for (int k = 1; k < MAX_PLAYERS; ++k) {
        if (k < n) {
            const int j = (player + k) % n;
            std::memcpy(out + k * PLAYER_BLOCK, a.blocks[j], PLAYER_BLOCK * sizeof(float));
        }
    }
    float* g = out + GLOBAL_OFFSET;
    for (int k = 0; k < 5; ++k) g[G::bank_wood + k] = (float)(s.bank[k] / (double)BANK_PER_RESOURCE);
    int deck_sum = 0;
    for (int k = 0; k < 5; ++k) {
        g[G::deck_knight + k] = (float)(s.dev_deck[k] / (double)DEV_DECK_COUNTS[k]);
        deck_sum += s.dev_deck[k];
    }
    g[G::deck_total] = (float)(deck_sum / 25.0);
    g[G::turn] = (float)std::min(1.0, s.turn / 200.0);
    g[G::stage] = (float)std::min(1.0, s.turn / (30.0 * std::max(1, n)));
    if (n >= 2 && n <= 4) g[G::players_2 + (n - 2)] = 1.0f;
    if (s.phase != PHASE_UNKNOWN) g[G::phase_setup_settlement + (int)s.phase] = 1.0f;
    g[G::my_turn] = (s.current == player) ? 1.0f : 0.0f;
    g[G::i_am_acting] = (s.acting_player() == player) ? 1.0f : 0.0f;
    g[G::dice] = (float)(s.dice / 12.0);
    g[G::free_roads] = (float)(s.free_roads / 2.0);
    g[G::dev_played_this_turn] = s.dev_played_this_turn ? 1.0f : 0.0f;
    g[G::trades_this_turn] = (float)(s.trades_this_turn / 4.0);
    // --- gaps (computed from the float32 block values, like the numpy reference) ---
    const double my_pub = a.blocks[player][P::public_vp];
    const double my_tot = a.blocks[player][P::total_vp];
    double best_pub = 0.0, best_tot = 0.0;
    int best_lr = 0, best_kn = 0;
    double leader = my_pub;
    for (int j = 0; j < n; ++j) {
        const double pub_j = a.blocks[j][P::public_vp];
        if (pub_j > leader) leader = pub_j;
        if (j == player) continue;
        if (pub_j > best_pub) best_pub = pub_j;
        const double tot_j = a.blocks[j][P::total_vp];
        if (tot_j > best_tot) best_tot = tot_j;
        if (a.lr[j] > best_lr) best_lr = a.lr[j];
        const int kn = s.players[j].played_knights;
        if (kn > best_kn) best_kn = kn;
    }
    g[G::vp_gap] = (float)(my_pub - best_pub);
    g[G::vp_gap_expected] = (float)(my_tot - best_tot);
    g[G::leader_vp] = (float)leader;
    g[G::i_am_leader] = (my_pub >= leader) ? 1.0f : 0.0f;
    g[G::lr_gap] = (float)((a.lr[player] - best_lr) / 5.0);
    g[G::knight_gap] = (float)((s.players[player].played_knights - best_kn) / 5.0);
    // --- pending trade -----------------------------------------------------
    if (s.has_pending_trade) {
        const TradeC& t = s.pending_trade;
        g[G::trade_pending] = 1.0f;
        g[G::trade_i_propose] = (t.proposer == player) ? 1.0f : 0.0f;
        g[G::trade_i_respond] = (s.phase == PHASE_TRADE_RESPONSE && s.trade_responder == player) ? 1.0f : 0.0f;
        for (int k = 0; k < 5; ++k) {
            g[G::trade_give_wood + k] = (float)(t.give[k] / 3.0);
            g[G::trade_get_wood + k] = (float)(t.get[k] / 3.0);
        }
    }
}

}  // namespace catanbot
