// C++ port of catanbot/engine.py.  See engine.hpp for the contract.
//
// Every handler is a line-by-line translation of the corresponding `_h_*`
// function: the same checks in the same order with the same messages, the same
// mutations in the same order, the same "touched" players (whose hand_size /
// dev_count are re-synced afterwards).  Ordered id lists (settlements, cities,
// roads) are kept in the Python order, membership tests use the bitsets that
// PlayerC maintains alongside them.
#include "engine.hpp"

#include <algorithm>
#include <cstring>

#include "features.hpp"  // features.longest_road_length == engine.longest_road_length

namespace catanbot {

// ---------------------------------------------------------------------------
// random sources
// ---------------------------------------------------------------------------
namespace {
inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
}  // namespace

void Xoshiro::reseed(uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; ++i) st[i] = splitmix64(x);
}

uint64_t Xoshiro::next() {
    const uint64_t result = rotl(st[1] * 5, 7) * 9;
    const uint64_t t = st[1] << 17;
    st[2] ^= st[0];
    st[3] ^= st[1];
    st[1] ^= st[2];
    st[0] ^= st[3];
    st[2] ^= t;
    st[3] = rotl(st[3], 45);
    return result;
}

int Xoshiro::randrange(int n) {
    if (n <= 0) throw std::invalid_argument("randrange(n) needs n >= 1");
    return (int)((unsigned __int128)next() * (uint64_t)n >> 64);  // Lemire's multiply-shift
}

int Xoshiro::randint(int lo, int hi) { return lo + randrange(hi - lo + 1); }

int ForcedDraw::randrange(int n) {
    ++calls;
    if (index < 0 || index >= n)
        throw std::invalid_argument("drawn_index " + std::to_string(index) + " out of range for " +
                                    std::to_string(n) + " possible draws");
    return index;
}

int ForcedDraw::randint(int lo, int hi) {
    const int span = hi - lo + 1;
    if (index < 0 || index >= span * span)
        throw std::invalid_argument("drawn_index " + std::to_string(index) + " out of range for a forced roll (0..35)");
    const int v = (calls++ == 0) ? index / span : index % span;
    return lo + v;
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
namespace {

[[noreturn]] inline void fail(const char* msg) { throw illegal_action(msg); }

inline uint32_t bit(int i) { return (uint32_t)1 << i; }

inline int floor_div2(int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); }

inline int pmod(int a, int n) {  // Python's a % n (n <= 0: ZeroDivisionError there, ValueError here)
    if (n <= 0) throw std::domain_error("modulo by zero (state has no players)");
    const int r = a % n;
    return r < 0 ? r + n : r;
}

// Python's state.players[i] raises IndexError outside the list (a negative index would
// wrap; such states are not produced by the engine and are rejected here as well).
inline PlayerC& player_at(GameStateC& s, int i) {
    if (i < 0 || i >= s.num_players) throw std::out_of_range("player index " + std::to_string(i) + " out of range");
    return s.players[i];
}
inline const PlayerC& player_at(const GameStateC& s, int i) {
    if (i < 0 || i >= s.num_players) throw std::out_of_range("player index " + std::to_string(i) + " out of range");
    return s.players[i];
}

// engine._vertex_owners / _vertex_owners_weights / _edge_owners (later players overwrite).
void vertex_owners(const GameStateC& s, int8_t* vo) {
    std::memset(vo, -1, NUM_VERTICES);
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& p = s.players[i];
        for (int k = 0; k < p.n_settlements; ++k) vo[p.settlements[k]] = (int8_t)i;
        for (int k = 0; k < p.n_cities; ++k) vo[p.cities[k]] = (int8_t)i;
    }
}

void vertex_owners_weights(const GameStateC& s, int8_t* vo, int8_t* vw) {
    std::memset(vo, -1, NUM_VERTICES);
    std::memset(vw, 0, NUM_VERTICES);
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& p = s.players[i];
        for (int k = 0; k < p.n_settlements; ++k) {
            vo[p.settlements[k]] = (int8_t)i;
            vw[p.settlements[k]] = 1;
        }
        for (int k = 0; k < p.n_cities; ++k) {
            vo[p.cities[k]] = (int8_t)i;
            vw[p.cities[k]] = 2;
        }
    }
}

void edge_owners(const GameStateC& s, int8_t* eo) {
    std::memset(eo, -1, NUM_EDGES);
    for (int i = 0; i < s.num_players; ++i) {
        const PlayerC& p = s.players[i];
        for (int k = 0; k < p.n_roads; ++k) eo[p.roads[k]] = (int8_t)i;
    }
}

// list.append / list.remove on the ordered id lists, keeping the bitsets in sync.
inline void add_settlement(PlayerC& p, int v) {
    if (p.n_settlements >= NUM_VERTICES) unsupported("Player.settlements is full");
    p.settlements[p.n_settlements++] = (uint8_t)v;
    p.settlement_bits |= (uint64_t)1 << v;
}
inline void add_city(PlayerC& p, int v) {
    if (p.n_cities >= NUM_VERTICES) unsupported("Player.cities is full");
    p.cities[p.n_cities++] = (uint8_t)v;
    p.city_bits |= (uint64_t)1 << v;
}
inline void add_road(PlayerC& p, int e) {
    if (p.n_roads >= NUM_EDGES) unsupported("Player.roads is full");
    p.roads[p.n_roads++] = (uint8_t)e;
    p.road_bits[e >> 6] |= (uint64_t)1 << (e & 63);
}
// list.remove(v): first occurrence; the bit stays while a duplicate remains.
inline void remove_settlement(PlayerC& p, int v) {
    int k = 0;
    while (k < p.n_settlements && p.settlements[k] != v) ++k;
    if (k == p.n_settlements) return;
    for (; k + 1 < p.n_settlements; ++k) p.settlements[k] = p.settlements[k + 1];
    --p.n_settlements;
    bool still = false;
    for (int j = 0; j < p.n_settlements; ++j)
        if (p.settlements[j] == v) still = true;
    if (!still) p.settlement_bits &= ~((uint64_t)1 << v);
}

inline void sync(PlayerC& p) {  // engine._sync
    p.hand_size = p.total_resources();
    p.dev_count = p.total_dev();
}

inline bool afford(const int32_t* res, const int* cost) {  // engine._afford
    for (int i = 0; i < NUM_RESOURCES; ++i)
        if (cost[i] && res[i] < cost[i]) return false;
    return true;
}

inline void pay(PlayerC& p, int32_t* bank, const int* cost) {  // engine._pay
    for (int i = 0; i < NUM_RESOURCES; ++i) {
        if (!cost[i]) continue;
        p.resources[i] -= cost[i];
        bank[i] += cost[i];
    }
}

inline bool can_pay(const PlayerC& p, const int32_t* counts) {  // engine._can_pay
    const int32_t* r = p.resources;
    return r[0] >= counts[0] && r[1] >= counts[1] && r[2] >= counts[2] && r[3] >= counts[3] && r[4] >= counts[4];
}

inline bool has_cards(const PlayerC& p) {
    const int32_t* r = p.resources;
    return r[0] || r[1] || r[2] || r[3] || r[4];
}

// engine._steal: one uniformly random card (rng.randrange(total)) from victim to thief.
bool steal(PlayerC& thief, PlayerC& victim, DrawSource& draw) {
    int32_t* vr = victim.resources;
    const int total = vr[0] + vr[1] + vr[2] + vr[3] + vr[4];
    if (total <= 0) return false;
    int idx = draw.randrange(total);
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        const int c = vr[r];
        if (idx < c) {
            vr[r] -= 1;
            thief.resources[r] += 1;
            return true;
        }
        idx -= c;
    }
    return false;
}

// engine._port_ratios: bank ratio per resource the player could give (4, 3 or 2).
void port_ratios(const GameStateC& s, int i, int* ratios) {
    for (int r = 0; r < NUM_RESOURCES; ++r) ratios[r] = 4;
    const PlayerC& p = s.players[i];
    bool generic = false;
    for (int k = 0; k < p.n_settlements; ++k) {
        const int t = s.ports[p.settlements[k]];
        if (t == PORT_GENERIC) generic = true;
        else if (t >= 0 && t < NUM_RESOURCES) ratios[t] = 2;
    }
    for (int k = 0; k < p.n_cities; ++k) {
        const int t = s.ports[p.cities[k]];
        if (t == PORT_GENERIC) generic = true;
        else if (t >= 0 && t < NUM_RESOURCES) ratios[t] = 2;
    }
    if (generic)
        for (int r = 0; r < NUM_RESOURCES; ++r)
            if (ratios[r] > 3) ratios[r] = 3;
}

// GameState.port_ratio(i, res): best ratio for giving `res` (settlements then cities, first 2:1 wins).
int port_ratio(const GameStateC& s, int i, int res) {
    const PlayerC& p = s.players[i];
    int ratio = 4;
    for (int k = 0; k < p.n_settlements + p.n_cities; ++k) {
        const int v = k < p.n_settlements ? p.settlements[k] : p.cities[k - p.n_settlements];
        const int t = s.ports[v];
        if (t < 0) continue;
        if (t == PORT_GENERIC) ratio = std::min(ratio, 3);
        else if (t == res) return 2;
    }
    return ratio;
}

// ---------------------------------------------------------------------------
// board-legality helpers (engine._settlement_vertices / _free_vertices / _road_edges / ...)
// ---------------------------------------------------------------------------
inline bool distance_ok(const int8_t* vo, int v) {
    for (int k = 0; k < VERTEX_NEIGHBORS_N[v]; ++k)
        if (vo[VERTEX_NEIGHBORS[v][k]] != -1) return false;
    return true;
}

// Vertices where `me` may build a settlement now (distance rule + own road), sorted.
int settlement_vertices(const GameStateC& s, int me, const int8_t* vo, int* out) {
    const PlayerC& p = s.players[me];
    uint8_t seen[NUM_VERTICES];
    std::memset(seen, 0, sizeof(seen));
    int n = 0;
    for (int k = 0; k < p.n_roads; ++k) {
        const int e = p.roads[k];
        for (int j = 0; j < 2; ++j) {
            const int v = EDGE_VERTICES[e][j];
            if (seen[v]) continue;
            seen[v] = 1;
            if (vo[v] != -1) continue;
            if (distance_ok(vo, v)) out[n++] = v;
        }
    }
    std::sort(out, out + n);
    return n;
}

// Vertices satisfying the distance rule (setup placement), ascending.
int free_vertices(const int8_t* vo, int* out) {
    int n = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (vo[v] != -1) continue;
        if (distance_ok(vo, v)) out[n++] = v;
    }
    return n;
}

// Unoccupied edges connected to an own road / building (opponent buildings cut), sorted.
int road_edges(const GameStateC& s, int me, const int8_t* vo, const int8_t* eo, int* out) {
    const PlayerC& p = s.players[me];
    uint8_t mark[NUM_EDGES];
    std::memset(mark, 0, sizeof(mark));
    int n = 0;
    for (int k = 0; k < p.n_roads; ++k) {
        const int e = p.roads[k];
        for (int j = 0; j < 2; ++j) {
            const int v = EDGE_VERTICES[e][j];
            const int o = vo[v];
            if (o == -1 || o == me) {
                for (int q = 0; q < VERTEX_EDGES_N[v]; ++q) {
                    const int f = VERTEX_EDGES[v][q];
                    if (eo[f] == -1 && !mark[f]) {
                        mark[f] = 1;
                        out[n++] = f;
                    }
                }
            }
        }
    }
    for (int k = 0; k < p.n_settlements + p.n_cities; ++k) {
        const int v = k < p.n_settlements ? p.settlements[k] : p.cities[k - p.n_settlements];
        for (int q = 0; q < VERTEX_EDGES_N[v]; ++q) {
            const int f = VERTEX_EDGES[v][q];
            if (eo[f] == -1 && !mark[f]) {
                mark[f] = 1;
                out[n++] = f;
            }
        }
    }
    std::sort(out, out + n);
    return n;
}

inline bool any_road_edge(const GameStateC& s, int me) {
    int8_t vo[NUM_VERTICES], eo[NUM_EDGES];
    vertex_owners(s, vo);
    edge_owners(s, eo);
    int tmp[NUM_EDGES];
    return road_edges(s, me, vo, eo, tmp) > 0;
}

// Edge `e` touches an own building or an own road at a non-opponent vertex.
bool road_connected(const GameStateC& s, int me, int e, const int8_t* vo) {
    const PlayerC& p = s.players[me];
    for (int j = 0; j < 2; ++j) {
        const int v = EDGE_VERTICES[e][j];
        const int o = vo[v];
        if (o == me) return true;
        if (o == -1) {
            for (int q = 0; q < VERTEX_EDGES_N[v]; ++q) {
                const int f = VERTEX_EDGES[v][q];
                if (f != e && p.has_road(f)) return true;
            }
        }
    }
    return false;
}

uint32_t robber_victims_mask(const GameStateC& s, int hex_id, int me, const int8_t* vo) {
    uint32_t mask = 0;
    for (int k = 0; k < 6; ++k) {
        const int o = vo[HEX_VERTICES[hex_id][k]];
        if (o >= 0 && o != me && !((mask >> o) & 1) && has_cards(s.players[o])) mask |= bit(o);
    }
    return mask;
}

// engine._robber_actions (MOVE_ROBBER or PLAY_KNIGHT rows).
void robber_actions(const GameStateC& s, int me, ActionKind kind, const int8_t* vo, ActionList& out) {
    const int n = s.num_players;
    bool cards[MAX_PLAYERS];
    for (int i = 0; i < n; ++i) cards[i] = has_cards(s.players[i]);
    for (int h = 0; h < NUM_HEXES; ++h) {
        if (h == s.robber) continue;
        uint32_t mask = 0;
        for (int k = 0; k < 6; ++k) {
            const int o = vo[HEX_VERTICES[h][k]];
            if (o >= 0 && o != me && cards[o]) mask |= bit(o);
        }
        if (mask) {
            for (int o = 0; o < n; ++o)
                if ((mask >> o) & 1) out.push(make_action(kind, h, o));
        } else {
            out.push(make_action(kind, h, -1));
        }
    }
}

// ---------------------------------------------------------------------------
// awards
// ---------------------------------------------------------------------------
void update_longest_road(GameStateC& s) {  // engine._update_longest_road
    const int n = s.num_players;
    int lengths[MAX_PLAYERS];
    int mx = 0;
    for (int i = 0; i < n; ++i) {
        lengths[i] = s.players[i].n_roads >= 5 ? engine_longest_road_length(s, i) : 0;
        if (lengths[i] > mx) mx = lengths[i];
    }
    const int holder = s.longest_road_owner;
    if (holder >= 0) {
        if (holder >= n) throw std::out_of_range("longest_road_owner out of range");
        const int hl = lengths[holder];
        if (hl >= 5 && hl >= mx) {
            s.longest_road_len = hl;
            return;
        }
    }
    int nw = -1;
    if (mx >= 5) {
        int count = 0, first = -1;
        for (int i = 0; i < n; ++i)
            if (lengths[i] == mx) {
                if (first < 0) first = i;
                ++count;
            }
        nw = count == 1 ? first : -1;
    }
    s.longest_road_owner = nw;
    s.longest_road_len = nw >= 0 ? lengths[nw] : 0;
}

void update_longest_road_after_road(GameStateC& s, int cur) {  // engine._update_longest_road_after_road
    if (s.players[cur].n_roads < 5) return;
    const int length = engine_longest_road_length(s, cur);
    const int holder = s.longest_road_owner;
    if (holder == cur) {
        if (length >= 5) {
            s.longest_road_len = length;
            return;
        }
    } else if (holder >= 0) {
        const int hl = engine_longest_road_length(s, holder);
        if (hl >= 5) {
            if (length > hl) {
                s.longest_road_owner = cur;
                s.longest_road_len = length;
            } else {
                s.longest_road_len = hl;
            }
            return;
        }
    } else if (length < 5) {
        return;
    }
    update_longest_road(s);
}

void update_largest_army(GameStateC& s, int cur) {  // engine._update_largest_army
    const PlayerC& p = s.players[cur];
    if (p.played_knights < 3) return;
    const int la = s.largest_army_owner;
    if (la == -1 || (la != cur && p.played_knights > player_at(s, la).played_knights)) s.largest_army_owner = cur;
}

// ---------------------------------------------------------------------------
// handlers (engine._h_*).  Each validates, mutates and returns the touched players.
// ---------------------------------------------------------------------------
uint32_t h_setup_settlement(GameStateC& s, const ActionC& a) {
    if (s.phase != PHASE_SETUP_SETTLEMENT || a.nargs != 1) fail("setup settlement not allowed now");
    const int v = a.a;
    if (v < 0 || v >= NUM_VERTICES) fail("bad vertex");
    int8_t vo[NUM_VERTICES];
    vertex_owners(s, vo);
    if (vo[v] != -1) fail("vertex occupied");
    if (!distance_ok(vo, v)) fail("distance rule");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    add_settlement(p, v);
    s.setup_last_settlement = v;
    s.phase = PHASE_SETUP_ROAD;
    if (s.setup_round == 1) {
        for (int k = 0; k < VERTEX_HEXES_N[v]; ++k) {
            const int res = s.hex_res[VERTEX_HEXES[v][k]];
            if (res != DESERT && s.bank[res] > 0) {
                s.bank[res] -= 1;
                p.resources[res] += 1;
            }
        }
        return bit(cur);
    }
    return 0;
}

uint32_t h_setup_road(GameStateC& s, const ActionC& a) {
    if (s.phase != PHASE_SETUP_ROAD || a.nargs != 1) fail("setup road not allowed now");
    const int e = a.a;
    const int last = s.setup_last_settlement;
    bool touches = false;
    if (last >= 0 && last < NUM_VERTICES && e >= 0 && e < NUM_EDGES)
        for (int k = 0; k < VERTEX_EDGES_N[last]; ++k)
            if (VERTEX_EDGES[last][k] == e) touches = true;
    if (!touches) fail("setup road must touch the settlement just placed");
    for (int i = 0; i < s.num_players; ++i)
        if (s.players[i].has_road(e)) fail("edge occupied");
    add_road(player_at(s, s.current), e);
    s.setup_last_settlement = -1;
    s.turn += 1;
    const int n = s.num_players;
    if (s.setup_round == 0) {
        if (s.current == n - 1) s.setup_round = 1;
        else s.current += 1;
        s.phase = PHASE_SETUP_SETTLEMENT;
    } else if (s.current == 0) {
        s.phase = PHASE_ROLL;
    } else {
        s.current -= 1;
        s.phase = PHASE_SETUP_SETTLEMENT;
    }
    return 0;
}

uint32_t h_roll(GameStateC& s, const ActionC& a, DrawSource& draw, bool* rolled) {
    if (s.phase != PHASE_ROLL) fail("cannot roll now");
    int value;
    if (a.nargs == 0) {
        const int d1 = draw.randint(1, 6);
        const int d2 = draw.randint(1, 6);
        value = d1 + d2;
    } else if (a.nargs == 1 && a.a >= 2 && a.a <= 12) {
        value = a.a;
    } else {
        fail("bad roll action");
    }
    s.dice = value;
    if (rolled) *rolled = true;
    const int n = s.num_players;
    if (value == 7) {
        const int cur = s.current;
        s.n_discard = 0;
        for (int k = 0; k < n; ++k) {
            const int i = pmod(cur + k, n);
            if (s.players[i].total_resources() > 7) s.discard_queue[s.n_discard++] = i;
        }
        s.phase = s.n_discard ? PHASE_DISCARD : PHASE_ROBBER;
        return 0;
    }
    int32_t gains[MAX_PLAYERS][NUM_RESOURCES];
    production_for_roll(s, value, gains);
    uint32_t touched = 0;
    for (int i = 0; i < n; ++i) {
        const int32_t* g = gains[i];
        if (g[0] || g[1] || g[2] || g[3] || g[4]) {
            int32_t* r = s.players[i].resources;
            for (int k = 0; k < NUM_RESOURCES; ++k) {
                if (g[k]) {
                    r[k] += g[k];
                    s.bank[k] -= g[k];
                }
            }
            touched |= bit(i);
        }
    }
    s.phase = PHASE_MAIN;
    return touched;
}

uint32_t h_discard(GameStateC& s, const ActionC& a) {
    if (s.phase != PHASE_DISCARD || s.n_discard == 0 || a.nargs != 1) fail("no discard pending");
    if (a.nvec1 != 5) fail("discard needs 5 counts");
    const int i = s.discard_queue[0];
    PlayerC& p = player_at(s, i);
    int32_t* res = p.resources;
    int total = 0;
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        const int c = a.vec1[r];
        if (c < 0 || c > res[r]) fail("cannot discard more than held");
        total += c;
    }
    if (total != floor_div2(p.total_resources())) fail("must discard half the hand (rounded down)");
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        const int c = a.vec1[r];
        if (c) {
            res[r] -= c;
            s.bank[r] += c;
        }
    }
    for (int k = 1; k < s.n_discard; ++k) s.discard_queue[k - 1] = s.discard_queue[k];
    --s.n_discard;
    if (s.n_discard == 0) s.phase = PHASE_ROBBER;
    return bit(i);
}

// engine._move_robber (shared by MOVE_ROBBER and PLAY_KNIGHT)
uint32_t move_robber(GameStateC& s, int h, int victim, int me, DrawSource& draw) {
    if (h < 0 || h >= NUM_HEXES || h == s.robber) fail("robber must move to a different hex");
    const uint32_t victims = robber_victims(s, h, me);
    if (victim == -1) {
        if (victims) fail("must steal from an adjacent opponent");
    } else if (victim < 0 || victim >= s.num_players || !((victims >> victim) & 1)) {
        fail("victim is not an adjacent opponent with cards");
    }
    s.robber = h;
    if (victim >= 0) {
        steal(player_at(s, me), s.players[victim], draw);
        return bit(me) | bit(victim);
    }
    return 0;
}

uint32_t h_move_robber(GameStateC& s, const ActionC& a, DrawSource& draw) {
    if (s.phase != PHASE_ROBBER || a.nargs != 2) fail("cannot move the robber now");
    const uint32_t touched = move_robber(s, a.a, a.b, s.current, draw);
    s.phase = PHASE_MAIN;
    return touched;
}

// engine._require_main: common precondition for main-phase actions other than BUILD_ROAD.
void require_main(GameStateC& s) {
    if (s.phase != PHASE_MAIN) fail("action only legal in the main phase");
    if (s.free_roads > 0) {
        if (player_at(s, s.current).n_roads < MAX_ROADS && any_road_edge(s, s.current))
            fail("free roads from Road Building must be placed first");
    }
}

uint32_t h_build_road(GameStateC& s, const ActionC& a) {
    if (s.phase != PHASE_MAIN || a.nargs != 1) fail("cannot build a road now");
    const int e = a.a;
    if (e < 0 || e >= NUM_EDGES) fail("bad edge");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (p.n_roads >= MAX_ROADS) fail("no road pieces left");
    const bool free = s.free_roads > 0;
    if (!free && !afford(p.resources, COST_ROAD)) fail("cannot afford a road");
    for (int i = 0; i < s.num_players; ++i)
        if (s.players[i].has_road(e)) fail("edge occupied");
    int8_t vo[NUM_VERTICES];
    vertex_owners(s, vo);
    if (!road_connected(s, cur, e, vo)) fail("road must connect to an own road or building");
    uint32_t touched = 0;
    if (free) {
        s.free_roads -= 1;
    } else {
        pay(p, s.bank, COST_ROAD);
        touched = bit(cur);
    }
    add_road(p, e);
    if (s.free_roads > 0) {
        bool forfeit = p.n_roads >= MAX_ROADS;
        if (!forfeit) {
            int8_t eo[NUM_EDGES];
            edge_owners(s, eo);
            int tmp[NUM_EDGES];
            forfeit = road_edges(s, cur, vo, eo, tmp) == 0;
        }
        if (forfeit) s.free_roads = 0;
    }
    update_longest_road_after_road(s, cur);
    return touched;
}

uint32_t h_build_settlement(GameStateC& s, const ActionC& a) {
    if (a.nargs != 1) fail("bad action");
    require_main(s);
    const int v = a.a;
    if (v < 0 || v >= NUM_VERTICES) fail("bad vertex");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (p.n_settlements >= MAX_SETTLEMENTS) fail("no settlement pieces left");
    if (!afford(p.resources, COST_SETTLEMENT)) fail("cannot afford a settlement");
    int8_t vo[NUM_VERTICES];
    vertex_owners(s, vo);
    if (vo[v] != -1) fail("vertex occupied");
    if (!distance_ok(vo, v)) fail("distance rule");
    bool touches_road = false;
    for (int q = 0; q < VERTEX_EDGES_N[v]; ++q)
        if (p.has_road(VERTEX_EDGES[v][q])) touches_road = true;
    if (!touches_road) fail("settlement must touch an own road");
    pay(p, s.bank, COST_SETTLEMENT);
    add_settlement(p, v);
    // The new building may cut an opponent's road: recompute if any opponent with a relevant
    // road network touches this vertex.
    for (int i = 0; i < s.num_players; ++i) {
        if (i == cur || s.players[i].n_roads < 5) continue;
        for (int q = 0; q < VERTEX_EDGES_N[v]; ++q) {
            if (s.players[i].has_road(VERTEX_EDGES[v][q])) {
                update_longest_road(s);
                return bit(cur);
            }
        }
    }
    return bit(cur);
}

uint32_t h_build_city(GameStateC& s, const ActionC& a) {
    if (a.nargs != 1) fail("bad action");
    require_main(s);
    const int v = a.a;
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (v < 0 || v >= NUM_VERTICES || !p.has_settlement(v)) fail("city must replace an own settlement");
    if (p.n_cities >= MAX_CITIES) fail("no city pieces left");
    if (!afford(p.resources, COST_CITY)) fail("cannot afford a city");
    pay(p, s.bank, COST_CITY);
    remove_settlement(p, v);
    add_city(p, v);
    return bit(cur);
}

uint32_t h_buy_dev(GameStateC& s, const ActionC& a, DrawSource& draw) {
    if (a.nargs != 0) fail("bad action");
    require_main(s);
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (!afford(p.resources, COST_DEV)) fail("cannot afford a development card");
    int32_t* deck = s.dev_deck;
    const int total = deck[0] + deck[1] + deck[2] + deck[3] + deck[4];
    if (total <= 0) fail("development deck is empty");
    int idx = draw.randrange(total);
    for (int t = 0; t < NUM_DEV; ++t) {
        if (idx < deck[t]) {
            deck[t] -= 1;
            p.dev_cards_new[t] += 1;
            break;
        }
        idx -= deck[t];
    }
    pay(p, s.bank, COST_DEV);
    return bit(cur);
}

uint32_t h_play_knight(GameStateC& s, const ActionC& a, DrawSource& draw) {
    if (a.nargs != 2) fail("bad action");
    if (s.phase == PHASE_MAIN) require_main(s);
    else if (s.phase != PHASE_ROLL) fail("knight may only be played before rolling or in the main phase");
    if (s.dev_played_this_turn) fail("already played a development card this turn");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (p.dev_cards[DEV_KNIGHT] <= 0) fail("no playable knight");
    const uint32_t touched = move_robber(s, a.a, a.b, cur, draw);
    p.dev_cards[DEV_KNIGHT] -= 1;
    p.played_knights += 1;
    s.dev_played_this_turn = true;
    update_largest_army(s, cur);
    return touched ? touched : bit(cur);
}

uint32_t h_play_road_building(GameStateC& s, const ActionC& a) {
    if (a.nargs != 0) fail("bad action");
    require_main(s);
    if (s.dev_played_this_turn) fail("already played a development card this turn");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (p.dev_cards[DEV_ROAD_BUILDING] <= 0) fail("no playable road building card");
    p.dev_cards[DEV_ROAD_BUILDING] -= 1;
    s.dev_played_this_turn = true;
    int free = std::min(2, MAX_ROADS - p.n_roads);
    if (free > 0 && !any_road_edge(s, cur)) free = 0;  // forfeited: no legal road
    s.free_roads = std::max(free, 0);
    return bit(cur);
}

uint32_t h_play_year_of_plenty(GameStateC& s, const ActionC& a) {
    if (a.nargs != 2) fail("bad action");
    require_main(s);
    if (s.dev_played_this_turn) fail("already played a development card this turn");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (p.dev_cards[DEV_YEAR_OF_PLENTY] <= 0) fail("no playable year of plenty card");
    const int r1 = a.a, r2 = a.b;
    if (!(0 <= r1 && r1 <= r2 && r2 < NUM_RESOURCES)) fail("resources must satisfy 0 <= res1 <= res2 < 5");
    if (r1 == r2) {
        if (s.bank[r1] < 2) fail("bank cannot supply the requested resources");
    } else if (s.bank[r1] < 1 || s.bank[r2] < 1) {
        fail("bank cannot supply the requested resources");
    }
    s.bank[r1] -= 1;
    s.bank[r2] -= 1;
    p.resources[r1] += 1;
    p.resources[r2] += 1;
    p.dev_cards[DEV_YEAR_OF_PLENTY] -= 1;
    s.dev_played_this_turn = true;
    return bit(cur);
}

uint32_t h_play_monopoly(GameStateC& s, const ActionC& a) {
    if (a.nargs != 1) fail("bad action");
    require_main(s);
    if (s.dev_played_this_turn) fail("already played a development card this turn");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    if (p.dev_cards[DEV_MONOPOLY] <= 0) fail("no playable monopoly card");
    const int r = a.a;
    if (r < 0 || r >= NUM_RESOURCES) fail("bad resource");
    int taken = 0;
    for (int i = 0; i < s.num_players; ++i) {
        if (i == cur) continue;
        taken += s.players[i].resources[r];
        s.players[i].resources[r] = 0;
    }
    p.resources[r] += taken;
    p.dev_cards[DEV_MONOPOLY] -= 1;
    s.dev_played_this_turn = true;
    return (uint32_t)((1u << s.num_players) - 1);
}

uint32_t h_bank_trade(GameStateC& s, const ActionC& a) {
    if (a.nargs != 2) fail("bad action");
    require_main(s);
    const int give = a.a, get = a.b;
    if (!(0 <= give && give < NUM_RESOURCES && 0 <= get && get < NUM_RESOURCES) || give == get) fail("bad bank trade");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    const int ratio = port_ratio(s, cur, give);
    if (p.resources[give] < ratio) fail("not enough cards for the bank ratio");
    if (s.bank[get] < 1) fail("bank is out of that resource");
    p.resources[give] -= ratio;
    s.bank[give] += ratio;
    s.bank[get] -= 1;
    p.resources[get] += 1;
    return bit(cur);
}

int next_responder(const GameStateC& s, const TradeC& offer) {  // engine._next_responder
    const int n = s.num_players;
    for (int k = 1; k < n; ++k) {
        const int i = pmod(offer.proposer + k, n);
        if (offer.responses[i] < 0) return i;
    }
    return -1;
}

void finish_responses(GameStateC& s, const TradeC& offer) {  // engine._finish_responses
    s.trade_responder = -1;
    bool any = false;
    for (int i = 0; i < MAX_PLAYERS; ++i)
        if (offer.responses[i] == 1) any = true;
    if (any) {
        s.phase = PHASE_TRADE_SELECT;
    } else {
        s.has_pending_trade = false;
        s.phase = PHASE_MAIN;
    }
}

uint32_t h_propose_trade(GameStateC& s, const ActionC& a) {
    if (a.nargs != 2) fail("bad action");
    require_main(s);
    if (s.trades_this_turn >= MAX_TRADE_PROPOSALS_PER_TURN) fail("no trade proposals left this turn");
    if (a.nvec1 != 5 || a.nvec2 != 5) fail("trade vectors need 5 counts");
    int give_total = 0, get_total = 0;
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        const int g = a.vec1[r], t = a.vec2[r];
        if (g < 0 || t < 0) fail("negative trade counts");
        if (g && t) fail("give and get must be disjoint");
        give_total += g;
        get_total += t;
    }
    if (give_total == 0 || get_total == 0) fail("trade must be non-empty on both sides");
    const int cur = s.current;
    if (!can_pay(player_at(s, cur), a.vec1)) fail("proposer does not hold the offered cards");
    TradeC offer;
    offer.proposer = cur;
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        offer.give[r] = a.vec1[r];
        offer.get[r] = a.vec2[r];
    }
    for (int i = 0; i < MAX_PLAYERS; ++i) offer.responses[i] = -1;
    const int n = s.num_players;
    for (int k = 1; k < n; ++k) {
        const int i = pmod(cur + k, n);
        if (!can_pay(s.players[i], a.vec2)) offer.responses[i] = 0;  // auto-reject: cannot pay
    }
    s.pending_trade = offer;
    s.has_pending_trade = true;
    s.trades_this_turn += 1;
    const int nxt = next_responder(s, s.pending_trade);
    if (nxt < 0) {
        finish_responses(s, s.pending_trade);
    } else {
        s.trade_responder = nxt;
        s.phase = PHASE_TRADE_RESPONSE;
    }
    return 0;
}

uint32_t respond(GameStateC& s, const ActionC& a, bool accepted) {  // engine._respond
    if (s.phase != PHASE_TRADE_RESPONSE || a.nargs != 0) fail("no trade response pending");
    const int i = s.trade_responder;
    if (!s.has_pending_trade || i < 0) fail("no trade response pending");
    if (i >= s.num_players) throw std::out_of_range("trade_responder " + std::to_string(i) + " out of range");
    TradeC& offer = s.pending_trade;
    if (offer.responses[i] >= 0) fail("no trade response pending");
    if (accepted && !can_pay(s.players[i], offer.get)) fail("responder cannot pay");
    offer.responses[i] = accepted ? 1 : 0;
    const int nxt = next_responder(s, offer);
    if (nxt < 0) finish_responses(s, offer);
    else s.trade_responder = nxt;
    return 0;
}

uint32_t h_execute_trade(GameStateC& s, const ActionC& a) {
    if (s.phase != PHASE_TRADE_SELECT || a.nargs != 1) fail("no trade to execute");
    const int partner = a.a;
    if (!s.has_pending_trade || partner < 0 || partner >= MAX_PLAYERS || s.pending_trade.responses[partner] != 1)
        fail("partner did not accept");
    const TradeC& offer = s.pending_trade;
    PlayerC& proposer = player_at(s, offer.proposer);
    PlayerC& other = player_at(s, partner);
    if (!can_pay(proposer, offer.give) || !can_pay(other, offer.get)) fail("cards no longer available");
    int32_t* pr = proposer.resources;
    int32_t* orr = other.resources;
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        const int g = offer.give[r], t = offer.get[r];
        if (g) {
            pr[r] -= g;
            orr[r] += g;
        }
        if (t) {
            orr[r] -= t;
            pr[r] += t;
        }
    }
    const uint32_t touched = bit(offer.proposer) | bit(partner);
    s.has_pending_trade = false;
    s.trade_responder = -1;
    s.phase = PHASE_MAIN;
    return touched;
}

uint32_t h_cancel_trade(GameStateC& s, const ActionC& a) {
    if (s.phase != PHASE_TRADE_SELECT || a.nargs != 0) fail("no trade to cancel");
    s.has_pending_trade = false;
    s.trade_responder = -1;
    s.phase = PHASE_MAIN;
    return 0;
}

void end_by_cap(GameStateC& s) {  // engine._end_by_cap
    const int n = s.num_players;
    int vps[MAX_PLAYERS];
    int best = 0;
    for (int i = 0; i < n; ++i) {
        vps[i] = count_vp(s, i, true);
        if (i == 0 || vps[i] > best) best = vps[i];
    }
    int count = 0, first = -1;
    for (int i = 0; i < n; ++i)
        if (vps[i] == best) {
            if (first < 0) first = i;
            ++count;
        }
    s.winner = count == 1 ? first : -1;
    s.phase = PHASE_GAME_OVER;
}

uint32_t h_end_turn(GameStateC& s, const ActionC& a) {
    if (s.phase != PHASE_MAIN || a.nargs != 0) fail("can only end the turn in the main phase");
    const int cur = s.current;
    PlayerC& p = player_at(s, cur);
    int32_t* nw = p.dev_cards_new;
    if (nw[0] || nw[1] || nw[2] || nw[3] || nw[4]) {
        for (int t = 0; t < NUM_DEV; ++t) {
            p.dev_cards[t] += nw[t];
            nw[t] = 0;
        }
    }
    s.dev_played_this_turn = false;
    s.free_roads = 0;
    s.trades_this_turn = 0;
    s.has_pending_trade = false;
    s.trade_responder = -1;
    s.dice = 0;
    s.turn += 1;
    s.current = pmod(cur + 1, s.num_players);
    s.phase = PHASE_ROLL;
    if (s.turn >= s.max_turns) end_by_cap(s);
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// public queries
// ---------------------------------------------------------------------------
int acting_player(const GameStateC& s) {
    if (s.phase == PHASE_DISCARD) return s.n_discard > 0 ? s.discard_queue[0] : s.current;
    if (s.phase == PHASE_TRADE_RESPONSE) return s.trade_responder;
    return s.current;
}

bool is_terminal(const GameStateC& s) { return s.phase == PHASE_GAME_OVER; }

int count_vp(const GameStateC& s, int player, bool include_hidden) {
    const PlayerC& p = player_at(s, player);
    int vp = p.n_settlements + 2 * p.n_cities;
    if (s.longest_road_owner == player) vp += 2;
    if (s.largest_army_owner == player) vp += 2;
    if (include_hidden) vp += p.dev_cards[DEV_VICTORY_POINT] + p.dev_cards_new[DEV_VICTORY_POINT];
    return vp;
}

void production_for_roll(const GameStateC& s, int value, int32_t gains[MAX_PLAYERS][NUM_RESOURCES]) {
    const int n = s.num_players;
    for (int i = 0; i < MAX_PLAYERS; ++i)
        for (int r = 0; r < NUM_RESOURCES; ++r) gains[i][r] = 0;
    if (value == 7) return;
    int8_t vo[NUM_VERTICES], vw[NUM_VERTICES];
    bool have_owners = false;
    int32_t owed[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    for (int h = 0; h < NUM_HEXES; ++h) {
        const int res = s.hex_res[h];
        if (s.hex_num[h] != value || h == s.robber || res == DESERT) continue;
        if (!have_owners) {
            vertex_owners_weights(s, vo, vw);
            have_owners = true;
        }
        for (int k = 0; k < 6; ++k) {
            const int v = HEX_VERTICES[h][k];
            const int o = vo[v];
            if (o >= 0) {
                const int amt = vw[v];
                gains[o][res] += amt;
                owed[res] += amt;
            }
        }
    }
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        if (owed[r] > s.bank[r]) {
            int count = 0, only = -1;
            for (int i = 0; i < n; ++i)
                if (gains[i][r]) {
                    only = i;
                    ++count;
                }
            if (count == 1) {
                gains[only][r] = s.bank[r];
            } else {
                for (int i = 0; i < n; ++i) gains[i][r] = 0;
            }
        }
    }
}

int engine_longest_road_length(const GameStateC& s, int player) {
    player_at(s, player);
    return longest_road_length(s, player);
}

uint32_t robber_victims(const GameStateC& s, int hex_id, int me) {
    int8_t vo[NUM_VERTICES];
    vertex_owners(s, vo);
    return robber_victims_mask(s, hex_id, me, vo);
}

// ---------------------------------------------------------------------------
// discard enumeration (engine.discard_options)
// ---------------------------------------------------------------------------
namespace {
struct DiscardEnum {
    int order[NUM_RESOURCES];
    int caps[NUM_RESOURCES];
    int suffix[NUM_RESOURCES + 1];
    int cur[NUM_RESOURCES];
    int cap;
    int n = 0;
    int32_t (*out)[NUM_RESOURCES];

    void rec(int j, int rem) {
        if (j == 4) {
            cur[4] = rem;
            int32_t* counts = out[n];
            for (int jj = 0; jj < NUM_RESOURCES; ++jj) counts[order[jj]] = cur[jj];
            ++n;
            return;
        }
        const int hi = caps[j] < rem ? caps[j] : rem;
        int lo = rem - suffix[j + 1];
        if (lo < 0) lo = 0;
        for (int x = hi; x >= lo; --x) {
            cur[j] = x;
            rec(j + 1, rem - x);
            if (n >= cap) return;
        }
    }
};
}  // namespace

int discard_options(const int32_t resources[NUM_RESOURCES], int k, int cap, int32_t (*out)[NUM_RESOURCES]) {
    DiscardEnum en;
    for (int i = 0; i < NUM_RESOURCES; ++i) en.order[i] = i;
    std::sort(en.order, en.order + NUM_RESOURCES, [&](int x, int y) {
        return resources[x] != resources[y] ? resources[x] > resources[y] : x < y;
    });
    for (int j = 0; j < NUM_RESOURCES; ++j) en.caps[j] = resources[en.order[j]];
    en.suffix[NUM_RESOURCES] = 0;
    for (int j = NUM_RESOURCES - 1; j >= 0; --j) en.suffix[j] = en.suffix[j + 1] + en.caps[j];
    if (k < 0 || en.suffix[0] < k) return 0;
    if (cap <= 0) return 0;
    en.cap = cap;
    en.out = out;
    en.rec(0, k);
    return en.n;
}

// ---------------------------------------------------------------------------
// legal actions
// ---------------------------------------------------------------------------
namespace {

void legal_roll(const GameStateC& s, ActionList& out) {  // engine._legal_roll
    out.push(make_action(ACT_ROLL));
    const PlayerC& p = player_at(s, s.current);
    if (!s.dev_played_this_turn && p.dev_cards[DEV_KNIGHT] > 0) {
        int8_t vo[NUM_VERTICES];
        vertex_owners(s, vo);
        robber_actions(s, s.current, ACT_PLAY_KNIGHT, vo, out);
    }
}

void legal_main(const GameStateC& s, ActionList& out) {  // engine._legal_main
    const int cur = s.current;
    const PlayerC& p = player_at(s, cur);
    const int32_t* res = p.resources;
    const int32_t* bank = s.bank;
    const bool roads_left = p.n_roads < MAX_ROADS;
    int8_t vo[NUM_VERTICES], eo[NUM_EDGES];
    int ids[NUM_EDGES];

    if (s.free_roads > 0 && roads_left) {
        vertex_owners(s, vo);
        edge_owners(s, eo);
        const int ne = road_edges(s, cur, vo, eo, ids);
        if (ne) {
            for (int k = 0; k < ne; ++k) out.push(make_action(ACT_BUILD_ROAD, ids[k]));
            out.push(make_action(ACT_END_TURN));
            return;
        }
    }

    const bool want_road = roads_left && res[WOOD] >= 1 && res[BRICK] >= 1;
    const bool want_settlement = p.n_settlements < MAX_SETTLEMENTS && res[WOOD] >= 1 && res[BRICK] >= 1 &&
                                 res[SHEEP] >= 1 && res[WHEAT] >= 1;
    const bool knight = !s.dev_played_this_turn && p.dev_cards[DEV_KNIGHT] > 0;
    if (want_road || want_settlement || knight) vertex_owners(s, vo);
    if (want_road) {
        edge_owners(s, eo);
        const int ne = road_edges(s, cur, vo, eo, ids);
        for (int k = 0; k < ne; ++k) out.push(make_action(ACT_BUILD_ROAD, ids[k]));
    }
    if (want_settlement) {
        const int nv = settlement_vertices(s, cur, vo, ids);
        for (int k = 0; k < nv; ++k) out.push(make_action(ACT_BUILD_SETTLEMENT, ids[k]));
    }
    if (p.n_cities < MAX_CITIES && res[WHEAT] >= 2 && res[ORE] >= 3) {
        int sv[NUM_VERTICES];
        for (int k = 0; k < p.n_settlements; ++k) sv[k] = p.settlements[k];
        std::sort(sv, sv + p.n_settlements);
        for (int k = 0; k < p.n_settlements; ++k) out.push(make_action(ACT_BUILD_CITY, sv[k]));
    }
    if (res[SHEEP] >= 1 && res[WHEAT] >= 1 && res[ORE] >= 1 &&
        s.dev_deck[0] + s.dev_deck[1] + s.dev_deck[2] + s.dev_deck[3] + s.dev_deck[4] > 0)
        out.push(make_action(ACT_BUY_DEV));
    if (!s.dev_played_this_turn) {
        const int32_t* dc = p.dev_cards;
        if (knight) robber_actions(s, cur, ACT_PLAY_KNIGHT, vo, out);
        if (dc[DEV_ROAD_BUILDING] > 0) out.push(make_action(ACT_PLAY_ROAD_BUILDING));
        if (dc[DEV_YEAR_OF_PLENTY] > 0) {
            for (int r1 = 0; r1 < NUM_RESOURCES; ++r1) {
                for (int r2 = r1; r2 < NUM_RESOURCES; ++r2) {
                    if (r1 == r2) {
                        if (bank[r1] >= 2) out.push(make_action(ACT_PLAY_YEAR_OF_PLENTY, r1, r2));
                    } else if (bank[r1] >= 1 && bank[r2] >= 1) {
                        out.push(make_action(ACT_PLAY_YEAR_OF_PLENTY, r1, r2));
                    }
                }
            }
        }
        if (dc[DEV_MONOPOLY] > 0)
            for (int r = 0; r < NUM_RESOURCES; ++r) out.push(make_action(ACT_PLAY_MONOPOLY, r));
    }
    // bank / port trades
    if (res[0] || res[1] || res[2] || res[3] || res[4]) {
        int ratios[NUM_RESOURCES];
        port_ratios(s, cur, ratios);
        for (int give = 0; give < NUM_RESOURCES; ++give) {
            if (res[give] >= ratios[give]) {
                for (int get = 0; get < NUM_RESOURCES; ++get)
                    if (get != give && bank[get] >= 1) out.push(make_action(ACT_BANK_TRADE, give, get));
            }
        }
        // player trade proposals (bounded candidate set)
        if (s.trades_this_turn < MAX_TRADE_PROPOSALS_PER_TURN) {
            bool opp_has[NUM_RESOURCES] = {false, false, false, false, false};
            for (int i = 0; i < s.num_players; ++i) {
                if (i == cur) continue;
                const int32_t* qr = s.players[i].resources;
                for (int r = 0; r < NUM_RESOURCES; ++r)
                    if (qr[r]) opp_has[r] = true;
            }
            for (int give = 0; give < NUM_RESOURCES; ++give) {
                const int n = res[give];
                if (n >= 1) {
                    for (int get = 0; get < NUM_RESOURCES; ++get) {
                        if (get != give && opp_has[get]) {
                            ActionC a = make_action(ACT_PROPOSE_TRADE, 0, 0);
                            a.nvec1 = a.nvec2 = 5;
                            a.vec1[give] = 1;
                            a.vec2[get] = 1;
                            out.push(a);
                            if (n >= 2) {
                                a.vec1[give] = 2;
                                out.push(a);
                            }
                        }
                    }
                }
            }
        }
    }
    out.push(make_action(ACT_END_TURN));
}

}  // namespace

void legal_actions(const GameStateC& s, ActionList& out) {
    out.n = 0;
    const Phase phase = s.phase;
    if (phase == PHASE_MAIN) {
        legal_main(s, out);
    } else if (phase == PHASE_ROLL) {
        legal_roll(s, out);
    } else if (phase == PHASE_SETUP_SETTLEMENT) {
        int8_t vo[NUM_VERTICES];
        vertex_owners(s, vo);
        int ids[NUM_VERTICES];
        const int nv = free_vertices(vo, ids);
        for (int k = 0; k < nv; ++k) out.push(make_action(ACT_SETUP_SETTLEMENT, ids[k]));
    } else if (phase == PHASE_SETUP_ROAD) {
        const int last = s.setup_last_settlement;
        if (last < 0) return;
        if (last >= NUM_VERTICES) throw std::out_of_range("setup_last_settlement out of range");
        int8_t eo[NUM_EDGES];
        edge_owners(s, eo);
        for (int k = 0; k < VERTEX_EDGES_N[last]; ++k) {
            const int e = VERTEX_EDGES[last][k];
            if (eo[e] == -1) out.push(make_action(ACT_SETUP_ROAD, e));
        }
    } else if (phase == PHASE_DISCARD) {
        if (s.n_discard == 0) return;
        const PlayerC& p = player_at(s, s.discard_queue[0]);
        const int k = floor_div2(p.total_resources());
        static thread_local int32_t opts[DISCARD_ENUM_CAP][NUM_RESOURCES];
        const int n = discard_options(p.resources, k, DISCARD_ENUM_CAP, opts);
        for (int i = 0; i < n; ++i) {
            ActionC a = make_action(ACT_DISCARD, 0);
            a.nvec1 = 5;
            for (int r = 0; r < NUM_RESOURCES; ++r) a.vec1[r] = opts[i][r];
            out.push(a);
        }
    } else if (phase == PHASE_ROBBER) {
        int8_t vo[NUM_VERTICES];
        vertex_owners(s, vo);
        robber_actions(s, s.current, ACT_MOVE_ROBBER, vo, out);
    } else if (phase == PHASE_TRADE_RESPONSE) {
        if (!s.has_pending_trade || s.trade_responder < 0) return;
        if (can_pay(player_at(s, s.trade_responder), s.pending_trade.get)) out.push(make_action(ACT_ACCEPT_TRADE));
        out.push(make_action(ACT_REJECT_TRADE));
    } else if (phase == PHASE_TRADE_SELECT) {
        if (s.has_pending_trade) {
            const TradeC& offer = s.pending_trade;
            const PlayerC& proposer = player_at(s, offer.proposer);
            for (int i = 0; i < MAX_PLAYERS; ++i) {
                if (offer.responses[i] < 0) continue;
                if (offer.responses[i] == 1 && can_pay(player_at(s, i), offer.get) && can_pay(proposer, offer.give))
                    out.push(make_action(ACT_EXECUTE_TRADE, i));
            }
        }
        out.push(make_action(ACT_CANCEL_TRADE));
    }
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------
void apply_inplace(GameStateC& s, const ActionC& a, DrawSource& draw, bool* rolled) {
    if (rolled) *rolled = false;
    if (s.phase == PHASE_GAME_OVER) fail("game is over");
    uint32_t touched = 0;
    switch (a.kind) {
        case ACT_SETUP_SETTLEMENT: touched = h_setup_settlement(s, a); break;
        case ACT_SETUP_ROAD: touched = h_setup_road(s, a); break;
        case ACT_ROLL: touched = h_roll(s, a, draw, rolled); break;
        case ACT_DISCARD: touched = h_discard(s, a); break;
        case ACT_MOVE_ROBBER: touched = h_move_robber(s, a, draw); break;
        case ACT_BUILD_ROAD: touched = h_build_road(s, a); break;
        case ACT_BUILD_SETTLEMENT: touched = h_build_settlement(s, a); break;
        case ACT_BUILD_CITY: touched = h_build_city(s, a); break;
        case ACT_BUY_DEV: touched = h_buy_dev(s, a, draw); break;
        case ACT_PLAY_KNIGHT: touched = h_play_knight(s, a, draw); break;
        case ACT_PLAY_ROAD_BUILDING: touched = h_play_road_building(s, a); break;
        case ACT_PLAY_YEAR_OF_PLENTY: touched = h_play_year_of_plenty(s, a); break;
        case ACT_PLAY_MONOPOLY: touched = h_play_monopoly(s, a); break;
        case ACT_BANK_TRADE: touched = h_bank_trade(s, a); break;
        case ACT_PROPOSE_TRADE: touched = h_propose_trade(s, a); break;
        case ACT_ACCEPT_TRADE: touched = respond(s, a, true); break;
        case ACT_REJECT_TRADE: touched = respond(s, a, false); break;
        case ACT_EXECUTE_TRADE: touched = h_execute_trade(s, a); break;
        case ACT_CANCEL_TRADE: touched = h_cancel_trade(s, a); break;
        case ACT_END_TURN: touched = h_end_turn(s, a); break;
        default: fail("unknown action");
    }
    if (touched) {
        for (int i = 0; i < s.num_players; ++i)
            if ((touched >> i) & 1) sync(s.players[i]);
    }
    // Win check: the current player wins as soon as they hold >= 10 VP during their own turn
    // (after END_TURN this is the *new* current player).
    const Phase phase = s.phase;
    if (phase != PHASE_GAME_OVER && phase != PHASE_SETUP_SETTLEMENT && phase != PHASE_SETUP_ROAD &&
        count_vp(s, s.current, true) >= VP_TO_WIN) {
        s.winner = s.current;
        s.phase = PHASE_GAME_OVER;
    }
}

// ---------------------------------------------------------------------------
// random playout
// ---------------------------------------------------------------------------
namespace {
// Forwards to the playout rng and remembers the last randrange draw (for the trace).
struct RecordingDraw : DrawSource {
    DrawSource& inner;
    int last = -1;
    explicit RecordingDraw(DrawSource& d) : inner(d) {}
    int randrange(int n) override { return last = inner.randrange(n); }
    int randint(int lo, int hi) override { return inner.randint(lo, hi); }
};
}  // namespace

long random_playout(GameStateC& s, DrawSource& rng, long max_actions, std::vector<PlayoutStep>* trace) {
    static thread_local ActionList acts;
    long n = 0, rolls = 0;
    while (s.phase != PHASE_GAME_OVER) {
        legal_actions(s, acts);
        if (acts.n == 0) {
            const char* name = s.phase >= 0 && s.phase < NUM_PHASES ? PHASE_NAMES[s.phase] : "?";
            throw std::runtime_error("no legal actions in phase '" + std::string(name) + "' for player " +
                                     std::to_string(acting_player(s)));
        }
        ActionC a = acts.items[rng.randrange(acts.n)];
        RecordingDraw rec(rng);
        bool rolled = false;
        apply_inplace(s, a, rec, &rolled);
        if (rolled) {
            ++rolls;
            if (a.nargs == 0) {  // record the rolled value so a replay needs no draw
                a.nargs = 1;
                a.a = s.dice;
            }
        }
        if (trace) trace->push_back(PlayoutStep{a, rec.last});
        if (++n > max_actions) throw std::runtime_error("random_playout exceeded max_actions");
    }
    return rolls;
}

}  // namespace catanbot
