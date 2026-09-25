// C++ port of catanbot/engine.py (the rules engine): legal actions, action
// application, production, awards, win detection and a random playout, all over
// the plain GameStateC struct of state.hpp.  See engine.cpp.
//
// Every function mirrors the Python function named in its comment: the same
// validation order, the same error messages (illegal_action::what() is the text
// of the Python IllegalActionError), the same action order in legal_actions and
// the same random draws (a DrawSource is asked exactly when and how the Python
// engine consults its random.Random: randint(1, 6) twice for a roll,
// randrange(total) for a steal and for a dev card draw).
//
// The header compiles without Python.h (define CATAN_NO_PYTHON before including
// state.hpp); the Python bindings live in module.cpp.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "board_tables.hpp"
#include "feature_layout.hpp"
#include "state.hpp"

namespace catanbot {

// engine.MAX_TRADE_PROPOSALS_PER_TURN / engine.DISCARD_ENUM_CAP
constexpr int MAX_TRADE_PROPOSALS_PER_TURN = 4;
constexpr int DISCARD_ENUM_CAP = 200;
// Upper bound on len(legal_actions) for a representable state: roads (72) + settlements (54)
// + cities (<= 54 in a hand-built state) + buy_dev + knights (19 hexes x 3 victims) +
// road_building + year_of_plenty (15) + monopoly (5) + bank trades (20) + proposals (40) +
// end_turn = 320; discards are capped at DISCARD_ENUM_CAP.
constexpr int MAX_LEGAL_ACTIONS = 400;

// Action kinds in the order of actions.ALL_KINDS.
enum ActionKind : int8_t {
    ACT_UNKNOWN = -1,
    ACT_SETUP_SETTLEMENT = 0,
    ACT_SETUP_ROAD,
    ACT_ROLL,
    ACT_DISCARD,
    ACT_MOVE_ROBBER,
    ACT_BUILD_ROAD,
    ACT_BUILD_SETTLEMENT,
    ACT_BUILD_CITY,
    ACT_BUY_DEV,
    ACT_PLAY_KNIGHT,
    ACT_PLAY_ROAD_BUILDING,
    ACT_PLAY_YEAR_OF_PLENTY,
    ACT_PLAY_MONOPOLY,
    ACT_BANK_TRADE,
    ACT_PROPOSE_TRADE,
    ACT_ACCEPT_TRADE,
    ACT_REJECT_TRADE,
    ACT_EXECUTE_TRADE,
    ACT_CANCEL_TRADE,
    ACT_END_TURN,
    NUM_ACTION_KINDS
};
constexpr const char* ACTION_KIND_NAMES[NUM_ACTION_KINDS] = {
    "setup_settlement", "setup_road", "roll", "discard", "move_robber", "build_road",
    "build_settlement", "build_city", "buy_dev", "play_knight", "play_road_building",
    "play_year_of_plenty", "play_monopoly", "bank_trade", "propose_trade", "accept_trade",
    "reject_trade", "execute_trade", "cancel_trade", "end_turn"};

// A Python action tuple (kind, *args).  `nargs` is the number of entries after the kind
// exactly as given (the handlers validate it like the Python ones do), `a` / `b` the first
// two scalar arguments, `vec1` / `vec2` the 5-vectors of DISCARD (vec1) and PROPOSE_TRADE
// (give, get) with their lengths as given in `nvec1` / `nvec2` (5 when valid).
struct ActionC {
    int8_t kind = ACT_UNKNOWN;
    int8_t nargs = 0;
    int8_t nvec1 = 0, nvec2 = 0;
    int32_t a = 0, b = 0;
    int32_t vec1[NUM_RESOURCES] = {0, 0, 0, 0, 0};
    int32_t vec2[NUM_RESOURCES] = {0, 0, 0, 0, 0};
};

inline ActionC make_action(ActionKind kind) {
    ActionC x;
    x.kind = kind;
    return x;
}
inline ActionC make_action(ActionKind kind, int a) {
    ActionC x;
    x.kind = kind;
    x.nargs = 1;
    x.a = a;
    return x;
}
inline ActionC make_action(ActionKind kind, int a, int b) {
    ActionC x;
    x.kind = kind;
    x.nargs = 2;
    x.a = a;
    x.b = b;
    return x;
}

struct ActionList {
    int n = 0;
    ActionC items[MAX_LEGAL_ACTIONS];
    void push(const ActionC& a) {
        if (n >= MAX_LEGAL_ACTIONS) unsupported("more than MAX_LEGAL_ACTIONS legal actions");
        items[n++] = a;
    }
};

// engine.IllegalActionError; what() is the Python message.
struct illegal_action : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Where the engine gets its random choices from (random.Random in Python).
struct DrawSource {
    virtual ~DrawSource() = default;
    virtual int randrange(int n) = 0;        // uniform in [0, n), n >= 1
    virtual int randint(int lo, int hi) = 0;  // uniform in [lo, hi]
};

// xoshiro256** seeded with splitmix64 (the C++ RNG of random_playout / apply(seed)).
struct Xoshiro : DrawSource {
    uint64_t st[4];
    explicit Xoshiro(uint64_t seed) { reseed(seed); }
    void reseed(uint64_t seed);
    uint64_t next();
    int randrange(int n) override;
    int randint(int lo, int hi) override;
};

// The forced draw of apply_forced: randrange(n) returns `index` (must be < n); the two
// randint(1, 6) calls of an unforced roll return 1 + index / 6 and 1 + index % 6 (index < 36).
struct ForcedDraw : DrawSource {
    int index;
    int calls = 0;
    explicit ForcedDraw(int idx) : index(idx) {}
    int randrange(int n) override;
    int randint(int lo, int hi) override;
};

// engine.acting_player / is_terminal / count_vp
int acting_player(const GameStateC& s);
bool is_terminal(const GameStateC& s);
int count_vp(const GameStateC& s, int player, bool include_hidden = true);

// engine.production_for_roll: per-player gains (robber and the bank-shortage rule applied).
void production_for_roll(const GameStateC& s, int value, int32_t gains[MAX_PLAYERS][NUM_RESOURCES]);

// engine.longest_road_length (identical to features.longest_road_length, which it reuses).
int engine_longest_road_length(const GameStateC& s, int player);

// engine.discard_options(resources, k, cap): writes at most `cap` count vectors, returns how many.
int discard_options(const int32_t resources[NUM_RESOURCES], int k, int cap, int32_t (*out)[NUM_RESOURCES]);

// engine.robber_victims as a bitmask over the players.
uint32_t robber_victims(const GameStateC& s, int hex_id, int me);

// engine.legal_actions: fills `out` in the Python order (empty when the game is over).
void legal_actions(const GameStateC& s, ActionList& out);

// engine.apply_inplace: validates and applies `a`; throws illegal_action (state untouched, the
// checks precede every mutation like in Python) or std::out_of_range for a player index the
// Python code would fail on with IndexError.  `*rolled` is set when the dice were rolled
// (rolls_history_len bookkeeping lives outside GameStateC).
void apply_inplace(GameStateC& s, const ActionC& a, DrawSource& draw, bool* rolled = nullptr);

// engine.random_playout on the C++ state: uniformly random legal actions (index drawn with
// rng.randrange, then the action applied with the same rng) until the game is over.  Throws
// std::runtime_error like Python when no action is legal or max_actions is exceeded.  With
// `trace`, every applied action is appended together with the random draw it consumed
// (-1 when none; rolls are recorded as (ROLL, value) so a replay needs no draw).  Returns
// the number of rolls performed.
struct PlayoutStep {
    ActionC action;
    int32_t draw;
};
long random_playout(GameStateC& s, DrawSource& rng, long max_actions, std::vector<PlayoutStep>* trace);

}  // namespace catanbot
