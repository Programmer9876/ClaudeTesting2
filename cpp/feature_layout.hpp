// GENERATED FILE - do not edit.
// Produced by scripts/gen_board_tables.py from catanbot/features.py (and state.py phases).
// Re-run `PYTHONPATH=. python3 scripts/gen_board_tables.py` after changing the feature layout.
#pragma once

namespace catan {

// ---- phases, in the order of the phase one-hot block ----------------------------
constexpr int NUM_PHASES = 9;
enum Phase : int {
    PHASE_SETUP_SETTLEMENT = 0,
    PHASE_SETUP_ROAD = 1,
    PHASE_ROLL = 2,
    PHASE_DISCARD = 3,
    PHASE_ROBBER = 4,
    PHASE_MAIN = 5,
    PHASE_TRADE_RESPONSE = 6,
    PHASE_TRADE_SELECT = 7,
    PHASE_GAME_OVER = 8,
    PHASE_UNKNOWN = -1
};
constexpr const char* PHASE_NAMES[NUM_PHASES] = {"setup_settlement", "setup_road", "roll", "discard", "robber", "main", "trade_response", "trade_select", "game_over"};

// ---- block sizes -----------------------------------------------------------------
constexpr int MAX_PLAYERS = 4;
constexpr int PLAYER_BLOCK = 63;
constexpr int GLOBAL_BLOCK = 50;
constexpr int GLOBAL_OFFSET = 252;
constexpr int NUM_FEATURES = 302;
static_assert(NUM_FEATURES == MAX_PLAYERS * PLAYER_BLOCK + GLOBAL_BLOCK, "layout");

// ---- offsets inside one player block (features._P) -------------------------------
namespace P {
constexpr int public_vp = 0;
constexpr int hidden_vp = 1;
constexpr int total_vp = 2;
constexpr int settlements = 3;
constexpr int cities = 4;
constexpr int roads = 5;
constexpr int res_wood = 6;
constexpr int res_brick = 7;
constexpr int res_sheep = 8;
constexpr int res_wheat = 9;
constexpr int res_ore = 10;
constexpr int hand_known = 11;
constexpr int hand_size = 12;
constexpr int cards_over_7 = 13;
constexpr int discard_exposure = 14;
constexpr int prod_wood = 15;
constexpr int prod_brick = 16;
constexpr int prod_sheep = 17;
constexpr int prod_wheat = 18;
constexpr int prod_ore = 19;
constexpr int prod_nr_wood = 20;
constexpr int prod_nr_brick = 21;
constexpr int prod_nr_sheep = 22;
constexpr int prod_nr_wheat = 23;
constexpr int prod_nr_ore = 24;
constexpr int prod_total = 25;
constexpr int prod_nr_total = 26;
constexpr int robber_loss = 27;
constexpr int prod_types = 28;
constexpr int prod_entropy = 29;
constexpr int robber_on_my_hex = 30;
constexpr int spots_now = 31;
constexpr int spots_1road = 32;
constexpr int spots_2roads = 33;
constexpr int best_spot_pips_now = 34;
constexpr int best_spot_pips_2roads = 35;
constexpr int can_build_road = 36;
constexpr int can_build_settlement = 37;
constexpr int can_build_city = 38;
constexpr int can_buy_dev = 39;
constexpr int longest_road = 40;
constexpr int has_longest_road = 41;
constexpr int knights = 42;
constexpr int has_largest_army = 43;
constexpr int dev_knight = 44;
constexpr int dev_victory_point = 45;
constexpr int dev_road_building = 46;
constexpr int dev_year_of_plenty = 47;
constexpr int dev_monopoly = 48;
constexpr int devnew_knight = 49;
constexpr int devnew_victory_point = 50;
constexpr int devnew_road_building = 51;
constexpr int devnew_year_of_plenty = 52;
constexpr int devnew_monopoly = 53;
constexpr int dev_total = 54;
constexpr int dev_known = 55;
constexpr int port_wood = 56;
constexpr int port_brick = 57;
constexpr int port_sheep = 58;
constexpr int port_wheat = 59;
constexpr int port_ore = 60;
constexpr int is_current = 61;
constexpr int present = 62;
}  // namespace P

// ---- offsets inside the global block (features._G) -------------------------------
namespace G {
constexpr int bank_wood = 0;
constexpr int bank_brick = 1;
constexpr int bank_sheep = 2;
constexpr int bank_wheat = 3;
constexpr int bank_ore = 4;
constexpr int deck_knight = 5;
constexpr int deck_victory_point = 6;
constexpr int deck_road_building = 7;
constexpr int deck_year_of_plenty = 8;
constexpr int deck_monopoly = 9;
constexpr int deck_total = 10;
constexpr int turn = 11;
constexpr int stage = 12;
constexpr int players_2 = 13;
constexpr int players_3 = 14;
constexpr int players_4 = 15;
constexpr int phase_setup_settlement = 16;
constexpr int phase_setup_road = 17;
constexpr int phase_roll = 18;
constexpr int phase_discard = 19;
constexpr int phase_robber = 20;
constexpr int phase_main = 21;
constexpr int phase_trade_response = 22;
constexpr int phase_trade_select = 23;
constexpr int phase_game_over = 24;
constexpr int my_turn = 25;
constexpr int i_am_acting = 26;
constexpr int dice = 27;
constexpr int free_roads = 28;
constexpr int dev_played_this_turn = 29;
constexpr int trades_this_turn = 30;
constexpr int vp_gap = 31;
constexpr int vp_gap_expected = 32;
constexpr int leader_vp = 33;
constexpr int i_am_leader = 34;
constexpr int lr_gap = 35;
constexpr int knight_gap = 36;
constexpr int trade_pending = 37;
constexpr int trade_i_propose = 38;
constexpr int trade_i_respond = 39;
constexpr int trade_give_wood = 40;
constexpr int trade_give_brick = 41;
constexpr int trade_give_sheep = 42;
constexpr int trade_give_wheat = 43;
constexpr int trade_give_ore = 44;
constexpr int trade_get_wood = 45;
constexpr int trade_get_brick = 46;
constexpr int trade_get_sheep = 47;
constexpr int trade_get_wheat = 48;
constexpr int trade_get_ore = 49;
}  // namespace G

// ---- full feature names (index = position in the vector) -------------------------
constexpr const char* FEATURE_NAMES[NUM_FEATURES] = {
    "me_public_vp",
    "me_hidden_vp",
    "me_total_vp",
    "me_settlements",
    "me_cities",
    "me_roads",
    "me_res_wood",
    "me_res_brick",
    "me_res_sheep",
    "me_res_wheat",
    "me_res_ore",
    "me_hand_known",
    "me_hand_size",
    "me_cards_over_7",
    "me_discard_exposure",
    "me_prod_wood",
    "me_prod_brick",
    "me_prod_sheep",
    "me_prod_wheat",
    "me_prod_ore",
    "me_prod_nr_wood",
    "me_prod_nr_brick",
    "me_prod_nr_sheep",
    "me_prod_nr_wheat",
    "me_prod_nr_ore",
    "me_prod_total",
    "me_prod_nr_total",
    "me_robber_loss",
    "me_prod_types",
    "me_prod_entropy",
    "me_robber_on_my_hex",
    "me_spots_now",
    "me_spots_1road",
    "me_spots_2roads",
    "me_best_spot_pips_now",
    "me_best_spot_pips_2roads",
    "me_can_build_road",
    "me_can_build_settlement",
    "me_can_build_city",
    "me_can_buy_dev",
    "me_longest_road",
    "me_has_longest_road",
    "me_knights",
    "me_has_largest_army",
    "me_dev_knight",
    "me_dev_victory_point",
    "me_dev_road_building",
    "me_dev_year_of_plenty",
    "me_dev_monopoly",
    "me_devnew_knight",
    "me_devnew_victory_point",
    "me_devnew_road_building",
    "me_devnew_year_of_plenty",
    "me_devnew_monopoly",
    "me_dev_total",
    "me_dev_known",
    "me_port_wood",
    "me_port_brick",
    "me_port_sheep",
    "me_port_wheat",
    "me_port_ore",
    "me_is_current",
    "me_present",
    "opp1_public_vp",
    "opp1_hidden_vp",
    "opp1_total_vp",
    "opp1_settlements",
    "opp1_cities",
    "opp1_roads",
    "opp1_res_wood",
    "opp1_res_brick",
    "opp1_res_sheep",
    "opp1_res_wheat",
    "opp1_res_ore",
    "opp1_hand_known",
    "opp1_hand_size",
    "opp1_cards_over_7",
    "opp1_discard_exposure",
    "opp1_prod_wood",
    "opp1_prod_brick",
    "opp1_prod_sheep",
    "opp1_prod_wheat",
    "opp1_prod_ore",
    "opp1_prod_nr_wood",
    "opp1_prod_nr_brick",
    "opp1_prod_nr_sheep",
    "opp1_prod_nr_wheat",
    "opp1_prod_nr_ore",
    "opp1_prod_total",
    "opp1_prod_nr_total",
    "opp1_robber_loss",
    "opp1_prod_types",
    "opp1_prod_entropy",
    "opp1_robber_on_my_hex",
    "opp1_spots_now",
    "opp1_spots_1road",
    "opp1_spots_2roads",
    "opp1_best_spot_pips_now",
    "opp1_best_spot_pips_2roads",
    "opp1_can_build_road",
    "opp1_can_build_settlement",
    "opp1_can_build_city",
    "opp1_can_buy_dev",
    "opp1_longest_road",
    "opp1_has_longest_road",
    "opp1_knights",
    "opp1_has_largest_army",
    "opp1_dev_knight",
    "opp1_dev_victory_point",
    "opp1_dev_road_building",
    "opp1_dev_year_of_plenty",
    "opp1_dev_monopoly",
    "opp1_devnew_knight",
    "opp1_devnew_victory_point",
    "opp1_devnew_road_building",
    "opp1_devnew_year_of_plenty",
    "opp1_devnew_monopoly",
    "opp1_dev_total",
    "opp1_dev_known",
    "opp1_port_wood",
    "opp1_port_brick",
    "opp1_port_sheep",
    "opp1_port_wheat",
    "opp1_port_ore",
    "opp1_is_current",
    "opp1_present",
    "opp2_public_vp",
    "opp2_hidden_vp",
    "opp2_total_vp",
    "opp2_settlements",
    "opp2_cities",
    "opp2_roads",
    "opp2_res_wood",
    "opp2_res_brick",
    "opp2_res_sheep",
    "opp2_res_wheat",
    "opp2_res_ore",
    "opp2_hand_known",
    "opp2_hand_size",
    "opp2_cards_over_7",
    "opp2_discard_exposure",
    "opp2_prod_wood",
    "opp2_prod_brick",
    "opp2_prod_sheep",
    "opp2_prod_wheat",
    "opp2_prod_ore",
    "opp2_prod_nr_wood",
    "opp2_prod_nr_brick",
    "opp2_prod_nr_sheep",
    "opp2_prod_nr_wheat",
    "opp2_prod_nr_ore",
    "opp2_prod_total",
    "opp2_prod_nr_total",
    "opp2_robber_loss",
    "opp2_prod_types",
    "opp2_prod_entropy",
    "opp2_robber_on_my_hex",
    "opp2_spots_now",
    "opp2_spots_1road",
    "opp2_spots_2roads",
    "opp2_best_spot_pips_now",
    "opp2_best_spot_pips_2roads",
    "opp2_can_build_road",
    "opp2_can_build_settlement",
    "opp2_can_build_city",
    "opp2_can_buy_dev",
    "opp2_longest_road",
    "opp2_has_longest_road",
    "opp2_knights",
    "opp2_has_largest_army",
    "opp2_dev_knight",
    "opp2_dev_victory_point",
    "opp2_dev_road_building",
    "opp2_dev_year_of_plenty",
    "opp2_dev_monopoly",
    "opp2_devnew_knight",
    "opp2_devnew_victory_point",
    "opp2_devnew_road_building",
    "opp2_devnew_year_of_plenty",
    "opp2_devnew_monopoly",
    "opp2_dev_total",
    "opp2_dev_known",
    "opp2_port_wood",
    "opp2_port_brick",
    "opp2_port_sheep",
    "opp2_port_wheat",
    "opp2_port_ore",
    "opp2_is_current",
    "opp2_present",
    "opp3_public_vp",
    "opp3_hidden_vp",
    "opp3_total_vp",
    "opp3_settlements",
    "opp3_cities",
    "opp3_roads",
    "opp3_res_wood",
    "opp3_res_brick",
    "opp3_res_sheep",
    "opp3_res_wheat",
    "opp3_res_ore",
    "opp3_hand_known",
    "opp3_hand_size",
    "opp3_cards_over_7",
    "opp3_discard_exposure",
    "opp3_prod_wood",
    "opp3_prod_brick",
    "opp3_prod_sheep",
    "opp3_prod_wheat",
    "opp3_prod_ore",
    "opp3_prod_nr_wood",
    "opp3_prod_nr_brick",
    "opp3_prod_nr_sheep",
    "opp3_prod_nr_wheat",
    "opp3_prod_nr_ore",
    "opp3_prod_total",
    "opp3_prod_nr_total",
    "opp3_robber_loss",
    "opp3_prod_types",
    "opp3_prod_entropy",
    "opp3_robber_on_my_hex",
    "opp3_spots_now",
    "opp3_spots_1road",
    "opp3_spots_2roads",
    "opp3_best_spot_pips_now",
    "opp3_best_spot_pips_2roads",
    "opp3_can_build_road",
    "opp3_can_build_settlement",
    "opp3_can_build_city",
    "opp3_can_buy_dev",
    "opp3_longest_road",
    "opp3_has_longest_road",
    "opp3_knights",
    "opp3_has_largest_army",
    "opp3_dev_knight",
    "opp3_dev_victory_point",
    "opp3_dev_road_building",
    "opp3_dev_year_of_plenty",
    "opp3_dev_monopoly",
    "opp3_devnew_knight",
    "opp3_devnew_victory_point",
    "opp3_devnew_road_building",
    "opp3_devnew_year_of_plenty",
    "opp3_devnew_monopoly",
    "opp3_dev_total",
    "opp3_dev_known",
    "opp3_port_wood",
    "opp3_port_brick",
    "opp3_port_sheep",
    "opp3_port_wheat",
    "opp3_port_ore",
    "opp3_is_current",
    "opp3_present",
    "g_bank_wood",
    "g_bank_brick",
    "g_bank_sheep",
    "g_bank_wheat",
    "g_bank_ore",
    "g_deck_knight",
    "g_deck_victory_point",
    "g_deck_road_building",
    "g_deck_year_of_plenty",
    "g_deck_monopoly",
    "g_deck_total",
    "g_turn",
    "g_stage",
    "g_players_2",
    "g_players_3",
    "g_players_4",
    "g_phase_setup_settlement",
    "g_phase_setup_road",
    "g_phase_roll",
    "g_phase_discard",
    "g_phase_robber",
    "g_phase_main",
    "g_phase_trade_response",
    "g_phase_trade_select",
    "g_phase_game_over",
    "g_my_turn",
    "g_i_am_acting",
    "g_dice",
    "g_free_roads",
    "g_dev_played_this_turn",
    "g_trades_this_turn",
    "g_vp_gap",
    "g_vp_gap_expected",
    "g_leader_vp",
    "g_i_am_leader",
    "g_lr_gap",
    "g_knight_gap",
    "g_trade_pending",
    "g_trade_i_propose",
    "g_trade_i_respond",
    "g_trade_give_wood",
    "g_trade_give_brick",
    "g_trade_give_sheep",
    "g_trade_give_wheat",
    "g_trade_give_ore",
    "g_trade_get_wood",
    "g_trade_get_brick",
    "g_trade_get_sheep",
    "g_trade_get_wheat",
    "g_trade_get_ore",
};

}  // namespace catan
