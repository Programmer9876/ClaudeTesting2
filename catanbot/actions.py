"""Action representation.

Actions are plain tuples ``(kind, *args)`` so they are hashable, cheap and
easy to serialise.  ``kind`` is one of the string constants below.

    (SETUP_SETTLEMENT, vertex)
    (SETUP_ROAD, edge)
    (ROLL,)                               # roll the dice (engine draws the roll from rng,
                                          #   or use ROLL with a forced value: (ROLL, value))
    (DISCARD, (w, b, s, wh, o))           # counts to discard; sum must be floor(hand/2)
    (MOVE_ROBBER, hex, victim)            # victim = player index or -1 (nobody to steal from)
    (BUILD_ROAD, edge)                    # also consumes a free road from Road Building if any
    (BUILD_SETTLEMENT, vertex)
    (BUILD_CITY, vertex)
    (BUY_DEV,)
    (PLAY_KNIGHT, hex, victim)            # move robber to hex and steal from victim (-1 = none)
    (PLAY_ROAD_BUILDING,)                 # grants free_roads = min(2, roads left)
    (PLAY_YEAR_OF_PLENTY, res1, res2)     # res1 <= res2
    (PLAY_MONOPOLY, res)
    (BANK_TRADE, give_res, get_res)       # trades ``ratio`` of give_res for 1 get_res
    (PROPOSE_TRADE, give_counts, get_counts)   # tuples of 5 ints; players respond
    (ACCEPT_TRADE,)                       # in PHASE_TRADE_RESPONSE
    (REJECT_TRADE,)                       # in PHASE_TRADE_RESPONSE
    (EXECUTE_TRADE, partner)              # in PHASE_TRADE_SELECT: proposer chooses a partner
    (CANCEL_TRADE,)                       # in PHASE_TRADE_SELECT
    (END_TURN,)
    (COUNTER_TRADE, give_counts, get_counts)   # rules variant (GameState.allow_counters): a responder answers
                                          #   the current player's offer with a modified deal; give / get
                                          #   are from the COUNTERER's side (what it hands over / wants).
                                          #   Not in ALL_KINDS (the default rules never produce it).
"""
from __future__ import annotations

from typing import Tuple

from . import board as B

SETUP_SETTLEMENT = "setup_settlement"
SETUP_ROAD = "setup_road"
ROLL = "roll"
DISCARD = "discard"
MOVE_ROBBER = "move_robber"
BUILD_ROAD = "build_road"
BUILD_SETTLEMENT = "build_settlement"
BUILD_CITY = "build_city"
BUY_DEV = "buy_dev"
PLAY_KNIGHT = "play_knight"
PLAY_ROAD_BUILDING = "play_road_building"
PLAY_YEAR_OF_PLENTY = "play_year_of_plenty"
PLAY_MONOPOLY = "play_monopoly"
BANK_TRADE = "bank_trade"
PROPOSE_TRADE = "propose_trade"
ACCEPT_TRADE = "accept_trade"
REJECT_TRADE = "reject_trade"
EXECUTE_TRADE = "execute_trade"
CANCEL_TRADE = "cancel_trade"
END_TURN = "end_turn"
# Rules variant (off by default): Colonist.io counter-offers, see engine._h_counter_trade.
COUNTER_TRADE = "counter_trade"

ALL_KINDS = [
    SETUP_SETTLEMENT, SETUP_ROAD, ROLL, DISCARD, MOVE_ROBBER, BUILD_ROAD, BUILD_SETTLEMENT,
    BUILD_CITY, BUY_DEV, PLAY_KNIGHT, PLAY_ROAD_BUILDING, PLAY_YEAR_OF_PLENTY, PLAY_MONOPOLY,
    BANK_TRADE, PROPOSE_TRADE, ACCEPT_TRADE, REJECT_TRADE, EXECUTE_TRADE, CANCEL_TRADE, END_TURN,
]
# Kinds that only exist under a non-default rules flag (kept out of ALL_KINDS: a default game never plays them).
VARIANT_KINDS = [COUNTER_TRADE]

Action = Tuple


def _counts_str(counts) -> str:
    parts = [f"{c} {B.RESOURCE_NAMES[i]}" for i, c in enumerate(counts) if c]
    return ", ".join(parts) if parts else "nothing"


def describe(action: Action, state=None) -> str:
    """Human readable one-line description of an action."""
    kind = action[0]
    names = None
    if state is not None:
        names = [p.name or p.color for p in state.players]

    def pname(i):
        if i is None or i < 0:
            return "nobody"
        return names[i] if names and i < len(names) else f"player {i}"

    def hexname(h):
        if state is not None:
            r, n = state.hexes[h]
            return f"hex {h} ({B.RESOURCE_NAMES[r]}{' ' + str(n) if n else ''})"
        return f"hex {h}"

    if kind == SETUP_SETTLEMENT:
        return f"Place starting settlement at vertex {action[1]}"
    if kind == SETUP_ROAD:
        return f"Place starting road on edge {action[1]}"
    if kind == ROLL:
        return "Roll the dice" if len(action) == 1 else f"Roll the dice (forced {action[1]})"
    if kind == DISCARD:
        return f"Discard {_counts_str(action[1])}"
    if kind == MOVE_ROBBER:
        return f"Move robber to {hexname(action[1])} and steal from {pname(action[2])}"
    if kind == BUILD_ROAD:
        return f"Build road on edge {action[1]}"
    if kind == BUILD_SETTLEMENT:
        return f"Build settlement at vertex {action[1]}"
    if kind == BUILD_CITY:
        return f"Upgrade settlement at vertex {action[1]} to a city"
    if kind == BUY_DEV:
        return "Buy a development card"
    if kind == PLAY_KNIGHT:
        return f"Play knight: move robber to {hexname(action[1])} and steal from {pname(action[2])}"
    if kind == PLAY_ROAD_BUILDING:
        return "Play Road Building (2 free roads)"
    if kind == PLAY_YEAR_OF_PLENTY:
        return f"Play Year of Plenty: take {B.RESOURCE_NAMES[action[1]]} + {B.RESOURCE_NAMES[action[2]]}"
    if kind == PLAY_MONOPOLY:
        return f"Play Monopoly on {B.RESOURCE_NAMES[action[1]]}"
    if kind == BANK_TRADE:
        ratio = state.port_ratio(state.current, action[1]) if state is not None else "?"
        return f"Trade with bank: {ratio} {B.RESOURCE_NAMES[action[1]]} -> 1 {B.RESOURCE_NAMES[action[2]]}"
    if kind == PROPOSE_TRADE:
        return f"Propose trade: give {_counts_str(action[1])} for {_counts_str(action[2])}"
    if kind == ACCEPT_TRADE:
        return "Accept the trade offer"
    if kind == REJECT_TRADE:
        return "Reject the trade offer"
    if kind == EXECUTE_TRADE:
        return f"Complete the trade with {pname(action[1])}"
    if kind == CANCEL_TRADE:
        return "Cancel the trade offer"
    if kind == COUNTER_TRADE:
        return f"Counter-offer: give {_counts_str(action[1])} for {_counts_str(action[2])}"
    if kind == END_TURN:
        return "End turn"
    return str(action)


def to_json(action: Action) -> list:
    """JSON-friendly form (tuples -> lists)."""
    return [list(a) if isinstance(a, tuple) else a for a in action]


def from_json(obj) -> Action:
    return tuple(tuple(a) if isinstance(a, list) else a for a in obj)
