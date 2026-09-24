"""Development card policy: when to buy, and how to use Monopoly / Year of Plenty.

Buying guidelines:

* At 8-9 VP the hidden VP cards can end the game unseen: gamble.
* Going for Largest Army (knights already played, award still open).
* Ore/sheep/wheat surplus that cannot become a city or settlement this turn.
* Card-count pressure (> 7 cards) - converting three cards into a dev card
  is a cheap way to dodge a 7.
* Never instead of an affordable city or a settlement on a good spot.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from . import board as B
from .counting import HandBelief, dev_draw_probabilities, monopoly_expected_take
from .placement import (RESOURCE_DEMAND, best_settlement_spots, buildable_settlements,
                        player_production, resource_scarcity)
from .state import GameState

# Rough value in VP-equivalents of each dev card type when drawn.
KNIGHT_VALUE = 0.55        # robber move + steal + progress to Largest Army
ROAD_BUILDING_VALUE = 0.7  # two roads = 2 wood + 2 brick
YEAR_OF_PLENTY_VALUE = 0.55
MONOPOLY_BASE_VALUE = 0.6


def _can_afford(p, cost) -> bool:
    return all(p.resources[r] >= cost[r] for r in range(5))


def dev_card_expected_value(state: GameState, player: int, belief: Optional[HandBelief] = None) -> float:
    """Expected VP-equivalent value of the next dev card for ``player``."""
    probs = dev_draw_probabilities(state)
    if sum(probs) <= 0:
        return 0.0
    p = state.players[player]
    knights = p.played_knights
    holder = state.largest_army_owner
    hk = state.players[holder].played_knights if holder >= 0 else 0
    # A knight is worth more when it moves us towards taking / defending the award.
    knight_v = KNIGHT_VALUE
    if holder != player and knights + 1 >= 3 and knights + 1 > hk:
        knight_v = 1.8
    elif holder != player and knights + 2 >= 3 and knights + 2 > hk:
        knight_v = 0.9
    mono_v = MONOPOLY_BASE_VALUE + 0.12 * max(monopoly_expected_take(state, player, r, belief) for r in range(5))
    return (probs[B.DEV_KNIGHT] * knight_v + probs[B.DEV_VP] * 1.0
            + probs[B.DEV_ROAD_BUILDING] * ROAD_BUILDING_VALUE
            + probs[B.DEV_YEAR_OF_PLENTY] * YEAR_OF_PLENTY_VALUE
            + probs[B.DEV_MONOPOLY] * mono_v)


def should_buy_dev(state: GameState, player: int, belief: Optional[HandBelief] = None) -> Tuple[bool, str]:
    """Rule-based recommendation for buying a development card now."""
    p = state.players[player]
    if sum(state.dev_deck) <= 0:
        return False, "development deck is empty"
    if not _can_afford(p, B.COST_DEV):
        return False, "cannot afford a development card"
    vp = state.total_vp(player)
    ev = dev_card_expected_value(state, player, belief)
    probs = dev_draw_probabilities(state)
    # Better uses of the same cards?
    if _can_afford(p, B.COST_CITY) and p.settlements and len(p.cities) < B.MAX_CITIES:
        after = [p.resources[r] - B.COST_CITY[r] for r in range(5)]
        if not all(after[r] >= B.COST_DEV[r] for r in range(5)):
            return False, "build the city first (2 VP beats a dev card)"
    spots = buildable_settlements(state, player) if len(p.settlements) < B.MAX_SETTLEMENTS else []
    if spots and _can_afford(p, B.COST_SETTLEMENT):
        after = [p.resources[r] - B.COST_SETTLEMENT[r] for r in range(5)]
        if not all(after[r] >= B.COST_DEV[r] for r in range(5)):
            return False, "build the settlement first"
    if vp >= 8 and probs[B.DEV_VP] > 0:
        return True, f"at {vp} VP a hidden victory point card ({probs[B.DEV_VP]:.0%} of the deck) can win unseen"
    knights, holder = p.played_knights, state.largest_army_owner
    hk = state.players[holder].played_knights if holder >= 0 else 0
    if holder != player and knights >= 1 and probs[B.DEV_KNIGHT] >= 0.35 and knights + 2 > hk:
        return True, f"pushing for Largest Army ({knights} knights played, {probs[B.DEV_KNIGHT]:.0%} knights left in the deck)"
    if p.total_resources > 7:
        return True, "converts 3 cards into a dev card before a 7 can hit"
    # Surplus of the dev-card resources with nothing better to spend them on.
    surplus = min(p.resources[B.SHEEP], p.resources[B.WHEAT], p.resources[B.ORE])
    if surplus >= 1 and not spots and (not p.settlements or not _can_afford(p, B.COST_CITY)):
        if ev >= 0.6:
            return True, f"no settlement spot / city affordable; expected dev value {ev:.2f} VP"
    if ev >= 0.85:
        return True, f"the remaining deck is rich (expected value {ev:.2f} VP)"
    return False, f"save the cards (expected dev value {ev:.2f} VP)"


def monopoly_value(state: GameState, player: int, res: int, belief: Optional[HandBelief] = None) -> float:
    """Expected cards gained by a Monopoly on ``res``, weighted by how useful they are."""
    take = monopoly_expected_take(state, player, res, belief)
    p = state.players[player]
    need = 0
    for cost in (B.COST_CITY, B.COST_SETTLEMENT, B.COST_DEV):
        need = max(need, cost[res] - p.resources[res])
    bonus = 1.0 + (0.5 if take >= need > 0 else 0.0)
    scarcity = resource_scarcity(state)
    return take * bonus * RESOURCE_DEMAND[res] * (scarcity[res] ** 0.3)


def best_monopoly(state: GameState, player: int, belief: Optional[HandBelief] = None) -> Tuple[int, float, str]:
    best_r, best_v = -1, -1.0
    for r in range(5):
        v = monopoly_value(state, player, r, belief)
        if v > best_v:
            best_r, best_v = r, v
    take = monopoly_expected_take(state, player, best_r, belief)
    return best_r, best_v, f"Monopoly on {B.RESOURCE_NAMES[best_r]} takes ~{take:.1f} cards"


def should_play_monopoly(state: GameState, player: int, belief: Optional[HandBelief] = None) -> Tuple[bool, int, str]:
    """Play when the expected take is big enough (or completes a build)."""
    p = state.players[player]
    if p.dev_cards[B.DEV_MONOPOLY] <= 0 or state.dev_played_this_turn:
        return False, -1, "no monopoly playable"
    r, v, reason = best_monopoly(state, player, belief)
    take = monopoly_expected_take(state, player, r, belief)
    if take >= 3.0:
        return True, r, reason
    for cost in (B.COST_CITY, B.COST_SETTLEMENT):
        missing = [max(0, cost[x] - p.resources[x]) for x in range(5)]
        if sum(missing) > 0 and missing[r] == sum(missing) and take >= missing[r]:
            return True, r, reason + " and completes a build"
    return False, r, f"wait: only ~{take:.1f} cards to take"


def best_year_of_plenty(state: GameState, player: int) -> Tuple[int, int, str]:
    """Pair of resources (r1 <= r2) that best completes a build; else the scarcest needs."""
    p = state.players[player]
    best = None
    best_score = -1e9
    spots = buildable_settlements(state, player)
    targets = []
    if p.settlements and len(p.cities) < B.MAX_CITIES:
        targets.append((B.COST_CITY, 2.0, "city"))
    if spots and len(p.settlements) < B.MAX_SETTLEMENTS:
        targets.append((B.COST_SETTLEMENT, 1.6, "settlement"))
    if sum(state.dev_deck) > 0:
        targets.append((B.COST_DEV, 0.7, "dev card"))
    if len(p.roads) < B.MAX_ROADS:
        targets.append((B.COST_ROAD, 0.4, "road"))
    prod = player_production(state, player, ignore_robber=True)
    scarcity = resource_scarcity(state)
    for r1 in range(5):
        if state.bank[r1] <= 0:
            continue
        for r2 in range(r1, 5):
            if state.bank[r2] <= (1 if r1 == r2 else 0):
                continue
            hand = list(p.resources)
            hand[r1] += 1
            hand[r2] += 1
            score = 0.0
            why = ""
            for cost, value, name in targets:
                missing_before = sum(max(0, cost[x] - p.resources[x]) for x in range(5))
                missing_after = sum(max(0, cost[x] - hand[x]) for x in range(5))
                if missing_before > 0 and missing_after == 0:
                    if value > score:
                        score = value
                        why = f"completes a {name}"
                elif missing_after < missing_before:
                    gain = 0.25 * value * (missing_before - missing_after)
                    if gain > score:
                        score = gain
                        why = f"gets closer to a {name}"
            # tie-break: scarce, high-demand resources we don't produce
            for r in (r1, r2):
                score += 0.05 * RESOURCE_DEMAND[r] * scarcity[r] / (1.0 + 6.0 * prod[r])
            if score > best_score:
                best_score = score
                best = (r1, r2, why or "takes the scarcest resources")
    if best is None:
        return 0, 0, "bank is empty"
    return best


def dev_card_advice(state: GameState, player: int, belief: Optional[HandBelief] = None) -> List[str]:
    """Human readable dev-card situation summary."""
    p = state.players[player]
    lines = []
    buy, why = should_buy_dev(state, player, belief)
    lines.append(("Buy a dev card: " if buy else "Don't buy a dev card now: ") + why)
    probs = dev_draw_probabilities(state)
    lines.append("Deck odds: " + ", ".join(f"{B.DEV_NAMES[t]} {probs[t]:.0%}" for t in range(5) if probs[t] > 0)
                 + f" ({sum(state.dev_deck)} cards left)")
    if p.dev_cards[B.DEV_MONOPOLY]:
        play, r, reason = should_play_monopoly(state, player, belief)
        lines.append(("Play Monopoly: " if play else "Hold Monopoly: ") + reason)
    if p.dev_cards[B.DEV_YEAR_OF_PLENTY]:
        r1, r2, why = best_year_of_plenty(state, player)
        lines.append(f"Year of Plenty: take {B.RESOURCE_NAMES[r1]} + {B.RESOURCE_NAMES[r2]} ({why})")
    return lines
