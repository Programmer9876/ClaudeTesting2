"""7-protection: discard choice and surplus dumping.

With more than 7 cards every opponent roll before our next turn has a 1/6
chance of costing us half our hand.  This module decides *what* to discard
when it happens and what to do *before* it happens (build, buy, bank-trade
surplus) so the hit is cheap.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from .actions import Action
from .placement import RESOURCE_DEMAND, buildable_settlements, player_production, resource_scarcity
from .state import GameState, PHASE_MAIN


def opponent_rolls_before_my_turn(state: GameState, player: int) -> int:
    """Number of opponent dice rolls that happen before ``player`` rolls next."""
    n = state.num_players
    if n <= 1:
        return 0
    if state.current == player:
        return n - 1
    k = (player - state.current) % n
    if state.dice:  # the current player already rolled this turn
        k -= 1
    return max(0, k)


def seven_risk(state: GameState, player: int) -> float:
    """P(at least one 7 is rolled by opponents before our next roll)."""
    k = opponent_rolls_before_my_turn(state, player)
    return 1.0 - (5.0 / 6.0) ** k


def discard_exposure(state: GameState, player: int) -> int:
    """Cards lost if a 7 is rolled right now (0 when hand <= 7)."""
    n = state.players[player].total_resources
    return n // 2 if n > 7 else 0


def default_keep_targets(state: GameState, player: int) -> List[Tuple[Sequence[int], str]]:
    """Builds we are saving for, most important first: ``[(cost, name), ...]``."""
    p = state.players[player]
    targets: List[Tuple[Sequence[int], str]] = []
    if p.settlements and len(p.cities) < B.MAX_CITIES:
        targets.append((B.COST_CITY, "city"))
    if len(p.settlements) < B.MAX_SETTLEMENTS and buildable_settlements(state, player):
        targets.insert(0, (B.COST_SETTLEMENT, "settlement"))
    if sum(state.dev_deck) > 0:
        targets.append((B.COST_DEV, "dev card"))
    if len(p.roads) < B.MAX_ROADS:
        targets.append((B.COST_ROAD, "road"))
    # Prefer the target we are closest to affording.
    def missing(cost):
        return sum(max(0, cost[r] - p.resources[r]) for r in range(5))
    targets.sort(key=lambda t: (missing(t[0]), -sum(t[0])))
    return targets


def needed_vector(state: GameState, player: int, keep_for=None) -> List[int]:
    """Cards worth keeping: the top target's cost plus what the second one needs."""
    if keep_for is None:
        keep_for = [c for c, _ in default_keep_targets(state, player)]
    needed = [0] * 5
    for k, cost in enumerate(keep_for[:2]):
        for r in range(5):
            want = cost[r] if k == 0 else -(-cost[r] // 2)   # ceil(cost / 2) for the second target
            needed[r] = max(needed[r], int(want))
    return needed


def _needed_tiers(state: GameState, player: int, keep_for=None):
    """(top-target cost, second-target half) used to score discards in two tiers."""
    if keep_for is None:
        keep_for = [c for c, _ in default_keep_targets(state, player)]
    top = list(keep_for[0]) if keep_for else [0] * 5
    second = [-(-keep_for[1][r] // 2) for r in range(5)] if len(keep_for) > 1 else [0] * 5
    return top, second


def choose_discard(state: GameState, player: int, keep_for=None,
                   legal_actions: Optional[List[Action]] = None) -> Action:
    """Return a ``(DISCARD, counts)`` action discarding floor(hand/2) cards.

    Keeps the cards needed for the planned build(s) and throws away surplus
    first, preferring cards we produce easily and that are worth least.
    """
    p = state.players[player]
    hand = list(p.resources)
    n = sum(hand) // 2
    top, second = _needed_tiers(state, player, keep_for)
    needed = [max(top[r], second[r]) for r in range(5)]
    prod = player_production(state, player, ignore_robber=True)
    scarcity = resource_scarcity(state)
    discard = [0] * 5
    for _ in range(n):
        best_r, best_score = -1, -1e9
        for r in range(5):
            if hand[r] <= 0:
                continue
            surplus = hand[r] - needed[r]
            # Two tiers: dipping below the top target's cost is much worse than below the
            # second target's half, and both are worse than discarding surplus.
            if surplus > 0:
                tier = 10.0
            elif hand[r] > top[r]:
                tier = 4.0
            else:
                tier = 0.0
            score = tier + 0.5 * surplus + 6.0 * prod[r] \
                - 1.5 * RESOURCE_DEMAND[r] * scarcity[r]
            if score > best_score:
                best_score, best_r = score, r
        if best_r < 0:
            break
        discard[best_r] += 1
        hand[best_r] -= 1
    action = (A.DISCARD, tuple(discard))
    if legal_actions is not None and action not in legal_actions:
        # Fall back to the closest legal discard (should not happen with a correct engine).
        best = None
        best_d = 1e9
        for a in legal_actions:
            if a[0] != A.DISCARD:
                continue
            d = sum(abs(a[1][r] - discard[r]) for r in range(5))
            if d < best_d:
                best, best_d = a, d
        if best is not None:
            return best
    return action


def surplus_dump_actions(state: GameState, player: int, legal_actions: List[Action]) -> List[Action]:
    """Actions that shrink a > 7 hand without hurting the plan, best first.

    Order: builds (they are always good), dev card purchase, then bank/port
    trades that convert surplus into something the plan needs.
    """
    p = state.players[player]
    if p.total_resources <= 7 or state.phase != PHASE_MAIN:
        return []
    needed = needed_vector(state, player)
    out: List[Action] = []
    kinds_order = [A.BUILD_CITY, A.BUILD_SETTLEMENT, A.BUILD_ROAD, A.BUY_DEV]
    for kind in kinds_order:
        for a in legal_actions:
            if a[0] == kind:
                out.append(a)
    # bank trades: give a resource we hold in surplus, get one we are missing
    trades = []
    for a in legal_actions:
        if a[0] != A.BANK_TRADE:
            continue
        give, get = a[1], a[2]
        ratio = state.port_ratio(player, give)
        surplus = p.resources[give] - needed[give]
        if surplus < ratio:
            continue
        gain = max(0, needed[get] - p.resources[get])
        trades.append((-(gain * 3) + ratio, a))
    trades.sort(key=lambda t: t[0])
    out.extend(a for _, a in trades)
    return out


def explain_seven_risk(state: GameState, player: int, politics=None) -> str:
    from .robber import steal_exposure
    n = state.players[player].total_resources
    if n <= 7:
        text = f"Hand of {n} cards is safe from a 7."
    else:
        risk = seven_risk(state, player)
        text = (f"Hand of {n} cards: {risk:.0%} chance an opponent rolls a 7 before your next roll "
                f"(you would lose {n // 2}). Consider building/buying or trading surplus down to 7.")
    if n >= 3:
        p_robbed, loss, detail = steal_exposure(state, player, politics)
        if p_robbed >= 0.2:
            text += f" Robber exposure: {detail}." + (
                " Spend before ending the turn or keep only cheap cards; the robber takes a random card."
                if loss >= 0.4 else "")
    return text
