"""Hand-written evaluation and move ordering.

``static_value`` scores a position for one player in "VP-equivalents";
``HeuristicEvaluator`` turns those scores into a win-probability-like number
in [0, 1] (softmax over players) so it is interchangeable with the value
net.  ``action_priors`` ranks legal actions for beam search / the heuristic
bot using the explicit strategy modules (placement, trading, discard,
robber, devcards).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import actions as A
from . import board as B
from .actions import Action
from .counting import expected_hidden_vp
from .devcards import best_year_of_plenty, should_buy_dev, should_play_monopoly
from .discard import choose_discard, needed_vector
from .placement import (RESOURCE_DEMAND, best_city_spots, best_settlement_spots, buildable_settlements,
                        player_production, reachable_spots, resource_scarcity, road_targets,
                        score_city, score_settlement_spot, setup_pick, setup_road_pick)
from .robber import best_robber_move, estimated_vp, hex_damage, should_play_knight, threat
from .state import GameState, PHASE_GAME_OVER
from .trading import candidate_offers, offer_is_feeding_leader, plan_trades, should_accept

BUILD_VALUE = {"city": 3.2, "settlement": 2.8, "dev card": 1.0, "road": 0.5}


def longest_road_length(state: GameState, player: int) -> int:
    """Longest simple path over the player's roads (opponent buildings break it)."""
    p = state.players[player]
    if not p.roads:
        return 0
    occ = state.occupied_vertices()
    adj: Dict[int, List[tuple]] = {}
    for e in p.roads:
        a, b = B.EDGE_VERTICES[e]
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))
    best = 0

    def dfs(v: int, used: set, length: int) -> None:
        nonlocal best
        if length > best:
            best = length
        if v in occ and occ[v] != player:
            return  # cannot continue through an opponent building
        for w, e in adj[v]:
            if e in used:
                continue
            used.add(e)
            dfs(w, used, length + 1)
            used.discard(e)

    for v in adj:
        dfs(v, set(), 0)
    return best


def _progress_to_build(state: GameState, player: int) -> float:
    """Value of partial progress towards the most sensible next build."""
    p = state.players[player]
    hand = p.resources
    best = 0.0
    spots = buildable_settlements(state, player) if len(p.settlements) < B.MAX_SETTLEMENTS else []
    options = []
    if p.settlements and len(p.cities) < B.MAX_CITIES:
        options.append((B.COST_CITY, BUILD_VALUE["city"]))
    if spots:
        options.append((B.COST_SETTLEMENT, BUILD_VALUE["settlement"]))
    if sum(state.dev_deck) > 0:
        options.append((B.COST_DEV, BUILD_VALUE["dev card"]))
    for cost, value in options:
        total = sum(cost)
        have = sum(min(hand[r], cost[r]) for r in range(5))
        frac = have / total
        best = max(best, value * 0.45 * frac * frac)
    return best


def static_value(state: GameState, player: int) -> float:
    """Position strength for ``player`` in VP-equivalents (unbounded)."""
    if state.phase == PHASE_GAME_OVER:
        if state.winner == player:
            return 1000.0
        if state.winner >= 0:
            return -1000.0
    p = state.players[player]
    vp = state.public_vp(player) + (p.vp_cards if p.dev_known else expected_hidden_vp(state, player))
    score = 10.0 * vp
    scarcity = resource_scarcity(state)
    prod = player_production(state, player, ignore_robber=False)
    prod_free = player_production(state, player, ignore_robber=True)
    pv = sum(prod[r] * 36.0 * RESOURCE_DEMAND[r] * math.sqrt(scarcity[r]) for r in range(5))
    pv_free = sum(prod_free[r] * 36.0 * RESOURCE_DEMAND[r] * math.sqrt(scarcity[r]) for r in range(5))
    score += 0.55 * pv + 0.25 * pv_free
    score += 0.4 * sum(1 for r in range(5) if prod_free[r] > 0)
    # Expansion potential.
    if len(p.settlements) < B.MAX_SETTLEMENTS:
        occ = state.occupied_vertices()
        reach = reachable_spots(state, player, max_roads=2, occ=occ)
        now = [v for v, (d, _) in reach.items() if d == 0]
        score += 0.6 * min(len(now), 3)
        best_spot = 0.0
        own_prod = prod_free
        for v, (d, _) in reach.items():
            s = score_settlement_spot(state, player, v, occ=occ, own_prod=own_prod, scarcity=scarcity) / (1.0 + 0.9 * d)
            if s > best_spot:
                best_spot = s
        score += 0.12 * best_spot
    # Hand: cards are worth something, too many are a liability.
    n = p.total_resources if p.hand_known else p.hand_size
    score += 0.12 * min(n, 7) - 0.25 * max(0, n - 7)
    if p.hand_known:
        score += _progress_to_build(state, player)
    # Dev cards.
    if p.dev_known:
        knights_held = p.dev_cards[B.DEV_KNIGHT] + p.dev_cards_new[B.DEV_KNIGHT]
        score += 0.7 * knights_held + 0.9 * (p.dev_cards[B.DEV_ROAD_BUILDING] + p.dev_cards_new[B.DEV_ROAD_BUILDING])
        score += 0.7 * (p.dev_cards[B.DEV_YEAR_OF_PLENTY] + p.dev_cards_new[B.DEV_YEAR_OF_PLENTY])
        score += 0.9 * (p.dev_cards[B.DEV_MONOPOLY] + p.dev_cards_new[B.DEV_MONOPOLY])
    else:
        score += 0.6 * p.dev_count
    # Longest road / largest army progress (the awards themselves are in vp).
    lr = longest_road_length(state, player)
    if state.longest_road_owner != player:
        holder_len = state.longest_road_len if state.longest_road_owner >= 0 else 4
        if lr >= holder_len:
            score += 0.35 * lr
        else:
            score += 0.15 * lr
    if state.largest_army_owner != player:
        score += 0.45 * p.played_knights
    # Ports that match our production.
    for v in p.settlements + p.cities:
        t = state.ports.get(v)
        if t is None:
            continue
        if t == B.PORT_GENERIC:
            score += 0.2
        else:
            score += 0.15 + 4.0 * prod_free[t]
    return score


class HeuristicEvaluator:
    """Softmax over players' static values -> win-probability-like numbers."""

    name = "heuristic"

    def __init__(self, temperature: float = 16.0):
        self.temperature = temperature

    def evaluate(self, states: Sequence[GameState], players: Sequence[int]) -> np.ndarray:
        out = np.zeros(len(states), dtype=np.float64)
        cache: Dict[int, List[float]] = {}
        for k, (s, pl) in enumerate(zip(states, players)):
            key = id(s)
            vals = cache.get(key)
            if vals is None:
                vals = [static_value(s, i) for i in range(s.num_players)]
                cache[key] = vals
            if s.phase == PHASE_GAME_OVER and s.winner >= 0:
                out[k] = 1.0 if s.winner == pl else 0.0
                continue
            m = max(vals)
            exps = [math.exp((v - m) / self.temperature) for v in vals]
            out[k] = exps[pl] / sum(exps)
        return out

    __call__ = evaluate


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------
def action_priors(state: GameState, actions: Sequence[Action], player: Optional[int] = None,
                  belief=None) -> List[float]:
    """Prior desirability of each action (higher = try first).  Same length as ``actions``."""
    if not actions:
        return []
    if player is None:
        from .engine import acting_player  # local import: engine depends on nothing here
        player = acting_player(state)
    p = state.players[player]
    kinds = {a[0] for a in actions}
    ctx: Dict[str, object] = {}
    if A.SETUP_SETTLEMENT in kinds:
        ctx["setup"] = dict(setup_pick(state, player, k=60))
    if A.SETUP_ROAD in kinds:
        ctx["setup_road"] = setup_road_pick(state, player, state.setup_last_settlement)[0]
    if A.BUILD_SETTLEMENT in kinds:
        ctx["spots"] = dict(best_settlement_spots(state, player, k=20, candidates=buildable_settlements(state, player)))
    if A.BUILD_CITY in kinds:
        ctx["cities"] = dict(best_city_spots(state, player, k=5))
    if A.BUILD_ROAD in kinds:
        targets = road_targets(state, player, max_roads=3, k=6)
        ctx["road_first"] = {t["first_edge"]: t["score"] for t in targets if t["first_edge"] >= 0}
        ctx["lr"] = longest_road_length(state, player)
    if A.BUY_DEV in kinds:
        ctx["buy_dev"] = should_buy_dev(state, player, belief)
    if A.PLAY_KNIGHT in kinds or A.MOVE_ROBBER in kinds:
        ctx["robber"] = best_robber_move(state, player)
        if A.PLAY_KNIGHT in kinds:
            ctx["knight"] = should_play_knight(state, player)
    if A.PLAY_MONOPOLY in kinds:
        ctx["mono"] = should_play_monopoly(state, player, belief)
    if A.PLAY_YEAR_OF_PLENTY in kinds:
        ctx["yop"] = best_year_of_plenty(state, player)
    if A.BANK_TRADE in kinds or A.PROPOSE_TRADE in kinds:
        needed = needed_vector(state, player)
        ctx["plan"] = {s["action"] for s in plan_trades(state, player, needed, belief)}
        ctx["offers"] = set(candidate_offers(state, player, needed, belief))
    if A.DISCARD in kinds:
        ctx["discard"] = choose_discard(state, player, legal_actions=list(actions))
    if A.ACCEPT_TRADE in kinds and state.pending_trade is not None:
        ctx["accept"] = should_accept(state, player, state.pending_trade)[0]

    out: List[float] = []
    for a in actions:
        k = a[0]
        if k == A.ROLL:
            v = 100.0
        elif k == A.SETUP_SETTLEMENT:
            v = 50.0 + ctx["setup"].get(a[1], 0.0)
        elif k == A.SETUP_ROAD:
            v = 80.0 if a[1] == ctx["setup_road"] else 20.0
        elif k == A.BUILD_CITY:
            v = 100.0 + ctx["cities"].get(a[1], score_city(state, player, a[1]))
        elif k == A.BUILD_SETTLEMENT:
            v = 95.0 + ctx["spots"].get(a[1], 0.0)
        elif k == A.BUILD_ROAD:
            v = 40.0 + 2.0 * ctx["road_first"].get(a[1], 0.0)
            if state.free_roads > 0:
                v += 50.0
            elif ctx["lr"] >= 4 and state.longest_road_owner != player:
                v += 8.0
        elif k == A.BUY_DEV:
            v = 80.0 if ctx["buy_dev"][0] else 20.0
        elif k == A.PLAY_KNIGHT:
            h, victim, _ = ctx["robber"]
            recommended = ctx["knight"][0]
            if a[1] == h and a[2] == victim:
                # The recommended knight outranks ROLL (100) so it is played before rolling.
                v = 110.0 if recommended else 50.0
            else:
                opp, own = hex_damage(state, a[1], player)
                v = (60.0 if recommended else 30.0) + min(15.0, 0.5 * (opp - 1.6 * own))
        elif k == A.MOVE_ROBBER:
            h, victim, _ = ctx["robber"]
            opp, own = hex_damage(state, a[1], player)
            v = 50.0 + (opp - 1.6 * own)
            if a[1] == h and a[2] == victim:
                v += 40.0
            elif a[2] >= 0:
                v += 2.0 + threat(state, a[2])
        elif k == A.PLAY_ROAD_BUILDING:
            v = 70.0 if len(p.roads) <= B.MAX_ROADS - 1 else 5.0
        elif k == A.PLAY_YEAR_OF_PLENTY:
            r1, r2, _ = ctx["yop"]
            v = 68.0 if (a[1], a[2]) == (r1, r2) else 25.0
        elif k == A.PLAY_MONOPOLY:
            play, r, _ = ctx["mono"]
            v = (75.0 if play else 15.0) if a[1] == r else 10.0
        elif k == A.BANK_TRADE:
            v = 60.0 if a in ctx["plan"] else 8.0
        elif k == A.PROPOSE_TRADE:
            v = 55.0 if a in ctx["plan"] else (35.0 if a in ctx["offers"] else 1.0)
        elif k == A.DISCARD:
            v = 90.0 if a == ctx["discard"] else 20.0 - 2.0 * sum(abs(a[1][r] - ctx["discard"][1][r]) for r in range(5))
        elif k == A.ACCEPT_TRADE:
            v = 80.0 if ctx.get("accept") else 10.0
        elif k == A.REJECT_TRADE:
            v = 10.0 if ctx.get("accept") else 80.0
        elif k == A.EXECUTE_TRADE:
            offer = state.pending_trade
            feeding = False
            if offer is not None:
                feeding = (offer_is_feeding_leader(state, player, a[1], offer.give)[0]
                           or estimated_vp(state, a[1]) >= B.VP_TO_WIN - 1)
            v = 1.0 if feeding else 70.0 - 10.0 * (threat(state, a[1]) - 1.0)
        elif k == A.CANCEL_TRADE:
            v = 5.0
        elif k == A.END_TURN:
            v = 12.0
        else:
            v = 1.0
        out.append(v)
    return out
