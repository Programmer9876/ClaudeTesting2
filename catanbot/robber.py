"""Robber placement, victim choice and knight timing.

Guidelines encoded here (they are priors for the search and the source of
the explanations shown to the user):

* Move the robber where it hurts the *leader* most and never on our own
  production unless there is no alternative.
* Steal from the leader, or from whoever has the most cards.
* Play a knight (possibly before rolling) when the robber blocks a critical
  tile of ours, when it takes / secures Largest Army, or to slow down a
  player who is about to win.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from . import board as B
from .counting import expected_hidden_vp
from .placement import RESOURCE_DEMAND, player_production, resource_scarcity
from .state import GameState, PHASE_ROLL, PHASE_MAIN


def estimated_vp(state: GameState, i: int) -> float:
    """Public VP plus expected hidden VP cards."""
    return state.public_vp(i) + expected_hidden_vp(state, i)


def threat(state: GameState, i: int) -> float:
    """Weight for how dangerous player ``i`` is (1.0 for a harmless player)."""
    vp = estimated_vp(state, i)
    t = 1.0 + 0.3 * max(0.0, vp - 4.0)
    if vp >= 8:
        t *= 1.8
    elif vp >= 7:
        t *= 1.3
    return t


def hex_pips_for_player(state: GameState, h: int, i: int) -> int:
    """Pips player ``i`` collects from hex ``h`` (cities double)."""
    res, num = state.hexes[h]
    if res == B.DESERT or num == 0:
        return 0
    p = state.players[i]
    pips = B.PIPS[num]
    total = 0
    for v in B.HEX_VERTICES[h]:
        if v in p.cities:
            total += 2 * pips
        elif v in p.settlements:
            total += pips
    return total


def production_blocked(state: GameState, player: int) -> float:
    """Pips-equivalent of our production currently blocked by the robber, demand weighted."""
    h = state.robber
    res, num = state.hexes[h]
    if res == B.DESERT or num == 0:
        return 0.0
    pips = hex_pips_for_player(state, h, player)
    scarcity = resource_scarcity(state)
    return pips * RESOURCE_DEMAND[res] * (scarcity[res] ** 0.5)


def hex_damage(state: GameState, h: int, player: int) -> Tuple[float, float]:
    """(damage to opponents weighted by threat, damage to ourselves) for the robber on ``h``."""
    res, num = state.hexes[h]
    if res == B.DESERT or num == 0:
        return 0.0, 0.0
    scarcity = resource_scarcity(state)
    w = RESOURCE_DEMAND[res] * (scarcity[res] ** 0.5)
    opp = 0.0
    own = 0.0
    for i in range(state.num_players):
        pips = hex_pips_for_player(state, h, i)
        if not pips:
            continue
        if i == player:
            own += pips * w
        else:
            opp += pips * w * threat(state, i)
    return opp, own


def steal_candidates(state: GameState, h: int, player: int) -> List[int]:
    """Opponents with a building on hex ``h`` and at least one card."""
    out = []
    for i, p in enumerate(state.players):
        if i == player:
            continue
        cards = p.total_resources if p.hand_known else p.hand_size
        if cards <= 0:
            continue
        if any(v in p.settlements or v in p.cities for v in B.HEX_VERTICES[h]):
            out.append(i)
    return out


def choose_victim(state: GameState, h: int, player: int) -> int:
    """Leader first (biggest threat), then the fattest hand; -1 if nobody."""
    cands = steal_candidates(state, h, player)
    if not cands:
        return -1

    def key(i):
        p = state.players[i]
        cards = p.total_resources if p.hand_known else p.hand_size
        return (threat(state, i), min(cards, 8), cards)

    return max(cands, key=key)


def best_robber_move(state: GameState, player: int, evaluator=None,
                     exclude: Optional[int] = None) -> Tuple[int, int, str]:
    """Best ``(hex, victim, reason)`` for moving the robber.

    ``exclude`` defaults to the current robber hex (it must move).  Hexes
    where we produce are penalised heavily; hexes with a steal target get a
    bonus because a stolen card is worth roughly 1 pip-equivalent per roll.
    """
    exclude = state.robber if exclude is None else exclude
    best = (-1, -1, "")
    best_score = -1e9
    for h in range(B.NUM_HEXES):
        if h == exclude:
            continue
        opp, own = hex_damage(state, h, player)
        victim = choose_victim(state, h, player)
        score = opp - 1.6 * own
        if victim >= 0:
            vp = state.players[victim]
            cards = vp.total_resources if vp.hand_known else vp.hand_size
            score += 1.5 + 0.35 * min(cards, 6) + 0.8 * (threat(state, victim) - 1.0)
        # tiny preference for the desert over blocking nothing while hurting us
        if score > best_score:
            best_score = score
            best = (h, victim, "")
    h, victim, _ = best
    res, num = state.hexes[h]
    parts = []
    if res != B.DESERT and num:
        hurt = [(i, hex_pips_for_player(state, h, i)) for i in range(state.num_players) if i != player]
        hurt = [(i, x) for i, x in hurt if x]
        if hurt:
            names = ", ".join(f"{state.players[i].name or state.players[i].color} (-{x} pips)" for i, x in hurt)
            parts.append(f"blocks {B.RESOURCE_NAMES[res]} {num}: {names}")
    if victim >= 0:
        v = state.players[victim]
        parts.append(f"steal from {v.name or v.color} ({estimated_vp(state, victim):.0f} VP, "
                     f"{v.total_resources if v.hand_known else v.hand_size} cards)")
    _, own = hex_damage(state, h, player)
    if own > 0:
        parts.append("(also blocks some of our own production)")
    return h, victim, "; ".join(parts) if parts else "nothing better available"


def largest_army_status(state: GameState, player: int) -> Tuple[int, int, int]:
    """(our knights played, holder index or -1, holder's knights)."""
    p = state.players[player]
    holder = state.largest_army_owner
    hk = state.players[holder].played_knights if holder >= 0 else 0
    return p.played_knights, holder, hk


def should_play_knight(state: GameState, player: int) -> Tuple[bool, str]:
    """Decide whether to play a knight now (works in PHASE_ROLL and PHASE_MAIN)."""
    p = state.players[player]
    if p.dev_cards[B.DEV_KNIGHT] <= 0:
        return False, "no playable knight"
    if state.dev_played_this_turn:
        return False, "already played a development card this turn"
    if state.phase not in (PHASE_ROLL, PHASE_MAIN):
        return False, "cannot play now"
    knights, holder, hk = largest_army_status(state, player)
    my_vp = state.total_vp(player)
    # 1. Winning move / Largest Army swing.
    takes_army = (knights + 1 >= 3) and (holder != player) and (knights + 1 > hk)
    if takes_army and my_vp + 2 >= B.VP_TO_WIN:
        return True, "playing the knight takes Largest Army and wins the game"
    if takes_army:
        return True, "takes Largest Army (+2 VP" + (f", -2 for {state.players[holder].name or state.players[holder].color}" if holder >= 0 else "") + ")"
    # 2. Robber on a critical tile of ours.
    blocked = production_blocked(state, player)
    if blocked >= 3.0:
        res, num = state.hexes[state.robber]
        return True, f"robber is blocking our {B.RESOURCE_NAMES[res]} {num} ({blocked:.1f} pips-equivalent)"
    # 3. Stop a player close to winning.
    leader = max((i for i in range(state.num_players) if i != player), key=lambda i: estimated_vp(state, i), default=-1)
    if leader >= 0:
        lvp = estimated_vp(state, leader)
        h, victim, reason = best_robber_move(state, player)
        opp, own = hex_damage(state, h, player)
        if lvp >= 8 and hex_pips_for_player(state, h, leader) >= 3:
            return True, f"slow down {state.players[leader].name or state.players[leader].color} at {lvp:.0f} VP: {reason}"
        # 4. Defend Largest Army lead when someone is catching up.
        if holder == player:
            close = max((q.played_knights for i, q in enumerate(state.players) if i != player), default=0)
            if close >= knights and knights + 1 > close:
                return True, "keeps Largest Army out of reach"
        # 5. Big damage with no cost to us and the steal helps.
        if own == 0 and opp >= 6.0 and victim >= 0:
            return True, f"strong robber move available: {reason}"
    # 6. Late game: work towards Largest Army instead of sitting on knights.
    if knights + p.dev_cards[B.DEV_KNIGHT] + p.dev_cards_new[B.DEV_KNIGHT] >= 3 and holder < 0 and state.turn > 6 * state.num_players:
        return True, "building towards Largest Army"
    if blocked > 0:
        return False, f"robber costs only {blocked:.1f} pips-equivalent; keep the knight for a better moment"
    return False, "robber is not hurting us; keep the knight for Largest Army or a critical block"
