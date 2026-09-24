"""Politics: coalitions, political capital and kingmaker-lite decisions.

In 3-4 player Catan the objective is *winning*, not VP.  The further ahead
you visibly are, the worse your trades get and the more the robber visits
you; Longest Road and Largest Army are big visible markers.  Every hostile
action (robbing, blocking, monopolising, refusing reasonable trades) costs
*political capital* with the affected player, and generous actions buy it.

This module keeps a ``capital[i][j]`` matrix ("how favourably j views i",
in [-1, 1], baseline low-to-moderate, decaying back to baseline) that is
updated from public actions.  The size of each update is scaled by how far
the action was from the actor's *selfish best* alternative (a cheap proxy
for "Nash distance"): robbing the leader because it is the best move is
business; robbing a trailing player while a better hex was available is
personal.

It also provides:

* ``target_pressure`` - how much the table wants to hit a player (relative
  position, visible awards, low capital);
* ``robber_target_weights`` - how an opponent picks victims (threat x
  grudge), used by the opponent simulation;
* ``trade_willingness`` - a multiplier on P(accept) between two players;
* ``political_trade_options`` - trades that are neutral-or-better for our
  win probability but let a trailing player take Longest Road / Largest
  Army from the leader, or otherwise slow the leader down ("buy runway").
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from .actions import Action
from .counting import expected_hidden_vp
from .placement import (RESOURCE_DEMAND, blocking_value, player_production, reachable_spots,
                        resource_scarcity, score_settlement_spot)
from .robber import hex_damage, hex_pips_for_player, threat
from .state import GameState, PHASE_GAME_OVER

BASELINE = 0.1          # low-to-moderate goodwill until proven otherwise
DECAY = 0.97            # per turn, deviation from baseline shrinks by this factor


def _vp(state: GameState, i: int) -> float:
    return state.public_vp(i) + expected_hidden_vp(state, i)


def _pname(state: GameState, i: int) -> str:
    p = state.players[i]
    return p.name or p.color


def relative_position(state: GameState, i: int) -> float:
    """How far ahead player ``i`` looks to the table (0 = not ahead, 1 = runaway leader).

    Counts VP lead over the second best, visible awards and production lead.
    """
    n = state.num_players
    vps = [_vp(state, j) for j in range(n)]
    mine = vps[i]
    others = [v for j, v in enumerate(vps) if j != i]
    best_other = max(others) if others else 0.0
    lead = mine - best_other
    score = 0.0
    score += 0.18 * max(0.0, lead) + 0.06 * max(0.0, mine - 5.0)
    if state.longest_road_owner == i:
        score += 0.12
    if state.largest_army_owner == i:
        score += 0.12
    prods = [sum(player_production(state, j, ignore_robber=True)) for j in range(n)]
    p_others = [p for j, p in enumerate(prods) if j != i]
    if p_others:
        score += 0.6 * max(0.0, prods[i] - max(p_others))
    return max(0.0, min(1.0, score))


class PoliticalState:
    """Political capital between every pair of players plus a short event log."""

    def __init__(self, n: int, baseline: float = BASELINE):
        self.n = n
        self.baseline = baseline
        self.capital: List[List[float]] = [[baseline] * n for _ in range(n)]
        self.events: List[str] = []

    # --- basic ---------------------------------------------------------------
    def get(self, i: int, j: int) -> float:
        """Capital of ``i`` in the eyes of ``j``."""
        if i == j or i < 0 or j < 0 or i >= self.n or j >= self.n:
            return 0.0
        return self.capital[i][j]

    def adjust(self, i: int, j: int, delta: float, why: str = "") -> None:
        if i == j or i < 0 or j < 0:
            return
        self.capital[i][j] = max(-1.0, min(1.0, self.capital[i][j] + delta))
        if why:
            self.events.append(why)
            if len(self.events) > 30:
                self.events = self.events[-30:]

    def decay(self, factor: float = DECAY) -> None:
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.capital[i][j] = self.baseline + (self.capital[i][j] - self.baseline) * factor

    def ensure(self, n: int) -> None:
        if n != self.n:
            self.__init__(n, self.baseline)

    # --- observation -----------------------------------------------------------
    def observe(self, state: GameState, action: Action, player: int, state_after: Optional[GameState] = None) -> None:
        """Update capital from a public action taken in ``state`` by ``player``."""
        self.ensure(state.num_players)
        kind = action[0]
        n = state.num_players
        if kind in (A.MOVE_ROBBER, A.PLAY_KNIGHT):
            h, victim = action[1], action[2]
            # Selfish best alternative: the hex with the best damage score for the actor.
            best = -1e9
            chosen = -1e9
            for hx in range(B.NUM_HEXES):
                if hx == state.robber:
                    continue
                opp, own = hex_damage(state, hx, player)
                sc = opp - 1.6 * own
                best = max(best, sc)
                if hx == h:
                    chosen = sc
            personal = 1.0 if best - chosen > 1.0 else 0.35   # sacrificing value to hit someone = personal
            hurt = {}
            for j in range(n):
                if j == player:
                    continue
                pips = hex_pips_for_player(state, h, j)
                if pips:
                    hurt[j] = pips
            for j, pips in hurt.items():
                is_leader = j == max(range(n), key=lambda k: _vp(state, k))
                mag = 0.05 * min(pips, 6) / 6.0 + (0.12 if j == victim else 0.0)
                mag *= personal if not is_leader else 0.5   # hitting the leader is expected
                self.adjust(player, j, -mag, f"{_pname(state, player)} robbed {_pname(state, j)}")
            # Not hitting someone who was the obvious selfish target buys a little goodwill.
            for j in range(n):
                if j == player or j in hurt:
                    continue
                if _vp(state, j) >= _vp(state, player) + 1:
                    self.adjust(player, j, 0.01)
        elif kind == A.PLAY_MONOPOLY:
            res = action[1]
            for j in range(n):
                if j == player:
                    continue
                pj = state.players[j]
                k = pj.resources[res] if pj.hand_known else 0
                if k > 0:
                    self.adjust(player, j, -0.04 * k, f"{_pname(state, player)} monopolised {k} {B.RESOURCE_NAMES[res]} from {_pname(state, j)}")
        elif kind in (A.BUILD_SETTLEMENT, A.BUILD_ROAD, A.SETUP_SETTLEMENT):
            # Blocking: did this take a spot / cut a road another player was heading for?
            occ = state.occupied_vertices()
            if kind in (A.BUILD_SETTLEMENT, A.SETUP_SETTLEMENT):
                v = action[1]
                for j in range(n):
                    if j == player:
                        continue
                    reach = reachable_spots(state, j, max_roads=1, occ=occ)
                    if v in reach:
                        val = score_settlement_spot(state, j, v, occ=occ)
                        self.adjust(player, j, -0.02 * min(val, 20) / 10.0,
                                    f"{_pname(state, player)} took a spot {_pname(state, j)} wanted")
            else:
                e = action[1]
                for j in range(n):
                    if j == player:
                        continue
                    reach = reachable_spots(state, j, max_roads=2, occ=occ)
                    if any(first == e for _, first in reach.values()):
                        self.adjust(player, j, -0.03, f"{_pname(state, player)} cut off {_pname(state, j)}'s road")
        elif kind == A.EXECUTE_TRADE:
            partner = action[1]
            offer = state.pending_trade
            if offer is not None:
                # Goodwill grows with how good the deal is for the partner.
                gain = sum((offer.give[r] - offer.get[r]) * RESOURCE_DEMAND[r] for r in range(5))
                d = 0.04 + 0.03 * max(0.0, gain)
                self.adjust(player, partner, d, f"{_pname(state, player)} traded with {_pname(state, partner)}")
                self.adjust(partner, player, d)
        elif kind == A.REJECT_TRADE:
            offer = state.pending_trade
            if offer is not None and offer.proposer != player:
                # Refusing an offer that was fair for us is a snub.
                gain = sum((offer.give[r] - offer.get[r]) * RESOURCE_DEMAND[r] for r in range(5))
                if gain >= -0.05:
                    self.adjust(player, offer.proposer, -0.02)
        elif kind == A.ACCEPT_TRADE:
            offer = state.pending_trade
            if offer is not None and offer.proposer != player:
                self.adjust(player, offer.proposer, 0.02)
        elif kind == A.END_TURN:
            self.decay()
        # Award transfers.
        if state_after is not None:
            for owner_attr, label in (("longest_road_owner", "Longest Road"), ("largest_army_owner", "Largest Army")):
                before = getattr(state, owner_attr)
                after = getattr(state_after, owner_attr)
                if before >= 0 and after == player and before != player:
                    self.adjust(player, before, -0.08, f"{_pname(state, player)} took {label} from {_pname(state, before)}")

    # --- queries -----------------------------------------------------------------
    def grudge(self, actor: int, victim: int) -> float:
        """How much ``actor`` wants to hurt ``victim`` because of past actions (>= 0)."""
        return max(0.0, self.baseline - self.get(victim, actor))

    def target_pressure(self, state: GameState, i: int) -> float:
        """0..1 - how much the table wants to hit player ``i`` (position + grudges)."""
        self.ensure(state.num_players)
        pos = relative_position(state, i)
        grudges = [self.grudge(j, i) for j in range(state.num_players) if j != i]
        g = sum(grudges) / max(1, len(grudges))
        return max(0.0, min(1.0, 0.75 * pos + 0.8 * g))

    def robber_target_weights(self, state: GameState, actor: int) -> List[float]:
        """Per-player multiplier for how attractive it is for ``actor`` to hit them."""
        self.ensure(state.num_players)
        out = []
        for j in range(state.num_players):
            if j == actor:
                out.append(0.0)
                continue
            w = threat(state, j)
            w *= 1.0 + 1.2 * self.grudge(actor, j)          # grudges
            w *= 1.0 - 0.5 * max(0.0, self.get(j, actor) - self.baseline)  # friends get hit less
            out.append(w)
        return out

    def trade_willingness(self, state: GameState, responder: int, proposer: int) -> float:
        """Multiplier on P(responder accepts proposer's offer): 0.4 .. 1.4."""
        self.ensure(state.num_players)
        cap = self.get(proposer, responder)          # how responder views proposer
        pos = relative_position(state, proposer)
        m = 1.0 + 0.6 * (cap - self.baseline) - 0.7 * pos
        return max(0.4, min(1.4, m))

    def summary(self, state: GameState, me: int) -> List[str]:
        self.ensure(state.num_players)
        lines = []
        pos = relative_position(state, me)
        pressure = self.target_pressure(state, me)
        lines.append(f"Your visible position: {pos:.0%} ahead; target pressure on you {pressure:.0%}"
                     + (" (expect robber hits and refused trades; avoid grabbing awards early)" if pressure > 0.45 else ""))
        for j in range(state.num_players):
            if j == me:
                continue
            cap = self.get(me, j)
            back = self.get(j, me)
            mood = "friendly" if cap > self.baseline + 0.1 else ("hostile" if cap < self.baseline - 0.1 else "neutral")
            lines.append(f"{_pname(state, j)}: {mood} towards you (capital {cap:+.2f}); "
                         f"you towards them {back:+.2f}; they are under {self.target_pressure(state, j):.0%} pressure")
        leader = max(range(state.num_players), key=lambda k: _vp(state, k))
        if leader != me and _vp(state, leader) >= 7:
            lines.append(f"Coalition target: {_pname(state, leader)} ({_vp(state, leader):.0f} VP)")
        if self.events:
            lines.append("Recent: " + "; ".join(self.events[-4:]))
        return lines

    # --- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"n": self.n, "baseline": self.baseline, "capital": self.capital, "events": self.events[-30:]}

    @staticmethod
    def from_dict(d: dict) -> "PoliticalState":
        p = PoliticalState(int(d["n"]), float(d.get("baseline", BASELINE)))
        p.capital = [[float(x) for x in row] for row in d.get("capital", p.capital)]
        p.events = list(d.get("events", []))
        return p


# ---------------------------------------------------------------------------
# Political (kingmaker-lite) options
# ---------------------------------------------------------------------------
def _longest_road_len(state: GameState, i: int) -> int:
    from .heuristic import longest_road_length
    return longest_road_length(state, i)


def award_threat_opportunities(state: GameState, me: int) -> List[dict]:
    """Trailing players who could take Longest Road / Largest Army from the leader.

    Returns dicts ``{"player": k, "award": "longest_road"|"largest_army", "needs": [5 counts],
    "gap": int, "leader": L}``.
    """
    out = []
    n = state.num_players
    if n < 3:
        return out
    leader = max(range(n), key=lambda k: _vp(state, k))
    if leader == me:
        return out
    # Longest road
    if state.longest_road_owner == leader:
        target_len = state.longest_road_len + 1
        for k in range(n):
            if k in (me, leader):
                continue
            pk = state.players[k]
            have = _longest_road_len(state, k)
            gap = target_len - have
            if 1 <= gap <= 2 and len(pk.roads) + gap <= B.MAX_ROADS:
                needs = [gap, gap, 0, 0, 0]
                out.append({"player": k, "award": "longest_road", "needs": needs, "gap": gap, "leader": leader})
    # Largest army
    if state.largest_army_owner == leader:
        target_k = state.players[leader].played_knights + 1
        for k in range(n):
            if k in (me, leader):
                continue
            pk = state.players[k]
            gap = target_k - pk.played_knights
            held = pk.dev_cards[B.DEV_KNIGHT] if pk.dev_known else min(pk.dev_count, 1)
            if 1 <= gap <= 2 and (held >= gap - 1):
                # they need to buy (gap - held) dev cards: 1 ore 1 sheep 1 wheat each
                buy = max(0, gap - held)
                needs = [0, 0, buy, buy, buy]
                out.append({"player": k, "award": "largest_army", "needs": needs, "gap": gap, "leader": leader})
    return out


def political_trade_options(state: GameState, me: int, evaluator=None, politics: Optional[PoliticalState] = None,
                            max_loss: float = 0.01, model=None) -> List[dict]:
    """Trades that help a trailing player take an award from the leader.

    Each option is ``{"action": PROPOSE_TRADE, "partner": k, "reason": str,
    "delta_me": float, "delta_leader": float}``; only options whose value for
    us (per ``evaluator``) drops by at most ``max_loss`` and that lower the
    leader's value are returned, best first.  Without an evaluator the
    heuristic evaluator is used.
    """
    if evaluator is None:
        from .heuristic import HeuristicEvaluator
        evaluator = HeuristicEvaluator()
    p = state.players[me]
    opts = award_threat_opportunities(state, me)
    if not opts or state.trades_this_turn >= 4:
        return []
    out = []
    scarcity = resource_scarcity(state)
    for o in opts:
        k = o["player"]
        pk = state.players[k]
        needs = o["needs"]
        if pk.hand_known:
            missing = [max(0, needs[r] - pk.resources[r]) for r in range(5)]
        else:
            missing = [1 if needs[r] > 0 else 0 for r in range(5)]
        if sum(missing) == 0 or sum(missing) > 2:
            continue
        # We give exactly the missing cards; we ask for the cheapest card they plausibly have.
        give = tuple(min(missing[r], p.resources[r]) for r in range(5))
        if sum(give) < sum(missing):
            continue
        if pk.hand_known:
            cands = [r for r in range(5) if pk.resources[r] > needs[r]]
        else:
            cands = [r for r in range(5) if pk.hand_size > 0]
        if not cands:
            continue
        cheapest = min(cands, key=lambda r: RESOURCE_DEMAND[r] * scarcity[r])
        get = tuple(1 if r == cheapest else 0 for r in range(5))
        if get == give:
            continue
        action = (A.PROPOSE_TRADE, give, get)
        # Evaluate: state after the trade (and after k spends the cards on the award).
        s2 = state.copy()
        for r in range(5):
            s2.players[me].resources[r] += get[r] - give[r]
            s2.players[k].resources[r] += give[r] - get[r]
        s3 = s2.copy()
        # Simulate the award changing hands (what the trade enables).
        if o["award"] == "longest_road":
            s3.longest_road_owner = k
            s3.longest_road_len = state.longest_road_len + 1
            for r in range(5):
                s3.players[k].resources[r] -= needs[r]
        else:
            s3.largest_army_owner = k
            s3.players[k].played_knights += o["gap"]
            for r in range(5):
                s3.players[k].resources[r] -= needs[r]
        L = o["leader"]
        vals = evaluator.evaluate([state, s3, state, s3], [me, me, L, L])
        d_me = float(vals[1] - vals[0])
        d_leader = float(vals[3] - vals[2])
        if d_me >= -max_loss and d_leader < -0.01:
            reason = (f"political: gives {_pname(state, k)} the cards to take "
                      f"{'Longest Road' if o['award'] == 'longest_road' else 'Largest Army'} from "
                      f"{_pname(state, L)} ({_vp(state, L):.0f} VP); our win chance {d_me:+.3f}, theirs {d_leader:+.3f}")
            out.append({"action": action, "partner": k, "reason": reason, "delta_me": d_me, "delta_leader": d_leader})
    out.sort(key=lambda d: d["delta_leader"] - 0.5 * d["delta_me"])
    return out


def runway_advice(state: GameState, me: int, politics: Optional[PoliticalState] = None) -> List[str]:
    """Human readable political situation and advice."""
    lines: List[str] = []
    n = state.num_players
    leader = max(range(n), key=lambda k: _vp(state, k))
    my_pos = relative_position(state, me)
    if my_pos > 0.5:
        lines.append("You are the visible leader: expect the robber, refused trades and blocking. "
                     "Prefer hidden progress (dev cards) over grabbing Longest Road / Largest Army early, "
                     "keep the hand small, and trade only when it completes a build.")
    elif leader != me and _vp(state, leader) >= 7:
        lines.append(f"{_pname(state, leader)} is running away ({_vp(state, leader):.0f} VP): rob them, refuse their trades, "
                     "and help whoever can take an award from them.")
    opts = award_threat_opportunities(state, me)
    for o in opts:
        who = _pname(state, o["player"])
        award = "Longest Road" if o["award"] == "longest_road" else "Largest Army"
        need = ", ".join(f"{o['needs'][r]} {B.RESOURCE_NAMES[r]}" for r in range(5) if o["needs"][r])
        lines.append(f"{who} is {o['gap']} step(s) from taking {award} off the leader (needs {need}).")
    if politics is not None:
        lines.extend(politics.summary(state, me))
    return lines
