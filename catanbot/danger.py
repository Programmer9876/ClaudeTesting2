"""Distance-to-win ("danger") model for every player.

Victory points alone say who is *ahead*; they do not say who is *close to
winning*.  A leader on 9 VP with no settlement spot left, every settlement
already a city and an empty hand is far from a win; a second-placed player
on 7 VP holding the cards for a city and a settlement is one or two rolls
from it.  This module estimates, for each player, the cheapest **win path**
(city upgrades, settlements on reachable spots, Longest Road, Largest Army,
hidden VP cards), the cards still **missing** for it, their production and
port substitutes per resource, the **rolls** that feed the path and the
number of **turns** until the path is affordable.  ``danger`` maps the turns
to 0..1 (1 = can win on their turn).

Consumers (``robber.py``, ``politics.py``):

* target weight = VP threat x (0.4 + 1.6 x danger): the leader stays the
  default priority but a loaded runner-up overtakes an overextended leader;
* need-aware blocking: a hex is worth blocking for the share of a player's
  supply of a *needed* resource it removes.  Blocking a resource the player
  already holds, or produces elsewhere, or can buy through a port from a
  surplus, is discounted;
* resource-aware stealing: a hand rich in what the victim needs (or in what
  we need) is a better steal than one full of cards nobody wants.

Hidden hands are taken as the production-weighted expectation
(``counting.hand_prior_weights``) or from a ``HandBelief`` when given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import board as B
from .counting import dev_pool, expected_hidden_vp, hand_prior_weights
from .engine import longest_road_length
from .placement import player_production, reachable_spots
from .state import GameState, PHASE_GAME_OVER

TURNS_HALF = 3.0        # danger = 0.5 when the win path is this many turns away
MAX_TURNS = 30.0
BLOCK_FLOOR = 0.35      # blocking a resource the target does not need still hurts a little
BLOCK_NEED = 2.0        # ... and a fully needed one hurts this much more
_CACHE_SIZE = 64


@dataclass
class WinPath:
    player: int
    vp: float                                  # estimated VP now (public + expected hidden)
    need_vp: int                               # VP still needed
    steps: List[str]                           # human readable path
    cost: List[float]                          # total resource cost of the path
    hand: List[float]                          # known or expected hand
    hand_known: bool
    missing: List[float]                       # max(0, cost - hand)
    need_share: List[float]                    # missing / sum(missing)
    eff_need: List[float]                      # need including port / bank routes from a surplus
    prod: List[float]                          # cards per roll per resource, robber ignored
    supply: List[float]                        # cards per roll incl. trades from surplus
    turns: float                               # estimated turns until the path is affordable
    min_turns: float                           # rule constraints (one knight per turn)
    rolls: Dict[int, List[int]] = field(default_factory=dict)   # needed resource -> numbers producing it
    can_win_now: bool = False                  # hand already covers the path
    danger: float = 0.0                        # 0..1
    blocked_now: float = 0.0                   # cards/roll the robber currently costs them

    def composition(self) -> List[float]:
        """Expected share of each resource in the hand (sums to 1; uniform when empty)."""
        tot = sum(self.hand)
        if tot <= 1e-9:
            return [0.2] * 5
        return [h / tot for h in self.hand]

    def summary(self, state: GameState) -> str:
        p = state.players[self.player]
        name = p.name or p.color
        if self.need_vp <= 0:
            return f"{name}: has won"
        if self.can_win_now:
            what = ", ".join(self.steps) or "nothing"
            return f"{name}: {self.vp:.0f} VP, CAN WIN ON THEIR TURN ({what})"
        needs = [(r, self.missing[r]) for r in range(5) if self.missing[r] > 0.05]
        need_txt = ", ".join(f"{m:.0f} {B.RESOURCE_NAMES[r]}" + (
            " (rolls " + "/".join(str(n) for n in self.rolls.get(r, [])) + ")" if self.rolls.get(r) else " (no production)")
            for r, m in needs) or "nothing"
        path = " + ".join(self.steps) if self.steps else "no path"
        tag = "loaded" if self.danger >= 0.5 else ("overextended" if self.danger < 0.2 else "building")
        return (f"{name}: {self.vp:.0f} VP, {tag}, ~{self.turns:.1f} turns to win via {path}; "
                f"missing {need_txt}; hand {sum(self.hand):.0f} cards")


def _signature(state: GameState) -> tuple:
    return (state.turn, state.current, state.phase, state.robber, state.dice, tuple(state.bank), tuple(state.dev_deck),
            state.longest_road_owner, state.longest_road_len, state.largest_army_owner,
            tuple((tuple(p.resources), p.hand_known, p.hand_size, p.dev_known, p.dev_count, tuple(p.dev_cards),
                   tuple(p.dev_cards_new), p.played_knights, tuple(p.settlements), tuple(p.cities), tuple(p.roads))
                  for p in state.players))


_cache: Dict[tuple, Dict[int, WinPath]] = {}


def win_paths(state: GameState, belief=None) -> Dict[int, WinPath]:
    """Win path of every player (cached per identical state when no belief is given)."""
    key = _signature(state) if belief is None else None
    if key is not None:
        hit = _cache.get(key)
        if hit is not None:
            return hit
    out = {i: win_path(state, i, belief) for i in range(state.num_players)}
    if key is not None:
        if len(_cache) >= _CACHE_SIZE:
            _cache.clear()
        _cache[key] = out
    return out


def _expected_hand(state: GameState, i: int, belief=None) -> Tuple[List[float], bool]:
    p = state.players[i]
    if p.hand_known:
        return [float(x) for x in p.resources], True
    if belief is not None:
        try:
            h = belief.expected_hand(i)
            if h is not None and abs(sum(h) - p.hand_size) < 1.5:
                return [float(x) for x in h], False
        except Exception:  # pragma: no cover - a belief that cannot answer falls back to the prior
            pass
    w = hand_prior_weights(state, i)
    return [p.hand_size * x for x in w], False


def _vp_sources(state: GameState, i: int) -> List[Tuple[float, List[float], int, str, float]]:
    """Ways for player i to gain VP: ``(cost_per_vp, cost_vector, vp, label, min_turns)``."""
    p = state.players[i]
    out: List[Tuple[float, List[float], int, str, float]] = []
    # City upgrades.
    for _ in range(min(len(p.settlements), B.MAX_CITIES - len(p.cities))):
        out.append((float(sum(B.COST_CITY)), [float(x) for x in B.COST_CITY], 1, "city", 0.0))
    # Settlements on reachable spots (0-2 roads away).
    slots = B.MAX_SETTLEMENTS - len(p.settlements)
    if slots > 0:
        spots = sorted(reachable_spots(state, i, max_roads=2).items(), key=lambda kv: kv[1][0])
        roads_left = B.MAX_ROADS - len(p.roads)
        for v, (d, _e) in spots[:slots]:
            if d > roads_left:
                continue
            cost = [float(B.COST_SETTLEMENT[r] + d * B.COST_ROAD[r]) for r in range(5)]
            out.append((sum(cost), cost, 1, "settlement" + (f" (+{d} road{'s' if d > 1 else ''})" if d else ""), 0.0))
    # Longest Road.
    if state.longest_road_owner != i and len(p.roads) >= 3 and len(p.roads) < B.MAX_ROADS:
        target = (state.longest_road_len if state.longest_road_owner >= 0 else 4) + 1
        k = target - longest_road_length(state, i)
        if 0 < k <= B.MAX_ROADS - len(p.roads):
            cost = [float(k * B.COST_ROAD[r]) for r in range(5)]
            out.append((sum(cost) / 2.0, cost, 2, f"longest road (+{k} roads)", 0.0))
    # Largest Army.
    pool = dev_pool(state)
    pool_total = sum(pool)
    p_knight = pool[B.DEV_KNIGHT] / pool_total if pool_total > 0 else 0.0
    p_vp = pool[B.DEV_VP] / pool_total if pool_total > 0 else 0.0
    if state.largest_army_owner != i:
        holder = state.largest_army_owner
        holder_k = state.players[holder].played_knights if holder >= 0 else 2
        k = holder_k + 1 - p.played_knights
        if p.dev_known:
            held = p.dev_cards[B.DEV_KNIGHT] + p.dev_cards_new[B.DEV_KNIGHT]
        else:
            held = p.dev_count * p_knight
        to_buy = max(0.0, k - held)
        if k > 0 and (to_buy <= 0 or p_knight > 0):
            per_card = (1.0 / p_knight) if p_knight > 0 else 0.0
            cost = [to_buy * per_card * B.COST_DEV[r] for r in range(5)]
            out.append((sum(cost) / 2.0, cost, 2, f"largest army (+{k} knight{'s' if k > 1 else ''})", float(k)))
    # Hidden VP cards still in the pool (each buy is a lottery ticket).
    if p_vp > 0:
        per_vp = [B.COST_DEV[r] / p_vp for r in range(5)]
        for _ in range(int(round(pool[B.DEV_VP]))):
            out.append((sum(per_vp), list(per_vp), 1, "VP card", 0.0))
    out.sort(key=lambda t: t[0])
    return out


def win_path(state: GameState, i: int, belief=None) -> WinPath:
    p = state.players[i]
    n = state.num_players
    vp = state.public_vp(i) + expected_hidden_vp(state, i)
    need_vp = int(B.VP_TO_WIN - round(vp))
    hand, known = _expected_hand(state, i, belief)
    prod = player_production(state, i, ignore_robber=True)
    prod_now = player_production(state, i, ignore_robber=False)
    blocked_now = sum(prod) - sum(prod_now)
    steps: List[str] = []
    cost = [0.0] * 5
    min_turns = 0.0
    if state.phase == PHASE_GAME_OVER and state.winner == i:
        need_vp = 0
    if need_vp > 0:
        # Greedy: the source whose cards are *closest to being in hand* (per VP) first - a hand
        # holding ore + wheat makes the city the next step even though a settlement is cheaper.
        left = need_vp
        sources = _vp_sources(state, i)
        remaining = list(hand)
        while left > 0 and sources:
            def gap(src):
                cvec, gain = src[1], src[2]
                return (sum(max(0.0, cvec[r] - remaining[r]) for r in range(5)) / gain, src[0])
            best = min(sources, key=gap)
            sources.remove(best)
            _cpv, cvec, gain, label, mt = best
            steps.append(label)
            for r in range(5):
                cost[r] += cvec[r]
                remaining[r] = max(0.0, remaining[r] - cvec[r])
            min_turns = max(min_turns, mt)
            left -= gain
        if left > 0:   # no path at all (nothing left to build): treat as very far
            steps.append(f"no way to gain {left} VP")
            cost = [c + 10.0 for c in cost]
    missing = [max(0.0, cost[r] - hand[r]) for r in range(5)]
    tot_missing = sum(missing)
    need_share = [m / tot_missing if tot_missing > 1e-9 else 0.0 for m in missing]
    # Supply per resource: own production plus what a surplus buys through the best port.
    surplus = [prod[q] * (1.0 - need_share[q]) for q in range(5)]
    ratio = [state.port_ratio(i, q) for q in range(5)]
    trade_in = [0.0] * 5
    for r in range(5):
        if state.bank[r] <= 0:
            continue
        trade_in[r] = sum(surplus[q] / ratio[q] for q in range(5) if q != r)
    supply = [prod[r] + trade_in[r] for r in range(5)]
    eff_need = list(need_share)
    for q in range(5):
        for r in range(5):
            if r == q or need_share[r] <= 0 or supply[r] <= 1e-9:
                continue
            eff_need[q] += need_share[r] * (surplus[q] / ratio[q]) / supply[r]
    # Turns: per-resource bottleneck vs. total throughput, per round of n rolls.
    turns = 0.0
    if need_vp > 0:
        for r in range(5):
            if missing[r] <= 1e-9:
                continue
            per_turn = supply[r] * n
            turns = max(turns, missing[r] / per_turn if per_turn > 1e-9 else MAX_TURNS)
        total_per_turn = sum(prod) * n
        if tot_missing > 1e-9:
            turns = max(turns, tot_missing / total_per_turn if total_per_turn > 1e-9 else MAX_TURNS)
        turns = max(turns, min_turns - 1.0)
        turns = min(MAX_TURNS, turns)
    can_win_now = need_vp > 0 and tot_missing <= 1e-9 and min_turns <= 1.0
    danger = 1.0 if need_vp <= 0 or can_win_now else 1.0 / (1.0 + turns / TURNS_HALF)
    # Rolls that feed the missing resources.
    rolls: Dict[int, List[int]] = {}
    for r in range(5):
        if missing[r] <= 0.05:
            continue
        nums = set()
        for v in list(p.settlements) + list(p.cities):
            for h in B.VERTEX_HEXES[v]:
                res, num = state.hexes[h]
                if res == r and num:
                    nums.add(num)
        rolls[r] = sorted(nums)
    return WinPath(player=i, vp=vp, need_vp=need_vp, steps=steps, cost=cost, hand=hand, hand_known=known,
                   missing=missing, need_share=need_share, eff_need=eff_need, prod=prod, supply=supply,
                   turns=turns, min_turns=min_turns, rolls=rolls, can_win_now=can_win_now, danger=danger,
                   blocked_now=blocked_now)


def danger_multiplier(wp: Optional[WinPath]) -> float:
    """Multiplier on the VP threat: 0.4 (far from a win) .. 2.0 (can win now)."""
    if wp is None:
        return 1.0
    return 0.4 + 1.6 * wp.danger


def block_factor(wp: Optional[WinPath], res: int, pips: int) -> float:
    """How much blocking ``pips`` of resource ``res`` hurts this player's win path (about 0.35 .. 2.5).

    Scales with the effective need for the resource (direct need plus its use
    as port currency for a needed one) and with the share of the player's
    supply of it that the hex represents: a resource they already hold, produce
    elsewhere or can buy from a surplus is discounted.
    """
    if wp is None or pips <= 0:
        return 1.0
    need = BLOCK_FLOOR + BLOCK_NEED * min(1.0, wp.eff_need[res])
    supply_pips = wp.supply[res] * 36.0
    share = pips / supply_pips if supply_pips > 1e-9 else 1.0
    return need * (0.6 + 0.4 * min(1.0, share))


def steal_factor(wp: Optional[WinPath], our_need: Optional[Sequence[float]] = None) -> float:
    """Value of one random card from this hand (about 0.5 .. 1.6): what they need, what we need."""
    if wp is None:
        return 1.0
    comp = wp.composition()
    f = 0.0
    for r in range(5):
        f += comp[r] * (0.6 + 0.9 * wp.need_share[r] + (0.5 * our_need[r] if our_need is not None else 0.0))
    return f


def rob_break_probability(wp: Optional[WinPath]) -> float:
    """Chance that stealing one random card breaks a can-win-now hand."""
    if wp is None or not wp.can_win_now:
        return 0.0
    tot = sum(wp.hand)
    if tot <= 0:
        return 0.0
    return min(1.0, sum(wp.cost) / tot)


def danger_lines(state: GameState, me: int, belief=None, k: int = 3) -> List[str]:
    """Advice lines: who is really close to winning and why (most dangerous first)."""
    paths = win_paths(state, belief)
    order = sorted((j for j in range(state.num_players) if j != me),
                   key=lambda j: (-paths[j].danger, -paths[j].vp))
    lines = [paths[j].summary(state) for j in order[:k]]
    mine = paths.get(me)
    if mine is not None and mine.need_vp > 0:
        lines.append("You: " + mine.summary(state).split(": ", 1)[1])
    return lines
