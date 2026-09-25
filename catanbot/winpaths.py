"""Win-path races with crowding: Longest Road / Largest Army as contested races (off by default).

``heuristic.static_value`` credits an award holder the full 2 VP (20 points) as if the award were
permanent, and every challenger a flat progress credit whether or not anyone else races for it
(0.15 / 0.35 per road length, 0.45 per played knight, 0.45 of the 0.7 per held knight).  Top players
treat the two awards as *zero-sum races*: a race is only worth entering when you can win it, a
crowded race wastes the cards of everyone in it, and the uncrowded path gets cheaper.

This module replaces those award credits, for every seat, with a race-aware expected value:

* ``P(hold the award at game end)``: a softmax over *projected levels* - the current level
  (road length / knights), cards in hand (weighted ``HAND_W``) and production-driven growth over
  the remaining horizon ``H`` (rounds), with one shared knight pool;
* a waste cost for the units spent fighting rivals who are close (the "crowded factor"
  ``N_close``: the soft count of rivals ahead or within one unit, which escalates the fight);
* a passive floor (``credit >= value of stopping now``): the option to quit the race.

The correction ``C_i = credit_i - S_i`` cancels static's implicit award credit ``S_i`` exactly (the
ledger), so the search compares its leaves on ``static + g * C`` (``PathsEvaluator``).  Optionally the
settlement-reach terms of static are rescaled by our chance of winning contested spots, and move
ordering is nudged (dev buys while Largest Army is open, no Longest Road bonus for roads in a lost
race).  ``race_lines`` prints the "Win paths" board of the CLI.

Invariants
----------
* ``SearchConfig.paths = 0`` (the default) runs no code of this module inside the bot.
* ``heuristic.static_value`` and ``cpp/*`` are untouched: the term wraps the evaluator.
* No RNG; every memo is a pure function of its key (cold and warm caches give identical values).
* Every module constant is read once per :class:`PathsContext` (``params``), so ``ParamBot``'s
  apply / restore around ``decide`` is sufficient; nothing is cached across decisions.

See docs/STRATEGY.md ("Win-path races") for the model, its limits and the knobs.
"""
from __future__ import annotations

import importlib
import math
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import accel as _accel
from . import actions as A
from . import board as B
from . import counting
from . import heuristic
from . import placement
from .state import GameState, PHASE_GAME_OVER, PHASE_MAIN, PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT

# ---------------------------------------------------------------------------
# Constants (units: static points, 10 per VP; time in rounds of n turns)
# ---------------------------------------------------------------------------
H_A = 0.5            # horizon: H = clamp(H_A + H_B * (10 - vmax), H_MIN, H_MAX) rounds
H_B = 2.0            #   (fitted on 7.6k turn-start states: remaining rounds = 0.13 + 2.42 (10 - vmax), MAE 3.2)
H_MIN = 1.0
H_MAX = 16.0
R_STAY = 2.0         # rounds the robber is expected to stay on a hex (share of the blocked income that counts)
TAU = 0.35           # share of the port-converted income that counts as supply of another resource
ETA = 0.6            # share of future roads that extend the longest trail
HAND_W = 0.5         # cards in hand count half: flexible, but not committed
TIE_LR = 1.5         # holder's tie bonus (a tie keeps the holder), Longest Road
TIE_LA = 1.0         # holder's tie bonus, Largest Army
MIN_LR = 5           # minimum level to claim the award
MIN_LA = 3
BETA = 1.2           # race temperature (softmax slope per unit of spread)
SD0 = 0.8            # spread of a race with no growth left
SD_GROW = 1.0        # spread added per unit of possible growth (hand + projected)
KAPPA = 0.5          # prize weight: holder 20 (1 - KAPPA (1 - P)), challenger 20 KAPPA P
KAPPA_F = 0.6        # points per future card
ROAD_RESIDUAL = 0.4  # share of a race road's value that remains (race roads also reach spots)
V_PROG_F = 0.4       # value of a progress card drawn instead of a knight, in future-card units
ESC = 0.5            # escalation of the waste per extra close rival
CROWD_K = 3.0        # slope of the "close rival" sigmoid
CROWD_CLOSE = 1.0    # a rival within this many units counts as close
LA_HELD_SHARE = 0.45  # the Largest Army share of static's 0.7 per held knight (fixed by static_value, not tunable)
LIVE_LR = 4          # Longest Road is live with a holder or once somebody's trail reaches this (0 = always)
LIVE_LA = 2          # Largest Army is live with a holder or once somebody has this many knights (0 = always)
SPOT_W = 1.0         # weight of the contested-spot term (cfg.paths_spots)
SPOT_MAX_T = 30.0    # cap on the rounds a player needs to afford a spot
PRIOR_SCALE = 4.0    # prior points per point of marginal credit
EXT_OFF = 0.3        # share of the road prior for a road that does not extend a dead end
ROAD_BONUS_CAP = 12.0
OLD_LR_BONUS = 8.0   # action_priors' Longest Road bonus that the race-aware one replaces
DEV_FLOOR = 45.0     # BUY_DEV prior when the marginal Largest Army credit per dev card is large ...
DEV_PROMOTE = 1.0    # ... i.e. g x p_knight x DeltaLA >= this
PLACEBO = 0          # 1 = control arm: every seat gets the next seat's correction
CACHE_CAP = 20000    # a memo dict is cleared when it grows beyond this
LR, LA = 0, 1

_PARAM_NAMES = ("H_A", "H_B", "H_MIN", "H_MAX", "R_STAY", "TAU", "ETA", "HAND_W", "TIE_LR", "TIE_LA", "MIN_LR",
                "MIN_LA", "BETA", "SD0", "SD_GROW", "KAPPA", "KAPPA_F", "ROAD_RESIDUAL", "V_PROG_F", "ESC", "CROWD_K",
                "CROWD_CLOSE", "LA_HELD_SHARE", "LIVE_LR", "LIVE_LA", "SPOT_W", "SPOT_MAX_T", "PRIOR_SCALE",
                "EXT_OFF", "ROAD_BONUS_CAP", "OLD_LR_BONUS", "DEV_FLOOR", "DEV_PROMOTE", "PLACEBO", "CACHE_CAP")

# static_value's own progress constants (mirrored, never tunable here): the ledger cancels exactly these.
_S_LR_AHEAD = 0.35
_S_LR_BEHIND = 0.15
_S_LA_PLAYED = 0.45
_AWARD = 20.0

_K, _VPC, _RBC, _YOP, _MONO = B.DEV_KNIGHT, B.DEV_VP, B.DEV_ROAD_BUILDING, B.DEV_YEAR_OF_PLENTY, B.DEV_MONOPOLY
_WOOD, _BRICK, _SHEEP, _WHEAT, _ORE = B.WOOD, B.BRICK, B.SHEEP, B.WHEAT, B.ORE
_SETUP = (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)


def current_params() -> Dict[str, Any]:
    """The module constants as they are now (a :class:`PathsContext` freezes them for its lifetime)."""
    g = globals()
    return {k: g[k] for k in _PARAM_NAMES}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
class RaceSolution(NamedTuple):
    P: tuple          # P(hold the award at game end) per seat
    P0: tuple         # the same if this seat stops growing now (rivals keep growing)
    nobody: float     # P(nobody reaches the minimum level) - 0 once somebody holds it
    credit: tuple     # max(V(P) - cost U, V(P0)) per seat, points
    U: tuple          # units spent fighting (with escalation)
    N_close: tuple    # soft count of rivals ahead or within CROWD_CLOSE units ("crowded factor")
    active: tuple     # True when racing beats the passive floor


class Supply(NamedTuple):
    inc: tuple        # cards per round by resource (robber-adjusted)
    s: tuple          # supply per round incl. port conversions, times the bank factor
    road_rate: float  # roads per round the supply can pay for
    dev_rate: float   # dev cards per round


class Races(NamedTuple):
    lr: RaceSolution
    la: RaceSolution
    S: List[float]    # static's implicit award credit per seat (LR + LA), cancelled by the ledger
    C: List[float]    # correction per seat (live races only, before the weight g and PLACEBO)
    inputs: dict


# ---------------------------------------------------------------------------
# Horizon and the race solver
# ---------------------------------------------------------------------------
def _vp_estimate(state: GameState, i: int) -> float:
    p = state.players[i]
    return state.public_vp(i) + (p.vp_cards if p.dev_known else counting.expected_hidden_vp(state, i))


def horizon(state: GameState, params: Optional[Dict[str, Any]] = None) -> float:
    """Remaining rounds: ``clamp(H_A + H_B (VP_TO_WIN - vmax), H_MIN, H_MAX)``."""
    p = params if params is not None else current_params()
    vmax = max(_vp_estimate(state, i) for i in range(state.num_players))
    h = p["H_A"] + p["H_B"] * (B.VP_TO_WIN - vmax)
    return min(p["H_MAX"], max(p["H_MIN"], h))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def solve_race(b: Sequence[float], a: Sequence[float], G: Sequence[float], holder: int, min_level: float,
               cost: float, params: Optional[Dict[str, Any]] = None) -> RaceSolution:
    """Win probabilities and credits of one race (pure; ``PathsContext`` memoizes it on rounded inputs).

    ``b`` current levels (hand and tie bonus included), ``a`` the hand part of ``b``, ``G`` projected
    growth over the horizon, ``holder`` the seat holding the award (-1 = nobody), ``min_level`` the
    level that claims it, ``cost`` points per unit spent fighting.  See docs/STRATEGY.md.
    """
    p = params if params is not None else current_params()
    kappa = p["KAPPA"]
    esc = p["ESC"]
    ck = p["CROWD_K"]
    cc = p["CROWD_CLOSE"]
    n = len(b)
    F = [b[i] + G[i] for i in range(n)]
    grow = max([a[k] + G[k] for k in range(n)] + [0.0])
    sigma = max(1e-9, math.sqrt(p["SD0"] * p["SD0"] + p["SD_GROW"] * grow))
    beta = p["BETA"] / sigma
    floor = min_level - 0.5
    M = max(F + [floor])
    e = [math.exp(beta * (f - M)) for f in F]
    null = math.exp(beta * (floor - M)) if holder < 0 else 0.0
    Z = sum(e) + null
    P = [x / Z for x in e]
    P0: List[float] = []
    credit: List[float] = []
    U: List[float] = []
    Nc: List[float] = []
    act: List[bool] = []
    for i in range(n):
        e0 = math.exp(beta * (b[i] - M))
        rest = null
        nc = 0.0
        top = -math.inf
        for j in range(n):
            if j == i:
                continue
            rest += e[j]
            nc += _sigmoid(ck * (F[j] - F[i] + cc))
            if F[j] > top:
                top = F[j]
        den = rest + e0
        p0 = e0 / den if den > 0.0 else P[i]
        gap = max(0.0, top + 1.0 - b[i]) if n > 1 else 0.0
        u = min(G[i], gap) * (1.0 + esc * max(0.0, nc - 1.0))
        if i == holder:
            v1 = _AWARD * (1.0 - kappa * (1.0 - P[i]))
            v0 = _AWARD * (1.0 - kappa * (1.0 - p0))
        else:
            v1 = _AWARD * kappa * P[i]
            v0 = _AWARD * kappa * p0
        racing = v1 - cost * u
        if racing > v0:
            credit.append(racing)
            act.append(True)
        else:
            credit.append(v0)
            act.append(False)
        P0.append(p0)
        U.append(u)
        Nc.append(nc)
    return RaceSolution(tuple(P), tuple(P0), null / Z, tuple(credit), tuple(U), tuple(Nc), tuple(act))


# ---------------------------------------------------------------------------
# Small helpers (static's own quantities)
# ---------------------------------------------------------------------------
def _accel_ok() -> bool:
    return _accel.AVAILABLE and (_accel._verified or _accel.verify())


def _heur_lr(state: GameState, i: int) -> int:
    """``heuristic.longest_road_length`` (the length static_value credits), C++ when usable."""
    if _accel_ok():
        try:
            return int(_accel._core.heuristic_longest_road_length(state, int(i)))
        except ValueError as exc:
            if not _accel._unsupported(exc):
                raise
    return heuristic.longest_road_length(state, i)


def _spot_score(state: GameState, me: int, v: int) -> float:
    """``placement.score_settlement_spot`` exactly as static_value computes it (C++ when usable)."""
    if _accel_ok():
        try:
            return float(_accel._core.score_settlement_spot(state, int(me), int(v)))
        except ValueError as exc:
            if not _accel._unsupported(exc):
                raise
    return placement.score_settlement_spot(state, me, v, occ=state.occupied_vertices(),
                                           own_prod=placement.player_production(state, me, ignore_robber=True),
                                           scarcity=placement.resource_scarcity(state))


def _osig(state: GameState) -> tuple:
    return tuple(tuple(sorted(p.settlements + p.cities)) for p in state.players)


def _name(state: GameState, i: int) -> str:
    p = state.players[i]
    return p.name or p.color


class _Leaf:
    """Race inputs of one state (per seat lists; the race tuples are rounded to 1e-4)."""

    __slots__ = ("osig", "hands", "kn", "rb", "L", "l", "room", "sup", "lr", "la", "lr_holder", "la_holder",
                 "played", "la_scale")


# ---------------------------------------------------------------------------
# Per-decision context
# ---------------------------------------------------------------------------
class PathsContext:
    """Everything fixed for one ``Searcher.search`` call (root quantities, frozen constants, memos).

    Opponents cannot build during our turn, so their supply / lengths / room hit the memos on every
    leaf after the first; per leaf only our own entries miss (plus an opponent's road length when our
    settlement cuts it).
    """

    def __init__(self, root: GameState, me: int, *, weight: float = 1.0, crowd: float = 1.0, spots: bool = False,
                 priors: bool = True, belief=None, params: Optional[Dict[str, Any]] = None):
        P = dict(params) if params is not None else current_params()
        self.params = P
        self.me = int(me)
        self.n = n = root.num_players
        self.weight = float(weight)
        self.crowd = float(crowd)
        self.spots = bool(spots)
        self.priors = bool(priors)
        self._root = root
        self._belief = belief
        self.stats: Dict[str, Any] = {"evals": 0, "race_hits": 0, "race_misses": 0, "lr_misses": 0, "room_misses": 0,
                                      "supply_misses": 0, "reach_misses": 0, "spot_misses": 0, "seconds": 0.0}
        self._cap = int(P["CACHE_CAP"])
        self._supply: Dict[tuple, Supply] = {}
        self._lr: Dict[tuple, Tuple[int, int]] = {}
        self._room: Dict[tuple, int] = {}
        self._race: Dict[tuple, RaceSolution] = {}
        self._reach: Dict[tuple, dict] = {}
        self._spot: Dict[tuple, float] = {}
        # frozen scalars used on every leaf
        self._hand_w = float(P["HAND_W"])
        self._eta = float(P["ETA"])
        self._tie_lr = float(P["TIE_LR"])
        self._tie_la = float(P["TIE_LA"])
        self._min_lr = P["MIN_LR"]
        self._min_la = P["MIN_LA"]
        self._placebo = int(P["PLACEBO"])
        self.vmax = max(_vp_estimate(root, i) for i in range(n))
        self.H = H = horizon(root, P)
        pool = counting.dev_pool(root)
        tot = sum(pool)
        self.pool = pool
        self.pool_total = tot
        if tot > 0:
            self.p_k = pool[_K] / tot
            self.p_vp = pool[_VPC] / tot
            self.p_rb = pool[_RBC] / tot
            self.p_yop = pool[_YOP] / tot
            self.p_mono = pool[_MONO] / tot
        else:
            self.p_k = self.p_vp = self.p_rb = self.p_yop = self.p_mono = 0.0
        self.p_prog = self.p_rb + self.p_yop + self.p_mono
        hidden_dev = sum(p.dev_count for p in root.players if not p.dev_known)
        self.K_left = pool[_K] * max(0.0, 1.0 - hidden_dev / max(1, tot))
        self._w: Dict[int, List[float]] = {j: counting.hand_prior_weights(root, j)
                                           for j in range(n) if not root.players[j].hand_known}
        # root incomes (no bank factor) -> the bank factor
        self.stay = min(1.0, P["R_STAY"] / H)
        incs = [self._income(root, j) for j in range(n)]
        tot_inc = [sum(inc[r] for inc in incs) for r in range(5)]
        self.bf = [min(1.0, root.bank[r] / (tot_inc[r] + 1.0)) for r in range(5)]
        osig = _osig(root)
        L_root = [self.lr_lengths(root, i, osig)[0] for i in range(n)]
        kn_root = [self._held_knights(root.players[i]) for i in range(n)]
        self.live_lr = (root.longest_road_owner >= 0) or (max(L_root) >= P["LIVE_LR"])
        self.live_la = (root.largest_army_owner >= 0) or (
            max(root.players[i].played_knights + kn_root[i] for i in range(n)) >= P["LIVE_LA"])
        self._room_root = {j: self._compute_room(root, j) for j in range(n) if j != self.me}
        self.c_lr = round(self.crowd * P["KAPPA_F"] * (2.0 / P["ETA"]) * (1.0 - P["ROAD_RESIDUAL"]), 4)
        if self.p_k > 0:
            raw = max(0.0, 3.0 * P["KAPPA_F"] - self.p_vp * 10.0 * P["KAPPA"] - self.p_prog * P["V_PROG_F"]) / self.p_k
            self.c_la = round(self.crowd * raw, 4)
        else:
            self.c_la = 0.0
        self._scarcity = None

    # --- memo plumbing ------------------------------------------------------------------
    def _put(self, d: dict, key, value):
        if len(d) >= self._cap:
            d.clear()
        d[key] = value
        return value

    # --- per-seat quantities --------------------------------------------------------------
    def _income(self, state: GameState, i: int) -> List[float]:
        pf = placement.player_production(state, i, ignore_robber=True)
        pn = placement.player_production(state, i, ignore_robber=False)
        n = self.n
        stay = self.stay
        return [n * (pf[r] - stay * (pf[r] - pn[r])) for r in range(5)]

    def supply(self, state: GameState, i: int) -> Supply:
        """Cards per round of seat ``i`` (memo: its buildings and the robber)."""
        p = state.players[i]
        key = (i, tuple(p.settlements), tuple(p.cities), state.robber)
        hit = self._supply.get(key)
        if hit is not None:
            return hit
        self.stats["supply_misses"] += 1
        inc = self._income(state, i)
        conv = [inc[q] / state.port_ratio(i, q) for q in range(5)]
        sc = sum(conv)
        tau = self.params["TAU"]
        bf = self.bf
        s = [(inc[r] + tau * (sc - conv[r])) * bf[r] for r in range(5)]
        tot = sum(inc)
        road_rate = min(s[_WOOD], s[_BRICK], tot / 2.0)
        dev_rate = min(s[_SHEEP], s[_WHEAT], s[_ORE], tot / 3.0)
        return self._put(self._supply, key, Supply(tuple(inc), tuple(s), road_rate, dev_rate))

    def lr_lengths(self, state: GameState, i: int, osig: tuple) -> Tuple[int, int]:
        """(official trail length that awards the card, static's heuristic length) of seat ``i``."""
        key = (i, tuple(state.players[i].roads), osig)
        hit = self._lr.get(key)
        if hit is not None:
            return hit
        self.stats["lr_misses"] += 1
        return self._put(self._lr, key, (int(_accel.longest_road_length(state, i)), int(_heur_lr(state, i))))

    @staticmethod
    def _compute_room(state: GameState, i: int) -> int:
        p = state.players[i]
        occ = state.occupied_vertices()
        eocc = state.occupied_edges()
        verts = set(p.settlements) | set(p.cities)
        for e in p.roads:
            verts.update(B.EDGE_VERTICES[e])
        edges = set()
        for v in verts:
            o = occ.get(v, i)
            if o != i:
                continue
            for e in B.VERTEX_EDGES[v]:
                if e not in eocc:
                    edges.add(e)
        return min(B.MAX_ROADS - len(p.roads), 2 * len(edges))

    def room(self, state: GameState, i: int, osig: tuple) -> int:
        """Roads seat ``i`` can still add to its network (ours per leaf, the opponents' from the root)."""
        if i != self.me:
            r = self._room_root.get(i)
            if r is not None:
                return r
            return self._compute_room(state, i)
        key = (tuple(state.players[i].roads), osig)
        hit = self._room.get(key)
        if hit is not None:
            return hit
        self.stats["room_misses"] += 1
        return self._put(self._room, key, self._compute_room(state, i))

    def _held_knights(self, p) -> float:
        if p.dev_known:
            return p.dev_cards[_K] + p.dev_cards_new[_K]
        return p.dev_count * self.p_k

    def _hand(self, i: int, p) -> List[float]:
        if self._belief is not None:
            try:
                h = self._belief.expected_hand(i)
                if h is not None and abs(sum(h) - p.hand_size) < 1.5:
                    return [float(x) for x in h]
            except Exception:  # pragma: no cover - a belief that cannot answer falls back to the prior
                pass
        w = self._w.get(i)
        if w is None:
            w = counting.hand_prior_weights(self._root, i)
            self._w[i] = w
        hs = p.hand_size
        return [hs * w[r] for r in range(5)]

    # --- race inputs ------------------------------------------------------------------------
    def _inputs(self, state: GameState) -> _Leaf:
        players = state.players
        n = self.n
        me = self.me
        H = self.H
        p_k = self.p_k
        p_rb = self.p_rb
        hw = self._hand_w
        eta_h = self._eta * H
        lr_holder = state.longest_road_owner
        la_holder = state.largest_army_owner
        osig = tuple(tuple(sorted(p.settlements + p.cities)) for p in players)
        hands = []
        kns = []
        rbs = []
        Ls = []
        ls = []
        rooms = []
        sups = []
        played = []
        lr_b = []
        lr_a = []
        lr_G = []
        la_b = []
        la_a = []
        la_G = []
        for i in range(n):
            p = players[i]
            hand = p.resources if p.hand_known else self._hand(i, p)
            if p.dev_known:
                dc = p.dev_cards
                dn = p.dev_cards_new
                k = dc[_K] + dn[_K]
                r = dc[_RBC] + dn[_RBC]
            else:
                k = p.dev_count * p_k
                r = p.dev_count * p_rb
            sup = self.supply(state, i)
            L, l = self.lr_lengths(state, i, osig)
            room = self.room(state, i, osig)
            a_raw = min(room, min(hand[_WOOD], hand[_BRICK]) + 2 * r)
            G = min(max(0, room - a_raw), eta_h * sup.road_rate)
            a = hw * a_raw
            lr_a.append(round(float(a), 4))
            lr_G.append(round(float(G), 4))
            lr_b.append(round(float(L + a + (self._tie_lr if i == lr_holder else 0.0)), 4))
            pk = p.played_knights
            a2 = hw * p_k * min(hand[_SHEEP], hand[_WHEAT], hand[_ORE])
            la_a.append(a2)
            la_G.append(min(1.0, p_k * sup.dev_rate) * H)
            la_b.append(round(float(pk + min(k, H) + a2 + (self._tie_la if i == la_holder else 0.0)), 4))
            hands.append(hand)
            kns.append(k)
            rbs.append(r)
            Ls.append(L)
            ls.append(l)
            rooms.append(room)
            sups.append(sup)
            played.append(pk)
        tot = sum(la_G)
        scale = 1.0
        if tot > 0.0 and self.K_left < tot:
            scale = self.K_left / tot
            la_G = [g * scale for g in la_G]
        leaf = _Leaf()
        leaf.osig = osig
        leaf.hands = hands
        leaf.kn = kns
        leaf.rb = rbs
        leaf.L = Ls
        leaf.l = ls
        leaf.room = rooms
        leaf.sup = sups
        leaf.played = played
        leaf.lr_holder = lr_holder
        leaf.la_holder = la_holder
        leaf.la_scale = scale
        leaf.lr = (tuple(lr_b), tuple(lr_a), tuple(lr_G))
        leaf.la = (tuple(la_b), tuple(round(float(x), 4) for x in la_a), tuple(round(float(x), 4) for x in la_G))
        return leaf

    def race_inputs(self, state: GameState) -> dict:
        """Per race ``(b, a, G, holder)`` plus the raw per-seat quantities (tests / diagnostics)."""
        leaf = self._inputs(state)
        return {"lr": leaf.lr + (leaf.lr_holder,), "la": leaf.la + (leaf.la_holder,), "L": list(leaf.L),
                "l": list(leaf.l), "room": list(leaf.room), "kn": list(leaf.kn), "rb": list(leaf.rb),
                "played": list(leaf.played), "hands": [list(h) for h in leaf.hands], "la_scale": leaf.la_scale,
                "supply": list(leaf.sup)}

    # --- solving --------------------------------------------------------------------------------
    def _solve(self, race: int, b: tuple, a: tuple, G: tuple, holder: int) -> RaceSolution:
        if race == LR:
            min_level, cost = self._min_lr, self.c_lr
        else:
            min_level, cost = self._min_la, self.c_la
        key = (race, b, a, G, holder, min_level, cost)
        sol = self._race.get(key)
        if sol is not None:
            self.stats["race_hits"] += 1
            return sol
        self.stats["race_misses"] += 1
        return self._put(self._race, key, solve_race(b, a, G, holder, min_level, cost, self.params))

    def _ledger(self, state: GameState, leaf: _Leaf) -> Tuple[List[float], List[float]]:
        """static's implicit award credit per seat: ``(S_LR, S_LA)``."""
        lr_h = leaf.lr_holder
        hl = state.longest_road_len if lr_h >= 0 else 4
        la_h = leaf.la_holder
        share = self.params["LA_HELD_SHARE"]
        s_lr = []
        s_la = []
        for i in range(self.n):
            if i == lr_h:
                s_lr.append(_AWARD)
            else:
                l = leaf.l[i]
                s_lr.append((_S_LR_AHEAD if l >= hl else _S_LR_BEHIND) * l)
            s_la.append((_AWARD if i == la_h else _S_LA_PLAYED * leaf.played[i]) + share * leaf.kn[i])
        return s_lr, s_la

    def _raw_C(self, state: GameState, leaf: _Leaf) -> List[float]:
        n = self.n
        C = [0.0] * n
        if not (self.live_lr or self.live_la):
            return C
        s_lr, s_la = self._ledger(state, leaf)
        if self.live_lr:
            sol = self._solve(LR, leaf.lr[0], leaf.lr[1], leaf.lr[2], leaf.lr_holder)
            cr = sol.credit
            for i in range(n):
                C[i] += cr[i] - s_lr[i]
        if self.live_la:
            sol = self._solve(LA, leaf.la[0], leaf.la[1], leaf.la[2], leaf.la_holder)
            cr = sol.credit
            for i in range(n):
                C[i] += cr[i] - s_la[i]
        return C

    def races(self, state: GameState) -> Races:
        """Both races solved (whether live or not), the ledger and the correction of live races."""
        leaf = self._inputs(state)
        lr = self._solve(LR, leaf.lr[0], leaf.lr[1], leaf.lr[2], leaf.lr_holder)
        la = self._solve(LA, leaf.la[0], leaf.la[1], leaf.la[2], leaf.la_holder)
        s_lr, s_la = self._ledger(state, leaf)
        S = [s_lr[i] + s_la[i] for i in range(self.n)]
        C = self._raw_C(state, leaf)
        inputs = {"lr": leaf.lr + (leaf.lr_holder,), "la": leaf.la + (leaf.la_holder,), "S_LR": s_lr, "S_LA": s_la}
        return Races(lr, la, S, C, inputs)

    # --- contested settlement spots -----------------------------------------------------------
    def _reach_of(self, state: GameState, j: int, osig: tuple) -> dict:
        key = (j, tuple(state.players[j].roads), osig, tuple(state.players[self.me].roads))
        hit = self._reach.get(key)
        if hit is not None:
            return hit
        self.stats["reach_misses"] += 1
        return self._put(self._reach, key, placement.reachable_spots(state, j, max_roads=2))

    def _spot_value(self, state: GameState, v: int, osig: tuple) -> float:
        p = state.players[self.me]
        # our roads and the award owners are in the key as well (the spot's expansion term reads every road,
        # its blockability term the opponents' VP): the memo stays a pure function of its key
        key = (v, tuple(p.settlements), tuple(p.cities), tuple(p.roads), osig, state.longest_road_owner,
               state.largest_army_owner)
        hit = self._spot.get(key)
        if hit is not None:
            return hit
        self.stats["spot_misses"] += 1
        return self._put(self._spot, key, _spot_score(state, self.me, v))

    def _turns_to_afford(self, hand: Sequence[float], s: Sequence[float], d: int) -> float:
        cost = (1 + d, 1 + d, 1, 1, 0)
        t = 0.0
        for r in range(5):
            miss = cost[r] - hand[r]
            if miss > 0.0:
                x = miss / max(s[r], 1e-6)
                if x > t:
                    t = x
        return min(self.params["SPOT_MAX_T"], t)

    def _spot_table(self, state: GameState, leaf: _Leaf) -> List[tuple]:
        """``[(v, d, P_me, [(rival, d_j, T_j + o_j), ...]), ...]`` for every spot within 2 roads of us."""
        me = self.me
        n = self.n
        osig = leaf.osig
        reach_me = self._reach_of(state, me, osig)
        if not reach_me:
            return []
        rivals = [(j, self._reach_of(state, j, osig)) for j in range(n) if j != me]
        t_me: Dict[int, float] = {}
        out = []
        for v, (d, _e) in reach_me.items():
            riv = [(j, rj[v][0]) for j, rj in rivals if v in rj]
            if not riv:
                out.append((v, d, 1.0, []))
                continue
            if d not in t_me:
                t_me[d] = self._turns_to_afford(leaf.hands[me], leaf.sup[me].s, d)
            r_me = 1.0 / (t_me[d] + 1.0)
            tot = r_me
            info = []
            for j, dj in riv:
                o = ((j - me) % n) / n
                tj = self._turns_to_afford(leaf.hands[j], leaf.sup[j].s, dj) + o
                tot += 1.0 / tj
                info.append((j, dj, tj))
            out.append((v, d, r_me / tot, info))
        return out

    def spot_correction(self, state: GameState, leaf: Optional[_Leaf] = None) -> float:
        """Rescaled settlement-reach terms of static for our seat (0 when no rival reaches our spots)."""
        p = state.players[self.me]
        if len(p.settlements) >= B.MAX_SETTLEMENTS:
            return 0.0
        leaf = leaf if leaf is not None else self._inputs(state)
        table = self._spot_table(state, leaf)
        if not table or all(not info for _v, _d, _pm, info in table):
            return 0.0
        best_now = 0.0
        best_base = 0.0
        sum_p0 = 0.0
        cnt0 = 0
        for v, d, pm, _info in table:
            sv = self._spot_value(state, v, leaf.osig) / (1.0 + 0.9 * d)
            if sv > best_base:
                best_base = sv
            if sv * pm > best_now:
                best_now = sv * pm
            if d == 0:
                sum_p0 += pm
                cnt0 += 1
        return 0.12 * (best_now - best_base) + 0.6 * (min(3.0, sum_p0) - min(3, cnt0))

    # --- the correction the evaluator adds ------------------------------------------------------
    def corrections(self, state: GameState) -> List[float]:
        """``g C_i + [i = me] g SPOT_W C_spot`` per seat (points; PLACEBO rotates C)."""
        self.stats["evals"] += 1
        n = self.n
        if not (self.live_lr or self.live_la or self.spots):
            return [0.0] * n            # nothing live: static applies unchanged (the gates are fixed per context)
        leaf = self._inputs(state)
        C = self._raw_C(state, leaf)
        if self._placebo:
            C = [C[(i + 1) % n] for i in range(n)]
        g = self.weight
        out = [g * c for c in C]
        if self.spots:
            out[self.me] += g * self.params["SPOT_W"] * self.spot_correction(state, leaf)
        return out

    # --- marginal credit of one more unit -----------------------------------------------------
    def _marginal(self, leaf: _Leaf, race: int, units: float = 1.0) -> float:
        me = self.me
        if race == LR:
            b, a, G = (list(x) for x in leaf.lr)
            base = self._solve(LR, leaf.lr[0], leaf.lr[1], leaf.lr[2], leaf.lr_holder).credit[me]
            hand = leaf.hands[me]
            room = max(0.0, leaf.room[me] - units)
            a_raw = min(room, min(hand[_WOOD], hand[_BRICK]) + 2 * leaf.rb[me])
            aa = self._hand_w * a_raw
            G[me] = round(float(min(max(0.0, room - a_raw), self._eta * self.H * leaf.sup[me].road_rate)), 4)
            a[me] = round(float(aa), 4)
            b[me] = round(float(leaf.L[me] + units + aa + (self._tie_lr if me == leaf.lr_holder else 0.0)), 4)
            new = self._solve(LR, tuple(b), tuple(a), tuple(G), leaf.lr_holder).credit[me]
            return new - base
        b, a, G = (list(x) for x in leaf.la)
        base = self._solve(LA, leaf.la[0], leaf.la[1], leaf.la[2], leaf.la_holder).credit[me]
        hand = leaf.hands[me]
        a2 = self._hand_w * self.p_k * min(hand[_SHEEP], hand[_WHEAT], hand[_ORE])
        b[me] = round(float(leaf.played[me] + min(leaf.kn[me] + units, self.H) + a2
                            + (self._tie_la if me == leaf.la_holder else 0.0)), 4)
        new = self._solve(LA, tuple(b), tuple(a), tuple(G), leaf.la_holder).credit[me]
        return new - base

    def marginal(self, state: GameState, race: int, units: float = 1.0) -> float:
        """Change of our credit in ``race`` (LR: +1 length and -1 room; LA: +1 knight held)."""
        return self._marginal(self._inputs(state), race, units)

    # --- move ordering --------------------------------------------------------------------------
    def adjust_priors(self, state: GameState, legal: Sequence, priors: Sequence[float]) -> List[float]:
        """Race-aware nudges of our own main-phase priors (BUY_DEV, paid BUILD_ROAD); others unchanged."""
        out = list(priors)
        me = self.me
        if state.phase != PHASE_MAIN or state.current != me:
            return out
        P = self.params
        want_dev = self.live_la and self.p_k > 0 and any(a[0] == A.BUY_DEV for a in legal)
        want_road = self.live_lr and state.free_roads == 0 and any(a[0] == A.BUILD_ROAD for a in legal)
        if not (want_dev or want_road):
            return out
        leaf = self._inputs(state)
        gp = min(1.0, self.weight)
        scale = P["PRIOR_SCALE"]
        if want_dev:
            d_la = self._marginal(leaf, LA)
            bonus = gp * scale * self.p_k * d_la
            promote = self.weight * self.p_k * d_la >= P["DEV_PROMOTE"]
            for i, a in enumerate(legal):
                if a[0] == A.BUY_DEV:
                    out[i] += bonus
                    if promote:
                        out[i] = max(out[i], P["DEV_FLOOR"])
        if want_road:
            d_lr = self._marginal(leaf, LR)
            old = P["OLD_LR_BONUS"] if (leaf.l[me] >= 4 and state.longest_road_owner != me) else 0.0
            deg: Dict[int, int] = {}
            for e in state.players[me].roads:
                for v in B.EDGE_VERTICES[e]:
                    deg[v] = deg.get(v, 0) + 1
            occ = state.occupied_vertices()
            cap = P["ROAD_BONUS_CAP"]
            for i, a in enumerate(legal):
                if a[0] != A.BUILD_ROAD:
                    continue
                ext = P["EXT_OFF"]
                for u in B.EDGE_VERTICES[a[1]]:
                    if deg.get(u, 0) == 1 and occ.get(u, me) == me:
                        ext = 1.0
                        break
                new = min(cap, max(0.0, scale * d_lr * ext))
                out[i] += gp * (new - old)
        return out

    # --- diagnostics ------------------------------------------------------------------------------
    def summary(self, state: GameState) -> dict:
        """Everything the model computes for ``state`` (tests, the advisor, diagnostics)."""
        leaf = self._inputs(state)
        rc = self.races(state)
        return {
            "H": self.H, "vmax": self.vmax, "live_lr": self.live_lr, "live_la": self.live_la,
            "c_lr": self.c_lr, "c_la": self.c_la, "p_k": self.p_k, "p_vp": self.p_vp, "K_left": self.K_left,
            "bf": list(self.bf), "inputs": self.race_inputs(state), "lr": rc.lr._asdict(), "la": rc.la._asdict(),
            "S": list(rc.S), "S_LR": list(rc.inputs["S_LR"]), "S_LA": list(rc.inputs["S_LA"]), "C": list(rc.C),
            "corrections": self.corrections(state),
            "spot": self.spot_correction(state, leaf) if self.spots else 0.0,
            "marginal_lr": self._marginal(leaf, LR), "marginal_la": self._marginal(leaf, LA),
            "stats": dict(self.stats),
        }


# ---------------------------------------------------------------------------
# The evaluator wrapper
# ---------------------------------------------------------------------------
def _base_kind(base) -> str:
    from .heuristic import HeuristicEvaluator
    if isinstance(base, HeuristicEvaluator) and type(base).evaluate is HeuristicEvaluator.evaluate:
        return "heuristic"
    h = getattr(base, "heuristic", None)
    if hasattr(base, "net") and hasattr(base, "alpha") and h is not None and hasattr(h, "temperature"):
        return "blend"
    return "other"


class PathsEvaluator:
    """``softmax((static + g C) / T)`` in place of the base evaluator's heuristic part (see module doc)."""

    name = "paths"

    def __init__(self, base, ctx: PathsContext):
        self.base = base
        self.ctx = ctx
        self.kind = _base_kind(base)
        if self.kind == "heuristic":
            self.temperature = float(base.temperature)
        elif self.kind == "blend":
            self.temperature = float(base.heuristic.temperature)
        else:
            self.temperature = 16.0

    @classmethod
    def for_search(cls, base, root: GameState, me: int, cfg) -> "PathsEvaluator":
        ctx = PathsContext(root, me, weight=cfg.paths_w, crowd=cfg.paths_crowd, spots=bool(cfg.paths_spots),
                           priors=bool(cfg.paths_priors))
        return cls(base, ctx)

    def _probs(self, s: GameState, corrected: bool = True) -> List[float]:
        V = _accel.static_values(s)
        if corrected:
            corr = self.ctx.corrections(s)
            z = [V[i] + corr[i] for i in range(len(V))]
        else:
            z = V
        m = max(z)
        T = self.temperature
        exps = [math.exp((v - m) / T) for v in z]
        tot = sum(exps)
        return [x / tot for x in exps]

    def _heuristic_part(self, states: Sequence[GameState], players: Sequence[int]):
        """Corrected heuristic values; ``None`` entries for setup states (the base handles those)."""
        out: List[Optional[float]] = []
        cache: Dict[int, List[float]] = {}
        for s, pl in zip(states, players):
            if s.phase == PHASE_GAME_OVER and s.winner >= 0:
                out.append(1.0 if s.winner == pl else 0.0)
                continue
            if s.phase in _SETUP:
                out.append(None)
                continue
            key = id(s)
            probs = cache.get(key)
            if probs is None:
                probs = self._probs(s)
                cache[key] = probs
            out.append(probs[pl])
        return out

    def evaluate(self, states: Sequence[GameState], players: Sequence[int]) -> np.ndarray:
        t0 = time.perf_counter()
        states = list(states)
        players = list(players)
        N = len(states)
        kind = self.kind
        try:
            if kind == "other":
                out = np.asarray(self.base.evaluate(states, players), dtype=np.float64).copy()
                cache: Dict[int, Tuple[List[float], List[float]]] = {}
                for k, (s, pl) in enumerate(zip(states, players)):
                    if (s.phase == PHASE_GAME_OVER and s.winner >= 0) or s.phase in _SETUP:
                        continue
                    key = id(s)
                    pair = cache.get(key)
                    if pair is None:
                        pair = (self._probs(s, True), self._probs(s, False))
                        cache[key] = pair
                    out[k] += pair[0][pl] - pair[1][pl]
                return out
            if kind == "blend":
                alpha = float(self.base.alpha)
                vn = np.asarray(self.base.net.evaluate(states, players), dtype=np.float64)
                if alpha >= 1.0:
                    return vn
            vh = self._heuristic_part(states, players)
            setup = [k for k in range(N) if vh[k] is None]
            if setup:
                heur = self.base.heuristic if kind == "blend" else self.base
                sv = heur.evaluate([states[k] for k in setup], [players[k] for k in setup])
                for k, v in zip(setup, sv):
                    vh[k] = float(v)
            h = np.asarray(vh, dtype=np.float64)
            if kind == "blend":
                return alpha * vn + (1.0 - alpha) * h
            return h
        finally:
            self.ctx.stats["seconds"] += time.perf_counter() - t0

    __call__ = evaluate


# ---------------------------------------------------------------------------
# Advisor ("Win paths" section of the CLI)
# ---------------------------------------------------------------------------
_RACE_NAMES = {LR: "Longest Road", LA: "Largest Army"}


def _verdict(ctx: PathsContext, sol: RaceSolution, holder: int, delta: float) -> str:
    me = ctx.me
    if holder == me:
        return "safe" if sol.P[me] >= 0.8 else "defend"
    if sol.active[me]:
        return f"OPEN - worth racing ({delta / 10.0:+.2f} VP per unit)"
    if sol.N_close[me] >= 1.5 or sol.P[me] < 0.10:
        return f"CROWDED - don't race ({delta / 10.0:+.2f} VP per unit)"
    return "passive"


def portfolio(state: GameState, me: int, ctx: Optional[PathsContext] = None) -> List[Tuple[str, float, str]]:
    """Value per card of each path for ``me``, best first: ``[(path, points per card, note), ...]``."""
    ctx = ctx if ctx is not None else PathsContext(state, me, spots=True)
    p = state.players[me]
    base = _accel.static_values(state)[me]
    out: List[Tuple[str, float, str]] = []
    if p.settlements and len(p.cities) < B.MAX_CITIES:
        spots = placement.best_city_spots(state, me, k=1)
        if spots:
            v = spots[0][0]
            s2 = state.copy()
            s2.players[me].settlements.remove(v)
            s2.players[me].cities.append(v)
            out.append(("city", (_accel.static_values(s2)[me] - base) / 5.0, f"vertex {v}"))
    if len(p.settlements) < B.MAX_SETTLEMENTS:
        best = None
        for v, (d, _e) in placement.reachable_spots(state, me, max_roads=2).items():
            s2 = state.copy()
            s2.players[me].settlements.append(v)
            val = (_accel.static_values(s2)[me] - base) / (4.0 + 2.0 * d)
            if best is None or val > best[0]:
                best = (val, v, d)
        if best is not None:
            out.append(("settlement", best[0], f"vertex {best[1]}, {best[2]} road(s) away"))
    if sum(state.dev_deck) > 0:
        d_la = ctx.marginal(state, LA)
        val = (ctx.p_k * (0.25 + d_la) + ctx.p_vp * 10.0 + ctx.p_rb * 0.9 + ctx.p_yop * 0.7 + ctx.p_mono * 0.9) / 3.0
        out.append(("dev card", val, f"knight {ctx.p_k:.0%}, VP {ctx.p_vp:.0%}"))
    if len(p.roads) < B.MAX_ROADS:
        out.append(("road-for-length", ctx.marginal(state, LR) / 2.0, "Longest Road credit per card"))
    out.sort(key=lambda t: -t[1])
    return out


def race_lines(state: GameState, me: int, belief=None) -> List[str]:
    """Advice lines for the "Win paths" section (root only, about a millisecond)."""
    if state.phase == PHASE_GAME_OVER:
        return []
    if state.phase in _SETUP:
        return ["(setup phase: the races start after the placements)"]
    ctx = PathsContext(state, me, weight=1.0, crowd=1.0, spots=True, priors=True, belief=belief)
    n = ctx.n
    leaf = ctx._inputs(state)
    lines = [f"Horizon ~{ctx.H:.1f} rounds (leader at {ctx.vmax:.1f} VP)."]
    crowded: List[str] = []
    race_rows = []
    for race, (b, a, G), holder, live in ((LR, leaf.lr, leaf.lr_holder, ctx.live_lr),
                                           (LA, leaf.la, leaf.la_holder, ctx.live_la)):
        sol = ctx._solve(race, b, a, G, holder)
        delta = ctx._marginal(leaf, race)
        unit = "roads" if race == LR else "knights"
        cur = leaf.L if race == LR else [leaf.played[i] for i in range(n)]
        if holder >= 0:
            head = f"held by {'you' if holder == me else _name(state, holder)} ({cur[holder]} {unit}, level {b[holder]:.1f})"
        else:
            head = f"unclaimed (needs {ctx._min_lr if race == LR else ctx._min_la})"
        rate = G[me] / ctx.H if ctx.H > 0 else 0.0
        verdict = _verdict(ctx, sol, holder, delta)
        if verdict.startswith("CROWDED"):
            crowded.append(f"{_RACE_NAMES[race]} ({sol.N_close[me]:.1f} close rivals)")
        race_rows.append((race, sol, delta))
        lines.append(f"{_RACE_NAMES[race]}: {head}; you {cur[me]} {unit} (level {b[me]:.1f}, +{rate:.2f}/round, "
                     f"projected {b[me] + G[me]:.1f}), P {sol.P[me]:.2f}, {sol.N_close[me]:.1f} close rivals"
                     f"{'' if live else ' (not live yet)'} -> {verdict}")
    # contested spots
    table = ctx._spot_table(state, leaf) if len(state.players[me].settlements) < B.MAX_SETTLEMENTS else []
    contested = [(v, d, pm, info) for v, d, pm, info in table if info]
    if not table:
        lines.append("Spots: no free settlement spot within 2 roads.")
    elif not contested:
        lines.append(f"Spots: {len(table)} spot(s) within 2 roads, none contested.")
    else:
        v, d, pm, info = max(contested, key=lambda t: ctx._spot_value(state, t[0], leaf.osig) / (1.0 + 0.9 * t[1]))
        first = min(info, key=lambda t: t[2])
        lines.append(f"Spots: best contested spot {v} ({d} road(s) away) - {_name(state, first[0])} gets there first "
                     f"(~{first[2]:.1f} rounds); your chance {pm:.0%}.")
    # VP cards
    pool_vp = ctx.pool[_VPC]
    rates = [leaf.sup[j].dev_rate for j in range(n)]
    tot_rate = sum(rates) * ctx.H
    share = min(1.0, ctx.pool_total / tot_rate) if tot_rate > 0 else 1.0
    e_vp = min(float(pool_vp), rates[me] * ctx.H * ctx.p_vp * share)
    lines.append(f"VP cards: {pool_vp} left in the pool ({ctx.p_vp:.0%} per dev card); your expected share over the "
                 f"horizon ~{e_vp:.1f} VP.")
    # cities: ore / wheat competition
    ow = [leaf.sup[j].inc[_ORE] + leaf.sup[j].inc[_WHEAT] for j in range(n)]
    tot_ow = sum(ow)
    our_share = ow[me] / tot_ow if tot_ow > 0 else 0.0
    robber_hex = state.hexes[state.robber] if 0 <= state.robber < len(state.hexes) else (B.DESERT, 0)
    on_our_ore = robber_hex[0] == _ORE and any(state.robber in B.VERTEX_HEXES[v]
                                               for v in state.players[me].settlements + state.players[me].cities)
    lines.append(f"Cities: you make {our_share:.0%} of the table's ore+wheat (fair share {1.0 / n:.0%}); bank ore "
                 f"{state.bank[_ORE]}, wheat {state.bank[_WHEAT]}" + ("; the robber sits on your ore." if on_our_ore else "."))
    # portfolio and the plan
    port = portfolio(state, me, ctx)
    if port:
        lines.append("Portfolio (points per card): " + " > ".join(f"{name} {val:.1f}" for name, val, _ in port))
        best = port[0]
        lines.insert(1, f"Your best path: {best[0]} ({best[1]:.1f} points per card, {best[2]}); crowded: "
                        + (", ".join(crowded) if crowded else "none") + ".")
    return lines


# ---------------------------------------------------------------------------
# Tunables (registered into catanbot.tuning.TUNABLES)
# ---------------------------------------------------------------------------
_ONLY = "only with search.paths=1 (base spec ...,paths=1 or --cand-set)"


def register_tunables(registry: Dict[str, object]) -> None:
    """Add the ``search.paths*`` knobs and the model constants to the ``tuning.TUNABLES`` registry."""
    tuning = sys.modules.get("catanbot.tuning") or importlib.import_module("catanbot.tuning")
    Tunable = tuning.Tunable
    for attr, default, cands, parse, desc in (
            ("paths", 0, [1], tuning._parse_int,
             "win-path races with crowding (catanbot/winpaths.py): race-aware Longest Road / Largest Army credit in "
             "the evaluator plus move-ordering nudges; 0 = off (the default bot); budgeted at depth 1"),
            ("paths_w", 1.0, [0.5, 1.5, 2.0], tuning._parse_float, "value weight g of the win-path correction; " + _ONLY),
            ("paths_crowd", 1.0, [0.0, 0.5, 2.0], tuning._parse_float,
             "crowding strength (scales the waste cost of a crowded race; 0 = probability share only); " + _ONLY),
            ("paths_priors", 1, [0], tuning._parse_int, "race-aware move-ordering nudges (0 = off); " + _ONLY),
            ("paths_spots", 0, [1], tuning._parse_int,
             "rescale static's settlement-reach terms by our chance to win contested spots; " + _ONLY)):
        t = Tunable(name=f"search.{attr}", module="catanbot.search", attr=attr, default=default, kind="search",
                    candidates=list(cands), requires_search=True, requires_depth=1, spec_key=attr, parse=parse,
                    description=desc)
        registry[t.name] = t
    g = globals()
    for attr, cands, parse, meaning in (
            ("KAPPA", [0.25, 0.75, 1.0], tuning._parse_float, "prize weight of the award races"),
            ("BETA", [0.8, 1.8], tuning._parse_float, "race softmax temperature"),
            ("H_B", [1.2, 2.8], tuning._parse_float, "horizon slope (rounds per VP the leader still needs)"),
            ("HAND_W", [0.0, 1.0], tuning._parse_float, "weight of cards in hand in the race levels"),
            ("ESC", [0.0, 1.0], tuning._parse_float, "waste escalation per extra close rival"),
            ("TIE_LR", [0.5, 2.5], tuning._parse_float, "holder tie bonus, Longest Road"),
            ("LIVE_LR", [0, 5], tuning._parse_int, "Longest Road liveness gate (trail length; 0 = always live)"),
            ("LIVE_LA", [0, 3], tuning._parse_int, "Largest Army liveness gate (knights; 0 = always live)"),
            ("PRIOR_SCALE", [2.0, 8.0], tuning._parse_float, "prior bonus per point of marginal race credit"),
            ("PLACEBO", [1], tuning._parse_int, "seat-rotated control: every seat gets the next seat's correction")):
        t = Tunable(name=f"winpaths.{attr}", module=__name__, attr=attr, default=g[attr], kind="weight",
                    candidates=list(cands), requires_search=True, needs_python_evaluator=False, clear_caches=False,
                    parse=parse, description=f"{meaning}; {_ONLY}")
        registry[t.name] = t
