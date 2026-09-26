"""Conversion cost: a port-aware diversification term for our own seat (off by default: ``search.conv``).

Area: diversification / expansion (separate from port access; linked to the Longest Road race, below).

The problem (docs/RESULTS.md, 2026-09-26 05:00): the bot plays ore/wheat cities on every board - openings of
11.4 wheat+ore pips vs 6.8 wood+brick, 3.85 distinct resources vs 4.67 for Catanatron's bots, the first extra
settlement at median round 12 vs 7-9, 75-78 % of its bank trades at 4:1.  The depth-1 search never sees the
multi-turn cost of a missing resource: the static value pays 0.4 per resource type produced and a small reach
credit for roads, while every card of a missing resource is bought later at 4:1.

The model (one term, feature id ``ports.conversion_cost``)
------------------------------------------------------------
For our seat, with robber-free production ``p_r`` (cards per roll of the dice, cities double; the robber is a
temporary block, static already charges it):

* need shares ``n_r`` = ``NEED`` normalised to sum 1 - by default ``placement.RESOURCE_DEMAND``, the build-cost
  mix of roads, settlements, cities and dev cards with a quarter extra on wheat and ore (0.187 / 0.187 / 0.168 /
  0.234 / 0.224).  Fixed shares, not the current plan: a plan-dependent mix jumps when a piece runs out and
  would give the search a reason to build (or avoid) the 5th settlement for the jump alone;
* income ``I = sum_r p_r``; shortfall ``s_r = max(0, n_r I - p_r)`` cards of ``r`` per roll that must be
  acquired; surplus ``u_r = max(0, p_r - n_r I)``;
* each acquired card costs ``rho - 1`` extra cards, ``rho`` the ratio of the route that pays for it.  Routes,
  cheapest first: a 2:1 port on a resource ``s`` carries at most ``u_s / 2`` cards per roll (it pays only with
  the surplus of ``s``); the rest goes through the generic route, 3:1 with a generic port and 4:1 at the bank,
  uncapped (the need is charged even when the surplus cannot pay it: the missing card is still missing).  So a
  missing card costs 3 / 2 / 1 extra cards at 4:1 / 3:1 / 2:1 - full, two-thirds and one-third weight;
* cards per roll ``c = sum over routes of carried x (rho - 1)`` (:func:`extra_cards`);
* horizon: ``R = n x H`` rolls left (every roll produces for us, as static's pips count every roll), with
  ``H = winpaths.horizon(root)`` rounds (clamp(0.5 + 2 (10 - vmax), 1, 16), fitted on 7.6k turn starts) frozen at
  the search root, so every node of one decision is valued at the same horizon;
* ``C_me = -KAPPA_CONV x R x (c(leaf) - c(root))`` static points, minus (``PORT_LEDGER``, default on) the change
  of static's own port credit (+0.2 per building on a 3:1 port, 0.15 + 4 x production of ``t`` on a 2:1 ``t``
  port): the conversion saving *replaces* that crude credit, so a port settlement gains exactly its trade
  savings and port access is not counted twice (docs/PRIORITY_PLAN.md: a port-value provider on the hub never
  stacks with static's port credit).  Anchoring at the root changes no ranking (a constant shift of our own
  static value never reorders softmax leaves) but keeps the correction exactly 0 on every leaf where we built
  nothing, so the hub's fast path returns the C++ values there.

``KAPPA_CONV = 0.12`` points per extra card is static's price of a held card.  Calibration: our typical
post-setup economy (wheat 6.3, ore 5.1, wood / brick 3.4, sheep 2.6 pips) has ``c = 0.156`` cards per roll at
4:1, 12.5 extra cards over an 80-roll game - the proof logs show ~15 (5 bank trades at 4:1 a game); a generic
port then saves 4.2 cards (measured ~5, docs/PRIORITY_PLAN.md acq.flow) and a 2:1 wheat port 3.2 (measured
3.6-3.8), so the model's card counts match the logs without a fitted factor.  At 0.12 a card they are worth
0.4-0.5 points (0.3-0.4 over the 64 rolls of the capped 16-round horizon), between static's 0.2 for a 3:1 port
and the 0.76-0.79 the ports design calibrated.  winpaths' ``KAPPA_F = 0.6`` (a future card spent in a race)
would make an early ore/wheat city cost more than its production credit.

Optional reach component (``REACH_W``, 0 = off): a road only changes static's reach terms, so the search sees the
conversion saving of a spot only when the settlement is built.  With ``REACH_W > 0`` a leaf also gets
``REACH_W x max_v saving(v) / (1 + 0.9 d_v)`` over the free spots ``v`` within two roads (static's own reach set
and distance discount), ``saving(v) = c(now) - c(with a settlement on v)`` including the port at ``v``, anchored at
the root like the main term.

Composition with winpaths' Longest Road credit (``paths=1,conv=1``: both are providers of one hub, summed per
leaf): the two value different things bought by the same roads.  This term is the value of *resource access* -
cards not lost to the bank, a function of our settlements, cities and ports (and, with ``REACH_W``, of the spots
our roads reach); winpaths' race credit is the value of the *prize* - P(hold the 2-VP card at game end) from
trail lengths, hand and road-building rate.  Neither reads the other's output, and their ledgers cancel disjoint
parts of static (winpaths: the award credits 0.15 / 0.35 per road length and 0.45 per knight; this term: the
port credit), so a road that extends the trail *and* heads for a new resource type earns both, once each.
They share inputs only: winpaths' road rate counts port-converted surplus (``TAU``), which is the race's
feasibility, not the saving.  Known overlap: with ``paths_spots=1`` winpaths rescales static's reach credit by
our chance to win contested spots, while the ``REACH_W`` saving is not rescaled.

Static's flat 0.4 per resource type produced stays: it is the diversity credit at every horizon, and the user's
point is that it is too small on its own, not wrong.

Own seat only: opponents cannot build during our turn and production is robber-free, so their terms would be a
constant per search whose only effect is to reweight which opponent the softmax prefers to damage - a
second-order effect of unclear sign on an opponent population (Catanatron) whose trade habits this model was
not fitted to.  The acq.progress design reached the same conclusion.

Invariants: ``SearchConfig.conv = 0`` (the default) imports and runs nothing of this module; ``static_value``,
the placement score and ``cpp/*`` are untouched; no RNG; every memo value is a pure function of its key; the
module constants are read once per :class:`ConversionContext` (``ParamBot`` apply / restore is enough).

The setup counterpart is the ``conversion`` opening policy (catanbot/openings.py), which adds the same cost change
of a candidate spot to the setup score (with the same port ledger on the spot score's port bonus).
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import board as B
from . import placement
from .state import GameState

# ---------------------------------------------------------------------------
# Constants (read once per ConversionContext; registered in tuning.py, read only with search.conv = 1)
# ---------------------------------------------------------------------------
KAPPA_CONV = 0.12                            # static points per extra card paid in conversions
REACH_W = 0.0                                # weight of the best reachable spot's conversion saving (0 = off)
NEED = list(placement.RESOURCE_DEMAND)       # need weights wood / brick / sheep / wheat / ore (normalised to shares)
PORT_LEDGER = 1                              # 1 = cancel static's own port credit for our seat (the conversion
#                                              saving replaces it); 0 = keep both (double counts port access)
CACHE_CAP = 20000                            # a memo dict is cleared when it grows beyond this

_PARAM_NAMES = ("KAPPA_CONV", "REACH_W", "NEED", "PORT_LEDGER", "CACHE_CAP")


def current_params() -> Dict[str, Any]:
    """The module constants as they are now (a :class:`ConversionContext` freezes them)."""
    g = globals()
    return {k: (list(g[k]) if isinstance(g[k], list) else g[k]) for k in _PARAM_NAMES}


# ---------------------------------------------------------------------------
# The model (pure functions)
# ---------------------------------------------------------------------------
def need_shares(need: Optional[Sequence[float]] = None) -> List[float]:
    """``need`` (default ``NEED``) normalised to sum 1."""
    w = [max(0.0, float(x)) for x in (NEED if need is None else need)]
    tot = sum(w)
    return [x / tot for x in w] if tot > 0 else [0.2] * 5


def shortfall_surplus(prod: Sequence[float], shares: Sequence[float]) -> Tuple[List[float], List[float]]:
    """``(s_r, u_r)``: cards per roll of ``r`` below / above the need share of our income."""
    inc = sum(prod)
    short = [0.0] * 5
    surplus = [0.0] * 5
    for r in range(5):
        x = shares[r] * inc - prod[r]
        if x > 0.0:
            short[r] = x
        elif x < 0.0:
            surplus[r] = -x
    return short, surplus


def extra_cards(short: Sequence[float], surplus: Sequence[float], ratios: Sequence[int]) -> float:
    """Extra cards per roll paid to acquire the shortfall: every card costs ``rho - 1`` on its route.

    Routes cheapest first: a resource ``s`` whose ratio is below the generic one (a 2:1 port) carries at most
    ``surplus[s] / rho_s`` cards; the rest pays the generic ratio ``g = max(ratios)`` (3 with a generic port,
    else 4), uncapped.  So with enough surplus on a 2:1 port the cost is 1/3 of the bank's, with a 3:1 port 2/3.
    """
    need = sum(short)
    if need <= 0.0:
        return 0.0
    g = max(ratios)
    cost = 0.0
    for rho, s in sorted((ratios[s], s) for s in range(5) if ratios[s] < g and surplus[s] > 0.0):
        carried = min(need, surplus[s] / rho)
        cost += carried * (rho - 1)
        need -= carried
        if need <= 0.0:
            return cost
    return cost + need * (g - 1)


def economy(state: GameState, settlements: Sequence[int], cities: Sequence[int]) -> Tuple[List[float], List[int]]:
    """Robber-free production per roll and bank ratios (per resource given) of a set of buildings."""
    prod = [0.0] * 5
    ratios = [4] * 5
    generic = False
    for mult, verts in ((1.0, settlements), (2.0, cities)):
        for v in verts:
            pr = placement.vertex_production(state, v, ignore_robber=True)
            for r in range(5):
                prod[r] += mult * pr[r]
            t = state.ports.get(v)
            if t is None:
                continue
            if t == B.PORT_GENERIC:
                generic = True
            else:
                ratios[t] = 2
    if generic:
        ratios = [min(x, 3) for x in ratios]
    return prod, ratios


def cost_per_roll(state: GameState, settlements: Sequence[int], cities: Sequence[int],
                  shares: Sequence[float]) -> float:
    """``c``: extra cards per roll the buildings' economy pays in conversions."""
    prod, ratios = economy(state, settlements, cities)
    short, surplus = shortfall_surplus(prod, shares)
    return extra_cards(short, surplus, ratios)


def static_port_credit(state: GameState, settlements: Sequence[int], cities: Sequence[int]) -> float:
    """``heuristic.static_value``'s port term for these buildings: +0.2 per building on a 3:1 port, 0.15 + 4 x the
    robber-free production of ``t`` per building on a 2:1 ``t`` port (what :data:`PORT_LEDGER` cancels) - with the
    ports area's switches (``heuristic.static_port_term``: the port constants, ``placement.PORT_MODEL``) whatever
    static's port term currently is, so the saving always replaces exactly static's credit (one owner)."""
    from .heuristic import static_port_term   # heuristic imports the strategy modules; this one stays light
    prod, _ratios = economy(state, settlements, cities)
    return static_port_term(state, list(settlements) + list(cities), prod)


def rolls_left(state: GameState) -> float:
    """``R = n x H``: dice rolls left in the game (``winpaths.horizon`` rounds of ``n`` rolls)."""
    from . import winpaths   # lazy: only with search.conv = 1 or the conversion opening policy
    return state.num_players * winpaths.horizon(state)


def spot_delta(state: GameState, player: int, v: int, shares: Optional[Sequence[float]] = None) -> float:
    """Change of ``c`` (cards per roll) if ``player`` adds a settlement on ``v`` (its production and port)."""
    p = state.players[player]
    shares = need_shares() if shares is None else shares
    s0, cs = sorted(p.settlements), sorted(p.cities)
    return cost_per_roll(state, sorted(s0 + [v]), cs, shares) - cost_per_roll(state, s0, cs, shares)


def breakdown(state: GameState, player: int, shares: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Every quantity of the model for ``player``'s current buildings (tests, diagnostics, the advisor)."""
    p = state.players[player]
    shares = need_shares() if shares is None else shares
    prod, ratios = economy(state, sorted(p.settlements), sorted(p.cities))
    short, surplus = shortfall_surplus(prod, shares)
    c = extra_cards(short, surplus, ratios)
    R = rolls_left(state)
    return {"prod": prod, "income": sum(prod), "shares": list(shares), "short": short, "surplus": surplus,
            "ratios": ratios, "cards_per_roll": c, "rolls_left": R, "cards": c * R,
            "points": KAPPA_CONV * c * R,
            "static_port_credit": static_port_credit(state, sorted(p.settlements), sorted(p.cities))}


# ---------------------------------------------------------------------------
# Per-decision context (a CorrectionHub provider)
# ---------------------------------------------------------------------------
class ConversionContext:
    """Our seat's conversion-cost correction for one ``Searcher.search`` call (see the module docstring)."""

    priors = False     # no move-ordering hook

    def __init__(self, root: GameState, me: int, params: Optional[Dict[str, Any]] = None):
        P = dict(params) if params is not None else current_params()
        self.params = P
        self.me = int(me)
        self.n = root.num_players
        self.kappa = float(P["KAPPA_CONV"])
        self.reach_w = float(P["REACH_W"])
        self.ledger = bool(P["PORT_LEDGER"])
        self.shares = need_shares(P["NEED"])
        self._cap = int(P["CACHE_CAP"])
        self.R = rolls_left(root)
        self.scale = self.kappa * self.R
        self.stats: Dict[str, Any] = {"evals": 0, "nonzero": 0, "cost_misses": 0, "reach_misses": 0}
        self._cost: Dict[tuple, float] = {}
        self._port: Dict[tuple, float] = {}
        self._reach: Dict[tuple, float] = {}
        self.v0 = self.value(root)

    def _put(self, d: dict, key, value):
        if len(d) >= self._cap:
            d.clear()
        d[key] = value
        return value

    def _cost_of(self, state: GameState, settlements: tuple, cities: tuple) -> float:
        key = (settlements, cities)
        hit = self._cost.get(key)
        if hit is not None:
            return hit
        self.stats["cost_misses"] += 1
        return self._put(self._cost, key, cost_per_roll(state, settlements, cities, self.shares))

    def cost(self, state: GameState) -> float:
        """``c`` of our seat in ``state`` (memo: our buildings; production is robber-free)."""
        p = state.players[self.me]
        return self._cost_of(state, tuple(sorted(p.settlements)), tuple(sorted(p.cities)))

    def port_credit(self, state: GameState) -> float:
        """Static's port credit of our seat (memo: our buildings)."""
        p = state.players[self.me]
        key = (tuple(sorted(p.settlements)), tuple(sorted(p.cities)))
        hit = self._port.get(key)
        if hit is None:
            hit = self._put(self._port, key, static_port_credit(state, key[0], key[1]))
        return hit

    def value(self, state: GameState) -> float:
        """Our seat's term before anchoring, static points: ``-scale c - [ledger] port credit + REACH_W scale
        reach``.  The correction is its change since the root."""
        v = -self.scale * self.cost(state)
        if self.ledger:
            v -= self.port_credit(state)
        if self.reach_w:
            v += self.reach_w * self.scale * self.reach_saving(state)
        return v

    def reach_saving(self, state: GameState) -> float:
        """``max(0, max_v (c(now) - c(with v)) / (1 + 0.9 d_v))`` over our free spots within two roads."""
        me = self.me
        p = state.players[me]
        if len(p.settlements) >= B.MAX_SETTLEMENTS:
            return 0.0
        osig = tuple(tuple(sorted(q.settlements + q.cities)) for q in state.players)
        rsig = tuple(tuple(q.roads) for q in state.players)
        key = (osig, rsig)
        hit = self._reach.get(key)
        if hit is not None:
            return hit
        self.stats["reach_misses"] += 1
        s0, cs = tuple(sorted(p.settlements)), tuple(sorted(p.cities))
        base = self._cost_of(state, s0, cs)
        best = 0.0
        occ = state.occupied_vertices()
        for v, (d, _e) in placement.reachable_spots(state, me, max_roads=2, occ=occ).items():
            saving = (base - self._cost_of(state, tuple(sorted(s0 + (v,))), cs)) / (1.0 + 0.9 * d)
            if saving > best:
                best = saving
        return self._put(self._reach, key, best)

    def corrections(self, state: GameState) -> Optional[List[float]]:
        """``[0, ..., value(state) - value(root), ..., 0]`` (our seat only); ``None`` when nothing changed since
        the root (no building of ours, and with ``REACH_W`` no road either)."""
        self.stats["evals"] += 1
        d = self.value(state) - self.v0
        if d == 0.0:
            return None
        self.stats["nonzero"] += 1
        out = [0.0] * self.n
        out[self.me] = d
        return out


# ---------------------------------------------------------------------------
# Self-play mechanism metrics (seat observer "economy", tuning.SEAT_OBSERVERS)
# ---------------------------------------------------------------------------
class EconomyObserver:
    """Per seat: bank trades at 4:1 / 3:1 / 2:1, settlements and cities built after setup, the turn of the first
    of each (-1 = never) and the distinct resource types produced at the seat's first roll."""

    def __init__(self, num_seats: int):
        n = int(num_seats)
        self.bank = [[0, 0, 0] for _ in range(n)]        # 4:1, 3:1, 2:1
        self.settlements = [0] * n
        self.cities = [0] * n
        self.first_settlement = [-1] * n
        self.first_city = [-1] * n
        self.types = [-1] * n

    def on_action(self, state: GameState, action, player: int) -> None:
        from . import actions as A
        k = action[0]
        if k == A.BANK_TRADE:
            self.bank[player][4 - state.port_ratio(player, action[1])] += 1
        elif k == A.BUILD_SETTLEMENT:
            self.settlements[player] += 1
            if self.first_settlement[player] < 0:
                self.first_settlement[player] = state.turn
        elif k == A.BUILD_CITY:
            self.cities[player] += 1
            if self.first_city[player] < 0:
                self.first_city[player] = state.turn
        elif k == A.ROLL and self.types[player] < 0:
            prod = placement.player_production(state, player, ignore_robber=True)
            self.types[player] = sum(1 for x in prod if x > 0)

    def result(self) -> Dict[str, List[int]]:
        return {"bank_4": [b[0] for b in self.bank], "bank_3": [b[1] for b in self.bank],
                "bank_2": [b[2] for b in self.bank], "settlements": list(self.settlements),
                "cities": list(self.cities), "first_settlement_turn": list(self.first_settlement),
                "first_city_turn": list(self.first_city), "types_after_setup": list(self.types)}


# ---------------------------------------------------------------------------
# Tunable registration (called from the last lines of catanbot/tuning.py)
# ---------------------------------------------------------------------------
_ONLY = "only with search.conv=1 (base spec ...,conv=1) or the conversion opening policy"


def register_tunables(registry: Dict[str, object]) -> None:
    """Add ``search.conv`` and the model constants to the ``tuning.TUNABLES`` registry."""
    tuning = sys.modules.get("catanbot.tuning") or importlib.import_module("catanbot.tuning")
    Tunable = tuning.Tunable
    t = Tunable(name="search.conv", module="catanbot.search", attr="conv", default=0, kind="search", candidates=[1],
                requires_search=True, requires_depth=1, spec_key="conv", parse=tuning._parse_int,
                description="conversion cost (catanbot/conversion.py): our missing resources cost 3 / 2 / 1 extra "
                            "cards each at 4:1 / 3:1 / 2:1 over the rolls left, as a leaf correction through the "
                            "hub; 0 = off (the default bot); budgeted at depth 1")
    registry[t.name] = t
    g = globals()
    for attr, cands, parse, meaning in (
            ("KAPPA_CONV", [0.06, 0.25, 0.5], tuning._parse_float, "static points per extra conversion card"),
            ("REACH_W", [0.12, 0.3], tuning._parse_float,
             "weight of the best reachable spot's conversion saving (road credit; 0 = off)"),
            ("NEED", [[1.0] * 5], tuning._parse_vector,
             "need weights wood/brick/sheep/wheat/ore (normalised to shares; default placement.RESOURCE_DEMAND)"),
            ("PORT_LEDGER", [0], tuning._parse_int,
             "1 = the conversion saving replaces static's port credit (and the setup spot score's port bonus) for "
             "our seat; 0 = both count")):
        default = list(g[attr]) if isinstance(g[attr], list) else g[attr]    # a list target is mutated in place
        t = Tunable(name=f"conversion.{attr}", module=__name__, attr=attr, default=default, kind="weight",
                    candidates=list(cands), requires_search=True, needs_python_evaluator=False, clear_caches=False,
                    parse=parse, description=f"{meaning}; {_ONLY}")
        registry[t.name] = t
    tuning.register_seat_observer("economy", EconomyObserver)     # self-play mechanism metrics (observers=...)
