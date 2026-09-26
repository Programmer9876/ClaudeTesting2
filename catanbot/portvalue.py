"""Port access: is a port spot worth it, and what are the ports we own worth (docs/PRIORITY_PLAN.md step 4, area
"ports"; the design is ``docs/designs/priority_areas_2026-09-26.json``, area "ports", as revised after both
critiques).  Everything here is off by default.

Area boundary: *diversification / expansion* (which resources we produce, how fast we expand, and what a missing
resource costs at the bank or a port) is catanbot/conversion.py; this module is *port access*: which port spots are
worth a weaker land spot, the roads and the blocking race to reach them, and the value of the ports we own.

``ports.flow_provider`` (``ports.FLOW_KAPPA``, a static-points weight, default 0 = off)
    A :class:`corrections.CorrectionHub` provider (C++ evaluator: a Python correction on the C++ static values, no
    ``needs_python_evaluator``).  For our own seat it *replaces* static's port credit (``heuristic.static_port_term``)
    by the trade-flow value of the ports we own, from the trades step's fitted table (``acquisition.acq.flow``):

        C_me = FLOW_KAPPA x (F(leaf) - F(root)) - (credit(leaf) - credit(root))
        F    = sum_r R(t) w_r (4 - rho_r)      cards our ports save over the rest of the game vs 4:1 everywhere

    ``R(t)`` = ``acquisition.flow_remaining`` (bank trades our bot still makes after ``t`` post-setup turns, ``t``
    frozen at the search root so every node of one decision has one horizon), ``w_r`` =
    ``acquisition.flow_weights`` (the share of those trades giving ``r``, from our robber-free production shares at
    the leaf) and ``rho`` our ratios at the leaf.  For one new port with unchanged production ``F`` rises by exactly
    ``acquisition.port_flow_value`` (tested); a second 3:1 adds nothing (the ratio vector is joint).  Main phase only
    (the hub passes setup states through): the setup spot score keeps static's constants.

    **One owner for our seat's port value.**  ``conv=1`` (catanbot/conversion.py) already credits ports for our seat
    through the conversion saving and cancels static's port credit (``conversion.PORT_LEDGER``).  With both on, the
    conversion term owns it and the flow provider stands down (it is not built: :meth:`CorrectionHub.for_search`
    records ``port_owner = "conv"``); otherwise static's credit would be cancelled twice and the port value counted
    twice.  The flow provider is never stacked with ``placement.PORT_MODEL`` 1 / 2 in a screened arm either (the
    plan's rule); if it were, it would cancel the calibrated static term like any other static port credit.

``ports.advice`` (``portvalue.ADVICE`` / the CLI's ``--port-advice``; display only)
    :func:`advice_lines`: for the port spots in play, "is this port worth it" - the cards the port saves over the
    rest of the game, the extra cards the best land spot would produce instead, the roads each needs, a verdict, and
    the race: our chance to get there first, which rivals reach it, in how many rounds, how much they want it
    (:func:`rival_want`) and whether someone leads (:func:`leader`).  Computed on the CLI advisor path only, never in
    ``Searcher.explain`` (which runs for every root action at every decision): bots never call it.

``ports.spot_want`` helper (F4, display now)
    :func:`rival_want` / :func:`will_block`: "will they block" by what the rival needs (their own spot score of the
    spot relative to their best reachable spot, distance-discounted) and the simple leader check.  Its use inside
    winpaths' spot race (``winpaths.SPOT_WANT``) is step 7 and waits for winpaths Stage 5.

The calibrated port model of ``ports.surplus_value`` (``placement.PORT_MODEL``) and the registered port constants
of ``ports.constants`` live in catanbot/placement.py and catanbot/heuristic.py (they are C++-mirrored); their
tunables register here.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import board as B
from . import placement as P
from .state import GameState, PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT

# ---------------------------------------------------------------------------
# Constants (registered in tuning.py through register_tunables)
# ---------------------------------------------------------------------------
FLOW_KAPPA = 0.0      # ports.flow_provider: static points per card our ports save over the game (0 = off)
ADVICE = 0            # ports.advice: 1 = the CLI advisor prints the port lines (display only)
WANT_FLOOR = 0.3      # F4 helper: a rival's want of a spot is at least this (a second choice is still taken)
LEADER_DENY = 0.6     # F4 helper: rivals block a leader at least this willingly (SPOT_LEADER mode 1)
LEADER_MARGIN = 1.0   # estimated VP ahead of every other seat that makes a leader
CACHE_CAP = 20000     # a memo dict is cleared when it grows beyond this

_PARAM_NAMES = ("FLOW_KAPPA", "CACHE_CAP")


def current_params() -> Dict[str, Any]:
    g = globals()
    return {k: g[k] for k in _PARAM_NAMES}


def flow_on() -> bool:
    """The flow provider is switched on (``FLOW_KAPPA != 0``)."""
    return bool(FLOW_KAPPA)


# ---------------------------------------------------------------------------
# Trade-flow value of owned ports (acq.flow)
# ---------------------------------------------------------------------------
def economy(state: GameState, settlements: Sequence[int], cities: Sequence[int]) -> Tuple[List[float], List[int]]:
    """Robber-free production per roll (cities double) and bank ratios of a set of buildings."""
    prod = [0.0] * 5
    for mult, verts in ((1.0, settlements), (2.0, cities)):
        for v in verts:
            pr = P.vertex_production(state, v, ignore_robber=True)
            for r in range(5):
                prod[r] += mult * pr[r]
    return prod, P.ratios_of(state, list(settlements) + list(cities))


def flow_cards(prod: Sequence[float], ratios: Sequence[int], city_open: bool, R: float) -> float:
    """``F = sum_r R w_r (4 - rho_r)``: cards the ports behind ``ratios`` save over ``R`` bank trades."""
    if R <= 0.0 or min(ratios) >= 4:
        return 0.0
    from .acquisition import flow_weights
    w = flow_weights(prod, city_open)
    return sum(R * w[r] * (4 - ratios[r]) for r in range(5))


def owned_flow_cards(state: GameState, i: int, t: Optional[float] = None) -> float:
    """Cards seat ``i``'s ports save over the rest of the game (``t`` post-setup turns; default: the state's)."""
    from .acquisition import flow_remaining, post_setup_turn
    p = state.players[i]
    prod, rho = economy(state, p.settlements, p.cities)
    t = post_setup_turn(state) if t is None else float(t)
    return flow_cards(prod, rho, bool(p.settlements) and len(p.cities) < B.MAX_CITIES, flow_remaining(t))


class PortFlowContext:
    """Our seat's port-flow correction for one ``Searcher.search`` call (a :class:`corrections.CorrectionHub`
    provider; see the module docstring)."""

    priors = False     # no move-ordering hook

    def __init__(self, root: GameState, me: int, params: Optional[Dict[str, Any]] = None):
        from .acquisition import flow_remaining, post_setup_turn
        Pm = dict(params) if params is not None else current_params()
        self.params = Pm
        self.me = int(me)
        self.n = root.num_players
        self.kappa = float(Pm["FLOW_KAPPA"])
        self._cap = int(Pm["CACHE_CAP"])
        self.t = post_setup_turn(root)
        self.R = flow_remaining(self.t)
        self.stats: Dict[str, Any] = {"evals": 0, "nonzero": 0, "misses": 0}
        self._memo: Dict[tuple, Tuple[float, float]] = {}
        self.v0 = self.value(root)

    def parts(self, state: GameState) -> Tuple[float, float]:
        """``(F cards, static's port credit)`` of our seat (memo: our buildings; production is robber-free)."""
        from .heuristic import static_port_term
        p = state.players[self.me]
        key = (tuple(sorted(p.settlements)), tuple(sorted(p.cities)))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        self.stats["misses"] += 1
        prod, rho = economy(state, key[0], key[1])
        cards = flow_cards(prod, rho, bool(key[0]) and len(key[1]) < B.MAX_CITIES, self.R)
        credit = static_port_term(state, list(key[0]) + list(key[1]), prod)
        if len(self._memo) >= self._cap:
            self._memo.clear()
        self._memo[key] = (cards, credit)
        return cards, credit

    def value(self, state: GameState) -> float:
        cards, credit = self.parts(state)
        return self.kappa * cards - credit

    def corrections(self, state: GameState) -> Optional[List[float]]:
        """``[0, ..., value(state) - value(root), ..., 0]`` (our seat); ``None`` when nothing changed."""
        self.stats["evals"] += 1
        d = self.value(state) - self.v0
        if d == 0.0:
            return None
        self.stats["nonzero"] += 1
        out = [0.0] * self.n
        out[self.me] = d
        return out


# ---------------------------------------------------------------------------
# F4 helper: will they block (rival need) and the leader check
# ---------------------------------------------------------------------------
def _vp(state: GameState, i: int) -> float:
    from .winpaths import _vp_estimate
    return _vp_estimate(state, i)


def leader(state: GameState, margin: Optional[float] = None) -> Optional[Tuple[int, float]]:
    """``(seat, lead)`` of the seat whose estimated VP (``winpaths._vp_estimate``: public VP + expected hidden VP
    cards) is at least ``margin`` (``LEADER_MARGIN``) above every other seat, else None."""
    m = LEADER_MARGIN if margin is None else margin
    vps = [_vp(state, i) for i in range(state.num_players)]
    top = max(range(len(vps)), key=lambda i: vps[i])
    rest = max(v for i, v in enumerate(vps) if i != top)
    lead = vps[top] - rest
    return (top, lead) if lead >= m else None


def rival_want(state: GameState, j: int, v: int, floor: Optional[float] = None,
               reach: Optional[Dict[int, Tuple[int, int]]] = None) -> Optional[float]:
    """``W_j(v) = clamp((score_j(v) / (1 + 0.9 d_j)) / best_j, floor, 1)``: how much seat ``j`` wants spot ``v``
    relative to its best spot within two roads (``score_j`` = ``winpaths._spot_score``, the static spot score from
    j's view: its production, missing types and port synergy - "what they need").  None when ``j`` cannot reach
    ``v`` within two roads."""
    from .winpaths import _spot_score
    fl = WANT_FLOOR if floor is None else floor
    reach = P.reachable_spots(state, j, max_roads=2) if reach is None else reach
    if v not in reach:
        return None
    best = 0.0
    mine = 0.0
    for x, (d, _e) in reach.items():
        s = _spot_score(state, j, x) / (1.0 + 0.9 * d)
        if s > best:
            best = s
        if x == v:
            mine = s
    if best <= 0.0:
        return 1.0
    return min(1.0, max(fl, mine / best))


def will_block(state: GameState, me: int, j: int, want: float, mode: int = 2,
               lead: Optional[Tuple[int, float]] = None) -> Tuple[float, str]:
    """``(W, why)`` after the leader check (the design's SPOT_LEADER): mode 1 - when we lead, every rival blocks us
    at least ``LEADER_DENY`` willingly; mode 2 - also, a leading rival races for anything (W = 1)."""
    if mode >= 1 and lead is not None and lead[0] == me and want < LEADER_DENY:
        return LEADER_DENY, "you lead"
    if mode >= 2 and lead is not None and lead[0] == j:
        return 1.0, "they lead"
    return want, ""


# ---------------------------------------------------------------------------
# F3: "is this port worth it" advice (CLI advisor path only)
# ---------------------------------------------------------------------------
def road_cards() -> float:
    """Demand-weighted cards of one road (a wood and a brick)."""
    return P.RESOURCE_DEMAND[B.WOOD] + P.RESOURCE_DEMAND[B.BRICK]


def pe_pips(state: GameState, v: int, scarcity: Sequence[float]) -> float:
    """``sum_r 36 prod_v[r] RESOURCE_DEMAND[r] sqrt(scarcity[r])``: demand-weighted pips of a settlement on ``v``."""
    pr = P.vertex_production(state, v, ignore_robber=True)
    return sum(36.0 * pr[r] * P.RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5) for r in range(5))


def _port_name(t: int) -> str:
    return "3:1" if t == B.PORT_GENERIC else f"2:1 {B.RESOURCE_NAMES[t]}"


def _seat_name(state: GameState, i: int) -> str:
    p = state.players[i]
    return p.name or p.color


def _candidate_spots(state: GameState, me: int) -> Dict[int, int]:
    """``{vertex: roads needed}``: every free vertex in the setup, else the spots within two roads of ours."""
    occ = state.occupied_vertices()
    if state.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
        return {v: 0 for v in range(B.NUM_VERTICES) if P.is_free_vertex(occ, v)}
    return {v: d for v, (d, _e) in P.reachable_spots(state, me, max_roads=2, occ=occ).items()}


def port_advice(state: GameState, me: int, v: int, spots: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """The numbers of the port line for ``me`` settling the port vertex ``v``:

    * ``G`` = ``placement.port_gain(own + prod_v; rho_now -> rho_with_v)`` (the F1 conversion model, whatever
      ``PORT_MODEL`` is), ``w`` its need weight, ``H = winpaths.horizon`` rounds, ``n`` players;
    * ``saved = n H G w`` needed-card equivalents over the rest of the game;
    * the land alternative ``u`` = the best non-port spot among ``spots`` (every free vertex in the setup, else our
      spots within two roads) by demand-weighted pips net of its roads;
    * ``extra = n H (PEpips(u) - PEpips(v)) / 36`` cards the land spot would produce instead, ``roads`` =
      ``road_cards() x (d_v - d_u)`` the extra road cards the port costs;
    * verdict: the port is worth it when ``saved > extra + roads``."""
    from .winpaths import horizon
    port = state.ports[v]
    spots = _candidate_spots(state, me) if spots is None else spots
    sc = P.resource_scarcity(state)
    own = P.player_production(state, me, ignore_robber=True)
    pv = P.vertex_production(state, v, ignore_robber=True)
    prod = [own[r] + pv[r] for r in range(5)]
    rho = P.port_ratios(state, me)
    rho2 = P.ratios_with(rho, port)
    G = P.port_gain(prod, rho, rho2)
    w, _c = P.port_need_weights(prod, sc)
    H = horizon(state)
    n = state.num_players
    rolls = n * H
    saved = rolls * G * w
    rc = road_cards()
    d_v = spots.get(v, 0)
    best = None
    for u, d in spots.items():
        if u == v or u in state.ports:
            continue
        cards = rolls * pe_pips(state, u, sc) / 36.0 - rc * d
        if best is None or cards > best[1] or (cards == best[1] and u < best[0]):
            best = (u, cards, d)
    out: Dict[str, Any] = {"vertex": v, "port": port, "port_name": _port_name(port), "G": G, "w": w, "H": H,
                           "rolls": rolls, "saved": saved, "roads": d_v, "ratios": rho, "ratios_with": rho2}
    if best is None:
        out.update({"land": None, "extra": 0.0, "road_cost": 0.0, "worth": saved > 0.0})
        return out
    u, _cards, d_u = best
    extra = rolls * (pe_pips(state, u, sc) - pe_pips(state, v, sc)) / 36.0
    road_cost = rc * (d_v - d_u)
    out.update({"land": u, "land_roads": d_u, "extra": extra, "road_cost": road_cost,
                "worth": saved > extra + road_cost})
    return out


def race_advice(state: GameState, me: int, v: int, ctx=None, mode: int = 2) -> Optional[Dict[str, Any]]:
    """The race for ``v`` (within two roads of ours): ``P_me`` from ``winpaths.PathsContext._spot_table`` (the
    conversion-aware clocks with the turn-order offsets) and per rival ``(seat, roads d_j, rounds T_j + o_j, want
    W_j, W after the leader check, why)``; None when ``v`` is not one of our spots within two roads."""
    from .winpaths import PathsContext
    ctx = PathsContext(state, me) if ctx is None else ctx
    leaf = ctx._inputs(state)
    row = next((r for r in ctx._spot_table(state, leaf) if r[0] == v), None)
    if row is None:
        return None
    _v, d, p_me, info = row
    lead = leader(state)
    rivals = []
    for j, dj, tj in info:
        want = rival_want(state, j, v)
        want = 1.0 if want is None else want
        eff, why = will_block(state, me, j, want, mode=mode, lead=lead)
        rivals.append({"seat": j, "name": _seat_name(state, j), "roads": dj, "rounds": tj, "want": want,
                       "will": eff, "why": why})
    return {"vertex": v, "roads": d, "p_me": p_me, "rivals": rivals, "leader": lead}


def _fmt_port_line(state: GameState, a: Dict[str, Any]) -> str:
    v = a["vertex"]
    roads = f"{a['roads']} road{'s' if a['roads'] != 1 else ''}"
    head = (f"{a['port_name']} port at {v} ({roads}): its better rates buy ~{a['saved']:.1f} more cards over "
            f"~{a['H']:.0f} rounds")
    if a["land"] is None:
        return head + ("; no land spot to compare -> port worth it" if a["worth"] else "")
    lr = a["land_roads"]
    land = f"best land spot {a['land']} ({lr} road{'s' if lr != 1 else ''}) produces ~{a['extra']:+.1f} cards more"
    roads_txt = f", the port's extra roads cost ~{a['road_cost']:.1f}" if a["road_cost"] > 0 else \
        (f", the land spot's extra roads cost ~{-a['road_cost']:.1f}" if a["road_cost"] < 0 else "")
    verdict = "port worth it" if a["worth"] else "take the land spot"
    return f"{head}; {land}{roads_txt} -> {verdict}"


def _fmt_race_line(state: GameState, me: int, r: Dict[str, Any]) -> str:
    lead = r["leader"]
    lead_txt = ""
    if lead is not None:
        lead_txt = f"; {'you lead' if lead[0] == me else _seat_name(state, lead[0]) + ' leads'} by {lead[1]:.1f} VP"
    if not r["rivals"]:
        return f"  race for {r['vertex']}: no rival reaches it within 2 roads (ours to take){lead_txt}"
    parts = []
    for x in r["rivals"]:
        will = f"wants it {x['want']:.1f}"
        if x["why"]:
            will += f" -> {x['will']:.1f} ({x['why']})"
        parts.append(f"{x['name']} {x['roads']} road{'s' if x['roads'] != 1 else ''}, affords it in "
                     f"~{x['rounds']:.1f} rounds, {will}")
    return f"  race for {r['vertex']}: we get there first ~{r['p_me']:.2f} ({'; '.join(parts)}){lead_txt}"


def advice_lines(state: GameState, me: int, actions: Sequence = (), k: int = 3) -> List[str]:
    """The advisor's 'Ports' lines: the port spots in play for ``me`` (those among the recommended ``actions``
    first, then the best of our port spots within two roads - or, in the setup, the best free port spots by our
    spot score), one worth-it line each plus the race line in the main phase.  Display only."""
    spots = _candidate_spots(state, me)
    ports = [v for v in spots if v in state.ports]
    if not ports:
        return []
    chosen: List[int] = []
    for a in actions:
        a = tuple(a) if not isinstance(a, tuple) else a
        if len(a) >= 2 and a[0] in ("setup_settlement", "build_settlement") and a[1] in state.ports \
                and a[1] not in chosen:
            chosen.append(int(a[1]))
    setup = state.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)
    if setup:
        sc = P.resource_scarcity(state)
        own = P.player_production(state, me, ignore_robber=True)
        ranked = sorted(ports, key=lambda v: (-P.score_settlement_spot(state, me, v, own_prod=own, scarcity=sc,
                                                                        setup=True), v))
    else:
        ranked = sorted(ports, key=lambda v: (spots[v], v))
    for v in ranked:
        if len(chosen) >= k:
            break
        if v not in chosen:
            chosen.append(v)
    lines: List[str] = []
    ctx = None
    for v in chosen[:k]:
        a = port_advice(state, me, v, spots)
        lines.append(_fmt_port_line(state, a))
        if not setup:
            try:
                if ctx is None:
                    from .winpaths import PathsContext
                    ctx = PathsContext(state, me)
                r = race_advice(state, me, v, ctx=ctx)
                if r is not None:
                    lines.append(_fmt_race_line(state, me, r))
            except Exception as ex:  # pragma: no cover - the race line is optional (design fallback)
                lines.append(f"  (race unavailable: {ex})")
    return lines


# ---------------------------------------------------------------------------
# Tunable registration (called from the last lines of catanbot/tuning.py)
# ---------------------------------------------------------------------------
_PYEVAL = ("C++-mirrored: cpp/heuristic.cpp keeps a constexpr copy of the default, so it needs the Python evaluator; "
           "a C++ change only after an ADOPT")
_ONLY_F1 = "only with placement.PORT_MODEL > 0"


def register_tunables(registry: Dict[str, object]) -> None:
    """Add the ports area's switches (docs/PRIORITY_PLAN.md step 4) to the ``tuning.TUNABLES`` registry."""
    tuning = sys.modules.get("catanbot.tuning") or importlib.import_module("catanbot.tuning")
    Tunable = tuning.Tunable
    from . import heuristic as Hm
    pl, hm = P.__name__, Hm.__name__

    def add(name, module, attr, cands, parse, desc, pyeval=True, search=False, kind="weight"):
        mod = sys.modules[module]
        t = Tunable(name=name, module=module, attr=attr, default=getattr(mod, attr), kind=kind,
                    candidates=list(cands), needs_python_evaluator=pyeval, requires_search=search,
                    clear_caches=False, parse=parse, description=desc)
        registry[t.name] = t

    # ports.constants (F2): today's literals, plus the count-3:1-once switch
    add("placement.PORT_SPOT_GENERIC", pl, "PORT_SPOT_GENERIC", [1.25, 0.0, 3.0], tuning._parse_float,
        "spot-score bonus of a 3:1 port spot (pips-equivalents); " + _PYEVAL)
    add("placement.PORT_SPOT_2TO1_BASE", pl, "PORT_SPOT_2TO1_BASE", [0.25, 0.0, 1.5], tuning._parse_float,
        "spot-score bonus of a 2:1 port spot: base; " + _PYEVAL)
    add("placement.PORT_SPOT_2TO1_SLOPE", pl, "PORT_SPOT_2TO1_SLOPE", [3.0, 0.0, 18.0], tuning._parse_float,
        "spot-score bonus of a 2:1 t port spot: slope per card per roll of t (ours + the spot's); " + _PYEVAL)
    add("heuristic.PORT_STATIC_GENERIC", hm, "PORT_STATIC_GENERIC", [0.8, 0.0], tuning._parse_float,
        "static_value per building on a 3:1 port (P3' candidate 0.8 with placement.PORT_GENERIC_ONCE=1); " + _PYEVAL)
    add("heuristic.PORT_STATIC_2TO1_BASE", hm, "PORT_STATIC_2TO1_BASE", [0.075, 0.0, 0.45], tuning._parse_float,
        "static_value per building on a 2:1 port: base; " + _PYEVAL)
    add("heuristic.PORT_STATIC_2TO1_SLOPE", hm, "PORT_STATIC_2TO1_SLOPE", [2.0, 0.0, 12.0], tuning._parse_float,
        "static_value per building on a 2:1 t port: slope per card per roll of t; " + _PYEVAL)
    add("placement.PORT_GENERIC_ONCE", pl, "PORT_GENERIC_ONCE", [1], tuning._parse_int,
        "1 = a 3:1 port counts once: static adds PORT_STATIC_GENERIC once, a 3:1 spot is worth 0 when we already own "
        "one; 0 = per building / per spot (today); " + _PYEVAL)
    # ports.surplus_value (F1): the calibrated conversion model
    add("placement.PORT_MODEL", pl, "PORT_MODEL", [1, 2, 3], tuning._parse_int,
        "port value from the calibrated conversion model (catanbot/placement.py port_gain): 1 = spot score and "
        "static_value, 2 = static only, 3 = 3:1 only (2:1 constants kept); 0 = the constants (today); " + _PYEVAL)
    for attr, cands, meaning in (
            ("PORT_A0", [0.03, 0.12], "cards per roll converted at 4:1 whatever the surplus (the fitted intercept)"),
            ("PORT_B", [0.6, 0.9], "share of the surplus converted at 4:1"),
            ("PORT_ELAST", [0.0, 0.5], "credit for the extra conversion at a cheaper ratio"),
            ("PORT_SPOT_W", [0.5, 1.5], "weight of the calibrated spot bonus"),
            ("PORT_STATIC_W", [0.4, 1.2], "static points per pips-equivalent of the calibrated port term")):
        add(f"placement.{attr}", pl, attr, cands, tuning._parse_float, f"F1 {meaning}; {_ONLY_F1}; {_PYEVAL}")
    # ports.flow_provider (a hub provider on the C++ evaluator) and ports.advice (display only)
    add("ports.FLOW_KAPPA", __name__, "FLOW_KAPPA", [0.18, 0.12, 0.36], tuning._parse_float,
        "ports.flow_provider (catanbot/portvalue.py): static points per card our ports save over the rest of the "
        "game (acquisition.port_flow_value's fitted table); replaces static's port credit for our seat through the "
        "correction hub; stands down when search.conv=1 owns our port value; 0 = off (the default bot); budgeted at "
        "depth 1", pyeval=False, search=True)
    add("portvalue.ADVICE", __name__, "ADVICE", [1], tuning._parse_int,
        "ports.advice: 1 = the CLI advisor prints 'is this port worth it' lines (cards saved vs the land spot, roads, "
        "the race, who wants it, the leader); display only, bots never read it", pyeval=False)
