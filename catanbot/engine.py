"""Rules engine: legal actions, action application, dice, awards.

This module implements the contract in ``docs/DESIGN.md`` section 3 for the
base game (3-4 players, Colonist.io defaults).  It is a *perfect information*
simulator: every card is known.  It is deterministic given a
``random.Random`` and never mutates its input in :func:`apply` (a faster
:func:`apply_inplace` is exposed for the self-play runner).

Design notes
------------
* Actions are the tuples defined in :mod:`catanbot.actions`.
* :func:`legal_actions` returns the complete list of legal actions for
  :func:`acting_player`.  Two documented exceptions to "complete":
  ``(ROLL, value)`` (forced roll, used by search) is accepted by
  :func:`apply` but only ``(ROLL,)`` is listed; and the ``DISCARD``
  enumeration is capped at :data:`DISCARD_ENUM_CAP` entries (see
  :func:`discard_options`).  ``PROPOSE_TRADE`` candidates are a bounded set
  (1-for-1 and 2-for-1) but :func:`apply` accepts any offer that satisfies the
  rules (non-empty, disjoint, affordable, at most
  :data:`MAX_TRADE_PROPOSALS_PER_TURN` per turn).
* While ``state.free_roads > 0`` (Road Building) the only legal actions in
  ``PHASE_MAIN`` are the free ``BUILD_ROAD`` placements and ``END_TURN``
  (which forfeits the remaining free roads).  Free roads are forfeited
  eagerly whenever no legal road exists.
* Occupancy is recomputed per call as two flat lists (vertex owner, edge
  owner); all action tuples that do not depend on the state are precomputed
  at import time so the hot paths allocate very little.
* Longest Road is recomputed after every road build (only the builder's and
  the holder's trails, see :func:`_update_longest_road_after_road`) and after
  a settlement that touches a road of an opponent who owns >= 5 roads (the
  only way a settlement can shorten a trail).
* Every rule violation raises :class:`IllegalActionError` (a ``ValueError``)
  and leaves the state untouched.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from .state import (
    PHASE_DISCARD,
    PHASE_GAME_OVER,
    PHASE_MAIN,
    PHASE_ROBBER,
    PHASE_ROLL,
    PHASE_SETUP_ROAD,
    PHASE_SETUP_SETTLEMENT,
    PHASE_TRADE_RESPONSE,
    PHASE_TRADE_SELECT,
    GameState,
    Player,
    TradeOffer,
)

__all__ = [
    "MAX_TRADE_PROPOSALS_PER_TURN",
    "DISCARD_ENUM_CAP",
    "IllegalActionError",
    "acting_player",
    "legal_actions",
    "apply",
    "apply_inplace",
    "is_terminal",
    "roll_outcomes",
    "apply_roll",
    "production_for_roll",
    "longest_road_length",
    "count_vp",
    "random_playout",
    "buildable_settlement_vertices",
    "buildable_road_edges",
    "buildable_city_vertices",
    "discard_options",
    "robber_victims",
]

Action = A.Action

#: Maximum number of ``PROPOSE_TRADE`` actions per turn.
MAX_TRADE_PROPOSALS_PER_TURN = 4
#: Maximum number of ``DISCARD`` actions enumerated by :func:`legal_actions`.
DISCARD_ENUM_CAP = 200


class IllegalActionError(ValueError):
    """Raised by :func:`apply` / :func:`apply_inplace` for an illegal action."""


# ---------------------------------------------------------------------------
# Precomputed tables (no per-call allocation for the common action tuples)
# ---------------------------------------------------------------------------
_NV = B.NUM_VERTICES
_NE = B.NUM_EDGES
_NH = B.NUM_HEXES
_EDGE_VERTICES = B.EDGE_VERTICES
_VERTEX_EDGES = B.VERTEX_EDGES
_VERTEX_NEIGHBORS = B.VERTEX_NEIGHBORS
_HEX_VERTICES = B.HEX_VERTICES
_VERTEX_HEXES = B.VERTEX_HEXES

_ROLL_ACTION: Action = (A.ROLL,)
_END_TURN_ACTION: Action = (A.END_TURN,)
_BUY_DEV_ACTION: Action = (A.BUY_DEV,)
_PLAY_ROAD_BUILDING_ACTION: Action = (A.PLAY_ROAD_BUILDING,)
_ACCEPT_ACTION: Action = (A.ACCEPT_TRADE,)
_REJECT_ACTION: Action = (A.REJECT_TRADE,)
_CANCEL_ACTION: Action = (A.CANCEL_TRADE,)

_SETUP_SETTLEMENT_ACTIONS: List[Action] = [(A.SETUP_SETTLEMENT, v) for v in range(_NV)]
_SETUP_ROAD_ACTIONS: List[Action] = [(A.SETUP_ROAD, e) for e in range(_NE)]
_BUILD_ROAD_ACTIONS: List[Action] = [(A.BUILD_ROAD, e) for e in range(_NE)]
_BUILD_SETTLEMENT_ACTIONS: List[Action] = [(A.BUILD_SETTLEMENT, v) for v in range(_NV)]
_BUILD_CITY_ACTIONS: List[Action] = [(A.BUILD_CITY, v) for v in range(_NV)]
_EXECUTE_TRADE_ACTIONS: List[Action] = [(A.EXECUTE_TRADE, i) for i in range(8)]
_MONOPOLY_ACTIONS: List[Action] = [(A.PLAY_MONOPOLY, r) for r in range(5)]
_YOP_ACTIONS: List[Action] = [(A.PLAY_YEAR_OF_PLENTY, r1, r2) for r1 in range(5) for r2 in range(r1, 5)]
_BANK_TRADE_ACTIONS: List[List[Action]] = [[(A.BANK_TRADE, g, t) for t in range(5)] for g in range(5)]
# robber tables: [hex][victim + 1] (index 0 = nobody), up to 8 players
_MOVE_ROBBER_ACTIONS: List[List[Action]] = [
    [(A.MOVE_ROBBER, h, i - 1) for i in range(9)] for h in range(_NH)
]
_PLAY_KNIGHT_ACTIONS: List[List[Action]] = [
    [(A.PLAY_KNIGHT, h, i - 1) for i in range(9)] for h in range(_NH)
]
_UNIT: List[Tuple[int, ...]] = [tuple(1 if i == r else 0 for i in range(5)) for r in range(5)]
_DOUBLE: List[Tuple[int, ...]] = [tuple(2 if i == r else 0 for i in range(5)) for r in range(5)]
# proposals: [give][amount-1][get]
_PROPOSE_ACTIONS: List[List[List[Action]]] = [
    [[(A.PROPOSE_TRADE, vec[g], _UNIT[t]) for t in range(5)] for vec in (_UNIT, _DOUBLE)]
    for g in range(5)
]
_ROLL_OUTCOMES: List[Tuple[float, int]] = [(B.ROLL_PROB[v], v) for v in range(2, 13)]

# Costs as (resource, amount) pairs for quick payment loops.
_COST_ROAD_ITEMS = tuple((i, c) for i, c in enumerate(B.COST_ROAD) if c)
_COST_SETTLEMENT_ITEMS = tuple((i, c) for i, c in enumerate(B.COST_SETTLEMENT) if c)
_COST_CITY_ITEMS = tuple((i, c) for i, c in enumerate(B.COST_CITY) if c)
_COST_DEV_ITEMS = tuple((i, c) for i, c in enumerate(B.COST_DEV) if c)

_SETUP_PHASES = (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _vertex_owners(state: GameState) -> List[int]:
    """Flat list vertex -> owner index (or -1)."""
    vo = [-1] * _NV
    for i, p in enumerate(state.players):
        for v in p.settlements:
            vo[v] = i
        for v in p.cities:
            vo[v] = i
    return vo


def _vertex_owners_weights(state: GameState) -> Tuple[List[int], List[int]]:
    """Vertex -> owner, vertex -> production weight (1 settlement, 2 city)."""
    vo = [-1] * _NV
    vw = [0] * _NV
    for i, p in enumerate(state.players):
        for v in p.settlements:
            vo[v] = i
            vw[v] = 1
        for v in p.cities:
            vo[v] = i
            vw[v] = 2
    return vo, vw


def _edge_owners(state: GameState) -> List[int]:
    """Flat list edge -> owner index (or -1)."""
    eo = [-1] * _NE
    for i, p in enumerate(state.players):
        for e in p.roads:
            eo[e] = i
    return eo


def _sync(p: Player) -> None:
    """Recompute ``hand_size`` / ``dev_count`` from the exact lists."""
    p.hand_size = sum(p.resources)
    p.dev_count = sum(p.dev_cards) + sum(p.dev_cards_new)


def _pay(p: Player, bank: List[int], items: Tuple[Tuple[int, int], ...]) -> None:
    res = p.resources
    for i, c in items:
        res[i] -= c
        bank[i] += c


def _afford(res: Sequence[int], items: Tuple[Tuple[int, int], ...]) -> bool:
    for i, c in items:
        if res[i] < c:
            return False
    return True


def _can_pay(p: Player, counts: Sequence[int]) -> bool:
    r = p.resources
    return r[0] >= counts[0] and r[1] >= counts[1] and r[2] >= counts[2] and r[3] >= counts[3] and r[4] >= counts[4]


def _steal(thief: Player, victim: Player, rng: random.Random) -> bool:
    """Move one uniformly random card from ``victim`` to ``thief``."""
    vr = victim.resources
    total = vr[0] + vr[1] + vr[2] + vr[3] + vr[4]
    if total <= 0:
        return False
    idx = rng.randrange(total)
    for r in range(5):
        c = vr[r]
        if idx < c:
            vr[r] -= 1
            thief.resources[r] += 1
            return True
        idx -= c
    return False  # pragma: no cover


def _port_ratios(state: GameState, i: int) -> List[int]:
    """Bank ratio for each resource the player could give (4, 3 or 2)."""
    ratios = [4, 4, 4, 4, 4]
    ports = state.ports
    if not ports:
        return ratios
    p = state.players[i]
    generic = False
    for v in p.settlements:
        t = ports.get(v)
        if t is not None:
            if t == B.PORT_GENERIC:
                generic = True
            else:
                ratios[t] = 2
    for v in p.cities:
        t = ports.get(v)
        if t is not None:
            if t == B.PORT_GENERIC:
                generic = True
            else:
                ratios[t] = 2
    if generic:
        for r in range(5):
            if ratios[r] > 3:
                ratios[r] = 3
    return ratios


def _rng(rng: Optional[random.Random]) -> random.Random:
    return rng if rng is not None else random.Random()


# ---------------------------------------------------------------------------
# Board-legality helpers (shared by legal_actions, apply and other modules)
# ---------------------------------------------------------------------------
def _settlement_vertices(state: GameState, me: int, vo: List[int]) -> List[int]:
    """Vertices where ``me`` may build a settlement now (distance rule + own road)."""
    p = state.players[me]
    seen = bytearray(_NV)
    out: List[int] = []
    for e in p.roads:
        for v in _EDGE_VERTICES[e]:
            if seen[v]:
                continue
            seen[v] = 1
            if vo[v] != -1:
                continue
            for w in _VERTEX_NEIGHBORS[v]:
                if vo[w] != -1:
                    break
            else:
                out.append(v)
    out.sort()
    return out


def _free_vertices(vo: List[int]) -> List[int]:
    """Vertices satisfying the distance rule (setup placement)."""
    out: List[int] = []
    for v in range(_NV):
        if vo[v] != -1:
            continue
        for w in _VERTEX_NEIGHBORS[v]:
            if vo[w] != -1:
                break
        else:
            out.append(v)
    return out


def _road_edges(state: GameState, me: int, vo: List[int], eo: List[int]) -> List[int]:
    """Unoccupied edges connected to an own road / building (opponent buildings cut)."""
    p = state.players[me]
    mark = bytearray(_NE)
    out: List[int] = []
    for e in p.roads:
        for v in _EDGE_VERTICES[e]:
            o = vo[v]
            if o == -1 or o == me:
                for f in _VERTEX_EDGES[v]:
                    if eo[f] == -1 and not mark[f]:
                        mark[f] = 1
                        out.append(f)
    for v in p.settlements:
        for f in _VERTEX_EDGES[v]:
            if eo[f] == -1 and not mark[f]:
                mark[f] = 1
                out.append(f)
    for v in p.cities:
        for f in _VERTEX_EDGES[v]:
            if eo[f] == -1 and not mark[f]:
                mark[f] = 1
                out.append(f)
    out.sort()
    return out


def _road_connected(state: GameState, me: int, e: int, vo: List[int]) -> bool:
    """Edge ``e`` touches an own building or an own road at a non-opponent vertex."""
    p = state.players[me]
    roads = p.roads
    for v in _EDGE_VERTICES[e]:
        o = vo[v]
        if o == me:
            return True
        if o == -1:
            for f in _VERTEX_EDGES[v]:
                if f != e and f in roads:
                    return True
    return False


def robber_victims(state: GameState, hex_id: int, me: int, vo: Optional[List[int]] = None) -> List[int]:
    """Opponents with a building on ``hex_id`` and at least one resource card (sorted)."""
    if vo is None:
        vo = _vertex_owners(state)
    players = state.players
    mask = 0
    for v in _HEX_VERTICES[hex_id]:
        o = vo[v]
        if o >= 0 and o != me and not (mask >> o) & 1:
            r = players[o].resources
            if r[0] or r[1] or r[2] or r[3] or r[4]:
                mask |= 1 << o
    return [i for i in range(len(players)) if (mask >> i) & 1]


def _robber_actions(state: GameState, me: int, table: List[List[Action]], vo: List[int]) -> List[Action]:
    players = state.players
    n = len(players)
    has_cards = [bool(p.resources[0] or p.resources[1] or p.resources[2] or p.resources[3] or p.resources[4])
                 for p in players]
    robber = state.robber
    out: List[Action] = []
    for h in range(_NH):
        if h == robber:
            continue
        mask = 0
        for v in _HEX_VERTICES[h]:
            o = vo[v]
            if o >= 0 and o != me and has_cards[o]:
                mask |= 1 << o
        row = table[h]
        if mask:
            for o in range(n):
                if (mask >> o) & 1:
                    out.append(row[o + 1])
        else:
            out.append(row[0])
    return out


def discard_options(resources: Sequence[int], k: int, cap: int = DISCARD_ENUM_CAP) -> List[Tuple[int, ...]]:
    """All distinct 5-vectors of counts summing to ``k`` with ``counts[i] <= resources[i]``.

    The enumeration is ordered so that combinations discarding more of the
    most-held resources come first (lexicographically descending along the
    resources sorted by held count), and it stops after ``cap`` entries.
    With the default cap of :data:`DISCARD_ENUM_CAP` (200) the list is
    complete for every hand of at most 16 cards (worst case 186 vectors);
    from 17 cards on (worst case 218, growing to 780 at 25 cards) only the
    first ``cap`` vectors, i.e. those dumping the most-held resources, are
    returned.  :func:`apply` still accepts *any* valid discard vector.
    """
    order = sorted(range(5), key=lambda i: (-resources[i], i))
    caps = [resources[i] for i in order]
    suffix = [0] * 6
    for j in range(4, -1, -1):
        suffix[j] = suffix[j + 1] + caps[j]
    if k < 0 or suffix[0] < k:
        return []
    out: List[Tuple[int, ...]] = []
    cur = [0] * 5

    def rec(j: int, rem: int) -> None:
        if j == 4:
            cur[4] = rem
            counts = [0] * 5
            for jj in range(5):
                counts[order[jj]] = cur[jj]
            out.append(tuple(counts))
            return
        hi = caps[j] if caps[j] < rem else rem
        lo = rem - suffix[j + 1]
        if lo < 0:
            lo = 0
        for x in range(hi, lo - 1, -1):
            cur[j] = x
            rec(j + 1, rem - x)
            if len(out) >= cap:
                return

    rec(0, k)
    return out


# ---------------------------------------------------------------------------
# Public helpers used by other modules
# ---------------------------------------------------------------------------
def buildable_settlement_vertices(state: GameState, player: int, setup: bool = False) -> List[int]:
    """Vertices where ``player`` could place a settlement (board legality only).

    Ignores resources, phase and piece limits.  With ``setup=True`` the
    connecting-road requirement is dropped (initial placement).
    """
    vo = _vertex_owners(state)
    if setup:
        return _free_vertices(vo)
    return _settlement_vertices(state, player, vo)


def buildable_road_edges(state: GameState, player: int) -> List[int]:
    """Edges where ``player`` could place a road (board legality only: free and connected)."""
    return _road_edges(state, player, _vertex_owners(state), _edge_owners(state))


def buildable_city_vertices(state: GameState, player: int) -> List[int]:
    """Vertices holding ``player``'s settlements (city upgrade candidates)."""
    return sorted(state.players[player].settlements)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def acting_player(state: GameState) -> int:
    """Index of the player who must act now."""
    phase = state.phase
    if phase == PHASE_DISCARD:
        return state.discard_queue[0] if state.discard_queue else state.current
    if phase == PHASE_TRADE_RESPONSE:
        return state.trade_responder
    return state.current


def is_terminal(state: GameState) -> bool:
    """True once the game is over."""
    return state.phase == PHASE_GAME_OVER


def roll_outcomes(state: GameState) -> List[Tuple[float, int]]:
    """``[(probability, total)]`` for the eleven 2d6 totals (2..12)."""
    return list(_ROLL_OUTCOMES)


def count_vp(state: GameState, player: int, include_hidden: bool = True) -> int:
    """Victory points of ``player`` (public + VP dev cards if ``include_hidden``)."""
    p = state.players[player]
    vp = len(p.settlements) + 2 * len(p.cities)
    if state.longest_road_owner == player:
        vp += 2
    if state.largest_army_owner == player:
        vp += 2
    if include_hidden:
        vp += p.dev_cards[B.DEV_VP] + p.dev_cards_new[B.DEV_VP]
    return vp


def production_for_roll(state: GameState, value: int) -> List[List[int]]:
    """Per-player resource gains for a roll of ``value`` (robber and bank shortage applied).

    If the bank cannot pay everyone for a resource: a single owed player gets
    what is left, otherwise nobody receives that resource.
    """
    players = state.players
    n = len(players)
    gains = [[0, 0, 0, 0, 0] for _ in range(n)]
    if value == 7:
        return gains
    robber = state.robber
    vo = vw = None
    owed = [0, 0, 0, 0, 0]
    for h, (res, num) in enumerate(state.hexes):
        if num != value or h == robber or res == B.DESERT:
            continue
        if vo is None:
            vo, vw = _vertex_owners_weights(state)
        for v in _HEX_VERTICES[h]:
            o = vo[v]
            if o >= 0:
                amt = vw[v]
                gains[o][res] += amt
                owed[res] += amt
    bank = state.bank
    for r in range(5):
        if owed[r] > bank[r]:
            recipients = [i for i in range(n) if gains[i][r]]
            if len(recipients) == 1:
                gains[recipients[0]][r] = bank[r]
            else:
                for i in recipients:
                    gains[i][r] = 0
    return gains


def longest_road_length(state: GameState, player: int) -> int:
    """Length of the longest trail over ``player``'s roads.

    Roads may not be reused; vertices may be revisited (official rule).  A
    vertex holding an opponent's building ends a path (the road leading into it
    still counts, the path cannot continue through it).
    """
    roads = state.players[player].roads
    if not roads:
        return 0
    blocked = bytearray(_NV)
    for i, q in enumerate(state.players):
        if i != player:
            for v in q.settlements:
                blocked[v] = 1
            for v in q.cities:
                blocked[v] = 1
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for k, e in enumerate(roads):
        a, b = _EDGE_VERTICES[e]
        bit = 1 << k
        adj.setdefault(a, []).append((bit, b))
        adj.setdefault(b, []).append((bit, a))

    def dfs(v: int, used: int) -> int:
        best = 0
        for bit, w in adj[v]:
            if used & bit:
                continue
            if blocked[w]:
                length = 1
            else:
                length = 1 + dfs(w, used | bit)
            if length > best:
                best = length
        return best

    total = len(roads)
    best = 0
    for v in adj:
        length = dfs(v, 0)
        if length > best:
            best = length
            if best >= total:
                break
    return best


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------
def legal_actions(state: GameState) -> List[Action]:
    """Complete list of legal actions for :func:`acting_player` (empty when over)."""
    phase = state.phase
    if phase == PHASE_MAIN:
        return _legal_main(state)
    if phase == PHASE_ROLL:
        return _legal_roll(state)
    if phase == PHASE_SETUP_SETTLEMENT:
        vo = _vertex_owners(state)
        return [_SETUP_SETTLEMENT_ACTIONS[v] for v in _free_vertices(vo)]
    if phase == PHASE_SETUP_ROAD:
        last = state.setup_last_settlement
        if last < 0:
            return []
        eo = _edge_owners(state)
        return [_SETUP_ROAD_ACTIONS[e] for e in _VERTEX_EDGES[last] if eo[e] == -1]
    if phase == PHASE_DISCARD:
        if not state.discard_queue:
            return []
        p = state.players[state.discard_queue[0]]
        k = sum(p.resources) // 2
        return [(A.DISCARD, c) for c in discard_options(p.resources, k)]
    if phase == PHASE_ROBBER:
        return _robber_actions(state, state.current, _MOVE_ROBBER_ACTIONS, _vertex_owners(state))
    if phase == PHASE_TRADE_RESPONSE:
        offer = state.pending_trade
        if offer is None or state.trade_responder < 0:
            return []
        if _can_pay(state.players[state.trade_responder], offer.get):
            return [_ACCEPT_ACTION, _REJECT_ACTION]
        return [_REJECT_ACTION]
    if phase == PHASE_TRADE_SELECT:
        offer = state.pending_trade
        out: List[Action] = []
        if offer is not None:
            proposer = state.players[offer.proposer]
            for i in sorted(offer.responses):
                if offer.responses[i] and _can_pay(state.players[i], offer.get) and _can_pay(proposer, offer.give):
                    out.append(_EXECUTE_TRADE_ACTIONS[i])
        out.append(_CANCEL_ACTION)
        return out
    return []


def _legal_roll(state: GameState) -> List[Action]:
    out: List[Action] = [_ROLL_ACTION]
    p = state.players[state.current]
    if not state.dev_played_this_turn and p.dev_cards[B.DEV_KNIGHT] > 0:
        out.extend(_robber_actions(state, state.current, _PLAY_KNIGHT_ACTIONS, _vertex_owners(state)))
    return out


def _legal_main(state: GameState) -> List[Action]:
    cur = state.current
    players = state.players
    p = players[cur]
    res = p.resources
    bank = state.bank
    roads_left = len(p.roads) < B.MAX_ROADS
    vo: Optional[List[int]] = None

    if state.free_roads > 0 and roads_left:
        vo = _vertex_owners(state)
        edges = _road_edges(state, cur, vo, _edge_owners(state))
        if edges:
            out = [_BUILD_ROAD_ACTIONS[e] for e in edges]
            out.append(_END_TURN_ACTION)
            return out

    out = []
    want_road = roads_left and res[B.WOOD] >= 1 and res[B.BRICK] >= 1
    want_settlement = (len(p.settlements) < B.MAX_SETTLEMENTS and res[B.WOOD] >= 1 and res[B.BRICK] >= 1
                       and res[B.SHEEP] >= 1 and res[B.WHEAT] >= 1)
    knight = not state.dev_played_this_turn and p.dev_cards[B.DEV_KNIGHT] > 0
    if want_road or want_settlement or knight:
        vo = _vertex_owners(state)
    if want_road:
        for e in _road_edges(state, cur, vo, _edge_owners(state)):  # type: ignore[arg-type]
            out.append(_BUILD_ROAD_ACTIONS[e])
    if want_settlement:
        for v in _settlement_vertices(state, cur, vo):  # type: ignore[arg-type]
            out.append(_BUILD_SETTLEMENT_ACTIONS[v])
    if len(p.cities) < B.MAX_CITIES and res[B.WHEAT] >= 2 and res[B.ORE] >= 3:
        for v in sorted(p.settlements):
            out.append(_BUILD_CITY_ACTIONS[v])
    if res[B.SHEEP] >= 1 and res[B.WHEAT] >= 1 and res[B.ORE] >= 1 and sum(state.dev_deck) > 0:
        out.append(_BUY_DEV_ACTION)
    if not state.dev_played_this_turn:
        dc = p.dev_cards
        if knight:
            out.extend(_robber_actions(state, cur, _PLAY_KNIGHT_ACTIONS, vo))  # type: ignore[arg-type]
        if dc[B.DEV_ROAD_BUILDING] > 0:
            out.append(_PLAY_ROAD_BUILDING_ACTION)
        if dc[B.DEV_YEAR_OF_PLENTY] > 0:
            for act in _YOP_ACTIONS:
                r1, r2 = act[1], act[2]
                if r1 == r2:
                    if bank[r1] >= 2:
                        out.append(act)
                elif bank[r1] >= 1 and bank[r2] >= 1:
                    out.append(act)
        if dc[B.DEV_MONOPOLY] > 0:
            out.extend(_MONOPOLY_ACTIONS)
    # bank / port trades
    if res[0] or res[1] or res[2] or res[3] or res[4]:
        ratios = _port_ratios(state, cur)
        for give in range(5):
            if res[give] >= ratios[give]:
                row = _BANK_TRADE_ACTIONS[give]
                for get in range(5):
                    if get != give and bank[get] >= 1:
                        out.append(row[get])
        # player trade proposals (bounded candidate set)
        if state.trades_this_turn < MAX_TRADE_PROPOSALS_PER_TURN:
            opp_has = [False, False, False, False, False]
            for i, q in enumerate(players):
                if i != cur:
                    qr = q.resources
                    for r in range(5):
                        if qr[r]:
                            opp_has[r] = True
            for give in range(5):
                n = res[give]
                if n >= 1:
                    row1 = _PROPOSE_ACTIONS[give][0]
                    row2 = _PROPOSE_ACTIONS[give][1] if n >= 2 else None
                    for get in range(5):
                        if get != give and opp_has[get]:
                            out.append(row1[get])
                            if row2 is not None:
                                out.append(row2[get])
    out.append(_END_TURN_ACTION)
    return out


# ---------------------------------------------------------------------------
# Awards
# ---------------------------------------------------------------------------
def _update_longest_road(state: GameState) -> None:
    """Recompute Longest Road ownership after a road build / settlement placement."""
    players = state.players
    n = len(players)
    lengths = [longest_road_length(state, i) if len(players[i].roads) >= 5 else 0 for i in range(n)]
    mx = max(lengths)
    holder = state.longest_road_owner
    if holder >= 0:
        hl = lengths[holder]
        if hl >= 5 and hl >= mx:
            state.longest_road_len = hl
            return
    if mx >= 5:
        cands = [i for i in range(n) if lengths[i] == mx]
        new = cands[0] if len(cands) == 1 else -1
    else:
        new = -1
    state.longest_road_owner = new
    state.longest_road_len = lengths[new] if new >= 0 else 0


def _update_longest_road_after_road(state: GameState, cur: int) -> None:
    """Cheaper Longest Road update when ``cur`` has just added a road.

    Only the builder's length can have changed (and only upwards).  While a
    holder exists the engine maintains the invariant "holder length >= every
    other length", so a builder who strictly exceeds the holder is the unique
    new maximum; the holder's length is re-derived rather than trusted so that
    externally constructed states (screenshots, JSON) are handled sensibly
    too: the shortcut is only taken while the recorded holder's trail is a
    valid award (>= 5), otherwise the full recompute decides, so the card is
    never awarded to, or left with, a trail shorter than 5.  Without a holder
    ties matter, hence the full recompute.
    """
    if len(state.players[cur].roads) < 5:
        return
    length = longest_road_length(state, cur)
    holder = state.longest_road_owner
    if holder == cur:
        if length >= 5:
            # adding a road never shortens the builder's trail: re-derive the length
            state.longest_road_len = length
            return
    elif holder >= 0:
        hl = longest_road_length(state, holder)
        if hl >= 5:
            if length > hl:
                state.longest_road_owner = cur
                state.longest_road_len = length
            else:
                state.longest_road_len = hl
            return
    elif length < 5:
        return
    _update_longest_road(state)


def _update_largest_army(state: GameState, cur: int) -> None:
    p = state.players[cur]
    if p.played_knights < 3:
        return
    la = state.largest_army_owner
    if la == -1 or (la != cur and p.played_knights > state.players[la].played_knights):
        state.largest_army_owner = cur


# ---------------------------------------------------------------------------
# Action handlers.  Each validates, mutates and returns the touched players.
# ---------------------------------------------------------------------------
def _fail(msg: str) -> None:
    raise IllegalActionError(msg)


def _h_setup_settlement(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_SETUP_SETTLEMENT or len(a) != 2:
        _fail("setup settlement not allowed now")
    v = a[1]
    if not (0 <= v < _NV):
        _fail("bad vertex")
    vo = _vertex_owners(s)
    if vo[v] != -1:
        _fail("vertex occupied")
    for w in _VERTEX_NEIGHBORS[v]:
        if vo[w] != -1:
            _fail("distance rule")
    cur = s.current
    p = s.players[cur]
    p.settlements.append(v)
    s.setup_last_settlement = v
    s.phase = PHASE_SETUP_ROAD
    if s.setup_round == 1:
        bank = s.bank
        for h in _VERTEX_HEXES[v]:
            res = s.hexes[h][0]
            if res != B.DESERT and bank[res] > 0:
                bank[res] -= 1
                p.resources[res] += 1
        return (cur,)
    return ()


def _h_setup_road(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_SETUP_ROAD or len(a) != 2:
        _fail("setup road not allowed now")
    e = a[1]
    last = s.setup_last_settlement
    if last < 0 or not (0 <= e < _NE) or e not in _VERTEX_EDGES[last]:
        _fail("setup road must touch the settlement just placed")
    for q in s.players:
        if e in q.roads:
            _fail("edge occupied")
    s.players[s.current].roads.append(e)
    s.setup_last_settlement = -1
    s.turn += 1
    n = len(s.players)
    if s.setup_round == 0:
        if s.current == n - 1:
            s.setup_round = 1
        else:
            s.current += 1
        s.phase = PHASE_SETUP_SETTLEMENT
    elif s.current == 0:
        s.phase = PHASE_ROLL
    else:
        s.current -= 1
        s.phase = PHASE_SETUP_SETTLEMENT
    return ()


def _h_roll(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_ROLL:
        _fail("cannot roll now")
    if len(a) == 1:
        r = _rng(rng)
        value = r.randint(1, 6) + r.randint(1, 6)
    elif len(a) == 2 and 2 <= a[1] <= 12:
        value = a[1]
    else:
        _fail("bad roll action")
    s.dice = value
    s.rolls_history_len += 1
    players = s.players
    n = len(players)
    if value == 7:
        cur = s.current
        queue = [i for i in ((cur + k) % n for k in range(n)) if sum(players[i].resources) > 7]
        s.discard_queue = queue
        s.phase = PHASE_DISCARD if queue else PHASE_ROBBER
        return ()
    gains = production_for_roll(s, value)
    bank = s.bank
    touched = []
    for i in range(n):
        g = gains[i]
        if g[0] or g[1] or g[2] or g[3] or g[4]:
            r = players[i].resources
            for k in range(5):
                c = g[k]
                if c:
                    r[k] += c
                    bank[k] -= c
            touched.append(i)
    s.phase = PHASE_MAIN
    return tuple(touched)


def _h_discard(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_DISCARD or not s.discard_queue or len(a) != 2:
        _fail("no discard pending")
    counts = a[1]
    if len(counts) != 5:
        _fail("discard needs 5 counts")
    i = s.discard_queue[0]
    p = s.players[i]
    res = p.resources
    total = 0
    for r in range(5):
        c = counts[r]
        if c < 0 or c > res[r]:
            _fail("cannot discard more than held")
        total += c
    if total != sum(res) // 2:
        _fail("must discard half the hand (rounded down)")
    bank = s.bank
    for r in range(5):
        c = counts[r]
        if c:
            res[r] -= c
            bank[r] += c
    s.discard_queue.pop(0)
    if not s.discard_queue:
        s.phase = PHASE_ROBBER
    return (i,)


def _move_robber(s: GameState, h: int, victim: int, me: int, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if not (0 <= h < _NH) or h == s.robber:
        _fail("robber must move to a different hex")
    victims = robber_victims(s, h, me)
    if victim == -1:
        if victims:
            _fail("must steal from an adjacent opponent")
    elif victim not in victims:
        _fail("victim is not an adjacent opponent with cards")
    s.robber = h
    if victim >= 0:
        _steal(s.players[me], s.players[victim], _rng(rng))
        return (me, victim)
    return ()


def _h_move_robber(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_ROBBER or len(a) != 3:
        _fail("cannot move the robber now")
    touched = _move_robber(s, a[1], a[2], s.current, rng)
    s.phase = PHASE_MAIN
    return touched


def _require_main(s: GameState) -> None:
    """Common precondition for main-phase actions other than ``BUILD_ROAD``.

    Pending free roads (Road Building) must be placed before anything else
    while a legal placement exists.  Stale free roads without a legal
    placement are simply ignored (they are forfeited eagerly by the road
    handlers anyway), so this check never mutates the state.
    """
    if s.phase != PHASE_MAIN:
        _fail("action only legal in the main phase")
    if s.free_roads > 0:
        if len(s.players[s.current].roads) < B.MAX_ROADS and _road_edges(
            s, s.current, _vertex_owners(s), _edge_owners(s)
        ):
            _fail("free roads from Road Building must be placed first")


def _h_build_road(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_MAIN or len(a) != 2:
        _fail("cannot build a road now")
    e = a[1]
    if not (0 <= e < _NE):
        _fail("bad edge")
    cur = s.current
    p = s.players[cur]
    if len(p.roads) >= B.MAX_ROADS:
        _fail("no road pieces left")
    free = s.free_roads > 0
    if not free and not _afford(p.resources, _COST_ROAD_ITEMS):
        _fail("cannot afford a road")
    for q in s.players:
        if e in q.roads:
            _fail("edge occupied")
    vo = _vertex_owners(s)
    if not _road_connected(s, cur, e, vo):
        _fail("road must connect to an own road or building")
    touched: Tuple[int, ...] = ()
    if free:
        s.free_roads -= 1
    else:
        _pay(p, s.bank, _COST_ROAD_ITEMS)
        touched = (cur,)
    p.roads.append(e)
    if s.free_roads > 0:
        if len(p.roads) >= B.MAX_ROADS or not _road_edges(s, cur, vo, _edge_owners(s)):
            s.free_roads = 0
    _update_longest_road_after_road(s, cur)
    return touched


def _h_build_settlement(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 2:
        _fail("bad action")
    _require_main(s)
    v = a[1]
    if not (0 <= v < _NV):
        _fail("bad vertex")
    cur = s.current
    p = s.players[cur]
    if len(p.settlements) >= B.MAX_SETTLEMENTS:
        _fail("no settlement pieces left")
    if not _afford(p.resources, _COST_SETTLEMENT_ITEMS):
        _fail("cannot afford a settlement")
    vo = _vertex_owners(s)
    if vo[v] != -1:
        _fail("vertex occupied")
    for w in _VERTEX_NEIGHBORS[v]:
        if vo[w] != -1:
            _fail("distance rule")
    roads = p.roads
    for f in _VERTEX_EDGES[v]:
        if f in roads:
            break
    else:
        _fail("settlement must touch an own road")
    _pay(p, s.bank, _COST_SETTLEMENT_ITEMS)
    p.settlements.append(v)
    # The new building may cut an opponent's road: recompute if any opponent
    # with a relevant road network touches this vertex.
    for i, q in enumerate(s.players):
        if i != cur and len(q.roads) >= 5:
            qr = q.roads
            for f in _VERTEX_EDGES[v]:
                if f in qr:
                    _update_longest_road(s)
                    return (cur,)
    return (cur,)


def _h_build_city(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 2:
        _fail("bad action")
    _require_main(s)
    v = a[1]
    cur = s.current
    p = s.players[cur]
    if v not in p.settlements:
        _fail("city must replace an own settlement")
    if len(p.cities) >= B.MAX_CITIES:
        _fail("no city pieces left")
    if not _afford(p.resources, _COST_CITY_ITEMS):
        _fail("cannot afford a city")
    _pay(p, s.bank, _COST_CITY_ITEMS)
    p.settlements.remove(v)
    p.cities.append(v)
    return (cur,)


def _h_buy_dev(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 1:
        _fail("bad action")
    _require_main(s)
    cur = s.current
    p = s.players[cur]
    if not _afford(p.resources, _COST_DEV_ITEMS):
        _fail("cannot afford a development card")
    deck = s.dev_deck
    total = deck[0] + deck[1] + deck[2] + deck[3] + deck[4]
    if total <= 0:
        _fail("development deck is empty")
    idx = _rng(rng).randrange(total)
    for t in range(5):
        if idx < deck[t]:
            deck[t] -= 1
            p.dev_cards_new[t] += 1
            break
        idx -= deck[t]
    _pay(p, s.bank, _COST_DEV_ITEMS)
    return (cur,)


def _h_play_knight(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 3:
        _fail("bad action")
    if s.phase == PHASE_MAIN:
        _require_main(s)
    elif s.phase != PHASE_ROLL:
        _fail("knight may only be played before rolling or in the main phase")
    if s.dev_played_this_turn:
        _fail("already played a development card this turn")
    cur = s.current
    p = s.players[cur]
    if p.dev_cards[B.DEV_KNIGHT] <= 0:
        _fail("no playable knight")
    touched = _move_robber(s, a[1], a[2], cur, rng)
    p.dev_cards[B.DEV_KNIGHT] -= 1
    p.played_knights += 1
    s.dev_played_this_turn = True
    _update_largest_army(s, cur)
    return touched if touched else (cur,)


def _h_play_road_building(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 1:
        _fail("bad action")
    _require_main(s)
    if s.dev_played_this_turn:
        _fail("already played a development card this turn")
    cur = s.current
    p = s.players[cur]
    if p.dev_cards[B.DEV_ROAD_BUILDING] <= 0:
        _fail("no playable road building card")
    p.dev_cards[B.DEV_ROAD_BUILDING] -= 1
    s.dev_played_this_turn = True
    free = min(2, B.MAX_ROADS - len(p.roads))
    if free > 0 and not _road_edges(s, cur, _vertex_owners(s), _edge_owners(s)):
        free = 0  # forfeited: no legal road
    s.free_roads = max(free, 0)
    return (cur,)


def _h_play_year_of_plenty(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 3:
        _fail("bad action")
    _require_main(s)
    if s.dev_played_this_turn:
        _fail("already played a development card this turn")
    cur = s.current
    p = s.players[cur]
    if p.dev_cards[B.DEV_YEAR_OF_PLENTY] <= 0:
        _fail("no playable year of plenty card")
    r1, r2 = a[1], a[2]
    if not (0 <= r1 <= r2 < 5):
        _fail("resources must satisfy 0 <= res1 <= res2 < 5")
    bank = s.bank
    if r1 == r2:
        if bank[r1] < 2:
            _fail("bank cannot supply the requested resources")
    elif bank[r1] < 1 or bank[r2] < 1:
        _fail("bank cannot supply the requested resources")
    bank[r1] -= 1
    bank[r2] -= 1
    p.resources[r1] += 1
    p.resources[r2] += 1
    p.dev_cards[B.DEV_YEAR_OF_PLENTY] -= 1
    s.dev_played_this_turn = True
    return (cur,)


def _h_play_monopoly(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 2:
        _fail("bad action")
    _require_main(s)
    if s.dev_played_this_turn:
        _fail("already played a development card this turn")
    cur = s.current
    p = s.players[cur]
    if p.dev_cards[B.DEV_MONOPOLY] <= 0:
        _fail("no playable monopoly card")
    r = a[1]
    if not (0 <= r < 5):
        _fail("bad resource")
    taken = 0
    for i, q in enumerate(s.players):
        if i != cur:
            taken += q.resources[r]
            q.resources[r] = 0
    p.resources[r] += taken
    p.dev_cards[B.DEV_MONOPOLY] -= 1
    s.dev_played_this_turn = True
    return tuple(range(len(s.players)))


def _h_bank_trade(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 3:
        _fail("bad action")
    _require_main(s)
    give, get = a[1], a[2]
    if not (0 <= give < 5 and 0 <= get < 5) or give == get:
        _fail("bad bank trade")
    cur = s.current
    p = s.players[cur]
    ratio = s.port_ratio(cur, give)
    if p.resources[give] < ratio:
        _fail("not enough cards for the bank ratio")
    if s.bank[get] < 1:
        _fail("bank is out of that resource")
    p.resources[give] -= ratio
    s.bank[give] += ratio
    s.bank[get] -= 1
    p.resources[get] += 1
    return (cur,)


def _next_responder(s: GameState, offer: TradeOffer) -> int:
    n = len(s.players)
    for k in range(1, n):
        i = (offer.proposer + k) % n
        if i not in offer.responses:
            return i
    return -1


def _finish_responses(s: GameState, offer: TradeOffer) -> None:
    """Called when no responder is left: go to TRADE_SELECT or back to MAIN."""
    s.trade_responder = -1
    if any(offer.responses.values()):
        s.phase = PHASE_TRADE_SELECT
    else:
        s.pending_trade = None
        s.phase = PHASE_MAIN


def _h_propose_trade(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if len(a) != 3:
        _fail("bad action")
    _require_main(s)
    if s.trades_this_turn >= MAX_TRADE_PROPOSALS_PER_TURN:
        _fail("no trade proposals left this turn")
    give, get = a[1], a[2]
    if len(give) != 5 or len(get) != 5:
        _fail("trade vectors need 5 counts")
    give_total = get_total = 0
    for r in range(5):
        g, t = give[r], get[r]
        if g < 0 or t < 0:
            _fail("negative trade counts")
        if g and t:
            _fail("give and get must be disjoint")
        give_total += g
        get_total += t
    if give_total == 0 or get_total == 0:
        _fail("trade must be non-empty on both sides")
    cur = s.current
    if not _can_pay(s.players[cur], give):
        _fail("proposer does not hold the offered cards")
    offer = TradeOffer(cur, list(give), list(get))
    n = len(s.players)
    for k in range(1, n):
        i = (cur + k) % n
        if not _can_pay(s.players[i], get):
            offer.responses[i] = False  # auto-reject: cannot pay
    s.pending_trade = offer
    s.trades_this_turn += 1
    nxt = _next_responder(s, offer)
    if nxt < 0:
        _finish_responses(s, offer)
    else:
        s.trade_responder = nxt
        s.phase = PHASE_TRADE_RESPONSE
    return ()


def _respond(s: GameState, a: Action, accepted: bool) -> Tuple[int, ...]:
    if s.phase != PHASE_TRADE_RESPONSE or len(a) != 1:
        _fail("no trade response pending")
    offer = s.pending_trade
    i = s.trade_responder
    if offer is None or i < 0 or i in offer.responses:
        _fail("no trade response pending")
    if accepted and not _can_pay(s.players[i], offer.get):
        _fail("responder cannot pay")
    offer.responses[i] = accepted
    nxt = _next_responder(s, offer)
    if nxt < 0:
        _finish_responses(s, offer)
    else:
        s.trade_responder = nxt
    return ()


def _h_accept_trade(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    return _respond(s, a, True)


def _h_reject_trade(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    return _respond(s, a, False)


def _h_execute_trade(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_TRADE_SELECT or len(a) != 2:
        _fail("no trade to execute")
    offer = s.pending_trade
    partner = a[1]
    if offer is None or not offer.responses.get(partner, False):
        _fail("partner did not accept")
    proposer = s.players[offer.proposer]
    other = s.players[partner]
    if not _can_pay(proposer, offer.give) or not _can_pay(other, offer.get):
        _fail("cards no longer available")
    pr, orr = proposer.resources, other.resources
    for r in range(5):
        g, t = offer.give[r], offer.get[r]
        if g:
            pr[r] -= g
            orr[r] += g
        if t:
            orr[r] -= t
            pr[r] += t
    s.pending_trade = None
    s.trade_responder = -1
    s.phase = PHASE_MAIN
    return (offer.proposer, partner)


def _h_cancel_trade(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_TRADE_SELECT or len(a) != 1:
        _fail("no trade to cancel")
    s.pending_trade = None
    s.trade_responder = -1
    s.phase = PHASE_MAIN
    return ()


def _h_end_turn(s: GameState, a: Action, rng: Optional[random.Random]) -> Tuple[int, ...]:
    if s.phase != PHASE_MAIN or len(a) != 1:
        _fail("can only end the turn in the main phase")
    cur = s.current
    p = s.players[cur]
    new = p.dev_cards_new
    if new[0] or new[1] or new[2] or new[3] or new[4]:
        dc = p.dev_cards
        for t in range(5):
            dc[t] += new[t]
            new[t] = 0
    s.dev_played_this_turn = False
    s.free_roads = 0
    s.trades_this_turn = 0
    s.pending_trade = None
    s.trade_responder = -1
    s.dice = 0
    s.turn += 1
    s.current = (cur + 1) % len(s.players)
    s.phase = PHASE_ROLL
    if s.turn >= s.max_turns:
        _end_by_cap(s)
    return ()


def _end_by_cap(s: GameState) -> None:
    n = len(s.players)
    vps = [count_vp(s, i) for i in range(n)]
    best = max(vps)
    winners = [i for i in range(n) if vps[i] == best]
    s.winner = winners[0] if len(winners) == 1 else -1
    s.phase = PHASE_GAME_OVER


_HANDLERS: Dict[str, Callable[[GameState, Action, Optional[random.Random]], Tuple[int, ...]]] = {
    A.SETUP_SETTLEMENT: _h_setup_settlement,
    A.SETUP_ROAD: _h_setup_road,
    A.ROLL: _h_roll,
    A.DISCARD: _h_discard,
    A.MOVE_ROBBER: _h_move_robber,
    A.BUILD_ROAD: _h_build_road,
    A.BUILD_SETTLEMENT: _h_build_settlement,
    A.BUILD_CITY: _h_build_city,
    A.BUY_DEV: _h_buy_dev,
    A.PLAY_KNIGHT: _h_play_knight,
    A.PLAY_ROAD_BUILDING: _h_play_road_building,
    A.PLAY_YEAR_OF_PLENTY: _h_play_year_of_plenty,
    A.PLAY_MONOPOLY: _h_play_monopoly,
    A.BANK_TRADE: _h_bank_trade,
    A.PROPOSE_TRADE: _h_propose_trade,
    A.ACCEPT_TRADE: _h_accept_trade,
    A.REJECT_TRADE: _h_reject_trade,
    A.EXECUTE_TRADE: _h_execute_trade,
    A.CANCEL_TRADE: _h_cancel_trade,
    A.END_TURN: _h_end_turn,
}


# ---------------------------------------------------------------------------
# apply / apply_inplace
# ---------------------------------------------------------------------------
def apply_inplace(state: GameState, action: Action, rng: Optional[random.Random] = None) -> GameState:
    """Apply ``action`` to ``state`` in place and return it.

    Raises :class:`IllegalActionError` when the action is not legal; every
    handler validates before it mutates, so a rejected action leaves the
    state unchanged.  ``rng`` is only consulted for random events (dice, dev
    card draws, robber steals); when ``None`` a fresh ``random.Random`` is
    used for those.
    """
    if state.phase == PHASE_GAME_OVER:
        raise IllegalActionError("game is over")
    try:
        handler = _HANDLERS[action[0]]
    except (KeyError, IndexError, TypeError):
        raise IllegalActionError(f"unknown action {action!r}") from None
    touched = handler(state, action, rng)
    if touched:
        players = state.players
        for i in touched:
            _sync(players[i])
    # Win check: the current player wins as soon as they hold >= 10 VP during
    # their own turn (after END_TURN this is the *new* current player, who may
    # have received Longest Road during the previous turn).
    phase = state.phase
    if phase != PHASE_GAME_OVER and phase not in _SETUP_PHASES and count_vp(state, state.current) >= B.VP_TO_WIN:
        state.winner = state.current
        state.phase = PHASE_GAME_OVER
    return state


def apply(state: GameState, action: Action, rng: Optional[random.Random] = None) -> GameState:
    """Return a new state with ``action`` applied (the input is not modified)."""
    return apply_inplace(state.copy(), action, rng)


def apply_roll(state: GameState, value: int, rng: Optional[random.Random] = None) -> GameState:
    """Apply a forced roll ``(ROLL, value)`` on a copy of ``state``."""
    return apply_inplace(state.copy(), (A.ROLL, value), rng)


# ---------------------------------------------------------------------------
# Random playout (tests / benchmarks / rollouts)
# ---------------------------------------------------------------------------
def random_playout(
    state: GameState,
    rng: random.Random,
    max_turns: Optional[int] = None,
    max_actions: int = 2_000_000,
) -> GameState:
    """Play uniformly random legal actions on a copy of ``state`` until the game ends.

    ``max_turns`` overrides ``state.max_turns`` when given.  ``max_actions`` is
    a safety valve (an engine bug that stalls a game raises ``RuntimeError``
    instead of hanging).
    """
    s = state.copy()
    if max_turns is not None:
        s.max_turns = max_turns
    choice = rng.choice
    n = 0
    while s.phase != PHASE_GAME_OVER:
        acts = legal_actions(s)
        if not acts:
            raise RuntimeError(f"no legal actions in phase {s.phase!r} for player {acting_player(s)}")
        apply_inplace(s, choice(acts), rng)
        n += 1
        if n > max_actions:
            raise RuntimeError("random_playout exceeded max_actions")
    return s
