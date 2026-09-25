"""GameState -> dense numpy feature vector from one player's perspective.

Layout (see :data:`FEATURE_NAMES`)
----------------------------------
The vector is ``4 * PLAYER_BLOCK + GLOBAL_BLOCK`` floats:

* block 0: "me" (the perspective player),
* blocks 1..3: the opponents in seat order *after* me (cyclic), padded with
  all-zero blocks (``present = 0``) when the game has fewer than 4 players,
* a global block: bank, dev deck, turn/stage, player count, phase one-hots,
  whose turn it is, VP / road / army gaps and the pending trade offer.

Every feature is scaled to roughly ``[0, 1]`` (or ``[-1, 1]`` for the
signed gap features).  The scale of each feature is documented next to its
name in :data:`_PLAYER_FEATURES` / :data:`_GLOBAL_FEATURES`.

Performance
-----------
``extract_batch`` does the expensive per-state work (occupancy, production,
reachable spots, longest roads) **once per distinct state object** and then
assembles each requested perspective by re-ordering the per-player blocks.
The self-play runner calls ``extract_batch([state] * n, range(n))`` which
therefore costs one state analysis, not ``n``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import accel as _accel  # optional C++ extension (catanbot_core); see docs/CPP.md
from . import board as B
from .state import (GameState, PHASE_DISCARD, PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER, PHASE_ROLL,
                    PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT, PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT)

__all__ = [
    "FEATURE_NAMES", "NUM_FEATURES", "PLAYER_BLOCK", "GLOBAL_BLOCK", "MAX_PLAYERS",
    "extract", "extract_batch", "feature_index", "longest_road_length", "production_per_roll",
    "settlement_spots", "vertex_production_tables",
]

MAX_PLAYERS = 4
_NV = B.NUM_VERTICES
_NE = B.NUM_EDGES
_RES = B.RESOURCE_NAMES[:5]
_DEV = B.DEV_NAMES

# Precomputed tables (tuples for fast iteration).
_VERTEX_HEXES = tuple(tuple(h) for h in B.VERTEX_HEXES)
_VERTEX_NEIGHBORS = tuple(tuple(n) for n in B.VERTEX_NEIGHBORS)
_VERTEX_EDGES = tuple(tuple(e) for e in B.VERTEX_EDGES)
_EDGE_VERTICES = tuple(B.EDGE_VERTICES)
_HEX_VERTICES = tuple(tuple(v) for v in B.HEX_VERTICES)
_PIPS = B.PIPS

# Phases in a fixed order for the one-hot block.
_PHASES = [PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD, PHASE_ROLL, PHASE_DISCARD, PHASE_ROBBER, PHASE_MAIN,
           PHASE_TRADE_RESPONSE, PHASE_TRADE_SELECT, PHASE_GAME_OVER]
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASES)}

# ---------------------------------------------------------------------------
# Feature naming / scales
# ---------------------------------------------------------------------------
# (name, scale description) for one player block.  The order here is the
# order in the vector; ``_P`` below maps name -> offset inside the block.
_PLAYER_FEATURES: List[Tuple[str, str]] = (
    [("public_vp", "/10"), ("hidden_vp", "/5 (own exact, opponents expected)"),
     ("total_vp", "(public + hidden estimate)/10"),
     ("settlements", "/5"), ("cities", "/4"), ("roads", "/15")]
    + [(f"res_{r}", "/10 (expected counts when hand unknown)") for r in _RES]
    + [("hand_known", "0/1"), ("hand_size", "/15"), ("cards_over_7", "max(0, hand-7)/8"),
       ("discard_exposure", "floor(hand/2) if hand>7 else 0, /8")]
    + [(f"prod_{r}", "expected cards per roll (robber-aware), /2") for r in _RES]
    + [(f"prod_nr_{r}", "expected cards per roll ignoring the robber, /2") for r in _RES]
    + [("prod_total", "/5"), ("prod_nr_total", "/5"), ("robber_loss", "production lost to the robber, /2"),
       ("prod_types", "resource types produced / 5"), ("prod_entropy", "normalised entropy of production, 0..1"),
       ("robber_on_my_hex", "0/1: robber touches one of my buildings")]
    + [("spots_now", "buildable settlement vertices now, /10"),
       ("spots_1road", "new spots reachable with exactly 1 more road, /10"),
       ("spots_2roads", "new spots reachable with exactly 2 more roads, /15"),
       ("best_spot_pips_now", "best pip total among spots_now, /15"),
       ("best_spot_pips_2roads", "best pip total among spots within 2 roads, /15")]
    + [("can_build_road", "0/1 (afford + piece left)"), ("can_build_settlement", "0/1 (afford + piece + spot)"),
       ("can_build_city", "0/1 (afford + piece + settlement to upgrade)"), ("can_buy_dev", "0/1 (afford + deck)")]
    + [("longest_road", "longest simple road path /15"), ("has_longest_road", "0/1"),
       ("knights", "played knights /5"), ("has_largest_army", "0/1")]
    + [(f"dev_{d}", "playable dev cards of type, /3 (expected when unknown)") for d in _DEV]
    + [(f"devnew_{d}", "dev cards bought this turn (not playable yet), /3") for d in _DEV]
    + [("dev_total", "unplayed dev cards /5"), ("dev_known", "0/1")]
    + [(f"port_{r}", "(4 - bank ratio)/2: 0 = 4:1, 0.5 = 3:1, 1 = 2:1") for r in _RES]
    + [("is_current", "0/1: this seat is state.current"), ("present", "0/1: seat exists (padding)")]
)
PLAYER_BLOCK = len(_PLAYER_FEATURES)
_P: Dict[str, int] = {name: i for i, (name, _) in enumerate(_PLAYER_FEATURES)}

_GLOBAL_FEATURES: List[Tuple[str, str]] = (
    [(f"bank_{r}", "/19") for r in _RES]
    + [(f"deck_{d}", "remaining / initial count of that type") for d in _DEV]
    + [("deck_total", "/25"), ("turn", "min(turn/200, 1)"), ("stage", "min(turn/(30*num_players), 1)"),
       ("players_2", "0/1"), ("players_3", "0/1"), ("players_4", "0/1")]
    + [(f"phase_{p}", "0/1") for p in _PHASES]
    + [("my_turn", "0/1: state.current == me"), ("i_am_acting", "0/1: I must act now (discard/respond/current)"),
       ("dice", "last roll /12"), ("free_roads", "/2"), ("dev_played_this_turn", "0/1"),
       ("trades_this_turn", "/4")]
    + [("vp_gap", "(my public vp - best opponent public vp)/10"),
       ("vp_gap_expected", "(my total vp - best opponent expected total vp)/10"),
       ("leader_vp", "max public vp over all players /10"), ("i_am_leader", "0/1 (ties count)"),
       ("lr_gap", "(my longest road - best opponent's)/5"), ("knight_gap", "(my knights - best opponent's)/5")]
    + [("trade_pending", "0/1"), ("trade_i_propose", "0/1"), ("trade_i_respond", "0/1: I am the responder")]
    + [(f"trade_give_{r}", "proposer gives, /3") for r in _RES]
    + [(f"trade_get_{r}", "proposer wants, /3") for r in _RES]
)
GLOBAL_BLOCK = len(_GLOBAL_FEATURES)
_G: Dict[str, int] = {name: i for i, (name, _) in enumerate(_GLOBAL_FEATURES)}

FEATURE_NAMES: List[str] = (
    [f"me_{n}" for n, _ in _PLAYER_FEATURES]
    + [f"opp{k}_{n}" for k in (1, 2, 3) for n, _ in _PLAYER_FEATURES]
    + [f"g_{n}" for n, _ in _GLOBAL_FEATURES]
)
NUM_FEATURES = len(FEATURE_NAMES)
_GLOBAL_OFFSET = MAX_PLAYERS * PLAYER_BLOCK
assert NUM_FEATURES == _GLOBAL_OFFSET + GLOBAL_BLOCK
assert NUM_FEATURES < 400

FEATURE_SCALES: Dict[str, str] = {}
for _n, _s in _PLAYER_FEATURES:
    FEATURE_SCALES[f"me_{_n}"] = _s
    for _k in (1, 2, 3):
        FEATURE_SCALES[f"opp{_k}_{_n}"] = _s
for _n, _s in _GLOBAL_FEATURES:
    FEATURE_SCALES[f"g_{_n}"] = _s


def feature_index(name: str) -> int:
    """Index of a feature by its full name (e.g. ``"me_prod_ore"``)."""
    return FEATURE_NAMES.index(name)


# ---------------------------------------------------------------------------
# Board-level tables (cached per hexes list)
# ---------------------------------------------------------------------------
_HEX_TABLE_CACHE: Dict[int, Tuple[list, np.ndarray, np.ndarray]] = {}


def vertex_production_tables(hexes: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """``(prod, pips)`` for a hexes list.

    ``prod`` is ``(54, 5)`` float64: expected cards of each resource per roll
    for a settlement on that vertex (pips / 36, robber ignored).  ``pips`` is
    ``(54,)``: pip total of the vertex.  Cached per ``hexes`` list object.
    """
    key = id(hexes)
    hit = _HEX_TABLE_CACHE.get(key)
    if hit is not None and hit[0] is hexes:
        return hit[1], hit[2]
    prod = np.zeros((_NV, 5), np.float64)
    pips = np.zeros(_NV, np.float64)
    for v in range(_NV):
        for h in _VERTEX_HEXES[v]:
            res, num = hexes[h]
            if res == B.DESERT:
                continue
            p = _PIPS.get(num, 0)
            prod[v, res] += p / 36.0
            pips[v] += p
    if len(_HEX_TABLE_CACHE) > 64:
        _HEX_TABLE_CACHE.clear()
    _HEX_TABLE_CACHE[key] = (hexes, prod, pips)  # keep a ref so id() stays unique
    return prod, pips


# ---------------------------------------------------------------------------
# Reusable helpers (other modules may import these)
# ---------------------------------------------------------------------------
def longest_road_length(state: GameState, player: int) -> int:
    """Longest trail (no edge reused) over ``player``'s roads.

    A vertex holding an opponent's building ends the path: the road leading
    into it counts, but the path cannot continue through it.  Vertices may be
    revisited (official rule); roads may not.
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
            length = 1 if blocked[w] else 1 + dfs(w, used | bit)
            if length > best:
                best = length
        return best

    total = len(roads)
    best = 0
    for v in adj:
        length = dfs(v, 0)
        if length > best:
            best = length
            if best == total:
                break
    return best


def production_per_roll(state: GameState, player: int, ignore_robber: bool = False) -> np.ndarray:
    """Expected cards per roll for each resource (5,), pips-weighted, x2 for cities."""
    prod, _ = vertex_production_tables(state.hexes)
    p = state.players[player]
    out = np.zeros(5, np.float64)
    for v in p.settlements:
        out += prod[v]
    for v in p.cities:
        out += 2.0 * prod[v]
    if not ignore_robber and 0 <= state.robber < B.NUM_HEXES:
        res, num = state.hexes[state.robber]
        if res != B.DESERT:
            pr = _PIPS.get(num, 0) / 36.0
            rv = _HEX_VERTICES[state.robber]
            n = sum(1 for v in p.settlements if v in rv) + 2 * sum(1 for v in p.cities if v in rv)
            out[res] = max(0.0, out[res] - n * pr)
    return out


def settlement_spots(state: GameState, player: int) -> Tuple[List[int], List[int], List[int]]:
    """``(now, one_road, two_roads)`` lists of vertices where ``player`` could settle.

    ``now``: distance >= 2 from every building and touching an own road.
    ``one_road`` / ``two_roads``: additional spots reachable by building
    exactly 1 / 2 more roads along free edges from the own road network
    (paths never continue through an opponent's building).
    """
    info = _analyse_players(state)
    spots = info["spots"][player]
    return spots[0], spots[1], spots[2]


# ---------------------------------------------------------------------------
# Per-state analysis
# ---------------------------------------------------------------------------
def _spots_for_player(player: int, roads: Sequence[int], owner: Dict[int, int], edge_taken: set,
                      settlement_ok: bytearray) -> Tuple[List[int], List[int], List[int]]:
    """Geometric settlement spots at road distance 0 / 1 / 2 from the network."""
    if not roads:
        return [], [], []
    seen = bytearray(_NV)
    frontier: List[int] = []
    for e in roads:
        for v in _EDGE_VERTICES[e]:
            if not seen[v]:
                seen[v] = 1
                frontier.append(v)
    now = [v for v in frontier if settlement_ok[v]]
    levels: List[List[int]] = []
    for _ in range(2):
        nxt: List[int] = []
        for v in frontier:
            o = owner.get(v, -1)
            if o != -1 and o != player:
                continue  # cannot build a road out of an opponent's building
            for e in _VERTEX_EDGES[v]:
                if e in edge_taken:
                    continue
                a, b = _EDGE_VERTICES[e]
                w = b if a == v else a
                if not seen[w]:
                    seen[w] = 1
                    nxt.append(w)
        levels.append([w for w in nxt if settlement_ok[w]])
        frontier = nxt
    return now, levels[0], levels[1]


def _analyse_players(state: GameState) -> dict:
    """All per-player quantities needed by the feature blocks."""
    players = state.players
    n = len(players)
    prod_tab, pips_tab = vertex_production_tables(state.hexes)
    owner: Dict[int, int] = {}
    edge_taken: set = set()
    for i, p in enumerate(players):
        for v in p.settlements:
            owner[v] = i
        for v in p.cities:
            owner[v] = i
        edge_taken.update(p.roads)
    settlement_ok = bytearray([1]) * _NV
    for v in owner:
        settlement_ok[v] = 0
        for w in _VERTEX_NEIGHBORS[v]:
            settlement_ok[w] = 0

    robber = state.robber
    rob_vertices: Tuple[int, ...] = ()
    prod_r_tab = prod_tab
    if 0 <= robber < B.NUM_HEXES:
        rob_vertices = _HEX_VERTICES[robber]
        rob_res, rob_num = state.hexes[robber]
        if rob_res != B.DESERT:
            # robber-aware table: the robber hex pays nothing at its 6 corners
            prod_r_tab = prod_tab.copy()
            rob_p = _PIPS.get(rob_num, 0) / 36.0
            for v in rob_vertices:
                prod_r_tab[v, rob_res] -= rob_p

    prod_nr = np.zeros((n, 5), np.float64)
    prod = np.zeros((n, 5), np.float64)
    robber_touch = [0.0] * n
    spots: List[Tuple[List[int], List[int], List[int]]] = []
    lr = [0] * n
    for i, p in enumerate(players):
        row_nr = prod_nr[i]
        row = prod[i]
        for v in p.settlements:
            row_nr += prod_tab[v]
            row += prod_r_tab[v]
            if v in rob_vertices:
                robber_touch[i] = 1.0
        for v in p.cities:
            row_nr += prod_tab[v]
            row_nr += prod_tab[v]
            row += prod_r_tab[v]
            row += prod_r_tab[v]
            if v in rob_vertices:
                robber_touch[i] = 1.0
        spots.append(_spots_for_player(i, p.roads, owner, edge_taken, settlement_ok))
        lr[i] = longest_road_length(state, i)
    return {"prod": prod, "prod_nr": prod_nr, "pips": pips_tab, "robber_touch": robber_touch,
            "spots": spots, "lr": lr}


def _hidden_vp_estimates(state: GameState) -> List[float]:
    """Expected VP cards per player (exact when ``dev_known``)."""
    players = state.players
    known_vp = 0
    unknown_total = 0
    for p in players:
        if p.dev_known:
            known_vp += p.vp_cards
        else:
            unknown_total += p.dev_count
    out = [0.0] * len(players)
    if unknown_total > 0:
        deck_vp = state.dev_deck[B.DEV_VP] if len(state.dev_deck) > B.DEV_VP else 0
        vp_unknown = min(unknown_total, max(0, B.DEV_DECK_COUNTS[B.DEV_VP] - known_vp - deck_vp))
        frac = vp_unknown / unknown_total
    else:
        frac = 0.0
    for i, p in enumerate(players):
        out[i] = float(p.vp_cards) if p.dev_known else p.dev_count * frac
    return out


def _player_blocks(state: GameState, info: dict) -> np.ndarray:
    """``(num_players, PLAYER_BLOCK)`` blocks for every seat."""
    players = state.players
    n = len(players)
    out = np.zeros((n, PLAYER_BLOCK), np.float32)
    prod = info["prod"]
    prod_nr = info["prod_nr"]
    pips = info["pips"]
    hidden = _hidden_vp_estimates(state)
    deck_total = sum(state.dev_deck)
    deck_left = deck_total > 0
    P = _P
    for i, p in enumerate(players):
        row = out[i]
        pub = state.public_vp(i)
        row[P["public_vp"]] = pub / 10.0
        row[P["hidden_vp"]] = hidden[i] / 5.0
        row[P["total_vp"]] = (pub + hidden[i]) / 10.0
        ns, nc, nr = len(p.settlements), len(p.cities), len(p.roads)
        row[P["settlements"]] = ns / 5.0
        row[P["cities"]] = nc / 4.0
        row[P["roads"]] = nr / 15.0
        # --- hand ---------------------------------------------------------
        res = p.resources
        hand = sum(res) if p.hand_known else p.hand_size
        r0 = P["res_wood"]
        if p.hand_known:
            for k in range(5):
                row[r0 + k] = res[k] / 10.0
        else:
            tot = float(prod_nr[i].sum())
            if tot > 0:
                for k in range(5):
                    row[r0 + k] = hand * prod_nr[i, k] / tot / 10.0
            else:
                for k in range(5):
                    row[r0 + k] = hand / 5.0 / 10.0
        row[P["hand_known"]] = 1.0 if p.hand_known else 0.0
        row[P["hand_size"]] = hand / 15.0
        if hand > 7:
            row[P["cards_over_7"]] = (hand - 7) / 8.0
            row[P["discard_exposure"]] = (hand // 2) / 8.0
        # --- production ---------------------------------------------------
        q0 = P["prod_wood"]
        qn = P["prod_nr_wood"]
        tot_r = 0.0
        tot_nr = 0.0
        types = 0
        ent = 0.0
        for k in range(5):
            a = prod[i, k]
            b = prod_nr[i, k]
            row[q0 + k] = a / 2.0
            row[qn + k] = b / 2.0
            tot_r += a
            tot_nr += b
            if b > 0:
                types += 1
        if tot_nr > 0:
            for k in range(5):
                b = prod_nr[i, k]
                if b > 0:
                    f = b / tot_nr
                    ent -= f * math.log(f)
            ent /= math.log(5.0)
        row[P["prod_total"]] = tot_r / 5.0
        row[P["prod_nr_total"]] = tot_nr / 5.0
        row[P["robber_loss"]] = (tot_nr - tot_r) / 2.0
        row[P["prod_types"]] = types / 5.0
        row[P["prod_entropy"]] = ent
        row[P["robber_on_my_hex"]] = info["robber_touch"][i]
        # --- expansion ----------------------------------------------------
        now, one, two = info["spots"][i]
        row[P["spots_now"]] = len(now) / 10.0
        row[P["spots_1road"]] = len(one) / 10.0
        row[P["spots_2roads"]] = len(two) / 15.0
        best_now = max((pips[v] for v in now), default=0.0)
        best_2 = best_now
        for lst in (one, two):
            for v in lst:
                if pips[v] > best_2:
                    best_2 = pips[v]
        row[P["best_spot_pips_now"]] = best_now / 15.0
        row[P["best_spot_pips_2roads"]] = best_2 / 15.0
        # --- affordability --------------------------------------------------
        w, b_, s, wh, o = res
        if p.hand_known:
            row[P["can_build_road"]] = 1.0 if (w >= 1 and b_ >= 1 and nr < B.MAX_ROADS) else 0.0
            row[P["can_build_settlement"]] = 1.0 if (w >= 1 and b_ >= 1 and s >= 1 and wh >= 1
                                                     and ns < B.MAX_SETTLEMENTS and now) else 0.0
            row[P["can_build_city"]] = 1.0 if (wh >= 2 and o >= 3 and ns > 0 and nc < B.MAX_CITIES) else 0.0
            row[P["can_buy_dev"]] = 1.0 if (s >= 1 and wh >= 1 and o >= 1 and deck_left) else 0.0
        # --- awards ----------------------------------------------------------
        row[P["longest_road"]] = info["lr"][i] / 15.0
        row[P["has_longest_road"]] = 1.0 if state.longest_road_owner == i else 0.0
        row[P["knights"]] = p.played_knights / 5.0
        row[P["has_largest_army"]] = 1.0 if state.largest_army_owner == i else 0.0
        # --- dev cards -------------------------------------------------------
        d0 = P["dev_knight"]
        dn = P["devnew_knight"]
        if p.dev_known:
            dc, dnew = p.dev_cards, p.dev_cards_new
            for k in range(5):
                row[d0 + k] = dc[k] / 3.0
                row[dn + k] = dnew[k] / 3.0
            row[P["dev_total"]] = p.total_dev / 5.0
            row[P["dev_known"]] = 1.0
        else:
            cnt = p.dev_count
            if cnt > 0:
                # expected composition: VP estimate from _hidden_vp_estimates, the
                # rest proportional to the remaining deck (uniform if deck empty).
                rest = max(0.0, cnt - hidden[i])
                deck = state.dev_deck
                non_vp = [deck[k] if k != B.DEV_VP else 0 for k in range(5)]
                tot = float(sum(non_vp))
                for k in range(5):
                    if k == B.DEV_VP:
                        row[d0 + k] = hidden[i] / 3.0
                    elif tot > 0:
                        row[d0 + k] = rest * non_vp[k] / tot / 3.0
                    else:
                        row[d0 + k] = rest / 4.0 / 3.0
            row[P["dev_total"]] = cnt / 5.0
        # --- ports -------------------------------------------------------------
        p0 = P["port_wood"]
        generic = False
        for v in p.settlements:
            t = state.ports.get(v)
            if t is None:
                continue
            if t == B.PORT_GENERIC:
                generic = True
            else:
                row[p0 + t] = 1.0
        for v in p.cities:
            t = state.ports.get(v)
            if t is None:
                continue
            if t == B.PORT_GENERIC:
                generic = True
            else:
                row[p0 + t] = 1.0
        if generic:
            for k in range(5):
                if row[p0 + k] < 0.5:
                    row[p0 + k] = 0.5
        row[P["is_current"]] = 1.0 if state.current == i else 0.0
        row[P["present"]] = 1.0
    return out


def _acting_player(state: GameState) -> int:
    if state.phase == PHASE_DISCARD and state.discard_queue:
        return state.discard_queue[0]
    if state.phase == PHASE_TRADE_RESPONSE:
        return state.trade_responder
    return state.current


def _global_block(state: GameState, player: int, info: dict, blocks: np.ndarray, out: np.ndarray) -> None:
    """Fill the perspective-dependent global block into ``out`` (length GLOBAL_BLOCK)."""
    G = _G
    n = len(state.players)
    bank = state.bank
    for k in range(5):
        out[G["bank_wood"] + k] = bank[k] / float(B.BANK_PER_RESOURCE)
    deck = state.dev_deck
    for k in range(5):
        out[G["deck_knight"] + k] = deck[k] / float(B.DEV_DECK_COUNTS[k])
    out[G["deck_total"]] = sum(deck) / 25.0
    out[G["turn"]] = min(1.0, state.turn / 200.0)
    out[G["stage"]] = min(1.0, state.turn / (30.0 * max(1, n)))
    if 2 <= n <= 4:
        out[G["players_2"] + (n - 2)] = 1.0
    pi = _PHASE_INDEX.get(state.phase)
    if pi is not None:
        out[G["phase_" + _PHASES[0]] + pi] = 1.0
    out[G["my_turn"]] = 1.0 if state.current == player else 0.0
    out[G["i_am_acting"]] = 1.0 if _acting_player(state) == player else 0.0
    out[G["dice"]] = state.dice / 12.0
    out[G["free_roads"]] = state.free_roads / 2.0
    out[G["dev_played_this_turn"]] = 1.0 if state.dev_played_this_turn else 0.0
    out[G["trades_this_turn"]] = state.trades_this_turn / 4.0
    # --- gaps ---------------------------------------------------------------
    P = _P
    pub = blocks[:, P["public_vp"]]
    totv = blocks[:, P["total_vp"]]
    lr = info["lr"]
    my_pub = float(pub[player])
    my_tot = float(totv[player])
    best_pub = 0.0
    best_tot = 0.0
    best_lr = 0
    best_kn = 0
    leader = my_pub
    for j in range(n):
        if pub[j] > leader:
            leader = float(pub[j])
        if j == player:
            continue
        if pub[j] > best_pub:
            best_pub = float(pub[j])
        if totv[j] > best_tot:
            best_tot = float(totv[j])
        if lr[j] > best_lr:
            best_lr = lr[j]
        kn = state.players[j].played_knights
        if kn > best_kn:
            best_kn = kn
    out[G["vp_gap"]] = my_pub - best_pub          # already /10
    out[G["vp_gap_expected"]] = my_tot - best_tot  # already /10
    out[G["leader_vp"]] = leader
    out[G["i_am_leader"]] = 1.0 if my_pub >= leader else 0.0
    out[G["lr_gap"]] = (lr[player] - best_lr) / 5.0
    out[G["knight_gap"]] = (state.players[player].played_knights - best_kn) / 5.0
    # --- pending trade -----------------------------------------------------
    t = state.pending_trade
    if t is not None:
        out[G["trade_pending"]] = 1.0
        out[G["trade_i_propose"]] = 1.0 if t.proposer == player else 0.0
        out[G["trade_i_respond"]] = 1.0 if (state.phase == PHASE_TRADE_RESPONSE
                                            and state.trade_responder == player) else 0.0
        g0, r0 = G["trade_give_wood"], G["trade_get_wood"]
        for k in range(5):
            out[g0 + k] = t.give[k] / 3.0
            out[r0 + k] = t.get[k] / 3.0


def _assemble(state: GameState, player: int, info: dict, blocks: np.ndarray, out: np.ndarray) -> None:
    """Write the full feature vector for ``player`` into ``out`` (length NUM_FEATURES)."""
    n = len(state.players)
    out[:_GLOBAL_OFFSET] = 0.0
    out[0:PLAYER_BLOCK] = blocks[player]
    for k in range(1, MAX_PLAYERS):
        if k < n:
            j = (player + k) % n
            out[k * PLAYER_BLOCK:(k + 1) * PLAYER_BLOCK] = blocks[j]
    out[_GLOBAL_OFFSET:] = 0.0
    _global_block(state, player, info, blocks, out[_GLOBAL_OFFSET:])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract(state: GameState, player: int) -> np.ndarray:
    """Feature vector (float32, shape ``(NUM_FEATURES,)``) from ``player``'s perspective."""
    if _accel.AVAILABLE:
        return _accel.extract(state, player)
    info = _analyse_players(state)
    blocks = _player_blocks(state, info)
    out = np.zeros(NUM_FEATURES, np.float32)
    _assemble(state, player, info, blocks, out)
    return out


def extract_batch(states: Sequence[GameState], players: Sequence[int]) -> np.ndarray:
    """Feature matrix ``(N, NUM_FEATURES)`` float32 for the (state, player) pairs.

    Consecutive identical state objects share one analysis, so
    ``extract_batch([s] * n, range(n))`` is about as cheap as one ``extract``.
    """
    if _accel.AVAILABLE:
        return _accel.extract_batch(states, players)
    n = len(states)
    if n != len(players):
        raise ValueError("states and players must have the same length")
    X = np.zeros((n, NUM_FEATURES), np.float32)
    last: Optional[GameState] = None
    info: Optional[dict] = None
    blocks: Optional[np.ndarray] = None
    for i in range(n):
        s = states[i]
        if s is not last:
            info = _analyse_players(s)
            blocks = _player_blocks(s, info)
            last = s
        _assemble(s, int(players[i]), info, blocks, X[i])  # type: ignore[arg-type]
    return X
