#!/usr/bin/env python3
"""Per-game mechanism metrics from a Catanatron action log (secondary endpoints of the test queue).

    python3 scripts/mechanics.py --proof proof/T1/logs                 # base rates: our seat vs the opponents
    python3 scripts/mechanics.py --proof proof/T1/logs --max-games 300 --json out.json

``game_mechanics(items, board, our_color)`` reads the encoded action log (``[colour, action type, value, result]``,
the format of ``catanbot/bench/catanatron_adapter.encode_log_entry`` and of the proof archive's
``catanbot-actionlog/1`` records) and the encoded board, replays the resource flow exactly (production with
catanatron's bank-depletion rule, setup yield, builds, trades, robber steals, discards, Monopoly, Year of Plenty)
and returns what the user's areas are supposed to change:

* expansion / ports: ``setup_distinct`` (distinct resources our buildings touch after setup: what we can
  produce), ``setup_pips`` per resource, ``first_settle_round`` / ``first_city_round`` (our own turn number of
  the first post-setup settlement / city), ``settle_before_city`` (1 if the first post-setup build of the two is
  a settlement), ``port_settled`` / ``port_round`` / ``port_kind`` (first building on a port node; round 0 =
  setup);
* bank trades: counts at 4:1 / 3:1 / 2:1, ``share_4to1``, cards given;
* player trades (3.3 domestic trading): offers made / received, trades done, cards gained;
* robber: moves by us, moves onto the VP leader's hexes (``robber_on_leader``) and the leader's pips blocked,
  rolls with the robber on our hexes, production cards we lost to the robber (``cards_lost_block``) and our
  income share under the robber (``robber_income_share`` = cards lost to blocks / (cards produced + lost));
* expansion / roads: roads and settlements built after setup, ``distinct_produced`` (resource types our rolls
  paid during the game), Longest Road held at the end;
* knights: played, still held at the end (``knights_held_end``), Largest Army held at the end;
* steals, discards, Monopoly haul, Year of Plenty plays, dev cards bought / held for 10+ player-turns, knights;
* titles at game end (Longest Road / Largest Army; from ``final`` when given).

It never changes a game or a verdict threshold: ``ablate_catanatron.py --mech`` stores the dict per game
(``record["mech"]``, not part of the arm key) and ``scripts/run_queue.py`` reports per-arm means / medians and
paired differences over a look's prefix (:func:`summarize`), drives the 'milder' fallback's overshoot rule
(:func:`overshoot`) and each port row's mechanism target.  ``hand_check`` compares the replayed hands with the
logged final hands when the log carries them (the proof archive does): a mismatch means the replay is wrong.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
RI = {r: i for i, r in enumerate(RESOURCES)}
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
COST = {"road": (1, 1, 0, 0, 0), "settlement": (1, 1, 1, 1, 0), "city": (0, 0, 0, 2, 3), "dev": (0, 0, 1, 1, 1)}
HELD_TURNS = 10
# Land hexes of catanatron's BASE map (cube coordinates) -> their six node ids.  The topology walk is the same on
# catanatron 3.2.1 and 3.3 (tests/test_mechanics.py rebuilds it from catanatron and compares).
HEX_NODES: Dict[Tuple[int, int, int], Tuple[int, ...]] = {
    (0, 0, 0): (0, 1, 2, 3, 4, 5), (1, -1, 0): (1, 2, 6, 7, 8, 9), (0, -1, 1): (2, 3, 9, 10, 11, 12),
    (-1, 0, 1): (3, 4, 12, 13, 14, 15), (-1, 1, 0): (4, 5, 15, 16, 17, 18), (0, 1, -1): (0, 5, 16, 19, 20, 21),
    (1, 0, -1): (0, 1, 6, 20, 22, 23), (2, -2, 0): (7, 8, 24, 25, 26, 27), (1, -2, 1): (8, 9, 10, 27, 28, 29),
    (0, -2, 2): (10, 11, 29, 30, 31, 32), (-1, -1, 2): (11, 12, 13, 32, 33, 34),
    (-2, 0, 2): (13, 14, 34, 35, 36, 37), (-2, 1, 1): (14, 15, 17, 37, 38, 39),
    (-2, 2, 0): (17, 18, 39, 40, 41, 42), (-1, 2, -1): (16, 18, 21, 40, 43, 44),
    (0, 2, -2): (19, 21, 43, 45, 46, 47), (1, 1, -2): (19, 20, 22, 46, 48, 49),
    (2, 0, -2): (22, 23, 49, 50, 51, 52), (2, -1, -1): (6, 7, 23, 24, 52, 53),
}
#: metrics whose per-arm summary is a median (turn numbers); everything else is a mean
MEDIAN_METRICS = ("first_settle_round", "first_city_round", "port_round")
#: the expansion / diversity / port readout (the coordinator's list) shown first in reports
KEY_METRICS = ("setup_distinct", "first_settle_round", "first_city_round", "settle_before_city", "port_settled",
               "share_4to1", "roads_built", "longest_road", "knights_played", "knights_held_end", "largest_army",
               "robber_income_share")


class Board:
    """Tiles, numbers and port nodes of an encoded board (``catanatron_adapter.encode_board``)."""

    def __init__(self, doc: Dict[str, Any]):
        self.tiles: Dict[Tuple[int, int, int], Tuple[Optional[str], Optional[int]]] = {}
        for t in doc.get("tiles", []):
            if t.get("t") == "land":
                self.tiles[tuple(t["c"])] = (t.get("resource"), t.get("number"))
        self.desert = next((c for c, (r, _) in self.tiles.items() if r is None), None)
        self.node_hexes: Dict[int, List[Tuple[int, int, int]]] = {}
        for c in self.tiles:
            for n in HEX_NODES[c]:
                self.node_hexes.setdefault(n, []).append(c)
        self.port_of: Dict[int, str] = {}
        for kind, nodes in (doc.get("ports") or {}).items():
            for n in nodes:
                self.port_of[int(n)] = kind


def _longest_road(edges: Sequence[Tuple[int, int]], blocked: set) -> int:
    """Longest trail over ``edges`` (distinct edges) that does not pass through a node in ``blocked``."""
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for i, (a, b) in enumerate(edges):
        adj.setdefault(a, []).append((b, i))
        adj.setdefault(b, []).append((a, i))
    best = 0

    def dfs(node: int, used: int, length: int, start: bool) -> None:
        nonlocal best
        best = max(best, length)
        if not start and node in blocked:
            return
        for nxt, i in adj.get(node, ()):
            if not used & (1 << i):
                dfs(nxt, used | (1 << i), length + 1, False)

    for n in adj:
        dfs(n, 0, 0, True)
    return best


class _Tracker:
    """Replays one game's resource flow and board for every colour."""

    def __init__(self, board: Board, colors: Sequence[str]):
        self.b = board
        self.colors = list(colors)
        self.hand = {c: [0] * 5 for c in colors}
        self.bank = [19] * 5
        self.buildings: Dict[int, Tuple[str, str]] = {}      # node -> (colour, "S" | "C")
        self.roads: Dict[str, List[Tuple[int, int]]] = {c: [] for c in colors}
        self.robber = board.desert
        self.knights = {c: 0 for c in colors}
        self.army: Optional[str] = None
        self.road_len = {c: 0 for c in colors}
        self.road_holder: Optional[str] = None
        self.setup = True
        self.free_roads = 0
        self.free_road_color: Optional[str] = None

    # -- public VP (settlements, cities, titles; hidden VP cards are not public) --------------------------------
    def public_vp(self, c: str) -> int:
        vp = sum(1 if k == "S" else 2 for (cc, k) in self.buildings.values() if cc == c)
        return vp + 2 * (self.road_holder == c) + 2 * (self.army == c)

    def _update_roads(self) -> None:
        for c in self.colors:
            blocked = {n for n, (cc, _) in self.buildings.items() if cc != c}
            self.road_len[c] = _longest_road(self.roads[c], blocked)
        best = max(self.road_len.values()) if self.road_len else 0
        holder = self.road_holder
        if holder is not None and self.road_len[holder] >= best and self.road_len[holder] >= 5:
            return
        tops = [c for c in self.colors if self.road_len[c] == best]
        self.road_holder = tops[0] if best >= 5 and len(tops) == 1 else (None if best < 5 else holder
                                                                          if holder in tops else None)

    def pay(self, c: str, cost: Sequence[int]) -> None:
        for i, n in enumerate(cost):
            self.hand[c][i] -= n
            self.bank[i] += n

    def give(self, c: str, i: int, n: int = 1) -> None:
        self.hand[c][i] += n
        self.bank[i] -= n

    def production(self, number: int) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Payout of a roll (catanatron's rule: a resource the bank cannot pay in full is paid to nobody)
        and what each colour lost to the robber."""
        want: Dict[str, List[int]] = {}
        blocked: Dict[str, List[int]] = {}
        totals = [0] * 5
        for coord, (res, num) in self.b.tiles.items():
            if num != number or res is None:
                continue
            for n in HEX_NODES[coord]:
                bld = self.buildings.get(n)
                if bld is None:
                    continue
                k = 1 if bld[1] == "S" else 2
                if coord == self.robber:
                    blocked.setdefault(bld[0], [0] * 5)[RI[res]] += k
                    continue
                want.setdefault(bld[0], [0] * 5)[RI[res]] += k
                totals[RI[res]] += k
        pay = {c: [x if totals[i] <= self.bank[i] else 0 for i, x in enumerate(v)] for c, v in want.items()}
        return pay, blocked


def _res_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [x for x in v if x is not None]


def game_mechanics(items: Sequence[Sequence[Any]], board: Dict[str, Any], our_color: str,
                   final: Optional[Dict[str, Any]] = None, colors: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Mechanism metrics of ``our_color`` in one game (see the module docstring).

    ``items``: encoded log entries ``[colour, type, value, result]``; ``board``: encoded board;
    ``final``: the logged final state summary (``state_summary``), used for titles and the hand check."""
    b = Board(board)
    if colors is None:
        colors = []
        for it in items:
            if it[0] not in colors:
                colors.append(it[0])
    tr = _Tracker(b, colors)
    me = our_color
    m: Dict[str, Any] = {k: 0 for k in (
        "bank_4to1", "bank_3to1", "bank_2to1", "bank_trades", "bank_cards_given", "offers_made", "offers_received",
        "trades_done", "trade_cards_gained", "robber_moves", "robber_on_leader", "robber_leader_pips",
        "robber_on_us_rolls", "cards_lost_block", "stolen_by_us", "stolen_from_us", "discarded",
        "monopoly_plays", "monopoly_haul", "yop_plays", "dev_bought", "dev_held10", "knights_played",
        "roads_built", "settlements_built", "cards_produced")}
    produced_types = set()
    my_rolls = 0
    first_settle = first_city = None      # (round, action index)
    port_round: Optional[int] = None
    port_kind: Optional[str] = None
    setup_nodes: List[int] = []
    devs: List[Tuple[str, int]] = []      # (card, player-turn index bought) of our unplayed cards
    held_ages: List[int] = []
    player_turns = 0
    settlements_placed = {c: 0 for c in colors}
    pending_offer: Optional[str] = None
    for idx, it in enumerate(items):
        c, t, v, res = it[0], it[1], it[2], (it[3] if len(it) > 3 else None)
        if t != "BUILD_ROAD" and not (t == "PLAY_ROAD_BUILDING"):
            if tr.free_road_color == c:
                tr.free_roads = 0
        if t == "ROLL":
            if tr.setup:
                tr.setup = False
                setup_nodes = [n for n, (cc, _) in tr.buildings.items() if cc == me]
            dice = res if res is not None else v
            number = int(dice[0]) + int(dice[1])
            if c == me:
                my_rolls += 1
            if tr.robber is not None and any(tr.buildings.get(n, ("", ""))[0] == me
                                              for n in HEX_NODES.get(tr.robber, ())):
                if b.tiles.get(tr.robber, (None, None))[1] is not None:
                    m["robber_on_us_rolls"] += 1
            if number != 7:
                pay, blocked = tr.production(number)
                for cc, vec in pay.items():
                    for i, x in enumerate(vec):
                        if x:
                            tr.give(cc, i, x)
                            if cc == me:
                                m["cards_produced"] += x
                                produced_types.add(i)
                m["cards_lost_block"] += sum(blocked.get(me, [0] * 5))
        elif t == "BUILD_SETTLEMENT":
            node = int(v)
            tr.buildings[node] = (c, "S")
            settlements_placed[c] += 1
            if tr.setup:
                if settlements_placed[c] == 2:
                    for coord in b.node_hexes.get(node, ()):
                        r = b.tiles[coord][0]
                        if r is not None:
                            tr.give(c, RI[r])
            else:
                tr.pay(c, COST["settlement"])
                if c == me:
                    m["settlements_built"] += 1
                if c == me and first_settle is None:
                    first_settle = (my_rolls, idx)
            if c == me and port_round is None and node in b.port_of:
                port_round = 0 if tr.setup else my_rolls
                port_kind = b.port_of[node]
            tr._update_roads()
        elif t == "BUILD_CITY":
            node = int(v)
            tr.buildings[node] = (c, "C")
            tr.pay(c, COST["city"])
            if c == me and first_city is None:
                first_city = (my_rolls, idx)
        elif t == "BUILD_ROAD":
            e = tuple(int(x) for x in v)
            tr.roads[c].append((e[0], e[1]))
            if c == me and not tr.setup:
                m["roads_built"] += 1
            if tr.setup:
                pass
            elif tr.free_roads > 0 and tr.free_road_color == c:
                tr.free_roads -= 1
            else:
                tr.pay(c, COST["road"])
            tr._update_roads()
        elif t == "BUY_DEVELOPMENT_CARD":
            tr.pay(c, COST["dev"])
            card = res if res is not None else v
            if c == me:
                m["dev_bought"] += 1
                devs.append((str(card), player_turns))
        elif t in ("PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING"):
            card = {"PLAY_KNIGHT_CARD": "KNIGHT", "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY",
                    "PLAY_MONOPOLY": "MONOPOLY", "PLAY_ROAD_BUILDING": "ROAD_BUILDING"}[t]
            if c == me:
                for j, (cd, t0) in enumerate(devs):
                    if cd == card:
                        held_ages.append(player_turns - t0)
                        devs.pop(j)
                        break
            if t == "PLAY_KNIGHT_CARD":
                tr.knights[c] += 1
                if c == me:
                    m["knights_played"] += 1
                top = tr.knights[c]
                if top >= 3 and (tr.army is None or top > tr.knights[tr.army]):
                    tr.army = c
            elif t == "PLAY_YEAR_OF_PLENTY":
                for r in _res_list(v):
                    tr.give(c, RI[r])
                if c == me:
                    m["yop_plays"] += 1
            elif t == "PLAY_MONOPOLY":
                i = RI[v if isinstance(v, str) else _res_list(v)[0]]
                haul = 0
                for cc in colors:
                    if cc != c:
                        haul += tr.hand[cc][i]
                        tr.hand[c][i] += tr.hand[cc][i]
                        tr.hand[cc][i] = 0
                if c == me:
                    m["monopoly_plays"] += 1
                    m["monopoly_haul"] += haul
            elif t == "PLAY_ROAD_BUILDING":
                tr.free_roads = 2
                tr.free_road_color = c
        elif t == "MOVE_ROBBER":
            coord = tuple(v[0])
            victim = v[1] if len(v) > 1 else None
            stolen = res if res is not None else (v[2] if len(v) > 2 else None)
            if c == me:
                m["robber_moves"] += 1
                opp = [cc for cc in colors if cc != me]
                top = max(tr.public_vp(cc) for cc in opp) if opp else 0
                leaders = {cc for cc in opp if tr.public_vp(cc) == top}
                on_leader = False
                pips = 0
                num = b.tiles.get(coord, (None, None))[1]
                for n in HEX_NODES.get(coord, ()):
                    bld = tr.buildings.get(n)
                    if bld and bld[0] in leaders:
                        on_leader = True
                        pips += PIPS.get(num, 0) * (1 if bld[1] == "S" else 2)
                m["robber_on_leader"] += int(on_leader)
                m["robber_leader_pips"] += pips
            tr.robber = coord
            if victim is not None and stolen is not None:
                i = RI[stolen]
                tr.hand[victim][i] -= 1
                tr.hand[c][i] += 1
                if c == me:
                    m["stolen_by_us"] += 1
                if victim == me:
                    m["stolen_from_us"] += 1
        elif t in ("DISCARD", "DISCARD_RESOURCE"):
            cards = _res_list(res if (t == "DISCARD" and res is not None) else v)
            for r in cards:
                tr.hand[c][RI[r]] -= 1
                tr.bank[RI[r]] += 1
            if c == me:
                m["discarded"] += len(cards)
        elif t == "MARITIME_TRADE":
            give = [x for x in v[:4] if x is not None]
            get = v[4]
            for r in give:
                tr.hand[c][RI[r]] -= 1
                tr.bank[RI[r]] += 1
            tr.give(c, RI[get])
            if c == me:
                m["bank_trades"] += 1
                m["bank_cards_given"] += len(give)
                m[{4: "bank_4to1", 3: "bank_3to1", 2: "bank_2to1"}.get(len(give), "bank_4to1")] += 1
        elif t == "OFFER_TRADE":
            pending_offer = c
            if c == me:
                m["offers_made"] += 1
            else:
                m["offers_received"] += 1
        elif t == "CONFIRM_TRADE":
            vals = list(v)
            give, get, partner = vals[:5], vals[5:10], vals[10]
            for i in range(5):
                tr.hand[c][i] += int(get[i]) - int(give[i])
                tr.hand[partner][i] += int(give[i]) - int(get[i])
            if me in (c, partner):
                m["trades_done"] += 1
                m["trade_cards_gained"] += int(sum(get)) if c == me else int(sum(give))
        elif t == "END_TURN":
            player_turns += 1
    # ---- per-game readout ------------------------------------------------------------------------------------
    if tr.setup:
        setup_nodes = [n for n, (cc, _) in tr.buildings.items() if cc == me]
    pips = [0] * 5
    for n in setup_nodes:
        for coord in b.node_hexes.get(n, ()):
            r, num = b.tiles[coord]
            if r is not None and num is not None:
                pips[RI[r]] += PIPS.get(int(num), 0)
    m["setup_distinct"] = sum(1 for x in pips if x > 0)
    m["setup_pips"] = {r: pips[i] for i, r in enumerate(RESOURCES)}
    m["first_settle_round"] = first_settle[0] if first_settle else None
    m["first_city_round"] = first_city[0] if first_city else None
    if first_settle and first_city:
        m["settle_before_city"] = int(first_settle[1] < first_city[1])
    elif first_settle or first_city:
        m["settle_before_city"] = int(first_settle is not None)
    else:
        m["settle_before_city"] = None
    m["port_settled"] = int(port_round is not None)
    m["port_round"] = port_round
    m["port_kind"] = port_kind
    m["share_4to1"] = (m["bank_4to1"] / m["bank_trades"]) if m["bank_trades"] else None
    m["robber_leader_share"] = (m["robber_on_leader"] / m["robber_moves"]) if m["robber_moves"] else None
    m["knights_held_end"] = sum(1 for cd, _ in devs if cd == "KNIGHT")
    m["distinct_produced"] = len(produced_types)
    lost = m["cards_lost_block"]
    m["robber_income_share"] = (lost / (lost + m["cards_produced"])) if (lost + m["cards_produced"]) else None
    held_ages.extend(player_turns - t0 for _, t0 in devs)
    m["dev_held10"] = sum(1 for a in held_ages if a >= HELD_TURNS)
    m["rounds"] = my_rolls
    lr = la = None
    hand_check = None
    if final:
        for p in final.get("players", []):
            if p.get("color") == me:
                lr, la = int(bool(p.get("has_longest_road"))), int(bool(p.get("has_largest_army")))
        hand_check = all(tr.hand[p["color"]] == [int(p["resources"][r]) for r in RESOURCES]
                         for p in final.get("players", []) if p.get("color") in tr.hand)
    else:
        lr, la = int(tr.road_holder == me), int(tr.army == me)
    m["longest_road"] = lr
    m["largest_army"] = la
    m["hand_check"] = hand_check
    return m


# ---------------------------------------------------------------------------
# Summaries over records / pairs
# ---------------------------------------------------------------------------
def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, (dict, list, str)):
        return None
    if isinstance(x, bool):
        return float(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _mean_se(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    n = len(xs)
    if not n:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def summarize(pairs: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]], metric: str) -> Dict[str, Any]:
    """Per-arm summary of ``metric`` over pairs of game records (records without ``mech`` are excluded) and the
    paired difference ``m_c - m_d`` with its se over the seeds where both are defined.  Turn-number metrics
    (:data:`MEDIAN_METRICS`) also get per-arm medians."""
    cs, ds, diffs = [], [], []
    for c, d in pairs:
        mc, md = (c.get("mech") or {}), (d.get("mech") or {})
        if not c.get("mech") or not d.get("mech"):
            continue
        xc, xd = _num(mc.get(metric)), _num(md.get(metric))
        if xc is not None:
            cs.append(xc)
        if xd is not None:
            ds.append(xd)
        if xc is not None and xd is not None:
            diffs.append(xc - xd)
    mc_, _ = _mean_se(cs)
    md_, _ = _mean_se(ds)
    dm, dse = _mean_se(diffs)
    out = {"metric": metric, "n_cand": len(cs), "n_def": len(ds), "cand": mc_, "def": md_, "diff": dm,
           "diff_se": dse, "n_pairs": len(diffs)}
    if metric in MEDIAN_METRICS:
        out["cand_median"] = statistics.median(cs) if cs else None
        out["def_median"] = statistics.median(ds) if ds else None
    return out


def summarize_all(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
                  metrics: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    pairs = list(pairs)
    if metrics is None:
        keys = set()
        for c, d in pairs:
            for r in (c, d):
                for k, v in (r.get("mech") or {}).items():
                    if _num(v) is not None or v is None:
                        keys.add(k)
        metrics = [k for k in KEY_METRICS if k in keys] + sorted(k for k in keys if k not in KEY_METRICS
                                                                   and k != "hand_check")
    return {k: summarize(pairs, k) for k in metrics}


def overshoot(summary: Dict[str, Any], direction: str, win_delta: Optional[float]) -> bool:
    """The 'milder' fallback's trigger: the mechanism moved in the declared ``direction`` ('+' / '-') by more
    than 2 paired se while the win-rate delta is <= 0 (the change did what it meant to do, too much)."""
    d, se = summary.get("diff"), summary.get("diff_se")
    if d is None or se is None or se <= 0 or win_delta is None or win_delta > 0:
        return False
    sign = 1.0 if direction.strip() in ("+", "up", "increase") else -1.0
    return sign * d > 2.0 * se


# ---------------------------------------------------------------------------
# Proof logs
# ---------------------------------------------------------------------------
def iter_proof_games(paths: Sequence[str], max_games: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    n = 0
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                yield doc
                n += 1
                if max_games is not None and n >= max_games:
                    return


def proof_game_metrics(doc: Dict[str, Any], seat: Optional[int] = None) -> Dict[str, Any]:
    """Metrics of one archived game for ``seat`` (default: our first seat)."""
    colors = list(doc["colors"])
    ours = json.loads(doc["our_seats"]) if isinstance(doc.get("our_seats"), str) else doc.get("our_seats", [0])
    s = ours[0] if seat is None else seat
    return game_mechanics(doc["actions"], doc["board"], colors[s], final=(doc.get("final") or {}).get("state"),
                          colors=colors)


def base_rates(paths: Sequence[str], max_games: Optional[int] = None) -> Dict[str, Any]:
    ours: List[Dict[str, Any]] = []
    opp: List[Dict[str, Any]] = []
    checks = [0, 0]
    for doc in iter_proof_games(paths, max_games):
        colors = list(doc["colors"])
        seats = json.loads(doc["our_seats"]) if isinstance(doc.get("our_seats"), str) else doc.get("our_seats", [0])
        for s in range(len(colors)):
            m = proof_game_metrics(doc, s)
            (ours if s in seats else opp).append(m)
            if m.get("hand_check") is not None:
                checks[0] += 1
                checks[1] += int(bool(m["hand_check"]))
    return {"ours": _agg(ours), "opponents": _agg(opp), "games": len(ours), "hand_checks": checks}


def _agg(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not rows:
        return out
    keys = [k for k in rows[0] if k not in ("hand_check", "port_kind", "setup_pips")]
    for k in keys:
        xs = [_num(r.get(k)) for r in rows]
        xs = [x for x in xs if x is not None]
        if not xs:
            continue
        out[k] = {"mean": sum(xs) / len(xs), "n": len(xs)}
        if k in MEDIAN_METRICS:
            out[k]["median"] = statistics.median(xs)
    tot4 = sum(r["bank_4to1"] for r in rows)
    tot = sum(r["bank_trades"] for r in rows)
    out["pooled_share_4to1"] = tot4 / tot if tot else None
    pips = [0.0] * 5
    for r in rows:
        for i, res in enumerate(RESOURCES):
            pips[i] += r["setup_pips"][res]
    out["setup_pips_mean"] = {res: pips[i] / len(rows) for i, res in enumerate(RESOURCES)}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proof", required=True, help="a proof logs directory (or a glob of *.jsonl.gz files)")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--json", help="write the base rates here")
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(args.proof, "*.jsonl*")) if os.path.isdir(args.proof)
                   else glob.glob(args.proof))
    if not paths:
        print(f"no logs under {args.proof}", file=sys.stderr)
        return 2
    br = base_rates(paths, args.max_games)
    print(f"{br['games']} games from {len(paths)} file(s); replayed hands equal the logged final hands in "
          f"{br['hand_checks'][1]}/{br['hand_checks'][0]} seat-games")
    print(f"{'metric':24} {'ours':>10} {'opponents':>10}")
    for k in list(KEY_METRICS) + ["pooled_share_4to1", "distinct_produced", "settlements_built", "bank_trades",
                                  "robber_moves", "robber_leader_share", "robber_on_us_rolls", "discarded",
                                  "stolen_by_us", "monopoly_haul", "dev_bought", "dev_held10"]:
        a, o = br["ours"].get(k), br["opponents"].get(k)

        def f(x):
            if x is None:
                return "n/a"
            if isinstance(x, dict):
                return f"{x.get('median', x['mean']):.2f}" + ("m" if "median" in x else "")
            return f"{x:.3f}"
        print(f"{k:24} {f(a):>10} {f(o):>10}")
    print("setup pips ours " + " ".join(f"{r[:2]} {v:.2f}" for r, v in br["ours"]["setup_pips_mean"].items())
          + " | opponents " + " ".join(f"{r[:2]} {v:.2f}" for r, v in br["opponents"]["setup_pips_mean"].items()))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(br, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
