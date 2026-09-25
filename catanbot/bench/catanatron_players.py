"""Strong opponents that run *inside* the catanatron engine.

Pure catanatron: this module imports nothing from ``catanbot`` so the players
can be dropped into any ``catanatron.game.Game`` (they are in the spirit of
the ``ValueFunctionPlayer`` / ``AlphaBetaPlayer`` of the Catanatron project,
which are not shipped on PyPI).

Contents
--------
* :func:`value_function` - cheap hand-written evaluation of a game from one
  colour's perspective (actual VP, robber-aware production, diversity,
  reachable settlement spots, longest road, largest army, dev cards, hand
  size / discard risk, progress toward the next build, ports matching
  production, and a relative term against the best opponent).
* :class:`ValueFunctionPlayer` - 1-ply greedy on the value function with
  exact expectations for chance actions (dev-card draws, robber steals) and a
  one-step look-ahead for actions that only pay off with a follow-up
  (maritime trades, Year of Plenty, Monopoly, knights, Road Building).
* :class:`AlphaBetaPlayer` - depth-2 (turn-level) expectimax: it searches
  its own action *sequences* for the current turn (beam-pruned, END_TURN
  included), then the next opponent's reply after a chance node over the
  most likely dice sums, with alpha pruning of the opponent node and a node
  budget (number of game copies) per decision.
* an opening book for the initial placements (production x diversity x
  scarcity, road toward the best free spot), robber targeting, discard
  planning (:func:`plan_discard`) and :func:`play_game`, a ``Game.play``
  replacement that lets players choose their discards.

Notes on the engine (catanatron 3.2.1)
--------------------------------------
* ``discard_possibilities`` only exposes ``Action(color, DISCARD, None)``
  (a random discard) and ``Game.play`` validates actions, so a player cannot
  choose its discard through ``decide``; :func:`play_game` applies the
  chosen discard with ``validate_action=False`` instead.
* All hands and the dev-card deck are visible in ``state``; the players below
  use the hand *composition* of a robbed player for the exact steal
  expectation (the stock ``VictoryPointPlayer`` sees the same state) but
  never peek at the order of the dev-card deck: dev-card purchases are
  averaged over the card types still in the deck with their starting
  probabilities.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.decks import starting_devcard_proba
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    ROAD,
    SETTLEMENT,
    Action,
    ActionPrompt,
    ActionType,
)
from catanatron.models.map import DICE_PROBAS
from catanatron.models.player import Color, Player
from catanatron.state_functions import player_key

__all__ = [
    "DEFAULT_WEIGHTS",
    "AlphaBetaPlayer",
    "ValueFunctionPlayer",
    "choose_initial_road",
    "choose_initial_settlement",
    "get_map_tables",
    "opening_spot_score",
    "plan_discard",
    "play_game",
    "robber_candidates",
    "value_function",
]

RESOURCE_INDEX: Dict[str, int] = {r: i for i, r in enumerate(RESOURCES)}
WOOD_I, BRICK_I, SHEEP_I, WHEAT_I, ORE_I = 0, 1, 2, 3, 4

ROAD_COST = (1, 1, 0, 0, 0)
SETTLEMENT_COST = (1, 1, 1, 1, 0)
CITY_COST = (0, 0, 0, 2, 3)
DEV_COST = (0, 0, 1, 1, 1)

# One representative dice pair per sum (the engine only uses the sum).
DICE_PAIR = {2: (1, 1), 3: (1, 2), 4: (2, 2), 5: (2, 3), 6: (3, 3), 7: (3, 4),
             8: (4, 4), 9: (4, 5), 10: (5, 5), 11: (5, 6), 12: (6, 6)}
# Dice sums ordered by probability (7 first; ties broken low-first).
DICE_SUMS_BY_PROBA = sorted(range(2, 13), key=lambda s: (-DICE_PROBAS[s], s))
# A "null roll": the engine only uses the dice sum, and no tile is numbered 0,
# so this marks the player as having rolled without any payout or robber.
NULL_DICE = (0, 0)

DEFAULT_WEIGHTS: Dict[str, object] = {
    "vp": 12.0,             # per actual victory point (heavily weighted)
    "win": 1000.0,          # terminal bonus / penalty
    "production": 12.0,     # per unit of probability-weighted production
    "resource_weights": (1.0, 1.05, 0.85, 1.2, 1.15),  # wood brick sheep wheat ore
    "diversity": 0.8,       # per distinct resource produced
    "spot_now": 1.2,        # per settlement spot buildable now (capped at 3)
    "spot_one_road": 0.5,   # per spot reachable with one road (capped at 4)
    "best_spot": 5.0,       # x production of the best reachable spot
    "road_length": 0.35,    # per road of the longest road
    "road_race": 1.5,       # bonus when within one road of taking Longest Road
    "knights": 0.9,         # per knight played (largest army progress)
    "army_race": 1.5,       # bonus when one knight from taking Largest Army
    "dev_card": 0.9,        # per non-VP development card in hand
    "card": 0.8,            # per resource card in hand
    "over_limit": 1.6,      # per card above the discard limit (discard risk)
    "progress": 4.0,        # x (progress toward the next build)^2
    "port": 6.0,            # x own production of a 2:1 port's resource
    "port_generic": 0.8,    # per 3:1 port
    "opponent": 0.5,        # relative term: minus this x the best opponent's score
}


def _merge_weights(weights: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not weights:
        return DEFAULT_WEIGHTS
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(weights)
    return merged


# --------------------------------------------------------------------------
# Per-player state keys (player_state is a flat dict prefixed by P0_, P1_ ...)
# --------------------------------------------------------------------------
class _Keys:
    __slots__ = ("vp", "vvp", "wood", "brick", "sheep", "wheat", "ore", "road_len", "knights",
                 "knight_h", "yop_h", "mono_h", "rb_h", "vp_h", "settle_av", "city_av",
                 "roads_av", "has_road", "has_army", "has_rolled", "played_dev")

    def __init__(self, idx: int):
        p = f"P{idx}_"
        self.vp = p + "ACTUAL_VICTORY_POINTS"
        self.vvp = p + "VICTORY_POINTS"
        self.wood, self.brick, self.sheep = p + "WOOD_IN_HAND", p + "BRICK_IN_HAND", p + "SHEEP_IN_HAND"
        self.wheat, self.ore = p + "WHEAT_IN_HAND", p + "ORE_IN_HAND"
        self.road_len = p + "LONGEST_ROAD_LENGTH"
        self.knights = p + "PLAYED_KNIGHT"
        self.knight_h, self.yop_h = p + "KNIGHT_IN_HAND", p + "YEAR_OF_PLENTY_IN_HAND"
        self.mono_h, self.rb_h = p + "MONOPOLY_IN_HAND", p + "ROAD_BUILDING_IN_HAND"
        self.vp_h = p + "VICTORY_POINT_IN_HAND"
        self.settle_av, self.city_av = p + "SETTLEMENTS_AVAILABLE", p + "CITIES_AVAILABLE"
        self.roads_av = p + "ROADS_AVAILABLE"
        self.has_road, self.has_army = p + "HAS_ROAD", p + "HAS_ARMY"
        self.has_rolled = p + "HAS_ROLLED"
        self.played_dev = p + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"


_KEYS = [_Keys(i) for i in range(4)]


def _hand(ps: dict, k: _Keys) -> List[int]:
    return [ps[k.wood], ps[k.brick], ps[k.sheep], ps[k.wheat], ps[k.ore]]


# --------------------------------------------------------------------------
# Per-map production tables (cached per CatanMap object)
# --------------------------------------------------------------------------
class MapTables:
    """Static per-map lookups: node production, adjacency, ports, scarcity."""

    __slots__ = ("map", "node_tiles", "node_vec", "node_total", "neighbors", "node_port",
                 "tile_nodes", "tile_proba", "board_vec", "scarcity")

    def __init__(self, catan_map):
        self.map = catan_map
        node_tiles: Dict[int, List[Tuple[tuple, int, float]]] = {}
        node_vec: Dict[int, List[float]] = {}
        board_vec = [0.0] * 5
        tile_nodes = {}
        tile_proba = {}
        for coord, tile in catan_map.land_tiles.items():
            nodes = tuple(tile.nodes.values())
            tile_nodes[coord] = nodes
            for node in nodes:
                node_tiles.setdefault(node, [])
                node_vec.setdefault(node, [0.0] * 5)
            if tile.resource is None:
                tile_proba[coord] = 0.0
                continue
            p = DICE_PROBAS[tile.number]
            tile_proba[coord] = p
            ri = RESOURCE_INDEX[tile.resource]
            board_vec[ri] += p
            for node in nodes:
                node_tiles[node].append((coord, ri, p))
                node_vec[node][ri] += p
        neighbors: Dict[int, set] = {n: set() for n in node_vec}
        for tile in catan_map.land_tiles.values():
            for a, b in tile.edges.values():
                neighbors[a].add(b)
                neighbors[b].add(a)
        node_port = {}
        for resource, nodes in catan_map.port_nodes.items():
            for n in nodes:
                node_port[n] = resource  # None means 3:1
        mean = sum(board_vec) / 5.0
        self.node_tiles = node_tiles
        self.node_vec = node_vec
        self.node_total = {n: sum(v) for n, v in node_vec.items()}
        self.neighbors = {n: tuple(sorted(s)) for n, s in neighbors.items()}
        self.node_port = node_port
        self.tile_nodes = tile_nodes
        self.tile_proba = tile_proba
        self.board_vec = board_vec
        self.scarcity = [min(1.6, max(0.6, mean / v)) if v > 0 else 1.6 for v in board_vec]


_MAP_CACHE: Dict[int, MapTables] = {}


def get_map_tables(catan_map) -> MapTables:
    """Per-node production tables for a map (built once per CatanMap object)."""
    tables = _MAP_CACHE.get(id(catan_map))
    if tables is None or tables.map is not catan_map:
        if len(_MAP_CACHE) > 64:
            _MAP_CACHE.clear()
        tables = MapTables(catan_map)
        _MAP_CACHE[id(catan_map)] = tables
    return tables


# --------------------------------------------------------------------------
# Value function
# --------------------------------------------------------------------------
_MISSING = object()


def _production(state, color, tables: MapTables, robber) -> List[float]:
    """Probability-weighted production per resource, ignoring the robber's tile."""
    prod = [0.0] * 5
    node_tiles = tables.node_tiles
    bbc = state.buildings_by_color[color]
    for node in bbc[SETTLEMENT]:
        for coord, ri, p in node_tiles[node]:
            if coord != robber:
                prod[ri] += p
    for node in bbc[CITY]:
        for coord, ri, p in node_tiles[node]:
            if coord != robber:
                prod[ri] += 2.0 * p
    return prod


def _reachable_spots(state, color, tables: MapTables):
    """(spots buildable now, spots reachable with one more road) as sets of nodes."""
    board = state.board
    components = board.connected_components.get(color, ())
    if not components:
        return set(), set()
    comp_nodes = set().union(*components)
    buildable = board.board_buildable_ids
    now = {n for n in comp_nodes if n in buildable}
    one_road = set()
    roads = board.roads
    neighbors = tables.neighbors
    buildings = board.buildings
    for n in comp_nodes:
        owner = buildings.get(n)
        if owner is not None and owner[0] != color:
            continue  # cannot expand through an enemy building
        for m in neighbors[n]:
            if m in buildable and m not in comp_nodes and (n, m) not in roads:
                one_road.add(m)
    return now, one_road


def _score(state, color, tables: MapTables, w: dict, full: bool) -> float:
    ps = state.player_state
    k = _KEYS[state.color_to_index[color]]
    board = state.board
    robber = board.robber_coordinate
    prod = _production(state, color, tables, robber)
    rw = w["resource_weights"]
    production = 0.0
    kinds = 0
    for i in range(5):
        p = prod[i]
        if p > 0.0:
            production += p * rw[i]
            kinds += 1
    score = w["vp"] * ps[k.vp] + w["production"] * production + w["diversity"] * kinds

    road_len = ps[k.road_len]
    score += w["road_length"] * road_len
    knights = ps[k.knights]
    score += w["knights"] * knights
    devs = ps[k.knight_h] + ps[k.yop_h] + ps[k.mono_h] + ps[k.rb_h]
    score += w["dev_card"] * devs

    hand = _hand(ps, k)
    n_cards = hand[0] + hand[1] + hand[2] + hand[3] + hand[4]
    score += w["card"] * n_cards
    if n_cards > state.discard_limit:
        score -= w["over_limit"] * (n_cards - state.discard_limit)
    if not full:
        return score

    # award races
    if not ps[k.has_road] and road_len >= 4 and road_len + 1 > board.road_length:
        score += w["road_race"]
    if not ps[k.has_army] and knights >= 2 and ps[k.knight_h] > 0:
        army_size = max((ps[_KEYS[i].knights] for i in range(len(state.colors))), default=0)
        if knights + 1 > army_size:
            score += w["army_race"]

    # expansion: settlement spots reachable now / with one road
    settle_av = ps[k.settle_av] > 0
    now, one_road = _reachable_spots(state, color, tables)
    node_total = tables.node_total
    if settle_av:
        best_now = max((node_total[s] for s in now), default=0.0)
        best_one = max((node_total[s] for s in one_road), default=0.0)
        score += w["spot_now"] * min(len(now), 3) + w["spot_one_road"] * min(len(one_road), 4)
        score += w["best_spot"] * max(best_now, 0.8 * best_one)

    # ports matching production
    node_port = tables.node_port
    bbc = state.buildings_by_color[color]
    for node in bbc[SETTLEMENT]:
        port = node_port.get(node, _MISSING)
        if port is _MISSING:
            continue
        score += w["port_generic"] if port is None else w["port"] * prod[RESOURCE_INDEX[port]]
    for node in bbc[CITY]:
        port = node_port.get(node, _MISSING)
        if port is _MISSING:
            continue
        score += w["port_generic"] if port is None else w["port"] * prod[RESOURCE_INDEX[port]]

    # progress toward the next build (convex so nearly-complete builds are kept)
    best = 0.0
    wp = w["progress"]
    if ps[k.city_av] > 0 and bbc[SETTLEMENT]:
        prog = (min(hand[3], 2) + min(hand[4], 3)) / 5.0
        best = wp * prog * prog
    if settle_av and (now or one_road):
        if now:
            prog = (min(hand[0], 1) + min(hand[1], 1) + min(hand[2], 1) + min(hand[3], 1)) / 4.0
            cand = wp * prog * prog
        else:
            prog = (min(hand[0], 2) + min(hand[1], 2) + min(hand[2], 1) + min(hand[3], 1)) / 6.0
            cand = 0.9 * wp * prog * prog
        if cand > best:
            best = cand
    if state.development_listdeck:
        prog = (min(hand[2], 1) + min(hand[3], 1) + min(hand[4], 1)) / 3.0
        cand = 0.1 * wp * prog * prog
        if cand > best:
            best = cand
    score += best
    return score


def value_function(game: Game, color: Color, weights: Optional[Dict[str, object]] = None) -> float:
    """Hand-written evaluation of ``game`` from ``color``'s perspective (higher is better)."""
    w = _merge_weights(weights)
    state = game.state
    tables = get_map_tables(state.board.map)
    own = _score(state, color, tables, w, True)
    best_opp = float("-inf")
    for other in state.colors:
        if other != color:
            s = _score(state, other, tables, w, False)
            if s > best_opp:
                best_opp = s
    value = own - w["opponent"] * (best_opp if best_opp != float("-inf") else 0.0)
    ps = state.player_state
    target = getattr(game, "vps_to_win", 10)
    if ps[_KEYS[state.color_to_index[color]].vp] >= target:
        value += w["win"]
    else:
        for other in state.colors:
            if other != color and ps[_KEYS[state.color_to_index[other]].vp] >= target:
                value -= w["win"]
                break
    return value


# --------------------------------------------------------------------------
# Opening book
# --------------------------------------------------------------------------
def opening_spot_score(tables: MapTables, board, node: int, own_prod: Sequence[float],
                       resource_weights=DEFAULT_WEIGHTS["resource_weights"]) -> float:
    """Production x diversity x scarcity score of an initial settlement spot."""
    vec = tables.node_vec[node]
    scarcity = tables.scarcity
    total = 0.0
    kinds = 0
    new_kinds = 0
    for ri in range(5):
        p = vec[ri]
        if p <= 0.0:
            continue
        kinds += 1
        if own_prod[ri] <= 0.0:
            new_kinds += 1
        total += p * resource_weights[ri] * scarcity[ri]
    if kinds == 0:
        return 0.0
    score = total * (1.0 + 0.15 * (kinds - 1))
    has_own = any(p > 0.0 for p in own_prod)
    if has_own:
        score *= 1.0 + 0.2 * new_kinds
        union = [own_prod[i] + vec[i] for i in range(5)]
        if union[WOOD_I] > 0 and union[BRICK_I] > 0 and union[SHEEP_I] > 0 and union[WHEAT_I] > 0:
            score *= 1.15
        if union[WHEAT_I] > 0 and union[ORE_I] > 0:
            score *= 1.08
    port = tables.node_port.get(node, _MISSING)
    if port is not _MISSING:
        if port is None:
            score += 0.012
        else:
            score += 0.02 if vec[RESOURCE_INDEX[port]] > 0 else 0.005
    # expansion potential: best free spot two edges away
    neighbors = tables.neighbors
    buildable = board.board_buildable_ids
    near = tables.neighbors[node]
    best_far = 0.0
    for n in near:
        for m in neighbors[n]:
            if m != node and m not in near and m in buildable:
                t = tables.node_total[m]
                if t > best_far:
                    best_far = t
    return score + 0.15 * best_far


def choose_initial_settlement(game: Game, color: Color, playable_actions: Sequence[Action]) -> Action:
    state = game.state
    tables = get_map_tables(state.board.map)
    own = _production(state, color, tables, None)
    best, best_score = playable_actions[0], float("-inf")
    for action in playable_actions:
        if action.action_type != ActionType.BUILD_SETTLEMENT:
            continue
        s = opening_spot_score(tables, state.board, action.value, own)
        if s > best_score:
            best, best_score = action, s
    return best


def choose_initial_road(game: Game, color: Color, playable_actions: Sequence[Action]) -> Action:
    """Road (from the settlement just placed) toward the best free spot two edges away."""
    state = game.state
    board = state.board
    tables = get_map_tables(board.map)
    own = _production(state, color, tables, None)
    settlement = state.buildings_by_color[color][SETTLEMENT][-1]
    buildable = board.board_buildable_ids
    neighbors = tables.neighbors
    best, best_score = playable_actions[0], float("-inf")
    for action in playable_actions:
        if action.action_type != ActionType.BUILD_ROAD:
            continue
        a, b = action.value
        far = b if a == settlement else a
        score = 0.0
        if board.buildings.get(far) is not None:
            score = -1.0
        for m in neighbors[far]:
            if m == settlement or m not in buildable:
                continue
            s = opening_spot_score(tables, board, m, own)
            # a spot three edges away counts a little, so dead ends are avoided
            for q in neighbors[m]:
                if q != far and q in buildable:
                    s = max(s, 0.5 * opening_spot_score(tables, board, q, own))
            if s > score:
                score = s
        if score > best_score:
            best, best_score = action, score
    return best


# --------------------------------------------------------------------------
# Robber targeting, discard planning
# --------------------------------------------------------------------------
def robber_candidates(game: Game, mover: Color, k: int = 4, target: Optional[Color] = None,
                      playable_actions: Optional[Sequence[Action]] = None) -> List[Action]:
    """Top-``k`` MOVE_ROBBER actions for ``mover`` by a cheap heuristic.

    Never a tile with one of the mover's own buildings (unless nothing else is
    available); prefers the tile where the leading opponents lose the most
    production and stealing from the leader.  With ``target`` set, only tiles
    hurting that colour are considered (used for the opponent's reply in the
    search).
    """
    state = game.state
    board = state.board
    tables = get_map_tables(board.map)
    ps = state.player_state
    actions = playable_actions if playable_actions is not None else state.playable_actions
    own_nodes = set(state.buildings_by_color[mover][SETTLEMENT]) | set(state.buildings_by_color[mover][CITY])
    vp_of = {c: ps[_KEYS[state.color_to_index[c]].vp] for c in state.colors}
    min_vp = min(vp_of.values())
    scored = []
    for action in actions:
        if action.action_type != ActionType.MOVE_ROBBER:
            continue
        coord, victim, _ = action.value
        nodes = tables.tile_nodes.get(coord)
        if nodes is None:
            continue
        p = tables.tile_proba[coord]
        loss = 0.0
        hits_target = victim == target
        own_hit = False
        for n in nodes:
            b = board.buildings.get(n)
            if b is None:
                continue
            if b[0] == mover:
                own_hit = True
                continue
            if target is not None and b[0] != target:
                continue
            hits_target = True
            mult = 2.0 if b[1] == CITY else 1.0
            loss += mult * p * (1.0 + 0.12 * (vp_of[b[0]] - min_vp))
        if target is not None and not hits_target:
            continue
        score = loss
        if victim is not None:
            cards = sum(_hand(ps, _KEYS[state.color_to_index[victim]]))
            score += 0.02 * (1.0 + 0.15 * (vp_of[victim] - min_vp)) + 0.003 * min(cards, 6)
        if own_hit:
            score -= 100.0
        scored.append((score, action))
    if not scored:
        return list(actions[:k])
    scored.sort(key=lambda t: -t[0])
    return [a for _, a in scored[:k]]


def plan_discard(game: Game, color: Color, num: Optional[int] = None) -> List[str]:
    """Cards to discard (``len(hand)//2`` by default): keep what the next build needs."""
    state = game.state
    ps = state.player_state
    k = _KEYS[state.color_to_index[color]]
    hand = _hand(ps, k)
    total = sum(hand)
    if num is None:
        num = total // 2
    num = max(0, min(num, total))
    if num == 0:
        return []
    tables = get_map_tables(state.board.map)
    now, one_road = _reachable_spots(state, color, tables)
    targets = []  # (progress, priority, cost)
    if ps[k.city_av] > 0 and state.buildings_by_color[color][SETTLEMENT]:
        targets.append(((min(hand[3], 2) + min(hand[4], 3)) / 5.0, 3, CITY_COST))
    if ps[k.settle_av] > 0 and (now or one_road):
        cost = SETTLEMENT_COST if now else (2, 2, 1, 1, 0)
        targets.append((sum(min(hand[i], cost[i]) for i in range(5)) / sum(cost), 2, cost))
    if state.development_listdeck:
        targets.append(((min(hand[2], 1) + min(hand[3], 1) + min(hand[4], 1)) / 3.0, 1, DEV_COST))
    if ps[k.roads_av] > 0 and not now:
        targets.append(((min(hand[0], 1) + min(hand[1], 1)) / 2.0, 0, ROAD_COST))
    targets.sort(key=lambda t: (-t[0], -t[1]))
    # score every physical card: needed by the first target > second target > value
    rw = DEFAULT_WEIGHTS["resource_weights"]
    cards = []
    for ri in range(5):
        need1 = targets[0][2][ri] if targets else 0
        need2 = targets[1][2][ri] if len(targets) > 1 else 0
        for copy_idx in range(hand[ri]):
            if copy_idx < need1:
                s = 10.0
            elif copy_idx < need2:
                s = 5.0
            else:
                s = rw[ri] * tables.scarcity[ri]
            cards.append((s - 0.01 * copy_idx, ri))
    cards.sort(key=lambda t: t[0])  # least valuable first
    return [RESOURCES[ri] for _, ri in cards[:num]]


def play_game(game: Game, smart_discard: bool = True, accumulators: Iterable = ()) -> Optional[Color]:
    """``Game.play`` replacement: players with ``choose_discard(game)`` pick their discards.

    Everything else goes through ``Game.play_tick`` (validated actions).  With
    ``smart_discard=False`` this is exactly ``Game.play``.
    """
    accumulators = list(accumulators)
    if not smart_discard:
        return game.play(accumulators=accumulators)
    for acc in accumulators:
        acc.before(game.copy())
    while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
        state = game.state
        player = state.current_player()
        chooser = getattr(player, "choose_discard", None)
        if state.current_prompt == ActionPrompt.DISCARD and chooser is not None:
            cards = list(chooser(game))
            action = Action(player.color, ActionType.DISCARD, cards)
            if accumulators:
                snapshot = game.copy()
                for acc in accumulators:
                    acc.step(snapshot, action)
            game.execute(action, validate_action=False)
        else:
            game.play_tick(accumulators=accumulators)
    for acc in accumulators:
        acc.after(game.copy())
    return game.winning_color()


# --------------------------------------------------------------------------
# Chance-action expansion
# --------------------------------------------------------------------------
def action_outcomes(game: Game, action: Action) -> List[Tuple[float, Action]]:
    """(probability, fully specified action) pairs for a possibly random action."""
    t = action.action_type
    state = game.state
    if t == ActionType.BUY_DEVELOPMENT_CARD and action.value is None:
        deck = state.development_listdeck
        present = [c for c in DEVELOPMENT_CARDS if c in deck]
        if not present:
            return [(1.0, action)]
        weights = [starting_devcard_proba(c) for c in present]
        z = sum(weights)
        return [(wt / z, Action(action.color, t, c)) for c, wt in zip(present, weights)]
    if t == ActionType.MOVE_ROBBER:
        coord, victim, resource = action.value
        if victim is None or resource is not None:
            return [(1.0, action)]
        hand = _hand(state.player_state, _KEYS[state.color_to_index[victim]])
        total = sum(hand)
        if total <= 0:
            return [(1.0, Action(action.color, t, (coord, None, None)))]
        return [(hand[i] / total, Action(action.color, t, (coord, victim, RESOURCES[i])))
                for i in range(5) if hand[i] > 0]
    return [(1.0, action)]


def _affordable(hand: Sequence[int], cost: Sequence[int]) -> bool:
    return all(hand[i] >= cost[i] for i in range(5))


# --------------------------------------------------------------------------
# Players
# --------------------------------------------------------------------------
class ValueFunctionPlayer(Player):
    """1-ply greedy player on :func:`value_function` (plus fixes of obvious errors).

    * never ends the turn while a settlement / city is affordable and placeable,
    * exact expectations for dev-card draws and robber steals,
    * maritime trades / Year of Plenty / Monopoly are valued together with the
      best build they enable, knights with the best robber move, Road Building
      with the two best free roads,
    * knights are played before rolling when the robber blocks own production
      or when the knight takes Largest Army,
    * initial placements from the opening book (``opening_book=True``),
    * :meth:`choose_discard` keeps the cards of the next build (used by
      :func:`play_game`; the engine's own loop discards at random).
    """

    def __init__(self, color: Color, weights: Optional[Dict[str, object]] = None,
                 opening_book: bool = True, robber_k: int = 6, is_bot: bool = True):
        super().__init__(color, is_bot)
        self.weights = _merge_weights(weights)
        self.opening_book = opening_book
        self.robber_k = robber_k
        self.nodes = 0  # game copies made during the last decision

    # -- engine hooks -----------------------------------------------------
    def decide(self, game: Game, playable_actions):
        self.nodes = 0
        playable_actions = list(playable_actions)
        if len(playable_actions) == 1:
            return playable_actions[0]
        state = game.state
        prompt = state.current_prompt
        if prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            if self.opening_book:
                return choose_initial_settlement(game, self.color, playable_actions)
            return self._greedy(game, playable_actions)
        if prompt == ActionPrompt.BUILD_INITIAL_ROAD:
            if self.opening_book:
                return choose_initial_road(game, self.color, playable_actions)
            return self._greedy(game, playable_actions)
        if prompt == ActionPrompt.DISCARD:
            return playable_actions[0]  # the engine only exposes the random discard
        if prompt == ActionPrompt.MOVE_ROBBER:
            return self._decide_robber(game, playable_actions)
        if prompt == ActionPrompt.PLAY_TURN and not state.player_state[
            _KEYS[state.color_to_index[self.color]].has_rolled
        ]:
            return self._decide_pre_roll(game, playable_actions)
        return self._decide_turn(game, playable_actions)

    def choose_discard(self, game: Game) -> List[str]:
        return plan_discard(game, self.color)

    # -- helpers ----------------------------------------------------------
    def _copy_exec(self, game: Game, action: Action) -> Game:
        self.nodes += 1
        child = game.copy()
        child.execute(action, validate_action=False)
        return child

    def _value(self, game: Game) -> float:
        return value_function(game, self.color, self.weights)

    def _keys(self, state) -> _Keys:
        return _KEYS[state.color_to_index[self.color]]

    def _decide_turn(self, game: Game, playable_actions: List[Action]) -> Action:
        return self._greedy(game, playable_actions)

    def _greedy(self, game: Game, playable_actions: List[Action]) -> Action:
        has_build = any(a.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY)
                        for a in playable_actions)
        best, best_value = None, float("-inf")
        for action in playable_actions:
            if action.action_type == ActionType.END_TURN and has_build:
                continue  # never end the turn while a settlement / city is buildable
            v = self.evaluate_action(game, action)
            if v > best_value:
                best, best_value = action, v
        return best if best is not None else playable_actions[0]

    def _decide_robber(self, game: Game, playable_actions: List[Action]) -> Action:
        cands = robber_candidates(game, self.color, self.robber_k, playable_actions=playable_actions)
        best, best_value = cands[0], float("-inf")
        for action in cands:
            v = self.expected_value(game, action)
            if v > best_value:
                best, best_value = action, v
        return best

    def _decide_pre_roll(self, game: Game, playable_actions: List[Action]) -> Action:
        roll = next((a for a in playable_actions if a.action_type == ActionType.ROLL), None)
        knight = next((a for a in playable_actions if a.action_type == ActionType.PLAY_KNIGHT_CARD), None)
        if roll is None or knight is None:
            return playable_actions[0]
        state = game.state
        ps = state.player_state
        k = self._keys(state)
        tables = get_map_tables(state.board.map)
        robber = state.board.robber_coordinate
        bbc = state.buildings_by_color[self.color]
        blocked = any(robber in (coord for coord, _, _ in tables.node_tiles[n])
                      for n in list(bbc[SETTLEMENT]) + list(bbc[CITY]))
        takes_army = False
        if not ps[k.has_army] and ps[k.knights] + 1 >= 3:
            army_size = max(ps[_KEYS[i].knights] for i in range(len(state.colors)))
            takes_army = ps[k.knights] + 1 > army_size
        if blocked or takes_army:
            after = self._copy_exec(game, knight)
            robber_action = self._decide_robber(after, after.state.playable_actions)
            if self.expected_value(after, robber_action) > self._value(game):
                return knight
        return roll

    # -- evaluation -------------------------------------------------------
    def expected_value(self, game: Game, action: Action) -> float:
        """Expectation of the value after ``action`` (chance actions averaged exactly)."""
        total = 0.0
        for p, a in action_outcomes(game, action):
            total += p * self._value(self._copy_exec(game, a))
        return total

    def evaluate_action(self, game: Game, action: Action) -> float:
        """1-ply value of ``action`` with the look-aheads described in the class doc."""
        t = action.action_type
        if t == ActionType.END_TURN:
            return self._value(game)
        if t in (ActionType.MARITIME_TRADE, ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY):
            after = self._copy_exec(game, action)
            return max(self._value(after), self._best_follow_up(after))
        if t == ActionType.PLAY_KNIGHT_CARD:
            after = self._copy_exec(game, action)
            cands = robber_candidates(after, self.color, min(3, self.robber_k))
            return max(self.expected_value(after, a) for a in cands)
        if t == ActionType.PLAY_ROAD_BUILDING:
            after = self._copy_exec(game, action)
            for _ in range(2):
                acts = after.state.playable_actions
                if not after.state.is_road_building or not acts:
                    break
                roads = self._road_candidates(after, acts, 3)
                if not roads:
                    break
                best, best_v = None, float("-inf")
                for r in roads:
                    child = self._copy_exec(after, r)
                    v = self._value(child)
                    if v > best_v:
                        best, best_v = child, v
                after = best
            return self._value(after)
        if t == ActionType.MOVE_ROBBER:
            return self.expected_value(game, action)
        if t == ActionType.BUY_DEVELOPMENT_CARD:
            return self.expected_value(game, action)
        return self._value(self._copy_exec(game, action))

    def _best_follow_up(self, game: Game) -> float:
        """Best value of an immediate build after a trade / dev-card play (or -inf)."""
        state = game.state
        if state.current_color() != self.color or state.current_prompt != ActionPrompt.PLAY_TURN:
            return float("-inf")
        tables = get_map_tables(state.board.map)
        best = float("-inf")
        cities = settlements = None
        buy = None
        for a in state.playable_actions:
            at = a.action_type
            if at == ActionType.BUILD_CITY:
                tot = tables.node_total[a.value]
                if cities is None or tot > cities[0]:
                    cities = (tot, a)
            elif at == ActionType.BUILD_SETTLEMENT:
                tot = tables.node_total[a.value]
                if settlements is None or tot > settlements[0]:
                    settlements = (tot, a)
            elif at == ActionType.BUY_DEVELOPMENT_CARD:
                buy = a
        for cand in (cities, settlements):
            if cand is not None:
                best = max(best, self._value(self._copy_exec(game, cand[1])))
        if buy is not None:
            best = max(best, self.expected_value(game, buy))
        return best

    def _road_candidates(self, game: Game, playable_actions: Sequence[Action], k: int) -> List[Action]:
        """Top-``k`` BUILD_ROAD actions by the spots they open (cheap, no copies)."""
        state = game.state
        board = state.board
        tables = get_map_tables(board.map)
        components = board.connected_components.get(self.color, ())
        comp_nodes = set().union(*components) if components else set()
        buildable = board.board_buildable_ids
        neighbors = tables.neighbors
        node_total = tables.node_total
        roads = board.roads
        k_ = self._keys(state)
        long_race = state.player_state[k_.road_len] >= 3
        scored = []
        for a in playable_actions:
            if a.action_type != ActionType.BUILD_ROAD:
                continue
            score = 0.05 + (0.15 if long_race else 0.0)
            for n in a.value:
                if n in comp_nodes:
                    continue
                if n in buildable:
                    score += 1.0 + 3.0 * node_total[n]
                for m in neighbors[n]:
                    if m in buildable and m not in comp_nodes and (n, m) not in roads:
                        score = max(score, 0.3 + 1.5 * node_total[m])
            scored.append((score, a))
        scored.sort(key=lambda t: -t[0])
        return [a for _, a in scored[:k]]


class AlphaBetaPlayer(ValueFunctionPlayer):
    """Turn-level depth-2 expectimax with alpha pruning and a node budget.

    At its own (post-roll) decisions the player searches sequences of its own
    actions for the rest of the turn (``beam`` candidates per node ordered by
    a cheap heuristic, at most ``depth`` actions before END_TURN), and values
    each END_TURN leaf by the next opponent's reply: a chance node over the
    ``dice_outcomes`` most likely dice sums (7 with discards / robber, 6, 8,
    ... with their true probabilities; the remaining mass is a "null roll")
    followed by a MIN node over that opponent's most damaging replies (city,
    settlement, knight + robber, longest-road road).  Chance actions of its
    own (dev-card purchase, steals) are averaged exactly.  ``budget`` caps the
    number of game copies per decision: the search deepens iteratively
    (1, 2, ... ``depth`` own actions) and keeps the last iteration that
    completed within the budget, so every root candidate is compared at the
    same depth.  Other prompts (robber, initial placements, pre-roll knight,
    road building) use the 1-ply logic of :class:`ValueFunctionPlayer`.
    """

    def __init__(self, color: Color, budget: int = 2000, beam: int = 8, depth: int = 3,
                 dice_outcomes: int = 3, opponent_replies: int = 4,
                 weights: Optional[Dict[str, object]] = None, opening_book: bool = True,
                 is_bot: bool = True):
        super().__init__(color, weights=weights, opening_book=opening_book, is_bot=is_bot)
        self.budget = budget
        self.beam = beam
        self.depth = depth
        self.dice_outcomes = max(1, dice_outcomes)
        self.opponent_replies = opponent_replies
        self.depth_reached = 0  # depth of the last completed iteration
        self._max_depth = depth
        self._exhausted = False
        sums = DICE_SUMS_BY_PROBA[: self.dice_outcomes]
        self._dice = [(DICE_PROBAS[s], DICE_PAIR[s]) for s in sums]
        rest = 1.0 - sum(p for p, _ in self._dice)
        if rest > 1e-9:
            # the other sums: a "null roll" (sum 0 yields nothing) with the remaining mass
            self._dice.append((rest, NULL_DICE))

    def _decide_turn(self, game: Game, playable_actions: List[Action]) -> Action:
        state = game.state
        if state.current_prompt != ActionPrompt.PLAY_TURN or state.is_road_building:
            return self._greedy(game, playable_actions)
        return self._search_root(game, playable_actions)

    # -- search -----------------------------------------------------------
    def _search_root(self, game: Game, playable_actions: List[Action]) -> Action:
        cands = self._own_candidates(game, playable_actions)
        if len(cands) == 1:
            return cands[0]
        best = cands[0]
        self.depth_reached = 0
        for d in range(1, self.depth + 1):
            self._max_depth = d
            self._exhausted = False
            results = []
            alpha = float("-inf")
            for action in cands:
                v = self._own_action_value(game, action, 1, alpha)
                if self._exhausted:
                    break
                results.append((v, action))
                if v > alpha:
                    alpha = v
            if results and (not self._exhausted or d == 1):
                best = max(results, key=lambda t: t[0])[1]
                self.depth_reached = d
            if self._exhausted:
                break
        return best

    def _own_action_value(self, game: Game, action: Action, depth: int, alpha: float) -> float:
        if action.action_type == ActionType.END_TURN:
            return self._opponent_node(game, action, alpha)
        total = 0.0
        for p, a in action_outcomes(game, action):
            child = self._copy_exec(game, a)
            total += p * self._max_node(child, depth, alpha)
        return total

    def _max_node(self, game: Game, depth: int, alpha: float) -> float:
        state = game.state
        ps = state.player_state
        if ps[self._keys(state).vp] >= getattr(game, "vps_to_win", 10):
            return self._value(game)
        if self.nodes >= self.budget:
            self._exhausted = True
            return self._value(game)
        prompt = state.current_prompt
        acts = state.playable_actions
        if state.current_color() != self.color or prompt == ActionPrompt.DISCARD:
            return self._value(game)  # should not happen mid-turn
        end_turn = next((a for a in acts if a.action_type == ActionType.END_TURN), None)
        if depth >= self._max_depth and end_turn is not None:
            return self._opponent_node(game, end_turn, alpha)
        cands = self._own_candidates(game, acts)
        best = float("-inf")
        for action in cands:
            v = self._own_action_value(game, action, depth + 1, max(alpha, best))
            if v > best:
                best = v
        return best if best != float("-inf") else self._value(game)

    def _opponent_node(self, game: Game, end_turn: Action, alpha: float) -> float:
        """Value of ending the turn: chance over dice, MIN over the opponent's replies."""
        if self.nodes >= self.budget:
            self._exhausted = True
            return self._value(game)
        after = self._copy_exec(game, end_turn)
        opp = after.state.current_color()
        if opp == self.color:
            return self._value(after)
        post = []
        for p, dice in self._dice:
            c = self._copy_exec(after, Action(opp, ActionType.ROLL, dice))
            self._resolve_discards(c)
            post.append((p, c, self._value(c)))
        total = 0.0
        remaining = sum(p * v for p, _, v in post)  # opponent replies only lower our value
        for p, c, v_upper in post:
            remaining -= p * v_upper
            vmin = v_upper
            if self.nodes < self.budget:
                for reply in self._opponent_replies(c, opp):
                    if self.nodes >= self.budget:
                        self._exhausted = True
                        break
                    v = self._reply_value(c, reply)
                    if v < vmin:
                        vmin = v
                        if total + p * vmin + remaining <= alpha:
                            return total + p * vmin + remaining  # cannot beat alpha
            total += p * vmin
        return total

    def _resolve_discards(self, game: Game) -> None:
        state = game.state
        guard = 0
        while state.current_prompt == ActionPrompt.DISCARD and guard < 8:
            color = state.current_color()
            cards = plan_discard(game, color)
            game.execute(Action(color, ActionType.DISCARD, cards), validate_action=False)
            guard += 1

    def _reply_value(self, game: Game, reply) -> float:
        """Our value after an opponent reply (a single action or a short sequence)."""
        g = game
        for a in reply:
            g = self._copy_exec(g, a)
        return self._value(g)

    def _opponent_replies(self, game: Game, opp: Color) -> List[Tuple[Action, ...]]:
        state = game.state
        acts = state.playable_actions
        tables = get_map_tables(state.board.map)
        replies: List[Tuple[float, Tuple[Action, ...]]] = []
        if state.current_prompt == ActionPrompt.MOVE_ROBBER:
            for a in robber_candidates(game, opp, 2, target=self.color, playable_actions=acts):
                replies.append((3.0, (a,)))
            return [r for _, r in replies[: self.opponent_replies]]
        best_city = best_settle = None
        knight = None
        road_race = False
        ps = state.player_state
        ko = _KEYS[state.color_to_index[opp]]
        if not ps[ko.has_road] and ps[ko.road_len] >= 4 and ps[ko.road_len] + 1 > state.board.road_length:
            road_race = True
        best_road = None
        for a in acts:
            t = a.action_type
            if t == ActionType.BUILD_CITY:
                tot = tables.node_total[a.value]
                if best_city is None or tot > best_city[0]:
                    best_city = (tot, a)
            elif t == ActionType.BUILD_SETTLEMENT:
                tot = tables.node_total[a.value]
                if best_settle is None or tot > best_settle[0]:
                    best_settle = (tot, a)
            elif t == ActionType.PLAY_KNIGHT_CARD:
                knight = a
            elif t == ActionType.BUILD_ROAD and road_race and best_road is None:
                best_road = a
        if best_city is not None:
            replies.append((4.0 + best_city[0], (best_city[1],)))
        if best_settle is not None:
            replies.append((3.0 + best_settle[0], (best_settle[1],)))
        if knight is not None:
            after = self._copy_exec(game, knight)
            for a in robber_candidates(after, opp, 1, target=self.color):
                # steal outcome: take the victim's most common resource (deterministic)
                coord, victim, _ = a.value
                if victim is not None:
                    hand = _hand(after.state.player_state, _KEYS[after.state.color_to_index[victim]])
                    res = RESOURCES[max(range(5), key=lambda i: hand[i])]
                    a = Action(a.color, a.action_type, (coord, victim, res))
                replies.append((2.5, (knight, a)))
        if best_road is not None:
            replies.append((2.0, (best_road,)))
        replies.sort(key=lambda t: -t[0])
        return [r for _, r in replies[: self.opponent_replies]]

    # -- own move ordering / pruning (no game copies) -----------------------
    def _own_candidates(self, game: Game, playable_actions: Sequence[Action]) -> List[Action]:
        state = game.state
        prompt = state.current_prompt
        if prompt == ActionPrompt.MOVE_ROBBER:
            return robber_candidates(game, self.color, 2, playable_actions=playable_actions)
        if state.is_road_building:
            return self._road_candidates(game, playable_actions, 2) or list(playable_actions[:1])
        tables = get_map_tables(state.board.map)
        ps = state.player_state
        k = self._keys(state)
        hand = _hand(ps, k)
        by_type: Dict[ActionType, List[Action]] = {}
        for a in playable_actions:
            by_type.setdefault(a.action_type, []).append(a)
        node_total = tables.node_total
        out: List[Action] = []

        cities = sorted(by_type.get(ActionType.BUILD_CITY, ()), key=lambda a: -node_total[a.value])[:2]
        settlements = sorted(by_type.get(ActionType.BUILD_SETTLEMENT, ()), key=lambda a: -node_total[a.value])[:2]
        out.extend(cities)
        out.extend(settlements)
        has_build = bool(cities or settlements)

        has_settlement = bool(state.buildings_by_color[self.color][SETTLEMENT])
        now, one_road = _reachable_spots(state, self.color, tables)
        can_city = ps[k.city_av] > 0 and has_settlement
        can_settle = ps[k.settle_av] > 0 and bool(now)
        can_dev = bool(state.development_listdeck)

        def enabled_build(new_hand) -> int:
            """3 city, 2 settlement, 1 dev card, 0 nothing (only builds not affordable now)."""
            if can_city and _affordable(new_hand, CITY_COST) and not _affordable(hand, CITY_COST):
                return 3
            if can_settle and _affordable(new_hand, SETTLEMENT_COST) and not _affordable(hand, SETTLEMENT_COST):
                return 2
            if can_dev and _affordable(new_hand, DEV_COST) and not _affordable(hand, DEV_COST):
                return 1
            return 0

        if ActionType.BUY_DEVELOPMENT_CARD in by_type:
            out.append(by_type[ActionType.BUY_DEVELOPMENT_CARD][0])
        if ActionType.PLAY_KNIGHT_CARD in by_type:
            out.append(by_type[ActionType.PLAY_KNIGHT_CARD][0])

        yops = []
        for a in by_type.get(ActionType.PLAY_YEAR_OF_PLENTY, ()):
            new_hand = list(hand)
            for r in a.value:
                new_hand[RESOURCE_INDEX[r]] += 1
            yops.append((enabled_build(new_hand), a))
        if yops:
            yops.sort(key=lambda t: -t[0])
            if yops[0][0] > 0:
                out.extend(a for e, a in yops[:2] if e > 0)
            else:
                # no build completed: the pair that most advances the best build
                def gain(a):
                    new_hand = list(hand)
                    for r in a.value:
                        new_hand[RESOURCE_INDEX[r]] += 1
                    return max(sum(min(new_hand[i], c[i]) for i in range(5)) / sum(c)
                               for c in (CITY_COST, SETTLEMENT_COST, DEV_COST))
                out.append(max((a for _, a in yops), key=gain))

        monos = by_type.get(ActionType.PLAY_MONOPOLY, ())
        if monos:
            def stolen(a):
                ri = RESOURCE_INDEX[a.value]
                return sum(_hand(ps, _KEYS[i])[ri] for i in range(len(state.colors))
                           if state.colors[i] != self.color)
            best_mono = max(monos, key=stolen)
            if stolen(best_mono) >= 2:
                out.append(best_mono)

        if ActionType.PLAY_ROAD_BUILDING in by_type:
            out.append(by_type[ActionType.PLAY_ROAD_BUILDING][0])

        roads = by_type.get(ActionType.BUILD_ROAD, ())
        if roads:
            out.extend(self._road_candidates(game, roads, 2))

        trades = []
        fallback = None
        for a in by_type.get(ActionType.MARITIME_TRADE, ()):
            offer = a.value
            new_hand = list(hand)
            given = [r for r in offer[:4] if r is not None]
            for r in given:
                new_hand[RESOURCE_INDEX[r]] -= 1
            new_hand[RESOURCE_INDEX[offer[4]]] += 1
            e = enabled_build(new_hand)
            if e > 0:
                trades.append((e, -len(given), new_hand[RESOURCE_INDEX[given[0]]], a))
            elif sum(hand) > state.discard_limit:
                gain = max(sum(min(new_hand[i], c[i]) for i in range(5)) / sum(c)
                           for c in (CITY_COST, SETTLEMENT_COST, DEV_COST))
                if fallback is None or gain > fallback[0]:
                    fallback = (gain, a)
        if trades:
            trades.sort(key=lambda t: (-t[0], -t[1], -t[2]))
            out.extend(a for _, _, _, a in trades[:2])
        elif fallback is not None:
            out.append(fallback[1])

        out = out[: self.beam]
        if not has_build and ActionType.END_TURN in by_type:
            out.append(by_type[ActionType.END_TURN][0])
        if not out:
            out = list(playable_actions[:1])
        return out
