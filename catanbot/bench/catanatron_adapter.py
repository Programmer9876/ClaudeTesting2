"""Bridge between the ``catanatron`` engine (PyPI ``catanatron``) and catanbot.

Three things live here:

1. **Board mapping** (:func:`derive_mapping`): catanatron addresses the 19
   land tiles by cube coordinates, the 54 vertices by node ids and the 72
   edges by node pairs.  catanbot uses hex indices 0..18 (rows 3-4-5-4-3,
   :data:`catanbot.board.HEX_COORDS`), vertex ids 0..53 and edge ids 0..71.
   The bijections are derived *structurally*: the 12 symmetries of the
   hexagonal board (6 rotations x mirror) are tried on the tile coordinates,
   node ids are matched to vertices through the tile incidence structure (a
   node shared by tiles A, B, C must map to the vertex shared by hexes f(A),
   f(B), f(C)) and the remaining coastal ambiguities are resolved through edge
   adjacency.  The result is verified to be a bijection on tiles, nodes and
   edges before use.  The identity symmetry is tried first so the natural
   orientation (catanatron ``NORTH`` = catanbot corner 0) wins when it fits.

2. **State / action conversion** (:func:`to_catanbot_state`,
   :func:`catanbot_action_to_key`, :func:`catanatron_action_to_catanbot`):
   a catanatron ``State`` becomes a fully observable :class:`GameState`
   (catanatron exposes every hand, so *all* players get ``hand_known=True``)
   and catanbot actions are matched against catanatron's ``playable_actions``.

3. :class:`CatanbotPlayer`: a catanatron ``Player`` whose ``decide`` runs a
   catanbot bot (default: the depth-1 heuristic :class:`SearchBot`) on the
   converted state, restricted to the legal actions that have a catanatron
   equivalent.  Actions logged by catanatron since our previous decision are
   replayed through ``Bot.observe`` so the opponent model / political
   tracking work exactly as in self-play.

Known semantic differences (see ``docs/BENCHMARKS.md``): player-to-player
trading is suppressed (catanatron 3.2.1 has none; 3.3's domestic-trade
prompts, which no stock player ever opens, are answered with ``REJECT_TRADE``
/ ``CANCEL_TRADE``), discards are chosen randomly by the catanatron 3.2.1
engine (its only ``DISCARD`` action has value ``None``) while on 3.3 the bot
picks its discard as one catanbot ``DISCARD`` and hands it over one
``DISCARD_RESOURCE`` prompt at a time, dev cards bought this turn are
playable immediately in catanatron (so they are placed in
``Player.dev_cards`` rather than ``dev_cards_new``), ``PLAY_ROAD_BUILDING``
is only offered by catanatron while the player also holds wood + brick, and
``PLAY_KNIGHT`` is two catanatron decisions (``PLAY_KNIGHT_CARD`` then
``MOVE_ROBBER``) - the robber target chosen by the search is remembered and
executed at the second prompt.  Two engine rules differ without any adapter
involvement: catanatron never counts a road that ends at an opponent's
building towards Longest Road (catanbot and the official rules do; the
awarded owner / length are copied from catanatron, so victory points always
agree with catanatron), and catanatron 3.2.1 prompts the *later* discarders
of a 7 with a hard-coded ``> 7`` regardless of its ``discard_limit``
(mirrored by :func:`state_to_catanbot`; 3.3 applies the limit to everyone).

Both catanatron generations are supported by feature detection (:data:`API_33`):
the PyPI 3.2.1 wheel (``State.actions``, ``State.playable_actions``,
``catanatron.state.apply_action``, one random ``DISCARD``, 3-tuple robber
values) and the 3.3 engine of the GitHub checkout (``State.action_records``
of ``ActionRecord(action, result)``, ``Game.playable_actions``,
``catanatron.apply_action.apply_action(state, action, record)``, per-card
``DISCARD_RESOURCE`` with ``State.discard_counts``, 2-tuple robber values,
``DECIDE_TRADE`` / ``DECIDE_ACCEPTEES`` prompts).  The helpers
:func:`action_log`, :func:`log_action`, :func:`playable_actions_of`,
:func:`apply_action` and :func:`replay_entry` hide the differences.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import (
    CITY,
    ROAD,
    SETTLEMENT,
    Action as CAction,
    ActionPrompt,
    ActionType,
)
from catanatron.models.map import CatanMap, NodeRef
from catanatron.models.player import Color, Player
from catanatron.state import State
try:  # catanatron <= 3.2 (PyPI wheel)
    from catanatron.state import apply_action as _apply_action
except ImportError:  # catanatron >= 3.3 (GitHub checkout)
    from catanatron.apply_action import apply_action as _apply_action
try:  # catanatron >= 3.3: the log holds ActionRecord(action, result) pairs
    from catanatron.models.enums import ActionRecord
except ImportError:  # catanatron 3.2.1: the log holds fully specified Actions
    ActionRecord = None

from .. import actions as A
from .. import board as B
from .. import engine as E
from ..agents.base import Bot
from ..agents.search_bot import SearchBot
from ..heuristic import action_priors
from ..selfplay import make_bot
from ..state import (
    PHASE_DISCARD,
    PHASE_GAME_OVER,
    PHASE_MAIN,
    PHASE_ROBBER,
    PHASE_ROLL,
    PHASE_SETUP_ROAD,
    PHASE_SETUP_SETTLEMENT,
    GameState,
    Player as CBPlayer,
)

__all__ = [
    "DEFAULT_SPEC",
    "COLORS",
    "COLOR_NAMES",
    "API_33",
    "CATANATRON_VERSION",
    "DISCARD_LEGACY",
    "DISCARD_RESOURCE",
    "DISCARD_TYPES",
    "TRADE_PROMPTS",
    "action_log",
    "log_action",
    "log_result",
    "playable_actions_of",
    "apply_action",
    "replay_entry",
    "BoardMapping",
    "board_symmetries",
    "derive_mapping",
    "mapping_for",
    "verify_mapping",
    "to_catanbot_state",
    "state_to_catanbot",
    "catanbot_action_to_key",
    "playable_key",
    "index_playable",
    "catanatron_action_to_catanbot",
    "fallback_action",
    "CatanbotPlayer",
    "make_game",
    "play_game",
]

DEFAULT_SPEC = "search:depth=1,evaluator=heuristic"

#: catanatron seat colours in the order the benchmark assigns them.
COLORS: Tuple[Color, ...] = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
COLOR_NAMES: Dict[Color, str] = {Color.RED: "red", Color.BLUE: "blue", Color.ORANGE: "orange", Color.WHITE: "white"}

# catanatron resource / dev-card strings <-> catanbot indices
RESOURCE_TO_CB: Dict[str, int] = {"WOOD": B.WOOD, "BRICK": B.BRICK, "SHEEP": B.SHEEP, "WHEAT": B.WHEAT, "ORE": B.ORE}
CB_TO_RESOURCE: List[str] = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEV_TO_CB: Dict[str, int] = {
    "KNIGHT": B.DEV_KNIGHT,
    "VICTORY_POINT": B.DEV_VP,
    "ROAD_BUILDING": B.DEV_ROAD_BUILDING,
    "YEAR_OF_PLENTY": B.DEV_YEAR_OF_PLENTY,
    "MONOPOLY": B.DEV_MONOPOLY,
}
CB_TO_DEV: List[str] = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]

Coordinate = Tuple[int, int, int]
NODE_REFS: List[NodeRef] = [NodeRef.NORTH, NodeRef.NORTHEAST, NodeRef.SOUTHEAST, NodeRef.SOUTH,
                            NodeRef.SOUTHWEST, NodeRef.NORTHWEST]


# ---------------------------------------------------------------------------
# Engine version differences (3.2.1 wheel vs 3.3 checkout), detected by feature
# ---------------------------------------------------------------------------
def _installed_version() -> str:
    try:
        from importlib.metadata import version
        return version("catanatron")
    except Exception:  # pragma: no cover - not installed as a distribution
        return "?"


#: The installed ``catanatron`` distribution version (informational; the API is detected by feature).
CATANATRON_VERSION: str = _installed_version()
#: 3.3 discards one ``DISCARD_RESOURCE`` (value: the resource) per prompt ...
DISCARD_RESOURCE = getattr(ActionType, "DISCARD_RESOURCE", None)
#: ... 3.2.1 has a single ``DISCARD`` whose playable value is ``None`` (random) and whose log value is the card list.
DISCARD_LEGACY = getattr(ActionType, "DISCARD", None)
DISCARD_TYPES: Tuple[ActionType, ...] = tuple(t for t in (DISCARD_LEGACY, DISCARD_RESOURCE) if t is not None)
#: True on the 3.3 API (``Game.playable_actions``, ``State.action_records``, per-card discards, trade prompts).
API_33: bool = DISCARD_RESOURCE is not None
#: 3.3 domestic-trade prompts (answered with ``REJECT_TRADE`` / ``CANCEL_TRADE`` by :class:`CatanbotPlayer`).
TRADE_PROMPTS: Tuple[ActionPrompt, ...] = tuple(
    p for p in (getattr(ActionPrompt, "DECIDE_TRADE", None), getattr(ActionPrompt, "DECIDE_ACCEPTEES", None))
    if p is not None)
_TRADE_ANSWERS: Tuple[ActionType, ...] = tuple(
    t for t in (getattr(ActionType, "REJECT_TRADE", None), getattr(ActionType, "CANCEL_TRADE", None))
    if t is not None)


def action_log(st: State) -> Sequence:
    """The engine's action log: ``State.actions`` (3.2.1) or ``State.action_records`` (3.3).

    Entries are fully specified either way (dice, drawn card, stolen card):
    use :func:`log_action` for the ``Action`` and :func:`replay_entry` to re-apply one.
    """
    recs = getattr(st, "action_records", None)
    return st.actions if recs is None else recs


def log_action(entry) -> CAction:
    """The ``Action`` of a log entry (an ``ActionRecord`` on 3.3, the action itself on 3.2.1)."""
    return getattr(entry, "action", entry)


def log_result(entry):
    """The chance result recorded with a log entry (3.3 ``ActionRecord.result``; ``None`` on
    3.2.1, whose logged actions carry the outcome in their value instead)."""
    return getattr(entry, "result", None)


def playable_actions_of(game: Game) -> List[CAction]:
    """The current playable actions (``Game.playable_actions`` on 3.3, ``State.playable_actions`` on 3.2.1)."""
    acts = getattr(game, "playable_actions", None)
    if acts is None:
        acts = game.state.playable_actions
    return acts


def apply_action(state: State, action: CAction, record=None):
    """Version-tolerant ``apply_action``; ``record`` (a 3.3 ``ActionRecord``) fixes the chance outcome.

    On 3.2.1 the outcome travels in the action value (dice, card, stolen
    resource, discarded cards) and ``record`` is ignored.
    """
    if ActionRecord is not None:
        return _apply_action(state, action, record)
    return _apply_action(state, action)


def replay_entry(state: State, entry) -> None:
    """Re-apply a log entry to ``state`` exactly as logged.

    On 3.3 the record must be passed along: ``apply_action`` ignores the value of a
    logged ``ROLL`` / ``MOVE_ROBBER`` and would draw fresh randomness (from the
    ``random.Random`` the state *shares* with its copies) without it.
    """
    if ActionRecord is not None and isinstance(entry, ActionRecord):
        _apply_action(state, entry.action, entry)
    else:
        _apply_action(state, log_action(entry))


# ---------------------------------------------------------------------------
# Board mapping
# ---------------------------------------------------------------------------
class MappingError(ValueError):
    """Raised when catanatron's board cannot be matched onto catanbot's topology."""


def _rot60(c: Coordinate) -> Coordinate:
    x, y, z = c
    return (-z, -x, -y)


def _mirror(c: Coordinate) -> Coordinate:
    x, y, z = c
    return (x, z, y)


def board_symmetries() -> List[Tuple[Tuple[int, bool], "callable"]]:
    """The 12 symmetries of the hexagonal board on cube coordinates, identity first.

    Returns ``[((rotation_steps, mirrored), f), ...]`` with ``f(cube) -> cube``.
    """
    syms = []
    for mirrored in (False, True):
        for k in range(6):
            def f(c: Coordinate, k: int = k, mirrored: bool = mirrored) -> Coordinate:
                if mirrored:
                    c = _mirror(c)
                for _ in range(k):
                    c = _rot60(c)
                return c
            syms.append(((k, mirrored), f))
    return syms


def _cube_to_hex(c: Coordinate) -> Optional[int]:
    """catanatron cube coordinate -> catanbot hex index (axial q = x, r = z)."""
    return B.HEX_INDEX.get((c[0], c[2]))


@dataclass
class BoardMapping:
    """Bijections between catanatron's and catanbot's board ids."""

    symmetry: Tuple[int, bool]                 # (rotation steps, mirrored) applied to cube coordinates
    coord_to_hex: Dict[Coordinate, int]        # catanatron cube coordinate -> hex index
    hex_to_coord: List[Coordinate]             # hex index -> cube coordinate
    tile_to_hex: Dict[int, int]                # catanatron tile id -> hex index
    hex_to_tile: List[int]                     # hex index -> tile id
    node_to_vertex: Dict[int, int]             # node id -> vertex id
    vertex_to_node: List[int]                  # vertex id -> node id
    edge_to_id: Dict[Tuple[int, int], int]     # (node, node) in either order -> edge id
    id_to_edge: List[Tuple[int, int]]          # edge id -> (node_a, node_b) with node_a < node_b

    def hex_of_coord(self, coord) -> int:
        return self.coord_to_hex[tuple(coord)]

    def edge_id(self, edge) -> int:
        return self.edge_to_id[(edge[0], edge[1])]


def _match_nodes(catan_map: CatanMap, tile_to_hex: Dict[int, int]) -> Dict[int, int]:
    """Node id -> vertex id from the incidence structure (raises :class:`MappingError`)."""
    node_hexes: Dict[int, set] = {}
    adj: Dict[int, set] = {}
    for tile in catan_map.land_tiles.values():
        h = tile_to_hex[tile.id]
        for n in tile.nodes.values():
            node_hexes.setdefault(n, set()).add(h)
        for a, b in tile.edges.values():
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    if len(node_hexes) != B.NUM_VERTICES:
        raise MappingError(f"{len(node_hexes)} land nodes, expected {B.NUM_VERTICES}")
    by_hexset: Dict[FrozenSet[int], List[int]] = {}
    for v in range(B.NUM_VERTICES):
        by_hexset.setdefault(frozenset(B.VERTEX_HEXES[v]), []).append(v)
    candidates: Dict[int, List[int]] = {}
    for n, hs in node_hexes.items():
        cands = by_hexset.get(frozenset(hs))
        if not cands:
            raise MappingError(f"node {n} touches hexes {sorted(hs)} which share no catanbot vertex")
        candidates[n] = list(cands)
    mapping: Dict[int, int] = {}
    used: set = set()
    for n, cands in candidates.items():
        if len(cands) == 1:
            v = cands[0]
            if v in used:
                raise MappingError(f"vertex {v} claimed by two nodes")
            mapping[n] = v
            used.add(v)
    # Coastal corners touching a single tile come in pairs that share the same
    # incidence set; edge adjacency to an already mapped neighbour tells them apart.
    pending = [n for n in candidates if n not in mapping]
    progress = True
    while pending and progress:
        progress = False
        rest = []
        for n in pending:
            opts = [v for v in candidates[n] if v not in used]
            for m in adj.get(n, ()):
                if m in mapping:
                    nb = B.VERTEX_NEIGHBORS[mapping[m]]
                    opts = [v for v in opts if v in nb]
            if len(opts) == 1:
                mapping[n] = opts[0]
                used.add(opts[0])
                progress = True
            elif not opts:
                raise MappingError(f"node {n} has no consistent vertex")
            else:
                rest.append(n)
        pending = rest
    if pending:
        raise MappingError(f"nodes {pending} remain ambiguous")
    if len(used) != B.NUM_VERTICES:
        raise MappingError("node -> vertex map is not a bijection")
    return mapping


def verify_mapping(catan_map: CatanMap, m: BoardMapping) -> List[str]:
    """Every incidence that must hold for ``m`` to be a board isomorphism; ``[]`` when OK."""
    problems: List[str] = []
    if sorted(m.tile_to_hex.values()) != list(range(B.NUM_HEXES)):
        problems.append("tile -> hex is not a bijection")
    if sorted(m.node_to_vertex.values()) != list(range(B.NUM_VERTICES)):
        problems.append("node -> vertex is not a bijection")
    if sorted(set(m.edge_to_id.values())) != list(range(B.NUM_EDGES)):
        problems.append("edge -> id does not cover every edge")
    for coord, tile in catan_map.land_tiles.items():
        h = m.coord_to_hex.get(tuple(coord))
        if h is None or m.tile_to_hex.get(tile.id) != h:
            problems.append(f"tile {tile.id} at {coord} has no consistent hex")
            continue
        want = set(B.HEX_VERTICES[h])
        got = {m.node_to_vertex.get(n, -1) for n in tile.nodes.values()}
        if want != got:
            problems.append(f"tile {tile.id}: nodes map to {sorted(got)} but hex {h} has {sorted(want)}")
        want_e = set(B.HEX_EDGES[h])
        got_e = set()
        for a, b in tile.edges.values():
            va, vb = m.node_to_vertex.get(a), m.node_to_vertex.get(b)
            try:
                eid = B.edge_between(va, vb)
            except (KeyError, TypeError):
                problems.append(f"tile {tile.id}: edge {(a, b)} maps to non-adjacent vertices {(va, vb)}")
                continue
            if m.edge_to_id.get((a, b)) != eid:
                problems.append(f"edge {(a, b)}: id {m.edge_to_id.get((a, b))} != {eid}")
            got_e.add(eid)
        if want_e != got_e:
            problems.append(f"tile {tile.id}: edges map to {sorted(got_e)} but hex {h} has {sorted(want_e)}")
    return problems


def derive_mapping(catan_map: CatanMap) -> BoardMapping:
    """Derive the tile / node / edge bijections for ``catan_map`` (standard 19-tile board)."""
    land = catan_map.land_tiles
    if len(land) != B.NUM_HEXES:
        raise MappingError(f"expected {B.NUM_HEXES} land tiles, got {len(land)}")
    failures: List[str] = []
    for sym, f in board_symmetries():
        coord_to_hex: Dict[Coordinate, int] = {}
        for coord in land:
            h = _cube_to_hex(f(tuple(coord)))
            if h is None:
                break
            coord_to_hex[tuple(coord)] = h
        if len(coord_to_hex) != B.NUM_HEXES or len(set(coord_to_hex.values())) != B.NUM_HEXES:
            failures.append(f"{sym}: tiles do not land on the 19 hexes")
            continue
        tile_to_hex = {tile.id: coord_to_hex[tuple(coord)] for coord, tile in land.items()}
        try:
            node_to_vertex = _match_nodes(catan_map, tile_to_hex)
        except MappingError as exc:
            failures.append(f"{sym}: {exc}")
            continue
        hex_to_coord: List[Coordinate] = [(0, 0, 0)] * B.NUM_HEXES
        hex_to_tile: List[int] = [-1] * B.NUM_HEXES
        for coord, tile in land.items():
            hex_to_coord[coord_to_hex[tuple(coord)]] = tuple(coord)
            hex_to_tile[coord_to_hex[tuple(coord)]] = tile.id
        vertex_to_node: List[int] = [-1] * B.NUM_VERTICES
        for n, v in node_to_vertex.items():
            vertex_to_node[v] = n
        edge_to_id: Dict[Tuple[int, int], int] = {}
        id_to_edge: List[Tuple[int, int]] = [(-1, -1)] * B.NUM_EDGES
        try:
            for tile in land.values():
                for a, b in tile.edges.values():
                    eid = B.edge_between(node_to_vertex[a], node_to_vertex[b])
                    edge_to_id[(a, b)] = eid
                    edge_to_id[(b, a)] = eid
                    id_to_edge[eid] = (min(a, b), max(a, b))
        except KeyError as exc:
            failures.append(f"{sym}: edge maps to non-adjacent vertices ({exc})")
            continue
        m = BoardMapping(sym, coord_to_hex, hex_to_coord, tile_to_hex, hex_to_tile, node_to_vertex,
                         vertex_to_node, edge_to_id, id_to_edge)
        problems = verify_mapping(catan_map, m)
        if problems:
            failures.append(f"{sym}: {problems[0]}")
            continue
        return m
    raise MappingError("no board symmetry maps catanatron's tiles onto catanbot's hexes: " + "; ".join(failures))


_MAPPING_CACHE: Dict[tuple, BoardMapping] = {}


def _map_key(catan_map: CatanMap) -> tuple:
    return tuple(sorted((tuple(coord), tile.id, tuple(tile.nodes[r] for r in NODE_REFS))
                        for coord, tile in catan_map.land_tiles.items()))


def mapping_for(catan_map: CatanMap) -> BoardMapping:
    """Cached :func:`derive_mapping` (node / tile ids are identical for every BASE map)."""
    key = _map_key(catan_map)
    m = _MAPPING_CACHE.get(key)
    if m is None:
        m = derive_mapping(catan_map)
        if len(_MAPPING_CACHE) > 8:
            _MAPPING_CACHE.clear()
        _MAPPING_CACHE[key] = m
    return m


# ---------------------------------------------------------------------------
# State conversion
# ---------------------------------------------------------------------------
def _last_roll_this_turn(log: Sequence) -> int:
    """Dice total of the current turn's ROLL (0 if the turn has not rolled yet); ``log`` as :func:`action_log`."""
    for entry in reversed(log):
        a = log_action(entry)
        t = a.action_type
        if t == ActionType.END_TURN:
            return 0
        if t == ActionType.ROLL:
            v = a.value
            return int(v[0]) + int(v[1]) if v else 0
    return 0


def state_to_catanbot(st: State, vps_to_win: int = 10, mapping: Optional[BoardMapping] = None,
                      suppress_trades: bool = True) -> GameState:
    """Convert a catanatron ``State`` into a fully observable catanbot :class:`GameState`.

    Seats follow ``st.colors``.  Every player is ``hand_known`` / ``dev_known``
    because catanatron exposes all hands.  With ``suppress_trades`` the state
    reports the maximum number of proposals already made this turn, so
    :func:`catanbot.engine.legal_actions` emits no ``PROPOSE_TRADE`` (there is
    no player trading in catanatron 3.2.1, and the benchmark never offers one
    on 3.3).  The 3.3 domestic-trade prompts (``DECIDE_TRADE`` for a
    responder, ``DECIDE_ACCEPTEES`` for the offerer) have no catanbot phase:
    they convert to the turn player's ``PHASE_MAIN`` state.  On 3.3 a
    ``DISCARD`` prompt reached *mid-way* through a player's per-card discards
    converts with the already reduced hand (the engine's remaining
    ``discard_counts`` are then what :class:`CatanbotPlayer` follows).
    """
    m = mapping or mapping_for(st.board.map)
    s = GameState()
    hexes: List[Tuple[int, int]] = [(B.DESERT, 0)] * B.NUM_HEXES
    for coord, tile in st.board.map.land_tiles.items():
        h = m.coord_to_hex[tuple(coord)]
        if tile.resource is None:
            hexes[h] = (B.DESERT, 0)
        else:
            hexes[h] = (RESOURCE_TO_CB[tile.resource], int(tile.number or 0))
    s.hexes = hexes
    s.robber = m.coord_to_hex[tuple(st.board.robber_coordinate)]
    ports: Dict[int, int] = {}
    for res, nodes in st.board.map.port_nodes.items():
        t = B.PORT_GENERIC if res is None else RESOURCE_TO_CB[res]
        for n in nodes:
            ports[m.node_to_vertex[n]] = t
    s.ports = ports

    ps = st.player_state
    n2v = m.node_to_vertex
    e2i = m.edge_to_id
    colors = st.colors
    n = len(colors)
    players: List[CBPlayer] = []
    for i, color in enumerate(colors):
        key = f"P{i}"
        p = CBPlayer(color=COLOR_NAMES.get(color, str(color.value).lower()),
                     name=COLOR_NAMES.get(color, str(color.value).lower()))
        p.resources = [int(ps[f"{key}_{r}_IN_HAND"]) for r in CB_TO_RESOURCE]
        held = [int(ps[f"{key}_{d}_IN_HAND"]) for d in CB_TO_DEV]
        if API_33:
            # 3.3 follows the official rule through ``{DEV}_OWNED_AT_START`` (set at the player's
            # END_TURN): a type bought this turn is not playable until the next turn, so it goes
            # to ``dev_cards_new`` (VP cards count either way).  A type owned at the start of the
            # turn is playable even if more copies were bought since (one play per turn anyway).
            p.dev_cards = [0] * 5
            p.dev_cards_new = [0] * 5
            for d, name in enumerate(CB_TO_DEV):
                if name == "VICTORY_POINT" or ps.get(f"{key}_{name}_OWNED_AT_START", True):
                    p.dev_cards[d] = held[d]
                else:
                    p.dev_cards_new[d] = held[d]
        else:
            # 3.2.1 lets a card bought this turn be played immediately -> all playable.
            p.dev_cards = held
            p.dev_cards_new = [0] * 5
        p.played_knights = int(ps[f"{key}_PLAYED_KNIGHT"])
        bb = st.buildings_by_color.get(color, {})
        p.settlements = [n2v[v] for v in bb.get(SETTLEMENT, [])]
        p.cities = [n2v[v] for v in bb.get(CITY, [])]
        p.roads = [e2i[(e[0], e[1])] for e in bb.get(ROAD, [])]
        p.hand_known = True
        p.hand_size = sum(p.resources)
        p.dev_known = True
        p.dev_count = sum(p.dev_cards)
        players.append(p)
        if ps[f"{key}_HAS_ROAD"]:
            s.longest_road_owner = i
            s.longest_road_len = int(ps[f"{key}_LONGEST_ROAD_LENGTH"])
        if ps[f"{key}_HAS_ARMY"]:
            s.largest_army_owner = i
    s.players = players
    s.bank = [int(x) for x in st.resource_freqdeck]
    deck = [0] * 5
    for card in st.development_listdeck:
        deck[DEV_TO_CB[card]] += 1
    s.dev_deck = deck

    cur = int(st.current_turn_index)
    s.current = cur
    s.turn = int(st.num_turns)
    s.max_turns = max(TURNS_LIMIT, s.turn + 2)
    cur_key = f"P{cur}"
    s.dev_played_this_turn = bool(ps[f"{cur_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])
    s.trades_this_turn = E.MAX_TRADE_PROPOSALS_PER_TURN if suppress_trades else 0
    s.free_roads = int(st.free_roads_available) if st.is_road_building else 0

    prompt = st.current_prompt
    cur_color = colors[cur]
    placed = len(st.buildings_by_color.get(cur_color, {}).get(SETTLEMENT, []))
    if prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        s.phase = PHASE_SETUP_SETTLEMENT
        s.setup_round = 1 if placed >= 1 else 0
        s.current = int(st.current_player_index)
    elif prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        s.phase = PHASE_SETUP_ROAD
        s.setup_round = 1 if placed >= 2 else 0
        s.current = int(st.current_player_index)
        last = st.buildings_by_color[cur_color][SETTLEMENT][-1]
        s.setup_last_settlement = n2v[last]
    elif prompt == ActionPrompt.DISCARD:
        s.phase = PHASE_DISCARD
        s.dice = 7
        cpi = int(st.current_player_index)
        queue = [cpi]
        discard_counts = getattr(st, "discard_counts", None)
        if discard_counts is not None:
            # 3.3 fixed every discarder (``> discard_limit``) and their counts at the ROLL.
            queue += [j for j in range(cpi + 1, n) if discard_counts[j] > 0]
        else:
            # 3.2.1 picks the *first* discarder with ``state.discard_limit`` (at the
            # ROLL) but advances to the later ones with a hard-coded ``> 7``
            # (``apply_action``'s DISCARD branch), whatever the configured limit.  The
            # queue mirrors the prompts catanatron will actually issue, so it is only
            # in the default ``discard_limit=7`` that both rules coincide.
            for j in range(cpi + 1, n):
                if players[j].total_resources > 7:
                    queue.append(j)
        s.discard_queue = queue
    elif prompt == ActionPrompt.MOVE_ROBBER:
        s.phase = PHASE_ROBBER
        s.dice = _last_roll_this_turn(action_log(st))
    else:  # PLAY_TURN (and the 3.3 trade prompts, which happen after the roll)
        if ps[f"{cur_key}_HAS_ROLLED"]:
            s.phase = PHASE_MAIN
            s.dice = _last_roll_this_turn(action_log(st))
        elif s.free_roads > 0:
            # 3.3 lets Road Building be played before the roll and then prompts the free
            # roads first: place them in a PHASE_MAIN state (dice 0); the ROLL prompt follows.
            s.phase = PHASE_MAIN
            s.dice = 0
        else:
            s.phase = PHASE_ROLL
            s.dice = 0
    for i in range(n):
        if ps[f"P{i}_ACTUAL_VICTORY_POINTS"] >= vps_to_win:
            s.winner = i
            s.phase = PHASE_GAME_OVER
    return s


def to_catanbot_state(game: Game, me_color: Optional[Color] = None, mapping: Optional[BoardMapping] = None,
                      suppress_trades: bool = True) -> GameState:
    """Convert a catanatron ``Game`` to a catanbot :class:`GameState`.

    ``me_color`` only validates that the colour is seated; catanatron's state is
    fully observable so every seat (ours and the opponents') is converted with
    exact hands (``hand_known=True``).  Seat ``i`` is ``game.state.colors[i]``.
    """
    if me_color is not None and me_color not in game.state.color_to_index:
        raise ValueError(f"{me_color} is not seated in this game")
    return state_to_catanbot(game.state, game.vps_to_win, mapping, suppress_trades)


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------
def playable_key(action: CAction) -> tuple:
    """Hashable identity of a catanatron action (road orientation / robber card normalised)."""
    t = action.action_type
    v = action.value
    if t == ActionType.BUILD_ROAD:
        return (t, (min(v), max(v)))
    if t == ActionType.MOVE_ROBBER:
        return (t, (tuple(v[0]), v[1]))
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        return (t, tuple(sorted(v)))
    if t == ActionType.MARITIME_TRADE:
        return (t, tuple(v))
    if t == DISCARD_LEGACY or t in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD):
        return (t, None)
    return (t, v)   # includes the 3.3 DISCARD_RESOURCE, whose value (the card) matters


def index_playable(playable: Sequence[CAction]) -> Dict[tuple, CAction]:
    """``{playable_key(a): a}`` for a catanatron ``playable_actions`` list."""
    return {playable_key(a): a for a in playable}


def catanbot_action_to_key(action: A.Action, state: GameState, mapping: BoardMapping,
                           colors: Sequence[Color]) -> Optional[tuple]:
    """The :func:`playable_key` a catanbot action corresponds to, or ``None`` if it has no
    catanatron equivalent (``PROPOSE_TRADE`` & friends, forced rolls).

    A catanbot ``DISCARD`` (all cards at once) is catanatron 3.2.1's single
    ``DISCARD None``; on 3.3, which discards one card per prompt, it is the
    ``DISCARD_RESOURCE`` of the plan's *first* card (:class:`CatanbotPlayer`
    queues the remaining cards for the following prompts).
    """
    kind = action[0]
    if kind in (A.SETUP_SETTLEMENT, A.BUILD_SETTLEMENT):
        return (ActionType.BUILD_SETTLEMENT, mapping.vertex_to_node[action[1]])
    if kind in (A.SETUP_ROAD, A.BUILD_ROAD):
        return (ActionType.BUILD_ROAD, mapping.id_to_edge[action[1]])
    if kind == A.BUILD_CITY:
        return (ActionType.BUILD_CITY, mapping.vertex_to_node[action[1]])
    if kind == A.ROLL:
        return (ActionType.ROLL, None) if len(action) == 1 else None
    if kind == A.END_TURN:
        return (ActionType.END_TURN, None)
    if kind == A.BUY_DEV:
        return (ActionType.BUY_DEVELOPMENT_CARD, None)
    if kind == A.PLAY_KNIGHT:
        return (ActionType.PLAY_KNIGHT_CARD, None)
    if kind == A.MOVE_ROBBER:
        victim = action[2]
        who = colors[victim] if victim is not None and victim >= 0 else None
        return (ActionType.MOVE_ROBBER, (mapping.hex_to_coord[action[1]], who))
    if kind == A.PLAY_ROAD_BUILDING:
        return (ActionType.PLAY_ROAD_BUILDING, None)
    if kind == A.PLAY_YEAR_OF_PLENTY:
        return (ActionType.PLAY_YEAR_OF_PLENTY, tuple(sorted((CB_TO_RESOURCE[action[1]], CB_TO_RESOURCE[action[2]]))))
    if kind == A.PLAY_MONOPOLY:
        return (ActionType.PLAY_MONOPOLY, CB_TO_RESOURCE[action[1]])
    if kind == A.BANK_TRADE:
        give, get = action[1], action[2]
        ratio = state.port_ratio(E.acting_player(state), give)
        offer = [CB_TO_RESOURCE[give]] * ratio + [None] * (4 - ratio) + [CB_TO_RESOURCE[get]]
        return (ActionType.MARITIME_TRADE, tuple(offer))
    if kind == A.DISCARD:
        if DISCARD_RESOURCE is None:
            return (DISCARD_LEGACY, None)
        counts = action[1] if len(action) > 1 else ()
        for r in range(min(5, len(counts))):
            if counts[r] > 0:
                return (DISCARD_RESOURCE, CB_TO_RESOURCE[r])
        return None
    return None


def catanatron_action_to_catanbot(action: CAction, state: GameState, mapping: BoardMapping,
                                  previous: Optional[CAction] = None) -> Optional[A.Action]:
    """Catanbot action for a catanatron action (logged or playable) taken in ``state``.

    ``previous`` is the action logged right before it: a ``MOVE_ROBBER`` that
    follows the same colour's ``PLAY_KNIGHT_CARD`` becomes ``PLAY_KNIGHT`` (the
    catanatron ``PLAY_KNIGHT_CARD`` itself maps to ``None``, as do single-card
    Year of Plenty picks which catanbot cannot express).  A discard becomes
    ``(DISCARD, counts)`` of the cards it names: all of them for a logged
    3.2.1 ``DISCARD``, one card for a 3.3 ``DISCARD_RESOURCE`` (the player
    merges a run of those into one observation), none for 3.2.1's playable
    ``DISCARD None``.  The 3.3 domestic-trade actions map to ``None``.
    """
    t = action.action_type
    v = action.value
    setup = state.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)
    if t == ActionType.BUILD_SETTLEMENT:
        return (A.SETUP_SETTLEMENT if setup else A.BUILD_SETTLEMENT, mapping.node_to_vertex[v])
    if t == ActionType.BUILD_ROAD:
        return (A.SETUP_ROAD if setup else A.BUILD_ROAD, mapping.edge_to_id[(v[0], v[1])])
    if t == ActionType.BUILD_CITY:
        return (A.BUILD_CITY, mapping.node_to_vertex[v])
    if t == ActionType.ROLL:
        return (A.ROLL,) if not v else (A.ROLL, int(v[0]) + int(v[1]))
    if t == ActionType.END_TURN:
        return (A.END_TURN,)
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return (A.BUY_DEV,)
    if t == ActionType.PLAY_KNIGHT_CARD:
        return None
    if t == ActionType.MOVE_ROBBER:
        coord, who = v[0], v[1]
        victim = -1
        if who is not None:
            for i, p in enumerate(state.players):
                if p.color == COLOR_NAMES.get(who, str(who.value).lower()):
                    victim = i
                    break
        kind = A.MOVE_ROBBER
        if (previous is not None and previous.action_type == ActionType.PLAY_KNIGHT_CARD
                and previous.color == action.color):
            kind = A.PLAY_KNIGHT
        return (kind, mapping.coord_to_hex[tuple(coord)], victim)
    if t == ActionType.PLAY_ROAD_BUILDING:
        return (A.PLAY_ROAD_BUILDING,)
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        picks = [RESOURCE_TO_CB[r] for r in v]
        if len(picks) != 2:
            return None
        r1, r2 = sorted(picks)
        return (A.PLAY_YEAR_OF_PLENTY, r1, r2)
    if t == ActionType.PLAY_MONOPOLY:
        return (A.PLAY_MONOPOLY, RESOURCE_TO_CB[v])
    if t == ActionType.MARITIME_TRADE:
        given = [r for r in v[:4] if r is not None]
        if not given:
            return None
        return (A.BANK_TRADE, RESOURCE_TO_CB[given[0]], RESOURCE_TO_CB[v[4]])
    if t in DISCARD_TYPES:
        counts = [0] * 5
        cards = v if isinstance(v, (list, tuple)) else ([v] if v else [])
        for r in cards:
            counts[RESOURCE_TO_CB[r]] += 1
        return (A.DISCARD, tuple(counts))
    return None


_FALLBACK_ORDER = [
    ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT, ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.BUILD_ROAD, ActionType.ROLL, ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT_CARD,
    ActionType.END_TURN,
]


def fallback_action(playable: Sequence[CAction]) -> CAction:
    """A sensible catanatron action when nothing better is available (never raises on a non-empty list)."""
    if not playable:
        raise ValueError("no playable actions")
    for t in _FALLBACK_ORDER:
        for a in playable:
            if a.action_type == t:
                return a
    return playable[0]


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------
class CatanbotPlayer(Player):
    """catanatron ``Player`` driven by a catanbot bot (default: depth-1 heuristic search).

    ``spec`` is a :func:`catanbot.selfplay.make_bot` spec; ``bot`` overrides it
    with a ready instance.  ``strict=True`` re-raises adapter errors (used by
    the tests); otherwise any failure falls back to :func:`fallback_action` and
    is counted in ``stats["errors"]`` so a benchmark never crashes.  ``observe``
    replays every catanatron-logged action through ``bot.observe`` (opponent
    model / political tracking).

    catanatron 3.3 specifics: a ``DISCARD`` prompt runs the bot once, on the
    full hand, for a catanbot ``(DISCARD, counts)``; its first card is played
    and the rest are queued (``stats["pending_discard"]``) for the engine's
    following ``DISCARD_RESOURCE`` prompts.  Observed runs of another
    player's ``DISCARD_RESOURCE`` actions are merged into one ``DISCARD``
    observation delivered with the state before the first card.  The
    domestic-trade prompts ``DECIDE_TRADE`` / ``DECIDE_ACCEPTEES`` are
    answered with ``REJECT_TRADE`` / ``CANCEL_TRADE`` (``stats["trade_prompts"]``).
    """

    def __init__(self, color: Color, spec: str = DEFAULT_SPEC, bot: Optional[Bot] = None, seed: int = 0,
                 strict: bool = False, suppress_trades: bool = True, observe: bool = True):
        super().__init__(color)
        self.spec = spec
        self.bot: Bot = bot if bot is not None else make_bot(spec)
        if isinstance(self.bot, SearchBot) and self.bot.config.trade_proposals != 0:
            self.bot.config = replace(self.bot.config, trade_proposals=0)
        self.seed = seed
        self.rng = random.Random(seed)
        self.strict = strict
        self.suppress_trades = suppress_trades
        self.observe_actions = observe
        self.last_explanation: Optional[str] = None
        self.stats: Dict[str, float] = {}
        self.unmapped_kinds: Dict[str, int] = {}   # kind of every top-ranked action without a catanatron equivalent
        self._game_id = None
        self._mapping: Optional[BoardMapping] = None
        self._shadow: Optional[State] = None
        self._observed = 0
        self._pending_robber: Optional[Tuple[int, int]] = None
        self._pending_discard: List[str] = []             # 3.3: cards still to hand over, one prompt each
        self._knight_state: Optional[GameState] = None   # state a PLAY_KNIGHT_CARD was decided in
        # 3.3: a run of one player's DISCARD_RESOURCE logs being merged: (state before, seat, counts, cards owed)
        self._discard_run: Optional[Tuple[GameState, int, List[int], int]] = None
        self._reset_stats()

    # -- lifecycle --------------------------------------------------------
    def _reset_stats(self) -> None:
        self.stats = {"decisions": 0, "trivial": 0, "searched": 0, "pending_robber": 0, "pending_discard": 0,
                      "trade_prompts": 0, "unmapped_top": 0, "fallback": 0, "errors": 0, "observe_errors": 0,
                      "observed": 0, "search_time": 0.0}

    def reset_state(self) -> None:
        """Forget the previous game (called by the bench; also triggered by a new ``game.id``)."""
        self.bot.reset()
        self.rng = random.Random(self.seed)
        self._game_id = None
        self._mapping = None
        self._shadow = None
        self._observed = 0
        self._pending_robber = None
        self._pending_discard = []
        self._knight_state = None
        self._discard_run = None
        self.last_explanation = None

    def _begin(self, game: Game) -> None:
        self.bot.reset()
        self._game_id = game.id
        self._mapping = mapping_for(game.state.board.map)
        self._shadow = game.state.copy()
        self._observed = len(action_log(game.state))
        self._pending_robber = None
        self._pending_discard = []
        self._knight_state = None
        self._discard_run = None

    # -- observation ------------------------------------------------------
    def _catch_up(self, st: State) -> None:
        """Replay the actions catanatron logged since our last look through ``bot.observe``.

        Every observation is delivered with the state the action was taken in
        (the self-play contract).  A knight is two logged catanatron actions
        (``PLAY_KNIGHT_CARD`` then ``MOVE_ROBBER``) but one catanbot action, so
        the merged ``PLAY_KNIGHT`` observation is delivered with the state the
        *card* was played in (``PHASE_ROLL`` / ``PHASE_MAIN``, knight still in
        hand) rather than the ``PHASE_ROBBER`` state after it, so that
        ``SearchBot.observe`` predicts from the same legal actions.  Likewise
        a 3.3 run of one player's ``DISCARD_RESOURCE`` actions is one catanbot
        ``DISCARD``: it is delivered with the state before the first card,
        where the merged counts are a legal discard (half the hand).  A run
        the log ends in the middle of (our own, while we are being prompted
        card by card) is completed on the next catch-up.
        """
        log = action_log(st)
        n = len(log)
        if self._shadow is None:
            self._shadow = st.copy()
            self._observed = n
            return
        i = self._observed
        while i < n:
            entry = log[i]
            a = log_action(entry)
            prev = log_action(log[i - 1]) if i > 0 else None
            try:
                cb = state_to_catanbot(self._shadow, mapping=self._mapping, suppress_trades=self.suppress_trades)
                if DISCARD_RESOURCE is not None and a.action_type == DISCARD_RESOURCE:
                    self._observe_discard_card(cb, a)
                else:
                    self._flush_discard_run()
                    cb_action = catanatron_action_to_catanbot(a, cb, self._mapping, prev)
                    if a.action_type == ActionType.PLAY_KNIGHT_CARD:
                        self._knight_state = cb
                    elif cb_action is not None:
                        seen = cb
                        if cb_action[0] == A.PLAY_KNIGHT and self._knight_state is not None:
                            seen = self._knight_state
                        self._knight_state = None
                        self.bot.observe(seen, cb_action, self._shadow.color_to_index[a.color])
                        self.stats["observed"] += 1
                    else:
                        self._knight_state = None
            except Exception:
                if self.strict:
                    raise
                self.stats["observe_errors"] += 1
            try:
                replay_entry(self._shadow, entry)
            except Exception:
                if self.strict:
                    raise
                # Shadow diverged: resynchronise from the live state and stop observing this gap.
                self._shadow = st.copy()
                self._observed = n
                self._knight_state = None
                self._discard_run = None
                self.stats["observe_errors"] += 1
                return
            i += 1
        self._observed = n

    def _observe_discard_card(self, cb: GameState, a: CAction) -> None:
        """Merge a logged 3.3 ``DISCARD_RESOURCE`` into the current run (flushed once complete)."""
        seat = self._shadow.color_to_index[a.color]
        run = self._discard_run
        if run is None or run[1] != seat:
            self._flush_discard_run()
            owed = cb.players[seat].total_resources // 2     # the engine's count, fixed at the roll
            run = self._discard_run = (cb, seat, [0] * 5, owed)
        run[2][RESOURCE_TO_CB[a.value]] += 1
        if sum(run[2]) >= run[3]:
            self._flush_discard_run()

    def _flush_discard_run(self) -> None:
        run = self._discard_run
        self._discard_run = None
        if run is None or not any(run[2]):
            return
        cb, seat, counts, _ = run
        self.bot.observe(cb, (A.DISCARD, tuple(counts)), seat)
        self.stats["observed"] += 1

    # -- decision ---------------------------------------------------------
    def decide(self, game: Game, playable_actions):
        self.stats["decisions"] += 1
        playable = list(playable_actions)
        try:
            if game.id != self._game_id or self._mapping is None:
                self._begin(game)
            if self.observe_actions:
                self._catch_up(game.state)
            return self._choose(game, playable)
        except Exception:
            if self.strict:
                raise
            self.stats["errors"] += 1
            self._pending_robber = None
            self._pending_discard = []
            return fallback_action(playable)

    def _choose(self, game: Game, playable: List[CAction]) -> CAction:
        st = game.state
        m = self._mapping
        assert m is not None
        prompt = st.current_prompt
        if DISCARD_RESOURCE is not None and prompt == ActionPrompt.DISCARD:
            return self._choose_discard(game, playable)
        if prompt in TRADE_PROMPTS:
            return self._answer_trade(playable)
        if len(playable) == 1:
            self.stats["trivial"] += 1
            return playable[0]
        index = index_playable(playable)
        if st.current_prompt == ActionPrompt.MOVE_ROBBER and self._pending_robber is not None:
            h, victim = self._pending_robber
            self._pending_robber = None
            who = st.colors[victim] if victim >= 0 else None
            a = index.get((ActionType.MOVE_ROBBER, (m.hex_to_coord[h], who)))
            if a is not None:
                self.stats["pending_robber"] += 1
                return a
        cb = to_catanbot_state(game, self.color, m, self.suppress_trades)
        legal = E.legal_actions(cb)
        lookup: Dict[A.Action, CAction] = {}
        cb_legal: List[A.Action] = []
        for a in legal:
            key = catanbot_action_to_key(a, cb, m, st.colors)
            if key is None:
                continue
            ca = index.get(key)
            if ca is not None:
                lookup[a] = ca
                cb_legal.append(a)
        if not cb_legal:
            self.stats["fallback"] += 1
            return fallback_action(playable)
        distinct = {id(ca) for ca in lookup.values()}
        if len(distinct) == 1:
            chosen = cb_legal[0]
            self.stats["trivial"] += 1
        else:
            t0 = time.perf_counter()
            decision = self.bot.decide(cb, cb_legal, self.rng)
            self.stats["search_time"] += time.perf_counter() - t0
            self.stats["searched"] += 1
            ranked = [r.action for r in (getattr(self.bot, "last_results", None) or [])]
            chosen = None
            for a in ranked:
                if a in lookup:
                    chosen = a
                    break
            if ranked and chosen is not ranked[0]:
                self.stats["unmapped_top"] += 1
                self.unmapped_kinds[ranked[0][0]] = self.unmapped_kinds.get(ranked[0][0], 0) + 1
            if chosen is None and decision in lookup:
                chosen = decision
            if chosen is None:
                self.stats["fallback"] += 1
                priors = action_priors(cb, cb_legal, E.acting_player(cb))
                chosen = cb_legal[max(range(len(cb_legal)), key=lambda i: priors[i])]
            try:
                self.last_explanation = self.bot.explain(cb)
            except Exception:
                self.last_explanation = None
        if chosen[0] == A.PLAY_KNIGHT:
            self._pending_robber = (int(chosen[1]), int(chosen[2]))
        return lookup[chosen]

    def _answer_trade(self, playable: List[CAction]) -> CAction:
        """3.3 domestic-trade prompts: decline (``REJECT_TRADE`` as a responder, ``CANCEL_TRADE`` as offerer)."""
        self.stats["trade_prompts"] += 1
        for t in _TRADE_ANSWERS:
            for a in playable:
                if a.action_type == t:
                    return a
        return fallback_action(playable)

    def _choose_discard(self, game: Game, playable: List[CAction]) -> CAction:
        """3.3 ``DISCARD`` prompt: plan the whole discard once, hand it over one ``DISCARD_RESOURCE`` at a time."""
        st = game.state
        m = self._mapping
        assert m is not None
        index = index_playable(playable)
        if self._pending_discard:
            a = index.get((DISCARD_RESOURCE, self._pending_discard[0]))
            if a is not None:
                self._pending_discard.pop(0)
                self.stats["pending_discard"] += 1
                return a
            self._pending_discard = []   # the plan no longer matches the hand: plan again
        if len(playable) == 1:
            self.stats["trivial"] += 1
            return playable[0]
        seat = st.color_to_index[self.color]
        owed = int(st.discard_counts[seat])
        cb = to_catanbot_state(game, self.color, m, self.suppress_trades)
        legal = [a for a in E.legal_actions(cb) if a[0] == A.DISCARD and sum(a[1]) == owed]
        if not legal:
            # Only reachable mid-run (the engine's remaining count is no longer half the hand).
            self.stats["fallback"] += 1
            return fallback_action(playable)
        if len(legal) == 1:
            decision = legal[0]
            self.stats["trivial"] += 1
        else:
            t0 = time.perf_counter()
            decision = self.bot.decide(cb, legal, self.rng)
            self.stats["search_time"] += time.perf_counter() - t0
            self.stats["searched"] += 1
            if decision not in legal:
                decision = legal[0]
            try:
                self.last_explanation = self.bot.explain(cb)
            except Exception:
                self.last_explanation = None
        plan = [CB_TO_RESOURCE[r] for r in range(5) for _ in range(int(decision[1][r]))]
        a = index.get((DISCARD_RESOURCE, plan[0])) if plan else None
        if a is None:
            self.stats["fallback"] += 1
            return fallback_action(playable)
        self._pending_discard = plan[1:]
        return a


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------
def make_game(players: Sequence[Player], seed: int, vps_to_win: int = 10, discard_limit: int = 7) -> Game:
    """``Game`` with ``players`` seated exactly in the given order.

    catanatron shuffles the seating with the game seed; since no action has been
    taken yet every per-seat field is still identical, so the seating can be
    reordered deterministically (``seed`` must be non-zero: catanatron treats 0
    as "pick a random seed").  ``discard_limit`` is catanatron's (it only
    governs who discards *first* on a 7, see :func:`state_to_catanbot`).
    """
    if not seed:
        raise ValueError("seed must be non-zero (catanatron treats 0 as random)")
    game = Game(list(players), seed=seed, vps_to_win=vps_to_win, discard_limit=discard_limit)
    st = game.state
    if list(st.players) != list(players):
        st.players = list(players)
        st.colors = tuple(p.color for p in players)
        st.color_to_index = {c: i for i, c in enumerate(st.colors)}
        acts = generate_playable_actions(st)
        if hasattr(game, "playable_actions"):   # 3.3 keeps them on the Game ...
            game.playable_actions = acts
        else:                                   # ... 3.2.1 on the State
            st.playable_actions = acts
    return game


def play_game(players: Sequence[Player], seed: int, vps_to_win: int = 10,
              discard_limit: int = 7) -> Dict[str, object]:
    """Play one seated game to the end; returns a summary dict (winner may be ``None`` at the turn cap)."""
    for p in players:
        if hasattr(p, 'reset_state'):
            p.reset_state()
    game = make_game(players, seed, vps_to_win, discard_limit)
    t0 = time.perf_counter()
    winner = game.play()
    st = game.state
    vps = [int(st.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"]) for i in range(len(st.colors))]
    return {
        "seed": seed,
        "winner": winner.value if winner is not None else None,
        "winner_seat": st.color_to_index[winner] if winner is not None else -1,
        "colors": [c.value for c in st.colors],
        "vps": vps,
        "turns": int(st.num_turns),
        "actions": len(action_log(st)),
        "duration": time.perf_counter() - t0,
    }
