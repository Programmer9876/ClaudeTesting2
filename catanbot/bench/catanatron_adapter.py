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

4. Domestic trading on 3.3 (``CatanbotPlayer(suppress_trades=False)``): a
   catanbot ``PROPOSE_TRADE`` is played as ``OFFER_TRADE`` (catanatron never
   lists offers in ``playable_actions``; the engine accepts one from the turn
   player after the roll, with no per-turn limit, so catanbot's own cap of
   :data:`catanbot.engine.MAX_TRADE_PROPOSALS_PER_TURN` is enforced through
   ``trades_this_turn``), ``DECIDE_TRADE`` converts to ``PHASE_TRADE_RESPONSE``
   and ``DECIDE_ACCEPTEES`` to ``PHASE_TRADE_SELECT`` so the bot decides them
   like in self-play (``EXECUTE_TRADE`` -> ``CONFIRM_TRADE`` with that partner),
   and the logged trade actions are observed (opponent model, politics).
   :class:`BenchOpponent` wraps a catanatron player for the bench: it times
   every decision and optionally replaces its answer to an offer by a rule
   (:data:`TRADE_MODES`), because catanatron's own players answer offers
   degenerately (``docs/BENCHMARKS.md``, "Domestic trading against catanatron 3.3").

5. Replayable game logs (``play_game(record_log=True)``, :func:`game_log`,
   :func:`rebuild_game`, :func:`replay_log_action`): the board, robber start,
   seating and shuffled development deck of a game plus every logged action
   with its chance outcome (3.3 ``ActionRecord.result``; on 3.2.1 read from the
   fully specified logged action) and the final state with a full-state
   :func:`state_fingerprint`.  A fresh game rebuilt on the logged board with the
   logged outcomes reproduces the original exactly (``scripts/replay_catanatron.py``).

6. Information modes (:data:`INFO_MODES`, ``CatanbotPlayer(info=...)``): ``full`` (the
   default) converts the true state with every card known, as described in 2.;
   ``counted`` hands the bot only what a Colonist.io player knows - public events counted
   from the action log by :class:`catanbot.bench.public_info.PublicInfoTracker`, opponents'
   hidden cards sampled from that posterior (``docs/BENCHMARKS.md``, "Information modes").

Known semantic differences (see ``docs/BENCHMARKS.md``): player-to-player
trading is off by default (catanatron 3.2.1 has none; on 3.3 the default
``suppress_trades=True`` never offers and answers the domestic-trade prompts
with ``REJECT_TRADE`` / ``CANCEL_TRADE``; catanatron's own players never
offer), discards are chosen randomly by the catanatron 3.2.1
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

Player count: 2 to 4 seats (``play_game`` / ``make_game`` seat whatever players they are
given; :func:`colors_for` gives the bench's colours for a count).  Nothing in the conversion
assumes four players: seats follow ``State.colors``, the discard queue is built from the seated
players (3.3's ``discard_counts`` / 3.2.1's hard-coded rule), a robber victim is mapped by colour
to its seat and catanbot's engine plays the base rules for any 2-4 players.  Both engines
play a 2-player game with the unchanged base rules (full board, 7-card discard limit, the
robber may steal from the only opponent, setup order 0-1-1-0), not the official two-player
variant.

Both catanatron generations are supported by feature detection (:data:`API_33`):
the PyPI 3.2.1 wheel (``State.actions``, ``State.playable_actions``,
``catanatron.state.apply_action``, one random ``DISCARD``, 3-tuple robber
values) and the 3.3 engine of the GitHub checkout (``State.action_records``
of ``ActionRecord(action, result)``, ``Game.playable_actions``,
``catanatron.apply_action.apply_action(state, action, record)``, per-card
``DISCARD_RESOURCE`` with ``State.discard_counts``, 2-tuple robber values,
``DECIDE_TRADE`` / ``DECIDE_ACCEPTEES`` prompts and the domestic-trade actions,
:data:`DOMESTIC_TRADING`).  The helpers
:func:`action_log`, :func:`log_action`, :func:`playable_actions_of`,
:func:`apply_action` and :func:`replay_entry` hide the differences.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

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
    PHASE_TRADE_RESPONSE,
    PHASE_TRADE_SELECT,
    GameState,
    Player as CBPlayer,
    TradeOffer,
)

__all__ = [
    "DEFAULT_SPEC",
    "COLORS",
    "COLOR_NAMES",
    "colors_for",
    "API_33",
    "CATANATRON_VERSION",
    "DISCARD_LEGACY",
    "DISCARD_RESOURCE",
    "DISCARD_TYPES",
    "TRADE_PROMPTS",
    "DOMESTIC_TRADING",
    "DOMESTIC_TRADE_TYPES",
    "TRADE_MODES",
    "INFO_MODES",
    "DEFAULT_INFO_SAMPLES",
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
    "BenchOpponent",
    "timing_summary",
    "make_game",
    "play_game",
    "LOG_FORMAT",
    "encode_log_entry",
    "decode_log_action",
    "decode_log_result",
    "encode_board",
    "decode_board",
    "state_summary",
    "state_fingerprint",
    "game_log_header",
    "game_log",
    "rebuild_game",
    "replay_log_action",
    "ReplayMismatch",
]

DEFAULT_SPEC = "search:depth=1,evaluator=heuristic"

#: catanatron seat colours in the order the benchmark assigns them.
COLORS: Tuple[Color, ...] = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
COLOR_NAMES: Dict[Color, str] = {Color.RED: "red", Color.BLUE: "blue", Color.ORANGE: "orange", Color.WHITE: "white"}


def colors_for(num_players: int = len(COLORS)) -> Tuple[Color, ...]:
    """The bench's seat colours for a ``num_players`` game (2-4): the first ``num_players`` of
    :data:`COLORS` (a 1v1 game seats RED and BLUE)."""
    if not 2 <= int(num_players) <= len(COLORS):
        raise ValueError(f"catanatron games have 2 to {len(COLORS)} players, got {num_players}")
    return COLORS[:int(num_players)]

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
#: 3.3 domestic-trade prompts: ``DECIDE_TRADE`` (a responder accepts / rejects ``State.current_trade``)
#: and ``DECIDE_ACCEPTEES`` (the offerer confirms one accepter or cancels).  ``None`` on 3.2.1.
DECIDE_TRADE = getattr(ActionPrompt, "DECIDE_TRADE", None)
DECIDE_ACCEPTEES = getattr(ActionPrompt, "DECIDE_ACCEPTEES", None)
TRADE_PROMPTS: Tuple[ActionPrompt, ...] = tuple(p for p in (DECIDE_TRADE, DECIDE_ACCEPTEES) if p is not None)
#: 3.3 domestic-trade action types (``None`` on 3.2.1).  Values: ``OFFER_TRADE`` the 10-tuple
#: (5 counts offered, 5 counts asked, in WOOD BRICK SHEEP WHEAT ORE order - catanbot's order);
#: ``ACCEPT_TRADE`` / ``REJECT_TRADE`` the 11-tuple ``State.current_trade`` (offer + offerer seat);
#: ``CONFIRM_TRADE`` the offer's 10 counts + the accepter's ``Color``; ``CANCEL_TRADE`` ``None``.
AT_OFFER = getattr(ActionType, "OFFER_TRADE", None)
AT_ACCEPT = getattr(ActionType, "ACCEPT_TRADE", None)
AT_REJECT = getattr(ActionType, "REJECT_TRADE", None)
AT_CONFIRM = getattr(ActionType, "CONFIRM_TRADE", None)
AT_CANCEL = getattr(ActionType, "CANCEL_TRADE", None)
DOMESTIC_TRADE_TYPES: Tuple[ActionType, ...] = tuple(
    t for t in (AT_OFFER, AT_ACCEPT, AT_REJECT, AT_CONFIRM, AT_CANCEL) if t is not None)
#: True when the engine has player-to-player trading (3.3).
DOMESTIC_TRADING: bool = AT_OFFER is not None and DECIDE_TRADE is not None
_TRADE_ANSWERS: Tuple[ActionType, ...] = tuple(t for t in (AT_REJECT, AT_CANCEL) if t is not None)
#: How the benchmark handles domestic trades (``scripts/bench_catanatron.py --trades``):
#: ``off`` - catanbot never offers (and declines any offer);
#: ``native`` - catanbot offers, each catanatron opponent answers with its own ``decide``
#: (a player that raises is counted in ``BenchOpponent.trade_stats["errors"]`` and rejects);
#: ``value`` - catanbot offers, opponents accept iff their value function rises with the trade;
#: ``fair`` - like ``value`` but they also refuse to give more cards than they get and refuse
#: a proposer within 2 VP of winning.  ``value`` / ``fair`` are OUR model of a sensible
#: opponent (:class:`BenchOpponent`), not catanatron's behaviour.
TRADE_MODES: Tuple[str, ...] = ("off", "native", "value", "fair")
#: What catanbot may know (``CatanbotPlayer(info=...)``, ``scripts/bench_catanatron.py --info``):
#: ``full`` - catanatron's true state, every hand and development card (the default, unchanged);
#: ``counted`` - what a Colonist.io player knows: public events counted from the action log,
#: opponents' hidden cards sampled from that posterior (:mod:`catanbot.bench.public_info`).
INFO_MODES: Tuple[str, ...] = ("full", "counted")
#: Determinizations searched per decision in the counted mode (``--info-samples``).
DEFAULT_INFO_SAMPLES = 4


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


def _offers_this_turn(log: Sequence) -> int:
    """Number of ``OFFER_TRADE`` actions logged since the last ``END_TURN`` (0 on 3.2.1)."""
    if AT_OFFER is None:
        return 0
    n = 0
    for entry in reversed(log):
        t = log_action(entry).action_type
        if t == ActionType.END_TURN:
            break
        if t == AT_OFFER:
            n += 1
    return n


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
    no player trading in catanatron 3.2.1, and the benchmark only offers on
    3.3 with ``--trades`` other than ``off``); otherwise ``trades_this_turn`` is
    the number of ``OFFER_TRADE`` logged this turn (capped at catanbot's
    per-turn maximum; catanatron itself has no limit).  The 3.3 domestic-trade
    prompts convert to catanbot's trade phases (:func:`_convert_trade_prompt`):
    ``DECIDE_TRADE`` to ``PHASE_TRADE_RESPONSE`` for the asked seat,
    ``DECIDE_ACCEPTEES`` to ``PHASE_TRADE_SELECT`` for the offerer.  On 3.3 a
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
    s.trades_this_turn = (E.MAX_TRADE_PROPOSALS_PER_TURN if suppress_trades
                          else min(E.MAX_TRADE_PROPOSALS_PER_TURN, _offers_this_turn(action_log(st))))
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
    elif prompt in TRADE_PROMPTS:
        _convert_trade_prompt(st, s, prompt)
    else:  # PLAY_TURN
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


def _convert_trade_prompt(st: State, s: GameState, prompt: ActionPrompt) -> None:
    """Fill ``s`` (turn player = offerer, after the roll) for a 3.3 domestic-trade prompt.

    ``State.current_trade`` is ``(5 offered, 5 asked, offerer seat)``.  catanatron asks
    the other seats in seat order (0, 1, ...; the offerer's own seat included when it
    is not seat 0 - an engine quirk) and records only the acceptances
    (``State.acceptees``), so every seat before the asked one has answered: accepted
    if its acceptee flag is set, rejected otherwise.  A seat still to be asked that
    cannot pay is marked rejected, as catanbot's engine does at the proposal.
    ``DECIDE_ACCEPTEES`` becomes ``PHASE_TRADE_SELECT`` with every answer known.
    """
    trade = st.current_trade
    proposer = int(trade[10])
    give = [int(x) for x in trade[:5]]
    get = [int(x) for x in trade[5:10]]
    n = len(s.players)
    acceptees = st.acceptees
    s.current = proposer
    s.dice = _last_roll_this_turn(action_log(st))
    offer = TradeOffer(proposer, give, get)
    if prompt == DECIDE_TRADE:
        responder = int(st.current_player_index)
        for i in range(n):
            if i == proposer or i == responder:
                continue
            if i < responder:
                offer.responses[i] = bool(acceptees[i])
            elif any(s.players[i].resources[r] < get[r] for r in range(5)):
                offer.responses[i] = False
        s.phase = PHASE_TRADE_RESPONSE
        s.trade_responder = responder
    else:
        for i in range(n):
            if i != proposer:
                offer.responses[i] = bool(acceptees[i])
        s.phase = PHASE_TRADE_SELECT
        s.trade_responder = -1
    s.pending_trade = offer


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
    """Hashable identity of a catanatron action (road orientation / robber card normalised;
    a trade answer is identified by its type alone, a ``CONFIRM_TRADE`` by the partner)."""
    t = action.action_type
    v = action.value
    if t in DOMESTIC_TRADE_TYPES:
        if t == AT_CONFIRM:
            return (t, v[10])
        if t == AT_OFFER:
            return (t, tuple(int(x) for x in v[:10]))
        return (t, None)
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
    catanatron equivalent (forced rolls; the player-trade actions on 3.2.1).

    On 3.3 ``PROPOSE_TRADE`` is ``OFFER_TRADE`` with the 10 counts (never in
    ``playable_actions``: :class:`CatanbotPlayer` builds the action itself),
    ``ACCEPT_TRADE`` / ``REJECT_TRADE`` / ``CANCEL_TRADE`` are the same-named
    types and ``(EXECUTE_TRADE, j)`` is ``CONFIRM_TRADE`` with seat ``j``'s colour.

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
    if not DOMESTIC_TRADING:
        return None
    if kind == A.PROPOSE_TRADE:
        return (AT_OFFER, tuple(int(x) for x in action[1]) + tuple(int(x) for x in action[2]))
    if kind == A.ACCEPT_TRADE:
        return (AT_ACCEPT, None)
    if kind == A.REJECT_TRADE:
        return (AT_REJECT, None)
    if kind == A.EXECUTE_TRADE:
        return (AT_CONFIRM, colors[action[1]])
    if kind == A.CANCEL_TRADE:
        return (AT_CANCEL, None)
    return None


def _seat_of(state: GameState, color: Color) -> int:
    """catanbot seat of a catanatron colour in a converted state (-1 if not seated)."""
    name = COLOR_NAMES.get(color, str(getattr(color, "value", color)).lower())
    for i, p in enumerate(state.players):
        if p.color == name:
            return i
    return -1


def _trade_to_catanbot(action: CAction, state: GameState) -> Optional[A.Action]:
    """catanbot action for a logged / playable 3.3 domestic-trade action in ``state`` (see
    :func:`catanatron_action_to_catanbot`)."""
    t = action.action_type
    v = action.value
    if t == AT_OFFER:
        return (A.PROPOSE_TRADE, tuple(int(x) for x in v[:5]), tuple(int(x) for x in v[5:10]))
    if t == AT_CANCEL:
        return (A.CANCEL_TRADE,)
    if t == AT_CONFIRM:
        partner = _seat_of(state, v[10])
        return (A.EXECUTE_TRADE, partner) if partner >= 0 else None
    # ACCEPT_TRADE / REJECT_TRADE
    offer = state.pending_trade
    seat = _seat_of(state, action.color)
    if state.phase != PHASE_TRADE_RESPONSE or offer is None or seat < 0 or seat == offer.proposer:
        return None   # catanatron also asks the offerer about its own offer: no catanbot equivalent
    if (t == AT_REJECT and state.players[seat].hand_known
            and any(state.players[seat].resources[r] < offer.get[r] for r in range(5))):
        return None   # a seat that cannot pay is auto-rejected by catanbot's engine, never asked
        #               (a hidden hand - the counted information mode - cannot tell: every answer counts)
    return (A.ACCEPT_TRADE,) if t == AT_ACCEPT else (A.REJECT_TRADE,)


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
    ``DISCARD None``.  The 3.3 domestic-trade actions map to ``PROPOSE_TRADE``,
    ``ACCEPT_TRADE`` / ``REJECT_TRADE`` (``state`` must be the converted
    ``DECIDE_TRADE`` state; ``None`` for the offerer's answer to its own offer
    and for the forced rejection of a seat that cannot pay, which catanbot's
    engine never asks), ``EXECUTE_TRADE`` and ``CANCEL_TRADE``.
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
    if t in DOMESTIC_TRADE_TYPES:
        return _trade_to_catanbot(action, state)
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
    observation delivered with the state before the first card.

    Domestic trades (3.3): with ``suppress_trades=True`` (the default) the bot
    never offers and the prompts ``DECIDE_TRADE`` / ``DECIDE_ACCEPTEES`` are
    answered with ``REJECT_TRADE`` / ``CANCEL_TRADE``.  With
    ``suppress_trades=False`` the bot's ``PROPOSE_TRADE`` choices are played as
    ``OFFER_TRADE`` (at most :data:`catanbot.engine.MAX_TRADE_PROPOSALS_PER_TURN`
    per turn, only cards we hold), an incoming offer is decided by the bot in
    ``PHASE_TRADE_RESPONSE`` and the accepters of our offer in
    ``PHASE_TRADE_SELECT``; the engine's question to the offerer about its own
    offer is answered ``REJECT_TRADE`` without a search.  ``stats`` counts
    ``offers``, ``offers_accepted`` (offers at least one seat accepted),
    ``trades_confirmed`` / ``trades_cancelled``, ``offers_received`` /
    ``offers_accepted_by_us`` and ``self_offer_prompts``.  On 3.2.1 trades are
    always suppressed.

    ``times`` / ``choice_times`` hold the wall time (seconds) of every
    ``decide`` call / of those with more than one playable action, including
    the adapter's conversion and observation work (the compute-fairness
    counterpart of :class:`BenchOpponent`'s timing).

    Information mode (:data:`INFO_MODES`, ``docs/BENCHMARKS.md`` "Information
    modes"): ``info="full"`` (the default) hands the bot catanatron's true state
    with every hand and development card known, exactly as before.
    ``info="counted"`` gives it what a Colonist.io player knows: a
    :class:`catanbot.bench.public_info.PublicInfoTracker` (``tracker``) follows
    the action log with public information only (``discards_public`` makes the
    cards of a discard public), every observation is delivered with the public
    view (opponents ``hand_known=False`` / ``dev_known=False`` with exact sizes
    and counts, cards zeroed; opponents' discards whose cards are hidden are not
    observed), our legal actions come from the tracker's canonical view (they
    depend on public information only) and every searched decision averages the
    bot's ranking over ``info_samples`` determinizations sampled from the
    tracker (mean value over the samples that ranked an action, actions ranked
    by at least half of them first).  ``stats`` then also counts
    ``info_samples`` (determinizations searched), ``info_uncertain`` (searched
    decisions with an opponent's hand not known exactly), ``info_errors`` and
    the tracker's hidden events.
    """

    def __init__(self, color: Color, spec: str = DEFAULT_SPEC, bot: Optional[Bot] = None, seed: int = 0,
                 strict: bool = False, suppress_trades: bool = True, observe: bool = True,
                 info: str = "full", info_samples: int = DEFAULT_INFO_SAMPLES, discards_public: bool = False):
        super().__init__(color)
        if info not in INFO_MODES:
            raise ValueError(f"info must be one of {INFO_MODES}, got {info!r}")
        if int(info_samples) < 1:
            raise ValueError(f"info_samples must be >= 1, got {info_samples!r}")
        self.info = info
        self.info_samples = int(info_samples)
        self.discards_public = bool(discards_public)
        self.tracker = None                 # PublicInfoTracker in the counted mode (created per game)
        self.spec = spec
        self.bot: Bot = bot if bot is not None else make_bot(spec)
        self.suppress_trades = bool(suppress_trades) or not DOMESTIC_TRADING
        if self.suppress_trades:
            # No proposal can be played: do not let the search spend nodes on them.
            for b in (self.bot, getattr(self.bot, "inner", None)):
                if isinstance(b, SearchBot) and b.config.trade_proposals != 0:
                    b.config = replace(b.config, trade_proposals=0)
        self.seed = seed
        self.rng = random.Random(seed)
        self.strict = strict
        self.observe_actions = observe
        self.last_explanation: Optional[str] = None
        self.stats: Dict[str, float] = {}
        self.times: List[float] = []          # wall seconds of every decide() call
        self.choice_times: List[float] = []   # ... of those with more than one playable action
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
                      "observed": 0, "search_time": 0.0,
                      "offers": 0, "offers_accepted": 0, "trades_confirmed": 0, "trades_cancelled": 0,
                      "offers_received": 0, "offers_accepted_by_us": 0, "self_offer_prompts": 0}
        if self.info == "counted":
            self.stats.update({"info_samples": 0, "info_uncertain": 0, "info_errors": 0, "info_resets": 0,
                               "info_max_hypotheses": 0, "hidden_steals": 0, "hidden_discards": 0,
                               "hidden_dev_draws": 0})
        self.times = []
        self.choice_times = []

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
        self.tracker = None

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
        self.tracker = None
        if self.info == "counted":
            from .public_info import PublicInfoTracker
            self._tracker_totals = {"hidden_steals": 0, "hidden_discards": 0, "hidden_dev_draws": 0,
                                    "info_resets": 0}
            self.tracker = PublicInfoTracker(self.color, discards_public=self.discards_public,
                                             vps_to_win=int(getattr(game, "vps_to_win", 10)))
            self.tracker.start(game.state)   # replays the log so far from the initial position

    def _follow_tracker(self, st: State) -> None:
        """Counted mode: bring the public-information tracker up to the live log (a divergence is
        an ``info_errors`` and a resync; ``strict`` re-raises)."""
        tr = self.tracker
        try:
            tr.follow(st)
        except Exception:
            if self.strict:
                raise
            self.stats["info_errors"] += 1
            tr.resync(st)
        # per-game tracker counts, accumulated into the player's stats across games
        tot = self._tracker_totals
        cur = {"hidden_steals": tr.stats["hidden_steals"], "hidden_discards": tr.stats["hidden_discards"],
               "hidden_dev_draws": tr.stats["hidden_dev_draws"], "info_resets": int(tr.counter.stats["resets"])}
        for k, v in cur.items():
            self.stats[k] += v - tot[k]
            tot[k] = v
        self.stats["info_max_hypotheses"] = max(self.stats["info_max_hypotheses"],
                                                int(tr.counter.stats["max_hypotheses"]))

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

        Counted information mode: every observation is delivered with the
        *public* view of its state (:func:`catanbot.bench.public_info.redact_state`)
        and an opponent's discard whose cards are hidden is not observed.
        """
        log = action_log(st)
        n = len(log)
        if self._shadow is None:
            self._shadow = st.copy()
            self._observed = n
            return
        redact = None
        if self.tracker is not None:
            from .public_info import redact_state as redact
        i = self._observed
        while i < n:
            entry = log[i]
            a = log_action(entry)
            prev = log_action(log[i - 1]) if i > 0 else None
            try:
                cb = state_to_catanbot(self._shadow, mapping=self._mapping, suppress_trades=self.suppress_trades)
                hidden_discard = False
                if redact is not None:
                    cb = redact(cb, self._shadow.color_to_index[self.color])
                    hidden_discard = (a.action_type in DISCARD_TYPES and a.color != self.color
                                      and not self.discards_public)
                if DISCARD_RESOURCE is not None and a.action_type == DISCARD_RESOURCE:
                    if hidden_discard:
                        self._flush_discard_run()
                    else:
                        self._observe_discard_card(cb, a)
                else:
                    self._flush_discard_run()
                    cb_action = (None if hidden_discard
                                 else catanatron_action_to_catanbot(a, cb, self._mapping, prev))
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
            p = cb.players[seat]
            owed = (p.total_resources if p.hand_known else p.hand_size) // 2   # the engine's count, fixed at the roll
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
        t0 = time.perf_counter()
        self.stats["decisions"] += 1
        playable = list(playable_actions)
        try:
            if game.id != self._game_id or self._mapping is None:
                self._begin(game)
            if self.tracker is not None:
                self._follow_tracker(game.state)
            if self.observe_actions:
                self._catch_up(game.state)
            return self._choose(game, playable)
        except Exception:
            if self.strict:
                raise
            self.stats["errors"] += 1
            self._pending_robber = None
            self._pending_discard = []
            return self._fallback(game, playable)
        finally:
            dt = time.perf_counter() - t0
            self.times.append(dt)
            if len(playable) > 1:
                self.choice_times.append(dt)

    @staticmethod
    def _fallback(game: Game, playable: List[CAction]) -> CAction:
        if game.state.current_prompt in TRADE_PROMPTS:
            for t in _TRADE_ANSWERS:          # never accept / confirm a trade by accident
                for a in playable:
                    if a.action_type == t:
                        return a
        return fallback_action(playable)

    # -- domestic trades (3.3) --------------------------------------------
    def _may_offer(self, st: State) -> bool:
        """True when catanatron would accept an ``OFFER_TRADE`` from us now and the per-turn cap allows one."""
        if self.suppress_trades or st.current_prompt != ActionPrompt.PLAY_TURN or st.is_road_building:
            return False
        seat = st.color_to_index[self.color]
        if int(st.current_turn_index) != seat or not st.player_state[f"P{seat}_HAS_ROLLED"]:
            return False
        if not any(st.player_state[f"P{seat}_{r}_IN_HAND"] for r in CB_TO_RESOURCE):
            return False
        return _offers_this_turn(action_log(st)) < E.MAX_TRADE_PROPOSALS_PER_TURN

    def _offer_action(self, action: A.Action, st: State) -> Optional[CAction]:
        """The ``OFFER_TRADE`` for a catanbot ``PROPOSE_TRADE`` (``None`` unless well formed and affordable)."""
        if len(action) != 3 or len(action[1]) != 5 or len(action[2]) != 5:
            return None
        give = tuple(int(x) for x in action[1])
        get = tuple(int(x) for x in action[2])
        if sum(give) <= 0 or sum(get) <= 0 or any(g < 0 for g in give + get):
            return None
        if any(give[r] and get[r] for r in range(5)):
            return None
        seat = st.color_to_index[self.color]
        if any(st.player_state[f"P{seat}_{CB_TO_RESOURCE[r]}_IN_HAND"] < give[r] for r in range(5)):
            return None
        return CAction(self.color, AT_OFFER, give + get)

    def _count_choice(self, chosen: A.Action) -> None:
        kind = chosen[0]
        if kind == A.PROPOSE_TRADE:
            self.stats["offers"] += 1
        elif kind == A.EXECUTE_TRADE:
            self.stats["trades_confirmed"] += 1
        elif kind == A.CANCEL_TRADE:
            self.stats["trades_cancelled"] += 1
        elif kind == A.ACCEPT_TRADE:
            self.stats["offers_accepted_by_us"] += 1

    def _choose(self, game: Game, playable: List[CAction]) -> CAction:
        st = game.state
        m = self._mapping
        assert m is not None
        prompt = st.current_prompt
        if DISCARD_RESOURCE is not None and prompt == ActionPrompt.DISCARD:
            return self._choose_discard(game, playable)
        if prompt in TRADE_PROMPTS:
            self.stats["trade_prompts"] += 1
            if self.suppress_trades:
                return self._answer_trade(playable)
            if prompt == DECIDE_TRADE:
                if int(st.current_trade[10]) == st.color_to_index[self.color]:
                    # catanatron asks the offerer too (seats after seat 0): never "accept" our own offer
                    self.stats["self_offer_prompts"] += 1
                    return self._answer_trade(playable)
                self.stats["offers_received"] += 1
            else:
                self.stats["offers_accepted"] += 1
        may_offer = self._may_offer(st)
        if len(playable) == 1 and not may_offer:
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
        cb, view = self._views(game)
        legal = E.legal_actions(view)
        lookup: Dict[A.Action, CAction] = {}
        cb_legal: List[A.Action] = []
        for a in legal:
            key = catanbot_action_to_key(a, view, m, st.colors)
            if key is None:
                continue
            ca = index.get(key)
            if ca is None and may_offer and a[0] == A.PROPOSE_TRADE:
                ca = self._offer_action(a, st)   # offers are never listed in playable_actions
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
            decision, ranked = self._bot_decide(cb, cb_legal)
            self.stats["search_time"] += time.perf_counter() - t0
            self.stats["searched"] += 1
            if may_offer:
                # The search may rank proposals outside the engine's bounded candidate list
                # (intermediary / political deals): playable if well formed and affordable.
                for a in ranked + [decision]:
                    if a and a[0] == A.PROPOSE_TRADE and a not in lookup:
                        ca = self._offer_action(a, st)
                        if ca is not None:
                            lookup[a] = ca
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
                priors = action_priors(view, cb_legal, E.acting_player(view))
                chosen = cb_legal[max(range(len(cb_legal)), key=lambda i: priors[i])]
            try:
                self.last_explanation = self.bot.explain(cb)
            except Exception:
                self.last_explanation = None
        if chosen[0] == A.PLAY_KNIGHT:
            self._pending_robber = (int(chosen[1]), int(chosen[2]))
        self._count_choice(chosen)
        return lookup[chosen]

    # -- information modes --------------------------------------------------
    def _views(self, game: Game) -> Tuple[GameState, GameState]:
        """``(state for the bot, state our legal actions are generated on)``.

        ``full``: catanatron's true state, twice (exactly the pre-information-mode behaviour).
        ``counted``: the tracker's public view (hidden cards zeroed) and its canonical view (the
        most likely hand hypothesis filled in: our legal actions only depend on public facts).
        """
        cb = to_catanbot_state(game, self.color, self._mapping, self.suppress_trades)
        if self.tracker is None:
            return cb, cb
        pub = self.tracker.public_view(cb)
        return pub, self.tracker.canonical_view(pub)

    def _bot_decide(self, state: GameState, legal: List[A.Action]) -> Tuple[A.Action, List[A.Action]]:
        """``(decision, ranked actions)`` of the bot on ``state`` (see :meth:`_decide_counted`)."""
        if self.tracker is None:
            decision = self.bot.decide(state, legal, self.rng)
            return decision, [r.action for r in (getattr(self.bot, "last_results", None) or [])]
        return self._decide_counted(state, legal)

    def _decide_counted(self, pub: GameState, legal: List[A.Action]) -> Tuple[A.Action, List[A.Action]]:
        """Counted mode: the bot decides on ``info_samples`` determinizations of the public view
        ``pub`` sampled from the tracker; each action's value is its mean over the samples that
        ranked it, and actions ranked by at least half the samples come first (then the mean,
        then the number of samples that chose it).  A bot without ``last_results`` is a vote."""
        from ..search import ScoredAction
        k_samples = self.info_samples
        values: Dict[A.Action, List[float]] = {}
        expl: Dict[A.Action, str] = {}
        votes: Dict[A.Action, int] = {}
        owner = self.bot
        while "last_results" not in vars(owner) and getattr(owner, "inner", None) is not None:
            owner = owner.inner        # ParamBot forwards last_results to the wrapped bot
        tr = self.tracker
        if not all(tr.is_exact(j) for j in range(len(pub.players)) if j != tr.me):
            self.stats["info_uncertain"] += 1
        for _ in range(k_samples):
            det = self.tracker.determinize(pub, self.rng)
            d = self.bot.decide(det, list(legal), self.rng)
            votes[d] = votes.get(d, 0) + 1
            for r in (getattr(self.bot, "last_results", None) or []):
                values.setdefault(r.action, []).append(float(r.value))
                expl.setdefault(r.action, r.explanation)
            self.stats["info_samples"] += 1
        if not values:
            ranked = sorted(votes, key=lambda a: -votes[a])
            return ranked[0], ranked
        need = (k_samples + 1) // 2
        ranked = sorted(values, key=lambda a: (len(values[a]) >= need, sum(values[a]) / len(values[a]),
                                               votes.get(a, 0)), reverse=True)
        if "last_results" in vars(owner):
            owner.last_results = [ScoredAction(a, sum(values[a]) / len(values[a]), expl.get(a, "")) for a in ranked]
        return ranked[0], ranked

    def _answer_trade(self, playable: List[CAction]) -> CAction:
        """3.3 domestic-trade prompts: decline (``REJECT_TRADE`` as a responder, ``CANCEL_TRADE`` as offerer)."""
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
        cb, view = self._views(game)
        legal = [a for a in E.legal_actions(view) if a[0] == A.DISCARD and sum(a[1]) == owed]
        if not legal:
            # Only reachable mid-run (the engine's remaining count is no longer half the hand).
            self.stats["fallback"] += 1
            return fallback_action(playable)
        if len(legal) == 1:
            decision = legal[0]
            self.stats["trivial"] += 1
        else:
            t0 = time.perf_counter()
            decision, _ = self._bot_decide(cb, legal)
            self.stats["search_time"] += time.perf_counter() - t0
            self.stats["searched"] += 1
            if decision not in legal:
                # The search's discard is not one the engine accepts: an illegal-action fallback.
                self.stats["fallback"] += 1
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
# Opponent wrapper (timing + domestic-trade response rule)
# ---------------------------------------------------------------------------
def timing_summary(times: Sequence[float]) -> Dict[str, float]:
    """``{n, mean_ms, p50_ms, p95_ms, max_ms, total_s}`` of per-decision wall times in seconds
    (nearest-rank percentiles; zeros for an empty list)."""
    n = len(times)
    if not n:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "total_s": 0.0}
    xs = sorted(times)
    total = float(sum(xs))

    def pct(q: float) -> float:
        return 1000.0 * xs[max(0, min(n - 1, int(math.ceil(q * n)) - 1))]

    return {"n": n, "mean_ms": 1000.0 * total / n, "p50_ms": pct(0.50), "p95_ms": pct(0.95),
            "max_ms": 1000.0 * xs[-1], "total_s": total}


def _trade_of(st: State) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """``(offered, asked, offerer seat)`` of the pending 3.3 ``State.current_trade``."""
    trade = st.current_trade
    return tuple(int(x) for x in trade[:5]), tuple(int(x) for x in trade[5:10]), int(trade[10])


class BenchOpponent(Player):
    """A catanatron player as a bench opponent: per-decision timing and the trade-response rule.

    Every ``decide`` is delegated to ``inner`` and its wall time recorded in
    ``times`` (and in ``choice_times`` when more than one action was playable).
    ``trade_rule`` (one of :data:`TRADE_MODES`; ``off`` behaves like
    ``native``) decides how a 3.3 ``DECIDE_TRADE`` prompt - an offer from
    catanbot - is answered:

    * ``native``: ``inner.decide`` answers.  catanatron's players do so
      degenerately (``docs/BENCHMARKS.md``): ``ValueFunctionPlayer`` always
      rejects (``ACCEPT_TRADE`` does not move cards, so both answers tie and
      the first listed, ``REJECT_TRADE``, wins), ``AlphaBetaPlayer`` /
      ``SameTurnAlphaBetaPlayer`` / ``MCTSPlayer`` raise ``RuntimeError``
      (their outcome expansion has no case for trade actions) - counted in
      ``trade_stats["errors"]`` and answered ``REJECT_TRADE`` here - and the
      random / ``VictoryPointPlayer`` players flip a coin.
    * ``value``: OUR model of a sensible opponent: accept iff the player's
      value function is strictly higher after the trade than before (both
      hands updated as ``CONFIRM_TRADE`` would).  The value function is the
      player's own when it has one (catanatron's value / alpha-beta players:
      their ``value_fn`` and weights; our stand-ins: their ``_value``),
      otherwise catanatron's ``base_fn`` with its default weights.
    * ``fair``: ``value``, and never give more cards than received, and never
      trade with a proposer who has ``vps_to_win - 2`` or more public VP.

    The seat asked about its own offer (a catanatron quirk) and a seat that
    cannot pay (only ``REJECT_TRADE`` playable) reject without consulting
    anything.  ``trade_stats`` counts ``asked`` (answerable offers),
    ``accepted``, ``rejected``, ``cannot_pay``, ``errors`` and ``self_offer``.
    Observer hooks (``before`` / ``step`` / ``after`` on 3.3,
    ``reset_state`` on 3.2.1) and unknown attributes are forwarded to ``inner``.
    """

    def __init__(self, inner: Player, trade_rule: str = "native", vps_to_win: int = 10):
        Player.__init__(self, inner.color)
        if trade_rule not in TRADE_MODES:
            raise ValueError(f"trade_rule must be one of {TRADE_MODES}, got {trade_rule!r}")
        self.inner = inner
        self.trade_rule = trade_rule
        self.vps_to_win = vps_to_win
        self.times: List[float] = []
        self.choice_times: List[float] = []
        self.trade_stats: Dict[str, int] = {"asked": 0, "accepted": 0, "rejected": 0, "cannot_pay": 0,
                                            "errors": 0, "self_offer": 0}
        self._value_fn: Optional[Callable] = None

    def __getattr__(self, name):
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __repr__(self) -> str:
        return f"BenchOpponent({self.inner!r}, trade_rule={self.trade_rule})"

    # -- forwarded hooks ----------------------------------------------------
    def before(self, game) -> None:
        hook = getattr(self.inner, "before", None)
        if hook is not None:
            hook(game)

    def step(self, game_before_action, action) -> None:
        hook = getattr(self.inner, "step", None)
        if hook is not None:
            hook(game_before_action, action)

    def after(self, game) -> None:
        hook = getattr(self.inner, "after", None)
        if hook is not None:
            hook(game)

    def reset_state(self) -> None:
        hook = getattr(self.inner, "reset_state", None)
        if hook is not None:
            hook()

    # -- decisions ----------------------------------------------------------
    def decide(self, game: Game, playable_actions):
        t0 = time.perf_counter()
        try:
            if DECIDE_TRADE is not None and game.state.current_prompt == DECIDE_TRADE:
                return self._answer_offer(game, list(playable_actions))
            return self.inner.decide(game, playable_actions)
        finally:
            dt = time.perf_counter() - t0
            self.times.append(dt)
            if len(playable_actions) > 1:
                self.choice_times.append(dt)

    def _answer_offer(self, game: Game, playable: List[CAction]) -> CAction:
        reject = next((a for a in playable if a.action_type == AT_REJECT), None)
        accept = next((a for a in playable if a.action_type == AT_ACCEPT), None)
        if reject is None:   # not a normal DECIDE_TRADE list: let the player handle it
            return self.inner.decide(game, playable)
        st = game.state
        _, _, proposer = _trade_of(st)
        if st.colors[proposer] == self.color:
            self.trade_stats["self_offer"] += 1
            return reject
        self.trade_stats["asked"] += 1
        if accept is None:
            self.trade_stats["cannot_pay"] += 1
            return reject
        if self.trade_rule in ("native", "off"):
            try:
                answer = self.inner.decide(game, playable)
            except Exception:
                self.trade_stats["errors"] += 1
                answer = reject
            if answer not in (accept, reject):
                self.trade_stats["errors"] += 1
                answer = reject
        else:
            answer = accept if self.rule_accepts(game) else reject
        self.trade_stats["accepted" if answer is accept else "rejected"] += 1
        return answer

    def value_fn(self) -> Callable:
        """``fn(game, color) -> float`` used by the ``value`` / ``fair`` rules (see the class doc)."""
        if self._value_fn is None:
            inner = self.inner
            if hasattr(inner, "value_fn_builder_name"):          # catanatron 3.3 value / alpha-beta players
                from catanatron.players.value import get_value_fn
                params = getattr(inner, "params", None)
                self._value_fn = get_value_fn(inner.value_fn_builder_name, getattr(params, "weights", None))
            elif callable(getattr(inner, "_value", None)):       # our stand-ins (catanbot.bench.catanatron_players)
                self._value_fn = lambda game, color: inner._value(game)
            else:
                from catanatron.players.value import DEFAULT_WEIGHTS, base_fn
                self._value_fn = base_fn(DEFAULT_WEIGHTS)
        return self._value_fn

    def rule_accepts(self, game: Game) -> bool:
        """The ``value`` / ``fair`` rule on the pending offer (this seat must be able to pay)."""
        from catanatron import state_functions as SF
        st = game.state
        receive, pay, proposer = _trade_of(st)
        other = st.colors[proposer]
        if self.trade_rule == "fair":
            if sum(receive) < sum(pay):
                return False
            if int(st.player_state[f"P{proposer}_VICTORY_POINTS"]) >= self.vps_to_win - 2:
                return False
        fn = self.value_fn()
        before = fn(game, self.color)
        after_game = game.copy()
        s2 = after_game.state
        SF.player_freqdeck_add(s2, self.color, list(receive))
        SF.player_freqdeck_subtract(s2, self.color, list(pay))
        SF.player_freqdeck_subtract(s2, other, list(receive))
        SF.player_freqdeck_add(s2, other, list(pay))
        return fn(after_game, self.color) > before


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------
def make_game(players: Sequence[Player], seed: int, vps_to_win: int = 10, discard_limit: int = 7,
              catan_map: Optional[CatanMap] = None) -> Game:
    """``Game`` with ``players`` seated exactly in the given order.

    catanatron shuffles the seating with the game seed; since no action has been
    taken yet every per-seat field is still identical, so the seating can be
    reordered deterministically (``seed`` must be non-zero: catanatron treats 0
    as "pick a random seed").  ``discard_limit`` is catanatron's (it only
    governs who discards *first* on a 7, see :func:`state_to_catanbot`).
    ``catan_map`` replaces the random board (used to rebuild a logged game,
    :func:`rebuild_game`); ``None`` keeps catanatron's seeded random board.
    """
    if not seed:
        raise ValueError("seed must be non-zero (catanatron treats 0 as random)")
    if catan_map is None:
        game = Game(list(players), seed=seed, vps_to_win=vps_to_win, discard_limit=discard_limit)
    else:
        game = Game(list(players), seed=seed, vps_to_win=vps_to_win, discard_limit=discard_limit,
                    catan_map=catan_map)
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
              discard_limit: int = 7, record_log: bool = False) -> Dict[str, object]:
    """Play one seated game to the end; returns a summary dict (winner may be ``None`` at the turn cap).

    ``record_log=True`` adds ``"log"``: the replayable record of :func:`game_log`
    (board, initial development deck, every action with its chance outcome, final
    state).  If the game raises, the partial record is attached to the exception
    as ``game_log`` before it propagates.
    """
    for p in players:
        if hasattr(p, 'reset_state'):
            p.reset_state()
    game = make_game(players, seed, vps_to_win, discard_limit)
    header = game_log_header(game) if record_log else None
    t0 = time.perf_counter()
    try:
        winner = game.play()
    except Exception as ex:
        if header is not None:
            try:
                ex.game_log = game_log(game, header, crashed=True)
            except Exception:   # noqa: BLE001 - never mask the original error
                pass
        raise
    st = game.state
    vps = [int(st.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"]) for i in range(len(st.colors))]
    res = {
        "seed": seed,
        "winner": winner.value if winner is not None else None,
        "winner_seat": st.color_to_index[winner] if winner is not None else -1,
        "colors": [c.value for c in st.colors],
        "vps": vps,
        "turns": int(st.num_turns),
        "actions": len(action_log(st)),
        "duration": time.perf_counter() - t0,
    }
    if header is not None:
        res["log"] = game_log(game, header)
    return res


# ---------------------------------------------------------------------------
# Replayable game logs (``scripts/bench_catanatron.py --log-actions``,
# ``scripts/replay_catanatron.py``)
# ---------------------------------------------------------------------------
#: Version tag of the per-game log records written by :func:`game_log`.
LOG_FORMAT = "catanbot-actionlog/1"


class ReplayMismatch(RuntimeError):
    """A logged action is not playable in the rebuilt state, or re-applies to a different record."""


def _plain(v):
    """JSON form of an action value / chance result: colours and enums by value, tuples as lists."""
    if isinstance(v, Color):
        return v.value
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(_plain(k)): _plain(x) for k, x in v.items()}
    if hasattr(v, "value") and hasattr(v, "name") and type(v).__module__.startswith("catanatron"):
        return v.value   # any other catanatron enum
    return v


def _tuplify(v):
    return tuple(_tuplify(x) for x in v) if isinstance(v, list) else v


def _legacy_result(action: CAction):
    """3.2.1 keeps the chance outcome in the logged action's value: extract it in 3.3's
    ``ActionRecord.result`` shape (dice, drawn card, stolen card, discarded cards)."""
    t = action.action_type
    v = action.value
    if t in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD) or (DISCARD_LEGACY is not None and t == DISCARD_LEGACY):
        return v
    if t == ActionType.MOVE_ROBBER:
        return v[2] if v is not None and len(v) > 2 else None
    return None


def encode_log_entry(entry) -> list:
    """``[colour, action type, value, result]`` (JSON-ready) of an engine log entry.

    ``result`` is the chance outcome as catanatron records it: 3.3's
    ``ActionRecord.result`` (dice, discarded / stolen card, drawn development
    card), or on 3.2.1 the same outcome read from the fully specified logged
    action (:func:`_legacy_result`; 3.2.1 records every outcome there,
    including its random discards)."""
    a = log_action(entry)
    result = log_result(entry) if ActionRecord is not None else _legacy_result(a)
    return [a.color.value, a.action_type.name, _plain(a.value), _plain(result)]


def decode_log_action(item: Sequence) -> CAction:
    """The catanatron ``Action`` of an :func:`encode_log_entry` item (runs on the engine that wrote it)."""
    color = Color(item[0])
    t = ActionType[item[1]]
    v = item[2]
    if v is not None:
        if t == ActionType.MOVE_ROBBER:
            who = Color(v[1]) if v[1] is not None else None
            v = (tuple(v[0]), who) + tuple(v[2:])
        elif AT_CONFIRM is not None and t == AT_CONFIRM:
            v = tuple(v[:10]) + (Color(v[10]),)
        elif DISCARD_LEGACY is not None and t == DISCARD_LEGACY:
            v = list(v)   # 3.2.1 logs its discards as a list
        else:
            v = _tuplify(v)
    return CAction(color, t, v)


def decode_log_result(action_type: ActionType, result):
    """Chance result of an :func:`encode_log_entry` item in the engine's own types."""
    if result is None:
        return None
    if action_type == ActionType.ROLL:
        return tuple(result)
    if DISCARD_LEGACY is not None and action_type == DISCARD_LEGACY:
        return list(result)
    return result


def encode_board(board) -> Dict[str, object]:
    """The random part of a catanatron board: every tile of the (BASE) topology in catanatron's
    order with its type, id, resource and number (land) or resource and direction (port),
    the robber's start and, for reading, the port nodes by resource (``"3:1"`` = generic)."""
    from catanatron.models.map import LandTile, Port
    cmap = board.map
    tiles = []
    for coord, tile in cmap.tiles.items():
        c = [int(x) for x in coord]
        if isinstance(tile, LandTile):
            tiles.append({"c": c, "t": "land", "id": int(tile.id), "resource": tile.resource,
                          "number": None if tile.number is None else int(tile.number)})
        elif isinstance(tile, Port):
            tiles.append({"c": c, "t": "port", "id": int(tile.id), "resource": tile.resource,
                          "direction": tile.direction.name})
        else:
            tiles.append({"c": c, "t": "water"})
    ports = {("3:1" if r is None else r): sorted(int(n) for n in nodes) for r, nodes in cmap.port_nodes.items()}
    return {"template": "BASE", "tiles": tiles, "robber": [int(x) for x in board.robber_coordinate],
            "ports": ports}


def decode_board(doc: Dict[str, object]) -> CatanMap:
    """Rebuild the ``CatanMap`` of :func:`encode_board`: the BASE template's topology is walked in
    catanatron's order (so node and edge ids come out identical) with the logged tiles."""
    from catanatron.models.map import BASE_MAP_TEMPLATE, LandTile, Port, Water, get_nodes_and_edges
    if doc.get("template", "BASE") != "BASE":
        raise ValueError(f"unsupported map template {doc.get('template')!r}")
    by_coord = {tuple(t["c"]): t for t in doc["tiles"]}
    if set(by_coord) != set(BASE_MAP_TEMPLATE.topology):
        raise ValueError("logged tiles do not cover the BASE map topology")
    all_tiles: Dict[Coordinate, object] = {}
    node_autoinc = 0
    for coordinate, tile_type in BASE_MAP_TEMPLATE.topology.items():
        nodes, edges, node_autoinc = get_nodes_and_edges(all_tiles, coordinate, node_autoinc)
        stored = by_coord[coordinate]
        if isinstance(tile_type, tuple):
            _, direction = tile_type
            if stored["t"] != "port" or stored["direction"] != direction.name:
                raise ValueError(f"tile at {coordinate}: logged {stored} but the template has a {direction.name} port")
            all_tiles[coordinate] = Port(stored["id"], stored["resource"], direction, nodes, edges)
        elif tile_type == LandTile:
            if stored["t"] != "land":
                raise ValueError(f"tile at {coordinate}: logged {stored} but the template has land")
            all_tiles[coordinate] = LandTile(stored["id"], stored["resource"], stored["number"], nodes, edges)
        else:
            all_tiles[coordinate] = Water(nodes, edges)
    return CatanMap.from_tiles(all_tiles)


def state_summary(st: State) -> Dict[str, object]:
    """Readable, JSON-ready summary of a catanatron state: per seat VP (actual / public), hand,
    development cards held and played, buildings, road / army titles; bank, deck, robber, prompt."""
    ps = st.player_state
    players = []
    for i, color in enumerate(st.colors):
        k = f"P{i}"
        bb = st.buildings_by_color.get(color, {})
        players.append({
            "seat": i,
            "color": color.value,
            "vp": int(ps[f"{k}_ACTUAL_VICTORY_POINTS"]),
            "public_vp": int(ps[f"{k}_VICTORY_POINTS"]),
            "resources": {r: int(ps[f"{k}_{r}_IN_HAND"]) for r in CB_TO_RESOURCE},
            "dev_cards": {d: int(ps[f"{k}_{d}_IN_HAND"]) for d in CB_TO_DEV},
            "dev_played": {d: int(ps.get(f"{k}_PLAYED_{d}", 0)) for d in CB_TO_DEV if f"{k}_PLAYED_{d}" in ps},
            "settlements": sorted(int(n) for n in bb.get(SETTLEMENT, [])),
            "cities": sorted(int(n) for n in bb.get(CITY, [])),
            "roads": sorted([int(min(e)), int(max(e))] for e in bb.get(ROAD, [])),
            "longest_road_length": int(ps[f"{k}_LONGEST_ROAD_LENGTH"]),
            "has_longest_road": bool(ps[f"{k}_HAS_ROAD"]),
            "has_largest_army": bool(ps[f"{k}_HAS_ARMY"]),
        })
    return {
        "turn": int(st.num_turns),
        "current_turn_seat": int(st.current_turn_index),
        "current_player_seat": int(st.current_player_index),
        "prompt": st.current_prompt.name,
        "robber": [int(x) for x in st.board.robber_coordinate],
        "bank": {r: int(x) for r, x in zip(CB_TO_RESOURCE, st.resource_freqdeck)},
        "dev_deck_left": len(st.development_listdeck),
        "actions": len(action_log(st)),
        "players": players,
    }


def _fingerprint_doc(st: State) -> Dict[str, object]:
    board = st.board
    doc = {
        "player_state": {k: _plain(v) for k, v in sorted(st.player_state.items())},
        "buildings": sorted([int(n), c.value, str(bt)] for n, (c, bt) in board.buildings.items()),
        "roads": sorted([int(e[0]), int(e[1]), c.value] for e, c in board.roads.items()),
        # a defaultdict: reading ``[color][CITY]`` creates an empty list, so empty entries are ignored
        "buildings_by_color": {c.value: {str(t): sorted(_plain(x) for x in xs) for t, xs in sorted(d.items()) if xs}
                               for c, d in st.buildings_by_color.items()},
        "robber": [int(x) for x in board.robber_coordinate],
        "road": [board.road_color.value if board.road_color else None, int(board.road_length)],
        "bank": [int(x) for x in st.resource_freqdeck],
        "dev_deck": list(st.development_listdeck),
        "turn": [int(st.num_turns), int(st.current_player_index), int(st.current_turn_index), st.current_prompt.name],
        "flags": [bool(st.is_initial_build_phase), bool(st.is_discarding), bool(st.is_moving_knight),
                  bool(st.is_road_building), int(st.free_roads_available)],
        "colors": [c.value for c in st.colors],
    }
    if hasattr(st, "discard_counts"):
        doc["discard_counts"] = [int(x) for x in st.discard_counts]
    if hasattr(st, "current_trade"):
        doc["trade"] = [_plain(st.current_trade), [bool(x) for x in st.acceptees], bool(st.is_resolving_trade)]
    return doc


def state_fingerprint(st: State) -> str:
    """SHA-256 (first 20 hex digits) of the complete game state: every ``player_state`` field,
    buildings, roads, robber, bank, development deck *in order*, turn / prompt / phase flags."""
    import hashlib
    import json
    text = json.dumps(_fingerprint_doc(st), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def game_log_header(game: Game) -> Dict[str, object]:
    """Everything random that is fixed before the first action: seating (colours in turn
    order), board, robber start and the shuffled development deck (in draw order: the
    engine pops from the end), plus the rules and engine version."""
    st = game.state
    if len(action_log(st)):
        raise ValueError("game_log_header must be taken before the first action")
    return {
        "format": LOG_FORMAT,
        "catanatron": CATANATRON_VERSION,
        "api": "3.3" if API_33 else "3.2",
        "seed": int(game.seed),
        "vps_to_win": int(game.vps_to_win),
        "discard_limit": int(st.discard_limit),
        "turns_limit": int(TURNS_LIMIT),
        "colors": [c.value for c in st.colors],
        "board": encode_board(st.board),
        "dev_deck": list(st.development_listdeck),
    }


def game_log(game: Game, header: Dict[str, object], crashed: bool = False) -> Dict[str, object]:
    """``header`` + every logged action (:func:`encode_log_entry`) + the final state
    (winner, VPs, turns, :func:`state_fingerprint`, :func:`state_summary`)."""
    st = game.state
    winner = game.winning_color()
    out = dict(header)
    out["actions"] = [encode_log_entry(e) for e in action_log(st)]
    out["final"] = {
        "winner": winner.value if winner is not None else None,
        "winner_seat": st.color_to_index[winner] if winner is not None else -1,
        "vps": [int(st.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"]) for i in range(len(st.colors))],
        "turns": int(st.num_turns),
        "num_actions": len(action_log(st)),
        "fingerprint": state_fingerprint(st),
        "state": state_summary(st),
    }
    if crashed:
        out["crashed"] = True
    return out


class _LogSeat(Player):
    """Placeholder seat of a rebuilt game (actions come from the log, never from ``decide``)."""

    def decide(self, game, playable_actions):
        raise RuntimeError("a replayed game takes its actions from the log")


def rebuild_game(doc: Dict[str, object]) -> Game:
    """A fresh catanatron game in the logged initial state: same board, robber, seating
    and development deck order, no action taken.  Raises ``ValueError`` when the log was
    written by the other catanatron API generation (3.2.1 vs 3.3)."""
    if doc.get("format") != LOG_FORMAT:
        raise ValueError(f"not a {LOG_FORMAT} record (format {doc.get('format')!r})")
    api = "3.3" if API_33 else "3.2"
    if doc.get("api") != api:
        raise ValueError(f"log written with catanatron {doc.get('catanatron')} ({doc.get('api')} API); this "
                         f"interpreter runs catanatron {CATANATRON_VERSION} ({api} API) - replay it with the "
                         f"interpreter that played it")
    cmap = decode_board(doc["board"])
    seats = [_LogSeat(Color(c)) for c in doc["colors"]]
    game = make_game(seats, seed=int(doc.get("seed") or 1), vps_to_win=int(doc["vps_to_win"]),
                     discard_limit=int(doc["discard_limit"]), catan_map=cmap)
    st = game.state
    st.development_listdeck = list(doc["dev_deck"])
    robber = tuple(doc["board"]["robber"])
    if tuple(st.board.robber_coordinate) != robber:
        raise ReplayMismatch(f"robber starts on {tuple(st.board.robber_coordinate)}, logged {robber}")
    return game


def replay_log_action(game: Game, item: Sequence, check: bool = True):
    """Apply one logged item to ``game`` with its logged chance outcome; returns the engine's
    record.  With ``check`` the item must be playable for the colour to move and must
    re-apply to exactly the logged item (else :class:`ReplayMismatch`)."""
    st = game.state
    action = decode_log_action(item)
    result = decode_log_result(action.action_type, item[3])
    if check:
        if action.color != st.current_color():
            raise ReplayMismatch(f"{item[0]} acts but {st.current_color().value} is to move")
        if AT_OFFER is not None and action.action_type == AT_OFFER:
            from catanatron.game import is_valid_action
            ok = is_valid_action(playable_actions_of(game), st, action)
        else:
            ok = any(a.color == action.color and playable_key(a) == playable_key(action)
                     for a in playable_actions_of(game))
        if not ok:
            raise ReplayMismatch(f"{list(item)} is not playable in the rebuilt state")
    if ActionRecord is not None:
        rec = game.execute(action, validate_action=False, action_record=ActionRecord(action, result))
    else:
        to_apply = action
        deck = st.development_listdeck
        if action.action_type == ActionType.BUY_DEVELOPMENT_CARD and deck and deck[-1] == action.value:
            to_apply = CAction(action.color, action.action_type, None)   # draw like the original: deck order kept
        rec = game.execute(to_apply, validate_action=False)
    if check:
        got = encode_log_entry(rec)
        if got != list(item):
            raise ReplayMismatch(f"logged {list(item)} re-applied as {got}")
    return rec
