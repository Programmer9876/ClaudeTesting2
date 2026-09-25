"""Static board topology for the standard 19-hex Catan board.

Coordinates
-----------
Hexes use axial coordinates ``(q, r)`` with pointy-top orientation (rows of
3-4-5-4-3 hexes stacked vertically, exactly how Colonist.io draws the board).
Hex index order is row by row, top to bottom, left to right:

    row r=-2:  0  1  2
    row r=-1:  3  4  5  6
    row r= 0:  7  8  9 10 11
    row r= 1: 12 13 14 15
    row r= 2: 16 17 18

Pixel-space geometry (unit hex size, screen coordinates with y pointing down):

    center_x = sqrt(3) * (q + r / 2)
    center_y = 1.5 * r
    corner k (k = 0..5, starting at the top, clockwise on screen):
        angle = 270 + 60 * k degrees
        (center_x + cos(angle), center_y + sin(angle))

Vertices are numbered 0..53 and edges 0..71 (sorted by (y, x) of their
position) so every module, the screenshot parser and the JSON state format
agree on ids.  All adjacency tables are precomputed at import time.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

SQRT3 = math.sqrt(3.0)

# ---------------------------------------------------------------------------
# Resources, dev cards, ports
# ---------------------------------------------------------------------------
WOOD, BRICK, SHEEP, WHEAT, ORE, DESERT = range(6)
RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore", "desert"]
RESOURCE_INDEX = {name: i for i, name in enumerate(RESOURCE_NAMES)}
# Common aliases accepted by parsers / the JSON loader.
RESOURCE_ALIASES = {
    "lumber": WOOD, "forest": WOOD, "wood": WOOD,
    "brick": BRICK, "hills": BRICK, "hill": BRICK, "clay": BRICK,
    "sheep": SHEEP, "wool": SHEEP, "pasture": SHEEP,
    "wheat": WHEAT, "grain": WHEAT, "fields": WHEAT, "field": WHEAT,
    "ore": ORE, "mountains": ORE, "mountain": ORE, "stone": ORE,
    "desert": DESERT,
}
NUM_RESOURCES = 5

DEV_KNIGHT, DEV_VP, DEV_ROAD_BUILDING, DEV_YEAR_OF_PLENTY, DEV_MONOPOLY = range(5)
DEV_NAMES = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]
DEV_DECK_COUNTS = [14, 5, 2, 2, 2]  # standard 25-card deck
NUM_DEV = 5

# Port types: 0..4 = 2:1 port for that resource, 5 = generic 3:1 port.
PORT_GENERIC = 5
PORT_NAMES = ["wood", "brick", "sheep", "wheat", "ore", "3:1"]

# Building costs indexed by resource (wood, brick, sheep, wheat, ore).
COST_ROAD = (1, 1, 0, 0, 0)
COST_SETTLEMENT = (1, 1, 1, 1, 0)
COST_CITY = (0, 0, 0, 2, 3)
COST_DEV = (0, 0, 1, 1, 1)

BANK_PER_RESOURCE = 19
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4
VP_TO_WIN = 10

# Number token -> pips (dots) = number of dice combinations.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
ROLL_PROB = {n: p / 36.0 for n, p in PIPS.items()}
ROLL_PROB[7] = 6 / 36.0
STANDARD_NUMBERS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
STANDARD_RESOURCE_COUNTS = {WOOD: 4, BRICK: 3, SHEEP: 4, WHEAT: 4, ORE: 3, DESERT: 1}

# ---------------------------------------------------------------------------
# Hex layout
# ---------------------------------------------------------------------------
HEX_COORDS: List[Tuple[int, int]] = [
    (0, -2), (1, -2), (2, -2),
    (-1, -1), (0, -1), (1, -1), (2, -1),
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    (-2, 1), (-1, 1), (0, 1), (1, 1),
    (-2, 2), (-1, 2), (0, 2),
]
NUM_HEXES = len(HEX_COORDS)
HEX_INDEX = {c: i for i, c in enumerate(HEX_COORDS)}
HEX_ROWS = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18]]


def hex_center(q: int, r: int, size: float = 1.0) -> Tuple[float, float]:
    """Pixel center of hex (q, r) with the given hex size (center-to-corner)."""
    return (size * SQRT3 * (q + r / 2.0), size * 1.5 * r)


def hex_corner(cx: float, cy: float, k: int, size: float = 1.0) -> Tuple[float, float]:
    """Corner ``k`` (0 = top, clockwise on screen) of a hex centred at (cx, cy)."""
    ang = math.radians(270 + 60 * k)
    return (cx + size * math.cos(ang), cy + size * math.sin(ang))


HEX_CENTERS: List[Tuple[float, float]] = [hex_center(q, r) for q, r in HEX_COORDS]


def _vertex_key(x: float, y: float) -> Tuple[int, int]:
    return (int(round(2 * x / SQRT3)), int(round(2 * y)))


_vkeys: Dict[Tuple[int, int], Tuple[float, float]] = {}
_hex_corner_keys: List[List[Tuple[int, int]]] = []
for _q, _r in HEX_COORDS:
    _cx, _cy = hex_center(_q, _r)
    _ks = []
    for _k in range(6):
        _x, _y = hex_corner(_cx, _cy, _k)
        _key = _vertex_key(_x, _y)
        _vkeys.setdefault(_key, (_x, _y))
        _ks.append(_key)
    _hex_corner_keys.append(_ks)

# Sort vertices by (y, x) so ids are stable and readable (top row first).
_sorted_keys = sorted(_vkeys, key=lambda k: (k[1], k[0]))
VERTEX_ID: Dict[Tuple[int, int], int] = {k: i for i, k in enumerate(_sorted_keys)}
NUM_VERTICES = len(_sorted_keys)
VERTEX_POS: List[Tuple[float, float]] = [_vkeys[k] for k in _sorted_keys]

# hex -> 6 vertex ids in corner order (0 = top, clockwise)
HEX_VERTICES: List[List[int]] = [[VERTEX_ID[k] for k in ks] for ks in _hex_corner_keys]

# edges
_edge_set: Dict[Tuple[int, int], None] = {}
for _vs in HEX_VERTICES:
    for _k in range(6):
        _a, _b = _vs[_k], _vs[(_k + 1) % 6]
        _edge_set[(min(_a, _b), max(_a, _b))] = None


def _edge_mid(e: Tuple[int, int]) -> Tuple[float, float]:
    (x1, y1), (x2, y2) = VERTEX_POS[e[0]], VERTEX_POS[e[1]]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


_sorted_edges = sorted(_edge_set, key=lambda e: (round(_edge_mid(e)[1], 6), round(_edge_mid(e)[0], 6)))
EDGE_ID: Dict[Tuple[int, int], int] = {e: i for i, e in enumerate(_sorted_edges)}
NUM_EDGES = len(_sorted_edges)
EDGE_VERTICES: List[Tuple[int, int]] = list(_sorted_edges)
EDGE_POS: List[Tuple[float, float]] = [_edge_mid(e) for e in _sorted_edges]


def edge_between(a: int, b: int) -> int:
    """Edge id joining vertices a and b (raises KeyError if not adjacent)."""
    return EDGE_ID[(min(a, b), max(a, b))]


# hex -> 6 edge ids (edge k joins corner k and corner k+1)
HEX_EDGES: List[List[int]] = [
    [edge_between(vs[k], vs[(k + 1) % 6]) for k in range(6)] for vs in HEX_VERTICES
]

# vertex -> hexes, vertex -> neighbour vertices, vertex -> edges
VERTEX_HEXES: List[List[int]] = [[] for _ in range(NUM_VERTICES)]
for _h, _vs in enumerate(HEX_VERTICES):
    for _v in _vs:
        VERTEX_HEXES[_v].append(_h)

VERTEX_NEIGHBORS: List[List[int]] = [[] for _ in range(NUM_VERTICES)]
VERTEX_EDGES: List[List[int]] = [[] for _ in range(NUM_VERTICES)]
for _e, (_a, _b) in enumerate(EDGE_VERTICES):
    VERTEX_NEIGHBORS[_a].append(_b)
    VERTEX_NEIGHBORS[_b].append(_a)
    VERTEX_EDGES[_a].append(_e)
    VERTEX_EDGES[_b].append(_e)

# edge -> hexes (1 for coastal edges, 2 for interior)
EDGE_HEXES: List[List[int]] = [[] for _ in range(NUM_EDGES)]
for _h, _es in enumerate(HEX_EDGES):
    for _e in _es:
        EDGE_HEXES[_e].append(_h)

# edge -> the (up to 4) edges sharing one endpoint with it
EDGE_NEIGHBORS: List[List[int]] = [
    sorted({x for v in EDGE_VERTICES[e] for x in VERTEX_EDGES[v] if x != e}) for e in range(NUM_EDGES)
]

# hex -> neighbouring hexes
HEX_NEIGHBORS: List[List[int]] = [[] for _ in range(NUM_HEXES)]
for _e, _hs in enumerate(EDGE_HEXES):
    if len(_hs) == 2:
        HEX_NEIGHBORS[_hs[0]].append(_hs[1])
        HEX_NEIGHBORS[_hs[1]].append(_hs[0])

# Coastal edges ordered clockwise around the board (start: top-left region).
def _angle(pos: Tuple[float, float]) -> float:
    # screen coords (y down): atan2(y, x) increases clockwise.  Start from
    # the top-left (angle -150 deg) so the sequence begins near hex 0's upper-left.
    a = math.degrees(math.atan2(pos[1], pos[0]))
    a = (a + 150.0) % 360.0
    return a


COASTAL_EDGES: List[int] = sorted(
    [e for e in range(NUM_EDGES) if len(EDGE_HEXES[e]) == 1], key=lambda e: _angle(EDGE_POS[e])
)
COASTAL_VERTICES: List[int] = sorted({v for e in COASTAL_EDGES for v in EDGE_VERTICES[e]})

# Default (standard beginner) port layout.  Ports are attached to a coastal
# edge; both endpoints of that edge give access to the port.  Colonist.io
# randomises ports, so parsers overwrite this via ``state.ports``.
_STANDARD_PORT_SLOTS = [1, 4, 8, 11, 15, 18, 22, 25, 28]  # indices into COASTAL_EDGES
_STANDARD_PORT_TYPES = [PORT_GENERIC, WHEAT, ORE, PORT_GENERIC, SHEEP, PORT_GENERIC, PORT_GENERIC, BRICK, WOOD]
STANDARD_PORT_EDGES: List[Tuple[int, int]] = [
    (COASTAL_EDGES[s], t) for s, t in zip(_STANDARD_PORT_SLOTS, _STANDARD_PORT_TYPES)
]


def ports_from_edges(port_edges) -> Dict[int, int]:
    """Expand ``[(edge_id, port_type), ...]`` into ``{vertex_id: port_type}``."""
    out: Dict[int, int] = {}
    for e, t in port_edges:
        for v in EDGE_VERTICES[e]:
            out[v] = t
    return out


STANDARD_PORTS: Dict[int, int] = ports_from_edges(STANDARD_PORT_EDGES)


def random_port_edges(rng) -> List[Tuple[int, int]]:
    """Standard harbour positions with the 9 port types shuffled (as Catanatron's base map does)."""
    types = list(_STANDARD_PORT_TYPES)
    rng.shuffle(types)
    return [(COASTAL_EDGES[s], t) for s, t in zip(_STANDARD_PORT_SLOTS, types)]


def random_ports(rng) -> Dict[int, int]:
    """``{vertex: port_type}`` for :func:`random_port_edges`."""
    return ports_from_edges(random_port_edges(rng))

# Official beginner layout (rows top to bottom): (resource, number)
STANDARD_HEXES: List[Tuple[int, int]] = [
    (ORE, 10), (SHEEP, 2), (WOOD, 9),
    (WHEAT, 12), (BRICK, 6), (SHEEP, 4), (BRICK, 10),
    (WHEAT, 9), (WOOD, 11), (DESERT, 0), (WOOD, 3), (ORE, 8),
    (WOOD, 8), (ORE, 3), (WHEAT, 4), (SHEEP, 5),
    (BRICK, 5), (WHEAT, 6), (SHEEP, 11),
]


def random_hexes(rng, no_adjacent_red: bool = True) -> List[Tuple[int, int]]:
    """Random standard board: shuffled resources + numbers, desert gets 0.

    With ``no_adjacent_red`` the 6s and 8s are never on neighbouring hexes
    (Colonist.io's default).  ``rng`` is a ``random.Random``.
    """
    resources = [r for r, n in STANDARD_RESOURCE_COUNTS.items() for _ in range(n)]
    for _ in range(1000):
        rng.shuffle(resources)
        numbers = list(STANDARD_NUMBERS)
        rng.shuffle(numbers)
        hexes: List[Tuple[int, int]] = []
        it = iter(numbers)
        for res in resources:
            hexes.append((res, 0) if res == DESERT else (res, next(it)))
        if not no_adjacent_red:
            return hexes
        ok = True
        for h, (res, num) in enumerate(hexes):
            if num in (6, 8):
                for nb in HEX_NEIGHBORS[h]:
                    if hexes[nb][1] in (6, 8):
                        ok = False
                        break
            if not ok:
                break
        if ok:
            return hexes
    return hexes  # pragma: no cover - practically unreachable


def vertex_pip_total(v: int, hexes) -> int:
    """Sum of pips of the hexes touching vertex v (desert = 0)."""
    return sum(PIPS.get(hexes[h][1], 0) for h in VERTEX_HEXES[v] if hexes[h][0] != DESERT)


# Precomputed sanity numbers for tests.
assert NUM_HEXES == 19
assert NUM_VERTICES == 54, NUM_VERTICES
assert NUM_EDGES == 72, NUM_EDGES
assert len(COASTAL_EDGES) == 30, len(COASTAL_EDGES)
