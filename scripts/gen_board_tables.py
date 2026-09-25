#!/usr/bin/env python3
"""Generate the C++ headers ``cpp/board_tables.hpp`` and ``cpp/feature_layout.hpp``.

The board topology (``catanbot/board.py``) and the feature layout
(``catanbot/features.py``) are the single source of truth.  This script
re-derives every table the C++ extension needs from those modules so the
headers can never drift from the Python reference:

* ``cpp/board_tables.hpp``: hex/vertex/edge adjacency, pips, costs, piece
  limits, standard layout, ports, coastal edges.
* ``cpp/feature_layout.hpp``: phase order (as used by the phase one-hots),
  ``PLAYER_BLOCK`` / ``GLOBAL_BLOCK`` / ``NUM_FEATURES``, the offset of every
  feature inside its block and the full ``FEATURE_NAMES`` list.

Usage::

    PYTHONPATH=. python3 scripts/gen_board_tables.py            # rewrite cpp/*.hpp
    PYTHONPATH=. python3 scripts/gen_board_tables.py --check    # exit 1 if stale

``tests/test_accel_features.py`` runs the ``--check`` mode.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from catanbot import board as B  # noqa: E402
from catanbot import features as F  # noqa: E402

BOARD_HEADER = os.path.join(ROOT, "cpp", "board_tables.hpp")
LAYOUT_HEADER = os.path.join(ROOT, "cpp", "feature_layout.hpp")


# ---------------------------------------------------------------------------
# emit helpers
# ---------------------------------------------------------------------------
def _row(vals: Iterable[int]) -> str:
    return "{" + ", ".join(str(int(v)) for v in vals) + "}"


def _pad(lst: Sequence[int], width: int, fill: int = -1) -> List[int]:
    out = list(lst) + [fill] * (width - len(lst))
    assert len(out) == width, (lst, width)
    return out


def _array1(name: str, vals: Sequence[int], ctype: str = "int") -> str:
    return f"constexpr {ctype} {name}[{len(vals)}] = {_row(vals)};\n"


def _array2(name: str, rows: Sequence[Sequence[int]], width: int, ctype: str = "int") -> str:
    body = ",\n".join("    " + _row(r) for r in rows)
    return f"constexpr {ctype} {name}[{len(rows)}][{width}] = {{\n{body}\n}};\n"


def _padded2(name: str, rows: Sequence[Sequence[int]], width: int) -> str:
    """Ragged table -> fixed width (padded with -1) plus a per-row count array."""
    counts = [len(r) for r in rows]
    assert max(counts) <= width, (name, max(counts), width)
    return _array1(name + "_N", counts) + _array2(name, [_pad(r, width) for r in rows], width)


# ---------------------------------------------------------------------------
# board_tables.hpp
# ---------------------------------------------------------------------------
def gen_board_tables() -> str:
    o: List[str] = []
    o.append("// GENERATED FILE - do not edit.\n")
    o.append("// Produced by scripts/gen_board_tables.py from catanbot/board.py.\n")
    o.append("// Re-run `PYTHONPATH=. python3 scripts/gen_board_tables.py` after changing board.py.\n")
    o.append("#pragma once\n\nnamespace catanbot {\n\n")
    o.append("// ---- sizes -------------------------------------------------------------\n")
    o.append(f"constexpr int NUM_HEXES = {B.NUM_HEXES};\n")
    o.append(f"constexpr int NUM_VERTICES = {B.NUM_VERTICES};\n")
    o.append(f"constexpr int NUM_EDGES = {B.NUM_EDGES};\n")
    o.append(f"constexpr int NUM_RESOURCES = {B.NUM_RESOURCES};\n")
    o.append(f"constexpr int NUM_DEV = {B.NUM_DEV};\n")
    o.append(f"constexpr int NUM_COASTAL_EDGES = {len(B.COASTAL_EDGES)};\n\n")
    o.append("// ---- resources / dev cards / ports ---------------------------------------\n")
    for i, n in enumerate(B.RESOURCE_NAMES):
        o.append(f"constexpr int {n.upper()} = {i};\n")
    o.append(f"constexpr int PORT_GENERIC = {B.PORT_GENERIC};\n")
    for i, n in enumerate(B.DEV_NAMES):
        o.append(f"constexpr int DEV_{n.upper()} = {i};\n")
    o.append("constexpr const char* RESOURCE_NAMES[%d] = {%s};\n" % (
        len(B.RESOURCE_NAMES), ", ".join(f'"{n}"' for n in B.RESOURCE_NAMES)))
    o.append("constexpr const char* DEV_NAMES[%d] = {%s};\n" % (
        len(B.DEV_NAMES), ", ".join(f'"{n}"' for n in B.DEV_NAMES)))
    o.append("constexpr const char* PORT_NAMES[%d] = {%s};\n\n" % (
        len(B.PORT_NAMES), ", ".join(f'"{n}"' for n in B.PORT_NAMES)))
    o.append("// ---- game constants --------------------------------------------------------\n")
    o.append(f"constexpr int BANK_PER_RESOURCE = {B.BANK_PER_RESOURCE};\n")
    o.append(f"constexpr int MAX_ROADS = {B.MAX_ROADS};\n")
    o.append(f"constexpr int MAX_SETTLEMENTS = {B.MAX_SETTLEMENTS};\n")
    o.append(f"constexpr int MAX_CITIES = {B.MAX_CITIES};\n")
    o.append(f"constexpr int VP_TO_WIN = {B.VP_TO_WIN};\n")
    o.append(f"constexpr int DEV_DECK_TOTAL = {sum(B.DEV_DECK_COUNTS)};\n")
    o.append(_array1("DEV_DECK_COUNTS", B.DEV_DECK_COUNTS))
    o.append(_array1("COST_ROAD", B.COST_ROAD))
    o.append(_array1("COST_SETTLEMENT", B.COST_SETTLEMENT))
    o.append(_array1("COST_CITY", B.COST_CITY))
    o.append(_array1("COST_DEV", B.COST_DEV))
    # PIPS indexed by number token 0..12 (0 where the number has no pips, incl. 7).
    pips = [B.PIPS.get(n, 0) for n in range(13)]
    o.append("// pips of number token n (index 0..12; 0 for 0, 1, 7 and the desert)\n")
    o.append(_array1("PIPS", pips))
    o.append(_array1("STANDARD_NUMBERS", B.STANDARD_NUMBERS))
    o.append("// resource -> number of hexes of that type on a standard board\n")
    o.append(_array1("STANDARD_RESOURCE_COUNTS", [B.STANDARD_RESOURCE_COUNTS[r] for r in range(6)]))
    o.append("\n// ---- topology ----------------------------------------------------------------\n")
    o.append("// axial (q, r) of each hex\n")
    o.append(_array2("HEX_COORDS", B.HEX_COORDS, 2))
    o.append("// hex -> 6 vertex ids in corner order (0 = top, clockwise)\n")
    o.append(_array2("HEX_VERTICES", B.HEX_VERTICES, 6))
    o.append("// hex -> 6 edge ids (edge k joins corner k and corner k+1)\n")
    o.append(_array2("HEX_EDGES", B.HEX_EDGES, 6))
    o.append("// vertex -> hexes (1..3, padded with -1; count in *_N)\n")
    o.append(_padded2("VERTEX_HEXES", B.VERTEX_HEXES, 3))
    o.append("// vertex -> neighbouring vertices (2..3, padded with -1)\n")
    o.append(_padded2("VERTEX_NEIGHBORS", B.VERTEX_NEIGHBORS, 3))
    o.append("// vertex -> incident edges (2..3, padded with -1); VERTEX_EDGES[v][k] joins v and VERTEX_NEIGHBORS[v][k]\n")
    o.append(_padded2("VERTEX_EDGES", B.VERTEX_EDGES, 3))
    o.append("// edge -> its two endpoints (a < b)\n")
    o.append(_array2("EDGE_VERTICES", B.EDGE_VERTICES, 2))
    o.append("// edge -> hexes (1 for coastal edges, 2 interior; padded with -1)\n")
    o.append(_padded2("EDGE_HEXES", B.EDGE_HEXES, 2))
    o.append("// edge -> edges sharing an endpoint (2..4, padded with -1)\n")
    o.append(_padded2("EDGE_NEIGHBORS", B.EDGE_NEIGHBORS, 4))
    o.append("// hex -> neighbouring hexes (2..6, padded with -1)\n")
    o.append(_padded2("HEX_NEIGHBORS", B.HEX_NEIGHBORS, 6))
    o.append("// coastal edges clockwise around the board\n")
    o.append(_array1("COASTAL_EDGES", B.COASTAL_EDGES))
    o.append("\n// ---- standard (beginner) layout -----------------------------------------------\n")
    o.append("// (resource, number) per hex\n")
    o.append(_array2("STANDARD_HEXES", B.STANDARD_HEXES, 2))
    o.append("// vertex -> port type (-1 = no port)\n")
    o.append(_array1("STANDARD_PORTS", [B.STANDARD_PORTS.get(v, -1) for v in range(B.NUM_VERTICES)]))
    o.append("// (coastal edge, port type) of the standard port layout\n")
    o.append(_array2("STANDARD_PORT_EDGES", B.STANDARD_PORT_EDGES, 2))
    o.append("\n}  // namespace catanbot\n")
    return "".join(o)


# ---------------------------------------------------------------------------
# feature_layout.hpp
# ---------------------------------------------------------------------------
def _cident(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if out[0].isdigit():
        out = "_" + out
    return out


def gen_feature_layout() -> str:
    o: List[str] = []
    o.append("// GENERATED FILE - do not edit.\n")
    o.append("// Produced by scripts/gen_board_tables.py from catanbot/features.py (and state.py phases).\n")
    o.append("// Re-run `PYTHONPATH=. python3 scripts/gen_board_tables.py` after changing the feature layout.\n")
    o.append("#pragma once\n\nnamespace catanbot {\n\n")
    o.append("// ---- phases, in the order of the phase one-hot block ----------------------------\n")
    phases = list(F._PHASES)
    o.append(f"constexpr int NUM_PHASES = {len(phases)};\n")
    o.append("enum Phase : int {\n")
    for i, p in enumerate(phases):
        o.append(f"    PHASE_{p.upper()} = {i},\n")
    o.append("    PHASE_UNKNOWN = -1\n};\n")
    o.append("constexpr const char* PHASE_NAMES[NUM_PHASES] = {%s};\n\n" % ", ".join(f'"{p}"' for p in phases))
    o.append("// ---- block sizes -----------------------------------------------------------------\n")
    o.append(f"constexpr int MAX_PLAYERS = {F.MAX_PLAYERS};\n")
    o.append(f"constexpr int PLAYER_BLOCK = {F.PLAYER_BLOCK};\n")
    o.append(f"constexpr int GLOBAL_BLOCK = {F.GLOBAL_BLOCK};\n")
    o.append(f"constexpr int GLOBAL_OFFSET = {F._GLOBAL_OFFSET};\n")
    o.append(f"constexpr int NUM_FEATURES = {F.NUM_FEATURES};\n")
    assert F.NUM_FEATURES == F.MAX_PLAYERS * F.PLAYER_BLOCK + F.GLOBAL_BLOCK
    o.append("static_assert(NUM_FEATURES == MAX_PLAYERS * PLAYER_BLOCK + GLOBAL_BLOCK, \"layout\");\n\n")
    o.append("// ---- offsets inside one player block (features._P) -------------------------------\n")
    o.append("namespace P {\n")
    for name, _ in F._PLAYER_FEATURES:
        o.append(f"constexpr int {_cident(name)} = {F._P[name]};\n")
    o.append("}  // namespace P\n\n")
    o.append("// ---- offsets inside the global block (features._G) -------------------------------\n")
    o.append("namespace G {\n")
    for name, _ in F._GLOBAL_FEATURES:
        o.append(f"constexpr int {_cident(name)} = {F._G[name]};\n")
    o.append("}  // namespace G\n\n")
    o.append("// ---- full feature names (index = position in the vector) -------------------------\n")
    o.append("constexpr const char* FEATURE_NAMES[NUM_FEATURES] = {\n")
    for name in F.FEATURE_NAMES:
        o.append(f'    "{name}",\n')
    o.append("};\n")
    o.append("\n}  // namespace catanbot\n")
    return "".join(o)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only verify that the headers are up to date")
    args = ap.parse_args(argv)
    outputs = {BOARD_HEADER: gen_board_tables(), LAYOUT_HEADER: gen_feature_layout()}
    stale = []
    for path, text in outputs.items():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                current = fh.read()
        except FileNotFoundError:
            current = None
        if current != text:
            stale.append(path)
            if not args.check:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                print(f"wrote {os.path.relpath(path, ROOT)}")
    if args.check:
        if stale:
            for p in stale:
                print(f"STALE: {os.path.relpath(p, ROOT)} (re-run scripts/gen_board_tables.py)")
            return 1
        print("headers up to date")
    elif not stale:
        print("headers already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
