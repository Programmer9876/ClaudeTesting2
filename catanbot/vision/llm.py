"""Claude-vision parser for Colonist.io screenshots.

The image is sent to the Anthropic Messages API together with a system prompt
that explains the Colonist.io board conventions and catanbot's indexing rules.
The model answers by calling a single tool, ``report_game_state``, whose
input schema is derived from :data:`catanbot.vision.schema.PARSE_SCHEMA`.

Because the model cannot know catanbot's vertex / edge numbering, pieces are
reported *relative to a hex*:

* a building is ``{"hex": h, "corner": k}`` with corner ``0`` the top corner
  of hex ``h`` and ``k`` increasing clockwise;
* a road is ``{"hex": h, "side": k}`` where side ``k`` joins corner ``k`` and
  corner ``k + 1``;
* a port is ``{"hex": h, "side": k, "type": ...}`` on the coastal side of a
  land hex.

:func:`convert_pieces` maps those to vertex / edge ids through
:data:`catanbot.board.HEX_VERTICES` and :data:`catanbot.board.HEX_EDGES`, after
which the result is an ordinary parsed-screenshot dict (DESIGN §7) that goes
through :func:`catanbot.vision.schema.validate` and
:func:`catanbot.vision.schema.parsed_to_state`.

With ``read_log=True`` (a card-counting session, :mod:`catanbot.colonist_log`) the prompt and the
tool also ask for the visible game-log entries (the optional ``log`` field of the parse); by default
the request is exactly the board / panel / hand transcription.

The ``anthropic`` package is an optional dependency and is imported lazily;
tests inject a fake client and never touch the network.  Nothing in this
module prints: problems are either raised as :class:`RuntimeError` (no
package, no credentials, API failure, no usable answer) or returned as
warnings on the :class:`~catanbot.vision.result.ParseResult`.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

from .. import board as B
from ..state import PLAYER_COLORS
from . import schema as S
from .result import ParseResult

__all__ = [
    "DEFAULT_MODEL",
    "TOOL_NAME",
    "MAX_IMAGE_SIDE",
    "CONFIDENCE_KEYS",
    "ParseResult",
    "build_prompt",
    "build_tool",
    "build_tool_schema",
    "convert_pieces",
    "corner_to_vertex",
    "side_to_edge",
    "encode_image",
    "load_image",
    "extract_tool_input",
    "parse_with_claude",
]

#: Default model.  Claude Opus 5 supports vision, forced tool use and adaptive
#: thinking; models that reject forced ``tool_choice`` (see
#: :data:`_NO_FORCED_TOOL_CHOICE_PREFIXES`) are handled automatically.
DEFAULT_MODEL = "claude-opus-5"
#: Name of the tool the model must call with the parsed screenshot.
TOOL_NAME = "report_game_state"
#: Longest image side sent to the API (Anthropic's recommended maximum).
MAX_IMAGE_SIDE = 1568
#: Confidence keys the model is asked to report (all 0..1).
CONFIDENCE_KEYS: Tuple[str, ...] = (
    "hexes", "numbers", "robber", "ports", "buildings", "roads", "players", "hand", "overall",
)

# Anthropic's request limit for base64 images is 5 MB; stay under it.
_MAX_IMAGE_BYTES = 4_500_000
# Model families that return HTTP 400 for ``tool_choice`` ``any`` / ``tool``.
_NO_FORCED_TOOL_CHOICE_PREFIXES = ("claude-fable-", "claude-mythos-", "claude-opus-5-5")

_ImageLike = Union[str, "os.PathLike[str]", bytes, bytearray, Image.Image]


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
def load_image(path_or_image: _ImageLike) -> Image.Image:
    """Open ``path_or_image`` (path, raw bytes or a PIL image) as an RGB image."""
    if isinstance(path_or_image, Image.Image):
        img = path_or_image
    elif isinstance(path_or_image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(path_or_image)))
    else:
        path = Path(path_or_image)
        if not path.is_file():
            raise FileNotFoundError(f"screenshot not found: {path}")
        img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def encode_image(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Tuple[str, str, Tuple[int, int]]:
    """Downsize ``image`` to ``max_side`` on the long side and base64-encode it.

    Returns ``(media_type, base64_data, (width, height))``.  PNG is used
    unless the encoded PNG would exceed the API's size limit, in which case
    a high-quality JPEG is sent instead.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / float(long_side)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        image = image.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    media_type = "image/png"
    if len(data) > _MAX_IMAGE_BYTES:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        media_type = "image/jpeg"
    return media_type, base64.standard_b64encode(data).decode("ascii"), image.size


# ---------------------------------------------------------------------------
# Prompt and tool definition
# ---------------------------------------------------------------------------
def _hex_layout_text() -> str:
    rows = []
    for r, row in enumerate(B.HEX_ROWS):
        pad = "  " * (2 - min(r, 4 - r))
        rows.append(pad + "  ".join(f"{h:2d}" for h in row))
    return "\n".join(rows)


#: Prompt section added with ``read_log=True``: transcribe the log panel into ``log``.
LOG_PROMPT = """

## Game log (card counting)
Also transcribe the game log panel (the list of recent game events, newest at the bottom) into `log`:
one object per visible log line, oldest first, exactly in the order shown. Card icons are resources:
lumber = wood, brick, wool = sheep, grain = wheat, ore; a face-down card back is an unknown card (count
it under "unknown"). Player names in the log are drawn in the player's colour: give `player` / `other`
as that colour (the name as written only if the colour is unclear). Map the lines to kinds:
- "X rolled [dice]" -> roll, value = the dice total
- "X got [cards]" -> gain with cards; "X received starting resources [cards]" -> gain with setup: true
- "X built a [road / settlement / city]" -> build with item; "X placed a ..." (setup) -> build, free: true
- "X bought [development card]" -> buy_dev; "X used [card]" -> play_dev with item (knight,
  road_building, year_of_plenty, monopoly, victory_point)
- "X stole N [resource]" right after a Monopoly -> monopoly with resource and count (the total)
- "X took from bank [cards]" (Year of Plenty) -> year_of_plenty with cards
- "X gave bank [cards] and took [cards]" -> bank_trade: cards = given, get = received
- "X traded [cards] for [cards] with Y" -> player_trade: cards = what X gave, get = what X got, other = Y
- "X wants to give [cards] for [cards]" -> offer (cards = offered, get = asked); a counter-offer -> counter
- "X stole [card] from Y" -> steal, other = Y, cards = the card if its face is shown, else {"unknown": 1}
- "X discarded [cards]" -> discard with cards, or count when only a number is shown
- "X moved Robber ..." -> robber; chat, awards and other messages -> other
Put the line as displayed (icons as resource words) in `text`. Never invent lines; if the log panel is
not visible, omit `log`."""


def build_prompt(read_log: bool = False) -> str:
    """System prompt explaining Colonist.io conventions and catanbot's indexing (``read_log``:
    plus :data:`LOG_PROMPT`, the game-log transcription)."""
    layout = _hex_layout_text()
    colors = ", ".join(PLAYER_COLORS)
    resources = ", ".join(B.RESOURCE_NAMES[:5])
    return f"""You are an expert reader of Colonist.io (online Settlers of Catan) screenshots.
Your only job is to transcribe EXACTLY what is visible in the screenshot into a structured
report by calling the `{TOOL_NAME}` tool once. Never answer in prose; never guess pieces that
you cannot see. When something is not visible, omit the optional field and lower the matching
confidence value instead of inventing data.

## Colonist.io board conventions
- The board is the standard 19-hex island drawn with POINTY-TOP hexes in five horizontal rows
  of 3, 4, 5, 4 and 3 hexes, surrounded by blue sea.
- Hex terrain colours: forest / wood (dark green trees), hills / brick (orange-red clay),
  pasture / sheep (light green meadow), fields / wheat (yellow), mountains / ore (grey rock),
  desert (pale sand, no number token). Report resources as one of: {resources}, desert.
- Every non-desert hex has a round cream number token (2..12, never 7); the 6 and 8 tokens are
  printed in red. The robber is a dark grey/black pawn standing on one hex.
- Settlements are small coloured houses, cities are larger coloured buildings (often with a
  tower / church-like shape), roads are coloured bars along hex edges. Piece colours are the
  player colours: {colors}.
- Ports (harbours) are drawn in the sea next to a coastal hex side: a generic port shows
  "3:1", a resource port shows "2:1" with a resource icon. Each port serves the two corners of
  that hex side.
- The player panel lists every player with name, colour, victory points (VP), number of
  resource cards, number of development cards, knights played, and badges / icons for
  Longest Road and Largest Army. The player on turn is usually highlighted, and the dice
  result is shown near the board or in the log. The hand bar at the bottom shows the exact
  resource cards (and development cards) of the player whose screen this is ("me").

## catanbot indexing rules (follow them exactly)
Hexes are numbered 0..18 row by row from the top row to the bottom row, left to right:

{layout}

So the top row is hexes 0,1,2, the second row 3,4,5,6, the middle row 7..11 (7 = far left,
11 = far right), the fourth row 12..15 and the bottom row 16,17,18. `hexes` must contain all
19 entries in this order.

Corners of a hex are numbered 0..5 starting at the TOP corner and going CLOCKWISE on screen:
corner 0 = top, 1 = upper-right, 2 = lower-right, 3 = bottom, 4 = lower-left, 5 = upper-left.
Sides of a hex are numbered 0..5 where side k is the edge joining corner k and corner k+1:
side 0 = upper-right edge, 1 = right (vertical) edge, 2 = lower-right edge,
3 = lower-left edge, 4 = left (vertical) edge, 5 = upper-left edge.

- Describe every settlement and city as {{"hex": h, "corner": k}} using ANY hex that touches
  that corner (shared corners may be given through whichever adjacent hex is most convenient;
  duplicates are merged). Describe every road as {{"hex": h, "side": k}} likewise.
- Describe each port as {{"hex": h, "side": k, "type": t}} where h is the LAND hex next to the
  port, k is the side of that hex that faces the port and t is "3:1" or the resource name.
  Include all ports only if they are visible; there are normally 9.
- `robber` is the hex index the robber pawn stands on.
- `players` are listed in the seat order shown in the panel (top to bottom). Use the piece
  colour of each player as `color`. Fill `vp`, `cards`, `dev_cards`, `knights`,
  `longest_road`, `largest_army` from the panel. For "me" (the screen owner) also fill
  `resources` with the exact hand from the hand bar; if the development cards in the hand bar
  are visible, give `dev_cards` as an object with exact counts by type, otherwise an integer.
- `me` is the colour of the screen owner; `current_player` is the colour of the player on
  turn; `dice` is the last dice total shown (omit if none is visible). Set `rolled` to true when
  the player on turn has already rolled this turn and to false when the roll is still pending
  (the dice may still display the previous player's roll); omit it if you cannot tell.
- If the bank stock or the size of the development deck is displayed, report `bank` /
  `dev_deck_remaining`; otherwise omit them.

## Confidence
Fill the `confidence` object with a number from 0 to 1 for each of:
{", ".join(CONFIDENCE_KEYS)}. Use 1.0 only when the item is clearly and completely readable.

Work carefully: first locate the five rows of hexes, then read the terrain and number of each
hex in order, then the robber, then walk around the coast for ports, then every piece, then the
player panel and the hand bar. Finally call `{TOOL_NAME}` exactly once.""" + (LOG_PROMPT if read_log else "")


def _hex_ref_schema(kind: str, what: str) -> Dict[str, Any]:
    """Schema for ``{"hex": h, "corner": k}`` / ``{"hex": h, "side": k}`` lists."""
    return {
        "type": "array",
        "description": what,
        "items": {
            "type": "object",
            "required": ["hex", kind],
            "additionalProperties": False,
            "properties": {
                "hex": {"type": "integer", "minimum": 0, "maximum": B.NUM_HEXES - 1,
                        "description": "Hex index 0..18 (row by row, top to bottom, left to right)."},
                kind: {"type": "integer", "minimum": 0, "maximum": 5,
                       "description": ("Corner 0..5 of that hex, 0 = top, clockwise." if kind == "corner"
                                       else "Side 0..5 of that hex; side k joins corner k and corner k+1.")},
            },
        },
    }


def build_tool_schema(read_log: bool = False) -> Dict[str, Any]:
    """``input_schema`` of the report tool: :data:`PARSE_SCHEMA` with hex-relative pieces.

    Buildings become ``{hex, corner}``, roads and ports ``{hex, side}``, and a
    ``confidence`` object is added.  Draft-specific keys (``$schema``,
    ``title``) are dropped because the API expects a plain object schema.  The optional
    game ``log`` is only part of it with ``read_log`` (card counting).
    """
    schema = copy.deepcopy(S.PARSE_SCHEMA)
    schema.pop("$schema", None)
    schema.pop("title", None)
    if not read_log:
        schema["properties"].pop("log", None)
    schema["description"] = ("Everything visible on a Colonist.io game screenshot. Hexes 0..18 in "
                             "row order; buildings / roads / ports relative to a hex (see the system prompt).")
    props = schema["properties"]
    props["ports"] = {
        "type": "array",
        "description": "Ports next to coastal hex sides. Omit if not visible.",
        "items": {
            "type": "object",
            "required": ["hex", "side", "type"],
            "additionalProperties": False,
            "properties": {
                "hex": {"type": "integer", "minimum": 0, "maximum": B.NUM_HEXES - 1,
                        "description": "Land hex next to the port."},
                "side": {"type": "integer", "minimum": 0, "maximum": 5,
                         "description": "Side of that hex facing the sea where the port is drawn."},
                "type": {"type": "string",
                         "description": "'3:1' for a generic port or the resource name of a 2:1 port."},
            },
        },
    }
    pprops = props["players"]["items"]["properties"]
    pprops["settlements"] = _hex_ref_schema("corner", "Settlements as {hex, corner}.")
    pprops["cities"] = _hex_ref_schema("corner", "Cities as {hex, corner}.")
    pprops["roads"] = _hex_ref_schema("side", "Roads as {hex, side}.")
    props["confidence"] = {
        "type": "object",
        "description": "Per-item confidence 0..1: " + ", ".join(CONFIDENCE_KEYS) + ".",
        "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in CONFIDENCE_KEYS},
        "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
    }
    schema["required"] = ["hexes", "players", "confidence"]
    return schema


def build_tool(read_log: bool = False) -> Dict[str, Any]:
    """Tool definition passed in ``tools=[...]`` to the Messages API."""
    return {
        "name": TOOL_NAME,
        "description": ("Report everything visible on the Colonist.io screenshot: the 19 hexes in "
                        "catanbot order, robber, ports, every player's pieces (relative to hexes), "
                        "panel counters, the screen owner's hand and per-item confidence."
                        + (" Also the visible game-log entries." if read_log else "")),
        "input_schema": build_tool_schema(read_log),
    }


# ---------------------------------------------------------------------------
# Conversion of hex-relative pieces to vertex / edge ids
# ---------------------------------------------------------------------------
def corner_to_vertex(hex_index: int, corner: int) -> int:
    """Vertex id of corner ``corner`` (0 = top, clockwise) of hex ``hex_index``."""
    return B.HEX_VERTICES[hex_index][corner % 6]


def side_to_edge(hex_index: int, side: int) -> int:
    """Edge id of side ``side`` (between corner ``side`` and ``side + 1``) of hex ``hex_index``."""
    return B.HEX_EDGES[hex_index][side % 6]


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ref_to_id(entry: Any, key: str, table: Sequence[Sequence[int]], maximum: int,
               label: str, warnings: List[str]) -> Optional[int]:
    """Resolve one ``{"hex": h, key: k}`` (or ``[h, k]`` or a bare id) to an id."""
    if isinstance(entry, dict):
        h = _as_int(entry.get("hex"))
        k = _as_int(entry.get(key))
        if k is None:
            k = _as_int(entry.get("side" if key == "corner" else "corner"))
        if h is None or k is None:
            warnings.append(f"{label}: malformed entry {entry!r}")
            return None
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        h, k = _as_int(entry[0]), _as_int(entry[1])
        if h is None or k is None:
            warnings.append(f"{label}: malformed entry {entry!r}")
            return None
    else:
        i = _as_int(entry)
        if i is None or not 0 <= i < maximum:
            warnings.append(f"{label}: malformed entry {entry!r}")
            return None
        return i
    if not 0 <= h < B.NUM_HEXES:
        warnings.append(f"{label}: hex {h} out of range 0..{B.NUM_HEXES - 1} in {entry!r}")
        return None
    if not 0 <= k <= 5:
        warnings.append(f"{label}: {key} {k} out of range 0..5 in {entry!r}")
        return None
    return table[h][k]


def _convert_list(entries: Any, key: str, table: Sequence[Sequence[int]], maximum: int,
                  label: str, warnings: List[str]) -> List[int]:
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)):
        warnings.append(f"{label}: expected a list, got {type(entries).__name__}")
        return []
    out: List[int] = []
    for entry in entries:
        i = _ref_to_id(entry, key, table, maximum, label, warnings)
        if i is not None and i not in out:
            out.append(i)
    return out


def convert_pieces(parsed_llm: Dict[str, Any], warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert the tool result (hex-relative pieces) into a DESIGN §7 parsed dict.

    ``settlements`` / ``cities`` become vertex ids, ``roads`` and ``ports``
    edge ids.  Bare integers are accepted as already-converted ids.  Malformed
    or out-of-range entries are dropped and described in ``warnings`` (a list
    that is appended to when given).  The ``confidence`` object is left in
    place; :func:`parse_with_claude` moves it to the result.  The input is not
    modified.
    """
    warnings = warnings if warnings is not None else []
    parsed: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in parsed_llm.items()
                              if k not in ("players", "ports")}
    ports_in = parsed_llm.get("ports")
    if ports_in is not None:
        ports_out: List[Dict[str, Any]] = []
        seen_edges: set = set()
        if isinstance(ports_in, (list, tuple)):
            for entry in ports_in:
                if isinstance(entry, dict) and "edge" in entry and "hex" not in entry:
                    e = _as_int(entry.get("edge"))
                    ptype = entry.get("type")
                else:
                    e = _ref_to_id(entry, "side", B.HEX_EDGES, B.NUM_EDGES, "port", warnings)
                    ptype = entry.get("type") if isinstance(entry, dict) else None
                if e is None or not 0 <= e < B.NUM_EDGES:
                    if e is not None:
                        warnings.append(f"port: edge {e} out of range")
                    continue
                if e in seen_edges:
                    continue
                seen_edges.add(e)
                ports_out.append({"edge": e, "type": ptype})
        else:
            warnings.append("ports: expected a list")
        parsed["ports"] = ports_out
    players_out: List[Dict[str, Any]] = []
    for i, rp in enumerate(parsed_llm.get("players") or []):
        if not isinstance(rp, dict):
            warnings.append(f"player {i}: malformed entry {rp!r}")
            continue
        label = str(rp.get("color") or rp.get("name") or f"player {i}")
        p: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in rp.items()
                             if k not in ("settlements", "cities", "roads")}
        p["settlements"] = _convert_list(rp.get("settlements"), "corner", B.HEX_VERTICES, B.NUM_VERTICES,
                                         f"{label} settlements", warnings)
        p["cities"] = _convert_list(rp.get("cities"), "corner", B.HEX_VERTICES, B.NUM_VERTICES,
                                    f"{label} cities", warnings)
        p["roads"] = _convert_list(rp.get("roads"), "side", B.HEX_EDGES, B.NUM_EDGES,
                                   f"{label} roads", warnings)
        # A vertex reported both as settlement and city is a city (the larger piece).
        p["settlements"] = [v for v in p["settlements"] if v not in p["cities"]]
        players_out.append(p)
    parsed["players"] = players_out
    return parsed


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------
def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from free text (fenced or bare)."""
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_tool_input(response: Any, tool_name: str = TOOL_NAME) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Find the ``tool_use`` input named ``tool_name`` in a Messages API response.

    Works with SDK objects (attributes) and plain dicts.  Falls back to a JSON
    object embedded in a text block (with a warning) when the model did not
    call the tool.  Returns ``(input_or_None, warnings)``.
    """
    warnings: List[str] = []
    content = _block_attr(response, "content", None) or []
    fallback: Optional[Dict[str, Any]] = None
    texts: List[str] = []
    for block in content:
        btype = _block_attr(block, "type")
        if btype == "tool_use":
            name = _block_attr(block, "name")
            inp = _block_attr(block, "input")
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except ValueError:
                    warnings.append("tool input was not valid JSON")
                    continue
            if name == tool_name and isinstance(inp, dict):
                return inp, warnings
            if isinstance(inp, dict) and fallback is None:
                fallback = inp
                warnings.append(f"model called tool {name!r} instead of {tool_name!r}")
        elif btype == "text":
            t = _block_attr(block, "text")
            if isinstance(t, str):
                texts.append(t)
    if fallback is not None:
        return fallback, warnings
    for t in texts:
        obj = _json_from_text(t)
        if obj is not None:
            warnings.append("model answered with JSON text instead of the tool call")
            return obj, warnings
    return None, warnings


def _normalise_confidence(raw: Any, warnings: List[str]) -> Dict[str, float]:
    conf: Dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            conf[str(k)] = min(1.0, max(0.0, f))
    else:
        warnings.append("model did not report confidence values")
    if "overall" not in conf and conf:
        conf["overall"] = min(conf.values())
    return conf


def _usage_dict(response: Any) -> Dict[str, Any]:
    usage = _block_attr(response, "usage", None)
    if usage is None:
        return {}
    out: Dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        val = _block_attr(usage, key, None)
        if isinstance(val, int):
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Client / API plumbing
# ---------------------------------------------------------------------------
def _import_anthropic() -> Any:
    try:
        import anthropic  # noqa: WPS433 (lazy optional dependency)
    except ImportError as exc:
        raise RuntimeError(
            "The LLM parser needs the 'anthropic' package, which is not installed. "
            "Install it with: pip install anthropic   (or: pip install 'catanbot[llm]')"
        ) from exc
    return anthropic


def _make_client(api_key: Optional[str]) -> Any:
    """Build the SDK client, letting the SDK resolve credentials itself.

    Besides ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` the SDK can use a
    stored ``ant auth login`` profile or workload-identity variables, so no
    environment pre-check is done here; the SDK's own credential error is
    translated into a hint.
    """
    anthropic = _import_anthropic()
    try:
        return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    except Exception as exc:  # the SDK raises AnthropicError when no credential resolves
        raise RuntimeError(
            f"No Anthropic credentials found ({exc}). Set an API key with:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "(or pass api_key=... to parse_with_claude, or log in with `ant auth login`)."
        ) from exc


def _api_error_types() -> Tuple[type, ...]:
    """Exception classes of the SDK worth translating (empty if the SDK is absent)."""
    try:
        import anthropic
    except ImportError:
        return ()
    types: List[type] = []
    for name in ("APIStatusError", "APIConnectionError", "APIError", "AnthropicError"):
        t = getattr(anthropic, name, None)
        if isinstance(t, type) and issubclass(t, BaseException):
            types.append(t)
    return tuple(types)


def _supports_forced_tool_choice(model: str) -> bool:
    return not model.startswith(_NO_FORCED_TOOL_CHOICE_PREFIXES)


def _describe_api_error(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    msg = getattr(exc, "message", None) or str(exc)
    name = type(exc).__name__
    if status == 401:
        return f"authentication failed ({name}): check ANTHROPIC_API_KEY. {msg}"
    if status == 429:
        return f"rate limited ({name}): retry later. {msg}"
    if isinstance(status, int) and status >= 500:
        return f"Anthropic server error {status} ({name}): retry later. {msg}"
    if status is not None:
        return f"API error {status} ({name}): {msg}"
    return f"could not reach the Anthropic API ({name}): {msg}"


def _call_model(client: Any, request: Dict[str, Any], forced: bool) -> Any:
    """Send the request, retrying once without forced ``tool_choice`` if rejected."""
    err_types = _api_error_types()
    try:
        return client.messages.create(**request)
    except err_types as exc:  # type: ignore[misc]
        status = getattr(exc, "status_code", None)
        text = (getattr(exc, "message", None) or str(exc)).lower()
        if forced and status == 400 and "tool_choice" in text:
            retry = dict(request)
            retry["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
            try:
                return client.messages.create(**retry)
            except err_types as exc2:  # type: ignore[misc]
                raise RuntimeError("Claude request failed: " + _describe_api_error(exc2)) from exc2
        raise RuntimeError("Claude request failed: " + _describe_api_error(exc)) from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def parse_with_claude(path_or_image: _ImageLike, me: Optional[str] = None, model: Optional[str] = None,
                      api_key: Optional[str] = None, client: Any = None, *, max_tokens: int = 16000,
                      effort: Optional[str] = None, extra_instructions: Optional[str] = None,
                      read_log: bool = False) -> ParseResult:
    """Parse a Colonist.io screenshot with Claude vision.

    Parameters
    ----------
    path_or_image:
        Image path, raw image bytes or a ``PIL.Image``.
    me:
        Colour of the screen owner; overrides what the model reports.
    model:
        Model id (default :data:`DEFAULT_MODEL`).
    api_key:
        API key; otherwise ``ANTHROPIC_API_KEY`` from the environment.
    client:
        Pre-built client exposing ``messages.create(**kwargs)``; used by
        tests to avoid the network.  When given, the package / key checks
        are skipped.
    max_tokens, effort:
        Passed to the API (``effort`` -> ``output_config.effort`` when given).
    extra_instructions:
        Optional text appended to the user message (e.g. "I am blue").
    read_log:
        Also transcribe the visible game log into ``parsed["log"]`` (card counting; normalised by
        :func:`catanbot.colonist_log.events_from_json`, problems become warnings).

    Raises
    ------
    RuntimeError
        If ``anthropic`` is not installed, no API key is available, the API
        call fails, or the model produced no usable report.
    FileNotFoundError
        If ``path_or_image`` is a path that does not exist.
    """
    model = model or DEFAULT_MODEL
    me = me.strip().lower() if isinstance(me, str) and me.strip() else None
    image = load_image(path_or_image)
    media_type, data, sent_size = encode_image(image)
    if client is None:
        client = _make_client(api_key)

    user_text = ("Transcribe this Colonist.io screenshot into the report_game_state tool call. "
                 "Read every hex, number token, the robber, ports, all pieces and the player panel.")
    if me:
        user_text += f" The screen owner ('me') is the {me} player."
    if read_log:
        user_text += " Also transcribe every visible game-log line into `log`."
    if extra_instructions:
        user_text += " " + extra_instructions.strip()
    forced = _supports_forced_tool_choice(model)
    request: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": build_prompt(read_log) if read_log else build_prompt(),
        "tools": [build_tool(read_log) if read_log else build_tool()],
        "tool_choice": ({"type": "tool", "name": TOOL_NAME} if forced
                        else {"type": "auto", "disable_parallel_tool_use": True}),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": user_text},
            ],
        }],
    }
    if effort:
        request["output_config"] = {"effort": effort}

    response = _call_model(client, request, forced)

    stop_reason = _block_attr(response, "stop_reason")
    if stop_reason == "refusal":
        details = _block_attr(response, "stop_details")
        explanation = _block_attr(details, "explanation", None) if details is not None else None
        raise RuntimeError("Claude declined to process the screenshot"
                           + (f": {explanation}" if explanation else "."))
    raw, warnings = extract_tool_input(response, TOOL_NAME)
    if raw is None:
        if stop_reason == "max_tokens":
            raise RuntimeError("Claude's answer was cut off by max_tokens before the report was complete; "
                               "retry with a larger max_tokens.")
        raise RuntimeError(f"Claude did not return a {TOOL_NAME} report (stop_reason={stop_reason!r}).")

    parsed = convert_pieces(raw, warnings)
    confidence = _normalise_confidence(parsed.pop("confidence", None), warnings)
    colors = [str(p.get("color", "")).strip().lower() for p in parsed.get("players", []) if isinstance(p, dict)]
    model_me = parsed.get("me")
    model_me = model_me.strip().lower() if isinstance(model_me, str) and model_me.strip() else None
    if me and (me in colors or not colors):
        parsed["me"] = me
        if model_me is not None and model_me != me:
            # the hand bar shows the screen owner's cards only: a hand the model attached to
            # another player was mis-attributed and must not count as exact knowledge
            for p in parsed.get("players", []):
                if isinstance(p, dict) and str(p.get("color", "")).strip().lower() != me and "resources" in p:
                    p.pop("resources", None)
                    warnings.append(f"model attributed the hand bar to {p.get('color')!r}; ignored because me={me!r}")
    elif me:
        # Same contract as the CV parser: an unknown colour is reported, never propagated,
        # so ``state.player_index(parsed["me"])`` cannot fail downstream.
        fallback = model_me if model_me in colors else colors[0]
        warnings.append(f"requested me={me!r} is not one of the detected players {colors}; using {fallback!r}")
        parsed["me"] = fallback
    elif model_me is None and colors:
        warnings.append("model did not identify the screen owner; assuming the first player")
        parsed["me"] = colors[0]
    elif model_me is not None and colors and model_me not in colors:
        warnings.append(f"model reported me={model_me!r} which is not one of the detected players {colors}; "
                        f"assuming {colors[0]!r}")
        parsed["me"] = colors[0]
    if read_log:
        from ..colonist_log import events_from_json
        if parsed.get("log") is None:
            warnings.append("model did not transcribe the game log (not visible?): card counting only "
                            "reconciles the hand sizes")
        else:
            events, log_warnings = events_from_json(parsed.get("log"))
            parsed["log"] = [e.to_dict() for e in events]
            warnings.extend("log: " + w for w in log_warnings)
    warnings.extend(S.validate(parsed))
    state = S.parsed_to_state(parsed)
    debug: Dict[str, Any] = {
        "model": _block_attr(response, "model", model),
        "stop_reason": stop_reason,
        "usage": _usage_dict(response),
        "image_size": list(sent_size),
        "media_type": media_type,
        "raw": raw,
    }
    return ParseResult(parsed=parsed, state=state, confidence=confidence, warnings=warnings, debug=debug)
