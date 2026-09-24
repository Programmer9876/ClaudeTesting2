"""Parsed-screenshot schema, conversion to :class:`GameState`, and validation.

A *parsed screenshot* is a plain JSON-compatible dict (see :data:`PARSE_SCHEMA`
and :data:`EXAMPLE_PARSED`) produced by the CV parser
(``catanbot.vision.colonist``), the LLM parser (``catanbot.vision.llm``) or a
human.  It contains only what is visible on a Colonist.io screen: the board,
the pieces, public per-player counters and *my* own hand.

:func:`parsed_to_state` turns such a dict into a :class:`GameState` by
applying standard priors for everything that is hidden:

* opponents get ``hand_known=False`` / ``dev_known=False`` with only the
  visible card counts;
* the bank is ``19`` minus every card whose type is known (the hands flagged
  ``hand_known``), never below zero, unless the parse carries a ``bank``;
* the development deck starts from the standard ``14/5/2/2/2`` counts,
  every played knight is removed from the knights, dev cards whose type is
  known are removed exactly, and the remaining *held-but-unknown* dev cards
  are removed proportionally to the remaining per-type counts (largest
  remainder rounding, never below zero).  When ``dev_deck_remaining`` is
  given, the number of unknown held cards is inferred from it instead of the
  players' ``dev_cards`` counters.

:func:`validate` returns human readable warnings about anything that looks
wrong in a parse (bad multisets, illegal piece positions, ...), and
:func:`state_to_parsed` is the inverse of :func:`parsed_to_state` used by the
synthetic renderer and the round-trip tests.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .. import board as B
from ..state import PHASE_MAIN, PLAYER_COLORS, GameState, Player

__all__ = [
    "PARSE_SCHEMA",
    "EXAMPLE_PARSED",
    "PORT_ALIASES",
    "parsed_to_state",
    "state_to_parsed",
    "validate",
    "resource_from_name",
    "port_type_from_name",
    "longest_road_length",
]

# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------
_RESOURCE_NAME_SCHEMA = {
    "type": "string",
    "description": "Resource name; aliases from board.RESOURCE_ALIASES are accepted "
                   "(lumber/forest, hills/clay, wool/pasture, grain/fields, mountains/stone).",
    "enum": sorted(set(B.RESOURCE_ALIASES)),
}

def _id_list_schema(maximum: int, what: str) -> Dict[str, Any]:
    """JSON schema for a list of unique integer ids in ``0..maximum``."""
    return {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": maximum},
        "uniqueItems": True,
        "description": what,
    }

PARSE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ParsedColonistScreenshot",
    "description": "Everything visible on a Colonist.io game screenshot, in catanbot board "
                   "index order (hexes 0..18 row by row, vertices 0..53, edges 0..71).",
    "type": "object",
    "required": ["hexes", "players"],
    "additionalProperties": True,
    "properties": {
        "hexes": {
            "type": "array",
            "minItems": B.NUM_HEXES,
            "maxItems": B.NUM_HEXES,
            "description": "19 hexes, row by row top to bottom, left to right (3-4-5-4-3).",
            "items": {
                "type": "object",
                "required": ["resource"],
                "properties": {
                    "resource": _RESOURCE_NAME_SCHEMA,
                    "number": {
                        "type": ["integer", "null"],
                        "minimum": 2,
                        "maximum": 12,
                        "description": "Number token (2..12 except 7); null/absent for the desert.",
                    },
                },
            },
        },
        "robber": {
            "type": "integer", "minimum": 0, "maximum": B.NUM_HEXES - 1,
            "description": "Hex index holding the robber (defaults to the desert).",
        },
        "ports": {
            "type": "array",
            "description": "Optional. Ports attached to coastal edges. Missing -> standard layout.",
            "items": {
                "type": "object",
                "required": ["edge", "type"],
                "properties": {
                    "edge": {"type": "integer", "minimum": 0, "maximum": B.NUM_EDGES - 1},
                    "type": {
                        "type": "string",
                        "description": "'3:1' (generic) or a resource name for a 2:1 port.",
                    },
                },
            },
        },
        "players": {
            "type": "array",
            "minItems": 1,
            "maxItems": len(PLAYER_COLORS),
            "description": "Players in seat (turn) order as shown in the player panel.",
            "items": {
                "type": "object",
                "required": ["color"],
                "properties": {
                    "color": {"type": "string", "enum": list(PLAYER_COLORS)},
                    "name": {"type": "string"},
                    "vp": {"type": "integer", "minimum": 0, "description": "Visible victory points."},
                    "cards": {"type": "integer", "minimum": 0, "description": "Resource cards in hand."},
                    "dev_cards": {
                        "type": ["integer", "object"],
                        "minimum": 0,
                        "description": "Unplayed dev cards held. An int (count only) or, for 'me', "
                                       "an object {knight, victory_point, road_building, "
                                       "year_of_plenty, monopoly} with exact counts.",
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                    "knights": {"type": "integer", "minimum": 0, "description": "Knights played."},
                    "longest_road": {"type": "boolean"},
                    "largest_army": {"type": "boolean"},
                    "settlements": _id_list_schema(B.NUM_VERTICES - 1, "Vertex ids with a settlement."),
                    "cities": _id_list_schema(B.NUM_VERTICES - 1, "Vertex ids with a city."),
                    "roads": _id_list_schema(B.NUM_EDGES - 1, "Edge ids with a road."),
                    "resources": {
                        "type": "object",
                        "description": "Exact hand by resource; only for 'me'.",
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                },
            },
        },
        "me": {"type": "string", "description": "Colour of the player whose screen this is."},
        "current_player": {"type": "string", "description": "Colour of the player on turn."},
        "dice": {"type": ["integer", "null"], "minimum": 0, "maximum": 12,
                 "description": "Last dice total shown (0/null if not rolled yet)."},
        "bank": {
            "type": "object",
            "description": "Optional. Bank stock per resource if visible.",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "dev_deck_remaining": {"type": ["integer", "null"], "minimum": 0, "maximum": 25,
                               "description": "Optional. Dev cards left in the deck if visible."},
        "turn": {"type": "integer", "minimum": 0, "description": "Optional. Turn counter if visible."},
    },
}

# Port name aliases (in addition to board.RESOURCE_ALIASES for 2:1 ports).
PORT_ALIASES: Dict[str, int] = {
    "3:1": B.PORT_GENERIC, "3": B.PORT_GENERIC, "generic": B.PORT_GENERIC, "any": B.PORT_GENERIC,
    "?": B.PORT_GENERIC, "3to1": B.PORT_GENERIC, "3-1": B.PORT_GENERIC,
}

_STANDARD_NUMBER_COUNTS = Counter(B.STANDARD_NUMBERS)
_COASTAL_EDGE_SET: Set[int] = set(B.COASTAL_EDGES)


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------
def resource_from_name(name: Union[str, int, None]) -> Optional[int]:
    """Resource index for a name/alias (or an int index); ``None`` if unknown."""
    if name is None:
        return None
    if isinstance(name, bool):
        return None
    if isinstance(name, int):
        return name if 0 <= name <= B.DESERT else None
    key = str(name).strip().lower()
    return B.RESOURCE_ALIASES.get(key)


def port_type_from_name(name: Union[str, int, None]) -> Optional[int]:
    """Port type index (0..4 = 2:1 resource port, 5 = generic) for a label.

    Accepts ``"3:1"``, ``"generic"``, ``"any"``, resource names and aliases,
    and labels like ``"2:1 wood"`` / ``"wood 2:1"`` / ``"ore port"``.
    """
    if name is None or isinstance(name, bool):
        return None
    if isinstance(name, int):
        return name if 0 <= name <= B.PORT_GENERIC else None
    key = str(name).strip().lower()
    if key in PORT_ALIASES:
        return PORT_ALIASES[key]
    cleaned = key.replace("2:1", " ").replace("2to1", " ").replace("port", " ").replace("harbor", " ")
    cleaned = cleaned.replace("harbour", " ").replace(":", " ").strip()
    for token in cleaned.split():
        if token in PORT_ALIASES:
            return PORT_ALIASES[token]
        res = B.RESOURCE_ALIASES.get(token)
        if res is not None and res != B.DESERT:
            return res
    return None


def _color_of(p: Any) -> str:
    """Normalised colour string of a parsed player entry."""
    return str(p.get("color", p.get("colour", "")) or "").strip().lower()


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _id_list(values: Any, maximum: int) -> List[int]:
    """Clean list of unique in-range ids, preserving first-seen order."""
    out: List[int] = []
    seen: Set[int] = set()
    for v in values or ():
        i = _int(v, -1)
        if 0 <= i < maximum and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _resource_counts(d: Any) -> List[int]:
    """``{"wood": 1, ...}`` (aliases ok) or a 5-list -> 5 counts."""
    counts = [0] * 5
    if isinstance(d, dict):
        for k, v in d.items():
            r = resource_from_name(k)
            if r is not None and r != B.DESERT:
                counts[r] += max(0, _int(v))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(list(d)[:5]):
            counts[i] = max(0, _int(v))
    return counts


def _dev_counts(d: Any) -> List[int]:
    counts = [0] * 5
    if isinstance(d, dict):
        for k, v in d.items():
            key = str(k).strip().lower()
            if key in B.DEV_NAMES:
                counts[B.DEV_NAMES.index(key)] += max(0, _int(v))
            elif key in ("vp", "victory", "victorypoint", "victory point"):
                counts[B.DEV_VP] += max(0, _int(v))
            elif key in ("yop", "plenty"):
                counts[B.DEV_YEAR_OF_PLENTY] += max(0, _int(v))
            elif key in ("roads", "road"):
                counts[B.DEV_ROAD_BUILDING] += max(0, _int(v))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(list(d)[:5]):
            counts[i] = max(0, _int(v))
    return counts


# ---------------------------------------------------------------------------
# Longest road (used for state.longest_road_len and validation)
# ---------------------------------------------------------------------------
def longest_road_length(roads: Iterable[int], blocked: Iterable[int] = ()) -> int:
    """Length of the longest simple trail through ``roads`` (edge ids).

    ``blocked`` are vertices holding an opponent building: a trail may end
    there but not pass through.  Edges are never reused; vertices may be.
    """
    road_set = set(roads)
    if not road_set:
        return 0
    blocked_set = set(blocked)
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for e in road_set:
        a, b = B.EDGE_VERTICES[e]
        adj.setdefault(a, []).append((e, b))
        adj.setdefault(b, []).append((e, a))
    best = 0
    used: Set[int] = set()

    def dfs(v: int, length: int) -> None:
        nonlocal best
        if length > best:
            best = length
        if length and v in blocked_set:
            return
        for e, w in adj[v]:
            if e not in used:
                used.add(e)
                dfs(w, length + 1)
                used.discard(e)

    for v in adj:
        dfs(v, 0)
    return best


# ---------------------------------------------------------------------------
# parsed -> GameState
# ---------------------------------------------------------------------------
def _resolve_me(parsed: Dict[str, Any], colors: List[str]) -> int:
    me = parsed.get("me")
    if isinstance(me, int) and not isinstance(me, bool) and 0 <= me < len(colors):
        return me
    if isinstance(me, str):
        key = me.strip().lower()
        if key in colors:
            return colors.index(key)
    return 0


def _apportion(total: int, weights: Sequence[int]) -> List[int]:
    """Split ``total`` over ``weights`` proportionally (largest remainder), capped by weights."""
    total = max(0, min(total, sum(weights)))
    if total == 0:
        return [0] * len(weights)
    wsum = float(sum(weights))
    raw = [total * w / wsum for w in weights]
    out = [int(x) for x in raw]
    remaining = total - sum(out)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order:
        if remaining <= 0:
            break
        if out[i] < weights[i]:
            out[i] += 1
            remaining -= 1
    return out


def parsed_to_state(parsed: Dict[str, Any]) -> GameState:
    """Build a :class:`GameState` from a parsed screenshot dict (see module doc).

    Never raises on missing optional fields; unknown resource names fall back
    to the desert, out-of-range ids are dropped.  Use :func:`validate` to get
    warnings about such problems.
    """
    s = GameState()
    # --- board ------------------------------------------------------------
    hexes: List[Tuple[int, int]] = []
    for h in (parsed.get("hexes") or [])[: B.NUM_HEXES]:
        if isinstance(h, dict):
            res = resource_from_name(h.get("resource"))
            num = _int(h.get("number"), 0)
        else:
            res = resource_from_name(h[0]) if h else None
            num = _int(h[1], 0) if h and len(h) > 1 else 0
        if res is None:
            res = B.DESERT
        hexes.append((res, 0 if res == B.DESERT else num))
    while len(hexes) < B.NUM_HEXES:
        hexes.append((B.DESERT, 0))
    s.hexes = hexes

    desert = next((i for i, (r, _) in enumerate(hexes) if r == B.DESERT), 9)
    robber = _int(parsed.get("robber"), -1)
    s.robber = robber if 0 <= robber < B.NUM_HEXES else desert

    ports = parsed.get("ports")
    if ports:
        port_edges: List[Tuple[int, int]] = []
        for entry in ports:
            if isinstance(entry, dict):
                e, t = _int(entry.get("edge"), -1), port_type_from_name(entry.get("type"))
            else:
                e, t = _int(entry[0], -1), port_type_from_name(entry[1])
            if 0 <= e < B.NUM_EDGES and t is not None:
                port_edges.append((e, t))
        s.ports = B.ports_from_edges(port_edges)
    else:
        s.ports = dict(B.STANDARD_PORTS)

    # --- players ----------------------------------------------------------
    raw_players = [p for p in (parsed.get("players") or []) if isinstance(p, dict)]
    colors = [_color_of(p) for p in raw_players]
    me = _resolve_me(parsed, colors) if raw_players else -1
    players: List[Player] = []
    for i, rp in enumerate(raw_players):
        p = Player(color=colors[i] or f"player{i}", name=str(rp.get("name") or colors[i] or f"player{i}"))
        p.settlements = _id_list(rp.get("settlements"), B.NUM_VERTICES)
        p.cities = _id_list(rp.get("cities"), B.NUM_VERTICES)
        p.roads = _id_list(rp.get("roads"), B.NUM_EDGES)
        p.played_knights = max(0, _int(rp.get("knights")))
        cards = max(0, _int(rp.get("cards")))
        resources = rp.get("resources")
        if i == me and isinstance(resources, (dict, list, tuple)):
            p.resources = _resource_counts(resources)
            p.hand_known = True
            p.hand_size = sum(p.resources)
        else:
            p.resources = [0] * 5
            p.hand_known = False
            p.hand_size = cards
        dev = rp.get("dev_cards")
        if isinstance(dev, (dict, list, tuple)):
            p.dev_cards = _dev_counts(dev)
            p.dev_known = True
            p.dev_count = sum(p.dev_cards)
        else:
            p.dev_cards = [0] * 5
            p.dev_known = False
            p.dev_count = max(0, _int(dev))
        players.append(p)
    s.players = players

    # --- bank -------------------------------------------------------------
    bank = parsed.get("bank")
    if isinstance(bank, (dict, list, tuple)) and bank:
        counts = _resource_counts(bank)
        if isinstance(bank, dict):
            # keep 19 for resources the parser did not mention
            mentioned = {resource_from_name(k) for k in bank}
            counts = [counts[r] if r in mentioned else B.BANK_PER_RESOURCE for r in range(5)]
        s.bank = counts
    else:
        s.bank = [B.BANK_PER_RESOURCE] * 5
        for p in players:
            if p.hand_known:
                for r in range(5):
                    s.bank[r] -= p.resources[r]
        s.bank = [max(0, x) for x in s.bank]

    # --- dev deck ---------------------------------------------------------
    dd = parsed.get("dev_deck")
    if isinstance(dd, (dict, list, tuple)) and dd:
        s.dev_deck = [max(0, x) for x in _dev_counts(dd)]
    else:
        deck = list(B.DEV_DECK_COUNTS)
        deck[B.DEV_KNIGHT] -= sum(p.played_knights for p in players)
        deck = [max(0, x) for x in deck]
        unknown_held = 0
        for p in players:
            if p.dev_known:
                for t in range(5):
                    deck[t] = max(0, deck[t] - p.dev_cards[t])
            else:
                unknown_held += p.dev_count
        remaining = parsed.get("dev_deck_remaining")
        if remaining is not None and not isinstance(remaining, bool):
            unknown_held = max(0, sum(deck) - _int(remaining))
        taken = _apportion(unknown_held, deck)
        s.dev_deck = [deck[t] - taken[t] for t in range(5)]

    # --- turn structure ---------------------------------------------------
    cur = parsed.get("current_player")
    cur_idx = -1
    if isinstance(cur, int) and not isinstance(cur, bool) and 0 <= cur < len(players):
        cur_idx = cur
    elif isinstance(cur, str) and cur.strip().lower() in colors:
        cur_idx = colors.index(cur.strip().lower())
    s.current = cur_idx if cur_idx >= 0 else max(0, me)
    s.phase = PHASE_MAIN
    s.dice = _int(parsed.get("dice"), 0)
    if not 0 <= s.dice <= 12:
        s.dice = 0
    s.turn = max(0, _int(parsed.get("turn"), 0))

    # --- awards -----------------------------------------------------------
    s.longest_road_owner = next((i for i, rp in enumerate(raw_players) if rp.get("longest_road")), -1)
    s.largest_army_owner = next((i for i, rp in enumerate(raw_players) if rp.get("largest_army")), -1)
    if s.longest_road_owner >= 0:
        owner = s.longest_road_owner
        blocked = [v for j, p in enumerate(players) if j != owner for v in p.settlements + p.cities]
        s.longest_road_len = max(5, longest_road_length(players[owner].roads, blocked))
    else:
        s.longest_road_len = 0
    return s


# ---------------------------------------------------------------------------
# GameState -> parsed
# ---------------------------------------------------------------------------
def _player_index(state: GameState, me: Union[int, str, None]) -> int:
    if me is None:
        return 0 if state.players else -1
    if isinstance(me, str):
        key = me.strip().lower()
        for i, p in enumerate(state.players):
            if p.color == key:
                return i
        return 0
    return int(me)


def _port_edge_list(ports: Dict[int, int]) -> List[Dict[str, Any]]:
    """Invert a ``{vertex: type}`` port map to ``[{edge, type}]`` on coastal edges."""
    out: List[Dict[str, Any]] = []
    consumed: Set[int] = set()
    for e in B.COASTAL_EDGES:
        a, b = B.EDGE_VERTICES[e]
        if a in consumed or b in consumed:
            continue
        ta, tb = ports.get(a), ports.get(b)
        if ta is not None and ta == tb:
            out.append({"edge": e, "type": B.PORT_NAMES[ta]})
            consumed.update((a, b))
    # ports whose vertices could not be paired on a single coastal edge: attach to any edge
    for v, t in sorted(ports.items()):
        if v in consumed:
            continue
        for e in B.VERTEX_EDGES[v]:
            if e in _COASTAL_EDGE_SET:
                out.append({"edge": e, "type": B.PORT_NAMES[t]})
                consumed.add(v)
                break
    return out


def state_to_parsed(state: GameState, me: Union[int, str, None] = None,
                    include_bank: bool = True) -> Dict[str, Any]:
    """Inverse of :func:`parsed_to_state`: what a screenshot of ``state`` shows.

    ``me`` is a player index or colour (default: player 0).  Only ``me`` gets
    exact ``resources`` (and exact ``dev_cards`` counts by type when known);
    opponents get card counts and public VP.  With ``include_bank`` the bank
    stock and dev-deck size are included (Colonist shows both).
    """
    me_idx = _player_index(state, me)
    players: List[Dict[str, Any]] = []
    for i, p in enumerate(state.players):
        entry: Dict[str, Any] = {
            "color": p.color,
            "name": p.name or p.color,
            "vp": state.total_vp(i) if i == me_idx else state.public_vp(i),
            "cards": sum(p.resources) if p.hand_known else p.hand_size,
            "dev_cards": p.total_dev if p.dev_known else p.dev_count,
            "knights": p.played_knights,
            "longest_road": state.longest_road_owner == i,
            "largest_army": state.largest_army_owner == i,
            "settlements": sorted(p.settlements),
            "cities": sorted(p.cities),
            "roads": sorted(p.roads),
        }
        if i == me_idx:
            if p.hand_known:
                entry["resources"] = {B.RESOURCE_NAMES[r]: p.resources[r] for r in range(5)}
            if p.dev_known:
                entry["dev_cards"] = {B.DEV_NAMES[t]: p.dev_cards[t] + p.dev_cards_new[t] for t in range(5)}
        players.append(entry)
    parsed: Dict[str, Any] = {
        "hexes": [{"resource": B.RESOURCE_NAMES[r], "number": (n if r != B.DESERT and n else None)}
                  for r, n in state.hexes],
        "robber": state.robber,
        "ports": _port_edge_list(state.ports),
        "players": players,
        "me": state.players[me_idx].color if 0 <= me_idx < len(state.players) else None,
        "current_player": state.players[state.current].color if 0 <= state.current < len(state.players) else None,
        "dice": state.dice,
        "turn": state.turn,
    }
    if include_bank:
        parsed["bank"] = {B.RESOURCE_NAMES[r]: state.bank[r] for r in range(5)}
        parsed["dev_deck_remaining"] = sum(state.dev_deck)
    return parsed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _multiset_diff(actual: Counter, expected: Counter) -> Tuple[List[Any], List[Any]]:
    missing = sorted((expected - actual).elements(), key=str)
    extra = sorted((actual - expected).elements(), key=str)
    return missing, extra


def validate(parsed: Dict[str, Any]) -> List[str]:
    """Check a parsed screenshot for inconsistencies; returns warning strings.

    An empty list means the parse looks like a legal standard-board position.
    Checks: hex count, resource / number multisets versus the standard set,
    numbers on the desert (or missing elsewhere), robber and port sanity,
    unknown / duplicate colours, piece ids out of range, pieces stacked on
    the same vertex or edge, the distance rule, roads disconnected from their
    owner's buildings, piece-count limits, VP consistent with pieces and
    awards, award flags versus knights / road length, and that ``me`` /
    ``current_player`` refer to listed players.
    """
    w: List[str] = []
    if not isinstance(parsed, dict):
        return ["parsed screenshot must be a dict"]

    # --- hexes ------------------------------------------------------------
    hexes = parsed.get("hexes")
    if not isinstance(hexes, list):
        w.append("missing 'hexes' list")
        hexes = []
    if len(hexes) != B.NUM_HEXES:
        w.append(f"expected {B.NUM_HEXES} hexes, got {len(hexes)}")
    res_ids: List[Optional[int]] = []
    numbers: List[int] = []
    for i, h in enumerate(hexes):
        if isinstance(h, dict):
            raw_res, raw_num = h.get("resource"), h.get("number")
        elif isinstance(h, (list, tuple)) and h:
            raw_res, raw_num = h[0], (h[1] if len(h) > 1 else None)
        else:
            w.append(f"hex {i}: malformed entry {h!r}")
            res_ids.append(None)
            continue
        r = resource_from_name(raw_res)
        if r is None:
            w.append(f"hex {i}: unknown resource {raw_res!r}")
        res_ids.append(r)
        n = _int(raw_num, 0) if raw_num is not None else 0
        if r == B.DESERT:
            if n:
                w.append(f"hex {i}: desert has a number token ({n})")
        elif r is not None:
            if n == 0:
                w.append(f"hex {i}: {B.RESOURCE_NAMES[r]} hex has no number token")
            elif n < 2 or n > 12 or n == 7:
                w.append(f"hex {i}: impossible number token {n}")
            else:
                numbers.append(n)
    known_res = Counter(r for r in res_ids if r is not None)
    if len(hexes) == B.NUM_HEXES and None not in res_ids:
        missing, extra = _multiset_diff(known_res, Counter(B.STANDARD_RESOURCE_COUNTS))
        if missing or extra:
            w.append("resource multiset differs from standard: missing "
                     f"{[B.RESOURCE_NAMES[r] for r in missing]}, extra {[B.RESOURCE_NAMES[r] for r in extra]}")
    if numbers:
        missing, extra = _multiset_diff(Counter(numbers), _STANDARD_NUMBER_COUNTS)
        if missing or extra:
            w.append(f"number multiset differs from standard: missing {missing}, extra {extra}")

    # --- robber -----------------------------------------------------------
    robber = parsed.get("robber")
    if robber is not None:
        rb = _int(robber, -1)
        if not 0 <= rb < B.NUM_HEXES:
            w.append(f"robber hex {robber!r} out of range")

    # --- ports ------------------------------------------------------------
    ports = parsed.get("ports")
    if ports is not None:
        if not isinstance(ports, list):
            w.append("'ports' must be a list of {edge, type}")
        else:
            seen_port_vertices: Set[int] = set()
            for entry in ports:
                if isinstance(entry, dict):
                    e, t = entry.get("edge"), entry.get("type")
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    e, t = entry
                else:
                    w.append(f"port entry malformed: {entry!r}")
                    continue
                ei = _int(e, -1)
                if ei not in _COASTAL_EDGE_SET:
                    w.append(f"port on edge {e!r} which is not a coastal edge")
                elif any(v in seen_port_vertices for v in B.EDGE_VERTICES[ei]):
                    w.append(f"port on edge {ei} shares a vertex with another port")
                else:
                    seen_port_vertices.update(B.EDGE_VERTICES[ei])
                if port_type_from_name(t) is None:
                    w.append(f"port on edge {e!r}: unknown type {t!r}")
            if len(ports) != 9:
                w.append(f"expected 9 ports, got {len(ports)}")

    # --- players ----------------------------------------------------------
    players = parsed.get("players")
    if not isinstance(players, list) or not players:
        w.append("missing 'players' list")
        players = []
    players = [p for p in players if isinstance(p, dict)]
    colors = [_color_of(p) for p in players]
    for i, c in enumerate(colors):
        if c not in PLAYER_COLORS:
            w.append(f"player {i}: unknown colour {c!r}")
    for c, n in Counter(colors).items():
        if n > 1:
            w.append(f"colour {c!r} used by {n} players")
    if len(players) > 4:
        w.append(f"{len(players)} players (base game has at most 4)")

    vertex_owner: Dict[int, int] = {}
    edge_owner: Dict[int, int] = {}
    per_player: List[Tuple[List[int], List[int], List[int]]] = []
    for i, p in enumerate(players):
        label = colors[i] or f"player {i}"
        s_raw, c_raw, r_raw = p.get("settlements") or [], p.get("cities") or [], p.get("roads") or []
        sett: List[int] = []
        cit: List[int] = []
        roads: List[int] = []
        for kind, raw, maximum, target in (("settlement", s_raw, B.NUM_VERTICES, sett),
                                           ("city", c_raw, B.NUM_VERTICES, cit),
                                           ("road", r_raw, B.NUM_EDGES, roads)):
            for v in raw:
                vi = _int(v, -1)
                if not 0 <= vi < maximum:
                    w.append(f"{label}: {kind} id {v!r} is not a valid "
                             f"{'edge' if kind == 'road' else 'vertex'} (0..{maximum - 1})")
                elif vi in target:
                    w.append(f"{label}: duplicate {kind} at {vi}")
                else:
                    target.append(vi)
        for v in sett + cit:
            if v in vertex_owner:
                other = colors[vertex_owner[v]] if vertex_owner[v] != i else label
                w.append(f"vertex {v}: building of {label} overlaps building of {other}")
            else:
                vertex_owner[v] = i
        for e in roads:
            if e in edge_owner:
                w.append(f"edge {e}: road of {label} overlaps road of {colors[edge_owner[e]]}")
            else:
                edge_owner[e] = i
        if len(sett) > B.MAX_SETTLEMENTS:
            w.append(f"{label}: {len(sett)} settlements (max {B.MAX_SETTLEMENTS})")
        if len(cit) > B.MAX_CITIES:
            w.append(f"{label}: {len(cit)} cities (max {B.MAX_CITIES})")
        if len(roads) > B.MAX_ROADS:
            w.append(f"{label}: {len(roads)} roads (max {B.MAX_ROADS})")
        for key in ("vp", "cards", "dev_cards", "knights"):
            val = p.get(key)
            if val is not None and not isinstance(val, dict) and _int(val, -1) < 0:
                w.append(f"{label}: negative or invalid {key} {val!r}")
        per_player.append((sett, cit, roads))

    # distance rule
    reported: Set[Tuple[int, int]] = set()
    for v, i in vertex_owner.items():
        for nb in B.VERTEX_NEIGHBORS[v]:
            j = vertex_owner.get(nb)
            if j is not None and (min(v, nb), max(v, nb)) not in reported:
                reported.add((min(v, nb), max(v, nb)))
                w.append(f"buildings on adjacent vertices {v} ({colors[i]}) and {nb} ({colors[j]}) "
                         "violate the distance rule")

    # road connectivity and awards / vp
    lr_holders = [i for i, p in enumerate(players) if p.get("longest_road")]
    la_holders = [i for i, p in enumerate(players) if p.get("largest_army")]
    if len(lr_holders) > 1:
        w.append(f"longest road flagged for several players: {[colors[i] for i in lr_holders]}")
    if len(la_holders) > 1:
        w.append(f"largest army flagged for several players: {[colors[i] for i in la_holders]}")
    for i, (sett, cit, roads) in enumerate(per_player):
        label = colors[i] or f"player {i}"
        buildings = set(sett) | set(cit)
        if roads:
            own_roads = set(roads)
            reached: Set[int] = set()
            frontier = list(buildings)
            seen_v: Set[int] = set(frontier)
            while frontier:
                v = frontier.pop()
                for e in B.VERTEX_EDGES[v]:
                    if e in own_roads and e not in reached:
                        reached.add(e)
                        for u in B.EDGE_VERTICES[e]:
                            if u not in seen_v:
                                seen_v.add(u)
                                frontier.append(u)
            loose = sorted(own_roads - reached)
            if loose:
                w.append(f"{label}: roads {loose} are not connected to any of their buildings")
        if buildings and not roads:
            w.append(f"{label}: has buildings but no roads")
        p = players[i]
        knights = _int(p.get("knights"), 0)
        if p.get("largest_army") and knights < 3:
            w.append(f"{label}: largest army flagged with only {knights} knights")
        if p.get("longest_road"):
            blocked = [v for j, (s2, c2, _) in enumerate(per_player) if j != i for v in s2 + c2]
            if longest_road_length(roads, blocked) < 5:
                w.append(f"{label}: longest road flagged but longest road is shorter than 5")
        vp_raw = p.get("vp")
        if vp_raw is not None:
            vp = _int(vp_raw, -1)
            public = len(sett) + 2 * len(cit) + (2 if p.get("longest_road") else 0) + (2 if p.get("largest_army") else 0)
            dev = p.get("dev_cards")
            dev_n = sum(_dev_counts(dev)) if isinstance(dev, (dict, list, tuple)) else _int(dev, 0)
            hidden_max = _dev_counts(dev)[B.DEV_VP] if isinstance(dev, (dict, list, tuple)) else dev_n
            if vp < public:
                w.append(f"{label}: vp {vp} is less than the {public} points visible on the board")
            elif vp > public + hidden_max:
                w.append(f"{label}: vp {vp} exceeds the {public} visible points plus {hidden_max} possible VP cards")

    # me / current player
    for key in ("me", "current_player"):
        val = parsed.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            if val.strip().lower() not in colors:
                w.append(f"'{key}' colour {val!r} is not one of the players {colors}")
        elif isinstance(val, bool) or not isinstance(val, int) or not 0 <= val < len(players):
            w.append(f"'{key}' {val!r} does not identify a player")
    dice = parsed.get("dice")
    if dice is not None and not isinstance(dice, bool):
        d = _int(dice, -1)
        if d not in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            w.append(f"dice value {dice!r} is not a possible 2d6 total")
    return w


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
EXAMPLE_PARSED: Dict[str, Any] = {
    "hexes": [
        {"resource": "ore", "number": 10}, {"resource": "sheep", "number": 2}, {"resource": "wood", "number": 9},
        {"resource": "wheat", "number": 12}, {"resource": "brick", "number": 6}, {"resource": "sheep", "number": 4},
        {"resource": "brick", "number": 10},
        {"resource": "wheat", "number": 9}, {"resource": "wood", "number": 11}, {"resource": "desert", "number": None},
        {"resource": "wood", "number": 3}, {"resource": "ore", "number": 8},
        {"resource": "wood", "number": 8}, {"resource": "ore", "number": 3}, {"resource": "wheat", "number": 4},
        {"resource": "sheep", "number": 5},
        {"resource": "brick", "number": 5}, {"resource": "wheat", "number": 6}, {"resource": "sheep", "number": 11},
    ],
    "robber": 13,
    "ports": [{"edge": e, "type": B.PORT_NAMES[t]} for e, t in B.STANDARD_PORT_EDGES],
    "players": [
        {"color": "red", "name": "Alice", "vp": 6, "cards": 4, "dev_cards": 1, "knights": 1,
         "longest_road": True, "largest_army": False,
         "settlements": [48, 17], "cities": [28], "roads": [34, 24, 18, 10, 6, 0, 63, 19],
         "resources": {"wood": 1, "brick": 1, "sheep": 0, "wheat": 2, "ore": 0}},
        {"color": "blue", "name": "Bob", "vp": 4, "cards": 6, "dev_cards": 2, "knights": 2,
         "longest_road": False, "largest_army": False,
         "settlements": [41, 18], "cities": [31], "roads": [37, 52, 20]},
        {"color": "orange", "name": "Carol", "vp": 3, "cards": 3, "dev_cards": 0, "knights": 0,
         "longest_road": False, "largest_army": False,
         "settlements": [39, 8, 20], "cities": [], "roads": [50, 7, 22]},
        {"color": "green", "name": "Dave", "vp": 5, "cards": 8, "dev_cards": 1, "knights": 3,
         "longest_road": False, "largest_army": True,
         "settlements": [14, 40, 49], "cities": [], "roads": [15, 51, 64]},
    ],
    "me": "red",
    "current_player": "red",
    "dice": 8,
}
