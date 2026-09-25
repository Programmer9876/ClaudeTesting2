"""Command line interface: ``python -m catanbot <command>``.

    analyze IMAGE      parse a Colonist.io screenshot and recommend a move
    recommend          recommend a move from a JSON state (GameState or parsed-screenshot format)
    watch              live advisor: capture the screen periodically and print advice
    play               simulate a game between bots
    eval               tournament between bot specs
    train              self-play training of the value net
    render             render a JSON state as a synthetic screenshot
    profiles           show a saved opponent-profile file
    outcome            record the winner of a logged game
    calibrate          score logged win estimates against outcomes / self-play calibration
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import replace as _dc_replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from . import engine as E
from .state import (GameState, PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROLL, PHASE_TRADE_RESPONSE, PLAYER_COLORS,
                    TradeOffer)

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "value_net.npz")

_COASTAL_EDGES = frozenset(B.COASTAL_EDGES)
_TRUE_WORDS = ("1", "true", "yes", "y", "on")
_FALSE_WORDS = ("0", "false", "no", "n", "off")
BOT_SPEC_HELP = ("bot specs: random[:end=0.3], heuristic[:temp=0.3,eps=0.05], "
                 "search[:depth=1,beam=4,expand=8,model=PATH|evaluator=heuristic,eps=0.05,temp=0.3,rolls=11,trades=3]")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class UsageError(Exception):
    """A problem with the user's input; ``main`` prints it as one line and exits with 2."""


def _res_index(name: str) -> int:
    r = B.RESOURCE_ALIASES.get(name.strip().lower())
    if r is None or r == B.DESERT:
        raise UsageError(f"unknown resource '{name}'")
    return r


def _parse_int(text: Any, what: str, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    """``int(text)`` with a readable UsageError ('hex index must be 0..18, got 99')."""
    rng = f" {lo}..{hi}" if lo is not None and hi is not None else (f" >= {lo}" if lo is not None else "")
    try:
        v = int(str(text).strip())
    except (TypeError, ValueError):
        raise UsageError(f"{what} must be a whole number{rng}, got '{text}'")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise UsageError(f"{what} must be{rng}, got {v}")
    return v


def _parse_bool(text: str, what: str) -> bool:
    t = text.strip().lower()
    if t in _TRUE_WORDS:
        return True
    if t in _FALSE_WORDS:
        return False
    raise UsageError(f"{what} must be 1/0, true/false or yes/no, got '{text}'")


def _parse_edge(text: str, what: str) -> int:
    """An edge id ('12') or a vertex pair ('3-7') -> edge id."""
    t = text.strip()
    if "-" in t:
        a_s, b_s = t.split("-", 1)
        a = _parse_int(a_s, f"{what} vertex", 0, B.NUM_VERTICES - 1)
        b = _parse_int(b_s, f"{what} vertex", 0, B.NUM_VERTICES - 1)
        try:
            return B.edge_between(a, b)
        except KeyError:
            raise UsageError(f"vertices {a} and {b} are not joined by an edge (see --ids)")
    return _parse_int(t, f"{what} edge id", 0, B.NUM_EDGES - 1)


def _counts_from_text(text: str) -> List[int]:
    """'wood:2,brick:1' or 'wood 2 brick 1' -> 5 counts."""
    out = [0] * 5
    toks = text.replace(":", " ").replace(",", " ").replace("=", " ").split()
    i = 0
    while i < len(toks):
        name = toks[i]
        n = 1
        if i + 1 < len(toks) and toks[i + 1].isdigit():
            n = int(toks[i + 1])
            i += 1
        out[_res_index(name)] += n
        i += 1
    return out


def _player_colors(parsed: dict) -> List[str]:
    return [str(p.get("color", "")).lower() for p in parsed.get("players", []) if isinstance(p, dict)]


def _player_entry(parsed: dict, color: str, create: bool = False) -> dict:
    """The parsed player with this colour (or name).  Only ``players=`` may create new players."""
    color = color.strip().lower()
    for p in parsed.get("players", []):
        if p.get("color", "").lower() == color or (p.get("name") or "").lower() == color:
            return p
    if not create:
        raise UsageError(f"unknown player '{color}' (players: {_player_colors(parsed)}; "
                         "add missing players first with --fix 'players=red,blue,...')")
    if color not in PLAYER_COLORS:
        raise UsageError(f"unknown player colour '{color}' (known colours: {PLAYER_COLORS})")
    p = {"color": color, "name": color, "vp": 0, "cards": 0, "dev_cards": 0, "knights": 0,
         "longest_road": False, "largest_army": False, "settlements": [], "cities": [], "roads": []}
    parsed.setdefault("players", []).append(p)
    return p


def _resolve_color(state: GameState, color: Any, what: str = "player") -> int:
    """Seat index of a colour or name (case-insensitive); UsageError listing the players otherwise."""
    key = str(color).strip().lower()
    for i, p in enumerate(state.players):
        if p.color.lower() == key or (p.name or "").lower() == key:
            return i
    raise UsageError(f"unknown {what} '{color}'; players in this game: {[p.color for p in state.players]}")


def apply_fix(parsed: dict, fix: str) -> None:
    """Apply one ``--fix`` correction to a parsed-screenshot dict (see docs/USAGE.md).

    Raises :class:`UsageError` for anything that cannot be honoured (unknown player,
    impossible number token, id out of range, ...) instead of producing a broken state.
    """
    text = fix.strip()
    if "=" not in text:
        raise UsageError(f"fix '{fix}' must look like KEY=VALUE (see docs/USAGE.md)")
    key, value = [t.strip() for t in text.split("=", 1)]
    kl = key.lower()
    vl = value.lower()
    if kl.startswith("hex "):
        h = _parse_int(kl.split(None, 1)[1], "hex index", 0, B.NUM_HEXES - 1)
        parts = vl.split()
        if not parts:
            raise UsageError(f"hex {h}: expected 'hex {h}=RESOURCE NUMBER' or 'hex {h}=desert'")
        hexes = parsed.setdefault("hexes", [])
        while len(hexes) < B.NUM_HEXES:
            hexes.append({"resource": "desert", "number": None})
        res = parts[0]
        if res == "desert":
            if len(parts) > 1:
                raise UsageError(f"hex {h}: the desert has no number token")
            hexes[h] = {"resource": "desert", "number": None}
            return
        rname = B.RESOURCE_NAMES[_res_index(res)]
        if len(parts) > 1:
            num = _parse_int(parts[1], f"hex {h} number token", 2, 12)
            if num == 7:
                raise UsageError(f"hex {h}: 7 is not a number token (use 2-6 or 8-12)")
        else:
            old = hexes[h].get("number") if isinstance(hexes[h], dict) else None
            num = int(old) if isinstance(old, int) and not isinstance(old, bool) and old in B.PIPS else 0
            if not num:
                raise UsageError(f"hex {h}: a {rname} hex needs a number token, e.g. 'hex {h}={rname} 8'")
        hexes[h] = {"resource": rname, "number": num}
        return
    if kl == "robber":
        parsed["robber"] = _parse_int(vl, "robber hex", 0, B.NUM_HEXES - 1)
        return
    if kl in ("me", "current", "current_player"):
        if parsed.get("players"):
            vl = _player_entry(parsed, vl)["color"]
        parsed["me" if kl == "me" else "current_player"] = vl
        return
    if kl == "dice":
        d = _parse_int(vl, "dice", 0, 12)
        if d == 1:
            raise UsageError("dice must be 2..12 (or 0 for 'not rolled yet')")
        parsed["dice"] = d
        return
    if kl in ("deck", "dev_deck", "dev_deck_remaining"):
        parsed["dev_deck_remaining"] = _parse_int(vl, "dev deck size", 0, sum(B.DEV_DECK_COUNTS))
        return
    if kl == "players":
        for c in vl.replace(",", " ").split():
            _player_entry(parsed, c, create=True)
        return
    if kl.startswith("port "):
        e = _parse_edge(kl.split(None, 1)[1], "port")
        if e not in _COASTAL_EDGES:
            raise UsageError(f"edge {e} is not a coastal edge, so it cannot hold a port (see --ids for the numbering)")
        ports = [p for p in (parsed.get("ports") or []) if int(p["edge"]) != e]
        if vl not in ("none", "remove", "", "-"):
            ports.append({"edge": e, "type": "3:1" if vl in ("3:1", "generic", "any") else B.RESOURCE_NAMES[_res_index(vl)]})
        parsed["ports"] = sorted(ports, key=lambda p: p["edge"])
        return
    if kl.startswith("bank."):
        rname = B.RESOURCE_NAMES[_res_index(kl[5:])]
        stock = _parse_int(vl, "bank stock", 0, B.BANK_PER_RESOURCE)
        parsed.setdefault("bank", {B.RESOURCE_NAMES[r]: B.BANK_PER_RESOURCE for r in range(5)})
        parsed["bank"][rname] = stock
        return
    if "." in kl:
        color, attr = kl.split(".", 1)
        p = _player_entry(parsed, color)
        if attr in ("hand", "resources"):
            counts = _counts_from_text(value)
            p["resources"] = {B.RESOURCE_NAMES[r]: counts[r] for r in range(5)}
            p["cards"] = sum(counts)
            return
        if attr == "cards":
            p["cards"] = _parse_int(vl, f"{color} card count", 0)
            return
        if attr in ("dev", "dev_cards"):
            p["dev_cards"] = _parse_int(vl, f"{color} dev card count", 0, sum(B.DEV_DECK_COUNTS))
            return
        if attr in ("devs", "dev_types"):
            # exact dev card types, e.g. 'red.devs=knight:1,vp:1,monopoly:1'
            counts: Dict[str, int] = {}
            toks = value.replace(":", " ").replace(",", " ").split()
            i = 0
            names = {"knight": "knight", "knights": "knight", "vp": "victory_point", "victory_point": "victory_point",
                     "rb": "road_building", "road_building": "road_building", "roads": "road_building",
                     "yop": "year_of_plenty", "year_of_plenty": "year_of_plenty", "plenty": "year_of_plenty",
                     "monopoly": "monopoly", "mono": "monopoly"}
            while i < len(toks):
                name = names.get(toks[i].lower())
                if name is None:
                    raise UsageError(f"unknown dev card type '{toks[i]}'")
                n = 1
                if i + 1 < len(toks) and toks[i + 1].isdigit():
                    n = int(toks[i + 1])
                    i += 1
                counts[name] = counts.get(name, 0) + n
                i += 1
            p["dev_cards"] = counts
            return
        if attr == "knights":
            p["knights"] = _parse_int(vl, f"{color} knights", 0, B.DEV_DECK_COUNTS[0])
            return
        if attr == "vp":
            p["vp"] = _parse_int(vl, f"{color} VP", 0, 15)
            return
        if attr in ("lr", "longest_road"):
            p["longest_road"] = _parse_bool(vl, f"{color} longest road flag")
            return
        if attr in ("la", "largest_army"):
            p["largest_army"] = _parse_bool(vl, f"{color} largest army flag")
            return
        if attr in ("settlements", "cities", "roads"):
            is_road = attr == "roads"
            what = f"{color} {attr[:-1]} " + ("edge" if is_road else "vertex") + " id"
            hi = (B.NUM_EDGES if is_road else B.NUM_VERTICES) - 1
            other = None if is_road else ("cities" if attr == "settlements" else "settlements")
            cur = list(p.get(attr, []))
            for tok in value.replace(",", " ").split():
                if tok.startswith("-"):
                    x = _parse_int(tok[1:], what, 0, hi)
                    cur = [c for c in cur if c != x]
                else:
                    x = _parse_int(tok.lstrip("+"), what, 0, hi)
                    if x not in cur:
                        cur.append(x)
                    if other:   # a city replaces the settlement on that vertex (and vice versa)
                        p[other] = [c for c in p.get(other, []) if c != x]
            p[attr] = sorted(cur)
            return
        raise UsageError(f"unknown player attribute '{attr}' in fix '{fix}'")
    raise UsageError(f"unrecognised fix '{fix}'")


def apply_fixes(parsed: dict, fixes: Optional[Sequence[str]]) -> None:
    """Apply every ``--fix`` in order; any problem becomes a UsageError naming the fix."""
    for fx in fixes or []:
        try:
            apply_fix(parsed, fx)
        except UsageError as ex:
            raise UsageError(f"bad --fix '{fx}': {ex}")
        except (ValueError, IndexError, KeyError, TypeError) as ex:
            raise UsageError(f"bad --fix '{fx}': {ex}")


def _load_json(path: str, what: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise UsageError(f"{path} is not a JSON {what}: {ex}")


def load_state_file(path: str) -> Tuple[GameState, Optional[dict]]:
    """Load either a GameState JSON or a parsed-screenshot JSON. Returns (state, parsed_or_None)."""
    d = _load_json(path, "state file")
    if isinstance(d, dict) and isinstance(d.get("parsed"), dict):
        d = d["parsed"]
    if not isinstance(d, dict):
        raise UsageError(f"{path}: expected a JSON object (a GameState or a parsed screenshot), "
                         f"got a JSON {type(d).__name__}")
    if "phase" in d or "version" in d:
        try:
            state = GameState.from_dict(d)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as ex:
            raise UsageError(f"{path}: not a valid GameState JSON ({type(ex).__name__}: {ex})")
        if not state.players:
            raise UsageError(f"{path}: the game state has no players")
        return state, None
    if not isinstance(d.get("hexes"), list) or not isinstance(d.get("players"), list):
        raise UsageError(f"{path}: not a game state - a GameState JSON has 'phase' and 'players', "
                         "a parsed screenshot has 'hexes' and 'players' (see catanbot/vision/schema.py)")
    from .vision.schema import parsed_to_state
    state = parsed_to_state(d)
    if not state.players:
        raise UsageError(f"{path}: the parsed screenshot lists no players")
    return state, d


def load_evaluator(model_path: Optional[str]) -> Tuple[Any, str]:
    """The value net at ``model_path`` (or the default one), else the heuristic evaluator.

    An explicitly given path that does not exist or cannot be loaded is an error; only the
    implicit default silently falls back to the heuristic evaluator (and says so in its name).
    """
    if model_path and model_path.lower() in ("heuristic", "none"):
        from .heuristic import HeuristicEvaluator
        return HeuristicEvaluator(), "heuristic evaluator"
    explicit = bool(model_path)
    path = model_path or DEFAULT_MODEL
    if os.path.exists(path):
        try:
            from .model import ValueNet
            return ValueNet.load(path), f"value net {path}"
        except Exception as ex:
            if explicit:
                raise UsageError(f"could not load value net {path}: {ex} (use --model heuristic for the heuristic evaluator)")
            print(f"could not load value net {path}: {ex}; using the heuristic evaluator", file=sys.stderr)
    elif explicit:
        raise UsageError(f"value net not found: {path} (use --model heuristic for the heuristic evaluator)")
    from .heuristic import HeuristicEvaluator
    return HeuristicEvaluator(), "heuristic evaluator (no trained value net found)"


def _is_profiles_dict(d: Any) -> bool:
    return isinstance(d, dict) and isinstance(d.get("profiles"), dict)


def load_profiles(path: Optional[str], state: GameState):
    from .opponent_model import OpponentModel
    from .politics import PoliticalState
    model = OpponentModel(state)
    politics = PoliticalState(state.num_players)
    if path and os.path.exists(path):
        d = _load_json(path, "profiles file")
        if not _is_profiles_dict(d):
            raise UsageError(f"{path} is not a profiles file (expected a JSON object with a 'profiles' key, "
                             "as written by --profiles FILE); refusing to overwrite it")
        model = OpponentModel.from_dict(d)
        model.attach(state)
        if d.get("politics"):
            politics = PoliticalState.from_named_dict(d["politics"], state)
    return model, politics


def save_profiles(path: str, model, politics, state: GameState) -> None:
    d = model.to_dict()
    d["politics"] = politics.to_named_dict(state)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


def split_bot_specs(text: str) -> List[str]:
    """Split a ``--bots`` list into specs.

    Specs are separated by ',', ';' or whitespace; a comma-separated token that is a
    ``key=value`` pair belongs to the previous spec, so
    ``"search:depth=1,model=x.npz,heuristic,random"`` is three bots.
    """
    specs: List[str] = []
    for chunk in re.split(r"[;\s]+", text.strip()):
        for tok in chunk.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" in tok.split(":", 1)[0] and specs:
                specs[-1] += "," + tok
            else:
                specs.append(tok)
    return specs


def _make_bots(specs: Sequence[str]) -> list:
    from .selfplay import make_bot
    if not specs:
        raise UsageError(f"--bots must name at least one bot ({BOT_SPEC_HELP})")
    bots = []
    for s in specs:
        try:
            bots.append(make_bot(s))
        except (ValueError, KeyError, TypeError, OSError) as ex:
            raise UsageError(f"bad bot spec '{s}': {ex} ({BOT_SPEC_HELP})")
    return bots


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
_RES_ABBR = {B.WOOD: "WO", B.BRICK: "BR", B.SHEEP: "SH", B.WHEAT: "WH", B.ORE: "OR", B.DESERT: "DE"}


def _port_edges(state: GameState) -> List[Tuple[Optional[int], int, Tuple[int, ...]]]:
    """``state.ports`` ({vertex: type}) as ``[(edge or None, type, vertices)]``."""
    out: List[Tuple[Optional[int], int, Tuple[int, ...]]] = []
    used: set = set()
    for e in B.COASTAL_EDGES:
        a, b = B.EDGE_VERTICES[e]
        t = state.ports.get(a)
        if t is not None and state.ports.get(b) == t and a not in used and b not in used:
            out.append((e, t, (a, b)))
            used.update((a, b))
    for v, t in sorted(state.ports.items()):
        if v not in used:
            out.append((None, t, (v,)))
    return out


def board_ascii(state: GameState) -> str:
    lines = []
    for row in B.HEX_ROWS:
        cells = []
        for h in row:
            res, num = state.hexes[h]
            label = f"{h:2d}:{_RES_ABBR[res]}{num if num else '':<2}"
            if h == state.robber:
                label += "*"
            else:
                label += " "
            cells.append(label)
        indent = " " * (3 * (5 - len(row)))
        lines.append(indent + "  ".join(cells))
    lines.append("(* = robber; ids are hex indices)")
    lines.append("Bank: " + ", ".join(f"{state.bank[r]} {B.RESOURCE_NAMES[r]}" for r in range(5))
                 + f"; dev deck {sum(state.dev_deck)} left"
                 + (" (" + ", ".join(f"{state.dev_deck[t]} {B.DEV_NAMES[t]}" for t in range(5) if state.dev_deck[t]) + ")"
                    if sum(state.dev_deck) else ""))
    if state.ports:
        by_type: Dict[str, List[str]] = {}
        for e, t, vs in _port_edges(state):
            by_type.setdefault(B.PORT_NAMES[t], []).append(
                f"edge {e} (vertices {vs[0]}-{vs[1]})" if e is not None else f"vertex {vs[0]}")
        lines.append("Ports: " + "; ".join(f"{t} at " + ", ".join(items) for t, items in sorted(by_type.items())))
        lines.append("(fix ports with --fix 'port EDGE=TYPE' or 'port V1-V2=TYPE'; --ids prints the vertex / edge numbering)")
    return "\n".join(lines)


def ids_ascii(state: Optional[GameState] = None) -> str:
    """Reference table of the fixed vertex (0..53) and edge (0..71) ids, hex by hex."""
    lines = ["Vertex / edge ids per hex (corners from the top clockwise; edge k joins corners k and k+1):",
             " hex        corners (vertex ids)     edges (edge ids)"]
    for h in range(B.NUM_HEXES):
        label = f"{h:2d}"
        if state is not None:
            res, num = state.hexes[h]
            label += f":{_RES_ABBR[res]}{num or ''}"
        lines.append(f" {label:<10} {' '.join(f'{v:2d}' for v in B.HEX_VERTICES[h]):<24} "
                     f"{' '.join(f'{e:2d}' for e in B.HEX_EDGES[h])}")
    lines.append("Coastal edges (where ports sit): " + " ".join(str(e) for e in B.COASTAL_EDGES))
    lines.append("Use these with --fix 'red.settlements=+V', 'red.cities=+V', 'red.roads=+E', 'port E=TYPE' "
                 "(or 'port V1-V2=TYPE'); --debug OUT.png draws them on the screenshot.")
    return "\n".join(lines)


def draw_ids(img, geometry: Optional[Dict[str, float]]):
    """Write vertex ids (v0..v53) and edge ids (e0..e71) onto a debug overlay in place."""
    if not geometry or img is None:
        return img
    from PIL import ImageDraw, ImageFont
    cx, cy, hs = float(geometry["cx"]), float(geometry["cy"]), float(geometry["hex_size"])
    draw = ImageDraw.Draw(img)
    try:
        from .vision.colonist import DEFAULT_FONT_PATH
        font = ImageFont.truetype(DEFAULT_FONT_PATH, max(10, int(0.17 * hs)))
    except Exception:
        font = ImageFont.load_default()

    def put(x: float, y: float, text: str, fill) -> None:
        try:
            draw.rectangle(draw.textbbox((x, y), text, font=font), fill=(0, 0, 0))
        except Exception:
            pass
        draw.text((x, y), text, fill=fill, font=font)

    for v, (px, py) in enumerate(B.VERTEX_POS):
        put(cx + px * hs + 0.06 * hs, cy + py * hs + 0.04 * hs, f"v{v}", (255, 255, 255))
    for e, (px, py) in enumerate(B.EDGE_POS):
        put(cx + px * hs - 0.16 * hs, cy + py * hs - 0.22 * hs, f"e{e}", (120, 255, 120))
    return img


def players_table(state: GameState, me: int) -> str:
    rows = ["  player      VP  cards dev  kn  LR LA  S/C/R   hand"]
    for i, p in enumerate(state.players):
        name = (p.name or p.color)[:10]
        vp = state.total_vp(i) if i == me else state.public_vp(i)
        cards = p.total_resources if p.hand_known else p.hand_size
        dev = p.total_dev if p.dev_known else p.dev_count
        hand = ", ".join(f"{p.resources[r]} {B.RESOURCE_NAMES[r]}" for r in range(5) if p.resources[r]) if p.hand_known else "?"
        mark = "*" if i == me else (">" if i == state.current else " ")
        rows.append(f"{mark} {name:<10} {vp:>3} {cards:>5} {dev:>4} {p.played_knights:>3}   "
                    f"{'x' if state.longest_road_owner == i else '-'}  {'x' if state.largest_army_owner == i else '-'}  "
                    f"{len(p.settlements)}/{len(p.cities)}/{len(p.roads):<3}  {hand}")
    rows.append("(* = you, > = player to move)")
    return "\n".join(rows)


def _section(title: str) -> str:
    return f"\n== {title} ==\n"


# ---------------------------------------------------------------------------
# core: recommend
# ---------------------------------------------------------------------------
def parse_offer(state: GameState, me: int, text: str) -> Tuple[int, List[int], List[int]]:
    """``'blue:give=wood:1;get=ore:1'`` -> (proposer, give, get).

    ``give`` is what the proposer hands over (you receive it), ``get`` what they want
    from you.  Both sides must be non-empty and disjoint, and known hands must cover them.
    """
    fmt = "expected --offer 'COLOR:give=RES:N,...;get=RES:N,...' (give = what they hand you, get = what they want)"
    if ":" not in text:
        raise UsageError(f"bad --offer '{text}': {fmt}")
    color, spec = text.split(":", 1)
    proposer = _resolve_color(state, color, "offer proposer")
    if proposer == me:
        raise UsageError(f"bad --offer '{text}': the proposer must be an opponent, not you ({state.players[me].color})")
    give: Optional[List[int]] = None
    get: Optional[List[int]] = None
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise UsageError(f"bad --offer '{text}': {fmt}")
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if k == "give":
            give = _counts_from_text(v)
        elif k == "get":
            get = _counts_from_text(v)
        else:
            raise UsageError(f"bad --offer '{text}': unknown part '{k}'; {fmt}")
    if give is None or get is None or sum(give) == 0 or sum(get) == 0:
        raise UsageError(f"bad --offer '{text}': both 'give' and 'get' must be non-empty; {fmt}")
    if any(give[r] and get[r] for r in range(5)):
        raise UsageError(f"bad --offer '{text}': the same resource cannot be on both sides")
    return proposer, give, get


def offer_affordability_warnings(state: GameState, me: int, proposer: int, give: Sequence[int],
                                 get: Sequence[int]) -> List[str]:
    """Warnings when a known hand cannot cover its side of an offer (accepting is then illegal)."""
    out: List[str] = []
    mep = state.players[me]
    if mep.hand_known and any(mep.resources[r] < get[r] for r in range(5)):
        out.append(f"you cannot pay {A._counts_str(get)} for this offer (your hand: {A._counts_str(mep.resources)}), "
                   f"so only rejecting is possible; if the hand was misread fix it with --fix '{mep.color}.hand=...'")
    pp = state.players[proposer]
    if pp.hand_known and any(pp.resources[r] < give[r] for r in range(5)):
        out.append(f"{pp.color} does not hold {A._counts_str(give)} (their hand: {A._counts_str(pp.resources)})")
    return out


def run_search(state: GameState, me: int, evaluator, cfg, args, model=None, politics=None, rng=None):
    """Search ``state`` for ``me``.  Hidden hands are averaged over ``args.samples``
    determinizations that share one ``--time`` budget: the remaining time is split over
    the samples still to run and no new sample starts once the budget is spent.

    Returns ``(ranked ScoredActions, info)`` with ``info = {seconds, samples, samples_done,
    time_limit, budget_hit}``.
    """
    from .inference import is_fully_known, sample_states
    from .search import ScoredAction, Searcher
    rng = rng or random.Random(0)
    time_limit = getattr(args, "time", None) or None
    n = max(1, int(getattr(args, "samples", 1) or 1))
    t0 = time.time()
    info: Dict[str, Any] = {"samples": 1, "samples_done": 1, "time_limit": time_limit, "budget_hit": False}
    if state.phase == PHASE_GAME_OVER:
        results: List[Any] = []
    elif is_fully_known(state):
        results = Searcher(evaluator, _dc_replace(cfg, time_limit=time_limit), model, None, politics).search(state, me, rng)
    else:
        states = sample_states(state, me, n, rng)
        agg: Dict[Any, List[float]] = {}
        expl: Dict[Any, str] = {}
        lines: Dict[Any, List[Any]] = {}
        done = 0
        for k, s in enumerate(states):
            cfg_k = cfg
            if time_limit:
                remaining = time_limit - (time.time() - t0)
                if k > 0 and remaining <= 0:
                    break
                cfg_k = _dc_replace(cfg, time_limit=max(0.02, remaining / (len(states) - k)))
            for r in Searcher(evaluator, cfg_k, model, None, politics).search(s, me, rng):
                agg.setdefault(r.action, []).append(r.value)
                expl.setdefault(r.action, r.explanation)
                lines.setdefault(r.action, r.line)
            done += 1
        results = [ScoredAction(a, sum(v) / len(v), expl[a], lines[a], sum(v) / len(v)) for a, v in agg.items()]
        results.sort(key=lambda r: -r.value)
        info["samples"], info["samples_done"] = len(states), done
    info["seconds"] = round(time.time() - t0, 2)
    if time_limit:
        info["budget_hit"] = info["samples_done"] < info["samples"] or info["seconds"] > time_limit + 0.1
    return results, info


def recommend_for_state(state: GameState, me: int, args, parsed: Optional[dict] = None) -> Dict[str, Any]:
    """Run the search + advice modules; returns a JSON-able report dict.

    ``state`` is not modified: the decision context (an incoming offer, "not my turn")
    is set up on a copy, and ``report["state"]`` is the state as given while
    ``report["decision"]`` says which situation was searched.
    """
    from .devcards import dev_card_advice
    from .discard import explain_seven_risk
    from .politics import political_trade_options, runway_advice
    from .robber import best_robber_move, should_play_knight
    from .danger import danger_lines
    from .search import SearchConfig
    from .trading import trade_advice

    report: Dict[str, Any] = {"warnings": [], "notes": []}
    evaluator, ev_name = load_evaluator(getattr(args, "model", None))
    report["evaluator"] = ev_name
    original = state
    state = state.copy()
    model, politics = load_profiles(getattr(args, "profiles", None), state)
    for ev in getattr(args, "event", None) or []:
        err = model.observe_event(state, ev)
        err2 = politics.observe_event(state, ev)
        if err and err2:
            report["warnings"].append(f"event '{ev}': {err}")
    # Which decision are we making?
    offer = getattr(args, "offer", None)
    if state.phase == PHASE_GAME_OVER:
        w = state.winner
        report["notes"].append("The game is over" + (f" ({state.players[w].name or state.players[w].color} won)"
                                                     if 0 <= w < state.num_players else "")
                               + "; there are no actions to recommend.")
    elif offer:
        proposer, give, get = parse_offer(state, me, offer)
        report["warnings"].extend(offer_affordability_warnings(state, me, proposer, give, get))
        state.pending_trade = TradeOffer(proposer, give, get)
        state.current = proposer
        state.phase = PHASE_TRADE_RESPONSE
        state.trade_responder = me
        state.dice = state.dice or 8
        state.discard_queue = []
        report["notes"].append(f"Evaluating the offer from {state.players[proposer].color}: you receive "
                               f"{A._counts_str(give)} and pay {A._counts_str(get)}.")
    elif state.current != me:
        report["notes"].append(f"It is {state.players[state.current].name or state.players[state.current].color}'s turn; "
                               "the recommendation is for your next turn with your current hand.")
        state.current = me
        state.phase = PHASE_ROLL
        state.dice = 0
        state.pending_trade = None
        state.discard_queue = []
    elif state.phase == PHASE_MAIN and not state.dice:
        state.phase = PHASE_ROLL
    if getattr(args, "phase", None) and not offer and state.phase != PHASE_GAME_OVER:
        state.phase = args.phase
        if args.phase == PHASE_MAIN and not state.dice:
            state.dice = 8
    cfg = SearchConfig(depth=args.depth, beam=args.beam, expand=max(6, args.beam + 4),
                       time_limit=getattr(args, "time", None), opp_roll_samples=12, finished_lookahead=0)
    rng = random.Random(getattr(args, "seed", 0) or 0)
    results, sinfo = run_search(state, me, evaluator, cfg, args, model, politics, rng)
    report["search_seconds"] = sinfo["seconds"]
    report["search"] = sinfo
    if sinfo["budget_hit"]:
        if sinfo["samples_done"] < sinfo["samples"]:
            report["notes"].append(f"--time {sinfo['time_limit']}s ran out: {sinfo['samples_done']} of "
                                   f"{sinfo['samples']} determinizations of the hidden hands were searched.")
        else:
            report["notes"].append(f"The search used {sinfo['seconds']}s, over the --time {sinfo['time_limit']}s budget "
                                   "(the deadline is checked between search levels; lower --depth / --beam / --samples).")
    report["actions"] = [r.to_dict(state) for r in results[:8]]
    report["decision"] = {"current": state.players[state.current].color, "phase": state.phase, "dice": state.dice}
    if any(r.action[0] == A.BUY_DEV for r in results[:5]):
        report["notes"].append("Dev cards bought this turn cannot be played until next turn (VP cards count immediately).")
    mep = state.players[me]
    if not mep.dev_known and mep.dev_count > 0:
        report["notes"].append(f"You hold {mep.dev_count} dev card(s) of unknown type; tell me with "
                               f"--fix '{mep.color}.devs=knight:1,vp:1' for exact knight / monopoly advice.")
    # advice sections
    advice: Dict[str, List[str]] = {}
    try:
        advice["trading"] = trade_advice(state, me, model=model)
    except Exception as ex:  # pragma: no cover
        advice["trading"] = [f"(trade advice unavailable: {ex})"]
    advice["seven_risk"] = [explain_seven_risk(state, me, politics)]
    if state.players[me].total_resources > 7 and state.phase == PHASE_MAIN and E.acting_player(state) == me:
        from .discard import choose_discard, surplus_dump_actions
        try:
            legal_now = E.legal_actions(state)
            dumps = surplus_dump_actions(state, me, legal_now)[:4]
            if dumps:
                advice["seven_risk"].append("Ways to get under 8 cards before the next roll: "
                                            + "; ".join(A.describe(a, state) for a in dumps))
            d = choose_discard(state, me)
            advice["seven_risk"].append("If a 7 hits now you would " + A.describe(d, state).lower()
                                        + " (keeping the cards for your next build).")
        except Exception:
            pass
    ok, why = should_play_knight(state, me)
    h, victim, reason = best_robber_move(state, me, target_weights=politics.robber_target_weights(state, me))
    advice["robber"] = [("Play a knight now: " if ok else "Hold the knight: ") + why,
                        f"Best robber target: {A.describe((A.MOVE_ROBBER, h, victim), state)} - {reason}"]
    try:
        advice["robber"].extend("Threat board: " + line for line in danger_lines(state, me))
    except Exception as ex:  # pragma: no cover
        advice["robber"].append(f"(threat board unavailable: {ex})")
    advice["dev_cards"] = dev_card_advice(state, me)
    advice["opponents"] = model.summary(state, me=me)
    # the per-opponent profile lines belong to the Opponents section only
    dup = {"Opponent " + l for l in advice["opponents"]}
    advice["trading"] = [l for l in advice["trading"] if l not in dup]
    pol = runway_advice(state, me, politics)
    try:
        for opt in political_trade_options(state, me, evaluator, politics, model=model)[:2]:
            pol.append("Option: " + A.describe(opt["action"], state) + " - " + opt["reason"])
    except Exception:
        pass
    advice["politics"] = pol
    report["advice"] = advice
    report["me"] = me
    report["state"] = original.to_dict()
    if parsed is not None:
        report["parsed"] = parsed
    log_path = getattr(args, "log", None)
    if log_path:
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "game": getattr(args, "game", None) or "default",
                    "me": state.players[me].color,
                    "turn": state.turn,
                    "vp": [state.public_vp(i) for i in range(state.num_players)],
                    "players": [p.color for p in state.players],
                    "win_prob": report["actions"][0]["value"] if report["actions"] else None,
                    "top_action": report["actions"][0]["text"] if report["actions"] else None,
                    "evaluator": report["evaluator"],
                }, default=float) + "\n")
        except OSError as ex:  # pragma: no cover
            report["warnings"].append(f"could not append to log {log_path}: {ex}")
    profiles = getattr(args, "profiles", None)
    if profiles:
        try:
            save_profiles(profiles, model, politics, state)
        except OSError as ex:  # pragma: no cover
            report["warnings"].append(f"could not save profiles: {ex}")
    return report


def print_report(state: GameState, me: int, report: Dict[str, Any], parse_warnings: Sequence[str] = (),
                 confidence: Optional[Dict[str, float]] = None) -> None:
    print(_section("Board"))
    print(board_ascii(state))
    print()
    print(players_table(state, me))
    if parse_warnings or confidence:
        print(_section("Parse warnings"))
        for w in parse_warnings:
            print(" - " + w)
        if confidence:
            low = [f"{k} {v:.0%}" for k, v in confidence.items() if v < 0.7]
            if low:
                print(" - low confidence: " + ", ".join(low))
        if not parse_warnings:
            print(" (none)")
    for n in report.get("notes", []):
        print("NOTE: " + n)
    for w in report.get("warnings", []):
        print("WARNING: " + w)
    print(_section(f"Recommended actions ({report['evaluator']}, {report['search_seconds']}s)"))
    if not report["actions"]:
        phase = (report.get("decision") or {}).get("phase", state.phase)
        print("  (no legal actions" + (": the game is over" if phase == PHASE_GAME_OVER else f" in phase {phase}") + ")")
    for k, a in enumerate(report["actions"][:5]):
        line = " -> ".join(a["line"][:4])
        print(f"{k + 1}. [{a['value']:.3f}] {a['text']}")
        if a["explanation"]:
            print(f"     why: {a['explanation']}")
        if len(a["line"]) > 1:
            print(f"     line: {line}")
    titles = [("trading", "Trading"), ("seven_risk", "7-protection"), ("robber", "Knight / robber"),
              ("dev_cards", "Development cards"), ("politics", "Politics"), ("opponents", "Opponents")]
    for key, title in titles:
        lines = report["advice"].get(key) or []
        if not lines:
            continue
        print(_section(title))
        for l in lines:
            print(l if l.startswith("  ") else " " + l)


def _emit(state: GameState, me: int, report: Dict[str, Any], args, parse_warnings: Sequence[str] = (),
          confidence: Optional[Dict[str, float]] = None) -> None:
    if args.json:
        print(json.dumps(report, indent=1, default=float))
        return
    print_report(state, me, report, parse_warnings, confidence)
    if getattr(args, "ids", False):
        print(_section("Vertex / edge ids"))
        print(ids_ascii(state))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_analyze(args) -> int:
    parser_kind = args.parser
    if parser_kind == "auto":
        parser_kind = "cv"
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                parser_kind = "llm"
            except ImportError:
                pass
    if parser_kind == "llm":
        try:
            from .vision.llm import parse_with_claude
            result = parse_with_claude(args.image, me=args.me)
        except RuntimeError as ex:
            print(f"LLM parser unavailable: {ex}\nFalling back to the computer-vision parser.", file=sys.stderr)
            parser_kind = "cv"
    if parser_kind == "cv":
        try:
            from .vision.colonist import parse_image
        except ImportError as ex:
            print(f"computer-vision parser needs opencv-python-headless / pillow: {ex}", file=sys.stderr)
            return 2
        try:
            result = parse_image(args.image, me=args.me, read_ui=not args.no_ui)
        except FileNotFoundError:
            raise
        except (ValueError, IndexError, OSError) as ex:
            raise UsageError(f"could not parse {args.image}: {ex}. Screenshot the whole game window (board with "
                             "number tokens, player panel, hand bar), try --parser llm, or enter the position "
                             "by hand: recommend --state STATE.json with --fix corrections.")
    parsed = result.parsed
    apply_fixes(parsed, args.fix)
    from .vision.schema import parsed_to_state, validate
    warnings = list(result.warnings)
    if args.fix:
        warnings = [w for w in warnings if not w.startswith("expected")] + validate(parsed)
    if args.no_ui:
        warnings.append("--no-ui: the player panel and hand bar were not read, so seat order, hands, VP, dev cards, "
                        "dice and whose turn it is are guesses; supply them with --fix 'players=red,blue,...', "
                        "--fix 'red.hand=wood:2,ore:1', --fix 'current=blue', --fix 'dice=8'")
    state = parsed_to_state(parsed)
    if not state.players:
        raise UsageError("no players were detected in the screenshot; add them with --fix 'players=red,blue,...' "
                         "and their pieces with --fix 'red.settlements=+V' etc. (see --ids), or use --parser llm")
    if parsed.get("dice"):
        state.phase = PHASE_MAIN
        state.dice = int(parsed["dice"])
    me = _resolve_color(state, args.me or parsed.get("me") or state.players[0].color, "player (--me)")
    if args.debug:
        try:
            from .vision.colonist import draw_debug
            img = draw_debug(args.image, result)
            draw_ids(img, (result.debug or {}).get("geometry"))
            img.save(args.debug)
            print(f"debug overlay written to {args.debug}", file=sys.stderr)
        except Exception as ex:  # pragma: no cover
            print(f"could not write debug overlay: {ex}", file=sys.stderr)
    if args.save_state:
        with open(args.save_state, "w") as f:
            json.dump({"parsed": parsed, "state": state.to_dict(), "warnings": warnings}, f, indent=1)
    report = recommend_for_state(state, me, args, parsed)
    report["parse_warnings"] = warnings
    report["confidence"] = {k: float(v) for k, v in result.confidence.items()}
    report["parser"] = parser_kind
    if not args.json:
        print(f"parser: {parser_kind}")
    _emit(state, me, report, args, warnings, result.confidence)
    return 0


def cmd_recommend(args) -> int:
    state, parsed = load_state_file(args.state)
    if args.fix:
        if parsed is None:
            raise UsageError("--fix only applies to parsed-screenshot JSON (the format analyze --save-state writes); "
                             f"{args.state} is a full GameState, so edit that JSON directly")
        apply_fixes(parsed, args.fix)
        from .vision.schema import parsed_to_state
        state = parsed_to_state(parsed)
        if not state.players:
            raise UsageError("the state has no players after the fixes")
        if parsed.get("dice"):
            state.phase = PHASE_MAIN
            state.dice = int(parsed["dice"])
    if args.me:
        me = _resolve_color(state, args.me, "player (--me)")
    elif parsed is not None and parsed.get("me"):
        me = _resolve_color(state, parsed["me"], "player ('me' in the state file)")
    else:
        me = E.acting_player(state) if state.phase != PHASE_GAME_OVER else 0
    report = recommend_for_state(state, me, args, parsed)
    _emit(state, me, report, args)
    return 0


def cmd_play(args) -> int:
    from .selfplay import play_game
    specs = split_bot_specs(args.bots)
    n = args.players or min(4, max(3, len(specs)))
    if len(specs) > n:
        raise UsageError(f"{len(specs)} bot specs given but the game has {n} players (3-4)")
    while len(specs) < n:
        specs.append(specs[-1] if specs else "heuristic")
    _make_bots(specs)   # validate every spec before the first game
    rng = random.Random(args.seed)
    for g in range(args.games):
        bots = _make_bots(specs)

        def on_action(state, action, player):
            if args.verbose:
                p = state.players[player]
                text = A.describe(action, state)
                expl = bots[player].explain(state) if hasattr(bots[player], "explain") else None
                print(f"t{state.turn:3d} {p.color:<7} {text}" + (f"   [{expl}]" if expl and action[0] != A.ROLL else ""))

        r = play_game(bots, rng=random.Random(rng.random()), max_turns=args.max_turns, specs=specs, on_action=on_action)
        winner = r.specs[r.winner] if r.winner >= 0 else "nobody"
        print(f"game {g + 1}: winner seat {r.winner} ({winner}) after {r.turns} turns, VP {r.vps}, "
              f"{r.actions} actions, {r.duration:.1f}s")
    return 0


def cmd_eval(args) -> int:
    from .selfplay import tournament
    specs = split_bot_specs(args.bots)
    _make_bots(specs)   # validate the specs here rather than inside a worker process
    res = tournament(specs, games=args.games, workers=args.workers, seed=args.seed, num_players=args.players,
                     max_turns=args.max_turns)
    print(f"{args.games} games, avg {res['avg_turns']:.0f} turns")
    print(f"{'bot':<50} games  win%   avgVP")
    for s, st in res["summary"].items():
        print(f"{s:<50} {st['games']:>5}  {st['win_rate'] * 100:5.1f}  {st['avg_vp']:5.2f}")
    return 0


def cmd_render(args) -> int:
    from .vision.synth import render_to_file
    state, parsed = load_state_file(args.state)
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", args.size or "")
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise UsageError(f"--size must be WIDTHxHEIGHT in pixels, e.g. 1280x800 (got '{args.size}')")
    w, h = int(m.group(1)), int(m.group(2))
    me = _resolve_color(state, args.me, "player (--me)") if args.me else 0
    render_to_file(state, args.out, size=(w, h), seed=args.seed, me=me, jitter=False)
    print(f"wrote {args.out}")
    return 0


def cmd_profiles(args) -> int:
    from .opponent_model import OpponentModel
    d = _load_json(args.file, "profiles file")
    if not _is_profiles_dict(d):
        raise UsageError(f"{args.file} is not a profiles file (expected a JSON object with a 'profiles' key, "
                         "as written by --profiles FILE)")
    model = OpponentModel.from_dict(d)
    if not model.profiles:
        print("no opponent profiles recorded yet")
    for name, prof in model.profiles.items():
        print(f"{name}: {prof.style_summary()} (observations {prof.observations})")
    pol = d.get("politics")
    if pol and pol.get("colors"):
        cols = pol["colors"]
        print("political capital (row = how the column player views the row player):")
        print("           " + " ".join(f"{c[:6]:>6}" for c in cols))
        for i, c in enumerate(cols):
            print(f"{c[:10]:<10} " + " ".join(f"{pol['capital'][i][j]:6.2f}" for j in range(len(cols))))
        events = pol.get("events") or []
        if events:
            print("recent political events: " + "; ".join(str(e) for e in events[-5:]))
        try:
            from .politics import PoliticalState
            from .state import new_game
            placeholder = new_game(len(cols), colors=list(cols))
            ps = PoliticalState.from_named_dict(pol, placeholder)
            for line in ps.coalitions.summary(placeholder, me=-1):
                print(line)
        except Exception as ex:  # pragma: no cover
            print(f"(coalition summary unavailable: {ex})")
    return 0


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as ex:
                raise UsageError(f"{path} line {k} is not JSON: {ex}")
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _reliability(pairs: List[Tuple[float, float]]) -> List[str]:
    """Brier score and reliability bins for (predicted win prob, outcome) pairs."""
    if not pairs:
        return ["no predictions with outcomes yet"]
    n = len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    base = sum(y for _, y in pairs) / n
    brier_ref = base * (1 - base)
    lines = [f"{n} predictions, Brier {brier:.3f} (always-base-rate {brier_ref:.3f}; lower is better)"]
    lines.append("  predicted   actual   n")
    edges = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        grp = [(p, y) for p, y in pairs if lo <= p < hi]
        if grp:
            lines.append(f"  {sum(p for p, _ in grp) / len(grp):9.2f} {sum(y for _, y in grp) / len(grp):8.2f} {len(grp):3d}")
    return lines


def cmd_outcome(args) -> int:
    """Record who won a logged game: catanbot outcome LOG --game ID --winner COLOR."""
    rows = _read_jsonl(args.log)
    recs = [r for r in rows if r.get("game") == args.game and "win_prob" in r]
    if not recs:
        games = sorted({str(r.get("game")) for r in rows if "win_prob" in r and r.get("game") is not None})
        raise UsageError(f"no logged positions for game '{args.game}' in {args.log}"
                         + (f" (logged games: {games})" if games else " (log positions first with analyze/recommend --log)"))
    winner = args.winner.strip().lower()
    players = [str(c).lower() for c in (recs[-1].get("players") or [])]
    if players and winner not in players:
        raise UsageError(f"'{args.winner}' is not a player of game {args.game} (players: {players})")
    with open(args.log, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "game": args.game, "outcome": winner}) + "\n")
    print(f"recorded: game {args.game} won by {winner}")
    return 0


def cmd_calibrate(args) -> int:
    """Score logged win estimates against recorded outcomes, or run a self-play calibration check."""
    if not args.log and not args.selfplay:
        raise UsageError("calibrate needs --log FILE (a JSONL written by analyze/recommend --log) and/or --selfplay N")
    pairs: List[Tuple[float, float]] = []
    if args.log:
        rows = _read_jsonl(args.log)
        outcomes = {r["game"]: r["outcome"] for r in rows if "outcome" in r}
        by_game: Dict[str, List[dict]] = {}
        for r in rows:
            if "win_prob" in r and r.get("win_prob") is not None:
                by_game.setdefault(r["game"], []).append(r)
        if not by_game:
            print(f"no logged positions in {args.log}")
        for g, recs in by_game.items():
            if g not in outcomes:
                print(f"game {g}: {len(recs)} positions, no outcome recorded yet (catanbot outcome {args.log} --game {g} --winner COLOR)")
                continue
            won = 1.0 if outcomes[g] == recs[0]["me"] else 0.0
            traj = " -> ".join(f"{r['win_prob']:.2f}" for r in recs)
            print(f"game {g}: {'won' if won else 'lost'}; win estimate trajectory {traj}")
            pairs.extend((float(r["win_prob"]), won) for r in recs)
        print("\n".join(_reliability(pairs)))
    if args.selfplay:
        from .selfplay import make_bot, play_game
        evaluator, ev_name = load_evaluator(args.model)
        print(f"self-play calibration of {ev_name}: {args.selfplay} games")
        sp: List[Tuple[float, float]] = []
        rng = random.Random(args.seed)
        for g in range(args.selfplay):
            bots = [make_bot(spec) for spec in ["heuristic:temp=0.3", "heuristic", "heuristic:temp=0.5", "heuristic:temp=0.15"][:4]]
            preds: List[Tuple[int, float]] = []

            def on_action(state, action, player, _preds=preds):
                if state.turn % 4 == 0 and action[0] != A.ROLL:
                    vals = evaluator.evaluate([state] * state.num_players, list(range(state.num_players)))
                    for i, v in enumerate(vals):
                        _preds.append((i, float(v)))

            r = play_game(bots, rng=random.Random(rng.random()), max_turns=300, on_action=on_action)
            if r.winner >= 0:
                sp.extend((v, 1.0 if i == r.winner else 0.0) for i, v in preds)
        print("\n".join(_reliability(sp)))
    return 0


def cmd_watch(args) -> int:
    """Live advisor: capture the screen (or read images from a directory) every few seconds,
    re-analyse when the position changed, print a compact recommendation.  You still make
    every move yourself - this only automates the screenshot + analysis loop."""
    import hashlib
    from .vision.colonist import parse_image
    from .vision.schema import parsed_to_state
    evaluator, ev_name = load_evaluator(args.model)
    print(f"watch mode ({ev_name}); every {args.interval}s; Ctrl-C to stop", flush=True)
    frames: List[Any] = []
    if args.from_dir:
        frames = sorted(os.path.join(args.from_dir, f) for f in os.listdir(args.from_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if not frames:
            print("no images in --from-dir")
            return 2
    else:
        try:
            import mss  # noqa: F401
        except ImportError:
            print("screen capture needs the 'mss' package: pip install mss   (or use --from-dir)")
            return 2
    last_sig = None
    seen = 0
    model, politics = load_profiles(args.profiles, GameState())  # placeholders until a state exists
    while True:
        if args.from_dir:
            if seen >= len(frames):
                break
            img = frames[seen]
            seen += 1
        else:
            import mss
            from PIL import Image
            with mss.mss() as sct:
                mon = sct.monitors[args.monitor] if args.monitor < len(sct.monitors) else sct.monitors[0]
                if args.region:
                    x, y, w, h = [int(v) for v in args.region.split(",")]
                    mon = {"left": x, "top": y, "width": w, "height": h}
                shot = sct.grab(mon)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        try:
            result = parse_image(img, me=args.me)
        except Exception as ex:
            print(f"[{time.strftime('%H:%M:%S')}] parse failed: {ex}")
            if not args.from_dir:
                time.sleep(args.interval)
            continue
        parsed = result.parsed
        try:
            apply_fixes(parsed, args.fix)
        except UsageError as ex:
            print(f"[{time.strftime('%H:%M:%S')}] {ex}")
        sig = hashlib.md5(json.dumps({k: parsed.get(k) for k in ("hexes", "robber", "players", "dice", "current_player")},
                                     sort_keys=True, default=str).encode()).hexdigest()
        if sig == last_sig:
            if not args.from_dir:
                time.sleep(args.interval)
            continue
        last_sig = sig
        state = parsed_to_state(parsed)
        if parsed.get("dice"):
            state.phase = PHASE_MAIN
            state.dice = int(parsed["dice"])
        try:
            if not state.players:
                raise UsageError("no players detected")
            me = _resolve_color(state, args.me or parsed.get("me") or state.players[0].color, "player (--me)")
            report = recommend_for_state(state, me, args, parsed)
        except UsageError as ex:
            print(f"[{time.strftime('%H:%M:%S')}] {ex}")
            if not args.from_dir:
                time.sleep(args.interval)
            continue
        top = report["actions"][:3]
        cur = state.players[state.current].name or state.players[state.current].color
        print(f"[{time.strftime('%H:%M:%S')}] turn of {cur}; you {state.total_vp(me)} VP; hand "
              + ", ".join(f"{state.players[me].resources[r]} {B.RESOURCE_NAMES[r][:2]}" for r in range(5) if state.players[me].resources[r]))
        for k, a in enumerate(top):
            print(f"   {k + 1}. [{a['value']:.2f}] {a['text']}" + (f" - {a['explanation'][:90]}" if a['explanation'] else ""))
        for n in report.get("notes", [])[:1]:
            print("   note: " + n)
        if result.warnings:
            print("   parse: " + "; ".join(result.warnings[:2]))
        if not args.from_dir:
            time.sleep(args.interval)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def _int_range(lo: int, hi: Optional[int] = None):
    """argparse type: a whole number in ``lo..hi``."""
    def conv(text: str) -> int:
        try:
            v = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got '{text}'")
        if v < lo or (hi is not None and v > hi):
            raise argparse.ArgumentTypeError(f"must be {lo}..{hi}" if hi is not None else f"must be >= {lo}")
        return v
    return conv


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of seconds, got '{text}'")
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0 seconds")
    return v


def _add_recommend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--me", help="your colour (default: from the screenshot / state)")
    p.add_argument("--depth", type=_int_range(1, 3), default=1, help="search depth in turns (1-3)")
    p.add_argument("--beam", type=_int_range(1), default=6, help="candidate sequences kept per search level (>= 1)")
    p.add_argument("--samples", type=_int_range(1), default=4, help="determinizations of hidden hands (>= 1)")
    p.add_argument("--model", help="value net path, or 'heuristic' (default: models/value_net.npz if present, else heuristic)")
    p.add_argument("--profiles", help="JSON file with opponent profiles + political capital (read and updated)")
    p.add_argument("--event", action="append", help="observed event, e.g. 'blue accepted give ore get wood' or 'blue robbed red'")
    p.add_argument("--fix", action="append", help="correct the parse, e.g. 'hex 4=wheat 8', 'red.hand=wood:2,ore:1'")
    p.add_argument("--offer", help="evaluate an incoming offer: 'blue:give=wood:1;get=ore:1' (blue gives you wood for your ore)")
    p.add_argument("--phase", choices=[PHASE_ROLL, PHASE_MAIN], help="force the decision phase")
    p.add_argument("--time", type=_positive_float, help="search time budget in seconds (shared by all --samples)")
    p.add_argument("--seed", type=int, default=0, help="random seed for the hidden-hand samples")
    p.add_argument("--json", action="store_true", help="machine readable output")
    p.add_argument("--ids", action="store_true", help="also print the vertex / edge id numbering (for --fix pieces and ports)")
    p.add_argument("--log", help="append this position's win estimate to a JSONL log (for `calibrate`)")
    p.add_argument("--game", help="game id used with --log / outcome")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catanbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="parse a Colonist.io screenshot and recommend a move")
    a.add_argument("image")
    a.add_argument("--parser", choices=["auto", "cv", "llm"], default="auto")
    a.add_argument("--save-state", help="write the parsed state JSON here")
    a.add_argument("--debug", help="write a debug overlay PNG here (detected elements + vertex / edge ids)")
    a.add_argument("--no-ui", action="store_true",
                   help="skip reading the player panel / hand bar (hands, VP, seat order and dice then need --fix)")
    _add_recommend_args(a)
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("recommend", help="recommend a move from a JSON state")
    r.add_argument("--state", required=True, help="GameState JSON or parsed-screenshot JSON (analyze --save-state)")
    _add_recommend_args(r)
    r.set_defaults(func=cmd_recommend)

    g = sub.add_parser("play", help="simulate a game")
    g.add_argument("--players", type=_int_range(3, 4), default=0, help="3 or 4 (default: number of bot specs)")
    g.add_argument("--bots", default="search,heuristic,heuristic,heuristic", help=BOT_SPEC_HELP)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-turns", type=_int_range(1), default=400)
    g.add_argument("--games", type=_int_range(1), default=1)
    g.add_argument("--verbose", action="store_true")
    g.set_defaults(func=cmd_play)

    ev = sub.add_parser("eval", help="tournament between bot specs")
    ev.add_argument("--bots", default="search,heuristic,random", help=BOT_SPEC_HELP)
    ev.add_argument("--games", type=_int_range(1), default=20)
    ev.add_argument("--workers", type=_int_range(1), default=max(1, (os.cpu_count() or 2) - 1))
    ev.add_argument("--players", type=_int_range(3, 4), default=4)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--max-turns", type=_int_range(1), default=400)
    ev.set_defaults(func=cmd_eval)

    from .train import build_parser as build_train_parser, train as train_fn
    t = build_train_parser(sub)
    t.set_defaults(func=lambda args: (train_fn(args), 0)[1])

    rd = sub.add_parser("render", help="render a JSON state as a synthetic screenshot")
    rd.add_argument("state")
    rd.add_argument("out")
    rd.add_argument("--size", default="1280x800", help="WIDTHxHEIGHT in pixels")
    rd.add_argument("--seed", type=int, default=0)
    rd.add_argument("--me", help="colour whose hand is drawn (default: seat 0)")
    rd.set_defaults(func=cmd_render)

    pr = sub.add_parser("profiles", help="show a saved opponent profile file")
    pr.add_argument("file")
    pr.set_defaults(func=cmd_profiles)

    wt = sub.add_parser("watch", help="live advisor: capture the screen periodically and print advice")
    wt.add_argument("--interval", type=float, default=6.0, help="seconds between captures")
    wt.add_argument("--monitor", type=int, default=1, help="mss monitor index (0 = all)")
    wt.add_argument("--region", help="capture region x,y,w,h in pixels")
    wt.add_argument("--from-dir", help="read screenshots from a directory instead of the screen (testing)")
    _add_recommend_args(wt)
    wt.set_defaults(func=cmd_watch)

    oc = sub.add_parser("outcome", help="record the winner of a logged game")
    oc.add_argument("log")
    oc.add_argument("--game", required=True)
    oc.add_argument("--winner", required=True, help="colour of the winner")
    oc.set_defaults(func=cmd_outcome)

    cb = sub.add_parser("calibrate", help="score logged win estimates against outcomes / self-play calibration")
    cb.add_argument("--log", help="JSONL written by analyze/recommend --log")
    cb.add_argument("--selfplay", type=int, default=0, help="also run N self-play games and score the evaluator")
    cb.add_argument("--model", help="value net path or 'heuristic' (for --selfplay)")
    cb.add_argument("--seed", type=int, default=0)
    cb.set_defaults(func=cmd_calibrate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except UsageError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    except FileNotFoundError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    except OSError as ex:   # unreadable / directory / permission problems with a user-given path
        print(f"error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
