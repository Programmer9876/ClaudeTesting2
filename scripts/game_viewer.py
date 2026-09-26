#!/usr/bin/env python3
"""Turn one logged Catanatron game into a self-contained replay page (HTML).

    /home/user/venv_cat33/bin/python scripts/game_viewer.py proof/T2/logs --game 71 --out game71.html

The game is rebuilt by replaying its action log with scripts/replay_catanatron.py (the same exact replay that
``--check`` uses), so every position on the page is the engine's own state.  The page holds the board, every
action with its outcome, and after each action every player's hand, development cards, victory points and
titles.  Logs of catanatron 3.3 need the 3.3 interpreter, logs of 3.2.1 need 3.2.1.

The page is written without <html>/<head>/<body> tags (a browser adds them; the artifact host wraps it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import replay_catanatron as R  # noqa: E402
from catanbot.bench import catanatron_adapter as AD  # noqa: E402

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEV = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
DEV_NAME = {"KNIGHT": "Knight", "VICTORY_POINT": "Victory Point", "ROAD_BUILDING": "Road Building",
            "YEAR_OF_PLENTY": "Year of Plenty", "MONOPOLY": "Monopoly"}
CORNER_DEG = {"NORTH": -90, "NORTHEAST": -30, "SOUTHEAST": 30, "SOUTH": 90, "SOUTHWEST": 150, "NORTHWEST": 210}
TEMPLATE = os.path.join(HERE, "game_viewer_template.html")


def center(c: Sequence[int]) -> Tuple[float, float]:
    """Pixel centre of a cube-coordinate tile (pointy-top hexes, unit circumradius, y down)."""
    x, _, z = c
    return math.sqrt(3) * (x + z / 2), 1.5 * z


def geometry(game) -> Tuple[Dict[int, List[float]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Node positions, land tiles and ports, from the rebuilt game's map.  Every node's position is computed
    from each tile it touches; they must agree, which checks the coordinate convention."""
    from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
    nodes: Dict[int, List[float]] = {}
    tiles, ports = [], []
    for coord, tile in game.state.board.map.tiles.items():
        cx, cy = center(coord)
        kind = type(tile).__name__
        for ref, nid in (getattr(tile, "nodes", None) or {}).items():
            a = math.radians(CORNER_DEG[ref.value])
            p = [cx + math.cos(a), cy + math.sin(a)]
            if nid in nodes:
                if math.dist(nodes[nid], p) > 1e-6:
                    raise SystemExit(f"node {nid}: positions disagree ({nodes[nid]} vs {p})")
            else:
                nodes[nid] = p
        if kind == "LandTile":
            tiles.append({"c": list(coord), "x": round(cx, 4), "y": round(cy, 4),
                          "res": tile.resource, "num": tile.number})
        elif kind == "Port":
            a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            ports.append({"x": round(cx, 4), "y": round(cy, 4), "res": tile.resource,
                          "nodes": [tile.nodes[a], tile.nodes[b]]})
    return {k: [round(v[0], 4), round(v[1], 4)] for k, v in nodes.items()}, tiles, ports


def res_name(r: Optional[str]) -> str:
    return (r or "any").lower()


def trade_text(v: Sequence[Optional[str]]) -> str:
    give = [x for x in v[:4] if x]
    return f"trades {len(give)} {res_name(give[0])} for 1 {res_name(v[4])} ({len(give)}:1)"


def gains_text(delta: Dict[int, List[int]], colors: Sequence[str]) -> str:
    parts = []
    for seat, d in delta.items():
        got = [f"+{n} {RES[i].lower()}" for i, n in enumerate(d) if n > 0]
        if got:
            parts.append(f"{colors[seat]} {' '.join(got)}")
    return "; ".join(parts)


def player_row(p: Dict[str, object]) -> List[object]:
    return [p["vp"], p["public_vp"], [p["resources"][r] for r in RES], [p["dev_cards"][d] for d in DEV],
            [p["dev_played"][d] for d in DEV], p["longest_road_length"], int(p["has_longest_road"]),
            int(p["has_largest_army"])]


def build(rec: Dict[str, object]) -> Dict[str, object]:
    game = AD.rebuild_game(rec)
    nodes, tiles, ports = geometry(game)
    colors = list(rec["colors"])
    seat_of = {c: i for i, c in enumerate(colors)}
    tile_at = {tuple(t["c"]): t for t in tiles}

    def where(c) -> str:
        t = tile_at.get(tuple(c))
        if not t:
            return "?"
        return "the desert" if t["res"] is None else f'{res_name(t["res"])} {t["num"]}'

    def pieces(s):
        out = set()
        for p in s["players"]:
            out |= {("s", p["seat"], n) for n in p["settlements"]}
            out |= {("c", p["seat"], n) for n in p["cities"]}
            out |= {("r", p["seat"], tuple(sorted(e))) for e in p["roads"]}
        return out

    prev = AD.state_summary(game.state)
    initial = {"P": [player_row(p) for p in prev["players"]], "robber": prev["robber"]}
    steps = []
    rolls = 0
    for i, (color, kind, value, result) in enumerate(rec["actions"]):
        AD.replay_log_action(game, [color, kind, value, result], check=True)
        cur = AD.state_summary(game.state)
        seat = seat_of[color]
        delta = {}
        for a, b in zip(prev["players"], cur["players"]):
            d = [b["resources"][r] - a["resources"][r] for r in RES]
            if any(d):
                delta[a["seat"]] = d
        text = kind.lower().replace("_", " ")
        if kind == "ROLL":
            rolls += 1
            dice = list(result or value)
            total = sum(dice)
            got = gains_text({k: v for k, v in delta.items()}, colors)
            text = f"rolls {total}" + (f" · {got}" if got else (" · the robber moves" if total == 7 else " · nobody produces"))
        elif kind == "BUILD_SETTLEMENT":
            text = "builds a settlement"
            got = gains_text(delta, colors) if rolls == 0 else ""
            if got and any(n > 0 for n in delta.get(seat, [])):
                text += f" · starting cards {got.split(' ', 1)[1]}"
        elif kind == "BUILD_CITY":
            text = "upgrades a settlement to a city"
        elif kind == "BUILD_ROAD":
            text = "builds a road"
        elif kind == "END_TURN":
            text = "ends the turn"
        elif kind == "MARITIME_TRADE":
            text = trade_text(value)
        elif kind == "MOVE_ROBBER":
            coord, victim = value[0], value[1]
            text = f"moves the robber to {where(coord)}"
            if victim:
                text += f" and steals {res_name(result) if result else 'a card'} from {victim}"
        elif kind == "BUY_DEVELOPMENT_CARD":
            card = result or value
            text = f"buys a development card ({DEV_NAME.get(card, card)})"
        elif kind == "PLAY_KNIGHT_CARD":
            text = "plays a Knight"
        elif kind == "PLAY_MONOPOLY":
            took = sum(delta.get(seat, [0] * 5))
            text = f"plays Monopoly on {res_name(value)} and collects {took}"
        elif kind == "PLAY_YEAR_OF_PLENTY":
            text = "plays Year of Plenty: " + " and ".join(res_name(r) for r in value)
        elif kind == "PLAY_ROAD_BUILDING":
            text = "plays Road Building"
        elif kind == "DISCARD_RESOURCE":
            text = f"discards {res_name(value)}"
        before, after = pieces(prev), pieces(cur)
        add = sorted([list(x[:2]) + [list(x[2]) if isinstance(x[2], tuple) else x[2]] for x in after - before],
                     key=str)
        gone = sorted([list(x[:2]) + [x[2]] for x in before - after if x[0] == "s"], key=str)
        step = {"a": seat, "k": kind, "x": text, "r": rolls, "P": [player_row(p) for p in cur["players"]],
                "rob": cur["robber"]}
        if kind == "ROLL":
            step["d"] = list(result or value)
        if delta:
            step["dl"] = {str(k): v for k, v in delta.items()}
        if add:
            step["add"] = add
        if gone:
            step["gone"] = gone
        steps.append(step)
        prev = cur
    fin = rec["final"]
    if [p["vp"] for p in prev["players"]] != list(fin["vps"]):
        raise SystemExit("replay does not end at the logged VPs")
    players = []
    for p in rec["players"]:
        if p.get("kind") == "catanbot":
            players.append({"color": p["color"], "who": "catanbot", "ours": True, "detail": p.get("spec")})
        else:
            cls = str(p.get("class", "")).split(":")[-1]
            players.append({"color": p["color"], "who": cls.replace("Player", "") or p.get("preset"),
                            "ours": False, "detail": p.get("class")})
    return {"meta": {"game": rec["game"], "seed": rec["seed"], "base_seed": rec["base_seed"],
                     "match": rec.get("match"), "catanatron": rec.get("catanatron"),
                     "vps_to_win": rec.get("vps_to_win"), "winner": fin["winner"], "vps": fin["vps"],
                     "turns": fin["turns"], "rolls": rolls},
            "players": players, "nodes": nodes, "tiles": tiles, "ports": ports,
            "initial": initial, "steps": steps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help=".jsonl.gz action logs or directories holding them")
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--label", default="", help="where the game comes from, shown under the title")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    _, rec = R.find_game(R.log_files(args.logs), args.game, args.base_seed)
    data = build(rec)
    data["meta"]["label"] = args.label
    with open(TEMPLATE) as f:
        page = f.read()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace("/*__GAME_DATA__*/null", blob)
    page = page.replace("<title>Catan Game Replay</title>", f"<title>Game {rec['game']} Replay</title>", 1)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"{args.out}: game {rec['game']}, {len(data['steps'])} actions, winner {data['meta']['winner']}, "
          f"{os.path.getsize(args.out) // 1024} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
