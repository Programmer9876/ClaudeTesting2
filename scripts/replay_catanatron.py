#!/usr/bin/env python3
"""Rebuild and inspect games from ``scripts/bench_catanatron.py --log-actions`` logs.

Usage::

    python3 scripts/replay_catanatron.py LOG [LOG ...] --list
    python3 scripts/replay_catanatron.py LOG --game N [--turn T | --action K] [--last 12] [--json]
    python3 scripts/replay_catanatron.py LOG [LOG ...] --check

``LOG`` is a ``.jsonl.gz`` action log or a directory searched recursively for
them.  Each line of a log is one game (format ``catanbot-actionlog/1``, see
``catanbot.bench.catanatron_adapter.game_log``): seeds, engine version, the
players in turn order (catanbot spec / opponent class and parameters), the
board (every tile of the BASE topology with resource and number or port
resource), the robber's start, the shuffled development deck, every action
with the acting colour and its chance outcome (dice, stolen card, drawn
development card, discarded cards) and the final state.

A game is rebuilt by *replaying the log*, not by re-running the players: a
fresh catanatron game is built on the logged board with the logged seating
and deck order, and every logged action is applied with its logged outcome
(catanatron 3.3: ``ActionRecord`` results; 3.2.1: the fully specified logged
action, which carries the dice / card / steal / random discard in its value).
Before an action is applied it must be playable for the colour to move, and
afterwards the engine's record must equal the logged item, so a replay is
exact or fails loudly.  Replays need the catanatron generation that played
the game (3.3 logs with the 3.3 interpreter, 3.2.1 logs with 3.2.1).

* ``--game N [--turn T | --action K]`` prints the board, then the state after
  ``K`` actions or when catanatron's turn counter (``num_turns``, which the
  engine also advances during the setup placements) first reaches ``T``
  (default: the final state): each player's hand, development cards held and
  played, buildings, roads, victory points (actual / public), titles, the
  bank, and the last ``--last`` actions with their outcomes.
* ``--check`` re-plays every game of every log and verifies that each action
  re-applies exactly and that the final victory points, winner, turn count
  and full-state fingerprint (``state_fingerprint``: every player field,
  buildings, roads, robber, bank, deck order, prompt) match the log; it
  reports the compressed log size per game.  Exit code 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanbot.bench import catanatron_adapter as AD  # noqa: E402


# ---------------------------------------------------------------------------
# Reading logs
# ---------------------------------------------------------------------------
def log_files(paths: Sequence[str]) -> List[str]:
    """The ``.jsonl.gz`` files named by ``paths`` (directories searched recursively), sorted."""
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files if f.endswith(".jsonl.gz"))
        elif os.path.exists(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return sorted(set(out))


def read_records(path: str) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """All complete game records of one log and a note if its tail is truncated (killed run)."""
    recs: List[Dict[str, object]] = []
    note = None
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    note = "truncated last line (interrupted run)"
    except (EOFError, zlib.error, OSError) as ex:
        note = f"truncated gzip tail (interrupted run): {ex}"
    return recs, note


def find_game(files: Sequence[str], game: int, seed: Optional[int] = None) -> Tuple[str, Dict[str, object]]:
    hits = []
    for f in files:
        recs, _ = read_records(f)
        for r in recs:
            if int(r.get("game", -1)) == game and (seed is None or int(r.get("base_seed", -1)) == seed):
                hits.append((f, r))
    if not hits:
        raise SystemExit(f"game {game} is not in {', '.join(files)}")
    if len(hits) > 1:
        where = "; ".join(f"{f} (base seed {r.get('base_seed')})" for f, r in hits)
        raise SystemExit(f"game {game} is in several logs: {where} - pass one file (or --base-seed)")
    return hits[0]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay(rec: Dict[str, object], upto_action: Optional[int] = None, upto_turn: Optional[int] = None,
           check: bool = True):
    """Rebuild ``rec``'s game and apply its logged actions (all, the first ``upto_action``, or until
    ``num_turns`` reaches ``upto_turn``).  Returns ``(game, k, turns)`` with ``k`` the number of
    actions applied and ``turns[i]`` the turn counter when action ``i`` was taken."""
    game = AD.rebuild_game(rec)
    acts = rec["actions"]
    turns: List[int] = []
    k = 0
    while k < len(acts):
        if upto_action is not None and k >= upto_action:
            break
        if upto_turn is not None and game.state.num_turns >= upto_turn:
            break
        turns.append(int(game.state.num_turns))
        AD.replay_log_action(game, acts[k], check=check)
        k += 1
    return game, k, turns


def check_record(rec: Dict[str, object]) -> Tuple[bool, str]:
    """Re-play one game completely; ``(ok, message)``."""
    try:
        game, k, _ = replay(rec)
    except (AD.ReplayMismatch, ValueError, KeyError) as ex:
        return False, f"replay failed: {ex}"
    final = rec.get("final") or {}
    st = game.state
    winner = game.winning_color()
    got = {
        "winner": winner.value if winner is not None else None,
        "vps": [int(st.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"]) for i in range(len(st.colors))],
        "turns": int(st.num_turns),
        "num_actions": k,
        "fingerprint": AD.state_fingerprint(st),
    }
    bad = [f"{key}: logged {final.get(key)!r}, replayed {val!r}" for key, val in got.items() if final.get(key) != val]
    if rec.get("crashed"):
        return (not bad), ("crashed game: partial log re-plays" if not bad else "crashed game: " + "; ".join(bad))
    if bad:
        return False, "; ".join(bad)
    return True, f"winner {got['winner']}, VPs {got['vps']}, {got['turns']} turns, {k} actions"


def check_files(files: Sequence[str], verbose: bool = False) -> int:
    """``--check``: re-play every game of every file; prints a line per file; returns the mismatch count."""
    total = bad = 0
    total_bytes = 0
    for f in files:
        recs, note = read_records(f)
        size = os.path.getsize(f)
        total_bytes += size
        fails = []
        for rec in recs:
            ok, msg = check_record(rec)
            if verbose or not ok:
                print(f"  game {rec.get('game')}: {'OK ' if ok else 'MISMATCH'} {msg}")
            if not ok:
                fails.append(rec.get("game"))
        total += len(recs)
        bad += len(fails)
        per = size / max(1, len(recs))
        status = "all OK" if not fails else f"{len(fails)} MISMATCH (games {fails})"
        print(f"{f}: {len(recs)} games, {status}; {size / 1024:.1f} KiB gzip = {per / 1024:.2f} KiB/game"
              + (f"; {note}" if note else ""))
    if len(files) > 1:
        print(f"total: {total} games in {len(files)} logs, {bad} mismatches, "
              f"{total_bytes / max(1, total) / 1024:.2f} KiB/game gzip")
    return bad


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    return json.dumps(v, separators=(",", ":")) if v is not None else "-"


def identity(p: Dict[str, object]) -> str:
    if p.get("kind") == "catanbot":
        return f'catanbot "{p.get("spec")}" (bot seed {p.get("bot_seed")})'
    params = p.get("params") or {}
    ptxt = ",".join(f"{k}={v}" for k, v in params.items()) or "defaults"
    return f'{p.get("preset")} = {p.get("class")} ({ptxt})'


def print_board(rec: Dict[str, object], robber: Sequence[int]) -> None:
    board = rec["board"]
    land = [t for t in board["tiles"] if t["t"] == "land"]
    print("board (land tiles by row z, cube coordinates (x,y,z); * = robber now):")
    for z in sorted({t["c"][2] for t in land}):
        row = sorted((t for t in land if t["c"][2] == z), key=lambda t: t["c"][0])
        cells = []
        for t in row:
            what = "DESERT" if t["resource"] is None else f'{t["resource"]} {t["number"]}'
            mark = "*" if list(t["c"]) == list(robber) else " "
            cells.append(f'{mark}({t["c"][0]},{t["c"][1]},{t["c"][2]}) {what}')
        print("   " + " | ".join(cells))
    ports = board.get("ports") or {}
    print("ports (nodes): " + "; ".join(f"{r} {nodes}" for r, nodes in ports.items()))
    print(f'robber start {tuple(board["robber"])}')


def print_state(rec: Dict[str, object], game, k: int, turns: Sequence[int], last: int = 12) -> None:
    st = game.state
    s = AD.state_summary(st)
    ident = {int(p["seat"]): p for p in rec.get("players", [])}
    print(f'game {rec.get("game")} (seed {rec.get("seed")}, base seed {rec.get("base_seed")}, '
          f'PYTHONHASHSEED={rec.get("hash_seed")}, catanatron {rec.get("catanatron")}, {rec.get("match", "?")}, '
          f'{rec.get("vps_to_win")} VP to win, discard above {rec.get("discard_limit")})')
    print("players in turn order:")
    for i, color in enumerate(rec["colors"]):
        p = ident.get(i, {})
        print(f"  seat {i} {color:<6} {identity(p) if p else '?'}")
    print_board(rec, s["robber"])
    n = len(rec["actions"])
    print(f'\nstate after {k}/{n} actions: turn {s["turn"]}, turn of seat {s["current_turn_seat"]} '
          f'({rec["colors"][s["current_turn_seat"]]}), to act: seat {s["current_player_seat"]} '
          f'({rec["colors"][s["current_player_seat"]]}), prompt {s["prompt"]}')
    winner = game.winning_color()
    if winner is not None:
        print(f"winner: {winner.value}")
    for p in s["players"]:
        tag = "catanbot" if ident.get(p["seat"], {}).get("kind") == "catanbot" else "opponent"
        titles = [t for t, on in (("longest road", p["has_longest_road"]), ("largest army", p["has_largest_army"])) if on]
        print(f'  seat {p["seat"]} {p["color"]:<6} [{tag}] VP {p["vp"]} (public {p["public_vp"]})'
              + (f' - {", ".join(titles)}' if titles else ""))
        print(f'      hand {p["resources"]} ({sum(p["resources"].values())} cards)')
        held = {d: c for d, c in p["dev_cards"].items() if c}
        played = {d: c for d, c in p["dev_played"].items() if c}
        print(f"      dev cards held {held or '{}'}; played {played or '{}'}")
        print(f'      settlements {p["settlements"]} cities {p["cities"]}')
        print(f'      roads {len(p["roads"])} (longest {p["longest_road_length"]}): {p["roads"]}')
    print(f'  bank {s["bank"]}; development deck {s["dev_deck_left"]} cards left')
    if k:
        lo = max(0, k - last)
        print(f"last {k - lo} action(s) (index, turn, colour, action, value -> chance result):")
        for i in range(lo, k):
            c, t, v, r = rec["actions"][i]
            print(f"  {i:5d} t{turns[i]:<4d} {c:<6} {t:<24} {_fmt(v)}" + (f" -> {_fmt(r)}" if r is not None else ""))


def list_games(files: Sequence[str]) -> None:
    for f in files:
        recs, note = read_records(f)
        print(f"{f}: {len(recs)} games" + (f" ({note})" if note else ""))
        for r in sorted(recs, key=lambda x: int(x.get("game", 0))):
            fin = r.get("final") or {}
            ours = r.get("our_seats", [])
            won = fin.get("winner_seat", -1) in ours
            print(f'  game {r.get("game"):>5} seed {r.get("seed")} ours {ours} winner {fin.get("winner")} '
                  f'({"ours" if won else "theirs" if fin.get("winner") else "none"}) VPs {fin.get("vps")} '
                  f'turns {fin.get("turns")} actions {len(r.get("actions", []))}' + (" CRASHED" if r.get("crashed") else ""))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", help=".jsonl.gz action logs or directories holding them")
    ap.add_argument("--game", type=int, default=None, help="game index to rebuild")
    ap.add_argument("--base-seed", type=int, default=None, help="pick the game among logs of several base seeds")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--turn", type=int, default=None, help="stop when catanatron's num_turns reaches T")
    g.add_argument("--action", type=int, default=None, help="stop after K actions")
    ap.add_argument("--last", type=int, default=12, help="how many of the last actions to print (default 12)")
    ap.add_argument("--json", action="store_true", help="print the rebuilt state as JSON (state_summary)")
    ap.add_argument("--check", action="store_true", help="re-play every game and verify final VPs / winner / state")
    ap.add_argument("--list", action="store_true", help="list the games in the logs")
    ap.add_argument("--verbose", action="store_true", help="--check: one line per game")
    args = ap.parse_args(argv)
    try:
        files = log_files(args.logs)
    except FileNotFoundError as ex:
        print(f"no such log: {ex}", file=sys.stderr)
        return 2
    if not files:
        print("no .jsonl.gz action logs found", file=sys.stderr)
        return 2
    if args.check:
        return 1 if check_files(files, args.verbose) else 0
    if args.list or args.game is None:
        list_games(files)
        return 0
    path, rec = find_game(files, args.game, args.base_seed)
    try:
        game, k, turns = replay(rec, upto_action=args.action, upto_turn=args.turn)
    except ValueError as ex:
        print(str(ex), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"file": path, "game": rec.get("game"), "actions_applied": k,
                          "state": AD.state_summary(game.state)}, indent=1))
    else:
        print(f"log: {path}")
        print_state(rec, game, k, turns, args.last)
    return 0


if __name__ == "__main__":
    sys.exit(main())
