#!/usr/bin/env python3
"""Bot server: one seat of a league match, played by *some commit's own code*.

Started by the match runner as a subprocess::

    PYTHONPATH=<champion tree> python3 <gate dir>/serve.py --spec SPEC [--tree DIR] [--label NAME] [--shim FILE]

This file deliberately imports nothing from ``catanbot.league``: it is run as
a script with ``PYTHONPATH`` pointing at the champion's materialised tree, so
``import catanbot`` loads *that* commit's engine, state, actions and bots,
even when the commit predates the league.  (The runner copies this file into
the gate directory at gate start, so a gate keeps one protocol for its whole
life while the working tree moves on.)

Protocol: line-delimited JSON, one request per line on stdin, exactly one
reply per request on stdout.  On start the server writes one ``hello`` line::

    {"hello": {"protocol": 1, "catanbot_file": ..., "accel": ..., "pid": ..., ...}}

Requests:

``{"cmd": "new_game", "seat": i, "seed": s, "num_players": n, "max_turns": 400}``
    builds a fresh bot with ``selfplay.make_bot(spec)``, calls ``reset()``,
    seeds the bot's private ``random.Random(s)``.  Reply
    ``{"ok": true, "bot": name, "wants_observe": bool}`` (``wants_observe`` is
    false when the bot class does not override ``Bot.observe``: the runner then
    skips observations for it - a pure optimisation).
``{"cmd": "decide", "state_id": k, "state": <GameState.to_dict()>, "legal": [actions]}``
    -> ``{"action": <action JSON>, "ms": decide time, "parse_ms": ...}``.
``{"cmd": "observe", "state_id": k, "state": ..., "action": a, "player": p}``
    calls ``bot.observe(state, action, player)`` (``state`` is the state the
    action was taken in, as in ``selfplay.play_game``) -> ``{"ok": true}``.
    ``state`` may be omitted when ``state_id`` equals the id of the last state
    this server received (the actor observes its own action in the state it
    just decided on).
``{"cmd": "ping"}`` -> ``{"ok": true}``;  ``{"cmd": "quit"}`` -> ``{"ok": true, "bye": true}`` and exit.

Errors are replies ``{"error": message, "kind": kind}`` with ``kind`` one of
``state_format`` (this commit cannot represent the state it was sent: version
skew - the runner aborts the gate loudly, naming the fields), ``bot`` (the bot
raised: counted as an error, the runner falls back to the first legal action)
and ``protocol``.

Version skew.  Every received state is round-tripped: ``from_dict`` then
``to_dict`` of *this* commit must reproduce the sent dict.  A field this
commit drops (a newer state format) or reads differently is reported by path,
e.g. ``players[2].dev_cards_new.knight`` or ``pending_trade.counter``.  If a
champion must keep playing a newer engine's states, write a **shim** (a small
Python file named in the champion's registry entry) and pass it with
``--shim``; it may define

* ``adapt_state(d: dict) -> dict``   - rewrite a new-format state for this commit,
* ``adapt_legal(legal: list) -> list`` - rewrite the legal action list,
* ``adapt_action(a: list, legal_json: list) -> list`` - map the returned action back,
* ``IGNORE_FIELDS``                  - dotted paths the round-trip check may ignore
  (fields you have verified are irrelevant to this commit's play).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Run as a script, this file's directory is sys.path[0]; drop it so that only PYTHONPATH decides what
# ``catanbot`` is (and sibling files cannot shadow anything).
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

import importlib.util  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

PROTOCOL = 1


def _diff(sent, back, path: str, out: list, ignore: set, limit: int = 8) -> None:
    """Paths where ``back`` (this commit's round trip) differs from ``sent``."""
    if len(out) >= limit or path in ignore:
        return
    if isinstance(sent, dict):
        if not isinstance(back, dict):
            out.append(f"{path or '<state>'}: sent an object, read back {type(back).__name__}")
            return
        for k in sent:
            p = f"{path}.{k}" if path else str(k)
            if p in ignore:
                continue
            if k not in back:
                out.append(f"{p}: field unknown to this commit (dropped by from_dict/to_dict)")
            else:
                _diff(sent[k], back[k], p, out, ignore, limit)
            if len(out) >= limit:
                return
        return
    if isinstance(sent, list):
        if not isinstance(back, list) or len(back) != len(sent):
            out.append(f"{path}: sent {json.dumps(sent)[:120]}, read back {json.dumps(back)[:120]}")
            return
        for i, (a, b) in enumerate(zip(sent, back)):
            _diff(a, b, f"{path}[{i}]", out, ignore, limit)
        return
    if sent != back:
        out.append(f"{path}: sent {json.dumps(sent)[:80]}, read back {json.dumps(back)[:80]}")


class StateFormatError(Exception):
    pass


class Server:
    def __init__(self, spec: str, label: str = "", tree: str = "", shim_path: str = ""):
        self.spec = spec
        self.label = label
        self.tree = tree
        import catanbot  # noqa: F401  (from PYTHONPATH: the champion's tree)
        from catanbot import selfplay
        from catanbot.state import GameState
        from catanbot.agents.base import Bot
        from catanbot import actions as A
        self.catanbot = catanbot
        self.selfplay = selfplay
        self.GameState = GameState
        self.BotBase = Bot
        self.to_json = getattr(A, "to_json", None) or (lambda a: [list(x) if isinstance(x, tuple) else x for x in a])
        self.from_json = getattr(A, "from_json", None) or (
            lambda o: tuple(tuple(x) if isinstance(x, list) else x for x in o))
        self.shim = None
        self.ignore: set = set()
        if shim_path:
            spec_ = importlib.util.spec_from_file_location("catanbot_league_shim", shim_path)
            mod = importlib.util.module_from_spec(spec_)
            spec_.loader.exec_module(mod)  # type: ignore[union-attr]
            self.shim = mod
            self.ignore = set(getattr(mod, "IGNORE_FIELDS", ()) or ())
        self.bot = None
        self.rng = random.Random(0)
        self.seat = -1
        self.max_turns = 400
        self.check = True
        self.state_id = None
        self.state_obj = None
        self.wants_observe = True

    # --- hello ----------------------------------------------------------
    def hello(self) -> dict:
        info = {"protocol": PROTOCOL, "label": self.label, "spec": self.spec, "pid": os.getpid(),
                "python": sys.version.split()[0], "executable": sys.executable,
                "catanbot_file": os.path.abspath(self.catanbot.__file__),
                "hash_seed": os.environ.get("PYTHONHASHSEED"),
                "catanbot_env": {k: v for k, v in os.environ.items() if k.startswith("CATANBOT_")},
                "shim": bool(self.shim)}
        try:
            from catanbot import accel
            ok = bool(accel.AVAILABLE) and (accel.verify() if hasattr(accel, "verify") else True)
            info["accel"] = bool(ok)
            if hasattr(accel, "native_search_available"):
                info["native_search"] = bool(accel.native_search_available())
            if hasattr(accel, "_core") and getattr(accel, "_core", None) is not None:
                info["accel_file"] = getattr(accel._core, "__file__", None)
        except Exception as exc:  # an old commit without accel
            info["accel"] = False
            info["accel_error"] = repr(exc)
        if self.tree:
            info["tree_ok"] = info["catanbot_file"].startswith(os.path.abspath(self.tree) + os.sep)
        return info

    # --- state handling ----------------------------------------------------
    def _state(self, msg: dict):
        sid = msg.get("state_id")
        d = msg.get("state")
        if d is None:
            if self.state_obj is None or sid is None or sid != self.state_id:
                raise ValueError(f"state omitted but state_id {sid!r} is not the cached one ({self.state_id!r})")
            return self.state_obj
        if self.shim is not None and hasattr(self.shim, "adapt_state"):
            d = self.shim.adapt_state(d)
        try:
            s = self.GameState.from_dict(d)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            raise StateFormatError(f"this commit's GameState.from_dict cannot read the state: {exc!r}"
                                   f" (missing / changed field?)") from exc
        if hasattr(s, "max_turns"):
            s.max_turns = self.max_turns
        if self.check:
            problems: list = []
            _diff(d, s.to_dict(), "", problems, self.ignore)
            if problems:
                raise StateFormatError("state format skew: " + "; ".join(problems))
        self.state_id = sid
        self.state_obj = s
        return s

    # --- commands ---------------------------------------------------------
    def new_game(self, msg: dict) -> dict:
        spec = msg.get("spec") or self.spec
        self.bot = self.selfplay.make_bot(spec)
        self.bot.reset()
        self.rng = random.Random(int(msg.get("seed", 0)))
        self.seat = int(msg.get("seat", -1))
        self.max_turns = int(msg.get("max_turns", 400))
        self.check = bool(msg.get("check", True))
        self.state_id = None
        self.state_obj = None
        obs = getattr(type(self.bot), "observe", None)
        self.wants_observe = obs is not None and obs is not getattr(self.BotBase, "observe", None)
        return {"ok": True, "bot": getattr(self.bot, "name", spec), "wants_observe": self.wants_observe}

    def decide(self, msg: dict) -> dict:
        t0 = time.perf_counter()
        s = self._state(msg)
        legal_json = msg["legal"]
        if self.shim is not None and hasattr(self.shim, "adapt_legal"):
            legal_json = self.shim.adapt_legal(legal_json)
        legal = [self.from_json(a) for a in legal_json]
        t1 = time.perf_counter()
        try:
            a = self.bot.decide(s, legal, self.rng)
        except Exception as exc:
            return {"error": f"bot.decide raised {exc!r}", "kind": "bot",
                    "traceback": traceback.format_exc(limit=6)[-1500:]}
        t2 = time.perf_counter()
        out = self.to_json(a)
        if self.shim is not None and hasattr(self.shim, "adapt_action"):
            out = self.shim.adapt_action(out, msg["legal"])
        return {"action": out, "ms": (t2 - t1) * 1e3, "parse_ms": (t1 - t0) * 1e3}

    def observe(self, msg: dict) -> dict:
        s = self._state(msg)
        try:
            self.bot.observe(s, self.from_json(msg["action"]), int(msg["player"]))
        except Exception as exc:
            return {"error": f"bot.observe raised {exc!r}", "kind": "bot",
                    "traceback": traceback.format_exc(limit=6)[-1500:]}
        return {"ok": True}

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        try:
            if cmd == "decide":
                return self.decide(msg)
            if cmd == "observe":
                return self.observe(msg)
            if cmd == "new_game":
                return self.new_game(msg)
            if cmd == "ping":
                return {"ok": True}
            if cmd == "quit":
                return {"ok": True, "bye": True}
            return {"error": f"unknown cmd {cmd!r}", "kind": "protocol"}
        except StateFormatError as exc:
            return {"error": str(exc), "kind": "state_format"}
        except Exception as exc:
            return {"error": f"{cmd}: {exc!r}", "kind": "protocol", "traceback": traceback.format_exc(limit=6)[-1500:]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--tree", default="", help="expected source tree (hello reports whether catanbot came from it)")
    ap.add_argument("--shim", default="")
    args = ap.parse_args(argv)
    # The protocol owns the real stdout; anything the bot code prints (Python or C++) goes to stderr.
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    inp = sys.stdin.buffer

    def send(obj) -> None:
        out.write(json.dumps(obj, separators=(",", ":")).encode() + b"\n")

    try:
        server = Server(args.spec, args.label, args.tree, args.shim)
        send({"hello": server.hello()})
    except Exception as exc:
        send({"error": f"server start failed: {exc!r}", "kind": "start",
              "traceback": traceback.format_exc(limit=8)[-2000:]})
        return 2
    while True:
        line = inp.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError as exc:
            send({"error": f"bad json: {exc}", "kind": "protocol"})
            continue
        reply = server.handle(msg)
        send(reply)
        if reply.get("bye"):
            return 0


if __name__ == "__main__":
    sys.exit(main())
