"""Match runner: the current tree's engine, every seat played by a bot server (or an in-process bot).

``play_match_game`` runs one game with the engine of *this* tree and asks
each seat for its decisions.  A seat is either

* a :class:`ServerSeat` - a :class:`BotServer` subprocess running
  ``serve.py`` with ``PYTHONPATH`` at some commit's tree, so the seat plays
  with that commit's own code (champions, and the candidate too so that every
  seat pays the same serialisation cost), or
* an :class:`InProcessSeat` - a bot object of this process (tests, quick
  experiments).

The loop mirrors ``selfplay.play_game``: the acting player's seat decides
(``decide(state, legal, rng)``), the action is validated against the engine's
legal list (plus well-formed trade proposals outside the engine's bounded
candidate list, which ``play_game`` also accepts), every seat observes it
(``observe(state_before, action, player)``) and the engine applies it with
its own ``random.Random(engine_seed)``.  Differences from ``play_game``, by
design: each seat's bot draws from its *own* ``random.Random(bot_seed)``
instead of the game's rng (so a seat's randomness does not depend on who
else is playing), and an illegal or failed decision is **counted** and
replaced by the first legal action instead of silently substituted.

A seat whose server crashes or times out voids the game (:class:`GameVoid`);
a server that cannot read the engine's state format raises
:class:`VersionSkewError` naming the offending fields - never a silent
fallback.
"""
from __future__ import annotations

import json
import os
import random
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .. import actions as A
from .. import engine as E
from ..state import PHASE_GAME_OVER, new_game

SERVE_SCRIPT = Path(__file__).resolve().with_name("serve.py")
MAX_ACTIONS = 20000   # same guard as selfplay.play_game


class ServerError(RuntimeError):
    def __init__(self, msg: str, kind: str = "protocol"):
        super().__init__(msg)
        self.kind = kind


class VersionSkewError(ServerError):
    """A bot server's commit cannot represent the engine's state (see ``serve.py``: write a shim)."""


class ServerCrash(ServerError):
    pass


class ServerTimeout(ServerError):
    pass


class GameVoid(RuntimeError):
    """The game cannot be completed (a server crashed / timed out); it is recorded as void, not scored."""

    def __init__(self, msg: str, seat: int):
        super().__init__(msg)
        self.seat = seat


def server_env(tree: os.PathLike, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment of a bot server: ``PYTHONPATH`` = its tree only, pinned hash seed, one BLAS thread,
    and none of the operator's ``CATANBOT_*`` switches (a champion runs with its defaults unless its
    registry entry says otherwise)."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CATANBOT_") and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(PYTHONPATH=str(tree), PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


class BotServer:
    """Client side of one ``serve.py`` subprocess (line-delimited JSON over its stdin / stdout)."""

    def __init__(self, tree: os.PathLike, spec: str, label: str = "", serve_script: os.PathLike = SERVE_SCRIPT,
                 python: str = sys.executable, env: Optional[Dict[str, str]] = None, shim: Optional[str] = None,
                 log_path: Optional[os.PathLike] = None, start_timeout: float = 120.0, timeout: float = 600.0,
                 require_accel: bool = False):
        self.tree = Path(tree)
        self.spec = spec
        self.label = label or spec
        self.serve_script = Path(serve_script)
        self.python = python
        self.extra_env = dict(env or {})
        self.shim = shim
        self.log_path = Path(log_path) if log_path else None
        self.start_timeout = start_timeout
        self.timeout = timeout
        self.require_accel = require_accel
        self.proc: Optional[subprocess.Popen] = None
        self.hello: Dict[str, Any] = {}
        self._buf = bytearray()
        self._log = None
        self.restarts = 0

    # --- lifecycle -----------------------------------------------------------------------
    def start(self) -> "BotServer":
        cmd = [self.python, str(self.serve_script), "--spec", self.spec, "--label", self.label, "--tree", str(self.tree)]
        if self.shim:
            cmd += ["--shim", str(self.shim)]
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log if self._log else subprocess.DEVNULL,
                                     cwd=str(self.tree), env=server_env(self.tree, self.extra_env), bufsize=0)
        self._buf = bytearray()
        msg = self._recv(self.start_timeout)
        if "hello" not in msg:
            self.close()
            raise ServerError(f"{self.label}: server failed to start: {msg.get('error')}\n{msg.get('traceback', '')}",
                              msg.get("kind", "start"))
        self.hello = msg["hello"]
        if self.hello.get("protocol") != 1:
            self.close()
            raise ServerError(f"{self.label}: protocol {self.hello.get('protocol')} != 1")
        if self.hello.get("tree_ok") is False:
            self.close()
            raise ServerError(f"{self.label}: catanbot was imported from {self.hello.get('catanbot_file')}, "
                              f"not from its tree {self.tree} (an installed package shadows PYTHONPATH?)")
        if self.require_accel and not self.hello.get("accel"):
            self.close()
            raise ServerError(f"{self.label}: the C++ extension is not available in {self.tree} "
                              f"(materialise with a build, or allow running without it)")
        return self

    def close(self) -> None:
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                self._send(b'{"cmd":"quit"}\n')
                p.wait(timeout=5)
            except Exception:
                p.kill()
                try:
                    p.wait(timeout=5)
                except Exception:
                    pass
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None

    def restart(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None
        self.restarts += 1
        self.start()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # --- I/O ------------------------------------------------------------------------------
    def _stderr_tail(self) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        with open(self.log_path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 1500))
            return f.read().decode(errors="replace")

    def _send(self, data: bytes) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise ServerCrash(f"{self.label}: server not running", "crash")
        try:
            view = memoryview(data)
            while view:
                n = self.proc.stdin.write(view)
                view = view[n:]
        except (BrokenPipeError, OSError) as exc:
            raise ServerCrash(f"{self.label}: server pipe closed ({exc}); stderr tail:\n{self._stderr_tail()}",
                              "crash") from None

    def _recv(self, timeout: Optional[float]) -> Dict[str, Any]:
        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = None if not timeout else time.monotonic() + timeout
        while True:
            i = self._buf.find(b"\n")
            if i >= 0:
                line = bytes(self._buf[:i])
                del self._buf[:i + 1]
                try:
                    return json.loads(line)
                except ValueError:
                    raise ServerError(f"{self.label}: unparsable reply {line[:200]!r}") from None
            wait = None if deadline is None else deadline - time.monotonic()
            if wait is not None and wait <= 0:
                raise ServerTimeout(f"{self.label}: no reply within {timeout:.0f} s", "timeout")
            r, _, _ = select.select([fd], [], [], wait)
            if not r:
                raise ServerTimeout(f"{self.label}: no reply within {timeout:.0f} s", "timeout")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                code = self.proc.poll()
                raise ServerCrash(f"{self.label}: server exited (code {code}); stderr tail:\n{self._stderr_tail()}",
                                  "crash")
            self._buf += chunk

    def send_raw(self, data: bytes) -> None:
        self._send(data)

    def recv(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        reply = self._recv(self.timeout if timeout is None else timeout)
        if "error" in reply:
            kind = reply.get("kind", "protocol")
            if kind == "state_format":
                raise VersionSkewError(
                    f"{self.label} (tree {self.tree}) cannot read the engine's state: {reply['error']}.  "
                    f"Its commit's GameState predates the current format: add a compatibility shim for it "
                    f"(see catanbot/league/serve.py and docs/LEAGUE.md, 'Version skew').", kind)
        return reply

    def request(self, obj: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        self._send(json.dumps(obj, separators=(",", ":")).encode() + b"\n")
        return self.recv(timeout)


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------
class _StateCache:
    """The state's JSON, serialised at most once per action whatever the number of servers."""

    __slots__ = ("state", "sid", "_json", "ser_s")

    def __init__(self, state, sid: int):
        self.state = state
        self.sid = sid
        self._json: Optional[bytes] = None
        self.ser_s = 0.0

    def json(self) -> bytes:
        if self._json is None:
            t = time.perf_counter()
            self._json = json.dumps(self.state.to_dict(), separators=(",", ":")).encode()
            self.ser_s += time.perf_counter() - t
        return self._json


@dataclass
class Decision:
    action: Any = None            # tuple, or None when the seat failed
    error: Optional[str] = None
    rt_ms: float = 0.0            # runner-side wall time of the decision (round trip for servers)
    bot_ms: float = 0.0           # time inside bot.decide (reported by the server)
    parse_ms: float = 0.0         # server-side JSON + from_dict (+ round-trip check)


class Seat:
    label: str = "seat"
    spec: str = ""
    wants_observe: bool = True

    def new_game(self, seat: int, seed: int, num_players: int, max_turns: int, check: bool = True) -> None:
        raise NotImplementedError

    def decide(self, cache: _StateCache, legal: List[tuple], legal_json: bytes) -> Decision:
        raise NotImplementedError

    def observe_send(self, cache: _StateCache, action: tuple, action_json: bytes, player: int) -> None:
        raise NotImplementedError

    def observe_recv(self) -> Optional[str]:
        return None

    def close(self) -> None:
        pass

    def info(self) -> Dict[str, Any]:
        return {"label": self.label, "spec": self.spec}


class InProcessSeat(Seat):
    """A bot of this process (``factory()`` returns a fresh bot each game, like ``make_bot``)."""

    def __init__(self, factory: Callable[[], Any], label: str = "", spec: str = ""):
        self.factory = factory
        self.label = label or spec or "inproc"
        self.spec = spec
        self.bot = None
        self.rng = random.Random(0)

    def new_game(self, seat, seed, num_players, max_turns, check=True):
        self.bot = self.factory()
        self.bot.reset()
        self.rng = random.Random(seed)
        from ..agents.base import Bot
        obs = getattr(type(self.bot), "observe", None)
        self.wants_observe = obs is not None and obs is not Bot.observe

    def decide(self, cache, legal, legal_json):
        t0 = time.perf_counter()
        try:
            a = self.bot.decide(cache.state, legal, self.rng)
            err = None
        except Exception as exc:
            a, err = None, f"bot.decide raised {exc!r}"
        ms = (time.perf_counter() - t0) * 1e3
        return Decision(a, err, ms, ms, 0.0)

    def observe_send(self, cache, action, action_json, player):
        self._obs_err = None
        try:
            self.bot.observe(cache.state, action, player)
        except Exception as exc:
            self._obs_err = f"bot.observe raised {exc!r}"

    def observe_recv(self):
        return getattr(self, "_obs_err", None)

    def info(self):
        return {"label": self.label, "spec": self.spec, "kind": "inprocess"}


class ServerSeat(Seat):
    """A seat played by a :class:`BotServer` (started on first use)."""

    def __init__(self, server: BotServer):
        self.server = server
        self.label = server.label
        self.spec = server.spec
        self._sid_sent: Optional[int] = None

    def new_game(self, seat, seed, num_players, max_turns, check=True):
        if not self.server.alive:
            if self.server.proc is not None:
                self.server.restart()
            else:
                self.server.start()
        r = self.server.request({"cmd": "new_game", "seat": seat, "seed": seed, "num_players": num_players,
                                 "max_turns": max_turns, "check": check}, timeout=self.server.start_timeout)
        if "error" in r:
            raise ServerError(f"{self.label}: new_game failed: {r['error']}\n{r.get('traceback', '')}", r.get("kind", ""))
        self.wants_observe = bool(r.get("wants_observe", True))
        self._sid_sent = None

    def _state_part(self, cache: _StateCache) -> bytes:
        if self._sid_sent == cache.sid:
            return b'"state_id":%d' % cache.sid
        self._sid_sent = cache.sid
        return b'"state_id":%d,"state":' % cache.sid + cache.json()

    def decide(self, cache, legal, legal_json):
        msg = b'{"cmd":"decide",' + self._state_part(cache) + b',"legal":' + legal_json + b"}\n"
        t0 = time.perf_counter()
        self.server.send_raw(msg)
        r = self.server.recv()
        rt = (time.perf_counter() - t0) * 1e3
        if "error" in r:
            return Decision(None, f"{r.get('kind')}: {r['error']}", rt, 0.0, 0.0)
        try:
            a = A.from_json(r["action"])
            hash(a)
        except Exception as exc:
            return Decision(None, f"unreadable action {r.get('action')!r}: {exc!r}", rt, float(r.get("ms", 0.0)), 0.0)
        return Decision(a, None, rt, float(r.get("ms", 0.0)), float(r.get("parse_ms", 0.0)))

    def observe_send(self, cache, action, action_json, player):
        msg = b'{"cmd":"observe",' + self._state_part(cache) + b',"action":' + action_json + b',"player":%d}\n' % player
        self.server.send_raw(msg)

    def observe_recv(self):
        r = self.server.recv()
        if "error" in r:
            return f"{r.get('kind')}: {r['error']}"
        return None

    def close(self):
        self.server.close()

    def info(self):
        h = self.server.hello
        return {"label": self.label, "spec": self.spec, "kind": "server", "tree": str(self.server.tree),
                "accel": h.get("accel"), "native_search": h.get("native_search"), "python": h.get("python"),
                "catanbot_file": h.get("catanbot_file")}


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------
def check_action(state, a, legal: Sequence[tuple], legal_set: set) -> Optional[str]:
    """``None`` when ``a`` is playable, else why not.  Trade proposals outside the engine's bounded
    candidate list are playable when the engine accepts them (as in ``selfplay.play_game``)."""
    try:
        if a in legal_set:
            return None
    except TypeError:
        return f"unhashable action {a!r}"
    if isinstance(a, tuple) and a and a[0] == A.PROPOSE_TRADE and any(x[0] == A.PROPOSE_TRADE for x in legal):
        try:
            E.apply(state, a, random.Random(0))
            return None
        except Exception as exc:
            return f"illegal trade proposal {a!r}: {exc}"
    return f"illegal action {a!r}"


@dataclass
class SeatStats:
    decisions: int = 0
    nontrivial: int = 0
    rt_ms: float = 0.0
    rt_ms_nontrivial: float = 0.0
    bot_ms: float = 0.0
    parse_ms: float = 0.0
    max_rt_ms: float = 0.0
    illegal: int = 0
    errors: int = 0            # bot exceptions in decide
    observe_errors: int = 0
    extra_proposals: int = 0
    observes: int = 0


def play_match_game(seats: Sequence[Seat], engine_seed: int, bot_seeds: Sequence[int], num_players: int,
                    max_turns: int = 400, check: bool = True, action_log: Optional[list] = None,
                    max_actions: int = MAX_ACTIONS) -> Dict[str, Any]:
    """Play one game; returns the outcome and per-seat timing / error statistics.

    Raises :class:`GameVoid` when a server crashes or times out mid-game and
    :class:`VersionSkewError` when a server cannot read the state format.
    """
    n = num_players
    if len(seats) != n:
        raise ValueError(f"{n} players but {len(seats)} seats")
    t_start = time.time()
    rng = random.Random(engine_seed)
    state = new_game(n, rng=rng)
    state.max_turns = max_turns
    for i, seat in enumerate(seats):
        try:
            seat.new_game(i, int(bot_seeds[i]), n, max_turns, check)
        except (ServerCrash, ServerTimeout) as exc:
            raise GameVoid(str(exc), i) from exc
    st = [SeatStats() for _ in range(n)]
    errors: List[str] = []
    ser_s = 0.0
    n_actions = 0
    sid = 0
    while state.phase != PHASE_GAME_OVER:
        i = E.acting_player(state)
        legal = E.legal_actions(state)
        if not legal:
            break
        cache = _StateCache(state, sid)
        t = time.perf_counter()
        legal_json = json.dumps([A.to_json(a) for a in legal], separators=(",", ":")).encode()
        ser_s += time.perf_counter() - t
        try:
            d = seats[i].decide(cache, legal, legal_json)
        except (ServerCrash, ServerTimeout) as exc:
            raise GameVoid(str(exc), i) from exc
        s = st[i]
        s.decisions += 1
        s.rt_ms += d.rt_ms
        s.bot_ms += d.bot_ms
        s.parse_ms += d.parse_ms
        s.max_rt_ms = max(s.max_rt_ms, d.rt_ms)
        if len(legal) > 1:
            s.nontrivial += 1
            s.rt_ms_nontrivial += d.rt_ms
        a = d.action
        if d.error is not None:
            s.errors += 1
            if len(errors) < 20:
                errors.append(f"seat {i} ({seats[i].label}) action {n_actions}: {d.error}")
            a = legal[0]
        else:
            why = check_action(state, a, legal, set(legal))
            if why is not None:
                s.illegal += 1
                if len(errors) < 20:
                    errors.append(f"seat {i} ({seats[i].label}) action {n_actions}: {why}; fallback {legal[0]!r}")
                a = legal[0]
            elif a not in legal:
                s.extra_proposals += 1
        action_json = json.dumps(A.to_json(a), separators=(",", ":")).encode()
        if action_log is not None:
            action_log.append((i, A.to_json(a)))
        watchers = [j for j, seat in enumerate(seats) if seat.wants_observe]
        try:
            for j in watchers:
                seats[j].observe_send(cache, a, action_json, i)
            for j in watchers:
                err = seats[j].observe_recv()
                st[j].observes += 1
                if err is not None:
                    st[j].observe_errors += 1
                    if len(errors) < 20:
                        errors.append(f"seat {j} ({seats[j].label}) observe {n_actions}: {err}")
        except (ServerCrash, ServerTimeout) as exc:
            raise GameVoid(str(exc), -1) from exc
        ser_s += cache.ser_s
        state = E.apply_inplace(state, a, rng)
        n_actions += 1
        sid += 1
        if n_actions > max_actions:
            break
    vps = [state.total_vp(k) for k in range(n)]
    return {
        "winner": int(state.winner), "vps": vps, "turns": int(state.turn), "actions": n_actions,
        "capped": state.winner < 0,
        "seats": [
            {"decisions": s.decisions, "nontrivial": s.nontrivial,
             "decide_ms": round(s.rt_ms / max(1, s.decisions), 3),
             "decide_ms_nontrivial": round(s.rt_ms_nontrivial / max(1, s.nontrivial), 3),
             "bot_ms": round(s.bot_ms / max(1, s.decisions), 3),
             "overhead_ms": round((s.rt_ms - s.bot_ms) / max(1, s.decisions), 4),
             "parse_ms": round(s.parse_ms / max(1, s.decisions), 4),
             "max_ms": round(s.max_rt_ms, 1), "total_s": round(s.rt_ms / 1e3, 3),
             "illegal": s.illegal, "errors": s.errors, "observe_errors": s.observe_errors,
             "extra_proposals": s.extra_proposals, "observes": s.observes}
            for s in st],
        "error_log": errors,
        "serialize_ms": round(ser_s * 1e3, 2),
        "duration_s": round(time.time() - t_start, 3),
    }
