#!/usr/bin/env python3
"""Benchmark catanbot against the players shipped with the ``catanatron`` engine.

Usage::

    python3 scripts/bench_catanatron.py --games 20 --opponent vp \\
        --spec "search:depth=1,evaluator=heuristic" [--seed 0] [--workers 1] [--json out.json] \\
        [--vps-to-win 10] [--discard-limit 7] [--trades off|native|value|fair] \\
        [--opponent-params KEY=VAL,...] [--hash-seed 0]
    python3 scripts/bench_catanatron.py --ladder strong --games 20 --workers 2
    python3 scripts/bench_catanatron.py --list-opponents
    python3 scripts/bench_catanatron.py --probe-trades 200 --opponent value,alphabeta,random

Every game seats one :class:`catanbot.bench.catanatron_adapter.CatanbotPlayer`
(built from ``--spec``, see ``catanbot.selfplay.make_bot``) against three
copies of the chosen catanatron opponent:

* ``vp``       - ``VictoryPointPlayer`` (greedy one-ply VP maximiser, the strongest
  stock player of the PyPI 3.2.1 wheel)
* ``weighted`` - ``WeightedRandomPlayer`` (random, biased to cities > settlements > dev cards)
* ``random``   - ``RandomPlayer`` (uniform random)
* ``vf`` / ``ab`` - our own value-function / alpha-beta players built inside the
  catanatron engine (``catanbot.bench.catanatron_players``; both versions)
* ``value`` / ``alphabeta`` / ``sameturn`` / ``playouts`` / ``mcts`` - catanatron's
  own strong players (``catanatron.players.value`` / ``minimax`` / ``playouts`` /
  ``mcts``), shipped by the 3.3 engine of the GitHub checkout only

``--opponent-params KEY=VAL,...`` passes constructor parameters to every
opponent: the fields of a 3.3 player's ``Params`` (``alphabeta``:
``depth=3,prunning=true``; ``mcts``: ``num_simulations=50``; ``playouts``:
``num_playouts=10``; ``value``: ``value_fn=contender``) or the keyword
arguments of an older-style class (``ab``: ``budget=4000,depth=3``), coerced
to the declared types.  An opponent that takes none of them is reported like
an unavailable one (exit code 2 when it is the only opponent).

Both catanatron generations work (3.2.1 wheel, 3.3 checkout; see the adapter).
A preset that the running catanatron does not ship fails with a one-line
"not available on catanatron X" message and exit code 2 when it is the only
opponent asked for; inside a comma list or a ``--ladder`` it is skipped with
the same message (exit code 2 only if nothing could be played).
``--list-opponents`` shows which presets resolve here.

Domestic trading (3.3 only, ``--trades``; see
``catanbot.bench.catanatron_adapter.TRADE_MODES`` and ``docs/BENCHMARKS.md``):
``off`` (default) - catanbot never offers; ``native`` - catanbot offers and the
opponents answer with their own ``decide`` (catanatron's players answer
degenerately: the value player always rejects, the alpha-beta / MCTS players
raise and are counted as rejecting); ``value`` - opponents accept iff their
value function rises with the trade; ``fair`` - ``value`` plus no n-for-fewer
against themselves and no deals with a proposer within 2 VP of winning.
``value`` / ``fair`` are OUR model of a sensible opponent, not catanatron's.
``--probe-trades N`` measures instead how each ``--opponent`` answers N offers
per category (1:1, favourable to them, unfavourable) sampled from mid-game
states, natively and under the two rules.

Seats rotate (game ``g`` puts catanbot in seat ``g % 4``), boards / dev decks
/ dice come from catanatron seeded per game.  catanatron iterates sets of
``Color`` enums, whose hashes depend on ``PYTHONHASHSEED``: the script
therefore re-executes itself with ``PYTHONHASHSEED`` set to ``--hash-seed``
(default 0; ``-1`` keeps the interpreter's random hashing), which makes a run
reproducible across processes and lets two runs with the same ``--seed`` be
compared game by game (e.g. ``--trades value`` vs ``--trades off``: the games
coincide until the first trade).  The script prints the win rate (a random
seat would win 25 %), average victory points, per-seat results, turn counts,
adapter statistics, the domestic-trade counts and the compute per decision of
both sides (mean / p95 ms, seconds per game), plus a Markdown table row ready
for ``docs/BENCHMARKS.md``.  ``--workers`` plays games in parallel processes.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanatron.game import TURNS_LIMIT  # noqa: E402
from catanatron.models.player import RandomPlayer  # noqa: E402
from catanatron.players.search import VictoryPointPlayer  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402

from catanbot.bench.catanatron_adapter import (  # noqa: E402
    CATANATRON_VERSION,
    COLORS,
    DEFAULT_SPEC,
    DOMESTIC_TRADING,
    TRADE_MODES,
    BenchOpponent,
    CatanbotPlayer,
    play_game,
    timing_summary,
)

#: Stock players present in every catanatron version.
OPPONENTS = {
    "vp": VictoryPointPlayer,
    "weighted": WeightedRandomPlayer,
    "random": RandomPlayer,
}

# Presets that exist only in some catanatron versions (the GitHub checkout has the strong
# players; PyPI 3.2.1 does not) or in this repo; resolved lazily by import path.
PRESETS = {
    "alphabeta": "catanatron.players.minimax:AlphaBetaPlayer",
    "sameturn": "catanatron.players.minimax:SameTurnAlphaBetaPlayer",
    "value": "catanatron.players.value:ValueFunctionPlayer",
    "mcts": "catanatron.players.mcts:MCTSPlayer",
    "playouts": "catanatron.players.playouts:GreedyPlayoutsPlayer",
    "vf": "catanbot.bench.catanatron_players:ValueFunctionPlayer",
    "ab": "catanbot.bench.catanatron_players:AlphaBetaPlayer",
}
LADDERS = {
    "controls": ["random", "weighted", "vp"],
    "standins": ["vf", "ab"],
    "strong": ["value", "alphabeta", "sameturn", "playouts", "mcts"],
    "full": ["random", "weighted", "vp", "vf", "ab", "value", "alphabeta", "sameturn", "playouts", "mcts"],
}


class OpponentUnavailable(SystemExit):
    """An opponent that cannot be used here: exit code 2 and a one-line ``message`` (no traceback)."""

    def __init__(self, message: str):
        super().__init__(2)
        self.message = message

    def __str__(self) -> str:
        return self.message


class BadOpponentParams(OpponentUnavailable):
    """``--opponent-params`` that the opponent class does not accept (handled like an unavailable opponent)."""


def _quiet_print(*args, **kwargs) -> None:
    """Replaces ``print`` in catanatron's playouts module (one line per decision otherwise)."""


def opponent_path(name: str) -> Optional[str]:
    """``module:Class`` of a preset or import path (``None`` for an unknown preset name)."""
    if name in OPPONENTS:
        cls = OPPONENTS[name]
        return f"{cls.__module__}:{cls.__name__}"
    path = PRESETS.get(name, name)
    if ":" in path:
        return path
    if "." in path:
        mod, cls = path.rsplit(".", 1)
        return f"{mod}:{cls}"
    return None


def resolve_opponent(name: str):
    """Opponent class from a preset key or an import path ``module:Class`` / ``module.Class``.

    Raises :class:`OpponentUnavailable` (a ``SystemExit`` with code 2 and a
    one-line message) for an unknown preset or one the running catanatron
    version does not ship.
    """
    if name in OPPONENTS:
        return OPPONENTS[name]
    path = opponent_path(name)
    if path is None:
        raise OpponentUnavailable(f"unknown opponent '{name}' (presets: {', '.join(sorted(OPPONENTS) + sorted(PRESETS))}, "
                                  "or an import path module:Class)")
    mod, cls = path.split(":", 1)
    try:
        module = importlib.import_module(mod)
        if hasattr(module, "USE_MULTIPROCESSING"):
            # catanatron 3.3's GreedyPlayoutsPlayer opens a Pool(cpu_count()) per decision, which
            # would exceed --workers and cannot run inside our (daemonic) worker processes; the
            # seeded playouts give the same result in-process.  It also prints one line per
            # decision: silence the module's print.
            module.USE_MULTIPROCESSING = False
            module.print = _quiet_print
        return getattr(module, cls)
    except (ImportError, AttributeError) as ex:
        hint = ("; the strong players need the 3.3 engine: pip install -e <GitHub clone of catanatron>"
                if mod.startswith("catanatron.") else "")
        raise OpponentUnavailable(f"opponent '{name}' ({path}) is not available on catanatron {CATANATRON_VERSION}: "
                                  f"{ex}{hint}")


# ---------------------------------------------------------------------------
# --opponent-params
# ---------------------------------------------------------------------------
def parse_opponent_params(text: Optional[str]) -> Dict[str, str]:
    """``"depth=3,prunning=true"`` -> ``{"depth": "3", "prunning": "true"}`` (values stay strings)."""
    out: Dict[str, str] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise BadOpponentParams(f"--opponent-params: '{item}' is not KEY=VAL")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _coerce(value: str, default):
    """Parse a ``--opponent-params`` string like the parameter's default value."""
    if isinstance(default, bool):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{value!r} is not a boolean")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if default is None or default is inspect.Parameter.empty:
        if value.strip().lower() == "none":
            return None
        for conv in (int, float):
            try:
                return conv(value)
            except ValueError:
                pass
    return value


def _init_params(cls) -> Dict[str, inspect.Parameter]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    return {k: p for k, p in sig.parameters.items()
            if k not in ("self", "color") and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _params_fields(cls) -> List[str]:
    params_cls = getattr(cls, "Params", None)
    if params_cls is None or not dataclasses.is_dataclass(params_cls):
        return []
    return [f.name for f in dataclasses.fields(params_cls)]


def opponent_factory(cls, params: Optional[Dict[str, str]] = None, name: Optional[str] = None) -> Callable:
    """``make(color) -> Player`` building ``cls`` with ``--opponent-params``.

    Keyword arguments of the class's ``__init__`` are used when they cover every
    key (older-style players, our stand-ins); otherwise the class's 3.3 ``Params``
    dataclass is built with ``catanatron.params.build_params``.  Raises
    :class:`BadOpponentParams` when neither accepts the keys or a value does
    not parse.  One instance is built here to validate.
    """
    label = name or cls.__name__
    if not params:
        return cls
    init = _init_params(cls)
    if all(k in init for k in params) and "params" not in params:
        try:
            kwargs = {k: _coerce(v, init[k].default) for k, v in params.items()}
            cls(COLORS[0], **kwargs)
        except Exception as ex:   # noqa: BLE001 - reported as a one-line usage error
            raise BadOpponentParams(f"opponent '{label}' ({cls.__name__}): bad --opponent-params {params}: {ex}")

        def make_kwargs(color):
            return cls(color, **kwargs)
        return make_kwargs
    fields = _params_fields(cls)
    if fields and all(k in fields for k in params):
        try:
            from catanatron.params import build_params
            built = build_params(cls, named=dict(params))
            cls(COLORS[0], built)
        except Exception as ex:   # noqa: BLE001
            raise BadOpponentParams(f"opponent '{label}' ({cls.__name__}): bad --opponent-params {params}: {ex}")

        def make_params(color):
            return cls(color, built)
        return make_params
    accepted = sorted(set(fields) | {k for k in init if k not in ("params", "is_bot")})
    raise BadOpponentParams(f"opponent '{label}' ({cls.__name__}) takes no parameter "
                            f"{', '.join(sorted(k for k in params if k not in fields and k not in init))} "
                            f"(accepted: {', '.join(accepted) or 'none'})")


def list_opponents() -> List[Tuple[str, str, Optional[type], str]]:
    """``(preset, module:Class, class or None, status)`` for every preset on the running version."""
    rows = []
    for name in list(OPPONENTS) + list(PRESETS):
        path = opponent_path(name) or "?"
        try:
            cls = resolve_opponent(name)
            rows.append((name, path, cls, "available"))
        except OpponentUnavailable as ex:
            rows.append((name, path, None, ex.message.split(": ", 1)[-1] if ": " in ex.message else ex.message))
    return rows


def print_opponents() -> None:
    print(f"opponent presets on catanatron {CATANATRON_VERSION}:")
    for name, path, cls, status in list_opponents():
        mark = "available" if cls is not None else f"not available on catanatron {CATANATRON_VERSION}"
        print(f"  {name:<10} {path:<56} {mark}")
    for ladder, names in LADDERS.items():
        print(f"ladder {ladder:<9}: {', '.join(names)}")


def game_seed(base_seed: int, g: int) -> int:
    """Non-zero per-game seed (catanatron treats seed 0 as 'random')."""
    return base_seed * 100003 + g + 1


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------
def run_one(job: tuple) -> Dict[str, object]:
    """Play game ``g`` of a batch; returns the summary dict plus adapter / trade / timing stats.

    ``job`` is ``(g, base_seed, spec, opponent, vps_to_win, discard_limit[, trades[, opponent_params]])``.
    The raw per-decision times travel in ``res["_times"]`` (dropped before JSON output).
    """
    g, base_seed, spec, opponent, vps_to_win, discard_limit = job[:6]
    trades = job[6] if len(job) > 6 else "off"
    opp_params = job[7] if len(job) > 7 else None
    seat = g % len(COLORS)
    seed = game_seed(base_seed, g)
    make = opponent_factory(resolve_opponent(opponent), opp_params, opponent)
    players = []
    opps: List[BenchOpponent] = []
    me = None
    for i, color in enumerate(COLORS):
        if i == seat:
            me = CatanbotPlayer(color, spec=spec, seed=seed, suppress_trades=(trades == "off"))
            players.append(me)
        else:
            o = BenchOpponent(make(color), trade_rule=trades, vps_to_win=vps_to_win)
            opps.append(o)
            players.append(o)
    res = play_game(players, seed=seed, vps_to_win=vps_to_win, discard_limit=discard_limit)
    assert me is not None
    res["game"] = g
    res["seat"] = seat
    res["our_vp"] = res["vps"][seat]
    res["opp_vps"] = [v for i, v in enumerate(res["vps"]) if i != seat]
    res["won"] = res["winner_seat"] == seat
    res["stats"] = dict(me.stats)
    res["unmapped_kinds"] = dict(me.unmapped_kinds)
    ot = {k: sum(o.trade_stats[k] for o in opps) for k in opps[0].trade_stats}
    st = me.stats
    res["trades"] = {
        "offers": int(st["offers"]), "offers_accepted": int(st["offers_accepted"]),
        "confirmed": int(st["trades_confirmed"]), "cancelled": int(st["trades_cancelled"]),
        "offers_received": int(st["offers_received"]), "accepted_by_us": int(st["offers_accepted_by_us"]),
        "opp_asked": ot["asked"], "opp_accepted": ot["accepted"], "opp_rejected": ot["rejected"],
        "opp_cannot_pay": ot["cannot_pay"], "opp_errors": ot["errors"],
    }
    opp_times = [t for o in opps for t in o.times]
    opp_choice = [t for o in opps for t in o.choice_times]
    res["timing"] = {
        "ours": timing_summary(me.times),
        "ours_choice": timing_summary(me.choice_times),
        "opp": timing_summary(opp_times),
        "opp_choice": timing_summary(opp_choice),
        "our_search_s": float(st["search_time"]),
        "opp_seat_s": [float(sum(o.times)) for o in opps],
    }
    res["_times"] = {"ours": list(me.times), "ours_choice": list(me.choice_times),
                     "opp": opp_times, "opp_choice": opp_choice}
    return res


STAT_KEYS = ["decisions", "searched", "trivial", "pending_robber", "pending_discard", "trade_prompts",
             "unmapped_top", "fallback", "errors", "observe_errors", "observed", "search_time",
             "offers", "offers_accepted", "trades_confirmed", "trades_cancelled", "offers_received",
             "offers_accepted_by_us", "self_offer_prompts"]
TRADE_KEYS = ["offers", "offers_accepted", "confirmed", "cancelled", "offers_received", "accepted_by_us",
              "opp_asked", "opp_accepted", "opp_rejected", "opp_cannot_pay", "opp_errors"]


def summarize(results: List[Dict[str, object]], spec: str, opponent: str, seed: int,
              trades: str = "off", opponent_params: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    n = len(results)
    wins = sum(1 for r in results if r["won"])
    seats = len(COLORS)
    by_seat = {s: [0, 0] for s in range(seats)}
    for r in results:
        by_seat[r["seat"]][1] += 1
        if r["won"]:
            by_seat[r["seat"]][0] += 1
    opp_avg = [sum(r["opp_vps"]) / len(r["opp_vps"]) for r in results]
    opp_best = [max(r["opp_vps"]) for r in results]
    stats = {k: sum(r["stats"].get(k, 0) for r in results) for k in STAT_KEYS}
    truncated = sum(1 for r in results if r["winner"] is None)
    unmapped: Dict[str, int] = {}
    for r in results:
        for k, v in r.get("unmapped_kinds", {}).items():
            unmapped[k] = unmapped.get(k, 0) + v
    trade_totals = {k: sum(r.get("trades", {}).get(k, 0) for r in results) for k in TRADE_KEYS}
    pooled: Dict[str, List[float]] = {"ours": [], "ours_choice": [], "opp": [], "opp_choice": []}
    for r in results:
        for k, xs in r.get("_times", {}).items():
            pooled.setdefault(k, []).extend(xs)
    opp_seats = seats - 1
    timing = {k: timing_summary(v) for k, v in pooled.items()}
    timing["our_s_per_game"] = timing["ours"]["total_s"] / n if n else 0.0
    timing["our_search_s_per_game"] = stats["search_time"] / n if n else 0.0
    timing["opp_s_per_game_per_seat"] = timing["opp"]["total_s"] / (n * opp_seats) if n else 0.0
    timing["our_decisions_per_game"] = timing["ours"]["n"] / n if n else 0.0
    timing["opp_decisions_per_game_per_seat"] = timing["opp"]["n"] / (n * opp_seats) if n else 0.0
    return {
        "spec": spec,
        "opponent": opponent,
        "opponent_class": resolve_opponent(opponent).__name__,
        "opponent_params": dict(opponent_params or {}),
        "trades": trades,
        "catanatron": CATANATRON_VERSION,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "seed": seed,
        "games": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "avg_vp": sum(r["our_vp"] for r in results) / n if n else 0.0,
        "avg_opp_vp": sum(opp_avg) / n if n else 0.0,
        "avg_best_opp_vp": sum(opp_best) / n if n else 0.0,
        "avg_turns": sum(r["turns"] for r in results) / n if n else 0.0,
        "avg_duration": sum(r["duration"] for r in results) / n if n else 0.0,
        "truncated": truncated,
        "by_seat": {s: {"wins": w, "games": g} for s, (w, g) in by_seat.items()},
        "stats": stats,
        "trade_totals": trade_totals,
        "timing": timing,
        "unmapped_kinds": unmapped,
        "results": results,
    }


def _timing_row(label: str, t: Dict[str, float], c: Dict[str, float], games: int, seats: int, s_per_game: float) -> str:
    per = max(1, games * seats)
    return (f"    {label:<34} {t['n'] / per:8.1f} {t['mean_ms']:9.2f} {t['p95_ms']:9.2f}   | "
            f"{c['n'] / per:8.1f} {c['mean_ms']:9.2f} {c['p95_ms']:9.2f}   | {s_per_game:9.2f}")


def print_summary(s: Dict[str, object], wall: float) -> None:
    n = max(1, int(s["games"]))
    st = s["stats"]
    params = s.get("opponent_params") or {}
    ptxt = f' ({",".join(f"{k}={v}" for k, v in params.items())})' if params else ""
    print(f'catanbot "{s["spec"]}" vs 3 x {s["opponent_class"]}{ptxt}: {s["games"]} games, seat rotation, '
          f'seed {s["seed"]} (catanatron {s.get("catanatron", CATANATRON_VERSION)}, trades {s.get("trades", "off")}, '
          f'PYTHONHASHSEED={s.get("hash_seed")})')
    print(f'  wins        : {s["wins"]}/{s["games"]} = {100.0 * s["win_rate"]:.1f}%   (a random seat wins 25%)')
    print(f'  avg VP      : catanbot {s["avg_vp"]:.2f} | opponents {s["avg_opp_vp"]:.2f} (best opponent {s["avg_best_opp_vp"]:.2f})')
    seats = ", ".join(f'seat{k} {v["wins"]}/{v["games"]}' for k, v in s["by_seat"].items())
    print(f"  by seat     : {seats}")
    print(f'  turns/game  : {s["avg_turns"]:.1f} (catanatron num_turns; {s["truncated"]} game(s) hit the {TURNS_LIMIT}-turn cap)')
    print(f'  time/game   : {s["avg_duration"]:.2f} s ({st["search_time"] / n:.2f} s in search, '
          f'{st["searched"] / n:.1f} searched + {st["trivial"] / n:.1f} trivial decisions per game); wall {wall:.1f} s')
    extra = ""
    if st.get("pending_discard"):
        extra += f', {int(st["pending_discard"])} planned discard cards'
    if st.get("trade_prompts"):
        how = "declined" if s.get("trades", "off") == "off" else "answered"
        extra += f', {int(st["trade_prompts"])} trade prompts {how}'
    print(f'  adapter     : {int(st["errors"])} errors, {int(st["fallback"])} fallbacks, '
          f'{int(st["unmapped_top"])} unmapped top actions, {int(st["observe_errors"])} observe errors, '
          f'{st["observed"] / n:.0f} observed actions/game{extra}'
          + (f'; unmapped top actions: {s["unmapped_kinds"]}' if s["unmapped_kinds"] else ""))
    tt = s.get("trade_totals") or {}
    if s.get("trades", "off") != "off" and tt:
        answerable = tt["opp_accepted"] + tt["opp_rejected"]
        rate = 100.0 * tt["opp_accepted"] / answerable if answerable else 0.0
        print(f'  trades      : catanbot offered {tt["offers"] / n:.1f}/game, {tt["offers_accepted"] / n:.1f} accepted by '
              f'>= 1 seat, {tt["confirmed"] / n:.1f} executed, {tt["cancelled"] / n:.1f} cancelled; '
              f'opponents ({s["trades"]}) accepted {tt["opp_accepted"]}/{answerable} answerable offers ({rate:.1f}%), '
              f'{tt["opp_cannot_pay"]} could not pay, {tt["opp_errors"]} opponent errors'
              + (f'; offers received {tt["offers_received"]} (accepted {tt["accepted_by_us"]})'
                 if tt["offers_received"] else ""))
    tm = s.get("timing")
    if tm:
        print("  compute     : per decision                  decisions/game   mean ms    p95 ms   | "
              "choices/g   mean ms    p95 ms   | s/game/seat")
        print(_timing_row("catanbot", tm["ours"], tm["ours_choice"], n, 1, tm["our_s_per_game"]))
        print(_timing_row(f'{s["opponent_class"]} (each of {len(COLORS) - 1})', tm["opp"], tm["opp_choice"], n,
                          len(COLORS) - 1, tm["opp_s_per_game_per_seat"]))
        print(f'    (catanbot search alone {tm["our_search_s_per_game"]:.2f} s/game; "choices" = decisions with '
              f'more than one playable action)')
    print("  markdown    : | `%s` | %s | %d | %d | %.0f%% | %.2f | %.2f | %.1f | %.2f |" % (
        s["spec"], s["opponent_class"], s["games"], s["wins"], 100.0 * s["win_rate"], s["avg_vp"],
        s["avg_opp_vp"], s["avg_turns"], s["avg_duration"]))


def _strip_raw(obj):
    """Drop the raw per-decision time lists (``_times``) from summaries before JSON output."""
    if isinstance(obj, list):
        return [_strip_raw(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_raw(v) for k, v in obj.items() if k != "_times"}
    return obj


# ---------------------------------------------------------------------------
# --probe-trades: how does each opponent answer an offer?
# ---------------------------------------------------------------------------
PROBE_CATEGORIES = ("1:1", "2:1", "1:2")   # cards the responder receives : cards it pays


def probe_positions(count: int, seed: int, min_turn: int = 20, max_turn: int = 90, every: int = 7) -> List:
    """``count`` mid-game positions: copies of catanatron games between four catanatron
    ``ValueFunctionPlayer`` (``WeightedRandomPlayer`` where absent) taken at post-roll
    ``PLAY_TURN`` prompts of turns ``min_turn``..``max_turn``, every ``every``-th such prompt."""
    from catanatron.models.enums import ActionPrompt
    from catanbot.bench.catanatron_adapter import make_game
    try:
        from catanatron.players.value import ValueFunctionPlayer as Gen
    except ImportError:   # 3.2.1
        Gen = WeightedRandomPlayer
    out = []
    g = 0
    while len(out) < count and g < 20 * count + 10:
        g += 1
        game = make_game([Gen(c) for c in COLORS], seed=seed * 7919 + g)
        k = 0
        while game.winning_color() is None and game.state.num_turns < max_turn and len(out) < count:
            st = game.state
            seat = st.current_turn_index
            if (st.current_prompt == ActionPrompt.PLAY_TURN and st.num_turns >= min_turn
                    and st.player_state[f"P{seat}_HAS_ROLLED"] and not st.is_road_building
                    and st.current_player_index == seat):
                k += 1
                if k % every == 0:
                    out.append(game.copy())
            game.play_tick()
    return out


def probe_offer(game, category: str, rng: random.Random) -> Optional[Tuple[int, ...]]:
    """A random ``OFFER_TRADE`` value (10 counts) from the turn player to the first seat catanatron
    asks, in ``category`` (responder receives : pays), that both sides can pay; ``None`` if none exists."""
    from catanatron.state_functions import get_player_freqdeck
    st = game.state
    offerer = st.current_turn_index
    responder = 0 if offerer != 0 else 1
    have_o = get_player_freqdeck(st, st.colors[offerer])
    have_r = get_player_freqdeck(st, st.colors[responder])
    n_recv, n_pay = {"1:1": (1, 1), "2:1": (2, 1), "1:2": (1, 2)}[category]
    for _ in range(60):
        pay_types = [r for r in range(5) if have_r[r] > 0]
        if not pay_types:
            return None
        pay = [0] * 5
        for _i in range(n_pay):
            opts = [r for r in pay_types if have_r[r] > pay[r]]
            if not opts:
                break
            pay[rng.choice(opts)] += 1
        if sum(pay) != n_pay:
            continue
        recv = [0] * 5
        for _i in range(n_recv):
            opts = [r for r in range(5) if have_o[r] > recv[r] and pay[r] == 0]
            if not opts:
                break
            recv[rng.choice(opts)] += 1
        if sum(recv) != n_recv:
            continue
        # OFFER_TRADE value: what the offerer gives (= responder receives), then what it asks
        return tuple(recv) + tuple(pay)
    return None


def run_probe(opponents: Sequence[str], n: int, seed: int, opp_params: Dict[str, str],
              vps_to_win: int = 10) -> Dict[str, object]:
    """Offer every opponent the same ``n`` sampled trades per category; tabulate the answers."""
    from catanatron.models.enums import Action as CAction, ActionPrompt, ActionType
    rng = random.Random(seed)
    positions = probe_positions(max(8, (n + 1) // 2), seed)
    samples = []   # (position index, category, offer value)
    for cat in PROBE_CATEGORIES:
        got = 0
        tries = 0
        while got < n and tries < 20 * n:
            tries += 1
            pi = rng.randrange(len(positions))
            offer = probe_offer(positions[pi], cat, rng)
            if offer is not None:
                samples.append((pi, cat, offer))
                got += 1
    table: Dict[str, object] = {}
    for name in opponents:
        try:
            make = opponent_factory(resolve_opponent(name), opp_params, name)
        except OpponentUnavailable as ex:
            print(f"skipping {name}: {ex.message}")
            continue
        rows = {cat: {"n": 0, "accept": 0, "reject": 0, "error": 0, "value_rule": 0, "fair_rule": 0,
                      "seconds": 0.0} for cat in PROBE_CATEGORIES}
        errors: Dict[str, int] = {}
        for pi, cat, offer in samples:
            game = positions[pi].copy()
            st = game.state
            offerer = st.colors[st.current_turn_index]
            game.execute(CAction(offerer, ActionType.OFFER_TRADE, offer))
            assert st.current_prompt == ActionPrompt.DECIDE_TRADE
            responder = st.colors[st.current_player_index]
            playable = list(game.playable_actions)
            assert {a.action_type for a in playable} == {ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE}
            player = make(responder)
            row = rows[cat]
            row["n"] += 1
            t0 = time.perf_counter()
            try:
                a = player.decide(game, playable)
                row["accept" if a.action_type == ActionType.ACCEPT_TRADE else "reject"] += 1
            except Exception as ex:   # noqa: BLE001 - catanatron's search players raise on trade actions
                row["error"] += 1
                key = f"{type(ex).__name__}: {ex}"[:120]
                errors[key] = errors.get(key, 0) + 1
            row["seconds"] += time.perf_counter() - t0
            for rule in ("value", "fair"):
                wrapped = BenchOpponent(player, trade_rule=rule, vps_to_win=vps_to_win)
                if wrapped.rule_accepts(game):
                    row[f"{rule}_rule"] += 1
        table[name] = {"class": resolve_opponent(name).__name__, "rows": rows, "errors": errors}
    return {"catanatron": CATANATRON_VERSION, "n_per_category": n, "seed": seed, "positions": len(positions),
            "opponent_params": opp_params, "table": table}


def print_probe(p: Dict[str, object]) -> None:
    print(f"== trade-response probe: {p['n_per_category']} offers per category from {p['positions']} mid-game "
          f"positions (catanatron {p['catanatron']}); columns: accept % native [errors] | value rule | fair rule")
    head = "  ".join(f"{c + ' (they receive:pay)':<30}" for c in PROBE_CATEGORIES)
    print(f"  {'opponent':<42} {head}  ms/answer")
    for name, entry in p["table"].items():
        cells = []
        secs = 0.0
        cnt = 0
        for cat in PROBE_CATEGORIES:
            r = entry["rows"][cat]
            m = max(1, r["n"])
            cells.append(f"{100.0 * r['accept'] / m:5.1f}% [{r['error']:3d}] | {100.0 * r['value_rule'] / m:5.1f}% | "
                         f"{100.0 * r['fair_rule'] / m:5.1f}%")
            secs += r["seconds"]
            cnt += r["n"]
        print(f"  {name + ' (' + entry['class'] + ')':<42} " + "  ".join(f"{c:<30}" for c in cells)
              + f"  {1000.0 * secs / max(1, cnt):8.1f}")
        for msg, k in entry["errors"].items():
            print(f"      {k} x {msg}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _reexec_with_hash_seed(hash_seed: int) -> None:
    """Re-run this script with ``PYTHONHASHSEED=hash_seed`` unless already set so (reproducible games)."""
    if hash_seed < 0 or os.environ.get("PYTHONHASHSEED") == str(hash_seed):
        return
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)


def main(argv=None, reexec: bool = False) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=20, help="number of games (default 20)")
    ap.add_argument("--opponent", default="vp",
                    help="opponent preset (vp, weighted, random, vf, ab, value, alphabeta, sameturn, playouts, mcts), "
                         "an import path module:Class, or a comma-separated list (default vp)")
    ap.add_argument("--opponent-params", default=None, metavar="KEY=VAL,...",
                    help="constructor parameters for every opponent (3.3 Params fields, e.g. depth=3 for alphabeta, "
                         "num_simulations=50 for mcts, num_playouts=10 for playouts; or __init__ keywords)")
    ap.add_argument("--ladder", choices=sorted(LADDERS), default=None,
                    help="run a preset ladder of opponents (controls | standins | strong | full); "
                         "opponents the running catanatron does not ship are skipped")
    ap.add_argument("--list-opponents", action="store_true",
                    help="print every preset with its class and whether it is available on the running catanatron")
    ap.add_argument("--spec", default=DEFAULT_SPEC, help=f'catanbot bot spec (default "{DEFAULT_SPEC}")')
    ap.add_argument("--seed", type=int, default=0, help="base seed (default 0)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    ap.add_argument("--vps-to-win", type=int, default=10, help="victory points needed (default 10)")
    ap.add_argument("--discard-limit", type=int, default=7,
                    help="catanatron discard limit on a 7 (default 7; on 3.2.1 only the first discarder honours it, "
                         "see docs/BENCHMARKS.md limitation 12)")
    ap.add_argument("--trades", choices=TRADE_MODES, default="off",
                    help="domestic trading (catanatron 3.3 only): off (default, catanbot never offers) | native "
                         "(opponents answer with their own decide) | value (opponents accept iff their value "
                         "function rises) | fair (value + no n-for-fewer, no deals with a player 2 VP from winning)")
    ap.add_argument("--hash-seed", type=int, default=0,
                    help="re-run with PYTHONHASHSEED=N so games are reproducible across processes (default 0; "
                         "-1 keeps random hashing); only when run as a script")
    ap.add_argument("--probe-trades", type=int, default=0, metavar="N",
                    help="instead of games: offer each --opponent N sampled trades per category (1:1, 2:1, 1:2) "
                         "from mid-game states and tabulate their answers (3.3 only)")
    ap.add_argument("--json", default=None, help="write the full results to this JSON file")
    ap.add_argument("--verbose", action="store_true", help="print one line per game")
    args = ap.parse_args(argv)

    if reexec:
        _reexec_with_hash_seed(args.hash_seed)

    if args.list_opponents:
        print_opponents()
        return 0
    try:
        opp_params = parse_opponent_params(args.opponent_params)
    except OpponentUnavailable as ex:
        print(ex.message, file=sys.stderr)
        return 2
    if (args.trades != "off" or args.probe_trades) and not DOMESTIC_TRADING:
        print(f"domestic trading (--trades {args.trades} / --probe-trades) needs catanatron 3.3; "
              f"catanatron {CATANATRON_VERSION} has no player-to-player trades", file=sys.stderr)
        return 2

    opponents = LADDERS[args.ladder] if args.ladder else [o.strip() for o in args.opponent.split(",") if o.strip()]
    single = args.ladder is None and len(opponents) == 1

    if args.probe_trades:
        probe = run_probe(opponents, args.probe_trades, args.seed, opp_params, args.vps_to_win)
        print_probe(probe)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(probe, fh, indent=1, default=str)
            print(f"  json        : {args.json}")
        return 0 if probe["table"] else 2

    summaries: List[Dict[str, object]] = []
    for opponent in opponents:
        try:
            cls = resolve_opponent(opponent)
            opponent_factory(cls, opp_params, opponent)
        except OpponentUnavailable as ex:
            if single:
                print(ex.message, file=sys.stderr)
                return 2
            print(f"skipping {opponent}: {ex.message}")
            continue
        ptxt = f" [{args.opponent_params}]" if opp_params else ""
        print(f"== catanbot [{args.spec}] vs 3x {cls.__name__} ({opponent}){ptxt}, {args.games} games, "
              f"catanatron {CATANATRON_VERSION}, trades {args.trades}, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")
        jobs = [(g, args.seed, args.spec, opponent, args.vps_to_win, args.discard_limit, args.trades, opp_params)
                for g in range(args.games)]
        results: List[Dict[str, object]] = []
        t0 = time.perf_counter()

        def report(r: Dict[str, object]) -> None:
            results.append(r)
            if args.verbose:
                tr = r.get("trades", {})
                ttxt = (f' offers={tr.get("offers", 0)} traded={tr.get("confirmed", 0)}'
                        if args.trades != "off" else "")
                print(f'  game {r["game"]:3d} seat {r["seat"]} seed {r["seed"]}: '
                      f'{"WIN " if r["won"] else "loss"} vp={r["vps"]} turns={r["turns"]} {r["duration"]:.1f}s{ttxt}',
                      flush=True)

        if args.workers > 1 and len(jobs) > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=args.workers) as pool:
                for r in pool.imap_unordered(run_one, jobs):
                    report(r)
        else:
            for job in jobs:
                report(run_one(job))
        results.sort(key=lambda r: r["game"])
        wall = time.perf_counter() - t0
        summary = summarize(results, args.spec, opponent, args.seed, args.trades, opp_params)
        summary["wall_time"] = wall
        print_summary(summary, wall)
        summaries.append(summary)
    if len(summaries) > 1:
        print(f"\n== ladder summary (win rate of catanbot in 4-player games; 25% = seat baseline; "
              f"catanatron {CATANATRON_VERSION})")
        for sm in summaries:
            print(f"  {sm['opponent_class']:<26} {sm['win_rate'] * 100:5.1f}%  avg VP {sm['avg_vp']:.2f} vs {sm['avg_opp_vp']:.2f}  ({sm['games']} games)")
    if args.json:
        with open(args.json, "w") as fh:
            out = summaries if len(summaries) > 1 else (summaries[0] if summaries else {})
            json.dump(_strip_raw(out), fh, indent=1, default=str)
        print(f"  json        : {args.json}")
    if not summaries:
        print(f"no opponent could be played on catanatron {CATANATRON_VERSION} (see --list-opponents)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(reexec=True))
