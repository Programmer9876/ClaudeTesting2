"""Registry of tunable strategy constants ("alpha strategies") and run-time overrides.

Every strategy module keeps its knobs as module-level constants (``danger.TURNS_HALF``,
``coalitions.SCALE``, ...).  This module lists them in :data:`TUNABLES` with their real
default, the candidate values worth testing and how to install an override, so that
``scripts/ablate.py`` can play *paired* games (the same bot with the constant at a
candidate value vs. at its default) and report whether the change is worth its cost.

How an override is installed
----------------------------
* ``kind="weight"`` / ``kind="flag"``: :func:`apply` sets every :class:`Target` of the
  tunable.  A target is a module attribute (``catanbot.danger.TURNS_HALF``), a copy of it
  at an import site (``robber.danger_multiplier`` is the name ``robber.py`` bound with
  ``from .danger import danger_multiplier``; setting only ``danger.danger_multiplier``
  would change nothing there) or the *default value of a function parameter* when the
  constant was bound at definition time (``def decay(self, factor=DECAY)`` freezes the
  value of ``DECAY``; the override rewrites ``decay.__defaults__``).  List-valued
  constants (``placement.RESOURCE_DEMAND``) are mutated in place because every import
  site shares the same list object.  :func:`apply` returns the previous values;
  :func:`restore` puts them back in reverse order, so overrides nest.
* ``kind="search"``: a ``SearchConfig`` field.  Nothing global is touched: the value is
  set on the bot's own ``config`` object (:func:`apply_to_bot`), which is what the bot spec
  ``search:depth=...,beam=...`` would have done.  Search knobs only exist on the search bot.
* ``needs_python_evaluator``: the constant is (also) read by ``heuristic.static_value``,
  whose C++ port (``cpp/heuristic.cpp``) reads no Python constants.  Such a tunable only
  takes effect with ``CATANBOT_NO_ACCEL=1`` set *before* ``catanbot`` is imported;
  ``scripts/ablate.py`` re-executes itself with that variable when needed.

Cross-talk between the two sides of a paired game is prevented by applying the
overrides only inside the candidate bot's own hooks (see ``agents/param_bot.py``) and by
clearing ``danger``'s win-path cache whenever a tunable that feeds it changes
(``clear_caches``).

Documented monkeypatches (the source files are not edited):
* ``danger.danger_multiplier`` flag off: ``danger.danger_multiplier`` and
  ``robber.danger_multiplier`` become ``lambda wp: 1.0`` so ``robber.target_weight`` uses
  the VP threat alone.
* ``opponent_model.stage_late_drop``: ``trade_stage_factor`` (in ``opponent_model``,
  ``search`` and ``trading``) is replaced by ``1 - drop * game_stage(state) ** 1.5``;
  the default 0.7 reproduces the original function exactly.
* ``trading.feed_leader_guard`` flag off: ``offer_is_feeding_leader`` (in ``trading`` and
  ``heuristic``) always answers ``(False, "")``.
* ``trading.accept_margin``: the default of ``should_accept``'s ``margin`` parameter.
"""
from __future__ import annotations

import importlib
import inspect
import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

KEEP = object()   # make(value) returns it when the candidate value means "leave the original in place"

DEFAULT_CHEAP_SPEC = "heuristic:temp=0.15"
DEFAULT_SEARCH_SPEC = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
DEFAULT_DEEP_SPEC = "search:depth=2,beam=4,expand=8,evaluator=heuristic"   # for knobs only depth >= 2 reads


def spec_depth(spec: str) -> int:
    """Search depth a bot spec produces (1 for the heuristic bot / an unspecified depth)."""
    kind, _, rest = spec.partition(":")
    if kind.strip() != "search":
        return 1
    for item in rest.split(","):
        k, _, v = item.partition("=")
        if k.strip() == "depth" and v.strip():
            return int(float(v))
    return 1


# ---------------------------------------------------------------------------
# Targets: where an override is written
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    module: str                 # dotted module name, e.g. "catanbot.danger"
    attr: str                   # attribute, or dotted path to a function ("PoliticalState.decay")
    param: Optional[str] = None  # when set: the default value of this parameter of the function at ``attr``

    def __str__(self) -> str:
        s = f"{self.module.split('.')[-1]}.{self.attr}"
        return f"{s}({self.param}=)" if self.param else s

    def owner(self):
        """The object holding the attribute and the attribute's last name component."""
        obj: Any = importlib.import_module(self.module)
        parts = self.attr.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]

    def get(self) -> Any:
        obj, name = self.owner()
        value = getattr(obj, name)
        if self.param is None:
            return list(value) if isinstance(value, list) else value
        return _get_default(value, self.param)

    def set(self, value: Any) -> None:
        obj, name = self.owner()
        if self.param is not None:
            _set_default(getattr(obj, name), self.param, value)
            return
        current = getattr(obj, name)
        if isinstance(current, list) and isinstance(value, (list, tuple)):
            current[:] = list(value)     # in place: every ``from x import LIST`` site shares the object
        else:
            setattr(obj, name, value)


def _function_of(fn):
    return getattr(fn, "__func__", fn)


# (function, parameter) -> ("kw", None) or ("pos", index into __defaults__).  ``inspect.signature``
# costs ~100 us; ParamBot applies and restores around every hook, so the slot is resolved once.
_DEFAULT_SLOTS: Dict[Tuple[int, str], Tuple[str, Optional[int]]] = {}


def _default_slot(fn, param: str) -> Tuple[str, Optional[int]]:
    key = (id(fn), param)
    slot = _DEFAULT_SLOTS.get(key)
    if slot is None:
        sig = inspect.signature(fn)
        p = sig.parameters[param]
        if p.default is inspect.Parameter.empty:
            raise ValueError(f"{fn} parameter {param!r} has no default")
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            slot = ("kw", None)
        else:
            positional = [n for n, q in sig.parameters.items()
                          if q.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            n_defaults = len(fn.__defaults__ or ())
            with_defaults = positional[len(positional) - n_defaults:]
            slot = ("pos", with_defaults.index(param))
        _DEFAULT_SLOTS[key] = slot
    return slot


def _get_default(fn, param: str) -> Any:
    fn = _function_of(fn)
    kind, idx = _default_slot(fn, param)
    if kind == "kw":
        return (fn.__kwdefaults__ or {})[param]
    return fn.__defaults__[idx]


def _set_default(fn, param: str, value: Any) -> None:
    fn = _function_of(fn)
    kind, idx = _default_slot(fn, param)
    if kind == "kw":
        kw = dict(fn.__kwdefaults__ or {})
        kw[param] = value
        fn.__kwdefaults__ = kw
        return
    defaults = list(fn.__defaults__)
    defaults[idx] = value
    fn.__defaults__ = tuple(defaults)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
def _parse_float(s: str) -> float:
    return float(s)


def _parse_int(s: str) -> int:
    return int(float(s))


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "on", "true", "yes", "default"):
        return True
    if v in ("0", "off", "false", "no"):
        return False
    raise ValueError(f"flag value must be on/off, got {s!r}")


def _parse_vector(s: str) -> List[float]:
    parts = [x for x in s.replace(":", "/").split("/") if x.strip()]
    if len(parts) != 5:
        raise ValueError(f"resource vector needs 5 values separated by '/', got {s!r}")
    return [float(x) for x in parts]


@dataclass
class Tunable:
    name: str
    module: str
    attr: str
    default: Any
    kind: str = "weight"                       # "weight" | "flag" | "search"
    candidates: List[Any] = field(default_factory=list)
    needs_python_evaluator: bool = False
    description: str = ""
    targets: Tuple[Target, ...] = ()           # everything apply() sets (default: the module attribute itself)
    make: Optional[Callable[[Any], Any]] = None  # candidate value -> object installed at the targets
    requires_search: bool = False              # has no effect on the heuristic bot (evaluator / search knob)
    requires_depth: int = 1                    # search knobs the searcher only reads at this depth or more
    clear_caches: bool = False                 # clear danger's win-path cache around apply / restore
    spec_key: Optional[str] = None             # search kind: equivalent selfplay bot-spec key, if any
    parse: Callable[[str], Any] = _parse_float

    def __post_init__(self) -> None:
        if not self.targets and self.kind != "search":
            self.targets = (Target(self.module, self.attr),)

    @property
    def qualname(self) -> str:
        return f"{self.module.split('.')[-1]}.{self.attr}"

    def installed_value(self, value: Any) -> Any:
        """The object that goes into the targets for candidate ``value`` (``KEEP`` = nothing)."""
        if self.kind == "flag" and bool(value):
            return KEEP
        return self.make(value) if self.make is not None else value

    def format(self, value: Any) -> str:
        if self.kind == "flag":
            return "on" if bool(value) else "off"
        if isinstance(value, (list, tuple)):
            return "/".join(f"{x:g}" for x in value)
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)


def _normalised(v: Sequence[float]) -> List[float]:
    m = sum(v) / len(v)
    return [x / m for x in v]


def _make_stage_factor(drop: float):
    from .opponent_model import game_stage

    def trade_stage_factor(state) -> float:
        return 1.0 - drop * (game_stage(state) ** 1.5)

    trade_stage_factor.__doc__ = f"trade_stage_factor with late drop {drop} (tuning override)"
    return trade_stage_factor


def _no_danger_multiplier(wp) -> float:
    return 1.0


def _no_feed_leader_guard(state, giver, receiver, give):
    return False, ""


def _scaled(default: float, factors: Sequence[float]) -> List[float]:
    return [round(default * f, 6) for f in factors]


def _build_registry() -> Dict[str, Tunable]:
    from . import coalitions, danger, devcards, heuristic, opponent_model, placement, politics, robber, trading
    from .selfplay import make_bot

    live = lambda mod, attr: Target(mod.__name__, attr).get()   # noqa: E731  (the real default, read now)
    reg: List[Tunable] = []

    def add(**kw) -> None:
        reg.append(Tunable(**kw))

    # --- danger (distance-to-win model) --------------------------------------------------
    add(name="danger.TURNS_HALF", module=danger.__name__, attr="TURNS_HALF", default=live(danger, "TURNS_HALF"),
        candidates=[1.5, 2.0, 4.5, 6.0], clear_caches=True,
        description="turns-to-win at which danger = 0.5 (steeper = only near-winners count as dangerous)")
    add(name="danger.BLOCK_FLOOR", module=danger.__name__, attr="BLOCK_FLOOR", default=live(danger, "BLOCK_FLOOR"),
        candidates=[0.0, 0.2, 0.5, 0.7], clear_caches=True,
        description="block_factor for a resource the target does not need")
    add(name="danger.BLOCK_NEED", module=danger.__name__, attr="BLOCK_NEED", default=live(danger, "BLOCK_NEED"),
        candidates=[0.0, 1.0, 3.0, 4.0], clear_caches=True,
        description="extra block_factor for a fully needed resource")
    add(name="danger.danger_multiplier", module=danger.__name__, attr="danger_multiplier", default=True, kind="flag",
        candidates=[False], parse=_parse_bool, clear_caches=True,
        targets=(Target(danger.__name__, "danger_multiplier"), Target(robber.__name__, "danger_multiplier")),
        make=lambda on: KEEP if on else _no_danger_multiplier,
        description="off: robber.target_weight uses the VP threat only (multiplier fixed at 1.0 instead of "
                    "0.4 + 1.6 x danger); implemented by monkeypatching danger.danger_multiplier and the copy "
                    "robber.py imported")
    # --- static evaluator ------------------------------------------------------------------
    add(name="heuristic.EXPOSURE_WEIGHT", module=heuristic.__name__, attr="EXPOSURE_WEIGHT",
        default=live(heuristic, "EXPOSURE_WEIGHT"), candidates=[0.0, 0.1, 0.4, 0.6],
        needs_python_evaluator=True, requires_search=True,
        description="weight of robber.steal_exposure_fast in static_value (only the search bot evaluates; "
                    "needs the Python evaluator)")
    # --- coalitions --------------------------------------------------------------------------
    add(name="coalitions.SCALE", module=coalitions.__name__, attr="SCALE", default=live(coalitions, "SCALE"),
        candidates=[0.25, 1.0, 2.0], description="EV sacrifice (card-value units) that counts as one full signal")
    add(name="coalitions.BLOC_THRESHOLD", module=coalitions.__name__, attr="BLOC_THRESHOLD",
        default=live(coalitions, "BLOC_THRESHOLD"), candidates=[0.5, 2.0, 100.0],
        targets=(Target(coalitions.__name__, "BLOC_THRESHOLD"),
                 Target(coalitions.__name__, "CoalitionDetector.allies", "threshold"),
                 Target(coalitions.__name__, "CoalitionDetector.blocs", "threshold"),
                 Target(coalitions.__name__, "CoalitionDetector.against", "threshold")),
        description="pairwise strength above which two players are treated as a bloc (100 = never); also "
                    "rewrites the def-time defaults of allies/blocs/against")
    add(name="coalitions.DECAY", module=coalitions.__name__, attr="DECAY", default=live(coalitions, "DECAY"),
        candidates=[0.9, 0.99],
        targets=(Target(coalitions.__name__, "DECAY"), Target(coalitions.__name__, "CoalitionDetector.decay", "factor")),
        description="per-turn decay of coalition signals (def-time default of CoalitionDetector.decay)")
    # --- politics ----------------------------------------------------------------------------
    add(name="politics.MAX_SLACK", module=politics.__name__, attr="MAX_SLACK", default=live(politics, "MAX_SLACK"),
        candidates=[0.0, 0.15, 0.6], description="cap on the favour slack a friend gets in should_accept")
    add(name="politics.BASELINE", module=politics.__name__, attr="BASELINE", default=live(politics, "BASELINE"),
        candidates=[0.0, 0.3],
        targets=(Target(politics.__name__, "BASELINE"), Target(politics.__name__, "PoliticalState.__init__", "baseline")),
        description="political capital everyone starts with (def-time default of PoliticalState.__init__)")
    add(name="politics.DECAY", module=politics.__name__, attr="DECAY", default=live(politics, "DECAY"),
        candidates=[0.9, 0.99],
        targets=(Target(politics.__name__, "DECAY"), Target(politics.__name__, "PoliticalState.decay", "factor")),
        description="per-turn decay of political capital towards the baseline")
    # --- opponent model ----------------------------------------------------------------------
    add(name="opponent_model.DECAY", module=opponent_model.__name__, attr="DECAY", default=live(opponent_model, "DECAY"),
        candidates=[0.7, 0.97],
        targets=(Target(opponent_model.__name__, "DECAY"), Target(opponent_model.__name__, "_EW.add", "decay")),
        description="per-observation decay of the opponent statistics (also _EW.add's def-time default)")
    add(name="opponent_model.VALUE_LR", module=opponent_model.__name__, attr="VALUE_LR",
        default=live(opponent_model, "VALUE_LR"), candidates=[0.0, 0.06, 0.25],
        description="learning rate of the opponents' implied resource valuations (0 = never learn)")
    add(name="opponent_model.stage_late_drop", module=opponent_model.__name__, attr="trade_stage_factor", default=0.7,
        candidates=[0.0, 0.4, 0.85],
        targets=(Target(opponent_model.__name__, "trade_stage_factor"), Target("catanbot.search", "trade_stage_factor"),
                 Target(trading.__name__, "trade_stage_factor")),
        make=_make_stage_factor,
        description="trade_stage_factor = 1 - drop x game_stage^1.5 (0 = trade willingness never drops late); "
                    "implemented by replacing the function at its three import sites")
    # --- trading -----------------------------------------------------------------------------
    add(name="trading.accept_margin", module=trading.__name__, attr="should_accept",
        default=Target(trading.__name__, "should_accept", "margin").get(), candidates=[0.0, 0.01, 0.03],
        targets=(Target(trading.__name__, "should_accept", "margin"),),
        description="base win-probability gain should_accept demands (the margin parameter's default)")
    add(name="trading.feed_leader_guard", module=trading.__name__, attr="offer_is_feeding_leader", default=True,
        kind="flag", candidates=[False], parse=_parse_bool,
        targets=(Target(trading.__name__, "offer_is_feeding_leader"), Target(heuristic.__name__, "offer_is_feeding_leader")),
        make=lambda on: KEEP if on else _no_feed_leader_guard,
        description="off: the don't-feed-the-leader rule never fires (offer_is_feeding_leader patched in trading "
                    "and heuristic; search/opponent_model import it lazily from trading)")
    # --- placement ---------------------------------------------------------------------------
    add(name="placement.RESOURCE_DEMAND", module=placement.__name__, attr="RESOURCE_DEMAND",
        default=live(placement, "RESOURCE_DEMAND"),
        candidates=[[1.0] * 5, _normalised([0.9, 0.9, 0.8, 1.4, 1.4]), _normalised([1.15, 1.15, 0.9, 1.0, 1.0])],
        needs_python_evaluator=True, make=_normalised, parse=_parse_vector,
        description="per-resource demand weights wood/brick/sheep/wheat/ore (mean 1; values a/b/c/d/e); "
                    "mutated in place, also read by static_value so it needs the Python evaluator")
    for attr, desc in (("PLACEMENT_BLOCK_WEIGHT", "weight of the robber-exposure penalty in settlement scoring"),
                       ("PLACEMENT_ROBBER_Q", "probability scale of the robber landing on a strong hex")):
        if hasattr(placement, attr):     # being added by another change; skip gracefully when absent
            d = live(placement, attr)
            add(name=f"placement.{attr}", module=placement.__name__, attr=attr, default=d,
                candidates=_scaled(d, [0.0, 0.5, 2.0]), description=desc)
    # --- dev cards ---------------------------------------------------------------------------
    add(name="devcards.KNIGHT_VALUE", module=devcards.__name__, attr="KNIGHT_VALUE", default=live(devcards, "KNIGHT_VALUE"),
        candidates=[0.3, 0.8], description="VP-equivalent value of a drawn knight in should_buy_dev")
    add(name="devcards.MONOPOLY_BASE_VALUE", module=devcards.__name__, attr="MONOPOLY_BASE_VALUE",
        default=live(devcards, "MONOPOLY_BASE_VALUE"), candidates=[0.3, 0.9],
        description="base VP-equivalent value of a drawn monopoly")
    # --- search knobs (SearchConfig fields; defaults are what the default search spec produces) ---
    # The opponents' turns (Searcher._future_values) are only simulated with depth >= 2: the four
    # opponent knobs do nothing at depth 1, so their ablation needs a depth-2 base spec (the script
    # picks DEFAULT_DEEP_SPEC and refuses a shallower one).
    cfg = make_bot(DEFAULT_SEARCH_SPEC).config
    for attr, key, cands, depth, desc in (
            ("trade_proposals", "trades", [0, 1, 5], 1, "PROPOSE_TRADE candidates per node (0 = never propose)"),
            ("dump_candidates", None, [0, 1, 5], 1, "surplus dumps tried with > 7 cards (0 = off)"),
            ("opponent_proposals", None, [0, 2], 2,
             "proposals a simulated opponent may make per turn (0 = never); depth >= 2 only"),
            ("opponent_actions", None, [2, 6], 2, "greedy actions per simulated opponent turn; depth >= 2 only"),
            ("opponent_expand", None, [3, 10], 2, "candidates evaluated per simulated opponent decision; depth >= 2 only"),
            ("opp_roll_samples", "opprolls", [1, 2, 6], 2, "sampled roll sequences for the opponents' turns; depth >= 2 only"),
            ("beam", "beam", [2, 6, 8], 1, "partial sequences kept per level"),
            ("expand", "expand", [4, 12, 16], 1, "actions tried per decision node"),
            ("depth", "depth", [2], 1, "turns of lookahead (2 = + opponents' turns)")):
        add(name=f"search.{attr}", module="catanbot.search", attr=attr, default=getattr(cfg, attr), kind="search",
            candidates=cands, requires_search=True, requires_depth=depth, spec_key=key, parse=_parse_int,
            description=desc)
    return {t.name: t for t in reg}


TUNABLES: Dict[str, Tunable] = _build_registry()


def find(name: str) -> Tunable:
    """Registry lookup by full name (``danger.TURNS_HALF``) or by a unique attribute name (``TURNS_HALF``)."""
    if name in TUNABLES:
        return TUNABLES[name]
    hits = [t for t in TUNABLES.values() if t.attr == name or t.name.split(".", 1)[1] == name]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"unknown tunable {name!r} (see --list)")
    raise KeyError(f"ambiguous tunable {name!r}: " + ", ".join(t.name for t in hits))


def verify_registry() -> List[str]:
    """Problems with the registry (each target must resolve; module attributes must equal the default)."""
    problems = []
    for t in TUNABLES.values():
        if t.kind == "search":
            from .search import SearchConfig
            if not hasattr(SearchConfig(), t.attr):
                problems.append(f"{t.name}: SearchConfig has no field {t.attr}")
            continue
        for tg in t.targets:
            try:
                cur = tg.get()
            except Exception as ex:  # noqa: BLE001
                problems.append(f"{t.name}: target {tg} does not resolve ({ex})")
                continue
            if t.make is None and t.kind == "weight" and cur != t.default:
                problems.append(f"{t.name}: target {tg} is {cur!r}, registry default {t.default!r}")
    return problems


# ---------------------------------------------------------------------------
# apply / restore
# ---------------------------------------------------------------------------
def _clear_caches() -> None:
    from . import danger
    cache = getattr(danger, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def apply(overrides: Dict[str, Any]) -> List[Tuple[Target, Any]]:
    """Install ``{tunable name: value}`` for the weight / flag tunables; returns the previous values.

    Search-kind tunables are skipped here (they live on a bot's ``SearchConfig``, see
    :func:`apply_to_bot`).  The returned token goes to :func:`restore`.
    """
    token: List[Tuple[Target, Any]] = []
    clear = False
    for name, value in overrides.items():
        t = find(name)
        if t.kind == "search":
            continue
        installed = t.installed_value(value)
        if installed is KEEP:
            continue
        clear = clear or t.clear_caches
        for tg in t.targets:
            token.append((tg, tg.get()))
            tg.set(installed)
    if clear:
        _clear_caches()
    return token


def restore(token: List[Tuple[Target, Any]]) -> None:
    """Undo :func:`apply` (reverse order, so nested applications unwind correctly)."""
    if not token:
        return
    for tg, prev in reversed(token):
        tg.set(prev)
    _clear_caches()


class overridden:
    """``with overridden({"danger.TURNS_HALF": 2.0}): ...`` - apply on entry, restore on exit."""

    def __init__(self, overrides: Dict[str, Any]):
        self.overrides = overrides
        self.token: List[Tuple[Target, Any]] = []

    def __enter__(self):
        self.token = apply(self.overrides)
        return self

    def __exit__(self, *exc):
        restore(self.token)
        return False


def apply_to_bot(bot, overrides: Dict[str, Any]) -> List[Tuple[Any, str, Any]]:
    """Set the search-kind overrides on ``bot.config``; returns ``[(config, attr, previous), ...]``."""
    token = []
    for name, value in overrides.items():
        t = find(name)
        if t.kind != "search":
            continue
        cfg = getattr(bot, "config", None)
        if cfg is None or not hasattr(cfg, t.attr):
            raise TypeError(f"{t.name} is a search knob but bot {getattr(bot, 'name', bot)!r} has no SearchConfig")
        token.append((cfg, t.attr, getattr(cfg, t.attr)))
        setattr(cfg, t.attr, value)
    return token


def restore_bot(token: List[Tuple[Any, str, Any]]) -> None:
    for cfg, attr, prev in reversed(token):
        setattr(cfg, attr, prev)


def check_overrides(bot, overrides: Dict[str, Any]) -> None:
    """Raise ``KeyError`` / ``TypeError`` for unknown names or search knobs on a bot without a SearchConfig."""
    for name in overrides:
        t = find(name)
        if t.kind == "search":
            cfg = getattr(bot, "config", None)
            if cfg is None or not hasattr(cfg, t.attr):
                raise TypeError(f"{t.name} is a search knob but bot {getattr(bot, 'name', bot)!r} has no SearchConfig")


# ---------------------------------------------------------------------------
# Paired games
# ---------------------------------------------------------------------------
# Seat patterns: 'C' = candidate value, 'D' = default.  Consecutive patterns are complements, so any
# even number of games gives both sides the same seats; 3-player alternates 2-vs-1 and 1-vs-2.
SEAT_PATTERNS = {
    2: ["CD", "DC"],
    3: ["CCD", "DDC", "CDC", "DCD", "DCC", "CDD"],
    4: ["CCDD", "DDCC", "CDCD", "DCDC", "CDDC", "DCCD"],
}


def seat_pattern(num_players: int, game_index: int) -> str:
    pats = SEAT_PATTERNS[num_players]
    return pats[game_index % len(pats)]


def game_seed(seed: int, game_index: int) -> int:
    """Identical across candidate values: the board and the dice only depend on (seed, game index)."""
    return seed * 100003 + game_index


def paired_jobs(games: int, seed: int, num_players: int) -> List[Tuple[str, int]]:
    return [(seat_pattern(num_players, g), game_seed(seed, g)) for g in range(games)]


def play_paired_game(base_spec: str, overrides: Dict[str, Any], pattern: str, seed: int,
                     max_turns: int = 400) -> Dict[str, Any]:
    """One game: seats marked 'C' get ``ParamBot(base bot, overrides)``, seats 'D' the bare base bot
    (wrapped too, so decision times are measured identically).  Returns a picklable summary."""
    from .agents.param_bot import ParamBot
    from .selfplay import make_bot, play_game
    bots = []
    for side in pattern:
        inner = make_bot(base_spec)
        bots.append(ParamBot(inner, overrides if side == "C" else {}, label="cand" if side == "C" else "default"))
    res = play_game(bots, rng=random.Random(seed), seed=seed, max_turns=max_turns)
    return {
        "seed": seed, "pattern": pattern, "winner": res.winner, "vps": list(res.vps), "turns": res.turns,
        "actions": res.actions, "duration": res.duration,
        "seat_times": [list(b.stats["times"]) for b in bots],          # inner.decide only (see ParamBot)
        "seat_decisions": [b.stats["decisions"] for b in bots],
        "seat_overhead": [b.stats["overhead"] for b in bots],          # apply / restore seconds, excluded above
        "evaluator_mode": evaluator_mode(),                            # the mode of the process that played it
        "pid": os.getpid(),
    }


def _paired_worker(args) -> Dict[str, Any]:
    base_spec, overrides, pattern, seed, max_turns = args
    return play_paired_game(base_spec, overrides, pattern, seed, max_turns)


def run_paired(base_spec: str, overrides: Dict[str, Any], games: int, seed: int, num_players: int,
               workers: int = 1, max_turns: int = 400,
               progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
    """Play ``games`` paired games (identical seeds for every call with the same ``seed``)."""
    args = [(base_spec, dict(overrides), pattern, gseed, max_turns) for pattern, gseed in paired_jobs(games, seed, num_players)]
    out: List[Dict[str, Any]] = []
    if workers <= 1 or len(args) <= 1:
        for k, a in enumerate(args):
            r = _paired_worker(a)
            out.append(r)
            if progress:
                progress(k + 1, len(args), r)
        return out
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_paired_worker, args)):
            out.append(r)
            if progress:
                progress(k + 1, len(args), r)
    out.sort(key=lambda r: r["seed"])
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _p95(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(0.95 * len(s)) - 1))]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired_stats(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary of paired games.

    Per game ``g`` the candidate side's seat win rate is ``c_g = 1/|C|`` if a candidate seat won
    (else 0) and ``d_g`` likewise for the default side; ``delta`` is the mean of the paired
    differences ``c_g - d_g`` (zero in expectation when both sides are equally strong, for 2-vs-2
    and for the rotating 2-vs-1 design alike), ``se`` its standard error over games and ``ci95``
    ``delta +- 1.96 se``.  In the 2-vs-2 design this is exactly the binomial standard error of the
    game-level side win rate (``side_win_rate``, ``side_se``) scaled to seat units.
    """
    G = len(results)
    cand_wins = def_wins = draws = 0
    cand_seats = def_seats = 0
    vp_c: List[float] = []
    vp_d: List[float] = []
    diffs: List[float] = []
    t_c: List[float] = []
    t_d: List[float] = []
    oh_c = oh_d = 0.0
    turns: List[int] = []
    records: List[Dict[str, Any]] = []
    modes = set()
    for r in results:
        pat = r["pattern"]
        C = [i for i, s in enumerate(pat) if s == "C"]
        D = [i for i, s in enumerate(pat) if s == "D"]
        cand_seats += len(C)
        def_seats += len(D)
        w = r["winner"]
        c_g = d_g = 0.0
        if w in C:
            cand_wins += 1
            c_g = 1.0 / len(C)
            side = "cand"
        elif w in D:
            def_wins += 1
            d_g = 1.0 / len(D)
            side = "default"
        else:
            draws += 1
            side = "draw"
        diffs.append(c_g - d_g)
        vp_c.extend(r["vps"][i] for i in C)
        vp_d.extend(r["vps"][i] for i in D)
        for i in C:
            t_c.extend(r["seat_times"][i])
        for i in D:
            t_d.extend(r["seat_times"][i])
        overhead = r.get("seat_overhead") or [0.0] * len(pat)
        oh_c += sum(overhead[i] for i in C)
        oh_d += sum(overhead[i] for i in D)
        turns.append(r["turns"])
        if r.get("evaluator_mode"):
            modes.add(r["evaluator_mode"])
        seat_ms = [[1000.0 * x for x in ts] for ts in r["seat_times"]]
        records.append({
            "seed": r["seed"], "pattern": pat, "winner": w, "winning_side": side, "diff": c_g - d_g,
            "vps": list(r["vps"]), "turns": r["turns"], "actions": r.get("actions"), "duration": r["duration"],
            "evaluator_mode": r.get("evaluator_mode"), "pid": r.get("pid"),
            "seat_decisions": list(r["seat_decisions"]),
            "seat_ms_mean": [_mean(ms) for ms in seat_ms], "seat_ms_p95": [_p95(ms) for ms in seat_ms],
            "seat_overhead_ms": [1000.0 * x for x in overhead],
        })
    delta = _mean(diffs) if diffs else float("nan")
    if G > 1:
        var = sum((x - delta) ** 2 for x in diffs) / (G - 1)
        se = math.sqrt(var / G)
    else:
        se = float("nan")
    decided = cand_wins + def_wins
    side_rate = cand_wins / decided if decided else float("nan")
    side_se = math.sqrt(side_rate * (1 - side_rate) / decided) if decided else float("nan")
    ms_c = [1000.0 * x for x in t_c]
    ms_d = [1000.0 * x for x in t_d]
    mean_c, mean_d = _mean(ms_c), _mean(ms_d)
    extra = mean_c - mean_d if ms_c and ms_d else float("nan")
    # "value per ms" only means something when the candidate is measurably costlier; a cheaper
    # candidate that is not worse is simply a win (see ``cost``).
    if math.isnan(extra):
        cost, value_per_ms = "unknown", None
    elif extra > COST_NOISE_MS:
        cost = "costlier"
        value_per_ms = (delta / extra) if not math.isnan(delta) else None
    elif extra < -COST_NOISE_MS:
        cost, value_per_ms = "cheaper", None
    else:
        cost, value_per_ms = "same cost", None
    if math.isnan(se):
        ci = [float("nan"), float("nan")]
    else:
        ci = [max(-1.0, delta - 1.96 * se), min(1.0, delta + 1.96 * se)]
    return {
        "games": G, "draws": draws,
        "cand_wins": cand_wins, "def_wins": def_wins, "cand_seats": cand_seats, "def_seats": def_seats,
        "win_rate_cand": cand_wins / cand_seats if cand_seats else float("nan"),
        "win_rate_def": def_wins / def_seats if def_seats else float("nan"),
        "delta": delta, "se": se, "ci95": ci,
        "side_win_rate": side_rate, "side_se": side_se,
        "avg_vp_cand": _mean(vp_c), "avg_vp_def": _mean(vp_d),
        "decisions_cand": len(ms_c), "decisions_def": len(ms_d),
        "ms_mean_cand": mean_c, "ms_p95_cand": _p95(ms_c), "ms_mean_def": mean_d, "ms_p95_def": _p95(ms_d),
        "extra_ms": extra, "cost": cost, "value_per_ms": value_per_ms,
        # apply / restore time per decision (NOT part of the decision times above); shows the harness's own cost
        "overhead_ms_cand": 1000.0 * oh_c / len(ms_c) if ms_c else float("nan"),
        "overhead_ms_def": 1000.0 * oh_d / len(ms_d) if ms_d else float("nan"),
        "avg_turns": _mean(turns), "seconds": sum(r["duration"] for r in results),
        "evaluator_modes": sorted(modes),      # as reported by the process that played each game
        "records": records,                    # one entry per game, in seed order
    }


COST_NOISE_MS = 0.05   # |extra ms per decision| below this counts as "same cost"


def verdict(st: Dict[str, Any], min_games: int = 30) -> str:
    """Plain-language reading of a stats row (smoke runs are always 'inconclusive')."""
    lo, hi = st["ci95"]
    if st["games"] < min_games or math.isnan(lo):
        return "inconclusive (too few games)"
    if lo > 0:
        return "candidate better"
    if hi < 0:
        return "default better"
    return "no significant difference"


def evaluator_mode() -> str:
    """Which static evaluator this process runs ("python (CATANBOT_NO_ACCEL=1)", "c++ (catanbot_core)", ...)."""
    from . import accel
    if accel.disabled_by_env():
        return "python (CATANBOT_NO_ACCEL=1)"
    if accel.AVAILABLE:
        return "c++ (catanbot_core)"
    return "python (catanbot_core not built)"
