"""``ParamBot``: a bot whose strategy constants differ from the module defaults.

``ParamBot(inner, overrides)`` wraps any :class:`Bot`.  Around *every* hook of the inner
bot (``reset``, ``decide``, ``observe``, ``explain``) it installs the overrides from
``catanbot.tuning`` (module constants, patched functions, ``SearchConfig`` fields) and
restores them in a ``finally`` block, so that in a game where some seats run the candidate
values and the others the defaults each bot sees its own constants and an exception in the
inner bot never leaks an override into the rest of the process.  ``play_game`` needs no
change: the wrapper has the same interface as the inner bot.

``stats`` records the wall time of every decision that had more than one legal action
(``times``, seconds), the number of such decisions and of trivial ones.  A decision's time
covers ``inner.decide`` only: the apply / restore of the overrides around it is measured
separately (``overhead``, seconds), so a candidate that needs many patches is not charged
for the harness's own work while the default side (empty overrides) pays none.

Bot spec form: ``selfplay.make_bot`` wraps the bot in a ``ParamBot`` when the spec has a
``tune=NAME:VALUE;NAME:VALUE`` key (:func:`parse_tune` / :func:`format_tune`), e.g.
``search:depth=1,beam=4,expand=8,evaluator=heuristic,tune=danger.TURNS_HALF:2.4;devcards.KNIGHT_VALUE:0.62``.
That is how a tuned parameter set (``scripts/tune_joint.py``) reaches the league gate and the
Catanatron benchmark as an ordinary spec; a spec without the key builds exactly the bot it always did.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .. import tuning
from ..actions import Action
from ..state import GameState
from .base import Bot


class ParamBot(Bot):
    def __init__(self, inner: Bot, overrides: Optional[Dict[str, Any]] = None, label: Optional[str] = None):
        self.inner = inner
        self.overrides: Dict[str, Any] = dict(overrides or {})
        tuning.check_overrides(inner, self.overrides)
        tag = label or ("cand" if self.overrides else "default")
        detail = ",".join(f"{tuning.find(k).name}={tuning.find(k).format(v)}" for k, v in self.overrides.items())
        self.name = f"{inner.name}[{tag}{(':' + detail) if detail else ''}]"
        self.stats: Dict[str, Any] = {"decisions": 0, "trivial": 0, "seconds": 0.0, "times": [], "overhead": 0.0}

    # --- override plumbing ---------------------------------------------------------------
    def _enter(self):
        token = tuning.apply(self.overrides)
        try:
            bot_token = tuning.apply_to_bot(self.inner, self.overrides)
        except BaseException:
            tuning.restore(token)
            raise
        return token, bot_token

    def _exit(self, tokens) -> None:
        token, bot_token = tokens
        try:
            tuning.restore_bot(bot_token)
        finally:
            tuning.restore(token)

    @contextlib.contextmanager
    def scope(self, prefixes: Optional[Sequence[str]] = None) -> Iterator[None]:
        """The overrides installed for the block (restored in ``finally``), for work done on the bot's
        behalf outside its hooks - the counted adapter's determinizations.  ``prefixes`` restricts
        them to the tunables whose name starts with one of them (``("devbelief.",)``); with no
        matching override nothing is applied at all."""
        if prefixes is None:
            tokens = self._enter()
        else:
            sub = {k: v for k, v in self.overrides.items() if tuning.find(k).name.startswith(tuple(prefixes))}
            if not sub:
                yield
                return
            token = tuning.apply(sub)
            try:
                bot_token = tuning.apply_to_bot(self.inner, sub)
            except BaseException:
                tuning.restore(token)
                raise
            tokens = (token, bot_token)
        try:
            yield
        finally:
            self._exit(tokens)

    def _call(self, fn, *args):
        tokens = self._enter()
        try:
            return fn(*args)
        finally:
            self._exit(tokens)

    # --- Bot protocol --------------------------------------------------------------------
    def reset(self) -> None:
        self._call(self.inner.reset)

    def decide(self, state: GameState, legal_actions: List[Action], rng) -> Action:
        t_enter = time.perf_counter()
        tokens = self._enter()
        t0 = time.perf_counter()
        try:
            return self.inner.decide(state, legal_actions, rng)
        finally:
            t1 = time.perf_counter()
            try:
                self._exit(tokens)
            finally:
                t2 = time.perf_counter()
                if len(legal_actions) > 1:
                    self.stats["decisions"] += 1
                    self.stats["seconds"] += t1 - t0
                    self.stats["times"].append(t1 - t0)              # inner.decide only
                    self.stats["overhead"] += (t0 - t_enter) + (t2 - t1)   # apply + restore, kept apart
                else:
                    self.stats["trivial"] += 1

    def observe(self, state: GameState, action: Action, player: int) -> None:
        self._call(self.inner.observe, state, action, player)

    def explain(self, state: GameState) -> Optional[str]:
        return self._call(self.inner.explain, state)

    def __getattr__(self, item: str):
        # Anything else (trade_bias, config, last_results, ...) comes from the inner bot.
        if item in ("inner", "overrides", "stats", "name"):
            raise AttributeError(item)
        return getattr(self.inner, item)

    def __repr__(self) -> str:
        return f"ParamBot({self.inner!r}, {self.overrides!r})"


# ---------------------------------------------------------------------------
# Overrides inside a bot spec (``tune=``)
# ---------------------------------------------------------------------------
def parse_tune(text: str) -> Dict[str, Any]:
    """``"danger.TURNS_HALF:2.4;devcards.KNIGHT_VALUE:0.62"`` -> ``{name: value}`` through the registry's
    parsers (flags ``on`` / ``off``, resource vectors ``a/b/c/d/e``)."""
    out: Dict[str, Any] = {}
    for item in text.split(";"):
        if not item.strip():
            continue
        name, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"tune= entries are NAME:VALUE separated by ';', got {item!r}")
        t = tuning.find(name.strip())
        out[t.name] = t.parse(value.strip())
    return out


def format_tune(overrides: Dict[str, Any]) -> str:
    """Inverse of :func:`parse_tune` (registry formatting: ``%g`` floats, ``on`` / ``off``, ``a/b/c/d/e``)."""
    return ";".join(f"{tuning.find(k).name}:{tuning.find(k).format(v)}" for k, v in overrides.items())


def tuned_spec(base_spec: str, overrides: Dict[str, Any]) -> str:
    """``base_spec`` with ``overrides`` built in: search knobs that have a spec key become that key
    (``trades=``, ``counter_margin=``, ...), everything else goes into ``tune=`` (merged with one already there)."""
    from ..selfplay import parse_spec
    name, kw = parse_spec(base_spec)
    rest = parse_tune(kw.pop("tune", ""))
    for k, v in overrides.items():
        t = tuning.find(k)
        if t.kind == "search" and t.spec_key:
            if name != "search":
                raise ValueError(f"{t.name} is a search knob but the base spec {base_spec!r} is not a search bot")
            kw[t.spec_key] = t.format(v)
        else:
            rest[t.name] = v
    if rest:
        kw["tune"] = format_tune(rest)
    return name + (":" + ",".join(f"{k}={v}" for k, v in kw.items()) if kw else "")
