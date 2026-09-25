"""``ParamBot``: a bot whose strategy constants differ from the module defaults.

``ParamBot(inner, overrides)`` wraps any :class:`Bot`.  Around *every* hook of the inner
bot (``reset``, ``decide``, ``observe``, ``explain``) it installs the overrides from
``catanbot.tuning`` (module constants, patched functions, ``SearchConfig`` fields) and
restores them in a ``finally`` block, so that in a game where some seats run the candidate
values and the others the defaults each bot sees its own constants and an exception in the
inner bot never leaks an override into the rest of the process.  ``play_game`` needs no
change: the wrapper has the same interface as the inner bot.

``stats`` records the wall time of every decision that had more than one legal action
(``times``, seconds), the number of such decisions and of trivial ones.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

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
        self.stats: Dict[str, Any] = {"decisions": 0, "trivial": 0, "seconds": 0.0, "times": []}

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
        t0 = time.perf_counter()
        try:
            return self._call(self.inner.decide, state, legal_actions, rng)
        finally:
            dt = time.perf_counter() - t0
            if len(legal_actions) > 1:
                self.stats["decisions"] += 1
                self.stats["seconds"] += dt
                self.stats["times"].append(dt)
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
