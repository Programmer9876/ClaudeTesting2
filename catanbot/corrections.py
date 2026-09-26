"""Leaf-value correction hub: one evaluator wrapper for every Python-side leaf correction (off by default).

Several strategy terms are search-time corrections on top of the static value: a per-seat number of static
points ``C_i`` added to ``heuristic.static_value`` before the evaluator's softmax, computed in Python on top of
the C++ static values, so that neither ``static_value`` nor ``cpp/*`` changes and no ``needs_python_evaluator``
re-exec is needed.  Win-path races (``winpaths.PathsContext``) were the first; the conversion cost
(``conversion.ConversionContext``) and acq.progress (``acquisition.AcqContext``, the trades area) followed; the ports
/ robber terms of docs/PRIORITY_PLAN.md (steps 4-5) are the next.  Instead of each wrapping the evaluator on its own
(and wrapping each other, three softmaxes per leaf), they are *providers* of one :class:`CorrectionHub`:

* a provider has ``corrections(state) -> per-seat points`` (``None`` or all zeros = nothing to add), and
  optionally ``adjust_priors(state, legal, priors)`` (move ordering of our own nodes; chained in provider order
  when the provider's ``priors`` attribute is true) and a ``stats`` dict;
* the hub sums the providers' corrections in list order and returns ``softmax((V + sum C) / T)[player]`` with
  ``V`` = the C++ static values of every seat and ``T`` the base evaluator's temperature (for a blended or
  other base, the base's value plus the change of that heuristic probability, as ``winpaths.PathsEvaluator``);
* **fast path**: a leaf whose corrections are all zero gets the base evaluator's own value (the batched C++
  ``heuristic_evaluate``), bit for bit, so a provider that is silent on most leaves costs only its own check;
* finished games and setup-phase states pass through to the base evaluator;
* **leaf-chance hook** (``chance`` providers, depth 1): a finished leaf ``s`` may be replaced by a mixture
  ``sum_k p_k v(s_k)`` of hub values of states the provider builds (for example "the robbed player kicks the
  robber with a knight before our next turn", docs/PRIORITY_PLAN.md ``search.knight_kick``).  The first chance
  provider that answers for a leaf wins: mixtures of the same mechanism must never be stacked.

Search integration (``catanbot/search.py``): ``Searcher._value_ev()`` is the one evaluator every leaf-value site
uses (``_eval``, ``_counter_filter``, the political trade options) and the native C++ lookahead is taken only when
it is the bare base evaluator.  ``paths = 1`` alone keeps building ``winpaths.PathsEvaluator`` exactly as before
(its values are bit-identical to the pre-hub code, tested); the hub is built only when a non-winpaths provider is
switched on, and then winpaths' ``PathsContext`` is its first provider (built through
``PathsEvaluator.for_search``, unchanged).  With every provider off nothing of this module is imported.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import accel as _accel
from .state import GameState, PHASE_GAME_OVER, PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT

_SETUP = (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)


def static_values(state: GameState) -> List[float]:
    """``accel.static_values`` (C++ ``static_value`` of every seat), called directly once verified."""
    if _accel.AVAILABLE and _accel._verified:
        try:
            return _accel._core.static_values(state)
        except ValueError as exc:
            if not _accel._unsupported(exc):
                raise
    return _accel.static_values(state)


def _softmax(z: Sequence[float], T: float) -> List[float]:
    m = max(z)
    exps = [math.exp((v - m) / T) for v in z]
    tot = sum(exps)
    return [x / tot for x in exps]


def base_kind(base) -> str:
    """"heuristic" (a plain ``HeuristicEvaluator``), "blend" (a ``BlendedEvaluator``) or "other" (a value net)."""
    from .heuristic import HeuristicEvaluator
    if isinstance(base, HeuristicEvaluator) and type(base).evaluate is HeuristicEvaluator.evaluate:
        return "heuristic"
    h = getattr(base, "heuristic", None)
    if hasattr(base, "net") and hasattr(base, "alpha") and h is not None and hasattr(h, "temperature"):
        return "blend"
    return "other"


class CorrectionHub:
    """The evaluator ``softmax((static + sum_p C_p) / T)`` over a provider list (see the module docstring)."""

    name = "hub"

    def __init__(self, base, providers: Sequence[Any], chance: Sequence[Any] = ()):
        self.base = base
        self.providers = list(providers)
        self.chance = list(chance)
        self.kind = base_kind(base)
        if self.kind == "heuristic":
            self.temperature = float(base.temperature)
        elif self.kind == "blend":
            self.temperature = float(base.heuristic.temperature)
        else:
            self.temperature = 16.0
        self.stats: Dict[str, Any] = {"evals": 0, "corrected": 0, "passed": 0, "chance_leaves": 0,
                                      "chance_states": 0, "seconds": 0.0}

    @classmethod
    def for_search(cls, base, root: GameState, me: int, cfg) -> "CorrectionHub":
        """The providers ``cfg`` switches on, in the fixed order: winpaths (``paths``), conversion (``conv``),
        acquisition (``acq``, catanbot/acquisition.py acq.progress)."""
        providers: List[Any] = []
        if getattr(cfg, "paths", 0):
            from . import winpaths   # the same constructor as paths = 1 alone
            providers.append(winpaths.PathsEvaluator.for_search(base, root, me, cfg).ctx)
        if getattr(cfg, "conv", 0):
            from .conversion import ConversionContext
            providers.append(ConversionContext(root, me))
        if getattr(cfg, "acq", 0):
            from .acquisition import AcqContext
            providers.append(AcqContext.for_search(root, me, cfg))
        return cls(base, providers)

    # --- corrections --------------------------------------------------------------------------
    def corrections(self, state: GameState) -> Optional[List[float]]:
        """Per-seat sum of the providers' corrections (list order), or ``None`` when every one is zero."""
        tot: Optional[List[float]] = None
        for p in self.providers:
            c = p.corrections(state)
            if not c or not any(c):
                continue
            if tot is None:
                tot = [float(x) for x in c]
            else:
                tot = [a + float(b) for a, b in zip(tot, c)]
        return tot

    def adjust_priors(self, state: GameState, legal: Sequence, priors: Sequence[float]) -> List[float]:
        """The providers' move-ordering hooks, chained in list order (those whose ``priors`` flag is set)."""
        out = list(priors)
        for p in self.providers:
            hook = getattr(p, "adjust_priors", None)
            if hook is not None and getattr(p, "priors", True):
                out = hook(state, legal, out)
        return out

    # --- evaluator interface ------------------------------------------------------------------
    def evaluate(self, states: Sequence[GameState], players: Sequence[int]) -> np.ndarray:
        t0 = time.perf_counter()
        try:
            return self._evaluate(list(states), list(players))
        finally:
            self.stats["seconds"] += time.perf_counter() - t0

    __call__ = evaluate

    def _evaluate(self, states: List[GameState], players: List[int]) -> np.ndarray:
        N = len(states)
        self.stats["evals"] += N
        corr: Dict[int, Optional[List[float]]] = {}
        todo: List[int] = []
        for k, s in enumerate(states):
            if (s.phase == PHASE_GAME_OVER and s.winner >= 0) or s.phase in _SETUP:
                continue
            key = id(s)
            if key not in corr:
                corr[key] = self.corrections(s)
            if corr[key] is not None:
                todo.append(k)
        self.stats["corrected"] += len(todo)
        self.stats["passed"] += N - len(todo)
        T = self.temperature
        probs: Dict[int, Tuple[List[float], Optional[List[float]]]] = {}

        def probs_of(s: GameState, with_base: bool) -> Tuple[List[float], Optional[List[float]]]:
            got = probs.get(id(s))
            if got is None:
                V = static_values(s)
                c = corr[id(s)]
                got = (_softmax([V[i] + c[i] for i in range(len(V))], T), _softmax(V, T) if with_base else None)
                probs[id(s)] = got
            return got

        if self.kind == "heuristic":
            out = np.empty(N, dtype=np.float64)
            if len(todo) < N:
                done = set(todo)
                rest = [k for k in range(N) if k not in done]
                vals = self.base.evaluate([states[k] for k in rest], [players[k] for k in rest])
                for k, v in zip(rest, vals):
                    out[k] = float(v)
            for k in todo:
                out[k] = probs_of(states[k], False)[0][players[k]]
            return out
        out = np.asarray(self.base.evaluate(states, players), dtype=np.float64).copy()
        w = 1.0 - float(self.base.alpha) if self.kind == "blend" else 1.0
        if w <= 0.0:
            return out
        for k in todo:
            new, old = probs_of(states[k], True)
            out[k] += w * (new[players[k]] - old[players[k]])
        return out

    # --- leaf-chance hook ---------------------------------------------------------------------
    def leaf_chance(self, states: Sequence[GameState], finished: Sequence[bool], me: int,
                    values: Sequence[float]) -> List[float]:
        """Values of a search level's new nodes with every finished leaf a chance provider answers for replaced
        by ``sum_k p_k v(s_k)`` (``v`` = this hub; an outcome that *is* the leaf reuses its value).  The extra
        states of the level are evaluated in one batch.  Unfinished nodes and finished games are unchanged."""
        out = [float(v) for v in values]
        if not self.chance:
            return out
        plan: List[Tuple[int, List[Tuple[float, GameState]]]] = []
        extra: List[GameState] = []
        for k, (s, fin) in enumerate(zip(states, finished)):
            if not fin or s.phase == PHASE_GAME_OVER or s.phase in _SETUP:
                continue
            for prov in self.chance:
                outs = prov.leaf_outcomes(s, me)
                if outs:
                    plan.append((k, list(outs)))
                    extra.extend(s2 for _, s2 in outs if s2 is not s)
                    break
        if not plan:
            return out
        self.stats["chance_leaves"] += len(plan)
        self.stats["chance_states"] += len(extra)
        ev = list(self.evaluate(extra, [me] * len(extra))) if extra else []
        j = 0
        for k, outs in plan:
            v = 0.0
            for p, s2 in outs:
                if s2 is states[k]:
                    v += p * out[k]
                else:
                    v += p * float(ev[j])
                    j += 1
            out[k] = v
        return out


# ---------------------------------------------------------------------------
# Toy providers (tests and diagnostics)
# ---------------------------------------------------------------------------
class ConstantProvider:
    """``corrections`` = a fixed per-seat vector, optionally only when ``when(state)`` holds (tests)."""

    def __init__(self, values: Sequence[float], when: Optional[Callable[[GameState], bool]] = None,
                 priors: Optional[Callable[[GameState, Sequence, List[float]], List[float]]] = None):
        self.values = [float(x) for x in values]
        self.when = when
        self._priors = priors
        self.priors = priors is not None
        self.calls = 0

    def corrections(self, state: GameState) -> List[float]:
        self.calls += 1
        if self.when is not None and not self.when(state):
            return [0.0] * len(self.values)
        return list(self.values)

    def adjust_priors(self, state: GameState, legal: Sequence, priors: Sequence[float]) -> List[float]:
        return self._priors(state, legal, list(priors)) if self._priors is not None else list(priors)
