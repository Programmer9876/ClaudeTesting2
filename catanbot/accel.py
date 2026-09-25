"""Optional C++ acceleration (the ``catanbot_core`` pybind11 extension).

``catanbot.features.extract`` / ``extract_batch`` and
``catanbot.heuristic.HeuristicEvaluator.evaluate`` call into the extension when
:data:`AVAILABLE` is true; the pure-Python code in ``features.py`` /
``heuristic.py`` stays the reference implementation and the fallback.

* Build: ``scripts/build_cpp.sh`` (see ``docs/CPP.md``).  The module is looked
  up as ``catanbot.catanbot_core`` first, then as a top-level ``catanbot_core``.
* Disable: set the environment variable ``CATANBOT_NO_ACCEL=1`` before
  importing catanbot (or assign ``catanbot.accel.AVAILABLE = False`` at run
  time; ``features`` reads the flag on every call).
* Safety: the first call checks that the extension was compiled for the
  current ``features.FEATURE_NAMES`` layout.  On a mismatch the extension is
  disabled with a ``RuntimeWarning`` and the Python path is used instead, so a
  stale build can never produce silently wrong features.
* Unsupported states: the extension raises ``catanbot_core.UnsupportedStateError``
  (a ``ValueError``) for a state its fixed-size structs cannot hold - more than
  4 players, lists of the wrong length, ids or integers out of the 32-bit
  range, more than 64 road entries for one player.  The wrappers below catch it
  and compute that call with the Python reference instead, so the public
  functions behave exactly as without the extension (only slower) for them.
"""
from __future__ import annotations

import os
import threading
import warnings
from typing import List, Optional, Sequence

__all__ = ["AVAILABLE", "extract", "extract_batch", "longest_road_length", "static_values", "static_value",
           "heuristic_evaluate", "load_core", "disabled_by_env", "verify",
           "ENGINE_ACTIVE", "ENGINE_ENV", "engine_enabled_by_env", "engine_available", "engine_legal_actions",
           "engine_apply", "engine_apply_inplace", "engine_apply_forced", "random_playout_fast",
           "NATIVE_SEARCH_ENV", "native_search_disabled_by_env", "native_search_available", "native_evaluator",
           "evaluator_key", "robber_weights_bundle", "future_values"]

# Entry points every usable build provides; an older build missing one is stale and gets disabled by verify().
_REQUIRED = ("extract_batch", "extract", "longest_road_length", "static_values", "static_value", "heuristic_evaluate",
             "UnsupportedStateError", "legal_actions", "apply", "apply_inplace", "apply_forced", "random_playout_fast",
             "future_values", "HeuristicEval", "MlpEval", "BlendEval")


def disabled_by_env() -> bool:
    """True when ``CATANBOT_NO_ACCEL`` is set to anything but empty / 0 / false / no."""
    return os.environ.get("CATANBOT_NO_ACCEL", "").strip().lower() not in ("", "0", "false", "no")


def load_core():
    """Import and return the extension module regardless of ``CATANBOT_NO_ACCEL``; ``None`` if not built."""
    try:
        from . import catanbot_core as core  # built into catanbot/ by scripts/build_cpp.sh
        return core
    except ImportError:
        pass
    try:
        import catanbot_core as core  # type: ignore[no-redef]  # built at the repository root
        return core
    except ImportError:
        return None


_core = None if disabled_by_env() else load_core()
AVAILABLE: bool = _core is not None
_verified = False
_fallback_lock = threading.RLock()


def verify() -> bool:
    """Check (once) that the extension matches ``features.FEATURE_NAMES``; disables it on mismatch."""
    global _verified, AVAILABLE
    if _verified:
        return AVAILABLE
    _verified = True
    if _core is None:
        AVAILABLE = False
        return False
    from . import features as F  # lazy: features imports this module at load time
    try:
        ok = (int(_core.num_features()) == F.NUM_FEATURES and list(_core.feature_names()) == list(F.FEATURE_NAMES)
              and all(hasattr(_core, name) for name in _REQUIRED))
    except Exception:  # pragma: no cover - a broken build
        ok = False
    if not ok:
        AVAILABLE = False
        warnings.warn("catanbot_core was built for a different feature layout or is an older build missing entry "
                      "points; falling back to the Python implementation.  Rebuild it with scripts/build_cpp.sh.",
                      RuntimeWarning, stacklevel=2)
    return AVAILABLE


def _unsupported(exc: BaseException) -> bool:
    """True when the extension refused the state as one it cannot represent (``UnsupportedStateError``)."""
    return isinstance(exc, getattr(_core, "UnsupportedStateError", ()))


def _python(fn, *args):
    """Run ``fn`` (a ``features`` / ``heuristic`` entry point) on the pure-Python path.

    Those functions route to the extension whenever ``AVAILABLE`` is true, so the flag is cleared for
    the duration of the call and restored afterwards (under a lock, so concurrent fallbacks restore it
    correctly; another thread extracting features meanwhile simply takes the Python path as well).
    """
    global AVAILABLE
    if not AVAILABLE:
        return fn(*args)
    with _fallback_lock:
        prev, AVAILABLE = AVAILABLE, False
        try:
            return fn(*args)
        finally:
            AVAILABLE = prev


def extract_batch(states: Sequence, players: Sequence[int]):
    """C++ ``features.extract_batch`` (Python for states the extension cannot represent, or when it is unusable)."""
    if AVAILABLE and (_verified or verify()):
        try:
            return _core.extract_batch(states, players)
        except ValueError as exc:
            if not _unsupported(exc):
                raise
    from . import features as F
    return _python(F.extract_batch, states, players)


def extract(state, player: int):
    """C++ ``features.extract`` (Python for states the extension cannot represent, or when it is unusable)."""
    if AVAILABLE and (_verified or verify()):
        try:
            return _core.extract(state, int(player))
        except ValueError as exc:
            if not _unsupported(exc):
                raise
    from . import features as F
    return _python(F.extract, state, player)


def longest_road_length(state, player: int) -> int:
    """C++ ``features.longest_road_length`` (Python for states the extension cannot represent, or when unusable)."""
    if AVAILABLE and (_verified or verify()):
        try:
            return int(_core.longest_road_length(state, int(player)))
        except ValueError as exc:
            if not _unsupported(exc):
                raise
    from . import features as F
    return F.longest_road_length(state, player)  # pure Python, never routes back here


def static_values(state) -> List[float]:
    """C++ ``heuristic.static_value`` for every player of ``state`` (Python when unsupported / unusable)."""
    if AVAILABLE and (_verified or verify()):
        try:
            return list(_core.static_values(state))
        except ValueError as exc:
            if not _unsupported(exc):
                raise
    from . import heuristic as H
    return [H.static_value(state, i) for i in range(state.num_players)]  # static_value is pure Python


def static_value(state, player: int) -> float:
    """C++ ``heuristic.static_value(state, player)`` (Python when unsupported / unusable)."""
    if AVAILABLE and (_verified or verify()):
        try:
            return float(_core.static_value(state, int(player)))
        except ValueError as exc:
            if not _unsupported(exc):
                raise
    from . import heuristic as H
    return H.static_value(state, player)


def heuristic_evaluate(states: Sequence, players: Sequence[int], temperature: float = 16.0):
    """C++ ``HeuristicEvaluator.evaluate`` -> float64 array (Python when unsupported / unusable)."""
    if AVAILABLE and (_verified or verify()):
        try:
            return _core.heuristic_evaluate(states, players, float(temperature))
        except ValueError as exc:
            if not _unsupported(exc):
                raise
    from . import heuristic as H
    return _python(H.HeuristicEvaluator(temperature).evaluate, states, players)


def core_file() -> Optional[str]:
    """Path of the loaded extension (for diagnostics)."""
    return getattr(_core, "__file__", None) if _core is not None else None


# ---------------------------------------------------------------------------
# Rules engine (C++ port of catanbot.engine: legal_actions / apply / apply_inplace / random_playout)
# ---------------------------------------------------------------------------
# The Python engine stays the default.  ``engine.legal_actions`` / ``apply`` / ``apply_inplace`` route
# to the extension only when :data:`ENGINE_ACTIVE` is true: the extension is loaded, it has the engine
# entry points and the environment variable ``CATANBOT_ACCEL_ENGINE`` is set (opt-in, default off).
# The wrappers return ``None`` for a state the extension cannot represent (``UnsupportedStateError``)
# so the caller falls back to the Python code; illegal actions raise ``engine.IllegalActionError``
# exactly like the Python engine (the extension raises that very class).
ENGINE_ENV = "CATANBOT_ACCEL_ENGINE"
_ENGINE_REQUIRED = ("legal_actions", "apply", "apply_inplace", "apply_forced", "random_playout_fast")


def engine_enabled_by_env() -> bool:
    """True when ``CATANBOT_ACCEL_ENGINE`` is set to anything but empty / 0 / false / no."""
    return os.environ.get(ENGINE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def engine_available() -> bool:
    """True when the loaded extension provides the engine entry points (independent of the env switch)."""
    return _core is not None and all(hasattr(_core, name) for name in _ENGINE_REQUIRED)


ENGINE_ACTIVE: bool = AVAILABLE and engine_enabled_by_env() and engine_available()


def engine_legal_actions(state):
    """C++ ``engine.legal_actions`` (same tuples, same order); ``None`` when the state is unsupported."""
    try:
        return _core.legal_actions(state)
    except ValueError as exc:
        if _unsupported(exc):
            return None
        raise


def engine_apply(state, action, rng=None):
    """C++ ``engine.apply``: a new ``GameState``; ``None`` when the state is unsupported.

    ``rng`` may be ``None``, an int seed (C++ generator) or a ``random.Random``, which the extension
    consults exactly like the Python engine does (same calls, same order, same results).
    """
    try:
        return _core.apply(state, action, rng)
    except ValueError as exc:
        if _unsupported(exc):
            return None
        raise


def engine_apply_inplace(state, action, rng=None):
    """C++ ``engine.apply_inplace`` (written back into the same objects); ``None`` when unsupported."""
    try:
        return _core.apply_inplace(state, action, rng)
    except ValueError as exc:
        if _unsupported(exc):
            return None
        raise


def engine_apply_forced(state, action, drawn_index: int):
    """``engine.apply`` with the random choice given (steal / dev draw index, or a 2d6 pair index for ``(ROLL,)``)."""
    return _core.apply_forced(state, action, int(drawn_index))


def random_playout_fast(state, seed=None, max_turns=None, max_actions: int = 2_000_000, trace: bool = False):
    """``engine.random_playout`` entirely in C++ (see ``core.random_playout_fast``); needs the extension."""
    return _core.random_playout_fast(state, seed, max_turns, int(max_actions), bool(trace))


# ---------------------------------------------------------------------------
# Native lookahead (C++ port of Searcher._future_values: opponents' greedy turns + reduced our-turn search)
# ---------------------------------------------------------------------------
# ``search.Searcher._future_values`` hands its batch of end-of-turn states to ``core.future_values`` when
# :func:`native_search_available` is true and the evaluator is one the extension knows
# (:func:`native_evaluator`).  The Python body stays the reference and the fallback: extension missing or
# stale, ``CATANBOT_NO_ACCEL=1``, ``CATANBOT_NO_NATIVE_SEARCH=1``, ``SearchConfig(native_future=False)``, an
# evaluator the extension does not know (any object other than a ``HeuristicEvaluator``, a float32 ``ValueNet``
# or a ``BlendedEvaluator`` of the two) or a state the structs cannot hold (``UnsupportedStateError`` ->
# :func:`future_values` returns ``None``).  See docs/CPP.md, section "Native lookahead".
NATIVE_SEARCH_ENV = "CATANBOT_NO_NATIVE_SEARCH"
_NATIVE_REQUIRED = ("future_values", "HeuristicEval", "MlpEval", "BlendEval", "reduced_search")


def native_search_disabled_by_env() -> bool:
    """True when ``CATANBOT_NO_NATIVE_SEARCH`` is set to anything but empty / 0 / false / no."""
    return os.environ.get(NATIVE_SEARCH_ENV, "").strip().lower() not in ("", "0", "false", "no")


def native_search_available() -> bool:
    """True when the extension is usable, provides the lookahead entry points and the env switch is off."""
    if not AVAILABLE or native_search_disabled_by_env():
        return False
    if not (_verified or verify()):
        return False
    return _core is not None and all(hasattr(_core, name) for name in _NATIVE_REQUIRED)


def _is_float32_net(ev) -> bool:
    import numpy as np
    try:
        if not (isinstance(ev.W, list) and isinstance(ev.b, list) and len(ev.W) >= 1 and len(ev.W) == len(ev.b)):
            return False
        if np.dtype(getattr(ev, "dtype", None)) != np.float32:
            return False
        arrays = list(ev.W) + list(ev.b) + [ev.mean, ev.std]
        return all(isinstance(x, np.ndarray) and x.dtype == np.float32 for x in arrays)
    except Exception:
        return False


def _is_plain(evaluator, cls) -> bool:
    """An instance of ``cls`` whose ``evaluate`` is the class's own (a subclass that overrides it - noise for
    exploration, a test double - has semantics the extension does not know and keeps the Python path)."""
    return isinstance(evaluator, cls) and getattr(type(evaluator), "evaluate", None) is cls.evaluate


def evaluator_key(evaluator):
    """What the native handle was built from (arrays replaced by ``set_params`` / ``load`` change it).

    In-place mutation of the same numpy arrays is not detected (documented); ``None`` for unknown objects,
    including subclasses that override ``evaluate``.  A ``ValueNet.input_mask`` is part of the key (the mask
    is folded into the first layer of the native twin, see :func:`native_evaluator`).
    """
    from .heuristic import HeuristicEvaluator   # lazy: heuristic imports this module at load time
    from .model import ValueNet
    from .selfplay import BlendedEvaluator
    if _is_plain(evaluator, HeuristicEvaluator) and hasattr(evaluator, "temperature"):
        return ("heuristic", float(evaluator.temperature))
    if _is_plain(evaluator, BlendedEvaluator) and hasattr(evaluator, "net") and hasattr(evaluator, "alpha") \
            and hasattr(evaluator, "heuristic"):
        kn = evaluator_key(evaluator.net)
        kh = evaluator_key(evaluator.heuristic)
        if kn is None or kn[0] != "mlp" or kh is None or kh[0] != "heuristic":
            return None
        return ("blend", float(evaluator.alpha), kn, kh)
    if _is_plain(evaluator, ValueNet) and hasattr(evaluator, "W") and hasattr(evaluator, "b") \
            and hasattr(evaluator, "mean") and hasattr(evaluator, "std") and hasattr(evaluator, "n_in") \
            and _is_float32_net(evaluator):
        mask = getattr(evaluator, "_input_mask", None)
        arrays = list(evaluator.W) + list(evaluator.b) + [evaluator.mean, evaluator.std]
        return ("mlp", id(evaluator), tuple((int(x.ctypes.data), x.shape) for x in arrays),
                None if mask is None else (int(mask.ctypes.data), mask.shape))
    return None


def _masked_layers(net):
    """``net.W`` with ``net.input_mask`` folded into the first layer: ``(H * m) @ W0 == H @ (m[:, None] * W0)``
    exactly (the mask is 0 / 1 in float32, so every product is either unchanged or exactly zero)."""
    import numpy as np
    W = list(net.W)
    mask = getattr(net, "_input_mask", None)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32).ravel()
        if m.shape != (W[0].shape[0],):
            raise ValueError("input mask shape mismatch")
        W[0] = np.ascontiguousarray(W[0] * m[:, None], dtype=np.float32)
    return W


def native_evaluator(evaluator):
    """The extension's twin of ``evaluator`` (a handle), or ``None`` when it has none.

    A ``HeuristicEvaluator`` (``temperature``), a float32 ``ValueNet`` (``W`` / ``b`` / ``mean`` / ``std`` /
    ``n_in``; an ``input_mask`` is folded into the first layer) or a ``BlendedEvaluator`` (``net`` / ``alpha`` /
    ``heuristic``) of those two, each with the class's own ``evaluate``.  Anything else - a timing wrapper, a
    subclass overriding ``evaluate``, a float64 net, a custom evaluator - gets ``None`` and the search keeps its
    Python path.  Like ``ValueNet.evaluate``, the native net returns 1 / 0 for a finished game.
    """
    if not native_search_available():
        return None
    key = evaluator_key(evaluator)
    if key is None:
        return None
    try:
        if key[0] == "heuristic":
            return _core.HeuristicEval(float(evaluator.temperature))
        if key[0] == "mlp":
            return _core.MlpEval(_masked_layers(evaluator), list(evaluator.b), evaluator.mean, evaluator.std)
        if key[0] == "blend":
            net = _core.MlpEval(_masked_layers(evaluator.net), list(evaluator.net.b), evaluator.net.mean,
                                evaluator.net.std)
            return _core.BlendEval(net, float(evaluator.alpha), _core.HeuristicEval(float(evaluator.heuristic.temperature)))
    except (ValueError, TypeError):
        return None
    return None


def robber_weights_bundle(state, politics, model) -> Optional[dict]:
    """The state-independent part of the simulated opponents' robber target weights.

    Mirrors ``politics.PoliticalState.robber_target_weights(state, actor, model)`` (``politics`` given) or
    ``OpponentModel.robber_habit_weights(state, actor)`` (only ``model`` given): per (actor, victim) the
    constant factors ``[1 + 1.2 grudge, 1 - 0.5 max(0, capital - baseline), habit base, 1 - 0.6 ally]``, the
    victim's leader-habit factor and per (victim, leader) the coalition factor (1.0 when the bloc is weak).  The
    extension multiplies them in the Python order onto ``target_weight`` / ``threat`` recomputed per simulated
    state, with the leader recomputed per state as well, so the weights are bit-identical.  ``None`` when
    neither is given (``best_robber_move`` then uses its own threat x danger weights, like Python).
    """
    import numpy as np
    n = state.num_players
    if politics is None and model is None:
        return None
    const = np.ones((n, n, 4), dtype=np.float64)
    habit_leader = np.ones((n, n), dtype=np.float64)
    coal = np.ones((n, n), dtype=np.float64)
    if model is not None:
        from .opponent_model import _pname
        for a in range(n):
            prof = model.profile_of(state, a)
            total = sum(prof.robbed.values())
            for j in range(n):
                if j == a:
                    continue
                f = 1.0
                if total > 0 and n > 2:
                    share = prof.robbed.get(_pname(state, j), 0.0) / total
                    trust = min(1.0, total / 3.0)
                    f *= max(0.5, min(2.5, 1.0 + 0.8 * trust * (share * (n - 1) - 1.0)))
                const[a, j, 2] = f
                if prof.robs_leader.weight >= 1:
                    habit_leader[a, j] = 0.6 + 0.8 * prof.robs_leader.mean()
    if politics is None:
        return {"mode": "habit", "n": n, "has_habit": True, "const": const, "habit_leader": habit_leader, "coal": coal}
    politics.ensure(n)
    for a in range(n):
        for j in range(n):
            if j == a:
                continue
            const[a, j, 0] = 1.0 + 1.2 * politics.grudge(a, j)
            const[a, j, 1] = 1.0 - 0.5 * max(0.0, politics.get(j, a) - politics.baseline)
            ally = min(1.0, politics.coalitions.strength(a, j) / 2.0)
            const[a, j, 3] = 1.0 - 0.6 * ally
    for j in range(n):
        for k in range(n):
            if j == k:
                continue
            st = politics.coalitions.strength(j, k)
            if st >= 1.0:
                coal[j, k] = 1.0 + 0.3 * min(2.0, st)
    return {"mode": "politics", "n": n, "has_habit": model is not None, "const": const, "habit_leader": habit_leader,
            "coal": coal}


def future_values(states, me: int, depth: int, levels, rolls, evaluator, robber=None, rng=None, deadline=None,
                  node_budget=None):
    """``core.future_values`` -> ``(values, nodes)``; ``None`` for a state the extension cannot represent."""
    try:
        vals, nodes = _core.future_values(states, int(me), int(depth), levels, rolls, evaluator, robber, rng, deadline,
                                          node_budget, False)
    except ValueError as exc:
        if _unsupported(exc):
            return None
        raise
    return [float(v) for v in vals], int(nodes)
