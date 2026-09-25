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
           "engine_apply", "engine_apply_inplace", "engine_apply_forced", "random_playout_fast"]

# Entry points every usable build provides; an older build missing one is stale and gets disabled by verify().
_REQUIRED = ("extract_batch", "extract", "longest_road_length", "static_values", "static_value", "heuristic_evaluate",
             "UnsupportedStateError", "legal_actions", "apply", "apply_inplace", "apply_forced", "random_playout_fast")


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
