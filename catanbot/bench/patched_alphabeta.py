"""``alphabeta_fixvp``: OUR patch of Catanatron's ``AlphaBetaPlayer`` (docs/SCRUTINY.md Q22).

THIS IS NOT CATANATRON'S PLAYER.  It is catanbot's modification of catanatron 3.3's
``catanatron.players.minimax.AlphaBetaPlayer``, written to check how much of our edge over AlphaBeta rests on two
gaps of the value function it scores positions with (``players/value.py``, ``base_fn``).  Report its results as
"AlphaBeta patched by us (alphabeta_fixvp)", never as Catanatron's AlphaBeta.

The two patches, and nothing else:

(a) **Its own hidden victory points.**  ``base_fn`` weighs the PUBLIC points (``P{i}_VICTORY_POINTS``, weight
    ``public_vps`` = 3e14), so a Victory Point card is worth 0 even in its own hand, which it can see.  The patch
    scores its own points as ``ACTUAL_VICTORY_POINTS`` (public points plus its own VP cards) at the same weight:
    ``+ public_vps x (ACTUAL - public)`` for its own colour.  ``base_fn`` never scores an opponent's points, so
    opponents' points stay public (their hidden VP cards stay invisible to it, as before).
(b) **Finished games.**  ``AlphaBetaPlayer.alphabeta`` scores a node where ``game.winning_color()`` is set with
    the same function and no win bonus.  The patch adds :data:`WIN_BONUS` (1e16, about 33 public points) when the
    winner is its own colour and subtracts it when another colour has won.  ``winning_color`` is catanatron's
    (actual points >= ``vps_to_win``); the unpatched player already stops its search there.

Everything else is catanatron's own code and defaults: ``decide`` / ``alphabeta`` are inherited unchanged (the
value function is swapped through catanatron's hook for it, ``use_value_function`` / ``value_function``), the
``Params`` defaults (depth 2, ``value_fn`` "base" = ``base_fn(DEFAULT_WEIGHTS)``, no pruning, no epsilon), the
chance expansion (``expand_spectrum``) and the 20 s search deadline.  ``--opponent-params`` works as for
``alphabeta``; with ``value_fn=contender`` the base is catanatron's ``contender_fn`` and (a) uses its
``public_vps`` weight.

Catanatron 3.2.1 ships no ``AlphaBetaPlayer`` (no ``catanatron.players.minimax``): importing this module there
raises ``ImportError``, which ``scripts/bench_catanatron.py`` (and ``scripts/ablate_catanatron.py`` through it)
reports as "not available on catanatron 3.2.1" with exit code 2.
"""
from __future__ import annotations

from typing import Callable, Optional

try:
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanatron.players.value import CONTENDER_WEIGHTS, DEFAULT_WEIGHTS, get_value_fn
    from catanatron.state_functions import player_key
except ImportError as _ex:   # catanatron 3.2.1 (PyPI wheel): no alpha-beta player to patch
    raise ImportError("alphabeta_fixvp is our patch of catanatron 3.3's AlphaBetaPlayer "
                      "(catanatron.players.minimax), which this catanatron does not ship "
                      f"({_ex}); use the 3.3 engine (/home/user/venv_cat33/bin/python)") from _ex

#: Added to the value of a finished game its own colour won, subtracted when another colour won.  Larger than
#: any unfinished position can score (about 33 public points at 3e14; a game ends by 10-12 points), and small
#: enough that the base terms stay resolved (float spacing at 1.3e16 is 2; production moves by ~1e6).
WIN_BONUS = 1e16

#: Preset name in scripts/bench_catanatron.py (PRESETS) and scripts/ablate_catanatron.py.
PRESET = "alphabeta_fixvp"


def own_hidden_vps(game, color) -> int:
    """Victory points of ``color`` that are not public: its Victory Point cards (``ACTUAL - VICTORY_POINTS``)."""
    ps = game.state.player_state
    key = player_key(game.state, color)
    return int(ps[f"{key}_ACTUAL_VICTORY_POINTS"]) - int(ps[f"{key}_VICTORY_POINTS"])


def patch_value(base_value: float, game, color, vp_weight: float) -> float:
    """``base_value`` (catanatron's value of ``game`` for ``color``) with patches (a) and (b).

    Equal to ``base_value`` (the same float) unless ``color`` holds hidden points or the game is over."""
    v = base_value
    hidden = own_hidden_vps(game, color)
    if hidden:
        v += hidden * vp_weight                       # (a) its own VP cards, at the public-points weight
    winner = game.winning_color()
    if winner is not None:
        v += WIN_BONUS if winner == color else -WIN_BONUS   # (b) a finished game: won / lost
    return v


class FixVPAlphaBetaPlayer(AlphaBetaPlayer):
    """catanatron 3.3's ``AlphaBetaPlayer`` with OUR value-function patch (module doc): it counts its own hidden
    Victory Point cards and scores a won / lost finished game with +/- :data:`WIN_BONUS`.  Not Catanatron's
    player; everything but ``value_function`` is inherited."""

    LABEL = "AlphaBeta + fix-VP (catanbot's patch, not Catanatron's player)"
    #: catanatron's hook (``AlphaBetaPlayer.alphabeta`` -> ``get_value_fn(..., self.value_function)``)
    use_value_function = True

    _base: Optional[Callable] = None

    def base_value_fn(self) -> Callable:
        """Catanatron's own value function for these params (``get_value_fn(builder, weights)``, as unpatched)."""
        if self._base is None:
            self._base = get_value_fn(self.value_fn_builder_name, self.params.weights)
        return self._base

    def vp_weight(self) -> float:
        """The ``public_vps`` weight the base function uses (``get_value_fn``: ``base_fn`` always takes
        ``DEFAULT_WEIGHTS``; ``contender_fn`` takes the params' weights, else ``CONTENDER_WEIGHTS``)."""
        if self.value_fn_builder_name == "base_fn":
            return float(DEFAULT_WEIGHTS["public_vps"])
        return float((self.params.weights or CONTENDER_WEIGHTS)["public_vps"])

    def value_function(self, game, p0_color):
        return patch_value(self.base_value_fn()(game, p0_color), game, p0_color, self.vp_weight())
