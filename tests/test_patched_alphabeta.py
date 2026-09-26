"""``alphabeta_fixvp`` (catanbot/bench/patched_alphabeta.py): OUR patch of catanatron 3.3's AlphaBetaPlayer.

Its value function must differ from catanatron's ``base_fn`` exactly on (a) its own hidden Victory Point cards and
(b) finished games (+/- WIN_BONUS for a won / lost game), and be the same float everywhere else.  On catanatron
3.2.1 (no AlphaBetaPlayer) the preset is refused with a one-line message and exit code 2.
"""
import importlib
import importlib.util
import math
import os
import random
import time

import pytest

pytest.importorskip("catanatron")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("bench_catanatron_fixvp",
                                               os.path.join(ROOT, "scripts", "bench_catanatron.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

from catanbot.bench import catanatron_adapter as AD  # noqa: E402

needs_33 = pytest.mark.skipif(not AD.API_33, reason=f"catanatron {AD.CATANATRON_VERSION} ships no AlphaBetaPlayer")
needs_321 = pytest.mark.skipif(AD.API_33, reason="catanatron 3.3 ships AlphaBetaPlayer")


@needs_321
def test_refused_cleanly_on_321():
    with pytest.raises(bench.OpponentUnavailable) as ex:
        bench.resolve_opponent("alphabeta_fixvp")
    msg = ex.value.message
    assert ex.value.code == 2 and "\n" not in msg
    assert f"not available on catanatron {AD.CATANATRON_VERSION}" in msg and "alphabeta_fixvp" in msg
    assert "3.3" in msg
    with pytest.raises(ImportError, match="our patch"):
        importlib.import_module("catanbot.bench.patched_alphabeta")


@needs_33
def test_preset_is_our_labelled_subclass_with_catanatrons_defaults():
    from catanatron.models.player import Color
    from catanatron.players.minimax import AlphaBetaPlayer
    from catanbot.bench import patched_alphabeta as P
    cls = bench.resolve_opponent("alphabeta_fixvp")
    assert cls is P.FixVPAlphaBetaPlayer and issubclass(cls, AlphaBetaPlayer) and cls is not AlphaBetaPlayer
    assert cls.__module__ == "catanbot.bench.patched_alphabeta" and "not Catanatron" in cls.LABEL
    assert bench.opponent_path("alphabeta_fixvp") == "catanbot.bench.patched_alphabeta:FixVPAlphaBetaPlayer"
    assert all("alphabeta_fixvp" not in names for names in bench.LADDERS.values())
    # the existing presets are untouched
    assert bench.PRESETS["alphabeta"] == "catanatron.players.minimax:AlphaBetaPlayer"
    assert bench.resolve_opponent("alphabeta") is AlphaBetaPlayer and AlphaBetaPlayer.use_value_function is None
    # only the value function is ours: search and decision code are catanatron's
    assert cls.decide is AlphaBetaPlayer.decide and cls.alphabeta is AlphaBetaPlayer.alphabeta
    assert cls.get_actions is AlphaBetaPlayer.get_actions and cls.Params is AlphaBetaPlayer.Params
    p, stock = cls(Color.RED), AlphaBetaPlayer(Color.RED)
    assert p.params == stock.params
    assert (p.params.depth, p.params.value_fn, p.params.prunning, p.params.epsilon) == (2, "base", False, None)
    assert p.value_fn_builder_name == "base_fn" and p.vp_weight() == 3e14
    # --opponent-params still work as for alphabeta
    make = bench.opponent_factory(cls, {"depth": "1"}, "alphabeta_fixvp")
    assert make(Color.BLUE).params.depth == 1 and isinstance(make(Color.BLUE), cls)


def _states(seed, every=40, limit=40):
    """Copies of a real 3.3 game (4 random players) every ``every`` plies (at most ``limit``) and its final state
    (the game is played to its end; catanatron's set iteration makes the game depend on PYTHONHASHSEED)."""
    from catanatron.game import Game
    from catanatron.models.player import RandomPlayer
    game = Game([RandomPlayer(c) for c in AD.COLORS], seed=seed)
    out, k = [], 0
    while game.winning_color() is None and k < 30000:
        game.play_tick()
        k += 1
        if k % every == 0 and len(out) < limit:
            out.append(game.copy())
    out.append(game.copy())
    return out


def _base_and_patched(game, color):
    from catanatron.players.value import DEFAULT_WEIGHTS, base_fn
    from catanbot.bench.patched_alphabeta import FixVPAlphaBetaPlayer
    return base_fn(DEFAULT_WEIGHTS)(game, color), FixVPAlphaBetaPlayer(color).value_function(game, color)


@needs_33
def test_value_differs_from_base_fn_exactly_on_hidden_vp_and_terminal_wins():
    from catanatron.state_functions import player_key
    from catanbot.bench.patched_alphabeta import WIN_BONUS, own_hidden_vps
    w = 3e14
    seen = {"same": 0, "hidden": 0, "won": 0, "lost": 0}
    for seed in range(3, 13):
        if seen["same"] > 50 and seen["hidden"] >= 10 and seen["won"] >= 2:
            break
        for g in _states(seed):
            winner = g.winning_color()
            for c in g.state.colors:
                base, patched = _base_and_patched(g, c)
                hidden = own_hidden_vps(g, c)
                if hidden == 0 and winner is None:
                    assert patched == base            # the same float
                    seen["same"] += 1
                    continue
                want = hidden * w + (0.0 if winner is None else (WIN_BONUS if winner == c else -WIN_BONUS))
                assert patched - base == pytest.approx(want, abs=8.0)
                seen["hidden"] += hidden > 0
                seen["won"] += winner == c
                seen["lost"] += winner is not None and winner != c
    assert seen["same"] > 50 and seen["hidden"] >= 10 and seen["won"] >= 2 and seen["lost"] >= 6

    # (a) injected: a VP card in its own hand counts at the public weight, an opponent's does not
    g = _states(7, every=200, limit=1)[0]
    assert g.winning_color() is None
    me, other = g.state.colors[0], g.state.colors[1]
    before_me, before_other = _base_and_patched(g, me), _base_and_patched(g, other)
    assert before_me[0] == before_me[1] or own_hidden_vps(g, me)
    ps, key = g.state.player_state, player_key(g.state, me)
    ps[f"{key}_VICTORY_POINT_IN_HAND"] += 1
    ps[f"{key}_ACTUAL_VICTORY_POINTS"] += 1
    base, patched = _base_and_patched(g, me)
    assert base - before_me[0] == pytest.approx(10.0)         # base_fn sees one more dev card (hand_devs), no point
    assert patched - base == pytest.approx((own_hidden_vps(g, me)) * w, abs=1.0)
    assert _base_and_patched(g, other) == before_other      # an opponent's values: unchanged

    # (b) injected: a finished game, won on public points, and one won with a hidden card
    for public, hidden in ((10, 0), (9, 1)):
        g2 = g.copy()
        ps2 = g2.state.player_state
        for c in g2.state.colors:
            k = player_key(g2.state, c)
            ps2[f"{k}_VICTORY_POINT_IN_HAND"] = 0
            ps2[f"{k}_ACTUAL_VICTORY_POINTS"] = ps2[f"{k}_VICTORY_POINTS"] = min(ps2[f"{k}_VICTORY_POINTS"], 7)
        ps2[f"{key}_VICTORY_POINTS"] = public
        ps2[f"{key}_ACTUAL_VICTORY_POINTS"] = public + hidden
        ps2[f"{key}_VICTORY_POINT_IN_HAND"] = hidden
        assert g2.winning_color() == me
        base, patched = _base_and_patched(g2, me)
        assert patched - base == pytest.approx(hidden * w + WIN_BONUS, abs=8.0)
        for c in g2.state.colors[1:]:
            base, patched = _base_and_patched(g2, c)
            assert patched - base == pytest.approx(-WIN_BONUS, abs=8.0)
        assert WIN_BONUS > 12 * w                  # a win outranks every unfinished position


@needs_33
def test_search_uses_the_patched_value_and_stock_alphabeta_does_not():
    from catanatron.players.minimax import AlphaBetaPlayer, DebugStateNode
    from catanbot.bench.patched_alphabeta import FixVPAlphaBetaPlayer
    g = next(s for s in _states(3) if s.winning_color() is None)
    c = g.state.colors[0]
    ps = g.state.player_state
    ps["P0_VICTORY_POINT_IN_HAND"] += 1
    ps["P0_ACTUAL_VICTORY_POINTS"] += 1
    base, patched = _base_and_patched(g, c)
    for cls, want in ((FixVPAlphaBetaPlayer, patched), (AlphaBetaPlayer, base)):
        p = cls(c)
        act, v = p.alphabeta(g.copy(), 0, -math.inf, math.inf, time.time() + 60, DebugStateNode("0", c))
        assert act is None and v == want
    # the bench's value / fair trade rule reads the player's own (patched) function; catanatron's players as before
    fix = AD.BenchOpponent(FixVPAlphaBetaPlayer(c), "value")
    assert fix.value_fn() == fix.inner.value_function
    assert AD.BenchOpponent(AlphaBetaPlayer(c), "value").value_fn().__qualname__ == "base_fn.<locals>.fn"


@needs_33
def test_plays_a_game_through_the_bench():
    from catanatron.models.player import Color  # noqa: F401
    res = bench.run_one((0, 426001, "heuristic:temp=0", "alphabeta_fixvp", 10, 7, "off", {"depth": "1"}))
    assert res["winner_seat"] in range(4) and res["stats"]["errors"] == 0
