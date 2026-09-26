"""Knight-kick leaf chance node (``search.knight_kick``, docs/PRIORITY_PLAN.md step 5): the alternative to the
robber persistence correction (``robber_eval``, R1a).  Off by default.

At depth 1 a finished leaf where the robber blocks a player who will kick it is valued by static_value as a
lasting ``0.55 x pv`` production loss for that player.  With ``SearchConfig.kick = KICK > 0`` such a leaf becomes an
exact chance node in the evaluator's own units (the :class:`catanbot.corrections.CorrectionHub` leaf-chance hook):

    value(s) = (1 - KICK q) v(s) + KICK q v(s')

* ``j*`` = the first player in roll order (``robber_eval.roll_offsets``) other than us whose block on the robber
  hex is worth at least ``KICK_MIN`` demand-weighted pips (the knight rule, ``robber.should_play_knight`` rule 2 -
  the same gate as R1a) and who holds a knight (``q`` = 1 for known cards; a seat whose dev cards were dealt by a
  determinization counts only through a supplied posterior, ``robber_eval.knight_hint``);
* ``s'`` = ``s`` with ``j*``'s knight played: the robber moved to ``j*``'s best robber hex (``robber.best_robber_move``
  on the root with the robber on ``h``, no politics or model weights, cached per ``(h, j*)`` per search), the knight
  spent (playable cards first), ``played_knights + 1`` and Largest Army updated (``engine._update_largest_army``).
  The steal is ignored;
* ``v`` = the hub's value (every provider applies, e.g. win-path races), batched per search level.

So the search stops paying for blocks that will not last, and sees where the kicked robber lands and the kicker's
Largest Army progress.  Depth 1 only: at depth >= 2 the simulated opponents' turns play knights themselves.  Never
stacked with R1a in one arm (the plan: knight_kick is R1a's fallback if R1a fails shadow gate G1 or G4).  Python
only; static_value, placement and the C++ evaluator are untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import board as B
from . import engine as E
from .placement import resource_scarcity
from .robber import best_robber_move, hex_pips_for_player
from .robber_eval import current_hint, hex_weight, knight_q, roll_offsets
from .state import GameState, PHASE_GAME_OVER

KICK_MIN = 3.0     # mirrors robber.should_play_knight rule 2 (and robber_eval.KICK_MIN's default); not a tunable

_K = B.DEV_KNIGHT


def kick_state(s: GameState, j: int, target: Tuple[int, int]) -> GameState:
    """``s`` after seat ``j`` plays a knight moving the robber to ``target = (hex, victim)`` (the steal ignored)."""
    s2 = s.copy()
    s2.robber = int(target[0])
    p = s2.players[j]
    if p.dev_cards[_K] > 0:
        p.dev_cards[_K] -= 1
    elif p.dev_cards_new[_K] > 0:
        p.dev_cards_new[_K] -= 1
    if p.dev_known:
        p.dev_count = sum(p.dev_cards) + sum(p.dev_cards_new)
    elif p.dev_count > 0:
        p.dev_count -= 1
    p.played_knights += 1
    E._update_largest_army(s2, j)
    return s2


class KickChance:
    """A chance provider of the hub (``leaf_outcomes``) for one ``Searcher.search`` call."""

    def __init__(self, root: GameState, me: int, kick: float, hint: Optional[Dict[str, Any]] = None):
        self.root = root
        self.me = int(me)
        self.kick = float(kick)
        self.hint = hint
        sc = resource_scarcity(root)
        self._hexw = [hex_weight(root, h, sc) for h in range(B.NUM_HEXES)]
        self._targets: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.stats: Dict[str, Any] = {"asked": 0, "kicks": 0, "target_misses": 0}

    @classmethod
    def for_search(cls, root: GameState, me: int, cfg) -> "KickChance":
        return cls(root, me, float(getattr(cfg, "kick", 0.0)), hint=current_hint())

    def target(self, h: int, j: int) -> Tuple[int, int]:
        """``j``'s kick: its best robber move from the root with the robber on ``h`` (cached per ``(h, j)``)."""
        key = (h, j)
        got = self._targets.get(key)
        if got is None:
            self.stats["target_misses"] += 1
            r = self.root.copy()
            r.robber = h
            hh, victim, _ = best_robber_move(r, j)
            got = (hh, victim)
            self._targets[key] = got
        return got

    def kicker(self, s: GameState, me: int) -> Optional[Tuple[int, float]]:
        """``(j*, q)``: the first would-kick knight holder on the robber hex in roll order, or None."""
        h = s.robber
        w = self._hexw[h]
        if w <= 0.0:
            return None
        t = None
        best = None
        for j in range(s.num_players):
            if j == me or hex_pips_for_player(s, h, j) * w < KICK_MIN:
                continue
            q = knight_q(s, j, me, self.hint)
            if q <= 0.0:
                continue
            if t is None:
                t = roll_offsets(s)
            if best is None or t[j] < t[best[0]]:
                best = (j, q)
        return best

    def leaf_outcomes(self, s: GameState, me: int) -> Optional[List[Tuple[float, GameState]]]:
        self.stats["asked"] += 1
        if self.kick <= 0.0 or s.phase == PHASE_GAME_OVER or s.current == me:
            return None
        got = self.kicker(s, me)
        if got is None:
            return None
        j, q = got
        s2 = kick_state(s, j, self.target(s.robber, j))
        p = min(1.0, self.kick * q)
        self.stats["kicks"] += 1
        return [(1.0 - p, s), (p, s2)]
