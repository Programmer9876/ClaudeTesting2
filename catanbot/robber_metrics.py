"""Observation-only robber counters for self-play games (``robber.shadow_metrics``, docs/PRIORITY_PLAN.md step 5).

:class:`RobberCounters` is a per-seat observer of ``tuning.play_paired_game`` (registered as the seat observer
"robber" by ``robber_eval.register_tunables``; pass ``observers=["robber"]``).  It reads the state before each
action (``on_action(state, action, player)``, called before the action is applied), never mutates it and never
draws a random number, so games are identical with and without it (tests/test_robber_metrics.py).  It is not a
bot feature: no tunable.  Against Catanatron the same mechanism metrics come from ``scripts/mechanics.py`` (the
action log); adapter stats keep only bot-internal counters (the plan's cross-area resolution).

Per seat (lists indexed by seat):

* ``rob_moves``: robber moves (a 7's MOVE_ROBBER or a PLAY_KNIGHT);
* ``rob_on_knight_holder``: moves whose hex blocks a would-kick knight holder (another seat whose block is worth
  at least ``robber_eval.KICK_MIN`` demand-weighted pips and who holds a knight: the knight rule), and
  ``rob_on_next_kicker`` those where such a holder kicks within one roll (``t <= 1``, the persistence gate G1's
  class);
* ``rob_on_leader``: moves onto a hex of an opponent with the top public VP (ties included, as scripts/mechanics.py);
* ``turn_starts``, ``blocked_turns`` (the robber blocks the seat at its turn start) and ``blocked_turns_knight``
  (... while it holds a playable knight);
* ``cards_lost_block``: production cards the robber denied the seat (a roll of the robber hex's number; the bank
  limit ignored), ``cards_lost_steal``: cards stolen from the seat;
* ``knights_played``, ``kicks`` (knights played while the robber blocked the player);
* ``knights_held_end`` (turns ended holding a playable knight) and ``knights_held_exposed`` (... as the strict
  public-VP leader, the natural robber target);
* ``army_end``: 1 for the Largest Army holder at the end of the game.
"""
from __future__ import annotations

from typing import Any, Dict, List

from . import actions as A
from . import board as B
from .robber_eval import block_values, kick_plan
from .state import GameState, PHASE_MAIN, PHASE_ROLL

_K = B.DEV_KNIGHT


class RobberCounters:
    KEYS = ("rob_moves", "rob_on_knight_holder", "rob_on_next_kicker", "rob_on_leader", "turn_starts",
            "blocked_turns", "blocked_turns_knight", "cards_lost_block", "cards_lost_steal", "knights_played", "kicks",
            "knights_held_end", "knights_held_exposed", "army_end")

    def __init__(self, num_seats: int):
        self.n = int(num_seats)
        self.c: Dict[str, List[int]] = {k: [0] * self.n for k in self.KEYS}
        self._roll_pending = False
        self._army = -1

    # --- helpers -------------------------------------------------------------------------------------------------
    @staticmethod
    def classify_move(state: GameState, action, mover: int) -> Dict[str, Any]:
        """The robber move ``action`` (MOVE_ROBBER / PLAY_KNIGHT) by ``mover`` from ``state``: its would-kick set
        on the new hex (``[(t, seat, kappa)]``, other seats only), whether a top-VP opponent is blocked, and the
        opponents' blocked value."""
        h = action[1]
        s = state.copy()
        s.robber = h
        if action[0] == A.PLAY_KNIGHT:
            s.dev_played_this_turn = True
        else:
            s.phase = PHASE_MAIN
        pv = block_values(s, h)
        plan = [x for x in kick_plan(s, mover, pv) if x[1] != mover]
        opp = [i for i in range(s.num_players) if i != mover]
        top = max(state.public_vp(i) for i in opp) if opp else 0
        on_leader = any(pv[i] > 0.0 and state.public_vp(i) == top for i in opp)
        return {"hex": h, "kick": plan, "on_leader": on_leader, "opp_pv": sum(pv[i] for i in opp)}

    def _roll_done(self, state: GameState) -> None:
        """Cards the robber denied on the roll just made (``state.dice``; nothing on a 7)."""
        self._roll_pending = False
        d = state.dice
        res, num = state.hexes[state.robber]
        if d == 7 or res == B.DESERT or num != d:
            return
        verts = B.HEX_VERTICES[state.robber]
        for i, p in enumerate(state.players):
            k = sum(1 for v in verts if v in p.settlements) + 2 * sum(1 for v in verts if v in p.cities)
            self.c["cards_lost_block"][i] += k

    # --- observer protocol ---------------------------------------------------------------------------------------
    def on_action(self, state: GameState, action, player: int) -> None:
        c = self.c
        if self._roll_pending:
            self._roll_done(state)
        self._army = state.largest_army_owner
        kind = action[0]
        if state.phase == PHASE_ROLL and state.dice == 0 and not state.dev_played_this_turn \
                and state.current == player:
            c["turn_starts"][player] += 1
            if block_values(state, state.robber)[player] > 0.0:
                c["blocked_turns"][player] += 1
                if state.players[player].dev_cards[_K] >= 1:
                    c["blocked_turns_knight"][player] += 1
        if kind == A.ROLL:
            if len(action) == 2:
                s = state.copy()
                s.dice = action[1]
                self._roll_done(s)
            else:
                self._roll_pending = True       # the dice are known at the next action
        elif kind in (A.MOVE_ROBBER, A.PLAY_KNIGHT) and len(action) >= 3:
            info = self.classify_move(state, action, player)
            c["rob_moves"][player] += 1
            c["rob_on_knight_holder"][player] += int(bool(info["kick"]))
            c["rob_on_next_kicker"][player] += int(any(t <= 1 for t, _k, _q in info["kick"]))
            c["rob_on_leader"][player] += int(info["on_leader"])
            victim = action[2]
            if victim is not None and 0 <= victim < self.n:
                vp = state.players[victim]
                if (vp.total_resources if vp.hand_known else vp.hand_size) > 0:
                    c["cards_lost_steal"][victim] += 1
            if kind == A.PLAY_KNIGHT:
                c["knights_played"][player] += 1
                if block_values(state, state.robber)[player] > 0.0:
                    c["kicks"][player] += 1
                # Largest Army after this knight (engine._update_largest_army), so a game-ending knight counts
                p = state.players[player]
                la = state.largest_army_owner
                if p.played_knights + 1 >= 3 and (la == -1 or (la != player and p.played_knights + 1 >
                                                              state.players[la].played_knights)):
                    self._army = player
        elif kind == A.END_TURN:
            p = state.players[player]
            if p.dev_cards[_K] >= 1:
                c["knights_held_end"][player] += 1
                vps = [state.public_vp(i) for i in range(self.n)]
                if all(vps[player] > vps[i] for i in range(self.n) if i != player):
                    c["knights_held_exposed"][player] += 1

    def result(self) -> Dict[str, List[int]]:
        out = {k: list(v) for k, v in self.c.items()}
        if 0 <= self._army < self.n:
            out["army_end"][self._army] = 1
        return out
