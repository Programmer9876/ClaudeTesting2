"""Robber leaf corrections (docs/PRIORITY_PLAN.md step 5, "Robber dynamics"; the full design is
``docs/designs/priority_areas_2026-09-26.json`` designs[3]).  Off by default.

``heuristic.static_value`` removes ``0.55 x pv_i(h)`` from every player blocked by the robber on hex ``h``
(``pv_i(h)`` = the player's pips on ``h`` x demand x sqrt(scarcity), exactly ``robber.production_blocked``), as if
the block were permanent.  This module is one :mod:`catanbot.corrections` hub provider, :class:`RobberContext`, with
three terms in static points (10 = 1 VP), each restoring or moving a share of that value:

* **R1a persistence** (``PERSIST_W``): a block on a player who will kick it at their next turn start is mostly
  undone before it costs production.  The would-kick set mirrors the knight rule (``robber.should_play_knight``
  rule 2: the block is worth at least ``KICK_MIN`` = 3.0 demand-weighted pips) and needs a known playable knight;
  each kicker kicks with ``KICK_LAMBDA`` (bots kick about 99.7% of the time, 0.95 is the conservative default).
  The first kicker in roll order ends the block after ``t`` more rolls, which avoids the share ``a^t`` of its
  loss (geometric lifetime, ``a = 1 - 1/R_REF``, ``R_REF`` = 4.34 rolls measured on the proof logs).  Every
  blocked player gets ``C = PERSIST_W x 0.55 x pv_i(h) x rho``, ``rho = sum_k P1_k a^t_k``.
* **R1b knight insurance** (``INSURANCE_W``, default 0): a held knight nearly cancels the next block, so it is
  worth more to a player likely to be robbed soon.  Centred on ``INS_OFFSET`` so that it moves knight value toward
  exposed positions instead of raising it.  The target model uses the opponent model's observed robber victims
  (with a trust ramp) and a prior from threat and blockable pips - never hand size.
* **R1c block duration** (``BLOCK_DUR_W``, default 0): restores a fixed share of every block (the leaf treats
  blocks as permanent); composes with R1a as ``rho + W (1 - rho)``.

Plus an optional retaliation weight ``RETAL_W`` (default 0; a politics-rule term deferred to human testing) and a
seat-rotated ``PLACEBO`` control.  Nothing here touches ``static_value``, ``steal_exposure_fast`` or placement (all
mirrored in C++): the hub adds the corrections in Python on top of the C++ static values, so no
``needs_python_evaluator`` and no C++ port.

Switch: ``SearchConfig.robber_corr`` (``search.robber_corr``, spec key ``robber_corr``, default 0).  With 0 no
context is built and no correction is computed, so the default bot is byte-identical.  ``robber_corr=1`` alone is
persistence only (``INSURANCE_W`` and ``BLOCK_DUR_W`` default to 0).  The module weights below are read only when
a context is built (frozen per search).

Information: a seat's knights count only when its cards are known (full information, self-play, our own seat).
In counted mode / the advisor the dev cards of some seats were *dealt* by a determinization; such seats count as
knight holders only through a supplied posterior ``P(knight)`` (:func:`knight_hint`), else not at all, so a
secret-VP leader is never spared.  ``search.search_determinized`` sets the hint for the advisor path; the counted
catanatron adapter sets it in step 6 (card counting), which also supplies the posterior.

Also here (shared with :mod:`catanbot.robber_metrics`, :mod:`catanbot.knightkick` and ``scripts/decision_shadow.py``):
the pure helpers :func:`hex_weight`, :func:`block_values`, :func:`roll_offsets`, :func:`knight_q` and
:func:`kick_plan` (the would-kick set of a robber position).
"""
from __future__ import annotations

import contextlib
import importlib
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import board as B
from .placement import RESOURCE_DEMAND, resource_scarcity
from .robber import hex_pips_for_player, threat
from .state import GameState, PHASE_GAME_OVER, PHASE_ROLL, PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT

# heuristic.static_value's weight of robber-blocked production (score += 0.55 pv + 0.25 pv_free; mirrored in
# cpp/heuristic.cpp).  Not a tunable: every term restores a share of exactly what the leaf removed.
BLOCK_SHARE = 0.55

# --- R1a persistence (search.robber_corr=1) ------------------------------------------------------------------------
PERSIST_W = 1.0       # weight of the persistence restore (1 = the calibrated model; 0 = off inside the provider)
KICK_LAMBDA = 0.95    # P(a would-kick holder kicks at their next turn start); bots measured 0.997
KICK_MIN = 3.0        # demand-weighted pips a block must be worth to be kicked (robber.should_play_knight rule 2)
R_REF = 4.34          # mean block lifetime in rolls (proof logs); a block survives a roll with 1 - 1/R_REF
RETAL_W = 0.0         # retaliation: the kicker's robber landing on us (politics rule: 0, deferred to humans)
PLACEBO = 0           # 1 = control arm: every seat gets the next seat's correction
# --- R1b knight insurance --------------------------------------------------------------------------------------------
INSURANCE_W = 0.0     # horizon multiplier (rounds of cover); 0 = off even with robber_corr=1; SPSA range 0-3
INS_AVERT = 0.85      # share of a block a held knight averts (0.18 of 1.34 cards lost with a knight)
INS_OFFSET = 0.0      # centring: the mean of D x P_hit over insured players at the shadow's leaves (0 = uncentred)
TGT_LEAD = 0.5        # prior target model: weight of "the clear leader" vs blockable pips
HIT_SHARE = 0.9       # share of a player's best blockable hex a hit denies (Catanatron: 7.7 of 8.5 pips)
KNIGHT_OFF = 0.5      # share of turns a knight holder spends it on offence (steal_exposure_fast's convention)
# --- R1c block duration ----------------------------------------------------------------------------------------------
BLOCK_DUR_W = 0.0     # share of every block's leaf value restored (0 = off); SPSA range 0-0.8
CACHE_CAP = 20000     # the per-search pv-table memo is cleared when it grows beyond this

_PARAM_NAMES = ("PERSIST_W", "KICK_LAMBDA", "KICK_MIN", "R_REF", "RETAL_W", "PLACEBO", "INSURANCE_W", "INS_AVERT",
                "INS_OFFSET", "TGT_LEAD", "HIT_SHARE", "KNIGHT_OFF", "BLOCK_DUR_W")

_K = B.DEV_KNIGHT


def current_params() -> Dict[str, float]:
    """The module weights now (a context freezes them at construction)."""
    g = globals()
    return {n: g[n] for n in _PARAM_NAMES}


# ---------------------------------------------------------------------------
# Information hint (counted mode / the advisor)
# ---------------------------------------------------------------------------
_HINT: Optional[Dict[str, Any]] = None


@contextlib.contextmanager
def knight_hint(dealt: Sequence[int] = (), p_knight: Optional[Dict[int, float]] = None) -> Iterator[None]:
    """Within the block, contexts built for a search treat the seats ``dealt`` (their dev cards were dealt by a
    determinization, not observed) as knight holders only with probability ``p_knight[seat]`` (a posterior from
    card counting; absent = 0).  Nests; restores the previous hint on exit.  RNG-free."""
    global _HINT
    prev = _HINT
    _HINT = {"dealt": frozenset(int(j) for j in dealt),
             "p_knight": {int(k): float(v) for k, v in (p_knight or {}).items()}}
    try:
        yield
    finally:
        _HINT = prev


def current_hint() -> Optional[Dict[str, Any]]:
    return _HINT


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def hex_weight(state: GameState, h: int, scarcity: Optional[Sequence[float]] = None) -> float:
    """Demand x sqrt(scarcity) of hex ``h``'s resource (0 for the desert / an unnumbered hex): the per-pip weight
    of ``robber.production_blocked`` and of static_value's pv."""
    res, num = state.hexes[h]
    if res == B.DESERT or num == 0:
        return 0.0
    sc = resource_scarcity(state) if scarcity is None else scarcity
    return RESOURCE_DEMAND[res] * (sc[res] ** 0.5)


def block_values(state: GameState, h: int, w: Optional[float] = None) -> List[float]:
    """``pv_i(h)`` for every seat: pips on ``h`` (cities double) x :func:`hex_weight`.  With ``h`` the robber hex
    this is ``robber.production_blocked`` for each seat, and ``0.55 x pv_i(h)`` is what static_value removed."""
    w = hex_weight(state, h) if w is None else w
    if w <= 0.0:
        return [0.0] * state.num_players
    return [hex_pips_for_player(state, h, i) * w for i in range(state.num_players)]


def roll_offsets(state: GameState) -> List[int]:
    """``t_k``: rolls before seat ``k``'s next turn start.  In the current player's roll phase the order is
    ``c, c+1, ...`` (``c`` may still play a knight before rolling, unless it already played a development card
    this turn: then its next chance is ``n`` rolls away); otherwise ``c+1, ..., c`` (``c`` has rolled)."""
    n = state.num_players
    c = state.current
    if state.phase == PHASE_ROLL:
        t = [(k - c) % n for k in range(n)]
        if state.dev_played_this_turn:
            t[c] = n
        return t
    return [(k - c - 1) % n for k in range(n)]


def knight_q(state: GameState, k: int, me: int, hint: Optional[Dict[str, Any]] = None) -> float:
    """Probability that seat ``k`` holds a knight it can play at its next turn start, as the searching player
    ``me`` may use it: 1/0 from its cards when they are known (our own seat, full information); a seat whose dev
    cards are unknown or were dealt by a determinization (``hint``) counts only through a supplied posterior."""
    p = state.players[k]
    if k != me:
        dealt = hint is not None and k in hint["dealt"]
        if dealt or not p.dev_known:
            return float(hint["p_knight"].get(k, 0.0)) if hint is not None else 0.0
    return 1.0 if p.dev_cards[_K] + p.dev_cards_new[_K] >= 1 else 0.0


def kick_plan(state: GameState, me: int, pv: Sequence[float], lam: float = None, kick_min: float = None,
              hint: Optional[Dict[str, Any]] = None) -> List[Tuple[int, int, float]]:
    """The would-kick set of the robber's current hex: ``[(t_k, k, kappa_k)]`` sorted by ``t_k`` for every seat with
    ``pv_k >= kick_min`` and ``knight_q > 0``; ``kappa_k = lam x q_k``.  ``pv`` = :func:`block_values` of the
    robber hex."""
    lam = KICK_LAMBDA if lam is None else lam
    kick_min = KICK_MIN if kick_min is None else kick_min
    out: List[Tuple[int, int, float]] = []
    t = None
    for k, v in enumerate(pv):
        if v <= 0.0 or v < kick_min:
            continue
        q = knight_q(state, k, me, hint)
        if q <= 0.0:
            continue
        if t is None:
            t = roll_offsets(state)
        out.append((t[k], k, lam * q))
    out.sort()
    return out


def first_kicks(plan: Sequence[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """``[(t_k, k, P1_k)]``: the probability that ``k`` is the first to kick (earlier rollers first)."""
    out = []
    surv = 1.0
    for t, k, kap in plan:
        out.append((t, k, kap * surv))
        surv *= 1.0 - kap
    return out


def restore_share(plan: Sequence[Tuple[int, int, float]], a: float) -> float:
    """``rho = sum_k P1_k a^t_k``: the expected share of the block's (permanent-view) loss a kick avoids."""
    return sum(p1 * a ** t for t, _k, p1 in first_kicks(plan))


def _osig(state: GameState) -> tuple:
    return tuple(tuple(sorted(p.settlements + p.cities)) for p in state.players)


# ---------------------------------------------------------------------------
# The hub provider
# ---------------------------------------------------------------------------
class RobberContext:
    """The robber terms for one ``Searcher.search`` call: a :class:`corrections.CorrectionHub` provider.

    Frozen at construction: the module weights, the hex weights (the board is fixed), the searching seat, the
    information hint and the opponent model's robber habits (:meth:`bind_model`).  Per leaf nothing is cached but
    the per-seat pv tables of every hex (keyed on every seat's buildings; only R1b / retaliation read them)."""

    priors = False     # no move-ordering hook

    def __init__(self, root: GameState, me: int, params: Optional[Dict[str, float]] = None,
                 hint: Optional[Dict[str, Any]] = None, model=None):
        P = dict(params) if params is not None else current_params()
        self.params = P
        self.me = int(me)
        self.n = root.num_players
        self.hint = hint
        self.persist_w = float(P["PERSIST_W"])
        self.lam = float(P["KICK_LAMBDA"])
        self.kick_min = float(P["KICK_MIN"])
        self.a = 1.0 - 1.0 / float(P["R_REF"]) if float(P["R_REF"]) > 0 else 0.0
        self.retal_w = float(P["RETAL_W"])
        self.placebo = int(P["PLACEBO"])
        self.ins_w = float(P["INSURANCE_W"])
        self.ins_avert = float(P["INS_AVERT"])
        self.ins_offset = float(P["INS_OFFSET"])
        self.tgt_lead = float(P["TGT_LEAD"])
        self.hit_share = float(P["HIT_SHARE"])
        self.knight_off = float(P["KNIGHT_OFF"])
        self.dur_w = float(P["BLOCK_DUR_W"])
        sc = resource_scarcity(root)
        self._hexw = [hex_weight(root, h, sc) for h in range(B.NUM_HEXES)]
        self._tables: Dict[tuple, List[List[float]]] = {}
        self._model = model
        self._habit: Optional[List[Tuple[float, List[float]]]] = None
        self.stats: Dict[str, Any] = {"evals": 0, "nonzero": 0, "blocked": 0, "kick_sets": 0, "rho_sum": 0.0,
                                      "ins_n": 0, "ins_sum_c": 0.0, "ins_sum_dp": 0.0, "table_misses": 0}

    @classmethod
    def for_search(cls, root: GameState, me: int, cfg) -> "RobberContext":
        return cls(root, me, hint=current_hint())

    def bind_model(self, model) -> None:
        """Use the searcher's opponent model (``None`` = the prior target model only).  Its observed robber victims
        per seat are frozen on first use as ``(tau_j, share_j[i])``, with the trust ramp ``tau_j = min(1, sum
        robbed_j / 3)`` (``OpponentProfile.robbed``: decayed counts by victim name)."""
        self._model = model
        self._habit = None

    def _habits(self, state: GameState) -> List[Tuple[float, List[float]]]:
        if self._habit is not None:
            return self._habit
        n = self.n
        out: List[Tuple[float, List[float]]] = []
        model = self._model
        for j in range(n):
            if model is None:
                out.append((0.0, [0.0] * n))
                continue
            prof = model.profiles.get(state.players[j].name or state.players[j].color)   # read only: no new profile
            robbed = prof.robbed if prof is not None else {}
            tot = sum(robbed.values())
            if tot <= 0.0:
                out.append((0.0, [0.0] * n))
                continue
            names = [(q.name or q.color) for q in state.players]
            out.append((min(1.0, tot / 3.0), [robbed.get(names[i], 0.0) / tot for i in range(n)]))
        self._habit = out
        return out

    # --- per-seat pv tables (R1b / retaliation) ---------------------------------------------------------------
    def pv_table(self, state: GameState) -> List[List[float]]:
        """``table[i][h] = pv_i(h)`` for every seat and hex (memo: every seat's buildings)."""
        key = _osig(state)
        got = self._tables.get(key)
        if got is not None:
            return got
        self.stats["table_misses"] += 1
        n = self.n
        table = [[0.0] * B.NUM_HEXES for _ in range(n)]
        for i in range(n):
            p = state.players[i]
            row = table[i]
            for v in p.settlements:
                for h in B.VERTEX_HEXES[v]:
                    row[h] += 1.0
            for v in p.cities:
                for h in B.VERTEX_HEXES[v]:
                    row[h] += 2.0
            for h in range(B.NUM_HEXES):
                if row[h]:
                    res, num = state.hexes[h]
                    row[h] = row[h] * B.PIPS[num] * self._hexw[h] if self._hexw[h] > 0.0 else 0.0
        if len(self._tables) >= CACHE_CAP:
            self._tables.clear()
        self._tables[key] = table
        return table

    def blockable(self, state: GameState, table: List[List[float]]) -> List[float]:
        """``bp_x = max over h' != robber of pv_x(h')``: the best block left for the next robber."""
        h0 = state.robber
        return [max([v for h, v in enumerate(row) if h != h0] or [0.0]) for row in table]

    def p_target(self, state: GameState, j: int, bp: Sequence[float], thr: Sequence[float]) -> List[float]:
        """``P_tgt_j(i)`` for every seat ``i`` (0 for ``j``): the prior (``TGT_LEAD`` on the clear leader by threat,
        the rest by blockable pips; never hand size) blended with ``j``'s observed victims by the trust ramp."""
        n = self.n
        others = [x for x in range(n) if x != j]
        if not others:
            return [0.0] * n
        leader = max(others, key=lambda x: thr[x])
        tot_bp = sum(bp[x] for x in others)
        tau, share = self._habits(state)[j]
        out = [0.0] * n
        for i in others:
            pb = bp[i] / tot_bp if tot_bp > 0.0 else 1.0 / len(others)
            prior = self.tgt_lead * (1.0 if i == leader else 0.0) + (1.0 - self.tgt_lead) * pb
            out[i] = (1.0 - tau) * prior + tau * share[i]
        return out

    def p_rob(self, state: GameState, j: int) -> float:
        """Chance seat ``j`` moves the robber on its next turn: a 7, or a known knight played offensively
        (``KNIGHT_OFF``; the steal_exposure_fast convention, reimplemented - that function is C++-mirrored)."""
        q = knight_q(state, j, self.me, self.hint)
        if state.dev_played_this_turn and j == state.current:
            q = 0.0
        return 1.0 / 6.0 + (5.0 / 6.0) * q * self.knight_off

    def insured(self, state: GameState, i: int, in_kick_set: bool) -> float:
        """Weight in [0, 1] that seat ``i`` holds a knight to spare: known cards need a knight beyond the one it
        will spend kicking the current block; a dealt / unknown seat counts only through a posterior (and never
        when it is itself a would-kick holder).  Only one knight counts."""
        p = state.players[i]
        if i != self.me:
            dealt = self.hint is not None and i in self.hint["dealt"]
            if dealt or not p.dev_known:
                if in_kick_set or self.hint is None:
                    return 0.0
                return float(self.hint["p_knight"].get(i, 0.0))
        k = p.dev_cards[_K] + p.dev_cards_new[_K] - (1 if in_kick_set else 0)
        return 1.0 if k >= 1 else 0.0

    # --- the correction ------------------------------------------------------------------------------------------
    def corrections(self, state: GameState) -> Optional[List[float]]:
        """Per-seat points (``None`` when every term is zero, which keeps the hub's bit-identical fast path)."""
        st = self.stats
        st["evals"] += 1
        if state.phase == PHASE_GAME_OVER or state.phase in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
            return None
        n = self.n
        C = [0.0] * n
        nz = False
        h = state.robber
        wh = self._hexw[h]
        pv = [hex_pips_for_player(state, h, i) * wh for i in range(n)] if wh > 0.0 else None
        plan: List[Tuple[int, int, float]] = []
        if pv is not None and any(pv):
            st["blocked"] += 1
            plan = kick_plan(state, self.me, pv, self.lam, self.kick_min, self.hint)
            rho_p = 0.0
            if plan:
                st["kick_sets"] += 1
                rho = restore_share(plan, self.a)
                st["rho_sum"] += rho
                rho_p = self.persist_w * rho
            share = rho_p + (self.dur_w * max(0.0, 1.0 - rho_p) if self.dur_w else 0.0)
            if share != 0.0:
                for i in range(n):
                    if pv[i] > 0.0:
                        C[i] += BLOCK_SHARE * pv[i] * share
                        nz = True
        if self.ins_w or (self.retal_w and plan):
            table = self.pv_table(state)
            bp = self.blockable(state, table)
            thr = [threat(state, x) for x in range(n)]
            tgt = [self.p_target(state, j, bp, thr) for j in range(n)]
            if self.ins_w:
                kick_set = {k for _t, k, _kap in plan}
                prob = [self.p_rob(state, j) for j in range(n)]
                for i in range(n):
                    w_ins = self.insured(state, i, i in kick_set)
                    if w_ins <= 0.0:
                        continue
                    p_safe = 1.0
                    for j in range(n):
                        if j != i:
                            p_safe *= 1.0 - prob[j] * tgt[j][i]
                    dp = self.hit_share * bp[i] * (1.0 - p_safe)
                    c = self.ins_w * BLOCK_SHARE * self.ins_avert * (dp - self.ins_offset) * w_ins
                    st["ins_n"] += 1
                    st["ins_sum_c"] += c
                    st["ins_sum_dp"] += dp
                    if c != 0.0:
                        C[i] += c
                        nz = True
            if self.retal_w and plan:
                me = self.me
                loss = BLOCK_SHARE * self.hit_share * bp[me]
                r = 0.0
                for t, k, p1 in first_kicks(plan):
                    if k != me:
                        r += p1 * self.a ** t * tgt[k][me]
                if r * loss:
                    C[me] -= self.retal_w * r * loss
                    nz = True
        if not nz:
            return None
        st["nonzero"] += 1
        if self.placebo:
            C = [C[(i + 1) % n] for i in range(n)]
        return C


# ---------------------------------------------------------------------------
# Tunable registration (called from the last lines of catanbot/tuning.py)
# ---------------------------------------------------------------------------
_ONLY = "only with search.robber_corr=1 (base spec ...,robber_corr=1)"


def register_tunables(registry: Dict[str, object]) -> None:
    """Add the robber switches (``search.robber_corr``, ``search.kick``), the ``robber_eval.*`` weights and the
    self-play seat observer "robber" (:class:`catanbot.robber_metrics.RobberCounters`)."""
    tuning = sys.modules.get("catanbot.tuning") or importlib.import_module("catanbot.tuning")
    Tunable = tuning.Tunable
    for attr, default, cands, parse, desc in (
            ("robber_corr", 0, [1], tuning._parse_int,
             "robber leaf corrections (catanbot/robber_eval.py; R1a persistence, plus R1b insurance / R1c block "
             "duration through their weights): a block on a would-kick knight holder is mostly temporary; "
             "0 = off (the default bot); budgeted at depth 1"),
            ("kick", 0.0, [0.95], tuning._parse_float,
             "knight-kick leaf chance node (catanbot/knightkick.py): with this probability the first blocked "
             "knight holder in roll order kicks the robber before our next turn (the alternative to robber_corr; "
             "never both); 0 = off; depth 1 only")):
        t = Tunable(name=f"search.{attr}", module="catanbot.search", attr=attr, default=default, kind="search",
                    candidates=list(cands), requires_search=True, requires_depth=1, spec_key=attr, parse=parse,
                    description=desc)
        registry[t.name] = t
    g = globals()
    for attr, cands, parse, meaning in (
            ("PERSIST_W", [0.0, 0.5], tuning._parse_float, "R1a persistence restore weight"),
            ("KICK_LAMBDA", [0.8, 1.0], tuning._parse_float, "R1a P(a would-kick knight holder kicks)"),
            ("KICK_MIN", [2.0, 4.0], tuning._parse_float, "R1a demand-weighted pips a block needs to be kicked"),
            ("R_REF", [2.5, 8.0], tuning._parse_float, "mean block lifetime in rolls (geometric keep curve)"),
            ("RETAL_W", [1.0], tuning._parse_float,
             "retaliation: the kicker's robber landing on us (politics rule; deferred to human testing)"),
            ("PLACEBO", [1], tuning._parse_int, "seat-rotated control: every seat gets the next seat's correction"),
            ("INSURANCE_W", [1.0, 2.0, 3.0], tuning._parse_float,
             "R1b knight insurance horizon weight (0 = off; an SPSA knob)"),
            ("INS_AVERT", [0.5, 1.0], tuning._parse_float, "R1b share of a block a held knight averts"),
            ("INS_OFFSET", [0.0, 2.0], tuning._parse_float, "R1b centring offset (mean D x P_hit, calibrated)"),
            ("TGT_LEAD", [0.25, 0.75], tuning._parse_float, "R1b prior weight of the clear leader as the target"),
            ("HIT_SHARE", [0.7, 1.0], tuning._parse_float, "R1b / retaliation share of the best blockable hex a hit denies"),
            ("KNIGHT_OFF", [0.25, 1.0], tuning._parse_float, "R1b share of turns a knight holder robs offensively"),
            ("BLOCK_DUR_W", [0.25, 0.5], tuning._parse_float,
             "R1c block-duration restore share (0 = off; an SPSA knob, 0-0.8)")):
        t = Tunable(name=f"robber_eval.{attr}", module=__name__, attr=attr, default=g[attr], kind="weight",
                    candidates=list(cands), requires_search=True, needs_python_evaluator=False, clear_caches=False,
                    parse=parse, description=f"{meaning}; {_ONLY}")
        registry[t.name] = t

    def _robber_counters(num_seats: int):
        from .robber_metrics import RobberCounters
        return RobberCounters(num_seats)

    tuning.register_seat_observer("robber", _robber_counters)   # self-play mechanism counters (observers=...)
