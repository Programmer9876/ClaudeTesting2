"""Counter-offers (Colonist.io) and out-of-turn trade analysis: which counter to make, and the advisor report.

Rules: the variant ``GameState.allow_counters`` (off by default), protocol in ``catanbot/engine.py``'s
docstring - a responder to the current player's offer may answer ``(COUNTER_TRADE, give, get)`` (from the
counterer's side); the current player sees each counter as an ordinary pending offer and may accept it (the
trade executes and the round closes) or reject it (the original offer continues).

* :func:`rank_counters` - the bot side.  Candidates are the engine's bounded edits of the offer
  (``engine.counter_candidates``: ask one more card, give one fewer, swap which card we give).  A counter is
  dropped when we cannot pay it, when it would hand the leader a build (``trading.offer_is_feeding_leader``)
  and entirely in the late-game situations where ``trading.should_accept`` refuses to trade with the proposer
  at all (they are ahead late, or about to win).  The rest are ranked by ``P(proposer accepts)^(1/aggr) x our
  gain`` (``OpponentModel.predict_counter_accept``; ``aggr`` = ``SearchConfig.counter_aggr``, 1 = expected
  gain) with the gain measured by the search's own evaluator.  The search expands the best
  ``counter_candidates`` of them as chance nodes (``Searcher._counter_outcomes``).
* :func:`offer_response_report` - the advisor (``cli.py --offer``): accept / reject / the best counters,
  each valued after the rest of the proposer's turn (``Searcher._finish_proposer_turn``) next to its value
  at the moment cards change hands, with the reason: what the trade lets the proposer build this turn,
  whether somebody else would take the deal if we refuse, and P(they take the counter).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from . import engine as E
from .actions import Action
from .state import GameState, PHASE_GAME_OVER, PHASE_TRADE_RESPONSE, TradeOffer


def _pname(state: GameState, i: int) -> str:
    p = state.players[i]
    return p.name or p.color


def counter_blocked(state: GameState, me: int, proposer: int) -> str:
    """Why we must not counter this proposer at all ("" when countering is allowed).

    The late-game rules of ``trading.should_accept``: no trades with a player about to win, nor (from 60 %
    game stage) with a player at >= 7 VP who is not behind us.
    """
    from .opponent_model import game_stage
    from .robber import estimated_vp
    pvp = estimated_vp(state, proposer)
    if pvp >= B.VP_TO_WIN - 1:
        return f"{_pname(state, proposer)} is about to win ({pvp:.0f} VP)"
    if game_stage(state) >= 0.6 and pvp >= estimated_vp(state, me) and pvp >= 7:
        return f"late game: no trades with a player ahead of us ({pvp:.0f} VP)"
    return ""


def describe_edit(offer: TradeOffer, give: Sequence[int], get: Sequence[int]) -> str:
    """How a counter differs from accepting ``offer`` (we would pay ``offer.get`` and receive ``offer.give``)."""
    parts = []
    base_give, base_get = offer.get, offer.give
    for r in range(5):
        if get[r] > base_get[r]:
            parts.append(f"asks {get[r] - base_get[r]} more {B.RESOURCE_NAMES[r]}")
        elif get[r] < base_get[r]:
            parts.append(f"asks {base_get[r] - get[r]} {B.RESOURCE_NAMES[r]} fewer")
    fewer = [r for r in range(5) if give[r] < base_give[r]]
    more = [r for r in range(5) if give[r] > base_give[r]]
    if fewer and more:
        parts.append("gives " + "/".join(B.RESOURCE_NAMES[r] for r in more) + " instead of "
                     + "/".join(B.RESOURCE_NAMES[r] for r in fewer))
    else:
        for r in fewer:
            parts.append(f"gives {base_give[r] - give[r]} {B.RESOURCE_NAMES[r]} fewer")
        for r in more:
            parts.append(f"gives {give[r] - base_give[r]} more {B.RESOURCE_NAMES[r]}")
    return ", ".join(parts) or "same deal"


def _after_trade(state: GameState, a: int, a_gives: Sequence[int], b: int, b_gives: Sequence[int]) -> GameState:
    s2 = state.copy()
    pa, pb = s2.players[a].resources, s2.players[b].resources
    for r in range(5):
        pa[r] += b_gives[r] - a_gives[r]
        pb[r] += a_gives[r] - b_gives[r]
    return s2


def counter_accept_probability(state: GameState, me: int, give: Sequence[int], get: Sequence[int], model=None,
                               belief=None, politics=None, alternatives: bool = False) -> float:
    """P(the proposer of the pending offer accepts our counter ``(give, get)`` (our side))."""
    offer = state.pending_trade
    cur = offer.proposer
    if model is not None:
        return model.predict_counter_accept(state, cur, offer, give, get, counterer=me, belief=belief,
                                            politics=politics, alternatives=alternatives)
    from .trading import should_accept
    ctr = TradeOffer(me, list(give), list(get))
    ok, _ = should_accept(state, cur, ctr, politics=politics)
    return (0.75 if ok else 0.1) * (0.5 if alternatives else 1.0)


def rank_counters(state: GameState, me: int, counters: Optional[Sequence[Action]] = None, evaluator=None,
                  model=None, belief=None, politics=None, aggr: float = 1.0) -> List[dict]:
    """Our counters to the pending offer, best first: ``{"action", "p_accept", "gain", "score", "reason"}``.

    ``counters`` defaults to ``engine.counter_candidates(state, me)``.  ``gain`` is the evaluator's value of
    the executed counter minus the value now (our perspective; with no evaluator a card-value estimate), and
    ``score = p_accept ** (1 / aggr) * max(0, gain)``.  Unsafe counters (we cannot pay, they would hand the
    leader a build) and every counter in :func:`counter_blocked` situations are left out.
    """
    offer = state.pending_trade
    if offer is None or offer.origin is not None or state.phase != PHASE_TRADE_RESPONSE:
        return []
    cur = offer.proposer
    if cur == me or counter_blocked(state, me, cur):
        return []
    if counters is None:
        counters = E.counter_candidates(state, me)
    from .trading import offer_is_feeding_leader
    hand = state.players[me].resources
    rows: List[dict] = []
    for a in counters:
        if a[0] != A.COUNTER_TRADE:
            continue
        give, get = a[1], a[2]
        if any(hand[r] < give[r] for r in range(5)):
            continue
        if offer_is_feeding_leader(state, me, cur, give)[0]:
            continue
        p = counter_accept_probability(state, me, give, get, model=model, belief=belief, politics=politics)
        rows.append({"action": a, "p_accept": float(p), "gain": 0.0, "score": 0.0, "reason": ""})
    if not rows:
        return []
    if evaluator is not None:
        states = [state] + [_after_trade(state, me, d["action"][1], cur, d["action"][2]) for d in rows]
        vals = [float(v) for v in evaluator.evaluate(states, [me] * len(states))]
        for d, v in zip(rows, vals[1:]):
            d["gain"] = v - vals[0]
    else:
        from .opponent_model import our_resource_values
        ours = our_resource_values(state, me)
        for d in rows:
            give, get = d["action"][1], d["action"][2]
            d["gain"] = 0.02 * sum(ours[r] * (get[r] - give[r]) for r in range(5))
    expo = 1.0 / max(1e-3, float(aggr))
    for d in rows:
        d["score"] = (d["p_accept"] ** expo) * max(0.0, d["gain"])
        give, get = d["action"][1], d["action"][2]
        d["reason"] = (f"counter ({describe_edit(offer, give, get)}): {_pname(state, cur)} takes it with "
                       f"~{d['p_accept']:.0%}; {d['gain']:+.3f} for us if taken")
    rows.sort(key=lambda d: (-d["score"], -d["gain"]))
    return rows


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------
def proposer_builds(before: GameState, after: GameState, j: int) -> List[Tuple[str, int]]:
    """What player ``j`` built between two states: ``[(kind, where)]`` (settlement / city vertex, road edge,
    dev card -1, "win" when the game ended with ``j`` winning)."""
    out: List[Tuple[str, int]] = []
    pb, pa = before.players[j], after.players[j]
    for v in pa.cities:
        if v not in pb.cities:
            out.append(("city", v))
    for v in pa.settlements:
        if v not in pb.settlements and v not in pa.cities:
            out.append(("settlement", v))
    for e in pa.roads:
        if e not in pb.roads:
            out.append(("road", e))
    nd = pa.total_dev - pb.total_dev
    for _ in range(max(0, nd)):
        out.append(("dev card", -1))
    if after.phase == PHASE_GAME_OVER and after.winner == j:
        out.append(("win", -1))
    return out


def _spot_numbers(state: GameState, v: int) -> str:
    nums = sorted(state.hexes[h][1] for h in B.VERTEX_HEXES[v] if state.hexes[h][1])
    return "-".join(str(n) for n in nums) or "desert"


def _build_text(state: GameState, builds: Sequence[Tuple[str, int]], me: int, targets: Dict[int, int]) -> str:
    parts = []
    roads = sum(1 for k, _ in builds if k == "road")
    for kind, where in builds:
        if kind == "win":
            parts.insert(0, "win the game this turn")
        elif kind == "settlement":
            spot = _spot_numbers(state, where)
            tag = f"the {spot} spot you are heading for" if where in targets else f"a {spot} spot"
            parts.append(f"a settlement on vertex {where} ({tag})")
        elif kind == "city":
            parts.append(f"a city on vertex {where} ({_spot_numbers(state, where)})")
        elif kind == "dev card":
            parts.append("a development card")
    if roads:
        parts.append(f"{roads} road{'s' if roads > 1 else ''}")
    return ", ".join(parts)


def offer_response_report(state: GameState, me: int, evaluator, model=None, politics=None, belief=None,
                          max_counters: int = 3, rng=None, config=None) -> dict:
    """Accept / reject / counter analysis of the pending offer for the advisor (``cli.py --offer``).

    ``state`` is the decision state (``PHASE_TRADE_RESPONSE``, we are the responder).  Every option is valued
    at two horizons with the search's machinery: when the cards change hands (``v_trade``, the old static
    view) and after the rest of the proposer's turn (``v_turn``, ``SearchConfig.respond_lookahead``), and
    ranked by ``v_turn``.  Returns ``{"options": [...], "best": text, "lines": [...]}``; hidden hands use one
    determinization.  The state is not modified.
    """
    from .inference import is_fully_known, sample_states
    from .placement import road_targets
    from .search import SearchConfig, Searcher
    from .trading import should_accept
    rng = rng or random.Random(0)
    offer = state.pending_trade
    if offer is None or state.phase != PHASE_TRADE_RESPONSE or E.acting_player(state) != me:
        return {"options": [], "best": "", "lines": []}
    s0 = state if is_fully_known(state) else sample_states(state, me, 1, rng)[0]
    s0 = s0.copy()
    s0.allow_counters = True           # Colonist.io lets every responder counter
    cur = offer.proposer
    who = _pname(s0, cur)
    cfg = config or SearchConfig(depth=1, counters=1, respond_lookahead=1)
    searcher = Searcher(evaluator, cfg, model, belief, politics)
    searcher._rng = random.Random(rng.random())
    try:
        targets = {t["vertex"]: t["roads"] for t in road_targets(s0, me, max_roads=3, k=4)}
    except Exception:
        targets = {}
    legal = E.legal_actions(s0)
    options: List[Tuple[Action, float, str]] = []
    if (A.ACCEPT_TRADE,) in legal:
        options.append(((A.ACCEPT_TRADE,), 1.0, ""))
    options.append(((A.REJECT_TRADE,), 1.0, ""))
    ranked = rank_counters(s0, me, [a for a in legal if a[0] == A.COUNTER_TRADE], evaluator=evaluator,
                           model=model, belief=belief, politics=politics, aggr=cfg.counter_aggr)
    blocked = counter_blocked(s0, me, cur)
    for d in ranked[:max_counters]:
        options.append((d["action"], d["p_accept"], describe_edit(offer, d["action"][1], d["action"][2])))
    # Other responders' answers to the original offer (as the search simulates them), for "they take it instead".
    takers = []
    for j in range(s0.num_players):
        if j in (me, cur) or j in offer.responses:
            continue
        ok, _ = should_accept(s0, j, offer, politics=politics)
        if ok:
            p = (model.predict_accept(s0, j, offer.give, offer.get, proposer=cur, belief=belief, politics=politics)
                 if model is not None else None)
            takers.append((j, p))
    rows = []
    lookahead_on = cfg.respond_lookahead
    for a, p_acc, edit in options:
        cfg.respond_lookahead = 0
        try:
            mids = searcher._outcomes(s0, a, me)
        except E.IllegalActionError:
            continue
        finally:
            cfg.respond_lookahead = lookahead_on
        ends = [(p, searcher._finish_proposer_turn(m, me)) for p, m in mids]
        vals = [float(v) for v in evaluator.evaluate([m for _, m in mids] + [e for _, e in ends],
                                                     [me] * (2 * len(mids)))]
        v_trade = sum(p * v for (p, _), v in zip(mids, vals[:len(mids)]))
        v_turn = sum(p * v for (p, _), v in zip(ends, vals[len(mids):]))
        # The most likely outcome in which the proposer traded (with us for accept / our counter taken).
        traded = [(p, m, e) for (p, m), (_, e) in zip(mids, ends)
                  if m.players[cur].resources != s0.players[cur].resources]
        builds: List[Tuple[str, int]] = []
        p_traded = sum(p for p, _, _ in traded)
        if traded:
            _, m, e = max(traded, key=lambda t: t[0])
            builds = proposer_builds(m, e, cur)
        rows.append({"action": a, "text": A.describe(a, s0), "v_trade": v_trade, "v_turn": v_turn,
                     "p_accept": p_acc if a[0] == A.COUNTER_TRADE else None, "edit": edit,
                     "p_trade": p_traded, "builds": builds, "ends": ends})
    # Builds the proposer makes anyway (after our reject): a build only counts as "enabled" when it is missing there.
    rej = next((r for r in rows if r["action"][0] == A.REJECT_TRADE), None)
    anyway = set()
    if rej is not None:
        for p, e in rej["ends"]:
            anyway.update(proposer_builds(s0, e, cur))
    lines: List[str] = []
    for r in rows:
        k = r["action"][0]
        enabled = [b for b in r["builds"] if b not in anyway or b[0] == "win"]
        what = _build_text(s0, enabled, me, targets)
        if k == A.ACCEPT_TRADE:
            txt = (f"Accept: {r['v_turn']:.3f} after {who}'s turn ({r['v_trade']:.3f} when the cards change "
                   f"hands)")
            if what:
                txt += f"; accepting lets {who} build {what}" if not what.startswith("win") else \
                    f"; accepting lets {who} {what}"
            else:
                txt += f"; {who} builds nothing extra with it this turn"
        elif k == A.REJECT_TRADE:
            txt = f"Reject: {r['v_turn']:.3f} after {who}'s turn ({r['v_trade']:.3f} now)"
            if takers:
                j, p = max(takers, key=lambda t: t[1] or 0.0)
                txt += (f"; rejecting: {_pname(s0, j)} will likely accept instead"
                        + (f" (~{p:.0%})" if p is not None else ""))
            else:
                txt += "; nobody else is likely to take the deal"
        else:
            give, get = r["action"][1], r["action"][2]
            txt = (f"Counter: you give {A._counts_str(give)}, ask {A._counts_str(get)} ({r['edit']}) - "
                   f"P({who} takes it) {r['p_accept']:.0%}; {r['v_turn']:.3f} after {who}'s turn")
            if what:
                txt += f"; if taken {who} can build {what}" if not what.startswith("win") else \
                    f"; if taken {who} can {what}"
        r["line"] = txt
        lines.append(txt)
    if blocked:
        lines.append(f"No counter-offer: {blocked}.")
    elif not any(r["action"][0] == A.COUNTER_TRADE for r in rows):
        lines.append("No counter-offer worth making (none is affordable, safe and better for us than now).")
    best = max(rows, key=lambda r: r["v_turn"]) if rows else None
    best_text = ""
    if best is not None:
        k = best["action"][0]
        best_text = ("accept" if k == A.ACCEPT_TRADE else "reject" if k == A.REJECT_TRADE
                     else f"counter: give {A._counts_str(best['action'][1])} for {A._counts_str(best['action'][2])}")
        static_best = max(rows, key=lambda r: r["v_trade"])
        lines.insert(0, f"Best answer after {who}'s turn: {best_text}"
                     + ("" if static_best is best else
                        f" (valued when the cards change hands it would be "
                        f"{A.describe(static_best['action'], s0).lower()})"))
    out_rows = [{k: v for k, v in r.items() if k != "ends"} for r in rows]
    for r in out_rows:
        r["action"] = A.to_json(r["action"])
        r["builds"] = [list(b) for b in r["builds"]]
    return {"options": out_rows, "best": best_text, "lines": lines}
