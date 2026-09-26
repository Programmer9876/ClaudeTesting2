#!/usr/bin/env python3
"""Self-play mechanism checks of the trades area (docs/PRIORITY_PLAN.md step 3): a few paired games, no verdict.

    nice python3 scripts/acq_mechanism.py --games 8 --cand shapes:acq_shapes=1 --json runs/acq_mech_shapes.json
    nice python3 scripts/acq_mechanism.py --games 8 --cand calib:opponent_model.CALIB_RATE=1 ...
    nice python3 scripts/acq_mechanism.py --games 8 --cand floor:acq_floor=1 ...

Four-player games of the search bot (``--spec``), candidate seats (``C``) and default seats (``D``) in the rotating
patterns of ``tuning.SEAT_PATTERNS`` (CCDD, DDCC, CDCD, ...), seeds ``--seed + g``; a candidate is
``LABEL:NAME=VALUE,...`` as in scripts/decision_shadow.py (a dotted NAME is a registry tunable installed
ParamBot-style around the candidate seats' hooks, a plain NAME a bot-spec key).  Per side (candidate / default seats) it reports:

* the ``trades`` seat observer (``acquisition.TradeObserver``): proposals (all, mixed-give, dominated by the
  proposer's own bank / port rate), answers (accepts, rejects, accepts dominated by the responder's own rate),
  trades executed as proposer / partner and how many were dominated for that side, cancels;
* mixed-give (injected) proposals accepted by anybody and executed (acq.breadth (B));
* the Brier score of the proposer's own P(any accept) at proposal time (its opponent model, calibrated or not,
  with the seat's overrides installed; responders who cannot pay excluded, as ``Searcher._trade_outcomes``), the
  mean prediction and the realised accept rate (acq.calib);
* how many of the side's proposals and accepts fail the acq_floor premium rule (``Searcher._floor_check`` with the
  seat's own evaluator, ``acquisition.W_PREMIUM``), whether or not the side runs it;
* process-time seconds per game and per decision of the side's seats.

Games and seats are paired, so side differences are within-game.  One process; the queue's command rows read the
JSON (``--json``) with ``verdict_from`` (keys under ``summary.<side>``).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TimedBot:
    """Wraps a bot: process time of every ``decide`` (and the count of decisions with more than one legal action)."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "bot")
        self.cpu = 0.0
        self.decisions = 0

    def reset(self):
        self.inner.reset()

    def decide(self, state, legal, rng):
        t0 = time.process_time()
        try:
            return self.inner.decide(state, legal, rng)
        finally:
            self.cpu += time.process_time() - t0
            self.decisions += int(len(legal) > 1)

    def observe(self, state, action, player):
        self.inner.observe(state, action, player)

    def explain(self, state):
        return self.inner.explain(state)

    def __getattr__(self, item):
        if item in ("inner", "cpu", "decisions", "name"):
            raise AttributeError(item)
        return getattr(self.inner, item)


def _search_bot(b):
    """The SearchBot inside TimedBot / ParamBot wrappers."""
    while hasattr(b, "inner"):
        b = b.inner
    return b


class Monitor:
    """``on_action`` hook: trade counters, Brier of the proposer's own P(any accept), mixed-offer outcomes and the
    premium rule's verdict on every proposal and accept (see the module docstring)."""

    def __init__(self, bots, pattern: str, floor_check: bool = True):
        from catanbot.acquisition import TradeObserver
        self.bots = bots
        self.pattern = pattern
        self.obs = TradeObserver(len(bots))
        self.floor = floor_check
        self.pending: Optional[Dict[str, Any]] = None
        self.brier: List[tuple] = []            # (seat, p_any, accepted)
        self.mixed: List[tuple] = []            # (seat, accepted, executed)
        self.floor_fail = {"proposals": [0] * len(bots), "accepts": [0] * len(bots), "checked_p": [0] * len(bots),
                           "checked_a": [0] * len(bots)}

    def _overrides(self, i: int):
        from catanbot import tuning
        b = self.bots[i].inner if hasattr(self.bots[i], "inner") else self.bots[i]
        ov = getattr(b, "overrides", {}) or {}
        return tuning.apply(ov), tuning.apply_to_bot(_search_bot(b), ov)

    def _restore(self, tok):
        from catanbot import tuning
        tuning.restore_bot(tok[1])
        tuning.restore(tok[0])

    def _searcher(self, i: int):
        from catanbot.search import Searcher
        sb = _search_bot(self.bots[i])
        return Searcher(sb.evaluator, sb.config, sb.model if sb.config.use_opponent_model else None, sb.belief,
                        sb.politics)

    def _close(self) -> None:
        pd = self.pending
        if pd is None:
            return
        if pd["p"] is not None:
            self.brier.append((pd["seat"], pd["p"], pd["accepted"]))
        if pd["mixed"]:
            self.mixed.append((pd["seat"], pd["accepted"], pd["executed"]))
        self.pending = None

    def on_action(self, state, action, player) -> None:
        from catanbot import actions as A
        self.obs.on_action(state, action, player)
        k = action[0]
        if k in (A.ACCEPT_TRADE, A.REJECT_TRADE) and self.pending is not None and state.pending_trade is not None \
                and state.pending_trade.origin is None:
            if k == A.ACCEPT_TRADE:
                self.pending["accepted"] = True
            if k == A.ACCEPT_TRADE and self.floor:
                self._floor_accept(state, player)
            return
        if k == A.EXECUTE_TRADE and self.pending is not None:
            self.pending["executed"] = True
            self._close()
            return
        self._close()
        if k != A.PROPOSE_TRADE:
            return
        give, get = action[1], action[2]
        sb = _search_bot(self.bots[player])
        p_any = None
        if sb.model is not None:
            tok = self._overrides(player)
            try:
                probs = []
                for j in range(state.num_players):
                    if j == player:
                        continue
                    q = state.players[j]
                    if q.hand_known and any(q.resources[r] < get[r] for r in range(5)):
                        continue
                    probs.append(sb.model.predict_accept(state, j, give, get, proposer=player, belief=sb.belief,
                                                         politics=sb.politics))
                if probs:
                    p = 1.0
                    for x in probs:
                        p *= 1.0 - x
                    p_any = 1.0 - p
            finally:
                self._restore(tok)
        self.pending = {"seat": player, "p": p_any, "accepted": False, "executed": False,
                        "mixed": sum(1 for x in give if x) >= 2}
        if self.floor:
            tok = self._overrides(player)
            try:
                sr = self._searcher(player)
                ok = sr._floor_proposals(state, [tuple(action)], player, [])
            finally:
                self._restore(tok)
            self.floor_fail["checked_p"][player] += 1
            self.floor_fail["proposals"][player] += int(tuple(action) not in ok)

    def _floor_accept(self, state, j) -> None:
        from catanbot import actions as A
        po = state.pending_trade
        tok = self._overrides(j)
        try:
            sr = self._searcher(j)
            res = sr._floor_check(state, j, [((A.ACCEPT_TRADE,), po.get, po.give, po.proposer, state.current != j)])
        finally:
            self._restore(tok)
        self.floor_fail["checked_a"][j] += 1
        self.floor_fail["accepts"][j] += int(not res[(A.ACCEPT_TRADE,)][0])

    def finish(self) -> Dict[str, Any]:
        self._close()
        return {"trades": self.obs.result(), "brier": self.brier, "mixed": self.mixed, "floor": self.floor_fail}


def play(spec: str, cand, games: int, seed: int, max_turns: int, floor_check: bool, log=sys.stdout):
    from catanbot import tuning
    from catanbot.agents.param_bot import ParamBot
    from catanbot.selfplay import make_bot, play_game
    cspec = spec + ("," if ":" in spec else ":") + ",".join(f"{k}={v}" for k, v in cand.spec_keys.items()) \
        if cand.spec_keys else spec
    out = []
    for g in range(games):
        pattern = tuning.seat_pattern(4, g)
        gseed = seed + g
        bots = [TimedBot(ParamBot(make_bot(cspec if side == "C" else spec), cand.overrides if side == "C" else {},
                                  label="cand" if side == "C" else "default")) for side in pattern]
        mon = Monitor(bots, pattern, floor_check)
        t0 = time.process_time()
        res = play_game(bots, rng=random.Random(gseed), seed=gseed, max_turns=max_turns, on_action=mon.on_action)
        rec = {"seed": gseed, "pattern": pattern, "winner": res.winner, "vps": list(res.vps), "turns": res.turns,
               "cpu": [b.cpu for b in bots], "decisions": [b.decisions for b in bots],
               "game_cpu": time.process_time() - t0, **mon.finish()}
        out.append(rec)
        print(f"game {g + 1}/{games} seed {gseed} {pattern}: winner {res.winner}, {res.turns} turns, "
              f"{rec['game_cpu']:.0f} CPU-s", file=log, flush=True)
    return out


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from catanbot.acquisition import TradeObserver
    sides: Dict[str, Dict[str, Any]] = {}
    for side in ("C", "D"):
        acc = {k: 0 for k in TradeObserver.KEYS}
        br: List[tuple] = []
        mixed = [0, 0, 0]
        fl = {"proposals": 0, "accepts": 0, "checked_p": 0, "checked_a": 0}
        cpu = dec = seats = 0.0
        wins = 0
        for r in records:
            idx = [i for i, s in enumerate(r["pattern"]) if s == side]
            seats += len(idx)
            wins += int(r["winner"] in idx)
            for k in TradeObserver.KEYS:
                acc[k] += sum(r["trades"][k][i] for i in idx)
            br.extend(b for b in r["brier"] if b[0] in idx)
            for s, a, e in r["mixed"]:
                if s in idx:
                    mixed[0] += 1
                    mixed[1] += int(a)
                    mixed[2] += int(e)
            for k in fl:
                fl[k] += sum(r["floor"][k][i] for i in idx)
            cpu += sum(r["cpu"][i] for i in idx)
            dec += sum(r["decisions"][i] for i in idx)
        g = len(records)
        n_b = len(br)
        sides["cand" if side == "C" else "default"] = {
            "seat_games": seats, "wins": wins,
            "per_seat_game": {k: (v / seats if seats else None) for k, v in acc.items()},
            "totals": acc,
            "mixed_proposals": mixed[0], "mixed_accepted": mixed[1], "mixed_executed": mixed[2],
            "brier": (sum((p - float(a)) ** 2 for _, p, a in br) / n_b) if n_b else None,
            "pred_mean": (sum(p for _, p, _a in br) / n_b) if n_b else None,
            "accept_rate": (sum(1 for *_x, a in br if a) / n_b) if n_b else None, "offers_scored": n_b,
            "floor_fail": fl,
            "floor_fail_share_proposals": (fl["proposals"] / fl["checked_p"]) if fl["checked_p"] else None,
            "floor_fail_share_accepts": (fl["accepts"] / fl["checked_a"]) if fl["checked_a"] else None,
            "cpu_s_per_seat_game": cpu / seats if seats else None,
            "ms_per_decision": 1000.0 * cpu / dec if dec else None,
            "games": g}
    c, d = sides["cand"], sides["default"]
    sides["cpu_ratio"] = (c["ms_per_decision"] / d["ms_per_decision"]) if c["ms_per_decision"] and \
        d["ms_per_decision"] else None
    return sides


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7700)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--cand", required=True, metavar="LABEL:NAME=VALUE,...")
    ap.add_argument("--no-floor-check", action="store_true", help="skip the premium rule's verdict on every trade")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    import importlib.util
    ds = sys.modules.get("_ds_for_mech")
    if ds is None:
        spec_ds = importlib.util.spec_from_file_location("_ds_for_mech",
                                                         os.path.join(ROOT, "scripts", "decision_shadow.py"))
        ds = importlib.util.module_from_spec(spec_ds)
        sys.modules["_ds_for_mech"] = ds
        spec_ds.loader.exec_module(ds)
    from catanbot import tuning
    spec = args.spec or tuning.DEFAULT_SEARCH_SPEC
    cand = ds.parse_candidate(args.cand)
    t0 = time.process_time()
    recs = play(spec, cand, args.games, args.seed, args.max_turns, not args.no_floor_check)
    summ = summarize(recs)
    print(f"\n{cand.label}: {args.games} games, {time.process_time() - t0:.0f} CPU-s; evaluator "
          f"{tuning.evaluator_mode()}")
    for side in ("cand", "default"):
        s = summ[side]
        ps = s["per_seat_game"]
        print(f"  {side:8} wins {s['wins']}/{s['games']}; per seat-game: proposals {ps['proposals']:.1f} (mixed "
              f"{ps['proposals_mixed']:.2f}, dominated {ps['proposals_dominated']:.2f}), accepts {ps['accepts']:.2f} "
              f"(dominated {ps['accepts_dominated']:.2f}), executed {ps['executed']:.2f}, partnered "
              f"{ps['partnered']:.2f}; mixed accepted {s['mixed_accepted']}/{s['mixed_proposals']} (executed "
              f"{s['mixed_executed']}); Brier {s['brier'] if s['brier'] is None else round(s['brier'], 3)} "
              f"(pred {s['pred_mean'] if s['pred_mean'] is None else round(s['pred_mean'], 3)} vs realised "
              f"{s['accept_rate'] if s['accept_rate'] is None else round(s['accept_rate'], 3)}, "
              f"{s['offers_scored']} offers); premium-rule fails: proposals {s['floor_fail']['proposals']}/"
              f"{s['floor_fail']['checked_p']}, accepts {s['floor_fail']['accepts']}/{s['floor_fail']['checked_a']}; "
              f"{s['cpu_s_per_seat_game']:.1f} CPU-s per seat-game, {s['ms_per_decision']:.1f} ms per decision")
    print(f"  CPU ratio (ms per decision, cand / default): {summ['cpu_ratio']:.2f}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json + ".tmp", "w") as fh:
            json.dump({"spec": spec, "cand": {"label": cand.label, "overrides": cand.overrides,
                                              "spec_keys": cand.spec_keys},
                       "games": args.games, "seed": args.seed, "summary": summ, "records": recs}, fh)
        os.replace(args.json + ".tmp", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
