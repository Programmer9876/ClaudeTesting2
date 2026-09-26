#!/usr/bin/env python3
"""acq.progress Stage 0 audit (docs/PRIORITY_PLAN.md step 3; zero games): trade-legal proof positions, real code.

    /home/user/venv_cat33/bin/python scripts/acq_audit.py --logs 'proof/T1/logs/*.gz' --positions 300 \\
        --json runs/acq_audit.json

The positions are our seat's main-phase decisions with a legal bank trade in archived proof games (the logged game
is replayed; the bot's answer is ignored), taken in log order until ``--positions`` are collected.  At each one the
default search (``--spec``) and every candidate re-search the root from the same ``random.Random(k)``
(scripts/decision_shadow.py's ShadowBot); the A/A candidate must change nothing.  Candidates (default): ``acq1``
(``acq=1``, conversions only), ``acq2`` (``acq=2``, the tested arm), ``conv`` (``conv=1``) and ``conv_acq2``
(``conv=1,acq=2``: the two hub providers together).

Reported per candidate: the change rate; flips by hand size (7 or fewer / above 7) and by kind; END_TURN flips
above 7 cards; for flips to END_TURN at 7 or fewer cards the share whose missing cards (after the conversions) are
rolled before our next build with P >= 0.5 (the design's "waitable": the card the default's bank trade buys is
rolled within the ``n`` rolls up to our next turn; also reported: P of rolling every card the best target still
misses after the conversions, ``acquisition.best_target_left``, exact over the same rolls); the
hub's correction calls, corrected leaves and cache misses per decision; process-time ms per decision (ratio to
the default).  Pre-registered pass (mode 2): 0 END_TURN flips above 7; >= 50 % of the END_TURN flips at <= 7 with
P >= 0.5; change rate >= 3 %; ms ratio <= 1.3.  Exit code 0 = PASS, 3 = FAIL or A/A broken.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_shadow():
    name = "_decision_shadow_for_acq"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "decision_shadow.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DS = _load_shadow()
DEFAULT_CANDS = ("acq1:acq=1", "acq2:acq=2", "conv:conv=1", "conv_acq2:conv=1,acq=2")
PASS_RULES = {"end_turn_above7": 0, "p_share_min": 0.5, "change_min": 0.03, "ms_ratio_max": 1.3}


class AuditBot(DS.ShadowBot):
    """ShadowBot restricted to trade-legal main-phase decisions (a legal bank trade), with a position cap and the
    audit's extra fields per row (hand size, hub counters, P(roll the missing cards) for END_TURN flips)."""

    cap = 300

    def decide(self, state, legal, rng):
        from catanbot import actions as A
        from catanbot.state import PHASE_MAIN
        if (len(legal) > 1 and state.phase == PHASE_MAIN and len(self.rows) < self.cap
                and any(a[0] == A.BANK_TRADE for a in legal)):
            self._shadow(state, legal, DS.classes_of(state, legal))
        return self.inner.decide(state, legal, rng)

    def _search_stats(self, state, me, cfg, k, overrides):
        from catanbot import tuning
        from catanbot.search import Searcher
        token = tuning.apply(overrides) if overrides else None
        try:
            sr = Searcher(self.inner.evaluator, cfg, self.inner.model, self.inner.belief, self.inner.politics)
            t0 = time.process_time()
            res = sr.search(state, me, random.Random(k))
            dt = time.process_time() - t0
        finally:
            if token is not None:
                tuning.restore(token)
        st: Dict[str, Any] = {}
        if sr._corr is not None:
            st["hub"] = {k2: v for k2, v in sr._corr.stats.items() if k2 != "seconds"}
            for p in sr._corr.providers:
                st[type(p).__name__] = dict(getattr(p, "stats", {}))
        return (list(res[0].action) if res else None), dt, st

    def _shadow(self, state, legal, cls) -> None:
        from catanbot import acquisition as Q
        from catanbot import engine as E
        inner = self.inner
        if inner.model is not None:
            inner.model.attach(state)
        inner._ensure_politics(state)
        me = E.acting_player(state)
        k = self.seed * 100003 + self.k
        self.k += 1
        a0, t0, _ = self._search_stats(state, me, inner.config, k, None)
        n = state.players[me].total_resources
        row = {"g": self.tag, "k": k, "turn": state.turn, "player": me, "cls": list(cls), "n": n, "def": a0,
               "ms_def": round(1000.0 * t0, 3), "cand": {}}
        for c in self.candidates:
            a1, t1, st = self._search_stats(state, me, self._config(c), k, dict(c.overrides))
            got = {"a": a1, "ms": round(1000.0 * t1, 3), "stats": st}
            if a1 != a0 and a1 and a1[0] == "end_turn" and n <= 7:
                name, left, p = Q.best_target_left(state, me, mode=2)
                got["end_turn_p"] = {"target": name, "left": left, "p_complete": round(p, 4)}
                if a0 and a0[0] == "bank_trade":      # the design's "waitable": roll the bought card before our turn
                    ys = Q.roll_yields(state, me)
                    got["end_turn_p"]["p"] = round(Q.roll_tails(ys, Q.horizon_k(state, me, me))[a0[2]][0], 4)
                else:
                    got["end_turn_p"]["p"] = round(p, 4)
            row["cand"][c.label] = got
        self.rows.append(row)


def run(spec: str, candidates, paths: Sequence[str], positions: int, log=sys.stdout):
    from catanatron.models.player import Color
    from catanbot.bench import catanatron_adapter as ad
    from catanbot.selfplay import make_bot
    api = "3.3" if ad.API_33 else "3.2"
    rows: List[Dict[str, Any]] = []
    games = []
    AuditBot.cap = positions
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc.get("api") != api:
                    raise SystemExit(f"error: {p} was written by catanatron {doc.get('catanatron')}; run with the "
                                     f"interpreter of that engine")
                seats = json.loads(doc["our_seats"]) if isinstance(doc["our_seats"], str) else doc["our_seats"]
                players = doc["players"]
                if isinstance(players, str):
                    import ast
                    players = ast.literal_eval(players)
                seat = int(seats[0])
                color = Color(doc["colors"][seat])
                bot_seed = int(players[seat].get("bot_seed", 0))
                game = ad.rebuild_game(doc)
                bot = AuditBot(make_bot(spec), candidates, ["main"], bot_seed, spec, rows, tag=f"pf{doc.get('game')}")
                me = ad.CatanbotPlayer(color, spec=spec, bot=bot, seed=bot_seed, suppress_trades=True)
                errors = 0
                for item in doc["actions"]:
                    if len(rows) >= positions:
                        break
                    if item[0] == color.value:
                        try:
                            me.decide(game, ad.playable_actions_of(game))
                        except Exception:  # noqa: BLE001  (the logged action is replayed anyway)
                            errors += 1
                    ad.replay_log_action(game, item, check=False)
                games.append({"game": doc.get("game"), "seat": seat, "errors": errors})
                print(f"proof game {doc.get('game')}: {len(rows)} positions", file=log, flush=True)
                if len(rows) >= positions:
                    return rows, games
    return rows, games


def summarize(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lab in labels:
        n = ch = 0
        by_hand = {"le7": [0, 0], "gt7": [0, 0]}
        kinds: Counter = Counter()
        end7 = []
        end7c = []
        end_above7 = 0
        ms_d, ms_c = [], []
        calls = corrected = tail_miss = target_miss = evals = gated = nonzero = 0
        for r in rows:
            got = r["cand"].get(lab)
            if got is None:
                continue
            n += 1
            flip = got["a"] != r["def"]
            ch += flip
            b = by_hand["le7" if r["n"] <= 7 else "gt7"]
            b[0] += 1
            b[1] += int(flip)
            if flip:
                kinds[f"{r['def'][0] if r['def'] else '-'} -> {got['a'][0] if got['a'] else '-'}"] += 1
                if got["a"] and got["a"][0] == "end_turn":
                    if r["n"] > 7:
                        end_above7 += 1
                    elif "end_turn_p" in got:
                        end7.append(got["end_turn_p"]["p"])
                        end7c.append(got["end_turn_p"]["p_complete"])
            ms_d.append(r["ms_def"])
            ms_c.append(got["ms"])
            hub = got["stats"].get("hub") or {}
            calls += hub.get("evals", 0)
            corrected += hub.get("corrected", 0)
            ac = got["stats"].get("AcqContext") or {}
            evals += ac.get("evals", 0)
            nonzero += ac.get("nonzero", 0)
            gated += ac.get("gated", 0)
            tail_miss += ac.get("tail_misses", 0)
            target_miss += ac.get("target_misses", 0)
        md = sum(ms_d) / len(ms_d) if ms_d else float("nan")
        mc = sum(ms_c) / len(ms_c) if ms_c else float("nan")
        per = (lambda x: round(x / n, 2)) if n else (lambda x: None)
        out[lab] = {"n": n, "changed": ch, "share": ch / n if n else None,
                    "by_hand": {k: {"n": v[0], "changed": v[1], "share": v[1] / v[0] if v[0] else None}
                                for k, v in by_hand.items()},
                    "kinds": dict(kinds.most_common()), "end_turn_above7": end_above7,
                    "end_turn_le7": len(end7), "end_turn_le7_p_ge_half": sum(1 for p in end7 if p >= 0.5),
                    "end_turn_le7_p_mean": (sum(end7) / len(end7)) if end7 else None,
                    "end_turn_le7_complete_p_ge_half": sum(1 for p in end7c if p >= 0.5),
                    "ms_def": md, "ms": mc, "ms_ratio": mc / md if md == md and md > 0 else None,
                    "per_decision": {"hub_evals": per(calls), "hub_corrected": per(corrected),
                                     "acq_evals": per(evals), "acq_nonzero": per(nonzero), "acq_gated": per(gated),
                                     "tail_misses": per(tail_miss), "target_misses": per(target_miss)}}
    return out


def verdict(s: Dict[str, Any]) -> Dict[str, Any]:
    """The pre-registered Stage 0 rule on one candidate's summary (mode 2)."""
    le7 = s["end_turn_le7"]
    share_p = (s["end_turn_le7_p_ge_half"] / le7) if le7 else None
    checks = {
        "end_turn_above7 == 0": s["end_turn_above7"] == PASS_RULES["end_turn_above7"],
        "END_TURN flips <= 7 with P >= 0.5: >= 50%": (share_p is None) or share_p >= PASS_RULES["p_share_min"],
        "change rate >= 3%": (s["share"] or 0.0) >= PASS_RULES["change_min"],
        "ms ratio <= 1.3": (s["ms_ratio"] or 99.0) <= PASS_RULES["ms_ratio_max"],
    }
    return {"pass": all(checks.values()), "checks": checks, "end_turn_le7_share_p_ge_half": share_p}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=os.path.join(ROOT, "proof", "T1", "logs", "*.gz"))
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--spec", default=None, help="the default bot spec (default: the shipped search bot)")
    ap.add_argument("--cand", action="append", default=None, metavar="LABEL:NAME=VALUE,...")
    ap.add_argument("--rule", default="acq2", metavar="LABEL",
                    help="the candidate the pre-registered rule decides (exit code); default acq2")
    ap.add_argument("--json", help="write the result here (rows included)")
    args = ap.parse_args(argv)
    from catanbot import tuning
    spec = args.spec or tuning.DEFAULT_SEARCH_SPEC
    cands = [DS.Candidate("aa", {}, {})] + [DS.parse_candidate(t) for t in (args.cand or DEFAULT_CANDS)]
    paths = sorted(glob.glob(args.logs))
    if not paths:
        raise SystemExit(f"error: no files match {args.logs}")
    cpu0, t0 = time.process_time(), time.time()
    rows, games = run(spec, cands, paths, args.positions)
    summ = summarize(rows, [c.label for c in cands])
    aa_ok = summ["aa"]["changed"] == 0
    print(f"\n{len(rows)} trade-legal positions in {len(games)} proof game(s); evaluator {tuning.evaluator_mode()}; "
          f"{time.process_time() - cpu0:.0f} CPU-s, {time.time() - t0:.0f} s wall")
    print(f"A/A: {summ['aa']['changed']} changed -> {'OK' if aa_ok else 'FAILED'}")
    le = sum(1 for r in rows if r["n"] <= 7)
    print(f"hand <= 7: {le}, > 7: {len(rows) - le}")
    for c in cands[1:]:
        s = summ[c.label]
        bh = s["by_hand"]
        print(f"  {c.label:10} changed {s['changed']}/{s['n']} ({(s['share'] or 0):.1%}); <=7 {bh['le7']['changed']}/"
              f"{bh['le7']['n']}, >7 {bh['gt7']['changed']}/{bh['gt7']['n']}; END_TURN >7: {s['end_turn_above7']}; "
              f"END_TURN <=7: {s['end_turn_le7']} (waitable P>=0.5: {s['end_turn_le7_p_ge_half']}, all missing P>=0.5: "
              f"{s['end_turn_le7_complete_p_ge_half']}); ms ratio "
              f"{(s['ms_ratio'] or float('nan')):.2f}; per decision {s['per_decision']}")
        for kname, cnt in list(s["kinds"].items())[:6]:
            print(f"      {cnt:4d}  {kname}")
    ver = verdict(summ[args.rule]) if args.rule in summ else None
    for c in cands[1:]:
        if c.label.startswith("acq"):
            v = verdict(summ[c.label])
            print(f"Stage 0 rule ({c.label}{', decides' if c.label == args.rule else ''}): "
                  f"{'PASS' if v['pass'] else 'FAIL'} {v['checks']}")
    if "conv" in summ and "conv_acq2" in summ and "acq2" in summ:
        both = sum(1 for r in rows if r["cand"]["conv"]["a"] != r["def"] and r["cand"]["acq2"]["a"] != r["def"])
        pair = sum(1 for r in rows if r["cand"]["conv_acq2"]["a"] != r["def"])
        agree = sum(1 for r in rows if r["cand"]["conv_acq2"]["a"] == r["cand"]["acq2"]["a"]
                    or r["cand"]["conv_acq2"]["a"] == r["cand"]["conv"]["a"])
        print(f"composition: conv & acq2 both change {both}; conv_acq2 changes {pair}; conv_acq2 equals acq2's or "
              f"conv's action in {agree}/{len(rows)}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json + ".tmp", "w") as fh:
            json.dump({"spec": spec, "evaluator": tuning.evaluator_mode(), "positions": len(rows), "games": games,
                       "aa_ok": aa_ok, "candidates": summ, "stage0": ver, "rule": args.rule, "rows": rows,
                       "cpu_s": time.process_time() - cpu0}, fh)
        os.replace(args.json + ".tmp", args.json)
    return 0 if aa_ok and (ver is None or ver["pass"]) else 3


if __name__ == "__main__":
    sys.exit(main())
