#!/usr/bin/env python3
"""Gate (A) of devbelief.hazard (docs/PRIORITY_PLAN.md step 6): calibration of the held-age posterior over
opponents' hidden development cards, on archived catanatron action logs, with zero new games.

    /home/user/venv_cat33/bin/python scripts/dev_belief_calibrate.py --logs 'proof/T*/logs/*.jsonl.gz' \\
        --json gate.json            # the interpreter must match the logs' engine (api 3.3 / 3.2)

Every game is rebuilt from its log and followed by the REAL counted tracker
(:class:`catanbot.bench.public_info.PublicInfoTracker`, our logged seat) and its
:class:`catanbot.devbelief.DevAgeModel`, so the implementation and the calibration are checked in one run.  At
every own-turn end of an opponent holding at least one hidden card, the exact posterior under the uniform deal
(``uniform``: no aging, the same win caps) and under the bot preset (``devbelief.H`` / ``EPS`` /
``H_KNIGHT_CTX``, ``--h`` / ``--eps`` / ``--h-knight-ctx``) predicts the holder's hidden VP cards; the truth is
read from the rebuilt state.

Metrics per holder class (``catanbot`` or the opponent preset) and age bucket of its oldest held card (own turns
since the purchase, the purchase turn included: 1, 2, 3-5, 6-9, 10+): hidden-VP mean squared error and the
log-loss of P(at least 1 VP); readouts: the log-loss of P(holds a knight) overall and in the knight-context
stratum (the turn started blocked as main robber victim or one knight from Largest Army).

PASS (the design's gate A) requires all of: hidden-VP MSE at least 30% below uniform in every class with at
least ``--min-n`` observations; log-loss no worse than uniform in every class x bucket cell with at least
``--min-n`` observations; 0 devbelief errors; fallbacks (no usable posterior) under 0.1%.  One process, no
workers; resumable only by re-running (about 10 CPU-min for T1-T11 niced).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 10 ** 9, "10+"))
MODELS = ("uniform", "hazard")


def bucket_of(age: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= age <= hi:
            return name
    return "1"


def _clip(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, p))


def logloss(p: float, y: bool) -> float:
    return -math.log(_clip(p) if y else 1.0 - _clip(p))


def holder_class(doc: Dict[str, Any], j: int) -> str:
    players = doc.get("players") or []
    if isinstance(players, str):
        import ast
        players = ast.literal_eval(players)
    if j < len(players) and isinstance(players[j], dict):
        p = players[j]
        return "catanbot" if p.get("kind") == "catanbot" else str(p.get("preset") or p.get("kind") or "opponent")
    return "unknown"


def iter_docs(paths: Sequence[str]):
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as fh:
            for line in fh:
                if line.strip():
                    yield p, json.loads(line)


def observe_game(doc: Dict[str, Any], sink: List[Dict[str, Any]]) -> Dict[str, int]:
    """Replay one logged game; append one observation per opponent own-turn end with hidden cards."""
    from catanatron.models.enums import ActionType
    from catanbot import devbelief as DB
    from catanbot.bench import catanatron_adapter as ad
    from catanbot.bench.public_info import PublicInfoTracker, snapshot
    seats = doc["our_seats"]
    if isinstance(seats, str):
        seats = json.loads(seats)
    me_seat = int(seats[0])
    game = ad.rebuild_game(doc)
    st = game.state
    # Colonist information (docs/PRIORITY_PLAN.md 'Rule for Colonist-information rows'): discards are public
    tr = PublicInfoTracker(st.colors[me_seat], vps_to_win=int(doc.get("vps_to_win", 10)), discards_public=True)
    tr.start(st)
    fallbacks = 0
    for item in doc["actions"]:
        ad.replay_log_action(game, item, check=False)
        st = game.state
        try:
            tr.follow(st)
        except Exception:  # noqa: BLE001  (as the counted adapter: count it and resync)
            tr.resync(st)
        a = ad.log_action(ad.action_log(st)[-1])
        if a.action_type != ActionType.END_TURN:
            continue
        j = st.color_to_index[a.color]
        if j == tr.me or j not in tr.unknown_dev_holders() or tr.dev_count[j] <= 0:
            continue
        snap = snapshot(st, tr.me)
        public = snap.vp
        ps = st.player_state
        truth_vp = int(ps[f"P{j}_VICTORY_POINT_IN_HAND"])
        truth_k = int(ps[f"P{j}_KNIGHT_IN_HAND"]) >= 1
        dm = tr.dev_age
        row = {"cls": holder_class(doc, j), "age": len(dm.turns[j]) - min(dm.recs[j]) if dm.recs[j] else 0,
               "n": tr.dev_count[j], "vp": truth_vp, "k": truth_k,
               "kctx": bool(dm.turns[j] and (dm.turns[j][-1][2] or dm.turns[j][-1][1] >= DB.KCTX_MIN_PIPS))}
        for name, pr in (("uniform", DB.UNIFORM), ("hazard", DB.BOT)):
            post = tr.dev_posterior(public, pr)
            if post is None or j not in post:
                fallbacks += 1
                row = None
                break
            d = post[j]
            row[name] = {"e": d["e_vp"], "p1": 1.0 - d["p_vp"][0], "pk": d["p_type"][ad.B.DEV_KNIGHT]}
        if row is not None:
            sink.append(row)
    return {"errors": int(tr.dev_age.stats["errors"]), "fallbacks": fallbacks}


def summarize(rows: Sequence[Dict[str, Any]], min_n: int) -> Dict[str, Any]:
    def cell(rs):
        out = {"n": len(rs)}
        for m in MODELS:
            out[m] = {"mse": sum((r[m]["e"] - r["vp"]) ** 2 for r in rs) / len(rs),
                      "logloss": sum(logloss(r[m]["p1"], r["vp"] >= 1) for r in rs) / len(rs),
                      "knight_logloss": sum(logloss(r[m]["pk"], r["k"]) for r in rs) / len(rs)}
        return out
    classes: Dict[str, Any] = {}
    fails: List[str] = []
    for cls in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == cls]
        c = cell(rs)
        c["buckets"] = {}
        if len(rs) >= min_n and not c["hazard"]["mse"] <= 0.7 * c["uniform"]["mse"]:
            fails.append(f"{cls}: hidden-VP MSE {c['hazard']['mse']:.3f} not 30% below uniform {c['uniform']['mse']:.3f}")
        for _lo, _hi, b in BUCKETS:
            rb = [r for r in rs if bucket_of(r["age"]) == b]
            if not rb:
                continue
            cb = cell(rb)
            c["buckets"][b] = cb
            if len(rb) >= min_n and cb["hazard"]["logloss"] > cb["uniform"]["logloss"] + 1e-12:
                fails.append(f"{cls} age {b}: log-loss {cb['hazard']['logloss']:.3f} > uniform "
                             f"{cb['uniform']['logloss']:.3f}")
        kc = [r for r in rs if r["kctx"]]
        if kc:
            c["knight_context"] = cell(kc)
        classes[cls] = c
    return {"classes": classes, "fails": fails}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", required=True, help="glob of catanbot-actionlog/1 files (proof/T1/logs/*.jsonl.gz)")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--h", type=float, default=None, help="devbelief.H (default: the module value)")
    ap.add_argument("--eps", type=float, default=None, help="devbelief.EPS")
    ap.add_argument("--h-knight-ctx", type=float, default=None, help="devbelief.H_KNIGHT_CTX")
    ap.add_argument("--min-n", type=int, default=30, help="observations a gate cell needs to count")
    ap.add_argument("--json", help="write the result here (key 'pass' / 'verdict')")
    args = ap.parse_args(argv)
    from catanbot import devbelief as DB
    from catanbot.bench import catanatron_adapter as ad
    for attr, v in (("H", args.h), ("EPS", args.eps), ("H_KNIGHT_CTX", args.h_knight_ctx)):
        if v is not None:
            setattr(DB, attr, float(v))
    paths = sorted(glob.glob(args.logs))
    if not paths:
        raise SystemExit(f"error: no file matches {args.logs!r}")
    api = "3.3" if ad.API_33 else "3.2"
    t0 = time.process_time()
    rows: List[Dict[str, Any]] = []
    errors = fallbacks = games = 0
    for p, doc in iter_docs(paths):
        if doc.get("api") != api:
            raise SystemExit(f"error: {p} was written by catanatron {doc.get('catanatron')} ({doc.get('api')} API); "
                             "run with the interpreter of that engine")
        res = observe_game(doc, rows)
        errors += res["errors"]
        fallbacks += res["fallbacks"]
        games += 1
        if args.max_games is not None and games >= args.max_games:
            break
    out = summarize(rows, args.min_n) if rows else {"classes": {}, "fails": ["no observation"]}
    n_obs = len(rows) + fallbacks
    if errors:
        out["fails"].append(f"{errors} devbelief errors")
    if n_obs and fallbacks / n_obs >= 0.001:
        out["fails"].append(f"fallbacks {fallbacks}/{n_obs} >= 0.1%")
    out.update({"games": games, "observations": len(rows), "errors": errors, "fallbacks": fallbacks,
                "params": {"H": DB.H, "EPS": DB.EPS, "H_KNIGHT_CTX": DB.H_KNIGHT_CTX}, "min_n": args.min_n,
                "cpu_s": round(time.process_time() - t0, 1)})
    out["pass"] = not out["fails"]
    out["verdict"] = "PASS" if out["pass"] else "FAIL"
    for cls, c in out["classes"].items():
        print(f"{cls:>10} n={c['n']:6d}  MSE uniform {c['uniform']['mse']:.3f} hazard {c['hazard']['mse']:.3f}  "
              f"log-loss {c['uniform']['logloss']:.3f} -> {c['hazard']['logloss']:.3f}  "
              f"knight {c['uniform']['knight_logloss']:.3f} -> {c['hazard']['knight_logloss']:.3f}")
    print(f"{out['verdict']}: {games} games, {len(rows)} observations, {errors} errors, {fallbacks} fallbacks, "
          f"{out['cpu_s']} CPU s" + ("".join("\n  " + f for f in out["fails"])))
    if args.json:
        tmp = args.json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
