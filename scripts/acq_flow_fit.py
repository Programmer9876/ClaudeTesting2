#!/usr/bin/env python3
"""acq.flow Stage 0 (docs/PRIORITY_PLAN.md step 3; zero games): fit the trade-flow table on proof logs, validate.

    python3 scripts/acq_flow_fit.py --fit 'proof/T1/logs/*.gz' --validate 'proof/T2/logs/*.gz' --json out.json

Replays our seat's buildings and bank trades from archived proof games (catanbot-actionlog/1; pure Python, no
catanatron needed; the board and node topology come from scripts/mechanics.py) and fits the two parts of
``acquisition.port_flow_value``:

* ``R(t)`` (``FLOW_T`` / ``FLOW_R``): bank trades (any ratio) our bot still makes after ``t`` post-setup player turns,
  averaged over the games still running at ``t`` (so a live game's port is valued for the rest of *that* game);
  the unconditional 4:1-only table is printed next to the design's (5.0 / 4.3 / 3.0 / 1.6 / 0.55 at 0 / 20 / 40 /
  60 / 80);
* ``w_r`` (``FLOW_W``): a multinomial logit of the resource a bank trade gives, on our production share of ``r``
  (robber-free pips) and on "``r`` is wheat / ore while a city can still be built", with per-resource intercepts
  (wood = 0), maximum likelihood (Newton steps, a small ridge).

Validation on the other log set: (i) the give split given the trades (per game, predicted vs realised gives per
resource: mean absolute error, against the flat give shares and production-share-only baselines); (ii) the whole
function at t = 0 and t = 40: ``R(t) w_r(features at t)`` vs the gives per resource still made after t, per game
alive at t (mean absolute error per resource per game); (iii) ``R(t)`` of the validation set vs the fitted table.
Prints the constants to paste into catanbot/acquisition.py (``--check`` compares them with the module's).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MECH = _load("_mechanics_for_flow", os.path.join(ROOT, "scripts", "mechanics.py"))
RI = MECH.RI
T_GRID = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
DESIGN_4TO1 = {0: 5.0, 20: 4.3, 40: 3.0, 60: 1.6, 80: 0.55}


def replay(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Our seat's bank trades ``(t, given resource, ratio, features)`` and the features at every ``T_GRID`` turn."""
    colors = list(doc["colors"])
    ours = json.loads(doc["our_seats"]) if isinstance(doc.get("our_seats"), str) else doc.get("our_seats", [0])
    me = colors[int(ours[0])]
    board = MECH.Board(doc["board"])
    pips = MECH.PIPS
    mine: Dict[int, int] = {}         # node -> 1 settlement / 2 city
    turns = 0
    setup = True
    snaps: Dict[int, tuple] = {}
    trades: List[tuple] = []

    def feats() -> tuple:
        prod = [0.0] * 5
        for node, mult in mine.items():
            for coord in board.node_hexes.get(node, ()):
                res, num = board.tiles[coord]
                if res is None or num is None:
                    continue
                prod[RI[res]] += mult * pips.get(int(num), 0) / 36.0
        n_set = sum(1 for m in mine.values() if m == 1)
        n_city = sum(1 for m in mine.values() if m == 2)
        return tuple(prod), (n_set > 0 and n_city < 4)

    for it in doc["actions"]:
        c, t = it[0], it[1]
        v = it[2]
        if t == "ROLL" and setup:
            setup = False
            snaps[0] = feats()
        if t == "BUILD_SETTLEMENT" and c == me:
            mine[int(v)] = 1
        elif t == "BUILD_CITY" and c == me:
            mine[int(v)] = 2
        elif t == "MARITIME_TRADE" and c == me:
            give = [x for x in v[:4] if x is not None]
            trades.append((turns, RI[give[0]], len(give), feats()))
        elif t == "END_TURN":
            turns += 1
            if turns in T_GRID:
                snaps[turns] = feats()
    return {"turns": turns, "trades": trades, "snaps": snaps}


def load(pattern: str, max_games: Optional[int]) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"error: no files match {pattern}")
    return [replay(doc) for doc in MECH.iter_proof_games(paths, max_games)]


def remaining_table(games: Sequence[Dict[str, Any]], only4: bool = False, conditional: bool = True) -> List[float]:
    out = []
    for t in T_GRID:
        pool = [g for g in games if g["turns"] > t] if conditional else list(games)
        if not pool:
            out.append(0.0)
            continue
        out.append(sum(sum(1 for (tt, _r, ratio, _f) in g["trades"] if tt >= t and (ratio == 4 or not only4))
                       for g in pool) / len(pool))
    return out


def features(f: tuple) -> List[Tuple[float, float]]:
    from catanbot.acquisition import flow_features
    prod, city_open = f
    return flow_features(prod, city_open)


def probs(w: Dict[str, Any], f: tuple) -> List[float]:
    from catanbot.acquisition import flow_weights
    return flow_weights(f[0], f[1], w)


def fit_logit(games: Sequence[Dict[str, Any]], ridge: float = 1e-3, iters: int = 60) -> Dict[str, Any]:
    """Maximum likelihood of ``softmax_r(b_r + s share_r + c city_r)`` over our bank trades (``b_wood = 0``)."""
    import numpy as np
    X, y = [], []
    for g in games:
        for (_t, r, _ratio, f) in g["trades"]:
            X.append(features(f))
            y.append(r)
    X = np.array(X, dtype=float)            # (n, 5, 2)
    y = np.array(y, dtype=int)
    n = len(y)

    def design(theta):
        b = np.concatenate([[0.0], theta[:4]])
        return b[None, :] + theta[4] * X[:, :, 0] + theta[5] * X[:, :, 1]

    theta = np.zeros(6)
    for _ in range(iters):
        z = design(theta)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        # gradient / Hessian over theta: d z_r / d theta = [e_r (r > 0) for the intercepts, share_r, city_r]
        D = np.zeros((n, 5, 6))
        for r in range(1, 5):
            D[:, r, r - 1] = 1.0
        D[:, :, 4] = X[:, :, 0]
        D[:, :, 5] = X[:, :, 1]
        onehot = np.zeros((n, 5))
        onehot[np.arange(n), y] = 1.0
        mean_d = (p[:, :, None] * D).sum(axis=1)                       # (n, 6)
        grad = ((onehot[:, :, None] * D).sum(axis=1) - mean_d).sum(axis=0) - ridge * theta
        H = -np.einsum("nr,nri,nrj->ij", p, D, D) + np.einsum("ni,nj->ij", mean_d, mean_d) - ridge * np.eye(6)
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-10:
            break
    z = design(theta)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    ll = float(np.log(p[np.arange(n), y]).sum())
    return {"intercept": [0.0] + [round(float(x), 3) for x in theta[:4]], "share": round(float(theta[4]), 3),
            "city": round(float(theta[5]), 3), "n": n, "loglik": ll}


def give_shares(games: Sequence[Dict[str, Any]]) -> List[float]:
    cnt = [0] * 5
    for g in games:
        for (_t, r, _ratio, _f) in g["trades"]:
            cnt[r] += 1
    tot = sum(cnt) or 1
    return [c / tot for c in cnt]


def split_mae(games, predict) -> float:
    """Per game, predicted vs realised gives per resource given the game's trades: mean |error| per resource."""
    err, n = 0.0, 0
    for g in games:
        pred = [0.0] * 5
        real = [0] * 5
        for (_t, r, _ratio, f) in g["trades"]:
            pr = predict(f)
            for q in range(5):
                pred[q] += pr[q]
            real[r] += 1
        err += sum(abs(pred[q] - real[q]) for q in range(5)) / 5.0
        n += 1
    return err / n if n else float("nan")


def function_mae(games, w, table, t: int) -> Tuple[float, float, int]:
    """``R(t) w_r(features at t)`` vs the gives per resource made after ``t``, per game alive at ``t``: (mean |error|
    per resource per game, same for the flat-shares fallback, games)."""
    from catanbot import acquisition as Q
    saved = (list(Q.FLOW_T), list(Q.FLOW_R))
    Q.FLOW_T[:] = T_GRID
    Q.FLOW_R[:] = table
    try:
        R = Q.flow_remaining(t)
    finally:
        Q.FLOW_T[:], Q.FLOW_R[:] = saved
    flat = give_shares(games)
    err = err_flat = 0.0
    n = 0
    for g in games:
        if g["turns"] <= t or t not in g["snaps"]:
            continue
        pr = probs(w, g["snaps"][t])
        real = [0] * 5
        for (tt, r, _ratio, _f) in g["trades"]:
            if tt >= t:
                real[r] += 1
        err += sum(abs(R * pr[q] - real[q]) for q in range(5)) / 5.0
        err_flat += sum(abs(R * flat[q] - real[q]) for q in range(5)) / 5.0
        n += 1
    return (err / n if n else float("nan")), (err_flat / n if n else float("nan")), n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default=os.path.join(ROOT, "proof", "T1", "logs", "*.gz"))
    ap.add_argument("--validate", default=os.path.join(ROOT, "proof", "T2", "logs", "*.gz"))
    ap.add_argument("--max-games", type=int, default=1000)
    ap.add_argument("--check", action="store_true", help="exit 3 when catanbot/acquisition.py's constants differ")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    fit = load(args.fit, args.max_games)
    val = load(args.validate, args.max_games)
    table = [round(x, 2) for x in remaining_table(fit)]
    table[-1] = 0.0                                   # beyond the last point nothing is left (interpolation end)
    w = fit_logit(fit)
    flat = give_shares(fit)
    from catanbot import acquisition as Q
    res: Dict[str, Any] = {"fit_games": len(fit), "validate_games": len(val), "FLOW_T": T_GRID, "FLOW_R": table,
                           "FLOW_W": {k: w[k] for k in ("intercept", "share", "city")}, "loglik": w["loglik"],
                           "trades_fit": w["n"], "flat_shares_fit": flat}
    res["remaining_fit_uncond_4to1"] = remaining_table(fit, only4=True, conditional=False)
    res["remaining_val_uncond_4to1"] = remaining_table(val, only4=True, conditional=False)
    res["remaining_val"] = remaining_table(val)
    res["trades_per_game"] = {"fit": sum(len(g["trades"]) for g in fit) / len(fit),
                              "validate": sum(len(g["trades"]) for g in val) / len(val)}
    wd = res["FLOW_W"]
    res["split_mae_validate"] = {
        "model": split_mae(val, lambda f: probs(wd, f)),
        "flat_shares": split_mae(val, lambda f: flat),
        "share_only": split_mae(val, lambda f: probs({"intercept": [0.0] * 5, "share": wd["share"], "city": 0.0}, f)),
    }
    res["function_mae_validate"] = {}
    for t in (0, 20, 40, 60):
        m, mf, n = function_mae(val, wd, table, t)
        res["function_mae_validate"][str(t)] = {"model": m, "flat": mf, "games": n}
    print(f"fit: {len(fit)} games, {w['n']} bank trades ({res['trades_per_game']['fit']:.2f} a game); validate: "
          f"{len(val)} games ({res['trades_per_game']['validate']:.2f} a game)")
    print("R(t) conditional (fit):   " + " ".join(f"{t}:{x:.2f}" for t, x in zip(T_GRID, table)))
    print("R(t) conditional (valid): " + " ".join(f"{t}:{x:.2f}" for t, x in zip(T_GRID, res["remaining_val"])))
    print("4:1 unconditional (fit) vs design: " + " ".join(
        f"{t}:{x:.2f}/{DESIGN_4TO1[t]}" for t, x in zip(T_GRID, res["remaining_fit_uncond_4to1"]) if t in DESIGN_4TO1))
    print("4:1 unconditional (valid):         " + " ".join(
        f"{t}:{x:.2f}" for t, x in zip(T_GRID, res["remaining_val_uncond_4to1"]) if t in DESIGN_4TO1))
    print(f"FLOW_W = {res['FLOW_W']} (loglik {w['loglik']:.1f}); flat give shares (fit) "
          + " ".join(f"{x:.3f}" for x in flat))
    print("give split MAE per resource per game (validate): " + ", ".join(
        f"{k} {v:.3f}" for k, v in res["split_mae_validate"].items()))
    for t, d in res["function_mae_validate"].items():
        print(f"port_flow_value inputs at t={t}: R(t) w_r vs gives after t, MAE per resource per game {d['model']:.3f} "
              f"(flat shares {d['flat']:.3f}; {d['games']} games)")
    same = (Q.FLOW_T == T_GRID and Q.FLOW_R == table and Q.FLOW_W["intercept"] == wd["intercept"]
            and Q.FLOW_W["share"] == wd["share"] and Q.FLOW_W["city"] == wd["city"])
    print(f"catanbot/acquisition.py constants {'match' if same else 'DIFFER'}:\nFLOW_R = {table}\nFLOW_W = {wd}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 3 if (args.check and not same) else 0


if __name__ == "__main__":
    sys.exit(main())
