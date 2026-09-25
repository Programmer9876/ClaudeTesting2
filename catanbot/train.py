"""Self-play training loop for the value network.

Each iteration:

1. **Generate** games whose seats are drawn from a pool of bots: the current
   best value net (with different exploration settings: epsilon random
   moves and a temperature over search values so trades / responses vary),
   heuristic bots with temperature, and older nets.  Player count is 3 or 4
   per game.  Every decision state is recorded from every player's
   perspective and labelled with the eventual winner.
2. **Fit** a new net on the replay buffer (warm-started from the best net).
   By default the net is *hand-blind*: its own resources / hand size /
   can-afford features are masked (``--mask-features``, see
   :data:`catanbot.model.HAND_BLIND_FEATURES`; ``--no-mask`` disables it).
   Self-play labels come from a policy that spends every affordable build at
   once, so an unmasked net learns that cards in hand are worth the builds
   they will become and, as a search evaluator, holds the cards instead of
   building; the heuristic half of the blend keeps affordability visible.
3. **Evaluate** the candidate against the current best in a tournament and
   promote it if it wins more.

Usage::

    python -m catanbot train --iters 4 --games 200 --workers 4
    # fit one net on an existing buffer, no games / tournament:
    python -m catanbot train --fit-only --replay models/value_net_replay.npz --out /tmp/net.npz
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .model import HAND_BLIND_FEATURES
from .selfplay import generate_dataset, tournament

BASE_SEARCH = "search:depth={depth},beam={beam},expand={expand}"
DEFAULT_MASK = ",".join(HAND_BLIND_FEATURES)   # --mask-features default: hand-blind net


def _log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def _model_spec(base: str, path: str, blend: float, extra: str = "") -> str:
    b = f",blend={blend}" if 0.0 < blend < 1.0 else ""
    return f"{base},model={path}{b}{extra}"


def build_pool(best_path: Optional[str], archive: List[str], depth: int, beam: int, expand: int,
               rng: random.Random, blend: float = 1.0) -> List[str]:
    base = BASE_SEARCH.format(depth=depth, beam=beam, expand=expand)
    pool: List[str] = []
    # Trading styles are randomised too (DESIGN section 11): a per-game acceptance bias, a
    # temperature over the offer ranking and random proposals / responses, so the value net sees
    # generous, stingy and erratic traders instead of one deterministic acceptance policy.
    if best_path:
        pool += [_model_spec(base, best_path, blend),
                 _model_spec(base, best_path, blend, ",eps=0.03,temp=0.3,accept_bias=0.3"),
                 _model_spec(base, best_path, blend, ",eps=0.08,temp=0.7,offer_temp=0.5,trade_eps=0.05"),
                 _model_spec(base, best_path, blend, ",temp=1.0,accept_bias=0.2,offer_temp=0.3")]
    else:
        pool += [base, f"{base},eps=0.03,temp=0.3,accept_bias=0.3",
                 f"{base},eps=0.08,temp=0.7,offer_temp=0.5,trade_eps=0.05"]
    pool += ["heuristic:temp=0.4,eps=0.05,accept_bias=0.3", "heuristic:temp=0.15,offer_temp=0.5,trade_eps=0.03"]
    for old in archive[-2:]:
        if old != best_path:
            pool.append(_model_spec(base, old, blend, ",eps=0.03,temp=0.3,accept_bias=0.2"))
    return pool


HEURISTIC_POOL = ["heuristic:temp=0.5,eps=0.08,accept_bias=0.3", "heuristic:temp=0.3,eps=0.03,offer_temp=0.5",
                  "heuristic:temp=0.15,accept_bias=0.2,trade_eps=0.03",
                  "heuristic:temp=0.8,eps=0.1,accept_bias=0.3,offer_temp=0.8,trade_eps=0.05"]


def _fit_summary(hist: Dict[str, object]) -> Dict[str, object]:
    """Last-epoch metrics of a fit history plus the metrics of the restored best epoch."""
    last = {k: (v[-1] if isinstance(v, list) and v else v) for k, v in hist.items()}
    be = hist.get("best_epoch", -1)
    if isinstance(be, int) and be >= 0:
        for k in ("val_loss", "val_auc", "val_acc"):
            v = hist.get(k)
            if isinstance(v, list) and len(v) > be:
                last[f"best_{k}"] = v[be]
    return last


def _bias_of(results) -> np.ndarray:
    """Per-sample acceptance bias of the recording bots (0 when a bot has none)."""
    parts = [r.bias if r.bias is not None else np.zeros(len(r.y), np.float32) for r in results if r.y is not None]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def fit_replay(X_buf: np.ndarray, y_buf: np.ndarray, g_buf: np.ndarray, args: argparse.Namespace,
               warm_from: Optional[str] = None, seed: int = 0, log=None):
    """Fit a net on the replay buffer; returns ``(net, history, n_train, n_val)``.

    Validation is 10 % of whole games (deterministic hash of the game id, so a
    game is always validation or always training).  ``warm_from`` continues
    from a saved net (its normalisation is kept); otherwise a fresh net with
    ``args.hidden`` is built.  ``args.mask_features`` (glob patterns over
    feature names, empty = none) is applied to the net in both cases.
    """
    from .features import NUM_FEATURES
    from .model import ValueNet, feature_mask

    n = len(y_buf)
    is_val = ((g_buf.astype(np.uint64) * np.uint64(2654435761) + np.uint64(args.seed)) >> np.uint64(7)) % np.uint64(10) == 0
    if not is_val.any():
        is_val[:max(1, n // 10)] = True
    val_idx, tr_idx = np.nonzero(is_val)[0], np.nonzero(~is_val)[0]
    # float16 slices: ValueNet.fit standardises into one float32 copy itself (the buffer is large)
    Xtr, ytr = X_buf[tr_idx], y_buf[tr_idx]
    Xva, yva = X_buf[val_idx], y_buf[val_idx]
    mask = feature_mask(getattr(args, "mask_features", None))
    if warm_from:
        net = ValueNet.load(warm_from)
    else:
        net = ValueNet(n_in=NUM_FEATURES, hidden=tuple(args.hidden), seed=seed)
    net.input_mask = mask
    if log:
        log(f"  input mask: {len(net.masked_features())} features hidden" +
            (f" ({', '.join(net.masked_features())})" if net.masked_features() else ""))
    hist = net.fit(Xtr, ytr, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   weight_decay=args.weight_decay, X_val=Xva, y_val=yva, patience=args.patience,
                   input_noise=args.noise, refit_norm=not warm_from, log=log)
    return net, hist, len(tr_idx), len(val_idx)


def train(args: argparse.Namespace) -> Dict[str, object]:
    from .features import NUM_FEATURES
    from .model import ValueNet

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    log_path = os.path.splitext(args.out)[0] + "_train.log"
    fh = open(log_path, "a")
    rng = random.Random(args.seed)
    best_path: Optional[str] = args.out if (args.resume and os.path.exists(args.out)) else None
    archive: List[str] = []
    history: List[dict] = []
    log_json = os.path.splitext(args.out)[0] + "_log.json"
    if args.resume and os.path.exists(log_json):
        try:
            with open(log_json) as f:
                history = json.load(f).get("iterations", [])
        except Exception:
            history = []
    X_buf = np.zeros((0, NUM_FEATURES), np.float16)
    y_buf = np.zeros(0, np.float32)
    g_buf = np.zeros(0, np.int32)          # game id of every sample (validation is split by game)
    b_buf = np.zeros(0, np.float32)        # acceptance bias of the sample's bot (trading-style metadata)
    buffer_path = os.path.splitext(args.out)[0] + "_replay.npz"
    load_path = getattr(args, "replay", None) or buffer_path
    if (args.resume or getattr(args, "replay", None) or getattr(args, "fit_only", False)) and os.path.exists(load_path):
        try:
            d = np.load(load_path)
            X_buf, y_buf = d["X"], d["y"]
            g_buf = d["g"] if "g" in d else np.arange(len(y_buf), dtype=np.int32)
            b_buf = d["bias"] if "bias" in d else np.zeros(len(y_buf), np.float32)
            _log(f"loaded replay buffer {load_path} with {len(y_buf)} samples", fh)
        except Exception:
            pass
    _log(f"training: iters={args.iters} games={args.games} workers={args.workers} depth={args.depth} "
         f"features={NUM_FEATURES} best={best_path} mask={args.mask_features!r}", fh)

    if getattr(args, "fit_only", False):
        # One fit on the loaded buffer (no games, no tournament); the net is written to --out.
        if len(y_buf) == 0:
            fh.close()
            raise SystemExit(f"--fit-only: no replay buffer at {load_path}")
        t_fit = time.time()
        net, hist, n_tr, n_va = fit_replay(X_buf, y_buf, g_buf, args,
                                           warm_from=best_path if args.warm_start else None,
                                           seed=args.seed, log=lambda m: _log(m, fh))
        last = _fit_summary(hist)
        _log(f"  fit on {n_tr} samples ({n_va} validation) in {time.time() - t_fit:.0f}s: {json.dumps(last)}", fh)
        net.save(args.out)
        _log(f"  wrote {args.out}", fh)
        fh.close()
        return {"best": args.out, "iterations": [], "fit": last, "samples": int(len(y_buf))}

    for it in range(len(history), len(history) + args.iters):
        t_it = time.time()
        pool = build_pool(best_path, archive, args.depth, args.beam, args.expand, rng, args.blend)
        _log(f"iter {it}: generating {args.games} games with pool {pool}", fh)

        def progress(k, n, r):
            if k % max(1, n // 10) == 0 or k == n:
                _log(f"  {k}/{n} games (last: {r.turns} turns, {r.duration:.1f}s, winner {r.winner})", fh)

        X, y, vpf, results = generate_dataset(pool, args.games, workers=args.workers, seed=args.seed * 1000 + it,
                                              num_players_choices=(3, 4) if args.mixed_players else (4,),
                                              max_turns=args.max_turns, sample_every=args.sample_every,
                                              progress=progress)
        if args.heur_games > 0:
            # Cheap, diverse extra games from heuristic bots: memorisation of game identity is the
            # main failure mode of the value net, and more distinct games is the best cure.
            t_h = time.time()
            Xh, yh, vh, rh = generate_dataset(HEURISTIC_POOL, args.heur_games, workers=args.workers,
                                              seed=args.seed * 1000 + it + 500,
                                              num_players_choices=(3, 4) if args.mixed_players else (4,),
                                              max_turns=args.max_turns, sample_every=args.sample_every)
            _log(f"  +{len(yh)} samples from {len(rh)} heuristic games in {time.time() - t_h:.0f}s", fh)
            X = np.concatenate([X, Xh]) if len(y) else Xh
            y = np.concatenate([y, yh]) if len(y) else yh
            results = list(results) + list(rh)
        wins: Dict[str, List[int]] = {}
        for r in results:
            for i, s in enumerate(r.specs):
                wins.setdefault(s, []).append(1 if r.winner == i else 0)
        gen_stats = {s: (sum(v) / len(v), len(v)) for s, v in wins.items()}
        avg_turns = sum(r.turns for r in results) / max(1, len(results))
        _log(f"  generated {len(y)} samples from {len(results)} games (avg {avg_turns:.0f} turns); "
             f"win rates: " + ", ".join(f"{s}: {w:.2f} ({n})" for s, (w, n) in gen_stats.items()), fh)
        gids = np.concatenate([np.full(len(r.y), it * 100000 + k, dtype=np.int32)
                               for k, r in enumerate(results) if r.y is not None]) if len(y) else np.zeros(0, np.int32)
        bias = _bias_of(results) if len(y) else np.zeros(0, np.float32)
        if len(bias) != len(y):
            bias = np.zeros(len(y), np.float32)
        X_buf = np.concatenate([X_buf, X])[-args.buffer:]
        y_buf = np.concatenate([y_buf, y])[-args.buffer:]
        g_buf = np.concatenate([g_buf, gids])[-args.buffer:]
        b_buf = np.concatenate([b_buf, bias])[-args.buffer:]
        np.savez_compressed(buffer_path, X=X_buf, y=y_buf, g=g_buf, bias=b_buf)
        if len(bias):
            _log(f"  trading styles: {float((bias != 0).mean()):.0%} of samples from biased traders, "
                 f"mean |bias| {float(np.abs(bias).mean()):.3f}", fh)

        # ---- fit (validation = 10 % of whole games, never positions of a training game) ----
        n = len(y_buf)
        t_fit = time.time()
        net, hist, n_tr, _ = fit_replay(X_buf, y_buf, g_buf, args,
                                        warm_from=best_path if (best_path and args.warm_start) else None,
                                        seed=args.seed + it)
        last = _fit_summary(hist)
        _log(f"  fit on {n_tr} samples in {time.time() - t_fit:.0f}s: {json.dumps(last)}", fh)
        cand_path = os.path.splitext(args.out)[0] + f"_candidate.npz"
        net.save(cand_path)

        # ---- evaluate -----------------------------------------------------
        base = BASE_SEARCH.format(depth=args.depth, beam=args.beam, expand=args.expand)
        cand_spec = _model_spec(base, cand_path, args.blend)
        best_spec = _model_spec(base, best_path, args.blend) if best_path else base
        t_ev = time.time()
        res = tournament([cand_spec, best_spec], games=args.eval_games, workers=args.workers,
                         seed=args.seed * 31 + it, num_players=4, max_turns=args.max_turns)
        cw = res["summary"][cand_spec]["win_rate"]
        bw = res["summary"][best_spec]["win_rate"]
        promoted = cw > bw * args.promote_ratio
        _log(f"  eval ({args.eval_games} games, {time.time() - t_ev:.0f}s): candidate {cw:.2f} vs best {bw:.2f} "
             f"(avg vp {res['summary'][cand_spec]['avg_vp']:.1f} vs {res['summary'][best_spec]['avg_vp']:.1f}) -> "
             f"{'PROMOTED' if promoted else 'rejected'}", fh)
        if promoted:
            shutil.copyfile(cand_path, args.out)
            arch = os.path.splitext(args.out)[0] + f"_iter{it}.npz"
            shutil.copyfile(cand_path, arch)
            archive.append(arch)
            best_path = args.out
        history.append({"iter": it, "samples": int(n), "avg_turns": avg_turns, "gen_win_rates": gen_stats,
                        "fit": last, "eval_candidate": cw, "eval_best": bw, "promoted": bool(promoted),
                        "trade_bias_abs_mean": float(np.abs(bias).mean()) if len(bias) else 0.0,
                        "seconds": time.time() - t_it})
        with open(log_json, "w") as f:
            json.dump({"iterations": history, "best": best_path}, f, indent=1)
    fh.close()
    return {"best": best_path, "iterations": history}


def build_parser(sub=None) -> argparse.ArgumentParser:
    p = sub.add_parser("train", help="self-play training of the value net") if sub is not None else \
        argparse.ArgumentParser(description="self-play training")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    p.add_argument("--out", default="models/value_net.npz")
    p.add_argument("--resume", action="store_true", help="continue from --out and its replay buffer")
    p.add_argument("--buffer", type=int, default=1500000)
    p.add_argument("--heur-games", type=int, default=600, help="extra cheap heuristic-bot games per iteration")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.25, help="gaussian input noise (standardised units)")
    p.add_argument("--blend", type=float, default=0.6, help="net weight in the net/heuristic blend used by bots (1 = net only)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--mask-features", default=DEFAULT_MASK, metavar="PATTERNS",
                   help="comma-separated glob patterns of feature names the net never sees (stored with the net); "
                        "default: the own-hand block (resources, hand size, 7-risk, can-afford flags)")
    p.add_argument("--no-mask", dest="mask_features", action="store_const", const="",
                   help="train on all features (the pre-mask behaviour)")
    p.add_argument("--replay", default=None, metavar="PATH",
                   help="replay buffer to load instead of <out>_replay.npz (read only)")
    p.add_argument("--fit-only", action="store_true",
                   help="fit one net on the loaded replay buffer and write it to --out (no games, no tournament)")
    p.add_argument("--eval-games", type=int, default=40)
    p.add_argument("--promote-ratio", type=float, default=1.0, help="promote if cand_win > ratio * best_win")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--expand", type=int, default=8)
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=400)
    p.add_argument("--mixed-players", action="store_true", default=True)
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false", default=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
