"""Self-play training loop for the value network.

Each iteration:

1. **Generate** games whose seats are drawn from a pool of bots: the current
   best value net (with different exploration settings: epsilon random
   moves and a temperature over search values so trades / responses vary),
   heuristic bots with temperature, and older nets.  Player count is 3 or 4
   per game.  Every decision state is recorded from every player's
   perspective and labelled with the eventual winner.
2. **Fit** a new net on the replay buffer (warm-started from the best net).
3. **Evaluate** the candidate against the current best in a tournament and
   promote it if it wins more.

Usage::

    python -m catanbot train --iters 4 --games 200 --workers 4
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

from .selfplay import generate_dataset, tournament

BASE_SEARCH = "search:depth={depth},beam={beam},expand={expand}"


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
    if best_path:
        pool += [_model_spec(base, best_path, blend),
                 _model_spec(base, best_path, blend, ",eps=0.03,temp=0.3"),
                 _model_spec(base, best_path, blend, ",eps=0.08,temp=0.7"),
                 _model_spec(base, best_path, blend, ",temp=1.0")]
    else:
        pool += [base, f"{base},eps=0.03,temp=0.3", f"{base},eps=0.08,temp=0.7"]
    pool += ["heuristic:temp=0.4,eps=0.05", "heuristic:temp=0.15"]
    for old in archive[-2:]:
        if old != best_path:
            pool.append(_model_spec(base, old, blend, ",eps=0.03,temp=0.3"))
    return pool


HEURISTIC_POOL = ["heuristic:temp=0.5,eps=0.08", "heuristic:temp=0.3,eps=0.03", "heuristic:temp=0.15",
                  "heuristic:temp=0.8,eps=0.1"]


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
    buffer_path = os.path.splitext(args.out)[0] + "_replay.npz"
    if args.resume and os.path.exists(buffer_path):
        try:
            d = np.load(buffer_path)
            X_buf, y_buf = d["X"], d["y"]
            g_buf = d["g"] if "g" in d else np.arange(len(y_buf), dtype=np.int32)
            _log(f"loaded replay buffer with {len(y_buf)} samples", fh)
        except Exception:
            pass
    _log(f"training: iters={args.iters} games={args.games} workers={args.workers} depth={args.depth} "
         f"features={NUM_FEATURES} best={best_path}", fh)

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
        X_buf = np.concatenate([X_buf, X])[-args.buffer:]
        y_buf = np.concatenate([y_buf, y])[-args.buffer:]
        g_buf = np.concatenate([g_buf, gids])[-args.buffer:]
        np.savez_compressed(buffer_path, X=X_buf, y=y_buf, g=g_buf)

        # ---- fit (validation = 10 % of whole games, never positions of a training game) ----
        n = len(y_buf)
        # Deterministic hash of the game id: a game is always validation or always training,
        # so warm-started nets are never validated on games they were fitted on.
        is_val = ((g_buf.astype(np.uint64) * np.uint64(2654435761) + np.uint64(args.seed)) >> np.uint64(7)) % np.uint64(10) == 0
        if not is_val.any():
            is_val[:max(1, n // 10)] = True
        val_idx, tr_idx = np.nonzero(is_val)[0], np.nonzero(~is_val)[0]
        Xtr, ytr = X_buf[tr_idx].astype(np.float32), y_buf[tr_idx]
        Xva, yva = X_buf[val_idx].astype(np.float32), y_buf[val_idx]
        if best_path and args.warm_start:
            net = ValueNet.load(best_path)
        else:
            net = ValueNet(n_in=NUM_FEATURES, hidden=tuple(args.hidden), seed=args.seed + it)
        t_fit = time.time()
        hist = net.fit(Xtr, ytr, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                       weight_decay=args.weight_decay, X_val=Xva, y_val=yva, patience=args.patience,
                       input_noise=args.noise, refit_norm=not (best_path and args.warm_start))
        last = {k: (v[-1] if isinstance(v, list) and v else v) for k, v in hist.items()}
        _log(f"  fit on {len(ytr)} samples in {time.time() - t_fit:.0f}s: {json.dumps(last)}", fh)
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
    p.add_argument("--noise", type=float, default=0.1, help="gaussian input noise (standardised units)")
    p.add_argument("--blend", type=float, default=0.6, help="net weight in the net/heuristic blend used by bots (1 = net only)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
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
