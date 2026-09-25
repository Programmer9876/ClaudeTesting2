"""Self-play training loop for the value network.

Each iteration:

1. **Generate** games whose seats are drawn from a pool of bots: the current
   best value net (with different exploration settings: epsilon random
   moves and a temperature over search values so trades / responses vary),
   heuristic bots with temperature, and older nets.  Player count is 3 or 4
   per game.  Every decision state is recorded from every player's
   perspective and labelled with the eventual winner.
2. **Fit** a new net on the replay buffer (warm-started from the best net):
   binary cross-entropy on the Monte-Carlo outcome labels plus a **pairwise
   ranking term** on *sibling afterstates* (``--rank-weight``, default on).
   The self-play labels come from a policy that spends every affordable
   build at once, so they hold no counterfactual for holding the cards, and
   a net fitted on them alone ranks "hold" level with "build" and drifts the
   search into passive lines (the 0.11-vs-0.39 rejection).  At a share
   ``--rank-rate`` of the main-phase decisions of the generated games the
   afterstate of every legal action is recorded with the heuristic
   evaluator's value; pairs (build vs END_TURN first) are fitted on their
   logit difference: ``--rank-loss delta`` (default) regresses
   ``z_better - z_worse`` onto the heuristic's own logit gap, so the net
   reproduces the heuristic's ordering *and* magnitudes (a road is worth a
   little, a settlement a lot, an offer a card or two) while the BCE keeps
   its absolute values calibrated; ``--rank-loss hinge`` (the first attempt)
   only orders, ``softplus(margin - (z_better - z_worse))``, and because the
   softplus never stops pushing it inflated every build-vs-hold gap until the
   search spent every card on roads (0.08 vs 0.42 with the net deciding the
   main phase alone).  The squared error alone, in turn, barely moves the
   small gaps that decide whether to *hold* the cards (END_TURN better than a
   road or a bank trade, 4-13 % of those pairs ordered right), so the delta
   mode adds a bounded sign hinge ``--rank-sign-margin`` (``relu(margin -
   difference)``, zero once the sign is right: 82-98 % of the hold pairs) and
   the hold pairs, outnumbered five to one, get ``--rank-hold-weight``.
   The same is recorded for the decisions *outside* the main phase that the
   search bot also makes with its evaluator: every setup placement
   (``--rank-setup-rate``: the best spots, each with its setup road) and a
   share ``--rank-other-rate`` of the incoming offers (accept / reject),
   robber moves and discards, whose pairs use the smaller ``--rank-gap-other``
   (an accept / reject pair differs by a card or two).  These phases, not
   the main phase, were where the 0.11-vs-0.39 candidate lost: with the net
   deciding only the main phase the bot was at parity with heuristic search,
   answering offers with it too dropped it to 0.12 vs 0.38 (12 games each),
   and at setup it picked the heuristic's best spot in a quarter of the
   nodes (6 % less production).  A net that orders the main phase well but
   answers offers and places settlements by noise still loses.
   ``--mask-features`` can additionally hide features from the net (e.g. the
   own-hand block, ``catanbot.model.HAND_BLIND_FEATURES``); off by default.
3. **Evaluate** the candidate against the current best in a tournament and
   promote it if it wins more.

Usage::

    python -m catanbot train --iters 4 --games 200 --workers 4
    # fit one net on an existing buffer (siblings from --rank-games fresh heuristic games), no tournament:
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .selfplay import SIBLING_OTHER_KINDS, generate_dataset, sibling_arrays, tournament

BASE_SEARCH = "search:depth={depth},beam={beam},expand={expand}"
DEFAULT_MASK = ""   # --mask-features default: none (catanbot.model.HAND_BLIND_FEATURES hides the own hand)
SIBLING_KEYS = ("Xs", "sn", "sh", "sk", "Xm")   # replay-buffer keys of the sibling afterstates (optional, additive)


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


def _val_split(ids: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic 10 % validation mask by id (a game / node is always validation or always training)."""
    return ((ids.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) >> np.uint64(7)) % np.uint64(10) == 0


def pair_margins(h_pos: np.ndarray, h_neg: np.ndarray, scale: float, max_margin: float,
                 min_margin: float = 0.05) -> np.ndarray:
    """Per-pair logit margins (hinge mode) or target differences (delta mode): ``scale`` times the
    heuristic's logit gap, clipped to ``[min_margin, max_margin]`` (``scale <= 0``: ``max_margin`` for
    every pair)."""
    if scale <= 0:
        return np.full(len(h_pos), float(max_margin))
    eps = 1e-4
    hp = np.clip(np.asarray(h_pos, np.float64), eps, 1 - eps)
    hn = np.clip(np.asarray(h_neg, np.float64), eps, 1 - eps)
    gap = np.log(hp / (1 - hp)) - np.log(hn / (1 - hn))
    return np.clip(scale * gap, min_margin, max_margin)


def offer_pairs(kind_pos: np.ndarray, kind_neg: np.ndarray) -> np.ndarray:
    """Mask of the incoming-offer pairs (accept vs reject of the same offer)."""
    from .selfplay import SIBLING_OFFER_KINDS
    return np.isin(np.asarray(kind_pos), SIBLING_OFFER_KINDS) & np.isin(np.asarray(kind_neg), SIBLING_OFFER_KINDS)


def pair_weights(kind_pos: np.ndarray, kind_neg: np.ndarray, offer_weight: float = 1.0,
                 setup_weight: float = 1.0, hold_weight: float = 1.0) -> np.ndarray:
    """Per-pair weights by decision type, normalised to mean 1: incoming-offer pairs (accept / reject)
    get ``offer_weight``, setup-placement pairs ``setup_weight``, pairs whose *better* side is END_TURN
    (hold the cards rather than build the road / make the bank trade) ``hold_weight``, all others 1.
    Offers are the most frequent decision of a game (about 70 per seat) but one pair per node, setup
    placements the rarest (2 per seat) but a dozen candidates each, so unweighted pairs are dominated by
    setup; the hold pairs are outnumbered five to one by build-better pairs, and a net fitted without the
    weight learns a road as a per-kind constant (+0.05) and never holds (4-13 % of the hold pairs right,
    then it builds roads with the settlement's cards in the search)."""
    from .selfplay import SIBLING_KIND_SETUP, SIBLING_OFFER_KINDS
    kp = np.asarray(kind_pos)
    kn = np.asarray(kind_neg)
    w = np.ones(len(kp), np.float64)
    offer = np.isin(kp, SIBLING_OFFER_KINDS) & np.isin(kn, SIBLING_OFFER_KINDS)
    setup = (kp == SIBLING_KIND_SETUP) & (kn == SIBLING_KIND_SETUP)
    w[offer] = float(offer_weight)
    w[setup] = float(setup_weight)
    w[kp == 0] = float(hold_weight)
    if len(w) and w.mean() > 0:
        w /= w.mean()
    return w


def build_pairs(node: np.ndarray, h: np.ndarray, kind: np.ndarray, gap: float = 0.02, per_node: int = 6,
                seed: int = 0, gap_other: Optional[float] = None,
                other_kinds: Sequence[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """Training pairs ``(better_idx, worse_idx)`` from sibling afterstates.

    Rows sharing a ``node`` id are the afterstates of one decision (each
    action followed by END_TURN, see ``selfplay.play_game``); ``h`` is the
    heuristic evaluator's value and ``kind`` the action kind
    (``selfplay.SIBLING_KINDS``, 0 = END_TURN).  A pair is formed only when
    the heuristic values differ by at least ``gap``.  Per node: the top
    afterstate vs END_TURN first (the build-vs-hold counterfactual), then up
    to ``per_node`` pairs involving the top afterstate or END_TURN, then
    other pairs.  Nodes whose rows are all of a kind in ``other_kinds``
    (incoming offers, robber moves, discards: ``selfplay.SIBLING_OTHER_KINDS``)
    use ``gap_other`` instead: an accept / reject pair differs by a card or
    two, about 0.001-0.003 of heuristic win probability, and the search
    decides them by that sign.
    """
    rng = random.Random(seed)
    node = np.asarray(node)
    order = np.argsort(node, kind="stable")
    bounds = np.flatnonzero(np.diff(node[order])) + 1
    other = set(int(k) for k in other_kinds)
    pos: List[int] = []
    neg: List[int] = []
    for grp in np.split(order, bounds):
        if len(grp) < 2:
            continue
        hv = h[grp]
        g = gap
        if gap_other is not None and other and all(int(k) in other for k in kind[grp]):
            g = gap_other
        t = int(np.argmax(hv))
        ends = np.flatnonzero(kind[grp] == 0)
        e = int(ends[0]) if len(ends) else -1
        chosen: List[Tuple[int, int]] = []
        if e >= 0 and t != e and hv[t] - hv[e] >= g:
            chosen.append((t, e))
        first: List[Tuple[int, int]] = []
        rest: List[Tuple[int, int]] = []
        for i in range(len(grp)):
            for j in range(len(grp)):
                if i == j or hv[i] - hv[j] < g or (i, j) in chosen:
                    continue
                (first if (i == t or j == t or i == e or j == e) else rest).append((i, j))
        rng.shuffle(first)
        rng.shuffle(rest)
        chosen += (first + rest)[:max(0, per_node - len(chosen))]
        for i, j in chosen:
            pos.append(int(grp[i]))
            neg.append(int(grp[j]))
    return np.asarray(pos, np.int64), np.asarray(neg, np.int64)


def fit_replay(X_buf: np.ndarray, y_buf: np.ndarray, g_buf: np.ndarray, args: argparse.Namespace,
               warm_from: Optional[str] = None, seed: int = 0, log=None, siblings=None):
    """Fit a net on the replay buffer; returns ``(net, history, n_train, n_val)``.

    Validation is 10 % of whole games (deterministic hash of the game id, so a
    game is always validation or always training).  ``warm_from`` continues
    from a saved net (its normalisation is kept); otherwise a fresh net with
    ``args.hidden`` is built.  ``args.mask_features`` (glob patterns over
    feature names, empty = none) is applied to the net in both cases.
    ``siblings = (Xs, node, h, kind, Xm)`` (see ``selfplay.sibling_arrays``)
    adds the pairwise ranking term with ``args.rank_weight`` / ``rank_margin``
    / ``rank_gap`` / ``rank_pairs`` and the mid-turn / END_TURN horizon
    consistency term with ``args.rank_consistency`` (validation pairs come
    from 10 % of the nodes); ``history["n_pairs"]`` / ``["n_val_pairs"]`` /
    ``["n_cons"]`` count them.
    """
    from .features import NUM_FEATURES
    from .model import ValueNet, feature_mask

    n = len(y_buf)
    is_val = _val_split(g_buf, args.seed)
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
    if log and net.masked_features():
        log(f"  input mask: {len(net.masked_features())} features hidden ({', '.join(net.masked_features())})")
    pairs = val_pairs = cons = val_cons = None
    n_pairs = n_val_pairs = n_cons = 0
    rank_weight = float(getattr(args, "rank_weight", 0.0) or 0.0)
    cons_weight = float(getattr(args, "rank_consistency", 0.0) or 0.0)
    pair_mode = getattr(args, "rank_loss", "hinge") or "hinge"
    delta = pair_mode == "delta"
    # delta mode regresses the difference onto the heuristic's (scaled) logit gap: every pair is informative
    # (a zero gap says "these two are equal"), so the pair gaps default to 0 and the target is not floored
    gap = args.rank_gap if args.rank_gap is not None else (0.0 if delta else 0.01)
    gap_other = getattr(args, "rank_gap_other", None)
    if gap_other is None:
        gap_other = 0.0 if delta else 0.001
    m_scale = getattr(args, "rank_margin_scale", 0.0)
    m_max = args.rank_margin
    m_min = 0.0 if delta else 0.05
    if delta:
        m_scale = getattr(args, "rank_delta_scale", 1.0)
        m_max = getattr(args, "rank_delta_cap", 2.0)
    if siblings is not None and len(siblings[0]) > 0 and (rank_weight > 0 or cons_weight > 0):
        Xs, sn, sh, sk = siblings[:4]
        Xm = siblings[4] if len(siblings) > 4 and siblings[4] is not None and len(siblings[4]) == len(sn) else None
        sval = _val_split(np.asarray(sn), args.seed)
        if rank_weight > 0:
            pos, neg = build_pairs(sn, sh, sk, gap=gap, per_node=args.rank_pairs, seed=seed,
                                   gap_other=gap_other, other_kinds=SIBLING_OTHER_KINDS)
            if len(pos):
                v = sval[pos]
                m = pair_margins(sh[pos], sh[neg], m_scale, m_max, m_min)
                offer_scale = float(getattr(args, "rank_offer_scale", 1.0) or 1.0)
                if delta and offer_scale != 1.0:
                    # an accept / reject pair differs by a card or two (about 0.02 heuristic logits, below what
                    # the net resolves): the target is amplified so the *sign* is learned, capped like the rest
                    off = offer_pairs(sk[pos], sk[neg])
                    m[off] = np.minimum(m[off] * offer_scale, m_max)
                w = pair_weights(sk[pos], sk[neg], getattr(args, "rank_offer_weight", 1.0),
                                 getattr(args, "rank_setup_weight", 1.0), getattr(args, "rank_hold_weight", 1.0))
                pairs = (Xs[pos[~v]], Xs[neg[~v]], m[~v], w[~v])
                val_pairs = (Xs[pos[v]], Xs[neg[v]], m[v], w[v])
                n_pairs, n_val_pairs = int((~v).sum()), int(v.sum())
            if log:
                log(f"  ranking pairs ({pair_mode}): {n_pairs} train / {n_val_pairs} val from {len(np.unique(sn))} nodes "
                    f"({len(sn)} afterstates); weight {rank_weight}, {'target' if delta else 'margin'} = heuristic "
                    f"logit gap x {m_scale} clipped to [{m_min}, {m_max}] (mean {float(m.mean()) if len(pos) else 0:.3f}), "
                    f"gap {gap} / {gap_other} (offers, robber, discards); "
                    f"pair weights: offers {getattr(args, 'rank_offer_weight', 1.0)}, setup "
                    f"{getattr(args, 'rank_setup_weight', 1.0)}, hold {getattr(args, 'rank_hold_weight', 1.0)}; "
                    f"offer target scale "
                    f"{getattr(args, 'rank_offer_scale', 1.0) if delta else 1.0}; sign hinge margin "
                    f"{getattr(args, 'rank_sign_margin', 0.0) if delta else 0.0} x {getattr(args, 'rank_sign_weight', 1.0)}")
                if len(pos):
                    from .selfplay import SIBLING_KINDS
                    counts = np.bincount(np.asarray(sk[pos], dtype=np.int64), minlength=len(SIBLING_KINDS) + 1)
                    mix = ", ".join(f"{SIBLING_KINDS[k] if k < len(SIBLING_KINDS) else 'other'} {c}"
                                    for k, c in enumerate(counts) if c)
                    log(f"  pairs by kind of the better afterstate: {mix}")
        if cons_weight > 0 and Xm is not None:
            cons = (Xm[~sval], Xs[~sval])
            val_cons = (Xm[sval], Xs[sval])
            n_cons = int((~sval).sum())
            if log:
                log(f"  horizon-consistency pairs (mid-turn vs after END_TURN): {n_cons} train / {int(sval.sum())} val; "
                    f"weight {cons_weight}")
    hist = net.fit(Xtr, ytr, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   weight_decay=args.weight_decay, X_val=Xva, y_val=yva, patience=args.patience,
                   input_noise=args.noise, refit_norm=not warm_from, log=log,
                   pairs=pairs, val_pairs=val_pairs, pair_weight=rank_weight,
                   pair_margin=getattr(args, "rank_margin", 0.5), pair_batch=getattr(args, "rank_batch", 256),
                   consistency=cons, val_consistency=val_cons, consistency_weight=cons_weight, pair_mode=pair_mode,
                   pair_sign_margin=float(getattr(args, "rank_sign_margin", 0.0) or 0.0),
                   pair_sign_weight=float(getattr(args, "rank_sign_weight", 1.0)))
    hist["n_pairs"] = n_pairs
    hist["n_val_pairs"] = n_val_pairs
    hist["n_cons"] = n_cons
    return net, hist, len(tr_idx), len(val_idx)


def _cap_siblings(Xs, sn, sh, sk, Xm, max_rows: int):
    """Keep the last ``max_rows`` sibling rows without splitting the first kept node."""
    if len(sn) <= max_rows:
        return Xs, sn, sh, sk, Xm
    start = len(sn) - max_rows
    while start < len(sn) and start > 0 and sn[start] == sn[start - 1]:
        start += 1
    return Xs[start:], sn[start:], sh[start:], sk[start:], Xm[start:]


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
    # sibling afterstates for the ranking term (selfplay.sibling_arrays): features, node id, heuristic value, kind
    Xs_buf = np.zeros((0, NUM_FEATURES), np.float16)
    sn_buf = np.zeros(0, np.int64)
    sh_buf = np.zeros(0, np.float32)
    sk_buf = np.zeros(0, np.int8)
    Xm_buf = np.zeros((0, NUM_FEATURES), np.float16)
    buffer_path = os.path.splitext(args.out)[0] + "_replay.npz"
    load_path = getattr(args, "replay", None) or buffer_path
    if (args.resume or getattr(args, "replay", None) or getattr(args, "fit_only", False)) and os.path.exists(load_path):
        try:
            d = np.load(load_path)
            X_buf, y_buf = d["X"], d["y"]
            g_buf = d["g"] if "g" in d else np.arange(len(y_buf), dtype=np.int32)
            b_buf = d["bias"] if "bias" in d else np.zeros(len(y_buf), np.float32)
            if all(k in d for k in SIBLING_KEYS):
                Xs_buf, sn_buf, sh_buf, sk_buf, Xm_buf = (d[k] for k in SIBLING_KEYS)
            _log(f"loaded replay buffer {load_path} with {len(y_buf)} samples and {len(sn_buf)} sibling rows", fh)
        except Exception:
            pass
    if getattr(args, "siblings", None):
        d = np.load(args.siblings)
        Xs_buf, sn_buf, sh_buf, sk_buf, Xm_buf = (d[k] for k in SIBLING_KEYS)
        _log(f"loaded {len(sn_buf)} sibling rows ({len(np.unique(sn_buf))} nodes) from {args.siblings}", fh)
    _log(f"training: iters={args.iters} games={args.games} workers={args.workers} depth={args.depth} "
         f"features={NUM_FEATURES} best={best_path} mask={args.mask_features!r}", fh)

    if getattr(args, "fit_only", False):
        # One fit on the loaded buffer (no games, no tournament); the net is written to --out.
        if len(y_buf) == 0:
            fh.close()
            raise SystemExit(f"--fit-only: no replay buffer at {load_path}")
        if args.rank_weight > 0 and len(sn_buf) == 0 and args.rank_games > 0 and args.rank_rate > 0:
            # the buffer predates sibling recording: sample nodes from cheap fresh heuristic games
            t_s = time.time()
            _, _, _, rs = generate_dataset(HEURISTIC_POOL, args.rank_games, workers=args.workers,
                                           seed=args.seed * 1000 + 900,
                                           num_players_choices=(3, 4) if args.mixed_players else (4,),
                                           max_turns=args.max_turns, sample_every=args.sample_every,
                                           sibling_rate=args.rank_rate, sibling_setup_rate=args.rank_setup_rate,
                                           sibling_other_rate=args.rank_other_rate)
            Xs_buf, sn_buf, sh_buf, sk_buf, Xm_buf = sibling_arrays(rs)
            sib_path = os.path.splitext(args.out)[0] + "_siblings.npz"
            np.savez_compressed(sib_path, Xs=Xs_buf, sn=sn_buf, sh=sh_buf, sk=sk_buf, Xm=Xm_buf)
            _log(f"  {len(sn_buf)} sibling rows ({len(np.unique(sn_buf))} nodes) from {len(rs)} heuristic games "
                 f"in {time.time() - t_s:.0f}s -> {sib_path} (reuse with --siblings)", fh)
        t_fit = time.time()
        net, hist, n_tr, n_va = fit_replay(X_buf, y_buf, g_buf, args,
                                           warm_from=best_path if args.warm_start else None,
                                           seed=args.seed, log=lambda m: _log(m, fh),
                                           siblings=(Xs_buf, sn_buf, sh_buf, sk_buf, Xm_buf))
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
                                              progress=progress, sibling_rate=args.rank_rate,
                                              sibling_setup_rate=args.rank_setup_rate,
                                              sibling_other_rate=args.rank_other_rate)
        if args.heur_games > 0:
            # Cheap, diverse extra games from heuristic bots: memorisation of game identity is the
            # main failure mode of the value net, and more distinct games is the best cure.
            t_h = time.time()
            Xh, yh, vh, rh = generate_dataset(HEURISTIC_POOL, args.heur_games, workers=args.workers,
                                              seed=args.seed * 1000 + it + 500,
                                              num_players_choices=(3, 4) if args.mixed_players else (4,),
                                              max_turns=args.max_turns, sample_every=args.sample_every,
                                              sibling_rate=args.rank_rate, sibling_setup_rate=args.rank_setup_rate,
                                           sibling_other_rate=args.rank_other_rate)
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
        if args.rank_rate > 0 or args.rank_setup_rate > 0 or (args.rank_other_rate or 0) > 0:
            Xs, sn, sh, sk, Xm = sibling_arrays(results, base_id=it * 10_000_000)
            Xs_buf, sn_buf, sh_buf, sk_buf, Xm_buf = _cap_siblings(
                np.concatenate([Xs_buf, Xs]), np.concatenate([sn_buf, sn]), np.concatenate([sh_buf, sh]),
                np.concatenate([sk_buf, sk]), np.concatenate([Xm_buf, Xm]), args.rank_buffer)
            _log(f"  +{len(sn)} sibling afterstates ({len(np.unique(sn))} nodes); buffer {len(sn_buf)} rows", fh)
        np.savez_compressed(buffer_path, X=X_buf, y=y_buf, g=g_buf, bias=b_buf,
                            Xs=Xs_buf, sn=sn_buf, sh=sh_buf, sk=sk_buf, Xm=Xm_buf)
        if len(bias):
            _log(f"  trading styles: {float((bias != 0).mean()):.0%} of samples from biased traders, "
                 f"mean |bias| {float(np.abs(bias).mean()):.3f}", fh)

        # ---- fit (validation = 10 % of whole games, never positions of a training game) ----
        n = len(y_buf)
        t_fit = time.time()
        net, hist, n_tr, _ = fit_replay(X_buf, y_buf, g_buf, args,
                                        warm_from=best_path if (best_path and args.warm_start) else None,
                                        seed=args.seed + it, log=lambda m: _log(m, fh),
                                        siblings=(Xs_buf, sn_buf, sh_buf, sk_buf, Xm_buf))
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
                        "default: none (catanbot.model.HAND_BLIND_FEATURES names the own-hand block)")
    p.add_argument("--no-mask", dest="mask_features", action="store_const", const="",
                   help="train on all features (the pre-mask behaviour)")
    p.add_argument("--rank-weight", type=float, default=1.0,
                   help="weight of the pairwise sibling-ranking term (0 = plain BCE, the pre-ranking behaviour)")
    p.add_argument("--rank-rate", type=float, default=0.03,
                   help="share of main-phase decisions of the generated games whose sibling afterstates are recorded")
    p.add_argument("--rank-setup-rate", type=float, default=1.0,
                   help="share of setup-settlement decisions whose placement afterstates are recorded as siblings "
                        "(the best 12 spots, each with its setup road; 0 = none)")
    p.add_argument("--rank-other-rate", type=float, default=None,
                   help="share of incoming-offer / robber / discard decisions recorded as siblings "
                        "(default 5 x --rank-rate; 0 = none)")
    p.add_argument("--rank-offer-weight", type=float, default=6.0,
                   help="weight of incoming-offer (accept / reject) pairs in the ranking term, relative to 1 for "
                        "main-phase / robber / discard pairs (weights are normalised to mean 1)")
    p.add_argument("--rank-setup-weight", type=float, default=0.5,
                   help="weight of setup-placement pairs in the ranking term (a dozen candidates per node)")
    p.add_argument("--rank-hold-weight", type=float, default=4.0,
                   help="weight of the pairs whose better side is END_TURN (hold the cards instead of a road / "
                        "bank trade / dev card): outnumbered by the build-better pairs, they carry the counterfactual")
    p.add_argument("--rank-gap-other", type=float, default=None,
                   help="minimum heuristic gap for pairs of offer / robber / discard siblings (their values differ "
                        "by a card or two, far less than --rank-gap; default 0.001 in hinge mode, 0 in delta mode)")
    p.add_argument("--rank-consistency", type=float, default=1.0,
                   help="weight of the mid-turn / END_TURN horizon-consistency term on the siblings (0 = off)")
    p.add_argument("--rank-loss", choices=["hinge", "delta"], default="delta",
                   help="pair loss: 'delta' regresses the net's sibling logit difference onto the heuristic's "
                        "(x --rank-delta-scale): ordering and magnitude; 'hinge' (softplus past a margin) orders "
                        "only and inflates every build-vs-END_TURN gap until the search spends every card on roads")
    p.add_argument("--rank-delta-scale", type=float, default=1.0,
                   help="delta mode: target difference = scale x heuristic logit gap")
    p.add_argument("--rank-delta-cap", type=float, default=2.0, help="delta mode: max target difference (logits)")
    p.add_argument("--rank-sign-margin", type=float, default=0.05,
                   help="delta mode: bounded sign hinge relu(margin - difference) on pairs with a positive target "
                        "(0 = off); the squared error alone barely moves the small hold-vs-road gaps")
    p.add_argument("--rank-sign-weight", type=float, default=1.0, help="weight of the sign hinge (delta mode)")
    p.add_argument("--rank-offer-scale", type=float, default=10.0,
                   help="delta mode: incoming-offer pairs get scale x their target (their heuristic gap is a card "
                        "or two, ~0.02 logits; the search only needs its sign)")
    p.add_argument("--rank-margin", type=float, default=0.5, help="hinge mode: max logit margin of the ranking term")
    p.add_argument("--rank-margin-scale", type=float, default=0.5,
                   help="per-pair margin = scale x heuristic logit gap, clipped to [0.05, --rank-margin] "
                        "(0 = fixed --rank-margin for every pair)")
    p.add_argument("--rank-gap", type=float, default=None,
                   help="minimum heuristic win-probability gap between two siblings to form a pair "
                        "(default 0.01 in hinge mode, 0 in delta mode)")
    p.add_argument("--rank-pairs", type=int, default=8, help="pairs per decision node")
    p.add_argument("--rank-batch", type=int, default=512, help="pairs per mini-batch step")
    p.add_argument("--rank-buffer", type=int, default=400000, help="max sibling rows kept in the replay buffer")
    p.add_argument("--rank-games", type=int, default=300,
                   help="--fit-only on a buffer without siblings: heuristic games to sample siblings from")
    p.add_argument("--siblings", default=None, metavar="PATH",
                   help="sibling afterstates (npz with Xs, sn, sh, sk) to use instead of the buffer's / fresh games")
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
