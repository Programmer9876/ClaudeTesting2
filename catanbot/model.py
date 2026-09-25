"""Pure-numpy MLP value network.

``ValueNet`` maps a feature vector (see :mod:`catanbot.features`) to a win
probability in ``[0, 1]``.  It is a plain multilayer perceptron:

* input standardisation (mean / std computed in :meth:`ValueNet.fit`, stored
  with the weights so raw features can be fed directly),
* ReLU hidden layers (default ``(256, 128)``), He initialisation,
* a single logit output with a sigmoid,
* binary cross-entropy loss (computed from the logit for stability),
* Adam with classic L2 weight decay (on weights, not biases), global-norm
  gradient clipping, mini-batches with shuffling, optional validation set
  with early stopping (patience) restoring the best weights,
* an optional **input mask** (:func:`feature_mask`): features the net must
  never see are zeroed after standardisation, at training and at inference
  alike, and the mask is stored with the weights.  The self-play labels are
  Monte-Carlo outcomes under a policy that spends every affordable build at
  once, so the net learns that cards in hand are worth the builds they will
  become and then, used as a search evaluator, ranks "hold the cards" level
  with "build now".  Masking the own-hand features (resources, hand size,
  can-afford flags) removes that confound; the heuristic side of the blend
  (``selfplay.BlendedEvaluator``) still sees affordability and the 7-risk.

* an optional **horizon-consistency term** (``fit(consistency=(X_a, X_b))``):
  ``(z_a - z_b)^2`` on the logits of the same sibling at its mid-turn horizon
  and after END_TURN.  The search compares lines its beam pruned (valued by a
  mid-turn static) with fully expanded lines (valued after END_TURN), which
  only works when the net gives both horizons the same value, as the
  heuristic does by construction.
* an optional **pairwise ranking term** (``fit(pairs=(X_pos, X_neg))``):
  ``softplus(margin - (z_pos - z_neg))`` on the logits of two sibling
  afterstates of the same decision, added to the BCE.  The search ranks
  siblings, and the self-play labels hold no counterfactual for the moves the
  behaviour policy never makes (holding an affordable build), so the net is
  told directly which sibling is better (the heuristic's ordering of concrete
  afterstates, see ``selfplay.play_game(sibling_rate=...)``) while the BCE keeps
  its absolute values calibrated.  The pairs cover every decision the search
  bot makes with the net - main-phase actions, setup placements, incoming
  offers, robber moves and discards - because the outcome labels resolve
  none of them: an offer's accept / reject afterstates differ by a card.

:meth:`ValueNet.evaluate` returns the exact value (1 / 0) for finished games,
like the heuristic evaluator, because game-over states are never recorded.

Weights are saved with ``np.savez`` and loaded with :meth:`ValueNet.load`.
"""
from __future__ import annotations

import fnmatch
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .features import FEATURE_NAMES, NUM_FEATURES, extract_batch
from .state import PHASE_GAME_OVER

__all__ = ["ValueNet", "binary_auc", "feature_mask", "pair_rank_loss", "HAND_BLIND_FEATURES", "MODEL_VERSION",
           "PAIR_MODES"]

# Files written by this module: version 1 has no input mask (readable by older code), version 2
# carries one (older loaders refuse it instead of silently using the net without its mask).
MODEL_VERSION = 2

# Own-hand features (see the module docstring): what a net trained on self-play outcomes must not
# see if it is to rank building now above holding the cards.  Glob patterns over FEATURE_NAMES.
HAND_BLIND_FEATURES = ("me_res_*", "me_hand_size", "me_cards_over_7", "me_discard_exposure",
                       "me_can_build_*", "me_can_buy_dev")


def feature_mask(patterns: Union[str, Iterable[str], None], names: Sequence[str] = FEATURE_NAMES) -> Optional[np.ndarray]:
    """Input mask (float32 ``(len(names),)``, 1 = keep, 0 = hide) from glob patterns over feature names.

    ``patterns`` is a comma-separated string or an iterable of glob patterns
    (``"me_res_*"``); every pattern must match at least one name (typos are
    errors).  ``None`` / an empty string / no patterns gives ``None`` (no mask).
    """
    if patterns is None:
        return None
    if isinstance(patterns, str):
        patterns = [p for p in patterns.split(",")]
    pats = [p.strip() for p in patterns if p and p.strip()]
    if not pats:
        return None
    mask = np.ones(len(names), np.float32)
    for pat in pats:
        hits = [i for i, n in enumerate(names) if fnmatch.fnmatchcase(n, pat)]
        if not hits:
            raise ValueError(f"feature mask pattern {pat!r} matches no feature name")
        mask[hits] = 0.0
    return mask


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _bce_from_logits(z: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy of logits ``z`` against targets ``y`` in [0, 1]."""
    # max(z, 0) - z*y + log(1 + exp(-|z|))
    loss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(loss.mean())


def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)), numerically stable."""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


PAIR_MODES = ("hinge", "delta")


def pair_rank_loss(z_pos: np.ndarray, z_neg: np.ndarray, margin, weight=None,
                   mode: str = "hinge", sign_margin: float = 0.0, sign_weight: float = 1.0
                   ) -> Tuple[float, np.ndarray]:
    """Pairwise loss on the logit difference ``z_pos - z_neg`` of two siblings and ``d loss / d z_pos``
    (``d loss / d z_neg`` is its negative).

    ``mode="hinge"``: ``mean(weight * softplus(margin - (z_pos - z_neg)))``, ordering only - the
    difference is pushed past ``margin`` and, the softplus never being zero, ever further.
    ``mode="delta"``: ``mean(weight * ((z_pos - z_neg) - margin) ** 2)``, i.e. the difference is
    regressed onto ``margin`` as its target: ordering *and* magnitude.  With ``sign_margin > 0`` the
    delta mode adds ``sign_weight * mean(weight * relu(sign_margin - (z_pos - z_neg)))`` over the pairs
    whose target is positive: a bounded hinge (zero once the sign is right by ``sign_margin``) that pushes
    with constant force where the squared error, proportional to a target of a few hundredths, would not.
    ``margin`` is a scalar or one value per pair, ``weight`` (one value per pair, default 1) scales each
    pair's term.
    """
    if mode not in PAIR_MODES:
        raise ValueError(f"unknown pair loss mode {mode!r}")
    diff = np.asarray(z_pos, np.float64) - np.asarray(z_neg, np.float64)
    m = np.asarray(margin, np.float64)
    w = np.ones_like(diff) if weight is None else np.asarray(weight, np.float64)
    n = max(1, len(diff))
    if mode == "delta":
        d = diff - m
        loss = float((w * d * d).mean())
        g = 2.0 * w * d / n
        if sign_margin > 0:
            viol = (np.broadcast_to(m, diff.shape) > 0) & (diff < sign_margin)
            loss += float(sign_weight * (w * np.where(viol, sign_margin - diff, 0.0)).mean())
            g = g - sign_weight * w * viol / n
        return loss, g
    d = m - diff
    return float((w * _softplus(d)).mean()), -w * _sigmoid(d) / n


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (rank statistic, ties get average rank).

    Returns 0.5 when only one class is present.
    """
    y = np.asarray(y_true).astype(np.float64).ravel()
    s = np.asarray(scores).astype(np.float64).ravel()
    n_pos = float((y > 0.5).sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), np.float64)
    # average ranks for ties
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, counts)
    sum_pos = ranks[y > 0.5].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class ValueNet:
    """Numpy MLP with sigmoid output; see the module docstring."""

    def __init__(self, n_in: int = NUM_FEATURES, hidden: Sequence[int] = (256, 128), seed: int = 0,
                 dtype=np.float32, input_mask: Optional[np.ndarray] = None):
        self.n_in = int(n_in)
        self.hidden: Tuple[int, ...] = tuple(int(h) for h in hidden)
        self.dtype = np.dtype(dtype)
        self.seed = int(seed)
        self._input_mask: Optional[np.ndarray] = None
        self.input_mask = input_mask
        rng = np.random.default_rng(seed)
        sizes = (self.n_in, *self.hidden, 1)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            std = math.sqrt(2.0 / fan_in)  # He init
            self.W.append((rng.standard_normal((fan_in, fan_out)) * std).astype(self.dtype))
            self.b.append(np.zeros(fan_out, self.dtype))
        self.mean = np.zeros(self.n_in, self.dtype)
        self.std = np.ones(self.n_in, self.dtype)
        self.norm_fitted = False
        # Adam state (created lazily in fit)
        self._m: Optional[List[np.ndarray]] = None
        self._v: Optional[List[np.ndarray]] = None
        self._t = 0

    # ------------------------------------------------------------------
    # input mask
    # ------------------------------------------------------------------
    @property
    def input_mask(self) -> Optional[np.ndarray]:
        """``(n_in,)`` 0/1 vector of the features the net uses (``None`` = all), see :func:`feature_mask`."""
        return self._input_mask

    @input_mask.setter
    def input_mask(self, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            self._input_mask = None
            return
        m = np.asarray(mask, dtype=self.dtype).ravel()
        if m.shape != (self.n_in,):
            raise ValueError(f"input mask must have shape ({self.n_in},), got {m.shape}")
        self._input_mask = None if bool(np.all(m == 1.0)) else (m != 0.0).astype(self.dtype)

    def masked_features(self, names: Sequence[str] = FEATURE_NAMES) -> List[str]:
        """Names of the hidden (masked) features."""
        if self._input_mask is None or len(names) != self.n_in:
            return []
        return [n for n, m in zip(names, self._input_mask) if m == 0.0]

    # ------------------------------------------------------------------
    # forward / backward
    # ------------------------------------------------------------------
    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_in:
            raise ValueError(f"expected {self.n_in} features, got {X.shape[1]}")
        # One float32 copy standardised in place (the replay buffer is ~1.5M rows: (X - mean) / std
        # would allocate two further full-size temporaries).
        H = np.array(X, dtype=self.dtype, copy=True)
        H -= self.mean
        H /= self.std
        if self._input_mask is not None:
            H *= self._input_mask   # hidden features are exactly 0 whatever the input holds
        return H

    def _forward(self, H: np.ndarray, keep: bool = False):
        """Logits ``(N,)`` for standardised input; with ``keep`` also the activations."""
        acts = [H] if keep else None
        n_layers = len(self.W)
        for i in range(n_layers):
            H = H @ self.W[i] + self.b[i]
            if i < n_layers - 1:
                H = np.maximum(H, 0.0)
                if keep:
                    acts.append(H)  # type: ignore[union-attr]
        z = H[:, 0]
        return (z, acts) if keep else z

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Raw pre-sigmoid outputs ``(N,)`` for raw (unstandardised) features."""
        return self._forward(self._prep(X))

    def predict(self, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """Win probabilities ``(N,)`` in ``[0, 1]`` for raw features ``X`` ``(N, n_in)``."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        n = X.shape[0]
        if n <= chunk:
            return _sigmoid(self.logits(X)).astype(np.float32)
        out = np.empty(n, np.float32)
        for s in range(0, n, chunk):
            out[s:s + chunk] = _sigmoid(self.logits(X[s:s + chunk]))
        return out

    __call__ = predict

    def evaluate(self, states, players) -> np.ndarray:
        """Evaluator interface for the search: ``extract_batch`` + :meth:`predict`.

        Finished games are exact (1 for the winner, 0 for everyone else): the
        self-play buffer holds no game-over states, so the net's own guess for
        a won afterstate (~0.6 on average) would rank a winning build below a
        trade proposal.
        """
        if len(states) == 0:
            return np.zeros(0, np.float32)
        out = self.predict(extract_batch(states, players))
        for k, (s, p) in enumerate(zip(states, players)):
            if s.phase == PHASE_GAME_OVER and s.winner >= 0:
                out[k] = 1.0 if s.winner == p else 0.0
        return out

    def _backprop(self, acts: List[np.ndarray], dz: np.ndarray, grads_W: List[np.ndarray],
                  grads_b: List[np.ndarray]) -> None:
        """Accumulate ``d loss / d W, b`` into ``grads_W`` / ``grads_b`` from ``dz`` ``(N, 1)`` (the loss
        gradient w.r.t. the logits) and the activations of the forward pass that produced them."""
        delta = dz
        for i in range(len(self.W) - 1, -1, -1):
            a = acts[i]
            grads_W[i] += a.T @ delta
            grads_b[i] += delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (a > 0)

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray, weight_decay: float = 0.0,
                       pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None, pair_weight: float = 1.0,
                       pair_margin: float = 0.5, consistency: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       consistency_weight: float = 1.0, pair_mode: str = "hinge", pair_sign_margin: float = 0.0,
                       pair_sign_weight: float = 1.0) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Mean BCE (+ L2 penalty, + the pairwise ranking / horizon-consistency terms of :meth:`fit` with
        ``pairs`` / ``consistency``) and its gradients w.r.t. ``W`` and ``b``.

        Exposed for gradient checking.  ``X`` is raw features (standardised
        internally with the stored mean/std).
        """
        H0 = self._prep(X)
        y = np.asarray(y, dtype=self.dtype).ravel()
        n = H0.shape[0]
        z, acts = self._forward(H0, keep=True)
        loss = _bce_from_logits(z, y)
        if weight_decay:
            loss += 0.5 * weight_decay * sum(float((W * W).sum()) for W in self.W)
        dz = ((_sigmoid(z) - y) / n).astype(self.dtype)[:, None]
        gW: List[np.ndarray] = [np.zeros_like(W) for W in self.W]
        gb: List[np.ndarray] = [np.zeros_like(b) for b in self.b]
        self._backprop(acts, dz, gW, gb)
        if pairs is not None and len(pairs[0]) > 0 and pair_weight > 0:
            zp, acts_p = self._forward(self._prep(np.asarray(pairs[0])), keep=True)
            zn, acts_n = self._forward(self._prep(np.asarray(pairs[1])), keep=True)
            pl, gp = pair_rank_loss(zp, zn, pairs[2] if len(pairs) > 2 and pairs[2] is not None else pair_margin,
                                    pairs[3] if len(pairs) > 3 else None, pair_mode, pair_sign_margin, pair_sign_weight)
            loss += pair_weight * pl
            gp = (gp * pair_weight).astype(self.dtype)[:, None]
            self._backprop(acts_p, gp, gW, gb)
            self._backprop(acts_n, -gp, gW, gb)
        if consistency is not None and len(consistency[0]) > 0 and consistency_weight > 0:
            za, acts_a = self._forward(self._prep(np.asarray(consistency[0])), keep=True)
            zb, acts_b = self._forward(self._prep(np.asarray(consistency[1])), keep=True)
            diff = za.astype(np.float64) - zb.astype(np.float64)
            loss += consistency_weight * float((diff * diff).mean())
            gc = (2.0 * diff / len(diff) * consistency_weight).astype(self.dtype)[:, None]
            self._backprop(acts_a, gc, gW, gb)
            self._backprop(acts_b, -gc, gW, gb)
        if weight_decay:
            for i in range(len(self.W)):
                gW[i] = gW[i] + weight_decay * self.W[i]
        return loss, gW, gb

    # ------------------------------------------------------------------
    # parameters as a flat vector (gradient checks, tests)
    # ------------------------------------------------------------------
    def get_params(self) -> np.ndarray:
        """All weights and biases as one flat vector (layer order W0, b0, W1, b1, ...)."""
        parts = []
        for W, b in zip(self.W, self.b):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_params(self, flat: np.ndarray) -> None:
        """Inverse of :meth:`get_params`."""
        flat = np.asarray(flat, dtype=self.dtype)
        k = 0
        for i in range(len(self.W)):
            n = self.W[i].size
            self.W[i] = flat[k:k + n].reshape(self.W[i].shape).copy()
            k += n
            n = self.b[i].size
            self.b[i] = flat[k:k + n].reshape(self.b[i].shape).copy()
            k += n
        if k != flat.size:
            raise ValueError("parameter vector has the wrong size")

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit_normalisation(self, X: np.ndarray, eps: float = 1e-6) -> None:
        """Compute and store input mean / std from ``X`` (raw features)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        n = X.shape[0]
        # Two-pass mean / std accumulated in float64 over row chunks: no full float64 copy of a
        # float16 replay buffer (which would be 4x its size).
        chunk = 65536
        mean = np.zeros(X.shape[1], np.float64)
        for s in range(0, n, chunk):
            mean += X[s:s + chunk].astype(np.float64).sum(axis=0)
        mean /= max(1, n)
        var = np.zeros(X.shape[1], np.float64)
        for s in range(0, n, chunk):
            d = X[s:s + chunk].astype(np.float64) - mean
            var += (d * d).sum(axis=0)
        std = np.sqrt(var / max(1, n))
        self.mean = mean.astype(self.dtype)
        std[std < eps] = 1.0  # constant features: leave as-is (centred to 0)
        self.std = std.astype(self.dtype)
        self.norm_fitted = True

    def _init_adam(self) -> None:
        self._m = [np.zeros_like(p) for p in (*self.W, *self.b)]
        self._v = [np.zeros_like(p) for p in (*self.W, *self.b)]
        self._t = 0

    def _adam_step(self, grads: List[np.ndarray], lr: float, beta1: float, beta2: float, eps: float) -> None:
        assert self._m is not None and self._v is not None
        self._t += 1
        t = self._t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        params = [*self.W, *self.b]
        for i, (p, g) in enumerate(zip(params, grads)):
            m = self._m[i]
            v = self._v[i]
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)
            p -= (lr * (m / bc1) / (np.sqrt(v / bc2) + eps)).astype(p.dtype)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3,
            weight_decay: float = 1e-4, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            log: Optional[Callable[[str], None]] = None, patience: int = 5, clip_norm: float = 5.0,
            shuffle: bool = True, seed: Optional[int] = None, refit_norm: bool = False,
            beta1: float = 0.9, beta2: float = 0.999, adam_eps: float = 1e-8,
            input_noise: float = 0.0, pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            pair_weight: float = 1.0, pair_margin: float = 0.5, pair_batch: int = 256,
            val_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            consistency: Optional[Tuple[np.ndarray, np.ndarray]] = None, consistency_weight: float = 1.0,
            val_consistency: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            pair_mode: str = "hinge", pair_sign_margin: float = 0.0, pair_sign_weight: float = 1.0
            ) -> Dict[str, object]:
        """Train with mini-batch Adam.

        ``X`` ``(N, n_in)`` raw features (any float dtype), ``y`` ``(N,)``
        targets in ``[0, 1]``.  Input standardisation statistics are computed
        from ``X`` on the first call (or when ``refit_norm``).  With a
        validation set, training stops after ``patience`` epochs without
        improvement of ``val_loss`` and the best weights are restored
        (``patience <= 0`` disables early stopping).  ``log`` receives one
        line per epoch.

        ``pairs = (X_pos, X_neg)`` (raw features, same length) adds the
        ranking term ``pair_weight * mean softplus(pair_margin - (z_pos -
        z_neg))``: every mini-batch step also takes the next ``pair_batch``
        pairs (cycling through a shuffled order).  A third element
        ``(X_pos, X_neg, margins)`` gives every pair its own margin instead
        of ``pair_margin``, a fourth ``(..., margins, weights)`` its own
        weight (the pair's term is multiplied by it; ``None`` = 1 for all).
        ``pair_mode`` selects :func:`pair_rank_loss`'s form: ``"hinge"``
        (ordering only) or ``"delta"`` (the margins are the *target*
        differences: ordering and magnitude, plus the bounded sign hinge
        ``pair_sign_margin`` / ``pair_sign_weight``).  ``consistency = (X_a, X_b)`` adds
        ``consistency_weight * mean (z_a - z_b)^2`` on ``pair_batch`` rows
        per step the same way (``cons_loss`` / ``val_cons_loss`` in the
        history).  ``val_pairs`` are scored
        each epoch (``val_pair_loss``, ``val_pair_acc`` = share ordered
        correctly) and, when given, early stopping uses ``val_loss +
        pair_weight * val_pair_loss``.

        Returns a history dict with lists ``train_loss``, ``val_loss``,
        ``val_auc``, ``val_acc`` (validation lists empty without a validation
        set), ``pair_loss`` / ``val_pair_loss`` / ``val_pair_acc`` (with
        pairs) plus ``epochs`` (run), ``best_epoch`` and ``stopped_early``.
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=self.dtype).ravel()
        n = X.shape[0]
        if n == 0:
            raise ValueError("empty training set")
        if len(y) != n:
            raise ValueError("X and y length mismatch")
        if not self.norm_fitted or refit_norm:
            self.fit_normalisation(X)
        Xn = self._prep(X)
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            Xv = self._prep(np.asarray(X_val))
            yv = np.asarray(y_val, dtype=self.dtype).ravel()
        rng = np.random.default_rng(self.seed if seed is None else seed)
        if self._m is None:
            self._init_adam()
        batch_size = max(1, min(int(batch_size), n))
        has_pairs = pairs is not None and len(pairs[0]) > 0 and pair_weight > 0
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"unknown pair loss mode {pair_mode!r}")
        if has_pairs:
            Pp = self._prep(np.asarray(pairs[0]))
            Pn = self._prep(np.asarray(pairs[1]))
            if len(Pp) != len(Pn):
                raise ValueError("pairs must have the same number of positive and negative rows")
            n_pairs = len(Pp)
            Pm = (np.asarray(pairs[2], np.float64).ravel() if len(pairs) > 2 and pairs[2] is not None
                  else np.full(n_pairs, float(pair_margin)))
            if len(Pm) != n_pairs:
                raise ValueError("one margin per pair expected")
            Pw = (np.asarray(pairs[3], np.float64).ravel() if len(pairs) > 3 and pairs[3] is not None
                  else np.ones(n_pairs))
            if len(Pw) != n_pairs:
                raise ValueError("one weight per pair expected")
            pair_batch = max(1, min(int(pair_batch), n_pairs))
            pidx = np.arange(n_pairs)
            p_pos = 0
        has_val_pairs = val_pairs is not None and len(val_pairs[0]) > 0
        if has_val_pairs:
            Vp = self._prep(np.asarray(val_pairs[0]))
            Vn = self._prep(np.asarray(val_pairs[1]))
            Vm = (np.asarray(val_pairs[2], np.float64).ravel() if len(val_pairs) > 2 and val_pairs[2] is not None
                  else float(pair_margin))
            Vw = (np.asarray(val_pairs[3], np.float64).ravel() if len(val_pairs) > 3 and val_pairs[3] is not None
                  else None)
        has_cons = consistency is not None and len(consistency[0]) > 0 and consistency_weight > 0
        if has_cons:
            Ca = self._prep(np.asarray(consistency[0]))
            Cb = self._prep(np.asarray(consistency[1]))
            if len(Ca) != len(Cb):
                raise ValueError("consistency pairs must have the same number of rows on both sides")
            n_cons = len(Ca)
            cons_batch = max(1, min(int(pair_batch), n_cons))
            cidx = np.arange(n_cons)
            c_pos = 0
        has_val_cons = val_consistency is not None and len(val_consistency[0]) > 0
        if has_val_cons:
            VCa = self._prep(np.asarray(val_consistency[0]))
            VCb = self._prep(np.asarray(val_consistency[1]))
        history: Dict[str, object] = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": [],
                                      "epochs": 0, "best_epoch": -1, "stopped_early": False}
        if has_pairs:
            history["pair_loss"] = []
        if has_val_pairs:
            history["val_pair_loss"] = []
            history["val_pair_acc"] = []
        if has_cons:
            history["cons_loss"] = []
        if has_val_cons:
            history["val_cons_loss"] = []
        best_val = float("inf")
        best_params: Optional[np.ndarray] = None
        bad = 0
        idx = np.arange(n)

        def noisy(H: np.ndarray) -> np.ndarray:
            if input_noise > 0:
                # Gaussian input noise (in standardised units) as a regulariser: positions
                # from the same game are near-duplicates, and without noise the net memorises
                # game identity instead of learning position value.
                H = H + rng.normal(0.0, input_noise, H.shape).astype(self.dtype)
                if self._input_mask is not None:
                    H = H * self._input_mask   # hidden features stay hidden (no noise either)
            return H

        for ep in range(int(epochs)):
            if shuffle:
                rng.shuffle(idx)
                if has_pairs:
                    rng.shuffle(pidx)
                if has_cons:
                    rng.shuffle(cidx)
            total = 0.0
            total_pair = 0.0
            n_pair_steps = 0
            total_cons = 0.0
            n_cons_steps = 0
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                Hb = noisy(Xn[bi])
                yb = y[bi]
                z, acts = self._forward(Hb, keep=True)
                total += _bce_from_logits(z, yb) * len(bi)
                dz = ((_sigmoid(z) - yb) / len(bi)).astype(self.dtype)[:, None]
                grads_W: List[np.ndarray] = [np.zeros_like(W) for W in self.W]
                grads_b: List[np.ndarray] = [np.zeros_like(b) for b in self.b]
                self._backprop(acts, dz, grads_W, grads_b)
                if has_pairs:
                    # next slice of pairs (cycling); the ranking gradient is added to the BCE gradient
                    if p_pos + pair_batch > n_pairs:
                        rng.shuffle(pidx)
                        p_pos = 0
                    pb = pidx[p_pos:p_pos + pair_batch]
                    p_pos += pair_batch
                    zp, acts_p = self._forward(noisy(Pp[pb]), keep=True)
                    zn, acts_n = self._forward(noisy(Pn[pb]), keep=True)
                    pl, gp = pair_rank_loss(zp, zn, Pm[pb], Pw[pb], pair_mode, pair_sign_margin, pair_sign_weight)
                    total_pair += pl
                    n_pair_steps += 1
                    gp = (gp * pair_weight).astype(self.dtype)[:, None]
                    self._backprop(acts_p, gp, grads_W, grads_b)
                    self._backprop(acts_n, -gp, grads_W, grads_b)
                if has_cons:
                    if c_pos + cons_batch > n_cons:
                        rng.shuffle(cidx)
                        c_pos = 0
                    cb = cidx[c_pos:c_pos + cons_batch]
                    c_pos += cons_batch
                    za, acts_a = self._forward(noisy(Ca[cb]), keep=True)
                    zb, acts_b = self._forward(noisy(Cb[cb]), keep=True)
                    diff = za.astype(np.float64) - zb.astype(np.float64)
                    total_cons += float((diff * diff).mean())
                    n_cons_steps += 1
                    gc = (2.0 * diff / len(cb) * consistency_weight).astype(self.dtype)[:, None]
                    self._backprop(acts_a, gc, grads_W, grads_b)
                    self._backprop(acts_b, -gc, grads_W, grads_b)
                if weight_decay:
                    for i in range(len(self.W)):
                        grads_W[i] += weight_decay * self.W[i]
                grads = [*grads_W, *grads_b]
                if clip_norm and clip_norm > 0:
                    gn = math.sqrt(sum(float((g * g).sum()) for g in grads))
                    if gn > clip_norm:
                        sc = clip_norm / (gn + 1e-12)
                        grads = [g * sc for g in grads]
                self._adam_step(grads, lr, beta1, beta2, adam_eps)
            train_loss = total / n
            history["train_loss"].append(train_loss)  # type: ignore[union-attr]
            history["epochs"] = ep + 1
            msg = f"epoch {ep + 1}/{epochs} train_loss={train_loss:.4f}"
            if has_pairs:
                pair_loss = total_pair / max(1, n_pair_steps)
                history["pair_loss"].append(pair_loss)  # type: ignore[union-attr]
                msg += f" pair_loss={pair_loss:.4f}"
            if has_cons:
                cons_loss = total_cons / max(1, n_cons_steps)
                history["cons_loss"].append(cons_loss)  # type: ignore[union-attr]
                msg += f" cons_loss={cons_loss:.4f}"
            if has_val_cons:
                dcv = self._forward(VCa).astype(np.float64) - self._forward(VCb).astype(np.float64)
                vcl = float((dcv * dcv).mean())
                history["val_cons_loss"].append(vcl)  # type: ignore[union-attr]
                msg += f" val_cons_loss={vcl:.4f}"
            vpl = 0.0
            if has_val_pairs:
                dv = self._forward(Vp) - self._forward(Vn)
                vpl, _ = pair_rank_loss(self._forward(Vp), self._forward(Vn), Vm, Vw, pair_mode, pair_sign_margin,
                                        pair_sign_weight)
                # share ordered correctly (in delta mode over the pairs whose target says pos > neg)
                strict = np.asarray(Vm, np.float64) > 0 if pair_mode == "delta" else np.ones(len(dv), bool)
                vpa = float((dv[strict] > 0).mean()) if strict.any() else 1.0
                history["val_pair_loss"].append(vpl)  # type: ignore[union-attr]
                history["val_pair_acc"].append(vpa)  # type: ignore[union-attr]
                msg += f" val_pair_loss={vpl:.4f} val_pair_acc={vpa:.3f}"
            if has_val:
                zv = self._forward(Xv)
                vl = _bce_from_logits(zv, yv)
                pv = _sigmoid(zv)
                acc = float(((pv >= 0.5) == (yv >= 0.5)).mean())
                auc = binary_auc(yv, pv)
                history["val_loss"].append(vl)  # type: ignore[union-attr]
                history["val_acc"].append(acc)  # type: ignore[union-attr]
                history["val_auc"].append(auc)  # type: ignore[union-attr]
                msg += f" val_loss={vl:.4f} val_acc={acc:.3f} val_auc={auc:.3f}"
                objective = vl + (pair_weight * vpl if (has_pairs and has_val_pairs) else 0.0)
                if objective < best_val - 1e-7:
                    best_val = objective
                    best_params = self.get_params()
                    history["best_epoch"] = ep
                    bad = 0
                else:
                    bad += 1
                    if patience > 0 and bad >= patience:
                        history["stopped_early"] = True
                        if log:
                            log(msg + " (early stop)")
                        break
            else:
                history["best_epoch"] = ep
            if log:
                log(msg)
        if has_val and best_params is not None:
            self.set_params(best_params)
        return history

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Write weights, normalisation and layer sizes to an ``.npz`` file."""
        arrays = {
            # nets without a mask stay readable by code that predates the mask (version 1)
            "version": np.asarray(MODEL_VERSION if self._input_mask is not None else 1, np.int64),
            "n_in": np.asarray(self.n_in, np.int64),
            "hidden": np.asarray(self.hidden, np.int64),
            "n_layers": np.asarray(len(self.W), np.int64),
            "mean": self.mean,
            "std": self.std,
            "norm_fitted": np.asarray(1 if self.norm_fitted else 0, np.int64),
            "seed": np.asarray(self.seed, np.int64),
        }
        if self._input_mask is not None:
            arrays["input_mask"] = self._input_mask
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @staticmethod
    def load(path: str, dtype=np.float32) -> "ValueNet":
        """Load a net saved by :meth:`save`."""
        with np.load(path) as z:
            version = int(z["version"])
            if version > MODEL_VERSION:
                raise ValueError(f"model version {version} is newer than supported {MODEL_VERSION}")
            n_in = int(z["n_in"])
            hidden = tuple(int(h) for h in z["hidden"])
            net = ValueNet(n_in=n_in, hidden=hidden, seed=int(z["seed"]) if "seed" in z else 0, dtype=dtype)
            n_layers = int(z["n_layers"])
            net.W = [z[f"W{i}"].astype(net.dtype) for i in range(n_layers)]
            net.b = [z[f"b{i}"].astype(net.dtype) for i in range(n_layers)]
            net.mean = z["mean"].astype(net.dtype)
            net.std = z["std"].astype(net.dtype)
            net.norm_fitted = bool(int(z["norm_fitted"])) if "norm_fitted" in z else True
            if "input_mask" in z:
                net.input_mask = z["input_mask"]
        return net

    def __repr__(self) -> str:
        masked = f", masked={int((self._input_mask == 0).sum())}" if self._input_mask is not None else ""
        return f"ValueNet(n_in={self.n_in}, hidden={self.hidden}, dtype={self.dtype.name}{masked})"
