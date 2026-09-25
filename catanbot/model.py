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
  with early stopping (patience) restoring the best weights.

Weights are saved with ``np.savez`` and loaded with :meth:`ValueNet.load`.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import NUM_FEATURES, extract_batch

__all__ = ["ValueNet", "binary_auc", "MODEL_VERSION"]

MODEL_VERSION = 1


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
                 dtype=np.float32):
        self.n_in = int(n_in)
        self.hidden: Tuple[int, ...] = tuple(int(h) for h in hidden)
        self.dtype = np.dtype(dtype)
        self.seed = int(seed)
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
    # forward / backward
    # ------------------------------------------------------------------
    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_in:
            raise ValueError(f"expected {self.n_in} features, got {X.shape[1]}")
        return (X - self.mean) / self.std

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
        """Evaluator interface for the search: ``extract_batch`` + :meth:`predict`."""
        if len(states) == 0:
            return np.zeros(0, np.float32)
        return self.predict(extract_batch(states, players))

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray, weight_decay: float = 0.0
                       ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Mean BCE (+ L2 penalty) and its gradients w.r.t. ``W`` and ``b``.

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
        gW: List[Optional[np.ndarray]] = [None] * len(self.W)
        gb: List[Optional[np.ndarray]] = [None] * len(self.b)
        delta = dz
        for i in range(len(self.W) - 1, -1, -1):
            a = acts[i]
            gW[i] = a.T @ delta
            gb[i] = delta.sum(axis=0)
            if weight_decay:
                gW[i] = gW[i] + weight_decay * self.W[i]
            if i > 0:
                delta = (delta @ self.W[i].T) * (a > 0)
        return loss, gW, gb  # type: ignore[return-value]

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
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0).astype(self.dtype)
        std = X.std(axis=0)
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
            beta1: float = 0.9, beta2: float = 0.999, adam_eps: float = 1e-8) -> Dict[str, object]:
        """Train with mini-batch Adam.

        ``X`` ``(N, n_in)`` raw features (any float dtype), ``y`` ``(N,)``
        targets in ``[0, 1]``.  Input standardisation statistics are computed
        from ``X`` on the first call (or when ``refit_norm``).  With a
        validation set, training stops after ``patience`` epochs without
        improvement of ``val_loss`` and the best weights are restored
        (``patience <= 0`` disables early stopping).  ``log`` receives one
        line per epoch.

        Returns a history dict with lists ``train_loss``, ``val_loss``,
        ``val_auc``, ``val_acc`` (validation lists empty without a validation
        set) plus ``epochs`` (run), ``best_epoch`` and ``stopped_early``.
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
        history: Dict[str, object] = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": [],
                                      "epochs": 0, "best_epoch": -1, "stopped_early": False}
        best_val = float("inf")
        best_params: Optional[np.ndarray] = None
        bad = 0
        idx = np.arange(n)
        for ep in range(int(epochs)):
            if shuffle:
                rng.shuffle(idx)
            total = 0.0
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                Hb = Xn[bi]
                yb = y[bi]
                z, acts = self._forward(Hb, keep=True)
                total += _bce_from_logits(z, yb) * len(bi)
                dz = ((_sigmoid(z) - yb) / len(bi)).astype(self.dtype)[:, None]
                grads_W: List[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
                grads_b: List[np.ndarray] = [None] * len(self.b)  # type: ignore[list-item]
                delta = dz
                for i in range(len(self.W) - 1, -1, -1):
                    a = acts[i]
                    gW = a.T @ delta
                    if weight_decay:
                        gW += weight_decay * self.W[i]
                    grads_W[i] = gW
                    grads_b[i] = delta.sum(axis=0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * (a > 0)
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
                if vl < best_val - 1e-7:
                    best_val = vl
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
            "version": np.asarray(MODEL_VERSION, np.int64),
            "n_in": np.asarray(self.n_in, np.int64),
            "hidden": np.asarray(self.hidden, np.int64),
            "n_layers": np.asarray(len(self.W), np.int64),
            "mean": self.mean,
            "std": self.std,
            "norm_fitted": np.asarray(1 if self.norm_fitted else 0, np.int64),
            "seed": np.asarray(self.seed, np.int64),
        }
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
        return net

    def __repr__(self) -> str:
        return f"ValueNet(n_in={self.n_in}, hidden={self.hidden}, dtype={self.dtype.name})"
