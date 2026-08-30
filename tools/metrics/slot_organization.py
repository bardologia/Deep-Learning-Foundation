"""Diagnostics of how a set-prediction head organises its Gaussian slots.

Measures how often each slot is used, whether the slots specialise by elevation
rank, and how consistently they map onto ground-truth slots, which is how slot
collapse is detected.
"""

from __future__ import annotations

import numpy as np

from tools.loss.param_loss           import ParamMatcher
from tools.metrics.gaussian_matching import GaussianMatcher


class SlotOrganization:
    """Slot usage, elevation-rank and assignment statistics of a prediction."""

    @staticmethod
    def usage_fractions(params_pred: np.ndarray, n_K: int, amp_threshold: float = ParamMatcher.ACTIVE_AMP_THR) -> np.ndarray:
        """Returns the fraction of pixels in which each slot is active.

        Args:
            params_pred   : Predicted parameter maps of shape (n_K * 3, height, width), channels ordered amplitude, mu, sigma per Gaussian.
            n_K           : Number of Gaussian slots.
            amp_threshold : Amplitude above which a slot counts as active.

        Returns:
            Float64 array of shape (n_K,).
        """
        amp = np.stack([params_pred[3 * k] for k in range(n_K)], axis=0).reshape(n_K, -1)
        return (amp >= amp_threshold).mean(axis=1).astype(np.float64)

    @staticmethod
    def usage_entropy(usage_fractions: np.ndarray) -> float:
        """Returns the entropy of the usage share, normalised to one for uniform usage.

        Args:
            usage_fractions : Per-slot usage fractions of shape (n_K,).

        Returns:
            Normalised entropy in [0, 1], zero when the total is non-positive
            or only one slot exists.
        """
        p     = np.asarray(usage_fractions, dtype=np.float64)
        total = float(p.sum())
        if total <= 0.0 or p.size <= 1:
            return 0.0

        q  = p / total
        nz = q[q > 0]
        h  = float(-(nz * np.log(nz)).sum())

        return h / float(np.log(p.size))

    @staticmethod
    def mu_rank_matrix(params_pred: np.ndarray, n_K: int, amp_threshold: float = ParamMatcher.ACTIVE_AMP_THR) -> np.ndarray:
        """Counts how often each slot takes each elevation rank among the active slots.

        Args:
            params_pred   : Predicted parameter maps of shape (n_K * 3, height, width).
            n_K           : Number of Gaussian slots.
            amp_threshold : Amplitude above which a slot counts as active.

        Returns:
            Float64 matrix of shape (n_K, n_K) whose entry (k, r) counts the
            pixels where slot k is active and holds elevation rank r, ranks
            ordered by ascending mu with inactive slots pushed to the end.
        """
        amp    = np.stack([params_pred[3 * k]     for k in range(n_K)], axis=0).reshape(n_K, -1)
        mu     = np.stack([params_pred[3 * k + 1] for k in range(n_K)], axis=0).reshape(n_K, -1)
        active = amp >= amp_threshold

        mu_masked = np.where(active, mu, np.inf)
        order     = np.argsort(mu_masked, axis=0, kind="stable")

        P    = mu.shape[1]
        rank = np.empty((n_K, P), dtype=np.int64)
        np.put_along_axis(rank, order, np.broadcast_to(np.arange(n_K)[:, None], (n_K, P)), axis=0)

        counts = np.zeros((n_K, n_K), dtype=np.float64)
        for k in range(n_K):
            a = active[k]
            if a.any():
                counts[k] += np.bincount(rank[k, a], minlength=n_K)

        return counts

    @staticmethod
    def assignment_matrix(params_pred: np.ndarray, params_gt: np.ndarray, n_K: int, amp_threshold: float = ParamMatcher.ACTIVE_AMP_THR, assignment: np.ndarray | None = None) -> np.ndarray:
        """Counts how often each predicted slot is matched to each ground-truth slot.

        Args:
            params_pred   : Predicted parameter maps of shape (n_K * 3, height, width).
            params_gt     : Ground-truth parameter maps of the same shape.
            n_K           : Number of Gaussian slots.
            amp_threshold : Amplitude above which a slot counts as active.
            assignment    : Precomputed assignment of shape (pixels, n_K); computed on demand when omitted.

        Returns:
            Float64 matrix of shape (n_K, n_K) whose entry (k, j) counts the
            pixels where predicted slot k matched active ground-truth slot j.
        """
        sel = GaussianMatcher(amp_threshold=amp_threshold).assignment(params_pred, params_gt, n_K) if assignment is None else assignment

        amp_pred = np.stack([params_pred[3 * k] for k in range(n_K)], axis=0).reshape(n_K, -1)
        amp_gt   = np.stack([params_gt  [3 * k] for k in range(n_K)], axis=0).reshape(n_K, -1)
        act_pred = amp_pred >= amp_threshold
        act_gt   = amp_gt   >= amp_threshold

        P      = amp_pred.shape[1]
        rows   = np.arange(P)
        counts = np.zeros((n_K, n_K), dtype=np.float64)
        for k in range(n_K):
            j       = sel[:, k]
            matched = act_pred[k] & act_gt[j, rows]
            if matched.any():
                counts[k] += np.bincount(j[matched], minlength=n_K)

        return counts

    @staticmethod
    def diagonality(counts: np.ndarray) -> float:
        """Returns the share of counts on the diagonal, or NaN for an all-zero matrix."""
        total = float(counts.sum())
        return float(np.trace(counts) / total) if total > 0.0 else float("nan")

    @staticmethod
    def row_normalized(counts: np.ndarray) -> np.ndarray:
        """Returns the count matrix with each row scaled to sum to one, all-zero rows left at zero."""
        row_sum = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
