from __future__ import annotations

import numpy as np

from tools.loss.param_loss           import ParamMatcher
from tools.metrics.gaussian_matching import GaussianMatcher


class SlotOrganization:
    @staticmethod
    def usage_fractions(params_pred: np.ndarray, n_K: int, amp_threshold: float = ParamMatcher.ACTIVE_AMP_THR) -> np.ndarray:
        amp = np.stack([params_pred[3 * k] for k in range(n_K)], axis=0).reshape(n_K, -1)
        return (amp >= amp_threshold).mean(axis=1).astype(np.float64)

    @staticmethod
    def usage_entropy(usage_fractions: np.ndarray) -> float:
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
        total = float(counts.sum())
        return float(np.trace(counts) / total) if total > 0.0 else float("nan")

    @staticmethod
    def row_normalized(counts: np.ndarray) -> np.ndarray:
        row_sum = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
