from __future__ import annotations

from itertools import permutations

import numpy as np

from tools.loss.param_loss import ParamMatcher


class GaussianMatcher:
    def __init__(self, ppg: int = 3, amp_threshold: float = ParamMatcher.ACTIVE_AMP_THR, big: float = 1e7, chunk_size: int = 250_000) -> None:
        self.ppg           = ppg
        self.amp_threshold = amp_threshold
        self.big           = big
        self.chunk_size    = chunk_size

    def _channel(self, params: np.ndarray, n_K: int, offset: int) -> np.ndarray:
        return np.stack([params[self.ppg * k + offset] for k in range(n_K)], axis=0).reshape(n_K, -1)

    @staticmethod
    def _best_perm(cost: np.ndarray, perms: np.ndarray, n_K: int) -> np.ndarray:
        idx  = np.arange(n_K)
        best = np.zeros(cost.shape[0], dtype=np.int64)
        low  = cost[:, idx, perms[0]].sum(axis=1)

        for position, perm in enumerate(perms[1:], start=1):
            candidate = cost[:, idx, perm].sum(axis=1)
            improved  = candidate < low

            low  = np.where(improved, candidate, low)
            best = np.where(improved, position, best)

        return best

    def assignment(self, params_pred: np.ndarray, params_gt: np.ndarray, n_K: int) -> np.ndarray:
        if n_K > ParamMatcher.MAX_GAUSSIANS:
            raise ValueError(f"GaussianMatcher.assignment enumerates n_K! permutations; n_K={n_K} exceeds MAX_GAUSSIANS={ParamMatcher.MAX_GAUSSIANS}")

        mu_pred  = self._channel(params_pred, n_K, 1)
        mu_gt    = self._channel(params_gt,   n_K, 1)
        act_pred = self._channel(params_pred, n_K, 0) >= self.amp_threshold
        act_gt   = self._channel(params_gt,   n_K, 0) >= self.amp_threshold

        perms    = np.asarray(list(permutations(range(n_K))))
        n_pixels = mu_pred.shape[1]
        sel      = np.empty((n_pixels, n_K), dtype=np.int64)

        for start in range(0, n_pixels, self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            mp = mu_pred [:, sl].T
            mg = mu_gt   [:, sl].T
            ap = act_pred[:, sl].T
            ag = act_gt  [:, sl].T

            base = np.nan_to_num(np.abs(mp[:, :, None] - mg[:, None, :]), nan=self.big, posinf=self.big)
            pair = ap[:, :, None] & ag[:, None, :]
            cost = np.where(pair, base, self.big)

            sel[sl] = perms[self._best_perm(cost, perms, n_K)]

        return sel

    def aligned_prediction(self, params_pred: np.ndarray, params_gt: np.ndarray, n_K: int, assignment: np.ndarray | None = None) -> np.ndarray:
        ppg  = self.ppg
        H, W = params_pred.shape[-2:]
        sel  = self.assignment(params_pred, params_gt, n_K) if assignment is None else assignment

        act_pred = self._channel(params_pred, n_K, 0) >= self.amp_threshold
        n_pixels = sel.shape[0]
        rows     = np.arange(n_pixels)
        aligned  = np.full((n_K, ppg, n_pixels), np.nan, dtype=np.float32)

        for i in range(n_K):
            active = act_pred[i]
            if not active.any():
                continue

            target = sel[active, i]
            block  = np.stack([params_pred[ppg * i + c].reshape(-1)[active] for c in range(ppg)], axis=1)
            aligned[target, :, rows[active]] = block

        return aligned.reshape(n_K * ppg, H, W)
