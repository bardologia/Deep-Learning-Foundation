"""Per-pixel matching of predicted Gaussian scatterers to ground-truth ones.

The evaluation-time counterpart of the training matcher: it works on stacked
numpy parameter maps, assigns slots by elevation proximity between active
scatterers, and rewrites the prediction into ground-truth slot order.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from tools.loss.param_loss import ParamMatcher


class GaussianMatcher:
    """Assigns predicted Gaussian slots to ground-truth slots by elevation proximity.

    Attributes:
        ppg           : Number of parameters per Gaussian, ordered amplitude, mu, sigma.
        amp_threshold : Amplitude above which a scatterer counts as active.
        big           : Sentinel cost assigned to pairs where either side is inactive or the elevation difference is not finite.
        chunk_size    : Number of pixels matched per batch, bounding peak memory.
    """

    def __init__(self, ppg: int = 3, amp_threshold: float = ParamMatcher.ACTIVE_AMP_THR, big: float = 1e7, chunk_size: int = 250_000) -> None:
        """Initialises the matcher.

        Args:
            ppg           : Number of parameters per Gaussian.
            amp_threshold : Amplitude above which a scatterer counts as active.
            big           : Sentinel cost for pairs that must not be matched.
            chunk_size    : Number of pixels matched per batch.
        """
        self.ppg           = ppg
        self.amp_threshold = amp_threshold
        self.big           = big
        self.chunk_size    = chunk_size

    def _channel(self, params: np.ndarray, n_K: int, offset: int) -> np.ndarray:
        """Returns one parameter channel of every Gaussian, flattened to (n_K, pixels)."""
        return np.stack([params[self.ppg * k + offset] for k in range(n_K)], axis=0).reshape(n_K, -1)

    @staticmethod
    def _best_perm(cost: np.ndarray, perms: np.ndarray, n_K: int) -> np.ndarray:
        """Returns the index of the lowest-cost permutation for each pixel.

        Args:
            cost  : Pairwise costs of shape (pixels, n_K, n_K).
            perms : All n_K! permutations, of shape (n_K!, n_K).
            n_K   : Number of Gaussian slots.

        Returns:
            Integer array of shape (pixels,) indexing into perms.
        """
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
        """Computes the ground-truth slot each predicted slot maps to, per pixel.

        The cost of a pair is the absolute elevation difference between the two
        mu values, replaced by the sentinel cost whenever either side is
        inactive, so inactive slots are only matched to each other.

        Args:
            params_pred : Predicted parameter maps of shape (n_K * ppg, height, width), channels ordered amplitude, mu, sigma per Gaussian.
            params_gt   : Ground-truth parameter maps of the same shape.
            n_K         : Number of Gaussian slots.

        Returns:
            Integer array of shape (pixels, n_K) giving, for each predicted
            slot, the ground-truth slot it was assigned to.

        Raises:
            ValueError : If n_K exceeds ParamMatcher.MAX_GAUSSIANS.
        """
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
        """Rewrites the predicted parameter maps into ground-truth slot order.

        Args:
            params_pred : Predicted parameter maps of shape (n_K * ppg, height, width).
            params_gt   : Ground-truth parameter maps of the same shape.
            n_K         : Number of Gaussian slots.
            assignment  : Precomputed assignment of shape (pixels, n_K); computed on demand when omitted.

        Returns:
            Array of shape (n_K * ppg, height, width) holding each predicted
            scatterer in its assigned ground-truth slot, with NaN wherever no
            active prediction was placed.
        """
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
