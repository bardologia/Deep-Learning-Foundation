"""Deterministic subsampling of dataset indices."""
from __future__ import annotations

import numpy as np


class Sampler:
    """Draws reproducible index subsets from a dataset."""
    @staticmethod
    def deterministic_indices(n_total: int, max_samples: int, seed: int = 42) -> np.ndarray:
        """Returns a sorted, reproducible subset of indices.

        Args:
            n_total     : Number of available indices.
            max_samples : Largest subset size; values of 0 or less keep everything.
            seed        : Seed of the drawing generator.

        Returns:
            Ascending indices of shape (min(n_total, max_samples),), or all indices
            when no subsampling is needed.
        """
        n_use = min(n_total, max_samples) if max_samples > 0 else n_total

        if n_use < n_total:
            rng     = np.random.default_rng(seed)
            indices = rng.choice(n_total, size=n_use, replace=False)
            indices.sort()
            return indices

        return np.arange(n_total, dtype=np.int64)
