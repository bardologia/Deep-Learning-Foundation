from __future__ import annotations

import numpy as np


class Sampler:
    @staticmethod
    def deterministic_indices(n_total: int, max_samples: int, seed: int = 42) -> np.ndarray:
        n_use = min(n_total, max_samples) if max_samples > 0 else n_total

        if n_use < n_total:
            rng     = np.random.default_rng(seed)
            indices = rng.choice(n_total, size=n_use, replace=False)
            indices.sort()
            return indices

        return np.arange(n_total, dtype=np.int64)
