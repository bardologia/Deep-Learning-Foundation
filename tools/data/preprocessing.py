from __future__ import annotations

import numpy as np


class ProfilePreprocessor:
    @staticmethod
    def apply(profile: np.ndarray, threshold_factor: float, truncation_index: int) -> np.ndarray:
        if truncation_index <= 0:
            raise ValueError(f"truncation_index must be >= 1, got {truncation_index}; it names the first zeroed z bin, so 0 blanks the whole profile and a negative value counts from the end (config field fit_truncation_index)")

        if not 0.0 <= threshold_factor < 1.0:
            raise ValueError(f"threshold_factor must lie in [0.0, 1.0), got {threshold_factor}; it is a fraction of the per-profile maximum, so 1.0 or more zeroes every sample (config field fit_threshold_factor)")

        out    = np.asarray(profile)
        copied = False

        if threshold_factor > 0.0:
            out    = np.where(out > out.max(axis=0, keepdims=True) * threshold_factor, out, 0.0)
            copied = True

        if truncation_index < out.shape[0]:
            if not copied:
                out = out.copy()
            out[truncation_index:] = 0.0

        return out


class ProfileNormalizer:
    @staticmethod
    def unit_area(cube: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
        cube  = np.asarray(cube, dtype=np.float32)
        total = cube.sum(axis=axis, keepdims=True)

        return (cube / np.clip(total, eps, None)).astype(np.float32)
