from __future__ import annotations

import numpy as np


class ProfilePreprocessor:
    @staticmethod
    def apply(profile: np.ndarray, threshold_factor: float, truncation_index: int) -> np.ndarray:
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
