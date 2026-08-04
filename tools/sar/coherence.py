from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from scipy.ndimage import uniform_filter


class ComplexSmoother:
    def __init__(self, window: Tuple[int, int]) -> None:
        self.window = tuple(int(w) for w in window)

    def smooth(self, array: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(array):
            return uniform_filter(array.real, self.window) + 1j * uniform_filter(array.imag, self.window)
        return uniform_filter(array, self.window)


class CoherenceEstimator:
    def __init__(self, window: Tuple[int, int] = (7, 7)) -> None:
        self.smoother = ComplexSmoother(window)

    def _complex_coherence(self, primary: np.ndarray, secondary: np.ndarray, flattening_phase: Optional[np.ndarray]) -> np.ndarray:
        cross = primary * np.conj(secondary)
        if flattening_phase is not None:
            cross = cross * np.exp(-1j * flattening_phase)

        numerator   = self.smoother.smooth(cross)
        denominator = np.sqrt(self.smoother.smooth(np.abs(primary) ** 2) * self.smoother.smooth(np.abs(secondary) ** 2))

        with np.errstate(all="ignore"):
            coherence = numerator / denominator

        coherence[np.abs(primary * secondary) == 0] = 0
        return coherence

    def estimate(self, primary: np.ndarray, secondary: np.ndarray, flattening_phase: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        coherence = self._complex_coherence(primary, secondary, flattening_phase)

        magnitude = np.clip(np.nan_to_num(np.abs(coherence)), 0.0, 1.0)
        phase     = np.nan_to_num(np.angle(coherence))

        return magnitude, phase

    def estimate_flattened(self, primary_amplitude: np.ndarray, interferogram: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.estimate(primary_amplitude, np.conj(interferogram))
