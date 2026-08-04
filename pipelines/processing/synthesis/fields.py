from __future__ import annotations

import numpy as np

from scipy.ndimage import gaussian_filter


class SmoothFieldSampler:
    def __init__(self, shape: tuple, smoothing_pixels: float, seed: int) -> None:
        self.shape            = tuple(int(value) for value in shape)
        self.smoothing_pixels = float(smoothing_pixels)
        self.seed             = int(seed)

    def _stream(self, stream: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, int(stream)])

    def unit(self, stream: int) -> np.ndarray:
        raw      = self._stream(stream).standard_normal(self.shape).astype(np.float32)
        smoothed = gaussian_filter(raw, sigma=self.smoothing_pixels, mode="wrap")

        low  = float(smoothed.min())
        high = float(smoothed.max())

        if high - low < 1e-9:
            raise ValueError(f"Smoothing a {self.shape} noise field with sigma={self.smoothing_pixels} collapsed it to a constant (range {high - low:.3e}); reduce field_smoothing_pixels or enlarge the scene.")

        return ((smoothed - low) / (high - low)).astype(np.float32)

    def bounded(self, stream: int, low: float, high: float) -> np.ndarray:
        return (float(low) + (float(high) - float(low)) * self.unit(stream)).astype(np.float32)

    def angle(self, stream: int) -> np.ndarray:
        return ((2.0 * np.pi) * self.unit(stream) - np.pi).astype(np.float32)

    def below_fraction(self, stream: int, fraction: float) -> np.ndarray:
        field = self.unit(stream)

        if fraction <= 0.0:
            return np.zeros(self.shape, dtype=bool)

        if fraction >= 1.0:
            return np.ones(self.shape, dtype=bool)

        return field < float(np.quantile(field, float(fraction)))


class WhiteNoiseSampler:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def _stream(self, stream: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, int(stream)])

    def real(self, stream: int, shape: tuple, scale) -> np.ndarray:
        return (self._stream(stream).standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

    def complex(self, stream: int, shape: tuple, scale) -> np.ndarray:
        generator = self._stream(stream)
        real      = generator.standard_normal(shape).astype(np.float32)
        imaginary = generator.standard_normal(shape).astype(np.float32)

        return ((real + 1j * imaginary).astype(np.complex64) * (scale / np.sqrt(2.0))).astype(np.complex64)

    def uniform_phase(self, stream: int, shape: tuple) -> np.ndarray:
        return self._stream(stream).uniform(-np.pi, np.pi, size=shape).astype(np.float32)
