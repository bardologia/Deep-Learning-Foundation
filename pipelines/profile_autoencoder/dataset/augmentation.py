from __future__ import annotations

import numpy as np

from configuration.dataset import ProfileAugmentationConfig


class ProfileAugmenter:
    def __init__(self, config: ProfileAugmentationConfig, logger, seed: int = 0):
        self.config = config
        self.logger = logger
        self.seed   = int(seed)
        self._rng   = np.random.default_rng(self.seed)

        self.logger.section("[Profile Augmentation]")
        self.logger.kv_table(
            {
                "Amplitude Scale" : f"range={self.config.amp_scale_range} p={self.config.p_amp_scale}",
                "Shift"           : f"max={self.config.max_shift} bins p={self.config.p_shift}",
                "Flip"            : self.config.p_flip,
                "Noise"           : f"std={self.config.noise_std} (normalized units) p={self.config.p_noise}",
            },
            title="Augmentation Config",
        )

    def _scale_amplitude(self, curve: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.config.p_amp_scale:
            return curve

        low, high = self.config.amp_scale_range
        factor    = self._rng.uniform(low, high)

        return curve * np.float32(factor)

    def _shift(self, curve: np.ndarray) -> np.ndarray:
        if self.config.max_shift <= 0 or self._rng.random() >= self.config.p_shift:
            return curve

        k = int(self._rng.integers(-self.config.max_shift, self.config.max_shift + 1))

        if k == 0:
            return curve

        return np.roll(curve, k)

    def _flip(self, curve: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.config.p_flip:
            return curve

        return curve[::-1]

    def add_noise(self, normalized_curve: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.config.p_noise:
            return normalized_curve

        noise = self._rng.normal(0.0, self.config.noise_std, normalized_curve.shape).astype(normalized_curve.dtype)

        return normalized_curve + noise

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, curve: np.ndarray) -> np.ndarray:
        curve = self._scale_amplitude(curve)
        curve = self._shift(curve)
        curve = self._flip(curve)

        return np.ascontiguousarray(curve, dtype=np.float32)
