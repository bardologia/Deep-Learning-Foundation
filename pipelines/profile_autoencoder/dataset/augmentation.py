"""Random augmentation of one-dimensional elevation profiles.

Applies amplitude scaling, integer height-bin shifts and flips to raw curves,
and additive Gaussian noise to already-normalised curves.
"""

from __future__ import annotations

import numpy as np

from configuration.dataset import ProfileAugmentationConfig


class ProfileAugmenter:
    """Augments profile curves with scale, shift, flip and noise transforms.

    Attributes:
        config : Probabilities and magnitudes of each transform.
        seed   : Seed of the internal numpy generator.
    """

    def __init__(self, config: ProfileAugmentationConfig, logger, seed: int = 0):
        """Seeds the generator and reports the augmentation settings.

        Args:
            config : Augmentation probabilities and magnitudes.
            logger : Logger used for the configuration table.
            seed   : Seed of the internal numpy generator.
        """
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
        """Multiplies a curve of shape (H,) by a random factor, or returns it unchanged."""
        if self._rng.random() >= self.config.p_amp_scale:
            return curve

        low, high = self.config.amp_scale_range
        factor    = self._rng.uniform(low, high)

        return curve * np.float32(factor)

    def _shift(self, curve: np.ndarray) -> np.ndarray:
        """Shifts a curve of shape (H,) by up to max_shift height bins, zero-filling."""
        if self.config.max_shift <= 0 or self._rng.random() >= self.config.p_shift:
            return curve

        k = int(self._rng.integers(-self.config.max_shift, self.config.max_shift + 1))

        if k == 0:
            return curve

        shifted = np.zeros_like(curve)

        if k > 0:
            shifted[k:] = curve[:-k]
        else:
            shifted[:k] = curve[-k:]

        return shifted

    def _flip(self, curve: np.ndarray) -> np.ndarray:
        """Reverses a curve of shape (H,) along elevation with probability p_flip."""
        if self._rng.random() >= self.config.p_flip:
            return curve

        return curve[::-1]

    def add_noise(self, normalized_curve: np.ndarray) -> np.ndarray:
        """Adds Gaussian noise to a normalised curve.

        Args:
            normalized_curve : Normalised profile of shape (H,), in normalised units.

        Returns:
            The curve with noise of standard deviation `noise_std` added, or the
            input unchanged when the draw falls outside `p_noise`.
        """
        if self._rng.random() >= self.config.p_noise:
            return normalized_curve

        noise = self._rng.normal(0.0, self.config.noise_std, normalized_curve.shape).astype(normalized_curve.dtype)

        return normalized_curve + noise

    def reseed(self, seed: int) -> None:
        """Restarts the internal generator from a new seed."""
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, curve: np.ndarray) -> np.ndarray:
        """Applies scaling, shifting and flipping to a raw profile.

        Args:
            curve : Raw profile of shape (H,) in physical intensity units.

        Returns:
            The augmented profile of shape (H,), contiguous float32.
        """
        curve = self._scale_amplitude(curve)
        curve = self._shift(curve)
        curve = self._flip(curve)

        return np.ascontiguousarray(curve, dtype=np.float32)
