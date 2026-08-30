"""Spatial and additive-noise augmentation for backbone training patches.

Augmentation is applied to the train split only, jointly to the input stack, the
ground-truth parameter maps and the optional per-pixel kz field so their spatial
alignment is preserved.
"""

from __future__ import annotations

import numpy as np

from configuration.dataset import AugmentationConfig


class SpatialAugmenter:
    """Applies flips, 90-degree rotations and Gaussian noise to training patches.

    Attributes:
        config : Probabilities and noise standard deviation driving the augmentation.
        logger : Logger used to report the augmentation configuration.
        seed   : Seed of the internal random generator.
    """

    def __init__(self, config: AugmentationConfig, logger, seed: int = 0):
        """Stores the augmentation configuration and logs it.

        Args:
            config : Augmentation probabilities and noise scale.
            logger : Logger receiving the configuration table.
            seed   : Seed for the numpy generator drawing the augmentation decisions.
        """
        self.config = config
        self.logger = logger
        self.seed   = int(seed)
        self._rng   = np.random.default_rng(self.seed)

        self.logger.section("[Data Augmentation]")
        self.logger.kv_table(
            {
                "Applies to"      : "train split only",
                "Flip Horizontal" : self.config.p_flip_h,
                "Flip Vertical"   : self.config.p_flip_v,
                "Rotate 90°"      : self.config.p_rot90,
                "Noise"           : f"std={self.config.noise_std} (normalized units) p={self.config.p_noise}",
            },
            title="Augmentation Config",
        )

    def _flip_horizontal(self, tensors: list) -> list:
        """Mirrors every tensor along the last (range) axis with probability p_flip_h."""
        if self._rng.random() >= self.config.p_flip_h:
            return tensors

        sl = (Ellipsis, slice(None), slice(None, None, -1))

        return [tensor[sl] for tensor in tensors]

    def _flip_vertical(self, tensors: list) -> list:
        """Mirrors every tensor along the second-to-last (row) axis with probability p_flip_v."""
        if self._rng.random() >= self.config.p_flip_v:
            return tensors

        sl = (Ellipsis, slice(None, None, -1), slice(None))

        return [tensor[sl] for tensor in tensors]

    def _rotate90(self, tensors: list) -> list:
        """Rotates every tensor by a random multiple of 90 degrees in the spatial plane with probability p_rot90."""
        if self.config.p_rot90 <= 0.0 or self._rng.random() >= self.config.p_rot90:
            return tensors

        k = int(self._rng.integers(1, 4))

        return [np.rot90(tensor, k=k, axes=(-2, -1)) for tensor in tensors]

    def add_noise(self, normalized_input: np.ndarray) -> np.ndarray:
        """Adds zero-mean Gaussian noise to a normalized input patch.

        Args:
            normalized_input : Normalized input tensor of shape (channels, rows, cols).

        Returns:
            The patch with noise of standard deviation noise_std (normalized units)
            added with probability p_noise, otherwise the untouched input.
        """
        if self._rng.random() >= self.config.p_noise:
            return normalized_input

        noise = self._rng.normal(0.0, self.config.noise_std, normalized_input.shape).astype(normalized_input.dtype)

        return normalized_input + noise

    def reseed(self, seed: int) -> None:
        """Restarts the internal random generator from the given seed."""
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, input_tensor: np.ndarray, gt_params: np.ndarray):
        """Applies the geometric augmentations jointly to a patch and its targets.

        Args:
            input_tensor : Input patch of shape (channels, rows, cols).
            gt_params    : Ground-truth Gaussian parameters of shape (params, rows, cols).

        Returns:
            The augmented (input_tensor, gt_params) pair.
        """
        tensors = [input_tensor, gt_params]

        tensors = self._flip_horizontal(tensors)
        tensors = self._flip_vertical(tensors)
        tensors = self._rotate90(tensors)

        tensors = [np.ascontiguousarray(tensor) for tensor in tensors]

        return tensors[0], tensors[1]
