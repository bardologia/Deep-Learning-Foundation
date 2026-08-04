from __future__ import annotations

import numpy as np

from configuration.dataset import AugmentationConfig


class SpatialAugmenter:
    def __init__(self, config: AugmentationConfig, logger, seed: int = 0, rot90_inactive: bool = False):
        self.config = config
        self.logger = logger
        self.seed   = int(seed)
        self._rng   = np.random.default_rng(self.seed)

        rot90 = f"{self.config.p_rot90} (inactive: skipped while a per-pixel geometry field is active)" if rot90_inactive else self.config.p_rot90

        self.logger.section("[Data Augmentation]")
        self.logger.kv_table(
            {
                "Applies to"      : "train split only",
                "Flip Horizontal" : self.config.p_flip_h,
                "Flip Vertical"   : self.config.p_flip_v,
                "Rotate 90°"      : rot90,
                "Noise"           : f"std={self.config.noise_std} (normalized units) p={self.config.p_noise}",
            },
            title="Augmentation Config",
        )

    def _flip_horizontal(self, tensors: list) -> list:
        if self._rng.random() >= self.config.p_flip_h:
            return tensors

        sl = (Ellipsis, slice(None), slice(None, None, -1))

        return [tensor[sl] for tensor in tensors]

    def _flip_vertical(self, tensors: list) -> list:
        if self._rng.random() >= self.config.p_flip_v:
            return tensors

        sl = (Ellipsis, slice(None, None, -1), slice(None))

        return [tensor[sl] for tensor in tensors]

    def _rotate90(self, tensors: list) -> list:
        if self.config.p_rot90 <= 0.0 or self._rng.random() >= self.config.p_rot90:
            return tensors

        k = int(self._rng.integers(1, 4))

        return [np.rot90(tensor, k=k, axes=(-2, -1)) for tensor in tensors]

    def add_noise(self, normalized_input: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.config.p_noise:
            return normalized_input

        noise = self._rng.normal(0.0, self.config.noise_std, normalized_input.shape).astype(normalized_input.dtype)

        return normalized_input + noise

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, input_tensor: np.ndarray, gt_params: np.ndarray, geometry: np.ndarray | None = None):
        tensors = [input_tensor, gt_params] if geometry is None else [input_tensor, gt_params, geometry]

        tensors = self._flip_horizontal(tensors)
        tensors = self._flip_vertical(tensors)

        if geometry is None:
            tensors = self._rotate90(tensors)

        tensors = [np.ascontiguousarray(tensor) for tensor in tensors]

        if geometry is None:
            return tensors[0], tensors[1]

        return tensors[0], tensors[1], tensors[2]
