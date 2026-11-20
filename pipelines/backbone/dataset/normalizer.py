from __future__ import annotations

from typing import Union

import numpy as np
import torch

from configuration.normalization      import ChannelStats
from pipelines.backbone.dataset.stats import Stats


Array = Union[np.ndarray, torch.Tensor]


class Normalizer:

    def __init__(self, stats: Stats) -> None:
        self.stats   = stats
        self._arrays = {"input": self._vectors(stats.input_stats), "target": self._vectors(stats.target_stats)}
        self._device = {}

    @staticmethod
    def _vectors(stats: ChannelStats) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(stats.loc, dtype=np.float32), np.asarray(stats.scale, dtype=np.float32)

    def _affine(self, side: str, tensor: Array) -> tuple:
        loc, scale = self._arrays[side]
        shape      = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)

        if not isinstance(tensor, torch.Tensor):
            return loc.reshape(shape), scale.reshape(shape)

        key = (side, str(tensor.device), tensor.ndim)
        if key not in self._device:
            self._device[key] = (torch.as_tensor(loc, device=tensor.device).reshape(shape), torch.as_tensor(scale, device=tensor.device).reshape(shape))

        return self._device[key]

    def _forward(self, side: str, tensor: Array) -> Array:
        loc, scale = self._affine(side, tensor)
        out        = (tensor - loc) / scale

        return out if isinstance(out, torch.Tensor) else np.ascontiguousarray(out, dtype=np.float32)

    def _inverse(self, side: str, tensor: Array) -> Array:
        loc, scale = self._affine(side, tensor)
        out        = tensor * scale + loc

        return out if isinstance(out, torch.Tensor) else np.ascontiguousarray(out, dtype=np.float32)

    def normalize_input(self, tensor: Array) -> Array:
        return self._forward("input", tensor)

    def normalize_target(self, tensor: Array) -> Array:
        return self._forward("target", tensor)

    def denormalize_input(self, tensor: Array) -> Array:
        return self._inverse("input", tensor)

    def denormalize_target(self, tensor: Array) -> Array:
        return self._inverse("target", tensor)
