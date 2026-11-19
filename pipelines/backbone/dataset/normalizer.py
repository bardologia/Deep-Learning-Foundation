
from __future__ import annotations

from typing import Union

import numpy as np
import torch

from configuration.normalization import ChannelStats
from tools.data.transforms       import Log1pTransform

from pipelines.backbone.dataset.stats import Stats


Array = Union[np.ndarray, torch.Tensor]


class Normalizer:

    def __init__(self, stats: Stats) -> None:
        self.stats = stats

        self._vectors: dict[int, dict]                         = {}
        self._device_vectors: dict[tuple[int, str, int], dict] = {}

    def _channel_vectors(self, stats: ChannelStats) -> dict:
        key = id(stats)
        if key in self._vectors:
            return self._vectors[key]

        loc       = np.asarray(stats.loc,       dtype=np.float32)
        scale     = np.asarray(stats.scale,     dtype=np.float32)
        log1p     = np.asarray([strat.apply_log1p for strat in stats.strategies], dtype=bool)
        clampable = np.asarray(stats.clampable, dtype=bool)
        inv_scale = (1.0 / scale).astype(np.float32)

        vectors = {
            "loc"         : loc,
            "scale"       : scale,
            "inv_scale"   : inv_scale,
            "log_idx"     : np.flatnonzero(log1p),
            "barrier_idx" : np.flatnonzero(clampable & ~log1p),
        }
        self._vectors[key] = vectors

        return vectors

    def _device_vectors_for(self, stats: ChannelStats, vectors: dict, device: torch.device, shape: tuple) -> dict:
        key = (id(stats), str(device), len(shape))
        if key in self._device_vectors:
            return self._device_vectors[key]

        built = {
            "loc"         : torch.as_tensor(vectors["loc"],       device=device).reshape(shape),
            "scale"       : torch.as_tensor(vectors["scale"],     device=device).reshape(shape),
            "inv_scale"   : torch.as_tensor(vectors["inv_scale"], device=device).reshape(shape),
            "log_idx"     : torch.as_tensor(vectors["log_idx"],     device=device),
            "barrier_idx" : torch.as_tensor(vectors["barrier_idx"], device=device),
        }
        self._device_vectors[key] = built

        return built

    def _apply_normalization(self, tensor: Array, stats: ChannelStats, inverse: bool, is_output: bool, leaky_slope: float = 0.0) -> Array:
        is_torch = isinstance(tensor, torch.Tensor)
        vectors  = self._channel_vectors(stats)
        clamp    = self.stats.clamp
        clamped  = clamp.enabled and is_output

        shape       = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)
        log_idx     = vectors["log_idx"]
        barrier_idx = vectors["barrier_idx"]

        def _select(idx):
            return (slice(None), idx) if tensor.ndim == 4 else (idx,)

        if is_torch:
            device    = tensor.device
            on_device = self._device_vectors_for(stats, vectors, device, shape)
            loc       = on_device["loc"]
            scale     = on_device["scale"]
            inv_scale = on_device["inv_scale"]

            if not inverse:
                x = tensor.clone()
                if log_idx.size:
                    sel    = _select(on_device["log_idx"])
                    x[sel] = Log1pTransform.compress(tensor[sel], leaky_slope)
                return (x - loc) * inv_scale

            x   = tensor * scale + loc
            out = x.clone()
            if log_idx.size:
                sel      = _select(on_device["log_idx"])
                out[sel] = Log1pTransform.decompress(x[sel], clamp.floor, clamp.ceil, clamped, leaky_slope)
            if clamped and barrier_idx.size:
                sel      = _select(on_device["barrier_idx"])
                out[sel] = Log1pTransform.decompress(Log1pTransform.compress(x[sel], leaky_slope), clamp.floor, clamp.ceil, clamped, leaky_slope)
            return out

        loc       = vectors["loc"].reshape(shape)
        scale     = vectors["scale"].reshape(shape)
        inv_scale = vectors["inv_scale"].reshape(shape)

        if not inverse:
            x = np.array(tensor, copy=True)
            if log_idx.size:
                sel    = _select(log_idx)
                x[sel] = Log1pTransform.compress(np.asarray(tensor)[sel], leaky_slope)
            out = (x - loc) * inv_scale
            return np.ascontiguousarray(out, dtype=np.float32)

        x   = np.asarray(tensor) * scale + loc
        out = np.array(x, copy=True)
        if log_idx.size:
            sel      = _select(log_idx)
            out[sel] = Log1pTransform.decompress(x[sel], clamp.floor, clamp.ceil, clamped, leaky_slope)
        if clamped and barrier_idx.size:
            sel      = _select(barrier_idx)
            out[sel] = Log1pTransform.decompress(Log1pTransform.compress(x[sel], leaky_slope), clamp.floor, clamp.ceil, clamped, leaky_slope)
        return np.ascontiguousarray(out, dtype=np.float32)

    def normalize_input(self, tensor: Array) -> Array:
        return self._apply_normalization(tensor, self.stats.input_stats, inverse=False, is_output=False)

    def normalize_output(self, tensor: Array, leaky_slope: float = 0.0) -> Array:
        return self._apply_normalization(tensor, self.stats.output_stats, inverse=False, is_output=True, leaky_slope=leaky_slope)

    def denormalize_input(self, tensor: Array) -> Array:
        return self._apply_normalization(tensor, self.stats.input_stats, inverse=True, is_output=False)

    def denormalize_output(self, tensor: Array, leaky_slope: float = 0.0) -> Array:
        return self._apply_normalization(tensor, self.stats.output_stats, inverse=True, is_output=True, leaky_slope=leaky_slope)
