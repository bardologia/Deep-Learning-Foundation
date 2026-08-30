"""Per-channel normalization and denormalization of backbone inputs and targets.

The transform is affine in a per-channel location and scale, optionally preceded by
a log1p compression, and applies the output clamp on the inverse direction so
denormalized parameters stay inside their physical barrier.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from configuration.normalization import ChannelStats
from tools.data.transforms       import Log1pTransform

from pipelines.backbone.dataset.stats import Stats


Array = Union[np.ndarray, torch.Tensor]


class Normalizer:
    """Applies the fitted per-channel normalization to numpy arrays and torch tensors.

    Attributes:
        stats : Fitted input and output channel statistics plus the output clamp.
    """

    def __init__(self, stats: Stats) -> None:
        """Stores the statistics and prepares the per-channel vector caches.

        Args:
            stats : Fitted normalization statistics of the run.
        """
        self.stats = stats

        self._vectors: dict[int, dict]                         = {}
        self._device_vectors: dict[tuple[int, str, int], dict] = {}

    def _channel_vectors(self, stats: ChannelStats) -> dict:
        """Builds and caches the numpy location, scale and channel-index vectors.

        Args:
            stats : Channel statistics to vectorize.

        Returns:
            Dictionary with the per-channel "loc", "scale" and "inv_scale" arrays of
            shape (channels,), the indices "log_idx" of the log1p-compressed channels
            and the indices "barrier_idx" of the clampable channels that are not
            log1p-compressed.

        Raises:
            ValueError : If the statistics carry no per-channel strategies or clampable flags.
        """
        key = id(stats)
        if key in self._vectors:
            return self._vectors[key]

        if stats.strategies is None:
            raise ValueError("ChannelStats is missing per-channel strategies; cannot build normalization vectors.")
        if stats.clampable is None:
            raise ValueError("ChannelStats is missing per-channel clampable flags; cannot build normalization vectors.")

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
        """Mirrors the per-channel vectors onto a torch device, cached per device and rank.

        Args:
            stats   : Channel statistics the vectors belong to.
            vectors : Numpy vectors returned by _channel_vectors.
            device  : Device the tensors are placed on.
            shape   : Broadcast shape for loc and scale, (1, -1, 1, 1) for batched tensors and (-1, 1, 1) for single patches.

        Returns:
            Dictionary of the same keys as the numpy vectors, holding torch tensors.
        """
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
        """Runs the affine and log1p transform in either direction, for numpy or torch input.

        Args:
            tensor      : Array of shape (channels, height, width) or (batch, channels, height, width).
            stats       : Channel statistics defining loc, scale, log1p flags and clampability.
            inverse     : Whether to denormalize (True) instead of normalize (False).
            is_output   : Whether the array holds parameter targets, which enables the clamp.
            leaky_slope : Slope of the leaky branch of the log1p compression outside the clamp.

        Returns:
            The transformed array, same shape and backend as the input.
        """
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

    @staticmethod
    def _require(stats: Optional[ChannelStats], role: str) -> ChannelStats:
        """Returns the statistics of a role, raising ValueError when they are absent."""
        if stats is None:
            raise ValueError(f"Normalizer has no {role} stats; cannot (de)normalize {role}.")

        return stats

    def normalize_input(self, tensor: Array) -> Array:
        """Normalizes an input stack of shape (channels, h, w) or (batch, channels, h, w)."""
        return self._apply_normalization(tensor, self._require(self.stats.input_stats, "input"), inverse=False, is_output=False)

    def normalize_output(self, tensor: Array, leaky_slope: float = 0.0) -> Array:
        """Normalizes Gaussian parameters of shape (params, h, w) or (batch, params, h, w).

        Args:
            tensor      : Parameters in physical units (amplitude, mu in metres, sigma in metres).
            leaky_slope : Slope of the leaky branch of the log1p compression.

        Returns:
            The parameters in normalized units, same shape.
        """
        return self._apply_normalization(tensor, self._require(self.stats.output_stats, "output"), inverse=False, is_output=True, leaky_slope=leaky_slope)

    def denormalize_input(self, tensor: Array) -> Array:
        """Returns a normalized input stack back in physical units, same shape."""
        return self._apply_normalization(tensor, self._require(self.stats.input_stats, "input"), inverse=True, is_output=False)

    def denormalize_output(self, tensor: Array, leaky_slope: float = 0.0) -> Array:
        """Returns normalized Gaussian parameters in physical units, clamped when the clamp is enabled.

        Args:
            tensor      : Parameters in normalized units, shape (params, h, w) or (batch, params, h, w).
            leaky_slope : Slope of the leaky branch outside the clamp floor and ceiling.

        Returns:
            Amplitudes and metre-valued mu and sigma, same shape.
        """
        return self._apply_normalization(tensor, self._require(self.stats.output_stats, "output"), inverse=True, is_output=True, leaky_slope=leaky_slope)
