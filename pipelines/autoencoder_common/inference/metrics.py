"""Reconstruction and latent metrics shared by the autoencoder families.

Concrete profile and image autoencoder metric classes extend this base with
their own domain-specific terms and a ``compute`` entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io import FileIO


class AeMetricsBase:
    """Per-sample reconstruction error and latent statistics for one split.

    Attributes:
        gt         : Ground-truth targets of shape (n_samples, *sample_shape).
        pred       : Reconstructions of the same shape as the targets.
        emb        : Latent embeddings of shape (n_samples, embedding_dim).
        normalized : Reconstruction errors measured in normalized units.
    """

    EPS = 1e-8

    def __init__(self, result) -> None:
        """Unpacks a predictor result into the arrays the metrics are computed on.

        Args:
            result : Predictor result carrying gt, pred, embeddings and normalized_errors.
        """
        self.gt         = result.gt
        self.pred       = result.pred
        self.emb        = result.embeddings
        self.normalized = result.normalized_errors

        self._sample_mse = None

    def per_sample_mse(self) -> np.ndarray:
        """Returns the cached mean squared error per sample, of shape (n_samples,)."""
        if self._sample_mse is None:
            diff             = self.pred - self.gt
            self._sample_mse = np.mean(diff ** 2, axis=tuple(range(1, diff.ndim)), dtype=np.float64)

        return self._sample_mse

    def _normalized_errors(self) -> dict:
        """Returns a copy of the reconstruction errors measured in normalized units."""
        return dict(self.normalized)

    def _embedding_stats(self) -> dict:
        """Returns latent norm, per-dimension spread and active-dimension fraction."""
        dim_std = np.std(self.emb, axis=0, dtype=np.float64)

        return {
            "embedding_norm_mean"           : float(np.mean(np.linalg.norm(self.emb, axis=1), dtype=np.float64)),
            "embedding_dim_std_mean"        : float(np.mean(dim_std)),
            "embedding_active_dim_fraction" : float(np.mean(dim_std > 1e-4)),
        }

    @staticmethod
    def write_json(metrics: dict, path: Path) -> Path:
        """Writes the metrics dictionary as indented JSON and returns the written path."""
        return FileIO.save_json(metrics, Path(path), indent=4)
