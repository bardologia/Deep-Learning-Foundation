from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io import FileIO


class AeMetricsBase:
    EPS = 1e-8

    def __init__(self, result) -> None:
        self.gt         = result.gt
        self.pred       = result.pred
        self.emb        = result.embeddings
        self.normalized = result.normalized_errors

    def _normalized_errors(self) -> dict:
        return dict(self.normalized)

    def _embedding_stats(self) -> dict:
        dim_std = np.std(self.emb, axis=0, dtype=np.float64)

        return {
            "embedding_norm_mean"           : float(np.mean(np.linalg.norm(self.emb, axis=1), dtype=np.float64)),
            "embedding_dim_std_mean"        : float(np.mean(dim_std)),
            "embedding_active_dim_fraction" : float(np.mean(dim_std > 1e-4)),
        }

    @staticmethod
    def write_json(metrics: dict, path: Path) -> Path:
        return FileIO.save_json(metrics, Path(path), indent=4)
