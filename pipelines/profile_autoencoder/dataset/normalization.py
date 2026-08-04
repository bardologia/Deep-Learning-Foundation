from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional

import numpy as np
import torch

from tools.data.gaussians    import GaussianMixture
from tools.data.io           import FileIO
from tools.data.sampling     import Sampler
from tools.data.transforms   import Log1pTransform
from tools.monitoring.logger import Logger


@dataclass
class ProfileStats:
    loc   : float
    scale : float

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        out_path  = directory / "profile_normalization_stats.json"

        payload = {
            "transform" : "log1p",
            "loc"       : self.loc,
            "scale"     : self.scale,
        }

        return FileIO.save_json(payload, out_path, indent=4)

    @classmethod
    def load(cls, directory: Path, logger: Optional[Logger] = None) -> "ProfileStats":
        path = Path(directory) / "profile_normalization_stats.json"
        if not path.exists():
            raise FileNotFoundError(f"Profile normalization stats not found at '{path}'.")

        payload = FileIO.load_json(path)

        if logger is not None:
            logger.section("[Profile normalization stats loaded]")
            logger.kv_table({
                "Stats path" : path,
                "loc"        : payload["loc"],
                "scale"      : payload["scale"],
            })

        return cls(loc = float(payload["loc"]), scale = float(payload["scale"]))


class ProfileStatsComputer:
    @staticmethod
    def _sample_indices(dataset, max_samples: int) -> np.ndarray:
        return Sampler.deterministic_indices(len(dataset), max_samples)

    @staticmethod
    def _evaluate_curves(dataset, indices: np.ndarray) -> np.ndarray:
        selected = dataset.index[indices]

        return GaussianMixture.evaluate_batch(
            dataset.x_axis,
            dataset.amps[selected],
            dataset.mus[selected],
            dataset.sigs[selected],
        )

    @staticmethod
    def compute(dataset, logger: Logger, max_samples: int = 100_000) -> ProfileStats:
        logger.section("[Profile Normalization Statistics]")

        indices = ProfileStatsComputer._sample_indices(dataset, max_samples)
        curves  = ProfileStatsComputer._evaluate_curves(dataset, indices)

        values = np.log1p(np.maximum(curves, 0.0)).astype(np.float64).ravel()
        loc    = float(values.mean())
        scale  = float(values.std())

        logger.kv_table({
            "Samples"   : f"{len(indices):,} / {len(dataset):,} profiles",
            "Transform" : "log1p then standardize",
            "loc"       : f"{loc:+.6f}",
            "scale"     : f"{scale:.6f}",
        })

        return ProfileStats(loc = loc, scale = max(scale, ProfileNormalizer.SCALE_FLOOR))


class ProfileNormalizer:
    SCALE_FLOOR = 1e-6

    def __init__(self, stats: ProfileStats) -> None:
        self.stats     = stats
        self.loc       = float(stats.loc)
        self.scale     = max(float(stats.scale), self.SCALE_FLOOR)
        self.inv_scale = 1.0 / self.scale

    def normalize(self, curve):
        x   = Log1pTransform.compress(curve)
        out = (x - self.loc) * self.inv_scale

        if isinstance(curve, torch.Tensor):
            return out

        return np.ascontiguousarray(out, dtype=np.float32)

    def denormalize(self, curve):
        x   = curve * self.scale + self.loc
        out = Log1pTransform.decompress(x)

        if isinstance(curve, torch.Tensor):
            return out

        return np.ascontiguousarray(out, dtype=np.float32)
