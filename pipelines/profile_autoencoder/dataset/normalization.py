"""Log1p standardisation of z profiles and its persisted statistics.

Fits the location and scale of log1p-compressed training curves, stores them
next to the run metadata, and applies them in both directions at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import ClassVar, Optional

import numpy as np
import torch

from tools.data.gaussians    import GaussianMixture
from tools.data.io           import FileIO
from tools.data.sampling     import Sampler
from tools.data.transforms   import Log1pTransform
from tools.monitoring.logger import Logger


@dataclass
class ProfileStats:
    """Location and scale of log1p-compressed profile amplitudes.

    Attributes:
        FILENAME : File name of the persisted statistics inside a metadata directory.
        loc      : Mean of the log1p-compressed training values.
        scale    : Standard deviation of the log1p-compressed training values.
    """

    FILENAME : ClassVar[str] = "profile_normalization_stats.json"

    loc   : float
    scale : float

    def payload(self) -> dict:
        """Returns the JSON payload persisted for these statistics."""
        return {
            "transform" : "log1p",
            "loc"       : self.loc,
            "scale"     : self.scale,
        }

    def save(self, directory: Path) -> Path:
        """Writes the statistics as profile_normalization_stats.json.

        Args:
            directory : Metadata directory the file is written into.

        Returns:
            Path of the written JSON file.
        """
        return FileIO.save_json(self.payload(), Path(directory) / self.FILENAME, indent=4)

    @classmethod
    def load(cls, directory: Path, logger: Optional[Logger] = None) -> "ProfileStats":
        """Reads the statistics back from a metadata directory.

        Args:
            directory : Directory holding profile_normalization_stats.json.
            logger    : Logger that reports the loaded values when given.

        Returns:
            The persisted `ProfileStats`.

        Raises:
            FileNotFoundError : If the statistics file is absent.
            ValueError        : If the persisted location is not finite or the scale is not finite and positive.
        """
        path = Path(directory) / cls.FILENAME
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

        loc   = float(payload["loc"])
        scale = float(payload["scale"])

        if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Corrupt profile normalization stats in '{path}': loc={loc}, scale={scale} must be finite with scale > 0.")

        return cls(loc = loc, scale = scale)


class ProfileStatsComputer:
    """Estimates profile normalisation statistics from a training dataset."""

    @staticmethod
    def _sample_indices(dataset, max_samples: int) -> np.ndarray:
        """Returns up to `max_samples` deterministic positions into the dataset."""
        return Sampler.deterministic_indices(len(dataset), max_samples)

    @staticmethod
    def _evaluate_curves(dataset, indices: np.ndarray) -> np.ndarray:
        """Evaluates the Gaussian mixtures at the sampled dataset positions.

        Args:
            dataset : Profile dataset holding the parameter arrays.
            indices : Positions into `dataset.index`.

        Returns:
            Raw curves of shape (len(indices), H).
        """
        selected = dataset.index[indices]

        return GaussianMixture.evaluate_batch(
            dataset.z_axis,
            dataset.amps[selected],
            dataset.mus[selected],
            dataset.sigs[selected],
        )

    @staticmethod
    def compute(dataset, logger: Logger, max_samples: int = 100_000) -> ProfileStats:
        """Fits the log1p location and scale over a sample of training curves.

        Args:
            dataset     : Training profile dataset.
            logger      : Logger for the statistics table.
            max_samples : Maximum number of curves sampled.

        Returns:
            Statistics with the measured location and scale.

        Raises:
            ValueError : If the sampled curves are (near-)constant, i.e. the measured scale is below `ProfileNormalizer.SCALE_FLOOR`; dividing by a floored scale would blow the normalized curves up by orders of magnitude.
        """
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

        if scale < ProfileNormalizer.SCALE_FLOOR:
            raise ValueError(f"The {len(indices):,} sampled training profiles are (near-)constant after log1p: measured std {scale:.3e} is below the scale floor {ProfileNormalizer.SCALE_FLOOR:g}, so standardizing by it would multiply every curve by more than {1.0 / ProfileNormalizer.SCALE_FLOOR:.0e}. The training split carries no usable profile variation; pick another split region or parameter run.")

        return ProfileStats(loc = loc, scale = scale)


class ProfileNormalizer:
    """Applies and inverts log1p standardisation of profile curves.

    Attributes:
        stats     : Statistics the transform was fitted with.
        loc       : Location subtracted after log1p compression.
        scale     : Standard deviation the compressed values are divided by.
        inv_scale : Reciprocal of `scale`.
    """

    SCALE_FLOOR = 1e-6

    def __init__(self, stats: ProfileStats) -> None:
        """Stores the statistics with the scale floored away from zero.

        Args:
            stats : Fitted log1p location and scale.

        Raises:
            ValueError : If the location or the scale is not finite.
        """
        loc   = float(stats.loc)
        scale = float(stats.scale)

        if not np.isfinite(loc) or not np.isfinite(scale):
            raise ValueError(f"ProfileNormalizer received non-finite stats: loc={loc}, scale={scale}")

        self.stats     = stats
        self.loc       = loc
        self.scale     = max(scale, self.SCALE_FLOOR)
        self.inv_scale = 1.0 / self.scale

    def normalize(self, curve):
        """Compresses and standardises a curve.

        Args:
            curve : Raw profile of shape (..., H) as a numpy array or torch tensor, in physical intensity units.

        Returns:
            The normalised profile with the same shape and container type.
        """
        x   = Log1pTransform.compress(curve)
        out = (x - self.loc) * self.inv_scale

        if isinstance(curve, torch.Tensor):
            return out

        return np.ascontiguousarray(out, dtype=np.float32)

    def denormalize(self, curve):
        """Inverts `normalize`, returning physical intensity units.

        Args:
            curve : Normalised profile of shape (..., H) as a numpy array or torch tensor.

        Returns:
            The raw profile with the same shape and container type.
        """
        x   = curve * self.scale + self.loc
        out = Log1pTransform.decompress(x)

        if isinstance(curve, torch.Tensor):
            return out

        return np.ascontiguousarray(out, dtype=np.float32)
