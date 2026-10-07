"""Shared fixtures for the configuration tests: crop regions, split regions and Gaussian settings."""

from __future__ import annotations

import dataclasses

from tools.data.regions                import CropRegion, SplitRegions
from configuration.dataset.gaussian import GaussianConfig


def make_crop() -> CropRegion:
    """Returns a crop region spanning row 1000-2000 and col 500-1000 in pixels."""
    return CropRegion(row_start=1000, row_end=2000, col_start=500, col_end=1000)


def make_split_regions() -> SplitRegions:
    """Returns contiguous train, validation and test crops over the same range extent."""
    return SplitRegions(
        train = CropRegion(row_start=1000, row_end=1300, col_start=500, col_end=1000),
        val   = CropRegion(row_start=1300, row_end=1450, col_start=500, col_end=1000),
        test  = CropRegion(row_start=1450, row_end=1600, col_start=500, col_end=1000),
    )


def make_gaussian() -> GaussianConfig:
    """Returns a Gaussian config with five slots over a height range of -20 to 80 metres."""
    return GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0, activity_threshold=1e-4)


def is_dataclass_type(obj) -> bool:
    """Returns whether the object is a dataclass class rather than an instance."""
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)
