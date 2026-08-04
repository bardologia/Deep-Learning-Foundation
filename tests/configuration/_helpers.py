from __future__ import annotations

import dataclasses

from tools.data.regions                import CropRegion, SplitRegions
from configuration.sar.gaussian_config import GaussianConfig


def make_crop() -> CropRegion:
    return CropRegion(azimuth_start=1000, azimuth_end=2000, range_start=500, range_end=1000)


def make_split_regions() -> SplitRegions:
    return SplitRegions(
        train = CropRegion(azimuth_start=1000, azimuth_end=1300, range_start=500, range_end=1000),
        val   = CropRegion(azimuth_start=1300, azimuth_end=1450, range_start=500, range_end=1000),
        test  = CropRegion(azimuth_start=1450, azimuth_end=1600, range_start=500, range_end=1000),
    )


def make_gaussian() -> GaussianConfig:
    return GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0)


def is_dataclass_type(obj) -> bool:
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)
