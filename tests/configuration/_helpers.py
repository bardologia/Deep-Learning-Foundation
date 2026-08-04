from __future__ import annotations

import dataclasses

from tools.data.regions                import CropRegion, SplitRegions
from configuration.dataset.gaussian import GaussianConfig


def make_crop() -> CropRegion:
    return CropRegion(row_start=1000, row_end=2000, col_start=500, col_end=1000)


def make_split_regions() -> SplitRegions:
    return SplitRegions(
        train = CropRegion(row_start=1000, row_end=1300, col_start=500, col_end=1000),
        val   = CropRegion(row_start=1300, row_end=1450, col_start=500, col_end=1000),
        test  = CropRegion(row_start=1450, row_end=1600, col_start=500, col_end=1000),
    )


def make_gaussian() -> GaussianConfig:
    return GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=80.0)


def is_dataclass_type(obj) -> bool:
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)
