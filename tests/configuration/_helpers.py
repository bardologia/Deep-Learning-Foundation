from __future__ import annotations

import dataclasses

from tools.data.regions import CropRegion, SplitRegions


def make_crop() -> CropRegion:
    return CropRegion(row_start=0, row_end=256, col_start=0, col_end=256)


def make_split_regions() -> SplitRegions:
    return SplitRegions(
        train = CropRegion(row_start=0,   row_end=180, col_start=0, col_end=256),
        val   = CropRegion(row_start=180, row_end=218, col_start=0, col_end=256),
        test  = CropRegion(row_start=218, row_end=256, col_start=0, col_end=256),
    )


def is_dataclass_type(obj) -> bool:
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)
