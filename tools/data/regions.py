from __future__ import annotations

from dataclasses import dataclass
from typing      import Tuple


@dataclass
class CropRegion:
    row_start : int
    row_end   : int
    col_start : int
    col_end   : int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.row_start, self.row_end, self.col_start, self.col_end)

    @property
    def row_size(self) -> int:
        return self.row_end - self.row_start

    @property
    def col_size(self) -> int:
        return self.col_end - self.col_start

    def contains(self, other: "CropRegion") -> bool:
        row_inside = self.row_start <= other.row_start and other.row_end <= self.row_end
        col_inside = self.col_start <= other.col_start and other.col_end <= self.col_end
        return row_inside and col_inside


@dataclass
class SplitRegions:
    train : CropRegion | list[CropRegion]
    val   : CropRegion | list[CropRegion]
    test  : CropRegion | list[CropRegion]

    @staticmethod
    def as_list(value) -> list[CropRegion]:
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @classmethod
    def from_bounds(cls, payload: dict) -> "SplitRegions":
        return cls(train=[CropRegion(*bounds) for bounds in payload["train"]], val=[CropRegion(*bounds) for bounds in payload["val"]], test=[CropRegion(*bounds) for bounds in payload["test"]])

    def bounds(self) -> dict:
        return {name: [list(region.as_tuple()) for region in regions] for name, regions in self.region_lists()}

    def items(self):
        return [("train", self.train), ("val", self.val), ("test", self.test)]

    def region_lists(self):
        return [(name, self.as_list(value)) for name, value in self.items()]

    def regions(self, split_name: str) -> list[CropRegion]:
        return self.as_list(dict(self.items())[split_name])

    def region_rows(self) -> list[dict]:
        rows = []
        for name, regions in self.region_lists():
            for index, region in enumerate(regions):
                label = name if len(regions) == 1 else f"{name}[{index}]"
                rows.append({"Split": label, "Crop": str(region.as_tuple()), "Rows": region.row_size, "Cols": region.col_size})
        return rows
