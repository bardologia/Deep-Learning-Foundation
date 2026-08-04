from __future__ import annotations

from dataclasses import dataclass
from typing      import Tuple


@dataclass
class CropRegion:
    row_start : int
    row_end   : int
    col_start : int
    col_end   : int

    def __post_init__(self) -> None:
        if self.row_start >= self.row_end:
            raise ValueError(f"row_start ({self.row_start}) must be < row_end ({self.row_end})")
        if self.col_start   >= self.col_end:
            raise ValueError(f"col_start ({self.col_start}) must be < col_end ({self.col_end})")

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.row_start, self.row_end, self.col_start, self.col_end)

    def as_identifier_string(self) -> str:
        return "a".join(str(value) for value in self.as_tuple())

    def as_labeled_string(self) -> str:
        return f"row{self.row_start}-{self.row_end}_col{self.col_start}-{self.col_end}"

    @property
    def row_size(self) -> int:
        return self.row_end - self.row_start

    @property
    def col_size(self) -> int:
        return self.col_end - self.col_start

    def overlaps(self, other: "CropRegion") -> bool:
        row_overlap = self.row_start < other.row_end and other.row_start < self.row_end
        col_overlap = self.col_start   < other.col_end   and other.col_start   < self.col_end
        return row_overlap and col_overlap

    def contains(self, other: "CropRegion") -> bool:
        row_inside = self.row_start <= other.row_start and other.row_end <= self.row_end
        col_inside = self.col_start   <= other.col_start   and other.col_end   <= self.col_end
        return row_inside and col_inside

    def local_slices(self, global_crop: "CropRegion") -> Tuple[slice, slice]:
        row_slice = slice(self.row_start - global_crop.row_start, self.row_end - global_crop.row_start)
        col_slice = slice(self.col_start   - global_crop.col_start,   self.col_end   - global_crop.col_start)
        return row_slice, col_slice

    def subdivide_by_rows(self, max_width: int) -> list["CropRegion"]:
        subsections = []
        current     = self.row_start

        while current < self.row_end:
            next_row = min(current + max_width, self.row_end)
            subsections.append(CropRegion(current, next_row, self.col_start, self.col_end))
            current = next_row

        return subsections


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

    def validate_disjoint(self) -> None:
        labeled = []
        for name, regions in self.region_lists():
            for index, region in enumerate(regions):
                label = name if len(regions) == 1 else f"{name}[{index}]"
                labeled.append((label, region))

        for i in range(len(labeled)):
            for j in range(i + 1, len(labeled)):
                name_a, region_a = labeled[i]
                name_b, region_b = labeled[j]
                if region_a.overlaps(region_b):
                    raise ValueError(f"Split regions '{name_a}' {region_a.as_tuple()} and '{name_b}' {region_b.as_tuple()} overlap; all train/val/test regions must be spatially disjoint.")

    def validate_within(self, global_crop: CropRegion) -> None:
        for name, regions in self.region_lists():
            for index, region in enumerate(regions):
                if not global_crop.contains(region):
                    label = name if len(regions) == 1 else f"{name}[{index}]"
                    raise ValueError(f"Split region '{label}' {region.as_tuple()} is not contained in the preprocessing global crop {global_crop.as_tuple()}; out-of-crop coordinates would silently wrap or truncate when sliced, loading pixels from the wrong area.")

    @classmethod
    def from_ratios(cls, global_crop: CropRegion, train_ratio: float = 0.70, val_ratio: float = 0.15) -> "SplitRegions":
        total_rows = global_crop.row_end - global_crop.row_start
        train_end  = global_crop.row_start + int(total_rows * train_ratio)
        val_end    = train_end + int(total_rows * val_ratio)

        return cls(
            train = CropRegion(global_crop.row_start, train_end,               global_crop.col_start, global_crop.col_end),
            val   = CropRegion(train_end,                 val_end,                 global_crop.col_start, global_crop.col_end),
            test  = CropRegion(val_end,                   global_crop.row_end, global_crop.col_start, global_crop.col_end),
        )
