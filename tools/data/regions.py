from __future__ import annotations

from dataclasses import dataclass
from typing      import Tuple


@dataclass
class CropRegion:
    azimuth_start : int
    azimuth_end   : int
    range_start   : int
    range_end     : int

    def __post_init__(self) -> None:
        if self.azimuth_start >= self.azimuth_end:
            raise ValueError(f"azimuth_start ({self.azimuth_start}) must be < azimuth_end ({self.azimuth_end})")
        if self.range_start   >= self.range_end:
            raise ValueError(f"range_start ({self.range_start}) must be < range_end ({self.range_end})")

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.azimuth_start, self.azimuth_end, self.range_start, self.range_end)

    def as_identifier_string(self) -> str:
        return "a".join(str(value) for value in self.as_tuple())

    def as_labeled_string(self) -> str:
        return f"az{self.azimuth_start}-{self.azimuth_end}_rg{self.range_start}-{self.range_end}"

    @property
    def azimuth_size(self) -> int:
        return self.azimuth_end - self.azimuth_start

    @property
    def range_size(self) -> int:
        return self.range_end - self.range_start

    def overlaps(self, other: "CropRegion") -> bool:
        azimuth_overlap = self.azimuth_start < other.azimuth_end and other.azimuth_start < self.azimuth_end
        range_overlap   = self.range_start   < other.range_end   and other.range_start   < self.range_end
        return azimuth_overlap and range_overlap

    def contains(self, other: "CropRegion") -> bool:
        azimuth_inside = self.azimuth_start <= other.azimuth_start and other.azimuth_end <= self.azimuth_end
        range_inside   = self.range_start   <= other.range_start   and other.range_end   <= self.range_end
        return azimuth_inside and range_inside

    def local_slices(self, global_crop: "CropRegion") -> Tuple[slice, slice]:
        azimuth_slice = slice(self.azimuth_start - global_crop.azimuth_start, self.azimuth_end - global_crop.azimuth_start)
        range_slice   = slice(self.range_start   - global_crop.range_start,   self.range_end   - global_crop.range_start)
        return azimuth_slice, range_slice

    def subdivide_by_azimuth(self, max_width: int) -> list["CropRegion"]:
        subsections = []
        current     = self.azimuth_start

        while current < self.azimuth_end:
            next_azimuth = min(current + max_width, self.azimuth_end)
            subsections.append(CropRegion(current, next_azimuth, self.range_start, self.range_end))
            current = next_azimuth

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
                rows.append({"Split": label, "Crop": str(region.as_tuple()), "Azimuth (lines)": region.azimuth_size, "Range (samples)": region.range_size})
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
        total_az  = global_crop.azimuth_end - global_crop.azimuth_start
        train_end = global_crop.azimuth_start + int(total_az * train_ratio)
        val_end   = train_end + int(total_az * val_ratio)

        return cls(
            train = CropRegion(global_crop.azimuth_start, train_end,               global_crop.range_start, global_crop.range_end),
            val   = CropRegion(train_end,                 val_end,                 global_crop.range_start, global_crop.range_end),
            test  = CropRegion(val_end,                   global_crop.azimuth_end, global_crop.range_start, global_crop.range_end),
        )
