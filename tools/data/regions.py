"""Row-column crop regions and the train/val/test splits built from them.

A region is a half-open rectangle in row-column array coordinates. Splits are collections
of such regions that must stay disjoint and inside the preprocessing crop, so
no pixel is shared between training and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing      import Tuple


@dataclass
class CropRegion:
    """Half-open rectangle in row-column array coordinates.

    Attributes:
        row_start : First row line, inclusive.
        row_end   : Last row line, exclusive.
        col_start   : First col sample, inclusive.
        col_end     : Last col sample, exclusive.
    """
    row_start : int
    row_end   : int
    col_start   : int
    col_end     : int

    def __post_init__(self) -> None:
        """Rejects empty or inverted regions.

        Raises:
            ValueError : If either axis does not start strictly before it ends.
        """
        if self.row_start >= self.row_end:
            raise ValueError(f"row_start ({self.row_start}) must be < row_end ({self.row_end})")
        if self.col_start   >= self.col_end:
            raise ValueError(f"col_start ({self.col_start}) must be < col_end ({self.col_end})")

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Returns the region as (row_start, row_end, col_start, col_end)."""
        return (self.row_start, self.row_end, self.col_start, self.col_end)

    def as_identifier_string(self) -> str:
        """Returns the region as a compact 'a'-joined identifier string."""
        return "a".join(str(value) for value in self.as_tuple())

    def as_labeled_string(self) -> str:
        """Returns the region as an axis-labelled string for file names."""
        return f"row{self.row_start}-{self.row_end}_col{self.col_start}-{self.col_end}"

    @property
    def row_size(self) -> int:
        """Extent of the region in row lines."""
        return self.row_end - self.row_start

    @property
    def col_size(self) -> int:
        """Extent of the region in col samples."""
        return self.col_end - self.col_start

    def overlaps(self, other: "CropRegion") -> bool:
        """Returns whether this region shares any pixel with another."""
        row_overlap = self.row_start < other.row_end and other.row_start < self.row_end
        col_overlap = self.col_start < other.col_end and other.col_start < self.col_end
        return row_overlap and col_overlap

    def contains(self, other: "CropRegion") -> bool:
        """Returns whether another region lies entirely inside this one."""
        row_inside = self.row_start <= other.row_start and other.row_end <= self.row_end
        col_inside = self.col_start <= other.col_start and other.col_end <= self.col_end
        return row_inside and col_inside

    def local_slices(self, global_crop: "CropRegion") -> Tuple[slice, slice]:
        """Returns this region as slices into an array cropped to ``global_crop``.

        Args:
            global_crop : Region the array was cropped to.

        Returns:
            The row and column slices addressing this region within that array.
        """
        row_slice = slice(self.row_start - global_crop.row_start, self.row_end - global_crop.row_start)
        col_slice = slice(self.col_start - global_crop.col_start, self.col_end - global_crop.col_start)
        return row_slice, col_slice

    def subdivide_by_rows(self, max_width: int) -> list["CropRegion"]:
        """Splits the region into consecutive row strips.

        Args:
            max_width : Largest row extent of a strip.

        Returns:
            The strips covering this region, in row order; the last may be shorter.
        """
        subsections = []
        current     = self.row_start

        while current < self.row_end:
            next_row = min(current + max_width, self.row_end)
            subsections.append(CropRegion(current, next_row, self.col_start, self.col_end))
            current = next_row

        return subsections


@dataclass
class SplitRegions:
    """Train, validation and test regions of one dataset.

    Attributes:
        train : Training region or list of regions.
        val   : Validation region or list of regions.
        test  : Test region or list of regions.
    """
    train : CropRegion | list[CropRegion]
    val   : CropRegion | list[CropRegion]
    test  : CropRegion | list[CropRegion]

    @staticmethod
    def as_list(value) -> list[CropRegion]:
        """Returns the value as a list of regions, wrapping a single region."""
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @classmethod
    def from_payload(cls, payload: dict) -> "SplitRegions":
        """Builds the split regions from their persisted form.

        Args:
            payload : Mapping of split name to one region dict or a list of them, as written into a run's dataset creation config.

        Returns:
            The split regions, each split a single region or a list of disjoint regions as stored.
        """
        return cls(train=cls._parse(payload["train"]), val=cls._parse(payload["val"]), test=cls._parse(payload["test"]))

    @staticmethod
    def _parse(value):
        """Returns one persisted split as a CropRegion, or a list of them when disjoint."""
        if isinstance(value, list):
            return [CropRegion(**region) for region in value]
        return CropRegion(**value)

    def items(self):
        """Returns the (split name, value) pairs in train, val, test order."""
        return [("train", self.train), ("val", self.val), ("test", self.test)]

    def region_lists(self):
        """Returns the (split name, region list) pairs in train, val, test order."""
        return [(name, self.as_list(value)) for name, value in self.items()]

    def regions(self, split_name: str) -> list[CropRegion]:
        """Returns the regions of one split as a list."""
        return self.as_list(dict(self.items())[split_name])

    def region_rows(self) -> list[dict]:
        """Returns one table row per region with its label, crop and extent in lines and samples."""
        rows = []
        for name, regions in self.region_lists():
            for index, region in enumerate(regions):
                label = name if len(regions) == 1 else f"{name}[{index}]"
                rows.append({"Split": label, "Crop": str(region.as_tuple()), "Rows": region.row_size, "Cols": region.col_size})
        return rows

    def validate_disjoint(self) -> None:
        """Checks that no two split regions overlap.

        Raises:
            ValueError : If any pair of regions shares a pixel.
        """
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
        """Checks that every split region lies inside the preprocessing crop.

        Args:
            global_crop : Region the preprocessed data covers.

        Raises:
            ValueError : If a split region is not contained in that crop.
        """
        for name, regions in self.region_lists():
            for index, region in enumerate(regions):
                if not global_crop.contains(region):
                    label = name if len(regions) == 1 else f"{name}[{index}]"
                    raise ValueError(f"Split region '{label}' {region.as_tuple()} is not contained in the preprocessing global crop {global_crop.as_tuple()}; out-of-crop coordinates would silently wrap or truncate when sliced, loading pixels from the wrong area.")

