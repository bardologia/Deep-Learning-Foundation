"""Cropping of extracted Gaussian parameter maps into split regions.

Validates that the train, validation and test regions sit inside the global crop
and do not overlap, then memory-maps the requested regions out of the parameter
stack.
"""

from __future__ import annotations

import numpy as np

from pipelines.shared.dataset.dataset_spatial import Layout
from tools.data.regions                       import SplitRegions
from tools.monitoring.logger                  import Logger


class ParameterCropper:
    """Reads per-split crops out of the extracted parameter maps.

    Attributes:
        layout        : Spatial layout of the preprocessing run.
        split_regions : Region definitions per split.
    """

    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger) -> None:
        """Validates the split regions against the global crop.

        Args:
            layout        : Spatial layout of the preprocessing run.
            split_regions : Region definitions per split.
            logger        : Logger for the region table.
        """
        self.layout        = layout
        self.split_regions = split_regions
        self.logger        = logger

        split_regions.validate_within(layout.global_crop)
        split_regions.validate_disjoint()

        self.logger.section("[ParameterCropper Initialized]")
        self.logger.metrics_table(split_regions.region_rows(), ["Split", "Crop", "Rows", "Cols"])

    def profile_length(self) -> int:
        """Returns the number of elevation bins of the preprocessing run."""
        return self.layout.profile_length

    def load_split(self, split_name: str) -> list[np.ndarray]:
        """Loads the parameter maps of every region belonging to one split.

        Args:
            split_name : "train", "val" or "test".

        Returns:
            One contiguous array per region, each of shape
            (3 * n_gaussians, azimuth, range).
        """
        regions    = self.split_regions.regions(split_name)
        parameters = np.load(str(self.layout.artifact_path("parameters")), mmap_mode="r", allow_pickle=False)

        arrays = []
        for region in regions:
            row_slice, col_slice = region.local_slices(self.layout.global_crop)
            arrays.append(np.ascontiguousarray(parameters[..., row_slice, col_slice]))

        total_pixels = sum(int(a.shape[-2] * a.shape[-1]) for a in arrays)

        self.logger.section(f"[Parameters Loaded: {split_name}]")
        self.logger.kv_table({
            "Regions"      : len(arrays),
            "Channels"     : int(arrays[0].shape[0]),
            "Total pixels" : f"{total_pixels:,}",
        })

        return arrays
