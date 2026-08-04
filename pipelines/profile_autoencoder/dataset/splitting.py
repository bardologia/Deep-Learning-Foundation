from __future__ import annotations

import numpy as np

from pipelines.shared.dataset.dataset_spatial import Layout
from tools.data.regions                       import SplitRegions
from tools.monitoring.logger                  import Logger


class ParameterCropper:
    def __init__(self, layout: Layout, split_regions: SplitRegions, logger: Logger) -> None:
        self.layout        = layout
        self.split_regions = split_regions
        self.logger        = logger

        split_regions.validate_within(layout.global_crop)
        split_regions.validate_disjoint()

        self.logger.section("[ParameterCropper Initialized]")
        self.logger.metrics_table(split_regions.region_rows(), ["Split", "Crop", "Azimuth (lines)", "Range (samples)"])

    def profile_length(self) -> int:
        return self.layout.profile_length

    def load_split(self, split_name: str) -> list[np.ndarray]:
        regions    = self.split_regions.regions(split_name)
        parameters = np.load(str(self.layout.artifact_path("parameters")), mmap_mode="r", allow_pickle=False)

        arrays = []
        for region in regions:
            az_slice, rg_slice = region.local_slices(self.layout.global_crop)
            arrays.append(np.ascontiguousarray(parameters[..., az_slice, rg_slice]))

        total_pixels = sum(int(a.shape[-2] * a.shape[-1]) for a in arrays)

        self.logger.section(f"[Parameters Loaded: {split_name}]")
        self.logger.kv_table({
            "Regions"      : len(arrays),
            "Channels"     : int(arrays[0].shape[0]),
            "Total pixels" : f"{total_pixels:,}",
        })

        return arrays
