from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io           import FileIO
from tools.data.regions      import CropRegion
from tools.monitoring.logger import Logger
from tools.baselines         import SecondarySelection


class Layout:
    def __init__(self, run_directory: Path, logger: Logger, parameters_path: Path) -> None:
        self.run_directory   = Path(run_directory)
        self.logger          = logger
        self.data_directory  = self.run_directory / "data"
        self.parameters_path = Path(parameters_path)

        layout_path = self.data_directory / "dataset.json"
        payload     = FileIO.load_json(layout_path)

        self.global_crop  : CropRegion  = CropRegion(*payload["global_crop"])
        self.tomogram_tag : str         = payload["tomogram_tag"]
        self.artifacts    : dict        = payload["artifacts"]
        self.pass_labels  : list | None = payload["pass_labels"]

        self.logger.section("[Layout Loaded]")
        self.logger.kv_table({
            "Run Directory"   : self.run_directory,
            "Global Crop"     : self.global_crop.as_tuple(),
            "Azimuth (lines)" : self.global_crop.azimuth_size,
            "Range (samples)" : self.global_crop.range_size,
            "Tomogram Tag"    : self.tomogram_tag,
            "Pass Labels"     : ", ".join(self.pass_labels) if self.pass_labels else "unavailable",
            "Parameters"      : self.parameters_path,
        })

    @property
    def profile_length(self) -> int:
        tomogram = np.load(str(self.artifact_path("tomogram_full")), mmap_mode="r", allow_pickle=False)
        return int(tomogram.shape[0])

    def artifact_path(self, artifact_key: str) -> Path:
        if artifact_key == "parameters":
            return self.parameters_path

        return self.data_directory / self.artifacts[artifact_key]

    def secondary_indices(self, secondary_labels) -> list | None:
        if secondary_labels is None:
            return None

        if not self.pass_labels:
            raise ValueError("Dataset records no pass labels in dataset.json; baseline extraction must succeed during pre-processing before secondaries can be selected by label.")

        return SecondarySelection.indices(self.pass_labels, secondary_labels)
