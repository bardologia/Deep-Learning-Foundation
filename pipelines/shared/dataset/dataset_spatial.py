
from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io           import FileIO
from tools.data.regions      import CropRegion
from tools.monitoring.logger import Logger


class Layout:

    LAYOUT_RELPATH = Path("data") / "dataset.json"

    def __init__(self, run_directory: Path, logger: Logger, parameters_path: Path) -> None:
        self.run_directory   = Path(run_directory)
        self.logger          = logger
        self.data_directory  = self.run_directory / "data"
        self.parameters_path = Path(parameters_path)

        layout_path = self.run_directory / self.LAYOUT_RELPATH
        payload     = FileIO.load_json(layout_path)

        self.global_crop    : CropRegion  = CropRegion(*payload["global_crop"])
        self.cube_tag       : str         = payload["cube_tag"]
        self.artifacts      : dict        = payload["artifacts"]
        self.channel_labels : list | None = payload["channel_labels"]

        self.logger.section("[Layout Loaded]")
        self.logger.kv_table({
            "Run Directory" : self.run_directory,
            "Global Crop"   : self.global_crop.as_tuple(),
            "Rows"          : self.global_crop.row_size,
            "Cols"          : self.global_crop.col_size,
            "Cube Tag"      : self.cube_tag,
            "Channel Labels": ", ".join(self.channel_labels) if self.channel_labels else "unavailable",
            "Parameters"    : self.parameters_path,
        })

    @property
    def profile_length(self) -> int:
        cube = np.load(str(self.artifact_path("cube_full")), mmap_mode="r", allow_pickle=False)
        return int(cube.shape[0])

    def artifact_path(self, artifact_key: str) -> Path:
        if artifact_key == "parameters":
            return self.parameters_path

        return self.data_directory / self.artifacts[artifact_key]

    def secondary_indices(self, secondary_labels) -> list | None:
        if secondary_labels is None:
            return None

        secondaries = list(self.channel_labels[1:])
        requested   = [str(label) for label in secondary_labels]

        return [secondaries.index(label) for label in requested]

    @classmethod
    def secondary_channel_labels(cls, run_directory) -> list:
        layout_path = Path(run_directory) / cls.LAYOUT_RELPATH

        return list(FileIO.load_json(layout_path)["channel_labels"][1:])
