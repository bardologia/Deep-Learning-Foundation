from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

import numpy as np

from tools.data.io      import FileIO
from tools.data.regions import CropRegion, SplitRegions


@dataclass
class DatasetFiles:
    LAYOUT  = "dataset.json"
    INPUTS  = "inputs.npy"
    TARGETS = "targets.npy"

    directory       : Path
    name            : str
    input_channels  : list[str]
    target_channels : list[str]
    rows            : int
    cols            : int
    splits          : SplitRegions
    generator       : dict

    @classmethod
    def read(cls, directory) -> "DatasetFiles":
        directory = Path(directory)
        payload   = FileIO.load_json(directory / cls.LAYOUT)

        return cls(
            directory       = directory,
            name            = str(payload["name"]),
            input_channels  = list(payload["input_channels"]),
            target_channels = list(payload["target_channels"]),
            rows            = int(payload["rows"]),
            cols            = int(payload["cols"]),
            splits          = SplitRegions.from_bounds(payload["splits"]),
            generator       = dict(payload["generator"]),
        )

    @classmethod
    def is_dataset(cls, directory) -> bool:
        return (Path(directory) / cls.LAYOUT).is_file()

    def payload(self) -> dict:
        return {
            "name"            : self.name,
            "input_channels"  : list(self.input_channels),
            "target_channels" : list(self.target_channels),
            "rows"            : self.rows,
            "cols"            : self.cols,
            "splits"          : self.splits.bounds(),
            "generator"       : self.generator,
        }

    def write(self, inputs: np.ndarray, targets: np.ndarray) -> Path:
        self.directory.mkdir(parents=True)

        np.save(self.directory / self.INPUTS,  np.asarray(inputs,  dtype=np.float32))
        np.save(self.directory / self.TARGETS, np.asarray(targets, dtype=np.float32))

        return FileIO.save_json(self.payload(), self.directory / self.LAYOUT)

    def inputs(self) -> np.ndarray:
        return np.load(self.directory / self.INPUTS, mmap_mode="r")

    def targets(self) -> np.ndarray:
        return np.load(self.directory / self.TARGETS, mmap_mode="r")

    @staticmethod
    def crop(array: np.ndarray, region: CropRegion) -> np.ndarray:
        return np.ascontiguousarray(array[:, region.row_start:region.row_end, region.col_start:region.col_end], dtype=np.float32)
