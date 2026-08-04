from __future__ import annotations

from pathlib import Path

from configuration.sar.processing_config import PathConfig, TomogramConfig
from tools.data.io                       import FileIO
from tools.monitoring.logger             import Logger


class GeneratorBase:
    def __init__(self, spec: dict, logger: Logger) -> None:
        self.spec   = spec
        self.logger = logger

    @classmethod
    def from_spec_file(cls, spec_path: str | Path, logger: Logger) -> "GeneratorBase":
        return cls(FileIO.load_json(Path(spec_path)), logger)

    def _paths(self) -> PathConfig:
        return PathConfig(
            main_directory   = Path(self.spec["main_directory"]),
            pyrat_directory  = Path(self.spec["pyrat_directory"]),
            run_subdirectory = self.spec["run_subdirectory"],
        )

    def _tomogram_config(self) -> TomogramConfig:
        return TomogramConfig(**self.spec["tomogram_config"])
