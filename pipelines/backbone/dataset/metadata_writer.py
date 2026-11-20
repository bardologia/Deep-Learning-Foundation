from __future__ import annotations

from pathlib import Path

from configuration.dataset              import DatasetConfig
from pipelines.backbone.dataset.spatial import GridInfo
from tools.data.io                      import FileIO
from tools.monitoring.logger            import Logger


class MetadataWriter:

    def __init__(self, run_directory: Path, logger: Logger) -> None:
        self.run_directory      = Path(run_directory)
        self.logger             = logger
        self.metadata_directory = self.run_directory / "meta"

        self.outpaths = {
            "dataset_configuration" : self.metadata_directory / "dataset_creation_config.json",
            "patch"                 : self.metadata_directory / "patch.json",
        }

        FileIO.ensure_dir(self.metadata_directory)

    def save_dataset_configuration(self, config: DatasetConfig) -> Path:
        return FileIO.save_json(config.as_dict(), self.outpaths["dataset_configuration"])

    def save_patch_metadata(self, grids: dict[str, list[GridInfo]]) -> Path:
        payload = {name: [grid.as_dict() for grid in value] for name, value in grids.items()}

        return FileIO.save_json(payload, self.outpaths["patch"])
