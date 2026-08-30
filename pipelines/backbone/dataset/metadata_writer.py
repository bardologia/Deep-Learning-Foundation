"""Persistence of the dataset-creation metadata of a backbone training run.

Writes the resolved dataset configuration, the crop regions of each split and the
patch grids under the run's meta directory, so inference can rebuild the exact
spatial layout the model was trained on.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path

from configuration.dataset   import DatasetConfig
from tools.data.regions      import CropRegion
from tools.data.io           import FileIO
from tools.monitoring.logger import Logger
from pipelines.backbone.dataset.spatial import GridInfo


class MetadataWriter:
    """Writes the dataset configuration, crop and patch metadata of a training run.

    Attributes:
        run_directory      : Training run directory that owns the metadata.
        metadata_directory : The run's meta subdirectory, created on construction.
        outpaths           : Output path of each metadata file, keyed by its role.
    """

    def __init__(self, run_directory: Path, logger: Logger) -> None:
        """Creates the metadata directory and resolves the output paths.

        Args:
            run_directory : Training run directory to write metadata into.
            logger        : Logger receiving the initialization report.
        """
        self.run_directory      = Path(run_directory)
        self.logger             = logger
        self.metadata_directory = self.run_directory / "meta"

        self.outpaths           = {
            "dataset_configuration" : self.metadata_directory / "dataset_creation_config.json",
            "crop"                  : self.metadata_directory / "crop.json",
            "patch"                 : self.metadata_directory / "patch.json",
        }

        FileIO.ensure_dir(self.metadata_directory)

        self.logger.section("[MetadataWriter Initialized]")
        self.logger.subsection(f"Metadata Directory : {self.metadata_directory} \n")

    def save_dataset_configuration(self, config: DatasetConfig) -> Path:
        """Serializes the dataset configuration to JSON.

        The preprocessing run path is stringified, the input and output configurations
        are expanded through their own serializers and the derived Gaussian x-axis is
        dropped, since it is rebuilt from the profile length.

        Args:
            config : Resolved dataset configuration of the run.

        Returns:
            Path of the written dataset_creation_config.json.
        """
        out_path = self.outpaths["dataset_configuration"]
        payload  = asdict(config)

        payload["preprocessing_run_directory"] = str(config.preprocessing_run_directory)
        payload["input_config"]                = config.input_config.as_dict()
        payload["output_config"]               = config.output_config.as_dict()
        payload.pop("z_axis", None)

        return FileIO.save_json(payload, out_path)

    def save_crop_metadata(self, global_crop: CropRegion, splits: dict[str, CropRegion]) -> Path:
        """Writes the preprocessing global crop and the per-split crop regions.

        Args:
            global_crop : Crop region the preprocessing artifacts cover.
            splits      : Crop region, or list of regions, of each split keyed by split name.

        Returns:
            Path of the written crop.json.
        """
        out_path = self.outpaths["crop"]
        payload  = {"global_crop" : list(global_crop.as_tuple()), "splits" : {name: self._region_payload(value) for name, value in splits.items()}}

        return FileIO.save_json(payload, out_path)

    def save_patch_metadata(self, grids: dict[str, GridInfo]) -> Path:
        """Writes the patch grid geometry of each split.

        Args:
            grids : Grid, or list of grids, of each split keyed by split name.

        Returns:
            Path of the written patch.json.
        """
        out_path = self.outpaths["patch"]
        payload  = {name: self._grid_payload(value) for name, value in grids.items()}

        return FileIO.save_json(payload, out_path)

    def _region_payload(self, value):
        """Returns one crop region, or a list of them, as plain coordinate lists."""
        if isinstance(value, (list, tuple)):
            return [list(region.as_tuple()) for region in value]
        return list(value.as_tuple())

    def _grid_payload(self, value):
        """Returns one grid, or a list of them, as JSON-ready dictionaries."""
        if isinstance(value, (list, tuple)):
            return [grid.as_dict() for grid in value]
        return value.as_dict()
