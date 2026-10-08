"""Spatial layout of a preprocessing run: crop extent, artifacts and pass labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.data.io           import FileIO
from tools.data.regions      import CropRegion
from tools.monitoring.logger import Logger


class Layout:
    """Reads dataset.json of a preprocessing run and resolves its artifact paths.

    Attributes:
        run_directory   : Root of the preprocessing run.
        data_directory  : Sub-directory holding the generated arrays.
        parameters_path : Path of the ground-truth Gaussian parameter array.
        global_crop     : Row/col crop the run was generated over.
        cube_tag        : Tag identifying the volumetric focusing settings.
        artifacts       : Mapping of artifact key to filename inside data/.
        channel_labels  : Channel labels of the stack, reference first, or None when unavailable.
    """

    LAYOUT_RELPATH = Path("data") / "dataset.json"

    def __init__(self, run_directory: Path, logger: Logger, parameters_path: Path) -> None:
        """Loads the dataset layout and logs its crop extent and artifact tags.

        Args:
            run_directory   : Root of the preprocessing run.
            logger          : Logger the layout summary is written to.
            parameters_path : Path of the ground-truth Gaussian parameter array.
        """
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
        """Number of z bins per profile, read from the full cube's first axis."""
        cube = np.load(str(self.artifact_path("cube_full")), mmap_mode="r", allow_pickle=False)
        return int(cube.shape[0])

    def artifact_path(self, artifact_key: str) -> Path:
        """Returns the file path of one dataset artifact.

        Args:
            artifact_key : Artifact name from dataset.json; the special key "parameters" resolves to the externally configured parameter file.
        """
        if artifact_key == "parameters":
            return self.parameters_path

        return self.data_directory / self.artifacts[artifact_key]

    def secondary_indices(self, secondary_labels) -> list | None:
        """Maps requested secondary channel labels onto track indices of the stack.

        Args:
            secondary_labels : Pass labels to use as secondaries, or None to keep the dataset's own secondary ordering.

        Returns:
            Track indices in the order the labels were requested, or None when no
            labels were given.

        Raises:
            ValueError : The dataset records no pass labels, so labels cannot be resolved to indices.
        """
        if secondary_labels is None:
            return None

        if not self.channel_labels:
            raise ValueError("Dataset records no channel labels in dataset.json; the dataset must list its channel labels before secondaries can be selected by label.")

        primary     = self.channel_labels[0]
        secondaries = list(self.channel_labels[1:])
        requested   = [str(label) for label in secondary_labels]

        if primary in requested:
            raise ValueError(f"Channel {primary} is the reference and is always included; remove it from secondary_labels")

        unknown = [label for label in requested if label not in secondaries]
        if unknown:
            raise ValueError(f"Unknown secondary labels {unknown}; secondaries are {secondaries}")

        if len(set(requested)) != len(requested):
            raise ValueError(f"secondary_labels contains duplicates: {requested}")

        return [secondaries.index(label) for label in requested]

    @classmethod
    def secondary_channel_labels(cls, run_directory) -> list:
        """Returns every secondary channel label a dataset records in data/dataset.json, in stack order.

        Reads only the layout file, so trial planners can enumerate the
        dataset's channels without loading the dataset or logging a layout.

        Args:
            run_directory : Root of the dataset run.

        Returns:
            The channel labels after the reference channel; empty for a single-channel stack.

        Raises:
            FileNotFoundError : If the dataset has no data/dataset.json.
            ValueError        : If dataset.json records no channel labels.
        """
        layout_path = Path(run_directory) / cls.LAYOUT_RELPATH
        if not layout_path.is_file():
            raise FileNotFoundError(f"{layout_path} does not exist; the dataset's channel labels are read from it, so point paths.dataset_path at a generated dataset")

        channel_labels = FileIO.load_json(layout_path)["channel_labels"]
        if not channel_labels:
            raise ValueError(f"{layout_path} records no channel labels; generate the dataset again so its secondary channels can be enumerated")

        return list(channel_labels[1:])
