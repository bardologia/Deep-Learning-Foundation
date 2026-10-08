"""Writes a synthetic dataset in the on-disk layout the training and inference pipelines read."""

from __future__ import annotations

import dataclasses

from datetime import datetime
from pathlib  import Path

import numpy as np

from configuration.synthesis import SynthesisConfig
from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


class DatasetWriter:
    """Saves the arrays, layout, configuration state and parameter run of a synthetic dataset.

    Attributes:
        ARTIFACTS : Array name to the file it is saved under in data/.
        SOURCE    : Metadata ``source`` value marking a parameter run written by the generator.
        config    : Synthesis configuration naming the output directories.
        logger    : Logger reporting every written file.
    """

    ARTIFACTS = {
        "primary"     : "primary.npy",
        "secondaries" : "secondaries.npy",
        "derived"     : "derived.npy",
        "aux_full"    : "aux_full.npy",
        "cube_full"   : "cube_full.npy",
    }

    SOURCE = "synthesis"

    def __init__(self, config: SynthesisConfig, logger: Logger) -> None:
        """Binds the writer to a synthesis configuration.

        Args:
            config : Synthesis configuration naming the output directories.
            logger : Logger reporting every written file.
        """
        self.config = config
        self.logger = logger

    def _save_array(self, name: str, array: np.ndarray) -> Path:
        """Saves one array under its artifact name and logs its shape, dtype and size."""
        path = self.config.data_directory / self.ARTIFACTS[name]
        np.save(str(path), np.ascontiguousarray(array), allow_pickle=False)

        self.logger.subsection(f"-> {path}  {array.shape}  {array.dtype}  ({array.nbytes / 1e6:.1f} MB)")
        return path

    def write_arrays(self, arrays: dict) -> None:
        """Saves every artifact array into data/.

        Args:
            arrays : Mapping holding one array per ``ARTIFACTS`` key.
        """
        FileIO.ensure_dir(self.config.data_directory)

        for name in self.ARTIFACTS:
            self._save_array(name, arrays[name])

    def write_layout(self) -> Path:
        """Writes data/dataset.json, the layout downstream pipelines read the dataset through.

        Returns:
            Path of the written layout file.
        """
        payload = {
            "global_crop"        : list(self.config.global_crop),
            "dataset_type"       : self.config.dataset_type,
            "cube_tag"           : self.config.cube_tag,
            "parameter_tag"      : self.config.parameter_tag,
            "max_amplitude_clip" : self.config.max_amplitude_clip,
            "channel_labels"     : self.config.channel_labels,
            "artifacts"          : dict(self.ARTIFACTS),
        }

        return FileIO.save_json(payload, self.config.data_directory / "dataset.json")

    def write_state(self) -> Path:
        """Writes meta/config_state.json with the crop, the cube z range and the synthesis settings.

        Returns:
            Path of the written configuration state file.
        """
        crop    = self.config.global_crop
        payload = {
            "crop" : {
                "row_start" : int(crop[0]),
                "row_end"   : int(crop[1]),
                "col_start" : int(crop[2]),
                "col_end"   : int(crop[3]),
            },
            "cube_config" : {
                "z_range" : list(self.config.z_range),
                "n_z"     : int(self.config.profile_length),
            },
            "synthesis_config" : {key: str(value) if isinstance(value, Path) else value for key, value in dataclasses.asdict(self.config).items()},
        }

        return FileIO.save_json(payload, self.config.metadata_directory / "config_state.json")

    def write_parameters(self, parameters: np.ndarray, diagnostics: dict) -> dict:
        """Saves the parameter run: the parameter stack, its diagnostics and its metadata.

        Args:
            parameters  : Parameter stack of shape (3 * k_max, rows, cols).
            diagnostics : Fit diagnostics arrays saved into fit_diagnostics.npz.

        Returns:
            Mapping with the parameter, diagnostics and metadata paths.
        """
        directory = self.config.parameters_directory
        FileIO.ensure_dir(directory)

        parameters_path  = directory / "parameters.npy"
        diagnostics_path = directory / "fit_diagnostics.npz"

        np.save(str(parameters_path), np.ascontiguousarray(parameters), allow_pickle=False)
        np.savez(str(diagnostics_path), **diagnostics)

        meta_path = FileIO.save_json(self._parameter_meta(parameters_path, diagnostics_path), directory / "param_extraction_meta.json")

        self.logger.subsection(f"-> {parameters_path}  {parameters.shape}  {parameters.dtype}")
        self.logger.subsection(f"-> {diagnostics_path}")
        self.logger.subsection(f"-> {meta_path}")

        return {"parameters": parameters_path, "diagnostics": diagnostics_path, "metadata": meta_path}

    def _parameter_meta(self, parameters_path: Path, diagnostics_path: Path) -> dict:
        """Returns the parameter run metadata in the form a fitted run records it."""
        config = self.config

        return {
            "timestamp"           : datetime.now().isoformat(timespec="seconds"),
            "source"              : self.SOURCE,
            "processed_data_path" : str(config.output_directory),
            "source_cube"         : str(config.data_directory / self.ARTIFACTS["cube_full"]),
            "output_directory"    : str(config.parameters_directory),
            "parameters_npy"      : parameters_path.name,
            "diagnostics_npz"     : diagnostics_path.name,
            "z_range"             : list(config.z_range),
            "k_max"               : int(config.k_max),
            "lambda_k"            : float(config.lambda_k),
            "activity_threshold"  : float(config.activity_threshold),
            "threshold_factor"    : float(config.threshold_factor),
            "truncation_index"    : int(config.truncation_index),
        }
