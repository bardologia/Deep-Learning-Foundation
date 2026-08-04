from __future__ import annotations

import dataclasses

from datetime import datetime
from pathlib  import Path

import numpy as np

from configuration.synthesis import SynthesisConfig
from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


class DatasetWriter:

    ARTIFACTS = {
        "primary"        : "primary.npy",
        "secondaries"    : "secondaries.npy",
        "interferograms" : "interferograms.npy",
        "dem_full"       : "dem_full.npy",
        "tomogram_full"  : "tomogram_full.npy",
    }

    SOURCE = "synthesis"

    def __init__(self, config: SynthesisConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def _save_array(self, name: str, array: np.ndarray) -> Path:
        path = self.config.data_directory / self.ARTIFACTS[name]
        np.save(str(path), np.ascontiguousarray(array), allow_pickle=False)

        self.logger.subsection(f"-> {path}  {array.shape}  {array.dtype}  ({array.nbytes / 1e6:.1f} MB)")
        return path

    def write_arrays(self, arrays: dict) -> None:
        FileIO.ensure_dir(self.config.data_directory)

        for name in self.ARTIFACTS:
            self._save_array(name, arrays[name])

    def write_layout(self) -> Path:
        payload = {
            "global_crop"        : list(self.config.global_crop),
            "dataset_type"       : self.config.dataset_type,
            "tomogram_tag"       : self.config.tomogram_tag,
            "parameter_tag"      : self.config.parameter_tag,
            "max_amplitude_clip" : self.config.max_amplitude_clip,
            "pass_labels"        : self.config.pass_labels,
            "artifacts"          : dict(self.ARTIFACTS),
        }

        return FileIO.save_json(payload, self.config.data_directory / "dataset.json")

    def write_state(self) -> Path:
        crop    = self.config.global_crop
        payload = {
            "crop" : {
                "azimuth_start" : int(crop[0]),
                "azimuth_end"   : int(crop[1]),
                "range_start"   : int(crop[2]),
                "range_end"     : int(crop[3]),
            },
            "tomogram_config" : {
                "height_range" : list(self.config.height_range),
                "n_heights"    : int(self.config.profile_length),
            },
            "synthesis_config" : {key: str(value) if isinstance(value, Path) else value for key, value in dataclasses.asdict(self.config).items()},
        }

        return FileIO.save_json(payload, self.config.metadata_directory / "config_state.json")

    def write_parameters(self, parameters: np.ndarray, diagnostics: dict) -> dict:
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
        config = self.config

        return {
            "timestamp"           : datetime.now().isoformat(timespec="seconds"),
            "source"              : self.SOURCE,
            "processed_data_path" : str(config.output_directory),
            "source_tomogram"     : str(config.data_directory / self.ARTIFACTS["tomogram_full"]),
            "output_directory"    : str(config.parameters_directory),
            "parameters_npy"      : parameters_path.name,
            "diagnostics_npz"     : diagnostics_path.name,
            "height_range"        : list(config.height_range),
            "k_max"               : int(config.k_max),
            "lambda_k"            : float(config.lambda_k),
            "activity_threshold"  : float(config.activity_threshold),
            "threshold_factor"    : float(config.threshold_factor),
            "truncation_index"    : int(config.truncation_index),
        }
