from __future__ import annotations

from datetime import datetime
from pathlib  import Path

import numpy as np

from configuration.param_extraction import ExtractionConfig
from tools.data.io           import FileIO
from tools.monitoring.logger import Logger


class ParameterRunMeta:
    FILENAME        = "param_extraction_meta.json"
    EXTERNAL_SOURCE = "external"

    @classmethod
    def is_external(cls, run_dir: Path) -> bool:
        meta_path = Path(run_dir) / cls.FILENAME

        if not meta_path.is_file():
            return False

        return FileIO.load_json(meta_path).get("source") == cls.EXTERNAL_SOURCE

    @classmethod
    def reject_external(cls, run_dir: Path, action: str) -> None:
        if cls.is_external(run_dir):
            raise ValueError(f"Parameter run {run_dir} was injected from external files, not fitted here, so it carries no fit diagnostics and cannot {action}. Point this step at a run produced by extract_params.")


class ExtractionMetadataManager:
    def __init__(self, config: ExtractionConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    def save_run_metadata(self, npy_path: Path, diagnostics_path: Path, tomogram_path: Path, height_range: tuple) -> Path:
        meta_path = self.config.output_directory / ParameterRunMeta.FILENAME
        ext       = self.config.fit_settings

        payload = {
            "timestamp"            : datetime.now().isoformat(timespec="seconds"),
            "processed_data_path"  : str(self.config.processed_data_path),
            "source_tomogram"      : str(tomogram_path),
            "height_range"         : list(height_range),
            "output_directory"     : str(self.config.output_directory),
            "output_prefix"        : self.config.output_prefix,
            "output_suffix"        : self.config.output_suffix_value,
            "parameters_npy"       : npy_path.name,
            "diagnostics_npz"      : diagnostics_path.name,
            "k_max"                : ext.fit_config.k_max,
            "lambda_k"             : ext.fit_config.lambda_k,
            "sigma_init_divisor"   : ext.fit_config.sigma_init_divisor,
            "prominence_frac"      : ext.fit_config.prominence_frac,
            "activity_threshold"   : ext.fit_config.activity_threshold,
            "threshold_factor"     : ext.fit_config.threshold_factor,
            "truncation_index"     : ext.fit_config.truncation_index,
            "adam_steps"           : self.config.adam_steps,
            "adam_lr"              : self.config.adam_lr,
            "adam_b1"              : self.config.adam_b1,
            "adam_b2"              : self.config.adam_b2,
            "range_batch_size"     : self.config.range_batch_size,
            "gpu_pixel_batch_size" : self.config.gpu_pixel_batch_size,
            "fit_sigma"            : ext.fit_config.fit_sigma,
            "fit_amplitude"        : ext.fit_config.fit_amplitude,
            "fit_mean"             : ext.fit_config.fit_mean,
            "fitting_method"       : ext.fitting_method,
        }

        FileIO.save_json(payload, meta_path)

        self.logger.subsection(f"-> Metadata written: {meta_path}")
        return meta_path


class ParameterIO:
    def __init__(self, logger : Logger) -> None:
        self.logger = logger

    def save_params(self, parameters_array : np.ndarray, npy_path : Path) -> Path:
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.subsection(f"Saving parameter stack of shape {parameters_array.shape} to disk")
        np.save(str(npy_path), np.ascontiguousarray(parameters_array), allow_pickle=False)

        return npy_path

    def load_params(self, npy_path : Path) -> np.ndarray:
        self.logger.subsection("Loading saved parameters for metrics and plots")
        return np.load(str(npy_path)).astype(np.float32, copy=False)

    def save_diagnostics(self, diagnostics : dict, npz_path : Path) -> Path:
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.subsection(f"Saving fit diagnostics ({', '.join(diagnostics.keys())}) to disk")
        np.savez(str(npz_path), **diagnostics)

        return npz_path

    def load_diagnostics(self, npz_path : Path) -> dict:
        self.logger.subsection("Loading fit diagnostics for metrics and plots")
        with np.load(str(npz_path)) as data:
            return {key: data[key] for key in data.files}

    def load_metadata(self, meta_path : Path) -> dict:
        return FileIO.load_json(meta_path)
