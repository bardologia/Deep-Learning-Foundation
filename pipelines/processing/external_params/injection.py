from __future__ import annotations

import shutil
from datetime import datetime
from pathlib  import Path
from typing   import Tuple

import numpy as np

from pipelines.processing.external_params.assembly import ChannelMapper, ParameterValidator, TargetAssembler
from pipelines.processing.param_extraction.io      import ParameterIO, ParameterRunMeta
from tools.data.gaussians                          import GaussianSlotSorter
from tools.data.io                                 import FileIO
from tools.data.regions                            import CropRegion
from tools.monitoring.logger                       import Logger


class DatasetTarget:

    def __init__(self, dataset_directory: Path) -> None:
        self.dataset_directory = Path(dataset_directory)
        self.global_crop       = self._read_global_crop()
        self.height_range      = self._read_height_range()

    @property
    def name(self) -> str:
        return self.dataset_directory.name

    @property
    def tomogram_path(self) -> Path:
        return self.dataset_directory / "data" / "tomogram_full.npy"

    def _read_global_crop(self) -> CropRegion:
        layout_path = self.dataset_directory / "data" / "dataset.json"

        if not layout_path.is_file():
            raise FileNotFoundError(f"No data/dataset.json under {self.dataset_directory}; the dataset must be pre-processed before external parameters can be injected into it")

        return CropRegion(*FileIO.load_json(layout_path)["global_crop"])

    def _read_height_range(self) -> Tuple[float, float]:
        state_path = self.dataset_directory / "meta" / "config_state.json"

        if not state_path.is_file():
            raise FileNotFoundError(f"No meta/config_state.json under {self.dataset_directory}; the tomogram height axis is needed to check the external heights share this dataset's convention")

        height_range = FileIO.load_json(state_path)["tomogram_config"]["height_range"]

        return (float(height_range[0]), float(height_range[1]))

    def profile_length(self) -> int:
        if not self.tomogram_path.is_file():
            raise FileNotFoundError(f"No data/tomogram_full.npy under {self.dataset_directory}; the dataset must be pre-processed before external parameters can be injected into it")

        return int(np.load(str(self.tomogram_path), mmap_mode="r").shape[0])

    def run_directory(self, output_prefix: str, output_suffix: str) -> Path:
        return self.dataset_directory / "params" / f"{output_prefix}_{output_suffix}"


class ExternalRunWriter:

    def __init__(self, target: DatasetTarget, run_directory: Path, overwrite: bool, logger: Logger) -> None:
        self.target        = target
        self.run_directory = Path(run_directory)
        self.overwrite     = overwrite
        self.logger        = logger
        self.parameter_io  = ParameterIO(logger=logger)

    def _prepare_directory(self) -> None:
        if not self.run_directory.exists():
            self.run_directory.mkdir(parents=True)
            return

        if not self.overwrite:
            raise FileExistsError(f"Parameter run {self.run_directory} already exists; pass overwrite=True to replace it, or choose another output_suffix")

        self.logger.subsection(f"Replacing existing parameter run {self.run_directory}")
        shutil.rmtree(self.run_directory)
        self.run_directory.mkdir(parents=True)

    def _write_parameters(self, parameters: np.ndarray) -> Path:
        return self.parameter_io.save_params(parameters, self.run_directory / "parameters.npy")

    def _write_metadata(self, parameters: np.ndarray, sources: list, mapper: ChannelMapper, activity_threshold: float, origin_note: str) -> Path:
        payload = {
            "timestamp"          : datetime.now().isoformat(timespec="seconds"),
            "source"             : ParameterRunMeta.EXTERNAL_SOURCE,
            "origin_note"        : origin_note,
            "processed_data_path": str(self.target.dataset_directory),
            "global_crop"        : list(self.target.global_crop.as_tuple()),
            "height_range"       : list(self.target.height_range),
            "output_directory"   : str(self.run_directory),
            "parameters_npy"     : "parameters.npy",
            "parameters_shape"   : list(parameters.shape),
            "k_max"              : mapper.n_slots,
            "fitted_gaussians"   : mapper.n_source,
            "padded_slots"       : mapper.n_padded_slots,
            "activity_threshold" : activity_threshold,
            "source_order"       : mapper.source_order,
            "source_files"       : [
                {
                    "path"     : str(source.path),
                    "window"   : source.window.as_identifier_string(),
                    "shape"    : list(source.cube.shape),
                    "dtype"    : str(source.cube.spec.dtype),
                }
                for source in sources
            ],
        }

        meta_path = self.run_directory / ParameterRunMeta.FILENAME
        FileIO.save_json(payload, meta_path)

        self.logger.subsection(f"-> Metadata written: {meta_path}")

        return meta_path

    def run(self, parameters: np.ndarray, sources: list, mapper: ChannelMapper, activity_threshold: float, origin_note: str) -> dict[str, Path]:
        self._prepare_directory()

        npy_path  = self._write_parameters(parameters)
        meta_path = self._write_metadata(parameters, sources, mapper, activity_threshold, origin_note)

        return {
            "parameters_npy"   : npy_path,
            "metadata"         : meta_path,
            "output_directory" : self.run_directory,
        }


class ExternalParamsInjectionPipeline:

    def __init__(self, config, dataset_directory: Path, sources: list, logger: Logger | None = None) -> None:
        self.config  = config
        self.target  = DatasetTarget(dataset_directory)
        self.sources = sources

        log_dir = self.target.dataset_directory / "params" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger        = logger or Logger(log_dir=str(log_dir), name="external_params_injection", level="INFO")
        self.mapper        = ChannelMapper(config.source_order, sources[0].n_gaussians, config.k_slots)
        self.run_directory = self.target.run_directory(config.output_prefix, config.output_suffix)

        self.logger.section("[External Parameter Injection Initialized]")
        self.logger.kv_table({
            "Dataset"          : str(self.target.dataset_directory),
            "Global crop"      : self.target.global_crop.as_labeled_string(),
            "Height range"     : f"{self.target.height_range[0]:g} .. {self.target.height_range[1]:g} m",
            "Profile bins"     : self.target.profile_length(),
            "Sources"          : len(self.sources),
            "Source order"     : config.source_order,
            "Fitted Gaussians" : self.mapper.n_source,
            "Slots written"    : self.mapper.n_slots,
            "Padded slots"     : self.mapper.n_padded_slots,
            "Run directory"    : str(self.run_directory),
        })

    def _stage_assemble(self) -> np.ndarray:
        return TargetAssembler(self.mapper, self.target.global_crop, self.logger).run(self.sources)

    def _stage_sort(self, parameters: np.ndarray) -> np.ndarray:
        self.logger.subsection("Sorting Gaussian slots by ascending mean, inactive slots last")

        return GaussianSlotSorter.by_mean(parameters, self.mapper.n_slots, self.config.activity_threshold)

    def _stage_validate(self, parameters: np.ndarray) -> None:
        ParameterValidator(self.mapper.n_slots, self.config.activity_threshold, self.target.height_range, self.logger).run(parameters)

    def _stage_write(self, parameters: np.ndarray) -> dict[str, Path]:
        writer = ExternalRunWriter(self.target, self.run_directory, self.config.overwrite, self.logger)

        return writer.run(parameters, self.sources, self.mapper, self.config.activity_threshold, self.config.origin_note)

    def run(self) -> dict[str, Path]:
        self.logger.section(f"[External Parameter Injection] {self.target.name}")

        parameters = self._stage_assemble()
        parameters = self._stage_sort(parameters)

        self._stage_validate(parameters)
        outputs = self._stage_write(parameters)

        self.logger.section("[External Parameter Injection Completed]")

        return outputs
