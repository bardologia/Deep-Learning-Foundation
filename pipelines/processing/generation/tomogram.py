from __future__ import annotations

import gc
import shutil
import sys
import tempfile
from pathlib import Path
from typing  import Tuple

import h5py
import numpy as np

from configuration.sar.processing_config  import (
    ParallelConfig,
    ProcessingConfig,
    TomogramConfig,
)
from pipelines.processing.generation.base import GeneratorBase
from tools.sar.pyrat_env                  import PyRatEnvironment
from tools.sar.tomogram_worker            import PyRatJob, run_pyrat_job
from tools.data.io                        import FileIO
from tools.orchestration.pool             import ProcessPoolRunner
from tools.monitoring.logger              import Logger
from tools.data.regions                   import CropRegion


class TomogramProcessor:
    def __init__(self, config: ProcessingConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

        self.logger.section("[TomogramProcessor Initialization]")
        self.logger.subsection(f"Max Azimuth Width : {self.config.tomogram_config.max_crop_azimuth_width}")
        self.logger.subsection(f"Parallel Effort   : {self.config.parallel.effort}")
        self.logger.subsection(f"Core Budget       : {self.config.parallel.core_budget()}")
        self.logger.subsection(f"Cores Available   : {self.config.parallel.available_cores()}")

    def _create_temp(self) -> Path:
        parent = self.config.paths.temporary_directory
        FileIO.ensure_dir(parent)
        temporary_directory = Path(tempfile.mkdtemp(prefix="tomo_", dir=str(parent)))
        return temporary_directory

    def _divide_crop(self, tomogram_config: TomogramConfig) -> list[Tuple[int, int, int, int]]:
        crop      = self.config.crop
        max_width = tomogram_config.max_crop_azimuth_width

        azimuth_start = crop.azimuth_start
        azimuth_end   = crop.azimuth_end
        total_width   = azimuth_end - azimuth_start

        if total_width <= max_width:
            self.logger.subsection(f"Crop width ({total_width}) fits within limit ({max_width}). Single section.")
            return [crop.as_tuple()]

        subsections = [region.as_tuple() for region in crop.subdivide_by_azimuth(max_width)]

        self.logger.subsection(f"Crop subdivided into {len(subsections)} sections.")

        return subsections

    def _dispatch_workers(
        self,
        subsections         : list[Tuple[int, int, int, int]],
        stack_identifier    : str,
        tomogram_config     : TomogramConfig,
        temporary_directory : Path,
    ) -> None:
        PyRatEnvironment.ensure_conda_lib_on_ld_path()

        parallel_config = self.config.parallel
        tasks           = []

        resolved_workers, worker_threads = parallel_config.resolve_plan(len(subsections))

        parent_sys_path = list(sys.path)

        for subsection_index, subsection_crop in enumerate(subsections):
            tasks.append(PyRatJob(
                pyrat_root_path       = str(self.config.paths.pyrat_directory),
                crop_tuple            = subsection_crop,
                suffix                = f"{subsection_index:04d}",
                fusar_project_path    = tomogram_config.fusar_project_path,
                stack_identifier      = stack_identifier,
                base_directory        = tomogram_config.base_directory,
                polarisation          = tomogram_config.polarisation,
                track_selection       = tomogram_config.track_selection,
                height_range          = tomogram_config.height_range,
                filter_method         = tomogram_config.filter_method,
                filter_arguments      = tomogram_config.filter_arguments,
                beamforming_method    = tomogram_config.beamforming_method,
                beamforming_arguments = tomogram_config.beamforming_arguments,
                output_directory      = str(temporary_directory),
                apply_resampling      = tomogram_config.apply_resampling,
                apply_presumming      = tomogram_config.apply_presumming,
                pyrat_threads         = worker_threads,
                parent_sys_path       = parent_sys_path,
            ))

        self.logger.subsection(f"Dispatching {len(tasks)} PyRat jobs across {resolved_workers} workers ({worker_threads} threads each, budget {parallel_config.core_budget()} of {parallel_config.available_cores()} cores)")

        runner    = ProcessPoolRunner(logger=self.logger, max_workers=resolved_workers)
        completed = runner.run(tasks, run_pyrat_job)

        for job, _ in completed:
            self.logger.subsection(f"PyRat job completed: subsection {job.suffix}")

    def _concatenate(self, temporary_directory: Path) -> Tuple[np.ndarray, np.ndarray]:
        partial_files_directory = temporary_directory / "TOMO" / "TOMO-SR"
        partial_file_paths      = sorted(partial_files_directory.iterdir())

        self.logger.subsection(f"[Concatenation] Merging {len(partial_file_paths)} subsection artifacts")

        dem_shapes      : list[Tuple[int, ...]] = []
        tomogram_shapes : list[Tuple[int, ...]] = []
        dem_dtype      = None
        tomogram_dtype = None

        for partial_file_path in partial_file_paths:
            with h5py.File(str(partial_file_path), "r") as hdf5_file:
                dem_shapes.append(hdf5_file["DEM"].shape)
                tomogram_shapes.append(hdf5_file["tomogram"].shape)
                dem_dtype      = hdf5_file["DEM"].dtype
                tomogram_dtype = hdf5_file["tomogram"].dtype

        AZIMUTH_AXIS = 1

        combined_dem_shape      = (sum(shape[0] for shape in dem_shapes),) + dem_shapes[0][1:]
        combined_tomogram_shape = tomogram_shapes[0][:AZIMUTH_AXIS] + (sum(shape[AZIMUTH_AXIS] for shape in tomogram_shapes),) + tomogram_shapes[0][AZIMUTH_AXIS + 1:]

        combined_dem      = np.empty(combined_dem_shape,      dtype=dem_dtype)
        combined_tomogram = np.empty(combined_tomogram_shape, dtype=tomogram_dtype)

        dem_offset      = 0
        tomogram_offset = 0

        for partial_file_path in partial_file_paths:
            with h5py.File(str(partial_file_path), "r") as hdf5_file:
                dem_chunk      = hdf5_file["DEM"][:]
                tomogram_chunk = hdf5_file["tomogram"][:]

            dem_width      = dem_chunk.shape[0]
            tomogram_width = tomogram_chunk.shape[1]

            combined_dem[dem_offset:dem_offset + dem_width]                 = dem_chunk
            combined_tomogram[:, tomogram_offset:tomogram_offset + tomogram_width] = tomogram_chunk

            dem_offset      += dem_width
            tomogram_offset += tomogram_width

            del dem_chunk, tomogram_chunk

        self.logger.subsection(f"Combined DEM shape      : {combined_dem.shape}")
        self.logger.subsection(f"Combined Tomogram shape : {combined_tomogram.shape}")

        return combined_dem, combined_tomogram

    def _save(self, tomogram_path: Path, dem_path: Path, tomogram_array: np.ndarray, dem_array: np.ndarray) -> None:
        FileIO.ensure_dir(tomogram_path.parent)
        np.save(str(tomogram_path), tomogram_array, allow_pickle=False)
        np.save(str(dem_path),      dem_array,      allow_pickle=False)

    def _cleanup_temp(self, temporary_directory: Path) -> None:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def run(self, tomogram_path: Path, dem_path: Path, stack_identifier: str, tomogram_config: TomogramConfig) -> Tuple[Path, Path]:
        self.logger.section("[Generating Tomogram]")
        self.logger.subsection(f"Target: {tomogram_path.name}")

        temporary_directory = self._create_temp()

        try:
            subsections = self._divide_crop(tomogram_config)
            self._dispatch_workers(subsections, stack_identifier, tomogram_config, temporary_directory)
            combined_dem, combined_tomogram = self._concatenate(temporary_directory)
            self._save(tomogram_path, dem_path, combined_tomogram, combined_dem)

            self.logger.subsection(f"Tomogram saved : {tomogram_path}")
            self.logger.subsection(f"DEM saved      : {dem_path}")
        finally:
            self._cleanup_temp(temporary_directory)
            self.logger.subsection("Temporary directory cleaned up. \n")
            gc.collect()

        return tomogram_path, dem_path


class TomogramGenerator(GeneratorBase):
    def _build_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            crop             = CropRegion(*self.spec["crop"]),
            tomogram_config  = self._tomogram_config(),
            parallel         = ParallelConfig(effort=self.spec["effort"]),
            paths            = self._paths(),
            dataset_type     = self.spec["dataset_type"],
            stack_identifier = self.spec["stack_identifier"],
        )

    def run(self) -> None:
        config = self._build_config()

        self.logger.section("[Tomogram Generation]")
        self.logger.kv_table({
            "Stack id"        : config.stack_identifier,
            "Track selection" : config.tomogram_config.track_selection,
            "Crop"            : config.crop.as_tuple(),
            "Output"          : self.spec["tomogram_path"],
        })

        TomogramProcessor(config, logger=self.logger).run(
            tomogram_path    = Path(self.spec["tomogram_path"]),
            dem_path         = Path(self.spec["dem_path"]),
            stack_identifier = config.stack_identifier,
            tomogram_config  = config.tomogram_config,
        )
