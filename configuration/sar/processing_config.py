from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import ClassVar, Dict, List, Optional, Tuple

from tools.runtime.run_tag import RunTag
from tools.data.regions    import CropRegion


@dataclass
class TomogramConfig:
    fusar_project_path     : str                 = ""
    base_directory         : str                 = "/ste/rnd/"
    polarisation           : str                 = "hv"
    track_selection        : str                 = "*"
    height_range           : Tuple[float, float] = (-20.0, 80.0)
    filter_method          : str                 = "Boxcar"
    filter_arguments       : Dict                = field(default_factory=lambda: {"win": [20, 10]})
    beamforming_method     : str                 = "Capon"
    beamforming_arguments  : List                = field(default_factory=list)
    max_crop_azimuth_width : int                 = 1000
    apply_resampling       : bool                = False
    apply_presumming       : bool                = False
    max_amplitude_clip     : float               = 1.25


@dataclass
class ParallelConfig:
    effort           : str           = "high"
    tomogram_workers : Optional[int] = None
    pyrat_threads    : Optional[int] = None

    EFFORT_FRACTIONS : ClassVar[Dict[str, float]] = {"low": 0.25, "medium": 0.5, "high": 0.8}
    THREAD_CAP       : ClassVar[int]              = 16

    @staticmethod
    def available_cores() -> int:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    def core_budget(self) -> int:
        if self.effort not in self.EFFORT_FRACTIONS:
            raise ValueError(f"Unknown effort '{self.effort}', expected one of {sorted(self.EFFORT_FRACTIONS)}")
        return max(1, int(self.available_cores() * self.EFFORT_FRACTIONS[self.effort]))

    def interferogram_threads(self) -> int:
        if self.pyrat_threads is not None:
            return max(1, self.pyrat_threads)
        return self.core_budget()

    def resolve_plan(self, subsection_count: int) -> Tuple[int, int]:
        budget = self.core_budget()

        if self.tomogram_workers is not None and self.pyrat_threads is not None:
            return max(1, min(subsection_count, self.tomogram_workers)), max(1, self.pyrat_threads)

        if self.tomogram_workers is not None:
            workers = max(1, min(subsection_count, self.tomogram_workers))
            return workers, max(1, min(self.THREAD_CAP, budget // workers))

        if self.pyrat_threads is not None:
            threads = max(1, self.pyrat_threads)
            return max(1, min(subsection_count, budget // threads)), threads

        best_plan  = None
        best_waves = None

        for workers in range(1, min(subsection_count, budget) + 1):
            threads = max(1, min(self.THREAD_CAP, budget // workers))
            waves   = math.ceil(subsection_count / workers)

            if best_plan is None or waves < best_waves or (waves == best_waves and threads > best_plan[1]):
                best_plan, best_waves = (workers, threads), waves

        return best_plan


@dataclass
class PathConfig:
    main_directory         : Path          = field(default_factory=lambda: Path("/ste/rnd/User/vice_vi/Dataset"))
    pyrat_directory        : Path          = field(default_factory=lambda: Path("/ste/rnd/User/vice_vi/pyrat"))
    data_subdirectory      : str           = "data"
    metadata_subdirectory  : str           = "meta"
    temporary_subdirectory : str           = "tmp"
    run_subdirectory       : Optional[str] = None

    @property
    def run_directory(self) -> Path:
        if self.run_subdirectory is None:
            return self.main_directory
        return self.main_directory / self.run_subdirectory

    @property
    def data_directory(self) -> Path:
        return self.run_directory / self.data_subdirectory

    @property
    def metadata_directory(self) -> Path:
        return self.run_directory / self.metadata_subdirectory

    @property
    def temporary_directory(self) -> Path:
        return self.run_directory / self.temporary_subdirectory


@dataclass
class ProcessingConfig:
    crop            : CropRegion
    tomogram_config : TomogramConfig = field(default_factory=TomogramConfig)
    parallel        : ParallelConfig = field(default_factory=ParallelConfig)
    paths           : PathConfig     = field(default_factory=PathConfig)

    dataset_type         : str = "FSAR"
    stack_identifier     : str = "1"
    tomogram_output_tag  : str = "Xtomo_id2X"
    parameter_output_tag : str = "Xparams_id2X"
    tomogram_env_name    : str = "stetools"

    @property
    def tomogram_tag(self) -> str:
        return f"{self.crop.as_identifier_string()}_{self.stack_identifier}_{self.tomogram_output_tag}"

    @property
    def parameter_tag(self) -> str:
        return f"{self.crop.as_identifier_string()}_{self.stack_identifier}_{self.parameter_output_tag}"

    def __post_init__(self) -> None:
        if self.paths.run_subdirectory is None:
            timestamp = RunTag.now()
            self.paths.run_subdirectory = f"run_{self.tomogram_tag}_{timestamp}"

@dataclass
class PreProcessEntryConfig:
    azimuth_start : int = 1000
    azimuth_end   : int = 16000
    range_start   : int = 500
    range_end     : int = 4000

    fusar_project_path : str = "/ste/rnd/User/sera_se/17sartom-traun_L.csv"
    base_directory     : str = "/ste/rnd/"
    track_selection    : str = "*"
    polarisation       : str = "hv"

    beamforming_method : str   = "Capon"
    filter_method      : str   = "Boxcar"
    height_range       : tuple = (-20.0, 80.0)
    win_list           : list  = field(default_factory=lambda: [[20, 10]])

    max_crop_azimuth_width : int   = 1000
    apply_resampling       : bool  = False
    apply_presumming       : bool  = False
    max_amplitude_clip     : float = 1.25

    effort           : str           = "high"
    tomogram_workers : Optional[int] = None
    pyrat_threads    : Optional[int] = None

    dataset_name         : Optional[str] = None
    dataset_type         : str           = "FSAR"
    stack_identifier     : str           = "1"
    tomogram_output_tag  : str           = "Xtomo_id2X"
    parameter_output_tag : str           = "Xparams_id2X"
    tomogram_env_name    : str           = "stetools"

    def resolve_dataset_name(self, win: List[int], run_identifier: str) -> str:
        win_string = "_".join(str(value) for value in win)

        if self.dataset_name:
            return self.dataset_name if len(self.win_list) == 1 else f"{self.dataset_name}_w{win_string}"

        crop   = CropRegion(azimuth_start=self.azimuth_start, azimuth_end=self.azimuth_end, range_start=self.range_start, range_end=self.range_end)
        source = Path(self.fusar_project_path).stem

        return f"{source}_{crop.as_labeled_string()}_w{win_string}_{self.polarisation}_{self.stack_identifier}_{run_identifier}"


@dataclass
class PreprocessInferenceConfig:
    runs_dir : Path = Path("/ste/rnd/User/vice_vi/Dataset")
    run_tags : list = field(default_factory=list)
