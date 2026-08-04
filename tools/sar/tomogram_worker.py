from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from typing      import Dict, List, Optional, Tuple

from tools.sar.pyrat_env import PyRatEnvironment


@dataclass
class PyRatJob:
    pyrat_root_path       : str
    crop_tuple            : Tuple[int, int, int, int]
    suffix                : str
    fusar_project_path    : str
    stack_identifier      : str
    base_directory        : str
    polarisation          : str
    track_selection       : str
    height_range          : Tuple[float, float]
    filter_method         : str
    filter_arguments      : Dict
    beamforming_method    : str
    beamforming_arguments : List
    output_directory      : str
    apply_resampling      : bool
    apply_presumming      : bool
    pyrat_threads         : int
    parent_sys_path       : Optional[list] = None


class PyRatWorker:
    def __init__(self, job: PyRatJob) -> None:
        self.job = job

    def _prepare_environment(self) -> None:
        if self.job.parent_sys_path is not None:
            sys.path[:] = self.job.parent_sys_path

        PyRatEnvironment.ensure(self.job.pyrat_root_path)

    def run(self) -> int:
        self._prepare_environment()

        from pyrat import pyrat_init, tomo
        pyrat_init(debug=True, nthreads=self.job.pyrat_threads, silent=True)

        tomo.fusartomo(
            FuSARproject = self.job.fusar_project_path,
            id           = self.job.stack_identifier,
            basedir      = self.job.base_directory,
            polarisation = self.job.polarisation,
            select       = self.job.track_selection,
            presum       = self.job.apply_presumming,
            crop         = self.job.crop_tuple,
            range        = list(self.job.height_range),
            filter       = self.job.filter_method,
            filargs      = self.job.filter_arguments,
            method       = self.job.beamforming_method,
            args         = self.job.beamforming_arguments,
            suffix       = self.job.suffix,
            dir          = self.job.output_directory,
            resampling   = self.job.apply_resampling,
        )

        gc.collect()

        return 0


def run_pyrat_job(job: PyRatJob) -> int:
    return PyRatWorker(job).run()
