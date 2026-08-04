from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path
from typing      import Optional, TYPE_CHECKING

from tools.runtime.conda_env import CondaJobDispatcher
from tools.data.io           import FileIO
from tools.monitoring.logger import Logger

if TYPE_CHECKING:
    from configuration.sar.processing_config import ProcessingConfig


class InterferogramLauncher:
    ENTRY = "main/processing/generate_interferograms.py"

    def __init__(self, env_name: str, logger: Logger, repo_root: Optional[Path] = None) -> None:
        self.logger     = logger
        self.dispatcher = CondaJobDispatcher(env_name, logger, repo_root)

    @staticmethod
    def build_spec(
        config              : "ProcessingConfig",
        primary_path        : Path,
        secondaries_path    : Path,
        interferograms_path : Path,
        baselines_path      : Path,
        profiles_path       : Path,
        parameters_path     : Path,
        result_path         : Path,
    ) -> dict:
        return {
            "tomogram_config"     : asdict(config.tomogram_config),
            "stack_identifier"    : config.stack_identifier,
            "dataset_type"        : config.dataset_type,
            "pyrat_directory"     : str(config.paths.pyrat_directory),
            "main_directory"      : str(config.paths.main_directory),
            "run_subdirectory"    : config.paths.run_subdirectory,
            "effort"              : config.parallel.effort,
            "pyrat_threads"       : config.parallel.pyrat_threads,
            "crop"                : list(config.crop.as_tuple()),
            "primary_path"        : str(primary_path),
            "secondaries_path"    : str(secondaries_path),
            "interferograms_path" : str(interferograms_path),
            "baselines_path"      : str(baselines_path),
            "profiles_path"       : str(profiles_path),
            "parameters_path"     : str(parameters_path),
            "result_path"         : str(result_path),
        }

    def generate(self, spec: dict, spec_path: Path) -> dict:
        self.dispatcher.dispatch(self.ENTRY, spec, spec_path)
        return FileIO.load_json(Path(spec["result_path"]))
