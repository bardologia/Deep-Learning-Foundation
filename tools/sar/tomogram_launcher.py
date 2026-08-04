from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path
from typing      import Optional, Tuple, TYPE_CHECKING

from tools.runtime.conda_env import CondaJobDispatcher
from tools.monitoring.logger import Logger

if TYPE_CHECKING:
    from configuration.sar.processing_config import ProcessingConfig


class TomogramLauncher:
    ENTRY = "main/processing/generate_tomogram.py"

    def __init__(self, env_name: str, logger: Logger, repo_root: Optional[Path] = None) -> None:
        self.logger     = logger
        self.dispatcher = CondaJobDispatcher(env_name, logger, repo_root)

    @staticmethod
    def build_spec(config: "ProcessingConfig", tomogram_path: Path, dem_path: Path) -> dict:
        return {
            "tomogram_config"  : asdict(config.tomogram_config),
            "stack_identifier" : config.stack_identifier,
            "dataset_type"     : config.dataset_type,
            "pyrat_directory"  : str(config.paths.pyrat_directory),
            "main_directory"   : str(config.paths.main_directory),
            "run_subdirectory" : config.paths.run_subdirectory,
            "effort"           : config.parallel.effort,
            "crop"             : list(config.crop.as_tuple()),
            "tomogram_path"    : str(tomogram_path),
            "dem_path"         : str(dem_path),
        }

    def generate(self, spec: dict, spec_path: Path) -> Tuple[Path, Path]:
        self.dispatcher.dispatch(self.ENTRY, spec, spec_path)
        return Path(spec["tomogram_path"]), Path(spec["dem_path"])
