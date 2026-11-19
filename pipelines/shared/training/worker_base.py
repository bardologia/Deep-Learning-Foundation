
from __future__ import annotations

from pathlib import Path


class WorkerBase:

    def __init__(self, config, run_tag: str) -> None:
        self.config  = config
        self.run_tag = run_tag
        self.run_dir = Path(config.paths.log_base_dir) / run_tag

    def skip_complete(self, run_directory: Path, log_name: str, resume: bool) -> bool:
        from tools.monitoring.logger  import Logger
        from tools.runtime.completion import CompletionMarker

        if not CompletionMarker.is_complete(run_directory):
            return False

        if not resume:
            raise SystemExit(f"{run_directory} already holds a completed run and resume is off; relaunch under a new run_tag to train it again, or delete the directory first, so a finished run is never overwritten in place")

        logger = Logger(log_dir=str(run_directory / "logs"), name=log_name, level="INFO")
        logger.info(f"{run_directory.name} already holds a completed run; this resume leaves it as it is")
        logger.close()

        return True
