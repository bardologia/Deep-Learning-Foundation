from __future__ import annotations

import shutil
from pathlib import Path

from tools.monitoring.logger  import Logger
from tools.runtime.completion import CompletionMarker


class UnitResume:
    def __init__(self, run_directory: Path, enabled: bool, trainer_resume: bool = False) -> None:
        self.run_directory  = Path(run_directory)
        self.enabled        = enabled
        self.trainer_resume = trainer_resume

    def _log(self, message: str) -> None:
        logger = Logger(log_dir=str(self.run_directory / "logs"), name="unit_resume", level="INFO")
        logger.info(message)
        logger.close()

    def skip_training(self) -> bool:
        if not self.enabled:
            return False

        if CompletionMarker.is_complete(self.run_directory):
            self._log(f"{self.run_directory.name}: completed training reused")
            return True

        if self.trainer_resume:
            return False

        if self.run_directory.is_dir():
            shutil.rmtree(self.run_directory)
            self._log(f"{self.run_directory.name}: unfinished run deleted, restarting from scratch")

        return False

    def skip_inference(self, split: str) -> bool:
        if not self.enabled:
            return False

        inference_dir = self.run_directory / "inference"
        if not inference_dir.is_dir():
            return False

        for candidate in sorted(inference_dir.iterdir()):
            if candidate.is_dir() and not CompletionMarker.is_complete(candidate):
                shutil.rmtree(candidate)
                self._log(f"{self.run_directory.name}: unfinished inference '{candidate.name}' deleted")

        completed = [candidate for candidate in inference_dir.iterdir() if candidate.is_dir() and CompletionMarker.is_complete(candidate)]

        if any(CompletionMarker.payload(candidate)["split"] == split for candidate in completed):
            self._log(f"{self.run_directory.name}: existing '{split}' inference reused")
            return True

        return False
