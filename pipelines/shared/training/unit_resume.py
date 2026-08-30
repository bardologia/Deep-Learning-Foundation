"""Resume policy for a single training or inference unit.

Decides whether a run directory can be reused, resumed from its checkpoint or
must be wiped and restarted, based on the completion marker written when a run
finishes. Unfinished inference stamps are deleted so partial outputs never look
complete.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pipelines.shared.inference.metadata import CompletedInference
from tools.monitoring.logger             import Logger
from tools.runtime.completion            import CompletionMarker


class UnitResume:
    """Decides whether a unit's training or inference can be skipped.

    Attributes:
        run_directory  : Run directory of the unit.
        enabled        : Whether resume logic applies at all.
        trainer_resume : Whether the trainer itself continues from a checkpoint, which keeps unfinished run directories in place.
        logger         : Logger created lazily on the first message.
    """

    def __init__(self, run_directory: Path, enabled: bool, trainer_resume: bool = False) -> None:
        """Stores the run directory and the resume flags for this unit."""
        self.run_directory  = Path(run_directory)
        self.enabled        = enabled
        self.trainer_resume = trainer_resume
        self.logger         = None

    def skip_training(self) -> bool:
        """Returns whether the completed run can be reused instead of retrained.

        A completed run is reused. An unfinished run is left alone when the
        trainer resumes from its checkpoint, and otherwise deleted so training
        restarts from scratch.
        """
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

    def skip_inference(self, split: str, subdir: str | None = None) -> bool:
        """Returns whether an existing inference for a split can be reused.

        Unfinished inference stamps are deleted first, so only completed
        stamps count as reusable.

        Args:
            split  : Dataset split the inference covers (e.g. ``test``).
            subdir : Optional subdirectory of ``inference`` holding the stamps.

        Returns:
            True when a completed stamp for the split already exists.
        """
        if not self.enabled:
            return False

        inference_dir = self.run_directory / "inference" / subdir if subdir else self.run_directory / "inference"
        if not inference_dir.is_dir():
            return False

        for candidate in sorted(inference_dir.iterdir()):
            if candidate.is_dir() and not CompletionMarker.is_complete(candidate):
                shutil.rmtree(candidate)
                self._log(f"{self.run_directory.name}: unfinished inference '{candidate.name}' deleted")

        if CompletedInference.stamps(inference_dir, split):
            self._log(f"{self.run_directory.name}: existing '{split}' inference reused")
            return True

        return False

    def _log(self, message: str) -> None:
        """Logs a resume decision, creating the appending logger on first use."""
        if self.logger is None:
            self.logger = Logger(log_dir=str(self.run_directory / "logs"), name="unit_resume", level="INFO", file_mode="a")

        self.logger.info(message)
