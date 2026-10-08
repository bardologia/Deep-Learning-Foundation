"""Discovery and layout of inference output stamps under a training run."""

from __future__ import annotations

from pathlib import Path

from tools.data.io                     import FileIO
from tools.runtime.completion          import CompletionMarker
from tools.runtime.environment_capture import EnvironmentCapture
from tools.runtime.run_tag             import RunTag


class CompletedInference:
    """Finds the completed inference stamps of a run for a given data split.

    Attributes:
        CHECKPOINT_KEY : Completion-marker entry recording the checkpoint a stamp was scored with.
    """

    CHECKPOINT_KEY = "checkpoint"

    @staticmethod
    def stamps(base: Path, split: str) -> list[Path]:
        """Returns the completed inference stamps scored on one split, oldest first.

        Args:
            base  : Inference directory of a run (or one of its sub-families).
            split : Data split the stamp must record ("train", "val", "test").

        Returns:
            Timestamped stamp directories carrying a completion marker for that
            split; empty when the base directory does not exist.

        Raises:
            KeyError : A completion marker predates split recording and must be backfilled.
        """
        base = Path(base)
        if not base.is_dir():
            return []

        completed = [candidate for candidate in sorted(base.iterdir()) if candidate.is_dir() and CompletionMarker.is_complete(candidate)]

        selected = []
        for stamp in completed:
            payload = CompletionMarker.payload(stamp)

            if "split" not in payload:
                raise KeyError(f"inference completion marker at {stamp} carries no 'split' key; re-run scripts/backfill_completion_markers.py to rewrite it from the recorded metrics")
            if payload["split"] == split:
                selected.append(stamp)

        return selected

    @staticmethod
    def latest(base: Path, split: str) -> Path | None:
        """Returns the newest completed stamp for a split, or None when there is none."""
        stamps = CompletedInference.stamps(base, split)

        return stamps[-1] if stamps else None


class InferenceMetadata:
    """Resolves the output paths of one inference run.

    Attributes:
        SUBDIR       : Family sub-directory under the run's inference/ tree.
        output_dir   : Stamp directory holding this inference run's artifacts.
        figures_dir  : Directory for the generated figures.
        logs_dir     : Directory for the run's text logs.
        metrics_path : Path of the metrics JSON file.
        report_path  : Path of the markdown report.
    """

    SUBDIR: str

    def __init__(self, config) -> None:
        """Derives the stamp directory and artifact paths from the inference config.

        Args:
            config : Inference configuration; an explicit output_subdir names the stamp, otherwise the current timestamp is used.
        """
        paths = config.paths

        base = config.run_directory / "inference" / self.SUBDIR

        self.output_dir   = base / config.output_subdir if config.output_subdir else base / RunTag.now()
        self.figures_dir  = self.output_dir / paths.figures_subdir
        self.logs_dir     = self.output_dir / paths.logs_subdir
        self.metrics_path = self.output_dir / paths.metrics_filename
        self.report_path  = self.output_dir / paths.report_filename

    def create_dirs(self) -> None:
        """Creates the stamp, figures and logs directories and records the environment snapshot."""
        FileIO.ensure_dirs(self.output_dir, self.figures_dir, self.logs_dir)
        FileIO.save_json(EnvironmentCapture.snapshot(), self.output_dir / "environment.json")
