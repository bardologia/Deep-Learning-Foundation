"""Runners that execute one training run, from run naming to pipeline call.

Resolve the run directory (fresh or resumed), consult the completion marker to
skip or restart it, mark the directory as settled for anything waiting on it,
run the pre-training preflight, persist the resolved entry config the preflight
has finished tuning, and hand off to the training pipeline.
"""

from __future__ import annotations

from pathlib import Path

from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.config.run_metadata          import TrainingRunMetadata
from pipelines.shared.training.pretrain_preflight  import PretrainPreflight
from pipelines.shared.training.unit_resume         import UnitResume
from tools.monitoring.logger                       import Logger
from tools.training.checkpoint                     import TrainerState


class RunDirectorySettledMarker:
    """Marker announcing that a run's resume gate has settled its run directory.

    A fresh run's resume gate deletes a stale unfinished run directory before
    re-creating it, so nothing outside rank zero may write into that directory
    until the gate has passed. This marker is the first file written after it,
    and is what the DDP fan-out waits on before launching the other ranks.

    Attributes:
        RELATIVE_PATH : Location of the marker inside a run directory.
    """

    RELATIVE_PATH = Path("docs") / "run_directory_settled.marker"

    @classmethod
    def path_for(cls, run_directory) -> Path:
        """Returns the marker path of a run directory.

        Args:
            run_directory : Run directory owning the marker.

        Returns:
            Absolute path of the marker file in that run.
        """
        return Path(run_directory) / cls.RELATIVE_PATH

    @classmethod
    def write(cls, run_directory) -> Path:
        """Creates the marker, and the docs directory holding it, in a settled run.

        Args:
            run_directory : Run directory whose resume gate has just passed.

        Returns:
            Path of the written marker.
        """
        marker = cls.path_for(run_directory)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("settled\n", encoding="utf-8")

        return marker


class SingleTrainRunner:
    """Base runner for a single training run of one entry configuration.

    Attributes:
        config        : Entry configuration of the run.
        run_directory : Resolved run directory, set once naming has run.
        unit_resume   : Resume policy for this run directory, set once built.
    """

    def __init__(self, config) -> None:
        """Stores the entry configuration with the run directory unresolved."""
        self.config        = config
        self.run_directory = None
        self.unit_resume   = None

    def _build_unit_resume(self) -> UnitResume:
        """Builds and stores the resume policy for the resolved run directory."""
        self.unit_resume = UnitResume(self.run_directory, enabled=self.config.resume, trainer_resume=self.config.training.resume, config=self.config)
        return self.unit_resume

    @property
    def label(self) -> str:
        """Human-readable label of the run; implemented by subclasses."""
        raise NotImplementedError

    def _resolve_run_name(self) -> str:
        """Returns the run name for a fresh run; implemented by subclasses."""
        raise NotImplementedError

    def _apply_resume_from(self) -> None:
        """Points the config at the pasted directory, which is the run directory exactly.

        The directory is used as it is, bypassing the naming scheme: an
        existing one is reused when complete and continued from its trainer
        state when it holds one; an existing one without a trainer state is
        restarted from scratch, and an absent one is created fresh at that
        path, which is how a sweep hands a job the directory it expects.

        Raises:
            FileNotFoundError : If neither ``resume_from`` nor its parent directory exists.
        """
        target = Path(self.config.resume_from).expanduser()
        if not target.is_dir() and not target.parent.is_dir():
            raise FileNotFoundError(f"resume_from points at '{target}', which does not exist and has no parent directory to be created in; paste the full path of the run to resume or to create")

        self.config.logdir          = target.parent
        self.config.run_name        = target.name
        self.config.resume          = True
        self.config.training.resume = TrainerState.path(target).is_file()

    def _resolve_run_directory(self) -> None:
        """Sets the run directory, either from ``resume_from`` or from naming."""
        if self.config.resume_from:
            self._apply_resume_from()
        else:
            self.config.run_name = self._resolve_run_name()

        self.run_directory = Path(self.config.logdir) / self.config.run_name

    def _pretrain_preflight(self) -> None:
        """Runs the batch-size and loader preflight for this run."""
        PretrainPreflight(
            pretrain_config = self.config.pretrain,
            training_config = self.config.training,
            build_trainer   = self._build_pretrain_trainer,
            run_directory   = self.run_directory,
            label           = self.label,
        ).run()

    def _build_pretrain_trainer(self, logger):
        """Builds (trainer, dataset, model) for the preflight; set by subclasses."""
        raise NotImplementedError

    def run(self):
        """Executes the training run; implemented by subclasses."""
        raise NotImplementedError


class EntryConfigTrainRunner(SingleTrainRunner):
    """Runner driving a training pipeline built straight from the entry config.

    Attributes:
        pipeline_class : Training pipeline class the runner instantiates.
    """

    pipeline_class = None

    def _resolve_run_name(self) -> str:
        """Returns the run name from the pipeline's label and the config override."""
        return TrainingRunMetadata.resolve_name(self.pipeline_class.run_label, self.config.run_name)

    def _build_pretrain_trainer(self, logger):
        """Builds the probe trainer, dataset and model in a throwaway context dir.

        Args:
            logger : Logger the probe trainer writes into.

        Returns:
            The (trainer, dataset, model) triple the preflight probes.
        """
        work_dir = self.run_directory / "pretrain" / "context"

        return self.pipeline_class(self.config).build_pretrain_trainer(work_dir, logger)

    def _skipped_result(self):
        """Returns the value stood in for a reused, already-complete run."""
        return None

    def _persist_entry_config(self) -> None:
        """Writes the entry config the preflight has finished tuning into the run's docs."""
        logger = Logger(log_dir="", name="train_runner")
        ResolvedEntryConfigWriter.save(self.config, self.run_directory, logger)

    def run(self):
        """Resolves the run directory, preflights, persists the entry config and trains.

        The entry config is written after the preflight, so the file records the
        batch size, loader settings and learning-rate multiplier the run really
        uses rather than the un-probed defaults.

        Returns:
            Whatever the training pipeline returns, or ``_skipped_result()``
            when the run directory already holds a completed run.
        """
        self._resolve_run_directory()

        if self._build_unit_resume().skip_training():
            return self._skipped_result()

        RunDirectorySettledMarker.write(self.run_directory)

        self._pretrain_preflight()
        self._persist_entry_config()

        return self.pipeline_class(self.config).run()
