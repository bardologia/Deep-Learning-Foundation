
from __future__ import annotations

from pathlib import Path

from pipelines.shared.config.resolved_entry_config import ResolvedEntryConfigWriter
from pipelines.shared.config.run_metadata          import TrainingRunMetadata
from pipelines.shared.training.pretrain_preflight  import PretrainPreflight
from pipelines.shared.training.unit_resume         import UnitResume
from tools.monitoring.logger                       import Logger
from tools.training.checkpoint                     import TrainerState


class RunDirectorySettledMarker:

    RELATIVE_PATH = Path("docs") / "run_directory_settled.marker"

    @classmethod
    def path_for(cls, run_directory) -> Path:
        return Path(run_directory) / cls.RELATIVE_PATH

    @classmethod
    def write(cls, run_directory) -> Path:
        marker = cls.path_for(run_directory)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("settled\n", encoding="utf-8")

        return marker


class SingleTrainRunner:

    def __init__(self, config) -> None:
        self.config        = config
        self.run_directory = None
        self.unit_resume   = None

    def _build_unit_resume(self) -> UnitResume:
        self.unit_resume = UnitResume(self.run_directory, enabled=self.config.resume, trainer_resume=self.config.training.resume, config=self.config)
        return self.unit_resume

    @property
    def label(self) -> str:
        raise NotImplementedError

    def _resolve_run_name(self) -> str:
        raise NotImplementedError

    def _apply_resume_from(self) -> None:
        target = Path(self.config.resume_from).expanduser()
        if not target.is_dir() and not target.parent.is_dir():
            raise FileNotFoundError(f"resume_from points at '{target}', which does not exist and has no parent directory to be created in; paste the full path of the run to resume or to create")

        self.config.logdir          = target.parent
        self.config.run_name        = target.name
        self.config.resume          = True
        self.config.training.resume = TrainerState.path(target).is_file()

    def _resolve_run_directory(self) -> None:
        if self.config.resume_from:
            self._apply_resume_from()
        else:
            self.config.run_name = self._resolve_run_name()

        self.run_directory = Path(self.config.logdir) / self.config.run_name

    def _pretrain_preflight(self) -> None:
        PretrainPreflight(
            pretrain_config = self.config.pretrain,
            training_config = self.config.training,
            build_trainer   = self._build_pretrain_trainer,
            run_directory   = self.run_directory,
            label           = self.label,
        ).run()

    def _build_pretrain_trainer(self, logger):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class EntryConfigTrainRunner(SingleTrainRunner):

    pipeline_class = None

    def _resolve_run_name(self) -> str:
        return TrainingRunMetadata.resolve_name(self.pipeline_class.run_label, self.config.run_name)

    def _build_pretrain_trainer(self, logger):
        work_dir = self.run_directory / "pretrain" / "context"

        return self.pipeline_class(self.config).build_pretrain_trainer(work_dir, logger)

    def _skipped_result(self):
        return None

    def _persist_entry_config(self) -> None:
        with Logger(log_dir=str(self.run_directory / "logs"), name="train_runner") as logger:
            ResolvedEntryConfigWriter.save(self.config, self.run_directory, logger)

    def run(self):
        self._resolve_run_directory()

        if self._build_unit_resume().skip_training():
            return self._skipped_result()

        RunDirectorySettledMarker.write(self.run_directory)

        self._pretrain_preflight()
        self._persist_entry_config()

        return self.pipeline_class(self.config).run()
