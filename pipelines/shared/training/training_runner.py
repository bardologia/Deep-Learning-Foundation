from __future__ import annotations

from pathlib import Path

from pipelines.shared.config.run_metadata         import TrainingRunMetadata
from pipelines.shared.training.pretrain_preflight import PretrainPreflight
from pipelines.shared.training.unit_resume        import UnitResume


class SingleTrainRunner:
    def __init__(self, config) -> None:
        self.config        = config
        self.run_directory = None
        self.unit_resume   = None

    def _build_unit_resume(self) -> UnitResume:
        self.unit_resume = UnitResume(self.run_directory, enabled=self.config.resume, trainer_resume=self.config.training.resume)
        return self.unit_resume

    @property
    def label(self) -> str:
        raise NotImplementedError

    def _resolve_run_name(self) -> str:
        raise NotImplementedError

    def _resolve_run_directory(self) -> None:
        self.config.run_name = self._resolve_run_name()
        self.run_directory   = Path(self.config.logdir) / self.config.run_name

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

    def run(self):
        self._resolve_run_directory()

        if self._build_unit_resume().skip_training():
            return self._skipped_result()

        self._pretrain_preflight()

        return self.pipeline_class(self.config).run()
