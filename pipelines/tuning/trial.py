from __future__ import annotations

import optuna

from pipelines.profile_autoencoder.training.trainer  import Trainer as AeTrainer
from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline as AeTrainingPipeline
from pipelines.image_autoencoder.training.trainer    import Trainer as ImageAeTrainer
from pipelines.image_autoencoder.training.pipeline   import TrainingPipeline as ImageAeTrainingPipeline
from pipelines.jepa.training.trainer                 import Trainer as JepaTrainer
from pipelines.jepa.training.pipeline                import TrainingPipeline as JepaTrainingPipeline
from pipelines.backbone.training.pipeline            import TrainingPipeline
from pipelines.backbone.training.trainer             import Trainer


class TrialTrainer(Trainer):
    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, emit_docs=emit_docs, **kwargs)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialPipeline(TrainingPipeline):
    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        self._trial     = trial
        self._emit_docs = emit_docs
        super().__init__(*args, **kwargs)

    def _make_trainer(self, model, model_cfg, x_axis, norm_stats, run_dir, logger, emit_docs=True):
        return TrialTrainer(
            model      = model,
            model_cfg  = model_cfg,
            x_axis     = x_axis,
            config     = self.trainer_config,
            run_dir    = run_dir,
            logger     = logger,
            norm_stats = norm_stats,
            trial      = self._trial,
            emit_docs  = self._emit_docs,
        )


class TrialProfileAeTrainer(AeTrainer):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        super()._after_eval(val_loss, epoch)
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialProfileAePipeline(AeTrainingPipeline):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, run_directory, logger, model, x_axis):
        return TrialProfileAeTrainer(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_directory, logger, trial=self._trial)


class TrialImageAeTrainer(ImageAeTrainer):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        super()._after_eval(val_loss, epoch)
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialImageAePipeline(ImageAeTrainingPipeline):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, run_directory, logger, model, x_axis):
        return TrialImageAeTrainer(model, self.autoencoder_cfg, x_axis, self.trainer_config, run_directory, logger, trial=self._trial)


class TrialJepaTrainer(JepaTrainer):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        super()._after_eval(val_loss, epoch)
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialJepaPipeline(JepaTrainingPipeline):
    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, model, backbone_cfg, x_axis, run_dir, logger, norm_stats, profile_normalizer):
        return TrialJepaTrainer(model, backbone_cfg, x_axis, self.trainer_config, run_dir, logger, norm_stats, profile_normalizer, trial=self._trial)
