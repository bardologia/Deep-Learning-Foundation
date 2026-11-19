
from __future__ import annotations

import optuna

from pipelines.backbone.training.pipeline import TrainingPipeline
from pipelines.backbone.training.trainer  import Trainer


class TrialPruning:

    @staticmethod
    def report(trial: optuna.Trial, val_loss: float, epoch: int, logger) -> None:
        trial.report(val_loss, epoch)
        if not trial.should_prune():
            return

        logger.warning(f"Trial {trial.number} pruned at epoch {epoch + 1}: val_loss={val_loss:.5g} is worse than the median of earlier trials at this step")
        raise optuna.exceptions.TrialPruned(f"pruned at epoch {epoch + 1} with val_loss {val_loss:.5g}")


class TrialTrainer(Trainer):

    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        self._trial = trial
        super().__init__(*args, emit_docs=emit_docs, **kwargs)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        TrialPruning.report(self._trial, val_loss, epoch, self.logger)


class TrialPipeline(TrainingPipeline):

    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        self._trial     = trial
        self._emit_docs = emit_docs
        super().__init__(*args, **kwargs)

    def _make_trainer(self, model, model_cfg, z_axis, norm_stats, run_dir, logger, emit_docs=True, writer=None):
        return TrialTrainer(
            model      = model,
            model_cfg  = model_cfg,
            z_axis     = z_axis,
            config     = self.trainer_config,
            run_dir    = run_dir,
            logger     = logger,
            norm_stats = norm_stats,
            trial      = self._trial,
            emit_docs  = emit_docs and self._emit_docs,
            writer     = writer,
        )
