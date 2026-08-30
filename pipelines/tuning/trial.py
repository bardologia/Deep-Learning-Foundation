"""Trial-aware subclasses of the training pipelines and trainers.

Each subclass reports the validation loss of every evaluated epoch to its
Optuna trial and raises TrialPruned as soon as the pruner rejects the trial, so
hopeless configurations stop early. One pair exists per training family:
backbone, profile autoencoder, image autoencoder and JEPA.
"""

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
    """Backbone trainer that reports to an Optuna trial and honours pruning."""

    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        """Stores the trial and builds the backbone trainer.

        Args:
            *args     : Positional arguments of the backbone trainer.
            trial     : Optuna trial this training belongs to.
            emit_docs : Whether the trainer writes its documentation artifacts.
            **kwargs  : Keyword arguments of the backbone trainer.
        """
        self._trial = trial
        super().__init__(*args, emit_docs=emit_docs, **kwargs)

    def _trial_callback(self, val_loss: float, epoch: int) -> None:
        """Reports the epoch's validation loss and prunes when Optuna says so.

        Args:
            val_loss : Validation loss of the epoch under the run's objective.
            epoch    : Zero-based epoch index used as the pruning step.

        Raises:
            optuna.exceptions.TrialPruned : If the pruner rejects the trial.
        """
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialPipeline(TrainingPipeline):
    """Backbone training pipeline that builds a trial-reporting trainer."""

    def __init__(self, *args, trial: optuna.Trial, emit_docs: bool = False, **kwargs) -> None:
        """Stores the trial and doc-emission flag, then builds the pipeline.

        Args:
            *args     : Positional arguments of the backbone training pipeline.
            trial     : Optuna trial this run belongs to.
            emit_docs : Whether trial runs write documentation artifacts.
            **kwargs  : Keyword arguments of the backbone training pipeline.
        """
        self._trial     = trial
        self._emit_docs = emit_docs
        super().__init__(*args, **kwargs)

    def _make_trainer(self, model, model_cfg, z_axis, norm_stats, run_dir, logger, emit_docs=True, writer=None):
        """Builds the trial-reporting backbone trainer for this run.

        Args:
            model      : Backbone model to train.
            model_cfg  : Model configuration of the trial.
            z_axis     : Z axis samples the profiles are defined on, in metres.
            norm_stats : Normalization statistics of the dataset.
            run_dir    : Run directory of the trial.
            logger     : Logger of the trial run.
            emit_docs  : Pipeline-side doc flag, ANDed with the tuner's own flag.
            writer     : TensorBoard writer, or None.

        Returns:
            The configured TrialTrainer.
        """
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


class TrialProfileAeTrainer(AeTrainer):
    """Profile-autoencoder trainer reporting to an Optuna trial."""

    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        """Stores the trial and builds the profile-autoencoder trainer."""
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        """Reports the epoch's validation loss and prunes when Optuna says so.

        Args:
            val_loss : Reconstruction validation loss of the epoch.
            epoch    : Zero-based epoch index used as the pruning step.

        Raises:
            optuna.exceptions.TrialPruned : If the pruner rejects the trial.
        """
        super()._after_eval(val_loss, epoch)
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialProfileAePipeline(AeTrainingPipeline):
    """Profile-autoencoder pipeline that builds a trial-reporting trainer."""

    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        """Stores the trial and builds the profile-autoencoder pipeline."""
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, run_directory, logger, model, z_axis, writer=None):
        """Builds the trial-reporting profile-autoencoder trainer.

        Args:
            run_directory : Run directory of the trial.
            logger        : Logger of the trial run.
            model         : Profile autoencoder to train.
            z_axis        : Z axis samples the profiles are defined on, in metres.
            writer        : TensorBoard writer, or None.

        Returns:
            The configured TrialProfileAeTrainer.
        """
        return TrialProfileAeTrainer(model, self.autoencoder_cfg, z_axis, self.trainer_config, run_directory, logger, writer, trial=self._trial)


class TrialImageAeTrainer(ImageAeTrainer):
    """Image-autoencoder trainer reporting to an Optuna trial."""

    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        """Stores the trial and builds the image-autoencoder trainer."""
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        """Reports the epoch's validation loss and prunes when Optuna says so.

        Args:
            val_loss : Reconstruction validation loss of the epoch.
            epoch    : Zero-based epoch index used as the pruning step.

        Raises:
            optuna.exceptions.TrialPruned : If the pruner rejects the trial.
        """
        super()._after_eval(val_loss, epoch)
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialImageAePipeline(ImageAeTrainingPipeline):
    """Image-autoencoder pipeline that builds a trial-reporting trainer."""

    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        """Stores the trial and builds the image-autoencoder pipeline."""
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, run_directory, logger, model, z_axis, writer=None):
        """Builds the trial-reporting image-autoencoder trainer.

        Args:
            run_directory : Run directory of the trial.
            logger        : Logger of the trial run.
            model         : Image autoencoder to train.
            z_axis        : Z axis samples the profiles are defined on, in metres.
            writer        : TensorBoard writer, or None.

        Returns:
            The configured TrialImageAeTrainer.
        """
        return TrialImageAeTrainer(model, self.autoencoder_cfg, z_axis, self.trainer_config, run_directory, logger, writer, trial=self._trial)


class TrialJepaTrainer(JepaTrainer):
    """JEPA trainer reporting to an Optuna trial."""

    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        """Stores the trial and builds the JEPA trainer."""
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _after_eval(self, val_loss: float, epoch: int) -> None:
        """Reports the epoch's validation loss and prunes when Optuna says so.

        Args:
            val_loss : Validation loss of the epoch under the JEPA objective.
            epoch    : Zero-based epoch index used as the pruning step.

        Raises:
            optuna.exceptions.TrialPruned : If the pruner rejects the trial.
        """
        super()._after_eval(val_loss, epoch)
        self._trial.report(val_loss, epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class TrialJepaPipeline(JepaTrainingPipeline):
    """JEPA training pipeline that builds a trial-reporting trainer."""

    def __init__(self, *args, trial: optuna.Trial, **kwargs) -> None:
        """Stores the trial and builds the JEPA training pipeline."""
        self._trial = trial
        super().__init__(*args, **kwargs)

    def _make_trainer(self, model, backbone_cfg, z_axis, run_dir, logger, norm_stats, profile_normalizer, writer=None):
        """Builds the trial-reporting JEPA trainer.

        Args:
            model              : JEPA model to train.
            backbone_cfg       : Backbone configuration of the trial.
            z_axis             : Z axis samples the profiles are defined on, in metres.
            run_dir            : Run directory of the trial.
            logger             : Logger of the trial run.
            norm_stats         : Normalization statistics of the dataset.
            profile_normalizer : Normalizer applied to the target profiles.
            writer             : TensorBoard writer, or None.

        Returns:
            The configured TrialJepaTrainer.
        """
        return TrialJepaTrainer(model, backbone_cfg, z_axis, self.trainer_config, run_dir, logger, norm_stats, profile_normalizer, writer, trial=self._trial)
