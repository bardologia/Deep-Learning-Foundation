"""Worker process running a share of one model's Optuna trials on a GPU.

Builds the entry template and tuner for the configured training type, attaches
a median pruner and a TPE sampler seeded per GPU, and optimizes the shared
study opened from the run's SQLite storage.
"""

from __future__ import annotations

from pathlib import Path

import optuna

from configuration.training                 import ImageAeEntryConfig, JepaEntryConfig, ProfileAeEntryConfig
from models                                 import BACKBONE_CONFIG_REGISTRY
from models.image_autoencoder               import IMAGE_AE_CONFIG_REGISTRY
from models.profile_autoencoder             import PROFILE_AE_CONFIG_REGISTRY
from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.model.model_builder   import ModelBuilder
from pipelines.tuning.storage               import TuningStorage
from pipelines.tuning.trial                 import TrialImageAePipeline, TrialProfileAePipeline
from pipelines.tuning.tuners                import AeTuner, JepaTuner, Tuner
from tools.monitoring.logger                import Logger


class TuningWorker:
    """Runs one GPU's share of the trials of a single model's study.

    Attributes:
        tag     : Run tag naming the tuning directory.
        config  : Tuning entry configuration.
        run_dir : Directory holding the per-model worker logs and results.
    """

    def __init__(self, tag: str, config) -> None:
        """Resolves the run directory and quiets Optuna's own logging.

        Args:
            tag    : Run tag naming the tuning directory.
            config : Tuning entry configuration.
        """
        self.tag     = tag
        self.config  = config
        self.run_dir = Path(config.paths.log_base_dir) / tag

        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _ae_entry_template(self):
        """Returns the profile-autoencoder entry config each trial deep-copies."""
        return ProfileAeEntryConfig(
            seed            = self.config.tuning.base_seed,
            pixel_subsample = self.config.pixel_subsample,
            keep_empty_frac = self.config.keep_empty_frac,
            ae_loss         = self.config.ae_loss,
            paths           = self.config.paths,
            training        = self.config.training,
        )

    def _image_ae_entry_template(self):
        """Returns the image-autoencoder entry config each trial deep-copies."""
        return ImageAeEntryConfig(
            seed          = self.config.tuning.base_seed,
            ae_loss       = self.config.image_ae_loss,
            paths         = self.config.paths,
            training      = self.config.training,
            normalization = self.config.normalization,
            augmentation  = self.config.augmentation,
        )

    def _jepa_entry_template(self):
        """Returns the JEPA entry config each trial deep-copies."""
        jepa = self.config.jepa

        return JepaEntryConfig(
            seed          = self.config.tuning.base_seed,
            paths         = self.config.paths,
            training      = self.config.training,
            normalization = self.config.normalization,
            augmentation  = self.config.augmentation,
            **jepa.entry_kwargs(),
        )

    def _build_base_configs(self):
        """Returns the (trainer config, dataset config) pair backbone trials copy."""
        factory = ConfigFactory(self.config)

        trainer_config            = factory.training_trainer_config(logdir=Path(self.config.paths.log_base_dir))
        trainer_config.curriculum = self.config.curriculum
        dataset_config            = factory.training_dataset_config()

        return trainer_config, dataset_config

    def _profile_ae_tuner(self, model_name: str, tune_cfg, logger: Logger):
        """Builds the tuner optimizing a profile autoencoder.

        Args:
            model_name : Profile autoencoder model name.
            tune_cfg   : Tuning settings.
            logger     : Logger of this worker.

        Returns:
            The configured AeTuner.
        """
        return AeTuner(
            model_name         = model_name,
            config_cls         = PROFILE_AE_CONFIG_REGISTRY[model_name],
            entry_template     = self._ae_entry_template(),
            trial_pipeline_cls = TrialProfileAePipeline,
            tune_cfg           = tune_cfg,
            log_dir            = str(self.run_dir / model_name),
            logger             = logger,
        )

    def _image_ae_tuner(self, model_name: str, tune_cfg, logger: Logger):
        """Builds the tuner optimizing an image autoencoder.

        Args:
            model_name : Image autoencoder model name.
            tune_cfg   : Tuning settings.
            logger     : Logger of this worker.

        Returns:
            The configured AeTuner.
        """
        return AeTuner(
            model_name         = model_name,
            config_cls         = IMAGE_AE_CONFIG_REGISTRY[model_name],
            entry_template     = self._image_ae_entry_template(),
            trial_pipeline_cls = TrialImageAePipeline,
            tune_cfg           = tune_cfg,
            log_dir            = str(self.run_dir / model_name),
            logger             = logger,
        )

    def _jepa_tuner(self, model_name: str, tune_cfg, logger: Logger):
        """Builds the tuner optimizing a JEPA backbone.

        Args:
            model_name : Composite ``<backbone>[-<head>]`` key.
            tune_cfg   : Tuning settings.
            logger     : Logger of this worker.

        Returns:
            The configured JepaTuner.
        """
        return JepaTuner(
            model_name       = model_name,
            model_config_cls = BACKBONE_CONFIG_REGISTRY[ModelBuilder.split_key(model_name)[0]],
            entry_template   = self._jepa_entry_template(),
            tune_cfg         = tune_cfg,
            log_dir          = str(self.run_dir / model_name),
            logger           = logger,
        )

    def _backbone_tuner(self, model_name: str, tune_cfg, logger: Logger):
        """Builds the tuner optimizing a supervised backbone.

        Args:
            model_name : Composite ``<backbone>[-<head>]`` key.
            tune_cfg   : Tuning settings.
            logger     : Logger of this worker.

        Returns:
            The configured Tuner.
        """
        trainer_cfg, dataset_cfg = self._build_base_configs()

        return Tuner(
            model_name          = model_name,
            model_config_cls    = BACKBONE_CONFIG_REGISTRY[ModelBuilder.split_key(model_name)[0]],
            base_trainer_config = trainer_cfg,
            base_dataset_config = dataset_cfg,
            training            = self.config.training,
            tune_cfg            = tune_cfg,
            log_dir             = str(self.run_dir / model_name),
            logger              = logger,
            emit_trial_docs     = tune_cfg.emit_trial_docs,
        )

    def _build_tuner(self, model_name: str, tune_cfg, logger: Logger):
        """Selects and builds the tuner matching the configured training type.

        Args:
            model_name : Model key to tune.
            tune_cfg   : Tuning settings.
            logger     : Logger of this worker.

        Returns:
            The tuner for this training type.

        Raises:
            ValueError : If the training type has no tuner.
        """
        dispatch = {
            "backbone"            : self._backbone_tuner,
            "image_autoencoder"   : self._image_ae_tuner,
            "jepa"                : self._jepa_tuner,
            "profile_autoencoder" : self._profile_ae_tuner,
        }

        training_type = self.config.training_type
        if training_type not in dispatch:
            raise ValueError(f"Unknown training_type '{training_type}', expected one of {sorted(dispatch)}")

        return dispatch[training_type](model_name, tune_cfg, logger)

    def run_worker(self, model_name: str, gpu_id: int, n_trials: int, study_name: str, storage_url: str) -> None:
        """Optimizes this worker's share of a model's study on one GPU.

        Args:
            model_name  : Model key to tune.
            gpu_id      : GPU index this worker owns; also offsets the sampler seed.
            n_trials    : Trials this worker contributes.
            study_name  : Name of the shared study to load.
            storage_url : SQLAlchemy URL of the shared Optuna storage.
        """
        tune_cfg      = self.config.tuning
        model_log_dir = str(self.run_dir / model_name / f"worker_gpu{gpu_id}")
        logger        = Logger(log_dir=model_log_dir, name=f"tune_worker_gpu{gpu_id}_{model_name}")

        logger.section(f"[GPU {gpu_id}] Tuning Worker — {model_name}")
        logger.kv_table({
            "GPU"        : gpu_id,
            "Trials"     : n_trials,
            "Study"      : study_name,
            "Storage"    : storage_url,
        })

        tuner = self._build_tuner(model_name, tune_cfg, logger)

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = tune_cfg.pruner_n_startup_trials,
            n_warmup_steps   = tune_cfg.pruner_n_warmup_steps,
        )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials = tune_cfg.pruner_n_startup_trials,
            multivariate     = True,
            constant_liar    = True,
            seed             = tune_cfg.base_seed + gpu_id,
        )

        study = optuna.load_study(study_name=study_name, storage=TuningStorage.open(storage_url), sampler=sampler, pruner=pruner)
        tuner.run(study, n_trials)

        logger.info(f"[GPU {gpu_id}] — {model_name}  DONE")
        logger.close()
