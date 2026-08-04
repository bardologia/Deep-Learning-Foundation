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
    def __init__(self, tag: str, config) -> None:
        self.tag     = tag
        self.config  = config
        self.run_dir = Path(config.paths.log_base_dir) / tag

        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _ae_entry_template(self):
        return ProfileAeEntryConfig(
            seed            = self.config.tuning.base_seed,
            pixel_subsample = self.config.pixel_subsample,
            keep_empty_frac = self.config.keep_empty_frac,
            ae_loss         = self.config.ae_loss,
            paths           = self.config.paths,
            training        = self.config.training,
        )

    def _image_ae_entry_template(self):
        return ImageAeEntryConfig(
            seed          = self.config.tuning.base_seed,
            ae_loss       = self.config.image_ae_loss,
            paths         = self.config.paths,
            training      = self.config.training,
            normalization = self.config.normalization,
            augmentation  = self.config.augmentation,
        )

    def _jepa_entry_template(self):
        jepa = self.config.jepa

        return JepaEntryConfig(
            seed                       = self.config.tuning.base_seed,
            profile_autoencoder_logdir = jepa.profile_autoencoder_logdir,
            profile_autoencoder_run    = jepa.profile_autoencoder_run,
            profile_autoencoder_mode   = jepa.profile_autoencoder_mode,
            image_autoencoder_logdir   = jepa.image_autoencoder_logdir,
            image_autoencoder_run      = jepa.image_autoencoder_run,
            image_autoencoder_mode     = jepa.image_autoencoder_mode,
            target_provider            = jepa.target_provider,
            embedding_loss             = jepa.embedding_loss,
            param_loss                 = jepa.param_loss,
            paths                      = self.config.paths,
            training                   = self.config.training,
            normalization              = self.config.normalization,
            augmentation               = self.config.augmentation,
        )

    def _build_base_configs(self):
        factory = ConfigFactory(self.config)

        trainer_config            = factory.training_trainer_config(logdir=Path(self.config.paths.log_base_dir))
        trainer_config.curriculum = self.config.curriculum
        dataset_config            = factory.training_dataset_config()

        return trainer_config, dataset_config

    def _profile_ae_tuner(self, model_name: str, tune_cfg, logger: Logger):
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
        return JepaTuner(
            model_name       = model_name,
            model_config_cls = BACKBONE_CONFIG_REGISTRY[ModelBuilder.split_key(model_name)[0]],
            entry_template   = self._jepa_entry_template(),
            tune_cfg         = tune_cfg,
            log_dir          = str(self.run_dir / model_name),
            logger           = logger,
        )

    def _backbone_tuner(self, model_name: str, tune_cfg, logger: Logger):
        trainer_cfg, dataset_cfg = self._build_base_configs()

        return Tuner(
            model_name          = model_name,
            model_config_cls    = BACKBONE_CONFIG_REGISTRY[ModelBuilder.split_key(model_name)[0]],
            base_trainer_config = trainer_cfg,
            base_dataset_config = dataset_cfg,
            tune_cfg            = tune_cfg,
            log_dir             = str(self.run_dir / model_name),
            logger              = logger,
            emit_trial_docs     = tune_cfg.emit_trial_docs,
        )

    def _build_tuner(self, model_name: str, tune_cfg, logger: Logger):
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
