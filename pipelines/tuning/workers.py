
from __future__ import annotations

import sys
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
from pipelines.tuning.tuners                import AeTuner, JepaTuner, StudyProgress, Tuner
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
            seed          = self.config.tuning.base_seed,
            paths         = self.config.paths,
            training      = self.config.training,
            normalization = self.config.normalization,
            augmentation  = self.config.augmentation,
            **jepa.entry_kwargs(),
        )

    def _build_base_configs(self):
        factory = ConfigFactory(self.config)

        trainer_config            = factory.training_trainer_config(logdir=Path(self.config.paths.log_base_dir))
        trainer_config.curriculum = self.config.curriculum
        dataset_config            = factory.training_dataset_config()

        return trainer_config, dataset_config

    def _profile_ae_tuner(self, model_name: str, tune_cfg, logger: Logger, worker_index: int):
        return AeTuner(
            model_name         = model_name,
            config_cls         = PROFILE_AE_CONFIG_REGISTRY[model_name],
            entry_template     = self._ae_entry_template(),
            trial_pipeline_cls = TrialProfileAePipeline,
            tune_cfg           = tune_cfg,
            log_dir            = str(self.run_dir / model_name),
            logger             = logger,
            worker_index       = worker_index,
        )

    def _image_ae_tuner(self, model_name: str, tune_cfg, logger: Logger, worker_index: int):
        return AeTuner(
            model_name         = model_name,
            config_cls         = IMAGE_AE_CONFIG_REGISTRY[model_name],
            entry_template     = self._image_ae_entry_template(),
            trial_pipeline_cls = TrialImageAePipeline,
            tune_cfg           = tune_cfg,
            log_dir            = str(self.run_dir / model_name),
            logger             = logger,
            worker_index       = worker_index,
        )

    def _jepa_tuner(self, model_name: str, tune_cfg, logger: Logger, worker_index: int):
        return JepaTuner(
            model_name       = model_name,
            model_config_cls = BACKBONE_CONFIG_REGISTRY[ModelBuilder.split_key(model_name)[0]],
            entry_template   = self._jepa_entry_template(),
            tune_cfg         = tune_cfg,
            log_dir          = str(self.run_dir / model_name),
            logger           = logger,
            worker_index     = worker_index,
        )

    def _backbone_tuner(self, model_name: str, tune_cfg, logger: Logger, worker_index: int):
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
            worker_index        = worker_index,
        )

    def _build_tuner(self, model_name: str, tune_cfg, logger: Logger, worker_index: int):
        dispatch = {
            "backbone"            : self._backbone_tuner,
            "image_autoencoder"   : self._image_ae_tuner,
            "jepa"                : self._jepa_tuner,
            "profile_autoencoder" : self._profile_ae_tuner,
        }

        training_type = self.config.training_type
        if training_type not in dispatch:
            raise ValueError(f"Unknown training_type '{training_type}', expected one of {sorted(dispatch)}")

        return dispatch[training_type](model_name, tune_cfg, logger, worker_index)

    def _report_worker(self, logger: Logger, summary: dict, model_name: str, worker_index: int) -> None:
        best = "none" if summary["best_trial"] is None else f"trial {summary['best_trial']}, value {summary['best_value']:.6f}"

        logger.kv_table({
            "Trials run"         : summary["trials_run"],
            "Complete"           : summary["complete"],
            "Pruned"             : summary["pruned"],
            "Failed"             : summary["failed"],
            "Stop reason"        : summary["stop_reason"],
            "Best (this worker)" : best,
        }, title=f"Worker {worker_index} summary")

        if summary["failed"] > 0:
            logger.warning(f"[Worker {worker_index}] {summary['failed']} of {summary['trials_run']} trial(s) failed; each failure is named on its trial line above")

        if summary["trials_run"] > 0 and summary["failed"] == summary["trials_run"]:
            sys.exit(f"ERROR: worker {worker_index} of {model_name}: every one of its {summary['trials_run']} trial(s) failed, so its share brought nothing to the study")

        logger.info(f"[Worker {worker_index}] — {model_name}  DONE")

    def run_worker(self, model_name: str, gpu_id: int, worker_index: int, n_trials: int, study_name: str, storage_url: str) -> None:
        tune_cfg      = self.config.tuning
        model_log_dir = str(self.run_dir / model_name / f"worker{worker_index:02d}")

        with Logger(log_dir=model_log_dir, name=f"tune_worker{worker_index:02d}_{model_name}") as logger:
            logger.adopt("optuna")
            logger.section(f"[Worker {worker_index}] Tuning Worker — {model_name}")

            tuner = self._build_tuner(model_name, tune_cfg, logger, worker_index)

            pruner = optuna.pruners.MedianPruner(
                n_startup_trials = tune_cfg.pruner_n_startup_trials,
                n_warmup_steps   = tune_cfg.pruner_n_warmup_steps,
            )

            sampler = optuna.samplers.TPESampler(
                n_startup_trials = tune_cfg.pruner_n_startup_trials,
                multivariate     = True,
                constant_liar    = True,
                seed             = tune_cfg.base_seed + worker_index,
            )

            study = optuna.load_study(study_name=study_name, storage=TuningStorage.open(storage_url), sampler=sampler, pruner=pruner)

            logger.kv_table({
                "Worker"          : worker_index,
                "GPU"             : gpu_id,
                "Trials"          : n_trials,
                "Study"           : study_name,
                "Storage"         : storage_url,
                "Study on attach" : StudyProgress.phrase(StudyProgress.counts(study), tune_cfg.n_trials),
            })

            summary = tuner.run(study, n_trials)

            self._report_worker(logger, summary, model_name, worker_index)
