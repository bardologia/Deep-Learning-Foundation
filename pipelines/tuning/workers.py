
from __future__ import annotations

import sys
from pathlib import Path

import optuna

from models                                 import BACKBONE_CONFIG_REGISTRY
from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.model.model_builder   import ModelBuilder
from pipelines.tuning.storage               import TuningStorage
from pipelines.tuning.tuners                import StudyProgress, Tuner
from tools.monitoring.logger                import Logger


class TuningWorker:

    def __init__(self, tag: str, config) -> None:
        self.tag     = tag
        self.config  = config
        self.run_dir = Path(config.paths.log_base_dir) / tag

        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _build_base_configs(self):
        factory = ConfigFactory(self.config)

        trainer_config            = factory.training_trainer_config(logdir=Path(self.config.paths.log_base_dir))
        trainer_config.loss = self.config.loss
        dataset_config            = factory.training_dataset_config()

        return trainer_config, dataset_config

    def _build_tuner(self, model_name: str, tune_cfg, logger: Logger, worker_index: int):
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
