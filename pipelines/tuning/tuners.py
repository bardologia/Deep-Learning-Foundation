from __future__ import annotations

import copy
from pathlib import Path

import optuna

from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.model.model_builder   import ModelBuilder
from pipelines.shared.training.run_naming   import RunNaming
from tools.data.io                          import FileIO


class ParamSampler:
    def sample(self, trial: optuna.Trial, space: dict) -> dict:
        sampled = {}

        for name, spec in space.items():
            kind = spec["type"]

            if kind == "float":
                sampled[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

            elif kind == "categorical":
                sampled[name] = trial.suggest_categorical(name, spec["choices"])

            elif kind == "indexed_categorical":
                idx           = trial.suggest_categorical(name + "__idx", list(range(len(spec["choices"]))))
                sampled[name] = spec["choices"][idx]

        return sampled

    def decode(self, params: dict, space: dict) -> dict:
        decoded = {}

        for name, value in params.items():
            if name.endswith("__idx"):
                param_name = name[:-5]
                spec       = space.get(param_name, {})
                if spec.get("type") == "indexed_categorical":
                    decoded[param_name] = spec["choices"][value]
            else:
                decoded[name] = value

        return decoded


class BestConfigWriter:
    def __init__(self, model_name: str, space: dict, path: Path) -> None:
        self.model_name = model_name
        self.space      = space
        self.path       = Path(path)
        self.sampler    = ParamSampler()

    def write(self, study: optuna.Study) -> dict | None:
        try:
            best = study.best_trial
        except ValueError:
            return None

        payload = {
            "model"    : self.model_name,
            "trial"    : best.number,
            "val_loss" : best.value,
            "params"   : self.sampler.decode(dict(best.params), self.space),
        }

        FileIO.save_json(payload, self.path, indent=2, atomic=True)

        return payload

    def __call__(self, study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
        self.write(study)


class Tuner:
    def __init__(
        self,
        model_name          : str,
        model_config_cls,
        base_trainer_config,
        base_dataset_config,
        tune_cfg,
        log_dir             : str,
        logger,
        emit_trial_docs     : bool = False,
    ) -> None:
        self.model_name          = model_name
        self.model_config_cls    = model_config_cls
        self.base_trainer_config = base_trainer_config
        self.base_dataset_config = base_dataset_config
        self.tune_cfg            = tune_cfg
        self.log_dir             = log_dir
        self.logger              = logger
        self.emit_trial_docs     = emit_trial_docs

        self.backbone_name, self.head = ModelBuilder.split_key(model_name)

        self.sampler     = ParamSampler()
        self.space       = {**model_config_cls.tunable_lr_params(), **model_config_cls.tunable_arch_params()}
        self.best_writer = BestConfigWriter(model_name, self.space, Path(log_dir) / "best_config.json")

    def _apply_params(self, trial: optuna.Trial, model_config) -> None:
        sampled = self.sampler.sample(trial, self.space)
        for k, v in sampled.items():
            setattr(model_config, k, v)

    def _objective(self, trial: optuna.Trial) -> float:
        from pipelines.tuning.trial import TrialPipeline

        model_config      = self.model_config_cls()
        model_config.head = self.head
        self._apply_params(trial, model_config)

        trainer_cfg = copy.deepcopy(self.base_trainer_config)
        dataset_cfg = copy.deepcopy(self.base_dataset_config)

        trainer_cfg.training.epochs         = self.tune_cfg.n_epochs
        trainer_cfg.scheduler.epochs        = self.tune_cfg.n_epochs
        trainer_cfg.early_stopping.patience = self.tune_cfg.early_stop_patience
        trainer_cfg.io.logdir               = str(Path(self.log_dir) / "trials" / f"trial_{trial.number:04d}")

        run_tag = RunNaming.training_tag(self.backbone_name, self.head, trainer_cfg.curriculum, trainer_cfg.gaussian.n_default_gaussians, dataset_cfg.augmentation)

        pipeline = TrialPipeline(
            trainer_config = trainer_cfg,
            dataset_config = dataset_cfg,
            backbone_name  = self.backbone_name,
            model_config   = model_config,
            seed           = self.tune_cfg.base_seed + trial.number,
            run_name       = RunNaming.compose(run_tag, f"trial_{trial.number:04d}"),
            trial          = trial,
            emit_docs      = self.emit_trial_docs,
        )

        _, _, best_val_loss = pipeline.run()

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> None:
        self.logger.section(f"[Tuner — {self.model_name}]")

        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
        }, title="Tuner config")

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True, callbacks=[self.best_writer])


class AeTuner:
    def __init__(
        self,
        model_name        : str,
        config_cls,
        entry_template,
        trial_pipeline_cls,
        tune_cfg,
        log_dir           : str,
        logger,
    ) -> None:
        self.model_name         = model_name
        self.config_cls         = config_cls
        self.entry_template     = entry_template
        self.trial_pipeline_cls = trial_pipeline_cls
        self.tune_cfg           = tune_cfg
        self.log_dir            = log_dir
        self.logger             = logger

        self.sampler     = ParamSampler()
        self.space       = {**config_cls.tunable_lr_params(), **config_cls.tunable_arch_params()}
        self.best_writer = BestConfigWriter(model_name, self.space, Path(log_dir) / "best_config.json")

    def _objective(self, trial: optuna.Trial) -> float:
        sampled = self.sampler.sample(trial, self.space)

        entry                 = copy.deepcopy(self.entry_template)
        entry.ae_model_name   = self.model_name
        entry.model_overrides = sampled
        entry.run_name        = f"trial_{trial.number:04d}"
        entry.seed            = self.tune_cfg.base_seed + trial.number
        entry.logdir          = Path(self.log_dir) / "trials"

        entry.training.epochs              = self.tune_cfg.n_epochs
        entry.training.scheduler_epochs    = self.tune_cfg.n_epochs
        entry.training.early_stop_patience = self.tune_cfg.early_stop_patience

        pipeline                 = self.trial_pipeline_cls(entry, trial=trial)
        (_, _, best_val_loss), _ = pipeline.run()

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> None:
        self.logger.section(f"[AeTuner — {self.model_name}]")

        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
        }, title="AeTuner config")

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True, callbacks=[self.best_writer])


class JepaTuner:
    def __init__(
        self,
        model_name     : str,
        model_config_cls,
        entry_template,
        tune_cfg,
        log_dir        : str,
        logger,
    ) -> None:
        self.model_name       = model_name
        self.model_config_cls = model_config_cls
        self.entry_template   = entry_template
        self.tune_cfg         = tune_cfg
        self.log_dir          = log_dir
        self.logger           = logger

        self.backbone_name, self.head = ModelBuilder.split_key(model_name)

        self.sampler     = ParamSampler()
        self.space       = model_config_cls.tunable_arch_params()
        self.best_writer = BestConfigWriter(model_name, self.space, Path(log_dir) / "best_config.json")

    def _objective(self, trial: optuna.Trial) -> float:
        from pipelines.tuning.trial import TrialJepaPipeline

        sampled     = self.sampler.sample(trial, self.space)
        n_gaussians = ConfigFactory(self.entry_template).gaussian_config().n_default_gaussians
        run_tag     = RunNaming.tag(self.backbone_name, self.head, self.entry_template.param_loss, n_gaussians, self.entry_template.augmentation)

        entry                 = copy.deepcopy(self.entry_template)
        entry.backbone_name   = self.backbone_name
        entry.backbone_head   = self.head
        entry.model_overrides = sampled
        entry.run_name        = RunNaming.compose(run_tag, f"trial_{trial.number:04d}")
        entry.seed            = self.tune_cfg.base_seed + trial.number
        entry.logdir          = Path(self.log_dir) / "trials"

        entry.training.epochs              = self.tune_cfg.n_epochs
        entry.training.scheduler_epochs    = self.tune_cfg.n_epochs
        entry.training.early_stop_patience = self.tune_cfg.early_stop_patience

        pipeline                 = TrialJepaPipeline(entry, trial=trial)
        (_, _, best_val_loss), _ = pipeline.run()

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> None:
        self.logger.section(f"[JepaTuner — {self.model_name}]")

        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
        }, title="JepaTuner config")

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True, callbacks=[self.best_writer])
