"""Search-space sampling and the per-family Optuna objective functions.

ParamSampler turns a declarative search space into Optuna suggestions and back,
BestConfigWriter persists the best trial after every trial, and one tuner class
per training family builds a shortened training run as the trial objective.
"""

from __future__ import annotations

import copy
from pathlib import Path

import optuna
from optuna.trial import TrialState

from pipelines.shared.config.config_factory import ConfigFactory
from pipelines.shared.model.model_builder   import ModelBuilder
from pipelines.shared.training.run_naming   import JepaNamingSpec, RunNaming
from tools.data.io                          import FileIO


class ParamSampler:
    """Samples a declarative search space and decodes stored trial parameters."""

    def sample(self, trial: optuna.Trial, space: dict) -> dict:
        """Draws one value per search dimension from the trial.

        Float dimensions are suggested directly, categorical ones by value, and
        ``indexed_categorical`` ones through an integer index so unhashable
        choices can be stored.

        Args:
            trial : Optuna trial doing the suggesting.
            space : Parameter name to specification mapping.

        Returns:
            Mapping from parameter name to the sampled value.
        """
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
        """Resolves stored ``__idx`` parameters back to their choice values.

        Args:
            params : Parameters as Optuna stored them for a trial.
            space  : Search space the trial was sampled from.

        Returns:
            Mapping from parameter name to its decoded value.

        Raises:
            KeyError : If an ``__idx`` parameter has no matching indexed_categorical dimension, meaning study and code diverged.
        """
        decoded = {}

        for name, value in params.items():
            if name.endswith("__idx"):
                param_name = name[:-5]

                if param_name not in space or space[param_name]["type"] != "indexed_categorical":
                    raise KeyError(f"Study parameter '{name}' has no indexed_categorical entry named '{param_name}' in the current search space; the study and the code have diverged, so the best configuration cannot be decoded")

                decoded[param_name] = space[param_name]["choices"][value]
            else:
                decoded[name] = value

        return decoded


class BestConfigWriter:
    """Writes the best trial of a study to JSON, also usable as a trial callback.

    Attributes:
        model_name : Model key the study belongs to.
        space      : Search space used to decode the stored parameters.
        path       : Path of the best-configuration JSON file.
        sampler    : Sampler providing the decoding.
    """

    def __init__(self, model_name: str, space: dict, path: Path) -> None:
        """Stores the model name, search space and output path."""
        self.model_name = model_name
        self.space      = space
        self.path       = Path(path)
        self.sampler    = ParamSampler()

    def write(self, study: optuna.Study) -> dict | None:
        """Writes the study's best trial to JSON.

        Args:
            study : Study whose best trial is persisted.

        Returns:
            The payload holding the model, trial number, validation loss and
            decoded parameters, or None when no trial completed yet.
        """
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
        """Rewrites the best configuration after a trial, as an Optuna callback.

        Args:
            study        : Study the finished trial belongs to.
            frozen_trial : The trial that just finished; unused.
        """
        self.write(study)


class StudyTargetStop:
    """Optuna callback stopping a worker once its study has reached the trial target.

    Every worker is handed its own share of the trials, and a second launch
    against the same study would add that share on top of what the first one
    already ran. Stopping on the study's own completed count keeps the target
    authoritative however many workers or launches attach to it.

    Attributes:
        target : Completed trials the study is optimised to; zero disables the stop.
    """

    def __init__(self, target: int) -> None:
        """Binds the callback to the study's trial target."""
        self.target = target

    def __call__(self, study: optuna.Study, trial) -> None:
        """Stops the study once it holds at least ``target`` completed trials."""
        if self.target and len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))) >= self.target:
            study.stop()


class Tuner:
    """Optuna objective running a shortened backbone training per trial.

    Attributes:
        model_name          : Composite ``<backbone>[-<head>]`` key being tuned.
        model_config_cls    : Model configuration class of the backbone.
        base_trainer_config : Trainer configuration each trial copies.
        base_dataset_config : Dataset configuration each trial copies.
        tune_cfg            : Tuning settings (epochs, patience, base seed).
        log_dir             : Directory holding the trial runs and the best configuration.
        logger              : Logger of the tuning worker.
        emit_trial_docs     : Whether trial runs write documentation artifacts.
        backbone_name       : Backbone part of the model key.
        head                : Head part of the model key.
        sampler             : Sampler drawing the trial parameters.
        space               : Search space over learning-rate and architecture parameters.
        best_writer         : Callback persisting the best trial after each trial.
    """

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
        """Builds the search space and the best-config writer for a backbone.

        Args:
            model_name          : Composite ``<backbone>[-<head>]`` key to tune.
            model_config_cls    : Model configuration class of the backbone.
            base_trainer_config : Trainer configuration each trial copies.
            base_dataset_config : Dataset configuration each trial copies.
            tune_cfg            : Tuning settings.
            log_dir             : Directory holding the trial runs.
            logger              : Logger of the tuning worker.
            emit_trial_docs     : Whether trial runs write documentation artifacts.
        """
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
        """Samples the search space and writes the values onto the model config."""
        sampled = self.sampler.sample(trial, self.space)
        for k, v in sampled.items():
            setattr(model_config, k, v)

    def _objective(self, trial: optuna.Trial) -> float:
        """Trains one sampled backbone configuration and returns its best loss.

        Args:
            trial : Optuna trial supplying the sampled parameters.

        Returns:
            The best validation loss reached by the shortened training run.
        """
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
        """Runs the requested number of backbone trials against the study.

        Args:
            study    : Shared Optuna study to optimize.
            n_trials : Trials this worker contributes.
        """
        self.logger.section(f"[Tuner — {self.model_name}]")

        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
        }, title="Tuner config")

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True, callbacks=[self.best_writer, StudyTargetStop(self.tune_cfg.n_trials)])


class AeTuner:
    """Optuna objective running a shortened autoencoder training per trial.

    Serves both the profile and the image autoencoder families, which differ
    only in their config registry, entry template and trial pipeline class.

    Attributes:
        model_name         : Autoencoder model name being tuned.
        config_cls         : Autoencoder configuration class supplying the search space.
        entry_template     : Entry configuration each trial deep-copies.
        trial_pipeline_cls : Trial-aware training pipeline class to run.
        tune_cfg           : Tuning settings (epochs, patience, base seed).
        log_dir            : Directory holding the trial runs and the best configuration.
        logger             : Logger of the tuning worker.
        sampler            : Sampler drawing the trial parameters.
        space              : Search space over learning-rate and architecture parameters.
        best_writer        : Callback persisting the best trial after each trial.
    """

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
        """Builds the search space and the best-config writer for an autoencoder.

        Args:
            model_name         : Autoencoder model name to tune.
            config_cls         : Autoencoder configuration class.
            entry_template     : Entry configuration each trial deep-copies.
            trial_pipeline_cls : Trial-aware training pipeline class.
            tune_cfg           : Tuning settings.
            log_dir            : Directory holding the trial runs.
            logger             : Logger of the tuning worker.
        """
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
        """Trains one sampled autoencoder configuration and returns its best loss.

        Args:
            trial : Optuna trial supplying the sampled parameters.

        Returns:
            The best reconstruction validation loss of the shortened run.
        """
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
        """Runs the requested number of autoencoder trials against the study.

        Args:
            study    : Shared Optuna study to optimize.
            n_trials : Trials this worker contributes.
        """
        self.logger.section(f"[AeTuner — {self.model_name}]")

        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
        }, title="AeTuner config")

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True, callbacks=[self.best_writer, StudyTargetStop(self.tune_cfg.n_trials)])


class JepaTuner:
    """Optuna objective running a shortened JEPA training per trial.

    Only architecture parameters are searched; the run tag is composed once
    from the coupled autoencoders so every trial lands under the same name.

    Attributes:
        model_name       : Composite ``<backbone>[-<head>]`` key being tuned.
        model_config_cls : Backbone configuration class.
        entry_template   : JEPA entry configuration each trial deep-copies.
        tune_cfg         : Tuning settings (epochs, patience, base seed).
        log_dir          : Directory holding the trial runs and the best configuration.
        logger           : Logger of the tuning worker.
        backbone_name    : Backbone part of the model key.
        head             : Head part of the model key.
        sampler          : Sampler drawing the trial parameters.
        space            : Search space over the architecture parameters.
        best_writer      : Callback persisting the best trial after each trial.
        run_tag          : Shared JEPA run tag the trial run names are composed from.
    """

    def __init__(
        self,
        model_name     : str,
        model_config_cls,
        entry_template,
        tune_cfg,
        log_dir        : str,
        logger,
    ) -> None:
        """Builds the search space, best-config writer and shared JEPA run tag.

        Args:
            model_name       : Composite ``<backbone>[-<head>]`` key to tune.
            model_config_cls : Backbone configuration class.
            entry_template   : JEPA entry configuration each trial deep-copies.
            tune_cfg         : Tuning settings.
            log_dir          : Directory holding the trial runs.
            logger           : Logger of the tuning worker.
        """
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

        n_gaussians  = ConfigFactory(entry_template).gaussian_config().n_default_gaussians
        self.run_tag = RunNaming.jepa_tag(self.backbone_name, self.head, JepaNamingSpec.resolve(entry_template), n_gaussians, entry_template.augmentation)

    def _objective(self, trial: optuna.Trial) -> float:
        """Trains one sampled JEPA configuration and returns its best loss.

        Args:
            trial : Optuna trial supplying the sampled parameters.

        Returns:
            The best validation loss of the shortened JEPA run.
        """
        from pipelines.tuning.trial import TrialJepaPipeline

        sampled = self.sampler.sample(trial, self.space)

        entry                 = copy.deepcopy(self.entry_template)
        entry.backbone_name   = self.backbone_name
        entry.backbone_head   = self.head
        entry.model_overrides = sampled
        entry.run_name        = RunNaming.compose(self.run_tag, f"trial_{trial.number:04d}")
        entry.seed            = self.tune_cfg.base_seed + trial.number
        entry.logdir          = Path(self.log_dir) / "trials"

        entry.training.epochs              = self.tune_cfg.n_epochs
        entry.training.scheduler_epochs    = self.tune_cfg.n_epochs
        entry.training.early_stop_patience = self.tune_cfg.early_stop_patience

        pipeline                 = TrialJepaPipeline(entry, trial=trial)
        (_, _, best_val_loss), _ = pipeline.run()

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> None:
        """Runs the requested number of JEPA trials against the study.

        Args:
            study    : Shared Optuna study to optimize.
            n_trials : Trials this worker contributes.
        """
        self.logger.section(f"[JepaTuner — {self.model_name}]")

        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
        }, title="JepaTuner config")

        study.optimize(self._objective, n_trials=n_trials, gc_after_trial=True, callbacks=[self.best_writer, StudyTargetStop(self.tune_cfg.n_trials)])
