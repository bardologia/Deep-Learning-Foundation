
from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib  import Path

import optuna
from optuna.trial import TrialState

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

            else:
                idx           = trial.suggest_categorical(name + "__idx", list(range(len(spec["choices"]))))
                sampled[name] = spec["choices"][idx]

        return sampled

    def decode(self, params: dict, space: dict) -> dict:
        decoded = {}

        for name, value in params.items():
            if name.endswith("__idx"):
                param_name = name[:-5]

                decoded[param_name] = space[param_name]["choices"][value]
            else:
                decoded[name] = value

        return decoded


class SearchSpace:

    BATCH  = "batch_size"
    GROUPS = (("search_lr", "tunable_lr_params", "learning-rate"), ("search_arch", "tunable_arch_params", "architecture"))

    @classmethod
    def _group(cls, config_cls, method: str, label: str) -> dict:
        params = getattr(config_cls, method)()

        return params

    @classmethod
    def of(cls, config_cls, tune_cfg) -> dict:
        space = {}

        for switch, method, label in cls.GROUPS:
            if getattr(tune_cfg, switch):
                space.update(cls._group(config_cls, method, label))

        if tune_cfg.search_batch:
            space[cls.BATCH] = {"type": "categorical", "choices": [int(size) for size in tune_cfg.batch_sizes]}

        return space

    @classmethod
    def describe(cls, tune_cfg) -> str:
        names = [switch.removeprefix("search_") for switch, _, _ in cls.GROUPS if getattr(tune_cfg, switch)]
        if tune_cfg.search_batch:
            names.append("batch")

        return "+".join(names)

    @classmethod
    def model_params(cls, params: dict) -> dict:
        return {name: value for name, value in params.items() if name != cls.BATCH}


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


class StudyTargetStop:

    COUNTED = (TrialState.COMPLETE, TrialState.PRUNED)

    def __init__(self, target: int, logger) -> None:
        self.target = target
        self.logger = logger
        self.fired  = False

    @classmethod
    def counted(cls, study: optuna.Study) -> int:
        return len(study.get_trials(deepcopy=False, states=cls.COUNTED))

    @classmethod
    def describe(cls, study: optuna.Study, target: int) -> str:
        complete = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        pruned   = len(study.get_trials(deepcopy=False, states=(TrialState.PRUNED,)))

        return f"{complete} complete + {pruned} pruned = {complete + pruned} of target {target}"

    def reached(self, study: optuna.Study) -> bool:
        return bool(self.target) and self.counted(study) >= self.target

    def __call__(self, study: optuna.Study, trial) -> None:
        if not self.reached(study):
            return

        self.fired = True
        self.logger.info(f"Study stop: target reached, {self.describe(study, self.target)}")
        study.stop()


class StudyPlateauStop:

    FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)

    def __init__(self, patience: int, min_delta: float, logger) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.logger    = logger
        self.fired     = False

    def walk(self, study: optuna.Study) -> tuple[int, int, float | None]:
        finished  = sorted(study.get_trials(deepcopy=False, states=self.FINISHED), key=lambda trial: trial.number)
        reference = None
        anchor    = -1

        for trial in finished:
            if trial.state != TrialState.COMPLETE:
                continue
            if reference is None or trial.value < reference - abs(reference) * self.min_delta:
                reference = trial.value
                anchor    = trial.number

        return sum(1 for trial in finished if trial.number > anchor), anchor, reference

    def stalled(self, study: optuna.Study) -> int:
        return self.walk(study)[0]

    def __call__(self, study: optuna.Study, trial) -> None:
        stalled, anchor, reference = self.walk(study)
        if not self.patience or stalled < self.patience:
            return

        since = f"since trial {anchor} without a {self.min_delta:.1%} improvement on {reference:.5g}" if reference is not None else "with no completed trial yet"

        self.fired = True
        self.logger.info(f"Study stop: plateau, {stalled} finished trials {since}")
        study.stop()


class StudyStopRules:

    TRIAL_FAILURES = (RuntimeError, FloatingPointError)

    TARGET       = "target"
    PLATEAU      = "plateau"
    TIME_BUDGET  = "time_budget"
    WORKER_SHARE = "worker_share"
    STUDY_ENDS   = (TARGET, PLATEAU, TIME_BUDGET)

    def __init__(self, tune_cfg, logger) -> None:
        self.tune_cfg = tune_cfg
        self.logger   = logger
        self.target   = StudyTargetStop(tune_cfg.n_trials, logger)
        self.plateau  = StudyPlateauStop(tune_cfg.plateau_patience, tune_cfg.plateau_min_delta, logger) if tune_cfg.plateau_patience else None

    @staticmethod
    def fail(trial: optuna.Trial, error: BaseException, logger) -> None:
        reason = f"{type(error).__name__}: {error}"

        trial.set_user_attr(StudyProgress.FAIL_REASON, reason)
        logger.error(f"Trial {trial.number} failed and is recorded as such: {reason}")

    def callbacks(self) -> list:
        return [self.target] if self.plateau is None else [self.target, self.plateau]

    def timeout_s(self) -> float | None:
        return self.tune_cfg.time_budget_h * 3600.0 if self.tune_cfg.time_budget_h else None

    def describe(self) -> dict:
        return {
            "Trial target"      : self.tune_cfg.n_trials,
            "Plateau patience"  : self.tune_cfg.plateau_patience or "off",
            "Plateau min delta" : self.tune_cfg.plateau_min_delta,
            "Time budget"       : f"{self.tune_cfg.time_budget_h} h" if self.tune_cfg.time_budget_h else "none",
        }

    def already_met(self, study: optuna.Study) -> str | None:
        if self.target.reached(study):
            return self.TARGET
        if self.plateau is not None and self.plateau.stalled(study) >= self.plateau.patience:
            return self.PLATEAU

        return None

    def reason(self, n_trials: int, trials_run: int, elapsed_s: float) -> str:
        if self.target.fired:
            return self.TARGET
        if self.plateau is not None and self.plateau.fired:
            return self.PLATEAU

        if trials_run >= n_trials:
            self.logger.info(f"Study stop: this worker's share of {n_trials} trial(s) is used up")
            return self.WORKER_SHARE

        timeout = self.timeout_s()
        if timeout is not None and elapsed_s >= timeout:
            self.logger.info(f"Study stop: time budget of {self.tune_cfg.time_budget_h} h used up after {trials_run} trial(s); no further trial started")
            return self.TIME_BUDGET

    def optimize(self, study: optuna.Study, objective, n_trials: int, progress: "StudyProgress", callbacks: list) -> str:
        met = self.already_met(study)
        if met is not None:
            self.logger.warning(f"Study already met its {met} rule before this worker ran a trial ({StudyProgress.phrase(StudyProgress.counts(study), self.tune_cfg.n_trials)}); this worker runs none")
            progress.finish(study, met)
            return met

        started = time.monotonic()
        study.optimize(objective, n_trials=n_trials, timeout=self.timeout_s(), gc_after_trial=True, catch=self.TRIAL_FAILURES, callbacks=[*callbacks, progress, *self.callbacks()])

        reason = self.reason(n_trials, progress.trials_run, time.monotonic() - started)
        progress.finish(study, reason)

        return reason


class StudyProgress:

    FILENAME    = "study_progress.json"
    FINISHED    = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    FAIL_REASON = "fail_reason"

    def __init__(self, log_dir, space: dict, rules: StudyStopRules, logger) -> None:
        self.path    = Path(log_dir) / self.FILENAME
        self.space   = space
        self.rules   = rules
        self.logger  = logger
        self.sampler = ParamSampler()

        self.started     = datetime.now()
        self.trials_run  = 0
        self.worker      = {"complete": 0, "pruned": 0, "fail": 0}
        self.worker_best = None
        self.stop_reason = None

    @staticmethod
    def counts(study: optuna.Study) -> dict:
        trials = study.get_trials(deepcopy=False)
        return {state.name.lower(): sum(1 for trial in trials if trial.state == state) for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING)}

    @staticmethod
    def phrase(counts: dict, target: int) -> str:
        return f"{counts['complete']} complete, {counts['pruned']} pruned, {counts['fail']} failed, {counts['running']} running of target {target}"

    def snapshot(self, study: optuna.Study) -> dict:
        trials   = study.get_trials(deepcopy=False)
        counts   = self.counts(study)
        finished = sum(1 for trial in trials if trial.state in self.FINISHED)
        recent   = sum(1 for trial in trials if trial.state in self.FINISHED and trial.datetime_complete is not None and trial.datetime_complete >= self.started)
        first    = min((trial.datetime_start for trial in trials if trial.datetime_start is not None), default=None)
        now      = datetime.now()
        elapsed  = (now - self.started).total_seconds() / 3600.0
        age      = (now - first).total_seconds() / 3600.0 if first is not None else 0.0
        rate     = recent / elapsed if elapsed > 0 else 0.0
        target   = self.rules.tune_cfg.n_trials
        left     = max(0, target - counts["complete"] - counts["pruned"])
        plateau  = self.rules.plateau

        best = None
        for trial in trials:
            if trial.state == TrialState.COMPLETE and (best is None or trial.value < best.value):
                best = trial

        return {
            "updated_at"       : now.isoformat(timespec="seconds"),
            "trials"           : counts,
            "finished"         : finished,
            "target"           : target,
            "best"             : None if best is None else {"trial": best.number, "value": best.value, "params": self.sampler.decode(dict(best.params), self.space)},
            "stalled"          : None if plateau is None else plateau.stalled(study),
            "plateau_patience" : None if plateau is None else plateau.patience,
            "elapsed_h"        : round(elapsed, 3),
            "study_age_h"      : round(age, 3),
            "trials_per_hour"  : round(rate, 2),
            "eta_target_h"     : None if rate <= 0 else round(left / rate, 2),
            "time_budget_h"    : self.rules.tune_cfg.time_budget_h or None,
            "stop_reason"      : self.stop_reason,
        }

    def _end_note(self, trial) -> str:
        if trial.state == TrialState.PRUNED and trial.last_step is not None:
            return f" at epoch {trial.last_step + 1}"

        if trial.state != TrialState.FAIL:
            return ""

        if self.FAIL_REASON in trial.user_attrs:
            return f" because {trial.user_attrs[self.FAIL_REASON]}"

        return " because Optuna rejected the objective value"

    def _describe(self, trial, snapshot: dict) -> str:
        params = self.sampler.decode(dict(trial.params), self.space)
        shown  = ", ".join(f"{name}={value:.4g}" if isinstance(value, float) else f"{name}={value}" for name, value in params.items())
        value  = f" value={trial.value:.5g}" if trial.value is not None else ""
        ended  = self._end_note(trial)
        counts = snapshot["trials"]
        best   = snapshot["best"]
        stall  = "off" if snapshot["stalled"] is None else f"{snapshot['stalled']}/{snapshot['plateau_patience']}"
        eta    = "n/a" if snapshot["eta_target_h"] is None else f"{snapshot['eta_target_h']:.1f} h"
        lead   = "none" if best is None else f"{best['value']:.5g} @ trial {best['trial']}"

        return f"Trial {trial.number} {trial.state.name}{value}{ended} ({shown}) | study: {counts['complete']} complete, {counts['pruned']} pruned, {counts['fail']} failed, {counts['running']} running of {snapshot['target']} | best {lead} | stalled {stall} | {snapshot['trials_per_hour']:.1f} trials/h over the {snapshot['elapsed_h']:.1f} h of this launch, study age {snapshot['study_age_h']:.1f} h, ETA to target {eta}"

    def _tally(self, trial) -> None:
        self.trials_run += 1
        self.worker[trial.state.name.lower()] += 1

        if trial.state == TrialState.COMPLETE and (self.worker_best is None or trial.value < self.worker_best[1]):
            self.worker_best = (trial.number, trial.value)

    def __call__(self, study: optuna.Study, trial) -> None:
        self._tally(trial)

        snapshot = self.snapshot(study)
        FileIO.save_json(snapshot, self.path, indent=2, atomic=True)
        self.logger.info(self._describe(trial, snapshot))

    def finish(self, study: optuna.Study, reason: str) -> None:
        self.stop_reason = reason

        snapshot = self.snapshot(study)
        FileIO.save_json(snapshot, self.path, indent=2, atomic=True)

        counts = snapshot["trials"]
        self.logger.info(f"Worker finished under the {reason} stop after {self.trials_run} trial(s): {self.worker['complete']} complete, {self.worker['pruned']} pruned, {self.worker['fail']} failed | study: {counts['complete']} complete, {counts['pruned']} pruned, {counts['fail']} failed, {counts['running']} running of {snapshot['target']}")

    def summary(self) -> dict:
        return {
            "trials_run"  : self.trials_run,
            "complete"    : self.worker["complete"],
            "pruned"      : self.worker["pruned"],
            "failed"      : self.worker["fail"],
            "best_trial"  : None if self.worker_best is None else self.worker_best[0],
            "best_value"  : None if self.worker_best is None else self.worker_best[1],
            "stop_reason" : self.stop_reason,
        }


class TrialCheckpoints:

    PATTERN = "**/*.pt"

    def __init__(self, keep: bool, logger) -> None:
        self.keep   = keep
        self.logger = logger

    @staticmethod
    def trial_dir(log_dir, trial_number: int) -> Path:
        return Path(log_dir) / "trials" / f"trial_{trial_number:04d}"

    def discard(self, trial_dir, trial_number: int) -> int:
        if self.keep:
            return 0

        freed = 0
        for path in sorted(Path(trial_dir).glob(self.PATTERN)):
            freed += path.stat().st_size
            path.unlink()

        self.logger.info(f"Trial {trial_number}: discarded {freed / 1e6:.0f} MB of checkpoints under {trial_dir}")

        return freed


class Tuner:

    def __init__(
        self,
        model_name          : str,
        model_config_cls,
        base_trainer_config,
        base_dataset_config,
        training,
        tune_cfg,
        log_dir             : str,
        logger,
        emit_trial_docs     : bool = False,
        worker_index        : int  = 0,
    ) -> None:
        self.model_name          = model_name
        self.model_config_cls    = model_config_cls
        self.base_trainer_config = base_trainer_config
        self.base_dataset_config = base_dataset_config
        self.training            = training
        self.tune_cfg            = tune_cfg
        self.log_dir             = log_dir
        self.logger              = logger
        self.emit_trial_docs     = emit_trial_docs
        self.worker_index        = worker_index

        self.sampler     = ParamSampler()
        self.space       = SearchSpace.of(model_config_cls, tune_cfg)
        self.best_writer = BestConfigWriter(model_name, self.space, Path(log_dir) / "best_config.json")
        self.checkpoints = TrialCheckpoints(tune_cfg.keep_trial_checkpoints, logger)

    def _apply_params(self, trial: optuna.Trial, model_config) -> dict:
        sampled = self.sampler.sample(trial, self.space)
        for name, value in SearchSpace.model_params(sampled).items():
            setattr(model_config, name, value)

        return sampled

    def _apply_batch(self, trainer_cfg, dataset_cfg, batch_size: int) -> None:
        dataset_cfg.batch_size         = batch_size
        trainer_cfg.optimizer.lr_scale = self.training.lr_scale(batch_size)

    def _objective(self, trial: optuna.Trial) -> float:
        from pipelines.tuning.trial import TrialPipeline

        trial.set_user_attr("worker", self.worker_index)

        model_config = self.model_config_cls()
        sampled      = self._apply_params(trial, model_config)

        trainer_cfg = copy.deepcopy(self.base_trainer_config)
        dataset_cfg = copy.deepcopy(self.base_dataset_config)
        trial_dir   = self.checkpoints.trial_dir(self.log_dir, trial.number)

        trainer_cfg.training.epochs         = self.tune_cfg.n_epochs
        trainer_cfg.scheduler.epochs        = self.tune_cfg.n_epochs
        trainer_cfg.early_stopping.patience = self.tune_cfg.early_stop_patience
        trainer_cfg.io.logdir               = str(trial_dir)

        if SearchSpace.BATCH in sampled:
            self._apply_batch(trainer_cfg, dataset_cfg, sampled[SearchSpace.BATCH])

        run_tag = RunNaming.tag(self.model_name, trainer_cfg.loss, dataset_cfg.augmentation)

        pipeline = TrialPipeline(
            trainer_config = trainer_cfg,
            dataset_config = dataset_cfg,
            backbone_name  = self.model_name,
            model_config   = model_config,
            seed           = self.tune_cfg.base_seed + trial.number,
            run_name       = RunNaming.compose(run_tag, f"trial_{trial.number:04d}"),
            trial          = trial,
            emit_docs      = self.emit_trial_docs,
        )

        try:
            _, _, best_val_loss = pipeline.run()
        except StudyStopRules.TRIAL_FAILURES as error:
            StudyStopRules.fail(trial, error, self.logger)
            raise
        finally:
            self.checkpoints.discard(trial_dir, trial.number)

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> dict:
        self.logger.section(f"[Tuner — {self.model_name}]")

        rules = StudyStopRules(self.tune_cfg, self.logger)
        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search groups"        : SearchSpace.describe(self.tune_cfg),
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
            **rules.describe(),
        }, title="Tuner config")

        progress = StudyProgress(self.log_dir, self.space, rules, self.logger)
        rules.optimize(study, self._objective, n_trials, progress, [self.best_writer])

        return progress.summary()
