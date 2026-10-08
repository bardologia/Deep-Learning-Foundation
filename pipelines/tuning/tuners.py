"""Search-space sampling, study stop rules and the per-family Optuna objective functions.

ParamSampler turns a declarative search space into Optuna suggestions and back,
SearchSpace picks the part of a model's declared space a study samples,
BestConfigWriter persists the best trial after every trial, StudyProgress logs
one line per finished trial, tallies the worker's own trials and keeps a
snapshot of the study beside it, the stop rules end a study at its trial
target, on a plateau of its best value or at a time budget and name the rule
that ended a worker's run, and one tuner class per training family builds a
shortened training run as the trial objective.
"""

from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib  import Path

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

        Raises:
            ValueError : If a dimension declares a type other than float, categorical or indexed_categorical.
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

            else:
                raise ValueError(f"Unknown search space parameter type '{kind}' for parameter '{name}'. Supported: float, categorical, indexed_categorical.")

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


class SearchSpace:
    """Assembles the parameters one study samples from the groups the tuning settings switch on.

    Two groups come from the model configuration class, the learning rates and
    the architecture; the batch size comes from the tuning settings and is kept
    apart under ``BATCH``, since it lands on the loader and the optimizer scale
    rather than on the model.

    Attributes:
        BATCH  : Name of the batch-size parameter in a study.
        GROUPS : Switch of the tuning settings, the config-class method declaring the group and its label, for the two model groups.
    """

    BATCH  = "batch_size"
    GROUPS = (("search_lr", "tunable_lr_params", "learning-rate"), ("search_arch", "tunable_arch_params", "architecture"))

    @classmethod
    def _group(cls, config_cls, method: str, label: str) -> dict:
        """Returns one declared parameter group of a model configuration class, refusing an empty one."""
        params = getattr(config_cls, method)()
        if not params:
            raise ValueError(f"{config_cls.__name__} declares no {label} parameter to search; switch that group off")

        return params

    @classmethod
    def of(cls, config_cls, tune_cfg) -> dict:
        """Returns the search space the tuning settings select for a model configuration class.

        Args:
            config_cls : Model configuration class declaring ``tunable_lr_params`` and ``tunable_arch_params``.
            tune_cfg   : Tuning settings carrying the group switches and the batch sizes.

        Returns:
            Mapping from parameter name to its sampling specification.

        Raises:
            ValueError : If every group is off, a switched-on model group is empty, or the batch group has no sizes to pick from.
        """
        space = {}

        for switch, method, label in cls.GROUPS:
            if getattr(tune_cfg, switch):
                space.update(cls._group(config_cls, method, label))

        if tune_cfg.search_batch:
            if not tune_cfg.batch_sizes:
                raise ValueError("search_batch is on but batch_sizes is empty; list the batch sizes the study may pick")
            space[cls.BATCH] = {"type": "categorical", "choices": [int(size) for size in tune_cfg.batch_sizes]}

        if not space:
            raise ValueError("every search group is off; switch on search_lr, search_arch or search_batch")

        return space

    @classmethod
    def arch_only(cls, config_cls) -> dict:
        """Returns the architecture group alone, which the JEPA studies always search."""
        return cls._group(config_cls, "tunable_arch_params", "architecture")

    @classmethod
    def describe(cls, tune_cfg) -> str:
        """Returns the switched-on groups as a short label for the logs."""
        names = [switch.removeprefix("search_") for switch, _, _ in cls.GROUPS if getattr(tune_cfg, switch)]
        if tune_cfg.search_batch:
            names.append("batch")

        return "+".join(names)

    @classmethod
    def model_params(cls, params: dict) -> dict:
        """Returns the sampled parameters that land on the model configuration, the batch size left out."""
        return {name: value for name, value in params.items() if name != cls.BATCH}


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
    already ran. Stopping on the study's own count keeps the target
    authoritative however many workers or launches attach to it. The count is
    the one the scheduler tops a study up by: completed and pruned trials, the
    evaluations that ran to a verdict, while a failed trial is left out so a
    top-up replaces it.

    Attributes:
        COUNTED : Trial states counted towards the target.
        target  : Completed and pruned trials the study is optimised to; zero disables the stop.
        logger  : Logger of the tuning worker, told when the rule fires.
        fired   : Whether this rule ended the worker's optimisation.
    """

    COUNTED = (TrialState.COMPLETE, TrialState.PRUNED)

    def __init__(self, target: int, logger) -> None:
        """Binds the callback to the study's trial target and the worker's logger."""
        self.target = target
        self.logger = logger
        self.fired  = False

    @classmethod
    def counted(cls, study: optuna.Study) -> int:
        """Returns the number of trials counted towards the target: the completed and the pruned ones."""
        return len(study.get_trials(deepcopy=False, states=cls.COUNTED))

    @classmethod
    def describe(cls, study: optuna.Study, target: int) -> str:
        """Returns the target count spelt out, as 'c complete + p pruned = n of target t'."""
        complete = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        pruned   = len(study.get_trials(deepcopy=False, states=(TrialState.PRUNED,)))

        return f"{complete} complete + {pruned} pruned = {complete + pruned} of target {target}"

    def reached(self, study: optuna.Study) -> bool:
        """Returns whether the study holds at least ``target`` completed and pruned trials; never with a zero target."""
        return bool(self.target) and self.counted(study) >= self.target

    def __call__(self, study: optuna.Study, trial) -> None:
        """Stops the study once it holds at least ``target`` completed and pruned trials, logging the stop."""
        if not self.reached(study):
            return

        self.fired = True
        self.logger.info(f"Study stop: target reached, {self.describe(study, self.target)}")
        study.stop()


class StudyPlateauStop:
    """Optuna callback stopping a study whose best value stopped improving.

    The finished trials are walked in trial order with a running reference: a
    completed trial beating the reference by at least the relative margin moves
    it, and every finished trial that follows the last move, pruned ones
    included, counts towards the patience. The verdict is derived from the
    shared study on every call, so every worker attached to the study reaches
    the same one without state of its own.

    Attributes:
        FINISHED  : Trial states that count as finished evaluations.
        patience  : Finished trials without a counted improvement before the study stops; zero disables the rule.
        min_delta : Smallest improvement, relative to the reference value, that counts.
        logger    : Logger of the tuning worker, told when the rule fires.
        fired     : Whether this rule ended the worker's optimisation.
    """

    FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)

    def __init__(self, patience: int, min_delta: float, logger) -> None:
        """Binds the callback to the patience, the relative margin and the worker's logger."""
        self.patience  = patience
        self.min_delta = min_delta
        self.logger    = logger
        self.fired     = False

    def walk(self, study: optuna.Study) -> tuple[int, int, float | None]:
        """Walks the finished trials in trial order and returns the stall count, the anchor and the reference.

        Args:
            study : Study whose finished trials are walked.

        Returns:
            How many finished trials followed the last counted improvement of
            the best value, the number of the trial that made that improvement
            (-1 before any completed trial) and the best value it set (None
            before any completed trial).
        """
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
        """Returns how many finished trials followed the last counted improvement of the best value."""
        return self.walk(study)[0]

    def __call__(self, study: optuna.Study, trial) -> None:
        """Stops the study once the patience is used up without a counted improvement, logging the stop."""
        stalled, anchor, reference = self.walk(study)
        if not self.patience or stalled < self.patience:
            return

        since = f"since trial {anchor} without a {self.min_delta:.1%} improvement on {reference:.5g}" if reference is not None else "with no completed trial yet"

        self.fired = True
        self.logger.info(f"Study stop: plateau, {stalled} finished trials {since}")
        study.stop()


class StudyStopRules:
    """Derives the stop rules every tuner runs ``study.optimize`` under, and names the rule that ended it.

    The trial target always stops a worker; the plateau rule joins it when the
    tuning settings set a patience, and the time budget becomes the timeout of
    the optimisation, after which a worker starts no further trial. Optuna
    leaves its loop for one of four reasons, checked in this order before
    every trial: a callback asked the study to stop (the trial target or the
    plateau rule), the worker ran its share of trials, or the time budget
    passed. Nothing in Optuna says which one it was, so the rules remember when
    they fire and ``reason`` works the rest out from the trials run and the
    time taken.

    Attributes:
        TRIAL_FAILURES : Exceptions a trial may raise that end the trial as failed instead of ending the worker: a diverging loss, which the trainer aborts with a FloatingPointError, and a configuration too large for the device, which CUDA raises as a RuntimeError.
        TARGET         : Reason code of a run ended by the trial target.
        PLATEAU        : Reason code of a run ended by the plateau rule.
        TIME_BUDGET    : Reason code of a run ended by the time budget.
        WORKER_SHARE   : Reason code of a run that used up the worker's share of trials.
        STUDY_ENDS     : Reason codes that end the study as configured; a used-up share leaves it short of its target.
        tune_cfg       : Tuning settings holding the trial target, the plateau rule and the time budget.
        logger         : Logger of the tuning worker.
        target         : Trial-target stop of this worker.
        plateau        : Plateau stop of this worker, or None when no patience is set.
    """

    TRIAL_FAILURES = (RuntimeError, FloatingPointError)

    TARGET       = "target"
    PLATEAU      = "plateau"
    TIME_BUDGET  = "time_budget"
    WORKER_SHARE = "worker_share"
    STUDY_ENDS   = (TARGET, PLATEAU, TIME_BUDGET)

    def __init__(self, tune_cfg, logger) -> None:
        """Builds the rule callbacks from the tuning settings.

        Args:
            tune_cfg : Tuning settings holding the trial target, the plateau rule and the time budget.
            logger   : Logger of the tuning worker, told when a rule fires.
        """
        self.tune_cfg = tune_cfg
        self.logger   = logger
        self.target   = StudyTargetStop(tune_cfg.n_trials, logger)
        self.plateau  = StudyPlateauStop(tune_cfg.plateau_patience, tune_cfg.plateau_min_delta, logger) if tune_cfg.plateau_patience else None

    @staticmethod
    def fail(trial: optuna.Trial, error: BaseException, logger) -> None:
        """Records why a trial failed: on the worker log, and as its ``fail_reason`` attribute, which the trial's progress line prints.

        Args:
            trial  : Trial whose objective raised one of TRIAL_FAILURES.
            error  : The exception, recorded as its type and message.
            logger : Logger of the tuning worker.
        """
        reason = f"{type(error).__name__}: {error}"

        trial.set_user_attr(StudyProgress.FAIL_REASON, reason)
        logger.error(f"Trial {trial.number} failed and is recorded as such: {reason}")

    def callbacks(self) -> list:
        """Returns the trial-target stop and, when a patience is set, the plateau stop."""
        return [self.target] if self.plateau is None else [self.target, self.plateau]

    def timeout_s(self) -> float | None:
        """Returns the time budget in seconds, or None when no budget is set."""
        return self.tune_cfg.time_budget_h * 3600.0 if self.tune_cfg.time_budget_h else None

    def describe(self) -> dict:
        """Returns the rules as rows of a tuner's configuration table."""
        return {
            "Trial target"      : self.tune_cfg.n_trials,
            "Plateau patience"  : self.tune_cfg.plateau_patience or "off",
            "Plateau min delta" : self.tune_cfg.plateau_min_delta,
            "Time budget"       : f"{self.tune_cfg.time_budget_h} h" if self.tune_cfg.time_budget_h else "none",
        }

    def already_met(self, study: optuna.Study) -> str | None:
        """Returns the rule a study satisfies before this worker runs a trial, TARGET or PLATEAU, else None.

        Optuna runs the stop callbacks only after a trial has finished, so a
        worker attaching to a study that already reached its target or plateau
        would train one more full trial before noticing.

        Args:
            study : Shared study as it stands on attach.
        """
        if self.target.reached(study):
            return self.TARGET
        if self.plateau is not None and self.plateau.stalled(study) >= self.plateau.patience:
            return self.PLATEAU

        return None

    def reason(self, n_trials: int, trials_run: int, elapsed_s: float) -> str:
        """Returns which rule ended ``study.optimize``, logging the two Optuna ends silently.

        Args:
            n_trials   : Trials this worker was handed.
            trials_run : Trials this worker ran, whatever their state.
            elapsed_s  : Seconds the optimisation took.

        Returns:
            One of TARGET, PLATEAU, WORKER_SHARE or TIME_BUDGET, in the order Optuna checks them.

        Raises:
            RuntimeError : If optimize returned although no rule fired, the share is not used up and the budget has not passed.
        """
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

        raise RuntimeError(f"study.optimize returned after {trials_run} of {n_trials} trial(s) and {elapsed_s / 3600.0:.2f} h although no stop rule fired, the share is not used up and no time budget passed")

    def optimize(self, study: optuna.Study, objective, n_trials: int, progress: "StudyProgress", callbacks: list) -> str:
        """Runs the study under the rules, records the reason the run ended and returns it.

        Args:
            study     : Shared study to optimize.
            objective : Trial objective of the tuner.
            n_trials  : Trials this worker contributes.
            progress  : Progress callback of this worker, which tallies its trials and keeps the study snapshot.
            callbacks : Further Optuna callbacks run before the progress and stop rules, such as the best-config writer.

        Returns:
            The reason code the run ended with; a study already at its target or plateau on attach ends with that reason and no trial run.
        """
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
    """Optuna callback logging one line per finished trial and writing a snapshot of the study.

    The line names the trial's state, value and parameters and the study's
    counts, best value, stall count against the plateau patience and finishing
    rate, so a worker log reads as a progress record of the whole study. The
    snapshot, ``study_progress.json`` beside the best configuration, holds the
    same numbers for the console or a shell to read while the study runs.
    Every worker attached to the study writes both from the shared study, so
    the snapshot reads the same whichever worker wrote it last.

    Attributes:
        FILENAME    : Name of the snapshot file inside the model's tuning directory.
        FINISHED    : Trial states counted as finished evaluations.
        FAIL_REASON : User attribute naming the error a failed trial raised.
        path        : Snapshot path.
        space       : Search space the stored parameters are decoded against.
        rules       : Stop rules, whose trial target and plateau patience the snapshot reports against.
        logger      : Logger of the tuning worker.
        sampler     : Sampler providing the decoding of stored parameters.
        started     : Moment this worker's launch began, from which the finishing rate and the time to the target are measured.
        trials_run  : Trials this worker ran, whatever their state.
        worker      : This worker's own trial counts keyed complete, pruned and fail.
        worker_best : (trial number, value) of this worker's best completed trial, or None.
        stop_reason : Reason code the worker's run ended with, None while it runs.
    """

    FILENAME    = "study_progress.json"
    FINISHED    = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    FAIL_REASON = "fail_reason"

    def __init__(self, log_dir, space: dict, rules: StudyStopRules, logger) -> None:
        """Stores the snapshot path, the search space, the stop rules and the logger, and starts this worker's tally."""
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
        """Returns the study's trial counts keyed complete, pruned, fail and running."""
        trials = study.get_trials(deepcopy=False)
        return {state.name.lower(): sum(1 for trial in trials if trial.state == state) for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING)}

    @staticmethod
    def phrase(counts: dict, target: int) -> str:
        """Returns the counts as one phrase of the form 'c complete, p pruned, f failed, r running of target n'."""
        return f"{counts['complete']} complete, {counts['pruned']} pruned, {counts['fail']} failed, {counts['running']} running of target {target}"

    def snapshot(self, study: optuna.Study) -> dict:
        """Returns the study's progress: state counts, best trial, stall count, rate and time to the target.

        The rate counts the study's trials, every worker's, that finished
        since this worker's launch began, over the time since then, so the
        idle gap before a resume does not dilute it; the study's age since its
        first trial is reported apart.

        Args:
            study : Shared study the numbers are read from.
        """
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
        """Returns why a failed trial ended, else an empty string.

        A failed trial without a recorded reason raised nothing: Optuna
        rejected the value its objective returned, such as a NaN.
        """
        if trial.state != TrialState.FAIL:
            return ""

        if self.FAIL_REASON in trial.user_attrs:
            return f" because {trial.user_attrs[self.FAIL_REASON]}"

        return " because Optuna rejected the objective value"

    def _describe(self, trial, snapshot: dict) -> str:
        """Returns the one-line progress record of a finished trial."""
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
        """Counts a trial this worker just ran under its state and keeps the worker's best completed value."""
        self.trials_run += 1
        self.worker[trial.state.name.lower()] += 1

        if trial.state == TrialState.COMPLETE and (self.worker_best is None or trial.value < self.worker_best[1]):
            self.worker_best = (trial.number, trial.value)

    def __call__(self, study: optuna.Study, trial) -> None:
        """Tallies the finished trial, logs its record and rewrites the snapshot, as an Optuna callback."""
        self._tally(trial)

        snapshot = self.snapshot(study)
        FileIO.save_json(snapshot, self.path, indent=2, atomic=True)
        self.logger.info(self._describe(trial, snapshot))

    def finish(self, study: optuna.Study, reason: str) -> None:
        """Records the reason the worker's run ended in the snapshot and logs the worker's tally against the study.

        Args:
            study  : Shared study the final snapshot is read from.
            reason : Reason code from the stop rules.
        """
        self.stop_reason = reason

        snapshot = self.snapshot(study)
        FileIO.save_json(snapshot, self.path, indent=2, atomic=True)

        counts = snapshot["trials"]
        self.logger.info(f"Worker finished under the {reason} stop after {self.trials_run} trial(s): {self.worker['complete']} complete, {self.worker['pruned']} pruned, {self.worker['fail']} failed | study: {counts['complete']} complete, {counts['pruned']} pruned, {counts['fail']} failed, {counts['running']} running of {snapshot['target']}")

    def summary(self) -> dict:
        """Returns this worker's tally: trials run by state, its best trial and the reason its run ended."""
        return {
            "trials_run"  : self.trials_run,
            "complete"    : self.worker["complete"],
            "pruned"      : self.worker["pruned"],
            "failed"      : self.worker["fail"],
            "best_trial"  : None if self.worker_best is None else self.worker_best[0],
            "best_value"  : None if self.worker_best is None else self.worker_best[1],
            "stop_reason" : self.stop_reason,
        }


class Tuner:
    """Optuna objective running a shortened backbone training per trial.

    Attributes:
        model_name          : Composite ``<backbone>[-<head>]`` key being tuned.
        model_config_cls    : Model configuration class of the backbone.
        base_trainer_config : Trainer configuration each trial copies.
        base_dataset_config : Dataset configuration each trial copies.
        training            : Training schedule the trials share, supplying the learning-rate scaling rule a sampled batch size follows.
        tune_cfg            : Tuning settings (epochs, patience, base seed, search groups).
        log_dir             : Directory holding the trial runs and the best configuration.
        logger              : Logger of the tuning worker.
        emit_trial_docs     : Whether trial runs write documentation artifacts.
        backbone_name       : Backbone part of the model key.
        head                : Head part of the model key.
        sampler             : Sampler drawing the trial parameters.
        space               : Search space over the switched-on groups of the backbone's tunable parameters and the batch size.
        best_writer         : Callback persisting the best trial after each trial.
    """

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
    ) -> None:
        """Builds the search space and the best-config writer for a backbone.

        Args:
            model_name          : Composite ``<backbone>[-<head>]`` key to tune.
            model_config_cls    : Model configuration class of the backbone.
            base_trainer_config : Trainer configuration each trial copies.
            base_dataset_config : Dataset configuration each trial copies.
            training            : Training schedule the trials share.
            tune_cfg            : Tuning settings.
            log_dir             : Directory holding the trial runs.
            logger              : Logger of the tuning worker.
            emit_trial_docs     : Whether trial runs write documentation artifacts.
        """
        self.model_name          = model_name
        self.model_config_cls    = model_config_cls
        self.base_trainer_config = base_trainer_config
        self.base_dataset_config = base_dataset_config
        self.training            = training
        self.tune_cfg            = tune_cfg
        self.log_dir             = log_dir
        self.logger              = logger
        self.emit_trial_docs     = emit_trial_docs

        self.backbone_name, self.head = ModelBuilder.split_key(model_name)

        self.sampler     = ParamSampler()
        self.space       = SearchSpace.of(model_config_cls, tune_cfg)
        self.best_writer = BestConfigWriter(model_name, self.space, Path(log_dir) / "best_config.json")

    def _apply_params(self, trial: optuna.Trial, model_config) -> dict:
        """Samples the search space, writes the model values onto the model config and returns everything sampled."""
        sampled = self.sampler.sample(trial, self.space)
        for name, value in SearchSpace.model_params(sampled).items():
            setattr(model_config, name, value)

        return sampled

    def _apply_batch(self, trainer_cfg, dataset_cfg, batch_size: int) -> None:
        """Puts a sampled batch size on the loader and rescales the optimizer as the training entries do.

        Args:
            trainer_cfg : Trial's trainer configuration, whose optimizer scale follows the batch.
            dataset_cfg : Trial's dataset configuration, whose loader takes the batch.
            batch_size  : Sampled batch size.
        """
        dataset_cfg.batch_size         = batch_size
        trainer_cfg.optimizer.lr_scale = self.training.lr_scale(batch_size)

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
        sampled           = self._apply_params(trial, model_config)

        trainer_cfg = copy.deepcopy(self.base_trainer_config)
        dataset_cfg = copy.deepcopy(self.base_dataset_config)

        trainer_cfg.training.epochs         = self.tune_cfg.n_epochs
        trainer_cfg.scheduler.epochs        = self.tune_cfg.n_epochs
        trainer_cfg.early_stopping.patience = self.tune_cfg.early_stop_patience
        trainer_cfg.io.logdir               = str(Path(self.log_dir) / "trials" / f"trial_{trial.number:04d}")

        if SearchSpace.BATCH in sampled:
            self._apply_batch(trainer_cfg, dataset_cfg, sampled[SearchSpace.BATCH])

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

        try:
            _, _, best_val_loss = pipeline.run()
        except StudyStopRules.TRIAL_FAILURES as error:
            StudyStopRules.fail(trial, error, self.logger)
            raise

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> dict:
        """Runs the requested number of backbone trials against the study and returns this worker's tally.

        Args:
            study    : Shared Optuna study to optimize.
            n_trials : Trials this worker contributes.

        Returns:
            The worker's trials run by state, its best trial and the reason its run ended.
        """
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
        space              : Search space over the switched-on groups of the autoencoder's tunable parameters and the batch size.
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
        self.space       = SearchSpace.of(config_cls, tune_cfg)
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
        entry.model_overrides = SearchSpace.model_params(sampled)
        entry.run_name        = f"trial_{trial.number:04d}"
        entry.seed            = self.tune_cfg.base_seed + trial.number
        entry.logdir          = Path(self.log_dir) / "trials"

        entry.training.epochs              = self.tune_cfg.n_epochs
        entry.training.scheduler_epochs    = self.tune_cfg.n_epochs
        entry.training.early_stop_patience = self.tune_cfg.early_stop_patience
        if SearchSpace.BATCH in sampled:
            entry.training.batch_size = sampled[SearchSpace.BATCH]

        pipeline = self.trial_pipeline_cls(entry, trial=trial)
        try:
            (_, _, best_val_loss), _ = pipeline.run()
        except StudyStopRules.TRIAL_FAILURES as error:
            StudyStopRules.fail(trial, error, self.logger)
            raise

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> dict:
        """Runs the requested number of autoencoder trials against the study and returns this worker's tally.

        Args:
            study    : Shared Optuna study to optimize.
            n_trials : Trials this worker contributes.

        Returns:
            The worker's trials run by state, its best trial and the reason its run ended.
        """
        self.logger.section(f"[AeTuner — {self.model_name}]")

        rules = StudyStopRules(self.tune_cfg, self.logger)
        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search groups"        : SearchSpace.describe(self.tune_cfg),
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
            **rules.describe(),
        }, title="AeTuner config")

        progress = StudyProgress(self.log_dir, self.space, rules, self.logger)
        rules.optimize(study, self._objective, n_trials, progress, [self.best_writer])

        return progress.summary()


class JepaTuner:
    """Optuna objective running a shortened JEPA training per trial.

    Only architecture parameters are searched whatever groups the tuning
    settings switch on, since the JEPA entry owns its own learning rates;
    the run tag is composed once from the coupled autoencoders so every trial
    lands under the same name.

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
        self.space       = SearchSpace.arch_only(model_config_cls)
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

        pipeline = TrialJepaPipeline(entry, trial=trial)
        try:
            (_, _, best_val_loss), _ = pipeline.run()
        except StudyStopRules.TRIAL_FAILURES as error:
            StudyStopRules.fail(trial, error, self.logger)
            raise

        return best_val_loss

    def run(self, study: optuna.Study, n_trials: int) -> dict:
        """Runs the requested number of JEPA trials against the study and returns this worker's tally.

        Args:
            study    : Shared Optuna study to optimize.
            n_trials : Trials this worker contributes.

        Returns:
            The worker's trials run by state, its best trial and the reason its run ended.
        """
        self.logger.section(f"[JepaTuner — {self.model_name}]")

        rules = StudyStopRules(self.tune_cfg, self.logger)
        self.logger.kv_table({
            "Trials (this worker)" : n_trials,
            "Epochs / trial"       : self.tune_cfg.n_epochs,
            "Early-stop patience"  : self.tune_cfg.early_stop_patience,
            "Search groups"        : "arch",
            "Search dimensions"    : len(self.space),
            "Log dir"              : self.log_dir,
            **rules.describe(),
        }, title="JepaTuner config")

        progress = StudyProgress(self.log_dir, self.space, rules, self.logger)
        rules.optimize(study, self._objective, n_trials, progress, [self.best_writer])

        return progress.summary()
