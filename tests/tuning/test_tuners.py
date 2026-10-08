"""Tests for parameter sampling, search-space selection, stop rules, best-config persistence and the Tuner.

Covers float, categorical and indexed categorical sampling and decoding, the
rejection of an unknown dimension type, the search groups a study may switch
on, the trial target, plateau and time-budget stop rules and the reason a
worker's run ended, the per-trial progress record and study snapshot, failing
trials, convergence on a synthetic objective, best-config writing and the
trial configs the backbone Tuner materializes.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta

import optuna
import pytest

import pipelines.tuning.trial as trial_mod

from optuna.trial import TrialState

from pipelines.tuning.tuners              import ParamSampler, BestConfigWriter, SearchSpace, StudyPlateauStop, StudyProgress, StudyStopRules, StudyTargetStop, Tuner
from pipelines.backbone.training.pipeline import TrainingPipeline
from configuration.architectures.backbone import UNetConfig
from configuration.training.general.run   import TrainingQueueConfig


FLOAT_SPACE = {
    "encoder_lr" : {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "dropout"    : {"type": "float", "low": 0.0, "high": 0.5},
}

CATEGORICAL_SPACE = {
    "activation" : {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
}

INDEXED_SPACE = {
    "features" : {"type": "indexed_categorical", "choices": [[32, 64], [64, 128], [48, 96]]},
}


def test_sampler_float_respects_bounds_and_log():
    """Verifies that sampled floats stay inside their bounds and keep the log flag."""
    space   = FLOAT_SPACE
    sampler = ParamSampler()
    seen    = {"encoder_lr": [], "dropout": []}

    def objective(trial):
        """Records the sampled parameters and returns a constant loss."""
        s = sampler.sample(trial, space)
        seen["encoder_lr"].append(s["encoder_lr"])
        seen["dropout"].append(s["dropout"])
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=20)

    assert all(1e-5 <= v <= 1e-2 for v in seen["encoder_lr"])
    assert all(0.0  <= v <= 0.5  for v in seen["dropout"])

    dist = study.trials[0].distributions["encoder_lr"]
    assert dist.log is True


def test_sampler_categorical_only_from_choices():
    """Verifies that categorical samples come only from the declared choices."""
    sampler = ParamSampler()
    seen    = []

    def objective(trial):
        """Records the sampled activation and returns a constant loss."""
        seen.append(sampler.sample(trial, CATEGORICAL_SPACE)["activation"])
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=15)

    assert set(seen) <= set(CATEGORICAL_SPACE["activation"]["choices"])
    assert "activation" in study.trials[0].params


def test_sampler_indexed_categorical_stores_index_returns_value():
    """Verifies that indexed categoricals store an index but hand back the choice value."""
    sampler = ParamSampler()
    sampled = {}

    def objective(trial):
        """Records the sampled parameters and returns a constant loss."""
        sampled.update(sampler.sample(trial, INDEXED_SPACE))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=2))
    study.optimize(objective, n_trials=1)

    assert sampled["features"] in INDEXED_SPACE["features"]["choices"]
    assert "features__idx" in study.trials[0].params
    assert "features"      not in study.trials[0].params


def test_sampler_rejects_an_unknown_parameter_type():
    """Verifies that a dimension with an undeclared type raises instead of being left out of the sample."""
    trial = optuna.create_study().ask()

    with pytest.raises(ValueError, match="Unknown search space parameter type 'int' for parameter 'depth'"):
        ParamSampler().sample(trial, {"depth": {"type": "int", "low": 2, "high": 8}})


def test_sampler_decode_roundtrips_indexed_categorical():
    """Verifies that decoding turns a stored index back into the choice value."""
    sampler = ParamSampler()
    params  = {"features__idx": 2, "encoder_lr": 0.001}
    decoded = sampler.decode(params, INDEXED_SPACE)

    assert decoded["features"]    == INDEXED_SPACE["features"]["choices"][2]
    assert decoded["encoder_lr"]  == 0.001
    assert "features__idx" not in decoded


def test_sampler_decode_rejects_an_index_the_space_no_longer_holds():
    """Verifies that an index for a parameter absent from the space raises KeyError."""
    sampler = ParamSampler()

    with pytest.raises(KeyError, match="renamed_features"):
        sampler.decode({"renamed_features__idx": 0}, INDEXED_SPACE)

    with pytest.raises(KeyError, match="encoder_lr"):
        sampler.decode({"encoder_lr__idx": 0}, FLOAT_SPACE)


def test_sampler_decode_passes_through_plain_params():
    """Verifies that non-indexed parameters pass through decoding unchanged."""
    sampler = ParamSampler()
    decoded = sampler.decode({"activation": "gelu", "dropout": 0.1}, CATEGORICAL_SPACE)

    assert decoded == {"activation": "gelu", "dropout": 0.1}


def test_search_space_joins_the_switched_on_groups(tune_cfg):
    """Both model groups by default, one group when the other is off, and the batch size as a categorical when asked."""
    assert SearchSpace.of(UNetConfig, tune_cfg) == {**UNetConfig.tunable_lr_params(), **UNetConfig.tunable_arch_params()}

    tune_cfg.search_arch = False
    assert SearchSpace.of(UNetConfig, tune_cfg) == UNetConfig.tunable_lr_params()

    tune_cfg.search_batch = True
    tune_cfg.batch_sizes  = [128, 256]
    space = SearchSpace.of(UNetConfig, tune_cfg)
    assert space["batch_size"] == {"type": "categorical", "choices": [128, 256]}
    assert SearchSpace.model_params({"encoder_lr": 1e-3, "batch_size": 128}) == {"encoder_lr": 1e-3}
    assert SearchSpace.describe(tune_cfg) == "lr+batch"


def test_search_space_refuses_nothing_to_search_and_empty_groups(tune_cfg):
    """Every group off, an empty batch list and a class declaring nothing of a switched-on group all fail loudly."""
    class ArchOnly:
        """Config stand-in declaring architecture parameters only."""
        @classmethod
        def tunable_lr_params(cls):
            """Declares no learning-rate parameter."""
            return {}

        @classmethod
        def tunable_arch_params(cls):
            """Declares one categorical architecture parameter."""
            return {"activation": {"type": "categorical", "choices": ["relu", "gelu"]}}

    tune_cfg.search_lr   = False
    tune_cfg.search_arch = False
    with pytest.raises(ValueError, match="every search group is off"):
        SearchSpace.of(UNetConfig, tune_cfg)

    tune_cfg.search_batch = True
    tune_cfg.batch_sizes  = []
    with pytest.raises(ValueError, match="batch_sizes is empty"):
        SearchSpace.of(UNetConfig, tune_cfg)

    tune_cfg.search_lr    = True
    tune_cfg.search_batch = False
    with pytest.raises(ValueError, match="declares no learning-rate parameter"):
        SearchSpace.of(ArchOnly, tune_cfg)

    assert SearchSpace.arch_only(ArchOnly) == ArchOnly.tunable_arch_params()


def _study_with_values(values):
    """Returns a minimising study holding one finished trial per value, pruned where the value is None."""
    study = optuna.create_study(direction="minimize")

    for value in values:
        if value is None:
            study.add_trial(optuna.trial.create_trial(state=TrialState.PRUNED, params={}, distributions={}))
        else:
            study.add_trial(optuna.trial.create_trial(value=value, state=TrialState.COMPLETE, params={}, distributions={}))

    return study


def test_plateau_stop_counts_finished_trials_since_the_last_counted_improvement(fake_logger):
    """Pruned trials count towards the patience, and a gain below the margin does not reset it."""
    rule = StudyPlateauStop(patience=3, min_delta=0.05, logger=fake_logger)

    assert rule.stalled(_study_with_values([1.0, 0.9, 0.95, None, 0.92])) == 3
    assert rule.stalled(_study_with_values([1.0, 0.9, 0.95, None, 0.80])) == 0
    assert rule.stalled(_study_with_values([None, None, 1.0]))            == 0
    assert rule.stalled(_study_with_values([]))                            == 0


def test_plateau_stop_ends_a_study_after_the_patience_without_progress(fake_logger):
    """A constant objective stops the study after the first trial plus the patience, and the rule says so with the anchor and reference."""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))
    rule  = StudyPlateauStop(patience=4, min_delta=0.0, logger=fake_logger)

    study.optimize(lambda trial: 1.0 + 0.0 * trial.suggest_float("x", 0.0, 1.0), n_trials=50, callbacks=[rule])

    assert len(study.trials) == 5
    assert rule.fired is True
    assert fake_logger.infos == ["Study stop: plateau, 4 finished trials since trial 0 without a 0.0% improvement on 1"]


def test_plateau_stop_keeps_running_while_the_best_keeps_improving(fake_logger):
    """An objective improving beyond the margin on every trial never trips the rule."""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))
    rule  = StudyPlateauStop(patience=3, min_delta=0.01, logger=fake_logger)

    study.optimize(lambda trial: 1.0 / (len(study.trials) + 0.0 * trial.suggest_float("x", 0.0, 1.0)), n_trials=20, callbacks=[rule])

    assert len(study.trials) == 20
    assert rule.fired is False
    assert fake_logger.infos == []


def test_target_stop_logs_the_stop_once_the_completed_count_reaches_the_target(fake_logger):
    """The target rule stops the worker at the completed count, says so, and reports the target as reached from then on."""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))
    rule  = StudyTargetStop(target=3, logger=fake_logger)

    assert rule.reached(study) is False

    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=10, callbacks=[rule])

    assert len(study.trials) == 3
    assert rule.fired is True
    assert rule.reached(study) is True
    assert fake_logger.infos == ["Study stop: target reached, 3 complete + 0 pruned = 3 of target 3"]


def test_target_stop_counts_pruned_trials_as_the_scheduler_does_but_not_failed_ones(fake_logger):
    """A study of one completed, one pruned and one failed trial has reached a target of two and not one of three."""
    study = _study_with_values([0.5, None])
    study.add_trial(optuna.trial.create_trial(state=TrialState.FAIL))

    assert StudyTargetStop(target=2, logger=fake_logger).reached(study) is True
    assert StudyTargetStop(target=3, logger=fake_logger).reached(study) is False


def test_stop_rules_name_the_rule_that_ended_the_run(fake_logger, tune_cfg):
    """The reason follows Optuna's order: a fired target or plateau, then a used-up share, then the time budget, and anything else is an error."""
    tune_cfg.plateau_patience = 2
    tune_cfg.time_budget_h    = 1.0
    rules = StudyStopRules(tune_cfg, fake_logger)

    assert rules.reason(n_trials=5, trials_run=5, elapsed_s=10.0) == StudyStopRules.WORKER_SHARE
    assert rules.reason(n_trials=5, trials_run=2, elapsed_s=3600.0) == StudyStopRules.TIME_BUDGET
    assert any("share of 5 trial(s) is used up" in message for message in fake_logger.infos)
    assert any("time budget of 1.0 h used up after 2 trial(s)" in message for message in fake_logger.infos)

    with pytest.raises(RuntimeError, match="no stop rule fired"):
        rules.reason(n_trials=5, trials_run=2, elapsed_s=10.0)

    rules.plateau.fired = True
    assert rules.reason(n_trials=5, trials_run=2, elapsed_s=10.0) == StudyStopRules.PLATEAU

    rules.target.fired = True
    assert rules.reason(n_trials=5, trials_run=2, elapsed_s=10.0) == StudyStopRules.TARGET


def test_optimize_records_the_stop_reason_in_the_snapshot_and_the_worker_tally(fake_logger, tune_cfg, tmp_path):
    """Running a study under the rules writes the reason into study_progress.json and the worker's tally names its trials, best value and reason."""
    tune_cfg.n_trials = 3
    rules    = StudyStopRules(tune_cfg, fake_logger)
    progress = StudyProgress(tmp_path, {"x": {"type": "float", "low": 0.0, "high": 1.0}}, rules, fake_logger)
    study    = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))

    reason  = rules.optimize(study, lambda trial: trial.suggest_float("x", 0.0, 1.0), 5, progress, [])
    written = json.loads((tmp_path / "study_progress.json").read_text())
    summary = progress.summary()

    assert reason                 == StudyStopRules.TARGET
    assert written["stop_reason"] == StudyStopRules.TARGET
    assert summary["trials_run"]  == 3
    assert summary["complete"]    == 3
    assert summary["failed"]      == 0
    assert summary["stop_reason"] == StudyStopRules.TARGET
    assert summary["best_value"]  == study.best_value
    assert summary["best_trial"]  == study.best_trial.number
    assert any(message.startswith("Worker finished under the target stop after 3 trial(s): 3 complete, 0 pruned, 0 failed") for message in fake_logger.infos)


def test_study_progress_snapshot_counts_states_best_stall_and_rate(fake_logger, tune_cfg, tmp_path):
    """The snapshot reports the state counts, the best completed trial decoded, the stall count (failed trials do not count) and a positive finishing rate."""
    tune_cfg.plateau_patience = 3
    space    = {"dropout": {"type": "float", "low": 0.0, "high": 0.5}, "features": {"type": "indexed_categorical", "choices": [[32, 64], [64, 128]]}}
    progress = StudyProgress(tmp_path, space, StudyStopRules(tune_cfg, fake_logger), fake_logger)
    study    = optuna.create_study(direction="minimize")
    dists    = {"dropout": optuna.distributions.FloatDistribution(0.0, 0.5), "features__idx": optuna.distributions.CategoricalDistribution([0, 1])}

    study.add_trial(optuna.trial.create_trial(value=0.5, state=TrialState.COMPLETE, params={"dropout": 0.1, "features__idx": 1}, distributions=dists))
    study.add_trial(optuna.trial.create_trial(value=0.3, state=TrialState.COMPLETE, params={"dropout": 0.2, "features__idx": 0}, distributions=dists))
    study.add_trial(optuna.trial.create_trial(state=TrialState.PRUNED, params={"dropout": 0.4, "features__idx": 0}, distributions=dists))
    study.add_trial(optuna.trial.create_trial(state=TrialState.FAIL))

    snapshot = progress.snapshot(study)

    assert snapshot["trials"]   == {"complete": 2, "pruned": 1, "fail": 1, "running": 0}
    assert snapshot["finished"] == 4
    assert snapshot["target"]   == tune_cfg.n_trials
    assert snapshot["best"]     == {"trial": 1, "value": 0.3, "params": {"dropout": 0.2, "features": [32, 64]}}
    assert snapshot["stalled"]  == 1
    assert snapshot["plateau_patience"] == 3
    assert snapshot["trials_per_hour"] > 0


def test_study_progress_rate_ignores_the_idle_gap_before_a_resume(fake_logger, tune_cfg, tmp_path):
    """Trials finished three days before this launch count towards the study's age, not towards the rate or the time to the target."""
    progress = StudyProgress(tmp_path, {}, StudyStopRules(tune_cfg, fake_logger), fake_logger)
    study    = optuna.create_study(direction="minimize")
    past     = datetime.now() - timedelta(days=3)

    for minutes in (0, 10):
        trial                   = optuna.trial.create_trial(value=0.5, state=TrialState.COMPLETE, params={}, distributions={})
        trial.datetime_start    = past + timedelta(minutes=minutes)
        trial.datetime_complete = past + timedelta(minutes=minutes + 5)
        study.add_trial(trial)

    progress.started = datetime.now() - timedelta(hours=1)
    assert progress.snapshot(study)["trials_per_hour"] == 0.0
    assert progress.snapshot(study)["eta_target_h"]    is None

    study.add_trial(optuna.trial.create_trial(value=0.4, state=TrialState.COMPLETE, params={}, distributions={}))
    snapshot = progress.snapshot(study)

    assert 0.9 < snapshot["trials_per_hour"] < 1.1
    assert 0.9 < snapshot["elapsed_h"]       < 1.1
    assert snapshot["study_age_h"] > 71.0
    assert 8.0 < snapshot["eta_target_h"] < 10.0


def test_study_progress_callback_logs_one_line_and_writes_the_snapshot(fake_logger, tune_cfg, tmp_path):
    """After every trial the callback appends one record to the log and rewrites study_progress.json."""
    space    = {"x": {"type": "float", "low": 0.0, "high": 1.0}}
    progress = StudyProgress(tmp_path, space, StudyStopRules(tune_cfg, fake_logger), fake_logger)
    study    = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))

    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=3, callbacks=[progress])

    lines = [message for message in fake_logger.infos if message.startswith("Trial ")]
    assert len(lines) == 3
    assert "COMPLETE value=" in lines[-1]
    assert "3 complete, 0 pruned, 0 failed, 0 running of 12" in lines[-1]
    assert "stalled off" in lines[-1]

    written = json.loads((tmp_path / "study_progress.json").read_text())
    assert written["trials"]["complete"] == 3
    assert written["best"]["value"]      == study.best_value


def test_a_failing_trial_is_recorded_as_failed_and_the_worker_goes_on(fake_logger, tune_cfg, tmp_path, monkeypatch):
    """A trial raising a divergence or device error ends as FAIL, is logged with its error, and the next trial still runs."""
    class FlakyPipeline:
        """Training pipeline stub failing on the first trial only."""
        def __init__(self, *args, **kwargs):
            """Records the trial number."""
            self.number = kwargs["trial"].number

        def run(self):
            """Raises the trainer's divergence error once, then returns a loss."""
            if self.number == 0:
                raise FloatingPointError("train loss is non-finite at step 7")
            return None, None, 0.4

    monkeypatch.setattr(trial_mod, "TrialPipeline", FlakyPipeline)

    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = _FakeTrainerConfig(),
        base_dataset_config = _FakeDatasetConfig(),
        training            = TrainingQueueConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    study   = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    summary = tuner.run(study, n_trials=2)

    assert [trial.state for trial in study.trials] == [TrialState.FAIL, TrialState.COMPLETE]
    assert any("Trial 0 failed and is recorded as such: FloatingPointError" in message for message in fake_logger.errors)
    assert any(message.startswith("Trial 0 FAIL because FloatingPointError: train loss is non-finite at step 7 (") for message in fake_logger.infos)
    assert study.trials[0].user_attrs["fail_reason"] == "FloatingPointError: train loss is non-finite at step 7"
    assert summary == {"trials_run": 2, "complete": 1, "pruned": 0, "failed": 1, "best_trial": 1, "best_value": 0.4, "stop_reason": StudyStopRules.WORKER_SHARE}


def test_a_failed_trial_without_an_exception_says_optuna_rejected_its_value(fake_logger, tune_cfg, tmp_path):
    """A NaN objective fails the trial without an exception, and the progress line says so instead of naming no reason."""
    progress = StudyProgress(tmp_path, {}, StudyStopRules(tune_cfg, fake_logger), fake_logger)
    study    = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: float("nan"), n_trials=1, callbacks=[progress])

    assert fake_logger.infos[-1].startswith("Trial 0 FAIL because Optuna rejected the objective value (")


def test_stop_rules_follow_the_tuning_settings(tune_cfg, fake_logger):
    """The target stop is always on, the plateau stop and the timeout follow their settings."""
    rules = StudyStopRules(tune_cfg, fake_logger)

    assert [type(rule) for rule in rules.callbacks()] == [StudyTargetStop]
    assert rules.timeout_s() is None
    assert rules.describe()["Plateau patience"] == "off"
    assert rules.describe()["Time budget"]      == "none"

    tune_cfg.plateau_patience  = 200
    tune_cfg.plateau_min_delta = 0.01
    tune_cfg.time_budget_h     = 1.5

    rules     = StudyStopRules(tune_cfg, fake_logger)
    callbacks = rules.callbacks()

    assert [type(rule) for rule in callbacks] == [StudyTargetStop, StudyPlateauStop]
    assert callbacks[1].patience  == 200
    assert callbacks[1].min_delta == 0.01
    assert rules.timeout_s()      == 5400.0
    assert rules.describe()["Plateau patience"] == 200
    assert rules.describe()["Time budget"]      == "1.5 h"


def _never_called(trial):
    """Objective that fails the test when a worker runs a trial it should not."""
    raise AssertionError("the worker ran a trial although the study was already at its stop")


def test_a_worker_attaching_to_a_study_at_its_target_runs_no_trial(fake_logger, tune_cfg, tmp_path):
    """A study that already holds the target's completed trials ends the attaching worker at once, with a warning and the target as its reason."""
    tune_cfg.n_trials = 2
    rules    = StudyStopRules(tune_cfg, fake_logger)
    progress = StudyProgress(tmp_path, {}, rules, fake_logger)
    study    = _study_with_values([0.5, 0.4])

    reason = rules.optimize(study, _never_called, 3, progress, [])

    assert reason == StudyStopRules.TARGET
    assert rules.already_met(study) == StudyStopRules.TARGET
    assert progress.summary()["trials_run"] == 0
    assert json.loads((tmp_path / "study_progress.json").read_text())["stop_reason"] == StudyStopRules.TARGET
    assert fake_logger.warnings == ["Study already met its target rule before this worker ran a trial (2 complete, 0 pruned, 0 failed, 0 running of target 2); this worker runs none"]


def test_a_worker_attaching_to_a_plateaued_study_runs_no_trial(fake_logger, tune_cfg, tmp_path):
    """A study whose best value has stalled past the patience ends the attaching worker at once with the plateau as its reason."""
    tune_cfg.n_trials         = 50
    tune_cfg.plateau_patience = 2
    rules    = StudyStopRules(tune_cfg, fake_logger)
    progress = StudyProgress(tmp_path, {}, rules, fake_logger)

    assert rules.already_met(_study_with_values([1.0, 0.9])) is None
    assert rules.optimize(_study_with_values([1.0, 1.0, None]), _never_called, 3, progress, []) == StudyStopRules.PLATEAU
    assert any("already met its plateau rule" in message for message in fake_logger.warnings)


def test_study_minimizes_synthetic_quadratic_to_known_optimum():
    """Verifies that TPE sampling over the sampler finds the synthetic optimum."""
    space   = {"x": {"type": "float", "low": -10.0, "high": 10.0}}
    sampler = ParamSampler()
    optimum = 3.0

    def objective(trial):
        """Returns the squared distance of the sampled x from the optimum."""
        s = sampler.sample(trial, space)
        return (s["x"] - optimum) ** 2

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
    study.optimize(objective, n_trials=120)

    assert study.best_value < 0.5
    assert abs(study.best_params["x"] - optimum) < 1.0


def test_best_config_writer_returns_none_without_trials(tmp_path):
    """Verifies that no file is written when the study has no completed trials."""
    study  = optuna.create_study(direction="minimize")
    writer = BestConfigWriter("unet", FLOAT_SPACE, tmp_path / "best.json")

    assert writer.write(study) is None
    assert not (tmp_path / "best.json").exists()


def test_best_config_writer_persists_decoded_best(tmp_path):
    """Verifies that the written payload holds the decoded best parameters."""
    sampler = ParamSampler()

    def objective(trial):
        """Returns the first element of the sampled feature list as the loss."""
        s = sampler.sample(trial, INDEXED_SPACE)
        return float(s["features"][0])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=12)

    path    = tmp_path / "best_config.json"
    writer  = BestConfigWriter("unet", INDEXED_SPACE, path)
    payload = writer.write(study)

    assert path.exists()
    assert payload["model"] == "unet"
    assert payload["trial"] == study.best_trial.number
    assert payload["val_loss"] == study.best_value
    assert payload["params"]["features"] in INDEXED_SPACE["features"]["choices"]

    on_disk = json.loads(path.read_text())
    assert on_disk["params"]["features"] == payload["params"]["features"]


def test_best_config_writer_callback_writes_during_optimize(tmp_path):
    """Verifies that the writer used as an Optuna callback writes during optimization."""
    sampler = ParamSampler()
    path    = tmp_path / "best.json"
    writer  = BestConfigWriter("unet", FLOAT_SPACE, path)

    def objective(trial):
        """Returns the sampled dropout as the loss."""
        return sampler.sample(trial, FLOAT_SPACE)["dropout"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5, callbacks=[writer])

    assert path.exists()


def test_tuner_space_is_union_of_lr_and_arch_params(fake_logger, tune_cfg, tmp_path):
    """Verifies that the tuner space is the LR and architecture union and names the output file."""
    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = None,
        base_dataset_config = None,
        training            = TrainingQueueConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    expected = {**UNetConfig.tunable_lr_params(), **UNetConfig.tunable_arch_params()}
    assert tuner.space == expected
    assert tuner.best_writer.path == tmp_path / "best_config.json"


def test_tuner_space_follows_the_switched_on_groups(fake_logger, tune_cfg, tmp_path):
    """With the architecture group off the tuner's space holds the rates alone."""
    tune_cfg.search_arch = False

    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = None,
        base_dataset_config = None,
        training            = TrainingQueueConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    assert tuner.space == UNetConfig.tunable_lr_params()
    assert "activation" not in tuner.space


def test_tuner_apply_params_sets_attrs_from_sampled(fake_logger, tune_cfg, tmp_path):
    """Verifies that sampled parameters are assigned onto the model config."""
    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = None,
        base_dataset_config = None,
        training            = TrainingQueueConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    cfg = UNetConfig()

    def objective(trial):
        """Applies the sampled parameters to the config and returns a constant loss."""
        tuner._apply_params(trial, cfg)
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=1)

    assert cfg.activation in UNetConfig.tunable_arch_params()["activation"]["choices"]
    assert cfg.features   in UNetConfig.tunable_arch_params()["features"]["choices"]
    assert 1e-5 <= cfg.encoder_lr <= 1e-2


def test_tuner_run_optimizes_with_injected_objective(fake_logger, tune_cfg, tmp_path, monkeypatch):
    """Verifies that the tuner drives the study toward the synthetic optimum and saves the best config."""
    optimum_dropout = 0.0

    class FakePipeline:
        """Training pipeline stub whose loss is driven by the sampled model config."""
        def __init__(self, *args, **kwargs):
            """Stores the model config handed to the pipeline."""
            self.model_config = kwargs["model_config"]

        def run(self):
            """Returns a loss increasing with the distance of dropout from the optimum."""
            loss = abs(self.model_config.dropout - optimum_dropout) + 0.01 * self.model_config.encoder_lr
            return None, None, float(loss)

    monkeypatch.setattr(trial_mod, "TrialPipeline", FakePipeline)

    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = _FakeTrainerConfig(),
        base_dataset_config = _FakeDatasetConfig(),
        training            = TrainingQueueConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8))
    tuner.run(study, n_trials=40)

    assert study.best_value < 0.2
    assert (tmp_path / "best_config.json").exists()
    assert any("Tuner" in s for s in fake_logger.sections)


def test_tuner_puts_a_sampled_batch_on_the_loader_and_rescales_the_optimizer(fake_logger, tune_cfg, tmp_path, monkeypatch):
    """A sampled batch size lands on the dataset config and the optimizer scale follows the batch rule, not on the model."""
    captured = {}

    class CapturePipeline:
        """Training pipeline stub capturing its keyword arguments."""
        def __init__(self, *args, **kwargs):
            """Records the construction keyword arguments."""
            captured.update(kwargs)

        def run(self):
            """Returns a fixed validation loss."""
            return None, None, 0.5

    monkeypatch.setattr(trial_mod, "TrialPipeline", CapturePipeline)

    tune_cfg.search_lr    = False
    tune_cfg.search_arch  = False
    tune_cfg.search_batch = True
    tune_cfg.batch_sizes  = [128]

    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = _FakeTrainerConfig(),
        base_dataset_config = _FakeDatasetConfig(),
        training            = TrainingQueueConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1)

    assert study.trials[0].params == {"batch_size": 128}
    assert captured["dataset_config"].batch_size         == 128
    assert captured["trainer_config"].optimizer.lr_scale == 128 / TrainingQueueConfig().lr_reference_batch_size
    assert not hasattr(captured["model_config"], "batch_size")
    assert captured["model_config"].encoder_lr == UNetConfig().encoder_lr


def test_tuner_objective_materializes_trial_config(fake_logger, tune_cfg, tmp_path, monkeypatch):
    """Verifies that the trial pipeline receives the run name, seed and epoch budget."""
    captured = {}

    class CapturePipeline:
        """Training pipeline stub capturing its keyword arguments."""
        def __init__(self, *args, **kwargs):
            """Records the construction keyword arguments."""
            captured.update(kwargs)

        def run(self):
            """Returns a fixed validation loss of 0.5."""
            return None, None, 0.5

    monkeypatch.setattr(trial_mod, "TrialPipeline", CapturePipeline)

    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = _FakeTrainerConfig(),
        base_dataset_config = _FakeDatasetConfig(),
        training            = TrainingQueueConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
        emit_trial_docs     = True,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1)

    assert captured["backbone_name"] == "unet"
    assert captured["seed"]          == tune_cfg.base_seed + 0
    assert captured["run_name"]      == "unet-conv-hungarian-K_3-hv-none-param_l1_0.1_trial_0000"
    assert captured["emit_docs"]     is True
    assert captured["trial"].number  == 0

    forwarded = set(captured) - {"trial", "emit_docs"}
    accepted  = set(inspect.signature(TrainingPipeline.__init__).parameters) - {"self"}
    assert forwarded <= accepted

    tcfg = captured["trainer_config"]
    assert tcfg.training.epochs         == tune_cfg.n_epochs
    assert tcfg.scheduler.epochs        == tune_cfg.n_epochs
    assert tcfg.early_stopping.patience == tune_cfg.early_stop_patience
    assert "trial_0000" in tcfg.io.logdir


class _FakeIO:
    """IO section stub carrying the log directory."""
    def __init__(self):
        """Initialises an empty log directory."""
        self.logdir = ""


class _FakeTrainingLoop:
    """Training loop section stub carrying the epoch budget."""
    def __init__(self):
        """Initialises the epoch budget to zero."""
        self.epochs = 0


class _FakeScheduler:
    """Scheduler section stub carrying the epoch budget."""
    def __init__(self):
        """Initialises the epoch budget to zero."""
        self.epochs = 0


class _FakeEarlyStop:
    """Early stopping section stub carrying the patience."""
    def __init__(self):
        """Initialises the patience to zero."""
        self.patience = 0


class _FakeOptimizer:
    """Optimizer section stub carrying the learning-rate scale."""
    def __init__(self):
        """Initialises the scale to one."""
        self.lr_scale = 1.0


class _FakeTrainerConfig:
    """Trainer config stub with the sections the tuner rewrites per trial."""
    def __init__(self):
        """Builds the training, scheduler, early stopping, IO, optimizer, curriculum and gaussian sections."""
        from configuration.dataset.gaussian import GaussianConfig
        from configuration.training import LossConfig, LossCurriculumConfig

        self.training       = _FakeTrainingLoop()
        self.scheduler      = _FakeScheduler()
        self.early_stopping = _FakeEarlyStop()
        self.io             = _FakeIO()
        self.optimizer      = _FakeOptimizer()
        self.curriculum     = LossCurriculumConfig(complete=LossConfig(use_param_l1=True))
        self.gaussian       = GaussianConfig(n_default_gaussians=3, x_min=-20.0, x_max=80.0, activity_threshold=1e-4)


class _FakeDatasetConfig:
    """Dataset config stub carrying an augmentation section and a batch size."""
    def __init__(self):
        """Builds the default augmentation section and the reference batch."""
        from configuration.dataset import AugmentationConfig

        self.augmentation = AugmentationConfig()
        self.batch_size   = 64
