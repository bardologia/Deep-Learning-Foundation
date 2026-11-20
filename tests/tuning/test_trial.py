
from __future__ import annotations

from pathlib import Path

import optuna
import pytest

import pipelines.tuning.trial as trial_mod

from pipelines.tuning.trial import TrialTrainer


def test_trial_callback_reports_intermediate_value(fake_logger):
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial
    trainer.logger = fake_logger

    trainer._trial_callback(0.42, 0)

    assert trial.storage.get_trial(trial._trial_id).intermediate_values[0] == 0.42


def test_trial_pipeline_honours_the_caller_emit_docs(monkeypatch):
    from pipelines.tuning.trial import TrialPipeline

    captured = []

    class CaptureTrainer:
        def __init__(self, **kwargs):
            captured.append(kwargs["emit_docs"])

    monkeypatch.setattr(trial_mod, "TrialTrainer", CaptureTrainer)

    pipeline                 = TrialPipeline.__new__(TrialPipeline)
    pipeline._trial          = None
    pipeline.trainer_config  = None

    for emit_trial_docs in (True, False):
        pipeline._emit_docs = emit_trial_docs

        pipeline._make_trainer(None, None, None, Path("run"), None)
        pipeline._make_trainer(None, None, None, Path("run"), None, emit_docs=False)

    assert captured == [True, False, False, False]


def test_trial_callback_no_prune_without_pruner(fake_logger):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    trial = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial
    trainer.logger = fake_logger

    trainer._trial_callback(1.0, 0)
    trainer._trial_callback(2.0, 1)


def test_trial_callback_prunes_on_bad_intermediate(fake_logger):
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial
    trainer.logger = fake_logger

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._trial_callback(5.0, 0)

    assert fake_logger.warnings == ["Trial 0 pruned at epoch 1: val_loss=5 is worse than the median of earlier trials at this step"]


def test_trial_callback_keeps_trial_under_threshold(fake_logger):
    pruner = optuna.pruners.ThresholdPruner(upper=10.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial
    trainer.logger = fake_logger

    trainer._trial_callback(0.5, 0)
