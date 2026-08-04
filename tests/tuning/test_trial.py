from __future__ import annotations

from pathlib import Path

import optuna
import pytest

import pipelines.tuning.trial as trial_mod

from pipelines.tuning.trial  import TrialTrainer, TrialProfileAeTrainer, TrialImageAeTrainer, TrialJepaTrainer
from pipelines.tuning.tuners import AeTuner, JepaTuner


def test_trial_callback_reports_intermediate_value():
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    trainer._trial_callback(0.42, 0)

    assert trial.storage.get_trial(trial._trial_id).intermediate_values[0] == 0.42


def test_trial_callback_no_prune_without_pruner():
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    trial = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    trainer._trial_callback(1.0, 0)
    trainer._trial_callback(2.0, 1)


def test_trial_callback_prunes_on_bad_intermediate():
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._trial_callback(5.0, 0)


def test_trial_callback_keeps_trial_under_threshold():
    pruner = optuna.pruners.ThresholdPruner(upper=10.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    trainer._trial_callback(0.5, 0)


def test_ae_trainer_after_eval_reports_and_prunes():
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialProfileAeTrainer.__new__(TrialProfileAeTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._after_eval(9.0, 0)

    assert trial.storage.get_trial(trial._trial_id).intermediate_values[0] == 9.0


def test_ae_trainer_after_eval_survives_good_value():
    pruner = optuna.pruners.ThresholdPruner(upper=10.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialProfileAeTrainer.__new__(TrialProfileAeTrainer)
    trainer._trial = trial

    trainer._after_eval(0.3, 0)


def test_jepa_trainer_after_eval_reports_and_prunes():
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialJepaTrainer.__new__(TrialJepaTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._after_eval(8.0, 0)


def test_image_ae_trainer_after_eval_reports_and_prunes():
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialImageAeTrainer.__new__(TrialImageAeTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._after_eval(7.0, 0)

    assert trial.storage.get_trial(trial._trial_id).intermediate_values[0] == 7.0


class FakeEntryTraining:
    def __init__(self):
        self.epochs              = 0
        self.scheduler_epochs    = 0
        self.early_stop_patience = 0


class FakeAeEntry:
    def __init__(self):
        self.training = FakeEntryTraining()


class FakeAeConfig:
    encoder_lr = 3e-4
    decoder_lr = 3e-4

    def __init__(self):
        self.encoder_lr    = 3e-4
        self.decoder_lr    = 3e-4
        self.encoder_wd    = 1e-4
        self.decoder_wd    = 1e-4
        self.embedding_dim = 24

    @classmethod
    def tunable_lr_params(cls):
        return {
            "encoder_lr"    : {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
            "embedding_dim" : {"type": "categorical", "choices": [16, 24, 32]},
        }

    @classmethod
    def tunable_arch_params(cls):
        return {"depth": {"type": "categorical", "choices": [3, 4, 6]}}


def test_ae_tuner_objective_materializes_entry(fake_logger, tune_cfg, tmp_path):
    captured = {}

    class CaptureAePipeline:
        def __init__(self, entry, trial):
            captured["entry"] = entry
            captured["trial"] = trial

        def run(self):
            return (None, None, 0.5), None

    tuner = AeTuner(
        model_name         = "mlp_ae",
        config_cls         = FakeAeConfig,
        entry_template     = FakeAeEntry(),
        trial_pipeline_cls = CaptureAePipeline,
        tune_cfg           = tune_cfg,
        log_dir            = str(tmp_path),
        logger             = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1)

    entry = captured["entry"]
    assert entry.ae_model_name == "mlp_ae"
    assert entry.run_name      == "trial_0000"
    assert entry.seed          == tune_cfg.base_seed + 0
    assert entry.logdir        == Path(tmp_path) / "trials"

    assert entry.training.epochs              == tune_cfg.n_epochs
    assert entry.training.scheduler_epochs    == tune_cfg.n_epochs
    assert entry.training.early_stop_patience == tune_cfg.early_stop_patience

    assert entry.model_overrides["embedding_dim"] in [16, 24, 32]
    assert entry.model_overrides["depth"]         in [3, 4, 6]
    assert set(entry.model_overrides) == {"encoder_lr", "embedding_dim", "depth"}


def test_ae_tuner_objective_returns_best_val_loss(fake_logger, tune_cfg, tmp_path):
    class FakeAePipeline:
        def __init__(self, entry, trial):
            self.entry = entry

        def run(self):
            return (None, None, 0.7), None

    tuner = AeTuner(
        model_name         = "mlp_ae",
        config_cls         = FakeAeConfig,
        entry_template     = FakeAeEntry(),
        trial_pipeline_cls = FakeAePipeline,
        tune_cfg           = tune_cfg,
        log_dir            = str(tmp_path),
        logger             = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1)

    assert study.best_value == 0.7


def test_ae_tuner_image_entry_gets_model_overrides(fake_logger, tune_cfg, tmp_path):
    captured = {}

    class CaptureImageAePipeline:
        def __init__(self, entry, trial):
            captured["entry"] = entry

        def run(self):
            return (None, None, 0.3), None

    tuner = AeTuner(
        model_name         = "conv2d_ae",
        config_cls         = FakeAeConfig,
        entry_template     = FakeAeEntry(),
        trial_pipeline_cls = CaptureImageAePipeline,
        tune_cfg           = tune_cfg,
        log_dir            = str(tmp_path),
        logger             = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1)

    entry = captured["entry"]
    assert entry.ae_model_name == "conv2d_ae"
    assert entry.model_overrides["embedding_dim"] in [16, 24, 32]


class FakeJepaConfig:
    @classmethod
    def tunable_arch_params(cls):
        return {
            "depth"      : {"type": "categorical", "choices": [2, 4, 8]},
            "hidden_dim" : {"type": "categorical", "choices": [64, 128]},
        }


class FakeJepaTraining:
    def __init__(self):
        self.epochs              = 0
        self.scheduler_epochs    = 0
        self.early_stop_patience = 0


class FakeJepaEntry:
    def __init__(self, test_data_dir: Path):
        from configuration.dataset               import AugmentationConfig
        from configuration.training              import LossConfig
        from configuration.training.general.run  import TrainingPathsConfig

        self.training     = FakeJepaTraining()
        self.param_loss   = LossConfig(use_param_l1=True)
        self.paths        = TrainingPathsConfig(dataset_path=test_data_dir, parameters_path=test_data_dir / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy")
        self.augmentation = AugmentationConfig()


@pytest.mark.real_data
def test_jepa_tuner_objective_sets_model_overrides(fake_logger, tune_cfg, tmp_path, monkeypatch, test_data_dir):
    captured = {}

    class CaptureJepaPipeline:
        def __init__(self, entry, trial):
            captured["entry"] = entry

        def run(self):
            return (None, None, 0.4), None

    monkeypatch.setattr(trial_mod, "TrialJepaPipeline", CaptureJepaPipeline)

    tuner = JepaTuner(
        model_name       = "vit",
        model_config_cls = FakeJepaConfig,
        entry_template   = FakeJepaEntry(test_data_dir),
        tune_cfg         = tune_cfg,
        log_dir          = str(tmp_path),
        logger           = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1)

    entry = captured["entry"]
    assert entry.backbone_name == "vit"
    assert entry.run_name      == "vit-conv-hungarian-K_5-hv-none-param_l1_0.1_trial_0000"
    assert set(entry.model_overrides.keys()) == {"depth", "hidden_dim"}
    assert entry.model_overrides["depth"]      in [2, 4, 8]
    assert entry.model_overrides["hidden_dim"] in [64, 128]


@pytest.mark.real_data
def test_jepa_tuner_space_uses_arch_params_only(fake_logger, tune_cfg, tmp_path, test_data_dir):
    tuner = JepaTuner(
        model_name       = "vit",
        model_config_cls = FakeJepaConfig,
        entry_template   = FakeJepaEntry(test_data_dir),
        tune_cfg         = tune_cfg,
        log_dir          = str(tmp_path),
        logger           = fake_logger,
    )

    assert tuner.space == FakeJepaConfig.tunable_arch_params()
