from __future__ import annotations

import inspect
import json

import optuna

import pipelines.tuning.trial as trial_mod

from pipelines.tuning.tuners              import ParamSampler, BestConfigWriter, Tuner
from pipelines.backbone.training.pipeline import TrainingPipeline
from configuration.architectures.backbone import UNetConfig


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
    space   = FLOAT_SPACE
    sampler = ParamSampler()
    seen    = {"encoder_lr": [], "dropout": []}

    def objective(trial):
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
    sampler = ParamSampler()
    seen    = []

    def objective(trial):
        seen.append(sampler.sample(trial, CATEGORICAL_SPACE)["activation"])
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=15)

    assert set(seen) <= set(CATEGORICAL_SPACE["activation"]["choices"])
    assert "activation" in study.trials[0].params


def test_sampler_indexed_categorical_stores_index_returns_value():
    sampler = ParamSampler()
    sampled = {}

    def objective(trial):
        sampled.update(sampler.sample(trial, INDEXED_SPACE))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=2))
    study.optimize(objective, n_trials=1)

    assert sampled["features"] in INDEXED_SPACE["features"]["choices"]
    assert "features__idx" in study.trials[0].params
    assert "features"      not in study.trials[0].params


def test_sampler_decode_roundtrips_indexed_categorical():
    sampler = ParamSampler()
    params  = {"features__idx": 2, "encoder_lr": 0.001}
    decoded = sampler.decode(params, INDEXED_SPACE)

    assert decoded["features"]    == INDEXED_SPACE["features"]["choices"][2]
    assert decoded["encoder_lr"]  == 0.001
    assert "features__idx" not in decoded


def test_sampler_decode_passes_through_plain_params():
    sampler = ParamSampler()
    decoded = sampler.decode({"activation": "gelu", "dropout": 0.1}, CATEGORICAL_SPACE)

    assert decoded == {"activation": "gelu", "dropout": 0.1}


def test_study_minimizes_synthetic_quadratic_to_known_optimum():
    space   = {"x": {"type": "float", "low": -10.0, "high": 10.0}}
    sampler = ParamSampler()
    optimum = 3.0

    def objective(trial):
        s = sampler.sample(trial, space)
        return (s["x"] - optimum) ** 2

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
    study.optimize(objective, n_trials=120)

    assert study.best_value < 0.5
    assert abs(study.best_params["x"] - optimum) < 1.0


def test_best_config_writer_returns_none_without_trials(tmp_path):
    study  = optuna.create_study(direction="minimize")
    writer = BestConfigWriter("unet", FLOAT_SPACE, tmp_path / "best.json")

    assert writer.write(study) is None
    assert not (tmp_path / "best.json").exists()


def test_best_config_writer_persists_decoded_best(tmp_path):
    sampler = ParamSampler()

    def objective(trial):
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
    sampler = ParamSampler()
    path    = tmp_path / "best.json"
    writer  = BestConfigWriter("unet", FLOAT_SPACE, path)

    def objective(trial):
        return sampler.sample(trial, FLOAT_SPACE)["dropout"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5, callbacks=[writer])

    assert path.exists()


def test_tuner_space_is_union_of_lr_and_arch_params(fake_logger, tune_cfg, tmp_path):
    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = None,
        base_dataset_config = None,
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    expected = {**UNetConfig.tunable_lr_params(), **UNetConfig.tunable_arch_params()}
    assert tuner.space == expected
    assert tuner.best_writer.path == tmp_path / "best_config.json"


def test_tuner_apply_params_sets_attrs_from_sampled(fake_logger, tune_cfg, tmp_path):
    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = None,
        base_dataset_config = None,
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    cfg = UNetConfig()

    def objective(trial):
        tuner._apply_params(trial, cfg)
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=1)

    assert cfg.activation in UNetConfig.tunable_arch_params()["activation"]["choices"]
    assert cfg.features   in UNetConfig.tunable_arch_params()["features"]["choices"]
    assert 1e-5 <= cfg.encoder_lr <= 1e-2


def test_tuner_run_optimizes_with_injected_objective(fake_logger, tune_cfg, tmp_path, monkeypatch):
    optimum_dropout = 0.0

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.model_config = kwargs["model_config"]

        def run(self):
            loss = abs(self.model_config.dropout - optimum_dropout) + 0.01 * self.model_config.encoder_lr
            return None, None, float(loss)

    monkeypatch.setattr(trial_mod, "TrialPipeline", FakePipeline)

    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = _FakeTrainerConfig(),
        base_dataset_config = _FakeDatasetConfig(),
        tune_cfg            = tune_cfg,
        log_dir             = str(tmp_path),
        logger              = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8))
    tuner.run(study, n_trials=40)

    assert study.best_value < 0.2
    assert (tmp_path / "best_config.json").exists()
    assert any("Tuner" in s for s in fake_logger.sections)


def test_tuner_objective_materializes_trial_config(fake_logger, tune_cfg, tmp_path, monkeypatch):
    captured = {}

    class CapturePipeline:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None, None, 0.5

    monkeypatch.setattr(trial_mod, "TrialPipeline", CapturePipeline)

    tuner = Tuner(
        model_name          = "unet",
        model_config_cls    = UNetConfig,
        base_trainer_config = _FakeTrainerConfig(),
        base_dataset_config = _FakeDatasetConfig(),
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
    def __init__(self):
        self.logdir = ""


class _FakeTrainingLoop:
    def __init__(self):
        self.epochs = 0


class _FakeScheduler:
    def __init__(self):
        self.epochs = 0


class _FakeEarlyStop:
    def __init__(self):
        self.patience = 0


class _FakeTrainerConfig:
    def __init__(self):
        from configuration.sar.gaussian_config import GaussianConfig
        from configuration.training import LossConfig, LossCurriculumConfig

        self.training       = _FakeTrainingLoop()
        self.scheduler      = _FakeScheduler()
        self.early_stopping = _FakeEarlyStop()
        self.io             = _FakeIO()
        self.curriculum     = LossCurriculumConfig(complete=LossConfig(use_param_l1=True))
        self.gaussian       = GaussianConfig(n_default_gaussians=3, x_min=-20.0, x_max=80.0)


class _FakeDatasetConfig:
    def __init__(self):
        from configuration.dataset import AugmentationConfig

        self.augmentation = AugmentationConfig()
