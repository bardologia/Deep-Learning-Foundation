"""Tests for trial trainers and the AE and JEPA tuners.

Covers intermediate value reporting and pruning from the trainer callbacks,
and the entry configs the tuners materialize for each trial.
"""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

import pipelines.tuning.trial as trial_mod

from pipelines.tuning.trial             import TrialTrainer, TrialProfileAeTrainer, TrialImageAeTrainer, TrialJepaTrainer
from pipelines.tuning.tuners            import AeTuner, JepaTuner
from configuration.training.general.run import DEFAULT_PARAMS_RUN
from configuration.synthesis            import SynthesisConfig


SCENE_K_TAG = f"K_{SynthesisConfig().k_max}"


def test_trial_callback_reports_intermediate_value():
    """Verifies that the trainer callback reports the loss as an intermediate value."""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    trainer._trial_callback(0.42, 0)

    assert trial.storage.get_trial(trial._trial_id).intermediate_values[0] == 0.42


def test_trial_pipeline_honours_the_caller_emit_docs(monkeypatch):
    """Verifies that the pipeline's emit_docs default is overridden by an explicit argument."""
    from pipelines.tuning.trial import TrialPipeline

    captured = []

    class CaptureTrainer:
        """Trainer stub recording the emit_docs flag it was constructed with."""
        def __init__(self, **kwargs):
            """Records the emit_docs keyword argument."""
            captured.append(kwargs["emit_docs"])

    monkeypatch.setattr(trial_mod, "TrialTrainer", CaptureTrainer)

    pipeline                 = TrialPipeline.__new__(TrialPipeline)
    pipeline._trial          = None
    pipeline.trainer_config  = None

    for emit_trial_docs in (True, False):
        pipeline._emit_docs = emit_trial_docs

        pipeline._make_trainer(None, None, None, None, Path("run"), None)
        pipeline._make_trainer(None, None, None, None, Path("run"), None, emit_docs=False)

    assert captured == [True, False, False, False]


def test_trial_callback_no_prune_without_pruner():
    """Verifies that no pruning occurs when the study uses a no-op pruner."""
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    trial = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    trainer._trial_callback(1.0, 0)
    trainer._trial_callback(2.0, 1)


def test_trial_callback_prunes_on_bad_intermediate():
    """Verifies that an intermediate value above the threshold raises TrialPruned."""
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._trial_callback(5.0, 0)


def test_trial_callback_keeps_trial_under_threshold():
    """Verifies that an intermediate value below the threshold leaves the trial running."""
    pruner = optuna.pruners.ThresholdPruner(upper=10.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialTrainer.__new__(TrialTrainer)
    trainer._trial = trial

    trainer._trial_callback(0.5, 0)


def test_ae_trainer_after_eval_reports_and_prunes():
    """Verifies that the profile AE trainer reports the value and prunes above threshold."""
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialProfileAeTrainer.__new__(TrialProfileAeTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._after_eval(9.0, 0)

    assert trial.storage.get_trial(trial._trial_id).intermediate_values[0] == 9.0


def test_ae_trainer_after_eval_survives_good_value():
    """Verifies that the profile AE trainer survives a value below the threshold."""
    pruner = optuna.pruners.ThresholdPruner(upper=10.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialProfileAeTrainer.__new__(TrialProfileAeTrainer)
    trainer._trial = trial

    trainer._after_eval(0.3, 0)


def test_jepa_trainer_after_eval_reports_and_prunes():
    """Verifies that the JEPA trainer prunes on an intermediate value above threshold."""
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialJepaTrainer.__new__(TrialJepaTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._after_eval(8.0, 0)


def test_image_ae_trainer_after_eval_reports_and_prunes():
    """Verifies that the image AE trainer reports the value and prunes above threshold."""
    pruner = optuna.pruners.ThresholdPruner(upper=1.0)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    trial  = study.ask()

    trainer = TrialImageAeTrainer.__new__(TrialImageAeTrainer)
    trainer._trial = trial

    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer._after_eval(7.0, 0)

    assert trial.storage.get_trial(trial._trial_id).intermediate_values[0] == 7.0


class FakeEntryTraining:
    """Training section stub carrying the epoch budget and patience fields."""
    def __init__(self):
        """Initialises the epoch and patience fields to zero."""
        self.epochs              = 0
        self.scheduler_epochs    = 0
        self.early_stop_patience = 0


class FakeAeEntry:
    """Autoencoder entry config stub carrying only a training section."""
    def __init__(self):
        """Builds the nested training section."""
        self.training = FakeEntryTraining()


class FakeAeConfig:
    """Autoencoder model config stub exposing a small tunable search space."""
    encoder_lr = 3e-4
    decoder_lr = 3e-4

    def __init__(self):
        """Initialises the learning rates, weight decays and embedding dimension."""
        self.encoder_lr    = 3e-4
        self.decoder_lr    = 3e-4
        self.encoder_wd    = 1e-4
        self.decoder_wd    = 1e-4
        self.embedding_dim = 24

    @classmethod
    def tunable_lr_params(cls):
        """Returns the tunable learning-rate search space."""
        return {
            "encoder_lr"    : {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
            "embedding_dim" : {"type": "categorical", "choices": [16, 24, 32]},
        }

    @classmethod
    def tunable_arch_params(cls):
        """Returns the tunable architecture search space."""
        return {"depth": {"type": "categorical", "choices": [3, 4, 6]}}


def test_ae_tuner_objective_materializes_entry(fake_logger, tune_cfg, tmp_path):
    """Verifies that the AE tuner fills run name, seed, log dir and model overrides."""
    captured = {}

    class CaptureAePipeline:
        """Trial pipeline stub capturing the entry config and trial it received."""
        def __init__(self, entry, trial):
            """Records the entry config and trial."""
            captured["entry"] = entry
            captured["trial"] = trial

        def run(self):
            """Returns a fixed validation loss of 0.5."""
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
    """Verifies that the objective returns the validation loss reported by the pipeline."""
    class FakeAePipeline:
        """Trial pipeline stub returning a fixed validation loss."""
        def __init__(self, entry, trial):
            """Stores the entry config."""
            self.entry = entry

        def run(self):
            """Returns a fixed validation loss of 0.7."""
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


def test_ae_tuner_names_the_error_of_a_failed_trial(fake_logger, tune_cfg, tmp_path):
    """A divergence or device error in an autoencoder trial is logged with its type and message and kept on the trial, as the backbone tuner does."""
    class DivergingAePipeline:
        """Trial pipeline stub whose run diverges."""
        def __init__(self, entry, trial):
            """Ignores the entry config."""

        def run(self):
            """Raises the trainer's divergence error."""
            raise FloatingPointError("reconstruction loss is non-finite at step 3")

    tuner = AeTuner(
        model_name         = "mlp_ae",
        config_cls         = FakeAeConfig,
        entry_template     = FakeAeEntry(),
        trial_pipeline_cls = DivergingAePipeline,
        tune_cfg           = tune_cfg,
        log_dir            = str(tmp_path),
        logger             = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1, catch=(FloatingPointError,))

    assert study.trials[0].state                     == optuna.trial.TrialState.FAIL
    assert study.trials[0].user_attrs["fail_reason"] == "FloatingPointError: reconstruction loss is non-finite at step 3"
    assert fake_logger.errors == ["Trial 0 failed and is recorded as such: FloatingPointError: reconstruction loss is non-finite at step 3"]


def test_ae_tuner_image_entry_gets_model_overrides(fake_logger, tune_cfg, tmp_path):
    """Verifies that the image AE entry receives the sampled model overrides."""
    captured = {}

    class CaptureImageAePipeline:
        """Trial pipeline stub capturing the entry config it received."""
        def __init__(self, entry, trial):
            """Records the entry config."""
            captured["entry"] = entry

        def run(self):
            """Returns a fixed validation loss of 0.3."""
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
    """JEPA model config stub exposing an architecture-only search space."""
    @classmethod
    def tunable_arch_params(cls):
        """Returns the tunable architecture search space."""
        return {
            "depth"      : {"type": "categorical", "choices": [2, 4, 8]},
            "hidden_dim" : {"type": "categorical", "choices": [64, 128]},
        }


class FakeJepaTraining:
    """Training section stub carrying the epoch budget and patience fields."""
    def __init__(self):
        """Initialises the epoch and patience fields to zero."""
        self.epochs              = 0
        self.scheduler_epochs    = 0
        self.early_stop_patience = 0


class FakeJepaEntry:
    """JEPA entry config stub with the fields the run-name builder reads."""
    def __init__(self, test_data_dir: Path, ae_logdir: Path | None = None):
        """Wires dataset paths, loss configs and optional autoencoder references."""
        from configuration.dataset               import AugmentationConfig
        from configuration.training              import EmbeddingLossConfig, LossConfig
        from configuration.training.general.run  import TrainingPathsConfig
        from configuration.training.jepa         import JepaMaskingConfig

        self.training     = FakeJepaTraining()
        self.param_loss   = LossConfig(use_param_l1=True)
        self.paths        = TrainingPathsConfig(dataset_path=test_data_dir, parameters_path=test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy")
        self.augmentation = AugmentationConfig()

        self.profile_autoencoder_logdir = ae_logdir
        self.profile_autoencoder_run    = "gru_run" if ae_logdir else None
        self.profile_autoencoder_mode   = "frozen"
        self.target_provider            = "stopgrad"
        self.image_autoencoder_logdir   = ae_logdir
        self.image_autoencoder_run      = None
        self.image_autoencoder_mode     = "frozen"
        self.embedding_loss             = EmbeddingLossConfig()
        self.masking                    = JepaMaskingConfig()


@pytest.mark.dataset
def test_jepa_tuner_objective_sets_model_overrides(fake_logger, tune_cfg, tmp_path, monkeypatch, test_data_dir):
    """Verifies that the JEPA tuner sets the backbone, run name and model overrides."""
    captured = {}

    class CaptureJepaPipeline:
        """Trial pipeline stub capturing the entry config it received."""
        def __init__(self, entry, trial):
            """Records the entry config."""
            captured["entry"] = entry

        def run(self):
            """Returns a fixed validation loss of 0.4."""
            return (None, None, 0.4), None

    monkeypatch.setattr(trial_mod, "TrialJepaPipeline", CaptureJepaPipeline)

    from configuration.architectures                import GruAutoencoderConfig
    from pipelines.shared.config.config_persistence import ProfileAutoencoderConfigIO

    ProfileAutoencoderConfigIO.save(GruAutoencoderConfig(), "gru_ae", tmp_path / "gru_run" / "meta")

    tuner = JepaTuner(
        model_name       = "vit",
        model_config_cls = FakeJepaConfig,
        entry_template   = FakeJepaEntry(test_data_dir, ae_logdir=tmp_path),
        tune_cfg         = tune_cfg,
        log_dir          = str(tmp_path),
        logger           = fake_logger,
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(tuner._objective, n_trials=1)

    entry = captured["entry"]
    assert entry.backbone_name == "vit"
    assert entry.run_name      == f"vit-conv-stopgrad-{SCENE_K_TAG}-hv-none-pae_gru.frozen-emb_mse_1-curve_mse_1_trial_0000"
    assert set(entry.model_overrides.keys()) == {"depth", "hidden_dim"}
    assert entry.model_overrides["depth"]      in [2, 4, 8]
    assert entry.model_overrides["hidden_dim"] in [64, 128]


@pytest.mark.dataset
def test_jepa_tuner_space_uses_arch_params_only(fake_logger, tune_cfg, tmp_path, test_data_dir):
    """Verifies that the JEPA search space holds architecture params only."""
    from configuration.architectures                import GruAutoencoderConfig
    from pipelines.shared.config.config_persistence import ProfileAutoencoderConfigIO

    ProfileAutoencoderConfigIO.save(GruAutoencoderConfig(), "gru_ae", tmp_path / "gru_run" / "meta")

    tuner = JepaTuner(
        model_name       = "vit",
        model_config_cls = FakeJepaConfig,
        entry_template   = FakeJepaEntry(test_data_dir, ae_logdir=tmp_path),
        tune_cfg         = tune_cfg,
        log_dir          = str(tmp_path),
        logger           = fake_logger,
    )

    assert tuner.space   == FakeJepaConfig.tunable_arch_params()
    assert tuner.run_tag == f"vit-conv-stopgrad-{SCENE_K_TAG}-hv-none-pae_gru.frozen-emb_mse_1-curve_mse_1"


@pytest.mark.dataset
def test_jepa_tuner_rejects_an_entry_it_cannot_name(fake_logger, tune_cfg, tmp_path, test_data_dir):
    """Verifies that an entry the run-name builder cannot describe raises ValueError."""
    with pytest.raises(ValueError, match="JEPA run naming requires"):
        JepaTuner(
            model_name       = "vit",
            model_config_cls = FakeJepaConfig,
            entry_template   = FakeJepaEntry(test_data_dir),
            tune_cfg         = tune_cfg,
            log_dir          = str(tmp_path),
            logger           = fake_logger,
        )
