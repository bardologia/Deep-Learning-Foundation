"""Tests for the benchmark worker processes.

Covers run naming, the size-match and max-batch artefacts a worker reads from the
run directory, the max-batch probe wrapper, and the entry configs a training
worker hands to the underlying pipelines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.benchmark.workers import (
    BenchmarkWorker,
    MaxBatchWorker,
)
from configuration.benchmark import BenchmarkConfig
from tools.data.io           import FileIO
from configuration.synthesis import SynthesisConfig


SCENE_K_TAG = f"K_{SynthesisConfig().k_max}"


@pytest.fixture
def config(tmp_path):
    """Returns a BenchmarkConfig logging into the temporary directory."""
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    return config


@pytest.mark.dataset
def test_run_name_encodes_model_head_matching_gaussians_aug_presence_component_and_seed(config, test_data_dir, params_dir):
    """Verifies the run name carries the model, head, matching, Gaussian count, augmentation and presence stem, plus the loss component and seed when given."""
    from pipelines.shared.training.run_naming import RunNaming

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"

    worker = BenchmarkWorker(config, "tag")
    tag    = RunNaming.benchmark_unit("unet", None, config.loss, SynthesisConfig().k_max, config.augmentation)

    assert worker._run_name("unet", None, None)       == tag
    assert worker._run_name("unet", None, 5)          == f"{tag}/seed5"
    assert worker._run_name("unet", "param_l1", None) == f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1"
    assert worker._run_name("unet", "param_l1", 5)    == f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1/seed5"
    assert tag.startswith(f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB-param_l1_1")


def test_size_overrides_empty_without_file(config):
    """Verifies no width overrides are applied when size_match.json is absent."""
    worker = BenchmarkWorker(config, "tag")

    assert worker._size_overrides("unet") == {}


def test_size_overrides_reads_overrides(config):
    """Verifies the stored width overrides for the model are returned."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "size_match.json"

    FileIO.save_json({"unet": {"overrides": {"features": [32, 64]}}}, path)

    assert worker._size_overrides("unet") == {"features": [32, 64]}


def test_backbone_training_rejects_a_stale_size_override(config):
    """Verifies a stored override naming a field the model config lost is rejected."""
    from pipelines.benchmark.workers import TrainingWorker

    worker = TrainingWorker(config, "tag")

    FileIO.save_json({"unet": {"overrides": {"renamed_features": [32, 64]}}}, worker.run_dir / "pipeline" / "size_match.json")

    with pytest.raises(AttributeError, match="Unknown model override 'renamed_features'"):
        worker._run_backbone("unet", run_name="unet", seed=None, loss_component=None)


def test_size_overrides_missing_model_exits(config):
    """Verifies a size-match file without an entry for the model exits."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "size_match.json"

    FileIO.save_json({"other_model": {"overrides": {}}}, path)

    with pytest.raises(SystemExit, match="missing an entry for 'unet'"):
        worker._size_overrides("unet")


def test_max_batch_size_none_without_file(config):
    """Verifies no batch size is reported when max_batch.json is absent."""
    worker = BenchmarkWorker(config, "tag")

    assert worker._max_batch_size("unet") is None


def test_max_batch_size_returns_measured_value(config):
    """Verifies the measured batch size of a passing probe is returned."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128}}, path)

    assert worker._max_batch_size("unet") == 128


def test_max_batch_size_raises_when_model_missing(config):
    """Verifies a max-batch file without an entry for the model exits."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"resunet": {"status": "PASS", "batch_size": 128}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_max_batch_size_raises_when_not_passed(config):
    """Verifies a failed max-batch probe result exits instead of being used."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "FAIL", "batch_size": None}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_lr_multiplier_none_without_file(config):
    """Verifies no LR multiplier is reported when max_batch.json is absent."""
    worker = BenchmarkWorker(config, "tag")

    assert worker._lr_multiplier("unet") is None


def test_lr_multiplier_returns_probed_value(config):
    """Verifies the multiplier the range test recorded for the model is returned."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128, "lr_multiplier": 0.25}}, path)

    assert worker._lr_multiplier("unet") == pytest.approx(0.25)


def test_lr_multiplier_none_when_the_test_selected_nothing(config):
    """Verifies a probe record without a stable multiplier yields None."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128, "lr_multiplier": None}}, path)

    assert worker._lr_multiplier("unet") is None


def test_lr_multiplier_raises_when_model_missing(config):
    """Verifies a max-batch file without an entry for the model exits."""
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"resunet": {"status": "PASS", "batch_size": 128, "lr_multiplier": 0.5}}, path)

    with pytest.raises(SystemExit):
        worker._lr_multiplier("unet")


def test_max_batch_worker_builds_the_entry_probe_for_jepa(config, monkeypatch):
    """Verifies a JEPA sweep dispatches to the entry probe wrapping the JEPA pipeline builder."""
    import pipelines.jepa.training.pipeline as jepa_pipeline_module
    from pipelines.benchmark.batch_probe import EntryMaxBatchProbe

    config.training_type = "jepa"

    class _FakePipeline:
        """JEPA pipeline stub exposing only the preflight trainer builder."""
        def __init__(self, entry) -> None:
            """Stores the entry config."""
            self.entry = entry

        def build_pretrain_trainer(self, work_dir, logger):
            """Never called in this test."""
            raise AssertionError("the dispatch test must not build the probe trainer")

    monkeypatch.setattr(jepa_pipeline_module, "TrainingPipeline", _FakePipeline)

    worker = MaxBatchWorker(config, "tag")
    probe  = worker._build_probe("resunet", worker.run_dir / "max_batch" / "resunet" / "probe")

    assert isinstance(probe, EntryMaxBatchProbe)
    assert probe.model_name == "resunet"


def test_max_batch_worker_builds_the_entry_probe_for_profile_autoencoder(config, monkeypatch):
    """Verifies a profile-AE sweep dispatches to the entry probe wrapping the AE pipeline builder."""
    import pipelines.profile_autoencoder.training.pipeline as pae_pipeline_module
    from pipelines.benchmark.batch_probe import EntryMaxBatchProbe

    config.training_type = "profile_autoencoder"

    class _FakePipeline:
        """Profile-AE pipeline stub exposing only the preflight trainer builder."""
        def __init__(self, entry) -> None:
            """Stores the entry config."""
            self.entry = entry

        def build_pretrain_trainer(self, work_dir, logger):
            """Never called in this test."""
            raise AssertionError("the dispatch test must not build the probe trainer")

    monkeypatch.setattr(pae_pipeline_module, "TrainingPipeline", _FakePipeline)

    worker = MaxBatchWorker(config, "tag")
    probe  = worker._build_probe("pae_mlp", worker.run_dir / "max_batch" / "pae_mlp" / "probe")

    assert isinstance(probe, EntryMaxBatchProbe)
    assert probe.model_name == "pae_mlp"


def test_jepa_training_applies_the_probed_batch_and_lr_multiplier(config, monkeypatch):
    """Verifies a JEPA unit trains under the probed batch size and folded LR multiplier."""
    from pipelines.benchmark.workers import TrainingWorker
    import pipelines.jepa.training.pipeline as jepa_pipeline_module

    config.training_type   = "jepa"
    config.find_lr.enabled = True

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet": {"status": "PASS", "batch_size": 128, "lr_multiplier": 0.5}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class _FakePipeline:
        """JEPA training pipeline stub capturing its entry config."""
        def __init__(self, entry) -> None:
            """Captures the entry config."""
            captured["entry"] = entry

        def run(self):
            """Returns nothing, standing in for a training run."""
            return None

    monkeypatch.setattr(jepa_pipeline_module, "TrainingPipeline", _FakePipeline)

    worker._run_jepa("resunet", "run", None)

    assert captured["entry"].training.batch_size    == 128
    assert captured["entry"].training.lr_multiplier == pytest.approx(0.5)


def test_profile_ae_training_applies_the_probed_batch_and_lr_multiplier(config, monkeypatch):
    """Verifies a profile-AE unit trains under the probed batch size and folded LR multiplier."""
    from pipelines.benchmark.workers import TrainingWorker
    import pipelines.profile_autoencoder.training.pipeline as pae_pipeline_module

    config.training_type   = "profile_autoencoder"
    config.find_lr.enabled = True

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"pae_mlp": {"status": "PASS", "batch_size": 4096, "lr_multiplier": 2.0}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class _FakePipeline:
        """Profile-AE training pipeline stub capturing its entry config."""
        def __init__(self, entry) -> None:
            """Captures the entry config."""
            captured["entry"] = entry

        def run(self):
            """Returns nothing, standing in for a training run."""
            return None

    monkeypatch.setattr(pae_pipeline_module, "TrainingPipeline", _FakePipeline)

    worker._run_profile_autoencoder("pae_mlp", "run", None)

    assert captured["entry"].training.batch_size    == 4096
    assert captured["entry"].training.lr_multiplier == pytest.approx(2.0)


def test_probe_config_is_disabled(config):
    """Verifies the worker's probe config is off by default and set to exit after probing."""
    probe = BenchmarkWorker(config, "tag").config.probe

    assert probe.enabled    is False
    assert probe.exit_after is True


def test_probe_config_is_the_configured_one(config):
    """Verifies an enabled probe config reaches the worker unchanged."""
    config.probe.enabled   = True
    config.probe.n_batches = 7

    probe = BenchmarkWorker(config, "tag").config.probe

    assert probe.enabled   is True
    assert probe.n_batches == 7


def test_max_batch_worker_hands_the_probe_its_model_and_work_dir(config, monkeypatch):
    """Verifies the probe receives the model name and its per-model work directory, and the result is written to disk."""
    worker = MaxBatchWorker(config, "tag")

    calls = {}

    class FakeProbe:
        """Stand-in probe that records its construction arguments and reports success."""
        def __init__(self, config, model_name, overrides, work_dir):
            """Records the model name, overrides and work directory handed to the probe."""
            calls["model_name"] = model_name
            calls["overrides"]  = overrides
            calls["work_dir"]   = work_dir
        def run(self):
            """Returns a passing max-batch result."""
            return {"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 12.0}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    worker.run("unet")

    assert calls["model_name"] == "unet"
    assert calls["work_dir"]   == worker.run_dir / "max_batch" / "unet" / "probe"
    result_path = worker.run_dir / "max_batch" / "unet" / "max_batch_result.json"
    assert result_path.exists()
    assert FileIO.load_json(result_path)["batch_size"] == 64


def test_max_batch_worker_raises_on_probe_fail(config, monkeypatch):
    """Verifies a failing probe makes the worker exit."""
    worker = MaxBatchWorker(config, "tag")

    class FakeProbe:
        """Stand-in probe that always reports failure."""
        def __init__(self, *args, **kwargs):
            """Accepts and discards the probe arguments."""
            pass
        def run(self):
            """Returns a failing max-batch result."""
            return {"model": "unet", "status": "FAIL", "batch_size": None, "peak_gb": None}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    with pytest.raises(SystemExit):
        worker.run("unet")


@pytest.mark.dataset
def test_training_worker_scales_the_lr_from_the_measured_batch(config, test_data_dir, params_dir, monkeypatch):
    """Verifies the trainer learning-rate scale and dataset batch size follow the measured max batch relative to the reference batch of 256."""
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"
    config.training.train_rows   = (1000, 1400)
    config.training.val_rows     = (1400, 1700)
    config.training.test_rows    = (1700, 2000)

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet-conv": {"status": "PASS", "batch_size": 500}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class FakePipeline:
        """Stand-in training pipeline that captures its constructor keyword arguments."""
        def __init__(self, **kwargs):
            """Captures the configs handed to the pipeline."""
            captured.update(kwargs)
        def run(self, probe_config=None):
            """Returns nothing, standing in for a training run."""
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / config.training.lr_reference_batch_size)
    assert captured["dataset_config"].batch_size         == 500


@pytest.mark.dataset
def test_training_worker_applies_the_probed_lr_multiplier(config, test_data_dir, params_dir, monkeypatch):
    """Verifies the probed multiplier folds into the trainer LR scale on top of the batch rule."""
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path     = test_data_dir
    config.paths.parameters_path  = params_dir / "parameters.npy"
    config.training.train_azimuth = (1000, 1400)
    config.training.val_azimuth   = (1400, 1700)
    config.training.test_azimuth  = (1700, 2000)
    config.find_lr.enabled        = True

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet-conv": {"status": "PASS", "batch_size": 500, "lr_multiplier": 0.25}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class FakePipeline:
        """Stand-in training pipeline that captures its constructor keyword arguments."""
        def __init__(self, **kwargs):
            """Captures the configs handed to the pipeline."""
            captured.update(kwargs)
        def run(self, probe_config=None):
            """Returns nothing, standing in for a training run."""
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / 256 * 0.25)


@pytest.mark.dataset
def test_training_worker_ignores_the_multiplier_when_the_test_is_disabled(config, test_data_dir, params_dir, monkeypatch):
    """Verifies a recorded multiplier is not applied while find_lr is off."""
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path     = test_data_dir
    config.paths.parameters_path  = params_dir / "parameters.npy"
    config.training.train_azimuth = (1000, 1400)
    config.training.val_azimuth   = (1400, 1700)
    config.training.test_azimuth  = (1700, 2000)

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet-conv": {"status": "PASS", "batch_size": 500, "lr_multiplier": 0.25}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class FakePipeline:
        """Stand-in training pipeline that captures its constructor keyword arguments."""
        def __init__(self, **kwargs):
            """Captures the configs handed to the pipeline."""
            captured.update(kwargs)
        def run(self, probe_config=None):
            """Returns nothing, standing in for a training run."""
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / 256)


def test_training_entry_configs_carry_the_overfit_check(config):
    """Verifies the autoencoder and JEPA entry configs share the worker's overfit-check config object."""
    from pipelines.benchmark.workers import TrainingWorker

    config.overfit_check.enabled = True
    worker = TrainingWorker(config, "tag")

    ae_entry   = worker._ae_entry_config("mlp_ae", worker.run_dir / "training")
    jepa_entry = worker._jepa_entry_config("resunet", worker.run_dir / "training")

    assert ae_entry.overfit_check         is config.overfit_check
    assert jepa_entry.overfit_check       is config.overfit_check
    assert ae_entry.overfit_check.enabled is True


def test_jepa_entry_config_carries_the_autoencoder_finetune_rates(config):
    """Verifies the JEPA entry config carries the profile and image autoencoder finetune learning rates and weight decays."""
    from pipelines.benchmark.workers import TrainingWorker

    config.jepa.profile_autoencoder_mode = "finetune"
    config.jepa.ae_finetune_lr           = 7e-5
    config.jepa.ae_finetune_wd           = 2e-3
    config.jepa.image_ae_finetune_lr     = 5e-5
    config.jepa.image_ae_finetune_wd     = 4e-3

    entry = TrainingWorker(config, "tag")._jepa_entry_config("resunet", config.paths.log_base_dir)

    assert entry.profile_autoencoder_mode == "finetune"
    assert entry.ae_finetune_lr           == 7e-5
    assert entry.ae_finetune_wd           == 2e-3
    assert entry.image_ae_finetune_lr     == 5e-5
    assert entry.image_ae_finetune_wd     == 4e-3


def test_jepa_entry_config_forwards_the_loss_scale_probe(config):
    """Verifies the sweep's loss-scale probe settings reach the JEPA entry config."""
    from pipelines.benchmark.workers import TrainingWorker

    config.probe.enabled = True

    entry = TrainingWorker(config, "tag")._jepa_entry_config("resunet", config.paths.log_base_dir)

    assert entry.probe is config.probe


def test_training_worker_rejects_unsupported_training_type(config):
    """Verifies an unrolled training type is rejected by the training worker."""
    from pipelines.benchmark.workers import TrainingWorker

    config.training_type = "nonexistent_type"
    worker = TrainingWorker(config, "tag")

    with pytest.raises(ValueError, match="nonexistent_type"):
        worker.run("gamma_net")
