from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.benchmark.workers import (
    BenchmarkWorker,
    MaxBatchWorker,
)
from configuration.benchmark     import BenchmarkConfig
from tools.data.io               import FileIO


@pytest.fixture
def config(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    return config


@pytest.mark.real_data
def test_run_name_encodes_model_head_matching_gaussians_aug_presence_component_and_seed(config, test_data_dir, params_dir):
    from pipelines.shared.training.run_naming import RunNaming

    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = params_dir / "parameters.npy"

    worker = BenchmarkWorker(config, "tag")
    tag    = RunNaming.benchmark_unit("unet", None, config.loss, 5, config.augmentation)

    assert worker._run_name("unet", None, None)       == tag
    assert worker._run_name("unet", None, 5)          == f"{tag}/seed5"
    assert worker._run_name("unet", "param_l1", None) == "unet-conv-hungarian-K_5-hv-A__param_l1"
    assert worker._run_name("unet", "param_l1", 5)    == "unet-conv-hungarian-K_5-hv-A__param_l1/seed5"
    assert tag.startswith("unet-conv-hungarian-K_5-hv-A-param_l1_1")


def test_size_overrides_empty_without_file(config):
    worker = BenchmarkWorker(config, "tag")

    assert worker._size_overrides("unet") == {}


def test_size_overrides_reads_overrides(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "size_match.json"

    FileIO.save_json({"unet": {"overrides": {"features": [32, 64]}}}, path)

    assert worker._size_overrides("unet") == {"features": [32, 64]}


def test_size_overrides_missing_model_exits(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "size_match.json"

    FileIO.save_json({"other_model": {"overrides": {}}}, path)

    with pytest.raises(SystemExit, match="missing an entry for 'unet'"):
        worker._size_overrides("unet")


def test_max_batch_size_none_without_file(config):
    worker = BenchmarkWorker(config, "tag")

    assert worker._max_batch_size("unet") is None


def test_max_batch_size_returns_measured_value(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "PASS", "batch_size": 128}}, path)

    assert worker._max_batch_size("unet") == 128


def test_max_batch_size_raises_when_model_missing(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"resunet": {"status": "PASS", "batch_size": 128}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_max_batch_size_raises_when_not_passed(config):
    worker = BenchmarkWorker(config, "tag")
    path   = worker.run_dir / "pipeline" / "max_batch.json"

    FileIO.save_json({"unet": {"status": "FAIL", "batch_size": None}}, path)

    with pytest.raises(SystemExit):
        worker._max_batch_size("unet")


def test_probe_config_is_disabled(config):
    worker = BenchmarkWorker(config, "tag")
    probe  = worker._probe_config()

    assert probe.enabled    is False
    assert probe.exit_after is True


def test_max_batch_worker_invokes_real_probe(config, monkeypatch):
    worker = MaxBatchWorker(config, "tag")

    calls = {}

    class FakeProbe:
        def __init__(self, config, model_name, overrides, work_dir):
            calls["model_name"] = model_name
            calls["overrides"]  = overrides
            calls["work_dir"]   = work_dir
        def run(self):
            return {"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 12.0}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    worker.run("unet")

    assert calls["model_name"] == "unet"
    assert calls["work_dir"]   == worker.run_dir / "max_batch" / "unet" / "probe"
    result_path = worker.run_dir / "max_batch" / "unet" / "max_batch_result.json"
    assert result_path.exists()
    assert FileIO.load_json(result_path)["batch_size"] == 64


def test_max_batch_worker_raises_on_probe_fail(config, monkeypatch):
    worker = MaxBatchWorker(config, "tag")

    class FakeProbe:
        def __init__(self, *args, **kwargs):
            pass
        def run(self):
            return {"model": "unet", "status": "FAIL", "batch_size": None, "peak_gb": None}

    monkeypatch.setattr("pipelines.benchmark.batch_probe.MaxBatchProbe", FakeProbe)

    with pytest.raises(SystemExit):
        worker.run("unet")


@pytest.mark.real_data
def test_training_worker_scales_the_lr_from_the_measured_batch(config, test_data_dir, params_dir, monkeypatch):
    from pipelines.benchmark.workers import TrainingWorker

    config.paths.dataset_path     = test_data_dir
    config.paths.parameters_path  = params_dir / "parameters.npy"
    config.training.train_azimuth = (1000, 1400)
    config.training.val_azimuth   = (1400, 1700)
    config.training.test_azimuth  = (1700, 2000)

    worker = TrainingWorker(config, "tag")
    FileIO.save_json({"resunet-conv": {"status": "PASS", "batch_size": 500}}, worker.run_dir / "pipeline" / "max_batch.json")

    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def run(self, probe_config=None):
            return None

    monkeypatch.setattr("pipelines.backbone.training.pipeline.TrainingPipeline", FakePipeline)

    worker.run("resunet-conv")

    assert captured["trainer_config"].optimizer.lr_scale == pytest.approx(500 / 256)
    assert captured["dataset_config"].batch_size         == 500


def test_training_entry_configs_carry_the_overfit_check(config):
    from pipelines.benchmark.workers import TrainingWorker

    config.overfit_check.enabled = True
    worker = TrainingWorker(config, "tag")

    ae_entry   = worker._ae_entry_config("mlp_ae", worker.run_dir / "training")
    jepa_entry = worker._jepa_entry_config("resunet", worker.run_dir / "training")

    assert ae_entry.overfit_check         is config.overfit_check
    assert jepa_entry.overfit_check       is config.overfit_check
    assert ae_entry.overfit_check.enabled is True


def test_training_worker_rejects_unsupported_training_type(config):
    from pipelines.benchmark.workers import TrainingWorker

    config.training_type = "unrolled"
    worker = TrainingWorker(config, "tag")

    with pytest.raises(ValueError, match="unrolled"):
        worker.run("gamma_net")
