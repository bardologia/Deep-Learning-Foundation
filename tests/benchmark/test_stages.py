from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pipelines.benchmark.stages import (
    ComparisonStage,
    InferenceStage,
    MaxBatchStage,
    SeedExpandedStage,
    SizeMatchStage,
    TrainingStage,
)
from configuration.benchmark    import BenchmarkConfig
from tools.data.io              import FileIO
from tools.orchestration        import QueuedInferenceStage, QueuedTrainingStage


@pytest.fixture
def config(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    config.resume             = False
    return config


@pytest.fixture
def data_config(config, test_data_dir):
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy"
    return config


ENTRY = Path("/entry/run_benchmark.py")


def test_maxbatch_job_command_uses_maxbatch_worker(config, logger_stub):
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    job = stage._job("unet")

    assert job.command[job.command.index("--worker") + 1] == "maxbatch"


def test_maxbatch_missing_result_carries_budget_and_ceiling(config, logger_stub):
    stage  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    record = stage._load_result("unet")

    assert record["status"]    == "FAIL"
    assert record["budget_gb"] == config.max_batch.vram_budget_gb
    assert record["ceiling"]   == config.max_batch.max_batch


def test_maxbatch_pass_result_is_reused_on_resume(config, logger_stub):
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    FileIO.save_json({"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 10.0}, stage._result_path("unet"))

    assert stage._has_result("unet") is True


def test_maxbatch_failed_result_is_reprobed_on_resume(config, logger_stub):
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    FileIO.save_json({"model": "unet", "status": "FAIL", "batch_size": None, "peak_gb": None, "error": "boom"}, stage._result_path("unet"))

    assert stage._has_result("unet") is False


def test_maxbatch_run_collects_records_keyed_by_model(config, logger_stub, monkeypatch):
    models = ["unet", "resunet"]
    stage  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=models, logger=logger_stub)

    def fake_queue(jobs):
        for job in jobs:
            path = stage._result_path(job.name)
            FileIO.save_json({"model": job.name, "status": "PASS", "batch_size": 64, "peak_gb": 10.0}, path)
        return []

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    records = stage.run()

    assert set(records) == set(models)
    assert records["unet"]["batch_size"] == 64
    assert stage.records_path.exists()


def test_sizematch_reference_record_is_zero_deviation(config, logger_stub):
    stage  = SizeMatchStage(config=config, run_tag="t", models=["unet"], logger=logger_stub)
    record = stage._reference_record("unet", 1234)

    assert record["model"]         == "unet"
    assert record["scale"]         == 1.0
    assert record["parameters"]    == 1234
    assert record["target"]        == 1234
    assert record["deviation_pct"] == 0.0
    assert record["iterations"]    == 0


@pytest.mark.real_data
@pytest.mark.slow
def test_sizematch_run_emits_reference_and_matched(monkeypatch, logger_stub, test_data_dir, tmp_path):
    config                       = BenchmarkConfig()
    config.paths.log_base_dir    = tmp_path
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy"
    config.resume                = False
    config.skip_models           = []

    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    records = stage.run()

    assert "unet" in records
    assert "resunet" in records
    assert records["unet"]["scale"] == 1.0
    assert stage.records_path.exists()


def test_sizematch_rejects_a_reference_that_is_not_swept(config, logger_stub):
    config.size_match.reference_model = "unet"
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet-set_pred", "resunet-set_pred"], logger=logger_stub)

    with pytest.raises(SystemExit, match="reference_model"):
        stage.run()


def test_sizematch_load_cached_requires_all_models(config, logger_stub):
    config.resume = True
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    FileIO.save_json({"unet": {}}, stage.records_path)

    assert stage._load_cached() is None


def test_sizematch_load_cached_returns_when_complete(config, logger_stub):
    config.resume = True
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    FileIO.save_json({"unet": {}, "resunet": {}}, stage.records_path)

    cached = stage._load_cached()
    assert set(cached) == {"unet", "resunet"}


@pytest.mark.real_data
def test_training_and_inference_stages_subclass_queued_bases(data_config, logger_stub):
    train = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    infer = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert isinstance(train, QueuedTrainingStage)
    assert isinstance(infer, QueuedInferenceStage)
    assert train.worker_action == "train"
    assert infer.worker_action == "infer"


def test_comparison_stage_run_invokes_collector_and_report(config, logger_stub, monkeypatch):
    stage = ComparisonStage(config=config, run_tag="t", logger=logger_stub, reference_model=config.size_match.reference_model, embed_images=config.comparison.embed_images)

    collected = {}

    class FakeCollector:
        def __init__(self, run_dir, logger):
            collected["run_dir"] = run_dir
            self.seed_dispersion = {"unet": {"n_seeds": 2}}
        def collect(self):
            return []

    class FakeReport:
        def __init__(self, **kwargs):
            collected["kwargs"] = kwargs
        def write_all(self):
            return [Path("/tmp/x.md")]

    monkeypatch.setattr("pipelines.benchmark.stages.BenchmarkSeedCollector", FakeCollector)
    monkeypatch.setattr("pipelines.benchmark.stages.ComparisonReport", FakeReport)

    out_dir = stage.run()

    assert collected["run_dir"] == stage.run_dir
    assert collected["kwargs"]["reference_model"] == config.size_match.reference_model
    assert collected["kwargs"]["seed_dispersion"] == {"unet": {"n_seeds": 2}}
    assert "comparison" in str(out_dir)


def test_sweep_components_empty_list_fails_for_backbone(config):
    config.sweep_loss_components = []

    with pytest.raises(SystemExit, match="sweep_loss_components is empty"):
        SeedExpandedStage.components(config)


def test_sweep_components_unknown_name_fails(config):
    config.sweep_loss_components = ["nope"]

    with pytest.raises(SystemExit, match="unknown loss component"):
        SeedExpandedStage.components(config)


def test_sweep_components_collapse_for_non_backbone(config):
    config.training_type         = "profile_autoencoder"
    config.sweep_loss_components = []

    assert SeedExpandedStage.components(config) == [None]


@pytest.mark.real_data
def test_training_seed_sweep_job_uses_base_model_and_seed(data_config, logger_stub):
    data_config.seeds = [7]
    stage = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    unit = "unet-conv-hungarian-K_5-hv-A__param_l1"
    assert stage.items == [f"{unit}/seed7"]

    job = stage._job(f"{unit}/seed7")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "7"
    assert job.command[job.command.index("--worker") + 1] == "train"
    assert job.log_path == stage.stage_dir / f"{unit}/seed7" / "worker.log"


@pytest.mark.real_data
def test_inference_seed_sweep_matches_training_run_names(data_config, logger_stub):
    data_config.seeds = [3, 4]
    stage = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    unit = "unet-conv-hungarian-K_5-hv-A__param_l1"
    assert stage.items == [f"{unit}/seed3", f"{unit}/seed4"]

    job = stage._job(f"{unit}/seed4")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "4"
    assert job.command[job.command.index("--worker") + 1] == "infer"


@pytest.mark.real_data
def test_unit_base_names_model_head_stem_and_component(data_config):
    unit = SeedExpandedStage.unit_base(data_config, "resunet-set_pred", "mse_curve")

    assert unit == "resunet-set_pred-hungarian-K_5-hv-A__mse_curve"
