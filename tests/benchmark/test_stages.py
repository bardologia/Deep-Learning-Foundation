"""Tests for the benchmark pipeline stages.

Covers the max-batch and size-match stages including their resume caches, the
queued training and inference stages with seed and loss-component expansion, and
the comparison stage wiring.
"""

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
from configuration.benchmark            import BenchmarkConfig
from tools.data.io                      import FileIO
from tools.orchestration                import QueuedInferenceStage, QueuedTrainingStage
from configuration.training.general.run import DEFAULT_PARAMS_RUN
from configuration.synthesis            import SynthesisConfig


SCENE_K_TAG = f"K_{SynthesisConfig().k_max}"


@pytest.fixture
def config(tmp_path):
    """Returns a BenchmarkConfig logging into the temporary directory with resume off."""
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    config.resume             = False
    return config


@pytest.fixture
def data_config(config, test_data_dir):
    """Returns the benchmark config pointed at the test dataset and its parameters."""
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    return config


ENTRY = Path("/entry/run_benchmark.py")


def test_maxbatch_job_command_uses_maxbatch_worker(config, logger_stub):
    """Verifies the max-batch job invokes the entry script with the maxbatch worker."""
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    job = stage._job("unet")

    assert job.command[job.command.index("--worker") + 1] == "maxbatch"


def test_maxbatch_missing_result_carries_budget_and_ceiling(config, logger_stub):
    """Verifies a missing probe result is reported as FAIL carrying the VRAM budget and batch ceiling."""
    stage  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    record = stage._load_result("unet")

    assert record["status"]    == "FAIL"
    assert record["budget_gb"] == config.max_batch.vram_budget_gb
    assert record["ceiling"]   == config.max_batch.max_batch


def test_maxbatch_pass_result_is_reused_on_resume(config, logger_stub):
    """Verifies a stored passing probe result taken under the current settings is reused."""
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    FileIO.save_json({"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 10.0, "settings": config.max_batch_settings()}, stage._result_path("unet"))

    assert stage._has_result("unet") is True


def test_maxbatch_result_is_reprobed_when_settings_changed(config, logger_stub):
    """Verifies a stored result is discarded when the VRAM budget or patch size changed."""
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    stored = config.max_batch_settings()
    FileIO.save_json({"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 10.0, "settings": stored}, stage._result_path("unet"))

    config.max_batch.vram_budget_gb = stored["budget_gb"] + 4.0
    assert stage._has_result("unet") is False

    config.max_batch.vram_budget_gb = stored["budget_gb"]
    config.training.patch_size      = [stored["patch_size"][0] * 2, stored["patch_size"][1]]
    assert stage._has_result("unet") is False

    config.training.patch_size = stored["patch_size"]
    assert stage._has_result("unet") is True


def test_maxbatch_result_is_reprobed_when_find_lr_is_toggled(config, logger_stub):
    """Verifies enabling the LR range test invalidates a result probed without it."""
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    FileIO.save_json({"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 10.0, "settings": config.max_batch_settings()}, stage._result_path("unet"))

    config.find_lr.enabled = True
    assert stage._has_result("unet") is False

    config.find_lr.enabled = False
    assert stage._has_result("unet") is True


def test_maxbatch_result_is_reprobed_on_an_older_settings_schema(config, logger_stub):
    """Verifies a cached result whose settings lack the current keys is re-probed."""
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    stored = config.max_batch_settings()
    del stored["find_lr"]
    FileIO.save_json({"model": "unet", "status": "PASS", "batch_size": 64, "peak_gb": 10.0, "settings": stored}, stage._result_path("unet"))

    assert stage._has_result("unet") is False


def test_maxbatch_failed_result_is_reprobed_on_resume(config, logger_stub):
    """Verifies a stored failing probe result is not reused on resume."""
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    FileIO.save_json({"model": "unet", "status": "FAIL", "batch_size": None, "peak_gb": None, "error": "boom"}, stage._result_path("unet"))

    assert stage._has_result("unet") is False


def test_maxbatch_run_collects_records_keyed_by_model(config, logger_stub, monkeypatch):
    """Verifies the stage collects one probe record per model and writes the records file."""
    models = ["unet", "resunet"]
    stage  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=models, logger=logger_stub)

    def fake_queue(jobs):
        """Writes a passing probe result for every queued job and reports no queue records."""
        for job in jobs:
            path = stage._result_path(job.name)
            FileIO.save_json({"model": job.name, "status": "PASS", "batch_size": 64, "peak_gb": 10.0, "lr_multiplier": None}, path)
        return []

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    records = stage.run()

    assert set(records) == set(models)
    assert records["unet"]["batch_size"] == 64
    assert stage.records_path.exists()


def test_sizematch_reference_record_is_zero_deviation(config, logger_stub):
    """Verifies the reference model records unit scale, zero deviation and no iterations."""
    stage  = SizeMatchStage(config=config, run_tag="t", models=["unet"], logger=logger_stub)
    record = stage._reference_record("unet", 1234)

    assert record["model"]         == "unet"
    assert record["scale"]         == 1.0
    assert record["parameters"]    == 1234
    assert record["target"]        == 1234
    assert record["deviation_pct"] == 0.0
    assert record["iterations"]    == 0


@pytest.mark.dataset
@pytest.mark.slow
def test_sizematch_run_emits_reference_and_matched(monkeypatch, logger_stub, test_data_dir, tmp_path):
    """Verifies the stage emits a unit-scale reference record plus a matched record and writes them to disk."""
    config                       = BenchmarkConfig()
    config.paths.log_base_dir    = tmp_path
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    config.resume                = False
    config.skip_models           = []

    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    records = stage.run()

    assert "unet" in records
    assert "resunet" in records
    assert records["unet"]["scale"] == 1.0
    assert stage.records_path.exists()


def test_sizematch_rejects_a_reference_that_is_not_swept(config, logger_stub):
    """Verifies a reference model absent from the swept models exits."""
    config.size_match.reference_model = "unet"
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet-set_pred", "resunet-set_pred"], logger=logger_stub)

    with pytest.raises(SystemExit, match="reference_model"):
        stage.run()


def test_sizematch_load_cached_requires_all_models(config, logger_stub):
    """Verifies a cached records file missing a swept model is rejected."""
    config.resume = True
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    FileIO.save_json({"unet": {}}, stage.records_path)

    assert stage._load_cached() is None


def test_sizematch_load_cached_returns_when_complete(config, logger_stub):
    """Verifies a cached records file covering every swept model is reused."""
    config.resume = True
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    settings = config.size_match_settings()
    FileIO.save_json({"unet": {"settings": settings}, "resunet": {"settings": settings}}, stage.records_path)

    cached = stage._load_cached()
    assert set(cached) == {"unet", "resunet"}


def test_sizematch_load_cached_rejects_changed_matching_inputs(config, logger_stub):
    """Verifies the cache is dropped when input channels, reference model or patch size changed."""
    config.resume = True
    stage = SizeMatchStage(config=config, run_tag="t", models=["unet", "resunet"], logger=logger_stub)

    settings = config.size_match_settings()
    FileIO.save_json({"unet": {"settings": settings}, "resunet": {"settings": settings}}, stage.records_path)

    config.size_match.in_channels = settings["in_channels"] + 1
    assert stage._load_cached() is None

    config.size_match.in_channels     = settings["in_channels"]
    config.size_match.reference_model = "resunet"
    assert stage._load_cached() is None

    config.size_match.reference_model = settings["reference_model"]
    config.training.patch_size        = [settings["patch_size"][0] * 2, settings["patch_size"][1]]
    assert stage._load_cached() is None

    config.training.patch_size = settings["patch_size"]
    assert stage._load_cached() is not None


@pytest.mark.dataset
def test_training_and_inference_stages_subclass_queued_bases(data_config, logger_stub):
    """Verifies the training and inference stages derive from the shared queued bases and carry their worker actions."""
    train = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    infer = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert isinstance(train, QueuedTrainingStage)
    assert isinstance(infer, QueuedInferenceStage)
    assert train.worker_action == "train"
    assert infer.worker_action == "infer"


def test_comparison_stage_run_invokes_collector_and_report(config, logger_stub, monkeypatch):
    """Verifies the comparison stage hands the collector the run directory and split, and forwards the reference model and seed dispersion to the report."""
    stage = ComparisonStage(config=config, run_tag="t", logger=logger_stub, reference_model=config.size_match.reference_model, embed_images=config.comparison.embed_images)

    collected = {}

    class FakeCollector:
        """Stand-in seed collector that records its arguments and returns no records."""
        def __init__(self, run_dir, logger, inference_split):
            """Records the run directory and inference split and exposes a fixed dispersion."""
            collected["run_dir"]         = run_dir
            collected["inference_split"] = inference_split
            self.seed_dispersion = {"unet": {"n_seeds": 2}}
        def collect(self):
            """Returns an empty record list."""
            return []

    class FakeReport:
        """Stand-in comparison report that records its keyword arguments."""
        def __init__(self, **kwargs):
            """Records the keyword arguments the stage passes to the report."""
            collected["kwargs"] = kwargs
        def write_all(self):
            """Returns a single fake written path."""
            return [Path("/tmp/x.md")]

    monkeypatch.setattr("pipelines.benchmark.stages.BenchmarkSeedCollector", FakeCollector)
    monkeypatch.setattr("pipelines.benchmark.stages.ComparisonReport", FakeReport)

    out_dir = stage.run()

    assert collected["run_dir"]         == stage.run_dir
    assert collected["inference_split"] == config.inference.split
    assert collected["kwargs"]["reference_model"] == config.size_match.reference_model
    assert collected["kwargs"]["seed_dispersion"] == {"unet": {"n_seeds": 2}}
    assert "comparison" in str(out_dir)


def test_sweep_components_empty_list_fails_for_backbone(config):
    """Verifies an empty loss-component sweep exits for backbone training."""
    config.sweep_loss_components = []

    with pytest.raises(SystemExit, match="sweep_loss_components is empty"):
        SeedExpandedStage.components(config)


def test_sweep_components_unknown_name_fails(config):
    """Verifies an unregistered loss-component name exits."""
    config.sweep_loss_components = ["nope"]

    with pytest.raises(SystemExit, match="unknown loss component"):
        SeedExpandedStage.components(config)


def test_sweep_components_collapse_for_non_backbone(config):
    """Verifies non-backbone training collapses the component sweep to a single unnamed arm."""
    config.training_type         = "profile_autoencoder"
    config.sweep_loss_components = []

    assert SeedExpandedStage.components(config) == [None]


@pytest.mark.dataset
def test_training_seed_sweep_job_uses_base_model_and_seed(data_config, logger_stub):
    """Verifies a seeded training unit runs the base model with its seed and logs under the unit's seed directory."""
    data_config.seeds = [7]
    stage = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    unit = f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1"
    assert stage.items == [f"{unit}/seed7"]

    job = stage._job(f"{unit}/seed7")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "7"
    assert job.command[job.command.index("--worker") + 1] == "train"
    assert job.log_path == stage.stage_dir / f"{unit}/seed7" / "worker.log"


@pytest.mark.dataset
def test_inference_seed_sweep_matches_training_run_names(data_config, logger_stub):
    """Verifies inference units carry the same seeded run names as training and invoke the infer worker."""
    data_config.seeds = [3, 4]
    stage = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    unit = f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1"
    assert stage.items == [f"{unit}/seed3", f"{unit}/seed4"]

    job = stage._job(f"{unit}/seed4")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "4"
    assert job.command[job.command.index("--worker") + 1] == "infer"


@pytest.mark.dataset
def test_unit_base_names_model_head_stem_and_component(data_config):
    """Verifies the unit base name joins model, head, matching, Gaussian count, augmentation and presence stem with the loss component."""
    unit = SeedExpandedStage.unit_base(data_config, "resunet-set_pred", "mse_curve")

    assert unit == f"resunet-set_pred-hungarian-{SCENE_K_TAG}-hv-AB__mse_curve"
