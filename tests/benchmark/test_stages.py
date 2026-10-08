"""Tests for the benchmark pipeline stages.

Covers the max-batch and size-match stages including their resume caches, the
queued training and inference stages with seed and loss-component expansion, and
the comparison stage wiring.
"""

from __future__ import annotations

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
from configuration.benchmark                      import BenchmarkConfig
from pipelines.shared.comparison.trial_collection import TrialRecord
from tools.data.io                                import FileIO
from tools.orchestration                          import QueuedInferenceStage, QueuedTrainingStage
from configuration.training.general.run           import DEFAULT_PARAMS_RUN
from configuration.synthesis                      import SynthesisConfig


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
    record = stage._load_result("unet", None)

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
            FileIO.save_json({"model": job.name, "status": "PASS", "batch_size": 64, "peak_gb": 10.0, "lr_multiplier": None, "settings": config.max_batch_settings()}, path)
        return []

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    records = stage.run()

    assert set(records) == set(models)
    assert records["unet"]["batch_size"] == 64
    assert stage.records_path.exists()


def _probe_record(config, status: str = "PASS", **fields) -> dict:
    """Returns a probe result record under the config's current settings, with the given fields overriding the defaults."""
    record = {"model": "unet", "status": status, "batch_size": 64, "peak_gb": 10.0, "lr_multiplier": None, "error": None, "settings": config.max_batch_settings()}
    record.update(fields)
    return record


def _queue_result(name: str, status: str, returncode: int, log_file: str) -> dict:
    """Returns the dictionary the GPU queue reports for one finished probe worker."""
    return {"name": name, "gpu": 0, "status": status, "returncode": returncode, "duration_s": 1.0, "log_file": log_file}


def test_maxbatch_discards_a_stale_result_before_reprobing_and_warns_once(config, recording_logger, monkeypatch):
    """A passing result measured under other settings is deleted before the worker runs, and the log explains the re-probe in exactly one line."""
    config.resume = True
    stage  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)
    stored = dict(config.max_batch_settings(), budget_gb=config.max_batch_settings()["budget_gb"] + 4.0)

    FileIO.save_json(_probe_record(config, settings=stored), stage._result_path("unet"))

    seen = {}

    def fake_queue(jobs):
        """Records whether the stale file was gone when the worker started and writes a fresh result."""
        seen["stale_present"] = stage._result_path("unet").exists()
        FileIO.save_json(_probe_record(config, batch_size=32), stage._result_path("unet"))
        return [_queue_result("unet", "DONE", 0, "/logs/unet.log")]

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    records  = stage.run()
    warnings = [message for level, message in recording_logger.messages if level == "warning" and message.startswith("unet:")]

    assert seen["stale_present"] is False
    assert records["unet"]["batch_size"] == 32
    assert len(warnings) == 1
    assert "measured under" in warnings[0]
    assert "not PASS" not in warnings[0]


def test_maxbatch_dead_worker_yields_a_fail_record_naming_its_exit_and_log(config, recording_logger, monkeypatch):
    """A worker that exits without writing a result leaves a FAIL record carrying its exit code and log path instead of whatever file was on disk."""
    config.resume = True
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    FileIO.save_json(_probe_record(config, settings=dict(config.max_batch_settings(), seed=99)), stage._result_path("unet"))

    monkeypatch.setattr(stage, "_run_queue", lambda jobs: [_queue_result("unet", "FAILED", 137, "/logs/unet/worker.log")])

    records = stage.run()

    assert records["unet"]["status"] == "FAIL"
    assert "worker exited 137" in records["unet"]["error"]
    assert "/logs/unet/worker.log" in records["unet"]["error"]
    assert FileIO.load_json(stage.records_path)["unet"]["status"] == "FAIL"


def test_maxbatch_load_result_rejects_a_result_measured_under_other_settings(config, recording_logger):
    """A result file whose settings differ from the current ones is turned into FAIL with an error line, never used as PASS."""
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    FileIO.save_json(_probe_record(config, settings=dict(config.max_batch_settings(), seed=99)), stage._result_path("unet"))

    record = stage._load_result("unet", None)

    assert record["status"] == "FAIL"
    assert "stale probe result" in record["error"]
    assert any(level == "error" and "measured under" in message for level, message in recording_logger.messages)


def test_maxbatch_resume_off_discards_every_previous_result(config, recording_logger, monkeypatch):
    """With resume off a previous result is deleted before probing and the log says so, so a dead worker cannot resurrect it."""
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    FileIO.save_json(_probe_record(config), stage._result_path("unet"))

    monkeypatch.setattr(stage, "_run_queue", lambda jobs: [_queue_result("unet", "FAILED", 1, "/logs/unet/worker.log")])

    records = stage.run()

    assert records["unet"]["status"] == "FAIL"
    assert ("warning", "unet: previous probe result discarded (resume is off), re-probing") in recording_logger.messages


def test_maxbatch_fresh_model_logs_no_discard_line(config, recording_logger, monkeypatch):
    """A model without any previous result is probed without a discard warning."""
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    def fake_queue(jobs):
        """Writes a passing result for the queued model."""
        FileIO.save_json(_probe_record(config), stage._result_path("unet"))
        return [_queue_result("unet", "DONE", 0, "/logs/unet.log")]

    monkeypatch.setattr(stage, "_run_queue", fake_queue)

    stage.run()

    assert not any(level == "warning" and "discarded" in message for level, message in recording_logger.messages)


def test_maxbatch_summary_warns_when_the_lr_range_test_selected_nothing(config, recording_logger):
    """With the LR range test on, a passing model without a selected multiplier is warned about, since it trains at the configured learning rate; one with a multiplier is not."""
    config.find_lr.enabled = True
    stage                  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet", "resunet"], logger=recording_logger)

    stage._log_summary({"unet": _probe_record(config), "resunet": _probe_record(config, model="resunet", lr_multiplier=0.5)})

    warnings = [message for level, message in recording_logger.messages if level == "warning"]

    assert warnings == [f"unet: LR range test found no usable response; training keeps lr_multiplier={config.training.lr_multiplier}"]


def test_maxbatch_summary_stays_quiet_about_the_lr_when_the_test_is_off(config, recording_logger):
    """With the LR range test off, a passing model without a multiplier is the expected case and draws no warning."""
    config.find_lr.enabled = False
    stage                  = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    stage._log_summary({"unet": _probe_record(config)})

    assert [message for level, message in recording_logger.messages if level == "warning"] == []


def test_maxbatch_summary_names_the_failure_reason_and_the_worker_log(config, recording_logger):
    """A failed model is listed with the last line of its error and the path of its probe worker log."""
    stage = MaxBatchStage(config=config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    stage._log_summary({"unet": _probe_record(config, status="FAIL", batch_size=None, peak_gb=None, error="Traceback (most recent call last):\n  ...\nRuntimeError: dataset split is empty\n")})

    assert ("error", f"FAILED  unet: RuntimeError: dataset split is empty  (see {stage.stage_dir / 'unet' / 'worker.log'})") in recording_logger.messages


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


@pytest.mark.dataset
def test_fresh_run_stages_report_no_unidentified_directory(data_config, logger_stub):
    """Verifies a run without resume still exposes an empty unidentified list, which the prepare stage's unit status reads."""
    data_config.resume = False

    train = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    infer = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert train.unidentified == []
    assert infer.unidentified == []


def test_comparison_stage_run_invokes_collector_and_report(config, logger_stub, monkeypatch):
    """Verifies the comparison stage hands the collector the run directory and split, and forwards the reference model and seed dispersion to the report."""
    stage = ComparisonStage(config=config, run_tag="t", logger=logger_stub, reference_model=config.size_match.reference_model, embed_images=config.comparison.embed_images, inference_split=config.inference.split)

    collected = {}

    class FakeCollector:
        """Stand-in seed collector that records its arguments and returns one record."""
        def __init__(self, run_dir, logger, inference_split):
            """Records the run directory and inference split and exposes a fixed dispersion."""
            collected["run_dir"]         = run_dir
            collected["inference_split"] = inference_split
            self.runs            = [TrialRecord(name="unet", run_dir=run_dir / "training" / "unet")]
            self.seed_dispersion = {"unet": {"n_seeds": 2}}
        def collect(self):
            """Returns the single record."""
            return list(self.runs)

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


@pytest.mark.dataset
def test_training_stage_reuses_a_unit_found_through_its_manifest(data_config, logger_stub):
    """Verifies a completed benchmark unit under another directory name is matched by its manifest and its worker pointed at that directory."""
    from pipelines.benchmark.stages import BenchmarkUnitManifest
    from tools.runtime.completion   import CompletionMarker

    data_config.resume = True
    data_config.seeds  = [7]

    old = data_config.paths.log_base_dir / "t" / "training" / "renamed_unit" / "seed7"
    BenchmarkUnitManifest.write(old, run_tag="t", unit="renamed_unit/seed7", model="unet", loss_component="param_l1", seed=7)
    CompletionMarker.stamp(old, {})

    stage   = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    planned = f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1/seed7"

    assert stage.unit_dirs == {planned: "renamed_unit/seed7"}
    assert stage._job(planned).command[stage._job(planned).command.index("--run-name") + 1] == "renamed_unit/seed7"
    assert stage._is_complete(planned)


@pytest.mark.dataset
def test_training_stage_hands_a_fresh_unit_its_own_name(data_config, logger_stub):
    """Verifies a unit without a directory is handed its own name as the run name."""
    data_config.seeds = [7]
    stage = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    job = stage._job(f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1/seed7")

    assert job.command[job.command.index("--run-name") + 1] == f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1/seed7"


def test_comparison_stage_builds_from_the_standalone_entry_config(logger_stub):
    """The compare_runs entry config, which carries no GPU pool and no inference block, constructs the stage and hands it the split explicitly."""
    from configuration.comparison import ComparisonEntryConfig

    entry = ComparisonEntryConfig(run_tag="t")
    stage = ComparisonStage(config=entry, run_tag="t", logger=logger_stub, reference_model=entry.reference_model, embed_images=entry.embed_images, inference_split=entry.inference_split)

    assert stage.inference_split == "test"
    assert stage.run_dir == Path(entry.paths.log_base_dir) / "t"


@pytest.mark.dataset
def test_resumed_stages_warn_about_run_directories_without_a_manifest(data_config, recording_logger):
    """Both the training and the inference stage warn, while reconciling, about a run directory no planned unit's manifest claims."""
    data_config.resume = True
    data_config.seeds  = [7]

    stray = data_config.paths.log_base_dir / "t" / "training" / "old_run" / "seed7"
    (stray / "meta").mkdir(parents=True)

    TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)
    InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    warnings = [message for level, message in recording_logger.messages if level == "warning"]

    assert any(message.startswith("TrainingStage: 1 run directory(ies)") for message in warnings)
    assert any(message.startswith("InferenceStage: 1 run directory(ies)") for message in warnings)
    assert warnings.count("old_run/seed7: claimed by no planned unit, ignored by the stage; the comparison still collects it beside the planned units") == 2


@pytest.mark.dataset
def test_a_manifestless_directory_at_a_planned_units_path_is_reported_as_that_units_run(data_config, recording_logger):
    """A directory without a manifest sitting at a planned unit's own path is reported as the run the stage will use for that unit."""
    from tools.runtime.completion import CompletionMarker

    data_config.resume = True
    data_config.seeds  = [7]

    planned = f"unet-conv-hungarian-{SCENE_K_TAG}-hv-AB__param_l1/seed7"
    unit    = data_config.paths.log_base_dir / "t" / "training" / planned
    (unit / "meta").mkdir(parents=True)
    CompletionMarker.stamp(unit, {"stage": "training"})

    TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    assert ("warning", f"{planned}: sits at planned unit '{planned}'s own path and holds a completion marker, so the stage uses it as that unit's completed run") in recording_logger.messages


def _comparison_stage(config, logger, planned_units, monkeypatch, runs):
    """Returns a comparison stage whose collector yields the given per-run records and whose report writes nothing."""
    stage = ComparisonStage(config=config, run_tag="t", logger=logger, reference_model="unet", embed_images=False, inference_split="test", planned_units=planned_units)

    class FakeCollector:
        """Collector stub exposing canned per-run records."""
        def __init__(self, run_dir, logger, inference_split):
            """Exposes the canned records and an empty dispersion."""
            self.runs            = runs
            self.seed_dispersion = {}
        def collect(self):
            """Returns the canned records."""
            return list(runs)

    class FakeReport:
        """Report stub writing nothing."""
        def __init__(self, **kwargs):
            """Accepts the report arguments."""
            pass
        def write_all(self):
            """Returns no written paths."""
            return []

    monkeypatch.setattr("pipelines.benchmark.stages.BenchmarkSeedCollector", FakeCollector)
    monkeypatch.setattr("pipelines.benchmark.stages.ComparisonReport", FakeReport)

    return stage


def _run_record(root, name, scored):
    """Returns a per-run record, with an inference directory attached when scored."""
    record = TrialRecord(name=name, run_dir=root / name)
    if scored:
        record.inference_dir = root / name / "inference" / "stamp"
    return record


def test_comparison_stage_names_planned_units_it_did_not_collect_and_runs_outside_the_plan(config, recording_logger, monkeypatch):
    """A planned unit without a run directory and a collected run no unit claims are each named in a warning."""
    root  = config.paths.log_base_dir / "t" / "training"
    runs  = [_run_record(root, "unet/seed0", True), _run_record(root, "stray/seed0", False)]
    stage = _comparison_stage(config, recording_logger, {"unet/seed0": "unet/seed0", "unet/seed1": "unet/seed1"}, monkeypatch, runs)

    stage.run()

    assert ("warning", "unet/seed1: planned unit not collected, no run at training/unet/seed1; absent from the reports") in recording_logger.messages
    assert ("warning", "stray/seed0: collected but planned by no unit; ranked in the reports beside the planned units") in recording_logger.messages


def test_comparison_stage_matches_planned_units_through_their_directories(config, recording_logger, monkeypatch):
    """A unit matched to a renamed directory on resume counts as collected under that directory."""
    root  = config.paths.log_base_dir / "t" / "training"
    runs  = [_run_record(root, "renamed/seed7", True)]
    stage = _comparison_stage(config, recording_logger, {"unet-conv__param_l1/seed7": "renamed/seed7"}, monkeypatch, runs)

    stage.run()

    assert [message for level, message in recording_logger.messages if level == "warning"] == []


def test_comparison_stage_refuses_an_empty_collection(config, recording_logger, monkeypatch):
    """Collecting nothing is an error instead of a set of empty reports."""
    stage = _comparison_stage(config, recording_logger, {"unet/seed0": "unet/seed0"}, monkeypatch, [])

    with pytest.raises(SystemExit, match="no run collected"):
        stage.run()


def test_comparison_stage_without_a_plan_compares_every_run(config, recording_logger, monkeypatch):
    """The standalone comparison, which knows no plan, compares whatever it collected without plan warnings."""
    root  = config.paths.log_base_dir / "t" / "training"
    stage = _comparison_stage(config, recording_logger, None, monkeypatch, [_run_record(root, "unet", True)])

    stage.run()

    assert [message for level, message in recording_logger.messages if level == "warning"] == []
