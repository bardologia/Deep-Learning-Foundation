
from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.benchmark.stages import (
    ComparisonStage,
    InferenceStage,
    SeedExpandedStage,
    TrainingStage,
)
from configuration.benchmark                      import BenchmarkConfig
from pipelines.shared.comparison.trial_collection import TrialRecord
from tools.orchestration                          import QueuedInferenceStage, QueuedTrainingStage




@pytest.fixture
def config(tmp_path):
    config                    = BenchmarkConfig()
    config.paths.log_base_dir = tmp_path
    config.resume             = False
    return config


@pytest.fixture
def data_config(config, test_data_dir):
    config.paths.dataset_path    = test_data_dir
    return config


ENTRY = Path("/entry/run_benchmark.py")


@pytest.mark.dataset
def test_training_and_inference_stages_subclass_queued_bases(data_config, logger_stub):
    train = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    infer = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert isinstance(train, QueuedTrainingStage)
    assert isinstance(infer, QueuedInferenceStage)
    assert train.worker_action == "train"
    assert infer.worker_action == "infer"


@pytest.mark.dataset
def test_fresh_run_stages_report_no_unidentified_directory(data_config, logger_stub):
    data_config.resume = False

    train = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    infer = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    assert train.unidentified == []
    assert infer.unidentified == []


def test_comparison_stage_run_invokes_collector_and_report(config, logger_stub, monkeypatch):
    stage = ComparisonStage(config=config, run_tag="t", logger=logger_stub, embed_images=config.comparison.embed_images, inference_split=config.inference.split)

    collected = {}

    class FakeCollector:
        def __init__(self, run_dir, logger, inference_split):
            collected["run_dir"]         = run_dir
            collected["inference_split"] = inference_split
            self.runs            = [TrialRecord(name="unet", run_dir=run_dir / "training" / "unet")]
            self.seed_dispersion = {"unet": {"n_seeds": 2}}
        def collect(self):
            return list(self.runs)

    class FakeReport:
        def __init__(self, **kwargs):
            collected["kwargs"] = kwargs
        def write_all(self):
            return [Path("/tmp/x.md")]

    monkeypatch.setattr("pipelines.benchmark.stages.BenchmarkSeedCollector", FakeCollector)
    monkeypatch.setattr("pipelines.benchmark.stages.ComparisonReport", FakeReport)

    out_dir = stage.run()

    assert collected["run_dir"]         == stage.run_dir
    assert collected["inference_split"] == config.inference.split
    assert collected["kwargs"]["seed_dispersion"] == {"unet": {"n_seeds": 2}}
    assert "comparison" in str(out_dir)


@pytest.mark.dataset
def test_training_seed_sweep_job_uses_base_model_and_seed(data_config, logger_stub):
    data_config.seeds = [7]
    stage = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    unit = "unet-mse-hv"
    assert stage.items == [f"{unit}/seed7"]

    job = stage._job(f"{unit}/seed7")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "7"
    assert job.command[job.command.index("--worker") + 1] == "train"
    assert job.log_path == stage.stage_dir / f"{unit}/seed7" / "worker.log"


@pytest.mark.dataset
def test_inference_seed_sweep_matches_training_run_names(data_config, logger_stub):
    data_config.seeds = [3, 4]
    stage = InferenceStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    unit = "unet-mse-hv"
    assert stage.items == [f"{unit}/seed3", f"{unit}/seed4"]

    job = stage._job(f"{unit}/seed4")
    assert job.command[job.command.index("--model") + 1] == "unet"
    assert job.command[job.command.index("--seed")  + 1] == "4"
    assert job.command[job.command.index("--worker") + 1] == "infer"


@pytest.mark.dataset
def test_unit_base_names_model_loss_kind_and_augmentation(data_config):
    unit = SeedExpandedStage.unit_base(data_config, "resunet", "l1")

    assert unit == "resunet-l1-hv"


@pytest.mark.dataset
def test_training_stage_reuses_a_unit_found_through_its_manifest(data_config, logger_stub):
    from pipelines.benchmark.stages import BenchmarkUnitManifest
    from tools.runtime.completion   import CompletionMarker

    data_config.resume = True
    data_config.seeds  = [7]

    old = data_config.paths.log_base_dir / "t" / "training" / "renamed_unit" / "seed7"
    BenchmarkUnitManifest.write(old, run_tag="t", unit="renamed_unit/seed7", model="unet", loss_component="mse", seed=7)
    CompletionMarker.stamp(old, {})

    stage   = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)
    planned = "unet-mse-hv/seed7"

    assert stage.unit_dirs == {planned: "renamed_unit/seed7"}
    assert stage._job(planned).command[stage._job(planned).command.index("--run-name") + 1] == "renamed_unit/seed7"
    assert stage._is_complete(planned)


@pytest.mark.dataset
def test_training_stage_hands_a_fresh_unit_its_own_name(data_config, logger_stub):
    data_config.seeds = [7]
    stage = TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=logger_stub)

    job = stage._job("unet-mse-hv/seed7")

    assert job.command[job.command.index("--run-name") + 1] == "unet-mse-hv/seed7"


def test_comparison_stage_builds_from_the_standalone_entry_config(logger_stub):
    from configuration.comparison import ComparisonEntryConfig

    entry = ComparisonEntryConfig(run_tag="t")
    stage = ComparisonStage(config=entry, run_tag="t", logger=logger_stub, embed_images=entry.embed_images, inference_split=entry.inference_split)

    assert stage.inference_split == "test"
    assert stage.run_dir == Path(entry.paths.log_base_dir) / "t"


@pytest.mark.dataset
def test_resumed_stages_warn_about_run_directories_without_a_manifest(data_config, recording_logger):
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
    from tools.runtime.completion import CompletionMarker

    data_config.resume = True
    data_config.seeds  = [7]

    planned = "unet-mse-hv/seed7"
    unit    = data_config.paths.log_base_dir / "t" / "training" / planned
    (unit / "meta").mkdir(parents=True)
    CompletionMarker.stamp(unit, {"stage": "training"})

    TrainingStage(config=data_config, entry_script=ENTRY, run_tag="t", models=["unet"], logger=recording_logger)

    assert ("warning", f"{planned}: sits at planned unit '{planned}'s own path and holds a completion marker, so the stage uses it as that unit's completed run") in recording_logger.messages


def _comparison_stage(config, logger, planned_units, monkeypatch, runs):
    stage = ComparisonStage(config=config, run_tag="t", logger=logger, embed_images=False, inference_split="test", planned_units=planned_units)

    class FakeCollector:
        def __init__(self, run_dir, logger, inference_split):
            self.runs            = runs
            self.seed_dispersion = {}
        def collect(self):
            return list(runs)

    class FakeReport:
        def __init__(self, **kwargs):
            pass
        def write_all(self):
            return []

    monkeypatch.setattr("pipelines.benchmark.stages.BenchmarkSeedCollector", FakeCollector)
    monkeypatch.setattr("pipelines.benchmark.stages.ComparisonReport", FakeReport)

    return stage


def _run_record(root, name, scored):
    record = TrialRecord(name=name, run_dir=root / name)
    if scored:
        record.inference_dir = root / name / "inference" / "stamp"
    return record


def test_comparison_stage_names_planned_units_it_did_not_collect_and_runs_outside_the_plan(config, recording_logger, monkeypatch):
    root  = config.paths.log_base_dir / "t" / "training"
    runs  = [_run_record(root, "unet/seed0", True), _run_record(root, "stray/seed0", False)]
    stage = _comparison_stage(config, recording_logger, {"unet/seed0": "unet/seed0", "unet/seed1": "unet/seed1"}, monkeypatch, runs)

    stage.run()

    assert ("warning", "unet/seed1: planned unit not collected, no run at training/unet/seed1; absent from the reports") in recording_logger.messages
    assert ("warning", "stray/seed0: collected but planned by no unit; ranked in the reports beside the planned units") in recording_logger.messages


def test_comparison_stage_matches_planned_units_through_their_directories(config, recording_logger, monkeypatch):
    root  = config.paths.log_base_dir / "t" / "training"
    runs  = [_run_record(root, "renamed/seed7", True)]
    stage = _comparison_stage(config, recording_logger, {"unet-mse-hv/seed7": "renamed/seed7"}, monkeypatch, runs)

    stage.run()

    assert [message for level, message in recording_logger.messages if level == "warning"] == []


def test_comparison_stage_without_a_plan_compares_every_run(config, recording_logger, monkeypatch):
    root  = config.paths.log_base_dir / "t" / "training"
    stage = _comparison_stage(config, recording_logger, None, monkeypatch, [_run_record(root, "unet", True)])

    stage.run()

    assert [message for level, message in recording_logger.messages if level == "warning"] == []
