from __future__ import annotations

from pathlib import Path

import pytest

import pipelines.cross_validation.stages as stages_module
from configuration.cross_validation     import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.folds   import FoldPlanner, FoldRunNaming
from pipelines.cross_validation.stages  import (
    CrossValidationReportStage,
    FoldInferenceStage,
    FoldTrainingStage,
)
from pipelines.cross_validation.workers import FoldTrainingWorker
from tools.monitoring.logger  import Logger
from tools.runtime.completion import CompletionMarker


def make_logger(tmp_path: Path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="stages_test")


def stage_config(tmp_path: Path, resume: bool = False) -> CrossValidationConfig:
    config                    = CrossValidationConfig()
    config.paths.log_base_dir = tmp_path
    config.folds              = FoldConfig(n_folds=5, azimuth_start=1000, azimuth_end=2000)
    config.resume             = resume
    config.inference_splits   = ["val", "test"]
    config.seeds              = []
    return config


def data_config(tmp_path: Path, test_data_dir: Path, resume: bool = False) -> CrossValidationConfig:
    config                       = stage_config(tmp_path, resume)
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy"
    return config


def fold_names(config: CrossValidationConfig) -> list[str]:
    return [run_name for _, _, run_name in FoldRunNaming(config).units()]


def make_planner(config: CrossValidationConfig) -> FoldPlanner:
    return FoldPlanner(config, range_start=500, range_end=1000)


def mark_complete(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "test"})


def queue_result(name: str, status: str = "DONE") -> dict:
    return {"name": name, "gpu": 0, "status": status, "returncode": 0, "duration_s": 1.0, "log_file": ""}


def install_fake_queue(stage, status: str = "DONE") -> dict:
    captured = {}

    def fake_queue(jobs):
        captured["jobs"] = jobs
        return [queue_result(job.name, status) for job in jobs]

    stage._run_queue = fake_queue
    return captured


@pytest.mark.real_data
def test_training_stage_builds_one_item_per_fold(tmp_path, test_data_dir):
    config = data_config(tmp_path, test_data_dir)
    stage  = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    assert stage.items == fold_names(config)
    assert [name.split("_fold_")[-1] for name in stage.items] == [str(index) for index in range(5)]


@pytest.mark.real_data
def test_training_stage_items_carry_the_training_tag_the_worker_trains_into(tmp_path, test_data_dir):
    config = data_config(tmp_path, test_data_dir)
    stage  = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))
    worker = FoldTrainingWorker(config=config, run_tag="rt")

    assert stage.items                                   == [worker.fold_run_name(index, None) for index in range(5)]
    assert stage.stage_dir / stage.items[3]              == worker.run_dir / "folds" / worker.fold_run_name(3, None)
    assert stage.items[3].startswith(f"{config.backbone_name}-{config.backbone_head}-")


@pytest.mark.real_data
def test_training_stage_job_carries_fold_index(tmp_path, test_data_dir):
    config = data_config(tmp_path, test_data_dir)
    stage  = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    job = stage._job(stage.items[3])

    assert job.command[job.command.index("--worker") + 1] == "train"
    assert job.command[job.command.index("--fold")   + 1] == "3"
    assert "--seed" not in job.command


@pytest.mark.real_data
def test_training_stage_seed_sweep_expands_fold_by_seed(tmp_path, test_data_dir):
    config       = data_config(tmp_path, test_data_dir)
    config.seeds = [1, 2]
    stage        = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))
    worker       = FoldTrainingWorker(config=config, run_tag="rt")

    assert stage.items == [worker.fold_run_name(index, seed) for index in range(5) for seed in (1, 2)]

    item = worker.fold_run_name(3, 2)
    job  = stage._job(item)
    assert job.command[job.command.index("--fold") + 1] == "3"
    assert job.command[job.command.index("--seed") + 1] == "2"
    assert job.log_path == stage.stage_dir / item / stage.worker_logname


@pytest.mark.real_data
def test_training_stage_subdir_and_results_path(tmp_path, test_data_dir):
    stage = FoldTrainingStage(config=data_config(tmp_path, test_data_dir), entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    assert stage.stage_dir.name            == "folds"
    assert stage.results_path.name         == "training_results.json"
    assert stage.results_path.parent.name  == "pipeline"


@pytest.mark.real_data
def test_training_stage_run_executes_all_folds(tmp_path, test_data_dir):
    config   = data_config(tmp_path, test_data_dir)
    stage    = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))
    captured = install_fake_queue(stage)

    results = stage.run()

    assert [job.name for job in captured["jobs"]] == fold_names(config)
    assert [r["name"] for r in results]           == fold_names(config)
    assert all(r["status"] == "DONE" for r in results)
    assert stage.results_path.exists()


@pytest.mark.real_data
def test_training_stage_job_command_carries_fold_flag(tmp_path, test_data_dir):
    stage    = FoldTrainingStage(config=data_config(tmp_path, test_data_dir), entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))
    captured = install_fake_queue(stage)

    stage.run()

    command = captured["jobs"][3].command
    assert "--fold" in command
    assert command[command.index("--fold") + 1] == "3"


@pytest.mark.real_data
def test_training_stage_reuses_completion_written_under_worker_naming(tmp_path, test_data_dir):
    config       = data_config(tmp_path, test_data_dir, resume=True)
    config.seeds = [1, 2]
    stage        = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))
    worker       = FoldTrainingWorker(config=config, run_tag="rt")

    trained = worker.fold_run_name(0, 1)
    mark_complete(worker.run_dir / "folds" / trained)

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert trained not in [job.name for job in captured["jobs"]]
    assert statuses[trained]                    == "DONE"
    assert statuses[worker.fold_run_name(0, 2)] == "DONE"
    assert (worker.run_dir / "folds" / trained).is_dir()


@pytest.mark.real_data
def test_inference_stage_one_job_per_fold_split_with_completed_training(tmp_path, test_data_dir):
    config  = data_config(tmp_path, test_data_dir)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))
    names   = fold_names(config)

    for name in names[:2]:
        mark_complete(stage.stage_dir / name)

    captured = install_fake_queue(stage)
    results  = stage.run()

    assert sorted(job.name for job in captured["jobs"]) == sorted(f"{name}:{split}" for name in names[:2] for split in ("val", "test"))

    statuses = {r["name"]: r["status"] for r in results}
    assert statuses[f"{names[0]}:val"]  == "DONE"
    assert statuses[f"{names[2]}:test"] == "SKIPPED"


@pytest.mark.real_data
def test_inference_stage_sees_completion_written_under_worker_naming(tmp_path, test_data_dir):
    config  = data_config(tmp_path, test_data_dir)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))
    worker  = FoldTrainingWorker(config=config, run_tag="rt")

    trained = worker.fold_run_name(0, None)
    mark_complete(worker.run_dir / "folds" / trained)

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert sorted(job.name for job in captured["jobs"]) == [f"{trained}:test", f"{trained}:val"]
    assert statuses[f"{trained}:val"]  == "DONE"
    assert statuses[f"{trained}:test"] == "DONE"


@pytest.mark.real_data
def test_inference_stage_skips_folds_without_completed_training(tmp_path, test_data_dir):
    config  = data_config(tmp_path, test_data_dir)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    interrupted_dir = stage.stage_dir / fold_names(config)[0]
    interrupted_dir.mkdir(parents=True)
    (interrupted_dir / "best_model.pt").write_text("x")

    captured = install_fake_queue(stage)
    results  = stage.run()

    assert captured.get("jobs", []) == []
    assert all(r["status"] == "SKIPPED" for r in results)
    assert len(results)             == 5 * 2


@pytest.mark.real_data
def test_inference_stage_reuses_existing_inference_on_resume(tmp_path, test_data_dir):
    config  = data_config(tmp_path, test_data_dir, resume=True)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))
    name    = fold_names(config)[0]

    fold_dir = stage.stage_dir / name
    mark_complete(fold_dir)

    for split in ("val", "test"):
        mark_complete(fold_dir / "inference" / split)

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert any(job.name.startswith(name) for job in captured.get("jobs", [])) is False
    assert statuses[f"{name}:val"]  == "DONE"
    assert statuses[f"{name}:test"] == "DONE"


@pytest.mark.real_data
def test_inference_stage_purges_unfinished_split_on_resume(tmp_path, test_data_dir):
    config  = data_config(tmp_path, test_data_dir, resume=True)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))
    name    = fold_names(config)[0]

    fold_dir = stage.stage_dir / name
    mark_complete(fold_dir)

    unfinished = fold_dir / "inference" / "val"
    unfinished.mkdir(parents=True)
    (unfinished / "metrics.json").write_text("{}")

    captured = install_fake_queue(stage)
    stage.run()

    assert f"{name}:val" in [job.name for job in captured["jobs"]]
    assert not unfinished.exists()


@pytest.mark.real_data
def test_inference_stage_job_command_carries_split(tmp_path, test_data_dir):
    config  = data_config(tmp_path, test_data_dir)
    planner = make_planner(config)
    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", planner=planner, logger=make_logger(tmp_path))
    name    = fold_names(config)[0]

    mark_complete(stage.stage_dir / name)

    captured = install_fake_queue(stage)
    stage.run()

    job     = next(job for job in captured["jobs"] if job.name == f"{name}:test")
    command = job.command
    assert "--split" in command
    assert command[command.index("--split") + 1] == "test"
    assert "--fold"  in command


def test_training_stage_keeps_bare_fold_names_for_profile_autoencoder(tmp_path):
    config               = stage_config(tmp_path)
    config.training_type = "profile_autoencoder"
    stage                = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_logger(tmp_path))

    assert stage.items == [f"fold_{index}" for index in range(5)]


def test_report_stage_invokes_collector_and_report(tmp_path, monkeypatch):
    config  = stage_config(tmp_path)
    planner = make_planner(config)
    stage   = CrossValidationReportStage(config=config, run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    seen = {}

    class FakeCollector:
        def __init__(self, run_dir, splits, logger):
            seen["splits"]       = splits
            self.seed_dispersion = {}

        def collect_by_split(self):
            return ["base"], {"test": ["rec"]}

    class FakeReport:
        def __init__(self, base_records, records_by_split, planner, out_dir, model_name, embed_images, logger, seed_dispersion=None):
            seen["base_records"]     = base_records
            seen["records_by_split"] = records_by_split
            seen["model_name"]       = model_name
            seen["out_dir"]          = out_dir
            seen["seed_dispersion"]  = seed_dispersion

        def write_all(self):
            seen["wrote"] = True
            return [seen["out_dir"] / "cv_aggregate_report.md"]

    monkeypatch.setattr(stages_module, "FoldCollector",         FakeCollector, raising=True)
    monkeypatch.setattr(stages_module, "CrossValidationReport", FakeReport,    raising=True)

    out_dir = stage.run()

    assert seen["splits"]           == ["val", "test"]
    assert seen["base_records"]     == ["base"]
    assert seen["records_by_split"] == {"test": ["rec"]}
    assert seen["model_name"]       == config.backbone_name
    assert seen["wrote"]            is True
    assert out_dir                  == seen["out_dir"]


def test_report_stage_model_name_for_profile_autoencoder(tmp_path, monkeypatch):
    config               = stage_config(tmp_path)
    config.training_type = "profile_autoencoder"
    planner              = make_planner(config)
    stage                = CrossValidationReportStage(config=config, run_tag="rt", planner=planner, logger=make_logger(tmp_path))

    seen = {}

    class FakeCollector:
        def __init__(self, run_dir, splits, logger):
            seen["splits"]       = splits
            self.seed_dispersion = {}

        def collect_by_split(self):
            return [], {}

    class FakeReport:
        def __init__(self, base_records, records_by_split, planner, out_dir, model_name, embed_images, logger, seed_dispersion=None):
            seen["model_name"] = model_name

        def write_all(self):
            return []

    monkeypatch.setattr(stages_module, "FoldCollector",         FakeCollector, raising=True)
    monkeypatch.setattr(stages_module, "CrossValidationReport", FakeReport,    raising=True)

    stage.run()

    assert seen["splits"]     == []
    assert seen["model_name"] == "profile_ae"
