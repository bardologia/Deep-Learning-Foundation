"""Tests for the fold training, fold inference and report stages of cross-validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib     import Path

import pytest

import pipelines.cross_validation.stages as stages_module
from configuration.cross_validation    import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.folds  import FoldPlanner, FoldRunNaming, FoldTreeSurvey, StoredFoldPlanner
from pipelines.cross_validation.stages import (
    CrossValidationReportRebuilder,
    CrossValidationReportStage,
    FoldInferenceStage,
    FoldTrainingStage,
)
from pipelines.cross_validation.workers import FoldTrainingWorker
from tools.data.io                      import FileIO
from tools.runtime.completion           import CompletionMarker
from configuration.training.general.run import DEFAULT_PARAMS_RUN

from tests.conftest import make_file_logger


def stage_config(tmp_path: Path, resume: bool = False) -> CrossValidationConfig:
    """Returns a cross-validation config with five folds writing under tmp_path."""
    config                    = CrossValidationConfig()
    config.paths.log_base_dir = tmp_path
    config.folds              = FoldConfig(n_folds=5, row_start=1000, row_end=1512)
    config.resume             = resume
    config.inference_splits   = ["val", "test"]
    config.seeds              = []
    return config


def data_config(tmp_path: Path, test_data_dir: Path, resume: bool = False) -> CrossValidationConfig:
    """Returns a stage config pointing at the real dataset and its extracted parameters."""
    config                       = stage_config(tmp_path, resume)
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    return config


def fold_names(config: CrossValidationConfig) -> list[str]:
    """Returns the fold run names the run naming derives from the config."""
    return [run_name for _, _, run_name in FoldRunNaming(config).units()]


def make_planner(config: CrossValidationConfig) -> FoldPlanner:
    """Returns a fold planner over the config with a fixed col extent."""
    return FoldPlanner(config, col_start=500, col_end=1000)


def mark_complete(directory: Path) -> None:
    """Creates the directory and stamps it with a completion marker."""
    directory.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(directory, {"stage": "test"})


def stamp_split_inference(fold_dir: Path, split: str) -> Path:
    """Stamps a per-split inference directory under the fold and returns it."""
    stamp = fold_dir / "inference" / f"stamp_{split}"
    stamp.mkdir(parents=True, exist_ok=True)
    CompletionMarker.stamp(stamp, {"stage": "inference", "split": split})
    return stamp


def build_finished_run(root: Path, run_tag: str, n_folds: int = 3, splits: tuple = ("test",), seeds: tuple = (), with_pipeline: bool = True) -> Path:
    """Writes a cross-validation run whose folds trained and inferred but whose report stage never ran.

    Every fold run carries what a training run persists and the fold-tree
    survey reads: the backbone architecture config, the run summary naming
    the model, the dataset creation config holding the fold's split regions
    from a real planner over the window [1000, 1512), a best checkpoint, a
    training completion marker and a stamped inference per split holding two
    metrics. The pipeline directory, with the stage state and the training
    results index, is optional so a run copied without it can be exercised.

    Args:
        root          : Directory the run folder is created in.
        run_tag       : Name of the run folder.
        n_folds       : Number of folds written.
        splits        : Splits every fold holds a completed inference for.
        seeds         : Seed directory names nested under every fold, empty for unseeded folds.
        with_pipeline : Whether the pipeline directory with the stage state and training results index is written.

    Returns:
        Path of the run directory.
    """
    import torch

    run_dir = root / run_tag

    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=n_folds, row_start=1000, row_end=1512)
    planner      = make_planner(config)

    results = []
    for index in range(n_folds):
        fold_dir = run_dir / "folds" / f"{config.backbone_name}_fold_{index}"
        run_dirs = [fold_dir / seed for seed in seeds] if seeds else [fold_dir]

        for unit_dir in run_dirs:
            (unit_dir / "meta").mkdir(parents=True)
            FileIO.save_json({}, unit_dir / "meta" / "model_config.json")
            FileIO.save_json({"model_name": config.backbone_name, "in_channels": 4, "out_channels": 6}, unit_dir / "meta" / "run_summary.json")
            FileIO.save_json({"split_regions": asdict(planner.plan(index).split_regions)}, unit_dir / "meta" / "dataset_creation_config.json")

            torch.save({"best_val_loss": 0.1 * (index + 1), "best_epoch": 3, "epoch": 5}, unit_dir / "best_model.pt")
            CompletionMarker.stamp(unit_dir, {"stage": "training", "epochs_completed": 5})

            for split in splits:
                stamp = stamp_split_inference(unit_dir, split)
                FileIO.save_json({"curve_rmse_gt": 0.1 * (index + 1), "overall_r2_gt": 0.9 - 0.05 * index}, stamp / "metrics.json")

            results.append({"name": str(unit_dir.relative_to(run_dir / "folds")), "status": "DONE", "duration_s": 60.0})

    if with_pipeline:
        FileIO.save_json({"run_tag": run_tag, "stages": {"training": {"status": "completed", "timestamp": "t"}}}, run_dir / "pipeline" / "state.json")
        FileIO.save_json(results, run_dir / "pipeline" / "training_results.json")

    return run_dir


def queue_result(name: str, status: str = "DONE") -> dict:
    """Returns a GPU queue result payload for the given job name and status."""
    return {"name": name, "gpu": 0, "status": status, "returncode": 0, "duration_s": 1.0, "log_file": ""}


def install_fake_queue(stage, status: str = "DONE") -> dict:
    """Replaces the stage's GPU queue with a fake that records the submitted jobs.

    Args:
        stage  : Stage whose _run_queue is replaced.
        status : Status reported for every submitted job.

    Returns:
        Dictionary that receives the submitted jobs under the 'jobs' key.
    """
    captured = {}

    def fake_queue(jobs):
        """Records the submitted jobs and returns one result per job."""
        captured["jobs"] = jobs
        return [queue_result(job.name, status) for job in jobs]

    stage._run_queue = fake_queue
    return captured


@pytest.mark.dataset
def test_training_stage_builds_one_item_per_fold(tmp_path, test_data_dir):
    """Verifies the training stage builds one item per fold, ordered by fold index."""
    config = data_config(tmp_path, test_data_dir)
    stage  = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))

    assert stage.items == fold_names(config)
    assert [name.split("_fold_")[-1] for name in stage.items] == [str(index) for index in range(5)]


@pytest.mark.dataset
def test_training_stage_items_carry_the_training_tag_the_worker_trains_into(tmp_path, test_data_dir):
    """Verifies the stage item names and directories match what the training worker writes."""
    config = data_config(tmp_path, test_data_dir)
    stage  = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    worker = FoldTrainingWorker(config=config, run_tag="rt")

    assert stage.items                                   == [worker.fold_run_name(index, None) for index in range(5)]
    assert stage.stage_dir / stage.items[3]              == worker.run_dir / "folds" / worker.fold_run_name(3, None)
    assert stage.items[3].startswith(f"{config.backbone_name}-{config.backbone_head}-")


@pytest.mark.dataset
def test_training_stage_job_carries_fold_index(tmp_path, test_data_dir):
    """Verifies a training job passes the fold index and no seed when no seeds are configured."""
    config = data_config(tmp_path, test_data_dir)
    stage  = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))

    job = stage._job(stage.items[3])

    assert job.command[job.command.index("--worker") + 1] == "train"
    assert job.command[job.command.index("--fold")   + 1] == "3"
    assert "--seed" not in job.command


@pytest.mark.dataset
def test_training_stage_seed_sweep_expands_fold_by_seed(tmp_path, test_data_dir):
    """Verifies configured seeds expand the items to the fold-by-seed cross product."""
    config       = data_config(tmp_path, test_data_dir)
    config.seeds = [1, 2]
    stage        = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    worker       = FoldTrainingWorker(config=config, run_tag="rt")

    assert stage.items == [worker.fold_run_name(index, seed) for index in range(5) for seed in (1, 2)]

    item = worker.fold_run_name(3, 2)
    job  = stage._job(item)
    assert job.command[job.command.index("--fold") + 1] == "3"
    assert job.command[job.command.index("--seed") + 1] == "2"
    assert job.log_path == stage.stage_dir / item / stage.worker_logname


@pytest.mark.dataset
def test_training_stage_subdir_and_results_path(tmp_path, test_data_dir):
    """Verifies the training stage writes into folds/ and reports results under pipeline/."""
    stage = FoldTrainingStage(config=data_config(tmp_path, test_data_dir), entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))

    assert stage.stage_dir.name            == "folds"
    assert stage.results_path.name         == "training_results.json"
    assert stage.results_path.parent.name  == "pipeline"


@pytest.mark.dataset
def test_training_stage_run_executes_all_folds(tmp_path, test_data_dir):
    """Verifies running the stage queues every fold and persists the results file."""
    config   = data_config(tmp_path, test_data_dir)
    stage    = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    captured = install_fake_queue(stage)

    results = stage.run()

    assert [job.name for job in captured["jobs"]] == fold_names(config)
    assert [r["name"] for r in results]           == fold_names(config)
    assert all(r["status"] == "DONE" for r in results)
    assert stage.results_path.exists()


@pytest.mark.dataset
def test_training_stage_job_command_carries_fold_flag(tmp_path, test_data_dir):
    """Verifies the queued job command carries the fold flag with the right index."""
    stage    = FoldTrainingStage(config=data_config(tmp_path, test_data_dir), entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    captured = install_fake_queue(stage)

    stage.run()

    command = captured["jobs"][3].command
    assert "--fold" in command
    assert command[command.index("--fold") + 1] == "3"


@pytest.mark.dataset
def test_training_stage_reuses_completion_written_under_worker_naming(tmp_path, test_data_dir):
    """Verifies a completed fold-seed run is reused instead of requeued on resume."""
    config       = data_config(tmp_path, test_data_dir, resume=True)
    config.seeds = [1, 2]
    stage        = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
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


@pytest.mark.dataset
def test_inference_stage_one_job_per_fold_with_completed_training(tmp_path, test_data_dir):
    """Verifies inference queues only folds with completed training and skips the rest."""
    config  = data_config(tmp_path, test_data_dir)
    stage = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    names   = fold_names(config)

    for name in names[:2]:
        mark_complete(stage.stage_dir / name)

    captured = install_fake_queue(stage)
    results  = stage.run()

    assert sorted(job.name for job in captured["jobs"]) == sorted(names[:2])

    statuses = {r["name"]: r["status"] for r in results}
    assert statuses[names[0]] == "DONE"
    assert statuses[names[2]] == "SKIPPED"
    assert len(results)       == 5


@pytest.mark.dataset
def test_inference_stage_sees_completion_written_under_worker_naming(tmp_path, test_data_dir):
    """Verifies the inference stage detects training completion written under the worker naming."""
    config  = data_config(tmp_path, test_data_dir)
    stage = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    worker  = FoldTrainingWorker(config=config, run_tag="rt")

    trained = worker.fold_run_name(0, None)
    mark_complete(worker.run_dir / "folds" / trained)

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert [job.name for job in captured["jobs"]] == [trained]
    assert statuses[trained]                      == "DONE"


@pytest.mark.dataset
def test_inference_stage_skips_folds_without_completed_training(tmp_path, test_data_dir):
    """Verifies a fold with a checkpoint but no completion marker is skipped."""
    config  = data_config(tmp_path, test_data_dir)
    stage = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))

    interrupted_dir = stage.stage_dir / fold_names(config)[0]
    interrupted_dir.mkdir(parents=True)
    (interrupted_dir / "best_model.pt").write_text("x")

    captured = install_fake_queue(stage)
    results  = stage.run()

    assert captured.get("jobs", []) == []
    assert all(r["status"] == "SKIPPED" for r in results)
    assert len(results)             == 5


@pytest.mark.dataset
def test_inference_stage_reuses_run_with_every_split_stamped_on_resume(tmp_path, test_data_dir):
    """Verifies a resumed run with every split stamped is reported done without requeuing."""
    config  = data_config(tmp_path, test_data_dir, resume=True)
    stage = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    name    = fold_names(config)[0]

    fold_dir = stage.stage_dir / name
    mark_complete(fold_dir)

    for split in ("val", "test"):
        stamp_split_inference(fold_dir, split)

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert name not in [job.name for job in captured.get("jobs", [])]
    assert statuses[name] == "DONE"


@pytest.mark.dataset
def test_inference_stage_requeues_run_with_a_missing_split_on_resume(tmp_path, test_data_dir):
    """Verifies a resumed run missing a split is requeued while the stamped split survives."""
    config  = data_config(tmp_path, test_data_dir, resume=True)
    stage = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    name    = fold_names(config)[0]

    fold_dir = stage.stage_dir / name
    mark_complete(fold_dir)
    stamp_split_inference(fold_dir, "val")

    captured = install_fake_queue(stage)
    stage.run()

    assert [job.name for job in captured["jobs"]] == [name]
    assert (fold_dir / "inference" / "stamp_val").exists()


@pytest.mark.dataset
def test_inference_stage_purges_unfinished_stamp_on_resume(tmp_path, test_data_dir):
    """Verifies an unstamped inference directory is deleted and the fold requeued on resume."""
    config  = data_config(tmp_path, test_data_dir, resume=True)
    stage = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    name    = fold_names(config)[0]

    fold_dir = stage.stage_dir / name
    mark_complete(fold_dir)

    unfinished = fold_dir / "inference" / "stamp_val"
    unfinished.mkdir(parents=True)
    (unfinished / "metrics.json").write_text("{}")

    captured = install_fake_queue(stage)
    stage.run()

    assert name in [job.name for job in captured["jobs"]]
    assert not unfinished.exists()


@pytest.mark.dataset
def test_inference_stage_job_command_carries_fold_without_split(tmp_path, test_data_dir):
    """Verifies the inference job channels the fold index and the infer worker without a split flag."""
    config  = data_config(tmp_path, test_data_dir)
    stage = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    name    = fold_names(config)[0]

    mark_complete(stage.stage_dir / name)

    captured = install_fake_queue(stage)
    stage.run()

    job     = next(job for job in captured["jobs"] if job.name == name)
    command = job.command
    assert "--split" not in command
    assert "--fold"      in command
    assert command[command.index("--fold") + 1]   == "0"
    assert command[command.index("--worker") + 1] == "infer"
    assert job.log_path == stage.stage_dir / name / stage.worker_logname


def test_training_stage_keeps_bare_fold_names_for_profile_autoencoder(tmp_path):
    """Verifies the profile autoencoder training type keeps plain fold_N item names."""
    config               = stage_config(tmp_path)
    config.training_type = "profile_autoencoder"
    stage                = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))

    assert stage.items == [f"fold_{index}" for index in range(5)]


def test_report_stage_invokes_collector_and_report(tmp_path, monkeypatch):
    """Verifies the report stage feeds the collected records into the cross-validation report."""
    config  = stage_config(tmp_path)
    planner = make_planner(config)
    stage   = CrossValidationReportStage(config=config, run_tag="rt", planner=planner, logger=make_file_logger(tmp_path, "stages_test"))

    seen = {}

    class FakeCollector:
        """Fold collector stub returning one base record and one test-split record."""
        def __init__(self, run_dir, splits, training_type, logger):
            """Records the requested splits and exposes an empty seed dispersion."""
            seen["splits"]       = splits
            self.seed_dispersion = {}

        def collect_by_split(self):
            """Returns the canned base records and per-split records."""
            return ["base"], {"test": ["rec"]}

    class FakeReport:
        """Cross-validation report stub recording its constructor arguments."""
        def __init__(self, base_records, records_by_split, planner, out_dir, model_name, embed_images, logger, seed_dispersion=None):
            """Records the records, model name, output directory and seed dispersion."""
            seen["base_records"]     = base_records
            seen["records_by_split"] = records_by_split
            seen["model_name"]       = model_name
            seen["out_dir"]          = out_dir
            seen["seed_dispersion"]  = seed_dispersion

        def write_all(self):
            """Records that writing happened and returns the aggregate report path."""
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
    """Verifies the report stage reports the autoencoder model name for profile autoencoder runs."""
    config                  = stage_config(tmp_path)
    config.training_type    = "profile_autoencoder"
    config.inference_splits = []
    planner                 = make_planner(config)
    stage                = CrossValidationReportStage(config=config, run_tag="rt", planner=planner, logger=make_file_logger(tmp_path, "stages_test"))

    seen = {}

    class FakeCollector:
        """Fold collector stub returning no records."""
        def __init__(self, run_dir, splits, training_type, logger):
            """Records the requested splits and exposes an empty seed dispersion."""
            seen["splits"]       = splits
            self.seed_dispersion = {}

        def collect_by_split(self):
            """Returns empty base records and an empty split mapping."""
            return [], {}

    class FakeReport:
        """Cross-validation report stub recording only the model name."""
        def __init__(self, base_records, records_by_split, planner, out_dir, model_name, embed_images, logger, seed_dispersion=None):
            """Records the model name passed by the stage."""
            seen["model_name"] = model_name

        def write_all(self):
            """Returns no written files."""
            return []

    monkeypatch.setattr(stages_module, "FoldCollector",         FakeCollector, raising=True)
    monkeypatch.setattr(stages_module, "CrossValidationReport", FakeReport,    raising=True)

    stage.run()

    assert seen["splits"]     == []
    assert seen["model_name"] == config.autoencoder.ae_model_name


def test_report_stage_honours_an_explicit_run_dir(tmp_path):
    """Verifies an explicit run directory replaces the one derived from the log base and run tag."""
    config = stage_config(tmp_path)
    stage  = CrossValidationReportStage(config=config, run_tag="rt", planner=make_planner(config), logger=make_file_logger(tmp_path, "stages_test"), run_dir=tmp_path / "workspace" / "cv_20260912")

    assert stage.run_dir == tmp_path / "workspace" / "cv_20260912"


def test_report_rebuilder_writes_the_report_from_the_fold_tree_alone(tmp_path):
    """Verifies a run holding nothing but its folds gets a full cross-fold report, fold plan included."""
    run_dir = build_finished_run(tmp_path / "workspace", "cv_20260912_100000", with_pipeline=False)

    out_dir = CrossValidationReportRebuilder(run_dir, make_file_logger(tmp_path, "rebuild_test")).rebuild()

    aggregate = (out_dir / "cv_aggregate_report.md").read_text(encoding="utf-8")

    assert not (run_dir / "pipeline").exists()
    assert out_dir.parent == run_dir / "reports"
    assert (out_dir / "cv_summary.json").is_file()
    assert (out_dir / "test" / "benchmark_overview.md").is_file()
    assert (out_dir / "test" / "metrics_comparison.md").is_file()
    assert aggregate.startswith("# Cross-Validation Aggregate Report")
    assert "Model `resunet`, 3 folds." in aggregate
    assert "| 0    | [1000, 1162) | [1178, 1332) | [1348, 1512)      |" in aggregate


def test_report_rebuilder_handles_seed_sweeps(tmp_path):
    """Verifies folds trained under several seeds are surveyed through their first seed and aggregated per fold."""
    run_dir = build_finished_run(tmp_path / "workspace", "cv_seeded", seeds=("seed0", "seed1"), with_pipeline=False)

    out_dir = CrossValidationReportRebuilder(run_dir, make_file_logger(tmp_path, "rebuild_test")).rebuild()

    aggregate = (out_dir / "cv_aggregate_report.md").read_text(encoding="utf-8")

    assert "Each fold was trained under multiple seeds." in aggregate
    assert "Model `resunet`, 3 folds." in aggregate


def test_report_rebuilder_aborts_when_a_fold_lacks_its_run_summary(tmp_path):
    """Verifies a fold run without the run summary naming its model aborts naming that file."""
    run_dir = build_finished_run(tmp_path / "workspace", "cv_20260912_100000")
    (run_dir / "folds" / "resunet_fold_0" / "meta" / "run_summary.json").unlink()

    with pytest.raises(FileNotFoundError, match="run_summary.json"):
        CrossValidationReportRebuilder(run_dir, make_file_logger(tmp_path, "rebuild_test")).rebuild()


def test_fold_tree_survey_reads_family_model_splits_and_plan(tmp_path):
    """Verifies the survey recovers the family, the model name, the scored splits and the stored fold plan."""
    run_dir   = build_finished_run(tmp_path / "workspace", "cv_20260912_100000", splits=("val", "test"))
    survey    = FoldTreeSurvey(run_dir)
    fold_runs = survey.fold_runs()
    first     = fold_runs[0]

    config       = stage_config(tmp_path)
    config.folds = FoldConfig(n_folds=3, row_start=1000, row_end=1512)
    expected     = make_planner(config)

    assert sorted(fold_runs) == [0, 1, 2]
    assert first == run_dir / "folds" / "resunet_fold_0"
    assert survey.training_type(first) == "backbone"
    assert survey.model_name(first) == "resunet"
    assert survey.splits(fold_runs, "backbone") == ["test", "val"]
    assert [plan.split_regions for plan in survey.planner(fold_runs).plans()] == [plan.split_regions for plan in expected.plans()]


def test_fold_tree_survey_rejects_a_gap_in_the_fold_indices(tmp_path):
    """Verifies folds whose indices do not run contiguously from zero are refused."""
    run_dir = build_finished_run(tmp_path / "workspace", "cv_20260912_100000")
    (run_dir / "folds" / "resunet_fold_2").rename(run_dir / "folds" / "resunet_fold_5")

    survey = FoldTreeSurvey(run_dir)

    with pytest.raises(ValueError, match="contiguously"):
        survey.planner(survey.fold_runs())


def test_stored_fold_planner_follows_the_planner_block_convention():
    """Verifies stored regions yield the same plans a configured planner builds, block assignment included."""
    config       = CrossValidationConfig()
    config.folds = FoldConfig(n_folds=4, row_start=1000, row_end=1512)
    configured   = make_planner(config)

    stored = StoredFoldPlanner({plan.fold_index: plan.split_regions for plan in configured.plans()})

    assert stored.n_folds == 4
    assert stored.plans() == configured.plans()


@pytest.mark.dataset
def test_training_stage_reuses_a_completed_unit_found_through_its_manifest(tmp_path, test_data_dir):
    """Verifies a completed fold run living under another directory name is matched by its manifest and reused, with the worker pointed at that directory."""
    from pipelines.cross_validation.folds import FoldUnitManifest

    config       = data_config(tmp_path, test_data_dir, resume=True)
    config.seeds = [1, 2]

    old = tmp_path / "rt" / "folds" / "old-label_fold_0" / "seed1"
    FoldUnitManifest.write(old, run_tag="rt", unit="old-label_fold_0/seed1", fold_index=0, seed=1, training_type="backbone", inference_regions={"test": (0, 1, 2, 3)})
    mark_complete(old)

    stage  = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    worker = FoldTrainingWorker(config=config, run_tag="rt")
    planned = worker.fold_run_name(0, 1)

    assert stage.unit_dirs == {planned: "old-label_fold_0/seed1"}
    assert stage.unit_dir(planned) == old
    assert stage._job(worker.fold_run_name(0, 2)).command[stage._job(worker.fold_run_name(0, 2)).command.index("--run-name") + 1] == worker.fold_run_name(0, 2)

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert planned not in [job.name for job in captured["jobs"]]
    assert statuses[planned] == "DONE"
    assert old.is_dir()


@pytest.mark.dataset
def test_training_stage_resumes_an_unfinished_unit_in_its_own_directory(tmp_path, test_data_dir):
    """Verifies an unfinished fold run with a trainer state is kept and its worker is handed that directory when the trainer resumes."""
    from pipelines.cross_validation.folds import FoldUnitManifest

    config                 = data_config(tmp_path, test_data_dir, resume=True)
    config.training.resume = True

    old = tmp_path / "rt" / "folds" / "old-label_fold_2"
    FoldUnitManifest.write(old, run_tag="rt", unit="old-label_fold_2", fold_index=2, seed=None, training_type="backbone", inference_regions={})
    (old / "last.pt").write_text("x")

    stage   = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    planned = FoldTrainingWorker(config=config, run_tag="rt").fold_run_name(2, None)

    captured = install_fake_queue(stage)
    stage.run()

    job = next(job for job in captured["jobs"] if job.name == planned)
    assert job.command[job.command.index("--run-name") + 1] == "old-label_fold_2"
    assert job.log_path == old / stage.worker_logname
    assert (old / "last.pt").exists()


@pytest.mark.dataset
def test_inference_stage_scores_the_directory_the_manifest_identifies(tmp_path, test_data_dir):
    """Verifies the inference stage reads training completion and inference stamps from the manifest-matched directory."""
    from pipelines.cross_validation.folds import FoldUnitManifest

    config = data_config(tmp_path, test_data_dir, resume=True)

    old = tmp_path / "rt" / "folds" / "old-label_fold_1"
    FoldUnitManifest.write(old, run_tag="rt", unit="old-label_fold_1", fold_index=1, seed=None, training_type="backbone", inference_regions={})
    mark_complete(old)
    stamp_split_inference(old, "val")

    stage   = FoldInferenceStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    planned = FoldTrainingWorker(config=config, run_tag="rt").fold_run_name(1, None)

    captured = install_fake_queue(stage)
    results  = stage.run()

    statuses = {r["name"]: r["status"] for r in results}
    assert statuses[planned] == "DONE"
    job = next(job for job in captured["jobs"] if job.name == planned)
    assert job.command[job.command.index("--run-name") + 1] == "old-label_fold_1"


@pytest.mark.dataset
def test_stage_reports_run_directories_without_a_manifest(tmp_path, test_data_dir):
    """Verifies fold runs carrying no manifest are listed as unidentified and never matched to a unit."""
    config = data_config(tmp_path, test_data_dir, resume=True)

    stray = tmp_path / "rt" / "folds" / "stray_fold_9"
    (stray / "meta").mkdir(parents=True)

    stage = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))

    assert stage.unit_dirs    == {}
    assert stage.unidentified == [stray]


@pytest.mark.dataset
def test_status_report_lists_every_unit_with_its_directory_and_state(tmp_path, test_data_dir):
    """Verifies the status report names the matched directory, the training state and the inference state of every unit."""
    from pipelines.cross_validation.folds import FoldUnitManifest

    config = data_config(tmp_path, test_data_dir, resume=True)

    old = tmp_path / "rt" / "folds" / "old-label_fold_0"
    FoldUnitManifest.write(old, run_tag="rt", unit="old-label_fold_0", fold_index=0, seed=None, training_type="backbone", inference_regions={})
    mark_complete(old)
    stamp_split_inference(old, "val")

    stage = FoldTrainingStage(config=config, entry_script=Path("e.py"), run_tag="rt", logger=make_file_logger(tmp_path, "stages_test"))
    rows  = {row["Unit"]: row for row in stage.status_report().rows()}
    first = rows[stage.items[0]]

    assert first["Directory"] == "old-label_fold_0"
    assert first["Training"]  == "complete"
    assert first["Inference"] == "val done, test missing"
    assert rows[stage.items[1]]["Directory"] == "(new)"
    assert rows[stage.items[1]]["Training"]  == "new"
