
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pipelines.backbone.inference.pipeline as pipeline_module
import pipelines.backbone.training.pipeline  as backbone_pipeline_module
from configuration.cross_validation     import CrossValidationConfig, FoldConfig
from pipelines.cross_validation.folds   import FoldNaming
from pipelines.cross_validation.workers import (
    CrossValidationWorker,
    FoldCollector,
    FoldInferenceWorker,
    FoldTrainingWorker,
)
from tools.data.io                      import FileIO
from tools.data.regions                 import SplitRegions
from tools.runtime.completion           import CompletionMarker
from configuration.synthesis            import SynthesisConfig
from configuration.training.general.run import DEFAULT_PARAMS_RUN


from tests.conftest import RecordingLogger, make_file_logger


def worker_config(test_data_dir: Path) -> CrossValidationConfig:
    config                       = CrossValidationConfig()
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    config.folds                 = FoldConfig(n_folds=5, row_start=1000, row_end=1512)
    return config


def stamp_split_inference(base: Path, split: str, metrics: dict) -> Path:
    stamp = base / f"stamp_{split}"
    stamp.mkdir(parents=True)

    CompletionMarker.stamp(stamp, {"stage": "inference", "split": split})
    FileIO.save_json(metrics, stamp / "metrics.json")

    return stamp


def build_run_dir(tmp_path: Path, fold_names: list[str], inference_folds: dict[str, dict]) -> Path:
    run_dir = tmp_path / "run"

    (run_dir / "pipeline").mkdir(parents=True)
    for name in fold_names:
        (run_dir / "folds" / name).mkdir(parents=True)

    training_results = [{"name": name, "status": "DONE", "duration_s": 10.0, "log_file": str(run_dir / "folds" / name / "worker.log")} for name in fold_names]
    FileIO.save_json(training_results, run_dir / "pipeline" / "training_results.json")

    for name, split_metrics in inference_folds.items():
        for split, metrics in split_metrics.items():
            stamp_split_inference(run_dir / "folds" / name / "inference", split, metrics)

    return run_dir


def test_collector_points_at_folds_subdir(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_file_logger(tmp_path, "workers_test"))

    assert collector.training_dir == run_dir / "folds"
    assert collector.splits       == ["test"]


def test_collector_collects_every_fold(tmp_path):
    names     = [FoldNaming.name(i) for i in range(5)]
    run_dir   = build_run_dir(tmp_path, names, {})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_file_logger(tmp_path, "workers_test"))

    base, by_split = collector.collect_by_split()

    assert [record.name for record in base] == names
    assert set(by_split)                    == {"test"}
    assert len(by_split["test"])            == 5


def test_collector_attaches_training_results(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    collector = FoldCollector(run_dir=run_dir, splits=[], logger=make_file_logger(tmp_path, "workers_test"))

    base, _ = collector.collect_by_split()

    assert base[0].training_result["status"]     == "DONE"
    assert base[0].training_result["duration_s"] == 10.0


def test_collector_split_view_loads_metrics_when_present(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {"fold_0": {"test": {"curve_rmse_gt": 2.5}}})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_file_logger(tmp_path, "workers_test"))

    _, by_split = collector.collect_by_split()
    records     = by_split["test"]

    assert records[0].metrics       == {"curve_rmse_gt": 2.5}
    assert records[0].inference_dir is not None
    assert CompletionMarker.payload(records[0].inference_dir)["split"] == "test"


def test_collector_split_view_empty_when_metrics_absent(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {"fold_0": {"test": {"curve_rmse_gt": 2.5}}})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_file_logger(tmp_path, "workers_test"))

    _, by_split = collector.collect_by_split()
    records     = by_split["test"]

    assert records[1].metrics       == {}
    assert records[1].inference_dir is None
    assert records[1].figures       == []


def test_collector_aggregates_seeds_per_fold(tmp_path):
    names     = ["fold_0/seed1", "fold_0/seed2", "fold_1/seed1", "fold_1/seed2"]
    inference = {
        "fold_0/seed1": {"test": {"curve_rmse_gt": 2.0}},
        "fold_0/seed2": {"test": {"curve_rmse_gt": 4.0}},
        "fold_1/seed1": {"test": {"curve_rmse_gt": 1.0}},
        "fold_1/seed2": {"test": {"curve_rmse_gt": 3.0}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_file_logger(tmp_path, "workers_test"))

    base, by_split = collector.collect_by_split()

    assert [record.name for record in base]             == ["fold_0", "fold_1"]
    assert by_split["test"][0].metrics["curve_rmse_gt"] == 3.0
    assert by_split["test"][1].metrics["curve_rmse_gt"] == 2.0

    dispersion = collector.seed_dispersion
    assert dispersion["fold_0"]["n_seeds"] == 2
    assert dispersion["fold_0"]["splits"]["test"]["curve_rmse_gt"] == pytest.approx(float(np.std([2.0, 4.0], ddof=1)))


def test_collector_multiple_splits_independent(tmp_path):
    run_dir = build_run_dir(
        tmp_path,
        ["fold_0", "fold_1"],
        {"fold_0": {"val": {"curve_rmse_gt": 1.0}, "test": {"curve_rmse_gt": 2.0}}},
    )
    collector = FoldCollector(run_dir=run_dir, splits=["val", "test"], logger=make_file_logger(tmp_path, "workers_test"))

    _, by_split = collector.collect_by_split()

    assert by_split["val"][0].metrics  == {"curve_rmse_gt": 1.0}
    assert by_split["test"][0].metrics == {"curve_rmse_gt": 2.0}


@pytest.mark.dataset
def test_cross_validation_worker_fold_name(test_data_dir):
    worker = CrossValidationWorker(worker_config(test_data_dir), run_tag="rt")

    assert worker.fold_name(0) == "fold_0"
    assert worker.fold_name(4) == "fold_4"


@pytest.mark.dataset
def test_training_worker_dispatches_to_backbone(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    captured = {}

    def fake_backbone(self, fold_index, seed, split_regions, run_name):
        captured["fold"]     = fold_index
        captured["seed"]     = seed
        captured["regions"]  = split_regions
        captured["run_name"] = run_name

    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", fake_backbone, raising=True)

    worker.run(2)

    assert captured["fold"]     == 2
    assert captured["seed"]     is None
    assert captured["run_name"] == worker.fold_run_name(2, None)
    assert isinstance(captured["regions"], SplitRegions)


@pytest.mark.dataset
def test_training_worker_writes_the_unit_manifest_before_training(test_data_dir, tmp_path, monkeypatch):
    from pipelines.cross_validation.folds             import FoldUnitManifest
    from pipelines.shared.orchestration.unit_manifest import UnitManifest

    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    seen = {}
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr, run_name: seen.update(manifest=UnitManifest.read(worker.unit_directory(run_name))), raising=True)

    worker.run(3, seed=7)

    manifest = seen["manifest"]
    assert manifest["kind"]       == FoldUnitManifest.KIND
    assert manifest["fold_index"] == 3
    assert manifest["seed"]       == 7
    assert manifest["run_tag"]    == "rt"
    assert manifest["unit"]       == worker.fold_run_name(3, 7)
    assert FoldUnitManifest.identity(manifest) == (3, 7)
    assert sorted(manifest["inference_regions"]) == ["test", "val"]
    assert manifest["inference_regions"]["test"] == list(worker.factory.planner().inference_region(3, "test").as_tuple())


@pytest.mark.dataset
def test_training_worker_leaves_a_finished_unit_alone(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    unit = worker.unit_directory(worker.fold_run_name(1, 4))
    unit.mkdir(parents=True)
    CompletionMarker.stamp(unit, {"stage": "training"})

    trained = []
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr, run_name: trained.append(run_name), raising=True)

    worker.run(1, seed=4)

    assert trained == []


@pytest.mark.dataset
def test_training_worker_trains_into_the_directory_the_scheduler_hands_over(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    seen = {}
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr, run_name: seen.update(run_name=run_name), raising=True)

    worker.run(1, seed=None, run_name="old-label_fold_1")

    assert seen["run_name"] == "old-label_fold_1"
    assert (worker.unit_directory("old-label_fold_1") / "meta" / "unit.json").is_file()


@pytest.mark.dataset
def test_training_worker_passes_planned_split_regions(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    captured = {}
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr, run_name: captured.update(sr=sr), raising=True)

    worker.run(2)

    train = sorted((r.row_start, r.row_end) for r in captured["sr"].regions("train"))
    assert train == [(1000, 1196), (1416, 1512)]
    assert worker.factory.planner().plan(2).split_regions.regions("test")[0].row_start == 1212


@pytest.mark.dataset
def test_inference_worker_builds_run_directory(test_data_dir, tmp_path, monkeypatch):
    config                       = worker_config(test_data_dir)
    config.paths.log_base_dir    = tmp_path
    config.inference_splits      = ["test"]
    worker                       = FoldInferenceWorker(config, run_tag="rt")

    captured = {}

    class FakePipeline:
        def __init__(self, inference_config):
            captured["config"] = inference_config

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(pipeline_module, "InferencePipeline", FakePipeline, raising=True)

    worker.run(1)

    expected_dir = worker.run_dir / "folds" / worker.fold_run_name(1, None)
    assert captured["ran"]                  is True
    assert captured["config"].run_directory == expected_dir
    assert expected_dir.name.endswith("_fold_1")
    assert expected_dir.name.startswith(f"{worker.config.backbone_name}-{worker.config.backbone_head}-")
    assert "-K_" in expected_dir.name
    assert captured["config"].split         == "test"
    assert captured["config"].output_subdir is None


@pytest.mark.dataset
def test_inference_worker_runs_every_configured_split(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    config.inference_splits   = ["val", "test"]
    worker                    = FoldInferenceWorker(config, run_tag="rt")

    ran = []

    class FakePipeline:
        def __init__(self, inference_config):
            self.split = inference_config.split

        def run(self):
            ran.append(self.split)

    monkeypatch.setattr(pipeline_module, "InferencePipeline", FakePipeline, raising=True)

    worker.run(1)

    assert ran == ["val", "test"]


@pytest.mark.dataset
def test_inference_worker_skips_completed_split_on_resume(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    config.inference_splits   = ["val", "test"]
    worker                    = FoldInferenceWorker(config, run_tag="rt")

    done = worker.run_dir / "folds" / worker.fold_run_name(1, None) / "inference" / "stamp_val"
    done.mkdir(parents=True)
    CompletionMarker.stamp(done, {"stage": "inference", "split": "val"})

    ran = []

    class FakePipeline:
        def __init__(self, inference_config):
            self.split = inference_config.split

        def run(self):
            ran.append(self.split)

    monkeypatch.setattr(pipeline_module, "InferencePipeline", FakePipeline, raising=True)

    worker.run(1)

    assert ran == ["test"]
    assert done.exists()


@pytest.mark.dataset
def test_backbone_fold_trainer_uses_the_entry_curriculum(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    config.prior_bias_init    = True
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    captured = {}

    class _FakePipeline:
        def __init__(self, trainer_config, dataset_config, backbone_name, model_config, seed, run_name, overfit_check, prior_bias_init):
            captured["trainer_config"]  = trainer_config
            captured["dataset_config"]  = dataset_config
            captured["run_name"]        = run_name
            captured["overfit_check"]   = overfit_check
            captured["prior_bias_init"] = prior_bias_init

        def run(self, probe_config=None):
            pass

    monkeypatch.setattr(backbone_pipeline_module, "TrainingPipeline", _FakePipeline, raising=True)

    worker.run(1, seed=7)

    trainer_config = captured["trainer_config"]
    assert trainer_config.curriculum is config.curriculum
    assert trainer_config.curriculum.enabled is True
    assert trainer_config.curriculum.complete.use_param_l1 is True
    from pipelines.shared.training.run_naming import RunNaming

    run_tag = RunNaming.training_tag(config.backbone_name, config.backbone_head, config.curriculum, SynthesisConfig().k_max, config.augmentation)
    assert captured["run_name"] == f"{run_tag}_fold_1/seed7"

    assert captured["overfit_check"] is config.overfit_check
    assert captured["prior_bias_init"] is True
    assert captured["dataset_config"].input_config is config.input


def test_collector_rebuilds_training_results_from_markers_when_file_absent(tmp_path):
    run_dir = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    (run_dir / "pipeline" / "training_results.json").unlink()

    CompletionMarker.stamp(run_dir / "folds" / "fold_0", {"stage": "training", "epochs_completed": 2})

    collector = FoldCollector(run_dir=run_dir, splits=[], logger=make_file_logger(tmp_path, "workers_test"))

    base, _ = collector.collect_by_split()

    assert base[0].training_result["status"] == "DONE"
    assert base[1].training_result["status"] == "PARTIAL"


def test_collector_warns_when_a_seed_has_no_metrics_and_records_the_coverage(tmp_path):
    names     = ["fold_0/seed1", "fold_0/seed2", "fold_1/seed1", "fold_1/seed2"]
    inference = {
        "fold_0/seed1": {"test": {"curve_rmse_gt": 2.0}},
        "fold_1/seed1": {"test": {"curve_rmse_gt": 1.0}},
        "fold_1/seed2": {"test": {"curve_rmse_gt": 3.0}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=logger)

    _, by_split = collector.collect_by_split()

    assert by_split["test"][0].metrics["curve_rmse_gt"] == 2.0
    assert ("warning", "fold_0 test: metrics averaged over 1/2 seeds; missing ['seed2'] (no completed inference with metrics)") in logger.messages
    assert collector.seed_dispersion["fold_0"]["n_used"]        == {"test": 1}
    assert collector.seed_dispersion["fold_0"]["missing_seeds"] == {"test": ["seed2"]}
    assert collector.seed_dispersion["fold_1"]["n_used"]        == {"test": 2}
    assert collector.seed_dispersion["fold_1"]["missing_seeds"] == {"test": []}
    assert any(level == "warning" and message.startswith("1 of 2 folds hold a seed that did not finish or average fewer seeds") for level, message in logger.messages)


def test_collector_marks_a_fold_partial_and_names_the_seed_that_did_not_finish(tmp_path):
    names   = ["fold_0/seed1", "fold_0/seed2"]
    run_dir = build_run_dir(tmp_path, names, {})

    results = FileIO.load_json(run_dir / "pipeline" / "training_results.json")
    results[1]["status"] = "FAILED"
    FileIO.save_json(results, run_dir / "pipeline" / "training_results.json")

    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=[], logger=logger)

    base, _ = collector.collect_by_split()

    assert base[0].training_result["status"]           == "PARTIAL"
    assert collector.seed_dispersion["fold_0"]["status"] == "PARTIAL"
    assert any(level == "warning" and message.startswith("fold_0: training status PARTIAL; seeds not DONE: ['seed2']") for level, message in logger.messages)


def test_collector_names_a_metric_that_is_non_finite_in_every_seed(tmp_path):
    names     = ["fold_0/seed1", "fold_0/seed2"]
    inference = {
        "fold_0/seed1": {"test": {"curve_rmse_gt": 2.0, "overall_r2_gt": float("nan")}},
        "fold_0/seed2": {"test": {"curve_rmse_gt": 4.0, "overall_r2_gt": float("nan")}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=logger)

    _, by_split = collector.collect_by_split()

    assert "overall_r2_gt" not in by_split["test"][0].metrics
    assert ("warning", "fold_0 test: ['overall_r2_gt'] non-finite in every seed, dropped from the fold mean") in logger.messages


def test_collector_stays_quiet_when_every_seed_contributed(tmp_path):
    import torch

    names     = ["fold_0/seed1", "fold_0/seed2"]
    inference = {name: {"test": {"curve_rmse_gt": 2.0}} for name in names}

    run_dir = build_run_dir(tmp_path, names, inference)
    for name in names:
        torch.save({"best_val_loss": 0.1, "best_epoch": 2, "epoch": 3}, run_dir / "folds" / name / "best_model.pt")

    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=logger)

    collector.collect_by_split()

    coverage = [message for level, message in logger.messages if level == "warning" and ("cover" in message or "fewer seeds" in message or "without inference" in message)]

    assert coverage == []
    assert collector.seed_dispersion["fold_0"]["status"] == "DONE"


def test_collector_names_a_metric_non_finite_in_every_fold_once_per_split(tmp_path):
    names     = ["fold_0", "fold_1"]
    inference = {
        "fold_0": {"test": {"curve_rmse_gt": 2.0, "overall_r2_gt": float("nan"), "curve_mae_gt": float("nan")}},
        "fold_1": {"test": {"curve_rmse_gt": 1.0, "overall_r2_gt": float("nan"), "curve_mae_gt": 0.1}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=logger)

    collector.collect_by_split()

    omitted = [message for level, message in logger.messages if level == "warning" and message.startswith("split test:")]

    assert omitted == ["split test: ['overall_r2_gt'] non-finite in every fold, omitted from the cross-fold aggregate (cv_aggregate_report.md, cv_summary.json)"]


@pytest.mark.dataset
def test_training_worker_refuses_a_finished_unit_when_resume_is_off(test_data_dir, tmp_path, monkeypatch):
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    config.resume             = False
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    unit = worker.unit_directory(worker.fold_run_name(1, 4))
    unit.mkdir(parents=True)
    CompletionMarker.stamp(unit, {"stage": "training"})

    trained = []
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr, run_name: trained.append(run_name), raising=True)

    with pytest.raises(SystemExit, match="already holds a completed run and resume is off"):
        worker.run(1, seed=4)

    assert trained == []
    assert CompletionMarker.is_complete(unit)


def test_collector_keeps_the_training_result_of_a_unit_in_a_renamed_directory(tmp_path):
    run_dir = build_run_dir(tmp_path, ["fold_0/seed1", "fold_1/seed1"], {})

    results = FileIO.load_json(run_dir / "pipeline" / "training_results.json")
    results[0]["name"] = "unet_fold_0/seed1"
    FileIO.save_json(results, run_dir / "pipeline" / "training_results.json")

    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=[], logger=logger)

    base, _ = collector.collect_by_split()

    assert base[0].training_result                       == {"status": "DONE", "duration_s": 10.0}
    assert collector.seed_dispersion["fold_0"]["status"] == "DONE"
    assert ("info", "1 unit result(s) matched to the renamed run directory holding their worker log: ['unet_fold_0/seed1 -> fold_0/seed1']") in logger.messages
