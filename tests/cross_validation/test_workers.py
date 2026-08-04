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
from tools.data.io           import FileIO
from tools.data.regions      import SplitRegions
from tools.monitoring.logger import Logger


def make_logger(tmp_path: Path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="workers_test")


def worker_config(test_data_dir: Path, training_type: str = "backbone") -> CrossValidationConfig:
    config                       = CrossValidationConfig(training_type=training_type)
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / "params_k5_lam0.01_sig4_sigma" / "parameters.npy"
    config.folds                 = FoldConfig(n_folds=5, azimuth_start=1000, azimuth_end=2000)
    return config


def build_run_dir(tmp_path: Path, fold_names: list[str], inference_folds: dict[str, dict]) -> Path:
    run_dir = tmp_path / "run"

    (run_dir / "pipeline").mkdir(parents=True)
    for name in fold_names:
        (run_dir / "folds" / name).mkdir(parents=True)

    training_results = [{"name": name, "status": "DONE", "duration_s": 10.0} for name in fold_names]
    FileIO.save_json(training_results, run_dir / "pipeline" / "training_results.json")

    for name, split_metrics in inference_folds.items():
        for split, metrics in split_metrics.items():
            inference_dir = run_dir / "folds" / name / "inference" / split
            inference_dir.mkdir(parents=True)
            FileIO.save_json(metrics, inference_dir / "metrics.json")

    return run_dir


def test_collector_points_at_folds_subdir(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_logger(tmp_path))

    assert collector.training_dir == run_dir / "folds"
    assert collector.splits       == ["test"]


def test_collector_collects_every_fold(tmp_path):
    names     = [FoldNaming.name(i) for i in range(5)]
    run_dir   = build_run_dir(tmp_path, names, {})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_logger(tmp_path))

    base, by_split = collector.collect_by_split()

    assert [record.name for record in base] == names
    assert set(by_split)                    == {"test"}
    assert len(by_split["test"])            == 5


def test_collector_attaches_training_results(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    collector = FoldCollector(run_dir=run_dir, splits=[], logger=make_logger(tmp_path))

    base, _ = collector.collect_by_split()

    assert base[0].training_result["status"]     == "DONE"
    assert base[0].training_result["duration_s"] == 10.0


def test_collector_split_view_loads_metrics_when_present(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {"fold_0": {"test": {"curve_rmse_gt": 2.5}}})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_logger(tmp_path))

    _, by_split = collector.collect_by_split()
    records     = by_split["test"]

    assert records[0].metrics            == {"curve_rmse_gt": 2.5}
    assert records[0].inference_dir      is not None
    assert records[0].inference_dir.name == "test"


def test_collector_split_view_empty_when_metrics_absent(tmp_path):
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {"fold_0": {"test": {"curve_rmse_gt": 2.5}}})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_logger(tmp_path))

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
    collector = FoldCollector(run_dir=run_dir, splits=["test"], logger=make_logger(tmp_path))

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
    collector = FoldCollector(run_dir=run_dir, splits=["val", "test"], logger=make_logger(tmp_path))

    _, by_split = collector.collect_by_split()

    assert by_split["val"][0].metrics  == {"curve_rmse_gt": 1.0}
    assert by_split["test"][0].metrics == {"curve_rmse_gt": 2.0}


@pytest.mark.real_data
def test_cross_validation_worker_fold_name(test_data_dir):
    worker = CrossValidationWorker(worker_config(test_data_dir), run_tag="rt")

    assert worker.fold_name(0) == "fold_0"
    assert worker.fold_name(4) == "fold_4"


@pytest.mark.real_data
def test_training_worker_dispatches_to_backbone(test_data_dir, monkeypatch):
    worker = FoldTrainingWorker(worker_config(test_data_dir, "backbone"), run_tag="rt")

    captured = {}

    def fake_backbone(self, fold_index, seed, split_regions):
        captured["fold"]    = fold_index
        captured["seed"]    = seed
        captured["regions"] = split_regions

    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", fake_backbone, raising=True)

    worker.run(2)

    assert captured["fold"] == 2
    assert captured["seed"] is None
    assert isinstance(captured["regions"], SplitRegions)


@pytest.mark.real_data
def test_training_worker_passes_planned_split_regions(test_data_dir, monkeypatch):
    worker = FoldTrainingWorker(worker_config(test_data_dir, "backbone"), run_tag="rt")

    captured = {}
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr: captured.update(sr=sr), raising=True)

    worker.run(2)

    train = sorted((r.azimuth_start, r.azimuth_end) for r in captured["sr"].regions("train"))
    assert train == [(1000, 1368), (1832, 2000)]
    assert worker.factory.planner().plan(2).split_regions.regions("test")[0].azimuth_start == 1432


@pytest.mark.real_data
def test_training_worker_dispatch_routes_by_type(test_data_dir, monkeypatch):
    seen = {}

    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone",            lambda self, i, seed, sr: seen.update(kind="backbone"), raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_jepa",               lambda self, i, seed, sr: seen.update(kind="jepa"),     raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_profile_autoencoder", lambda self, i, seed, sr: seen.update(kind="ae"),       raising=True)

    FoldTrainingWorker(worker_config(test_data_dir, "backbone"), run_tag="rt").run(0)
    assert seen["kind"] == "backbone"

    FoldTrainingWorker(worker_config(test_data_dir, "jepa"), run_tag="rt").run(0)
    assert seen["kind"] == "jepa"

    FoldTrainingWorker(worker_config(test_data_dir, "profile_autoencoder"), run_tag="rt").run(0)
    assert seen["kind"] == "ae"


@pytest.mark.real_data
def test_training_worker_rejects_unknown_type(test_data_dir):
    config               = worker_config(test_data_dir)
    config.training_type = "bogus"

    worker = FoldTrainingWorker(config, run_tag="rt")

    with pytest.raises(ValueError, match="Unknown training_type"):
        worker.run(0)


@pytest.mark.real_data
def test_inference_worker_builds_run_directory(test_data_dir, monkeypatch):
    worker = FoldInferenceWorker(worker_config(test_data_dir), run_tag="rt")

    captured = {}

    class FakePipeline:
        def __init__(self, inference_config, components=None):
            captured["config"]     = inference_config
            captured["components"] = components

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(pipeline_module, "InferencePipeline", FakePipeline, raising=True)

    worker.run(1, "test")

    expected_dir = worker.run_dir / "folds" / worker.fold_run_name(1, None)
    assert captured["ran"]                  is True
    assert captured["config"].run_directory == expected_dir
    assert expected_dir.name.endswith("_fold_1")
    assert expected_dir.name.startswith(f"{worker.config.backbone_name}-{worker.config.backbone_head}-")
    assert "-K_" in expected_dir.name
    assert captured["config"].split         == "test"
    assert captured["config"].output_subdir == "test"


@pytest.mark.real_data
def test_backbone_fold_trainer_uses_the_entry_curriculum(test_data_dir, monkeypatch):
    config = worker_config(test_data_dir, "backbone")
    worker = FoldTrainingWorker(config, run_tag="rt")

    captured = {}

    class _FakePipeline:
        def __init__(self, trainer_config, dataset_config, backbone_name, model_config, seed, run_name):
            captured["trainer_config"] = trainer_config
            captured["run_name"]       = run_name

        def run(self, probe_config=None):
            pass

    monkeypatch.setattr(backbone_pipeline_module, "TrainingPipeline", _FakePipeline, raising=True)

    worker.run(1, seed=7)

    trainer_config = captured["trainer_config"]
    assert trainer_config.curriculum is config.curriculum
    assert trainer_config.curriculum.enabled is True
    assert trainer_config.curriculum.complete.use_covariance_match is True
    assert trainer_config.geometry.baselines_origin == str(config.geometry.baselines_file(test_data_dir))
    from pipelines.shared.training.run_naming import RunNaming

    run_tag = RunNaming.training_tag(config.backbone_name, config.backbone_head, config.curriculum, 5, config.augmentation)
    assert captured["run_name"] == f"{run_tag}_fold_1/seed7"
