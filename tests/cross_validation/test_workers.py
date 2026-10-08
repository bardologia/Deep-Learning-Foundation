"""Tests for the cross-validation fold workers and the fold result collector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pipelines.backbone.inference.pipeline as pipeline_module
import pipelines.backbone.training.pipeline  as backbone_pipeline_module
from configuration.cross_validation     import CrossValidationConfig, FoldConfig
from configuration.dataset              import InputConfig, ProfileAugmentationConfig
from configuration.training.jepa        import JepaCurriculumConfig
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


def worker_config(test_data_dir: Path, training_type: str = "backbone") -> CrossValidationConfig:
    """Returns a cross-validation config bound to the real dataset and parameter run."""
    config                       = CrossValidationConfig(training_type=training_type)
    config.paths.dataset_path    = test_data_dir
    config.paths.parameters_path = test_data_dir / "params" / DEFAULT_PARAMS_RUN / "parameters.npy"
    config.folds                 = FoldConfig(n_folds=5, row_start=1000, row_end=1512)
    return config


def jepa_worker_config(test_data_dir: Path, tmp_path: Path) -> CrossValidationConfig:
    """Returns a JEPA cross-validation config coupled to a stand-in profile-autoencoder run.

    Args:
        test_data_dir : Root of the real preprocessed dataset.
        tmp_path      : Directory receiving the stand-in autoencoder run and the fold outputs.

    Returns:
        The configuration, carrying the conv head a coupled profile encoder requires.
    """
    from configuration.architectures                import GruAutoencoderConfig
    from pipelines.shared.config.config_persistence import ProfileAutoencoderConfigIO

    ProfileAutoencoderConfigIO.save(GruAutoencoderConfig(), "gru_ae", tmp_path / "gru_run" / "meta")

    config                                 = worker_config(test_data_dir, "jepa")
    config.backbone_head                   = "conv"
    config.paths.log_base_dir              = tmp_path
    config.jepa.profile_autoencoder_logdir = tmp_path
    config.jepa.profile_autoencoder_run    = "gru_run"

    return config


def stamp_split_inference(base: Path, split: str, metrics: dict) -> Path:
    """Creates a completion-stamped inference directory holding the given metrics.

    Args:
        base    : Inference base directory of the fold.
        split   : Split name recorded in the completion payload.
        metrics : Metric payload written to metrics.json.

    Returns:
        The stamped directory.
    """
    stamp = base / f"stamp_{split}"
    stamp.mkdir(parents=True)

    CompletionMarker.stamp(stamp, {"stage": "inference", "split": split})
    FileIO.save_json(metrics, stamp / "metrics.json")

    return stamp


def build_run_dir(tmp_path: Path, fold_names: list[str], inference_folds: dict[str, dict]) -> Path:
    """Builds a cross-validation run directory with fold folders and stamped inference.

    Args:
        tmp_path        : Parent directory of the synthetic run.
        fold_names      : Fold directory names created under folds/.
        inference_folds : Per-fold mapping of split name to metric payload.

    Returns:
        The run directory.
    """
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
    """Verifies the collector reads training runs from the folds subdirectory."""
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    assert collector.training_dir == run_dir / "folds"
    assert collector.splits       == ["test"]


def test_collector_collects_every_fold(tmp_path):
    """Verifies every fold yields a base record and a split record."""
    names     = [FoldNaming.name(i) for i in range(5)]
    run_dir   = build_run_dir(tmp_path, names, {})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    base, by_split = collector.collect_by_split()

    assert [record.name for record in base] == names
    assert set(by_split)                    == {"test"}
    assert len(by_split["test"])            == 5


def test_collector_attaches_training_results(tmp_path):
    """Verifies the training result payload is attached to the base records."""
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    collector = FoldCollector(run_dir=run_dir, splits=[], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    base, _ = collector.collect_by_split()

    assert base[0].training_result["status"]     == "DONE"
    assert base[0].training_result["duration_s"] == 10.0


def test_collector_split_view_loads_metrics_when_present(tmp_path):
    """Verifies a stamped split directory supplies the metrics and inference directory."""
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {"fold_0": {"test": {"curve_rmse_gt": 2.5}}})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    _, by_split = collector.collect_by_split()
    records     = by_split["test"]

    assert records[0].metrics       == {"curve_rmse_gt": 2.5}
    assert records[0].inference_dir is not None
    assert CompletionMarker.payload(records[0].inference_dir)["split"] == "test"


def test_collector_reads_autoencoder_metrics_under_the_family_subdir(tmp_path):
    """Verifies autoencoder metrics are read from the family inference subdirectory."""
    run_dir = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})

    for name in ("fold_0", "fold_1"):
        stamp_split_inference(run_dir / "folds" / name / "inference" / "profile_ae", "test", {"mse_mean": 0.5})

    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="profile_autoencoder", logger=make_file_logger(tmp_path, "workers_test"))

    _, by_split = collector.collect_by_split()

    assert by_split["test"][0].metrics                   == {"mse_mean": 0.5}
    assert by_split["test"][0].inference_dir.parent.name == "profile_ae"


def test_collector_split_view_empty_when_metrics_absent(tmp_path):
    """Verifies a fold without stamped inference yields empty metrics and no figures."""
    run_dir   = build_run_dir(tmp_path, ["fold_0", "fold_1"], {"fold_0": {"test": {"curve_rmse_gt": 2.5}}})
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    _, by_split = collector.collect_by_split()
    records     = by_split["test"]

    assert records[1].metrics       == {}
    assert records[1].inference_dir is None
    assert records[1].figures       == []


def test_collector_aggregates_seeds_per_fold(tmp_path):
    """Verifies per-seed fold runs are averaged into one record with a seed dispersion."""
    names     = ["fold_0/seed1", "fold_0/seed2", "fold_1/seed1", "fold_1/seed2"]
    inference = {
        "fold_0/seed1": {"test": {"curve_rmse_gt": 2.0}},
        "fold_0/seed2": {"test": {"curve_rmse_gt": 4.0}},
        "fold_1/seed1": {"test": {"curve_rmse_gt": 1.0}},
        "fold_1/seed2": {"test": {"curve_rmse_gt": 3.0}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    base, by_split = collector.collect_by_split()

    assert [record.name for record in base]             == ["fold_0", "fold_1"]
    assert by_split["test"][0].metrics["curve_rmse_gt"] == 3.0
    assert by_split["test"][1].metrics["curve_rmse_gt"] == 2.0

    dispersion = collector.seed_dispersion
    assert dispersion["fold_0"]["n_seeds"] == 2
    assert dispersion["fold_0"]["splits"]["test"]["curve_rmse_gt"] == pytest.approx(float(np.std([2.0, 4.0], ddof=1)))


def test_collector_multiple_splits_independent(tmp_path):
    """Verifies each split keeps its own metrics for the same fold."""
    run_dir = build_run_dir(
        tmp_path,
        ["fold_0", "fold_1"],
        {"fold_0": {"val": {"curve_rmse_gt": 1.0}, "test": {"curve_rmse_gt": 2.0}}},
    )
    collector = FoldCollector(run_dir=run_dir, splits=["val", "test"], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    _, by_split = collector.collect_by_split()

    assert by_split["val"][0].metrics  == {"curve_rmse_gt": 1.0}
    assert by_split["test"][0].metrics == {"curve_rmse_gt": 2.0}


@pytest.mark.dataset
def test_cross_validation_worker_fold_name(test_data_dir):
    """Verifies the worker derives the plain fold name from the fold index."""
    worker = CrossValidationWorker(worker_config(test_data_dir), run_tag="rt")

    assert worker.fold_name(0) == "fold_0"
    assert worker.fold_name(4) == "fold_4"


@pytest.mark.dataset
def test_training_worker_dispatches_to_backbone(test_data_dir, tmp_path, monkeypatch):
    """Verifies a backbone run dispatches to the backbone trainer with the planned split regions and the composed run name."""
    config                    = worker_config(test_data_dir, "backbone")
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    captured = {}

    def fake_backbone(self, fold_index, seed, split_regions, run_name):
        """Records the fold index, seed, split regions and run name passed to the backbone run."""
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
    """Verifies the worker records the fold, seed and scored regions in the unit directory before the pipeline runs."""
    from pipelines.cross_validation.folds             import FoldUnitManifest
    from pipelines.shared.orchestration.unit_manifest import UnitManifest

    config                    = worker_config(test_data_dir, "backbone")
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
    """Verifies a unit that already completed is not trained again, which a repeated dispatch relies on."""
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
    """Verifies an explicit run name replaces the composed one, so a unit found by its manifest keeps its directory."""
    config                    = worker_config(test_data_dir, "backbone")
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    seen = {}
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr, run_name: seen.update(run_name=run_name), raising=True)

    worker.run(1, seed=None, run_name="old-label_fold_1")

    assert seen["run_name"] == "old-label_fold_1"
    assert (worker.unit_directory("old-label_fold_1") / "meta" / "unit.json").is_file()


@pytest.mark.dataset
def test_training_worker_passes_planned_split_regions(test_data_dir, tmp_path, monkeypatch):
    """Verifies the split regions handed to training are the guarded regions of the fold plan."""
    config                    = worker_config(test_data_dir, "backbone")
    config.paths.log_base_dir = tmp_path
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    captured = {}
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr, run_name: captured.update(sr=sr), raising=True)

    worker.run(2)

    train = sorted((r.row_start, r.row_end) for r in captured["sr"].regions("train"))
    assert train == [(1000, 1196), (1416, 1512)]
    assert worker.factory.planner().plan(2).split_regions.regions("test")[0].row_start == 1212


@pytest.mark.dataset
def test_training_worker_dispatch_routes_by_type(test_data_dir, tmp_path, monkeypatch):
    """Verifies each training type routes to its own training method."""
    from configuration.architectures                import GruAutoencoderConfig
    from pipelines.shared.config.config_persistence import ProfileAutoencoderConfigIO

    seen = {}

    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone",            lambda self, i, seed, sr, run_name: seen.update(kind="backbone"),  raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_dual",                lambda self, i, seed, sr, run_name: seen.update(kind="dual"),      raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_jepa",                lambda self, i, seed, sr, run_name: seen.update(kind="jepa"),      raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_profile_autoencoder", lambda self, i, seed, sr, run_name: seen.update(kind="ae"),        raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_image_autoencoder",   lambda self, i, seed, sr, run_name: seen.update(kind="image_ae"),  raising=True)

    def rooted(training_type: str) -> CrossValidationConfig:
        """Returns a worker config of the family writing under the temporary directory."""
        config                    = worker_config(test_data_dir, training_type)
        config.paths.log_base_dir = tmp_path
        return config

    FoldTrainingWorker(rooted("backbone"), run_tag="rt").run(0)
    assert seen["kind"] == "backbone"

    FoldTrainingWorker(rooted("dual"), run_tag="rt").run(0)
    assert seen["kind"] == "dual"

    ProfileAutoencoderConfigIO.save(GruAutoencoderConfig(), "gru_ae", tmp_path / "gru_run" / "meta")
    jepa_config                                 = rooted("jepa")
    jepa_config.jepa.profile_autoencoder_logdir = tmp_path
    jepa_config.jepa.profile_autoencoder_run    = "gru_run"

    FoldTrainingWorker(jepa_config, run_tag="rt").run(0)
    assert seen["kind"] == "jepa"

    FoldTrainingWorker(rooted("profile_autoencoder"), run_tag="rt").run(0)
    assert seen["kind"] == "ae"

    FoldTrainingWorker(rooted("image_autoencoder"), run_tag="rt").run(0)
    assert seen["kind"] == "image_ae"


@pytest.mark.dataset
def test_training_worker_rejects_unknown_type(test_data_dir):
    """Verifies an unrecognised training type is rejected."""
    config               = worker_config(test_data_dir)
    config.training_type = "bogus"

    with pytest.raises(ValueError, match="Unknown training_type"):
        FoldTrainingWorker(config, run_tag="rt").run(0)


@pytest.mark.dataset
def test_dual_fold_run_name_carries_trunks_and_routing(test_data_dir):
    """Verifies a dual fold run name carries both trunks, the routing tag and the fold index."""
    worker = FoldTrainingWorker(worker_config(test_data_dir, "dual"), run_tag="rt")

    name = worker.fold_run_name(1, None)

    assert name.startswith("dual_unet_skip-set_pred-")
    assert "-full.full-" in name
    assert name.endswith("_fold_1")


@pytest.mark.dataset
def test_dual_model_config_binds_both_trunks(test_data_dir):
    """Verifies the dual model config binds the parameter and existence trunks and inputs."""
    config                          = worker_config(test_data_dir, "dual")
    config.dual.existence_backbone  = "unet"
    config.dual.existence_input     = ["derived"]

    model_config = FoldTrainingWorker(config, run_tag="rt")._dual_model_config()

    assert model_config.params_backbone    == "unet_skip"
    assert model_config.existence_backbone == "unet"
    assert model_config.existence_input    == ("derived",)


@pytest.mark.dataset
def test_dual_model_config_rejects_trunk_keys_in_model_overrides(test_data_dir):
    """Verifies trunk keys passed through model_overrides are rejected."""
    config                  = worker_config(test_data_dir, "dual")
    config.model_overrides  = {"params_backbone": "unet"}

    with pytest.raises(ValueError, match="dedicated dual"):
        FoldTrainingWorker(config, run_tag="rt")._dual_model_config()


@pytest.mark.dataset
def test_ae_fold_entry_carries_the_cross_validated_profile_augmentation(test_data_dir):
    """Verifies the profile-augmentation recipe of the sweep reaches the fold entry config."""
    config                                  = worker_config(test_data_dir, "profile_autoencoder")
    config.autoencoder.profile_augmentation = ProfileAugmentationConfig(p_flip=0.25, p_shift=0.75, max_shift=7, p_amp_scale=0.4, p_noise=0.6, noise_std=0.02)

    entry = FoldTrainingWorker(config, run_tag="rt")._ae_entry_config("fold_0", 3)

    assert entry.profile_augmentation is config.autoencoder.profile_augmentation


@pytest.mark.dataset
def test_ae_fold_pipeline_samples_with_the_cross_validated_profile_augmentation(test_data_dir, tmp_path):
    """Verifies the forwarded augmentation reaches the profile dataset every fold trains on."""
    from pipelines.profile_autoencoder.training.pipeline import TrainingPipeline

    config                                  = worker_config(test_data_dir, "profile_autoencoder")
    config.paths.log_base_dir               = tmp_path
    config.autoencoder.profile_augmentation = ProfileAugmentationConfig(p_flip=0.25, p_shift=0.75, p_noise=0.6, noise_std=0.02)

    worker   = FoldTrainingWorker(config, run_tag="rt")
    entry    = worker._ae_entry_config(worker.fold_run_name(0, None), None)
    pipeline = TrainingPipeline(entry, split_regions=worker.factory.planner().plan(0).split_regions)

    assert pipeline.dataset_config.augmentation.p_flip    == 0.25
    assert pipeline.dataset_config.augmentation.p_shift   == 0.75
    assert pipeline.dataset_config.augmentation.p_noise   == 0.6
    assert pipeline.dataset_config.augmentation.noise_std == 0.02


@pytest.mark.dataset
def test_jepa_fold_entry_carries_the_cross_validated_input_channels(test_data_dir, tmp_path):
    """Verifies the channel set the sweep records reaches the JEPA fold entry config."""
    config       = jepa_worker_config(test_data_dir, tmp_path)
    config.input = InputConfig(use_primary=True, use_secondaries=False, use_derived=False, use_aux=True)

    entry = FoldTrainingWorker(config, run_tag="rt")._jepa_entry_config("fold_0", 3)

    assert entry.input is config.input


@pytest.mark.dataset
def test_jepa_fold_entry_carries_the_cross_validated_curriculum(test_data_dir, tmp_path):
    """Verifies the JEPA loss curriculum of the sweep reaches the fold entry config."""
    config                 = jepa_worker_config(test_data_dir, tmp_path)
    config.jepa.curriculum = JepaCurriculumConfig(enabled=True, swap_epoch=4, reset_lr=True, reset_warmup=True, reset_optimizer=True)

    entry = FoldTrainingWorker(config, run_tag="rt")._jepa_entry_config("fold_0", 3)

    assert entry.curriculum is config.jepa.curriculum


@pytest.mark.dataset
def test_jepa_fold_pipeline_trains_on_the_cross_validated_channels_and_curriculum(test_data_dir, tmp_path):
    """Verifies both forwarded settings reach the dataset and trainer configs of the fold."""
    from pipelines.jepa.training.pipeline import TrainingPipeline

    config                 = jepa_worker_config(test_data_dir, tmp_path)
    config.input           = InputConfig(use_primary=True, use_secondaries=False, use_derived=False, use_aux=True)
    config.jepa.curriculum = JepaCurriculumConfig(enabled=True, swap_epoch=4, reset_lr=True)

    worker   = FoldTrainingWorker(config, run_tag="rt")
    entry    = worker._jepa_entry_config(worker.fold_run_name(0, None), None)
    pipeline = TrainingPipeline(entry, split_regions=worker.factory.planner().plan(0).split_regions)

    assert pipeline.dataset_config.input_config.use_aux            is True
    assert pipeline.dataset_config.input_config.use_derived is False
    assert pipeline.dataset_config.input_config.use_secondaries    is False
    assert pipeline.trainer_config.curriculum.enabled              is True
    assert pipeline.trainer_config.curriculum.swap_epoch           == 4
    assert pipeline.trainer_config.curriculum.reset_lr             is True


@pytest.mark.dataset
def test_inference_worker_builds_run_directory(test_data_dir, tmp_path, monkeypatch):
    """Verifies the inference worker points the pipeline at the fold run directory and split."""
    config                       = worker_config(test_data_dir)
    config.paths.log_base_dir    = tmp_path
    config.inference_splits      = ["test"]
    worker                       = FoldInferenceWorker(config, run_tag="rt")

    captured = {}

    class FakePipeline:
        """Inference pipeline stub recording its configuration and run call."""
        def __init__(self, inference_config, components=None):
            """Records the inference configuration and component overrides."""
            captured["config"]     = inference_config
            captured["components"] = components

        def run(self):
            """Records that the pipeline ran."""
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
    """Verifies inference runs once per configured split in order."""
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    config.inference_splits   = ["val", "test"]
    worker                    = FoldInferenceWorker(config, run_tag="rt")

    ran = []

    class FakePipeline:
        """Inference pipeline stub recording the split it was configured with."""
        def __init__(self, inference_config, components=None):
            """Stores the split of the inference configuration."""
            self.split = inference_config.split

        def run(self):
            """Records the split as having run."""
            ran.append(self.split)

    monkeypatch.setattr(pipeline_module, "InferencePipeline", FakePipeline, raising=True)

    worker.run(1)

    assert ran == ["val", "test"]


@pytest.mark.dataset
def test_inference_worker_skips_completed_split_on_resume(test_data_dir, tmp_path, monkeypatch):
    """Verifies a split already stamped complete is skipped when resuming."""
    config                    = worker_config(test_data_dir)
    config.paths.log_base_dir = tmp_path
    config.inference_splits   = ["val", "test"]
    worker                    = FoldInferenceWorker(config, run_tag="rt")

    done = worker.run_dir / "folds" / worker.fold_run_name(1, None) / "inference" / "stamp_val"
    done.mkdir(parents=True)
    CompletionMarker.stamp(done, {"stage": "inference", "split": "val"})

    ran = []

    class FakePipeline:
        """Inference pipeline stub recording the split it was configured with."""
        def __init__(self, inference_config, components=None):
            """Stores the split of the inference configuration."""
            self.split = inference_config.split

        def run(self):
            """Records the split as having run."""
            ran.append(self.split)

    monkeypatch.setattr(pipeline_module, "InferencePipeline", FakePipeline, raising=True)

    worker.run(1)

    assert ran == ["test"]
    assert done.exists()


@pytest.mark.dataset
def test_inference_worker_routes_autoencoders_to_their_pipelines(test_data_dir, tmp_path, monkeypatch):
    """Verifies autoencoder folds route to their own inference pipelines."""
    import pipelines.image_autoencoder.inference.pipeline   as image_ae_module
    import pipelines.profile_autoencoder.inference.pipeline as profile_ae_module

    seen = {}

    def fake(kind, module, attribute):
        """Installs a pipeline stub in place of the given module attribute.

        Args:
            kind      : Label recorded when the stub is constructed.
            module    : Module whose attribute is replaced.
            attribute : Name of the pipeline class to replace.
        """
        class FakePipeline:
            """Inference pipeline stub recording its kind, configuration and run call."""
            def __init__(self, config):
                """Records the kind label and the inference configuration."""
                seen["kind"]   = kind
                seen["config"] = config

            def run(self):
                """Records that the pipeline ran."""
                seen["ran"] = True

        monkeypatch.setattr(module, attribute, FakePipeline, raising=True)

    fake("profile_ae", profile_ae_module, "ProfileAeInferencePipeline")
    fake("image_ae",   image_ae_module,   "ImageAeInferencePipeline")

    profile_config                    = worker_config(test_data_dir, "profile_autoencoder")
    profile_config.paths.log_base_dir = tmp_path
    profile_config.inference_splits   = ["val"]

    FoldInferenceWorker(profile_config, run_tag="rt").run(1)
    assert seen["kind"]                 == "profile_ae"
    assert seen["config"].split         == "val"
    assert seen["config"].output_subdir is None
    assert seen["ran"]                  is True

    image_config                    = worker_config(test_data_dir, "image_autoencoder")
    image_config.paths.log_base_dir = tmp_path
    image_config.inference_splits   = ["test"]

    FoldInferenceWorker(image_config, run_tag="rt").run(1)
    assert seen["kind"]                 == "image_ae"
    assert seen["config"].split         == "test"
    assert seen["config"].output_subdir is None


@pytest.mark.dataset
def test_backbone_fold_trainer_uses_the_entry_curriculum(test_data_dir, tmp_path, monkeypatch):
    """Verifies the backbone fold trainer inherits the entry curriculum, geometry origin, run name and prior-bias flag."""
    config                    = worker_config(test_data_dir, "backbone")
    config.paths.log_base_dir = tmp_path
    config.prior_bias_init    = True
    worker                    = FoldTrainingWorker(config, run_tag="rt")

    captured = {}

    class _FakePipeline:
        """Backbone training pipeline stub recording its constructor arguments."""
        def __init__(self, trainer_config, dataset_config, backbone_name, model_config, seed, run_name, overfit_check, prior_bias_init):
            """Records the trainer and dataset configs, the run name, the overfit check and the prior-bias flag."""
            captured["trainer_config"]  = trainer_config
            captured["dataset_config"]  = dataset_config
            captured["run_name"]        = run_name
            captured["overfit_check"]   = overfit_check
            captured["prior_bias_init"] = prior_bias_init

        def run(self, probe_config=None):
            """Accepts an optional probe config and does nothing."""
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
    """Verifies missing training results are rebuilt from the per-fold completion markers."""
    run_dir = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})
    (run_dir / "pipeline" / "training_results.json").unlink()

    CompletionMarker.stamp(run_dir / "folds" / "fold_0", {"stage": "training", "epochs_completed": 2})

    collector = FoldCollector(run_dir=run_dir, splits=[], training_type="backbone", logger=make_file_logger(tmp_path, "workers_test"))

    base, _ = collector.collect_by_split()

    assert base[0].training_result["status"] == "DONE"
    assert base[1].training_result["status"] == "PARTIAL"


def test_collector_warns_when_a_seed_has_no_metrics_and_records_the_coverage(tmp_path):
    """A seed without a scored split is named in a warning, and the fold's dispersion records how many seeds the mean used."""
    names     = ["fold_0/seed1", "fold_0/seed2", "fold_1/seed1", "fold_1/seed2"]
    inference = {
        "fold_0/seed1": {"test": {"curve_rmse_gt": 2.0}},
        "fold_1/seed1": {"test": {"curve_rmse_gt": 1.0}},
        "fold_1/seed2": {"test": {"curve_rmse_gt": 3.0}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=logger)

    _, by_split = collector.collect_by_split()

    assert by_split["test"][0].metrics["curve_rmse_gt"] == 2.0
    assert ("warning", "fold_0 test: metrics averaged over 1/2 seeds; missing ['seed2'] (no completed inference with metrics)") in logger.messages
    assert collector.seed_dispersion["fold_0"]["n_used"]        == {"test": 1}
    assert collector.seed_dispersion["fold_0"]["missing_seeds"] == {"test": ["seed2"]}
    assert collector.seed_dispersion["fold_1"]["n_used"]        == {"test": 2}
    assert collector.seed_dispersion["fold_1"]["missing_seeds"] == {"test": []}
    assert any(level == "warning" and message.startswith("1 of 2 folds hold a seed that did not finish or average fewer seeds") for level, message in logger.messages)


def test_collector_marks_a_fold_partial_and_names_the_seed_that_did_not_finish(tmp_path):
    """A seed whose training did not end DONE turns its fold PARTIAL in the dispersion and is named in a warning."""
    names   = ["fold_0/seed1", "fold_0/seed2"]
    run_dir = build_run_dir(tmp_path, names, {})

    results = FileIO.load_json(run_dir / "pipeline" / "training_results.json")
    results[1]["status"] = "FAILED"
    FileIO.save_json(results, run_dir / "pipeline" / "training_results.json")

    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=[], training_type="backbone", logger=logger)

    base, _ = collector.collect_by_split()

    assert base[0].training_result["status"]           == "PARTIAL"
    assert collector.seed_dispersion["fold_0"]["status"] == "PARTIAL"
    assert any(level == "warning" and message.startswith("fold_0: training status PARTIAL; seeds not DONE: ['seed2']") for level, message in logger.messages)


def test_collector_names_a_metric_that_is_non_finite_in_every_seed(tmp_path):
    """A metric no seed of a fold reports as a finite value is dropped from the fold mean with a warning naming it."""
    names     = ["fold_0/seed1", "fold_0/seed2"]
    inference = {
        "fold_0/seed1": {"test": {"curve_rmse_gt": 2.0, "overall_r2_gt": float("nan")}},
        "fold_0/seed2": {"test": {"curve_rmse_gt": 4.0, "overall_r2_gt": float("nan")}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=logger)

    _, by_split = collector.collect_by_split()

    assert "overall_r2_gt" not in by_split["test"][0].metrics
    assert ("warning", "fold_0 test: ['overall_r2_gt'] non-finite in every seed, dropped from the fold mean") in logger.messages


def test_collector_stays_quiet_when_every_seed_contributed(tmp_path):
    """Folds whose seeds all finished, scored and left a checkpoint produce no coverage warning."""
    import torch

    names     = ["fold_0/seed1", "fold_0/seed2"]
    inference = {name: {"test": {"curve_rmse_gt": 2.0}} for name in names}

    run_dir = build_run_dir(tmp_path, names, inference)
    for name in names:
        torch.save({"best_val_loss": 0.1, "best_epoch": 2, "epoch": 3}, run_dir / "folds" / name / "best_model.pt")

    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=logger)

    collector.collect_by_split()

    coverage = [message for level, message in logger.messages if level == "warning" and ("cover" in message or "fewer seeds" in message or "without inference" in message)]

    assert coverage == []
    assert collector.seed_dispersion["fold_0"]["status"] == "DONE"


def test_collector_names_a_metric_non_finite_in_every_fold_once_per_split(tmp_path):
    """A metric that every fold drops as non-finite is named once for its split as omitted from the cross-fold aggregate; one finite in some fold is not."""
    names     = ["fold_0", "fold_1"]
    inference = {
        "fold_0": {"test": {"curve_rmse_gt": 2.0, "overall_r2_gt": float("nan"), "curve_mae_gt": float("nan")}},
        "fold_1": {"test": {"curve_rmse_gt": 1.0, "overall_r2_gt": float("nan"), "curve_mae_gt": 0.1}},
    }

    run_dir   = build_run_dir(tmp_path, names, inference)
    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["test"], training_type="backbone", logger=logger)

    collector.collect_by_split()

    omitted = [message for level, message in logger.messages if level == "warning" and message.startswith("split test:")]

    assert omitted == ["split test: ['overall_r2_gt'] non-finite in every fold, omitted from the cross-fold aggregate (cv_aggregate_report.md, cv_summary.json)"]


def test_collector_sees_autoencoder_inference_under_the_family_subdir_and_counts_every_split(tmp_path):
    """An autoencoder fold run scored under its family subdirectory is collected as inferred, and each configured split gets one coverage line naming the runs it misses."""
    run_dir = build_run_dir(tmp_path, ["fold_0", "fold_1"], {})

    for name in ("fold_0", "fold_1"):
        stamp_split_inference(run_dir / "folds" / name / "inference" / "profile_ae", "val", {"mse_mean": 0.5})
    stamp_split_inference(run_dir / "folds" / "fold_0" / "inference" / "profile_ae", "test", {"mse_mean": 0.4})

    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=["val", "test"], training_type="profile_autoencoder", logger=logger)
    records   = collector.collect()

    collector.collect_by_split()

    assert all(record.has_inference for record in records)
    assert collector.scored_splits == {"fold_0": ["val", "test"], "fold_1": ["val"]}
    assert ("info", "split val: 2/2 fold runs hold a completed inference with metrics") in logger.messages
    assert ("warning", "split test: 1/2 fold runs hold a completed inference with metrics; missing: ['fold_1']") in logger.messages


@pytest.mark.dataset
def test_training_worker_refuses_a_finished_unit_when_resume_is_off(test_data_dir, tmp_path, monkeypatch):
    """With resume off a fold unit that already completed makes the worker exit instead of training over it or claiming it trained."""
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
    """A unit a resume matched to a renamed directory keeps its stored status and duration, since its result follows its worker log."""
    run_dir = build_run_dir(tmp_path, ["fold_0/seed1", "fold_1/seed1"], {})

    results = FileIO.load_json(run_dir / "pipeline" / "training_results.json")
    results[0]["name"] = "unet_fold_0/seed1"
    FileIO.save_json(results, run_dir / "pipeline" / "training_results.json")

    logger    = RecordingLogger()
    collector = FoldCollector(run_dir=run_dir, splits=[], training_type="backbone", logger=logger)

    base, _ = collector.collect_by_split()

    assert base[0].training_result                       == {"status": "DONE", "duration_s": 10.0}
    assert collector.seed_dispersion["fold_0"]["status"] == "DONE"
    assert ("info", "1 unit result(s) matched to the renamed run directory holding their worker log: ['unet_fold_0/seed1 -> fold_0/seed1']") in logger.messages
