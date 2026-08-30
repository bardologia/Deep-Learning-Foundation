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


from tests.conftest import make_file_logger


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

    training_results = [{"name": name, "status": "DONE", "duration_s": 10.0} for name in fold_names]
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
def test_training_worker_dispatches_to_backbone(test_data_dir, monkeypatch):
    """Verifies a backbone run dispatches to the backbone trainer with the planned split regions."""
    worker = FoldTrainingWorker(worker_config(test_data_dir, "backbone"), run_tag="rt")

    captured = {}

    def fake_backbone(self, fold_index, seed, split_regions):
        """Records the fold index, seed and split regions passed to the backbone run."""
        captured["fold"]    = fold_index
        captured["seed"]    = seed
        captured["regions"] = split_regions

    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", fake_backbone, raising=True)

    worker.run(2)

    assert captured["fold"] == 2
    assert captured["seed"] is None
    assert isinstance(captured["regions"], SplitRegions)


@pytest.mark.dataset
def test_training_worker_passes_planned_split_regions(test_data_dir, monkeypatch):
    """Verifies the split regions handed to training are the guarded regions of the fold plan."""
    worker = FoldTrainingWorker(worker_config(test_data_dir, "backbone"), run_tag="rt")

    captured = {}
    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone", lambda self, i, seed, sr: captured.update(sr=sr), raising=True)

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

    monkeypatch.setattr(FoldTrainingWorker, "_run_backbone",            lambda self, i, seed, sr: seen.update(kind="backbone"),  raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_dual",                lambda self, i, seed, sr: seen.update(kind="dual"),      raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_jepa",                lambda self, i, seed, sr: seen.update(kind="jepa"),      raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_profile_autoencoder", lambda self, i, seed, sr: seen.update(kind="ae"),        raising=True)
    monkeypatch.setattr(FoldTrainingWorker, "_run_image_autoencoder",   lambda self, i, seed, sr: seen.update(kind="image_ae"),  raising=True)

    FoldTrainingWorker(worker_config(test_data_dir, "backbone"), run_tag="rt").run(0)
    assert seen["kind"] == "backbone"

    FoldTrainingWorker(worker_config(test_data_dir, "dual"), run_tag="rt").run(0)
    assert seen["kind"] == "dual"

    ProfileAutoencoderConfigIO.save(GruAutoencoderConfig(), "gru_ae", tmp_path / "gru_run" / "meta")
    jepa_config                                 = worker_config(test_data_dir, "jepa")
    jepa_config.jepa.profile_autoencoder_logdir = tmp_path
    jepa_config.jepa.profile_autoencoder_run    = "gru_run"

    FoldTrainingWorker(jepa_config, run_tag="rt").run(0)
    assert seen["kind"] == "jepa"

    FoldTrainingWorker(worker_config(test_data_dir, "profile_autoencoder"), run_tag="rt").run(0)
    assert seen["kind"] == "ae"

    FoldTrainingWorker(worker_config(test_data_dir, "image_autoencoder"), run_tag="rt").run(0)
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
def test_backbone_fold_trainer_uses_the_entry_curriculum(test_data_dir, monkeypatch):
    """Verifies the backbone fold trainer inherits the entry curriculum, geometry origin and run name."""
    config = worker_config(test_data_dir, "backbone")
    worker = FoldTrainingWorker(config, run_tag="rt")

    captured = {}

    class _FakePipeline:
        """Backbone training pipeline stub recording its constructor arguments."""
        def __init__(self, trainer_config, dataset_config, backbone_name, model_config, seed, run_name, overfit_check):
            """Records the trainer and dataset configs, the run name and the overfit check."""
            captured["trainer_config"] = trainer_config
            captured["dataset_config"] = dataset_config
            captured["run_name"]       = run_name
            captured["overfit_check"]  = overfit_check

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
