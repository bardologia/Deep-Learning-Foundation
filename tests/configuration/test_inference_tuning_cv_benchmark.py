from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from configuration.inference.general             import InferencePaths, InferenceConfig
from configuration.inference.backbone            import BackboneInferenceEntryConfig
from configuration.inference.image_autoencoder   import ImageAeInferenceEntryConfig
from configuration.inference.profile_autoencoder import ProfileAeInferenceEntryConfig
from configuration.inference.image_autoencoder   import ImageAeInferenceConfig
from configuration.inference.profile_autoencoder import ProfileAeInferenceConfig

from configuration.tuning.general import TuningConfig, TuningEntryConfig
from configuration.tuning.jepa    import JepaTuneConfig

from configuration.cross_validation.general             import FoldConfig, CrossValidationConfig
from configuration.cross_validation.jepa                import JepaCvConfig
from configuration.cross_validation.profile_autoencoder import AeCvConfig

from configuration.benchmark.general import (
    MaxBatchConfig,
    SizeMatchConfig,
    InferenceQueueConfig,
    ComparisonReportConfig,
    BenchmarkConfig,
)
from configuration.benchmark.jepa               import JepaBenchConfig
from configuration.benchmark.dataloader_tuning  import DataLoaderTuningEntryConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig
from configuration.training.profile_autoencoder import ProfileAeLossConfig

from configuration.dataset               import AugmentationConfig, InputConfig
from configuration.normalization.general import NormalizationConfig
from configuration.sar.geometry_config   import GeometryConfig


DEFAULT_CONFIGS = [
    InferencePaths,
    BackboneInferenceEntryConfig,
    ProfileAeInferenceEntryConfig,
    ImageAeInferenceEntryConfig,
    TuningConfig,
    TuningEntryConfig,
    JepaTuneConfig,
    FoldConfig,
    CrossValidationConfig,
    JepaCvConfig,
    AeCvConfig,
    RunPathsConfig,
    MaxBatchConfig,
    SizeMatchConfig,
    TrainingQueueConfig,
    InferenceQueueConfig,
    ComparisonReportConfig,
    BenchmarkConfig,
    JepaBenchConfig,
    DataLoaderTuningEntryConfig,
]

IDS = [c.__name__ for c in DEFAULT_CONFIGS]


@pytest.mark.parametrize("config_cls", DEFAULT_CONFIGS, ids=IDS)
def test_default_config_instantiates(config_cls):
    instance = config_cls()
    assert dataclasses.is_dataclass(instance)


def test_inference_config_requires_run_directory():
    with pytest.raises(TypeError):
        InferenceConfig()

    cfg = InferenceConfig(run_directory=Path("."))
    assert cfg.device == "cuda"
    assert cfg.batch_size is None
    assert isinstance(cfg.paths, InferencePaths)
    assert isinstance(cfg.gif_axes, list)


def test_inference_config_profile_counts_positive():
    cfg = InferenceConfig(run_directory=Path("."))
    assert cfg.n_best_profiles   > 0
    assert cfg.n_worst_profiles  > 0
    assert cfg.n_random_profiles > 0


def test_backbone_inference_entry_config_holds_inference():
    cfg = BackboneInferenceEntryConfig()
    assert isinstance(cfg.inference, InferenceConfig)
    assert cfg.inference.gif_axes == ["elevation", "range", "azimuth"]
    assert isinstance(cfg.run_filter, list)
    assert cfg.runs_dir.endswith("/runs")


def test_ae_inference_entry_configs_expose_only_their_tree():
    profile = ProfileAeInferenceEntryConfig()
    image   = ImageAeInferenceEntryConfig()

    assert isinstance(profile.profile_inference, ProfileAeInferenceConfig)
    assert isinstance(image.image_inference, ImageAeInferenceConfig)
    assert not hasattr(profile, "inference")
    assert not hasattr(image, "inference")
    assert not hasattr(BackboneInferenceEntryConfig(), "profile_inference")
    assert profile.runs_dir.endswith("/profile_autoencoder")
    assert image.runs_dir.endswith("/image_autoencoder")


def test_tuning_config_positive_trials():
    cfg = TuningConfig()
    assert cfg.n_trials > 0
    assert cfg.n_epochs > 0
    assert cfg.early_stop_patience > 0


def test_tuning_entry_config_subconfigs():
    cfg = TuningEntryConfig()
    assert isinstance(cfg.tuning, TuningConfig)
    assert isinstance(cfg.jepa, JepaTuneConfig)
    assert isinstance(cfg.paths, RunPathsConfig)
    assert isinstance(cfg.gpus, list)
    assert cfg.training.batch_size > 0


def test_tuning_entry_ae_loss_default_factory():
    cfg = TuningEntryConfig()
    assert isinstance(cfg.ae_loss, ProfileAeLossConfig)


def test_fold_config_defaults():
    cfg = FoldConfig()
    assert cfg.n_folds > 0
    assert cfg.azimuth_end > cfg.azimuth_start


def test_cross_validation_config_subconfigs():
    cfg = CrossValidationConfig()
    assert isinstance(cfg.folds, FoldConfig)
    assert isinstance(cfg.jepa, JepaCvConfig)
    assert isinstance(cfg.autoencoder, AeCvConfig)
    assert cfg.inference_splits == ["val", "test"]


def test_cross_validation_runs_inference_logic():
    assert CrossValidationConfig(training_type="backbone").runs_inference() is True
    assert CrossValidationConfig(training_type="profile_autoencoder").runs_inference() is False


def test_benchmark_paths_defaults_are_paths():
    cfg = RunPathsConfig()
    assert isinstance(cfg.dataset_path, Path)
    assert isinstance(cfg.parameters_path, Path)
    assert isinstance(cfg.secondary_labels, tuple)


def test_training_queue_config_axis_tuples():
    cfg = TrainingQueueConfig()
    assert cfg.epochs > 0
    assert cfg.batch_size > 0
    assert len(cfg.patch_size) == 2
    assert len(cfg.train_azimuth) == 2
    assert cfg.train_azimuth[1] > cfg.train_azimuth[0]


def test_size_match_locked_params_tuple():
    cfg = SizeMatchConfig()
    assert isinstance(cfg.locked_params, tuple)
    assert cfg.scale_high > cfg.scale_low
    assert cfg.in_channels > 0


def test_max_batch_config_budget_positive():
    cfg = MaxBatchConfig()
    assert cfg.vram_budget_gb > 0
    assert cfg.max_batch > 0


def test_benchmark_config_dispatch_methods():
    backbone = BenchmarkConfig(training_type="backbone")
    assert backbone.runs_size_match() is True
    assert backbone.runs_max_batch() is True
    assert backbone.runs_inference() is True

    pae = BenchmarkConfig(training_type="profile_autoencoder")
    assert pae.runs_size_match() is False
    assert pae.runs_inference() is False

    jepa = BenchmarkConfig(training_type="jepa")
    assert jepa.runs_inference() is True


def test_benchmark_config_subconfigs():
    cfg = BenchmarkConfig()
    assert isinstance(cfg.max_batch, MaxBatchConfig)
    assert isinstance(cfg.size_match, SizeMatchConfig)
    assert isinstance(cfg.jepa, JepaBenchConfig)


def test_benchmark_config_exposes_training_recipe_sections():
    cfg = BenchmarkConfig()
    assert isinstance(cfg.input, InputConfig)
    assert isinstance(cfg.geometry, GeometryConfig)
    assert isinstance(cfg.normalization, NormalizationConfig)
    assert isinstance(cfg.augmentation, AugmentationConfig)


def test_dataloader_tuning_entry_lists_independent():
    a = DataLoaderTuningEntryConfig()
    b = DataLoaderTuningEntryConfig()
    a.batch_sizes.append(99999)
    assert 99999 not in b.batch_sizes
    assert isinstance(a.output_dir, Path)
