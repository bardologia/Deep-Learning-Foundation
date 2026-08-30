"""Tests for the inference, tuning, cross-validation and benchmark configurations.

Covers that each entry config builds from defaults, that inference defaults come
from one factory across standalone and training entries, and that the tuning,
fold and benchmark configs carry their expected sub-configs and dispatch rules.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from configuration.inference.general             import InferencePaths, InferenceConfig
from configuration.inference.backbone            import BackboneInferenceEntryConfig
from configuration.inference.dual                import DualInferenceEntryConfig
from configuration.inference.image_autoencoder   import ImageAeInferenceEntryConfig
from configuration.inference.profile_autoencoder import ProfileAeInferenceEntryConfig
from configuration.inference.image_autoencoder   import ImageAeInferenceConfig
from configuration.inference.profile_autoencoder import ProfileAeInferenceConfig

from configuration.tuning.general import TuningConfig, TuningEntryConfig

from configuration.cross_validation.general             import FoldConfig, CrossValidationConfig
from configuration.cross_validation.profile_autoencoder import AeCvConfig

from configuration.benchmark.general import (
    MaxBatchConfig,
    SizeMatchConfig,
    InferenceQueueConfig,
    ComparisonReportConfig,
    BenchmarkConfig,
)
from configuration.benchmark.dataloader_tuning  import DataLoaderTuningEntryConfig
from configuration.training.general.run         import RunPathsConfig, TrainingQueueConfig
from configuration.training.jepa               import JepaCouplingConfig
from configuration.training.profile_autoencoder import ProfileAeLossConfig

from configuration.dataset                  import AugmentationConfig, InputConfig
from configuration.training.general.runtime import OverfitCheckConfig
from configuration.normalization.general import NormalizationConfig


DEFAULT_CONFIGS = [
    InferencePaths,
    BackboneInferenceEntryConfig,
    ProfileAeInferenceEntryConfig,
    ImageAeInferenceEntryConfig,
    TuningConfig,
    TuningEntryConfig,
    JepaCouplingConfig,
    FoldConfig,
    CrossValidationConfig,
    AeCvConfig,
    RunPathsConfig,
    MaxBatchConfig,
    SizeMatchConfig,
    TrainingQueueConfig,
    InferenceQueueConfig,
    ComparisonReportConfig,
    BenchmarkConfig,
    DataLoaderTuningEntryConfig,
]

IDS = [c.__name__ for c in DEFAULT_CONFIGS]


@pytest.mark.parametrize("config_cls", DEFAULT_CONFIGS, ids=IDS)
def test_default_config_instantiates(config_cls):
    """Verifies every listed entry config builds from defaults as a dataclass."""
    instance = config_cls()
    assert dataclasses.is_dataclass(instance)


def test_inference_config_requires_run_directory():
    """Verifies the inference config demands a run directory and defaults to CUDA with an unset batch size."""
    with pytest.raises(TypeError):
        InferenceConfig()

    cfg = InferenceConfig(run_directory=Path("."))
    assert cfg.device == "cuda"
    assert cfg.batch_size is None
    assert isinstance(cfg.paths, InferencePaths)
    assert isinstance(cfg.gif_axes, list)


def test_inference_config_profile_counts_positive():
    """Verifies the best, worst and random profile counts plotted at inference are positive."""
    cfg = InferenceConfig(run_directory=Path("."))
    assert cfg.n_best_profiles   > 0
    assert cfg.n_worst_profiles  > 0
    assert cfg.n_random_profiles > 0


def test_backbone_inference_entry_config_holds_inference():
    """Verifies the backbone inference entry carries an inference config, the three GIF axes and a runs directory."""
    cfg = BackboneInferenceEntryConfig()
    assert isinstance(cfg.inference, InferenceConfig)
    assert cfg.inference.gif_axes == ["z", "col", "row"]
    assert isinstance(cfg.run_filter, list)
    assert cfg.runs_dir == "runs"


def test_run_directory_inference_defaults_come_from_one_factory():
    """Verifies standalone and training inference configs share the same defaults and differ only in whether reduced products are computed."""
    from configuration.training.backbone import BackboneEntryConfig
    from configuration.training.dual     import DualEntryConfig
    from configuration.training.jepa     import JepaEntryConfig

    standalone = [BackboneInferenceEntryConfig().inference, DualInferenceEntryConfig().inference]
    training   = [BackboneEntryConfig().inference, DualEntryConfig().inference, JepaEntryConfig().inference]

    for cfg in standalone + training:
        assert cfg.run_directory == Path(".")
        assert cfg.save_cubes  is True
        assert cfg.cpu_workers == 16
        assert cfg.gif_axes    == ["z", "col", "row"]



def test_ae_inference_entry_configs_expose_only_their_tree():
    """Verifies the autoencoder inference entries expose only their own inference tree and runs directory."""
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
    """Verifies the tuning defaults carry positive trial, epoch and patience counts."""
    cfg = TuningConfig()
    assert cfg.n_trials > 0
    assert cfg.n_epochs > 0
    assert cfg.early_stop_patience > 0


def test_tuning_entry_config_subconfigs():
    """Verifies the tuning entry carries the tuning, JEPA coupling, paths and GPU sub-configs with a positive batch size."""
    cfg = TuningEntryConfig()
    assert isinstance(cfg.tuning, TuningConfig)
    assert isinstance(cfg.jepa, JepaCouplingConfig)
    assert isinstance(cfg.paths, RunPathsConfig)
    assert isinstance(cfg.gpus, list)
    assert cfg.training.batch_size > 0


def test_tuning_entry_ae_loss_default_factory():
    """Verifies the tuning entry builds its own profile autoencoder loss config."""
    cfg = TuningEntryConfig()
    assert isinstance(cfg.ae_loss, ProfileAeLossConfig)


def test_fold_config_defaults():
    """Verifies the fold defaults carry a positive fold count over a non-empty row extent."""
    cfg = FoldConfig()
    assert cfg.n_folds > 0
    assert cfg.row_end > cfg.row_start


def test_cross_validation_config_subconfigs():
    """Verifies the cross-validation config carries its fold, JEPA and autoencoder sub-configs and evaluates both val and test."""
    cfg = CrossValidationConfig()
    assert isinstance(cfg.folds, FoldConfig)
    assert isinstance(cfg.jepa, JepaCouplingConfig)
    assert isinstance(cfg.autoencoder, AeCvConfig)
    assert cfg.inference_splits == ["val", "test"]


def test_cross_validation_validates_the_standard_backbone():
    """Verifies cross-validation defaults to the same backbone and head as backbone training."""
    from configuration.training.backbone import BackboneEntryConfig

    cfg = CrossValidationConfig()

    assert cfg.backbone_name == BackboneEntryConfig.backbone_name
    assert cfg.backbone_head == BackboneEntryConfig.backbone_head


def test_cross_validation_carries_the_input_stack_and_overfit_gate():
    """Verifies cross-validation carries the input stack with secondaries enabled and the overfit-check gate."""
    cfg = CrossValidationConfig()

    assert isinstance(cfg.input, InputConfig)
    assert cfg.input.use_secondaries is True
    assert isinstance(cfg.overfit_check, OverfitCheckConfig)


def test_cross_validation_fold_inference_saves_cubes():
    """Verifies fold inference saves cubes while benchmark inference does not."""
    assert CrossValidationConfig().inference.save_cubes is True
    assert BenchmarkConfig().inference.save_cubes is False


def test_cross_validation_runs_inference_logic():
    """Verifies inference runs for backbone and profile autoencoder training but not when the split list is empty."""
    assert CrossValidationConfig(training_type="backbone").runs_inference() is True
    assert CrossValidationConfig(training_type="profile_autoencoder").runs_inference() is True
    assert CrossValidationConfig(training_type="image_autoencoder", inference_splits=[]).runs_inference() is False


def test_benchmark_paths_defaults_are_paths():
    """Verifies the run paths config exposes path objects and a tuple of secondary labels."""
    cfg = RunPathsConfig()
    assert isinstance(cfg.dataset_path, Path)
    assert isinstance(cfg.parameters_path, Path)
    assert isinstance(cfg.secondary_labels, tuple)


def test_training_queue_config_axis_tuples():
    """Verifies the queued training defaults carry positive epochs and batch size plus two-element patch and row extents."""
    cfg = TrainingQueueConfig()
    assert cfg.epochs > 0
    assert cfg.batch_size > 0
    assert len(cfg.patch_size) == 2
    assert len(cfg.train_rows) == 2
    assert cfg.train_rows[1] > cfg.train_rows[0]


def test_size_match_locked_params_tuple():
    """Verifies the size-match defaults lock a tuple of parameters over a valid scale range with positive input channels."""
    cfg = SizeMatchConfig()
    assert isinstance(cfg.locked_params, tuple)
    assert cfg.scale_high > cfg.scale_low
    assert cfg.in_channels > 0


def test_max_batch_config_budget_positive():
    """Verifies the max-batch defaults carry a positive VRAM budget and batch ceiling."""
    cfg = MaxBatchConfig()
    assert cfg.vram_budget_gb > 0
    assert cfg.max_batch > 0


def test_benchmark_config_dispatch_methods():
    """Verifies which stages each training type runs: size match for backbone only, no inference for the profile autoencoder, inference for JEPA."""
    backbone = BenchmarkConfig(training_type="backbone")
    assert backbone.runs_size_match() is True
    assert backbone.runs_inference() is True

    pae = BenchmarkConfig(training_type="profile_autoencoder")
    assert pae.runs_size_match() is False
    assert pae.runs_inference() is False

    jepa = BenchmarkConfig(training_type="jepa")
    assert jepa.runs_inference() is True


def test_benchmark_config_subconfigs():
    """Verifies the benchmark config carries the max-batch, size-match and JEPA coupling sub-configs."""
    cfg = BenchmarkConfig()
    assert isinstance(cfg.max_batch, MaxBatchConfig)
    assert isinstance(cfg.size_match, SizeMatchConfig)
    assert isinstance(cfg.jepa, JepaCouplingConfig)


def test_benchmark_config_exposes_training_recipe_sections():
    """Verifies the benchmark config exposes the input, geometry, normalization and augmentation recipe sections."""
    cfg = BenchmarkConfig()
    assert isinstance(cfg.input, InputConfig)
    assert isinstance(cfg.normalization, NormalizationConfig)
    assert isinstance(cfg.augmentation, AugmentationConfig)


def test_dataloader_tuning_entry_lists_independent():
    """Verifies two dataloader tuning entries hold independent batch-size lists and a path output directory."""
    a = DataLoaderTuningEntryConfig()
    b = DataLoaderTuningEntryConfig()
    a.batch_sizes.append(99999)
    assert 99999 not in b.batch_sizes
    assert isinstance(a.output_dir, Path)
