
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from configuration.inference.general  import InferencePaths, InferenceConfig
from configuration.inference.backbone import BackboneInferenceEntryConfig

from configuration.tuning.general import TuningConfig, TuningEntryConfig

from configuration.cross_validation.general import FoldConfig, CrossValidationConfig

from configuration.benchmark.general import (
    MaxBatchConfig,
    SizeMatchConfig,
    InferenceQueueConfig,
    ComparisonReportConfig,
    BenchmarkConfig,
)
from configuration.benchmark.dataloader_tuning import DataLoaderTuningEntryConfig
from configuration.training.general.run        import RunPathsConfig, TrainingQueueConfig

from configuration.dataset                  import AugmentationConfig, InputConfig
from configuration.training.general.runtime import OverfitCheckConfig
from configuration.normalization.general import NormalizationConfig


DEFAULT_CONFIGS = [
    InferencePaths,
    BackboneInferenceEntryConfig,
    TuningConfig,
    TuningEntryConfig,
    FoldConfig,
    CrossValidationConfig,
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
    assert cfg.inference.gif_axes == ["z", "col", "row"]
    assert isinstance(cfg.run_filter, list)
    assert cfg.runs_dir == "runs"


def test_run_directory_inference_defaults_come_from_one_factory():
    from configuration.training.backbone import BackboneEntryConfig

    standalone = [BackboneInferenceEntryConfig().inference]
    training   = [BackboneEntryConfig().inference]

    for cfg in standalone + training:
        assert cfg.run_directory == Path(".")
        assert cfg.save_cubes  is True
        assert cfg.cpu_workers == 16
        assert cfg.gif_axes    == ["z", "col", "row"]


def test_tuning_config_positive_trials():
    cfg = TuningConfig()
    assert cfg.n_trials > 0
    assert cfg.n_epochs > 0
    assert cfg.early_stop_patience > 0


def test_tuning_config_searches_the_model_groups_by_default():
    cfg = TuningConfig()
    assert cfg.search_lr    is True
    assert cfg.search_arch  is True
    assert cfg.search_batch is False
    assert cfg.batch_sizes  == [64, 128, 256, 512, 1024]


def test_tuning_config_stop_rules_default_off():
    cfg = TuningConfig()
    assert cfg.plateau_patience  == 0
    assert cfg.plateau_min_delta == 0.0
    assert cfg.time_budget_h     == 0.0


def test_tuning_config_discards_trial_checkpoints_by_default():
    assert TuningConfig().keep_trial_checkpoints is False


def test_training_queue_lr_scale_follows_the_batch_rule():
    cfg = TrainingQueueConfig(batch_size=256, lr_reference_batch_size=256, lr_multiplier=2.0)

    assert cfg.lr_scale()    == 2.0
    assert cfg.lr_scale(128) == 1.0

    cfg.scale_lr_with_batch = False
    assert cfg.lr_scale(64) == 2.0


def test_tuning_entry_config_subconfigs():
    cfg = TuningEntryConfig()
    assert isinstance(cfg.tuning, TuningConfig)
    assert isinstance(cfg.paths, RunPathsConfig)
    assert isinstance(cfg.gpus, list)
    assert cfg.training.batch_size > 0


def test_fold_config_defaults():
    cfg = FoldConfig()
    assert cfg.n_folds > 0
    assert cfg.row_end > cfg.row_start


def test_cross_validation_config_subconfigs():
    cfg = CrossValidationConfig()
    assert isinstance(cfg.folds, FoldConfig)
    assert cfg.inference_splits == ["val", "test"]


def test_cross_validation_validates_the_standard_backbone():
    from configuration.training.backbone import BackboneEntryConfig

    cfg = CrossValidationConfig()

    assert cfg.backbone_name == BackboneEntryConfig.backbone_name
    assert cfg.backbone_head == BackboneEntryConfig.backbone_head


def test_cross_validation_carries_the_input_stack_and_overfit_gate():
    cfg = CrossValidationConfig()

    assert isinstance(cfg.input, InputConfig)
    assert cfg.input.use_secondaries is True
    assert isinstance(cfg.overfit_check, OverfitCheckConfig)


def test_cross_validation_fold_inference_saves_cubes():
    assert CrossValidationConfig().inference.save_cubes is True
    assert BenchmarkConfig().inference.save_cubes is False


def test_cross_validation_runs_inference_logic():
    assert CrossValidationConfig().runs_inference() is True
    assert CrossValidationConfig(inference_splits=[]).runs_inference() is False


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
    assert len(cfg.train_rows) == 2
    assert cfg.train_rows[1] > cfg.train_rows[0]


def test_size_match_locked_params_tuple():
    cfg = SizeMatchConfig()
    assert isinstance(cfg.locked_params, tuple)
    assert cfg.scale_high > cfg.scale_low
    assert cfg.in_channels > 0


def test_max_batch_config_budget_positive():
    cfg = MaxBatchConfig()
    assert cfg.vram_budget_gb > 0
    assert cfg.max_batch > 0


def test_benchmark_config_runs_inference_when_infer_after_is_on():
    assert BenchmarkConfig().runs_inference() is True
    assert BenchmarkConfig(infer_after=False).runs_inference() is False


def test_benchmark_config_subconfigs():
    cfg = BenchmarkConfig()
    assert isinstance(cfg.max_batch, MaxBatchConfig)
    assert isinstance(cfg.size_match, SizeMatchConfig)


def test_benchmark_config_exposes_training_recipe_sections():
    cfg = BenchmarkConfig()
    assert isinstance(cfg.input, InputConfig)
    assert isinstance(cfg.normalization, NormalizationConfig)
    assert isinstance(cfg.augmentation, AugmentationConfig)


def test_dataloader_tuning_entry_lists_independent():
    a = DataLoaderTuningEntryConfig()
    b = DataLoaderTuningEntryConfig()
    a.batch_sizes.append(99999)
    assert 99999 not in b.batch_sizes
    assert isinstance(a.output_dir, Path)
