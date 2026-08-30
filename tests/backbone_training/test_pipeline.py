"""Tests for the backbone and dual training pipelines.

Covers the end-to-end run on real preprocessing data (checkpoint, metadata and config
patch-squaring rule applied for backbones that only accept square inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configuration.architectures          import DualResUNetConfig
from configuration.dataset                import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.dataset.gaussian    import GaussianConfig
from configuration.training.backbone      import BackboneTrainerConfig
from pipelines.backbone.training.pipeline import TrainingPipeline
from pipelines.dual.training.pipeline     import DualTrainingPipeline
from tools.data.regions                   import CropRegion



pytestmark = pytest.mark.usefixtures("force_cpu")


def _dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    """Returns a small five-Gaussian dataset config over three adjacent azimuth split regions."""
    input_config = InputConfig(
        use_primary     = True, primary_representation        = Representation.MAG_ONLY,
        use_secondaries = True, secondaries_representation    = Representation.MAG_ONLY,
        use_derived     = True, derived_representation = Representation.ANGLE_ONLY,
    )

    splits = SplitRegions(
        train = CropRegion(1000, 1024, 500, 524),
        val   = CropRegion(1024, 1040, 500, 524),
        test  = CropRegion(1040, 1056, 500, 524),
    )

    return DatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = splits,
        secondary_labels            = ("CH01", "CH02"),
        patch                       = PatchConfig(size=(8, 8), stride=(8, 8)),
        input_config                = input_config,
        batch_size                  = 2,
        num_workers                 = 0,
        prefetch_factor             = 8,
        n_gaussians                 = 5,
    )


def _trainer_config(test_data_dir, params_dir, tmp_path) -> BackboneTrainerConfig:
    """Returns a one-epoch backbone trainer config with only the param_l1 term active."""
    gaussian = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy")
    config   = BackboneTrainerConfig(gaussian=gaussian)

    config.io.logdir                     = str(tmp_path)
    config.training.epochs               = 1
    config.training.validation_frequency = 1
    config.resources.enabled             = False

    config.curriculum.complete.use_param_l1    = True
    config.curriculum.complete.weight_param_l1 = 1.0

    return config


@pytest.mark.dataset
@pytest.mark.slow
def test_training_pipeline_end_to_end_produces_checkpoint(test_data_dir, params_dir, tmp_path):
    """Verifies a full pipeline run writes the best checkpoint and the metadata directory."""
    dataset_config = _dataset_config(test_data_dir, params_dir)
    trainer_config = _trainer_config(test_data_dir, params_dir, tmp_path)

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        backbone_name  = "resunet",
        model_config   = None,
        seed           = 0,
        run_name       = "pipeline_smoke",
    )

    train_losses, val_losses, best_val = pipeline.run(probe_config=None)

    run_directory = pipeline.run_metadata.run_directory

    assert (run_directory / "best_model.pt").exists()
    assert (run_directory / "meta").is_dir()
    assert len(train_losses) == 1
    assert val_losses is not None


@pytest.mark.dataset
@pytest.mark.slow
def test_training_pipeline_saves_configs(test_data_dir, params_dir, tmp_path):
    """Verifies the run writes configuration artifacts into the metadata directory."""
    dataset_config = _dataset_config(test_data_dir, params_dir)
    trainer_config = _trainer_config(test_data_dir, params_dir, tmp_path)

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        backbone_name  = "resunet",
        model_config   = None,
        seed           = 0,
        run_name       = "pipeline_cfg",
    )

    pipeline.run(probe_config=None)

    meta_dir = pipeline.run_metadata.run_directory / "meta"
    written  = list(meta_dir.glob("*.json")) + list(meta_dir.glob("*.yaml")) + list(meta_dir.glob("*.txt"))

    assert meta_dir.is_dir()
    assert len(written) > 0


@pytest.mark.dataset
@pytest.mark.slow
def test_training_pipeline_overfit_check_gates_and_reports(test_data_dir, params_dir, tmp_path):
    """Verifies a passing overfit check reports its sanitized overrides, leaves no checkpoints behind and lets training proceed."""
    import json

    from configuration.training import OverfitCheckConfig

    dataset_config = _dataset_config(test_data_dir, params_dir)
    trainer_config = _trainer_config(test_data_dir, params_dir, tmp_path)

    overfit_check = OverfitCheckConfig(enabled=True, n_examples=2, max_steps=4, steps_per_epoch=2, pass_loss_ratio=1.0)

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        backbone_name  = "resunet",
        model_config   = None,
        seed           = 0,
        run_name       = "pipeline_overfit_check",
        overfit_check  = overfit_check,
    )

    train_losses, val_losses, best_val = pipeline.run(probe_config=None)

    run_directory = pipeline.run_metadata.run_directory
    report_path   = run_directory / "meta" / "overfit_report.json"

    assert report_path.is_file()

    report = json.loads(report_path.read_text())

    assert report["passed"] is True
    assert report["n_examples"] == 2
    assert report["sanitized_overrides"]["optimizer.weight_decay"] == 0.0
    assert report["sanitized_overrides"]["gate_loss_stage"] == "complete"
    assert report["sanitized_overrides"]["gate_stage.use_active_normalization"] is False
    assert report["sanitized_overrides"]["model.dropout"] == 0.0
    assert report["sanitized_overrides"]["augmentation"] == "disabled"

    assert not (run_directory / "overfit_check" / "best_model.pt").exists()
    assert not (run_directory / "overfit_check" / "last.pt").exists()

    assert (run_directory / "best_model.pt").exists()
    assert len(train_losses) == 1


@pytest.mark.dataset
def test_training_pipeline_overfit_check_abort_leaves_configs(test_data_dir, params_dir, tmp_path):
    """Verifies a failing overfit check aborts the run but still leaves the report and config artifacts."""
    from configuration.training import OverfitCheckConfig

    dataset_config = _dataset_config(test_data_dir, params_dir)
    trainer_config = _trainer_config(test_data_dir, params_dir, tmp_path)

    overfit_check = OverfitCheckConfig(enabled=True, n_examples=2, max_steps=2, steps_per_epoch=2, pass_loss_ratio=0.0, stop_threshold=0.0)

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        backbone_name  = "resunet",
        model_config   = None,
        seed           = 0,
        run_name       = "pipeline_overfit_abort",
        overfit_check  = overfit_check,
    )

    with pytest.raises(RuntimeError, match="Overfit check failed"):
        pipeline.run(probe_config=None)

    run_directory = pipeline.run_metadata.run_directory

    assert (run_directory / "meta" / "overfit_report.json").is_file()
    assert (run_directory / "docs" / "trainer_config.json").is_file()
    assert (run_directory / "meta" / "run_summary.json").is_file()
    assert not (run_directory / "best_model.pt").exists()


def _square_patch_dataset_config() -> DatasetConfig:
    """Returns a dataset config with a rectangular (64, 32) patch and (32, 16) stride."""
    splits = SplitRegions(
        train = CropRegion(0, 8, 0, 8),
        val   = CropRegion(8, 16, 0, 8),
        test  = CropRegion(16, 24, 0, 8),
    )

    return DatasetConfig(preprocessing_run_directory=Path("/x"), split_regions=splits, patch=PatchConfig(size=(64, 32), stride=(32, 16)))


def _square_patch_trainer_config() -> BackboneTrainerConfig:
    """Returns a minimal five-Gaussian backbone trainer config."""
    return BackboneTrainerConfig(gaussian=GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=40.0))


def test_pipeline_squares_the_patch_for_square_only_backbones():
    """Verifies a square-only backbone widens the patch and stride to the larger axis."""
    pipeline = TrainingPipeline(trainer_config=_square_patch_trainer_config(), dataset_config=_square_patch_dataset_config(), backbone_name="swin_unet")

    assert pipeline.dataset_config.patch.size   == (64, 64)
    assert pipeline.dataset_config.patch.stride == (32, 32)
    assert pipeline.patch_size                  == (64, 64)


def test_pipeline_keeps_rectangular_patches_for_convolutional_backbones():
    """Verifies a convolutional backbone keeps the rectangular patch and stride."""
    pipeline = TrainingPipeline(trainer_config=_square_patch_trainer_config(), dataset_config=_square_patch_dataset_config(), backbone_name="resunet")

    assert pipeline.dataset_config.patch.size   == (64, 32)
    assert pipeline.dataset_config.patch.stride == (32, 16)


def test_dual_pipeline_squares_the_patch_when_a_trunk_is_square_only():
    """Verifies the dual pipeline squares the patch when either trunk is square-only."""
    model_config                    = DualResUNetConfig()
    model_config.params_backbone    = "resunet"
    model_config.existence_backbone = "swin_unet"

    pipeline = DualTrainingPipeline(trainer_config=_square_patch_trainer_config(), dataset_config=_square_patch_dataset_config(), backbone_name="dual_resunet", model_config=model_config)

    assert pipeline.dataset_config.patch.size   == (64, 64)
    assert pipeline.dataset_config.patch.stride == (32, 32)
