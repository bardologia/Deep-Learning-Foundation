from __future__ import annotations

import pytest

from configuration.dataset                import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.sar.gaussian_config    import GaussianConfig
from configuration.training.backbone      import BackboneTrainerConfig
from pipelines.backbone.training.pipeline import TrainingPipeline
from tools.data.regions                   import CropRegion

from tests.backbone_training._helpers import geometry_config


pytestmark = pytest.mark.usefixtures("force_cpu")


def _dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    input_config = InputConfig(
        use_primary        = True, primary_representation        = Representation.MAG_ONLY,
        use_secondaries    = True, secondaries_representation    = Representation.MAG_ONLY,
        use_interferograms = True, interferograms_representation = Representation.ANGLE_ONLY,
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
        secondary_labels            = ("FL01_PS04", "FL01_PS06"),
        patch                       = PatchConfig(size=(8, 8), stride=(8, 8)),
        input_config                = input_config,
        batch_size                  = 2,
        num_workers                 = 0,
        prefetch_factor             = 8,
        n_gaussians                 = 5,
    )


def _trainer_config(test_data_dir, params_dir, tmp_path) -> BackboneTrainerConfig:
    gaussian = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy")
    config   = BackboneTrainerConfig(gaussian=gaussian)

    config.io.logdir                     = str(tmp_path)
    config.io.writer                     = None
    config.training.epochs               = 1
    config.training.validation_frequency = 1
    config.resources.enabled             = False
    config.geometry                      = geometry_config()

    config.curriculum.complete.use_param_l1    = True
    config.curriculum.complete.weight_param_l1 = 1.0

    return config


@pytest.mark.real_data
@pytest.mark.slow
def test_training_pipeline_end_to_end_produces_checkpoint(test_data_dir, params_dir, tmp_path):
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


@pytest.mark.real_data
@pytest.mark.slow
def test_training_pipeline_runs_per_pixel_physics_geometry(test_data_dir, params_dir, tmp_path):
    dataset_config = _dataset_config(test_data_dir, params_dir)
    trainer_config = _trainer_config(test_data_dir, params_dir, tmp_path)

    trainer_config.curriculum.complete.use_covariance_match    = True
    trainer_config.curriculum.complete.weight_covariance_match = 1.0
    trainer_config.geometry.height_axis_convention             = "height"

    pipeline = TrainingPipeline(
        trainer_config = trainer_config,
        dataset_config = dataset_config,
        backbone_name  = "resunet",
        model_config   = None,
        seed           = 0,
        run_name       = "pipeline_physics",
    )

    train_losses, val_losses, best_val = pipeline.run(probe_config=None)

    assert pipeline.dataset_pipeline.geometry_field is not None
    assert pipeline.dataset_pipeline.geometry_field.n_tracks == 3
    assert (pipeline.run_metadata.run_directory / "best_model.pt").exists()
    assert len(train_losses) == 1


@pytest.mark.real_data
@pytest.mark.slow
def test_training_pipeline_saves_configs(test_data_dir, params_dir, tmp_path):
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


@pytest.mark.real_data
@pytest.mark.slow
def test_training_pipeline_overfit_check_gates_and_reports(test_data_dir, params_dir, tmp_path):
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


@pytest.mark.real_data
def test_training_pipeline_overfit_check_abort_leaves_configs(test_data_dir, params_dir, tmp_path):
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
