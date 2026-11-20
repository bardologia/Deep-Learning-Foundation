
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from configuration.dataset                import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.dataset.gaussian    import GaussianConfig
from configuration.training.backbone      import BackboneTrainerConfig
from pipelines.backbone.training.pipeline import TrainingPipeline
from pipelines.shared.model.model_builder import ModelBuilder
from tools.data.gaussians                 import GaussianActivity, GaussianSlotSorter
from tools.data.io                        import FileIO
from tools.data.regions                   import CropRegion



pytestmark = pytest.mark.usefixtures("force_cpu")


def _dataset_config(test_data_dir, params_dir) -> DatasetConfig:
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

    train_losses, val_losses, best_val = pipeline.run()

    run_directory = pipeline.run_metadata.run_directory

    assert (run_directory / "best_model.pt").exists()
    assert (run_directory / "meta").is_dir()
    assert len(train_losses) == 1
    assert val_losses is not None


@pytest.mark.dataset
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

    pipeline.run()

    meta_dir = pipeline.run_metadata.run_directory / "meta"
    written  = list(meta_dir.glob("*.json")) + list(meta_dir.glob("*.yaml")) + list(meta_dir.glob("*.txt"))

    assert meta_dir.is_dir()
    assert len(written) > 0


def _prior_bias_pipeline(test_data_dir, params_dir, tmp_path, prior_bias_init: bool) -> TrainingPipeline:
    return TrainingPipeline(
        trainer_config  = _trainer_config(test_data_dir, params_dir, tmp_path),
        dataset_config  = _dataset_config(test_data_dir, params_dir),
        backbone_name   = "unet",
        model_config    = ModelBuilder.config_from_registry("unet", {"features": [8, 16]}, head="conv"),
        seed            = 0,
        run_name        = f"pipeline_prior_bias_{prior_bias_init}",
        prior_bias_init = prior_bias_init,
    )


@pytest.mark.dataset
@pytest.mark.slow
def test_training_pipeline_starts_the_conv_head_at_the_training_label_prior(test_data_dir, params_dir, tmp_path):
    pipeline = _prior_bias_pipeline(test_data_dir, params_dir, tmp_path, prior_bias_init=True)
    pipeline.run()

    record      = FileIO.load_json(pipeline.run_metadata.run_directory / "meta" / "head_prior_bias.json")
    threshold   = pipeline.trainer_config.gaussian.activity_threshold
    n_gaussians = pipeline.trainer_config.gaussian.n_default_gaussians

    labels     = np.load(params_dir / "parameters.npy", mmap_mode="r")
    rows, cols = pipeline.dataset_config.split_regions.train.local_slices(pipeline.dataset_pipeline.layout.global_crop)
    ordered    = GaussianSlotSorter.by_mean(np.asarray(labels[:3 * n_gaussians, rows, cols], dtype=np.float32), n_gaussians, threshold)
    active     = [int(GaussianActivity.is_active(ordered[3 * slot], threshold).sum()) for slot in range(n_gaussians)]

    assert record["activity_threshold"] == threshold
    assert record["training_pixels"]    == 24 * 24
    assert record["active_pixels"]      == active
    assert len(record["prior_bias"])    == 3 * n_gaussians
    assert np.isfinite(record["prior_bias"]).all()
    assert not np.allclose(record["prior_bias"], record["default_bias"])


@pytest.mark.dataset
@pytest.mark.slow
def test_training_pipeline_keeps_the_default_head_bias_when_the_prior_is_off(test_data_dir, params_dir, tmp_path):
    pipeline = _prior_bias_pipeline(test_data_dir, params_dir, tmp_path, prior_bias_init=False)
    pipeline.run()

    assert (pipeline.run_metadata.run_directory / "meta" / "run_summary.json").is_file()
    assert not (pipeline.run_metadata.run_directory / "meta" / "head_prior_bias.json").exists()


def test_training_pipeline_records_a_dataset_stage_abort_in_its_run_log(tmp_path, monkeypatch):
    trainer_config                   = _square_patch_trainer_config()
    trainer_config.io.logdir         = str(tmp_path)
    trainer_config.resources.enabled = False

    dataset_config                     = _square_patch_dataset_config()
    dataset_config.fingerprint_dataset = False

    pipeline = TrainingPipeline(trainer_config=trainer_config, dataset_config=dataset_config, backbone_name="resunet", run_name="dataset_abort")

    monkeypatch.setattr(pipeline, "_build_dataset_pipeline", Mock(side_effect=ValueError("dataset stage broke")))

    with pytest.raises(ValueError, match="dataset stage broke"):
        pipeline.run()

    text = (pipeline.run_metadata.run_directory / "logs" / "resunet_metadata.log").read_text(encoding="utf-8")

    assert "[Aborted] ValueError: dataset stage broke" in text
    assert "Traceback (most recent call last)" in text
    assert ">>> ABORTED resunet_metadata" in text
    assert ">>> END" not in text


def _square_patch_dataset_config() -> DatasetConfig:
    splits = SplitRegions(
        train = CropRegion(0, 8, 0, 8),
        val   = CropRegion(8, 16, 0, 8),
        test  = CropRegion(16, 24, 0, 8),
    )

    return DatasetConfig(preprocessing_run_directory=Path("/x"), split_regions=splits, patch=PatchConfig(size=(64, 32), stride=(32, 16)))


def _square_patch_trainer_config() -> BackboneTrainerConfig:
    return BackboneTrainerConfig(gaussian=GaussianConfig(n_default_gaussians=5, x_min=-20.0, x_max=40.0, activity_threshold=1e-4))


def test_pipeline_squares_the_patch_for_square_only_backbones():
    pipeline = TrainingPipeline(trainer_config=_square_patch_trainer_config(), dataset_config=_square_patch_dataset_config(), backbone_name="swin_unet")

    assert pipeline.dataset_config.patch.size   == (64, 64)
    assert pipeline.dataset_config.patch.stride == (32, 32)
    assert pipeline.patch_size                  == (64, 64)
    assert pipeline.configured_patch            == ((64, 32), (32, 16))
    assert pipeline.patch_squared               is True


def test_pipeline_names_the_configured_and_effective_geometry_when_squared():
    pipeline = TrainingPipeline(trainer_config=_square_patch_trainer_config(), dataset_config=_square_patch_dataset_config(), backbone_name="swin_unet")

    assert pipeline._patch_geometry_label()   == "configured patch (64, 32) stride (32, 16) -> effective patch (64, 64) stride (32, 32) (swin_unet needs a square input)"
    assert pipeline._patch_geometry_payload() == {"configured_size": [64, 32], "configured_stride": [32, 16], "size": [64, 64], "stride": [32, 32], "squared": True}


def test_pipeline_keeps_rectangular_patches_for_convolutional_backbones():
    pipeline = TrainingPipeline(trainer_config=_square_patch_trainer_config(), dataset_config=_square_patch_dataset_config(), backbone_name="resunet")

    assert pipeline.dataset_config.patch.size   == (64, 32)
    assert pipeline.dataset_config.patch.stride == (32, 16)
    assert pipeline.patch_squared               is False
    assert pipeline._patch_geometry_label()     == "patch (64, 32) stride (32, 16)"
    assert pipeline._patch_geometry_payload()["squared"] is False
