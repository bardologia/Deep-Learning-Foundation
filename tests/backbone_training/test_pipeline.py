from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from configuration.dataset                  import DatasetConfig, PatchConfig
from configuration.training.backbone        import BackboneTrainerConfig
from pipelines.backbone.training.pipeline   import TrainingPipeline
from pipelines.shared.dataset.dataset_files import DatasetFiles
from tools.data.regions                     import CropRegion, SplitRegions


pytestmark = pytest.mark.usefixtures("force_cpu")


def _dataset_config(test_data_dir) -> DatasetConfig:
    return DatasetConfig(
        dataset_directory = test_data_dir,
        split_regions     = DatasetFiles.read(test_data_dir).splits,
        patch             = PatchConfig(size=(16, 16), stride=(16, 16)),
        batch_size        = 8,
        num_workers       = 0,
    )


def _trainer_config(tmp_path, epochs: int = 1) -> BackboneTrainerConfig:
    config = BackboneTrainerConfig()

    config.io.logdir                     = str(tmp_path)
    config.training.epochs               = epochs
    config.training.validation_frequency = 1
    config.resources.enabled             = False

    return config


def _pipeline(test_data_dir, tmp_path, run_name: str, epochs: int = 1) -> TrainingPipeline:
    return TrainingPipeline(trainer_config=_trainer_config(tmp_path, epochs), dataset_config=_dataset_config(test_data_dir), backbone_name="local_cnn", model_config=None, seed=0, run_name=run_name)


@pytest.mark.dataset
def test_training_pipeline_learns_the_synthetic_task_and_writes_the_run(test_data_dir, tmp_path):
    pipeline = _pipeline(test_data_dir, tmp_path, "pipeline_smoke", epochs=6)

    train_losses, val_losses, best_val = pipeline.run()

    run_directory = pipeline.run_metadata.run_directory
    summary       = json.loads((run_directory / "meta" / "run_summary.json").read_text())

    assert (run_directory / "best_model.pt").is_file()
    assert (run_directory / "meta" / "normalization_stats.json").is_file()
    assert (run_directory / "meta" / "dataset_creation_config.json").is_file()
    assert summary["input_channels"]  == ["x0", "x1", "x2", "x3"]
    assert summary["target_channels"] == ["pointwise", "neighbourhood"]
    assert (summary["in_channels"], summary["out_channels"]) == (4, 2)
    assert len(train_losses) == 6
    assert train_losses[-1] < train_losses[0]
    assert best_val < val_losses[0]


def test_training_pipeline_records_a_dataset_stage_abort_in_its_run_log(test_data_dir, tmp_path, monkeypatch):
    dataset_config                     = _dataset_config(test_data_dir)
    dataset_config.fingerprint_dataset = False

    pipeline = TrainingPipeline(trainer_config=_trainer_config(tmp_path), dataset_config=dataset_config, backbone_name="resunet", run_name="dataset_abort")

    monkeypatch.setattr("pipelines.backbone.training.pipeline.DatasetPipeline", Mock(side_effect=ValueError("dataset stage broke")))

    with pytest.raises(ValueError, match="dataset stage broke"):
        pipeline.run()

    text = (pipeline.run_metadata.run_directory / "logs" / "resunet_metadata.log").read_text(encoding="utf-8")

    assert "[Aborted] ValueError: dataset stage broke" in text
    assert "Traceback (most recent call last)" in text
    assert ">>> ABORTED resunet_metadata" in text
    assert ">>> END" not in text


def _square_patch_dataset_config() -> DatasetConfig:
    splits = SplitRegions(train=[CropRegion(0, 8, 0, 8)], val=[CropRegion(8, 16, 0, 8)], test=[CropRegion(16, 24, 0, 8)])

    return DatasetConfig(dataset_directory=Path("/x"), split_regions=splits, patch=PatchConfig(size=(64, 32), stride=(32, 16)))


def test_pipeline_squares_the_patch_for_square_only_backbones():
    pipeline = TrainingPipeline(trainer_config=BackboneTrainerConfig(), dataset_config=_square_patch_dataset_config(), backbone_name="swin_unet")

    assert pipeline.dataset_config.patch.size   == (64, 64)
    assert pipeline.dataset_config.patch.stride == (32, 32)
    assert pipeline.patch_size                  == (64, 64)
    assert pipeline.configured_patch            == ((64, 32), (32, 16))
    assert pipeline.patch_squared               is True


def test_pipeline_names_the_configured_and_effective_geometry_when_squared():
    pipeline = TrainingPipeline(trainer_config=BackboneTrainerConfig(), dataset_config=_square_patch_dataset_config(), backbone_name="swin_unet")

    assert pipeline._patch_geometry_label()   == "configured patch (64, 32) stride (32, 16) -> effective patch (64, 64) stride (32, 32) (swin_unet needs a square input)"
    assert pipeline._patch_geometry_payload() == {"configured_size": [64, 32], "configured_stride": [32, 16], "size": [64, 64], "stride": [32, 32], "squared": True}


def test_pipeline_keeps_rectangular_patches_for_convolutional_backbones():
    pipeline = TrainingPipeline(trainer_config=BackboneTrainerConfig(), dataset_config=_square_patch_dataset_config(), backbone_name="resunet")

    assert pipeline.dataset_config.patch.size   == (64, 32)
    assert pipeline.dataset_config.patch.stride == (32, 16)
    assert pipeline.patch_squared               is False
    assert pipeline._patch_geometry_label()     == "patch (64, 32) stride (32, 16)"
    assert pipeline._patch_geometry_payload()["squared"] is False
