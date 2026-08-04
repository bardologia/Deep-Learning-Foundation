from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from configuration.dataset               import DatasetConfig, InputConfig, PatchConfig, Representation, SplitRegions
from configuration.sar.gaussian_config   import GaussianConfig
from pipelines.backbone.dataset.pipeline import BackboneDatasetPreparation
from tools.data.regions                  import CropRegion
from tools.monitoring.logger             import Logger


def _dataset_config(test_data_dir, params_dir) -> DatasetConfig:
    ic = InputConfig(
        use_primary=True,        primary_representation=Representation.MAG_ONLY,
        use_secondaries=True,    secondaries_representation=Representation.MAG_ONLY,
        use_interferograms=True, interferograms_representation=Representation.ANGLE_ONLY,
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
        input_config                = ic,
        batch_size                  = 2,
        num_workers                 = 0,
        prefetch_factor             = 8,
        n_gaussians                 = 5,
    )


@pytest.mark.real_data
@pytest.mark.slow
def test_backbone_preparation_returns_loaders_axis_and_datasets(test_data_dir, params_dir, tmp_path):
    dataset_config = _dataset_config(test_data_dir, params_dir)
    gaussian       = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy")
    trainer_config = SimpleNamespace(gaussian=gaussian)
    logger         = Logger(log_dir=str(tmp_path / "logs"), name="prep", level="ERROR")

    prep = BackboneDatasetPreparation(dataset_config, trainer_config, tmp_path, logger, seed=0)

    loaders, datasets, x_axis, x_len = prep.run()

    assert len(loaders) == 3
    assert set(datasets.keys()) == {"train", "val", "test"}

    assert x_len == 150
    assert x_axis.shape == (150,)
    assert np.isclose(x_axis[0],  gaussian.x_min)
    assert np.isclose(x_axis[-1], gaussian.x_max)

    assert dataset_config.n_gaussians == gaussian.n_default_gaussians
    assert dataset_config.x_axis is x_axis


@pytest.mark.real_data
@pytest.mark.slow
def test_backbone_preparation_loader_batches_consistent(test_data_dir, params_dir, tmp_path):
    dataset_config = _dataset_config(test_data_dir, params_dir)
    gaussian       = GaussianConfig.from_dataset(test_data_dir, params_dir / "parameters.npy")
    trainer_config = SimpleNamespace(gaussian=gaussian)
    logger         = Logger(log_dir=str(tmp_path / "logs"), name="prep2", level="ERROR")

    prep = BackboneDatasetPreparation(dataset_config, trainer_config, tmp_path, logger, seed=0)
    (train_loader, val_loader, test_loader), datasets, _, _ = prep.run()

    x, y = next(iter(train_loader))

    assert x.shape[0] == 2
    assert x.shape[1] == datasets["train"].input_channels
    assert y.shape[1] == datasets["train"].gt_channels
    assert len(val_loader)  >= 1
    assert len(test_loader) >= 1
