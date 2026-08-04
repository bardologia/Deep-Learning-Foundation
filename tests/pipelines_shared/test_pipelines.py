from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.dataset                          import DatasetConfig, InputConfig, PatchConfig, ProfileDatasetConfig, Representation, SplitRegions
from pipelines.backbone.dataset.pipeline            import DatasetPipeline
from pipelines.profile_autoencoder.dataset.pipeline import ProfileDatasetPipeline
from tools.data.regions                             import CropRegion
from tools.monitoring.logger                        import Logger


def _logger(tmp_path, name) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name=name, level="ERROR")


def _backbone_config(test_data_dir, params_dir) -> DatasetConfig:
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
        shuffle_train               = True,
        n_gaussians                 = 5,
    )


@pytest.mark.real_data
@pytest.mark.slow
def test_backbone_pipeline_writes_metadata_and_stats(test_data_dir, params_dir, tmp_path):
    config   = _backbone_config(test_data_dir, params_dir)
    pipeline = DatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "bb_pipe"), seed=0)

    train_loader, val_loader, test_loader, datasets = pipeline.run()

    meta = tmp_path / "meta"

    assert (meta / "normalization_stats.json").is_file()
    assert (meta / "dataset_creation_config.json").is_file()
    assert (meta / "crop.json").is_file()
    assert (meta / "patch.json").is_file()

    assert pipeline.profile_length == 150
    assert datasets["train"].normalizer is not None
    assert len(train_loader) >= 1


@pytest.mark.real_data
@pytest.mark.slow
def test_backbone_pipeline_train_order_reproducible(test_data_dir, params_dir, tmp_path):
    torch.use_deterministic_algorithms(False)

    def _first_batch(run_dir):
        config   = _backbone_config(test_data_dir, params_dir)
        pipeline = DatasetPipeline(config, run_dir, logger=_logger(run_dir, "bb_repro"), seed=7)
        loaders  = pipeline.run()
        x, _     = next(iter(loaders[0]))
        return x.numpy()

    a = _first_batch(tmp_path / "a")
    b = _first_batch(tmp_path / "b")

    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.real_data
@pytest.mark.slow
def test_backbone_pipeline_rectangular_patches_yield_rectangular_batches(test_data_dir, params_dir, tmp_path):
    config       = _backbone_config(test_data_dir, params_dir)
    config.patch = PatchConfig(size=(8, 16), stride=(4, 8))

    pipeline = DatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "bb_rect"), seed=0)

    train_loader, _val_loader, _test_loader, _datasets = pipeline.run()
    x, y = next(iter(train_loader))

    assert tuple(x.shape[-2:]) == (8, 16)
    assert tuple(y.shape[-2:]) == (8, 16)


@pytest.mark.real_data
@pytest.mark.slow
def test_backbone_pipeline_crops_only_the_enabled_input_groups(test_data_dir, params_dir, tmp_path):
    full    = _backbone_config(test_data_dir, params_dir)
    trimmed = _backbone_config(test_data_dir, params_dir)

    trimmed.input_config = InputConfig(
        use_primary=False,
        use_secondaries=False,
        use_interferograms=True, interferograms_representation=Representation.ANGLE_ONLY,
    )

    full_datasets    = DatasetPipeline(full,    tmp_path / "full",    logger=_logger(tmp_path, "bb_full"),    seed=0).run()[3]
    trimmed_datasets = DatasetPipeline(trimmed, tmp_path / "trimmed", logger=_logger(tmp_path, "bb_trimmed"), seed=0).run()[3]

    full_train    = full_datasets["train"]
    trimmed_train = trimmed_datasets["train"]

    assert (full_train.n_primary, full_train.n_secondaries, full_train.n_interferograms)          == (1, 2, 2)
    assert (trimmed_train.n_primary, trimmed_train.n_secondaries, trimmed_train.n_interferograms) == (0, 0, 2)

    assert full_train.inputs.shape[0]    == 5
    assert trimmed_train.inputs.shape[0] == 2
    assert trimmed_train.input_channels  == 2

    np.testing.assert_array_equal(trimmed_train.inputs, full_train.inputs[3:])


@pytest.mark.real_data
def test_backbone_pipeline_rejects_rot90_with_rectangular_patch(test_data_dir, params_dir, tmp_path):
    config                      = _backbone_config(test_data_dir, params_dir)
    config.patch                = PatchConfig(size=(8, 16), stride=(4, 8))
    config.augmentation.p_rot90 = 0.5

    with pytest.raises(ValueError, match="rectangular patch"):
        DatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "bb_rot90"), seed=0)


@pytest.mark.real_data
@pytest.mark.slow
def test_profile_pipeline_runs_and_returns_normalizer(test_data_dir, params_dir, tmp_path):
    config = ProfileDatasetConfig(
        preprocessing_run_directory = test_data_dir,
        parameters_path             = params_dir / "parameters.npy",
        split_regions               = SplitRegions(
            train = CropRegion(1000, 1040, 500, 540),
            val   = CropRegion(1040, 1060, 500, 540),
            test  = CropRegion(1060, 1080, 500, 540),
        ),
        n_gaussians       = 5,
        batch_size        = 16,
        num_workers       = 0,
        prefetch_factor   = 2,
        stats_max_samples = 500,
    )

    pipeline = ProfileDatasetPipeline(config, tmp_path, logger=_logger(tmp_path, "prof_pipe"), seed=0)

    (train_loader, val_loader, test_loader), datasets, x_axis, x_len, normalizer = pipeline.run()

    assert x_len == 150
    assert x_axis.shape == (150,)
    assert (tmp_path / "meta" / "profile_normalization_stats.json").is_file()
    assert np.isfinite(normalizer.loc)
    assert normalizer.scale >= normalizer.SCALE_FLOOR

    batch = next(iter(train_loader))
    assert batch.shape[1] == 150
    assert len(val_loader)  >= 1
    assert len(test_loader) >= 1


@pytest.mark.real_data
@pytest.mark.slow
def test_profile_pipeline_train_order_reproducible(test_data_dir, params_dir, tmp_path):
    def _first_batch(run_dir):
        config = ProfileDatasetConfig(
            preprocessing_run_directory = test_data_dir,
            parameters_path             = params_dir / "parameters.npy",
            split_regions               = SplitRegions(
                train = CropRegion(1000, 1040, 500, 540),
                val   = CropRegion(1040, 1060, 500, 540),
                test  = CropRegion(1060, 1080, 500, 540),
            ),
            n_gaussians       = 5,
            batch_size        = 16,
            num_workers       = 0,
            prefetch_factor   = 2,
            stats_max_samples = 500,
        )
        pipeline = ProfileDatasetPipeline(config, run_dir, logger=_logger(run_dir, "prof_repro"), seed=3)
        loaders  = pipeline.run()[0]
        return next(iter(loaders[0])).numpy()

    a = _first_batch(tmp_path / "a")
    b = _first_batch(tmp_path / "b")

    np.testing.assert_allclose(a, b, atol=1e-5)
