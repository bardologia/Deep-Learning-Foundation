"""Tests for the profile parameter cropper over explicit row splits."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.shared.dataset.dataset_spatial        import Layout
from pipelines.profile_autoencoder.dataset.splitting import ParameterCropper
from tools.data.regions                              import CropRegion, SplitRegions
from tools.monitoring.logger                         import Logger


@pytest.mark.dataset
def test_parameter_cropper_loads_disjoint_pixel_counts(test_data_dir, params_dir, parameters, scene_config, tmp_path):
    """The cropper loads each split at its own row extent and reports the profile length."""
    logger = Logger(log_dir=str(tmp_path / "logs"), name="split_scene", level="ERROR")
    layout = Layout(test_data_dir, logger=logger, parameters_path=params_dir / "parameters.npy")

    splits = SplitRegions(
        train = CropRegion(1000, 1060, 500, 560),
        val   = CropRegion(1060, 1080, 500, 560),
        test  = CropRegion(1080, 1100, 500, 560),
    )

    cropper = ParameterCropper(layout, splits, logger=logger)

    train = cropper.load_split("train")
    val   = cropper.load_split("val")
    test  = cropper.load_split("test")

    assert train[0].shape == (parameters.shape[0], 60, 60)
    assert val[0].shape   == (parameters.shape[0], 20, 60)
    assert test[0].shape  == (parameters.shape[0], 20, 60)
    assert cropper.profile_length() == scene_config.profile_length


@pytest.mark.dataset
def test_parameter_cropper_regions_load_distinct_data(test_data_dir, params_dir, tmp_path):
    """Adjacent split regions load different parameter data."""
    logger = Logger(log_dir=str(tmp_path / "logs"), name="split_real2", level="ERROR")
    layout = Layout(test_data_dir, logger=logger, parameters_path=params_dir / "parameters.npy")

    splits = SplitRegions(
        train = CropRegion(1000, 1020, 500, 520),
        val   = CropRegion(1020, 1040, 500, 520),
        test  = CropRegion(1040, 1060, 500, 520),
    )

    cropper = ParameterCropper(layout, splits, logger=logger)

    train = cropper.load_split("train")[0]
    val   = cropper.load_split("val")[0]

    assert not np.array_equal(train, val)


@pytest.mark.dataset
def test_parameter_cropper_rejects_overlapping_splits(test_data_dir, params_dir, tmp_path):
    """Overlapping split regions raise ValueError."""
    logger = Logger(log_dir=str(tmp_path / "logs"), name="split_overlap", level="ERROR")
    layout = Layout(test_data_dir, logger=logger, parameters_path=params_dir / "parameters.npy")

    splits = SplitRegions(
        train = CropRegion(1000, 1060, 500, 560),
        val   = CropRegion(1050, 1080, 500, 560),
        test  = CropRegion(1080, 1100, 500, 560),
    )

    with pytest.raises(ValueError, match="overlap"):
        ParameterCropper(layout, splits, logger=logger)
