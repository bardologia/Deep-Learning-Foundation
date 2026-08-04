from __future__ import annotations

import numpy as np
import pytest

from pipelines.shared.dataset.dataset_spatial        import Layout
from pipelines.profile_autoencoder.dataset.splitting import ParameterCropper
from tools.data.regions                              import CropRegion, SplitRegions
from tools.monitoring.logger                         import Logger


def _ranges(region: CropRegion) -> set[int]:
    return set(range(region.azimuth_start, region.azimuth_end))


def test_from_ratios_partitions_are_disjoint():
    crop   = CropRegion(0, 1000, 0, 500)
    splits = SplitRegions.from_ratios(crop, train_ratio=0.7, val_ratio=0.15)

    train, val, test = splits.train, splits.val, splits.test

    assert _ranges(train).isdisjoint(_ranges(val))
    assert _ranges(train).isdisjoint(_ranges(test))
    assert _ranges(val).isdisjoint(_ranges(test))


def test_from_ratios_cover_full_azimuth():
    crop   = CropRegion(0, 1000, 0, 500)
    splits = SplitRegions.from_ratios(crop, train_ratio=0.7, val_ratio=0.15)

    covered = _ranges(splits.train) | _ranges(splits.val) | _ranges(splits.test)

    assert covered == set(range(0, 1000))


def test_from_ratios_respects_ratios():
    crop   = CropRegion(0, 1000, 0, 500)
    splits = SplitRegions.from_ratios(crop, train_ratio=0.7, val_ratio=0.15)

    assert splits.train.azimuth_size == 700
    assert splits.val.azimuth_size   == 150
    assert splits.test.azimuth_size  == 150


def test_from_ratios_deterministic():
    crop = CropRegion(100, 900, 50, 300)

    a = SplitRegions.from_ratios(crop, 0.6, 0.2)
    b = SplitRegions.from_ratios(crop, 0.6, 0.2)

    assert a.train.as_tuple() == b.train.as_tuple()
    assert a.val.as_tuple()   == b.val.as_tuple()
    assert a.test.as_tuple()  == b.test.as_tuple()


def test_from_ratios_share_range_extent():
    crop   = CropRegion(0, 1000, 25, 475)
    splits = SplitRegions.from_ratios(crop)

    for region in (splits.train, splits.val, splits.test):
        assert region.range_start == 25
        assert region.range_end   == 475


@pytest.mark.real_data
def test_parameter_cropper_loads_disjoint_pixel_counts(test_data_dir, params_dir, tmp_path):
    logger = Logger(log_dir=str(tmp_path / "logs"), name="split_real", level="ERROR")
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

    assert train[0].shape == (15, 60, 60)
    assert val[0].shape   == (15, 20, 60)
    assert test[0].shape  == (15, 20, 60)
    assert cropper.profile_length() == 150


@pytest.mark.real_data
def test_parameter_cropper_regions_load_distinct_data(test_data_dir, params_dir, tmp_path):
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


@pytest.mark.real_data
def test_parameter_cropper_rejects_overlapping_splits(test_data_dir, params_dir, tmp_path):
    logger = Logger(log_dir=str(tmp_path / "logs"), name="split_overlap", level="ERROR")
    layout = Layout(test_data_dir, logger=logger, parameters_path=params_dir / "parameters.npy")

    splits = SplitRegions(
        train = CropRegion(1000, 1060, 500, 560),
        val   = CropRegion(1050, 1080, 500, 560),
        test  = CropRegion(1080, 1100, 500, 560),
    )

    with pytest.raises(ValueError, match="overlap"):
        ParameterCropper(layout, splits, logger=logger)
