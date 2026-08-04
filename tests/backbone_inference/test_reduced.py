from __future__ import annotations

import types

import numpy as np
import pytest

from tools.data.regions                   import CropRegion
from pipelines.backbone.inference.reduced import ReducedTomogramSynthesizer

from tests.conftest import SilentLogger


def _make_synth(x_axis_length: int, region: CropRegion, secondary_labels=None) -> ReducedTomogramSynthesizer:
    synth        = ReducedTomogramSynthesizer.__new__(ReducedTomogramSynthesizer)
    synth._run   = types.SimpleNamespace(
        x_axis_length    = x_axis_length,
        split_region     = region,
        secondary_labels = secondary_labels,
    )
    synth.logger = SilentLogger()
    return synth


def test_validate_alignment_match():
    region  = CropRegion(azimuth_start=0, azimuth_end=8, range_start=0, range_end=6)
    synth   = _make_synth(x_axis_length=10, region=region)
    reduced = np.zeros((10, 8, 6), dtype=np.float32)

    synth._validate_alignment(reduced)


def test_validate_alignment_elevation_mismatch_raises():
    region  = CropRegion(azimuth_start=0, azimuth_end=8, range_start=0, range_end=6)
    synth   = _make_synth(x_axis_length=10, region=region)
    reduced = np.zeros((7, 8, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="elevation bins"):
        synth._validate_alignment(reduced)


def test_validate_alignment_spatial_mismatch_raises():
    region  = CropRegion(azimuth_start=0, azimuth_end=8, range_start=0, range_end=6)
    synth   = _make_synth(x_axis_length=10, region=region)
    reduced = np.zeros((10, 9, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="spatial shape"):
        synth._validate_alignment(reduced)


def test_cache_key_is_deterministic_and_sorted():
    region = CropRegion(azimuth_start=0, azimuth_end=8, range_start=0, range_end=6)
    synth  = _make_synth(x_axis_length=10, region=region)

    key_a = synth._cache_key([3, 1, 2], region)
    key_b = synth._cache_key([1, 2, 3], region)

    assert key_a == key_b
    assert "sel3-1-2-3" in key_a
    assert region.as_identifier_string() in key_a


def test_build_spec_carries_track_selection_and_crop():
    region = CropRegion(azimuth_start=2, azimuth_end=10, range_start=4, range_end=10)
    synth  = _make_synth(x_axis_length=10, region=region)
    synth.cfg = types.SimpleNamespace(
        reduced_pyrat_dir = None,
        reduced_effort    = "high",
        run_directory     = "/tmp/run",
    )

    state = {
        "tomogram_config"  : {"height_range": [-20.0, 80.0]},
        "stack_identifier" : "stackX",
        "dataset_type"     : "L",
        "paths"            : {"pyrat_directory": "/pyrat"},
    }

    spec = synth._build_spec(state, [1, 4], region, tomogram_path="/t.npy", dem_path="/d.npy")

    assert spec["tomogram_config"]["track_selection"] == [1, 4]
    assert spec["pyrat_directory"]  == "/pyrat"
    assert spec["crop"]             == list(region.as_tuple())
    assert spec["effort"]           == "high"
    assert spec["stack_identifier"] == "stackX"


def test_report_orientation_returns_persistable_correlations():
    region = CropRegion(azimuth_start=0, azimuth_end=4, range_start=0, range_end=4)
    synth  = _make_synth(x_axis_length=20, region=region)

    x       = np.linspace(0, 1, 20)
    profile = np.exp(-((x - 0.3) ** 2) / 0.01).astype(np.float32)
    gt      = np.broadcast_to(profile[:, None, None], (20, 4, 4)).astype(np.float32)
    reduced = gt.copy()

    orientation = synth._report_orientation(reduced, gt)

    assert orientation["reduced_orientation_corr_aligned"] == pytest.approx(1.0, abs=1e-5)
    assert orientation["reduced_orientation_corr_aligned"] > orientation["reduced_orientation_corr_flipped"]


def test_report_orientation_detects_flipped_axis():
    region = CropRegion(azimuth_start=0, azimuth_end=4, range_start=0, range_end=4)
    synth  = _make_synth(x_axis_length=20, region=region)

    x       = np.linspace(0, 1, 20)
    profile = np.exp(-((x - 0.3) ** 2) / 0.01).astype(np.float32)
    gt      = np.broadcast_to(profile[:, None, None], (20, 4, 4)).astype(np.float32)
    reduced = gt[::-1].copy()

    orientation = synth._report_orientation(reduced, gt)

    assert orientation["reduced_orientation_corr_flipped"] > orientation["reduced_orientation_corr_aligned"]
