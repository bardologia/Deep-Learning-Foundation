from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.profile_autoencoder.dataset.datasets import ProfileDataset
from pipelines.profile_autoencoder.dataset.normalization import (
    ProfileNormalizer,
    ProfileStats,
    ProfileStatsComputer,
)
from tools.monitoring.logger import Logger


def test_roundtrip_numpy():
    norm  = ProfileNormalizer(ProfileStats(loc=0.3, scale=0.7))
    curve = np.abs(np.random.default_rng(0).standard_normal(150)).astype(np.float32) * 5.0

    back = norm.denormalize(norm.normalize(curve))

    np.testing.assert_allclose(back, curve, atol=1e-3)


def test_roundtrip_torch():
    norm  = ProfileNormalizer(ProfileStats(loc=0.1, scale=1.3))
    curve = torch.abs(torch.randn(150)) * 4.0

    back = norm.denormalize(norm.normalize(curve))

    assert torch.allclose(back, curve, atol=1e-3)


def test_normalized_output_matches_log1p_formula():
    norm  = ProfileNormalizer(ProfileStats(loc=0.2, scale=0.5))
    curve = np.abs(np.random.default_rng(1).standard_normal(64)).astype(np.float32)

    out      = norm.normalize(curve)
    expected = (np.log1p(curve) - 0.2) / 0.5

    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_scale_floor_applied_for_zero_scale():
    norm = ProfileNormalizer(ProfileStats(loc=0.0, scale=0.0))

    assert norm.scale == ProfileNormalizer.SCALE_FLOOR
    assert np.isfinite(norm.inv_scale)


def test_normalize_dtype_is_float32_numpy():
    norm = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    out  = norm.normalize(np.ones(10, dtype=np.float64))

    assert out.dtype == np.float32


@pytest.mark.real_data
@pytest.mark.slow
def test_fit_real_profiles_finite_and_roundtrip(parameters, tmp_path):
    params = [np.ascontiguousarray(np.asarray(parameters[:, :40, :40]))]
    x_axis = np.linspace(0.0, 1.0, 150, dtype=np.float32)
    logger = Logger(log_dir=str(tmp_path / "logs"), name="prof_norm", level="ERROR")

    ds = ProfileDataset(
        param_arrays=params, x_axis=x_axis, n_gaussians=5, split_name="train",
        amp_zero_thr=1e-3, keep_empty_frac=0.05, pixel_subsample=1.0, seed=0, logger=logger,
    )

    stats = ProfileStatsComputer.compute(ds, logger, max_samples=500)

    assert np.isfinite(stats.loc)
    assert np.isfinite(stats.scale)
    assert stats.scale >= ProfileNormalizer.SCALE_FLOOR

    norm = ProfileNormalizer(stats)
    raw  = np.asarray(ds[0])

    out  = norm.normalize(raw)
    back = norm.denormalize(out)

    assert np.all(np.isfinite(raw))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(back, raw, atol=1e-3)
