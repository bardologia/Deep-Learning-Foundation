"""Tests for the profile autoencoder log1p normalizer and its fitted statistics."""

from __future__ import annotations

import json

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
    """Normalizing then denormalizing a numpy profile recovers the original curve."""
    norm  = ProfileNormalizer(ProfileStats(loc=0.3, scale=0.7))
    curve = np.abs(np.random.default_rng(0).standard_normal(150)).astype(np.float32) * 5.0

    back = norm.denormalize(norm.normalize(curve))

    np.testing.assert_allclose(back, curve, atol=1e-3)


def test_roundtrip_torch():
    """Normalizing then denormalizing a torch profile recovers the original curve."""
    norm  = ProfileNormalizer(ProfileStats(loc=0.1, scale=1.3))
    curve = torch.abs(torch.randn(150)) * 4.0

    back = norm.denormalize(norm.normalize(curve))

    assert torch.allclose(back, curve, atol=1e-3)


def test_normalized_output_matches_log1p_formula():
    """Profile normalization equals (log1p(x) - loc) / scale."""
    norm  = ProfileNormalizer(ProfileStats(loc=0.2, scale=0.5))
    curve = np.abs(np.random.default_rng(1).standard_normal(64)).astype(np.float32)

    out      = norm.normalize(curve)
    expected = (np.log1p(curve) - 0.2) / 0.5

    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_scale_floor_applied_for_zero_scale():
    """A zero fitted scale is raised to the scale floor, keeping the inverse finite."""
    norm = ProfileNormalizer(ProfileStats(loc=0.0, scale=0.0))

    assert norm.scale == ProfileNormalizer.SCALE_FLOOR
    assert np.isfinite(norm.inv_scale)


class _ConstantDataset:
    """Profile dataset stand-in whose every pixel renders the same curve.

    Attributes:
        z_axis : Z axis of shape (H,).
        index  : Pixel index served by the dataset.
        amps   : Amplitudes of shape (n_pixels, 1), all equal.
        mus    : Means of shape (n_pixels, 1), all equal.
        sigs   : Widths of shape (n_pixels, 1), all equal.
    """

    def __init__(self, n_pixels: int, amplitude: float):
        """Builds n_pixels identical Gaussians of the given amplitude."""
        self.z_axis = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        self.index  = np.arange(n_pixels)
        self.amps   = np.full((n_pixels, 1), amplitude, dtype=np.float32)
        self.mus    = np.full((n_pixels, 1), 0.5,       dtype=np.float32)
        self.sigs   = np.full((n_pixels, 1), 0.1,       dtype=np.float32)

    def __len__(self) -> int:
        """Returns the pixel count."""
        return int(self.index.shape[0])


def test_compute_refuses_a_near_constant_training_sample(tmp_path):
    """Curves with no variation after log1p raise, naming the sample size and the measured std, instead of persisting a floored scale."""
    logger = Logger(log_dir=str(tmp_path / "logs"), name="prof_norm", level="ERROR")

    with pytest.raises(ValueError, match=r"8 sampled training profiles .* below the scale floor"):
        ProfileStatsComputer.compute(_ConstantDataset(8, amplitude=0.0), logger, max_samples=8)

    logger.close()


def test_compute_keeps_the_measured_scale_when_curves_vary(tmp_path):
    """A sample with real variation returns its measured scale unchanged."""
    logger  = Logger(log_dir=str(tmp_path / "logs"), name="prof_norm", level="ERROR")
    dataset = _ConstantDataset(8, amplitude=5.0)

    stats = ProfileStatsComputer.compute(dataset, logger, max_samples=8)
    logger.close()

    assert stats.scale > ProfileNormalizer.SCALE_FLOOR
    assert np.isfinite(stats.loc)


@pytest.mark.parametrize("loc, scale", [(float("nan"), 1.0), (0.0, float("inf"))])
def test_non_finite_stats_are_rejected(loc, scale):
    """A NaN or infinite location or scale is rejected instead of normalizing to non-finite values."""
    with pytest.raises(ValueError, match="non-finite stats"):
        ProfileNormalizer(ProfileStats(loc=loc, scale=scale))


@pytest.mark.parametrize("loc, scale", [(0.0, -1.0), (0.0, 0.0), (float("nan"), 1.0)])
def test_load_rejects_corrupt_stats_file(tmp_path, loc, scale):
    """Loading a stats file with a non-finite location or a non-positive scale raises."""
    (tmp_path / ProfileStats.FILENAME).write_text(json.dumps({"loc": loc, "scale": scale}))

    with pytest.raises(ValueError, match="Corrupt profile normalization stats"):
        ProfileStats.load(tmp_path)


def test_normalize_dtype_is_float32_numpy():
    """Normalizing a float64 profile returns float32."""
    norm = ProfileNormalizer(ProfileStats(loc=0.0, scale=1.0))
    out  = norm.normalize(np.ones(10, dtype=np.float64))

    assert out.dtype == np.float32


@pytest.mark.dataset
@pytest.mark.slow
def test_fit_real_profiles_finite_and_roundtrip(parameters, tmp_path):
    """Statistics fitted on real profiles are finite, respect the scale floor, and round-trip."""
    params = [np.ascontiguousarray(np.asarray(parameters[:, :40, :40]))]
    z_axis = np.linspace(0.0, 1.0, 150, dtype=np.float32)
    logger = Logger(log_dir=str(tmp_path / "logs"), name="prof_norm", level="ERROR")

    ds = ProfileDataset(
        param_arrays=params, z_axis=z_axis, n_gaussians=params[0].shape[0] // 3, split_name="train",
        activity_threshold=1e-3, keep_empty_frac=0.05, pixel_subsample=1.0, seed=0, logger=logger,
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
