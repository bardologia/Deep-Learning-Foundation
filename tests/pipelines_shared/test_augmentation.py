from __future__ import annotations

import numpy as np

from configuration.dataset                              import AugmentationConfig, ProfileAugmentationConfig
from pipelines.backbone.dataset.augmentation            import SpatialAugmenter
from pipelines.profile_autoencoder.dataset.augmentation import ProfileAugmenter
from tools.monitoring.logger                            import Logger


def _logger(tmp_path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="aug_test", level="ERROR")


def _spatial(tmp_path, **kw) -> SpatialAugmenter:
    return SpatialAugmenter(AugmentationConfig(**kw), logger=_logger(tmp_path), seed=0)


def _profile(tmp_path, **kw) -> ProfileAugmenter:
    return ProfileAugmenter(ProfileAugmentationConfig(**kw), logger=_logger(tmp_path), seed=0)


def test_spatial_identity_when_all_probabilities_zero(tmp_path):
    aug = _spatial(tmp_path, p_flip_h=0.0, p_flip_v=0.0, p_rot90=0.0, p_noise=0.0)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    y   = np.random.default_rng(1).standard_normal((3, 8, 8)).astype(np.float32)

    xa, ya = aug(x, y)

    np.testing.assert_array_equal(xa, x)
    np.testing.assert_array_equal(ya, y)


def test_spatial_flip_preserves_shape(tmp_path):
    aug = _spatial(tmp_path, p_flip_h=1.0, p_flip_v=1.0, p_rot90=0.0, p_noise=0.0)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    y   = np.random.default_rng(1).standard_normal((3, 8, 8)).astype(np.float32)

    xa, ya = aug(x, y)

    assert xa.shape == x.shape
    assert ya.shape == y.shape


def test_spatial_double_horizontal_flip_is_identity(tmp_path):
    x = np.random.default_rng(0).standard_normal((2, 8, 8)).astype(np.float32)

    flipped  = x[..., :, ::-1]
    restored = flipped[..., :, ::-1]

    np.testing.assert_array_equal(restored, x)


def test_spatial_rot90_keeps_square_shape(tmp_path):
    aug = _spatial(tmp_path, p_flip_h=0.0, p_flip_v=0.0, p_rot90=1.0, p_noise=0.0)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    y   = np.random.default_rng(1).standard_normal((3, 8, 8)).astype(np.float32)

    xa, ya = aug(x, y)

    assert xa.shape == x.shape
    assert ya.shape == y.shape


def test_spatial_deterministic_for_fixed_seed(tmp_path):
    x = np.random.default_rng(5).standard_normal((4, 8, 8)).astype(np.float32)
    y = np.random.default_rng(6).standard_normal((3, 8, 8)).astype(np.float32)

    a1 = _spatial(tmp_path, p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.5)
    a2 = _spatial(tmp_path, p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.5)

    out1 = a1(x.copy(), y.copy())
    out2 = a2(x.copy(), y.copy())

    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])


def test_spatial_flip_applies_to_geometry(tmp_path):
    aug = _spatial(tmp_path, p_flip_h=1.0, p_flip_v=0.0, p_rot90=0.0, p_noise=0.0)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    y   = np.random.default_rng(1).standard_normal((3, 8, 8)).astype(np.float32)
    g   = np.random.default_rng(2).standard_normal((5, 8, 8)).astype(np.float32)

    xa, ya, ga = aug(x, y, g)

    np.testing.assert_array_equal(xa, x[..., :, ::-1])
    np.testing.assert_array_equal(ga, g[..., :, ::-1])


def test_spatial_rot90_skipped_when_geometry_present(tmp_path):
    aug = _spatial(tmp_path, p_flip_h=0.0, p_flip_v=0.0, p_rot90=1.0, p_noise=0.0)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    y   = np.random.default_rng(1).standard_normal((3, 8, 8)).astype(np.float32)
    g   = np.random.default_rng(2).standard_normal((5, 8, 8)).astype(np.float32)

    xa, ya, ga = aug(x, y, g)

    np.testing.assert_array_equal(xa, x)
    np.testing.assert_array_equal(ga, g)


def test_spatial_geometry_returns_three_tuple(tmp_path):
    aug = _spatial(tmp_path, p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.0, p_noise=0.0)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    y   = np.random.default_rng(1).standard_normal((3, 8, 8)).astype(np.float32)
    g   = np.random.default_rng(2).standard_normal((5, 8, 8)).astype(np.float32)

    out = aug(x, y, g)

    assert len(out) == 3
    assert out[2].shape == g.shape


def test_spatial_noise_zero_probability_is_identity(tmp_path):
    aug = _spatial(tmp_path, p_noise=0.0)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)

    np.testing.assert_array_equal(aug.add_noise(x), x)


def test_spatial_noise_bounded_by_std(tmp_path):
    aug = _spatial(tmp_path, p_noise=1.0, noise_std=0.01)
    x   = np.zeros((4, 64, 64), dtype=np.float32)

    out = aug.add_noise(x)

    assert out.shape == x.shape
    assert np.std(out) < 0.05


def test_spatial_reseed_resets_stream(tmp_path):
    aug = _spatial(tmp_path, p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.5)
    x   = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    y   = np.random.default_rng(1).standard_normal((3, 8, 8)).astype(np.float32)

    first = aug(x.copy(), y.copy())
    aug.reseed(0)
    again = aug(x.copy(), y.copy())

    np.testing.assert_array_equal(first[0], again[0])


def test_profile_identity_when_all_probabilities_zero(tmp_path):
    aug   = _profile(tmp_path, p_amp_scale=0.0, p_shift=0.0, p_flip=0.0, p_noise=0.0)
    curve = np.random.default_rng(0).standard_normal(150).astype(np.float32)

    np.testing.assert_array_equal(aug(curve), curve)


def test_profile_shift_preserves_length_and_values(tmp_path):
    aug   = _profile(tmp_path, p_amp_scale=0.0, p_shift=1.0, max_shift=4, p_flip=0.0, p_noise=0.0)
    curve = np.arange(150, dtype=np.float32)

    out = aug(curve)

    assert out.shape == curve.shape
    assert np.allclose(np.sort(out), np.sort(curve))


def test_profile_amplitude_scale_within_range(tmp_path):
    aug   = _profile(tmp_path, p_amp_scale=1.0, amp_scale_range=(0.5, 2.0), p_shift=0.0, p_flip=0.0, p_noise=0.0)
    curve = np.ones(150, dtype=np.float32)

    out = aug(curve)

    assert np.all(out >= 0.5 - 1e-5)
    assert np.all(out <= 2.0 + 1e-5)


def test_profile_flip_reverses_curve(tmp_path):
    aug   = _profile(tmp_path, p_amp_scale=0.0, p_shift=0.0, p_flip=1.0, p_noise=0.0)
    curve = np.arange(150, dtype=np.float32)

    out = aug(curve)

    np.testing.assert_array_equal(out, curve[::-1])


def test_profile_deterministic_for_fixed_seed(tmp_path):
    curve = np.random.default_rng(7).standard_normal(150).astype(np.float32)

    a1 = _profile(tmp_path, p_amp_scale=0.5, p_shift=0.5, p_flip=0.5)
    a2 = _profile(tmp_path, p_amp_scale=0.5, p_shift=0.5, p_flip=0.5)

    np.testing.assert_array_equal(a1(curve.copy()), a2(curve.copy()))


def test_profile_output_is_contiguous_float32(tmp_path):
    aug   = _profile(tmp_path, p_flip=1.0)
    curve = np.arange(150, dtype=np.float32)

    out = aug(curve)

    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
