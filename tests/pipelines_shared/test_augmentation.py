
from __future__ import annotations

import numpy as np

from configuration.dataset                   import AugmentationConfig
from pipelines.backbone.dataset.augmentation import SpatialAugmenter
from tools.monitoring.logger                 import Logger


def _logger(tmp_path) -> Logger:
    return Logger(log_dir=str(tmp_path / "logs"), name="aug_test", level="ERROR")


def _spatial(tmp_path, **kw) -> SpatialAugmenter:
    return SpatialAugmenter(AugmentationConfig(**kw), logger=_logger(tmp_path), seed=0)


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
