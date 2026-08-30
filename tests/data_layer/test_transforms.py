"""Tests for the log1p amplitude compression used on SAR magnitudes.

Covers numpy and torch parity with log1p, clamping at the physical floor and
ceiling, round-trip fidelity, the leaky-slope variants that keep gradients
alive outside the bounds, monotonicity, and behaviour on a real amplitude window.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.data.transforms import Log1pTransform


def test_compress_numpy_matches_log1p():
    """Numpy compression of nonnegative amplitudes equals np.log1p."""
    x = np.array([0.0, 1.0, 9.0, 100.0], dtype=np.float64)

    assert np.allclose(Log1pTransform.compress(x), np.log1p(x))


def test_compress_numpy_clamps_negatives():
    """Negative amplitudes compress to zero while positives follow log1p."""
    x = np.array([-5.0, -0.1, 0.0, 3.0])

    out = Log1pTransform.compress(x)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert np.isclose(out[3], np.log1p(3.0))


def test_compress_torch_matches_log1p():
    """Torch compression of nonnegative amplitudes equals torch.log1p."""
    x   = torch.tensor([0.0, 1.0, 9.0, 100.0])
    out = Log1pTransform.compress(x)

    assert torch.allclose(out, torch.log1p(x))


def test_compress_torch_clamps_negatives():
    """Negative torch amplitudes compress to zero."""
    x   = torch.tensor([-2.0, 0.0, 7.0])
    out = Log1pTransform.compress(x)

    assert out[0].item() == 0.0


def test_roundtrip_numpy():
    """Decompressing a compressed numpy array recovers the original amplitudes."""
    x          = np.array([0.0, 0.5, 5.0, 50.0], dtype=np.float64)
    recovered  = Log1pTransform.decompress(Log1pTransform.compress(x))

    assert np.allclose(recovered, x, atol=1e-6)


def test_roundtrip_torch():
    """Decompressing a compressed torch tensor recovers the original amplitudes."""
    x         = torch.tensor([0.0, 0.5, 5.0, 50.0])
    recovered = Log1pTransform.decompress(Log1pTransform.compress(x))

    assert torch.allclose(recovered, x, atol=1e-5)


def test_decompress_numpy_clamps_to_physical_ceil():
    """A huge compressed numpy value decompresses to the physical amplitude ceiling."""
    big = np.array([1e9])
    out = Log1pTransform.decompress(big)

    assert np.isclose(out[0], Log1pTransform.CEIL)


def test_decompress_torch_clamps_to_physical_ceil():
    """A huge compressed torch value decompresses to the physical amplitude ceiling."""
    big = torch.tensor([1e9])
    out = Log1pTransform.decompress(big)

    assert torch.isclose(out[0], torch.tensor(Log1pTransform.CEIL))


def test_decompress_custom_ceil_is_physical():
    """An explicit ceiling clamps in amplitude units, not in compressed units."""
    values = np.log1p(np.array([10.0, 400.0]))
    out    = Log1pTransform.decompress(values, floor=0.0, ceil=100.0)

    assert np.isclose(out[0], 10.0)
    assert np.isclose(out[1], 100.0)


def test_decompress_clamps_negatives():
    """Negative compressed values decompress to zero in both numpy and torch."""
    out_np = Log1pTransform.decompress(np.array([-3.0]))
    out_t  = Log1pTransform.decompress(torch.tensor([-3.0]))

    assert out_np[0] == 0.0
    assert out_t[0].item() == 0.0


def test_decompress_leaky_slope_leaks_past_ceiling():
    """A nonzero leaky slope lets decompressed values exceed the ceiling."""
    hi  = np.log1p(Log1pTransform.CEIL)
    out = Log1pTransform.decompress(np.array([hi + 1.0]), leaky_slope=0.1)

    assert out[0] > Log1pTransform.CEIL


def test_decompress_leaky_slope_keeps_gradient_above_ceiling():
    """Decompression above the ceiling keeps a positive gradient under a leaky slope."""
    x = torch.tensor([np.log1p(Log1pTransform.CEIL) + 2.0], requires_grad=True)

    Log1pTransform.decompress(x, leaky_slope=0.01).sum().backward()

    assert x.grad[0].item() > 0.0


def test_decompress_leaky_slope_keeps_gradient_below_floor():
    """Decompression below the floor keeps a positive gradient under a leaky slope."""
    x = torch.tensor([-3.0], requires_grad=True)

    Log1pTransform.decompress(x, leaky_slope=0.01).sum().backward()

    assert x.grad[0].item() > 0.0


def test_decompress_leaky_slope_identity_inside_bounds():
    """The leaky slope leaves values inside the physical bounds unchanged."""
    inside = np.log1p(np.array([1.0, 150.0]))

    hard  = Log1pTransform.decompress(inside)
    leaky = Log1pTransform.decompress(inside, leaky_slope=0.1)

    assert np.allclose(hard, leaky)


def test_compress_leaky_slope_leaks_below_floor():
    """A nonzero leaky slope lets compression of a negative amplitude go below zero."""
    out = Log1pTransform.compress(np.array([-2.0]), leaky_slope=0.1)

    assert out[0] < 0.0


def test_compress_leaky_slope_keeps_gradient_below_floor():
    """Compression of a negative amplitude keeps a positive gradient under a leaky slope."""
    x = torch.tensor([-2.0], requires_grad=True)

    Log1pTransform.compress(x, leaky_slope=0.01).sum().backward()

    assert x.grad[0].item() > 0.0


def test_compress_leaky_slope_identity_for_nonnegative():
    """The leaky slope leaves compression of nonnegative amplitudes unchanged."""
    x = np.array([0.0, 1.0, 50.0])

    assert np.allclose(Log1pTransform.compress(x, leaky_slope=0.1), Log1pTransform.compress(x))


def test_compress_monotonic():
    """Compression is strictly increasing over an amplitude ramp."""
    x   = np.linspace(0.0, 100.0, 50)
    out = Log1pTransform.compress(x)

    assert np.all(np.diff(out) > 0)


@pytest.mark.dataset
def test_compress_scene_magnitude_window(secondaries):
    """Compressing a scene amplitude window preserves shape and stays finite, nonnegative and bounded by log1p of the maximum."""
    mag        = np.abs(np.asarray(secondaries[0, :32, :32])).astype(np.float64)
    compressed = Log1pTransform.compress(mag)

    assert compressed.shape == mag.shape
    assert np.all(compressed >= 0.0)
    assert np.all(np.isfinite(compressed))
    assert np.all(compressed <= np.log1p(mag.max()) + 1e-9)
