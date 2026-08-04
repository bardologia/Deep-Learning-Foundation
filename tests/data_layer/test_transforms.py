from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.data.transforms import Log1pTransform


def test_compress_numpy_matches_log1p():
    x = np.array([0.0, 1.0, 9.0, 100.0], dtype=np.float64)

    assert np.allclose(Log1pTransform.compress(x), np.log1p(x))


def test_compress_numpy_clamps_negatives():
    x = np.array([-5.0, -0.1, 0.0, 3.0])

    out = Log1pTransform.compress(x)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert np.isclose(out[3], np.log1p(3.0))


def test_compress_torch_matches_log1p():
    x   = torch.tensor([0.0, 1.0, 9.0, 100.0])
    out = Log1pTransform.compress(x)

    assert torch.allclose(out, torch.log1p(x))


def test_compress_torch_clamps_negatives():
    x   = torch.tensor([-2.0, 0.0, 7.0])
    out = Log1pTransform.compress(x)

    assert out[0].item() == 0.0


def test_roundtrip_numpy():
    x          = np.array([0.0, 0.5, 5.0, 50.0], dtype=np.float64)
    recovered  = Log1pTransform.decompress(Log1pTransform.compress(x))

    assert np.allclose(recovered, x, atol=1e-6)


def test_roundtrip_torch():
    x         = torch.tensor([0.0, 0.5, 5.0, 50.0])
    recovered = Log1pTransform.decompress(Log1pTransform.compress(x))

    assert torch.allclose(recovered, x, atol=1e-5)


def test_decompress_numpy_clamps_to_physical_ceil():
    big = np.array([1e9])
    out = Log1pTransform.decompress(big)

    assert np.isclose(out[0], Log1pTransform.CEIL)


def test_decompress_torch_clamps_to_physical_ceil():
    big = torch.tensor([1e9])
    out = Log1pTransform.decompress(big)

    assert torch.isclose(out[0], torch.tensor(Log1pTransform.CEIL))


def test_decompress_custom_ceil_is_physical():
    values = np.log1p(np.array([10.0, 400.0]))
    out    = Log1pTransform.decompress(values, floor=0.0, ceil=100.0)

    assert np.isclose(out[0], 10.0)
    assert np.isclose(out[1], 100.0)


def test_decompress_clamps_negatives():
    out_np = Log1pTransform.decompress(np.array([-3.0]))
    out_t  = Log1pTransform.decompress(torch.tensor([-3.0]))

    assert out_np[0] == 0.0
    assert out_t[0].item() == 0.0


def test_decompress_leaky_slope_leaks_past_ceiling():
    hi  = np.log1p(Log1pTransform.CEIL)
    out = Log1pTransform.decompress(np.array([hi + 1.0]), leaky_slope=0.1)

    assert out[0] > Log1pTransform.CEIL


def test_decompress_leaky_slope_keeps_gradient_above_ceiling():
    x = torch.tensor([np.log1p(Log1pTransform.CEIL) + 2.0], requires_grad=True)

    Log1pTransform.decompress(x, leaky_slope=0.01).sum().backward()

    assert x.grad[0].item() > 0.0


def test_decompress_leaky_slope_keeps_gradient_below_floor():
    x = torch.tensor([-3.0], requires_grad=True)

    Log1pTransform.decompress(x, leaky_slope=0.01).sum().backward()

    assert x.grad[0].item() > 0.0


def test_decompress_leaky_slope_identity_inside_bounds():
    inside = np.log1p(np.array([1.0, 150.0]))

    hard  = Log1pTransform.decompress(inside)
    leaky = Log1pTransform.decompress(inside, leaky_slope=0.1)

    assert np.allclose(hard, leaky)


def test_compress_leaky_slope_leaks_below_floor():
    out = Log1pTransform.compress(np.array([-2.0]), leaky_slope=0.1)

    assert out[0] < 0.0


def test_compress_leaky_slope_keeps_gradient_below_floor():
    x = torch.tensor([-2.0], requires_grad=True)

    Log1pTransform.compress(x, leaky_slope=0.01).sum().backward()

    assert x.grad[0].item() > 0.0


def test_compress_leaky_slope_identity_for_nonnegative():
    x = np.array([0.0, 1.0, 50.0])

    assert np.allclose(Log1pTransform.compress(x, leaky_slope=0.1), Log1pTransform.compress(x))


def test_compress_monotonic():
    x   = np.linspace(0.0, 100.0, 50)
    out = Log1pTransform.compress(x)

    assert np.all(np.diff(out) > 0)


@pytest.mark.real_data
def test_compress_real_magnitude_window(secondaries):
    mag        = np.abs(np.asarray(secondaries[0, :32, :32])).astype(np.float64)
    compressed = Log1pTransform.compress(mag)

    assert compressed.shape == mag.shape
    assert np.all(compressed >= 0.0)
    assert np.all(np.isfinite(compressed))
    assert np.all(compressed <= np.log1p(mag.max()) + 1e-9)
