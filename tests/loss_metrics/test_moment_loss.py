
from __future__ import annotations

import pytest
import torch

from tools.loss.curve_loss import MomentLoss


def _curves(seed: int, b=2, n=12, h=3, w=3) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(b, n, h, w, generator=gen, dtype=torch.float64) + 0.1


def _z_axis(n=12) -> torch.Tensor:
    return torch.linspace(-20.0, 80.0, n, dtype=torch.float64)


def test_masked_mean_basic():
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    assert MomentLoss.masked_mean(vals, mask).item() == 1.5


def test_masked_mean_empty_mask_no_div_zero():
    vals = torch.ones(4, dtype=torch.float64)
    mask = torch.zeros(4, dtype=torch.float64)
    assert MomentLoss.masked_mean(vals, mask).item() == 0.0


def test_valid_fraction_is_the_share_of_pixels_above_the_floor():
    target = _curves(10)
    z      = _z_axis()
    dx     = float(z[1] - z[0])

    target[:, :, 0, :] = 0.0

    mask = MomentLoss.valid_mask(target, dx, 1e-6)

    assert mask.shape == (2, 3, 3)
    assert torch.equal(mask, ((target.sum(dim=1) * dx) > 1e-6).to(target.dtype))
    assert MomentLoss.valid_fraction(target, dx, 1e-6).item() == pytest.approx(2.0 / 3.0)
    assert MomentLoss.valid_fraction(torch.zeros_like(target), dx, 1e-6).item() == 0.0
    assert MomentLoss.total_power(target.clone(), target, dx, 1e-6).item() == pytest.approx(0.0)


def test_terms_return_zero_when_no_pixel_clears_the_floor():
    pred   = _curves(11)
    target = torch.zeros_like(pred)
    z      = _z_axis()
    dx     = float(z[1] - z[0])

    assert MomentLoss.valid_fraction(target, dx, 1e-3).item() == 0.0
    assert MomentLoss.total_power(pred, target, dx, 1e-3).item() == 0.0
    assert MomentLoss.moments(pred, target, z, dx, 1e-3, (1.0, 1.0, 1.0)).item() == 0.0


def test_moment_sums_shapes_and_mass():
    curves = _curves(0)
    z      = _z_axis()
    dx     = float(z[1] - z[0])
    s0, s1, s2 = MomentLoss.moment_sums(curves, z, dx)
    assert s0.shape == (2, 3, 3)
    expected_s0 = curves.sum(dim=1) * dx
    assert torch.allclose(s0, expected_s0)


def test_total_power_zero_on_identical():
    target = _curves(1)
    z      = _z_axis()
    dx     = float(z[1] - z[0])
    out    = MomentLoss.total_power(target.clone(), target, dx, 1e-6)
    assert out.item() < 1e-9


def test_total_power_nonnegative():
    pred   = _curves(2)
    target = _curves(3)
    z      = _z_axis()
    dx     = float(z[1] - z[0])
    out    = MomentLoss.total_power(pred, target, dx, 1e-6)
    assert out.item() >= 0.0


def test_moments_zero_on_identical():
    target = _curves(4)
    z      = _z_axis()
    dx     = float(z[1] - z[0])
    out    = MomentLoss.moments(target.clone(), target, z, dx, 1e-6, (1.0, 1.0, 1.0))
    assert out.item() < 1e-7


def test_moments_nonnegative_and_scalar():
    pred   = _curves(5)
    target = _curves(6)
    z      = _z_axis()
    dx     = float(z[1] - z[0])
    out    = MomentLoss.moments(pred, target, z, dx, 1e-6, (1.0, 1.0, 1.0))
    assert out.ndim == 0
    assert out.item() >= 0.0


def test_moments_weight_normalisation_invariant():
    pred   = _curves(7)
    target = _curves(8)
    z      = _z_axis()
    dx     = float(z[1] - z[0])
    a      = MomentLoss.moments(pred, target, z, dx, 1e-6, (1.0, 1.0, 1.0))
    b      = MomentLoss.moments(pred, target, z, dx, 1e-6, (2.0, 2.0, 2.0))
    assert torch.allclose(a, b, atol=1e-9)
