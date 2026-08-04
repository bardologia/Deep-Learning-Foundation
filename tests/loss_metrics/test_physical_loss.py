from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.loss.physical_loss import PhysicalLoss


def _curves(seed: int, b=2, n=12, h=3, w=3) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(b, n, h, w, generator=gen, dtype=torch.float64) + 0.1


def _x_axis(n=12) -> torch.Tensor:
    return torch.linspace(-20.0, 80.0, n, dtype=torch.float64)


def _steering(n_tracks=6, n=12) -> torch.Tensor:
    gen   = torch.Generator().manual_seed(99)
    phase = torch.rand(n_tracks, n, generator=gen, dtype=torch.float64) * 6.0
    return torch.polar(torch.ones_like(phase), phase)


def _outer(steering: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ik,jk->ijk", steering, steering.conj())


def test_masked_mean_basic():
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    assert PhysicalLoss.masked_mean(vals, mask).item() == 1.5


def test_masked_mean_empty_mask_no_div_zero():
    vals = torch.ones(4, dtype=torch.float64)
    mask = torch.zeros(4, dtype=torch.float64)
    assert PhysicalLoss.masked_mean(vals, mask).item() == 0.0


def test_moment_sums_shapes_and_mass():
    curves = _curves(0)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    s0, s1, s2 = PhysicalLoss.moment_sums(curves, x, dx)
    assert s0.shape == (2, 3, 3)
    expected_s0 = curves.sum(dim=1) * dx
    assert torch.allclose(s0, expected_s0)


def test_total_power_zero_on_identical():
    target = _curves(1)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    out    = PhysicalLoss.total_power(target.clone(), target, dx, 1e-6)
    assert out.item() < 1e-9


def test_total_power_nonnegative():
    pred   = _curves(2)
    target = _curves(3)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    out    = PhysicalLoss.total_power(pred, target, dx, 1e-6)
    assert out.item() >= 0.0


def test_moments_zero_on_identical():
    target = _curves(4)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    out    = PhysicalLoss.moments(target.clone(), target, x, dx, 1e-6, (1.0, 1.0, 1.0))
    assert out.item() < 1e-7


def test_moments_nonnegative_and_scalar():
    pred   = _curves(5)
    target = _curves(6)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    out    = PhysicalLoss.moments(pred, target, x, dx, 1e-6, (1.0, 1.0, 1.0))
    assert out.ndim == 0
    assert out.item() >= 0.0


def test_moments_weight_normalisation_invariant():
    pred   = _curves(7)
    target = _curves(8)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    a      = PhysicalLoss.moments(pred, target, x, dx, 1e-6, (1.0, 1.0, 1.0))
    b      = PhysicalLoss.moments(pred, target, x, dx, 1e-6, (2.0, 2.0, 2.0))
    assert torch.allclose(a, b, atol=1e-9)


def test_coherence_resynthesis_zero_on_identical():
    target   = _curves(9)
    x        = _x_axis()
    dx       = float(x[1] - x[0])
    steering = _steering()
    out      = PhysicalLoss.coherence_resynthesis(target.clone(), target, steering, dx, 1e-6)
    assert out.item() < 1e-9


def test_coherence_resynthesis_nonnegative():
    pred     = _curves(10)
    target   = _curves(11)
    x        = _x_axis()
    dx       = float(x[1] - x[0])
    steering = _steering()
    out      = PhysicalLoss.coherence_resynthesis(pred, target, steering, dx, 1e-6)
    assert out.item() >= 0.0


def test_covariance_matching_zero_on_identical():
    target = _curves(12)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    outer  = _outer(_steering())
    out    = PhysicalLoss.covariance_matching(target.clone(), target, outer, dx, 1e-6)
    assert out.item() < 1e-9


def test_covariance_matching_nonnegative():
    pred   = _curves(13)
    target = _curves(14)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    outer  = _outer(_steering())
    out    = PhysicalLoss.covariance_matching(pred, target, outer, dx, 1e-6)
    assert out.item() >= 0.0


def test_total_power_gradient_flow():
    pred   = _curves(15).requires_grad_(True)
    target = _curves(16)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    PhysicalLoss.total_power(pred, target, dx, 1e-6).backward()
    assert torch.isfinite(pred.grad).all()


def test_moments_gradient_flow():
    pred   = _curves(17).requires_grad_(True)
    target = _curves(18)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    PhysicalLoss.moments(pred, target, x, dx, 1e-6, (1.0, 1.0, 1.0)).backward()
    assert torch.isfinite(pred.grad).all()


def test_covariance_matching_gradient_flow():
    pred   = _curves(19).requires_grad_(True)
    target = _curves(20)
    x      = _x_axis()
    dx     = float(x[1] - x[0])
    outer  = _outer(_steering())
    PhysicalLoss.covariance_matching(pred, target, outer, dx, 1e-6).backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.slow
def test_capon_cycle_runs_and_nonnegative():
    pred     = _curves(21, n=12)
    target   = _curves(22, n=12)
    x        = _x_axis()
    dx       = float(x[1] - x[0])
    steering = _steering()
    outer    = _outer(steering)
    out      = PhysicalLoss.capon_cycle(pred, target, steering, outer, dx, 0.01, 1e-6)
    assert out.ndim == 0
    assert out.item() >= 0.0


@pytest.mark.slow
def test_capon_cycle_gradient_flow():
    pred     = _curves(23, n=12).requires_grad_(True)
    target   = _curves(24, n=12)
    x        = _x_axis()
    dx       = float(x[1] - x[0])
    steering = _steering()
    outer    = _outer(steering)
    PhysicalLoss.capon_cycle(pred, target, steering, outer, dx, 0.05, 1e-6).backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.real_data
@pytest.mark.slow
def test_physical_losses_on_real_tomogram(tomogram_full):
    win = np.abs(np.asarray(tomogram_full[:12, :6, :6])).astype(np.float64)
    cur = torch.from_numpy(win)[None] + 1e-3

    x        = _x_axis(12)
    dx       = float(x[1] - x[0])
    steering = _steering(6, 12)
    outer    = _outer(steering)

    pred = cur * 1.1

    assert PhysicalLoss.total_power(pred, cur, dx, 1e-6).item() >= 0.0
    assert PhysicalLoss.moments(pred, cur, x, dx, 1e-6, (1.0, 1.0, 1.0)).item() >= 0.0
    assert PhysicalLoss.coherence_resynthesis(pred, cur, steering, dx, 1e-6).item() >= 0.0
    assert PhysicalLoss.covariance_matching(pred, cur, outer, dx, 1e-6).item() >= 0.0
    assert PhysicalLoss.total_power(cur, cur.clone(), dx, 1e-6).item() < 1e-9
