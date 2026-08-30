"""Tests for the CurveLoss elevation-profile terms.

Covers the difference-based terms (MSE, L1, Huber, Charbonnier, smooth L1) against their
definitions, the shape-only cosine term and the scale-sensitive log-energy term that complements
it, gradient flow, scalar reduction, behaviour on a real tomogram window, and the second
difference used by the Sobolev regulariser."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tools.loss.curve_loss import CurveLoss


def _curves(seed: int, shape=(2, 8, 4, 4)) -> torch.Tensor:
    """Returns a reproducible random float64 profile cube of the given shape, seeded by `seed`."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g, dtype=torch.float64)


def test_mse_diff_zero_on_identical():
    """A zero difference gives zero MSE."""
    diff = torch.zeros(2, 8, 4, 4, dtype=torch.float64)
    assert CurveLoss.mse_diff(diff).item() == 0.0


def test_mse_diff_nonnegative():
    """The MSE term is never negative."""
    diff = _curves(0) - 0.5
    assert CurveLoss.mse_diff(diff).item() >= 0.0


def test_mse_diff_matches_definition():
    """The MSE term equals the mean of the squared difference."""
    diff     = _curves(1) - 0.5
    expected = (diff * diff).mean()
    assert torch.allclose(CurveLoss.mse_diff(diff), expected)


def test_l1_diff_zero_and_nonnegative():
    """The L1 term is zero on a zero difference and never negative otherwise."""
    zero = torch.zeros(1, 4, 3, 3, dtype=torch.float64)
    diff = _curves(2) - 0.5
    assert CurveLoss.l1_diff(zero).item() == 0.0
    assert CurveLoss.l1_diff(diff).item() >= 0.0


def test_l1_diff_matches_definition():
    """The L1 term equals the mean absolute difference."""
    diff = _curves(3) - 0.5
    assert torch.allclose(CurveLoss.l1_diff(diff), diff.abs().mean())


def test_huber_quadratic_below_delta():
    """Below delta the Huber term is half the squared difference."""
    diff  = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float64)
    delta = 1.0
    out   = CurveLoss.huber_diff(diff, delta)
    assert torch.allclose(out, 0.5 * diff * diff)


def test_huber_linear_above_delta():
    """Above delta the Huber term is linear in the absolute difference."""
    diff     = torch.full((1, 1, 1, 1), 5.0, dtype=torch.float64)
    delta    = 1.0
    out      = CurveLoss.huber_diff(diff, delta)
    expected = delta * (diff.abs() - 0.5 * delta)
    assert torch.allclose(out, expected)


def test_huber_zero_on_identical():
    """A zero difference gives zero Huber loss."""
    diff = torch.zeros(2, 4, 2, 2, dtype=torch.float64)
    assert CurveLoss.huber_diff(diff, 1.0).item() == 0.0


def test_charbonnier_nonnegative_and_floor():
    """The Charbonnier term floors at epsilon on a zero difference."""
    eps  = 1e-3
    zero = torch.zeros(2, 4, 2, 2, dtype=torch.float64)
    out  = CurveLoss.charbonnier_diff(zero, eps)
    assert out.item() >= 0.0
    assert torch.allclose(out, torch.tensor(eps, dtype=torch.float64))


def test_smooth_l1_quadratic_and_linear():
    """Smooth L1 is quadratic below beta and linear above it."""
    beta  = 1.0
    small = torch.full((1, 1, 1, 1), 0.2, dtype=torch.float64)
    large = torch.full((1, 1, 1, 1), 4.0, dtype=torch.float64)
    assert torch.allclose(CurveLoss.smooth_l1_diff(small, beta), 0.5 * small * small / beta)
    assert torch.allclose(CurveLoss.smooth_l1_diff(large, beta), large.abs() - 0.5 * beta)


def test_cosine_zero_on_identical():
    """The cosine term vanishes when prediction and target coincide."""
    target = _curves(4)
    out    = CurveLoss.cosine(target.clone(), target, axis=1)
    assert out.item() < 1e-6


def test_cosine_nonnegative():
    """The cosine term is never negative."""
    pred   = _curves(5)
    target = _curves(6)
    out    = CurveLoss.cosine(pred, target, axis=1)
    assert out.item() >= 0.0


def test_cosine_scale_invariant():
    """Scaling the prediction leaves the cosine term unchanged."""
    pred   = _curves(7)
    target = _curves(8)
    base   = CurveLoss.cosine(pred, target, axis=1)
    scaled = CurveLoss.cosine(pred * 3.0, target, axis=1)
    assert torch.allclose(base, scaled, atol=1e-6)


def test_log_energy_zero_on_identical():
    """The log-energy term vanishes when prediction and target coincide."""
    target = _curves(9)
    out    = CurveLoss.log_energy(target.clone(), target, axis=1)
    assert out.item() < 1e-6


def test_log_energy_nonnegative():
    """The log-energy term is never negative."""
    pred   = _curves(10)
    target = _curves(11)
    out    = CurveLoss.log_energy(pred, target, axis=1)
    assert out.item() >= 0.0


def test_log_energy_detects_scale_error_where_cosine_is_blind():
    """A threefold scaling is invisible to cosine but scores log 3 in log energy."""
    target = _curves(12)
    scaled = target * 3.0

    cosine     = CurveLoss.cosine(scaled, target, axis=1)
    log_energy = CurveLoss.log_energy(scaled, target, axis=1)

    assert cosine.item() < 1e-6
    assert torch.allclose(log_energy, torch.tensor(math.log(3.0), dtype=torch.float64), atol=1e-4)


def test_log_energy_symmetric_in_scale_direction():
    """Scaling up and down by the same factor scores the same log-energy value."""
    target = _curves(13)
    up     = CurveLoss.log_energy(target * 4.0, target, axis=1)
    down   = CurveLoss.log_energy(target / 4.0, target, axis=1)
    assert torch.allclose(up, down, atol=1e-4)


def test_log_energy_ignores_empty_target_pixels():
    """Pixels whose target profile is all zero contribute nothing to the log-energy term."""
    target             = _curves(14)
    target[:, :, 0, 0] = 0.0

    pred             = target.clone()
    pred[:, :, 0, 0] = 5.0

    out = CurveLoss.log_energy(pred, target, axis=1)
    assert out.item() < 1e-6


def test_mse_diff_gradient_flow():
    """Backward through the MSE term leaves a finite gradient on the difference."""
    diff = (_curves(16) - 0.5).requires_grad_(True)
    CurveLoss.mse_diff(diff).backward()
    assert diff.grad is not None
    assert torch.isfinite(diff.grad).all()


def test_cosine_gradient_flow():
    """Backward through the cosine term leaves a finite gradient on the prediction."""
    pred   = _curves(17).requires_grad_(True)
    target = _curves(18)
    CurveLoss.cosine(pred, target, axis=1).backward()
    assert torch.isfinite(pred.grad).all()


def test_log_energy_gradient_flow():
    """Backward through the log-energy term leaves a finite gradient on the prediction."""
    pred   = _curves(19).requires_grad_(True)
    target = _curves(20)
    CurveLoss.log_energy(pred, target, axis=1).backward()
    assert torch.isfinite(pred.grad).all()


def test_outputs_are_scalars():
    """Every curve term reduces to a zero-dimensional tensor."""
    diff   = _curves(21) - 0.5
    pred   = _curves(22, (1, 8, 3, 3))
    target = _curves(23, (1, 8, 3, 3))
    for out in (
        CurveLoss.mse_diff(diff),
        CurveLoss.l1_diff(diff),
        CurveLoss.huber_diff(diff, 1.0),
        CurveLoss.charbonnier_diff(diff, 1e-3),
        CurveLoss.smooth_l1_diff(diff, 1.0),
        CurveLoss.cosine(pred, target, axis=1),
        CurveLoss.log_energy(pred, target, axis=1),
    ):
        assert out.ndim == 0


@pytest.mark.dataset
def test_curve_losses_on_scene_cube(cube_full):
    """On a scene cube window the terms stay non-negative and vanish at identity."""
    win = np.abs(np.asarray(cube_full[:16, :8, :8])).astype(np.float64)
    cur = torch.from_numpy(win)[None]

    pred = cur * 1.05

    assert CurveLoss.mse_diff(pred - cur).item() >= 0.0
    assert CurveLoss.cosine(pred, cur, axis=1).item() >= 0.0
    assert CurveLoss.mse_diff(cur - cur).item() == 0.0


def test_second_difference_zero_on_linear_profile():
    """A linear elevation ramp has a vanishing second difference."""
    ramp  = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    curve = ramp.view(1, 8, 1, 1).expand(2, 8, 4, 4)

    assert CurveLoss.second_difference(curve).abs().max().item() == pytest.approx(0.0, abs=1e-12)


def test_second_difference_matches_manual():
    """The second difference matches the manual three-point stencil and shortens the profile axis by two."""
    curve  = _curves(4)
    manual = curve[:, 2:] - 2.0 * curve[:, 1:-1] + curve[:, :-2]

    assert torch.equal(CurveLoss.second_difference(curve), manual)
    assert CurveLoss.second_difference(curve).shape == (2, 6, 4, 4)


def test_second_difference_short_profile_raises():
    """A profile shorter than three elevation bins raises ValueError."""
    with pytest.raises(ValueError, match="at least 3"):
        CurveLoss.second_difference(torch.zeros(1, 2, 1, 1))
