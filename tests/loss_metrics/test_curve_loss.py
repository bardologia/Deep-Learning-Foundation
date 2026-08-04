from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tools.loss.curve_loss import CurveLoss


def _curves(seed: int, shape=(2, 8, 4, 4)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g, dtype=torch.float64)


def test_mse_diff_zero_on_identical():
    diff = torch.zeros(2, 8, 4, 4, dtype=torch.float64)
    assert CurveLoss.mse_diff(diff).item() == 0.0


def test_mse_diff_nonnegative():
    diff = _curves(0) - 0.5
    assert CurveLoss.mse_diff(diff).item() >= 0.0


def test_mse_diff_matches_definition():
    diff     = _curves(1) - 0.5
    expected = (diff * diff).mean()
    assert torch.allclose(CurveLoss.mse_diff(diff), expected)


def test_l1_diff_zero_and_nonnegative():
    zero = torch.zeros(1, 4, 3, 3, dtype=torch.float64)
    diff = _curves(2) - 0.5
    assert CurveLoss.l1_diff(zero).item() == 0.0
    assert CurveLoss.l1_diff(diff).item() >= 0.0


def test_l1_diff_matches_definition():
    diff = _curves(3) - 0.5
    assert torch.allclose(CurveLoss.l1_diff(diff), diff.abs().mean())


def test_huber_quadratic_below_delta():
    diff  = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float64)
    delta = 1.0
    out   = CurveLoss.huber_diff(diff, delta)
    assert torch.allclose(out, 0.5 * diff * diff)


def test_huber_linear_above_delta():
    diff     = torch.full((1, 1, 1, 1), 5.0, dtype=torch.float64)
    delta    = 1.0
    out      = CurveLoss.huber_diff(diff, delta)
    expected = delta * (diff.abs() - 0.5 * delta)
    assert torch.allclose(out, expected)


def test_huber_zero_on_identical():
    diff = torch.zeros(2, 4, 2, 2, dtype=torch.float64)
    assert CurveLoss.huber_diff(diff, 1.0).item() == 0.0


def test_charbonnier_nonnegative_and_floor():
    eps  = 1e-3
    zero = torch.zeros(2, 4, 2, 2, dtype=torch.float64)
    out  = CurveLoss.charbonnier_diff(zero, eps)
    assert out.item() >= 0.0
    assert torch.allclose(out, torch.tensor(eps, dtype=torch.float64))


def test_smooth_l1_quadratic_and_linear():
    beta  = 1.0
    small = torch.full((1, 1, 1, 1), 0.2, dtype=torch.float64)
    large = torch.full((1, 1, 1, 1), 4.0, dtype=torch.float64)
    assert torch.allclose(CurveLoss.smooth_l1_diff(small, beta), 0.5 * small * small / beta)
    assert torch.allclose(CurveLoss.smooth_l1_diff(large, beta), large.abs() - 0.5 * beta)


def test_cosine_zero_on_identical():
    target = _curves(4)
    out    = CurveLoss.cosine(target.clone(), target, axis=1)
    assert out.item() < 1e-6


def test_cosine_nonnegative():
    pred   = _curves(5)
    target = _curves(6)
    out    = CurveLoss.cosine(pred, target, axis=1)
    assert out.item() >= 0.0


def test_cosine_scale_invariant():
    pred   = _curves(7)
    target = _curves(8)
    base   = CurveLoss.cosine(pred, target, axis=1)
    scaled = CurveLoss.cosine(pred * 3.0, target, axis=1)
    assert torch.allclose(base, scaled, atol=1e-6)


def test_mse_diff_gradient_flow():
    diff = (_curves(16) - 0.5).requires_grad_(True)
    CurveLoss.mse_diff(diff).backward()
    assert diff.grad is not None
    assert torch.isfinite(diff.grad).all()


def test_cosine_gradient_flow():
    pred   = _curves(17).requires_grad_(True)
    target = _curves(18)
    CurveLoss.cosine(pred, target, axis=1).backward()
    assert torch.isfinite(pred.grad).all()


def test_outputs_are_scalars():
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
    ):
        assert out.ndim == 0


@pytest.mark.real_data
def test_curve_losses_on_real_tomogram(tomogram_full):
    win = np.abs(np.asarray(tomogram_full[:16, :8, :8])).astype(np.float64)
    cur = torch.from_numpy(win)[None]

    pred = cur * 1.05

    assert CurveLoss.mse_diff(pred - cur).item() >= 0.0
    assert CurveLoss.cosine(pred, cur, axis=1).item() >= 0.0
    assert CurveLoss.mse_diff(cur - cur).item() == 0.0
