"""Safety tests for the curve and parameter loss terms on real Gaussian parameter data.

Each term is checked for a finite non-negative scalar forward value, a zero (or epsilon floor) at
identity, monotonic growth with injected noise, finite non-zero gradients, determinism, and
safety on all-zero curves. Also covers the active-slot masking of mu and sigma and the
ground-truth mu sorting performed by ParamMatcher."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.data.gaussians  import GaussianClamp, GaussianCurve
from tools.loss.curve_loss import CurveLoss as CL
from tools.loss.param_loss import ParamLoss as PL
from tools.loss.param_loss import ParamMatcher as PM

PPG     = 3
AMP_MAX = 1000.0
HUBER_D = 1.0
CHARB_E = 1e-3

CURVE_ZERO_AT_IDENTITY = ["mse", "l1", "huber", "cosine"]
CURVE_ALL              = CURVE_ZERO_AT_IDENTITY + ["charbonnier"]
PARAM_ALL              = ["param_l1", "param_huber", "param_mse", "tv"]


def _axis():
    """Returns the z axis of 150 bins spanning -20 to 80 metres."""
    return torch.linspace(-20.0, 80.0, 150, dtype=torch.float32)


def _gt_params(parameters):
    """Returns a 32x32 parameter crop of shape (2, 3 * n_gaussians, 32, 32) repeated over a batch of two."""
    crop = np.asarray(parameters[:, 400:432, 200:232]).copy()
    return torch.from_numpy(crop).unsqueeze(0).repeat(2, 1, 1, 1).float()


def _curves(params):
    """Returns the Gaussian mixture curves reconstructed from the parameter cube on the z axis."""
    return GaussianCurve.reconstruct(params, _axis(), PPG)


def _pred_curves(gt_params, noise, seed=0):
    """Returns a noisy, clamped, gradient-tracking parameter cube and the curves reconstructed from it.

    Args:
        gt_params : Ground-truth parameter cube of shape (batch, 3 * n_gaussians, height, width).
        noise     : Standard deviation of the Gaussian perturbation added to every parameter.
        seed      : Torch manual seed used for the perturbation.

    Returns:
        Tuple of the perturbed parameter cube and its reconstructed curve cube.
    """
    torch.manual_seed(seed)
    p = gt_params + noise * torch.randn_like(gt_params)
    p = GaussianClamp.apply(p, z_axis=_axis(), amp_max=AMP_MAX, ppg=PPG, leaky_slope=0.01)
    p = p.clone().requires_grad_(True)
    return p, _curves(p)


def _curve(name, pc, target):
    """Returns the named curve loss term evaluated between the predicted and target curve cubes.

    Args:
        name   : Term to evaluate, one of mse, l1, huber, charbonnier or cosine.
        pc     : Predicted curve cube of shape (batch, n_elev, height, width).
        target : Target curve cube of the same shape.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError : If the term name is unknown.
    """
    if name == "mse":
        return CL.mse_diff(pc - target)
    if name == "l1":
        return CL.l1_diff(pc - target)
    if name == "huber":
        return CL.huber_diff(pc - target, HUBER_D)
    if name == "charbonnier":
        return CL.charbonnier_diff(pc - target, CHARB_E)
    if name == "cosine":
        return CL.cosine(pc, target, 1)

    raise ValueError(name)


def _gt5(parameters):
    """Returns the ground-truth parameters regrouped to shape (batch, 5, 3, height, width)."""
    gt         = _gt_params(parameters)
    b, c, h, w = gt.shape
    return gt.reshape(b, c // PPG, PPG, h, w)


def _weights(gt5):
    """Returns per-parameter weights that zero mu and sigma wherever the slot amplitude is inactive."""
    active = (gt5[:, :, 0:1] > 1e-3).float()
    mask   = torch.ones_like(gt5)
    mask[:, :, 1:] = active.expand_as(gt5[:, :, 1:])
    return torch.ones(1, 1, PPG, 1, 1) * mask


def _pred5(gt5, noise, seed=0):
    """Returns a noisy, gradient-tracking copy of the grouped ground-truth parameters."""
    torch.manual_seed(seed)
    return (gt5 + noise * torch.randn_like(gt5)).clone().requires_grad_(True)


def _param(name, p, gt5, weights):
    """Returns the named parameter loss term evaluated between the grouped prediction and ground truth.

    Args:
        name    : Term to evaluate, one of param_l1, param_huber, param_mse or tv.
        p       : Predicted parameters of shape (batch, n_gaussians, 3, height, width).
        gt5     : Ground-truth parameters of the same shape.
        weights : Per-parameter weights broadcast over the same shape.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError : If the term name is unknown.
    """
    if name == "param_l1":
        return PL.l1(p, gt5, weights, ["amp", "mu", "sigma"])[0]
    if name == "param_huber":
        return PL.huber(p, gt5, weights, 0.5)
    if name == "param_mse":
        return PL.mse(p, gt5, weights)
    if name == "tv":
        return PL.tv(p)

    raise ValueError(name)


@pytest.mark.dataset
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_forward_finite_nonnegative(parameters, name):
    """Every curve term returns a finite non-negative scalar on real data."""
    gt     = _gt_params(parameters)
    target = _curves(gt)
    _, pc  = _pred_curves(gt, 0.2)

    val = _curve(name, pc, target)

    assert val.ndim == 0
    assert torch.isfinite(val)
    assert float(val.detach()) >= -1e-9


@pytest.mark.dataset
@pytest.mark.parametrize("name", CURVE_ZERO_AT_IDENTITY)
def test_curve_term_zero_when_prediction_equals_target(parameters, name):
    """Curve terms that vanish at identity score zero when prediction equals target."""
    gt     = _gt_params(parameters)
    target = _curves(gt)

    val = _curve(name, target, target)

    assert float(val) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.dataset
def test_charbonnier_floors_at_eps_at_identity(parameters):
    """The Charbonnier term floors at its epsilon rather than zero at identity."""
    gt     = _gt_params(parameters)
    target = _curves(gt)

    val = _curve("charbonnier", target, target)

    assert float(val) == pytest.approx(CHARB_E, rel=1e-3)


@pytest.mark.dataset
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_monotonic_in_error(parameters, name):
    """Every curve term grows as the injected parameter noise grows."""
    gt     = _gt_params(parameters)
    target = _curves(gt)

    vals = []
    for nz in (0.05, 0.2, 0.6):
        _, pc = _pred_curves(gt, nz, seed=1)
        vals.append(float(_curve(name, pc, target).detach()))

    assert vals[0] < vals[1] < vals[2]


@pytest.mark.dataset
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_gradient_finite_and_nonzero(parameters, name):
    """Every curve term backpropagates a finite, non-zero gradient to the parameters."""
    gt     = _gt_params(parameters)
    target = _curves(gt)
    p, pc  = _pred_curves(gt, 0.2)

    _curve(name, pc, target).backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.norm()) > 0.0


@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_finite_on_zero_curves(name):
    """Every curve term stays finite on all-zero curves."""
    z = torch.zeros(2, 150, 8, 8)

    val = _curve(name, z, z)

    assert torch.isfinite(val)
    assert not torch.isnan(val)


@pytest.mark.dataset
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_deterministic(parameters, name):
    """Evaluating a curve term twice on the same inputs gives the identical value."""
    gt     = _gt_params(parameters)
    target = _curves(gt)
    _, pc  = _pred_curves(gt, 0.2, seed=4)

    a = float(_curve(name, pc, target).detach())
    b = float(_curve(name, pc, target).detach())

    assert a == b


@pytest.mark.dataset
def test_cosine_is_bounded(parameters):
    """The cosine term stays within zero and two."""
    gt     = _gt_params(parameters)
    target = _curves(gt)
    _, pc  = _pred_curves(gt, 0.6)

    assert 0.0 <= float(CL.cosine(pc, target, 1).detach()) <= 2.0


@pytest.mark.dataset
@pytest.mark.parametrize("name", PARAM_ALL)
def test_param_term_forward_finite_nonnegative(parameters, name):
    """Every parameter term returns a finite non-negative scalar on real data."""
    gt5 = _gt5(parameters)
    w   = _weights(gt5)
    p   = _pred5(gt5, 0.2)

    val = _param(name, p, gt5, w)

    assert val.ndim == 0
    assert torch.isfinite(val)
    assert float(val.detach()) >= 0.0


@pytest.mark.dataset
@pytest.mark.parametrize("name", ["param_l1", "param_huber", "param_mse"])
def test_param_term_zero_when_prediction_equals_target(parameters, name):
    """The fidelity parameter terms score zero when prediction equals ground truth."""
    gt5 = _gt5(parameters)
    w   = _weights(gt5)

    val = _param(name, gt5, gt5, w)

    assert float(val) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.dataset
def test_tv_zero_on_spatially_constant_map(parameters):
    """Total variation vanishes on a spatially constant parameter map."""
    gt5      = _gt5(parameters)
    constant = torch.ones_like(gt5) * gt5.mean()

    assert float(PL.tv(constant)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.dataset
@pytest.mark.parametrize("name", PARAM_ALL)
def test_param_term_monotonic_in_error(parameters, name):
    """Every parameter term grows as the injected parameter noise grows."""
    gt5 = _gt5(parameters)
    w   = _weights(gt5)

    vals = []
    for nz in (0.05, 0.2, 0.6):
        p = _pred5(gt5, nz, seed=2)
        vals.append(float(_param(name, p, gt5, w).detach()))

    assert vals[0] < vals[1] < vals[2]


@pytest.mark.dataset
@pytest.mark.parametrize("name", PARAM_ALL)
def test_param_term_gradient_finite_and_nonzero(parameters, name):
    """Every parameter term backpropagates a finite, non-zero gradient."""
    gt5 = _gt5(parameters)
    w   = _weights(gt5)
    p   = _pred5(gt5, 0.2)

    _param(name, p, gt5, w).backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.norm()) > 0.0


@pytest.mark.dataset
def test_param_l1_returns_per_parameter_breakdown(parameters):
    """The L1 parameter term reports a finite value for each of amplitude, mu and sigma."""
    gt5 = _gt5(parameters)
    w   = _weights(gt5)
    p   = _pred5(gt5, 0.3)

    total, per_param = PL.l1(p, gt5, w, ["amp", "mu", "sigma"])

    assert set(per_param.keys()) == {"amp", "mu", "sigma"}
    assert all(torch.isfinite(v) for v in per_param.values())


def test_param_l1_masks_out_inactive_gaussian_mu_sigma():
    """With inactive slots masked, any mu and sigma prediction scores zero."""
    gt   = torch.zeros(1, 5, 3, 4, 4)
    mask = torch.ones_like(gt)
    mask[:, :, 1:] = 0.0
    w    = torch.ones(1, 1, 3, 1, 1) * mask

    p1 = torch.zeros(1, 5, 3, 4, 4)
    p1[:, :, 1:] = 7.0
    p2 = torch.zeros(1, 5, 3, 4, 4)
    p2[:, :, 1:] = -3.0

    v1 = float(PL.l1(p1, gt, w, ["amp", "mu", "sigma"])[0])
    v2 = float(PL.l1(p2, gt, w, ["amp", "mu", "sigma"])[0])

    assert v1 == pytest.approx(0.0, abs=1e-9)
    assert v2 == pytest.approx(0.0, abs=1e-9)


def test_match_sorts_gt_by_mu_among_active():
    """Matching sorts the active ground-truth slots into ascending mu order."""
    pred      = torch.randn(1, 5, 3, 4, 4)
    pred_phys = pred.clone()

    gt      = torch.randn(1, 5, 3, 4, 4)
    gt_phys = gt.clone()

    gt_phys[:, :, 0] = torch.tensor([2.0, 0.0, 3.0, 0.0, 1.0]).reshape(1, 5, 1, 1)
    gt_phys[:, :, 1] = torch.tensor([5.0, 9.0, 1.0, 8.0, 3.0]).reshape(1, 5, 1, 1)

    _, _, g, gp = PM.match(pred, pred_phys, gt, gt_phys)

    active_mus = gp[0, :, 1, 0, 0][gp[0, :, 0, 0, 0] > 1e-3]
    assert torch.all(active_mus[1:] >= active_mus[:-1])


def test_match_gt_sort_is_idempotent():
    """Matching an already-sorted ground truth leaves it unchanged."""
    pred    = torch.randn(1, 5, 3, 3, 3)
    gt      = torch.randn(1, 5, 3, 3, 3)
    gt_phys = torch.rand(1, 5, 3, 3, 3) + 0.5

    once  = PM.match(pred, pred.clone(), gt, gt_phys)
    twice = PM.match(once[0], once[1], once[2], once[3])

    assert torch.equal(once[2], twice[2])
    assert torch.equal(once[3], twice[3])
