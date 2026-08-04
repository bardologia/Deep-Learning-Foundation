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
    return torch.linspace(-20.0, 80.0, 150, dtype=torch.float32)


def _gt_params(parameters):
    crop = np.asarray(parameters[:, 400:432, 200:232]).copy()
    return torch.from_numpy(crop).unsqueeze(0).repeat(2, 1, 1, 1).float()


def _curves(params):
    return GaussianCurve.reconstruct(params, _axis(), PPG)


def _pred_curves(gt_params, noise, seed=0):
    torch.manual_seed(seed)
    p = gt_params + noise * torch.randn_like(gt_params)
    p = GaussianClamp.apply(p, x_axis=_axis(), amp_max=AMP_MAX, ppg=PPG, leaky_slope=0.01)
    p = p.clone().requires_grad_(True)
    return p, _curves(p)


def _curve(name, pc, target):
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
    gt         = _gt_params(parameters)
    b, _, h, w = gt.shape
    return gt.reshape(b, 5, PPG, h, w)


def _weights(gt5):
    active = (gt5[:, :, 0:1] > 1e-3).float()
    mask   = torch.ones_like(gt5)
    mask[:, :, 1:] = active.expand_as(gt5[:, :, 1:])
    return torch.ones(1, 1, PPG, 1, 1) * mask


def _pred5(gt5, noise, seed=0):
    torch.manual_seed(seed)
    return (gt5 + noise * torch.randn_like(gt5)).clone().requires_grad_(True)


def _param(name, p, gt5, weights):
    if name == "param_l1":
        return PL.l1(p, gt5, weights, ["amp", "mu", "sigma"])[0]
    if name == "param_huber":
        return PL.huber(p, gt5, weights, 0.5)
    if name == "param_mse":
        return PL.mse(p, gt5, weights)
    if name == "tv":
        return PL.tv(p)

    raise ValueError(name)


@pytest.mark.real_data
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_forward_finite_nonnegative(parameters, name):
    gt     = _gt_params(parameters)
    target = _curves(gt)
    _, pc  = _pred_curves(gt, 0.2)

    val = _curve(name, pc, target)

    assert val.ndim == 0
    assert torch.isfinite(val)
    assert float(val.detach()) >= -1e-9


@pytest.mark.real_data
@pytest.mark.parametrize("name", CURVE_ZERO_AT_IDENTITY)
def test_curve_term_zero_when_prediction_equals_target(parameters, name):
    gt     = _gt_params(parameters)
    target = _curves(gt)

    val = _curve(name, target, target)

    assert float(val) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.real_data
def test_charbonnier_floors_at_eps_at_identity(parameters):
    gt     = _gt_params(parameters)
    target = _curves(gt)

    val = _curve("charbonnier", target, target)

    assert float(val) == pytest.approx(CHARB_E, rel=1e-3)


@pytest.mark.real_data
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_monotonic_in_error(parameters, name):
    gt     = _gt_params(parameters)
    target = _curves(gt)

    vals = []
    for nz in (0.05, 0.2, 0.6):
        _, pc = _pred_curves(gt, nz, seed=1)
        vals.append(float(_curve(name, pc, target).detach()))

    assert vals[0] < vals[1] < vals[2]


@pytest.mark.real_data
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_gradient_finite_and_nonzero(parameters, name):
    gt     = _gt_params(parameters)
    target = _curves(gt)
    p, pc  = _pred_curves(gt, 0.2)

    _curve(name, pc, target).backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.norm()) > 0.0


@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_finite_on_zero_curves(name):
    z = torch.zeros(2, 150, 8, 8)

    val = _curve(name, z, z)

    assert torch.isfinite(val)
    assert not torch.isnan(val)


@pytest.mark.real_data
@pytest.mark.parametrize("name", CURVE_ALL)
def test_curve_term_deterministic(parameters, name):
    gt     = _gt_params(parameters)
    target = _curves(gt)
    _, pc  = _pred_curves(gt, 0.2, seed=4)

    a = float(_curve(name, pc, target).detach())
    b = float(_curve(name, pc, target).detach())

    assert a == b


@pytest.mark.real_data
def test_cosine_is_bounded(parameters):
    gt     = _gt_params(parameters)
    target = _curves(gt)
    _, pc  = _pred_curves(gt, 0.6)

    assert 0.0 <= float(CL.cosine(pc, target, 1).detach()) <= 2.0


@pytest.mark.real_data
@pytest.mark.parametrize("name", PARAM_ALL)
def test_param_term_forward_finite_nonnegative(parameters, name):
    gt5 = _gt5(parameters)
    w   = _weights(gt5)
    p   = _pred5(gt5, 0.2)

    val = _param(name, p, gt5, w)

    assert val.ndim == 0
    assert torch.isfinite(val)
    assert float(val.detach()) >= 0.0


@pytest.mark.real_data
@pytest.mark.parametrize("name", ["param_l1", "param_huber", "param_mse"])
def test_param_term_zero_when_prediction_equals_target(parameters, name):
    gt5 = _gt5(parameters)
    w   = _weights(gt5)

    val = _param(name, gt5, gt5, w)

    assert float(val) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.real_data
def test_tv_zero_on_spatially_constant_map(parameters):
    gt5      = _gt5(parameters)
    constant = torch.ones_like(gt5) * gt5.mean()

    assert float(PL.tv(constant)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.real_data
@pytest.mark.parametrize("name", PARAM_ALL)
def test_param_term_monotonic_in_error(parameters, name):
    gt5 = _gt5(parameters)
    w   = _weights(gt5)

    vals = []
    for nz in (0.05, 0.2, 0.6):
        p = _pred5(gt5, nz, seed=2)
        vals.append(float(_param(name, p, gt5, w).detach()))

    assert vals[0] < vals[1] < vals[2]


@pytest.mark.real_data
@pytest.mark.parametrize("name", PARAM_ALL)
def test_param_term_gradient_finite_and_nonzero(parameters, name):
    gt5 = _gt5(parameters)
    w   = _weights(gt5)
    p   = _pred5(gt5, 0.2)

    _param(name, p, gt5, w).backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.norm()) > 0.0


@pytest.mark.real_data
def test_param_l1_returns_per_parameter_breakdown(parameters):
    gt5 = _gt5(parameters)
    w   = _weights(gt5)
    p   = _pred5(gt5, 0.3)

    total, per_param = PL.l1(p, gt5, w, ["amp", "mu", "sigma"])

    assert set(per_param.keys()) == {"amp", "mu", "sigma"}
    assert all(torch.isfinite(v) for v in per_param.values())


def test_param_l1_masks_out_inactive_gaussian_mu_sigma():
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
    pred    = torch.randn(1, 5, 3, 3, 3)
    gt      = torch.randn(1, 5, 3, 3, 3)
    gt_phys = torch.rand(1, 5, 3, 3, 3) + 0.5

    once  = PM.match(pred, pred.clone(), gt, gt_phys)
    twice = PM.match(once[0], once[1], once[2], once[3])

    assert torch.equal(once[2], twice[2])
    assert torch.equal(once[3], twice[3])
