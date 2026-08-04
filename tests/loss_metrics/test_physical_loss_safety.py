from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.sar.geometry_config import GeometryConfig
from tools.data.gaussians              import GaussianClamp, GaussianCurve
from tools.loss.physical_loss          import PhysicalLoss as PL
from tools.sar.tomo_geometry           import TomoGeometry

FLOOR   = 1e-3
LOADING = 1e-2
MW      = (1.0, 1.0, 1.0)
PPG     = 3
AMP_MAX = 1000.0

ZERO_AT_IDENTITY = ["total_power", "moments", "coherence_resyn", "covariance_match"]
ALL_TERMS        = ZERO_AT_IDENTITY + ["capon_cycle"]
STEERING_TERMS   = ["coherence_resyn", "covariance_match", "capon_cycle"]


def _axis():
    return torch.linspace(-20.0, 80.0, 150, dtype=torch.float32)


def _dx():
    x = _axis()
    return float(x[1] - x[0])


def _geometry(test_data_dir):
    return TomoGeometry(GeometryConfig().resolved(str(test_data_dir)), _axis())


def _gt(parameters):
    crop = np.asarray(parameters[:, 400:416, 200:216]).copy()
    gp   = torch.from_numpy(crop).unsqueeze(0).repeat(2, 1, 1, 1).float()
    return gp, GaussianCurve.reconstruct(gp, _axis(), PPG)


def _pred(gt_params, noise, seed=0):
    torch.manual_seed(seed)
    p = gt_params + noise * torch.randn_like(gt_params)
    p = GaussianClamp.apply(p, x_axis=_axis(), amp_max=AMP_MAX, ppg=PPG, leaky_slope=0.01)
    p = p.clone().requires_grad_(True)
    return p, GaussianCurve.reconstruct(p, _axis(), PPG)


def _apply(name, pred_curves, target, geom):
    x  = _axis()
    dx = _dx()

    if name == "total_power":
        return PL.total_power(pred_curves, target, dx, FLOOR)
    if name == "moments":
        return PL.moments(pred_curves, target, x, dx, FLOOR, MW)
    if name == "coherence_resyn":
        return PL.coherence_resynthesis(pred_curves, target, geom.steering, dx, FLOOR)
    if name == "covariance_match":
        return PL.covariance_matching(pred_curves, target, geom.outer, dx, FLOOR)
    if name == "capon_cycle":
        return PL.capon_cycle(pred_curves, target, geom.steering, geom.outer, dx, LOADING, FLOOR)

    raise ValueError(name)


def _flip(geom):
    s2 = torch.polar(torch.ones_like(geom.steering.abs()), -torch.angle(geom.steering))
    o2 = torch.einsum("ik,jk->ijk", s2, s2.conj())
    return s2, o2


@pytest.mark.real_data
def test_gt_curves_are_nonnegative_and_finite(parameters):
    _, target = _gt(parameters)

    assert torch.isfinite(target).all()
    assert float(target.min()) >= 0.0


@pytest.mark.real_data
def test_geometry_full_stack_kz_spans_baselines(parameters, test_data_dir):
    geom = _geometry(test_data_dir)

    assert geom.n_tracks == 29
    assert float(geom.kz.min()) == 0.0
    assert float(geom.kz.max()) > 1.0


@pytest.mark.real_data
@pytest.mark.parametrize("name", ALL_TERMS)
def test_term_forward_is_finite_scalar_nonnegative(parameters, test_data_dir, name):
    geom         = _geometry(test_data_dir)
    gt_params, t = _gt(parameters)
    _, pc        = _pred(gt_params, 0.2)

    val = _apply(name, pc, t, geom)

    assert val.ndim == 0
    assert torch.isfinite(val)
    assert float(val.detach()) >= 0.0


@pytest.mark.real_data
@pytest.mark.parametrize("name", ZERO_AT_IDENTITY)
def test_term_zero_when_prediction_equals_target(parameters, test_data_dir, name):
    geom         = _geometry(test_data_dir)
    gt_params, t = _gt(parameters)

    val = _apply(name, t, t, geom)

    assert float(val) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.real_data
def test_capon_cycle_is_small_when_prediction_equals_target(parameters, test_data_dir):
    geom         = _geometry(test_data_dir)
    gt_params, t = _gt(parameters)

    val = _apply("capon_cycle", t, t, geom)

    assert torch.isfinite(val)
    assert float(val) < 1e-2


@pytest.mark.real_data
@pytest.mark.parametrize("name", ALL_TERMS)
def test_term_increases_monotonically_with_error(parameters, test_data_dir, name):
    geom         = _geometry(test_data_dir)
    gt_params, t = _gt(parameters)

    vals = []
    for nz in (0.05, 0.2, 0.6):
        _, pc = _pred(gt_params, nz, seed=1)
        vals.append(float(_apply(name, pc, t, geom).detach()))

    assert vals[0] < vals[1] < vals[2]


@pytest.mark.real_data
@pytest.mark.parametrize("name", ALL_TERMS)
def test_term_gradient_is_finite_and_nonzero(parameters, test_data_dir, name):
    geom         = _geometry(test_data_dir)
    gt_params, t = _gt(parameters)
    p, pc        = _pred(gt_params, 0.2)

    _apply(name, pc, t, geom).backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.norm()) > 0.0


@pytest.mark.real_data
@pytest.mark.parametrize("name", ALL_TERMS)
def test_term_finite_zero_on_all_inactive_window(parameters, test_data_dir, name):
    geom = _geometry(test_data_dir)
    z    = torch.zeros(2, 150, 8, 8)

    val = _apply(name, z, z, geom)

    assert torch.isfinite(val)
    assert not torch.isnan(val)
    assert float(val) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.real_data
@pytest.mark.parametrize("name", ALL_TERMS)
def test_term_is_deterministic(parameters, test_data_dir, name):
    geom         = _geometry(test_data_dir)
    gt_params, t = _gt(parameters)
    _, pc        = _pred(gt_params, 0.2, seed=7)

    a = float(_apply(name, pc, t, geom).detach())
    b = float(_apply(name, pc, t, geom).detach())

    assert a == b


@pytest.mark.real_data
def test_capon_cycle_stable_on_degenerate_rank1_covariance(test_data_dir):
    geom = _geometry(test_data_dir)

    p = torch.zeros(1, 150, 4, 4)
    p[:, 75] = 5.0
    p = p.clone().requires_grad_(True)

    t = torch.zeros(1, 150, 4, 4)
    t[:, 80] = 5.0

    val = PL.capon_cycle(p, t, geom.steering, geom.outer, _dx(), LOADING, FLOOR)
    val.backward()

    assert torch.isfinite(val)
    assert torch.isfinite(p.grad).all()
    assert float(p.grad.norm()) > 0.0


@pytest.mark.real_data
@pytest.mark.parametrize("name", STEERING_TERMS)
def test_steering_terms_invariant_to_phase_sign(parameters, test_data_dir, name):
    geom         = _geometry(test_data_dir)
    gt_params, t = _gt(parameters)
    _, pc        = _pred(gt_params, 0.2, seed=3)
    pc           = pc.detach()

    s2, o2  = _flip(geom)
    flipped = type(geom).__new__(type(geom))
    flipped.steering = s2
    flipped.outer    = o2

    base = float(_apply(name, pc, t, geom))
    flip = float(_apply(name, pc, t, flipped))

    assert flip == pytest.approx(base, rel=1e-5, abs=1e-9)
