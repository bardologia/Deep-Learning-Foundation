"""Tests covering the Gaussian parameter loss terms and the slot matcher.

Exercises the L1, Huber and total-variation terms of ``ParamLoss``, its presence
and focal weighting, and the Hungarian and sorted-by-mu assignment strategies of
``ParamMatcher``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from tools.loss.param_loss import ParamLoss, ParamMatcher


PARAM_NAMES = ["amp", "mu", "sigma"]


def _params(seed: int, b=2, g=5, h=4, w=4) -> torch.Tensor:
    """Returns uniform random Gaussian parameters of shape (b, g, 3, h, w)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(b, g, 3, h, w, generator=gen, dtype=torch.float64)


def _weights(b=2, g=5, h=4, w=4) -> torch.Tensor:
    """Returns an all-ones per-parameter weight tensor of shape (b, g, 3, h, w)."""
    return torch.ones(b, g, 3, h, w, dtype=torch.float64)


def test_l1_zero_on_identical():
    """Verifies the L1 term and every per-parameter part vanish when prediction equals ground truth."""
    p = _params(0)
    total, per = ParamLoss.l1(p, p.clone(), _weights(), PARAM_NAMES)
    assert total.item() == 0.0
    for v in per.values():
        assert v.item() == 0.0


def test_l1_nonnegative():
    """Verifies the L1 term is never negative for unrelated parameter tensors."""
    total, _ = ParamLoss.l1(_params(1), _params(2), _weights(), PARAM_NAMES)
    assert total.item() >= 0.0


def test_l1_per_param_keys_and_sum():
    """Verifies the per-parameter dictionary is keyed by parameter name and averages to the total."""
    pred = _params(3)
    gt   = _params(4)
    total, per = ParamLoss.l1(pred, gt, _weights(), PARAM_NAMES)
    assert set(per.keys()) == set(PARAM_NAMES)

    recombined = sum(p.item() for p in per.values()) / len(PARAM_NAMES)
    assert abs(recombined - total.item()) < 1e-9


def test_l1_weight_zero_kills_term():
    """Verifies zeroing the weight of the mu channel removes its contribution while amplitude still counts."""
    pred = _params(5)
    gt   = _params(6)
    w    = _weights()
    w[:, :, 1] = 0.0
    _, per = ParamLoss.l1(pred, gt, w, PARAM_NAMES)
    assert per["mu"].item() == 0.0
    assert per["amp"].item() > 0.0


def test_l1_gradient_flow():
    """Verifies the L1 term backpropagates finite gradients into the predicted parameters."""
    pred = _params(7).requires_grad_(True)
    gt   = _params(8)
    total, _ = ParamLoss.l1(pred, gt, _weights(), PARAM_NAMES)
    total.backward()
    assert torch.isfinite(pred.grad).all()


def test_huber_zero_on_identical():
    """Verifies the Huber term vanishes when prediction equals ground truth."""
    p   = _params(9)
    out = ParamLoss.huber(p, p.clone(), _weights(), 1.0)
    assert out.item() == 0.0


def test_huber_nonnegative():
    """Verifies the Huber term is never negative."""
    out = ParamLoss.huber(_params(10), _params(11), _weights(), 1.0)
    assert out.item() >= 0.0


def test_huber_quadratic_branch():
    """Verifies residuals below delta follow the quadratic half-squared branch."""
    pred = torch.full((1, 1, 3, 1, 1), 0.1, dtype=torch.float64)
    gt   = torch.zeros_like(pred)
    out  = ParamLoss.huber(pred, gt, torch.ones_like(pred), 1.0)
    diff = pred - gt
    assert torch.allclose(out, (0.5 * diff * diff).mean())


def test_huber_gradient_flow():
    """Verifies the Huber term backpropagates finite gradients."""
    pred = _params(12).requires_grad_(True)
    gt   = _params(13)
    ParamLoss.huber(pred, gt, _weights(), 1.0).backward()
    assert torch.isfinite(pred.grad).all()


def test_tv_zero_on_constant():
    """Verifies the total-variation term is zero on a spatially constant parameter map."""
    flat = torch.ones(1, 5, 3, 6, 6, dtype=torch.float64) * 0.3
    assert ParamLoss.tv(flat).item() == 0.0


def test_tv_nonnegative_and_positive_on_variation():
    """Verifies the total-variation term is strictly positive on a spatially varying map."""
    out = ParamLoss.tv(_params(14, h=6, w=6))
    assert out.item() > 0.0


def test_tv_gradient_flow():
    """Verifies the total-variation term backpropagates finite gradients."""
    p = _params(15, h=6, w=6).requires_grad_(True)
    ParamLoss.tv(p).backward()
    assert torch.isfinite(p.grad).all()


def test_match_sorts_gt_by_mu():
    """Verifies the matcher returns ground-truth slots ordered by ascending mu."""
    b, g, h, w = 1, 3, 1, 1
    gt = torch.zeros(b, g, 3, h, w, dtype=torch.float64)
    gt[0, :, 0, 0, 0] = torch.tensor([1.0, 1.0, 1.0])
    gt[0, :, 1, 0, 0] = torch.tensor([5.0, -2.0, 1.0])
    gt_phys = gt.clone()

    pred = torch.zeros_like(gt)

    _, _, sorted_gt, _ = ParamMatcher.match(pred, pred.clone(), gt, gt_phys)
    mus = sorted_gt[0, :, 1, 0, 0]
    assert torch.all(mus[:-1] <= mus[1:])


def test_match_pushes_inactive_gt_last():
    """Verifies zero-amplitude ground-truth slots are ordered after the active ones."""
    b, g, h, w = 1, 3, 1, 1
    gt = torch.zeros(b, g, 3, h, w, dtype=torch.float64)
    gt[0, :, 0, 0, 0] = torch.tensor([1.0, 0.0, 1.0])
    gt[0, :, 1, 0, 0] = torch.tensor([5.0, -8.0, 1.0])
    gt_phys = gt.clone()
    pred    = torch.zeros_like(gt)

    _, _, sorted_gt, sorted_phys = ParamMatcher.match(pred, pred.clone(), gt, gt_phys)
    last_amp = sorted_phys[0, -1, 0, 0, 0]
    assert last_amp.item() == 0.0


def test_match_realigns_pred():
    """Verifies a permuted prediction is realigned onto the sorted ground-truth slots."""
    b, g, h, w = 2, 3, 4, 4
    gt      = _params(20, b, g, h, w)
    gt_phys = gt.clone()
    gt_phys[:, :, 0] = 1.0

    perm      = [2, 0, 1]
    pred      = gt[:, perm].clone()
    pred_phys = gt_phys[:, perm].clone()

    matched, _, sorted_gt, _ = ParamMatcher.match(pred, pred_phys, gt, gt_phys)
    assert (matched - sorted_gt).abs().max().item() < 1e-9


def test_match_assignment_reaches_the_exact_cost_minimum():
    """Verifies the Hungarian assignment attains the brute-force minimum over all slot permutations."""
    b, g, h, w = 2, 3, 3, 3
    pred       = _params(31, b, g, h, w)
    gt         = _params(32, b, g, h, w)
    gt_phys    = gt.clone()
    gt_phys[:, :, 0] = 1.0

    matched, _, sorted_gt, _ = ParamMatcher.match(pred, pred.clone(), gt, gt_phys)

    achieved  = (matched - sorted_gt).abs().sum(2).sum(1)
    per_perm  = [(pred[:, list(perm)] - sorted_gt).abs().sum(2).sum(1) for perm in itertools.permutations(range(g))]
    brute     = torch.stack(per_perm).min(dim=0).values

    assert torch.allclose(achieved, brute, atol=1e-12)


def test_match_ignores_inactive_gt():
    """Verifies matching stays exact on the active slots when one ground-truth slot is empty."""
    b, g, h, w = 2, 3, 4, 4
    gt      = _params(21, b, g, h, w)
    gt_phys = gt.clone()
    gt_phys[:, :, 0] = 1.0
    gt_phys[:, 2, :] = 0.0

    perm = [2, 0, 1]
    matched, _, sorted_gt, sorted_phys = ParamMatcher.match(gt[:, perm].clone(), gt_phys[:, perm].clone(), gt, gt_phys)
    active = sorted_phys[:, :, 0] > 1e-3
    err    = ((matched - sorted_gt).abs().sum(2) * active)[active.bool()]
    assert err.max().item() < 1e-9


def test_match_rejects_large_g():
    """Verifies matching refuses more slots than MAX_GAUSSIANS."""
    b, g, h, w = 1, ParamMatcher.MAX_GAUSSIANS + 1, 1, 1
    t = _params(22, b, g, h, w)
    with pytest.raises(ValueError):
        ParamMatcher.match(t, t.clone(), t, t.clone())


def test_match_rejects_unknown_method():
    """Verifies matching refuses an unrecognised assignment method name."""
    b, g, h, w = 1, 3, 1, 1
    t = _params(23, b, g, h, w)
    with pytest.raises(ValueError):
        ParamMatcher.match(t, t.clone(), t, t.clone(), method="bogus")


def test_sorted_gt_sorts_gt_but_leaves_pred():
    """Verifies the sorted-GT method orders ground truth by mu and returns the prediction unchanged."""
    b, g, h, w = 2, 3, 4, 4
    gt      = _params(24, b, g, h, w)
    gt_phys = gt.clone()
    gt_phys[:, :, 0] = 1.0

    perm      = [2, 0, 1]
    pred      = gt[:, perm].clone()
    pred_phys = gt_phys[:, perm].clone()

    matched, _, sorted_gt, _ = ParamMatcher.match(pred, pred_phys, gt, gt_phys, method=ParamMatcher.SORTED_GT)

    mus = sorted_gt[:, :, 1]
    assert torch.all(mus[:, :-1] <= mus[:, 1:])
    assert torch.equal(matched, pred)


def test_sorted_gt_pushes_inactive_gt_last():
    """Verifies the sorted-GT method also pushes zero-amplitude ground-truth slots last."""
    b, g, h, w = 1, 3, 1, 1
    gt = torch.zeros(b, g, 3, h, w, dtype=torch.float64)
    gt[0, :, 0, 0, 0] = torch.tensor([1.0, 0.0, 1.0])
    gt[0, :, 1, 0, 0] = torch.tensor([5.0, -8.0, 1.0])
    gt_phys = gt.clone()
    pred    = torch.zeros_like(gt)

    _, _, _, sorted_phys = ParamMatcher.match(pred, pred.clone(), gt, gt_phys, method=ParamMatcher.SORTED_GT)
    assert sorted_phys[0, -1, 0, 0, 0].item() == 0.0


def test_presence_scale_balances_within_each_slot():
    """Verifies presence balancing equalises the total active and inactive weight inside one slot."""
    b, g, h, w = 2, 2, 4, 4
    active = torch.zeros(b, g, 1, h, w, dtype=torch.float64)
    active[:, 0]           = 1.0
    active[:, 1, :, :1, :] = 1.0

    scale          = ParamLoss.presence_scale(active, balance=True, active_weight=1.0, inactive_weight=1.0)
    slot1_active   = scale[:, 1][active[:, 1].bool()]
    slot1_inactive = scale[:, 1][active[:, 1] == 0]

    assert slot1_active[0].item()    == pytest.approx(0.5 / 0.25)
    assert slot1_inactive[0].item()  == pytest.approx(0.5 / 0.75)
    assert slot1_active.sum().item() == pytest.approx(slot1_inactive.sum().item())


def test_presence_scale_fraction_is_independent_per_slot():
    """Verifies each slot's presence fraction is computed from that slot alone."""
    b, g, h, w = 2, 2, 4, 4
    active = torch.zeros(b, g, 1, h, w, dtype=torch.float64)
    active[:, 0]           = 1.0
    active[:, 1, :, :1, :] = 1.0

    scale = ParamLoss.presence_scale(active, balance=True, active_weight=1.0, inactive_weight=1.0)
    slot0 = scale[:, 0]

    assert torch.allclose(slot0, torch.full_like(slot0, 0.5 / (1.0 - ParamLoss.FRAC_CLAMP)))


def test_presence_scale_clamps_degenerate_fractions():
    """Verifies fully active or fully empty slots stay finite through the fraction clamp."""
    b, g, h, w = 1, 2, 4, 4
    active = torch.zeros(b, g, 1, h, w, dtype=torch.float64)
    active[:, 0] = 1.0

    scale = ParamLoss.presence_scale(active, balance=True, active_weight=1.0, inactive_weight=1.0)

    assert torch.isfinite(scale).all()
    assert scale.max().item() <= 0.5 / ParamLoss.FRAC_CLAMP
    assert scale[:, 0].max().item() == pytest.approx(0.5 / (1.0 - ParamLoss.FRAC_CLAMP))
    assert scale[:, 1].max().item() == pytest.approx(0.5 / (1.0 - ParamLoss.FRAC_CLAMP))


def test_presence_scale_default_is_identity():
    """Verifies presence scaling is all ones when balancing is disabled and both weights are unity."""
    active = torch.tensor([1.0, 0.0]).reshape(1, 2, 1, 1, 1)
    scale  = ParamLoss.presence_scale(active, balance=False, active_weight=1.0, inactive_weight=1.0)
    assert torch.allclose(scale, torch.ones_like(scale))


def test_focal_scale_downweights_easy_zeros():
    """Verifies the focal scale is larger for a missed strong component than for a missed negligible one."""
    amp_pred  = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64)
    amp_big   = torch.ones_like(amp_pred)
    amp_small = torch.full_like(amp_pred, 1e-3)

    hard = ParamLoss.focal_scale(amp_pred, amp_big,   gamma=2.0, delta=0.5)
    easy = ParamLoss.focal_scale(amp_pred, amp_small, gamma=2.0, delta=0.5)
    assert hard.item() > easy.item()
    assert easy.item() < 1e-3


def test_focal_scale_gamma_zero_is_identity():
    """Verifies a focal exponent of zero leaves the scale at one."""
    amp_pred = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64)
    amp_gt   = torch.ones_like(amp_pred)
    scale    = ParamLoss.focal_scale(amp_pred, amp_gt, gamma=0.0, delta=0.5)
    assert torch.allclose(scale, torch.ones_like(scale))


def test_active_norm_matches_mean_for_uniform_weights():
    """Verifies active normalisation equals the plain mean when every weight is one."""
    pred = _params(23)
    gt   = _params(24)
    w    = _weights()
    t_mean, _ = ParamLoss.l1(pred, gt, w, PARAM_NAMES, active_norm=False)
    t_norm, _ = ParamLoss.l1(pred, gt, w, PARAM_NAMES, active_norm=True)
    assert abs(t_mean.item() - t_norm.item()) < 1e-9


def test_active_norm_rescales_by_active_fraction():
    """Verifies active normalisation raises the loss when part of the weight mask is zero."""
    pred = _params(25)
    gt   = _params(26)
    w    = _weights()
    w[:, 2:] = 0.0

    t_mean, _ = ParamLoss.l1(pred, gt, w, PARAM_NAMES, active_norm=False)
    t_norm, _ = ParamLoss.l1(pred, gt, w, PARAM_NAMES, active_norm=True)
    assert t_norm.item() > t_mean.item()


@pytest.mark.dataset
def test_param_loss_on_scene_parameters(parameters):
    """Verifies the L1 term on a scene parameter crop isolates the perturbed amplitude channel."""
    win = np.asarray(parameters[:, :8, :8]).astype(np.float64)
    t   = torch.from_numpy(win).reshape(win.shape[0] // 3, 3, 8, 8)[None]

    pred = t.clone()
    pred[:, :, 0] += 0.02

    total, per = ParamLoss.l1(pred, t, torch.ones_like(t), PARAM_NAMES)
    assert total.item() > 0.0
    assert per["mu"].item() < 1e-9
    assert per["amp"].item() > 0.0

    self_loss, _ = ParamLoss.l1(t, t.clone(), torch.ones_like(t), PARAM_NAMES)
    assert self_loss.item() == 0.0
