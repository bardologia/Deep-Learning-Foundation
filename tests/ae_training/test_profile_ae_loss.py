"""Tests for the profile autoencoder reconstruction loss.

Covers the summable term set (mse, l1, huber, charbonnier, cosine, log_energy, tv, sobolev):
the returned dictionary structure, agreement with the manual formulas, weighted summation,
gradient flow, and the smoothness semantics of the TV and Sobolev terms on the residual.
"""

from __future__ import annotations

import pytest
import torch

from configuration.training.profile_autoencoder  import ProfileAeLossConfig
from pipelines.profile_autoencoder.training.loss import Loss


POINTWISE = ["mse", "l1", "huber", "charbonnier"]


def _single(kind, **kwargs):
    """Returns a loss with only the given curve term enabled."""
    flags = {"use_mse_curve": kind == "mse", f"use_{kind}_curve": True}
    return Loss(ProfileAeLossConfig(**flags, **kwargs))


def _curve_pair(seed=0):
    """Returns a (reconstruction, curve) pair of random profiles of shape (4, 16, 1, 1)."""
    g     = torch.Generator().manual_seed(seed)
    recon = torch.randn(4, 16, 1, 1, generator=g)
    curve = torch.randn(4, 16, 1, 1, generator=g)
    return recon, curve


def test_loss_generation_is_zero():
    """The profile autoencoder loss reports generation zero, since it trains no Gaussians."""
    assert Loss(ProfileAeLossConfig()).loss_generation == 0


@pytest.mark.parametrize("kind", POINTWISE + ["cosine", "log_energy", "tv"])
def test_forward_returns_finite_scalar(kind):
    """Each term returns a finite non-negative scalar total loss."""
    recon, curve = _curve_pair()
    out          = _single(kind)(recon, curve)

    total = out["total_loss"]

    assert total.ndim == 0
    assert torch.isfinite(total)
    assert total.item() >= 0.0


@pytest.mark.parametrize("kind", POINTWISE)
def test_dict_structure_and_consistency(kind):
    """The loss dictionary carries the fixed key set, the single enabled component, and a total equal to it."""
    recon, curve = _curve_pair()
    out          = _single(kind)(recon, curve)

    assert set(out.keys()) == {"total_loss", "components", "monitor", "occupancy", "physical"}
    assert set(out["components"].keys()) == {f"curve_{kind}"}
    assert out["monitor"] == {}

    assert torch.equal(out["total_loss"], out["components"][f"curve_{kind}"])


def test_default_config_enables_only_mse():
    """The default configuration reproduces the plain MSE objective."""
    recon, curve = _curve_pair()
    out          = Loss(ProfileAeLossConfig())(recon, curve)

    manual = ((recon - curve) ** 2).mean()

    assert set(out["components"]) == {"curve_mse"}
    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


@pytest.mark.parametrize("kind", ["mse", "l1", "huber", "tv"])
def test_zero_on_identical_input(kind):
    """The unfloored kinds vanish when the reconstruction equals the curve."""
    curve = torch.randn(4, 16, 1, 1)
    out   = _single(kind)(curve.clone(), curve.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("kind", POINTWISE + ["cosine", "log_energy", "tv"])
def test_gradient_flow(kind):
    """Each kind backpropagates finite non-zero gradients into the reconstruction."""
    recon, curve = _curve_pair()
    recon        = recon.clone().requires_grad_(True)

    out = _single(kind)(recon, curve)
    out["total_loss"].backward()

    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()
    assert recon.grad.abs().sum() > 0.0


def test_l1_matches_manual():
    """The l1 term equals the mean absolute difference computed directly."""
    recon, curve = _curve_pair()
    out          = _single("l1")(recon, curve)

    manual = (recon - curve).abs().mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_tv_matches_manual():
    """The tv term equals the mean absolute first difference of the residual along the height axis."""
    recon, curve = _curve_pair()
    out          = _single("tv")(recon, curve)

    diff   = recon - curve
    manual = (diff[:, 1:] - diff[:, :-1]).abs().mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_huber_delta_changes_value():
    """Changing the huber delta changes the loss value on the same pair."""
    recon, curve = _curve_pair()

    small = _single("huber", huber_delta=0.1)(recon, curve)
    large = _single("huber", huber_delta=5.0)(recon, curve)

    assert small["total_loss"].item() != large["total_loss"].item()


def test_charbonnier_eps_floor_on_identical():
    """On an identical pair the charbonnier term returns its epsilon floor rather than zero."""
    curve = torch.zeros(4, 16, 1, 1)
    out   = _single("charbonnier", charbonnier_eps=1e-2)(curve, curve)

    assert out["total_loss"].item() == pytest.approx(1e-2, abs=1e-5)


def test_cosine_zero_on_identical_shapes():
    """The cosine term vanishes when the reconstruction is a positive rescaling of the curve."""
    curve = torch.rand(4, 16, 1, 1) + 0.5
    out   = _single("cosine")(2.0 * curve, curve)

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_log_energy_detects_rescaling():
    """The log-energy term equals |log 2| when the reconstruction doubles the curve energy."""
    curve = torch.rand(4, 16, 1, 1) + 0.5
    out   = _single("log_energy")(2.0 * curve, curve)

    assert out["total_loss"].item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-4)


def test_multi_term_total_is_weighted_sum():
    """Enabling several terms folds each component into the total at its configured weight."""
    recon, curve = _curve_pair()
    cfg          = ProfileAeLossConfig(use_mse_curve=True, weight_mse_curve=1.0, use_l1_curve=True, weight_l1_curve=0.5, use_cosine_curve=True, weight_cosine_curve=0.25)
    out          = Loss(cfg)(recon, curve)

    assert set(out["components"]) == {"curve_mse", "curve_l1", "curve_cosine"}

    expected = out["components"]["curve_mse"] + 0.5 * out["components"]["curve_l1"] + 0.25 * out["components"]["curve_cosine"]
    assert out["total_loss"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_no_enabled_term_raises():
    """A configuration with every term disabled is rejected at construction."""
    with pytest.raises(ValueError):
        Loss(ProfileAeLossConfig(use_mse_curve=False))


def test_sobolev_off_by_default():
    """No Sobolev component is reported unless the term is switched on."""
    out = Loss(ProfileAeLossConfig())(*_curve_pair())

    assert set(out["components"]) == {"curve_mse"}


def test_sobolev_adds_component_and_weighted_total():
    """Enabling the Sobolev term adds its component and folds it into the total at its weight."""
    recon, curve = _curve_pair()
    out          = Loss(ProfileAeLossConfig(use_sobolev=True, weight_sobolev=2.0))(recon, curve)

    assert set(out["components"]) == {"curve_mse", "curve_sobolev"}

    expected = out["components"]["curve_mse"] + 2.0 * out["components"]["curve_sobolev"]
    assert out["total_loss"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_sobolev_zero_for_linear_residual():
    """A constant-slope residual along the z axis leaves the Sobolev term at zero."""
    curve = _curve_pair()[1]
    ramp  = torch.linspace(0.0, 1.0, curve.shape[1]).view(1, -1, 1, 1)
    out   = Loss(ProfileAeLossConfig(use_sobolev=True, weight_sobolev=1.0))(curve + ramp, curve)

    assert out["components"]["curve_mse"].item() > 0.0
    assert out["components"]["curve_sobolev"].item() == pytest.approx(0.0, abs=1e-10)


def test_sobolev_penalizes_jitter_over_smooth_error_at_equal_mse():
    """At equal reconstruction error, alternating bin-to-bin jitter costs more in total than a constant offset."""
    target = torch.zeros(1, 16, 1, 1)
    offset = torch.full((1, 16, 1, 1), 0.5)
    jitter = 0.5 * torch.tensor([1.0, -1.0]).repeat(8).view(1, 16, 1, 1)
    loss   = Loss(ProfileAeLossConfig(use_sobolev=True, weight_sobolev=1.0))

    smooth_out = loss(target + offset, target)
    jitter_out = loss(target + jitter, target)

    assert smooth_out["components"]["curve_mse"].item() == pytest.approx(jitter_out["components"]["curve_mse"].item(), rel=1e-6)
    assert jitter_out["total_loss"].item() > smooth_out["total_loss"].item()


def test_tv_penalizes_jitter_over_constant_offset():
    """The TV term is zero for a constant offset and positive for bin-to-bin jitter."""
    target = torch.zeros(1, 16, 1, 1)
    offset = torch.full((1, 16, 1, 1), 0.5)
    jitter = 0.5 * torch.tensor([1.0, -1.0]).repeat(8).view(1, 16, 1, 1)
    loss   = _single("tv")

    assert loss(target + offset, target)["total_loss"].item() == pytest.approx(0.0, abs=1e-8)
    assert loss(target + jitter, target)["total_loss"].item() > 0.0
