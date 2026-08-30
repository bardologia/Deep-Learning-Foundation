"""Tests for the image autoencoder reconstruction loss.

Covers the four reconstruction kinds (mse, l1, huber, charbonnier): the returned dictionary
structure, agreement with the manual formulas, gradient flow, the effect of the huber delta
and the charbonnier epsilon floor, invariance to batch ordering, and the rejection of
the curve-only kinds that would reduce along the channel axis.
"""

from __future__ import annotations

import pytest
import torch

from configuration.training.image_autoencoder  import ImageAeLossConfig
from pipelines.image_autoencoder.training.loss import Loss


RECON_KINDS = ["mse", "l1", "huber", "charbonnier"]


def _loss(kind):
    """Returns an image autoencoder loss configured with the given reconstruction kind."""
    return Loss(ImageAeLossConfig(recon_kind=kind))


def _image_pair(seed=0):
    """Returns a (reconstruction, image) pair of random patches of shape (3, 2, 8, 8)."""
    g     = torch.Generator().manual_seed(seed)
    image = torch.randn(3, 2, 8, 8, generator=g)
    recon = torch.randn(3, 2, 8, 8, generator=g)
    return recon, image


def test_loss_generation_is_zero():
    """The image autoencoder loss reports generation zero, since it trains no Gaussians."""
    assert _loss("mse").loss_generation == 0


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_forward_returns_finite_scalar(kind):
    """Each reconstruction kind returns a finite non-negative scalar total loss."""
    recon, image = _image_pair()
    out          = _loss(kind)(recon, image)

    total = out["total_loss"]

    assert total.ndim == 0
    assert torch.isfinite(total)
    assert total.item() >= 0.0


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_dict_structure_and_consistency(kind):
    """The loss dictionary carries the fixed key set, a single image_recon component, and a total equal to it."""
    recon, image = _image_pair()
    out          = _loss(kind)(recon, image)

    assert set(out.keys()) == {"total_loss", "components", "monitor", "occupancy", "physical"}
    assert set(out["components"].keys()) == {"image_recon"}
    assert out["monitor"] == {}

    assert torch.equal(out["total_loss"], out["components"]["image_recon"])


@pytest.mark.parametrize("kind", ["mse", "l1", "huber"])
def test_zero_on_identical_input(kind):
    """The unfloored kinds vanish when the reconstruction equals the image."""
    image = torch.randn(2, 2, 8, 8)
    out   = _loss(kind)(image.clone(), image.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_gradient_flow(kind):
    """Each kind backpropagates finite non-zero gradients into the reconstruction."""
    recon, image = _image_pair()
    recon        = recon.clone().requires_grad_(True)

    out = _loss(kind)(recon, image)
    out["total_loss"].backward()

    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()
    assert recon.grad.abs().sum() > 0.0


def test_mse_matches_manual():
    """The mse kind equals the mean squared difference computed directly."""
    recon, image = _image_pair()
    out          = _loss("mse")(recon, image)

    manual = ((recon - image) ** 2).mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_l1_matches_manual():
    """The l1 kind equals the mean absolute difference computed directly."""
    recon, image = _image_pair()
    out          = _loss("l1")(recon, image)

    manual = (recon - image).abs().mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_huber_delta_changes_value():
    """Changing the huber delta changes the loss value on the same pair."""
    recon, image = _image_pair()

    small = Loss(ImageAeLossConfig(recon_kind="huber", huber_delta=0.1))(recon, image)
    large = Loss(ImageAeLossConfig(recon_kind="huber", huber_delta=5.0))(recon, image)

    assert small["total_loss"].item() != large["total_loss"].item()


def test_charbonnier_eps_floor_on_identical():
    """On an identical pair the charbonnier kind returns its epsilon floor rather than zero."""
    image = torch.zeros(2, 2, 4, 4)
    eps   = 1e-2
    out   = Loss(ImageAeLossConfig(recon_kind="charbonnier", charbonnier_eps=eps))(image, image)

    assert out["total_loss"].item() == pytest.approx(eps, abs=1e-5)


def test_unknown_kind_raises():
    """An unregistered reconstruction kind raises when the loss is evaluated."""
    with pytest.raises(ValueError):
        _loss("does_not_exist")(*_image_pair())


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_invariant_to_batch_order(kind):
    """Permuting the batch leaves every reconstruction kind unchanged."""
    recon, image = _image_pair()

    perm     = torch.tensor([2, 0, 1])
    straight = _loss(kind)(recon, image)["total_loss"]
    shuffled = _loss(kind)(recon[perm], image[perm])["total_loss"]

    assert straight.item() == pytest.approx(shuffled.item(), rel=1e-6)


@pytest.mark.parametrize("kind", ["cosine", "log_energy", "tv", "sobolev"])
def test_curve_only_kinds_are_rejected(kind):
    """The shape, energy and smoothness kinds reduce along the channel axis and are refused."""
    with pytest.raises(ValueError):
        _loss(kind)
