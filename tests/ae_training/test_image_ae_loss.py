from __future__ import annotations

import pytest
import torch

from configuration.training.image_autoencoder  import ImageAeLossConfig
from pipelines.image_autoencoder.training.loss import Loss


RECON_KINDS = ["mse", "l1", "huber", "charbonnier"]


def _loss(kind):
    return Loss(ImageAeLossConfig(recon_kind=kind))


def _image_pair(seed=0):
    g     = torch.Generator().manual_seed(seed)
    image = torch.randn(3, 2, 8, 8, generator=g)
    recon = torch.randn(3, 2, 8, 8, generator=g)
    return recon, image


def test_loss_generation_is_zero():
    assert _loss("mse").loss_generation == 0


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_forward_returns_finite_scalar(kind):
    recon, image = _image_pair()
    out          = _loss(kind)(recon, image)

    total = out["total_loss"]

    assert total.ndim == 0
    assert torch.isfinite(total)
    assert total.item() >= 0.0


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_dict_structure_and_consistency(kind):
    recon, image = _image_pair()
    out          = _loss(kind)(recon, image)

    assert set(out.keys()) == {"total_loss", "components", "monitor", "occupancy", "physical"}
    assert set(out["components"].keys()) == {"image_recon"}
    assert out["monitor"] == {}

    assert torch.equal(out["total_loss"], out["components"]["image_recon"])


@pytest.mark.parametrize("kind", ["mse", "l1", "huber"])
def test_zero_on_identical_input(kind):
    image = torch.randn(2, 2, 8, 8)
    out   = _loss(kind)(image.clone(), image.clone())

    assert out["total_loss"].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_gradient_flow(kind):
    recon, image = _image_pair()
    recon        = recon.clone().requires_grad_(True)

    out = _loss(kind)(recon, image)
    out["total_loss"].backward()

    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()
    assert recon.grad.abs().sum() > 0.0


def test_mse_matches_manual():
    recon, image = _image_pair()
    out          = _loss("mse")(recon, image)

    manual = ((recon - image) ** 2).mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_l1_matches_manual():
    recon, image = _image_pair()
    out          = _loss("l1")(recon, image)

    manual = (recon - image).abs().mean()

    assert out["total_loss"].item() == pytest.approx(manual.item(), rel=1e-6)


def test_huber_delta_changes_value():
    recon, image = _image_pair()

    small = Loss(ImageAeLossConfig(recon_kind="huber", huber_delta=0.1))(recon, image)
    large = Loss(ImageAeLossConfig(recon_kind="huber", huber_delta=5.0))(recon, image)

    assert small["total_loss"].item() != large["total_loss"].item()


def test_charbonnier_eps_floor_on_identical():
    image = torch.zeros(2, 2, 4, 4)
    eps   = 1e-2
    out   = Loss(ImageAeLossConfig(recon_kind="charbonnier", charbonnier_eps=eps))(image, image)

    assert out["total_loss"].item() == pytest.approx(eps, abs=1e-5)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        _loss("does_not_exist")(*_image_pair())


@pytest.mark.parametrize("kind", RECON_KINDS)
def test_invariant_to_batch_order(kind):
    recon, image = _image_pair()

    perm     = torch.tensor([2, 0, 1])
    straight = _loss(kind)(recon, image)["total_loss"]
    shuffled = _loss(kind)(recon[perm], image[perm])["total_loss"]

    assert straight.item() == pytest.approx(shuffled.item(), rel=1e-6)
