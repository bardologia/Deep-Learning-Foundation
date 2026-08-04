from __future__ import annotations

import numpy as np
import pytest
import torch

from models.image_autoencoder import IMAGE_AE_CONFIG_REGISTRY, get_image_autoencoder


IN_CHANNELS   = 2
EMBEDDING_DIM = 8
PATCH         = 16
NAMES         = sorted(IMAGE_AE_CONFIG_REGISTRY.keys())


def _tiny_config(name):
    cfg               = IMAGE_AE_CONFIG_REGISTRY[name](in_channels=IN_CHANNELS, embedding_dim=EMBEDDING_DIM)
    cfg.base_channels = 8
    cfg.depth         = 1
    if hasattr(cfg, "hidden_dim"):
        cfg.hidden_dim = 16
    if hasattr(cfg, "num_heads"):
        cfg.num_heads = 2
    if hasattr(cfg, "patch_size"):
        cfg.patch_size = 4
    if hasattr(cfg, "dilation_depth"):
        cfg.dilation_depth = 2
    return cfg


def _build(name):
    model, cfg = get_image_autoencoder(name, _tiny_config(name))
    return model, cfg


def _image_input(interferograms):
    block  = np.asarray(interferograms[:IN_CHANNELS, :PATCH, : 2 * PATCH])
    mag    = np.abs(block).astype(np.float32)
    tensor = torch.from_numpy(mag).reshape(IN_CHANNELS, PATCH, 2, PATCH).permute(2, 0, 1, 3).contiguous()
    return tensor


def _finite(t):
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("name", NAMES)
def test_registry_entries_build(name):
    model, cfg = _build(name)
    assert isinstance(model, torch.nn.Module)
    assert cfg.in_channels == IN_CHANNELS


@pytest.mark.real_data
@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_shape(name, interferograms):
    model, _ = _build(name)
    model.eval()

    x = _image_input(interferograms)
    with torch.no_grad():
        x_hat, z = model.reconstruct(x)

    assert x_hat.shape == x.shape
    assert z.shape[0] == x.shape[0]
    assert z.shape[1] == EMBEDDING_DIM
    assert _finite(x_hat)
    assert _finite(z)


@pytest.mark.real_data
@pytest.mark.parametrize("name", NAMES)
def test_latent_dim(name, interferograms):
    model, _ = _build(name)
    model.eval()

    x = _image_input(interferograms)
    with torch.no_grad():
        z = model.encode(x)

    assert z.shape[1] == EMBEDDING_DIM


@pytest.mark.real_data
@pytest.mark.parametrize("name", NAMES)
def test_encode_features_resizes(name, interferograms):
    model, _ = _build(name)
    model.eval()

    x      = _image_input(interferograms)
    out_hw = (PATCH, PATCH)
    with torch.no_grad():
        z = model.encode_features(x, out_hw)

    assert z.shape[-2:] == out_hw
    assert z.shape[1] == EMBEDDING_DIM
    assert _finite(z)


@pytest.mark.real_data
@pytest.mark.parametrize("name", NAMES)
def test_backward_finite_grads(name, interferograms):
    model, _ = _build(name)
    model.train()

    x     = _image_input(interferograms)
    x_hat = model(x)
    loss  = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()

    assert _finite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(_finite(g) for g in grads)


@pytest.mark.parametrize("name", NAMES)
def test_eval_deterministic(name):
    model, _ = _build(name)
    model.eval()

    x = torch.randn(2, IN_CHANNELS, PATCH, PATCH)
    with torch.no_grad():
        first  = model(x)
        second = model(x)

    assert torch.allclose(first, second)


@pytest.mark.parametrize("name", NAMES)
def test_zero_input_finite(name):
    model, _ = _build(name)
    model.eval()

    x = torch.zeros(2, IN_CHANNELS, PATCH, PATCH)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)


@pytest.mark.parametrize("name", NAMES)
def test_constant_input_finite(name):
    model, _ = _build(name)
    model.eval()

    x = torch.full((2, IN_CHANNELS, PATCH, PATCH), 0.5)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)


def test_vit_rejects_input_not_divisible_by_patch_size():
    model, _ = _build("vit_ae")
    model.eval()

    with pytest.raises(ValueError, match="patch_size"):
        with torch.no_grad():
            model.reconstruct(torch.randn(1, IN_CHANNELS, PATCH - 1, PATCH))


@pytest.mark.parametrize("name", ["convnext2d_ae", "vit_ae"])
def test_layernorm_models_reject_normalization_override(name):
    cfg               = _tiny_config(name)
    cfg.normalization = "batch"

    with pytest.raises(ValueError, match="normalization"):
        get_image_autoencoder(name, cfg)
