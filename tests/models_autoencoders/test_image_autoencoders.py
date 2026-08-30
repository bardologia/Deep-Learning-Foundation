"""Tests covering the registry of image autoencoders over derived channel patches.

Every registered model is built at a tiny size and checked for reconstruction shape,
latent dimensionality, feature resizing, gradient health, determinism, degenerate
inputs and rejection of config overrides it never reads.
"""

from __future__ import annotations

import pytest
import torch

from models.image_autoencoder import IMAGE_AE_CONFIG_REGISTRY, get_image_autoencoder


IN_CHANNELS   = 2
EMBEDDING_DIM = 8
PATCH         = 16
NAMES         = sorted(IMAGE_AE_CONFIG_REGISTRY.keys())


UNREAD_FIELDS = {
    "dilated2d_ae" : ("depth",),
    "vit_ae"       : ("base_channels",),
}


def _tiny_config(name):
    """Returns a minimal config for the named image autoencoder, skipping fields it never reads."""
    cfg    = IMAGE_AE_CONFIG_REGISTRY[name](in_channels=IN_CHANNELS, embedding_dim=EMBEDDING_DIM)
    unread = UNREAD_FIELDS.get(name, ())

    if "base_channels" not in unread:
        cfg.base_channels = 8
    if "depth" not in unread:
        cfg.depth = 1
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
    """Returns the named image autoencoder and its resolved config at tiny size."""
    model, cfg = get_image_autoencoder(name, _tiny_config(name))
    return model, cfg


def _image_input():
    """Returns a deterministic random image batch of shape (2, in_channels, PATCH, PATCH)."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, IN_CHANNELS, PATCH, PATCH, generator=generator, dtype=torch.float32)


def _finite(t):
    """Returns True when every entry of the tensor is finite."""
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("name", NAMES)
def test_registry_entries_build(name):
    """Verifies every registry entry constructs a module carrying the requested input channel count."""
    model, cfg = _build(name)
    assert isinstance(model, torch.nn.Module)
    assert cfg.in_channels == IN_CHANNELS


@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_shape(name):
    """Verifies reconstruction preserves the input shape and returns an embedding of the configured width."""
    model, _ = _build(name)
    model.eval()

    x = _image_input()
    with torch.no_grad():
        x_hat, z = model.reconstruct(x)

    assert x_hat.shape == x.shape
    assert z.shape[0] == x.shape[0]
    assert z.shape[1] == EMBEDDING_DIM
    assert _finite(x_hat)
    assert _finite(z)


@pytest.mark.parametrize("name", NAMES)
def test_latent_dim(name):
    """Verifies the encoder emits the configured embedding dimension."""
    model, _ = _build(name)
    model.eval()

    x = _image_input()
    with torch.no_grad():
        z = model.encode(x)

    assert z.shape[1] == EMBEDDING_DIM


@pytest.mark.parametrize("name", NAMES)
def test_encode_features_resizes(name):
    """Verifies encoded features are resized to the requested spatial size at the embedding width."""
    model, _ = _build(name)
    model.eval()

    x      = _image_input()
    out_hw = (PATCH, PATCH)
    with torch.no_grad():
        z = model.encode_features(x, out_hw)

    assert z.shape[-2:] == out_hw
    assert z.shape[1] == EMBEDDING_DIM
    assert _finite(z)


@pytest.mark.parametrize("name", NAMES)
def test_backward_finite_grads(name):
    """Verifies the reconstruction loss backpropagates finite gradients to the model parameters."""
    model, _ = _build(name)
    model.train()

    x     = _image_input()
    x_hat = model(x)
    loss  = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()

    assert _finite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(_finite(g) for g in grads)


@pytest.mark.parametrize("name", NAMES)
def test_eval_deterministic(name):
    """Verifies two evaluation-mode forward passes on the same input agree."""
    model, _ = _build(name)
    model.eval()

    x = torch.randn(2, IN_CHANNELS, PATCH, PATCH)
    with torch.no_grad():
        first  = model(x)
        second = model(x)

    assert torch.allclose(first, second)


@pytest.mark.parametrize("name", NAMES)
def test_zero_input_finite(name):
    """Verifies an all-zero patch reconstructs to a finite tensor of the same shape."""
    model, _ = _build(name)
    model.eval()

    x = torch.zeros(2, IN_CHANNELS, PATCH, PATCH)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)


@pytest.mark.parametrize("name", NAMES)
def test_constant_input_finite(name):
    """Verifies a constant-valued patch reconstructs to a finite tensor of the same shape."""
    model, _ = _build(name)
    model.eval()

    x = torch.full((2, IN_CHANNELS, PATCH, PATCH), 0.5)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)


def test_vit_rejects_input_not_divisible_by_patch_size():
    """Verifies the ViT autoencoder refuses a patch that its patch size does not divide."""
    model, _ = _build("vit_ae")
    model.eval()

    with pytest.raises(ValueError, match="patch_size"):
        with torch.no_grad():
            model.reconstruct(torch.randn(1, IN_CHANNELS, PATCH - 1, PATCH))


@pytest.mark.parametrize("name", ["convnext2d_ae", "vit_ae"])
def test_layernorm_models_reject_normalization_override(name):
    """Verifies LayerNorm-based models refuse a normalisation override they cannot honour."""
    cfg               = _tiny_config(name)
    cfg.normalization = "batch"

    with pytest.raises(ValueError, match="normalization"):
        get_image_autoencoder(name, cfg)


@pytest.mark.parametrize("name,field,value", [
    ("vit_ae",       "downsample_factor", 4),
    ("vit_ae",       "base_channels",     64),
    ("dilated2d_ae", "downsample_factor", 2),
    ("dilated2d_ae", "depth",             3),
])
def test_models_reject_overrides_they_never_read(name, field, value):
    """Verifies a model raises when given a config field its architecture ignores."""
    cfg = _tiny_config(name)
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        get_image_autoencoder(name, cfg)
