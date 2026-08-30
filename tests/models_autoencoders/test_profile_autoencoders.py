"""Tests covering the registry of elevation-profile autoencoders.

Every registered model is built at a tiny size and checked for reconstruction shape,
latent dimensionality, gradient health, determinism, embedding normalisation, complete
parameter-group coverage and rejection of unsupported settings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.profile_autoencoder import PROFILE_AE_CONFIG_REGISTRY, get_profile_autoencoder


PROFILE_LENGTH = 16
EMBEDDING_DIM  = 8
NAMES          = sorted(PROFILE_AE_CONFIG_REGISTRY.keys())


def _tiny_config(name):
    """Returns a minimal config for the named profile autoencoder."""
    cfg = PROFILE_AE_CONFIG_REGISTRY[name](profile_length=PROFILE_LENGTH, embedding_dim=EMBEDDING_DIM)
    if hasattr(cfg, "hidden_dim"):
        cfg.hidden_dim = 16
    if hasattr(cfg, "seq_channels"):
        cfg.seq_channels = 8
    if hasattr(cfg, "num_heads"):
        cfg.num_heads = 2
    if hasattr(cfg, "depth"):
        cfg.depth = 2
    if hasattr(cfg, "patch_size"):
        cfg.patch_size = 4
    return cfg


def _build(name):
    """Returns the named profile autoencoder and its resolved config at tiny size."""
    model, cfg = get_profile_autoencoder(name, _tiny_config(name))
    return model, cfg


def _real_batch():
    """Returns the (batch, height, width) dimensions used for the real-data tensors."""
    return (2, 3, 3)


def _profile_input(cube_full, dims):
    """Returns real cube magnitudes as a tensor of shape (B, PROFILE_LENGTH, H, W)."""
    B, H, W = dims
    block   = np.asarray(cube_full[:PROFILE_LENGTH, :H, : B * W])
    mag     = np.abs(block).astype(np.float32)
    tensor  = torch.from_numpy(mag).reshape(PROFILE_LENGTH, H, B, W).permute(2, 0, 1, 3).contiguous()
    return tensor


def _finite(t):
    """Returns True when every entry of the tensor is finite."""
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("name", NAMES)
def test_registry_entries_build(name):
    """Verifies every registry entry constructs a module carrying the requested profile length."""
    model, cfg = _build(name)
    assert isinstance(model, torch.nn.Module)
    assert cfg.profile_length == PROFILE_LENGTH


@pytest.mark.dataset
@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_shape(name, cube_full):
    """Verifies reconstruction preserves the profile shape and yields a per-pixel embedding map."""
    model, cfg = _build(name)
    model.eval()

    dims = _real_batch()
    x    = _profile_input(cube_full, dims)

    with torch.no_grad():
        x_hat, z = model.reconstruct(x)

    assert x_hat.shape == x.shape
    B, H, W = dims
    assert z.shape == (B, EMBEDDING_DIM, H, W)
    assert _finite(x_hat)
    assert _finite(z)


@pytest.mark.dataset
@pytest.mark.parametrize("name", NAMES)
def test_latent_dim(name, cube_full):
    """Verifies the encoder emits the configured embedding dimension."""
    model, _ = _build(name)
    model.eval()

    x = _profile_input(cube_full, _real_batch())
    with torch.no_grad():
        z = model.encode(x)

    assert z.shape[1] == EMBEDDING_DIM


@pytest.mark.dataset
@pytest.mark.parametrize("name", NAMES)
def test_backward_finite_grads(name, cube_full):
    """Verifies the reconstruction loss backpropagates finite gradients to the model parameters."""
    model, _ = _build(name)
    model.train()

    x     = _profile_input(cube_full, _real_batch())
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

    x = torch.randn(2, PROFILE_LENGTH, 3, 3)
    with torch.no_grad():
        first  = model(x)
        second = model(x)

    assert torch.allclose(first, second)


@pytest.mark.parametrize("name", NAMES)
def test_zero_input_finite(name):
    """Verifies an all-zero profile batch reconstructs to a finite tensor of the same shape."""
    model, _ = _build(name)
    model.eval()

    x = torch.zeros(2, PROFILE_LENGTH, 3, 3)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)


@pytest.mark.parametrize("name", NAMES)
def test_constant_input_finite(name):
    """Verifies a constant-valued profile batch reconstructs to a finite tensor of the same shape."""
    model, _ = _build(name)
    model.eval()

    x = torch.full((2, PROFILE_LENGTH, 3, 3), 0.7)
    with torch.no_grad():
        x_hat = model(x)

    assert x_hat.shape == x.shape
    assert _finite(x_hat)


@pytest.mark.parametrize("name", NAMES)
def test_layernorm_embedding_is_registered_module(name):
    """Verifies the LayerNorm embedding normaliser appears in the state dict as a registered submodule."""
    cfg                = _tiny_config(name)
    cfg.embedding_norm = "layernorm"
    model, _           = get_profile_autoencoder(name, cfg)

    keys = model.state_dict().keys()
    assert any("embedding_layernorm" in k for k in keys)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("embedding_norm", ["none", "l2", "layernorm"])
def test_param_groups_cover_every_parameter(name, embedding_norm):
    """Verifies the optimiser parameter groups cover every model parameter for each embedding normaliser."""
    cfg                = _tiny_config(name)
    cfg.embedding_norm = embedding_norm
    model, config      = get_profile_autoencoder(name, cfg)

    grouped   = {id(p) for group in config.get_param_groups(model) for p in group["params"]}
    uncovered = [param_name for param_name, p in model.named_parameters() if id(p) not in grouped]

    assert uncovered == []


@pytest.mark.parametrize("name", NAMES)
def test_embedding_layernorm_affine_is_optimized(name):
    """Verifies one optimiser step updates the embedding LayerNorm affine weight."""
    cfg                = _tiny_config(name)
    cfg.embedding_norm = "layernorm"
    model, config      = get_profile_autoencoder(name, cfg)
    model.train()

    optimizer = torch.optim.AdamW(config.get_param_groups(model))
    before    = model.embedding_layernorm.norm.weight.detach().clone()

    x    = torch.randn(2, PROFILE_LENGTH, 3, 3)
    loss = torch.nn.functional.mse_loss(model(x), x)
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, model.embedding_layernorm.norm.weight.detach())


@pytest.mark.parametrize("name", NAMES)
def test_collapsed_embedding_backward_finite(name):
    """Verifies a collapsed embedding still yields a finite loss and finite gradients."""
    cfg                = _tiny_config(name)
    cfg.embedding_norm = "layernorm"
    model, _           = get_profile_autoencoder(name, cfg)
    model.train()

    original_encode = model.encode

    def collapsed_encode(curve):
        """Returns a nearly constant embedding to emulate a collapsed latent space."""
        z = original_encode(curve)
        return torch.full_like(z, 0.5) + 1e-6 * z

    model.encode = collapsed_encode

    x     = torch.randn(2, PROFILE_LENGTH, 3, 3)
    x_hat = model.decode(model.encode(x))
    loss  = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert _finite(loss)
    assert all(_finite(g) for g in grads)


def test_transformer1d_rejects_unsupported_activation():
    """Verifies the 1D transformer autoencoder refuses an activation it cannot use."""
    cfg            = _tiny_config("transformer1d_ae")
    cfg.activation = "silu"

    with pytest.raises(ValueError, match="activation"):
        get_profile_autoencoder("transformer1d_ae", cfg)


def test_transformer1d_attention_heads_do_not_change_the_model():
    """Verifies the head count leaves the 1D transformer autoencoder output bit-identical."""
    outputs = []
    for heads in (2, 4):
        cfg           = _tiny_config("transformer1d_ae")
        cfg.num_heads = heads

        torch.manual_seed(0)
        model, _ = get_profile_autoencoder("transformer1d_ae", cfg)
        model.eval()

        torch.manual_seed(1)
        with torch.no_grad():
            outputs.append(model(torch.randn(2, PROFILE_LENGTH, 3, 3)))

    assert torch.allclose(outputs[0], outputs[1], atol=0.0, rtol=0.0)


def test_transformer1d_does_not_advertise_attention_heads_as_a_tuner_lever():
    """Verifies num_heads is tunable for the attention CNN but not for the 1D transformer."""
    tunable = PROFILE_AE_CONFIG_REGISTRY["transformer1d_ae"].tunable_arch_params()

    assert "num_heads" not in tunable
    assert "num_heads"     in PROFILE_AE_CONFIG_REGISTRY["cnn_attn_ae"].tunable_arch_params()


@pytest.mark.parametrize("name", ["conv1d_ae", "tcn_ae", "cnn_attn_ae"])
def test_xavier_init_reaches_conv1d_layers(name):
    """Verifies Xavier initialisation zeroes the biases of every Conv1d layer in the model."""
    cfg           = _tiny_config(name)
    cfg.init_mode = "xavier"
    model, _      = get_profile_autoencoder(name, cfg)

    conv_biases = [m.bias for m in model.modules() if isinstance(m, torch.nn.Conv1d) and m.bias is not None]

    assert conv_biases
    assert all(torch.all(bias == 0.0) for bias in conv_biases)


def _forward_ok(name, cfg):
    """Builds the model from the config and checks a synthetic round trip stays shaped and finite."""
    model, cfg = get_profile_autoencoder(name, cfg)
    model.eval()

    x = torch.rand(2, PROFILE_LENGTH, 2, 2)
    with torch.no_grad():
        recon, z = model.reconstruct(x)

    assert recon.shape == x.shape
    assert z.shape == (2, EMBEDDING_DIM, 2, 2)
    assert _finite(recon) and _finite(z)
    return model


@pytest.mark.parametrize("name", ["mlp_ae", "conv1d_ae", "tcn_ae"])
def test_batchnorm_normalization_builds_and_runs(name):
    """The batchnorm normalization mode adds norm layers and keeps the round trip finite."""
    cfg = _tiny_config(name)
    cfg.normalization = "batchnorm"

    model    = _forward_ok(name, cfg)
    norms_1d = [m for m in model.modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]

    assert norms_1d


def test_mlp_layernorm_normalization_builds_and_runs():
    """The layernorm normalization mode of the MLP autoencoder runs a finite round trip."""
    cfg = _tiny_config("mlp_ae")
    cfg.normalization = "layernorm"
    _forward_ok("mlp_ae", cfg)


def test_resmlp_none_normalization_drops_the_layernorm():
    """Disabling the resmlp pre-norm replaces every block norm with an identity."""
    cfg = _tiny_config("resmlp_ae")
    cfg.normalization = "none"

    model = _forward_ok("resmlp_ae", cfg)

    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.encoder.blocks.modules())


@pytest.mark.parametrize("name", ["mlp_ae", "conv1d_ae", "tcn_ae", "resmlp_ae"])
def test_unsupported_normalization_raises(name):
    """An unsupported normalization mode is rejected at construction."""
    cfg = _tiny_config(name)
    cfg.normalization = "instance"

    with pytest.raises(ValueError):
        get_profile_autoencoder(name, cfg)


@pytest.mark.parametrize("name", ["conv1d_ae", "tcn_ae", "cnn_attn_ae"])
@pytest.mark.parametrize("pooling", ["max", "attention"])
def test_pooling_modes_build_and_run(name, pooling):
    """The max and attention pooling modes keep the round trip shaped and finite."""
    cfg = _tiny_config(name)
    cfg.pooling = pooling
    _forward_ok(name, cfg)


@pytest.mark.parametrize("name", ["conv1d_ae", "tcn_ae", "cnn_attn_ae"])
def test_unknown_pooling_raises(name):
    """An unknown pooling mode is rejected at construction."""
    cfg = _tiny_config(name)
    cfg.pooling = "nope"

    with pytest.raises(ValueError):
        get_profile_autoencoder(name, cfg)


def test_attention_pooling_adds_parameters():
    """Attention pooling introduces the scoring parameters average pooling does not have."""
    plain = _tiny_config("conv1d_ae")
    attn  = _tiny_config("conv1d_ae")
    attn.pooling = "attention"

    n_plain = sum(p.numel() for p in get_profile_autoencoder("conv1d_ae", plain)[0].parameters())
    n_attn  = sum(p.numel() for p in get_profile_autoencoder("conv1d_ae", attn)[0].parameters())

    assert n_attn > n_plain


def test_conv1d_depth_grows_the_body():
    """Raising the conv1d depth adds convolution layers to the encoder body."""
    deep = _tiny_config("conv1d_ae")
    deep.depth = 4

    model = _forward_ok("conv1d_ae", deep)
    convs = [m for m in model.encoder.body if isinstance(m, torch.nn.Conv1d)]

    assert len(convs) == 4


def test_conv1d_dropout_inserts_dropout_layers():
    """A positive conv1d dropout inserts dropout modules into the body."""
    cfg = _tiny_config("conv1d_ae")
    cfg.dropout = 0.2

    model = _forward_ok("conv1d_ae", cfg)

    assert any(isinstance(m, torch.nn.Dropout) for m in model.encoder.body)


def test_conv1d_channel_list_sets_per_layer_widths():
    """An explicit channel list overrides the uniform width layer by layer."""
    cfg = _tiny_config("conv1d_ae")
    cfg.seq_channels_list = [8, 12, 16]

    model = _forward_ok("conv1d_ae", cfg)
    convs = [m for m in model.encoder.body if isinstance(m, torch.nn.Conv1d)]

    assert [c.out_channels for c in convs] == [8, 12, 16]
    assert model.encoder.head.in_features == 16


def test_mlp_hidden_dims_sets_per_layer_widths():
    """An explicit hidden width list shapes the MLP encoder layer by layer."""
    cfg = _tiny_config("mlp_ae")
    cfg.hidden_dims = [16, 12]

    model = _forward_ok("mlp_ae", cfg)
    convs = [m for m in model.encoder.net if isinstance(m, torch.nn.Conv2d)]

    assert [c.out_channels for c in convs] == [16, 12, EMBEDDING_DIM]


def test_transformer1d_ffn_ratio_changes_feedforward_width():
    """Raising the transformer feed-forward ratio grows the parameter count."""
    narrow = _tiny_config("transformer1d_ae")
    wide   = _tiny_config("transformer1d_ae")
    wide.ffn_ratio = 4.0

    n_narrow = sum(p.numel() for p in get_profile_autoencoder("transformer1d_ae", narrow)[0].parameters())
    n_wide   = sum(p.numel() for p in get_profile_autoencoder("transformer1d_ae", wide)[0].parameters())

    assert n_wide > n_narrow


@pytest.mark.parametrize("mode", ["sinusoidal", "none"])
def test_cnn_attn_positional_modes_build_and_run(mode):
    """The sinusoidal and disabled positional modes run a finite round trip without learned tables."""
    cfg = _tiny_config("cnn_attn_ae")
    cfg.pos_embedding = mode

    model = _forward_ok("cnn_attn_ae", cfg)

    assert not any(name.endswith("pos_embed") for name, _ in model.named_parameters())
    assert not any(name.endswith("pos_embed") for name in model.state_dict())


def test_cnn_attn_unknown_positional_mode_raises():
    """An unknown positional embedding mode is rejected at construction."""
    cfg = _tiny_config("cnn_attn_ae")
    cfg.pos_embedding = "nope"

    with pytest.raises(ValueError):
        get_profile_autoencoder("cnn_attn_ae", cfg)
