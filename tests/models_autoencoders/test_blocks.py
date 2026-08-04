from __future__ import annotations

import pytest
import torch

from models.blocks import (
    ChannelLayerNorm,
    ConvBlock,
    ConvNeXtBlock,
    Decoder,
    DropPath,
    Encoder,
    MultiHeadSelfAttention,
    OutputHeadsMixin,
    PatchEmbedding,
    PixelMLP,
    ResidualConvBlock,
    TransformerBlock,
    build_activation,
    build_norm2d,
    build_upsample,
    downsample_stages,
    initialize_weights,
    match_spatial_size,
    scaled_dot_product,
    tokens_to_feature_map,
)


ACTIVATIONS    = ["relu", "leaky_relu", "gelu", "elu", "silu"]
NORMALIZATIONS = ["batch", "instance", "group", "none"]


def _finite(t):
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("factor,stages", [(1, 0), (2, 1), (4, 2), (8, 3)])
def test_downsample_stages(factor, stages):
    assert downsample_stages(factor) == stages


@pytest.mark.parametrize("bad", [0, 3, 6])
def test_downsample_stages_invalid(bad):
    with pytest.raises(ValueError):
        downsample_stages(bad)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_build_activation(name):
    act = build_activation(name)
    out = act(torch.randn(2, 4))
    assert out.shape == (2, 4)
    assert _finite(out)


def test_build_activation_invalid():
    with pytest.raises(ValueError):
        build_activation("nope")


@pytest.mark.parametrize("name", NORMALIZATIONS)
def test_build_norm2d(name):
    norm = build_norm2d(name, 8)
    out  = norm(torch.randn(2, 8, 6, 6))
    assert out.shape == (2, 8, 6, 6)
    assert _finite(out)


def test_build_norm2d_invalid():
    with pytest.raises(ValueError):
        build_norm2d("nope", 8)


@pytest.mark.parametrize("mode", ["convtranspose", "bilinear"])
def test_build_upsample(mode):
    up  = build_upsample(mode, 6, 3, scale_factor=2)
    out = up(torch.randn(2, 6, 4, 4))
    assert out.shape == (2, 3, 8, 8)
    assert _finite(out)


def test_build_upsample_invalid():
    with pytest.raises(ValueError):
        build_upsample("nope", 6, 3)


def test_drop_path_eval_identity():
    block = DropPath(0.5).eval()
    x     = torch.randn(4, 3, 5, 5)
    assert torch.equal(block(x), x)


def test_drop_path_zero_prob_identity():
    block = DropPath(0.0).train()
    x     = torch.randn(4, 3, 5, 5)
    assert torch.equal(block(x), x)


@pytest.mark.parametrize("mode", ["default", "kaiming", "xavier"])
def test_initialize_weights(mode):
    module = ConvBlock(3, 4)
    initialize_weights(module, mode)
    out = module(torch.randn(2, 3, 8, 8))
    assert _finite(out)


def test_match_spatial_size_resizes():
    source    = torch.randn(2, 3, 4, 4)
    reference = torch.randn(2, 5, 8, 8)
    out       = match_spatial_size(source, reference)
    assert out.shape[2:] == reference.shape[2:]
    assert _finite(out)


def test_match_spatial_size_passthrough():
    source    = torch.randn(2, 3, 8, 8)
    reference = torch.randn(2, 5, 8, 8)
    out       = match_spatial_size(source, reference)
    assert out is source


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_conv_block(activation, normalization):
    block = ConvBlock(3, 6, dropout=0.1, activation=activation, normalization=normalization)
    out   = block(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 6, 8, 8)
    assert _finite(out)


@pytest.mark.parametrize("stride,first_unit", [(1, False), (1, True), (2, False)])
def test_residual_conv_block(stride, first_unit):
    block    = ResidualConvBlock(4, 8, stride=stride, first_unit=first_unit)
    out      = block(torch.randn(2, 4, 8, 8))
    expected = 8 // stride
    assert out.shape == (2, 8, expected, expected)
    assert _finite(out)


def test_residual_conv_block_identity_shortcut():
    block = ResidualConvBlock(6, 6, first_unit=True)
    out   = block(torch.randn(2, 6, 8, 8))
    assert out.shape == (2, 6, 8, 8)
    assert _finite(out)


def test_channel_layer_norm():
    block = ChannelLayerNorm(8)
    out   = block(torch.randn(2, 8, 5, 5))
    assert out.shape == (2, 8, 5, 5)
    assert _finite(out)


@pytest.mark.parametrize("layer_scale_init", [0.0, 1e-6])
def test_convnext_block(layer_scale_init):
    block = ConvNeXtBlock(8, ffn_ratio=4.0, drop_path=0.0, ffn_activation="gelu", layer_scale_init=layer_scale_init)
    out   = block(torch.randn(2, 8, 6, 6))
    assert out.shape == (2, 8, 6, 6)
    assert _finite(out)


def test_pixel_mlp():
    block = PixelMLP(4, 16, 3)
    out   = block(torch.randn(2, 4, 7, 7))
    assert out.shape == (2, 3, 7, 7)
    assert _finite(out)


def test_encoder_decoder_roundtrip():
    feature_sizes = [4, 8, 16]
    encoder       = Encoder(3, feature_sizes)
    decoder       = Decoder(feature_sizes[::-1])

    x                 = torch.randn(2, 3, 16, 16)
    bottleneck, skips = encoder(x)
    assert bottleneck.shape == (2, 16, 2, 2)
    assert len(skips) == len(feature_sizes)
    assert _finite(bottleneck)

    out = decoder(bottleneck, skips[::-1][1:])
    assert _finite(out)


def test_scaled_dot_product():
    q   = torch.randn(2, 2, 5, 4)
    k   = torch.randn(2, 2, 5, 4)
    v   = torch.randn(2, 2, 5, 4)
    out = scaled_dot_product(q, k, v, scale=0.5, attention_dropout=torch.nn.Dropout(0.0))
    assert out.shape == (2, 2, 5, 4)
    assert _finite(out)


def test_multi_head_self_attention():
    block = MultiHeadSelfAttention(16, num_heads=4)
    out   = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
    assert _finite(out)


def test_multi_head_self_attention_invalid():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(16, num_heads=5)


@pytest.mark.parametrize("drop_path_rate", [0.0, 0.1])
def test_transformer_block(drop_path_rate):
    block = TransformerBlock(16, num_heads=4, drop_path_rate=drop_path_rate).eval()
    out   = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
    assert _finite(out)


def test_patch_embedding_and_back():
    block          = PatchEmbedding(3, 16, patch_size=4)
    tokens, gh, gw = block(torch.randn(2, 3, 16, 16))
    assert (gh, gw) == (4, 4)
    assert tokens.shape == (2, 16, 16)
    assert _finite(tokens)

    fmap = tokens_to_feature_map(tokens, gh, gw)
    assert fmap.shape == (2, 16, 4, 4)
    assert _finite(fmap)


def test_block_backward_finite_grads():
    block = ConvBlock(3, 6)
    x     = torch.randn(2, 3, 8, 8, requires_grad=True)
    block(x).pow(2).mean().backward()
    assert _finite(x.grad)


class _GaussianModule(OutputHeadsMixin):
    def __init__(self, head: str = "conv"):
        class _Cfg:
            params_per_gaussian = 3
            out_channels        = 9
            activation          = "relu"
        _Cfg.head               = head
        self.config             = _Cfg()
        self.embedding_channels = 8
        self.hidden_channels    = 16
        self._resolve_gaussian_layout()


def test_gaussian_heads_triple():
    module = _GaussianModule()
    module._build_triple_heads()
    out = module._triple_head_forward(torch.randn(2, 8, 5, 5))
    assert out.shape == (2, 9, 5, 5)
    assert _finite(out)


def test_gaussian_heads_per_gaussian():
    module = _GaussianModule()
    module._build_per_gaussian_heads()
    out = module._per_gaussian_forward(torch.randn(2, 8, 5, 5))
    assert out.shape == (2, 9, 5, 5)
    assert _finite(out)


@pytest.mark.parametrize("head", ["conv", "multihead", "per_gaussian", "set_pred"])
def test_output_head_dispatch(head):
    module = _GaussianModule(head=head)
    module._build_output_head()

    out = module._head_forward(torch.randn(2, 8, 5, 5))

    assert out.shape == (2, 9, 5, 5)
    assert _finite(out)
    assert len(module.head_parameters()) > 0


def test_output_head_unknown_head_raises():
    module = _GaussianModule(head="dense")
    with pytest.raises(ValueError):
        module._build_output_head()
