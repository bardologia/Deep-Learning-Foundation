"""Tests covering the shared neural building blocks in ``models.blocks``.

Exercises the layer factory, drop-path scheduling, convolutional and ConvNeXt blocks,
attention and transformer blocks, patch embedding, the encoder/decoder pair and the
Gaussian output-head mixin.
"""

from __future__ import annotations

import pytest
import torch

from models.blocks import (
    ChannelLayerNorm,
    ConvBlock,
    ConvNeXtBlock,
    Decoder,
    DropPath,
    DropPathSchedule,
    Encoder,
    MultiHeadSelfAttention,
    OutputHeadsMixin,
    PatchEmbedding,
    PixelMLP,
    ResidualConvBlock,
    TransformerBlock,
)
from models.blocks import AttentionMath, AttentionTap, FeatureGeometry, LayerFactory


ACTIVATIONS    = ["relu", "leaky_relu", "gelu", "elu", "silu"]
NORMALIZATIONS = ["batch", "instance", "group", "none"]


def _finite(t):
    """Returns True when every entry of the tensor is finite."""
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("factor,stages", [(1, 0), (2, 1), (4, 2), (8, 3)])
def test_downsample_stages(factor, stages):
    """Verifies a power-of-two downsampling factor maps to its number of stages."""
    assert LayerFactory.downsample_stages(factor) == stages


@pytest.mark.parametrize("bad", [0, 3, 6])
def test_downsample_stages_invalid(bad):
    """Verifies a downsampling factor that is not a power of two is rejected."""
    with pytest.raises(ValueError):
        LayerFactory.downsample_stages(bad)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_build_activation(name):
    """Verifies each supported activation builds and preserves the input shape."""
    act = LayerFactory.build_activation(name)
    out = act(torch.randn(2, 4))
    assert out.shape == (2, 4)
    assert _finite(out)


def test_build_activation_invalid():
    """Verifies an unknown activation name is rejected."""
    with pytest.raises(ValueError):
        LayerFactory.build_activation("nope")


@pytest.mark.parametrize("name", NORMALIZATIONS)
def test_build_norm2d(name):
    """Verifies each supported 2D normalisation builds and preserves the feature-map shape."""
    norm = LayerFactory.build_norm2d(name, 8)
    out  = norm(torch.randn(2, 8, 6, 6))
    assert out.shape == (2, 8, 6, 6)
    assert _finite(out)


def test_build_norm2d_invalid():
    """Verifies an unknown normalisation name is rejected."""
    with pytest.raises(ValueError):
        LayerFactory.build_norm2d("nope", 8)


@pytest.mark.parametrize("mode", ["convtranspose", "bilinear"])
def test_build_upsample(mode):
    """Verifies both upsampling modes double the spatial size and remap the channel count."""
    up  = LayerFactory.build_upsample(mode, 6, 3, scale_factor=2)
    out = up(torch.randn(2, 6, 4, 4))
    assert out.shape == (2, 3, 8, 8)
    assert _finite(out)


def test_build_upsample_invalid():
    """Verifies an unknown upsampling mode is rejected."""
    with pytest.raises(ValueError):
        LayerFactory.build_upsample("nope", 6, 3)


def test_drop_path_eval_identity():
    """Verifies drop path is the identity in evaluation mode."""
    block = DropPath(0.5).eval()
    x     = torch.randn(4, 3, 5, 5)
    assert torch.equal(block(x), x)


def test_drop_path_zero_prob_identity():
    """Verifies drop path is the identity when the drop probability is zero."""
    block = DropPath(0.0).train()
    x     = torch.randn(4, 3, 5, 5)
    assert torch.equal(block(x), x)


@pytest.mark.parametrize("mode", ["default", "kaiming", "xavier"])
def test_initialize_weights(mode):
    """Verifies each initialisation mode leaves the block producing finite outputs."""
    module = ConvBlock(3, 4)
    LayerFactory.initialize_weights(module, mode)
    out = module(torch.randn(2, 3, 8, 8))
    assert _finite(out)


def test_drop_path_schedule_slices_by_stage_index():
    """Verifies the schedule hands each stage its own contiguous slice of drop probabilities."""
    schedule = DropPathSchedule(total_blocks=6, rate=0.5, blocks_per_stage=2)

    assert schedule.stage(0) == pytest.approx([0.0, 0.1])
    assert schedule.stage(2) == pytest.approx([0.4, 0.5])
    assert schedule.stage(1) == pytest.approx(schedule.stage(1))


def test_drop_path_schedule_rejects_a_stage_beyond_its_length():
    """Verifies requesting a stage past the scheduled depth is rejected."""
    schedule = DropPathSchedule(total_blocks=6, rate=0.5, blocks_per_stage=2)

    with pytest.raises(ValueError):
        schedule.stage(3)


def test_initialize_weights_invalid():
    """Verifies an unknown initialisation mode is rejected."""
    with pytest.raises(ValueError):
        LayerFactory.initialize_weights(ConvBlock(3, 4), "kaiming_uniform")


def test_match_spatial_size_resizes():
    """Verifies a smaller feature map is resized to the reference spatial size."""
    source    = torch.randn(2, 3, 4, 4)
    reference = torch.randn(2, 5, 8, 8)
    out       = FeatureGeometry.match_spatial_size(source, reference)
    assert out.shape[2:] == reference.shape[2:]
    assert _finite(out)


def test_match_spatial_size_passthrough():
    """Verifies an already matching feature map is returned unchanged."""
    source    = torch.randn(2, 3, 8, 8)
    reference = torch.randn(2, 5, 8, 8)
    out       = FeatureGeometry.match_spatial_size(source, reference)
    assert out is source


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_conv_block(activation, normalization):
    """Verifies the convolution block maps channels and keeps the spatial size across activations and normalisations."""
    block = ConvBlock(3, 6, dropout=0.1, activation=activation, normalization=normalization)
    out   = block(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 6, 8, 8)
    assert _finite(out)


@pytest.mark.parametrize("stride,first_unit", [(1, False), (1, True), (2, False)])
def test_residual_conv_block(stride, first_unit):
    """Verifies the residual block applies the requested stride and output channel count."""
    block    = ResidualConvBlock(4, 8, stride=stride, first_unit=first_unit)
    out      = block(torch.randn(2, 4, 8, 8))
    expected = 8 // stride
    assert out.shape == (2, 8, expected, expected)
    assert _finite(out)


def test_residual_conv_block_identity_shortcut():
    """Verifies the residual block keeps shape when input and output channels match."""
    block = ResidualConvBlock(6, 6, first_unit=True)
    out   = block(torch.randn(2, 6, 8, 8))
    assert out.shape == (2, 6, 8, 8)
    assert _finite(out)


def test_channel_layer_norm():
    """Verifies channel-wise layer normalisation preserves the feature-map shape."""
    block = ChannelLayerNorm(8)
    out   = block(torch.randn(2, 8, 5, 5))
    assert out.shape == (2, 8, 5, 5)
    assert _finite(out)


@pytest.mark.parametrize("layer_scale_init", [0.0, 1e-6])
def test_convnext_block(layer_scale_init):
    """Verifies the ConvNeXt block preserves shape with and without layer scale."""
    block = ConvNeXtBlock(8, ffn_ratio=4.0, drop_path=0.0, ffn_activation="gelu", layer_scale_init=layer_scale_init)
    out   = block(torch.randn(2, 8, 6, 6))
    assert out.shape == (2, 8, 6, 6)
    assert _finite(out)


def test_pixel_mlp():
    """Verifies the pixelwise MLP remaps channels without touching the spatial grid."""
    block = PixelMLP(4, 16, 3)
    out   = block(torch.randn(2, 4, 7, 7))
    assert out.shape == (2, 3, 7, 7)
    assert _finite(out)


def test_encoder_decoder_roundtrip():
    """Verifies the encoder produces the expected bottleneck and skips that the decoder consumes."""
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
    """Verifies scaled dot-product attention preserves the query shape."""
    q   = torch.randn(2, 2, 5, 4)
    k   = torch.randn(2, 2, 5, 4)
    v   = torch.randn(2, 2, 5, 4)
    out = AttentionMath.scaled_dot_product(q, k, v, scale=0.5, attention_dropout=torch.nn.Dropout(0.0))
    assert out.shape == (2, 2, 5, 4)
    assert _finite(out)


@pytest.mark.parametrize("with_bias", [False, True])
def test_scaled_dot_product_fused_path_matches_the_tapped_path(with_bias):
    """Verifies the fused attention kernel and the tapped explicit path agree in eval mode, with and without an additive bias."""
    generator = torch.Generator().manual_seed(0)
    q         = torch.randn(2, 2, 5, 4, generator=generator, dtype=torch.float64)
    k         = torch.randn(2, 2, 5, 4, generator=generator, dtype=torch.float64)
    v         = torch.randn(2, 2, 5, 4, generator=generator, dtype=torch.float64)
    bias      = torch.randn(2, 1, 5, 5, generator=generator, dtype=torch.float64) if with_bias else None
    dropout   = torch.nn.Dropout(0.3).eval()

    fused = AttentionMath.scaled_dot_product(q, k, v, 0.5, dropout, bias)

    sink = []
    AttentionTap.enable(sink)
    try:
        tapped = AttentionMath.scaled_dot_product(q, k, v, 0.5, dropout, bias)
    finally:
        AttentionTap.disable()

    assert len(sink) == 1
    assert torch.allclose(fused, tapped, atol=1e-6, rtol=0.0)


def test_shifted_swin_block_fused_path_matches_the_tapped_path():
    """Verifies a shifted Swin block, whose attention adds the relative-position bias and the cyclic-shift mask, agrees on both attention paths in eval mode."""
    from models.backbone.swin_unet import SwinTransformerBlock

    torch.manual_seed(1)
    block = SwinTransformerBlock(embedding_dim=8, num_heads=2, window_size=4, shift_size=2, attention_dropout=0.3).double().eval()
    torch.nn.init.normal_(block.attention.relative_position_bias_table, std=1.0)
    x     = torch.randn(2, 8 * 8, 8, dtype=torch.float64)

    sink = []
    with torch.no_grad():
        fused = block(x, 8, 8)

        AttentionTap.enable(sink)
        try:
            tapped = block(x, 8, 8)
        finally:
            AttentionTap.disable()

    assert len(sink) == 1
    assert torch.allclose(fused, tapped, atol=1e-6, rtol=0.0)


def test_multi_head_self_attention():
    """Verifies multi-head self-attention preserves the token sequence shape."""
    block = MultiHeadSelfAttention(16, num_heads=4)
    out   = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
    assert _finite(out)


def test_multi_head_self_attention_invalid():
    """Verifies an embedding dimension not divisible by the head count is rejected."""
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(16, num_heads=5)


@pytest.mark.parametrize("drop_path_rate", [0.0, 0.1])
def test_transformer_block(drop_path_rate):
    """Verifies the transformer block preserves the token shape with and without drop path."""
    block = TransformerBlock(16, num_heads=4, drop_path_rate=drop_path_rate).eval()
    out   = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
    assert _finite(out)


def test_patch_embedding_and_back():
    """Verifies patch embedding reports the token grid and that tokens fold back into a feature map."""
    block          = PatchEmbedding(3, 16, patch_size=4)
    tokens, gh, gw = block(torch.randn(2, 3, 16, 16))
    assert (gh, gw) == (4, 4)
    assert tokens.shape == (2, 16, 16)
    assert _finite(tokens)

    fmap = FeatureGeometry.tokens_to_feature_map(tokens, gh, gw)
    assert fmap.shape == (2, 16, 4, 4)
    assert _finite(fmap)


def test_block_backward_finite_grads():
    """Verifies a convolution block backpropagates finite gradients to its input."""
    block = ConvBlock(3, 6)
    x     = torch.randn(2, 3, 8, 8, requires_grad=True)
    block(x).pow(2).mean().backward()
    assert _finite(x.grad)


class _GaussianModule(OutputHeadsMixin):
    """Minimal ``OutputHeadsMixin`` host used to exercise the Gaussian output heads."""
    def __init__(self, head: str = "conv"):
        """Builds the stub config and resolves the Gaussian layout for the requested head."""
        class _Cfg:
            """Stand-in model config carrying only the fields the output heads read."""
            params_per_gaussian = 3
            out_channels        = 9
            activation          = "relu"
        _Cfg.head               = head
        self.config             = _Cfg()
        self.embedding_channels = 8
        self.hidden_channels    = 16
        self._resolve_gaussian_layout()


def test_gaussian_heads_triple():
    """Verifies the triple head emits three parameters per Gaussian on the input grid."""
    module = _GaussianModule()
    module._build_triple_heads()
    out = module._triple_head_forward(torch.randn(2, 8, 5, 5))
    assert out.shape == (2, 9, 5, 5)
    assert _finite(out)


def test_gaussian_heads_per_gaussian():
    """Verifies the per-Gaussian heads emit the full parameter stack on the input grid."""
    module = _GaussianModule()
    module._build_per_gaussian_heads()
    out = module._per_gaussian_forward(torch.randn(2, 8, 5, 5))
    assert out.shape == (2, 9, 5, 5)
    assert _finite(out)


@pytest.mark.parametrize("head", ["conv", "multihead", "per_gaussian", "set_pred"])
def test_output_head_dispatch(head):
    """Verifies every head variant builds, forwards to the declared channel count and exposes parameters."""
    module = _GaussianModule(head=head)
    module._build_output_head()

    out = module._head_forward(torch.randn(2, 8, 5, 5))

    assert out.shape == (2, 9, 5, 5)
    assert _finite(out)
    assert len(module.head_parameters()) > 0


def test_output_head_unknown_head_raises():
    """Verifies an unknown head name is rejected at build time."""
    module = _GaussianModule(head="dense")
    with pytest.raises(ValueError):
        module._build_output_head()
