"""Tests covering every backbone in the model zoo and its output heads.

Each registered backbone is built at a tiny size and checked for shape-preserving
forward passes, finite values and gradients, deterministic evaluation, complete
parameter-group coverage, and architecture-specific validation rules.
"""

from __future__ import annotations

import pytest
import numpy as np
import torch

from models.backbone import BACKBONE_MODEL_REGISTRY, BACKBONE_CONFIG_REGISTRY, BACKBONE_HEADS, BACKBONE_IMAGE_SIZE_MODELS, get_backbone

from tests.models_backbone._helpers import SMALL_OVERRIDES, WINDOW


DEVICE = torch.device("cpu")
BATCH  = 2

ALL_NAMES      = sorted(BACKBONE_MODEL_REGISTRY.keys())
GAUSSIAN_HEADS = [head for head in BACKBONE_HEADS if head != "conv"]


def _build(name, head: str = "conv"):
    """Returns the named backbone with the given head, shrunk by the shared overrides and set to eval."""
    overrides     = dict(SMALL_OVERRIDES.get(name, {}))
    model, config = get_backbone(name, head=head, **overrides)
    model         = model.to(DEVICE).eval()
    return model, config


def _input_window(in_channels):
    """Returns a deterministic standardised random batch of shape (BATCH, in_channels, WINDOW, WINDOW)."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(BATCH, in_channels, WINDOW, WINDOW, generator=generator, dtype=torch.float32).to(DEVICE)


def _synthetic_window(in_channels):
    """Returns a seeded Gaussian input window of shape (BATCH, in_channels, WINDOW, WINDOW)."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(BATCH, in_channels, WINDOW, WINDOW, generator=generator).to(DEVICE)


@pytest.fixture
def make_input():
    """Yields a factory that builds a synthetic input window matching a backbone config."""
    def _factory(config):
        """Returns an input window sized for the given backbone config."""
        return _input_window(config.in_channels)
    return _factory


@pytest.mark.parametrize("name", ALL_NAMES)
def test_construct_and_param_count(name):
    """Verifies every registered backbone builds with at least one parameter."""
    model, _ = _build(name)
    total    = sum(p.numel() for p in model.parameters())
    assert total > 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_shape_preserved(name, make_input):
    """Verifies the forward pass keeps the patch size and emits the configured output channels."""
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert y.shape[0] == BATCH
    assert y.shape[1] == config.out_channels
    assert y.shape[2] == WINDOW
    assert y.shape[3] == WINDOW


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_finite(name, make_input):
    """Verifies the forward pass produces only finite values."""
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all()


@pytest.mark.dataset
def test_forward_on_a_scene_derived_window(derived):
    """Verifies the UNet backbone runs on a scene derived-channel window with the expected shape."""
    model, config = _build("unet")

    raw   = np.asarray(derived[:1, :WINDOW, :WINDOW])
    mag   = np.abs(raw).astype(np.float32)
    mean  = mag.mean(axis=(1, 2), keepdims=True)
    std   = mag.std(axis=(1, 2), keepdims=True) + 1e-6
    norm  = np.broadcast_to((mag - mean) / std, (config.in_channels, WINDOW, WINDOW))
    x     = torch.from_numpy(np.repeat(norm[None, ...], BATCH, axis=0).copy()).to(DEVICE)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (BATCH, config.out_channels, WINDOW, WINDOW)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_backward_finite_gradients(name, make_input):
    """Verifies the backward pass produces finite gradients on every trainable parameter."""
    model, config = _build(name)
    model.train()
    x             = make_input(config)

    y    = model(x)
    loss = y.float().pow(2).mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]

    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(loss).item()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_deterministic_eval(name, make_input):
    """Verifies two evaluation-mode forward passes on the same input are bit-identical."""
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        a = model(x)
        b = model(x)

    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_cpu_roundtrip(name):
    """Verifies moving a backbone to CPU leaves every parameter on CPU."""
    model, _ = _build(name)

    model.to("cpu")
    devices = {p.device.type for p in model.parameters()}

    assert devices == {"cpu"}


def test_registry_covers_all_models():
    """Verifies the model and config registries share keys and cover the image-size models."""
    assert set(BACKBONE_MODEL_REGISTRY.keys()) == set(BACKBONE_CONFIG_REGISTRY.keys())
    assert len(ALL_NAMES) > 0
    assert BACKBONE_IMAGE_SIZE_MODELS.issubset(set(ALL_NAMES))


@pytest.mark.parametrize("head", GAUSSIAN_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_gaussian_head_forward_shape(name, head, make_input):
    """Verifies each Gaussian head emits the configured output channels at the patch size."""
    model, config = _build(name, head=head)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (BATCH, config.out_channels, WINDOW, WINDOW)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("head", GAUSSIAN_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_equals_head_applied_to_encode_decode(name, head, make_input):
    """Verifies the forward pass equals the head applied to the trunk embedding."""
    model, config = _build(name, head=head)
    x             = make_input(config)

    with torch.no_grad():
        direct = model(x)
        staged = model._head_forward(model.encode_decode(x))

    assert staged.shape == direct.shape
    assert torch.allclose(direct, staged, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("head", BACKBONE_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_param_groups_cover_every_parameter(name, head):
    """Verifies the optimiser parameter groups cover every parameter exactly once for each head."""
    model, config = _build(name, head=head)

    grouped = sum(len(group["params"]) for group in config.get_param_groups(model))
    total   = sum(1 for _ in model.parameters())

    assert grouped == total


@pytest.mark.parametrize("name", ALL_NAMES)
def test_unknown_head_raises(name):
    """Verifies an unknown head name is rejected for every backbone."""
    with pytest.raises(ValueError):
        _build(name, head="dense")


def test_local_cnn_block_kernels_control_the_receptive_field():
    """Verifies a 3x3 then 1x1 local CNN cannot propagate a corner poke to the patch centre."""
    torch.manual_seed(0)
    model, _ = get_backbone("local_cnn", in_channels=3, out_channels=6, features=[8, 8], block_kernels=[3, 1])
    model    = model.eval()

    base  = torch.zeros(1, 3, 17, 17)
    poked = base.clone()
    poked[0, :, 0, 0] = 5.0

    with torch.no_grad():
        delta = (model(base) - model(poked)).abs()

    assert delta[0, :, 8, 8].max().item() == 0.0
    assert delta.max().item() > 0.0


def test_local_cnn_all_1x1_kernels_are_pixelwise():
    """Verifies an all-1x1 local CNN changes only the poked pixel."""
    torch.manual_seed(0)
    model, _ = get_backbone("local_cnn", in_channels=3, out_channels=6, features=[8, 8], block_kernels=[1, 1])
    model    = model.eval()

    base  = torch.zeros(1, 3, 5, 5)
    poked = base.clone()
    poked[0, :, 1, 1] = 5.0

    with torch.no_grad():
        delta = (model(base) - model(poked)).abs().amax(dim=1)

    assert delta[0, 1, 1].item() > 0.0
    delta[0, 1, 1] = 0.0
    assert delta.max().item() == 0.0


def test_local_cnn_rejects_mismatched_block_kernels():
    """Verifies the local CNN refuses a kernel list shorter than its feature ladder."""
    with pytest.raises(ValueError, match="exactly one kernel size"):
        get_backbone("local_cnn", features=[8, 8], block_kernels=[3])


def test_convnext_unet_drop_path_rates_follow_the_stage_schedule():
    """Verifies the ConvNeXt UNet assigns drop-path rates stage by stage from the shared schedule."""
    from models.blocks import DropPathSchedule

    model, _ = get_backbone("convnext_unet", features=[8, 16], blocks_per_stage=2, bottleneck_factor=1, stochastic_depth_rate=0.4)
    schedule = DropPathSchedule(total_blocks=2 * (2 * 2 + 1), rate=0.4, blocks_per_stage=2)

    stages = list(model.encoder_stages) + [model.bottleneck] + list(model.decoder_stages)

    for index, stage in enumerate(stages):
        assert [block.drop_path.drop_prob for block in stage.blocks] == pytest.approx(schedule.stage(index))


def test_swin_block_reuses_the_shifted_attention_mask_per_spatial_size():
    """Verifies the Swin block caches its shifted-window mask per spatial size and stays deterministic."""
    from models.backbone.swin_unet import SwinTransformerBlock

    block = SwinTransformerBlock(embedding_dim=8, num_heads=2, window_size=4, shift_size=2).eval()
    x     = torch.randn(2, 8 * 8, 8)

    with torch.no_grad():
        first  = block(x, 8, 8)
        mask   = block._attention_mask(8, 8, x.device)
        second = block(x, 8, 8)

    assert block._attention_mask(8, 8, x.device) is mask
    assert block._attention_mask(12, 12, x.device) is not mask
    assert torch.allclose(first, second, atol=0.0, rtol=0.0)


def test_transunet_rejects_an_image_too_small_for_its_cnn_stages():
    """Verifies TransUNet refuses an image that the CNN downsampling stages reduce below one pixel."""
    with pytest.raises(ValueError, match="is too small for 4 CNN downsampling stages"):
        get_backbone("transunet", image_size=8, patch_size=1, cnn_features=[16, 32, 64, 128])


def test_transunet_rejects_a_patch_larger_than_the_bottleneck():
    """Verifies TransUNet refuses a patch size larger than the CNN bottleneck feature map."""
    with pytest.raises(ValueError, match="cannot exceed bottleneck feature size"):
        get_backbone("transunet", image_size=64, patch_size=16, cnn_features=[16, 32, 64])


@pytest.mark.parametrize("heights", [(4, 3), (4, 3, 2, 2)])
def test_u2net_rejects_rsu_heights_that_do_not_match_its_stages(heights):
    """Verifies U2-Net refuses an RSU height list that does not have three entries."""
    with pytest.raises(ValueError, match="rsu_heights must contain exactly three"):
        get_backbone("u2net", features=[16, 32, 64, 128], rsu_heights=heights)


def test_multihead_rejects_four_parameters_per_gaussian():
    """Verifies the multihead output head refuses anything but three parameters per Gaussian."""
    with pytest.raises(ValueError, match="params_per_gaussian must be 3"):
        get_backbone("unet", head="multihead", params_per_gaussian=4, out_channels=8, features=[8, 16], bottleneck_factor=1)


def test_unetr_rejects_patch_size_below_the_decoder_ladder():
    """Verifies UNETR refuses a patch size smaller than its decoder upsampling ladder demands."""
    with pytest.raises(ValueError, match="multiple of 2 \\*\\* len\\(decoder_features\\)"):
        get_backbone("unetr", image_size=WINDOW, patch_size=8, decoder_features=[32, 16, 8, 8])


def test_unetr_rejects_patch_size_indivisible_by_the_decoder_ladder():
    """Verifies UNETR refuses a patch size not divisible by its decoder upsampling ladder."""
    with pytest.raises(ValueError, match="multiple of 2 \\*\\* len\\(decoder_features\\)"):
        get_backbone("unetr", image_size=48, patch_size=24, decoder_features=[32, 16, 8, 8])


def test_unetr_rejects_non_positive_patch_size():
    """Verifies UNETR rejects non-positive patch sizes."""
    with pytest.raises(ValueError, match="patch_size must be a positive integer"):
        get_backbone("unetr", image_size=64, patch_size=0, embedding_dim=64, transformer_layers=4, transformer_heads=4, decoder_features=[32, 16, 8, 8])


def test_unetr_rejects_patch_size_not_dividing_image_size():
    """Verifies UNETR rejects patch sizes that do not evenly divide the image size."""
    with pytest.raises(ValueError, match="image_size must be divisible by patch_size"):
        get_backbone("unetr", image_size=64, patch_size=7, embedding_dim=64, transformer_layers=4, transformer_heads=4, decoder_features=[32, 16, 8, 8])


def test_conv_block_rejects_even_kernels():
    """Verifies the convolution block refuses an even kernel size."""
    from models.blocks import ConvBlock

    with pytest.raises(ValueError, match="positive odd"):
        ConvBlock(input_channels=4, output_channels=4, kernel_size=2)
