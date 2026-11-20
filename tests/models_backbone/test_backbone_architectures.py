
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
    overrides     = dict(SMALL_OVERRIDES.get(name, {}))
    model, config = get_backbone(name, head=head, **overrides)
    model         = model.to(DEVICE).eval()
    return model, config


def _input_window(in_channels):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(BATCH, in_channels, WINDOW, WINDOW, generator=generator, dtype=torch.float32).to(DEVICE)


def _synthetic_window(in_channels):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(BATCH, in_channels, WINDOW, WINDOW, generator=generator).to(DEVICE)


@pytest.fixture
def make_input():
    def _factory(config):
        return _input_window(config.in_channels)
    return _factory


@pytest.mark.parametrize("name", ALL_NAMES)
def test_construct_and_param_count(name):
    model, _ = _build(name)
    total    = sum(p.numel() for p in model.parameters())
    assert total > 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_shape_preserved(name, make_input):
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
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert torch.isfinite(y).all()


@pytest.mark.dataset
def test_forward_on_a_scene_window(inputs):
    model, config = _build("unet")

    window = np.asarray(inputs[:1, :WINDOW, :WINDOW], dtype=np.float32)
    norm   = np.broadcast_to(window, (config.in_channels, WINDOW, WINDOW))
    x      = torch.from_numpy(np.repeat(norm[None, ...], BATCH, axis=0).copy()).to(DEVICE)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (BATCH, config.out_channels, WINDOW, WINDOW)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_backward_finite_gradients(name, make_input):
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
    model, config = _build(name)
    x             = make_input(config)

    with torch.no_grad():
        a = model(x)
        b = model(x)

    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_cpu_roundtrip(name):
    model, _ = _build(name)

    model.to("cpu")
    devices = {p.device.type for p in model.parameters()}

    assert devices == {"cpu"}


def test_registry_covers_all_models():
    assert set(BACKBONE_MODEL_REGISTRY.keys()) == set(BACKBONE_CONFIG_REGISTRY.keys())
    assert len(ALL_NAMES) > 0
    assert BACKBONE_IMAGE_SIZE_MODELS.issubset(set(ALL_NAMES))


@pytest.mark.parametrize("head", GAUSSIAN_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_gaussian_head_forward_shape(name, head, make_input):
    model, config = _build(name, head=head)
    x             = make_input(config)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (BATCH, config.out_channels, WINDOW, WINDOW)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("head", GAUSSIAN_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_equals_head_applied_to_encode_decode(name, head, make_input):
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
    model, config = _build(name, head=head)

    grouped = sum(len(group["params"]) for group in config.get_param_groups(model))
    total   = sum(1 for _ in model.parameters())

    assert grouped == total


def test_local_cnn_block_kernels_control_the_receptive_field():
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


def test_convnext_unet_drop_path_rates_follow_the_stage_schedule():
    from models.blocks import DropPathSchedule

    model, _ = get_backbone("convnext_unet", features=[8, 16], blocks_per_stage=2, bottleneck_factor=1, stochastic_depth_rate=0.4)
    schedule = DropPathSchedule(total_blocks=2 * (2 * 2 + 1), rate=0.4, blocks_per_stage=2)

    stages = list(model.encoder_stages) + [model.bottleneck] + list(model.decoder_stages)

    for index, stage in enumerate(stages):
        assert [block.drop_path.drop_prob for block in stage.blocks] == pytest.approx(schedule.stage(index))


def test_swin_block_reuses_the_shifted_attention_mask_per_spatial_size():
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


