from __future__ import annotations

import numpy as np
import pytest
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


def _input_window(interferograms, in_channels):
    raw   = np.asarray(interferograms[:in_channels, :WINDOW, :WINDOW])
    mag   = np.abs(raw).astype(np.float32)
    mean  = mag.mean(axis=(1, 2), keepdims=True)
    std   = mag.std(axis=(1, 2), keepdims=True) + 1e-6
    norm  = (mag - mean) / std
    batch = np.repeat(norm[None, ...], BATCH, axis=0)
    return torch.from_numpy(batch).to(DEVICE)


@pytest.fixture
def make_input(interferograms):
    def _factory(config):
        return _input_window(interferograms, config.in_channels)
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


@pytest.mark.parametrize("head", BACKBONE_HEADS)
@pytest.mark.parametrize("name", ALL_NAMES)
def test_param_groups_cover_every_parameter(name, head):
    model, config = _build(name, head=head)

    grouped = sum(len(group["params"]) for group in config.get_param_groups(model))
    total   = sum(1 for _ in model.parameters())

    assert grouped == total


@pytest.mark.parametrize("name", ALL_NAMES)
def test_unknown_head_raises(name):
    with pytest.raises(ValueError):
        _build(name, head="dense")


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


def test_local_cnn_rejects_mismatched_block_kernels():
    with pytest.raises(ValueError, match="exactly one kernel size"):
        get_backbone("local_cnn", features=[8, 8], block_kernels=[3])


def test_conv_block_rejects_even_kernels():
    from models.blocks import ConvBlock

    with pytest.raises(ValueError, match="positive odd"):
        ConvBlock(input_channels=4, output_channels=4, kernel_size=2)
