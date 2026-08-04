from __future__ import annotations

import pytest
import torch

from models.backbone import get_backbone


WINDOW = 32
BATCH  = 2

SETPRED_NAMES = ["unet", "resunet"]

OVERRIDES = {"head": "set_pred", "features": [8, 16], "bottleneck_factor": 1, "dropout": 0.0}


def _build(name):
    model, config = get_backbone(name, **OVERRIDES)
    return model.eval(), config


def _force_gate(model, logit_value: float) -> None:
    final_conv = model.existence_head.mlp[-1]
    torch.nn.init.zeros_(final_conv.weight)
    torch.nn.init.constant_(final_conv.bias, logit_value)


@pytest.mark.parametrize("name", SETPRED_NAMES)
def test_closed_gate_pins_amplitude_to_off_level(name):
    model, config = _build(name)
    _force_gate(model, logit_value=-30.0)

    with torch.no_grad():
        model.amp_off.copy_(torch.tensor([-1.5, 0.75]))
        out = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))

    ppg = config.params_per_gaussian
    amp = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)[:, :, 0]

    assert torch.allclose(amp[:, 0], torch.full_like(amp[:, 0], -1.5), atol=1e-5)
    assert torch.allclose(amp[:, 1], torch.full_like(amp[:, 1],  0.75), atol=1e-5)


@pytest.mark.parametrize("name", SETPRED_NAMES)
def test_open_gate_passes_raw_amplitude(name):
    model, config = _build(name)
    _force_gate(model, logit_value=30.0)

    x = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        out       = model(x)
        embedding = model.encode_decode(x)
        raw       = torch.stack([head(embedding) for head in model.gaussian_heads], dim=1)

    ppg = config.params_per_gaussian
    amp = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)[:, :, 0]

    assert torch.allclose(amp, raw[:, :, 0], atol=1e-5)


@pytest.mark.parametrize("name", SETPRED_NAMES)
def test_gate_leaves_mu_sigma_untouched(name):
    model, config = _build(name)
    _force_gate(model, logit_value=-30.0)

    x = torch.randn(BATCH, config.in_channels, WINDOW, WINDOW)

    with torch.no_grad():
        out       = model(x)
        embedding = model.encode_decode(x)
        raw       = torch.stack([head(embedding) for head in model.gaussian_heads], dim=1)

    ppg      = config.params_per_gaussian
    reshaped = out.reshape(BATCH, out.shape[1] // ppg, ppg, WINDOW, WINDOW)

    assert torch.allclose(reshaped[:, :, 1:], raw[:, :, 1:], atol=1e-5)


@pytest.mark.parametrize("name", SETPRED_NAMES)
def test_gate_gradients_reach_existence_head(name):
    model, config = _build(name)
    model.train()

    out  = model(torch.randn(BATCH, config.in_channels, WINDOW, WINDOW))
    loss = out.pow(2).mean()
    loss.backward()

    grads = [p.grad for p in model.existence_head.parameters()]

    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert model.amp_off.grad is not None
