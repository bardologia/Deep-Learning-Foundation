from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from pipelines.jepa.training.coupling import CouplingMode, TargetProvider


def make_encoder():
    return nn.Sequential(nn.Conv2d(2, 2, kernel_size=1))


def test_coupling_mode_rejects_unknown_kind():
    with pytest.raises(ValueError):
        CouplingMode("scratch", "profile autoencoder")


def test_coupling_mode_frozen_disables_grad_and_eval():
    mode    = CouplingMode("frozen", "profile autoencoder")
    encoder = make_encoder()
    mode.apply(encoder)

    assert mode.trainable is False
    assert all(not p.requires_grad for p in encoder.parameters())
    assert encoder.training is False


def test_coupling_mode_finetune_enables_grad():
    mode    = CouplingMode("finetune", "profile autoencoder")
    encoder = make_encoder()
    mode.apply(encoder)

    assert mode.trainable is True
    assert all(p.requires_grad for p in encoder.parameters())


def test_coupling_mode_param_groups_empty_when_frozen():
    mode    = CouplingMode("frozen", "profile autoencoder")
    encoder = make_encoder()
    mode.apply(encoder)

    assert mode.param_groups(encoder, lr=1e-3, wd=1e-4) == []


def test_coupling_mode_param_groups_carry_lr_and_wd_when_finetune():
    mode    = CouplingMode("finetune", "profile autoencoder")
    encoder = make_encoder()
    mode.apply(encoder)

    groups = mode.param_groups(encoder, lr=7e-4, wd=2e-5)

    assert len(groups)               == 1
    assert groups[0]["lr"]           == 7e-4
    assert groups[0]["weight_decay"] == 2e-5
    assert groups[0]["name"]         == "profile autoencoder"
    assert len(groups[0]["params"]) > 0


def test_target_provider_live_is_differentiable():
    encoder  = make_encoder()
    provider = TargetProvider("live")
    curve    = torch.randn(1, 2, 3, 3, requires_grad=True)

    out = provider.target(encoder, curve)

    assert out.requires_grad is True


def test_target_provider_stopgrad_detaches_target():
    encoder  = make_encoder()
    provider = TargetProvider("stopgrad")
    curve    = torch.randn(1, 2, 3, 3, requires_grad=True)

    out = provider.target(encoder, curve)

    assert out.requires_grad is False


def test_target_provider_rejects_unknown_kind():
    with pytest.raises(ValueError):
        TargetProvider("ema")

    with pytest.raises(ValueError):
        TargetProvider("stopgradd")


def test_target_provider_does_not_mutate_online_encoder():
    encoder  = make_encoder()
    provider = TargetProvider("stopgrad")
    snapshot = copy.deepcopy(encoder.state_dict())

    provider.target(encoder, torch.randn(1, 2, 3, 3))

    for key, value in encoder.state_dict().items():
        assert torch.allclose(value, snapshot[key])
