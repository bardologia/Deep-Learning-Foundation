"""Tests for the stochastic weight average: math, application and BN refresh."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tools.training.checkpoint import WeightSwa


class _BnNet(nn.Module):
    """One conv with batch norm, exercising both parameters and running buffers."""

    def __init__(self):
        """Builds the conv-BN pair."""
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.bn   = nn.BatchNorm2d(2)

    def forward(self, x):
        """Returns the normalised convolution output."""
        return self.bn(self.conv(x))


def test_update_averages_snapshots_equally():
    """Two snapshots average to their mean, regardless of order."""
    model = nn.Linear(2, 2, bias=False)
    swa   = WeightSwa(model, enabled=True)

    with torch.no_grad():
        model.weight.fill_(1.0)
    swa.update(model)

    with torch.no_grad():
        model.weight.fill_(3.0)
    swa.update(model)

    assert swa.n_averaged == 2
    assert torch.allclose(swa.shadow["weight"], torch.full((2, 2), 2.0))


def test_applied_swaps_weights_and_restores_buffers():
    """Applying the average swaps parameters in and restores parameters and buffers on exit."""
    model = _BnNet()
    swa   = WeightSwa(model, enabled=True)
    swa.update(model)

    live_weight = model.conv.weight.detach().clone()
    live_mean   = model.bn.running_mean.detach().clone()

    with swa.applied(model):
        model.train()
        model(torch.randn(8, 1, 6, 6))
        assert not torch.allclose(model.bn.running_mean, live_mean)

    assert torch.allclose(model.conv.weight, live_weight)
    assert torch.allclose(model.bn.running_mean, live_mean)


def test_applied_without_updates_fails_loudly():
    """Applying an empty average raises instead of silently using zero weights."""
    model = nn.Linear(2, 2)
    swa   = WeightSwa(model, enabled=True)

    with pytest.raises(ValueError, match="no snapshot"):
        with swa.applied(model):
            pass


def test_refresh_bn_reestimates_running_stats():
    """One refresh pass rewrites the BN running statistics from the loader batches."""
    torch.manual_seed(0)
    model = _BnNet()

    batches = [(torch.randn(16, 1, 6, 6) + 5.0,) for _ in range(4)]
    before  = model.bn.running_mean.detach().clone()

    refreshed = WeightSwa.refresh_bn(model, batches, lambda batch: model(batch[0]))

    assert refreshed == 1
    assert not torch.allclose(model.bn.running_mean, before)
    assert model.bn.momentum is not None


def test_refresh_bn_skips_models_without_batchnorm():
    """A model with no BN modules reports zero refreshed and consumes no batches."""
    consumed = []

    refreshed = WeightSwa.refresh_bn(nn.Linear(2, 2), consumed, lambda batch: None)

    assert refreshed == 0


def test_state_round_trip():
    """The averaged weights and count survive a state-dict round trip."""
    model = nn.Linear(2, 2, bias=False)
    swa   = WeightSwa(model, enabled=True)
    swa.update(model)

    restored = WeightSwa(model, enabled=True)
    restored.load_state_dict(swa.state_dict())

    assert restored.n_averaged == 1
    assert torch.allclose(restored.shadow["weight"], swa.shadow["weight"])


def test_state_mismatch_fails_loudly():
    """Restoring an enabled state into a disabled average is rejected."""
    model = nn.Linear(2, 2)
    swa   = WeightSwa(model, enabled=True)
    swa.update(model)

    disabled = WeightSwa(model, enabled=False)

    with pytest.raises(ValueError, match="use_swa"):
        disabled.load_state_dict(swa.state_dict())
