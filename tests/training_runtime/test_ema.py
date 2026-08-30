"""Tests for the exponential moving average of model weights.

Covers shadow updates, the temporary parameter swap context manager and
state dict round trips of WeightEma.
"""

from __future__ import annotations

import pytest
import torch

from tools.training.checkpoint import WeightEma


def _model(fill: float = 1.0) -> torch.nn.Module:
    """Returns a bias-free 2x2 linear model whose weight is filled with `fill`."""
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(fill)
    return model


def test_update_moves_shadow_toward_params():
    """Verifies that shadow weights move toward the live parameters at the decay rate."""
    model = _model(fill=0.0)
    ema   = WeightEma(model, decay=0.5, enabled=True, warmup=False)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)

    assert torch.allclose(ema.shadow["weight"], torch.full((2, 2), 0.5))

    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], torch.full((2, 2), 0.75))


def test_disabled_ema_keeps_no_shadow_and_update_is_noop():
    """Verifies that a disabled EMA keeps an empty shadow and ignores updates."""
    model = _model()
    ema   = WeightEma(model, decay=0.5, enabled=False, warmup=False)

    ema.update(model)

    assert ema.shadow == {}


def test_applied_swaps_shadow_in_and_restores_after():
    """Verifies that the context manager installs shadow weights and restores originals."""
    model = _model(fill=0.0)
    ema   = WeightEma(model, decay=0.0, enabled=True, warmup=False)

    with torch.no_grad():
        model.weight.fill_(3.0)

    with ema.applied(model):
        assert torch.allclose(model.weight, torch.zeros(2, 2))

    assert torch.allclose(model.weight, torch.full((2, 2), 3.0))


def test_applied_restores_params_when_body_raises():
    """Verifies that live parameters are restored when the context body raises."""
    model = _model(fill=2.0)
    ema   = WeightEma(model, decay=0.0, enabled=True, warmup=False)

    with torch.no_grad():
        model.weight.fill_(5.0)

    with pytest.raises(RuntimeError):
        with ema.applied(model):
            raise RuntimeError("boom")

    assert torch.allclose(model.weight, torch.full((2, 2), 5.0))


def test_disabled_applied_is_noop():
    """Verifies that a disabled EMA leaves parameters untouched inside the context."""
    model = _model(fill=4.0)
    ema   = WeightEma(model, decay=0.5, enabled=False, warmup=False)

    with ema.applied(model):
        assert torch.allclose(model.weight, torch.full((2, 2), 4.0))


def test_state_roundtrip():
    """Verifies that shadow weights and the update count survive a state dict save and load."""
    model = _model(fill=1.0)
    ema   = WeightEma(model, decay=0.5, enabled=True, warmup=False)

    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)

    other = WeightEma(model, decay=0.5, enabled=True, warmup=False)
    other.load_state_dict(ema.state_dict())

    assert torch.allclose(other.shadow["weight"], ema.shadow["weight"])
    assert other.num_updates == 1


def test_decay_warmup_follows_classic_schedule():
    """Verifies that the effective decay ramps as min(decay, (1+t)/(10+t)) over the updates."""
    model = _model(fill=0.0)
    ema   = WeightEma(model, decay=0.999, enabled=True, warmup=True)

    assert ema.effective_decay() == pytest.approx(0.1)

    ema.update(model)
    assert ema.num_updates      == 1
    assert ema.last_decay       == pytest.approx(0.1)
    assert ema.effective_decay() == pytest.approx(2.0 / 11.0)

    ema.num_updates = 100_000
    assert ema.effective_decay() == pytest.approx(0.999)


def test_decay_warmup_first_update_blend():
    """Verifies that the first warmup update blends 90 percent of the live weights into the shadow."""
    model = _model(fill=0.0)
    ema   = WeightEma(model, decay=0.999, enabled=True, warmup=True)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)

    assert torch.allclose(ema.shadow["weight"], torch.full((2, 2), 0.9))


def test_decay_warmup_off_uses_target_decay():
    """Verifies that the target decay applies from the first update when warmup is off."""
    model = _model(fill=0.0)
    ema   = WeightEma(model, decay=0.999, enabled=True, warmup=False)

    assert ema.effective_decay() == pytest.approx(0.999)


def test_state_enabled_mismatch_raises():
    """Verifies that loading a state with a different enabled flag raises ValueError."""
    model    = _model()
    enabled  = WeightEma(model, decay=0.5, enabled=True, warmup=False)
    disabled = WeightEma(model, decay=0.5, enabled=False, warmup=False)

    with pytest.raises(ValueError):
        disabled.load_state_dict(enabled.state_dict())

    with pytest.raises(ValueError):
        enabled.load_state_dict(disabled.state_dict())
