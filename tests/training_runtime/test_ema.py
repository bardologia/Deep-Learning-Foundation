from __future__ import annotations

import pytest
import torch

from tools.training.checkpoint import WeightEma


def _model(fill: float = 1.0) -> torch.nn.Module:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(fill)
    return model


def test_update_moves_shadow_toward_params():
    model = _model(fill=0.0)
    ema   = WeightEma(model, decay=0.5, enabled=True)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)

    assert torch.allclose(ema.shadow["weight"], torch.full((2, 2), 0.5))

    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], torch.full((2, 2), 0.75))


def test_disabled_ema_keeps_no_shadow_and_update_is_noop():
    model = _model()
    ema   = WeightEma(model, decay=0.5, enabled=False)

    ema.update(model)

    assert ema.shadow == {}


def test_applied_swaps_shadow_in_and_restores_after():
    model = _model(fill=0.0)
    ema   = WeightEma(model, decay=0.0, enabled=True)

    with torch.no_grad():
        model.weight.fill_(3.0)

    with ema.applied(model):
        assert torch.allclose(model.weight, torch.zeros(2, 2))

    assert torch.allclose(model.weight, torch.full((2, 2), 3.0))


def test_applied_restores_params_when_body_raises():
    model = _model(fill=2.0)
    ema   = WeightEma(model, decay=0.0, enabled=True)

    with torch.no_grad():
        model.weight.fill_(5.0)

    with pytest.raises(RuntimeError):
        with ema.applied(model):
            raise RuntimeError("boom")

    assert torch.allclose(model.weight, torch.full((2, 2), 5.0))


def test_disabled_applied_is_noop():
    model = _model(fill=4.0)
    ema   = WeightEma(model, decay=0.5, enabled=False)

    with ema.applied(model):
        assert torch.allclose(model.weight, torch.full((2, 2), 4.0))


def test_state_roundtrip():
    model = _model(fill=1.0)
    ema   = WeightEma(model, decay=0.5, enabled=True)

    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)

    other = WeightEma(model, decay=0.5, enabled=True)
    other.load_state_dict(ema.state_dict())

    assert torch.allclose(other.shadow["weight"], ema.shadow["weight"])


def test_state_enabled_mismatch_raises():
    model    = _model()
    enabled  = WeightEma(model, decay=0.5, enabled=True)
    disabled = WeightEma(model, decay=0.5, enabled=False)

    with pytest.raises(ValueError):
        disabled.load_state_dict(enabled.state_dict())

    with pytest.raises(ValueError):
        enabled.load_state_dict(disabled.state_dict())
