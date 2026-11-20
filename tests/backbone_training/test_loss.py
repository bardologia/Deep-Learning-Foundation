from __future__ import annotations

import pytest
import torch

from configuration.training.general.loss import LossConfig, LossKind
from pipelines.backbone.training.loss    import Loss

from tests.conftest import SilentLogger


def _loss(**overrides) -> Loss:
    return Loss(LossConfig(**overrides), ["a", "b"], SilentLogger())


def _tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets     = torch.zeros(1, 2, 2, 2)
    predictions = torch.zeros(1, 2, 2, 2)

    predictions[0, 0] = 2.0
    predictions[0, 1] = 0.5

    return predictions, targets, torch.ones(1, 1, 2, 2)


@pytest.mark.parametrize("kind, a, b", [(LossKind.MSE, 4.0, 0.25), (LossKind.L1, 2.0, 0.5), (LossKind.HUBER, 1.5, 0.125)])
def test_every_kind_scores_each_channel_and_averages_them(kind, a, b):
    out = _loss(kind=kind)(*_tensors())

    assert out["channels"]["a"].value.item() == pytest.approx(a)
    assert out["channels"]["b"].value.item() == pytest.approx(b)
    assert out["total_loss"].value.item()    == pytest.approx((a + b) / 2.0)


def test_channel_weights_reweigh_the_total():
    out = _loss(channel_weights=[3.0, 1.0])(*_tensors())

    assert out["total_loss"].value.item() == pytest.approx((3.0 * 4.0 + 0.25) / 4.0)


def test_pixel_weights_leave_padding_out_and_pool_by_weight():
    predictions, targets, weights = _tensors()

    predictions[0, 0, 0, 0] = 100.0
    weights[0, 0, 0, 0]     = 0.0

    out = _loss()(predictions, targets, weights)

    assert out["channels"]["a"].value.item() == pytest.approx(4.0)
    assert out["total_loss"].weight.item()   == pytest.approx(3.0)


def test_the_total_backpropagates_to_the_predictions():
    predictions, targets, weights = _tensors()
    predictions.requires_grad_(True)

    _loss()(predictions, targets, weights)["total_loss"].value.backward()

    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
