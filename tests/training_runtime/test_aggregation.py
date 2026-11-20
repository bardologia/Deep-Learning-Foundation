from __future__ import annotations

import re

import pytest
import torch

from tools.training.aggregation import MetricAggregator, ScalarsPage, WeightedMean


def _mean(value: float, weight: float) -> WeightedMean:
    return WeightedMean(torch.tensor(value * weight), torch.tensor(weight))


def _loss_dict(total: float, weight: float, **channels: float) -> dict:
    return {"total_loss": _mean(total, weight), "channels": {name: _mean(value, weight) for name, value in channels.items()}}


def test_weighted_mean_of_values_and_weights():
    mean = WeightedMean.of(torch.tensor([[1.0, 3.0], [5.0, 7.0]]), torch.tensor([[1.0, 0.0], [1.0, 2.0]]))

    assert mean.total.item()  == pytest.approx(20.0)
    assert mean.weight.item() == pytest.approx(4.0)
    assert mean.value.item()  == pytest.approx(5.0)


def test_weighted_mean_of_zero_weight_stays_finite():
    assert WeightedMean(torch.tensor(0.0), torch.tensor(0.0)).value.item() == 0.0


def test_a_single_batch_reduces_to_its_own_values():
    agg = MetricAggregator()
    agg.add(_loss_dict(2.0, 10.0, a=1.0, b=3.0))

    assert agg.reduce_total()    == pytest.approx(2.0)
    assert agg.reduce_channels() == pytest.approx({"a": 1.0, "b": 3.0})


def test_batches_pool_by_their_pixel_weight_not_by_their_count():
    agg = MetricAggregator()
    agg.add(_loss_dict(1.0, 30.0, a=1.0))
    agg.add(_loss_dict(5.0, 10.0, a=5.0))

    assert agg.reduce_total()         == pytest.approx((1.0 * 30.0 + 5.0 * 10.0) / 40.0)
    assert agg.reduce_channels()["a"] == pytest.approx(2.0)


def test_the_scalars_page_lays_the_loss_and_optimiser_families_out_as_multiline_charts():
    layout = ScalarsPage().loss().optimiser().layout()

    assert list(layout) == ["Loss", "Optimiser"]
    assert list(layout["Loss"]) == ["Total loss by stage", "Loss per target channel, training", "Loss per target channel, validation"]
    assert layout["Loss"]["Total loss by stage"] == ["Multiline", ["loss/train$", "loss/val$", "loss/test$"]]
    assert layout["Optimiser"]["Gradient norm, total and per group"] == ["Multiline", ["optim/grad_norm$", "optim/grad_norm/"]]

    for charts in layout.values():
        for kind, patterns in charts.values():
            assert kind == "Multiline"
            assert all(re.compile(pattern) for pattern in patterns)


def test_the_scalars_page_patterns_match_the_tags_the_trainer_writes():
    layout   = ScalarsPage().loss().optimiser().layout()
    channels = layout["Loss"]["Loss per target channel, validation"][1][0]
    groups   = layout["Optimiser"]["Learning rate per group"][1][0]
    total    = layout["Loss"]["Total loss by stage"][1][0]

    assert re.match(channels, "loss_channels/pointwise/val")
    assert not re.match(channels, "loss_channels/pointwise/train") and not re.match(channels, "loss_channels/pointwise/val/extra")
    assert re.match(groups, "optim/lr/encoder") and re.match(groups, "optim/lr/output_head")
    assert re.match(total, "loss/train") and not re.match(total, "loss/train_extra")
