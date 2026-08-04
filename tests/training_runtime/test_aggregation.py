from __future__ import annotations

import pytest
import torch

from tools.training.aggregation import MetricAggregator


def _loss_dict(components, monitor, occupancy=None, physical=None):
    return {"components": components, "monitor": monitor, "occupancy": occupancy or {}, "physical": physical or {}}


def test_single_add_reduces_to_same_values():
    agg = MetricAggregator()
    agg.add(_loss_dict({"a": 2.0}, {"m": 6.0}))

    assert agg.reduce_components()  == {"a": 2.0}
    assert agg.reduce_monitor()     == {"m": 6.0}


def test_mean_over_multiple_batches():
    agg = MetricAggregator()
    agg.add(_loss_dict({"a": 1.0}, {"m": 100.0}))
    agg.add(_loss_dict({"a": 3.0}, {"m": 200.0}))
    agg.add(_loss_dict({"a": 5.0}, {"m": 300.0}))

    assert agg.reduce_components()["a"] == pytest.approx(3.0)
    assert agg.reduce_monitor()["m"]    == pytest.approx(200.0)


def test_occupancy_channel_reduces_independently():
    agg = MetricAggregator()
    agg.add(_loss_dict({}, {}, {"count/exact_frac": 0.4}))
    agg.add(_loss_dict({}, {}, {"count/exact_frac": 0.6}))

    assert agg.reduce_occupancy()["count/exact_frac"] == pytest.approx(0.5)


def test_disjoint_keys_average_over_their_own_batches():
    agg = MetricAggregator()
    agg.add(_loss_dict({"a": 2.0}, {}))
    agg.add(_loss_dict({"b": 4.0}, {}))

    reduced = agg.reduce_components()
    assert reduced["a"] == pytest.approx(2.0)
    assert reduced["b"] == pytest.approx(4.0)


def test_sparse_occupancy_key_not_diluted_by_absent_batches():
    agg = MetricAggregator()
    agg.add(_loss_dict({}, {}, {"count/exact_frac": 0.5, "count/acc_gt3": 1.0}))
    agg.add(_loss_dict({}, {}, {"count/exact_frac": 0.5}))
    agg.add(_loss_dict({}, {}, {"count/exact_frac": 0.5, "count/acc_gt3": 0.0}))

    reduced = agg.reduce_occupancy()
    assert reduced["count/exact_frac"] == pytest.approx(0.5)
    assert reduced["count/acc_gt3"]    == pytest.approx(0.5)


def test_reduce_on_empty_returns_empty():
    agg = MetricAggregator()
    assert agg.reduce_components() == {}


def test_int_values_coerced_to_float():
    agg = MetricAggregator()
    agg.add(_loss_dict({"a": 2}, {"m": 8}))

    assert isinstance(agg.components_sum["a"], float)
    assert agg.components_sum["a"] == 2.0


def test_tensor_values_accumulate_without_immediate_conversion():
    agg = MetricAggregator()
    agg.add(_loss_dict({"a": torch.tensor(2.0, requires_grad=True)}, {}))
    agg.add(_loss_dict({"a": torch.tensor(4.0)}, {}))

    assert isinstance(agg.components_sum["a"], torch.Tensor)
    assert agg.components_sum["a"].requires_grad is False
    assert agg.reduce_components()["a"] == pytest.approx(3.0)


def test_physical_channel_reduces_independently():
    agg = MetricAggregator()
    agg.add(_loss_dict({}, {}, physical={"mu_mae_m": torch.tensor(2.0)}))
    agg.add(_loss_dict({}, {}, physical={"mu_mae_m": torch.tensor(4.0)}))

    assert agg.reduce_physical()["mu_mae_m"] == pytest.approx(3.0)
