
from __future__ import annotations

import re

import pytest
import torch

from tools.training.aggregation import MetricAggregator, ScalarsPage


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
    agg.add(_loss_dict({}, {}, physical={"mu_mae": torch.tensor(2.0)}))
    agg.add(_loss_dict({}, {}, physical={"mu_mae": torch.tensor(4.0)}))

    assert agg.reduce_physical()["mu_mae"] == pytest.approx(3.0)


def test_the_scalars_page_lays_the_base_families_out_as_multiline_charts():
    layout = ScalarsPage().loss().optimiser().layout()

    assert list(layout) == ["Loss", "Optimiser"]
    assert list(layout["Loss"]) == ["Total loss by stage", "Loss terms, training", "Loss terms, validation", "Monitors, training", "Monitors, validation"]
    assert layout["Loss"]["Total loss by stage"]  == ["Multiline", ["loss/train$", "loss/val$", "loss/test$"]]
    assert layout["Loss"]["Loss terms, training"] == ["Multiline", ["loss_terms/.+/train$"]]
    assert layout["Optimiser"]["Gradient norm, total and per group"] == ["Multiline", ["optim/grad_norm$", "optim/grad_norm/"]]

    for charts in layout.values():
        for kind, patterns in charts.values():
            assert kind == "Multiline"
            assert all(re.compile(pattern) for pattern in patterns)


def test_the_scalars_page_patterns_match_the_tags_the_trainers_write_from_the_start():
    layout = ScalarsPage().loss().optimiser().parameter_errors().occupancy().slots().layout()
    terms  = layout["Loss"]["Loss terms, validation"][1][0]
    groups = layout["Optimiser"]["Learning rate per group"][1][0]
    total  = layout["Loss"]["Total loss by stage"][1][0]
    slots  = layout["Slots"]["Gradient norm per slot"][1][0]

    assert list(layout) == ["Loss", "Optimiser", "Diagnostics", "Slots"]
    assert list(layout["Diagnostics"]) == ["Parameter errors, training", "Parameter errors, validation", "Occupancy, training", "Occupancy, validation"]
    assert re.match(terms, "loss_terms/param_l1/val") and re.match(terms, "loss_terms/param_l1/amp/val")
    assert not re.match(terms, "loss_terms/param_l1/train") and not re.match(terms, "loss_terms/param_l1/val/extra")
    assert re.match(groups, "optim/lr/encoder") and re.match(groups, "optim/lr/output_head")
    assert re.match(total, "loss/train") and not re.match(total, "loss/train_extra")
    assert re.match(slots, "slot_vitals/grad_norm/slot_0/train") and not re.match(slots, "slot_vitals/amp_mean/slot_0/val")


def _drawn(patterns: list, tag: str) -> bool:
    return any(re.match(pattern, tag) for pattern in patterns)


def test_the_scalars_page_draws_slot_widths_and_centres_over_all_and_declared_pixels_on_one_chart_each():
    slots   = ScalarsPage().slots().layout()["Slots"]
    widths  = slots["Mean width per slot, all and declared pixels (m)"][1]
    centres = slots["Mean centre per slot, all and declared pixels (m)"][1]

    assert _drawn(widths, "slot_vitals/sigma_mean/slot_1/val") and _drawn(widths, "slot_vitals/sigma_mean_declared/slot_1/val")
    assert not _drawn(widths, "slot_vitals/mu_mean/slot_1/val") and not _drawn(widths, "slot_vitals/mu_mean_declared/slot_1/val")
    assert _drawn(centres, "slot_vitals/mu_mean/slot_0/val") and _drawn(centres, "slot_vitals/mu_mean_declared/slot_0/val")
    assert not _drawn(centres, "slot_vitals/sigma_mean/slot_0/val")
