
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest           import RecordingWriter
from tools.monitoring.tracker import Tracker
from unittest.mock import Mock


def test_active_false_without_writer():
    assert Tracker().active is False


def test_active_true_with_writer():
    assert Tracker(writer=RecordingWriter()).active is True


def test_set_step_and_advance():
    t = Tracker()
    t.set_step(10)

    assert t._step      == 10
    assert t.advance()  == 11
    assert t.advance(4) == 15
    assert t._step      == 15


def test_set_step_casts_to_int():
    t = Tracker()
    t.set_step(3.9)

    assert t._step == 3


def test_scope_nesting_and_tagging():
    t = Tracker()

    assert t._tag("x") == "x"

    with t.scope("a"):
        assert t._tag("x") == "a/x"
        with t.scope("b"):
            assert t._tag("x") == "a/b/x"
        assert t._tag("x") == "a/x"

    assert t._tag("x") == "x"
    assert t._scopes   == []


def test_scope_pops_on_exception():
    t = Tracker()
    with pytest.raises(RuntimeError):
        with t.scope("a"):
            Mock(side_effect=RuntimeError("boom"))()

    assert t._scopes == []


def test_resolve_step_default_and_override():
    t = Tracker()
    t.set_step(7)

    assert t._resolve(None) == 7
    assert t._resolve(2)    == 2


def test_log_scalar_records_exact_value():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.set_step(3)
    t.log_scalar("loss", 0.5)

    assert w.scalars == [("loss", 0.5, 3)]


def test_log_scalar_casts_to_float():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_scalar("n", 4, step=1)

    tag, value, step = w.scalars[0]

    assert isinstance(value, float)
    assert value == 4.0


def test_log_scalar_explicit_step_overrides():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.set_step(100)
    t.log_scalar("m", 1.0, step=9)

    assert w.scalars[0] == ("m", 1.0, 9)


def test_log_scalar_uses_scope_in_tag():
    w = RecordingWriter()
    t = Tracker(writer=w)
    with t.scope("train"):
        t.log_scalar("loss", 0.2, step=0)

    assert w.scalars[0][0] == "train/loss"


def test_log_metrics_prefixes_and_records_all():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_metrics("sys", {"a": 1.0, "b": 2.0}, step=5)

    recorded = {(tag, val) for tag, val, _ in w.scalars}

    assert ("sys/a", 1.0) in recorded
    assert ("sys/b", 2.0) in recorded
    assert all(step == 5 for _, _, step in w.scalars)


def test_log_metrics_skips_unconvertible():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_metrics("sys", {"good": 3.0, "bad": "not_a_number", "none": None}, step=0)

    tags = [tag for tag, _, _ in w.scalars]

    assert tags == ["sys/good"]


def test_log_histogram_records_float32_array():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_histogram("weights", [[1, 2], [3, 4]], step=2)

    tag, values, step, bins = w.histograms[0]

    assert tag == "weights"
    assert values.dtype == np.float32
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert step == 2
    assert bins == "auto"


def test_log_histogram_drops_non_finite_values_and_counts_them():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_histogram("param_hist/mu", [1.0, np.nan, np.inf, 2.0, -np.inf], step=3)

    tag, values, step, _bins = w.histograms[0]

    assert tag == "param_hist/mu"
    assert values.tolist() == [1.0, 2.0]
    assert step == 3
    assert ("diagnostics/param_hist/mu_nonfinite", 3.0, 3) in w.scalars


def test_log_histogram_of_nothing_finite_logs_an_empty_marker_instead():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_histogram("grad", [], step=1)
    t.log_histogram("grad", [np.nan, np.nan], step=2)

    assert w.histograms == []
    assert ("diagnostics/grad_empty", 1.0, 1) in w.scalars
    assert ("diagnostics/grad_empty", 1.0, 2) in w.scalars
    assert ("diagnostics/grad_nonfinite", 2.0, 2) in w.scalars
    assert not any(tag == "diagnostics/grad_nonfinite" and step == 1 for tag, _value, step in w.scalars)


def test_log_histogram_of_finite_values_logs_no_diagnostics():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_histogram("weights", [0.5, 0.25], step=1)

    assert len(w.histograms) == 1
    assert w.scalars == []


def test_log_layout_forwards_the_custom_scalars_layout():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.log_layout({"Loss": {"Total loss by stage": ["Multiline", ["loss/train$", "loss/val$"]]}})

    assert w.layouts == [{"Loss": {"Total loss by stage": ["Multiline", ["loss/train$", "loss/val$"]]}}]


def test_inactive_tracker_records_nothing():
    t = Tracker()
    t.log_scalar("loss", 1.0)
    t.log_metrics("sys", {"a": 1.0})
    t.log_histogram("h", [1, 2, 3])
    t.log_layout({"Loss": {}})


def test_flush_and_close_delegate():
    w = RecordingWriter()
    t = Tracker(writer=w)
    t.flush()
    t.close()

    assert w.flushed == 1
    assert w.closed  == 1


def test_flush_and_close_noop_without_writer():
    t = Tracker()
    t.flush()
    t.close()
