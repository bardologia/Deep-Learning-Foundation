"""Tests for activation recording and the activation x-ray issue detector.

Covers hook coverage of leaf modules, statistic accumulation across batches,
dead-channel and channel-concentration measures, the store mode that keeps raw
layer outputs safe from in-place activations, and the severity rules that turn
layer statistics into flagged reports and a run summary.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.diagnostics                  import ActivationXrayConfig
from tools.diagnostics.activation_recorder      import ActivationRecorder
from tools.diagnostics.activation_xray_analysis import ActivationIssueDetector, ActivationXraySummarizer


class _TinyNet(torch.nn.Module):
    """Three-layer convolutional network used as a recording target."""
    def __init__(self) -> None:
        """Builds the convolution, ReLU and 1x1 head layers."""
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.act  = torch.nn.ReLU()
        self.head = torch.nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, x):
        """Returns the head output for an input of shape (batch, 2, height, width)."""
        return self.head(self.act(self.conv(x)))


def _record(model, batches) -> dict:
    """Returns the per-layer activation statistics after running the batches through the model."""
    recorder = ActivationRecorder(model)
    recorder.attach_stats()

    with torch.no_grad():
        for batch in batches:
            model(batch)

    recorder.detach()
    return recorder.stats()


def test_recorder_covers_all_leaf_modules():
    """Statistics are collected for every leaf module with the right channel and batch counts."""
    stats = _record(_TinyNet(), [torch.randn(2, 2, 8, 8)])

    assert set(stats) == {"conv", "act", "head"}
    assert stats["conv"]["n_channels"] == 4
    assert stats["conv"]["n_batches"]  == 1


def test_recorder_accumulates_across_batches():
    """Batch and element counts accumulate over successive forward passes."""
    stats = _record(_TinyNet(), [torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)])

    assert stats["conv"]["n_batches"]  == 2
    assert stats["conv"]["n_elements"] == 2 * 2 * 4 * 8 * 8


def test_recorder_detects_dead_channels():
    """A zeroed convolution channel with strong negative bias shows up as a dead ReLU channel."""
    model = _TinyNet()
    with torch.no_grad():
        model.conv.weight[0] = 0.0
        model.conv.bias[0]   = -100.0

    stats = _record(model, [torch.rand(2, 2, 8, 8)])

    assert stats["act"]["dead_channels"]     >= 1
    assert stats["act"]["dead_channel_frac"] >= 0.25


def test_recorder_stats_are_exact_for_known_values():
    """Zero fraction, mean and maximum absolute value match hand-computed values on a fixed tensor."""
    class _Identity(torch.nn.Module):
        """Module wrapping a single Identity layer so its output can be recorded."""
        def __init__(self) -> None:
            """Builds the identity layer."""
            super().__init__()
            self.layer = torch.nn.Identity()

        def forward(self, x):
            """Returns the input unchanged through the identity layer."""
            return self.layer(x)

    values = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 4)
    stats  = _record(_Identity(), [values])["layer"]

    assert stats["zero_frac"] == pytest.approx(0.25)
    assert stats["mean"]      == pytest.approx(1.5)
    assert stats["max_abs"]   == pytest.approx(3.0)


def test_store_mode_returns_layer_outputs():
    """Store mode keeps the raw outputs of the selected layers with their full shapes."""
    model    = _TinyNet()
    recorder = ActivationRecorder(model)
    recorder.attach_store(["conv", "act"])

    with torch.no_grad():
        model(torch.randn(1, 2, 8, 8))

    recorder.detach()
    stored = recorder.stored()

    assert set(stored) == {"conv", "act"}
    assert stored["conv"].shape == (1, 4, 8, 8)
    assert torch.equal(stored["act"], stored["conv"].clamp_min(0.0))


def test_store_mode_unknown_module_raises():
    """Requesting storage for a module name that does not exist raises KeyError."""
    recorder = ActivationRecorder(_TinyNet())

    with pytest.raises(KeyError):
        recorder.attach_store(["missing_layer"])


def _layer_stats(name: str, **overrides) -> dict:
    """Returns a healthy layer statistics record with the given name and overrides applied."""
    stats = {
        "name": name, "module_type": "Conv2d", "shape": [4, 8, 8], "n_batches": 1, "n_elements": 100,
        "nonfinite_frac": 0.0, "zero_frac": 0.3, "mean": 0.5, "std": 1.0, "max_abs": 4.0,
        "abs_p50": 0.5, "abs_p99": 3.0, "dynamic_range": 6.0, "n_channels": 4, "dead_channels": 0,
        "dead_channel_frac": 0.0, "effective_channel_frac": 0.9, "channel_gini": 0.1, "hist_counts": [0] * 24,
    }
    stats.update(overrides)
    return stats


def test_issue_detector_flags_dead_and_exploding_layers():
    """A dead layer is critical, an exploding layer is a warning, and the summary counts both as flagged."""
    config = ActivationXrayConfig()
    stats  = {
        "dead"    : _layer_stats("dead", module_type="ReLU", zero_frac=0.995, mean=0.0, std=0.001, max_abs=0.01, abs_p50=0.001, abs_p99=0.01, dead_channels=4, dead_channel_frac=1.0, effective_channel_frac=0.0, channel_gini=None),
        "explode" : _layer_stats("explode", mean=1.0, std=5.0, max_abs=1e6, abs_p99=1e5, zero_frac=0.0),
        "clean"   : _layer_stats("clean"),
    }

    reports  = ActivationIssueDetector(config).run(stats)
    by_name  = {report.name: report for report in reports}
    summary  = ActivationXraySummarizer().build(reports, "/tmp/run", n_batches=1)

    assert by_name["dead"].severity    == "critical"
    assert by_name["explode"].severity == "warning"
    assert by_name["clean"].severity   == "ok"
    assert summary["verdict"]          == "critical issues detected"
    assert summary["flagged_layers"]   == 2


def test_nonfinite_activations_are_critical():
    """A layer with non-finite activations is reported as critical with the matching issue code."""
    config = ActivationXrayConfig()
    stats  = {"naninf": _layer_stats("naninf", shape=[4], nonfinite_frac=0.02, zero_frac=0.0, max_abs=2.0, abs_p99=1.5)}

    reports = ActivationIssueDetector(config).run(stats)

    assert reports[0].severity == "critical"
    assert any(issue.code == "nonfinite_activations" for issue in reports[0].issues)


def test_channel_collapse_is_flagged_only_on_live_layers():
    """Channel collapse is reported for concentrated live layers but suppressed on dead or healthy ones."""
    config = ActivationXrayConfig()
    stats  = {
        "collapsed" : _layer_stats("collapsed", n_channels=64, effective_channel_frac=0.05, channel_gini=0.9),
        "dead"      : _layer_stats("dead", zero_frac=0.995, n_channels=64, effective_channel_frac=0.05, channel_gini=0.9, dead_channels=60, dead_channel_frac=0.9375),
        "healthy"   : _layer_stats("healthy", n_channels=64, effective_channel_frac=0.6, channel_gini=0.2),
    }

    reports = ActivationIssueDetector(config).run(stats)
    by_name = {report.name: report for report in reports}

    assert any(issue.code == "channel_collapse" for issue in by_name["collapsed"].issues)
    assert not any(issue.code == "channel_collapse" for issue in by_name["dead"].issues)
    assert not any(issue.code == "channel_collapse" for issue in by_name["healthy"].issues)


def test_summary_ranks_worst_layers_by_severity():
    """The summary lists the critical layer ahead of the merely warned one."""
    config = ActivationXrayConfig()
    stats  = {
        "warned"   : _layer_stats("warned", max_abs=1e6),
        "critical" : _layer_stats("critical", zero_frac=0.995, effective_channel_frac=0.0, channel_gini=None),
    }

    reports = ActivationIssueDetector(config).run(stats)
    summary = ActivationXraySummarizer().build(reports, "/tmp/run", n_batches=1)

    assert summary["worst_layers"][0] == "critical"
    assert "warned" in summary["worst_layers"]


def test_store_mode_survives_inplace_activations():
    """Stored pre-activation outputs keep their negative values despite the in-place ReLU."""
    class _InplaceNet(torch.nn.Module):
        """Convolution followed by an in-place ReLU, used to test output cloning."""
        def __init__(self) -> None:
            """Builds the 1x1 convolution and the in-place ReLU."""
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 4, kernel_size=1, bias=True)
            self.act  = torch.nn.ReLU(inplace=True)

        def forward(self, x):
            """Returns the in-place activated convolution output."""
            return self.act(self.conv(x))

    torch.manual_seed(0)
    model    = _InplaceNet()
    recorder = ActivationRecorder(model)
    recorder.attach_store(["conv"])

    with torch.no_grad():
        model(torch.randn(1, 2, 6, 6))

    recorder.detach()
    stored = recorder.stored()["conv"]

    assert float(stored.min()) < 0.0


def test_stats_record_forward_order():
    """First-seen counters order the layers along the forward pass."""
    stats = _record(_TinyNet(), [torch.randn(1, 2, 8, 8)])

    assert stats["conv"]["first_seen"] < stats["act"]["first_seen"] < stats["head"]["first_seen"]


def test_abs_percentile_reports_overflow_via_max_abs():
    """The 99th absolute percentile saturates at the maximum absolute value for huge activations."""
    stats = _record(_TinyNet(), [torch.randn(2, 2, 8, 8) * 1e6])

    assert stats["conv"]["abs_p99"] == stats["conv"]["max_abs"]


class _PassThrough(torch.nn.Module):
    """Module wrapping a single Identity layer for channel-statistic tests."""
    def __init__(self) -> None:
        """Builds the identity layer."""
        super().__init__()
        self.layer = torch.nn.Identity()

    def forward(self, x):
        """Returns the input unchanged through the identity layer."""
        return self.layer(x)


def test_channel_concentration_separates_uniform_from_collapsed():
    """Uniform channels give full effective fraction and zero Gini, a single dominant channel gives the opposite."""
    uniform   = torch.ones(1, 4, 4, 4)
    collapsed = torch.zeros(1, 4, 4, 4)
    collapsed[:, 0] = 1000.0
    collapsed[:, 1:] = 0.001

    uniform_stats   = _record(_PassThrough(), [uniform])["layer"]
    collapsed_stats = _record(_PassThrough(), [collapsed])["layer"]

    assert uniform_stats["effective_channel_frac"] == pytest.approx(1.0)
    assert uniform_stats["channel_gini"]           == pytest.approx(0.0, abs=1e-9)
    assert collapsed_stats["effective_channel_frac"] < 0.3
    assert collapsed_stats["channel_gini"]           > 0.7


def test_all_dead_channels_report_zero_effective_fraction():
    """An all-zero activation reports zero effective channels and an undefined Gini coefficient."""
    stats = _record(_PassThrough(), [torch.zeros(1, 4, 4, 4)])["layer"]

    assert stats["effective_channel_frac"] == 0.0
    assert stats["channel_gini"] is None


def test_constant_activations_have_unit_dynamic_range():
    """A constant activation has unit dynamic range and equal median and 99th absolute percentiles."""
    stats = _record(_PassThrough(), [torch.full((1, 4, 4, 4), 2.0)])["layer"]

    assert stats["dynamic_range"] == pytest.approx(1.0)
    assert stats["abs_p50"] == stats["abs_p99"]
