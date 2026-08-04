from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.diagnostics                  import ActivationXrayConfig
from tools.diagnostics.activation_recorder      import ActivationRecorder
from tools.diagnostics.activation_xray_analysis import ActivationIssueDetector, ActivationXraySummarizer


class _TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.act  = torch.nn.ReLU()
        self.head = torch.nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, x):
        return self.head(self.act(self.conv(x)))


def _record(model, batches) -> dict:
    recorder = ActivationRecorder(model)
    recorder.attach_stats()

    with torch.no_grad():
        for batch in batches:
            model(batch)

    recorder.detach()
    return recorder.stats()


def test_recorder_covers_all_leaf_modules():
    stats = _record(_TinyNet(), [torch.randn(2, 2, 8, 8)])

    assert set(stats) == {"conv", "act", "head"}
    assert stats["conv"]["n_channels"] == 4
    assert stats["conv"]["n_batches"]  == 1


def test_recorder_accumulates_across_batches():
    stats = _record(_TinyNet(), [torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)])

    assert stats["conv"]["n_batches"]  == 2
    assert stats["conv"]["n_elements"] == 2 * 2 * 4 * 8 * 8


def test_recorder_detects_dead_channels():
    model = _TinyNet()
    with torch.no_grad():
        model.conv.weight[0] = 0.0
        model.conv.bias[0]   = -100.0

    stats = _record(model, [torch.rand(2, 2, 8, 8)])

    assert stats["act"]["dead_channels"]     >= 1
    assert stats["act"]["dead_channel_frac"] >= 0.25


def test_recorder_stats_are_exact_for_known_values():
    class _Identity(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Identity()

        def forward(self, x):
            return self.layer(x)

    values = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 4)
    stats  = _record(_Identity(), [values])["layer"]

    assert stats["zero_frac"] == pytest.approx(0.25)
    assert stats["mean"]      == pytest.approx(1.5)
    assert stats["max_abs"]   == pytest.approx(3.0)


def test_store_mode_returns_layer_outputs():
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
    recorder = ActivationRecorder(_TinyNet())

    with pytest.raises(KeyError):
        recorder.attach_store(["missing_layer"])


def test_issue_detector_flags_dead_and_exploding_layers():
    config = ActivationXrayConfig()
    stats  = {
        "dead": {
            "name": "dead", "module_type": "ReLU", "shape": [4, 8, 8], "n_batches": 1, "n_elements": 100,
            "nonfinite_frac": 0.0, "zero_frac": 0.995, "mean": 0.0, "std": 0.001, "max_abs": 0.01,
            "abs_p99": 0.01, "n_channels": 4, "dead_channels": 4, "dead_channel_frac": 1.0, "hist_counts": [0] * 24,
        },
        "explode": {
            "name": "explode", "module_type": "Conv2d", "shape": [4, 8, 8], "n_batches": 1, "n_elements": 100,
            "nonfinite_frac": 0.0, "zero_frac": 0.0, "mean": 1.0, "std": 5.0, "max_abs": 1e6,
            "abs_p99": 1e5, "n_channels": 4, "dead_channels": 0, "dead_channel_frac": 0.0, "hist_counts": [0] * 24,
        },
        "clean": {
            "name": "clean", "module_type": "Conv2d", "shape": [4, 8, 8], "n_batches": 1, "n_elements": 100,
            "nonfinite_frac": 0.0, "zero_frac": 0.3, "mean": 0.5, "std": 1.0, "max_abs": 4.0,
            "abs_p99": 3.0, "n_channels": 4, "dead_channels": 0, "dead_channel_frac": 0.0, "hist_counts": [0] * 24,
        },
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
    config = ActivationXrayConfig()
    stats  = {
        "naninf": {
            "name": "naninf", "module_type": "Conv2d", "shape": [4], "n_batches": 1, "n_elements": 100,
            "nonfinite_frac": 0.02, "zero_frac": 0.0, "mean": 0.5, "std": 1.0, "max_abs": 2.0,
            "abs_p99": 1.5, "n_channels": 4, "dead_channels": 0, "dead_channel_frac": 0.0, "hist_counts": [0] * 24,
        },
    }

    reports = ActivationIssueDetector(config).run(stats)

    assert reports[0].severity == "critical"
    assert any(issue.code == "nonfinite_activations" for issue in reports[0].issues)


def test_store_mode_survives_inplace_activations():
    class _InplaceNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 4, kernel_size=1, bias=True)
            self.act  = torch.nn.ReLU(inplace=True)

        def forward(self, x):
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
    stats = _record(_TinyNet(), [torch.randn(1, 2, 8, 8)])

    assert stats["conv"]["first_seen"] < stats["act"]["first_seen"] < stats["head"]["first_seen"]


def test_abs_percentile_reports_overflow_via_max_abs():
    stats = _record(_TinyNet(), [torch.randn(2, 2, 8, 8) * 1e6])

    assert stats["conv"]["abs_p99"] == stats["conv"]["max_abs"]
