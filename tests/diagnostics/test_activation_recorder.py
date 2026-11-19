
from __future__ import annotations

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


def _layer_stats(name: str, **overrides) -> dict:
    stats = {
        "name": name, "module_type": "Conv2d", "shape": [4, 8, 8], "n_batches": 1, "n_elements": 100,
        "nonfinite_frac": 0.0, "zero_frac": 0.3, "mean": 0.5, "std": 1.0, "max_abs": 4.0,
        "abs_p50": 0.5, "abs_p99": 3.0, "dynamic_range": 6.0, "n_channels": 4, "dead_channels": 0,
        "dead_channel_frac": 0.0, "effective_channel_frac": 0.9, "channel_gini": 0.1, "hist_counts": [0] * 24,
    }
    stats.update(overrides)
    return stats


def test_issue_detector_flags_dead_and_exploding_layers():
    config = ActivationXrayConfig()
    stats  = {
        "dead"    : _layer_stats("dead", module_type="ReLU", zero_frac=0.995, mean=0.0, std=0.001, max_abs=0.01, abs_p50=0.001, abs_p99=0.01, dead_channels=4, dead_channel_frac=1.0, effective_channel_frac=0.0, channel_gini=None),
        "explode" : _layer_stats("explode", mean=1.0, std=5.0, max_abs=1e6, abs_p99=1e5, zero_frac=0.0),
        "clean"   : _layer_stats("clean"),
    }

    reports  = ActivationIssueDetector(config).run(stats)
    by_name  = {report.name: report for report in reports}
    summary  = ActivationXraySummarizer(ActivationXrayConfig()).build(reports, "/tmp/run", n_batches=1, reused_modules={}, silent_layers={})

    assert by_name["dead"].severity    == "critical"
    assert by_name["explode"].severity == "warning"
    assert by_name["clean"].severity   == "ok"
    assert summary["verdict"]          == "critical issues detected"
    assert summary["flagged_layers"]   == 2


def test_nonfinite_activations_are_critical():
    config = ActivationXrayConfig()
    stats  = {"naninf": _layer_stats("naninf", shape=[4], nonfinite_frac=0.02, zero_frac=0.0, max_abs=2.0, abs_p99=1.5)}

    reports = ActivationIssueDetector(config).run(stats)

    assert reports[0].severity == "critical"
    assert any(issue.code == "nonfinite_activations" for issue in reports[0].issues)


def test_channel_collapse_is_flagged_only_on_live_layers():
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
    config = ActivationXrayConfig()
    stats  = {
        "warned"   : _layer_stats("warned", max_abs=1e6),
        "critical" : _layer_stats("critical", zero_frac=0.995, effective_channel_frac=0.0, channel_gini=None),
    }

    reports = ActivationIssueDetector(config).run(stats)
    summary = ActivationXraySummarizer(ActivationXrayConfig()).build(reports, "/tmp/run", n_batches=1, reused_modules={}, silent_layers={})

    assert summary["worst_layers"][0] == "critical"
    assert "warned" in summary["worst_layers"]


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


class _PassThrough(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Identity()

    def forward(self, x):
        return self.layer(x)


def test_channel_concentration_separates_uniform_from_collapsed():
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
    stats = _record(_PassThrough(), [torch.zeros(1, 4, 4, 4)])["layer"]

    assert stats["effective_channel_frac"] == 0.0
    assert stats["channel_gini"] is None


def test_constant_activations_have_unit_dynamic_range():
    stats = _record(_PassThrough(), [torch.full((1, 4, 4, 4), 2.0)])["layer"]

    assert stats["dynamic_range"] == pytest.approx(1.0)
    assert stats["abs_p50"] == stats["abs_p99"]


def test_abs_percentiles_describe_the_non_zero_magnitudes():
    values          = torch.zeros(1, 10)
    values[0, :4]   = 0.5
    stats           = _record(_PassThrough(), [values])["layer"]

    assert stats["zero_frac"]     == pytest.approx(0.6)
    assert stats["abs_p50"]       == pytest.approx(0.5)
    assert stats["abs_p99"]       == pytest.approx(0.5)
    assert stats["dynamic_range"] == pytest.approx(1.0)


def test_all_zero_layer_has_no_dynamic_range():
    stats = _record(_PassThrough(), [torch.zeros(1, 8)])["layer"]

    assert stats["abs_p50"]       == 0.0
    assert stats["abs_p99"]       == 0.0
    assert stats["dynamic_range"] is None


class _SharedActNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_a = torch.nn.Conv2d(2, 4, kernel_size=1)
        self.conv_b = torch.nn.Conv2d(4, 8, kernel_size=1)
        self.act    = torch.nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_b(self.act(self.conv_a(x))))


def test_reused_leaf_module_is_recorded_per_call_site():
    model    = _SharedActNet()
    recorder = ActivationRecorder(model)
    recorder.attach_stats()

    with torch.no_grad():
        model(torch.randn(2, 2, 6, 6))
        model(torch.randn(2, 2, 6, 6))

    recorder.detach()
    stats = recorder.stats()

    assert {"act", "act#1"} <= set(stats)
    assert stats["act"]["n_channels"]   == 4
    assert stats["act#1"]["n_channels"] == 8
    assert stats["act"]["n_batches"]    == 2
    assert stats["act#1"]["n_batches"]  == 2
    assert stats["act#1"]["module_type"] == "ReLU"
    assert recorder.reused_modules() == {"act": 2}


def test_unshared_modules_are_not_reported_as_reused():
    recorder = ActivationRecorder(_TinyNet())
    recorder.attach_stats()

    with torch.no_grad():
        recorder.model(torch.randn(1, 2, 8, 8))

    recorder.detach()

    assert recorder.reused_modules() == {}


def test_summary_carries_the_reused_modules():
    reports = ActivationIssueDetector(ActivationXrayConfig()).run({"clean": _layer_stats("clean")})
    summary = ActivationXraySummarizer(ActivationXrayConfig()).build(reports, "/tmp/run", n_batches=1, reused_modules={"pool": 4}, silent_layers={})

    assert summary["reused_modules"] == {"pool": 4}


class _UnwiredNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv   = torch.nn.Conv2d(2, 4, kernel_size=1)
        self.unused = torch.nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_silent_leaf_modules_are_named_and_left_out_of_the_stats():
    model    = _UnwiredNet()
    recorder = ActivationRecorder(model)
    recorder.attach_stats()

    with torch.no_grad():
        model(torch.randn(1, 2, 4, 4))

    recorder.detach()

    assert set(recorder.stats())  == {"conv"}
    assert recorder.silent_layers() == {"unused": "Conv2d"}


def test_summary_counts_hooked_and_silent_layers():
    reports = ActivationIssueDetector(ActivationXrayConfig()).run({"clean": _layer_stats("clean")})
    summary = ActivationXraySummarizer(ActivationXrayConfig()).build(reports, "/tmp/run", n_batches=1, reused_modules={}, silent_layers={"unused": "Conv2d"})

    assert summary["layers"]        == 1
    assert summary["hooked_layers"] == 2
    assert summary["silent_layers"] == {"unused": "Conv2d"}


def test_summary_records_the_capture_settings_thresholds_and_provenance():
    config  = ActivationXrayConfig(split="val", batch_size=2, max_batches=3, dead_zero_frac_warn=0.8)
    reports = ActivationIssueDetector(config).run({"clean": _layer_stats("clean")})
    summary = ActivationXraySummarizer(config).build(reports, "/tmp/run", n_batches=3, reused_modules={}, silent_layers={})

    assert summary["checkpoint"]  == "best_model.pt"
    assert summary["split"]       == "val"
    assert summary["batch_size"]  == 2
    assert summary["max_batches"] == 3
    assert summary["thresholds"]["dead_zero_frac_warn"] == 0.8
    assert set(summary["thresholds"]) == set(ActivationXraySummarizer.THRESHOLDS)
    assert summary["provenance"]["git_commit"]


def test_hooked_layer_count_counts_a_reused_module_once():
    model    = _SharedActNet()
    recorder = ActivationRecorder(model)
    recorder.attach_stats()

    with torch.no_grad():
        model(torch.randn(2, 2, 6, 6))

    recorder.detach()
    reports = ActivationIssueDetector(ActivationXrayConfig()).run(recorder.stats())
    summary = ActivationXraySummarizer(ActivationXrayConfig()).build(reports, "/tmp/run", n_batches=1, reused_modules=recorder.reused_modules(), silent_layers=recorder.silent_layers())

    assert summary["layers"]        == 4
    assert summary["hooked_layers"] == 3
    assert ActivationRecorder.module_of("act#1") == "act"
    assert ActivationRecorder.module_of("conv_a") == "conv_a"
