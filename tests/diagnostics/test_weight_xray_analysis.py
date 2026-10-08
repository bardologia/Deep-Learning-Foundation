"""Tests for the weight x-ray statistics, size-limited checks and summary.

Covers the duplicate-unit and spectral checks that a size limit skips, so a
tensor the x-ray did not test is recorded as unchecked instead of clean, the
single-unit case that has no pair to compare, and the transposed convolutions
whose output units sit on dim 1 of the weight.
"""

from __future__ import annotations

import torch

from configuration.diagnostics              import WeightXrayConfig
from tools.diagnostics.weight_xray_analysis import IssueDetector, ModuleTypes, WeightAnalyzer, XraySummarizer


def _analyze(config: WeightXrayConfig, state_dict: dict) -> dict:
    """Returns the detected reports of a state dict keyed by tensor name."""
    reports = WeightAnalyzer(config, ModuleTypes.unavailable()).analyze(state_dict)
    reports = IssueDetector(config.thresholds).run(reports, state_dict)
    return {report.name: report for report in reports}


def test_duplicate_check_above_the_unit_limit_is_recorded_as_skipped():
    """A weight with more output units than duplicate_max_units carries the skip reason and no pair count."""
    config = WeightXrayConfig(duplicate_max_units=4)
    torch.manual_seed(0)

    reports = _analyze(config, {"wide.weight": torch.randn(8, 3), "narrow.weight": torch.randn(4, 3)})

    assert reports["wide.weight"].duplicate_units is None
    assert reports["wide.weight"].skipped_checks == {"duplicates": "8 output units > duplicate_max_units 4"}
    assert reports["narrow.weight"].duplicate_units == 0
    assert reports["narrow.weight"].skipped_checks == {}


def test_spectral_check_above_the_svd_limit_is_recorded_as_skipped():
    """A matrix whose smaller side exceeds svd_max_dim has no rank statistics and names the limit."""
    config = WeightXrayConfig(svd_max_dim=2)
    torch.manual_seed(0)

    reports = _analyze(config, {"big.weight": torch.randn(3, 3)})

    assert reports["big.weight"].rank_ratio is None
    assert reports["big.weight"].skipped_checks == {"spectral": "min(out_units, fan_in) 3 > svd_max_dim 2"}


def test_single_output_unit_has_zero_duplicate_pairs_and_no_skip():
    """A one-unit weight has no pair to compare, which is a zero count rather than a skipped check."""
    reports = _analyze(WeightXrayConfig(), {"head.weight": torch.randn(1, 5)})

    assert reports["head.weight"].duplicate_units == 0
    assert reports["head.weight"].skipped_checks == {}


def _transposed_types(key: str) -> ModuleTypes:
    """Returns module types declaring one key as a transposed-convolution weight."""
    return ModuleTypes(architecture="unet", by_key={key: "ConvTranspose2d"}, transposed={key})


def test_transposed_convolution_reads_its_output_units_on_dim_1():
    """An (in, out, kH, kW) weight reports out units and an in*kH*kW fan-in with module types, and the reverse without."""
    torch.manual_seed(0)
    weight = torch.randn(8, 4, 2, 2)

    with_types    = WeightAnalyzer(WeightXrayConfig(), _transposed_types("up.weight")).analyze({"up.weight": weight})[0]
    without_types = WeightAnalyzer(WeightXrayConfig(), ModuleTypes.unavailable()).analyze({"up.weight": weight})[0]

    assert (with_types.out_units,    with_types.fan_in,    with_types.unit_axis,    with_types.module_type)    == (4, 32, 1, "ConvTranspose2d")
    assert (without_types.out_units, without_types.fan_in, without_types.unit_axis, without_types.module_type) == (8, 16, 0, None)


def test_dead_output_channel_of_a_transposed_convolution_is_flagged_only_with_module_types():
    """A zeroed output channel of an upsampler is a dead output unit on dim 1 but an ignored dead column on dim 0."""
    torch.manual_seed(0)
    weight       = torch.randn(8, 4, 2, 2)
    weight[:, 0] = 0.0
    state        = {"up.weight": weight}
    config       = WeightXrayConfig()

    with_types    = IssueDetector(config.thresholds).run(WeightAnalyzer(config, _transposed_types("up.weight")).analyze(state), state)[0]
    without_types = IssueDetector(config.thresholds).run(WeightAnalyzer(config, ModuleTypes.unavailable()).analyze(state), state)[0]

    assert with_types.dead_output_units == 1
    assert any(issue.code == "dead_neurons" for issue in with_types.issues)
    assert without_types.dead_output_units == 0
    assert not any(issue.code == "dead_neurons" for issue in without_types.issues)


def test_summary_counts_unchecked_tensors_separately_from_clean_ones():
    """The summary reports how many tensors each size-limited check skipped while their severity stays ok."""
    config = WeightXrayConfig(duplicate_max_units=4, svd_max_dim=2)
    torch.manual_seed(0)

    state   = {"wide.weight": torch.randn(8, 3), "narrow.weight": torch.randn(2, 3), "bias": torch.randn(8)}
    reports = list(_analyze(config, state).values())
    summary = XraySummarizer().build(reports, config.resolved_checkpoint_path)

    assert summary["unchecked"] == {"spectral": 1, "duplicates": 1}
    assert summary["severity_counts"]["ok"] == 3
