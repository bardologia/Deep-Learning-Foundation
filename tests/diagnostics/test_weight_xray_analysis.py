
from __future__ import annotations

import torch

from configuration.diagnostics              import WeightXrayConfig
from tools.diagnostics.weight_xray_analysis import IssueDetector, ModuleTypes, WeightAnalyzer, XraySummarizer


def _analyze(config: WeightXrayConfig, state_dict: dict) -> dict:
    reports = WeightAnalyzer(config, ModuleTypes.unavailable()).analyze(state_dict)
    reports = IssueDetector(config.thresholds).run(reports, state_dict)
    return {report.name: report for report in reports}


def test_duplicate_check_above_the_unit_limit_is_recorded_as_skipped():
    config = WeightXrayConfig(duplicate_max_units=4)
    torch.manual_seed(0)

    reports = _analyze(config, {"wide.weight": torch.randn(8, 3), "narrow.weight": torch.randn(4, 3)})

    assert reports["wide.weight"].duplicate_units is None
    assert reports["wide.weight"].skipped_checks == {"duplicates": "8 output units > duplicate_max_units 4"}
    assert reports["narrow.weight"].duplicate_units == 0
    assert reports["narrow.weight"].skipped_checks == {}


def test_spectral_check_above_the_svd_limit_is_recorded_as_skipped():
    config = WeightXrayConfig(svd_max_dim=2)
    torch.manual_seed(0)

    reports = _analyze(config, {"big.weight": torch.randn(3, 3)})

    assert reports["big.weight"].rank_ratio is None
    assert reports["big.weight"].skipped_checks == {"spectral": "min(out_units, fan_in) 3 > svd_max_dim 2"}


def test_single_output_unit_has_zero_duplicate_pairs_and_no_skip():
    reports = _analyze(WeightXrayConfig(), {"head.weight": torch.randn(1, 5)})

    assert reports["head.weight"].duplicate_units == 0
    assert reports["head.weight"].skipped_checks == {}


def _transposed_types(key: str) -> ModuleTypes:
    return ModuleTypes(architecture="unet", by_key={key: "ConvTranspose2d"}, transposed={key})


def test_transposed_convolution_reads_its_output_units_on_dim_1():
    torch.manual_seed(0)
    weight = torch.randn(8, 4, 2, 2)

    with_types    = WeightAnalyzer(WeightXrayConfig(), _transposed_types("up.weight")).analyze({"up.weight": weight})[0]
    without_types = WeightAnalyzer(WeightXrayConfig(), ModuleTypes.unavailable()).analyze({"up.weight": weight})[0]

    assert (with_types.out_units,    with_types.fan_in,    with_types.unit_axis,    with_types.module_type)    == (4, 32, 1, "ConvTranspose2d")
    assert (without_types.out_units, without_types.fan_in, without_types.unit_axis, without_types.module_type) == (8, 16, 0, None)


def test_dead_output_channel_of_a_transposed_convolution_is_flagged_only_with_module_types():
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
    config = WeightXrayConfig(duplicate_max_units=4, svd_max_dim=2)
    torch.manual_seed(0)

    state   = {"wide.weight": torch.randn(8, 3), "narrow.weight": torch.randn(2, 3), "bias": torch.randn(8)}
    reports = list(_analyze(config, state).values())
    summary = XraySummarizer().build(reports, config.resolved_checkpoint_path)

    assert summary["unchecked"] == {"spectral": 1, "duplicates": 1}
    assert summary["severity_counts"]["ok"] == 3
