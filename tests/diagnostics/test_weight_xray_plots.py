from __future__ import annotations

import pytest
import torch

from configuration.diagnostics              import WeightXrayConfig
from tools.diagnostics.weight_xray_analysis import IssueDetector, WeightAnalyzer
from tools.diagnostics.weight_xray_plots    import WeightXrayPlots


@pytest.fixture
def state_dict():
    torch.manual_seed(0)

    dead       = torch.zeros(16, 8)
    dead[0, 0] = 1.0

    return {
        "encoder.blocks.0.conv.weight" : torch.randn(16, 8) * 0.05,
        "encoder.blocks.0.conv.bias"   : torch.randn(16) * 0.01,
        "decoder.head.weight"          : dead,
    }


def test_xray_plots_render_series_histogram_and_flagged_layers(tmp_path, state_dict):
    config  = WeightXrayConfig(output_dir=tmp_path)
    reports = WeightAnalyzer(config).analyze(state_dict)
    reports = IssueDetector(config.thresholds).run(reports, state_dict)

    paths = WeightXrayPlots(config).render(reports, state_dict)
    names = {path.name for path in paths}

    assert {"layer_l2_norm.png", "layer_std.png", "layer_sparsity.png", "layer_rank_ratio.png", "weight_histogram.png"} <= names
    assert all(path.is_file() for path in paths)
    assert any(path.parent.name == "layer_histograms" for path in paths)


def test_long_dotted_titles_wrap_at_name_boundaries(tmp_path):
    plots = WeightXrayPlots(WeightXrayConfig(output_dir=tmp_path))
    title = "encoder.encoder_blocks.3.residual_convs.1.conv_block.block.0.weight (warning)"

    wrapped = plots._wrap_title(title)

    assert "\n" in wrapped
    assert wrapped.replace("\n", "") == title
    assert all(len(line) <= plots.TITLE_WRAP + 1 for line in wrapped.split("\n"))


def test_short_titles_stay_unwrapped(tmp_path):
    plots = WeightXrayPlots(WeightXrayConfig(output_dir=tmp_path))

    assert plots._wrap_title("Pooled weight distribution") == "Pooled weight distribution"
