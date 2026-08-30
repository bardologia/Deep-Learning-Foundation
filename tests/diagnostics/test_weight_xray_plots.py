"""Tests for the weight x-ray plot renderer.

Covers the per-metric layer series and histogram outputs, dotted-title wrapping
at name boundaries, and the budgeted pooled sample drawn across every tensor.
"""

from __future__ import annotations

import pytest
import torch

from configuration.diagnostics              import WeightXrayConfig
from tools.diagnostics.weight_xray_analysis import IssueDetector, WeightAnalyzer
from tools.diagnostics.weight_xray_plots    import WeightXrayPlots


@pytest.fixture
def state_dict():
    """Returns a small state dict with a random convolution layer, its bias and an almost entirely dead head."""
    torch.manual_seed(0)

    dead       = torch.zeros(16, 8)
    dead[0, 0] = 1.0

    return {
        "encoder.blocks.0.conv.weight" : torch.randn(16, 8) * 0.05,
        "encoder.blocks.0.conv.bias"   : torch.randn(16) * 0.01,
        "decoder.head.weight"          : dead,
    }


def test_xray_plots_render_series_histogram_and_flagged_layers(tmp_path, state_dict):
    """Rendering writes the layer metric series, the pooled histogram and per-layer histograms to disk."""
    config  = WeightXrayConfig(output_dir=tmp_path)
    reports = WeightAnalyzer(config).analyze(state_dict)
    reports = IssueDetector(config.thresholds).run(reports, state_dict)

    paths = WeightXrayPlots(config).render(reports, state_dict)
    names = {path.name for path in paths}

    assert {"layer_l2_norm.png", "layer_std.png", "layer_sparsity.png", "layer_rank_ratio.png", "weight_histogram.png"} <= names
    assert all(path.is_file() for path in paths)
    assert any(path.parent.name == "layer_histograms" for path in paths)


def test_long_dotted_titles_wrap_at_name_boundaries(tmp_path):
    """A long dotted parameter name wraps into lines within the width limit without losing characters."""
    plots = WeightXrayPlots(WeightXrayConfig(output_dir=tmp_path))
    title = "encoder.encoder_blocks.3.residual_convs.1.conv_block.block.0.weight (warning)"

    wrapped = plots._wrap_title(title)

    assert "\n" in wrapped
    assert wrapped.replace("\n", "") == title
    assert all(len(line) <= plots.TITLE_WRAP + 1 for line in wrapped.split("\n"))


def test_short_titles_stay_unwrapped(tmp_path):
    """A title shorter than the wrap width is returned unchanged."""
    plots = WeightXrayPlots(WeightXrayConfig(output_dir=tmp_path))

    assert plots._wrap_title("Pooled weight distribution") == "Pooled weight distribution"


def test_global_sample_respects_the_budget_and_covers_every_tensor(tmp_path):
    """The pooled histogram sample stays within budget and draws from each tensor in proportion to its size."""
    state = {
        "a.weight" : torch.full((20, 20), 1.0),
        "b.weight" : torch.full((30, 20), 2.0),
    }

    config  = WeightXrayConfig(output_dir=tmp_path, histogram_sample_max=100)
    reports = WeightAnalyzer(config).analyze(state)
    pooled  = WeightXrayPlots(config)._global_sample(reports, state)

    assert pooled.size <= 100
    assert (pooled == 1.0).sum() == pytest.approx(40, abs=1)
    assert (pooled == 2.0).sum() == pytest.approx(60, abs=1)
