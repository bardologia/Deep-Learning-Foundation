
from __future__ import annotations

import numpy as np
import pytest
import torch

from configuration.diagnostics                               import InputAttributionConfig
from pipelines.backbone.inference.analysis.input_attribution import ChannelOcclusion, GradientAttribution, InputAttributionRun
from tools.metrics.scoring                                   import RankCorrelation

from tests.backbone_inference._helpers import StageLogger, build_run_directory
from tests.conftest                    import RecordingLogger


WINDOW  = 16
TARGETS = ["first", "second", "constant"]


class _ChannelZeroModel(torch.nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + 0.5
        out[:, 1] = 3.0 * x[:, 0]
        out[:, 2] = 2.0 + 0.0 * x[:, 0]

        return out


class _TwoChannelModel(torch.nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + x[:, 1]
        out[:, 1] = 5.0 + 0.0 * x[:, 0]
        out[:, 2] = 2.0 + 0.0 * x[:, 0]

        return out


def _windows(n_channels: int = 3, n_probes: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.uniform(0.2, 1.0, size=(n_probes, n_channels, WINDOW, WINDOW))).float()


def test_gradient_attribution_concentrates_on_the_used_channel():
    importance, _kept = GradientAttribution(_ChannelZeroModel(), WINDOW, TARGETS).channel_importance(_windows())

    assert importance["first"]["share"][0] == pytest.approx(1.0)
    assert importance["first"]["share"][1] == pytest.approx(0.0)
    assert importance["second"]["share"][0]  == pytest.approx(1.0)
    assert np.allclose(importance["first"]["share"].sum(), 1.0)
    assert np.allclose(importance["first"]["share_std"], 0.0)


def test_gradient_attribution_of_a_constant_target_is_nan():
    importance, _kept = GradientAttribution(_ChannelZeroModel(), WINDOW, TARGETS).channel_importance(_windows())

    assert np.all(np.isnan(importance["constant"]["share"]))
    assert np.all(np.isnan(importance["constant"]["grad_x_input"]))


def test_gradient_attribution_counts_the_probes_each_target_kept():
    _importance, kept = GradientAttribution(_ChannelZeroModel(), WINDOW, TARGETS).channel_importance(_windows(n_probes=3))

    assert kept["first"]    == {"share": 3, "grad_x_input": 3}
    assert kept["constant"] == {"share": 0, "grad_x_input": 0}


def test_dropped_probes_are_warned_per_target(tmp_path):
    logger = RecordingLogger()
    run    = InputAttributionRun(tmp_path, InputAttributionConfig(), logger)

    run._warn_dropped_probes({"first": {"share": 4, "grad_x_input": 4}, "constant": {"share": 1, "grad_x_input": 1}}, 4)

    assert logger.messages == [("warning", f"{tmp_path.name}: gradient attribution of constant: 3/4 probes had a zero gradient (share over 1, gradient x input over 1 probes)")]


def test_grad_x_input_downweights_quiet_channels():
    windows        = _windows()
    windows[:, 1] *= 0.01

    importance, _kept = GradientAttribution(_TwoChannelModel(), WINDOW, TARGETS).channel_importance(windows)

    share = importance["first"]["share"]
    gxi   = importance["first"]["grad_x_input"]

    assert share[0] == pytest.approx(0.5, abs=0.01)
    assert share[1] == pytest.approx(0.5, abs=0.01)
    assert gxi[0]   > 0.95
    assert gxi[1]   < 0.05


def test_combined_averages_only_live_targets():
    importance, _kept = GradientAttribution(_ChannelZeroModel(), WINDOW, TARGETS).channel_importance(_windows())
    combined          = GradientAttribution.combined(importance, "share")

    assert combined[0] == pytest.approx(1.0)
    assert np.isfinite(combined).all()


def test_occlusion_shifts_only_for_used_channels():
    loader = [(torch.from_numpy(np.random.default_rng(1).uniform(0.3, 1.0, size=(2, 3, 8, 8))).float(), None, None)]

    passes = []
    deltas = ChannelOcclusion(_ChannelZeroModel(), lambda: passes.append(1)).deltas(loader, max_batches=1, n_channels=3)

    assert deltas[0] > 1e-6
    assert deltas[1] == pytest.approx(0.0, abs=1e-12)
    assert deltas[2] == pytest.approx(0.0, abs=1e-12)
    assert len(passes) == 3


def test_spearman_detects_perfect_and_inverted_rankings():
    a = np.array([0.1, 0.4, 0.2, 0.9])

    assert RankCorrelation.spearman(a, a)        == pytest.approx(1.0)
    assert RankCorrelation.spearman(a, -a)       == pytest.approx(-1.0)


@pytest.mark.dataset
@pytest.mark.slow
def test_attribution_pass_tracks_the_occlusion_and_times_both_stages(tmp_path, test_data_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    config  = InputAttributionConfig(window=8, n_row_probes=2, n_col_probes=2, batch_size=2, max_batches=1)
    logger  = StageLogger()

    InputAttributionRun(run_dir, config, logger).run()

    assert len(logger.totals) == 1 and logger.totals[0] == logger.advances
    assert logger.timers[0] == "Input attribution: gradient pass (4 probe windows)"
    assert logger.timers[1] == f"Input attribution: occlusion ({logger.advances} passes)"
