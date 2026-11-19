
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configuration.diagnostics                               import InputAttributionConfig
from pipelines.backbone.inference.analysis.input_attribution import ChannelGroups, ChannelLabeler, ChannelOcclusion, GradientAttribution, InputAttributionRun
from pipelines.backbone.inference.probes                     import PredictionCurves
from tools.metrics.scoring                                   import RankCorrelation

from tests.backbone_inference._helpers import StageLogger, build_run_directory
from tests.conftest                    import RecordingLogger


WINDOW = 16


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
    importance, _kept = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())

    assert importance["amp"]["share"][0] == pytest.approx(1.0)
    assert importance["amp"]["share"][1] == pytest.approx(0.0)
    assert importance["mu"]["share"][0]  == pytest.approx(1.0)
    assert np.allclose(importance["amp"]["share"].sum(), 1.0)
    assert np.allclose(importance["amp"]["share_std"], 0.0)


def test_gradient_attribution_constant_family_is_nan():
    importance, _kept = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())

    assert np.all(np.isnan(importance["sigma"]["share"]))
    assert np.all(np.isnan(importance["sigma"]["grad_x_input"]))


def test_gradient_attribution_counts_the_probes_each_family_kept():
    _importance, kept = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows(n_probes=3))

    assert kept["amp"] == {"share": 3, "grad_x_input": 3}
    assert kept["sigma"] == {"share": 0, "grad_x_input": 0}


def test_dropped_probes_are_warned_per_family(tmp_path):
    logger = RecordingLogger()
    run    = InputAttributionRun(tmp_path, InputAttributionConfig(), logger)

    run._warn_dropped_probes({"amp": {"share": 4, "grad_x_input": 4}, "sigma": {"share": 1, "grad_x_input": 1}}, 4)

    assert logger.messages == [("warning", f"{tmp_path.name}: gradient attribution of sigma: 3/4 probes had a zero gradient (share over 1, gradient x input over 1 probes)")]


def test_grad_x_input_downweights_quiet_channels():
    windows        = _windows()
    windows[:, 1] *= 0.01

    importance, _kept = GradientAttribution(_TwoChannelModel(), WINDOW).channel_importance(windows)

    share = importance["amp"]["share"]
    gxi   = importance["amp"]["grad_x_input"]

    assert share[0] == pytest.approx(0.5, abs=0.01)
    assert share[1] == pytest.approx(0.5, abs=0.01)
    assert gxi[0]   > 0.95
    assert gxi[1]   < 0.05


def test_combined_averages_only_live_families():
    importance, _kept = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())
    combined          = GradientAttribution.combined(importance, "share")

    assert combined[0] == pytest.approx(1.0)
    assert np.isfinite(combined).all()


def _model_wrapper():
    model = _ChannelZeroModel()
    return lambda images: model(torch.as_tensor(images)).numpy()


def test_occlusion_shifts_only_for_used_channel_sets():
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, 12))
    loader   = [(torch.from_numpy(np.random.default_rng(1).uniform(0.3, 1.0, size=(2, 3, 8, 8))).float(), None)]

    passes = []
    deltas = ChannelOcclusion(_model_wrapper(), renderer, lambda: passes.append(1)).deltas(loader, max_batches=1, channel_sets=[[0], [1], [2], [1, 2]])

    assert deltas[0] > 1e-6
    assert deltas[1] == pytest.approx(0.0, abs=1e-12)
    assert deltas[2] == pytest.approx(0.0, abs=1e-12)
    assert deltas[3] == pytest.approx(0.0, abs=1e-12)
    assert len(passes) == 4


def test_spearman_detects_perfect_and_inverted_rankings():
    a = np.array([0.1, 0.4, 0.2, 0.9])

    assert RankCorrelation.spearman(a, a)        == pytest.approx(1.0)
    assert RankCorrelation.spearman(a, -a)       == pytest.approx(-1.0)


def _labeled_run(use_aux: bool = False) -> SimpleNamespace:
    input_config = SimpleNamespace(
        use_primary                     = True,
        use_secondaries                 = True,
        use_derived                     = True,
        use_aux                         = use_aux,
        primary_channels_per_source     = 1,
        secondaries_channels_per_source = 1,
        derived_channels_per_source     = 1,
    )

    n_channels = 1 + 2 + 2 + (1 if use_aux else 0)

    return SimpleNamespace(
        dataset          = SimpleNamespace(input_config=input_config),
        secondary_labels = ["CH01", "CH02"],
        n_secondaries    = 2,
        in_channels      = n_channels,
    )


def test_channel_labels_follow_the_input_stack_order():
    labels = ChannelLabeler.build(_labeled_run(use_aux=True))

    assert labels == ["primary", "sec CH01", "sec CH02", "derived CH01", "derived CH02", "aux"]


def test_channel_groups_partition_the_stack():
    run    = _labeled_run(use_aux=True)
    groups = ChannelGroups.build(run)

    assert groups == {"primary": [0], "secondaries": [1, 2], "derived": [3, 4], "aux": [5]}
    assert ChannelGroups.of_channels(groups, 6) == ["primary", "secondaries", "secondaries", "derived", "derived", "aux"]


@pytest.mark.dataset
@pytest.mark.slow
def test_attribution_pass_tracks_the_occlusion_and_times_both_stages(tmp_path, test_data_dir, params_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    config  = InputAttributionConfig(window=32, n_row_probes=2, n_col_probes=2, batch_size=2, max_batches=1)
    logger  = StageLogger()

    InputAttributionRun(run_dir, config, logger).run()

    assert len(logger.totals) == 1 and logger.totals[0] == logger.advances
    assert logger.timers[0] == "Input attribution: gradient pass (4 probe windows)"
    assert logger.timers[1] == f"Input attribution: occlusion ({logger.advances} passes)"
