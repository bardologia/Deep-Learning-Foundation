"""Tests for attributing backbone predictions to individual input channels.

Covers gradient and gradient-times-input importance per parameter family, channel
occlusion deltas measured in curve space, Spearman rank agreement, and the channel
labelling and grouping derived from the dataset input configuration.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.input_attribution import ChannelGroups, ChannelLabeler, ChannelOcclusion, GradientAttribution
from tools.metrics.scoring                                  import RankCorrelation
from pipelines.backbone.inference.probes            import PredictionCurves


WINDOW = 16


class _ChannelZeroModel(torch.nn.Module):
    """Model whose amplitude and centre depend only on channel 0 and whose width is constant."""
    def forward(self, x):
        """Returns parameters of shape (batch, 3, height, width) derived from channel 0 of x."""
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + 0.5
        out[:, 1] = 3.0 * x[:, 0]
        out[:, 2] = 2.0 + 0.0 * x[:, 0]

        return out


class _TwoChannelModel(torch.nn.Module):
    """Model whose amplitude is the sum of channels 0 and 1, with constant centre and width."""
    def forward(self, x):
        """Returns parameters of shape (batch, 3, height, width) whose amplitude sums channels 0 and 1."""
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + x[:, 1]
        out[:, 1] = 5.0 + 0.0 * x[:, 0]
        out[:, 2] = 2.0 + 0.0 * x[:, 0]

        return out


def _windows(n_channels: int = 3, n_probes: int = 2) -> torch.Tensor:
    """Returns uniform random probe windows of shape (n_probes, n_channels, WINDOW, WINDOW)."""
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.uniform(0.2, 1.0, size=(n_probes, n_channels, WINDOW, WINDOW))).float()


def test_gradient_attribution_concentrates_on_the_used_channel():
    """All amplitude and centre importance lands on the single channel the model reads."""
    importance = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())

    assert importance["amp"]["share"][0] == pytest.approx(1.0)
    assert importance["amp"]["share"][1] == pytest.approx(0.0)
    assert importance["mu"]["share"][0]  == pytest.approx(1.0)
    assert np.allclose(importance["amp"]["share"].sum(), 1.0)
    assert np.allclose(importance["amp"]["share_std"], 0.0)


def test_gradient_attribution_constant_family_is_nan():
    """A parameter family with no input dependence reports NaN importance."""
    importance = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())

    assert np.all(np.isnan(importance["sigma"]["share"]))
    assert np.all(np.isnan(importance["sigma"]["grad_x_input"]))


def test_gradient_attribution_fully_dead_model_raises():
    """A model with zero gradients on every family raises ValueError."""
    class _ConstantModel(torch.nn.Module):
        """Model whose output is identically zero, so every input gradient vanishes."""
        def forward(self, x):
            """Returns zeros of shape (batch, 3, height, width) detached from x in value but not in graph."""
            return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3]) + 0.0 * x.sum()

    with pytest.raises(ValueError):
        GradientAttribution(_ConstantModel(), WINDOW).channel_importance(_windows())


def test_grad_x_input_downweights_quiet_channels():
    """Plain gradient share stays balanced while gradient-times-input follows the channel magnitudes."""
    windows        = _windows()
    windows[:, 1] *= 0.01

    importance = GradientAttribution(_TwoChannelModel(), WINDOW).channel_importance(windows)

    share = importance["amp"]["share"]
    gxi   = importance["amp"]["grad_x_input"]

    assert share[0] == pytest.approx(0.5, abs=0.01)
    assert share[1] == pytest.approx(0.5, abs=0.01)
    assert gxi[0]   > 0.95
    assert gxi[1]   < 0.05


def test_combined_averages_only_live_families():
    """Combining across families skips the all-NaN family and stays finite."""
    importance = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())
    combined   = GradientAttribution.combined(importance, "share")

    assert combined[0] == pytest.approx(1.0)
    assert np.isfinite(combined).all()


def _model_wrapper():
    """Returns a numpy-in, numpy-out callable wrapping the channel-zero model."""
    model = _ChannelZeroModel()
    return lambda images: model(torch.as_tensor(images)).numpy()


def test_occlusion_shifts_only_for_used_channel_sets():
    """Occluding the read channel moves the rendered curves while occluding unused channels does not."""
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, 12))
    loader   = [(torch.from_numpy(np.random.default_rng(1).uniform(0.3, 1.0, size=(2, 3, 8, 8))).float(), None)]

    deltas = ChannelOcclusion(_model_wrapper(), renderer).deltas(loader, max_batches=1, channel_sets=[[0], [1], [2], [1, 2]])

    assert deltas[0] > 1e-6
    assert deltas[1] == pytest.approx(0.0, abs=1e-12)
    assert deltas[2] == pytest.approx(0.0, abs=1e-12)
    assert deltas[3] == pytest.approx(0.0, abs=1e-12)


def test_occlusion_without_batches_raises():
    """Occlusion over an empty loader raises ValueError."""
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, 12))

    with pytest.raises(ValueError):
        ChannelOcclusion(_model_wrapper(), renderer).deltas([], max_batches=0, channel_sets=[[0]])


def test_spearman_detects_perfect_and_inverted_rankings():
    """Spearman correlation is one against itself and minus one against the negated vector."""
    a = np.array([0.1, 0.4, 0.2, 0.9])

    assert RankCorrelation.spearman(a, a)        == pytest.approx(1.0)
    assert RankCorrelation.spearman(a, -a)       == pytest.approx(-1.0)


def test_spearman_needs_matched_vectors():
    """Vectors shorter than the minimum sample count are rejected."""
    with pytest.raises(ValueError):
        RankCorrelation.spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def _labeled_run(use_aux: bool = False) -> SimpleNamespace:
    """Builds a run stub whose input configuration describes a primary, two secondaries and two derived.

    Args:
        use_aux : Whether a Aux channel is appended to the input stack.

    Returns:
        Run stub with the input configuration, the secondary labels and the total channel count.
    """
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
    """Channel labels are emitted in primary, secondary, derived, Aux order."""
    labels = ChannelLabeler.build(_labeled_run(use_aux=True))

    assert labels == ["primary", "sec CH01", "sec CH02", "derived CH01", "derived CH02", "aux"]


def test_channel_labels_mismatch_raises():
    """A channel count that disagrees with the input configuration raises ValueError."""
    run             = _labeled_run()
    run.in_channels = 99

    with pytest.raises(ValueError):
        ChannelLabeler.build(run)


def test_channel_groups_partition_the_stack():
    """Channel groups partition the stack into contiguous index ranges and map back to per-channel names."""
    run    = _labeled_run(use_aux=True)
    groups = ChannelGroups.build(run)

    assert groups == {"primary": [0], "secondaries": [1, 2], "interferograms": [3, 4], "aux": [5]}
    assert ChannelGroups.of_channels(groups, 6) == ["primary", "secondaries", "secondaries", "interferograms", "interferograms", "aux"]


def test_channel_groups_mismatch_raises():
    """Asking for more channels than the groups cover raises ValueError."""
    groups = ChannelGroups.build(_labeled_run())

    with pytest.raises(ValueError):
        ChannelGroups.of_channels(groups, 7)
