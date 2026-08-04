from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.input_attribution import ChannelLabeler, ChannelOcclusion, GradientAttribution
from pipelines.backbone.inference.probes            import PredictionCurves


WINDOW = 16


class _ChannelZeroModel(torch.nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        out        = torch.zeros(B, 3, H, W)

        out[:, 0] = x[:, 0] + 0.5
        out[:, 1] = 3.0 * x[:, 0]
        out[:, 2] = 2.0 + 0.0 * x[:, 0]

        return out


def _windows(n_channels: int = 3, n_probes: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.uniform(0.2, 1.0, size=(n_probes, n_channels, WINDOW, WINDOW))).float()


def test_gradient_attribution_concentrates_on_the_used_channel():
    importance = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())

    assert importance["amp"][0] == pytest.approx(1.0)
    assert importance["amp"][1] == pytest.approx(0.0)
    assert importance["mu"][0]  == pytest.approx(1.0)
    assert np.allclose(sum(importance["amp"]), 1.0)


def test_gradient_attribution_constant_family_is_nan():
    importance = GradientAttribution(_ChannelZeroModel(), WINDOW).channel_importance(_windows())

    assert np.all(np.isnan(importance["sigma"]))


def test_gradient_attribution_fully_dead_model_raises():
    class _ConstantModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3]) + 0.0 * x.sum()

    with pytest.raises(ValueError):
        GradientAttribution(_ConstantModel(), WINDOW).channel_importance(_windows())


def _model_wrapper():
    model = _ChannelZeroModel()
    return lambda images: model(torch.as_tensor(images)).numpy()


def test_occlusion_shifts_only_for_used_channels():
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, 12))
    loader   = [(torch.from_numpy(np.random.default_rng(1).uniform(0.3, 1.0, size=(2, 3, 8, 8))).float(), None)]

    deltas = ChannelOcclusion(_model_wrapper(), renderer).deltas(loader, max_batches=1, n_channels=3)

    assert deltas[0] > 1e-6
    assert deltas[1] == pytest.approx(0.0, abs=1e-12)
    assert deltas[2] == pytest.approx(0.0, abs=1e-12)


def test_occlusion_without_batches_raises():
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, 12))

    with pytest.raises(ValueError):
        ChannelOcclusion(_model_wrapper(), renderer).deltas([], max_batches=0, n_channels=3)


def _labeled_run(use_dem: bool = False) -> SimpleNamespace:
    input_config = SimpleNamespace(
        use_primary                      = True,
        use_secondaries                  = True,
        use_interferograms               = True,
        use_dem                          = use_dem,
        primary_channels_per_pass        = 1,
        secondaries_channels_per_pass    = 1,
        interferograms_channels_per_pass = 1,
    )

    n_channels = 1 + 2 + 2 + (1 if use_dem else 0)

    return SimpleNamespace(
        dataset          = SimpleNamespace(input_config=input_config),
        secondary_labels = ["PS04", "PS06"],
        n_secondaries    = 2,
        in_channels      = n_channels,
    )


def test_channel_labels_follow_the_input_stack_order():
    labels = ChannelLabeler.build(_labeled_run(use_dem=True))

    assert labels == ["primary", "sec PS04", "sec PS06", "ifg PS04", "ifg PS06", "dem"]


def test_channel_labels_mismatch_raises():
    run             = _labeled_run()
    run.in_channels = 99

    with pytest.raises(ValueError):
        ChannelLabeler.build(run)
