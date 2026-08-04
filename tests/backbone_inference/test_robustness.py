from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.input_attribution import TrackChannels
from pipelines.backbone.inference.analysis.robustness        import RobustnessCore
from pipelines.backbone.inference.probes                     import PredictionCurves

from types import SimpleNamespace


H, W, N_ELEV = 8, 8, 10


def _model(images):
    x      = np.asarray(images, dtype=np.float32)
    params = np.zeros((x.shape[0], 3, H, W), dtype=np.float32)

    params[:, 0] = np.abs(x[:, 1]) + np.abs(x[:, 2]) + 0.5
    params[:, 1] = 5.0
    params[:, 2] = 2.0

    return params


def _batches():
    rng      = np.random.default_rng(0)
    images   = torch.from_numpy(rng.uniform(0.4, 1.0, size=(2, 3, H, W))).float()
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, N_ELEV))

    gt_curves = renderer.render(_model(images))
    return [(images, gt_curves)], renderer


def test_noise_curve_starts_clean_and_degrades():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.noise_curve(batches, [0.0, 0.5, 2.0])

    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"] > 0.0
    assert rows[2]["mse"] > rows[1]["mse"]


def test_drop_curve_degrades_with_more_tracks():
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.drop_curve(batches, [[1], [2]], draws=2)

    assert rows[0]["dropped"] == 0
    assert rows[0]["mse"]     == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"]     > 0.0
    assert rows[2]["mse"]     > rows[1]["mse"]
    assert rows[2]["dropped"] == 2


def test_drop_curve_evaluates_each_distinct_subset_once():
    batches, renderer = _batches()
    calls             = []

    def counting_model(images):
        calls.append(images)
        return _model(images)

    core = RobustnessCore(counting_model, renderer)
    rows = core.drop_curve(batches, [[1], [2]], draws=8)

    assert len(calls) == 4

    assert rows[1]["mse_std"] is not None
    assert rows[2]["mse_std"] is None


def test_track_channels_map_secondaries_and_interferograms():
    input_config = SimpleNamespace(
        use_primary                      = True,
        use_secondaries                  = True,
        use_interferograms               = True,
        primary_channels_per_pass        = 1,
        secondaries_channels_per_pass    = 1,
        interferograms_channels_per_pass = 1,
    )
    run = SimpleNamespace(dataset=SimpleNamespace(input_config=input_config), n_secondaries=2)

    per_track = TrackChannels.build(run)

    assert per_track == [[1, 3], [2, 4]]


def test_track_channels_without_tracks_raise():
    input_config = SimpleNamespace(
        use_primary                      = True,
        use_secondaries                  = False,
        use_interferograms               = False,
        primary_channels_per_pass        = 1,
        secondaries_channels_per_pass    = 1,
        interferograms_channels_per_pass = 1,
    )
    run = SimpleNamespace(dataset=SimpleNamespace(input_config=input_config), n_secondaries=0)

    with pytest.raises(ValueError):
        TrackChannels.build(run)
