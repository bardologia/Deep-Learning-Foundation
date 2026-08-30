"""Tests covering noise, gain, track-drop and shift robustness curves plus track-channel mapping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.input_attribution import TrackChannels
from pipelines.backbone.inference.analysis.robustness        import RobustnessCore
from pipelines.backbone.inference.probes                     import PredictionCurves

from types import SimpleNamespace


H, W, N_Z = 8, 8, 10


def _model(images):
    """Predicts one Gaussian whose amplitude follows the last two input channels.

    Args:
        images : Input stack of shape (batch, 3, H, W).

    Returns:
        Parameter cube of shape (batch, 3, H, W) holding amplitude, a fixed centre
        of 5 metres and a fixed width of 2 metres.
    """
    x      = np.asarray(images, dtype=np.float32)
    params = np.zeros((x.shape[0], 3, H, W), dtype=np.float32)

    params[:, 0] = np.abs(x[:, 1]) + np.abs(x[:, 2]) + 0.5
    params[:, 1] = 5.0
    params[:, 2] = 2.0

    return params


def _batches():
    """Builds one evaluation batch and the curve renderer used to score it.

    Returns:
        Tuple of a single-batch list pairing images of shape (2, 3, H, W) with
        clean curves of shape (2, N_ELEV, H, W), and the renderer that made them.
    """
    rng      = np.random.default_rng(0)
    images   = torch.from_numpy(rng.uniform(0.4, 1.0, size=(2, 3, H, W))).float()
    renderer = PredictionCurves(1, np.linspace(-5.0, 15.0, N_Z))

    gt_curves = renderer.render(_model(images))
    return [(images, gt_curves)], renderer


def test_noise_curve_starts_clean_and_degrades():
    """The clean anchor scores zero error and error grows with the noise sigma."""
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.noise_curve(batches, [0.0, 0.5, 2.0])

    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"] > 0.0
    assert rows[2]["mse"] > rows[1]["mse"]


def test_noise_curve_requires_the_clean_anchor():
    """A sigma list without the clean anchor is rejected."""
    batches, renderer = _batches()

    with pytest.raises(ValueError):
        RobustnessCore(_model, renderer).noise_curve(batches, [0.5, 2.0])


def test_noise_curve_orders_an_out_of_order_sigma_list():
    """Sigmas are sorted before evaluation so the clean anchor comes first."""
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.noise_curve(batches, [0.5, 0.0, 2.0])

    assert [row["sigma"] for row in rows] == [0.0, 0.5, 2.0]
    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)


def test_tolerance_sigma_rejects_rows_that_do_not_start_clean_and_rise():
    """Rows that do not begin at the clean baseline are rejected."""
    with pytest.raises(ValueError, match="clean baseline"):
        RobustnessCore.tolerance_sigma([{"sigma": 0.5, "mse": 1.0}, {"sigma": 0.0, "mse": 1.0}])


def test_noise_on_unused_channels_is_harmless():
    """Noise restricted to channels the model ignores leaves the error at zero."""
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    unused = core.noise_curve(batches, [0.0, 1.0], channels=[0])
    used   = core.noise_curve(batches, [0.0, 1.0], channels=[1, 2])

    assert unused[1]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert used[1]["mse"]   > 0.0


def test_noise_draws_report_spread():
    """Repeated draws report a standard deviation for the noisy rows but not for the clean anchor."""
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.noise_curve(batches, [0.0, 0.5], draws=3)

    assert rows[0]["mse_std"] is None
    assert rows[1]["mse_std"] is not None


def test_gain_curve_always_holds_the_reference():
    """The gain sweep inserts the unit-gain reference, which alone reproduces the clean curves."""
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows  = core.gain_curve(batches, [0.5, 2.0])
    gains = [row["gain"] for row in rows]

    assert gains == [0.5, 1.0, 2.0]
    assert rows[1]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["mse"] > 0.0
    assert rows[2]["mse"] > 0.0


def test_drop_curve_degrades_with_more_tracks():
    """Dropping more tracks raises the error above the zero-drop baseline."""
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.drop_curve(batches, [[1], [2]], draws=2)

    assert rows[0]["dropped"] == 0
    assert rows[0]["mse"]     == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"]     > 0.0
    assert rows[2]["mse"]     > rows[1]["mse"]
    assert rows[2]["dropped"] == 2


def test_drop_curve_evaluates_each_distinct_subset_once():
    """Each distinct drop subset is evaluated once, so the full-drop row carries no spread."""
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


def test_shift_consistency_is_zero_for_a_pointwise_model():
    """A pointwise model is exactly shift-consistent."""
    batches, renderer = _batches()
    core = RobustnessCore(_model, renderer)

    rows = core.shift_consistency(batches, [1, 3])

    assert rows[0]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["mse"] == pytest.approx(0.0, abs=1e-12)


def test_shift_consistency_detects_position_dependence():
    """A model whose output depends on the row index breaks shift consistency."""
    batches, renderer = _batches()

    def positional_model(images):
        params = _model(images)
        params[:, 0] += np.arange(H, dtype=np.float32)[None, :, None]
        return params

    rows = RobustnessCore(positional_model, renderer).shift_consistency(batches, [2])

    assert rows[0]["mse"] > 1e-6


def test_shift_consistency_rejects_out_of_range_shifts():
    """A shift as large as the image extent is rejected."""
    batches, renderer = _batches()

    with pytest.raises(ValueError):
        RobustnessCore(_model, renderer).shift_consistency(batches, [H])


def test_tolerance_sigma_interpolates_and_censors():
    """The tolerance sigma interpolates the threshold crossing, or reports the last sigma as censored."""
    crossing = [
        {"sigma": 0.0, "mse": 1.0, "mse_std": None},
        {"sigma": 1.0, "mse": 1.5, "mse_std": None},
        {"sigma": 2.0, "mse": 3.5, "mse_std": None},
    ]
    flat = [
        {"sigma": 0.0, "mse": 1.0, "mse_std": None},
        {"sigma": 1.0, "mse": 1.2, "mse_std": None},
    ]

    tolerance = RobustnessCore.tolerance_sigma(crossing)

    assert tolerance["sigma"]    == pytest.approx(1.25)
    assert tolerance["censored"] is False

    censored = RobustnessCore.tolerance_sigma(flat)

    assert censored["sigma"]    == pytest.approx(1.0)
    assert censored["censored"] is True


def test_track_channels_map_secondaries_and_derived_channels():
    """Each secondary track maps to its secondary and derived channel indices."""
    input_config = SimpleNamespace(
        use_primary                     = True,
        use_secondaries                 = True,
        use_derived                     = True,
        primary_channels_per_source     = 1,
        secondaries_channels_per_source = 1,
        derived_channels_per_source     = 1,
    )
    run = SimpleNamespace(dataset=SimpleNamespace(input_config=input_config), n_secondaries=2)

    per_track = TrackChannels.build(run)

    assert per_track == [[1, 3], [2, 4]]


def test_track_channels_without_tracks_raise():
    """A configuration with no secondary tracks is rejected."""
    input_config = SimpleNamespace(
        use_primary                     = True,
        use_secondaries                 = False,
        use_derived                     = False,
        primary_channels_per_source     = 1,
        secondaries_channels_per_source = 1,
        derived_channels_per_source     = 1,
    )
    run = SimpleNamespace(dataset=SimpleNamespace(input_config=input_config), n_secondaries=0)

    with pytest.raises(ValueError):
        TrackChannels.build(run)
