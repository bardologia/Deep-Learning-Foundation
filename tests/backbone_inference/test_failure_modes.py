"""Tests for the per-pixel failure-mode classification of predicted Gaussian mixtures.

Checks that `FailureModes` labels each pixel as ok, missed, hallucinated, position,
width or amplitude, and that the miss rate rises for closely spaced scatterers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.backbone.inference.failure_modes import FailureModes


H, W = 10, 8


def _params(slots: list[tuple[float, float, float]], H_: int = H, W_: int = W) -> np.ndarray:
    """Builds a constant parameter cube from a list of (amplitude, mu, sigma) slots.

    Args:
        slots : One (amplitude, mu in metres, sigma in metres) triple per Gaussian slot.
        H_    : Azimuth extent of the cube.
        W_    : Range extent of the cube.

    Returns:
        Parameter cube of shape (3 * n_slots, H_, W_) with every pixel carrying the same slots.
    """
    cube = np.zeros((len(slots) * 3, H_, W_), dtype=np.float64)

    for k, (amp, mu, sigma) in enumerate(slots):
        cube[3 * k]     = amp
        cube[3 * k + 1] = mu
        cube[3 * k + 2] = sigma

    return cube


def _mode_name(mode_map: np.ndarray, row: int = 0, col: int = 0) -> str:
    """Returns the failure-mode label at one pixel of the mode map."""
    return FailureModes.MODES[int(mode_map[row, col])]


def test_perfect_prediction_is_ok_everywhere():
    """An exact prediction yields mode 0 everywhere and zero miss and hallucination rates."""
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = gt.copy()

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2).run()

    assert np.all(mode_map == 0)
    assert scalars["failure_miss_rate"]      == pytest.approx(0.0)
    assert scalars["failure_halluc_rate"]    == pytest.approx(0.0)
    assert scalars["failure_frac_ok"]        == pytest.approx(1.0)


def test_missing_component_is_flagged():
    """Dropping one of two ground-truth components is labelled \"missed\" with a 0.5 miss rate."""
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = _params([(1.0, 10.0, 3.0), (0.0, 0.0, 0.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2).run()

    assert _mode_name(mode_map) == "missed"
    assert scalars["failure_miss_rate"] == pytest.approx(0.5)


def test_hallucinated_component_is_flagged():
    """Predicting a component absent from the ground truth is labelled \"hallucinated\"."""
    gt   = _params([(1.0, 10.0, 3.0), (0.0, 0.0, 0.0)])
    pred = _params([(1.0, 10.0, 3.0), (2.0, 60.0, 4.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2).run()

    assert _mode_name(mode_map) == "hallucinated"
    assert scalars["failure_halluc_rate"] == pytest.approx(0.5)


def test_position_shift_within_tolerance_is_flagged_as_position():
    """A matched scatterer displaced by 3 m is labelled \"position\"."""
    gt   = _params([(1.0, 10.0, 3.0)])
    pred = _params([(1.0, 13.0, 3.0)])

    mode_map, _by_sep, _scalars = FailureModes(pred, gt, 1).run()

    assert _mode_name(mode_map) == "position"


def test_width_and_amplitude_errors_are_flagged():
    """A doubled sigma is labelled \"width\" and an inflated amplitude is labelled \"amplitude\"."""
    gt         = _params([(1.0, 10.0, 3.0)])
    pred_width = _params([(1.0, 10.0, 6.0)])
    pred_amp   = _params([(2.5, 10.0, 3.0)])

    width_map, _s1, _x1 = FailureModes(pred_width, gt, 1).run()
    amp_map,   _s2, _x2 = FailureModes(pred_amp, gt, 1).run()

    assert _mode_name(width_map) == "width"
    assert _mode_name(amp_map)   == "amplitude"


def test_missed_takes_priority_over_matched_errors():
    """A missed scatterer outranks a position error on the same pixel."""
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = _params([(1.0, 13.0, 3.0), (0.0, 0.0, 0.0)])

    mode_map, _by_sep, _scalars = FailureModes(pred, gt, 2).run()

    assert _mode_name(mode_map) == "missed"


def test_far_prediction_counts_as_miss_plus_hallucination():
    """A scatterer predicted 40 m away counts as both a miss and a hallucination."""
    gt   = _params([(1.0, 10.0, 3.0)])
    pred = _params([(1.0, 50.0, 3.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 1).run()

    assert _mode_name(mode_map) == "missed"
    assert scalars["failure_miss_rate"]   == pytest.approx(1.0)
    assert scalars["failure_halluc_rate"] == pytest.approx(1.0)


def test_miss_by_separation_rises_for_close_components():
    """The miss rate binned by component separation is higher for close pairs than for far pairs."""
    H_, W_ = 40, 30
    gt   = np.zeros((6, H_, W_))
    pred = np.zeros((6, H_, W_))

    gt[0] = 1.0
    gt[1] = 10.0
    gt[2] = 3.0
    gt[3] = 1.0
    gt[5] = 3.0

    gt[4, : H_ // 2] = 14.0
    gt[4, H_ // 2 :] = 40.0

    pred[:3]             = gt[:3]
    pred[3:, H_ // 2 :]  = gt[3:, H_ // 2 :]

    _mode_map, by_sep, _scalars = FailureModes(pred, gt, 2).run()

    assert by_sep
    close = [row for row in by_sep if row["center"] < 20.0]
    far   = [row for row in by_sep if row["center"] > 20.0]
    assert close and far
    assert min(row["miss_rate"] for row in close) > max(row["miss_rate"] for row in far)
