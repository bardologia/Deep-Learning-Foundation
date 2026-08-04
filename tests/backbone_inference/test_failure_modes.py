from __future__ import annotations

import numpy as np
import pytest

from pipelines.backbone.inference.failure_modes import FailureModes


H, W = 10, 8


def _params(slots: list[tuple[float, float, float]], H_: int = H, W_: int = W) -> np.ndarray:
    cube = np.zeros((len(slots) * 3, H_, W_), dtype=np.float64)

    for k, (amp, mu, sigma) in enumerate(slots):
        cube[3 * k]     = amp
        cube[3 * k + 1] = mu
        cube[3 * k + 2] = sigma

    return cube


def _mode_name(mode_map: np.ndarray, az: int = 0, rg: int = 0) -> str:
    return FailureModes.MODES[int(mode_map[az, rg])]


def test_perfect_prediction_is_ok_everywhere():
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = gt.copy()

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2).run()

    assert np.all(mode_map == 0)
    assert scalars["failure_miss_rate"]      == pytest.approx(0.0)
    assert scalars["failure_halluc_rate"]    == pytest.approx(0.0)
    assert scalars["failure_frac_ok"]        == pytest.approx(1.0)


def test_missing_scatterer_is_flagged():
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = _params([(1.0, 10.0, 3.0), (0.0, 0.0, 0.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2).run()

    assert _mode_name(mode_map) == "missed"
    assert scalars["failure_miss_rate"] == pytest.approx(0.5)


def test_hallucinated_scatterer_is_flagged():
    gt   = _params([(1.0, 10.0, 3.0), (0.0, 0.0, 0.0)])
    pred = _params([(1.0, 10.0, 3.0), (2.0, 60.0, 4.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2).run()

    assert _mode_name(mode_map) == "hallucinated"
    assert scalars["failure_halluc_rate"] == pytest.approx(0.5)


def test_position_shift_within_tolerance_is_flagged_as_position():
    gt   = _params([(1.0, 10.0, 3.0)])
    pred = _params([(1.0, 13.0, 3.0)])

    mode_map, _by_sep, _scalars = FailureModes(pred, gt, 1).run()

    assert _mode_name(mode_map) == "position"


def test_width_and_amplitude_errors_are_flagged():
    gt         = _params([(1.0, 10.0, 3.0)])
    pred_width = _params([(1.0, 10.0, 6.0)])
    pred_amp   = _params([(2.5, 10.0, 3.0)])

    width_map, _s1, _x1 = FailureModes(pred_width, gt, 1).run()
    amp_map,   _s2, _x2 = FailureModes(pred_amp, gt, 1).run()

    assert _mode_name(width_map) == "width"
    assert _mode_name(amp_map)   == "amplitude"


def test_missed_takes_priority_over_matched_errors():
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = _params([(1.0, 13.0, 3.0), (0.0, 0.0, 0.0)])

    mode_map, _by_sep, _scalars = FailureModes(pred, gt, 2).run()

    assert _mode_name(mode_map) == "missed"


def test_far_prediction_counts_as_miss_plus_hallucination():
    gt   = _params([(1.0, 10.0, 3.0)])
    pred = _params([(1.0, 50.0, 3.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 1).run()

    assert _mode_name(mode_map) == "missed"
    assert scalars["failure_miss_rate"]   == pytest.approx(1.0)
    assert scalars["failure_halluc_rate"] == pytest.approx(1.0)


def test_miss_by_separation_rises_for_close_scatterers():
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
