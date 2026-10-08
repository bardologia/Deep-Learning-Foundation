"""Tests for the per-pixel failure-mode classification of predicted Gaussian mixtures.

Checks that `FailureModes` labels each pixel as ok, missed, hallucinated, position,
width or amplitude, and that the miss rate rises for closely spaced components.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.failure_modes      import FailureModes
from pipelines.backbone.inference.pipeline           import InferencePipeline
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata

from tests.conftest import RecordingLogger


H, W = 10, 8


def _params(slots: list[tuple[float, float, float]], H_: int = H, W_: int = W) -> np.ndarray:
    """Builds a constant parameter cube from a list of (amplitude, mu, sigma) slots.

    Args:
        slots : One (amplitude, mu in metres, sigma in metres) triple per Gaussian slot.
        H_    : Row extent of the cube.
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

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2, 1e-4).run()

    assert np.all(mode_map == 0)
    assert scalars["failure_miss_rate"]      == pytest.approx(0.0)
    assert scalars["failure_halluc_rate"]    == pytest.approx(0.0)
    assert scalars["failure_frac_ok"]        == pytest.approx(1.0)


def test_missing_component_is_flagged():
    """Dropping one of two ground-truth components is labelled \"missed\" with a 0.5 miss rate."""
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = _params([(1.0, 10.0, 3.0), (0.0, 0.0, 0.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2, 1e-4).run()

    assert _mode_name(mode_map) == "missed"
    assert scalars["failure_miss_rate"] == pytest.approx(0.5)


def test_hallucinated_component_is_flagged():
    """Predicting a component absent from the ground truth is labelled \"hallucinated\"."""
    gt   = _params([(1.0, 10.0, 3.0), (0.0, 0.0, 0.0)])
    pred = _params([(1.0, 10.0, 3.0), (2.0, 60.0, 4.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 2, 1e-4).run()

    assert _mode_name(mode_map) == "hallucinated"
    assert scalars["failure_halluc_rate"] == pytest.approx(0.5)


def test_position_shift_within_tolerance_is_flagged_as_position():
    """A matched component displaced by 3 m is labelled \"position\"."""
    gt   = _params([(1.0, 10.0, 3.0)])
    pred = _params([(1.0, 13.0, 3.0)])

    mode_map, _by_sep, _scalars = FailureModes(pred, gt, 1, 1e-4).run()

    assert _mode_name(mode_map) == "position"


def test_width_and_amplitude_errors_are_flagged():
    """A doubled sigma is labelled \"width\" and an inflated amplitude is labelled \"amplitude\"."""
    gt         = _params([(1.0, 10.0, 3.0)])
    pred_width = _params([(1.0, 10.0, 6.0)])
    pred_amp   = _params([(2.5, 10.0, 3.0)])

    width_map, _s1, _x1 = FailureModes(pred_width, gt, 1, 1e-4).run()
    amp_map,   _s2, _x2 = FailureModes(pred_amp, gt, 1, 1e-4).run()

    assert _mode_name(width_map) == "width"
    assert _mode_name(amp_map)   == "amplitude"


def test_missed_takes_priority_over_matched_errors():
    """A missed component outranks a position error on the same pixel."""
    gt   = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])
    pred = _params([(1.0, 13.0, 3.0), (0.0, 0.0, 0.0)])

    mode_map, _by_sep, _scalars = FailureModes(pred, gt, 2, 1e-4).run()

    assert _mode_name(mode_map) == "missed"


def test_far_prediction_counts_as_miss_plus_hallucination():
    """A component predicted 40 m away counts as both a miss and a hallucination."""
    gt   = _params([(1.0, 10.0, 3.0)])
    pred = _params([(1.0, 50.0, 3.0)])

    mode_map, _by_sep, scalars = FailureModes(pred, gt, 1, 1e-4).run()

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

    _mode_map, by_sep, _scalars = FailureModes(pred, gt, 2, 1e-4).run()

    assert by_sep
    close = [row for row in by_sep if row["center"] < 20.0]
    far   = [row for row in by_sep if row["center"] > 20.0]
    assert close and far
    assert min(row["miss_rate"] for row in close) > max(row["miss_rate"] for row in far)


def test_miss_by_separation_status_reports_the_computed_bins():
    """A profile that binned the separations records how many bins it computed over how many two-component pixels."""
    H_, W_ = 40, 30
    gt     = np.zeros((6, H_, W_))

    gt[0] = 1.0
    gt[1] = 10.0
    gt[2] = 3.0
    gt[3] = 1.0
    gt[5] = 3.0
    gt[4] = np.linspace(14.0, 60.0, H_)[:, None]

    _mode_map, by_sep, scalars = FailureModes(gt.copy(), gt, 2, 1e-4).run()

    assert by_sep
    assert scalars["failure_sep_status"] == f"computed: {len(by_sep)} bins over {H_ * W_} two-component pixels, 0 sparse bin(s) under 30 pixels dropped"


def test_miss_by_separation_status_names_why_the_profile_was_skipped():
    """Too few two-component pixels, or separations without spread, leave the rows empty with a status naming the reason."""
    single = _params([(1.0, 10.0, 3.0)])
    paired = _params([(1.0, 10.0, 3.0), (2.0, 40.0, 4.0)])

    _m1, by_sep_single, scalars_single = FailureModes(single.copy(), single, 1, 1e-4).run()
    _m2, by_sep_paired, scalars_paired = FailureModes(paired.copy(), paired, 2, 1e-4).run()

    assert by_sep_single == [] and by_sep_paired == []
    assert scalars_single["failure_sep_status"] == "skipped: only 0 pixels carry two active GT components, need 30"
    assert scalars_paired["failure_sep_status"] == f"skipped: the separations of the {H * W} two-component pixels take 1 distinct quantile value(s), too few to bin"


def test_pipeline_stage_records_and_logs_why_the_separation_profile_is_missing(tmp_path):
    """The failure-mode stage stores failure_sep_status in metrics.json and logs it, so a missing section 5.8 has a stated reason."""
    gt = _params([(1.0, 10.0, 3.0)])

    config = InferenceConfig(run_directory=tmp_path, output_subdir="failure", device="cpu", save_cubes=False)
    meta   = InferenceMetadata(config)
    meta.create_dirs()

    result  = SimpleNamespace(params_pred=gt.copy(), params_gt=gt, assignment=lambda n_gaussians, threshold: None)
    run     = SimpleNamespace(n_gaussians=1, activity_threshold=1e-4)
    logger  = RecordingLogger()
    metrics = {}

    InferencePipeline(config)._classify_failures(config, meta, run, result, metrics, logger)

    assert result.miss_by_sep == []
    assert metrics["failure_sep_status"] == "skipped: only 0 pixels carry two active GT components, need 30"
    assert ("subsection", "Miss rate by GT separation : skipped: only 0 pixels carry two active GT components, need 30") in logger.messages
