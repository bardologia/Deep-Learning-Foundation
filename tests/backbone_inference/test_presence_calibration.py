"""Tests covering presence-calibration precision bins and risk-coverage diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.backbone.inference.analysis.seed_disagreement import RiskCoverage
from pipelines.backbone.inference.failure_modes              import FailureModes
from pipelines.backbone.inference.presence_calibration       import PresenceCalibration


H, W = 40, 30


def _cube(amp: np.ndarray, mu: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Builds a single-slot parameter cube of shape (3, H, W).

    Args:
        amp   : Amplitude map of shape (H, W).
        mu    : Gaussian centre map of shape (H, W), in metres.
        sigma : Constant Gaussian width, in metres.

    Returns:
        Cube stacking amplitude, centre and width along the first axis.
    """
    cube    = np.zeros((3, H, W), dtype=np.float64)
    cube[0] = amp
    cube[1] = mu
    cube[2] = sigma

    return cube


def _calibration(params_pred: np.ndarray, params_gt: np.ndarray) -> PresenceCalibration:
    """Returns a calibration over Hungarian-matched pairs, mirroring the production pipeline path."""
    return PresenceCalibration(FailureModes(params_pred, params_gt, 1, 1e-4).pair_predictions())


def test_perfect_predictions_have_full_precision():
    """Predictions identical to the ground truth score precision one in every bin."""
    rng = np.random.default_rng(0)
    amp = rng.uniform(0.5, 2.0, size=(H, W))
    mu  = rng.uniform(0.0, 50.0, size=(H, W))

    rows, scalars = _calibration(_cube(amp, mu), _cube(amp, mu)).run()

    assert scalars["presence_overall_precision"] == pytest.approx(1.0)
    assert all(row["precision"] == pytest.approx(1.0) for row in rows)


def test_low_amplitude_predictions_that_miss_reduce_low_bin_precision():
    """Spurious low-amplitude detections depress low-bin precision while the high bin stays perfect."""
    rng = np.random.default_rng(1)
    amp = rng.uniform(0.01, 2.0, size=(H, W))
    mu  = np.full((H, W), 25.0)

    gt_amp = np.where(amp > 1.0, amp, 0.0)

    rows, scalars = _calibration(_cube(amp, mu), _cube(gt_amp, mu)).run()

    assert scalars["presence_precision_low"]  < 0.2
    assert scalars["presence_precision_high"] == pytest.approx(1.0)
    assert scalars["presence_rank_corr"]      > 0.5


def test_too_few_predictions_raise():
    """Too few predicted components to populate the bins raises."""
    amp        = np.zeros((H, W))
    amp[0, :3] = 1.0
    mu         = np.full((H, W), 25.0)

    with pytest.raises(ValueError):
        _calibration(_cube(amp, mu), _cube(amp, mu)).run()


def test_risk_coverage_curve_is_monotone_for_informative_confidence():
    """Disagreement that tracks error yields a rising risk curve and an AURC below the full-coverage risk."""
    rng          = np.random.default_rng(2)
    disagreement = rng.uniform(0.0, 1.0, size=(H, W))
    risk         = disagreement * 5.0 + rng.uniform(0.0, 0.01, size=(H, W))

    rows, scalars = RiskCoverage(disagreement, risk).run()

    risks = [row["risk"] for row in rows]

    assert risks == sorted(risks)
    assert scalars["aurc"] < scalars["full_coverage_risk"]
    assert scalars["risk_at_half_coverage"] < scalars["full_coverage_risk"]
    assert scalars["disagreement_error_spearman"] > 0.99


def test_risk_coverage_shape_mismatch_raises():
    """Disagreement and risk maps of different shapes are rejected."""
    with pytest.raises(ValueError):
        RiskCoverage(np.zeros((4, 4)), np.zeros((5, 4)))


def test_uninformative_confidence_gives_flat_curve():
    """Constant risk makes the AURC equal the full-coverage risk."""
    rng          = np.random.default_rng(3)
    disagreement = rng.uniform(0.0, 1.0, size=(H, W))
    risk         = np.ones((H, W))

    rows, scalars = RiskCoverage(disagreement, risk).run()

    assert scalars["aurc"] == pytest.approx(scalars["full_coverage_risk"])
    assert all(row["risk"] == pytest.approx(1.0) for row in rows)


def test_label_suspects_flag_confident_wrong_pixels():
    """Only pixels combining low disagreement with high error are flagged as label suspects."""
    from pipelines.backbone.inference.analysis.seed_disagreement import LabelSuspects

    disagreement = np.full((H, W), 0.5)
    risk         = np.full((H, W), 1.0)

    disagreement[0, 0] = 0.0
    risk[0, 0]         = 100.0
    risk[0, 1]         = 100.0
    disagreement[0, 1] = 5.0

    mask, scalars = LabelSuspects(disagreement, risk).run()

    assert mask[0, 0] == 1
    assert mask[0, 1] == 0
    assert scalars["label_suspect_n"] == 1.0


def test_label_suspects_report_label_r2_of_flagged_pixels():
    """Flagged pixels report their own label R2 mean alongside the overall map mean."""
    from pipelines.backbone.inference.analysis.seed_disagreement import LabelSuspects

    disagreement       = np.full((H, W), 0.5)
    risk               = np.full((H, W), 1.0)
    label_r2           = np.full((H, W), 0.9)

    disagreement[0, 0] = 0.0
    risk[0, 0]         = 100.0
    label_r2[0, 0]     = 0.1

    _mask, scalars = LabelSuspects(disagreement, risk, label_r2).run()

    assert scalars["label_suspect_label_r2_mean"]    == pytest.approx(0.1)
    assert scalars["label_suspect_label_r2_overall"] >  0.85
