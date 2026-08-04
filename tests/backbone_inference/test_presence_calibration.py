from __future__ import annotations

import numpy as np
import pytest

from pipelines.backbone.inference.analysis.seed_disagreement import RiskCoverage
from pipelines.backbone.inference.presence_calibration       import PresenceCalibration


H, W = 40, 30


def _cube(amp: np.ndarray, mu: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    cube    = np.zeros((3, H, W), dtype=np.float64)
    cube[0] = amp
    cube[1] = mu
    cube[2] = sigma

    return cube


def test_perfect_predictions_have_full_precision():
    rng = np.random.default_rng(0)
    amp = rng.uniform(0.5, 2.0, size=(H, W))
    mu  = rng.uniform(0.0, 50.0, size=(H, W))

    rows, scalars = PresenceCalibration(_cube(amp, mu), _cube(amp, mu), 1).run()

    assert scalars["presence_overall_precision"] == pytest.approx(1.0)
    assert all(row["precision"] == pytest.approx(1.0) for row in rows)


def test_low_amplitude_predictions_that_miss_reduce_low_bin_precision():
    rng = np.random.default_rng(1)
    amp = rng.uniform(0.01, 2.0, size=(H, W))
    mu  = np.full((H, W), 25.0)

    gt_amp = np.where(amp > 1.0, amp, 0.0)

    rows, scalars = PresenceCalibration(_cube(amp, mu), _cube(gt_amp, mu), 1).run()

    assert scalars["presence_precision_low"]  < 0.2
    assert scalars["presence_precision_high"] == pytest.approx(1.0)
    assert scalars["presence_rank_corr"]      > 0.5


def test_too_few_predictions_raise():
    amp        = np.zeros((H, W))
    amp[0, :3] = 1.0
    mu         = np.full((H, W), 25.0)

    with pytest.raises(ValueError):
        PresenceCalibration(_cube(amp, mu), _cube(amp, mu), 1).run()


def test_risk_coverage_curve_is_monotone_for_informative_confidence():
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
    with pytest.raises(ValueError):
        RiskCoverage(np.zeros((4, 4)), np.zeros((5, 4)))


def test_uninformative_confidence_gives_flat_curve():
    rng          = np.random.default_rng(3)
    disagreement = rng.uniform(0.0, 1.0, size=(H, W))
    risk         = np.ones((H, W))

    rows, scalars = RiskCoverage(disagreement, risk).run()

    assert scalars["aurc"] == pytest.approx(scalars["full_coverage_risk"])
    assert all(row["risk"] == pytest.approx(1.0) for row in rows)


def test_label_suspects_flag_confident_wrong_pixels():
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
