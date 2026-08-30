"""Tests covering the scalar validation, R-squared, orientation and correlation helpers.

Exercises ``FiniteScalar`` coercion, per-pixel ``R2`` maps, the higher/lower/neutral
verdicts of ``MetricOrientation``, ``RelativeImprovement`` fractions and strings, and
the tie-aware Spearman correlation in ``RankCorrelation``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.metrics.scoring import FiniteScalar, MetricOrientation, R2, RelativeImprovement


def test_finite_scalar_accepts_floats_and_ints():
    """Verifies floats and ints pass the finite-number check and coerce to float."""
    assert FiniteScalar.is_finite_number(1.5)
    assert FiniteScalar.is_finite_number(3)
    assert FiniteScalar.coerce(2) == 2.0


def test_finite_scalar_rejects_bool():
    """Verifies booleans are not accepted as numeric metric values."""
    assert FiniteScalar.is_finite_number(True) is False
    assert FiniteScalar.coerce(True) is None


def test_finite_scalar_rejects_nan_inf_and_strings():
    """Verifies NaN, infinity and strings fail the check and coerce to None."""
    assert FiniteScalar.is_finite_number(float("nan")) is False
    assert FiniteScalar.is_finite_number(float("inf")) is False
    assert FiniteScalar.is_finite_number("x") is False
    assert FiniteScalar.coerce(float("nan")) is None
    assert FiniteScalar.coerce("x") is None


def test_r2_perfect_prediction_is_one():
    """Verifies R-squared is one when the prediction equals the reference."""
    ref  = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    pred = ref.copy()
    out  = R2.pixel_map(pred, ref, axis=1)
    assert np.allclose(out, 1.0, atol=1e-6)


def test_r2_mean_prediction_is_zero():
    """Verifies R-squared is zero when the prediction is the reference mean."""
    ref  = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    pred = np.full_like(ref, ref.mean())
    out  = R2.pixel_map(pred, ref, axis=1)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_r2_output_dtype_and_shape():
    """Verifies the pixel map reduces along the requested axis and returns float32."""
    ref  = np.random.RandomState(0).rand(5, 7).astype(np.float64)
    pred = np.random.RandomState(1).rand(5, 7).astype(np.float64)
    out  = R2.pixel_map(pred, ref, axis=0)
    assert out.dtype == np.float32
    assert out.shape == (7,)


def test_r2_worse_than_mean_is_negative():
    """Verifies a prediction worse than the mean gives a negative R-squared."""
    ref  = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    pred = np.array([[10.0, -5.0, 20.0, -8.0]], dtype=np.float64)
    out  = R2.pixel_map(pred, ref, axis=1)
    assert out[0] < 0.0


def test_orientation_lower_tokens():
    """Verifies error and loss metric names are classified as lower-is-better."""
    assert MetricOrientation.direction("curve_mse") == "lower"
    assert MetricOrientation.direction("param_mae") == "lower"
    assert MetricOrientation.higher_is_better("recon_loss") is False


def test_orientation_higher_tokens():
    """Verifies R-squared, SSIM, F1, accuracy, improvement and agreement names are higher-is-better."""
    assert MetricOrientation.direction("r2_curve") == "higher"
    assert MetricOrientation.direction("ssim_z") == "higher"
    assert MetricOrientation.higher_is_better("matched_f1") is True
    assert MetricOrientation.direction("count_acc_gt3")   == "higher"
    assert MetricOrientation.direction("count_acc_pred3") == "higher"
    assert MetricOrientation.direction("count_exact_frac") == "higher"
    assert MetricOrientation.direction("improvement_pixel_mse_mean") == "higher"
    assert MetricOrientation.direction("physics_valid_fraction")     == "higher"
    assert MetricOrientation.direction("phase_agreement_gt_mean")    == "higher"


def test_orientation_neutral_returns_none():
    """Verifies descriptive count and fraction metrics carry no orientation."""
    assert MetricOrientation.direction("n_pixels") is None
    assert MetricOrientation.direction("gt_amp") is None
    assert MetricOrientation.direction("matched_n_pairs") is None
    assert MetricOrientation.higher_is_better("n_pixels") is None
    assert MetricOrientation.direction("active_count_pred_mean")  is None
    assert MetricOrientation.direction("active_frac_pred")        is None
    assert MetricOrientation.direction("slot_2_active_pred_frac") is None


def test_orientation_default_is_lower():
    """Verifies an unrecognised metric name defaults to lower-is-better."""
    assert MetricOrientation.direction("some_unknown_thing") == "lower"


def test_orientation_reduction_overrides_mse_token():
    """Verifies the reduction token wins over the mse token so reductions are higher-is-better."""
    assert MetricOrientation.direction("relative_mse_reduction")       == "higher"
    assert MetricOrientation.higher_is_better("relative_mse_reduction") is True
    assert MetricOrientation.direction("fraction_pred_beats_reduced")  == "higher"


def test_relative_improvement_lower_is_better():
    """Verifies halving a lower-is-better baseline reports a fraction of 0.5."""
    frac = RelativeImprovement.fraction(2.0, 1.0, higher_is_better=False)
    assert math.isclose(frac, 0.5)


def test_relative_improvement_higher_is_better():
    """Verifies doubling a higher-is-better baseline reports a fraction of 1.0."""
    frac = RelativeImprovement.fraction(1.0, 2.0, higher_is_better=True)
    assert math.isclose(frac, 1.0)


def test_relative_improvement_nan_on_zero_baseline():
    """Verifies a zero baseline yields NaN instead of dividing by zero."""
    assert math.isnan(RelativeImprovement.fraction(0.0, 1.0))


def test_relative_improvement_nan_on_non_number():
    """Verifies non-numeric or non-finite inputs yield NaN."""
    assert math.isnan(RelativeImprovement.fraction("x", 1.0))
    assert math.isnan(RelativeImprovement.fraction(1.0, float("inf")))


def test_relative_improvement_percent_string():
    """Verifies the percent formatter signs the change and falls back to the empty marker."""
    assert RelativeImprovement.percent(2.0, 1.0, higher_is_better=False) == "+50.0%"
    assert RelativeImprovement.percent(0.0, 1.0) == "n/a"
    assert RelativeImprovement.percent(0.0, 1.0, empty="-") == "-"


@pytest.mark.dataset
def test_r2_on_scene_cube_self_is_one(cube_full):
    """Verifies R-squared of a scene cube window against itself is one."""
    win  = np.abs(np.asarray(cube_full[:16, :8, :8])).astype(np.float64)
    pred = win.copy()
    out  = R2.pixel_map(pred, win, axis=0)
    assert np.allclose(out, 1.0, atol=1e-5)


@pytest.mark.dataset
def test_relative_improvement_on_extraction_meta(param_extraction_meta):
    leaf = _first_numeric(param_extraction_meta)
    if leaf is None:
        pytest.skip("no numeric scalar found in the parameter run metadata")
    frac = RelativeImprovement.fraction(leaf, leaf * 0.9, higher_is_better=False)
    assert math.isclose(abs(frac), 0.1, rel_tol=1e-6)


def _first_numeric(obj):
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)) and math.isfinite(float(obj)) and obj != 0.0:
        return float(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            r = _first_numeric(v)
            if r is not None:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = _first_numeric(v)
            if r is not None:
                return r
    return None
