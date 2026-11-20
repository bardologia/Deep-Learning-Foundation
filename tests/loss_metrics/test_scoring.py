
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.metrics.scoring import FiniteScalar, MetricOrientation, R2, RankCorrelation, RelativeImprovement, SeedAggregation


def test_finite_scalar_accepts_floats_and_ints():
    assert FiniteScalar.is_finite_number(1.5)
    assert FiniteScalar.is_finite_number(3)
    assert FiniteScalar.coerce(2) == 2.0


def test_finite_scalar_rejects_bool():
    assert FiniteScalar.is_finite_number(True) is False
    assert FiniteScalar.coerce(True) is None


def test_finite_scalar_rejects_nan_inf_and_strings():
    assert FiniteScalar.is_finite_number(float("nan")) is False
    assert FiniteScalar.is_finite_number(float("inf")) is False
    assert FiniteScalar.is_finite_number("x") is False
    assert FiniteScalar.coerce(float("nan")) is None
    assert FiniteScalar.coerce("x") is None


def test_r2_perfect_prediction_is_one():
    ref  = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    pred = ref.copy()
    out  = R2.pixel_map(pred, ref, axis=1)
    assert np.allclose(out, 1.0, atol=1e-6)


def test_r2_mean_prediction_is_zero():
    ref  = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    pred = np.full_like(ref, ref.mean())
    out  = R2.pixel_map(pred, ref, axis=1)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_r2_output_dtype_and_shape():
    ref  = np.random.RandomState(0).rand(5, 7).astype(np.float64)
    pred = np.random.RandomState(1).rand(5, 7).astype(np.float64)
    out  = R2.pixel_map(pred, ref, axis=0)
    assert out.dtype == np.float32
    assert out.shape == (7,)


def test_r2_worse_than_mean_is_negative():
    ref  = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    pred = np.array([[10.0, -5.0, 20.0, -8.0]], dtype=np.float64)
    out  = R2.pixel_map(pred, ref, axis=1)
    assert out[0] < 0.0


def test_r2_constant_reference_is_undefined():
    ref  = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float64)
    pred = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]], dtype=np.float64)
    out  = R2.pixel_map(pred, ref, axis=1)

    assert out[0] == pytest.approx(1.0)
    assert math.isnan(out[1])
    assert math.isnan(out[2])
    assert R2.undefined_count(out) == 2


def test_orientation_lower_tokens():
    assert MetricOrientation.direction("curve_mse") == "lower"
    assert MetricOrientation.direction("param_mae") == "lower"
    assert MetricOrientation.higher_is_better("recon_loss") is False


def test_orientation_higher_tokens():
    assert MetricOrientation.direction("r2_curve")                   == "higher"
    assert MetricOrientation.direction("ssim_z")                     == "higher"
    assert MetricOrientation.direction("ssim_cube_norm_col_mean")    == "higher"
    assert MetricOrientation.higher_is_better("matched_f1")          is True
    assert MetricOrientation.direction("count_acc_gt3")              == "higher"
    assert MetricOrientation.direction("count_acc_pred3")            == "higher"
    assert MetricOrientation.direction("count_exact_frac")           == "higher"
    assert MetricOrientation.direction("improvement_pixel_mse_mean") == "higher"


def test_orientation_neutral_returns_none():
    assert MetricOrientation.direction("n_pixels") is None
    assert MetricOrientation.direction("gt_amp") is None
    assert MetricOrientation.direction("matched_n_pairs") is None
    assert MetricOrientation.higher_is_better("n_pixels") is None
    assert MetricOrientation.direction("active_count_pred_mean")  is None
    assert MetricOrientation.direction("active_frac_pred")        is None
    assert MetricOrientation.direction("slot_2_active_pred_frac") is None


def test_orientation_undefined_counts_are_neutral():
    for key in ("pixel_r2_gt_undefined_count", "label_r2_undefined_count", "r2_undefined_count"):
        assert MetricOrientation.direction(key) is None, key


def test_orientation_slot_organization_and_failure_ok_are_higher():
    for key in ("slot_gt_alignment", "slot_mu_rank_diag", "slot_usage_entropy", "failure_frac_ok", "presence_rank_corr"):
        assert MetricOrientation.direction(key) == "higher", key


def test_orientation_fractions_of_faults_are_lower():
    for key in ("count_over_frac", "count_under_frac", "clamp_amp_ceil_frac", "failure_frac_missed", "failure_miss_rate", "norm_in_ch2_overflow_frac", "norm_in_ch1_nonfinite_frac", "norm_in_nonfinite_frac", "norm_in_worst_abs_mean", "z_ce_gt_12", "pixel_peak_idx_d_gt_p95", "pixel_peak_idx_d_gt_nonempty_mean", "pixel_peak_err_units_mean_gt_nonempty", "roughness_excess_frac", "label_r2_frac_below_half"):
        assert MetricOrientation.direction(key) == "lower", key


def test_orientation_descriptive_slot_and_audit_keys_are_neutral():
    for key in ("slot_1_amp_active_gt_mean", "slot_3_mu_global_pred_std", "slot_0_mu_gt_mean", "norm_in_ch0_mean", "clamp_n_active", "activity_threshold", "roughness_ratio", "checkpoint_epoch", "n_pixels_gt_empty", "gt_empty_frac", "strat_label_r2_spearman", "strat_gt_active_count_n_bins", "strat_label_r2_pixels_dropped", "ssim_gt_z_n_scored", "ssim_norm_col_n_nonfinite", "ssim_cube_norm_row_n_nonfinite"):
        assert MetricOrientation.direction(key) is None, key


def test_orientation_spreads_are_not_ranked():
    for key in ("pixel_cosine_gt_std", "pixel_cosine_gt_nonempty_std", "pixel_r2_gt_std", "pixel_mae_gt_std", "pixel_peak_idx_d_gt_std"):
        assert MetricOrientation.direction(key) is None, key

    assert MetricOrientation.direction("pixel_cosine_gt_mean") == "higher"
    assert MetricOrientation.direction("pixel_mae_gt_mean")    == "lower"


def test_orientation_reconstruction_error_keys_are_declared():
    for key in ("rmse", "mse_mean", "mse_median", "mse_mean_normalized", "mae_mean", "mae_mean_normalized", "relative_l2_mean", "relative_l2_median", "curve_l1", "curve_l1_p90"):
        assert MetricOrientation.direction(key) == "lower", key
    for key in ("pearson_mean", "pearson_median"):
        assert MetricOrientation.direction(key) == "higher", key
    for key in ("n_curves", "n_active_curves", "n_valid_pixels", "num_batches", "profile_length", "patch_height", "seed"):
        assert MetricOrientation.direction(key) is None, key


def test_orientation_reduction_overrides_mse_token():
    assert MetricOrientation.direction("relative_mse_reduction")       == "higher"
    assert MetricOrientation.higher_is_better("relative_mse_reduction") is True
    assert MetricOrientation.direction("fraction_pred_beats_reduced")  == "higher"


def test_relative_improvement_lower_is_better():
    frac = RelativeImprovement.fraction(2.0, 1.0, higher_is_better=False)
    assert math.isclose(frac, 0.5)


def test_relative_improvement_higher_is_better():
    frac = RelativeImprovement.fraction(1.0, 2.0, higher_is_better=True)
    assert math.isclose(frac, 1.0)


def test_relative_improvement_nan_on_zero_baseline():
    assert math.isnan(RelativeImprovement.fraction(0.0, 1.0))


def test_relative_improvement_nan_on_non_number():
    assert math.isnan(RelativeImprovement.fraction("x", 1.0))
    assert math.isnan(RelativeImprovement.fraction(1.0, float("inf")))


def test_relative_improvement_percent_string():
    assert RelativeImprovement.percent(2.0, 1.0, higher_is_better=False) == "+50.0%"
    assert RelativeImprovement.percent(0.0, 1.0) == "n/a"
    assert RelativeImprovement.percent(0.0, 1.0, empty="-") == "-"


def test_rank_correlation_averages_tied_ranks():
    assert RankCorrelation.ranks(np.array([1.0, 1.0, 2.0])).tolist() == [0.5, 0.5, 2.0]
    assert RankCorrelation.ranks(np.array([[3.0, 1.0], [1.0, 7.0]])).tolist() == [2.0, 0.5, 0.5, 3.0]


def test_rank_correlation_is_unaffected_by_tie_order():
    values = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    errors = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    shuffled        = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    reversed_errors = np.array([3.0, 4.0, 5.0, 2.0, 1.0])

    assert RankCorrelation.spearman(values, errors) == pytest.approx(RankCorrelation.spearman(shuffled, reversed_errors))


def test_rank_correlation_of_monotone_and_inverted_vectors():
    a = np.array([0.1, 0.4, 0.2, 0.9])

    assert RankCorrelation.spearman(a,  a) == pytest.approx(1.0)
    assert RankCorrelation.spearman(a, -a) == pytest.approx(-1.0)


def test_rank_correlation_ignores_non_finite_pairs():
    a = np.array([1.0, 2.0, 3.0, np.nan])
    b = np.array([1.0, 2.0, 3.0, 10.0])

    assert RankCorrelation.spearman(a, b) == pytest.approx(1.0)


def test_rank_correlation_is_nan_without_variance():
    constant = np.zeros(5)

    assert math.isnan(RankCorrelation.spearman(constant, np.arange(5.0)))
    assert math.isnan(RankCorrelation.spearman(np.array([1.0, np.nan, np.nan, 4.0, 5.0]), np.array([1.0, 2.0, 3.0, np.nan, np.nan])))


def test_relative_improvement_scales_with_the_baseline_magnitude():
    for baseline in (1e-6, 0.375, 12.5, 4.2e3):
        assert math.isclose(RelativeImprovement.fraction(baseline, baseline * 0.9, higher_is_better=False), 0.1, rel_tol=1e-9)
        assert RelativeImprovement.percent(baseline, baseline * 0.9, higher_is_better=False) == "+10.0%"


def test_seed_aggregation_counts_contributors_and_names_the_shortfall():
    seeds = [{"f1": 0.8, "mae": 0.1}, {"f1": float("nan"), "mae": 0.2}, {"mae": 0.3}]
    names = ["seed0", "seed1", "seed2"]

    assert SeedAggregation.contributors(seeds, ["f1", "mae", "absent"]) == {"f1": 1, "mae": 3, "absent": 0}
    assert SeedAggregation.shortfall(seeds, ["f1", "mae"], names) == {"f1": ["seed1", "seed2"]}


def test_unknown_metric_has_no_orientation():
    with pytest.raises(StopIteration):
        MetricOrientation.direction("zzz_unclassified_metric")
