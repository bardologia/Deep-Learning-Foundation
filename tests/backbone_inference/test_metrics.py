from __future__ import annotations

import numpy as np
import pytest

from pipelines.backbone.inference.metrics import Metrics, Result
from tools.loss.param_loss                import ParamMatcher


N_GAUSSIANS = 5
N_ELEV      = 24


def _x_axis(n: int = N_ELEV) -> np.ndarray:
    return np.linspace(-20.0, 80.0, n).astype(np.float32)


def _make_result(pred_curves, gt_curves, params_pred=None, params_gt=None) -> Result:
    pixel_maps = Metrics.curve_pixel_metrics(pred_curves, gt_curves)

    return Result(
        pred_curves        = pred_curves,
        gt_curves          = gt_curves,
        pixel_mse          = pixel_maps["mse"],
        pixel_mae          = pixel_maps["mae"],
        pixel_r2           = pixel_maps["r2"],
        pixel_cosine       = pixel_maps["cos"],
        pixel_peak_err_idx = pixel_maps["peak"].astype(np.int32),
        cube_directory     = None,
        azimuth_offset     = 0,
        range_offset       = 0,
        params_pred        = params_pred,
        params_gt          = params_gt,
    )


def _curves_from_params(params, x_axis, n_gaussians) -> np.ndarray:
    C, H, W = params.shape
    curve   = np.zeros((x_axis.size, H, W), dtype=np.float32)

    for k in range(n_gaussians):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = x_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))

    return curve.astype(np.float32)


def test_curve_pixel_metrics_identical_is_perfect():
    rng   = np.random.default_rng(0)
    curve = rng.random((N_ELEV, 6, 7)).astype(np.float32) + 0.1

    out = Metrics.curve_pixel_metrics(curve, curve)

    assert np.allclose(out["mse"], 0.0)
    assert np.allclose(out["mae"], 0.0)
    assert np.allclose(out["r2"], 1.0, atol=1e-5)
    assert np.allclose(out["cos"], 1.0, atol=1e-5)
    assert np.all(out["peak"] == 0)


def test_curve_pixel_metrics_more_noise_is_worse():
    rng   = np.random.default_rng(1)
    ref   = rng.random((N_ELEV, 5, 5)).astype(np.float32) + 0.5
    small = ref + 0.01 * rng.standard_normal(ref.shape).astype(np.float32)
    large = ref + 0.50 * rng.standard_normal(ref.shape).astype(np.float32)

    mse_small = Metrics.curve_pixel_metrics(small, ref)["mse"].mean()
    mse_large = Metrics.curve_pixel_metrics(large, ref)["mse"].mean()

    assert mse_small < mse_large


def test_curve_pixel_metrics_mse_mae_exact_on_tiny_input():
    ref  = np.array([[[0.0]], [[0.0]], [[0.0]]], dtype=np.float32)
    pred = np.array([[[1.0]], [[2.0]], [[2.0]]], dtype=np.float32)

    out = Metrics.curve_pixel_metrics(pred, ref)

    assert out["mse"][0, 0] == pytest.approx((1.0 + 4.0 + 4.0) / 3.0)
    assert out["mae"][0, 0] == pytest.approx((1.0 + 2.0 + 2.0) / 3.0)


def test_curve_pixel_metrics_peak_index_difference():
    ref     = np.zeros((4, 1, 1), dtype=np.float32)
    pred    = np.zeros((4, 1, 1), dtype=np.float32)
    ref[1]  = 1.0
    pred[3] = 1.0

    out = Metrics.curve_pixel_metrics(pred, ref)

    assert int(out["peak"][0, 0]) == 2


def test_curve_pixel_metrics_cosine_orthogonal():
    ref     = np.zeros((4, 1, 1), dtype=np.float32)
    pred    = np.zeros((4, 1, 1), dtype=np.float32)
    ref[0]  = 1.0
    pred[1] = 1.0

    out = Metrics.curve_pixel_metrics(pred, ref)

    assert out["cos"][0, 0] == pytest.approx(0.0, abs=1e-6)


def test_curve_scalar_metrics_identical():
    rng   = np.random.default_rng(2)
    curve = rng.random((N_ELEV, 4, 4)).astype(np.float32) + 0.2
    m     = Metrics(_make_result(curve, curve), _x_axis(), N_GAUSSIANS)

    scal = m._curve_scalar_metrics(curve, curve, suffix="gt")

    assert scal["curve_mse_gt"]  == pytest.approx(0.0, abs=1e-10)
    assert scal["curve_mae_gt"]  == pytest.approx(0.0, abs=1e-10)
    assert scal["curve_rmse_gt"] == pytest.approx(0.0, abs=1e-10)
    assert scal["overall_r2_gt"] == pytest.approx(1.0, abs=1e-6)
    assert scal["psnr_db_gt"]    == float("inf")


def test_curve_scalar_metrics_rmse_is_sqrt_mse():
    rng  = np.random.default_rng(3)
    ref  = rng.random((N_ELEV, 3, 3)).astype(np.float32)
    pred = ref + 0.1 * rng.standard_normal(ref.shape).astype(np.float32)
    m    = Metrics(_make_result(pred, ref), _x_axis(), N_GAUSSIANS)

    scal = m._curve_scalar_metrics(pred, ref, suffix="gt")

    assert scal["curve_rmse_gt"] == pytest.approx(np.sqrt(scal["curve_mse_gt"]))


def test_psnr_constant_reference_is_nan():
    ref  = np.ones((4, 2, 2), dtype=np.float32)
    pred = ref + 0.1

    assert np.isnan(Metrics._psnr(pred, ref))


def test_basic_stats_and_percentiles_exact():
    x     = np.arange(0.0, 101.0, dtype=np.float64).reshape(101, 1, 1)
    stats = Metrics._basic_stats(x)
    pcts  = Metrics._percentiles(x)

    assert stats["min"]    == 0.0
    assert stats["max"]    == 100.0
    assert stats["median"] == 50.0
    assert stats["mean"]   == pytest.approx(50.0)
    assert pcts["p50"]     == pytest.approx(50.0)
    assert pcts["p25"]     == pytest.approx(25.0)


def test_select_pixels_best_worst_random_disjoint():
    rng        = np.random.default_rng(4)
    metric_map = rng.random((10, 10)).astype(np.float32)

    sel = Metrics.select_pixels(metric_map, n_best=3, n_worst=3, n_random=4, seed=0)

    best_vals  = [metric_map[y, x] for y, x in sel["best"]]
    worst_vals = [metric_map[y, x] for y, x in sel["worst"]]

    assert best_vals == sorted(best_vals)
    assert max(best_vals) <= min(worst_vals)
    assert sel["best"].shape  == (3, 2)
    assert sel["worst"].shape == (3, 2)

    all_coords = {tuple(c) for c in np.concatenate([sel["best"], sel["worst"], sel["random"]])}
    assert len(all_coords) == 3 + 3 + 4


def test_select_pixels_zero_counts_select_nothing():
    rng        = np.random.default_rng(4)
    metric_map = rng.random((10, 10)).astype(np.float32)

    sel = Metrics.select_pixels(metric_map, n_best=0, n_worst=0, n_random=0, seed=0)

    assert sel["best"].shape[0]   == 0
    assert sel["worst"].shape[0]  == 0
    assert sel["random"].shape[0] == 0

    sel = Metrics.select_pixels(metric_map, n_best=2, n_worst=0, n_random=5, seed=0)

    assert sel["best"].shape[0]   == 2
    assert sel["worst"].shape[0]  == 0
    assert sel["random"].shape[0] == 5


def test_flat_to_yx_roundtrip():
    width = 7
    flat  = np.array([0, 6, 7, 20], dtype=np.int64)
    yx    = Metrics._flat_to_yx(flat, width)

    assert list(yx[0]) == [0, 0]
    assert list(yx[1]) == [0, 6]
    assert list(yx[2]) == [1, 0]
    assert list(yx[3]) == [2, 6]


def test_elev_metrics_identical_perfect():
    rng   = np.random.default_rng(5)
    curve = rng.random((N_ELEV, 4, 4)).astype(np.float32) + 0.3
    m     = Metrics(_make_result(curve, curve), _x_axis(), N_GAUSSIANS)

    elev = m._elev_metrics(curve, curve, suffix="gt")

    assert np.allclose(elev["elev_mae_gt"],  0.0, atol=1e-6)
    assert np.allclose(elev["elev_rmse_gt"], 0.0, atol=1e-6)
    assert np.allclose(elev["elev_r2_gt"],   1.0, atol=1e-5)


def test_active_count_stats_identical_perfect_agreement():
    H, W   = 6, 6
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    params[0] = 1.0
    params[3] = 1.0
    params[6] = 0.0

    res = _make_result(np.zeros((N_ELEV, H, W), np.float32), np.zeros((N_ELEV, H, W), np.float32), params_pred=params.copy(), params_gt=params.copy())
    out = Metrics(res, _x_axis(), N_GAUSSIANS)._active_count_stats()

    assert out["count_exact_frac"]      == pytest.approx(1.0)
    assert out["count_under_frac"]      == pytest.approx(0.0)
    assert out["count_over_frac"]       == pytest.approx(0.0)
    assert out["active_count_gt_mean"]  == pytest.approx(2.0)
    assert out["slot_0_active_gt_frac"] == pytest.approx(1.0)
    assert out["slot_2_active_gt_frac"] == pytest.approx(0.0)
    assert out["count_acc_gt2"]         == pytest.approx(1.0)
    assert out["count_acc_pred2"]       == pytest.approx(1.0)
    assert "count_acc_gt1" not in out
    assert "count_acc_pred1" not in out


def test_active_count_stats_per_slot_pred_is_permutation_invariant():
    H, W = 5, 5
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0],  gt[1] = 1.0, -10.0
    gt[3],  gt[4] =  1.0,  10.0
    gt[6],  gt[7] =  1.0,  40.0

    order = [2, 0, 1, 3, 4]
    pred  = _reorder_groups(gt, order)

    out = Metrics(_make_result(np.zeros((N_ELEV, H, W), np.float32), np.zeros((N_ELEV, H, W), np.float32), params_pred=pred, params_gt=gt), _x_axis(), N_GAUSSIANS)._active_count_stats()

    for k in range(N_GAUSSIANS):
        assert out[f"slot_{k}_active_pred_frac"] == pytest.approx(out[f"slot_{k}_active_gt_frac"])


def test_active_count_stats_undercount_when_pred_drops_slot():
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0] = 1.0
    gt[3] = 1.0

    pred    = gt.copy()
    pred[3] = 0.0

    res = _make_result(np.zeros((N_ELEV, H, W), np.float32), np.zeros((N_ELEV, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS)._active_count_stats()

    assert out["count_under_frac"]       == pytest.approx(1.0)
    assert out["count_exact_frac"]       == pytest.approx(0.0)
    assert out["active_count_pred_mean"] == pytest.approx(1.0)
    assert out["active_frac_pred"]       == pytest.approx(1.0 / N_GAUSSIANS)
    assert out["count_acc_gt2"]          == pytest.approx(0.0)


def _reorder_groups(params: np.ndarray, order: list) -> np.ndarray:
    K       = len(order)
    grouped = params.reshape(K, 3, *params.shape[1:])
    return grouped[order].reshape(K * 3, *params.shape[1:])


def test_param_population_stats_active_inactive_global_means():
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0, :, :2] = 2.0
    gt[1, :, :2] = 10.0
    gt[2, :, :2] = 4.0

    pred            = gt.copy()
    pred[0, :, :2] += 0.5
    pred[1, :, :2] += 1.0

    res = _make_result(np.zeros((N_ELEV, H, W), np.float32), np.zeros((N_ELEV, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS)._param_population_stats()

    assert out["slot_0_amp_active_gt_mean"]   == pytest.approx(2.0)
    assert out["slot_0_amp_inactive_gt_mean"] == pytest.approx(0.0)
    assert out["slot_0_amp_global_gt_mean"]   == pytest.approx(1.0)
    assert out["slot_0_mu_active_gt_mean"]    == pytest.approx(10.0)
    assert out["slot_0_sig_active_gt_std"]    == pytest.approx(0.0)

    assert "slot_0_mu_inactive_gt_mean"  not in out
    assert "slot_0_mu_global_gt_mean"    not in out
    assert "slot_0_sig_inactive_gt_mean" not in out
    assert "slot_0_sig_global_gt_std"    not in out

    assert out["slot_0_amp_active_pred_mean"] == pytest.approx(2.5)
    assert out["slot_0_mu_active_pred_mean"]  == pytest.approx(11.0)
    assert np.isnan(out["slot_0_amp_inactive_pred_mean"])
    assert out["slot_0_amp_global_pred_mean"] == pytest.approx(2.5)
    assert out["slot_0_mu_global_pred_mean"]  == pytest.approx(11.0)

    assert np.isnan(out["slot_1_amp_active_gt_mean"])
    assert out["slot_1_amp_global_gt_mean"] == pytest.approx(0.0)


def test_param_population_stats_on_nan_masked_ground_truth():
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0, :, :2] = 2.0
    gt[1, :, :2] = 10.0
    gt[2, :, :2] = 4.0

    pred = gt.copy()

    for k in range(N_GAUSSIANS):
        inactive = ~ParamMatcher.is_active(gt[3 * k])
        gt[3 * k + 1][inactive] = np.nan
        gt[3 * k + 2][inactive] = np.nan

    res = _make_result(np.zeros((N_ELEV, H, W), np.float32), np.zeros((N_ELEV, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS)._param_population_stats()

    assert out["slot_0_mu_active_gt_mean"]  == pytest.approx(10.0)
    assert out["slot_0_sig_active_gt_mean"] == pytest.approx(4.0)

    assert out["slot_0_amp_inactive_gt_mean"] == pytest.approx(0.0)
    assert out["slot_0_amp_global_gt_mean"]   == pytest.approx(1.0)

    assert not any(key.endswith("_gt_mean") and np.isnan(value) for key, value in out.items() if key.startswith("slot_0_"))


def test_param_population_stats_pred_is_permutation_invariant():
    H, W = 5, 5
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0], gt[1], gt[2] = 1.0, -10.0, 3.0
    gt[3], gt[4], gt[5] = 1.0,  10.0, 4.0
    gt[6], gt[7], gt[8] = 1.0,  40.0, 5.0

    order = [2, 0, 1, 3, 4]
    pred  = _reorder_groups(gt, order)

    res = _make_result(np.zeros((N_ELEV, H, W), np.float32), np.zeros((N_ELEV, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS)._param_population_stats()

    for k in range(3):
        for param in ("amp", "mu", "sig"):
            assert out[f"slot_{k}_{param}_active_pred_mean"] == pytest.approx(out[f"slot_{k}_{param}_active_gt_mean"])


def _matched(pred, gt, n_gaussians=N_GAUSSIANS, tol=5.0):
    res = _make_result(np.zeros((N_ELEV, *pred.shape[1:]), np.float32), np.zeros((N_ELEV, *pred.shape[1:]), np.float32), params_pred=pred, params_gt=gt)
    return Metrics(res, _x_axis(), n_gaussians)._matched_gaussian_metrics(match_tol=tol)


def test_matched_identical_is_perfect():
    H, W   = 6, 6
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    for k in range(N_GAUSSIANS):
        params[3 * k]     = 1.0
        params[3 * k + 1] = float(k * 12)
        params[3 * k + 2] = 3.0

    out = _matched(params.copy(), params.copy())

    assert out["matched_mu_mae"]    == pytest.approx(0.0, abs=1e-6)
    assert out["matched_sig_mae"]   == pytest.approx(0.0, abs=1e-6)
    assert out["matched_amp_mae"]   == pytest.approx(0.0, abs=1e-6)
    assert out["matched_recall"]    == pytest.approx(1.0, abs=1e-6)
    assert out["matched_precision"] == pytest.approx(1.0, abs=1e-6)


def test_matched_is_invariant_to_pred_slot_permutation():
    rng    = np.random.default_rng(21)
    H, W   = 8, 8
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    for k in range(N_GAUSSIANS):
        params[3 * k]     = 1.0
        params[3 * k + 1] = float(k * 20)
        params[3 * k + 2] = 3.0

    pred = params.copy()
    for k in range(N_GAUSSIANS):
        pred[3 * k]     += 0.1 * rng.standard_normal((H, W)).astype(np.float32)
        pred[3 * k + 1] += 0.5 * rng.standard_normal((H, W)).astype(np.float32)
        pred[3 * k + 2] += 0.2 * rng.standard_normal((H, W)).astype(np.float32)

    base = _matched(pred.copy(),                          params.copy())
    reor = _matched(_reorder_groups(pred, [2, 0, 4, 1, 3]), params.copy())

    assert reor["matched_mu_mae"]  == pytest.approx(base["matched_mu_mae"],  rel=1e-5)
    assert reor["matched_sig_mae"] == pytest.approx(base["matched_sig_mae"], rel=1e-5)
    assert reor["matched_amp_mae"] == pytest.approx(base["matched_amp_mae"], rel=1e-5)
    assert reor["matched_recall"]  == pytest.approx(base["matched_recall"],  rel=1e-5)


def test_matched_recall_penalises_filler_far_from_gt():
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)
    pred = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0], gt[1], gt[2] = 1.0, 0.0,  3.0
    gt[3], gt[4], gt[5] = 1.0, 40.0, 3.0

    pred[0], pred[1], pred[2] = 1.0, 0.0, 3.0
    pred[3], pred[4], pred[5] = 1.0, 5.0, 3.0

    out = _matched(pred, gt, tol=5.0)

    assert out["matched_recall_gt2"]    == pytest.approx(0.5, abs=1e-6)
    assert out["matched_precision_gt2"] == pytest.approx(0.5, abs=1e-6)
    assert out["matched_mu_mae_gt2"]    == pytest.approx((0.0 + 35.0) / 2.0, abs=1e-5)


def test_matched_amp_error_over_matched_pairs():
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)
    pred = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0], gt[1], gt[2] = 1.0, 0.0,  3.0
    gt[3], gt[4], gt[5] = 0.8, 40.0, 3.0

    pred[0], pred[1], pred[2] = 0.7, 0.0,  3.0
    pred[3], pred[4], pred[5] = 1.0, 40.0, 3.0

    out = _matched(pred, gt, tol=5.0)

    assert out["matched_amp_mae"]     == pytest.approx((0.3 + 0.2) / 2.0, abs=1e-6)
    assert out["matched_amp_rmse"]    == pytest.approx(np.sqrt((0.3 ** 2 + 0.2 ** 2) / 2.0), abs=1e-6)
    assert out["matched_amp_mae_gt2"] == pytest.approx((0.3 + 0.2) / 2.0, abs=1e-6)


@pytest.mark.real_data
def test_metrics_identical_real_parameter_window_is_perfect(parameters):
    params = np.asarray(parameters[:, :16, :16], dtype=np.float32)
    x_axis = _x_axis(40)
    curve  = _curves_from_params(params, x_axis, N_GAUSSIANS)

    res = _make_result(curve, curve, params_pred=params.copy(), params_gt=params.copy())
    m   = Metrics(res, x_axis, N_GAUSSIANS)

    metrics = m.compute(param_space=True)

    assert metrics["curve_mse_gt"]     == pytest.approx(0.0, abs=1e-8)
    assert metrics["overall_r2_gt"]    == pytest.approx(1.0, abs=1e-5)
    assert metrics["pixel_r2_gt_mean"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.real_data
def test_metrics_perturbed_real_window_is_worse(parameters):
    rng    = np.random.default_rng(11)
    params = np.asarray(parameters[:, :16, :16], dtype=np.float32)
    x_axis = _x_axis(40)
    gt     = _curves_from_params(params, x_axis, N_GAUSSIANS)

    pred_small = gt + 0.001 * rng.standard_normal(gt.shape).astype(np.float32) * gt.std()
    pred_large = gt + 0.200 * rng.standard_normal(gt.shape).astype(np.float32) * gt.std()

    m_small = Metrics(_make_result(pred_small, gt), x_axis, N_GAUSSIANS)
    m_large = Metrics(_make_result(pred_large, gt), x_axis, N_GAUSSIANS)

    s = m_small.compute(param_space=False)
    l = m_large.compute(param_space=False)

    assert s["curve_mse_gt"]  < l["curve_mse_gt"]
    assert s["overall_r2_gt"] > l["overall_r2_gt"]


@pytest.mark.real_data
def test_reduced_comparison_identical_reduced_zero_improvement(parameters):
    params = np.asarray(parameters[:, :12, :12], dtype=np.float32)
    x_axis = _x_axis(40)
    gt     = _curves_from_params(params, x_axis, N_GAUSSIANS)
    pred   = gt.copy()

    m    = Metrics(_make_result(pred, gt), x_axis, N_GAUSSIANS)
    comp = m.reduced_comparison(gt.copy())

    assert comp.metrics["improvement_pixel_mse_mean"] == pytest.approx(0.0, abs=1e-6)
    assert comp.reduced_norm.shape == gt.shape


def test_label_quality_perfect_fit_gives_r2_one():
    rng    = np.random.default_rng(0)
    curves = rng.uniform(0.5, 2.0, size=(N_ELEV, 4, 3)).astype(np.float32)
    result = _make_result(curves, curves)

    r2_map, stats = Metrics(result, _x_axis(), N_GAUSSIANS).label_quality(curves.copy())

    assert r2_map.shape == (4, 3)
    assert np.allclose(r2_map, 1.0, atol=1e-5)
    assert stats["label_r2_mean"]          == pytest.approx(1.0, abs=1e-5)
    assert stats["label_r2_frac_below_half"] == pytest.approx(0.0)


def test_label_quality_noisier_labels_score_lower():
    rng   = np.random.default_rng(1)
    raw   = rng.uniform(0.5, 2.0, size=(N_ELEV, 4, 3)).astype(np.float32)
    clean = raw + rng.normal(0.0, 0.01, size=raw.shape).astype(np.float32)
    noisy = raw + rng.normal(0.0, 0.60, size=raw.shape).astype(np.float32)

    _map_clean, stats_clean = Metrics(_make_result(clean, clean), _x_axis(), N_GAUSSIANS).label_quality(raw)
    _map_noisy, stats_noisy = Metrics(_make_result(noisy, noisy), _x_axis(), N_GAUSSIANS).label_quality(raw)

    assert stats_clean["label_r2_mean"] > stats_noisy["label_r2_mean"]


def test_label_quality_shape_mismatch_raises():
    curves = np.ones((N_ELEV, 4, 3), dtype=np.float32)
    result = _make_result(curves, curves)

    with pytest.raises(ValueError):
        Metrics(result, _x_axis(), N_GAUSSIANS).label_quality(np.ones((N_ELEV, 5, 3), dtype=np.float32))
