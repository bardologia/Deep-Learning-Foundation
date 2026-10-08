"""Tests for the metric layer of backbone inference.

Covers per-pixel and scalar curve metrics, z-resolved metrics, active-slot
counting and per-slot parameter populations, Hungarian-matched Gaussian metrics and
their permutation invariance, label quality against the raw cube, curve roughness,
and the switches that gate parameter-space and slot-organisation metrics.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.metrics            import BoundedProcessPool, Metrics, Result
from pipelines.backbone.inference.pipeline           import InferencePipeline
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from tools.data.gaussians                            import GaussianActivity
from tools.data.regions                              import CropRegion

from tests.conftest import RecordingLogger, SilentLogger


N_GAUSSIANS = 5
N_Z         = 24


def _x_axis(n: int = N_Z) -> np.ndarray:
    """Returns an z axis of n samples spanning -20 m to 80 m."""
    return np.linspace(-20.0, 80.0, n).astype(np.float32)


def _make_result(pred_curves, gt_curves, params_pred=None, params_gt=None) -> Result:
    """Builds a Result whose pixel maps are derived from the given curve cubes.

    Args:
        pred_curves : Predicted profiles of shape (n_z, height, width).
        gt_curves   : Ground-truth profiles of shape (n_z, height, width).
        params_pred : Optional predicted parameter cube of shape (3 * n_gaussians, height, width).
        params_gt   : Optional ground-truth parameter cube of the same shape.

    Returns:
        A Result carrying the curves, the per-pixel metric maps and the parameters.
    """
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
        row_offset         = 0,
        col_offset         = 0,
        params_pred        = params_pred,
        params_gt          = params_gt,
    )


def _curves_from_params(params, z_axis, n_gaussians) -> np.ndarray:
    """Renders a Gaussian mixture parameter cube onto the z axis.

    Args:
        params      : Parameter cube of shape (3 * n_gaussians, z, width), grouped amplitude, mu, sigma.
        z_axis      : Z axis in metres, of shape (n_z,).
        n_gaussians : Number of slots to sum.

    Returns:
        Curve cube of shape (n_z, z, width).
    """
    C, H, W = params.shape
    curve   = np.zeros((z_axis.size, H, W), dtype=np.float32)

    for k in range(n_gaussians):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = z_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))

    return curve.astype(np.float32)


def _gaussian_params(seed: int, height: int = 4, width: int = 3) -> np.ndarray:
    """Draws a parameter cube whose first two of N_GAUSSIANS slots are randomly active.

    Args:
        seed   : Seed of the random generator.
        height : Row extent of the cube.
        width  : Range extent of the cube.

    Returns:
        Parameter cube of shape (3 * N_GAUSSIANS, height, width), with centres in
        metres between -10 and 60 and widths in metres between 4 and 9.
    """
    rng    = np.random.default_rng(seed)
    params = np.zeros((N_GAUSSIANS * 3, height, width), dtype=np.float32)

    for k in range(2):
        params[3 * k]     = rng.uniform(0.5, 2.0, (height, width))
        params[3 * k + 1] = rng.uniform(-10.0, 60.0, (height, width))
        params[3 * k + 2] = rng.uniform(4.0, 9.0, (height, width))

    return params


def test_curve_pixel_metrics_identical_is_perfect():
    """Identical curves give zero error, unit R2 and cosine, and no peak displacement."""
    rng   = np.random.default_rng(0)
    curve = rng.random((N_Z, 6, 7)).astype(np.float32) + 0.1

    out = Metrics.curve_pixel_metrics(curve, curve)

    assert np.allclose(out["mse"], 0.0)
    assert np.allclose(out["mae"], 0.0)
    assert np.allclose(out["r2"], 1.0, atol=1e-5)
    assert np.allclose(out["cos"], 1.0, atol=1e-5)
    assert np.all(out["peak"] == 0)


def test_curve_pixel_metrics_more_noise_is_worse():
    """Mean pixel MSE grows with the noise added to the prediction."""
    rng   = np.random.default_rng(1)
    ref   = rng.random((N_Z, 5, 5)).astype(np.float32) + 0.5
    small = ref + 0.01 * rng.standard_normal(ref.shape).astype(np.float32)
    large = ref + 0.50 * rng.standard_normal(ref.shape).astype(np.float32)

    mse_small = Metrics.curve_pixel_metrics(small, ref)["mse"].mean()
    mse_large = Metrics.curve_pixel_metrics(large, ref)["mse"].mean()

    assert mse_small < mse_large


def test_curve_pixel_metrics_mse_mae_exact_on_tiny_input():
    """Pixel MSE and MAE match the values computed by hand on a three-bin profile."""
    ref  = np.array([[[0.0]], [[0.0]], [[0.0]]], dtype=np.float32)
    pred = np.array([[[1.0]], [[2.0]], [[2.0]]], dtype=np.float32)

    out = Metrics.curve_pixel_metrics(pred, ref)

    assert out["mse"][0, 0] == pytest.approx((1.0 + 4.0 + 4.0) / 3.0)
    assert out["mae"][0, 0] == pytest.approx((1.0 + 2.0 + 2.0) / 3.0)


def test_curve_pixel_metrics_peak_index_difference():
    """The peak map reports the signed distance in bins between the predicted and true maxima."""
    ref     = np.zeros((4, 1, 1), dtype=np.float32)
    pred    = np.zeros((4, 1, 1), dtype=np.float32)
    ref[1]  = 1.0
    pred[3] = 1.0

    out = Metrics.curve_pixel_metrics(pred, ref)

    assert int(out["peak"][0, 0]) == 2


def test_curve_pixel_metrics_cosine_orthogonal():
    """Profiles peaking in different bins have zero cosine similarity."""
    ref     = np.zeros((4, 1, 1), dtype=np.float32)
    pred    = np.zeros((4, 1, 1), dtype=np.float32)
    ref[0]  = 1.0
    pred[1] = 1.0

    out = Metrics.curve_pixel_metrics(pred, ref)

    assert out["cos"][0, 0] == pytest.approx(0.0, abs=1e-6)


def test_curve_pixel_metrics_constant_reference_pixel_has_undefined_r2():
    """A pixel whose reference profile is all zero gets a NaN R2 while its neighbours and the other maps stay finite."""
    rng  = np.random.default_rng(5)
    ref  = rng.random((N_Z, 3, 3)).astype(np.float32) + 0.1
    ref[:, 1, 1] = 0.0
    pred = ref + 0.05

    out = Metrics.curve_pixel_metrics(pred, ref)

    assert np.isnan(out["r2"][1, 1])
    assert np.isfinite(np.delete(out["r2"].reshape(-1), 4)).all()
    assert np.isfinite(out["mse"]).all()
    assert np.isfinite(out["cos"]).all()


def test_unit_area_cubes_are_normalised_once_and_shared():
    """Unit-area cubes are cached per side, are distinct objects, and integrate to one along z."""
    rng    = np.random.default_rng(5)
    curve  = rng.random((N_Z, 4, 4)).astype(np.float32) + 0.1
    result = _make_result(curve.copy(), curve.copy())

    pred_norm = result.pred_unit_area()
    gt_norm   = result.gt_unit_area()

    assert result.pred_unit_area() is pred_norm
    assert result.gt_unit_area()   is gt_norm
    assert pred_norm is not gt_norm
    assert pred_norm.sum(axis=0) == pytest.approx(np.ones((4, 4)), abs=1e-5)


def test_ssim_pool_follows_the_configured_cpu_workers():
    """The worker count is taken from the configuration and capped at the CPU count."""
    rng   = np.random.default_rng(4)
    curve = rng.random((N_Z, 4, 4)).astype(np.float32)
    result = _make_result(curve, curve)

    assert Metrics(result, _x_axis(), N_GAUSSIANS, cpu_workers=3, active_amp_thr=1e-4, logger=SilentLogger()).num_workers  == 3
    assert Metrics(result, _x_axis(), N_GAUSSIANS, cpu_workers=10_000, active_amp_thr=1e-4, logger=SilentLogger()).num_workers == (os.cpu_count() or 1)


def test_curve_scalar_metrics_identical():
    """Identical curves give zero MSE, MAE and RMSE, unit R2 and infinite PSNR."""
    rng   = np.random.default_rng(2)
    curve = rng.random((N_Z, 4, 4)).astype(np.float32) + 0.2
    m     = Metrics(_make_result(curve, curve), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())

    scal = m._curve_scalar_metrics(curve, curve, suffix="gt")

    assert scal["curve_mse_gt"]  == pytest.approx(0.0, abs=1e-10)
    assert scal["curve_mae_gt"]  == pytest.approx(0.0, abs=1e-10)
    assert scal["curve_rmse_gt"] == pytest.approx(0.0, abs=1e-10)
    assert scal["overall_r2_gt"] == pytest.approx(1.0, abs=1e-6)
    assert scal["psnr_db_gt"]    == float("inf")


def test_curve_scalar_metrics_rmse_is_sqrt_mse():
    """Curve RMSE is the square root of curve MSE."""
    rng  = np.random.default_rng(3)
    ref  = rng.random((N_Z, 3, 3)).astype(np.float32)
    pred = ref + 0.1 * rng.standard_normal(ref.shape).astype(np.float32)
    m    = Metrics(_make_result(pred, ref), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())

    scal = m._curve_scalar_metrics(pred, ref, suffix="gt")

    assert scal["curve_rmse_gt"] == pytest.approx(np.sqrt(scal["curve_mse_gt"]))


def test_psnr_constant_reference_is_nan():
    """A constant reference has no dynamic range, so PSNR is NaN."""
    ref  = np.ones((4, 2, 2), dtype=np.float32)
    pred = ref + 0.1

    assert np.isnan(Metrics._psnr(pred, ref))


def test_basic_stats_and_percentiles_exact():
    """Basic statistics and percentiles match the analytic values of a linear ramp."""
    x     = np.arange(0.0, 101.0, dtype=np.float64).reshape(101, 1, 1)
    stats = Metrics._basic_stats(x)
    pcts  = Metrics._percentiles(x)

    assert stats["min"]    == 0.0
    assert stats["max"]    == 100.0
    assert stats["median"] == 50.0
    assert stats["mean"]   == pytest.approx(50.0)
    assert pcts["p50"]     == pytest.approx(50.0)
    assert pcts["p25"]     == pytest.approx(25.0)


def test_pixel_stats_skip_undefined_pixels_and_compute_counts_them():
    """Per-pixel statistics cover the finite entries only, and compute() reports how many R2 pixels were undefined."""
    rng = np.random.default_rng(6)
    gt  = rng.random((N_Z, 3, 3)).astype(np.float32) + 0.1
    gt[:, 0, 0] = 0.0
    pred = gt + 0.01 * rng.standard_normal(gt.shape).astype(np.float32)

    result  = _make_result(pred, gt)
    metrics = Metrics(result, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).compute(param_space=False)
    defined = result.pixel_r2[np.isfinite(result.pixel_r2)]

    assert metrics["pixel_r2_gt_undefined_count"] == 1
    assert metrics["pixel_r2_gt_mean"] == pytest.approx(float(defined.mean()), abs=1e-6)
    assert metrics["pixel_r2_gt_min"]  == pytest.approx(float(defined.min()),  abs=1e-6)
    assert np.isfinite(metrics["pixel_r2_gt_p1"])


def test_pixel_stats_without_finite_pixels_raise():
    """A map with no finite entry raises instead of producing NaN statistics."""
    curves  = np.ones((N_Z, 2, 2), dtype=np.float32)
    metrics = Metrics(_make_result(curves, curves), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())

    with pytest.raises(ValueError, match="no finite pixel"):
        metrics._expand_pixel_stats((("pixel_r2_gt", np.full((2, 2), np.nan, dtype=np.float32)),))


def test_select_pixels_best_worst_random_disjoint():
    """Best and worst selections are sorted and separated, and all selected pixels are distinct."""
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
    """Zero requested pixels return empty selections while the other categories keep their size."""
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
    """Flat indices unravel to the expected (row, column) pairs for the given width."""
    width = 7
    flat  = np.array([0, 6, 7, 20], dtype=np.int64)
    yx    = Metrics._flat_to_yx(flat, width)

    assert list(yx[0]) == [0, 0]
    assert list(yx[1]) == [0, 6]
    assert list(yx[2]) == [1, 0]
    assert list(yx[3]) == [2, 6]


def test_z_metrics_identical_perfect():
    """Per-z MAE and RMSE vanish and R2 is one for identical curves."""
    rng   = np.random.default_rng(5)
    curve = rng.random((N_Z, 4, 4)).astype(np.float32) + 0.3
    m     = Metrics(_make_result(curve, curve), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())

    z = m._z_metrics(curve, curve, suffix="gt")

    assert np.allclose(z["z_mae_gt"],  0.0, atol=1e-6)
    assert np.allclose(z["z_rmse_gt"], 0.0, atol=1e-6)
    assert np.allclose(z["z_r2_gt"],   1.0, atol=1e-5)


def test_active_count_stats_identical_perfect_agreement():
    """Identical parameters give exact component counts, per-slot activity fractions and count accuracy."""
    H, W   = 6, 6
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    params[0] = 1.0
    params[3] = 1.0
    params[6] = 0.0

    res = _make_result(np.zeros((N_Z, H, W), np.float32), np.zeros((N_Z, H, W), np.float32), params_pred=params.copy(), params_gt=params.copy())
    out = Metrics(res, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._active_count_stats()

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
    """Per-slot activity fractions are unchanged by permuting the predicted slots."""
    H, W = 5, 5
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0],  gt[1] = 1.0, -10.0
    gt[3],  gt[4] =  1.0,  10.0
    gt[6],  gt[7] =  1.0,  40.0

    order = [2, 0, 1, 3, 4]
    pred  = _reorder_groups(gt, order)

    out = Metrics(_make_result(np.zeros((N_Z, H, W), np.float32), np.zeros((N_Z, H, W), np.float32), params_pred=pred, params_gt=gt), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._active_count_stats()

    for k in range(N_GAUSSIANS):
        assert out[f"slot_{k}_active_pred_frac"] == pytest.approx(out[f"slot_{k}_active_gt_frac"])


def test_active_count_stats_undercount_when_pred_drops_slot():
    """Dropping a predicted slot is reported as a full undercount with halved mean activity."""
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0] = 1.0
    gt[3] = 1.0

    pred    = gt.copy()
    pred[3] = 0.0

    res = _make_result(np.zeros((N_Z, H, W), np.float32), np.zeros((N_Z, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._active_count_stats()

    assert out["count_under_frac"]       == pytest.approx(1.0)
    assert out["count_exact_frac"]       == pytest.approx(0.0)
    assert out["active_count_pred_mean"] == pytest.approx(1.0)
    assert out["active_frac_pred"]       == pytest.approx(1.0 / N_GAUSSIANS)
    assert out["count_acc_gt2"]          == pytest.approx(0.0)


def _reorder_groups(params: np.ndarray, order: list) -> np.ndarray:
    """Permutes the Gaussian slots of a parameter cube.

    Args:
        params : Parameter cube of shape (3 * n_slots, height, width).
        order  : New slot order, as a permutation of range(n_slots).

    Returns:
        Parameter cube of the same shape with the slot triples reordered.
    """
    K       = len(order)
    grouped = params.reshape(K, 3, *params.shape[1:])
    return grouped[order].reshape(K * 3, *params.shape[1:])


def test_param_population_stats_active_inactive_global_means():
    """Amplitude carries active, inactive and global statistics while centre and width are reported on active pixels only."""
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0, :, :2] = 2.0
    gt[1, :, :2] = 10.0
    gt[2, :, :2] = 4.0

    pred            = gt.copy()
    pred[0, :, :2] += 0.5
    pred[1, :, :2] += 1.0

    res = _make_result(np.zeros((N_Z, H, W), np.float32), np.zeros((N_Z, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._param_population_stats()

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
    """NaN-masked inactive centres and widths leave the active and amplitude statistics finite."""
    H, W = 4, 4
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0, :, :2] = 2.0
    gt[1, :, :2] = 10.0
    gt[2, :, :2] = 4.0

    pred = gt.copy()

    for k in range(N_GAUSSIANS):
        inactive = ~GaussianActivity.is_active(gt[3 * k], 1e-4)
        gt[3 * k + 1][inactive] = np.nan
        gt[3 * k + 2][inactive] = np.nan

    res = _make_result(np.zeros((N_Z, H, W), np.float32), np.zeros((N_Z, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._param_population_stats()

    assert out["slot_0_mu_active_gt_mean"]  == pytest.approx(10.0)
    assert out["slot_0_sig_active_gt_mean"] == pytest.approx(4.0)

    assert out["slot_0_amp_inactive_gt_mean"] == pytest.approx(0.0)
    assert out["slot_0_amp_global_gt_mean"]   == pytest.approx(1.0)

    assert not any(key.endswith("_gt_mean") and np.isnan(value) for key, value in out.items() if key.startswith("slot_0_"))


def test_param_population_stats_pred_is_permutation_invariant():
    """Per-slot parameter statistics are unchanged by permuting the predicted slots."""
    H, W = 5, 5
    gt   = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)

    gt[0], gt[1], gt[2] = 1.0, -10.0, 3.0
    gt[3], gt[4], gt[5] = 1.0,  10.0, 4.0
    gt[6], gt[7], gt[8] = 1.0,  40.0, 5.0

    order = [2, 0, 1, 3, 4]
    pred  = _reorder_groups(gt, order)

    res = _make_result(np.zeros((N_Z, H, W), np.float32), np.zeros((N_Z, H, W), np.float32), params_pred=pred, params_gt=gt)
    out = Metrics(res, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._param_population_stats()

    for k in range(3):
        for param in ("amp", "mu", "sig"):
            assert out[f"slot_{k}_{param}_active_pred_mean"] == pytest.approx(out[f"slot_{k}_{param}_active_gt_mean"])


def _matched(pred, gt, n_gaussians=N_GAUSSIANS, tol=5.0):
    """Computes the Hungarian-matched Gaussian metrics for a prediction and its ground truth.

    Args:
        pred        : Predicted parameter cube of shape (3 * n_gaussians, z, width).
        gt          : Ground-truth parameter cube of the same shape.
        n_gaussians : Number of Gaussian slots.
        tol         : Matching tolerance on the Gaussian centre, in metres.

    Returns:
        Mapping of matched-metric name to value.
    """
    res = _make_result(np.zeros((N_Z, *pred.shape[1:]), np.float32), np.zeros((N_Z, *pred.shape[1:]), np.float32), params_pred=pred, params_gt=gt)
    return Metrics(res, _x_axis(), n_gaussians, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).matched_gaussian_metrics(match_tol=tol)


def test_matched_identical_is_perfect():
    """Identical parameters give zero matched errors with unit recall and precision."""
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
    """Matched errors and recall are unchanged by permuting the predicted slots."""
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
    """A second prediction placed outside the tolerance halves recall and precision on two-component pixels."""
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
    """Matched amplitude MAE and RMSE average only over the matched pairs."""
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


@pytest.mark.dataset
def test_metrics_identical_scene_parameter_window_is_perfect(parameters):
    """Curves rendered from scene parameters score perfectly against themselves."""
    params      = np.asarray(parameters[:, :16, :16], dtype=np.float32)
    z_axis      = _x_axis(40)
    n_gaussians = params.shape[0] // 3
    curve       = _curves_from_params(params, z_axis, n_gaussians)

    res = _make_result(curve, curve, params_pred=params.copy(), params_gt=params.copy())
    m   = Metrics(res, z_axis, n_gaussians, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())

    metrics = m.compute(param_space=True)

    assert metrics["curve_mse_gt"]     == pytest.approx(0.0, abs=1e-8)
    assert metrics["overall_r2_gt"]    == pytest.approx(1.0, abs=1e-5)
    assert metrics["pixel_r2_gt_mean"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.dataset
def test_metrics_perturbed_scene_window_is_worse(parameters):
    """Stronger perturbation of scene-parameter curves raises MSE and lowers R2."""
    rng         = np.random.default_rng(11)
    params      = np.asarray(parameters[:, :16, :16], dtype=np.float32)
    z_axis      = _x_axis(40)
    n_gaussians = params.shape[0] // 3
    gt          = _curves_from_params(params, z_axis, n_gaussians)

    pred_small = gt + 0.001 * rng.standard_normal(gt.shape).astype(np.float32) * gt.std()
    pred_large = gt + 0.200 * rng.standard_normal(gt.shape).astype(np.float32) * gt.std()

    m_small = Metrics(_make_result(pred_small, gt), z_axis, n_gaussians, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())
    m_large = Metrics(_make_result(pred_large, gt), z_axis, n_gaussians, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())

    s = m_small.compute(param_space=False)
    l = m_large.compute(param_space=False)

    assert s["curve_mse_gt"]  < l["curve_mse_gt"]
    assert s["overall_r2_gt"] > l["overall_r2_gt"]




def test_label_quality_perfect_fit_gives_r2_one():
    """Labels identical to the raw cube score a unit R2 map with no pixel below one half."""
    rng    = np.random.default_rng(0)
    curves = rng.uniform(0.5, 2.0, size=(N_Z, 4, 3)).astype(np.float32)
    result = _make_result(curves, curves)

    r2_map, stats = Metrics(result, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).label_quality(curves.copy())

    assert r2_map.shape == (4, 3)
    assert np.allclose(r2_map, 1.0, atol=1e-5)
    assert stats["label_r2_mean"]          == pytest.approx(1.0, abs=1e-5)
    assert stats["label_r2_frac_below_half"] == pytest.approx(0.0)


def test_label_quality_noisier_labels_score_lower():
    """Noisier labels score a lower mean label R2 against the same raw cube."""
    rng   = np.random.default_rng(1)
    raw   = rng.uniform(0.5, 2.0, size=(N_Z, 4, 3)).astype(np.float32)
    clean = raw + rng.normal(0.0, 0.01, size=raw.shape).astype(np.float32)
    noisy = raw + rng.normal(0.0, 0.60, size=raw.shape).astype(np.float32)

    _map_clean, stats_clean = Metrics(_make_result(clean, clean), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).label_quality(raw)
    _map_noisy, stats_noisy = Metrics(_make_result(noisy, noisy), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).label_quality(raw)

    assert stats_clean["label_r2_mean"] > stats_noisy["label_r2_mean"]


def test_label_quality_counts_pixels_with_a_constant_raw_profile():
    """A raw cube pixel with a constant profile yields an undefined label R2 that is counted and left out of the mean."""
    rng = np.random.default_rng(2)
    raw = rng.uniform(0.5, 2.0, size=(N_Z, 4, 3)).astype(np.float32)
    raw[:, 0, 0] = 0.0

    r2_map, stats = Metrics(_make_result(raw.copy(), raw), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).label_quality(raw)

    assert np.isnan(r2_map[0, 0])
    assert stats["label_r2_undefined_count"] == 1
    assert stats["label_r2_mean"] == pytest.approx(1.0, abs=1e-5)


def _slice_cubes(seed: int = 0):
    """Builds a prediction and unrelated labels large enough for a 7-pixel SSIM window on every slice.

    Args:
        seed : Seed of the random generator.

    Returns:
        Tuple of prediction and label cubes, each of shape (N_Z, 12, 10).
    """
    rng  = np.random.default_rng(seed)
    pred = rng.uniform(0.5, 2.0, size=(N_Z, 12, 10)).astype(np.float32)
    gt   = rng.uniform(0.5, 2.0, size=(N_Z, 12, 10)).astype(np.float32)

    return pred, gt


def _all_indices(cube: np.ndarray) -> dict:
    """Returns every slice index of each axis of a cube, keyed as the pipeline passes them."""
    n_z, n_rows, n_cols = cube.shape
    return dict(z_indices=np.arange(n_z), col_indices=np.arange(n_cols), row_indices=np.arange(n_rows))


def test_slice_ssim_counts_and_warns_about_non_finite_slices():
    """Slices whose reference and prediction are both constant score NaN, are counted per axis, left out of the mean and warned about."""
    pred, gt = _slice_cubes(seed=3)

    pred[:, :, :2] = 0.0
    gt[:, :, :2]   = 0.0

    logger  = RecordingLogger()
    metrics = Metrics(_make_result(pred, gt), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=logger)

    out = metrics._slice_ssim(pred, gt, prefix="gt", **_all_indices(pred))

    finite = [out[f"ssim_gt_col_{i}"] for i in range(2, pred.shape[2])]

    assert out["ssim_gt_col_n_nonfinite"] == 2
    assert out["ssim_gt_col_n_scored"]    == pred.shape[2] - 2
    assert out["ssim_gt_col_mean"]        == pytest.approx(float(np.mean(finite)))
    assert out["ssim_gt_z_n_nonfinite"]   == 0
    assert out["ssim_gt_z_n_scored"]      == N_Z
    assert [msg for level, msg in logger.messages if level == "warning"] == [f"SSIM (gt) col: 2 of {pred.shape[2]} slices scored non-finite (degenerate slice, such as a constant reference) and are excluded from ssim_gt_col_mean"]


def test_gt_empty_pixels_are_counted_and_left_out_of_the_nonempty_statistics():
    """Pixels with an all-zero GT profile are counted, keep the headline cosine at 0, and are excluded from the _nonempty statistics."""
    rng  = np.random.default_rng(7)
    gt   = rng.uniform(0.5, 2.0, size=(N_Z, 4, 5)).astype(np.float32)
    pred = gt.copy()

    gt[:, 0, :2] = 0.0

    stats = Metrics(_make_result(pred, gt), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._gt_empty_stats()

    assert stats["n_pixels_gt_empty"]                     == 2
    assert stats["gt_empty_frac"]                         == pytest.approx(2 / 20)
    assert stats["pixel_cosine_gt_nonempty_min"]          == pytest.approx(1.0, abs=1e-5)
    assert stats["pixel_peak_idx_d_gt_nonempty_max"]      == 0.0
    assert stats["pixel_peak_err_units_mean_gt_nonempty"] == 0.0


def test_metrics_stage_warns_about_gt_empty_pixels(tmp_path, monkeypatch):
    """The pipeline's metric stage warns with the empty-GT count and the headline statistics it biases, and stays quiet without empty pixels."""
    config = InferenceConfig(run_directory=tmp_path, output_subdir="metrics", device="cpu", cpu_workers=1)
    meta   = InferenceMetadata(config)
    meta.create_dirs()

    run    = SimpleNamespace(n_gaussians=N_GAUSSIANS, activity_threshold=1e-4, split_name="test", split_region=CropRegion(0, 4, 0, 5), core_region=None, checkpoint_meta={"epoch": 3, "best_epoch": 3, "best_val_loss": 0.1})
    result = SimpleNamespace(params_pred=None, params_gt=None, params_source="model", extract_r2=None, extract_floor=None)

    for n_empty, expected in ((3, 1), (0, 0)):
        monkeypatch.setattr(Metrics, "compute", lambda self, **kwargs: {"n_pixels": 20, "n_pixels_gt_empty": n_empty, "gt_empty_frac": n_empty / 20})
        logger = RecordingLogger()

        InferencePipeline(config)._evaluate_metrics(config, result, _x_axis(), run, meta, {"all_z_idx": None, "all_col_idx": None, "all_row_idx": None}, logger)

        warnings = [msg for level, msg in logger.messages if level == "warning"]

        assert len(warnings) == expected
        if expected:
            assert warnings[0].startswith("3 of 20 pixels (15.00%) have an empty GT profile")


def test_label_quality_stage_opens_its_own_section(tmp_path, monkeypatch):
    """The label-quality stage opens its own section before scoring, so its time is not filed under the previous step."""
    config = InferenceConfig(run_directory=tmp_path, output_subdir="labels", device="cpu", cpu_workers=1, save_cubes=False)
    meta   = InferenceMetadata(config)
    meta.create_dirs()

    monkeypatch.setattr(Metrics, "label_quality", lambda self, full_curves: (np.ones((2, 2)), {"label_r2_mean": 0.9, "label_r2_frac_below_half": 0.1}))

    run    = SimpleNamespace(n_gaussians=N_GAUSSIANS, activity_threshold=1e-4, full_curves=None)
    result = SimpleNamespace()
    logger = RecordingLogger()

    InferencePipeline(config)._evaluate_label_quality(config, meta, run, result, _x_axis(), {}, logger)

    assert logger.messages[0] == ("section", "[Inference: Label Quality]")


def test_gt_empty_split_raises():
    """A split whose every GT profile is empty has nothing to score and raises."""
    gt = np.zeros((N_Z, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="empty ground-truth profile"):
        Metrics(_make_result(gt + 1.0, gt), _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger())._gt_empty_stats()


def test_label_quality_shape_mismatch_raises():
    """A raw cube whose spatial shape differs from the labels raises ValueError."""
    curves = np.ones((N_Z, 4, 3), dtype=np.float32)
    result = _make_result(curves, curves)

    with pytest.raises(ValueError):
        Metrics(result, _x_axis(), N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).label_quality(np.ones((N_Z, 5, 3), dtype=np.float32))


def test_roughness_map_is_zero_for_a_linear_ramp():
    """A linear z ramp has zero second-difference roughness."""
    ramp = np.linspace(0.0, 1.0, N_Z, dtype=np.float32).reshape(-1, 1, 1) * np.ones((1, 3, 2), dtype=np.float32)

    assert Metrics.roughness_map(ramp).shape == (3, 2)
    assert np.allclose(Metrics.roughness_map(ramp), 0.0, atol=1e-10)


def test_roughness_map_grows_with_bin_to_bin_jitter():
    """Alternating bin-to-bin jitter is rougher than a flat profile."""
    smooth = np.zeros((N_Z, 2, 2), dtype=np.float32)
    jitter = np.tile(np.array([1.0, -1.0], dtype=np.float32), N_Z // 2).reshape(-1, 1, 1) * np.ones((1, 2, 2), dtype=np.float32)

    assert Metrics.roughness_map(jitter).mean() > Metrics.roughness_map(smooth).mean()


def test_roughness_metrics_report_unit_ratio_for_identical_curves():
    """Identical curves give a roughness ratio of one and no excess-roughness pixels."""
    z_axis = _x_axis()
    params = _gaussian_params(seed=1)
    curves = _curves_from_params(params, z_axis, N_GAUSSIANS)

    metrics = Metrics(_make_result(curves.copy(), curves), z_axis, N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).compute(param_space=False)

    assert metrics["roughness_ratio"]       == pytest.approx(1.0, rel=1e-6)
    assert metrics["roughness_excess_frac"] == pytest.approx(0.0)


def test_roughness_ratio_exceeds_one_when_the_prediction_carries_jitter():
    """Adding z jitter to the prediction drives the roughness ratio and excess fraction up."""
    z_axis = _x_axis()
    params = _gaussian_params(seed=2)
    gt     = _curves_from_params(params, z_axis, N_GAUSSIANS)

    jitter = np.tile(np.array([1.0, -1.0], dtype=np.float32), N_Z // 2).reshape(-1, 1, 1)
    pred   = gt + 0.2 * jitter

    metrics = Metrics(_make_result(pred, gt), z_axis, N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).compute(param_space=False)

    assert metrics["roughness_ratio"] > 2.0
    assert metrics["roughness_excess_frac"] > 0.5


def test_slot_organization_can_be_switched_off():
    """Disabling slot organisation drops its metrics while the matched metrics remain."""
    z_axis = _x_axis()
    params = _gaussian_params(seed=4)
    curves = _curves_from_params(params, z_axis, N_GAUSSIANS)
    result = _make_result(curves.copy(), curves, params_pred=params.copy(), params_gt=params)

    with_slots    = Metrics(result, z_axis, N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).compute(param_space=True, slot_organization=True)
    without_slots = Metrics(result, z_axis, N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).compute(param_space=True, slot_organization=False)

    assert "slot_usage_entropy" in with_slots
    assert "slot_usage_entropy" not in without_slots
    assert "matched_mu_mae"     in without_slots


def _legacy_cubes(seed: int = 0, n_slices: int = 3) -> tuple:
    """Draws an estimate/reference cube pair whose slices hold the 27-pixel legacy window.

    Args:
        seed     : Seed of the random generator.
        n_slices : Extent of the last cube axis.

    Returns:
        Tuple of estimate and reference cubes, each of shape (32, 32, n_slices).
    """
    rng      = np.random.default_rng(seed)
    estimate = rng.uniform(0.0, 1.0, (32, 32, n_slices))

    return estimate, estimate + 0.1 * rng.standard_normal((32, 32, n_slices))


def _identity_worker(payload):
    """Returns its payload, standing in for a slice scorer inside the pool."""
    return payload


def test_bounded_pool_tracks_the_sweep_on_the_run_logger():
    """The slice pool draws its progress through the run logger and records how many slices each sweep scored."""
    logger = RecordingLogger()
    out    = BoundedProcessPool(2, logger).results(_identity_worker, iter(range(5)), 5, "SSIM (gt)")

    assert out == [0, 1, 2, 3, 4]
    assert ("info", "SSIM (gt): 5 slices scored") in logger.messages
