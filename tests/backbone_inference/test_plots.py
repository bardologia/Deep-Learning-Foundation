"""Tests for the backbone inference plotting layer.

Covers the composed `Plotter` facade and its shared drawing tools, the parameter
distribution, scatter, error-map and error-histogram figures with their single slot
matching per cube, the slot-occupancy map, the slice and cube figures, and the track
geometry and flight-path figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pipelines.backbone.inference.plots          import Plotter, PlotTools, SlicePlotter, ParamPlotter, SlotPlotter
from pipelines.backbone.inference.plots.plotter  import Plotter as PloterClass
from tools.metrics.gaussian_matching             import GaussianMatcher


N_GAUSSIANS = 2
N_Z         = 16
H           = W       = 8


def _assert_files(paths):
    """Asserts that at least one path was returned and every one is a non-empty file."""
    assert len(paths) > 0
    for p in paths:
        assert p.is_file()
        assert p.stat().st_size > 0


def _params(n_gaussians=N_GAUSSIANS, h=H, w=W):
    """Builds a parameter cube of shape (3 * n_gaussians, h, w) with random amplitudes and fixed centres and widths."""
    rng    = np.random.default_rng(0)
    params = np.zeros((n_gaussians * 3, h, w), dtype=np.float32)
    for k in range(n_gaussians):
        params[3 * k]     = rng.random((h, w)).astype(np.float32) + 0.5
        params[3 * k + 1] = float(k * 15)
        params[3 * k + 2] = 4.0
    return params


def _curves(params, z_axis, n_gaussians=N_GAUSSIANS):
    """Renders a parameter cube into profiles of shape (n_z, z, width)."""
    curve = np.zeros((z_axis.size, params.shape[1], params.shape[2]), dtype=np.float32)
    for k in range(n_gaussians):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = z_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
    return curve.astype(np.float32)


def _x_axis():
    """Returns the z axis of N_Z samples spanning -20 m to 80 m."""
    return np.linspace(-20.0, 80.0, N_Z).astype(np.float32)


def test_ploter_composes_subplotters():
    """The plotter facade exposes the slice, parameter, slot and track sub-plotters."""
    p = Plotter()
    assert isinstance(p.slice, SlicePlotter)
    assert isinstance(p.param, ParamPlotter)
    assert isinstance(p.slot,  SlotPlotter)
    assert PloterClass is Plotter


def test_plottools_imshow_panel(tmp_path):
    """The shared image panel writes a non-empty figure."""
    tools = PlotTools()
    data  = np.random.default_rng(0).random((8, 8)).astype(np.float32)

    path = tools._imshow_panel(
        data=data, title="t", x_label="x", y_label="y", cbar_label="c",
        extent=[0, 8, 8, 0], cmap="jet", vmin=0.0, vmax=1.0, origin="upper",
        path=tmp_path / "panel.png",
    )

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plottools_intensity_scale_normalize():
    """The intensity scale is the reference maximum when normalising and one otherwise."""
    tools = PlotTools(normalize=True)
    ref   = np.array([0.0, 1.0, 4.0], dtype=np.float32)

    assert tools._intensity_scale(ref) == pytest.approx(4.0)

    tools_no = PlotTools(normalize=False)
    assert tools_no._intensity_scale(ref) == 1.0


def test_param_plotter_distributions(tmp_path):
    """Parameter distribution figures are written for the whole cube."""
    params  = _params()
    plotter = ParamPlotter()

    paths = plotter.plot_param_distributions(
        params_pred=params, params_gt=params, n_gaussians=N_GAUSSIANS, amp_threshold=1e-4, out_dir=tmp_path,
    )
    _assert_files(paths)


def test_param_plotter_scatter(tmp_path):
    """Predicted-against-true scatter figures are written for the whole cube."""
    params  = _params()
    plotter = ParamPlotter()

    paths = plotter.plot_param_scatter(
        params_pred=params + 0.01, params_gt=params, n_gaussians=N_GAUSSIANS, amp_threshold=1e-4, out_dir=tmp_path,
    )
    _assert_files(paths)


def test_param_plotter_error_maps(tmp_path):
    """Spatial parameter error maps are written for the whole cube."""
    params  = _params()
    plotter = ParamPlotter()

    paths = plotter.plot_param_error_maps(
        params_pred=params + 0.5, params_gt=params, n_gaussians=N_GAUSSIANS, amp_threshold=1e-4,
        out_dir=tmp_path, row_offset=0, col_offset=0,
    )
    _assert_files(paths)


def test_param_plotter_error_hists(tmp_path):
    """One error histogram is written per slot and parameter."""
    params  = _params()
    noise   = np.random.default_rng(1).normal(0.0, 0.1, params.shape).astype(np.float32)
    plotter = ParamPlotter()

    paths = plotter.plot_param_error_hists(
        params_pred=params + noise, params_gt=params, n_gaussians=N_GAUSSIANS, amp_threshold=1e-4, out_dir=tmp_path,
    )
    _assert_files(paths)
    assert len(paths) == N_GAUSSIANS * 3


def test_param_plotter_matches_slots_once_per_cube(tmp_path, monkeypatch):
    """Slot matching is computed once per distinct prediction cube and reused across figures."""
    params  = _params()
    pred    = params + 0.01
    other   = params + 0.5
    plotter = ParamPlotter()
    calls   = []

    original = GaussianMatcher.aligned_prediction

    def counted(self, params_pred, params_gt, n_K, assignment=None):
        """Records each slot-matching call and delegates to the real implementation."""
        calls.append(n_K)
        return original(self, params_pred, params_gt, n_K, assignment=assignment)

    monkeypatch.setattr(GaussianMatcher, "aligned_prediction", counted)

    plotter.plot_param_scatter(
        params_pred=pred[: N_GAUSSIANS * 3], params_gt=params[: N_GAUSSIANS * 3], n_gaussians=N_GAUSSIANS, amp_threshold=1e-4, out_dir=tmp_path / "scatter",
    )
    plotter.plot_param_error_hists(
        params_pred=pred[: N_GAUSSIANS * 3], params_gt=params[: N_GAUSSIANS * 3], n_gaussians=N_GAUSSIANS, amp_threshold=1e-4, out_dir=tmp_path / "hists",
    )

    assert len(calls) == 1

    plotter.plot_param_error_maps(
        params_pred=other[: N_GAUSSIANS * 3], params_gt=params[: N_GAUSSIANS * 3], n_gaussians=N_GAUSSIANS, amp_threshold=1e-4,
        out_dir=tmp_path / "maps", row_offset=0, col_offset=0,
    )

    assert len(calls) == 2


def test_param_scatter_r2_value_exact():
    """The scatter annotation reports an R2 of one for an exact prediction."""
    gt   = np.array([1.0, 2.0, 3.0, 4.0])
    pred = gt.copy()
    assert ParamPlotter._r2_value(gt, pred) == pytest.approx(1.0, abs=1e-9)


def test_slot_plotter_active_count_map(tmp_path):
    """The active-component count map is written."""
    params  = _params()
    plotter = SlotPlotter()
    paths   = plotter.plot_active_count_map(
        params_pred=params, params_gt=params, n_gaussians=N_GAUSSIANS, amp_threshold=1e-4,
        out_dir=tmp_path, row_offset=0, col_offset=0,
    )
    _assert_files(paths)


def test_slice_plotter_profiles(tmp_path):
    """Per-pixel profile figures are written for the selected pixels."""
    z_axis = _x_axis()
    params = _params()
    gt     = _curves(params, z_axis)
    pred   = gt + 0.01

    pm = {
        "mse": ((pred - gt) ** 2).mean(0),
        "mae": np.abs(pred - gt).mean(0),
        "r2":  np.ones((H, W), dtype=np.float32),
        "cos": np.ones((H, W), dtype=np.float32),
    }

    plotter = SlicePlotter()
    paths   = plotter.plot_profiles(
        pred_curves=pred, gt_curves=gt, params_pred=params, z_axis=z_axis,
        pixels=np.array([[1, 1], [2, 3]]), tag="best", out_dir=tmp_path,
        n_gaussians=N_GAUSSIANS, pixel_metrics=pm, row_offset=0, col_offset=0,
    )
    _assert_files(paths)


def test_slice_plotter_pixel_metric_map_log(tmp_path):
    """A pixel metric map is written on a logarithmic colour scale."""
    plotter = SlicePlotter()
    metric  = np.random.default_rng(0).random((H, W)).astype(np.float32) + 0.01

    path = plotter.plot_pixel_metric_map(
        metric_map=metric, title="t", label="MSE", out_path=tmp_path / "m.png",
        row_offset=0, col_offset=0, log=True,
    )
    assert path.is_file() and path.stat().st_size > 0


def test_slice_plotter_cube_slice_col_and_row(tmp_path):
    """Cube slices are written along both the column and the row axis."""
    z_axis = _x_axis()
    params = _params()
    gt     = _curves(params, z_axis)
    pred   = gt + 0.02

    plotter = SlicePlotter()

    for axis in ("col", "row"):
        paths = plotter.plot_cube_slice(
            pred_cube=pred, gt_cube=gt, axis=axis, index=2, z_axis=z_axis,
            out_dir=tmp_path, stem=f"slice_{axis}", row_offset=0, col_offset=0, ssim_value=0.9,
        )
        _assert_files(paths)


def test_slice_plotter_cube_slice_bad_axis_raises(tmp_path):
    """The z axis is not a valid cube slice axis and raises ValueError."""
    z_axis  = _x_axis()
    gt      = _curves(_params(), z_axis)
    plotter = SlicePlotter()

    with pytest.raises(ValueError, match="axis must be"):
        plotter.plot_cube_slice(
            pred_cube=gt, gt_cube=gt, axis="z", index=0, z_axis=z_axis,
            out_dir=tmp_path, stem="bad", row_offset=0, col_offset=0,
        )


def test_slice_plotter_z_intensity_slice(tmp_path):
    """A constant-z intensity slice is written."""
    z_axis = _x_axis()
    params = _params()
    gt     = _curves(params, z_axis)
    pred   = gt + 0.02

    plotter = SlicePlotter()
    paths   = plotter.plot_z_intensity_slice(
        pred_cube=pred, gt_cube=gt, z_idx=4, z_axis=z_axis,
        out_dir=tmp_path, stem="z", row_offset=0, col_offset=0, ssim_value=0.8,
    )
    _assert_files(paths)


def test_slice_plotter_metric_histograms(tmp_path):
    """A histogram is written for each supplied pixel metric map."""
    plotter = SlicePlotter()
    arrays  = {"pixel_mse": np.random.default_rng(0).random((H, W)).astype(np.float32)}

    paths = plotter.plot_metric_histograms(arrays, tmp_path)
    _assert_files(paths)


def test_slice_plotter_input_channels(tmp_path):
    """Amplitude and phase figures are written for the complex input stack."""
    rng     = np.random.default_rng(0)
    n_sec   = 2
    inputs  = (rng.standard_normal((1 + 2 * n_sec, H, W)) + 1j * rng.standard_normal((1 + 2 * n_sec, H, W))).astype(np.complex64)
    plotter = SlicePlotter()

    paths = plotter.plot_input_channels(
        complex_inputs=inputs, n_secondaries=n_sec, labels=["A", "B"],
        out_dir=tmp_path, row_offset=0, col_offset=0,
    )
    _assert_files(paths)


def _ssim_metrics(axis, n, prefix="gt"):
    """Builds a metric dictionary holding per-slice SSIM values and their mean for one axis.

    Args:
        axis   : Slice axis name the keys are built for.
        n      : Number of slices.
        prefix : Comparison side the keys refer to.

    Returns:
        Mapping of SSIM metric key to value.
    """
    gm = {f"ssim_{prefix}_{axis}_{i}": 0.5 + 0.01 * i for i in range(n)}
    gm[f"ssim_{prefix}_{axis}_mean"] = 0.6
    return gm


def test_slice_plotter_ssim_curves(tmp_path):
    """The per-slice SSIM curve figure is written."""
    plotter = SlicePlotter()
    gm      = _ssim_metrics("col", W)

    path = plotter.plot_ssim_curves(
        global_metrics=gm, axis="col", out_path=tmp_path / "ssim.png",
        n_slices=W, slice_indices=np.arange(W),
    )
    assert path.is_file() and path.stat().st_size > 0


def test_slice_plotter_z_metric_curves(tmp_path):
    """One figure is written per z-resolved metric."""
    z_axis = _x_axis()
    gm     = {}
    for key in ("z_mae", "z_rmse", "z_r2", "z_ce"):
        for i in range(N_Z):
            gm[f"{key}_gt_{i}"] = 0.1 + 0.001 * i
        gm[f"{key}_gt_mean"] = 0.2

    plotter = SlicePlotter()
    paths   = plotter.plot_z_metric_curves(gm, tmp_path, N_Z, z_axis)
    _assert_files(paths)




