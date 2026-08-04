from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pipelines.backbone.inference.plots          import Plotter, PlotTools, SlicePlotter, ParamPlotter, SlotPlotter, TrackPlotter
from pipelines.backbone.inference.plots.plotter  import Plotter as PloterClass
from tools.baselines.containers                  import TrackBaselines, TrackProfiles
from tools.metrics.gaussian_matching             import GaussianMatcher


N_GAUSSIANS = 2
N_ELEV      = 16
H = W       = 8


def _assert_files(paths):
    assert len(paths) > 0
    for p in paths:
        assert p.is_file()
        assert p.stat().st_size > 0


def _params(n_gaussians=N_GAUSSIANS, h=H, w=W):
    rng    = np.random.default_rng(0)
    params = np.zeros((n_gaussians * 3, h, w), dtype=np.float32)
    for k in range(n_gaussians):
        params[3 * k]     = rng.random((h, w)).astype(np.float32) + 0.5
        params[3 * k + 1] = float(k * 15)
        params[3 * k + 2] = 4.0
    return params


def _curves(params, x_axis, n_gaussians=N_GAUSSIANS):
    curve = np.zeros((x_axis.size, params.shape[1], params.shape[2]), dtype=np.float32)
    for k in range(n_gaussians):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = x_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
    return curve.astype(np.float32)


def _x_axis():
    return np.linspace(-20.0, 80.0, N_ELEV).astype(np.float32)


def test_ploter_composes_subplotters():
    p = Plotter()
    assert isinstance(p.slice, SlicePlotter)
    assert isinstance(p.param, ParamPlotter)
    assert isinstance(p.slot,  SlotPlotter)
    assert isinstance(p.track, TrackPlotter)
    assert PloterClass is Plotter


def test_plottools_imshow_panel(tmp_path):
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
    tools = PlotTools(normalize=True)
    ref   = np.array([0.0, 1.0, 4.0], dtype=np.float32)

    assert tools._intensity_scale(ref) == pytest.approx(4.0)

    tools_no = PlotTools(normalize=False)
    assert tools_no._intensity_scale(ref) == 1.0


def test_param_plotter_distributions(tmp_path):
    params  = _params()
    plotter = ParamPlotter()

    paths = plotter.plot_param_distributions(
        params_pred=params, params_gt=params, n_gaussians=N_GAUSSIANS, out_dir=tmp_path,
    )
    _assert_files(paths)


def test_param_plotter_scatter(tmp_path):
    params  = _params()
    plotter = ParamPlotter()

    paths = plotter.plot_param_scatter(
        params_pred=params + 0.01, params_gt=params, n_gaussians=N_GAUSSIANS, out_dir=tmp_path,
    )
    _assert_files(paths)


def test_param_plotter_error_maps(tmp_path):
    params  = _params()
    plotter = ParamPlotter()

    paths = plotter.plot_param_error_maps(
        params_pred=params + 0.5, params_gt=params, n_gaussians=N_GAUSSIANS,
        out_dir=tmp_path, az_offset=0, rg_offset=0,
    )
    _assert_files(paths)


def test_param_plotter_error_hists(tmp_path):
    params  = _params()
    noise   = np.random.default_rng(1).normal(0.0, 0.1, params.shape).astype(np.float32)
    plotter = ParamPlotter()

    paths = plotter.plot_param_error_hists(
        params_pred=params + noise, params_gt=params, n_gaussians=N_GAUSSIANS, out_dir=tmp_path,
    )
    _assert_files(paths)
    assert len(paths) == N_GAUSSIANS * 3


def test_param_plotter_matches_slots_once_per_cube(tmp_path, monkeypatch):
    params  = _params()
    pred    = params + 0.01
    other   = params + 0.5
    plotter = ParamPlotter()
    calls   = []

    original = GaussianMatcher.aligned_prediction

    def counted(self, params_pred, params_gt, n_K, assignment=None):
        calls.append(n_K)
        return original(self, params_pred, params_gt, n_K, assignment=assignment)

    monkeypatch.setattr(GaussianMatcher, "aligned_prediction", counted)

    plotter.plot_param_scatter(
        params_pred=pred[: N_GAUSSIANS * 3], params_gt=params[: N_GAUSSIANS * 3], n_gaussians=N_GAUSSIANS, out_dir=tmp_path / "scatter",
    )
    plotter.plot_param_error_hists(
        params_pred=pred[: N_GAUSSIANS * 3], params_gt=params[: N_GAUSSIANS * 3], n_gaussians=N_GAUSSIANS, out_dir=tmp_path / "hists",
    )

    assert len(calls) == 1

    plotter.plot_param_error_maps(
        params_pred=other[: N_GAUSSIANS * 3], params_gt=params[: N_GAUSSIANS * 3], n_gaussians=N_GAUSSIANS,
        out_dir=tmp_path / "maps", az_offset=0, rg_offset=0,
    )

    assert len(calls) == 2


def test_param_scatter_r2_value_exact():
    gt   = np.array([1.0, 2.0, 3.0, 4.0])
    pred = gt.copy()
    assert ParamPlotter._r2_value(gt, pred) == pytest.approx(1.0, abs=1e-9)


def test_slot_plotter_active_count_map(tmp_path):
    params  = _params()
    plotter = SlotPlotter()
    paths   = plotter.plot_active_count_map(
        params_pred=params, params_gt=params, n_gaussians=N_GAUSSIANS,
        out_dir=tmp_path, az_offset=0, rg_offset=0,
    )
    _assert_files(paths)


def test_slice_plotter_profiles(tmp_path):
    x_axis = _x_axis()
    params = _params()
    gt     = _curves(params, x_axis)
    pred   = gt + 0.01

    pm = {
        "mse": ((pred - gt) ** 2).mean(0),
        "mae": np.abs(pred - gt).mean(0),
        "r2":  np.ones((H, W), dtype=np.float32),
        "cos": np.ones((H, W), dtype=np.float32),
    }

    plotter = SlicePlotter()
    paths   = plotter.plot_profiles(
        pred_curves=pred, gt_curves=gt, params_pred=params, x_axis=x_axis,
        pixels=np.array([[1, 1], [2, 3]]), tag="best", out_dir=tmp_path,
        n_gaussians=N_GAUSSIANS, pixel_metrics=pm, az_offset=0, rg_offset=0,
    )
    _assert_files(paths)


def test_slice_plotter_pixel_metric_map_log(tmp_path):
    plotter = SlicePlotter()
    metric  = np.random.default_rng(0).random((H, W)).astype(np.float32) + 0.01

    path = plotter.plot_pixel_metric_map(
        metric_map=metric, title="t", label="MSE", out_path=tmp_path / "m.png",
        az_offset=0, rg_offset=0, log=True,
    )
    assert path.is_file() and path.stat().st_size > 0


def test_slice_plotter_tomogram_slice_range_and_azimuth(tmp_path):
    x_axis = _x_axis()
    params = _params()
    gt     = _curves(params, x_axis)
    pred   = gt + 0.02

    plotter = SlicePlotter()

    for axis in ("range", "azimuth"):
        paths = plotter.plot_tomogram_slice(
            pred_cube=pred, gt_cube=gt, axis=axis, index=2, x_axis=x_axis,
            out_dir=tmp_path, stem=f"slice_{axis}", az_offset=0, rg_offset=0, ssim_value=0.9,
        )
        _assert_files(paths)


def test_slice_plotter_tomogram_slice_bad_axis_raises(tmp_path):
    x_axis  = _x_axis()
    gt      = _curves(_params(), x_axis)
    plotter = SlicePlotter()

    with pytest.raises(ValueError, match="axis must be"):
        plotter.plot_tomogram_slice(
            pred_cube=gt, gt_cube=gt, axis="elevation", index=0, x_axis=x_axis,
            out_dir=tmp_path, stem="bad", az_offset=0, rg_offset=0,
        )


def test_slice_plotter_elevation_intensity_slice(tmp_path):
    x_axis = _x_axis()
    params = _params()
    gt     = _curves(params, x_axis)
    pred   = gt + 0.02

    plotter = SlicePlotter()
    paths   = plotter.plot_elevation_intensity_slice(
        pred_cube=pred, gt_cube=gt, elev_idx=4, x_axis=x_axis,
        out_dir=tmp_path, stem="elev", az_offset=0, rg_offset=0, ssim_value=0.8,
    )
    _assert_files(paths)


def test_slice_plotter_metric_histograms(tmp_path):
    plotter = SlicePlotter()
    arrays  = {"pixel_mse": np.random.default_rng(0).random((H, W)).astype(np.float32)}

    paths = plotter.plot_metric_histograms(arrays, tmp_path)
    _assert_files(paths)


def test_slice_plotter_input_channels(tmp_path):
    rng     = np.random.default_rng(0)
    n_sec   = 2
    inputs  = (rng.standard_normal((1 + 2 * n_sec, H, W)) + 1j * rng.standard_normal((1 + 2 * n_sec, H, W))).astype(np.complex64)
    plotter = SlicePlotter()

    paths = plotter.plot_input_channels(
        complex_inputs=inputs, n_secondaries=n_sec, labels=["A", "B"],
        out_dir=tmp_path, az_offset=0, rg_offset=0,
    )
    _assert_files(paths)


def _ssim_metrics(axis, n, prefix="gt"):
    gm = {f"ssim_{prefix}_{axis}_{i}": 0.5 + 0.01 * i for i in range(n)}
    gm[f"ssim_{prefix}_{axis}_mean"] = 0.6
    return gm


def test_slice_plotter_ssim_curves(tmp_path):
    plotter = SlicePlotter()
    gm      = _ssim_metrics("range", W)

    path = plotter.plot_ssim_curves(
        global_metrics=gm, axis="range", out_path=tmp_path / "ssim.png",
        n_slices=W, slice_indices=np.arange(W),
    )
    assert path.is_file() and path.stat().st_size > 0


def test_slice_plotter_elev_metric_curves(tmp_path):
    x_axis = _x_axis()
    gm     = {}
    for key in ("elev_mae", "elev_rmse", "elev_r2", "elev_ce"):
        for i in range(N_ELEV):
            gm[f"{key}_gt_{i}"] = 0.1 + 0.001 * i
        gm[f"{key}_gt_mean"] = 0.2

    plotter = SlicePlotter()
    paths   = plotter.plot_elev_metric_curves(gm, tmp_path, N_ELEV, x_axis)
    _assert_files(paths)


@pytest.mark.real_data
def test_track_plotter_geometry(tmp_path, baselines_json):
    baselines = TrackBaselines.from_payload(baselines_json)
    plotter   = TrackPlotter()

    path = plotter.plot_track_geometry(baselines, tmp_path / "geom.png")
    assert path.is_file() and path.stat().st_size > 0


@pytest.mark.real_data
def test_track_plotter_profiles_and_flight(tmp_path, track_profiles):
    profiles = TrackProfiles(
        labels        = [str(label) for label in track_profiles["labels"]],
        horizontal    = np.asarray(track_profiles["horizontal"], dtype=float),
        vertical      = np.asarray(track_profiles["vertical"],   dtype=float),
        azimuth_start = int(track_profiles["azimuth_start"]),
    )
    plotter = TrackPlotter()

    paths = plotter.plot_track_profiles(profiles, tmp_path, split_azimuth=(1000, 1500))
    _assert_files(paths)

    flight = plotter.plot_track_flight_3d(profiles, tmp_path / "flight.png")
    assert flight.is_file() and flight.stat().st_size > 0
