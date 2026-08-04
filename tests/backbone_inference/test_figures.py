from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from configuration.inference                         import InferenceConfig
from tools.data.regions                              import CropRegion
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics            import Metrics, Result
from pipelines.backbone.inference.figures            import Animator, FigureComposer
from pipelines.backbone.inference.plots              import Plotter
from tools.baselines                                 import TrackBaselines, TrackProfiles

from tests.conftest import SilentLogger


N_GAUSSIANS = 2
N_ELEV      = 12
H = W       = 8


def _x_axis():
    return np.linspace(-20.0, 80.0, N_ELEV).astype(np.float32)


def _params():
    rng    = np.random.default_rng(0)
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)
    for k in range(N_GAUSSIANS):
        params[3 * k]     = rng.random((H, W)).astype(np.float32) + 0.5
        params[3 * k + 1] = float(k * 15)
        params[3 * k + 2] = 4.0
    return params


def _curves(params, x_axis):
    curve = np.zeros((x_axis.size, H, W), dtype=np.float32)
    for k in range(N_GAUSSIANS):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = x_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
    return curve.astype(np.float32)


def _result_and_metrics():
    x_axis = _x_axis()
    params = _params()
    gt     = _curves(params, x_axis)
    pred   = gt + 0.01 * np.random.default_rng(1).standard_normal(gt.shape).astype(np.float32)

    pixel = Metrics.curve_pixel_metrics(pred, gt)
    res   = Result(
        pred_curves        = pred,
        gt_curves          = gt,
        pixel_mse          = pixel["mse"],
        pixel_mae          = pixel["mae"],
        pixel_r2           = pixel["r2"],
        pixel_cosine       = pixel["cos"],
        pixel_peak_err_idx = pixel["peak"].astype(np.int32),
        cube_directory     = None,
        azimuth_offset     = 0,
        range_offset       = 0,
        params_pred        = params.copy(),
        params_gt          = params.copy(),
    )

    indices = {
        "slice_elev_idx"  : np.array([2, 6]),
        "slice_range_idx" : np.array([2, 5]),
        "slice_az_idx"    : np.array([2, 5]),
        "all_elev_idx"    : np.arange(N_ELEV),
        "all_range_idx"   : np.arange(W),
        "all_az_idx"      : np.arange(H),
    }

    gm = Metrics(res, x_axis, N_GAUSSIANS).compute(
        elev_indices  = indices["all_elev_idx"],
        range_indices = indices["all_range_idx"],
        az_indices    = indices["all_az_idx"],
        param_space   = True,
    )

    return res, gm, indices, x_axis


def _track_baselines():
    return TrackBaselines(
        labels         = ["PS02", "PS04", "PS06"],
        vertical       = [0.0, 4.0, 8.0],
        horizontal     = [0.0, 1.0, 2.0],
        vertical_std   = [0.0, 0.2, 0.3],
        horizontal_std = [0.0, 0.1, 0.2],
    )


def _track_profiles():
    along = np.linspace(0.0, 1.0, H, dtype=np.float32)
    return TrackProfiles(
        labels        = ["PS02", "PS04", "PS06"],
        horizontal    = np.stack([along * k       for k in range(3)]).astype(np.float32),
        vertical      = np.stack([along * k + 4.0 * k for k in range(3)]).astype(np.float32),
        azimuth_start = 0,
    )


def _run_stub():
    region = CropRegion(azimuth_start=0, azimuth_end=H, range_start=0, range_end=W)
    return types.SimpleNamespace(
        n_gaussians      = N_GAUSSIANS,
        split_region     = region,
        full_curves      = _curves(_params(), _x_axis()),
        track_baselines  = _track_baselines(),
        track_profiles   = _track_profiles(),
        complex_inputs   = None,
        n_secondaries    = 0,
        secondary_labels = None,
    )


def _composer(tmp_path):
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="fig", device="cpu", normalize_intensity=False)
    meta = InferenceMetadata(cfg)
    meta.create_dirs()
    plotter = Plotter(normalize=False)
    return FigureComposer(plotter=plotter, meta=meta, logger=SilentLogger(), cfg=cfg), meta


def test_animator_build_axis_bad_axis():
    animator = Animator(SilentLogger())
    cubes    = (np.zeros((N_ELEV, H, W), np.float32), np.zeros((N_ELEV, H, W), np.float32))
    with pytest.raises(ValueError, match="axis must be"):
        animator._build_axis("bogus", cubes, _x_axis(), 0, 0)


def test_animator_rejects_non_positive_max_frames():
    with pytest.raises(ValueError, match="max_frames"):
        Animator(SilentLogger(), max_frames=0)


@pytest.mark.slow
def test_animator_walk_gif(tmp_path):
    x_axis = _x_axis()
    params = _params()
    gt     = _curves(params, x_axis)
    pred   = gt + 0.05

    animator = Animator(SilentLogger(), max_frames=4, num_workers=1)
    out      = animator.walk_gif(
        pred_cube=pred, gt_cube=gt, axis="elevation",
        out_path=tmp_path / "walk.gif", x_axis=x_axis, az_offset=0, rg_offset=0,
    )

    assert out.is_file()
    assert out.stat().st_size > 0


@pytest.mark.slow
def test_figure_composer_compose_creates_files(tmp_path):
    res, gm, indices, x_axis = _result_and_metrics()
    composer, meta = _composer(tmp_path)
    run            = _run_stub()

    figure_paths = composer.compose(
        result=res, run=run, global_metrics=gm, x_axis_np=x_axis, indices=indices, param_space=True,
    )

    assert "profiles_best"      in figure_paths
    assert "pixel_mse_map"      in figure_paths
    assert "param_distributions" in figure_paths
    assert "param_error_hists"  in figure_paths
    assert "active_count_map"   in figure_paths
    assert "slices_range"       in figure_paths
    assert "ssim_range"         in figure_paths

    flat = [p for paths in figure_paths.values() for p in paths]
    assert len(flat) > 0
    for p in flat:
        assert p.is_file()
        assert p.stat().st_size > 0


@pytest.mark.slow
def test_figure_composer_compose_with_reduced(tmp_path):
    res, gm, indices, x_axis = _result_and_metrics()

    m    = Metrics(res, x_axis, N_GAUSSIANS)
    comp = m.reduced_comparison(
        res.gt_curves + 0.02,
        elev_indices  = indices["all_elev_idx"],
        range_indices = indices["all_range_idx"],
        az_indices    = indices["all_az_idx"],
    )
    res.reduced = comp
    gm.update(comp.metrics)

    composer, meta = _composer(tmp_path)
    run            = _run_stub()

    figure_paths = composer.compose(
        result=res, run=run, global_metrics=gm, x_axis_np=x_axis, indices=indices, param_space=True,
    )

    assert "improvement_map"       in figure_paths
    assert "reduced_pixel_mse_map" in figure_paths
    for p in figure_paths["improvement_map"]:
        assert p.is_file() and p.stat().st_size > 0

    assert "slices_range_reduced"       in figure_paths
    assert "slices_azimuth_reduced"     in figure_paths
    assert "slices_elev_reduced"        in figure_paths
    assert "ssim_range_reduced"         in figure_paths
    assert "ssim_azimuth_reduced"       in figure_paths
    assert "ssim_elev_reduced"          in figure_paths
    assert "elev_metric_curves_reduced" in figure_paths

    assert len(figure_paths["slices_range_reduced"]) > 0
    for p in figure_paths["slices_range_reduced"]:
        assert "reduced" in str(p)
        assert p.is_file() and p.stat().st_size > 0

    for p in figure_paths["ssim_range_reduced"]:
        assert p.is_file() and p.stat().st_size > 0
