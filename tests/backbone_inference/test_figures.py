"""Tests for the figure and animation layer of backbone inference.

Covers `Animator` argument validation and GIF writing, and the set of figure groups
`FigureComposer` emits for plain runs, reduced-baseline comparisons, JEPA embedding
maps and runs whose parameters were refitted from the curves.
"""

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

from tests.conftest import SilentLogger


N_GAUSSIANS = 2
N_Z         = 12
H           = W       = 8


def _x_axis():
    """Returns the z axis of N_Z samples spanning -20 m to 80 m."""
    return np.linspace(-20.0, 80.0, N_Z).astype(np.float32)


def _params():
    """Returns a parameter cube of shape (3 * N_GAUSSIANS, H, W) with random amplitudes and fixed centres and widths."""
    rng    = np.random.default_rng(0)
    params = np.zeros((N_GAUSSIANS * 3, H, W), dtype=np.float32)
    for k in range(N_GAUSSIANS):
        params[3 * k]     = rng.random((H, W)).astype(np.float32) + 0.5
        params[3 * k + 1] = float(k * 15)
        params[3 * k + 2] = 4.0
    return params


def _curves(params, z_axis):
    """Renders a parameter cube into z profiles.

    Args:
        params : Parameter cube of shape (3 * N_GAUSSIANS, H, W), grouped amplitude, mu, sigma.
        z_axis : Z axis in metres, of shape (n_z,).

    Returns:
        Curve cube of shape (n_z, H, W).
    """
    curve = np.zeros((z_axis.size, H, W), dtype=np.float32)
    for k in range(N_GAUSSIANS):
        a   = np.maximum(params[3 * k], 0.0)[None]
        mu  = params[3 * k + 1][None]
        sig = params[3 * k + 2][None]
        x   = z_axis.reshape(-1, 1, 1)
        curve += a * np.exp(-((x - mu) ** 2) / (2.0 * sig * sig + 1e-8))
    return curve.astype(np.float32)


def _result_and_metrics():
    """Builds a slightly noisy prediction together with its metrics and slice indices.

    Returns:
        Tuple of the Result, the global metrics dict, the index dict consumed by the
        composer and the z axis in metres.
    """
    z_axis = _x_axis()
    params = _params()
    gt     = _curves(params, z_axis)
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
        row_offset         = 0,
        col_offset         = 0,
        params_pred        = params.copy(),
        params_gt          = params.copy(),
    )

    indices = {
        "slice_z_idx"   : np.array([2, 6]),
        "slice_col_idx" : np.array([2, 5]),
        "slice_row_idx" : np.array([2, 5]),
        "all_z_idx"     : np.arange(N_Z),
        "all_col_idx"   : np.arange(W),
        "all_row_idx"   : np.arange(H),
    }

    gm = Metrics(res, z_axis, N_GAUSSIANS, cpu_workers=2, active_amp_thr=1e-4, logger=SilentLogger()).compute(
        z_indices   = indices["all_z_idx"],
        col_indices = indices["all_col_idx"],
        row_indices = indices["all_row_idx"],
        param_space = True,
    )

    return res, gm, indices, z_axis




def _run_stub():
    """Returns a run stub carrying the split region, full curves and channel geometry the composer reads."""
    region = CropRegion(row_start=0, row_end=H, col_start=0, col_end=W)
    return types.SimpleNamespace(
        n_gaussians        = N_GAUSSIANS,
        activity_threshold = 1e-4,
        split_region       = region,
        full_curves        = _curves(_params(), _x_axis()),
        complex_inputs     = None,
        n_secondaries      = 0,
        secondary_labels   = None,
    )


def _composer(tmp_path):
    """Builds a figure composer writing under a freshly created output tree.

    Args:
        tmp_path : Run directory the inference output subdirectory is created under.

    Returns:
        Tuple of the composer and the inference metadata holding its output paths.
    """
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="fig", device="cpu", normalize_intensity=False)
    meta = InferenceMetadata(cfg)
    meta.create_dirs()
    plotter = Plotter(normalize=False)
    return FigureComposer(plotter=plotter, meta=meta, logger=SilentLogger(), cfg=cfg), meta


def test_animator_build_axis_bad_axis():
    """An unknown walk axis raises ValueError."""
    animator = Animator(SilentLogger())
    cubes    = (np.zeros((N_Z, H, W), np.float32), np.zeros((N_Z, H, W), np.float32))
    with pytest.raises(ValueError, match="axis must be"):
        animator._build_axis("bogus", cubes, _x_axis(), 0, 0)


def test_animator_rejects_non_positive_max_frames():
    """Constructing an animator with max_frames of zero raises ValueError."""
    with pytest.raises(ValueError, match="max_frames"):
        Animator(SilentLogger(), max_frames=0)


@pytest.mark.slow
def test_animator_walk_gif(tmp_path):
    """Walking the z axis writes a non-empty GIF."""
    z_axis = _x_axis()
    params = _params()
    gt     = _curves(params, z_axis)
    pred   = gt + 0.05

    animator = Animator(SilentLogger(), max_frames=4, num_workers=1)
    out      = animator.walk_gif(
        pred_cube=pred, gt_cube=gt, axis="z",
        out_path=tmp_path / "walk.gif", z_axis=z_axis, row_offset=0, col_offset=0,
    )

    assert out.is_file()
    assert out.stat().st_size > 0


@pytest.mark.slow
def test_figure_composer_compose_creates_files(tmp_path):
    """Composing a parameter-space run emits every expected figure group as non-empty files."""
    res, gm, indices, z_axis = _result_and_metrics()
    composer, meta = _composer(tmp_path)
    run            = _run_stub()

    figure_paths = composer.compose(
        result=res, run=run, global_metrics=gm, z_axis_np=z_axis, indices=indices, param_space=True,
    )

    assert "profiles_best"      in figure_paths
    assert "pixel_mse_map"      in figure_paths
    assert "param_distributions" in figure_paths
    assert "param_error_hists"  in figure_paths
    assert "active_count_map"   in figure_paths
    assert "slices_col"       in figure_paths
    assert "ssim_col"         in figure_paths

    flat = [p for paths in figure_paths.values() for p in paths]
    assert len(flat) > 0
    for p in flat:
        assert p.is_file()
        assert p.stat().st_size > 0




class _EmbeddingMaps:
    """Stand-in for the JEPA embedding maps attached to a Result."""
    def __init__(self):
        """Fills the four embedding maps of shape (H, W) with random values."""
        rng = np.random.default_rng(3)

        self.embedding_error  = rng.random((H, W)).astype(np.float32)
        self.embedding_cosine = rng.uniform(0.5, 1.0, (H, W)).astype(np.float32)
        self.decode_mse       = rng.random((H, W)).astype(np.float32)
        self.chain_mse        = rng.random((H, W)).astype(np.float32)


@pytest.mark.slow
def test_figure_composer_writes_embedding_maps(tmp_path):
    """Embedding maps on the result produce the four map figures and the histogram figure."""
    res, gm, indices, z_axis = _result_and_metrics()
    res.embedding_maps       = _EmbeddingMaps()

    composer, _meta = _composer(tmp_path)

    figure_paths = composer.compose(
        result=res, run=_run_stub(), global_metrics=gm, z_axis_np=z_axis, indices=indices, param_space=True,
    )

    for key in ("embedding_error_map", "embedding_cosine_map", "embedding_decode_map", "embedding_chain_map", "embedding_histograms"):
        assert key in figure_paths
        for path in figure_paths[key]:
            assert path.is_file() and path.stat().st_size > 0


@pytest.mark.slow
def test_figure_composer_skips_slot_organization_for_extracted_parameters(tmp_path):
    """Refitted parameters swap the slot-organisation figures for the extraction R2 map."""
    res, gm, indices, z_axis = _result_and_metrics()
    res.params_source        = "extracted"
    res.extract_r2           = np.random.default_rng(4).uniform(0.8, 1.0, (H, W)).astype(np.float32)

    composer, _meta = _composer(tmp_path)

    figure_paths = composer.compose(
        result=res, run=_run_stub(), global_metrics=gm, z_axis_np=z_axis, indices=indices, param_space=True,
    )

    assert "param_distributions" in figure_paths
    assert "extract_r2_map"      in figure_paths
    assert "slot_usage"      not in figure_paths
    assert "slot_mu_rank"    not in figure_paths


@pytest.mark.slow
def test_figure_composer_keeps_slot_organization_for_model_parameters(tmp_path):
    """Model-predicted parameters keep the slot-usage figure."""
    res, gm, indices, z_axis = _result_and_metrics()

    composer, _meta = _composer(tmp_path)

    figure_paths = composer.compose(
        result=res, run=_run_stub(), global_metrics=gm, z_axis_np=z_axis, indices=indices, param_space=True,
    )

    assert "slot_usage" in figure_paths
