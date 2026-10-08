"""Tests for the figure and animation layer of backbone inference.

Covers `Animator` argument validation and GIF writing, the set of figure groups
`FigureComposer` emits for plain runs, reduced-baseline comparisons, JEPA embedding
maps and runs whose parameters were refitted from the curves, and the composer
drawing the slices of the index dict `InferencePipeline` builds.
"""

from __future__ import annotations

import contextlib
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from configuration.inference                         import InferenceConfig
from tools.data.regions                              import CropRegion
from pipelines.backbone.inference                    import animation
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics            import Metrics, Result
from pipelines.backbone.inference.figures            import Animator, FigureComposer
from pipelines.backbone.inference.pipeline           import InferencePipeline
from pipelines.backbone.inference.plots              import Plotter

from tests.conftest import RecordingLogger, SilentLogger


class _RecordingProgress:
    """Progress-bar stub that records the tasks added to it.

    Attributes:
        tasks : Tuples of (description, total) in the order the tasks were added.
    """

    def __init__(self) -> None:
        """Starts with no recorded task."""
        self.tasks = []

    def add_task(self, description, total=None, **kwargs):
        """Records a task and returns its index."""
        self.tasks.append((description, total))
        return len(self.tasks) - 1

    def advance(self, task, advance=1):
        """Ignores a progress advance."""
        pass


class _TrackingLogger(RecordingLogger):
    """Recording logger that also keeps the progress tasks opened through track.

    Attributes:
        progress : Progress stub shared by every track block, holding the added tasks.
    """

    def __init__(self) -> None:
        """Starts with an empty message log and progress stub."""
        super().__init__()
        self.progress = _RecordingProgress()

    @contextlib.contextmanager
    def track(self, transient: bool = False):
        """Yields the shared progress stub in place of a live bar."""
        yield self.progress

    def of(self, level: str) -> list:
        """Returns the recorded messages of one level."""
        return [message for recorded, message in self.messages if recorded == level]


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


def _slice_indices(n_col_slices: int = 2, n_row_slices: int = 2, n_z_slices: int = 2) -> dict:
    """Returns the slice-index dict InferencePipeline builds for the test cube of shape (N_Z, H, W).

    Args:
        n_col_slices : Number of col slices plotted.
        n_row_slices : Number of row slices plotted.
        n_z_slices   : Number of z slices plotted.

    Returns:
        The pipeline's mapping of plotted and full index arrays per axis.
    """
    cfg = InferenceConfig(run_directory=Path("run"), device="cpu", n_col_slices=n_col_slices, n_row_slices=n_row_slices, n_z_slices=n_z_slices)

    return InferencePipeline(cfg)._compute_slice_indices(cfg, N_Z, H, W)


def _result_and_metrics():
    """Builds a slightly noisy prediction together with its metrics and slice indices.

    Returns:
        Tuple of the Result, the global metrics dict, the index dict InferencePipeline builds for the composer and the z axis in metres.
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

    indices = _slice_indices()

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


def _composer(tmp_path, logger=None):
    """Builds a figure composer writing under a freshly created output tree.

    Args:
        tmp_path : Run directory the inference output subdirectory is created under.
        logger   : Logger handed to the composer, a silent one when omitted.

    Returns:
        Tuple of the composer and the inference metadata holding its output paths.
    """
    cfg  = InferenceConfig(run_directory=tmp_path, output_subdir="fig", device="cpu", normalize_intensity=False)
    meta = InferenceMetadata(cfg)
    meta.create_dirs()
    plotter = Plotter(normalize=False)
    return FigureComposer(plotter=plotter, meta=meta, logger=SilentLogger() if logger is None else logger, cfg=cfg), meta


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
    written  = animator.walk_gif(
        pred_cube=pred, gt_cube=gt, axis="z",
        out_path=tmp_path / "walk.gif", z_axis=z_axis, row_offset=0, col_offset=0,
    )

    assert list(written) == ["walk"]
    assert written["walk"].is_file()
    assert written["walk"].stat().st_size > 0


@pytest.mark.slow
def test_animator_renders_frames_through_the_logger_bar_and_logs_frame_counts(tmp_path):
    """Frame progress goes through the logger's bar, named after the GIF, and the written line carries frames, delay and time."""
    z_axis = _x_axis()
    params = _params()
    gt     = _curves(params, z_axis)
    logger = _TrackingLogger()

    animator = Animator(logger, max_frames=4, num_workers=1)
    animator.walk_gif(
        pred_cube=gt + 0.05, gt_cube=gt, axis="z",
        out_path=tmp_path / "walk.gif", z_axis=z_axis, row_offset=0, col_offset=0,
    )

    assert logger.progress.tasks == [("GIF walk.gif", 4)]
    assert not hasattr(animation, "tqdm")

    written = [line for line in logger.of("subsection") if line.startswith("GIF (z  ) : ")]
    assert len(written) == 1
    assert f"[4 of {N_Z} slices, 83 ms/frame, " in written[0]
    assert written[0].rstrip().endswith("s]")


def test_animator_rejects_non_positive_duration():
    """Constructing an animator with a zero duration raises ValueError."""
    with pytest.raises(ValueError, match="duration"):
        Animator(SilentLogger(), duration=0.0)


def test_animator_frame_delay_from_duration_or_fps():
    """The frame delay follows the total duration when set and the fps otherwise."""
    assert Animator(SilentLogger(), fps=10)._frame_delay_ms(50)               == 100
    assert Animator(SilentLogger(), fps=10, duration=5.0)._frame_delay_ms(50) == 100
    assert Animator(SilentLogger(), fps=10, duration=2.0)._frame_delay_ms(8)  == 250


def test_animator_panel_figsize_follows_slice_aspect():
    """The single-panel figure keeps the slice aspect ratio with the long side fixed and the short side floored."""
    wide_w, wide_h = Animator._panel_figsize(100, 400)
    tall_w, tall_h = Animator._panel_figsize(400, 100)
    thin_w, thin_h = Animator._panel_figsize(10, 400)

    assert wide_w > wide_h
    assert tall_h > tall_w
    assert thin_h == pytest.approx(Animator.PANEL_SHORT_SIDE_IN + Animator.MARGIN_H_IN)


@pytest.mark.slow
def test_animator_walk_gif_separate_panels(tmp_path):
    """Separate panels write one single-cube GIF per available cube next to the composite."""
    z_axis = _x_axis()
    params = _params()
    gt     = _curves(params, z_axis)
    pred   = gt + 0.05

    animator = Animator(SilentLogger(), max_frames=3, num_workers=1, separate_panels=True, duration=1.5)
    written  = animator.walk_gif(
        pred_cube=pred, gt_cube=gt, axis="col",
        out_path=tmp_path / "walk_col.gif", z_axis=z_axis, row_offset=0, col_offset=0,
        full_cube=gt * 1.1,
    )

    assert list(written) == ["walk_col", "walk_col_pred", "walk_col_gt", "walk_col_full"]
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0


@pytest.mark.slow
def test_figure_composer_compose_creates_files(tmp_path):
    """Composing a parameter-space run emits every expected figure group as non-empty files and closes with the per-family table and total."""
    res, gm, indices, z_axis = _result_and_metrics()
    logger                   = _TrackingLogger()
    composer, meta           = _composer(tmp_path, logger)
    run                      = _run_stub()

    figure_paths = composer.compose(
        result=res, run=run, global_metrics=gm, z_axis_np=z_axis, indices=indices, param_space=True,
    )

    assert "profiles_best"       in figure_paths
    assert "pixel_mse_map"       in figure_paths
    assert "param_distributions" in figure_paths
    assert "param_error_hists"   in figure_paths
    assert "active_count_map"    in figure_paths
    assert "slices_col"          in figure_paths
    assert "ssim_col"            in figure_paths

    flat = [p for paths in figure_paths.values() for p in paths]
    assert len(flat) > 0
    for p in flat:
        assert p.is_file()
        assert p.stat().st_size > 0

    assert ("metrics_table", "Figures written") in logger.messages
    assert logger.of("ok")      == [f"{len(flat)} figures in {len(figure_paths)} families under {meta.figures_dir}"]
    assert logger.of("warning") == []


@pytest.mark.slow
def test_figure_composer_names_the_pixel_maps_drawn_and_reports_an_empty_separation_profile(tmp_path):
    """The pixel-map line names every map drawn, the histograms are counted, and an empty miss-by-separation profile is warned about and summarised."""
    res, _gm, _indices, _z_axis = _result_and_metrics()
    res.label_r2                = res.pixel_r2.copy()
    res.miss_by_sep             = []
    logger                      = _TrackingLogger()
    composer, meta              = _composer(tmp_path, logger)
    figure_paths                = {}

    composer._compose_pixel_maps(res, figure_paths)
    composer._summarize(figure_paths)

    assert f"Pixel maps : mse, r2, peak_error, label_r2 written to {meta.figures_dir / 'pixel_maps'}" in logger.of("subsection")
    assert f"Metric histograms : 3 figures in {meta.figures_dir / 'histograms'}" in logger.of("subsection")
    assert "miss_by_separation" not in figure_paths
    assert logger.of("warning") == ["Miss-by-separation figure skipped: no GT-separation bin of at least 30 pixels with two active GT components"]
    assert ("metrics_table", "Figures written") in logger.messages

    total = sum(len(paths) for paths in figure_paths.values())
    assert logger.of("ok") == [f"{total} figures in {len(figure_paths)} families under {meta.figures_dir}; empty or incomplete: miss_by_separation"]


@pytest.mark.slow
def test_figure_composer_tracks_each_slice_set_on_the_logger_bar(tmp_path):
    """Both slice sets open one progress task each, sized to the col, row and z slices they draw."""
    res, gm, indices, z_axis = _result_and_metrics()
    logger                   = _TrackingLogger()
    composer, _meta          = _composer(tmp_path, logger)
    figure_paths             = {}

    composer._compose_slices(res, _run_stub(), gm, z_axis, indices, figure_paths)

    assert logger.progress.tasks == [("Slice figures (slices)", 6), ("Slice figures (slices_norm)", 6)]


@pytest.mark.slow
def test_figure_composer_draws_the_slices_the_pipeline_indexes(tmp_path):
    """The index dict InferencePipeline builds feeds the slice figures as is, with one figure stem per plotted col, row and z index."""
    res, gm, _indices, z_axis = _result_and_metrics()
    indices                   = _slice_indices(n_col_slices=3, n_row_slices=2, n_z_slices=4)
    composer, _meta           = _composer(tmp_path)
    figure_paths              = {}

    composer._compose_slices(res, _run_stub(), gm, z_axis, indices, figure_paths)

    drawn = {family: {path.name.rsplit("_", 1)[0] for path in figure_paths[family]} for family in ("slices_col", "slices_row", "slices_z")}

    assert drawn["slices_col"] == {f"col_{int(i)}"   for i in indices["slice_col_idx"]}
    assert drawn["slices_row"] == {f"row_{int(i)}"   for i in indices["slice_row_idx"]}
    assert drawn["slices_z"]   == {f"z_idx_{int(i)}" for i in indices["slice_z_idx"]}
    assert len(drawn["slices_col"]) == 3
    assert len(drawn["slices_row"]) == 2
    assert len(drawn["slices_z"])   == 4


@pytest.mark.slow
def test_figure_composer_counts_param_figures_against_three_per_slot(tmp_path):
    """With every GT slot matched, each parameter family reports 3 K of 3 K figures and nothing is skipped."""
    res, _gm, _indices, _z_axis = _result_and_metrics()
    logger                      = _TrackingLogger()
    composer, meta              = _composer(tmp_path, logger)
    figure_paths                = {}

    composer._compose_param_plots(res, _run_stub(), figure_paths)

    assert logger.of("subsection") == [f"Param plots : distributions 6/6, scatter 6/6, error maps 6/6, error histograms 6/6, active-count maps 2 written to {meta.figures_dir}"]
    assert logger.of("warning")    == []


@pytest.mark.slow
def test_figure_composer_explains_the_param_figures_a_collapsed_slot_loses(tmp_path):
    """A GT slot with no matched active prediction loses its scatter, error-map and error-histogram figures under one warning."""
    res, _gm, _indices, _z_axis = _result_and_metrics()
    res.params_pred[3]          = 0.0
    logger                      = _TrackingLogger()
    composer, _meta             = _composer(tmp_path, logger)
    figure_paths                = {}

    composer._compose_param_plots(res, _run_stub(), figure_paths)

    assert len(figure_paths["param_distributions"]) == 6
    assert len(figure_paths["param_scatter"])       == 3
    assert len(figure_paths["param_error_maps"])    == 3
    assert len(figure_paths["param_error_hists"])   == 3

    warnings = logger.of("warning")
    assert len(warnings) == 1
    assert warnings[0].startswith("Param figures skipped for 9 (family, slot, parameter) triples: ")
    assert f"scatter g2/a+mu+sigma: no matched active prediction on {H * W:,} GT-active pixels" in warnings[0]
    assert f"error_hists g2/a+mu+sigma: no matched finite prediction on {H * W:,} GT-active pixels" in warnings[0]
    assert logger.of("subsection")[0].startswith("Param plots : distributions 6/6, scatter 3/6, error maps 3/6, error histograms 3/6")

    composer._summarize(figure_paths)
    assert logger.of("ok")[0].endswith("; empty or incomplete: param_error_hists, param_error_maps, param_scatter")




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
