"""Tests for refitting Gaussian parameters from predicted z profiles.

Covers `CurveParamExtractor` on synthetic mixtures (accuracy, ordering, chunking,
determinism, axis validation) and the inference-pipeline stage that runs it, fills
`params_pred` and measures the extraction floor on the ground-truth curves.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.backbone.inference.pipeline as pipeline_module

from configuration.inference                    import InferenceConfig
from pipelines.backbone.inference.curve_params  import CurveParamExtractor
from pipelines.backbone.inference.metrics       import Metrics, Result
from pipelines.backbone.inference.pipeline      import InferenceComponents, InferencePipeline

from tests.conftest import SilentLogger


N_Z = 120


def _axis() -> np.ndarray:
    """Returns the ascending z axis of N_Z samples spanning -20 m to 20 m."""
    return np.linspace(-20.0, 20.0, N_Z).astype(np.float32)


def _mixture(amps: np.ndarray, mus: np.ndarray, sigs: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """Renders a Gaussian mixture onto the z axis.

    Args:
        amps   : Amplitudes of shape (n_components, height, width).
        mus    : Centres in metres, of shape (n_components, height, width).
        sigs   : Widths in metres, of shape (n_components, height, width).
        z_axis : Z axis in metres, of shape (n_z,).

    Returns:
        Curve cube of shape (n_z, height, width).
    """
    curves = np.zeros((z_axis.size, *amps.shape[1:]), dtype=np.float32)

    for k in range(amps.shape[0]):
        offset  = z_axis.reshape(-1, 1, 1) - mus[k][None]
        curves += amps[k][None] * np.exp(-0.5 * (offset / sigs[k][None]) ** 2)

    return curves


def _separated_scene(seed: int = 0, height: int = 4, width: int = 3):
    """Builds a two-component scene whose Gaussians sit on opposite halves of the axis.

    Args:
        seed   : Seed of the random generator drawing the component parameters.
        height : Row extent of the scene.
        width  : Range extent of the scene.

    Returns:
        Tuple of the curve cube of shape (n_z, height, width), the amplitudes,
        the centres in metres, the widths in metres and the z axis.
    """
    rng    = np.random.default_rng(seed)
    z_axis = _axis()

    amps = rng.uniform(0.5, 2.0, (2, height, width)).astype(np.float32)
    mus  = np.stack([rng.uniform(-14.0, -6.0, (height, width)), rng.uniform(6.0, 14.0, (height, width))]).astype(np.float32)
    sigs = rng.uniform(1.5, 3.0, (2, height, width)).astype(np.float32)

    return _mixture(amps, mus, sigs, z_axis), amps, mus, sigs, z_axis


def test_extractor_recovers_separated_gaussians():
    """Centres, widths and amplitudes of well-separated Gaussians are recovered within tight tolerances."""
    curves, amps, mus, sigs, z_axis = _separated_scene()

    result = CurveParamExtractor(z_axis, 2, 1e-4).run(curves)
    params = result.params

    assert params.shape == (6, *curves.shape[1:])

    assert np.abs(params[1] - mus[0]).max()  < 0.5
    assert np.abs(params[4] - mus[1]).max()  < 0.5
    assert np.abs(params[2] - sigs[0]).max() < 0.5
    assert np.abs(params[0] - amps[0]).max() < 0.2


def test_extractor_fit_quality_is_high_on_exact_mixtures():
    """Per-pixel fit R2 exceeds 0.99 when the curves are exact Gaussian mixtures."""
    curves, _amps, _mus, _sigs, z_axis = _separated_scene()

    result = CurveParamExtractor(z_axis, 2, 1e-4).run(curves)

    assert result.fit_r2.shape == curves.shape[1:]
    assert result.fit_r2.min() > 0.99


def test_extractor_orders_components_by_position():
    """Extracted slots come out sorted by ascending centre."""
    curves, _amps, _mus, _sigs, z_axis = _separated_scene(seed=3)

    params = CurveParamExtractor(z_axis, 2, 1e-4).run(curves).params

    assert np.all(params[1] <= params[4])


def test_extractor_render_round_trips_its_own_parameters():
    """Rendering the extracted parameters reproduces the input curves."""
    curves, _amps, _mus, _sigs, z_axis = _separated_scene(seed=5)

    extractor = CurveParamExtractor(z_axis, 2, 1e-4)
    params    = extractor.run(curves).params
    rendered  = extractor.render(params)

    assert rendered.shape == curves.shape
    assert np.abs(rendered - curves).max() < 0.25


def test_extractor_returns_inactive_amplitudes_on_empty_profiles():
    """All-zero profiles yield zero amplitudes in every slot."""
    z_axis = _axis()
    curves = np.zeros((N_Z, 3, 2), dtype=np.float32)

    params = CurveParamExtractor(z_axis, 2, 1e-4).run(curves).params

    assert np.all(params[0] == 0.0)
    assert np.all(params[3] == 0.0)


def test_extractor_is_deterministic():
    """Two runs on the same curves give bit-identical parameters."""
    curves, _amps, _mus, _sigs, z_axis = _separated_scene(seed=7)

    first  = CurveParamExtractor(z_axis, 2, 1e-4).run(curves).params
    second = CurveParamExtractor(z_axis, 2, 1e-4).run(curves).params

    np.testing.assert_array_equal(first, second)


def test_extractor_chunking_does_not_change_the_result():
    """Chunking the pixel dimension leaves parameters and fit R2 unchanged."""
    curves, _amps, _mus, _sigs, z_axis = _separated_scene(seed=9, height=6, width=5)

    whole   = CurveParamExtractor(z_axis, 2, 1e-4, chunk_pixels=10_000).run(curves)
    chunked = CurveParamExtractor(z_axis, 2, 1e-4, chunk_pixels=7).run(curves)

    np.testing.assert_allclose(whole.params, chunked.params, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(whole.fit_r2, chunked.fit_r2, rtol=1e-5, atol=1e-6)


def test_extractor_rejects_a_mismatched_z_axis():
    """Curves whose z length differs from the axis raise ValueError."""
    curves = np.zeros((N_Z + 1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="different z grids"):
        CurveParamExtractor(_axis(), 2, 1e-4).run(curves)


def test_extractor_rejects_a_degenerate_axis():
    """An z axis shorter than three samples is rejected at construction."""
    with pytest.raises(ValueError, match="at least 3 samples"):
        CurveParamExtractor(np.array([0.0, 1.0], dtype=np.float32), 1, 1e-4)


def test_wider_gaussians_yield_wider_extracted_sigmas():
    """Extracted sigma tracks the true width for narrow and wide single Gaussians."""
    z_axis = _axis()
    amps   = np.ones((1, 2, 2), dtype=np.float32)
    mus    = np.zeros((1, 2, 2), dtype=np.float32)

    narrow = CurveParamExtractor(z_axis, 1, 1e-4).run(_mixture(amps, mus, np.full((1, 2, 2), 1.5, np.float32), z_axis)).params
    wide   = CurveParamExtractor(z_axis, 1, 1e-4).run(_mixture(amps, mus, np.full((1, 2, 2), 4.0, np.float32), z_axis)).params

    assert wide[2].mean() > narrow[2].mean()
    assert np.abs(narrow[2] - 1.5).max() < 0.4
    assert np.abs(wide[2]   - 4.0).max() < 0.5


def _pipeline_pieces(tmp_path, seed=11):
    """Assembles an inference pipeline and a matching result for the extraction stage.

    Args:
        tmp_path : Directory used as run directory and cube directory.
        seed     : Seed of the synthetic two-component scene.

    Returns:
        Tuple of the pipeline, its configuration, the metadata stub, the run stub, a
        Result whose predicted curves equal the ground truth, and the z axis.
    """
    curves, amps, mus, sigs, z_axis = _separated_scene(seed=seed, height=4, width=3)

    params_gt = np.zeros((6, *curves.shape[1:]), dtype=np.float32)
    for k in range(2):
        params_gt[3 * k]     = amps[k]
        params_gt[3 * k + 1] = mus[k]
        params_gt[3 * k + 2] = sigs[k]

    pixel_maps = Metrics.curve_pixel_metrics(curves, curves)
    result     = Result(
        pred_curves        = curves,
        gt_curves          = curves,
        pixel_mse          = pixel_maps["mse"],
        pixel_mae          = pixel_maps["mae"],
        pixel_r2           = pixel_maps["r2"],
        pixel_cosine       = pixel_maps["cos"],
        pixel_peak_err_idx = pixel_maps["peak"].astype(np.int32),
        cube_directory     = tmp_path,
        row_offset     = 0,
        col_offset       = 0,
        params_gt          = params_gt,
    )

    pipeline = InferencePipeline(
        InferenceConfig(run_directory=tmp_path, save_cubes=False, cpu_workers=2),
        InferenceComponents(param_space=False),
    )

    run  = SimpleNamespace(n_gaussians=2, activity_threshold=1e-4, split_name="test", split_region=SimpleNamespace(as_tuple=lambda: (0, 4, 0, 3)), core_region=None, track_baselines=None, track_profiles=None)
    meta = SimpleNamespace(cube_dir=tmp_path, metrics_path=tmp_path / "metrics.json")

    return pipeline, pipeline.config, meta, run, result, z_axis


def test_extraction_stage_fills_predicted_parameters_and_the_floor(tmp_path):
    """The stage writes extracted parameters, the fit R2 map and the extraction floor onto the result."""
    pipeline, cfg, meta, run, result, z_axis = _pipeline_pieces(tmp_path)

    pipeline._extract_curve_params(cfg, meta, run, result, z_axis, SilentLogger())

    assert result.params_pred.shape == (6, 4, 3)
    assert result.params_source     == "extracted"
    assert result.extract_r2.shape  == (4, 3)
    assert result.extract_floor["matched_mu_mae"] < 0.5


def test_extraction_stage_parallelises_over_the_configured_cpu_workers(tmp_path, monkeypatch):
    """The extractor is constructed with the worker count taken from the inference configuration."""
    pipeline, cfg, meta, run, result, z_axis = _pipeline_pieces(tmp_path)

    captured = {}

    def _spy(**kwargs):
        """Records the extractor keyword arguments and forwards them to the real extractor."""
        captured.update(kwargs)
        return CurveParamExtractor(**kwargs)

    monkeypatch.setattr(pipeline_module, "CurveParamExtractor", _spy)

    pipeline._extract_curve_params(cfg, meta, run, result, z_axis, SilentLogger())

    assert captured["n_workers"] == cfg.cpu_workers == 2


def test_extraction_stage_is_skipped_when_disabled(tmp_path):
    """With extract_curve_params off, no parameters are extracted and the source stays \"model\"."""
    pipeline, cfg, meta, run, result, z_axis = _pipeline_pieces(tmp_path)
    cfg.extract_curve_params = False

    pipeline._extract_curve_params(cfg, meta, run, result, z_axis, SilentLogger())

    assert result.params_pred is None
    assert result.params_source == "model"


def test_extraction_stage_leaves_parameter_models_untouched(tmp_path):
    """Parameter-space models keep their own head output instead of refitted parameters."""
    pipeline, cfg, meta, run, result, z_axis = _pipeline_pieces(tmp_path)
    pipeline.components = InferenceComponents(param_space=True)

    pipeline._extract_curve_params(cfg, meta, run, result, z_axis, SilentLogger())

    assert result.params_pred is None


def test_extracted_parameters_unlock_the_param_space_metrics(tmp_path):
    """Extracted parameters enable the matched and floor metrics while slot-organisation metrics stay suppressed."""
    pipeline, cfg, meta, run, result, z_axis = _pipeline_pieces(tmp_path)

    pipeline._extract_curve_params(cfg, meta, run, result, z_axis, SilentLogger())

    indices = {"all_z_idx": np.arange(N_Z), "all_col_idx": np.arange(3), "all_row_idx": np.arange(4)}
    metrics = pipeline._evaluate_metrics(cfg, result, z_axis.astype(np.float64), run, meta, indices, SilentLogger())

    assert metrics["params_source"] == "extracted"
    assert "matched_mu_mae"         in metrics
    assert "active_frac_gt"         in metrics
    assert "extract_floor_matched_mu_mae" in metrics
    assert "extract_fit_r2_median"        in metrics
    assert "slot_usage_entropy"     not in metrics


def test_extractor_handles_a_descending_z_axis():
    """A descending z axis still recovers the true centre and width."""
    z_axis = _axis()[::-1].copy()
    amps   = np.ones((1, 2, 2), dtype=np.float32)
    mus    = np.full((1, 2, 2), 3.0, dtype=np.float32)
    sigs   = np.full((1, 2, 2), 2.5, dtype=np.float32)

    params = CurveParamExtractor(z_axis, 1, 1e-4).run(_mixture(amps, mus, sigs, z_axis)).params

    assert np.abs(params[1] - 3.0).max() < 0.5
    assert np.abs(params[2] - 2.5).max() < 0.5


def test_extraction_stage_requires_the_exact_gt_parameters(tmp_path):
    """Missing ground-truth parameters make the stage raise, since the floor cannot be measured."""
    pipeline, cfg, meta, run, result, z_axis = _pipeline_pieces(tmp_path)
    result.params_gt = None

    with pytest.raises(ValueError, match="measure its own floor"):
        pipeline._extract_curve_params(cfg, meta, run, result, z_axis, SilentLogger())


def test_extractor_does_not_fork_a_process_pool_by_default():
    """The peak initialiser is built single-threaded and forks no process pool."""
    from pipelines.processing.param_extraction.sigma.initialiser import PeakInitialiser

    created = []
    build   = PeakInitialiser.__init__

    def spy(self, n_workers=1):
        """Records the worker count the initialiser is built with and asserts no pool is created."""
        created.append(n_workers)
        build(self, n_workers=n_workers)
        assert self._pool is None

    PeakInitialiser.__init__ = spy
    try:
        curves, _amps, _mus, _sigs, z_axis = _separated_scene(seed=13)
        CurveParamExtractor(z_axis, 2, 1e-4).run(curves)
    finally:
        PeakInitialiser.__init__ = build

    assert created == [1]
