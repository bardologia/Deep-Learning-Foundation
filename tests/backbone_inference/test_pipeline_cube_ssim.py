"""Tests of the raw-cube SSIM stage of the backbone inference pipeline.

The stage is exercised in isolation on synthetic cubes: the reference it scores against,
the metric keys it writes in raw intensity and unit area, the log section it opens and
the skip status it records when the sweep is switched off.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from configuration.benchmark                         import InferenceQueueConfig
from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.metrics            import Metrics, Result
from pipelines.backbone.inference.pipeline           import InferencePipeline
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata

from tests.conftest import RecordingLogger


N_Z    = 16
N_ROWS = 12
N_COLS = 10


def _result(pred: np.ndarray, gt: np.ndarray) -> Result:
    """Builds a Result whose pixel maps are derived from the given cubes.

    Args:
        pred : Predicted cube of shape (N_Z, N_ROWS, N_COLS).
        gt   : Ground-truth cube of the same shape.

    Returns:
        A Result carrying both cubes and their per-pixel metric maps.
    """
    maps = Metrics.curve_pixel_metrics(pred, gt)

    return Result(
        pred_curves        = pred,
        gt_curves          = gt,
        pixel_mse          = maps["mse"],
        pixel_mae          = maps["mae"],
        pixel_r2           = maps["r2"],
        pixel_cosine       = maps["cos"],
        pixel_peak_err_idx = maps["peak"].astype(np.int32),
        cube_directory     = None,
        row_offset         = 0,
        col_offset         = 0,
    )


def _stage(tmp_path, *, enabled: bool):
    """Prepares the pipeline, config, metadata, run stub, result and indices for the stage.

    The prediction equals the raw cube while the ground truth is unrelated, so a unit
    score can only come from scoring against the raw cube.

    Args:
        tmp_path : Directory standing in for the training run directory.
        enabled  : Value of compute_cube_ssim.

    Returns:
        Tuple of pipeline, config, metadata, run stub, result, z axis and slice indices.
    """
    rng  = np.random.default_rng(0)
    full = rng.uniform(0.5, 2.0, (N_Z, N_ROWS, N_COLS)).astype(np.float32)
    gt   = rng.uniform(0.5, 2.0, (N_Z, N_ROWS, N_COLS)).astype(np.float32)

    config = InferenceConfig(
        run_directory     = tmp_path,
        output_subdir     = "cube_ssim",
        device            = "cpu",
        cpu_workers       = 1,
        compute_cube_ssim = enabled,
    )

    meta = InferenceMetadata(config)
    meta.create_dirs()

    run     = types.SimpleNamespace(full_curves=full, n_gaussians=2, activity_threshold=1e-4)
    result  = _result(full.copy(), gt)
    z_axis  = np.linspace(-20.0, 80.0, N_Z)
    indices = {"all_z_idx": np.arange(N_Z), "all_col_idx": np.arange(N_COLS), "all_row_idx": np.arange(N_ROWS)}

    return InferencePipeline(config), config, meta, run, result, z_axis, indices


def test_cube_ssim_is_off_by_default():
    """The stage adds two full SSIM sweeps, so neither the inference nor the benchmark inference config runs it unless asked."""
    assert InferenceConfig(run_directory=Path(".")).compute_cube_ssim is False
    assert InferenceQueueConfig().compute_cube_ssim                   is False


def test_cube_ssim_stage_scores_against_the_raw_cube(tmp_path):
    """The prediction is scored against run.full_curves in both forms, not against the Gaussian ground truth."""
    pipeline, config, meta, run, result, z_axis, indices = _stage(tmp_path, enabled=True)
    metrics = {}
    logger  = RecordingLogger()

    pipeline._evaluate_cube_ssim(config, meta, run, result, z_axis, metrics, indices, logger)

    assert metrics["cube_ssim_status"] == "computed"
    assert logger.messages[0]          == ("section", "[Inference: Cube SSIM]")

    written = json.loads(meta.metrics_path.read_text())
    for axis, n_slices in (("z", N_Z), ("col", N_COLS), ("row", N_ROWS)):
        assert metrics[f"ssim_cube_{axis}_mean"]      == pytest.approx(1.0, abs=1e-6)
        assert metrics[f"ssim_cube_norm_{axis}_mean"] == pytest.approx(1.0, abs=1e-6)
        assert f"ssim_cube_{axis}_{n_slices - 1}" in metrics
        assert written[f"ssim_cube_{axis}_mean"]      == pytest.approx(1.0, abs=1e-6)


def test_cube_ssim_stage_records_its_skip(tmp_path):
    """Disabling the sweep logs a skip section and records the status instead of staying silent."""
    pipeline, config, meta, run, result, z_axis, indices = _stage(tmp_path, enabled=False)
    metrics = {}
    logger  = RecordingLogger()

    pipeline._evaluate_cube_ssim(config, meta, run, result, z_axis, metrics, indices, logger)

    assert metrics["cube_ssim_status"] == "skipped: compute_cube_ssim disabled"
    assert not any(key.startswith("ssim_cube") for key in metrics)

    assert any(level == "section" and "Cube SSIM skipped" in msg for level, msg in logger.messages)
    assert json.loads(meta.metrics_path.read_text())["cube_ssim_status"] == "skipped: compute_cube_ssim disabled"
