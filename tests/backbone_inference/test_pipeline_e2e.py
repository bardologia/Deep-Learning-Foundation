"""End-to-end tests of the backbone inference pipeline on the synthetic test scene.

A miniature U-Net run is inferred over the test split, then re-inferred with a
region override that extends the split outward, and the equal-spacing slice
index helper is checked in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configuration.inference               import InferenceConfig
from pipelines.backbone.inference.pipeline import InferencePipeline

from tests.backbone_inference._helpers import N_GAUSSIANS, N_Z, SECONDARY_LABELS, build_run_directory


@pytest.mark.dataset
@pytest.mark.slow
def test_inference_pipeline_end_to_end(tmp_path, test_data_dir, params_dir):
    """A full run writes the report, finite curve and physics metrics, per-track agreement and the saved cubes."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)

    config = InferenceConfig(
        run_directory            = run_dir,
        output_subdir            = "e2e",
        device                   = "cpu",
        split                    = "test",
        num_workers              = 0,
        cpu_workers              = 2,
        save_plots               = False,
        save_animations          = False,
        save_cubes               = True,
    )

    report_path = InferencePipeline(config).run()

    output_dir = run_dir / "inference" / "e2e"
    assert report_path == output_dir / "report.md"
    assert report_path.is_file()

    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert metrics["split"] == "test"

    for key in ("curve_mse_gt", "overall_r2_gt", "pixel_mse_gt_mean", "ssim_norm_z_mean", "active_frac_gt", "matched_recall"):
        assert np.isfinite(metrics[key]), key

    cubes = output_dir / "cubes"
    assert np.load(cubes / "pred_curves.npy").shape == (N_Z, 96, 96)
    assert np.load(cubes / "gt_curves.npy").shape   == (N_Z, 96, 96)
    assert np.load(cubes / "params_pred.npy").shape == (3 * N_GAUSSIANS, 96, 96)
    assert np.load(cubes / "params_gt.npy").shape   == (3 * N_GAUSSIANS, 96, 96)

    report = report_path.read_text()
    assert "unet" in report


@pytest.mark.real_data
@pytest.mark.slow
def test_inference_pipeline_with_region_override(tmp_path, test_data_dir, params_dir):
    """A region override widens the inferred area while the stored split region stays the reported core region."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)

    config = InferenceConfig(
        run_directory            = run_dir,
        output_subdir            = "extended",
        device                   = "cpu",
        split                    = "test",
        region_override          = [1180, 1296, 500, 596],
        num_workers              = 0,
        cpu_workers              = 2,
        save_plots               = False,
        save_animations          = False,
        save_cubes               = True,
    )

    InferencePipeline(config).run()

    output_dir = run_dir / "inference" / "extended"
    metrics    = json.loads((output_dir / "metrics.json").read_text())

    assert metrics["split_region"] == [1180, 1296, 500, 596]
    assert metrics["core_region"]  == [1200, 1296, 500, 596]

    assert np.load(output_dir / "cubes" / "pred_curves.npy").shape == (N_Z, 116, 96)
    assert np.load(output_dir / "cubes" / "gt_curves.npy").shape   == (N_Z, 116, 96)


@pytest.mark.real_data
def test_region_override_must_contain_the_stored_split_region(tmp_path, test_data_dir, params_dir):
    """A region override disjoint from the stored split region raises ValueError."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)

    config = InferenceConfig(
        run_directory   = run_dir,
        output_subdir   = "shifted",
        device          = "cpu",
        split           = "test",
        region_override = [1000, 1096, 500, 596],
        num_workers     = 0,
        save_plots      = False,
        save_animations = False,
    )

    with pytest.raises(ValueError, match="must contain the stored"):
        InferencePipeline(config).run()


def test_equal_indices_are_unique_and_inside_the_axis():
    """Equally spaced slice indices are unique and stay inside the axis for every axis and slice count."""
    for n_total in (7, 10, 64, 150):
        for n_slices in (1, 3, 5, n_total):
            indices = InferencePipeline._equal_indices(n_total, n_slices)

            assert indices.size == np.unique(indices).size
            assert indices.min() >= 0
            assert indices.max() < n_total
