"""End-to-end tests of the backbone inference pipeline on the synthetic test scene.

A miniature U-Net run is inferred over the test split, then re-inferred with a
region override that extends the split outward, and the equal-spacing slice
index helper is checked in isolation.
"""

from __future__ import annotations

import json
import numbers

import numpy as np
import pytest

from configuration.inference               import InferenceConfig
from pipelines.backbone.inference.pipeline import InferencePipeline
from pipelines.shared.inference.metadata   import CompletedInference
from tools.metrics.scoring                 import MetricOrientation

from tests.backbone_inference._helpers import N_GAUSSIANS, N_Z, build_run_directory

@pytest.mark.dataset
@pytest.mark.slow
def test_inference_pipeline_end_to_end(tmp_path, test_data_dir, params_dir, param_extraction_meta):
    """A full run writes the report, finite curve metrics, the label run's activity threshold and the saved cubes."""
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

    assert metrics["split"]              == "test"
    assert metrics["activity_threshold"] == float(param_extraction_meta["activity_threshold"])

    for key in ("curve_mse_gt", "overall_r2_gt", "pixel_mse_gt_mean", "ssim_norm_z_mean", "active_frac_gt", "matched_recall", "pixel_cosine_gt_nonempty_mean", "gt_empty_frac"):
        assert np.isfinite(metrics[key]), key

    for key, value in metrics.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            MetricOrientation.direction(key)

    cubes = output_dir / "cubes"
    assert np.load(cubes / "pred_curves.npy").shape == (N_Z, 96, 96)
    assert np.load(cubes / "gt_curves.npy").shape   == (N_Z, 96, 96)
    assert np.load(cubes / "params_pred.npy").shape == (3 * N_GAUSSIANS, 96, 96)
    assert np.load(cubes / "params_gt.npy").shape   == (3 * N_GAUSSIANS, 96, 96)

    report = report_path.read_text()
    assert "unet" in report

    marker   = json.loads((output_dir / "complete.json").read_text())
    resolved = json.loads((output_dir / "inference_config.json").read_text())

    assert marker["checkpoint"]["name"]       == "best_model.pt"
    assert marker["checkpoint"]["epoch"] + 1  == metrics["checkpoint_epoch"]
    assert marker["inference_config"]         == "inference_config.json"
    assert marker["provenance"]["git_commit"] == metrics["code_commit"]
    assert metrics["checkpoint_name"]         == "best_model.pt"
    assert resolved["render_amp_floor"]       == config.render_amp_floor
    assert resolved["run_directory"]          == str(run_dir)
    assert "code_commit" in report and "render_amp_floor" in report

    log = (output_dir / "logs" / "inference.log").read_text()
    assert f"Cubes   : {len(list(cubes.glob('*.npy')))} .npy files in {cubes}" in log


@pytest.mark.dataset
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


@pytest.mark.dataset
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


@pytest.mark.dataset
@pytest.mark.slow
def test_backbone_runner_scores_each_region_of_a_split_stored_as_two(tmp_path, test_data_dir, params_dir):
    """A test split stored as two disjoint regions is scored once per region, each pass reporting its own region as split and core region and its region pass."""
    from configuration.inference                       import BackboneInferenceEntryConfig
    from pipelines.shared.inference.inference_dispatch import BackboneInferenceRunner

    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    regions = [[1200, 1296, 500, 596], [1300, 1396, 500, 596]]

    dataset_path = run_dir / "meta" / "dataset_creation_config.json"
    payload      = json.loads(dataset_path.read_text())
    payload["split_regions"]["test"] = [dict(zip(("row_start", "row_end", "col_start", "col_end"), region)) for region in regions]
    dataset_path.write_text(json.dumps(payload))

    inference = InferenceConfig(
        run_directory   = run_dir,
        device          = "cpu",
        split           = "test",
        num_workers     = 0,
        cpu_workers     = 2,
        save_plots      = False,
        save_animations = False,
        save_cubes      = False,
    )

    BackboneInferenceRunner(BackboneInferenceEntryConfig(inference=inference)).run(run_dir)

    stamps = sorted((run_dir / "inference").glob("*_region*"))
    assert [stamp.name[-len("_region0"):] for stamp in stamps] == ["_region0", "_region1"]

    group = stamps[0].name[:-len("_region0")]

    for index, (stamp, region) in enumerate(zip(stamps, regions)):
        metrics = json.loads((stamp / "metrics.json").read_text())
        marker  = json.loads((stamp / "complete.json").read_text())
        assert metrics["split_region"] == region
        assert metrics["core_region"]  == region
        assert metrics["region_pass"]  == {"index": index, "count": 2, "group": group}
        assert marker["region_pass"]   == {"index": index, "count": 2, "group": group}

    assert CompletedInference.stamps(run_dir / "inference", "test") == stamps
    assert "Cubes   : not saved (save_cubes=False)" in (stamps[0] / "logs" / "inference.log").read_text()


def test_equal_indices_are_unique_and_inside_the_axis():
    """Equally spaced slice indices are unique and stay inside the axis for every axis and slice count."""
    for n_total in (7, 10, 64, 150):
        for n_slices in (1, 3, 5, n_total):
            indices = InferencePipeline._equal_indices(n_total, n_slices)

            assert indices.size == np.unique(indices).size
            assert indices.min() >= 0
            assert indices.max() < n_total
