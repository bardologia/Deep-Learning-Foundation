"""End-to-end tests of scoring a trained backbone run on a dataset it was not trained on.

A miniature U-Net run built over the synthetic test scene is scored on a second
synthetic scene, generated with another seed and four other secondary channels,
over a crop outside the run's splits. The pass must stamp under the run's
cross-dataset tree, record the target in its marker, metrics and report, score
against the target's labels, leave the run's own inference untouched, and refuse a
target whose labels or channel count do not fit the run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest

from configuration.inference                        import InferenceConfig, TargetDatasetConfig
from configuration.synthesis                        import SynthesisConfig
from pipelines.backbone.inference.pipeline          import InferencePipeline
from pipelines.processing.synthesis.pipeline        import SyntheticDatasetPipeline
from pipelines.shared.inference.inference_scheduler import InferenceScheduler
from pipelines.shared.inference.metadata            import TargetDataset
from pipelines.shared.inference.run_classifier      import InferenceLayout, RunType

from tests.backbone_inference._helpers import N_GAUSSIANS, N_Z, build_run_directory
from tests.conftest                    import SilentLogger


TARGET_CHANNELS = ("CH05", "CH06", "CH07", "CH08")
TARGET_CROP     = (1400, 1496, 600, 696)


def _generate_scene(root: Path, name: str, **overrides) -> Path:
    """Generates a synthetic scene under the root with the given configuration overrides and returns its dataset directory."""
    config = dataclasses.replace(SynthesisConfig(), output_directory=root / name, **overrides)

    SyntheticDatasetPipeline(config, logger=SilentLogger()).run()

    return config.output_directory


@pytest.fixture(scope="module")
def second_scene(tmp_path_factory) -> Path:
    """Returns a scene generated with another seed and other secondary channel labels than the test scene."""
    return _generate_scene(tmp_path_factory.mktemp("target"), "second_scene", seed=1, secondary_labels=TARGET_CHANNELS)


def _config(run_dir: Path, scene: Path, label_run: str = SynthesisConfig().params_run_name, channels=TARGET_CHANNELS) -> InferenceConfig:
    """Returns an inference configuration scoring the run on the scene's crop through the given secondary channels."""
    return InferenceConfig(
        run_directory   = run_dir,
        output_subdir   = "target",
        device          = "cpu",
        num_workers     = 0,
        cpu_workers     = 2,
        save_plots      = False,
        save_animations = False,
        save_cubes      = True,
        target          = TargetDatasetConfig(
            enabled          = True,
            dataset_path     = scene,
            parameters_path  = scene / "params" / label_run / "parameters.npy",
            secondary_labels = list(channels),
            row_start        = TARGET_CROP[0],
            row_end          = TARGET_CROP[1],
            col_start        = TARGET_CROP[2],
            col_end          = TARGET_CROP[3],
        ),
    )


@pytest.mark.dataset
@pytest.mark.slow
def test_a_run_scored_on_a_target_dataset_stamps_under_its_cross_dataset_tree(tmp_path, test_data_dir, params_dir, second_scene):
    """The pass scores the target crop through the target channels and records the target everywhere it is read back."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    config  = _config(run_dir, second_scene)

    report_path = InferencePipeline(config).run()

    output_dir = run_dir / InferenceLayout.TARGET_DIRNAME / "second_scene" / "target"
    assert report_path == output_dir / "report.md"
    assert not (run_dir / InferenceLayout.DIRNAME).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    marker  = json.loads((output_dir / "complete.json").read_text())
    target  = TargetDataset.of(config).payload()

    assert metrics[TargetDataset.KEY]  == target
    assert marker[TargetDataset.KEY]   == target
    assert metrics["training_dataset"] == str(test_data_dir)
    assert metrics["split"]            == "test"
    assert metrics["split_region"]     == list(TARGET_CROP)

    for key in ("curve_mse_gt", "pixel_cosine_gt_nonempty_mean", "matched_recall"):
        assert np.isfinite(metrics[key]), key

    labels   = np.load(second_scene / "params" / SynthesisConfig().params_run_name / "parameters.npy")
    rows     = slice(TARGET_CROP[0] - SynthesisConfig().global_crop[0], TARGET_CROP[1] - SynthesisConfig().global_crop[0])
    cols     = slice(TARGET_CROP[2] - SynthesisConfig().global_crop[2], TARGET_CROP[3] - SynthesisConfig().global_crop[2])
    expected = labels[:, rows, cols]
    gt       = np.load(output_dir / "cubes" / "params_gt.npy")
    known    = np.isfinite(gt)

    assert np.load(output_dir / "cubes" / "pred_curves.npy").shape == (N_Z, 96, 96)
    assert gt.shape == (3 * N_GAUSSIANS, 96, 96)
    assert known[0::3].all()
    np.testing.assert_allclose(gt[known], expected[known], rtol=1e-5, atol=1e-6)

    report = report_path.read_text()
    assert "second_scene" in report
    assert "training_dataset" in report
    assert ", ".join(TARGET_CHANNELS) in report

    entry     = SimpleNamespace(runs_dir=str(run_dir), run_filter=[], gpus=[0], gpus_file="", poll_interval_s=1, skip_completed=True)
    block     = dataclasses.replace(config, run_directory=Path("."), output_subdir=None)
    scheduler = InferenceScheduler(entry, Path("main/x.py"), RunType.BACKBONE, block=block)

    assert scheduler._inference_state(run_dir, SilentLogger()) == ("done", True)


@pytest.mark.dataset
@pytest.mark.slow
def test_a_target_feeding_fewer_channels_than_the_run_consumes_is_refused(tmp_path, test_data_dir, params_dir, second_scene):
    """Three target channels give fewer input channels than the four-secondary run was trained on."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)

    with pytest.raises(ValueError, match="input channels"):
        InferencePipeline(_config(run_dir, second_scene, channels=TARGET_CHANNELS[:3])).run()


@pytest.mark.dataset
@pytest.mark.slow
def test_target_labels_fitted_at_another_k_are_refused(tmp_path, test_data_dir, params_dir):
    """The K=3 label run of a scene cannot score a run that predicts K=2 Gaussians."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    scene   = _generate_scene(tmp_path, "k3_scene", k_max=N_GAUSSIANS + 1, params_run_name=f"params_k{N_GAUSSIANS + 1}")

    with pytest.raises(ValueError, match=f"predicts K={N_GAUSSIANS} Gaussians"):
        InferencePipeline(_config(run_dir, scene, label_run=f"params_k{N_GAUSSIANS + 1}")).run()
