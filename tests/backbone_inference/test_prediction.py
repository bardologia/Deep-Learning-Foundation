from __future__ import annotations

import numpy as np
import pytest

from pipelines.backbone.dataset.spatial     import Patcher
from pipelines.backbone.inference.loader    import RunLoader
from pipelines.backbone.inference.metrics   import Metrics
from pipelines.backbone.inference.predictor import Predictor, Stitcher

from tests.backbone_inference._helpers import build_run_directory
from tests.conftest                    import SilentLogger


@pytest.mark.parametrize("kind", ["hann", "uniform", "triangular"])
def test_stitching_constant_patches_returns_the_constant(kind):
    grid     = Patcher.build(spatial_size=(37, 50), patch_size=(16, 16), stride=(8, 8)).grid
    stitcher = Stitcher(grid, n_channels=2, window_kind=kind)

    for index in range(grid.number_of_patches):
        stitcher.add_patch(index, np.full((2, 16, 16), 3.5, dtype=np.float32))

    stitched = stitcher.finalize()

    assert stitched.shape == (2, 37, 50)
    assert np.allclose(stitched, 3.5)


def test_stitching_patches_cut_from_a_map_rebuilds_the_map():
    rng      = np.random.default_rng(0)
    image    = rng.normal(size=(3, 41, 29)).astype(np.float32)
    patcher  = Patcher.build(spatial_size=(41, 29), patch_size=(16, 16), stride=(8, 8))
    stitcher = Stitcher(patcher.grid, n_channels=3)

    for index in range(patcher.grid.number_of_patches):
        stitcher.add_patch(index, patcher.extract(image, index))

    assert np.allclose(stitcher.finalize(), image, atol=1e-5)


def test_metrics_score_every_channel_and_their_mean():
    targets     = np.stack([np.arange(12.0).reshape(3, 4), np.ones((3, 4))]).astype(np.float32)
    predictions = targets + np.stack([np.full((3, 4), 0.5), np.zeros((3, 4))]).astype(np.float32)

    metrics = Metrics.compute(predictions, targets, ["a", "b"])

    assert metrics["rmse_a"] == pytest.approx(0.5)
    assert metrics["bias_a"] == pytest.approx(0.5)
    assert metrics["r2_a"]   == pytest.approx(1.0 - 3.0 / 143.0)
    assert metrics["rmse_b"] == pytest.approx(0.0)
    assert np.isnan(metrics["r2_b"])
    assert Metrics.channel_values(metrics, "a")["mae"] == pytest.approx(0.5)
    assert metrics["rmse_mean"] == pytest.approx(0.25)
    assert metrics["n_pixels"]  == 12


def test_the_loader_rebuilds_the_run_and_the_predictor_stitches_its_split(tmp_path, test_data_dir, targets):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    run     = RunLoader(run_dir, SilentLogger()).load(split="test", batch_size=None, num_workers=0, device="cpu", checkpoint_name="best_model.pt")
    region  = run.split_region

    predictions = Predictor(run, SilentLogger()).predict()

    assert run.target_channels == ["pointwise", "neighbourhood"]
    assert run.checkpoint_meta == {"epoch": 2, "best_val_loss": 0.1, "best_epoch": 2}
    assert predictions.shape == (2, region.row_size, region.col_size)
    assert np.isfinite(predictions).all()
    assert np.array_equal(run.targets, targets[:, region.row_start:region.row_end, region.col_start:region.col_end])
