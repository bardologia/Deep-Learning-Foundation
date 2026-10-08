"""Tests covering training-evolution frame rendering and epoch-checkpoint discovery."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.training_evolution import EvolutionFrames, TrainingEvolutionRun
from configuration.diagnostics                                import TrainingEvolutionConfig

from tests.backbone_inference._helpers import StageLogger, build_run_directory
from tests.conftest                    import SilentLogger


def test_evolution_frame_renders_an_image():
    """A frame renders a non-trivial RGB image of the ground-truth and predicted profiles."""
    frames = EvolutionFrames()
    z_axis = np.linspace(-10.0, 30.0, 24)
    gt     = np.exp(-((z_axis - 10.0) ** 2) / 8.0)
    pred   = 0.5 * gt

    image = frames.frame(z_axis, gt, pred, epoch=5, ylim=1.5)

    assert image.size[0] > 100 and image.size[1] > 100
    assert image.mode == "RGB"


def test_missing_snapshots_raise(tmp_path):
    """An empty checkpoint directory is rejected."""
    (tmp_path / "checkpoints").mkdir()

    run = TrainingEvolutionRun(tmp_path, TrainingEvolutionConfig(), SilentLogger())

    with pytest.raises(FileNotFoundError):
        run._snapshots()


def test_snapshots_are_discovered_in_epoch_order(tmp_path):
    """Epoch checkpoints are returned sorted by epoch, not by discovery order."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    for epoch in (10, 2, 30):
        torch.save({"epoch": epoch - 1, "params": {}}, ckpt_dir / f"epoch_{epoch:04d}.pt")

    run       = TrainingEvolutionRun(tmp_path, TrainingEvolutionConfig(), SilentLogger())
    snapshots = run._snapshots()

    assert [path.name for path in snapshots] == ["epoch_0002.pt", "epoch_0010.pt", "epoch_0030.pt"]


@pytest.mark.dataset
@pytest.mark.slow
def test_evolution_pass_announces_tracks_and_times_replay_and_frames(tmp_path, test_data_dir, params_dir):
    """Replaying two snapshots at two probe pixels announces the snapshot range, advances one bar per snapshot and one per frame, and times both stages."""
    run_dir = build_run_directory(tmp_path, test_data_dir, params_dir)
    (run_dir / "checkpoints").mkdir()
    for epoch in (1, 2):
        shutil.copy(run_dir / "best_model.pt", run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")

    logger = StageLogger()
    TrainingEvolutionRun(run_dir, TrainingEvolutionConfig(window=32, n_row_probes=2, n_col_probes=1), logger).run()

    assert ("info", f"{run_dir.name}: replaying 2 snapshots (epoch_0001 .. epoch_0002) at 2 probe pixels, then rendering 4 frames") in logger.messages
    assert logger.totals == [2, 4] and logger.advances == 6
    assert logger.timers == ["Training evolution: snapshot replay (2 snapshots)", "Training evolution: frame rendering (4 frames)"]
