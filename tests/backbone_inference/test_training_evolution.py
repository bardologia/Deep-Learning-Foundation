
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
    prediction = np.random.default_rng(0).normal(size=(12, 16))

    image = EvolutionFrames().frame(prediction, "pointwise", epoch=5, vmin=-2.0, vmax=2.0)

    assert image.size[0] > 100 and image.size[1] > 100
    assert image.mode == "RGB"


def test_snapshots_are_discovered_in_epoch_order(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    for epoch in (10, 2, 30):
        torch.save({"epoch": epoch - 1, "params": {}}, ckpt_dir / f"epoch_{epoch:04d}.pt")

    run       = TrainingEvolutionRun(tmp_path, TrainingEvolutionConfig(), SilentLogger())
    snapshots = run._snapshots()

    assert [path.name for path in snapshots] == ["epoch_0002.pt", "epoch_0010.pt", "epoch_0030.pt"]


@pytest.mark.dataset
@pytest.mark.slow
def test_evolution_pass_announces_tracks_and_times_replay_and_frames(tmp_path, test_data_dir):
    run_dir = build_run_directory(tmp_path, test_data_dir)
    (run_dir / "checkpoints").mkdir()
    for epoch in (1, 2):
        shutil.copy(run_dir / "best_model.pt", run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")

    logger  = StageLogger()
    outcome = TrainingEvolutionRun(run_dir, TrainingEvolutionConfig(batch_size=4), logger).run()

    assert ("info", f"{run_dir.name}: replaying 2 snapshots (epoch_0001 .. epoch_0002) over the 'test' split") in logger.messages
    assert logger.timers == ["Training evolution: snapshot replay (2 snapshots)", "Training evolution: frame rendering (4 frames)"]
    assert outcome.payload["epochs"] == [2, 2]
    assert sorted(path.name for path in (run_dir / "training_evolution").glob("*.gif")) == ["evolution_neighbourhood.gif", "evolution_pointwise.gif"]
