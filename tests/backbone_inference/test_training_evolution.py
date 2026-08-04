from __future__ import annotations

from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import pytest
import torch

from pipelines.backbone.inference.analysis.training_evolution import EvolutionFrames, TrainingEvolutionRun
from configuration.diagnostics                       import TrainingEvolutionConfig

from tests.conftest import SilentLogger


def test_evolution_frame_renders_an_image():
    frames = EvolutionFrames()
    x_axis = np.linspace(-10.0, 30.0, 24)
    gt     = np.exp(-((x_axis - 10.0) ** 2) / 8.0)
    pred   = 0.5 * gt

    image = frames.frame(x_axis, gt, pred, epoch=5, ylim=1.5)

    assert image.size[0] > 100 and image.size[1] > 100
    assert image.mode == "RGB"


def test_missing_snapshots_raise(tmp_path):
    (tmp_path / "checkpoints").mkdir()

    run = TrainingEvolutionRun(tmp_path, TrainingEvolutionConfig(), SilentLogger())

    with pytest.raises(FileNotFoundError):
        run._snapshots()


def test_snapshots_are_discovered_in_epoch_order(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    for epoch in (10, 2, 30):
        torch.save({"epoch": epoch - 1, "params": {}}, ckpt_dir / f"epoch_{epoch:04d}.pt")

    run       = TrainingEvolutionRun(tmp_path, TrainingEvolutionConfig(), SilentLogger())
    snapshots = run._snapshots()

    assert [path.name for path in snapshots] == ["epoch_0002.pt", "epoch_0010.pt", "epoch_0030.pt"]
