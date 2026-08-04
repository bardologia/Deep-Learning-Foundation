from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pipelines.backbone.dataset.spatial            import Patcher
from pipelines.backbone.inference.flip_consistency import FlipConsistencyEvaluator

from tests.conftest import SilentLogger


H, W, N_ELEV = 8, 6, 10


def _equivariant_model(images: torch.Tensor) -> np.ndarray:
    x      = images.numpy()
    params = np.zeros((x.shape[0], 3, H, W), dtype=np.float32)

    params[:, 0] = np.abs(x[:, 0]) + 0.5
    params[:, 1] = 10.0 * x[:, 0]
    params[:, 2] = 3.0

    return params


def _position_model(images: torch.Tensor) -> np.ndarray:
    B      = images.shape[0]
    params = np.zeros((B, 3, H, W), dtype=np.float32)

    cols         = np.arange(W, dtype=np.float32)[None, None, :]
    params[:, 0] = 1.0
    params[:, 1] = np.broadcast_to(5.0 * cols, (B, H, W))
    params[:, 2] = 3.0

    return params


def _run(model) -> SimpleNamespace:
    rng    = np.random.default_rng(0)
    images = torch.from_numpy(rng.uniform(0.2, 1.0, size=(1, 2, H, W))).float()
    grid   = Patcher.build(spatial_size=(H, W), patch_size=(H, W), stride=(H, W)).grid

    return SimpleNamespace(
        model       = model,
        n_gaussians = 1,
        x_axis      = np.linspace(-5.0, 15.0, N_ELEV).astype(np.float32),
        grid        = grid,
        loader      = [(images, None)],
    )


def test_equivariant_model_has_zero_disagreement():
    evaluator = FlipConsistencyEvaluator(_run(_equivariant_model), SilentLogger(), window_kind="uniform")

    flip_map = evaluator.run()

    assert flip_map.shape == (H, W)
    assert np.allclose(flip_map, 0.0, atol=1e-10)


def test_position_dependent_model_disagrees():
    evaluator = FlipConsistencyEvaluator(_run(_position_model), SilentLogger(), window_kind="uniform")

    flip_map = evaluator.run()

    assert flip_map.mean() > 1e-4
    center_col = (W - 1) / 2.0
    assert flip_map[:, 0].mean() > flip_map[:, int(round(center_col))].mean()
