"""Tests that flip consistency measures a model's equivariance to horizontal flips.

A model whose output follows the flipped input must score zero disagreement, while a
model keyed to absolute column position must disagree most at the image edges. Both
the parameter-head and the direct curve-head paths are covered.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pipelines.backbone.dataset.spatial            import Patcher
from pipelines.backbone.inference.flip_consistency import FlipConsistencyEvaluator

from tests.conftest import SilentLogger


H, W, N_Z = 8, 6, 10


def _equivariant_model(images: torch.Tensor) -> np.ndarray:
    """Predicts Gaussian parameters that depend only on the local input value.

    Args:
        images : Input windows of shape (batch, channels, H, W).

    Returns:
        Parameter cube of shape (batch, 3, H, W) holding amplitude, mu and sigma.
    """
    x      = images.numpy()
    params = np.zeros((x.shape[0], 3, H, W), dtype=np.float32)

    params[:, 0] = np.abs(x[:, 0]) + 0.5
    params[:, 1] = 10.0 * x[:, 0]
    params[:, 2] = 3.0

    return params


def _position_model(images: torch.Tensor) -> np.ndarray:
    """Predicts a centre that ramps with the absolute range column, ignoring the input.

    Args:
        images : Input windows of shape (batch, channels, H, W).

    Returns:
        Parameter cube of shape (batch, 3, H, W) whose mu grows by 5 m per column.
    """
    B      = images.shape[0]
    params = np.zeros((B, 3, H, W), dtype=np.float32)

    cols         = np.arange(W, dtype=np.float32)[None, None, :]
    params[:, 0] = 1.0
    params[:, 1] = np.broadcast_to(5.0 * cols, (B, H, W))
    params[:, 2] = 3.0

    return params


def _run(model) -> SimpleNamespace:
    """Wraps a callable model in the run fields the flip evaluator reads.

    Args:
        model : Callable mapping a batch of input windows to predictions.

    Returns:
        Run stub with a single-patch grid covering the whole (H, W) scene and a
        one-batch loader.
    """
    rng    = np.random.default_rng(0)
    images = torch.from_numpy(rng.uniform(0.2, 1.0, size=(1, 2, H, W))).float()
    grid   = Patcher.build(spatial_size=(H, W), patch_size=(H, W), stride=(H, W)).grid

    return SimpleNamespace(
        model       = model,
        n_gaussians = 1,
        z_axis      = np.linspace(-5.0, 15.0, N_Z).astype(np.float32),
        grid        = grid,
        loader      = [(images, None)],
    )


def test_equivariant_model_has_zero_disagreement():
    """A locally equivariant parameter model yields an all-zero flip map."""
    evaluator = FlipConsistencyEvaluator(_run(_equivariant_model), SilentLogger(), window_kind="uniform")

    flip_map = evaluator.run()

    assert flip_map.shape == (H, W)
    assert np.allclose(flip_map, 0.0, atol=1e-10)


def test_position_dependent_model_disagrees():
    """A position-keyed parameter model disagrees on average and most strongly at the range edge."""
    evaluator = FlipConsistencyEvaluator(_run(_position_model), SilentLogger(), window_kind="uniform")

    flip_map = evaluator.run()

    assert flip_map.mean() > 1e-4
    center_col = (W - 1) / 2.0
    assert flip_map[:, 0].mean() > flip_map[:, int(round(center_col))].mean()


def _equivariant_curve_model(images: torch.Tensor) -> np.ndarray:
    """Predicts curves that copy the local input value into every elevation bin.

    Args:
        images : Input windows of shape (batch, channels, H, W).

    Returns:
        Curve cube of shape (batch, N_Z, H, W).
    """
    x      = images.numpy()
    curves = np.abs(x[:, 0:1]) + 0.5

    return np.repeat(curves, N_Z, axis=1).astype(np.float32)


def _position_curve_model(images: torch.Tensor) -> np.ndarray:
    """Predicts curves whose value is the absolute range column index.

    Args:
        images : Input windows of shape (batch, channels, H, W).

    Returns:
        Curve cube of shape (batch, N_Z, H, W).
    """
    B      = images.shape[0]
    cols   = np.arange(W, dtype=np.float32)[None, None, None, :]
    curves = np.broadcast_to(cols, (B, N_Z, H, W))

    return np.ascontiguousarray(curves, dtype=np.float32)


def test_curve_model_skips_param_rendering_and_scores_zero_when_equivariant():
    """With render_params off no renderer is built and an equivariant curve model scores zero."""
    evaluator = FlipConsistencyEvaluator(_run(_equivariant_curve_model), SilentLogger(), window_kind="uniform", render_params=False)

    assert evaluator.renderer is None

    flip_map = evaluator.run()

    assert flip_map.shape == (H, W)
    assert np.allclose(flip_map, 0.0, atol=1e-10)


def test_curve_model_disagrees_when_position_dependent():
    """A position-keyed curve model disagrees most strongly at the range edge."""
    evaluator = FlipConsistencyEvaluator(_run(_position_curve_model), SilentLogger(), window_kind="uniform", render_params=False)

    flip_map = evaluator.run()

    assert flip_map.mean() > 1e-4
    assert flip_map[:, 0].mean() > flip_map[:, int(round((W - 1) / 2.0))].mean()
