from __future__ import annotations

import numpy as np
import torch

from pipelines.backbone.inference.predictor import CubeStitcher
from pipelines.backbone.inference.probes    import PredictionCurves


class FlipConsistencyEvaluator:

    FLIP_AXES = {"azimuth": 2, "range": 3}

    def __init__(self, run, logger, *, window_kind: str, render_amp_floor: float = 0.0) -> None:
        self.loaded      = run
        self.logger      = logger
        self.window_kind = window_kind
        self.renderer    = PredictionCurves(run.n_gaussians, run.x_axis, render_amp_floor)

    def _curves(self, params: np.ndarray) -> np.ndarray:
        return self.renderer.render(params)

    def _flipped_params(self, images: torch.Tensor, axis: int) -> np.ndarray:
        flipped = torch.flip(images, dims=[axis])
        params  = self.loaded.model(flipped)

        return np.flip(params, axis=axis)

    def _batch_disagreement(self, images: torch.Tensor) -> np.ndarray:
        base_curves = self._curves(self.loaded.model(images))

        disagreement = np.zeros(base_curves.shape[0:1] + base_curves.shape[2:], dtype=np.float64)
        for axis in self.FLIP_AXES.values():
            flip_curves   = self._curves(self._flipped_params(images, axis))
            disagreement += ((base_curves - flip_curves) ** 2).mean(axis=1)

        return (disagreement / len(self.FLIP_AXES)).astype(np.float32)

    def run(self) -> np.ndarray:
        run      = self.loaded
        stitcher = CubeStitcher(grid=run.grid, n_channels=1, window_kind=self.window_kind, dtype="float32")

        sample_count = 0
        with self.logger.track(transient=True) as prog:
            task = prog.add_task("[section]Flip Consistency[/section]", total=len(run.loader))
            for batch in run.loader:
                images       = batch[0]
                disagreement = self._batch_disagreement(images)

                for b in range(disagreement.shape[0]):
                    stitcher.add_patch(sample_count + b, disagreement[b:b + 1])

                sample_count += disagreement.shape[0]
                prog.advance(task)

        return stitcher.finalize_cube()[0]
