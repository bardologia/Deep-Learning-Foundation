"""Measures how far a backbone's predictions drift under azimuth and range flips.

The model is applied to each patch and to its flipped copies; unflipping the outputs
and comparing the rendered curves gives a per-pixel disagreement map that requires no
ground truth, so it can be read as a self-consistency diagnostic.
"""

from __future__ import annotations

import numpy as np
import torch

from pipelines.backbone.inference.predictor import CubeStitcher
from pipelines.backbone.inference.probes    import PredictionCurves


class FlipConsistencyEvaluator:
    """Stitches a per-pixel flip-equivariance disagreement map for a loaded run.

    Attributes:
        loaded        : Loaded run providing the model, patch loader and stitching grid.
        logger        : Logger driving the progress display.
        window_kind   : Stitching window used to blend overlapping patches.
        render_params : Whether model outputs are Gaussian parameters needing rendering to curves.
        renderer      : Curve renderer used when `render_params` is set, otherwise None.
    """

    FLIP_AXES = {"row": 2, "col": 3}

    def __init__(self, run, logger, *, window_kind: str, render_amp_floor: float = 0.0, render_params: bool = True) -> None:
        """Stores the run and builds the curve renderer for parameter-space models.

        Args:
            run              : Loaded run carrying the model, dataloader, grid, Gaussian count and z axis.
            logger           : Logger used for the progress bar.
            window_kind      : Stitching window name passed to the cube stitcher.
            render_amp_floor : Amplitude floor below which a rendered Gaussian is suppressed.
            render_params    : Whether the model emits Gaussian parameters rather than curves.
        """

        self.loaded        = run
        self.logger        = logger
        self.window_kind   = window_kind
        self.render_params = render_params
        self.renderer      = PredictionCurves(run.n_gaussians, run.z_axis, render_amp_floor) if render_params else None

    def _curves(self, output: np.ndarray) -> np.ndarray:
        """Returns z curves of shape (batch, n_z, row, col) from a model output batch."""

        if not self.render_params:
            return np.asarray(output, dtype=np.float32)

        return self.renderer.render(output)

    def _flipped_output(self, images: torch.Tensor, axis: int) -> np.ndarray:
        """Predicts on the batch flipped along one spatial axis and unflips the output back.

        Args:
            images : Input patch batch of shape (batch, channels, azimuth, range).
            axis   : Tensor axis to flip, 2 for azimuth and 3 for range.

        Returns:
            Model output realigned with the unflipped input, of shape (batch, out_channels, azimuth, range).
        """

        flipped = torch.flip(images, dims=[axis])
        output  = self.loaded.model(flipped)

        return np.flip(output, axis=axis)

    def _batch_disagreement(self, images: torch.Tensor) -> np.ndarray:
        """Returns the mean over flip axes of the curve MSE between the base and flip-restored predictions.

        Args:
            images : Input patch batch of shape (batch, channels, azimuth, range).

        Returns:
            Per-pixel disagreement of shape (batch, azimuth, range).
        """

        base_curves = self._curves(self.loaded.model(images))

        disagreement = np.zeros(base_curves.shape[0:1] + base_curves.shape[2:], dtype=np.float64)
        for axis in self.FLIP_AXES.values():
            flip_curves   = self._curves(self._flipped_output(images, axis))
            disagreement += ((base_curves - flip_curves) ** 2).mean(axis=1)

        return (disagreement / len(self.FLIP_AXES)).astype(np.float32)

    def run(self) -> np.ndarray:
        """Runs the flip evaluation over every patch and stitches the disagreement into one map.

        Returns:
            Flip-equivariance disagreement map of shape (azimuth, range) covering the split region.
        """

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
