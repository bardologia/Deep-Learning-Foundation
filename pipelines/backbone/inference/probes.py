"""Window sampling and curve rendering helpers shared by the analysis probes.

Provides the pieces the receptive-field, attribution and activation analyses need: the
device a model lives on, evenly spread square windows cut from the inference split, and
rendering of predicted Gaussian parameters into elevation curves.
"""

from __future__ import annotations

import numpy as np
import torch

from pipelines.backbone.inference.predictor import Predictor
from tools.data.gaussians                   import GaussianReconstructor


class ModelDevice:
    """Resolves the device a model's parameters live on."""

    @staticmethod
    def of(model: torch.nn.Module) -> torch.device:
        """Returns the device of the first parameter, or CPU for a parameterless model."""
        parameters = list(model.parameters())
        return parameters[0].device if parameters else torch.device("cpu")


class ProbeWindows:
    """Cuts square input windows out of the inference split at a regular grid of centres.

    Attributes:
        run    : Loaded run supplying the split region, the complex stack and the dataset assembler.
        window : Side length of every probe window in pixels.
    """

    def __init__(self, run, window: int) -> None:
        """Stores the run handle and the probe window side length in pixels."""
        self.run    = run
        self.window = int(window)

    def centers(self, n_row_probes: int, n_col_probes: int) -> list[tuple[int, int]]:
        """Returns window centres spread evenly over the split region.

        Args:
            n_row_probes : Number of centres along the rows.
            n_col_probes : Number of centres along the columns.

        Returns:
            Centre pixels as (row, col) index pairs, in row-major order, placed so
            every window stays inside the split region.

        Raises:
            ValueError : If the split region is smaller than the probe window.
        """
        half   = self.window // 2
        n_rows = self.run.split_region.row_size
        n_cols = self.run.split_region.col_size

        if n_rows < self.window or n_cols < self.window:
            raise ValueError(f"Split region {n_rows}x{n_cols} is smaller than the {self.window}px probe window; choose a smaller window or a larger split")

        row_centers = np.linspace(half, n_rows - self.window + half, n_row_probes).round().astype(int)
        col_centers = np.linspace(half, n_cols - self.window + half, n_col_probes).round().astype(int)

        return [(int(row), int(col)) for row in row_centers for col in col_centers]

    def assemble(self, centers: list[tuple[int, int]]) -> torch.Tensor:
        """Assembles the model input tensor for every requested window centre.

        Args:
            centers : Window centres as (azimuth, range) index pairs into the split region.

        Returns:
            Batched model inputs of shape (n_centers, in_channels, window, window), built by
            the dataset's own channel assembly from the complex stack and the DEM.
        """
        half    = self.window // 2
        aux     = self.run.dataset.aux
        windows = []

        for row, col in centers:
            row0, col0       = row - half, col - half
            complex_window = self.run.complex_inputs[:, row0:row0 + self.window, col0:col0 + self.window]
            aux_window     = aux[row0:row0 + self.window, col0:col0 + self.window] if aux is not None else None
            windows.append(self.run.dataset.assemble_window(complex_window, aux_window))

        return torch.from_numpy(np.stack(windows)).float()


class PredictionCurves:
    """Renders predicted Gaussian parameters into elevation curves on a fixed axis.

    Attributes:
        n_gaussians      : Number of Gaussian slots read from the parameter tensor.
        x_axis           : Elevation sampling axis in metres, of shape (n_elevation,).
        render_amp_floor : Amplitude below which a Gaussian is suppressed before rendering.
    """

    def __init__(self, n_gaussians: int, z_axis: np.ndarray, render_amp_floor: float = 0.0) -> None:
        """Stores the slot count, the z axis in metres and the render amplitude floor."""
        self.n_gaussians      = int(n_gaussians)
        self.z_axis           = np.asarray(z_axis, dtype=np.float32)
        self.render_amp_floor = float(render_amp_floor)

    def render(self, params: np.ndarray) -> np.ndarray:
        """Reconstructs elevation curves from a batch of predicted Gaussian parameters.

        Args:
            params : Predicted parameters of shape (batch, channels, height, width), the first 3 * n_gaussians channels interleaving amplitude, mu in metres and sigma in metres.

        Returns:
            Rendered curves of shape (batch, n_elevation, height, width).
        """
        n_K        = self.n_gaussians
        B, _, H, W = params.shape

        x     = self.z_axis.reshape(1, 1, -1, 1, 1)
        gauss = params[:, :n_K * 3].reshape(B, n_K, 3, H, W).astype(np.float32)
        gauss = Predictor._render_masked(gauss, self.render_amp_floor)

        return GaussianReconstructor.reconstruct_batch(gauss, x)
