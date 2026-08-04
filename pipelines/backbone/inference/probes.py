from __future__ import annotations

import numpy as np
import torch

from pipelines.backbone.inference.predictor import Predictor
from tools.data.gaussians                   import GaussianReconstructor


class ModelDevice:

    @staticmethod
    def of(model: torch.nn.Module) -> torch.device:
        parameters = list(model.parameters())
        return parameters[0].device if parameters else torch.device("cpu")


class ProbeWindows:

    def __init__(self, run, window: int) -> None:
        self.run    = run
        self.window = int(window)

    def centers(self, n_row_probes: int, n_col_probes: int) -> list[tuple[int, int]]:
        half   = self.window // 2
        n_rows = self.run.split_region.row_size
        n_cols = self.run.split_region.col_size

        if n_rows < self.window or n_cols < self.window:
            raise ValueError(f"Split region {n_rows}x{n_cols} is smaller than the {self.window}px probe window; choose a smaller window or a larger split")

        row_centers = np.linspace(half, n_rows - half - 1, n_row_probes).round().astype(int)
        col_centers = np.linspace(half, n_cols - half - 1, n_col_probes).round().astype(int)

        return [(int(row), int(col)) for row in row_centers for col in col_centers]

    def assemble(self, centers: list[tuple[int, int]]) -> torch.Tensor:
        half    = self.window // 2
        aux     = self.run.dataset.aux
        windows = []

        for row, col in centers:
            complex_window = self.run.complex_inputs[:, row - half:row + half, col - half:col + half]
            aux_window     = aux[row - half:row + half, col - half:col + half] if aux is not None else None
            windows.append(self.run.dataset.assemble_window(complex_window, aux_window))

        return torch.from_numpy(np.stack(windows)).float()


class PredictionCurves:

    def __init__(self, n_gaussians: int, z_axis: np.ndarray, render_amp_floor: float = 0.0) -> None:
        self.n_gaussians      = int(n_gaussians)
        self.z_axis           = np.asarray(z_axis, dtype=np.float32)
        self.render_amp_floor = float(render_amp_floor)

    def render(self, params: np.ndarray) -> np.ndarray:
        n_K        = self.n_gaussians
        B, _, H, W = params.shape

        x     = self.z_axis.reshape(1, 1, -1, 1, 1)
        gauss = params[:, :n_K * 3].reshape(B, n_K, 3, H, W).astype(np.float32)
        gauss = Predictor._render_masked(gauss, self.render_amp_floor)

        return GaussianReconstructor.reconstruct_batch(gauss, x)
