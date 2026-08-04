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

    def centers(self, n_azimuth: int, n_range: int) -> list[tuple[int, int]]:
        half = self.window // 2
        n_az = self.run.split_region.azimuth_size
        n_rg = self.run.split_region.range_size

        if n_az < self.window or n_rg < self.window:
            raise ValueError(f"Split region {n_az}x{n_rg} is smaller than the {self.window}px probe window; choose a smaller window or a larger split")

        az_centers = np.linspace(half, n_az - half - 1, n_azimuth).round().astype(int)
        rg_centers = np.linspace(half, n_rg - half - 1, n_range).round().astype(int)

        return [(int(az), int(rg)) for az in az_centers for rg in rg_centers]

    def assemble(self, centers: list[tuple[int, int]]) -> torch.Tensor:
        half    = self.window // 2
        dem     = self.run.dataset.dem
        windows = []

        for az, rg in centers:
            complex_window = self.run.complex_inputs[:, az - half:az + half, rg - half:rg + half]
            dem_window     = dem[az - half:az + half, rg - half:rg + half] if dem is not None else None
            windows.append(self.run.dataset.assemble_window(complex_window, dem_window))

        return torch.from_numpy(np.stack(windows)).float()


class PredictionCurves:

    def __init__(self, n_gaussians: int, x_axis: np.ndarray, render_amp_floor: float = 0.0) -> None:
        self.n_gaussians      = int(n_gaussians)
        self.x_axis           = np.asarray(x_axis, dtype=np.float32)
        self.render_amp_floor = float(render_amp_floor)

    def render(self, params: np.ndarray) -> np.ndarray:
        n_K        = self.n_gaussians
        B, _, H, W = params.shape

        x     = self.x_axis.reshape(1, 1, -1, 1, 1)
        gauss = params[:, :n_K * 3].reshape(B, n_K, 3, H, W).astype(np.float32)
        gauss = Predictor._render_masked(gauss, self.render_amp_floor)

        return GaussianReconstructor.reconstruct_batch(gauss, x)
