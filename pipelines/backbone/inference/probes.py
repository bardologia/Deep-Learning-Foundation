
from __future__ import annotations

import numpy as np
import torch


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

        row_centers = np.linspace(half, n_rows - self.window + half, n_row_probes).round().astype(int)
        col_centers = np.linspace(half, n_cols - self.window + half, n_col_probes).round().astype(int)

        return [(int(row), int(col)) for row in row_centers for col in col_centers]

    def assemble(self, centers: list[tuple[int, int]]) -> torch.Tensor:
        half    = self.window // 2
        windows = []

        for row, col in centers:
            row0, col0 = row - half, col - half
            windows.append(self.run.dataset.window(self.run.inputs[:, row0:row0 + self.window, col0:col0 + self.window]))

        return torch.from_numpy(np.stack(windows)).float()
