from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from pipelines.backbone.dataset.spatial  import GridInfo
from pipelines.backbone.inference.loader import Run
from tools.monitoring.logger             import Logger


class Stitcher:

    def __init__(self, grid: GridInfo, n_channels: int, window_kind: str = "hann") -> None:
        self.grid   = grid
        self.window = Stitcher.make_patch_window(grid.patch_size, kind=window_kind)

        padded_h, padded_w = grid.padded_size

        self._accum  = np.zeros((int(n_channels), padded_h, padded_w), dtype=np.float64)
        self._weight = np.zeros((padded_h, padded_w), dtype=np.float64)

    @staticmethod
    def make_patch_window(patch_size: Tuple[int, int], kind: str = "hann") -> np.ndarray:
        patch_h, patch_w = patch_size

        if kind == "uniform":
            return np.ones((patch_h, patch_w), dtype=np.float32)

        rows = (np.arange(patch_h) + 0.5) / patch_h
        cols = (np.arange(patch_w) + 0.5) / patch_w

        if kind == "hann":
            window_rows, window_cols = 0.5 - 0.5 * np.cos(2.0 * np.pi * rows), 0.5 - 0.5 * np.cos(2.0 * np.pi * cols)
        else:
            window_rows, window_cols = {"triangular": (1.0 - np.abs(rows * 2.0 - 1.0), 1.0 - np.abs(cols * 2.0 - 1.0))}[kind]

        return np.outer(np.clip(window_rows, 1e-3, None), np.clip(window_cols, 1e-3, None)).astype(np.float32)

    def add_patch(self, idx: int, patch: np.ndarray) -> None:
        v0, h0           = self.grid.patch_origin(idx)
        patch_h, patch_w = self.grid.patch_size

        self._accum[:, v0:v0 + patch_h, h0:h0 + patch_w] += patch * self.window[None]
        self._weight[v0:v0 + patch_h, h0:h0 + patch_w]   += self.window

    def finalize(self) -> np.ndarray:
        H, W         = self.grid.spatial_size
        pad_t, pad_l = self.grid.pad_top, self.grid.pad_left

        stitched = self._accum[:, pad_t:pad_t + H, pad_l:pad_l + W] / self._weight[None, pad_t:pad_t + H, pad_l:pad_l + W]

        return np.ascontiguousarray(stitched, dtype=np.float32)


class Predictor:

    def __init__(self, run: Run, logger: Logger, window_kind: str = "hann") -> None:
        self.run         = run
        self.logger      = logger
        self.window_kind = window_kind

    @torch.no_grad()
    def predict(self) -> np.ndarray:
        stitcher = Stitcher(self.run.grid, self.run.out_channels, window_kind=self.window_kind)
        model    = self.run.model
        index    = 0

        with self.logger.track(transient=True) as progress:
            task = progress.add_task("[section]Predict[/section]", total=len(self.run.loader))

            for inputs, _targets, _weights in self.run.loader:
                outputs = model.denormalize_target(model.module(inputs.to(model.device))).cpu().numpy()

                for patch in outputs:
                    stitcher.add_patch(index, patch)
                    index += 1

                progress.advance(task)

        self.logger.subsection(f"Stitched {index} patches into {self.run.out_channels} target maps of {self.run.grid.spatial_size} with a {self.window_kind} window")

        return stitcher.finalize()
