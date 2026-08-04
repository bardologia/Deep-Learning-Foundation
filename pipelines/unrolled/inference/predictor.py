from __future__ import annotations

from dataclasses import dataclass
from typing      import Optional

import numpy as np
import torch

from configuration.inference.unrolled    import UnrolledInferenceConfig
from pipelines.unrolled.inference.loader import UnrolledRun
from pipelines.unrolled.synthesis        import MeasurementSynthesiser
from tools.monitoring.logger             import Logger


@dataclass
class UnrolledPrediction:
    curve_l1_map         : np.ndarray
    curve_mse_map        : np.ndarray
    peak_error_map       : np.ndarray
    gt_peak_height_map   : np.ndarray
    pred_peak_height_map : np.ndarray
    valid_mask           : np.ndarray
    profile_cube         : Optional[np.ndarray]


class UnrolledPredictor:
    def __init__(self, run: UnrolledRun, config: UnrolledInferenceConfig, logger: Logger) -> None:
        self.run    = run
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)

        self.x_axis      = torch.from_numpy(np.asarray(run.x_axis, dtype=np.float32)).to(self.device)
        self.synthesiser = MeasurementSynthesiser(self.x_axis, run.ppg, run.power_floor, run.noise_std)

        self.cached_chunk = None
        self.cached_pair  = None

        self.run.model.to(self.device)

    def _chunk_rows(self) -> int:
        _, height, width = self.run.gt_parameters.shape
        n_bins           = int(self.x_axis.shape[0])

        return max(1, min(height, int(self.config.chunk_cells) // max(1, n_bins * width)))

    def _noise_generator(self, row_start: int) -> Optional[torch.Generator]:
        if self.run.noise_std <= 0.0:
            return None

        generator = torch.Generator(device=self.device)
        generator.manual_seed(row_start)

        return generator

    @torch.no_grad()
    def _predict_rows(self, row_start: int, row_end: int) -> tuple:
        gt = torch.from_numpy(np.ascontiguousarray(self.run.gt_parameters[:, row_start:row_end])).unsqueeze(0).to(self.device)
        kz = torch.from_numpy(np.ascontiguousarray(self.run.kz_field[:, row_start:row_end])).unsqueeze(0).to(self.device)

        measurements, target, mask = self.synthesiser.synthesise(gt, kz, self._noise_generator(row_start))
        pred                       = self.run.model(measurements, kz, self.x_axis)

        return pred[0], target[0], mask[0]

    @torch.no_grad()
    def run_inference(self) -> UnrolledPrediction:
        _, height, width = self.run.gt_parameters.shape

        l1_map    = np.zeros((height, width), dtype=np.float32)
        mse_map   = np.zeros((height, width), dtype=np.float32)
        peak_map  = np.zeros((height, width), dtype=np.float32)
        gt_peak   = np.zeros((height, width), dtype=np.float32)
        pred_peak = np.zeros((height, width), dtype=np.float32)
        valid     = np.zeros((height, width), dtype=bool)

        cube = np.zeros((int(self.x_axis.shape[0]), height, width), dtype=np.float32) if self.config.save_profile_cube else None

        rows = self._chunk_rows()
        self.logger.subsection(f"Predicting {height}x{width} pixels in row chunks of {rows}")

        for row_start in range(0, height, rows):
            row_end = min(row_start + rows, height)

            pred, target, mask = self._predict_rows(row_start, row_end)
            diff               = pred - target

            pred_heights = self.x_axis[pred.argmax(dim=0)]
            gt_heights   = self.x_axis[target.argmax(dim=0)]

            l1_map   [row_start:row_end] = diff.abs().mean(dim=0).cpu().numpy()
            mse_map  [row_start:row_end] = diff.pow(2).mean(dim=0).cpu().numpy()
            peak_map [row_start:row_end] = (pred_heights - gt_heights).abs().cpu().numpy()
            gt_peak  [row_start:row_end] = gt_heights.cpu().numpy()
            pred_peak[row_start:row_end] = pred_heights.cpu().numpy()
            valid    [row_start:row_end] = mask.cpu().numpy()

            if cube is not None:
                cube[:, row_start:row_end] = pred.cpu().numpy()

        return UnrolledPrediction(
            curve_l1_map         = l1_map,
            curve_mse_map        = mse_map,
            peak_error_map       = peak_map,
            gt_peak_height_map   = gt_peak,
            pred_peak_height_map = pred_peak,
            valid_mask           = valid,
            profile_cube         = cube,
        )

    @torch.no_grad()
    def profile_pair(self, azimuth: int, range_index: int) -> dict:
        _, height, _ = self.run.gt_parameters.shape

        rows        = self._chunk_rows()
        chunk_start = (int(azimuth) // rows) * rows
        chunk_end   = min(chunk_start + rows, height)
        chunk_row   = int(azimuth) - chunk_start

        if self.cached_chunk != (chunk_start, chunk_end):
            pred, target, _mask = self._predict_rows(chunk_start, chunk_end)
            self.cached_chunk   = (chunk_start, chunk_end)
            self.cached_pair    = (pred, target)

        pred, target = self.cached_pair

        return {
            "azimuth" : int(azimuth),
            "range"   : int(range_index),
            "gt"      : target[:, chunk_row, range_index].cpu().numpy(),
            "pred"    : pred[:, chunk_row, range_index].cpu().numpy(),
        }
