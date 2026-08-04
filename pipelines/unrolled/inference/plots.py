from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np

from pipelines.unrolled.inference.predictor import UnrolledPrediction
from tools.reporting.plotting               import PlotBase


class UnrolledPlots(PlotBase):
    ERROR_CMAP  = "magma"
    HEIGHT_CMAP = "viridis"

    def __init__(self, fig_dpi: int = 150, save_dpi: int = 300) -> None:
        self.fig_dpi  = fig_dpi
        self.save_dpi = save_dpi
        self._apply_style()

    @staticmethod
    def _masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out         = values.astype(np.float32).copy()
        out[~valid] = np.nan

        return out

    def _error_maps(self, prediction: UnrolledPrediction, figures_dir: Path) -> Dict[str, List[Path]]:
        l1   = self._masked(prediction.curve_l1_map,   prediction.valid_mask)
        peak = self._masked(prediction.peak_error_map, prediction.valid_mask)
        cmap = self._cmap_with_bad(self.ERROR_CMAP)

        l1_path   = self._imshow_figure(l1,   x_label="range (px)", y_label="azimuth (px)", title="Per-pixel curve L1 error",     cmap=cmap, colorbar_label="L1 error",           path=figures_dir / "curve_l1_map.png")
        peak_path = self._imshow_figure(peak, x_label="range (px)", y_label="azimuth (px)", title="Per-pixel peak position error", cmap=cmap, colorbar_label="peak error (m)",     path=figures_dir / "peak_error_map.png")

        return {"curve_l1_map": [l1_path], "peak_error_map": [peak_path]}

    def _error_histogram(self, prediction: UnrolledPrediction, figures_dir: Path) -> Path:
        values = np.log10(np.maximum(prediction.curve_l1_map[prediction.valid_mask], 1e-12))

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(values, bins=min(60, max(1, values.size)), color="#34495e", edgecolor="white", linewidth=0.3)

        ax.set_xlabel(r"$\log_{10}$ per-pixel curve L1 error")
        ax.set_ylabel("count")
        ax.set_title("Curve error distribution")

        return self._save(fig, figures_dir / "error_histogram.png")

    def _peak_height_maps(self, prediction: UnrolledPrediction, figures_dir: Path) -> List[Path]:
        gt   = self._masked(prediction.gt_peak_height_map,   prediction.valid_mask)
        pred = self._masked(prediction.pred_peak_height_map, prediction.valid_mask)

        vmin, vmax = self._shared_clim(gt, pred)
        cmap       = self._cmap_with_bad(self.HEIGHT_CMAP)

        gt_path   = self._imshow_figure(gt,   x_label="range (px)", y_label="azimuth (px)", title="Ground-truth peak height",  cmap=cmap, vmin=vmin, vmax=vmax, colorbar_label="height (m)", path=figures_dir / "gt_peak_height.png")
        pred_path = self._imshow_figure(pred, x_label="range (px)", y_label="azimuth (px)", title="Predicted peak height",     cmap=cmap, vmin=vmin, vmax=vmax, colorbar_label="height (m)", path=figures_dir / "pred_peak_height.png")

        return [gt_path, pred_path]

    def _profile_overlay(self, example: dict, x_axis: np.ndarray, label: str, rank: int, figures_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

        ax.plot(x_axis, example["gt"],   color="#1f4e79", linewidth=1.4, label="ground truth")
        ax.plot(x_axis, example["pred"], color="#b03a2e", linewidth=1.4, linestyle="--", label="prediction")

        l1 = float(np.abs(example["pred"] - example["gt"]).mean())

        ax.set_xlabel("height (m)")
        ax.set_ylabel("normalized power")
        ax.set_title(f"{label} #{rank + 1}  pixel ({example['azimuth']}, {example['range']})  L1 = {l1:.4g}")
        ax.legend(frameon=False)

        return self._save(fig, figures_dir / "profiles" / f"{label}_{rank + 1:02d}.png")

    def compose(self, prediction: UnrolledPrediction, examples: Dict[str, List[dict]], x_axis: np.ndarray, figures_dir: Path) -> Dict[str, List[Path]]:
        figures = self._error_maps(prediction, figures_dir)

        figures["error_histogram"] = [self._error_histogram(prediction, figures_dir)]
        figures["peak_heights"]    = self._peak_height_maps(prediction, figures_dir)

        for label, pixel_examples in examples.items():
            figures[label] = [self._profile_overlay(example, x_axis, label, rank, figures_dir) for rank, example in enumerate(pixel_examples)]

        return figures
