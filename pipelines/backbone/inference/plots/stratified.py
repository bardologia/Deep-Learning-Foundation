from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipelines.backbone.inference.plots.base import PlotTools


class StratifiedErrorPlotter(PlotTools):

    COVARIATE_LABELS = {
        "gt_active_count"   : "GT active Gaussian count",
        "primary_amplitude" : "Primary SLC amplitude",
        "coherence"         : "Mean coherence across secondaries",
        "dem_slope"         : "DEM slope magnitude [m/px]",
        "label_r2"          : "Label fit R²",
    }

    def plot_error_curve(self, rows: list[dict], covariate: str, out_path: Path, discrete: bool) -> Path:
        self._apply_style()

        centers = [row["center"] for row in rows]
        medians = [row["median"] for row in rows]
        q25     = [row["q25"] for row in rows]
        q75     = [row["q75"] for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.fill_between(centers, q25, q75, alpha=0.25, color="#0072B2", linewidth=0)
        ax.plot(centers, medians, marker="o", color="#0072B2", linewidth=1.4)

        ax.set_xlabel(self.COVARIATE_LABELS.get(covariate, covariate))
        ax.set_ylabel("Pixel curve MSE (median, IQR band)")
        ax.set_yscale("log")
        ax.set_title(f"Error stratified by {self.COVARIATE_LABELS.get(covariate, covariate).lower()}")
        if discrete:
            ax.set_xticks(centers)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_failure_mode_map(self, mode_map, mode_names: tuple, out_path: Path, az_offset: int, rg_offset: int) -> Path:
        H, W   = mode_map.shape
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        legend = ", ".join(f"{index}={name}" for index, name in enumerate(mode_names))

        return self._imshow_figure(
            mode_map.astype(np.int64),
            x_label        = "Range [px]",
            y_label        = "Azimuth [px]",
            title          = "Dominant failure mode per pixel",
            cmap           = None,
            extent         = extent,
            discrete       = True,
            levels         = list(range(len(mode_names))),
            colorbar_label = "Mode",
            text_overlay   = legend,
            path           = out_path,
        )

    def plot_presence_reliability(self, rows: list[dict], out_path: Path) -> Path:
        self._apply_style()

        amps       = [row["amp_mean"] for row in rows]
        precisions = [row["precision"] * 100.0 for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(amps, precisions, marker="o", color="#009E73", linewidth=1.4)
        ax.set_xlabel("Predicted amplitude (bin mean)")
        ax.set_ylabel("Matched to a GT scatterer [%]")
        ax.set_title("Presence reliability: precision by predicted amplitude")
        ax.set_ylim(0.0, 102.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_miss_by_separation(self, rows: list[dict], out_path: Path) -> Path:
        self._apply_style()

        centers = [row["center"] for row in rows]
        rates   = [row["miss_rate"] * 100.0 for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(centers, rates, marker="o", color="#D55E00", linewidth=1.4)
        ax.set_xlabel("Minimum GT scatterer separation [m]")
        ax.set_ylabel("Pixels with a missed scatterer [%]")
        ax.set_title("Miss rate by ground-truth scatterer separation")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)
