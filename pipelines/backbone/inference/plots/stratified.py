"""Figures for error stratification, failure modes and presence calibration.

Turns the binned tables produced by the stratified-error, failure-mode and
presence-calibration analyses into the publication figures of the inference report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipelines.backbone.inference.plots.base import PlotTools


class StratifiedErrorPlotter(PlotTools):
    """Plots binned error curves, failure-mode maps and detection reliability curves.

    Attributes:
        COVARIATE_LABELS : Axis labels, with units, for each supported stratification covariate.
    """

    COVARIATE_LABELS = {
        "gt_active_count"   : "GT active Gaussian count",
        "primary_amplitude" : "Primary channel amplitude",
        "aux_gradient"      : "Aux slope magnitude [per px]",
        "label_r2"          : "Label fit R²",
    }

    def plot_error_curve(self, rows: list[dict], covariate: str, out_path: Path, discrete: bool) -> Path:
        """Plots median pixel curve MSE with an interquartile band against a scene covariate.

        Args:
            rows      : Per-bin records carrying "center", "median", "q25" and "q75".
            covariate : Covariate name, looked up in COVARIATE_LABELS for the x-axis label.
            out_path  : Destination path of the figure.
            discrete  : Whether to place x ticks exactly on the bin centres.

        Returns:
            Path of the saved figure.
        """
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

    def plot_failure_mode_map(self, mode_map, mode_names: tuple, out_path: Path, row_offset: int, col_offset: int) -> Path:
        """Draws the dominant failure mode of every pixel as a discrete categorical map.

        Args:
            mode_map   : Integer mode index per pixel, of shape (row, col), indexing mode_names.
            mode_names : Failure-mode names in index order, rendered as the colourbar legend.
            out_path   : Destination path of the figure.
            row_offset  : Row index of the first row, used for the axis extent.
            col_offset  : Range index of the first column, used for the axis extent.

        Returns:
            Path of the saved figure.
        """
        H, W   = mode_map.shape
        extent = [col_offset, col_offset + W, row_offset + H, row_offset]
        legend = ", ".join(f"{index}={name}" for index, name in enumerate(mode_names))

        return self._imshow_figure(
            mode_map.astype(np.int64),
            x_label        = "Col [px]",
            y_label        = "Row [px]",
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
        """Plots the share of predicted Gaussians matching a ground-truth component per amplitude bin.

        Args:
            rows     : Per-bin records carrying "amp_mean" and "precision" (a fraction in [0, 1]).
            out_path : Destination path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        amps       = [row["amp_mean"] for row in rows]
        precisions = [row["precision"] * 100.0 for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(amps, precisions, marker="o", color="#009E73", linewidth=1.4)
        ax.set_xlabel("Predicted amplitude (bin mean)")
        ax.set_ylabel("Matched to a GT component [%]")
        ax.set_title("Presence reliability: precision by predicted amplitude")
        ax.set_ylim(0.0, 102.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_miss_by_separation(self, rows: list[dict], out_path: Path) -> Path:
        """Plots the fraction of pixels with a missed component against ground-truth component separation.

        Args:
            rows     : Per-bin records carrying "center" (separation in metres) and "miss_rate" in [0, 1].
            out_path : Destination path of the figure.

        Returns:
            Path of the saved figure.
        """
        self._apply_style()

        centers = [row["center"] for row in rows]
        rates   = [row["miss_rate"] * 100.0 for row in rows]

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))
        ax.plot(centers, rates, marker="o", color="#D55E00", linewidth=1.4)
        ax.set_xlabel("Minimum GT component separation")
        ax.set_ylabel("Pixels with a missed component [%]")
        ax.set_title("Miss rate by ground-truth component separation")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, out_path)
