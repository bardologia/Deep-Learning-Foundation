"""Draws how a set-prediction head organises its raw output slots.

The figures show which slots the model actually uses, what parameter ranges each slot
covers, and whether slot index tracks either the within-pixel μ ordering or the matched
ground-truth index -- the signatures of slot collapse and of slot specialisation.
"""

from __future__ import annotations

from pathlib import Path
from typing  import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.backbone.inference.plots.base import PlotTools
from tools.metrics.slot_organization         import SlotOrganization


class SlotOrganizationPlotter(PlotTools):
    """Plots slot-usage, per-slot parameter distributions and slot-ordering matrices."""

    def plot_slot_usage(
        self,
        params_pred   : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        amp_threshold : float,
    ) -> List[Path]:
        """Plots the fraction of pixels on which each raw output slot is active.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figure is written into.
            amp_threshold : Amplitude above which a slot counts as active, as recorded by the label run.

        Returns:
            Single-element list holding the written figure path.
        """

        usage   = SlotOrganization.usage_fractions(params_pred, n_gaussians, amp_threshold)
        entropy = SlotOrganization.usage_entropy(usage)
        slots   = np.arange(n_gaussians)
        colors  = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, n_gaussians))

        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.bar(slots, usage, color=colors, edgecolor="black", linewidth=0.4)

        uniform = float(usage.mean())
        ax.axhline(uniform, color="0.3", linestyle="--", linewidth=0.9, label=f"mean use = {uniform:.3f}")

        ax.set_xticks(slots)
        ax.set_xlabel("raw predicted slot (output-channel index)")
        ax.set_ylabel("active-pixel fraction")
        ax.set_title(f"Per-slot activation frequency  (usage entropy = {entropy:.3f})", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        fig.tight_layout()

        return [self._save(fig, out_dir / "slot_usage.png")]

    def plot_slot_param_distributions(
        self,
        params_pred   : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        amp_threshold : float,
        bins          : int = 80,
    ) -> List[Path]:
        """Plots one overlaid histogram per parameter, with a curve per raw slot over active pixels.

        Amplitude is binned and drawn on a logarithmic axis; μ and σ use linear bins over
        the central 99% of their values.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figures are written into.
            amp_threshold : Amplitude above which a slot counts as active, as recorded by the label run.
            bins          : Number of histogram bins.

        Returns:
            Paths of the written figures, one per parameter that had populated slots.
        """

        amp_all = np.stack([params_pred[3 * k] for k in range(n_gaussians)], axis=0).reshape(n_gaussians, -1)
        active  = amp_all >= amp_threshold
        colors  = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, n_gaussians))
        paths   = []

        for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
            short    = self.PARAM_SHORT[j]
            per_slot = []
            for k in range(n_gaussians):
                v = params_pred[3 * k + j].reshape(-1)[active[k]]
                per_slot.append(v[np.isfinite(v)])

            populated = [v for v in per_slot if v.size]
            if not populated:
                continue
            combined = np.concatenate(populated)

            is_amp = j == 0
            if is_amp:
                positive = combined[combined > 0]
                if positive.size == 0:
                    continue
                lo        = max(float(np.percentile(positive, 0.5)), 1e-6)
                hi        = float(positive.max()) * 1.02
                bin_edges = np.geomspace(lo, hi, bins + 1)
            else:
                lo        = float(np.percentile(combined, 0.5))
                hi        = float(np.percentile(combined, 99.5))
                bin_edges = np.linspace(lo, hi, bins + 1)

            fig, ax = plt.subplots(figsize=(5.6, 3.8))
            for k in range(n_gaussians):
                v = per_slot[k]
                v = v[(v >= bin_edges[0]) & (v <= bin_edges[-1])]
                if v.size == 0:
                    continue
                ax.hist(v, bins=bin_edges, density=True, histtype="step", linewidth=1.3, color=colors[k], label=f"slot {k}")

            if is_amp:
                ax.set_xscale("log")

            ax.set_xlabel(short)
            ax.set_ylabel("density")
            ax.set_title(f"Per-slot {lbl}  (raw slots, active pixels)", fontsize=10)
            ax.legend(fontsize=7, framealpha=0.9, ncol=2)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"slot_dist_{fname}.png"))

        return paths

    def plot_mu_rank_matrix(
        self,
        params_pred   : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        amp_threshold : float,
    ) -> List[Path]:
        """Plots the row-normalised matrix of raw slot index against within-pixel μ rank.

        A diagonal matrix means the head emits its Gaussians in z order.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figure is written into.
            amp_threshold : Amplitude above which a slot counts as active, as recorded by the label run.

        Returns:
            Single-element list holding the written figure path.
        """

        counts = SlotOrganization.mu_rank_matrix(params_pred, n_gaussians, amp_threshold)
        diag   = SlotOrganization.diagonality(counts)
        matrix = SlotOrganization.row_normalized(counts)

        path = self._matrix_panel(
            matrix  = matrix,
            title   = "Slot index vs μ-rank",
            x_label = "μ-rank within pixel (0 = lowest μ)",
            y_label = "raw predicted slot",
            diag    = diag,
            path    = out_dir / "slot_mu_rank_matrix.png",
        )

        return [path]

    def plot_assignment_matrix(
        self,
        params_pred   : np.ndarray,
        params_gt     : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        amp_threshold : float,
        assignment    : np.ndarray | None = None,
    ) -> List[Path]:
        """Plots the row-normalised matrix of raw slot index against matched ground-truth index.

        A diagonal matrix means each slot consistently claims the same GT component.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            params_gt     : Ground-truth parameter cube of the same shape.
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figure is written into.
            amp_threshold : Amplitude above which a slot counts as active, as recorded by the label run.
            assignment    : Precomputed predicted-to-GT assignment, or None to match here.

        Returns:
            Single-element list holding the written figure path.
        """

        counts = SlotOrganization.assignment_matrix(params_pred, params_gt, n_gaussians, amp_threshold, assignment=assignment)
        diag   = SlotOrganization.diagonality(counts)
        matrix = SlotOrganization.row_normalized(counts)

        path = self._matrix_panel(
            matrix  = matrix,
            title   = "Slot index vs matched GT index",
            x_label = "matched GT slot (μ-sorted index)",
            y_label = "raw predicted slot",
            diag    = diag,
            path    = out_dir / "slot_assignment_matrix.png",
        )

        return [path]

    def _matrix_panel(self, matrix: np.ndarray, title: str, x_label: str, y_label: str, diag: float, path: Path) -> Path:
        """Renders a square annotated probability matrix with its diagonality in the title.

        Args:
            matrix  : Row-normalised matrix of shape (K, K).
            title   : Panel title, appended with the diagonality score.
            x_label : Horizontal axis label.
            y_label : Vertical axis label.
            diag    : Diagonality score in [0, 1] quoted in the title.
            path    : Destination file path.

        Returns:
            The written file path.
        """

        n = matrix.shape[0]

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im      = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal", origin="upper")

        for i in range(n):
            for k in range(n):
                val = float(matrix[i, k])
                ax.text(k, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val < 0.6 else "black")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}  (diagonality = {diag:.3f})", fontsize=10)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("row-normalised probability")
        fig.tight_layout()

        return self._save(fig, path)
