"""Figures comparing the number of active Gaussian slots per pixel against ground truth.

Renders the two active-count maps of the inference report: a categorical agreement map
(correct / under / over) and the signed count difference between prediction and labels.
"""

from __future__ import annotations

from pathlib import Path
from typing  import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import numpy                as np
from matplotlib.patches     import Patch

from pipelines.backbone.inference.plots.base import PlotTools
from tools.loss.param_loss                   import ParamMatcher


class SlotPlotter(PlotTools):
    """Draws per-pixel active-Gaussian-count maps for a stitched inference cube."""

    def plot_active_count_map(
        self,
        params_pred   : np.ndarray,
        params_gt     : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        row_offset    : int,
        col_offset    : int,
        amp_threshold : float = ParamMatcher.ACTIVE_AMP_THR,
    ) -> List[Path]:
        """Saves the active-count agreement map and the signed count-difference map.

        Args:
            params_pred   : Predicted Gaussian parameter cube of shape (3 * n_gaussians, rows, cols), interleaved as (amplitude, mu, sigma) per slot.
            params_gt     : Ground-truth Gaussian parameter cube in the same layout as params_pred.
            n_gaussians   : Number of Gaussian slots to count over.
            out_dir       : Directory the two PNG figures are written into.
            az_offset     : Azimuth index of the first row of the cube, used for axis extents.
            rg_offset     : Range index of the first column of the cube, used for axis extents.
            amp_threshold : Amplitude above which a slot counts as active.

        Returns:
            Paths of the agreement map and the signed-difference map, in that order.
        """

        gt_count   = np.zeros(params_gt  .shape[-2:], dtype=np.int32)
        pred_count = np.zeros(params_pred.shape[-2:], dtype=np.int32)

        for k in range(n_gaussians):
            gt_count   += (params_gt  [3 * k] >= amp_threshold).astype(np.int32)
            pred_count += (params_pred[3 * k] >= amp_threshold).astype(np.int32)

        diff = pred_count - gt_count

        H, W = diff.shape
        rgb  = np.zeros((H, W, 3), dtype=np.float32)
        rgb[diff == 0] = [0.20, 0.75, 0.20]
        rgb[diff <  0] = [0.20, 0.45, 0.90]
        rgb[diff >  0] = [0.90, 0.25, 0.25]

        extent = [col_offset, col_offset + W, row_offset + H, row_offset]

        n_total   = H * W
        n_correct = int((diff == 0).sum())
        n_under   = int((diff <  0).sum())
        n_over    = int((diff >  0).sum())

        paths = []

        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=extent)
        ax.set_xlabel("col index")
        ax.set_ylabel("row index")
        ax.set_title("Active-count agreement per pixel")

        legend_els = [
            Patch(facecolor=[0.20, 0.75, 0.20], label=f"correct  ({n_correct / n_total * 100:.1f}%)"),
            Patch(facecolor=[0.20, 0.45, 0.90], label=f"under    ({n_under   / n_total * 100:.1f}%)"),
            Patch(facecolor=[0.90, 0.25, 0.25], label=f"over     ({n_over    / n_total * 100:.1f}%)"),
        ]
        ax.legend(handles=legend_els, loc="lower right", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "active_count_agreement.png"))

        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        vabs = max(1, int(np.abs(diff).max()))
        im   = ax.imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto", interpolation="nearest", extent=extent)
        ax.set_xlabel("col index")
        ax.set_ylabel("row index")
        ax.set_title("Signed count difference  (pred − GT)")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("pred − GT  [#Gaussians]")
        cb.set_ticks(range(-vabs, vabs + 1))
        fig.tight_layout()
        paths.append(self._save(fig, out_dir / "active_count_difference.png"))

        return paths
