"""Draws Gaussian-parameter accuracy after matching predicted slots to ground-truth slots.

Every figure is scoped to the pixels where the ground-truth Gaussian is active, so
inactive slots cannot dilute the comparison, and all of them read the same cached
Hungarian alignment.
"""

from __future__ import annotations

from pathlib import Path
from typing  import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors    as mcolors
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.backbone.inference.plots.base import PlotTools
from tools.loss.param_loss                   import ParamMatcher
from tools.metrics.gaussian_matching         import GaussianMatcher


class SlotAlignment:
    """Caches the slot-aligned prediction so the parameter figures match only once.

    Attributes:
        params_pred   : Predicted parameter cube the cached alignment was built from.
        params_gt     : Ground-truth parameter cube the cached alignment was built against.
        n_gaussians   : Gaussian count K of the cached alignment.
        amp_threshold : Activity threshold of the cached alignment.
        aligned       : Predicted parameters reordered into GT slot order.
    """

    def __init__(self) -> None:
        """Starts with an empty cache."""

        self.params_pred   = None
        self.params_gt     = None
        self.n_gaussians   = 0
        self.amp_threshold = None
        self.aligned       = None

    @staticmethod
    def _is_same(cached: Optional[np.ndarray], candidate: np.ndarray) -> bool:
        """Returns whether the candidate array is the identical buffer, shape and strides as the cached one."""

        if cached is None:
            return False

        return cached.shape == candidate.shape and cached.strides == candidate.strides and cached.ctypes.data == candidate.ctypes.data

    def of(self, params_pred: np.ndarray, params_gt: np.ndarray, n_gaussians: int, amp_threshold: float, assignment: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the slot-aligned prediction, reusing the cache when the same cubes come back.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            params_gt     : Ground-truth parameter cube of the same shape.
            n_gaussians   : Gaussian count K.
            amp_threshold : Amplitude above which a slot counts as active.
            assignment    : Precomputed predicted-to-GT assignment, or None to match here.

        Returns:
            Predicted parameters reordered into GT slot order, of shape (3 * K, rows, cols).
        """

        if self.n_gaussians == n_gaussians and self.amp_threshold == amp_threshold and self._is_same(self.params_pred, params_pred) and self._is_same(self.params_gt, params_gt):
            return self.aligned

        self.params_pred   = params_pred
        self.params_gt     = params_gt
        self.n_gaussians   = n_gaussians
        self.amp_threshold = amp_threshold
        self.aligned       = GaussianMatcher(amp_threshold=amp_threshold).aligned_prediction(params_pred, params_gt, n_gaussians, assignment=assignment)

        return self.aligned


class ParamPlotter(PlotTools):
    """Plots matched Gaussian-parameter distributions, scatters, error maps and error histograms.

    Attributes:
        alignment : Cache of the slot-aligned prediction shared by every figure family here.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Builds the plotter with the shared plot style and an empty alignment cache."""

        super().__init__(*args, **kwargs)
        self.alignment = SlotAlignment()

    def plot_param_distributions(
        self,
        params_pred   : np.ndarray,
        params_gt     : Optional[np.ndarray],
        n_gaussians   : int,
        out_dir       : Path,
        assignment    : Optional[np.ndarray] = None,
        bins          : int   = 80,
        amp_threshold : float = ParamMatcher.ACTIVE_AMP_THR,
    ) -> List[Path]:
        """Plots predicted-versus-GT histograms of each parameter, one figure per Gaussian and parameter.

        With ground truth present the prediction is slot-aligned first and both
        distributions are restricted to the pixels where that GT Gaussian is active;
        without it, the raw predicted distribution is drawn alone. Amplitude uses
        logarithmic bins.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            params_gt     : Ground-truth parameter cube of the same shape, or None.
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figures are written into.
            assignment    : Precomputed predicted-to-GT assignment, or None to match here.
            bins          : Number of histogram bins.
            amp_threshold : Amplitude above which a slot counts as active.

        Returns:
            Paths of the written figures.
        """

        matched     = params_gt is not None
        pred_source = self.alignment.of(params_pred, params_gt, n_gaussians, amp_threshold, assignment) if matched else params_pred
        paths       = []

        for k in range(n_gaussians):
            amp_ch = 3 * k
            if matched and amp_ch < params_gt.shape[0]:
                gt_amp_flat = params_gt[amp_ch].reshape(-1)
                active_mask = np.isfinite(gt_amp_flat) & ParamMatcher.is_active(gt_amp_flat, amp_threshold)
            else:
                active_mask = None

            for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                pred_flat = pred_source[ch].reshape(-1)
                if active_mask is not None:
                    pred_flat = pred_flat[active_mask]
                pred = pred_flat[np.isfinite(pred_flat)]

                gt = np.empty(0, dtype=np.float64)
                if params_gt is not None and ch < params_gt.shape[0]:
                    gt_flat = params_gt[ch].reshape(-1)
                    if active_mask is not None:
                        gt_flat = gt_flat[active_mask]
                    gt = gt_flat[np.isfinite(gt_flat)]

                has_pred = pred.size > 0
                has_gt   = gt.size > 0
                if not has_pred and not has_gt:
                    continue

                combined = np.concatenate([arr for arr in (pred, gt) if arr.size])
                is_amp   = j == 0

                if is_amp:
                    positive  = combined[combined > 0]
                    if positive.size == 0:
                        continue
                    lo        = max(float(np.percentile(positive, 0.5)), 1e-6)
                    hi        = float(positive.max()) * 1.02
                    bin_edges = np.geomspace(lo, hi, bins + 1)
                else:
                    lo        = float(np.percentile(combined, 0.5))
                    hi        = float(np.percentile(combined, 99.5))
                    bin_edges = np.linspace(lo, hi, bins + 1)

                gt_in   = gt  [(gt   >= bin_edges[0]) & (gt   <= bin_edges[-1])]
                pred_in = pred[(pred >= bin_edges[0]) & (pred <= bin_edges[-1])]
                if gt_in.size == 0 and pred_in.size == 0:
                    continue

                fig, ax = plt.subplots(figsize=(4.8, 3.4))

                if gt_in.size:
                    ax.hist(gt_in, bins=bin_edges, density=True, color="C0", alpha=0.55, label="GT", edgecolor="none")
                    ax.axvline(float(np.median(gt_in)), color="C0", linestyle="--", linewidth=0.9, label=f"med GT={np.median(gt_in):.3g}")

                if pred_in.size:
                    pred_label = "Matched pred" if matched else "Pred"
                    ax.hist(pred_in, bins=bin_edges, density=True, color="C3", alpha=0.55, label=pred_label, edgecolor="none")
                    ax.axvline(float(np.median(pred_in)), color="C3", linestyle="--", linewidth=0.9, label=f"med {pred_label}={np.median(pred_in):.3g}")

                tag   = "GT g" if matched else "g"
                scope = "matched, active" if matched else "pred only"

                if is_amp:
                    ax.set_xscale("log")
                    ax.set_title(f"{tag}{k + 1} — {lbl}  ({scope}, max={float(combined.max()):.3g})", fontsize=10)
                else:
                    ax.set_title(f"{tag}{k + 1} — {lbl}  ({scope})", fontsize=10)

                ax.set_xlabel(short)
                ax.set_ylabel("density")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
                fig.tight_layout()

                paths.append(self._save(fig, out_dir / f"g{k + 1}_{fname}.png"))

        return paths

    @staticmethod
    def _bin_edges(values: np.ndarray, bins: int, q: float = 0.5) -> np.ndarray:
        """Returns linear bin edges spanning the central percentile range of the values.

        Args:
            values : Flat value array.
            bins   : Number of bins, so the result has `bins` + 1 edges.
            q      : Percentile trimmed from each tail.

        Returns:
            Monotone bin edges, widened to the full range and then to unit width if degenerate.
        """

        lo = float(np.percentile(values, q))
        hi = float(np.percentile(values, 100.0 - q))

        if hi - lo < 1e-12:
            lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            hi = lo + 1.0

        return np.linspace(lo, hi, bins + 1)

    @staticmethod
    def _point_density(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Returns the 2-D histogram count of the bin each scatter point falls in.

        Args:
            gt   : Ground-truth values of shape (n_points,).
            pred : Matched predicted values of the same shape.

        Returns:
            Per-point bin count of shape (n_points,), floored at 1 so it can be log-scaled.
        """

        bins = int(np.clip(np.sqrt(gt.size / 2.0), 16, 128))

        gt_edges   = ParamPlotter._bin_edges(gt,   bins)
        pred_edges = ParamPlotter._bin_edges(pred, bins)

        counts, _, _ = np.histogram2d(gt, pred, bins=[gt_edges, pred_edges])
        gt_bin       = np.clip(np.digitize(gt,   gt_edges)   - 1, 0, bins - 1)
        pred_bin     = np.clip(np.digitize(pred, pred_edges) - 1, 0, bins - 1)

        return np.maximum(counts[gt_bin, pred_bin], 1.0)

    def _scatter_panel(self, ax, gt: np.ndarray, pred: np.ndarray, label: str):
        """Draws a density-coloured scatter of prediction against ground truth, densest points on top.

        Args:
            ax    : Axes to draw into.
            gt    : Ground-truth values of shape (n_points,).
            pred  : Matched predicted values of the same shape.
            label : Legend label, annotated with the R² of the pair.

        Returns:
            The matplotlib PathCollection, so the caller can attach a colorbar.
        """

        r2      = self._r2_value(gt, pred)
        density = self._point_density(gt, pred)
        order   = np.argsort(density)
        norm    = mcolors.LogNorm(vmin=1.0, vmax=max(float(density.max()), 2.0))

        return ax.scatter(
            gt[order], pred[order], c=density[order], s=4, alpha=0.7, norm=norm,
            cmap="viridis", edgecolors="none", rasterized=True, label=f"{label}  R²={r2:.3f}",
        )

    @staticmethod
    def _robust_square_limits(*arrays: np.ndarray, q: float = 0.5) -> tuple:
        """Returns padded shared axis limits covering the central percentile range of all inputs.

        Args:
            *arrays : Value arrays sharing one axis range.
            q       : Percentile trimmed from each tail before padding.

        Returns:
            Tuple of lower and upper limit.
        """

        flat = np.concatenate([arr for arr in arrays])
        lo   = float(np.percentile(flat, q))
        hi   = float(np.percentile(flat, 100.0 - q))
        pad  = (hi - lo) * 0.04 if hi > lo else 1.0

        return lo - pad, hi + pad

    @staticmethod
    def _identity_line(ax, lo: float, hi: float) -> None:
        """Draws the dashed y = x reference line between `lo` and `hi` on the axes."""

        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--", label="identity")

    def plot_param_scatter(
        self,
        params_pred   : np.ndarray,
        params_gt     : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        assignment    : Optional[np.ndarray] = None,
        max_points    : int   = 8_000,
        seed          : int   = 0,
        amp_threshold : float = ParamMatcher.ACTIVE_AMP_THR,
    ) -> List[Path]:
        """Plots matched prediction against ground truth as a density scatter, one figure per Gaussian and parameter.

        Points come only from pixels where the GT Gaussian is active, and are subsampled
        jointly across the pair so the axes stay paired.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            params_gt     : Ground-truth parameter cube of the same shape.
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figures are written into.
            assignment    : Precomputed predicted-to-GT assignment, or None to match here.
            max_points    : Maximum number of scatter points drawn per figure.
            seed          : Seed of the subsampling generator.
            amp_threshold : Amplitude above which a slot counts as active.

        Returns:
            Paths of the written figures.
        """

        aligned = self.alignment.of(params_pred, params_gt, n_gaussians, amp_threshold, assignment)
        paths   = []

        for k in range(n_gaussians):
            gt_amp_flat = params_gt[3 * k].reshape(-1)
            is_active   = np.isfinite(gt_amp_flat) & ParamMatcher.is_active(gt_amp_flat, amp_threshold)

            for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                if ch >= params_gt.shape[0] or ch >= aligned.shape[0]:
                    continue

                gt_all   = params_gt[ch].reshape(-1)
                pred_all = aligned  [ch].reshape(-1)
                matched  = is_active & np.isfinite(pred_all)

                gt, pred = self._paired_subsample([gt_all[matched], pred_all[matched]], max_points, seed)
                if gt.size == 0:
                    continue

                fig, ax = plt.subplots(figsize=(5.2, 4.4))
                sc      = self._scatter_panel(ax, gt, pred, "matched")
                lo, hi  = self._robust_square_limits(gt, pred)
                self._identity_line(ax, lo, hi)

                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal", adjustable="box")

                fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("point density (count per 2-D bin, log scale)")

                r2_str = self._r2_value(gt, pred)
                ax.set_title(f"GT g{k + 1} — {lbl}  (matched, R²={r2_str:.3f})", fontsize=10)
                ax.set_xlabel(f"GT {short}")
                ax.set_ylabel(f"Matched pred {short}")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
                fig.tight_layout()

                paths.append(self._save(fig, out_dir / f"g{k + 1}_{fname}.png"))

        return paths

    @staticmethod
    def _r2_value(gt: np.ndarray, pred: np.ndarray) -> float:
        """Returns the coefficient of determination of `pred` against `gt`."""

        ss_res = float(np.sum((gt - pred) ** 2))
        ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))

        return 1.0 - ss_res / (ss_tot + 1e-12)

    def plot_param_error_maps(
        self,
        params_pred   : np.ndarray,
        params_gt     : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        row_offset     : int,
        col_offset     : int,
        assignment    : Optional[np.ndarray] = None,
        amp_threshold : float = ParamMatcher.ACTIVE_AMP_THR,
    ) -> List[Path]:
        """Maps the absolute matched parameter error over the scene, one figure per Gaussian and parameter.

        Pixels where the GT Gaussian is inactive are left blank, and the colour scale is
        capped at the 99th percentile of the finite errors.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            params_gt     : Ground-truth parameter cube of the same shape.
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figures are written into.
            row_offset     : Row index of the crop origin, used for the axis extent.
            rg_offset     : Range index of the crop origin, used for the axis extent.
            assignment    : Precomputed predicted-to-GT assignment, or None to match here.
            amp_threshold : Amplitude above which a slot counts as active.

        Returns:
            Paths of the written figures.
        """

        aligned = self.alignment.of(params_pred, params_gt, n_gaussians, amp_threshold, assignment)
        H, W    = params_pred.shape[-2:]
        extent  = [col_offset, col_offset + W, row_offset + H, row_offset]
        paths   = []

        for k in range(n_gaussians):
            gt_active = ParamMatcher.is_active(params_gt[3 * k], amp_threshold)

            for j, (fname, _) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                if ch >= params_gt.shape[0] or ch >= aligned.shape[0]:
                    continue

                err       = np.where(gt_active, np.abs(aligned[ch] - params_gt[ch]), np.nan).astype(np.float32)
                valid_err = err[np.isfinite(err)]
                if valid_err.size == 0:
                    continue

                vmax = float(np.percentile(valid_err, 99.0))

                paths.append(self._imshow_panel(
                    data       = err,
                    title      = f"|Δ{short}| — GT g{k + 1}  (matched, p99={vmax:.3g})",
                    x_label    = "col index",
                    y_label    = "row index",
                    cbar_label = f"|Δ{short}|",
                    extent     = extent,
                    cmap       = self.err_cmap,
                    vmin       = 0.0,
                    vmax       = vmax,
                    origin     = "upper",
                    path       = out_dir / f"g{k + 1}_{fname}.png",
                ))

        return paths

    def plot_param_error_hists(
        self,
        params_pred   : np.ndarray,
        params_gt     : np.ndarray,
        n_gaussians   : int,
        out_dir       : Path,
        assignment    : Optional[np.ndarray] = None,
        bins          : int   = 80,
        amp_threshold : float = ParamMatcher.ACTIVE_AMP_THR,
    ) -> List[Path]:
        """Plots the signed matched parameter error histogram, one figure per Gaussian and parameter.

        The axis is centred on zero and clipped symmetrically at the 99.5th percentile of
        the absolute error; median and mean lines expose any systematic bias.

        Args:
            params_pred   : Predicted parameter cube of shape (3 * K, rows, cols).
            params_gt     : Ground-truth parameter cube of the same shape.
            n_gaussians   : Gaussian count K.
            out_dir       : Directory the figures are written into.
            assignment    : Precomputed predicted-to-GT assignment, or None to match here.
            bins          : Number of histogram bins.
            amp_threshold : Amplitude above which a slot counts as active.

        Returns:
            Paths of the written figures.
        """

        aligned = self.alignment.of(params_pred, params_gt, n_gaussians, amp_threshold, assignment)
        paths   = []

        for k in range(n_gaussians):
            gt_amp_flat = params_gt[3 * k].reshape(-1)
            is_active   = np.isfinite(gt_amp_flat) & ParamMatcher.is_active(gt_amp_flat, amp_threshold)

            for j, (fname, lbl) in enumerate(self.PARAM_LABELS):
                ch    = 3 * k + j
                short = self.PARAM_SHORT[j]

                if ch >= params_gt.shape[0] or ch >= aligned.shape[0]:
                    continue

                gt_flat   = params_gt[ch].reshape(-1)
                pred_flat = aligned  [ch].reshape(-1)
                matched   = is_active & np.isfinite(gt_flat) & np.isfinite(pred_flat)

                err = (pred_flat - gt_flat)[matched]
                if err.size == 0:
                    continue

                half      = max(float(np.percentile(np.abs(err), 99.5)), 1e-6)
                bin_edges = np.linspace(-half, half, bins + 1)
                err_in    = err[(err >= bin_edges[0]) & (err <= bin_edges[-1])]
                if err_in.size == 0:
                    continue

                bias   = float(np.mean(err))
                median = float(np.median(err))
                mae    = float(np.mean(np.abs(err)))

                fig, ax = plt.subplots(figsize=(4.8, 3.4))

                ax.hist(err_in, bins=bin_edges, density=True, color="C3", alpha=0.65, edgecolor="none")
                ax.axvline(0.0,    color="black", linestyle="-",  linewidth=0.9, label="zero error")
                ax.axvline(median, color="C0",    linestyle="--", linewidth=0.9, label=f"median={median:.3g}")
                ax.axvline(bias,   color="C3",    linestyle="--", linewidth=0.9, label=f"mean={bias:.3g}")

                ax.set_title(f"GT g{k + 1} — {lbl} error  (matched, active, MAE={mae:.3g})", fontsize=10)
                ax.set_xlabel(f"Δ{short} (matched pred − GT)")
                ax.set_ylabel("density")
                ax.legend(fontsize=7, framealpha=0.9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
                fig.tight_layout()

                paths.append(self._save(fig, out_dir / f"g{k + 1}_{fname}.png"))

        return paths
