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
    def __init__(self) -> None:
        self.params_pred = None
        self.params_gt   = None
        self.n_gaussians = 0
        self.aligned     = None

    @staticmethod
    def _is_same(cached: Optional[np.ndarray], candidate: np.ndarray) -> bool:
        if cached is None:
            return False

        return cached.shape == candidate.shape and cached.strides == candidate.strides and cached.ctypes.data == candidate.ctypes.data

    def of(self, params_pred: np.ndarray, params_gt: np.ndarray, n_gaussians: int, assignment: Optional[np.ndarray] = None) -> np.ndarray:
        if self.n_gaussians == n_gaussians and self._is_same(self.params_pred, params_pred) and self._is_same(self.params_gt, params_gt):
            return self.aligned

        self.params_pred = params_pred
        self.params_gt   = params_gt
        self.n_gaussians = n_gaussians
        self.aligned     = GaussianMatcher().aligned_prediction(params_pred, params_gt, n_gaussians, assignment=assignment)

        return self.aligned


class ParamPlotter(PlotTools):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alignment = SlotAlignment()

    def plot_param_distributions(
        self,
        params_pred : np.ndarray,
        params_gt   : Optional[np.ndarray],
        n_gaussians : int,
        out_dir     : Path,
        assignment  : Optional[np.ndarray] = None,
        bins        : int = 80,
    ) -> List[Path]:

        matched     = params_gt is not None
        pred_source = self.alignment.of(params_pred, params_gt, n_gaussians, assignment) if matched else params_pred
        paths       = []

        for k in range(n_gaussians):
            amp_ch = 3 * k
            if matched and amp_ch < params_gt.shape[0]:
                gt_amp_flat = params_gt[amp_ch].reshape(-1)
                active_mask = np.isfinite(gt_amp_flat) & ParamMatcher.is_active(gt_amp_flat)
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
        lo = float(np.percentile(values, q))
        hi = float(np.percentile(values, 100.0 - q))

        if hi - lo < 1e-12:
            lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            hi = lo + 1.0

        return np.linspace(lo, hi, bins + 1)

    @staticmethod
    def _point_density(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        bins = int(np.clip(np.sqrt(gt.size / 2.0), 16, 128))

        gt_edges   = ParamPlotter._bin_edges(gt,   bins)
        pred_edges = ParamPlotter._bin_edges(pred, bins)

        counts, _, _ = np.histogram2d(gt, pred, bins=[gt_edges, pred_edges])
        gt_bin       = np.clip(np.digitize(gt,   gt_edges)   - 1, 0, bins - 1)
        pred_bin     = np.clip(np.digitize(pred, pred_edges) - 1, 0, bins - 1)

        return np.maximum(counts[gt_bin, pred_bin], 1.0)

    def _scatter_panel(self, ax, gt: np.ndarray, pred: np.ndarray, label: str):
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
        flat = np.concatenate([arr for arr in arrays])
        lo   = float(np.percentile(flat, q))
        hi   = float(np.percentile(flat, 100.0 - q))
        pad  = (hi - lo) * 0.04 if hi > lo else 1.0

        return lo - pad, hi + pad

    @staticmethod
    def _identity_line(ax, lo: float, hi: float) -> None:
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--", label="identity")

    def plot_param_scatter(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_dir     : Path,
        assignment  : Optional[np.ndarray] = None,
        max_points  : int = 8_000,
        seed        : int = 0,
    ) -> List[Path]:

        aligned = self.alignment.of(params_pred, params_gt, n_gaussians, assignment)
        paths   = []

        for k in range(n_gaussians):
            gt_amp_flat = params_gt[3 * k].reshape(-1)
            is_active   = np.isfinite(gt_amp_flat) & ParamMatcher.is_active(gt_amp_flat)

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
        ss_res = float(np.sum((gt - pred) ** 2))
        ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))

        return 1.0 - ss_res / (ss_tot + 1e-12)

    def plot_param_error_maps(
        self,
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_dir     : Path,
        az_offset   : int,
        rg_offset   : int,
        assignment  : Optional[np.ndarray] = None,
    ) -> List[Path]:

        aligned = self.alignment.of(params_pred, params_gt, n_gaussians, assignment)
        H, W    = params_pred.shape[-2:]
        extent  = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        paths   = []

        for k in range(n_gaussians):
            gt_active = ParamMatcher.is_active(params_gt[3 * k])

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
                    x_label    = "range index",
                    y_label    = "azimuth index",
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
        params_pred : np.ndarray,
        params_gt   : np.ndarray,
        n_gaussians : int,
        out_dir     : Path,
        assignment  : Optional[np.ndarray] = None,
        bins        : int = 80,
    ) -> List[Path]:

        aligned = self.alignment.of(params_pred, params_gt, n_gaussians, assignment)
        paths   = []

        for k in range(n_gaussians):
            gt_amp_flat = params_gt[3 * k].reshape(-1)
            is_active   = np.isfinite(gt_amp_flat) & ParamMatcher.is_active(gt_amp_flat)

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
