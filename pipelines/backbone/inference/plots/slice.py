from __future__ import annotations

from pathlib import Path
from typing  import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm        as cm
import matplotlib.pyplot    as plt
import numpy                as np

from pipelines.backbone.inference.plots.base import PlotTools


class SlicePlotter(PlotTools):
    def plot_profiles(
        self,
        pred_curves    : np.ndarray,
        gt_curves      : np.ndarray,
        params_pred    : Optional[np.ndarray],
        x_axis         : np.ndarray,
        pixels         : np.ndarray,
        tag            : str,
        out_dir        : Path,
        n_gaussians    : int,
        pixel_metrics  : Dict[str, np.ndarray],
        az_offset      : int,
        rg_offset      : int,
        reduced_curves : Optional[np.ndarray] = None,
        full_curves    : Optional[np.ndarray] = None,
    ) -> List[Path]:

        base_colors = [cm.tab10(i) for i in range(10)]
        paths       = []

        for p, (y, x) in enumerate(pixels):
            gt    = gt_curves  [:, y, x]
            pred  = pred_curves[:, y, x]
            comps = self._gaussian_components(params_pred[:, y, x], x_axis, n_gaussians) if params_pred is not None else []

            fig, ax = plt.subplots(figsize=(5.2, 3.4))

            if full_curves is not None:
                ax.plot(x_axis, full_curves[:, y, x], color="0.55", linewidth=1.0, label="Full tomo (raw)", zorder=1)

            ax.plot(x_axis, gt,   color="black", linewidth=1.4, label="GT",   zorder=3)
            ax.plot(x_axis, pred, color="C3",    linewidth=1.2, label="Pred", linestyle="--", zorder=4)

            if reduced_curves is not None:
                reduced   = reduced_curves[:, y, x]
                red_peak  = float(np.nanmax(reduced))
                gt_peak   = float(np.nanmax(gt))
                red_scale = (gt_peak / red_peak) if red_peak > 1e-12 else 1.0
                ax.plot(x_axis, reduced * red_scale, color="C2", linewidth=1.1, label="Reduced (GT-peak)", linestyle=":", zorder=2)

            for k, comp in enumerate(comps):
                ax.plot(x_axis, comp, color=base_colors[k % len(base_colors)], linewidth=0.8, alpha=0.75, label=f"$g_{{{k+1}}}$")

            ax.fill_between(x_axis, pred - gt, 0.0, color="C0", alpha=0.10, linewidth=0)

            mse = float(pixel_metrics["mse"][y, x])
            r2  = float(pixel_metrics["r2"] [y, x])
            cos = float(pixel_metrics["cos"][y, x])

            ax.set_title(f"az={y + az_offset}, rg={x + rg_offset}   MSE={mse:.3g}  R²={r2:.3f}  cos={cos:.3f}", fontsize=9)
            ax.set_xlabel("elevation [m]")
            ax.set_ylabel("intensity")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            ax.legend(loc="best", framealpha=0.85, fontsize=7, ncol=2)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{tag}_{p + 1:02d}_az{y + az_offset}_rg{x + rg_offset}.png"))

        return paths

    def plot_pixel_metric_map(
        self,
        metric_map : np.ndarray,
        title      : str,
        label      : str,
        out_path   : Path,
        az_offset  : int,
        rg_offset  : int,
        cmap       : str   = "jet",
        log        : bool  = False,
        q_low      : float = 1.0,
        q_high     : float = 99.0,
    ) -> Path:

        H, W   = metric_map.shape
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]

        data = metric_map.copy()
        if log:
            data = np.log10(np.clip(data, 1e-12, None))

        vmin, vmax = self._shared_clim(data, q_low=q_low, q_high=q_high)

        return self._imshow_panel(
            data       = data,
            title      = title,
            x_label    = "range index",
            y_label    = "azimuth index",
            cbar_label = label + (" [log10]" if log else ""),
            extent     = extent,
            cmap       = cmap,
            vmin       = vmin,
            vmax       = vmax,
            origin     = "upper",
            path       = out_path,
            figsize    = (7, 5),
        )

    def plot_input_channels(
        self,
        complex_inputs : np.ndarray,
        n_secondaries  : int,
        labels         : Optional[List[str]],
        out_dir        : Path,
        az_offset      : int,
        rg_offset      : int,
        primary_label  : str = "primary",
    ) -> List[Path]:

        H, W   = complex_inputs.shape[-2:]
        extent = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        names  = list(labels) if labels else [f"secondary {i + 1}" for i in range(n_secondaries)]
        paths  = []

        paths.append(self._amplitude_panel(complex_inputs[0], f"Primary {primary_label} — amplitude", out_dir / "pass_primary_amplitude.png", extent))

        for i in range(n_secondaries):
            paths.append(self._amplitude_panel(complex_inputs[1 + i], f"Secondary {names[i]} — amplitude", out_dir / f"pass_{names[i]}_amplitude.png", extent))

        for i in range(n_secondaries):
            phase = np.angle(complex_inputs[1 + n_secondaries + i]).astype(np.float32)

            paths.append(self._imshow_panel(
                data       = phase,
                title      = f"Interferogram {primary_label} × {names[i]} — phase",
                x_label    = "range index",
                y_label    = "azimuth index",
                cbar_label = "phase [rad]",
                extent     = extent,
                cmap       = self._phase_cmap(),
                vmin       = -np.pi,
                vmax       = np.pi,
                origin     = "upper",
                path       = out_dir / f"interferogram_{names[i]}_phase.png",
            ))

        return paths

    def _amplitude_panel(self, channel: np.ndarray, title: str, path: Path, extent: list) -> Path:
        amplitude  = np.abs(channel).astype(np.float32)
        vmin, vmax = self._amplitude_clim(amplitude)

        return self._imshow_panel(
            data       = amplitude,
            title      = title,
            x_label    = "range index",
            y_label    = "azimuth index",
            cbar_label = "amplitude",
            extent     = extent,
            cmap       = "gray",
            vmin       = vmin,
            vmax       = vmax,
            origin     = "upper",
            path       = path,
            figsize    = (7, 5),
        )

    def _render_slice_panels(
        self,
        pred_slice : np.ndarray,
        gt_slice   : np.ndarray,
        extent     : list,
        x_label    : str,
        y_label    : str,
        origin     : str,
        title_pos  : str,
        ssim_value : Optional[float],
        out_dir    : Path,
        stem       : str,
        ref_title  : str = "GT (Gaussian)",
        pred_title : str = "Prediction",
        err_title  : str = "|Pred − GT|",
        full_slice : Optional[np.ndarray] = None,
        full_title : str = "Full tomogram (raw)",
    ) -> List[Path]:

        err_gt_slice = np.abs(pred_slice - gt_slice)

        scale        = self._intensity_scale(gt_slice)
        gt_slice     = self._rescale(gt_slice,     scale)
        pred_slice   = self._rescale(pred_slice,   scale)
        err_gt_slice = self._rescale(err_gt_slice, scale)

        vmin, vmax = self._shared_clim(gt_slice, pred_slice)

        err_finite = err_gt_slice[np.isfinite(err_gt_slice)]
        if err_finite.size == 0:
            raise ValueError(f"Slice '{stem}' has no finite error values; the prediction slice is entirely NaN against the reference.")

        emax_gt  = float(np.percentile(err_finite, 99.0))
        ssim_str = f"   SSIM = {ssim_value:.4f}" if ssim_value is not None and np.isfinite(ssim_value) else ""

        panels = [
            (gt_slice,     f"{ref_title} — {title_pos}",          self.cmap,     vmin, vmax,    self._int_label, "gt"),
            (pred_slice,   f"{pred_title} — {title_pos}{ssim_str}", self.cmap,   vmin, vmax,    self._int_label, "pred"),
            (err_gt_slice, f"{err_title} — {title_pos}",           self.err_cmap, 0.0,  emax_gt, self._err_label, "error"),
        ]

        if full_slice is not None:
            full_rescaled = self._rescale(full_slice, scale)
            panels.insert(0, (full_rescaled, f"{full_title} — {title_pos}", self.cmap, vmin, vmax, self._int_label, "full"))

        return [
            self._imshow_panel(
                data       = data,
                title      = title,
                x_label    = x_label,
                y_label    = y_label,
                cbar_label = cbar,
                extent     = extent,
                cmap       = cmap_used,
                vmin       = vlo,
                vmax       = vhi,
                origin     = origin,
                path       = out_dir / f"{stem}_{kind}.png",
            )
            for data, title, cmap_used, vlo, vhi, cbar, kind in panels
        ]

    def plot_tomogram_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        axis       : str,
        index      : int,
        x_axis     : np.ndarray,
        out_dir    : Path,
        stem       : str,
        az_offset  : int,
        rg_offset  : int,
        ssim_value : Optional[float] = None,
        ref_title  : str = "GT (Gaussian)",
        pred_title : str = "Prediction",
        err_title  : str = "|Pred − GT|",
        full_cube  : Optional[np.ndarray] = None,
        full_title : str = "Full tomogram (raw)",
    ) -> List[Path]:

        if axis == "range":
            pred_slice = pred_cube[:, :, index]
            gt_slice   = gt_cube  [:, :, index]
            full_slice = full_cube[:, :, index] if full_cube is not None else None
            x_label    = "azimuth index"
            x_extent_lo, x_extent_hi = az_offset, az_offset + pred_slice.shape[1]
            title_pos                = f"range = {index + rg_offset}"

        elif axis == "azimuth":
            pred_slice = pred_cube[:, index, :]
            gt_slice   = gt_cube  [:, index, :]
            full_slice = full_cube[:, index, :] if full_cube is not None else None
            x_label    = "range index"
            x_extent_lo, x_extent_hi = rg_offset, rg_offset + pred_slice.shape[1]
            title_pos                = f"azimuth = {index + az_offset}"

        else:
            raise ValueError(f"axis must be 'range' or 'azimuth', got {axis!r}")

        sort_idx      = np.argsort(x_axis)
        x_axis_sorted = x_axis[sort_idx]
        pred_slice    = pred_slice[sort_idx]
        gt_slice      = gt_slice  [sort_idx]
        full_slice    = full_slice[sort_idx] if full_slice is not None else None

        extent = [x_extent_lo, x_extent_hi, float(x_axis_sorted[0]), float(x_axis_sorted[-1])]

        return self._render_slice_panels(
            pred_slice = pred_slice,
            gt_slice   = gt_slice,
            extent     = extent,
            x_label    = x_label,
            y_label    = "elevation [m]",
            origin     = "lower",
            title_pos  = title_pos,
            ssim_value = ssim_value,
            out_dir    = out_dir,
            stem       = stem,
            ref_title  = ref_title,
            pred_title = pred_title,
            err_title  = err_title,
            full_slice = full_slice,
            full_title = full_title,
        )

    def plot_elevation_intensity_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        elev_idx   : int,
        x_axis     : np.ndarray,
        out_dir    : Path,
        stem       : str,
        az_offset  : int,
        rg_offset  : int,
        ssim_value : Optional[float] = None,
        ref_title  : str = "GT (Gaussian)",
        pred_title : str = "Prediction",
        err_title  : str = "|Pred − GT|",
        full_cube  : Optional[np.ndarray] = None,
        full_title : str = "Full tomogram (raw)",
    ) -> List[Path]:

        pred_slice = pred_cube[elev_idx]
        gt_slice   = gt_cube  [elev_idx]
        full_slice = full_cube[elev_idx] if full_cube is not None else None

        H, W      = pred_slice.shape
        extent    = [rg_offset, rg_offset + W, az_offset + H, az_offset]
        title_pos = f"elev = {x_axis[elev_idx]:.2f} m (idx {elev_idx})"

        return self._render_slice_panels(
            pred_slice = pred_slice,
            gt_slice   = gt_slice,
            extent     = extent,
            x_label    = "range index",
            y_label    = "azimuth index",
            origin     = "upper",
            title_pos  = title_pos,
            ssim_value = ssim_value,
            out_dir    = out_dir,
            stem       = stem,
            ref_title  = ref_title,
            pred_title = pred_title,
            err_title  = err_title,
            full_slice = full_slice,
            full_title = full_title,
        )

    def plot_metric_histograms(self, metric_arrays: Dict[str, np.ndarray], out_dir: Path) -> List[Path]:
        paths = []

        for name, arr in metric_arrays.items():
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                continue

            lo, hi    = np.percentile(flat, [0.5, 99.5])
            clipped   = np.clip(flat, lo, hi)
            n_outside = int(np.count_nonzero((flat < lo) | (flat > hi)))
            fig, ax   = plt.subplots(figsize=(4.8, 3.4))
            ax.hist(clipped, bins=80, color="C0", edgecolor="white", linewidth=0.3)
            ax.axvline(float(np.median(flat)), color="black", linestyle="--", linewidth=1.0, label=f"median={np.median(flat):.3g}")
            ax.set_title(f"{name} (denorm)  [{n_outside} of {flat.size} outliers piled at edges]", fontsize=8)
            ax.set_xlabel(name)
            ax.set_ylabel("count")
            ax.set_yscale("log")
            ax.legend(framealpha=0.9)
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{name}.png"))

        return paths

    def plot_ssim_curves(
        self,
        global_metrics : dict,
        axis           : str,
        out_path       : Path,
        n_slices       : int,
        slice_indices  : np.ndarray,
        ax_offset      : int = 0,
        prefix         : str = "gt",
        series_label   : str = "pred × GT (Gaussian)",
    ) -> Path:

        vals_gt = np.array([global_metrics[f"ssim_{prefix}_{axis}_{i}"]  for i in range(n_slices)], dtype=np.float64)
        x_phys  = slice_indices.astype(np.float64) + ax_offset

        mean_gt = float(global_metrics[f"ssim_{prefix}_{axis}_mean"])

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(x_phys, vals_gt, color="C0", linewidth=0.9, label=series_label, alpha=0.9)
        ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4f}")

        tick_locs = np.linspace(x_phys[0], x_phys[-1], min(20, n_slices)).round().astype(int)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_locs, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(f"{axis} index")
        ax.set_ylabel("SSIM")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"SSIM per {axis}-slice  (n={n_slices})")
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
        fig.tight_layout()

        return self._save(fig, out_path)

    def plot_elev_metric_curves(
        self,
        global_metrics : dict,
        out_dir        : Path,
        n_elev         : int,
        x_axis         : np.ndarray,
        suffix         : str = "gt",
        series_label   : str = "pred × GT (Gaussian)",
    ) -> List[Path]:

        metric_specs = [
            ("elev_mae",  "mae",           "MAE",           "mean absolute error"),
            ("elev_rmse", "rmse",          "RMSE",          "root mean squared error"),
            ("elev_r2",   "r2",            "R²",            "coefficient of determination"),
            ("elev_ce",   "cross_entropy", "cross-entropy", "cross-entropy (normalised profiles)"),
        ]

        paths = []
        for key, fname, ylabel, desc in metric_specs:
            vals_gt = np.array([global_metrics[f"{key}_{suffix}_{i}"]  for i in range(n_elev)], dtype=np.float64)
            mean_gt = float(global_metrics[f"{key}_{suffix}_mean"])

            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            ax.plot(x_axis, vals_gt, color="C0", linewidth=0.9, label=series_label, alpha=0.9)
            ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4g}")

            ax.set_xlabel("elevation [m]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} per elevation bin — {desc}")
            ax.legend(framealpha=0.9)
            ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{fname}.png"))

        return paths
