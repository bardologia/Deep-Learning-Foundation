"""Draws the cube-space inference figures: profiles, slices, pixel maps and metric curves.

Every slice is written as separate reference, prediction and error panels rather than one
composite, and the reference and prediction panels always share a colour scale so the two
can be read against each other.
"""

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
    """Plots z profiles, cube slices, per-pixel maps, histograms and metric curves."""

    def plot_profiles(
        self,
        pred_curves    : np.ndarray,
        gt_curves      : np.ndarray,
        params_pred    : Optional[np.ndarray],
        z_axis         : np.ndarray,
        pixels         : np.ndarray,
        tag            : str,
        out_dir        : Path,
        n_gaussians    : int,
        pixel_metrics  : Dict[str, np.ndarray],
        row_offset     : int,
        col_offset     : int,
        full_curves    : Optional[np.ndarray] = None,
    ) -> List[Path]:
        """Plots one z profile per selected pixel, overlaying prediction, ground truth and components.

        The classical baseline curve, when given, is rescaled to the GT peak because it
        carries no absolute intensity calibration.

        Args:
            pred_curves    : Predicted cube of shape (n_z, rows, cols).
            gt_curves      : Ground-truth cube of the same shape.
            params_pred    : Predicted parameter cube of shape (3 * K, rows, cols), or None to omit components.
            x_axis         : Elevation axis in metres, of shape (n_z,).
            pixels         : Selected (rows, cols) index pairs of shape (n_pixels, 2).
            tag            : Selection name ("best", "worst" or "random") used in the file names.
            out_dir        : Directory the figures are written into.
            n_gaussians    : Gaussian count K.
            pixel_metrics  : Per-pixel maps keyed "mse", "mae", "r2" and "cos", each of shape (rows, cols).
            az_offset      : Azimuth index of the crop origin, added to the titles and file names.
            rg_offset      : Range index of the crop origin, added to the titles and file names.
            reduced_curves : Classical baseline cube of the same shape, or None.
            full_curves    : Raw cube of the same shape, or None.

        Returns:
            Paths of the written figures, one per selected pixel.
        """

        base_colors = [cm.tab10(i) for i in range(10)]
        paths       = []

        for p, (y, x) in enumerate(pixels):
            gt    = gt_curves  [:, y, x]
            pred  = pred_curves[:, y, x]
            comps = self._gaussian_components(params_pred[:, y, x], z_axis, n_gaussians) if params_pred is not None else []

            fig, ax = plt.subplots(figsize=(5.2, 3.4))

            if full_curves is not None:
                ax.plot(z_axis, full_curves[:, y, x], color="0.55", linewidth=1.0, label="Full cube (raw)", zorder=1)

            ax.plot(z_axis, gt,   color="black", linewidth=1.4, label="GT",   zorder=3)
            ax.plot(z_axis, pred, color="C3",    linewidth=1.2, label="Pred", linestyle="--", zorder=4)

            for k, comp in enumerate(comps):
                ax.plot(z_axis, comp, color=base_colors[k % len(base_colors)], linewidth=0.8, alpha=0.75, label=f"$g_{{{k+1}}}$")

            ax.fill_between(z_axis, pred - gt, 0.0, color="C0", alpha=0.10, linewidth=0)

            mse = float(pixel_metrics["mse"][y, x])
            r2  = float(pixel_metrics["r2"] [y, x])
            cos = float(pixel_metrics["cos"][y, x])

            ax.set_title(f"row={y + row_offset}, col={x + col_offset}   MSE={mse:.3g}  R²={r2:.3f}  cos={cos:.3f}", fontsize=9)
            ax.set_xlabel("z")
            ax.set_ylabel("intensity")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
            ax.legend(loc="best", framealpha=0.85, fontsize=7, ncol=2)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{tag}_{p + 1:02d}_row{y + row_offset}_col{x + col_offset}.png"))

        return paths

    def plot_pixel_metric_map(
        self,
        metric_map : np.ndarray,
        title      : str,
        label      : str,
        out_path   : Path,
        row_offset : int,
        col_offset : int,
        cmap       : str   = "jet",
        log        : bool  = False,
        q_low      : float = 1.0,
        q_high     : float = 99.0,
    ) -> Path:
        """Maps a per-pixel metric over the scene in row-range coordinates.

        Args:
            metric_map : Per-pixel metric of shape (rows, cols).
            title      : Panel title.
            label      : Colorbar label, suffixed with "[log10]" when `log` is set.
            out_path   : Destination file path.
            az_offset  : Azimuth index of the crop origin, used for the axis extent.
            rg_offset  : Range index of the crop origin, used for the axis extent.
            cmap       : Colormap name.
            log        : Whether to display the base-10 logarithm of the clipped values.
            q_low      : Lower percentile setting the colour limit.
            q_high     : Upper percentile setting the colour limit.

        Returns:
            The written file path.
        """

        H, W   = metric_map.shape
        extent = [col_offset, col_offset + W, row_offset + H, row_offset]

        data = metric_map.copy()
        if log:
            data = np.log10(np.clip(data, 1e-12, None))

        vmin, vmax = self._shared_clim(data, q_low=q_low, q_high=q_high)

        return self._imshow_panel(
            data       = data,
            title      = title,
            x_label    = "col index",
            y_label    = "row index",
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
        row_offset     : int,
        col_offset     : int,
        primary_label  : str = "primary",
    ) -> List[Path]:
        """Plots the amplitude of every channel and the phase of every derived channel in the input stack.

        The stack is laid out as the primary, then the secondaries, then the derived channels
        formed against the primary.

        Args:
            complex_inputs : Complex channel stack of shape (channels, rows, cols).
            n_secondaries  : Number of secondary tracks in the stack.
            labels         : Track labels of the secondaries, or None to number them.
            out_dir        : Directory the figures are written into.
            az_offset      : Azimuth index of the crop origin, used for the axis extent.
            rg_offset      : Range index of the crop origin, used for the axis extent.
            primary_label  : Track label of the primary channel.

        Returns:
            Paths of the written figures: one amplitude map per pass and one phase map per derived channel.
        """

        H, W   = complex_inputs.shape[-2:]
        extent = [col_offset, col_offset + W, row_offset + H, row_offset]
        names  = list(labels) if labels else [f"secondary {i + 1}" for i in range(n_secondaries)]
        paths  = []

        paths.append(self._amplitude_panel(complex_inputs[0], f"Primary {primary_label} — amplitude", out_dir / "channel_primary_amplitude.png", extent))

        for i in range(n_secondaries):
            paths.append(self._amplitude_panel(complex_inputs[1 + i], f"Secondary {names[i]} — amplitude", out_dir / f"channel_{names[i]}_amplitude.png", extent))

        for i in range(n_secondaries):
            phase = np.angle(complex_inputs[1 + n_secondaries + i]).astype(np.float32)

            paths.append(self._imshow_panel(
                data       = phase,
                title      = f"Derived {primary_label} × {names[i]} — phase",
                x_label    = "col index",
                y_label    = "row index",
                cbar_label = "phase [rad]",
                extent     = extent,
                cmap       = self._phase_cmap(),
                vmin       = -np.pi,
                vmax       = np.pi,
                origin     = "upper",
                path       = out_dir / f"derived_{names[i]}_phase.png",
            ))

        return paths

    def _amplitude_panel(self, channel: np.ndarray, title: str, path: Path, extent: list) -> Path:
        """Renders the amplitude of one complex channel in greyscale with the standard amplitude clim.

        Args:
            channel : Complex channel of shape (rows, cols).
            title   : Panel title.
            path    : Destination file path.
            extent  : Axis extent as [x_min, x_max, y_min, y_max].

        Returns:
            The written file path.
        """

        amplitude  = np.abs(channel).astype(np.float32)
        vmin, vmax = self._amplitude_clim(amplitude)

        return self._imshow_panel(
            data       = amplitude,
            title      = title,
            x_label    = "col index",
            y_label    = "row index",
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
        full_title : str = "Full cube (raw)",
    ) -> List[Path]:
        """Writes the reference, prediction, error and optional raw panels of one 2-D slice.

        Reference and prediction share a colour scale; the error panel is capped at the
        99th percentile of its finite values.

        Args:
            pred_slice : Predicted 2-D slice.
            gt_slice   : Reference 2-D slice of the same shape.
            extent     : Axis extent as [x_min, x_max, y_min, y_max].
            x_label    : Horizontal axis label.
            y_label    : Vertical axis label.
            origin     : Image origin, "upper" or "lower".
            title_pos  : Slice position string appended to every panel title.
            ssim_value : SSIM of this slice annotated on the prediction panel, or None.
            out_dir    : Directory the panels are written into.
            stem       : File stem the panel kind is appended to.
            ref_title  : Panel title for the reference.
            pred_title : Panel title for the prediction.
            err_title  : Panel title for the absolute error.
            full_slice : Raw cube slice of the same shape, or None.
            full_title : Panel title for the raw cube.

        Returns:
            Paths of the written panels.

        Raises:
            ValueError : If the error slice holds no finite value.
        """

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

    def plot_cube_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        axis       : str,
        index      : int,
        z_axis     : np.ndarray,
        out_dir    : Path,
        stem       : str,
        row_offset : int,
        col_offset : int,
        ssim_value : Optional[float] = None,
        ref_title  : str = "GT (Gaussian)",
        pred_title : str = "Prediction",
        err_title  : str = "|Pred − GT|",
        full_cube  : Optional[np.ndarray] = None,
        full_title : str = "Full cube (raw)",
    ) -> List[Path]:
        """Plots a vertical cube slice at a fixed range or row index.

        The z axis is sorted ascending before display, so the panels read bottom-up
        in metres regardless of the stored bin order.

        Args:
            pred_cube  : Predicted cube of shape (n_z, rows, cols).
            gt_cube    : Reference cube of the same shape.
            axis       : Axis held fixed, "range" or "row".
            index      : Index along `axis` within the crop.
            x_axis     : Elevation axis in metres, of shape (n_z,).
            out_dir    : Directory the panels are written into.
            stem       : File stem the panel kind is appended to.
            az_offset  : Azimuth index of the crop origin.
            rg_offset  : Range index of the crop origin.
            ssim_value : SSIM of this slice annotated on the prediction panel, or None.
            ref_title  : Panel title for the reference.
            pred_title : Panel title for the prediction.
            err_title  : Panel title for the absolute error.
            full_cube  : Raw cube of the same shape, or None.
            full_title : Panel title for the raw cube.

        Returns:
            Paths of the written panels.

        Raises:
            ValueError : If `axis` is neither "range" nor "row".
        """

        if axis == "col":
            pred_slice = pred_cube[:, :, index]
            gt_slice   = gt_cube  [:, :, index]
            full_slice = full_cube[:, :, index] if full_cube is not None else None
            x_label    = "row index"
            x_extent_lo, x_extent_hi = row_offset, row_offset + pred_slice.shape[1]
            title_pos                = f"col = {index + col_offset}"

        elif axis == "row":
            pred_slice = pred_cube[:, index, :]
            gt_slice   = gt_cube  [:, index, :]
            full_slice = full_cube[:, index, :] if full_cube is not None else None
            x_label    = "col index"
            x_extent_lo, x_extent_hi = col_offset, col_offset + pred_slice.shape[1]
            title_pos                = f"row = {index + row_offset}"

        else:
            raise ValueError(f"axis must be 'col' or 'row', got {axis!r}")

        sort_idx      = np.argsort(z_axis)
        x_axis_sorted = z_axis[sort_idx]
        pred_slice    = pred_slice[sort_idx]
        gt_slice      = gt_slice  [sort_idx]
        full_slice    = full_slice[sort_idx] if full_slice is not None else None

        extent = [x_extent_lo, x_extent_hi, float(x_axis_sorted[0]), float(x_axis_sorted[-1])]

        return self._render_slice_panels(
            pred_slice = pred_slice,
            gt_slice   = gt_slice,
            extent     = extent,
            x_label    = x_label,
            y_label    = "z",
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

    def plot_z_intensity_slice(
        self,
        pred_cube  : np.ndarray,
        gt_cube    : np.ndarray,
        z_idx      : int,
        z_axis     : np.ndarray,
        out_dir    : Path,
        stem       : str,
        row_offset : int,
        col_offset : int,
        ssim_value : Optional[float] = None,
        ref_title  : str = "GT (Gaussian)",
        pred_title : str = "Prediction",
        err_title  : str = "|Pred − GT|",
        full_cube  : Optional[np.ndarray] = None,
        full_title : str = "Full cube (raw)",
    ) -> List[Path]:
        """Plots the horizontal intensity map of one z bin over the whole crop.

        Args:
            pred_cube  : Predicted cube of shape (n_z, rows, cols).
            gt_cube    : Reference cube of the same shape.
            elev_idx   : Elevation bin index to display.
            x_axis     : Elevation axis in metres, of shape (n_z,), used for the title.
            out_dir    : Directory the panels are written into.
            stem       : File stem the panel kind is appended to.
            az_offset  : Azimuth index of the crop origin, used for the axis extent.
            rg_offset  : Range index of the crop origin, used for the axis extent.
            ssim_value : SSIM of this slice annotated on the prediction panel, or None.
            ref_title  : Panel title for the reference.
            pred_title : Panel title for the prediction.
            err_title  : Panel title for the absolute error.
            full_cube  : Raw cube of the same shape, or None.
            full_title : Panel title for the raw cube.

        Returns:
            Paths of the written panels.
        """

        pred_slice = pred_cube[z_idx]
        gt_slice   = gt_cube  [z_idx]
        full_slice = full_cube[z_idx] if full_cube is not None else None

        H, W      = pred_slice.shape
        extent    = [col_offset, col_offset + W, row_offset + H, row_offset]
        title_pos = f"z = {z_axis[z_idx]:.2f} m (idx {z_idx})"

        return self._render_slice_panels(
            pred_slice = pred_slice,
            gt_slice   = gt_slice,
            extent     = extent,
            x_label    = "col index",
            y_label    = "row index",
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
        """Plots one log-count histogram per per-pixel metric map.

        Values outside the central 99% are clipped onto the edge bins and their number is
        stated in the title, so tails stay visible without stretching the axis.

        Args:
            metric_arrays : Mapping of metric name to a per-pixel map of shape (rows, cols).
            out_dir       : Directory the figures are written into.

        Returns:
            Paths of the written figures, skipping maps with no finite value.
        """

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
        """Plots SSIM against slice index along one axis, with the axis mean as a reference line.

        Args:
            global_metrics : Metric dictionary holding the per-slice SSIM values.
            axis           : Axis name ("range", "row" or "elev").
            out_path       : Destination file path.
            n_slices       : Number of slices along the axis.
            slice_indices  : Slice indices within the crop, of shape (n_slices,).
            ax_offset      : Crop origin along the axis, added to the plotted indices.
            prefix         : Metric-key prefix selecting the reference variant.
            series_label   : Legend label for the series.

        Returns:
            The written file path.
        """

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

    def plot_z_metric_curves(
        self,
        global_metrics : dict,
        out_dir        : Path,
        n_z            : int,
        z_axis         : np.ndarray,
        suffix         : str = "gt",
        series_label   : str = "pred × GT (Gaussian)",
    ) -> List[Path]:
        """Plots MAE, RMSE, R² and cross-entropy against z, one figure per metric.

        Args:
            global_metrics : Metric dictionary holding the per-z series.
            out_dir        : Directory the figures are written into.
            n_elev         : Number of z bins.
            x_axis         : Elevation axis in metres, of shape (n_z,).
            suffix         : Metric-key suffix selecting the reference variant.
            series_label   : Legend label for the series.

        Returns:
            Paths of the four written figures.
        """

        metric_specs = [
            ("z_mae",  "mae",           "MAE",           "mean absolute error"),
            ("z_rmse", "rmse",          "RMSE",          "root mean squared error"),
            ("z_r2",   "r2",            "R²",            "coefficient of determination"),
            ("z_ce",   "cross_entropy", "cross-entropy", "cross-entropy (normalised profiles)"),
        ]

        paths = []
        for key, fname, ylabel, desc in metric_specs:
            vals_gt = np.array([global_metrics[f"{key}_{suffix}_{i}"]  for i in range(n_z)], dtype=np.float64)
            mean_gt = float(global_metrics[f"{key}_{suffix}_mean"])

            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            ax.plot(z_axis, vals_gt, color="C0", linewidth=0.9, label=series_label, alpha=0.9)
            ax.axhline(mean_gt, color="C0", linestyle="--", linewidth=1.0, label=f"mean = {mean_gt:.4g}")

            ax.set_xlabel("z")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} per z bin — {desc}")
            ax.legend(framealpha=0.9)
            ax.grid(True, axis="y", which="major", linewidth=0.3, alpha=0.5)
            fig.tight_layout()

            paths.append(self._save(fig, out_dir / f"{fname}.png"))

        return paths
