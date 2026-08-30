"""Composes the full backbone-inference figure set from a stitched inference result.

The composer walks every optional evaluation carried on the `Result` (parameters,
embedding maps) and writes one figure per
file into the run's figures directory, returning the paths grouped by figure family.
"""

from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import numpy as np

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.animation          import Animator
from pipelines.backbone.inference.failure_modes      import FailureModes
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics            import Metrics, Result
from pipelines.backbone.inference.plots              import Plotter
from tools.data.preprocessing                        import ProfileNormalizer
from tools.monitoring.logger                         import Logger


class FigureComposer:
    """Writes every inference figure and animation for one backbone run.

    Attributes:
        plotter        : Bundle of per-family plotters (slice, param, slot, organization, stratified).
        meta           : Run metadata holding the output directory layout.
        logger         : Logger used to announce each figure family as it is written.
        cfg            : Inference configuration selecting colormaps, slice counts and GIF settings.
        full_norm_cube : Cached unit-area normalisation of the raw cube, built on first use.
    """

    def __init__(
        self,
        plotter : Plotter,
        meta    : InferenceMetadata,
        logger  : Logger,
        cfg     : InferenceConfig,
    ) -> None:
        """Stores the plotters, run metadata, logger and inference configuration.

        Args:
            plotter : Plotter bundle providing the per-family plotting methods.
            meta    : Inference metadata resolving the figure and animation directories.
            logger  : Logger for per-family progress lines.
            cfg     : Inference configuration.
        """

        self.plotter        = plotter
        self.meta           = meta
        self.logger         = logger
        self.cfg            = cfg
        self.full_norm_cube = None

    def _compose_input_channels(self, result: Result, run, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes amplitude and derived-channel-phase maps of the complex input stack.

        Nothing is written when the run carries no complex inputs.

        Args:
            result       : Inference result supplying the row and column offsets of the crop.
            run          : Loaded run carrying the complex input stack of shape (channels, rows, cols).
            figure_paths : Mapping of figure family to written paths, extended in place.
        """

        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger

        if run.complex_inputs is not None and run.n_secondaries > 0:
            figure_paths["input_channels"] = slice_plotter.plot_input_channels(
                complex_inputs = run.complex_inputs,
                n_secondaries  = run.n_secondaries,
                labels         = run.secondary_labels,
                out_dir        = meta.figures_dir / "input_channels",
                row_offset     = result.row_offset,
                col_offset     = result.col_offset,
            )
            logger.subsection(f"Input channels : {len(figure_paths['input_channels'])} figures in {meta.figures_dir / 'input_channels'}")

    def _compose_profiles(self, result: Result, run, z_axis_np: np.ndarray, param_space: bool, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes predicted-vs-GT z profiles for best, worst and random pixels.

        Args:
            result       : Inference result carrying the predicted and GT curve cubes and per-pixel metrics.
            run          : Loaded run supplying the Gaussian count and the raw cube curves.
            z_axis_np    : Z axis in metres, of shape (n_z,).
            param_space  : Whether predicted Gaussian parameters exist and should be drawn as components.
            figure_paths : Mapping of figure family to written paths, extended in place.
        """

        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger
        cfg           = self.cfg

        pixel_metrics_for_plot = {
            "mse" : result.pixel_mse,
            "mae" : result.pixel_mae,
            "r2"  : result.pixel_r2,
            "cos" : result.pixel_cosine,
        }

        selected = Metrics.select_pixels(
            result.pixel_mse,
            n_best   = cfg.n_best_profiles,
            n_worst  = cfg.n_worst_profiles,
            n_random = cfg.n_random_profiles,
            seed     = cfg.profile_seed,
        )

        for tag, pixels in (("best", selected["best"]), ("worst", selected["worst"]), ("random", selected["random"])):
            figure_paths[f"profiles_{tag}"] = slice_plotter.plot_profiles(
                pred_curves   = result.pred_curves,
                gt_curves     = result.gt_curves,
                params_pred   = (result.params_pred if param_space else None),
                z_axis        = z_axis_np,
                pixels        = pixels,
                tag           = tag,
                out_dir       = meta.figures_dir / "profiles",
                n_gaussians   = run.n_gaussians,
                pixel_metrics = pixel_metrics_for_plot,
                row_offset    = result.row_offset,
                col_offset    = result.col_offset,
                full_curves   = run.full_curves,
            )

            logger.subsection(f"Profiles ({tag:<6}) : {len(figure_paths[f'profiles_{tag}'])} figures in {meta.figures_dir / 'profiles'}")

    def _compose_pixel_maps(self, result: Result, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes the per-pixel metric maps, stratified error curves, failure-mode map and metric histograms.

        Maps for label fit R², extraction fit R² and flip consistency are added only when
        the corresponding evaluation ran.

        Args:
            result       : Inference result carrying the per-pixel metric maps of shape (rows, cols).
            figure_paths : Mapping of figure family to written paths, extended in place.
        """

        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger
        cfg           = self.cfg

        pixel_map_specs = [
            ("pixel_mse_map",  "mse",        result.pixel_mse,                             "Per-pixel curve MSE (denorm)",          "MSE",          {"cmap": cfg.cmap_error, "log": True}),
            ("pixel_r2_map",   "r2",         result.pixel_r2,                              "Per-pixel R² (denorm)",                 "R²",           {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}),
            ("pixel_peak_map", "peak_error", result.pixel_peak_err_idx.astype(np.float32), "Peak-location absolute error (denorm)", "|Δ peak idx|", {"cmap": cfg.cmap_error}),
        ]

        if result.label_r2 is not None:
            pixel_map_specs.append(("label_r2_map", "label_r2", result.label_r2, "Label fit R² (GT curves vs raw cube)", "R²", {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}))

        if result.extract_r2 is not None:
            pixel_map_specs.append(("extract_r2_map", "extract_r2", result.extract_r2, "Extraction fit R² (refit Gaussians vs predicted curve)", "R²", {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}))

        if result.flip_consistency is not None:
            pixel_map_specs.append(("flip_consistency_map", "flip_consistency", result.flip_consistency, "Flip-equivariance disagreement (curve MSE)", "MSE", {"cmap": cfg.cmap_error, "log": True}))

        if result.stratified is not None:
            figure_paths["stratified_errors"] = [self.plotter.stratified.plot_error_curve(
                rows      = payload["rows"],
                covariate = name,
                out_path  = meta.figures_dir / "stratified" / f"{name}.png",
                discrete  = payload["discrete"],
            ) for name, payload in sorted(result.stratified.items())]

        if result.failure_mode_map is not None:
            figure_paths["failure_mode_map"] = [self.plotter.stratified.plot_failure_mode_map(
                result.failure_mode_map,
                mode_names = FailureModes.MODES,
                out_path   = meta.figures_dir / "pixel_maps" / "failure_mode.png",
                row_offset = result.row_offset,
                col_offset = result.col_offset,
            )]

        if result.miss_by_sep:
            figure_paths["miss_by_separation"] = [self.plotter.stratified.plot_miss_by_separation(
                result.miss_by_sep,
                out_path = meta.figures_dir / "pixel_maps" / "miss_by_separation.png",
            )]

        if result.presence_rows:
            figure_paths["presence_reliability"] = [self.plotter.stratified.plot_presence_reliability(
                result.presence_rows,
                out_path = meta.figures_dir / "pixel_maps" / "presence_reliability.png",
            )]

        for key, fname, data, title, label, extra in pixel_map_specs:
            figure_paths[key] = [slice_plotter.plot_pixel_metric_map(
                metric_map = data,
                title      = title,
                label      = label,
                out_path   = meta.figures_dir / "pixel_maps" / f"{fname}.png",
                row_offset = result.row_offset,
                col_offset = result.col_offset,
                **extra,
            )]

        logger.subsection(f"Pixel maps : mse, r2, peak written to {meta.figures_dir / 'pixel_maps'}")

        histogram_arrays = {
            "pixel_mse"    : result.pixel_mse,
            "pixel_r2"     : result.pixel_r2,
            "pixel_cosine" : result.pixel_cosine,
        }

        figure_paths["metric_histograms"] = slice_plotter.plot_metric_histograms(
            histogram_arrays,
            meta.figures_dir / "histograms",
        )

    def _compose_embedding_maps(self, result: Result, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes the JEPA embedding-error, cosine, decoder-floor and full-chain maps plus their histograms.

        Returns immediately when the run carried no embedding evaluation.

        Args:
            result       : Inference result whose `embedding_maps` holds the per-pixel maps of shape (rows, cols).
            figure_paths : Mapping of figure family to written paths, extended in place.
        """

        if result.embedding_maps is None:
            return

        maps          = result.embedding_maps
        slice_plotter = self.plotter.slice
        cfg           = self.cfg
        out_dir       = self.meta.figures_dir / "embedding"

        specs = (
            ("embedding_error_map",  "embedding_error",  maps.embedding_error,  "Per-pixel embedding MSE (predicted vs target embedding)", "MSE",    {"cmap": cfg.cmap_error, "log": True}),
            ("embedding_cosine_map", "embedding_cosine", maps.embedding_cosine, "Per-pixel embedding cosine similarity",                   "cosine", {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}),
            ("embedding_decode_map", "decode_floor",     maps.decode_mse,       "Decoder-only MSE (decode(encode(GT)) vs GT, normalised)", "MSE",    {"cmap": cfg.cmap_error, "log": True}),
            ("embedding_chain_map",  "chain_error",      maps.chain_mse,        "Full-chain MSE (decode(prediction) vs GT, normalised)",   "MSE",    {"cmap": cfg.cmap_error, "log": True}),
        )

        for key, fname, data, title, label, extra in specs:
            figure_paths[key] = [slice_plotter.plot_pixel_metric_map(
                metric_map = data,
                title      = title,
                label      = label,
                out_path   = out_dir / f"{fname}.png",
                row_offset = result.row_offset,
                col_offset = result.col_offset,
                **extra,
            )]

        figure_paths["embedding_histograms"] = slice_plotter.plot_metric_histograms(
            {"embedding_error": maps.embedding_error, "embedding_cosine": maps.embedding_cosine},
            out_dir / "histograms",
        )

        self.logger.subsection(f"Embedding maps : error, cosine, decoder floor, full chain written to {out_dir}")

    def _compose_param_plots(self, result: Result, run, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes the Gaussian-parameter distributions, scatters, error maps, error histograms and active-count map.

        Args:
            result       : Inference result carrying the predicted and GT parameter cubes of shape (3 * K, rows, cols).
            run          : Loaded run supplying the Gaussian count K.
            figure_paths : Mapping of figure family to written paths, extended in place.
        """

        param_plotter = self.plotter.param
        slot_plotter  = self.plotter.slot
        meta          = self.meta
        logger        = self.logger
        assignment    = result.assignment(run.n_gaussians, self.cfg.active_amp_thr) if result.params_gt is not None else None

        figure_paths["param_distributions"] = param_plotter.plot_param_distributions(
            params_pred   = result.params_pred[: run.n_gaussians * 3],
            params_gt     = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians   = run.n_gaussians,
            out_dir       = meta.figures_dir / "param_distributions",
            assignment    = assignment,
            amp_threshold = self.cfg.active_amp_thr,
        )

        figure_paths["param_scatter"] = param_plotter.plot_param_scatter(
            params_pred   = result.params_pred[: run.n_gaussians * 3],
            params_gt     = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians   = run.n_gaussians,
            out_dir       = meta.figures_dir / "param_scatter",
            assignment    = assignment,
            amp_threshold = self.cfg.active_amp_thr,
        )

        figure_paths["param_error_maps"] = param_plotter.plot_param_error_maps(
            params_pred   = result.params_pred[: run.n_gaussians * 3],
            params_gt     = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians   = run.n_gaussians,
            out_dir       = meta.figures_dir / "param_error_maps",
            row_offset     = result.row_offset,
            col_offset     = result.col_offset,
            assignment    = assignment,
            amp_threshold = self.cfg.active_amp_thr,
        )

        figure_paths["param_error_hists"] = param_plotter.plot_param_error_hists(
            params_pred   = result.params_pred[: run.n_gaussians * 3],
            params_gt     = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians   = run.n_gaussians,
            out_dir       = meta.figures_dir / "param_error_hists",
            assignment    = assignment,
            amp_threshold = self.cfg.active_amp_thr,
        )

        figure_paths["active_count_map"] = slot_plotter.plot_active_count_map(
            params_pred   = result.params_pred[: run.n_gaussians * 3],
            params_gt     = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians   = run.n_gaussians,
            out_dir       = meta.figures_dir / "slots",
            row_offset     = result.row_offset,
            col_offset     = result.col_offset,
            amp_threshold = self.cfg.active_amp_thr,
        )

        logger.subsection(f"Param plots : distributions, scatter, error maps, error histograms, active-count map written to {meta.figures_dir}")

    def _compose_slot_organization(self, result: Result, run, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes the raw-slot usage bars, per-slot parameter distributions, μ-rank matrix and GT-assignment matrix.

        Args:
            result       : Inference result carrying the predicted and GT parameter cubes of shape (3 * K, rows, cols).
            run          : Loaded run supplying the Gaussian count K.
            figure_paths : Mapping of figure family to written paths, extended in place.
        """

        org_plotter = self.plotter.organization
        meta        = self.meta
        logger      = self.logger
        n_K         = run.n_gaussians
        out_dir     = meta.figures_dir / "slot_organization"

        params_pred = result.params_pred[: n_K * 3]
        params_gt   = result.params_gt  [: n_K * 3] if result.params_gt is not None else None

        figure_paths["slot_usage"]      = org_plotter.plot_slot_usage(params_pred, n_K, out_dir, amp_threshold=self.cfg.active_amp_thr)
        figure_paths["slot_param_dist"] = org_plotter.plot_slot_param_distributions(params_pred, n_K, out_dir, amp_threshold=self.cfg.active_amp_thr)
        figure_paths["slot_mu_rank"]    = org_plotter.plot_mu_rank_matrix(params_pred, n_K, out_dir, amp_threshold=self.cfg.active_amp_thr)

        if params_gt is not None:
            figure_paths["slot_assignment"] = org_plotter.plot_assignment_matrix(params_pred, params_gt, n_K, out_dir, assignment=result.assignment(n_K, self.cfg.active_amp_thr), amp_threshold=self.cfg.active_amp_thr)

        logger.subsection(f"Slot organization : usage, per-slot distributions, μ-rank and GT-assignment matrices written to {out_dir}")

    def _compose_slices(self, result: Result, run, global_metrics: dict, z_axis_np: np.ndarray, indices: dict, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes col, row and z slices twice: in denormalised intensity and in unit-area form.

        Args:
            result         : Inference result carrying the predicted and GT curve cubes of shape (n_z, row, col).
            run            : Loaded run supplying the raw cube.
            global_metrics : Metric dictionary holding the per-slice SSIM values keyed by axis and index.
            z_axis_np      : Z axis in metres, of shape (n_z,).
            indices        : Slice-index arrays produced by the pipeline, keyed by axis.
            figure_paths   : Mapping of figure family to written paths, extended in place.
        """

        meta   = self.meta
        logger = self.logger
        cfg    = self.cfg

        self._slice_set(
            pred_cube      = result.pred_curves,
            gt_cube        = result.gt_curves,
            full_cube      = run.full_curves,
            result         = result,
            z_axis_np      = z_axis_np,
            indices        = indices,
            global_metrics = global_metrics,
            ssim_prefix    = "gt",
            out_dir        = meta.figures_dir / "slices",
            stem_prefix    = "",
            group_suffix   = "",
            ref_title      = "GT (Gaussian)",
            pred_title     = "Prediction",
            err_title      = "|Pred − GT|",
            full_title     = "Full cube (raw)",
            figure_paths   = figure_paths,
        )

        pred_norm = result.pred_unit_area()
        gt_norm   = result.gt_unit_area()
        full_norm = self._full_unit_area(run)

        self._slice_set(
            pred_cube      = pred_norm,
            gt_cube        = gt_norm,
            full_cube      = full_norm,
            result         = result,
            z_axis_np      = z_axis_np,
            indices        = indices,
            global_metrics = global_metrics,
            ssim_prefix    = "norm",
            out_dir        = meta.figures_dir / "slices_norm",
            stem_prefix    = "norm_",
            group_suffix   = "_norm",
            ref_title      = "GT (unit-area)",
            pred_title     = "Prediction (unit-area)",
            err_title      = "|Pred − GT|",
            full_title     = "Full cube (unit-area)",
            figure_paths   = figure_paths,
        )

        logger.subsection(f"Slices written : range={cfg.n_col_slices} row={cfg.n_row_slices} z={cfg.n_z_slices} (denorm + unit-area, gt/pred/error each)")

    def _slice_set(
        self,
        pred_cube      : np.ndarray,
        gt_cube        : np.ndarray,
        full_cube,
        result         : Result,
        z_axis_np      : np.ndarray,
        indices        : dict,
        global_metrics : dict,
        ssim_prefix    : str,
        out_dir        : Path,
        stem_prefix    : str,
        group_suffix   : str,
        ref_title      : str,
        pred_title     : str,
        err_title      : str,
        full_title     : str,
        figure_paths   : Dict[str, List[Path]],
    ) -> None:
        """Writes one slice figure per selected range, row and z index for a single cube variant.

        Args:
            pred_cube      : Predicted cube of shape (n_z, rows, cols).
            gt_cube        : Ground-truth cube of the same shape as `pred_cube`.
            full_cube      : Raw cube of the same shape, drawn as an extra reference panel.
            result         : Inference result supplying the row and column offsets of the crop.
            x_axis_np      : Elevation axis in metres, of shape (n_z,).
            indices        : Slice-index arrays keyed by axis.
            global_metrics : Metric dictionary holding the per-slice SSIM values.
            ssim_prefix    : Metric-key prefix selecting which SSIM series to annotate ("gt", "norm" or "red").
            out_dir        : Directory the slice figures are written into.
            stem_prefix    : Prefix prepended to each figure file stem.
            group_suffix   : Suffix appended to the figure-family keys in `figure_paths`.
            ref_title      : Panel title for the reference cube.
            pred_title     : Panel title for the predicted cube.
            err_title      : Panel title for the absolute-error panel.
            full_title     : Panel title for the raw cube panel.
            figure_paths   : Mapping of figure family to written paths, extended in place.
        """

        slice_plotter = self.plotter.slice
        slice_col_idx = indices["slice_col_idx"]
        slice_row_idx = indices["slice_row_idx"]
        slice_z_idx   = indices["slice_z_idx"]

        figure_paths[f"slices_col{group_suffix}"]   = []
        figure_paths[f"slices_row{group_suffix}"] = []
        figure_paths[f"slices_z{group_suffix}"]    = []

        for axis, indices_arr, stem_fn, group in (
            ("col", slice_col_idx, lambda i: f"{stem_prefix}col_{int(i) + result.col_offset}",     f"slices_col{group_suffix}"),
            ("row", slice_row_idx,    lambda i: f"{stem_prefix}row_{int(i) + result.row_offset}", f"slices_row{group_suffix}"),
        ):
            for i in indices_arr:
                figure_paths[group] += slice_plotter.plot_cube_slice(
                    pred_cube  = pred_cube,
                    gt_cube    = gt_cube,
                    axis       = axis,
                    index      = int(i),
                    z_axis     = z_axis_np,
                    out_dir    = out_dir,
                    stem       = stem_fn(i),
                    row_offset = result.row_offset,
                    col_offset = result.col_offset,
                    ssim_value = global_metrics[f"ssim_{ssim_prefix}_{axis}_{int(i)}"],
                    ref_title  = ref_title,
                    pred_title = pred_title,
                    err_title  = err_title,
                    full_cube  = full_cube,
                    full_title = full_title,
                )

        for i in slice_z_idx:
            figure_paths[f"slices_z{group_suffix}"] += slice_plotter.plot_z_intensity_slice(
                pred_cube  = pred_cube,
                gt_cube    = gt_cube,
                z_idx      = int(i),
                z_axis     = z_axis_np,
                out_dir    = out_dir,
                stem       = f"{stem_prefix}z_idx_{int(i)}",
                row_offset = result.row_offset,
                col_offset = result.col_offset,
                ssim_value = global_metrics[f"ssim_{ssim_prefix}_z_{int(i)}"],
                ref_title  = ref_title,
                pred_title = pred_title,
                err_title  = err_title,
                full_cube  = full_cube,
                full_title = full_title,
            )

    def _compose_ssim(self, result: Result, global_metrics: dict, z_axis_np: np.ndarray, indices: dict, figure_paths: Dict[str, List[Path]]) -> None:
        """Writes the per-slice SSIM curves for both cube variants and the per-z metric curves.

        Args:
            result         : Inference result supplying the cube shape and crop offsets.
            global_metrics : Metric dictionary holding the per-slice SSIM and per-z series.
            z_axis_np      : Z axis in metres, of shape (n_z,).
            indices        : Slice-index arrays keyed by axis.
            figure_paths   : Mapping of figure family to written paths, extended in place.
        """

        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger

        _N_z, _row, _col = result.pred_curves.shape
        offsets           = {"col": result.col_offset, "row": result.row_offset}

        self._ssim_curves_set(
            global_metrics = global_metrics,
            indices        = indices,
            shape          = result.pred_curves.shape,
            offsets        = offsets,
            ssim_prefix    = "gt",
            group_suffix   = "",
            series_label   = "pred × GT (Gaussian)",
            out_dir        = meta.figures_dir / "ssim",
            figure_paths   = figure_paths,
        )

        self._ssim_curves_set(
            global_metrics = global_metrics,
            indices        = indices,
            shape          = result.pred_curves.shape,
            offsets        = offsets,
            ssim_prefix    = "norm",
            group_suffix   = "_norm",
            series_label   = "pred × GT (unit-area)",
            out_dir        = meta.figures_dir / "ssim_norm",
            figure_paths   = figure_paths,
        )

        logger.subsection(f"SSIM plots : range, row, z (denorm + unit-area) written to {meta.figures_dir}\n")

        figure_paths["z_metric_curves"] = slice_plotter.plot_z_metric_curves(
            global_metrics = global_metrics,
            out_dir        = meta.figures_dir / "z_metrics",
            n_z            = _N_z,
            z_axis         = z_axis_np,
        )

        logger.subsection(f"Z metric curves (MAE, RMSE, R², CE) written to {meta.figures_dir / 'z_metrics'}\n")

    def _ssim_curves_set(
        self,
        global_metrics : dict,
        indices        : dict,
        shape          : tuple,
        offsets        : dict,
        ssim_prefix    : str,
        group_suffix   : str,
        series_label   : str,
        out_dir        : Path,
        figure_paths   : Dict[str, List[Path]],
    ) -> None:
        """Writes one SSIM-versus-slice-index curve per axis for a single cube variant.

        Args:
            global_metrics : Metric dictionary holding the per-slice SSIM values.
            indices        : Full index arrays for range, row and z.
            shape          : Cube shape (n_z, rows, cols) the curves span.
            offsets        : Crop offsets keyed "range" and "row", added to the plotted indices.
            ssim_prefix    : Metric-key prefix selecting which SSIM series to plot.
            group_suffix   : Suffix appended to the figure-family keys in `figure_paths`.
            series_label   : Legend label for the plotted series.
            out_dir        : Directory the curves are written into.
            figure_paths   : Mapping of figure family to written paths, extended in place.
        """

        slice_plotter     = self.plotter.slice
        _N_z, _row, _col = shape

        for axis, n_slices, indices_arr, offset in (
            ("col", _col,     indices["all_col_idx"], offsets["col"]),
            ("row", _row,     indices["all_row_idx"],    offsets["row"]),
            ("z",   _N_z, indices["all_z_idx"],  0),
        ):
            figure_paths[f"ssim_{axis}{group_suffix}"] = [slice_plotter.plot_ssim_curves(
                global_metrics = global_metrics,
                axis           = axis,
                out_path       = out_dir / f"{axis}.png",
                n_slices       = n_slices,
                slice_indices  = indices_arr,
                ax_offset      = offset,
                prefix         = ssim_prefix,
                series_label   = series_label,
            )]

    def _full_unit_area(self, run) -> np.ndarray:
        """Returns the raw cube normalised to unit area per profile, cached after the first call."""

        if self.full_norm_cube is None:
            self.full_norm_cube = ProfileNormalizer.unit_area(run.full_curves)

        return self.full_norm_cube

    def compose(
        self,
        result         : Result,
        run,
        global_metrics : dict,
        z_axis_np      : np.ndarray,
        indices        : dict,
        param_space    : bool = True,
    ) -> Dict[str, List[Path]]:
        """Composes every figure family available for this run and returns their paths.

        Args:
            result         : Inference result carrying the cubes, per-pixel maps and optional evaluations.
            run            : Loaded run supplying the raw cube and the Gaussian count.
            global_metrics : Metric dictionary written by the pipeline before plotting.
            x_axis_np      : Elevation axis in metres, of shape (n_z,).
            indices        : Slice-index arrays keyed by axis.
            param_space    : Whether Gaussian-parameter figures should be produced.

        Returns:
            Mapping of figure-family name to the list of figure paths written for it.
        """

        figure_paths: Dict[str, List[Path]] = {}

        self._compose_input_channels(result, run, figure_paths)
        self._compose_profiles(result, run, z_axis_np, param_space, figure_paths)
        self._compose_pixel_maps(result, figure_paths)
        self._compose_embedding_maps(result, figure_paths)

        if param_space:
            self._compose_param_plots(result, run, figure_paths)

            if result.params_source == "model":
                self._compose_slot_organization(result, run, figure_paths)
            else:
                self.logger.subsection("Slot-organization figures skipped: the parameters were extracted from the predicted curves and sorted by position, so slot index carries no model-side meaning.")

        self._compose_slices(result, run, global_metrics, z_axis_np, indices, figure_paths)
        self._compose_ssim(result, global_metrics, z_axis_np, indices, figure_paths)

        return figure_paths

    def animate(self, result: Result, run, z_axis_np: np.ndarray) -> Dict[str, Path]:
        """Renders one walk-through GIF per configured axis over the predicted and GT cubes.

        Args:
            result    : Inference result carrying the predicted and GT curve cubes of shape (n_z, row, col).
            run       : Loaded run supplying the raw cube.
            z_axis_np : Z axis in metres, of shape (n_z,).

        Returns:
            Mapping of "walk_<axis>" to the written GIF path.
        """

        cfg    = self.cfg
        meta   = self.meta
        logger = self.logger

        animator = Animator(
            logger      = logger,
            cmap        = cfg.cmap_intensity,
            err_cmap    = cfg.cmap_error,
            dpi         = cfg.gif_dpi,
            fps         = cfg.gif_fps,
            max_frames  = cfg.gif_max_frames,
            num_workers = cfg.gif_workers,
        )

        gif_paths: Dict[str, Path] = {}
        for axis in cfg.gif_axes:
            logger.subsection(f"Generating walk GIF along {axis} axis")
            gif_paths[f"walk_{axis}"] = animator.walk_gif(
                pred_cube  = result.pred_curves,
                gt_cube    = result.gt_curves,
                axis       = axis,
                out_path   = meta.animations_dir / f"walk_{axis}.gif",
                z_axis     = z_axis_np,
                row_offset = result.row_offset,
                col_offset = result.col_offset,
                full_cube  = run.full_curves,
            )

        logger.subsection("")

        return gif_paths
