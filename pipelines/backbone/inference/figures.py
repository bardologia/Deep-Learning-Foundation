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
    def __init__(
        self,
        plotter : Plotter,
        meta    : InferenceMetadata,
        logger  : Logger,
        cfg     : InferenceConfig,
    ) -> None:

        self.plotter = plotter
        self.meta    = meta
        self.logger  = logger
        self.cfg     = cfg

    def _compose_tracks(self, run, figure_paths: Dict[str, List[Path]]) -> None:
        track_plotter = self.plotter.track
        meta          = self.meta
        logger        = self.logger

        figure_paths["track_geometry"] = [track_plotter.plot_track_geometry(
            baselines = run.track_baselines,
            out_path  = meta.figures_dir / "tracks" / "track_geometry.png",
        )]
        logger.subsection(f"Track geometry : {meta.figures_dir / 'tracks'}")

        figure_paths["track_profiles"] = track_plotter.plot_track_profiles(
            profiles      = run.track_profiles,
            out_dir       = meta.figures_dir / "tracks",
            split_azimuth = (run.split_region.azimuth_start, run.split_region.azimuth_end),
        )
        logger.subsection(f"Track profiles : {len(figure_paths['track_profiles'])} figures in {meta.figures_dir / 'tracks'}")

        figure_paths["track_flight_3d"] = [track_plotter.plot_track_flight_3d(
            profiles = run.track_profiles,
            out_path = meta.figures_dir / "tracks" / "flight_tracks_3d.png",
        )]
        logger.subsection(f"Flight tracks 3D : {meta.figures_dir / 'tracks' / 'flight_tracks_3d.png'}")

    def _compose_input_channels(self, result: Result, run, figure_paths: Dict[str, List[Path]]) -> None:
        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger

        if run.complex_inputs is not None and run.n_secondaries > 0:
            figure_paths["input_channels"] = slice_plotter.plot_input_channels(
                complex_inputs = run.complex_inputs,
                n_secondaries  = run.n_secondaries,
                labels         = run.secondary_labels,
                out_dir        = meta.figures_dir / "input_channels",
                az_offset      = result.azimuth_offset,
                rg_offset      = result.range_offset,
                primary_label  = run.track_baselines.reference,
            )
            logger.subsection(f"Input channels : {len(figure_paths['input_channels'])} figures in {meta.figures_dir / 'input_channels'}")

    def _compose_profiles(self, result: Result, run, x_axis_np: np.ndarray, param_space: bool, figure_paths: Dict[str, List[Path]]) -> None:
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

        reduced_curves_for_plot = result.reduced.reduced_curves if result.reduced is not None else None

        for tag, pixels in (("best", selected["best"]), ("worst", selected["worst"]), ("random", selected["random"])):
            figure_paths[f"profiles_{tag}"] = slice_plotter.plot_profiles(
                pred_curves    = result.pred_curves,
                gt_curves      = result.gt_curves,
                params_pred    = (result.params_pred if param_space else None),
                x_axis         = x_axis_np,
                pixels         = pixels,
                tag            = tag,
                out_dir        = meta.figures_dir / "profiles",
                n_gaussians    = run.n_gaussians,
                pixel_metrics  = pixel_metrics_for_plot,
                az_offset      = result.azimuth_offset,
                rg_offset      = result.range_offset,
                reduced_curves = reduced_curves_for_plot,
                full_curves    = run.full_curves,
            )

            logger.subsection(f"Profiles ({tag:<6}) : {len(figure_paths[f'profiles_{tag}'])} figures in {meta.figures_dir / 'profiles'}")

    def _compose_pixel_maps(self, result: Result, figure_paths: Dict[str, List[Path]]) -> None:
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
            pixel_map_specs.append(("label_r2_map", "label_r2", result.label_r2, "Label fit R² (GT curves vs raw tomogram)", "R²", {"cmap": "RdYlGn", "q_low": 2.0, "q_high": 98.0}))

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
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
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
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
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

    def _compose_param_plots(self, result: Result, run, figure_paths: Dict[str, List[Path]]) -> None:
        param_plotter = self.plotter.param
        slot_plotter  = self.plotter.slot
        meta          = self.meta
        logger        = self.logger
        assignment    = result.assignment(run.n_gaussians) if result.params_gt is not None else None

        figure_paths["param_distributions"] = param_plotter.plot_param_distributions(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = (result.params_gt[: run.n_gaussians * 3] if result.params_gt is not None else None),
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_distributions",
            assignment  = assignment,
        )

        figure_paths["param_scatter"] = param_plotter.plot_param_scatter(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_scatter",
            assignment  = assignment,
        )

        figure_paths["param_error_maps"] = param_plotter.plot_param_error_maps(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_error_maps",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
            assignment  = assignment,
        )

        figure_paths["param_error_hists"] = param_plotter.plot_param_error_hists(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "param_error_hists",
            assignment  = assignment,
        )

        figure_paths["active_count_map"] = slot_plotter.plot_active_count_map(
            params_pred = result.params_pred[: run.n_gaussians * 3],
            params_gt   = result.params_gt  [: run.n_gaussians * 3],
            n_gaussians = run.n_gaussians,
            out_dir     = meta.figures_dir / "slots",
            az_offset   = result.azimuth_offset,
            rg_offset   = result.range_offset,
        )

        logger.subsection(f"Param plots : distributions, scatter, error maps, error histograms, active-count map written to {meta.figures_dir}")

    def _compose_slot_organization(self, result: Result, run, figure_paths: Dict[str, List[Path]]) -> None:
        org_plotter = self.plotter.organization
        meta        = self.meta
        logger      = self.logger
        n_K         = run.n_gaussians
        out_dir     = meta.figures_dir / "slot_organization"

        params_pred = result.params_pred[: n_K * 3]
        params_gt   = result.params_gt  [: n_K * 3] if result.params_gt is not None else None

        figure_paths["slot_usage"]      = org_plotter.plot_slot_usage(params_pred, n_K, out_dir)
        figure_paths["slot_param_dist"] = org_plotter.plot_slot_param_distributions(params_pred, n_K, out_dir)
        figure_paths["slot_mu_rank"]    = org_plotter.plot_mu_rank_matrix(params_pred, n_K, out_dir)

        if params_gt is not None:
            figure_paths["slot_assignment"] = org_plotter.plot_assignment_matrix(params_pred, params_gt, n_K, out_dir, assignment=result.assignment(n_K))

        logger.subsection(f"Slot organization : usage, per-slot distributions, μ-rank and GT-assignment matrices written to {out_dir}")

    def _compose_slices(self, result: Result, run, global_metrics: dict, x_axis_np: np.ndarray, indices: dict, figure_paths: Dict[str, List[Path]]) -> None:
        meta   = self.meta
        logger = self.logger
        cfg    = self.cfg

        self._slice_set(
            pred_cube      = result.pred_curves,
            gt_cube        = result.gt_curves,
            full_cube      = run.full_curves,
            result         = result,
            x_axis_np      = x_axis_np,
            indices        = indices,
            global_metrics = global_metrics,
            ssim_prefix    = "gt",
            out_dir        = meta.figures_dir / "slices",
            stem_prefix    = "",
            group_suffix   = "",
            ref_title      = "GT (Gaussian)",
            pred_title     = "Prediction",
            err_title      = "|Pred − GT|",
            full_title     = "Full tomogram (raw)",
            figure_paths   = figure_paths,
        )

        pred_norm = ProfileNormalizer.unit_area(result.pred_curves)
        gt_norm   = ProfileNormalizer.unit_area(result.gt_curves)
        full_norm = ProfileNormalizer.unit_area(run.full_curves)

        self._slice_set(
            pred_cube      = pred_norm,
            gt_cube        = gt_norm,
            full_cube      = full_norm,
            result         = result,
            x_axis_np      = x_axis_np,
            indices        = indices,
            global_metrics = global_metrics,
            ssim_prefix    = "norm",
            out_dir        = meta.figures_dir / "slices_norm",
            stem_prefix    = "norm_",
            group_suffix   = "_norm",
            ref_title      = "GT (unit-area)",
            pred_title     = "Prediction (unit-area)",
            err_title      = "|Pred − GT|",
            full_title     = "Full tomogram (unit-area)",
            figure_paths   = figure_paths,
        )

        logger.subsection(f"Slices written : range={cfg.n_range_slices} azimuth={cfg.n_azimuth_slices} elev={cfg.n_elevation_slices} (denorm + unit-area, gt/pred/error each)")

    def _slice_set(
        self,
        pred_cube      : np.ndarray,
        gt_cube        : np.ndarray,
        full_cube,
        result         : Result,
        x_axis_np      : np.ndarray,
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

        slice_plotter   = self.plotter.slice
        slice_range_idx = indices["slice_range_idx"]
        slice_az_idx    = indices["slice_az_idx"]
        slice_elev_idx  = indices["slice_elev_idx"]

        figure_paths[f"slices_range{group_suffix}"]   = []
        figure_paths[f"slices_azimuth{group_suffix}"] = []
        figure_paths[f"slices_elev{group_suffix}"]    = []

        for axis, indices_arr, stem_fn, group in (
            ("range",   slice_range_idx, lambda i: f"{stem_prefix}range_{int(i) + result.range_offset}",     f"slices_range{group_suffix}"),
            ("azimuth", slice_az_idx,    lambda i: f"{stem_prefix}azimuth_{int(i) + result.azimuth_offset}", f"slices_azimuth{group_suffix}"),
        ):
            for i in indices_arr:
                figure_paths[group] += slice_plotter.plot_tomogram_slice(
                    pred_cube  = pred_cube,
                    gt_cube    = gt_cube,
                    axis       = axis,
                    index      = int(i),
                    x_axis     = x_axis_np,
                    out_dir    = out_dir,
                    stem       = stem_fn(i),
                    az_offset  = result.azimuth_offset,
                    rg_offset  = result.range_offset,
                    ssim_value = global_metrics[f"ssim_{ssim_prefix}_{axis}_{int(i)}"],
                    ref_title  = ref_title,
                    pred_title = pred_title,
                    err_title  = err_title,
                    full_cube  = full_cube,
                    full_title = full_title,
                )

        for i in slice_elev_idx:
            figure_paths[f"slices_elev{group_suffix}"] += slice_plotter.plot_elevation_intensity_slice(
                pred_cube  = pred_cube,
                gt_cube    = gt_cube,
                elev_idx   = int(i),
                x_axis     = x_axis_np,
                out_dir    = out_dir,
                stem       = f"{stem_prefix}elev_idx_{int(i)}",
                az_offset  = result.azimuth_offset,
                rg_offset  = result.range_offset,
                ssim_value = global_metrics[f"ssim_{ssim_prefix}_elev_{int(i)}"],
                ref_title  = ref_title,
                pred_title = pred_title,
                err_title  = err_title,
                full_cube  = full_cube,
                full_title = full_title,
            )

    def _compose_ssim(self, result: Result, global_metrics: dict, x_axis_np: np.ndarray, indices: dict, figure_paths: Dict[str, List[Path]]) -> None:
        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger

        _N_elev, _az, _rg = result.pred_curves.shape
        offsets           = {"range": result.range_offset, "azimuth": result.azimuth_offset}

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

        logger.subsection(f"SSIM plots : range, azimuth, elev (denorm + unit-area) written to {meta.figures_dir}\n")

        figure_paths["elev_metric_curves"] = slice_plotter.plot_elev_metric_curves(
            global_metrics = global_metrics,
            out_dir        = meta.figures_dir / "elev_metrics",
            n_elev         = _N_elev,
            x_axis         = x_axis_np,
        )

        logger.subsection(f"Elev metric curves (MAE, RMSE, R², CE) written to {meta.figures_dir / 'elev_metrics'}\n")

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

        slice_plotter     = self.plotter.slice
        _N_elev, _az, _rg = shape

        for axis, n_slices, indices_arr, offset in (
            ("range",   _rg,     indices["all_range_idx"], offsets["range"]),
            ("azimuth", _az,     indices["all_az_idx"],    offsets["azimuth"]),
            ("elev",    _N_elev, indices["all_elev_idx"],  0),
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

    def _compose_reduced(
        self,
        result         : Result,
        run,
        global_metrics : dict,
        x_axis_np      : np.ndarray,
        indices        : dict,
        figure_paths   : Dict[str, List[Path]],
    ) -> None:

        slice_plotter = self.plotter.slice
        meta          = self.meta
        logger        = self.logger
        reduced       = result.reduced

        gt_n   = reduced.gt_norm
        red_n  = reduced.reduced_norm
        full_n = ProfileNormalizer.unit_area(run.full_curves)

        _N_elev, _az, _rg = red_n.shape

        reduced_dir = meta.figures_dir / "reduced"

        figure_paths["improvement_map"] = [slice_plotter.plot_pixel_metric_map(
            metric_map = reduced.improvement,
            title      = "NN improvement over classical baseline (per-pixel ΔMSE, unit-area profiles)",
            label      = "MSE(reduced) − MSE(pred)",
            out_path   = reduced_dir / "improvement_map.png",
            az_offset  = result.azimuth_offset,
            rg_offset  = result.range_offset,
            cmap       = "RdBu_r",
            q_low      = 2.0,
            q_high     = 98.0,
        )]

        figure_paths["reduced_pixel_mse_map"] = [slice_plotter.plot_pixel_metric_map(
            metric_map = reduced.err_reduced,
            title      = "Per-pixel reduced-vs-GT MSE (unit-area profiles, log scale)",
            label      = "MSE",
            out_path   = reduced_dir / "reduced_mse_map.png",
            az_offset  = result.azimuth_offset,
            rg_offset  = result.range_offset,
            cmap       = self.cfg.cmap_error,
            log        = True,
        )]

        self._slice_set(
            pred_cube      = red_n,
            gt_cube        = gt_n,
            full_cube      = full_n,
            result         = result,
            x_axis_np      = x_axis_np,
            indices        = indices,
            global_metrics = global_metrics,
            ssim_prefix    = "red",
            out_dir        = reduced_dir / "slices",
            stem_prefix    = "reduced_",
            group_suffix   = "_reduced",
            ref_title      = "GT (unit-area)",
            pred_title     = "Reduced (Capon, unit-area)",
            err_title      = "|Reduced − GT|",
            full_title     = "Full tomogram (unit-area)",
            figure_paths   = figure_paths,
        )

        self._ssim_curves_set(
            global_metrics = global_metrics,
            indices        = indices,
            shape          = red_n.shape,
            offsets        = {"range": result.range_offset, "azimuth": result.azimuth_offset},
            ssim_prefix    = "red",
            group_suffix   = "_reduced",
            series_label   = "reduced × GT (unit-area)",
            out_dir        = reduced_dir / "ssim",
            figure_paths   = figure_paths,
        )

        figure_paths["elev_metric_curves_reduced"] = slice_plotter.plot_elev_metric_curves(
            global_metrics = global_metrics,
            out_dir        = reduced_dir / "elev_metrics",
            n_elev         = _N_elev,
            x_axis         = x_axis_np,
            suffix         = "red",
            series_label   = "reduced × GT (unit-area)",
        )

        logger.subsection(f"Reduced baseline figures written to {reduced_dir}")

    def _compose_data_consistency(self, result: Result, figure_paths: Dict[str, List[Path]]) -> None:
        slice_plotter = self.plotter.slice
        track_plotter = self.plotter.track
        meta          = self.meta
        consistency   = result.data_consistency

        physics_dir = meta.figures_dir / "data_consistency"

        figure_paths["coherence_error_map"] = [slice_plotter.plot_pixel_metric_map(
            metric_map = consistency.coherence_error_map,
            title      = "Per-pixel coherence-resynthesis error (pred vs GT, log scale)",
            label      = "mean |Δγ|² over tracks",
            out_path   = physics_dir / "coherence_error_map.png",
            az_offset  = result.azimuth_offset,
            rg_offset  = result.range_offset,
            cmap       = self.cfg.cmap_error,
            log        = True,
        )]

        figure_paths["covariance_error_map"] = [slice_plotter.plot_pixel_metric_map(
            metric_map = consistency.covariance_error_map,
            title      = "Per-pixel covariance-matching error (pred vs GT, log scale)",
            label      = "relative covariance error",
            out_path   = physics_dir / "covariance_error_map.png",
            az_offset  = result.azimuth_offset,
            rg_offset  = result.range_offset,
            cmap       = self.cfg.cmap_error,
            log        = True,
        )]

        secondaries = consistency.track_labels[1:]
        metrics     = consistency.metrics
        agreement   = []

        for source in ("gt", "pred"):
            aligned = [metrics[f"phase_agreement_{source}_track_{label}"]         for label in secondaries]
            flipped = [metrics[f"phase_agreement_{source}_flipped_track_{label}"] for label in secondaries]

            agreement.append(track_plotter.plot_phase_agreement(
                labels   = secondaries,
                aligned  = aligned,
                flipped  = flipped,
                source   = source,
                out_path = physics_dir / f"phase_agreement_{source}.png",
            ))

        figure_paths["phase_agreement"] = agreement

        if consistency.track_rows:
            figure_paths["physics_spectroscopy"] = [
                track_plotter.plot_residual_spectroscopy(consistency.track_rows, physics_dir / "residual_vs_kz.png"),
                track_plotter.plot_phase_spread_spectrum(consistency.track_rows, physics_dir / "phase_spread_vs_kz.png"),
            ]

        self.logger.subsection(f"Data-consistency figures : {physics_dir}")

    def compose(
        self,
        result         : Result,
        run,
        global_metrics : dict,
        x_axis_np      : np.ndarray,
        indices        : dict,
        param_space    : bool = True,
    ) -> Dict[str, List[Path]]:

        figure_paths: Dict[str, List[Path]] = {}

        self._compose_tracks(run, figure_paths)
        self._compose_input_channels(result, run, figure_paths)
        self._compose_profiles(result, run, x_axis_np, param_space, figure_paths)
        self._compose_pixel_maps(result, figure_paths)

        if param_space:
            self._compose_param_plots(result, run, figure_paths)
            self._compose_slot_organization(result, run, figure_paths)

        self._compose_slices(result, run, global_metrics, x_axis_np, indices, figure_paths)
        self._compose_ssim(result, global_metrics, x_axis_np, indices, figure_paths)

        if result.reduced is not None:
            self._compose_reduced(result, run, global_metrics, x_axis_np, indices, figure_paths)
        else:
            self.logger.subsection("Reduced-baseline figures skipped: no reduced comparison was computed for this run.")

        if result.data_consistency is not None:
            self._compose_data_consistency(result, figure_paths)
        else:
            self.logger.subsection("Data-consistency figures skipped: no interferometric-consistency evaluation was computed for this run.")

        return figure_paths

    def animate(self, result: Result, run, x_axis_np: np.ndarray) -> Dict[str, Path]:
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
                pred_cube    = result.pred_curves,
                gt_cube      = result.gt_curves,
                axis         = axis,
                out_path     = meta.animations_dir / f"walk_{axis}.gif",
                x_axis       = x_axis_np,
                az_offset    = result.azimuth_offset,
                rg_offset    = result.range_offset,
                full_cube    = run.full_curves,
            )

        logger.subsection("")

        return gif_paths
