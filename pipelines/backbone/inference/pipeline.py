
from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict, List, Type

import numpy as np

from configuration.inference                           import InferenceConfig
from pipelines.backbone.inference.curve_params         import CurveParamExtractor
from pipelines.backbone.inference.figures              import FigureComposer
from pipelines.backbone.inference.flip_consistency     import FlipConsistencyEvaluator
from pipelines.backbone.inference.normalization_audit  import NormalizationAudit
from pipelines.backbone.inference.loader               import RunLoader
from pipelines.backbone.inference.run_metadata_paths   import InferenceMetadata
from pipelines.backbone.inference.metrics              import Metrics, Result
from pipelines.backbone.inference.plots                import Plotter
from pipelines.backbone.inference.predictor            import Predictor
from pipelines.backbone.inference.failure_modes        import FailureModes
from pipelines.backbone.inference.presence_calibration import PresenceCalibration
from pipelines.backbone.inference.report               import Report, ReportPayloadBuilder
from pipelines.backbone.inference.stratified           import StratifiedErrors
from pipelines.shared.inference.metadata               import CompletedInference, RegionPass, TargetDataset
from tools.data.regions                                import CropRegion
from tools.monitoring.logger                           import Logger
from tools.reporting.plotting                          import PlotBase
from tools.runtime.completion                          import CompletionMarker
from tools.runtime.config_cli                          import ConfigCli
from tools.runtime.git_state                           import GitState


@dataclass(frozen=True)
class InferenceComponents:

    loader_cls              : Type[RunLoader] = RunLoader
    predictor_cls           : Type[Predictor] = Predictor
    param_space             : bool            = True
    embedding_evaluator_cls : Type | None     = None


class InferencePipeline:

    def __init__(self, config: InferenceConfig, components: InferenceComponents | None = None, region_pass: RegionPass | None = None) -> None:

        self.config      = config
        self.components  = components if components is not None else InferenceComponents()
        self.region_pass = region_pass
        self.target      = TargetDataset.of(config)

    def _setup(self, cfg: InferenceConfig) -> tuple[InferenceMetadata, Logger, Plotter]:

        meta = InferenceMetadata(cfg, self.target)
        meta.create_dirs()
        ConfigCli.save_resolved(cfg, meta.config_path)
        np.random.seed(cfg.seed)

        logger = Logger(log_dir=str(meta.logs_dir), name="inference", level=cfg.log_level)
        logger.section("[Inference Pipeline]")
        logger.kv_table(self._setup_rows(cfg, meta))

        PlotBase.use_style(cfg.figure_style)

        plotter = Plotter(
            cmap      = cfg.cmap_intensity,
            err_cmap  = cfg.cmap_error,
            normalize = cfg.normalize_intensity,
            fig_dpi   = cfg.fig_dpi,
            save_dpi  = cfg.save_dpi,
        )

        return meta, logger, plotter

    def _setup_rows(self, cfg: InferenceConfig, meta: InferenceMetadata) -> dict:

        evaluator = self.components.embedding_evaluator_cls

        return {
            "Run Directory"       : cfg.run_directory,
            "Output Dir"          : meta.output_dir,
            "Split"               : cfg.split,
            "Region override"     : cfg.region_override if cfg.region_override else "none (stored split region)",
            "Region pass"         : f"region {self.region_pass.index} of {self.region_pass.count} (group {self.region_pass.group})" if self.region_pass is not None else "single pass",
            "Target dataset"      : self.target.describe() if self.target is not None else "none (scored on its own split)",
            "Device"              : cfg.device,
            "Checkpoint"          : cfg.checkpoint_name,
            "Render amp floor"    : cfg.render_amp_floor,
            "Code commit"         : GitState.commit(),
            "Resolved config"     : meta.config_path,
            "Loader"              : self.components.loader_cls.__name__,
            "Predictor"           : self.components.predictor_cls.__name__,
            "Param Space"         : self.components.param_space,
            "Embedding Evaluator" : evaluator.__name__ if evaluator is not None else "none",
        }

    def _load_run(self, cfg: InferenceConfig, logger: Logger):

        loader = self.components.loader_cls(cfg.run_directory, logger=logger)
        return loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
            checkpoint_name = cfg.checkpoint_name,
            region_override = CropRegion(*cfg.region_override) if cfg.region_override else None,
            target          = self.target,
        )

    def _build_embedding_evaluator(self, cfg: InferenceConfig, run, logger: Logger):

        if self.components.embedding_evaluator_cls is None:
            return None

        return self.components.embedding_evaluator_cls(run, logger, window_kind=cfg.stitch_window)

    def _predict(self, cfg: InferenceConfig, meta: InferenceMetadata, run, evaluator, logger: Logger):

        evaluator_kwargs = {} if evaluator is None else {"embedding_evaluator": evaluator}

        predictor = self.components.predictor_cls(
            run              = run,
            logger           = logger,
            window_kind      = cfg.stitch_window,
            cube_dtype       = cfg.cube_dtype,
            save_cubes       = cfg.save_cubes,
            meta             = meta,
            cpu_workers      = cfg.cpu_workers,
            render_amp_floor = cfg.render_amp_floor,
            **evaluator_kwargs,
        )
        return predictor.run_inference()

    @staticmethod
    def _equal_indices(n_total: int, n_slices: int) -> np.ndarray:

        n_slices = max(1, min(n_slices, n_total))
        return np.unique(np.linspace(n_total * 0.1, n_total * 0.9, n_slices).round().astype(int))

    def _compute_slice_indices(self, cfg: InferenceConfig, n_z: int, n_rows: int, n_cols: int) -> dict:

        return {
            "slice_z_idx"   : self._equal_indices(n_z,    cfg.n_z_slices),
            "slice_col_idx" : self._equal_indices(n_cols, cfg.n_col_slices),
            "slice_row_idx" : self._equal_indices(n_rows, cfg.n_row_slices),
            "all_z_idx"     : np.arange(n_z),
            "all_col_idx"   : np.arange(n_cols),
            "all_row_idx"   : np.arange(n_rows),
        }

    def _extract_curve_params(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, z_axis_np: np.ndarray, logger: Logger) -> None:

        if self.components.param_space:
            return

        if not cfg.extract_curve_params:
            logger.section("[Inference: Curve Parameter Extraction skipped]")
            logger.subsection("extract_curve_params is disabled; this run predicts curves directly, so every Gaussian-parameter analysis stays absent from metrics, figures, and report.")
            return

        if result.params_gt is None:
            raise ValueError("Curve-parameter extraction needs the exact GT parameters to measure its own floor, but the predictor produced no params_gt cube; a curve-space predictor must stitch the GT parameters alongside the curves.")

        logger.section("[Inference: Curve Parameter Extraction]")

        extractor = CurveParamExtractor(
            z_axis             = z_axis_np,
            n_gaussians        = run.n_gaussians,
            activity_threshold = run.activity_threshold,
            prominence_frac    = cfg.extract_prominence_frac,
            n_workers          = cfg.cpu_workers,
        )

        predicted = extractor.run(result.pred_curves)
        reference = extractor.run(result.gt_curves)

        result.params_pred   = predicted.params
        result.params_source = "extracted"
        result.extract_r2    = predicted.fit_r2
        result.extract_floor = self._extraction_floor(result, reference.params, z_axis_np, run.n_gaussians, cfg.cpu_workers, run.activity_threshold, logger)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "params_pred.npy", predicted.params)
            np.save(meta.cube_dir / "extract_r2.npy",  predicted.fit_r2)

        logger.subsection(f"Extracted {run.n_gaussians} Gaussians per pixel from the predicted curves : median fit R² = {float(np.median(predicted.fit_r2)):.4f}")
        logger.subsection(f"Extraction floor (same extractor on GT curves vs exact GT parameters) : μ MAE = {result.extract_floor['matched_mu_mae']:.4g}, σ MAE = {result.extract_floor['matched_sig_mae']:.4g}")

    @staticmethod
    def _extraction_floor(result, reference_params: np.ndarray, z_axis_np: np.ndarray, n_gaussians: int, cpu_workers: int, active_amp_thr: float, logger: Logger) -> dict:

        floor = Result(
            pred_curves        = result.gt_curves,
            gt_curves          = result.gt_curves,
            pixel_mse          = result.pixel_mse,
            pixel_mae          = result.pixel_mae,
            pixel_r2           = result.pixel_r2,
            pixel_cosine       = result.pixel_cosine,
            pixel_peak_err_idx = result.pixel_peak_err_idx,
            cube_directory     = result.cube_directory,
            row_offset         = result.row_offset,
            col_offset         = result.col_offset,
            params_pred        = reference_params,
            params_gt          = result.params_gt,
        )

        return Metrics(floor, z_axis_np, n_gaussians, cpu_workers, active_amp_thr, logger).matched_gaussian_metrics()

    def _evaluate_metrics(self, cfg: InferenceConfig, result, z_axis_np: np.ndarray, run, meta: InferenceMetadata, indices: dict, logger: Logger) -> dict:

        param_space = result.params_pred is not None and result.params_gt is not None

        global_metrics = Metrics(result, z_axis_np, run.n_gaussians, cfg.cpu_workers, run.activity_threshold, logger).compute(
            z_indices         = indices["all_z_idx"],
            col_indices       = indices["all_col_idx"],
            row_indices       = indices["all_row_idx"],
            param_space       = param_space,
            slot_organization = result.params_source == "model",
        )

        global_metrics["params_source"] = result.params_source

        if global_metrics["n_pixels_gt_empty"]:
            logger.warning(f"{global_metrics['n_pixels_gt_empty']} of {global_metrics['n_pixels']} pixels ({global_metrics['gt_empty_frac'] * 100.0:.2f}%) have an empty GT profile (every label Gaussian at zero amplitude); they score cosine 0 and a peak error measured from bin 0 inside the headline pixel_cosine_gt_* and pixel_peak_* statistics, which the pixel_*_nonempty statistics leave out")

        if result.extract_r2 is not None:
            finite = result.extract_r2[np.isfinite(result.extract_r2)]
            global_metrics["extract_fit_r2_mean"]   = float(finite.mean())
            global_metrics["extract_fit_r2_median"] = float(np.median(finite))
            global_metrics["extract_fit_r2_p5"]     = float(np.percentile(finite, 5.0))

        if result.extract_floor is not None:
            for key, value in result.extract_floor.items():
                global_metrics[f"extract_floor_{key}"] = value

        global_metrics["split"]              = run.split_name
        global_metrics["split_region"]       = list(run.split_region.as_tuple())
        global_metrics["activity_threshold"] = run.activity_threshold
        global_metrics["checkpoint_name"]    = cfg.checkpoint_name
        global_metrics["checkpoint_epoch"]   = run.checkpoint_meta["epoch"] + 1
        global_metrics["code_commit"]        = GitState.commit()

        if self.region_pass is not None:
            global_metrics[RegionPass.KEY] = self.region_pass.payload()

        if self.target is not None:
            global_metrics[TargetDataset.KEY]  = self.target.payload()
            global_metrics["training_dataset"] = str(run.training_dataset)

        if run.core_region is not None:
            global_metrics["core_region"] = list(run.core_region.as_tuple())

        Metrics.write_json(global_metrics, meta.metrics_path)
        return global_metrics

    def _evaluate_embeddings(self, cfg: InferenceConfig, meta: InferenceMetadata, evaluator, result, global_metrics: dict, logger: Logger) -> None:

        if evaluator is None:
            return

        logger.section("[Inference: Embedding Evaluation]")

        metrics, maps = evaluator.run(pixel_mse=result.pixel_mse)

        result.embedding_maps = maps

        global_metrics.update(metrics)
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "embedding_error.npy",  maps.embedding_error)
            np.save(meta.cube_dir / "embedding_cosine.npy", maps.embedding_cosine)
            np.save(meta.cube_dir / "decode_mse.npy",       maps.decode_mse)
            np.save(meta.cube_dir / "chain_mse.npy",        maps.chain_mse)

        logger.subsection(f"Embeddings evaluated : {len(metrics)} metrics merged into global metrics")

    def _evaluate_label_quality(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, z_axis_np: np.ndarray, global_metrics: dict, logger: Logger) -> None:

        logger.section("[Inference: Label Quality]")

        r2_map, stats = Metrics(result, z_axis_np, run.n_gaussians, cfg.cpu_workers, run.activity_threshold, logger).label_quality(run.full_curves)

        result.label_r2 = r2_map
        global_metrics["label_quality_status"] = "computed"
        global_metrics.update(stats)
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "label_r2.npy", r2_map)

        logger.subsection(f"Label quality : mean fit R² = {stats['label_r2_mean']:.4f}, {stats['label_r2_frac_below_half'] * 100.0:.1f}% of labels below R² 0.5")

    def _evaluate_cube_ssim(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, z_axis_np: np.ndarray, global_metrics: dict, indices: dict, logger: Logger) -> None:

        if not cfg.compute_cube_ssim:
            logger.section("[Inference: Cube SSIM skipped]")
            logger.subsection("compute_cube_ssim is disabled; the standard SSIM against the raw cube is absent from metrics and report.")
            global_metrics["cube_ssim_status"] = "skipped: compute_cube_ssim disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Cube SSIM]")

        stats = Metrics(result, z_axis_np, run.n_gaussians, cfg.cpu_workers, run.activity_threshold, logger).cube_ssim(
            run.full_curves,
            z_indices   = indices["all_z_idx"],
            col_indices = indices["all_col_idx"],
            row_indices = indices["all_row_idx"],
        )

        global_metrics["cube_ssim_status"] = "computed"
        global_metrics.update(stats)
        Metrics.write_json(global_metrics, meta.metrics_path)

        logger.subsection(f"Cube SSIM : z {stats['ssim_cube_z_mean']:.4f}, col {stats['ssim_cube_col_mean']:.4f}, row {stats['ssim_cube_row_mean']:.4f} vs the raw cube")

    def _evaluate_flip_consistency(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:

        if not cfg.compute_flip_consistency:
            logger.section("[Inference: Flip Consistency skipped]")
            logger.subsection("compute_flip_consistency is disabled; the flip-equivariance disagreement map is absent from metrics, figures, and report.")
            global_metrics["flip_consistency_status"] = "skipped: compute_flip_consistency disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Flip Consistency]")

        evaluator = FlipConsistencyEvaluator(run, logger, window_kind=cfg.stitch_window, render_amp_floor=cfg.render_amp_floor, render_params=self.components.param_space)
        flip_map  = evaluator.run()

        result.flip_consistency = flip_map

        finite = flip_map[np.isfinite(flip_map)]
        if not finite.size:
            raise ValueError("Flip-consistency map holds no finite value; the stitched disagreement is degenerate")

        err_flat  = result.pixel_mse.reshape(-1).astype(np.float64)
        flip_flat = flip_map.reshape(-1).astype(np.float64)
        both      = np.isfinite(err_flat) & np.isfinite(flip_flat)
        corr      = float(np.corrcoef(flip_flat[both], err_flat[both])[0, 1]) if both.sum() > 2 else float("nan")

        global_metrics["flip_consistency_status"] = "computed"
        global_metrics["flip_mse_mean"]           = float(finite.mean())
        global_metrics["flip_mse_median"]         = float(np.median(finite))
        global_metrics["flip_mse_p95"]            = float(np.percentile(finite, 95.0))
        global_metrics["flip_error_corr"]         = corr
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "flip_consistency.npy", flip_map)

        logger.subsection(f"Flip consistency : mean disagreement {global_metrics['flip_mse_mean']:.4g}, correlation with pixel MSE {corr:.3f}")

    def _stratify_errors(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:

        if not cfg.compute_stratified:
            logger.section("[Inference: Stratified Errors skipped]")
            logger.subsection("compute_stratified is disabled; the covariate-stratified error curves are absent from metrics, figures, and report.")
            global_metrics["stratified_status"] = "skipped: compute_stratified disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Stratified Errors]")

        tables, scalars   = StratifiedErrors(run, result).run()
        result.stratified = tables

        global_metrics["stratified_status"] = "computed"
        global_metrics.update(scalars)
        Metrics.write_json(global_metrics, meta.metrics_path)

        for name, table in sorted(tables.items()):
            n_nonfinite = int(scalars[f"strat_{name}_pixels_nonfinite"])

            if table["dropped"]:
                bins = ", ".join(f"{entry['center']:.4g} (n={entry['n']})" for entry in table["dropped"])
                logger.warning(f"Stratified errors, {name}: {len(table['dropped'])} bin(s) under {StratifiedErrors.MIN_BIN_COUNT} pixels left out of the table and figure, {int(scalars[f'strat_{name}_pixels_dropped'])} pixels in all: {bins}")

            if n_nonfinite:
                logger.warning(f"Stratified errors, {name}: {n_nonfinite} pixels with a non-finite covariate or error left out of every bin")

        spearman  = {name: scalars[f"strat_{name}_spearman"] for name in sorted(tables)}
        finite    = {name: value for name, value in spearman.items() if np.isfinite(value)}
        undefined = [name for name in spearman if name not in finite]

        if undefined:
            logger.subsection(f"No rank correlation with error for {undefined}: the covariate or the error is constant on this region")

        if not finite:
            logger.warning(f"Stratified over {sorted(tables)}, but every rank correlation with error is undefined; no covariate can be named as the strongest")
            return

        strongest = max(finite, key=lambda name: abs(finite[name]))
        logger.subsection(f"Stratified over {sorted(tables)}; strongest rank correlation with error: {strongest} ({finite[strongest]:.3f})")

    def _classify_failures(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:

        if result.params_pred is None or result.params_gt is None:
            logger.section("[Inference: Failure Modes skipped]")
            logger.subsection("No parameter cubes are available; the failure-mode classification is absent from metrics, figures, and report.")
            global_metrics["failure_mode_status"] = "skipped: no parameter cubes"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Failure Modes]")

        modes = FailureModes(result.params_pred, result.params_gt, run.n_gaussians, assignment=result.assignment(run.n_gaussians, run.activity_threshold), active_amp_thr=run.activity_threshold)
        pairs = modes.pair_predictions()

        mode_map, by_sep, scalars = modes.classify(pairs)

        result.failure_mode_map = mode_map
        result.failure_pairs    = pairs
        result.miss_by_sep      = by_sep

        global_metrics["failure_mode_status"] = "computed"
        global_metrics.update(scalars)
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "failure_mode.npy", mode_map)

        logger.subsection(f"Failure modes : miss rate {scalars['failure_miss_rate'] * 100.0:.2f}%, hallucination rate {scalars['failure_halluc_rate'] * 100.0:.2f}%, clean pixels {scalars['failure_frac_ok'] * 100.0:.1f}%")
        logger.subsection(f"Miss rate by GT separation : {scalars['failure_sep_status']}")

    def _calibrate_presence(self, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:

        if result.failure_pairs is None:
            logger.section("[Inference: Presence Calibration skipped]")
            logger.subsection("No parameter cubes are available; the presence reliability curve is absent from metrics, figures, and report.")
            global_metrics["presence_status"] = "skipped: no parameter cubes"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Presence Calibration]")

        rows, scalars        = PresenceCalibration(result.failure_pairs).run()
        result.presence_rows = rows

        global_metrics["presence_status"] = "computed"
        global_metrics.update(scalars)
        Metrics.write_json(global_metrics, meta.metrics_path)

        logger.subsection(f"Presence calibration : overall precision {scalars['presence_overall_precision'] * 100.0:.1f}%, amp-precision rank correlation {scalars['presence_rank_corr']:.3f}")

    def _audit_normalization(self, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:

        logger.section("[Inference: Normalization Audit]")

        stats = NormalizationAudit(run, result).run()
        global_metrics.update(stats)
        Metrics.write_json(global_metrics, meta.metrics_path)

        logger.subsection(f"Input drift : worst |mean| {stats['norm_in_worst_abs_mean']:.3f}, worst |std-1| {stats['norm_in_worst_std_dev']:.3f}, overflow {stats['norm_in_overflow_frac'] * 100.0:.3f}%, non-finite {stats['norm_in_nonfinite_frac'] * 100.0:.4f}%")

        if "clamp_amp_ceil_frac" in stats:
            logger.subsection(f"Clamp hits  : amp ceil {stats['clamp_amp_ceil_frac'] * 100.0:.2f}%, sigma floor {stats['clamp_sigma_floor_frac'] * 100.0:.2f}%, mu at axis edge {stats['clamp_mu_edge_frac'] * 100.0:.2f}%")

        for line in NormalizationAudit.warning_lines(stats):
            logger.warning(f"Normalization audit, {line}")

    def _compose_figures(self, composer: FigureComposer, result, run, global_metrics: dict, z_axis_np: np.ndarray, indices: dict) -> Dict[str, List[Path]]:

        return composer.compose(
            result         = result,
            run            = run,
            global_metrics = global_metrics,
            z_axis_np      = z_axis_np,
            indices        = indices,
            param_space    = result.params_pred is not None and result.params_gt is not None,
        )

    def _build_report(
        self,
        meta           : InferenceMetadata,
        run,
        cfg            : InferenceConfig,
        z_axis_np      : np.ndarray,
        global_metrics : dict,
        figure_paths   : Dict[str, List[Path]],
        gif_paths      : Dict[str, Path],
    ) -> Path:

        return Report(
            output_dir       = meta.output_dir,
            run_summary      = ReportPayloadBuilder.run_summary(run, z_axis_np),
            inference_config = ReportPayloadBuilder.inference_config(cfg, run),
            checkpoint_meta  = run.checkpoint_meta,
            global_metrics   = global_metrics,
            figure_paths     = figure_paths,
            gif_paths        = gif_paths,
            report_path      = meta.report_path,
        ).assemble()

    def run(self) -> Path:

        cfg                   = self.config
        meta, logger, plotter = self._setup(cfg)
        try:
            CompletionMarker.clear(meta.output_dir)
            run       = self._load_run(cfg, logger)
            evaluator = self._build_embedding_evaluator(cfg, run, logger)
            result    = self._predict(cfg, meta, run, evaluator, logger)

            z_axis_np        = np.asarray(run.z_axis, dtype=np.float64)
            _N_z, _row, _col = result.pred_curves.shape

            self._extract_curve_params(cfg, meta, run, result, z_axis_np, logger)

            logger.section("[Inference: Metrics]")
            indices        = self._compute_slice_indices(cfg, _N_z, _row, _col)
            global_metrics = self._evaluate_metrics(cfg, result, z_axis_np, run, meta, indices, logger)
            logger.subsection(f"Global metrics written : {meta.metrics_path}")

            self._evaluate_embeddings(cfg, meta, evaluator, result, global_metrics, logger)
            self._evaluate_label_quality(cfg, meta, run, result, z_axis_np, global_metrics, logger)
            self._evaluate_cube_ssim(cfg, meta, run, result, z_axis_np, global_metrics, indices, logger)
            self._evaluate_flip_consistency(cfg, meta, run, result, global_metrics, logger)
            self._stratify_errors(cfg, meta, run, result, global_metrics, logger)
            self._classify_failures(cfg, meta, run, result, global_metrics, logger)
            self._calibrate_presence(meta, run, result, global_metrics, logger)
            self._audit_normalization(meta, run, result, global_metrics, logger)

            figure_paths = {}
            gif_paths    = {}

            if cfg.save_plots or cfg.save_animations:
                composer = FigureComposer(plotter=plotter, meta=meta, logger=logger, cfg=cfg)

                if cfg.save_plots:
                    logger.section("[Inference: Plots]")
                    figure_paths = self._compose_figures(composer, result, run, global_metrics, z_axis_np, indices)
                else:
                    logger.section("[Inference: Plots skipped]")

                if cfg.save_animations:
                    logger.section("[Inference: Animations]")
                    gif_paths = composer.animate(result, run, z_axis_np)
                else:
                    logger.section("[Inference: Animations skipped]")
            else:
                logger.section("[Inference: Plots and animations skipped]")

            logger.section("[Inference: Report]")
            report_path = self._build_report(
                meta           = meta,
                run            = run,
                cfg            = cfg,
                z_axis_np      = z_axis_np,
                global_metrics = global_metrics,
                figure_paths   = figure_paths,
                gif_paths      = gif_paths,
            )

            marker = {
                "stage"                           : "inference",
                "split"                           : run.split_name,
                "inference_config"                : meta.config_path.name,
                CompletedInference.CHECKPOINT_KEY : {"name": cfg.checkpoint_name, **run.checkpoint_meta},
            }

            if self.region_pass is not None:
                marker[RegionPass.KEY] = self.region_pass.payload()

            if self.target is not None:
                marker[TargetDataset.KEY] = self.target.payload()

            CompletionMarker.stamp(meta.output_dir, marker)

            logger.section("[Inference Pipeline Done]")
            logger.kv_table({
                "pixel_mse_gt_mean" : global_metrics["pixel_mse_gt_mean"],
                "pixel_mae_gt_mean" : global_metrics["pixel_mae_gt_mean"],
                "ssim_gt_z_mean"    : global_metrics["ssim_gt_z_mean"],
            })
            logger.subsection(f"Report  : {report_path}")
            logger.subsection(f"Metrics : {meta.metrics_path}")
            if cfg.save_cubes:
                logger.subsection(f"Cubes   : {len(list(result.cube_directory.glob('*.npy')))} .npy files in {result.cube_directory}\n")
            else:
                logger.subsection("Cubes   : not saved (save_cubes=False)\n")

            return report_path
        finally:
            logger.close()
