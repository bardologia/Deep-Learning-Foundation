from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict, List, Type

import numpy as np

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.data_consistency   import DataConsistencyEvaluator
from pipelines.backbone.inference.figures            import FigureComposer
from pipelines.backbone.inference.flip_consistency   import FlipConsistencyEvaluator
from pipelines.backbone.inference.normalization_audit import NormalizationAudit
from pipelines.backbone.inference.loader             import RunLoader
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.backbone.inference.metrics            import Metrics
from pipelines.backbone.inference.plots              import Plotter
from pipelines.backbone.inference.predictor          import Predictor
from pipelines.backbone.inference.reduced            import ReducedTomogramSynthesizer
from pipelines.backbone.inference.failure_modes      import FailureModes
from pipelines.backbone.inference.presence_calibration import PresenceCalibration
from pipelines.backbone.inference.report             import Report, ReportPayloadBuilder
from pipelines.backbone.inference.stratified         import StratifiedErrors
from tools.monitoring.logger                         import Logger
from tools.reporting.plotting                        import PlotBase
from tools.runtime.completion                        import CompletionMarker


@dataclass(frozen=True)
class InferenceComponents:
    loader_cls              : Type[RunLoader] = RunLoader
    predictor_cls           : Type[Predictor] = Predictor
    param_space             : bool            = True
    embedding_evaluator_cls : Type | None     = None


class InferencePipeline:
    def __init__(self, config: InferenceConfig, components: InferenceComponents | None = None) -> None:
        self.config     = config
        self.components = components if components is not None else InferenceComponents()

    def _setup(self, cfg: InferenceConfig) -> tuple[InferenceMetadata, Logger, Plotter]:
        meta = InferenceMetadata(cfg)
        meta.create_dirs()
        np.random.seed(cfg.seed)

        evaluator = self.components.embedding_evaluator_cls

        logger = Logger(log_dir=str(meta.logs_dir), name="inference", level=cfg.log_level)
        logger.section("[Inference Pipeline]")
        logger.kv_table({
            "Run Directory"       : cfg.run_directory,
            "Output Dir"          : meta.output_dir,
            "Split"               : cfg.split,
            "Device"              : cfg.device,
            "Checkpoint"          : cfg.checkpoint_name,
            "Loader"              : self.components.loader_cls.__name__,
            "Predictor"           : self.components.predictor_cls.__name__,
            "Param Space"         : self.components.param_space,
            "Embedding Evaluator" : evaluator.__name__ if evaluator is not None else "none",
        })

        PlotBase.use_style(cfg.figure_style)

        plotter = Plotter(
            cmap      = cfg.cmap_intensity,
            err_cmap  = cfg.cmap_error,
            normalize = cfg.normalize_intensity,
            fig_dpi   = cfg.fig_dpi,
            save_dpi  = cfg.save_dpi,
        )

        return meta, logger, plotter

    def _load_run(self, cfg: InferenceConfig, logger: Logger):
        loader = self.components.loader_cls(cfg.run_directory, logger=logger)
        return loader.load(
            split           = cfg.split,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            device          = cfg.device,
            checkpoint_name = cfg.checkpoint_name,
        )

    def _predict(self, cfg: InferenceConfig, meta: InferenceMetadata, run, logger: Logger):
        predictor = self.components.predictor_cls(
            run         = run,
            logger      = logger,
            window_kind      = cfg.stitch_window,
            cube_dtype       = cfg.cube_dtype,
            save_cubes       = cfg.save_cubes,
            meta             = meta,
            cpu_workers      = cfg.cpu_workers,
            render_amp_floor = cfg.render_amp_floor,
        )
        return predictor.run_inference()

    @staticmethod
    def _equal_indices(n_total: int, n_slices: int) -> np.ndarray:
        n_slices = max(1, min(n_slices, n_total))
        return np.unique(np.linspace(n_total * 0.1, n_total * 0.9, n_slices).round().astype(int))

    def _compute_slice_indices(self, cfg: InferenceConfig, n_elev: int, n_az: int, n_rg: int) -> dict:
        return {
            "slice_elev_idx"  : self._equal_indices(n_elev, cfg.n_elevation_slices),
            "slice_range_idx" : self._equal_indices(n_rg,   cfg.n_range_slices),
            "slice_az_idx"    : self._equal_indices(n_az,   cfg.n_azimuth_slices),
            "all_elev_idx"    : np.arange(n_elev),
            "all_range_idx"   : np.arange(n_rg),
            "all_az_idx"      : np.arange(n_az),
        }

    def _evaluate_metrics(self, result, x_axis_np: np.ndarray, run, meta: InferenceMetadata, indices: dict) -> dict:
        global_metrics = Metrics(result, x_axis_np, run.n_gaussians).compute(
            elev_indices  = indices["all_elev_idx"],
            range_indices = indices["all_range_idx"],
            az_indices    = indices["all_az_idx"],
            param_space   = self.components.param_space,
        )

        global_metrics["split"]        = run.split_name
        global_metrics["split_region"] = list(run.split_region.as_tuple())

        if run.track_baselines is not None:
            global_metrics["tracks"] = run.track_baselines.to_payload()

        if run.track_profiles is not None:
            global_metrics["track_positions"] = run.track_profiles.position_summary()

        Metrics.write_json(global_metrics, meta.metrics_path)
        return global_metrics

    def _evaluate_embeddings(self, meta: InferenceMetadata, run, global_metrics: dict, logger: Logger) -> None:
        if self.components.embedding_evaluator_cls is None:
            return

        evaluator = self.components.embedding_evaluator_cls(run, logger)

        global_metrics.update(evaluator.run())
        Metrics.write_json(global_metrics, meta.metrics_path)

    def _synthesize_reduced(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, x_axis_np: np.ndarray, global_metrics: dict, indices: dict, logger: Logger) -> None:
        if not cfg.compute_reduced:
            logger.section("[Inference: Reduced Capon Baseline skipped]")
            logger.subsection("compute_reduced is disabled; the NN-vs-reduced-vs-GT comparison is absent from metrics, figures, and report.")
            global_metrics["reduced_status"] = "skipped: compute_reduced disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        if run.secondary_labels is None:
            logger.section("[Inference: Reduced Capon Baseline skipped]")
            logger.subsection("Run uses the full secondary stack; the reduced tomogram would equal the ground-truth tomogram, so no baseline comparison exists.")
            global_metrics["reduced_status"] = "skipped: full secondary stack"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        synth     = ReducedTomogramSynthesizer(run, meta, cfg, logger)
        synthesis = synth.run(result.gt_curves)

        comparison = Metrics(result, x_axis_np, run.n_gaussians).reduced_comparison(
            synthesis.curves,
            elev_indices  = indices["all_elev_idx"],
            range_indices = indices["all_range_idx"],
            az_indices    = indices["all_az_idx"],
        )

        result.reduced = comparison
        global_metrics["reduced_status"] = "computed"
        global_metrics.update(synthesis.orientation)
        global_metrics.update(comparison.metrics)
        Metrics.write_json(global_metrics, meta.metrics_path)

        logger.subsection(f"Reduced baseline merged : relative MSE reduction = {comparison.metrics['relative_mse_reduction']:.4f}, NN beats reduced on {comparison.metrics['fraction_pred_beats_reduced'] * 100.0:.1f}% of pixels")

    def _evaluate_data_consistency(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, x_axis_np: np.ndarray, global_metrics: dict, logger: Logger) -> None:
        if not cfg.compute_data_consistency:
            logger.section("[Inference: Data Consistency skipped]")
            logger.subsection("compute_data_consistency is disabled; interferometric-consistency metrics are absent from metrics, figures, and report.")
            global_metrics["data_consistency_status"] = "skipped: compute_data_consistency disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        evaluator   = DataConsistencyEvaluator(run, cfg, logger)
        consistency = evaluator.run(result.pred_curves, result.gt_curves, x_axis_np)

        result.data_consistency = consistency
        global_metrics["data_consistency_status"] = "computed"
        global_metrics.update(consistency.metrics)
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "physics_coherence_error.npy",  consistency.coherence_error_map)
            np.save(meta.cube_dir / "physics_covariance_error.npy", consistency.covariance_error_map)
            np.save(meta.cube_dir / "physics_valid_mask.npy",       consistency.valid_mask)

    def _evaluate_label_quality(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, x_axis_np: np.ndarray, global_metrics: dict, logger: Logger) -> None:
        r2_map, stats = Metrics(result, x_axis_np, run.n_gaussians).label_quality(run.full_curves)

        result.label_r2 = r2_map
        global_metrics["label_quality_status"] = "computed"
        global_metrics.update(stats)
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "label_r2.npy", r2_map)

        logger.subsection(f"Label quality : mean fit R² = {stats['label_r2_mean']:.4f}, {stats['label_r2_frac_below_half'] * 100.0:.1f}% of labels below R² 0.5")

    def _evaluate_flip_consistency(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:
        if not cfg.compute_flip_consistency:
            logger.section("[Inference: Flip Consistency skipped]")
            logger.subsection("compute_flip_consistency is disabled; the flip-equivariance disagreement map is absent from metrics, figures, and report.")
            global_metrics["flip_consistency_status"] = "skipped: compute_flip_consistency disabled"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Flip Consistency]")

        evaluator = FlipConsistencyEvaluator(run, logger, window_kind=cfg.stitch_window, render_amp_floor=cfg.render_amp_floor)
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

        strongest = max(tables, key=lambda name: abs(scalars[f"strat_{name}_spearman"]))
        logger.subsection(f"Stratified over {sorted(tables)}; strongest rank correlation with error: {strongest} ({scalars[f'strat_{strongest}_spearman']:.3f})")

    def _classify_failures(self, cfg: InferenceConfig, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:
        if result.params_pred is None or result.params_gt is None:
            logger.section("[Inference: Failure Modes skipped]")
            logger.subsection("No parameter cubes are available; the failure-mode classification is absent from metrics, figures, and report.")
            global_metrics["failure_mode_status"] = "skipped: no parameter cubes"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Failure Modes]")

        mode_map, by_sep, scalars = FailureModes(result.params_pred, result.params_gt, run.n_gaussians, assignment=result.assignment(run.n_gaussians)).run()

        result.failure_mode_map = mode_map
        result.miss_by_sep      = by_sep

        global_metrics["failure_mode_status"] = "computed"
        global_metrics.update(scalars)
        Metrics.write_json(global_metrics, meta.metrics_path)

        if cfg.save_cubes:
            np.save(meta.cube_dir / "failure_mode.npy", mode_map)

        logger.subsection(f"Failure modes : miss rate {scalars['failure_miss_rate'] * 100.0:.2f}%, hallucination rate {scalars['failure_halluc_rate'] * 100.0:.2f}%, clean pixels {scalars['failure_frac_ok'] * 100.0:.1f}%")

    def _calibrate_presence(self, meta: InferenceMetadata, run, result, global_metrics: dict, logger: Logger) -> None:
        if result.params_pred is None or result.params_gt is None:
            logger.section("[Inference: Presence Calibration skipped]")
            logger.subsection("No parameter cubes are available; the presence reliability curve is absent from metrics, figures, and report.")
            global_metrics["presence_status"] = "skipped: no parameter cubes"
            Metrics.write_json(global_metrics, meta.metrics_path)
            return

        logger.section("[Inference: Presence Calibration]")

        rows, scalars        = PresenceCalibration(result.params_pred, result.params_gt, run.n_gaussians, assignment=result.assignment(run.n_gaussians)).run()
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

        logger.subsection(
            f"Input drift : worst |mean| {stats['norm_in_worst_abs_mean']:.3f}, worst |std-1| {stats['norm_in_worst_std_dev']:.3f}, "
            f"overflow {stats['norm_in_overflow_frac'] * 100.0:.3f}%"
        )

        if "clamp_amp_ceil_frac" in stats:
            logger.subsection(f"Clamp hits  : amp ceil {stats['clamp_amp_ceil_frac'] * 100.0:.2f}%, sigma floor {stats['clamp_sigma_floor_frac'] * 100.0:.2f}%, mu at axis edge {stats['clamp_mu_edge_frac'] * 100.0:.2f}%")

    def _compose_figures(self, composer: FigureComposer, result, run, global_metrics: dict, x_axis_np: np.ndarray, indices: dict) -> Dict[str, List[Path]]:
        return composer.compose(
            result         = result,
            run            = run,
            global_metrics = global_metrics,
            x_axis_np      = x_axis_np,
            indices        = indices,
            param_space    = self.components.param_space,
        )

    def _build_report(
        self,
        meta           : InferenceMetadata,
        run,
        cfg            : InferenceConfig,
        x_axis_np      : np.ndarray,
        global_metrics : dict,
        figure_paths   : Dict[str, List[Path]],
        gif_paths      : Dict[str, Path],
    ) -> Path:

        return Report(
            output_dir       = meta.output_dir,
            run_summary      = ReportPayloadBuilder.run_summary(run, x_axis_np),
            inference_config = ReportPayloadBuilder.inference_config(cfg, run),
            checkpoint_meta  = run.checkpoint_meta,
            global_metrics   = global_metrics,
            figure_paths     = figure_paths,
            gif_paths        = gif_paths,
            report_path      = meta.report_path,
        ).assemble()

    def run(self) -> Path:
        cfg                    = self.config
        meta, logger, plotter  = self._setup(cfg)
        CompletionMarker.clear(meta.output_dir)
        run    = self._load_run(cfg, logger)
        result = self._predict(cfg, meta, run, logger)

        x_axis_np         = np.asarray(run.x_axis, dtype=np.float64)
        _N_elev, _az, _rg = result.pred_curves.shape

        logger.section("[Inference: Metrics]")
        indices        = self._compute_slice_indices(cfg, _N_elev, _az, _rg)
        global_metrics = self._evaluate_metrics(result, x_axis_np, run, meta, indices)
        logger.subsection(f"Global metrics written : {meta.metrics_path}")

        self._evaluate_embeddings(meta, run, global_metrics, logger)
        self._synthesize_reduced(cfg, meta, run, result, x_axis_np, global_metrics, indices, logger)
        self._evaluate_data_consistency(cfg, meta, run, result, x_axis_np, global_metrics, logger)
        self._evaluate_label_quality(cfg, meta, run, result, x_axis_np, global_metrics, logger)
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
                figure_paths = self._compose_figures(composer, result, run, global_metrics, x_axis_np, indices)
            else:
                logger.section("[Inference: Plots skipped]")

            if cfg.save_animations:
                logger.section("[Inference: Animations]")
                gif_paths = composer.animate(result, run, x_axis_np)
            else:
                logger.section("[Inference: Animations skipped]")
        else:
            logger.section("[Inference: Plots and animations skipped]")

        logger.section("[Inference: Report]")
        report_path = self._build_report(
            meta           = meta,
            run            = run,
            cfg            = cfg,
            x_axis_np      = x_axis_np,
            global_metrics = global_metrics,
            figure_paths   = figure_paths,
            gif_paths      = gif_paths,
        )

        CompletionMarker.stamp(meta.output_dir, {
            "stage" : "inference",
            "split" : run.split_name,
        })

        logger.section("[Inference Pipeline Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {meta.metrics_path}")
        logger.subsection(f"Cubes   : {result.cube_directory}\n")
        logger.close()

        return report_path
