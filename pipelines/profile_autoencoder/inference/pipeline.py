from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.inference.profile_autoencoder         import ProfileAeInferenceConfig
from pipelines.profile_autoencoder.inference.loader      import ProfileAeRunLoader
from pipelines.profile_autoencoder.inference.metrics     import ProfileAeMetrics
from pipelines.profile_autoencoder.inference.plots       import ProfileAePlots
from pipelines.profile_autoencoder.inference.predictor   import ProfileAePredictor
from pipelines.profile_autoencoder.inference.report      import ProfileAeReport
from pipelines.shared.inference.metadata                 import InferenceMetadata
from tools.monitoring.logger                             import Logger
from tools.reporting.plotting                            import PlotBase


class ProfileAeInferenceMetadata(InferenceMetadata):
    SUBDIR = "profile_ae"


class ProfileAeInferencePipeline:
    def __init__(self, config: ProfileAeInferenceConfig) -> None:
        self.config = config

    def _setup(self) -> tuple[ProfileAeInferenceMetadata, Logger]:
        meta = ProfileAeInferenceMetadata(self.config)
        meta.create_dirs()

        logger = Logger(log_dir=str(meta.logs_dir), name="profile_ae_inference", level=self.config.log_level)
        logger.section("[Profile AE Inference Pipeline]")
        logger.kv_table({
            "Run Directory" : self.config.run_directory,
            "Output Dir"    : meta.output_dir,
            "Split"         : self.config.split,
            "Device"        : self.config.device,
        })

        return meta, logger

    def _load_run(self, logger: Logger):
        loader = ProfileAeRunLoader(self.config.run_directory, logger=logger)
        return loader.load(config=self.config, device=self.config.device)

    def _predict(self, run, logger: Logger):
        return ProfileAePredictor(run, device=self.config.device, logger=logger).run_inference()

    def _persist_embeddings(self, result, run, meta: ProfileAeInferenceMetadata, logger: Logger) -> Path:
        path       = meta.output_dir / "embeddings.npy"
        index_path = meta.output_dir / "pixel_indices.npy"

        np.save(path, result.embeddings)
        np.save(index_path, run.dataset.index)

        logger.subsection(f"Embeddings saved    : {path}  shape {result.embeddings.shape}")
        logger.subsection(f"Pixel indices saved : {index_path}  shape {run.dataset.index.shape}")
        return path

    def _evaluate_metrics(self, result, run, meta: ProfileAeInferenceMetadata, logger: Logger):
        metrics_obj = ProfileAeMetrics(result, run.x_axis, run.amp_zero_thr)
        metrics     = metrics_obj.compute()

        metrics["split"]         = run.split_name
        metrics["split_regions"] = [list(region.as_tuple()) for region in run.split_regions]

        ProfileAeMetrics.write_json(metrics, meta.metrics_path)

        logger.section("[Profile AE Inference: Metrics]")
        logger.kv_table({
            "MSE"          : f"{metrics['mse_mean']:.6g}",
            "R2"           : f"{metrics['r2']:.4f}",
            "Pearson"      : f"{metrics['pearson_mean']:.4f}",
            "Relative L2"  : f"{metrics['relative_l2_mean']:.4f}",
            "Peak loc MAE" : f"{metrics['peak_location_mae']:.4f}",
        })

        return metrics, metrics_obj

    def _compose_figures(self, result, run, metrics_obj, meta: ProfileAeInferenceMetadata, logger: Logger) -> dict:
        if not self.config.save_plots:
            logger.section("[Profile AE Inference: Plots skipped]")
            return {}

        logger.section("[Profile AE Inference: Plots]")
        PlotBase.use_style(self.config.figure_style)
        plots = ProfileAePlots(fig_dpi=self.config.fig_dpi, save_dpi=self.config.save_dpi)

        return plots.compose(result, np.asarray(run.x_axis, dtype=np.float64), metrics_obj.per_curve_mse(), self.config, meta.figures_dir, run.amp_zero_thr)

    def _build_report(self, meta: ProfileAeInferenceMetadata, run, metrics: dict, figures: dict) -> Path:
        return ProfileAeReport(
            output_dir  = meta.output_dir,
            run         = run,
            config      = self.config,
            metrics     = metrics,
            figures     = figures,
            report_path = meta.report_path,
        ).assemble()

    def run(self) -> Path:
        meta, logger = self._setup()

        run    = self._load_run(logger)
        result = self._predict(run, logger)

        self._persist_embeddings(result, run, meta, logger)

        metrics, metrics_obj = self._evaluate_metrics(result, run, meta, logger)
        figures              = self._compose_figures(result, run, metrics_obj, meta, logger)

        logger.section("[Profile AE Inference: Report]")
        report_path = self._build_report(meta, run, metrics, figures)

        logger.section("[Profile AE Inference Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {meta.metrics_path}\n")
        logger.close()

        return report_path
