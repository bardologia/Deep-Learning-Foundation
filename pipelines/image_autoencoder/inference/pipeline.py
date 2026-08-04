from __future__ import annotations

from pathlib  import Path

import numpy as np

from configuration.inference.image_autoencoder        import ImageAeInferenceConfig
from pipelines.image_autoencoder.inference.loader     import ImageAeRunLoader
from pipelines.image_autoencoder.inference.metrics    import ImageAeMetrics
from pipelines.image_autoencoder.inference.plots      import ImageAePlots
from pipelines.image_autoencoder.inference.predictor  import ImageAePredictor
from pipelines.image_autoencoder.inference.report     import ImageAeReport
from pipelines.shared.inference.metadata              import InferenceMetadata
from tools.monitoring.logger                          import Logger
from tools.reporting.plotting                         import PlotBase


class ImageAeInferenceMetadata(InferenceMetadata):
    SUBDIR = "image_ae"


class ImageAeInferencePipeline:
    def __init__(self, config: ImageAeInferenceConfig) -> None:
        self.config = config

    def _setup(self) -> tuple[ImageAeInferenceMetadata, Logger]:
        meta = ImageAeInferenceMetadata(self.config)
        meta.create_dirs()

        logger = Logger(log_dir=str(meta.logs_dir), name="image_ae_inference", level=self.config.log_level)
        logger.section("[Image AE Inference Pipeline]")
        logger.kv_table({
            "Run Directory" : self.config.run_directory,
            "Output Dir"    : meta.output_dir,
            "Split"         : self.config.split,
            "Device"        : self.config.device,
        })

        return meta, logger

    def _load_run(self, logger: Logger):
        loader = ImageAeRunLoader(self.config.run_directory, logger=logger)
        return loader.load(config=self.config, device=self.config.device)

    def _predict(self, run, logger: Logger):
        return ImageAePredictor(run, device=self.config.device, logger=logger).run_inference()

    def _persist_embeddings(self, result, meta: ImageAeInferenceMetadata, logger: Logger) -> Path:
        path = meta.output_dir / "embeddings.npy"
        np.save(path, result.embeddings)
        logger.subsection(f"Embeddings saved : {path}  shape {result.embeddings.shape}")
        return path

    def _evaluate_metrics(self, result, run, meta: ImageAeInferenceMetadata, logger: Logger):
        metrics_obj = ImageAeMetrics(result)
        metrics     = metrics_obj.compute()

        metrics["split"]        = run.split_name
        metrics["split_region"] = list(run.split_region.as_tuple())

        ImageAeMetrics.write_json(metrics, meta.metrics_path)

        logger.section("[Image AE Inference: Metrics]")
        logger.kv_table({
            "MSE"  : f"{metrics['mse_mean']:.6g}",
            "RMSE" : f"{metrics['rmse']:.6g}",
            "R2"   : f"{metrics['r2']:.4f}",
            "PSNR" : f"{metrics['psnr']:.2f}",
        })

        return metrics, metrics_obj

    def _compose_figures(self, result, metrics: dict, metrics_obj, meta: ImageAeInferenceMetadata, logger: Logger) -> dict:
        if not self.config.save_plots:
            logger.section("[Image AE Inference: Plots skipped]")
            return {}

        logger.section("[Image AE Inference: Plots]")
        PlotBase.use_style(self.config.figure_style)
        plots = ImageAePlots(fig_dpi=self.config.fig_dpi, save_dpi=self.config.save_dpi)

        return plots.compose(result, metrics["channel_mse"], metrics_obj.per_patch_mse(), self.config, meta.figures_dir)

    def _build_report(self, meta: ImageAeInferenceMetadata, run, metrics: dict, figures: dict) -> Path:
        return ImageAeReport(
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

        self._persist_embeddings(result, meta, logger)

        metrics, metrics_obj = self._evaluate_metrics(result, run, meta, logger)
        figures              = self._compose_figures(result, metrics, metrics_obj, meta, logger)

        logger.section("[Image AE Inference: Report]")
        report_path = self._build_report(meta, run, metrics, figures)

        logger.section("[Image AE Inference Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {meta.metrics_path}\n")
        logger.close()

        return report_path
