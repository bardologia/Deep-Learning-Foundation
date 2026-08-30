"""Image-autoencoder inference pipeline and its artifact metadata layout.

Binds the image-specific loader, predictor, metrics, plots and report into the
shared autoencoder inference pipeline.
"""

from __future__ import annotations

from pipelines.autoencoder_common.inference.pipeline  import AeInferencePipeline
from pipelines.image_autoencoder.inference.loader     import ImageAeRunLoader
from pipelines.image_autoencoder.inference.metrics    import ImageAeMetrics
from pipelines.image_autoencoder.inference.plots      import ImageAePlots
from pipelines.image_autoencoder.inference.predictor  import ImageAePredictor
from pipelines.image_autoencoder.inference.report     import ImageAeReport
from pipelines.shared.inference.metadata              import InferenceMetadata
from pipelines.shared.inference.run_classifier        import RunType


class ImageAeInferenceMetadata(InferenceMetadata):
    """Inference artifact layout written under the image-autoencoder subdirectory."""

    SUBDIR = RunType.IMAGE_AE


class ImageAeInferencePipeline(AeInferencePipeline):
    """Runs inference on a trained image autoencoder and writes its metrics, figures and report."""

    FAMILY          = "Image AE"
    LOGGER_NAME     = "image_ae_inference"
    METADATA_CLASS  = ImageAeInferenceMetadata
    LOADER_CLASS    = ImageAeRunLoader
    PREDICTOR_CLASS = ImageAePredictor
    REPORT_CLASS    = ImageAeReport

    def _metrics_object(self, result, run) -> ImageAeMetrics:
        """Returns the metrics computer for a predictor result."""
        return ImageAeMetrics(result)

    def _region_metrics(self, run) -> dict:
        """Returns the crop region of the inferred split as `(azimuth_start, azimuth_end, range_start, range_end)`."""
        return {"split_region": list(run.split_region.as_tuple())}

    def _metrics_summary(self, metrics: dict) -> dict:
        """Returns the formatted headline metrics logged at the end of the run."""
        return {
            "MSE"  : f"{metrics['mse_mean']:.6g}",
            "RMSE" : f"{metrics['rmse']:.6g}",
            "R2"   : f"{metrics['r2']:.4f}",
            "PSNR" : f"{metrics['psnr']:.2f}",
        }

    def _plot_figures(self, result, run, metrics: dict, metrics_obj, meta) -> dict:
        """Renders the reconstruction, error and embedding figures into the run's figure directory.

        Args:
            result      : Predictor result holding ground truth, reconstructions and embeddings.
            run         : Loaded run bundle.
            metrics     : Computed metrics dictionary.
            metrics_obj : Metrics computer, used for the per-patch errors.
            meta        : Inference metadata holding the output directories.

        Returns:
            Figure paths grouped by figure kind.
        """
        plots = ImageAePlots(fig_dpi=self.config.fig_dpi, save_dpi=self.config.save_dpi)

        return plots.compose(result, metrics["channel_mse"], metrics_obj.per_sample_mse(), self.config, meta.figures_dir)
