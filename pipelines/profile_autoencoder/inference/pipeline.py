"""Inference pipeline for the standalone profile autoencoder.

Binds the profile loader, predictor, metrics, plots and report onto the shared
autoencoder inference pipeline and persists the per-pixel index alongside the
embeddings so latent maps can be placed back into the scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipelines.autoencoder_common.inference.pipeline     import AeInferencePipeline
from pipelines.profile_autoencoder.inference.loader      import ProfileAeRunLoader
from pipelines.profile_autoencoder.inference.metrics     import ProfileAeMetrics
from pipelines.profile_autoencoder.inference.plots       import ProfileAePlots
from pipelines.profile_autoencoder.inference.predictor   import ProfileAePredictor
from pipelines.profile_autoencoder.inference.report      import ProfileAeReport
from pipelines.shared.inference.metadata                 import InferenceMetadata
from pipelines.shared.inference.run_classifier           import RunType
from tools.monitoring.logger                             import Logger


class ProfileAeInferenceMetadata(InferenceMetadata):
    """Inference metadata rooted in the profile-autoencoder output subdirectory."""

    SUBDIR = RunType.PROFILE_AE


class ProfileAeInferencePipeline(AeInferencePipeline):
    """Runs reconstruction, metrics, figures and reporting for a profile AE run."""

    FAMILY          = "Profile AE"
    LOGGER_NAME     = "profile_ae_inference"
    METADATA_CLASS  = ProfileAeInferenceMetadata
    LOADER_CLASS    = ProfileAeRunLoader
    PREDICTOR_CLASS = ProfileAePredictor
    REPORT_CLASS    = ProfileAeReport

    def _persist_embeddings(self, result, run, meta, logger: Logger) -> Path:
        """Saves the embeddings plus the flat pixel index they belong to.

        Args:
            result : Reconstruction result holding the embeddings.
            run    : Loaded profile-autoencoder run.
            meta   : Inference metadata providing the output directory.
            logger : Logger for the save report.

        Returns:
            Path of the saved embedding array.
        """
        path       = super()._persist_embeddings(result, run, meta, logger)
        index_path = meta.output_dir / "pixel_indices.npy"

        np.save(index_path, run.dataset.index)
        logger.subsection(f"Pixel indices saved : {index_path}  shape {run.dataset.index.shape}")

        return path

    def _metrics_object(self, result, run) -> ProfileAeMetrics:
        """Returns the metrics calculator bound to this run's axis and threshold."""
        return ProfileAeMetrics(result, run.x_axis, run.amp_zero_thr)

    def _region_metrics(self, run) -> dict:
        """Returns the split crop regions as plain tuples for the metrics file."""
        return {"split_regions": [list(region.as_tuple()) for region in run.split_regions]}

    def _metrics_summary(self, metrics: dict) -> dict:
        """Returns the headline metrics formatted for the console summary."""
        return {
            "MSE"          : f"{metrics['mse_mean']:.6g}",
            "R2"           : f"{metrics['r2']:.4f}",
            "Pearson"      : f"{metrics['pearson_mean']:.4f}",
            "Relative L2"  : f"{metrics['relative_l2_mean']:.4f}",
            "Peak loc MAE" : f"{metrics['peak_location_mae']:.4f}",
        }

    def _plot_figures(self, result, run, metrics: dict, metrics_obj, meta) -> dict:
        """Renders every figure group and returns their paths keyed by group.

        Args:
            result      : Reconstruction result.
            run         : Loaded profile-autoencoder run.
            metrics     : Computed metric dictionary.
            metrics_obj : Metrics calculator supplying the per-curve MSE.
            meta        : Inference metadata providing the figures directory.

        Returns:
            Mapping from figure-group name to the list of written paths.
        """
        plots = ProfileAePlots(fig_dpi=self.config.fig_dpi, save_dpi=self.config.save_dpi)

        return plots.compose(result, np.asarray(run.z_axis, dtype=np.float64), metrics_obj.per_sample_mse(), self.config, meta.figures_dir, run.amp_zero_thr, pixel_index=run.dataset.index, regions=run.split_regions)
