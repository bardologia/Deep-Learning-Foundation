"""Shared inference orchestration for the profile and image autoencoders.

Subclasses bind the family-specific metadata, loader, predictor and report
classes as class attributes and implement the metric and plotting hooks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipelines.autoencoder_common.inference.metrics import AeMetricsBase
from tools.monitoring.logger                        import Logger
from tools.reporting.plotting                       import PlotBase
from tools.runtime.completion                       import CompletionMarker


class AeInferencePipeline:
    """Runs a trained autoencoder over a split and writes metrics, plots and a report.

    Attributes:
        config : Inference entry configuration for the run.
    """

    FAMILY          = None
    LOGGER_NAME     = None
    METADATA_CLASS  = None
    LOADER_CLASS    = None
    PREDICTOR_CLASS = None
    REPORT_CLASS    = None

    def __init__(self, config) -> None:
        """Stores the inference configuration for the run.

        Args:
            config : Inference entry configuration naming the run directory, split, device and output options.
        """
        self.config = config

    def _setup(self):
        """Creates the output directories and logger, returning the metadata and logger."""
        meta = self.METADATA_CLASS(self.config)
        meta.create_dirs()

        logger = Logger(log_dir=str(meta.logs_dir), name=self.LOGGER_NAME, level=self.config.log_level)
        logger.section(f"[{self.FAMILY} Inference Pipeline]")
        logger.kv_table({
            "Run Directory" : self.config.run_directory,
            "Output Dir"    : meta.output_dir,
            "Split"         : self.config.split,
            "Device"        : self.config.device,
        })

        return meta, logger

    def _load_run(self, logger: Logger):
        """Returns the loaded run bundle of model, normalizer and split loader."""
        loader = self.LOADER_CLASS(self.config.run_directory, logger=logger)
        return loader.load(config=self.config, device=self.config.device)

    def _predict(self, run, logger: Logger):
        """Returns the reconstruction result for the whole split."""
        return self.PREDICTOR_CLASS(run, device=self.config.device, logger=logger).run_inference()

    def _persist_embeddings(self, result, run, meta, logger: Logger) -> Path:
        """Saves the latent embeddings of shape (n_samples, embedding_dim) and returns the file path."""
        path = meta.output_dir / "embeddings.npy"
        np.save(path, result.embeddings)

        logger.subsection(f"Embeddings saved    : {path}  shape {result.embeddings.shape}")

        return path

    def _metrics_object(self, result, run):
        """Returns the family-specific metrics object; implemented by subclasses."""
        raise NotImplementedError

    def _region_metrics(self, run) -> dict:
        """Returns the split region descriptors to merge into the metrics; implemented by subclasses."""
        raise NotImplementedError

    def _metrics_summary(self, metrics: dict) -> dict:
        """Returns the subset of metrics shown in the log table; implemented by subclasses."""
        raise NotImplementedError

    def _scored_weights(self, run) -> dict:
        """Returns the checkpoint name and one-based epoch of the weights being scored.

        Args:
            run : Loaded run bundle whose checkpoint_meta carries the zero-based checkpoint epoch.

        Returns:
            Mapping with 'checkpoint' and 'checkpoint_epoch', so a final_model.pt
            inference stays distinguishable from a best_model.pt one.
        """
        return {
            "checkpoint"       : self.config.checkpoint_name,
            "checkpoint_epoch" : int(run.checkpoint_meta["epoch"]) + 1,
        }

    def _provenance_metrics(self, run) -> dict:
        """Returns the run-level fields recorded next to the metrics: the scored weights, by default."""
        return self._scored_weights(run)

    def _evaluate_metrics(self, result, run, meta, logger: Logger):
        """Computes, writes and logs the metrics, returning them with the metrics object."""
        metrics_obj = self._metrics_object(result, run)
        metrics     = metrics_obj.compute()

        metrics["split"] = run.split_name
        metrics.update(self._region_metrics(run))
        metrics.update(self._provenance_metrics(run))

        AeMetricsBase.write_json(metrics, meta.metrics_path)

        logger.section(f"[{self.FAMILY} Inference: Metrics]")
        logger.kv_table(self._metrics_summary(metrics))

        return metrics, metrics_obj

    def _plot_figures(self, result, run, metrics: dict, metrics_obj, meta) -> dict:
        """Returns the figure name to path mapping; implemented by subclasses."""
        raise NotImplementedError

    def _compose_figures(self, result, run, metrics: dict, metrics_obj, meta, logger: Logger) -> dict:
        """Returns the rendered figures, or an empty mapping when plotting is disabled."""
        if not self.config.save_plots:
            logger.section(f"[{self.FAMILY} Inference: Plots skipped]")
            return {}

        logger.section(f"[{self.FAMILY} Inference: Plots]")
        PlotBase.use_style(self.config.figure_style)

        return self._plot_figures(result, run, metrics, metrics_obj, meta)

    def _build_report(self, meta, run, metrics: dict, figures: dict) -> Path:
        """Assembles the markdown inference report and returns its path."""
        return self.REPORT_CLASS(
            output_dir  = meta.output_dir,
            run         = run,
            config      = self.config,
            metrics     = metrics,
            figures     = figures,
            report_path = meta.report_path,
        ).assemble()

    def _infer(self, meta, logger: Logger) -> Path:
        """Runs load, predict, metrics, plots and report, stamping the completion marker.

        Args:
            meta   : Inference metadata holding the output directory and paths.
            logger : Logger writing into the run's log directory.

        Returns:
            Path of the assembled report.
        """
        CompletionMarker.clear(meta.output_dir)

        run    = self._load_run(logger)
        result = self._predict(run, logger)

        self._persist_embeddings(result, run, meta, logger)

        metrics, metrics_obj = self._evaluate_metrics(result, run, meta, logger)
        figures              = self._compose_figures(result, run, metrics, metrics_obj, meta, logger)

        logger.section(f"[{self.FAMILY} Inference: Report]")
        report_path = self._build_report(meta, run, metrics, figures)

        CompletionMarker.stamp(meta.output_dir, {
            "stage" : "inference",
            "split" : run.split_name,
            **self._scored_weights(run),
        })

        logger.section(f"[{self.FAMILY} Inference Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {meta.metrics_path}\n")

        return report_path

    def run(self) -> Path:
        """Executes the inference pipeline, closing the logger on every exit path.

        Returns:
            Path of the assembled report.

        Raises:
            BaseException : Re-raises any failure after logging it.
        """
        meta, logger = self._setup()

        try:
            return self._infer(meta, logger)
        except BaseException as error:
            logger.error(f"[{self.FAMILY} Inference Failed] {type(error).__name__}: {error}")
            raise
        finally:
            logger.close()
