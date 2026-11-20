from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.inference                         import InferenceConfig
from pipelines.backbone.inference.figures            import TargetMapFigures
from pipelines.backbone.inference.loader             import Run, RunLoader
from pipelines.backbone.inference.metrics            import Metrics
from pipelines.backbone.inference.predictor          import Predictor
from pipelines.backbone.inference.report             import Report
from pipelines.backbone.inference.run_metadata_paths import InferenceMetadata
from pipelines.shared.inference.metadata             import CompletedInference, RegionPass, TargetDataset
from tools.data.regions                              import CropRegion
from tools.monitoring.logger                         import Logger
from tools.reporting.plotting                        import PlotBase
from tools.runtime.completion                        import CompletionMarker
from tools.runtime.config_cli                        import ConfigCli
from tools.runtime.git_state                         import GitState


class InferencePipeline:

    def __init__(self, config: InferenceConfig, region_pass: RegionPass | None = None) -> None:
        self.config      = config
        self.region_pass = region_pass
        self.target      = TargetDataset.of(config)

    def _setup(self) -> tuple[InferenceMetadata, Logger]:
        meta = InferenceMetadata(self.config, self.target)
        meta.create_dirs()
        ConfigCli.save_resolved(self.config, meta.config_path)
        np.random.seed(self.config.seed)

        logger = Logger(log_dir=str(meta.logs_dir), name="inference", level=self.config.log_level)
        logger.section("[Inference Pipeline]")
        logger.kv_table({
            "Run Directory"   : self.config.run_directory,
            "Output Dir"      : meta.output_dir,
            "Split"           : self.config.split,
            "Region override" : self.config.region_override if self.config.region_override else "none (stored split region)",
            "Region pass"     : f"region {self.region_pass.index} of {self.region_pass.count} (group {self.region_pass.group})" if self.region_pass is not None else "single pass",
            "Target dataset"  : self.target.describe() if self.target is not None else "none (scored on its own split)",
            "Device"          : self.config.device,
            "Checkpoint"      : self.config.checkpoint_name,
            "Code commit"     : GitState.commit(),
            "Resolved config" : meta.config_path,
        })

        PlotBase.use_style(self.config.figure_style)

        return meta, logger

    def _load_run(self, logger: Logger) -> Run:
        return RunLoader(self.config.run_directory, logger=logger).load(
            split           = self.config.split,
            batch_size      = self.config.batch_size,
            num_workers     = self.config.num_workers,
            device          = self.config.device,
            checkpoint_name = self.config.checkpoint_name,
            region_override = CropRegion(*self.config.region_override) if self.config.region_override else None,
            target          = self.target,
        )

    def _save_maps(self, meta: InferenceMetadata, predictions: np.ndarray, run: Run, logger: Logger) -> None:
        if not self.config.save_maps:
            logger.subsection("Maps : not saved (save_maps=False)")
            return

        np.save(meta.maps_dir / "predictions.npy", predictions)
        np.save(meta.maps_dir / "targets.npy", run.targets)
        logger.subsection(f"Maps : predictions.npy and targets.npy, shape {predictions.shape}, in {meta.maps_dir}")

    def _evaluate(self, meta: InferenceMetadata, predictions: np.ndarray, run: Run, logger: Logger) -> dict:
        metrics = Metrics.compute(predictions, run.targets, run.target_channels)

        metrics["run_directory"]    = str(self.config.run_directory)
        metrics["split"]            = run.split_name
        metrics["split_region"]     = list(run.split_region.as_tuple())
        metrics["checkpoint_name"]  = self.config.checkpoint_name
        metrics["checkpoint_epoch"] = run.checkpoint_meta["epoch"] + 1
        metrics["code_commit"]      = GitState.commit()

        if run.core_region is not None:
            metrics["core_region"] = list(run.core_region.as_tuple())

        if self.region_pass is not None:
            metrics[RegionPass.KEY] = self.region_pass.payload()

        if self.target is not None:
            metrics[TargetDataset.KEY]  = self.target.payload()
            metrics["training_dataset"] = str(run.training_dataset)

        Metrics.write_json(metrics, meta.metrics_path)

        values = {name: Metrics.channel_values(metrics, name) for name in run.target_channels}
        rows   = [{"Channel": name, "RMSE": f"{entry['rmse']:.5g}", "MAE": f"{entry['mae']:.5g}", "Bias": f"{entry['bias']:+.5g}", "R2": f"{entry['r2']:.4f}"} for name, entry in values.items()]
        logger.metrics_table(rows, ["Channel", "RMSE", "MAE", "Bias", "R2"], title=f"Metrics on the {run.split_name} region")

        return metrics

    def _stamp(self, meta: InferenceMetadata, run: Run) -> None:
        marker = {
            "stage"                           : "inference",
            "split"                           : run.split_name,
            "inference_config"                : meta.config_path.name,
            CompletedInference.CHECKPOINT_KEY : {"name": self.config.checkpoint_name, **run.checkpoint_meta},
        }

        if self.region_pass is not None:
            marker[RegionPass.KEY] = self.region_pass.payload()

        if self.target is not None:
            marker[TargetDataset.KEY] = self.target.payload()

        CompletionMarker.stamp(meta.output_dir, marker)

    def run(self) -> Path:
        meta, logger = self._setup()
        try:
            CompletionMarker.clear(meta.output_dir)

            run         = self._load_run(logger)
            predictions = Predictor(run, logger, window_kind=self.config.stitch_window).predict()

            logger.section("[Inference: Metrics]")
            self._save_maps(meta, predictions, run, logger)
            metrics = self._evaluate(meta, predictions, run, logger)

            figure_paths = TargetMapFigures(self.config).compose(predictions, run.targets, run.target_channels, meta.figures_dir) if self.config.save_plots else {}
            report_path  = Report(run, metrics, figure_paths, meta.output_dir).write(meta.report_path)

            self._stamp(meta, run)

            logger.section("[Inference Pipeline Done]")
            logger.kv_table({
                "RMSE (mean over channels)" : f"{metrics['rmse_mean']:.5g}",
                "R2 (mean over channels)"   : f"{metrics['r2_mean']:.4f}",
                "Report"                    : report_path,
                "Metrics"                   : meta.metrics_path,
            })

            return report_path
        finally:
            logger.close()
