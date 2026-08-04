from __future__ import annotations

from pathlib import Path
from typing  import Dict, List

import numpy as np

from configuration.inference.unrolled       import UnrolledInferenceConfig
from pipelines.shared.inference.metadata    import InferenceMetadata
from pipelines.unrolled.inference.loader    import UnrolledRun, UnrolledRunLoader
from pipelines.unrolled.inference.metrics   import UnrolledMetrics
from pipelines.unrolled.inference.plots     import UnrolledPlots
from pipelines.unrolled.inference.predictor import UnrolledPrediction, UnrolledPredictor
from pipelines.unrolled.inference.report    import UnrolledReport
from tools.data.io                          import FileIO
from tools.monitoring.logger                import Logger
from tools.reporting.plotting               import PlotBase
from tools.runtime.reproducibility          import Reproducibility


class UnrolledInferenceMetadata(InferenceMetadata):
    SUBDIR = "unrolled"


class UnrolledInferencePipeline:
    def __init__(self, config: UnrolledInferenceConfig) -> None:
        self.config = config

        Reproducibility.seed_everything(config.seed)

    def _setup(self) -> tuple[UnrolledInferenceMetadata, Logger]:
        meta = UnrolledInferenceMetadata(self.config)
        meta.create_dirs()

        logger = Logger(log_dir=str(meta.logs_dir), name="unrolled_inference", level=self.config.log_level)
        logger.section("[Unrolled Inference Pipeline]")
        logger.kv_table({
            "Run Directory" : self.config.run_directory,
            "Output Dir"    : meta.output_dir,
            "Split"         : self.config.split,
            "Device"        : self.config.device,
        })

        return meta, logger

    def _load_run(self, logger: Logger) -> UnrolledRun:
        loader = UnrolledRunLoader(self.config.run_directory, logger=logger)

        return loader.load(config=self.config, device=self.config.device)

    def _persist_profile_cube(self, prediction: UnrolledPrediction, meta: UnrolledInferenceMetadata, logger: Logger) -> None:
        if prediction.profile_cube is None:
            return

        path = meta.output_dir / "profile_cube.npy"
        np.save(path, prediction.profile_cube)
        logger.subsection(f"Profile cube saved : {path}  shape {prediction.profile_cube.shape}")

    def _evaluate_metrics(self, prediction: UnrolledPrediction, run: UnrolledRun, meta: UnrolledInferenceMetadata, logger: Logger) -> dict:
        metrics = UnrolledMetrics(prediction, run).compute()

        FileIO.save_json(metrics, meta.metrics_path)

        logger.section("[Unrolled Inference: Metrics]")
        logger.kv_table({
            "Loss"           : f"{metrics['loss']:.6g}",
            "Curve L1"       : f"{metrics['curve_l1']:.6g}",
            "Curve RMSE"     : f"{metrics['curve_rmse']:.6g}",
            "Peak MAE (m)"   : f"{metrics['peak_mae_m']:.4g}",
            "Valid fraction" : f"{metrics['valid_fraction']:.4f}",
        })

        return metrics

    def _select_examples(self, prediction: UnrolledPrediction, predictor: UnrolledPredictor) -> Dict[str, List[dict]]:
        if not self.config.save_plots or self.config.n_example_profiles < 1:
            return {}

        pixel_indices = np.argwhere(prediction.valid_mask)
        errors        = prediction.curve_l1_map[prediction.valid_mask]
        order         = np.argsort(errors)

        n            = min(self.config.n_example_profiles, order.size)
        median_start = max(0, (order.size - n) // 2)

        picks = {
            "best"   : order[:n],
            "median" : order[median_start:median_start + n],
            "worst"  : order[::-1][:n],
        }

        return {label: [predictor.profile_pair(*pixel_indices[index]) for index in indices] for label, indices in picks.items()}

    def _compose_figures(self, prediction: UnrolledPrediction, examples: Dict[str, List[dict]], run: UnrolledRun, meta: UnrolledInferenceMetadata, logger: Logger) -> dict:
        if not self.config.save_plots:
            logger.section("[Unrolled Inference: Plots skipped]")
            return {}

        logger.section("[Unrolled Inference: Plots]")
        PlotBase.use_style(self.config.figure_style)
        plots = UnrolledPlots(fig_dpi=self.config.fig_dpi, save_dpi=self.config.save_dpi)

        return plots.compose(prediction, examples, run.x_axis, meta.figures_dir)

    def _build_report(self, meta: UnrolledInferenceMetadata, run: UnrolledRun, metrics: dict, figures: dict) -> Path:
        return UnrolledReport(
            output_dir  = meta.output_dir,
            run         = run,
            config      = self.config,
            metrics     = metrics,
            figures     = figures,
            report_path = meta.report_path,
        ).assemble()

    def run(self) -> Path:
        meta, logger = self._setup()

        run        = self._load_run(logger)
        predictor  = UnrolledPredictor(run, self.config, logger)
        prediction = predictor.run_inference()

        self._persist_profile_cube(prediction, meta, logger)

        metrics  = self._evaluate_metrics(prediction, run, meta, logger)
        examples = self._select_examples(prediction, predictor)
        figures  = self._compose_figures(prediction, examples, run, meta, logger)

        logger.section("[Unrolled Inference: Report]")
        report_path = self._build_report(meta, run, metrics, figures)

        logger.section("[Unrolled Inference Done]")
        logger.subsection(f"Report  : {report_path}")
        logger.subsection(f"Metrics : {meta.metrics_path}\n")
        logger.close()

        return report_path
