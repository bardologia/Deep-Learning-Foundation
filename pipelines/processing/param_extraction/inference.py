from __future__ import annotations

import gc
from pathlib import Path

from pipelines.processing.param_extraction.metrics  import FittingMetricsCalculator
from pipelines.processing.param_extraction.io       import ParameterIO, ParameterRunMeta
from pipelines.processing.param_extraction.plots    import FittingResultPlotter
from pipelines.shared.orchestration.session_scheduler import SequentialSessionScheduler
from tools.data.io                                  import FileIO
from tools.monitoring.logger                        import Logger


class ParamRunInferencePipeline:

    META_FILENAME    = ParameterRunMeta.FILENAME
    SUMMARY_FILENAME = "fit_metrics_summary.json"

    def __init__(self, run_dir: Path, logger: Logger, make_plots: bool = True) -> None:
        self.run_dir      = Path(run_dir)
        self.logger       = logger
        self.make_plots   = make_plots
        self.parameter_io = ParameterIO(logger=logger)

        self.logger.section("[Param Extraction Inference]")
        self.logger.subsection(f"Run directory : {self.run_dir}")

    def _load(self) -> tuple:
        meta = self.parameter_io.load_metadata(self.run_dir / self.META_FILENAME)

        parameters    = self.parameter_io.load_params(self.run_dir / meta["parameters_npy"])
        diagnostics   = self.parameter_io.load_diagnostics(self.run_dir / meta["diagnostics_npz"])
        tomogram_path = Path(meta["source_tomogram"])

        return meta, parameters, diagnostics, tomogram_path

    def _metrics(self, parameters, meta: dict, diagnostics: dict, tomogram_path: Path) -> dict:
        calculator = FittingMetricsCalculator(
            n_gaussians      = int(meta["k_max"]),
            logger           = self.logger,
            threshold_factor = float(meta["threshold_factor"]),
            truncation_index = int(meta["truncation_index"]),
            amp_threshold    = float(meta["activity_threshold"]),
        )

        return calculator.run(parameters, meta, tomogram_path, diagnostics)

    def _summary(self, metrics_dict: dict) -> Path:
        payload = {
            "global_summary" : metrics_dict["global_summary"],
            "per_k_summary"  : metrics_dict["per_k_summary"],
            "snr_summary"    : metrics_dict["snr_summary"],
        }

        summary_path = self.run_dir / self.SUMMARY_FILENAME
        FileIO.save_json(payload, summary_path)

        self.logger.subsection(f"-> Metrics summary written: {summary_path}")
        return summary_path

    def _plots(self, parameters, meta: dict, metrics_dict: dict, tomogram_path: Path) -> dict[str, Path]:
        if not self.make_plots:
            return {}

        plotter = FittingResultPlotter(
            output_directory = self.run_dir,
            n_gaussians      = int(meta["k_max"]),
            logger           = self.logger,
            threshold_factor = float(meta["threshold_factor"]),
            truncation_index = int(meta["truncation_index"]),
            amp_threshold    = float(meta["activity_threshold"]),
        )

        return plotter.run(parameters, metrics_dict, meta, tomogram_path)

    def run(self) -> dict[str, Path]:
        meta, parameters, diagnostics, tomogram_path = self._load()

        metrics_dict = self._metrics(parameters, meta, diagnostics, tomogram_path)
        summary_path = self._summary(metrics_dict)
        plot_paths   = self._plots(parameters, meta, metrics_dict, tomogram_path)

        del parameters, diagnostics
        gc.collect()

        self.logger.section("[Param Extraction Inference Completed]")

        return {
            "metrics_summary"  : summary_path,
            "output_directory" : self.run_dir,
            "plots"            : plot_paths,
        }


class ParamInferenceTrialCollector:

    def __init__(self, params_dir: Path, run_tags: list[str], logger: Logger) -> None:
        self.params_dir = Path(params_dir)
        self.run_tags   = run_tags
        self.logger     = logger

    def _discover_tags(self) -> list[str]:
        if self.run_tags:
            return list(self.run_tags)

        return [
            str(marker.parent.relative_to(self.params_dir))
            for marker in sorted(self.params_dir.rglob(ParamRunInferencePipeline.META_FILENAME))
            if (marker.parent / "parameters.npy").exists() and not ParameterRunMeta.is_external(marker.parent)
        ]

    def collect(self) -> list[Path]:
        self.logger.section("Collecting parameter-extraction trials")

        run_dirs = []
        for tag in self._discover_tags():
            run_dir = self.params_dir / tag

            if not (run_dir / ParamRunInferencePipeline.META_FILENAME).exists():
                raise FileNotFoundError(f"No {ParamRunInferencePipeline.META_FILENAME} under {run_dir}; cannot run parameter-extraction inference for trial '{tag}'.")

            ParameterRunMeta.reject_external(run_dir, "be scored against the fit it never went through")

            self.logger.info(tag)
            run_dirs.append(run_dir)

        if not run_dirs:
            self.logger.error(f"No parameter-extraction trials found under {self.params_dir}")

        return run_dirs


class ParamInferenceSession:
    def __init__(self, run_dir: Path, make_plots: bool) -> None:
        self.run_dir    = Path(run_dir)
        self.make_plots = make_plots

    def execute(self) -> dict[str, Path]:
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = Logger(log_dir=str(log_dir), name="param_extraction_inference", level="INFO")

        return ParamRunInferencePipeline(self.run_dir, logger=logger, make_plots=self.make_plots).run()


def run_param_inference_session(session: ParamInferenceSession) -> dict[str, Path]:
    return session.execute()


class ParamExtractionInferenceScheduler(SequentialSessionScheduler):
    EMPTY_MESSAGE = "No parameter-extraction trials to infer"
    SESSION_NOUN  = "trials"

    def __init__(self, config, logger: Logger) -> None:
        super().__init__(logger)
        self.config = config

    def _sessions(self) -> list[ParamInferenceSession]:
        run_dirs = ParamInferenceTrialCollector(Path(self.config.params_dir), list(self.config.run_tags), self.logger).collect()
        return [ParamInferenceSession(run_dir, self.config.make_plots) for run_dir in run_dirs]

    def _session_runner(self):
        return run_param_inference_session

    def _result_key(self, session) -> str:
        return session.run_dir.name

    def _completion_message(self, session) -> str:
        return f"[Trial] {session.run_dir.name} completed"

    def _outputs_table(self, outputs: dict) -> dict:
        return {name: str(path) for name, path in outputs.items() if name != "plots"}
