"""Inference stage over fitted Gaussian parameter runs: metrics and plots.

Scores a parameter-extraction run against the cube it was fitted to, analyses the
writes the summary JSON
and figures. External parameter injections carry no fit diagnostics and are rejected.
"""

from __future__ import annotations

import gc
from pathlib import Path

from pipelines.processing.param_extraction.metrics    import FittingMetricsCalculator
from pipelines.processing.param_extraction.io         import ParameterIO, ParameterRunMeta
from pipelines.processing.param_extraction.plots      import FittingResultPlotter
from pipelines.shared.orchestration.session_scheduler import SequentialSessionScheduler
from tools.data.io                                    import FileIO
from tools.monitoring.logger                          import Logger


class ParamRunInferencePipeline:
    """Scores and plots one fitted parameter-extraction run.

    Attributes:
        run_dir         : Parameter run directory holding the fitted arrays and metadata.
        logger          : Logger for the stage report.
        make_plots      : Whether the figure stages run.
        parameter_io    : Reader for the parameter, diagnostics and metadata files.
    """

    META_FILENAME    = ParameterRunMeta.FILENAME
    SUMMARY_FILENAME = "fit_metrics_summary.json"

    def __init__(self, run_dir: Path, logger: Logger, make_plots: bool = True) -> None:
        """Configures the inference pipeline for one parameter run directory.

        Args:
            run_dir         : Parameter run directory produced by ``extract_params``.
            logger          : Logger for the stage report.
            make_plots      : Whether the figure stages run.
        """
        self.run_dir         = Path(run_dir)
        self.logger          = logger
        self.make_plots      = make_plots
        self.parameter_io    = ParameterIO(logger=logger)

        self.logger.section("[Param Extraction Inference]")
        self.logger.subsection(f"Run directory : {self.run_dir}")

    def _load(self) -> tuple:
        """Loads the run metadata, parameter stack, fit diagnostics and cube path.

        Returns:
            Tuple of the metadata dictionary, the parameter stack of shape
            (3 * k_max, rows, cols), the diagnostics dictionary and the source
            cube path.
        """
        meta = self.parameter_io.load_metadata(self.run_dir / self.META_FILENAME)

        parameters  = self.parameter_io.load_params(self.run_dir / meta["parameters_npy"])
        diagnostics = self.parameter_io.load_diagnostics(self.run_dir / meta["diagnostics_npz"])
        cube_path   = Path(meta["source_cube"])

        return meta, parameters, diagnostics, cube_path

    def _metrics(self, parameters, meta: dict, diagnostics: dict, cube_path: Path) -> dict:
        """Computes the fit-quality maps and summaries for the run.

        Args:
            parameters    : Parameter stack of shape (3 * k_max, row, col).
            meta          : Run metadata carrying the fit settings and z col.
            diagnostics   : Per-K MSE and model-order selection arrays from the fitter.
            cube_path : Path of the cube the fit was performed on.

        Returns:
            Dictionary of metric maps and summary dictionaries.
        """
        calculator = FittingMetricsCalculator(
            n_gaussians      = int(meta["k_max"]),
            logger           = self.logger,
            threshold_factor = float(meta["threshold_factor"]),
            truncation_index = int(meta["truncation_index"]),
            amp_threshold    = float(meta["activity_threshold"]),
        )

        return calculator.run(parameters, meta, cube_path, diagnostics)

    def _summary(self, metrics_dict: dict) -> Path:
        """Writes the metrics summary JSON.

        Args:
            metrics_dict : Metric maps and summaries from the metrics stage.

        Returns:
            Path of the written summary JSON.
        """
        payload = {
            "global_summary" : metrics_dict["global_summary"],
            "per_k_summary"  : metrics_dict["per_k_summary"],
            "snr_summary"    : metrics_dict["snr_summary"],
        }

        summary_path = self.run_dir / self.SUMMARY_FILENAME
        FileIO.save_json(payload, summary_path)

        self.logger.subsection(f"-> Metrics summary written: {summary_path}")
        return summary_path

    def _plots(self, parameters, meta: dict, metrics_dict: dict, cube_path: Path) -> dict[str, Path]:
        """Renders the fit-result figures unless plotting is disabled.

        Args:
            parameters    : Parameter stack of shape (3 * k_max, row, col).
            meta          : Run metadata carrying the fit settings.
            metrics_dict  : Metric maps and summaries from the metrics stage.
            cube_path : Path of the cube the fit was performed on.

        Returns:
            Mapping from figure name to the saved PNG path; empty when plotting is off.
        """
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

        return plotter.run(parameters, metrics_dict, meta, cube_path)

    def run(self) -> dict[str, Path]:
        """Runs the metrics, summary and plot stages for the run.

        Returns:
            Dictionary with the summary JSON path, the output directory and the
            mapping of saved figure paths.
        """
        meta, parameters, diagnostics, cube_path = self._load()

        metrics_dict = self._metrics(parameters, meta, diagnostics, cube_path)
        summary_path = self._summary(metrics_dict)
        plot_paths   = self._plots(parameters, meta, metrics_dict, cube_path)

        del parameters, diagnostics
        gc.collect()

        self.logger.section("[Param Extraction Inference Completed]")

        return {
            "metrics_summary"  : summary_path,
            "output_directory" : self.run_dir,
            "plots"            : plot_paths,
        }


class ParamInferenceTrialCollector:
    """Resolves the parameter-extraction runs that inference should visit.

    Attributes:
        params_dir : Directory holding the parameter runs, possibly nested.
        run_tags   : Explicit run paths relative to ``params_dir``; empty means discover.
        logger     : Logger used to report the collected runs.
    """

    def __init__(self, params_dir: Path, run_tags: list[str], logger: Logger) -> None:
        """Binds the collector to a params directory and an optional tag filter.

        Args:
            params_dir : Directory holding the parameter runs.
            run_tags   : Run paths relative to ``params_dir``; empty enables discovery.
            logger     : Logger for the collection report.
        """
        self.params_dir = Path(params_dir)
        self.run_tags   = run_tags
        self.logger     = logger

    def _discover_tags(self) -> list[str]:
        """Returns the configured tags, or every fitted (non-external) run found below."""
        if self.run_tags:
            return list(self.run_tags)

        return [
            str(marker.parent.relative_to(self.params_dir))
            for marker in sorted(self.params_dir.rglob(ParamRunInferencePipeline.META_FILENAME))
            if (marker.parent / "parameters.npy").exists() and not ParameterRunMeta.is_external(marker.parent)
        ]

    def collect(self) -> list[Path]:
        """Returns the run directories of every requested parameter-extraction trial.

        Returns:
            List of run directories, in discovery order.

        Raises:
            FileNotFoundError : If a requested tag has no run metadata on disk.
            ValueError        : If a requested run was injected from external parameters.
        """
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
    """One schedulable unit of parameter-extraction inference over a single run.

    Attributes:
        run_dir    : Parameter run directory this session processes.
        make_plots : Whether the figure stages run.
    """

    def __init__(self, run_dir: Path, make_plots: bool) -> None:
        """Binds the session to a parameter run.

        Args:
            run_dir    : Parameter run directory to process.
            make_plots : Whether the figure stages run.
        """
        self.run_dir    = Path(run_dir)
        self.make_plots = make_plots

    def execute(self) -> dict[str, Path]:
        """Creates the session logger and runs the parameter inference pipeline.

        The run's own log closes with the exception, its traceback and an
        ABORTED banner when a stage raises, so the cause stays beside the run.

        Returns:
            The result dictionary returned by ``ParamRunInferencePipeline.run``.
        """
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with Logger(log_dir=str(log_dir), name="param_extraction_inference", level="INFO") as logger:
            return ParamRunInferencePipeline(self.run_dir, logger=logger, make_plots=self.make_plots).run()


def run_param_inference_session(session: ParamInferenceSession) -> dict[str, Path]:
    """Module-level entry point so a session can be dispatched by a scheduler."""
    return session.execute()


class ParamExtractionInferenceScheduler(SequentialSessionScheduler):
    """Runs parameter-extraction inference sequentially over every collected run.

    Attributes:
        config : Entry configuration carrying the params directory and analysis settings.
    """

    EMPTY_MESSAGE = "No parameter-extraction trials to infer"
    SESSION_NOUN  = "trials"

    def __init__(self, config, logger: Logger) -> None:
        """Binds the scheduler to the parameter-inference entry configuration.

        Args:
            logger : Logger shared with the base scheduler.
        """
        super().__init__(logger)
        self.config = config

    def _sessions(self) -> list[ParamInferenceSession]:
        """Returns one session per collected parameter-extraction run."""
        run_dirs = ParamInferenceTrialCollector(Path(self.config.params_dir), list(self.config.run_tags), self.logger).collect()

        return [ParamInferenceSession(run_dir, self.config.make_plots) for run_dir in run_dirs]

    def _session_runner(self):
        """Returns the callable that executes one session."""
        return run_param_inference_session

    def _result_key(self, session) -> str:
        """Returns the run name used to key this session's results."""
        return session.run_dir.name

    def _completion_message(self, session) -> str:
        """Returns the log line announcing that a trial finished."""
        return f"[Trial] {session.run_dir.name} completed"

    def _outputs_table(self, outputs: dict) -> dict:
        """Returns the session outputs as a printable table, dropping the plot mapping."""
        return {name: str(path) for name, path in outputs.items() if name != "plots"}
