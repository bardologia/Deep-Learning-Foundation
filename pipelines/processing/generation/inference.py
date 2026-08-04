from __future__ import annotations

import gc
from pathlib import Path

from pipelines.processing.generation.distributions    import StackDistributionAnalyzer
from pipelines.processing.generation.plots            import StackPlotter
from pipelines.shared.orchestration.session_scheduler import SequentialSessionScheduler
from tools.data.io                                    import FileIO
from tools.monitoring.logger                          import Logger


class StackInferencePipeline:

    LAYOUT_FILENAME = "dataset.json"

    def __init__(self, run_dir: Path, logger: Logger) -> None:
        self.run_dir  = Path(run_dir)
        self.data_dir = self.run_dir / "data"
        self.logger   = logger

        self.logger.section("[Pre-Processing Inference]")
        self.logger.subsection(f"Run directory : {self.run_dir}")

    def _layout(self) -> dict:
        return FileIO.load_json(self.data_dir / self.LAYOUT_FILENAME)

    def _plot(self, layout: dict) -> dict[str, Path]:
        artifacts = layout["artifacts"]

        plotter = StackPlotter(
            run_directory      = self.run_dir,
            max_amplitude_clip = float(layout["max_amplitude_clip"]),
            logger             = self.logger,
        )

        return plotter.run(
            primary_path        = self.data_dir / artifacts["primary"],
            secondaries_path    = self.data_dir / artifacts["secondaries"],
            interferograms_path = self.data_dir / artifacts["interferograms"],
            dem_path            = self.data_dir / artifacts["dem_full"],
            pass_labels         = layout.get("pass_labels"),
        )

    def _analyze_distributions(self, layout: dict) -> dict[str, Path]:
        artifacts = layout["artifacts"]

        analyzer = StackDistributionAnalyzer(
            run_directory      = self.run_dir,
            max_amplitude_clip = float(layout["max_amplitude_clip"]),
            logger             = self.logger,
        )

        return analyzer.run(
            primary_path        = self.data_dir / artifacts["primary"],
            secondaries_path    = self.data_dir / artifacts["secondaries"],
            interferograms_path = self.data_dir / artifacts["interferograms"],
            dem_path            = self.data_dir / artifacts["dem_full"],
            pass_labels         = layout.get("pass_labels"),
        )

    def run(self) -> dict[str, Path]:
        layout        = self._layout()
        saved         = self._plot(layout)
        distributions = self._analyze_distributions(layout)

        gc.collect()

        self.logger.section("[Pre-Processing Inference Completed]")

        return {
            "images"             : self.run_dir / "images",
            "run_directory"      : self.run_dir,
            "figures"            : len(saved),
            "distributions_json" : distributions["value_distributions_json"],
        }


class StackInferenceTrialCollector:

    def __init__(self, runs_dir: Path, run_tags: list[str], logger: Logger) -> None:
        self.runs_dir = Path(runs_dir)
        self.run_tags = run_tags
        self.logger   = logger

    def _discover_tags(self) -> list[str]:
        if self.run_tags:
            return list(self.run_tags)

        return [
            entry.name
            for entry in sorted(self.runs_dir.iterdir())
            if entry.is_dir() and (entry / "data" / StackInferencePipeline.LAYOUT_FILENAME).exists()
        ]

    def collect(self) -> list[Path]:
        self.logger.section("Collecting preprocessing trials")

        run_dirs = []
        for tag in self._discover_tags():
            run_dir = self.runs_dir / tag

            if not (run_dir / "data" / StackInferencePipeline.LAYOUT_FILENAME).exists():
                raise FileNotFoundError(f"No data/{StackInferencePipeline.LAYOUT_FILENAME} under {run_dir}; cannot run preprocessing inference for trial '{tag}'.")

            self.logger.info(tag)
            run_dirs.append(run_dir)

        if not run_dirs:
            self.logger.error(f"No preprocessing trials found under {self.runs_dir}")

        return run_dirs


class StackInferenceSession:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    def execute(self) -> dict[str, Path]:
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = Logger(log_dir=str(log_dir), name="preprocessing_inference", level="INFO")

        return StackInferencePipeline(self.run_dir, logger=logger).run()


def run_stack_inference_session(session: StackInferenceSession) -> dict[str, Path]:
    return session.execute()


class PreprocessingInferenceScheduler(SequentialSessionScheduler):
    EMPTY_MESSAGE = "No preprocessing trials to infer"
    SESSION_NOUN  = "trials"

    def __init__(self, config, logger: Logger) -> None:
        super().__init__(logger)
        self.config = config

    def _sessions(self) -> list[StackInferenceSession]:
        run_dirs = StackInferenceTrialCollector(Path(self.config.runs_dir), list(self.config.run_tags), self.logger).collect()
        return [StackInferenceSession(run_dir) for run_dir in run_dirs]

    def _session_runner(self):
        return run_stack_inference_session

    def _result_key(self, session) -> str:
        return session.run_dir.name

    def _completion_message(self, session) -> str:
        return f"[Trial] {session.run_dir.name} completed"
