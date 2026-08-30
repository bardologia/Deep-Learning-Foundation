"""GPU-parallel scheduling of inference over every run of one model family.

Walks a runs directory, keeps the runs of the requested type, and dispatches one
worker subprocess per run through the shared GPU queue.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.inference.run_classifier import RunClassifier, RunDirectoryWalk
from tools.monitoring.logger                   import Logger
from tools.orchestration.gpu_queue             import GpuJob, GpuJobResult, GpuPoolFile, GpuQueue
from tools.runtime.config_cli                  import ConfigCli
from tools.runtime.run_tag                     import RunTag


class InferenceScheduler:
    """Fans inference over a runs tree out across the configured GPU pool.

    Attributes:
        config       : Entry-point configuration with runs_dir, run filter, GPU list and polling interval.
        entry_script : Entry script re-invoked in worker mode, once per run.
        run_type     : Model family that runs must classify as to be scheduled.
        runs_dir     : Root of the run tree being walked.
        work_dir     : Timestamped directory holding worker logs and the resolved config.
        pool_file    : File describing the live GPU pool.
    """

    def __init__(self, config, entry_script: Path, run_type: str) -> None:
        """Resolves the working directory and GPU pool file for this scheduling session.

        Args:
            config       : Entry-point configuration for the inference family.
            entry_script : Entry script the workers re-invoke.
            run_type     : Model family to select from the runs tree.
        """
        self.config       = config
        self.entry_script = entry_script
        self.run_type     = run_type
        self.runs_dir     = Path(config.runs_dir)
        self.work_dir     = Path("logs") / "inference" / run_type / RunTag.now()
        self.pool_file    = GpuPoolFile.resolve(config.gpus_file, self.work_dir)

    def _root(self) -> Path:
        """Returns the runs directory.

        Raises:
            NotADirectoryError : The configured runs directory does not exist.
        """
        if not self.runs_dir.is_dir():
            raise NotADirectoryError(f"runs_dir does not exist: {self.runs_dir}")

        return self.runs_dir

    def _candidate_dirs(self) -> list[Path]:
        """Returns the run directories named by the filter, or every walked run.

        Raises:
            FileNotFoundError : A filtered name has no directory under the root.
        """
        root = self._root()

        if self.config.run_filter:
            selected = [root / name for name in self.config.run_filter if (root / name).is_dir()]
            missing  = [name for name in self.config.run_filter if not (root / name).is_dir()]
            if missing:
                raise FileNotFoundError(f"No run directory named {missing} found under {root}")
            return sorted(selected)

        return sorted(RunDirectoryWalk.walk(root))

    def _run_dirs(self, logger: Logger) -> list[Path]:
        """Returns the candidate runs that classify as the requested family.

        Args:
            logger : Logger noting how many candidates were rejected.

        Raises:
            FileNotFoundError : Every candidate was rejected, so nothing to score.
        """
        candidates = self._candidate_dirs()
        matched    = [directory for directory in candidates if RunClassifier.is_type(directory, self.run_type)]

        skipped = len(candidates) - len(matched)
        if skipped:
            logger.subsection(f"Skipping {skipped} run(s) not of type '{self.run_type}'")

        if not matched:
            raise FileNotFoundError(f"No run of type '{self.run_type}' found under {self.runs_dir}: {len(candidates)} candidate director(ies) were walked and all were rejected. Check runs_dir and the entry point family.")

        return matched

    def _worker_config(self) -> Path:
        """Writes the fully resolved configuration the workers read, and returns its path."""
        path = self.work_dir / "resolved_config.json"
        return ConfigCli.save_resolved(self.config, path)

    def _jobs(self, run_dirs: list[Path], config_path: Path) -> list[GpuJob]:
        """Builds one worker job per run directory.

        Args:
            run_dirs    : Run directories to score.
            config_path : Resolved configuration file passed to every worker.

        Returns:
            GPU jobs named by the run path relative to the runs root, each
            writing to its own log file under the work directory.
        """
        jobs = []
        for run_dir in run_dirs:
            name    = run_dir.relative_to(self.runs_dir).as_posix()
            command = [sys.executable, str(self.entry_script), "--worker", "--run-dir", str(run_dir), "--config", str(config_path)]
            jobs.append(GpuJob(name=name, command=command, log_path=self.work_dir / f"{name.replace('/', '__')}.out"))
        return jobs

    def _report(self, logger: Logger, results: list[GpuJobResult]) -> None:
        """Logs one line per finished job and flags the failures.

        Args:
            logger  : Logger the per-job lines are written to.
            results : Finished job results from the GPU queue.
        """
        for result in results:
            logger.info(f"{result.name}  :  GPU {result.gpu}  {result.status}  ({result.duration_s / 60:.1f} min)  {result.log_file}")

        failed = [result for result in results if result.status != "DONE"]
        if failed:
            logger.error(f"{len(failed)} of {len(results)} runs failed; see the per-run logs under {self.work_dir}")

    def run(self) -> list[GpuJobResult]:
        """Selects the runs, dispatches the worker jobs and reports the outcome.

        Returns:
            One result per scheduled run, carrying its GPU, status, duration and
            log file.
        """
        with Logger(log_dir=str(self.work_dir), name="inference") as logger:
            logger.section("Inference")

            run_dirs    = self._run_dirs(logger)
            config_path = self._worker_config()
            jobs        = self._jobs(run_dirs, config_path)

            logger.kv_table({
                "Run type"      : self.run_type,
                "Runs"          : len(run_dirs),
                "GPUs"          : self.config.gpus,
                "GPU pool file" : str(self.pool_file),
                "Runs dir"      : str(self.runs_dir),
                "Filter"        : self.config.run_filter or "all run directories",
                "Workers"       : str(self.work_dir),
            }, title="Configuration")

            queue   = GpuQueue(gpus=self.config.gpus, logger=logger, poll_interval_s=self.config.poll_interval_s, pool_file=self.pool_file)
            results = queue.run(jobs)

            self._report(logger, results)

        return results
