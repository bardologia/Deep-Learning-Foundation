from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.inference.run_classifier import RunClassifier, RunDirectoryWalk
from tools.monitoring.logger                   import Logger
from tools.orchestration.gpu_queue             import GpuJob, GpuJobResult, GpuPoolFile, GpuQueue
from tools.runtime.config_cli                  import ConfigCli
from tools.runtime.run_tag                     import RunTag


class InferenceScheduler:

    def __init__(self, config, entry_script: Path, run_type: str) -> None:
        self.config       = config
        self.entry_script = entry_script
        self.run_type     = run_type
        self.runs_dir     = Path(config.runs_dir)
        self.work_dir     = Path("logs") / "inference" / run_type / RunTag.now()
        self.pool_file    = GpuPoolFile.resolve(config.gpus_file, self.work_dir)

    def _root(self, logger: Logger) -> Path | None:
        if self.runs_dir.is_dir():
            return self.runs_dir

        logger.subsection(f"Runs directory does not exist: {self.runs_dir}")
        return None

    def _candidate_dirs(self, logger: Logger) -> list[Path]:
        root = self._root(logger)
        if root is None:
            return []

        if self.config.run_filter:
            selected = [root / name for name in self.config.run_filter if (root / name).is_dir()]
            missing  = [name for name in self.config.run_filter if not (root / name).is_dir()]
            if missing:
                raise FileNotFoundError(f"No run directory named {missing} found under {root}")
            return sorted(selected)

        return sorted(RunDirectoryWalk.walk(root))

    def _run_dirs(self, logger: Logger) -> list[Path]:
        candidates = self._candidate_dirs(logger)
        matched    = [directory for directory in candidates if RunClassifier.is_type(directory, self.run_type)]

        skipped = len(candidates) - len(matched)
        if skipped:
            logger.subsection(f"Skipping {skipped} run(s) not of type '{self.run_type}'")

        return matched

    def _worker_config(self) -> Path:
        path = self.work_dir / "resolved_config.json"
        return ConfigCli.save_resolved(self.config, path)

    def _jobs(self, run_dirs: list[Path], config_path: Path) -> list[GpuJob]:
        jobs = []
        for run_dir in run_dirs:
            command = [sys.executable, str(self.entry_script), "--worker", "--run-dir", str(run_dir), "--config", str(config_path)]
            jobs.append(GpuJob(name=f"{run_dir.parent.name}/{run_dir.name}", command=command, log_path=self.work_dir / f"{run_dir.parent.name}__{run_dir.name}.out"))
        return jobs

    def _report(self, logger: Logger, results: list[GpuJobResult]) -> None:
        for result in results:
            logger.info(f"{result.name}  :  GPU {result.gpu}  {result.status}  ({result.duration_s / 60:.1f} min)  {result.log_file}")

        failed = [result for result in results if result.status != "DONE"]
        if failed:
            logger.error(f"{len(failed)} of {len(results)} runs failed; see the per-run logs under {self.work_dir}")

    def run(self) -> list[GpuJobResult]:
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
