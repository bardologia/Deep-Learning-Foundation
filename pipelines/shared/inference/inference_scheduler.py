
from __future__ import annotations

import sys
from pathlib import Path

from configuration.inference.general           import InferenceConfig
from pipelines.shared.inference.metadata       import CompletedInference, TargetDataset
from pipelines.shared.inference.run_classifier import InferenceLayout, RunClassifier, RunDirectoryWalk
from tools.monitoring.logger                   import Logger
from tools.orchestration.gpu_queue             import GpuJob, GpuJobResult, GpuPoolFile, GpuQueue
from tools.runtime.config_cli                  import ConfigCli
from tools.runtime.run_tag                     import RunTag


class RunSelection:

    def __init__(self, root: Path, runs: list[Path]) -> None:
        self.root = Path(root)
        self.runs = list(runs)

    def relative(self, run_dir: Path) -> str:
        return run_dir.name if run_dir == self.root else run_dir.relative_to(self.root).as_posix()

    def _matches(self, run_dir: Path, name: str) -> bool:
        relative = self.relative(run_dir)
        token    = name.strip("/")

        return relative == token or relative.startswith(f"{token}/") or run_dir.name == token

    def resolve(self, run_filter: list[str]) -> list[Path]:
        if not run_filter:
            return list(self.runs)

        selected = []
        for name in run_filter:
            hits = [run_dir for run_dir in self.runs if self._matches(run_dir, name)]
            if not hits:
                found = ", ".join(self.relative(run_dir) for run_dir in self.runs) or "none"
                raise FileNotFoundError(f"run_filter entry '{name}' names no run under {self.root}; runs found there: {found}")
            selected.extend(hits)

        return list(dict.fromkeys(selected))


class InferenceScheduler:

    def __init__(self, config, entry_script: Path, run_type: str, block) -> None:
        self.config          = config
        self.entry_script    = entry_script
        self.run_type        = run_type
        self.block           = block
        self.split           = block.split
        self.checkpoint_name = block.checkpoint_name
        self.target          = TargetDataset.of(block) if isinstance(block, InferenceConfig) else None
        self.runs_dir        = Path(config.runs_dir).expanduser()
        self.work_dir        = Path("logs") / "inference" / run_type / RunTag.now()
        self.pool_file       = GpuPoolFile.resolve(config.gpus_file, self.work_dir)
        self.skipped         = []

    def _check_block(self) -> None:
        if self.block.output_subdir:
            raise ValueError(f"The inference block sets output_subdir='{self.block.output_subdir}', but the scheduler names every stamp itself (a timestamp under <run>/inference/, one per run found under runs_dir), so that name would be ignored; leave output_subdir unset.")

        if Path(self.block.run_directory) != Path("."):
            raise ValueError(f"The inference block sets run_directory='{self.block.run_directory}', but the scheduler replaces it with every run found under runs_dir, so that path would be ignored; leave run_directory at '.' and point runs_dir at the run (or the tree) to score.")

    def _root(self) -> Path:
        if not self.runs_dir.is_dir():
            raise NotADirectoryError(f"runs_dir does not exist: {self.runs_dir}")

        return self.runs_dir

    def _candidate_dirs(self) -> list[Path]:
        root = self._root()
        runs = [root] if (root / RunDirectoryWalk.MARKER).is_dir() else sorted(RunDirectoryWalk.walk(root))

        return RunSelection(root, runs).resolve(list(self.config.run_filter))

    def _name(self, run_dir: Path) -> str:
        return RunSelection(self.runs_dir, []).relative(run_dir)

    def _classified(self, candidates: list[Path], logger: Logger) -> list[Path]:
        matched  = []
        rejected = []

        for directory in candidates:
            try:
                family = RunClassifier.classify(directory)
            except ValueError as error:
                logger.warning(f"{self._name(directory)} is left out: {error}")
                rejected.append({"Run": self._name(directory), "Family": "unclassifiable", "Reason": "no recognised model config under meta/"})
                continue

            if family == self.run_type:
                matched.append(directory)
            else:
                rejected.append({"Run": self._name(directory), "Family": family, "Reason": f"not of type '{self.run_type}'"})

        if rejected:
            logger.metrics_table(rejected, columns=["Run", "Family", "Reason"], title=f"Skipping {len(rejected)} run(s) not of type '{self.run_type}'")

        return matched

    def _inference_state(self, run_dir: Path, logger: Logger) -> tuple[str, bool]:
        base   = InferenceLayout.base(run_dir, self.run_type) if self.target is None else InferenceLayout.target_base(run_dir, self.run_type, self.target.name)
        stamps = CompletedInference.stamps(base, self.split)

        if self.target is not None:
            stamps = [stamp for stamp in stamps if CompletedInference.target(stamp) == self.target.payload()]

        if stamps:
            recorded = [CompletedInference.checkpoint_name(stamp) for stamp in stamps]

            if self.checkpoint_name in recorded:
                return "done", True

            if None in recorded:
                return "done (checkpoint not recorded)", True

            others = ", ".join(sorted(set(recorded)))
            logger.warning(f"{self._name(run_dir)}: its stamps on split '{self.split}' were scored with {others}, not the requested {self.checkpoint_name}; the run is scored again")

            return f"done (other checkpoint: {others})", False

        incomplete = CompletedInference.incomplete_groups(base, self.split)
        if not incomplete:
            return "missing", False

        group          = sorted(incomplete)[-1]
        present, count = incomplete[group]
        missing        = [index for index in range(count) if index not in present]

        logger.warning(f"{self._name(run_dir)}: the region group {group} on split '{self.split}' stamped regions {present} of {count}; regions {missing} were never scored, so the run is scored again")

        return f"partial ({len(present)}/{count} regions)", False

    def _run_dirs(self, logger: Logger) -> list[Path]:
        candidates = self._candidate_dirs()
        matched    = self._classified(candidates, logger)

        if not matched:
            raise FileNotFoundError(f"No run of type '{self.run_type}' found under {self.runs_dir}: {len(candidates)} candidate director(ies) were walked and all were rejected. Check runs_dir and the entry point family.")

        column = f"Inference ({self.split})" if self.target is None else f"Inference on {self.target.name}"
        rows   = []
        for directory in matched:
            state, done = self._inference_state(directory, logger)
            skip        = done and self.config.skip_completed

            rows.append({"Run": self._name(directory), column: state, "Action": "skip" if skip else "score"})

            if skip:
                self.skipped.append(directory)

        logger.metrics_table(rows, columns=["Run", column, "Action"], title=f"{len(matched)} {self.run_type} run(s) under {self.runs_dir}")

        return [directory for directory in matched if directory not in self.skipped]

    def _worker_config(self) -> Path:
        path = self.work_dir / "resolved_config.json"
        return ConfigCli.save_resolved(self.config, path)

    def _jobs(self, run_dirs: list[Path], config_path: Path) -> list[GpuJob]:
        jobs = []
        for run_dir in run_dirs:
            name    = self._name(run_dir)
            command = [sys.executable, str(self.entry_script), "--worker", "--run-dir", str(run_dir), "--config", str(config_path)]
            jobs.append(GpuJob(name=name, command=command, log_path=self.work_dir / f"{name.replace('/', '__')}.out"))
        return jobs

    def _report(self, logger: Logger, results: list[GpuJobResult]) -> None:
        for result in results:
            logger.info(f"{result.name}  :  GPU {result.gpu}  {result.status}  ({result.duration_s / 60:.1f} min)  {result.log_file}")

        failed = [result for result in results if result.status != "DONE"]
        if failed:
            logger.error(f"{len(failed)} of {len(results)} runs failed; see the per-run logs under {self.work_dir}")

    def run(self) -> list[GpuJobResult]:
        self._check_block()

        with Logger(log_dir=str(self.work_dir), name="inference") as logger:
            logger.section("Inference")

            run_dirs = self._run_dirs(logger)
            scored   = f"'{self.split}'" if self.target is None else f"the target {self.target.name} (same labels, secondary channels and crop)"

            logger.kv_table({
                "Run type"       : self.run_type,
                "Split"          : self.split,
                "Target dataset" : self.target.describe() if self.target is not None else "none (runs scored on their own split)",
                "Checkpoint"     : self.checkpoint_name,
                "Runs to score"  : len(run_dirs),
                "Runs skipped"   : f"{len(self.skipped)} (already scored on {scored} with {self.checkpoint_name} or an unrecorded checkpoint)",
                "GPUs"           : self.config.gpus,
                "GPU pool file"  : str(self.pool_file),
                "Runs dir"       : str(self.runs_dir),
                "Filter"         : self.config.run_filter or "all runs found",
                "Skip completed" : self.config.skip_completed,
                "Workers"        : str(self.work_dir),
            }, title="Configuration")

            if not run_dirs:
                logger.ok(f"every selected run already holds a completed inference on {scored}; nothing to score")
                return []

            config_path = self._worker_config()
            jobs        = self._jobs(run_dirs, config_path)

            queue   = GpuQueue(gpus=self.config.gpus, logger=logger, poll_interval_s=self.config.poll_interval_s, pool_file=self.pool_file)
            results = queue.run(jobs)

            self._report(logger, results)

        return results
