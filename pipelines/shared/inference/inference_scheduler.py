"""GPU-parallel scheduling of inference over every run of one model family.

Walks whatever path it is given, keeps the runs of the requested family, leaves
alone the ones already scored on the configured split with the requested
checkpoint, and dispatches one worker subprocess per remaining run through the
shared GPU queue.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipelines.shared.inference.metadata       import CompletedInference
from pipelines.shared.inference.run_classifier import InferenceLayout, RunClassifier, RunDirectoryWalk
from tools.monitoring.logger                   import Logger
from tools.orchestration.gpu_queue             import GpuJob, GpuJobResult, GpuPoolFile, GpuQueue
from tools.runtime.config_cli                  import ConfigCli
from tools.runtime.run_tag                     import RunTag


class RunSelection:
    """Resolves the runs a filter names among the runs found under a root.

    A filter entry names a run by its path relative to the root, a directory
    below the root whose runs are all taken, or a run's own directory name.

    Attributes:
        root : Directory the run paths are relative to.
        runs : Every run found under the root, in walk order.
    """

    def __init__(self, root: Path, runs: list[Path]) -> None:
        """Binds the selection to the root and the runs found under it."""
        self.root = Path(root)
        self.runs = list(runs)

    def relative(self, run_dir: Path) -> str:
        """Returns a run's path relative to the root, its own name when it is the root."""
        return run_dir.name if run_dir == self.root else run_dir.relative_to(self.root).as_posix()

    def _matches(self, run_dir: Path, name: str) -> bool:
        """Returns whether one filter entry names a run."""
        relative = self.relative(run_dir)
        token    = name.strip("/")

        return relative == token or relative.startswith(f"{token}/") or run_dir.name == token

    def resolve(self, run_filter: list[str]) -> list[Path]:
        """Returns the runs the filter names, every run for an empty filter.

        Args:
            run_filter : Filter entries as described on the class.

        Raises:
            FileNotFoundError : If an entry names no run under the root.
        """
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
    """Fans inference over a runs tree out across the configured GPU pool.

    Attributes:
        config          : Entry-point configuration with runs_dir, run filter, GPU list, polling interval and the skip switch.
        entry_script    : Entry script re-invoked in worker mode, once per run.
        run_type        : Model family that runs must classify as to be scheduled.
        block           : Family inference block of the entry configuration, carrying the split and the checkpoint the workers score.
        split           : Dataset split the family block scores, deciding which runs already hold their inference.
        checkpoint_name : Checkpoint file the family block scores; a run whose stamps record only other checkpoints is scored again.
        runs_dir        : Path being walked: a run, a sweep, an experiment or a whole runs tree.
        work_dir        : Timestamped directory holding worker logs and the resolved config.
        pool_file       : File describing the live GPU pool.
        skipped         : Runs left alone because they already hold a completed inference on the split, filled by `run`.
    """

    def __init__(self, config, entry_script: Path, run_type: str, block) -> None:
        """Resolves the working directory and GPU pool file for this scheduling session.

        Args:
            config       : Entry-point configuration for the inference family.
            entry_script : Entry script the workers re-invoke.
            run_type     : Model family to select from the runs tree.
            block        : Family inference block of the configuration, naming the split and checkpoint the workers score.
        """
        self.config          = config
        self.entry_script    = entry_script
        self.run_type        = run_type
        self.block           = block
        self.split           = block.split
        self.checkpoint_name = block.checkpoint_name
        self.runs_dir        = Path(config.runs_dir).expanduser()
        self.work_dir        = Path("logs") / "inference" / run_type / RunTag.now()
        self.pool_file       = GpuPoolFile.resolve(config.gpus_file, self.work_dir)
        self.skipped         = []

    def _check_block(self) -> None:
        """Refuses a family block whose output_subdir or run_directory is set, since the per-run fan-out overwrites both.

        Raises:
            ValueError : If output_subdir is set, or run_directory differs from the "." placeholder the scheduler replaces with every run it finds.
        """
        if self.block.output_subdir:
            raise ValueError(f"The inference block sets output_subdir='{self.block.output_subdir}', but the scheduler names every stamp itself (a timestamp under <run>/inference/, one per run found under runs_dir), so that name would be ignored; leave output_subdir unset.")

        if Path(self.block.run_directory) != Path("."):
            raise ValueError(f"The inference block sets run_directory='{self.block.run_directory}', but the scheduler replaces it with every run found under runs_dir, so that path would be ignored; leave run_directory at '.' and point runs_dir at the run (or the tree) to score.")

    def _root(self) -> Path:
        """Returns the runs directory.

        Raises:
            NotADirectoryError : The configured runs directory does not exist.
        """
        if not self.runs_dir.is_dir():
            raise NotADirectoryError(f"runs_dir does not exist: {self.runs_dir}")

        return self.runs_dir

    def _candidate_dirs(self) -> list[Path]:
        """Returns the runs the filter names among every run at or below the root.

        The root itself counts as a run when it holds a meta directory, so one
        run directory can be given directly.

        Raises:
            FileNotFoundError : A filter entry names no run under the root.
        """
        root = self._root()
        runs = [root] if (root / RunDirectoryWalk.MARKER).is_dir() else sorted(RunDirectoryWalk.walk(root))

        return RunSelection(root, runs).resolve(list(self.config.run_filter))

    def _name(self, run_dir: Path) -> str:
        """Returns the job name of a run: its path relative to the root, its own name when it is the root."""
        return RunSelection(self.runs_dir, []).relative(run_dir)

    def _classified(self, candidates: list[Path], logger: Logger) -> list[Path]:
        """Returns the candidates of the requested family, listing every other candidate with its family or the reason it cannot be classified.

        A candidate without a recognised model config (an interrupted run, a
        preprocessing or analysis directory holding meta/) is warned about with the
        classifier's message, so it is not mistaken for a run of another family.

        Args:
            candidates : Run directories selected by the filter.
            logger     : Logger receiving the rejection table and the unclassifiable-run warnings.
        """
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
        """Returns how far a run is scored on the configured split and whether it counts as scored.

        A run counts as scored when one of its completed stamps on the split records
        the requested checkpoint. Stamps that record only other checkpoints are warned
        about and the run is scored again; stamps whose marker records no checkpoint
        (autoencoder stamps, older backbone stamps) still count as scored
        and say so in the table. A split scored region by region whose launch stamped
        only some of its regions reads as partial, is warned about with the missing
        region indices and is scored again.

        Args:
            run_dir : Run directory whose inference stamps are inspected.
            logger  : Logger receiving the other-checkpoint and partial-group warnings.

        Returns:
            Tuple of the state shown in the run table ("done", "done (checkpoint not recorded)", "done (other checkpoint: X)", "partial (k/N regions)" or "missing") and whether the run already holds a completed inference with the requested checkpoint.
        """
        base   = InferenceLayout.base(run_dir, self.run_type)
        stamps = CompletedInference.stamps(base, self.split)

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
        """Returns the candidate runs of the requested family that still need scoring.

        Args:
            logger : Logger receiving the rejection table, the run table and the warnings.

        Raises:
            FileNotFoundError : No candidate classifies as the requested family.
        """
        candidates = self._candidate_dirs()
        matched    = self._classified(candidates, logger)

        if not matched:
            raise FileNotFoundError(f"No run of type '{self.run_type}' found under {self.runs_dir}: {len(candidates)} candidate director(ies) were walked and all were rejected. Check runs_dir and the entry point family.")

        column = f"Inference ({self.split})"
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
            name    = self._name(run_dir)
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
        """Checks the family block, selects the runs, dispatches the worker jobs and reports the outcome.

        Returns:
            One result per scored run, carrying its GPU, status, duration and
            log file; empty when every selected run already held its inference.

        Raises:
            ValueError : If the family block sets output_subdir or a run_directory, which the per-run fan-out would silently overwrite.
        """
        self._check_block()

        with Logger(log_dir=str(self.work_dir), name="inference") as logger:
            logger.section("Inference")

            run_dirs = self._run_dirs(logger)

            logger.kv_table({
                "Run type"       : self.run_type,
                "Split"          : self.split,
                "Checkpoint"     : self.checkpoint_name,
                "Runs to score"  : len(run_dirs),
                "Runs skipped"   : f"{len(self.skipped)} (already scored on '{self.split}' with {self.checkpoint_name} or an unrecorded checkpoint)",
                "GPUs"           : self.config.gpus,
                "GPU pool file"  : str(self.pool_file),
                "Runs dir"       : str(self.runs_dir),
                "Filter"         : self.config.run_filter or "all runs found",
                "Skip completed" : self.config.skip_completed,
                "Workers"        : str(self.work_dir),
            }, title="Configuration")

            if not run_dirs:
                logger.ok("every selected run already holds a completed inference on this split; nothing to score")
                return []

            config_path = self._worker_config()
            jobs        = self._jobs(run_dirs, config_path)

            queue   = GpuQueue(gpus=self.config.gpus, logger=logger, poll_interval_s=self.config.poll_interval_s, pool_file=self.pool_file)
            results = queue.run(jobs)

            self._report(logger, results)

        return results
