"""Launching of training and analysis entry points from the console.

Resolves the interpreter and the TensorBoard log directory of an entry point,
hands the run to the process manager either immediately or through the queue, and
starts TensorBoard for training runs in the background.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib  import Path

from process_manager        import ProcessManager
from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver
from tensorboard_manager    import TensorboardManager
from web_logger             import WebLogger


class RunLauncher:
    """Validates interpreters, resolves log directories and launches console runs.

    Attributes:
        paths       : Project paths providing entry points and interpreter discovery.
        logger      : Console logger.
        resolver    : Config resolver used to read an entry's default log directory.
        processes   : Process manager executing or queueing the run.
        tensorboard : TensorBoard manager started for training entry points.
    """

    def __init__(self, paths: ProjectPaths, logger: WebLogger, resolver: ScriptConfigResolver, processes: ProcessManager, tensorboard: TensorboardManager) -> None:
        """Stores the paths, logger, resolver, process manager and TensorBoard manager."""
        self.paths       = paths
        self.logger      = logger
        self.resolver    = resolver
        self.processes   = processes
        self.tensorboard = tensorboard

    def preferred_interpreter(self, script_key: str = "") -> str:
        """Returns the interpreter preferred for an entry point among the discovered ones."""
        interpreters = self.paths.discover_interpreters()
        return self.paths.preferred_interpreter(interpreters, script_key)

    def interpreter_error(self, interpreter: str) -> str:
        """Returns an error message when the interpreter is not one of the discovered ones."""
        if any(item["path"] == interpreter for item in self.paths.discover_interpreters()):
            return ""
        return f"unknown interpreter '{interpreter}'; pick one of the environments listed by the console"

    def training_logdir(self, key: str, overrides: dict, interpreter: str) -> str | None:
        """Returns the TensorBoard log directory an entry point will write to.

        Takes the first log-directory config leaf that the launch overrides set, and
        otherwise falls back to the value resolved from the entry's config defaults.

        Args:
            key         : Entry-point key.
            overrides   : Config leaf overrides chosen for this launch.
            interpreter : Interpreter used to resolve the entry's config defaults.

        Returns:
            The log directory, or None when the entry writes no TensorBoard logs or its
            config cannot be resolved.
        """
        leaf_keys = self.tensorboard.logdir_keys(key)
        if not leaf_keys:
            return None

        for leaf in leaf_keys:
            value = (overrides or {}).get(leaf)
            if value:
                return str(value)

        resolved = self.resolver.resolve(key, interpreter)
        if not resolved.get("ok"):
            return None

        leaves = {item["path"]: item["value"] for item in resolved["leaves"]}
        for leaf in leaf_keys:
            if leaves.get(leaf):
                return str(leaves[leaf])

        return None

    def runs_root(self, key: str, interpreter: str) -> str | None:
        """Returns the parent of an entry point's default log directory, or None."""
        logdir = self.training_logdir(key, {}, interpreter)
        if not logdir:
            return None
        return str(Path(logdir).parent)

    def tensorboard_capable(self, key: str) -> bool:
        """Returns whether an entry point writes TensorBoard event files."""
        return self.tensorboard.logdir_keys(key) is not None

    def job_run_spec(self, job_id: str, root: Path) -> dict | None:
        """Returns the run directories of one console job's fan-out units, by label.

        Each unit is resolved to the run directory holding its event files, twin
        units of the same run collapse into one entry, and the labels are the run
        directories' paths relative to the launch's log directory.

        Args:
            job_id : Job whose unit registry is resolved.
            root   : Log directory the launch resolved to.

        Returns:
            Mapping of run label to run directory, or None when the job has no
            units or none of them resolved to a directory with event files.
        """
        run_dirs = []
        for unit in self._job_units(job_id):
            run_dir = self._unit_run_dir(unit, root)
            if run_dir is not None and run_dir not in run_dirs:
                run_dirs.append(run_dir)

        if not run_dirs or run_dirs == [root]:
            return None

        spec = {}
        for run_dir in run_dirs:
            base  = str(run_dir.relative_to(root)) if root in run_dir.parents else run_dir.name
            label = base
            index = 2
            while label in spec:
                label  = f"{base}_{index}"
                index += 1
            spec[label] = str(run_dir)

        return spec

    def _job_units(self, job_id: str) -> list[dict]:
        """Returns the fan-out unit records of a job, empty without a progress snapshot."""
        info = self.processes.progress(job_id)
        if not info.get("ok") or "progress" not in info:
            return []
        return info["progress"].get("units") or []

    def _unit_run_dir(self, unit: dict, root: Path) -> Path | None:
        """Returns the run directory holding one fan-out unit's event files.

        Flat fan-out layouts name each unit after its run, so the directory named
        like the unit under an ancestor of its log is checked first; nested
        layouts put the log inside the run directory itself, so the log's own
        directory is the fallback. Either match must already hold event files.

        Args:
            unit : Unit record carrying the run name and log path.
            root : Log directory of the launch, bounding the ancestor walk.

        Returns:
            The unit's run directory, or None when no directory with event files
            is found, as for a unit that has not started writing yet.
        """
        raw = unit.get("log")
        if not raw:
            return None

        log = Path(raw)
        log = log if log.is_absolute() else self.paths.repo_root / log

        name = unit.get("name") or ""
        for directory in self._bounded_ancestors(log.parent, root):
            candidate = directory / name
            if name and candidate.is_dir() and self.tensorboard.has_event_files(candidate):
                return candidate

        if self.tensorboard.has_event_files(log.parent):
            return log.parent

        return None

    @staticmethod
    def _bounded_ancestors(start: Path, root: Path) -> list[Path]:
        """Returns `start` and its ancestors up to and including `root`.

        Args:
            start : Directory the walk begins at.
            root  : Directory the walk stops at; a start outside it yields only itself.
        """
        if root not in start.parents and start != root:
            return [start]

        chain = [start]
        while chain[-1] != root:
            chain.append(chain[-1].parent)
        return chain

    def _pid_run_dir(self, record: dict, root: Path) -> Path | None:
        """Returns the run directory of an in-process single run, found by writer pid.

        A single training run writes its event files from the job's own process,
        and event file names carry the writer's start timestamp and pid, so the
        run directory is the one holding event files stamped with the job's pid
        since shortly before its start.

        Args:
            record : Job record carrying the pid and start timestamp.
            root   : Log directory the launch resolved to.

        Returns:
            The run directory, or None when the job has no pid, nothing under the
            root matches, or the match is not a single directory below the root.
        """
        pid = record.get("pid")
        if not pid:
            return None

        since    = datetime.fromisoformat(record["started"]).timestamp() - 60.0
        run_dirs = [run_dir for run_dir in self.tensorboard.run_dirs_for_pid(root, pid, since) if run_dir != root]

        return run_dirs[0] if len(run_dirs) == 1 else None

    def job_tensorboard(self, record: dict) -> dict:
        """Starts, or reuses, a TensorBoard instance scoped to one console job's runs.

        A fan-out job whose units resolved to run directories is served exactly
        those directories, an in-process single run is served the run directory
        its pid wrote event files into, and any other job is served the log
        directory its launch resolved to.

        Args:
            record : Job record carrying the script key, overrides, interpreter, job id, pid and start timestamp.

        Returns:
            The TensorBoard manager's ensure result, or a failed result when the
            entry writes no event files or its log directory cannot be resolved.
        """
        if not self.tensorboard_capable(record["script"]):
            return {"ok": False, "error": f"{record['script']} writes no TensorBoard event files"}

        root = self.training_logdir(record["script"], record["overrides"], record["interpreter"])
        if not root:
            return {"ok": False, "error": "could not resolve the job's log directory"}

        spec = self.job_run_spec(record["job_id"], Path(root))
        if spec:
            return self.tensorboard.ensure_spec(spec, record["interpreter"])

        run_dir = self._pid_run_dir(record, Path(root))
        if run_dir is not None:
            return self.tensorboard.ensure(str(run_dir), record["interpreter"])

        return self.tensorboard.ensure(root, record["interpreter"])

    def _autostart_tensorboard(self, key: str, overrides: dict, interpreter: str) -> None:
        """Ensures a TensorBoard instance for the launched run, logging any failure."""
        try:
            logdir = self.training_logdir(key, overrides, interpreter)
            if logdir:
                self.tensorboard.ensure(logdir, interpreter)
        except Exception as exc:
            self.logger.error(f"tensorboard autostart failed: {exc}")

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up: str | None, detach: bool, queue: bool) -> dict:
        """Launches or queues an entry point and starts TensorBoard for training runs.

        Args:
            key         : Entry-point key to run.
            interpreter : Python interpreter to run it under.
            overrides   : Config leaf overrides passed on the command line.
            follow_up   : Entry point to run once this one succeeds, or None.
            detach      : Whether the run survives the console process.
            queue       : Whether to enqueue the run instead of starting it immediately.

        Returns:
            The process manager's launch result, or a failed result when the interpreter
            is unknown.
        """
        error = self.interpreter_error(interpreter)
        if error:
            return {"ok": False, "error": error}

        if queue:
            result = self.processes.enqueue(key, interpreter, overrides, follow_up, detach)
        else:
            result = self.processes.launch(key, interpreter, overrides, follow_up, detach)

        if result.get("ok") and self.tensorboard.logdir_keys(key):
            threading.Thread(target=self._autostart_tensorboard, args=(key, overrides, interpreter), daemon=True).start()

        return result
