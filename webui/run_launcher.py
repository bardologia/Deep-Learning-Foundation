
from __future__ import annotations

import threading
from datetime import datetime
from pathlib  import Path

from finalize_chain         import FinalizeChain, RunFinalizer
from process_manager        import ProcessManager
from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver
from tensorboard_manager    import TensorboardManager
from web_logger             import WebLogger


class RunLauncher:


    def __init__(self, paths: ProjectPaths, logger: WebLogger, resolver: ScriptConfigResolver, processes: ProcessManager, tensorboard: TensorboardManager, notifier) -> None:
        self.paths       = paths
        self.logger      = logger
        self.resolver    = resolver
        self.processes   = processes
        self.tensorboard = tensorboard
        self.finalizer   = RunFinalizer(self, processes, resolver, notifier, logger)

    def start(self) -> None:
        self.finalizer.start()

    def preferred_interpreter(self, script_key: str = "") -> str:
        interpreters = self.paths.discover_interpreters()
        return self.paths.preferred_interpreter(interpreters, script_key)

    def interpreter_error(self, interpreter: str) -> str:
        if any(item["path"] == interpreter for item in self.paths.discover_interpreters()):
            return ""
        return f"unknown interpreter '{interpreter}'; pick one of the environments listed by the console"

    def training_logdir(self, key: str, overrides: dict, interpreter: str) -> str | None:
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
        logdir = self.training_logdir(key, {}, interpreter)
        if not logdir:
            return None
        return str(Path(logdir).parent)

    def tensorboard_capable(self, key: str) -> bool:
        return self.tensorboard.logdir_keys(key) is not None

    def _job_units(self, job_id: str) -> list[dict]:
        info = self.processes.progress(job_id)
        if not info.get("ok") or "progress" not in info:
            return []
        return info["progress"]["units"]

    def job_run_spec(self, units: list, root: Path) -> dict | None:
        run_dirs = []
        for unit in units:
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

    def _unit_run_dir(self, unit: dict, root: Path) -> Path | None:
        log = self._unit_log(unit)
        if log is None:
            return None

        name = unit.get("name") or ""
        for directory in self._bounded_ancestors(log.parent, root):
            candidate = directory / name
            if name and candidate.is_dir() and self.tensorboard.has_event_files(candidate):
                return candidate

        if self.tensorboard.has_event_files(log.parent):
            return log.parent

        return None

    def _unit_log(self, unit: dict) -> Path | None:
        raw = unit.get("log")
        if not raw:
            return None

        log = Path(raw)
        return log if log.is_absolute() else self.paths.repo_root / log

    @staticmethod
    def _bounded_ancestors(start: Path, root: Path) -> list[Path]:
        if root not in start.parents and start != root:
            return [start]

        chain = [start]
        while chain[-1] != root:
            chain.append(chain[-1].parent)
        return chain

    def _pid_run_dir(self, record: dict, root: Path) -> Path | None:
        pid = record.get("pid")
        if not pid:
            return None

        since    = datetime.fromisoformat(record["started"]).timestamp() - 60.0
        run_dirs = [run_dir for run_dir in self.tensorboard.run_dirs_for_pid(root, pid, since) if run_dir != root]

        return run_dirs[0] if len(run_dirs) == 1 else None

    def job_tensorboard(self, record: dict) -> dict:
        if not self.tensorboard_capable(record["script"]):
            return {"ok": False, "error": f"{record['script']} writes no TensorBoard event files"}

        root = self.training_logdir(record["script"], record["overrides"], record["interpreter"])
        if not root:
            return {"ok": False, "error": "could not resolve the job's log directory"}

        units = self._job_units(record["job_id"])
        spec  = self.job_run_spec(units, Path(root))
        if spec:
            return self.tensorboard.ensure_spec(spec, record["interpreter"])

        run_dir = self._pid_run_dir(record, Path(root))
        if run_dir is not None:
            return self.tensorboard.ensure(str(run_dir), record["interpreter"])

        return self.tensorboard.ensure(root, record["interpreter"])

    def _autostart_tensorboard(self, key: str, overrides: dict, interpreter: str) -> None:
        try:
            logdir = self.training_logdir(key, overrides, interpreter)
            if logdir:
                self.tensorboard.ensure(logdir, interpreter)
        except Exception as exc:
            self.logger.error(f"tensorboard autostart failed: {exc}")

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up: str | None, detach: bool, queue: bool, finalize: dict | None = None) -> dict:
        error = self.interpreter_error(interpreter)
        if error:
            return {"ok": False, "error": error}

        plan, error = self.finalizer.plan(key, interpreter, overrides, finalize)
        if error:
            return {"ok": False, "error": error}

        if queue:
            result = self.processes.enqueue(key, interpreter, overrides, follow_up, detach)
        else:
            result = self.processes.launch(key, interpreter, overrides, follow_up, detach)

        if not result.get("ok"):
            return result

        self.processes.attach_finalize(result["job_id"], FinalizeChain.flags(finalize))

        if plan is not None:
            self.finalizer.watch(result["job_id"], plan)

        if self.tensorboard.logdir_keys(key):
            threading.Thread(target=self._autostart_tensorboard, args=(key, overrides, interpreter), daemon=True).start()

        return result
