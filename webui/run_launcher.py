from __future__ import annotations

import threading
from pathlib import Path

from process_manager        import ProcessManager
from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver
from tensorboard_manager    import TensorboardManager
from web_logger             import WebLogger


class RunLauncher:

    def __init__(self, paths: ProjectPaths, logger: WebLogger, resolver: ScriptConfigResolver, processes: ProcessManager, tensorboard: TensorboardManager) -> None:
        self.paths       = paths
        self.logger      = logger
        self.resolver    = resolver
        self.processes   = processes
        self.tensorboard = tensorboard

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

    def _autostart_tensorboard(self, key: str, overrides: dict, interpreter: str) -> None:
        try:
            logdir = self.training_logdir(key, overrides, interpreter)
            if logdir:
                self.tensorboard.ensure(logdir, interpreter)
        except Exception as exc:
            self.logger.error(f"tensorboard autostart failed: {exc}")

    def execute(self, key: str, interpreter: str, overrides: dict, follow_up: str | None, detach: bool, queue: bool) -> dict:
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
