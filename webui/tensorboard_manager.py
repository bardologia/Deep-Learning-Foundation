
from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib  import Path

from project_paths import ProjectPaths
from web_logger    import WebLogger


class TensorboardManager:


    TRAINING_LOGDIRS = {
        "train_backbone"            : ("logdir",),
        "benchmark"                 : ("paths.log_base_dir",),
        "cross_validate"            : ("paths.log_base_dir",),
        "tune"                      : ("paths.log_base_dir",),
    }

    RUN_EVENT_FOLDER  = "tensorboard"
    STARTUP_TIMEOUT_S = 90.0
    PROBE_INTERVAL_S  = 0.5
    STOP_GRACE_S      = 10.0

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:

        self.paths     = paths
        self.logger    = logger
        self.instances = {}
        self.lock      = threading.Lock()

    def logdir_keys(self, script_key: str) -> tuple | None:

        return self.TRAINING_LOGDIRS.get(script_key)

    def _alive(self, record: dict) -> bool:

        if record["status"] not in ("starting", "running"):
            return False
        process = record["process"]
        return process is None or process.poll() is None

    def ensure(self, logdir: str, interpreter: str) -> dict:

        return self._ensure(str(logdir), ["--logdir", str(logdir)], interpreter)

    def ensure_spec(self, spec: dict, interpreter: str) -> dict:

        broken = [name for name in spec if ":" in name or "," in name] + [path for path in spec.values() if "," in str(path)]
        if broken:
            return {"ok": False, "error": f"cannot serve via --logdir_spec, ':' or ',' in {broken[0]!r}"}

        joined = ",".join(f"{name}:{path}" for name, path in sorted(spec.items()))
        return self._ensure(joined, ["--logdir_spec", joined], interpreter)

    def _ensure(self, logdir: str, location: list, interpreter: str) -> dict:

        port   = self._free_port()
        tb_id  = uuid.uuid4().hex[:8]

        stderr_path = Path(tempfile.gettempdir()) / f"tensorboard_{tb_id}.log"

        record = {
            "id"          : tb_id,
            "logdir"      : logdir,
            "port"        : port,
            "pid"         : None,
            "status"      : "starting",
            "started"     : datetime.now().isoformat(timespec="seconds"),
            "process"     : None,
            "stderr_path" : stderr_path,
        }

        with self.lock:
            for existing in self.instances.values():
                if existing["logdir"] == logdir and self._alive(existing):
                    return {"ok": True, **self._view(existing)}
            self.instances[tb_id] = record

        tb_bin  = shutil.which("tensorboard")
        command = [tb_bin] if tb_bin else [interpreter, "-m", "tensorboard.main"]

        argv = command + [
            *location,
            "--host",            "127.0.0.1",
            "--port",            str(port),
            "--path_prefix",     f"/tb/{tb_id}",
            "--reload_interval", "30",
        ]

        with open(stderr_path, "wb") as stderr_file:
            try:
                process = subprocess.Popen(argv, cwd=str(self.paths.repo_root), stdout=stderr_file, stderr=stderr_file)
            except OSError as exc:
                with self.lock:
                    self.instances.pop(tb_id, None)
                return {"ok": False, "error": str(exc)}

        with self.lock:
            record["process"] = process
            record["pid"]     = process.pid

        self.logger.ok(f"tensorboard {tb_id} starting on port {port} for {logdir}")

        threading.Thread(target=self._probe, args=(record,), daemon=True).start()

        return {"ok": True, **self._view(record)}

    def running_logdir(self, logdir: str) -> str | None:

        logdir = str(logdir)
        with self.lock:
            for record in self.instances.values():
                if record["logdir"] == logdir and self._alive(record):
                    return record["id"]
        return None

    def run_tree(self, runs_root: str) -> dict:

        root = Path(runs_root).expanduser()
        if not root.is_dir():
            return {"ok": False, "error": f"runs root not found: {runs_root}"}

        tree = self._tree_children(root)

        self.logger.info(f"tensorboard run tree: {sum(node['run_count'] for node in tree)} runs under {root}")
        return {"ok": True, "runs_root": str(root), "tree": tree}

    def _tree_children(self, directory: Path) -> list:

        nodes = []
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(".") or not entry.is_dir():
                continue

            node = self._tree_node(entry)
            if node["run_count"]:
                nodes.append(node)

        return nodes

    def _tree_node(self, directory: Path) -> dict:

        events = directory / self.RUN_EVENT_FOLDER
        if events.is_dir():
            return self._tree_entry(directory, "run", int(self.has_event_files(events)), [])

        children = self._tree_children(directory)
        count    = sum(child["run_count"] for child in children)

        if all(child["kind"] == "run" for child in children):
            return self._tree_entry(directory, "set", count, [])
        return self._tree_entry(directory, "group", count, children)

    def _tree_entry(self, directory: Path, kind: str, run_count: int, children: list) -> dict:

        return {
            "name"      : directory.name,
            "path"      : str(directory),
            "kind"      : kind,
            "run_count" : run_count,
            "running"   : self.running_logdir(str(directory)),
            "children"  : children,
        }

    def has_event_files(self, directory: str | Path) -> bool:

        return next(Path(directory).rglob("*tfevents*"), None) is not None

    def run_dirs_for_pid(self, root: str | Path, pid: int, since: float) -> list:

        root     = Path(root)
        run_dirs = []

        for event in root.rglob("*tfevents*"):
            if not event.is_file():
                continue

            stamp, writer = self._parse_event_name(event.name)
            if writer != pid or stamp is None or stamp < since:
                continue

            relative = event.relative_to(root)
            run_dir  = root / relative.parts[0] if len(relative.parts) > 1 else root
            if run_dir not in run_dirs:
                run_dirs.append(run_dir)

        return run_dirs

    @staticmethod
    def _parse_event_name(name: str) -> tuple:

        parts = name.split(".")
        if parts and parts[-1] == "v2":
            parts = parts[:-1]

        pid = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else None

        try:
            anchor = parts.index("tfevents")
        except ValueError:
            return None, pid

        stamp = next((float(part) for part in parts[anchor + 1:] if part.isdigit()), None)
        return stamp, pid

    def get(self, tb_id: str) -> dict | None:

        with self.lock:
            return self.instances.get(tb_id)

    def list_instances(self) -> list[dict]:

        with self.lock:
            dead = [record["id"] for record in self.instances.values() if record["status"] == "running" and record["process"] is not None and record["process"].poll() is not None]
            for tb_id in dead:
                self.instances.pop(tb_id)["status"] = "stopped"
            records = [dict(record) for record in self.instances.values()]

        for tb_id in dead:
            self.logger.muted(f"tensorboard {tb_id} exited on its own, dropping it from the console")

        return [self._view(r) for r in sorted(records, key=lambda r: r["started"], reverse=True)]

    def stop(self, tb_id: str) -> dict:

        with self.lock:
            record = self.instances.pop(tb_id, None)
            if record is not None:
                record["status"] = "stopped"

        if record is None:
            return {"ok": False, "error": "unknown tensorboard instance"}

        self._reap(record)
        self.logger.warning(f"tensorboard {tb_id} stopped")
        return {"ok": True}

    def stop_all(self) -> None:

        with self.lock:
            records = list(self.instances.values())
            for record in records:
                record["status"] = "stopped"
            self.instances.clear()

        for record in records:
            self._reap(record)

    def _reap(self, record: dict) -> None:

        process = record["process"]
        if process is None or process.poll() is not None:
            return

        process.terminate()

        try:
            process.wait(timeout=self.STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            self.logger.warning(f"tensorboard {record['id']} ignored the terminate for {self.STOP_GRACE_S:.0f}s, killing it")
            process.kill()
            process.wait(timeout=self.STOP_GRACE_S)

    def _mark(self, record: dict, from_status: str, to_status: str) -> bool:

        with self.lock:
            if record["status"] != from_status:
                return False
            record["status"] = to_status
            return True

    def _probe(self, record: dict) -> None:

        url      = f"http://127.0.0.1:{record['port']}/tb/{record['id']}/"
        deadline = time.monotonic() + self.STARTUP_TIMEOUT_S

        while time.monotonic() < deadline:
            with self.lock:
                if record["status"] != "starting":
                    return

            if record["process"].poll() is not None:
                if self._mark(record, "starting", "failed"):
                    self.logger.error(f"tensorboard {record['id']} exited during startup: {self._stderr_tail(record)}")
                return

            try:
                with urllib.request.urlopen(url, timeout=2):
                    if self._mark(record, "starting", "running"):
                        self.logger.ok(f"tensorboard {record['id']} ready at /tb/{record['id']}/")
                    return
            except OSError:
                time.sleep(self.PROBE_INTERVAL_S)

        if self._mark(record, "starting", "failed"):
            self._reap(record)
            self.logger.error(f"tensorboard {record['id']} failed to start within {self.STARTUP_TIMEOUT_S:.0f}s: {self._stderr_tail(record)}")

    def _view(self, record: dict) -> dict:

        return {
            "id"      : record["id"],
            "logdir"  : record["logdir"],
            "port"    : record["port"],
            "pid"     : record["pid"],
            "status"  : record["status"],
            "started" : record["started"],
            "url"     : f"/tb/{record['id']}/",
        }

    def _stderr_tail(self, record: dict, max_chars: int = 500) -> str:

        try:
            text = record["stderr_path"].read_text(errors="replace").strip()
        except OSError:
            return "no stderr captured"
        return text[-max_chars:] if text else "no stderr captured"

    @staticmethod
    def _free_port() -> int:

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
