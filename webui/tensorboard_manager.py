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
        "train_profile_autoencoder" : ("logdir",),
        "train_image_autoencoder"   : ("logdir",),
        "train_jepa"                : ("logdir",),
        "train_dual"                : ("logdir",),
        "benchmark"                 : ("paths.log_base_dir",),
        "cross_validate"            : ("paths.log_base_dir",),
        "sweep_patches"             : ("paths.log_base_dir",),
        "tune"                      : ("paths.log_base_dir",),
    }

    STARTUP_TIMEOUT_S = 90.0
    PROBE_INTERVAL_S  = 0.5

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
        logdir = str(logdir)
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
            "--logdir",          logdir,
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

    def list_logdirs(self, runs_root: str) -> dict:
        root = Path(runs_root).expanduser()
        if not root.is_dir():
            return {"ok": False, "error": f"runs root not found: {runs_root}"}

        entries = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            entries.append({
                "name"      : entry.name,
                "path"      : str(entry),
                "run_count" : self._count_event_runs(entry),
                "running"   : self.running_logdir(str(entry)),
            })

        self.logger.info(f"tensorboard logdirs: listed {len(entries)} under {root}")
        return {"ok": True, "runs_root": str(root), "logdirs": entries}

    def _count_event_runs(self, directory: Path) -> int:
        parents = set()
        for event in directory.rglob("*tfevents*"):
            if event.is_file():
                parents.add(event.parent)
        return len(parents)

    def get(self, tb_id: str) -> dict | None:
        with self.lock:
            return self.instances.get(tb_id)

    def list_instances(self) -> list[dict]:
        with self.lock:
            for record in self.instances.values():
                if record["status"] == "running" and record["process"] is not None and record["process"].poll() is not None:
                    record["status"] = "stopped"
            records = [dict(record) for record in self.instances.values()]

        return [self._view(r) for r in sorted(records, key=lambda r: r["started"], reverse=True)]

    def stop(self, tb_id: str) -> dict:
        with self.lock:
            record = self.instances.get(tb_id)
            if record is not None:
                record["status"] = "stopped"

        if record is None:
            return {"ok": False, "error": "unknown tensorboard instance"}

        if record["process"] is not None and record["process"].poll() is None:
            record["process"].terminate()

        self.logger.warning(f"tensorboard {tb_id} stopped")
        return {"ok": True}

    def stop_all(self) -> None:
        with self.lock:
            records = list(self.instances.values())
            for record in records:
                record["status"] = "stopped"

        for record in records:
            if record["process"] is not None and record["process"].poll() is None:
                record["process"].terminate()

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
            if record["process"].poll() is None:
                record["process"].terminate()
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
