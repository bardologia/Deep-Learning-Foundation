"""TensorBoard subprocess lifecycle management for the web console.

Starts one TensorBoard server per log directory behind a ``/tb/<id>/`` path
prefix, probes it until it answers HTTP, and reaps it on shutdown.
"""

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
    """Starts, tracks and stops TensorBoard servers on behalf of the web UI.

    Attributes:
        paths     : Project path resolver used as the working directory for servers.
        logger    : Console logger the lifecycle events are written to.
        instances : Mapping of instance id to the mutable record of that server.
        lock      : Guard protecting ``instances`` against the probe threads.
    """


    TRAINING_LOGDIRS = {
        "train_backbone"            : ("logdir",),
        "train_profile_autoencoder" : ("logdir",),
        "train_image_autoencoder"   : ("logdir",),
        "train_jepa"                : ("logdir",),
        "train_unrolled"            : ("logdir",),
        "train_dual"                : ("logdir",),
        "benchmark"                 : ("paths.log_base_dir",),
        "cross_validate"            : ("paths.log_base_dir",),
        "tune"                      : ("paths.log_base_dir",),
    }

    STARTUP_TIMEOUT_S = 90.0
    PROBE_INTERVAL_S  = 0.5
    STOP_GRACE_S      = 10.0

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        """Stores the path resolver and logger and prepares the instance table.

        Args:
            paths  : Project path resolver providing the repository root.
            logger : Console logger for start, stop and failure messages.
        """

        self.paths     = paths
        self.logger    = logger
        self.instances = {}
        self.lock      = threading.Lock()

    def logdir_keys(self, script_key: str) -> tuple | None:
        """Returns the config keys holding the log directory for a script key."""

        return self.TRAINING_LOGDIRS.get(script_key)

    def _alive(self, record: dict) -> bool:
        """Returns whether the recorded server is still startable or serving.

        Args:
            record : Instance record as stored in ``instances``.

        Returns:
            True when the record is in a live state and its process has not exited.
        """

        if record["status"] not in ("starting", "running"):
            return False
        process = record["process"]
        return process is None or process.poll() is None

    def ensure(self, logdir: str, interpreter: str) -> dict:
        """Returns a server for the given log directory, starting one if needed.

        Args:
            logdir      : Directory of TensorBoard event files to serve.
            interpreter : Python interpreter used when no ``tensorboard`` binary is on PATH.

        Returns:
            Dictionary with ``ok`` plus the public instance view, or ``ok`` False and
            an ``error`` message when the process could not be spawned.
        """

        return self._ensure(str(logdir), ["--logdir", str(logdir)], interpreter)

    def ensure_spec(self, spec: dict, interpreter: str) -> dict:
        """Returns a server over exactly the given run directories, starting one if needed.

        The directories are served through ``--logdir_spec``, so TensorBoard shows
        each one under its label instead of globbing a whole runs root.

        Args:
            spec        : Mapping of run label to the run directory served under it; labels must not contain ``:`` or ``,`` and directories must not contain ``,``.
            interpreter : Python interpreter used when no ``tensorboard`` binary is on PATH.

        Returns:
            Dictionary with ``ok`` plus the public instance view, or ``ok`` False and
            an ``error`` message for an unservable spec or a failed spawn.
        """

        broken = [name for name in spec if ":" in name or "," in name] + [path for path in spec.values() if "," in str(path)]
        if broken:
            return {"ok": False, "error": f"cannot serve via --logdir_spec, ':' or ',' in {broken[0]!r}"}

        joined = ",".join(f"{name}:{path}" for name, path in sorted(spec.items()))
        return self._ensure(joined, ["--logdir_spec", joined], interpreter)

    def _ensure(self, logdir: str, location: list, interpreter: str) -> dict:
        """Starts, or reuses, the instance identified by a log scope.

        An already live instance serving the same scope is reused. Otherwise a
        TensorBoard process is spawned on a free loopback port with a unique path
        prefix, and a background thread probes it until it answers.

        Args:
            logdir      : Canonical scope string identifying the instance, either a plain directory or a joined ``--logdir_spec`` value.
            location    : Argv fragment pointing TensorBoard at the scope.
            interpreter : Python interpreter used when no ``tensorboard`` binary is on PATH.

        Returns:
            Dictionary with ``ok`` plus the public instance view, or ``ok`` False and
            an ``error`` message when the process could not be spawned.
        """

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
        """Returns the id of the live instance serving this log directory, if any."""

        logdir = str(logdir)
        with self.lock:
            for record in self.instances.values():
                if record["logdir"] == logdir and self._alive(record):
                    return record["id"]
        return None

    def list_logdirs(self, runs_root: str) -> dict:
        """Lists candidate log directories directly under a runs root.

        Args:
            runs_root : Directory whose immediate subdirectories are training runs.

        Returns:
            Dictionary with ``ok``, the resolved ``runs_root`` and a ``logdirs`` list of
            entries carrying name, path, event-run count and any running instance id;
            ``ok`` is False with an ``error`` when the root does not exist.
        """

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

    def has_event_files(self, directory: str | Path) -> bool:
        """Returns whether any TensorBoard event file exists under a directory."""

        return next(Path(directory).rglob("*tfevents*"), None) is not None

    def run_dirs_for_pid(self, root: str | Path, pid: int, since: float) -> list:
        """Returns the run directories under a root whose event files a process wrote.

        Event file names embed the writer's start timestamp, host and pid; a file
        counts when its pid matches and its timestamp is not older than ``since``.
        The run directory is the first path component below the root, or the root
        itself for an event file sitting directly in it.

        Args:
            root  : Runs root to scan recursively.
            pid   : Writer process id the event files must carry.
            since : Unix timestamp; event files stamped earlier are ignored.

        Returns:
            Matching run directories in scan order, without duplicates.
        """

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
        """Returns the (start timestamp, writer pid) embedded in an event file name.

        Args:
            name : Event file name of the ``events.out.tfevents.<time>.<host>.<pid>.<n>`` form, optionally with a trailing ``.v2``.

        Returns:
            Tuple of timestamp and pid, either being None when the name does not
            carry that field in a parseable form.
        """

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

    def _count_event_runs(self, directory: Path) -> int:
        """Returns the number of distinct directories holding tfevents files."""

        parents = set()
        for event in directory.rglob("*tfevents*"):
            if event.is_file():
                parents.add(event.parent)
        return len(parents)

    def get(self, tb_id: str) -> dict | None:
        """Returns the raw instance record for an id, or None when unknown."""

        with self.lock:
            return self.instances.get(tb_id)

    def list_instances(self) -> list[dict]:
        """Returns public views of all tracked instances, newest first.

        Instances whose process exited on its own are dropped from the table before
        the views are built.

        Returns:
            List of instance view dictionaries sorted by descending start time.
        """

        with self.lock:
            dead = [record["id"] for record in self.instances.values() if record["status"] == "running" and record["process"] is not None and record["process"].poll() is not None]
            for tb_id in dead:
                self.instances.pop(tb_id)["status"] = "stopped"
            records = [dict(record) for record in self.instances.values()]

        for tb_id in dead:
            self.logger.muted(f"tensorboard {tb_id} exited on its own, dropping it from the console")

        return [self._view(r) for r in sorted(records, key=lambda r: r["started"], reverse=True)]

    def stop(self, tb_id: str) -> dict:
        """Stops a tracked instance and removes it from the table.

        Args:
            tb_id : Instance identifier returned by ``ensure``.

        Returns:
            Dictionary with ``ok`` True, or ``ok`` False and an ``error`` for an
            unknown identifier.
        """

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
        """Stops every tracked instance and clears the table."""

        with self.lock:
            records = list(self.instances.values())
            for record in records:
                record["status"] = "stopped"
            self.instances.clear()

        for record in records:
            self._reap(record)

    def _reap(self, record: dict) -> None:
        """Terminates the recorded process, killing it if it ignores the terminate.

        Args:
            record : Instance record whose process should be shut down.
        """

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
        """Transitions a record's status under the lock and reports whether it applied.

        Args:
            record      : Instance record to transition.
            from_status : Status the record must currently hold.
            to_status   : Status to write when the precondition holds.

        Returns:
            True when the transition was applied, False when the record had moved on.
        """

        with self.lock:
            if record["status"] != from_status:
                return False
            record["status"] = to_status
            return True

    def _probe(self, record: dict) -> None:
        """Polls a starting server until it answers HTTP, exits, or times out.

        Runs on a background thread; marks the record running on success and failed
        on early exit or timeout, reaping the process in the timeout case.

        Args:
            record : Instance record of the freshly spawned server.
        """

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
        """Returns the JSON-safe public view of an instance record."""

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
        """Returns the tail of the captured stderr log for failure reporting.

        Args:
            record    : Instance record holding the stderr file path.
            max_chars : Maximum number of trailing characters to return.

        Returns:
            Trailing text of the log, or a placeholder when nothing was captured.
        """

        try:
            text = record["stderr_path"].read_text(errors="replace").strip()
        except OSError:
            return "no stderr captured"
        return text[-max_chars:] if text else "no stderr captured"

    @staticmethod
    def _free_port() -> int:
        """Returns a currently free TCP port on the loopback interface."""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
