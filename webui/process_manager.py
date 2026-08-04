from __future__ import annotations

import codecs
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime    import datetime, timedelta
from pathlib     import Path

from tools.orchestration.gpu_queue import GpuPoolFile, GpuProgressFile

from job_describer import JobDescriber
from notifier      import JobNotifier
from proc_stats    import ProcStats
from project_paths import ProjectPaths
from web_logger    import WebLogger


class JobStream:

    def __init__(self) -> None:
        self.buffer      = deque(maxlen=4000)
        self.subscribers = []
        self.lock        = threading.Lock()

    def publish(self, event: dict) -> None:
        with self.lock:
            self.buffer.append(event)
            for sub in list(self.subscribers):
                try:
                    sub.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self) -> queue.Queue:
        sub = queue.Queue(maxsize=8000)
        with self.lock:
            for event in self.buffer:
                sub.put_nowait(event)
            self.subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: queue.Queue) -> None:
        with self.lock:
            if sub in self.subscribers:
                self.subscribers.remove(sub)


class ProcessManager:

    OVERRIDE_NAME    = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
    DETACHED_PID     = re.compile(r"detached .* as pid (\d+)")
    ORPHAN_MIN_AGE_S = 15.0
    ORPHAN_RESCAN_S  = 3.0
    HISTORY_LIMIT    = 50
    ENDED_STATES     = ("finished", "failed", "cancelled")
    POOL_SCRIPTS     = ("train_backbone", "train_dual", "train_jepa", "train_image_autoencoder", "train_profile_autoencoder", "train_unrolled", "sweep_patches", "benchmark", "cross_validate", "tune", "infer_backbone", "infer_profile_autoencoder", "infer_image_autoencoder", "infer_unrolled", "infer_dual")
    POOL_FIELD       = "gpus_file"

    def __init__(self, paths: ProjectPaths, logger: WebLogger, notifier: JobNotifier, describer: JobDescriber) -> None:
        self.paths         = paths
        self.logger        = logger
        self.notifier      = notifier
        self.describer     = describer
        self.jobs          = {}
        self.streams       = {}
        self.launch_queue  = deque()
        self.lock          = threading.Lock()
        self.uid           = os.getuid()
        self.clk           = os.sysconf("SC_CLK_TCK")
        self.orphan_scan_t = 0.0

    def launch(self, key: str, interpreter: str, overrides: dict | None = None, follow_up: str | None = None, detach: bool = False) -> dict:
        script = self.paths.script_entry(key)["path"]
        if not script.exists():
            return {"ok": False, "error": "script not found"}

        if follow_up and not detach:
            if not self.paths.has_script(follow_up):
                return {"ok": False, "error": f"unknown follow-up script '{follow_up}'"}
            if not self.paths.script_entry(follow_up)["path"].exists():
                return {"ok": False, "error": f"follow-up script '{follow_up}' not found"}

        record = self._make_record(key, interpreter, self._clean_overrides(overrides), detach)
        stream = JobStream()

        with self.lock:
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream

        error = self._start(record, stream)
        if error is not None:
            with self.lock:
                self.jobs.pop(record["job_id"], None)
                self.streams.pop(record["job_id"], None)
            return {"ok": False, "error": error}

        if follow_up and detach:
            self.logger.warning(f"follow-up {follow_up} ignored, {key} launched detached")
        elif follow_up:
            self._schedule(record, follow_up)

        return {"ok": True, "job_id": record["job_id"]}

    def enqueue(self, key: str, interpreter: str, overrides: dict | None = None, follow_up: str | None = None, detach: bool = False) -> dict:
        script = self.paths.script_entry(key)["path"]
        if not script.exists():
            return {"ok": False, "error": "script not found"}

        if follow_up and detach:
            self.logger.warning(f"follow-up {follow_up} ignored, {key} queued detached")
            follow_up = None
        elif follow_up:
            if not self.paths.has_script(follow_up):
                return {"ok": False, "error": f"unknown follow-up script '{follow_up}'"}
            if not self.paths.script_entry(follow_up)["path"].exists():
                return {"ok": False, "error": f"follow-up script '{follow_up}' not found"}

        record                 = self._make_record(key, interpreter, self._clean_overrides(overrides), detach)
        record["status"]       = "queued"
        record["queue_follow"] = follow_up
        stream                 = JobStream()

        with self.lock:
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream
            self.launch_queue.append(record["job_id"])
            position = len(self.launch_queue)

        stream.publish({"type": "status", "status": "queued", "position": position})
        self.logger.muted(f"queued {key} as job {record['job_id']} at position {position}{self._described(record)}")

        self._advance_queue()

        with self.lock:
            still_queued = record["status"] == "queued"
        return {"ok": True, "job_id": record["job_id"], "queued": still_queued}

    def _make_record(self, key: str, interpreter: str, overrides: dict, detach: bool = False) -> dict:
        job_id    = uuid.uuid4().hex[:12]
        overrides = self._with_pool_file(key, overrides, job_id)

        return {
            "job_id"       : job_id,
            "script"       : key,
            "command"      : self._render_command(interpreter, key, overrides, detach),
            "description"  : self.describer.describe(key, interpreter, overrides),
            "interpreter"  : interpreter,
            "overrides"    : overrides,
            "detach"       : detach,
            "status"       : "pending",
            "pid"          : None,
            "started"      : datetime.now().isoformat(timespec="seconds"),
            "exit_code"    : None,
            "follow_of"    : None,
            "follow_up"    : None,
            "queue_follow" : None,
        }

    def _with_pool_file(self, key: str, overrides: dict, job_id: str) -> dict:
        if key not in self.POOL_SCRIPTS or overrides.get(self.POOL_FIELD):
            return overrides

        return {**overrides, self.POOL_FIELD: str(self.paths.gpu_pools_dir / f"{job_id}.json")}

    def _runtime_env(self, interpreter: str) -> dict:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"]      = "1"
        env["COLUMNS"]          = "120"
        env["LINES"]            = "32"
        env.setdefault("QT_QPA_PLATFORM", "offscreen")

        library_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), "lib")
        library_path = env.get("LD_LIBRARY_PATH", "")
        if library_dir not in library_path.split(":"):
            env["LD_LIBRARY_PATH"] = library_dir + (":" + library_path if library_path else "")

        return env

    def _start(self, record: dict, stream: JobStream) -> str | None:
        entry = self.paths.script_entry(record["script"])
        argv  = [record["interpreter"], "-u", str(entry["path"])]
        for path, value in record["overrides"].items():
            argv += [f"--{path}", value]
        if record.get("detach"):
            argv.append("--detach")

        env = self._runtime_env(record["interpreter"])

        try:
            process = subprocess.Popen(
                argv,
                cwd               = str(self.paths.repo_root),
                stdout            = subprocess.PIPE,
                stderr            = subprocess.STDOUT,
                env               = env,
                start_new_session = True,
            )
        except OSError as exc:
            return str(exc)

        with self.lock:
            record["status"]  = "running"
            record["pid"]     = process.pid
            record["started"] = datetime.now().isoformat(timespec="seconds")
            snapshot          = dict(record)

        self.logger.ok(f"launched {record['script']} as job {record['job_id']} (pid {process.pid}){self._described(record)}")
        stream.publish({"type": "status", "status": "running", "pid": process.pid})
        self.notifier.job_started(snapshot)

        worker = threading.Thread(target=self._pump, args=(record["job_id"], process, stream), daemon=True)
        worker.start()
        return None

    def _schedule(self, parent: dict, key: str) -> None:
        record              = self._make_record(key, parent["interpreter"], {})
        record["status"]    = "scheduled"
        record["follow_of"] = parent["job_id"]

        stream = JobStream()
        stream.publish({"type": "status", "status": "scheduled", "after": parent["script"]})

        with self.lock:
            parent["follow_up"]            = record["job_id"]
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream

        self.logger.muted(f"scheduled {key} as job {record['job_id']} after {parent['script']} ({parent['job_id']}){self._described(record)}")

    def _described(self, record: dict) -> str:
        description = record.get("description") or ""
        return f" — {description}" if description else ""

    def _resolve_follow_up(self, follow_id: str, code: int) -> None:
        with self.lock:
            record = self.jobs.get(follow_id)
            stream = self.streams.get(follow_id)
        if record is None or stream is None or record["status"] != "scheduled":
            return

        if code == 0:
            error = self._start(record, stream)
            if error is None:
                return
            with self.lock:
                record["status"] = "failed"
                snapshot         = dict(record)
            stream.publish({"type": "status", "status": "failed", "code": None, "verdict": "error"})
            stream.publish({"type": "end"})
            self.logger.error(f"scheduled job {follow_id} failed to start: {error}")
            self.notifier.job_finished(snapshot)
            return

        with self.lock:
            record["status"] = "cancelled"
        stream.publish({"type": "status", "status": "cancelled", "code": None})
        stream.publish({"type": "end"})
        self.logger.warning(f"scheduled job {follow_id} cancelled, parent exited with code {code}")

    def _advance_queue(self) -> None:
        while True:
            with self.lock:
                busy = any(r["status"] in ("pending", "running") for r in self.jobs.values())
                if busy or not self.launch_queue:
                    return
                job_id = self.launch_queue.popleft()
                record = self.jobs.get(job_id)
                stream = self.streams.get(job_id)
                if record is None or stream is None or record["status"] != "queued":
                    continue
                record["status"] = "pending"

            error = self._start(record, stream)
            if error is None:
                if record["queue_follow"]:
                    self._schedule(record, record["queue_follow"])
                return

            with self.lock:
                record["status"] = "failed"
                snapshot         = dict(record)
            stream.publish({"type": "status", "status": "failed", "code": None, "verdict": "error"})
            stream.publish({"type": "end"})
            self.logger.error(f"queued job {job_id} failed to start: {error}")
            self.notifier.job_finished(snapshot)

    def _clean_overrides(self, overrides: dict | None) -> dict:
        cleaned = {}
        for path, value in (overrides or {}).items():
            if not isinstance(path, str) or not self.OVERRIDE_NAME.match(path):
                raise ValueError(f"invalid override key '{path}'")
            cleaned[path] = str(value)
        return cleaned

    def _render_command(self, interpreter: str, key: str, overrides: dict, detach: bool = False) -> str:
        entry = self.paths.script_entry(key)
        parts = [interpreter, "-u", entry["rel"]]
        for path, value in overrides.items():
            parts += [f"--{path}", shlex.quote(value)]
        if detach:
            parts.append("--detach")
        return " ".join(parts)

    def _pump(self, job_id: str, process: subprocess.Popen, stream: JobStream) -> None:
        fd      = process.stdout.fileno()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        with self.lock:
            detached = bool(self.jobs.get(job_id, {}).get("detach"))
        head = ""

        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                stream.publish({"type": "chunk", "data": text})
                if detached and len(head) < 4096:
                    head += text

        tail = decoder.decode(b"", final=True)
        if tail:
            stream.publish({"type": "chunk", "data": tail})
            if detached and len(head) < 4096:
                head += tail

        process.wait()
        code = process.returncode

        detached_pid = self._parse_detached_pid(head) if detached else None
        if detached_pid and code == 0:
            self._adopt_detached(job_id, detached_pid, stream)
            return

        with self.lock:
            record = self.jobs.get(job_id)
            if record is not None:
                record["status"]    = "finished" if code == 0 else "failed"
                record["exit_code"] = code
            follow_id = record["follow_up"] if record else None
            snapshot  = dict(record) if record else None

        if follow_id:
            self._resolve_follow_up(follow_id, code)

        verdict = "ok" if code == 0 else "error"
        self.logger.muted(f"job {job_id} exited with code {code}")
        stream.publish({"type": "status", "status": record["status"], "code": code, "verdict": verdict})
        stream.publish({"type": "end"})

        if snapshot is not None:
            snapshot["progress"] = self._final_progress(job_id)
            self.notifier.job_finished(snapshot)

        self._advance_queue()
        self._prune_history()

    def _parse_detached_pid(self, text: str) -> int | None:
        match = self.DETACHED_PID.search(text)
        return int(match.group(1)) if match else None

    def _adopt_detached(self, job_id: str, pid: int, stream: JobStream) -> None:
        token = self._pid_start_token(pid)

        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            record["status"]           = "running"
            record["pid"]              = pid
            record["detached_running"] = True

        self.logger.ok(f"job {job_id} detached, now tracking live pid {pid}")
        stream.publish({"type": "status", "status": "running", "pid": pid, "detached": True})

        watcher = threading.Thread(target=self._watch_detached, args=(job_id, pid, token, stream), daemon=True)
        watcher.start()

    def _watch_detached(self, job_id: str, pid: int, token: tuple | None, stream: JobStream) -> None:
        while self._pid_alive(pid) and self._pid_start_token(pid) == token:
            time.sleep(2.0)
            with self.lock:
                if self.jobs.get(job_id) is None:
                    return

        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            record["status"]    = "finished"
            record["exit_code"] = None
            snapshot            = dict(record)

        self.logger.muted(f"detached job {job_id} (pid {pid}) exited, exit status unknown")
        stream.publish({"type": "status", "status": "finished", "code": None, "verdict": "unknown"})
        stream.publish({"type": "end"})

        snapshot["progress"] = self._final_progress(job_id)
        self.notifier.job_finished(snapshot)

        self._advance_queue()
        self._prune_history()

    def _final_progress(self, job_id: str) -> dict | None:
        info = self.progress(job_id)
        return info["progress"] if info.get("ok") and "progress" in info else None

    def _prune_history(self) -> None:
        with self.lock:
            ended   = sorted((record for record in self.jobs.values() if record["status"] in self.ENDED_STATES), key=lambda r: r["started"])
            dropped = ended[: max(0, len(ended) - self.HISTORY_LIMIT)]
            for record in dropped:
                self.jobs.pop(record["job_id"], None)
                self.streams.pop(record["job_id"], None)

        if dropped:
            self.logger.muted(f"dropped {len(dropped)} ended job(s) and their output from the console, keeping the {self.HISTORY_LIMIT} most recent")

    def adopt_orphans(self) -> int:
        now = time.monotonic()
        with self.lock:
            if now - self.orphan_scan_t < self.ORPHAN_RESCAN_S:
                return 0
            self.orphan_scan_t = now

        orphans = self._orphan_candidates()
        for pid, info in orphans.items():
            self._adopt_orphan(pid, info)
        return len(orphans)

    def _orphan_candidates(self) -> dict:
        found = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == os.getpid():
                continue

            try:
                if os.stat(f"/proc/{pid}").st_uid != self.uid:
                    continue
                raw = open(f"/proc/{pid}/cmdline", "rb").read()
            except OSError:
                continue

            tokens = [token.decode(errors="replace") for token in raw.split(b"\x00") if token]
            if not tokens or not os.path.basename(tokens[0]).startswith("python"):
                continue

            script = self._orphan_script(pid, tokens)
            if script is None:
                continue

            age = self._pid_age_s(pid)
            if age is None or age < self.ORPHAN_MIN_AGE_S:
                continue
            if self.job_for_pid(pid) is not None:
                continue

            found[pid] = {"script": script, "cmd": " ".join(tokens), "tokens": tokens, "interpreter": tokens[0], "age_s": age}

        return {pid: info for pid, info in found.items() if self._orphan_ancestor(pid, set(found)) is None}

    def _orphan_script(self, pid: int, tokens: list[str]) -> str | None:
        for token in tokens[1:]:
            if not token.endswith(".py"):
                continue

            path = Path(token)
            if not path.is_absolute():
                try:
                    path = Path(os.readlink(f"/proc/{pid}/cwd")) / path
                except OSError:
                    return None

            try:
                path = path.resolve()
            except OSError:
                return None
            return path.stem if path.is_relative_to(self.paths.main_dir) else None
        return None

    def _pid_age_s(self, pid: int) -> float | None:
        try:
            stat   = open(f"/proc/{pid}/stat").read()
            fields = stat[stat.rindex(")") + 2 :].split()
            uptime = float(open("/proc/uptime").read().split()[0])
            return uptime - int(fields[19]) / self.clk
        except (OSError, ValueError, IndexError):
            return None

    def _orphan_ancestor(self, pid: int, known: set) -> int | None:
        parent = ProcStats.ppid(pid)
        hops   = 0
        while parent > 1 and hops < 16:
            if parent in known:
                return parent
            parent = ProcStats.ppid(parent)
            hops  += 1
        return None

    def _orphan_overrides(self, tokens: list[str]) -> dict:
        args = []
        for index, token in enumerate(tokens):
            if token.endswith(".py"):
                args = tokens[index + 1 :]
                break

        overrides = {}
        index     = 0
        while index < len(args):
            if not args[index].startswith("--"):
                index += 1
                continue
            name = args[index][2:]
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                overrides[name] = args[index + 1]
                index += 2
            else:
                overrides[name] = "True"
                index += 1
        return overrides

    def _adopt_orphan(self, pid: int, info: dict) -> None:
        overrides = self._orphan_overrides(info["tokens"])
        record = {
            "job_id"           : uuid.uuid4().hex[:12],
            "script"           : info["script"],
            "command"          : info["cmd"],
            "description"      : "",
            "interpreter"      : info["interpreter"],
            "overrides"        : overrides,
            "detach"           : False,
            "status"           : "running",
            "pid"              : pid,
            "started"          : (datetime.now() - timedelta(seconds=info["age_s"])).isoformat(timespec="seconds"),
            "exit_code"        : None,
            "follow_of"        : None,
            "follow_up"        : None,
            "adopted"          : True,
            "detached_running" : True,
        }

        stream = JobStream()
        stream.publish({"type": "status", "status": "running", "pid": pid, "adopted": True})

        with self.lock:
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream

        self.logger.warning(f"adopted untracked process {info['script']} (pid {pid}) into the console")

        threading.Thread(target=self._describe_adopted, args=(record["job_id"], info, overrides), daemon=True).start()

        token   = self._pid_start_token(pid)
        watcher = threading.Thread(target=self._watch_detached, args=(record["job_id"], pid, token, stream), daemon=True)
        watcher.start()

    def _describe_adopted(self, job_id: str, info: dict, overrides: dict) -> None:
        description = self.describer.describe(info["script"], info["interpreter"], overrides)

        with self.lock:
            record = self.jobs.get(job_id)
            if record is not None:
                record["description"] = description

    def _pid_alive(self, pid: int) -> bool:
        try:
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            return False
        try:
            return stat[stat.rindex(")") + 2] != "Z"
        except (ValueError, IndexError):
            return False

    def _pid_start_token(self, pid: int) -> tuple | None:
        try:
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            return None
        try:
            fields = stat[stat.rindex(")") + 2 :].split()
            return (pid, fields[19])
        except (ValueError, IndexError):
            return None

    def stop(self, job_id: str) -> dict:
        with self.lock:
            record = self.jobs.get(job_id)
            stream = self.streams.get(job_id)
        if record is None:
            return {"ok": False, "error": "unknown job"}

        if record["status"] == "queued":
            with self.lock:
                record["status"] = "cancelled"
                if job_id in self.launch_queue:
                    self.launch_queue.remove(job_id)
            if stream is not None:
                stream.publish({"type": "status", "status": "cancelled", "code": None})
                stream.publish({"type": "end"})
            self.logger.warning(f"queued job {job_id} cancelled by user")
            return {"ok": True}

        if record["status"] == "scheduled":
            with self.lock:
                record["status"] = "cancelled"
            if stream is not None:
                stream.publish({"type": "status", "status": "cancelled", "code": None})
                stream.publish({"type": "end"})
            self.logger.warning(f"scheduled job {job_id} cancelled by user")
            return {"ok": True}

        if record["status"] != "running":
            return {"ok": False, "error": "job is not running"}

        with self.lock:
            self.jobs[job_id]["stopped"] = True
        self._signal_group(record["pid"], signal.SIGTERM)
        self.logger.warning(f"stop requested for job {job_id}")
        return {"ok": True}

    def gpu_pool(self, job_id: str) -> dict:
        with self.lock:
            record = self.jobs.get(job_id)

        if record is None:
            return {"ok": False, "error": "unknown job"}

        path = record["overrides"].get(self.POOL_FIELD)
        if not path:
            return {"ok": True, "supported": False, "live": False}

        pool = Path(path)
        if not pool.is_file():
            return {"ok": True, "supported": True, "live": False, "path": str(pool)}

        try:
            gpus = GpuPoolFile.validate(json.loads(pool.read_text(encoding="utf-8")))
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as error:
            return {"ok": False, "error": f"unreadable GPU pool file: {error}", "path": str(pool)}

        return {"ok": True, "supported": True, "live": True, "gpus": gpus, "path": str(pool)}

    def progress(self, job_id: str) -> dict:
        with self.lock:
            record = self.jobs.get(job_id)

        if record is None:
            return {"ok": False, "error": "unknown job"}

        path = record["overrides"].get(self.POOL_FIELD)
        if not path:
            return {"ok": True, "supported": False, "live": False}

        progress_path = GpuProgressFile.resolve(Path(path))
        if not progress_path.is_file():
            return {"ok": True, "supported": True, "live": False, "path": str(progress_path)}

        try:
            snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            return {"ok": False, "error": f"unreadable progress file: {error}", "path": str(progress_path)}

        return {"ok": True, "supported": True, "live": record["status"] == "running", "progress": snapshot, "path": str(progress_path)}

    def set_gpus(self, job_id: str, gpus, park: bool = False) -> dict:
        with self.lock:
            record = self.jobs.get(job_id)

        if record is None:
            return {"ok": False, "error": "unknown job"}

        if record["status"] != "running":
            return {"ok": False, "error": "job is not running"}

        path = record["overrides"].get(self.POOL_FIELD)
        if not path:
            return {"ok": False, "error": f"{record['script']} was not launched with a live GPU pool"}

        pool = Path(path)
        if not pool.is_file():
            return {"ok": False, "error": "this job has no live GPU pool yet; the pool appears once a fan-out over trials or seeds starts dispatching"}

        try:
            requested = GpuPoolFile.validate({"gpus": gpus})
        except (ValueError, TypeError) as error:
            return {"ok": False, "error": str(error)}

        if not requested and not park:
            return {"ok": False, "error": "select at least one GPU, or confirm parking to hold the experiment"}

        GpuPoolFile(pool, self.logger).write(requested)

        if requested:
            self.logger.ok(f"job {job_id} GPU pool set to {requested}")
        else:
            self.logger.warning(f"job {job_id} parked — no GPUs left in the pool, runs in flight will finish and nothing new starts")

        return {"ok": True, "gpus": requested, "parked": not requested}

    def clear_queue(self) -> int:
        with self.lock:
            pending = [self.jobs[job_id] for job_id in self.launch_queue if job_id in self.jobs]
            streams = [self.streams.get(record["job_id"]) for record in pending]
            self.launch_queue.clear()
            for record in pending:
                record["status"] = "cancelled"

        for record, stream in zip(pending, streams):
            if stream is not None:
                stream.publish({"type": "status", "status": "cancelled", "code": None})
                stream.publish({"type": "end"})
            self.logger.warning(f"queued job {record['job_id']} cancelled, launch queue cleared")

        return len(pending)

    def stop_all(self, grace: float = 8.0) -> int:
        self.clear_queue()

        with self.lock:
            running = [dict(r) for r in self.jobs.values() if r["status"] == "running"]

        if not running:
            return 0

        for record in running:
            with self.lock:
                self.jobs[record["job_id"]]["stopped"] = True
            self._signal_group(record["pid"], signal.SIGTERM)
            self.logger.warning(f"watchdog stop for job {record['job_id']} (pid {record['pid']})")

        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            with self.lock:
                alive = [r for r in running if self.jobs[r["job_id"]]["status"] == "running"]
            if not alive:
                return len(running)
            time.sleep(0.5)

        with self.lock:
            stubborn = [r for r in running if self.jobs[r["job_id"]]["status"] == "running"]
        for record in stubborn:
            self._signal_group(record["pid"], signal.SIGKILL)
            self.logger.error(f"force kill for job {record['job_id']} (pid {record['pid']})")

        return len(running)

    @staticmethod
    def _signal_group(pid: int, sig: signal.Signals) -> None:
        try:
            os.killpg(pid, sig)
            return
        except (ProcessLookupError, PermissionError):
            pass

        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def job_for_pid(self, pid: int) -> str | None:
        with self.lock:
            running = {record["pid"]: record["job_id"] for record in self.jobs.values() if record["status"] == "running" and record["pid"] is not None}

        hops = 0
        while pid > 1 and hops < 16:
            if pid in running:
                return running[pid]
            pid   = ProcStats.ppid(pid)
            hops += 1
        return None

    def job_fate(self, job_id: str, pid: int) -> str:
        with self.lock:
            record = dict(self.jobs.get(job_id) or {})

        if not record:
            return "unknown"
        if record["status"] == "running":
            return "pending" if record["pid"] == pid else "released"
        if record["status"] == "failed":
            return "stopped" if record.get("stopped") else "crashed"
        if record["status"] == "finished":
            return "finished" if record["exit_code"] == 0 else "unknown"
        return "unknown"

    def list_jobs(self) -> list[dict]:
        with self.lock:
            records = [dict(record) for record in sorted(self.jobs.values(), key=lambda r: r["started"], reverse=True)]

        for record in records:
            info               = self.progress(record["job_id"]) if record["status"] == "running" else None
            record["progress"] = info["progress"] if info is not None and info.get("ok") and info.get("live") and "progress" in info else None

        return records

    def get_stream(self, job_id: str) -> JobStream | None:
        with self.lock:
            return self.streams.get(job_id)


class ProcessNuke:

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger
        self.uid    = os.getuid()

    def _spared_pids(self) -> set[int]:
        spared = {0, 1}
        pid    = os.getpid()

        while pid > 1:
            spared.add(pid)
            pid = ProcStats.ppid(pid)

        return spared

    def _targets(self, spared: set[int]) -> list[int]:
        return [pid for pid in self._user_pids() if pid not in spared]

    def _user_pids(self) -> list[int]:
        pids = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                if os.stat(f"/proc/{pid}").st_uid == self.uid:
                    pids.append(pid)
            except OSError:
                continue
        return pids

    def _terminate(self, targets: list[int], sig: signal.Signals) -> int:
        hit = 0
        for pid in targets:
            try:
                os.kill(pid, sig)
                hit += 1
            except (ProcessLookupError, PermissionError):
                continue
        return hit

    def _alive(self, targets: list[int]) -> list[int]:
        return [pid for pid in targets if os.path.exists(f"/proc/{pid}")]

    def nuke(self, grace: float = 4.0) -> dict:
        spared  = self._spared_pids()
        targets = self._targets(spared)

        if not targets:
            self.logger.muted("nuke requested, no processes to kill")
            return {"ok": True, "signalled": 0, "killed": 0}

        signalled = self._terminate(targets, signal.SIGTERM)
        self.logger.error(f"NUKE: SIGTERM sent to {signalled} of {len(targets)} user processes")

        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not self._alive(targets):
                return {"ok": True, "signalled": signalled, "killed": 0}
            time.sleep(0.25)

        stubborn = self._alive(targets)
        killed   = self._terminate(stubborn, signal.SIGKILL)
        self.logger.error(f"NUKE: SIGKILL sent to {killed} surviving processes")

        return {"ok": True, "signalled": signalled, "killed": killed}


class ServerDetacher:

    APPLY_TIMEOUT_S = 5.0
    LOG_NAME        = "webui_server.log"

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.log_path  = paths.logs_dir / self.LOG_NAME
        self.requested = threading.Event()
        self.applied   = threading.Event()
        self.stopping  = threading.Event()

    def state(self) -> dict:
        return {"detached": self.applied.is_set(), "pid": os.getpid(), "log_path": str(self.log_path)}

    def detach(self) -> dict:
        if not self.applied.is_set():
            self.requested.set()
            if not self.applied.wait(self.APPLY_TIMEOUT_S):
                return {"ok": False, "error": "detach was not applied by the main thread"}
        return {"ok": True, **self.state()}

    def shutdown(self) -> None:
        self.stopping.set()
        self.requested.set()

    def _apply(self) -> None:
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        handle = open(self.log_path, "ab", buffering=0)

        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        os.dup2(handle.fileno(), sys.stdout.fileno())
        os.dup2(handle.fileno(), sys.stderr.fileno())
        handle.close()

        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, sys.stdin.fileno())
        os.close(devnull)

        self.logger.enabled = False

    def wait_loop(self) -> None:
        while True:
            self.requested.wait()
            self.requested.clear()
            if self.stopping.is_set():
                self.logger.warning(f"front end shutdown requested from the console (pid {os.getpid()}); running jobs are their own session leaders and keep going")
                return
            if not self.applied.is_set():
                self._apply()
                self.applied.set()
                self.logger.warning(f"server detached from the terminal (pid {os.getpid()}), output continues in {self.log_path}")
