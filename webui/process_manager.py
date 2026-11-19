
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
from functools   import partial
from pathlib     import Path
from typing      import Callable

from tools.orchestration.gpu_queue import GpuPoolFile, GpuProgressFile

from job_describer  import JobDescriber
from log_transcript import AnsiTranscript
from notifier       import JobNotifier
from politeness     import JobPoliteness
from proc_stats     import ProcStats
from project_paths  import ProjectPaths
from web_logger     import WebLogger


class JobStream:

    QUEUE_LIMIT = 8000

    def __init__(self, logger: WebLogger) -> None:
        self.logger        = logger
        self.buffer        = deque(maxlen=4000)
        self.subscribers   = []
        self.dropped       = 0
        self.lock          = threading.Lock()
        self.event_counter = 0

    def publish(self, event: dict) -> None:
        with self.lock:
            self.event_counter += 1
            event["id"]         = self.event_counter
            self.buffer.append(event)
            overflowed = 0
            for sub in list(self.subscribers):
                try:
                    sub.put_nowait(event)
                except queue.Full:
                    overflowed += self._truncate(sub, event)

            self.dropped += overflowed

        if overflowed:
            self.logger.warning(f"a console viewer fell {overflowed} events behind and was truncated; the on-disk log keeps the full output")

    def _truncate(self, sub: queue.Queue, event: dict) -> int:
        dropped = 0

        while True:
            try:
                sub.get_nowait()
                dropped += 1
            except queue.Empty:
                break

        sub.put_nowait({"type": "chunk", "data": f"\r\n\x1b[2;31m── stream truncated: {dropped} event(s) dropped, this viewer could not keep up; the on-disk log is complete ──\x1b[0m\r\n"})
        sub.put_nowait(event)

        return dropped

    def attach(self, sink, last_event_id: int = 0) -> None:
        with self.lock:
            for event in self.buffer:
                if event["id"] > last_event_id:
                    sink.put_nowait(event)
            self.subscribers.append(sink)

    def subscribe(self, last_event_id: int = 0) -> queue.Queue:
        sub = queue.Queue(maxsize=self.QUEUE_LIMIT)
        self.attach(sub, last_event_id)
        return sub

    def unsubscribe(self, sub) -> None:
        with self.lock:
            if sub in self.subscribers:
                self.subscribers.remove(sub)


class LogTailer:

    POLL_S      = 0.5
    READ_SIZE   = 262144
    CATCHUP_MAX = 1048576

    def __init__(self, path: Path, stream: JobStream, alive: Callable[[], bool]) -> None:
        self.path    = path
        self.stream  = stream
        self.alive   = alive
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def _seek_catchup(self, handle) -> None:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - self.CATCHUP_MAX))

    def _drain(self, handle) -> None:
        while True:
            block = handle.read(self.READ_SIZE)
            if not block:
                return

            text = self.decoder.decode(block)
            if text:
                self.stream.publish({"type": "chunk", "data": text})

    def follow(self) -> None:
        with open(self.path, "rb") as handle:
            self._seek_catchup(handle)

            while True:
                self._drain(handle)
                if not self.alive():
                    self._drain(handle)
                    return
                time.sleep(self.POLL_S)


class ProcessManager:

    OVERRIDE_NAME    = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
    DETACHED_PID     = re.compile(r"detached .* as pid (\d+)")
    DETACHED_LOG     = re.compile(r"^output: (.+)$", re.M)
    ORPHAN_MIN_AGE_S = 15.0
    ORPHAN_RESCAN_S  = 3.0
    HISTORY_LIMIT    = 50
    LOG_VIEW_BYTES   = 4194304
    ENDED_STATES     = ("finished", "failed", "cancelled")
    POOL_SCRIPTS     = ("train_backbone", "train_dual", "train_jepa", "train_image_autoencoder", "train_profile_autoencoder", "benchmark", "cross_validate", "tune", "infer_backbone", "infer_profile_autoencoder", "infer_image_autoencoder", "infer_dual")
    POOL_FIELD       = "gpus_file"

    def __init__(self, paths: ProjectPaths, logger: WebLogger, notifier: JobNotifier, describer: JobDescriber, politeness: JobPoliteness) -> None:
        self.paths         = paths
        self.logger        = logger
        self.notifier      = notifier
        self.describer     = describer
        self.politeness    = politeness
        self.jobs          = {}
        self.streams       = {}
        self.launch_queue  = deque()
        self.queue_held    = None
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

        record = self._make_record(key, interpreter, self._clean_overrides(overrides), detach, follow_up)
        stream = JobStream(self.logger)

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

        record                 = self._make_record(key, interpreter, self._clean_overrides(overrides), detach, follow_up)
        record["status"]       = "queued"
        record["queue_follow"] = follow_up
        stream                 = JobStream(self.logger)

        with self.lock:
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream
            self.launch_queue.append(record["job_id"])
            position = len(self.launch_queue)
            blocker  = self._blocker()

        stream.publish({"type": "status", "status": "queued", "position": position, "blocker": blocker})
        self.logger.muted(f"queued {key} as job {record['job_id']} at position {position}{self._described(record)}")

        self._advance_queue()

        with self.lock:
            still_queued = record["status"] == "queued"
        return {"ok": True, "job_id": record["job_id"], "queued": still_queued}

    def _make_record(self, key: str, interpreter: str, overrides: dict, detach: bool = False, follow_up: str | None = None) -> dict:
        job_id    = uuid.uuid4().hex[:12]
        request   = {"overrides": dict(overrides), "follow_up": None if detach else follow_up, "finalize": {}}
        overrides = self._with_pool_file(key, overrides, job_id)

        return {
            "job_id"       : job_id,
            "script"       : key,
            "command"      : self._render_command(interpreter, key, overrides, detach),
            "description"  : self.describer.describe(key, interpreter, overrides),
            "interpreter"  : interpreter,
            "overrides"    : overrides,
            "request"      : request,
            "detach"       : detach,
            "status"       : "pending",
            "pid"          : None,
            "started"      : datetime.now().isoformat(timespec="seconds"),
            "exit_code"    : None,
            "follow_of"    : None,
            "follow_up"    : None,
            "queue_follow" : None,
            "adopted"      : False,
            "energy_wh"    : 0.0,
        }

    def _with_pool_file(self, key: str, overrides: dict, job_id: str) -> dict:
        if key not in self.POOL_SCRIPTS or overrides.get(self.POOL_FIELD):
            return overrides

        return {**overrides, self.POOL_FIELD: str(self.paths.gpu_pools_dir / f"{job_id}.json")}

    def _without_pool_file(self, overrides: dict) -> dict:
        pool = overrides.get(self.POOL_FIELD)
        if pool is None or Path(pool).parent != self.paths.gpu_pools_dir:
            return dict(overrides)
        return {path: value for path, value in overrides.items() if path != self.POOL_FIELD}

    def attach_finalize(self, job_id: str, flags: dict) -> None:
        with self.lock:
            self.jobs[job_id]["request"]["finalize"] = dict(flags)

    def _runtime_env(self, interpreter: str) -> dict:
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
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
        argv  = self.politeness.prefix() + [record["interpreter"], "-u", str(entry["path"])]
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

        stream = JobStream(self.logger)
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
            live   = record is not None and stream is not None and record["status"] == "scheduled"
            if live and code == 0:
                record["status"] = "pending"

        if not live:
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
                blocker = self._blocker()

                if blocker is not None or not self.launch_queue:
                    held            = blocker if blocker is not None and self.launch_queue and self.queue_held != blocker["job_id"] else None
                    self.queue_held = blocker["job_id"] if blocker is not None else None
                    break

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

        if held is not None:
            origin = "adopted process" if held["adopted"] else "job"
            where  = f" (pid {held['pid']})" if held["pid"] else ""
            self.logger.muted(f"launch queue held by {origin} {held['script']}{where}")

    def _blocker(self) -> dict | None:
        for record in self.jobs.values():
            if record["status"] in ("pending", "running"):
                return {"job_id": record["job_id"], "script": record["script"], "pid": record["pid"], "adopted": record["adopted"]}

        return None

    def _clean_overrides(self, overrides: dict | None) -> dict:
        cleaned = {}
        for path, value in (overrides or {}).items():
            if not isinstance(path, str) or not self.OVERRIDE_NAME.match(path):
                raise ValueError(f"invalid override key '{path}'")
            cleaned[path] = str(value)
        return cleaned

    def _render_command(self, interpreter: str, key: str, overrides: dict, detach: bool = False) -> str:
        entry = self.paths.script_entry(key)
        parts = self.politeness.prefix() + [interpreter, "-u", entry["rel"]]
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
            self._adopt_detached(job_id, detached_pid, self._parse_detached_log(head), stream)
            return

        status = "finished" if code == 0 else "failed"

        with self.lock:
            record = self.jobs.get(job_id)
            if record is not None:
                record["status"]    = status
                record["exit_code"] = code
            follow_id = record["follow_up"] if record else None
            snapshot  = dict(record) if record else None

        if follow_id:
            self._resolve_follow_up(follow_id, code)

        verdict = "ok" if code == 0 else "error"
        self.logger.muted(f"job {job_id} exited with code {code}")
        stream.publish({"type": "status", "status": status, "code": code, "verdict": verdict})
        stream.publish({"type": "end"})

        if snapshot is not None:
            snapshot["progress"] = self._final_progress(job_id)
            self.notifier.job_finished(snapshot)

        self._advance_queue()
        self._prune_history()

    def _parse_detached_pid(self, text: str) -> int | None:
        match = self.DETACHED_PID.search(text)
        return int(match.group(1)) if match else None

    def _parse_detached_log(self, text: str) -> Path | None:
        match = self.DETACHED_LOG.search(text)
        if match is None:
            return None

        path = Path(match.group(1).strip())
        path = path if path.is_absolute() else self.paths.repo_root / path
        return path if path.is_file() else None

    def _adopt_detached(self, job_id: str, pid: int, log: Path | None, stream: JobStream) -> None:
        token = self._pid_start_token(pid)

        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            record["status"]           = "running"
            record["pid"]              = pid
            record["detached_running"] = True
            record["log_path"]         = str(log) if log else None

        self.logger.ok(f"job {job_id} detached, now tracking live pid {pid}{f' and tailing {log}' if log else ''}")
        stream.publish({"type": "status", "status": "running", "pid": pid, "detached": True, "log": str(log) if log else None})

        tailer  = self._start_tailer(log, pid, token, stream)
        watcher = threading.Thread(target=self._watch_detached, args=(job_id, pid, token, stream, tailer), daemon=True)
        watcher.start()

    def _start_tailer(self, log: Path | None, pid: int, token: tuple | None, stream: JobStream) -> threading.Thread | None:
        if log is None:
            return None

        tailer = LogTailer(log, stream, partial(self._pid_running, pid, token))
        thread = threading.Thread(target=tailer.follow, daemon=True)
        thread.start()
        return thread

    def _watch_detached(self, job_id: str, pid: int, token: tuple | None, stream: JobStream, tailer: threading.Thread | None = None) -> None:
        while self._pid_running(pid, token):
            time.sleep(2.0)
            with self.lock:
                if self.jobs.get(job_id) is None:
                    return

        if tailer is not None:
            tailer.join(timeout=LogTailer.POLL_S * 6)

        progress        = self._final_progress(job_id)
        status, verdict = self._detached_verdict(progress)

        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            record["status"]    = status
            record["exit_code"] = None
            record["verdict"]   = verdict
            snapshot            = dict(record)

        self.logger.muted(f"detached job {job_id} (pid {pid}) exited, verdict {verdict} from its completion snapshot" if verdict != "unknown" else f"detached job {job_id} (pid {pid}) exited, exit status unknown")
        stream.publish({"type": "status", "status": status, "code": None, "verdict": verdict})
        stream.publish({"type": "end"})

        snapshot["progress"] = progress
        self.notifier.job_finished(snapshot)

        self._advance_queue()
        self._prune_history()

    def _detached_verdict(self, progress: dict | None) -> tuple[str, str]:
        if not progress or not progress.get("total"):
            return "finished", "unknown"
        if progress.get("done", 0) + progress.get("failed", 0) < progress["total"]:
            return "failed", "died"
        if progress.get("failed"):
            return "failed", "error"
        return "finished", "ok"

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
        found   = {}
        running = self._running_pids()
        parents = {}

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
            if self._ancestor_match(pid, running, parents) is not None:
                continue

            found[pid] = {"script": script, "cmd": " ".join(tokens), "tokens": tokens, "interpreter": tokens[0], "age_s": age}

        known = set(found)

        return {pid: info for pid, info in found.items() if self._ancestor_match(self._parent(pid, parents), known, parents) is None}

    def _running_pids(self) -> set:
        with self.lock:
            return {record["pid"] for record in self.jobs.values() if record["status"] == "running" and record["pid"] is not None}

    def _parent(self, pid: int, parents: dict) -> int:
        if pid not in parents:
            parents[pid] = ProcStats.ppid(pid)

        return parents[pid]

    def _ancestor_match(self, pid: int, known: set, parents: dict) -> int | None:
        hops = 0

        while pid > 1 and hops < 16:
            if pid in known:
                return pid
            pid   = self._parent(pid, parents)
            hops += 1

        return None

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

    def _pid_log_path(self, pid: int) -> Path | None:
        try:
            target = Path(os.readlink(f"/proc/{pid}/fd/1"))
        except OSError:
            return None
        return target if target.is_file() else None

    def _adopt_orphan(self, pid: int, info: dict) -> None:
        overrides = self._orphan_overrides(info["tokens"])
        log       = self._pid_log_path(pid)
        record = {
            "job_id"           : uuid.uuid4().hex[:12],
            "script"           : info["script"],
            "command"          : info["cmd"],
            "description"      : "",
            "interpreter"      : info["interpreter"],
            "overrides"        : overrides,
            "request"          : {"overrides": self._without_pool_file(overrides), "follow_up": None, "finalize": {}},
            "detach"           : False,
            "status"           : "running",
            "pid"              : pid,
            "started"          : (datetime.now() - timedelta(seconds=info["age_s"])).isoformat(timespec="seconds"),
            "exit_code"        : None,
            "follow_of"        : None,
            "follow_up"        : None,
            "adopted"          : True,
            "detached_running" : True,
            "log_path"         : str(log) if log else None,
            "energy_wh"        : 0.0,
        }

        stream = JobStream(self.logger)
        stream.publish({"type": "status", "status": "running", "pid": pid, "adopted": True, "log": str(log) if log else None})

        with self.lock:
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream

        self.logger.warning(f"adopted untracked process {info['script']} (pid {pid}) into the console{f', tailing {log}' if log else ''}")

        threading.Thread(target=self._describe_adopted, args=(record["job_id"], info, overrides), daemon=True).start()

        token   = self._pid_start_token(pid)
        tailer  = self._start_tailer(log, pid, token, stream)
        watcher = threading.Thread(target=self._watch_detached, args=(record["job_id"], pid, token, stream, tailer), daemon=True)
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

    def _pid_running(self, pid: int, token: tuple | None) -> bool:
        return self._pid_alive(pid) and self._pid_start_token(pid) == token

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
        pid = None

        with self.lock:
            record = self.jobs.get(job_id)
            stream = self.streams.get(job_id)
            status = record["status"] if record is not None else None

            if status in ("queued", "scheduled"):
                record["status"] = "cancelled"
                if job_id in self.launch_queue:
                    self.launch_queue.remove(job_id)
            elif status in ("pending", "running"):
                record["stopped"] = True
                pid               = record["pid"]

        if record is None:
            return {"ok": False, "error": "unknown job"}

        if status in ("queued", "scheduled"):
            if stream is not None:
                stream.publish({"type": "status", "status": "cancelled", "code": None})
                stream.publish({"type": "end"})
            self.logger.warning(f"{status} job {job_id} cancelled by user")
            return {"ok": True}

        if status not in ("pending", "running"):
            return {"ok": False, "error": "job is not running"}

        pid = pid if pid is not None else self._await_pid(job_id)
        if pid is not None:
            self._signal_group(pid, signal.SIGTERM)
        self.logger.warning(f"stop requested for job {job_id}")
        return {"ok": True}

    def _await_pid(self, job_id: str) -> int | None:
        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:
            with self.lock:
                record = self.jobs.get(job_id)
                if record is None or record["status"] not in ("pending", "running"):
                    return None
                if record["pid"] is not None:
                    return record["pid"]
            time.sleep(0.05)

        return None

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

    def unit_entry(self, job_id: str, index: int) -> dict:
        info = self.progress(job_id)
        if not info.get("ok"):
            return info
        if "progress" not in info:
            return {"ok": False, "error": "this job has no progress snapshot yet — units appear once the fan-out starts dispatching"}

        units = info["progress"].get("units")
        if units is None:
            return {"ok": False, "error": "this progress snapshot has no unit registry — it was written by an older run"}
        if not 0 <= index < len(units):
            return {"ok": False, "error": f"unit index {index} is out of range, the registry holds {len(units)} units"}

        unit = units[index]
        path = Path(unit["log"])
        path = path if path.is_absolute() else self.paths.repo_root / path
        return {"ok": True, "unit": unit, "path": path}

    def _unit_live(self, job_id: str, index: int, stop: threading.Event) -> bool:
        if stop.is_set():
            return False

        with self.lock:
            record = self.jobs.get(job_id)
        if record is None or record["status"] != "running":
            return False

        entry = self.unit_entry(job_id, index)
        return entry.get("ok", False) and entry["unit"]["status"] in ("queued", "running")

    def unit_stream(self, job_id: str, index: int) -> dict:
        entry = self.unit_entry(job_id, index)
        if not entry.get("ok"):
            return entry

        unit   = entry["unit"]
        path   = entry["path"]
        stream = JobStream(self.logger)
        stop   = threading.Event()

        stream.publish({"type": "unit", "name": unit["name"], "log": str(path), "status": unit["status"], "gpu": unit["gpu"]})

        if not path.is_file():
            stream.publish({"type": "status", "status": unit["status"], "code": None, "missing": True})
            stream.publish({"type": "end"})
            return {"ok": True, "stream": stream, "stop": stop}

        worker = threading.Thread(target=self._follow_unit, args=(job_id, index, path, stream, stop), daemon=True)
        worker.start()
        return {"ok": True, "stream": stream, "stop": stop}

    def _follow_unit(self, job_id: str, index: int, path: Path, stream: JobStream, stop: threading.Event) -> None:
        LogTailer(path, stream, partial(self._unit_live, job_id, index, stop)).follow()

        entry = self.unit_entry(job_id, index)
        unit  = entry["unit"] if entry.get("ok") else {}
        stream.publish({"type": "status", "status": unit.get("status", "unknown"), "code": unit.get("returncode")})
        stream.publish({"type": "end"})

    def job_log(self, job_id: str, unit: int | None = None) -> dict:
        with self.lock:
            record = self.jobs.get(job_id)

        if record is None:
            return {"ok": False, "error": "unknown job"}

        if unit is None:
            path = record.get("log_path")
            if not path:
                return {"ok": True, "available": False, "reason": "this job writes to a pipe or a terminal, not to a log file"}
            target = Path(path)
        else:
            entry = self.unit_entry(job_id, unit)
            if not entry.get("ok"):
                return entry
            target = entry["path"]
            if not target.is_file():
                return {"ok": True, "available": False, "reason": f"{entry['unit']['name']} has not started yet — its log file does not exist"}

        if not target.is_file():
            return {"ok": False, "error": f"log file is gone: {target}", "path": str(target)}

        size = target.stat().st_size
        with open(target, "rb") as handle:
            handle.seek(max(0, size - self.LOG_VIEW_BYTES))
            raw = handle.read()

        return {
            "ok"        : True,
            "available" : True,
            "path"      : str(target),
            "size"      : size,
            "shown"     : len(raw),
            "truncated" : size > self.LOG_VIEW_BYTES,
            "text"      : AnsiTranscript().flatten(raw.decode("utf-8", errors="replace")),
        }

    def set_gpus(self, job_id: str, gpus, park: bool = False, preempt=None) -> dict:
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
            edit = GpuPoolFile.parse({"gpus": gpus, "preempt": [] if preempt is None else preempt})
        except (ValueError, TypeError) as error:
            return {"ok": False, "error": str(error)}

        if not edit.gpus and not park:
            return {"ok": False, "error": "select at least one GPU, or confirm parking to hold the experiment"}

        try:
            GpuPoolFile(pool, self.logger).write(edit.gpus, edit.preempt)
        except OSError as exc:
            return {"ok": False, "error": f"failed to write pool file: {exc}"}

        if edit.preempt:
            self.logger.warning(f"job {job_id} frees GPU {edit.preempt} now — the runs on them stop and restart from the head of the queue")

        if edit.gpus:
            self.logger.ok(f"job {job_id} GPU pool set to {edit.gpus}")
        else:
            self.logger.warning(f"job {job_id} parked — no GPUs left in the pool, nothing new starts until a device is added back")

        return {"ok": True, "gpus": edit.gpus, "parked": not edit.gpus, "preempted": edit.preempt}

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
                known = self.jobs.get(record["job_id"])
                if known is not None:
                    known["stopped"] = True
            self._signal_group(record["pid"], signal.SIGTERM)
            self.logger.warning(f"watchdog stop for job {record['job_id']} (pid {record['pid']})")

        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not self._still_running(running):
                return len(running)
            time.sleep(0.5)

        for record in self._still_running(running):
            self._signal_group(record["pid"], signal.SIGKILL)
            self.logger.error(f"force kill for job {record['job_id']} (pid {record['pid']})")

        return len(running)

    def _still_running(self, snapshot: list[dict]) -> list[dict]:
        alive = []

        with self.lock:
            for record in snapshot:
                known = self.jobs.get(record["job_id"])
                if known is not None and known["status"] == "running":
                    alive.append(record)

        return alive

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

        owner = self._ancestor_match(pid, set(running), {})

        return None if owner is None else running[owner]

    def add_energy(self, job_id: str, wh: float) -> None:
        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                return
            record["energy_wh"] = round(record["energy_wh"] + wh, 3)

    def job_fate(self, job_id: str, pid: int) -> str:
        with self.lock:
            record = dict(self.jobs.get(job_id) or {})

        if not record:
            return "unknown"
        if record["status"] == "running":
            return "pending" if record["pid"] == pid else "released"
        if record["status"] == "failed":
            if record.get("stopped"):
                return "stopped"
            return "finished" if record.get("verdict") == "error" else "crashed"
        if record["status"] == "finished":
            return "finished" if record["exit_code"] == 0 or record.get("verdict") == "ok" else "unknown"
        return "unknown"

    def job_record(self, job_id: str) -> dict | None:
        with self.lock:
            record = self.jobs.get(job_id)
            return dict(record) if record else None

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

    def targets(self) -> list[int]:
        return self._targets(self._spared_pids())

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
        targets = self.targets()

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

    APPLY_TIMEOUT_S  = 30.0
    ESCAPE_TIMEOUT_S = 5.0
    WAKE_S           = 0.25
    LOG_NAME         = "webui_server.log"
    SCOPE_PREFIX     = "ml-template-webui"
    SERVE_SCRIPT     = Path(__file__).resolve().with_name("serve.py")
    SHIELDED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)

    def __init__(self, paths: ProjectPaths, logger: WebLogger) -> None:
        self.paths        = paths
        self.logger       = logger
        self.log_path     = paths.logs_dir / self.LOG_NAME
        self.scope        = ""
        self.shield_error = ""
        self.requested    = threading.Event()
        self.applied      = threading.Event()
        self.stopping     = threading.Event()

    def state(self) -> dict:
        return {"detached": self.applied.is_set(), "pid": os.getpid(), "log_path": str(self.log_path), "scope": self.scope, "shield_error": self.shield_error}

    def detach(self) -> dict:
        if not self.applied.is_set():
            self.requested.set()
            if not self.applied.wait(self.APPLY_TIMEOUT_S):
                return {"ok": False, "error": "detach was not applied by the main thread"}
        return {"ok": True, **self.state()}

    def sibling_pids(self) -> list[int]:
        own  = os.getpid()
        uid  = os.getuid()
        pids = []

        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) == own:
                continue

            pid = int(entry)
            try:
                if os.stat(f"/proc/{pid}").st_uid != uid:
                    continue
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except OSError:
                continue

            if self._served_root(argv) == self.paths.repo_root:
                pids.append(pid)

        return sorted(pids)

    def _served_root(self, argv: list[bytes]) -> Path | None:
        parts = [part.decode(errors="replace") for part in argv if part]
        if not any(part.endswith("serve.py") and Path(part).resolve() == self.SERVE_SCRIPT for part in parts):
            return None

        for index, part in enumerate(parts):
            if part == "--root" and index + 1 < len(parts):
                return Path(parts[index + 1]).resolve()
            if part.startswith("--root="):
                return Path(part[len("--root="):]).resolve()

        return self.SERVE_SCRIPT.parent.parent

    def kill_siblings(self) -> list[int]:
        killed = []

        for pid in self.sibling_pids():
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                continue

        if killed:
            self.logger.warning(f"kill front end: SIGKILL sent to {len(killed)} other console server(s) on this root, pids {', '.join(map(str, killed))}")

        return killed

    def shutdown(self) -> None:
        self.stopping.set()
        self.requested.set()

    def _shield_handler(self, signum: int, frame) -> None:
        self.logger.warning(f"{signal.Signals(signum).name} ignored by the keep-alive shield; stop the server through the console's kill front end button or SIGKILL")

    def _escape_session(self) -> None:
        scope = f"{self.SCOPE_PREFIX}-{os.getpid()}.scope"
        argv  = ["busctl", "--user", "call", "org.freedesktop.systemd1", "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager", "StartTransientUnit", "ssa(sv)a(sa(sv))", scope, "fail", "2", "PIDs", "au", "1", str(os.getpid()), "CollectMode", "s", "inactive-or-failed", "0"]

        try:
            moved = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.shield_error = f"session escape failed: {exc}"
            self.logger.error(self.shield_error)
            return
        if moved.returncode != 0:
            self.shield_error = f"session escape failed: {moved.stderr.strip() or moved.stdout.strip() or f'busctl exited {moved.returncode}'}"
            self.logger.error(self.shield_error)
            return

        deadline = time.monotonic() + self.ESCAPE_TIMEOUT_S
        while True:
            try:
                cgroup_text = Path("/proc/self/cgroup").read_text()
            except OSError as exc:
                self.shield_error = f"failed to read /proc/self/cgroup: {exc}"
                self.logger.error(self.shield_error)
                return

            if scope in cgroup_text:
                break
            if time.monotonic() > deadline:
                self.shield_error = f"session escape did not land within {self.ESCAPE_TIMEOUT_S:.0f}s: process still in {cgroup_text.strip()}"
                self.logger.error(self.shield_error)
                return
            time.sleep(0.1)

        self.scope = scope

        try:
            linger = subprocess.run(["loginctl", "enable-linger"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.shield_error = f"lingering could not be enabled: {exc}"
            self.logger.error(self.shield_error)
            return
        if linger.returncode != 0:
            self.shield_error = f"lingering could not be enabled: {linger.stderr.strip() or linger.stdout.strip() or f'loginctl exited {linger.returncode}'}"
            self.logger.error(self.shield_error)

    def _apply(self) -> None:
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        handle = open(self.log_path, "ab", buffering=0)

        for signum in self.SHIELDED_SIGNALS:
            signal.signal(signum, self._shield_handler)
        os.dup2(handle.fileno(), sys.stdout.fileno())
        os.dup2(handle.fileno(), sys.stderr.fileno())
        handle.close()

        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, sys.stdin.fileno())
        os.close(devnull)

        self._escape_session()
        self.logger.enabled = False

    def wait_loop(self) -> None:
        while True:
            if not self.requested.wait(self.WAKE_S):
                continue
            self.requested.clear()
            if self.stopping.is_set():
                self.logger.warning(f"front end shutdown requested from the console (pid {os.getpid()}); running jobs are their own session leaders and keep going")
                return
            if not self.applied.is_set():
                self._apply()
                self.applied.set()
                shield = f"shielded in {self.scope}" if self.scope and not self.shield_error else f"shield incomplete: {self.shield_error}"
                self.logger.warning(f"server detached from the terminal (pid {os.getpid()}), {shield}, output continues in {self.log_path}")
