
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque

from gpu_activity  import GpuActivityLog
from host_history  import HostHistoryLog
from proc_stats    import CpuCounters, MemInfo, ProcStats, ProcSweep
from telemetry_log import CpuBusyMeter, TelemetrySampler
from web_logger    import WebLogger


class GpuProbe:

    DEVICE_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,uuid"
    APP_QUERY    = "pid,used_gpu_memory,gpu_uuid"
    CACHE_TTL_S  = 0.5
    TIMEOUT_S    = 3

    def __init__(self, logger: WebLogger) -> None:
        self.logger    = logger
        self.lock      = threading.Lock()
        self.cache     = {}
        self.installed = shutil.which("nvidia-smi") is not None
        self.failed    = False

    def devices(self) -> list[list[str]] | None:
        return self._cached(f"--query-gpu={self.DEVICE_QUERY}", 9)

    def apps(self) -> list[list[str]] | None:
        return self._cached(f"--query-compute-apps={self.APP_QUERY}", 3)

    def _cached(self, flag: str, width: int) -> list[list[str]] | None:
        now = time.monotonic()

        with self.lock:
            entry = self.cache.get(flag)
            if entry is not None and now - entry[0] < self.CACHE_TTL_S:
                return entry[1]

        rows = self._rows(flag, width)

        with self.lock:
            self.cache[flag] = (time.monotonic(), rows)

        return rows

    def _rows(self, flag: str, width: int) -> list[list[str]] | None:
        if not self.installed:
            return []

        try:
            out = subprocess.run(["nvidia-smi", flag, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=self.TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as error:
            self._failed(f"nvidia-smi {flag} did not answer ({error})")
            return None

        if out.returncode != 0:
            self._failed(f"nvidia-smi {flag} exited with code {out.returncode} ({out.stderr.strip() or 'no stderr'})")
            return None

        rows = []
        for line in out.stdout.strip().splitlines():
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) >= width:
                rows.append(cells)

        self._recovered()
        return rows

    def _failed(self, reason: str) -> None:
        with self.lock:
            first       = not self.failed
            self.failed = True

        if first:
            self.logger.error(f"{reason}: GPU occupancy is unknown, so the schedule, charity and intrusion checks stand down until it answers again")

    def _recovered(self) -> None:
        with self.lock:
            recovered   = self.failed
            self.failed = False

        if recovered:
            self.logger.ok("nvidia-smi answers again, GPU occupancy is known")


class SystemMonitor:

    PROC_LIMIT   = 30
    DU_REFRESH_S = 600.0

    def __init__(self, paths, logger: WebLogger) -> None:
        self.paths       = paths
        self.logger      = logger
        self.lock        = threading.Lock()
        self.probe       = GpuProbe(logger)
        self.prev_cpu    = {}
        self.prev_proc   = {}
        self.prev_proc_t = 0.0
        self.uid         = os.getuid()
        self.user        = ProcStats.username(self.uid)
        self.clk         = os.sysconf("SC_CLK_TCK")
        self.page        = os.sysconf("SC_PAGE_SIZE")
        self.user_root   = self._user_root()
        self.du_usage    = {"user": None, "repo": None}
        self.history     = SystemHistory(self)
        self.users       = ActiveUsers(self)
        self.activity    = GpuActivityLog(self, paths, logger)
        self.host_hist   = HostHistoryLog(self, paths, logger)

        threading.Thread(target=self._du_loop, daemon=True).start()
        threading.Thread(target=self.history.sample_loop, daemon=True).start()
        threading.Thread(target=self.users.sample_loop, daemon=True).start()
        threading.Thread(target=self.activity.sample_loop, daemon=True).start()
        threading.Thread(target=self.host_hist.sample_loop, daemon=True).start()

    def _cpu_percents(self) -> tuple[list[float], float]:
        cores = []
        total = 0.0

        for key, (busy, whole) in CpuCounters.read().items():
            prev = self.prev_cpu.get(key)
            self.prev_cpu[key] = (busy, whole)

            pct = 0.0
            if prev is not None and whole > prev[1]:
                pct = round(100.0 * (busy - prev[0]) / (whole - prev[1]), 1)

            if key == "cpu":
                total = pct
            else:
                cores.append(pct)

        return cores, total

    def _procs(self, gpu_mem: dict) -> list[dict]:
        now  = time.monotonic()
        dt   = now - self.prev_proc_t
        rows = []
        seen = set()

        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                if os.stat(f"/proc/{pid}").st_uid != self.uid:
                    continue
            except OSError:
                continue

            stat = ProcStats.stat(pid)
            if stat is None:
                continue

            prev = self.prev_proc.get(pid)
            cpu  = 0.0
            if prev is not None and self.prev_proc_t > 0 and dt > 0:
                cpu = max(0.0, round(100.0 * (stat["jiffies"] - prev) / self.clk / dt, 1))
            self.prev_proc[pid] = stat["jiffies"]
            seen.add(pid)

            cmd = self._pid_cmd(pid)

            rows.append({
                "pid"     : pid,
                "state"   : stat["state"],
                "cpu"     : cpu,
                "rss"     : stat["rss"],
                "threads" : stat["threads"],
                "gpu"     : gpu_mem.get(pid, 0),
                "cmd"     : (cmd or stat["comm"])[:200],
            })

        self.prev_proc_t = now
        self.prev_proc   = {p: j for p, j in self.prev_proc.items() if p in seen}

        rows.sort(key=lambda r: (-r["cpu"], -r["gpu"], -r["rss"]))
        top = rows[: self.PROC_LIMIT]

        for row in top:
            pss         = ProcStats.pss(row["pid"])
            row["pss"]  = pss if pss is not None else row["rss"]

        return top

    def _gpu_devices(self) -> list[dict] | None:
        rows = self.probe.devices()
        if rows is None:
            return None

        devices = []
        for cells in rows:
            devices.append({
                "index"       : self._num(cells[0]),
                "name"        : cells[1],
                "util"        : self._num(cells[2]),
                "mem_used"    : self._num(cells[3]),
                "mem_total"   : self._num(cells[4]),
                "temp"        : self._num(cells[5]),
                "power"       : self._num(cells[6]),
                "power_limit" : self._num(cells[7]),
                "uuid"        : cells[8],
            })
        return devices

    def _compute_apps(self) -> dict | None:
        rows = self.probe.apps()
        if rows is None:
            return None

        grouped = {}
        for cells in rows:
            try:
                pid = int(cells[0])
                mem = float(cells[1])
            except ValueError:
                continue

            grouped.setdefault(cells[2], []).append({
                "pid"   : pid,
                "mem"   : mem,
                "owner" : self.pid_owner(pid),
                "cmd"   : self._pid_cmd(pid),
            })

        return grouped

    def pid_owner(self, pid: int) -> str | None:
        try:
            uid = os.stat(f"/proc/{pid}").st_uid
        except OSError:
            return None

        stat = ProcStats.stat(pid)
        if stat is None or stat["state"] == "Z":
            return None

        return ProcStats.username(uid)

    def _pid_cmd(self, pid: int) -> str:
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            return ""
        return raw.replace(b"\x00", b" ").decode(errors="replace").strip()

    def gpu_occupancy(self) -> list[dict] | None:
        apps    = self._compute_apps()
        devices = self._gpu_devices()

        if apps is None or devices is None:
            return None
        return [{**device, "procs": apps.get(device["uuid"], [])} for device in devices]

    def _memory(self) -> dict:
        info = MemInfo.fields()
        if not info:
            return {}

        return {
            "total"      : info.get("MemTotal", 0),
            "available"  : info.get("MemAvailable", 0),
            "swap_total" : info.get("SwapTotal", 0),
            "swap_free"  : info.get("SwapFree", 0),
        }

    def _disk(self) -> dict:
        try:
            usage = shutil.disk_usage(self.paths.repo_root)
        except OSError:
            return {}

        return {
            "path"      : str(self.paths.repo_root),
            "total"     : usage.total,
            "used"      : usage.used,
            "free"      : usage.free,
            "user_path" : str(self.user_root),
            "user_used" : self.du_usage["user"],
            "repo_path" : str(self.paths.repo_root),
            "repo_used" : self.du_usage["repo"],
        }

    def _user_root(self):
        current = self.paths.repo_root.resolve()

        while current != current.parent:
            parent = current.parent
            try:
                if os.stat(parent).st_uid != self.uid:
                    break
            except OSError:
                break
            current = parent

        return current

    def _du_loop(self) -> None:
        while True:
            try:
                self.du_usage["repo"] = self._du(self.paths.repo_root)
                self.du_usage["user"] = self._du(self.user_root) if self.user_root != self.paths.repo_root else self.du_usage["repo"]
            except Exception as error:
                self.logger.error(f"disk usage sweep failed: {error}")

            time.sleep(self.DU_REFRESH_S)

    def _du(self, path) -> int | None:
        try:
            out = subprocess.run(["du", "-s", "--block-size=1", str(path)], capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired):
            return None

        try:
            return int(out.stdout.split()[0])
        except (ValueError, IndexError):
            return None

    def _uptime(self) -> float:
        try:
            return float(open("/proc/uptime").read().split()[0])
        except (OSError, ValueError, IndexError):
            return 0.0

    def _gpu_cards(self, occupancy: list[dict]) -> list[dict]:
        cards = []
        for device in occupancy:
            mine    = False
            others  = False
            stale   = False
            holders = []

            for proc in device["procs"]:
                owner = proc["owner"]
                if owner is None:
                    stale = True
                elif owner == self.user:
                    mine = True
                    if owner not in holders:
                        holders.append(owner)
                else:
                    others = True
                    if owner not in holders:
                        holders.append(owner)

            cards.append({
                "index"       : device["index"],
                "name"        : device["name"],
                "util"        : device["util"],
                "mem_used"    : device["mem_used"],
                "mem_total"   : device["mem_total"],
                "temp"        : device["temp"],
                "power"       : device["power"],
                "power_limit" : device["power_limit"],
                "mine"        : mine,
                "others"      : others,
                "stale"       : stale,
                "holders"     : holders,
            })
        return cards

    def _gpu_mem_by_pid(self, occupancy: list[dict]) -> dict:
        usage = {}
        for device in occupancy:
            for proc in device["procs"]:
                usage[proc["pid"]] = usage.get(proc["pid"], 0) + proc["mem"]
        return usage

    def _num(self, raw: str):
        try:
            value = float(raw)
        except ValueError:
            return None
        return int(value) if value.is_integer() else value

    def snapshot(self) -> dict:
        occupancy = self.gpu_occupancy()
        known     = occupancy is not None
        gpu_mem   = self._gpu_mem_by_pid(occupancy) if known else {}

        with self.lock:
            cores, total = self._cpu_percents()
            procs        = self._procs(gpu_mem)

        return {
            "host"       : socket.gethostname(),
            "user"       : self.user,
            "uptime"     : self._uptime(),
            "cpu"        : {"count": os.cpu_count() or len(cores), "total": total, "cores": cores, "load": list(os.getloadavg())},
            "mem"        : self._memory(),
            "disk"       : self._disk(),
            "gpus"       : self._gpu_cards(occupancy) if known else [],
            "gpus_known" : known,
            "procs"      : procs,
            "users"      : self.users.state(),
            "history"    : self.history.state(),
        }


class SystemHistory(TelemetrySampler):

    SAMPLE_PERIOD_S = 0.5
    GPU_PERIOD_S    = 2.0
    MAX_SAMPLES     = 144
    LABEL           = "system history"

    def __init__(self, monitor: SystemMonitor) -> None:
        self.monitor    = monitor
        self.logger     = monitor.logger
        self.lock       = threading.Lock()
        self.cpu_meter  = CpuBusyMeter()
        self.cpu        = deque(maxlen=self.MAX_SAMPLES)
        self.ram        = deque(maxlen=self.MAX_SAMPLES)
        self.gpus       = {}
        self.devices    = None
        self.devices_at = None

    def _ram_percent(self) -> float:
        mem = self.monitor._memory()
        if not mem.get("total"):
            return 0.0
        return round(100.0 * (mem["total"] - mem["available"]) / mem["total"], 1)

    def _devices(self) -> list[dict] | None:
        now = time.monotonic()
        if self.devices_at is not None and now - self.devices_at < self.GPU_PERIOD_S:
            return self.devices

        self.devices_at = now
        self.devices    = self.monitor._gpu_devices()
        return self.devices

    def _track(self, key: str) -> dict:
        return self.gpus.setdefault(key, {"util": deque(maxlen=self.MAX_SAMPLES), "mem": deque(maxlen=self.MAX_SAMPLES)})

    def sample(self) -> None:
        cpu     = self.cpu_meter.percent()
        ram     = self._ram_percent()
        devices = self._devices()

        with self.lock:
            self.cpu.append(cpu)
            self.ram.append(ram)

            if devices is None:
                return

            sampled = set()
            for device in devices:
                if device["index"] is None:
                    continue

                key   = str(device["index"])
                track = self._track(key)
                util  = device["util"] if device["util"] is not None else 0.0
                mpct  = round(100.0 * device["mem_used"] / device["mem_total"], 1) if device["mem_total"] and device["mem_used"] is not None else 0.0
                track["util"].append(util)
                track["mem"].append(mpct)
                sampled.add(key)

            for key, track in self.gpus.items():
                if key not in sampled:
                    track["util"].append(0.0)
                    track["mem"].append(0.0)

    def state(self) -> dict:
        with self.lock:
            return {
                "period_s"    : self.SAMPLE_PERIOD_S,
                "max_samples" : self.MAX_SAMPLES,
                "cpu"         : list(self.cpu),
                "ram"         : list(self.ram),
                "gpus"        : {key: {"util": list(track["util"]), "mem": list(track["mem"])} for key, track in self.gpus.items()},
            }


class ActiveUsers(TelemetrySampler):

    SAMPLE_PERIOD_S = 2.0
    MIN_UID         = 1000
    CPU_FLOOR_PCT   = 1.0
    MEM_FLOOR       = 1 << 30
    LABEL           = "active users"

    def __init__(self, monitor: SystemMonitor) -> None:
        self.monitor = monitor
        self.logger  = monitor.logger
        self.lock    = threading.Lock()
        self.prev    = {}
        self.prev_t  = 0.0
        self.rows    = []

    def _scan(self) -> tuple[dict, float]:
        now   = time.monotonic()
        dt    = now - self.prev_t
        prev  = self.prev
        cur   = {}
        users = {}

        for row in ProcSweep.rows():
            pid      = row["pid"]
            jiffies  = row["stat"]["jiffies"]
            cur[pid] = jiffies
            agg      = users.setdefault(row["uid"], {"nproc": 0, "mem": 0, "jdelta": 0})

            agg["nproc"] += 1
            agg["mem"]   += row["attributed"] or 0
            if pid in prev:
                agg["jdelta"] += max(0, jiffies - prev[pid])

        self.prev   = cur
        self.prev_t = now
        return users, dt

    def _sessions(self) -> dict:
        try:
            out = subprocess.run(["who"], capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            return {}

        if out.returncode != 0:
            return {}

        counts = {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts:
                counts[parts[0]] = counts.get(parts[0], 0) + 1
        return counts

    def _gpu_by_user(self) -> dict:
        occupancy = self.monitor.gpu_occupancy()
        if occupancy is None:
            return {}

        usage = {}
        for device in occupancy:
            for proc in device["procs"]:
                if proc["owner"] is None:
                    continue
                held         = usage.setdefault(proc["owner"], {"mem": 0.0, "gpus": set()})
                held["mem"] += proc["mem"]
                held["gpus"].add(device["index"])
        return usage

    def _rows(self, users: dict, dt: float, sessions: dict, gpu: dict) -> list[dict]:
        memory   = self.monitor._memory()
        mem_used = max(0, memory.get("total", 0) - memory.get("available", 0))
        rows     = []

        for uid, agg in users.items():
            name = ProcStats.username(uid)
            held = gpu.get(name, {"mem": 0.0, "gpus": set()})
            sess = sessions.get(name, 0)
            cpu  = round(100.0 * agg["jdelta"] / self.monitor.clk / dt, 1) if dt > 0 else 0.0

            if sess == 0 and uid < self.MIN_UID and cpu < self.CPU_FLOOR_PCT and held["mem"] <= 0 and agg["mem"] < self.MEM_FLOOR:
                continue

            rows.append({
                "user"      : name,
                "uid"       : uid,
                "me"        : uid == self.monitor.uid,
                "sessions"  : sess,
                "nproc"     : agg["nproc"],
                "cpu"       : cpu,
                "mem"       : agg["mem"],
                "mem_share" : min(100.0, round(100.0 * agg["mem"] / mem_used, 1)) if mem_used else 0.0,
                "gpu_mem"   : held["mem"],
                "gpus"      : sorted(index for index in held["gpus"] if index is not None),
            })

        rows.sort(key=lambda r: (-r["cpu"], -r["gpu_mem"], -r["mem"]))
        return rows

    def sample(self) -> None:
        users, dt = self._scan()
        rows      = self._rows(users, dt, self._sessions(), self._gpu_by_user())

        with self.lock:
            self.rows = rows

    def state(self) -> list[dict]:
        with self.lock:
            return [dict(row) for row in self.rows]

