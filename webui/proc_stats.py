
from __future__ import annotations

import os
import pwd
import threading
import time


class MemInfo:

    @staticmethod
    def fields() -> dict:
        info = {}
        try:
            for line in open("/proc/meminfo").read().splitlines():
                key, _, rest = line.partition(":")
                info[key]    = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return {}
        return info

    @staticmethod
    def used_percent() -> float | None:
        info  = MemInfo.fields()
        total = info.get("MemTotal", 0)
        if total <= 0:
            return None
        return 100.0 * (total - info.get("MemAvailable", 0)) / total


class CpuCounters:

    @staticmethod
    def read() -> dict:
        counters = {}
        try:
            lines = open("/proc/stat").read().splitlines()
        except OSError:
            return counters

        for line in lines:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            vals  = [int(v) for v in parts[1:9]]
            busy  = sum(vals) - vals[3] - vals[4]
            counters[parts[0]] = (busy, sum(vals))

        return counters


class PssCache:

    TTL_S   = 5.0
    PRUNE_S = 60.0

    def __init__(self) -> None:
        self.lock     = threading.Lock()
        self.entries  = {}
        self.pruned_t = 0.0

    def _read(self, pid: int) -> int | None:
        try:
            for line in open(f"/proc/{pid}/smaps_rollup"):
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _prune(self, now: float) -> None:
        if now - self.pruned_t < self.PRUNE_S:
            return

        self.pruned_t = now
        self.entries  = {pid: entry for pid, entry in self.entries.items() if now - entry[0] < self.PRUNE_S}

    def value(self, pid: int) -> int | None:
        now = time.monotonic()

        with self.lock:
            entry = self.entries.get(pid)
            if entry is not None and now - entry[0] < self.TTL_S:
                return entry[1]

        pss = self._read(pid)

        with self.lock:
            self.entries[pid] = (time.monotonic(), pss)
            self._prune(now)

        return pss


class ProcStats:

    PAGE  = os.sysconf("SC_PAGE_SIZE")
    CACHE = PssCache()

    @staticmethod
    def username(uid: int) -> str:
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)

    @staticmethod
    def stat(pid: int) -> dict | None:
        try:
            raw    = open(f"/proc/{pid}/stat").read()
            close  = raw.rindex(")")
            fields = raw[close + 2 :].split()
            return {
                "comm"    : raw[raw.index("(") + 1 : close],
                "state"   : fields[0],
                "ppid"    : int(fields[1]),
                "jiffies" : int(fields[11]) + int(fields[12]),
                "threads" : int(fields[17]),
                "started" : fields[19],
                "rss"     : int(fields[21]) * ProcStats.PAGE,
            }
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def pss(pid: int) -> int | None:
        return ProcStats.CACHE.value(pid)

    @staticmethod
    def private(pid: int) -> int | None:
        try:
            parts    = open(f"/proc/{pid}/statm").read().split()
            resident = int(parts[1])
            shared   = int(parts[2])
        except (OSError, ValueError, IndexError):
            return None
        return max(0, resident - shared) * ProcStats.PAGE

    @staticmethod
    def attributed(pid: int) -> int | None:
        pss = ProcStats.pss(pid)
        if pss is not None:
            return pss
        return ProcStats.private(pid)

    @staticmethod
    def ppid(pid: int) -> int:
        stat = ProcStats.stat(pid)
        return stat["ppid"] if stat is not None else 0


class ProcSweep:

    @staticmethod
    def rows() -> list[dict]:
        rows = []

        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue

            pid = int(entry)
            try:
                uid = os.stat(f"/proc/{pid}").st_uid
            except OSError:
                continue

            stat = ProcStats.stat(pid)
            if stat is None:
                continue

            rows.append({"pid": pid, "uid": uid, "stat": stat, "attributed": ProcStats.attributed(pid)})

        return rows


class UserSlices:

    ROOT      = "/sys/fs/cgroup/user.slice"
    PSI_ROOT  = "/proc/pressure"
    RESOURCES = (("cpu", "cpu"), ("mem", "memory"), ("io", "io"))

    @staticmethod
    def host_available() -> bool:
        return os.path.isfile(os.path.join(UserSlices.PSI_ROOT, "cpu"))

    @staticmethod
    def available() -> bool:
        return os.path.isdir(UserSlices.ROOT)

    @staticmethod
    def pressure(path: str) -> dict:
        out = {"some": 0.0, "full": 0.0}
        for line in open(path).read().splitlines():
            head, *tokens = line.split()
            fields        = dict(token.split("=") for token in tokens if "=" in token)
            if head in out:
                out[head] = float(fields["avg10"])
        return out

    @staticmethod
    def host_pressure() -> dict:
        return {key: UserSlices.pressure(os.path.join(UserSlices.PSI_ROOT, name)) for key, name in UserSlices.RESOURCES}

    @staticmethod
    def self_slice_uid() -> int | None:
        try:
            line = open("/proc/self/cgroup").read()
        except OSError:
            return None

        marker = "/user.slice/user-"
        start  = line.find(marker)
        if start < 0:
            return None
        digits = line[start + len(marker):].split(".slice", 1)[0]
        return int(digits) if digits.isdigit() else None

    @staticmethod
    def _counter(path: str, keys: tuple) -> dict:
        out = {key: 0 for key in keys}
        for line in open(path).read().splitlines():
            key, _, value = line.partition(" ")
            if key in out:
                out[key] = int(value)
        return out

    @staticmethod
    def rows() -> list[dict]:
        rows = []

        for entry in os.listdir(UserSlices.ROOT):
            if not (entry.startswith("user-") and entry.endswith(".slice")):
                continue
            digits = entry[len("user-"):-len(".slice")]
            if not digits.isdigit():
                continue

            base = os.path.join(UserSlices.ROOT, entry)
            try:
                cpu   = UserSlices._counter(os.path.join(base, "cpu.stat"), ("usage_usec",))
                mem   = UserSlices._counter(os.path.join(base, "memory.stat"), ("anon", "shmem"))
                stall = {key: UserSlices.pressure(os.path.join(base, f"{name}.pressure")) for key, name in UserSlices.RESOURCES}
            except OSError:
                continue

            rows.append({"uid": int(digits), "cpu_usec": cpu["usage_usec"], "mem": mem["anon"] + mem["shmem"], "stall": stall})

        return rows
