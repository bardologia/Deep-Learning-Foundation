
from __future__ import annotations

import json
import threading
import time

from proc_stats import CpuCounters
from web_logger import WebLogger


class CpuBusyMeter:

    def __init__(self) -> None:
        self.prev = None

    def percent(self) -> float:
        current   = CpuCounters.read().get("cpu")
        prev      = self.prev
        self.prev = current

        if current is None or prev is None or current[1] <= prev[1]:
            return 0.0
        return round(100.0 * (current[0] - prev[0]) / (current[1] - prev[1]), 1)


class TelemetrySampler:

    def sample_loop(self) -> None:
        while True:
            try:
                self.sample()
            except Exception as error:
                self.logger.error(f"{self.LABEL} sampling failed: {error}")

            time.sleep(self.SAMPLE_PERIOD_S)


class BucketedTelemetryLog(TelemetrySampler):

    SAMPLE_PERIOD_S = 30.0
    BUCKET_S        = 300
    RETENTION_S     = 30 * 86400

    SPANS = {
        "day"   : (86400,      300),
        "week"  : (7 * 86400,  3600),
        "month" : (30 * 86400, 4 * 3600),
    }

    def __init__(self, monitor, paths, logger: WebLogger) -> None:
        self.monitor = monitor
        self.logger  = logger
        self.path    = paths.logs_dir / self.FILE_NAME
        self.lock    = threading.Lock()
        self.buckets = {}
        self.names   = {}
        self.open_at = None

        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return

        cutoff = time.time() - self.RETENTION_S
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            record = json.loads(line)
            if record["t"] < cutoff:
                continue

            self.buckets[record["t"]] = {key: value for key, value in record.items() if key != "t"}
            for key, entry in record["gpus"].items():
                self.names[key] = entry["name"]

    def _prune(self, now: float) -> None:
        cutoff = now - self.RETENTION_S
        for start in [start for start in self.buckets if start < cutoff]:
            del self.buckets[start]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines   = [json.dumps({"t": start, **bucket}) for start, bucket in sorted(self.buckets.items())]
        staging = self.path.with_suffix(".tmp")
        staging.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        staging.replace(self.path)

    def _fold(self, now: float, sample) -> None:
        start  = int(now // self.BUCKET_S) * self.BUCKET_S
        rolled = self.open_at is not None and start != self.open_at

        with self.lock:
            self._accumulate(start, sample)
            self._prune(now)
            self.open_at = start

            if rolled:
                self._persist()
