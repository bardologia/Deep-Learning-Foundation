
from __future__ import annotations

import time

from telemetry_log import BucketedTelemetryLog


class GpuActivityLog(BucketedTelemetryLog):

    BUSY_UTIL = 5.0
    FILE_NAME = "gpu_activity.jsonl"
    LABEL     = "gpu activity"

    def _states(self) -> list[dict] | None:
        occupancy = self.monitor.gpu_occupancy()
        if occupancy is None:
            return None

        states = []
        for device in occupancy:
            if device["index"] is None:
                continue

            holders = []
            mine    = False
            others  = False

            for proc in device["procs"]:
                owner = proc["owner"]
                if owner is None:
                    continue

                if owner == self.monitor.user:
                    mine = True
                else:
                    others = True

                if owner not in holders:
                    holders.append(owner)

            util = device["util"] if device["util"] is not None else 0.0
            states.append({
                "index"   : device["index"],
                "name"    : device["name"],
                "util"    : util,
                "busy"    : bool(device["procs"]) or util >= self.BUSY_UTIL,
                "mine"    : mine,
                "others"  : others,
                "holders" : holders,
            })
        return states

    def _accumulate(self, start: int, states: list[dict]) -> None:
        bucket = self.buckets.setdefault(start, {"gpus": {}})
        for state in states:
            key   = str(state["index"])
            entry = bucket["gpus"].setdefault(key, {"name": state["name"], "n": 0, "busy": 0, "mine": 0, "others": 0, "util": 0.0, "holders": []})

            entry["name"]    = state["name"]
            entry["n"]      += 1
            entry["busy"]   += 1 if state["busy"] else 0
            entry["mine"]   += 1 if state["mine"] else 0
            entry["others"] += 1 if state["others"] else 0
            entry["util"]   += state["util"]

            for holder in state["holders"]:
                if holder not in entry["holders"]:
                    entry["holders"].append(holder)

            self.names[key] = state["name"]

    def sample(self) -> None:
        states = self._states()
        if states is None:
            return

        self._fold(time.time(), states)

    def state(self, span: str = "day") -> dict:
        window_s, step_s = self.SPANS[span]

        with self.lock:
            now    = time.time()
            cutoff = now - window_s

            merged = {}
            for start, bucket in sorted(self.buckets.items()):
                if start + self.BUCKET_S <= cutoff:
                    continue

                slot = int(start // step_s) * step_s
                for key, entry in bucket["gpus"].items():
                    acc = merged.setdefault(key, {}).setdefault(slot, {"n": 0, "busy": 0, "mine": 0, "others": 0, "util": 0.0, "holders": []})

                    acc["n"]      += entry["n"]
                    acc["busy"]   += entry["busy"]
                    acc["mine"]   += entry["mine"]
                    acc["others"] += entry["others"]
                    acc["util"]   += entry["util"]

                    for holder in entry["holders"]:
                        if holder not in acc["holders"]:
                            acc["holders"].append(holder)

            devices = {}
            for key, slots in merged.items():
                for slot, acc in sorted(slots.items()):
                    count = acc["n"] or 1
                    devices.setdefault(key, []).append({
                        "t"       : slot,
                        "busy"    : round(acc["busy"] / count, 3),
                        "mine"    : round(acc["mine"] / count, 3),
                        "others"  : round(acc["others"] / count, 3),
                        "util"    : round(acc["util"] / count, 1),
                        "holders" : list(acc["holders"]),
                    })

            return {
                "span"     : span,
                "bucket_s" : step_s,
                "window_s" : window_s,
                "now"      : now,
                "gpus"     : [{"index": int(key), "name": self.names.get(key, ""), "buckets": buckets} for key, buckets in sorted(devices.items(), key=lambda item: int(item[0]))],
            }
