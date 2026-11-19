
from __future__ import annotations

import json
import subprocess
import threading

from web_logger import WebLogger


class JobPoliteness:

    SETTINGS_NAME = "politeness.json"
    LEVELS        = {
        "off"        : {"nice": 0,  "io_class": None, "io_level": None, "label": "no nice, no ionice"},
        "polite"     : {"nice": 10, "io_class": 2,    "io_level": 7,    "label": "nice 10, ionice best-effort 7"},
        "background" : {"nice": 19, "io_class": 3,    "io_level": None, "label": "nice 19, ionice idle"},
    }
    DEFAULTS      = {"level": "polite", "auto_throttle": True}

    def __init__(self, paths, logger: WebLogger) -> None:
        self.paths    = paths
        self.logger   = logger
        self.lock     = threading.Lock()
        self.path     = paths.logs_dir / self.SETTINGS_NAME
        self.settings = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return dict(self.DEFAULTS)

        loaded = json.loads(self.path.read_text())
        level  = loaded["level"]
        if level not in self.LEVELS:
            raise ValueError(f"{self.path} names unknown politeness level {level!r}")
        return {"level": level, "auto_throttle": bool(loaded["auto_throttle"])}

    def level(self) -> str:
        with self.lock:
            return self.settings["level"]

    def auto_throttle(self) -> bool:
        with self.lock:
            return self.settings["auto_throttle"]

    def prefix(self) -> list[str]:
        spec = self.LEVELS[self.level()]
        if spec["nice"] == 0:
            return []

        argv = ["nice", "-n", str(spec["nice"]), "ionice", "-c", str(spec["io_class"])]
        if spec["io_level"] is not None:
            argv += ["-n", str(spec["io_level"])]
        return argv

    def state(self) -> dict:
        with self.lock:
            settings = dict(self.settings)
        spec = self.LEVELS[settings["level"]]
        return {**settings, "nice": spec["nice"], "label": spec["label"], "prefix": " ".join(self.prefix()), "levels": list(self.LEVELS), "settings_path": str(self.path)}

    def configure(self, payload: dict) -> dict:
        with self.lock:
            current = dict(self.settings)

        level    = str(payload.get("level", current["level"]))
        throttle = bool(payload.get("auto_throttle", current["auto_throttle"]))

        if level not in self.LEVELS:
            return {"ok": False, "error": f"unknown politeness level {level!r}; choose one of {', '.join(self.LEVELS)}"}

        with self.lock:
            self.settings = {"level": level, "auto_throttle": throttle}
            self._persist()

        self.logger.ok(f"politeness set to {level} ({self.LEVELS[level]['label']}), auto-throttle {'on' if throttle else 'off'}")
        return {"ok": True, **self.state()}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.settings, indent=2) + "\n")


class JobThrottle:

    NICE      = 19
    IO_CLASS  = 3
    TIMEOUT_S = 15.0

    def __init__(self, nuke, logger: WebLogger) -> None:
        self.nuke   = nuke
        self.logger = logger

    def _run(self, argv: list[str]) -> tuple[int, str]:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=self.TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, str(exc)

        lines = done.stderr.strip().splitlines()
        return done.returncode, (lines[-1] if lines else "")

    def apply(self) -> dict:
        targets = self.nuke.targets()
        if not targets:
            self.logger.muted("throttle requested, no processes to lower")
            return {"ok": True, "processes": 0, "renice": 0, "ionice": 0, "error": ""}

        pids           = [str(pid) for pid in targets]
        renice, err_r  = self._run(["renice", "-n", str(self.NICE), "-p", *pids])
        ionice, err_i  = self._run(["ionice", "-c", str(self.IO_CLASS), "-p", *pids])
        error          = err_r or err_i
        ok             = renice == 0 and ionice == 0

        if ok:
            self.logger.warning(f"THROTTLE: {len(targets)} user processes dropped to nice {self.NICE} and idle I/O")
        else:
            self.logger.error(f"THROTTLE failed on {len(targets)} processes (renice {renice}, ionice {ionice}): {error}")

        return {"ok": ok, "processes": len(targets), "renice": renice, "ionice": ionice, "error": error}
