from __future__ import annotations

import sys
from datetime import datetime


class WebLogger:

    COLORS = {
        "INFO"  : "\033[36m",
        "OK"    : "\033[32m",
        "WARN"  : "\033[33m",
        "ERROR" : "\033[31m",
        "MUTED" : "\033[90m",
    }
    RESET = "\033[0m"

    def __init__(self, name: str = "webui") -> None:
        self.name    = name
        self.enabled = sys.stdout.isatty()

    def _emit(self, level: str, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        color = self.COLORS.get(level, "") if self.enabled else ""
        reset = self.RESET if self.enabled else ""
        line  = f"{color}[{stamp}] {level:<5}{reset} {message}"
        print(line, flush=True)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def ok(self, message: str) -> None:
        self._emit("OK", message)

    def warning(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def muted(self, message: str) -> None:
        self._emit("MUTED", message)

    def banner(self, title: str, lines: list[str]) -> None:
        width = max([len(title)] + [len(item) for item in lines]) + 4
        bar   = "=" * width
        self._emit("OK", bar)
        self._emit("OK", f"  {title}")
        self._emit("OK", bar)
        for item in lines:
            self._emit("INFO", f"  {item}")
