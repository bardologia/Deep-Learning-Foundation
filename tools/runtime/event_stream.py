from __future__ import annotations

import json
import sys


class JsonEventStream:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def emit(self, kind: str, payload: dict) -> None:
        line = f"{self.marker} {kind} {json.dumps(payload)}"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
