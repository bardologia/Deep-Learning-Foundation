from __future__ import annotations

import json
import math
import mimetypes
from pathlib      import Path
from urllib.parse import parse_qs, urlparse


class HttpExchange:

    def __init__(self, handler, method: str, raw_path: str, path: str, query: dict) -> None:
        self.handler  = handler
        self.method   = method
        self.raw_path = raw_path
        self.path     = path
        self.query    = query
        self.body     = {}

    @classmethod
    def of(cls, handler) -> HttpExchange:
        parsed = urlparse(handler.path)
        return cls(handler, handler.command, parsed.path, parsed.path.rstrip("/") or "/", parse_qs(parsed.query))

    @classmethod
    def jsonsafe(cls, value):
        if isinstance(value, dict):
            return {key: cls.jsonsafe(child) for key, child in value.items()}
        if isinstance(value, list):
            return [cls.jsonsafe(child) for child in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def text(self, name: str, default: str = "") -> str:
        return (self.query.get(name) or [default])[0]

    def texts(self, name: str) -> list[str]:
        return self.query.get(name) or []

    def integer(self, name: str, default: str = "0") -> int:
        return int(self.text(name, default))

    def number(self, name: str, default: str = "0") -> float:
        return float(self.text(name, default))

    def optional_number(self, name: str) -> float | None:
        raw = self.text(name)
        return float(raw) if raw else None

    def read_body(self) -> None:
        length = int(self.handler.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            self.body = {}
            return

        raw = self.handler.rfile.read(length)
        try:
            self.body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.body = {}

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(self.jsonsafe(payload)).encode("utf-8")

        self.handler.send_response(status)
        self.handler.send_header("Content-Type", "application/json")
        self.handler.send_header("Content-Length", str(len(data)))
        self.handler.send_header("Access-Control-Allow-Origin", "*")
        self.handler.end_headers()
        self.handler.wfile.write(data)

    def send_result(self, result: dict, error: int = 400) -> None:
        self.send_json(result, 200 if result.get("ok") else error)

    def not_found(self) -> None:
        self.send_json({"error": "not found"}, 404)

    def send_png(self, png: bytes | None) -> None:
        if png is None:
            self.not_found()
            return

        self.send_payload("image/png", png, "no-cache")

    def send_bytes(self, blob: bytes | None) -> None:
        if blob is None:
            self.not_found()
            return

        self.send_payload("application/octet-stream", blob, "no-cache")

    def send_file(self, target: Path, cache: str) -> None:
        data = target.read_bytes()
        self.send_payload(mimetypes.guess_type(str(target))[0] or "application/octet-stream", data, cache)

    def send_payload(self, content_type: str, data: bytes, cache: str) -> None:
        self.handler.send_response(200)
        self.handler.send_header("Content-Type", content_type)
        self.handler.send_header("Content-Length", str(len(data)))
        self.handler.send_header("Cache-Control", cache)
        self.handler.end_headers()
        self.handler.wfile.write(data)


class RouteConflict(RuntimeError):
    pass


class RouteTable:

    def __init__(self) -> None:
        self.exact     = {}
        self.wildcards = []

    def add(self, method: str, path: str, action) -> None:
        if (method, path) in self.exact:
            raise RouteConflict(f"{method} {path} is declared twice")

        self.exact[(method, path)] = action

    def wildcard(self, method: str, prefix: str, suffix: str, action) -> None:
        if any(entry[:3] == (method, prefix, suffix) for entry in self.wildcards):
            raise RouteConflict(f"{method} {prefix}*{suffix} is declared twice")

        self.wildcards.append((method, prefix, suffix, action))

    def action_for(self, method: str, path: str) -> tuple:
        action = self.exact.get((method, path))
        if action is not None:
            return action, None

        for candidate, prefix, suffix, action in self.wildcards:
            if candidate != method or not path.startswith(prefix) or not path.endswith(suffix):
                continue

            return action, path[len(prefix):len(path) - len(suffix)]

        return None, None

    def dispatch(self, exchange: HttpExchange) -> bool:
        action, key = self.action_for(exchange.method, exchange.path)
        if action is None:
            return False

        if key is None:
            action(exchange)
        else:
            action(exchange, key)

        return True


class SubRouter:

    def __init__(self, sections: tuple, raw_sections: tuple = ()) -> None:
        self.sections     = sections
        self.raw_sections = raw_sections
        self.table        = RouteTable()

        self.declare(self.table)

    def declare(self, table: RouteTable) -> None:
        raise NotImplementedError

    def claims_raw(self, raw_path: str) -> bool:
        return bool(self.raw_sections) and raw_path.startswith(self.raw_sections)

    def handle_raw(self, exchange: HttpExchange) -> None:
        raise NotImplementedError

    def handle(self, exchange: HttpExchange) -> bool:
        return self.table.dispatch(exchange)
