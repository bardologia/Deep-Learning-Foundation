
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime

from web_logger import WebLogger


class ConfigTemplateStore:

    ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")

    def __init__(self, paths, logger: WebLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.lock      = threading.Lock()
        self.directory = paths.templates_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict) -> dict:
        key = str(payload.get("script_key", ""))
        if not self.paths.has_script(key):
            return {"ok": False, "error": f"unknown script '{key}'"}

        name = str(payload.get("name", "")).strip()
        if not name:
            return {"ok": False, "error": "no template name given"}

        existing = self._named(key, name)
        if existing is not None and not bool(payload.get("replace")):
            return {"ok": False, "error": f"a template named '{name}' already exists for {key}"}

        overrides = {str(path): str(value) for path, value in dict(payload.get("overrides") or {}).items()}

        entry = {
            "template_id" : existing["template_id"] if existing is not None else uuid.uuid4().hex[:12],
            "script"      : key,
            "title"       : str(payload.get("title", "")).strip() or key,
            "name"        : name,
            "overrides"   : overrides,
            "saved_at"    : datetime.now().isoformat(timespec="seconds"),
        }

        with self.lock:
            self._path(entry["template_id"]).write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")

        self.logger.ok(f"{'replaced' if existing is not None else 'saved'} config template '{name}' ({entry['template_id']}) for {key} with {len(overrides)} override(s)")
        return {"ok": True, "entry": entry, "replaced": existing is not None}

    def list(self) -> dict:
        entries = self._entries()
        entries.sort(key=lambda entry: (entry["saved_at"], entry["template_id"]), reverse=True)
        return {"templates": entries}

    def get(self, template_id: str) -> dict | None:
        if not self.ID_PATTERN.match(template_id):
            return None

        with self.lock:
            path = self._path(template_id)
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

    def rename(self, template_id: str, raw_name: str) -> dict:
        name = str(raw_name).strip()
        if not name:
            return {"ok": False, "error": "no template name given"}

        entry = self.get(template_id)
        if entry is None:
            return {"ok": False, "error": "template not found"}

        clash = self._named(entry["script"], name)
        if clash is not None and clash["template_id"] != template_id:
            return {"ok": False, "error": f"a template named '{name}' already exists for {entry['script']}"}

        renamed = {**entry, "name": name}

        with self.lock:
            self._path(template_id).write_text(json.dumps(renamed, indent=2) + "\n", encoding="utf-8")

        self.logger.ok(f"renamed config template {template_id} to '{name}'")
        return {"ok": True, "entry": renamed}

    def delete(self, template_id: str) -> dict:
        if not self.ID_PATTERN.match(template_id):
            return {"ok": False, "error": "template not found"}

        with self.lock:
            path = self._path(template_id)
            if not path.is_file():
                return {"ok": False, "error": "template not found"}
            entry = json.loads(path.read_text(encoding="utf-8"))
            path.unlink()

        self.logger.muted(f"deleted config template '{entry['name']}' ({template_id})")
        return {"ok": True}

    def _named(self, script: str, name: str) -> dict | None:
        for entry in self._entries():
            if entry["script"] == script and entry["name"] == name:
                return entry
        return None

    def _entries(self) -> list:
        with self.lock:
            return [json.loads(path.read_text(encoding="utf-8")) for path in self.directory.glob("*.json")]

    def _path(self, template_id: str):
        return self.directory / f"{template_id}.json"
