"""On-disk store of reusable configuration templates captured from the config forms.

Each template is one JSON file named after its identifier, holding the entry
point it was captured on and the config overrides the form carried at save time.
A template is imported back into a form rather than launched: it seeds the
fields of a new run and leaves the interpreter, execution and finalize controls
of the page importing it untouched, which is what separates it from a saved run.
Names are the handle the templates are retrieved by, so they are unique per
entry point and a save over an existing name has to say that it replaces it.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime

from web_logger import WebLogger


class ConfigTemplateStore:
    """Persists, lists, renames and deletes config templates.

    Attributes:
        paths     : Project paths providing the store directory and entry-point checks.
        logger    : Console logger.
        lock      : Guards the directory against concurrent readers and writers.
        directory : Directory holding one JSON file per template.
    """

    ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")

    def __init__(self, paths, logger: WebLogger) -> None:
        """Stores the paths and logger and creates the template directory."""
        self.paths     = paths
        self.logger    = logger
        self.lock      = threading.Lock()
        self.directory = paths.templates_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict) -> dict:
        """Writes one template built from a config form's current overrides.

        Args:
            payload : Form contents with `script_key`, `name`, optional `title` and `overrides`, and `replace` set when an existing template of that name is meant to be overwritten.

        Returns:
            Result carrying the stored entry and whether it replaced one, or `ok`
            False with an error when the entry point is unknown, the name is blank
            or the name is already taken on that entry point without `replace`.
        """
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
        """Returns every template, newest first by save time and identifier."""
        entries = self._entries()
        entries.sort(key=lambda entry: (entry["saved_at"], entry["template_id"]), reverse=True)
        return {"templates": entries}

    def get(self, template_id: str) -> dict | None:
        """Returns one template, or None when the identifier is malformed or absent."""
        if not self.ID_PATTERN.match(template_id):
            return None

        with self.lock:
            path = self._path(template_id)
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

    def rename(self, template_id: str, raw_name: str) -> dict:
        """Gives one template a new name, keeping its identifier and overrides.

        Args:
            template_id : Identifier of the template to rename.
            raw_name    : New name, trimmed before it is stored.

        Returns:
            Result carrying the renamed entry, or `ok` False with an error when the
            name is blank, the template is absent or the name is taken by another
            template on the same entry point.
        """
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
        """Deletes one template, or reports that it was not found."""
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
        """Returns the template of one entry point carrying this name, or None."""
        for entry in self._entries():
            if entry["script"] == script and entry["name"] == name:
                return entry
        return None

    def _entries(self) -> list:
        """Returns every stored template, in directory order."""
        with self.lock:
            return [json.loads(path.read_text(encoding="utf-8")) for path in self.directory.glob("*.json")]

    def _path(self, template_id: str):
        """Returns the JSON file path holding the template with this identifier."""
        return self.directory / f"{template_id}.json"
