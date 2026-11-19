
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime

from finalize_chain import FinalizeChain
from web_logger     import WebLogger


class SequenceStore:

    ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")

    def __init__(self, paths, logger: WebLogger, decisions) -> None:
        self.paths     = paths
        self.logger    = logger
        self.decisions = decisions
        self.lock      = threading.Lock()
        self.directory = paths.sequences_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, name: str) -> dict:
        stamp = datetime.now().isoformat(timespec="seconds")
        entry = {
            "sequence_id" : uuid.uuid4().hex[:12],
            "name"        : str(name).strip() or f"sequence {stamp}",
            "dataset"     : {"dataset_path": "", "parameters_path": ""},
            "detach"      : True,
            "isolate"     : True,
            "blocks"      : [],
            "created_at"  : stamp,
            "updated_at"  : stamp,
        }

        with self.lock:
            self._write(entry)

        self.logger.ok(f"created sequence '{entry['name']}' ({entry['sequence_id']})")
        return {"ok": True, "entry": entry}

    def save(self, sequence_id: str, payload: dict) -> dict:
        with self.lock:
            current = self._read(sequence_id)
            if current is None:
                return {"ok": False, "error": "sequence not found"}

            blocks = [self._clean_block(block) for block in list(payload.get("blocks") or [])]
            error  = self._validate(blocks)
            if error:
                return {"ok": False, "error": error}

            error = self.decisions.apply(blocks, self.order(blocks))
            if error:
                return {"ok": False, "error": error}

            dataset = dict(payload.get("dataset") or {})
            entry   = {
                "sequence_id" : sequence_id,
                "name"        : str(payload.get("name", "")).strip() or current["name"],
                "dataset"     : {"dataset_path": str(dataset.get("dataset_path", "")), "parameters_path": str(dataset.get("parameters_path", ""))},
                "detach"      : bool(payload.get("detach")),
                "isolate"     : bool(payload.get("isolate")),
                "blocks"      : blocks,
                "created_at"  : current["created_at"],
                "updated_at"  : datetime.now().isoformat(timespec="seconds"),
            }
            self._write(entry)

        return {"ok": True, "entry": entry}

    def decide(self, sequence_id: str, block_id: str, path: str, choice: str) -> dict:
        entry = self.get(sequence_id)
        if entry is None:
            return {"ok": False, "error": "sequence not found"}

        block = next((item for item in entry["blocks"] if item["block_id"] == block_id), None)
        if block is None:
            return {"ok": False, "error": "block not found"}

        decision = next((item for item in block["decisions"] if item["path"] == path), None)
        if decision is None:
            return {"ok": False, "error": f"block '{block['title']}' does not decide '{path}'"}

        decision["choice"] = str(choice)
        if decision["choice"] and decision["choice"] not in decision["options"]:
            decision["options"].append(decision["choice"])
        result = self.save(sequence_id, entry)

        if result["ok"]:
            self.logger.ok(f"sequence {sequence_id}: '{block['name'] or block['title']}' {'decided ' + path + ' = ' + str(choice) if choice else 'reopened ' + path}")
        return result

    def _clean_block(self, block: dict) -> dict:
        return {
            "block_id"     : str(block.get("block_id") or uuid.uuid4().hex[:8]),
            "script"       : str(block.get("script", "")),
            "title"        : str(block.get("title", "")).strip() or str(block.get("script", "")),
            "name"         : str(block.get("name", "")).strip(),
            "interpreter"  : str(block.get("interpreter", "")).strip(),
            "overrides"    : {str(path): str(value) for path, value in dict(block.get("overrides") or {}).items()},
            "field_notes"  : {},
            "decisions"    : self.decisions.clean(block),
            "predecessors" : [str(pred) for pred in list(block.get("predecessors") or [])],
            "checkpoint"   : bool(block.get("checkpoint")),
            "skip"         : bool(block.get("skip")),
            "finalize"     : FinalizeChain.flags(block.get("finalize")),
            "target"       : "local",
            "sweep"        : bool(block.get("sweep")),
            "resources"    : dict(block.get("resources") or {}),
            "x"            : float(block.get("x", 0.0)),
            "y"            : float(block.get("y", 0.0)),
        }

    def _validate(self, blocks: list) -> str:
        ids = [block["block_id"] for block in blocks]
        if len(ids) != len(set(ids)):
            return "duplicate block ids"

        for block in blocks:
            if not self.paths.has_script(block["script"]):
                return f"unknown script '{block['script']}'"
            missing = [pred for pred in block["predecessors"] if pred not in ids]
            if missing:
                return f"block '{block['title']}' links to missing block(s): {', '.join(missing)}"
            if block["block_id"] in block["predecessors"]:
                return f"block '{block['title']}' links to itself"

        return "the sequence has a cycle" if self.order(blocks) is None else ""

    def order(self, blocks: list) -> list | None:
        remaining = {block["block_id"]: set(block["predecessors"]) for block in blocks}
        ordered   = []

        while remaining:
            ready = sorted(block_id for block_id, preds in remaining.items() if not preds)
            if not ready:
                return None

            ordered += ready
            for block_id in ready:
                remaining.pop(block_id)
            for preds in remaining.values():
                preds.difference_update(ready)

        return ordered

    def list(self) -> dict:
        with self.lock:
            entries = []
            for path in sorted(self.directory.glob("*.json")):
                try:
                    entry = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    self.logger.error(f"skipping unreadable sequence file {path.name}: {error}")
                    continue

                missing = [key for key in ("sequence_id", "updated_at") if not isinstance(entry, dict) or key not in entry]
                if missing:
                    self.logger.error(f"skipping unreadable sequence file {path.name}: missing {', '.join(missing)}")
                    continue

                sequence_id = str(entry["sequence_id"])

                target = self._path(sequence_id)
                if path != target:
                    if target.exists():
                        self.logger.error(f"skipping {path.name}: sequence {sequence_id} already stored as {target.name}; delete one of the two files")
                        continue
                    path.rename(target)
                    self.logger.ok(f"adopted sequence file {path.name} as {target.name}")

                entries.append(entry)

        entries.sort(key=lambda entry: (entry["updated_at"], entry["sequence_id"]), reverse=True)
        return {"sequences": entries}

    def get(self, sequence_id: str) -> dict | None:
        with self.lock:
            return self._read(sequence_id)

    def delete(self, sequence_id: str) -> dict:
        with self.lock:
            entry = self._read(sequence_id)
            if entry is None:
                return {"ok": False, "error": "sequence not found"}
            self._path(sequence_id).unlink()

        self.logger.muted(f"deleted sequence '{entry['name']}' ({sequence_id})")
        return {"ok": True}

    def _read(self, sequence_id: str) -> dict | None:
        if not self.ID_PATTERN.match(sequence_id):
            return None

        path = self._path(sequence_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, entry: dict) -> None:
        self._path(entry["sequence_id"]).write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")

    def _path(self, sequence_id: str):
        return self.directory / f"{sequence_id}.json"
