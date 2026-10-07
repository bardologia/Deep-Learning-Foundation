"""On-disk store of experiment sequences built in the console.

A sequence is a directed acyclic graph of blocks, each block one entry-point
launch with its own config overrides, linked by predecessor edges. The sequence
carries a shared dataset selection that is injected into every block at launch
time, a detach flag, an isolation flag that gives every execution its own fresh
run directory, and per-block checkpoint marks that hold the chain until
released. A block carries one finalize switch per post-experiment step (the
seed comparison, the trial comparison and the report mailing); the runner
chains the enabled steps onto its end, inside the isolated workspace when the
sequence has one and on the entry's own sweep directory otherwise. A block
may also declare the configuration axes it decides; every
save stamps those decisions onto the blocks they govern, which writes the
per-block field notes labelling config paths as ``frozen`` (an earlier block's
verdict) or ``waiting`` (a later block decides it). One JSON file per sequence
lives under ``logs/sequences``.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime

from finalize_chain import FinalizeChain
from web_logger     import WebLogger


class SequenceStore:
    """Persists, validates, lists, reads and deletes sequence definitions.

    Attributes:
        paths     : Project paths providing the store directory and entry-point checks.
        logger    : Console logger.
        decisions : Engine stamping each block's declared decision axes onto the blocks they govern.
        lock      : Guards the directory against concurrent readers and writers.
        directory : Directory holding one JSON file per sequence.
    """

    ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")

    def __init__(self, paths, logger: WebLogger, decisions) -> None:
        """Stores the paths, logger and decision engine, and creates the sequences directory."""
        self.paths     = paths
        self.logger    = logger
        self.decisions = decisions
        self.lock      = threading.Lock()
        self.directory = paths.sequences_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, name: str) -> dict:
        """Writes a new empty sequence and returns it.

        Args:
            name : Display name of the sequence; a stamp-based name when blank.

        Returns:
            Result carrying the stored entry under ``entry``.
        """
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
        """Replaces one sequence's definition with a validated payload.

        Args:
            sequence_id : Identifier of the sequence being replaced.
            payload     : Full definition with ``name``, ``dataset``, ``detach``, ``isolate`` and ``blocks``; blocks without a ``block_id`` are assigned one.

        Returns:
            Result carrying the stored entry, or ``ok`` False with an error when
            the sequence is unknown, a block script does not exist, an edge points
            at a missing block, the graph has a cycle, or an entry point of a
            block cannot be resolved to stamp the declared decisions.
        """
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
        """Records the verdict of one decision axis and re-stamps the sequence.

        Args:
            sequence_id : Identifier of the sequence holding the deciding block.
            block_id    : Identifier of the block that decides the axis.
            path        : Configuration path of the decided axis.
            choice      : Winning value, or empty to reopen the question.

        Returns:
            Result carrying the re-stamped entry, or ``ok`` False with an error
            when the sequence, block or axis is unknown.
        """
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
        result             = self.save(sequence_id, entry)

        if result["ok"]:
            self.logger.ok(f"sequence {sequence_id}: '{block['name'] or block['title']}' {'decided ' + path + ' = ' + str(choice) if choice else 'reopened ' + path}")
        return result

    def _clean_block(self, block: dict) -> dict:
        """Returns one block normalised to the stored schema, assigning a fresh id when absent.

        Field notes are not read from the payload: they are rewritten from the
        declared decisions once the whole block list is validated.
        """
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
        """Returns an error message when the block list is not a launchable DAG, else an empty string."""
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
        """Returns the block ids in a topological order, or None when the graph has a cycle.

        Args:
            blocks : Cleaned block list whose predecessor ids all exist.
        """
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
        """Returns every sequence, newest first by update time and identifier.

        A file whose name does not match its ``sequence_id`` is renamed to the
        id on the way through, so a definition dropped into the directory under
        any name (e.g. copied from ``docs/sequences``) becomes addressable.
        Unreadable files and id collisions are reported and skipped instead of
        taking the whole listing down.
        """
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
        """Returns one sequence, or None when the identifier is malformed or absent."""
        with self.lock:
            return self._read(sequence_id)

    def delete(self, sequence_id: str) -> dict:
        """Deletes one sequence, or reports that it was not found."""
        with self.lock:
            entry = self._read(sequence_id)
            if entry is None:
                return {"ok": False, "error": "sequence not found"}
            self._path(sequence_id).unlink()

        self.logger.muted(f"deleted sequence '{entry['name']}' ({sequence_id})")
        return {"ok": True}

    def _read(self, sequence_id: str) -> dict | None:
        """Returns one sequence read under the caller's lock, or None when it is unknown."""
        if not self.ID_PATTERN.match(sequence_id):
            return None

        path = self._path(sequence_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, entry: dict) -> None:
        """Writes one sequence's JSON file under the caller's lock."""
        self._path(entry["sequence_id"]).write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")

    def _path(self, sequence_id: str):
        """Returns the JSON file path holding the sequence with this identifier."""
        return self.directory / f"{sequence_id}.json"
