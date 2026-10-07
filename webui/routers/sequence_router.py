"""API routes for building, saving and executing experiment sequences.

Serves the sequence store (create, save, list, delete) and the sequence runner
(start, resume scan, stop, checkpoint release, the study snapshots of a running
tuning block), attaching each sequence's live execution state to its
definition.
"""

from __future__ import annotations

from routers.dispatch import HttpExchange, RouteTable, SubRouter
from sequence_runner  import SequenceRunner
from sequence_store   import SequenceStore


class SequenceRouter(SubRouter):
    """Serves the sequence builder: definitions on disk and live execution control.

    Attributes:
        sequences : Store holding the sequence definitions.
        runner    : Runner advancing started sequences over the process manager.
    """

    def __init__(self, sequences: SequenceStore, runner: SequenceRunner) -> None:
        """Stores the sequence store and runner, then declares the routes."""
        self.sequences = sequences
        self.runner    = runner

        super().__init__(("/api/sequences",))

    def declare(self, table: RouteTable) -> None:
        """Registers the listing, create, study, detail, save, delete and execution routes; the study route precedes the detail route, whose empty suffix would otherwise claim it."""
        table.add("GET",  "/api/sequences", self.listing)
        table.add("POST", "/api/sequences", self.create)
        table.wildcard("GET",  "/api/sequences/", "/study",   self.study)
        table.wildcard("GET",  "/api/sequences/", "",         self.detail)
        table.wildcard("POST", "/api/sequences/", "/save",    self.save)
        table.wildcard("POST", "/api/sequences/", "/decide",  self.decide)
        table.wildcard("POST", "/api/sequences/", "/delete",  self.remove)
        table.wildcard("POST", "/api/sequences/", "/resume-scan", self.resume_scan)
        table.wildcard("POST", "/api/sequences/", "/start",   self.start)
        table.wildcard("POST", "/api/sequences/", "/stop",    self.stop)
        table.wildcard("POST", "/api/sequences/", "/clear",   self.clear)
        table.wildcard("POST", "/api/sequences/", "/release", self.release)

    def listing(self, exchange: HttpExchange) -> None:
        """Answers with every sequence, newest first, each with its live state or None."""
        listed = self.sequences.list()
        states = self.runner.states()

        exchange.send_json({"sequences": [{**entry, "state": states.get(entry["sequence_id"])} for entry in listed["sequences"]]})

    def create(self, exchange: HttpExchange) -> None:
        """Creates a new empty sequence named in the body."""
        exchange.send_result(self.sequences.create(exchange.body.get("name", "")))

    def detail(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Answers with one sequence's definition and live state, or 404 when unknown."""
        entry = self.sequences.get(sequence_id)
        if entry is None:
            exchange.send_json({"error": "sequence not found"}, 404)
            return

        exchange.send_json({"ok": True, "entry": entry, "state": self.runner.state(sequence_id)})

    def study(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Answers with the study snapshots of the tuning block named by the ``block`` query parameter."""
        exchange.send_result(self.runner.study_progress(sequence_id, exchange.text("block")))

    def save(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Replaces one sequence's definition with the validated body."""
        exchange.send_result(self.sequences.save(sequence_id, exchange.body))

    def decide(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Records one block's verdict on a decision axis and freezes it into the blocks after it."""
        exchange.send_result(self.sequences.decide(sequence_id, str(exchange.body.get("block_id", "")), str(exchange.body.get("path", "")), str(exchange.body.get("choice", ""))))

    def remove(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Deletes one sequence, refusing while it is still running or held."""
        state = self.runner.state(sequence_id)
        if state is not None and state["status"] in ("running", "held"):
            exchange.send_json({"ok": False, "error": "stop the sequence before deleting it"}, 400)
            return

        exchange.send_result(self.sequences.delete(sequence_id))

    def resume_scan(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Reads which blocks a previous workspace records as finished, so a resumed start can bypass them."""
        exchange.send_result(self.runner.resume_scan(sequence_id, str(exchange.body.get("resume_dir", "")).strip()))

    def start(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Starts one sequence, resuming the workspace named in the body when one is given and bypassing the listed blocks for that execution only."""
        resume_dir = str(exchange.body.get("resume_dir", "")).strip()
        bypass     = [str(block_id) for block_id in list(exchange.body.get("bypass") or [])]

        exchange.send_result(self.runner.start_sequence(sequence_id, resume_dir, bypass))

    def stop(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Stops one running sequence."""
        exchange.send_result(self.runner.stop_sequence(sequence_id))

    def clear(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Discards one sequence's finished execution state, refusing while it is still active."""
        exchange.send_result(self.runner.clear_state(sequence_id))

    def release(self, exchange: HttpExchange, sequence_id: str) -> None:
        """Releases the held checkpoint block named in the body."""
        exchange.send_result(self.runner.release(sequence_id, str(exchange.body.get("block_id", ""))))
