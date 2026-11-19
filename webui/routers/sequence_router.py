
from __future__ import annotations

from routers.dispatch import HttpExchange, RouteTable, SubRouter
from sequence_runner  import SequenceRunner
from sequence_store   import SequenceStore


class SequenceRouter(SubRouter):

    def __init__(self, sequences: SequenceStore, runner: SequenceRunner) -> None:
        self.sequences = sequences
        self.runner    = runner

        super().__init__(("/api/sequences",))

    def declare(self, table: RouteTable) -> None:
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
        listed = self.sequences.list()
        states = self.runner.states()

        exchange.send_json({"sequences": [{**entry, "state": states.get(entry["sequence_id"])} for entry in listed["sequences"]]})

    def create(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.sequences.create(exchange.body.get("name", "")))

    def detail(self, exchange: HttpExchange, sequence_id: str) -> None:
        entry = self.sequences.get(sequence_id)
        if entry is None:
            exchange.send_json({"error": "sequence not found"}, 404)
            return

        exchange.send_json({"ok": True, "entry": entry, "state": self.runner.state(sequence_id)})

    def study(self, exchange: HttpExchange, sequence_id: str) -> None:
        exchange.send_result(self.runner.study_progress(sequence_id, exchange.text("block")))

    def save(self, exchange: HttpExchange, sequence_id: str) -> None:
        exchange.send_result(self.sequences.save(sequence_id, exchange.body))

    def decide(self, exchange: HttpExchange, sequence_id: str) -> None:
        exchange.send_result(self.sequences.decide(sequence_id, str(exchange.body.get("block_id", "")), str(exchange.body.get("path", "")), str(exchange.body.get("choice", ""))))

    def remove(self, exchange: HttpExchange, sequence_id: str) -> None:
        state = self.runner.state(sequence_id)
        if state is not None and state["status"] in ("running", "held"):
            exchange.send_json({"ok": False, "error": "stop the sequence before deleting it"}, 400)
            return

        exchange.send_result(self.sequences.delete(sequence_id))

    def resume_scan(self, exchange: HttpExchange, sequence_id: str) -> None:
        exchange.send_result(self.runner.resume_scan(sequence_id, str(exchange.body.get("resume_dir", "")).strip()))

    def start(self, exchange: HttpExchange, sequence_id: str) -> None:
        resume_dir = str(exchange.body.get("resume_dir", "")).strip()
        bypass     = [str(block_id) for block_id in list(exchange.body.get("bypass") or [])]

        exchange.send_result(self.runner.start_sequence(sequence_id, resume_dir, bypass))

    def stop(self, exchange: HttpExchange, sequence_id: str) -> None:
        exchange.send_result(self.runner.stop_sequence(sequence_id))

    def clear(self, exchange: HttpExchange, sequence_id: str) -> None:
        exchange.send_result(self.runner.clear_state(sequence_id))

    def release(self, exchange: HttpExchange, sequence_id: str) -> None:
        exchange.send_result(self.runner.release(sequence_id, str(exchange.body.get("block_id", ""))))
