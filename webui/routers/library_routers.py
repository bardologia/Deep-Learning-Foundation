from __future__ import annotations

from model_library_base import ModelNoteLibrary
from routers.dispatch   import HttpExchange, RouteTable, SubRouter


class ContentLibraryRouter(SubRouter):

    def __init__(self, section: str, library, envelope: str = "") -> None:
        self.section  = section
        self.library  = library
        self.envelope = envelope

        super().__init__((section,))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", self.section, self.content)

    def content(self, exchange: HttpExchange) -> None:
        collected = self.library.collect()
        exchange.send_json({self.envelope: collected} if self.envelope else collected)


class ModelLibraryRouter(SubRouter):

    def __init__(self, section: str, library: ModelNoteLibrary) -> None:
        self.section = section
        self.library = library

        super().__init__((section,))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", self.section, self.families)
        table.wildcard("GET", f"{self.section}/", "/note", self.note)

    def families(self, exchange: HttpExchange) -> None:
        exchange.send_json({"families": self.library.collect()})

    def note(self, exchange: HttpExchange, key: str) -> None:
        note = self.library.note(key)
        if note is None:
            exchange.send_json({"error": "not found"}, 404)
            return

        exchange.send_json(note)


class BackboneRouter(ModelLibraryRouter):

    def families(self, exchange: HttpExchange) -> None:
        exchange.send_json({"families": self.library.collect(), "heads": self.library.heads()})
