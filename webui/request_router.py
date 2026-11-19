
from __future__ import annotations

import traceback
from urllib.parse import urlparse

from routers.dispatch import HttpExchange, RouteConflict, SubRouter
from web_logger       import WebLogger


class RequestRouter:

    def __init__(self, logger: WebLogger, routers: list[SubRouter]) -> None:
        self.logger   = logger
        self.routers  = list(routers)
        self.sections = self._index()

    def _index(self) -> dict:
        index = {}

        for router in self.routers:
            for section in router.sections:
                owner = index.get(section)
                if owner is not None:
                    raise RouteConflict(f"section '{section}' is claimed by both {type(owner).__name__} and {type(router).__name__}")
                index[section] = router

        return index

    @staticmethod
    def _section_of(path: str) -> str:
        parts = path.split("/")
        depth = 3 if len(parts) > 2 and parts[1] == "api" else 2

        return "/".join(parts[:depth])

    def _proxied(self, exchange: HttpExchange) -> bool:
        for router in self.routers:
            if router.claims_raw(exchange.raw_path):
                router.handle_raw(exchange)
                return True

        return False

    def _dispatch(self, exchange: HttpExchange) -> None:
        if exchange.method == "POST":
            try:
                exchange.read_body()
            except (ValueError, UnicodeDecodeError) as exc:
                exchange.send_json({"error": f"unreadable request body: {exc}"}, 400)
                return

        router = self.sections.get(self._section_of(exchange.path))
        if router is None or not router.handle(exchange):
            exchange.not_found()

    def _same_origin(self, exchange: HttpExchange) -> bool:
        origin = exchange.header("Origin")
        if not origin:
            return True

        return urlparse(origin).netloc == exchange.header("Host")

    def _report(self, exchange: HttpExchange) -> None:
        if exchange.status < 400:
            return

        detail = f": {exchange.error}" if exchange.error else ""
        self.logger.warning(f"{exchange.method} {exchange.raw_path} -> {exchange.status}{detail}")

    def route(self, handler) -> None:
        exchange = HttpExchange.of(handler)

        try:
            if not self._same_origin(exchange):
                exchange.send_json({"error": "cross-origin request refused"}, 403)
            elif not self._proxied(exchange):
                self._dispatch(exchange)

            self._report(exchange)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.logger.error(f"router error on {exchange.method} {exchange.raw_path}:\n{traceback.format_exc()}")
            try:
                exchange.send_json({"error": str(exc)}, 500)
            except Exception as reply_exc:
                self.logger.error(f"could not deliver the 500 reply on {exchange.method} {exchange.raw_path}: {reply_exc}")
