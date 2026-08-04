from __future__ import annotations

from urllib.parse import unquote

from project_paths    import ProjectPaths
from results_browser  import ResultsBrowser
from routers.dispatch import HttpExchange, RouteTable, SubRouter


class StaticRouter(SubRouter):

    def __init__(self, paths: ProjectPaths, results: ResultsBrowser) -> None:
        self.paths   = paths
        self.results = results

        super().__init__(("/", "/static", "/resultsmedia"))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/",             self.index)
        table.add("GET", "/resultsmedia", self.media)
        table.wildcard("GET", "/static/", "", self.asset)

    def index(self, exchange: HttpExchange) -> None:
        self.serve(exchange, "index.html")

    def media(self, exchange: HttpExchange) -> None:
        target = self.results.file_path(exchange.text("path"))
        if target is None:
            exchange.not_found()
            return

        exchange.send_file(target, "max-age=60")

    def asset(self, exchange: HttpExchange, relative: str) -> None:
        self.serve(exchange, relative)

    def serve(self, exchange: HttpExchange, relative: str) -> None:
        target = (self.paths.static_dir / unquote(relative)).resolve()

        if not target.is_relative_to(self.paths.static_dir.resolve()):
            exchange.send_json({"error": "forbidden"}, 403)
            return

        if not target.is_file():
            exchange.not_found()
            return

        exchange.send_file(target, "no-cache")
