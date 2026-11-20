
from __future__ import annotations

from dataset_browser  import DatasetBrowser
from results_browser  import ResultsBrowser
from routers.dispatch import HttpExchange, RouteTable, SubRouter


class ResultsRouter(SubRouter):

    def __init__(self, results: ResultsBrowser) -> None:
        self.results = results

        super().__init__(("/api/results",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/results/tree",    self.tree)
        table.add("GET", "/api/results/folder",  self.folder)
        table.add("GET", "/api/results/catalog", self.catalog)
        table.add("GET", "/api/results/gallery", self.gallery)

    def tree(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.results.tree(exchange.text("path")), 404)

    def folder(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.results.folder(exchange.text("root"), exchange.text("rel")), 404)

    def catalog(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.results.catalog(exchange.text("datasets"), exchange.text("logs")))

    def gallery(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.results.gallery(exchange.text("root")), 404)


class DatasetRouter(SubRouter):

    def __init__(self, datasets: DatasetBrowser) -> None:
        self.datasets = datasets

        super().__init__(("/api/fs",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/fs/datasets",     self.dataset_roots)
        table.add("GET", "/api/fs/runs",         self.runs)
        table.add("GET", "/api/fs/run_groups",   self.run_groups)
        table.add("GET", "/api/fs/experiments",  self.experiments)
        table.add("GET", "/api/fs/params",       self.params)
        table.add("GET", "/api/fs/secondaries",  self.secondaries)

    def dataset_roots(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.datasets.datasets(exchange.text("base")))

    def runs(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.datasets.runs(exchange.texts("base"), seed_units=exchange.text("units", "0") == "1", split=exchange.text("split", "")))

    def run_groups(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.datasets.run_groups(exchange.texts("base")))

    def experiments(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.datasets.experiments(exchange.text("base"), exchange.text("marker")))

    def params(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.datasets.params(exchange.text("dataset")))

    def secondaries(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.datasets.secondaries(exchange.text("dataset")))
