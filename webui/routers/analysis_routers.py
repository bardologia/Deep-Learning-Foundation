from __future__ import annotations

from ab_autopsy       import AbAutopsy
from fit_lab          import FitLab
from model_probe      import ModelProbe
from routers.dispatch import HttpExchange, RouteTable, SubRouter
from triage_board     import TriageBoard


class FitLabRouter(SubRouter):

    def __init__(self, fitlab: FitLab) -> None:
        self.fitlab = fitlab

        super().__init__(("/api/fitlab",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/fitlab/datasets",   self.datasets)
        table.add("GET",  "/api/fitlab/status",     self.status)
        table.add("GET",  "/api/fitlab/map",        self.map_view)
        table.add("GET",  "/api/fitlab/fit_status", self.fit_status)
        table.add("GET",  "/api/fitlab/fit_result", self.fit_result)
        table.add("POST", "/api/fitlab/load",       self.load)
        table.add("POST", "/api/fitlab/fit",        self.fit)

    def datasets(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.fitlab.datasets(exchange.text("base")))

    def status(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.fitlab.load_status())

    def map_view(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.fitlab.map_png(exchange.text("src", "slc")))

    def fit_status(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.fitlab.fit_status())

    def fit_result(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.fitlab.fit_result_payload(), 404)

    def load(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.fitlab.start_load(exchange.body.get("path", "")))

    def fit(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.fitlab.start_fit(exchange.body))


class ProbeRouter(SubRouter):

    def __init__(self, probe: ModelProbe) -> None:
        self.probe = probe

        super().__init__(("/api/probe",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/probe/runs",     self.runs)
        table.add("GET",  "/api/probe/status",   self.status)
        table.add("GET",  "/api/probe/layers",   self.layers)
        table.add("GET",  "/api/probe/map",      self.map_view)
        table.add("GET",  "/api/probe/features", self.features)
        table.add("POST", "/api/probe/load",        self.load)
        table.add("POST", "/api/probe/predict",     self.predict)
        table.add("POST", "/api/probe/attribution", self.attribution)
        table.add("POST", "/api/probe/whatif",      self.whatif)

    def runs(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.runs(exchange.text("base")))

    def status(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.probe.load_status())

    def layers(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.layers())

    def map_view(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.probe.map_png())

    def features(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.probe.features_png(exchange.integer("az"), exchange.integer("rg"), exchange.text("layer")))

    def load(self, exchange: HttpExchange) -> None:
        body = exchange.body
        exchange.send_result(self.probe.start_load(body.get("path", ""), body.get("split", "test"), body.get("device", "cpu")))

    def predict(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.predict(exchange.body))

    def attribution(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.attribution(exchange.body))

    def whatif(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.whatif(exchange.body))


class TriageRouter(SubRouter):

    def __init__(self, triage: TriageBoard) -> None:
        self.triage = triage

        super().__init__(("/api/triage",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/triage/cases",    self.cases)
        table.add("GET",  "/api/triage/thumb",    self.thumb)
        table.add("GET",  "/api/triage/profile",  self.profile)
        table.add("POST", "/api/triage/annotate", self.annotate)

    def thumb(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.triage.thumb(exchange.text("id"), exchange.integer("az0"), exchange.integer("rg0")))

    def profile(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.triage.profile(exchange.text("id"), exchange.integer("az"), exchange.integer("rg")))

    def cases(self, exchange: HttpExchange) -> None:
        result = self.triage.cases(
            cube_id = exchange.text("id"),
            top_n   = exchange.integer("n", "40"),
        )
        exchange.send_result(result)

    def annotate(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.triage.annotate(exchange.body))


class AutopsyRouter(SubRouter):

    def __init__(self, autopsy: AbAutopsy) -> None:
        self.autopsy = autopsy

        super().__init__(("/api/autopsy",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/autopsy/runs",    self.runs)
        table.add("GET", "/api/autopsy/compare", self.compare)
        table.add("GET", "/api/autopsy/profile", self.profile)

    def runs(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.autopsy.runs(exchange.text("base")))

    def compare(self, exchange: HttpExchange) -> None:
        result = self.autopsy.compare(
            a = exchange.text("a"),
            b = exchange.text("b"),
        )
        exchange.send_result(result)

    def profile(self, exchange: HttpExchange) -> None:
        result = self.autopsy.profile(
            a  = exchange.text("a"),
            b  = exchange.text("b"),
            az = exchange.integer("az"),
            rg = exchange.integer("rg"),
        )
        exchange.send_result(result)
