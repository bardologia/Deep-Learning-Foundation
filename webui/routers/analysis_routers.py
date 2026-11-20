
from __future__ import annotations

from ab_autopsy       import AbAutopsy
from model_probe      import ModelProbe
from model_survey     import ModelSurvey
from routers.dispatch import HttpExchange, RouteTable, SubRouter


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
        table.add("GET",  "/api/probe/kernels",  self.kernels)
        table.add("POST", "/api/probe/load",        self.load)
        table.add("POST", "/api/probe/predict",     self.predict)
        table.add("POST", "/api/probe/fields",      self.fields)
        table.add("POST", "/api/probe/attribution", self.attribution)
        table.add("POST", "/api/probe/ablation",    self.ablation)
        table.add("POST", "/api/probe/occlusion",   self.occlusion)
        table.add("POST", "/api/probe/flips",       self.flips)
        table.add("POST", "/api/probe/vitals",      self.vitals)

    def runs(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.runs(exchange.text("base")))

    def status(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.probe.load_status())

    def layers(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.layers())

    def map_view(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.probe.map_png())

    def features(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.probe.features_png(exchange.integer("row"), exchange.integer("col"), exchange.text("layer")))

    def kernels(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.probe.kernels_png(exchange.text("layer")))

    def load(self, exchange: HttpExchange) -> None:
        body = exchange.body
        exchange.send_result(self.probe.start_load(body.get("path", ""), body.get("split", "test"), body.get("device", "cpu")))

    def predict(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.predict(exchange.body))

    def fields(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.fields(exchange.body))

    def attribution(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.attribution(exchange.body))

    def ablation(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.ablation(exchange.body))

    def occlusion(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.occlusion(exchange.body))

    def flips(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.flips(exchange.body))

    def vitals(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.probe.vitals(exchange.body))


class SurveyRouter(SubRouter):

    def __init__(self, survey: ModelSurvey) -> None:
        self.survey = survey

        super().__init__(("/api/survey",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/survey/runs",   self.runs)
        table.add("GET",  "/api/survey/status", self.status)
        table.add("GET",  "/api/survey/result", self.result)
        table.add("POST", "/api/survey/start",  self.start)
        table.add("POST", "/api/survey/cancel", self.cancel)

    def runs(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.survey.runs(exchange.text("base")))

    def status(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.survey.survey_status())

    def result(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.survey.survey_result(), 404)

    def start(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.survey.start(exchange.body))

    def cancel(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.survey.cancel())


class AutopsyRouter(SubRouter):

    def __init__(self, autopsy: AbAutopsy) -> None:
        self.autopsy = autopsy

        super().__init__(("/api/autopsy",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/autopsy/runs",    self.runs)
        table.add("GET", "/api/autopsy/compare", self.compare)
        table.add("GET", "/api/autopsy/pixel",   self.pixel)

    def runs(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.autopsy.runs(exchange.text("base")))

    def compare(self, exchange: HttpExchange) -> None:
        result = self.autopsy.compare(
            a = exchange.text("a"),
            b = exchange.text("b"),
        )
        exchange.send_result(result)

    def pixel(self, exchange: HttpExchange) -> None:
        result = self.autopsy.pixel(
            a   = exchange.text("a"),
            b   = exchange.text("b"),
            row = exchange.integer("row"),
            col = exchange.integer("col"),
        )
        exchange.send_result(result)


class AnalysisPayloadRouter(SubRouter):

    PREFIX: str

    def __init__(self, browser) -> None:
        self.browser = browser

        super().__init__((self.PREFIX,))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", f"{self.PREFIX}/analyses", self.analyses)
        table.add("GET", f"{self.PREFIX}/report",   self.report)

    def analyses(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.browser.analyses(exchange.text("base")))

    def report(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.browser.report(exchange.text("id")))
