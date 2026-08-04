from __future__ import annotations

import io
import json

import pytest

from request_router             import RequestRouter
from routers.analysis_routers   import AutopsyRouter, FitLabRouter, ProbeRouter, TriageRouter
from routers.cube_routers       import CubeRouter, SliceRouter
from routers.dispatch           import HttpExchange, RouteConflict, RouteTable, SubRouter
from routers.launch_routers     import CatalogRouter, JobRouter, SavedRunRouter
from routers.library_routers    import BackboneRouter, ContentLibraryRouter, ModelLibraryRouter
from routers.results_routers    import CurvesRouter, DatasetRouter, LeaderboardRouter, ResultsRouter
from routers.static_router      import StaticRouter
from routers.system_router      import SystemRouter
from routers.tensorboard_router import TensorboardRouter

ROUTE_COUNT   = 107
SECTION_COUNT = 34

RESOLUTIONS = [
    ("GET",  "/",                                  StaticRouter),
    ("GET",  "/static/js/app.js",                  StaticRouter),
    ("GET",  "/resultsmedia",                      StaticRouter),
    ("GET",  "/api/results/tree",                  ResultsRouter),
    ("GET",  "/api/fs/runs",                       DatasetRouter),
    ("GET",  "/api/leaderboard/diff",              LeaderboardRouter),
    ("GET",  "/api/curves",                        CurvesRouter),
    ("GET",  "/api/cubes/plane",                   CubeRouter),
    ("POST", "/api/cubes/save_slices",             CubeRouter),
    ("GET",  "/api/slices/slice",                  SliceRouter),
    ("POST", "/api/slices/collect",                SliceRouter),
    ("GET",  "/api/fitlab/map",                    FitLabRouter),
    ("POST", "/api/probe/predict",                 ProbeRouter),
    ("GET",  "/api/probe/runs",                    ProbeRouter),
    ("GET",  "/api/triage/cases",                  TriageRouter),
    ("GET",  "/api/triage/thumb",                  TriageRouter),
    ("GET",  "/api/triage/profile",                TriageRouter),
    ("GET",  "/api/autopsy/compare",               AutopsyRouter),
    ("GET",  "/api/autopsy/runs",                  AutopsyRouter),
    ("GET",  "/api/backbones",                     BackboneRouter),
    ("GET",  "/api/backbones/unet/note",           BackboneRouter),
    ("GET",  "/api/jepa-variants/jepa_vit/note",   ModelLibraryRouter),
    ("GET",  "/api/configs",                       ContentLibraryRouter),
    ("GET",  "/api/physics-loss",                  ContentLibraryRouter),
    ("GET",  "/api/project",                       CatalogRouter),
    ("GET",  "/api/scripts/train_backbone",        CatalogRouter),
    ("GET",  "/api/scripts/train_backbone/config", CatalogRouter),
    ("POST", "/api/run",                           JobRouter),
    ("GET",  "/api/jobs/job-1/stream",             JobRouter),
    ("POST", "/api/saved-runs/abc/delete",         SavedRunRouter),
    ("GET",  "/api/saved-runs",                    SavedRunRouter),
    ("GET",  "/api/system",                        SystemRouter),
    ("POST", "/api/gpu-schedule",                  SystemRouter),
    ("GET",  "/api/gpu-guard/history",             SystemRouter),
    ("POST", "/api/tensorboard/tb-1/stop",         TensorboardRouter),
]


class FakeHandler:

    def __init__(self, command: str, path: str, body: bytes = b"") -> None:
        self.command = command
        self.path    = path
        self.headers = {"Content-Length": str(len(body))}
        self.rfile   = io.BytesIO(body)
        self.wfile   = io.BytesIO()
        self.status  = 0
        self.sent    = {}

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.sent[name] = value

    def end_headers(self) -> None:
        return


class EchoRouter(SubRouter):

    def __init__(self, sections: tuple) -> None:
        self.seen = []
        super().__init__(sections)

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/echo", self.plain)
        table.wildcard("GET", "/api/echo/", "/tail", self.tail)

    def plain(self, exchange: HttpExchange) -> None:
        self.seen.append(None)
        exchange.send_json({"ok": True})

    def tail(self, exchange: HttpExchange, key: str) -> None:
        self.seen.append(key)
        exchange.send_json({"ok": True, "key": key})


def build_routers() -> list:
    return [
        StaticRouter(None, None),
        ResultsRouter(None),
        DatasetRouter(None),
        LeaderboardRouter(None),
        CurvesRouter(None),
        CubeRouter(None),
        SliceRouter(None),
        FitLabRouter(None),
        ProbeRouter(None),
        TriageRouter(None),
        AutopsyRouter(None),
        ContentLibraryRouter("/api/equations",    None, "groups"),
        ContentLibraryRouter("/api/physics-loss", None),
        ContentLibraryRouter("/api/flows",        None, "flows"),
        ContentLibraryRouter("/api/pipelines",    None, "pipelines"),
        ContentLibraryRouter("/api/repomap",      None, "folders"),
        ContentLibraryRouter("/api/configs",      None, "groups"),
        BackboneRouter("/api/backbones", None),
        ModelLibraryRouter("/api/profile-autoencoders", None),
        ModelLibraryRouter("/api/image-autoencoders",   None),
        ModelLibraryRouter("/api/jepa-variants",        None),
        CatalogRouter(None, None, None, None, None, None),
        JobRouter(None, None, None),
        SavedRunRouter(None, None, None),
        SystemRouter(None, None, None, None, None, None, None, None, None),
        TensorboardRouter(None, None),
    ]


def response_of(handler: FakeHandler) -> dict:
    return json.loads(handler.wfile.getvalue().decode("utf-8"))


def test_jsonsafe_replaces_non_finite_floats():
    payload = {
        "ok"     : True,
        "value"  : float("nan"),
        "high"   : float("inf"),
        "low"    : float("-inf"),
        "fine"   : 1.5,
        "count"  : 3,
        "name"   : "run",
        "nested" : {"mu": float("nan"), "list": [1.0, float("inf"), {"deep": float("nan")}]},
    }

    safe = HttpExchange.jsonsafe(payload)

    assert safe["ok"] is True
    assert safe["value"] is None
    assert safe["high"] is None
    assert safe["low"] is None
    assert safe["fine"] == 1.5
    assert safe["count"] == 3
    assert safe["name"] == "run"
    assert safe["nested"]["mu"] is None
    assert safe["nested"]["list"] == [1.0, None, {"deep": None}]


def test_jsonsafe_keeps_bools_and_none():
    assert HttpExchange.jsonsafe({"flag": False, "none": None}) == {"flag": False, "none": None}


def test_query_accessors_apply_defaults():
    exchange = HttpExchange.of(FakeHandler("GET", "/api/cubes/plane?id=run&az=7&frac=0.5&vmax="))

    assert exchange.text("id")             == "run"
    assert exchange.text("space", "phys")  == "phys"
    assert exchange.texts("id")            == ["run"]
    assert exchange.integer("az")          == 7
    assert exchange.number("frac")         == 0.5
    assert exchange.number("keep", "-inf") == float("-inf")
    assert exchange.optional_number("vmax") is None


def test_route_table_prefers_exact_then_wildcard():
    router = EchoRouter(("/api/echo",))

    action, key = router.table.action_for("GET", "/api/echo")
    assert key is None and action is not None

    action, key = router.table.action_for("GET", "/api/echo/abc/tail")
    assert key == "abc"

    assert router.table.action_for("POST", "/api/echo") == (None, None)


def test_route_table_rejects_duplicate_declarations():
    table = RouteTable()
    table.add("GET", "/api/echo", print)

    with pytest.raises(RouteConflict):
        table.add("GET", "/api/echo", print)

    table.wildcard("GET", "/api/echo/", "/tail", print)

    with pytest.raises(RouteConflict):
        table.wildcard("GET", "/api/echo/", "/tail", print)


def test_router_rejects_two_owners_of_a_section():
    with pytest.raises(RouteConflict):
        RequestRouter(None, [EchoRouter(("/api/echo",)), EchoRouter(("/api/echo",))])


def test_route_inventory_is_stable():
    routers = build_routers()
    router  = RequestRouter(None, routers)

    declared = sum(len(sub.table.exact) + len(sub.table.wildcards) for sub in routers)
    raw      = sum(len(sub.raw_sections) for sub in routers)

    assert declared == ROUTE_COUNT
    assert len(router.sections) == SECTION_COUNT
    assert raw == 1


@pytest.mark.parametrize("method,path,owner", RESOLUTIONS)
def test_every_frontend_path_resolves_to_one_sub_router(method, path, owner):
    routers = build_routers()
    router  = RequestRouter(None, routers)

    owners = [sub for sub in routers if sub.table.action_for(method, path)[0] is not None]

    assert len(owners) == 1, path
    assert type(owners[0]) is owner
    assert router.sections[RequestRouter._section_of(path)] is owners[0]


def test_unknown_path_and_method_fall_back():
    router = RequestRouter(None, [EchoRouter(("/api/echo",))])

    missing = FakeHandler("GET", "/api/nope")
    router.route(missing)
    assert missing.status == 404

    wrong = FakeHandler("PUT", "/api/echo")
    router.route(wrong)
    assert wrong.status == 405
    assert response_of(wrong) == {"error": "method not allowed"}


def test_trailing_slash_and_body_are_normalised():
    echo   = EchoRouter(("/api/echo",))
    router = RequestRouter(None, [echo])

    handler = FakeHandler("GET", "/api/echo/?x=1")
    router.route(handler)

    assert handler.status == 200
    assert echo.seen == [None]
