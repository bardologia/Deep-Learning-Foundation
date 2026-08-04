from __future__ import annotations

import io
import json

import pytest

from request_router          import RequestRouter
from routers.library_routers import BackboneRouter, ContentLibraryRouter, ModelLibraryRouter
from system_monitor          import ActiveUsers, SystemHistory, SystemMonitor
from web_ui_server           import WebUIServer


FRONTEND_PATHS = [
    "/api/equations",
    "/api/physics-loss",
    "/api/flows",
    "/api/pipelines",
    "/api/repomap",
    "/api/configs",
    "/api/backbones",
    "/api/profile-autoencoders",
    "/api/image-autoencoders",
    "/api/jepa-variants",
]


class FakeHandler:

    def __init__(self, path: str) -> None:
        self.command = "GET"
        self.path    = path
        self.headers = {"Content-Length": "0"}
        self.rfile   = io.BytesIO(b"")
        self.wfile   = io.BytesIO()
        self.status  = 0

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        return

    def end_headers(self) -> None:
        return


@pytest.fixture(scope="module")
def server():
    original = (SystemMonitor._du_loop, SystemHistory.sample_loop, ActiveUsers.sample_loop)

    SystemMonitor._du_loop      = lambda self: None
    SystemHistory.sample_loop   = lambda self: None
    ActiveUsers.sample_loop     = lambda self: None
    try:
        yield WebUIServer(host="127.0.0.1", port=0)
    finally:
        SystemMonitor._du_loop, SystemHistory.sample_loop, ActiveUsers.sample_loop = original


def _get(server: WebUIServer, path: str) -> tuple[int, object]:
    handler = FakeHandler(path)
    server.router.route(handler)
    return handler.status, json.loads(handler.wfile.getvalue().decode("utf-8"))


def test_the_composed_router_owns_every_section_once(server):
    sections = [section for sub in server.router.routers for section in sub.raw_sections]

    assert sorted(sections) == sorted(set(sections))
    assert len(server.router.sections) == len(set(server.router.sections))


@pytest.mark.parametrize("path", FRONTEND_PATHS)
def test_every_library_endpoint_answers_with_content(server, path):
    status, payload = _get(server, path)

    assert status == 200
    assert payload


def test_the_library_envelopes_match_what_the_frontend_reads(server):
    assert list(_get(server, "/api/flows")[1])     == ["flows"]
    assert list(_get(server, "/api/pipelines")[1]) == ["pipelines"]
    assert list(_get(server, "/api/equations")[1]) == ["groups"]
    assert list(_get(server, "/api/repomap")[1])   == ["folders"]
    assert list(_get(server, "/api/configs")[1])   == ["groups"]

    assert sorted(_get(server, "/api/physics-loss")[1]) == ["comparison", "config", "dataset", "intro", "operator", "terms"]


def test_the_model_endpoints_answer_with_families(server):
    assert list(_get(server, "/api/profile-autoencoders")[1]) == ["families"]
    assert list(_get(server, "/api/image-autoencoders")[1])   == ["families"]
    assert list(_get(server, "/api/jepa-variants")[1])        == ["families"]
    assert sorted(_get(server, "/api/backbones")[1])          == ["families", "heads"]


def test_a_model_note_is_served_and_an_unknown_key_is_a_404(server):
    status, payload = _get(server, "/api/profile-autoencoders/mlp_ae/note")

    assert status == 200
    assert payload["key"] == "mlp_ae"
    assert payload["markdown"].strip()

    assert _get(server, "/api/profile-autoencoders/no_such_model/note")[0] == 404


def test_every_library_router_is_bound_to_a_real_library(server):
    routers = [sub for sub in server.router.routers if isinstance(sub, (ContentLibraryRouter, ModelLibraryRouter, BackboneRouter))]

    assert len(routers) == 10
    assert all(sub.library is not None for sub in routers)
    assert all(callable(getattr(sub.library, "collect")) for sub in routers)


def test_an_unknown_endpoint_is_still_a_404(server):
    assert _get(server, "/api/does-not-exist")[0] == 404


def test_the_router_is_a_request_router(server):
    assert isinstance(server.router, RequestRouter)
    assert server.paths.repo_root.is_dir()
