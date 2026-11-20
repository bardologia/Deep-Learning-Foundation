from __future__ import annotations

import pytest

from gpu_activity            import GpuActivityLog
from host_history            import HostHistoryLog
from request_router          import RequestRouter
from routers.library_routers import BackboneRouter, ContentLibraryRouter
from system_monitor          import ActiveUsers, SystemHistory, SystemMonitor
from web_ui_server           import WebUIServer

from tests.webui.conftest import FakeHandler


FRONTEND_PATHS = [
    "/api/configs",
    "/api/backbones",
]


@pytest.fixture(scope="module")
def server():
    original = (SystemMonitor._du_loop, SystemHistory.sample_loop, ActiveUsers.sample_loop, GpuActivityLog.sample_loop, HostHistoryLog.sample_loop)

    SystemMonitor._du_loop      = lambda self: None
    SystemHistory.sample_loop   = lambda self: None
    ActiveUsers.sample_loop     = lambda self: None
    GpuActivityLog.sample_loop  = lambda self: None
    HostHistoryLog.sample_loop  = lambda self: None
    try:
        yield WebUIServer(host="127.0.0.1", port=0)
    finally:
        SystemMonitor._du_loop, SystemHistory.sample_loop, ActiveUsers.sample_loop, GpuActivityLog.sample_loop, HostHistoryLog.sample_loop = original


def _get(server: WebUIServer, path: str) -> tuple[int, object]:
    handler = FakeHandler("GET", path)
    server.router.route(handler)
    return handler.status, handler.payload()


def test_the_composed_router_owns_every_section_once(server):
    claimed = [section for sub in server.router.routers for section in sub.sections]
    raw     = [section for sub in server.router.routers for section in sub.raw_sections]

    assert len(claimed) == len(set(claimed))
    assert sorted(claimed) == sorted(server.router.sections)
    assert all(server.router.sections[section] is sub for sub in server.router.routers for section in sub.sections)
    assert raw == ["/tb/"]


@pytest.mark.parametrize("path", FRONTEND_PATHS)
def test_every_library_endpoint_answers_with_content(server, path):
    status, payload = _get(server, path)

    assert status == 200
    assert payload


def test_the_library_envelopes_match_what_the_frontend_reads(server):
    assert list(_get(server, "/api/configs")[1]) == ["groups"]


def test_the_model_endpoint_answers_with_families_and_heads(server):
    assert sorted(_get(server, "/api/backbones")[1]) == ["families", "heads"]


def test_a_model_note_is_served_and_an_unknown_key_is_a_404(server):
    status, payload = _get(server, "/api/backbones/unet/note")

    assert status == 200
    assert payload["key"] == "unet"
    assert payload["markdown"].strip()

    assert _get(server, "/api/backbones/no_such_model/note")[0] == 404


def test_every_library_router_is_bound_to_a_real_library(server):
    routers = [sub for sub in server.router.routers if isinstance(sub, (ContentLibraryRouter, BackboneRouter))]

    assert len(routers) == 2
    assert all(sub.library is not None for sub in routers)
    assert all(callable(getattr(sub.library, "collect")) for sub in routers)


def test_an_unknown_endpoint_is_still_a_404(server):
    assert _get(server, "/api/does-not-exist")[0] == 404


def test_the_router_is_a_request_router(server):
    assert isinstance(server.router, RequestRouter)
    assert server.paths.repo_root.is_dir()
