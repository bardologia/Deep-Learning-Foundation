"""Unit tests for the SSE stream pump and the numbered job stream behind it.

Covers the running event ids a job stream assigns, the replay a reconnecting
viewer gets after the last id it holds, the id lines the pump writes, the
Last-Event-ID header a reconnect sends and the 400 a malformed one gets.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

from process_manager         import JobStream
from routers.dispatch        import HttpExchange
from routers.launch_routers  import JobRouter
from web_logger              import WebLogger


def _exchange(headers: dict) -> HttpExchange:
    """Returns an exchange over a fake handler carrying the given request headers and a byte sink."""
    handler         = MagicMock()
    handler.headers = headers
    handler.wfile   = io.BytesIO()

    return HttpExchange(handler, "GET", "/api/jobs/job/stream", "/api/jobs/job/stream", {})


def _stream(n_chunks: int) -> JobStream:
    """Returns a job stream holding ``n_chunks`` output chunks followed by the end marker."""
    stream = JobStream(WebLogger())
    for index in range(n_chunks):
        stream.publish({"type": "chunk", "data": f"line {index}\n"})
    stream.publish({"type": "end"})
    return stream


def test_published_events_carry_running_ids():
    """Every published event is numbered from 1 in publication order."""
    stream = _stream(3)

    assert [event["id"] for event in stream.buffer] == [1, 2, 3, 4]


def test_a_reconnect_replays_only_the_events_after_its_last_id():
    """A subscriber naming id 2 gets events 3 and 4, a fresh one gets the whole buffer."""
    stream = _stream(3)

    fresh  = stream.subscribe()
    resume = stream.subscribe(2)

    assert fresh.qsize() == 4
    assert [resume.get_nowait()["id"] for _ in range(resume.qsize())] == [3, 4]


def test_the_pump_writes_id_lines_and_resumes_from_the_last_event_id():
    """A job stream request with Last-Event-ID 2 subscribes after event 2 and writes each event with its id line."""
    stream   = _stream(3)
    backend  = MagicMock()
    backend.get_stream.return_value = stream
    exchange = _exchange({"Last-Event-ID": "2"})

    router = JobRouter(MagicMock(), backend, MagicMock())
    router.stream(exchange, "job")

    written = exchange.handler.wfile.getvalue().decode("utf-8")
    assert "id: 3\ndata: " in written
    assert "id: 4\ndata: " in written
    assert "id: 1\n" not in written
    exchange.handler.send_response.assert_called_once_with(200)


def test_a_malformed_last_event_id_is_refused():
    """A Last-Event-ID that is not a non-negative integer answers 400 before any stream is opened."""
    backend  = MagicMock()
    exchange = _exchange({"Last-Event-ID": "ten"})

    router = JobRouter(MagicMock(), backend, MagicMock())
    router.stream(exchange, "job")

    assert exchange.status == 400
    backend.get_stream.assert_not_called()
