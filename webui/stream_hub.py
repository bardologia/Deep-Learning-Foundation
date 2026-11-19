
from __future__ import annotations

import queue
import threading
import uuid

from process_manager import JobStream
from web_logger      import WebLogger


class ChannelSink:

    def __init__(self, channel: StreamChannel, key: str) -> None:
        self.channel = channel
        self.key     = key
        self.ended   = False

    def put_nowait(self, event: dict) -> None:
        if event["type"] == "end":
            self.ended = True

        self.channel.offer({"key": self.key, **event})


class ChannelSubscription:

    def __init__(self, stream: JobStream, sink: ChannelSink, stop: threading.Event | None) -> None:
        self.stream = stream
        self.sink   = sink
        self.stop   = stop

    def release(self) -> None:
        self.stream.unsubscribe(self.sink)

        if self.stop is not None:
            self.stop.set()


class StreamChannel:

    QUEUE_LIMIT = 8000
    CLOSE       = {"type": "close"}

    def __init__(self, channel_id: str, logger: WebLogger) -> None:
        self.channel_id    = channel_id
        self.logger        = logger
        self.events        = queue.Queue(maxsize=self.QUEUE_LIMIT)
        self.subscriptions = {}
        self.dropped       = 0
        self.closed        = False
        self.offer_lock    = threading.Lock()
        self.lock          = threading.Lock()

    def offer(self, event: dict) -> None:
        with self.offer_lock:
            try:
                self.events.put_nowait(event)
                return
            except queue.Full:
                pass

            dropped       = self._drain()
            self.dropped += dropped
            self.events.put_nowait({"key": None, "type": "truncated", "dropped": dropped})
            self.events.put_nowait(event)

        self.logger.warning(f"a console channel fell {dropped} events behind and was truncated; the on-disk logs keep the full output")

    def next(self, timeout: float) -> dict | None:
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def attach(self, key: str, stream: JobStream, stop: threading.Event | None, after: int) -> None:
        sink = ChannelSink(self, key)

        with self.lock:
            previous                = self.subscriptions.pop(key, None)
            self.subscriptions[key] = ChannelSubscription(stream, sink, stop)

        if previous is not None:
            previous.release()

        stream.attach(sink, after)

    def detach(self, key: str) -> bool:
        with self.lock:
            subscription = self.subscriptions.pop(key, None)

        if subscription is None:
            return False

        subscription.release()
        return True

    def prune(self) -> list[str]:
        with self.lock:
            ended = [key for key, subscription in self.subscriptions.items() if subscription.sink.ended]
            gone  = [self.subscriptions.pop(key) for key in ended]

        for subscription in gone:
            subscription.release()

        return ended

    def close(self) -> None:
        with self.lock:
            subscriptions      = list(self.subscriptions.values())
            self.subscriptions = {}
            self.closed        = True

        for subscription in subscriptions:
            subscription.release()

        with self.offer_lock:
            try:
                self.events.put_nowait(self.CLOSE)
            except queue.Full:
                self._drain()
                self.events.put_nowait(self.CLOSE)

    def _drain(self) -> int:
        dropped = 0

        while True:
            try:
                self.events.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped


class StreamHub:

    def __init__(self, logger: WebLogger) -> None:
        self.logger   = logger
        self.boot     = uuid.uuid4().hex
        self.channels = {}
        self.lock     = threading.Lock()

    def open(self, channel_id: str) -> StreamChannel:
        channel = StreamChannel(channel_id, self.logger)

        with self.lock:
            previous                  = self.channels.get(channel_id)
            self.channels[channel_id] = channel

        if previous is not None:
            previous.close()

        return channel

    def channel(self, channel_id: str) -> StreamChannel | None:
        with self.lock:
            return self.channels.get(channel_id)

    @staticmethod
    def key_of(job_id: str, unit: int | None) -> str:
        return job_id if unit is None else f"{job_id}/{unit}"

    def subscribe(self, channel_id: str, backend, job_id: str, unit: int | None, after: int) -> dict:
        channel = self.channel(channel_id)

        if channel is None:
            return {"ok": False, "error": "unknown channel", "stale": True}

        channel.prune()
        key = self.key_of(job_id, unit)

        if unit is None:
            stream = backend.get_stream(job_id)
            if stream is None:
                return {"ok": False, "error": "unknown job"}
            channel.attach(key, stream, None, after)
            return {"ok": True, "key": key}

        opened = backend.unit_stream(job_id, unit)
        if not opened.get("ok"):
            return opened

        channel.attach(key, opened["stream"], opened["stop"], 0)
        return {"ok": True, "key": key}

    def unsubscribe(self, channel_id: str, job_id: str, unit: int | None) -> dict:
        channel = self.channel(channel_id)

        if channel is None:
            return {"ok": False, "error": "unknown channel", "stale": True}

        key      = self.key_of(job_id, unit)
        attached = channel.detach(key)
        channel.prune()

        return {"ok": True, "key": key, "attached": attached}

    def release(self, channel: StreamChannel) -> None:
        with self.lock:
            current = self.channels.get(channel.channel_id) is channel
            if current:
                del self.channels[channel.channel_id]

        if current:
            channel.close()
