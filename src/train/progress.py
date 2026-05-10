"""Process-wide progress / event bus.

Producers (downloaders, sample generators, training loop) push events here.
Consumers (the SSE endpoint) subscribe and forward events to the browser.

The bus is callable from any thread: the worker training loop posts events
without going through the asyncio loop, and SSE consumers receive them through
asyncio.Queue. Cross-thread enqueue is handled via loop.call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    kind: str  # "log" | "progress" | "metric" | "phase" | "complete" | "error"
    payload: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ts": self.timestamp, **self.payload}


class _Subscriber:
    __slots__ = ("queue", "loop")

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self.queue = queue
        self.loop = loop


class EventBus:
    """In-process pub/sub. Replays a small history to new subscribers."""

    def __init__(self, history_size: int = 200) -> None:
        self._subscribers: set[_Subscriber] = set()
        self._history: deque[Event] = deque(maxlen=history_size)
        # threading.Lock protects mutations from any thread.
        self._lock = threading.Lock()

    async def subscribe(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        with self._lock:
            replay = list(self._history)
            self._subscribers.add(_Subscriber(q, loop))
        for ev in replay:
            await q.put(ev)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = {s for s in self._subscribers if s.queue is not q}

    def publish(self, kind: str, **payload: Any) -> None:
        ev = Event(kind=kind, payload=payload, timestamp=time.time())
        with self._lock:
            self._history.append(ev)
            subs = list(self._subscribers)

        # Fan out without awaiting; drop on full to avoid blocking the producer.
        for sub in subs:
            try:
                sub.loop.call_soon_threadsafe(_safe_put_nowait, sub.queue, ev)
            except RuntimeError:
                # Event loop closed; subscriber will be cleaned up on its handler exit.
                pass

    # Convenience helpers --------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        self.publish("log", message=message, level=level)

    def phase(self, name: str, detail: str = "") -> None:
        self.publish("phase", name=name, detail=detail)

    def progress(self, name: str, fraction: float, detail: str = "") -> None:
        self.publish("progress", name=name, fraction=max(0.0, min(1.0, fraction)), detail=detail)

    def metric(self, **values: Any) -> None:
        self.publish("metric", **values)

    def complete(self, **payload: Any) -> None:
        self.publish("complete", **payload)

    def error(self, message: str, **extra: Any) -> None:
        # Named "run_error" so it doesn't collide with EventSource's built-in
        # connection-error event in the browser.
        self.publish("run_error", message=message, **extra)


def _safe_put_nowait(q: asyncio.Queue, ev: Event) -> None:
    try:
        q.put_nowait(ev)
    except asyncio.QueueFull:
        logger.warning("SSE subscriber queue full; dropping event")


# Module-level singleton. Imported by both the training pipeline and the SSE route.
bus = EventBus()
