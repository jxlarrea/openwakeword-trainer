"""Process-wide progress / event bus.

Producers (downloaders, sample generators, training loop) push events here.
Consumers (the SSE endpoint) subscribe and forward events to the browser.

The bus is callable from any thread: the worker training loop posts events
without going through the asyncio loop, and SSE consumers receive them through
asyncio.Queue. Cross-thread enqueue is handled via loop.call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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
        self._logs: deque[dict[str, Any]] = deque(maxlen=3000)
        self._phase: dict[str, Any] | None = None
        self._progress: dict[str, dict[str, Any]] = {}
        self._metrics: dict[str, Any] = {}
        self._run_id: str | None = None
        self._persist_path: Path | None = None
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
            self._update_snapshot_locked(ev)
            persist_path = self._persist_path
            subs = list(self._subscribers)
        if persist_path is not None:
            self._persist_event(persist_path, ev)

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

    # Persistent run snapshot ---------------------------------------------

    def start_run(self, run_id: str, run_dir: Path) -> None:
        """Start a durable per-run event log and clear the live snapshot."""
        path = run_dir / "live_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        with self._lock:
            self._history.clear()
            self._logs.clear()
            self._phase = None
            self._progress = {}
            self._metrics = {}
            self._run_id = run_id
            self._persist_path = path

    def finish_run(self) -> None:
        with self._lock:
            self._persist_path = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self._run_id,
                "phase": dict(self._phase) if self._phase else None,
                "progress": [dict(v) for v in self._progress.values()],
                "metrics": dict(self._metrics),
                "logs": list(self._logs),
            }

    def _update_snapshot_locked(self, ev: Event) -> None:
        if ev.kind == "run_started":
            self._run_id = ev.payload.get("run_id") or self._run_id
        elif ev.kind == "phase":
            self._phase = {
                "name": ev.payload.get("name", ""),
                "detail": ev.payload.get("detail", ""),
                "ts": ev.timestamp,
            }
        elif ev.kind == "progress":
            name = str(ev.payload.get("name", ""))
            if name:
                self._progress[name] = {
                    "name": name,
                    "fraction": ev.payload.get("fraction", 0.0),
                    "detail": ev.payload.get("detail", ""),
                    "ts": ev.timestamp,
                }
        elif ev.kind == "metric":
            for key, value in ev.payload.items():
                if key not in ("kind", "ts"):
                    self._metrics[key] = value
        elif ev.kind == "log":
            self._logs.append(
                {
                    "level": ev.payload.get("level", "info"),
                    "message": ev.payload.get("message", ""),
                    "ts": ev.timestamp,
                }
            )

    @staticmethod
    def _persist_event(path: Path, ev: Event) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to persist progress event to %s", path, exc_info=True)


def _safe_put_nowait(q: asyncio.Queue, ev: Event) -> None:
    try:
        q.put_nowait(ev)
    except asyncio.QueueFull:
        logger.warning("SSE subscriber queue full; dropping event")


# Module-level singleton. Imported by both the training pipeline and the SSE route.
bus = EventBus()


# ----------------------------------------------------------------------------- #
# Logging bridge: forward Python log records into the EventBus.
# ----------------------------------------------------------------------------- #

# Loggers whose records should NOT be forwarded to the UI (access logs etc).
_NOISY_LOGGER_PREFIXES = (
    "uvicorn",
    "starlette",
    "watchfiles",
    "asyncio",
)


class BusLoggingHandler(logging.Handler):
    """Logging handler that publishes records to the in-process EventBus.

    Lets every src.* module use `logger.info(...)` and have those lines show up
    in the Web UI's live-log panel without each call site explicitly calling
    bus.log().
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level=level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if any(record.name.startswith(p) for p in _NOISY_LOGGER_PREFIXES):
                return
            msg = record.getMessage()
            # Prefix the logger name so the UI shows where the message came from.
            short_name = record.name.replace("src.", "")
            bus.log(f"[{short_name}] {msg}", level=record.levelname.lower())
        except Exception:  # noqa: BLE001
            self.handleError(record)
