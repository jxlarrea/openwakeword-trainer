"""Logging configuration."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Also forward INFO+ records to the EventBus so the Web UI live-log shows
    # downloader / orchestrator messages without each module having to call
    # bus.log() explicitly.
    from src.train.progress import BusLoggingHandler

    root.addHandler(BusLoggingHandler(level=logging.INFO))

    # Tame chatty libraries.
    for noisy in (
        "urllib3",
        "httpx",
        "httpcore",
        "filelock",
        "huggingface_hub",
        "datasets",
        "matplotlib",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
