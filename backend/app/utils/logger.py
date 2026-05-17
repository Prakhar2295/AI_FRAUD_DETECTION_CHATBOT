"""Application logging utilities.

Provides a centralized logger factory that writes both to console and a
rotating log file. Designed for local development debugging without any
external telemetry or tracing dependencies.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict


LOG_DIR = Path("backend/logs")
LOG_FILE = LOG_DIR / "fraud_detection_system.log"


def _ensure_log_dir() -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort: if we cannot create it, continue with console logging
        pass


_GLOBAL_HANDLERS: Dict[str, logging.Handler] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for `name`.

    Creates a stream handler and a rotating file handler (UTF-8). Handlers
    are created once and reused across calls.
    """
    _ensure_log_dir()

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Basic readable formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    _GLOBAL_HANDLERS["stream"] = stream_handler

    # Rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(LOG_FILE), mode="a", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        _GLOBAL_HANDLERS["file"] = file_handler
    except Exception:
        # If file handler cannot be created, fall back to console only.
        logger.exception("Failed to create file handler for logs; continuing with console only")

    # Set default level and avoid double propagation
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Quiet noisy third-party loggers that clutter output. Keep general uvicorn
    # and uvicorn.error at INFO so startup/completion messages are visible.
    for noisy in ("uvicorn.access", "httpx", "asyncio", "chardet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger

