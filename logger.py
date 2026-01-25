"""logger.py — Unified logging + optional JSON logs (institutional baseline)

Goals:
- One-time root logger configuration (console or JSON)
- Optional rotating file handler for audit/ops logs
- Backward-compatible `get_logger()` helper used by legacy modules

Environment variables:
- LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default INFO)
- LOG_FORMAT: console|plain|json (default console)
- LOG_COLOR: 1/0 (default 1 when console)
- LOG_FILE: path to rotating log file (optional)
- LOG_FILE_MAX_BYTES: default 5_000_000
- LOG_FILE_BACKUPS: default 3
- LOG_FORCE_RECONFIG: 1/0 (default 0) reconfigure even if handlers exist
- SERVICE_NAME: emitted in JSON logs (default bitget-bot)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------
# Env controls
# ---------------------------------------------------------------------

_SERVICE_NAME = str(os.getenv("SERVICE_NAME", "bitget-bot")).strip() or "bitget-bot"
_LOG_LEVEL = str(os.getenv("LOG_LEVEL", "INFO")).strip().upper()
_LOG_FORMAT = str(os.getenv("LOG_FORMAT", "console")).strip().lower()
_LOG_COLOR = str(os.getenv("LOG_COLOR", "1")).strip() == "1"
_LOG_FILE = str(os.getenv("LOG_FILE", "")).strip()
_LOG_FILE_MAX_BYTES = int(float(os.getenv("LOG_FILE_MAX_BYTES", "5000000")))
_LOG_FILE_BACKUPS = int(float(os.getenv("LOG_FILE_BACKUPS", "3")))
_LOG_FORCE_RECONFIG = str(os.getenv("LOG_FORCE_RECONFIG", "0")).strip() == "1"

# Custom SUCCESS level between INFO and WARNING
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def _parse_level(level: str) -> int:
    try:
        return int(getattr(logging, level.upper()))
    except Exception:
        return logging.INFO


# ---------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------

_STD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JsonFormatter(logging.Formatter):
    """Structured JSON logs suitable for ingestion (ELK/Datadog/Loki)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "ts": ts,
            "service": _SERVICE_NAME,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        # Merge non-standard extras
        try:
            for k, v in record.__dict__.items():
                if k in _STD_ATTRS:
                    continue
                try:
                    json.dumps(v, default=str)
                    payload[k] = v
                except Exception:
                    payload[k] = str(v)
        except Exception:
            pass

        if record.exc_info:
            try:
                payload["exc"] = self.formatException(record.exc_info)
            except Exception:
                payload["exc"] = "<exc_format_failed>"

        return json.dumps(payload, ensure_ascii=False, default=str)


_COLOR_CODES = {
    "DEBUG": "\033[90m",
    "INFO": "\033[94m",
    "SUCCESS": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[91m",
    "END": "\033[0m",
}


class ConsoleFormatter(logging.Formatter):
    def __init__(self, colored: bool):
        super().__init__("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        self._colored = bool(colored)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not self._colored:
            return msg
        color = _COLOR_CODES.get(record.levelname, "")
        end = _COLOR_CODES.get("END", "")
        return f"{color}{msg}{end}" if color else msg


# ---------------------------------------------------------------------
# Root configuration (idempotent)
# ---------------------------------------------------------------------

_LOCK = threading.Lock()
_CONFIGURED = False


def setup_logging() -> None:
    """Configure root logging exactly once (unless LOG_FORCE_RECONFIG=1)."""
    global _CONFIGURED

    with _LOCK:
        if _CONFIGURED and not _LOG_FORCE_RECONFIG:
            return

        root = logging.getLogger()

        if root.handlers and not _LOG_FORCE_RECONFIG:
            # Respect existing configuration (e.g. when embedded in another runtime)
            _CONFIGURED = True
            return

        if _LOG_FORCE_RECONFIG and root.handlers:
            for h in list(root.handlers):
                try:
                    root.removeHandler(h)
                except Exception:
                    pass

        root.setLevel(_parse_level(_LOG_LEVEL))

        handlers = []

        # Console handler always on
        sh = logging.StreamHandler(sys.stdout)
        if _LOG_FORMAT == "json":
            sh.setFormatter(JsonFormatter())
        elif _LOG_FORMAT in ("plain", "text"):
            sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            colored = bool(_LOG_COLOR and hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            sh.setFormatter(ConsoleFormatter(colored=colored))
        handlers.append(sh)

        # Optional rotating file handler (always JSON, operational/audit-friendly)
        if _LOG_FILE:
            try:
                os.makedirs(os.path.dirname(_LOG_FILE) or ".", exist_ok=True)
                fh = RotatingFileHandler(
                    _LOG_FILE,
                    maxBytes=max(100_000, int(_LOG_FILE_MAX_BYTES)),
                    backupCount=max(1, int(_LOG_FILE_BACKUPS)),
                    encoding="utf-8",
                )
                fh.setFormatter(JsonFormatter())
                handlers.append(fh)
            except Exception:
                # Do not break the bot if file logging is misconfigured.
                pass

        for h in handlers:
            root.addHandler(h)

        _CONFIGURED = True


# ---------------------------------------------------------------------
# Backward-compatible helper (legacy modules)
# ---------------------------------------------------------------------

def _fmt(msg: Any, args: Tuple[Any, ...]) -> str:
    if not args:
        return str(msg)
    try:
        return str(msg) % args
    except Exception:
        return f"{msg} " + " ".join(map(str, args))


class _Compat:
    """Compatibility shim: provides .success() and legacy-style formatting."""

    def __init__(self, name: str):
        setup_logging()
        self._name = str(name or "app")
        self._logger = logging.getLogger(self._name)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.info(_fmt(msg, args))

    def success(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.log(SUCCESS_LEVEL, _fmt(msg, args))

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(_fmt(msg, args))

    def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.error(_fmt(msg, args))

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(_fmt(msg, args))


def get_logger(name: str = "app") -> _Compat:
    return _Compat(name)
