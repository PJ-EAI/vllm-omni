# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Structured gateway logs: one event per line, tagged with who and what served it.

Every log line carries the fields bound with ``context(...)`` in the current
asyncio task, so a session's lines all carry its ``session``, ``robot``,
``model``, ``revision``, ``checkpoint``, ``model_type`` and ``backend`` without
each call site repeating them. ``event(...)`` adds a stable ``event`` name
(``session.open``, ``act.request``, ``route.revision``, ...) and its own fields.

Configured under ``logging:`` in the gateway YAML (``--log-format`` /
``--log-file`` override it)::

    logging:
      format: text        # stderr: text (key=value suffix) | json (one object per line)
      file: logs/gateway.jsonl   # optional; always JSON lines, next to stderr
      level: INFO
      requests: all       # all = one act.request / openpi.request line per inference; errors = failures only
      max_mb: 100         # rotation size of the file
      backups: 5

Find everything one robot saw with, e.g.,
``jq -c 'select(.session == "<id>")' logs/gateway.jsonl``; the robot prints the
same session id when it connects.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_context: ContextVar[dict[str, Any]] = ContextVar("vla_maas_log_context", default={})
# Context keys first, in this order, so text lines line up.
_ORDER = ("session", "robot", "model", "revision", "checkpoint", "model_type", "backend")
REQUEST_LEVELS = {"all": logging.INFO, "errors": logging.DEBUG}


def _clean(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


@contextmanager
def context(**fields: Any) -> Iterator[None]:
    """Tag every log line in this block (and the task it runs in) with ``fields``."""
    token = _context.set({**_context.get(), **_clean(fields)})
    try:
        yield
    finally:
        _context.reset(token)


def bind(**fields: Any) -> None:
    """Add fields learned mid-block (e.g. the backend's reported checkpoint) to the current context."""
    _context.set({**_context.get(), **_clean(fields)})


def current() -> dict[str, Any]:
    return dict(_context.get())


def event(logger: logging.Logger, name: str, msg: str = "", level: int = logging.INFO, **fields: Any) -> None:
    if logger.isEnabledFor(level):
        logger.log(level, msg or name, extra={"event": name, "fields": _clean(fields)})


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    ctx = _context.get()
    out: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
        "level": record.levelname,
        "event": getattr(record, "event", None) or record.name,
        "msg": record.getMessage(),
    }
    out.update((k, ctx[k]) for k in _ORDER if k in ctx)
    out.update((k, v) for k, v in ctx.items() if k not in _ORDER)
    out.update(getattr(record, "fields", {}))
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = _payload(record)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = _payload(record)
        head = f"{out.pop('ts')} {out.pop('level')} {out.pop('event')} {out.pop('msg')}"
        tail = " ".join(f"{k}={json.dumps(v, default=str) if not isinstance(v, str) else v}" for k, v in out.items())
        line = f"{head} [{tail}]" if tail else head
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure(cfg: dict | None = None) -> int:
    """Install handlers on the root logger; returns the level ``act.request`` lines log at."""
    cfg = dict(cfg or {})
    fmt = cfg.get("format", "text")
    if fmt not in ("json", "text"):
        raise ValueError(f"logging.format must be json or text, got {fmt!r}")
    requests = cfg.get("requests", "all")
    if requests not in REQUEST_LEVELS:
        raise ValueError(f"logging.requests must be one of {sorted(REQUEST_LEVELS)}, got {requests!r}")
    stream = logging.StreamHandler()
    stream.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handlers: list[logging.Handler] = [stream]
    if cfg.get("file"):
        path = Path(cfg["file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        file = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=int(float(cfg.get("max_mb", 100)) * 2**20),
            backupCount=int(cfg.get("backups", 5)),
            encoding="utf-8",
        )
        file.setFormatter(JsonFormatter())
        handlers.append(file)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(cfg.get("level", "INFO"))
    # The websockets library logs every failed handshake at INFO; keep those out of the event stream.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    return REQUEST_LEVELS[requests]
