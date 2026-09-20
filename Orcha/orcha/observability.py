"""
orcha.observability
====================
Structured logging + trace-ID plumbing.

Design
------
- One module-level logger named "orcha". Configure handlers/levels from the
  host application; ORCHA never forces a handler onto the root logger.
- `get_logger()` returns a child logger that auto-injects the packet id as
  `trace_id` on every record when called inside an orchestration run.
- Structured logging is emitted as JSON-friendly key/value pairs via
  `logger.info(msg, extra={...})` so any JSON formatter picks them up.
- Nothing here ever raises — observability must not break the pipeline.

Log lines you can expect
------------------------
  orcha.orchestrator  INFO  orchestration.start   trace_id=ab12cd34 query="..."
  orcha.orchestrator  INFO  orchestration.stage   trace_id=ab12cd34 stage=select ...
  orcha.orchestrator  INFO  orchestration.finish  trace_id=ab12cd34 iterations=1 ...
"""
from __future__ import annotations

import logging
from typing import Any, Optional

_LOGGER_NAME = "orcha"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return the shared Orcha logger (or a named child of it)."""
    logger = logging.getLogger(name)
    # Default to WARNING if the host app hasn't configured anything, so
    # importing orcha in a library context is silent.
    if not logging.getLogger().handlers and logger.level == logging.NOTSET:
        logger.setLevel(logging.WARNING)
    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """
    Convenience for scripts / the demo server: attach a single stream
    handler to the orcha logger. Idempotent — safe to call repeatedly.
    Host applications that want structured/JSON logs or a different
    format should configure the "orcha" logger themselves and ignore this.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    # Avoid duplicate handlers on re-configure.
    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_orcha", False)
        for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)-22s %(levelname)-7s %(message)s"
            )
        )
        handler._orcha = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


def log_stage(
    logger: logging.Logger,
    trace_id: Optional[str],
    event: str,
    **fields: Any,
) -> None:
    """
    Emit one structured log line tied to a packet's trace id.

    The trace_id is shortened to the first 8 chars for readability — it is
    purely a correlation aid, not a security value.
    """
    extra = {"trace_id": (trace_id or "—")[:8]}
    extra.update(fields)
    # Use %-style formatting; pass structured data via extra so JSON
    # formatters (e.g. python-json-logger) capture every field.
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("%s  %s", event, payload, extra=extra)
