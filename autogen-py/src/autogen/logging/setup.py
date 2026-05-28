"""Structured logging configuration using structlog.

Mirrors autogen.net's structured logging setup with JSON output,
context binding (app_id, session_id, correlation_id), and
dev-friendly console rendering during development.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str = "DEBUG") -> None:
    """Configure structlog as the global logging framework.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound to the given module name.

    Args:
        name: Usually __name__ from the calling module.

    Returns:
        A BoundLogger that supports .info(), .error(), .bind(), etc.
    """
    return structlog.get_logger(name or __name__)
