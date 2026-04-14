from __future__ import annotations

import contextlib
import contextvars
import logging


_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id_var.get() or "-"
        return True


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if any(isinstance(existing_filter, CorrelationIdFilter) for existing_filter in root_logger.filters):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] correlation_id=%(correlation_id)s %(message)s"
        )
    )
    handler.addFilter(CorrelationIdFilter())
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    root_logger.addFilter(CorrelationIdFilter())


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


@contextlib.contextmanager
def correlation_scope(correlation_id: str):
    token = _correlation_id_var.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id_var.reset(token)