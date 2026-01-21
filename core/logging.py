"""
Production-ready structured logging using structlog.
Provides JSON logging with trace IDs, context propagation, and performance metrics.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import structlog
from structlog.types import EventDict, Processor

from legal_assistant.core.config import get_settings

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def add_request_context(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Add request context (request_id, user_id) to every log entry.
    """
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    
    user_id = user_id_var.get()
    if user_id:
        event_dict["user_id"] = user_id
    
    return event_dict


def add_app_context(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Add application metadata to logs.
    """
    settings = get_settings()
    event_dict["app_name"] = settings.app_name
    event_dict["app_version"] = settings.app_version
    event_dict["environment"] = settings.environment
    return event_dict


def drop_color_message_key(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Remove colored message key used by ConsoleRenderer.
    Prevents duplicate messages in JSON logs.
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging() -> None:
    """
    Configure structlog for production use.
    
    Features:
    - JSON output for production
    - Colored console output for development
    - Request ID tracking
    - Performance metrics
    - Exception formatting
    """
    settings = get_settings()
    
    # Determine processors based on environment
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_request_context,
        add_app_context,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.monitoring.log_format == "json" or settings.is_production:
        # Production: JSON logging
        processors = shared_processors + [
            drop_color_message_key,
            structlog.processors.JSONRenderer(sort_keys=True),
        ]
    else:
        # Development: Colored console logging
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    log_level = getattr(logging, settings.monitoring.log_level)
    
    # Root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # File logging (optional)
    if settings.monitoring.log_file:
        handler = logging.FileHandler(
            settings.monitoring.log_file,
            encoding="utf-8"
        )
        handler.setLevel(log_level)
        logging.root.addHandler(handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured structlog logger
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("user_query", query="найти договор", user_id="user123")
    """
    return structlog.get_logger(name)


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracing.
    
    Returns:
        UUID4 string without dashes
    """
    return uuid.uuid4().hex


def set_request_context(request_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """
    Set request context for the current execution.
    
    Args:
        request_id: Request ID (generated if None)
        user_id: User ID (optional)
    
    Returns:
        The request_id that was set
    
    Example:
        >>> request_id = set_request_context(user_id="user123")
        >>> logger.info("processing_query")  # Will include request_id and user_id
    """
    if request_id is None:
        request_id = generate_request_id()
    
    request_id_var.set(request_id)
    
    if user_id:
        user_id_var.set(user_id)
    
    return request_id


def clear_request_context() -> None:
    """
    Clear request context.
    Should be called at the end of request processing.
    """
    request_id_var.set(None)
    user_id_var.set(None)


class LoggerMixin:
    """
    Mixin to add logger to any class.
    
    Example:
        >>> class MyService(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("processing")
    """
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class"""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Get logger
    logger = get_logger(__name__)
    
    # Test basic logging
    logger.info("application_started")
    
    # Test with context
    set_request_context(user_id="user123")
    logger.info(
        "user_query_received",
        query="найти договор аренды",
        query_length=20
    )
    
    # Test error logging
    try:
        1 / 0
    except Exception:
        logger.exception(
            "division_error",
            numerator=1,
            denominator=0
        )
    
    # Test performance logging
    import time
    start = time.time()
    time.sleep(0.1)
    duration = time.time() - start
    logger.info(
        "operation_completed",
        operation="document_parsing",
        duration_seconds=duration,
        success=True
    )
    
    # Clear context
    clear_request_context()
    logger.info("request_completed")
