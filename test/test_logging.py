"""
Tests for logging module.
Tests structured logging, context propagation, and request tracking.
"""

import json
import logging
from io import StringIO
from typing import Generator
import pytest
import structlog

from legal_assistant.core.logging import (
    get_logger,
    setup_logging,
    set_request_context,
    clear_request_context,
    generate_request_id,
    LoggerMixin,
    request_id_var,
    user_id_var,
)


@pytest.fixture
def reset_logging() -> Generator[None, None, None]:
    """Reset logging configuration after test"""
    yield
    # Reset structlog
    structlog.reset_defaults()
    # Reset logging
    logging.root.handlers = []


@pytest.fixture
def capture_logs() -> Generator[StringIO, None, None]:
    """Capture log output to StringIO"""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    
    # Add to root logger
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)
    
    yield stream
    
    # Cleanup
    logging.root.removeHandler(handler)


class TestLoggerCreation:
    """Test logger creation and configuration"""
    
    def test_get_logger(self, reset_logging):
        """Test getting a logger instance"""
        logger = get_logger(__name__)
        
        assert logger is not None
        assert isinstance(logger, structlog.stdlib.BoundLogger)
    
    def test_logger_names(self, reset_logging):
        """Test logger with different names"""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        # Different loggers for different names
        assert logger1 is not None
        assert logger2 is not None
        # Note: structlog may return same instance, but names are tracked
    
    def test_setup_logging(self, reset_logging, monkeypatch):
        """Test logging setup"""
        # Mock settings
        from legal_assistant.core.config import Settings, MonitoringSettings
        
        mock_monitoring = MonitoringSettings(
            log_level="INFO",
            log_format="json",
            enable_langsmith=False,
            enable_prometheus=False
        )
        
        class MockSettings:
            monitoring = mock_monitoring
            app_name = "Test App"
            app_version = "1.0.0"
            environment = "test"
            
            @property
            def is_production(self):
                return False
        
        # No errors during setup
        setup_logging()


class TestRequestContext:
    """Test request context management"""
    
    def test_generate_request_id(self):
        """Test request ID generation"""
        request_id = generate_request_id()
        
        assert request_id is not None
        assert len(request_id) == 32  # UUID4 hex without dashes
        assert "-" not in request_id
        
        # Should be unique
        request_id2 = generate_request_id()
        assert request_id != request_id2
    
    def test_set_request_context_with_id(self):
        """Test setting context with specific request ID"""
        request_id = "test-request-123"
        user_id = "user-456"
        
        returned_id = set_request_context(
            request_id=request_id,
            user_id=user_id
        )
        
        assert returned_id == request_id
        assert request_id_var.get() == request_id
        assert user_id_var.get() == user_id
        
        # Cleanup
        clear_request_context()
    
    def test_set_request_context_auto_generate(self):
        """Test auto-generating request ID"""
        user_id = "user-789"
        
        request_id = set_request_context(user_id=user_id)
        
        assert request_id is not None
        assert len(request_id) == 32
        assert request_id_var.get() == request_id
        assert user_id_var.get() == user_id
        
        # Cleanup
        clear_request_context()
    
    def test_set_request_context_without_user(self):
        """Test setting context without user ID"""
        request_id = set_request_context()
        
        assert request_id is not None
        assert request_id_var.get() == request_id
        assert user_id_var.get() is None
        
        # Cleanup
        clear_request_context()
    
    def test_clear_request_context(self):
        """Test clearing context"""
        set_request_context(
            request_id="test-123",
            user_id="user-456"
        )
        
        # Verify set
        assert request_id_var.get() is not None
        assert user_id_var.get() is not None
        
        # Clear
        clear_request_context()
        
        # Verify cleared
        assert request_id_var.get() is None
        assert user_id_var.get() is None


class TestStructuredLogging:
    """Test structured logging output"""
    
    def test_log_with_context(self, reset_logging, capture_logs):
        """Test logging with request context"""
        setup_logging()
        logger = get_logger(__name__)
        
        # Set context
        request_id = set_request_context(
            request_id="req-123",
            user_id="user-456"
        )
        
        # Log message
        logger.info("test_event", key1="value1", key2=42)
        
        # Get output
        output = capture_logs.getvalue()
        
        # Should contain event and context
        assert "test_event" in output
        assert "req-123" in output or "request_id" in output
        
        # Cleanup
        clear_request_context()
    
    def test_log_without_context(self, reset_logging, capture_logs):
        """Test logging without context"""
        setup_logging()
        logger = get_logger(__name__)
        
        # Clear any existing context
        clear_request_context()
        
        # Log without context
        logger.info("no_context_event", data="test")
        
        output = capture_logs.getvalue()
        assert "no_context_event" in output
    
    def test_log_levels(self, reset_logging, capture_logs):
        """Test different log levels"""
        setup_logging()
        logger = get_logger(__name__)
        
        # Log at different levels
        logger.debug("debug_message")
        logger.info("info_message")
        logger.warning("warning_message")
        logger.error("error_message")
        
        output = capture_logs.getvalue()
        
        # Check all levels present (depending on log level config)
        assert "info_message" in output
        assert "warning_message" in output
        assert "error_message" in output
    
    def test_log_with_exception(self, reset_logging, capture_logs):
        """Test logging exceptions"""
        setup_logging()
        logger = get_logger(__name__)
        
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception("division_error", numerator=1, denominator=0)
        
        output = capture_logs.getvalue()
        
        assert "division_error" in output
        assert "ZeroDivisionError" in output or "exception" in output.lower()
    
    def test_log_with_structured_data(self, reset_logging, capture_logs):
        """Test logging structured data"""
        setup_logging()
        logger = get_logger(__name__)
        
        data = {
            "user_id": "123",
            "action": "query",
            "duration_ms": 150,
            "success": True
        }
        
        logger.info("user_action", **data)
        
        output = capture_logs.getvalue()
        assert "user_action" in output


class TestLoggerMixin:
    """Test LoggerMixin for classes"""
    
    def test_logger_mixin_basic(self, reset_logging):
        """Test basic LoggerMixin usage"""
        class MyService(LoggerMixin):
            def process(self):
                self.logger.info("processing")
                return "done"
        
        service = MyService()
        result = service.process()
        
        assert result == "done"
        assert hasattr(service, "logger")
    
    def test_logger_mixin_cached(self, reset_logging):
        """Test that logger is cached"""
        class MyService(LoggerMixin):
            pass
        
        service = MyService()
        logger1 = service.logger
        logger2 = service.logger
        
        # Same instance
        assert logger1 is logger2
    
    def test_logger_mixin_name(self, reset_logging):
        """Test logger gets correct name"""
        class MyTestService(LoggerMixin):
            pass
        
        service = MyTestService()
        logger = service.logger
        
        # Logger should be created (no error)
        assert logger is not None


class TestContextPropagation:
    """Test context propagation across async operations"""
    
    @pytest.mark.asyncio
    async def test_async_context_propagation(self):
        """Test context is preserved in async functions"""
        import asyncio
        
        # Set context
        request_id = set_request_context(
            request_id="async-123",
            user_id="async-user"
        )
        
        async def async_operation():
            # Context should be available here
            return request_id_var.get(), user_id_var.get()
        
        # Run async operation
        rid, uid = await async_operation()
        
        assert rid == "async-123"
        assert uid == "async-user"
        
        # Cleanup
        clear_request_context()
    
    @pytest.mark.asyncio
    async def test_concurrent_contexts(self):
        """Test that contexts are isolated in concurrent tasks"""
        import asyncio
        
        async def task_with_context(task_id: str):
            # Each task sets its own context
            set_request_context(request_id=f"req-{task_id}")
            await asyncio.sleep(0.01)
            # Should still have its own context
            return request_id_var.get()
        
        # Run multiple tasks
        tasks = [
            task_with_context("1"),
            task_with_context("2"),
            task_with_context("3")
        ]
        results = await asyncio.gather(*tasks)
        
        # Each should have correct context
        assert "req-1" in results
        assert "req-2" in results
        assert "req-3" in results
        
        # Cleanup
        clear_request_context()


class TestLoggingPerformance:
    """Test logging performance"""
    
    def test_logger_creation_performance(self, reset_logging, benchmark):
        """Test logger creation is fast"""
        def create_logger():
            return get_logger(__name__)
        
        # Should be very fast (microseconds)
        result = benchmark(create_logger)
        assert result is not None
    
    def test_logging_performance(self, reset_logging, capture_logs, benchmark):
        """Test logging performance"""
        setup_logging()
        logger = get_logger(__name__)
        
        def log_message():
            logger.info("test_message", key="value", number=42)
        
        # Should be fast (< 1ms per log)
        benchmark(log_message)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_log_with_none_values(self, reset_logging, capture_logs):
        """Test logging None values"""
        setup_logging()
        logger = get_logger(__name__)
        
        logger.info("test_event", value=None, other="data")
        
        # Should not crash
        output = capture_logs.getvalue()
        assert "test_event" in output
    
    def test_log_with_complex_types(self, reset_logging, capture_logs):
        """Test logging complex types"""
        setup_logging()
        logger = get_logger(__name__)
        
        # Log with list, dict, etc
        logger.info(
            "complex_data",
            list_data=[1, 2, 3],
            dict_data={"key": "value"},
            tuple_data=(1, 2)
        )
        
        # Should not crash
        output = capture_logs.getvalue()
        assert "complex_data" in output
    
    def test_unicode_in_logs(self, reset_logging, capture_logs):
        """Test Unicode characters in logs"""
        setup_logging()
        logger = get_logger(__name__)
        
        logger.info("unicode_test", message="Привет мир 你好世界 🎉")
        
        # Should handle Unicode
        output = capture_logs.getvalue()
        assert "unicode_test" in output


class TestIntegration:
    """Integration tests for logging system"""
    
    def test_full_logging_flow(self, reset_logging, capture_logs):
        """Test complete logging flow"""
        # Setup
        setup_logging()
        logger = get_logger(__name__)
        
        # Set context
        request_id = set_request_context(user_id="integration-user")
        
        # Log various events
        logger.info("request_started", endpoint="/query")
        logger.info("processing", step="parsing", duration=0.5)
        logger.info("request_completed", status="success")
        
        # Get output
        output = capture_logs.getvalue()
        
        # Verify all events logged
        assert "request_started" in output
        assert "processing" in output
        assert "request_completed" in output
        
        # Cleanup
        clear_request_context()
    
    def test_error_flow(self, reset_logging, capture_logs):
        """Test error logging flow"""
        setup_logging()
        logger = get_logger(__name__)
        
        set_request_context(request_id="error-test")
        
        try:
            # Simulate error
            raise ValueError("Test error")
        except ValueError:
            logger.exception("error_occurred", component="test")
        
        output = capture_logs.getvalue()
        assert "error_occurred" in output
        
        clear_request_context()


# Run with: pytest tests/core/test_logging.py -v
