"""
Tests for metrics module.
Tests Prometheus metrics, tracking helpers, and exporters.
"""

import time
import pytest
from prometheus_client import REGISTRY as DEFAULT_REGISTRY

from legal_assistant.core.metrics import (
    REGISTRY,
    request_count,
    request_duration,
    llm_request_count,
    llm_tokens_used,
    llm_cost_usd,
    document_processed,
    vector_search_count,
    errors_total,
    track_time,
    track_active,
    track_llm_tokens,
    track_error,
    export_metrics,
    MetricStatus,
    active_requests,
)
from legal_assistant.core.exceptions import FileValidationError


@pytest.fixture
def reset_metrics():
    """Reset metrics before each test"""
    # Note: In real tests, we'd use a separate test registry
    # For now, we accept that metrics accumulate
    yield
    # Metrics can't be truly reset without restarting Python
    # But tests should still pass as they check relative changes


class TestMetricCreation:
    """Test metric creation and types"""
    
    def test_counter_exists(self):
        """Test that counters are created"""
        assert request_count is not None
        assert llm_request_count is not None
        assert errors_total is not None
    
    def test_histogram_exists(self):
        """Test that histograms are created"""
        assert request_duration is not None
    
    def test_gauge_exists(self):
        """Test that gauges are created"""
        assert active_requests is not None
    
    def test_metric_registry(self):
        """Test custom registry is used"""
        assert REGISTRY is not None
        assert REGISTRY != DEFAULT_REGISTRY


class TestRequestMetrics:
    """Test request metrics"""
    
    def test_request_count_increment(self, reset_metrics):
        """Test incrementing request counter"""
        # Get initial value
        before = request_count.labels(
            endpoint="/test",
            method="POST",
            status="success"
        )._value.get()
        
        # Increment
        request_count.labels(
            endpoint="/test",
            method="POST",
            status="success"
        ).inc()
        
        # Check increased
        after = request_count.labels(
            endpoint="/test",
            method="POST",
            status="success"
        )._value.get()
        
        assert after == before + 1
    
    def test_request_duration_observe(self, reset_metrics):
        """Test observing request duration"""
        # Observe a duration
        request_duration.labels(
            endpoint="/query",
            method="POST"
        ).observe(1.5)
        
        # Should not raise error
        # Actual values are accumulated, so just test it works
    
    def test_active_requests_gauge(self, reset_metrics):
        """Test active requests gauge"""
        initial = active_requests.labels(endpoint="/test")._value.get()
        
        # Increment
        active_requests.labels(endpoint="/test").inc()
        assert active_requests.labels(endpoint="/test")._value.get() == initial + 1
        
        # Decrement
        active_requests.labels(endpoint="/test").dec()
        assert active_requests.labels(endpoint="/test")._value.get() == initial


class TestLLMMetrics:
    """Test LLM-specific metrics"""
    
    def test_llm_request_count(self, reset_metrics):
        """Test LLM request counter"""
        before = llm_request_count.labels(
            provider="openai",
            model="gpt-4",
            status="success"
        )._value.get()
        
        llm_request_count.labels(
            provider="openai",
            model="gpt-4",
            status="success"
        ).inc()
        
        after = llm_request_count.labels(
            provider="openai",
            model="gpt-4",
            status="success"
        )._value.get()
        
        assert after == before + 1
    
    def test_llm_tokens_tracking(self, reset_metrics):
        """Test token usage tracking"""
        before_prompt = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="prompt"
        )._value.get()
        
        before_completion = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="completion"
        )._value.get()
        
        # Track tokens
        llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="prompt"
        ).inc(500)
        
        llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="completion"
        ).inc(200)
        
        # Check increased
        after_prompt = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="prompt"
        )._value.get()
        
        after_completion = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="completion"
        )._value.get()
        
        assert after_prompt == before_prompt + 500
        assert after_completion == before_completion + 200
    
    def test_llm_cost_tracking(self, reset_metrics):
        """Test cost tracking"""
        before = llm_cost_usd.labels(
            provider="openai",
            model="gpt-4"
        )._value.get()
        
        llm_cost_usd.labels(
            provider="openai",
            model="gpt-4"
        ).inc(0.015)
        
        after = llm_cost_usd.labels(
            provider="openai",
            model="gpt-4"
        )._value.get()
        
        assert after >= before + 0.014  # Float precision


class TestDocumentMetrics:
    """Test document processing metrics"""
    
    def test_document_processed_counter(self, reset_metrics):
        """Test document processed counter"""
        before = document_processed.labels(
            parser="llamaparse",
            status="success"
        )._value.get()
        
        document_processed.labels(
            parser="llamaparse",
            status="success"
        ).inc()
        
        after = document_processed.labels(
            parser="llamaparse",
            status="success"
        )._value.get()
        
        assert after == before + 1


class TestVectorMetrics:
    """Test vector store metrics"""
    
    def test_vector_search_counter(self, reset_metrics):
        """Test vector search counter"""
        before = vector_search_count.labels(
            collection="legal_documents",
            status="success"
        )._value.get()
        
        vector_search_count.labels(
            collection="legal_documents",
            status="success"
        ).inc()
        
        after = vector_search_count.labels(
            collection="legal_documents",
            status="success"
        )._value.get()
        
        assert after == before + 1


class TestErrorMetrics:
    """Test error tracking metrics"""
    
    def test_errors_counter(self, reset_metrics):
        """Test error counter"""
        before = errors_total.labels(
            component="test",
            error_type="TestError"
        )._value.get()
        
        errors_total.labels(
            component="test",
            error_type="TestError"
        ).inc()
        
        after = errors_total.labels(
            component="test",
            error_type="TestError"
        )._value.get()
        
        assert after == before + 1


class TestTrackingHelpers:
    """Test metric tracking helper functions"""
    
    def test_track_time_context_manager(self, reset_metrics):
        """Test track_time context manager"""
        with track_time(
            request_duration,
            {"endpoint": "/test", "method": "GET"}
        ):
            time.sleep(0.01)
        
        # Should have recorded duration
        # We can't easily assert exact value, but it shouldn't error
    
    def test_track_active_context_manager(self, reset_metrics):
        """Test track_active context manager"""
        initial = active_requests.labels(endpoint="/test")._value.get()
        
        with track_active(active_requests, {"endpoint": "/test"}):
            # Inside context, should be incremented
            current = active_requests.labels(endpoint="/test")._value.get()
            assert current == initial + 1
        
        # After context, should be back to initial
        final = active_requests.labels(endpoint="/test")._value.get()
        assert final == initial
    
    def test_track_active_exception_handling(self, reset_metrics):
        """Test track_active handles exceptions"""
        initial = active_requests.labels(endpoint="/test")._value.get()
        
        try:
            with track_active(active_requests, {"endpoint": "/test"}):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should still decrement after exception
        final = active_requests.labels(endpoint="/test")._value.get()
        assert final == initial
    
    def test_track_llm_tokens_helper(self, reset_metrics):
        """Test track_llm_tokens helper function"""
        before_prompt = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="prompt"
        )._value.get()
        
        before_completion = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="completion"
        )._value.get()
        
        before_cost = llm_cost_usd.labels(
            provider="openai",
            model="gpt-4"
        )._value.get()
        
        # Track tokens
        track_llm_tokens(
            provider="openai",
            model="gpt-4",
            prompt_tokens=500,
            completion_tokens=200,
            cost_usd=0.015
        )
        
        # Check all metrics updated
        after_prompt = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="prompt"
        )._value.get()
        
        after_completion = llm_tokens_used.labels(
            provider="openai",
            model="gpt-4",
            token_type="completion"
        )._value.get()
        
        after_cost = llm_cost_usd.labels(
            provider="openai",
            model="gpt-4"
        )._value.get()
        
        assert after_prompt == before_prompt + 500
        assert after_completion == before_completion + 200
        assert after_cost >= before_cost + 0.014
    
    def test_track_error_helper(self, reset_metrics):
        """Test track_error helper function"""
        before = errors_total.labels(
            component="test_component",
            error_type="FileValidationError"
        )._value.get()
        
        # Track error
        exc = FileValidationError(
            message="Test error",
            filename="test.pdf"
        )
        track_error("test_component", exc)
        
        after = errors_total.labels(
            component="test_component",
            error_type="FileValidationError"
        )._value.get()
        
        assert after == before + 1


class TestMetricStatus:
    """Test MetricStatus enum"""
    
    def test_metric_status_values(self):
        """Test MetricStatus enum values"""
        assert MetricStatus.SUCCESS == "success"
        assert MetricStatus.ERROR == "error"
        assert MetricStatus.TIMEOUT == "timeout"
        assert MetricStatus.FALLBACK == "fallback"
    
    def test_metric_status_usage(self):
        """Test using MetricStatus in metrics"""
        request_count.labels(
            endpoint="/test",
            method="POST",
            status=MetricStatus.SUCCESS.value
        ).inc()
        
        # Should not error


class TestMetricsExport:
    """Test metrics export"""
    
    def test_export_metrics_format(self, reset_metrics):
        """Test export_metrics returns correct format"""
        data, content_type = export_metrics()
        
        assert isinstance(data, bytes)
        assert content_type is not None
        assert "text/plain" in content_type or "prometheus" in content_type
    
    def test_export_metrics_content(self, reset_metrics):
        """Test export_metrics contains metrics"""
        # Add some metrics
        request_count.labels(
            endpoint="/test",
            method="GET",
            status="success"
        ).inc()
        
        data, _ = export_metrics()
        content = data.decode('utf-8')
        
        # Should contain metric names
        assert "legal_assistant" in content
    
    def test_export_metrics_parseable(self, reset_metrics):
        """Test exported metrics are parseable"""
        data, _ = export_metrics()
        content = data.decode('utf-8')
        
        # Should have HELP and TYPE lines
        lines = content.split('\n')
        assert any(line.startswith('# HELP') for line in lines)
        assert any(line.startswith('# TYPE') for line in lines)


class TestMetricsIntegration:
    """Integration tests for metrics system"""
    
    def test_full_request_flow(self, reset_metrics):
        """Test complete request metrics flow"""
        endpoint = "/integration_test"
        
        # Track full flow
        with track_active(active_requests, {"endpoint": endpoint}):
            with track_time(
                request_duration,
                {"endpoint": endpoint, "method": "POST"}
            ):
                time.sleep(0.01)
                
                # Simulate success
                request_count.labels(
                    endpoint=endpoint,
                    method="POST",
                    status="success"
                ).inc()
        
        # All metrics should be updated
        # Can't assert exact values but shouldn't error
    
    def test_error_tracking_flow(self, reset_metrics):
        """Test error tracking flow"""
        component = "integration_test"
        
        try:
            raise FileValidationError(
                message="Test error",
                filename="test.pdf"
            )
        except FileValidationError as e:
            track_error(component, e)
        
        # Error should be tracked
        after = errors_total.labels(
            component=component,
            error_type="FileValidationError"
        )._value.get()
        
        assert after > 0


class TestPerformance:
    """Test metrics performance"""
    
    def test_counter_performance(self, benchmark):
        """Test counter increment performance"""
        def increment_counter():
            request_count.labels(
                endpoint="/perf",
                method="GET",
                status="success"
            ).inc()
        
        # Should be very fast (microseconds)
        benchmark(increment_counter)
    
    def test_histogram_performance(self, benchmark):
        """Test histogram observe performance"""
        def observe_duration():
            request_duration.labels(
                endpoint="/perf",
                method="GET"
            ).observe(1.0)
        
        benchmark(observe_duration)


# Run with: pytest tests/core/test_metrics.py -v
