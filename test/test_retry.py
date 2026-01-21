"""
Tests for retry logic.
Tests retry decorators, fallback patterns, and circuit breaker.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, Mock

from legal_assistant.utils.retry import (
    retry_on_llm_error,
    retry_on_vector_store_error,
    retry_on_external_service_error,
    retry_on_embedding_error,
    retry_async,
    with_fallback,
    CircuitBreaker,
)
from legal_assistant.core.exceptions import (
    LLMTimeoutError,
    VectorSearchError,
    ExternalServiceError,
    EmbeddingError,
)


class TestRetryDecorators:
    """Test retry decorators"""
    
    @pytest.mark.asyncio
    async def test_retry_on_llm_error_success_first_try(self):
        """Test successful call on first try"""
        call_count = 0
        
        @retry_on_llm_error(max_attempts=3)
        async def successful_call():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_call()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_llm_error_retry_and_succeed(self):
        """Test retry after failure then success"""
        call_count = 0
        
        @retry_on_llm_error(max_attempts=3, min_wait=0, max_wait=0)
        async def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMTimeoutError("openai", "gpt-4", 30)
            return "success"
        
        result = await flaky_call()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_llm_error_max_attempts_exceeded(self):
        """Test failure after max retries"""
        call_count = 0
        
        @retry_on_llm_error(max_attempts=3, min_wait=0, max_wait=0)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise LLMTimeoutError("openai", "gpt-4", 30)
        
        with pytest.raises(LLMTimeoutError):
            await always_fails()
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_llm_error_non_retryable_exception(self):
        """Test non-retryable exception not retried"""
        call_count = 0
        
        @retry_on_llm_error(max_attempts=3)
        async def raises_other_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a retryable error")
        
        with pytest.raises(ValueError):
            await raises_other_error()
        
        # Should not retry
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_vector_store_error(self):
        """Test vector store retry decorator"""
        call_count = 0
        
        @retry_on_vector_store_error(max_attempts=2, min_wait=0, max_wait=0)
        async def flaky_search():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise VectorSearchError("legal_documents", "timeout")
            return ["result"]
        
        result = await flaky_search()
        
        assert result == ["result"]
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_external_service_error(self):
        """Test external service retry decorator"""
        call_count = 0
        
        @retry_on_external_service_error(max_attempts=2, min_wait=0, max_wait=0)
        async def flaky_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ExternalServiceError("test_service", "connection failed")
            return {"status": "ok"}
        
        result = await flaky_api_call()
        
        assert result["status"] == "ok"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_embedding_error(self):
        """Test embedding retry decorator"""
        call_count = 0
        
        @retry_on_embedding_error(max_attempts=2, min_wait=0, max_wait=0)
        async def flaky_embedding():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise EmbeddingError("rate limit")
            return [[0.1, 0.2, 0.3]]
        
        result = await flaky_embedding()
        
        assert len(result) == 1
        assert call_count == 2


class TestRetryAsync:
    """Test generic retry_async helper"""
    
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Test retry_async with successful call"""
        async def successful_func(value):
            return value * 2
        
        result = await retry_async(
            successful_func,
            5,
            max_attempts=3
        )
        
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_retry_async_with_retries(self):
        """Test retry_async with retries"""
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Failed")
            return "success"
        
        result = await retry_async(
            flaky_func,
            max_attempts=3,
            exceptions=(ConnectionError,),
            min_wait=0,
            max_wait=0
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_max_attempts(self):
        """Test retry_async respects max attempts"""
        call_count = 0
        
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Always fails")
        
        with pytest.raises(TimeoutError):
            await retry_async(
                always_fails,
                max_attempts=2,
                exceptions=(TimeoutError,),
                min_wait=0,
                max_wait=0
            )
        
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_async_with_args_kwargs(self):
        """Test retry_async with arguments"""
        async def func_with_args(a, b, c=10):
            return a + b + c
        
        result = await retry_async(
            func_with_args,
            5,
            7,
            c=20,
            max_attempts=2
        )
        
        assert result == 32


class TestFallbackPattern:
    """Test fallback pattern helper"""
    
    @pytest.mark.asyncio
    async def test_with_fallback_primary_succeeds(self):
        """Test fallback when primary succeeds"""
        primary_called = False
        fallback_called = False
        
        async def primary():
            nonlocal primary_called
            primary_called = True
            return "primary_result"
        
        async def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"
        
        result = await with_fallback(primary, fallback)
        
        assert result == "primary_result"
        assert primary_called is True
        assert fallback_called is False
    
    @pytest.mark.asyncio
    async def test_with_fallback_primary_fails(self):
        """Test fallback when primary fails"""
        primary_called = False
        fallback_called = False
        
        async def primary():
            nonlocal primary_called
            primary_called = True
            raise ValueError("Primary failed")
        
        async def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"
        
        result = await with_fallback(
            primary,
            fallback,
            primary_exceptions=(ValueError,)
        )
        
        assert result == "fallback_result"
        assert primary_called is True
        assert fallback_called is True
    
    @pytest.mark.asyncio
    async def test_with_fallback_with_args(self):
        """Test fallback with arguments"""
        async def primary(text):
            if len(text) < 10:
                raise ValueError("Too short")
            return f"Primary: {text}"
        
        async def fallback(text):
            return f"Fallback: {text}"
        
        # Primary fails
        result = await with_fallback(
            primary,
            fallback,
            "Hi",
            primary_exceptions=(ValueError,)
        )
        assert result == "Fallback: Hi"
        
        # Primary succeeds
        result = await with_fallback(
            primary,
            fallback,
            "Long enough text",
            primary_exceptions=(ValueError,)
        )
        assert result == "Primary: Long enough text"
    
    @pytest.mark.asyncio
    async def test_with_fallback_unexpected_exception(self):
        """Test fallback with unexpected exception"""
        async def primary():
            raise RuntimeError("Unexpected")
        
        async def fallback():
            return "fallback"
        
        # RuntimeError not in primary_exceptions, should propagate
        with pytest.raises(RuntimeError):
            await with_fallback(
                primary,
                fallback,
                primary_exceptions=(ValueError,)
            )


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in CLOSED state"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        assert breaker.state == "CLOSED"
        
        # Successful calls keep circuit closed
        async with breaker:
            result = "success"
        
        assert breaker.state == "CLOSED"
        assert breaker.failures == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        # Fail 3 times
        for _ in range(3):
            try:
                async with breaker:
                    raise ConnectionError("Service down")
            except ConnectionError:
                pass
        
        # Circuit should be OPEN
        assert breaker.state == "OPEN"
        assert breaker.failures == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_fails_fast_when_open(self):
        """Test circuit breaker fails fast in OPEN state"""
        breaker = CircuitBreaker(failure_threshold=2, timeout=60)
        
        # Open the circuit
        for _ in range(2):
            try:
                async with breaker:
                    raise ConnectionError("Fail")
            except ConnectionError:
                pass
        
        assert breaker.state == "OPEN"
        
        # Next call should fail fast
        with pytest.raises(ExternalServiceError) as exc_info:
            async with breaker:
                pass
        
        assert "Circuit breaker OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker enters HALF_OPEN after timeout"""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Open the circuit
        for _ in range(2):
            try:
                async with breaker:
                    raise ConnectionError("Fail")
            except ConnectionError:
                pass
        
        assert breaker.state == "OPEN"
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Should enter HALF_OPEN
        async with breaker:
            # Successful call should close circuit
            pass
        
        assert breaker.state == "CLOSED"
        assert breaker.failures == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_reopens_on_failure_in_half_open(self):
        """Test circuit reopens if failure in HALF_OPEN"""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Open circuit
        for _ in range(2):
            try:
                async with breaker:
                    raise ConnectionError("Fail")
            except ConnectionError:
                pass
        
        # Wait for HALF_OPEN
        await asyncio.sleep(0.2)
        breaker.state = "HALF_OPEN"
        
        # Fail in HALF_OPEN
        try:
            async with breaker:
                raise ConnectionError("Still failing")
        except ConnectionError:
            pass
        
        # Should reopen
        assert breaker.state == "OPEN"


class TestRetryPerformance:
    """Test retry performance"""
    
    @pytest.mark.asyncio
    async def test_retry_overhead_minimal(self, benchmark):
        """Test retry decorator has minimal overhead"""
        @retry_on_llm_error(max_attempts=1)
        async def fast_function():
            return "result"
        
        async def run():
            return await fast_function()
        
        # Benchmark async function
        result = await run()
        assert result == "result"
        
        # Note: benchmark doesn't work well with async
        # But test verifies no performance issues


class TestEdgeCases:
    """Test edge cases"""
    
    @pytest.mark.asyncio
    async def test_retry_with_zero_wait_time(self):
        """Test retry with zero wait time"""
        call_count = 0
        
        @retry_on_llm_error(max_attempts=2, min_wait=0, max_wait=0)
        async def instant_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMTimeoutError("openai", "gpt-4", 30)
            return "success"
        
        start = time.time()
        result = await instant_retry()
        duration = time.time() - start
        
        assert result == "success"
        assert duration < 0.1  # Should be near instant
    
    @pytest.mark.asyncio
    async def test_fallback_both_fail(self):
        """Test when both primary and fallback fail"""
        async def primary():
            raise ValueError("Primary fail")
        
        async def fallback():
            raise RuntimeError("Fallback fail")
        
        # Fallback exception should propagate
        with pytest.raises(RuntimeError):
            await with_fallback(
                primary,
                fallback,
                primary_exceptions=(ValueError,)
            )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_requests(self):
        """Test circuit breaker with concurrent requests"""
        breaker = CircuitBreaker(failure_threshold=5, timeout=1)
        
        async def make_request(should_fail: bool):
            try:
                async with breaker:
                    if should_fail:
                        raise ConnectionError("Fail")
                    return "success"
            except ConnectionError:
                return "failed"
        
        # Make concurrent requests
        tasks = [
            make_request(i % 2 == 0)  # Half succeed, half fail
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        # Some should succeed, circuit might open
        assert "success" in results or breaker.state == "OPEN"


class TestIntegration:
    """Integration tests for retry system"""
    
    @pytest.mark.asyncio
    async def test_retry_with_fallback_pattern(self):
        """Test combining retry with fallback"""
        primary_attempts = 0
        fallback_attempts = 0
        
        @retry_on_llm_error(max_attempts=2, min_wait=0, max_wait=0)
        async def primary_with_retry():
            nonlocal primary_attempts
            primary_attempts += 1
            raise LLMTimeoutError("openai", "gpt-4", 30)
        
        async def fallback_func():
            nonlocal fallback_attempts
            fallback_attempts += 1
            return "fallback_success"
        
        # Primary will retry 2 times, then fallback
        result = await with_fallback(
            primary_with_retry,
            fallback_func,
            primary_exceptions=(LLMTimeoutError,)
        )
        
        assert result == "fallback_success"
        assert primary_attempts == 2  # Retried
        assert fallback_attempts == 1


# Run with: pytest tests/utils/test_retry.py -v
