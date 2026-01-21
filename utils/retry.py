"""
Retry logic with exponential backoff using tenacity.
Handles transient failures with smart retry strategies.
"""

from functools import wraps
from typing import Any, Callable, Optional, Type

from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    before_sleep_log,
    after_log,
)
import logging

from legal_assistant.core.logging import get_logger
from legal_assistant.core.exceptions import (
    LLMTimeoutError,
    VectorSearchError,
    ExternalServiceError,
    RerankError,
    EmbeddingError,
)

logger = get_logger(__name__)


# ============================================================================
# RETRY DECORATORS
# ============================================================================

def retry_on_llm_error(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10
):
    """
    Retry decorator for LLM requests.
    
    Args:
        max_attempts: Maximum retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
    
    Example:
        >>> @retry_on_llm_error(max_attempts=3)
        ... async def call_llm(prompt: str):
        ...     return await client.chat.completions.create(...)
    """
    return retry(
        retry=retry_if_exception_type((
            LLMTimeoutError,
            ConnectionError,
            TimeoutError,
        )),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=min_wait,
            max=max_wait
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
        reraise=True
    )


def retry_on_vector_store_error(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 5
):
    """
    Retry decorator for vector store operations.
    
    Example:
        >>> @retry_on_vector_store_error()
        ... async def search_vectors(query: str):
        ...     return await qdrant.search(...)
    """
    return retry(
        retry=retry_if_exception_type((
            VectorSearchError,
            ConnectionError,
        )),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=min_wait,
            max=max_wait
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


def retry_on_external_service_error(
    max_attempts: int = 2,
    min_wait: int = 2,
    max_wait: int = 8
):
    """
    Retry decorator for external service calls.
    
    Example:
        >>> @retry_on_external_service_error()
        ... async def rerank_results(query: str, docs: list):
        ...     return await cohere.rerank(...)
    """
    return retry(
        retry=retry_if_exception_type((
            ExternalServiceError,
            RerankError,
            ConnectionError,
            TimeoutError,
        )),
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(
            multiplier=1,
            min=min_wait,
            max=max_wait
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


def retry_on_embedding_error(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 5
):
    """
    Retry decorator for embedding generation.
    
    Example:
        >>> @retry_on_embedding_error()
        ... async def generate_embeddings(texts: list[str]):
        ...     return await openai.embeddings.create(...)
    """
    return retry(
        retry=retry_if_exception_type((
            EmbeddingError,
            ConnectionError,
            TimeoutError,
        )),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=min_wait,
            max=max_wait
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


# ============================================================================
# ASYNC RETRY HELPERS
# ============================================================================

async def retry_async(
    func: Callable,
    *args: Any,
    max_attempts: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    min_wait: int = 1,
    max_wait: int = 10,
    **kwargs: Any
) -> Any:
    """
    Generic async retry helper.
    
    Args:
        func: Async function to retry
        args: Function args
        max_attempts: Max retry attempts
        exceptions: Tuple of exceptions to retry on
        min_wait: Min wait seconds
        max_wait: Max wait seconds
        kwargs: Function kwargs
    
    Returns:
        Function result
    
    Example:
        >>> result = await retry_async(
        ...     call_api,
        ...     endpoint="/search",
        ...     max_attempts=3,
        ...     exceptions=(ConnectionError, TimeoutError)
        ... )
    """
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(exceptions),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    ):
        with attempt:
            result = await func(*args, **kwargs)
            return result


# ============================================================================
# FALLBACK PATTERN
# ============================================================================

async def with_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    *args: Any,
    primary_exceptions: tuple[Type[Exception], ...] = (Exception,),
    **kwargs: Any
) -> Any:
    """
    Execute primary function with fallback on failure.
    
    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function if primary fails
        args: Function arguments
        primary_exceptions: Exceptions that trigger fallback
        kwargs: Function keyword arguments
    
    Returns:
        Result from primary or fallback
    
    Example:
        >>> result = await with_fallback(
        ...     llamaparse.parse,
        ...     vision_parser.parse,
        ...     file_path,
        ...     primary_exceptions=(ParseError, TimeoutError)
        ... )
    """
    try:
        logger.info("attempting_primary_function", func=primary_func.__name__)
        result = await primary_func(*args, **kwargs)
        logger.info("primary_function_succeeded", func=primary_func.__name__)
        return result
    except primary_exceptions as e:
        logger.warning(
            "primary_function_failed_using_fallback",
            primary_func=primary_func.__name__,
            fallback_func=fallback_func.__name__,
            error=str(e)
        )
        result = await fallback_func(*args, **kwargs)
        logger.info("fallback_function_succeeded", func=fallback_func.__name__)
        return result


# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================

class CircuitBreaker:
    """
    Simple circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Service unavailable, fail fast
    - HALF_OPEN: Testing if service recovered
    
    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        >>> 
        >>> async def call_service():
        ...     async with breaker:
        ...         return await external_api.call()
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        name: str = "circuit_breaker"
    ):
        """
        Args:
            failure_threshold: Failures before opening circuit
            timeout: Seconds before attempting recovery
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(f"{__name__}.{name}")
    
    async def __aenter__(self):
        """Enter circuit breaker context"""
        import time
        
        if self.state == "OPEN":
            # Check if timeout expired
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.timeout
            ):
                self.state = "HALF_OPEN"
                self.logger.info("circuit_breaker_half_open", name=self.name)
            else:
                self.logger.warning("circuit_breaker_open_failing_fast", name=self.name)
                raise ExternalServiceError(
                    service=self.name,
                    reason="Circuit breaker OPEN"
                )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context"""
        import time
        
        if exc_type is None:
            # Success
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
                self.logger.info("circuit_breaker_closed", name=self.name)
            return False
        
        # Failure
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.error(
                "circuit_breaker_opened",
                name=self.name,
                failures=self.failures,
                threshold=self.failure_threshold
            )
        
        return False  # Don't suppress exception


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Example 1: Retry decorator
    @retry_on_llm_error(max_attempts=3)
    async def flaky_llm_call():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise LLMTimeoutError("openai", "gpt-4", 30)
        return "Success!"
    
    # Example 2: Generic retry
    async def flaky_api_call():
        import random
        if random.random() < 0.5:
            raise ConnectionError("Network error")
        return {"data": "Success"}
    
    # Example 3: Fallback pattern
    async def primary_parser(text: str):
        if len(text) < 10:
            raise ValueError("Text too short")
        return f"Primary: {text}"
    
    async def fallback_parser(text: str):
        return f"Fallback: {text}"
    
    # Example 4: Circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, timeout=5, name="test_service")
    
    async def test_circuit_breaker():
        for i in range(10):
            try:
                async with breaker:
                    import random
                    if random.random() < 0.6:
                        raise ConnectionError("Service down")
                    print(f"Call {i}: Success")
            except Exception as e:
                print(f"Call {i}: Failed - {e}")
            await asyncio.sleep(0.5)
    
    # Run examples
    async def main():
        # Test retry
        try:
            result = await flaky_llm_call()
            print(f"LLM call result: {result}")
        except Exception as e:
            print(f"LLM call failed: {e}")
        
        # Test generic retry
        try:
            result = await retry_async(
                flaky_api_call,
                max_attempts=3,
                exceptions=(ConnectionError,)
            )
            print(f"API call result: {result}")
        except Exception as e:
            print(f"API call failed: {e}")
        
        # Test fallback
        result = await with_fallback(
            primary_parser,
            fallback_parser,
            "Hi",
            primary_exceptions=(ValueError,)
        )
        print(f"Parser result: {result}")
        
        # Test circuit breaker
        print("\nTesting circuit breaker:")
        await test_circuit_breaker()
    
    asyncio.run(main())
