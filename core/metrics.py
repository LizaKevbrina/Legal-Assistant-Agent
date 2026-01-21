"""
Prometheus metrics for monitoring application performance.
Tracks latency, errors, token usage, and business metrics.
"""

import time
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Generator, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from legal_assistant.core.config import get_settings
from legal_assistant.core.logging import get_logger

logger = get_logger(__name__)

# Create custom registry (allows multiple apps in same process)
REGISTRY = CollectorRegistry()

# Application Info
app_info = Info(
    "legal_assistant_info",
    "Application information",
    registry=REGISTRY
)

settings = get_settings()
app_info.info({
    "app_name": settings.app_name,
    "version": settings.app_version,
    "environment": settings.environment,
})


# ============================================================================
# REQUEST METRICS
# ============================================================================

request_count = Counter(
    "legal_assistant_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
    registry=REGISTRY
)

request_duration = Histogram(
    "legal_assistant_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)

active_requests = Gauge(
    "legal_assistant_active_requests",
    "Number of active requests",
    ["endpoint"],
    registry=REGISTRY
)


# ============================================================================
# LLM METRICS
# ============================================================================

llm_request_count = Counter(
    "legal_assistant_llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"],
    registry=REGISTRY
)

llm_request_duration = Histogram(
    "legal_assistant_llm_request_duration_seconds",
    "LLM request duration",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    registry=REGISTRY
)

llm_tokens_used = Counter(
    "legal_assistant_llm_tokens_total",
    "Total tokens consumed",
    ["provider", "model", "token_type"],
    registry=REGISTRY
)

llm_cost_usd = Counter(
    "legal_assistant_llm_cost_usd_total",
    "Total LLM cost in USD",
    ["provider", "model"],
    registry=REGISTRY
)

llm_active_requests = Gauge(
    "legal_assistant_llm_active_requests",
    "Active LLM requests",
    ["provider"],
    registry=REGISTRY
)


# ============================================================================
# DOCUMENT PROCESSING METRICS
# ============================================================================

document_processed = Counter(
    "legal_assistant_documents_processed_total",
    "Documents processed",
    ["parser", "status"],
    registry=REGISTRY
)

document_processing_duration = Histogram(
    "legal_assistant_document_processing_seconds",
    "Document processing duration",
    ["parser"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=REGISTRY
)

document_pages = Summary(
    "legal_assistant_document_pages",
    "Number of pages per document",
    registry=REGISTRY
)

document_size_bytes = Summary(
    "legal_assistant_document_size_bytes",
    "Document size in bytes",
    registry=REGISTRY
)


# ============================================================================
# VECTOR STORE METRICS
# ============================================================================

vector_search_count = Counter(
    "legal_assistant_vector_searches_total",
    "Vector searches performed",
    ["collection", "status"],
    registry=REGISTRY
)

vector_search_duration = Histogram(
    "legal_assistant_vector_search_seconds",
    "Vector search duration",
    ["collection"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY
)

vector_results_count = Summary(
    "legal_assistant_vector_results",
    "Number of results returned",
    registry=REGISTRY
)

chunks_indexed = Counter(
    "legal_assistant_chunks_indexed_total",
    "Total chunks indexed",
    ["collection"],
    registry=REGISTRY
)


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

retrieval_count = Counter(
    "legal_assistant_retrievals_total",
    "Retrieval operations",
    ["stage", "status"],
    registry=REGISTRY
)

retrieval_duration = Histogram(
    "legal_assistant_retrieval_seconds",
    "Retrieval duration by stage",
    ["stage"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY
)

rerank_score = Summary(
    "legal_assistant_rerank_score",
    "Reranking scores",
    registry=REGISTRY
)


# ============================================================================
# QUALITY METRICS
# ============================================================================

query_confidence = Summary(
    "legal_assistant_query_confidence",
    "Query response confidence scores",
    registry=REGISTRY
)

hitl_triggered = Counter(
    "legal_assistant_hitl_triggered_total",
    "HITL triggers",
    ["reason"],
    registry=REGISTRY
)

ragas_scores = Gauge(
    "legal_assistant_ragas_score",
    "RAGAS evaluation scores",
    ["metric"],
    registry=REGISTRY
)


# ============================================================================
# ERROR METRICS
# ============================================================================

errors_total = Counter(
    "legal_assistant_errors_total",
    "Total errors",
    ["component", "error_type"],
    registry=REGISTRY
)

pii_detected = Counter(
    "legal_assistant_pii_detected_total",
    "PII detections",
    ["entity_type"],
    registry=REGISTRY
)


# ============================================================================
# CACHE METRICS
# ============================================================================

cache_hits = Counter(
    "legal_assistant_cache_hits_total",
    "Cache hits",
    ["cache_type"],
    registry=REGISTRY
)

cache_misses = Counter(
    "legal_assistant_cache_misses_total",
    "Cache misses",
    ["cache_type"],
    registry=REGISTRY
)


# ============================================================================
# BUSINESS METRICS
# ============================================================================

queries_per_document_type = Counter(
    "legal_assistant_queries_by_doc_type_total",
    "Queries by document type",
    ["doc_type"],
    registry=REGISTRY
)

user_sessions = Gauge(
    "legal_assistant_active_user_sessions",
    "Active user sessions",
    registry=REGISTRY
)


# ============================================================================
# METRIC HELPERS
# ============================================================================

class MetricStatus(str, Enum):
    """Standard status codes for metrics"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"


@contextmanager
def track_time(
    histogram: Histogram,
    labels: Optional[dict[str, str]] = None
) -> Generator[None, None, None]:
    """
    Context manager to track operation duration.
    
    Example:
        >>> with track_time(llm_request_duration, {"provider": "openai"}):
        ...     result = call_llm()
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        if labels:
            histogram.labels(**labels).observe(duration)
        else:
            histogram.observe(duration)


@contextmanager
def track_active(
    gauge: Gauge,
    labels: Optional[dict[str, str]] = None
) -> Generator[None, None, None]:
    """
    Context manager to track active operations.
    
    Example:
        >>> with track_active(active_requests, {"endpoint": "/query"}):
        ...     process_request()
    """
    if labels:
        gauge.labels(**labels).inc()
    else:
        gauge.inc()
    try:
        yield
    finally:
        if labels:
            gauge.labels(**labels).dec()
        else:
            gauge.dec()


def track_llm_tokens(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float
) -> None:
    """
    Track LLM token usage and cost.
    
    Args:
        provider: LLM provider (openai, anthropic)
        model: Model name
        prompt_tokens: Input tokens
        completion_tokens: Output tokens
        cost_usd: Cost in USD
    """
    llm_tokens_used.labels(
        provider=provider,
        model=model,
        token_type="prompt"
    ).inc(prompt_tokens)
    
    llm_tokens_used.labels(
        provider=provider,
        model=model,
        token_type="completion"
    ).inc(completion_tokens)
    
    llm_cost_usd.labels(
        provider=provider,
        model=model
    ).inc(cost_usd)
    
    logger.info(
        "llm_tokens_tracked",
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost_usd
    )


def track_error(component: str, error: Exception) -> None:
    """
    Track error occurrence.
    
    Args:
        component: Component where error occurred
        error: Exception instance
    """
    error_type = error.__class__.__name__
    errors_total.labels(
        component=component,
        error_type=error_type
    ).inc()
    
    logger.error(
        "error_tracked",
        component=component,
        error_type=error_type,
        error_message=str(error)
    )


def metrics_middleware(func: Callable) -> Callable:
    """
    Decorator to automatically track function metrics.
    
    Example:
        >>> @metrics_middleware
        ... async def process_query(query: str):
        ...     return result
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        
        with track_active(active_requests, {"endpoint": func_name}):
            start = time.time()
            status = MetricStatus.SUCCESS
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = MetricStatus.ERROR
                track_error(func_name, e)
                raise
            finally:
                duration = time.time() - start
                request_duration.labels(
                    endpoint=func_name,
                    method="async"
                ).observe(duration)
                
                request_count.labels(
                    endpoint=func_name,
                    method="async",
                    status=status.value
                ).inc()
    
    return wrapper


def export_metrics() -> tuple[bytes, str]:
    """
    Export metrics in Prometheus format.
    
    Returns:
        Tuple of (metrics_data, content_type)
    
    Example:
        >>> data, content_type = export_metrics()
        >>> # Send as HTTP response
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Simulate some metrics
    
    # Request metrics
    with track_time(request_duration, {"endpoint": "/query", "method": "POST"}):
        time.sleep(0.5)
    request_count.labels(endpoint="/query", method="POST", status="success").inc()
    
    # LLM metrics
    track_llm_tokens(
        provider="openai",
        model="gpt-4-turbo",
        prompt_tokens=500,
        completion_tokens=200,
        cost_usd=0.015
    )
    
    # Document metrics
    document_processed.labels(parser="llamaparse", status="success").inc()
    document_pages.observe(15)
    document_size_bytes.observe(2_500_000)
    
    # Vector metrics
    vector_search_count.labels(collection="legal_documents", status="success").inc()
    vector_results_count.observe(20)
    chunks_indexed.labels(collection="legal_documents").inc(45)
    
    # Quality metrics
    query_confidence.observe(0.85)
    ragas_scores.labels(metric="faithfulness").set(0.92)
    
    # Error metrics
    try:
        raise ValueError("Test error")
    except Exception as e:
        track_error("test_component", e)
    
    # Export metrics
    metrics_data, content_type = export_metrics()
    print(metrics_data.decode())
