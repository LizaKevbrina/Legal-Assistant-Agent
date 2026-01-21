"""
Health check and monitoring endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from legal_assistant.core import get_settings, get_logger
from legal_assistant.core.metrics import export_metrics
from legal_assistant.api.models.responses import HealthResponse

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["Health"])


async def check_qdrant() -> str:
    """Check Qdrant health"""
    try:
        # TODO: Implement actual Qdrant health check
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_redis() -> str:
    """Check Redis health"""
    try:
        # TODO: Implement actual Redis health check
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_llm() -> str:
    """Check LLM availability"""
    try:
        # TODO: Implement actual LLM health check
        return "healthy"
    except Exception:
        return "unhealthy"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    System health check.
    Returns health status of all services.
    
    Returns:
        Health status
    """
    # Check all services
    services = {
        "api": "healthy",
        "vector_store": await check_qdrant(),
        "cache": await check_redis(),
        "llm": await check_llm(),
    }
    
    # Overall status
    status_value = "healthy" if all(
        s == "healthy" for s in services.values()
    ) else "unhealthy"
    
    logger.info("health_check", status=status_value, services=services)
    
    return HealthResponse(
        status=status_value,
        version=settings.app_version,
        services=services,
        timestamp=datetime.utcnow()
    )


@router.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus format.
    
    Returns:
        Prometheus metrics
    """
    data, content_type = export_metrics()
    return PlainTextResponse(
        content=data.decode('utf-8'),
        media_type=content_type
    )


@router.get("/")
async def root():
    """
    Root endpoint.
    Returns basic API information.
    """
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }
