"""
FastAPI application entry point.
Production-ready Legal Assistant API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from legal_assistant.core import (
    initialize_core,
    get_settings,
    get_logger,
)
from legal_assistant.api.middleware.error_handler import ErrorHandlerMiddleware
from legal_assistant.api.middleware.logging import LoggingMiddleware
from legal_assistant.api.middleware.rate_limit import RateLimitMiddleware
from legal_assistant.api.routes import auth, health

# Initialize core components
initialize_core()
settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "api_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment
    )
    
    # TODO: Initialize connections
    # - Redis connection pool
    # - Qdrant client
    # - Database connection
    
    yield
    
    # Shutdown
    logger.info("api_shutting_down")
    
    # TODO: Close connections
    # - Close Redis pool
    # - Close Qdrant client
    # - Close database


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Production-ready Legal Assistant API with multimodal RAG.
    
    ## Features
    * 🔐 JWT Authentication
    * 📄 Document Upload & Processing
    * 🔍 Semantic Search with RAG
    * 💬 Q&A with Source Citations
    * 📊 RAGAS Evaluation
    * 🎯 HITL for Low Confidence
    
    ## Tech Stack
    * FastAPI + Pydantic V2
    * LlamaIndex + LlamaParse
    * Qdrant Vector Database
    * Redis Rate Limiting
    * Prometheus Metrics
    * Structured Logging
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# ============================================================================
# MIDDLEWARE (Order matters!)
# ============================================================================

# 1. Error Handler (catches all exceptions)
app.add_middleware(ErrorHandlerMiddleware)

# 2. Logging (logs all requests)
app.add_middleware(LoggingMiddleware)

# 3. Rate Limiting (before auth to limit brute force)
# Note: Requires Redis connection
# app.add_middleware(RateLimitMiddleware, redis_client=redis_client)

# 4. CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit"],
)


# ============================================================================
# ROUTES
# ============================================================================

# Health & Monitoring
app.include_router(health.router)

# Authentication
app.include_router(auth.router)

# Documents (TODO: Uncomment when created)
# from legal_assistant.api.routes import documents
# app.include_router(documents.router)

# Query (TODO: Uncomment when created)
# from legal_assistant.api.routes import query
# app.include_router(query.router)

# Admin (TODO: Uncomment when created)
# from legal_assistant.api.routes import admin
# app.include_router(admin.router)


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to docs"""
    return RedirectResponse(url="/docs")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(
        "starting_uvicorn",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.reload
    )
    
    uvicorn.run(
        "legal_assistant.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_config=None,  # Use our custom logging
    )
