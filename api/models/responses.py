"""
Response models for API endpoints.
Structured responses with consistent format.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict

from legal_assistant.api.models.requests import (
    DocumentType,
    Jurisdiction,
    LegalArea,
)


# ============================================================================
# BASE RESPONSE MODELS
# ============================================================================

class BaseResponse(BaseModel):
    """Base response model"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "request_id": "abc123def456"
            }
        }
    )
    
    success: bool = Field(
        ...,
        description="Operation success status"
    )
    message: Optional[str] = Field(
        None,
        description="Human-readable message"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request trace ID"
    )


class ErrorResponse(BaseResponse):
    """Error response model"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "message": "File validation failed",
                "request_id": "abc123",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "details": {"filename": "doc.pdf", "reason": "File too large"},
                    "recoverable": False
                }
            }
        }
    )
    
    success: bool = Field(default=False)
    error: dict[str, Any] = Field(
        ...,
        description="Error details"
    )


# ============================================================================
# AUTHENTICATION RESPONSES
# ============================================================================

class TokenResponse(BaseResponse):
    """JWT token response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "access_token": "eyJhbGciOiJIUzI1NiIs...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }
    )
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class UserInfoResponse(BaseResponse):
    """User information response"""
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    created_at: datetime = Field(..., description="Account creation date")
    is_active: bool = Field(default=True, description="Account active status")


# ============================================================================
# DOCUMENT RESPONSES
# ============================================================================

class DocumentMetadata(BaseModel):
    """Document metadata"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_abc123",
                "title": "Договор аренды помещения",
                "doc_type": "contract",
                "filename": "contract_2025.pdf",
                "file_size_mb": 2.5,
                "pages": 15,
                "jurisdiction": "moscow",
                "legal_area": "civil",
                "date": "2025-01-15T00:00:00Z",
                "parties": ["ООО Арендодатель", "ООО Арендатор"],
                "tags": ["аренда", "коммерческая"],
                "uploaded_at": "2025-01-15T10:30:00Z",
                "processed": True
            }
        }
    )
    
    document_id: str = Field(..., description="Document ID")
    title: Optional[str] = Field(None, description="Document title")
    doc_type: DocumentType = Field(..., description="Document type")
    filename: str = Field(..., description="Original filename")
    file_size_mb: float = Field(..., description="File size in MB")
    pages: Optional[int] = Field(None, description="Number of pages")
    
    # Legal metadata
    jurisdiction: Optional[Jurisdiction] = None
    legal_area: Optional[LegalArea] = None
    date: Optional[datetime] = None
    parties: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    notes: Optional[str] = None
    
    # System metadata
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    uploaded_by: str = Field(..., description="User ID")
    processed: bool = Field(default=False, description="Processing status")
    processing_error: Optional[str] = Field(
        None,
        description="Processing error message"
    )
    chunks_count: Optional[int] = Field(
        None,
        description="Number of indexed chunks"
    )


class DocumentUploadResponse(BaseResponse):
    """Document upload response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Document uploaded successfully",
                "document_id": "doc_abc123",
                "metadata": {
                    "document_id": "doc_abc123",
                    "filename": "contract.pdf",
                    "file_size_mb": 2.5,
                    "processed": False
                }
            }
        }
    )
    
    document_id: str = Field(..., description="Document ID")
    metadata: DocumentMetadata = Field(..., description="Document metadata")


class DocumentListResponse(BaseResponse):
    """List of documents response"""
    documents: list[DocumentMetadata] = Field(
        ...,
        description="List of documents"
    )
    total: int = Field(..., description="Total documents count")
    limit: int = Field(..., description="Results per page")
    offset: int = Field(..., description="Pagination offset")


class DocumentDeleteResponse(BaseResponse):
    """Document deletion response"""
    document_id: str = Field(..., description="Deleted document ID")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")


# ============================================================================
# QUERY RESPONSES
# ============================================================================

class SourceDocument(BaseModel):
    """Source document reference"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_abc123",
                "title": "Договор аренды",
                "chunk_id": "chunk_001",
                "text": "Арендная плата составляет 100000 рублей...",
                "page": 5,
                "score": 0.92,
                "metadata": {"doc_type": "contract", "date": "2025-01-15"}
            }
        }
    )
    
    document_id: str = Field(..., description="Document ID")
    title: Optional[str] = Field(None, description="Document title")
    chunk_id: str = Field(..., description="Chunk ID")
    text: str = Field(..., description="Relevant text excerpt")
    page: Optional[int] = Field(None, description="Page number")
    score: float = Field(..., description="Relevance score (0-1)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class QueryResponse(BaseResponse):
    """Query answer response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "query_id": "query_abc123",
                "query": "Какая арендная плата?",
                "answer": "Согласно договору, арендная плата составляет 100000 рублей в месяц.",
                "confidence": 0.89,
                "sources": [
                    {
                        "document_id": "doc_123",
                        "title": "Договор аренды",
                        "text": "Арендная плата...",
                        "score": 0.92
                    }
                ],
                "processing_time_ms": 1234,
                "tokens_used": 750
            }
        }
    )
    
    query_id: str = Field(..., description="Query ID for tracking")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(
        ...,
        description="Confidence score (0-1)",
        ge=0.0,
        le=1.0
    )
    
    # Sources
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents"
    )
    sources_count: int = Field(
        default=0,
        description="Number of sources used"
    )
    
    # Metadata
    processing_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds"
    )
    tokens_used: int = Field(..., description="Total tokens used")
    model_used: str = Field(..., description="LLM model used")
    
    # Warnings
    warnings: Optional[list[str]] = Field(
        None,
        description="Warning messages"
    )
    requires_review: bool = Field(
        default=False,
        description="Requires human review (HITL)"
    )


# ============================================================================
# FEEDBACK RESPONSES
# ============================================================================

class FeedbackResponse(BaseResponse):
    """Feedback submission response"""
    feedback_id: str = Field(..., description="Feedback ID")
    query_id: str = Field(..., description="Associated query ID")
    recorded_at: datetime = Field(..., description="Timestamp")


# ============================================================================
# EVALUATION RESPONSES
# ============================================================================

class RAGASMetrics(BaseModel):
    """RAGAS evaluation metrics"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "faithfulness": 0.92,
                "answer_relevancy": 0.88,
                "context_precision": 0.85,
                "context_recall": 0.90
            }
        }
    )
    
    faithfulness: Optional[float] = Field(
        None,
        description="Faithfulness score (0-1)",
        ge=0.0,
        le=1.0
    )
    answer_relevancy: Optional[float] = Field(
        None,
        description="Answer relevancy score (0-1)",
        ge=0.0,
        le=1.0
    )
    context_precision: Optional[float] = Field(
        None,
        description="Context precision score (0-1)",
        ge=0.0,
        le=1.0
    )
    context_recall: Optional[float] = Field(
        None,
        description="Context recall score (0-1)",
        ge=0.0,
        le=1.0
    )


class EvaluationResponse(BaseResponse):
    """Evaluation response"""
    evaluation_id: str = Field(..., description="Evaluation ID")
    metrics: RAGASMetrics = Field(..., description="RAGAS metrics")
    overall_score: float = Field(
        ...,
        description="Overall score (0-1)",
        ge=0.0,
        le=1.0
    )


# ============================================================================
# HEALTH & STATUS RESPONSES
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "services": {
                    "database": "healthy",
                    "vector_store": "healthy",
                    "cache": "healthy",
                    "llm": "healthy"
                },
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }
    )
    
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    services: dict[str, str] = Field(
        ...,
        description="Service health statuses"
    )
    timestamp: datetime = Field(..., description="Check timestamp")


class StatsResponse(BaseResponse):
    """Statistics response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "documents_total": 1523,
                "chunks_total": 45690,
                "queries_total": 3421,
                "queries_today": 127,
                "avg_confidence": 0.87,
                "storage_used_mb": 2048.5
            }
        }
    )
    
    documents_total: int = Field(..., description="Total documents")
    chunks_total: int = Field(..., description="Total indexed chunks")
    queries_total: int = Field(..., description="Total queries processed")
    queries_today: int = Field(..., description="Queries today")
    avg_confidence: float = Field(
        ...,
        description="Average confidence score"
    )
    storage_used_mb: float = Field(..., description="Storage used in MB")


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

class BatchOperationResponse(BaseResponse):
    """Batch operation response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "batch_id": "batch_abc123",
                "total": 10,
                "succeeded": 8,
                "failed": 2,
                "results": [
                    {"id": "1", "status": "success"},
                    {"id": "2", "status": "failed", "error": "Parse error"}
                ]
            }
        }
    )
    
    batch_id: str = Field(..., description="Batch operation ID")
    total: int = Field(..., description="Total items")
    succeeded: int = Field(..., description="Successful items")
    failed: int = Field(..., description="Failed items")
    results: list[dict[str, Any]] = Field(
        ...,
        description="Individual results"
    )
