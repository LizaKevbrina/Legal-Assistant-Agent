"""
Custom exceptions with structured error handling.
Provides clear error types, HTTP status codes, and recovery strategies.
"""

from typing import Any, Optional


class LegalAssistantException(Exception):
    """
    Base exception for all application errors.
    
    Attributes:
        message: Human-readable error message
        details: Additional context (dict)
        status_code: HTTP status code
        error_code: Machine-readable error code
        recoverable: Whether error can be retried
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        recoverable: bool = False
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.error_code = error_code
        self.recoverable = recoverable
        super().__init__(self.message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": {
                "message": self.message,
                "code": self.error_code,
                "details": self.details,
                "recoverable": self.recoverable
            }
        }


# ============================================================================
# INPUT VALIDATION ERRORS (400)
# ============================================================================

class ValidationError(LegalAssistantException):
    """Input validation failed"""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=400,
            error_code="VALIDATION_ERROR",
            recoverable=False
        )


class FileValidationError(ValidationError):
    """File validation failed (size, type, content)"""
    
    def __init__(self, message: str, filename: str, **kwargs):
        super().__init__(
            message=message,
            details={"filename": filename, **kwargs}
        )


class QueryValidationError(ValidationError):
    """Query validation failed (length, content)"""
    
    def __init__(self, message: str, query: str, **kwargs):
        super().__init__(
            message=message,
            details={"query": query[:100], **kwargs}
        )


class PIIDetectedError(ValidationError):
    """PII detected in input"""
    
    def __init__(self, entities: list[str]):
        super().__init__(
            message="PII detected in input. Please remove sensitive information.",
            details={"detected_entities": entities}
        )


# ============================================================================
# AUTHENTICATION/AUTHORIZATION ERRORS (401/403)
# ============================================================================

class AuthenticationError(LegalAssistantException):
    """Authentication failed"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            recoverable=False
        )


class AuthorizationError(LegalAssistantException):
    """Authorization failed"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            recoverable=False
        )


# ============================================================================
# RATE LIMITING ERRORS (429)
# ============================================================================

class RateLimitError(LegalAssistantException):
    """Rate limit exceeded"""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: int
    ):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after_seconds": retry_after
            },
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            recoverable=True
        )


class TokenBudgetExceeded(LegalAssistantException):
    """Daily token budget exceeded"""
    
    def __init__(self, used: int, budget: int):
        super().__init__(
            message=f"Daily token budget exceeded: {used}/{budget}",
            details={"used_tokens": used, "budget": budget},
            status_code=429,
            error_code="TOKEN_BUDGET_EXCEEDED",
            recoverable=False
        )


# ============================================================================
# DOCUMENT PROCESSING ERRORS (500/503)
# ============================================================================

class DocumentProcessingError(LegalAssistantException):
    """Document processing failed"""
    
    def __init__(
        self,
        message: str,
        filename: str,
        parser: str,
        recoverable: bool = True
    ):
        super().__init__(
            message=message,
            details={"filename": filename, "parser": parser},
            status_code=503 if recoverable else 500,
            error_code="DOCUMENT_PROCESSING_ERROR",
            recoverable=recoverable
        )


class ParseError(DocumentProcessingError):
    """Document parsing failed"""
    
    def __init__(self, filename: str, parser: str, reason: str):
        super().__init__(
            message=f"Failed to parse document: {reason}",
            filename=filename,
            parser=parser,
            recoverable=True
        )


class ChunkingError(DocumentProcessingError):
    """Document chunking failed"""
    
    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"Failed to chunk document: {reason}",
            filename=filename,
            parser="chunker",
            recoverable=False
        )


class OCRError(DocumentProcessingError):
    """OCR processing failed"""
    
    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"OCR failed: {reason}",
            filename=filename,
            parser="tesseract",
            recoverable=True
        )


# ============================================================================
# LLM ERRORS (500/503)
# ============================================================================

class LLMError(LegalAssistantException):
    """LLM request failed"""
    
    def __init__(
        self,
        message: str,
        provider: str,
        model: str,
        recoverable: bool = True
    ):
        super().__init__(
            message=message,
            details={"provider": provider, "model": model},
            status_code=503 if recoverable else 500,
            error_code="LLM_ERROR",
            recoverable=recoverable
        )


class LLMTimeoutError(LLMError):
    """LLM request timed out"""
    
    def __init__(self, provider: str, model: str, timeout_seconds: int):
        super().__init__(
            message=f"LLM request timed out after {timeout_seconds}s",
            provider=provider,
            model=model,
            recoverable=True
        )


class LLMContextLengthError(LLMError):
    """Context length exceeded"""
    
    def __init__(self, provider: str, model: str, tokens: int, limit: int):
        super().__init__(
            message=f"Context length exceeded: {tokens} > {limit}",
            provider=provider,
            model=model,
            recoverable=False
        )


class LLMContentFilterError(LLMError):
    """Content filtered by LLM provider"""
    
    def __init__(self, provider: str, model: str, reason: str):
        super().__init__(
            message=f"Content filtered: {reason}",
            provider=provider,
            model=model,
            recoverable=False
        )


# ============================================================================
# VECTOR STORE ERRORS (500/503)
# ============================================================================

class VectorStoreError(LegalAssistantException):
    """Vector store operation failed"""
    
    def __init__(
        self,
        message: str,
        collection: str,
        operation: str,
        recoverable: bool = True
    ):
        super().__init__(
            message=message,
            details={"collection": collection, "operation": operation},
            status_code=503 if recoverable else 500,
            error_code="VECTOR_STORE_ERROR",
            recoverable=recoverable
        )


class VectorSearchError(VectorStoreError):
    """Vector search failed"""
    
    def __init__(self, collection: str, reason: str):
        super().__init__(
            message=f"Vector search failed: {reason}",
            collection=collection,
            operation="search",
            recoverable=True
        )


class VectorIndexError(VectorStoreError):
    """Vector indexing failed"""
    
    def __init__(self, collection: str, reason: str):
        super().__init__(
            message=f"Vector indexing failed: {reason}",
            collection=collection,
            operation="index",
            recoverable=True
        )


# ============================================================================
# RETRIEVAL ERRORS (500/503)
# ============================================================================

class RetrievalError(LegalAssistantException):
    """Retrieval pipeline failed"""
    
    def __init__(
        self,
        message: str,
        stage: str,
        recoverable: bool = True
    ):
        super().__init__(
            message=message,
            details={"stage": stage},
            status_code=503 if recoverable else 500,
            error_code="RETRIEVAL_ERROR",
            recoverable=recoverable
        )


class EmbeddingError(RetrievalError):
    """Embedding generation failed"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Embedding generation failed: {reason}",
            stage="embedding",
            recoverable=True
        )


class RerankError(RetrievalError):
    """Reranking failed"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Reranking failed: {reason}",
            stage="rerank",
            recoverable=True
        )


class NoResultsError(RetrievalError):
    """No results found"""
    
    def __init__(self, query: str):
        super().__init__(
            message="No relevant documents found for query",
            stage="search",
            recoverable=False
        )
        self.details["query"] = query[:100]


# ============================================================================
# EXTERNAL SERVICE ERRORS (502/503)
# ============================================================================

class ExternalServiceError(LegalAssistantException):
    """External service unavailable"""
    
    def __init__(self, service: str, reason: str):
        super().__init__(
            message=f"{service} unavailable: {reason}",
            details={"service": service},
            status_code=503,
            error_code="EXTERNAL_SERVICE_ERROR",
            recoverable=True
        )


class QdrantUnavailable(ExternalServiceError):
    """Qdrant service unavailable"""
    
    def __init__(self, reason: str):
        super().__init__(service="Qdrant", reason=reason)


class RedisUnavailable(ExternalServiceError):
    """Redis service unavailable"""
    
    def __init__(self, reason: str):
        super().__init__(service="Redis", reason=reason)


# ============================================================================
# CONFIGURATION ERRORS (500)
# ============================================================================

class ConfigurationError(LegalAssistantException):
    """Configuration error"""
    
    def __init__(self, message: str, config_key: str):
        super().__init__(
            message=message,
            details={"config_key": config_key},
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            recoverable=False
        )


# ============================================================================
# QUALITY/CONFIDENCE ERRORS (422)
# ============================================================================

class LowConfidenceError(LegalAssistantException):
    """Response confidence too low"""
    
    def __init__(self, confidence: float, threshold: float):
        super().__init__(
            message=f"Response confidence too low: {confidence:.2f} < {threshold:.2f}",
            details={
                "confidence": confidence,
                "threshold": threshold,
                "requires_hitl": True
            },
            status_code=422,
            error_code="LOW_CONFIDENCE",
            recoverable=False
        )


class HallucinationDetected(LegalAssistantException):
    """Potential hallucination detected"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Potential hallucination: {reason}",
            details={"requires_hitl": True},
            status_code=422,
            error_code="HALLUCINATION_DETECTED",
            recoverable=False
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Raise and catch exceptions
    
    try:
        raise FileValidationError(
            message="File too large",
            filename="contract.pdf",
            size_mb=75,
            max_size_mb=50
        )
    except FileValidationError as e:
        print(f"Status: {e.status_code}")
        print(f"Code: {e.error_code}")
        print(f"Recoverable: {e.recoverable}")
        print(f"Dict: {e.to_dict()}")
    
    try:
        raise LLMTimeoutError(
            provider="openai",
            model="gpt-4-turbo",
            timeout_seconds=30
        )
    except LLMTimeoutError as e:
        print(f"\nLLM Error: {e.message}")
        print(f"Recoverable: {e.recoverable}")
    
    try:
        raise LowConfidenceError(confidence=0.65, threshold=0.7)
    except LowConfidenceError as e:
        print(f"\nLow Confidence: {e.message}")
        print(f"HITL Required: {e.details['requires_hitl']}")
