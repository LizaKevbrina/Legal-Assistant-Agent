"""
Tests for custom exceptions.
Tests exception hierarchy, error codes, and structured responses.
"""

import pytest

from legal_assistant.core.exceptions import (
    LegalAssistantException,
    ValidationError,
    FileValidationError,
    QueryValidationError,
    PIIDetectedError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    TokenBudgetExceeded,
    DocumentProcessingError,
    ParseError,
    ChunkingError,
    OCRError,
    LLMError,
    LLMTimeoutError,
    LLMContextLengthError,
    LLMContentFilterError,
    VectorStoreError,
    VectorSearchError,
    VectorIndexError,
    RetrievalError,
    EmbeddingError,
    RerankError,
    NoResultsError,
    ExternalServiceError,
    QdrantUnavailable,
    RedisUnavailable,
    ConfigurationError,
    LowConfidenceError,
    HallucinationDetected,
)


class TestBaseException:
    """Test base LegalAssistantException"""
    
    def test_basic_exception(self):
        """Test creating basic exception"""
        exc = LegalAssistantException(
            message="Test error",
            status_code=500,
            error_code="TEST_ERROR"
        )
        
        assert str(exc) == "Test error"
        assert exc.status_code == 500
        assert exc.error_code == "TEST_ERROR"
        assert exc.recoverable is False
        assert exc.details == {}
    
    def test_exception_with_details(self):
        """Test exception with details"""
        details = {"key": "value", "count": 42}
        exc = LegalAssistantException(
            message="Error with details",
            details=details
        )
        
        assert exc.details == details
    
    def test_exception_to_dict(self):
        """Test converting exception to dict"""
        exc = LegalAssistantException(
            message="Test error",
            details={"info": "data"},
            status_code=400,
            error_code="TEST",
            recoverable=True
        )
        
        result = exc.to_dict()
        
        assert "error" in result
        assert result["error"]["message"] == "Test error"
        assert result["error"]["code"] == "TEST"
        assert result["error"]["details"] == {"info": "data"}
        assert result["error"]["recoverable"] is True
    
    def test_exception_inheritance(self):
        """Test that custom exception inherits from Exception"""
        exc = LegalAssistantException("test")
        
        assert isinstance(exc, Exception)
        assert isinstance(exc, LegalAssistantException)


class TestValidationErrors:
    """Test validation exception types"""
    
    def test_file_validation_error(self):
        """Test FileValidationError"""
        exc = FileValidationError(
            message="File too large",
            filename="document.pdf",
            size_mb=75,
            max_size_mb=50
        )
        
        assert exc.status_code == 400
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.details["filename"] == "document.pdf"
        assert exc.details["size_mb"] == 75
        assert exc.details["max_size_mb"] == 50
        assert exc.recoverable is False
    
    def test_query_validation_error(self):
        """Test QueryValidationError"""
        long_query = "a" * 1500
        exc = QueryValidationError(
            message="Query too long",
            query=long_query,
            length=1500,
            max_length=1000
        )
        
        assert exc.status_code == 400
        # Query should be truncated in details
        assert len(exc.details["query"]) == 100
        assert exc.details["length"] == 1500
    
    def test_pii_detected_error(self):
        """Test PIIDetectedError"""
        entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"]
        exc = PIIDetectedError(entities=entities)
        
        assert exc.status_code == 400
        assert "PII detected" in exc.message
        assert exc.details["detected_entities"] == entities


class TestAuthErrors:
    """Test authentication/authorization errors"""
    
    def test_authentication_error_default(self):
        """Test default AuthenticationError"""
        exc = AuthenticationError()
        
        assert exc.status_code == 401
        assert exc.error_code == "AUTHENTICATION_ERROR"
        assert "Authentication failed" in exc.message
        assert exc.recoverable is False
    
    def test_authentication_error_custom(self):
        """Test custom AuthenticationError message"""
        exc = AuthenticationError(message="Invalid token")
        
        assert "Invalid token" in exc.message
        assert exc.status_code == 401
    
    def test_authorization_error(self):
        """Test AuthorizationError"""
        exc = AuthorizationError(message="Access denied")
        
        assert exc.status_code == 403
        assert exc.error_code == "AUTHORIZATION_ERROR"
        assert "Access denied" in exc.message


class TestRateLimitErrors:
    """Test rate limiting errors"""
    
    def test_rate_limit_error(self):
        """Test RateLimitError"""
        exc = RateLimitError(
            limit=10,
            window_seconds=60,
            retry_after=45
        )
        
        assert exc.status_code == 429
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
        assert exc.recoverable is True
        assert exc.details["limit"] == 10
        assert exc.details["window_seconds"] == 60
        assert exc.details["retry_after_seconds"] == 45
        assert "10 requests per 60s" in exc.message
    
    def test_token_budget_exceeded(self):
        """Test TokenBudgetExceeded"""
        exc = TokenBudgetExceeded(used=1_200_000, budget=1_000_000)
        
        assert exc.status_code == 429
        assert exc.error_code == "TOKEN_BUDGET_EXCEEDED"
        assert exc.recoverable is False  # Can't retry until next day
        assert exc.details["used_tokens"] == 1_200_000
        assert exc.details["budget"] == 1_000_000


class TestDocumentProcessingErrors:
    """Test document processing errors"""
    
    def test_parse_error_recoverable(self):
        """Test ParseError as recoverable"""
        exc = ParseError(
            filename="contract.pdf",
            parser="llamaparse",
            reason="Timeout"
        )
        
        assert exc.status_code == 503  # Recoverable
        assert exc.error_code == "DOCUMENT_PROCESSING_ERROR"
        assert exc.recoverable is True
        assert exc.details["filename"] == "contract.pdf"
        assert exc.details["parser"] == "llamaparse"
        assert "Timeout" in exc.message
    
    def test_chunking_error_not_recoverable(self):
        """Test ChunkingError as non-recoverable"""
        exc = ChunkingError(
            filename="doc.pdf",
            reason="Invalid text structure"
        )
        
        assert exc.status_code == 500  # Not recoverable
        assert exc.recoverable is False
    
    def test_ocr_error(self):
        """Test OCRError"""
        exc = OCRError(
            filename="scan.pdf",
            reason="Low quality image"
        )
        
        assert exc.details["parser"] == "tesseract"
        assert exc.recoverable is True


class TestLLMErrors:
    """Test LLM-related errors"""
    
    def test_llm_timeout_error(self):
        """Test LLMTimeoutError"""
        exc = LLMTimeoutError(
            provider="openai",
            model="gpt-4-turbo",
            timeout_seconds=30
        )
        
        assert exc.status_code == 503
        assert exc.error_code == "LLM_ERROR"
        assert exc.recoverable is True
        assert exc.details["provider"] == "openai"
        assert exc.details["model"] == "gpt-4-turbo"
        assert "30s" in exc.message
    
    def test_llm_context_length_error(self):
        """Test LLMContextLengthError"""
        exc = LLMContextLengthError(
            provider="openai",
            model="gpt-4",
            tokens=150000,
            limit=128000
        )
        
        assert exc.status_code == 500
        assert exc.recoverable is False  # Can't retry, need to reduce tokens
        assert "150000 > 128000" in exc.message
    
    def test_llm_content_filter_error(self):
        """Test LLMContentFilterError"""
        exc = LLMContentFilterError(
            provider="openai",
            model="gpt-4",
            reason="Violence detected"
        )
        
        assert exc.recoverable is False
        assert "Violence detected" in exc.message


class TestVectorStoreErrors:
    """Test vector store errors"""
    
    def test_vector_search_error(self):
        """Test VectorSearchError"""
        exc = VectorSearchError(
            collection="legal_documents",
            reason="Connection timeout"
        )
        
        assert exc.status_code == 503
        assert exc.error_code == "VECTOR_STORE_ERROR"
        assert exc.recoverable is True
        assert exc.details["collection"] == "legal_documents"
        assert exc.details["operation"] == "search"
    
    def test_vector_index_error(self):
        """Test VectorIndexError"""
        exc = VectorIndexError(
            collection="legal_documents",
            reason="Invalid vector dimension"
        )
        
        assert exc.details["operation"] == "index"
        assert exc.recoverable is True


class TestRetrievalErrors:
    """Test retrieval pipeline errors"""
    
    def test_embedding_error(self):
        """Test EmbeddingError"""
        exc = EmbeddingError(reason="API rate limit")
        
        assert exc.status_code == 503
        assert exc.error_code == "RETRIEVAL_ERROR"
        assert exc.recoverable is True
        assert exc.details["stage"] == "embedding"
    
    def test_rerank_error(self):
        """Test RerankError"""
        exc = RerankError(reason="Service unavailable")
        
        assert exc.details["stage"] == "rerank"
        assert exc.recoverable is True
    
    def test_no_results_error(self):
        """Test NoResultsError"""
        query = "найти договор аренды"
        exc = NoResultsError(query=query)
        
        assert exc.status_code == 503
        assert exc.recoverable is False
        assert exc.details["stage"] == "search"
        assert query in exc.details["query"]


class TestExternalServiceErrors:
    """Test external service errors"""
    
    def test_qdrant_unavailable(self):
        """Test QdrantUnavailable"""
        exc = QdrantUnavailable(reason="Connection refused")
        
        assert exc.status_code == 503
        assert exc.error_code == "EXTERNAL_SERVICE_ERROR"
        assert exc.recoverable is True
        assert exc.details["service"] == "Qdrant"
        assert "Connection refused" in exc.message
    
    def test_redis_unavailable(self):
        """Test RedisUnavailable"""
        exc = RedisUnavailable(reason="Timeout")
        
        assert exc.details["service"] == "Redis"
        assert exc.recoverable is True


class TestQualityErrors:
    """Test quality/confidence errors"""
    
    def test_low_confidence_error(self):
        """Test LowConfidenceError"""
        exc = LowConfidenceError(confidence=0.65, threshold=0.7)
        
        assert exc.status_code == 422
        assert exc.error_code == "LOW_CONFIDENCE"
        assert exc.recoverable is False
        assert exc.details["confidence"] == 0.65
        assert exc.details["threshold"] == 0.7
        assert exc.details["requires_hitl"] is True
        assert "0.65 < 0.70" in exc.message
    
    def test_hallucination_detected(self):
        """Test HallucinationDetected"""
        exc = HallucinationDetected(reason="Contradicts source")
        
        assert exc.status_code == 422
        assert exc.error_code == "HALLUCINATION_DETECTED"
        assert exc.details["requires_hitl"] is True
        assert "Contradicts source" in exc.message


class TestConfigurationError:
    """Test configuration errors"""
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        exc = ConfigurationError(
            message="Missing API key",
            config_key="OPENAI_API_KEY"
        )
        
        assert exc.status_code == 500
        assert exc.error_code == "CONFIGURATION_ERROR"
        assert exc.recoverable is False
        assert exc.details["config_key"] == "OPENAI_API_KEY"


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy"""
    
    def test_validation_error_hierarchy(self):
        """Test ValidationError hierarchy"""
        exc = FileValidationError(
            message="test",
            filename="test.pdf"
        )
        
        assert isinstance(exc, FileValidationError)
        assert isinstance(exc, ValidationError)
        assert isinstance(exc, LegalAssistantException)
        assert isinstance(exc, Exception)
    
    def test_llm_error_hierarchy(self):
        """Test LLMError hierarchy"""
        exc = LLMTimeoutError(
            provider="openai",
            model="gpt-4",
            timeout_seconds=30
        )
        
        assert isinstance(exc, LLMTimeoutError)
        assert isinstance(exc, LLMError)
        assert isinstance(exc, LegalAssistantException)
    
    def test_vector_error_hierarchy(self):
        """Test VectorStoreError hierarchy"""
        exc = VectorSearchError(
            collection="test",
            reason="timeout"
        )
        
        assert isinstance(exc, VectorSearchError)
        assert isinstance(exc, VectorStoreError)
        assert isinstance(exc, LegalAssistantException)


class TestErrorHandling:
    """Test error handling patterns"""
    
    def test_catch_specific_exception(self):
        """Test catching specific exception type"""
        def raise_file_error():
            raise FileValidationError(
                message="File too large",
                filename="test.pdf"
            )
        
        with pytest.raises(FileValidationError) as exc_info:
            raise_file_error()
        
        assert exc_info.value.status_code == 400
        assert "File too large" in str(exc_info.value)
    
    def test_catch_base_exception(self):
        """Test catching base exception"""
        def raise_any_error():
            raise LLMTimeoutError(
                provider="openai",
                model="gpt-4",
                timeout_seconds=30
            )
        
        with pytest.raises(LegalAssistantException) as exc_info:
            raise_any_error()
        
        assert exc_info.value.recoverable is True
    
    def test_recoverable_flag_usage(self):
        """Test using recoverable flag"""
        def process():
            try:
                raise LLMTimeoutError(
                    provider="openai",
                    model="gpt-4",
                    timeout_seconds=30
                )
            except LegalAssistantException as e:
                return e.recoverable
        
        # Should be recoverable
        assert process() is True
    
    def test_error_dict_for_api_response(self):
        """Test converting error to API response"""
        exc = RateLimitError(
            limit=10,
            window_seconds=60,
            retry_after=45
        )
        
        response = exc.to_dict()
        
        # Should be JSON-serializable
        import json
        json_str = json.dumps(response)
        assert json_str is not None
        
        # Should have expected structure
        assert "error" in response
        assert "message" in response["error"]
        assert "code" in response["error"]
        assert "details" in response["error"]


class TestEdgeCases:
    """Test edge cases"""
    
    def test_exception_with_empty_details(self):
        """Test exception with empty details"""
        exc = LegalAssistantException(message="test")
        
        assert exc.details == {}
        
        result = exc.to_dict()
        assert result["error"]["details"] == {}
    
    def test_exception_with_none_message(self):
        """Test exception message handling"""
        exc = LegalAssistantException(message="")
        
        assert exc.message == ""
        assert str(exc) == ""
    
    def test_multiple_exceptions_same_type(self):
        """Test creating multiple exceptions of same type"""
        exc1 = FileValidationError(
            message="Error 1",
            filename="file1.pdf"
        )
        exc2 = FileValidationError(
            message="Error 2",
            filename="file2.pdf"
        )
        
        assert exc1.details["filename"] == "file1.pdf"
        assert exc2.details["filename"] == "file2.pdf"
        assert exc1 is not exc2


# Run with: pytest tests/core/test_exceptions.py -v
