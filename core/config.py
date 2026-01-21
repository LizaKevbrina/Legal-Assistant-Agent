"""
Production-ready configuration management using Pydantic Settings.
Loads from environment variables with validation and type safety.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field, SecretStr, validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API server configuration"""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=4, ge=1, le=32, description="Uvicorn workers")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # Security
    secret_key: SecretStr = Field(
        ...,
        description="JWT secret key",
        min_length=32
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        description="JWT expiration time"
    )


class LLMSettings(BaseSettings):
    """LLM provider configuration"""
    
    # OpenAI (Primary)
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model name"
    )
    openai_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    openai_max_tokens: int = Field(
        default=2048,
        ge=1,
        le=16000,
        description="Max tokens per response"
    )
    
    # Anthropic (Fallback)
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key"
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model name"
    )
    
    # Embeddings
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model"
    )
    embedding_dimensions: int = Field(
        default=3072,
        ge=256,
        description="Embedding vector dimensions"
    )
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Batch size for embeddings"
    )
    
    # Rate Limiting
    llm_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Max LLM requests per minute"
    )
    llm_tokens_per_day: int = Field(
        default=1_000_000,
        ge=1000,
        description="Daily token budget"
    )


class LlamaParseSettings(BaseSettings):
    """LlamaParse configuration"""
    
    llamaparse_api_key: SecretStr = Field(..., description="LlamaParse API key")
    llamaparse_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Parse timeout in seconds"
    )
    llamaparse_max_pages: int = Field(
        default=100,
        ge=1,
        description="Max pages per document"
    )
    llamaparse_result_type: Literal["text", "markdown"] = Field(
        default="markdown",
        description="Output format"
    )
    
    # Vision Fallback
    use_vision_fallback: bool = Field(
        default=True,
        description="Enable GPT-4V fallback"
    )
    vision_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Vision API timeout"
    )
    
    # OCR Fallback
    use_ocr_fallback: bool = Field(
        default=True,
        description="Enable Tesseract OCR fallback"
    )
    tesseract_lang: str = Field(
        default="rus+eng",
        description="Tesseract languages"
    )


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration"""
    
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    qdrant_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Qdrant API key (for cloud)"
    )
    
    # Collections
    collection_name: str = Field(
        default="legal_documents",
        description="Main collection name"
    )
    image_collection_name: str = Field(
        default="legal_images",
        description="Image collection name"
    )
    
    # Vector Config
    vector_size: int = Field(
        default=3072,
        ge=128,
        description="Vector dimensions (must match embeddings)"
    )
    distance_metric: Literal["cosine", "euclid", "dot"] = Field(
        default="cosine",
        description="Distance metric"
    )
    
    # HNSW Index
    hnsw_m: int = Field(
        default=16,
        ge=4,
        le=64,
        description="HNSW M parameter"
    )
    hnsw_ef_construct: int = Field(
        default=200,
        ge=16,
        le=512,
        description="HNSW ef_construct parameter"
    )
    
    # Search
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Initial retrieval top-k"
    )
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Final top-k after reranking"
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )


class CohereSettings(BaseSettings):
    """Cohere reranking configuration"""
    
    cohere_api_key: SecretStr = Field(..., description="Cohere API key")
    cohere_model: str = Field(
        default="rerank-multilingual-v3.0",
        description="Rerank model"
    )
    cohere_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Rerank timeout"
    )


class RedisSettings(BaseSettings):
    """Redis configuration for caching and rate limiting"""
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_password: Optional[SecretStr] = Field(
        default=None,
        description="Redis password"
    )
    
    # Rate Limiting
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        description="Rate limit window in seconds"
    )
    rate_limit_max_requests: int = Field(
        default=10,
        ge=1,
        description="Max requests per window per user"
    )
    
    # Caching
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Cache embeddings"
    )


class SecuritySettings(BaseSettings):
    """Security and PII protection configuration"""
    
    enable_pii_detection: bool = Field(
        default=True,
        description="Enable PII detection"
    )
    pii_entities: list[str] = Field(
        default=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "LOCATION",
            "ORGANIZATION",
            "IBAN_CODE",
            "CREDIT_CARD",
            "CRYPTO",
            "IP_ADDRESS"
        ],
        description="PII entities to detect"
    )
    
    # Input Validation
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max upload file size"
    )
    allowed_file_types: list[str] = Field(
        default=[".pdf", ".png", ".jpg", ".jpeg", ".tiff"],
        description="Allowed file extensions"
    )
    max_query_length: int = Field(
        default=1000,
        ge=10,
        le=5000,
        description="Max query length in characters"
    )
    
    # Malware Scanning
    enable_malware_scan: bool = Field(
        default=False,
        description="Enable ClamAV scanning"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    
    # LangSmith
    langsmith_api_key: Optional[SecretStr] = Field(
        default=None,
        description="LangSmith API key"
    )
    langsmith_project: str = Field(
        default="legal-assistant-prod",
        description="LangSmith project name"
    )
    enable_langsmith: bool = Field(
        default=True,
        description="Enable LangSmith tracing"
    )
    
    # Prometheus
    enable_prometheus: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Prometheus exporter port"
    )
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log format"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path (None = stdout)"
    )


class EvaluationSettings(BaseSettings):
    """Evaluation and testing configuration"""
    
    enable_ragas: bool = Field(
        default=True,
        description="Enable RAGAS evaluation"
    )
    ragas_batch_size: int = Field(
        default=10,
        ge=1,
        description="RAGAS batch size"
    )
    
    # Test Datasets
    test_dataset_path: Path = Field(
        default=Path("data/test_datasets"),
        description="Test dataset directory"
    )
    
    # LLM-as-Judge
    judge_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Model for LLM-as-Judge"
    )
    
    # HITL
    enable_hitl: bool = Field(
        default=True,
        description="Enable Human-in-the-Loop"
    )
    hitl_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for HITL"
    )


class ChunkingSettings(BaseSettings):
    """Document chunking configuration"""
    
    chunk_size: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=128,
        ge=0,
        le=512,
        description="Overlap between chunks"
    )
    sentence_window_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Sentence window for context"
    )
    preserve_clause_boundaries: bool = Field(
        default=True,
        description="Preserve legal clause boundaries"
    )


class Settings(BaseSettings):
    """Main application settings - aggregates all sub-settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    # Application
    app_name: str = Field(
        default="Legal Assistant",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    # Sub-settings (nested)
    api: APISettings = Field(default_factory=APISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    llamaparse: LlamaParseSettings = Field(default_factory=LlamaParseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    cohere: CohereSettings = Field(default_factory=CohereSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure production settings are strict"""
        if v == "production":
            # Add production-specific validations
            pass
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This ensures settings are loaded only once.
    """
    return Settings()


# Convenience function for getting settings
settings = get_settings()
