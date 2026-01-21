"""
Request models for API endpoints.
Pydantic V2 models with validation and documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Literal
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
)


class DocumentType(str, Enum):
    """Legal document types"""
    CONTRACT = "contract"
    LAW = "law"
    REGULATION = "regulation"
    CASE = "case"
    OTHER = "other"


class Jurisdiction(str, Enum):
    """Legal jurisdictions"""
    RF = "rf"  # Russian Federation
    MOSCOW = "moscow"
    SPB = "spb"  # Saint Petersburg
    OTHER = "other"


class LegalArea(str, Enum):
    """Legal practice areas"""
    CIVIL = "civil"
    CRIMINAL = "criminal"
    CORPORATE = "corporate"
    TAX = "tax"
    LABOR = "labor"
    REAL_ESTATE = "real_estate"
    IP = "intellectual_property"
    OTHER = "other"


# ============================================================================
# AUTHENTICATION REQUESTS
# ============================================================================

class LoginRequest(BaseModel):
    """User login request"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "email": "user@example.com",
            "password": "SecurePassword123!"
        }
    })
    
    email: str = Field(
        ...,
        description="User email",
        min_length=3,
        max_length=255,
        json_schema_extra={"example": "user@example.com"}
    )
    password: str = Field(
        ...,
        description="User password",
        min_length=8,
        max_length=128,
        json_schema_extra={"example": "SecurePassword123!"}
    )
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()


class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(
        ...,
        description="Refresh token",
        min_length=32
    )


# ============================================================================
# DOCUMENT UPLOAD REQUESTS
# ============================================================================

class DocumentUploadMetadata(BaseModel):
    """Metadata for document upload"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "doc_type": "contract",
            "title": "Договор аренды помещения",
            "jurisdiction": "moscow",
            "legal_area": "civil",
            "date": "2025-01-15",
            "parties": ["ООО Арендодатель", "ООО Арендатор"]
        }
    })
    
    doc_type: DocumentType = Field(
        default=DocumentType.OTHER,
        description="Document type"
    )
    title: Optional[str] = Field(
        None,
        description="Document title",
        max_length=500,
        json_schema_extra={"example": "Договор аренды нежилого помещения"}
    )
    jurisdiction: Optional[Jurisdiction] = Field(
        None,
        description="Legal jurisdiction"
    )
    legal_area: Optional[LegalArea] = Field(
        None,
        description="Legal practice area"
    )
    date: Optional[datetime] = Field(
        None,
        description="Document date"
    )
    parties: Optional[list[str]] = Field(
        None,
        description="Contract parties",
        max_length=10,
        json_schema_extra={"example": ["ООО Компания А", "ООО Компания Б"]}
    )
    tags: Optional[list[str]] = Field(
        None,
        description="Custom tags",
        max_length=20,
        json_schema_extra={"example": ["аренда", "коммерческая", "москва"]}
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes",
        max_length=2000
    )
    
    @field_validator("parties")
    @classmethod
    def validate_parties(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate parties list"""
        if v:
            if len(v) > 10:
                raise ValueError("Maximum 10 parties allowed")
            # Remove empty strings
            v = [p.strip() for p in v if p.strip()]
        return v if v else None
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate tags"""
        if v:
            if len(v) > 20:
                raise ValueError("Maximum 20 tags allowed")
            # Lowercase and deduplicate
            v = list(set(tag.lower().strip() for tag in v if tag.strip()))
        return v if v else None


# ============================================================================
# QUERY REQUESTS
# ============================================================================

class QueryRequest(BaseModel):
    """Question answering request"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "Найти все договоры аренды, заключенные в 2025 году",
            "filters": {
                "doc_type": ["contract"],
                "jurisdiction": ["moscow"],
                "date_from": "2025-01-01"
            },
            "top_k": 5,
            "enable_rerank": True
        }
    })
    
    query: str = Field(
        ...,
        description="User query",
        min_length=3,
        max_length=1000,
        json_schema_extra={"example": "Какие условия аренды указаны в договоре?"}
    )
    
    # Filters
    filters: Optional["QueryFilters"] = Field(
        None,
        description="Search filters"
    )
    
    # Retrieval parameters
    top_k: int = Field(
        default=5,
        description="Number of documents to retrieve",
        ge=1,
        le=20
    )
    enable_rerank: bool = Field(
        default=True,
        description="Enable reranking"
    )
    
    # Generation parameters
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response"
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Max tokens for response",
        ge=100,
        le=2048
    )
    temperature: Optional[float] = Field(
        None,
        description="LLM temperature",
        ge=0.0,
        le=1.0
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean query"""
        # Remove extra whitespace
        v = " ".join(v.split())
        
        # Basic SQL injection check
        dangerous_patterns = ["DROP", "DELETE", "UPDATE", "INSERT", "--", "/*"]
        v_upper = v.upper()
        for pattern in dangerous_patterns:
            if pattern in v_upper:
                raise ValueError(f"Query contains prohibited pattern: {pattern}")
        
        return v


class QueryFilters(BaseModel):
    """Search filters for queries"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "doc_type": ["contract", "regulation"],
            "jurisdiction": ["moscow"],
            "legal_area": ["civil"],
            "date_from": "2024-01-01",
            "date_to": "2025-12-31",
            "parties": ["ООО Компания"],
            "tags": ["аренда"]
        }
    })
    
    doc_type: Optional[list[DocumentType]] = Field(
        None,
        description="Filter by document types"
    )
    jurisdiction: Optional[list[Jurisdiction]] = Field(
        None,
        description="Filter by jurisdictions"
    )
    legal_area: Optional[list[LegalArea]] = Field(
        None,
        description="Filter by legal areas"
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Start date"
    )
    date_to: Optional[datetime] = Field(
        None,
        description="End date"
    )
    parties: Optional[list[str]] = Field(
        None,
        description="Filter by parties",
        max_length=5
    )
    tags: Optional[list[str]] = Field(
        None,
        description="Filter by tags",
        max_length=10
    )
    document_ids: Optional[list[str]] = Field(
        None,
        description="Specific document IDs",
        max_length=10
    )
    
    @field_validator("date_from", "date_to")
    @classmethod
    def validate_dates(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Validate dates are not in future"""
        if v and v > datetime.utcnow():
            raise ValueError("Date cannot be in the future")
        return v


# ============================================================================
# DOCUMENT MANAGEMENT REQUESTS
# ============================================================================

class DocumentUpdateRequest(BaseModel):
    """Update document metadata"""
    title: Optional[str] = Field(None, max_length=500)
    doc_type: Optional[DocumentType] = None
    jurisdiction: Optional[Jurisdiction] = None
    legal_area: Optional[LegalArea] = None
    date: Optional[datetime] = None
    parties: Optional[list[str]] = Field(None, max_length=10)
    tags: Optional[list[str]] = Field(None, max_length=20)
    notes: Optional[str] = Field(None, max_length=2000)


class DocumentSearchRequest(BaseModel):
    """Search documents by metadata"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "аренда",
            "doc_type": ["contract"],
            "date_from": "2025-01-01",
            "limit": 20
        }
    })
    
    title: Optional[str] = Field(
        None,
        description="Search in title (partial match)",
        max_length=200
    )
    doc_type: Optional[list[DocumentType]] = None
    jurisdiction: Optional[list[Jurisdiction]] = None
    legal_area: Optional[list[LegalArea]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    parties: Optional[list[str]] = Field(None, max_length=5)
    tags: Optional[list[str]] = Field(None, max_length=10)
    
    # Pagination
    limit: int = Field(
        default=20,
        description="Results per page",
        ge=1,
        le=100
    )
    offset: int = Field(
        default=0,
        description="Pagination offset",
        ge=0
    )
    
    # Sorting
    sort_by: Literal["date", "title", "created_at"] = Field(
        default="created_at",
        description="Sort field"
    )
    sort_order: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort order"
    )


# ============================================================================
# FEEDBACK REQUESTS
# ============================================================================

class FeedbackRequest(BaseModel):
    """User feedback on query response"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query_id": "abc123",
            "rating": 4,
            "feedback_text": "Ответ был полезен, но не хватало деталей",
            "issues": ["incomplete", "missing_sources"]
        }
    })
    
    query_id: str = Field(
        ...,
        description="Query/response ID",
        min_length=1,
        max_length=100
    )
    rating: int = Field(
        ...,
        description="Rating 1-5",
        ge=1,
        le=5
    )
    feedback_text: Optional[str] = Field(
        None,
        description="Feedback text",
        max_length=2000
    )
    issues: Optional[list[str]] = Field(
        None,
        description="Issue categories",
        json_schema_extra={"example": ["inaccurate", "incomplete", "hallucination"]}
    )
    helpful: bool = Field(
        default=True,
        description="Was response helpful?"
    )


# ============================================================================
# EVALUATION REQUESTS
# ============================================================================

class EvaluationRequest(BaseModel):
    """Request RAGAS evaluation"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "Какие условия аренды?",
            "response": "Арендная плата составляет 100000 рублей",
            "contexts": ["Договор аренды... арендная плата 100000 руб"],
            "ground_truth": "Арендная плата 100000 рублей в месяц"
        }
    })
    
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    contexts: list[str] = Field(..., description="Retrieved contexts")
    ground_truth: Optional[str] = Field(
        None,
        description="Ground truth answer (if available)"
    )


# Update forward references
QueryRequest.model_rebuild()
